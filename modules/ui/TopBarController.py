import copy
import json
import os
import shutil
import traceback
from contextlib import suppress

from modules.util import path_util
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import default_optimizer_config
from modules.util.path_util import write_json_atomic
from modules.util.ui.browser_util import open_url

# base models for the model types that have no built-in preset (the others use their presets' choice)
FALLBACK_BASE_MODELS = {
    ModelType.STABLE_DIFFUSION_20: "sd2-community/stable-diffusion-2",
    ModelType.STABLE_DIFFUSION_35: "stabilityai/stable-diffusion-3.5-large",
    ModelType.FLUX_FILL_DEV_1: "black-forest-labs/FLUX.1-Fill-dev",
}


class TopBarController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_model_types(self) -> list[tuple[str, ModelType]]:
        return [  #TODO simplify
            ("SD1.5", ModelType.STABLE_DIFFUSION_15),
            ("SD1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("SD2.0", ModelType.STABLE_DIFFUSION_20),
            ("SD2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("SD2.1", ModelType.STABLE_DIFFUSION_21),
            ("SD3", ModelType.STABLE_DIFFUSION_3),
            ("SD3.5", ModelType.STABLE_DIFFUSION_35),
            ("SDXL", ModelType.STABLE_DIFFUSION_XL_10_BASE),
            ("SDXL Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
            ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
            ("Stable Cascade", ModelType.STABLE_CASCADE_1),
            ("PixArt Alpha", ModelType.PIXART_ALPHA),
            ("PixArt Sigma", ModelType.PIXART_SIGMA),
            ("Flux Dev.1", ModelType.FLUX_DEV_1),
            ("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1),
            ("Flux 2 [Dev, Klein]", ModelType.FLUX_2),
            ("Sana", ModelType.SANA),
            ("Hunyuan Video", ModelType.HUNYUAN_VIDEO),
            ("HiDream Full", ModelType.HI_DREAM_FULL),
            ("Chroma1", ModelType.CHROMA_1),
            ("QwenImage", ModelType.QWEN),
            ("Anima", ModelType.ANIMA),
            ("Krea 2", ModelType.KREA_2),
            ("Z-Image", ModelType.Z_IMAGE),
            ("Ernie Image", ModelType.ERNIE),
            ("Ideogram 4", ModelType.IDEOGRAM_4),
        ]

    def get_training_methods(self, model_type: ModelType) -> list[tuple[str, TrainingMethod]]:
        labels = {
            TrainingMethod.FINE_TUNE: "Fine Tune",
            TrainingMethod.LORA: "LoRA",
            TrainingMethod.EMBEDDING: "Embedding",
            TrainingMethod.FINE_TUNE_VAE: "Fine Tune VAE",
        }
        return [(labels[m], m) for m in model_type.supported_training_methods()]

    def load_preset_tree(self, dir: str = "training_presets") -> list[tuple[str, str | list]]:
        # mirrors whatever directory structure happens to exist under `dir`; a node is either
        # (display_name, path) for a leaf preset or (display_name, children) for a group.
        # "#" marks a built-in preset (vs. a user file); "#.json" is the last-session state,
        # not a preset. Same convention as load_config_from_file's is_built_in_preset check.
        nodes = []
        if os.path.isdir(dir):
            for entry in sorted(os.scandir(dir), key=lambda e: e.name.lower()):
                if entry.is_dir():
                    children = self.load_preset_tree(entry.path)
                    if children:
                        nodes.append((entry.name, children))
                elif entry.name.startswith("#") and entry.name != "#.json" and entry.name.endswith(".json"):
                    nodes.append((os.path.splitext(entry.name)[0], path_util.canonical_join(dir, entry.name)))
        return nodes

    def _built_in_presets(self, dir: str) -> list[dict]:
        presets = []
        for root, _, files in os.walk(dir):
            for name in files:
                if name.startswith("#") and name != "#.json" and name.endswith(".json"):
                    with suppress(Exception), open(os.path.join(root, name), "r", encoding="utf-8") as f:
                        presets.append(json.load(f))
        return presets

    def preset_base_models(self, dir: str = "training_presets") -> dict[ModelType, list[str]]:
        # the base models the built-in presets use per model type, most used first
        counts: dict[ModelType, dict[str, int]] = {}
        for preset in self._built_in_presets(dir):
            with suppress(ValueError):
                model_type = ModelType(preset.get("model_type", str(ModelType.STABLE_DIFFUSION_15)))
                base_model = preset.get("base_model_name")
                if base_model:
                    type_counts = counts.setdefault(model_type, {})
                    type_counts[base_model] = type_counts.get(base_model, 0) + 1
        base_models = {model_type: sorted(names, key=lambda n: -names[n]) for model_type, names in counts.items()}
        for model_type, base_model in FALLBACK_BASE_MODELS.items():
            base_models.setdefault(model_type, [base_model])
        return base_models

    def preset_part_models(self, dir: str = "training_presets") -> dict[ModelType, dict[str, str]]:
        # the models the built-in presets set on the other parts per model type: Wuerstchen/Stable Cascade's decoder
        # and effnet encoder, HiDream's Llama text encoder. The diffusion model's override is a variant choice
        # (e.g. Z-Image De-Turbo), not a default, so it is left out.
        part_models: dict[ModelType, dict[str, str]] = {}
        for preset in self._built_in_presets(dir):
            with suppress(ValueError):
                model_type = ModelType(preset.get("model_type", str(ModelType.STABLE_DIFFUSION_15)))
                for part in model_type.model_parts()[1:]:
                    part_config = preset.get(part)
                    model_name = part_config.get("model_name") if isinstance(part_config, dict) else None
                    if model_name:
                        part_models.setdefault(model_type, {}).setdefault(part, model_name)
        return part_models

    def update_base_model_for(self, model_type: ModelType, ui_state):
        # Switching the model type keeps the base model, so a base model left from another model type would be loaded
        # (and downloaded) as this one. Replace it only while it is another model type's preset default (or empty);
        # a model the user picked stays.
        preset_base_models = self.preset_base_models()
        current = self.train_config.base_model_name
        own_defaults = preset_base_models.get(model_type, [])
        other_defaults = {name for other_type, names in preset_base_models.items() if other_type != model_type
                          for name in names}
        if current in own_defaults or (current and current not in other_defaults):
            return
        new = own_defaults[0] if own_defaults else ""
        if new != current:
            self.train_config.base_model_name = new  # config first: the var's callbacks run before its trace
            ui_state.get_var("base_model_name").set(new)

        # the parts that need their own model (decoder, effnet encoder, HiDream's Llama), by the same rule
        part_models = self.preset_part_models()
        other_part_models = {(part, name) for other_type, parts in part_models.items() if other_type != model_type
                             for part, name in parts.items()}
        for part in model_type.model_parts()[1:]:
            part_config = getattr(self.train_config, part, None)
            if part_config is None:
                continue
            current = part_config.model_name
            new = part_models.get(model_type, {}).get(part, "")
            if current == new or (current and (part, current) not in other_part_models):
                continue
            part_config.model_name = new
            ui_state.get_var(f"{part}.model_name").set(new)

    def save_to_file(self, name) -> str:
        name = path_util.safe_filename(name)
        path = path_util.canonical_join("training_presets", f"{name}.json")
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))
        return path

    def save_config_to_path(self, path: str) -> None:
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))

    def load_config_from_file(self, filename: str) -> TrainConfig | None:
        try:
            basename = os.path.basename(filename)
            is_built_in_preset = basename.startswith("#") and basename != "#.json"

            with open(filename, "r", encoding="utf-8") as f:
                loaded_dict = json.load(f)
            # built-in configs are always saved in the most recent version, so migration can be skipped
            migrate = not is_built_in_preset
            # (the migrations edit nested dicts in place, so each pass gets its own copy)
            loaded_config = TrainConfig.default_values().from_dict(copy.deepcopy(loaded_dict), migrate=migrate)
            # as scripts/train.py does: load again on top of the optimizer's defaults, so the optimizer settings the
            # file leaves out get the UI's defaults and the ones it sets are kept
            default_config = TrainConfig.default_values()
            default_config.optimizer = default_optimizer_config(loaded_config.optimizer.optimizer)
            loaded_config = default_config.from_dict(copy.deepcopy(loaded_dict), migrate=migrate).to_unpacked_config()

            with suppress(FileNotFoundError), open("secrets.json", "r", encoding="utf-8") as f:
                secrets_dict = json.load(f)
                loaded_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)

            self.train_config.from_dict(loaded_config.to_dict())
            return loaded_config
        except FileNotFoundError:
            return None
        except Exception:
            print(traceback.format_exc())
            if os.path.basename(filename) == "#.json":
                # the UI keeps its defaults and overwrites #.json with them on close, so keep the unreadable state
                with suppress(OSError):
                    shutil.copyfile(filename, filename + ".bak")
                    print(f"Could not restore the last session. A copy of {filename} was saved as {filename}.bak")
            return None

    def save_secrets(self, path) -> str:
        write_json_atomic(path, self.train_config.secrets.to_dict())
        return path

    def open_wiki(self):
        open_url("https://github.com/Nerogar/OneTrainer/wiki")

    def save_default(self):
        self.save_to_file("#")
        self.save_secrets("secrets.json")
