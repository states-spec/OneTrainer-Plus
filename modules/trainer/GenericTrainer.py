import copy
import json
import math
import os
import shutil
import traceback
from collections.abc import Callable
from pathlib import Path

import modules.util.multi_gpu_util as multi
import modules.util.nvcomp_util as nvcomp_util
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import concept_stats, create, huggingface_util, path_util
from modules.util.bf16_stochastic_rounding import set_seed as bf16_stochastic_rounding_set_seed
from modules.util.cache_util import changed_cache_settings, save_cache_settings
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.compile_util import init_compile, reset_compile
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig, get_output_model_destination
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import PeftType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.profiling_util import PeakMemoryRecorder, TorchMemoryRecorder, TorchProfiler
from modules.util.run_diagnostics import apply_reproducibility_env, diagnostics
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import fp8_matmul_supported, torch_gc
from modules.util.tqdm_util import tqdm
from modules.util.training_seed import (
    apply_training_seed,
    check_training_seed,
    seed_adv_optm_generators,
    training_seed_offset,
)
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

# OT_DEBUG_PROFILES=1 dumps a CUDA memory snapshot for the first two steps, where the allocator is still
# growing, and a profiler trace at steps 10 and 40, past compilation and warmup.
_DEBUG_PROFILES = os.environ.get("OT_DEBUG_PROFILES") == "1"
_MEMORY_PROFILE_STEPS = (0, 1) if _DEBUG_PROFILES else ()
_PROFILE_STEPS = (10, 11, 40, 41) if _DEBUG_PROFILES else ()
# optimizer steps into a run after which a D-adapting optimizer still at Initial D is reported (create.d_at_initial_groups)
_D_AT_INITIAL_CHECK_STEPS = 100


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]
    default_sample_queued: bool

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)
        # torch._dynamo.config overrides are thread-local, so init_compile() must be called in the training thread/process.
        reset_compile()
        init_compile()

        if multi.is_master():
            tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
            os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, config.run_name))
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []

    @diagnostics.record_errors
    def start(self):
        # checked up front: the saver only rejects an unsupported format when it first saves, after the training
        formats = self.config.model_type.supported_output_formats(self.config.training_method)
        if self.config.output_model_format not in formats:
            raise ValueError(
                f"Output format {self.config.output_model_format} can't save {self.config.training_method} training of "
                f"{self.config.model_type}. Set output_model_format to one of: {', '.join(str(f) for f in formats)}."
            )
        # a value saved for another model type stays when the type changes; the loss function raised at the first step
        loss_weights = self.config.model_type.supported_loss_weights()
        if self.config.loss_weight_fn not in loss_weights:
            raise ValueError(
                f"Loss weight function {self.config.loss_weight_fn} doesn't work with {self.config.model_type}. "
                f"Set loss_weight_fn to one of: {', '.join(str(w) for w in loss_weights)}."
            )
        # a limit of 0 divides 0 by 0 on the zero-initialized rotations of the first step (NaN loss after caching)
        if self.config.training_method == TrainingMethod.LORA and self.config.peft_type == PeftType.OFT_2 \
                and self.config.oft_clipped_norm is not None and self.config.oft_clipped_norm <= 0:
            raise ValueError("oft_clipped_norm must be greater than 0. Leave it empty to turn spectral norm clipping off.")
        # HiDream's mixture-of-experts layers pick their experts by the data (boolean indexing in training, a
        # .cpu().numpy() loop at sampling), which a fullgraph compile can't trace: the first step failed in inductor
        if self.config.compile and self.config.model_type.is_hi_dream():
            raise ValueError("Compile transformer blocks doesn't work with HiDream: its mixture-of-experts layers choose "
                             "experts from the data, which torch.compile can't trace. Turn it off on the model tab.")
        self.__check_device_support()
        create.check_optimizer_config(self.config.optimizer)
        check_training_seed(self.config)
        diagnostics.close()  # a run in this process (the UI) that ended in an error left its log open
        diagnostics.start(self.config, torch.device(self.config.train_device))
        # OT_SEED (diagnostics, rocm_check) replaces the training seed for the whole run
        apply_training_seed(self.config)
        apply_reproducibility_env()

        if multi.is_master():
            self.__save_config_to_workspace()
            self.__check_concept_paths()
            if self.config.masked_training:
                self.__report_images_without_mask()

            if self.config.latent_caching:
                if self.config.clear_cache_before_training:
                    self.__clear_cache()
                else:
                    changed = changed_cache_settings(self.config)
                    if changed is None:
                        print("The cache has no record of the settings it was made with, so it is rebuilt.")
                        self.__clear_cache()
                    elif changed:
                        print(f"The cache was made with different settings ({', '.join(changed)}), so it is rebuilt.")
                        self.__clear_cache()
                save_cache_settings(self.config)

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path
                    # the trained transformer is part of the backup, a configured override would replace it
                    model_names.transformer_model = ""

                description = self.config.describe_backup(last_backup_path)
                print(f"Continuing training from backup '{last_backup_path}'"
                      f"{f' ({description})' if description else ''}...")
            else:
                print("No backup found, continuing without backup...")

        huggingface_util.configure_hub(
            self.config.secrets.huggingface_token,
            offline_mode=self.config.offline_mode,
            cache_dir=self.config.huggingface_cache_dir,
            on_status=self.callbacks.on_update_status,
        )

        self.callbacks.on_update_status("loading the model")

        # always derive from the current cache_dir: a value persisted from an earlier run (e.g. a copied config)
        # would otherwise keep pointing at that run's cache, or at a local path on a cloud machine
        self.config.quantization.cache_dir = self.config.cache_dir + "/quantization"
        os.makedirs(self.config.quantization.cache_dir, exist_ok=True)

        with diagnostics.phase("load model"):
            self.model = self.model_loader.load(
                model_type=self.config.model_type,
                model_names=model_names,
                weight_dtypes=self.config.weight_dtypes(),
                quantization=self.config.quantization,
                stream_from_disk=self.config.streams_from_disk(),
                cache_in_ram=self.config.cache_in_ram(),
            )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        with diagnostics.phase("model setup"):
            self.model_setup.setup_optimizations(self.model, self.config)
            self.model_setup.setup_train_device(self.model, self.config)
            self.model_setup.setup_model(self.model, self.config)
        self.model.eval()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()
        # the saver's format checks need the trained parts set up, so they run here, before the cache is built and
        # the training starts: at the final save, a refusal came after the whole run and left no output
        self.model_saver.check_can_save(self.model, self.config.output_model_format)

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []
        self.default_sample_queued = False

        self.parameters = self.model.parameters.parameters()

        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True
            )

    def __check_device_support(self):
        # settings the GPU or the PyTorch build can't run fail only at the model's first forward pass, after the
        # model is loaded and the cache is built
        if self.config.attention_mechanism == AttentionMechanism.CUDNN and torch.version.hip is not None:
            raise ValueError(
                "cuDNN attention is NVIDIA-only, and this PyTorch is built for ROCm. Select torch SDPA, which uses the "
                "attention kernels PyTorch has for AMD GPUs."
            )

        fp8_parts = []
        for part in self.config.model_type.model_parts():
            weight_dtype = getattr(self.config, part).weight_dtype
            if weight_dtype.quantize_fpW8A8() or weight_dtype in (DataType.GGUF_A8_FLOAT, DataType.BFLOAT_16_A8_FLOAT):
                fp8_parts.append(f"{part}: {weight_dtype}")
        train_device = torch.device(self.config.train_device)
        if fp8_parts and not fp8_matmul_supported(train_device):
            device_name = torch.cuda.get_device_name(train_device) if train_device.type == "cuda" else str(train_device)
            raise ValueError(
                f"The float W8A8 / A8 weight data types ({', '.join(fp8_parts)}) compute in fp8, which {device_name} "
                f"can't: that needs an NVIDIA GPU with compute capability 8.9 or newer, or an AMD Instinct MI300/MI350 "
                f"or Radeon RX 9000 series GPU. Use float8 (W8) to store the weights in 8 bits and compute in the "
                f"train data type, or int W8A8 / GGUF A8 int / bfloat16 A8 int for int8 compute."
            )

        # the compressed weight data types need nvCOMP (NVIDIA only). Without it the loader stops while it builds
        # the first compressed layer (quantization_util); stopped here, nothing has loaded yet.
        compressed_parts = [part for part in self.config.model_type.model_parts()
                            if getattr(self.config, part).weight_dtype.is_compressed()]
        if compressed_parts and not nvcomp_util.available():
            raise ValueError(
                f"The compressed weight data types "
                f"({', '.join(f'{part}: {getattr(self.config, part).weight_dtype}' for part in compressed_parts)}) "
                f"need nvCOMP, which isn't available: {nvcomp_util.unavailable_reason()}. Use the same data type "
                f"without compression "
                f"({', '.join(f'{part}: {getattr(self.config, part).weight_dtype.uncompressed()}' for part in compressed_parts)}"
                f", the same choice without 'compressed' on the model tab): the same weights, uncompressed."
            )

    def __load_concepts(self) -> list[ConceptConfig] | None:
        concepts = self.config.concepts
        if concepts is None:
            try:
                with open(self.config.concept_file_name, "r", encoding="utf-8") as f:
                    concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]
            except (OSError, ValueError):
                return None  # the data loader reports a missing or unreadable concepts file
        return concepts

    def __check_concept_paths(self):
        # the data loader lists each enabled concept's folder when the first epoch starts, after the model has loaded,
        # and an empty path failed there as "No such file or directory: ''" without naming the concept
        for concept in self.__load_concepts() or []:
            if not concept.enabled:
                continue
            label = f"Concept '{concept.name}'" if concept.name else "A concept"
            if self.config.concepts is None:
                label += f" in {self.config.concept_file_name}"
            if not concept.path:
                raise ValueError(f"{label} has no folder (its path is empty). Choose the folder with its images on the "
                                 f"concepts tab, or disable it.")
            if not os.path.isdir(concept.path):
                raise ValueError(f"{label} points to {concept.path!r}, which isn't a folder (relative paths start at "
                                 f"{os.getcwd()}). Choose the folder with its images on the concepts tab, or disable it.")

    def __report_images_without_mask(self):
        # masked training gives an image without a mask file an all-white mask, which trains the whole image
        concepts = self.__load_concepts()
        if concepts is None:
            return
        for concept, missing, total, examples in concept_stats.images_without_mask(
                concepts, allow_videos=self.config.model_type.is_video_model()):
            print(f"Masked training: {missing} of {total} images in concept '{concept.name or concept.path}' "
                  f"{'has' if missing == 1 else 'have'} no mask (<image name>-masklabel.png next to the image), so "
                  f"{'it trains' if missing == 1 else 'they train'} on the whole image: {', '.join(examples)}"
                  f"{', ...' if missing > len(examples) else ''}")

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{self.config.run_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.config.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith('epoch-') or filename in ['image', 'text']):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        for dirpath in self.config.get_backup_paths()[backups_to_keep:]:
            try:
                shutil.rmtree(dirpath)
            except OSError:  # noqa: PERF203 - continue pruning after a filesystem error
                tqdm.write(f"Could not delete old rolling backup {dirpath}")

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        with PeakMemoryRecorder("sampling", enabled=False):
            for fun in self.sample_queue:
                fun()
        self.sample_queue = []
        self.default_sample_queued = False

    def __sample_loop(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_config_list: list[SampleConfig],
            ema_applied: bool,
            folder_postfix: str = "",
            is_custom_sample: bool = False,
    ):
        for i, sample_config in multi.distributed(
            [(i, sample_config) for i, sample_config in enumerate(sample_config_list) if sample_config.enabled],
            distribute=not self.config.samples_to_tensorboard and not ema_applied
        ):
            try:
                safe_prompt = path_util.safe_filename(sample_config.prompt)

                if is_custom_sample:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        "custom",
                    )
                else:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        f"{str(i)} - {safe_prompt}{folder_postfix}",
                    )

                sample_path = os.path.join(
                    sample_dir,
                    f"{self.config.get_timestamped_run_name()}-training-sample-{train_progress.filename_string()}"
                )

                def on_sample_default(sampler_output: ModelSamplerOutput):
                    if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                        self.tensorboard.add_image(
                            f"sample{str(i)} - {safe_prompt}", pil_to_tensor(sampler_output.data),  # noqa: B023
                            train_progress.global_step
                        )
                    self.callbacks.on_sample_default(sampler_output)

                def on_sample_custom(sampler_output: ModelSamplerOutput):
                    self.callbacks.on_sample_custom(sampler_output)

                on_sample = on_sample_custom if is_custom_sample else on_sample_default
                on_update_progress = self.callbacks.on_update_sample_custom_progress if is_custom_sample else self.callbacks.on_update_sample_default_progress

                self.model.eval()

                sample_config = copy.copy(sample_config)
                sample_config.from_train_config(self.config)

                self.model_sampler.sample(
                    sample_config=sample_config,
                    destination=sample_path,
                    image_format=self.config.sample_image_format,
                    video_format=self.config.sample_video_format,
                    audio_format=self.config.sample_audio_format,
                    on_sample=on_sample,
                    on_update_progress=on_update_progress,
                )
            except Exception:
                traceback.print_exc()
                tqdm.write("Error during sampling, proceeding without sampling")

            torch_gc()

    @diagnostics.phase_method("sampling")
    def __sample_during_training(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("Sampling ...")

        is_custom_sample = False
        if sample_params_list:
            is_custom_sample = True
        elif self.config.samples is not None:
            sample_params_list = self.config.samples
        else:
            try:
                with open(self.config.sample_definition_file_name, 'r', encoding="utf-8") as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values(self.config.model_type).from_dict(samples[i])
                    sample_params_list = samples
            # We absolutely do not want to fail training just because the sample definition file becomes missing or broken right before sampling.
            except Exception:
                traceback.print_exc()
                tqdm.write("Error during loading the sample definition file, proceeding without sampling")
                sample_params_list = []

        if self.model.ema:
            #the EMA model only exists in the master process, so EMA sampling is done on one GPU only
            #non-EMA sampling is done on all GPUs
            assert multi.is_master() and self.config.ema != EMAMode.OFF
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            is_custom_sample=is_custom_sample,
            ema_applied = self.config.ema != EMAMode.OFF
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if ema is enabled:
        if self.config.ema != EMAMode.OFF and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                folder_postfix=" - no-ema",
                ema_applied = False,
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            with diagnostics.phase("validation"):
                self.__run_validation(train_progress)

    def __run_validation(self, train_progress: TrainProgress):
        self.validation_data_loader.get_data_set().start_next_epoch()
        current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

        if current_epoch_length_validation == 0:
            return

        self.callbacks.on_update_status("Calculating validation loss")
        self.model_setup.setup_train_device(self.model, self.config)
        # evaluated like samples: without dropout, and on the weights that are sampled and saved (schedule-free
        # optimizers train on an interpolated point and switch to their averaged weights for evaluation)
        self.model.eval()
        self.__before_eval()

        torch_gc()

        step_tqdm_validation = tqdm(
            self.validation_data_loader.get_data_loader(),
            desc="validation_step",
            total=current_epoch_length_validation)

        accumulated_loss_per_concept = {}
        concept_counts = {}
        mapping_seed_to_label = {}
        mapping_label_to_seed = {}

        for validation_batch in step_tqdm_validation:
            if self.__needs_gc(train_progress):
                torch_gc()

            with torch.no_grad():
                model_output_data = self.model_setup.predict(
                    self.model, validation_batch, self.config, train_progress, deterministic=True)
                loss_validation = self.model_setup.calculate_loss(
                    self.model, validation_batch, model_output_data, self.config)

            # since validation batch size = 1
            concept_name = validation_batch["concept_name"][0]
            concept_path = validation_batch["concept_path"][0]
            concept_seed = validation_batch["concept_seed"].item()
            loss = loss_validation.item()

            label = concept_name if concept_name else os.path.basename(concept_path)
            # check and fix collision to display both graphs in tensorboard
            if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                suffix = 1
                new_label = f"{label}({suffix})"
                while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                    suffix += 1
                    new_label = f"{label}({suffix})"
                label = new_label

            if concept_seed not in mapping_seed_to_label:
                mapping_seed_to_label[concept_seed] = label
                mapping_label_to_seed[label] = concept_seed

            accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
            concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

        self.model_setup.setup_train_device(self.model, self.config)
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        for concept_seed, total_loss in accumulated_loss_per_concept.items():
            average_loss = total_loss / concept_counts[concept_seed]

            self.tensorboard.add_scalar(f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                                        average_loss,
                                        train_progress.global_step)

        if len(concept_counts) > 1:
            total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
            total_count = sum(concept_counts[key] for key in concept_counts)
            total_average_loss = total_loss / total_count

            self.tensorboard.add_scalar("loss/validation_step/total_average",
                                        total_average_loss,
                                        train_progress.global_step)

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    @diagnostics.phase_method("backup")
    def __backup(self, train_progress: TrainProgress, print_msg: bool = True):
        torch_gc()

        self.callbacks.on_update_status("Creating backup")

        save_name = path_util.safe_filename(self.config.get_timestamped_run_name(), max_length=None)
        backup_name = f"{save_name}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                tqdm.write("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
        except Exception:
            traceback.print_exc()
            tqdm.write("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                tqdm.write("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    @diagnostics.phase_method("save")
    def __save(self, train_progress: TrainProgress, print_msg: bool = True):
        torch_gc()

        self.callbacks.on_update_status("Saving")

        save_name = path_util.safe_filename(self.config.get_timestamped_run_name(), max_length=None)
        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{save_name}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}"
        )
        if print_msg:
            tqdm.write("Saving " + save_path)

        try:
            # Special case for schedule-free optimizers. eval() goes before the EMA weights are copied in, as in end()
            # and sampling: after them, it moved the EMA weights towards the optimizer's z sequence
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )
        except Exception:
            traceback.print_exc()
            tqdm.write("Could not save model. Check your disk space!")
            try:
                if os.path.isdir(save_path):
                    shutil.rmtree(save_path)
                elif os.path.isfile(save_path):
                    os.remove(save_path)
            except Exception:
                traceback.print_exc()
                tqdm.write("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)
            # also after a failed save, which used to leave the optimizer in eval mode for the rest of the training
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first", self.config.sample_skip_first, self.config.sample_after_unit, train_progress
        ) and self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(self, scaler):
        fused_optimizer_step = self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass
        fused_reduce = self.config.multi_gpu and self.config.fused_gradient_reduce
        if fused_optimizer_step:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating Fused Back Pass with Accumulation Steps > 1 does not reduce VRAM usage.")
            if self.config.clip_grad_norm is not None:
                # each tensor is stepped in its own grad hook, before the other gradients exist
                print(f"Note: with Fused Back Pass, Clip Grad Norm clips each parameter's gradient to a norm of "
                      f"{self.config.clip_grad_norm} on its own, not the norm of all gradients together, which is only "
                      f"known after the back pass. This clips less than the same value without Fused Back Pass.")
            if self.config.multi_gpu and not fused_reduce:
                raise ValueError("if Fused Back Pass and Multi-GPU is enabled, Fused Reduce must also be enabled")
        elif not fused_reduce:
            return

        for param_group in self.model.optimizer.param_groups:
            for i, parameter in enumerate(param_group["params"]):
                # TODO: Find a better check instead of "parameter.requires_grad".
                #       This will break if the some parameters don't require grad during the first training step.
                if parameter.requires_grad:
                    if scaler:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            scaler.unscale_parameter_(tensor, self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                            tensor.grad = None
                    else:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            self.model.optimizer.step_parameter(tensor, param_group, i)
                            tensor.grad = None

                    # the step function is bound as a default: a closure would see only the one defined for the last
                    # parameter, so every parameter was stepped with the last param group's lr, betas and decay
                    def __grad_hook(tensor: Tensor, optimizer_step=__optimizer_step):
                        init_compile()  # workaround for https://github.com/pytorch/pytorch/issues/186537
                        if self.__is_update_step(self.model.train_progress):
                            if fused_reduce:
                                multi.reduce_grads_mean(
                                    [tensor],
                                    self.config.gradient_reduce_precision,
                                    after_reduce=optimizer_step if fused_optimizer_step else None,
                                    async_op=self.config.async_gradient_reduce,
                                    max_buffer=self.config.async_gradient_reduce_buffer * 1024 * 1024,
                                )
                            elif fused_optimizer_step:
                                optimizer_step(tensor)

                    handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                    self.grad_hook_handles.append(handle)


    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    @diagnostics.record_errors
    def train(self):
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress

        if self.config.only_cache:
            if multi.is_master():
                self.callbacks.on_update_status("Caching")
                for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
                    with diagnostics.phase("caching", epoch=_epoch):
                        self.data_loader.get_data_set().start_next_epoch()
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = torch.tensor(0.0, device=train_device)
        ema_loss = None
        ema_loss_steps = 0
        optimizer_steps = 0
        epochs = range(train_progress.epoch, self.config.epochs, 1)

        if multi.is_master():
            epochs = tqdm(epochs, desc="epoch", initial=train_progress.epoch, total=self.config.epochs)
        for _epoch in epochs:
            multi.sync_commands(self.commands)
            if self.commands.get_stop_command():
                return
            self.callbacks.on_update_status("Starting epoch/caching")

            #call start_next_epoch with only one process at first, because it might write to the cache. All subsequent processes can read in parallel:
            for _ in multi.master_first():
                with diagnostics.phase("epoch start (caching)", epoch=train_progress.epoch):
                    if self.config.latent_caching:
                        self.data_loader.get_data_set().start_next_epoch()
                        self.model_setup.setup_train_device(self.model, self.config)
                    else:
                        self.model_setup.setup_train_device(self.model, self.config)
                        self.data_loader.get_data_set().start_next_epoch()

            if self.config.debug_mode:
                multi.warn_parameter_divergence(self.parameters, train_device)

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            # no torch_gc here: setup_train_device above ends in materialize_only(), which collects whenever a
            # part actually moved. On an epoch that changed nothing (the usual case once the caches are warm)
            # there is nothing to reclaim, and this collection cost ~350ms of the epoch gap on a large model.

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()

            if multi.is_master():
                batches = step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step", total=current_epoch_length,
                                 initial=train_progress.epoch_step)
            else:
                batches = self.data_loader.get_data_loader()
            epoch_started_at_step = train_progress.epoch_step
            epoch_batch_count = 0
            for batch in batches:
                epoch_batch_count += 1
                multi.sync_commands(self.commands)
                if self.commands.get_stop_command():
                    multi.warn_parameter_divergence(self.parameters, train_device)

                # A "sample now" request on a step that samples anyway, or while a sample still waits in the queue
                # (with gradient accumulation it runs at the next update), is that same sample, so it is used up
                # here without a second one. The request used to stay set behind the scheduled sample's `or`, and
                # sampled the same weights again on the next step.
                scheduled_sample = not self.commands.get_stop_command() and self.__needs_sample(train_progress)
                requested_sample = self.commands.get_and_reset_sample_default_command()
                if (scheduled_sample or requested_sample) and not self.default_sample_queued:
                    self.default_sample_queued = True
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )
                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:
                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    backup = self.commands.get_and_reset_backup_command()
                    save = self.commands.get_and_reset_save_command()
                    if multi.is_master() and (backup or save):
                        self.model.evict()
                        if backup:
                            self.__backup(train_progress, True)
                        if save:
                            self.__save(train_progress, True)
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("Training ...")

                with (
                    TorchMemoryRecorder(enabled=multi.is_master() and train_progress.global_step in _MEMORY_PROFILE_STEPS, filename=f"memory-step{train_progress.global_step}-{get_string_timestamp()}.pickle"),
                    TorchProfiler      (enabled=multi.is_master() and train_progress.global_step in _PROFILE_STEPS, filename=f"profile-step{train_progress.global_step}-{get_string_timestamp()}.json"),
                ):
                    step_seed = train_progress.global_step + training_seed_offset(self.config)
                    bf16_stochastic_rounding_set_seed(step_seed, train_device)
                    seed_adv_optm_generators(step_seed, train_device)

                    prior_pred_indices = [i for i in range(self.config.batch_size)
                                          if ConceptType(batch['concept_type'][i]) == ConceptType.PRIOR_PREDICTION]
                    if len(prior_pred_indices) > 0 \
                            or (self.config.masked_training
                                and self.config.masked_prior_preservation_weight > 0
                                and self.config.training_method == TrainingMethod.LORA):
                        with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                            #do NOT create a subbatch using the indices, even though it would be more efficient:
                            #different timesteps are used for a smaller subbatch by predict(), but the conditioning must match exactly:
                            prior_model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                        prior_model_prediction = prior_model_output_data['predicted'].to(dtype=model_output_data['target'].dtype)
                        model_output_data['target'][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
                        model_output_data['prior_target'] = prior_model_prediction
                    else:
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                    loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)

                    loss = loss / self.config.gradient_accumulation_steps
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    has_gradient = True
                    detached_loss = loss.detach()
                    multi.reduce_tensor_mean(detached_loss)
                    accumulated_loss += detached_loss

                    if self.__is_update_step(train_progress):
                        # checked before the optimizer step: a NaN loss means NaN gradients, which would go into the
                        # weights and the optimizer state (and can crash the GPU runtime). The loss is already reduced
                        # over all GPUs, so every rank stops. A fused back pass has already applied them by now.
                        # An infinite loss does the same damage, except with fp16 training, whose grad scaler skips
                        # steps with infinite gradients itself.
                        accumulated_loss_cpu = accumulated_loss.item()
                        if math.isnan(accumulated_loss_cpu) or (math.isinf(accumulated_loss_cpu) and not (scaler and scaler.is_enabled())):
                            raise RuntimeError(f"Training loss became {'NaN' if math.isnan(accumulated_loss_cpu) else 'infinite'}. This may be due to invalid parameters, precision issues, or a bug in the loss computation.")

                        if self.config.fused_gradient_reduce:
                            multi.finish_async(self.config.gradient_reduce_precision)
                        else:
                            multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)

                        if scaler and self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()

                        lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                        self.model.optimizer.zero_grad(set_to_none=True)
                        has_gradient = False

                        if multi.is_master():
                            self.model_setup.report_to_tensorboard(
                                self.model, self.config, lr_scheduler, self.tensorboard
                            )

                            self.tensorboard.add_scalar("loss/train_step",accumulated_loss_cpu , train_progress.global_step)
                            ema_loss = ema_loss or accumulated_loss_cpu
                            ema_loss_steps += 1
                            ema_loss_decay = min(0.99, 1 - (1 / ema_loss_steps))
                            ema_loss = (ema_loss * ema_loss_decay) + (accumulated_loss_cpu * (1 - ema_loss_decay))
                            step_tqdm.set_postfix({
                                'loss': accumulated_loss_cpu,
                                'smooth loss': ema_loss,
                            })
                            self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)

                        accumulated_loss = 0.0
                        self.model_setup.after_optimizer_step(self.model, self.config, train_progress)

                        if self.model.ema:
                            assert multi.is_master()
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay",
                                self.model.ema.get_current_decay(update_step),
                                train_progress.global_step
                            )
                            self.model.ema.step(
                                self.parameters,
                                update_step
                            )

                        self.one_step_trained = True
                        diagnostics.step(train_progress.global_step, train_progress.epoch, accumulated_loss_cpu)

                        optimizer_steps += 1
                        if optimizer_steps == _D_AT_INITIAL_CHECK_STEPS and multi.is_master() \
                                and self.config.optimizer.optimizer.is_adaptive:
                            message = create.d_at_initial_training_warning(
                                self.config, self.model.optimizer, _D_AT_INITIAL_CHECK_STEPS)
                            if message:
                                tqdm.write(message)

                if self.config.validation and multi.is_master():
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    return

            if epoch_batch_count == 0 and epoch_started_at_step == 0:
                # an empty epoch trains nothing, and the following ones are normally just as empty, so the run would
                # count through its epochs without a step and end without saving
                no_images = ("the enabled concepts have no images (check their paths, and Include Subdirectories for "
                             "images in subfolders)")
                if self.config.batch_size <= 1:
                    raise RuntimeError(f"This epoch has no batches, so nothing can be trained: {no_images}.")
                raise RuntimeError(
                    f"This epoch has no batches, so nothing can be trained: either {no_images}, or no aspect ratio "
                    f"bucket has {self.config.batch_size} samples (batches never mix buckets, and leftover samples are "
                    f"skipped). Lower the batch size or add images"
                    f"{', or turn off aspect ratio bucketing' if self.config.aspect_ratio_bucketing else ''}."
                )

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                return

    def end(self):
        if self.one_step_trained:
            self.model.evict()

            if self.config.backup_before_save and multi.is_master():
                self.__backup(self.model.train_progress)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            if multi.is_master():
                self.callbacks.on_update_status("Saving the final model")

                if self.model.ema:
                    self.model.ema.copy_ema_to(self.parameters, store_temp=False)
                save_path = get_output_model_destination(
                    self.config.final_output_dir,
                    self.config.run_name,
                    self.config.output_model_format,
                )
                print("Saving " + save_path)

                with diagnostics.phase("final save", path=save_path):
                    self.model_saver.save(
                        model=self.model,
                        model_type=self.config.model_type,
                        output_model_format=self.config.output_model_format,
                        output_model_destination=save_path,
                        dtype=self.config.output_dtype.torch_dtype()
                    )
                diagnostics.output(save_path)

        if self.model is not None:
            self.model.evict()

        if multi.is_master():
            self.tensorboard.close()

            if self.config.tensorboard and not self.config.tensorboard_always_on:
                super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
        diagnostics.close()
