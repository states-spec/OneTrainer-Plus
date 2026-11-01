import ast
import importlib
from collections.abc import Iterable

import modules.util.multi_gpu_util as multi
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.EMAModule import EMAModuleWrapper
from modules.util import factory
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig, TrainOptimizerConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.lr_scheduler_util import (
    lr_lambda_constant,
    lr_lambda_cosine,
    lr_lambda_cosine_with_hard_restarts,
    lr_lambda_cosine_with_restarts,
    lr_lambda_linear,
    lr_lambda_rex,
    lr_lambda_warmup,
)
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer.adafactor_extensions import patch_adafactor
from modules.util.optimizer.adam_extensions import patch_adam
from modules.util.optimizer.adamw_extensions import patch_adamw
from modules.util.optimizer.muon_util import split_parameters_for_muon
from modules.util.TrainProgress import TrainProgress
from modules.zluda import ZLUDA

import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, SequentialLR

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    SchedulerMixin,
    UniPCMultistepScheduler,
)

factory.import_dir("modules/modelSampler", "modules.modelSampler")
factory.import_dir("modules/modelLoader", "modules.modelLoader")
factory.import_dir("modules/modelSaver", "modules.modelSaver")
factory.import_dir("modules/modelSetup", "modules.modelSetup")
factory.import_dir("modules/dataLoader", "modules.dataLoader")

def create_model_loader(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelLoader | None:
    cls = factory.get(BaseModelLoader, model_type, training_method)
    return cls() if cls is not None else None


def create_model_saver(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSaver | None:
    cls = factory.get(BaseModelSaver, model_type, training_method)
    return cls() if cls is not None else None

def get_model_setup_class(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> type | None:
    return factory.get(BaseModelSetup, model_type, training_method)

def create_model_setup(
        model_type: ModelType,
        train_device: torch.device,
        temp_device: torch.device,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        debug_mode: bool = False,
) -> BaseModelSetup | None:
    cls = factory.get(BaseModelSetup, model_type, training_method)
    return cls(train_device, temp_device, debug_mode) if cls is not None else None

def create_model_sampler(
        train_device: torch.device,
        temp_device: torch.device,
        model: BaseModel,
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSampler:
    cls = factory.get(BaseModelSampler, model_type, training_method)
    if cls is None:
        cls = factory.get(BaseModelSampler, model_type)
    return cls(train_device, temp_device, model, model_type) if cls is not None else None

def create_data_loader(
        train_device: torch.device,
        temp_device: torch.device,
        model: BaseModel,
        model_type: ModelType,
        model_setup: BaseModelSetup,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        config: TrainConfig = None,
        train_progress: TrainProgress | None = None,
        is_validation: bool = False
) -> BaseDataLoader | None:
    # Layer offloading uses a non-thread-safe conductor. Only text encoders run inside the caching dataloader's
    # worker threads (to produce the text cache), so only their offload_fraction can conflict with threading.
    if config.dataloader_threads > 1 and any(
        getattr(config, name).offload_fraction > 0 for name in model_type.text_encoder_parts()
    ):
        raise RuntimeError('layer offloading can not be activated for a text encoder if "dataloader_threads" > 1')

    if train_progress is None:
        train_progress = TrainProgress()

    cls = factory.get(BaseDataLoader, model_type, training_method)
    if cls is None:
        cls = factory.get(BaseDataLoader, model_type)
    return cls(train_device, temp_device, config, model, model_setup, train_progress, is_validation) if cls is not None else None


def _adv_orthograd_mode(value) -> str:
    # adv_optm >= 2.5 takes an OrthoGrad mode string; a bool (a config that skipped migration) means the former OrthoGrad
    if isinstance(value, bool) or value in ("True", "False"):
        return "flattened" if value in (True, "True") else "disabled"
    return value if value is not None else "disabled"


def _bnb_lars_on_momentum_kernels(optimizer):
    # bitsandbytes' LARS is SGD with momentum whose update is limited to max_unorm times the weight norm, run on
    # momentum's kernels. bitsandbytes 0.49 has no "lars" entry in its 32-bit kernel table (its 8-bit table still maps
    # "lars" to momentum's kernels, as older releases did for both), so every step on 32-bit state raised "Unsupported
    # optimizer name: lars": all of LARS, and LARS_8BIT's tensors smaller than min_8bit_size. Momentum's name selects
    # the same kernels on every path. bitsandbytes doesn't save the name with the optimizer state.
    optimizer.optimizer_name = "momentum"
    return optimizer


def _adv_state_precision(optimizer: Optimizer, state_precision: str | None, name: str = "state_precision") -> str:
    from modules.util.optimizer_util import adv_state_precisions
    state_precision = state_precision.lower() if state_precision is not None else "auto"
    if state_precision not in adv_state_precisions(optimizer):
        raise ValueError(f'{name} "{state_precision}" does not work with {optimizer} in adv_optm '
                         f'(use one of {", ".join(adv_state_precisions(optimizer))})')
    return state_precision


def _adv_common_kwargs(optimizer_config: TrainOptimizerConfig, nesterov_default: bool = False) -> dict:
    # settings every adv_optm optimizer except Lion_adv takes
    state_precision = _adv_state_precision(optimizer_config.optimizer, optimizer_config.state_precision)
    if optimizer_config.optimizer == Optimizer.SINKSGD_ADV and not optimizer_config.momentum:
        # without momentum SinkSGD_adv keeps no state to factor, and adv_optm 2.5.13-2.6.0 fails on "factored" then
        state_precision = "auto" if state_precision == "factored" else state_precision
    return {
        "orthogonal_gradient": _adv_orthograd_mode(optimizer_config.orthogonal_gradient),
        "nesterov": optimizer_config.nesterov if optimizer_config.nesterov is not None else nesterov_default,
        "nesterov_coef": optimizer_config.nesterov_coef,
        "state_precision": state_precision,
        "centered_wd": optimizer_config.centered_wd if optimizer_config.centered_wd is not None else 0.0,
        "centered_wd_mode": optimizer_config.centered_wd_mode if optimizer_config.centered_wd_mode is not None else "full",
        "spectral_normalization": optimizer_config.spectral_normalization if optimizer_config.spectral_normalization is not None else False,
        **_adv_scaled_wd_kwargs(optimizer_config),
    }


def _adv_scaled_wd_kwargs(optimizer_config: TrainOptimizerConfig) -> dict:
    # every adv_optm optimizer takes scaled_wd since 2.6.0. Passed only when it is on, so a venv that still has
    # adv_optm 2.5 (a git pull without update.sh) trains as before; check_optimizer_config stops if it is on there
    return {"scaled_wd": True} if optimizer_config.scaled_wd else {}


def _adv_optm_has_scaled_wd() -> bool:
    import inspect

    from adv_optm import AdamW_adv
    return "scaled_wd" in inspect.signature(AdamW_adv.__init__).parameters


def _sinksgd_oft_mode_groups(parameters: list[dict]) -> list[str]:
    # adv_optm 2.6's SinkSGD_adv normalizes OFT updates with a skew-symmetric Sinkhorn (oft_mode), which treats every
    # matrix of the group as OFT blocks of shape (r, b * (b - 1) / 2). Upstream turns it on for the whole optimizer
    # whenever the config's adapter type is OFT, which also reaches full fine-tuning and embedding runs of a config that
    # still has OFT selected, and the additional embeddings trained next to an OFT LoRA: their matrices fail in the
    # first step (index_put shape mismatch), or are normalized as rotations if a width happens to fit. So it is set per
    # group, only where every matrix is a tagged OFT block (tag_util); vectors, such as DoRA-OFT's magnitudes, take
    # SinkSGD_adv's sign path either way
    oft_groups = []
    for group in parameters:
        matrices = [p for p in group['params'] if p.ndim >= 2 and not getattr(p, '_is_dora_scale', False)
                    and not getattr(p, 'is_vector', False)]
        group['oft_mode'] = bool(matrices) and all(getattr(p, '_is_oft', False) for p in matrices)
        if group['oft_mode']:
            oft_groups.append(str(group.get('name')))
    return oft_groups


def _adv_adam_kwargs(optimizer_config: TrainOptimizerConfig) -> dict:
    # settings only the Adam-style adv_optm optimizers (AdamW, Adopt, Prodigy) take
    return {
        "fisher_wd": optimizer_config.fisher_wd if optimizer_config.fisher_wd is not None else False,
        "factored_2nd": optimizer_config.factored_2nd if optimizer_config.factored_2nd is not None else False,
    }


def _adv_optm_version() -> str:
    import adv_optm
    return getattr(adv_optm, "__version__", "unknown")


# adv_optm minor versions whose optimizer states are interchangeable. 2.6.0 only added settings (scaled_wd,
# SinkSGD_adv's oft_mode), which a group saved without them gets from the config (create_optimizer adds missing
# settings), and stores every state tensor as 2.5 does
_ADV_OPTM_COMPATIBLE_STATES = ({"2.5", "2.6"},)


def _adv_optm_state_family(version: str) -> set[str]:
    minor = ".".join(version.split(".")[:2])
    return next((family for family in _ADV_OPTM_COMPATIBLE_STATES if minor in family), {minor})


def _check_adv_optm_state_version(state_dict: dict) -> dict | None:
    # adv_optm changes its state layout between some minor versions (2.5 can not load the state of 2.2), so the state
    # of an incompatible version is dropped: the training resumes with fresh optimizer statistics instead of failing
    saved_version = state_dict.get("adv_optm_version")
    current_version = _adv_optm_version()
    if saved_version is not None and _adv_optm_state_family(saved_version) == _adv_optm_state_family(current_version):
        return state_dict
    print(f"Warning: the saved optimizer state was written by adv_optm {saved_version or 'older than 2.5'}, "
          f"which adv_optm {current_version} can not load. The optimizer state is reset; "
          f"the model weights, EMA and training progress are still restored.")
    return None


def _restore_adv_optm_state_dtypes(optimizer: torch.optim.Optimizer, state_dict: dict):
    # torch's load_state_dict casts every state tensor to the dtype of its parameter, and adv_optm 2.5 then casts
    # them to one dtype per group. That turns signed int8 states (int8_sr) into uint8, rounds fp32 states of bf16
    # weights to bf16 precision, and gives the fp32 states of vectors in "factored" mode the weight dtype, which fails
    # in the next step. The state was saved by an adv_optm version with the same state layout (checked before), so the
    # saved tensors are exactly what the optimizer uses: put them back.
    for saved_group, group in zip(state_dict['param_groups'], optimizer.param_groups, strict=True):
        for saved_index, p in zip(saved_group['params'], group['params'], strict=True):
            saved_state = state_dict['state'].get(saved_index)
            state = optimizer.state.get(p)
            if not saved_state or not state:
                continue
            for key, saved_value in saved_state.items():
                value = state.get(key)
                if isinstance(saved_value, torch.Tensor) and isinstance(value, torch.Tensor) \
                        and value.shape == saved_value.shape:
                    state[key] = saved_value.to(device=value.device, copy=True)


# adv_optm's Prodigy_adv (2.5.13-2.6.0) keeps its momentum scaled by d, but its Nesterov step mixes in the unscaled gradient
# (exp_avg.lerp_(grad, 1 - coef)). That term is 1/d times too large, so every step moves each weight by about
# 0.1 * lr whatever d is: at lr 1 a LoRA turned to noise within the first steps. The other adv_optm optimizers keep
# their momentum in gradient units and are not affected.
_PRODIGY_ADV_NESTEROV = ("Nesterov momentum is broken in PRODIGY_ADV (adv_optm 2.5.13 and every later release and "
                         "branch so far): it moves every weight by about 0.1 x the learning rate per step regardless of "
                         "Prodigy's step size, which destroys the model")


def check_optimizer_config(optimizer_config: TrainOptimizerConfig):
    # turns off settings known to break training; GenericTrainer.start calls it before the model loads. The UI hides
    # these settings for the optimizer, so a value saved earlier could not be changed there.
    if optimizer_config.optimizer == Optimizer.PRODIGY_ADV and optimizer_config.nesterov:
        print(f"Warning: {_PRODIGY_ADV_NESTEROV}. The config has it on; it is turned off for this run.")
        optimizer_config.nesterov = False

    if optimizer_config.optimizer.is_adv_optm and optimizer_config.scaled_wd and not _adv_optm_has_scaled_wd():
        raise ValueError(f"Scaled Weight Decay needs adv_optm 2.6.0, but {_adv_optm_version()} is installed. Run "
                         f"update.sh (update.bat on Windows) to install the current requirements.")

    # adv_optm 2.5.13-2.6.0: Adopt_adv's fp16 state fails in its first step together with these (a dtype error,
    # "expected dtype c10::Half for `end`"), after the model has loaded and the cache is built
    if optimizer_config.optimizer == Optimizer.ADOPT_ADV and (optimizer_config.state_precision or "").lower() == "fp16":
        clashing = [title for title, on in (("Nesterov", optimizer_config.nesterov),
                                            ("Factored 2nd Moment", optimizer_config.factored_2nd)) if on]
        if clashing:
            raise ValueError(f'State Precision "fp16" does not work with {" or ".join(clashing)} in ADOPT_ADV: '
                             f'adv_optm fails in the first step. Pick another State Precision, or turn '
                             f'{" and ".join(clashing)} off.')

    # torch's fused and foreach Adam(W) steps can't step one parameter at a time or round stochastically.
    # create_optimizer refuses that too, but only after the model has loaded
    if optimizer_config.optimizer in (Optimizer.ADAM, Optimizer.ADAMW) \
            and (optimizer_config.fused or optimizer_config.foreach):
        needing = [title for title, on in (("Fused Back Pass", optimizer_config.fused_back_pass),
                                           ("Stochastic Rounding", optimizer_config.stochastic_rounding)) if on]
        if needing:
            raise ValueError(f'{" and ".join(needing)} only work{"s" if len(needing) == 1 else ""} with '
                             f'{optimizer_config.optimizer}\'s plain implementation: turn Fused and ForEach off in '
                             f'the optimizer settings, or turn {" and ".join(needing)} off.')


def _disable_prodigy_adv_nesterov(optimizer: torch.optim.Optimizer):
    # a resumed group keeps the backup's settings, so a backup made with Nesterov on would still use it
    if any(group.get('nesterov') for group in optimizer.param_groups):
        print(f"Warning: {_PRODIGY_ADV_NESTEROV}. The backup was made with it on; it is turned off for this run.")
        for group in optimizer.param_groups:
            group['nesterov'] = False


def _effective_snr_cond(optimizer_config: TrainOptimizerConfig) -> bool:
    # adv_optm applies SNR preconditioning only with normed momentum and momentum > 0, but its constructor raises when
    # the option is set without both of them; the UI greys the option out then, so a value saved earlier stays set
    return bool(optimizer_config.snr_cond and optimizer_config.normed_momentum and (optimizer_config.momentum or 0) > 0)


def _restore_kourkoutas_helper(optimizer: torch.optim.Optimizer):
    # adv_optm creates the Kourkoutas-beta helper only in the constructor, when kourkoutas_beta is set there, but each
    # step reads the flag from the param group. A resumed group keeps the backup's flag, so resuming a Kourkoutas
    # backup with the option turned off failed in the first step ("no attribute 'kourkoutas_helper'").
    if getattr(optimizer, 'kourkoutas_helper', None) is None and any(
            group.get('kourkoutas_beta') or group.get('adam_kourkoutas_beta') for group in optimizer.param_groups):
        from adv_optm.util.Kourkoutas import KourkoutasHelper
        optimizer.kourkoutas_helper = KourkoutasHelper(optimizer)


def _drop_mismatched_kourkoutas_state(optimizer: torch.optim.Optimizer):
    # Kourkoutas-beta keeps one running gradient norm per layer, per row for tagged LoRA/OFT parameters (see
    # tag_util). A backup made before the parameters were tagged has one value per layer, which fails in the first
    # step ("output with shape [] doesn't match the broadcast shape [8, 1]"). Those averages are dropped and rebuilt.
    dropped = 0
    for p, state in optimizer.state.items():
        r_ema = state.get('kourkoutas_r_ema')
        if r_ema is None:
            continue
        if getattr(p, '_is_oft', False) or getattr(p, '_is_lora_A', False):
            shape = (p.shape[0], 1)
        elif getattr(p, '_is_lora_B', False):
            shape = (1, p.shape[1])
        else:
            shape = ()
        if tuple(r_ema.shape) != shape:
            del state['kourkoutas_r_ema']
            dropped += 1
    if dropped:
        print(f"Warning: the backup's Kourkoutas-beta averages of {dropped} parameters were made before the adapter "
              f"parameters were tagged for adv_optm. They are reset and rebuilt over the next steps.")


# param group keys that hold a setting from the config (TrainOptimizerConfig fields, plus the ones the optimizers
# name differently); the other keys are the optimizer's own running values, like Prodigy's d
_OPTIMIZER_SETTING_ALIASES = {"betas", "compiled_optimizer"}


def _changed_optimizer_settings(group_name: str, saved_group: dict, new_group: dict) -> list[str]:
    # "k" is also Prodigy's step counter
    settings = (set(TrainOptimizerConfig.default_values().types) | _OPTIMIZER_SETTING_ALIASES) - {"optimizer", "k"}
    changed = []
    for key in sorted(settings & saved_group.keys() & new_group.keys()):
        try:
            differs = bool(saved_group[key] != new_group[key])
        except (RuntimeError, TypeError, ValueError):
            continue
        if differs:
            changed.append(f"{group_name}.{key}: backup {saved_group[key]!r}, config {new_group[key]!r}")
    return changed


def _d_restart_note(optimizer_config: TrainOptimizerConfig, saved_group: dict | None, learning_rate: float) -> str:
    # Resuming the backup's weights without its optimizer state is a short re-warmup for Adam-like optimizers, but one
    # that adapts its step size D (Prodigy, D-Adaptation) starts again at Initial D and measures D by how far the
    # weights move from where it starts. From weights that are already trained they move little: a Chroma LoRA
    # resumed from an ADOPT_ADV backup kept D at Initial D (1e-4) for 450 steps, a fresh one reached 1e-2.
    if not optimizer_config.optimizer.is_adaptive:
        return ""
    note = (f" {optimizer_config.optimizer} starts its step size D again at Initial D ({optimizer_config.d0}) and "
            f"estimates it from how far the weights move from the backup's, which are already trained, so D can stay "
            f"at Initial D and the training barely moves.")
    saved_lr = None if saved_group is None or saved_group.get("lr") is None \
        else saved_group["lr"] * saved_group.get("d", 1.0)
    if saved_lr and learning_rate:
        note += (f" The backup's run trained with a step size of {saved_lr:.3g}: an Initial D of about "
                 f"{saved_lr / learning_rate:.3g} (at learning rate {learning_rate:g}) continues at that pace.")
    return note


def create_optimizer(
        parameter_group_collection: NamedParameterGroupCollection,
        state_dict: dict | None,
        config: TrainConfig,
        layer_key_fn: dict[int, str] | None = None,
) -> torch.optim.Optimizer | None:
    optimizer = None
    optimizer_config = config.optimizer

    if optimizer_config.optimizer is None:
        return None

    # unset settings (e.g. from a hand-written config) use the UI's defaults, not the fallbacks below
    from modules.util.optimizer_util import fill_unset_optimizer_settings
    optimizer_config = fill_unset_optimizer_settings(optimizer_config)

    # a trained, layer-offloaded part has its params evicted during the back pass, so it needs fused_back_pass
    if any(part.offload_fraction > 0 and part.train for part in config.model_part_configs()):
        if (not optimizer_config.optimizer.supports_fused_back_pass() or not optimizer_config.fused_back_pass) \
                and config.training_method == TrainingMethod.FINE_TUNE:
            raise RuntimeError('layer offloading can only be used for fine tuning when using an optimizer that supports "fused_back_pass"')

    parameters = parameter_group_collection.parameters_for_optimizer(config)

    match config.optimizer.optimizer:

        # SGD Optimizer
        case Optimizer.SGD:
            optimizer = torch.optim.SGD(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                dampening=optimizer_config.dampening if optimizer_config.dampening is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else False,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False,
                maximize=optimizer_config.maximize if optimizer_config.maximize is not None else False,
                differentiable=optimizer_config.differentiable if optimizer_config.differentiable is not None else False,
            )

        # SGD_8BIT Optimizer
        case Optimizer.SGD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.SGD8bit(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                dampening=optimizer_config.dampening if optimizer_config.dampening is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else False,
            )

        # ADAM Optimizer
        case Optimizer.ADAM:
            if optimizer_config.stochastic_rounding \
                    and (optimizer_config.fused or optimizer_config.foreach):
                raise RuntimeError('"stochastic_rounding" is only allowed when "fused" and "foreach" are disabled')

            if optimizer_config.fused_back_pass \
                    and (optimizer_config.fused or optimizer_config.foreach):
                raise RuntimeError('"fused_back_pass" is only allowed when "fused" and "foreach" are disabled')

            optimizer = torch.optim.Adam(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False,
                maximize=optimizer_config.maximize if optimizer_config.maximize is not None else False,
                capturable=optimizer_config.capturable if optimizer_config.capturable is not None else False,
                differentiable=optimizer_config.differentiable if optimizer_config.differentiable is not None else False,
                fused=optimizer_config.fused if optimizer_config.fused is not None else False,
            )

            if optimizer_config.stochastic_rounding or optimizer_config.fused_back_pass:
                patch_adam(optimizer, optimizer_config.stochastic_rounding)

        # ADAMW Optimizer
        case Optimizer.ADAMW:
            if optimizer_config.stochastic_rounding \
                    and (optimizer_config.fused or optimizer_config.foreach):
                raise RuntimeError('"stochastic_rounding" is only allowed when "fused" and "foreach" are disabled')

            if optimizer_config.fused_back_pass \
                    and (optimizer_config.fused or optimizer_config.foreach):
                raise RuntimeError('"fused_back_pass" is only allowed when "fused" and "foreach" are disabled')

            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False,
                maximize=optimizer_config.maximize if optimizer_config.maximize is not None else False,
                capturable=optimizer_config.capturable if optimizer_config.capturable is not None else False,
                differentiable=optimizer_config.differentiable if optimizer_config.differentiable is not None else False,
                fused=optimizer_config.fused if optimizer_config.fused is not None else False,
            )

            if optimizer_config.stochastic_rounding or optimizer_config.fused_back_pass:
                patch_adamw(optimizer, optimizer_config.stochastic_rounding)

        # ADAM_8BIT Optimizer
        case Optimizer.ADAM_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # ADAMW_8BIT Optimizer
        case Optimizer.ADAMW_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # AdEMAMix_8BIT Optimizer
        case Optimizer.AdEMAMix_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdEMAMix8bit(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta1 is not None else 0.9999,),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 5,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # AdEMAMix Optimizer
        case Optimizer.AdEMAMix:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdEMAMix(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta1 is not None else 0.9999,),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 5,
                optim_bits=optimizer_config.optim_bits if optimizer_config.optim_bits is not None else 32,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # ADAGRAD Optimizer
        case Optimizer.ADAGRAD:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-10,
                lr_decay=optimizer_config.lr_decay if optimizer_config.lr_decay is not None else 0,
                initial_accumulator_value=optimizer_config.initial_accumulator_value if optimizer_config.initial_accumulator_value is not None else 0,
            )

        # ADAGRAD_8BIT Optimizer
        case Optimizer.ADAGRAD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad8bit(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-10,
                lr_decay=optimizer_config.lr_decay if optimizer_config.lr_decay is not None else 0,
                initial_accumulator_value=optimizer_config.initial_accumulator_value if optimizer_config.initial_accumulator_value is not None else 0,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
            )

        # RMSPROP Optimizer
        case Optimizer.RMSPROP:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 0.99,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                centered=optimizer_config.centered if optimizer_config.centered is not None else False,
            )

        # RMSPROP_8BIT Optimizer
        case Optimizer.RMSPROP_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop8bit(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 0.99,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                centered=optimizer_config.centered if optimizer_config.centered is not None else False,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
            )

        # LION Optimizer
        case Optimizer.LION:
            import lion_pytorch as lp
            optimizer = lp.Lion(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                use_triton=optimizer_config.use_triton if optimizer_config.use_triton is not None else False,
            )

        # LARS Optimizer
        case Optimizer.LARS:
            import bitsandbytes as bnb
            optimizer = _bnb_lars_on_momentum_kernels(bnb.optim.LARS(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                dampening=optimizer_config.dampening if optimizer_config.dampening is not None else 0,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else False,
                max_unorm=optimizer_config.max_unorm if optimizer_config.max_unorm is not None else 0.02,
            ))

        # LARS_8BIT Optimizer
        case Optimizer.LARS_8BIT:
            import bitsandbytes as bnb
            optimizer = _bnb_lars_on_momentum_kernels(bnb.optim.LARS8bit(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                dampening=optimizer_config.dampening if optimizer_config.dampening is not None else 0,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else False,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                max_unorm=optimizer_config.max_unorm if optimizer_config.max_unorm is not None else 0.02,
            ))

        # LAMB Optimizer
        case Optimizer.LAMB:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB(
                params=parameters,
                lr=config.learning_rate,
                bias_correction=optimizer_config.bias_correction if optimizer_config.bias_correction is not None else True,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                adam_w_mode=optimizer_config.adam_w_mode if optimizer_config.adam_w_mode is not None else True,
                optim_bits=optimizer_config.optim_bits if optimizer_config.optim_bits is not None else 32,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else False,
                max_unorm=optimizer_config.max_unorm if optimizer_config.max_unorm is not None else 1.0,
            )

        # LAMB_8BIT Optimizer
        case Optimizer.LAMB_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB8bit(
                params=parameters,
                lr=config.learning_rate,
                bias_correction=optimizer_config.bias_correction if optimizer_config.bias_correction is not None else True,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                adam_w_mode=optimizer_config.adam_w_mode if optimizer_config.adam_w_mode is not None else True,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else False,
                max_unorm=optimizer_config.max_unorm if optimizer_config.max_unorm is not None else 1.0,
            )

        # LION_8BIT Optimizer
        case Optimizer.LION_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Lion8bit(
                params=parameters,
                lr=config.learning_rate if config.learning_rate is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # Schedule-free AdamW
        case Optimizer.SCHEDULE_FREE_ADAMW:
            if config.model_type.is_wuerstchen_v2() or config.model_type.is_stable_cascade():
                raise NotImplementedError("Cannot use schedule-free optimizers with Wuerstchen-based models.")
            from schedulefree import AdamWScheduleFree
            optimizer = AdamWScheduleFree(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                warmup_steps=config.learning_rate_warmup_steps,
                r=optimizer_config.r if optimizer_config.r is not None else 0,
                weight_lr_power=optimizer_config.weight_lr_power if optimizer_config.weight_lr_power is not None else 2.0,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False
            )

        # Schedule-free SGD
        case Optimizer.SCHEDULE_FREE_SGD:
            if config.model_type.is_wuerstchen_v2() or config.model_type.is_stable_cascade():
                raise NotImplementedError("Cannot use schedule-free optimizers with Wuerstchen models.")
            from schedulefree import SGDScheduleFree
            optimizer = SGDScheduleFree(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                warmup_steps=config.learning_rate_warmup_steps,
                r=optimizer_config.r if optimizer_config.r is not None else 0,
                weight_lr_power=optimizer_config.weight_lr_power if optimizer_config.weight_lr_power is not None else 2.0,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False
            )

        # DADAPT_SGD Optimizer
        case Optimizer.DADAPT_SGD:
            import dadaptation as da
            optimizer = da.DAdaptSGD(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0.0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                fsdp_in_use=optimizer_config.fsdp_in_use if optimizer_config.fsdp_in_use is not None else False,
            )

        # DADAPT_ADAM Optimizer
        case Optimizer.DADAPT_ADAM:
            import dadaptation as da
            optimizer = da.DAdaptAdam(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                decouple=optimizer_config.decouple if optimizer_config.decouple is not None else False,
                use_bias_correction=optimizer_config.use_bias_correction if optimizer_config.use_bias_correction is not None else False,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                fsdp_in_use=optimizer_config.fsdp_in_use if optimizer_config.fsdp_in_use is not None else False,
            )

        # DADAPT_ADAN Optimizer
        case Optimizer.DADAPT_ADAN:
            import dadaptation as da
            optimizer = da.DAdaptAdan(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.98,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.92,
                       optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.99),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.02,
                no_prox=optimizer_config.no_prox if optimizer_config.no_prox is not None else False,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
            )

        # DADAPT_ADA_GRAD Optimizer
        case Optimizer.DADAPT_ADA_GRAD:
            import dadaptation as da
            optimizer = da.DAdaptAdaGrad(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 0.0,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
            )

        # DADAPT_LION Optimizer
        case Optimizer.DADAPT_LION:
            import dadaptation as da
            optimizer = da.DAdaptLion(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                fsdp_in_use=optimizer_config.fsdp_in_use if optimizer_config.fsdp_in_use is not None else False,
            )

        # PRODIGY Optimizer
        case Optimizer.PRODIGY:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                beta3=optimizer_config.beta3 if optimizer_config.beta3 is not None else None,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                decouple=optimizer_config.decouple if optimizer_config.decouple is not None else True,
                use_bias_correction=optimizer_config.use_bias_correction if optimizer_config.use_bias_correction is not None else False,
                safeguard_warmup=optimizer_config.safeguard_warmup if optimizer_config.safeguard_warmup is not None else False,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                d_coef=optimizer_config.d_coef if optimizer_config.d_coef is not None else 1.0,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                fsdp_in_use=optimizer_config.fsdp_in_use if optimizer_config.fsdp_in_use is not None else False,
                slice_p=optimizer_config.slice_p if optimizer_config.slice_p is not None else 1,
            )

        # PRODIGY_PLUS_SCHEDULE_FREE Optimizer
        case Optimizer.PRODIGY_PLUS_SCHEDULE_FREE:
            from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
            optimizer = ProdigyPlusScheduleFree(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                beta3=optimizer_config.beta3 if optimizer_config.beta3 is not None else None,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                weight_decay_by_lr=optimizer_config.weight_decay_by_lr if optimizer_config.weight_decay_by_lr is not None else True,
                use_bias_correction=optimizer_config.use_bias_correction if optimizer_config.use_bias_correction is not None else False,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                d_coef=optimizer_config.d_coef if optimizer_config.d_coef is not None else 1.0,
                prodigy_steps=optimizer_config.prodigy_steps if optimizer_config.prodigy_steps is not None else 0,
                use_speed=optimizer_config.use_speed if optimizer_config.use_speed is not None else False,
                eps=optimizer_config.eps if optimizer_config.eps is not None else None,
                split_groups=optimizer_config.split_groups if optimizer_config.split_groups is not None else True,
                split_groups_mean=optimizer_config.split_groups_mean if optimizer_config.split_groups_mean is not None else True,
                factored=optimizer_config.factored if optimizer_config.factored is not None else True,
                factored_fp32=optimizer_config.factored_fp32 if optimizer_config.factored_fp32 is not None else True,
                fused_back_pass=optimizer_config.fused_back_pass if optimizer_config.fused_back_pass is not None else False,
                use_stableadamw=optimizer_config.use_stableadamw if optimizer_config.use_stableadamw is not None else True,
                use_cautious=optimizer_config.use_cautious if optimizer_config.use_cautious is not None else False,
                use_grams=optimizer_config.use_grams if optimizer_config.use_grams is not None else False,
                use_adopt=optimizer_config.use_adopt if optimizer_config.use_adopt is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding if optimizer_config.stochastic_rounding is not None else True,
                d_limiter=optimizer_config.d_limiter if optimizer_config.d_limiter is not None else True,
                use_schedulefree=optimizer_config.use_schedulefree if optimizer_config.use_schedulefree is not None else True,
                schedulefree_c=optimizer_config.schedulefree_c if optimizer_config.schedulefree_c is not None else 0.0,
                use_orthograd=optimizer_config.use_orthograd if optimizer_config.use_orthograd is not None else False,
            )

        # ADAFactor Optimizer
        case Optimizer.ADAFACTOR:
            from transformers.optimization import Adafactor

            if optimizer_config.relative_step:
                for parameter in parameters:
                    if isinstance(parameter, dict) and 'lr' in parameter:
                        parameter.pop('lr')

            optimizer = Adafactor(
                params=parameters,
                lr=None if optimizer_config.relative_step is True else config.learning_rate,
                eps=(optimizer_config.eps if optimizer_config.eps is not None else 1e-30,
                     optimizer_config.eps2 if optimizer_config.eps2 is not None else 1e-3),
                clip_threshold=optimizer_config.clip_threshold if optimizer_config.clip_threshold is not None else 1.0,
                decay_rate=optimizer_config.decay_rate if optimizer_config.decay_rate is not None else -0.8,
                beta1=optimizer_config.beta1 if optimizer_config.beta1 is not None else None,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                scale_parameter=optimizer_config.scale_parameter if optimizer_config.scale_parameter is not None else True,
                relative_step=optimizer_config.relative_step if optimizer_config.relative_step is not None else True,
                warmup_init=optimizer_config.warmup_init if optimizer_config.warmup_init is not None else False,
            )

            patch_adafactor(optimizer, optimizer_config.stochastic_rounding)

        # CAME Optimizer
        case Optimizer.CAME:
            from modules.util.optimizer.CAME import CAME
            optimizer = CAME(
                params=parameters,
                lr=config.learning_rate,
                eps=(optimizer_config.eps if optimizer_config.eps is not None else 1e-30,
                     optimizer_config.eps2 if optimizer_config.eps2 is not None else 1e-16),
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.9999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                use_cautious=optimizer_config.use_cautious,
            )

        # CAME_8BIT Optimizer
        case Optimizer.CAME_8BIT:
            from modules.util.optimizer.CAME8bit import CAME8bit
            optimizer = CAME8bit(
                params=parameters,
                lr=config.learning_rate,
                eps=(optimizer_config.eps if optimizer_config.eps is not None else 1e-30,
                     optimizer_config.eps2 if optimizer_config.eps2 is not None else 1e-16),
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.9999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 16384,
                quant_block_size=optimizer_config.quant_block_size if optimizer_config.quant_block_size is not None else 2048
            )

        # ADAMW_ADV Optimizer
        case Optimizer.ADAMW_ADV:
            from adv_optm import AdamW_adv
            optimizer = AdamW_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                use_atan2=optimizer_config.use_atan2 if optimizer_config.use_atan2 is not None else False,
                normed_momentum=optimizer_config.normed_momentum if optimizer_config.normed_momentum is not None else False,
                kourkoutas_beta=optimizer_config.kourkoutas_beta if optimizer_config.kourkoutas_beta is not None else False,
                k_warmup_steps=(config.learning_rate_warmup_steps / config.gradient_accumulation_steps),
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                **_adv_common_kwargs(optimizer_config),
                **_adv_adam_kwargs(optimizer_config),
            )

        # ADOPT_ADV Optimizer
        case Optimizer.ADOPT_ADV:
            from adv_optm import Adopt_adv
            optimizer = Adopt_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.9999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-6,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                use_atan2=optimizer_config.use_atan2 if optimizer_config.use_atan2 is not None else False,
                kourkoutas_beta=optimizer_config.kourkoutas_beta if optimizer_config.kourkoutas_beta is not None else False,
                k_warmup_steps=(config.learning_rate_warmup_steps / config.gradient_accumulation_steps),
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                **_adv_common_kwargs(optimizer_config),
                **_adv_adam_kwargs(optimizer_config),
            )

        # PRODIGY_ADV Optimizer
        case Optimizer.PRODIGY_ADV:
            from adv_optm import Prodigy_adv
            check_optimizer_config(optimizer_config)
            optimizer = Prodigy_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                beta3=optimizer_config.beta3 if optimizer_config.beta3 is not None else None,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                d_coef=optimizer_config.d_coef if optimizer_config.d_coef is not None else 1.0,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                slice_p=optimizer_config.slice_p if optimizer_config.slice_p is not None else 1,
                prodigy_steps=optimizer_config.prodigy_steps if optimizer_config.prodigy_steps is not None else 0,
                d_limiter=optimizer_config.d_limiter if optimizer_config.d_limiter is not None else False,
                use_atan2=optimizer_config.use_atan2 if optimizer_config.use_atan2 is not None else False,
                kourkoutas_beta=optimizer_config.kourkoutas_beta if optimizer_config.kourkoutas_beta is not None else False,
                k_warmup_steps=(config.learning_rate_warmup_steps / config.gradient_accumulation_steps),
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                **_adv_common_kwargs(optimizer_config),
                **_adv_adam_kwargs(optimizer_config),
            )

        # SignSGD_ADV Optimizer
        case Optimizer.SIGNSGD_ADV:
            from adv_optm import SignSGD_adv
            optimizer = SignSGD_adv(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                geometric_wd=optimizer_config.geometric_wd if optimizer_config.geometric_wd is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                stochastic_sign=optimizer_config.stochastic_sign if optimizer_config.stochastic_sign is not None else False,
                normed_momentum=optimizer_config.normed_momentum if optimizer_config.normed_momentum is not None else False,
                snr_cond=_effective_snr_cond(optimizer_config),
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                **_adv_common_kwargs(optimizer_config),
            )

        # SinkSGD_ADV Optimizer
        case Optimizer.SINKSGD_ADV:
            from adv_optm import SinkSGD_adv
            if oft_groups := _sinksgd_oft_mode_groups(parameters):
                print(f"Note: SINKSGD_ADV normalizes the OFT updates of {', '.join(oft_groups)} with adv_optm's "
                      f"skew-symmetric Sinkhorn (its OFT mode).")
            optimizer = SinkSGD_adv(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0.0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                sinkhorn_iterations=optimizer_config.sinkhorn_iterations if optimizer_config.sinkhorn_iterations is not None else 5,
                orthogonal_sinkhorn=optimizer_config.orthogonal_sinkhorn if optimizer_config.orthogonal_sinkhorn is not None else False,
                normed_momentum=optimizer_config.normed_momentum if optimizer_config.normed_momentum is not None else False,
                snr_cond=_effective_snr_cond(optimizer_config),
                geometric_wd=optimizer_config.geometric_wd if optimizer_config.geometric_wd is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                **_adv_common_kwargs(optimizer_config),
            )

        # LION_ADV Optimizer
        case Optimizer.LION_ADV:
            from adv_optm import Lion_adv
            optimizer = Lion_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                orthogonal_gradient=_adv_orthograd_mode(optimizer_config.orthogonal_gradient),
                auto_kappa_p=optimizer_config.auto_kappa_p if optimizer_config.auto_kappa_p is not None else False,
                stochastic_sign=optimizer_config.stochastic_sign if optimizer_config.stochastic_sign is not None else False,
                centered_wd=optimizer_config.centered_wd if optimizer_config.centered_wd is not None else 0.0,
                centered_wd_mode=optimizer_config.centered_wd_mode if optimizer_config.centered_wd_mode is not None else "full",
                spectral_normalization=optimizer_config.spectral_normalization if optimizer_config.spectral_normalization is not None else False,
                **_adv_scaled_wd_kwargs(optimizer_config),
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
            )

        # MUON_ADV Optimizer
        case Optimizer.MUON_ADV:
            import inspect

            from adv_optm import Muon_adv

            params_for_optimizer, MuonWithAuxAdam = split_parameters_for_muon(parameters, layer_key_fn, config)

            # Prepare Adam-specific keyword arguments from the config
            adam_kwargs = {}
            if MuonWithAuxAdam:
                adam_config = optimizer_config.muon_adam_config
                adam_config_dict = adam_config if isinstance(adam_config, dict) else adam_config.to_dict()

                valid_adam_keys = {k for k in inspect.signature(Muon_adv.__init__).parameters if k.startswith('adam_')}
                adam_kwargs = {
                    key: adam_config_dict[key.removeprefix('adam_')]
                    for key in valid_adam_keys
                    if key.removeprefix('adam_') in adam_config_dict and adam_config_dict[key.removeprefix('adam_')] is not None
                }
                # Manually construct adam_betas from beta1 and beta2
                beta1_adam = adam_config_dict.get('beta1')
                beta2_adam = adam_config_dict.get('beta2')
                adam_kwargs['adam_betas'] = (
                    beta1_adam if beta1_adam is not None else 0.9,
                    beta2_adam if beta2_adam is not None else 0.99
                )
                adam_kwargs['adam_orthogonal_gradient'] = _adv_orthograd_mode(adam_kwargs.get('adam_orthogonal_gradient'))
                adam_kwargs['adam_state_precision'] = _adv_state_precision(
                    Optimizer.ADAMW_ADV, adam_kwargs.get('adam_state_precision'), "the auxiliary Adam's state_precision")
            optimizer = Muon_adv(
                params=params_for_optimizer,
                lr=config.learning_rate,
                beta1=optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                ns_steps=optimizer_config.ns_steps if optimizer_config.ns_steps is not None else 5,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                rms_rescaling=optimizer_config.rms_rescaling if optimizer_config.rms_rescaling is not None else True,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                normuon_variant=optimizer_config.normuon_variant if optimizer_config.normuon_variant is not None else False,
                beta2_normuon=optimizer_config.beta2_normuon if optimizer_config.beta2_normuon is not None else 0.95,
                low_rank_ortho=optimizer_config.low_rank_ortho if optimizer_config.low_rank_ortho is not None else False,
                ortho_rank=optimizer_config.ortho_rank if optimizer_config.ortho_rank is not None else 128,
                accelerated_ns=optimizer_config.accelerated_ns if optimizer_config.accelerated_ns is not None else False,
                approx_mars=optimizer_config.approx_mars if optimizer_config.approx_mars is not None else False,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                **_adv_common_kwargs(optimizer_config, nesterov_default=True),
                **adam_kwargs
            )

        # ADAMUON_ADV Optimizer
        case Optimizer.ADAMUON_ADV:
            import inspect

            from adv_optm import AdaMuon_adv

            params_for_optimizer, MuonWithAuxAdam = split_parameters_for_muon(parameters, layer_key_fn, config)

            # Prepare Adam-specific keyword arguments from the config
            adam_kwargs = {}
            if MuonWithAuxAdam:
                adam_config = optimizer_config.muon_adam_config
                # Handle both dict (from JSON/Config) and Object (legacy/runtime)
                adam_config_dict = adam_config if isinstance(adam_config, dict) else adam_config.to_dict()

                valid_adam_keys = {k for k in inspect.signature(AdaMuon_adv.__init__).parameters if k.startswith('adam_')}
                adam_kwargs = {
                    key: adam_config_dict[key.removeprefix('adam_')]
                    for key in valid_adam_keys
                    if key.removeprefix('adam_') in adam_config_dict and adam_config_dict[key.removeprefix('adam_')] is not None
                }
                # Manually construct adam_betas from beta1 and beta2
                adam_beta1 = adam_config_dict.get('beta1')
                adam_beta2 = adam_config_dict.get('beta2')
                adam_kwargs['adam_betas'] = (
                    adam_beta1 if adam_beta1 is not None else 0.9,
                    adam_beta2 if adam_beta2 is not None else 0.99
                )
                adam_kwargs['adam_orthogonal_gradient'] = _adv_orthograd_mode(adam_kwargs.get('adam_orthogonal_gradient'))
                adam_kwargs['adam_state_precision'] = _adv_state_precision(
                    Optimizer.ADAMW_ADV, adam_kwargs.get('adam_state_precision'), "the auxiliary Adam's state_precision")
            optimizer = AdaMuon_adv(
                params=params_for_optimizer,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                    optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                ns_steps=optimizer_config.ns_steps if optimizer_config.ns_steps is not None else 5,
                rms_rescaling=optimizer_config.rms_rescaling if optimizer_config.rms_rescaling is not None else True,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                use_atan2=optimizer_config.use_atan2 if optimizer_config.use_atan2 is not None else False,
                low_rank_ortho=optimizer_config.low_rank_ortho if optimizer_config.low_rank_ortho is not None else False,
                ortho_rank=optimizer_config.ortho_rank if optimizer_config.ortho_rank is not None else 128,
                normuon_variant=optimizer_config.normuon_variant if optimizer_config.normuon_variant is not None else False,
                accelerated_ns=optimizer_config.accelerated_ns if optimizer_config.accelerated_ns is not None else False,
                approx_mars=optimizer_config.approx_mars if optimizer_config.approx_mars is not None else False,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                factored_2nd=optimizer_config.factored_2nd if optimizer_config.factored_2nd is not None else False,
                **_adv_common_kwargs(optimizer_config, nesterov_default=True),
                **adam_kwargs
            )

        # MUON Optimizer
        case Optimizer.MUON:

            from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

            params_for_optimizer, ___ = split_parameters_for_muon(parameters, layer_key_fn, config)

            final_param_groups  = []
            for group in params_for_optimizer:
                is_muon = group.get('optim_type') == 'muon'

                if is_muon:
                    final_group = {
                        'params': group['params'],
                        'lr': group['lr'],
                        'use_muon': True,
                        'momentum': optimizer_config.momentum if optimizer_config.momentum is not None else 0.95,
                        'weight_decay': optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                    }
                else:  # is adam
                    adam_config = optimizer_config.muon_adam_config
                    if not isinstance(adam_config, dict):
                        adam_config = adam_config.to_dict()

                    beta1 = adam_config.get('beta1')
                    beta2 = adam_config.get('beta2')
                    eps = adam_config.get('eps')
                    weight_decay = adam_config.get('weight_decay')

                    final_group = {
                        'params': group['params'],
                        'lr': group['lr'],
                        'use_muon': False,
                        'betas': (beta1 if beta1 is not None else 0.9,
                                  beta2 if beta2 is not None else 0.95),
                        'eps': eps if eps is not None else 1e-10,
                        'weight_decay': weight_decay if weight_decay is not None else 0.0,
                    }
                final_param_groups.append(final_group)

            OptimizerClass = MuonWithAuxAdam if multi.world_size() > 1 else SingleDeviceMuonWithAuxAdam
            optimizer = OptimizerClass(param_groups=final_param_groups )

            # Add metadata back to the optimizer's param_groups for the framework to use.
            for i, group in enumerate(optimizer.param_groups):
                original_group = params_for_optimizer[i]
                group['initial_lr'] = original_group.get('initial_lr', original_group['lr'])
                group['name'] = original_group.get('name')
                group['optim_type'] = original_group.get('optim_type')


        # ADABELIEF Optimizer
        case Optimizer.ADABELIEF:
            from timm.optim.adabelief import AdaBelief
            optimizer = AdaBelief(
                params=parameters,
                lr=config.learning_rate if config.learning_rate is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-16,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                decoupled_decay=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else True,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
                rectify=optimizer_config.rectify if optimizer_config.rectify is not None else True,
                degenerated_to_sgd=optimizer_config.degenerated_to_sgd if optimizer_config.degenerated_to_sgd is not None else True,
            )

        # TIGER Optimizer
        case Optimizer.TIGER:
            from pytorch_optimizer.optimizer.tiger import Tiger
            optimizer = Tiger(
                params=parameters,
                lr=config.learning_rate if config.learning_rate is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                beta=optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                weight_decouple=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else True,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
            )

        # AIDA Optimizer
        case Optimizer.AIDA:
            from pytorch_optimizer.optimizer.aida import Aida
            optimizer = Aida(
                params=parameters,
                lr=config.learning_rate if config.learning_rate is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decouple=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else True,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
                k=optimizer_config.k if optimizer_config.k is not None else 2,
                xi=optimizer_config.xi if optimizer_config.xi is not None else 1e-20,
                rectify=optimizer_config.rectify if optimizer_config.rectify is not None else False,
                n_sma_threshold=optimizer_config.n_sma_threshold if optimizer_config.n_sma_threshold is not None else 5,
                degenerated_to_sgd=optimizer_config.degenerated_to_sgd if optimizer_config.degenerated_to_sgd is not None else True,
                ams_bound=optimizer_config.ams_bound if optimizer_config.ams_bound is not None else False,
                r=optimizer_config.r if optimizer_config.r is not None else 0.95,
                adanorm=optimizer_config.adanorm if optimizer_config.adanorm is not None else False,
                adam_debias=optimizer_config.adam_debias if optimizer_config.adam_debias is not None else False,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
            )

        # ADOPT Optimizer
        case Optimizer.ADOPT:
            from pytorch_optimizer.optimizer.adopt import ADOPT
            optimizer = ADOPT(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.9999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                weight_decouple=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else False,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
                cautious=optimizer_config.cautious if optimizer_config.cautious is not None else False,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-6,
            )

        # YOGI Optimizer
        case Optimizer.YOGI:
            from pytorch_optimizer.optimizer.yogi import Yogi
            optimizer = Yogi(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                weight_decouple=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else True,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
                r=optimizer_config.r if optimizer_config.r is not None else 0.95,
                adanorm=optimizer_config.adanorm if optimizer_config.adanorm is not None else False,
                adam_debias=optimizer_config.adam_debias if optimizer_config.adam_debias is not None else False,
                initial_accumulator=optimizer_config.initial_accumulator if optimizer_config.initial_accumulator is not None else 1e-6,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-3,
            )

    if state_dict is not None and optimizer is not None and optimizer_config.optimizer.is_adv_optm:
        saved_group = (state_dict.get("param_groups") or [None])[0]
        same_optimizer = (state_dict.get("param_group_optimizer_mapping") or [None])[0] == str(optimizer_config.optimizer)
        state_dict = _check_adv_optm_state_version(state_dict)
        if state_dict is None and optimizer_config.optimizer.is_adaptive and same_optimizer and saved_group \
                and "d" in saved_group:
            # the per-parameter statistics can't be loaded, but Prodigy's step size D is one number per group: keep the
            # D the run reached (and its step counter), or a resume in the middle of a run starts again at Initial D
            for group in optimizer.param_groups:
                for key in ("d", "d_max", "k"):
                    if key in saved_group:
                        group[key] = saved_group[key]
            print(f"Warning: {optimizer_config.optimizer} continues with the backup's step size D "
                  f"({saved_group['d']:.3g}); its other statistics start again.")
        elif state_dict is None and (note := _d_restart_note(optimizer_config, saved_group, config.learning_rate)):
            print("Warning:" + note)

    if state_dict is not None and optimizer is not None:
        if 'param_group_mapping' not in state_dict:
            # Old method of loading the optimizer state. This only works if the param groups did not change.
            for i, params in enumerate(parameters):
                state_dict['param_groups'][i]['lr'] = params['lr']
                state_dict['param_groups'][i]['initial_lr'] = params['initial_lr']
        else:
            # New method of loading the optimizer state. Each group is mapped by a unique name.
            old_state = state_dict['state']
            old_param_groups = state_dict['param_groups']
            old_group_mapping = state_dict['param_group_mapping']
            old_group_optimizer_mapping = state_dict['param_group_optimizer_mapping']

            new_param_groups = optimizer.state_dict()['param_groups']
            # the same naming as init_model_parameters uses to save the mapping: every group of the Muon optimizers
            # has an optim_type, also without the auxiliary Adam
            if any('optim_type' in group for group in optimizer.param_groups):
                new_group_mapping = []
                for group in optimizer.param_groups:
                    original_name = group.get('name')

                    optim_type = group.get('optim_type', 'unknown')
                    unique_name = f"{original_name}_{optim_type}"
                    new_group_mapping.append(unique_name)
            else:
                new_group_mapping = parameter_group_collection.unique_name_mapping

            state = {}
            param_groups = []
            state_index = 0
            changed_settings = []
            # groups that start with a fresh state although training continues from the backup, and why
            fresh_groups, other_optimizer_group = [], None

            for new_group_index, unique_group_name in enumerate(new_group_mapping):
                if (unique_group_name in old_group_mapping and str(config.optimizer.optimizer) ==
                        old_group_optimizer_mapping[old_group_mapping.index(unique_group_name)]):
                    # the group state was saved in state_dict
                    old_group_index = old_group_mapping.index(unique_group_name)
                    new_group = new_param_groups[new_group_index]
                    old_group = old_param_groups[old_group_index]
                    changed_settings += _changed_optimizer_settings(unique_group_name, old_group, new_group)
                    for i, old_state_index in enumerate(old_group['params']):
                        if old_state_index in old_state:
                            state[state_index] = old_state[old_state_index]
                        old_group['params'][i] = state_index
                        state_index += 1
                    param_groups.append(old_group)

                    # settings a newer optimizer version added since the state was saved
                    for key, value in new_group.items():
                        old_group.setdefault(key, value)
                    old_group['lr'] = new_group['lr']
                    old_group['initial_lr'] = new_group['initial_lr']
                else:
                    # the group state was not saved, initialize with an empty group state
                    if unique_group_name in old_group_mapping:
                        old_group_index = old_group_mapping.index(unique_group_name)
                        fresh_groups.append(f"{unique_group_name} was trained with "
                                            f"{old_group_optimizer_mapping[old_group_index]}")
                        other_optimizer_group = other_optimizer_group or old_param_groups[old_group_index]
                    else:
                        fresh_groups.append(f"{unique_group_name} is not in the backup")
                    new_group = new_param_groups[new_group_index]
                    new_group['params'][:] = range(state_index, state_index + len(new_group['params']))
                    state_index += len(new_group['params'])
                    param_groups.append(new_group)

            state_dict['state'] = state
            state_dict['param_groups'] = param_groups

            if fresh_groups:
                # silent before: a run continued from another run's backup trained on with a fresh optimizer state
                print(f"Warning: the backup's optimizer state is not used for {'; '.join(fresh_groups)}. Training "
                      f"continues on the backup's weights with a fresh {config.optimizer.optimizer} state."
                      + (_d_restart_note(optimizer_config, other_optimizer_group, config.learning_rate)
                         + " Or continue with the optimizer the backup was made with."
                         if other_optimizer_group is not None and optimizer_config.optimizer.is_adaptive else ""))

            if changed_settings:
                print("Warning: these optimizer settings differ between the backup and the config. The training "
                      "continues with the backup's values (only the learning rate is taken from the config); start "
                      "without the backup to use the new ones:\n  " + "\n  ".join(changed_settings))

        if optimizer_config.optimizer.is_adv_optm:
            # adv_optm 2.5 reads this group key when loading, but only sets it for groups with non-factored states
            # (never for Lion_adv or state_precision="factored"), so those states could not be resumed
            for group in state_dict['param_groups']:
                if group.get('optim_type') == 'adam':
                    precision = group.get('adam_actual_state_precision', group.get('adam_state_precision', 'auto'))
                else:
                    precision = group.get('state_precision', 'auto')
                group.setdefault('actual_state_precision', 'auto' if precision == 'factored' else precision)

        optimizer.load_state_dict(state_dict)

        if optimizer_config.optimizer.is_adv_optm:
            _restore_adv_optm_state_dtypes(optimizer, state_dict)
            _restore_kourkoutas_helper(optimizer)
            _drop_mismatched_kourkoutas_state(optimizer)
            if optimizer_config.optimizer == Optimizer.PRODIGY_ADV:
                _disable_prodigy_adv_nesterov(optimizer)

        if optimizer_config.optimizer.is_adaptive:
            stuck = [group for group in d_at_initial_groups(optimizer, min_steps=20) if group[3] <= 0]
            if stuck:
                print(f"Warning: the backup's {optimizer_config.optimizer} state still has its step size D at Initial "
                      f"D, and its estimate is negative ({_describe_d_at_initial(stuck)}), so D will not grow from "
                      f"here: the training continues at learning rate x Initial D. For D to adapt again, start without "
                      f"this backup (Continue From Last Backup off, or a new workspace), or set Initial D to the step "
                      f"size you want.")

    return optimizer


def d_at_initial_groups(optimizer: torch.optim.Optimizer, min_steps: int) -> list[tuple[str, float, int, float]]:
    # Prodigy only ever raises D: to its estimate d_numerator / sum|s| (times d_coef) when that exceeds D. With D still
    # at Initial D after many steps, the estimate never reached it, so the run trains at a fixed lr x Initial D, which
    # is larger than the step size Prodigy estimates. d_numerator sums dot(grad, start - weights); once it is negative
    # (the steps already went too far, the gradients point back towards the start) D can't grow at all. The user's
    # Chroma LoRA trained like that with LoRA alpha 32 and Initial D 1e-4 (d_numerator negative, D 1e-4 for 450 steps;
    # alpha 1 and Initial D 1e-6 adapted from step ~30), and a backup resumed such a state run after run.
    # The (name, Initial D, steps, d_numerator) of each param group still at Initial D after min_steps steps.
    # D-Adaptation keeps no d0/d_numerator in its groups and is never listed.
    groups = []
    for group in optimizer.param_groups:
        d, d0, k, numerator = group.get("d"), group.get("d0"), group.get("k", 0), group.get("d_numerator")
        if d is None or d0 is None or numerator is None or d != d0 or k < min_steps:
            continue
        groups.append((group.get("name", "?"), d0, k, numerator))
    return groups


def _describe_d_at_initial(groups: list[tuple[str, float, int, float]]) -> str:
    return "; ".join(f"{name}: D {d0:g} after {k} steps, d_numerator {numerator:.3g}" for name, d0, k, numerator in groups)


def d_at_initial_training_warning(config: TrainConfig, optimizer: torch.optim.Optimizer, steps: int) -> str | None:
    # asked once, `steps` optimizer steps into a run. A run that adapts raises D within a few dozen steps (the user's
    # Chroma LoRA from ~30); early on the estimate can also be slightly negative just from noise (3 random steps:
    # -2.5e-12), so the run is given time first
    groups = d_at_initial_groups(optimizer, min_steps=steps)
    if not groups:
        return None
    negative = " A negative d_numerator means D can't grow at all any more." \
        if any(numerator <= 0 for _, _, _, numerator in groups) else ""
    lora = " or the LoRA alpha (a LoRA step's effect scales with alpha / rank)" \
        if config.training_method == TrainingMethod.LORA else ""
    return (f"Warning: {config.optimizer.optimizer}'s step size D is still at Initial D after {steps} steps of this "
            f"run ({_describe_d_at_initial(groups)}). Prodigy only raises D, so its estimate stayed below Initial D: "
            f"the run trains at a fixed learning rate x Initial D, a larger step than Prodigy estimates.{negative} If "
            f"that isn't intended, lower Initial D (for example to 1e-6){lora}.")


def create_ema(
        parameters: Iterable[Parameter] | list[dict],
        state_dict: dict | None,
        config: TrainConfig,
) -> EMAModuleWrapper | None:
    if config.ema == EMAMode.GPU:
        device = torch.device(config.train_device)
    elif config.ema == EMAMode.CPU:
        device = torch.device("cpu")
    else:
        return None

    ema = EMAModuleWrapper(
        parameters=parameters,
        decay=config.ema_decay,
        update_step_interval=config.ema_update_step_interval,
        device=device,
        ema_stochastic_rounding=config.ema_stochastic_rounding,
    )

    if state_dict is not None:
        ema.load_state_dict(state_dict)

    return ema


def create_lr_scheduler(
        config: TrainConfig,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: LearningRateScheduler,
        warmup_steps: int | float,
        num_cycles: float,
        min_factor: float,
        num_epochs: int,
        batch_size: int,
        approximate_epoch_length: int,
        gradient_accumulation_steps: int,
        global_step: int = 0,
) -> LRScheduler:
    steps_per_epoch = approximate_epoch_length
    total_steps = int(steps_per_epoch * num_epochs / gradient_accumulation_steps)

    if warmup_steps > 1:   #values > 1 are literal step count
        warmup_steps = int(warmup_steps / gradient_accumulation_steps)
    elif 0 < warmup_steps <= 1:  #values between 0-1 are treated as percentage
        warmup_steps = int(warmup_steps * total_steps)
    else:   #catch any invalid inputs or negative values
        warmup_steps = 0

    scheduler_steps = total_steps - warmup_steps

    # Force schedule-free algorithms to constant schedule.
    if config.optimizer.optimizer.is_schedule_free:
        learning_rate_scheduler = LearningRateScheduler.CONSTANT

    match learning_rate_scheduler:
        case LearningRateScheduler.CONSTANT:
            lr_lambda = lr_lambda_constant()

        case LearningRateScheduler.LINEAR:
            lr_lambda = lr_lambda_linear(
                scheduler_steps, min_factor
            )

        case LearningRateScheduler.COSINE:
            lr_lambda = lr_lambda_cosine(
                scheduler_steps, min_factor
            )

        case LearningRateScheduler.COSINE_WITH_RESTARTS:
            lr_lambda = lr_lambda_cosine_with_restarts(
                scheduler_steps, num_cycles, min_factor
            )

        case LearningRateScheduler.COSINE_WITH_HARD_RESTARTS:
            lr_lambda = lr_lambda_cosine_with_hard_restarts(
                scheduler_steps, num_cycles, min_factor
            )

        case LearningRateScheduler.REX:
            lr_lambda = lr_lambda_rex(
                scheduler_steps, min_factor
            )

        case LearningRateScheduler.ADAFACTOR:
            from transformers.optimization import AdafactorSchedule
            return AdafactorSchedule(
                optimizer,
                initial_lr=optimizer.state_dict()['param_groups'][0]['initial_lr'],
            )
        case LearningRateScheduler.CUSTOM:
            # Special case. Unlike the others, we return from here.
            if not config.custom_learning_rate_scheduler:
                raise AssertionError("Must specify a class when using a custom LR scheduler.")
            if "." not in config.custom_learning_rate_scheduler:
                raise AssertionError("Custom class name must be in the format <module>.<class>")
            klass = config.custom_learning_rate_scheduler.split(".")[-1]
            module = config.custom_learning_rate_scheduler.removesuffix("." + klass)
            module = importlib.import_module(module)
            klass = getattr(module, klass)
            # Compile arguments into single dict.
            args = {}
            for pd in config.scheduler_params:
                key = pd["key"]
                value = pd["value"]
                # Special values
                match value:
                    case "%LR%":
                        value = config.learning_rate
                    case "%EPOCHS%":
                        value = num_epochs
                    case "%STEPS_PER_EPOCH%":
                        value = steps_per_epoch
                    case "%TOTAL_STEPS%":
                        value = total_steps
                    case "%SCHEDULER_STEPS%":
                        value = scheduler_steps
                    case _:
                        value = ast.literal_eval(value)
                args[key] = value
            scheduler = klass(optimizer=optimizer,
                              last_epoch=int(global_step / gradient_accumulation_steps) - 1,
                              **args)
            if warmup_steps > 0:
                warmup_scheduler = LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lr_lambda_warmup(warmup_steps, lr_lambda_constant()),
                    last_epoch=int(global_step / gradient_accumulation_steps) - 1)
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_steps],
                    last_epoch=int(global_step / gradient_accumulation_steps) - 1)
            return scheduler
        case _:
            lr_lambda = lr_lambda_constant()

    if warmup_steps > 0 and not config.optimizer.optimizer.is_schedule_free:
        lr_lambda = lr_lambda_warmup(warmup_steps, lr_lambda)

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
        last_epoch=int(global_step / gradient_accumulation_steps) - 1,
    )


def create_noise_scheduler(
        noise_scheduler: NoiseScheduler,
        original_noise_scheduler: SchedulerMixin = None,
        num_inference_timesteps: int = None,
):
    scheduler = None

    num_inference_timesteps = num_inference_timesteps or 20
    num_train_timesteps = original_noise_scheduler.config.num_train_timesteps if hasattr(
        original_noise_scheduler.config, "num_train_timesteps") else 1000
    beta_start = original_noise_scheduler.config.beta_start if hasattr(original_noise_scheduler.config,
                                                                       "beta_start") else 0.00085
    beta_end = original_noise_scheduler.config.beta_end if hasattr(original_noise_scheduler.config,
                                                                   "beta_end") else 0.012
    beta_schedule = original_noise_scheduler.config.beta_schedule if hasattr(original_noise_scheduler.config,
                                                                             "beta_schedule") else "scaled_linear"
    prediction_type = original_noise_scheduler.config.prediction_type if hasattr(original_noise_scheduler.config,
                                                                                 "prediction_type") else "epsilon"

    match noise_scheduler:
        case NoiseScheduler.DDIM:
            scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type=prediction_type,
            )
        case NoiseScheduler.EULER:
            scheduler = EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=False,
            )
        case NoiseScheduler.EULER_A:
            scheduler = EulerAncestralDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
            )
        case NoiseScheduler.DPMPP:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=0,
                prediction_type=prediction_type,
                use_karras_sigmas=False,
                algorithm_type="dpmsolver++"
            )
        case NoiseScheduler.DPMPP_SDE:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=0,
                prediction_type=prediction_type,
                use_karras_sigmas=False,
                algorithm_type="sde-dpmsolver++"
            )
        case NoiseScheduler.UNIPC:
            scheduler = UniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=False,
            )
        case NoiseScheduler.EULER_KARRAS:
            scheduler = EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=True,
            )
        case NoiseScheduler.DPMPP_KARRAS:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
        case NoiseScheduler.DPMPP_SDE_KARRAS:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++"
            )
        case NoiseScheduler.UNIPC_KARRAS:
            scheduler = UniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=True,
            )

    if scheduler:
        scheduler.set_timesteps(num_inference_timesteps)

    return scheduler

def create_trainer(
        config: TrainConfig,
        callbacks: TrainCallbacks,
        commands: TrainCommands,
        reattach: bool = False,
):
    if config.cloud.enabled:
        from modules.trainer.CloudTrainer import CloudTrainer
        trainer = CloudTrainer(config, callbacks, commands, reattach=reattach)
    elif config.multi_gpu:
        from modules.trainer.MultiTrainer import MultiTrainer
        trainer = MultiTrainer(config, callbacks, commands)
    else:
        ZLUDA.initialize_devices(config)
        from modules.trainer.GenericTrainer import GenericTrainer
        trainer = GenericTrainer(config, callbacks, commands)
    return trainer
