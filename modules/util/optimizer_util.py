import copy

import modules.util.multi_gpu_util as multi
from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig, TrainOptimizerConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer.muon_util import build_muon_adam_key_fn
from modules.util.torch_util import optimizer_to_device_

import torch


def default_optimizer_config(optimizer: Optimizer) -> TrainOptimizerConfig:
    # the settings an optimizer starts from when it is selected in the UI
    optimizer_config = TrainOptimizerConfig.default_values()
    optimizer_config.from_dict(OPTIMIZER_DEFAULT_PARAMETERS[optimizer])
    optimizer_config.optimizer = optimizer
    return optimizer_config


# settings that create_optimizer passes on as None, where None is a choice rather than "use the default"
# (e.g. eps=None enables Adam-atan2 in prodigy-plus-schedule-free)
_NONE_IS_A_SETTING = {
    Optimizer.PRODIGY: {"beta3"},
    Optimizer.PRODIGY_PLUS_SCHEDULE_FREE: {"beta3", "eps"},
    Optimizer.PRODIGY_ADV: {"beta3"},
    Optimizer.ADAFACTOR: {"beta1"},
}


def adv_state_precisions(optimizer: Optimizer) -> list[str]:
    # the adv_optm state_precision modes that train with this optimizer. In adv_optm 2.5.13-2.6.0 "fp16" fails in
    # every optimizer except Adopt_adv (a dtype error, or NaN in Prodigy_adv). "factored" failed in AdaMuon_adv up to
    # 2.5.13 (original_shape unbound) and works since 2.5.16
    precisions = ["auto", "fp32", "factored", "bf16_sr", "int8_sr"]
    if optimizer == Optimizer.ADOPT_ADV:
        precisions.append("fp16")
    return precisions


def fill_unset_optimizer_settings(optimizer_config: TrainOptimizerConfig) -> TrainOptimizerConfig:
    # a copy where every unset (None) setting of the optimizer's defaults is replaced by the default the UI starts from
    optimizer = optimizer_config.optimizer
    filled_config = copy.deepcopy(optimizer_config)
    for key, value in OPTIMIZER_DEFAULT_PARAMETERS[optimizer].items():
        if getattr(filled_config, key, None) is None and key not in _NONE_IS_A_SETTING.get(optimizer, set()):
            setattr(filled_config, key, value)
    return filled_config


def change_optimizer(train_config: TrainConfig) -> TrainOptimizerConfig:
    optimizer = train_config.optimizer.optimizer

    optimizer_config = default_optimizer_config(optimizer)

    if str(optimizer) in train_config.optimizer_defaults:
        saved_optimizer_config = train_config.optimizer_defaults[str(optimizer)]
        optimizer_config.from_dict(saved_optimizer_config.to_dict())

    return optimizer_config


def load_optimizer_defaults(train_config: TrainConfig) -> TrainOptimizerConfig:
    optimizer = train_config.optimizer.optimizer

    optimizer_config = default_optimizer_config(optimizer)

    if str(optimizer) in train_config.optimizer_defaults:
        train_config.optimizer_defaults.pop(str(optimizer))

    return optimizer_config


def update_optimizer_config(train_config: TrainConfig):
    optimizer = train_config.optimizer.optimizer

    if str(optimizer) in train_config.optimizer_defaults:
        saved_optimizer_config = train_config.optimizer_defaults[str(optimizer)]
        saved_optimizer_config.from_dict(train_config.optimizer.to_dict())
    else:
        optimizer_donfig = TrainOptimizerConfig.default_values()
        optimizer_donfig.from_dict(train_config.optimizer.to_dict())
        train_config.optimizer_defaults[str(optimizer)] = optimizer_donfig


def init_model_parameters(
        model: BaseModel,
        parameters: NamedParameterGroupCollection,
        train_device: torch.device,
):
    model.parameters = parameters
    #random (LoRA) initialisation can differ, broadcast from GPU #0 to all others
    #to be safe, do that before the optimizer is created because the optimizer could take copies
    multi.broadcast_parameters(parameters.parameters(), train_device)

    layer_key_fn = None
    if model.train_config.optimizer.MuonWithAuxAdam:
        print("INFO: Creating layer keys for MuonWithAuxAdam.")
        layer_key_fn = build_muon_adam_key_fn(model, model.train_config)

    from modules.util import create

    if model.train_config.optimizer.optimizer.is_adv_optm:
        from modules.util.optimizer.tag_util import tag_peft_parameters
        tag_peft_parameters(model, model.train_config.oft_scaled)

    model.optimizer = create.create_optimizer(
        parameters, model.optimizer_state_dict, model.train_config, layer_key_fn
    )

    if model.optimizer is not None:
        optimizer_to_device_(model.optimizer, train_device)
    model.optimizer_state_dict = None

    if multi.is_master():
        model.ema = create.create_ema(parameters.parameters(), model.ema_state_dict, model.train_config)
    else:
        model.ema = None
    model.ema_state_dict = None

    if model.optimizer is not None and any('optim_type' in g for g in model.optimizer.param_groups):
        new_param_group_mapping = []
        for group in model.optimizer.param_groups:
            original_name = group.get('name')

            optim_type = group.get('optim_type', 'unknown')
            unique_name = f"{original_name}_{optim_type}"
            new_param_group_mapping.append(unique_name)
        model.param_group_mapping = new_param_group_mapping
    else:
        model.param_group_mapping = parameters.unique_name_mapping


# Optimizer Key map with defaults
OPTIMIZER_DEFAULT_PARAMETERS = {
    Optimizer.ADAFACTOR: {
        "eps": 1e-30,
        "eps2": 1e-3,
        "clip_threshold": 1.0,
        "decay_rate": -0.8,
        "beta1": None,
        "weight_decay": 0.0,
        "scale_parameter": False,
        "relative_step": False,
        "warmup_init": False,
        "stochastic_rounding": True,
        "fused_back_pass": False,
    },
    Optimizer.ADAGRAD: {
        "lr_decay": 0,
        "weight_decay": 0,
        "initial_accumulator_value": 0,
        "eps": 1e-10,
    },
    Optimizer.ADAGRAD_8BIT: {
        "lr_decay": 0,
        "weight_decay": 0,
        "initial_accumulator_value": 0,
        "eps": 1e-10,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "fused_back_pass": False,
    },
    Optimizer.ADAM_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.ADAMW_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.MUON: {
        "momentum": 0.95,
        "weight_decay": 0.0,
        "MuonWithAuxAdam": True,
        "muon_hidden_layers": None,
        "muon_adam_regex": False,
        "muon_adam_lr": 3e-4,
        "muon_te1_adam_lr": None,
        "muon_te2_adam_lr": None,
        "muon_adam_config": {},
    },
    Optimizer.AdEMAMix_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "eps": 1e-8,
        "alpha": 5,
        "weight_decay": 1e-2,
        "min_8bit_size": 4096,
        "is_paged": False,
    },
    Optimizer.AdEMAMix: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "eps": 1e-8,
        "alpha": 5,
        "weight_decay": 1e-2,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "is_paged": False,
    },
    Optimizer.ADOPT: {
        "beta1": 0.9,
        "beta2": 0.9999,
        "weight_decay": 0.0,
        "decoupled_decay": False,
        "fixed_decay": False,
        "cautious": False,
        "eps": 1e-6,
    },
    Optimizer.LAMB: {
        "bias_correction": True,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "adam_w_mode": True,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": False,
        "max_unorm": 1.0,
    },
    Optimizer.LAMB_8BIT: {
        "bias_correction": True,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "adam_w_mode": True,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": False,
        "max_unorm": 1.0,
    },
    Optimizer.LARS: {
        "momentum": 0.9,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "max_unorm": 0.02,
    },
    Optimizer.LARS_8BIT: {
        "momentum": 0.9,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "max_unorm": 0.02,
    },
    Optimizer.LION_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.RMSPROP: {
        "alpha": 0.99,
        "eps": 1e-8,
        "weight_decay": 0,
        "momentum": 0,
        "centered": False,
    },
    Optimizer.RMSPROP_8BIT: {
        "alpha": 0.99,
        "eps": 1e-8,
        "weight_decay": 0,
        "momentum": 0,
        "centered": False,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.SGD_8BIT: {
        "momentum": 0.9,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
    },
    Optimizer.SCHEDULE_FREE_ADAMW: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "r": 0.0,
        "weight_lr_power": 2.0,
        "foreach": False,
    },
    Optimizer.SCHEDULE_FREE_SGD: {
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "r": 0.0,
        "weight_lr_power": 2.0,
        "foreach": False,
    },
    Optimizer.PRODIGY: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": None,
        "eps": 1e-8,
        "weight_decay": 0,
        "decouple": True,
        "use_bias_correction": False,
        "safeguard_warmup": False,
        "d0": 1e-6,
        "d_coef": 1.0,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
        "slice_p": 11,
    },
    Optimizer.PRODIGY_PLUS_SCHEDULE_FREE: {
        "beta1": 0.9,
        "beta2": 0.99,
        "beta3": None,
        "weight_decay": 0.0,
        "weight_decay_by_lr": True,
        "use_bias_correction": False,
        "d0": 1e-6,
        "d_coef": 1.0,
        "prodigy_steps": 0,
        "use_speed": False,
        "eps": 1e-8,
        "split_groups": True,
        "split_groups_mean": False,
        "factored": True,
        "factored_fp32": True,
        "fused_back_pass": False,
        "use_stableadamw": True,
        "use_cautious": False,
        "use_grams": False,
        "use_adopt": False,
        "d_limiter": True,
        "stochastic_rounding": True,
        "use_schedulefree": True,
        "schedulefree_c": 0.0,
        "use_orthograd": False,
    },
    Optimizer.DADAPT_ADA_GRAD: {
        "momentum": 0,
        "log_every": 0,
        "weight_decay": 0.0,
        "eps": 1e-6,
        "d0": 1e-6,
        "growth_rate": float('inf'),
    },
    Optimizer.DADAPT_ADAN: {
        "beta1": 0.98,
        "beta2": 0.92,
        "beta3": 0.99,
        "eps": 1e-8,
        "weight_decay": 0.02,
        "no_prox": False,
        "log_every": 0,
        "d0": 1e-6,
        "growth_rate": float('inf'),
    },
    Optimizer.DADAPT_ADAM: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "log_every": 0,
        "decouple": False,
        "use_bias_correction": False,
        "d0": 1e-6,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
    },
    Optimizer.DADAPT_SGD: {
        "momentum": 0.0,
        "weight_decay": 0,
        "log_every": 0,
        "d0": 1e-6,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
    },
    Optimizer.DADAPT_LION: {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "log_every": 0,
        "d0": 1e-6,
        "fsdp_in_use": False,
    },
    Optimizer.ADAM: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "foreach": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": True,
        "stochastic_rounding": False,
        "fused_back_pass": False,
    },
    Optimizer.ADAMW: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "amsgrad": False,
        "foreach": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": True,
        "stochastic_rounding": False,
        "fused_back_pass": False,
    },
    Optimizer.SGD: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "foreach": False,
        "maximize": False,
        "differentiable": False,
    },
    Optimizer.LION: {
        "beta1": 0.9,
        "beta2": 0.99,
        "weight_decay": 0.0,
        "use_triton": False,
    },
    Optimizer.CAME: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "eps": 1e-30,
        "eps2": 1e-16,
        "weight_decay": 1e-2,
        "stochastic_rounding": False,
        "use_cautious": False,
        "fused_back_pass": False,
    },
    Optimizer.CAME_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "eps": 1e-30,
        "eps2": 1e-16,
        "weight_decay": 1e-2,
        "stochastic_rounding": False,
        "fused_back_pass": False,
        "min_8bit_size": 16384,
        "quant_block_size": 2048
    },
    Optimizer.ADAMW_ADV: {
        "beta1": 0.9,
        "beta2": 0.99,
        "eps": 1e-8,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "fisher_wd": False,
        "centered_wd": 0.0,
        "centered_wd_mode": "full",
        "state_precision": "auto",
        "factored_2nd": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "use_atan2": False,
        "orthogonal_gradient": "disabled",
        "nesterov": False,
        "nesterov_coef": None,
        "normed_momentum": False,
        "spectral_normalization": False,
        "kourkoutas_beta": False,
        "scaled_wd": False,
    },
    Optimizer.ADOPT_ADV: {
        "beta1": 0.9,
        "beta2": 0.9999,
        "eps": 1e-6,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "fisher_wd": False,
        "centered_wd": 0.0,
        "centered_wd_mode": "full",
        "state_precision": "auto",
        "factored_2nd": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "use_atan2": True,
        "orthogonal_gradient": "disabled",
        "nesterov": False,
        "nesterov_coef": None,
        "spectral_normalization": False,
        "kourkoutas_beta": False,
        "scaled_wd": False,
    },
    Optimizer.PRODIGY_ADV: {
        "beta1": 0.9,
        "beta2": 0.99,
        "beta3": None,
        "eps": 1e-8,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "fisher_wd": False,
        "centered_wd": 0.0,
        "centered_wd_mode": "full",
        "state_precision": "auto",
        "factored_2nd": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "d0": 1e-6,
        "d_coef": 1.0,
        "growth_rate": float('inf'),
        "slice_p": 11,
        "prodigy_steps": 0,
        "d_limiter": False,
        "use_atan2": False,
        "orthogonal_gradient": "disabled",
        "spectral_normalization": False,
        "kourkoutas_beta": False,
        "scaled_wd": False,
    },
    Optimizer.SIGNSGD_ADV: {
        "momentum": 0.95,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "geometric_wd": False,
        "centered_wd": 0.0,
        # the settings window fills left, right, left, ...: this order keeps the dropdowns in the left column, where
        # the entries make it wide (in the right column, next to only checkboxes, they were squeezed)
        "stochastic_rounding": True,
        "centered_wd_mode": "full",
        "compile": False,
        "state_precision": "auto",
        "fused_back_pass": False,
        "orthogonal_gradient": "disabled",
        "nesterov": False,
        "nesterov_coef": None,
        "normed_momentum": False,
        "snr_cond": False,
        "stochastic_sign": False,
        "spectral_normalization": False,
        "scaled_wd": False,
    },
    Optimizer.SINKSGD_ADV: {
        "momentum": 0.95,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "geometric_wd": False,
        "centered_wd": 0.0,
        # dropdowns in the left column, see SIGNSGD_ADV
        "stochastic_rounding": True,
        "centered_wd_mode": "full",
        "compile": False,
        "state_precision": "auto",
        "fused_back_pass": False,
        "sinkhorn_iterations": 5,
        "orthogonal_sinkhorn": False,
        "orthogonal_gradient": "disabled",
        "nesterov": True,
        "nesterov_coef": None,
        "normed_momentum": False,
        "snr_cond": False,
        "spectral_normalization": False,
        "scaled_wd": False,
    },
    Optimizer.LION_ADV: {
        "beta1": 0.9,
        "beta2": 0.99,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "centered_wd": 0.0,
        "centered_wd_mode": "full",
        "nnmf_factor": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "orthogonal_gradient": "disabled",
        "auto_kappa_p": True,
        "stochastic_sign": False,
        "spectral_normalization": False,
        "scaled_wd": False,
    },
    Optimizer.MUON_ADV: {
        "beta1": 0.9,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "centered_wd": 0.0,
        "centered_wd_mode": "full",
        "state_precision": "auto",
        "accelerated_ns": False,
        "ns_steps": 5,
        "low_rank_ortho": False,
        "ortho_rank": 128,
        "rms_rescaling": True,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "MuonWithAuxAdam": True,
        "muon_hidden_layers": None,
        "muon_adam_regex": False,
        "muon_adam_lr": 1e-6,
        "muon_te1_adam_lr": None,
        "muon_te2_adam_lr": None,
        "nesterov": True,
        "nesterov_coef": None,
        "normuon_variant": True,
        "beta2_normuon": 0.95,
        "orthogonal_gradient": "disabled",
        "approx_mars": False,
        "spectral_normalization": False,
        "muon_adam_config": {},
        "scaled_wd": False,
    },
    Optimizer.ADAMUON_ADV: {
        "beta1": 0.95,
        "beta2": 0.95,
        "eps": 1e-8,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "centered_wd": 0.0,
        "centered_wd_mode": "full",
        "state_precision": "auto",
        "factored_2nd": False,
        "accelerated_ns": False,
        "ns_steps": 5,
        "low_rank_ortho": False,
        "ortho_rank": 128,
        "rms_rescaling": True,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "MuonWithAuxAdam": True,
        "muon_hidden_layers": None,
        "muon_adam_regex": False,
        "muon_adam_lr": 1e-6,
        "muon_te1_adam_lr": None,
        "muon_te2_adam_lr": None,
        "nesterov": True,
        "nesterov_coef": None,
        "use_atan2": False,
        "normuon_variant": True,
        "orthogonal_gradient": "disabled",
        "approx_mars": False,
        "spectral_normalization": False,
        "muon_adam_config": {},
        "scaled_wd": False,
    },
    Optimizer.ADABELIEF: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-16,
        "weight_decay": 0,
        "amsgrad": False,
        "decoupled_decay": True,
        "fixed_decay": False,
        "rectify": True,
        "degenerated_to_sgd": True,
    },
    Optimizer.TIGER: {
        "beta1": 0.965,
        "weight_decay": 0.01,
        "decoupled_decay": True,
        "fixed_decay": False,
    },
    Optimizer.AIDA: {
        "beta1": 0.9,
        "beta2": 0.999,
        "k": 2,
        "xi": 1e-20,
        "weight_decay": 0.0,
        "decoupled_decay": False,
        "fixed_decay": False,
        "rectify": False,
        "n_sma_threshold": 5,
        "degenerated_to_sgd": True,
        "ams_bound": False,
        "r": 0.95,
        "adanorm": False,
        "adam_debias": False,
        "eps": 1e-8,
    },
    Optimizer.YOGI: {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "decoupled_decay": True,
        "fixed_decay": False,
        "r": 0.95,
        "adanorm": False,
        "adam_debias": False,
        "initial_accumulator": 1e-6,
        "eps": 1e-3,
    },
}
