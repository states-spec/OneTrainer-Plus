
from collections.abc import Callable
from typing import Any

from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import (
    OPTIMIZER_DEFAULT_PARAMETERS,
    adv_state_precisions,
)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _on(value: Any) -> bool:
    return value.lower() == "true" if isinstance(value, str) else bool(value)


def _decays(value: Callable[[str], Any]) -> bool:
    return _number(value('weight_decay')) > 0 or _number(value('centered_wd')) > 0


def _plain_implementation(value: Callable[[str], Any]) -> bool:
    # ADAM/ADAMW: torch's fused and foreach steps can't step one parameter at a time or round stochastically, and
    # create_optimizer refuses either option together with them
    return not _on(value('fused')) and not _on(value('foreach'))


# settings that only take effect together with another one: the optimizer windows grey them out while it is off.
# Only where the setting provably has no effect then (adv_optm 2.5.13-2.6.0)
OPTION_DEPENDENCIES: dict[str, Callable[[Callable[[str], Any]], bool]] = {
    'nesterov_coef': lambda value: _on(value('nesterov')),
    'centered_wd_mode': lambda value: _number(value('centered_wd')) > 0,
    # adv_optm applies it only with normed momentum and momentum > 0
    'snr_cond': lambda value: _on(value('normed_momentum')) and _number(value('momentum')) > 0,
    'min_beta1': lambda value: _number(value('beta1_warmup')) > 0,
    # they modify the (centered) weight decay
    'cautious_wd': _decays,
    'fisher_wd': _decays,
    'geometric_wd': _decays,
    'scaled_wd': _decays,
    'fused_back_pass': _plain_implementation,
    'stochastic_rounding': _plain_implementation,
}


class OptionDependencies:
    def __init__(self, components, ui_state, shown_keys):
        self.components = components
        self.ui_state = ui_state
        self.shown_keys = set(shown_keys)
        self.widgets = []

    def add(self, key: str, widget, label=None, rule: Callable[[Callable[[str], Any]], bool] | None = None):
        # the label too: a disabled checkbox looks almost like an unchecked one in the dark theme (Qt greys the text;
        # a CTk label has no disabled color)
        rule = rule or OPTION_DEPENDENCIES.get(key)
        if rule is not None:
            self.widgets += [(w, rule) for w in (widget, label) if w is not None]

    def value(self, key: str) -> Any:
        # a setting the window doesn't show has no effect for this optimizer, even if the config holds a value
        if key not in self.shown_keys:
            return None
        try:
            return self.ui_state.get_var(key).get()
        except KeyError:
            return None

    def update(self, *args):
        for widget, rule in self.widgets:
            self.components.set_widget_enabled(widget, rule(self.value))


class BaseOptimizerParamsWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, controller, ui_state, optimizer_ui_state,
                      on_optimizer_change_cb, load_defaults_cb):
        # Optimizer
        self.components.label(frame, 0, 0, "Optimizer",
                              tooltip="The type of optimizer")

        # Create the optimizer dropdown menu and set the command
        self.components.options(frame, 0, 1, [str(x) for x in list(Optimizer)], optimizer_ui_state, "optimizer",
                                command=on_optimizer_change_cb)

        # Defaults Button
        self.components.label(frame, 0, 3, "Optimizer Defaults",
                              tooltip="Load default settings for the selected optimizer")
        self.components.button(frame, 0, 4, "Load Defaults", load_defaults_cb,
                               tooltip="Load default settings for the selected optimizer")

    def build_dynamic_content(self, master, controller, optimizer_ui_state,
                              update_user_pref_cb, open_muon_adam_cb):
        # Lookup for the title and tooltip for a key
        # @formatter:off
        KEY_DETAIL_MAP = {
            'adam_w_mode': {'title': 'Adam W Mode', 'tooltip': 'Whether to use weight decay correction for Adam optimizer.', 'type': 'bool'},
            'alpha': {'title': 'Alpha', 'tooltip': 'Smoothing parameter for RMSprop and others.', 'type': 'float'},
            'amsgrad': {'title': 'AMSGrad', 'tooltip': 'Whether to use the AMSGrad variant for Adam.', 'type': 'bool'},
            'beta1': {'title': 'Beta1', 'tooltip': 'Momentum term.', 'type': 'float'},
            'beta2': {'title': 'Beta2', 'tooltip': 'Coefficients for computing running averages of gradient.', 'type': 'float'},
            'beta3': {'title': 'Beta3', 'tooltip': 'Coefficient for computing the Prodigy stepsize.', 'type': 'float'},
            'bias_correction': {'title': 'Bias Correction', 'tooltip': 'Whether to use bias correction in optimization algorithms like Adam.', 'type': 'bool'},
            'block_wise': {'title': 'Block Wise', 'tooltip': 'Whether to perform block-wise model update.', 'type': 'bool'},
            'capturable': {'title': 'Capturable', 'tooltip': 'Whether some property of the optimizer can be captured.', 'type': 'bool'},
            'centered': {'title': 'Centered', 'tooltip': 'Whether to center the gradient before scaling. Great for stabilizing the training process.', 'type': 'bool'},
            'clip_threshold': {'title': 'Clip Threshold', 'tooltip': 'Clipping value for gradients.', 'type': 'float'},
            'd0': {'title': 'Initial D', 'tooltip': 'Initial D estimate for D-adaptation.', 'type': 'float'},
            'd_coef': {'title': 'D Coefficient', 'tooltip': 'Coefficient in the expression for the estimate of d.', 'type': 'float'},
            'dampening': {'title': 'Dampening', 'tooltip': 'Dampening for momentum.', 'type': 'float'},
            'decay_rate': {'title': 'Decay Rate', 'tooltip': 'Rate of decay for moment estimation.', 'type': 'float'},
            'decouple': {'title': 'Decouple', 'tooltip': 'Use AdamW style decoupled weight decay.', 'type': 'bool'},
            'differentiable': {'title': 'Differentiable', 'tooltip': 'Whether the optimization function is differentiable.', 'type': 'bool'},
            'eps': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'eps2': {'title': 'EPS 2', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'foreach': {'title': 'ForEach', 'tooltip': 'Whether to use a foreach implementation if available. This implementation is usually faster.', 'type': 'bool'},
            'fsdp_in_use': {'title': 'FSDP in Use', 'tooltip': 'Flag for using sharded parameters.', 'type': 'bool'},
            'fused': {'title': 'Fused', 'tooltip': 'Whether to use a fused implementation if available. This implementation is usually faster and requires less memory.', 'type': 'bool'},
            'fused_back_pass': {'title': 'Fused Back Pass', 'tooltip': 'Whether to fuse the back propagation pass with the optimizer step. This reduces VRAM usage, but is not compatible with gradient accumulation. Clip Grad Norm then clips each parameter on its own instead of all gradients together. ADAM and ADAMW need Fused and ForEach off for it.', 'type': 'bool'},
            'growth_rate': {'title': 'Growth Rate', 'tooltip': 'Limit for D estimate growth rate.', 'type': 'float'},
            'initial_accumulator_value': {'title': 'Initial Accumulator Value', 'tooltip': 'Initial value for Adagrad optimizer.', 'type': 'float'},
            'initial_accumulator': {'title': 'Initial Accumulator', 'tooltip': 'Sets the starting value for both moment estimates to ensure numerical stability and balanced adaptive updates early in training.', 'type': 'float'},
            'is_paged': {'title': 'Is Paged', 'tooltip': 'Whether the optimizer\'s internal state should be paged to CPU.', 'type': 'bool'},
            'log_every': {'title': 'Log Every', 'tooltip': 'Intervals at which logging should occur.', 'type': 'int'},
            'lr_decay': {'title': 'LR Decay', 'tooltip': 'Rate at which learning rate decreases.', 'type': 'float'},
            'max_unorm': {'title': 'Max Unorm', 'tooltip': 'Maximum value for gradient clipping by norms.', 'type': 'float'},
            'maximize': {'title': 'Maximize', 'tooltip': 'Whether to maximize the optimization function.', 'type': 'bool'},
            'min_8bit_size': {'title': 'Min 8bit Size', 'tooltip': 'Minimum tensor size for 8-bit quantization.', 'type': 'int'},
            'quant_block_size': {'title': 'Quant Block Size', 'tooltip': 'Size of a block of normalized 8-bit quantization data. Larger values increase memory efficiency at the cost of data precision.', 'type': 'int'},
            'momentum': {'title': 'Momentum', 'tooltip': 'Factor to accelerate SGD in relevant direction.', 'type': 'float'},
            'nesterov': {'title': 'Nesterov', 'tooltip': 'Whether to use Nesterov momentum. For the ADV optimizers its mixing is set by Nesterov Coefficient; this replaces the former (Simplified) AdEMAMix options.', 'type': 'bool'},
            'no_prox': {'title': 'No Prox', 'tooltip': 'Whether to use proximity updates or not.', 'type': 'bool'},
            'optim_bits': {'title': 'Optim Bits', 'tooltip': 'Number of bits used for optimization.', 'type': 'int'},
            'percentile_clipping': {'title': 'Percentile Clipping', 'tooltip': 'Gradient clipping based on percentile values.', 'type': 'int'},
            'relative_step': {'title': 'Relative Step', 'tooltip': 'Whether to use a relative step size.', 'type': 'bool'},
            'safeguard_warmup': {'title': 'Safeguard Warmup', 'tooltip': 'Avoid issues during warm-up stage.', 'type': 'bool'},
            'scale_parameter': {'title': 'Scale Parameter', 'tooltip': 'Whether to scale the parameter or not.', 'type': 'bool'},
            'stochastic_rounding': {'title': 'Stochastic Rounding', 'tooltip': 'Stochastic rounding for weight updates. Improves quality when using bfloat16 weights. ADAM and ADAMW need Fused and ForEach off for it.', 'type': 'bool'},
            'use_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Turn on Adam\'s bias correction.', 'type': 'bool'},
            'use_triton': {'title': 'Use Triton', 'tooltip': 'Whether Triton optimization should be used.', 'type': 'bool'},
            'warmup_init': {'title': 'Warmup Initialization', 'tooltip': 'Whether to warm-up the optimizer initialization.', 'type': 'bool'},
            'weight_decay': {'title': 'Weight Decay', 'tooltip': 'Regularization to prevent overfitting.', 'type': 'float'},
            'weight_lr_power': {'title': 'Weight LR Power', 'tooltip': 'During warmup, the weights in the average will be equal to lr raised to this power. Set to 0 for no weighting.', 'type': 'float'},
            'decoupled_decay': {'title': 'Decoupled Decay', 'tooltip': 'If set as True, then the optimizer uses decoupled weight decay as in AdamW.', 'type': 'bool'},
            'fixed_decay': {'title': 'Fixed Decay', 'tooltip': '(When Decoupled Decay is True:) Applies fixed weight decay when True; scales decay with learning rate when False.', 'type': 'bool'},
            'rectify': {'title': 'Rectify', 'tooltip': 'Perform the rectified update similar to RAdam.', 'type': 'bool'},
            'degenerated_to_sgd': {'title': 'Degenerated to SGD', 'tooltip': 'Performs SGD update when gradient variance is high.', 'type': 'bool'},
            'k': {'title': 'K', 'tooltip': 'Number of vector projected per iteration.', 'type': 'int'},
            'xi': {'title': 'Xi', 'tooltip': 'Term used in vector projections to avoid division by zero.', 'type': 'float'},
            'n_sma_threshold': {'title': 'N SMA Threshold', 'tooltip': 'Number of SMA threshold.', 'type': 'int'},
            'ams_bound': {'title': 'AMS Bound', 'tooltip': 'Whether to use the AMSBound variant.', 'type': 'bool'},
            'r': {'title': 'R', 'tooltip': 'EMA factor.', 'type': 'float'},
            'adanorm': {'title': 'AdaNorm', 'tooltip': 'Whether to use the AdaNorm variant', 'type': 'bool'},
            'adam_debias': {'title': 'Adam Debias', 'tooltip': 'Only correct the denominator to avoid inflating step sizes early in training.', 'type': 'bool'},
            'slice_p': {'title': 'Slice parameters', 'tooltip': 'Reduce memory usage by calculating LR adaptation statistics on only every pth entry of each tensor. For values greater than 1 this is an approximation to standard Prodigy. Values ~11 are reasonable.', 'type': 'int'},
            'cautious': {'title': 'Cautious', 'tooltip': 'Whether to use the Cautious variant', 'type': 'bool'},
            'weight_decay_by_lr': {'title': 'Weight Decay by LR', 'tooltip': "Multiply weight decay by the adaptive learning rate, as PyTorch's AdamW does. Off: weight decay is not scaled by the LR.", 'type': 'bool'},
            'prodigy_steps': {'title': 'Prodigy Steps', 'tooltip': 'Turn off Prodigy after N steps', 'type': 'int'},
            'use_speed': {'title': 'SPEED', 'tooltip': "Highly experimental. Replaces Prodigy's step size estimate with SPEED (Simplified Prodigy with rElativE D): less memory and scale-insensitive, but can be unstable with weight decay.", 'type': 'bool'},
            'split_groups': {'title': 'Split Groups', 'tooltip': 'Calculate d for each parameter group separately, e.g. when training a text encoder beside the UNet. Off: one d across all groups, as in the original Prodigy.', 'type': 'bool'},
            'split_groups_mean': {'title': 'Split Groups Mean', 'tooltip': "With Split Groups: use the harmonic mean of d across all groups times each group's LR, instead of each group's own d.", 'type': 'bool'},
            'factored': {'title': 'Factored', 'tooltip': "Factored approximation of the second moment, like Adafactor. Saves memory. Turn off if training gives NaNs or the learning rate doesn't grow.", 'type': 'bool'},
            'factored_fp32': {'title': 'Factored FP32', 'tooltip': "Keep the factored second moment in float32 for stability. Off: use the gradient's dtype (slightly less memory). Ignored without Factored.", 'type': 'bool'},
            'use_stableadamw': {'title': 'StableAdamW', 'tooltip': "Scale updates by their RMS, like Adafactor's update scaling. Turn off if the adaptive learning rate never improves or is over-estimated.", 'type': 'bool'},
            'use_cautious': {'title': 'Cautious', 'tooltip': 'Experimental. Cautious updates (arXiv 2411.16085): keep and boost only the update components that agree with the current gradient.', 'type': 'bool'},
            'use_grams': {'title': 'Grams', 'tooltip': "Experimental. Grams updates (arXiv 2412.17107): take the update's signs from the current gradient.", 'type': 'bool'},
            'use_adopt': {'title': 'ADOPT', 'tooltip': 'Experimental. Partial ADOPT (arXiv 2411.02853): update the second moment after the step, so the current gradient is not in the denominator.', 'type': 'bool'},
            'd_limiter': {'title': 'D Limiter', 'tooltip': 'Prevent over-estimated LRs when gradients and EMA are still stabilizing', 'type': 'bool'},
            'use_schedulefree': {'title': 'Schedule-Free', 'tooltip': 'Use the Schedule-Free version of the optimizer. Off: a modified reference Prodigy that may need an LR schedule (cosine is recommended).', 'type': 'bool'},
            'use_orthograd': {'title': 'OrthoGrad', 'tooltip': 'Experimental. OrthoGrad (arXiv 2501.04697): update with only the gradient component orthogonal to the weights. Can reduce overfitting.', 'type': 'bool'},
            'nnmf_factor': {'title': 'Factored Optimizer', 'tooltip': 'Enables a memory-efficient mode by applying fast low-rank factorization to the optimizers states. It combines factorization for magnitudes with 1-bit compression for signs, drastically reducing VRAM usage and allowing for larger models or batch sizes. This is an approximation which may slightly alter training dynamics.', 'type': 'bool'},
            'nesterov_coef': {'title': 'Nesterov Coefficient', 'tooltip': 'Mixing coefficient of the Nesterov update: coef * momentum + (1 - coef) * gradient. Empty uses Beta1 (or the momentum value). Lower values weight the current gradient more; this replaces the former Simplified AdEMAMix "Grad α". Only used when Nesterov is on and Beta1/momentum is above 0.', 'type': 'float'},
            'state_precision': {'title': 'State Precision', 'tooltip': 'Storage format of the optimizer state. auto: the parameter dtype (the former behavior). fp32: full precision. factored: rank-1 factored low-memory state (the former Factored Optimizer option). bf16_sr / int8_sr: bfloat16 / 8-bit state with stochastic rounding. fp16: half precision (ADOPT_ADV only). Only the modes that work with the selected optimizer are listed.', 'type': 'choice', 'values': adv_state_precisions},
            'factored_2nd': {'title': 'Factored 2nd Moment', 'tooltip': 'Factorizes only the second moment (the variance estimate) to save memory, while the first moment keeps the State Precision format. Works together with any State Precision except fp16 (ADOPT_ADV).', 'type': 'bool'},
            'fisher_wd': {'title': 'Fisher Weight Decay', 'tooltip': 'Scales the weight decay by the second-moment (Fisher) estimate, as in the FAdam paper, instead of applying it uniformly.', 'type': 'bool'},
            'centered_wd': {'title': 'Centered Weight Decay', 'tooltip': 'Decays the weights toward their values at the start of training (the anchor) instead of toward zero. Can be combined with the normal weight decay. 0 or empty disables it. Stores a copy of the trained weights, see Anchor Precision.', 'type': 'float'},
            'centered_wd_mode': {'title': 'Anchor Precision', 'tooltip': 'Storage format of the Centered Weight Decay anchor. full: the parameter dtype (default). float8: an unscaled float8 e4m3 cast, about 3-11% off the weights and small weights become 0. int8 / int4: block-wise quantized. Only used when Centered Weight Decay is above 0, and not for LoRA/OFT weights (their decay is already centered).', 'type': 'choice', 'values': ['full', 'float8', 'int8', 'int4']},
            'spectral_normalization': {'title': 'Spectral Scaling', 'tooltip': 'Scales each update by its spectral norm (estimated by power iteration), which makes the update size independent of the layer width or LoRA rank.', 'type': 'bool'},
            'normed_momentum': {'title': 'Normed Momentum', 'tooltip': 'Normalizes the gradient before it enters the momentum (normalization then momentum) instead of normalizing the momentum.', 'type': 'bool'},
            'snr_cond': {'title': 'SNR Preconditioning', 'tooltip': 'Variance/confidence preconditioning: scales each update by how consistent its gradient is. Requires Normed Momentum.', 'type': 'bool'},
            'stochastic_sign': {'title': 'Stochastic Sign', 'tooltip': 'Uses an adaptive stochastic sign operator with L-infinity preconditioning instead of the plain sign of the update.', 'type': 'bool'},
            'geometric_wd': {'title': 'Geometric Weight Decay', 'tooltip': 'Decays weights in dominant rows/columns more strongly and protects under-used ones, instead of a uniform weight decay.', 'type': 'bool'},
            'scaled_wd': {'title': 'Scaled Weight Decay', 'tooltip': 'Scales the weight decay to each weight: a matrix gets it divided by its input width (an OFT block by half its block size), and vectors (DoRA magnitudes, biases) get no normal weight decay. It follows the learning rate schedule (the learning rate divided by its starting value; Prodigy: times its step size D) instead of being multiplied by the learning rate itself. Meant for much larger values than plain weight decay: adv_optm suggests starting at 0.1, together with Spectral Scaling.', 'type': 'bool'},
            'sinkhorn_iterations': {'title': 'Sinkhorn Iterations', 'tooltip': 'Number of Sinkhorn row/column normalization iterations applied to each update.', 'type': 'int'},
            'orthogonal_sinkhorn': {'title': 'Orthogonal Sinkhorn', 'tooltip': 'Uses the orthogonal variant of the Sinkhorn normalization.', 'type': 'bool'},
            'orthogonal_gradient': {'title': 'OrthoGrad', 'tooltip': 'Removes the gradient component parallel to the weight, which reduces overfitting and improves generalization. flattened: the original OrthoGrad over the whole tensor. iterative: a matrix-wise variant (adv_optm 2.5). disabled: off.', 'type': 'choice', 'values': ['disabled', 'flattened', 'iterative']},
            'use_atan2': {'title': 'Atan2 Scaling', 'tooltip': 'A robust replacement for eps, which also incorporates gradient clipping, bounding and stabilizing the optimizer updates.', 'type': 'bool'},
            'beta1_warmup': {'title': 'Beta1 Warmup Steps', 'tooltip': 'Number of warmup steps to gradually increase beta1 from Minimum Beta1 Value to its final value. During warmup, beta1 increases linearly. leave it empty to disable warmup and use constant beta1.', 'type': 'int'},
            'min_beta1': {'title': 'Minimum Beta1', 'tooltip': 'Starting beta1 value for warmup scheduling. Used only when beta1 warmup is enabled. Lower values allow faster initial adaptation, while higher values provide more smoothing. The final beta1 value is specified in the beta1 parameter.', 'type': 'float'},
            'kourkoutas_beta': {'title': 'Kourkoutas Beta', 'tooltip': 'Enables a layer-wise dynamic β₂ adaptation. This feature makes the optimizer more responsive to "spiky" gradients by lowering β₂ during periods of high variance, and more stable during calm periods by raising β₂ towards its maximum. It can significantly improve training stability and final loss.', 'type': 'bool'},
            'schedulefree_c': {'title': 'Schedule free averaging strength', 'tooltip': 'Larger values = more responsive (shorter averaging window); smaller values = smoother (longer window). Set to 0 to disable and use the original Schedule-Free rule. Short small batches (≈6-12); long/large-batch (≈50-200).', 'type': 'float'},
            'ns_steps': {'title': 'Newton-Schulz Iterations', 'tooltip': 'Controls the number of iterations for update orthogonalization. Higher values improve the updates quality but make each step slower. Lower values are faster per step but may be less effective.', 'type': 'int'},
            'MuonWithAuxAdam': {'title': 'MuonWithAuxAdam', 'tooltip': 'Whether to use the standard way of Muon. Non-hidden layers fallback to ADAMW, and MUON takes the rest. Note: The auxiliary Adam (ADAMW) is typically only relevant for training "full" LoRA (LoRA for all layers) or full finetune and is irrelevant for most common LoRA use cases.', 'type': 'bool'},
            'muon_hidden_layers': {'title': 'Hidden Layers', 'tooltip': 'Comma-separated list of hidden layers to train using Muon. Regular expressions (if toggled) are supported. Any model layer with a matching name will be trained using Muon. If None is provided it will default to using automatic way of finding hidden layers.', 'type': 'str'},
            'muon_adam_regex': {'title': 'Use Regex', 'tooltip': 'Whether to use regular expressions for hidden layers.', 'type': 'bool'},
            'muon_adam_lr': {'title': 'Auxiliary Adam LR', 'tooltip': 'Learning rate for the auxiliary AdamW optimizer. If empty, it will use the main learning rate.', 'type': 'float'},
            'muon_te1_adam_lr': {'title': 'AuxAdam TE1 LR', 'tooltip': 'Learning rate for the auxiliary AdamW optimizer for the first text encoder. If empty, it will use the Auxiliary Adam LR.', 'type': 'float'},
            'muon_te2_adam_lr': {'title': 'AuxAdam TE2 LR', 'tooltip': 'Learning rate for the auxiliary AdamW optimizer for the second text encoder. If empty, it will use the Auxiliary Adam LR.', 'type': 'float'},
            'rms_rescaling': {'title': 'RMS Rescaling', 'tooltip': 'Muon already scales its updates to approximate and use the same learning rate (LR) as Adam. This option integrates a more accurate method to match the Adam LR, but it is slower.', 'type': 'bool'},
            'normuon_variant': {'title': 'NorMuon Variant', 'tooltip': 'Enables the NorMuon optimizer variant, which combines Muon orthogonalization with per-neuron adaptive learning rates for better convergence and balanced parameter updates. Costs only one scalar state buffer per parameter group, size few KBs, maintaining high memory efficiency.', 'type': 'bool'},
            'beta2_normuon': {'title': 'NorMuon Beta2', 'tooltip': 'Exponential decay rate for the neuron-wise second-moment estimator in NorMuon (analogous to Adams beta2). Controls how past squared updates influence current normalization.', 'type': 'float'},
            'low_rank_ortho': {'title': 'Low-rank Orthogonalization', 'tooltip': 'Use low-rank orthogonalization to accelerate Muon by orthogonalizing only in a low-dimensional subspace, improving speed and noise robustness.', 'type': 'bool'},
            'ortho_rank': {'title': 'Ortho Rank', 'tooltip': 'Target rank for low-rank orthogonalization. Controls the dimensionality of the subspace used for efficient and noise-robust orthogonalization.', 'type': 'int'},
            'accelerated_ns': {'title': 'Accelerated Newton-Schulz', 'tooltip': 'Applies an enhanced Newton-Schulz variant that replaces heuristic coefficients with optimal coefficients derived at each step. This improves performance and convergence by reducing the number of required operations.', 'type': 'bool'},
            'cautious_wd': {'title': 'Cautious Weight Decay', 'tooltip': 'Applies weight decay only to parameter coordinates whose signs align with the optimizer update direction. This preserves the original optimization objective while still benefiting from regularization effects, leading to improved convergence and better final performance.', 'type': 'bool'},
            'approx_mars': {'title': 'Approx MARS-M', 'tooltip': 'Enables Approximated MARS-M, a variance reduction technique. It uses the previous step\'s gradient to correct the current update, leading to lower losses and improved convergence stability. This requires additional state to store the previous gradient.', 'type': 'bool'},
            'auto_kappa_p': {'title': 'Auto Lion-K', 'tooltip': 'Automatically determines the optimal P-value based on layer dimensions. Uses p=2.0 (Spherical) for 4D (Conv) tensors for stability and rotational invariance, and p=1.0 (Sign) for 2D (Linear) tensors for sparsity. Overrides the manual P-value. Recommend for unet models.', 'type': 'bool'},
            'compile': {'title': 'Compiled Optimizer', 'tooltip': 'Enables PyTorch compilation for the optimizer internal step logic. This is intended to improve performance by allowing PyTorch to fuse operations and optimize the computational graph.', 'type': 'bool'},
        }
        # @formatter:on

        selected_optimizer = controller.config.optimizer.optimizer
        dependencies = OptionDependencies(self.components, optimizer_ui_state,
                                          OPTIMIZER_DEFAULT_PARAMETERS[selected_optimizer].keys())

        def on_change(*args):
            update_user_pref_cb(*args)
            dependencies.update()

        # Extract the keys for the selected optimizer
        for index, key in enumerate(OPTIMIZER_DEFAULT_PARAMETERS[selected_optimizer].keys()):
            if key not in KEY_DETAIL_MAP:
                continue
            arg_info = KEY_DETAIL_MAP[key]

            title = arg_info['title']
            tooltip = arg_info['tooltip']
            type = arg_info['type']

            row = (index // 2) + 1
            col = 3 * (index % 2)

            label = self.components.label(master, row, col, title, tooltip=tooltip)

            if key == 'MuonWithAuxAdam':
                frame = self.components.inline_frame(master, row, col + 1, columnspan=2)

                self.components.switch(frame, 0, 0, optimizer_ui_state, key, command=on_change)

                self.muon_adam_button = self.components.button(
                    frame, 0, 1, "...", open_muon_adam_cb,
                    tooltip="Configure the auxiliary AdamW_adv optimizer",
                    width=20, padx=5)
                dependencies.add(key, self.muon_adam_button, rule=lambda value: _on(value('MuonWithAuxAdam')))
            elif type == 'choice':
                values = arg_info['values'](selected_optimizer) if callable(arg_info['values']) else arg_info['values']
                dependencies.add(key, self.components.options(master, row, col + 1, values, optimizer_ui_state, key,
                                                              command=on_change), label)
            elif type != 'bool':
                dependencies.add(key, self.components.entry(master, row, col + 1, optimizer_ui_state, key,
                                                            command=on_change), label)
            else:
                dependencies.add(key, self.components.switch(master, row, col + 1, optimizer_ui_state, key,
                                                             command=on_change), label)

        dependencies.update()
