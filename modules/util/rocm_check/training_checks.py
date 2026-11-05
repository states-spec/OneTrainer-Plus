"""The training-side code on the GPU: adapters, optimizers (through OneTrainer's own create_optimizer, with a
save/resume round trip), layer offloading through the real LayerOffloadConductor, pinned memory, and concurrent
VAE encodes (the ROCm NaN-latent bug the caching lock works around).
"""
import copy
import importlib.util
import io
import itertools
import os
import threading
import time

from modules.util import rocm_sdpa_fix
from modules.util.rocm_check.framework import Ctx, Rec, check, finite, rel_err

import torch
import torch.nn.functional as F
from torch import nn

SECTION = "3 training code"


def _rand(ctx: Ctx, *shape, scale=1.0, seed=None) -> torch.Tensor:
    return torch.randn(*shape, generator=ctx.generator(seed)) * scale


def _adapter_factories(device: torch.device):
    from modules.module.LoRAModule import DoRAModule, LoHaModule, LoKrModule, LoRAModule, OFTModule

    return {
        "LoRA": lambda lin, dev: LoRAModule("p", lin, 16, 1.0),
        "DoRA": lambda lin, dev: DoRAModule("p", lin, 16, 1.0, norm_epsilon=False, decompose_output_axis=False,
                                            train_device=dev),
        "LoKr (factor 16)": lambda lin, dev: LoKrModule("p", lin, 16, 1.0, False, 16, False, False, False, False, dev),
        "LoKr vec trick": lambda lin, dev: LoKrModule("p", lin, 16, 1.0, False, 16, False, False, False, False, dev,
                                                      lokr_vec_trick=True),
        "LoKr + DoRA": lambda lin, dev: LoKrModule("p", lin, 16, 1.0, False, 16, False, True, False, False, dev),
        "LoHa": lambda lin, dev: LoHaModule("p", lin, 16, 1.0),
        "OFT v2": lambda lin, dev: OFTModule("p", lin, 32, False, False, dropout_probability=0.0),
        # the check's weights give a rotation norm of ~0.2, so a limit of 0.05 keeps the clipping active
        "OFT v2 clipped": lambda lin, dev: OFTModule("p", lin, 32, False, False, oft_clipped_norm=0.05,
                                                     dropout_probability=0.0),
    }


def _build_adapter(ctx: Ctx, factory, device: torch.device, in_f: int, out_f: int, base_dtype: torch.dtype):
    torch.manual_seed(ctx.seed)  # the adapters initialize from the global generator: same start for every build
    lin = nn.Linear(in_f, out_f)
    with torch.no_grad():
        lin.weight.copy_(_rand(ctx, out_f, in_f, scale=in_f ** -0.5, seed=1))
        lin.bias.copy_(_rand(ctx, out_f, scale=0.1, seed=2))
    lin.requires_grad_(False)
    adapter = factory(lin, torch.device("cpu"))
    with torch.no_grad():  # move every adapter weight off its init (LoRA up = 0) so the delta is exercised
        for i, p in enumerate(adapter.parameters()):
            p.add_(_rand(ctx, *p.shape, scale=0.02, seed=10 + i))
    lin.to(device, base_dtype)
    adapter.to(device)
    if hasattr(adapter, "train_device"):
        adapter.train_device = device
    adapter.hook_to_module()
    return lin, adapter


@check(SECTION, "adapters forward+backward (LoRA, DoRA, LoKr, LoHa, OFT)")
def adapters(ctx: Ctx, rec: Rec):
    in_f, out_f, tokens = 256, 384, 64
    x = _rand(ctx, 2, tokens, in_f, seed=3)
    g = _rand(ctx, 2, tokens, out_f, seed=4)
    on_gpu = ctx.device.type == "cuda"
    for name, factory in _adapter_factories(ctx.device).items():
        ref_lin, ref_adapter = _build_adapter(ctx, factory, torch.device("cpu"), in_f, out_f, torch.float32)
        y_ref = ref_lin(x)
        y_ref.backward(g)

        lin, adapter = _build_adapter(ctx, factory, ctx.device, in_f, out_f,
                                      torch.bfloat16 if on_gpu else torch.float32)
        with torch.autocast(ctx.device.type, dtype=torch.bfloat16, enabled=on_gpu):
            y = lin(x.to(ctx.device))
        y.float().backward(g.to(ctx.device))
        err_y = rel_err(y, y_ref)
        err_g = max(rel_err(p.grad, r.grad) for p, r in zip(adapter.parameters(), ref_adapter.parameters(),
                                                          strict=True))
        ok = finite(y) and all(finite(p.grad) for p in adapter.parameters())
        tol = 0.03 if on_gpu else 1e-4
        rec.expect(ok and err_y < tol and err_g < tol * 2,
                   f"{name}: out rel err {err_y:.1e}, grads {err_g:.1e} (limits {tol:.0e}/{tol * 2:.0e}, "
                   f"{'bf16 autocast' if on_gpu else 'fp32'} vs fp32 CPU)")

    if ctx.full and on_gpu:
        in_f = out_f = 3072
        x = _rand(ctx, 2, 1024, in_f, seed=5).to(ctx.device, torch.bfloat16)
        base = nn.Linear(in_f, out_f).to(ctx.device, torch.bfloat16).requires_grad_(False)
        x.requires_grad_()
        # the GPU raises its clocks only after some time under load: without this, the first timing (no adapter)
        # came out slower than LoRA and OFT on the 7900 XTX
        warm_until = time.perf_counter() + 0.5
        while time.perf_counter() < warm_until:
            base(x).float().sum().backward()
        ctx.sync()
        t_base = ctx.bench(lambda: base(x).float().sum().backward())
        rec.metric("fwd+bwd ms 2x1024x3072 no adapter", t_base)
        for name, factory in _adapter_factories(ctx.device).items():
            lin, _ = _build_adapter(ctx, factory, ctx.device, in_f, out_f, torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = ctx.bench(lambda lin=lin: lin(x).float().sum().backward())
            rec.metric(f"fwd+bwd ms 2x1024x3072 {name}", t)


def _bitsandbytes_installed() -> bool:
    # e.g. AMD's ROCm 10.1 builds (HIP 7.16): no bitsandbytes release has a build for them (docs/RocmVersionComparison.md)
    return importlib.util.find_spec("bitsandbytes") is not None


def _optimizer_list(ctx: Ctx):
    from modules.util.enum.Optimizer import Optimizer
    from modules.util.optimizer_util import adv_state_precisions

    optimizers = [Optimizer.ADOPT_ADV, Optimizer.PRODIGY_ADV, Optimizer.ADAMW]
    if ctx.full:
        optimizers += [Optimizer.ADAMW_ADV, Optimizer.PRODIGY_PLUS_SCHEDULE_FREE, Optimizer.SCHEDULE_FREE_ADAMW]
        if ctx.is_gpu and _bitsandbytes_installed():
            optimizers.append(Optimizer.ADAMW_8BIT)
    out = []
    for opt in optimizers:
        if opt.is_adv_optm:
            precisions = adv_state_precisions(opt) if ctx.full else ["auto", "int8_sr"]
            out += [(opt, p) for p in precisions]
        else:
            out.append((opt, None))
    return out


class _OptimizerRun:
    """one optimizer on a LoRA-like parameter set, built the way training builds it"""

    SHAPES = [(16, 512), (512, 16), (512, 1), (512,)]

    def __init__(self, ctx: Ctx, optimizer, precision, device, dtype, params=None, state_dict=None, overrides=None):
        from modules.util import create
        from modules.util.config.TrainConfig import TrainConfig
        from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
        from modules.util.optimizer_util import default_optimizer_config
        from modules.util.torch_util import optimizer_to_device_

        self.ctx, self.optimizer_enum, self.device = ctx, optimizer, device
        config = TrainConfig.default_values()
        config.optimizer = default_optimizer_config(optimizer)
        if precision is not None:
            config.optimizer.state_precision = precision
        for key, value in (overrides or {}).items():
            setattr(config.optimizer, key, value)
        config.learning_rate = 1.0 if "PRODIGY" in str(optimizer) else 1e-3
        config.learning_rate_warmup_steps = 0
        config.train_device = str(device)
        if params is None:
            params = [_rand(ctx, *s, scale=0.1, seed=20 + i) for i, s in enumerate(self.SHAPES)]
        self.params = [nn.Parameter(p.detach().clone().to(device, dtype)) for p in params]
        self.collection = NamedParameterGroupCollection()
        self.collection.add_group(NamedParameterGroup("lora", self.params, None, "lora"))
        self.opt = create.create_optimizer(self.collection, state_dict, config)
        optimizer_to_device_(self.opt, device)
        if hasattr(self.opt, "train"):
            self.opt.train()  # schedule-free optimizers

    def step(self, i: int):
        for j, p in enumerate(self.params):
            p.grad = _rand(self.ctx, *p.shape, seed=1000 * i + j).to(self.device, p.dtype)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def saved_state(self) -> dict:
        # like InternalModelSaverMixin: the extra keys, then a torch.save / torch.load round trip
        state = self.opt.state_dict()
        if any('optim_type' in group for group in self.opt.param_groups):
            # Muon's groups, named as optimizer_util.init_model_parameters names them for the saver
            state["param_group_mapping"] = [f"{group.get('name')}_{group.get('optim_type', 'unknown')}"
                                            for group in self.opt.param_groups]
        else:
            state["param_group_mapping"] = self.collection.unique_name_mapping
        state["param_group_optimizer_mapping"] = [str(self.optimizer_enum)] * len(state["param_group_mapping"])
        if self.optimizer_enum.is_adv_optm:
            import adv_optm
            state["adv_optm_version"] = getattr(adv_optm, "__version__", "unknown")
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        return torch.load(buffer, weights_only=True)

    def weights(self) -> list[torch.Tensor]:
        if hasattr(self.opt, "eval"):
            self.opt.eval()
        w = [p.detach().float().cpu().clone() for p in self.params]
        if hasattr(self.opt, "train"):
            self.opt.train()
        return w


@check(SECTION, "optimizers step, state precision and save/resume (create_optimizer)")
def optimizers(ctx: Ctx, rec: Rec):
    dtypes = [torch.float32, torch.bfloat16]
    for optimizer, precision in _optimizer_list(ctx):
        label_opt = f"{optimizer}{'' if precision is None else f' {precision}'}"
        for dtype in dtypes:
            label = f"{label_opt} {str(dtype)[6:]}"
            try:
                run = _OptimizerRun(ctx, optimizer, precision, ctx.device, dtype)
            except Exception as e:
                rec.fail(f"{label}: create_optimizer failed: {type(e).__name__}: {e}")
                continue
            for i in range(3):
                run.step(i)
            resumed = _OptimizerRun(ctx, optimizer, precision, ctx.device, dtype, params=run.params,
                                    state_dict=run.saved_state())
            for i in range(3, 6):
                run.step(i)
                resumed.step(i)
            w, w_resumed = run.weights(), resumed.weights()
            ok = all(finite(t) for t in w + w_resumed)
            diff = max(rel_err(a, b) for a, b in zip(w_resumed, w, strict=True))
            stochastic = precision is not None and precision.endswith("_sr") or dtype == torch.bfloat16
            limit = 1e-2 if stochastic else 1e-6
            if not ok:
                rec.fail(f"{label}: NaN/inf in the weights")
            elif diff > limit:
                rec.fail(f"{label}: resumed run differs from the uninterrupted one by rel {diff:.1e} (limit {limit})")
            else:
                rec.line(f"ok: {label}: 6 steps, resume diff {diff:.1e}")

        if ctx.device.type != "cpu" and precision in (None, "auto", "fp32"):
            # the same fp32 run on the GPU and on the CPU should end at (nearly) the same weights
            runs = {}
            for device in (ctx.device, torch.device("cpu")):
                try:
                    r = _OptimizerRun(ctx, optimizer, precision, device, torch.float32)
                    for i in range(6):
                        r.step(i)
                except RuntimeError as e:
                    if device.type != "cpu":
                        raise
                    # e.g. bitsandbytes' 8-bit optimizers run only on the GPU
                    rec.line(f"{label_opt}: no CPU run to compare with ({str(e).splitlines()[0][:80]})")
                    break
                runs[device.type] = r.weights()
            if "cpu" not in runs:
                continue
            diff = max(rel_err(a, b) for a, b in zip(runs["cuda"], runs["cpu"], strict=True))
            rec.expect(diff < 1e-3, f"{label_opt} fp32: GPU vs CPU after 6 steps rel diff {diff:.1e} (limit 1e-3)")


# the optimizers create.py builds from bitsandbytes classes, 32-bit ones included (CAME_8BIT is OneTrainer's own);
# they run only on the GPU
_BNB_OPTIMIZERS = ("ADAGRAD", "ADAGRAD_8BIT", "ADAM_8BIT", "ADAMW_8BIT", "AdEMAMix", "AdEMAMix_8BIT", "LAMB",
                   "LAMB_8BIT", "LARS", "LARS_8BIT", "LION_8BIT", "RMSPROP", "RMSPROP_8BIT", "SGD_8BIT")


@check(SECTION, "every optimizer with its UI defaults: create, step, save/resume, fused back pass (create_optimizer)")
def all_optimizers(ctx: Ctx, rec: Rec):
    from modules.util.enum.Optimizer import Optimizer

    # 2D parameters only, as a LoRA has: Muon and AdaMuon without the auxiliary Adam refuse 1D ones
    shapes = [(16, 64), (64, 16), (64, 1)]
    names = []
    bnb_installed = _bitsandbytes_installed()
    for optimizer in Optimizer:
        if optimizer.name in _BNB_OPTIMIZERS and not (ctx.is_gpu and bnb_installed):
            names.append(f"{optimizer} (bitsandbytes, {'GPU only' if bnb_installed else 'not installed'}: not run)")
            continue
        params = [_rand(ctx, *shape, scale=0.1, seed=40 + i) for i, shape in enumerate(shapes)]
        try:
            run = _OptimizerRun(ctx, optimizer, None, ctx.device, torch.float32, params=params)
            for i in range(3):
                run.step(i)
            resumed = _OptimizerRun(ctx, optimizer, None, ctx.device, torch.float32, params=run.params,
                                    state_dict=run.saved_state())
            for i in range(3, 6):
                run.step(i)
                resumed.step(i)
            w, w_resumed = run.weights(), resumed.weights()
            if not all(finite(t) for t in w + w_resumed):
                rec.fail(f"{optimizer}: NaN/inf in the weights")
                continue
            diff = max(rel_err(a, b) for a, b in zip(w_resumed, w, strict=True))
            moved = max(rel_err(a, b.to(a.device)) for a, b in zip(w, params, strict=True))
            if moved == 0:
                rec.fail(f"{optimizer}: 6 steps left the weights unchanged")
                continue
            if diff > 1e-5:
                # a state that save/load doesn't restore exactly; not always wrong (e.g. state outside state_dict)
                rec.warn(f"{optimizer}: the resumed run differs from the uninterrupted one by rel {diff:.1e}")
            text = f"{optimizer}: moved rel {moved:.1e} in 6 steps, resume diff {diff:.1e}"
            if optimizer.supports_fused_back_pass():
                # ADAM/ADAMW step parameters one at a time only with their plain implementation
                fused = _OptimizerRun(ctx, optimizer, None, ctx.device, torch.float32, params=params,
                                      overrides={"fused_back_pass": True, "fused": False, "foreach": False})
                for i in range(2):
                    for group in fused.opt.param_groups:
                        for j, p in enumerate(group["params"]):
                            p.grad = _rand(ctx, *p.shape, seed=2000 * i + j).to(ctx.device)
                            fused.opt.step_parameter(p, group, j)
                            p.grad = None
                if not all(finite(t) for t in fused.weights()):
                    rec.fail(f"{optimizer}: fused back pass gave NaN/inf weights")
                    continue
                text += ", fused back pass ok"
            names.append(text)
        except Exception as e:
            rec.fail(f"{optimizer}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    for text in names:
        rec.line(text)


@check(SECTION, "PRODIGY_ADV Nesterov: its step against AdamW_adv's Nesterov with D held fixed")
def prodigy_adv_nesterov(ctx: Ctx, rec: Rec):
    # With D held at Initial D (d_coef 0), Prodigy is Adam without bias correction at learning rate D * lr: its
    # moments are D and D^2 times Adam's. So its Nesterov step has to give AdamW_adv's Nesterov step. Before 2.6.1,
    # adv_optm's Prodigy_adv mixed the raw gradient into its D-scaled momentum (training refuses Nesterov there)
    import adv_optm
    from adv_optm import AdamW_adv, Prodigy_adv

    d0, lr, steps = 3e-4, 1.0, 40

    def run(nesterov, reference=False, **options):
        g = torch.Generator().manual_seed(0)
        params = [nn.Parameter((torch.randn(32, 48, generator=g) * 0.1).to(ctx.device)),
                  nn.Parameter((torch.randn(48, generator=g) * 0.1).to(ctx.device))]
        start = [p.detach().clone() for p in params]
        common = dict(betas=(0.9, 0.99), eps=1e-8, nesterov=nesterov, stochastic_rounding=False, **options)
        optimizer = AdamW_adv(params, lr=d0 * lr, use_bias_correction=False, **common) if reference \
            else Prodigy_adv(params, lr=lr, d0=d0, d_coef=0.0, **common)
        g = torch.Generator().manual_seed(1)
        for _ in range(steps):
            for p in params:
                p.grad = torch.randn(p.shape, generator=g).to(ctx.device)
            optimizer.step()
        weights = torch.cat([p.detach().flatten() for p in params])
        moved = torch.cat([(p.detach() - s).flatten() for p, s in zip(params, start, strict=True)]).abs().max()
        return weights, moved.item()

    rec.line(f"adv_optm {adv_optm.__version__}")
    for label, options in (("defaults", {}), ("coefficient 0.5", {"nesterov_coef": 0.5}),
                           ("factored state", {"state_precision": "factored"}), ("atan2", {"use_atan2": True})):
        reference, moved = run(True, reference=True, **options)
        prodigy, _ = run(True, **options)
        err = (prodigy - reference).abs().max().item() / moved
        rec.expect(err < 1e-4, f"{label}: Nesterov step vs AdamW_adv's after {steps} steps: {err:.1e} of the distance "
                               f"moved (limit 1e-4)")


@check(SECTION, "ADOPT_ADV Nesterov: the same step at gradient scales 1e-2 and 1e2")
def adopt_adv_nesterov(ctx: Ctx, rec: Rec):
    # ADOPT normalizes the gradient by its running second moment, so the same gradients scaled by any factor move the
    # weights the same way, and its Nesterov step has to keep that. Before adv_optm 2.6.2 it mixed the raw gradient into
    # the normalized momentum: Nesterov then only shrank the step with small gradients and blew it up with large ones
    # (training refuses Nesterov there)
    import adv_optm
    from adv_optm import Adopt_adv

    def run(scale, **options):
        g = torch.Generator().manual_seed(0)
        params = [nn.Parameter((torch.randn(32, 48, generator=g) * 0.1).to(ctx.device)),
                  nn.Parameter((torch.randn(48, generator=g) * 0.1).to(ctx.device))]
        start = [p.detach().clone() for p in params]
        optimizer = Adopt_adv(params, lr=1e-3, nesterov=True, stochastic_rounding=False, **options)
        g = torch.Generator().manual_seed(1)
        for _ in range(30):
            for p in params:
                p.grad = (torch.randn(p.shape, generator=g) * scale).to(ctx.device)
            optimizer.step()
        return torch.cat([(p.detach() - s).flatten() for p, s in zip(params, start, strict=True)])

    rec.line(f"adv_optm {adv_optm.__version__}")
    for label, options in (("defaults", {}), ("atan2", {"use_atan2": True}), ("coefficient 0.5", {"nesterov_coef": 0.5}),
                           ("factored state", {"state_precision": "factored"})):
        small, large = run(1e-2, **options), run(1e2, **options)
        err = ((large - small).abs().max() / small.abs().max()).item()
        rec.expect(err < 1e-3, f"{label}: step at gradient scale 1e2 vs 1e-2: {err:.1e} of the distance moved "
                               f"(limit 1e-3)")


@check(SECTION, "timestep sampling stays inside the noise schedule (every distribution, wide settings, shifts)")
def timestep_range(ctx: Ctx, rec: Rec):
    from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.TimestepDistribution import TimestepDistribution

    # The sampled timestep indexes the noise schedule, and wide settings reach its end: LOGIT_NORMAL at the Chroma
    # presets' noising weight 7.7 drew num_train_timesteps itself for 2% of the samples, one past the last entry (a
    # device-side assert on the GPU, which aborted the training run under ROCm). The samples are only compared here,
    # never used as an index, so a regression fails this check instead of ending rocm_check.
    settings = [(0.0, 0.0), (3.0, 1.0), (7.7, 0.0), (20.0, -2.0), (1.0, 0.5), (0.5, 1.0)]
    sampled, refused = 0, []
    for distribution, (weight, bias), shift in itertools.product(TimestepDistribution, settings, (1.0, 3.0)):
        config = TrainConfig.default_values()
        config.timestep_distribution, config.noising_weight, config.noising_bias = distribution, weight, bias
        config.timestep_shift = shift
        for n in (1000, 10000):  # 10000: the continuous models' discretization (_get_timestep_continuous)
            sampler = type("Sampler", (ModelSetupNoiseMixin,), {})()  # a fresh one: the discrete ones cache weights
            generator = torch.Generator(device=ctx.device).manual_seed(ctx.seed + sampled)
            try:
                timestep = sampler._get_timestep_discrete(n, False, generator, 16384, config)
            except ValueError:  # no valid weights (INVERTED_PARABOLA with its peak far outside): refused with a message
                refused.append(f"{distribution} {weight}/{bias}/{shift}")
                continue
            sampled += 1
            low, high = int(timestep.min()), int(timestep.max())
            if low < 0 or high >= n:
                rec.fail(f"{distribution}, noising weight {weight}, bias {bias}, shift {shift}: timesteps {low}..{high}, "
                         f"outside the schedule's 0..{n - 1}")
    rec.line(f"{sampled} settings x 16384 samples on {ctx.device}; refused for having no valid weights: "
             f"{', '.join(sorted(set(refused))) or 'none'}")


class _Block(nn.Module):
    def __init__(self, dim, expansion, adapter):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim * expansion)
        self.lin2 = nn.Linear(dim * expansion, dim)
        self._adapter = [adapter]  # kept out of the module tree, like a LoRA

    def forward(self, hidden_states, temb):
        h = self.lin2(F.gelu(self.lin1(hidden_states)))
        return hidden_states + h * (temb + self._adapter[0])


class _Toy(nn.Module):
    def __init__(self, dim, layers, expansion):
        super().__init__()
        self.adapters = nn.ParameterList([nn.Parameter(torch.randn(dim) * 0.1) for _ in range(layers)])
        self.blocks = nn.ModuleList([_Block(dim, expansion, a) for a in self.adapters])
        self.blocks.requires_grad_(False)

    def forward(self, x, temb):
        for b in self.blocks:
            x = b(x, temb)
        return x


@check(SECTION, "layer + activation offloading through LayerOffloadConductor")
def offloading(ctx: Ctx, rec: Rec):
    from modules.util.checkpointing_util import enable_checkpointing
    from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig

    dim, layers, expansion = (1024, 12, 4) if ctx.full else (512, 8, 4)
    variants = [(f, a, s) for f in (0.5, 0.9) for a in (False, True) for s in (True, False)]
    if not ctx.full:
        variants = [(0.5, True, True), (0.9, False, True), (0.5, False, False)]
    if not ctx.is_gpu:
        variants = [v for v in variants if not v[2]]  # transfers are synchronous without a GPU
    for fraction, act, async_on in variants:
        transfers = "async" if async_on and ctx.is_gpu else "sync"
        label = f"offload {fraction}, activations {'on' if act else 'off'}, {transfers} transfers"
        torch.manual_seed(0)
        model = _Toy(dim, layers, expansion)
        ref = copy.deepcopy(model).to(ctx.device)
        originals = [copy.deepcopy(b.state_dict()) for b in model.blocks]
        model.adapters.to(ctx.device)
        config = TrainConfig.default_values()
        # on a CPU-only run, "cpu:1" is a different device to the conductor (same memory): it runs its real logic
        temp_device = "cpu:1" if ctx.device.type == "cpu" else "cpu"
        config.train_device, config.temp_device, config.async_offloading = str(ctx.device), temp_device, async_on
        part = TrainModelPartConfig.default_values()
        part.offload_fraction, part.activation_offloading, part.gradient_checkpointing = fraction, act, True
        conductor = enable_checkpointing(model, config, part, False, [(model.blocks, ["hidden_states"])])
        before = torch.cuda.memory_allocated(ctx.device) if ctx.is_gpu else 0
        conductor.materialize()
        if ctx.is_gpu:
            # the values below would still match if offloaded weights stayed on the GPU; the memory wouldn't: only the
            # loaded layers (plus the cache's rounding) may be there
            total = sum(p.numel() * p.element_size() for p in model.blocks.parameters())
            used = torch.cuda.memory_allocated(ctx.device) - before
            bound = (1 - fraction) * total * 1.1 + 2 * total / layers
            rec.expect(used <= bound, f"{label}: {used / 2**20:.0f} MiB of the {total / 2**20:.0f} MiB of layers on the GPU "
                                      f"after materialize (at most {bound / 2**20:.0f} MiB with offload {fraction})")
        worst = 0.0
        for step in range(4):
            x = _rand(ctx, 2, [256, 1024, 64, 512][step], dim, seed=step).to(ctx.device)
            temb = _rand(ctx, dim, seed=100 + step).to(ctx.device)
            for p in list(model.adapters) + list(ref.adapters):
                p.grad = None
            loss = model(x, temb).pow(2).mean()
            loss.backward()
            ref_loss = ref(x, temb).pow(2).mean()
            ref_loss.backward()
            worst = max(worst, rel_err(loss, ref_loss),
                        *(rel_err(a.grad, b.grad) for a, b in zip(model.adapters, ref.adapters, strict=True)))
        rec.expect(worst < 1e-4, f"{label}: loss/grads vs no offloading rel err {worst:.1e}")
        t_off = ctx.bench(lambda model=model, x=x, temb=temb: model(x, temb).pow(2).mean().backward(), iters=5)
        t_ref = ctx.bench(lambda ref=ref, x=x, temb=temb: ref(x, temb).pow(2).mean().backward(), iters=5)
        rec.metric(f"{label}: step ms", t_off)
        rec.metric(f"{label}: step ms without offloading", t_ref)

        identical = True
        for _ in range(3):
            conductor.evict()
            conductor.materialize()
        conductor.evict()
        for block, original in zip(model.blocks, originals, strict=True):
            for key, value in block.state_dict().items():
                identical &= torch.equal(value.cpu(), original[key])
        rec.expect(identical, f"{label}: weights bit-identical after 3 materialize/evict cycles")
        del model, ref, conductor


class _StreamToy(nn.Module):
    """a frozen part with a remainder outside the layer list (input projection, final norm), as transformers have;
    the adapters live outside it, like a LoRA"""

    def __init__(self, dim, expansion, adapters):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([_Block(dim, expansion, a) for a in adapters])
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, temb):
        x = self.proj_in(x)
        for b in self.blocks:
            x = b(x, temb)
        # the blocks return fp32 (the adapters are fp32); the norm runs in its weight dtype, as in a bf16 model
        return self.norm_out(x.to(self.norm_out.weight.dtype))


@check(SECTION, "disk streaming (stream_from_disk): streamed vs loaded into RAM, with and without offloading",
       gpu_only=True, note="streams a part from a .safetensors file the way the model loaders do")
def disk_streaming(ctx: Ctx, rec: Rec):
    """a part streamed from its checkpoint (stream_from_disk, the default) must end up with the same weights and give the
    same results as the part loaded into RAM, converted and moved to the GPU (stream_from_disk off): whole, and layer by
    layer under offloading (simplex too), with cache in RAM on (evicted to RAM) and off (evicted to meta, re-streamed)"""
    import tempfile

    from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin, _stamp_skeleton_float_dtypes
    from modules.util.checkpointing_util import enable_checkpointing
    from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig
    from modules.util.disk_stream import stream_module_to
    from modules.util.enum.DataType import DataType
    from modules.util.quantization_util import quantize_layers, replace_linear_with_quantized_layers
    from modules.util.torch_util import torch_gc

    from accelerate import init_empty_weights
    from safetensors.torch import load_file, save_file

    # the loaders' own streaming closure and RAM-load conversion (private; neither uses the loader instance)
    finish_load = HFModelLoaderMixin._HFModelLoaderMixin__finish_sub_module_load
    convert_to_dtype = HFModelLoaderMixin._HFModelLoaderMixin__convert_sub_module_to_dtype

    dim, layers, expansion = (1024, 12, 4) if ctx.full else (512, 8, 4)
    train_dtype = DataType.BFLOAT_16
    # "cpu:1" on a CPU-only dry run, as in the offloading check (this check needs a GPU: streaming materializes only
    # onto a device of another type than the temp device, like TrainConfig.streams_from_disk)
    temp = torch.device("cpu:1" if ctx.device.type == "cpu" else "cpu")
    # (weight dtype, cache in RAM, offload fraction, simplex)
    variants = [(d, c, 0.0, False) for d in (DataType.BFLOAT_16, DataType.FLOAT_8) for c in (True, False)] \
        + [(d, c, 0.5, s) for d in (DataType.BFLOAT_16, DataType.FLOAT_8) for c in (True, False) for s in (False, True)]
    if not ctx.full:
        variants = [(DataType.BFLOAT_16, True, 0.0, False), (DataType.BFLOAT_16, False, 0.0, False),
                    (DataType.FLOAT_8, False, 0.0, False), (DataType.BFLOAT_16, True, 0.5, False),
                    (DataType.BFLOAT_16, False, 0.5, False), (DataType.BFLOAT_16, True, 0.5, True),
                    (DataType.FLOAT_8, False, 0.5, False)]

    def new_adapters():
        return [nn.Parameter(_rand(ctx, dim, scale=0.1, seed=50 + i).to(ctx.device)) for i in range(layers)]

    def run(model, adapters, step):
        for p in adapters:
            p.grad = None
        x = _rand(ctx, 2, [256, 1024, 64, 512][step % 4], dim, seed=step).to(ctx.device)
        temb = _rand(ctx, dim, seed=100 + step).to(ctx.device)
        with torch.autocast(ctx.device.type, dtype=train_dtype.torch_dtype()):
            loss = model(x, temb).float().pow(2).mean()
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in adapters]

    with tempfile.TemporaryDirectory(dir=ctx.out_dir if os.path.isdir(ctx.out_dir) else None) as tmp:
        path = os.path.join(tmp, "part.safetensors")
        torch.manual_seed(0)
        save_file({k: v.contiguous() for k, v in _StreamToy(dim, expansion, new_adapters()).state_dict().items()}, path)

        for dtype, cache_in_ram, fraction, simplex in variants:
            label = (f"{dtype}, cache in RAM {'on' if cache_in_ram else 'off'}, "
                     + (f"offload {fraction}{' simplex' if simplex else ''}" if fraction else "whole part"))
            # stream_from_disk off: the whole file into RAM, converted, quantized, moved to the GPU
            ref_adapters = new_adapters()
            ref = _StreamToy(dim, expansion, ref_adapters)
            ref.load_state_dict(load_file(path))
            convert_to_dtype(None, ref, dtype, train_dtype, [], None)
            ref.requires_grad_(False)
            quantize_layers(ref, ctx.device, train_dtype, None)
            ref.to(ctx.device)
            expected = ref.state_dict()  # with the fp8 scale buffers

            # stream_from_disk on: a meta skeleton, filled from the file on the GPU (quantized there)
            adapters = new_adapters()
            with init_empty_weights():
                part = _StreamToy(dim, expansion, adapters)
            replace_linear_with_quantized_layers(part, dtype, [], None, copy_parameters=False)
            _stamp_skeleton_float_dtypes(part, dtype, None, [], DataType.BFLOAT_16)
            key_to_file = dict.fromkeys(load_file(path), path)
            part, materialize_fn = finish_load(None, part, dtype, train_dtype, [], key_to_file)
            part.requires_grad_(False)  # after the quantized layers replaced the linears, as the model setup does

            conductor = None
            if fraction:
                config = TrainConfig.default_values()
                config.train_device, config.temp_device = str(ctx.device), str(temp)
                config.async_offloading, config.stream_from_disk = True, True
                part_config = TrainModelPartConfig.default_values()
                part_config.offload_fraction, part_config.gradient_checkpointing = fraction, True
                part_config.cache_in_ram, part_config.simplex_offloading = cache_in_ram, simplex
                conductor = enable_checkpointing(part, config, part_config, False, [(part.blocks, ["hidden_states"])])

            def materialize(part=part, conductor=conductor, materialize_fn=materialize_fn, cache_in_ram=cache_in_ram):
                # as BaseModel._move_part does
                if conductor is not None:
                    conductor.materialize(train_dtype, name="test part", materialize_fn=materialize_fn,
                                         cache_in_ram=cache_in_ram)
                else:
                    stream_module_to(part, ctx.device, materialize_fn, train_dtype, cache_in_ram, "test part", temp)

            def evict(part=part, conductor=conductor, materialize_fn=materialize_fn, cache_in_ram=cache_in_ram):
                if conductor is not None:
                    conductor.evict(to_meta=not cache_in_ram)
                else:
                    stream_module_to(part, temp, materialize_fn, train_dtype, cache_in_ram, "test part", temp)

            materialize()
            if conductor is None:
                got = part.state_dict()
                same = got.keys() == expected.keys() and all(
                    v.device == expected[k].device and v.dtype == expected[k].dtype and torch.equal(v, expected[k])
                    for k, v in got.items())
                rec.expect(same, f"{label}: streamed weights bit-identical to the RAM-loaded part, same device and dtype")
            worst, evicted_right = 0.0, True
            for cycle in range(3):
                for step in range(2):
                    loss, grads = run(part, adapters, step)
                    ref_loss, ref_grads = run(ref, ref_adapters, step)
                    worst = max(worst, rel_err(loss, ref_loss),
                                *(rel_err(a, b) for a, b in zip(grads, ref_grads, strict=True)))
                if cycle < 2:
                    evict()
                    # cache in RAM keeps the weights on the temp device; without it they go back to meta
                    if conductor is None:
                        evicted_right &= any(p.is_meta for p in part.parameters()) != cache_in_ram
                    materialize()
            if conductor is None:
                rec.expect(evicted_right, f"{label}: evicted to {'RAM' if cache_in_ram else 'meta'} and streamed back")
            rec.expect(worst < 1e-5, f"{label}: loss/grads vs the RAM-loaded part over 3 load cycles, rel err {worst:.1e}")
            evict()
            del ref, part, conductor, materialize_fn
            torch_gc()


@check(SECTION, "pinned host memory and transfer speed", gpu_only=True)
def pinned_memory(ctx: Ctx, rec: Rec):
    from modules.util.torch_util import pin_tensor_, unpin_tensor_

    size = 256 * 2**20
    host = torch.empty(size // 4, dtype=torch.float32)
    pin_tensor_(host)  # cudaHostRegister / hipHostRegister, as the offload caches do
    rec.expect(host.is_pinned(), "pin_tensor_ registers the memory as pinned")
    device_buf = torch.empty_like(host, device=ctx.device)
    pageable = torch.empty_like(host)
    for name, src in (("pinned", host), ("pageable", pageable)):
        ms = ctx.bench(lambda src=src: device_buf.copy_(src, non_blocking=True), iters=10)
        rec.metric(f"host->GPU GB/s ({name})", size / ms / 1e6)
        ms = ctx.bench(lambda src=src: src.copy_(device_buf, non_blocking=True), iters=10)
        rec.metric(f"GPU->host GB/s ({name})", size / ms / 1e6)
    ctx.sync()
    unpin_tensor_(host)
    rec.expect(not host.is_pinned(), "unpin_tensor_ releases it")


def _vae_from_file(path: str):
    """a VAE from one .safetensors file in the original (ComfyUI, A1111, BFL ae.safetensors) or diffusers layout, also
    inside a full checkpoint. The config is read from the tensor shapes: diffusers' from_single_file fetches it from
    Hugging Face instead, which fails offline and for gated repos."""
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        prefix = next((p for p in ("", "first_stage_model.", "vae.") if p + "encoder.conv_in.weight" in keys), None)
        if prefix is None:
            raise ValueError(f"{path} holds no VAE weights (no encoder.conv_in.weight): it is probably a transformer or "
                             "UNet only file. Pass the VAE file (e.g. ComfyUI models/vae/ae.safetensors for "
                             "Flux/Chroma) or a diffusers model folder with vae/")
        parts = tuple(prefix + part for part in ("encoder.", "decoder.", "quant_conv.", "post_quant_conv."))
        state = {k[len(prefix):]: f.get_tensor(k) for k in keys if k.startswith(parts)}

    diffusers_layout = "encoder.down_blocks.0.resnets.0.conv1.weight" in state
    block = "encoder.down_blocks.{}.resnets.0.conv2.weight" if diffusers_layout else "encoder.down.{}.block.0.conv2.weight"
    block_out_channels = []
    while block.format(len(block_out_channels)) in state:
        block_out_channels.append(state[block.format(len(block_out_channels))].shape[0])
    resnet = "encoder.down_blocks.0.resnets.{}.conv1.weight" if diffusers_layout else "encoder.down.0.block.{}.conv1.weight"
    layers_per_block = 0
    while resnet.format(layers_per_block) in state:
        layers_per_block += 1
    config = {
        "in_channels": state["encoder.conv_in.weight"].shape[1],
        "out_channels": state["decoder.conv_out.weight"].shape[0],
        "latent_channels": state["encoder.conv_out.weight"].shape[0] // 2,
        "down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "up_block_types": ["UpDecoderBlock2D"] * len(block_out_channels),
        "block_out_channels": block_out_channels,
        "layers_per_block": layers_per_block,
        "use_quant_conv": "quant_conv.weight" in state,
        "use_post_quant_conv": "post_quant_conv.weight" in state,
    }
    if not diffusers_layout:
        state = convert_ldm_vae_checkpoint(state, config)
    vae = AutoencoderKL(**config)
    vae.load_state_dict(state, strict=True)
    return vae.float()


def _vae(ctx: Ctx, device: torch.device | None = None):
    """the VAE for the VAE checks, in fp32 as training caches with it: --vae if given, else the 16-channel layout of
    the Flux/Chroma/SD3-era models, randomly initialized (no download needed)"""
    from diffusers import AutoencoderKL

    device = device or ctx.device
    if ctx.vae_path:
        if os.path.isfile(ctx.vae_path):
            vae = _vae_from_file(ctx.vae_path)
        elif os.path.isdir(os.path.join(ctx.vae_path, "vae")):
            vae = AutoencoderKL.from_pretrained(ctx.vae_path, subfolder="vae", torch_dtype=torch.float32)
        else:
            vae = AutoencoderKL.from_pretrained(ctx.vae_path, torch_dtype=torch.float32)
    else:
        torch.manual_seed(0)
        vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=16,
                            down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
                            block_out_channels=[128, 256, 512, 512], layers_per_block=2,
                            use_quant_conv=False, use_post_quant_conv=False)
    return vae.to(device).eval().requires_grad_(False)


def _vae_image(ctx: Ctx, res: int, seed: int) -> torch.Tensor:
    # a smooth random image in [-1, 1]: closer to a photo than per-pixel noise
    low = torch.rand(1, 3, max(res // 16, 2), max(res // 16, 2), generator=ctx.generator(seed))
    return (F.interpolate(low, size=(res, res), mode="bicubic", align_corners=False).clamp(0, 1) * 2 - 1)


_LAYER_TYPES = (nn.Conv2d, nn.GroupNorm, nn.Linear)


def _encode_with_layer_errors(vae, img: torch.Tensor, reference: dict | None):
    """encodes img; with a reference (layer name -> CPU output), returns each layer's rel error to it in call order.
    Without one, returns the layer outputs themselves (as the reference)."""
    from diffusers.models.attention_processor import Attention

    records, hooks = [], []
    for name, module in vae.encoder.named_modules():
        if isinstance(module, _LAYER_TYPES + (Attention,)):
            def hook(mod, args, out, name=name):
                out = out[0] if isinstance(out, tuple) else out
                if reference is None:
                    records.append((name, out.detach().float().cpu().clone()))
                else:
                    records.append((name, type(mod).__name__, rel_err(out, reference[name]), finite(out)))
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            latent = vae.encode(img).latent_dist.mean.float().cpu()
    finally:
        for h in hooks:
            h.remove()
    return (dict(records) if reference is None else records), latent


@check(SECTION, "VAE encode per layer: main thread vs worker threads, MIOpen and attention toggles",
       note="compares every layer of the VAE encoder against a CPU reference; the CPU part takes a while")
def vae_layers(ctx: Ctx, rec: Rec):
    """caching runs the VAE in worker threads. This finds the first layer (if any) whose output differs from a CPU
    float64-checked reference, per run, and which runtime toggle makes the difference go away."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    res = 128 if ctx.device.type == "cpu" else (512 if ctx.full else 256)
    img = _vae_image(ctx, res, seed=7)
    rec.line(f"{'VAE ' + ctx.vae_path if ctx.vae_path else 'random-init 16-channel VAE'}, fp32, {res}x{res} image")

    reference, ref_latent = _encode_with_layer_errors(_vae(ctx, torch.device("cpu")), img, None)
    vae = _vae(ctx)
    img_d = img.to(ctx.device)
    miopen_default = torch.backends.cudnn.enabled

    def run(label: str, thread: bool, miopen: bool = True, math_attention: bool = False, sdpa_fix: bool = True):
        out = {}

        def body():
            try:
                if math_attention:
                    with sdpa_kernel([SDPBackend.MATH]):
                        out["r"] = _encode_with_layer_errors(vae, img_d, reference)
                else:
                    out["r"] = _encode_with_layer_errors(vae, img_d, reference)
            except Exception as e:
                out["e"] = f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"

        torch.backends.cudnn.enabled = miopen
        fix_was_installed = rocm_sdpa_fix.installed()
        if not sdpa_fix:
            rocm_sdpa_fix.uninstall()
        try:
            if thread:
                t = threading.Thread(target=body)
                t.start()
                t.join()
            else:
                body()
        finally:
            torch.backends.cudnn.enabled = miopen_default
            if fix_was_installed:
                rocm_sdpa_fix.install()
        # without the fix, PyTorch's own kernel at head size 512 gives wrong numbers (AOTriton 0.12b) or fails
        # (AOTriton 0.14, which has no gfx110x kernels above 256): both are the known bug
        report = rec.warn if not sdpa_fix and rocm_sdpa_fix.applies_to(ctx.device) else rec.fail
        if "e" in out:
            report(f"{label}: {out['e']}")
            return None
        records, latent = out["r"]
        final = rel_err(latent, ref_latent)
        bad = [r for r in records if not r[3] or r[2] > 1e-3]
        if not bad and final < 1e-3:
            rec.line(f"ok: {label}: every layer matches the CPU (latent rel err {final:.1e})")
            return True
        name, kind, err, is_finite = bad[0] if bad else ("(none)", "", final, True)
        types = sorted({r[1] for r in bad})
        report(f"{label}: latent rel err {final:.2e}; first wrong layer {name} ({kind}): "
                 f"{'NaN/inf' if not is_finite else f'rel err {err:.2e}'}; wrong layer types: {', '.join(types)}")
        return False

    results = {
        "main": run("main thread", False),
        "main again": run("main thread, again", False),
        "worker": run("worker thread", True),
        "worker again": run("another worker thread", True),
    }
    if ctx.is_gpu:
        results["worker no MIOpen"] = run("worker thread, MIOpen off (torch.backends.cudnn.enabled=False)", True,
                                          miopen=False)
        results["worker math attention"] = run("worker thread, SDPA math kernel", True, math_attention=True)
        results["main no MIOpen"] = run("main thread, MIOpen off", False, miopen=False)
        if rocm_sdpa_fix.applies_to(ctx.device):
            run("main thread, without OneTrainer's attention fix (rocm_sdpa_fix; expected wrong here)", False,
                sdpa_fix=False)
    fixes = [k for k, v in results.items() if v and not results.get(k.split(" no ")[0].split(" math")[0], True)]
    if fixes:
        rec.line("the runs that match where the plain run doesn't: " + ", ".join(fixes))


@check(SECTION, "concurrent VAE encodes (caching with dataloader_threads > 1)")
def concurrent_encodes(ctx: Ctx, rec: Rec):
    """caching encodes images in several threads on the same GPU; on ROCm that once gave random all-NaN latents.
    OneTrainer now encodes one at a time (DataLoaderMgdsMixin._run_models_one_at_a_time). This measures whether
    unlocked concurrent encodes still go wrong on this GPU, and that locked ones don't."""
    encoder = _vae(ctx)
    res = 512 if ctx.full else 256
    if ctx.device.type == "cpu":
        res = 128
    count = 8 if ctx.full else 4
    rounds = 6 if ctx.full else 2
    images = [_vae_image(ctx, res, seed=i).to(ctx.device) for i in range(count)]

    def encode(img):
        with torch.no_grad():
            return encoder.encode(img).latent_dist.mean

    reference = [encode(img).cpu() for img in images]
    serial_bad = []
    for i, (img, ref) in enumerate(zip(images, reference, strict=True)):
        again = encode(img)
        if not finite(ref) or not finite(again):
            serial_bad.append(f"image {i}: NaN/inf")
        elif rel_err(again, ref) > 1e-3:
            serial_bad.append(f"image {i}: rel err {rel_err(again, ref):.1e}")
    rec.expect(not serial_bad, f"serial encodes finite and repeatable in the main thread ({len(serial_bad)} bad of "
                               f"{count}{': ' + ', '.join(serial_bad) if serial_bad else ''})")

    for threads in (2, 4):
        for locked in (False, True):
            lock = threading.Lock()
            bad, errors = [], []

            def worker(t, lock=lock, bad=bad, errors=errors, locked=locked, threads=threads):
                try:
                    for r in range(rounds):
                        for i in range(t, count, threads) if r % 2 else range(count):
                            if locked:
                                with lock:
                                    latent = encode(images[i])
                            else:
                                latent = encode(images[i])
                            if not finite(latent):
                                bad.append((i, "NaN/inf"))
                            elif rel_err(latent, reference[i]) > 1e-3:
                                bad.append((i, f"rel err {rel_err(latent, reference[i]):.1e}"))
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")

            t0 = time.perf_counter()
            pool = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
            for t in pool:
                t.start()
            for t in pool:
                t.join()
            seconds = time.perf_counter() - t0
            label = f"{threads} threads, {'locked' if locked else 'unlocked'}"
            rec.metric(f"{label}: seconds", seconds)
            if errors:
                rec.fail(f"{label}: {errors[0]}")
            elif bad:
                msg = f"{label}: {len(bad)} bad latents, e.g. image {bad[0][0]}: {bad[0][1]}"
                if locked:
                    rec.fail(msg)
                else:
                    rec.warn(msg + " -- concurrent encodes break on this GPU; keep the caching lock")
            else:
                rec.line(f"ok: {label}: all latents finite and matching")
