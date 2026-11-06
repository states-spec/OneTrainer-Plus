# Runs a real training config for a few optimizer steps and records what the optimizer sees: per step the data
# (masks, latents), the gradients and parameter movement per kind of adapter parameter, Prodigy's D statistics, whether
# anything outside the optimizer changed the trained weights between steps, and a gradient check (the loss change a
# step predicts from the gradient vs the change measured on the same deterministic batch). Nothing is saved.
import math
import os
import time
import types
from collections import defaultdict

from modules.module.FusedModule import FusedModuleGroup
from modules.util import create, path_util
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimeUnit import TimeUnit

import torch


def _kind(name: str) -> str:
    if name.endswith("lora_down.weight"):
        return "LoRA down (A)"
    if name.endswith("lora_up.weight"):
        return "LoRA up (B)"
    if name.endswith("dora_scale"):
        return "DoRA magnitude"
    return "other"


class _Probe:
    def __init__(self, trainer, config: TrainConfig, steps: int, check_every: int, log):
        self.trainer = trainer
        self.config = config
        self.steps = steps
        self.check_every = check_every
        self.log = log
        self.records = []
        self.last_batch = None
        self.batch_stats = []
        self.after_step = None  # CPU copies of the trained weights right after the previous step
        self.optimizer_steps = 0

    # ---------------------------------------------------------------- setup checks
    def static_checks(self):
        model = self.trainer.model
        optimizer = model.optimizer
        self.names = {}
        self.dora_checks = []
        hooked = total = dummies = fused = 0
        for wrapper in model.adapters():
            for prefix, module in wrapper.lora_modules.items():
                total += 1
                if isinstance(module, wrapper.dummy_klass):
                    dummies += 1
                    continue
                if isinstance(module, FusedModuleGroup):
                    # fused qkv (ORIGINAL/COMFY/KOHYA formats): one adapter on the concatenated leaves, each leaf's
                    # forward replaced by the group
                    hooked += bool(module.is_applied)
                    fused += 1
                    peft = module.module
                else:
                    hooked += bool(module.is_applied and module.orig_module.forward == module.forward)
                    peft = module
                for name, p in module.named_parameters():
                    self.names[id(p)] = f"{prefix}.{name}"
                if getattr(peft, "dora_scale", None) is not None and len(self.dora_checks) < 8:
                    self.dora_checks.append((prefix, peft))

        optimizer_params = [p for g in optimizer.param_groups for p in g["params"]]
        in_adapters = sum(id(p) in self.names for p in optimizer_params)
        trainable = [p for p in optimizer_params if p.requires_grad]
        self.log(f"optimizer: {type(optimizer).__name__}, {len(optimizer.param_groups)} group(s), "
                 f"{len(optimizer_params)} tensors ({len(trainable)} require grad), "
                 f"{in_adapters} of them are adapter weights the model's LoRA modules use")
        self.log(f"adapter modules: {total} ({dummies} dummies, {fused} fused qkv groups), {hooked} hooked into "
                 f"their layer")
        if in_adapters != len(optimizer_params):
            self.log("  WARNING: the optimizer holds tensors that are not the LoRA modules' weights")
        dtypes = defaultdict(int)
        for p in optimizer_params:
            dtypes[(str(p.dtype), str(p.device))] += 1
        self.log(f"  dtypes/devices: {dict(dtypes)}")
        for g in optimizer.param_groups:
            keys = ("lr", "initial_lr", "d", "d0", "d_coef", "growth_rate", "d_limiter", "slice_p", "use_atan2",
                    "betas", "beta3", "weight_decay", "cautious_wd", "prodigy_steps", "safeguard_warmup")
            self.log("  group " + ", ".join(f"{k}={g[k]}" for k in keys if k in g))

        restored = sum(1 for p in optimizer_params if optimizer.state.get(p))
        if restored:
            g0 = optimizer.param_groups[0]
            self.log(f"  RESTORED optimizer state (a backup): {restored} of {len(optimizer_params)} tensors have state "
                     f"before the first step; k={g0.get('k')}, d={g0.get('d')}, d_numerator={g0.get('d_numerator')}")
        # a DoRA magnitude starts as the norm of its base weight (a resumed LoRA has trained since): a wrong start would mean the adapters were built on
        # weights that weren't loaded yet
        from modules.util.quantization_util import get_unquantized_weight
        for prefix, module in self.dora_checks:
            w = get_unquantized_weight(module.orig_module, torch.float, module.dora_scale.device)
            if module.decompose_output_axis:
                norm = w.reshape(w.shape[0], -1).norm(dim=1)
            else:
                norm = w.transpose(1, 0).reshape(w.shape[1], -1).norm(dim=1)
            scale = module.dora_scale.detach().float().flatten()
            rel = ((scale - norm).norm() / norm.norm().clamp_min(1e-12)).item()
            self.log(f"  DoRA start check {prefix}: |magnitude - base weight norm| / |norm| = {rel:.2e} "
                     f"(0 on a fresh start; base weight std {w.std().item():.3e})")

    # ---------------------------------------------------------------- hooks
    def install(self):
        setup = self.trainer.model_setup
        optimizer = self.trainer.model.optimizer
        original_predict = setup.predict
        original_step = optimizer.step

        def predict(model, batch, config, train_progress, **kwargs):
            if not kwargs.get("deterministic", False) and torch.is_grad_enabled():
                self.last_batch = batch
                self.batch_stats.append(self._batch_stats(batch))
            return original_predict(model, batch, config, train_progress, **kwargs)

        # bound to the optimizer: the lr scheduler wraps optimizer.step and expects a method
        def step(_optimizer, *args, **kwargs):
            pre = self._before_step()
            check = self._gradient_check_start() if self._check_now() else None
            result = original_step(*args, **kwargs)
            self._after_step(pre, check)
            return result

        setup.predict = predict
        optimizer.step = types.MethodType(step, optimizer)

    def _check_now(self) -> bool:
        return self.check_every > 0 and self.last_batch is not None \
            and (self.optimizer_steps < 3 or self.optimizer_steps % self.check_every == 0)

    def _batch_stats(self, batch: dict) -> dict:
        stats = {}
        for key in ("latent_mask", "latent_image"):
            t = batch.get(key)
            if isinstance(t, torch.Tensor):
                t = t.float()
                per_sample = t.flatten(1)
                stats[key] = {
                    "mean": [round(v, 4) for v in per_sample.mean(1).tolist()],
                    "std": [round(v, 4) for v in per_sample.std(1).tolist()],
                    "finite": bool(torch.isfinite(t).all()),
                }
        for key in ("image_path", "prompt"):
            if key in batch:
                stats[key] = [str(v)[-80:] for v in batch[key]]
        return stats

    def _params(self):
        for g in self.trainer.model.optimizer.param_groups:
            for p in g["params"]:
                yield g, p

    def _before_step(self) -> dict:
        optimizer = self.trainer.model.optimizer
        per_kind = defaultdict(lambda: defaultdict(float))
        drift = 0.0
        drift_tensors = 0
        no_grad = 0
        for i, (g, p) in enumerate(self._params()):
            kind = _kind(self.names.get(id(p), ""))
            k = per_kind[kind]
            k["tensors"] += 1
            if self.after_step is not None:
                d = (p.detach().float().cpu() - self.after_step[i]).norm().item()
                if d > 0:
                    drift += d ** 2
                    drift_tensors += 1
            if p.grad is None:
                no_grad += 1
                continue
            grad = p.grad.detach().float()
            k["grad_sq"] += grad.pow(2).sum().item()
            state = optimizer.state.get(p, {})
            p0 = state.get("p0")
            if p0 is not None and "d" in g:
                slice_p = g.get("slice_p", 1)
                g_slice = grad.flatten()[::slice_p]
                p_slice = p.detach().flatten()[::slice_p].float()
                dot = torch.dot(g_slice, p0.float() - p_slice).item()
                dlr = g["d"] * g["lr"]
                k["prodigy_numerator"] += (g["d"] / g["d0"]) * dlr * dot
                k["dot_g_p0_minus_p"] += dot
                k["moved_sq"] += (p0.float() - p_slice).pow(2).sum().item()
        return {"per_kind": per_kind, "drift": math.sqrt(drift), "drift_tensors": drift_tensors, "no_grad": no_grad,
                "before": [p.detach().to("cpu", torch.float32, copy=True) for _, p in self._params()]}

    def _deterministic_loss(self) -> float:
        trainer = self.trainer
        data = trainer.model_setup.predict(trainer.model, self.last_batch, self.config, trainer.model.train_progress,
                                           deterministic=True)
        return trainer.model_setup.calculate_loss(trainer.model, self.last_batch, data, self.config)

    def _gradient_check_start(self):
        # the training gradients stay untouched: they are set aside, a deterministic loss is backpropagated into
        # fresh ones, and the training ones are put back
        params = [p for _, p in self._params()]
        saved = [p.grad for p in params]
        for p in params:
            p.grad = None
        torch.clear_autocast_cache()
        with torch.enable_grad():
            loss = self._deterministic_loss()
            loss.backward()
        det_grads = [None if p.grad is None else p.grad.detach().to("cpu", torch.float32, copy=True) for p in params]
        for p, s in zip(params, saved, strict=True):
            p.grad = s
        torch.clear_autocast_cache()
        return {"loss": loss.item(), "grads": det_grads}

    def _after_step(self, pre: dict, check: dict | None):
        optimizer = self.trainer.model.optimizer
        self.optimizer_steps += 1
        after = [p.detach().to("cpu", torch.float32, copy=True) for _, p in self._params()]
        g0 = optimizer.param_groups[0]
        record = {
            "step": self.optimizer_steps,
            "d": g0.get("d"),
            "d_max": g0.get("d_max"),
            "d_numerator": g0.get("d_numerator"),
            "tensors_without_grad": pre["no_grad"],
            "weights_changed_outside_optimizer": {"norm": pre["drift"], "tensors": pre["drift_tensors"]},
            "per_kind": {},
        }
        update_sq = defaultdict(float)
        for (_, p), b, a in zip(self._params(), pre["before"], after, strict=True):
            update_sq[_kind(self.names.get(id(p), ""))] += (a - b).pow(2).sum().item()
        for kind, k in pre["per_kind"].items():
            record["per_kind"][kind] = {
                "tensors": int(k["tensors"]),
                "grad_norm": math.sqrt(k["grad_sq"]),
                "update_norm": math.sqrt(update_sq[kind]),
                "distance_from_start": math.sqrt(k["moved_sq"]),
                "dot(grad, start - weights)": k["dot_g_p0_minus_p"],
                "prodigy_numerator_term": k["prodigy_numerator"],
            }
        if check is not None:
            torch.clear_autocast_cache()
            with torch.no_grad():
                loss_after = self._deterministic_loss().item()
            predicted = sum((g * (a - b)).sum().item() for g, a, b in zip(check["grads"], after, pre["before"],
                                                                          strict=True) if g is not None)
            actual = loss_after - check["loss"]
            record["gradient_check"] = {"loss_before": check["loss"], "loss_after": loss_after,
                                        "predicted_change": predicted, "actual_change": actual,
                                        "ratio": actual / predicted if predicted != 0 else None}
        if self.batch_stats:
            record["batch"] = self.batch_stats[-1]
        self.after_step = after
        self.records.append(record)
        self._print(record)

    def _print(self, r: dict):
        d = r["d"]
        line = f"step {r['step']:4d}"
        if d is not None:
            line += f"  d={d:.4e}  d_numerator={r['d_numerator']:+.3e}"
        drift = r["weights_changed_outside_optimizer"]
        if drift["tensors"]:
            line += f"  CHANGED OUTSIDE THE OPTIMIZER: {drift['tensors']} tensors, norm {drift['norm']:.3e}"
        if r["tensors_without_grad"]:
            line += f"  {r['tensors_without_grad']} tensors without grad"
        self.log(line)
        for kind, k in sorted(r["per_kind"].items()):
            self.log(f"    {kind:15s} grad {k['grad_norm']:.3e}  update {k['update_norm']:.3e}  "
                     f"from start {k['distance_from_start']:.3e}  dot(g, start-w) {k['dot(grad, start - weights)']:+.3e}"
                     f"  numerator term {k['prodigy_numerator_term']:+.3e}")
        c = r.get("gradient_check")
        if c:
            ratio = "n/a" if c["ratio"] is None else f"{c['ratio']:+.3f}"
            self.log(f"    gradient check: loss {c['loss_before']:.6f} -> {c['loss_after']:.6f}, change "
                     f"{c['actual_change']:+.3e}, predicted from the gradient {c['predicted_change']:+.3e}, "
                     f"ratio {ratio} (near +1 = the gradient matches the loss; noisy while steps are tiny, as at d0)")
        b = r.get("batch", {})
        if "latent_mask" in b:
            self.log(f"    batch mask mean {b['latent_mask']['mean']}  latent std {b.get('latent_image', {}).get('std')}"
                     f"  finite {b.get('latent_image', {}).get('finite')}")


def run_probe(config: TrainConfig, steps: int, check_every: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    text_path = os.path.join(out_dir, "train_probe.txt")
    with open(text_path, "w", encoding="utf-8") as text:

        def log(line: str):
            print(line, flush=True)
            text.write(line + "\n")
            text.flush()

        log(f"train probe: {config.model_type}, {config.training_method}, {config.peft_type}, "
            f"optimizer {config.optimizer.optimizer}, {steps} optimizer steps, torch {torch.__version__}"
            + (f" (HIP {torch.version.hip})" if torch.version.hip else ""))
        if config.optimizer.fused_back_pass:
            log("note: fused back pass is on, so the gradients are gone by the optimizer step; the per-step gradient "
                "numbers and the gradient check need it off")

        # where a run can get trained weights and optimizer state from: the newest complete backup in the workspace
        # (Continue From Last Backup) or the base LoRA (a backup folder carries its optimizer state too)
        log(f"workspace: {os.path.abspath(config.workspace_dir)}, cache: {os.path.abspath(config.cache_dir)}")
        # (get_backup_paths is newer than some checkouts this probe is copied into to compare versions)
        backup_dir = os.path.join(config.workspace_dir, "backup")
        backups = config.get_backup_paths() if hasattr(config, "get_backup_paths") else sorted(
            (os.path.join(backup_dir, name) for name in os.listdir(backup_dir)
             if os.path.isdir(os.path.join(backup_dir, name))), reverse=True) if os.path.isdir(backup_dir) else []
        log(f"backups in the workspace before the run: {len(backups)}"
            + "".join(f"\n    {path}{'' if os.path.isfile(os.path.join(path, 'meta.json')) else ' (incomplete)'}"
                      for path in backups))
        selected = getattr(config, "continue_backup_path", "")
        if selected:
            log(f"continue from backup: {os.path.abspath(selected)} (selected; Continue From Last Backup doesn't apply)")
        else:
            log(f"continue from last backup: {config.continue_last_backup}"
                + (f" -> resumes {config.get_last_backup_path()}" if config.continue_last_backup and backups else ""))
        lora = config.lora_model_name
        if lora:
            is_backup = os.path.isfile(os.path.join(lora, "meta.json"))
            log(f"base LoRA: {os.path.abspath(lora)}" + (" (a backup folder: its optimizer state and progress load too)"
                                                         if is_backup else ""))
        else:
            log("base LoRA: none")

        # nothing goes into the workspace's backups or saves: a timed backup written during a probe was resumed by
        # the next run (Continue From Last Backup), which then started from this probe's state
        config.backup_after_unit = TimeUnit.NEVER
        config.save_every_unit = TimeUnit.NEVER
        config.backup_before_save = False

        callbacks = TrainCallbacks()
        commands = TrainCommands()
        trainer = create.create_trainer(config, callbacks, commands)
        probe = None

        def on_progress(train_progress, max_step, max_epoch):
            if probe is not None and probe.optimizer_steps >= steps:
                commands.stop()

        callbacks.set_on_update_train_progress(on_progress)
        started = time.time()
        try:
            trainer.start()
            probe = _Probe(trainer, config, steps, check_every, log)
            probe.static_checks()
            probe.install()
            trainer.train()
        finally:
            trainer._stop_tensorboard()
            if probe is not None:
                path_util.write_json_atomic(os.path.join(out_dir, "train_probe.json"), probe.records)
            log(f"done in {time.time() - started:.0f} s; nothing was saved. Report: {text_path}")
