"""The main workload end to end, at a tiny scale: a Chroma model with random weights and a small masked dataset, built
offline (tiny_chroma.py), trained through the real scripts/train.py with the workload's settings (Chroma LoRA preset,
masked training, EMA, ADOPT_ADV, no warmup, sampling, backups). Each variant that changes how GPU memory is handled
(MemPools, disk streaming, layer offloading, resuming a backup) is compared with the baseline and with a repeat of it,
which measures the run-to-run noise; every run writes an OT_DIAG_LOG, whose memory numbers are reported. And the same
model's components (T5, VAE, transformer with a LoRA) on the GPU against the CPU, with the same inputs.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from modules.util.rocm_check.framework import Ctx, Rec, check, finite, output_tail, rel_err

import torch

SECTION = "5 training run (tiny Chroma)"
_SEED = "42"
_PRESET = os.path.join("training_presets", "Chroma", "#chroma LoRA 24GB.json")


def _repo_root(ctx: Ctx) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(ctx.script_path)))


def _base_config(ctx: Ctx, model: str, data: str, run_dir: str, epochs: int) -> dict:
    with open(os.path.join(_repo_root(ctx), _PRESET), encoding="utf-8") as f:
        config = json.load(f)
    gpu = ctx.device.type == "cuda"
    with open(os.path.join(run_dir, "concepts.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": "data", "path": data, "enabled": True, "seed": 1, "text": {"prompt_source": "sample"}}], f)
    with open(os.path.join(run_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump([{"prompt": "a red cat on a table", "height": 256, "width": 256, "seed": 7, "diffusion_steps": 4,
                    "cfg_scale": 1.0, "enabled": True}], f)
    config.update({
        "__version": 13, "base_model_name": model, "resolution": "256", "epochs": epochs,
        # the main workload: masked training, EMA, ADOPT_ADV, no warmup (the preset gives the rest: LoRA, bf16, batch 2)
        "masked_training": True, "ema": "GPU" if gpu else "CPU", "ema_update_step_interval": 1,
        "optimizer": {"optimizer": "ADOPT_ADV"}, "learning_rate_warmup_steps": 0,
        "train_device": str(ctx.device), "temp_device": "cpu",
        "sample_after": 1, "sample_after_unit": "EPOCH", "sample_image_format": "PNG",
        "backup_after": 1, "backup_after_unit": "EPOCH", "tensorboard": False, "tensorboard_always_on": False,
        "workspace_dir": os.path.join(run_dir, "workspace"), "cache_dir": os.path.join(run_dir, "cache"),
        "final_output_dir": os.path.join(run_dir, "output"), "run_name_mode": "CUSTOM", "run_name": "lora",
        "concept_file_name": os.path.join(run_dir, "concepts.json"),
        "sample_definition_file_name": os.path.join(run_dir, "samples.json"),
    })
    return config


def _train(ctx: Ctx, run_dir: str, config: dict, env: dict[str, str], values: list[str], log_name: str) -> dict:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    diag = os.path.join(run_dir, f"{log_name}.jsonl")
    cmd = [sys.executable, os.path.join(_repo_root(ctx), "scripts", "train.py"), "--config-path", config_path]
    for value in values:
        cmd += ["--config-value", value]
    run_env = {**os.environ, "OT_SEED": _SEED, **env, "OT_DIAG_LOG": diag, "OT_DIAG_LABEL": log_name}
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=_repo_root(ctx), env=run_env, capture_output=True, text=True, timeout=1800)
    with open(os.path.join(run_dir, f"{log_name}.log"), "w", encoding="utf-8") as f:
        f.write(result.stdout + result.stderr)
    return {"rc": result.returncode, "seconds": time.perf_counter() - start, "diag": diag,
            "tail": output_tail(result.stdout + result.stderr),
            "output": os.path.join(run_dir, "output", "lora.safetensors"),
            "samples": os.path.join(run_dir, "workspace", "samples")}


def _weights(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return {k: v.double() for k, v in load_file(path).items()}


def _weights_diff(a: dict, b: dict) -> float:
    if a.keys() != b.keys():
        return float("inf")
    return max(((a[k] - b[k]).norm() / a[k].norm().clamp_min(1e-30)).item() for k in a)


def _losses(diag: str) -> dict[int, float]:
    out = {}
    if os.path.exists(diag):
        with open(diag, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break
                if r["kind"] == "header":
                    out = {}  # a later run appended to the log: its losses only
                elif r["kind"] == "step":
                    out[r["step"]] = r["loss"]
    return out


def _loss_diff(a: dict, b: dict) -> tuple[float, int | None, int]:
    common = sorted(set(a) & set(b))
    worst = max((abs(a[s] - b[s]) / max(abs(a[s]), 1e-12) for s in common), default=0.0)
    first = next((s for s in common if a[s] != b[s]), None)
    return worst, first, len(common)


def _sample_images(path: str) -> tuple[int, int]:
    import numpy as np
    from PIL import Image

    images = [os.path.join(r, n) for r, _, ns in os.walk(path) for n in ns if n.endswith(".png")] \
        if os.path.isdir(path) else []
    blank = sum(1 for p in images if np.asarray(Image.open(p)).astype(np.float32).std() < 1.0)
    return len(images), blank


def _variants(ctx: Ctx) -> list[tuple[str, dict, list[str]]]:
    offload = ["transformer.offload_fraction=0.5", "transformer.activation_offloading=true"]
    if not ctx.is_gpu:
        return [("repeat", {}, [])]
    variants = [("repeat", {}, [])]
    if ctx.is_rocm:  # MemPools are always used on NVIDIA
        variants += [("MemPools", {"OT_ROCM_MEM_POOLS": "true"}, []),
                     ("MemPools + offloading", {"OT_ROCM_MEM_POOLS": "true"}, offload)]
    variants += [("streaming off", {}, ["stream_from_disk=false"]),
                 ("layer offloading 0.5 + activations", {}, offload)]
    if ctx.full:
        variants += [("cache in RAM off", {}, ["transformer.cache_in_ram=false", "text_encoder.cache_in_ram=false"]),
                     ("simplex offloading", {}, ["transformer.offload_fraction=0.5",
                                                 "transformer.simplex_offloading=true"]),
                     ("synchronous offloading", {}, [*offload, "async_offloading=false"]),
                     ("deterministic algorithms", {"OT_DETERMINISTIC": "true"}, [])]
    return variants


@check(SECTION, "training runs: tiny Chroma LoRA as the main workload (repeat, MemPools, streaming, offloading, "
                "resume)", timeout=5400.0,
       note="trains a tiny random Chroma through scripts/train.py once per variant: a few minutes")
def training_runs(ctx: Ctx, rec: Rec):
    from modules.util.diag_report import load, summarize
    from modules.util.rocm_check.tiny_chroma import build_dataset, build_tiny_chroma

    base_dir = tempfile.mkdtemp(prefix="rocm_check_training_", dir=ctx.out_dir if os.path.isdir(ctx.out_dir) else None)
    failed = False
    try:
        model = build_tiny_chroma(os.path.join(base_dir, "model"))
        data = build_dataset(os.path.join(base_dir, "data"), count=8, size=256)
        epochs = 3

        def run(name: str, env: dict, values: list[str], epochs: int = epochs, run_dir: str | None = None,
                log_name: str = "diag") -> dict:
            run_dir = run_dir or os.path.join(base_dir, name.replace(" ", "_").replace("+", "and"))
            os.makedirs(run_dir, exist_ok=True)
            config = _base_config(ctx, model, data, run_dir, epochs)
            config["lora_model_name"] = initial_lora
            return _train(ctx, run_dir, config, env, values, log_name)

        # every run starts from the same LoRA: a LoRA layer is initialized on the device its base layer is on, so with
        # layer offloading the offloaded layers draw from the CPU's generator instead of the GPU's, and the runs would
        # start apart even with OT_SEED. A run at learning rate 0 saves the start (one epoch, no samples or backups).
        initial_lora = ""
        init_dir = os.path.join(base_dir, "initial_lora")
        os.makedirs(init_dir, exist_ok=True)
        init = _train(ctx, init_dir, _base_config(ctx, model, data, init_dir, 1), {},
                      ["learning_rate=0", "sample_after_unit=NEVER", "backup_after_unit=NEVER",
                       "backup_before_save=false"], "diag")
        if init["rc"] != 0 or not os.path.exists(init["output"]):
            failed = True
            rec.fail(f"the run that makes the initial LoRA failed (exit code {init['rc']}): "
                     + init["tail"])
            return
        initial_lora = init["output"]

        results = {"baseline": run("baseline", {}, [])}
        for name, env, values in _variants(ctx):
            results[name] = run(name, env, values)
        # resume: stop after 2 of the 3 epochs (the backup at the end), then continue from that backup
        resume_dir = os.path.join(base_dir, "resume")
        first = run("resume", {}, [], epochs=epochs - 1, run_dir=resume_dir, log_name="first")
        results["resumed from a backup"] = \
            run("resume", {}, ["continue_last_backup=true"], run_dir=resume_dir, log_name="diag") \
            if first["rc"] == 0 else first

        # every run: finished, finite weights, sample images
        weights = {}
        for name, r in results.items():
            if r["rc"] != 0 or not os.path.exists(r["output"]):
                failed = True
                rec.fail(f"{name}: training failed (exit code {r['rc']}): " + r["tail"])
                continue
            w = _weights(r["output"])
            weights[name] = w
            ok = all(finite(v) for v in w.values())
            count, blank = _sample_images(r["samples"])
            rec.expect(ok and count > 0 and blank == 0,
                       f"{name}: finished in {r['seconds']:.0f} s, {len(w)} LoRA tensors "
                       f"{'finite' if ok else 'with NaN/inf'}, {count} sample images ({blank} blank)")
            # keep the runs for a look at them: missing or blank samples fail the check as NaN weights do
            failed |= not (ok and count > 0 and blank == 0)

        if "baseline" not in weights:
            return
        base_losses = _losses(results["baseline"]["diag"])
        floor = _weights_diff(weights["baseline"], weights["repeat"]) if "repeat" in weights else 0.0
        floor_loss = _loss_diff(base_losses, _losses(results["repeat"]["diag"]))[0] if "repeat" in weights else 0.0
        rec.line(f"run-to-run noise (baseline vs repeat): weights {floor:.1e}, losses {floor_loss:.1e}"
                 + (" (bit-identical)" if floor == 0 else ""))
        limit = max(10 * floor, 1e-6)
        for name, w in weights.items():
            if name in ("baseline", "repeat"):
                continue
            diff = _weights_diff(weights["baseline"], w)
            worst, first_step, steps = _loss_diff(base_losses, _losses(results[name]["diag"]))
            text = (f"{name} vs baseline: weights {diff:.1e}, losses {worst:.1e} over {steps} steps"
                    + (", bit-identical weights" if diff == 0 else
                       f", losses part at step {first_step}" if first_step is not None else ""))
            if diff <= limit:
                rec.line(f"ok: {text}")
            elif diff <= 1e-2:
                rec.warn(f"{text}: beyond 10x the run-to-run noise (limit {limit:.1e})")
            else:
                failed = True
                rec.fail(f"{text}: far beyond the run-to-run noise (limit {limit:.1e})")

        # the memory each variant used (tiny model: the allocator's behavior, not a 24 GB workload's numbers)
        for name, r in results.items():
            if not os.path.exists(r["diag"]):
                continue
            s = summarize(load(r["diag"]))
            if "peak reserved GiB" not in s:
                continue
            rec.line(f"{name}: peak reserved {s['peak reserved GiB'] * 1024:.0f} MiB, held unused at the end "
                     f"{(s['held unused at end GiB'] or 0) * 1024:.0f} MiB, alloc retries {s['alloc retries']}, "
                     f"{s['part moves']} part moves in {s['part move s']:.1f} s, step {s['step s median'] or 0:.3f} s")
            rec.metric(f"{name}: peak reserved MiB", s["peak reserved GiB"] * 1024)
            if s["step s median"]:
                rec.metric(f"{name}: step ms", s["step s median"] * 1000)
            rec.metric(f"{name}: run seconds", r["seconds"])
    finally:
        if failed:
            rec.line(f"the runs are kept in {base_dir} (each run's .log and .jsonl, compare with diag_report)")
        else:
            shutil.rmtree(base_dir, ignore_errors=True)


def _outputs(run_dir: str) -> dict[str, torch.Tensor]:
    # every tensor a run saved in its output folder: a LoRA or embedding file, or a diffusers folder
    from safetensors.torch import load_file
    out = os.path.join(run_dir, "output")
    tensors = {}
    for root, _, names in os.walk(out):
        for name in sorted(names):
            if name.endswith(".safetensors"):
                path = os.path.join(root, name)
                tensors.update({f"{os.path.relpath(path, out)}:{k}": v for k, v in load_file(path).items()})
    return tensors


_EMBEDDING = {"uuid": "6c1f4f6e-0d2b-4c55-9a0e-3b1d2f7c9e10", "model_name": "", "placeholder": "<emb>", "train": True,
              "token_count": 2, "initial_embedding_text": "cat", "is_output_embedding": False}


def _feature_variants() -> list[tuple[str, dict, list[str], dict]]:
    # (name, config changes, --config-value overrides, options). Options: "round_trip" loads the saved LoRA back at
    # learning rate 0 and expects the same tensors saved again; "dataset" (data, data_nomask, data_emb), "concepts"
    # (extra (name, type) concepts on the same images), "sample_prompt", "epochs", "expect" (text the log must contain)
    te_lora = ["text_encoder.train=true"]
    emb = {"dataset": "data_emb", "sample_prompt": "<emb> on a table"}
    return [
        ("PRODIGY_ADV", {"optimizer": {"optimizer": "PRODIGY_ADV"}, "learning_rate": 1.0}, [], {}),
        ("MUON_ADV", {"optimizer": {"optimizer": "MUON_ADV"}, "learning_rate": 1e-3}, [], {}),
        ("ADAMUON_ADV with the auxiliary Adam, DoRA",
         {"optimizer": {"optimizer": "ADAMUON_ADV", "MuonWithAuxAdam": True}, "lora_decompose": True,
          "learning_rate": 1e-3}, [], {}),
        ("COMFY_LORA with a text encoder LoRA", {"output_model_format": "COMFY_LORA"}, te_lora, {"round_trip": True}),
        ("KOHYA_LORA", {"output_model_format": "KOHYA_LORA"}, [], {"round_trip": True}),
        ("ORIGINAL_LORA", {"output_model_format": "ORIGINAL_LORA"}, [], {"round_trip": True}),
        ("LEGACY_LORA", {"output_model_format": "LEGACY_LORA"}, [], {"round_trip": True}),
        ("DoRA", {"lora_decompose": True}, [], {"round_trip": True}),
        ("LoHa", {"peft_type": "LOHA"}, [], {"round_trip": True}),
        ("LoKr with DoRA", {"peft_type": "LOKR", "lora_decompose": True, "lokr_dim": 4}, [], {"round_trip": True}),
        ("OFT with DoRA-OFT, CANS and spectral norm clipping",
         {"peft_type": "OFT_2", "dora_oft": True, "oft_cans": True, "oft_clipped_norm": 0.5, "oft_block_size": 8},
         [], {"round_trip": True}),
        ("LoRA with a bundled additional embedding", {"additional_embeddings": [_EMBEDDING]}, [],
         {**emb, "round_trip": True}),
        # adv_optm's OFT mode only on the OFT group: the embedding's group would fail in the first step with it
        ("SINKSGD_ADV on OFT with an additional embedding",
         {"peft_type": "OFT_2", "oft_block_size": 8, "optimizer": {"optimizer": "SINKSGD_ADV"},
          "additional_embeddings": [_EMBEDDING], "learning_rate": 1e-3}, [],
         {**emb, "round_trip": True, "expect": "skew-symmetric Sinkhorn"}),
        ("ADOPT_ADV with scaled weight decay",
         {"optimizer": {"optimizer": "ADOPT_ADV", "weight_decay": 0.1, "scaled_wd": True}}, [], {}),
        ("embedding training", {"training_method": "EMBEDDING", "output_model_format": "SAFETENSORS",
                                "embedding": _EMBEDDING, "learning_rate": 1e-2}, [], emb),
        ("fine-tune with an additional embedding", {"training_method": "FINE_TUNE", "output_model_format": "DIFFUSERS",
                                                    "additional_embeddings": [_EMBEDDING], "learning_rate": 1e-4},
         [], emb),
        ("MUON fine-tune with the auxiliary Adam",
         {"training_method": "FINE_TUNE", "output_model_format": "DIFFUSERS",
          "optimizer": {"optimizer": "MUON", "MuonWithAuxAdam": True}, "learning_rate": 1e-4}, [], {}),
        ("validation every step", {"validation": True, "validate_after": 1, "validate_after_unit": "STEP"}, [],
         {"concepts": [("val", "VALIDATION")]}),
        ("prior-prediction concept", {}, [], {"concepts": [("prior", "PRIOR_PREDICTION")]}),
        ("masked prior preservation with an image without a mask",
         {"masked_prior_preservation_weight": 1.0, "normalize_masked_area_loss": True, "unmasked_weight": 0.1}, [],
         {"dataset": "data_nomask", "expect": "has no mask"}),
        ("MAE, Huber and log-cosh losses, sigma weighting, no masking",
         {"masked_training": False, "mse_strength": 0.5, "mae_strength": 0.3, "huber_strength": 0.2,
          "log_cosh_strength": 0.1, "loss_weight_fn": "SIGMA"}, [], {}),
        ("CIOP, CEP, immiscible and offset noise, logit-normal timesteps",
         {"ciop_noise_weight": 0.1, "ciop_p": 0.5, "cep_gamma": 0.1, "k_noise_sampling": 4,
          "offset_noise_weight": 0.05, "timestep_distribution": "LOGIT_NORMAL"}, [], {}),
        ("batch 1, no aspect ratio bucketing, LoRA dropout",
         {"batch_size": 1, "aspect_ratio_bucketing": False, "dropout_probability": 0.1}, [], {}),
        ("gradient accumulation 2 with fused back pass and clipping",
         {"optimizer": {"optimizer": "ADOPT_ADV", "fused_back_pass": True}, "gradient_accumulation_steps": 2,
          "clip_grad_norm": 1.0}, [], {"epochs": 3}),
        ("cosine with restarts, warmup over 30% of the steps",
         {"learning_rate_scheduler": "COSINE_WITH_RESTARTS", "learning_rate_warmup_steps": 0.3,
          "learning_rate_cycles": 2}, [], {"epochs": 3}),
        ("a save every step with EMA", {"save_every": 1, "save_every_unit": "STEP", "save_skip_first": 0}, [], {}),
        ("text encoder LoRA stopped after an epoch", {"output_model_format": "COMFY_LORA"},
         [*te_lora, "text_encoder.stop_training_after=1", "text_encoder.stop_training_after_unit=EPOCH"],
         {"epochs": 3}),
        ("float8 transformer weights", {}, ["transformer.weight_dtype=FLOAT_8"], {}),
    ]


def _feature_run(ctx: Ctx, model: str, data: str, run_dir: str, changes: dict, values: list[str], options: dict,
                 epochs: int, extra: dict | None = None) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    config = _base_config(ctx, model, data, run_dir, epochs)
    concepts = [{"name": "data", "path": data, "enabled": True, "seed": 1, "text": {"prompt_source": "sample"}}]
    concepts += [{"name": name, "path": data, "enabled": True, "seed": 2 + i, "type": kind,
                  "text": {"prompt_source": "sample"}} for i, (name, kind) in enumerate(options.get("concepts", []))]
    with open(config["concept_file_name"], "w", encoding="utf-8") as f:
        json.dump(concepts, f)
    if "sample_prompt" in options:
        with open(config["sample_definition_file_name"], "w", encoding="utf-8") as f:
            json.dump([{"prompt": options["sample_prompt"], "height": 256, "width": 256, "seed": 7,
                        "diffusion_steps": 4, "cfg_scale": 1.0, "enabled": True}], f)
    config.update(json.loads(json.dumps(changes)))  # a copy: the dicts are shared with the variant list
    config.update(extra or {})
    result = _train(ctx, run_dir, config, {}, values, "diag")
    with open(os.path.join(run_dir, "diag.log"), encoding="utf-8", errors="replace") as f:
        result["log"] = f.read()
    result["tensors"] = _outputs(run_dir) if result["rc"] == 0 else {}
    return result


@check(SECTION, "training features: optimizers, LoRA formats and adapter types loaded back, embeddings, fine-tuning, "
                "validation, masking, losses and noise options (tiny Chroma)", full_only=True, timeout=5400.0,
       note="trains the tiny Chroma about 35 times through scripts/train.py: 10-15 minutes")
def training_features(ctx: Ctx, rec: Rec):
    from modules.util.rocm_check.tiny_chroma import build_dataset, build_tiny_chroma

    base_dir = tempfile.mkdtemp(prefix="rocm_check_features_", dir=ctx.out_dir if os.path.isdir(ctx.out_dir) else None)
    failed = False
    try:
        model = build_tiny_chroma(os.path.join(base_dir, "model"))
        datasets = {"data": build_dataset(os.path.join(base_dir, "data"))}
        # one image without its mask file, and captions that use the embedding's placeholder
        datasets["data_nomask"] = os.path.join(base_dir, "data_nomask")
        shutil.copytree(datasets["data"], datasets["data_nomask"])
        os.remove(os.path.join(datasets["data_nomask"], "img0-masklabel.png"))
        datasets["data_emb"] = os.path.join(base_dir, "data_emb")
        shutil.copytree(datasets["data"], datasets["data_emb"])
        for name in os.listdir(datasets["data_emb"]):
            if name.endswith(".txt"):
                path = os.path.join(datasets["data_emb"], name)
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                with open(path, "w", encoding="utf-8") as f:
                    f.write("<emb> " + text)

        for i, (name, changes, values, options) in enumerate(_feature_variants()):
            run_dir = os.path.join(base_dir, f"{i:02d}")
            data = datasets[options.get("dataset", "data")]
            result = _feature_run(ctx, model, data, run_dir, changes, values, options, options.get("epochs", 2))
            tensors = result["tensors"]
            if result["rc"] != 0 or not tensors:
                failed = True
                rec.fail(f"{name}: training failed (exit code {result['rc']}): " + result["tail"])
                continue
            if not all(finite(t) for t in tensors.values()):
                failed = True
                rec.fail(f"{name}: NaN/inf in the saved tensors")
                continue
            if options.get("expect") and options["expect"] not in result["log"]:
                failed = True
                rec.fail(f"{name}: the log doesn't say '{options['expect']}'")
                continue
            text = f"{name}: {result['seconds']:.0f} s, {len(tensors)} saved tensors, all finite"
            if options.get("round_trip"):
                # the saved file as the starting LoRA, nothing trained (learning rate 0, no EMA): the same tensors
                # have to come out again, so every saved layer was found and loaded into its place
                again = _feature_run(ctx, model, data, os.path.join(run_dir, "reload"), changes, values, options, 1, {
                    "lora_model_name": result["output"], "learning_rate": 0.0, "embedding_learning_rate": 0.0,
                    "ema": "OFF", "sample_after_unit": "NEVER", "backup_after_unit": "NEVER",
                    "backup_before_save": False})
                if again["rc"] != 0 or not again["tensors"]:
                    failed = True
                    rec.fail(f"{name}: training from the saved LoRA failed (exit code {again['rc']}): "
                             + again["tail"])
                    continue
                common = tensors.keys() & again["tensors"].keys()
                worst = max(((tensors[k].float() - again["tensors"][k].float()).abs().max().item() for k in common),
                            default=0.0)
                if tensors.keys() != again["tensors"].keys() or worst != 0:
                    failed = True
                    rec.fail(f"{name}: loaded back and saved again, the file differs ("
                             f"{len(tensors.keys() ^ again['tensors'].keys())} keys in only one of them, max abs "
                             f"diff {worst:.3g})")
                    continue
                text += ", loaded back and saved again bit-exactly"
            rec.line("ok: " + text)
    finally:
        if failed:
            rec.line(f"the runs are kept in {base_dir} (each run's diag.log and diag.jsonl)")
        else:
            shutil.rmtree(base_dir, ignore_errors=True)


@check(SECTION, "tiny Chroma components (T5, VAE, transformer + LoRA) on the GPU vs the CPU", gpu_only=True)
def components(ctx: Ctx, rec: Rec):
    """the model loaded through the real ChromaModelLoader on both devices, the same inputs: the text encoder through
    ChromaModel.encode_text (padding mask, cut to the longest caption), the VAE both ways, and the transformer called as
    BaseChromaSetup.predict calls it, with a LoRA on the workload's layer filter, forward and backward. fp32 on the GPU
    must match the CPU closely; bf16 weights under bf16 autocast (the workload) within bf16 precision"""
    from random import Random

    from modules.model.ChromaModel import ChromaModel
    from modules.modelLoader.chroma.ChromaModelLoader import ChromaModelLoader
    from modules.module.LoRAModule import LoRAModuleWrapper
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ModelType import ModelType
    from modules.util.ModelNames import ModelNames
    from modules.util.rocm_check.tiny_chroma import build_tiny_chroma

    with tempfile.TemporaryDirectory(dir=ctx.out_dir if os.path.isdir(ctx.out_dir) else None) as tmp:
        path = build_tiny_chroma(os.path.join(tmp, "model"))
        with open(os.path.join(_repo_root(ctx), _PRESET), encoding="utf-8") as f:
            preset = json.load(f)

        def load(dtype: DataType) -> tuple[ChromaModel, TrainConfig]:
            config = TrainConfig.default_values()
            config.model_type = ModelType.CHROMA_1
            config.transformer.weight_dtype = config.text_encoder.weight_dtype = dtype
            config.vae.weight_dtype = DataType.FLOAT_32
            config.train_dtype = dtype
            config.layer_filter = preset["layer_filter"]
            model = ChromaModel(ModelType.CHROMA_1)
            ChromaModelLoader().load(model, ModelType.CHROMA_1, ModelNames(base_model=path), config.weight_dtypes(),
                                     config.quantization, stream_from_disk=False)
            return model, config

        prompts = ["a red cat on a table", "photo of a small house near a river under a dark night sky with a moon"]
        g = torch.Generator().manual_seed(ctx.seed)
        image = torch.rand(2, 3, 256, 256, generator=g) * 2 - 1
        latents = torch.randn(2, 16, 32, 32, generator=g)
        timestep = torch.tensor([300.0, 800.0])
        target = torch.randn(2, 32 * 32 // 4, 64, generator=g)

        def run(model: ChromaModel, config: TrainConfig, device: torch.device, autocast: bool, lora_state=None):
            model.text_encoder.to(device)
            model.vae.to(device)
            model.transformer.to(device)
            lora = LoRAModuleWrapper(model.transformer, "transformer", config, config.layer_filter.split(","))
            if lora_state is None:
                gl = torch.Generator().manual_seed(ctx.seed + 1)
                with torch.no_grad():
                    for p in lora.parameters():
                        p.copy_(torch.randn(p.shape, generator=gl) * 0.05)
            else:
                lora.load_state_dict(lora_state)
            lora.to(device)
            lora.hook_to_module()
            dtype = config.train_dtype.torch_dtype()
            ac = torch.autocast(device.type, dtype=dtype, enabled=autocast)
            out = {}
            with torch.no_grad(), ac:
                text, mask = model.encode_text(device, batch_size=2, rand=Random(0), text=prompts)
                out["text"] = text.float().cpu()
                out["mask"] = mask.cpu()
                out["vae latent"] = model.vae.encode(image.to(device)).latent_dist.mean.float().cpu()
                out["vae image"] = model.vae.decode(latents.to(device)).sample.float().cpu()
            # the transformer gets the CPU's text embedding, so its comparison doesn't carry the text encoder's error
            text_in = out_ref["text"].to(device) if out_ref else text
            mask_in = out_ref["mask"].to(device) if out_ref else mask
            packed = model.pack_latents(latents.to(device))
            image_ids = model.prepare_latent_image_ids(32, 32, device, dtype)
            text_ids = torch.zeros(text_in.shape[1], 3, device=device)
            attention_mask = torch.cat([mask_in, torch.ones(2, packed.shape[1], dtype=torch.bool, device=device)], 1)
            with ac:
                pred = model.transformer(
                    hidden_states=packed.to(dtype), timestep=timestep.to(device) / 1000,
                    encoder_hidden_states=text_in.to(dtype), txt_ids=text_ids.to(dtype), img_ids=image_ids.to(dtype),
                    attention_mask=attention_mask, joint_attention_kwargs=None, return_dict=True).sample
                loss = torch.nn.functional.mse_loss(pred.float(), target.to(device))
            loss.backward()
            out["transformer out"] = pred.float().detach().cpu()
            out["LoRA grads"] = torch.cat([p.grad.float().flatten().cpu() for p in lora.parameters()])
            out["lora_state"] = {k: v.detach().cpu() for k, v in lora.state_dict().items()}
            return out

        out_ref = None
        cpu_model, cpu_config = load(DataType.FLOAT_32)
        out_ref = run(cpu_model, cpu_config, torch.device("cpu"), autocast=False)
        rec.expect(bool(out_ref["mask"][0].sum() < out_ref["mask"][1].sum()),
                   f"the two captions give different text lengths ({int(out_ref['mask'][0].sum())} and "
                   f"{int(out_ref['mask'][1].sum())} tokens of {out_ref['mask'].shape[1]}), so the attention is masked")
        for label, dtype, autocast, limit in (("fp32", DataType.FLOAT_32, False, 1e-4),
                                              ("bf16 weights + autocast", DataType.BFLOAT_16, True, 5e-2)):
            model, config = load(dtype)
            out = run(model, config, ctx.device, autocast, lora_state=out_ref["lora_state"])
            for key in ("text", "vae latent", "vae image", "transformer out", "LoRA grads"):
                err = rel_err(out[key], out_ref[key])
                rec.expect(finite(out[key]) and err < limit * (2 if key == "LoRA grads" else 1),
                           f"{label}: {key} rel err {err:.1e} (limit {limit * (2 if key == 'LoRA grads' else 1):.0e})")
            del model
