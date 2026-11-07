"""MOSS-Image on the real model, for scripts/moss_check.py (the koratahiu-moss-implementation-experimental branch).

Section 1 loads the real weights into this process and compares the device (bf16, as training runs them) with float32
on the CPU, given the same inputs: the text encoder (only Qwen3.5's language model is loaded; fla's linear-attention
kernels against PyTorch's; a padded batch, as sampling encodes prompt and negative prompt, against single captions),
the FLUX.2 VAE (encode, decode, round trip, the 2x2 latent packing) and the transformer (two texts of different lengths
in one batch, so with the text mask, against each sample alone: the fork's batched forward on the real weights).

Section 2 trains LoRAs on the real model through scripts/train.py, all from one saved start, with a validation concept
and a sample (a prompt with a longer negative prompt: classifier-free guidance runs as a batch of two with the text
mask). Each variant is compared with the baseline and with a repeat of it (the run-to-run noise): layer + activation
offloading, disk streaming off, text encoder offloading, resuming a backup (Continue from last backup and Continue from
Backup), compiled transformer blocks, and the saved LoRA loaded back at learning rate 0. --full adds a short full
fine-tune (Adafactor, fused back pass, layer offloading).
"""
import contextlib
import copy
import gc
import json
import math
import os
import random
import re
import shutil
import time
import traceback

from modules.util.rocm_check.framework import Ctx, Rec, check, child_task, finite, rel_err, run_child
from modules.util.rocm_check.workload_checks import (
    _compile_stats,
    _loss_diff,
    _losses,
    _repo_root,
    _train,
    _weights,
    _weights_diff,
)

import torch

SECTION_COMPONENTS = "1 MOSS components (real weights)"
SECTION_TRAINING = "2 MOSS training (real model)"
DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Image-Base"
_PRESET = os.path.join("training_presets", "MOSS", "#moss LoRA.json")

_CAPTIONS = {
    "short": "a red apple on a wooden table",
    "medium": "A cozy reading nook by a rain-streaked window at dusk: a knitted blanket over a leather armchair, a "
              "stack of old books and a steaming mug of tea on a small side table, warm lamplight and soft shadows.",
    "long": "An overhead photograph of a busy farmers market stall on a bright autumn morning. Wooden crates overflow "
            "with red and green apples, orange pumpkins of every size, bundles of carrots with their leafy tops still "
            "attached, purple cabbages and baskets of dark blackberries. A hand-painted chalkboard sign in the corner "
            "lists prices in neat white letters. Burlap sacks of potatoes lean against the table legs, and a striped "
            "canvas awning casts a soft diagonal shadow across half of the produce. In the background, out of focus, "
            "shoppers in wool coats carry tote bags, and a child points at a pile of gourds. The light is warm and "
            "low, with long shadows and a slight haze, and the colors are rich and saturated. Shot on a 35mm lens, "
            "high detail, natural colors, no text other than the chalkboard.",
}
# the sample every training run makes: the base model's image of it is the step-0 sample (the start LoRA changes
# nothing yet), for a look at whether the prompt is followed
_SAMPLE_PROMPT = "a red apple on a wooden table next to a glass of water, soft window light, photograph"
_SAMPLE_NEGATIVE = "blurry, low quality, oversaturated, watermark, text, deformed, cartoon, painting, dark, noisy"
_WORDS = ["red", "green", "blue", "golden", "silver", "cat", "dog", "horse", "house", "tree", "car", "sky", "sun",
          "moon", "river", "stone", "bird", "flower", "table", "chair", "window", "mountain", "forest", "city",
          "street", "lamp", "night", "morning", "rain", "snow", "beach", "ocean", "boat", "bridge", "castle",
          "garden", "portrait", "woman", "man", "child", "old", "young", "smiling", "sitting", "standing", "walking",
          "wooden", "glass", "metal", "soft", "bright", "dark", "warm", "cold", "detailed", "photograph", "painting",
          "sketch", "close-up", "wide", "shot"]


def preflight(repo_root: str) -> tuple[str | None, str | None]:
    """(an error that stops the check, a note) about the Python running it: the training runs start scripts/train.py
    with this same Python, so one without OneTrainer's requirements (another venv or conda env than the one the
    launchers activate) would fail every run, after the component checks"""
    import sys

    try:
        import modules.util.config.TrainConfig  # noqa: F401
        import modules.util.create  # noqa: F401
    except ImportError as e:
        return (f"this Python can't run OneTrainer's training ({e}).\n  Python: {sys.executable}, torch "
                f"{torch.__version__}\n  The training runs start scripts/train.py with the same Python. Start the "
                f"check through the launcher, which activates OneTrainer's environment:\n"
                f"    ./run-cmd.sh moss_check <the same arguments>"), None
    requirements = "requirements-rocm.txt" if torch.version.hip else "requirements-cuda.txt" if torch.version.cuda \
        else "requirements-default.txt"
    with contextlib.suppress(OSError):
        with open(os.path.join(repo_root, requirements), encoding="utf-8") as f:
            pinned = next((m.group(1) for m in (re.match(r"torch==([^;\s#]+)", line) for line in f) if m), None)
        if pinned and pinned != torch.__version__:
            return None, (f"note: torch {torch.__version__} in {sys.executable}, {requirements} pins {pinned}: the "
                          f"results describe this build")
    return None, None


def _model(ctx: Ctx) -> str:
    return ctx.cache.get("moss_model") or DEFAULT_MODEL


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bf16(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16)


def _sync(ctx: Ctx):
    ctx.sync()


def _child_env(ctx: Ctx, **extra: str) -> dict[str, str]:
    # a child process gets the model and resolution through its environment (run_child passes device and out dir)
    return {"OT_MOSS_CHECK_MODEL": _model(ctx), "OT_MOSS_CHECK_RESOLUTION": str(ctx.cache.get("moss_resolution") or 512),
            **extra}


def _in_child(timeout: float = 3600.0):
    """runs the check in a process of its own: whatever GPU memory it used (the text encoder alone is 15 GiB in bf16)
    is the driver's again when it ends, not held by this process while the training runs need the whole GPU"""
    def wrap(body):
        task = f"moss_{body.__name__}"

        @child_task(task)
        def child(ctx: Ctx) -> dict:
            rec = Rec()
            try:
                body(ctx, rec)
            except torch.OutOfMemoryError as e:
                rec.fail(f"out of memory: {str(e).splitlines()[0]}")
            except Exception as e:
                rec.fail(f"{type(e).__name__}: {e}")
                rec.lines += ["  " + line for line in traceback.format_exc().strip().splitlines()[-8:]]
            return {"status": rec.status, "lines": rec.lines, "metrics": rec.metrics}

        def run(ctx: Ctx, rec: Rec):
            result = run_child(ctx, task, _child_env(ctx), timeout=timeout)
            if "error" in result:
                rec.fail(f"the check's process failed: {result['error']}")
                return
            rec.lines += result["lines"]
            rec.metrics.update(result["metrics"])
            rec._set(result["status"])

        run.__name__ = body.__name__
        return run
    return wrap


def _text_path(ctx: Ctx) -> str:
    # the text encoder check's float32 hidden states, for the transformer check in another process
    return os.path.join(ctx.out_dir, "moss_text_reference.pt")


# ---------------------------------------------------------------------------------------------------------------------
# section 1: components


class _LinearAttentionKernels:
    """switches every Gated DeltaNet layer of a Qwen3.5 model between the kernels transformers picked (fla's, when it
    is installed and a GPU is there) and PyTorch's own (the chunked loop and the gated RMS norm), keeping the weights"""

    def __init__(self, text_encoder: torch.nn.Module):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

        self.qwen = qwen
        self.layers = [m for m in text_encoder.modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]
        self.picked = {id(m): (m.norm, m.chunk_gated_delta_rule) for m in self.layers}
        self.torch_norm = {id(m): qwen.Qwen3_5RMSNormGated(m.head_v_dim, eps=m.layer_norm_epsilon) for m in self.layers}
        self.has_fla = any(rule is not qwen.torch_chunk_gated_delta_rule for _, rule in self.picked.values())

    def use(self, picked: bool):
        for m in self.layers:
            norm, rule = self.picked[id(m)] if picked else (self.torch_norm[id(m)], self.qwen.torch_chunk_gated_delta_rule)
            if norm is not m.norm:
                weight = m.norm.weight.detach()
                norm.weight.data = weight.clone()  # the device and dtype the model is in now
                m.norm = norm
            m.chunk_gated_delta_rule = rule


def _encode(model, device: torch.device, texts: list[str], autocast: bool) -> list[torch.Tensor]:
    """MossModel.encode_text as training and sampling call it; each text's hidden states over its real tokens"""
    model.text_encoder_autocast_context = _bf16(device) if autocast else contextlib.nullcontext()
    with torch.no_grad():
        out, mask = model.encode_text(train_device=device, text=texts)
    return [out[i][mask[i]].float().cpu() for i in range(len(texts))]


@check(SECTION_COMPONENTS, "text encoder: Qwen3.5's language model in bf16 on the device against float32 on the CPU, "
                           "fla against the PyTorch kernels, a padded batch against single captions",
       timeout=3600.0, note="loads the text encoder in float32 (~32 GB of RAM) and runs it on the CPU: a few minutes")
@_in_child()
def text_encoder(ctx: Ctx, rec: Rec):
    from modules.model.MossModel import MossModel
    from modules.modelLoader.moss.MossModelLoader import MossModelLoader
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ModelType import ModelType

    from transformers import Qwen2Tokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    path = _model(ctx)
    model = MossModel(ModelType.MOSS)
    model.tokenizer = Qwen2Tokenizer.from_pretrained(path, subfolder="tokenizer")
    start = time.perf_counter()
    te, _ = MossModelLoader()._load_text_encoder(Qwen3_5TextModel, DataType.FLOAT_32, DataType.FLOAT_32, path,
                                                 "text_encoder", stream_from_disk=False,
                                                 checkpoint_prefix="language_model.")
    te = te.to("cpu").eval()
    model.text_encoder = te
    params = sum(p.numel() for p in te.parameters())
    vision = [n for n, _ in te.named_parameters() if "visual" in n]
    rec.expect(not vision, f"loaded the language model only: {params / 1e9:.2f}B parameters, "
                           f"{len(vision)} vision-tower tensors, in {time.perf_counter() - start:.0f} s")
    kernels = _LinearAttentionKernels(te)
    rec.line(f"{len(kernels.layers)} Gated DeltaNet layers; transformers picked "
             f"{'fla' if kernels.has_fla else 'the PyTorch'} kernels")
    if not kernels.has_fla and ctx.is_gpu:
        rec.warn("fla-core isn't used: text caching runs the slow PyTorch loop (requirements-rocm.txt installs it)")

    names = list(_CAPTIONS)
    try:
        # the reference: float32 on the CPU with PyTorch's kernels, each caption alone
        kernels.use(picked=False)
        cpu = torch.device("cpu")
        start = time.perf_counter()
        ref = {n: _encode(model, cpu, [_CAPTIONS[n]], autocast=False)[0] for n in names}
        rec.line("float32 on the CPU: " + ", ".join(f"{n} {ref[n].shape[0]} tokens" for n in names)
                 + f" ({time.perf_counter() - start:.0f} s)")
        for n in names:
            if not finite(ref[n]):
                rec.fail(f"the float32 reference of the {n} caption has NaN/inf")
                return
        torch.save(ref, _text_path(ctx))
        # a padded batch (sampling encodes prompt and negative prompt together): the real tokens as when alone
        padded = _encode(model, cpu, [_CAPTIONS["short"], _CAPTIONS["long"]], autocast=False)
        err = max(rel_err(padded[0], ref["short"]), rel_err(padded[1], ref["long"]))
        rec.expect(err < 1e-4, f"float32 on the CPU, short + long caption in one padded batch vs each alone: {err:.1e}")

        te.to(ctx.device, torch.bfloat16)
        errors, padded_errors = {}, {}
        for label, picked in ([("fla kernels", True)] if kernels.has_fla else []) + [("PyTorch kernels", False)]:
            kernels.use(picked)
            out = {n: _encode(model, ctx.device, [_CAPTIONS[n]], autocast=True)[0] for n in names}
            _sync(ctx)
            start = time.perf_counter()
            for _ in range(3):
                _encode(model, ctx.device, [_CAPTIONS["long"]], autocast=True)
            _sync(ctx)
            ms = (time.perf_counter() - start) / 3 * 1000
            errors[label] = {n: rel_err(out[n], ref[n]) for n in names}
            text = (f"{label}, bf16 on {ctx.device}, vs the float32 reference: "
                    + ", ".join(f"{n} {errors[label][n]:.1e}" for n in names) + f"; the long caption in {ms:.0f} ms")
            worst = max(errors[label].values())
            if not all(finite(o) for o in out.values()) or worst > 0.1:
                rec.fail(f"{text}: far from the reference")
            elif worst > 0.05:
                rec.warn(f"{text}: further from the reference than bf16 rounding usually leaves")
            else:
                rec.line(f"ok: {text}")
            rec.metric(f"{label}: long caption ms", ms)
            # a padded batch, as sampling encodes the prompt and the negative prompt together
            padded = _encode(model, ctx.device, [_CAPTIONS["short"], _CAPTIONS["long"]], autocast=True)
            padded_errors[label] = ({"short": rel_err(padded[0], ref["short"]), "long": rel_err(padded[1], ref["long"])},
                                    max(rel_err(padded[0], out["short"]), rel_err(padded[1], out["long"])))
        if "fla kernels" in errors:
            fla, torch_errors = errors["fla kernels"], errors["PyTorch kernels"]
            worse = [n for n in names if fla[n] > 1.5 * torch_errors[n] + 2e-3]
            rec.expect(not worse, "fla's kernels as close to the reference as PyTorch's in bf16"
                       + (f" (not for: {', '.join(worse)})" if worse else ""))

        # the bf16 noise: the largest error of a caption encoded alone, with either kernel set. A padding that reached
        # the real tokens would be far beyond it (on the 7900 XTX, fla: 1.1-1.4e-2 alone, 2.4e-2 in the padded batch;
        # PyTorch's kernels 2.4e-2 alone for the long caption)
        noise = max(e for per_caption in errors.values() for e in per_caption.values())
        for label, (per_caption, vs_alone) in padded_errors.items():
            worst = max(per_caption.values())
            text = (f"{label}, short + long caption in one padded batch vs the float32 reference: short "
                    f"{per_caption['short']:.1e}, long {per_caption['long']:.1e} (vs each alone {vs_alone:.1e}; the "
                    f"largest error of a caption alone {noise:.1e})")
            if worst <= 1.5 * noise + 2e-3:
                rec.line(f"ok: {text}")
            elif worst <= 0.1:
                rec.warn(f"{text}: further from the reference than any caption alone")
            else:
                rec.fail(f"{text}: the padding changes the real tokens")
    finally:
        del te, model, kernels
        _free()


def _test_image(size: int) -> torch.Tensor:
    # smooth color fields with edges and a fine pattern: something to reconstruct, in [-1, 1]
    y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    r = 0.6 * torch.sin(6.0 * x + 2.0 * y) + 0.3 * ((x - 0.35) ** 2 + (y - 0.45) ** 2 < 0.06).float()
    g = 0.5 * torch.cos(5.0 * y - 3.0 * x) - 0.4 * ((x > 0.6) & (y > 0.55)).float()
    b = 0.4 * torch.sin(40.0 * x) * torch.sin(40.0 * y) + 0.3 * (y - 0.5)
    return torch.stack([r, g, b]).clamp(-1, 1)[None]


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.double().cpu() - b.double().cpu()) / 2).pow(2).mean().item()  # images in [-1, 1]
    return float("inf") if mse == 0 else -10.0 * torch.log10(torch.tensor(mse)).item()


@check(SECTION_COMPONENTS, "VAE: FLUX.2 VAE encode and decode on the device against the CPU, round trip, latent "
                           "packing", timeout=1800.0)
@_in_child()
def vae(ctx: Ctx, rec: Rec):
    from modules.model.MossModel import MossModel
    from modules.util.enum.ModelType import ModelType

    from diffusers import AutoencoderKLFlux2

    vae_model = AutoencoderKLFlux2.from_pretrained(_model(ctx), subfolder="vae", torch_dtype=torch.float32).eval()
    size = ctx.cache.get("moss_resolution") or 512
    image = _test_image(size)
    try:
        with torch.no_grad():
            cpu_latent = vae_model.encode(image).latent_dist.mode()
            cpu_decoded = vae_model.decode(cpu_latent).sample
            vae_model.to(ctx.device)
            latent = vae_model.encode(image.to(ctx.device)).latent_dist.mode()
            decoded = vae_model.decode(latent).sample
            with _bf16(ctx.device):  # caching encodes under the training autocast (bf16) with the float32 VAE
                bf16_latent = vae_model.encode(image.to(ctx.device)).latent_dist.mode().float()
        ok = finite(latent) and finite(decoded) and finite(bf16_latent)
        enc, dec = rel_err(latent, cpu_latent), rel_err(decoded, cpu_decoded)
        rec.expect(ok and enc < 1e-3 and dec < 1e-3,
                   f"float32 on {ctx.device} vs the CPU at {size} px: latent {enc:.1e}, decoded image {dec:.1e}")
        bf16_err = rel_err(bf16_latent, cpu_latent)
        text = f"latent encoded under bf16 autocast (as caching does) vs float32 on the CPU: {bf16_err:.1e}"
        if bf16_err < 5e-2:
            rec.line(f"ok: {text}")
        else:
            rec.warn(f"{text}: more than bf16 rounding usually leaves")
        psnr = _psnr(decoded, image)
        text = f"round trip (encode, decode) PSNR {psnr:.1f} dB (CPU: {_psnr(cpu_decoded, image):.1f} dB)"
        if psnr >= 25:
            rec.line(f"ok: {text}")
        else:
            rec.warn(f"{text}: low for this VAE")

        model = MossModel(ModelType.MOSS)
        model.vae = vae_model
        packed = model.scale_latents(latent)
        back = model.unscale_latents(packed)
        rec.expect(tuple(packed.shape[1:]) == (latent.shape[1] * 4, latent.shape[2] // 2, latent.shape[3] // 2)
                   and rel_err(back, latent) < 1e-6,
                   f"latent packing: {tuple(latent.shape[1:])} -> {tuple(packed.shape[1:])}, unpacked back "
                   f"{rel_err(back, latent):.1e}; packed mean {packed.mean().item():.2f}, std {packed.std().item():.2f}")

    finally:
        del vae_model
        _free()


@check(SECTION_COMPONENTS, "transformer: two texts of different lengths in one batch (the text mask) against each "
                           "sample alone, bf16 on the device against float32 on the CPU",
       timeout=3600.0, note="loads the transformer in float32 (~20 GB of RAM) and runs it on the CPU: a few minutes")
@_in_child()
def transformer(ctx: Ctx, rec: Rec):
    from modules.model.moss.pipeline_moss_image import MOSSImageTransformer

    start = time.perf_counter()
    model = MOSSImageTransformer.from_pretrained(_model(ctx), subfolder="transformer",
                                                 torch_dtype=torch.float32).eval()
    model.set_attention_backend("native")  # torch SDPA, as OneTrainer's SDP setting
    rec.line(f"loaded {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters, "
             f"{len(model.layers)} blocks, in {time.perf_counter() - start:.0f} s")
    try:
        text = torch.load(_text_path(ctx), weights_only=True) if os.path.exists(_text_path(ctx)) else None
        if text is None:
            rec.line("the text encoder check didn't run: random text embeddings")
            gen = ctx.generator(2)
            text = {"short": torch.randn(40, model.config.text_dim, generator=gen),
                    "medium": torch.randn(70, model.config.text_dim, generator=gen)}
        a, b = text["short"], text["medium"]
        # 256 px (16x16 latent tokens) keeps the float32 run on the CPU short; the padding is what is tested
        side = max(4, (ctx.cache.get("moss_resolution") or 512) // 32)
        latents = torch.randn(2, model.config.latent_channels, side, side, generator=ctx.generator(3))
        timesteps = torch.tensor([300, 750])
        context = torch.nn.utils.rnn.pad_sequence([a, b], batch_first=True)
        mask = torch.zeros(context.shape[:2], dtype=torch.bool)
        mask[0, :a.shape[0]] = True
        mask[1, :b.shape[0]] = True

        def run(device: torch.device, dtype: torch.dtype) -> tuple[list[torch.Tensor], torch.Tensor]:
            cast = {"device": device, "dtype": dtype}
            autocast = _bf16(device) if dtype == torch.bfloat16 else contextlib.nullcontext()
            with torch.no_grad(), autocast:
                alone = [model(latents[i:i + 1].to(**cast), timesteps[i:i + 1].to(device), t[None].to(**cast),
                               return_dict=False)[0].float().cpu() for i, t in enumerate((a, b))]
                batch = model(latents.to(**cast), timesteps.to(device), context.to(**cast),
                              encoder_attention_mask=mask.to(device), return_dict=False)[0].float().cpu()
            return alone, batch

        start = time.perf_counter()
        ref, ref_batch = run(torch.device("cpu"), torch.float32)
        cpu_err = max(rel_err(ref_batch[i], ref[i][0]) for i in range(2))
        rec.expect(finite(ref_batch) and cpu_err < 1e-4,
                   f"float32 on the CPU ({a.shape[0]} and {b.shape[0]} text tokens, {side}x{side} latent): the batch "
                   f"with the text mask vs each sample alone {cpu_err:.1e} ({time.perf_counter() - start:.0f} s)")

        model.to(ctx.device, torch.bfloat16)
        alone, batch = run(ctx.device, torch.bfloat16)
        alone_err = max(rel_err(alone[i][0], ref[i][0]) for i in range(2))
        batch_err = max(rel_err(batch[i], ref[i][0]) for i in range(2))
        text = (f"bf16 on {ctx.device} vs float32 on the CPU: each sample alone {alone_err:.1e}, in the masked batch "
                f"{batch_err:.1e}")
        if not finite(batch) or alone_err > 0.1:
            rec.fail(f"{text}: far from the reference")
        elif batch_err <= 1.5 * alone_err + 2e-3:
            rec.line(f"ok: {text}")
        elif batch_err <= 3 * alone_err + 1e-2:
            rec.warn(f"{text}: the batch is further from the reference than single samples")
        else:
            rec.fail(f"{text}: the masked batch is wrong")

        if ctx.is_gpu:
            size = 64  # a 1024 px image
            big = torch.randn(2, model.config.latent_channels, size, size, generator=ctx.generator(4))
            args = (big.to(ctx.device, torch.bfloat16), timesteps.to(ctx.device), context.to(ctx.device, torch.bfloat16))
            torch.cuda.reset_peak_memory_stats(ctx.device)
            with torch.no_grad(), _bf16(ctx.device):
                ms = ctx.bench(lambda: model(*args, encoder_attention_mask=mask.to(ctx.device), return_dict=False),
                               iters=3, warmup=1)
            rec.line(f"info: forward pass of a batch of 2 at 1024 px (masked text) {ms:.0f} ms, peak allocated "
                     f"{torch.cuda.max_memory_allocated(ctx.device) / 2 ** 30:.1f} GiB")
            rec.metric("1024 px batch-2 forward ms", ms)
    finally:
        model = None  # noqa: F841 (frees the weights before the next check)
        _free()


def _miopen_dirs() -> dict[str, str]:
    # where MIOpen keeps what it found (the user find-db) and the kernels it compiled, unless told otherwise
    home = os.path.expanduser("~")
    return {"find-db": os.environ.get("MIOPEN_USER_DB_PATH") or os.path.join(home, ".config", "miopen"),
            "kernel cache": os.environ.get("MIOPEN_CUSTOM_CACHE_DIR") or os.path.join(home, ".cache", "miopen")}


def _dir_stats(path: str) -> list | None:
    if not os.path.isdir(path):
        return None
    files = [os.path.join(root, name) for root, _, names in os.walk(path) for name in names]
    return [len(files), sum(os.path.getsize(f) for f in files if os.path.isfile(f))]


@child_task("moss_vae_decode")
def _vae_decode(ctx: Ctx) -> dict:
    # one fresh process: the FLUX.2 VAE decodes a 1024 px latent twice
    from diffusers import AutoencoderKLFlux2

    if os.environ.get("OT_MOSS_CHECK_MIOPEN_IMMEDIATE") == "1":
        torch.backends.miopen.immediate = True
    dirs = _miopen_dirs()
    before = {name: _dir_stats(path) for name, path in dirs.items()}
    vae_model = AutoencoderKLFlux2.from_pretrained(_model(ctx), subfolder="vae", torch_dtype=torch.float32).eval()
    vae_model.to(ctx.device)
    latent = torch.randn(1, vae_model.config.latent_channels, 128, 128, generator=torch.Generator().manual_seed(1))
    latent = latent.to(ctx.device)
    times = []
    with torch.no_grad():
        for _ in range(2):
            _sync(ctx)
            start = time.perf_counter()
            out = vae_model.decode(latent).sample
            _sync(ctx)
            times.append(time.perf_counter() - start)
    path = os.path.join(ctx.out_dir, f"moss_vae_decode_{os.environ.get('OT_MOSS_CHECK_LABEL', 'x')}.pt")
    torch.save(out.cpu(), path)
    return {"times": times, "path": path, "finite": finite(out), "dirs": dirs, "before": before,
            "after": {name: _dir_stats(p) for name, p in dirs.items()}}


@check(SECTION_COMPONENTS, "VAE decode at 1024 px in fresh processes: the first decode's cost with your MIOpen caches "
                           "(twice), with empty ones, MIOPEN_FIND_MODE=FAST and MIOpen's immediate mode",
       gpu_only=True, timeout=5400.0, note="five processes that decode twice each: up to ~12 minutes")
def vae_first_decode(ctx: Ctx, rec: Rec):
    # run A's first sample decode at 1024 px took 89 s, later ones 1.3 s; the VAE check's (a new process) 137 s. A
    # cost every process pays again, or one MIOpen keeps in its caches? And do its fast modes avoid it?
    fresh = os.path.join(ctx.out_dir, "miopen_empty")
    shutil.rmtree(fresh, ignore_errors=True)

    def empty(name: str) -> dict:
        return {"MIOPEN_USER_DB_PATH": os.path.join(fresh, name, "db"),
                "MIOPEN_CUSTOM_CACHE_DIR": os.path.join(fresh, name, "cache")}

    variants = [("your MIOpen caches", {}), ("your MIOpen caches, a second process", {}),
                ("empty MIOpen caches", empty("default")),
                ("empty caches, MIOPEN_FIND_MODE=FAST", {**empty("fast"), "MIOPEN_FIND_MODE": "FAST"}),
                ("empty caches, MIOpen immediate mode", {**empty("immediate"), "OT_MOSS_CHECK_MIOPEN_IMMEDIATE": "1"})]
    reference = None
    try:
        for i, (label, env) in enumerate(variants):
            print(f"    decoding: {label}", flush=True)
            result = run_child(ctx, "moss_vae_decode", _child_env(ctx, OT_MOSS_CHECK_LABEL=str(i), **env),
                               timeout=1800.0)
            if "error" in result:
                rec.fail(f"{label}: {result['error']}")
                continue
            image = torch.load(result["path"], weights_only=True)
            os.remove(result["path"])
            if reference is None:
                reference = image
            first, second = result["times"]
            caches = "; ".join(f"{name} {result['before'][name]} -> {result['after'][name]} (files, bytes)"
                               for name in result["dirs"])
            text = (f"{label}: first decode {first:.1f} s, second {second:.2f} s; the image vs the first process's "
                    f"{rel_err(image, reference):.1e}; {caches}")
            if not result["finite"] or rel_err(image, reference) > 1e-3:
                rec.fail(f"{text}: a different image")
            else:
                rec.line(f"ok: {text}")
            rec.metric(f"{label}: first decode s", first)
        rec.line(f"MIOpen's caches by default: {_miopen_dirs()}")
    finally:
        shutil.rmtree(fresh, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------
# section 2: training through scripts/train.py


def _build_dataset(path: str, count: int, size: int, seed: int):
    import numpy as np
    from PIL import Image, ImageDraw

    os.makedirs(path, exist_ok=True)
    rng = random.Random(seed)
    for i in range(count):
        base = np.asarray([rng.randint(30, 220) for _ in range(3)], dtype=np.float32)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = base + 50 * np.sin(xx / (9 + 2 * i))[..., None] + 50 * np.cos(yy / (7 + i))[..., None]
        img = Image.fromarray(image.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        draw.ellipse((size // 5, size // 4, 3 * size // 4, 4 * size // 5), fill=tuple(rng.randint(0, 255) for _ in range(3)))
        draw.rectangle((size // 2, size // 8, 7 * size // 8, size // 3), fill=tuple(rng.randint(0, 255) for _ in range(3)))
        img.save(os.path.join(path, f"img{i}.png"))
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rectangle((size // 10, size // 10, 9 * size // 10, 9 * size // 10), fill=255)
        mask.save(os.path.join(path, f"img{i}-masklabel.png"))
        # captions from 3 to ~70 words: a batch of two has texts of different lengths, so the text mask is used
        with open(os.path.join(path, f"img{i}.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(rng.choice(_WORDS) for _ in range(3 + (i * 67) % 68)))


def _default_settings(ctx: Ctx) -> dict:
    """the training settings when no --config is given: the MOSS LoRA preset with bf16 weights (the user's runs; the
    preset's INT_W8A8 transformer autotunes its Triton kernels for minutes first), DoRA, ADOPT_ADV, EMA, masks"""
    with open(os.path.join(_repo_root(ctx), _PRESET), encoding="utf-8") as f:
        config = json.load(f)
    config.update({
        "__version": 13,
        "transformer": {"train": True, "weight_dtype": "BFLOAT_16"},
        "text_encoder": {"train": False, "weight_dtype": "BFLOAT_16"},
        "vae": {"weight_dtype": "FLOAT_32"},
        "batch_size": 2, "learning_rate": 1e-4, "optimizer": {"optimizer": "ADOPT_ADV"},
        "lora_rank": 16, "lora_alpha": 16.0, "lora_decompose": True,
        "masked_training": True, "ema": "GPU" if ctx.is_gpu else "CPU", "ema_update_step_interval": 1,
        "learning_rate_warmup_steps": 0,
    })
    return config


def _settings(ctx: Ctx) -> dict:
    if "moss_settings" not in ctx.cache:
        user = ctx.cache.get("moss_config")
        if user:
            with open(user, encoding="utf-8") as f:
                ctx.cache["moss_settings"] = json.load(f)
        else:
            ctx.cache["moss_settings"] = _default_settings(ctx)
    return copy.deepcopy(ctx.cache["moss_settings"])


def _data(ctx: Ctx) -> tuple[str, str, int]:
    base = ctx.cache["moss_runs"]
    size = ctx.cache.get("moss_resolution") or 512
    data, val = os.path.join(base, "data"), os.path.join(base, "validation_data")
    if not os.path.isdir(data):
        _build_dataset(data, 8, size, seed=1)
        _build_dataset(val, 2, size, seed=2)
    return data, val, size


def _run_config(ctx: Ctx, run_dir: str, epochs: int, lora: str) -> dict:
    data, val, size = _data(ctx)
    config = _settings(ctx)
    with open(os.path.join(run_dir, "concepts.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": "train", "path": data, "enabled": True, "seed": 1, "text": {"prompt_source": "sample"}},
                   {"name": "validation", "path": val, "enabled": True, "seed": 2, "type": "VALIDATION",
                    "text": {"prompt_source": "sample"}}], f)
    with open(os.path.join(run_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump([{"prompt": _SAMPLE_PROMPT, "negative_prompt": _SAMPLE_NEGATIVE, "height": size, "width": size,
                    "seed": 7, "diffusion_steps": 20 if ctx.is_gpu else 3, "cfg_scale": 4.0, "enabled": True}], f)
    config.update({
        "model_type": "MOSS", "training_method": "LORA", "base_model_name": _model(ctx), "lora_model_name": lora,
        "resolution": str(size), "epochs": epochs, "compile": False, "multi_gpu": False, "only_cache": False,
        "train_device": str(ctx.device), "temp_device": "cpu", "output_dtype": "FLOAT_32",
        "validation": True, "validate_after": 1, "validate_after_unit": "EPOCH",
        "sample_after": 1, "sample_after_unit": "EPOCH", "sample_skip_first": 0, "sample_image_format": "PNG",
        "non_ema_sampling": True, "samples_to_tensorboard": False,
        # backups only where a run is resumed (each holds the LoRA, its EMA and the optimizer state)
        "backup_after": 1, "backup_after_unit": "NEVER", "backup_before_save": False,
        "save_every_unit": "NEVER", "continue_last_backup": False, "continue_backup_path": "",
        "tensorboard": False, "tensorboard_always_on": False, "clear_cache_before_training": True,
        "workspace_dir": os.path.join(run_dir, "workspace"), "cache_dir": os.path.join(run_dir, "cache"),
        "final_output_dir": os.path.join(run_dir, "output"), "run_name_mode": "CUSTOM", "run_name": "lora",
        "concept_file_name": os.path.join(run_dir, "concepts.json"),
        "sample_definition_file_name": os.path.join(run_dir, "samples.json"),
        "concepts": None, "samples": None,
    })
    for part in ("transformer", "text_encoder"):
        config.setdefault(part, {}).update({"offload_fraction": 0.0})
    config["text_encoder"]["dropout_probability"] = 0.0  # MOSS refuses caption dropout
    config.setdefault("__version", 13)
    return config


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _run(ctx: Ctx, name: str, values: list[str], env: dict | None = None, epochs: int = 2, lora: str | None = None,
         run_dir: str | None = None, log_name: str = "diag") -> dict:
    own_dir = run_dir is None
    run_dir = run_dir or os.path.join(ctx.cache["moss_runs"], _slug(name))

    def attempt() -> dict:
        os.makedirs(run_dir, exist_ok=True)
        config = _run_config(ctx, run_dir, epochs, ctx.cache.get("moss_initial_lora", "") if lora is None else lora)
        free = _wait_for_free_memory(ctx)
        print(f"    training: {name}" + (f" ({free[0]:.1f} of {free[1]:.1f} GiB free on the GPU)" if free else ""),
              flush=True)
        # OT_LOG_VRAM: every load and unload prints the GPU memory, also the device's free memory, into the run's log
        result = _train(ctx, run_dir, config, {**(env or {}), "OT_LOG_VRAM": "true"}, values, log_name)
        result["dir"] = run_dir
        result["free_before"] = free
        with open(os.path.join(run_dir, f"{log_name}.log"), encoding="utf-8", errors="replace") as f:
            result["log"] = f.read()
        return result

    result = attempt()
    if result["rc"] != 0 and own_dir and "OutOfMemoryError" in result["log"]:
        # other programs share the GPU, and their use changes while the check runs: one more try, from scratch
        first = result
        oom = os.path.join(ctx.cache["moss_runs"], f"{_slug(name)}_out_of_memory")
        shutil.rmtree(oom, ignore_errors=True)
        shutil.move(run_dir, oom)  # kept for a look: its log has the GPU memory at every load and unload
        time.sleep(10.0)
        result = attempt()
        result["first_attempt"] = {"free_before": first["free_before"], "tail": first["tail"], "dir": oom}
    return result


def _device_memory(ctx: Ctx) -> tuple[float, float] | None:
    """free and total GiB on the device, other processes' use included"""
    if not ctx.is_gpu:
        return None
    free, total = torch.cuda.mem_get_info(ctx.device)
    return free / 2 ** 30, total / 2 ** 30


def _wait_for_free_memory(ctx: Ctx, seconds: float = 30.0) -> tuple[float, float] | None:
    # the driver frees a finished run's GPU memory on its own time: wait up to 30 s for the free memory to come back to
    # what it was before the training runs (other programs on the desktop use the GPU too, so not for all of it)
    memory = _device_memory(ctx)
    target = ctx.cache.get("moss_free_at_start")
    deadline = time.monotonic() + seconds
    while memory and target and memory[0] < target - 1.0 and time.monotonic() < deadline:
        time.sleep(2.0)
        memory = _device_memory(ctx)
    return memory


def _samples(path: str) -> dict[str, "object"]:
    """sample images by folder and training progress (the file names start with the run's timestamp)"""
    import numpy as np
    from PIL import Image

    images = {}
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            for name in names:
                match = re.search(r"-training-sample-(\d+-\d+-\d+)\.png$", name)
                if match:
                    key = f"{os.path.basename(root)} @ {match.group(1)}"
                    images[key] = np.asarray(Image.open(os.path.join(root, name)).convert("RGB")).astype(np.float32)
    return images


def _validation_losses(diag: str) -> dict[int, dict[str, float]]:
    out = {}
    if os.path.exists(diag):
        with open(diag, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break
                if r["kind"] == "header":
                    out = {}
                elif r["kind"] == "validation":
                    out[r["step"]] = r["losses"]
    return out


def _summary(r: dict) -> dict:
    from modules.util.diag_report import load, summarize
    return summarize(load(r["diag"])) if os.path.exists(r["diag"]) else {}


def _finished(rec: Rec, name: str, r: dict, expect_samples: bool = True, expect_validation: bool = True):
    """the saved LoRA if the run finished with finite weights, sample images that aren't black or flat, and validation
    losses; else None (and a FAIL)"""
    if r["rc"] != 0 or not os.path.exists(r["output"]):
        free = r.get("free_before")
        rec.fail(f"{name}: training failed (exit code {r['rc']})"
                 + (f", {free[0]:.1f} of {free[1]:.1f} GiB of the GPU were free when it started" if free else "")
                 + (", also on a second try" if r.get("first_attempt") else "") + f": {r['tail']}")
        return None
    weights = _weights(r["output"])
    problems = []
    if not all(finite(v) for v in weights.values()):
        problems.append("NaN/inf in the saved LoRA")
    images = _samples(r["samples"])
    dark = [k for k, v in images.items() if v.mean() < 10]
    flat = [k for k, v in images.items() if v.std() < 2]
    if expect_samples and not images:
        problems.append("no sample images")
    if dark or flat:
        problems.append(f"{len(dark)} black and {len(flat)} flat sample images")
    validation = _validation_losses(r["diag"])
    bad_validation = [step for step, losses in validation.items() if not all(map(math.isfinite, losses.values()))]
    if expect_validation and not validation:
        problems.append("no validation loss (the validation concept wasn't evaluated)")
    if bad_validation:
        problems.append(f"NaN/inf validation loss at steps {bad_validation}")
    s = _summary(r)
    memory = (f", peak reserved {s['peak reserved GiB']:.1f} GiB" if "peak reserved GiB" in s else "") \
        + (f", step {s['step s median']:.2f} s" if s.get("step s median") else "")
    first = r.get("first_attempt")
    retry = ""
    if first:
        retry = ("; the first attempt ran out of GPU memory"
                 + (f" ({first['free_before'][0]:.1f} GiB were free when it started)" if first["free_before"] else "")
                 + f", its log in {first['dir']}")
    text = (f"{name}: finished in {r['seconds']:.0f} s, {len(weights)} tensors, {len(images)} sample images, "
            f"validation at steps {sorted(validation)}{memory}{retry}")
    if problems:
        rec.fail(f"{text}: {'; '.join(problems)}")
        return None
    rec.line(f"ok: {text}")
    return weights


def _compare(rec: Rec, name: str, base: dict, w_base: dict, r: dict, w: dict, floor: float) -> bool:
    """a variant against the baseline: weights, losses, validation losses and sample pixels; True if it matches"""
    diff = _weights_diff(w_base, w)
    worst, first_step, steps = _loss_diff(_losses(base["diag"]), _losses(r["diag"]))
    val_base, val = _validation_losses(base["diag"]), _validation_losses(r["diag"])
    common = sorted(set(val_base) & set(val))
    val_diff = max((abs(val_base[s][k] - val[s][k]) / max(abs(val_base[s][k]), 1e-12)
                    for s in common for k in val_base[s] if k in val[s]), default=0.0)
    images_base, images = _samples(base["samples"]), _samples(r["samples"])
    keys = sorted(set(images_base) & set(images))
    pixels = max((float(abs(images_base[k] - images[k]).max()) for k in keys), default=None)
    text = (f"{name} vs baseline: weights {diff:.1e}, losses {worst:.1e} over {steps} steps"
            + (f" (apart from step {first_step})" if first_step is not None else "")
            + f", validation {val_diff:.1e} at steps {common}"
            + (f", samples {len(keys)} compared, largest pixel difference {pixels:.0f}" if pixels is not None else
               ", no samples to compare"))
    limit = max(10 * floor, 1e-6)
    if diff == 0 and pixels:
        # the same weights must sample the same images (this is where offloading or a part move in sampling shows)
        rec.warn(f"{text}: the samples differ though the weights are bit-identical")
        return False
    if diff <= limit:
        rec.line(f"ok: {text}")
        return True
    if diff <= 1e-2:
        rec.warn(f"{text}: beyond 10x the run-to-run noise (limit {limit:.1e})")
    else:
        rec.fail(f"{text}: far beyond the run-to-run noise (limit {limit:.1e})")
    return False


@check(SECTION_TRAINING, "LoRA training runs: baseline, repeat, layer offloading, streaming off, text encoder "
                         "offloading, resumed from a backup", timeout=14400.0,
       note="trains the real model 8 times through scripts/train.py: about 20 minutes on a 7900 XTX")
def training_runs(ctx: Ctx, rec: Rec):
    memory = _device_memory(ctx)
    if memory:
        ctx.cache["moss_free_at_start"] = memory[0]
        rec.line(f"info: before the training runs {memory[0]:.1f} of {memory[1]:.1f} GiB of the GPU were free (this "
                 f"process: {torch.cuda.memory_reserved(ctx.device) / 2 ** 30:.1f} GiB; other programs use the rest)")
    # on the CPU, offloading moves between cpu and cpu:1; nothing streams from disk there, which Simplex needs
    on_cpu = [] if ctx.is_gpu else ["temp_device=cpu:1", "transformer.simplex_offloading=false",
                                    "text_encoder.simplex_offloading=false"]
    # every run starts from one saved LoRA (a run at learning rate 0): a LoRA layer is initialized on its base layer's
    # device, so with layer offloading the offloaded layers would draw from another generator
    init = _run(ctx, "start LoRA (learning rate 0)", ["learning_rate=0", "sample_after_unit=NEVER", "validation=false"],
                epochs=1, lora="")
    if init["rc"] != 0 or not os.path.exists(init["output"]):
        rec.fail(f"the run that saves the start LoRA failed (exit code {init['rc']}): {init['tail']}")
        ctx.cache["moss_failed"] = True
        return
    ctx.cache["moss_initial_lora"] = init["output"]
    settings = _settings(ctx)
    rec.line(f"settings: {'--config ' + ctx.cache['moss_config'] if ctx.cache.get('moss_config') else 'built in'}; "
             f"{settings.get('optimizer', {}).get('optimizer')}, lr {settings.get('learning_rate')}, "
             f"{settings.get('peft_type', 'LORA')} rank {settings.get('lora_rank')}"
             f"{' (DoRA)' if settings.get('lora_decompose') else ''}, batch {settings.get('batch_size')}, "
             f"{ctx.cache.get('moss_resolution') or 512} px, EMA {settings.get('ema')}")

    results = {"baseline": _run(ctx, "baseline", [])}
    results["repeat"] = _run(ctx, "repeat", [])
    simplex = ctx.is_gpu and settings.get("transformer", {}).get("simplex_offloading", False)
    results["layer offloading 0.5 + activations" + (" (simplex)" if simplex else "")] = _run(
        ctx, "layer offloading", ["transformer.offload_fraction=0.5", "transformer.activation_offloading=true",
                                  *on_cpu])
    results["disk streaming off"] = _run(ctx, "streaming off", ["stream_from_disk=false"])
    # caching with a partly offloaded text encoder needs a single data loader thread
    results["text encoder offloading 0.5"] = _run(
        ctx, "text encoder offloading", ["text_encoder.offload_fraction=0.5", "dataloader_threads=1", *on_cpu])
    # stop after the first epoch (a backup at the end), then continue from it: once with Continue from last backup,
    # once with Continue from Backup naming that folder (in another workspace)
    resume_dir = os.path.join(ctx.cache["moss_runs"], "resume")
    first = _run(ctx, "resume, first epoch", ["backup_before_save=true"], epochs=1, run_dir=resume_dir,
                 log_name="first")
    backups = os.path.join(resume_dir, "workspace", "backup")
    backup = os.path.join(backups, sorted(os.listdir(backups))[-1]) if os.path.isdir(backups) and os.listdir(backups) \
        else None
    if first["rc"] != 0 or backup is None:
        rec.fail(f"resume: the first epoch failed or left no backup (exit code {first['rc']}): {first['tail']}")
    else:
        results["resumed (Continue from last backup)"] = _run(ctx, "resume", ["continue_last_backup=true"],
                                                              run_dir=resume_dir)
        results["resumed (Continue from Backup)"] = _run(ctx, "continue from backup",
                                                         [f"continue_backup_path={backup}"])

    weights = {}
    for name, r in results.items():
        w = _finished(rec, name, r)
        if w is not None:
            weights[name] = w
    if "baseline" not in weights:
        ctx.cache["moss_failed"] = True
        return
    ctx.cache["moss_baseline"] = results["baseline"]
    images = _samples(results["baseline"]["samples"])
    for key in sorted(images):
        rec.line(f"baseline sample {key}: mean {images[key].mean():.0f}, std {images[key].std():.0f} (0-255)")

    floor = _weights_diff(weights["baseline"], weights["repeat"]) if "repeat" in weights else 0.0
    rec.line(f"run-to-run noise (baseline vs repeat): weights {floor:.1e}"
             + (" (bit-identical)" if floor == 0 else ""))
    matched = True
    for name, w in weights.items():
        if name != "baseline":
            matched &= _compare(rec, name, results["baseline"], weights["baseline"], results[name], w, floor)
    if not matched or rec.status == "FAIL":
        ctx.cache["moss_failed"] = True
    for name, r in results.items():
        s = _summary(r)
        if "peak reserved GiB" in s:
            rec.metric(f"{name}: peak reserved GiB", s["peak reserved GiB"])
        if s.get("step s median"):
            rec.metric(f"{name}: step s", s["step s median"])


def _nudged_lora(src: str, dst: str, relative: float, seed: int):
    """src with every value moved by `relative` of itself, up or down at random (zeros stay zero)"""
    from safetensors import safe_open
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    with safe_open(src, framework="pt") as f:
        metadata = f.metadata()
        for key in sorted(f.keys()):
            t = f.get_tensor(key)
            if t.is_floating_point():
                sign = torch.randint(0, 2, t.shape, generator=generator, dtype=torch.float64) * 2 - 1
                t = (t.double() * (1 + relative * sign)).to(t.dtype)
            tensors[key] = t
    save_file(tensors, dst, metadata=metadata)


def _versus_eager(base: dict, w_base: dict, start_base: dict, r: dict, w: dict, start: dict) -> dict:
    """a run that rounds differently against the eager baseline: the first step's loss (the same weights, so the forward
    pass alone), the losses, the last validation loss, the weights, and the change from each run's own start LoRA"""
    keys = sorted(w_base)
    eager = torch.cat([(w_base[k] - start_base[k]).flatten() for k in keys])
    other = torch.cat([(w[k] - start[k]).flatten() for k in keys])
    base_losses, losses = _losses(base["diag"]), _losses(r["diag"])
    rel = [abs(base_losses[s] - losses[s]) / max(abs(base_losses[s]), 1e-12)
           for s in sorted(base_losses.keys() & losses.keys())]
    val_base, val = _validation_losses(base["diag"]), _validation_losses(r["diag"])
    last = max(val_base.keys() & val.keys(), default=None)
    return {
        "first loss": rel[0] if rel else float("inf"),
        "losses": max(rel, default=float("inf")),
        "validation": max((abs(val_base[last][k] - val[last][k]) / max(abs(val_base[last][k]), 1e-12)
                           for k in val_base[last] if k in val[last]), default=float("nan"))
        if last is not None else float("nan"),
        "validation step": last,
        "weights": _weights_diff(w_base, w),
        "cosine": torch.nn.functional.cosine_similarity(eager, other, dim=0).item(),
        "size": (other.norm() / eager.norm().clamp_min(1e-30)).item(),
    }


def _versus_text(v: dict) -> str:
    return (f"first loss {v['first loss']:.1e}, losses up to {v['losses']:.1e}, validation {v['validation']:.1e} at "
            f"step {v['validation step']}, weights {v['weights']:.1e}; the change from its start LoRA points as "
            f"eager's does to cosine {v['cosine']:.3f}, its size {v['size']:.2f}x eager's")


@check(SECTION_TRAINING, "compiled training: Compile transformer blocks against the eager baseline, with an eager run "
                         "from a start LoRA moved by rounding as the yardstick", timeout=3600.0,
       note="two training runs: the eager yardstick and the compiled one")
def compiled_training(ctx: Ctx, rec: Rec):
    base = ctx.cache.get("moss_baseline")
    if base is None:
        rec.skip("needs the baseline of the training runs check")
        return
    # compiling changes the rounding (fused kernels keep float32 in between), so a compiled run can't match eager bit by
    # bit, and an adaptive optimizer turns rounding into another trajectory: Prodigy starts at D = 1e-6, its first
    # updates are about lr * D * sign(gradient) per value (Adam's first step), so a value whose gradient is near zero
    # moves the full step one way or the other, and rounding moves when D grows (the user's config on the 7900 XTX:
    # weights 1.5 apart after 8 steps, the change from the start pointing as eager's to cosine 0.67, the same size). So
    # the yardstick for the compiled run is how far rounding alone takes the eager run: the same run from the start LoRA
    # with every value moved by 2^-8 of itself (about one bfloat16 rounding step), up or down at random
    start_path = ctx.cache["moss_initial_lora"]
    nudged_path = os.path.join(ctx.cache["moss_runs"], "start_lora_nudged.safetensors")
    _nudged_lora(start_path, nudged_path, 2.0 ** -8, seed=3)
    ref = _run(ctx, "eager, start LoRA nudged", [], lora=nudged_path)
    r = _run(ctx, "compiled", ["compile=true"], env={"TORCH_LOGS": "recompiles,graph_breaks"})
    w_ref = _finished(rec, "eager, start LoRA nudged by 2^-8", ref)
    w = _finished(rec, "compiled", r)
    if w_ref is None:
        ctx.cache["moss_failed"] = True
    w_base = _weights(base["output"])
    if w is None or w.keys() != w_base.keys():
        if w is not None:
            rec.fail(f"compiled: the saved LoRA has other tensors than eager's ({len(w)} vs {len(w_base)})")
        ctx.cache["moss_failed"] = True
        return
    start = _weights(start_path)
    v = _versus_eager(base, w_base, start, r, w, start)
    yardstick = _versus_eager(base, w_base, start, ref, w_ref, _weights(nudged_path)) \
        if w_ref is not None and w_ref.keys() == w_base.keys() else None
    if yardstick:
        rec.line(f"yardstick: eager from the nudged start vs eager: {_versus_text(yardstick)}")
    graphs, breaks = _compile_stats(r["log"])
    recompiles = len(re.findall(r"Recompiling function", r["log"]))
    s_eager, s = _summary(ref if w_ref is not None else base), _summary(r)
    text = (f"compiled vs eager: {_versus_text(v)}; {graphs} compiled variants ({recompiles} recompiles), {breaks} graph "
            f"breaks; step {s.get('step s median') or 0:.2f} s vs {s_eager.get('step s median') or 0:.2f} s eager")
    # 1 - cosine: how far the change from the start turned away from eager's
    spread = 1 - v["cosine"]
    ref_spread = 1 - yardstick["cosine"] if yardstick else None
    # a yardstick that moved the first loss far less than compiling did is smaller than compiling's rounding, and a
    # direction beyond it then only warns
    small = yardstick is not None and yardstick["first loss"] < v["first loss"] / 4
    problem = ("nothing was compiled" if graphs == 0 else "graph breaks" if breaks else
               "the compiled forward pass differs" if v["first loss"] > 2e-2 else
               "the compiled run's update has another size" if not 0.5 <= v["size"] <= 2.0 else
               "the compiled run trained in another direction than rounding takes the eager run"
               if ref_spread is not None and spread > 2 * ref_spread + 0.1 and not small else None)
    if problem:
        ctx.cache["moss_failed"] = True
        rec.fail(f"{text}: {problem}")
    elif ref_spread is None:
        rec.warn(f"{text}: no yardstick to judge the direction by (the nudged eager run failed)")
    elif spread > ref_spread + 0.05:
        rec.warn(f"{text}: further from eager than the nudged start takes it (cosine {yardstick['cosine']:.3f})"
                 + ("; the nudge changed the first loss less than a quarter as much as compiling did, so the "
                    "yardstick may be smaller than compiling's rounding" if small else ""))
    else:
        rec.line(f"ok: {text}: within what rounding does to the eager run (cosine {yardstick['cosine']:.3f})")


@check(SECTION_TRAINING, "the saved LoRA loaded back at learning rate 0 saves the same tensors", timeout=3600.0)
def lora_round_trip(ctx: Ctx, rec: Rec):
    base = ctx.cache.get("moss_baseline")
    if base is None:
        rec.skip("needs the baseline of the training runs check")
        return
    r = _run(ctx, "LoRA loaded back", ["learning_rate=0", "ema=OFF", "sample_after_unit=NEVER", "validation=false"],
             epochs=1, lora=base["output"])
    w = _finished(rec, "LoRA loaded back", r, expect_samples=False, expect_validation=False)
    if w is None:
        ctx.cache["moss_failed"] = True
        return
    w_base = _weights(base["output"])
    same_keys = w.keys() == w_base.keys()
    diff = _weights_diff(w_base, w) if same_keys else float("inf")
    rec.expect(same_keys and diff == 0, f"{len(w)} tensors saved after loading {len(w_base)}: "
                                        + ("the same keys" if same_keys else "other keys") + f", largest change {diff:.1e}")
    if not (same_keys and diff == 0):
        ctx.cache["moss_failed"] = True


@check(SECTION_TRAINING, "full fine-tune: Adafactor with fused back pass and layer offloading, saved as the "
                         "transformer alone", full_only=True, timeout=3600.0,
       note="one short fine-tune; the saved transformer is ~10 GB")
def fine_tune(ctx: Ctx, rec: Rec):
    run_dir = os.path.join(ctx.cache["moss_runs"], "fine_tune")
    os.makedirs(run_dir, exist_ok=True)
    config = _run_config(ctx, run_dir, 1, "")
    config.update({
        "training_method": "FINE_TUNE", "output_model_format": "ORIGINAL_TRANSFORMER", "output_dtype": "BFLOAT_16",
        "optimizer": {"optimizer": "ADAFACTOR", "fused_back_pass": True, "stochastic_rounding": True},
        "learning_rate": 1e-6, "ema": "OFF", "sample_after_unit": "NEVER", "validation": False,
    })
    # a fine-tuned part can't use Simplex (its weights would live in a RAM buffer filled from the checkpoint)
    config["transformer"].update({"offload_fraction": 0.5, "activation_offloading": True, "simplex_offloading": False})
    values = [] if ctx.is_gpu else ["temp_device=cpu:1"]
    print("    training: full fine-tune", flush=True)
    r = _train(ctx, run_dir, config, {}, values, "diag")
    if r["rc"] != 0 or not os.path.exists(r["output"]):
        ctx.cache["moss_failed"] = True
        rec.fail(f"fine-tune failed (exit code {r['rc']}): {r['tail']}")
        return
    from safetensors import safe_open

    count, bad = 0, []
    with safe_open(r["output"], framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 (safe_open has no __iter__)
            count += 1
            if not finite(f.get_tensor(key)):
                bad.append(key)
    losses = _losses(r["diag"])
    s = _summary(r)
    rec.expect(not bad and losses and all(v == v for v in losses.values()),
               f"fine-tune: {len(losses)} steps, losses {[round(v, 4) for v in losses.values()]}, saved "
               f"{os.path.getsize(r['output']) / 2 ** 30:.1f} GiB ({count} tensors, {len(bad)} with NaN/inf)"
               + (f", peak reserved {s['peak reserved GiB']:.1f} GiB, step {s.get('step s median') or 0:.2f} s"
                  if "peak reserved GiB" in s else ""))
    if bad:
        ctx.cache["moss_failed"] = True


def finish(ctx: Ctx, keep: bool) -> str:
    """after the checks: keep every run's sample images (and, after a failure or with --keep, the whole runs)"""
    runs = ctx.cache.get("moss_runs")
    if not runs or not os.path.isdir(runs):
        return ""
    samples = os.path.join(ctx.out_dir, "samples")
    shutil.rmtree(samples, ignore_errors=True)
    for name in sorted(os.listdir(runs)):
        source = os.path.join(runs, name, "workspace", "samples")
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(samples, name))
    if keep or ctx.cache.get("moss_failed"):
        return f"the runs are kept in {runs} (each run's diag.log and diag.jsonl); sample images in {samples}"
    shutil.rmtree(runs, ignore_errors=True)
    return f"sample images of every run in {samples}"
