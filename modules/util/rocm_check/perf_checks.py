"""Speed of the kernels training spends its time in, per model family and resolution, and what the ROCm runtime
switches (BLAS library, TunableOp, MIOpen find mode, allocator settings, torch.compile) change. Numbers only: a slow
result is reported, not failed, except where a setting breaks.
"""
import glob
import os
import time

from modules.util.rocm_check.framework import Ctx, Rec, check, child_task, finite, rel_err, run_child
from modules.util.rocm_check.shapes import FAMILIES

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

SECTION = "4 speed"


def _rand(ctx: Ctx, *shape, dtype=torch.bfloat16, scale=1.0, seed=0) -> torch.Tensor:
    g = torch.Generator(device=ctx.device).manual_seed(seed)
    return (torch.randn(*shape, generator=g, device=ctx.device) * scale).to(dtype)


def _tflops(flops: float, ms: float) -> float:
    return flops / ms / 1e9


@check(SECTION, "attention fwd+bwd per model family and backend", gpu_only=True)
def attention_speed(ctx: Ctx, rec: Rec):
    rec.info()
    backends = [None, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    if ctx.full:
        backends.append(SDPBackend.MATH)
    for family in FAMILIES:
        for res in ctx.resolutions:
            s = family.seq(res)
            q, k, v = (_rand(ctx, 1, family.heads, s, family.head_dim, seed=i).requires_grad_() for i in range(3))
            flops = 4 * family.heads * s * s * family.head_dim * 3.5  # fwd (2 matmuls) + bwd (~2.5x)
            parts, times = [], {}
            for backend in backends:
                name = "default" if backend is None else backend.name.replace("_ATTENTION", "").lower()
                if backend == SDPBackend.MATH and s > 4608:
                    continue  # the math kernel materializes seq x seq scores: skip what can't fit
                torch.cuda.reset_peak_memory_stats(ctx.device)
                before = torch.cuda.memory_allocated(ctx.device)

                def run(backend=backend, q=q, k=k, v=v):
                    if backend is None:
                        out = F.scaled_dot_product_attention(q, k, v)
                    else:
                        with sdpa_kernel([backend]):
                            out = F.scaled_dot_product_attention(q, k, v)
                    out.backward(torch.ones_like(out))

                try:
                    ms = ctx.bench(run, iters=10 if ctx.full else 4, warmup=2)
                except torch.OutOfMemoryError:  # a RuntimeError subclass: must come first
                    parts.append(f"{name} OOM")
                    torch.cuda.empty_cache()
                    continue
                except RuntimeError as e:
                    if any(s in str(e) for s in ("No available kernel", "No viable backend")):
                        parts.append(f"{name} n/a")
                        continue
                    raise
                peak = (torch.cuda.max_memory_allocated(ctx.device) - before) / 2**20
                parts.append(f"{name} {ms:.2f} ms ({_tflops(flops, ms):.0f} TFLOPS, +{peak:.0f} MiB)")
                rec.metric(f"{family.name} @{res} {name} ms", ms)
                times[name] = ms
            rec.line(f"{family.name} @ {res}px (1x{family.heads}x{s}x{family.head_dim} bf16): " + "; ".join(parts))
            fused = [times[n] for n in ("flash", "efficient") if n in times]
            if "default" in times and fused and times["default"] > 2 * min(fused):
                rec.warn(f"{family.name} @ {res}px: the default kernel is {times['default'] / min(fused):.1f}x slower "
                         f"than the fastest fused one, so training would run the slow path here")
            del q, k, v


@check(SECTION, "linear layers fwd+bwd per model family (bf16)", gpu_only=True)
def linear_speed(ctx: Ctx, rec: Rec):
    rec.info()
    for family in FAMILIES:
        res = ctx.resolutions[0]
        tokens = family.seq(res)
        x = _rand(ctx, tokens, family.hidden, seed=1).requires_grad_()
        results = []
        for in_f, out_f in ((family.hidden, family.hidden * 3), (family.hidden, family.mlp), (family.mlp, family.hidden)):
            w = _rand(ctx, out_f, in_f, scale=in_f ** -0.5, seed=2)  # frozen base weight, like LoRA training
            xi = x if in_f == family.hidden else _rand(ctx, tokens, in_f, seed=3).requires_grad_()
            grad = torch.ones(tokens, out_f, device=ctx.device, dtype=torch.bfloat16)
            ms = ctx.bench(lambda xi=xi, w=w, grad=grad: F.linear(xi, w).backward(grad))
            tf = _tflops(2 * tokens * in_f * out_f * 2, ms)  # forward + input gradient
            results.append(f"{in_f}->{out_f} {ms:.2f} ms ({tf:.0f} TFLOPS)")
            rec.metric(f"{family.name} {tokens}x{in_f}->{out_f} TFLOPS", tf)
        rec.line(f"{family.name} @ {res}px, {tokens} tokens: " + "; ".join(results))


@check(SECTION, "VAE convolutions (MIOpen) per resolution", gpu_only=True)
def conv_speed(ctx: Ctx, rec: Rec):
    rec.info()
    for dtype in (torch.float32, torch.bfloat16):
        for res in ctx.resolutions:
            x = _rand(ctx, 1, 128, res, res, dtype=dtype, seed=1)
            w = _rand(ctx, 128, 128, 3, 3, dtype=dtype, scale=(128 * 9) ** -0.5, seed=2)
            t0 = time.perf_counter()
            F.conv2d(x, w, padding=1)
            ctx.sync()
            first = (time.perf_counter() - t0) * 1000
            ms = ctx.bench(lambda x=x, w=w: F.conv2d(x, w, padding=1))
            name = str(dtype)[6:]
            rec.line(f"{name} 128ch 3x3 at {res}x{res}: {ms:.2f} ms (first call {first:.0f} ms incl. kernel search)")
            rec.metric(f"conv {name} {res}px ms", ms)


@child_task("core_bench")
def core_bench(ctx: Ctx) -> dict:
    """a fixed mix of what training runs, timed in a fresh process (so its environment variables apply)"""
    out = {}
    t0 = time.perf_counter()
    for name, (m, k, n) in {"linear 1536x3072x12288": (1536, 3072, 12288),
                            "linear 4096x640x5120": (4096, 640, 5120),
                            "linear 1357x1536x6144": (1357, 1536, 6144)}.items():
        x = _rand(ctx, m, k, seed=1).requires_grad_()
        w = _rand(ctx, n, k, scale=k ** -0.5, seed=2)
        g = torch.ones(m, n, device=ctx.device, dtype=torch.bfloat16)
        out[name + " ms"] = ctx.bench(lambda x=x, w=w, g=g: F.linear(x, w).backward(g), iters=10 if ctx.full else 5)
        # the kernels picked for this process (TunableOp's tuned ones after the timing above) against an fp64 CPU
        # reference: a switch that is faster but wrong shows here, not only as training runs that drift apart
        x.grad = None
        y = F.linear(x, w)
        y.backward(g)
        x64, w64 = x.detach().to("cpu", torch.float64), w.to("cpu", torch.float64)
        out[name + " out err"] = rel_err(y, x64 @ w64.T)
        out[name + " dx err"] = rel_err(x.grad, g.to("cpu", torch.float64) @ w64)
    q = _rand(ctx, 1, 24, 1536, 128, seed=3).requires_grad_()
    out["attention 24x1536x128 ms"] = ctx.bench(
        lambda: F.scaled_dot_product_attention(q, q, q).backward(torch.ones_like(q)), iters=10 if ctx.full else 5)
    x = _rand(ctx, 1, 128, 512, 512, dtype=torch.float32, seed=4)
    w = _rand(ctx, 128, 128, 3, 3, dtype=torch.float32, scale=0.03, seed=5)
    t1 = time.perf_counter()
    F.conv2d(x, w, padding=1)
    ctx.sync()
    out["conv first call ms"] = (time.perf_counter() - t1) * 1000
    out["conv fp32 512px ms"] = ctx.bench(lambda: F.conv2d(x, w, padding=1))
    out["total s"] = time.perf_counter() - t0
    out["blas"] = str(torch.backends.cuda.preferred_blas_library()).removeprefix("_BlasBackend.")
    return out


@check(SECTION, "runtime switches: BLAS library, TunableOp, MIOpen find mode", gpu_only=True, timeout=1800)
def runtime_switches(ctx: Ctx, rec: Rec):
    rec.info()
    variants = {"default": {}}
    if ctx.is_rocm:
        # the library PyTorch doesn't pick for this GPU: on gfx1100 the default is already rocBLAS (Cublas), so
        # TORCH_BLAS_PREFER_HIPBLASLT=0 changed nothing (bit-identical training runs)
        if "Cublaslt" in str(torch.backends.cuda.preferred_blas_library()):
            variants["rocBLAS instead of hipBLASLt (TORCH_BLAS_PREFER_HIPBLASLT=0)"] = {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}
        else:
            variants["hipBLASLt instead of rocBLAS (TORCH_BLAS_PREFER_HIPBLASLT=1)"] = {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}
        variants["MIOPEN_FIND_MODE=FAST"] = {"MIOPEN_FIND_MODE": "FAST"}
    tuned_before = []
    if ctx.full:
        # TunableOp reads the results file an earlier rocm_check run left in the output folder and tunes only the
        # shapes it doesn't hold, so its timing includes a tuning only when there was no such file
        tuned_before = sorted(os.path.basename(f) for f in glob.glob(os.path.join(ctx.out_dir, "rocm_check_tunableop*.csv")))
        variants["TunableOp (PYTORCH_TUNABLEOP_ENABLED=1)"] = {
            "PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1",
            "PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS": "100",
            "PYTORCH_TUNABLEOP_FILENAME": os.path.join(ctx.out_dir, "rocm_check_tunableop%d.csv"),
        }
    # every run starts without the switches compared here, so "default" is PyTorch's default even when the shell sets
    # one (TunableOp from the environment would also tune into, and read, the training runs' results file)
    switches = sorted({"TORCH_BLAS_PREFER_HIPBLASLT", "MIOPEN_FIND_MODE", "PYTORCH_TUNABLEOP_ENABLED",
                       "PYTORCH_TUNABLEOP_TUNING", "PYTORCH_TUNABLEOP_FILENAME",
                       *(key for env in variants.values() for key in env)})
    inherited = [f"{key}={os.environ[key]}" for key in switches if key in os.environ]
    if inherited:
        rec.line(f"set in this environment and left out of every run here, so that 'default' is PyTorch's default: "
                 f"{', '.join(inherited)}")
    variants = {name: {**dict.fromkeys(switches), **env} for name, env in variants.items()}
    if ctx.is_rocm:
        # each run gets an empty MIOpen tuning database, so "conv first call" compares like with like
        for i, env in enumerate(variants.values()):
            db = os.path.join(ctx.out_dir, "rocm_check_miopen_db", str(i))
            os.makedirs(db, exist_ok=True)
            env["MIOPEN_USER_DB_PATH"] = db
    results = {name: run_child(ctx, "core_bench", env) for name, env in variants.items()}
    base = results["default"]
    if "error" in base:
        rec.fail(f"default run failed: {base['error']}")
        return
    for name, result in results.items():
        if "error" in result:
            rec.warn(f"{name}: {result['error']}")
            continue
        parts = []
        for key, value in result.items():
            if key.endswith(" ms") and key in base:
                ratio = base[key] / value if value else float("nan")
                parts.append(f"{key.removesuffix(' ms')} {value:.2f} ms" + ("" if name == "default" else f" ({ratio:.2f}x)"))
                rec.metric(f"{name}: {key}", value)
        rec.line(f"{name} (BLAS {result.get('blas', '?')}, total {result.get('total s', 0):.0f} s): " + "; ".join(parts))
        if "BLAS" in name and result.get("blas") and result.get("blas") == base.get("blas"):
            # PyTorch keeps its default where it doesn't allow the other library (hipBLASLt on gfx1100 in torch 2.13)
            rec.line(f"{name}: PyTorch ignored the setting on this GPU and still used {result['blas']}, so the "
                     f"numbers above are the default library again (run-to-run spread)")
        errors = {k: v for k, v in result.items() if k.endswith(" err")}
        if errors:
            worst = max(errors, key=errors.get)
            # bf16 inputs: the same limit as the BLAS correctness check
            rec.expect(errors[worst] < 2e-2, f"{name}: linear forward/backward vs fp64, worst rel err "
                                             f"{errors[worst]:.2e} ({worst.removesuffix(' err')}, limit 2e-02)")
    if ctx.full:
        rec.line(f"TunableOp used the results an earlier run saved in {', '.join(tuned_before)} (delete it to time a "
                 f"new tuning)" if tuned_before else
                 "TunableOp tuned in this run (its total includes the tuning); the results are saved in "
                 "rocm_check_tunableop0.csv and used by the next run")
    rec.line("(x = speed relative to default; >1 is faster. The default run's times vary between runs, e.g. 0.68 vs "
             "1.17 ms for the same 4096x640x5120 kernel in two runs on an RX 7900 XTX: compare ratios within a run)")


@child_task("alloc_pattern")
def alloc_pattern(ctx: Ctx) -> dict:
    """allocates and frees tensors of changing sizes, like steps over aspect ratio buckets and model swaps"""
    free, _ = torch.cuda.mem_get_info(ctx.device)
    torch.cuda.reset_peak_memory_stats(ctx.device)
    kept = []
    for round_ in range(6):
        # sizes relative to the free memory, growing each round: at most ~45% of it is held at once
        for fraction in (0.04, 0.008, 0.08, 0.004, 0.06, 0.02, 0.12, 0.003):
            kept.append(torch.empty(int(free * fraction * (1 + 0.1 * round_)) // 2, dtype=torch.bfloat16,
                                    device=ctx.device))
            if len(kept) > 4:
                kept.pop(0)
    kept.clear()
    big = torch.empty(int(free * 0.4) // 2, dtype=torch.bfloat16, device=ctx.device)  # a model part after the churn
    del big
    return {"peak allocated GiB": torch.cuda.max_memory_allocated(ctx.device) / 2**30,
            "peak reserved GiB": torch.cuda.max_memory_reserved(ctx.device) / 2**30,
            "alloc conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}


def _without_expandable(conf: str) -> str:
    return ",".join(part for part in conf.split(",") if part and not part.strip().startswith("expandable_segments"))


# the checks that gave wrong results with expandable_segments on the RX 7900 XTX (and passed without it)
_ALLOCATOR_CHECKS = ["convolution, group norm, upsample", "VAE encode per layer", "concurrent VAE encodes"]
_ALLOCATOR_CHECKS_FULL = ["layer + activation offloading"]


@child_task("alloc_checks")
def alloc_checks(ctx: Ctx) -> dict:
    """the correctness checks that broke under expandable_segments, run in this process's allocator setting after an
    allocation churn (a synthetic conv test alone passed under it while these failed)"""
    from modules.util.rocm_check.framework import REGISTRY, Rec

    alloc_pattern(ctx)
    torch.cuda.empty_cache()
    wanted = _ALLOCATOR_CHECKS + (_ALLOCATOR_CHECKS_FULL if ctx.full else [])
    results = {}
    for c in REGISTRY:
        if not any(c.name.startswith(w) for w in wanted):
            continue
        rec = Rec()
        try:
            c.fn(ctx, rec)
        except Exception as e:
            rec.fail(f"{type(e).__name__}: {e}")
        failures = [line for line in rec.lines if line.startswith("FAIL")]
        results[c.name] = {"status": rec.status, "failures": failures}
    return {"checks": results, "alloc conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}


@check(SECTION, "allocator: default vs expandable_segments (OT_EXPANDABLE_SEGMENTS), memory and correctness",
       gpu_only=True, note="runs the VAE, conv and (with --full) offloading checks again under each allocator")
def allocator(ctx: Ctx, rec: Rec):
    rec.info()
    # PyTorch reads the first of these that is set; wrong results under expandable_segments fail the check only when
    # this setup uses it, else they warn (it is off by default)
    current = next((os.environ[k] for k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
                    if os.environ.get(k)), "")
    in_use = any(part.replace(" ", "").lower() == "expandable_segments:true" for part in current.split(","))
    default = _without_expandable(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
    expandable = ",".join(filter(None, [default, "expandable_segments:True"]))
    wrong = {}
    for name, conf in (("default", default), ("expandable_segments", expandable)):
        # an empty value makes PyTorch fall back to PYTORCH_HIP_ALLOC_CONF/PYTORCH_ALLOC_CONF; clear those too
        env = {"PYTORCH_CUDA_ALLOC_CONF": conf, "PYTORCH_HIP_ALLOC_CONF": conf, "PYTORCH_ALLOC_CONF": conf}
        result = run_child(ctx, "alloc_pattern", env)
        if "error" in result:
            (rec.fail if name == "default" else rec.warn)(f"{name}: {result['error']}")
        else:
            waste = result["peak reserved GiB"] - result["peak allocated GiB"]
            rec.line(f"{name} ({conf or 'no settings'}): peak {result['peak allocated GiB']:.2f} GiB in tensors, "
                     f"{result['peak reserved GiB']:.2f} GiB reserved ({waste:.2f} GiB held unused)")
            rec.metric(f"{name} reserved-unused GiB", waste)
        result = run_child(ctx, "alloc_checks", env, timeout=1800.0)
        if "error" in result:
            rec.fail(f"{name} correctness run: {result['error']}")
            continue
        wrong[name] = 0
        for check_name, r in result["checks"].items():
            wrong[name] += len(r["failures"])
            text = f"{name}: {check_name}: {r['status']}" \
                + (f" ({len(r['failures'])} failures, first: {r['failures'][0][6:]})" if r["failures"] else "")
            if r["failures"] and name == "expandable_segments" and not in_use:
                rec.warn(text)
            else:
                rec.expect(not r["failures"], text)
    if wrong.get("expandable_segments") and wrong.get("default") == 0:
        (rec.fail if in_use else rec.warn)(
            "expandable_segments gives wrong results on this GPU while the default allocator doesn't: don't set "
            "OT_EXPANDABLE_SEGMENTS=true (or expandable_segments in PYTORCH_*ALLOC_CONF)"
            + (". This setup uses it now" if in_use else "; this setup doesn't use it"))
    elif wrong.get("default"):
        rec.fail("wrong results even with the default allocator: a driver, firmware or hardware (VRAM clock, "
                 "undervolt, temperature) problem, not an allocator setting")


@child_task("mem_pool_checks")
def mem_pool_checks(ctx: Ctx) -> dict:
    """the offloading and disk-streaming checks (their caches and streamed layers go into MemPools) and a model part
    moved into its own MemPool and back, as BaseModel._move_part does, in this process's OT_ROCM_MEM_POOLS setting,
    after an allocation churn"""
    import copy

    from modules.util.rocm_check.framework import REGISTRY
    from modules.util.torch_util import create_mem_pool, mem_pool_context, supports_mem_pool

    alloc_pattern(ctx)
    torch.cuda.empty_cache()
    results = {}
    for c in REGISTRY:
        if not c.name.startswith(("layer + activation offloading", "disk streaming")):
            continue
        rec = Rec()
        try:
            c.fn(ctx, rec)
        except Exception as e:
            rec.fail(f"{type(e).__name__}: {e}")
        results[c.name] = {"status": rec.status, "failures": [line for line in rec.lines if line.startswith("FAIL")]}

    rec = Rec()
    try:
        torch.manual_seed(0)
        part = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
        original = copy.deepcopy(part.state_dict())
        x = _rand(ctx, 256, 1024, dtype=torch.float32, seed=1)
        expected = copy.deepcopy(part).to(ctx.device)(x)
        worst, identical = 0.0, True
        for _ in range(3):
            pool = create_mem_pool(ctx.device)
            with mem_pool_context(pool, device=ctx.device):
                part.to(ctx.device)
            worst = max(worst, rel_err(part(x), expected))
            part.to("cpu")
            del pool
            torch.cuda.empty_cache()
            identical &= all(torch.equal(v, original[k]) for k, v in part.state_dict().items())
        rec.expect(worst < 1e-5, f"output vs a part outside a MemPool rel err {worst:.1e}")
        rec.expect(identical, "weights bit-identical after 3 moves into a MemPool and back")
    except Exception as e:
        rec.fail(f"{type(e).__name__}: {e}")
    results["model part in its own MemPool"] = {"status": rec.status,
                                                "failures": [line for line in rec.lines if line.startswith("FAIL")]}
    return {"checks": results, "pools": supports_mem_pool(ctx.device)}


@check(SECTION, "mem_pool: allocations with and without MemPools (OT_ROCM_MEM_POOLS), correctness", gpu_only=True,
       note="runs the offloading and disk-streaming checks again with and without the MemPools they use")
def mem_pools(ctx: Ctx, rec: Rec):
    rec.info()
    wrong = {}
    for name, value in (("without MemPools", "false"), ("MemPools", "true")):
        result = run_child(ctx, "mem_pool_checks", {"OT_ROCM_MEM_POOLS": value}, timeout=1800.0)
        if "error" in result:
            rec.fail(f"{name}: {result['error']}")
            continue
        if ctx.is_rocm:
            rec.expect(result["pools"] == (value == "true"),
                       f"OT_ROCM_MEM_POOLS={value}: OneTrainer {'uses' if result['pools'] else 'does not use'} MemPools")
        wrong[name] = 0
        for check_name, r in result["checks"].items():
            wrong[name] += len(r["failures"])
            rec.expect(not r["failures"], f"{name}: {check_name}: {r['status']}"
                       + (f" ({len(r['failures'])} failures, first: {r['failures'][0][6:]})" if r["failures"] else ""))
    if not ctx.is_rocm:
        rec.line("(outside ROCm OneTrainer always uses MemPools on the native allocator; OT_ROCM_MEM_POOLS changes nothing)")
    elif wrong.get("MemPools") and wrong.get("without MemPools") == 0:
        rec.fail("MemPools give wrong results on this GPU while normal allocations don't: keep OT_ROCM_MEM_POOLS unset")
    elif wrong.get("MemPools") == 0 and wrong.get("without MemPools") == 0:
        rec.line("MemPools gave correct results here: OT_ROCM_MEM_POOLS=true can be used")


class _DiTBlock(nn.Module):
    """a generic pre-norm transformer block (attention + MLP), the shape most of the DiT models share"""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.mlp1, self.mlp2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, d)
        x = x + self.out(a)
        return x + self.mlp2(F.gelu(self.mlp1(self.norm2(x)), approximate="tanh"))


@check(SECTION, "torch.compile on a transformer block (the 'compile' option)", gpu_only=True, full_only=True,
       timeout=1800)
def compile_speed(ctx: Ctx, rec: Rec):
    rec.info()
    from modules.util.compile_util import init_compile

    init_compile()
    torch.manual_seed(0)
    block = _DiTBlock(1536, 24).to(ctx.device, torch.bfloat16).requires_grad_(False)
    x = _rand(ctx, 1, 1357, 1536, seed=1).requires_grad_()
    ref = block(x)
    eager_ms = ctx.bench(lambda: block(x).float().sum().backward())
    from torch._dynamo.utils import counters

    counters.clear()
    compiled = torch.compile(block, fullgraph=True)
    t0 = time.perf_counter()
    out = compiled(x)
    out.float().sum().backward()
    ctx.sync()
    compile_s = time.perf_counter() - t0
    graphs = counters["stats"].get("unique_graphs", 0)
    cache_hits = sum(v for k, v in counters["inductor"].items() if "cache_hit" in k)
    rec.line(f"dynamo compiled {graphs} graph(s), {sum(counters['graph_break'].values())} graph break(s), "
             f"{cache_hits} inductor cache hit(s) (a hit loads kernels compiled by an earlier run)")
    if graphs == 0:
        rec.warn("torch.compile compiled nothing: the timing below is eager")
    err = rel_err(out, ref)
    rec.expect(finite(out) and err < 3e-2, f"compiled output vs eager: rel err {err:.1e}")
    compiled_ms = ctx.bench(lambda: compiled(x).float().sum().backward())
    rec.line(f"eager {eager_ms:.2f} ms, compiled {compiled_ms:.2f} ms ({eager_ms / compiled_ms:.2f}x), "
             f"first compile {compile_s:.0f} s")
    rec.metric("eager ms", eager_ms)
    rec.metric("compiled ms", compiled_ms)
    rec.metric("compile seconds", compile_s)

    # what the option compiles in training: the same block with a LoRA (rank 16, fp32 weights) on every linear layer,
    # under bf16 autocast, inside OneTrainer's checkpoint layer (checkpointing on), and only the LoRA trained
    from modules.module.LoRAModule import LoRAModule
    from modules.util.checkpointing_util import create_checkpoint

    def training_block(compile_: bool):
        torch.manual_seed(0)
        base = _DiTBlock(1536, 24).to(ctx.device, torch.bfloat16).requires_grad_(False)
        loras = []
        for name, module in list(base.named_modules()):
            if isinstance(module, nn.Linear):
                lora = LoRAModule(name, module, 16, 16.0)
                with torch.no_grad():  # LoRA up starts at 0: move it so the delta is exercised
                    lora.lora_up.weight.copy_(_rand(ctx, *lora.lora_up.weight.shape, dtype=torch.float32,
                                                    scale=0.01, seed=7))
                lora.to(ctx.device)
                lora.hook_to_module()
                loras.append(lora)
        return create_checkpoint(base, checkpointing=True, compile=compile_), loras

    def step(block, loras):
        for lora in loras:
            for p in lora.parameters():
                p.grad = None
        with torch.autocast(ctx.device.type, dtype=torch.bfloat16):
            out = block(x)
        out.float().sum().backward()
        return out

    results = {}
    for compile_ in (False, True):
        block, loras = training_block(compile_)
        if ctx.is_gpu:
            torch.cuda.reset_peak_memory_stats(ctx.device)
        t0 = time.perf_counter()
        out = step(block, loras)
        ctx.sync()
        first_s = time.perf_counter() - t0
        grads = [p.grad.float().clone() for lora in loras for p in lora.parameters()]
        ms = ctx.bench(lambda block=block, loras=loras: step(block, loras))
        peak = torch.cuda.max_memory_allocated(ctx.device) / 2**20 if ctx.is_gpu else 0.0
        results[compile_] = (out.detach().float(), grads, ms, peak, first_s)
    (out_e, grads_e, ms_e, peak_e, _), (out_c, grads_c, ms_c, peak_c, first_c) = results[False], results[True]
    grad_err = max(rel_err(c, e) for c, e in zip(grads_c, grads_e, strict=True))
    rec.expect(finite(out_c) and rel_err(out_c, out_e) < 3e-2 and grad_err < 5e-2,
               f"with LoRA and checkpointing (the training path): compiled vs eager output "
               f"{rel_err(out_c, out_e):.1e}, LoRA gradients {grad_err:.1e}")
    rec.line(f"with LoRA and checkpointing: eager {ms_e:.2f} ms, compiled {ms_c:.2f} ms ({ms_e / ms_c:.2f}x), "
             f"peak allocated {peak_e:.0f} vs {peak_c:.0f} MiB, first compile {first_c:.0f} s")
    rec.metric("LoRA + checkpointing eager ms", ms_e)
    rec.metric("LoRA + checkpointing compiled ms", ms_c)
    rec.metric("LoRA + checkpointing compile seconds", first_c)
