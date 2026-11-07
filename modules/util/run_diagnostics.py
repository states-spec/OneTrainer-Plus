"""OT_DIAG_LOG: a JSON-lines record of one training run (GPU memory, fragmentation, timing, loss), for comparing runs with
scripts/diag_report.py, e.g. the same config with and without OT_ROCM_MEM_POOLS.

OT_DIAG_LOG=true writes diagnostics/runs/<timestamp>.jsonl in the OneTrainer folder, any other value is the file to write. Every line
is one record and is flushed at once, so a crash or a GPU hang still leaves everything up to it. Records:
  header   versions, GPU, the allocator and GPU environment variables, whether MemPools are used, the config (no secrets)
  step     every optimizer step: loss, seconds since the previous step, memory now and its peak since the last record
  phase    a stretch of the run (loading, caching, sampling, validation, backup, save, final save): its duration, the
           memory before and after, the peak inside it, and the fragmentation after it
  move     model parts loaded onto or unloaded from the GPU (BaseModel.materialize/evict), with the same numbers
  validation  each validation concept's loss at a validation run
  error    the exception that ended the run (an out-of-memory error included), with the memory state at that moment
"""
import contextlib
import functools
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import torch

_TRUE = ("1", "true", "yes")
_ENV_PREFIXES = ("OT_", "PYTORCH_", "HIP_", "ROCR_", "HSA_", "MIOPEN_", "TORCH_", "AMD_", "CUDA_", "HCC_", "GPU_")
_ENV_SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
# the packages whose versions decide how a run computes, beside torch itself: runs from different environments (diag_ab
# python:PATH variants) differ in these too
_PACKAGES = ("torchvision", "triton", "pytorch-triton-rocm", "rocm-sdk-core", "diffusers", "transformers", "mgds",
             "adv_optm", "bitsandbytes", "safetensors", "accelerate")


def _package_versions() -> dict[str, str]:
    import importlib.metadata

    versions = {}
    for name in _PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return versions


def _private_ram_bytes() -> int | None:
    try:
        from modules.util.profiling_util import _private_ram_bytes as ram
        return ram()
    except Exception:
        return None


def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                               text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "?"


def _weights_digest(path: str) -> tuple[str, int]:
    import hashlib

    from safetensors import safe_open

    if os.path.isdir(path):
        files = sorted(os.path.join(root, name) for root, _, names in os.walk(path) for name in names
                       if name.endswith(".safetensors"))
    else:
        files = [path]
    if not files or not all(f.endswith(".safetensors") for f in files):
        return "no .safetensors weights to digest", 0
    sha = hashlib.sha256()
    count = 0
    for file in files:
        sha.update(os.path.relpath(file, path if os.path.isdir(path) else os.path.dirname(file)).encode())
        with safe_open(file, framework="pt") as f:
            for key in sorted(f.keys()):
                t = f.get_tensor(key).contiguous()
                sha.update(f"{key}|{t.dtype}|{tuple(t.shape)}".encode())
                sha.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
                count += 1
    return sha.hexdigest(), count


def apply_reproducibility_env():
    """OT_DETERMINISTIC=true asks torch for deterministic kernels (use_deterministic_algorithms, warn_only: an op without
    one warns and runs as before), so GPU runs differ less. OT_SEED=<int> replaces the run's training seed
    (training_seed.apply_training_seed)."""
    if os.environ.get("OT_DETERMINISTIC", "").lower() in _TRUE:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("OT_DETERMINISTIC: deterministic algorithms requested (ops without one warn)")


class RunDiagnostics:
    def __init__(self):
        self.__file = None
        self.__device: torch.device | None = None
        self.__start = 0.0
        self.__last_step_time: float | None = None
        self.__events_since_step = 0
        self.__depth = 0  # nested phases: only the outermost resets the peak counters
        self.path: str | None = None

    @property
    def enabled(self) -> bool:
        return self.__file is not None

    def start(self, config, train_device: torch.device):
        """open the log if OT_DIAG_LOG is set; only the first process of a multi-GPU run writes"""
        value = os.environ.get("OT_DIAG_LOG", "").strip()
        if self.__file is not None or not value or value.lower() in ("0", "false", "no"):
            return
        from modules.util import multi_gpu_util as multi
        if not multi.is_master():
            return
        from modules.util import path_util
        from modules.util.time_util import get_string_timestamp
        if value.lower() in _TRUE:
            path = path_util.diagnostics_path("runs", f"{get_string_timestamp()}.jsonl")
        else:
            path = value
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.__file = open(path, "a", encoding="utf-8")  # noqa: SIM115 (open for the whole run)
        self.path = path
        self.__device = train_device
        self.__start = time.perf_counter()
        self.__last_step_time = None
        print(f"Writing run diagnostics to {path}")
        self.__write("header", **self.__header(config))
        self.__reset_peaks()

    def close(self):
        if self.__file is not None:
            self.__write("end")
            self.__file.close()
            self.__file = None

    def step(self, global_step: int, epoch: int, loss: float):
        if self.__file is None:
            return
        now = time.perf_counter()
        seconds = None if self.__last_step_time is None else now - self.__last_step_time
        # a step whose interval also holds sampling, a backup or a part move is no measure of the step time
        with self.__guard(f"step {global_step}"):
            self.__write("step", step=global_step, epoch=epoch, loss=loss, seconds=seconds,
                         clean=self.__events_since_step == 0, **self.__memory(full=False),
                         **(self.__sensors() if global_step % 20 == 0 else {}))
        self.__last_step_time = now
        self.__events_since_step = 0
        if self.__depth == 0:
            self.__reset_peaks()

    @contextlib.contextmanager
    def phase(self, name: str, **fields):
        if self.__file is None:
            yield
            return
        outer = self.__depth == 0
        before = {}
        with self.__guard(f"the start of the {name} phase"):
            before = self.__memory(full=False)
            if outer:
                self.__reset_peaks()
        self.__depth += 1
        start = time.perf_counter()
        try:
            yield
        finally:
            self.__depth -= 1
            with self.__guard(f"the {name} phase"):
                after = self.__memory(full=True)
                self.__write("phase", name=name, seconds=time.perf_counter() - start,
                             before={k: before.get(k) for k in ("allocated", "reserved")}, **fields, **after)
                self.__events_since_step += 1
                if outer:
                    self.__reset_peaks()

    def phase_method(self, name: str):
        """decorator: the method's whole call is one phase"""
        def wrap(fn):
            @functools.wraps(fn)
            def inner(*args, **kwargs):
                with self.phase(name):
                    return fn(*args, **kwargs)
            return inner
        return wrap

    def record_errors(self, fn):
        """decorator: an exception leaving the method is logged (with the memory state) before it propagates"""
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except BaseException as e:
                with self.__guard("the error"):
                    self.error(e)
                raise
        return inner

    @contextlib.contextmanager
    def __guard(self, what: str):
        # the log must never end a run or replace the exception that is ending one: after a GPU fault the memory
        # queries raise too, and from inside a finally or except block that error would be the one reported
        try:
            yield
        except Exception as e:
            print(f"Run diagnostics: could not record {what}: {type(e).__name__}: {e}")

    def move(self, direction: str, parts: list[str], moved: bool, seconds: float):
        if self.__file is None or not moved:
            return
        self.__events_since_step += 1
        with self.__guard(f"the {direction} of {', '.join(parts)}"):
            self.__write("move", direction=direction, parts=list(parts), seconds=seconds, **self.__memory(full=True))

    def output(self, path: str):
        """a digest of the saved weights (tensor names, dtypes, shapes and data, not the file's metadata, which holds
        the date): two runs that saved bit-identical weights have the same one"""
        if self.__file is None:
            return
        start = time.perf_counter()
        try:
            digest, tensors = _weights_digest(path)
        except Exception as e:
            digest, tensors = f"not readable: {type(e).__name__}: {e}", 0
        self.__write("output", path=os.path.abspath(path), digest=digest, tensors=tensors,
                     seconds=time.perf_counter() - start)

    def validation(self, global_step: int, losses: dict[str, float]):
        """the validation loss of each validation concept (its label as in TensorBoard), averaged over its images"""
        if self.__file is None:
            return
        with self.__guard(f"the validation at step {global_step}"):
            self.__write("validation", step=global_step, losses=losses)

    def error(self, exc: BaseException):
        if self.__file is None:
            return
        self.__write("error", type=type(exc).__name__, message=str(exc)[:2000],
                     out_of_memory=isinstance(exc, torch.OutOfMemoryError), **self.__memory(full=True))

    def __write(self, kind: str, **fields):
        record = {"kind": kind, "t": round(time.perf_counter() - self.__start, 4), **fields}
        self.__file.write(json.dumps(record, default=str) + "\n")
        self.__file.flush()

    def __gpu(self) -> bool:
        return self.__device is not None and self.__device.type == "cuda" and torch.cuda.is_available()

    def __reset_peaks(self):
        if self.__gpu():
            torch.cuda.reset_peak_memory_stats(self.__device)

    def __memory(self, full: bool) -> dict[str, Any]:
        out: dict[str, Any] = {"ram": _private_ram_bytes()}
        if not self.__gpu():
            return out
        device = self.__device
        stats = torch.cuda.memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        out.update(
            allocated=stats.get("allocated_bytes.all.current", 0),
            reserved=stats.get("reserved_bytes.all.current", 0),
            peak_allocated=stats.get("allocated_bytes.all.peak", 0),
            peak_reserved=stats.get("reserved_bytes.all.peak", 0),
            # free space inside segments that also hold live tensors: reusable only by allocations that fit the holes
            inactive_split=stats.get("inactive_split_bytes.all.current", 0),
            # the allocator freed its cache and retried a failed cudaMalloc/hipMalloc: memory pressure or fragmentation
            alloc_retries=stats.get("num_alloc_retries", 0),
            ooms=stats.get("num_ooms", 0),
            device_allocs=stats.get("num_device_alloc", 0),
            device_frees=stats.get("num_device_free", 0),
            device_free=free,
            device_total=total,
        )
        if full:
            out.update(self.__sensors())
            segments = torch.cuda.memory_snapshot()
            segments = [s for s in segments if s.get("device", device.index or 0) == (device.index or 0)]
            holes = [b["size"] for s in segments for b in s["blocks"] if b["state"] != "active_allocated"]
            free_in_cache = sum(holes)
            # MemPool segments carry the pool's id; the default pool's is (0, 0)
            pooled = [s for s in segments if tuple(s.get("segment_pool_id", (0, 0))) != (0, 0)]
            out.update(
                segments=len(segments),
                largest_hole=max(holes, default=0),
                # the share of the cached free memory not in its largest hole: near 1 means it is all small pieces
                ext_frag=(1 - max(holes, default=0) / free_in_cache) if free_in_cache else 0.0,
                pool_segments=len(pooled),
                pool_reserved=sum(s["total_size"] for s in pooled),
            )
        return out

    def __sensors(self) -> dict[str, Any]:
        # temperature, clock and power, through amdsmi (ROCm) or pynvml (CUDA) when installed: throttling shows up as
        # slower steps that have nothing to do with the settings compared
        if not self.__gpu():
            return {}
        out = {}
        for name, fn in (("temperature_c", torch.cuda.temperature), ("clock_mhz", torch.cuda.clock_rate),
                         ("power_mw", torch.cuda.power_draw)):
            with contextlib.suppress(Exception):  # no amdsmi/pynvml, or the driver doesn't report it
                out[name] = fn(self.__device)
        return out

    def __header(self, config) -> dict[str, Any]:
        from modules.util.torch_util import supports_mem_pool

        device = self.__device
        info: dict[str, Any] = {
            "commit": _git_commit(),
            "argv": sys.argv,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "rocm": getattr(torch.version, "rocm", None),  # AMD's own builds (ROCm 10)
            "cuda": torch.version.cuda,
            "packages": _package_versions(),
            "device": str(device),
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith(_ENV_PREFIXES) and not any(w in k for w in _ENV_SECRET_WORDS)},
        }
        if self.__gpu():
            props = torch.cuda.get_device_properties(device)
            info.update(
                gpu=props.name,
                arch=props.gcnArchName.split(":")[0] if torch.version.hip else f"sm_{props.major}{props.minor}",
                gpu_count=torch.cuda.device_count(),
                allocator=torch.cuda.memory.get_allocator_backend(),
                mem_pools=supports_mem_pool(device),
            )
        try:
            info["config"] = config.to_settings_dict(secrets=False)
        except Exception as e:
            info["config"] = f"not readable: {e}"
        return info


diagnostics = RunDiagnostics()
