"""Check registry, runner and report for scripts/rocm_check.py.

Every check gets a fresh Rec, runs in a try block, and is followed by a device sync and cache release, so one failing
check (an exception, an OOM) doesn't stop the others. The report files are rewritten after every check: if the GPU
hangs or the process dies, the report still has everything up to the check that was running. A watchdog prints the
stack of a check that runs longer than its limit.

Checks that need a different process environment (allocator settings, TunableOp, the BLAS library, MIOpen's find
mode) are read by PyTorch or ROCm only at startup, so they run in child processes of this script (run_child).
"""
import faulthandler
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

PASS, WARN, FAIL, SKIP, INFO = "PASS", "WARN", "FAIL", "SKIP", "INFO"
_SEVERITY = {PASS: 0, INFO: 0, SKIP: 1, WARN: 2, FAIL: 3}

CHILD_RESULT_PREFIX = "ROCM_CHECK_CHILD_RESULT "


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return "?"


def onetrainer_version() -> str:
    """the branch and commit of this checkout, and whether tracked files differ from it"""
    return (f"{_git('rev-parse', '--abbrev-ref', 'HEAD')} @ {_git('rev-parse', '--short', 'HEAD')}"
            f"{' (uncommitted changes)' if _git('status', '--porcelain', '--untracked-files=no') else ''}")


def torch_stack() -> dict:
    """the PyTorch build a report was made with: torch.version.rocm exists in AMD's own builds (ROCm 10)"""
    return {"torch": torch.__version__, "hip": torch.version.hip, "rocm": getattr(torch.version, "rocm", None),
            "cuda": torch.version.cuda}


def stack_label(stack: dict) -> str:
    """'torch 2.13.0+rocm7.2 (HIP 7.2.x)' from torch_stack() or a report's header; empty for reports made before the
    header had it"""
    if not stack.get("torch"):
        return ""
    gpu = [f"HIP {stack['hip']}" if stack.get("hip") else "", f"ROCm {stack['rocm']}" if stack.get("rocm") else "",
           f"CUDA {stack['cuda']}" if stack.get("cuda") else ""]
    gpu = [part for part in gpu if part]
    return f"torch {stack['torch']}" + (f" ({', '.join(gpu)})" if gpu else "")


class Rec:
    """What a check reports: a status (the worst one set), detail lines and named numbers."""

    def __init__(self):
        self.status = PASS
        self.lines: list[str] = []
        self.metrics: dict = {}

    def _set(self, status: str):
        if _SEVERITY[status] > _SEVERITY[self.status] or self.status == PASS and status == INFO:
            self.status = status

    def line(self, text: str):
        self.lines.append(text)

    def info(self, text: str | None = None):
        self._set(INFO)
        if text:
            self.lines.append(text)

    def warn(self, text: str):
        self._set(WARN)
        self.lines.append("WARN: " + text)

    def fail(self, text: str):
        self._set(FAIL)
        self.lines.append("FAIL: " + text)

    def skip(self, text: str):
        self._set(SKIP)
        self.lines.append("skipped: " + text)

    def metric(self, name: str, value):
        if isinstance(value, float) and not math.isfinite(value):
            value = str(value)
        self.metrics[name] = value

    def expect(self, ok: bool, text: str):
        # a pass/fail line: kept on success too, so the report shows what was compared
        if ok:
            self.lines.append("ok: " + text)
        else:
            self.fail(text)


@dataclass
class Check:
    section: str
    name: str
    fn: Callable
    gpu_only: bool = False
    full_only: bool = False
    timeout: float = 600.0
    note: str = ""  # printed when the check starts, e.g. that it takes long the first time


REGISTRY: list[Check] = []


def check(section: str, name: str, gpu_only: bool = False, full_only: bool = False, timeout: float = 600.0,
          note: str = ""):
    def register(fn):
        REGISTRY.append(Check(section, name, fn, gpu_only, full_only, timeout, note))
        return fn
    return register


CHILD_TASKS: dict[str, Callable] = {}


def child_task(name: str):
    def register(fn):
        CHILD_TASKS[name] = fn
        return fn
    return register


@dataclass
class Ctx:
    device: torch.device
    full: bool
    out_dir: str
    script_path: str
    resolutions: list[int]
    seed: int = 0
    vae_path: str | None = None  # a real VAE for the VAE checks instead of a randomly initialized one
    cache: dict = field(default_factory=dict)

    @property
    def is_gpu(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    @property
    def is_rocm(self) -> bool:
        return torch.version.hip is not None

    @property
    def arch(self) -> str:
        if not self.is_gpu:
            return self.device.type
        props = torch.cuda.get_device_properties(self.device)
        if self.is_rocm:
            return props.gcnArchName.split(":")[0]
        return f"sm_{props.major}{props.minor}"

    def sync(self):
        if self.is_gpu:
            torch.cuda.synchronize(self.device)

    def generator(self, seed: int | None = None) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed if seed is None else seed)

    def bench(self, fn: Callable, iters: int | None = None, warmup: int | None = None) -> float:
        """median milliseconds per call of fn"""
        iters = iters or (20 if self.full else 8)
        warmup = warmup if warmup is not None else (5 if self.full else 2)
        for _ in range(warmup):
            fn()
        self.sync()
        times = []
        for _ in range(iters):
            if self.is_gpu:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return times[len(times) // 2]


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().to("cpu", torch.float64)
    expected = expected.detach().to("cpu", torch.float64)
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


def finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def output_tail(output: str, lines: int = 6) -> str:
    """the end of a process's output, for a report line. A crash in native code ends with faulthandler's report (every
    thread's stack, then the extension modules loaded), whose last lines say nothing: for one, the runtime's message
    before it (a failed assert in a GPU kernel, a HIP/HSA error, an uncaught C++ exception) and where the threads
    were (each one's innermost OneTrainer frame, else its innermost frame) instead"""
    text = [line.strip() for line in output.splitlines() if line.strip()]
    fatal = next((i for i, line in enumerate(text) if line.startswith("Fatal Python error")), None)
    if fatal is None:
        return " | ".join(text[-lines:])[-600:]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + os.sep
    # a kernel's failed assert is printed through the device printf (stdout), the runtime's own errors go to stderr
    markers = ("ssertion", "HSA_STATUS", "Memory access fault", "HIP error", "CUDA error", "terminate called", "what():")
    runtime = [line[:300] for line in text[:fatal] if any(marker in line for marker in markers)][-3:]
    parts = (runtime or [line[:300] for line in text[max(0, fatal - 2):fatal]]) + [text[fatal]]
    frames = []
    for i, line in enumerate(text[fatal + 1:], fatal + 1):
        if line.startswith(("Thread 0x", "Current thread 0x")):
            stack = []
            for frame in text[i + 1:]:
                if not frame.startswith("File "):
                    break
                stack.append(frame)
            ours = [frame for frame in stack if repo in frame]
            if stack:
                frames.append((not ours, not line.startswith("Current"), (ours or stack)[0].replace(repo, "")))
    shown = []
    for *_, frame in sorted(frames, key=lambda f: f[:2]):
        if frame not in shown and len(shown) < 4:
            shown.append(frame)
    return " | ".join(parts + shown)[:900]


def run_child(ctx: Ctx, task: str, env: dict[str, str | None], timeout: float = 900.0) -> dict:
    """runs one CHILD_TASKS entry in a fresh process with extra environment variables (None removes one), returns its
    result dict"""
    child_env = os.environ.copy()
    for key, value in env.items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value
    cmd = [sys.executable, ctx.script_path, "--child", task, "--device", str(ctx.device),
           "--out-dir", ctx.out_dir, "--seed", str(ctx.seed)]
    if ctx.full:
        cmd.append("--full")
    try:
        proc = subprocess.run(cmd, env=child_env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"no result after {timeout:.0f} s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(CHILD_RESULT_PREFIX):
            return json.loads(line[len(CHILD_RESULT_PREFIX):])
    return {"error": f"exit code {proc.returncode}: " + output_tail(proc.stdout + proc.stderr, 3)}


def child_main(task: str, ctx: Ctx):
    try:
        result = CHILD_TASKS[task](ctx)
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    print(CHILD_RESULT_PREFIX + json.dumps(result), flush=True)


class Report:
    def __init__(self, ctx: Ctx, argv: list[str], name: str = "rocm_check", title: str = "OneTrainer ROCm/GPU check"):
        self.ctx = ctx
        self.title = title
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")
        self.argv = argv
        # in the header, so every report (also one made with --only, which skips the environment check) says which
        # code and which PyTorch build it comes from
        self.onetrainer = onetrainer_version()
        self.stack = torch_stack()
        self.results: list[dict] = []
        self.running: str | None = None
        self.txt_path = os.path.join(ctx.out_dir, f"{name}_report.txt")
        self.json_path = os.path.join(ctx.out_dir, f"{name}_report.json")

    def write(self):
        data = {
            "started": self.started,
            "command": " ".join(self.argv),
            "device": str(self.ctx.device),
            "mode": "full" if self.ctx.full else "quick",
            "onetrainer": self.onetrainer,
            **self.stack,
            "platform": platform.platform(),
            "running_when_written": self.running,
            "results": self.results,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write(self.text())

    def text(self) -> str:
        out = [f"{self.title}  ({'full' if self.ctx.full else 'quick'} mode, device {self.ctx.device})",
               f"started {self.started}",
               f"OneTrainer {self.onetrainer}, {stack_label(self.stack)}",
               f"command: {' '.join(self.argv)}", ""]
        counts = {}
        for r in self.results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        out.append("summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        if self.running:
            out.append(f"!! the run stopped or is still in: {self.running}")
        problems = [r for r in self.results if r["status"] in (FAIL, WARN)]
        if problems:
            out.append("")
            out.append("problems:")
            out.extend(f"  [{r['status']}] {r['section']} / {r['name']}: {_first_problem(r['status'], r['lines'])}"
                       for r in problems)
        section = None
        for r in self.results:
            if r["section"] != section:
                section = r["section"]
                out += ["", f"=== {section} ==="]
            out.append(f"[{r['status']}] {r['name']}  ({r['seconds']:.1f} s)")
            out += [f"    {line}" for line in r["lines"]]
            for k, v in r["metrics"].items():
                out.append(f"    {k}: {_fmt(v)}")
        return "\n".join(out) + "\n"


def _first_problem(status: str, lines: list[str]) -> str:
    # the check's first line with its status (e.g. "FAIL: ..."), without that prefix: the later lines are often "ok"
    # lines or other details, which made the summary show a passing comparison for a failed check
    line = next((line for line in lines if line.startswith(status + ": ")), "")
    return line.removeprefix(status + ": ")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    return str(v)


def _global_state() -> dict:
    # process-wide switches a check may flip for a test; a leftover change would skew every later check (and a
    # leak of the same kind in OneTrainer's code would slow or break training)
    from modules.util import rocm_sdpa_fix
    return {"flash SDPA": torch.backends.cuda.flash_sdp_enabled(),
            "memory-efficient SDPA": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math SDPA": torch.backends.cuda.math_sdp_enabled(),
            "MIOpen/cuDNN": torch.backends.cudnn.enabled,
            "rocm_sdpa_fix installed": rocm_sdpa_fix.installed()}


def _restore_global_state(state: dict):
    from modules.util import rocm_sdpa_fix
    torch.backends.cuda.enable_flash_sdp(state["flash SDPA"])
    torch.backends.cuda.enable_mem_efficient_sdp(state["memory-efficient SDPA"])
    torch.backends.cuda.enable_math_sdp(state["math SDPA"])
    torch.backends.cudnn.enabled = state["MIOpen/cuDNN"]
    if state["rocm_sdpa_fix installed"]:
        rocm_sdpa_fix.install()
    else:
        rocm_sdpa_fix.uninstall()


def run_checks(ctx: Ctx, report: Report, only: list[str] | None, skip: list[str] | None):
    def selected(c: Check) -> bool:
        key = f"{c.section} / {c.name}".lower()
        if only and not any(s.lower() in key for s in only):
            return False
        return not (skip and any(s.lower() in key for s in skip))

    checks = [c for c in sorted(REGISTRY, key=lambda c: c.section) if selected(c)]  # stable: file order within a section
    for i, c in enumerate(checks):
        label = f"{c.section} / {c.name}"
        print(f"[{i + 1}/{len(checks)}] {label} ...", flush=True)
        if c.note and not (c.gpu_only and not ctx.is_gpu) and not (c.full_only and not ctx.full):
            print(f"    ({c.note})", flush=True)
        rec = Rec()
        t0 = time.perf_counter()
        if c.gpu_only and not ctx.is_gpu:
            rec.skip("needs a GPU")
        elif c.full_only and not ctx.full:
            rec.skip("runs with --full only")
        else:
            report.running = label
            report.write()
            state = _global_state()
            faulthandler.dump_traceback_later(c.timeout * (3 if ctx.full else 1), exit=False)
            try:
                c.fn(ctx, rec)
            except torch.OutOfMemoryError as e:
                rec.fail(f"out of memory: {str(e).splitlines()[0]}")
            except Exception as e:
                rec.fail(f"{type(e).__name__}: {e}")
                rec.lines += ["  " + line for line in traceback.format_exc().strip().splitlines()[-8:]]
            finally:
                faulthandler.cancel_dump_traceback_later()
            changed = {k: v for k, v in _global_state().items() if state[k] != v}
            if changed:
                rec.fail("left process-wide settings changed (restored for the next checks): "
                         + ", ".join(f"{k} {state[k]} -> {v}" for k, v in changed.items()))
                _restore_global_state(state)
            try:
                ctx.sync()
                if ctx.is_gpu:
                    torch.cuda.empty_cache()
            except Exception as e:
                rec.fail(f"device error after the check: {type(e).__name__}: {e}")
        report.results.append({"section": c.section, "name": c.name, "status": rec.status, "lines": rec.lines,
                               "metrics": rec.metrics, "seconds": time.perf_counter() - t0})
        report.running = None
        report.write()
        problem = _first_problem(rec.status, rec.lines) if rec.status in (FAIL, WARN) else ""
        print(f"    -> {rec.status}" + (f": {problem}" if problem else ""), flush=True)
