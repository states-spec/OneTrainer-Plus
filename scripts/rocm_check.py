from util.import_util import script_imports

script_imports()

import argparse
import os
import sys

import modules.util.rocm_check.compile_checks  # noqa: F401
import modules.util.rocm_check.env_checks  # noqa: F401 (registers the checks, in report order)
import modules.util.rocm_check.kernel_checks  # noqa: F401
import modules.util.rocm_check.perf_checks  # noqa: F401
import modules.util.rocm_check.training_checks  # noqa: F401
import modules.util.rocm_check.workload_checks  # noqa: F401
from modules.util import path_util
from modules.util.rocm_check.framework import FAIL, REGISTRY, Ctx, Report, child_main, run_checks

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Checks and measures what OneTrainer runs on the GPU: kernel correctness against CPU references, "
                    "the training code (adapters, optimizers, offloading, caching), and speed per model family. "
                    "Made for ROCm, works on CUDA and (partly) on the CPU. Writes rocm_check_report.txt and .json to diagnostics/rocm_check.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to test (default: cuda if available, else cpu)")
    parser.add_argument("--full", action="store_true",
                        help="more dtypes, sizes, optimizers and repeats, plus TunableOp, torch.compile and the "
                             "training features (about 45-60 minutes instead of about 5)")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 1024],
                        help="training resolutions the speed checks size their tensors for (default: 512 1024)")
    parser.add_argument("--out-dir", default=path_util.diagnostics_path("rocm_check"),
                        help="where the report files go (default: diagnostics/rocm_check in the OneTrainer folder)")
    parser.add_argument("--only", nargs="+", help="run only checks whose 'section / name' contains one of these")
    parser.add_argument("--skip", nargs="+", help="skip checks whose 'section / name' contains one of these")
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae", help="a VAE for the VAE checks instead of a randomly initialized one: a diffusers "
                                      "model folder (with vae/), a VAE folder or a .safetensors file holding a VAE (read offline)")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list:
        for c in sorted(REGISTRY, key=lambda c: c.section):
            flags = [f for f, on in (("GPU only", c.gpu_only), ("--full only", c.full_only)) if on]
            print(f"{c.section} / {c.name}" + (f"  ({', '.join(flags)})" if flags else ""))
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        torch.cuda.set_device(device)
    ctx = Ctx(device=device, full=args.full, out_dir=os.path.abspath(args.out_dir),
              script_path=os.path.abspath(__file__), resolutions=args.resolutions, seed=args.seed,
              vae_path=args.vae)

    if args.child:
        child_main(args.child, ctx)
        return 0

    report = Report(ctx, [os.path.basename(sys.executable)] + sys.argv)
    run_checks(ctx, report, args.only, args.skip)
    print()
    print(report.text())
    print(f"report written to {report.txt_path} and {report.json_path}")
    return 1 if any(r["status"] == FAIL for r in report.results) else 0


if __name__ == "__main__":
    sys.exit(main())
