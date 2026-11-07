from util.import_util import script_imports

script_imports()

import argparse
import os
import sys

import modules.util.rocm_check.moss_checks as moss_checks
from modules.util import path_util
from modules.util.rocm_check.framework import FAIL, REGISTRY, Ctx, Report, child_main, run_checks

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Checks MOSS-Image on the real model: its text encoder, VAE and transformer on the GPU against "
                    "float32 on the CPU, then LoRA training through scripts/train.py with offloading, streaming, "
                    "resuming, compiling and a round trip of the saved LoRA, each compared with a baseline. About "
                    "30 minutes on a 7900 XTX; needs ~50 GB of RAM and the model (27 GB download). Writes "
                    "moss_check_report.txt and .json and every run's sample images to diagnostics/moss_check.")
    # the component checks run in child processes, which get the model and resolution through their environment
    parser.add_argument("--model", default=os.environ.get("OT_MOSS_CHECK_MODEL", moss_checks.DEFAULT_MODEL),
                        help=f"the MOSS-Image model: a Hugging Face repo or a local folder (default: "
                             f"{moss_checks.DEFAULT_MODEL})")
    parser.add_argument("--config", help="a training config (e.g. saved from the UI) to take the training settings "
                                         "from: optimizer, learning rate, LoRA, dtypes, EMA, masking. The runs still "
                                         "set the data, schedule, resolution and output themselves")
    parser.add_argument("--resolution", type=int, default=int(os.environ.get("OT_MOSS_CHECK_RESOLUTION", "512")),
                        help="training and sample resolution of the runs (default: 512)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true", help="also a short full fine-tune of the transformer")
    parser.add_argument("--keep", action="store_true",
                        help="keep the training runs (LoRAs, backups, logs), not only their sample images")
    parser.add_argument("--out-dir", default=path_util.diagnostics_path("moss_check"))
    parser.add_argument("--only", nargs="+", help="run only checks whose 'section / name' contains one of these")
    parser.add_argument("--skip", nargs="+", help="skip checks whose 'section / name' contains one of these")
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # the helpers moss_checks shares with rocm_check register rocm_check's own training checks too
    REGISTRY[:] = [c for c in REGISTRY if c.section in (moss_checks.SECTION_COMPONENTS, moss_checks.SECTION_TRAINING)]

    if args.list:
        for c in sorted(REGISTRY, key=lambda c: c.section):
            print(f"{c.section} / {c.name}" + ("  (--full only)" if c.full_only else ""))
        return 0

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        torch.cuda.set_device(device)
    ctx = Ctx(device=device, full=args.full, out_dir=out_dir, script_path=os.path.abspath(__file__),
              resolutions=[args.resolution], seed=args.seed)
    if args.child:
        ctx.cache.update({"moss_model": args.model, "moss_resolution": args.resolution})
        child_main(args.child, ctx)
        return 0

    error, note = moss_checks.preflight(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if error:
        print(f"moss_check: {error}", file=sys.stderr)
        return 2
    if note:
        print(note)

    runs = os.path.join(out_dir, "runs")
    if os.path.isdir(runs):
        import shutil
        shutil.rmtree(runs)
    if os.path.exists(moss_checks._text_path(ctx)):
        os.remove(moss_checks._text_path(ctx))  # the text encoder check writes it anew for the transformer check
    ctx.cache.update({"moss_model": args.model, "moss_config": os.path.abspath(args.config) if args.config else None,
                      "moss_resolution": args.resolution, "moss_runs": runs})

    report = Report(ctx, [os.path.basename(sys.executable)] + sys.argv, name="moss_check",
                    title="OneTrainer MOSS-Image check")
    run_checks(ctx, report, args.only, args.skip)
    note = moss_checks.finish(ctx, args.keep)
    print()
    print(report.text())
    print(f"report written to {report.txt_path} and {report.json_path}")
    if note:
        print(note)
    return 1 if any(r["status"] == FAIL for r in report.results) else 0


if __name__ == "__main__":
    sys.exit(main())
