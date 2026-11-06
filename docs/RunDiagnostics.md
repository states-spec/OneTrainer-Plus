# Run diagnostics: recording and comparing training runs

Three tools for finding out what a setting does to a real training run: how much GPU memory it takes, how
fragmented that memory gets, how fast the steps are, and whether the results change.

- `OT_DIAG_LOG=true` makes a training run write a diagnostics log.
- `./run-cmd.sh diag_report` summarizes one log or compares several.
- `./run-cmd.sh diag_ab` runs one config under several settings and compares the runs.

The GPU check (`./run-cmd.sh rocm_check`, see [GpuCheck.md](GpuCheck.md)) does the same with a tiny model, in a few
minutes and without your model or dataset. Use it first; use these tools for the real thing.

## Where the output goes

Everything these tools write goes into one folder, `diagnostics/` in the OneTrainer folder (git ignores it):

| folder or file | written by |
|---|---|
| `rocm_check/` | `rocm_check`: `rocm_check_report.txt` and `.json`, its TunableOp results and MIOpen database, and the tiny-Chroma training runs when that check fails (`--out-dir` changes it) |
| `diag_ab/<timestamp>/` | `diag_ab`: one folder per run (workspace, cache, output, `diag.jsonl`, `output.log`) and `report.txt` (`--out-dir` changes it) |
| `runs/<timestamp>.jsonl` | a training run with `OT_DIAG_LOG=true` |
| `tunableop/tunableop_results0.csv` | PyTorch's TunableOp (`PYTORCH_TUNABLEOP_ENABLED=1`), unless `PYTORCH_TUNABLEOP_FILENAME` is set; deleting it only costs a new tuning |
| `debug_report.log` | `generate_debug_report` |
| `stacks.txt` | the profiling window's stack dump |

Everything in it can be deleted; only the TunableOp results are read again, by later runs.

## The diagnostics log

```sh
OT_DIAG_LOG=true ./start-ui.sh                                   # or the CLI:
OT_DIAG_LOG=/path/run.jsonl ./run-cmd.sh train --config-path my_config.json
```

`true` writes `diagnostics/runs/<timestamp>.jsonl` in the OneTrainer folder; any other value is the file to write. Each line is one
record, written at once, so a crash or a GPU hang still leaves everything before it:

| record | when | what |
|---|---|---|
| `header` | start | versions, GPU, every `OT_`/`PYTORCH_`/`HIP_`/`ROCR_`/`HSA_`/`MIOPEN_`/`TORCH_` environment variable, whether MemPools are used, the whole config (without secrets) |
| `step` | every optimizer step | loss, seconds since the previous step, memory now and its peak since the last record |
| `phase` | loading, model setup, each epoch start (caching), sampling, validation, backup, save, final save | duration, memory before and after, the peak inside it, fragmentation after it |
| `move` | a model part loaded onto or unloaded from the GPU | parts, duration, memory and fragmentation after it |
| `output` | after the final save | a digest of the saved weights: two runs that saved the same weights have the same one |
| `error` | the exception that ended the run, out of memory included | message and the memory state at that moment |

The memory numbers come from PyTorch's allocator: memory in tensors (`allocated`), memory PyTorch holds (`reserved`),
their peaks, free space inside partly used blocks (`inactive_split`), how often the allocator had to free its cache
and retry an allocation (`alloc_retries`), out-of-memory errors (`ooms`), GPU allocations and releases, free memory on
the device, and, after phases and part moves, the number of memory segments, the largest free hole, how fragmented
the free memory is, and how much sits in MemPools. With amdsmi (ROCm) or pynvml (NVIDIA) installed, the GPU's
temperature, clock and power are recorded too (after phases and part moves, and every 20th step): a run that ran hotter
and clocked lower has slower steps for that reason alone. Writing the log costs nothing noticeable per step; the
fragmentation figures read the allocator's segment list, which is done only after phases and part moves.

## Comparing runs: `diag_report`

```sh
./run-cmd.sh diag_report run_a.jsonl run_b.jsonl [more.jsonl ...] [--json numbers.json]
```

The first log is the reference. The report shows:

- each run's status: finished, an error (out of memory named), or incomplete (crashed, hung or killed);
- the settings that differ between the runs (environment and config; the per-run folders are left out);
- per run: steps, mean loss, median and 90th percentile step time (the first two steps, and steps with sampling,
  backups or part moves in between, don't count), peak RAM, and on a GPU: peak allocated and reserved memory, the
  most memory held without tensors, fragmentation, inactive split memory, MemPool memory, allocation retries, OOMs,
  GPU allocations and releases, part moves and their time;
- per phase: how often, how long, its peak reserved memory, in brackets how much the phase itself added to what
  was reserved before it (the peak also counts the cache training left behind), and the most memory held unused
  after it;
- for each run against the first: where the losses part (the difference at that first step, the mean and the
  largest, and both mean losses), and whether the saved weights are bit-identical (if not, the largest and the median
  relative difference of a tensor, read from the saved files; LoRA B tensors start at zero, so theirs is the largest);
- for repeated runs of the same variant: the same comparison among them, which is the run-to-run noise to judge the
  differences by, and each variant's mean, minimum and maximum.

## Running an A/B: `diag_ab`

```sh
./run-cmd.sh diag_ab --config-path my_config.json --config-value epochs=1
```

Without `--variant` this is the MemPool comparison: `baseline` against `OT_ROCM_MEM_POOLS=true`, twice each, in the
order baseline, MemPools, baseline, MemPools (alternating spreads drift, such as GPU temperature, over both). Each run
gets its own workspace, cache and output folder under `diagnostics/diag_ab/<timestamp>/`, so every run loads, caches and trains
the same way, and `OT_SEED=42` (unless you set another), which replaces the training seed of every run, so every run starts
from the same LoRA and steps with the same noise. At the end it prints
the report and writes it to `report.txt` in that folder.

Other comparisons: `--variant 'NAME env:KEY=VALUE config:key=value ...'`, as often as needed:

```sh
./run-cmd.sh diag_ab --config-path my_config.json --variant baseline --variant 'nostream config:stream_from_disk=false'
./run-cmd.sh diag_ab --config-path my_config.json \
    --variant 'baseline env:OT_DETERMINISTIC=true' \
    --variant 'mempools env:OT_DETERMINISTIC=true env:OT_ROCM_MEM_POOLS=true'
```

`--config-value` applies to every run (shorten the run with `epochs=1`, as with `train`), `--repeats` sets the runs per
variant (default 2), `--out-dir` the folder, `--preset-path` a built-in preset applied first.

`python:PATH` runs a variant with another environment's interpreter: this checkout's code with that venv's PyTorch and
ROCm build. That compares two builds on your real workload, in alternating order:

```sh
./run-cmd.sh diag_ab --config-path my_config.json --config-value epochs=1 \
    --variant 'rocm72 python:~/OT_KP2/venv/bin/python' \
    --variant 'rocm101 python:~/OT_ROCm10/venv/bin/python'
```

The other venv needs OneTrainer's requirements from a compatible commit (each log records the interpreter, torch and
the main package versions, and the report lists those that differ). Its first run fills that build's kernel caches
(MIOpen, Triton, TunableOp if on), so compare the repeats.

### Reading an A/B

- **Correctness first.** Two runs of the same variant show the noise. A setting that only moves memory around (MemPools,
  streaming, offloading) should leave the losses and weights within that noise. On the CPU, and on a GPU with
  deterministic kernels, the same seed gives bit-identical runs, and so should such a setting. GPU kernels that sum in
  a varying order make every run a little different; `OT_DETERMINISTIC=true` on both variants removes most of that.
- **Offloading variants need a fixed start.** A new LoRA layer is initialized on the device its base layer is on, so
  with layer offloading the offloaded layers draw from the CPU's random generator instead of the GPU's, and the runs
  start from different LoRAs even with `OT_SEED`. To compare offloading settings, save a starting LoRA first (one run
  with `learning_rate=0`) and give every variant `config:lora_model_name=<that file>`, as `rocm_check` does.
- **Memory.** The peak reserved memory decides whether a run fits. "Held unused" is memory PyTorch keeps without
  tensors: it can't hold a model part that needs one large piece when it is fragmented ("frag" near 1). Allocation
  retries mean the allocator ran out and had to free its cache, usually just before an out-of-memory error, or
  instead of one.
- **Speed.** The step time median, and the time part moves take (sampling, backups and caching load and unload parts).

Compare runs of the same commit and the same config: the report lists everything that differs.

## Terminal messages that look like errors

- `HIPCachingAllocator.cpp: memory allocation failed with OOM on device 0 while trying to allocate N bytes (free: ...)`
  (`CUDACachingAllocator.cpp` on NVIDIA), while the run goes on: PyTorch 2.13 prints this at every failed attempt to
  reserve GPU memory, also the first one. After it, the allocator frees the memory it keeps cached but unused and
  tries again; MIOpen convolutions fall back to an algorithm with a smaller workspace. Only when that fails too does
  the run stop with an out-of-memory error. The message says that one large block (often a convolution workspace in
  the VAE, or a layer-offloading buffer) didn't fit next to the cache; `OT_LOG_VRAM=true` shows the memory around it.
  Freeing the cache costs a device sync, so a run that prints it often is a little slower, and in the diagnostics log
  it shows up as allocator retries.
