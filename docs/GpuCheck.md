# GPU check (`scripts/rocm_check.py`)

Checks and measures what OneTrainer runs on your GPU, without a model download or a dataset. It was made for AMD
GPUs on ROCm, where kernels are less tested than on CUDA, but it runs on CUDA too, and partly on the CPU.

```sh
./run-cmd.sh rocm_check            # quick mode, about 10 minutes (half of it the tiny-model training runs)
./run-cmd.sh rocm_check --full     # more dtypes, sizes and optimizers, plus TunableOp, torch.compile and training features (45-60 min)
./run-cmd.sh rocm_check --list     # the checks
./run-cmd.sh rocm_check --only sdpa offloading     # checks whose "section / name" contains one of these
./run-cmd.sh rocm_check --resolutions 512 768 1024 # sizes for the speed checks (default 512 1024)
./run-cmd.sh rocm_check --only "training run"      # the tiny-model training runs and component checks (section 5)
./run-cmd.sh rocm_check --only vae --vae /path/to/model   # the VAE checks with a real VAE (model folder with vae/,
                                                          # a VAE folder, or a .safetensors file holding a VAE, e.g.
                                                          # ComfyUI models/vae/ae.safetensors; read offline)
```

It writes `rocm_check_report.txt` (for reading) and `rocm_check_report.json` (for comparing runs) to
`diagnostics/rocm_check/` in the OneTrainer folder, or to `--out-dir`. The files are rewritten after every check, so a crash or a GPU hang still leaves the
results up to the check that was running (the report names it). The exit code is 1 if any check failed.

## What it checks

1. **Environment**: PyTorch and its ROCm/CUDA build, the GPU (name, gfx arch, VRAM; every visible GPU, so an integrated one shows up), the BLAS and flash attention
   libraries, Triton, bitsandbytes, package versions and the GPU environment variables. It warns when AOTriton
   treats the GPU as experimental (then SDPA silently uses its slow math kernel), and reports whether the float W8A8
   weight types can run.
2. **Kernel correctness**, each against a float64 reference on the CPU:
   - matmul/linear forward and backward in fp32/bf16/fp16 (hipBLASLt or cuBLAS)
   - int8 matmul (`torch._int_mm`) and OneTrainer's Triton 8-bit kernel (the W8A8 backward), W8A8 layers, and the
     bfloat16 A8 layers (timed against a bf16 layer that also computes the weight gradient)
   - scaled dot product attention per backend (flash, memory-efficient and math, PyTorch's kernels called directly)
     and the call OneTrainer makes, per head size used by the model families and the VAEs (up to 512), with no mask, key-padding masks and the query x key mask whose padded rows
     have no valid key (NaN there would spread through the model)
   - which attention backend PyTorch picks per head size, dtype and mask, and per model family
   - convolution, group norm and upsampling (the VAE's ops, MIOpen on ROCm)
   - bitsandbytes: NFLOAT_4 and INT_8 layers, 8-bit AdamW
   - bf16 stochastic rounding
3. **Training code** on the GPU:
   - LoRA, DoRA, LoKr (also with DoRA and the vec trick), LoHa and OFT v2 (also with spectral norm clipping), forward and backward under bf16 autocast
   - optimizers built by OneTrainer's own `create_optimizer` (ADV optimizers with every state precision, AdamW,
     schedule-free ones), with a save/resume round trip like a backup
   - every optimizer with the UI's default settings: created, stepped, saved and resumed like a backup (the resumed
     run has to match the uninterrupted one), and the fused back pass where it is supported (the bitsandbytes
     optimizers only on the GPU)
   - timestep sampling with every distribution, wide noising weights/biases and shifts: no sample outside the
     noise schedule (one past its end is a device-side assert, which aborts the process under ROCm)
   - layer and activation offloading through the real `LayerOffloadConductor`: same loss and gradients as without
     offloading, only the loaded layers on the GPU after loading, weights bit-identical after load/unload cycles
   - disk streaming (`stream_from_disk`, on by default): a part streamed from a `.safetensors` file through the model
     loaders' own streaming code, against the same part loaded into RAM and moved to the GPU: the same weights
     (bf16 and fp8), loss and gradients, whole and layer by layer under offloading (simplex too), with Cache In RAM
     on and off, over three unload/reload cycles
   - pinned host memory and transfer speed
   - concurrent VAE encodes, as caching with `dataloader_threads` > 1 does, with and without the lock OneTrainer
     now uses (on ROCm, unlocked concurrent encodes once gave random NaN latents)
   - the VAE encoder layer by layer against a CPU reference, in the main thread and in worker threads (where
     caching runs it), and with MIOpen off or the math attention kernel: it names the first layer that goes wrong
     and which switch avoids it
4. **Speed**: attention (per backend) and linear layers per model family and resolution, VAE convolutions, and the
   runtime switches that can only be set at startup, each in a fresh process (and each one's linear layers checked
   against an fp64 reference, so a faster but wrong setting fails): the BLAS library PyTorch doesn't pick by default
   (hipBLASLt on an RX 7900 XTX, whose default is rocBLAS),
   TunableOp, MIOpen's find mode, the allocator with `expandable_segments` (its memory use, and the conv, VAE
   and, with `--full`, offloading checks run again under each allocator; wrong results under `expandable_segments`
   fail the check only when your setup uses it, and warn otherwise), the offloading and disk-streaming
   checks and a model part move with and without dedicated MemPools (`OT_ROCM_MEM_POOLS`), and `torch.compile`.

5. **Training run (tiny Chroma)**: a Chroma model with random weights and a small masked, captioned dataset, built
   offline in a few seconds, trained through the real `scripts/train.py` with the main workload's settings (the Chroma
   LoRA preset, masked training, EMA, ADOPT_ADV, no warmup, sampling and backups every epoch). Every run is seeded
   (`OT_SEED`) and writes a diagnostics log ([RunDiagnostics.md](RunDiagnostics.md)). The baseline runs twice (the
   run-to-run noise), then once per variant that changes how memory is handled: MemPools (`OT_ROCM_MEM_POOLS`, with
   and without layer offloading), disk streaming off, layer and activation offloading, and resuming from a backup
   (stopped after two of the three epochs); `--full` adds cache in RAM off, simplex offloading, synchronous
   offloading and deterministic algorithms. Each variant's saved LoRA and losses are compared with the baseline's,
   against the noise; it also reports each run's peak and unused memory, allocation retries and part moves. And the
   model's components on the GPU against the CPU, with the same inputs: the text encoder (captions of two lengths, so
   the attention is masked), the VAE both ways, and the transformer with a LoRA forward and backward, in fp32 and in
   bf16 under autocast. With `--full`, the same tiny model also trains once per feature (about 35 runs): other
   optimizers (Prodigy_adv, Muon, AdaMuon with the auxiliary Adam, SinkSGD on OFT next to an additional embedding,
   Adopt with scaled weight decay), every LoRA output format and adapter type (LoRA, DoRA, LoHa, LoKr, OFT), an
   additional embedding, embedding training, fine-tuning, validation, prior prediction, masked prior preservation with
   an image that has no mask, other losses and noise options, gradient accumulation with the fused back pass, a
   restarting scheduler, saves every step with EMA, float8 weights. Each run has to finish
   with finite weights, and each saved LoRA is loaded back as the starting LoRA of a run at learning rate 0, which has
   to save exactly the same tensors (so no layer, adapter part or bundled embedding is lost or misplaced on loading).
   When a run fails, the folder with every run is kept (the report names it), each with its whole output in
   `diag.log`. A run that crashes in native code (ROCm aborts the process on a GPU fault or a failed assert in a
   kernel: exit code -6) shows the runtime's message and where its threads were in the report line.

## Reading the results

- `PASS`: correct within the tolerance shown. `FAIL`: wrong result, NaN/inf or an error. `WARN`: works, but
  something is off (a slow fallback, concurrent encodes breaking). `INFO`: measurements only. `SKIP`: needs a GPU or
  `--full`.
- Correctness numbers are relative errors: the norm of the difference divided by the norm of the reference. bf16
  results land around 1e-3 to 1e-2, fp32 around 1e-7.
- Speed ratios are relative to the default settings; above 1 is faster.
- The first time Triton runs (torch.compile, the 8-bit matmul kernels), it builds a small HIP helper with your C
  compiler, and the compiler's warnings are printed in between the checks: for example
  `warning: '_POSIX_C_SOURCE' redefined` from `hip_utils.c`, where the Python headers (here Anaconda's
  Python 3.13) and a newer glibc define it differently. They are not errors, the check's own line (`-> INFO`,
  `-> PASS`) is the result, and Triton keeps the built helper in `~/.triton/cache`, so it doesn't build it again.

## Comparing PyTorch/ROCm builds

Run the full check in each environment (each its own clone and venv, the same OneTrainer commit), then put the reports
side by side; the first is the reference:

```sh
./run-cmd.sh rocm_check_compare ~/OT_KP2/diagnostics/rocm_check/rocm_check_report.json \
    ~/OT_ROCm10/diagnostics/rocm_check/rocm_check_report.json
```

It lists each report's PyTorch build and commit, every check's status side by side (`*` where they differ), the
problems, and every speed number with its speed relative to the first report, plus a geometric mean per check (above 1
= faster; attention uses PyTorch's own backend choice only). It writes the same to
`diagnostics/rocm_check/rocm_check_compare.txt`. One report per build is one sample: the run-time switches' child
processes vary between runs, and the first run in a new environment fills its kernel caches (first-call and compile
times). For the real workload, train your config in both environments with `diag_ab` and `python:` variants
([RunDiagnostics.md](RunDiagnostics.md)).

## Adding to it

- A new model family with other attention or MLP sizes: add a row to `modules/util/rocm_check/shapes.py`.
- A new check: a function `(ctx, rec)` with the `@check(section, name)` decorator in one of the modules under
  `modules/util/rocm_check/`. `rec.expect(ok, text)` records a pass/fail line, `rec.warn`/`rec.info`/`rec.metric`
  the rest; exceptions and out-of-memory errors are caught and reported as failures. A setting that PyTorch or ROCm
  reads only at startup needs a `@child_task` and `run_child(ctx, task, env)`.
