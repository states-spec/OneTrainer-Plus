def script_imports(allow_zluda: bool = True):
    import faulthandler
    import importlib.util
    import logging
    import os
    import re
    import sys
    import warnings
    from pathlib import Path

    # a crash in native code (segmentation fault in the GPU runtime, a kernel or a library) prints the Python stack of
    # every thread to stderr, instead of only the shell's "Segmentation fault" line. No cost while nothing crashes.
    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=True)

    if sys.platform == 'win32':
        # PyTorch's Windows allocator can retain freed loading/quantization buffers in mimalloc arenas.
        # Return unused pages promptly, including while the UI is idle. Set this before importing torch;
        # explicit user allocator settings take precedence. This does not change the CUDA allocator.
        os.environ.setdefault('MIMALLOC_PURGE_DELAY', '0')

    # Suppress the Triton startup warning and leftover xformers installations warnings.
    # Filter out the Triton warning on startup.
    # xformers is not installed anymore, but might still exist for some installations.
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    # Silence non-actionable startup/compile warnings unless OT_DEBUG_WARNINGS is set.
    # Filter emitting loggers directly; parent filters miss child records.
    if not os.environ.get("OT_DEBUG_WARNINGS"):
        # Suppress noisy diffusers/transformers import/load logs; regexes match runtime values.
        logging.getLogger("diffusers.configuration_utils").addFilter(
            lambda record: not re.search(
                r"The config attributes .* were passed to .*, but are not expected and will be ignored",
                record.getMessage(),
            )
        )
        logging.getLogger("diffusers.models.modeling_utils").addFilter(
            lambda record: 'Attention backends are an experimental feature' not in record.getMessage()
        )
        # logged when diffusers.modular_pipelines is imported (AnimaModel's AnimaAutoBlocks), i.e. at every start
        logging.getLogger("diffusers.modular_pipelines").addFilter(
            lambda record: 'Modular Diffusers is currently an experimental feature' not in record.getMessage()
        )
        logging.getLogger("transformers.modeling_utils").addFilter(
            lambda record: not re.search(
                r"`loss_type=.*` was set in the config but it is unrecognized", record.getMessage()
            )
        )

        # Qwen3.5 (MOSS's text encoder) says its "fast path is not available" unless both fla-core and causal-conv1d
        # are installed, but it takes each kernel separately: with fla-core, only the short convolution runs on
        # PyTorch's conv1d (causal-conv1d is a CUDA extension). Without fla-core the message stays (slow fallback).
        if importlib.util.find_spec("fla") is not None:
            logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").addFilter(
                lambda record: 'The fast path is not available' not in record.getMessage()
            )

        # A dependency still passes the deprecated local_dir_use_symlinks to hf_hub_download.
        warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")

        # diffusers' own Chroma transformer still imports FluxPosEmbed from its old place (printed at every load)
        warnings.filterwarnings("ignore", message=r"`FluxPosEmbed` is deprecated", category=FutureWarning)

        # Suppress harmless torch.compile fallback notes: complex operators use warnings.warn(),
        # while insufficient SMs use logger.warning().
        warnings.filterwarnings("ignore", message=r".*does not support code generation for complex operators.*")
        logging.getLogger("torch._inductor.utils").addFilter(
            lambda record: 'Not enough SMs to use max_autotune_gemm mode' not in record.getMessage()
        )

    # Every torch.load in the process refuses pickled code (weights_only=True), also where a library asks for
    # weights_only=False: mgds loads its cache files that way, so a planted or shared cache file ran code. Everything
    # OneTrainer+ loads with torch.load holds only tensors and plain values (caches, optimizer and EMA state, LoRAs,
    # embeddings, .ckpt models). PyTorch refuses this together with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, so a user who
    # set that one keeps it.
    if not os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")

    # Prioritize local modules to prevent shadowing; three parents reach the repo root.
    onetrainer_lib_path = Path(__file__).absolute().parent.parent.parent
    sys.path.insert(0, str(onetrainer_lib_path))

    # TunableOp (PYTORCH_TUNABLEOP_ENABLED=1) keeps its results with the other diagnostic output instead of the working
    # directory, unless PYTORCH_TUNABLEOP_FILENAME names a file. Set before torch is imported, which reads it.
    # PyTorch turns TunableOp on only for exactly "1" (it warns about and ignores "true" and the like).
    if os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1" and not os.environ.get("PYTORCH_TUNABLEOP_FILENAME"):
        from modules.util import path_util
        os.makedirs(path_util.diagnostics_path("tunableop"), exist_ok=True)
        os.environ["PYTORCH_TUNABLEOP_FILENAME"] = path_util.diagnostics_path("tunableop", "tunableop_results%d.csv")

    # torch.compile (Compile transformer blocks, Compiled Optimizer) keeps its compiled kernels in
    # $TMPDIR/torchinductor_<user> by default, and /tmp is cleared at every boot on many Linux systems (a RAM-backed
    # tmpfs on Fedora), so the first compiled run after each reboot compiled everything again (minutes on a large
    # model). A cache under ~/.cache survives that; TORCHINDUCTOR_CACHE_DIR still wins when set.
    if not sys.platform.startswith('win') and not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        inductor_cache = os.path.join(cache_home, "OneTrainer-Plus", "torchinductor")
        try:
            os.makedirs(inductor_cache, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
        except OSError:
            pass  # an unwritable home: PyTorch's own default

    if allow_zluda and sys.platform.startswith('win'):
        from modules.zluda import ZLUDAInstaller

        zluda_path = ZLUDAInstaller.get_path()

        if os.path.exists(zluda_path):
            try:
                ZLUDAInstaller.load(zluda_path)
                print(f'Using ZLUDA in {zluda_path}')
            except Exception as e:
                print(f'Failed to load ZLUDA: {e}')

            from modules.zluda import ZLUDA

            ZLUDA.initialize()

    # RDNA3 under ROCm: attention with heads above 256 (every VAE) goes to the math kernel, see rocm_sdpa_fix
    from modules.util import rocm_sdpa_fix
    rocm_sdpa_fix.install()

    # transformers before 5.10: a model's chat template names could write files outside the save folder (CVE-2026-9856)
    from modules.util import chat_template_fix
    chat_template_fix.install()
