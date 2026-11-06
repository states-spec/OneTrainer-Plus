from modules.util.tqdm_util import tqdm

import torch
import torch._dynamo.callback
import torch.utils._sympy.functions

# Mod.eval accepting negative number literals, to avoid https://github.com/Nerogar/OneTrainer/issues/1126. torch 2.12
# merged the fix PR https://github.com/pytorch/pytorch/pull/169726, but Mod.eval itself still raises on negative numbers
# (2.13 too), and inductor's tiling_utils still calls sympy is_constant() (which substitutes negative test values) on
# expressions that can contain Mod. Can be removed once Mod.eval accepts negative numbers in a torch version we use.
_original_mod_eval = torch.utils._sympy.functions.Mod.eval.__func__


@classmethod
def Mod_patched_eval(cls, p, q):
    # torch's own Mod.eval, except for two number literals with p < 0, where it raises AssertionError(p): those get
    # p % q, as the copy of it this used to be returned (torch 2.12 and 2.13 differ from that copy in this case only).
    # Wrapping it keeps whatever later torch versions change in the rest of the function.
    if q.is_Number and p.is_Number and p < 0 and q >= 1:
        return p % q
    return _original_mod_eval(cls, p, q)


# set by CompressedWeightMixin._compress_weight() when a weight is stored nvCOMP-compressed;
# gates the force_parameter_static_shapes relaxation in init_compile() below.
needs_dynamic_parameter_shapes = False


def reset_compile():
    # needs_dynamic_parameter_shapes is a process global, but only describes the current run. in the
    # long-lived UI process a compressed run would otherwise keep relaxing force_parameter_static_shapes
    # for every later run in the same session. call this once per run, before init_compile().
    global needs_dynamic_parameter_shapes
    needs_dynamic_parameter_shapes = False


_RECOMPILE_LIMIT = 8192


def init_compile():
    # since torch 2.12, dynamo config overrides are stored in ContextVars (pytorch/pytorch#173568), which threads don't
    # inherit. init_compile() is therefore called in every thread that can trigger a compilation - including the
    # autograd worker threads, which recompute checkpointed modules during the backward pass.
    # Every compiled block of every model shares one function (the checkpoint layer's forward), so all their variants
    # count against both limits: recompile_limit (8 by default) caps that function's variants, and
    # accumulated_recompile_limit (256) its compiles in the whole process. fullgraph raises at either. A UI session
    # reuses a config's compiled variants, but every other model type, adapter or data type adds its own.
    for name in ("recompile_limit", "accumulated_recompile_limit"):
        setattr(torch._dynamo.config, name, _RECOMPILE_LIMIT)
        # also as the default every thread sees, including the ones torch starts itself: on a GPU the backward runs in
        # an autograd worker thread, where diffusers' own checkpointing (SD, SDXL) runs the compiled blocks again
        entry = getattr(torch._dynamo.config, "_config", {}).get(name)
        if entry is not None and hasattr(entry, "default"):
            entry.default = _RECOMPILE_LIMIT

    # same thread-local-config problem: force_parameter_static_shapes defaults to True and would
    # override maybe_mark_dynamic on the compressed weight, so the reentrant-checkpoint backward
    # (running on autograd worker threads that don't see _compress_weight()'s main-thread setting)
    # recompiles per compressed length forever. only relax it when compression actually needs it.
    if needs_dynamic_parameter_shapes:
        torch._dynamo.config.force_parameter_static_shapes = False


def _on_compile_start(args: "torch._dynamo.callback.CallbackArgs") -> None:
    frame_id, _, frame_compile_id = args.compile_id.partition("/")
    direction = "backward" if args.callback_trigger == torch._dynamo.callback.CallbackTrigger.LAZY_BACKWARD else "forward"
    tqdm.show_status(f"compiling kernel {frame_id} {direction} (variant #{frame_compile_id or 0})...")


torch._dynamo.callback.on_compile_start(_on_compile_start)

torch.utils._sympy.functions.Mod.eval = Mod_patched_eval
