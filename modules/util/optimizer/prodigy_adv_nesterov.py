"""
Nesterov momentum for adv_optm's Prodigy_adv.

Prodigy keeps its first moment in units of its step size D (exp_avg <- beta1 * exp_avg + (1 - beta1) * d * grad), but
Prodigy_adv's Nesterov step mixes in the raw gradient: update = exp_avg.lerp(grad, 1 - coef). That term is 1/d times
too large, so each step moved every weight by about (1 - coef) * lr whatever D was (adv_optm 2.5.13 to 2.6.0, and every
branch of Koratahiu/Advanced_Optimizers as of 2026-11-02): a Chroma LoRA at learning rate 1 turned to noise within the
first steps. The other adv_optm optimizers keep their momentum in gradient units and are not affected.

prodigy_adv_class() returns the class create_optimizer builds:
- adv_optm's own Prodigy_adv, if its Nesterov step is correct (a release that fixed it),
- otherwise a subclass whose _step_parameter is adv_optm's own source with the gradient term scaled by d (the two
  lerp lines, one per state layout), if both are found exactly,
- otherwise adv_optm's class, reported as broken: create.check_optimizer_config turns Nesterov off then.
A one-step probe decides whether a class is correct. With D held fixed, the fixed step equals AdamW_adv's Nesterov at
learning rate D * lr (rocm_check tests it), and with Nesterov off it is adv_optm's step, bit for bit.
"""
import functools
import inspect
import sys
import textwrap

import torch
from torch import nn

# adv_optm's lines (2.5.13 to 2.6.0) and their fix: the gradient term in D units, like exp_avg
_FIXES = (
    ("update_mt = update_mt.lerp_(grad_reshaped, 1-nv_coef)", "update_mt = update_mt.lerp_(grad_reshaped * d, 1-nv_coef)"),
    ("update_mt = update_mt.lerp_(grad, 1-nv_coef)", "update_mt = update_mt.lerp_(grad * d, 1-nv_coef)"),
)


def nesterov_step_is_correct(prodigy_class) -> bool:
    # one step from D = Initial D = 1e-6 at learning rate 1 with a constant gradient. A correct step moves each weight
    # by about 2e-6 (Nesterov's 0.19 * d * g over sqrt(v) = 0.1 * d * |g|, times d * lr); with the raw gradient term
    # it moves it by about 1
    p = nn.Parameter(torch.full((4, 4), 0.1))
    optimizer = prodigy_class([p], lr=1.0, d0=1e-6, betas=(0.9, 0.99), nesterov=True, stochastic_rounding=False)
    p.grad = torch.ones_like(p)
    optimizer.step()
    return (p.detach() - 0.1).abs().max().item() < 1e-3


def _patched_class(prodigy_class):
    module = sys.modules[prodigy_class.__module__]
    try:
        source = textwrap.dedent(inspect.getsource(prodigy_class._step_parameter))
    except (OSError, TypeError):
        return None
    for old, new in _FIXES:
        if source.count(old) != 1:
            return None
        source = source.replace(old, new)
    namespace = dict(vars(module))  # the function sees adv_optm's own helpers
    exec(compile(source, f"{module.__file__} (OneTrainer+: Nesterov term in D units)", "exec"), namespace)
    return type(prodigy_class.__name__, (prodigy_class,), {
        "_step_parameter": namespace["_step_parameter"],
        "__module__": __name__,
        "__doc__": prodigy_class.__doc__,
    })


@functools.cache
def prodigy_adv_class() -> tuple[type, bool]:
    """(the Prodigy_adv class to build, whether its Nesterov step is correct)"""
    from adv_optm import Prodigy_adv

    if nesterov_step_is_correct(Prodigy_adv):
        return Prodigy_adv, True
    patched = _patched_class(Prodigy_adv)
    if patched is not None and nesterov_step_is_correct(patched):
        return patched, True
    return Prodigy_adv, False
