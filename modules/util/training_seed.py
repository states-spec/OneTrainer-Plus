"""
The training seed (training tab: Training Seed, a mode and a value). Everything random in a training run comes from
this one number:
- the global generators (Python, NumPy, torch on every device), seeded with it when training starts: the LoRA and
  embedding initialization, dropout, OFT's power-iteration start, adv_optm's stochastic rounding;
- each step's generator (noise, timesteps, offset noise, CIOP/CEP, Immiscible noise) in every model setup's predict,
  seeded with the step (and the GPU's rank) + a 32-bit hash of the seed. torch's CPU generator keeps only the low 32
  bits of a seed, so the seed's part has to be in them: an earlier seed * 2^32 left every CPU per-step generator the
  same whatever the seed. A step's seed is distinct within a run (the step + a constant, modulo 2^32);
- the bf16 stochastic rounding of OneTrainer's own optimizer extensions, the EMA and multi-GPU gradient reduction,
  seeded the same way at each step;
- adv_optm's own generators (stochastic rounding of bf16 weights and bf16_sr/int8_sr states, SignSGD's stochastic
  sign, Muon's low-rank orthogonalization, spectral normalization's start), which adv_optm seeds with 42 once per
  process: reseeded the same way at each step, so they follow the seed, a second run in the same UI session starts
  where the first did, and a resumed run continues where the backup left off.
Two runs with the same seed and settings start and step alike (bit-identical on the CPU and on the 7900 XTX).
Validation keeps its own fixed seed, so it compares runs on the same noise. The data order and the augmentations
follow each concept's own seed (fixed values in the concept file): they come before the latent cache. OT_SEED (run
diagnostics, rocm_check, diag_ab) replaces the training seed for the whole run.

The seed for a run is chosen once, where a run starts (the UI's Start Training, scripts/train.py), never in the
trainer, so a cloud run trains with the seed picked on this PC:
- Fixed (the default, 42): the value, every run.
- Increment / Decrement: the value; after the run the UI moves it one up / down for the next run.
- Random: one new value per run, drawn once and written into the field before the run starts, so the run can be
  repeated with Fixed; everything above derives from that one value.
"""
import random

from modules.util.enum.TrainingSeedMode import TrainingSeedMode

import torch

MAX_TRAINING_SEED = (1 << 31) - 1


def _mode(mode) -> TrainingSeedMode:
    return mode if isinstance(mode, TrainingSeedMode) else TrainingSeedMode[str(mode or TrainingSeedMode.FIXED)]


def seed_for_new_run(mode, seed: int) -> int:
    """the seed a run that starts now trains with"""
    if _mode(mode) == TrainingSeedMode.RANDOM:
        return random.SystemRandom().randint(0, MAX_TRAINING_SEED)
    return seed


def seed_after_run(mode, seed: int) -> int:
    """the value shown for the next run, once a run has used this one"""
    mode = _mode(mode)
    if mode == TrainingSeedMode.INCREMENT:
        return 0 if seed >= MAX_TRAINING_SEED else seed + 1
    if mode == TrainingSeedMode.DECREMENT:
        return MAX_TRAINING_SEED if seed <= 0 else seed - 1
    return seed


def check_training_seed(config):
    if not 0 <= config.training_seed <= MAX_TRAINING_SEED:
        raise ValueError(f"Training Seed must be between 0 and {MAX_TRAINING_SEED}, not {config.training_seed}.")


def apply_training_seed(config):
    """seeds the global generators when training starts (GenericTrainer.start). OT_SEED, if set, replaces the seed for
    the whole run (the per-step seeds read config.training_seed too), so a run never mixes two seeds"""
    import os

    import numpy as np

    env_seed = os.environ.get("OT_SEED", "").strip()
    if env_seed:
        config.training_seed = int(env_seed)
        print(f"OT_SEED: the training seed for this run is {config.training_seed}")
    check_training_seed(config)

    random.seed(config.training_seed)
    np.random.seed(config.training_seed)  # noqa: NPY002 (the legacy global generator is the one other code draws from)
    torch.manual_seed(config.training_seed)
    print(f"Training seed: {config.training_seed} ({str(config.training_seed_mode).capitalize()})")


def training_seed_offset(config) -> int:
    """added to each step's seed: every model setup's predict, the bf16 stochastic rounding generator and adv_optm's.
    A 32-bit hash of the seed (SplitMix64's finalizer), so it lands in the bits every generator reads"""
    x = (config.training_seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFF


def _exact_device(device: torch.device) -> torch.device:
    # adv_optm keys its generators by a tensor's device, which always has an index ("cuda:0", not "cuda")
    if device.type != "cpu" and device.index is None:
        return torch.device(device.type, torch.cuda.current_device() if device.type == "cuda" else 0)
    return device


_adv_optm_warned = False


def seed_adv_optm_generators(step_seed: int, train_device: torch.device):
    """reseeds adv_optm's per-device generators (GenericTrainer.train, every step, with the step's seed); nothing if
    adv_optm isn't loaded. They are adv_optm internals (adv_optm.util.param_update._generators, a dict of device ->
    torch.Generator, 2.5-2.6.3): if a later adv_optm moves them, this warns once instead of doing nothing silently
    (rocm_check section 3 "adv_optm's random generators follow the training seed" fails then too)"""
    import sys

    global _adv_optm_warned
    if "adv_optm" not in sys.modules:
        return
    param_update = sys.modules.get("adv_optm.util.param_update")
    generators = getattr(param_update, "_generators", None)
    if not isinstance(generators, dict):
        if not _adv_optm_warned:
            _adv_optm_warned = True
            version = getattr(sys.modules["adv_optm"], "__version__", "unknown")
            print(f"Warning: adv_optm {version} keeps its random generators somewhere OneTrainer+ doesn't know, so "
                  f"the training seed doesn't reach its stochastic rounding (and SignSGD's stochastic sign, Muon's "
                  f"low-rank orthogonalization). Training works, but runs with the same seed may differ there. "
                  f"modules/util/training_seed.py needs an update for this adv_optm.")
        return
    for device in {_exact_device(torch.device(train_device)), torch.device("cpu")}:
        if device not in generators:
            generators[device] = torch.Generator(device=device)
        generators[device].manual_seed(step_seed)
