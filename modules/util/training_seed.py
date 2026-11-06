"""
The training seed (training tab: Training Seed, a mode and a value). Everything random in a training run comes from
this one number:
- the global generators (Python, NumPy, torch on every device), seeded with it when training starts: the LoRA and
  embedding initialization, dropout, OFT's power-iteration start, adv_optm's stochastic rounding;
- each step's generator (noise, timesteps, offset noise, CIOP/CEP, Immiscible noise) in every model setup's predict,
  seeded with seed * 2^32 + the step (and the GPU's rank);
- the bf16 stochastic rounding of OneTrainer's own optimizer extensions, the EMA and multi-GPU gradient reduction,
  seeded the same way at each step.
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
    """added to each step's seed: every model setup's predict, and the bf16 stochastic rounding generator"""
    return config.training_seed << 32
