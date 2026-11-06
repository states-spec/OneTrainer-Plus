"""
The training seed (training tab: Training Seed, a mode and a value).

Off (the default) trains as before: the LoRA/embedding initialization comes from the unseeded global generators, and
each step's noise and timesteps from a generator seeded with the step number alone, the same for every run. Any other
mode seeds the global generators (Python, NumPy, torch) with the seed when training starts, and adds an offset of
seed * 2^32 to each step's seed, so a seed gives its own initialization, dropout and noise sequence, and two runs with
the same seed and settings start and step alike (bit-identical on the CPU; GPU kernels may round differently).
Validation keeps its fixed seed, so it compares runs on the same noise whatever their seed. The data order and the
augmentations stay seeded by each concept's own seed: they come before the latent cache.

The seed for a run is chosen where a run starts (the UI's Start Training, scripts/train.py), never in the trainer, so a
cloud run trains with the seed picked on this PC:
- Fixed: the value, every run.
- Increment / Decrement: the value; after the run the UI moves it one up / down for the next run.
- Random: a new value for each run, written into the field, so the run can be repeated with Fixed.
"""
import random

from modules.util.enum.TrainingSeedMode import TrainingSeedMode

import torch

MAX_TRAINING_SEED = (1 << 31) - 1


def _mode(mode) -> TrainingSeedMode:
    return mode if isinstance(mode, TrainingSeedMode) else TrainingSeedMode(str(mode or TrainingSeedMode.OFF))


def is_active(config) -> bool:
    return _mode(config.training_seed_mode) != TrainingSeedMode.OFF


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
    if is_active(config) and not 0 <= config.training_seed <= MAX_TRAINING_SEED:
        raise ValueError(f"Training Seed must be between 0 and {MAX_TRAINING_SEED}, not {config.training_seed}.")


def apply_training_seed(config):
    """seeds the global generators when training starts (GenericTrainer.start); nothing in mode Off"""
    if not is_active(config):
        return
    import numpy as np

    random.seed(config.training_seed)
    np.random.seed(config.training_seed)  # noqa: NPY002 (the legacy global generator is the one other code draws from)
    torch.manual_seed(config.training_seed)
    print(f"Training seed: {config.training_seed} ({str(config.training_seed_mode).capitalize()})")


def training_seed_offset(config) -> int:
    """added to each step's seed in every model setup's predict: 0 in mode Off, so the noise is as before"""
    return config.training_seed << 32 if is_active(config) else 0
