from enum import Enum


class TrainingSeedMode(Enum):
    FIXED = 'FIXED'
    INCREMENT = 'INCREMENT'
    DECREMENT = 'DECREMENT'
    RANDOM = 'RANDOM'
    # removed mode (unseeded initialization, the same noise for every run): an alias, so a config saved with it loads
    # as Fixed; list(TrainingSeedMode) leaves aliases out, so the UI doesn't offer it
    OFF = 'FIXED'  # noqa: PIE796 (the alias is the point)

    def __str__(self):
        return self.value
