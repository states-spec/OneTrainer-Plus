from enum import Enum


class TrainingSeedMode(Enum):
    OFF = 'OFF'
    FIXED = 'FIXED'
    INCREMENT = 'INCREMENT'
    DECREMENT = 'DECREMENT'
    RANDOM = 'RANDOM'

    def __str__(self):
        return self.value
