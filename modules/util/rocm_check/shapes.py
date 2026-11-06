"""Tensor shapes of the model families OneTrainer trains, for the speed and correctness checks.

Approximate on purpose: the checks measure the kernels a family's attention and MLP layers hit (head count, head size,
sequence length, hidden size), not the model itself. Head sizes cover the odd ones too (40 for SD1.5, 72 for PixArt),
because attention kernels have size-specific paths. Add a family here when a new model type has a different shape.
"""
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str
    heads: int
    head_dim: int
    hidden: int
    mlp: int  # the widest MLP projection
    image_tokens: Callable[[int], int]  # tokens of a resolution x resolution image in the largest attention layer
    text_tokens: int  # text tokens joined to the image tokens (0 for cross-attention-only models)

    def seq(self, resolution: int) -> int:
        return self.image_tokens(resolution) + self.text_tokens


FAMILIES = [
    Family("SD1.5/SD2 UNet (64x64 level)", 8, 40, 320, 2560, lambda r: (r // 8) ** 2, 0),
    Family("SDXL UNet (2nd level)", 10, 64, 640, 5120, lambda r: (r // 16) ** 2, 0),
    Family("PixArt/Sana-like DiT", 16, 72, 1152, 4608, lambda r: (r // 16) ** 2, 0),
    Family("SD3/SD3.5 Medium MMDiT", 24, 64, 1536, 6144, lambda r: (r // 16) ** 2, 333),
    Family("Flux/Chroma/Qwen-Image MMDiT", 24, 128, 3072, 12288, lambda r: (r // 16) ** 2, 512),
    Family("Z-Image DiT", 30, 128, 3840, 10240, lambda r: (r // 16) ** 2, 256),
    # the fused up/gate projection is the widest (2 x 8192); the text is a chat-templated prompt
    Family("MOSS-Image DiT", 24, 128, 3072, 16384, lambda r: (r // 16) ** 2, 256),
]

HEAD_DIMS = sorted({f.head_dim for f in FAMILIES})
