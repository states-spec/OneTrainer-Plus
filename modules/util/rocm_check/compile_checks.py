"""Compile transformer blocks on every model family OneTrainer compiles, without model files or configs.

Each family's denoiser class (diffusers) is built with random weights: a tiny config for the correctness part, and on a
GPU also a config at the family's real width (few blocks) for the speed part. The blocks are wrapped by OneTrainer's
own `enable_checkpointing_for_*` function, as training does (gradient checkpointing on, compile on or off), a LoRA
(rank 4) is hooked into every linear layer, and one training step (forward under the train dtype's autocast, backward
into the LoRA) runs compiled and eager from the same weights and inputs.
"""
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass

from modules.util.rocm_check.framework import Ctx, Rec, check, finite, rel_err

import torch
from torch import nn

SECTION = "6 compile per model family"


def _randn(shape, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=gen)


def _ids(tokens: int, h: int, w: int, axes: int = 3) -> torch.Tensor:
    # flux-style position ids: text rows all 0, image rows (0, y, x)
    ids = torch.zeros(tokens, axes)
    if h * w:
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        ids[-h * w:, axes - 2] = ys.flatten().float()
        ids[-h * w:, axes - 1] = xs.flatten().float()
    return ids


# --- families: build(real) -> (model, inputs(gen) -> dict). `real` gives the family's real width with few blocks ---

def _sd15(real: bool):
    from diffusers import UNet2DConditionModel
    if real:  # diffusers' defaults are SD1.5's UNet, but for the text width
        model, size, cross, tokens = UNet2DConditionModel(sample_size=64, cross_attention_dim=768), 64, 768, 77
    else:
        model = UNet2DConditionModel(
            sample_size=16, block_out_channels=(32, 64), layers_per_block=1, cross_attention_dim=32,
            attention_head_dim=8, norm_num_groups=32,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"))
        size, cross, tokens = 16, 32, 8
    return model, lambda g: {"sample": _randn((1, 4, size, size), g), "timestep": torch.tensor([500]),
                             "encoder_hidden_states": _randn((1, tokens, cross), g)}


def _sdxl(real: bool):
    from diffusers import UNet2DConditionModel
    common = {"down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
              "up_block_types": ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
              "addition_embed_type": "text_time", "use_linear_projection": True}
    if real:
        model = UNet2DConditionModel(
            sample_size=128, block_out_channels=(320, 640, 1280), layers_per_block=2, cross_attention_dim=2048,
            transformer_layers_per_block=(1, 2, 10), attention_head_dim=(5, 10, 20), addition_time_embed_dim=256,
            projection_class_embeddings_input_dim=2816, **common)
        size, cross, tokens, pooled = 64, 2048, 77, 1280
    else:
        model = UNet2DConditionModel(
            sample_size=16, block_out_channels=(32, 32, 64), layers_per_block=1, cross_attention_dim=32,
            transformer_layers_per_block=(1, 1, 2), attention_head_dim=(2, 2, 4), addition_time_embed_dim=8,
            projection_class_embeddings_input_dim=8 * 6 + 32, norm_num_groups=32, **common)
        size, cross, tokens, pooled = 16, 32, 8, 32
    return model, lambda g: {"sample": _randn((1, 4, size, size), g), "timestep": torch.tensor([500]),
                             "encoder_hidden_states": _randn((1, tokens, cross), g),
                             "added_cond_kwargs": {"text_embeds": _randn((1, pooled), g),
                                                   "time_ids": torch.tensor([[size * 8.0, size * 8, 0, 0,
                                                                              size * 8, size * 8]])}}


def _pixart(real: bool):
    from diffusers import Transformer2DModel
    common = {"attention_bias": True, "activation_fn": "gelu-approximate", "norm_type": "ada_norm_single",
              "norm_elementwise_affine": False, "norm_eps": 1e-6, "use_additional_conditions": False}
    if real:
        model = Transformer2DModel(sample_size=64, num_layers=2, patch_size=2, attention_head_dim=72,
                                   num_attention_heads=16, in_channels=4, out_channels=8, cross_attention_dim=1152,
                                   caption_channels=4096, num_embeds_ada_norm=1000, **common)
        size, caption, tokens = 64, 4096, 120
    else:
        model = Transformer2DModel(sample_size=8, num_layers=3, patch_size=2, attention_head_dim=2,
                                   num_attention_heads=2, in_channels=4, out_channels=8, cross_attention_dim=8,
                                   caption_channels=None, num_embeds_ada_norm=8, **common)
        size, caption, tokens = 8, 8, 8
    return model, lambda g: {"hidden_states": _randn((1, 4, size, size), g), "timestep": torch.tensor([500]),
                             "encoder_hidden_states": _randn((1, tokens, caption), g),
                             "added_cond_kwargs": {"aspect_ratio": None, "resolution": None}}


def _sana(real: bool):
    from diffusers import SanaTransformer2DModel
    if real:  # Sana 1600M, 1024 px (32x32 latent of the DC-AE)
        model = SanaTransformer2DModel(patch_size=1, in_channels=32, out_channels=32, num_layers=2,
                                       attention_head_dim=32, num_attention_heads=70, num_cross_attention_heads=20,
                                       cross_attention_head_dim=112, cross_attention_dim=2240, caption_channels=2304,
                                       mlp_ratio=2.5, sample_size=32)
        size, channels, caption, tokens = 32, 32, 2304, 300
    else:
        model = SanaTransformer2DModel(patch_size=1, in_channels=4, out_channels=4, num_layers=3, attention_head_dim=4,
                                       num_attention_heads=2, num_cross_attention_heads=2, cross_attention_head_dim=4,
                                       cross_attention_dim=8, caption_channels=8, sample_size=16)
        size, channels, caption, tokens = 16, 4, 8, 8
    return model, lambda g: {"hidden_states": _randn((1, channels, size, size), g), "timestep": torch.tensor([500]),
                             "encoder_hidden_states": _randn((1, tokens, caption), g)}


def _sd3(real: bool):
    from diffusers import SD3Transformer2DModel
    if real:  # SD3.5 Medium's block types (dual attention, RMS norm), 2 blocks
        model = SD3Transformer2DModel(sample_size=128, patch_size=2, in_channels=16, num_layers=2,
                                      attention_head_dim=64, num_attention_heads=24, joint_attention_dim=4096,
                                      caption_projection_dim=1536, pooled_projection_dim=2048, out_channels=16,
                                      pos_embed_max_size=384, dual_attention_layers=(0,), qk_norm="rms_norm")
        size, channels, joint, pooled, tokens = 64, 16, 4096, 2048, 333
    else:
        model = SD3Transformer2DModel(sample_size=16, patch_size=1, in_channels=4, num_layers=3, attention_head_dim=8,
                                      num_attention_heads=4, caption_projection_dim=32, joint_attention_dim=32,
                                      pooled_projection_dim=64, out_channels=4, pos_embed_max_size=96,
                                      dual_attention_layers=(0,), qk_norm="rms_norm")
        size, channels, joint, pooled, tokens = 16, 4, 32, 64, 16
    return model, lambda g: {"hidden_states": _randn((1, channels, size, size), g), "timestep": torch.tensor([500]),
                             "encoder_hidden_states": _randn((1, tokens, joint), g),
                             "pooled_projections": _randn((1, pooled), g)}


def _flux_like(cls_name: str, real: bool):
    import diffusers
    cls = getattr(diffusers, cls_name)
    chroma = cls_name.startswith("Chroma")
    if real:  # Flux.1 / Chroma width, 512 px (32x32 tokens), 2 double + 2 single blocks
        kwargs = {"patch_size": 1, "in_channels": 64, "num_layers": 2, "num_single_layers": 2,
                  "attention_head_dim": 128, "num_attention_heads": 24, "joint_attention_dim": 4096,
                  "axes_dims_rope": (16, 56, 56)}
        hw, channels, joint, pooled, tokens = 32, 64, 4096, 768, 256
    else:
        kwargs = {"patch_size": 1, "in_channels": 4, "num_layers": 2, "num_single_layers": 2, "attention_head_dim": 16,
                  "num_attention_heads": 2, "joint_attention_dim": 32, "axes_dims_rope": (4, 4, 8)}
        hw, channels, joint, pooled, tokens = 4, 4, 32, 32, 16
    if chroma:
        kwargs.update(approximator_num_channels=64, approximator_hidden_dim=5120, approximator_layers=5) if real else \
            kwargs.update(approximator_num_channels=8, approximator_hidden_dim=16, approximator_layers=1)
    else:
        kwargs.update(pooled_projection_dim=pooled, guidance_embeds=True)
    model = cls(**kwargs)

    def inputs(g):
        d = {"hidden_states": _randn((1, hw * hw, channels), g), "encoder_hidden_states": _randn((1, tokens, joint), g),
             "img_ids": _ids(hw * hw, hw, hw), "txt_ids": torch.zeros(tokens, 3), "timestep": torch.tensor([0.5])}
        if not chroma:
            d.update(pooled_projections=_randn((1, pooled), g), guidance=torch.tensor([3.5]))
        return d
    return model, inputs


def _flux2(real: bool):
    from diffusers import Flux2Transformer2DModel
    if real:  # Flux.2 klein-like width, 512 px
        model = Flux2Transformer2DModel(patch_size=1, in_channels=128, num_layers=2, num_single_layers=2,
                                        attention_head_dim=128, num_attention_heads=24, joint_attention_dim=7680,
                                        timestep_guidance_channels=256, axes_dims_rope=(32, 32, 32, 32))
        hw, channels, joint, tokens = 32, 128, 7680, 256
    else:
        model = Flux2Transformer2DModel(patch_size=1, in_channels=4, num_layers=2, num_single_layers=2,
                                        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                        timestep_guidance_channels=256, axes_dims_rope=(4, 4, 4, 4))
        hw, channels, joint, tokens = 4, 4, 32, 16

    def inputs(g):
        img = torch.cartesian_prod(torch.arange(1), torch.arange(hw), torch.arange(hw), torch.arange(1)).unsqueeze(0)
        txt = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(tokens)).unsqueeze(0)
        return {"hidden_states": _randn((1, hw * hw, channels), g), "encoder_hidden_states": _randn((1, tokens, joint), g),
                "img_ids": img, "txt_ids": txt, "timestep": torch.tensor([0.5]), "guidance": torch.tensor([1.0])}
    return model, inputs


def _qwen(real: bool):
    from diffusers import QwenImageTransformer2DModel
    if real:  # Qwen-Image width, 512 px, 2 blocks
        model = QwenImageTransformer2DModel(patch_size=2, in_channels=64, out_channels=16, num_layers=2,
                                            attention_head_dim=128, num_attention_heads=24, joint_attention_dim=3584,
                                            guidance_embeds=False, axes_dims_rope=(16, 56, 56))
        hw, channels, joint, tokens = 32, 64, 3584, 256
    else:
        model = QwenImageTransformer2DModel(patch_size=2, in_channels=16, out_channels=4, num_layers=3,
                                            attention_head_dim=16, num_attention_heads=4, joint_attention_dim=16,
                                            guidance_embeds=False, axes_dims_rope=(8, 4, 4))
        hw, channels, joint, tokens = 4, 16, 16, 8
    return model, lambda g: {"hidden_states": _randn((1, hw * hw, channels), g),
                             "encoder_hidden_states": _randn((1, tokens, joint), g),
                             "encoder_hidden_states_mask": torch.ones((1, tokens), dtype=torch.long),
                             "timestep": torch.tensor([0.5]), "img_shapes": [(1, hw, hw)]}


def _anima(real: bool):
    from diffusers import CosmosTransformer3DModel
    common = {"in_channels": 16 if real else 4, "out_channels": 16 if real else 4, "patch_size": (1, 2, 2),
              "concat_padding_mask": True, "extra_pos_embed_type": "learnable" if not real else None}
    if real:  # Anima / Cosmos-Predict2 2B width, 512 px, 2 blocks
        model = CosmosTransformer3DModel(num_attention_heads=16, attention_head_dim=128, num_layers=2, mlp_ratio=4.0,
                                         text_embed_dim=1024, adaln_lora_dim=256, max_size=(128, 240, 240),
                                         rope_scale=(1.0, 4.0, 4.0), **common)
        size, channels, text, tokens = 64, 16, 1024, 256
    else:
        model = CosmosTransformer3DModel(num_attention_heads=2, attention_head_dim=12, num_layers=3, mlp_ratio=2,
                                         text_embed_dim=16, adaln_lora_dim=4, max_size=(4, 32, 32),
                                         rope_scale=(2.0, 1.0, 1.0), **common)
        size, channels, text, tokens = 16, 4, 16, 12
    return model, lambda g: {"hidden_states": _randn((1, channels, 1, size, size), g), "timestep": torch.tensor([0.5]),
                             "encoder_hidden_states": _randn((1, tokens, text), g),
                             "padding_mask": torch.zeros(1, 1, size * 8, size * 8)}


def _z_image(real: bool):
    from diffusers import ZImageTransformer2DModel
    if real:  # Z-Image width, 512 px, 2 layers + 1 refiner each
        model = ZImageTransformer2DModel(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=3840,
                                         n_layers=2, n_refiner_layers=1, n_heads=30, n_kv_heads=30, qk_norm=True,
                                         cap_feat_dim=2560, rope_theta=256.0, t_scale=1000.0, axes_dims=[32, 48, 48],
                                         axes_lens=[1536, 512, 512])
        size, cap, tokens = 64, 2560, 256
    else:
        model = ZImageTransformer2DModel(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=16,
                                         n_layers=1, n_refiner_layers=1, n_heads=1, n_kv_heads=2, qk_norm=True,
                                         cap_feat_dim=16, rope_theta=256.0, t_scale=1000.0, axes_dims=[8, 4, 4],
                                         axes_lens=[256, 32, 32])
        size, cap, tokens = 16, 16, 16
    return model, lambda g: {"x": [_randn((16, 1, size, size), g)], "cap_feats": [_randn((tokens, cap), g)],
                             "t": torch.tensor([0.5])}


def _hidream(real: bool):
    from diffusers import HiDreamImageTransformer2DModel
    if real:  # HiDream-I1 width, 512 px, 1 double + 2 single blocks
        model = HiDreamImageTransformer2DModel(patch_size=2, in_channels=16, out_channels=16, num_layers=1,
                                               num_single_layers=2, attention_head_dim=128, num_attention_heads=20,
                                               caption_channels=[4096, 4096], text_emb_dim=2048, num_routed_experts=4,
                                               num_activated_experts=2, axes_dims_rope=(64, 32, 32),
                                               max_resolution=(128, 128), llama_layers=(0, 1))
        size, t5, llama, pooled, tokens = 64, 4096, 4096, 2048, 128
    else:
        model = HiDreamImageTransformer2DModel(patch_size=2, in_channels=4, out_channels=4, num_layers=1,
                                               num_single_layers=1, attention_head_dim=8, num_attention_heads=4,
                                               caption_channels=[8, 4], text_emb_dim=8, num_routed_experts=2,
                                               num_activated_experts=2, axes_dims_rope=(4, 2, 2),
                                               max_resolution=(32, 32), llama_layers=(0, 1))
        size, t5, llama, pooled, tokens = 32, 8, 4, 8, 8
    return model, lambda g: {"hidden_states": _randn((1, 16 if real else 4, size, size), g),
                             "timesteps": torch.tensor([500]),
                             "encoder_hidden_states_t5": _randn((1, tokens, t5), g),
                             "encoder_hidden_states_llama3": _randn((2, 1, tokens, llama), g),
                             "pooled_embeds": _randn((1, pooled), g)}


def _hunyuan_video(real: bool):
    from diffusers import HunyuanVideoTransformer3DModel
    if real:  # HunyuanVideo width, one 512 px frame, 1 double + 2 single blocks
        model = HunyuanVideoTransformer3DModel(in_channels=16, out_channels=16, num_attention_heads=24,
                                               attention_head_dim=128, num_layers=1, num_single_layers=2,
                                               num_refiner_layers=2, patch_size=2, patch_size_t=1,
                                               guidance_embeds=True, text_embed_dim=4096, pooled_projection_dim=768,
                                               rope_axes_dim=(16, 56, 56))
        size, channels, text, pooled, tokens = 64, 16, 4096, 768, 256
    else:
        model = HunyuanVideoTransformer3DModel(in_channels=4, out_channels=4, num_attention_heads=2,
                                               attention_head_dim=10, num_layers=1, num_single_layers=1,
                                               num_refiner_layers=1, patch_size=1, patch_size_t=1,
                                               guidance_embeds=True, text_embed_dim=16, pooled_projection_dim=8,
                                               rope_axes_dim=(2, 4, 4))
        size, channels, text, pooled, tokens = 16, 4, 16, 8, 12
    return model, lambda g: {"hidden_states": _randn((1, channels, 1, size, size), g),
                             "timestep": torch.tensor([500]), "encoder_hidden_states": _randn((1, tokens, text), g),
                             "pooled_projections": _randn((1, pooled), g),
                             "encoder_attention_mask": torch.ones((1, tokens)), "guidance": torch.tensor([3500])}


def _ernie(real: bool):
    from diffusers import ErnieImageTransformer2DModel
    model = ErnieImageTransformer2DModel(hidden_size=16, num_attention_heads=1, num_layers=3, ffn_hidden_size=16,
                                         in_channels=16, out_channels=16, patch_size=1, text_in_dim=16, rope_theta=256,
                                         rope_axes_dim=(8, 4, 4), eps=1e-6, qk_layernorm=True)
    return model, lambda g: {"hidden_states": _randn((1, 16, 16, 16), g), "timestep": torch.tensor([1.0]),
                             "text_bth": _randn((1, 16, 16), g), "text_lens": torch.tensor([16])}


def _ideogram(real: bool):
    from diffusers import Ideogram4Transformer2DModel
    from diffusers.models.transformers.transformer_ideogram4 import (
        IMAGE_POSITION_OFFSET,
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
    )
    channels, llm, text, grid = 16, 24, 4, 2
    model = Ideogram4Transformer2DModel(in_channels=channels, num_layers=3, attention_head_dim=8, num_attention_heads=4,
                                        intermediate_size=32, adaln_dim=16, llm_features_dim=llm, rope_theta=10_000,
                                        mrope_section=(2, 1, 1), norm_eps=1e-5)

    def inputs(g):
        image = grid * grid
        seq = text + image
        hidden = torch.zeros(1, seq, channels)
        hidden[:, text:] = _randn((1, image, channels), g)
        encoder = torch.zeros(1, seq, llm)
        encoder[:, :text] = _randn((1, text, llm), g)
        pos = torch.zeros(1, seq, 3, dtype=torch.long)
        pos[:, :text] = torch.arange(text).unsqueeze(-1)
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        pos[:, text:, 0] = IMAGE_POSITION_OFFSET
        pos[:, text:, 1] = ys.flatten() + IMAGE_POSITION_OFFSET
        pos[:, text:, 2] = xs.flatten() + IMAGE_POSITION_OFFSET
        indicator = torch.full((1, seq), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
        indicator[:, :text] = LLM_TOKEN_INDICATOR
        return {"hidden_states": hidden, "encoder_hidden_states": encoder, "timestep": torch.tensor([0.5]),
                "position_ids": pos, "segment_ids": torch.ones(1, seq, dtype=torch.long), "indicator": indicator}
    return model, inputs


def _krea2(real: bool):
    from diffusers import Krea2Transformer2DModel
    channels, text_dim, text_layers, grid, text = 16, 16, 3, 2, 4
    model = Krea2Transformer2DModel(in_channels=channels, num_layers=2, attention_head_dim=8, num_attention_heads=4,
                                    num_key_value_heads=2, intermediate_size=32, timestep_embed_dim=8,
                                    text_hidden_dim=text_dim, num_text_layers=text_layers, text_num_attention_heads=2,
                                    text_num_key_value_heads=1, text_intermediate_size=16, num_layerwise_text_blocks=1,
                                    num_refiner_text_blocks=1, axes_dims_rope=(4, 2, 2), rope_theta=1000.0,
                                    norm_eps=1e-5)

    def inputs(g):
        image = grid * grid
        pos = torch.zeros(text + image, 3)
        pos[text:, 1] = torch.arange(grid).repeat_interleave(grid).float()
        pos[text:, 2] = torch.arange(grid).repeat(grid).float()
        mask = torch.ones(1, text, dtype=torch.bool)
        mask[:, -1] = False
        return {"hidden_states": _randn((1, image, channels), g),
                "encoder_hidden_states": _randn((1, text, text_layers, text_dim), g), "timestep": torch.tensor([0.5]),
                "position_ids": pos, "encoder_attention_mask": mask}
    return model, inputs


def _moss(real: bool):
    # OneTrainer's copy of the MOSS-Image transformer (no diffusers class yet). Batch 2 with texts of two lengths: the
    # whole batch goes through each block at once with the padded text masked (MOSSImageTransformer.forward), which
    # is what layer offloading needs
    from modules.model.moss.pipeline_moss_image import MOSSImageTransformer
    if real:  # MOSS-Image-Base's widths, two blocks
        model = MOSSImageTransformer(dim=3072, ffn_dim=8192, num_heads=24, num_layers=2, text_dim=4096)
        grid, text, text_dim = 32, 64, 4096
    else:
        model = MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=4, num_layers=3, text_dim=32, freq_dim=32)
        grid, text, text_dim = 3, 6, 32

    def inputs(g):
        mask = torch.ones(2, text, dtype=torch.bool)
        mask[1, -2:] = False
        return {"hidden_states": _randn((2, 128, grid, grid), g), "timestep": torch.tensor([300.0, 700.0]),
                "encoder_hidden_states": _randn((2, text, text_dim), g), "encoder_attention_mask": mask}
    return model, inputs


@dataclass(frozen=True)
class Family:
    name: str
    build: Callable  # build(real) -> (model, inputs) or None
    enable: str  # the checkpointing_util function training uses for this family
    part: str  # the TrainConfig part holding its settings
    has_real: bool = True  # a config at the family's real width exists (not for the newest families yet)
    refused: str = ""  # why GenericTrainer refuses Compile for this family; its run is expected to fail


FAMILIES = [
    Family("SD 1.5/2.x UNet", _sd15, "enable_checkpointing_for_stable_diffusion_unet", "unet"),
    Family("SDXL UNet", _sdxl, "enable_checkpointing_for_stable_diffusion_unet", "unet"),
    Family("PixArt", _pixart, "enable_checkpointing_for_basic_transformer_blocks", "transformer"),
    Family("Sana", _sana, "enable_checkpointing_for_sana_transformer", "transformer"),
    Family("SD3/3.5", _sd3, "enable_checkpointing_for_stable_diffusion_3_transformer", "transformer"),
    Family("Flux.1", lambda real: _flux_like("FluxTransformer2DModel", real),
           "enable_checkpointing_for_flux_transformer", "transformer"),
    Family("Chroma", lambda real: _flux_like("ChromaTransformer2DModel", real),
           "enable_checkpointing_for_chroma_transformer", "transformer"),
    Family("Flux.2", _flux2, "enable_checkpointing_for_flux2_transformer", "transformer"),
    Family("Qwen-Image", _qwen, "enable_checkpointing_for_qwen_transformer", "transformer"),
    Family("Anima (Cosmos)", _anima, "enable_checkpointing_for_qwen_transformer", "transformer"),
    Family("Z-Image", _z_image, "enable_checkpointing_for_z_image_transformer", "transformer"),
    Family("HiDream", _hidream, "enable_checkpointing_for_hi_dream_transformer", "transformer", has_real=False,
           refused="its mixture-of-experts layers choose experts from the data"),
    Family("Hunyuan Video", _hunyuan_video, "enable_checkpointing_for_hunyuan_video_transformer", "transformer"),
    Family("Ernie Image", _ernie, "enable_checkpointing_for_ernie_transformer", "transformer", has_real=False),
    Family("Ideogram 4", _ideogram, "enable_checkpointing_for_ideogram_transformer", "transformer", has_real=False),
    Family("Krea 2", _krea2, "enable_checkpointing_for_krea2_transformer", "transformer", has_real=False),
    Family("MOSS", _moss, "enable_checkpointing_for_moss_transformer", "transformer"),
]


def _to(value, device: torch.device, dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device, dtype) if value.is_floating_point() else value.to(device)
    if isinstance(value, list):
        return [_to(v, device, dtype) for v in value]
    if isinstance(value, dict):
        return {k: _to(v, device, dtype) for k, v in value.items()}
    return value


def _output(result) -> torch.Tensor:
    while isinstance(result, (tuple, list)) or hasattr(result, "to_tuple"):
        if hasattr(result, "to_tuple"):
            result = result.to_tuple()
        tensors = [r for r in result if isinstance(r, (torch.Tensor, list, tuple))]
        if len(tensors) > 1 and all(isinstance(r, torch.Tensor) for r in tensors):
            return torch.cat([r.flatten() for r in tensors])
        result = tensors[0]
    return result


class _Setup:
    """one family's model, compiled or eager, wrapped the way training wraps it, with a LoRA on every linear layer"""

    def __init__(self, ctx: Ctx, family: Family, real: bool, compile_: bool, dtype: torch.dtype | None = None,
                 offload: bool = False, activations: bool = True, async_transfers: bool = True):
        from modules.module.LoRAModule import LoRAModule
        from modules.util import checkpointing_util
        from modules.util.config.TrainConfig import TrainConfig
        from modules.util.enum.TrainingMethod import TrainingMethod

        self.ctx = ctx
        self.dtype = dtype or (torch.bfloat16 if ctx.is_gpu else torch.float32)
        torch.manual_seed(0)
        built = family.build(real)
        model, self.make_inputs = built
        model = model.to(ctx.device, self.dtype).requires_grad_(False)
        config = TrainConfig.default_values()
        config.training_method = TrainingMethod.LORA
        config.compile = compile_
        config.train_device = str(ctx.device)
        config.temp_device = "cpu" if ctx.is_gpu else "cpu:1"  # cpu:1: another device to the conductor, same memory
        part = getattr(config, family.part)
        part.gradient_checkpointing = True
        config.async_offloading = async_transfers
        part.offload_fraction = 0.5 if offload else 0.0
        part.activation_offloading = offload and activations
        self.conductor = getattr(checkpointing_util, family.enable)(model, config, part)
        if self.conductor is not None:
            self.conductor.materialize()  # as BaseModel.materialize does when a part goes to the train device
        self.model = model
        self.loras = []
        gen = torch.Generator().manual_seed(7)
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                lora = LoRAModule(name, module, 4, 4.0)
                with torch.no_grad():
                    # both from one CPU generator: LoRAModule draws lora_down on its base layer's device, so with
                    # offloading the offloaded layers' LoRAs would start elsewhere (CPU vs GPU generator) and an
                    # offloaded run would be compared from different weights. LoRA up starts at 0: moved, so the
                    # delta and the down gradients are exercised.
                    lora.lora_down.weight.copy_(torch.randn(lora.lora_down.weight.shape, generator=gen)
                                                / lora.lora_down.weight.shape[1] ** 0.5)
                    lora.lora_up.weight.copy_(torch.randn(lora.lora_up.weight.shape, generator=gen) * 0.01)
                lora.to(ctx.device)
                lora.hook_to_module()
                self.loras.append(lora)
        self.inputs = _to(self.make_inputs(torch.Generator().manual_seed(1)), ctx.device, self.dtype)
        self.kwargs = {"return_dict": False} if "return_dict" in inspect.signature(model.forward).parameters else {}

    def offloaded_layers(self) -> str:
        try:  # private, for the report only: layers minus the ones kept loaded
            layers = self.conductor._LayerOffloadConductor__layers
            kept = self.conductor._LayerOffloadConductor__offload_strategy.initial_loaded_layers
            return f"{len(layers) - len(kept)} of {len(layers)} layers offloaded"
        except AttributeError:
            return "offloaded layers unknown"

    def close(self):
        # as training does when a part leaves the train device: frees the caches and the pinned memory, which a
        # dropped conductor kept registered (hipErrorInvalidValue in a later run)
        if self.conductor is not None:
            self.conductor.evict()
            self.ctx.sync()
            self.conductor = None

    def step(self) -> torch.Tensor:
        for lora in self.loras:
            for p in lora.parameters():
                p.grad = None
        with torch.autocast(self.ctx.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = _output(self.model(**self.inputs, **self.kwargs))
        out.float().square().mean().backward()
        return out.detach().float()

    def grads(self) -> torch.Tensor:
        # a LoRA on a layer whose output nothing uses gets no gradient eagerly, a zero one compiled
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten()
                          for lora in self.loras for p in lora.parameters()])

    def sample(self) -> torch.Tensor:
        # sampling runs the compiled blocks too: eval mode, no gradients (a separate compiled variant)
        self.model.eval()
        try:
            with torch.no_grad(), torch.autocast(self.ctx.device.type, dtype=self.dtype,
                                                 enabled=self.dtype != torch.float32):
                return _output(self.model(**self.inputs, **self.kwargs)).float()
        finally:
            self.model.train()


def _compare(ctx: Ctx, rec: Rec, family: Family, real: bool) -> None:
    from modules.util.compile_util import init_compile

    from torch._dynamo.utils import counters

    if real and not family.has_real:
        return
    init_compile()
    torch._dynamo.reset()
    counters.clear()
    label = f"{family.name} ({'real width' if real else 'tiny'})"
    tolerance = (3e-2, 1e-1) if ctx.is_gpu else (1e-4, 1e-3)

    def judge(text: str, outs: list[torch.Tensor], out_err: float, grad_err: float | None, graphs: int,
              breaks: int, recompiles: int):
        if graphs == 0:
            rec.fail(f"{text}: nothing was compiled (the family's blocks weren't wrapped)")
        elif breaks:
            rec.fail(f"{text}: graph breaks")
        elif recompiles:
            rec.fail(f"{text}: the second step compiled again")
        elif not all(finite(o) for o in outs) or out_err > tolerance[0] or (grad_err or 0) > tolerance[1]:
            rec.fail(f"{text}: compiled and eager differ more than rounding")
        else:
            rec.line(f"ok: {text}")

    eager = _Setup(ctx, family, real, compile_=False)
    out_e, grads_e = eager.step(), eager.grads()
    sample_e = eager.sample()
    compiled = _Setup(ctx, family, real, compile_=True)
    t0 = time.perf_counter()
    out_c = compiled.step()
    ctx.sync()
    compile_s = time.perf_counter() - t0
    grads_c = compiled.grads()
    graphs = counters["stats"].get("unique_graphs", 0)
    compiled.step()
    recompiles = counters["stats"].get("unique_graphs", 0) - graphs
    sample_c = compiled.sample()
    graphs_all = counters["stats"].get("unique_graphs", 0)
    breaks = sum(counters["graph_break"].values())
    out_err, grad_err, sample_err = rel_err(out_c, out_e), rel_err(grads_c, grads_e), rel_err(sample_c, sample_e)
    text = (f"{label}: compiled vs eager output {out_err:.1e}, LoRA gradients {grad_err:.1e}, sampling "
            f"{sample_err:.1e}; {graphs_all} graphs, {breaks} graph breaks, first step {compile_s:.0f} s")
    if real and ctx.is_gpu:
        def timed(setup):
            torch.cuda.reset_peak_memory_stats(ctx.device)
            ms = ctx.bench(setup.step, iters=10, warmup=2)
            return ms, torch.cuda.max_memory_allocated(ctx.device) / 2**30
        (ms_e, mem_e), (ms_c, mem_c) = timed(eager), timed(compiled)
        text += (f"; step {ms_c:.1f} vs {ms_e:.1f} ms eager ({ms_e / ms_c:.2f}x), peak allocated {mem_c:.2f} vs "
                 f"{mem_e:.2f} GiB")
        rec.metric(f"{family.name}: eager ms", ms_e)
        rec.metric(f"{family.name}: compiled ms", ms_c)
    judge(text, [out_c, grads_c, sample_c], max(out_err, sample_err), grad_err, graphs_all, breaks, recompiles)
    del compiled

    if not real:
        # layer + activation offloading: the other compiled path (BoundaryOffloadCheckpointLayer), against the
        # same eager reference
        torch._dynamo.reset()
        counters.clear()
        offloaded = _Setup(ctx, family, real, compile_=True, offload=True)
        out_o, grads_o = offloaded.step(), offloaded.grads()
        layers_text = offloaded.offloaded_layers()
        graphs = counters["stats"].get("unique_graphs", 0)
        breaks = sum(counters["graph_break"].values())
        out_err, grad_err = rel_err(out_o, out_e), rel_err(grads_o, grads_e)
        judge(f"{label}, compiled with layer + activation offloading ({layers_text}): output vs eager {out_err:.1e}, "
              f"LoRA gradients {grad_err:.1e}; {graphs} graphs, {breaks} graph breaks", [out_o, grads_o], out_err,
              grad_err, graphs, breaks, 0)
        offloaded.close()
        del offloaded
        if ctx.is_gpu and rec.lines[-1].startswith("FAIL"):
            _offload_breakdown(ctx, rec, family, label, out_e, grads_e)
    del eager
    if ctx.is_gpu:
        torch.cuda.empty_cache()


def _offload_breakdown(ctx: Ctx, rec: Rec, family: Family, label: str, out_e: torch.Tensor, grads_e: torch.Tensor):
    # which part of offloading the failure comes from, against the same eager reference without offloading: on a CPU
    # "cpu:1" is the same memory, so a layer used before its transfer finished still computes right there and only a
    # GPU shows such a problem
    for name, compile_, activations, async_transfers in (
            ("eager, layer + activation offloading", False, True, True),
            ("eager, layer offloading only", False, False, True),
            ("compiled, layer offloading only", True, False, True),
            ("compiled, layer + activation offloading, synchronous transfers", True, True, False),
            ("eager, layer offloading only, synchronous transfers", False, False, False)):
        try:
            torch._dynamo.reset()
            setup = _Setup(ctx, family, False, compile_=compile_, offload=True, activations=activations,
                           async_transfers=async_transfers)
            out, grads = setup.step(), setup.grads()
            second, second_grads = setup.step(), setup.grads()
            rec.line(f"  {label}, {name}: output {rel_err(out, out_e):.1e}, LoRA gradients "
                     f"{rel_err(grads, grads_e):.1e}; second step vs first {rel_err(second, out):.1e}, "
                     f"{rel_err(second_grads, grads):.1e}")
            setup.close()
            del setup
        except Exception as e:
            rec.line(f"  {label}, {name}: {type(e).__name__}: {e}"[:400])
        torch.cuda.empty_cache()


def _expect_refused(ctx: Ctx, rec: Rec, family: Family):
    # GenericTrainer refuses Compile for this family; check that it still fails, so the refusal can go once a new
    # diffusers or torch makes it compile
    torch._dynamo.reset()
    try:
        setup = _Setup(ctx, family, False, compile_=True)
        setup.step()
        setup.sample()
    except Exception as e:
        rec.line(f"ok: {family.name}: Compile fails as expected ({type(e).__name__}), and training refuses it before "
                 f"loading: {family.refused}")
        return
    rec.warn(f"{family.name} compiles now: the refusal in GenericTrainer.start (Compile with {family.name}) can go")


@check(SECTION, "Compile transformer blocks on every model family (random weights, no model files)",
       full_only=True, timeout=7200.0,
       note="builds each family's denoiser with random weights and trains one LoRA step compiled and eager: "
            "~15 min on the CPU, longer on the GPU (real widths)")
def compile_families(ctx: Ctx, rec: Rec):
    # the tiny configs check that compiling works and matches eager on every family; on the GPU the real-width
    # configs (a few blocks at the family's real hidden size, 512 px) give the speed and memory that matter
    for family in FAMILIES:
        for real in ((False, True) if ctx.is_gpu else (False,)):
            if family.refused:
                _expect_refused(ctx, rec, family)
                break
            try:
                _compare(ctx, rec, family, real)
            except Exception as e:  # one family failing must not hide the others
                rec.fail(f"{family.name} ({'real width' if real else 'tiny'}): {type(e).__name__}: {e}"[:600])
            finally:
                if ctx.is_gpu:
                    torch.cuda.empty_cache()
