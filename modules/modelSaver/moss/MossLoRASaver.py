from modules.model.MossModel import MossModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin

from torch import Tensor


class MossLoRASaver(
    LoRASaverMixin
):
    def __init__(self):
        super().__init__()

    def _get_state_dict(
            self,
            model: MossModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        # upstream's MOSS port trained additional embeddings next to the LoRA but saved them nowhere
        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                if embedding.append_token is not None:
                    state_dict[f"bundle_emb.{embedding.placeholder}.moss"] = embedding.append_token

        return state_dict
