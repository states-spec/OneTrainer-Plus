"""Backport of transformers 5.10's fix for CVE-2026-9856 (GHSA-xrqw-3rrv-vx5w) to the pinned transformers 5.5.4.

save_pretrained writes a tokenizer's or processor's named chat templates to additional_chat_templates/<name>.jinja,
and before 5.10 the name, which comes from the model repository's tokenizer_config.json, could climb out of that
folder: a full fine-tune saved in diffusers format from a malicious base model wrote .jinja files elsewhere.
OneTrainer+ stays on 5.5.4 because 5.6 restructured CLIPTextModel (it lost its text_model. prefix), which changes the
layer names that LoRA keys, embeddings, checkpointing and offloading use. install() adds 5.10's check: a name whose
file wouldn't land directly in the template folder stops the save before anything is written.
"""
import inspect
import os
from pathlib import Path

import transformers
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import CHAT_TEMPLATE_DIR

from packaging.version import Version

_original_save_chat_templates = PreTrainedTokenizerBase.save_chat_templates
_original_processor_save_pretrained = ProcessorMixin.save_pretrained
_save_chat_templates_signature = inspect.signature(_original_save_chat_templates)


def _check_template_names(chat_template, template_dir: str):
    if not isinstance(chat_template, dict):
        return
    for name in chat_template:
        if name == "default":
            continue
        # as transformers 5.10 checks it
        if Path(os.path.join(template_dir, f"{name}.jinja")).resolve().parent != Path(template_dir).resolve():
            raise ValueError(f"Invalid chat template name: {name!r}")


def _save_chat_templates(self, *args, **kwargs):
    arguments = _save_chat_templates_signature.bind(self, *args, **kwargs).arguments
    if arguments.get("save_jinja_files"):
        prefix = arguments.get("filename_prefix")
        _check_template_names(self.chat_template, os.path.join(
            arguments["save_directory"], (prefix + "-" if prefix else "") + CHAT_TEMPLATE_DIR))
    return _original_save_chat_templates(self, *args, **kwargs)


def _processor_save_pretrained(self, save_directory, *args, **kwargs):
    _check_template_names(getattr(self, "chat_template", None), os.path.join(save_directory, CHAT_TEMPLATE_DIR))
    return _original_processor_save_pretrained(self, save_directory, *args, **kwargs)


def needed() -> bool:
    return Version(transformers.__version__) < Version("5.10")


def install():
    if needed():
        PreTrainedTokenizerBase.save_chat_templates = _save_chat_templates
        ProcessorMixin.save_pretrained = _processor_save_pretrained


def installed() -> bool:
    return PreTrainedTokenizerBase.save_chat_templates is _save_chat_templates
