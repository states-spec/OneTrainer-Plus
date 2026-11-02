"""The messages between OneTrainer+ and its trainer on a cloud machine, one JSON object per line.

The trainer on the cloud machine reports progress, status and samples to this PC (callbacks), and receives the UI's
commands (stop, sample, backup, save). Both directions used to be pickled, and unpickling runs whatever the sender packs
into the data: the cloud machine, anyone between it and this PC, or a config's install command could run code here.
JSON carries only data, and only the callbacks listed in _CALLBACKS can be called.
"""
import base64
import io
import json

from modules.modelSampler.BaseModelSampler import ModelSamplerOutput
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from PIL import Image

# a callback file larger than this is dropped (a sample image is ~100 KiB as base64 JPEG)
MAX_CALLBACK_BYTES = 64 * 2**20

_PROGRESS_FIELDS = ("epoch", "epoch_step", "epoch_sample", "global_step")


def _number(value) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    return value


def _text(value) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected text, got {type(value).__name__}")
    return value


def _encode_progress(train_progress: TrainProgress, max_step, max_epoch) -> list:
    return [{field: getattr(train_progress, field) for field in _PROGRESS_FIELDS}, max_step, max_epoch]


def _decode_progress(args: list) -> list:
    progress, max_step, max_epoch = args
    return [TrainProgress(**{field: _number(progress[field]) for field in _PROGRESS_FIELDS}),
            _number(max_step), _number(max_epoch)]


def _encode_sampler_output(sampler_output: ModelSamplerOutput) -> list:
    data = None
    if sampler_output.file_type == FileType.IMAGE and sampler_output.data is not None:
        # a JPEG, as the pickled form had; videos and audio are not sent (the workspace sync brings their files)
        buffer = io.BytesIO()
        sampler_output.data.save(buffer, format="JPEG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return [{"file_type": str(sampler_output.file_type), "data": data}]


def _decode_sampler_output(args: list) -> list:
    value = args[0]
    file_type = FileType(_text(value["file_type"]))
    data = value["data"]
    if file_type == FileType.IMAGE and data is not None:
        image = Image.open(io.BytesIO(base64.b64decode(_text(data), validate=True)), formats=["JPEG"])
        image.load()
        return [ModelSamplerOutput(file_type, image)]
    return [ModelSamplerOutput(file_type, None)]


def _encode_sample_progress(step, max_step) -> list:
    return [step, max_step]


def _decode_sample_progress(args: list) -> list:
    step, max_step = args
    return [_number(step), _number(max_step)]


# callback name -> (encode its arguments into JSON values, decode and check them)
_CALLBACKS = {
    "on_update_train_progress": (_encode_progress, _decode_progress),
    "on_update_status": (lambda status: [status], lambda args: [_text(args[0])]),
    "on_sample_default": (_encode_sampler_output, _decode_sampler_output),
    "on_update_sample_default_progress": (_encode_sample_progress, _decode_sample_progress),
    "on_sample_custom": (_encode_sampler_output, _decode_sampler_output),
    "on_update_sample_custom_progress": (_encode_sample_progress, _decode_sample_progress),
}


def encode_callback(name: str, *args) -> bytes:
    encode, _ = _CALLBACKS[name]
    return (json.dumps({"name": name, "args": encode(*args)}) + "\n").encode("utf-8")


def decode_callbacks(data: bytes) -> list[tuple[str, list]]:
    """The callbacks in a callback file the cloud machine sent; anything else is skipped with a message."""
    calls = []
    skipped = 0
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            name = message["name"]
            _, decode = _CALLBACKS[name]
            calls.append((name, decode(message["args"])))
        except Exception:
            skipped += 1
    if skipped:
        print(f"Ignored {skipped} message(s) from the cloud machine that aren't known callbacks (a different "
              f"OneTrainer+ version there sends them in another format)")
    return calls


def encode_commands(commands: TrainCommands) -> bytes:
    return (json.dumps(commands.to_dict()) + "\n").encode("utf-8")


def decode_commands(data: bytes, model_type: ModelType) -> TrainCommands:
    """The commands in what the command pipe received (one JSON object per write)."""
    commands = TrainCommands()
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        message = json.loads(line)
        if message.get("stop") is True:
            commands.stop()
        for sample in message.get("sample_custom", []):
            if not isinstance(sample, dict):
                raise ValueError(f"expected a sample definition, got {type(sample).__name__}")
            commands.sample_custom(SampleConfig.default_values(model_type).from_dict(sample))
        if message.get("sample_default") is True:
            commands.sample_default()
        if message.get("backup") is True:
            commands.backup()
        if message.get("save") is True:
            commands.save()
    return commands
