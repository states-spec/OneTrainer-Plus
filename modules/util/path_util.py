import json
import os.path
from contextlib import suppress
from pathlib import Path
from typing import Any


def safe_filename(
        text: str,
        allow_spaces: bool = True,
        max_length: int | None = 32,
):
    legal_chars = [' ', '.', '_', '-', '#']
    if not allow_spaces:
        text = text.replace(' ', '_')

    text = ''.join(filter(lambda x: str.isalnum(x) or x in legal_chars, text)).strip()

    if max_length is not None:
        text = text[0: max_length]

    return text.strip()


def canonical_join(base_path: str, *paths: str):
    # Creates a canonical path name that can be used for comparisons.
    # Also, Windows does understand / instead of \, so these paths can be used as usual.

    joined = os.path.join(base_path, *paths)
    return joined.replace('\\', '/')


# Everything the diagnostic tools write goes into one folder in the OneTrainer folder: rocm_check reports, diag_ab runs,
# OT_DIAG_LOG=true logs, TunableOp results, the debug report and the profiling window's stack dumps.
DIAGNOSTICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "diagnostics")


def diagnostics_path(*parts: str) -> str:
    return os.path.join(DIAGNOSTICS_DIR, *parts)


def write_json_atomic(path: str, obj: Any, private: bool = False):
    if private:
        # readable only by its owner from the moment it exists (secrets.json). os.open's mode applies only to a new
        # file, so a temp file left by an earlier failed write goes first.
        with suppress(FileNotFoundError):
            os.remove(path + ".write")
        with os.fdopen(os.open(path + ".write", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w",
                       encoding="utf-8") as f:
            json.dump(obj, f, indent=4)
    else:
        with open(path + ".write", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)
    os.replace(path + ".write", path)


def restrict_to_owner(path: str):
    """Takes group and other access off an existing file: secrets.json used to be created readable by everyone."""
    if os.name != "posix":
        return
    with suppress(OSError):
        mode = os.stat(path).st_mode & 0o777
        if mode & 0o077:
            os.chmod(path, mode & 0o700)


SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.avif'}
SUPPORTED_VIDEO_EXTENSIONS = {'.webm', '.mkv', '.flv', '.avi', '.mov', '.wmv', '.mp4', '.mpeg', '.m4v'}
SUPPORTED_CAPTION_EXTENSIONS = {'.txt'}


def supported_image_extensions() -> set[str]:
    return SUPPORTED_IMAGE_EXTENSIONS


def is_supported_image_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_IMAGE_EXTENSIONS


def supported_video_extensions() -> set[str]:
    return SUPPORTED_VIDEO_EXTENSIONS


def is_supported_video_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_VIDEO_EXTENSIONS


def supported_caption_extensions() -> set[str]:
    return SUPPORTED_CAPTION_EXTENSIONS


def json_path_modifier(x: str | Path) -> Path:
    x = Path(x).absolute()
    return x.parent if x.suffix == ".json" else x
