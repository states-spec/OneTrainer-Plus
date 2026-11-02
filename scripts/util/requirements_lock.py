"""Hash-locked requirements for install.sh and update.sh (lib.include.sh).

    python scripts/util/requirements_lock.py lock [--upgrade] [requirements-rocm.txt ...]
    python scripts/util/requirements_lock.py check requirements-rocm.txt

requirements-lock/<platform file> holds every package that requirements-global.txt and that platform file install,
with their dependencies, each at one version with the SHA-256 of each of its files. The launch scripts install
requirements-lock/build-tools.txt first (pip, setuptools, setuptools-scm, wheel), then the lock file with
`pip install --require-hashes --no-build-isolation`: pip then refuses a file that was replaced on PyPI or the PyTorch
index after the lock was made, and never picks up a newer release. Packages built from source (two in the lock files,
and the git packages) build with those hash-checked build tools: pip's isolated build environments download theirs
without checking any hash, even in --require-hashes mode. requirements-unlocked.txt (the git packages, pinned by
commit, and yt-dlp) comes last, without dependencies; their dependencies are in the lock file.

lock: makes the lock files with uv (https://docs.astral.sh/uv/), for every OS and Python 3.11 to 3.13. It keeps the
versions in the existing lock files where the requirement files still allow them; --upgrade resolves every package
anew. The PyTorch index lists some files without a SHA-256 (every torch 2.13.0+rocm7.2 wheel): uv would download each
of them to hash it, 6 GB per ROCm wheel, unpacked into its cache. So uv reads the extra indexes through a local
stand-in, which hashes those files of the pinned versions itself, streaming each once and keeping only the hash and
the package metadata (cached in ~/.cache/onetrainer-plus/requirements-lock), and leaves out other files without one.

check: run by the launch scripts. Succeeds (exit code 0) if there is a lock file for the platform file, and it and
build-tools.txt were made from the current requirement files. Only the standard library: it runs before the
requirements are installed.
"""

import argparse
import base64
import collections
import concurrent.futures
import hashlib
import html
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLOBAL_FILE = "requirements-global.txt"
UNLOCKED_FILE = "requirements-unlocked.txt"
LOCK_DIR = "requirements-lock"
PLATFORM_FILES = ["requirements-default.txt", "requirements-cuda.txt", "requirements-cuda-legacy.txt",
                  "requirements-rocm.txt"]
BUILD_TOOLS_LOCK = os.path.join(LOCK_DIR, "build-tools.txt")
# what the packages built from source need: setuptools (diffusers, Muon, antlr4-python3-runtime, dadaptation),
# setuptools-scm (mgds), and pip itself. setuptools gets the version requirements-global.txt pins. A package that
# builds with another backend (hatchling, flit-core, ...) needs it added here.
BUILD_TOOLS = ["pip", "setuptools", "setuptools-scm>=8", "wheel"]
INPUTS_PREFIX = "# inputs-sha256: "
PYTHON_MINIMUM = "3.11"
# the CPython versions the launch scripts accept (3.11 to 3.13): their wheels are hashed when the index has no hash
PYTHON_TAGS = {"cp311", "cp312", "cp313"}
CACHE_DIR = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "onetrainer-plus",
                         "requirements-lock")
# a wheel's metadata and its central directory are at its end
TAIL_BYTES = 64 * 2**20

_COMMENT = re.compile(r"(^|\s)#.*$")
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*===?\s*([^\s;,]+)")
_GIT = re.compile(r"^(?:-e\s+|--editable\s+)?(git\+\S+?)#egg=([A-Za-z0-9][A-Za-z0-9._-]*)$")
_INDEX = re.compile(r"^(--index-url|--extra-index-url|-i)(?:\s+|=)(\S+)$")
_ANCHOR = re.compile(r"<a\s[^>]*>.*?</a>", re.IGNORECASE | re.DOTALL)
_HREF = re.compile(r'href\s*=\s*"([^"]*)"', re.IGNORECASE)


def inputs_sha256(*names: str) -> str:
    """The SHA-256 of what a lock file is made from (requirement files and BUILD_TOOLS), so check() notices a change."""
    sha = hashlib.sha256()
    for name in names:
        with open(name, "rb") as f:
            sha.update(name.encode("utf-8") + b"\0" + f.read().replace(b"\r\n", b"\n") + b"\0")
    sha.update("\n".join(BUILD_TOOLS).encode("utf-8"))
    return sha.hexdigest()


def _recorded_inputs_sha256(lock_file: str) -> str | None:
    with open(lock_file, encoding="utf-8") as f:
        return next((line[len(INPUTS_PREFIX):].strip() for line in f if line.startswith(INPUTS_PREFIX)), None)


def check(platform_file: str) -> int:
    platform_file = os.path.normpath(platform_file)
    lock_file = os.path.join(LOCK_DIR, platform_file)
    if (os.path.dirname(platform_file) or not os.path.isfile(platform_file) or not os.path.isfile(lock_file)
            or not os.path.isfile(BUILD_TOOLS_LOCK)):
        print(f"There is no hash-locked version of {platform_file}: installing it without hash checking.",
              file=sys.stderr)
        return 2
    if (_recorded_inputs_sha256(lock_file) != inputs_sha256(GLOBAL_FILE, UNLOCKED_FILE, platform_file)
            or _recorded_inputs_sha256(BUILD_TOOLS_LOCK) != inputs_sha256(GLOBAL_FILE)):
        print(f"{lock_file} or {BUILD_TOOLS_LOCK} was made from other requirement files than the current ones: "
              f"installing without hash checking. To make them again: python scripts/util/requirements_lock.py lock",
              file=sys.stderr)
        return 1
    print(f"Installing the hash-locked requirements: {BUILD_TOOLS_LOCK}, then {lock_file}.")
    return 0


def _lines(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = _COMMENT.sub("", line).strip()
            if line:
                yield line


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _unlocked_entries() -> tuple[list[str], list[str]]:
    """requirements-unlocked.txt as uv reads it (a git package as `name @ git+url`), and its package names."""
    lines, names = [], []
    for line in _lines(UNLOCKED_FILE):
        if git := _GIT.match(line):
            lines.append(f"{git.group(2)} @ {git.group(1)}")
            names.append(git.group(2))
        elif (name := _NAME.match(line)) is not None:
            lines.append(line)
            names.append(name.group(1))
        else:
            raise ValueError(f"{UNLOCKED_FILE}: can't lock around this line: {line}")
    return lines, names


def _pins(*paths: str) -> dict[str, str]:
    pins = {}
    for path in paths:
        for line in _lines(path):
            if pin := _PIN.match(line):
                pins[_normalize(pin.group(1))] = pin.group(2).lower()
    return pins


class _TailFile(io.RawIOBase):
    """The end of a file at its real offsets, for zipfile (zip64 offsets are absolute)."""

    def __init__(self, tail: bytes, size: int):
        super().__init__()
        self.tail, self.size, self.start, self.pos = tail, size, size - len(tail), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if self.pos < self.start and end > self.pos:
            raise OSError(f"read at offset {self.pos}, before the {len(self.tail)} bytes kept from the end")
        data = self.tail[self.pos - self.start:end - self.start]
        self.pos = end
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def _stream_wheel(url: str) -> tuple[str, int, bytes]:
    sha = hashlib.sha256()
    tail, tail_size, size = collections.deque(), 0, 0
    with urllib.request.urlopen(url, timeout=120) as response:
        expected = response.headers.get("Content-Length")
        while chunk := response.read(4 * 2**20):
            sha.update(chunk)
            size += len(chunk)
            tail.append(chunk)
            tail_size += len(chunk)
            while tail_size - len(tail[0]) >= TAIL_BYTES:
                tail_size -= len(tail.popleft())
    # http.client ends a response that stops early like a complete one: a cut-off download would get a wrong hash
    if expected is not None and size != int(expected):
        raise OSError(f"the download stopped after {size} of {expected} bytes")
    with zipfile.ZipFile(_TailFile(b"".join(tail), size)) as archive:
        names = [name for name in archive.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)]
        if len(names) != 1:
            raise RuntimeError(f"{url}: expected one .dist-info/METADATA, found {names}")
        return sha.hexdigest(), size, archive.read(names[0])


_print_lock = threading.Lock()


def _say(message: str):
    """Prints a line from any thread (hashing and the local index run in several) without mixing lines."""
    with _print_lock:
        print(message, file=sys.stderr, flush=True)


def _file_hash_and_metadata(url: str) -> tuple[str, bytes]:
    """The SHA-256 and the .dist-info/METADATA of a wheel, from one streamed download that keeps only its end."""
    cache_file = os.path.join(CACHE_DIR, hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json")
    with suppress(OSError, ValueError, KeyError):
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        if cached["url"] == url:
            return cached["sha256"], base64.b64decode(cached["metadata"])

    name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    _say(f"Hashing {name}: the index lists it without a SHA-256. One download, nothing of it is kept but the hash and "
         f"the package metadata...")
    for attempt in range(1, 4):
        try:
            sha256, size, metadata = _stream_wheel(url)
            break
        except (OSError, http.client.HTTPException, zipfile.BadZipFile) as e:
            if attempt == 3:
                raise
            _say(f"{name}: {e}. Trying again...")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"url": url, "sha256": sha256, "size": size,
                   "metadata": base64.b64encode(metadata).decode("ascii")}, f)
    return sha256, metadata


class _IndexMirror:
    """A local stand-in for package indexes, for uv: each project page as the index serves it, except for the files
    it lists without a SHA-256. Wheels of a pinned version for a supported Python get their hash and metadata
    (PEP 658) from here, so uv never downloads them; the others are left out, so uv can't pick a file nobody hashed."""

    def __init__(self):
        self.indexes: list[str] = []
        self.pins: dict[str, str] = {}
        self.metadata: dict[str, bytes] = {}
        self.hash_lock = threading.Lock()
        self.reported: set[tuple[str, int]] = set()
        mirror = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status, body = mirror.respond(urllib.parse.unquote(urllib.parse.urlsplit(self.path).path))
                self.send_response(status)
                self.send_header("Content-Type", "text/html" if self.path.endswith("/") else "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def url(self, index_url: str) -> str:
        if index_url not in self.indexes:
            self.indexes.append(index_url)
        return f"{self.base_url}/{self.indexes.index(index_url)}/"

    def respond(self, path: str) -> tuple[int, bytes]:
        try:
            if path in self.metadata:
                return 200, self.metadata[path]
            parts = path.strip("/").split("/")
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) < len(self.indexes):
                page = self.page(int(parts[0]), parts[1])
                if page is not None:
                    return 200, page.encode("utf-8")
        except Exception as e:
            # uv only sees a missing page: print the reason
            _say(f"Local index stand-in: {path}: {e}")
            return 500, b""
        return 404, b""

    def prepare(self, index_url: str):
        """Hashes the files of the pinned projects that need it before uv runs, which would time out waiting."""
        for project in sorted(self.pins):
            self.page(self.indexes.index(index_url), project)

    def _wanted(self, project: str, filename: str) -> bool:
        """A wheel of the pinned version of a project for a supported Python (not a free-threaded build)."""
        pinned = self.pins.get(_normalize(project))
        parts = filename[:-len(".whl")].split("-")
        if pinned is None or not filename.endswith(".whl") or len(parts) not in (5, 6) or parts[1].lower() != pinned:
            return False
        python_tags, abi_tags = parts[-3].split("."), parts[-2].split(".")
        if any(abi.endswith("t") for abi in abi_tags):
            return False
        return (any(tag in PYTHON_TAGS or tag == "py3" for tag in python_tags)
                or ("abi3" in abi_tags and any(tag.startswith("cp3") for tag in python_tags)))

    def page(self, index: int, project: str) -> str | None:
        url = urllib.parse.urljoin(self.indexes[index], urllib.parse.quote(project) + "/")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "text/html"}),
                                        timeout=60) as response:
                page_url, text = response.geturl(), response.read().decode("utf-8")
        except urllib.error.HTTPError:
            return None

        pinned = self.pins.get(_normalize(project))
        anchors, unhashed, left_out = [], [], 0
        for anchor in _ANCHOR.findall(text):
            href = _HREF.search(anchor)
            if href is None:
                continue
            file_url = urllib.parse.urljoin(page_url, html.unescape(href.group(1)))
            filename = html.unescape(re.sub(r"<[^>]*>", "", anchor)).strip()
            if "#sha256=" in file_url:
                anchors.append(anchor.replace(href.group(0), f'href="{html.escape(file_url)}"'))
            elif self._wanted(project, filename):
                unhashed.append((filename, urllib.parse.urldefrag(file_url).url))
            elif pinned is not None and [part.lower() for part in filename.split("-")[1:2]] == [pinned]:
                left_out += 1

        if left_out and (project, index) not in self.reported:
            self.reported.add((project, index))
            _say(f"{self.indexes[index]}: left out {left_out} files of {project} {pinned} without a SHA-256 that are "
                 f"not for Python {', '.join(sorted(PYTHON_TAGS))}")
        hashed = []
        if unhashed:
            with self.hash_lock, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                hashed = list(pool.map(_file_hash_and_metadata, [file_url for _, file_url in unhashed]))
        for (filename, _), (sha256, metadata) in zip(unhashed, hashed, strict=True):
            local = f"/files/{index}/{filename}"
            self.metadata[local + ".metadata"] = metadata
            metadata_sha256 = hashlib.sha256(metadata).hexdigest()
            anchors.append(f'<a href="{self.base_url}{urllib.parse.quote(local)}#sha256={sha256}" '
                           f'data-dist-info-metadata="sha256={metadata_sha256}" '
                           f'data-core-metadata="sha256={metadata_sha256}">{html.escape(filename)}</a>')
        return "<!DOCTYPE html>\n<html><body>\n" + "<br/>\n".join(anchors) + "\n</body></html>\n"


def _check_hashed(text: str):
    """Every requirement in uv's output has hashes: pip would refuse the whole file otherwise."""
    entry = None
    for line in [*text.splitlines(), "end"]:
        if line and not line[0].isspace() and not line.startswith(("#", "-")):
            if entry is not None and "--hash=sha256:" not in entry:
                raise RuntimeError(f"no hash for {entry.split()[0]}")
            entry = line
        elif entry is not None:
            entry += line


def _compile(uv: str, tmp: str, inputs: list[str], lock_file: str, upgrade: bool, options: list[str]) -> str:
    """uv pip compile of input files in tmp: every OS, Python 3.11 to 3.13, hashes; keeps the existing versions."""
    output = os.path.join(tmp, "lock.txt")
    if os.path.isfile(lock_file) and not upgrade:
        # uv keeps the versions in an existing output file where the requirements allow them
        shutil.copyfile(lock_file, output)
    command = [uv, "pip", "compile", *inputs, "--output-file", output, "--universal", "--python-version",
               PYTHON_MINIMUM, "--generate-hashes", "--no-header", "--no-config", "--quiet", *options]
    if upgrade:
        command.append("--upgrade")
    no_proxy = ",".join(filter(None, [os.environ.get("NO_PROXY") or os.environ.get("no_proxy"), "127.0.0.1"]))
    subprocess.run(command, cwd=tmp, check=True,
                   env={**os.environ, "NO_PROXY": no_proxy, "no_proxy": no_proxy, "UV_HTTP_TIMEOUT": "600"})
    with open(output, encoding="utf-8") as f:
        body = f.read()
    _check_hashed(body)
    return body


def _write_lock(lock_file: str, header: list[str], body: str):
    os.makedirs(LOCK_DIR, exist_ok=True)
    with open(lock_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n\n" + body)
    _say(f"Wrote {lock_file}")


def lock_build_tools(upgrade: bool, uv: str):
    setuptools = _pins(GLOBAL_FILE).get("setuptools")
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "build-tools.txt"), "w", encoding="utf-8") as f:
            for tool in BUILD_TOOLS:
                f.write(f"setuptools=={setuptools}\n" if tool == "setuptools" and setuptools else f"{tool}\n")
        _say(f"Locking the build tools ({', '.join(BUILD_TOOLS)})...")
        body = _compile(uv, tmp, ["build-tools.txt"], BUILD_TOOLS_LOCK, upgrade, [])
    _write_lock(BUILD_TOOLS_LOCK, [
        "# Hash-locked build tools. install.sh and update.sh install them first (pip install --require-hashes), so that",
        "# every package built from source after them builds with these (--no-build-isolation): pip's isolated build",
        "# environments download their build tools without checking any hash, even in --require-hashes mode.",
        "# Made by `python scripts/util/requirements_lock.py lock` (uv): don't edit it by hand.",
        f"{INPUTS_PREFIX}{inputs_sha256(GLOBAL_FILE)}",
    ], body)


def _versions_only(lock_text: str) -> str:
    """The `name==version ; marker` lines of a lock file, without its hashes and comments (for a constraints file)."""
    return "".join(line.removesuffix("\\").strip() + "\n" for line in lock_text.splitlines()
                   if line and not line[0].isspace() and not line.startswith(("#", "-")))


def lock(platform_file: str, upgrade: bool, uv: str, mirror: _IndexMirror):
    lock_file = os.path.join(LOCK_DIR, platform_file)
    unlocked_lines, unlocked_names = _unlocked_entries()
    with tempfile.TemporaryDirectory() as tmp:
        # uv reads -e git+ only for local folders: requirements-unlocked.txt goes in as `name @ git+url`, the extra
        # indexes as their local stand-ins. The build tools are constraints, so that what the requirements share with
        # them (setuptools, wheel, packaging) gets the same versions.
        shutil.copyfile(GLOBAL_FILE, os.path.join(tmp, GLOBAL_FILE))
        with open(os.path.join(tmp, UNLOCKED_FILE), "w", encoding="utf-8") as f:
            f.write("\n".join(unlocked_lines) + "\n")
        with open(BUILD_TOOLS_LOCK, encoding="utf-8") as f, \
                open(os.path.join(tmp, "build-tools.txt"), "w", encoding="utf-8") as out:
            out.write(_versions_only(f.read()))
        mirror.pins = _pins(GLOBAL_FILE, platform_file)
        index_lines = []
        with open(platform_file, encoding="utf-8") as f, \
                open(os.path.join(tmp, platform_file), "w", encoding="utf-8") as out:
            for line in f:
                if index := _INDEX.match(_COMMENT.sub("", line).strip()):
                    option = "--index-url" if index.group(1) == "-i" else index.group(1)
                    index_lines.append(f"{option} {index.group(2)}")
                    line = f"{option} {mirror.url(index.group(2))}\n"
                    mirror.prepare(index.group(2))
                out.write(line)
        _say(f"Locking {platform_file}...")
        options = ["--constraint", "build-tools.txt", "--index-strategy", "unsafe-best-match"]
        for name in unlocked_names:
            options += ["--no-emit-package", name]
        body = _compile(uv, tmp, [GLOBAL_FILE, platform_file], lock_file, upgrade, options)

    if mirror.base_url in body:
        raise RuntimeError("the address of the local index stand-in ended up in the lock file")
    _write_lock(lock_file, [
        f"# Hash-locked requirements for {platform_file}: every package that it and {GLOBAL_FILE} install, with their",
        "# dependencies, each with the SHA-256 of its files. install.sh and update.sh install build-tools.txt first, then",
        f"# this file with `pip install --require-hashes --no-build-isolation`, then {UNLOCKED_FILE} without dependencies.",
        "# Made by `python scripts/util/requirements_lock.py lock` (uv): don't edit it by hand.",
        f"{INPUTS_PREFIX}{inputs_sha256(GLOBAL_FILE, UNLOCKED_FILE, platform_file)}",
        *index_lines,
    ], body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    lock_parser = commands.add_parser("lock", help="make the lock files with uv")
    lock_parser.add_argument("platform_files", nargs="*", default=PLATFORM_FILES)
    lock_parser.add_argument("--upgrade", action="store_true", help="resolve every package anew")
    lock_parser.add_argument("--uv", default="uv", help="the uv command (default: uv)")
    check_parser = commands.add_parser("check", help="succeed if a platform file has a current lock file")
    check_parser.add_argument("platform_file")
    args = parser.parse_args()

    os.chdir(REPO_DIR)
    if args.command == "check":
        return check(args.platform_file)

    uv = shutil.which(args.uv)
    if uv is None:
        print("lock needs uv: https://docs.astral.sh/uv/getting-started/installation/ (or pip install uv)",
              file=sys.stderr)
        return 1
    if args.upgrade and sorted(args.platform_files) != sorted(PLATFORM_FILES):
        print("--upgrade resolves the build tools anew, which every lock file shares: run it for all of them",
              file=sys.stderr)
        return 1
    mirror = _IndexMirror()
    try:
        lock_build_tools(args.upgrade, uv)
        for platform_file in args.platform_files:
            lock(platform_file, args.upgrade, uv, mirror)
    finally:
        mirror.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
