import ctypes
import hashlib
import os
import platform
import shutil
import urllib.request
import zipfile

RELEASE = f"rel.{os.environ.get('ZLUDA_HASH', '11cc5844514f93161e0e74387f04e2c537705a82')}"
# SHA-256 of each known release's ZLUDA-windows-amd64.zip. Its DLLs are loaded into OneTrainer+'s process, and the
# files of a GitHub release can be replaced after it was published. Another release (ZLUDA_HASH) needs ZLUDA_SHA256.
RELEASE_SHA256 = {
    "rel.11cc5844514f93161e0e74387f04e2c537705a82": "7a1731725f0300b7ecf17b9ac07854c986488428f6c619e0d535e564fc267799",
}
DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
HIP_TARGETS = ['rocblas.dll', 'rocsolver.dll', 'hiprtc0507.dll',]
ZLUDA_TARGETS = ('nvcuda.dll', 'nvml.dll',)


def get_path() -> str:
    return os.path.abspath(os.environ.get('ZLUDA', '.zluda'))


def find_hip_sdk() -> str | None:
    program_files = os.environ.get('ProgramFiles', r'C:\Program Files')  # noqa: SIM112
    hip_path_default = rf'{program_files}\AMD\ROCm\5.7'
    if not os.path.exists(hip_path_default):
        hip_path_default = None
    return os.environ.get('HIP_PATH', hip_path_default)


def install(zluda_path: os.PathLike) -> None:
    if os.path.exists(zluda_path):
        return

    if platform.system() != 'Windows': # TODO
        return

    urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/{RELEASE}/ZLUDA-windows-amd64.zip', '_zluda')
    expected = os.environ.get('ZLUDA_SHA256', RELEASE_SHA256.get(RELEASE, '')).strip().lower()
    sha256 = hashlib.sha256()
    with open('_zluda', 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            sha256.update(chunk)
    if sha256.hexdigest() != expected:
        os.remove('_zluda')
        if not expected:
            raise RuntimeError(f'No SHA-256 is known for ZLUDA {RELEASE}: set ZLUDA_SHA256 to the checksum of its '
                               f'ZLUDA-windows-amd64.zip')
        raise RuntimeError(f'The ZLUDA {RELEASE} download has the SHA-256 {sha256.hexdigest()} instead of {expected}: '
                           f'it was not installed')
    with zipfile.ZipFile('_zluda', 'r') as archive:
        infos = archive.infolist()
        for info in infos:
            if not info.is_dir():
                info.filename = os.path.basename(info.filename)
                archive.extract(info, '.zluda')
    os.remove('_zluda')


def make_copy(zluda_path: os.PathLike) -> None:
    for k, v in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(zluda_path, v)):
            try:
                os.link(os.path.join(zluda_path, k), os.path.join(zluda_path, v))
            except Exception:
                shutil.copyfile(os.path.join(zluda_path, k), os.path.join(zluda_path, v))


def load(zluda_path: os.PathLike) -> None:
    hip_path = find_hip_sdk()
    if hip_path is None:
        raise RuntimeError('Could not find AMD HIP SDK, please install it from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html')
    for v in HIP_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(hip_path, 'bin', v))
    for v in ZLUDA_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
