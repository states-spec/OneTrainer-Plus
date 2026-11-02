import os
import re
import subprocess

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_git_commit() -> str | None:
    """The full hash of the commit this OneTrainer+ checkout is at, or None (not a git checkout, or no git)."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_REPO_ROOT,
                                         stderr=subprocess.DEVNULL).decode('ascii').strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) else None


def has_uncommitted_changes() -> bool:
    try:
        return bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=_REPO_ROOT,
                                            stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return False


def get_git_branch() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "git not installed"


def get_git_revision() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "git not installed"
