from pathlib import Path
from typing import Any

from modules.util.app_info import APP_NAME, REPOSITORY_URL
from modules.util.config.BaseConfig import BaseConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType

# A cloud machine installs OneTrainer+ from its own repository. The install command runs in the folder above
# onetrainer_dir and has to create it: git clone names the folder after the repository.
DEFAULT_ONETRAINER_DIR = "/workspace/" + REPOSITORY_URL.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
DEFAULT_INSTALL_CMD = f"git clone {REPOSITORY_URL}"
# OneTrainer's defaults, which installed upstream OneTrainer; configs saved before this fork changed them hold these
_UPSTREAM_ONETRAINER_DIR = "/workspace/OneTrainer"
_UPSTREAM_INSTALL_CMD = "git clone https://github.com/Nerogar/OneTrainer"
# notes already printed: the UI and scripts/train.py both read a config twice
_printed_notes: set[str] = set()


class CloudSecretsConfig(BaseConfig):
    api_key: str
    host: str
    port: int
    user: str
    id: str
    key_file: str
    password: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        data.append(("api_key", "", str, False))
        data.append(("id", "", str, False))
        data.append(("host", "", str, False))
        data.append(("port", 0, str, False))
        data.append(("user", "root", str, False))
        data.append(("key_file", "", str, False)) # whilst not a secret, makes more semantic sense here
        data.append(("password", "", str, False))
        return CloudSecretsConfig(data)

    def expanded_key_file(self) -> str:
        key_file = getattr(self, "key_file", "").strip()
        if key_file == "":
            return ""
        return str(Path(key_file).expanduser())

    def connect_kwargs(self) -> dict:
        kwargs = {}
        key_file = self.expanded_key_file()
        if key_file:
            kwargs["key_filename"] = key_file
        password = getattr(self, "password", "").strip()
        if password:
            kwargs["password"] = password
        # Never query the ssh-agent: a broken agent can make paramiko's Agent() raise and abort the connect.
        kwargs["allow_agent"] = False
        return kwargs


class CloudConfig(BaseConfig):
    enabled: bool
    type: CloudType
    file_sync : CloudFileSync
    create : bool
    name: str
    tensorboard_tunnel: bool
    sub_type: str
    gpu_type: str
    volume_size: int
    min_download: int
    remote_dir: str
    huggingface_cache_dir: str
    onetrainer_dir: str
    install_onetrainer: bool
    update_onetrainer: bool
    install_cmd: str
    detach_trainer: bool
    run_id: str
    download_sampels : bool
    download_output_model : bool
    download_saves : bool
    download_backups : bool
    download_tensorboard : bool
    on_finish: CloudAction
    on_error: CloudAction
    on_detached_finish: CloudAction
    on_detached_error: CloudAction


    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    def from_dict(self, data: dict, migrate: bool = True) -> 'CloudConfig':
        super().from_dict(data, migrate)
        # upstream's default pair (still in configs saved before) installs upstream OneTrainer: replace it with
        # OneTrainer+'s own. Anything else was set by hand and stays.
        if self.install_cmd == _UPSTREAM_INSTALL_CMD and self.onetrainer_dir == _UPSTREAM_ONETRAINER_DIR:
            self.install_cmd = DEFAULT_INSTALL_CMD
            self.onetrainer_dir = DEFAULT_ONETRAINER_DIR
            note = (f"Cloud: the config names upstream OneTrainer's default repository and folder; installing "
                    f"{APP_NAME} from {REPOSITORY_URL} into {DEFAULT_ONETRAINER_DIR} instead")
            if self.enabled and note not in _printed_notes:
                _printed_notes.add(note)
                print(note)
        return self

    def describe_run(self, secrets: CloudSecretsConfig) -> str:
        """What a cloud run with these settings does, shown before it starts: a config file brings its cloud
        settings along (install command, GPU, end actions), and the saved cloud login from secrets.json is added."""
        actions = {CloudAction.NONE: "keep it running", CloudAction.STOP: "stop it", CloudAction.DELETE: "delete it"}
        lines = []
        if self.type == CloudType.RUNPOD:
            if secrets.id:
                lines.append(f"Machine: the RunPod pod {secrets.id}")
            elif self.create:
                lines.append(f"Machine: a new RunPod pod ({self.gpu_type or 'no GPU type set'}, "
                             f"{(self.sub_type or 'any').lower()} cloud, {self.volume_size} GB volume), billed to "
                             f"your RunPod account")
            else:
                lines.append("Machine: RunPod, but no pod id is set and creating a pod is off")
        else:
            lines.append(f"Machine: the Linux server {secrets.user}@{secrets.host}, port {secrets.port}")
        if self.install_onetrainer or self.update_onetrainer:
            note = "" if self.install_cmd == DEFAULT_INSTALL_CMD else "  <- not the default command"
            lines.append(f"Installs and runs {APP_NAME} from {self.onetrainer_dir} there; if that folder doesn't "
                         f"exist yet, it runs:\n    {self.install_cmd}{note}")
        else:
            lines.append(f"Runs the {APP_NAME} already in {self.onetrainer_dir} there")
        lines.append(f"When training finishes: {actions[self.on_finish]}; after an error: {actions[self.on_error]}")
        if self.detach_trainer:
            lines.append(f"If this PC is disconnected (detached run '{self.run_id}'): "
                         f"{actions[self.on_detached_finish]} when it finishes, "
                         f"{actions[self.on_detached_error]} after an error")
        if self.delete_workspace:
            lines.append("The workspace on the cloud machine is deleted afterwards")
        return "\n".join(lines)

    @staticmethod
    def default_values():
        data = []

        data.append(("enabled", False, bool, False))
        data.append(("type", CloudType.RUNPOD, CloudType, False))
        data.append(("file_sync", CloudFileSync.NATIVE_SCP, CloudFileSync, False))
        data.append(("create", False, bool, False))
        data.append(("name", "OneTrainer", str, False))
        data.append(("tensorboard_tunnel", True, bool, False))
        data.append(("sub_type", "", str, False))
        data.append(("gpu_type", "", str, False))
        data.append(("volume_size", 100, int, False))
        data.append(("min_download", 0, int, False))
        data.append(("remote_dir", "/workspace", str, False))
        data.append(("huggingface_cache_dir", "/workspace/huggingface_cache", str, False))
        data.append(("onetrainer_dir", DEFAULT_ONETRAINER_DIR, str, False))
        data.append(("install_cmd", DEFAULT_INSTALL_CMD, str, False))
        data.append(("install_onetrainer", True, bool, False))
        data.append(("update_onetrainer", True, bool, False))
        data.append(("detach_trainer", False, bool, False))
        data.append(("run_id", "job1", str, False))
        data.append(("download_samples", True, bool, False))
        data.append(("download_output_model", True, bool, False))
        data.append(("download_saves", True, bool, False))
        data.append(("download_backups", False, bool, False))
        data.append(("download_tensorboard", False, bool, False))
        data.append(("delete_workspace", False, bool, False))
        data.append(("on_finish", CloudAction.NONE, CloudAction, False))
        data.append(("on_error", CloudAction.NONE, CloudAction, False))
        data.append(("on_detached_finish", CloudAction.NONE, CloudAction, False))
        data.append(("on_detached_error", CloudAction.NONE, CloudAction, False))
        return CloudConfig(data)
