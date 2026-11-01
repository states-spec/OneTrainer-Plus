
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudType import CloudType
from modules.util.ui.browser_util import open_url


class CloudTabController:
    def __init__(self, config: TrainConfig, parent):
        self.config = config
        self.parent = parent
        self.reattach = False

    def do_reattach(self):
        self.reattach = True
        try:
            self.parent.start_training()
        finally:
            self.reattach = False

    def get_gpu_types(self) -> list[str]:
        if self.config.cloud.type == CloudType.RUNPOD:
            import runpod
            runpod.api_key = self.config.secrets.cloud.api_key
            try:
                gpus = runpod.get_gpus()
            except Exception as e:  # no or a wrong API key, or no connection
                print(f"Could not get the GPU types from RunPod (is the API key set?): {e}")
                return []
            return [gpu['id'] for gpu in gpus]
        return []

    def open_create_cloud_url(self):
        if self.config.cloud.type == CloudType.RUNPOD:
            open_url("https://www.runpod.io/console/deploy?template=1a33vbssq9&type=gpu")
