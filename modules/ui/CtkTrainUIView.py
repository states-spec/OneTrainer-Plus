import ctypes
import platform
import tkinter as tk
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from tkinter import filedialog, messagebox

from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.BaseTrainUIView import BaseTrainUIView
from modules.ui.CloudTabController import CloudTabController
from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.CtkAdditionalEmbeddingsTabView import CtkAdditionalEmbeddingsTabView
from modules.ui.CtkCaptionUIView import CtkCaptionUIView
from modules.ui.CtkCloudTabView import CtkCloudTabView
from modules.ui.CtkConceptTabView import CtkConceptTabView
from modules.ui.CtkConvertModelUIView import CtkConvertModelUIView
from modules.ui.CtkLoraTabView import CtkLoraTabView
from modules.ui.CtkModelTabView import CtkModelTabView
from modules.ui.CtkProfilingWindowView import CtkProfilingWindowView
from modules.ui.CtkSampleWindowView import CtkSampleWindowView
from modules.ui.CtkSamplingTabView import CtkSamplingTabView
from modules.ui.CtkSettingsTabView import CtkSettingsTabView, apply_ui_scale, apply_ui_theme
from modules.ui.CtkTopBarView import CtkTopBarView
from modules.ui.CtkTrainingTabView import CtkTrainingTabView
from modules.ui.CtkVideoToolUIView import CtkVideoToolUIView
from modules.ui.LoraTabController import LoraTabController
from modules.ui.ModelTabController import ModelTabController
from modules.ui.ProfilingWindowController import ProfilingWindowController
from modules.ui.SamplingTabController import SamplingTabController
from modules.ui.SettingsTabController import SettingsTabController
from modules.ui.TopBarController import TopBarController
from modules.ui.TrainingTabController import TrainingTabController
from modules.ui.TrainUIController import TrainUIController
from modules.util.app_info import APP_NAME
from modules.util.config.TrainConfig import TrainConfig
from modules.util.config.UISettingsConfig import UISettingsConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk

# chunk for forcing Windows to ignore DPI scaling when moving between monitors
# fixes the long standing transparency bug https://github.com/Nerogar/OneTrainer/issues/90
if platform.system() == "Windows":
    with suppress(Exception):
        # https://learn.microsoft.com/en-us/windows/win32/hidpi/setting-the-default-dpi-awareness-for-a-process#setting-default-awareness-programmatically
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE


class CtkTrainUIView(BaseTrainUIView, ctk.CTk):
    set_step_progress: Callable[[int, int], None]
    set_epoch_progress: Callable[[int, int], None]

    status_label: ctk.CTkLabel | None
    training_button: ctk.CTkButton | None

    _TRAIN_BUTTON_STYLES = {
        "idle": {
            "text": "Start Training",
            "state": "normal",
            "fg_color": "#198754",
            "hover_color": "#146c43",
            "text_color": "white",
            "text_color_disabled": "white",
        },
        "running": {
            "text": "Stop Training",
            "state": "normal",
            "fg_color": "#dc3545",
            "hover_color": "#bb2d3b",
            "text_color": "white",
        },
        "stopping": {
            "text": "Stopping...",
            "state": "disabled",
            "fg_color": "#dc3545",
            "hover_color": "#dc3545",
            "text_color": "white",
            "text_color_disabled": "white",
        },
    }

    def __init__(self):
        ctk.CTk.__init__(self)

        train_config = TrainConfig.default_values()
        ui_state = CtkUIState(self, train_config)
        controller = TrainUIController(train_config)

        BaseTrainUIView.__init__(self, ctk_components, controller, ui_state)
        self.controller.view = self

        self.title(APP_NAME)
        self.geometry("1100x740")

        self.after(100, lambda: self._set_icon())

        # UI preferences (theme, scale, window size), stored apart from the training config
        self.ui_settings = UISettingsConfig.load()
        self.ui_settings_state = CtkUIState(self, self.ui_settings)
        apply_ui_theme(self.ui_settings)
        apply_ui_scale(self.ui_settings)
        ctk.set_default_color_theme("blue")
        if self.ui_settings.remember_window_size and self.ui_settings.window_width > 0 and self.ui_settings.window_height > 0:
            self.geometry(f"{self.ui_settings.window_width}x{self.ui_settings.window_height}")
        if self.ui_settings.start_maximized:
            self.after(0, self.__maximize)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.status_label = None
        self.eta_label = None
        self.training_button = None
        self.export_button = None
        self.tabview = None

        self.model_tab = None
        self.training_tab = None
        self.lora_tab = None
        self.cloud_tab = None
        self.additional_embeddings_tab = None
        self.settings_tab = None

        self.top_bar_component = self.top_bar(self)
        self.content_frame(self)
        self.bottom_bar(self)

        self.controller._check_start_always_on_tensorboard()

        self.workspace_dir_trace_id = self.ui_state.add_var_trace("workspace_dir", self.controller._on_workspace_dir_change_trace)

        # Persistent profiling window.
        self._profiling_controller = ProfilingWindowController()
        self.profiling_window = self._profiling_controller.create_window(self, CtkProfilingWindowView)

        self.protocol("WM_DELETE_WINDOW", self.__close)

    def __close(self):
        self.top_bar_component.save_default()
        self.__save_window_size()
        self.controller._stop_always_on_tensorboard()
        if hasattr(self, 'workspace_dir_trace_id'):
            self.ui_state.remove_var_trace("workspace_dir", self.workspace_dir_trace_id)
        self.quit()

    def __maximize(self):
        try:
            self.state("zoomed")  # Windows, macOS
        except tk.TclError:
            with suppress(tk.TclError):
                self.attributes("-zoomed", True)  # X11

    def __is_maximized(self) -> bool:
        with suppress(tk.TclError):
            if self.state() == "zoomed" or self.attributes("-zoomed"):
                return True
        return False

    def __save_window_size(self):
        if not self.ui_settings.remember_window_size or self.__is_maximized():
            return
        with suppress(ValueError):
            # geometry() returns "WxH+X+Y" without CTk's window scaling
            width, height = (int(v) for v in self.geometry().split("+")[0].split("x"))
            self.ui_settings.window_width = width
            self.ui_settings.window_height = height
            self.ui_settings.save()

    # --- BaseTrainUIView abstract method implementations ---

    def on_update_status(self, status: str):
        self.status_label.configure(text=status)

    def on_training_started(self):
        self._set_training_button_style("running")

    def on_training_stopped(self, error_caught: bool):
        self.eta_label.configure(text="")
        self._set_training_button_style("idle")

    def on_training_stopping(self):
        self._set_training_button_style("stopping")

    def on_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None):
        self.set_step_progress(epoch_step, max_step)
        self.set_epoch_progress(epoch, max_epoch)
        if eta_str is not None:
            self.eta_label.configure(text=f"ETA: {eta_str}")
        else:
            self.eta_label.configure(text="")

    def schedule_on_main_thread(self, fn: Callable):
        self.after(0, fn)

    def get_cloud_reattach(self) -> bool:
        return self.cloud_tab.reattach

    def save_default(self):
        self.top_bar_component.save_default()
        self.concepts_tab.save_current_config()
        self.sampling_tab.save_current_config()
        if self.additional_embeddings_tab:
            self.additional_embeddings_tab.save_current_config()

    def show_validation_errors(self, errors: list[str]):
        bullet_list = "\n".join(f"• {e}" for e in errors)
        messagebox.showerror(
            "Cannot Start Training",
            f"Please fix the following errors before training:\n\n{bullet_list}",
        )

    def confirm(self, title: str, message: str) -> bool:
        return messagebox.askokcancel(title, message)

    def open_dataset_tool(self):
        self.wait_window(self.controller.open_dataset_tool(self, CtkCaptionUIView))

    def open_video_tool(self):
        self.wait_window(self.controller.open_video_tool(self, CtkVideoToolUIView))

    def open_convert_model_tool(self):
        self.wait_window(self.controller.open_convert_model_tool(self, CtkConvertModelUIView))

    def open_sampling_tool(self):
        self.controller.open_sampling_tool(self, CtkSampleWindowView)

    def open_manual_sample_window(self):
        self.controller.open_manual_sample_window(self, CtkSampleWindowView)

    def wait_window(self, window):
        ctk.CTk.wait_window(self, window)

    def show_window(self, window):
        window.focus_set()

    def connect_window_closed(self, window, callback):
        # <Destroy> fires for the toplevel and every descendant widget; only react to the window itself
        window.bind("<Destroy>", lambda e: callback() if e.widget is window else None)

    # --- CTK layout and frame builders ---

    def _set_icon(self):
        """Set the window icon safely after window is ready"""
        set_window_icon(self)

    def top_bar(self, master):
        return CtkTopBarView(
            master,
            TopBarController(self.controller.train_config),
            self.ui_state,
            self.change_model_type,
            self.change_training_method,
            self.load_preset,
        )

    def bottom_bar(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=2, column=0, sticky="nsew")

        # status + ETA container
        status_frame = ctk.CTkFrame(frame, corner_radius=0, fg_color="transparent")
        status_frame.grid(row=0, column=1, sticky="w")
        status_frame.grid_rowconfigure(0, weight=0)
        status_frame.grid_rowconfigure(1, weight=0)
        status_frame.grid_columnconfigure(0, weight=1)

        # padding
        frame.grid_columnconfigure(2, weight=1)

        self.build_bottom_bar_content(frame, status_frame, self.controller, self.ui_state)
        self._set_training_button_style("idle")  # centralized styling

        return frame

    def content_frame(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(frame)
        self.tabview.grid(row=0, column=0, sticky="nsew")

        self.general_tab = self.create_general_tab(self.tabview.add("general"))
        self.model_tab = self.create_model_tab(self.tabview.add("model"))
        self.data_tab = self.create_data_tab(self.tabview.add("data"))
        self.concepts_tab = self.create_concepts_tab(self.tabview.add("concepts"))
        self.training_tab = self.create_training_tab(self.tabview.add("training"))
        self.sampling_tab = self.create_sampling_tab(self.tabview.add("sampling"))
        self.backup_tab = self.create_backup_tab(self.tabview.add("backup"))
        self.tools_tab = self.create_tools_tab(self.tabview.add("tools"))
        self.additional_embeddings_tab = self.create_additional_embeddings_tab(self.tabview.add("additional embeddings"))
        self.cloud_tab = self.create_cloud_tab(self.tabview.add("cloud"))
        # always shown, so the LoRA settings stay reachable while another training method is selected
        self.lora_tab = CtkLoraTabView(self.tabview.add("LoRA"), LoraTabController(self.controller.train_config), self.ui_state)
        self.settings_tab = CtkSettingsTabView(
            self.tabview.add("settings"),
            SettingsTabController(self.ui_settings, apply_ui_scale, needs_restart=("theme",), supports_font_size=False),
            self.ui_settings_state,
        )

        self.change_training_method(self.controller.train_config.training_method)
        self._update_additional_embeddings_tab(self.controller.train_config.model_type)

        return frame

    def create_general_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)
        self.build_general_tab_content(frame, self.controller, self.ui_state)
        frame.pack(fill="both", expand=1)
        return frame

    def create_model_tab(self, master):
        return CtkModelTabView(master, ModelTabController(self.controller.train_config), self.ui_state)

    def create_data_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)
        self.build_data_tab_content(frame, self.controller, self.ui_state)
        frame.pack(fill="both", expand=1)
        return frame

    def create_concepts_tab(self, master):
        return CtkConceptTabView(master, ConceptTabController(self.controller.train_config), self.ui_state)

    def create_training_tab(self, master) -> CtkTrainingTabView:
        return CtkTrainingTabView(master, TrainingTabController(self.controller.train_config), self.ui_state)

    def create_cloud_tab(self, master) -> CtkCloudTabView:
        return CtkCloudTabView(master, CloudTabController(self.controller.train_config, parent=self), self.ui_state)

    def create_sampling_tab(self, master):
        master.grid_rowconfigure(0, weight=0)
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(master=master, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="nsew")
        sub_frame = ctk.CTkFrame(master=top_frame, corner_radius=0, fg_color="transparent")
        sub_frame.grid(row=1, column=0, sticky="nsew", columnspan=6)

        self.build_sampling_tab_header(top_frame, sub_frame, self.controller, self.ui_state)

        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        return CtkSamplingTabView(frame, SamplingTabController(self.controller.train_config), self.ui_state)

    def create_backup_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)
        self.build_backup_tab_content(frame, self.controller, self.ui_state)
        frame.pack(fill="both", expand=1)
        return frame

    def embedding_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)
        self.build_embedding_tab_content(frame, self.controller, self.ui_state)
        frame.pack(fill="both", expand=1)
        return frame

    def create_additional_embeddings_tab(self, master):
        return CtkAdditionalEmbeddingsTabView(master, AdditionalEmbeddingsTabController(self.controller.train_config), self.ui_state)

    def create_tools_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)
        self.build_tools_tab_content(frame, self.controller, self.ui_state)
        frame.pack(fill="both", expand=1)
        return frame

    def open_profiling_tool(self):
        self.profiling_window.deiconify()

    def change_model_type(self, model_type: ModelType):
        if self.model_tab:
            self.model_tab.refresh_ui()

        if self.training_tab:
            self.training_tab.refresh_ui()

        if self.lora_tab:
            self.lora_tab.refresh_ui()

        if self.tabview and "embedding" in self.tabview._tab_dict:
            self._refresh_embedding_tab()

        self._update_additional_embeddings_tab(model_type)

    def _update_additional_embeddings_tab(self, model_type: ModelType):
        # always shown; it says so when the model type doesn't support embedding training
        if self.additional_embeddings_tab:
            self.additional_embeddings_tab.update_supported()

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return

        if self.model_tab:
            self.model_tab.refresh_ui()

        if self.training_tab:
            # its masking settings depend on the training method
            self.training_tab.refresh_ui()

        if self.lora_tab:
            self.lora_tab.refresh_ui()

        self._refresh_embedding_tab()

    def _refresh_embedding_tab(self):
        # always shown, like the LoRA tab. Rebuilt on changes, because its notice depends on the training method and
        # the model type.
        exists = "embedding" in self.tabview._tab_dict
        was_current = exists and self.tabview.get() == "embedding"
        index = self.tabview.index("embedding" if exists else "settings")
        if exists:
            self.tabview.delete("embedding")
        self.embedding_tab(self.tabview.insert(index, "embedding"))
        if was_current:
            self.tabview.set("embedding")

    def load_preset(self):
        if not self.tabview:
            return

        if self.additional_embeddings_tab:
            self.additional_embeddings_tab.refresh_ui()
        # the concept and sample files the loaded config names (at startup the tabs are built from them afterwards)
        for tab in (getattr(self, "concepts_tab", None), getattr(self, "sampling_tab", None)):
            if tab is not None:
                tab.reload_from_config()

    def _set_training_button_style(self, mode: str):
        if not self.training_button:
            return
        style = self._TRAIN_BUTTON_STYLES.get(mode)
        if not style:
            return
        self.training_button.configure(**style)

    def export_training(self):
        file_path = filedialog.asksaveasfilename(filetypes=[
            ("All Files", "*.*"),
            ("json", "*.json"),
        ], initialdir=".", initialfile="config.json")
        if file_path:
            self.controller.export_training(file_path)

    def generate_debug_package(self):
        zip_path = filedialog.askdirectory(
            initialdir=".",
            title="Select Directory to Save Debug Package"
        )
        if not zip_path:
            return
        self.controller.generate_debug_package(Path(zip_path) / "OneTrainer_debug_report.zip")
