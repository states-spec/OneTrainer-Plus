import subprocess
import sys

from modules.ui.BaseSettingsTabView import BaseSettingsTabView
from modules.ui.SettingsTabController import SettingsTabController
from modules.util.config.UISettingsConfig import UISettingsConfig
from modules.util.enum.UITheme import UITheme
from modules.util.ui import ctk_components

import customtkinter as ctk
import darkdetect


def apply_ui_theme(settings: UISettingsConfig):
    # only at startup: labels in the transparent scrollable tabs keep the background they were created with when the
    # appearance mode changes later
    if settings.theme == UITheme.SYSTEM:
        # more efficient version of ctk.set_appearance_mode("System"), which retrieves the system theme on each main loop iteration
        ctk.set_appearance_mode("Light" if _system_prefers_light() else "Dark")
    else:
        ctk.set_appearance_mode("Dark" if settings.theme == UITheme.DARK else "Light")


def _system_prefers_light() -> bool:
    # an explicit light preference only: no preference is dark, as in the Qt UI. On Linux darkdetect (customtkinter's
    # detection) also says Light for GNOME's "Default" style (color-scheme 'default', i.e. no preference) and when it
    # can't tell, so GNOME's setting is read here
    if sys.platform.startswith("linux"):
        try:
            result = subprocess.run(["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                                    capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return False
        return result.stdout.strip().strip("'") == "prefer-light"
    return darkdetect.theme() == "Light"


def apply_ui_scale(settings: UISettingsConfig):
    if ctk.ScalingTracker.widget_scaling != settings.ui_scale:
        ctk.set_widget_scaling(settings.ui_scale)
        ctk.set_window_scaling(settings.ui_scale)


class CtkSettingsTabView(BaseSettingsTabView):
    def __init__(self, master, controller: SettingsTabController, ui_state):
        BaseSettingsTabView.__init__(self, ctk_components)

        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=1)
        self.build(frame, controller, ui_state)
        frame.pack(fill="both", expand=1)
        controller.watch(ui_state)
