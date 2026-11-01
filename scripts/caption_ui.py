from util.import_util import script_imports

script_imports()

from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.CtkCaptionUIView import CtkCaptionUIView
from modules.ui.CtkSettingsTabView import apply_ui_scale, apply_ui_theme
from modules.util.args.CaptionUIArgs import CaptionUIArgs
from modules.util.config.UISettingsConfig import UISettingsConfig

import customtkinter as ctk


def main():
    args = CaptionUIArgs.parse_args()

    # the theme and scale of the UI settings, as in the training UI (customtkinter's own default follows the OS)
    ui_settings = UISettingsConfig.load()
    apply_ui_theme(ui_settings)
    apply_ui_scale(ui_settings)

    # CtkCaptionUIView is a CTkToplevel, so it needs a CTk root. Hide the root and run the
    # toplevel as a standalone window, tearing the root down once the window is closed.
    root = ctk.CTk()
    root.withdraw()
    ui = CaptionUIController(args.dir, args.include_subdirectories).create_window(root, CtkCaptionUIView)
    root.wait_window(ui)
    root.destroy()


if __name__ == '__main__':
    main()
