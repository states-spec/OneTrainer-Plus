from modules.util.app_info import APP_NAME


class BaseSettingsTabView:
    def __init__(self, components):
        self.components = components

    def build(self, frame, controller, ui_state):
        row = 0

        # theme
        self.components.label(frame, row, 0, "Theme",
                              tooltip="Color theme of the UI. System follows the light or dark mode of the operating system.")
        self.components.options_kv(frame, row, 1, controller.get_themes(), ui_state, "theme")
        self.__restart_note(frame, row, controller, "theme")
        row += 1

        # scale
        self.components.label(frame, row, 0, "UI Scale",
                              tooltip="Size of the whole UI: text, widgets and windows.")
        self.components.options_kv(frame, row, 1, controller.get_ui_scales(), ui_state, "ui_scale")
        self.__restart_note(frame, row, controller, "ui_scale")
        row += 1

        # font size
        if controller.supports_font_size:
            self.components.label(frame, row, 0, "Font Size",
                                  tooltip="Size of the UI text. Default uses the operating system's font size. Larger text needs a larger window; to enlarge everything evenly, use UI Scale instead.")
            self.components.options_kv(frame, row, 1, controller.get_font_sizes(), ui_state, "font_size")
            row += 1

        # window
        self.components.label(frame, row, 0, "Remember Window Size",
                              tooltip="Open the main window with the size it had when it was last closed.")
        self.components.switch(frame, row, 1, ui_state, "remember_window_size")
        row += 1

        self.components.label(frame, row, 0, "Start Maximized",
                              tooltip="Open the main window maximized.")
        self.components.switch(frame, row, 1, ui_state, "start_maximized")
        row += 1

        if controller.supports_system_file_dialogs:
            self.components.label(frame, row, 0, "System File Dialogs",
                                  tooltip="Use the desktop's file and folder dialogs (GTK, portal or KDE) for the \"...\" buttons instead of "
                                          f"{APP_NAME}'s own. Off by default: on some Linux desktops the system dialog freezes or crashes {APP_NAME}. "
                                          "On GTK desktops (GNOME, Cinnamon, MATE, Xfce) the GTK dialog needs a restart of the UI after switching this on.")
            self.components.switch(frame, row, 1, ui_state, "system_file_dialogs")
            row += 1

        self.components.button(frame, row, 1, "Reset to Defaults", lambda: controller.reset(ui_state),
                               tooltip="Restore the default UI settings (dark theme, 100% scale).")
        row += 1

        self.components.label(frame, row, 0, "Saved in",
                              tooltip="UI settings are kept apart from the training config, so presets and saved configs don't change them.")
        self.components.label(frame, row, 1, controller.get_settings_path())

    def __restart_note(self, frame, row, controller, name):
        if name in controller.needs_restart:
            self.components.label(frame, row, 2, f"Applies after restarting {APP_NAME}")
