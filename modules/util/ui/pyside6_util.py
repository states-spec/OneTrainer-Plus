import html
import locale
import os
import re
import signal
import sys
from abc import ABCMeta

from modules.util.config.UISettingsConfig import UISettingsConfig
from modules.util.enum.UITheme import UITheme

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPalette, qGray
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QProgressBar,
    QStackedWidget,
    QStyleFactory,
    QTabWidget,
    QToolTip,
    QWidget,
)


class QtABCMeta(type(QWidget), ABCMeta):
    # Combined metaclass that resolves the conflict between Qt's Shiboken metaclass and ABCMeta.
    pass


# set once, at startup: changing the application style sheet makes Qt polish every widget again once per ancestor
# (see _restyle_widgets), so nothing in it may depend on the theme or the font size. The checkbox rule
# (_checkbox_indicator_style_sheet) is added then.
_STYLESHEET = """
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
        padding: 2px 2px;
    }
    QToolButton {
        padding-top: 0px;
        padding-bottom: 0px;
        padding-right: 40px;
    }
    QToolButton::menu-indicator {
        subcontrol-origin: padding;
        subcontrol-position: right center;
        width: 12px;
        height: 12px;
        right: 10px;
    }
"""

# the trough of a progress bar, which depends on the theme, so each bar has it in its own style sheet
_PROGRESS_BAR_STYLESHEET = "QProgressBar { background-color: %s; }"

# The background of every tab page. Pages without one of their own (LoRA, settings) showed the tab widget's frame, which
# Fusion paints lighter than the window in the dark theme, while scroll areas fill theirs with the window color. The
# page area of each tab widget takes this role (style_tab_widget), and pages and the scroll areas in them inherit it:
# the window color in the dark theme, the tab frame's color in the light one. Neither Fusion nor OneTrainer uses it
# otherwise (it is the color of every other row in item views that alternate their row colors; none here do).
_TAB_PAGE_ROLE = QPalette.ColorRole.AlternateBase

# the light look OneTrainer always had, captured in create_application; the font size the platform picked and the
# height of a line in it
_light_palette: QPalette | None = None
_default_font_point_size: float | None = None
_default_line_height = 16
_current_theme: UITheme | None = None
_dark = False  # whether the dark palette is applied
_requesting_color_scheme = False
_system_file_dialogs = False


def _dark_palette() -> QPalette:
    window = QColor(45, 45, 45)
    button = QColor(53, 53, 53)
    base = QColor(30, 30, 30)
    text = QColor(230, 230, 230)
    disabled_text = QColor(128, 128, 128)
    highlight = QColor(42, 130, 218)

    palette = QPalette()
    for role, color in (
            (QPalette.ColorRole.Window, window),
            (QPalette.ColorRole.WindowText, text),
            (QPalette.ColorRole.Base, base),
            (_TAB_PAGE_ROLE, window),
            (QPalette.ColorRole.ToolTipBase, button),
            (QPalette.ColorRole.ToolTipText, text),
            (QPalette.ColorRole.PlaceholderText, disabled_text),
            (QPalette.ColorRole.Text, text),
            (QPalette.ColorRole.Button, button),
            (QPalette.ColorRole.ButtonText, text),
            (QPalette.ColorRole.BrightText, QColor(255, 80, 80)),
            (QPalette.ColorRole.Link, QColor(86, 160, 240)),
            (QPalette.ColorRole.Highlight, highlight),
            (QPalette.ColorRole.HighlightedText, QColor("white")),
            (QPalette.ColorRole.Light, QColor(75, 75, 75)),
            (QPalette.ColorRole.Midlight, QColor(62, 62, 62)),
            (QPalette.ColorRole.Mid, QColor(40, 40, 40)),
            (QPalette.ColorRole.Dark, QColor(25, 25, 25)),
            (QPalette.ColorRole.Shadow, QColor(10, 10, 10)),
    ):
        palette.setColor(role, color)

    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(52, 52, 52))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(48, 48, 48))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, disabled_text)
    return palette


def _tab_frame_color(palette: QPalette) -> QColor:
    # the color Fusion fills a tab widget's frame with (QFusionStylePrivate::tabFrameColor, Qt 6.11)
    color = palette.button().color()
    color = color.lighter(100 + max(1, (180 - qGray(color.rgb())) // 6))
    color.setHsv(color.hue(), int(color.saturation() * 0.75), color.value(), color.alpha())
    return color.lighter(104)


def style_tab_widget(tab_widget: QTabWidget):
    # the same background on every page, see _TAB_PAGE_ROLE
    pages = tab_widget.findChild(QStackedWidget, "qt_tabwidget_stackedwidget", Qt.FindChildOption.FindDirectChildrenOnly)
    pages.setBackgroundRole(_TAB_PAGE_ROLE)
    pages.setAutoFillBackground(True)


def _checkbox_indicator_style_sheet(size_at_default_font: int) -> str:
    # A checkbox indicator that is size_at_default_font pixels with the platform's default font size and grows and
    # shrinks with the text when the Font Size setting changes it: em is the line height of the application font in
    # Qt style sheets, recomputed when the widgets are polished again
    size = f"{size_at_default_font / _default_line_height:.4f}em"
    return f"QCheckBox::indicator {{ width: {size}; height: {size}; }}"


def _is_dark(app: QApplication, theme: UITheme) -> bool:
    if theme == UITheme.SYSTEM:
        # dark unless the OS asks for light: GNOME's "Default" style and desktops without a color scheme setting report
        # no preference (Unknown)
        return app.styleHints().colorScheme() != Qt.ColorScheme.Light
    return theme == UITheme.DARK


def format_tooltip(text: str) -> str:
    # wrap tooltips after 45 characters, supported by studys size
    wrapped = re.sub(r"(.{65}\S*)[ \t]+", r"\1\n", text)
    return f'<p style="white-space: pre;">{html.escape(wrapped)}</p>'


def is_dark_theme() -> bool:
    # whether the UI shows the dark palette now (with the System theme, whatever the OS reports)
    app = QApplication.instance()
    return app is not None and _current_theme is not None and _is_dark(app, _current_theme)


def apply_theme(app: QApplication, theme: UITheme):
    global _current_theme, _dark, _requesting_color_scheme
    _current_theme = theme

    # also tells the platform, e.g. for a dark title bar on Windows
    _requesting_color_scheme = True
    try:
        if theme == UITheme.SYSTEM:
            app.styleHints().unsetColorScheme()
        else:
            app.styleHints().setColorScheme(Qt.ColorScheme.Dark if theme == UITheme.DARK else Qt.ColorScheme.Light)
    finally:
        _requesting_color_scheme = False

    _dark = _is_dark(app, theme)
    palette = _dark_palette() if _dark else QPalette(_light_palette)
    app.setPalette(palette)
    QToolTip.setPalette(palette)
    if not app.styleSheet():
        # at startup, before there are widgets
        app.setStyleSheet(_STYLESHEET + _checkbox_indicator_style_sheet(16))
    _restyle_widgets(app)


def apply_font_size(app: QApplication, point_size: int):
    font = app.font()
    font.setPointSizeF(point_size if point_size > 0 else _default_font_point_size)
    if font != app.font():
        app.setFont(font)
        _restyle_widgets(app)


def style_progress_bar(bar: QProgressBar):
    # every QProgressBar goes through this when it is created; theme changes call it again
    style_sheet = _PROGRESS_BAR_STYLESHEET % ("#505050" if _dark else "#c8c8c8")
    if bar.styleSheet() != style_sheet:
        bar.setStyleSheet(style_sheet)


def _restyle_widgets(app: QApplication):
    # A widget polished by the application style sheet keeps the palette and font it had then, so a new palette or
    # font only shows after every widget is polished again. Setting the style sheet again does that, but Qt
    # (QStyleSheetStyle::repolish) polishes the subtree of every widget it lists, and it lists them all: each widget
    # is polished once per ancestor, 0.5 s for the main window with a dozen concepts and 1.5-1.8 s with 150 (3000
    # widgets). Here each widget is polished once, parents first (a widget takes its font and palette from its
    # parent then), and gets the StyleChange event after its children (a scroll area sizes its contents then, also
    # in a hidden tab): the same fonts, palettes, sizes and pixels in a fraction of the time.
    polished = []
    stack = [widget for widget in app.topLevelWidgets() if widget.parentWidget() is None]
    while stack:
        widget = stack.pop()
        if isinstance(widget, QProgressBar):
            style_progress_bar(widget)
        widget.style().polish(widget)
        polished.append(widget)
        stack.extend(widget.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly))
    style_change = QEvent(QEvent.Type.StyleChange)
    for widget in reversed(polished):
        QCoreApplication.sendEvent(widget, style_change)


def apply_ui_settings(settings: UISettingsConfig):
    # the UI scale is applied by create_application, Qt can't change it while running
    global _system_file_dialogs
    _system_file_dialogs = settings.system_file_dialogs
    app = QApplication.instance()
    if settings.theme != _current_theme:
        apply_theme(app, settings.theme)
    apply_font_size(app, settings.font_size)


def _on_color_scheme_changed(_scheme):
    # the OS switched between light and dark. Also emitted while apply_theme requests a scheme from the platform,
    # which applies the palette itself, and then nothing changes here
    app = QApplication.instance()
    if _current_theme == UITheme.SYSTEM and not _requesting_color_scheme and _is_dark(app, UITheme.SYSTEM) != _dark:
        apply_theme(app, UITheme.SYSTEM)


def _gtk_based_desktop() -> bool:
    # the desktops for which Qt picks its GTK3 platform theme (QGenericUnixTheme::themeNames)
    desktops = os.environ.get("XDG_CURRENT_DESKTOP", "").upper().split(":")
    return any(d in ("GNOME", "X-CINNAMON", "UNITY", "MATE", "XFCE", "LXDE", "PANTHEON") for d in desktops)


def create_application() -> QApplication:
    global _light_palette, _default_font_point_size, _default_line_height

    # Restore the OS default SIGINT handler so Ctrl+C terminates the process
    # directly at the C level. Qt's event loop blocks inside C++, so Python's
    # own SIGINT handler would never get a chance to run while app.exec() is
    # active and Ctrl+C would be ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    settings = UISettingsConfig.load()
    global _system_file_dialogs
    _system_file_dialogs = settings.system_file_dialogs
    # Qt reads the scale factor once, when the application is created. A value set in the environment wins.
    if settings.ui_scale != 1.0 and "QT_SCALE_FACTOR" not in os.environ:
        os.environ["QT_SCALE_FACTOR"] = str(settings.ui_scale)
    # On GTK-based desktops Qt loads its GTK3 platform theme, which starts the desktop's GTK (and its threads) inside
    # the pip-installed Qt. On Fedora GNOME a thread then jumped to a garbage address and crashed the UI after a
    # dropdown click. Qt's built-in GNOME theme uses no GTK. The GTK theme stays when System File Dialogs is on (it
    # provides the GTK dialog) or QT_QPA_PLATFORMTHEME is set.
    if sys.platform.startswith("linux") and not settings.system_file_dialogs \
            and "QT_QPA_PLATFORMTHEME" not in os.environ and _gtk_based_desktop():
        os.environ["QT_QPA_PLATFORMTHEME"] = "gnome"
    # Qt messages that tell a user nothing they could act on (OT_DEBUG_WARNINGS shows them): on desktops without the
    # xdg-desktop-portal Settings interface, two "qt.qpa.theme.gnome: dbus reply error ...
    # org.freedesktop.portal.Settings" lines while Qt probes for the system color scheme. Qt reads the rules when its
    # logging initializes at QApplication init; rules already in QT_LOGGING_RULES come after these, so they win.
    if not os.environ.get("OT_DEBUG_WARNINGS"):
        rules = ["qt.qpa.theme.gnome=false"]
        if os.environ.get("QT_LOGGING_RULES"):
            rules.append(os.environ["QT_LOGGING_RULES"])
        os.environ["QT_LOGGING_RULES"] = ";".join(rules)

    app = QApplication(sys.argv)
    # QApplication initializes the C locale from the environment (setlocale(LC_ALL, "")), which sets LC_NUMERIC
    # to a locale whose decimal separator may be a comma. C libraries then misparse '.' floats: protobuf's upb
    # backend rejects sentencepiece's schema ("Invalid default '0.9995'"), breaking every sentencepiece tokenizer
    # (T5/Chroma/Flux), and the Kineto profiler writes comma decimals into its JSON traces. Restore the C numeric
    # locale, which Python itself uses by default. Qt's own display uses QLocale, which is independent of this and
    # keeps formatting numbers per the system locale.
    locale.setlocale(locale.LC_NUMERIC, "C")
    # Force Fusion everywhere: native styles (e.g. windowsvista) draw standard
    # controls via OS theme APIs, which breaks once an application stylesheet
    # is set, producing a flatter look than Fusion's own stylesheet-aware painting.
    app.setStyle(QStyleFactory.create("Fusion"))
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    _light_palette = QPalette(app.palette())
    _light_palette.setColor(QPalette.ColorRole.Base, QColor("white"))
    _light_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    _light_palette.setColor(_TAB_PAGE_ROLE, _tab_frame_color(_light_palette))
    _default_font_point_size = app.font().pointSizeF()
    _default_line_height = max(1, QFontMetrics(app.font()).height())

    apply_theme(app, settings.theme)
    apply_font_size(app, settings.font_size)
    app.styleHints().colorSchemeChanged.connect(_on_color_scheme_changed)

    return app


def file_dialog_options(options: QFileDialog.Option | None = None) -> QFileDialog.Option:
    # On Linux, Qt's "native" dialog is the desktop's dialog code (GTK, the xdg portal, or KDE's plugins) loaded into
    # the pip-installed Qt, which freezes or crashes OneTrainer on some desktops. Qt's own dialog doesn't depend on it.
    if options is None:
        options = QFileDialog.Option(0)
    if sys.platform.startswith("linux") and not _system_file_dialogs:
        options |= QFileDialog.Option.DontUseNativeDialog
    return options
