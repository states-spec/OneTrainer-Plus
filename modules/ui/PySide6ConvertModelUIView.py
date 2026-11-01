from modules.ui.BaseConvertModelUIView import BaseConvertModelUIView
from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QDialog, QGridLayout, QWidget


class PySide6ConvertModelUIView(BaseConvertModelUIView, QDialog):
    def __init__(self, parent, controller: ConvertModelUIController):
        QDialog.__init__(self, parent)
        BaseConvertModelUIView.__init__(self, pyside6_components)

        # delete on close: a parented QDialog is otherwise kept after exec() returns, one more each time the tool opens
        # (the conversion runs on the UI thread, so nothing reports to the window after it closed)
        self.finished.connect(self.deleteLater)

        self.controller = controller
        self.ui_state = PySide6UIState(controller.convert_model_args)
        self._dynamic_widgets = []

        self.setWindowTitle("Convert models")
        self.resize(600, 380)

        _pad = pyside6_components.PAD
        outer = QGridLayout(self)
        outer.setContentsMargins(_pad, _pad, _pad, _pad)

        self._frame = QWidget(self)
        self._layout = pyside6_components._layout(self._frame)
        self._layout.setColumnStretch(1, 1)
        outer.addWidget(self._frame, 0, 0)

        self.build_content(self._frame, controller, self.ui_state, self._rebuild_dynamic_ui)
        self._rebuild_dynamic_ui()
        self._layout.setRowStretch(self._layout.rowCount(), 1)

    def _rebuild_dynamic_ui(self, *args):
        for widget in self._dynamic_widgets:
            widget.hide()
            widget.deleteLater()

        before = set(self._frame.children())
        self.build_dynamic_content(self._frame, self.controller, self.ui_state)
        self._dynamic_widgets = [c for c in self._frame.children() if c not in before and isinstance(c, QWidget)]

    def set_converting(self, active):
        self.button.setEnabled(not active)
