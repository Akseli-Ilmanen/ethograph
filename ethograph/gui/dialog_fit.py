"""Every ethograph dialog is resizable and fits the screen.

A dialog's minimum size is its layout's minimum, so a tall dialog on a small
screen can neither be shrunk nor scrolled. The first time a dialog is shown,
its content moves into a scroll area (the trailing button box stays pinned
below it), a size grip is enabled, and the window is capped to the screen.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_FITTED = "_ethograph_fitted"
#: Share of the screen's available area a dialog may take on first show.
_SCREEN_FRACTION = 0.92


def _is_ours(dialog: QDialog) -> bool:
    """Qt's own dialogs (message boxes, file/colour pickers) manage their own size."""
    return type(dialog).__module__.startswith("ethograph.")


def _already_scrolls(layout: QLayout) -> bool:
    items = [layout.itemAt(i) for i in range(layout.count())]
    widgets = [item.widget() for item in items if item is not None]
    content = [w for w in widgets if w is not None and not isinstance(w, QDialogButtonBox)]
    return len(content) == 1 and isinstance(content[0], QScrollArea)


def fit_dialog(dialog: QDialog) -> None:
    """Make ``dialog`` resizable, scrollable and no larger than its screen."""
    dialog.setProperty(_FITTED, True)
    dialog.setSizeGripEnabled(True)
    layout = dialog.layout()
    if layout is None:
        return
    if layout.sizeConstraint() == QLayout.SizeConstraint.SetFixedSize:
        layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

    natural = dialog.size().expandedTo(dialog.sizeHint())
    if not _already_scrolls(layout):
        scroll = _wrap_in_scroll_area(dialog, layout)
        # Room for the vertical bar, so a clamped height never adds a horizontal one.
        bar = scroll.verticalScrollBar()
        assert bar is not None
        natural.setWidth(natural.width() + bar.sizeHint().width())

    screen = (dialog.parentWidget() or dialog).screen()
    if screen is None:
        return
    available = screen.availableGeometry().size() * _SCREEN_FRACTION
    dialog.setMinimumSize(dialog.minimumSize().boundedTo(available))
    dialog.resize(natural.boundedTo(available))


def _wrap_in_scroll_area(dialog: QDialog, layout: QLayout) -> QScrollArea:
    buttons = None
    last = layout.itemAt(layout.count() - 1) if layout.count() else None
    if last is not None and isinstance(last.widget(), QDialogButtonBox):
        buttons = last.widget()
        layout.removeWidget(buttons)

    margins = layout.contentsMargins()
    content = QWidget()
    # Steals the layout from the dialog and reparents its widgets into content.
    content.setLayout(layout)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)

    outer = QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, margins.bottom())
    outer.setSpacing(0)
    outer.addWidget(scroll)
    if buttons is not None:
        outer.addSpacing(margins.bottom())
        button_row = QVBoxLayout()
        button_row.setContentsMargins(margins.left(), 0, margins.right(), 0)
        button_row.addWidget(buttons)
        outer.addLayout(button_row)
    return scroll


class DialogFitter(QObject):
    """Application event filter applying :func:`fit_dialog` on a dialog's first show."""

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if (
            event is not None
            and event.type() == QEvent.Type.Show
            and isinstance(obj, QDialog)
            and obj.isWindow()
            and obj.windowType() != Qt.WindowType.Popup
            and not obj.property(_FITTED)
            and _is_ours(obj)
        ):
            fit_dialog(obj)
        return False


def install_dialog_fitter(app: QApplication) -> DialogFitter:
    fitter = DialogFitter(app)
    app.installEventFilter(fitter)
    return fitter
