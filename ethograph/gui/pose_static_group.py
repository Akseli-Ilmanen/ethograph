"""Static keypoints of the Refine Pose stage: tag them, then propagate them.

A landmark that never moves — the corner of the table, a fixed marker — is
labelled once and read as that position on every frame
(:attr:`KeypointStore.static_keypoints`). This sidebar group is where a
keypoint is tagged static (it turns bold and coloured in the list, so the
tag visibly took) and where **Propagate from this frame** copies every
static keypoint's position on the frame on screen to all frames: label them
all once in Sequential mode, press the button, and only the moving
keypoints are left to place on the other frames.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import QGroupBox, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout

from ethograph.gui.notify import notify

_STATIC_COLOR = "#f5c542"


class StaticKeypointsGroup(QGroupBox):
    """The sidebar group; :meth:`bind` points it at the dialog being edited."""

    def __init__(self, parent=None):
        super().__init__("Static keypoints", parent)
        self._dialog = None
        self._syncing = False
        box = QVBoxLayout(self)
        hint = QLabel("Tick a landmark that never moves. Label it once, then propagate it to every frame.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgba(255,255,255,150);")
        box.addWidget(hint)
        self.list = QListWidget()
        self.list.setToolTip("Ticked keypoints are static: one position, drawn on every frame.")
        self.list.itemChanged.connect(self._on_item_changed)
        box.addWidget(self.list)
        self.propagate_btn = QPushButton("Propagate static landmarks from this frame to all frames")
        self.propagate_btn.setToolTip(
            "Take every static keypoint's position on the frame on screen\n"
            "and use it on all frames of this folder — the same as labelling\n"
            "it once, made explicit for landmarks the model already found."
        )
        self.propagate_btn.clicked.connect(self.propagate)
        box.addWidget(self.propagate_btn)
        self.refresh()

    def bind(self, dialog) -> None:
        """Edit *dialog*'s store (``None`` empties the group)."""
        self._dialog = dialog
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the list from the bound store's schema and static tags."""
        self._syncing = True
        try:
            self.list.clear()
            store = self._dialog.store if self._dialog is not None else None
            self.propagate_btn.setEnabled(store is not None and bool(store.static_keypoints))
            if store is None:
                return
            for name in store.keypoint_names:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                static = store.is_static(name)
                item.setCheckState(Qt.Checked if static else Qt.Unchecked)
                self._style(item, static)
                self.list.addItem(item)
        finally:
            self._syncing = False

    @staticmethod
    def _style(item: QListWidgetItem, static: bool) -> None:
        font = QFont(item.font())
        font.setBold(static)
        item.setFont(font)
        item.setForeground(QColor(_STATIC_COLOR) if static else QColor("white"))
        item.setText(f"📌 {item.text().lstrip('📌 ')}" if static else item.text().lstrip("📌 "))

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._syncing or self._dialog is None:
            return
        name = item.text().lstrip("📌 ")
        static = item.checkState() == Qt.Checked
        store = self._dialog.store
        if static != store.is_static(name):
            store.set_static(name, static)
            self._dialog.on_static_changed()
        self._syncing = True
        self._style(item, static)
        self._syncing = False
        self.propagate_btn.setEnabled(bool(store.static_keypoints))

    def propagate(self) -> None:
        """Every static keypoint's position on the current frame becomes its position everywhere."""
        if self._dialog is None:
            return
        placed = self._dialog.propagate_static_from_current_frame()
        if placed:
            notify(f"Propagated {placed} static landmark(s) to every frame of this folder.", "info")
        else:
            notify("No static keypoint is placed on this frame — label them here first.", "warning")
