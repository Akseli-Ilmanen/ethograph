"""Add-panel popup: a transient, searchable list of Media / Feature sources.

Opened from the "➕ Add panel" button in the bottom playback bar (or Shift+N).
The popup overlays the plot area and vanishes on focus-out. Two workflows:

* **Drag** an entry onto the plot container (:class:`UnifiedPanelContainer`)
  — drops a ``kind|name`` payload; the container calls back into
  :meth:`MetaWidget._on_source_dropped`, which opens :class:`PlotTypePicker`
  (navigable with ↑/↓) to choose the plot type and then creates the panel.
* **Enter / double-click** an entry — same flow, but the panel is created at
  its default location; the user arranges it afterwards by dragging the
  panel's dock title bar.

The filter box has keyboard focus on open; ↑/↓ move the list selection while
typing. Feature plot-type options are gated by data shape ``(T, N)``:

* ``Lineplot`` — always available.
* ``Heatmap`` / ``Space (2D)`` — need ``N >= 2``.
* ``Space (3D)`` — needs ``N >= 3``.
* ``Radial`` — needs the feature's dims to pin down to ONE column whose values
  span a full turn (360° or 2π), which is what distinguishes a heading from any
  other 1-D signal.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import QMimeData, QPoint, Qt
from qtpy.QtGui import QDrag
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.video_feature_files import video_feature_sources

from .plots_radial import feature_angular_unit

logger = logging.getLogger(__name__)

SOURCE_MIME = "application/x-ethograph-source"

#: Sentinel source name for the popup's "Image — browse…" entry.
IMAGE_BROWSE = "__browse__"
#: The popup's "Video features — browse…" entry: a folder of ``{video stem}.npy`` becomes a heatmap.
VIDEO_FEATURES_BROWSE = "__browse_video_features__"

_ROLE_KIND = Qt.UserRole
_ROLE_NAME = Qt.UserRole + 1
_ROLE_HEADER = Qt.UserRole + 2


def feature_ncols(app_state, feature: str) -> int:
    """Number of non-time columns ``N`` for a feature (product of extra dims).

    Asks the loader's ``feature_dims`` first — ``app_state.ds`` is None for
    pynapple data, so reading only the dataset answered 1 for every pynapple
    feature and hid the Heatmap/Space options exactly there.
    """
    loader = getattr(app_state, "data_loader", None)
    derived = getattr(loader, "derived", None)
    if isinstance(derived, dict) and feature in derived:
        return derived[feature].n_columns
    if loader is not None and hasattr(loader, "feature_dims"):
        dims = loader.feature_dims(feature)
        if dims:
            n = 1
            for values in dims.values():
                n *= max(1, len(values))
            return n
        # {} can also mean "loader knows nothing" — fall through to the ds.
    ds = getattr(app_state, "ds", None)
    if ds is None or feature not in getattr(ds, "data_vars", {}):
        return 1
    da = ds[feature]
    n = 1
    for dim, size in zip(da.dims, da.shape):
        if "time" not in str(dim).lower():
            n *= int(size)
    return max(1, n)


def allowed_plot_types(kind: str, name: str, app_state) -> list[str]:
    """Plot types offered for a dropped source, gated by data shape."""
    if kind == "audio":
        return ["Audio Trace", "Spectrogram Trace"]
    if kind == "video":
        return ["Camera view"]
    if kind == "image":
        return ["Image"]
    if kind == "video_features":
        return ["Heatmap"]
    if kind == "neo":
        return ["Neo Trace"]
    if kind == "console":
        return ["Python console"]
    if kind == "labels":
        return ["Label timeline"]
    if kind == "predictions":
        return ["Prediction timeline"]
    if kind == "phy":
        return ["Phy TraceView"]
    if kind == "feature":
        options = ["Lineplot"]
        n = feature_ncols(app_state, name)
        if n >= 2:
            options += ["Heatmap", "Space (2D)"]
        if n >= 3:
            options.append("Space (3D)")
        # A compass shows ONE heading, so it is offered for any feature whose
        # dims pin down to a single column covering a full turn — anything else
        # has no direction. The raw column count says nothing here: a heading
        # normally carries a keypoint or individual dim like every other
        # feature, and gating on N == 1 hid the option from exactly those.
        if feature_angular_unit(app_state, name) is not None:
            options.append("Radial")
        return options
    return []


class PlotTypePicker(QDialog):
    """Tiny modal list to choose one option (plot type, channel, …);
    ↑/↓ to move, Enter to accept."""

    def __init__(self, options: list[str], parent=None, title: str = "Plot type"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.choice: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._list = QListWidget()
        self._list.addItems(options)
        if options:
            self._list.setCurrentRow(0)
        self._list.itemActivated.connect(self._accept_item)
        self._list.itemDoubleClicked.connect(self._accept_item)
        layout.addWidget(self._list)
        self._list.setFocus()
        self.resize(200, 30 + 22 * max(1, len(options)))

    def _accept_item(self, item):
        self.choice = item.text()
        self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            item = self._list.currentItem()
            if item is not None:
                self._accept_item(item)
            return
        super().keyPressEvent(event)


class ChannelSelectDialog(QDialog):
    """Multi-select channel picker for a Neo stream. Defaults to all channels;
    returns the chosen 0-based channel indices (a Neo panel shows them as a
    stacked multi-channel trace)."""

    def __init__(self, n_channels: int, channel_names=None, parent=None, title: str = "Select channels"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(240, min(460, 120 + 22 * max(1, n_channels)))
        names = channel_names or [f"Ch {i}" for i in range(n_channels)]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._all_cb = QCheckBox("(All)")
        self._all_cb.setChecked(True)
        self._all_cb.toggled.connect(self._on_all)
        layout.addWidget(self._all_cb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(1)
        self._checks: list[QCheckBox] = []
        for i in range(n_channels):
            cb = QCheckBox(str(names[i]) if i < len(names) else f"Ch {i}")
            cb.setChecked(True)
            cb.toggled.connect(self._on_item)
            holder_layout.addWidget(cb)
            self._checks.append(cb)
        holder_layout.addStretch()
        scroll.setWidget(holder)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_all(self, checked: bool):
        for cb in self._checks:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

    def _on_item(self, _):
        self._all_cb.blockSignals(True)
        self._all_cb.setChecked(all(cb.isChecked() for cb in self._checks))
        self._all_cb.blockSignals(False)

    def selected_channels(self) -> list[int] | None:
        """Chosen channel indices; None when all are selected (show everything)."""
        chosen = [i for i, cb in enumerate(self._checks) if cb.isChecked()]
        if len(chosen) == len(self._checks):
            return None
        return chosen


class _SourceList(QListWidget):
    """A list whose data items (not headers) are draggable as sources."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def startDrag(self, _actions):
        item = self.currentItem()
        if item is None or item.data(_ROLE_HEADER):
            return
        kind = item.data(_ROLE_KIND)
        name = item.data(_ROLE_NAME)
        mime = QMimeData()
        mime.setData(SOURCE_MIME, f"{kind}|{name}".encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        # Hide the popup so the plot area is unobstructed while dragging.
        self.window().hide()
        drag.exec_(Qt.CopyAction)


class SourcePopup(QWidget):
    """Transient searchable list of Media + Feature sources.

    ``on_activate(kind, name)`` (set by MetaWidget) is called when an entry
    is chosen with Enter or a double-click; dragging out uses the normal
    drag-and-drop path into the plot container.
    """

    _WIDTH = 230
    _MAX_HEIGHT = 420
    _ROW_HEIGHT = 20

    def __init__(self, app_state, parent=None):
        super().__init__(parent, Qt.Popup)
        self.app_state = app_state
        self.on_activate = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._filter.installEventFilter(self)
        layout.addWidget(self._filter)

        self._list = _SourceList()
        self._list.itemActivated.connect(self._activate_item)
        self._list.itemDoubleClicked.connect(self._activate_item)
        layout.addWidget(self._list)

        hint = QLabel("Drag onto the plot area, or press Enter.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgba(255,255,255,140); font-size: 8pt;")
        layout.addWidget(hint)

    # ------------------------------------------------------------------
    # Population (same catalog-driven list the old left sidebar showed)
    # ------------------------------------------------------------------

    def _add_header(self, text: str):
        item = QListWidgetItem(text)
        item.setData(_ROLE_HEADER, True)
        item.setFlags(Qt.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self._list.addItem(item)

    def _add_source(self, label: str, kind: str, name: str) -> QListWidgetItem:
        item = QListWidgetItem(f"  {label}")
        item.setData(_ROLE_HEADER, False)
        item.setData(_ROLE_KIND, kind)
        item.setData(_ROLE_NAME, name)
        self._list.addItem(item)
        return item

    def refresh(self, catalog=None, neo_streams=None, phy_available=False):
        """Repopulate from the current session (cameras, mics, features).

        *catalog* is the loaded DataCatalog: its ``feature_choices()`` is the
        canonical feature list (same one the features combo displays), so
        every offered feature is representable in the sidebar controls.
        *neo_streams* are Neo stream/modality display names (EMG, accelerometer,
        …); each is a "neo" source whose channels are picked on drop.
        *phy_available* adds the raw-data "Ephys (Phy-like viewer)" source.
        """
        self._list.clear()
        sio = getattr(self.app_state, "nwb_alignment", None)
        cameras = list(getattr(sio, "cameras", []) or []) if sio else []
        mics = list(getattr(sio, "mics", []) or []) if sio else []

        # An empty time axis carrying only the label overlay — always offered,
        # first, because it needs no data: it is how a video-only session
        # gets to see the labels it places.
        self._add_header("Labels")
        self._add_source("Label timeline", "labels", "labels")
        for prediction_set in getattr(self.app_state, "prediction_sets", None) or []:
            self._add_source(f"Predictions ({prediction_set.name})", "predictions", str(prediction_set.path))

        self._add_header("Media")
        for cam in cameras:
            self._add_source(f"Video ({cam})", "video", str(cam))
        # One entry per mic/audio file (all its channels). The channel is
        # picked when the panel is created and can be changed later via the
        # sidebar's Channel combo.
        audio_names = [str(m) for m in mics]
        if not audio_names:
            audio_names = list(getattr(self.app_state, "audio_mic_channels", None) or {})
        if not audio_names:
            audio_names = list(getattr(self.app_state, "audio_source_map", None) or {})
        for name in audio_names:
            self._add_source(f"Audio ({name})", "audio", name)
        # Static images (arena photo, reference frame): each dropped/browsed
        # image is a Media source; a browse entry lets the user add one now.
        for img in getattr(self.app_state, "image_paths", None) or []:
            self._add_source(f"Image ({Path(img).name})", "image", str(img))
        self._add_source("Static image — browse…", "image", IMAGE_BROWSE)

        # Ephys sources: the raw-data Phy trace (singleton, re-showable) plus
        # one Neo source per stream/modality (EMG, accelerometer, amplifier…).
        # Dropping a Neo source opens a channel picker; each drop = a new panel.
        if phy_available or neo_streams:
            self._add_header("Ephys")
            if phy_available:
                self._add_source("Phy (Multi-channel trace)", "phy", "phy")
            for stream in neo_streams or []:
                self._add_source(f"Neo ({stream})", "neo", str(stream))

        features: list[str] = catalog.feature_choices() if catalog is not None else []
        ds = getattr(self.app_state, "ds", None)
        if not features and ds is not None:
            features = list(ds.data_vars)
        # Video features from files (a folder of {video stem}.npy, e.g. an
        # embedding export) are listed with where they came from, so two
        # exports of one model with different settings stay tellable apart;
        # the browse entry adds another — dropping a folder on the window
        # does the same.
        folders = video_feature_sources(ds) if ds is not None else {}
        self._add_header("Features")
        for feat in features:
            if feat not in folders:
                self._add_source(feat, "feature", str(feat))
        self._add_header("Video features")
        for feat in features:
            if feat in folders:
                item = self._add_source(f"{feat}  ({folders[feat]})", "feature", str(feat))
                item.setToolTip(folders[feat])
        item = self._add_source("Video features — browse a folder of .npy…", "video_features", VIDEO_FEATURES_BROWSE)
        item.setToolTip(
            "One {video stem}.npy per trial, matched to the session's videos by name. "
            "Dropping a folder on the window does the same."
        )

        # A dockable Python console over the plotted arrays: click a feature
        # panel to bind what it shows, assign to make new features.
        self._add_header("Tools")
        self._add_source("Python console", "console", "console")

        self._apply_filter(self._filter.text())

    # ------------------------------------------------------------------
    # Filtering + keyboard navigation
    # ------------------------------------------------------------------

    def _apply_filter(self, text: str):
        """Hide non-matching data items; hide headers whose group is empty."""
        needle = text.strip().lower()
        header_item = None
        header_has_match = False
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(_ROLE_HEADER):
                if header_item is not None:
                    header_item.setHidden(not header_has_match)
                header_item, header_has_match = item, False
                continue
            match = not needle or needle in item.text().lower()
            item.setHidden(not match)
            header_has_match = header_has_match or match
        if header_item is not None:
            header_item.setHidden(not header_has_match)
        self._select_first_visible()

    def _visible_data_rows(self) -> list[int]:
        return [
            i
            for i in range(self._list.count())
            if not self._list.item(i).isHidden() and not self._list.item(i).data(_ROLE_HEADER)
        ]

    def _select_first_visible(self):
        rows = self._visible_data_rows()
        self._list.setCurrentRow(rows[0] if rows else -1)

    def _move_selection(self, delta: int):
        rows = self._visible_data_rows()
        if not rows:
            return
        current = self._list.currentRow()
        pos = rows.index(current) if current in rows else 0
        self._list.setCurrentRow(rows[(pos + delta) % len(rows)])

    def eventFilter(self, obj, event):
        """↑/↓/Enter in the filter box drive the list selection."""
        if obj is self._filter and event.type() == event.Type.KeyPress:
            key = event.key()
            if key == Qt.Key_Down:
                self._move_selection(+1)
                return True
            if key == Qt.Key_Up:
                self._move_selection(-1)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                item = self._list.currentItem()
                if item is not None:
                    self._activate_item(item)
                return True
        return super().eventFilter(obj, event)

    def _activate_item(self, item):
        if item is None or item.data(_ROLE_HEADER):
            return
        kind = item.data(_ROLE_KIND)
        name = item.data(_ROLE_NAME)
        self.hide()
        if callable(self.on_activate):
            self.on_activate(kind, name)

    # ------------------------------------------------------------------
    # Showing
    # ------------------------------------------------------------------

    def popup_at(self, global_pos: QPoint, open_upward: bool = False):
        """Show the popup at *global_pos* (top-left; bottom-left if upward)."""
        n_rows = self._list.count()
        height = min(self._MAX_HEIGHT, 80 + self._ROW_HEIGHT * max(3, n_rows))
        self.resize(self._WIDTH, height)
        pos = global_pos - QPoint(0, height) if open_upward else global_pos
        self.move(pos)
        self._filter.clear()
        self.show()
        self._filter.setFocus()
        self._select_first_visible()
