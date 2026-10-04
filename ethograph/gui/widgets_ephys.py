"""Ephys widget — trace controls, neuron jumping (Kilosort/Pynapple)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
import pyqtgraph as pg
from qtpy.QtCore import (
    QItemSelection,
    QItemSelectionModel,
    QRectF,
    Qt,
    Signal,
)
from qtpy.QtGui import QBrush, QColor, QPen, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter1d

from ethograph.features.neural import SpikeTable, build_tsgroup, instantaneous_rate
from ethograph.gui.notify import notify
from ethograph.gui.table_filter import (
    SORT_ROLE,
    CategoryFilterDialog,
    FilterHeaderView,
    MultiColumnFilterProxy,
    NumericFilterDialog,
)
from ethograph.gui.unit_events import UnitEventsArea
from ethograph.labels.feature_events import classes_along
from ethograph.utils.qt import (
    find_combo_index,
    get_combo_value,
    set_combo_to_value,
)

from ..io.catalog import PlotData
from ..io.derived import DerivedFeature, derived_loader_for
from ..io.ephys_loader import _NeoWrapper as _RefNeo
from ..io.ephys_loader import load_ephys
from ..io.plot_sources import FileSource
from ..io.validation import EPHYS_EXTENSIONS_RAW
from .app_constants import CLUSTER_TABLE_MAX_HEIGHT, CLUSTER_TABLE_ROW_HEIGHT
<<<<<<< HEAD
from .heatmap_sort import RASTERMAP_MAX_SAMPLES, RASTERMAP_MIN_ROWS, argmax_window_order, rastermap_order
from .raster_render import (
    MAX_TICK_WIDTH,
    PEAK_ORDERS,
=======
from .heatmap_sort import RASTERMAP_MAX_SAMPLES, RASTERMAP_MIN_ROWS, rastermap_order
from .raster_render import (
    MAX_TICK_WIDTH,
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
    RENDER_MODES,
    ROW_ORDERS,
    TICK_WIDTH_AUTO,
    group_by_color,
    order_units,
)
<<<<<<< HEAD
from .right_context import CONTEXT_TITLE_STYLE
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4

logger = logging.getLogger(__name__)

#: The feature the firing rates are shown as: one column per unit, re-binned for every trial.
FIRING_RATE_FEATURE = "firing_rate"
#: The same units' rate as 1 / inter-spike interval — read across trials only, never a panel's feature.
INSTANT_RATE_FEATURE = "instantaneous_rate"
#: The rates a :class:`FiringRateReader` reads off the spikes.
SPIKE_RATE_FEATURES = (FIRING_RATE_FEATURE, INSTANT_RATE_FEATURE)
#: The dim a firing rate's columns are picked by when it is read across trials.
FIRING_RATE_UNIT_DIM = "unit"
#: Bins x units above which the firing rate is refused — the matrix would not fit in memory.
_FIRING_RATE_MAX_CELLS = 50_000_000

_CLUSTER_COLORS = [
    (228, 26, 28),  # red
    (55, 126, 184),  # blue
    (77, 175, 74),  # green
    (152, 78, 163),  # purple
    (255, 127, 0),  # orange
    (255, 255, 51),  # yellow
    (166, 86, 40),  # brown
    (247, 129, 191),  # pink
    (153, 153, 153),  # grey
    (0, 210, 213),  # cyan
    (180, 210, 36),  # lime
    (240, 60, 100),  # magenta
    (100, 180, 255),  # sky
    (200, 130, 0),  # amber
    (100, 220, 150),  # mint
    (180, 100, 220),  # lavender
    (220, 180, 100),  # sand
    (100, 140, 80),  # olive
    (220, 100, 100),  # coral
    (80, 180, 180),  # teal
]

_RAWIO_TO_DISPLAY = {
    "IntanRawIO": "Intan",
    "OpenEphysBinaryRawIO": "OpenEphys",
    "OpenEphysRawIO": "OpenEphys",
    "NWBIO": "NWB",
    "BlackrockRawIO": "Blackrock",
    "AxonRawIO": "Axon",
    "AxographRawIO": "Axograph",
    "EDFRawIO": "EDF",
    "BrainVisionRawIO": "BrainVision",
    "Spike2RawIO": "Spike2",
    "NeuralynxRawIO": "Neuralynx",
    "MicromedRawIO": "Micromed",
    "PlexonRawIO": "Plexon",
    "Plexon2RawIO": "Plexon2",
    "SpikeGadgetsRawIO": "SpikeGadgets",
    "SpikeGLXRawIO": "SpikeGLX",
    "MedRawIO": "MED",
    "WinEdrRawIO": "WinEDR",
    "WinWcpRawIO": "WinWCP",
    "NeuroNexusRawIO": "NeuroNexus",
    "TdtRawIO": "TDT",
}


_PROBE_COLOR_SELECTED = QColor(0x00, 0xBB, 0xFF)
_PROBE_COLOR_UNSELECTED = QColor(140, 140, 140)
_PROBE_DOT_SIZE = 12
_PROBE_LABEL_FONT = pg.Qt.QtGui.QFont("monospace", 7)


class _RectangleSelector:
    """Rubber-band rectangle drawn on a pyqtgraph ViewBox."""

    def __init__(self, view_box: pg.ViewBox, on_release):
        self._vb = view_box
        self._on_release = on_release
        self._rect_item: pg.QtWidgets.QGraphicsRectItem | None = None
        self._origin = None
        view_box.scene().sigMouseClicked.connect(self._noop)
        view_box.mouseDragEvent = self._drag_event

    @staticmethod
    def _noop(evt):
        pass

    def _drag_event(self, evt):
        evt.accept()
        pos = evt.pos()
        if evt.isStart():
            self._origin = self._vb.mapToView(pos)
            if self._rect_item is not None:
                self._vb.removeItem(self._rect_item)
            self._rect_item = pg.QtWidgets.QGraphicsRectItem()
            self._rect_item.setPen(QPen(QColor(255, 255, 100, 200), 1))
            self._rect_item.setBrush(QBrush(QColor(255, 255, 100, 40)))
            self._vb.addItem(self._rect_item, ignoreBounds=True)
        elif evt.isFinish():
            end = self._vb.mapToView(pos)
            rect = QRectF(self._origin, end).normalized()
            if self._rect_item is not None:
                self._vb.removeItem(self._rect_item)
                self._rect_item = None
            self._on_release(rect)
        else:
            if self._origin is not None:
                current = self._vb.mapToView(pos)
                r = QRectF(self._origin, current).normalized()
                self._rect_item.setRect(r)


class ProbeChannelDialog(QDialog):
    """Interactive probe map for selecting channels by spatial position."""

    def __init__(
        self,
        channel_positions: np.ndarray,
        channel_map: np.ndarray | None,
        hw_names: dict[int, str] | None,
        selected_hw: set[int] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select channels on probe")
        self.resize(800, 600)

        self._positions = channel_positions
        self._channel_map = channel_map if channel_map is not None else np.arange(len(channel_positions))
        self._hw_names = hw_names or {}
        self._n_sites = len(self._channel_map)
        self._selected: set[int] = set(selected_hw) if selected_hw else set()

        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Channel", "Selected"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self._populate_table()
        left_layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(self._select_all)
        deselect_all_btn = QPushButton("Deselect all")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(deselect_all_btn)
        left_layout.addLayout(btn_row)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Drag a rectangle to select channels:"))
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", "X position")
        self._plot_widget.setLabel("left", "Y position (depth)")
        right_layout.addWidget(self._plot_widget)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._scatter = None
        self._text_items: list[pg.TextItem] = []
        self._draw_probe()
        vb = self._plot_widget.getPlotItem().getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        self._selector = _RectangleSelector(vb, self._on_rect_select)

    def _channel_label(self, idx: int) -> str:
        hw_ch = int(self._channel_map[idx])
        name = self._hw_names.get(hw_ch)
        if name and name != f"Ch {hw_ch}":
            return f"{name} ({hw_ch})"
        return f"Ch {hw_ch}"

    def _populate_table(self):
        self._table.setRowCount(self._n_sites)
        for i in range(self._n_sites):
            label_item = QTableWidgetItem(self._channel_label(i))
            self._table.setItem(i, 0, label_item)
            status_item = QTableWidgetItem()
            self._table.setItem(i, 1, status_item)
        self._update_table_colors()

    def _update_table_colors(self):
        for i in range(self._n_sites):
            is_sel = i in self._selected
            color = _PROBE_COLOR_SELECTED if is_sel else _PROBE_COLOR_UNSELECTED
            for col in range(2):
                item = self._table.item(i, col)
                if item:
                    item.setBackground(QBrush(color))
            status = self._table.item(i, 1)
            if status:
                status.setText("Yes" if is_sel else "")

    def _draw_probe(self):
        x = self._positions[:, 0] if len(self._positions) > 0 else np.array([])
        y = self._positions[:, 1] if len(self._positions) > 0 else np.array([])

        brushes = []
        for i in range(self._n_sites):
            if i in self._selected:
                brushes.append(pg.mkBrush(_PROBE_COLOR_SELECTED))
            else:
                brushes.append(pg.mkBrush(_PROBE_COLOR_UNSELECTED))

        if self._scatter is not None:
            self._plot_widget.removeItem(self._scatter)

        self._scatter = pg.ScatterPlotItem(
            x=x,
            y=y,
            size=_PROBE_DOT_SIZE,
            pen=pg.mkPen(color="w", width=0.5),
            brush=brushes,
            hoverable=True,
            hoverSize=_PROBE_DOT_SIZE + 4,
            tip=lambda x, y, data: "",
        )
        self._scatter.sigClicked.connect(self._on_dot_clicked)
        self._plot_widget.addItem(self._scatter)

        for item in self._text_items:
            self._plot_widget.removeItem(item)
        self._text_items.clear()

        for i in range(self._n_sites):
            hw_ch = int(self._channel_map[i])
            ti = pg.TextItem(
                text=str(hw_ch),
                color="w",
                anchor=(0.5, 0.5),
                fill=pg.mkBrush(0, 0, 0, 160),
            )
            ti.setFont(_PROBE_LABEL_FONT)
            ti.setPos(float(x[i]), float(y[i]))
            self._plot_widget.addItem(ti)
            self._text_items.append(ti)

        self._plot_widget.autoRange()

    def _on_dot_clicked(self, _scatter, points, _ev):
        for pt in points:
            idx = pt.index()
            if idx in self._selected:
                self._selected.discard(idx)
            else:
                self._selected.add(idx)
        self._refresh()

    def _on_rect_select(self, rect: QRectF):
        for i in range(self._n_sites):
            px = self._positions[i, 0]
            py = self._positions[i, 1]
            if rect.contains(px, py):
                self._selected.add(i)
        self._refresh()

    def _refresh(self):
        self._update_table_colors()
        brushes = []
        for i in range(self._n_sites):
            if i in self._selected:
                brushes.append(pg.mkBrush(_PROBE_COLOR_SELECTED))
            else:
                brushes.append(pg.mkBrush(_PROBE_COLOR_UNSELECTED))
        if self._scatter is not None:
            self._scatter.setBrush(brushes)

    def _select_all(self):
        self._selected = set(range(self._n_sites))
        self._refresh()

    def _deselect_all(self):
        self._selected.clear()
        self._refresh()

    def get_selected_hw_channels(self) -> np.ndarray | None:
        if not self._selected:
            return None
        y_coords = self._positions[:, 1]
        selected_list = sorted(self._selected, key=lambda i: -y_coords[i])
        return self._channel_map[selected_list].astype(int)


class ParamsDialog(QDialog):
    """Dialog to collect params.py fields when the file is missing or dat_path is invalid."""

    def __init__(self, parent=None, defaults: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Ephys params.py configuration")
        self.resize(500, 260)
        defaults = defaults or {}

        layout = QVBoxLayout(self)

        form = QVBoxLayout()

        # dat_path
        dat_row = QHBoxLayout()
        dat_row.addWidget(QLabel("dat_path:"))
        self.dat_path_edit = QLineEdit(defaults.get("dat_path", ""))
        dat_row.addWidget(self.dat_path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dat)
        dat_row.addWidget(browse_btn)
        form.addLayout(dat_row)

        # n_channels_dat
        nch_row = QHBoxLayout()
        nch_row.addWidget(QLabel("n_channels_dat:"))
        self.n_channels_spin = QSpinBox()
        self.n_channels_spin.setRange(1, 10000)
        self.n_channels_spin.setValue(defaults.get("n_channels_dat", 64))
        nch_row.addWidget(self.n_channels_spin)
        nch_row.addStretch()
        form.addLayout(nch_row)

        # sample_rate
        sr_row = QHBoxLayout()
        sr_row.addWidget(QLabel("sample_rate:"))
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(1.0, 1_000_000.0)
        self.sample_rate_spin.setDecimals(1)
        self.sample_rate_spin.setValue(defaults.get("sample_rate", 30000.0))
        sr_row.addWidget(self.sample_rate_spin)
        sr_row.addStretch()
        form.addLayout(sr_row)

        # offset
        off_row = QHBoxLayout()
        off_row.addWidget(QLabel("offset:"))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(0, 1_000_000)
        self.offset_spin.setValue(defaults.get("offset", 0))
        off_row.addWidget(self.offset_spin)
        off_row.addStretch()
        form.addLayout(off_row)

        # dtype
        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("dtype:"))
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["int16", "float32", "float64", "int32", "uint16"])
        self.dtype_combo.setCurrentText(defaults.get("dtype", "int16"))
        dt_row.addWidget(self.dtype_combo)
        dt_row.addStretch()
        form.addLayout(dt_row)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _browse_dat(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select raw ephys data file",
            "",
            "Raw data (*.dat *.bin *.raw);;All files (*)",
        )
        if path:
            self.dat_path_edit.setText(path)

    def get_params(self) -> dict:
        return {
            "dat_path": self.dat_path_edit.text().strip(),
            "n_channels_dat": self.n_channels_spin.value(),
            "sample_rate": self.sample_rate_spin.value(),
            "offset": self.offset_spin.value(),
            "dtype": self.dtype_combo.currentText(),
        }


def _write_params_py(folder: Path, params: dict):
    """Write a params.py file to the given folder."""
    lines = [
        f"dat_path = r'{params.get('dat_path', '')}'",
        f"n_channels_dat = {params.get('n_channels_dat', 64)}",
        f"dtype = '{params.get('dtype', 'int16')}'",
        f"offset = {params.get('offset', 0)}",
        f"sample_rate = {params.get('sample_rate', 30000.0)}",
        "hp_filtered = False",
    ]
    (folder / "params.py").write_text("\n".join(lines) + "\n")


_COLOR_ROLE = Qt.UserRole + 2

<<<<<<< HEAD
#: Raster dots of a unit that passes the neuron table's filters but is not selected.
=======
#: Raster dots of a unit that passes the cluster table's filters but is not selected.
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
_UNSELECTED_DOT_COLOR = (0, 0, 0)


class _ChannelFilterProxy(MultiColumnFilterProxy):
    """Column filtering plus the probe view's "only these channels" restriction.

    The channel restriction is not a header filter — it is driven by the probe
    map selection — so it lives here rather than in the shared proxy.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._visible_channels: set[int] | None = None
        self._ch_col: int | None = None

    def set_visible_channel_filter(self, channels: set[int] | None, ch_col: int | None):
        self._visible_channels = channels
        self._ch_col = ch_col
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent):
        if not super().filterAcceptsRow(source_row, source_parent):
            return False
        if self._visible_channels is None or self._ch_col is None:
            return True
        try:
            channel = int(float(self.sourceModel().index(source_row, self._ch_col, source_parent).data(SORT_ROLE)))
        except (ValueError, TypeError):
            return False
        return channel in self._visible_channels


class _ClusterIdDelegate(QStyledItemDelegate):
    """Paints cluster_id cells with a solid background color stored in _COLOR_ROLE.

    This bypasses the QSS `QTableView::item` rules that would otherwise ignore
    the model's BackgroundRole.
    """

    def paint(self, painter, option, index):
        color: QColor | None = index.data(_COLOR_ROLE)
        if color is not None:
            painter.save()
            painter.fillRect(option.rect, QBrush(color))
            if option.state & QStyle.StateFlag.State_Selected:
                painter.fillRect(option.rect, QBrush(QColor(255, 255, 255, 50)))
            r, g, b = color.red(), color.green(), color.blue()
            text_color = QColor(0, 0, 0) if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else QColor(255, 255, 255)
            painter.setPen(text_color)
            text = str(index.data(Qt.DisplayRole) or "")
            painter.drawText(option.rect.adjusted(4, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
            painter.restore()
        else:
            super().paint(painter, option, index)


class FiringRateReader:
    """Firing rates over any window behind ``select()`` — the part of a loader a trial sweep reads.

    Either of :data:`SPIKE_RATE_FEATURES`: the binned, smoothed rate of the
    Firing rates panel, or the instantaneous one (``1 / ISI``), read at that
    panel's bin spacing so both land on the same time grid. The dim is
    :data:`FIRING_RATE_UNIT_DIM`: pinned to one unit it returns that unit's
    rate, left free every unit's, one column each.
    """

    def __init__(self, widget: EphysWidget, unit_ids: list[int]) -> None:
        self._widget = widget
        self._unit_ids = unit_ids
        self._trains: dict[int, np.ndarray] = {}

    def _train(self, unit_id: int) -> np.ndarray:
        """One unit's spike times over the whole session, read once."""
        if unit_id not in self._trains:
            self._trains[unit_id] = np.asarray(self._widget._tsgroup[unit_id].times(), dtype=np.float64)
        return self._trains[unit_id]

    def _instantaneous(self, unit_ids: list[int], t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """``1 / ISI`` of *unit_ids* on the panel's bin grid over ``[t0, t1]``.

        The interval a grid time falls in may start before the window, so
        each unit's whole train is consulted, not the window's spikes.
        """
        bin_size = self._widget.fr_bin_spin.value()
        centers = t0 + (np.arange(int(np.floor((t1 - t0) / bin_size + 1e-9))) + 0.5) * bin_size
        rates = np.column_stack([instantaneous_rate(self._train(cid), centers) for cid in unit_ids])
        return rates.astype(np.float32), centers

    def select(self, feature: str, selections: dict[str, str], t0: float, t1: float) -> PlotData | None:
        pinned = selections.get(FIRING_RATE_UNIT_DIM)
        unit_ids = self._unit_ids if pinned is None else [cid for cid in self._unit_ids if str(cid) == pinned]
        if feature not in SPIKE_RATE_FEATURES or not unit_ids:
            return None
        if feature == INSTANT_RATE_FEATURE:
            rates, centers = self._instantaneous(unit_ids, t0, t1)
        else:
            rates, centers = self._widget._bin_rates(unit_ids, t0, t1)
        if pinned is not None:
            return PlotData(time=centers, data=rates[:, 0])
        return PlotData(time=centers, data=rates, dim_labels=[str(cid) for cid in unit_ids])


class EphysWidget(QWidget):
    """Ephys controls with toggle-button tabs: Ephys trace | Neuron jumping."""

    cluster_selected = Signal(int)  # emitted when a single cluster row is selected
<<<<<<< HEAD
    unit_filter_changed = Signal()  # the neuron table's filters changed which units pass
    unit_rows_changed = Signal()  # which units have a row, or the order of the rows, changed
=======
    unit_filter_changed = Signal()  # the cluster table's filters changed which units pass
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4

    def __init__(self, shell, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.shell = shell
        self.plot_container = None
        self.meta_widget = None
        self.data_widget = None
        self.io_widget = None

        self._cluster_df: pd.DataFrame | None = None
        self._spike_clusters: np.ndarray | None = None
        self._spike_samples: np.ndarray | None = None  # raw integer Kilosort sample indices
        self._spike_times_s: np.ndarray | None = None  # float64 seconds, derived on load
        self._channel_positions: np.ndarray | None = None
        self._channel_map: np.ndarray | None = None
        self._probe_channel_order: np.ndarray | None = None
        self._custom_channel_set: np.ndarray | None = None
        self._templates: np.ndarray | None = None
        self._ephys_n_channels = 0
        self._tsgroup = None
        self._neurons_source: str | None = None  # "kilosort" or "pynapple"
        self._pynapple_cid_to_row: dict[int, int] = {}  # cluster_id → raster row index
        # Every spike in one time-sorted table, built once per loaded TsGroup.
        self._spike_table_cache: tuple[nap.TsGroup, SpikeTable] | None = None
        # Unit ids in the order the last Rastermap fit gave them.
        self._rastermap_units: list[int] | None = None
        self._current_cluster_id_for_psth: int | None = None
        self._psth_dialog = None
        self._kilosort_sr: float | None = None
        self._fr_cache_key: tuple | None = None
        self._kilosort_params: dict | None = None
        self._phy_reader = None
        self._phy_loader = None
        self._phy_n_channels: int | None = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_traceview_panel(main_layout)
        self._create_firing_rate_panel(main_layout)

        # traceview_panel is borrowed into the right sidebar's "Phy viewer"
        # context; firing_rate_panel is popped from the top-bar Neural menu.
        self.traceview_panel.show()
        self.firing_rate_panel.show()
        self.setEnabled(False)

    def _open_psth(self):
        from .widgets_psth import PSTHDialog

        if self._psth_dialog is None or not self._psth_dialog.isVisible():
            nav = getattr(self.data_widget, "navigation_widget", None)
            labels_w = getattr(self.data_widget, "labels_widget", None) if self.data_widget else None
            self._psth_dialog = PSTHDialog(self.app_state, self, labels_w, nav, parent=self)
            self._psth_dialog.trial_jump_requested.connect(self._on_psth_trial_jump)

        self._psth_dialog.show()
        self._psth_dialog.raise_()
        self._psth_dialog.activateWindow()

    def _on_psth_trial_jump(self, trial_id: str):
        nav = getattr(self.data_widget, "navigation_widget", None)
        if nav is not None:
            nav.navigate_to_trial(trial_id)

    def _refresh_layout(self):
        if self.meta_widget:
            self.meta_widget.refresh_widget_layout(self)

    # ------------------------------------------------------------------
    # Ephys trace panel (channel,  gain, range)
    # ------------------------------------------------------------------

    def _create_traceview_panel(self, main_layout):
        self.traceview_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.traceview_panel.setLayout(layout)

        group = QGroupBox("Ephys trace controls")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)
        group_layout.setContentsMargins(2, 2, 2, 2)
        group.setLayout(group_layout)
        layout.addWidget(group)

        # Channel spinbox
        self.ephys_channel_label = QLabel("Ephys channel:")
        self.ephys_channel_spin = QSpinBox()
        self.ephys_channel_spin.setObjectName("ephys_channel_spin")
        self.ephys_channel_spin.setRange(0, 0)
        self.ephys_channel_spin.setPrefix("Ch ")
        self.ephys_channel_spin.setToolTip("Select ephys channel to display")
        self.ephys_channel_spin.valueChanged.connect(self._on_ephys_channel_changed)

        # Gain spinbox
        self.ephys_gain_label = QLabel("Gain:")
        self.ephys_gain_spin = QDoubleSpinBox()
        self.ephys_gain_spin.setObjectName("ephys_gain_spin")
        self.ephys_gain_spin.setRange(-100.0, 100.0)
        self.ephys_gain_spin.setSingleStep(0.1)
        self.ephys_gain_spin.setDecimals(1)
        self.ephys_gain_spin.setValue(0.0)
        self.ephys_gain_spin.setToolTip("Display gain: negative = amplify, positive = attenuate (Ctrl+Wheel)")
        self.ephys_gain_spin.valueChanged.connect(self._on_ephys_gain_changed)

        self.ephys_auto_gain_cb = QCheckBox("Auto gain")
        self.ephys_auto_gain_cb.setToolTip(
            "Quantile-based auto-scaling (Phy method: 1st/99th percentile after median subtraction)"
        )
        self.ephys_auto_gain_cb.setChecked(True)
        self.ephys_auto_gain_cb.toggled.connect(self._on_auto_gain_toggled)

        ch_row = QHBoxLayout()
        ch_row.addWidget(self.ephys_gain_label)
        ch_row.addWidget(self.ephys_gain_spin)
        ch_row.addWidget(self.ephys_auto_gain_cb)
        ch_row.addStretch()
        group_layout.addLayout(ch_row)

        self.pyramid_cb = QCheckBox("Pyramid downsampling")
        self.pyramid_cb.setChecked(True)
        self.pyramid_cb.setToolTip(
            "Use precomputed min/max pyramid for efficient rendering of long recordings. "
            "Disable to use direct strided loading (slower but useful for diagnosing rendering issues)."
        )
        self.pyramid_cb.toggled.connect(self._on_pyramid_toggled)
        group_layout.addWidget(self.pyramid_cb)

        self._probe_row = QWidget()
        probe_row_layout = QHBoxLayout()
        probe_row_layout.setContentsMargins(0, 0, 0, 0)
        probe_row_layout.setSpacing(5)
        self._probe_row.setLayout(probe_row_layout)

        self.probe_select_btn = QPushButton("Select channels on probe")
        self.probe_select_btn.setToolTip("Open probe map to select channels by spatial position")
        self.probe_select_btn.clicked.connect(self._open_probe_channel_dialog)
        probe_row_layout.addWidget(self.probe_select_btn)

        probe_row_layout.addWidget(QLabel("N closest:"))
        self.n_closest_spin = QSpinBox()
        self.n_closest_spin.setRange(1, 384)
        self.n_closest_spin.setValue(5)
        self.n_closest_spin.setToolTip(
            "Number of spatially closest channels for waveform display. Only works for non split/merged clusters."
        )
        self.n_closest_spin.valueChanged.connect(self._update_highlight_label)
        probe_row_layout.addWidget(self.n_closest_spin)
        probe_row_layout.addStretch()

        self._probe_row.hide()
        group_layout.addWidget(self._probe_row)

        _sel_row = QHBoxLayout()
        _sel_row.setSpacing(4)
        _sel_row.setContentsMargins(0, 0, 0, 0)

        self._highlight_label = QLabel()
        self._update_highlight_label()
        _sel_row.addWidget(self._highlight_label)

        self._select_visible_btn = QPushButton("All visible rows")
        self._select_visible_btn.setToolTip(
            "Highlight all rows currently visible after filtering, then disable auto-highlight"
        )
        self._select_visible_btn.clicked.connect(self._select_clusters_all_visible)
        _sel_row.addWidget(self._select_visible_btn)

        self._unselect_btn = QPushButton("Unselect")
        self._unselect_btn.setToolTip("Clear spike overlays and table selection")
        self._unselect_btn.clicked.connect(self._unselect_clusters)
        _sel_row.addWidget(self._unselect_btn)

        self._selection_row = _sel_row

        self._multi_cluster_colors: dict[int, tuple] = {}
        # Unit ids by when their rate peaks, as last sorted (the PEAK_ORDERS row orders).
        self._peak_units: list[int] | None = None
        # The units drawn on each raster row, by row key (see _raster_rows).
        self._raster_row_units: dict[int, list[int]] = {}
        # True while a click on a plot row changes the table's selection: the view stays where it is.
        self._selecting_from_plot = False

        self._cluster_model = QStandardItemModel()
        self._cluster_proxy = _ChannelFilterProxy()
        self._cluster_proxy.setSourceModel(self._cluster_model)

        self.cluster_table = QTableView()
        self.cluster_table.setModel(self._cluster_proxy)
        self.cluster_table.verticalHeader().setVisible(False)
        self.cluster_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cluster_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cluster_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cluster_table.setSortingEnabled(True)
        self.cluster_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cluster_table.verticalHeader().setDefaultSectionSize(CLUSTER_TABLE_ROW_HEIGHT)
        self.cluster_table.setMaximumHeight(CLUSTER_TABLE_MAX_HEIGHT)

        self._filter_col_cats: dict[int, list[str]] = {}
        self._filter_cat_active: dict[int, set[str]] = {}
        self._filter_num_active: dict[int, tuple[str, float]] = {}
        self._filter_cat_cols: set[int] = set()
        self._filter_num_cols: set[int] = set()

        self._cluster_id_delegate = _ClusterIdDelegate(self.cluster_table)

        self._filter_header = FilterHeaderView(set(), set())
        self._filter_header.setDefaultSectionSize(40)
        self._filter_header.setMinimumSectionSize(20)
        self._filter_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self._filter_header.setStretchLastSection(False)
        self._filter_header.filter_requested.connect(self._on_filter_header_clicked)
        self.cluster_table.setHorizontalHeader(self._filter_header)
        self._filter_header.setSortIndicator(-1, Qt.AscendingOrder)
        self.cluster_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.cluster_table.setStyleSheet("""
            QTableView { gridline-color: transparent; background: #444; color: #fff; }
            QTableView::item { padding: 0px 2px; color: #fff; }
            QTableView::item:selected { background: #3a5070; color: #fff; }
            QHeaderView::section {
                padding: 2px 4px;
                background: #888;
                color: #fff;
                border: none;
                border-right: 1px solid #666;
                font-family: sans-serif;
                font-size: 11px;
            }
            QHeaderView::section:last { border-right: none; }
        """)
        self.cluster_table.selectionModel().selectionChanged.connect(self._on_cluster_row_selected)
        # layoutChanged, not the header's sortIndicatorChanged: that one fires before the rows have moved.
        self._cluster_proxy.layoutChanged.connect(self._on_cluster_table_sorted)
<<<<<<< HEAD
=======

        raster_row = QHBoxLayout()
        raster_row.setSpacing(4)
        raster_row.setContentsMargins(0, 0, 0, 0)
        raster_row.addWidget(QLabel("Raster:"))
        self.raster_render_combo = QComboBox()
        for key, label in RENDER_MODES.items():
            self.raster_render_combo.addItem(label, key)
        self.raster_render_combo.setToolTip(
            "Auto: a tick per spike, and spike counts per pixel once the view is too crowded for ticks.\n"
            "Ticks / Density: always that one."
        )
        self.raster_render_combo.setCurrentIndex(
            max(0, self.raster_render_combo.findData(self.app_state.get_with_default("raster_render_mode")))
        )
        self.raster_render_combo.currentIndexChanged.connect(self._on_raster_render_changed)
        raster_row.addWidget(self.raster_render_combo)

        raster_row.addWidget(QLabel("Tick width:"))
        self.raster_tick_width_spin = QSpinBox()
        self.raster_tick_width_spin.setRange(TICK_WIDTH_AUTO, MAX_TICK_WIDTH)
        self.raster_tick_width_spin.setSpecialValueText("Auto")
        self.raster_tick_width_spin.setSuffix(" px")
        self.raster_tick_width_spin.setToolTip(
            "Auto: as wide as the view allows, bold when spikes are sparse and thin when they are close together.\n"
            "Any other value is used as it is."
        )
        self.raster_tick_width_spin.setValue(self.app_state.get_with_default("raster_tick_width"))
        self.raster_tick_width_spin.valueChanged.connect(self._on_raster_tick_width_changed)
        raster_row.addWidget(self.raster_tick_width_spin)

        raster_row.addWidget(QLabel("Rows:"))
        self.raster_row_order_combo = QComboBox()
        for key, label in ROW_ORDERS.items():
            self.raster_row_order_combo.addItem(label, key)
        self.raster_row_order_combo.setToolTip(
            "Probe depth: a row per channel, aligned with the trace panel (Kilosort only).\n"
            "Cluster table: a row per unit, in the table's order — sort the table to reorder.\n"
            "Rastermap: a row per unit, units with similar activity next to each other."
        )
        self.raster_row_order_combo.currentIndexChanged.connect(self._on_raster_row_order_changed)
        raster_row.addWidget(self.raster_row_order_combo)

        self.rastermap_fit_btn = QPushButton("Refit")
        self.rastermap_fit_btn.setToolTip("Fit Rastermap again on the units and trial shown now")
        self.rastermap_fit_btn.clicked.connect(self._fit_rastermap)
        raster_row.addWidget(self.rastermap_fit_btn)
        raster_row.addStretch()
        layout.addLayout(raster_row)
        self._sync_row_order_combo()
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4

        # Raster-only settings: how its spikes are drawn.
        self.raster_panel = QGroupBox("Raster display")
        raster_row = QHBoxLayout()
        raster_row.setSpacing(4)
        raster_row.setContentsMargins(2, 2, 2, 2)
        self.raster_panel.setLayout(raster_row)
        raster_row.addWidget(QLabel("Raster:"))
        self.raster_render_combo = QComboBox()
        for key, label in RENDER_MODES.items():
            self.raster_render_combo.addItem(label, key)
        self.raster_render_combo.setToolTip(
            "Auto: a tick per spike, and spike counts per pixel once the view is too crowded for ticks.\n"
            "Ticks / Density: always that one."
        )
        self.raster_render_combo.setCurrentIndex(
            max(0, self.raster_render_combo.findData(self.app_state.get_with_default("raster_render_mode")))
        )
        self.raster_render_combo.currentIndexChanged.connect(self._on_raster_render_changed)
        raster_row.addWidget(self.raster_render_combo)

        raster_row.addWidget(QLabel("Tick width:"))
        self.raster_tick_width_spin = QSpinBox()
        self.raster_tick_width_spin.setRange(TICK_WIDTH_AUTO, MAX_TICK_WIDTH)
        self.raster_tick_width_spin.setSpecialValueText("Auto")
        self.raster_tick_width_spin.setSuffix(" px")
        self.raster_tick_width_spin.setToolTip(
            "Auto: as wide as the view allows, bold when spikes are sparse and thin when they are close together.\n"
            "Any other value is used as it is."
        )
        self.raster_tick_width_spin.setValue(self.app_state.get_with_default("raster_tick_width"))
        self.raster_tick_width_spin.valueChanged.connect(self._on_raster_tick_width_changed)
        raster_row.addWidget(self.raster_tick_width_spin)
        raster_row.addStretch()

        # What every neuron panel's sidebar shows: the one row order they all
        # follow, and the table that filters, sorts and selects the units.
        self.neuron_table_panel = QWidget()
        neuron_layout = QVBoxLayout()
        neuron_layout.setSpacing(2)
        neuron_layout.setContentsMargins(0, 0, 0, 0)
        self.neuron_table_panel.setLayout(neuron_layout)
        neuron_title = QLabel("Neuron table settings")
        neuron_title.setStyleSheet(CONTEXT_TITLE_STYLE)
        neuron_layout.addWidget(neuron_title)

        order_row = QHBoxLayout()
        order_row.setSpacing(4)
        order_row.setContentsMargins(0, 0, 0, 0)
        order_row.addWidget(QLabel("Row order:"))
        self.raster_row_order_combo = QComboBox()
        for key, label in ROW_ORDERS.items():
            self.raster_row_order_combo.addItem(label, key)
        self.raster_row_order_combo.setToolTip(
            "The order of the units in every panel that lists them (raster, firing rates, per-unit labels).\n"
            "Probe depth: a row per channel, aligned with the trace panel (Kilosort only).\n"
            "Neuron table: the table's order — sort the table to reorder.\n"
            "Rastermap: units with similar activity next to each other; fitted once, then kept.\n"
            "Peak time (each trial): by when each unit's rate peaks, earliest on top; re-sorted on every trial.\n"
            "Peak time (visible window): the same over what is on screen, sorted when the button is pressed."
        )
        self.raster_row_order_combo.currentIndexChanged.connect(self._on_raster_row_order_changed)
        order_row.addWidget(self.raster_row_order_combo)

        self.order_window_spin = QDoubleSpinBox()
        self.order_window_spin.setRange(0.01, 600.0)
        self.order_window_spin.setDecimals(2)
        self.order_window_spin.setSuffix(" s")
        self.order_window_spin.setToolTip("Length of the window a unit's peak is looked for in")
        self.order_window_spin.setValue(self.app_state.get_with_default("heatmap_sort_window_s"))
        self.order_window_spin.valueChanged.connect(self._on_order_window_changed)
        order_row.addWidget(self.order_window_spin)

        self.order_apply_btn = QPushButton("Refit")
        self.order_apply_btn.setToolTip("Fit or sort again on the units and the trial or window shown now")
        self.order_apply_btn.clicked.connect(self._on_order_apply_clicked)
        order_row.addWidget(self.order_apply_btn)
        order_row.addStretch()
        neuron_layout.addLayout(order_row)
        neuron_layout.addLayout(self._selection_row)
        neuron_layout.addWidget(self.cluster_table)
        self.unit_events = UnitEventsArea(self.app_state, self, FIRING_RATE_UNIT_DIM)
        neuron_layout.addWidget(self.unit_events)
        self._sync_row_order_combo()

        main_layout.addWidget(self.raster_panel)
        main_layout.addWidget(self.neuron_table_panel)

        main_layout.addWidget(self.traceview_panel)

    # ------------------------------------------------------------------
    # Ephys trace handlers
    # ------------------------------------------------------------------

    def _resolve_phy_loader(self) -> tuple:
        """Resolve the Phy-Viewer loader from the current Neo source (no Kilosort).

        Called once at Kilosort load time to populate _phy_loader, and on every
        trial change when no Kilosort is loaded.
        """
        if self._phy_reader is not None:
            return self._phy_reader, 0

        ephys_path, stream_id, channel_idx = self.app_state.get_ephys_source()
        if not ephys_path:
            return None, 0

        if self._kilosort_params and Path(ephys_path).suffix.lower() in EPHYS_EXTENSIONS_RAW:
            n_ch = self._kilosort_params.get("n_channels_dat")
            if n_ch is None and self._channel_map is not None:
                n_ch = int(self._channel_map.max()) + 1
            if n_ch is None:
                return None, 0
            try:
                return load_ephys(
                    ephys_path,
                    stream_id,
                    n_channels=n_ch,
                    sampling_rate=self._kilosort_params.get("sample_rate"),
                ), channel_idx
            except Exception:
                return None, 0

        try:
            stream_id = self._match_neural_stream(ephys_path, stream_id)
            return load_ephys(ephys_path, stream_id), channel_idx
        except Exception:
            return None, 0

    def _match_neural_stream(self, ephys_path: str, stream_id: str) -> str:
        # NOTE: Claude-authored — unsure whether to keep. Revisit.
        """Pick the Neo stream that matches the probe when Kilosort is loaded.

        The Neo stream combo may point at an auxiliary stream (e.g. a 2-channel
        Intan digital-input stream). Pairing that with a full-probe channel
        order breaks the ephys trace rendering, so resolve to the stream whose
        channel count / sample rate matches the Kilosort params instead.
        """
        if not self._kilosort_params:
            return stream_id
        ks_nch = self._kilosort_params.get("n_channels_dat")
        ks_sr = self._kilosort_params.get("sample_rate")
        try:
            info = load_ephys(ephys_path, str(stream_id)).stream_info
        except Exception:
            return stream_id
        if not info:
            return stream_id
        if ks_nch is not None:
            matches = [sid for sid, meta in info.items() if meta.get("n_channels") == ks_nch]
            if matches:
                if ks_sr is not None:
                    for sid in matches:
                        if abs(info[sid].get("rate", 0.0) - ks_sr) < 1.0:
                            return sid
                return matches[0]
        return max(info.items(), key=lambda kv: kv[1].get("n_channels", 0))[0]

    def has_phy_trace(self) -> bool:
        """True when a raw-data Phy trace can be shown (Kilosort raw .bin/.dat
        loader resolved). Gates the "Ephys (Phy-like viewer)" popup source."""
        if self._phy_loader is not None:
            return True
        loader, _ = self._resolve_phy_loader()
        return loader is not None

    def configure_ephys_trace_plot(self):
        # When Kilosort is loaded, _phy_loader is captured once at load time
        # so switching the Neo stream combo cannot affect the Phy-Viewer rate.
        if self._phy_loader is not None:
            loader, channel_idx = self._phy_loader, 0
        else:
            loader, channel_idx = self._resolve_phy_loader()

        if loader is None:
            return

        self.plot_container.ephys_trace_plot.set_loader(loader, channel_idx)
        start_time = getattr(loader, "starting_time", 0.0)
        ephys_source = FileSource("ephys", loader, start_time=start_time)
        self.plot_container.ephys_trace_plot.set_source(ephys_source)
        self.plot_container.raster_plot.set_source(ephys_source)

        n_ch = loader.n_channels
        self._ephys_n_channels = n_ch
        self.ephys_gain_label.show()
        self.ephys_gain_spin.show()

        if self.plot_container.is_ephystrace():
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

        if self._tsgroup is not None:
            self._draw_raster()

    def _on_ephys_channel_changed(self, channel: int):
        stream_sel = getattr(self.app_state, "ephys_stream_sel", None)
        source_map = getattr(self.app_state, "ephys_source_map", {})
        if stream_sel not in source_map:
            return
        filename, stream_id, _ = source_map[stream_sel]
        source_map[stream_sel] = (filename, stream_id, channel)

        if self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.set_channel(channel)
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

    def n_units(self) -> int:
        """How many spike-sorted units are loaded (0 gates the raster source off)."""
        return 0 if self._tsgroup is None else len(self._tsgroup)

    def show_raster(self) -> None:
        """Show the raster panel, filled for the current window."""
        self.plot_container.set_raster_visible(True)
        self.apply_probe_order()
        self.refresh_raster()

    def show_firing_rates(self) -> str | None:
        """Make the firing rates a feature for the current trial; its name, or ``None`` if there is nothing to bin."""
        return FIRING_RATE_FEATURE if self._compute_firing_rates() else None

    def refresh_raster(self) -> None:
        """Redraw the raster for the current window: the selected clusters, else every unit."""
        if self._tsgroup is None:
            return
        if self._multi_cluster_colors:
            self._redraw_selected_clusters()
        else:
            self._draw_raster()

    def _on_ephys_gain_changed(self, value: float):
        if self.ephys_auto_gain_cb.isChecked():
            self.ephys_auto_gain_cb.blockSignals(True)
            self.ephys_auto_gain_cb.setChecked(False)
            self.ephys_auto_gain_cb.blockSignals(False)
        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.buffer.display_gain = value
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

    def _on_auto_gain_toggled(self, checked: bool):
        if checked:
            self._apply_auto_gain()

    def _on_pyramid_toggled(self, checked: bool):
        if self.plot_container:
            plots = [getattr(self.plot_container, "ephys_trace_plot", None)]
            plots += list(getattr(self.plot_container, "neo_trace_plots", ()) or ())
            for plot in plots:
                if plot is not None:
                    plot.buffer.use_pyramid = checked
                    plot.buffer._invalidate_cache()
                    xmin, xmax = self.plot_container.get_current_xlim()
                    plot.update_plot_content(xmin, xmax)

    def _apply_auto_gain(self):
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return
        ephys_plot = self.plot_container.ephys_trace_plot
        new_gain = ephys_plot.auto_gain()
        self.ephys_gain_spin.blockSignals(True)
        self.ephys_gain_spin.setValue(new_gain)
        self.ephys_gain_spin.blockSignals(False)

    def _open_probe_channel_dialog(self):
        if self._channel_positions is None or self._channel_map is None:
            notify(
                "No channel positions loaded — load a Kilosort folder first (not available for Pynapple).",
                "warning",
            )
            return

        current_selected = None
        if self._custom_channel_set is not None:
            idx_set = set()
            for hw in self._custom_channel_set:
                matches = np.where(self._channel_map == hw)[0]
                if len(matches):
                    idx_set.add(int(matches[0]))
            current_selected = idx_set

        hw_names = self.get_hw_names(self._channel_map)
        dialog = ProbeChannelDialog(
            self._channel_positions,
            self._channel_map,
            hw_names,
            selected_hw=current_selected,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        hw_channels = dialog.get_selected_hw_channels()
        if hw_channels is None or len(hw_channels) == 0:
            self._custom_channel_set = None
            if self.plot_container and self.plot_container.is_ephystrace():
                self.plot_container.ephys_trace_plot.set_custom_channel_set(None)
            self._apply_probe_channel_filter()
            return

        self._custom_channel_set = hw_channels
        self._apply_probe_channel_filter()

        if self.plot_container and self.plot_container.is_ephystrace():
            ephys_plot = self.plot_container.ephys_trace_plot
            ephys_plot.set_custom_channel_set(hw_channels)
            ephys_plot.auto_channel_spacing()
            if self.ephys_auto_gain_cb.isChecked():
                self._apply_auto_gain()
            ephys_plot.autoscale()

    def hide_ephys_channel_controls(self):
        self.ephys_channel_label.hide()
        self.ephys_channel_spin.hide()
        self.ephys_gain_label.hide()
        self.ephys_gain_spin.hide()

    # ------------------------------------------------------------------
    # Neuron jumping panel (Kilosort / Pynapple)
    # ------------------------------------------------------------------

    def populate_ephys_default_path(self):
        neurons_path = self.io_widget.neurons_path_edit.text().strip()
        if neurons_path:
            self._load_neurons()

    def _load_neurons(self):
        """Dispatcher: detect whether path is a Kilosort folder or Pynapple file."""
        path_str = self.io_widget.neurons_path_edit.text().strip()
        if not path_str:
            return
        path = Path(path_str)
        if path.is_dir():
            self._load_kilosort_folder(path)
        elif path.is_file():
            self._load_pynapple_file(path)
        else:
            notify(f"Path not found: {path}", "warning")

    def _parse_kilosort_params(self, folder: Path) -> dict | None:
        params_file = folder / "params.py"
        if not params_file.exists():
            return None
        try:
            namespace = {}
            exec(params_file.read_text(), namespace)
            sr = namespace.get("sample_rate")
            if sr is None:
                return None
            result = {"sample_rate": float(sr)}
            n_ch = namespace.get("n_channels_dat")
            if n_ch is not None:
                result["n_channels_dat"] = int(n_ch)
            dat_path = namespace.get("dat_path")
            if dat_path is not None:
                result["dat_path"] = str(dat_path)
            result["dtype"] = str(namespace.get("dtype", "int16"))
            return result
        except (OSError, ValueError, KeyError, TypeError) as e:
            notify(f"Failed to parse {params_file.name}: {e}", "warning")
            return None

    def _validate_kilosort_sr(self, kilosort_sr: float) -> bool:
        loader = self._get_any_ephys_loader()
        if loader is None:
            return True
        ephys_sr = loader.rate
        if abs(kilosort_sr - ephys_sr) > 1.0:
            notify(
                f"Sample rate mismatch: Kilosort params.py says {kilosort_sr:.0f} Hz "
                f"but ephys loader reports {ephys_sr:.0f} Hz. Check your data.",
                "warning",
            )
            return False
        return True

    def _load_kilosort_folder(self, folder: Path):
        self.app_state.neurons_path = str(folder)

        required_files = [
            "spike_times.npy",
            "spike_clusters.npy",
            "channel_positions.npy",
            "channel_map.npy",
        ]
        missing = [f for f in required_files if not (folder / f).exists()]
        if missing:
            notify(
                "Kilosort folder is missing required files:\n" + "\n".join(f"  - {f}" for f in missing),
                "warning",
            )
            return

        ks_params = self._parse_kilosort_params(folder)
        if ks_params is None:
            dialog = ParamsDialog(self)
            if dialog.exec_() != QDialog.Accepted:
                return
            ks_params = dialog.get_params()
            _write_params_py(folder, ks_params)
            notify(f"Saved params.py to {folder}")

        dat_path_str = ks_params.get("dat_path", "")
        if dat_path_str and not Path(dat_path_str).is_file():
            # params.py records the sorting machine's path; look for the same
            # recording next to the Kilosort output before bothering the user.
            relocated = self._resolve_dat_path(folder, ks_params)
            if relocated is not None:
                ks_params["dat_path"] = str(relocated)
                _write_params_py(folder, ks_params)
                notify(f"Raw data relocated to {relocated}")
            else:
                notify(
                    f"dat_path not found on this machine:\n{dat_path_str}\n\n"
                    "Please update the path to the raw data file.",
                    "warning",
                )
                dialog = ParamsDialog(self, defaults=ks_params)
                if dialog.exec_() != QDialog.Accepted:
                    return
                ks_params = dialog.get_params()
                _write_params_py(folder, ks_params)
                notify(f"Updated params.py in {folder}")

        ks_sr = ks_params.get("sample_rate")
        if ks_sr is None:
            notify("No sample_rate in params — cannot load kilosort folder.", "warning")
            return
        ks_sr = float(ks_sr)
        if not self._validate_kilosort_sr(ks_sr):
            return
        self._kilosort_sr = ks_sr
        self._kilosort_params = ks_params

        cluster_info_path = folder / "cluster_info.tsv"
        self._cluster_df = (
            self._load_file(cluster_info_path, pd.read_csv, sep="\t") if cluster_info_path.exists() else None
        )

        self._spike_clusters = self._load_file(folder / "spike_clusters.npy", np.load, flatten=True)
        self._spike_samples = self._load_file(folder / "spike_times.npy", np.load, flatten=True)
        if self._spike_samples is not None and self._spike_clusters is not None:
            self._spike_times_s = self._spike_samples.astype(np.float64) / ks_sr
            self._tsgroup = build_tsgroup(self._spike_times_s, self._spike_clusters)
            if self._cluster_df is None:
                # The table is the unit filter, so it lists the units even without Phy's metadata.
<<<<<<< HEAD
                notify("No cluster_info.tsv found — the neuron table lists the units without metadata.", "warning")
=======
                notify("No cluster_info.tsv found — the cluster table lists the units without metadata.", "warning")
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
                self._cluster_df = self._build_cluster_df_from_tsgroup(self._tsgroup)
        self._channel_positions = self._load_file(folder / "channel_positions.npy", np.load)
        self._channel_map = self._load_file(folder / "channel_map.npy", np.load, flatten=True)
        self._templates = self._load_file(folder / "templates.npy", np.load)

        self._neurons_source = "kilosort"
        self._reorder_probe_by_position()

        if self._cluster_df is not None:
            self._populate_cluster_table(self._cluster_df)

        if self._channel_positions is not None and self._channel_map is not None:
            self._probe_row.show()

        self._register_dat_fallback(folder)
        # Capture the Phy loader once now — before the user can change the Neo
        # stream combo — so configure_ephys_trace_plot() always uses the neural stream.
        if self._phy_reader is not None:
            self._phy_loader = self._phy_reader
        else:
            self._phy_loader, _ = self._resolve_phy_loader()
        # Pre-wire the Phy loader/source so the panel renders instantly when the
        # user adds it from the popup — but do NOT show it (it is heavy; the user
        # opts in via "➕ Add panel" → "Ephys (Phy-like viewer)").
        if self._phy_loader is not None and self.plot_container:
            self.configure_ephys_trace_plot()

        self._rastermap_units = None
        self._sync_row_order_combo()
        if self._tsgroup is not None:
            self._draw_raster()

        self.app_state.has_neurons = True

        if self.data_widget:
            self.data_widget.on_kilosort_loaded()
        # A raw .bin/.dat Phy trace is now addable — refresh the add-panel popup.
        if self.meta_widget is not None:
            self.meta_widget.refresh_source_popup()

    def _load_pynapple_file(self, path: Path):
        """Load neuron data from a Pynapple-compatible file (.npz or .nwb)."""
        self.app_state.neurons_path = str(path)
        try:
            data = nap.load_file(str(path))
        except Exception as e:
            notify(f"Failed to load Pynapple file: {e}", "warning")
            return

        # An .npz holds one object, so a saved TsGroup comes back as itself
        # (its keys are unit ids); an NWB file is a container with a 'units' entry.
        if isinstance(data, nap.TsGroup):
            tsgroup = data
        elif isinstance(data, nap.NWBFile):
            if "units" not in data:
                notify(
                    f"No 'units' key found in {path.name}.\nAvailable keys: {list(data.keys())}",
                    "warning",
                )
                return
            tsgroup = data["units"]
            if not isinstance(tsgroup, nap.TsGroup):
                notify(f"'units' is not a TsGroup (got {type(tsgroup).__name__})", "warning")
                return
        else:
            notify(
                f"{path.name} holds a {type(data).__name__}, not a TsGroup of spike times.",
                "warning",
            )
            return

        self._tsgroup = tsgroup
        self._neurons_source = "pynapple"

        # No raw trace data for Pynapple
        self._spike_clusters = None
        self._spike_samples = None
        self._spike_times_s = None
        self._channel_positions = None
        self._channel_map = None
        self._templates = None
        self._kilosort_sr = None
        self._kilosort_params = None
        self._phy_reader = None
        self._phy_loader = None

        cluster_df = self._build_cluster_df_from_tsgroup(tsgroup)
        self._cluster_df = cluster_df
        if cluster_df is not None:
            self._populate_cluster_table(cluster_df)

        self._probe_row.hide()

        self._rastermap_units = None
        self._sync_row_order_combo()
        self._draw_raster()
        self.app_state.has_neurons = True
        if self.meta_widget is not None:
            self.meta_widget.refresh_source_popup()

        # Check for IntervalSets in the loaded data
        self._check_pynapple_intervalsets(data, path.name)

        notify(f"Loaded {len(tsgroup)} units from {path.name}")

    def _build_cluster_df_from_tsgroup(self, tsgroup: nap.TsGroup) -> pd.DataFrame:
        """Build a cluster DataFrame from TsGroup metadata columns."""
        cluster_ids = list(tsgroup.keys())
        df = pd.DataFrame({"cluster_id": cluster_ids})

        metadata = tsgroup.metadata
        for col in tsgroup.metadata_columns:
            df[col] = metadata[col].loc[cluster_ids].values

        df["n_spikes"] = [len(tsgroup[cid]) for cid in cluster_ids]
        return df

    def _check_pynapple_intervalsets(self, data, filename: str):
        """Show popup listing any IntervalSets found in a Pynapple file."""
        intervalsets = {}
        for key in data.keys():
            try:
                val = data[key]
            except Exception:
                continue
            if isinstance(val, nap.IntervalSet):
                intervalsets[key] = val

        if not intervalsets:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("IntervalSets found")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Found {len(intervalsets)} IntervalSet(s) in {filename}:"))
        for name, iset in intervalsets.items():
            layout.addWidget(QLabel(f"  {name} — {len(iset)} intervals"))
        btn = QDialogButtonBox(QDialogButtonBox.Ok)
        btn.accepted.connect(dialog.accept)
        layout.addWidget(btn)
        dialog.exec_()

    def _trial_start_session(self) -> float:
        trial = getattr(self.app_state, "trials_sel", None)
        align = getattr(self.app_state, "nwb_alignment", None)
        if trial is not None and align is not None:
            return float(align.start_time(trial) or 0.0)
        return 0.0

    def _trial_ep(self) -> nap.IntervalSet | None:
        """The visible window mapped onto the spike/recording clock.

        Uses the window's actual start AND end — a window that doesn't begin
        at display t=0 (label/sequence windows, panned fixed windows, session
        basis) must restrict to its real span, not ``[0, duration]``.
        """
        window_bounds = self.app_state.window_bounds
        if window_bounds is None:
            return None
        offset = self._ephys_offset()
        return nap.IntervalSet(offset + window_bounds.start_s, offset + window_bounds.end_s)

    def _ephys_offset(self) -> float:
        """Spike/recording-clock time corresponding to display-clock t=0.

        Spike times are session-absolute (plus the user's scalar
        ``ephys_offset``); the display axis is trial-relative in trial basis
        and session-absolute in session basis, so the trial start is added
        only in trial basis. Add to a display time to reach the spike clock;
        subtract from spike times to draw them on the axis.
        """
        ephys_offset = float(getattr(self.app_state, "ephys_offset", 0.0) or 0.0)
        if self.app_state.display_basis == "session":
            return ephys_offset
        return self._trial_start_session() + ephys_offset

    def filtered_unit_ids(self) -> list[int]:
<<<<<<< HEAD
        """Ids of the units passing the neuron table's filters — the one unit filter.
=======
        """Ids of the units passing the cluster table's filters — the one unit filter.
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4

        Every consumer of units (raster, firing rates, PSTH) reads this, the
        way every trial operation reads ``app_state.trials``. Table order.
        """
        if self._tsgroup is None:
            return []
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return []
        loaded = {int(cid) for cid in self._tsgroup.keys()}
        ids = []
        for row in range(self._cluster_proxy.rowCount()):
            item = self._source_item(row, cid_col)
            if item is not None and item.text() and int(item.text()) in loaded:
                ids.append(int(item.text()))
        return ids

    def _on_unit_filter_changed(self) -> None:
<<<<<<< HEAD
        if self._row_order() == "peak_trial":
            self._sort_by_peak()
        self._draw_raster()
        self._refresh_firing_rates()
        self.unit_filter_changed.emit()
        self.unit_rows_changed.emit()
=======
        self._draw_raster()
        self._refresh_firing_rates()
        self.unit_filter_changed.emit()
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4

    def _spike_table(self) -> SpikeTable:
        """Every spike of the loaded units, time-sorted; rebuilt only when the TsGroup is replaced."""
        if self._spike_table_cache is None or self._spike_table_cache[0] is not self._tsgroup:
            self._spike_table_cache = (self._tsgroup, SpikeTable.from_tsgroup(self._tsgroup))
        return self._spike_table_cache[1]

    def _row_order(self) -> str:
        """The row order in effect (a ``ROW_ORDERS`` key): depth needs a probe."""
        order = self.app_state.get_with_default("raster_row_order")
        if order == "depth" and self._neurons_source != "kilosort":
            return "table"
        return order

    def _depth_ordered_channels(self) -> np.ndarray:
        """The probe's hardware channels, top of the probe first."""
        all_ch = self.plot_container.ephys_trace_plot._all_ordered_channels()
        if len(all_ch) == 0 and self._probe_channel_order is not None:
            # No raw recording behind the trace panel: the probe's own depth order.
            all_ch = self._probe_channel_order
        return all_ch

    def _unit_channels(self, unit_ids: list[int]) -> dict[int, int]:
        """The hardware channel each Kilosort unit sits on: its template's best, else the table's."""
        best = self._build_cluster_best_channel_map()
        table_ch: dict[int, int] = {}
        if self._cluster_df is not None and {"cluster_id", "ch"} <= set(self._cluster_df.columns):
            known = self._cluster_df.dropna(subset=["cluster_id", "ch"])
            table_ch = dict(zip(known["cluster_id"].astype(int), known["ch"].astype(int)))
        return {cid: best.get(cid, table_ch.get(cid, 0)) for cid in unit_ids}

    def ordered_unit_ids(self) -> list[int]:
        """The filtered units in the raster's row order, top row first — the one unit order.

        Everything that lists units by row (raster, firing rates) reads this,
        so their rows agree.
        """
        unit_ids = self.filtered_unit_ids()
        order = self._row_order()
        if order == "rastermap":
            return order_units(unit_ids, self._rastermap_units)
<<<<<<< HEAD
        if order in PEAK_ORDERS:
            return order_units(unit_ids, self._peak_units)
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        if order == "depth":
            channels = self._unit_channels(unit_ids)
            rank = {int(hw): i for i, hw in enumerate(self._depth_ordered_channels())}
            return sorted(unit_ids, key=lambda cid: rank.get(channels[cid], len(rank)))
        return unit_ids

    def _raster_rows(self, unit_ids: list[int]) -> dict[int, int]:
        """The raster row key of each unit, after syncing the raster's y-axis to them.

        By depth, Kilosort units sit at their best channel (the trace panel's
        y-space); in every other order each unit gets a row of its own, in the
<<<<<<< HEAD
        order ``unit_ids`` come in. Either way a row is labelled with the ids
        of the units on it.
=======
        order ``unit_ids`` come in.
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        """
        raster = self.plot_container.raster_plot
        if self._row_order() == "depth":
            raster.follows_trace_y = True
            all_ch = self._depth_ordered_channels()
            total = len(all_ch)
            if total > 0:
                spacing = self.plot_container.ephys_trace_plot.buffer.channel_spacing
                raster.sync_y_axis({int(hw): (total - 1 - i) * spacing for i, hw in enumerate(all_ch)}, spacing, total)
<<<<<<< HEAD
            channels = self._unit_channels(unit_ids)
            on_channel: dict[int, list[int]] = {}
            for cid in unit_ids:
                on_channel.setdefault(channels[cid], []).append(cid)
            self._raster_row_units = on_channel
            raster.set_row_labels({hw: ", ".join(map(str, ids)) for hw, ids in on_channel.items()})
            return channels
=======
            return self._unit_channels(unit_ids)
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        raster.follows_trace_y = False
        rows = {cid: i for i, cid in enumerate(unit_ids)}
        total = len(rows)
        raster.sync_y_axis({i: float(total - 1 - i) for i in range(total)}, 1.0, total)
<<<<<<< HEAD
        raster.set_row_labels({i: str(cid) for cid, i in rows.items()})
        self._raster_row_units = {i: [cid] for cid, i in rows.items()}
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        self._pynapple_cid_to_row = rows
        return rows

    def _draw_raster(self) -> None:
        """The filtered units in grey; the selected ones among them in their colours."""
        if not self.plot_container or self._tsgroup is None:
            return
        trial_ep = self._trial_ep()
        if trial_ep is None:
            return
        rows = self._raster_rows(self.ordered_unit_ids())
        times, units = self._spike_table().window(float(trial_ep.start[0]), float(trial_ep.end[0]))
        entries = group_by_color(
            times - self._ephys_offset(), units, rows, self._multi_cluster_colors, _UNSELECTED_DOT_COLOR
        )
        self.plot_container.raster_plot.set_multi_cluster_spike_data(entries)

    # ------------------------------------------------------------------
    # Raster rendering + row order
    # ------------------------------------------------------------------

    def _sync_row_order_combo(self) -> None:
        """Show the row order in effect, and grey out probe depth for units without a probe."""
        combo = self.raster_row_order_combo
        combo.model().item(combo.findData("depth")).setEnabled(self._neurons_source == "kilosort")
        order = self._row_order()
        combo.blockSignals(True)
        combo.setCurrentIndex(combo.findData(order))
        combo.blockSignals(False)
<<<<<<< HEAD
        self.order_apply_btn.setText("Refit" if order == "rastermap" else "Sort now")
        self.order_apply_btn.setEnabled(order in ("rastermap", "peak_window"))
        self.order_window_spin.setVisible(order in PEAK_ORDERS)
=======
        self.rastermap_fit_btn.setEnabled(order == "rastermap")
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4

    def _on_raster_render_changed(self, _index: int) -> None:
        self.app_state.raster_render_mode = self.raster_render_combo.currentData()
        if self.plot_container:
            self.plot_container.raster_plot.refresh()

    def _on_raster_tick_width_changed(self, width: int) -> None:
        self.app_state.raster_tick_width = width
        if self.plot_container:
            self.plot_container.raster_plot.refresh()

    def _on_raster_row_order_changed(self, _index: int) -> None:
        self.app_state.raster_row_order = self.raster_row_order_combo.currentData()
        self._sync_row_order_combo()
<<<<<<< HEAD
        order = self._row_order()
        if order == "rastermap" and self._rastermap_units is None:
            # Picking the order asks for the fit; the button is for fitting it again.
            self._fit_rastermap()
            return
        if order in PEAK_ORDERS:
            self._sort_by_peak()
        self._apply_unit_order()

    def _apply_unit_order(self) -> None:
        """Redraw every view that lists units by row, in the order now in effect."""
        self._draw_raster()
        self._refresh_firing_rates()
        self.unit_rows_changed.emit()

    def _on_cluster_table_sorted(self, *_) -> None:
        if self._row_order() == "table":
            self._apply_unit_order()

    def _on_order_apply_clicked(self) -> None:
        if self._row_order() == "rastermap":
            self._fit_rastermap()
            return
        self._sort_by_peak()
        self._apply_unit_order()

    def _on_order_window_changed(self, window_s: float) -> None:
        self.app_state.heatmap_sort_window_s = window_s
        if self._row_order() == "peak_trial":
            self._sort_by_peak()
            self._apply_unit_order()

    def _peak_sort_span(self) -> tuple[float, float] | None:
        """The stretch of the spike clock a peak order is read from: the trial, or what is on screen."""
        span = self._trial_span_spike_clock()
        if span is None or self._row_order() != "peak_window" or not self.plot_container:
            return span
        offset = self._ephys_offset()
        x0, x1 = self.plot_container.raster_plot.get_current_xlim()
        t0, t1 = max(x0 + offset, span[0]), min(x1 + offset, span[1])
        return (t0, t1) if t1 > t0 else span

    def _sort_by_peak(self) -> None:
        """Order the filtered units by the window their firing rate is highest in: earliest first."""
        span = self._peak_sort_span()
        unit_ids = self.filtered_unit_ids()
        if self._tsgroup is None or span is None or not unit_ids:
            return
        rates, centers = self._bin_rates(unit_ids, *span)
        if len(centers) == 0:
            return
        order = argmax_window_order(
            rates,
            centers,
            float(self.app_state.get_with_default("heatmap_sort_window_s")),
            float(self.app_state.get_with_default("heatmap_sort_overlap")),
        )
        self._peak_units = [unit_ids[i] for i in order]
=======
        if self._row_order() == "rastermap" and self._rastermap_units is None:
            # Picking the order asks for the fit; the button is for fitting it again.
            self._fit_rastermap()
            return
        self._draw_raster()
        self._refresh_firing_rates()

    def _on_cluster_table_sorted(self, *_) -> None:
        if self._row_order() == "table":
            self._draw_raster()
            self._refresh_firing_rates()
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4

    def _fit_rastermap(self) -> None:
        """Fit Rastermap on the filtered units' spike counts over the visible window; keep the order."""
        if self._tsgroup is None:
            return
        unit_ids = self.filtered_unit_ids()
        trial_ep = self._trial_ep()
        if trial_ep is None:
            return
        if len(unit_ids) < RASTERMAP_MIN_ROWS:
            notify(f"Rastermap needs at least {RASTERMAP_MIN_ROWS} units; {len(unit_ids)} pass the filters.", "warning")
            return
        t_start, t_stop = float(trial_ep.start[0]), float(trial_ep.end[0])
        # The user's bin, widened only when the window would exceed what the fit takes.
        bin_size = max(self.app_state.fr_bin_size, (t_stop - t_start) / RASTERMAP_MAX_SAMPLES)
        counts, _ = self._spike_table().count(unit_ids, t_start, t_stop, bin_size)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            order = rastermap_order(counts)
        except ValueError as e:
            notify(str(e), "warning")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._rastermap_units = [unit_ids[i] for i in order]
        self._draw_raster()
        self._refresh_firing_rates()
<<<<<<< HEAD
        self.unit_rows_changed.emit()
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        notify(f"Rastermap order fitted on {len(unit_ids)} units")

    def _build_cluster_best_channel_map(self) -> dict[int, int]:
        if self._templates is None or self._channel_map is None:
            return {}
        best_map: dict[int, int] = {}
        for cluster_id in range(self._templates.shape[0]):
            template = self._templates[cluster_id]
            amplitude = np.max(template, axis=0) - np.min(template, axis=0)
            site_idx = int(np.argmax(amplitude))
            if site_idx < len(self._channel_map):
                best_map[cluster_id] = int(self._channel_map[site_idx])
            else:
                best_map[cluster_id] = site_idx
        return best_map

    def _load_file(self, path: Path, loader, flatten: bool = False, **kwargs):
        if not path.exists():
            return None
        try:
            data = loader(path, **kwargs)
            return data.flatten() if flatten else data
        except (OSError, ValueError) as e:
            notify(f"Failed to load {path.name}: {e}", "warning")
            return None

    def _resolve_dat_path(self, ks_folder: Path, params: dict | None = None) -> Path | None:
        """Locate the raw recording a Kilosort folder was sorted from.

        ``params.py`` records ``dat_path`` as it was on the sorting machine, so
        it routinely points at a drive that does not exist here.  The raw file
        normally sits in the Kilosort folder or its parent (Kilosort writes its
        output into a subfolder of the recording), so both are searched.
        """
        search_dirs = [ks_folder, ks_folder.parent]

        dat_path_str = (params or self._kilosort_params or {}).get("dat_path")
        if dat_path_str:
            candidate = Path(dat_path_str)
            if candidate.is_file():
                return candidate
            for folder in search_dirs:
                for relative in (folder / candidate.name, folder / candidate):
                    if relative.is_file():
                        return relative

        # No usable dat_path: fall back to the largest raw file, since a
        # recording sits alongside much smaller aux/digital-in companions.
        for folder in search_dirs:
            try:
                raw = [e for e in folder.iterdir() if e.is_file() and e.suffix.lower() in EPHYS_EXTENSIONS_RAW]
            except OSError:
                continue
            if raw:
                return max(raw, key=lambda e: e.stat().st_size)

        ephys_path_str = getattr(self.app_state, "ephys_path", None)
        if ephys_path_str:
            ephys_path = Path(ephys_path_str)
            if ephys_path.is_file() and ephys_path.suffix.lower() in EPHYS_EXTENSIONS_RAW:
                return ephys_path

        return None

    def _register_dat_fallback(self, ks_folder: Path):
        """Set up the Phy-Viewer panel using phylib reader for the raw .dat file."""
        if not self._kilosort_params:
            return

        dat_path = self._resolve_dat_path(ks_folder)
        if dat_path is None:
            return

        sr = self._kilosort_params["sample_rate"]
        n_channels = self._kilosort_params.get("n_channels_dat")

        if n_channels is None and self._channel_map is not None:
            n_channels = int(self._channel_map.max()) + 1

        if n_channels is None:
            notify("Cannot determine n_channels_dat — skipping Phy viewer.", "warning")
            return

        try:
            loader = load_ephys(
                dat_path,
                stream_id="0",
                n_channels=n_channels,
                sampling_rate=sr,
            )
        except Exception:
            notify(f"Failed to open .dat file: {dat_path}", "warning")
            return

        self._phy_reader = loader
        self._phy_n_channels = n_channels

        if not self.app_state.ephys_source_map:
            display_name = "Ephys Waveform"
            self.app_state.ephys_source_map[display_name] = (str(dat_path), "0", 0)
            self.app_state.ephys_stream_sel = display_name

        self.app_state.has_neo = True
        self.app_state.has_neurons = True
        notify(f"Phy viewer: {dat_path.name} ({n_channels} ch, {sr:.0f} Hz)")

    def _get_hardware_label(self) -> str:
        ephys_path, stream_id, _ = self.app_state.get_ephys_source()
        if not ephys_path:
            return "Hardware"
        ext = Path(ephys_path).suffix.lower()
        rawio_name = _RefNeo.KNOWN_EXTENSIONS.get(ext)
        return _RAWIO_TO_DISPLAY.get(rawio_name, "Hardware")

    def get_hw_names(self, channel_map: np.ndarray | None) -> dict[int, str] | None:
        if channel_map is None:
            return None
        ephys_path, stream_id, _ = self.app_state.get_ephys_source()
        if not ephys_path:
            return None
        try:
            loader = load_ephys(ephys_path, stream_id)
        except Exception:
            return None
        if not hasattr(loader, "channel_names"):
            return None
        channel_names = loader.channel_names
        if channel_names is None:
            return None
        return {int(ch): channel_names[ch] if ch < len(channel_names) else f"Ch {ch}" for ch in channel_map}

    @staticmethod
    def _make_item(text: str, sort_value: float | None = None, user_data=None) -> QStandardItem:
        item = QStandardItem(text)
        item.setEditable(False)
        if sort_value is not None:
            item.setData(sort_value, SORT_ROLE)
        if user_data is not None:
            item.setData(user_data, Qt.UserRole)
        return item

    @staticmethod
    def _format_value(value) -> tuple[str, float | None]:
        """Return (display_text, numeric_sort_value_or_None).

        Integers display without decimals; floats display with 3 d.p.
        Non-numeric values return (str, None).
        """
        if pd.isna(value):
            return "", None
        try:
            fval = float(value)
            if fval == int(fval):
                return str(int(fval)), fval
            return f"{fval:.3f}", fval
        except (ValueError, TypeError):
            return str(value), None

    def _compute_isi_per_cluster(self) -> dict[int, float]:
        """Mean ISI in ms for each cluster."""
        if self._tsgroup is None:
            return {}
        isi_map: dict[int, float] = {}
        for cid in self._tsgroup.keys():
            intervals = np.diff(self._tsgroup[cid].times())
            isi_map[int(cid)] = float(np.mean(intervals) * 1000.0) if len(intervals) > 0 else np.nan
        return isi_map

    def _populate_cluster_table(self, df: pd.DataFrame):
        _PRIORITY = [
            "cluster_id",
            "ch",
            "sh",
            "KSLabel",
            "group",
            "fr",
            "Amplitude",
            "n_spikes",
        ]
        _EXCLUDE = {"amp", "id_orig", "group_order"} | {c for c in df.columns if c.startswith("Unnamed")}

        ordered_cols = [c for c in _PRIORITY if c in df.columns]
        extra_cols = [c for c in df.columns if c not in set(_PRIORITY) and c not in _EXCLUDE]
        ordered_cols.extend(extra_cols)

        # Insert ISI column after n_spikes (or after last priority col)
        isi_map = self._compute_isi_per_cluster()
        if isi_map:
            insert_at = ordered_cols.index("n_spikes") + 1 if "n_spikes" in ordered_cols else len(ordered_cols)
            ordered_cols.insert(insert_at, "ISI (ms)")

        has_ch = "ch" in ordered_cols
        hw_names = self.get_hw_names(self._channel_map) if has_ch else None
        has_distinct_hw = hw_names is not None and any(name != f"Ch {hw}" for hw, name in hw_names.items())
        hw_label = self._get_hardware_label() if has_distinct_hw else ""

        # Build header labels (group → Human, KSLabel → KS, cluster_id → id, ch → ch(KS) + ch(hw))
        # Trailing space creates a small gap between text and the filter-zone separator.
        header_labels: list[str] = []
        for col in ordered_cols:
            if col == "group":
                header_labels.append("Human  ")
            elif col == "KSLabel":
                header_labels.append("KS  ")
            elif col == "cluster_id":
                header_labels.append("id  ")
            elif col == "ch" and has_distinct_hw:
                header_labels.append("ch (KS)  ")
                header_labels.append(f"ch ({hw_label})  ")
            else:
                header_labels.append(col + "  ")

        self.cluster_table.setSortingEnabled(False)
        model = self._cluster_model
        model.clear()
        model.setHorizontalHeaderLabels(header_labels)

        for _, row in df.iterrows():
            cluster_id = int(row["cluster_id"]) if "cluster_id" in row.index and pd.notna(row["cluster_id"]) else None
            row_items: list[QStandardItem] = []
            for col in ordered_cols:
                if col == "ISI (ms)":
                    isi_val = isi_map.get(cluster_id, np.nan) if cluster_id is not None else np.nan
                    if pd.isna(isi_val):
                        row_items.append(self._make_item(""))
                    else:
                        text, sv = self._format_value(isi_val)
                        row_items.append(self._make_item(text, sv))
                elif col == "ch" and has_distinct_hw:
                    value = row[col]
                    ks_ch = int(value) if pd.notna(value) else 0
                    hw_name = hw_names.get(ks_ch) if hw_names else None
                    hw_display = hw_name if hw_name else str(ks_ch)
                    row_items.append(self._make_item(str(ks_ch), float(ks_ch)))
                    row_items.append(self._make_item(hw_display, float(ks_ch), user_data=ks_ch))
                else:
                    value = row[col] if col in row.index else None
                    text, sv = self._format_value(value)
                    row_items.append(self._make_item(text, sv))
            model.appendRow(row_items)

        self.cluster_table.setSortingEnabled(True)
        self._setup_filter_header(header_labels)
        self._apply_default_human_label_filter(header_labels)
        stripped = [h.strip() for h in header_labels]
        cid_view_col = stripped.index("id") if "id" in stripped else None
        if cid_view_col is not None:
            self.cluster_table.setItemDelegateForColumn(cid_view_col, self._cluster_id_delegate)

    def _setup_filter_header(self, col_names: list[str]):
        # ch/sh/id are always categorical even though they hold integers
        _force_cat = {"ch", "sh", "id"}
        model = self._cluster_model
        cat_cols: set[int] = set()
        num_cols: set[int] = set()
        self._filter_col_cats.clear()

        for col_idx, col_name in enumerate(col_names):
            name = col_name.strip()
            force_cat = name in _force_cat or name.startswith("ch (")
            if force_cat:
                unique_vals: list[str] = []
                for row in range(model.rowCount()):
                    item = model.item(row, col_idx)
                    if item and item.text() and item.text() not in unique_vals:
                        unique_vals.append(item.text())
                self._filter_col_cats[col_idx] = sorted(unique_vals)
                cat_cols.add(col_idx)
            else:
                # Check whether the column has numeric sort values
                is_numeric = any(
                    model.item(row, col_idx) is not None and model.item(row, col_idx).data(SORT_ROLE) is not None
                    for row in range(model.rowCount())
                )
                if is_numeric:
                    num_cols.add(col_idx)
                else:
                    unique_vals = []
                    for row in range(model.rowCount()):
                        item = model.item(row, col_idx)
                        if item and item.text() and item.text() not in unique_vals:
                            unique_vals.append(item.text())
                    self._filter_col_cats[col_idx] = sorted(unique_vals)
                    cat_cols.add(col_idx)

        self._filter_cat_cols = cat_cols
        self._filter_num_cols = num_cols
        self._filter_header.set_filterable(cat_cols, num_cols)
        self._update_header_active_filters()

    def _apply_default_human_label_filter(self, col_names: list[str]):
        stripped = [h.strip() for h in col_names]
        if "Human" not in stripped:
            return
        col_idx = stripped.index("Human")
        if col_idx not in self._filter_cat_cols:
            return
        vals = self._filter_col_cats.get(col_idx, [])
        if not any(v for v in vals):
            return
        if "good" in vals:
            allowed = {"good"}
            self._filter_cat_active[col_idx] = allowed
            self._cluster_proxy.set_cat_filter(col_idx, allowed)
            self._update_header_active_filters()

    def _on_filter_header_clicked(self, logical_col: int):
        header_item = self._cluster_model.horizontalHeaderItem(logical_col)
        col_name = (header_item.text() if header_item else "").strip()
        if col_name in {"ch", "ch (KS)"} or (col_name.startswith("ch (") and not col_name.startswith("ch (KS")):
            self._open_probe_channel_dialog()
            return
        if logical_col in self._filter_cat_cols:
            values = self._filter_col_cats.get(logical_col, [])
            active = self._filter_cat_active.get(logical_col, set())
            dialog = CategoryFilterDialog(logical_col, values, active, self)
            if dialog.exec_() == QDialog.Accepted:
                allowed = dialog.get_allowed()
                if allowed:
                    self._filter_cat_active[logical_col] = allowed
                else:
                    self._filter_cat_active.pop(logical_col, None)
                self._cluster_proxy.set_cat_filter(logical_col, allowed)
                self._update_header_active_filters()
                self._on_unit_filter_changed()
        elif logical_col in self._filter_num_cols:
            current = self._filter_num_active.get(logical_col)
            dialog = NumericFilterDialog(logical_col, current, self)
            if dialog.exec_() == QDialog.Accepted:
                f = dialog.get_filter()
                if f is None:
                    self._filter_num_active.pop(logical_col, None)
                    self._cluster_proxy.set_numeric_filter(logical_col, None, None)
                else:
                    op, val = f
                    self._filter_num_active[logical_col] = f
                    self._cluster_proxy.set_numeric_filter(logical_col, op, val)
                self._update_header_active_filters()
                self._on_unit_filter_changed()

    def _update_header_active_filters(self):
        active = set(self._filter_cat_active.keys()) | set(self._filter_num_active.keys())
        self._filter_header.set_active_filters(active)

    def _find_col_by_header(self, prefix: str, exact: str | None = None) -> int | None:
        model = self._cluster_model
        for col in range(model.columnCount()):
            h = model.horizontalHeaderItem(col)
            if h is None:
                continue
            text = h.text().strip()
            if exact is not None and text == exact:
                return col
            if exact is None and text.startswith(prefix) and text != "ch (KS)":
                return col
        return None

    def _source_item(self, proxy_row: int, col: int) -> QStandardItem | None:
        proxy_idx = self._cluster_proxy.index(proxy_row, col)
        source_idx = self._cluster_proxy.mapToSource(proxy_idx)
        return self._cluster_model.itemFromIndex(source_idx)

    def selected_unit_ids(self) -> list[int]:
        """Ids of the units whose rows are selected in the neuron table, in table order."""
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return []
        rows = sorted(index.row() for index in self.cluster_table.selectionModel().selectedRows())
        items = [self._source_item(row, cid_col) for row in rows]
        return [int(item.text()) for item in items if item is not None]

    def _on_unit_row_clicked(self, click_info: dict) -> None:
        """Ctrl+click on a row of the raster or of a firing-rate heatmap toggles its units in the table."""
        ctrl = click_info.get("modifiers", Qt.NoModifier) & Qt.ControlModifier
        if click_info.get("button") != Qt.LeftButton or not ctrl or self.app_state.label_drawing_armed:
            return
        plot = click_info.get("plot")
        if plot is self.plot_container.raster_plot:
            unit_ids = self._raster_row_units.get(plot.row_at(click_info["y"]), [])
        elif plot in self.plot_container.heatmap_plots and plot._effective_feature() == FIRING_RATE_FEATURE:
            label = plot.row_label_at(click_info["y"])
            unit_ids = [] if label is None else [int(label)]
        else:
            return
        self.toggle_units_selected(unit_ids)

    def toggle_units_selected(self, unit_ids: list[int]) -> None:
        """Flip the table selection of *unit_ids*, leaving the rest of the selection as it is."""
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None or not unit_ids:
            return
        wanted = set(unit_ids)
        proxy = self._cluster_proxy
        selection = QItemSelection()
        for row in range(proxy.rowCount()):
            item = self._source_item(row, cid_col)
            if item is not None and int(item.text()) in wanted:
                selection.select(proxy.index(row, 0), proxy.index(row, 0))
        if selection.isEmpty():
            return
        self._selecting_from_plot = True
        try:
            self.cluster_table.selectionModel().select(
                selection, QItemSelectionModel.Toggle | QItemSelectionModel.Rows
            )
        finally:
            self._selecting_from_plot = False
        self.cluster_table.scrollTo(selection.indexes()[0])

    def _on_cluster_row_selected(self, _selected=None, _deselected=None):
        indexes = self.cluster_table.selectionModel().selectedRows()
        if not indexes:
            if self._multi_cluster_colors:
                self._clear_multi_cluster_mode()
            return

        hw_col_idx = self._find_col_by_header("ch (")
        ch_col_idx = hw_col_idx or self._find_col_by_header("", exact="ch")
        cluster_id_col_idx = self._find_col_by_header("", exact="id")

        # Navigate ephys channel to first selected row
        first_row = indexes[0].row()
        if ch_col_idx is not None:
            ch_item = self._source_item(first_row, ch_col_idx)
            if ch_item is not None:
                channel = ch_item.data(Qt.UserRole)
                if channel is None:
                    try:
                        channel = int(ch_item.text())
                    except (ValueError, TypeError):
                        channel = None
                if channel is not None:
                    self._apply_ephys_channel(int(channel))

        if cluster_id_col_idx is None:
            return

        if len(indexes) == 1:
            self._on_single_cluster_selected(first_row, cluster_id_col_idx, ch_col_idx)
        else:
            self._on_multi_cluster_selected(indexes, cluster_id_col_idx, ch_col_idx)

    def _on_single_cluster_selected(self, proxy_row: int, cid_col: int, hw_col_idx: int | None):
        cid_item = self._source_item(proxy_row, cid_col)
        if cid_item is None:
            return
        try:
            cluster_id = int(cid_item.text())
        except (ValueError, TypeError):
            return

        self._multi_cluster_colors.clear()
        self._multi_cluster_colors[cluster_id] = _CLUSTER_COLORS[0]
        self._apply_cluster_colors_to_table()

        ks_ch = self._get_ks_channel_for_row(proxy_row, hw_col_idx)
        self._draw_spikes_for_cluster(cluster_id, ks_ch)
        if not self._selecting_from_plot:
            self._jump_to_first_spike()
        self._sync_cluster_id_to_combo(cluster_id)

        self._current_cluster_id_for_psth = cluster_id
        self.cluster_selected.emit(cluster_id)

    def _on_multi_cluster_selected(self, indexes, cid_col: int, hw_col_idx: int | None):
        if not self.plot_container:
            return

        has_ephys_trace = self._neurons_source == "kilosort" and self.plot_container.is_ephystrace()
        sr = None
        if has_ephys_trace:
            ephys_plot = self.plot_container.ephys_trace_plot
            sr = self._kilosort_sr
            if sr is None or sr <= 0:
                has_ephys_trace = False

        trial_ep = self._trial_ep()
        if trial_ep is None:
            return
        offset = self._ephys_offset()

        self._multi_cluster_colors.clear()
        cluster_entries = []

        for i, idx in enumerate(indexes):
            proxy_row = idx.row()
            cid_item = self._source_item(proxy_row, cid_col)
            if cid_item is None:
                continue
            try:
                cluster_id = int(cid_item.text())
            except (ValueError, TypeError):
                continue

            color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            self._multi_cluster_colors[cluster_id] = color

            if has_ephys_trace:
                times_global = self._tsgroup[cluster_id].restrict(trial_ep).times()
                samples_abs = np.round(times_global * sr).astype(np.int64)
                ks_ch = self._get_ks_channel_for_row(proxy_row, hw_col_idx)
                channels = self._best_channels_for_cluster(cluster_id, ks_ch)
                cluster_entries.append((times_global - offset, samples_abs, channels, color))

        self._apply_cluster_colors_to_table()
        if has_ephys_trace:
            ephys_plot.set_multi_cluster_spike_data(cluster_entries)
        self._draw_raster()

    def _on_visible_channels_changed(self, _first: int, _last: int):
        pass

    def _apply_probe_channel_filter(self):
        ch_col = self._find_col_by_header("", exact="ch") or self._find_col_by_header("", exact="ch (KS)")
        if self._custom_channel_set is not None and ch_col is not None:
            self._cluster_proxy.set_visible_channel_filter(set(int(c) for c in self._custom_channel_set), ch_col)
        else:
            self._cluster_proxy.set_visible_channel_filter(None, None)
        self._on_unit_filter_changed()

    def _select_clusters_all_visible(self):
        """Select all filtered-visible rows, highlight their spikes, then disable auto-highlight."""
        if self._cluster_proxy.rowCount() == 0:
            notify("No clusters in current view.")
            return
        self.cluster_table.selectAll()

    def _update_highlight_label(self, _=None):
        n = self.n_closest_spin.value()
        self._highlight_label.setText(f"Highlight clusters (on {n} closest):")

    def _unselect_clusters(self):
        """Clear spike overlays, table selection, and disable auto-highlight."""
        self.cluster_table.clearSelection()
        self._clear_multi_cluster_mode()

    def _clear_multi_cluster_mode(self):
        self._multi_cluster_colors.clear()
        self._apply_cluster_colors_to_table()
        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.clear_spike_overlays()
        self._draw_raster()

    def _apply_cluster_colors_to_table(self):
        self.sync_firing_rate_panels()
        model = self._cluster_model
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return

        for row in range(model.rowCount()):
            item = model.item(row, cid_col)
            if item is None:
                continue
            try:
                cluster_id = int(item.text())
            except (ValueError, TypeError):
                continue

            color = self._multi_cluster_colors.get(cluster_id)
            cid_item = model.item(row, cid_col)
            if cid_item:
                cid_item.setData(QColor(*color[:3]) if color else None, _COLOR_ROLE)

    def _jump_to_first_spike(self):
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return
        self.plot_container.ephys_trace_plot.jump_to_spike(delta=0)

    def _reorder_probe_by_position(self):
        if self._channel_positions is None:
            return
        y_coords = self._channel_positions[:, 1]
        depth_order = np.argsort(y_coords)[::-1]
        if self._channel_map is not None:
            self._probe_channel_order = self._channel_map[depth_order].astype(int)
        else:
            self._probe_channel_order = depth_order.astype(int)
        self.apply_probe_order()

    def apply_probe_order(self):
        if self._probe_channel_order is None:
            return
        if not self.plot_container or not self.plot_container.is_ephystrace():
            return
        self.plot_container.ephys_trace_plot.set_probe_channel_order(self._probe_channel_order)

    @staticmethod
    def _get_closest_channels(
        channel_positions: np.ndarray,
        channel_index: int,
        n: int | None = None,
    ) -> np.ndarray:
        """Get the n channels closest to *channel_index* on the probe.

        Direct port of ``phylib.io.model.get_closest_channels``.
        """
        x = channel_positions[:, 0]
        y = channel_positions[:, 1]
        x0, y0 = channel_positions[channel_index]
        d = (x - x0) ** 2 + (y - y0) ** 2
        out = np.argsort(d)
        if n:
            out = out[:n]
        return out

    def _find_best_channels(
        self,
        template: np.ndarray,
        n_closest_channels: int = 12,
        amplitude_threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Find the best channels for a given template.

        Copied: ``phylib.io.model.TemplateModel._find_best_channels``.
        https://github.com/cortex-lab/phylib/blob/master/phylib/io/model.py

        Parameters
        ----------
        template : (n_samples, n_channels) array
            Mean waveform from ``templates.npy[template_id]``.
        n_closest_channels : int
            Max spatial neighbours to consider (Phy default 12).
        amplitude_threshold : float
            Fraction of peak amplitude; channels below are excluded.
            0 keeps all n_closest_channels (Phy default).

        Returns
        -------
        channel_ids : array of int
            Selected channel indices, sorted by descending amplitude.
        amplitude : array of float
            Amplitude on each selected channel.
        best_channel : int
            Channel with maximum amplitude.
        """
        assert template.ndim == 2
        amplitude = template.max(axis=0) - template.min(axis=0)

        best_channel = int(np.argmax(amplitude))
        max_amp = amplitude[best_channel]

        peak_channels = np.nonzero(amplitude >= amplitude_threshold * max_amp)[0]

        close_channels = self._get_closest_channels(
            self._channel_positions,
            best_channel,
            n_closest_channels,
        )

        channel_ids = np.intersect1d(peak_channels, close_channels)

        order = np.argsort(amplitude[channel_ids])[::-1]
        channel_ids = channel_ids[order]
        amplitude_out = amplitude[channel_ids]

        return channel_ids, amplitude_out, best_channel

    def _draw_spikes_for_cluster(self, cluster_id: int, channel: int):
        if self._tsgroup is None or not self.plot_container:
            return

        trial_ep = self._trial_ep()
        if trial_ep is None:
            return
        # Draw waveforms on ephys trace (Kilosort only)
        if self._neurons_source == "kilosort" and self.plot_container.is_ephystrace():
            sr = self._kilosort_sr
            if sr is not None and sr > 0:
                times_global = self._tsgroup[cluster_id].restrict(trial_ep).times()
                samples_abs = np.round(times_global * sr).astype(np.int64)
                channels = self._best_channels_for_cluster(cluster_id, channel)
                self.plot_container.ephys_trace_plot.set_spike_data(
                    times_global - self._ephys_offset(), samples_abs, channels
                )
        self._draw_raster()

    def _best_channels_for_cluster(self, cluster_id: int, fallback_channel: int) -> list[int]:
        if (
            self._templates is not None
            and self._channel_positions is not None
            and self._channel_map is not None
            and cluster_id < self._templates.shape[0]
        ):
            template = self._templates[cluster_id]
            n_closest = self.n_closest_spin.value()
            site_indices, _amp, _best = self._find_best_channels(
                template,
                n_closest_channels=n_closest,
            )
            hw_channels = [int(self._channel_map[i]) for i in site_indices if i < len(self._channel_map)]
            return hw_channels if hw_channels else [fallback_channel]
        return [fallback_channel]

    def _get_ks_channel_for_row(self, proxy_row: int, hw_col_idx: int | None) -> int:
        if hw_col_idx is not None:
            ch_item = self._source_item(proxy_row, hw_col_idx)
            if ch_item is not None:
                val = ch_item.data(Qt.UserRole)
                if val is not None:
                    return int(val)
                try:
                    return int(ch_item.text())
                except (ValueError, TypeError):
                    pass
        return self.ephys_channel_spin.value()

    def _apply_ephys_channel(self, channel: int):
        stream_sel = getattr(self.app_state, "ephys_stream_sel", None)
        source_map = getattr(self.app_state, "ephys_source_map", {})
        if not stream_sel or stream_sel not in source_map:
            return

        filename, stream_id, _ = source_map[stream_sel]
        source_map[stream_sel] = (filename, stream_id, channel)

        self.ephys_channel_spin.blockSignals(True)
        self.ephys_channel_spin.setValue(channel)
        self.ephys_channel_spin.blockSignals(False)

        if self.plot_container and self.plot_container.is_ephystrace():
            self.plot_container.ephys_trace_plot.set_channel(channel)
            xmin, xmax = self.plot_container.get_current_xlim()
            self.plot_container.ephys_trace_plot.update_plot_content(xmin, xmax)

    # ------------------------------------------------------------------
    # Firing rate panel
    # ------------------------------------------------------------------

    def _create_firing_rate_panel(self, main_layout):
        self.firing_rate_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.firing_rate_panel.setLayout(layout)

        group = QGroupBox("Firing rates")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)
        group_layout.setContentsMargins(2, 2, 2, 2)
        group.setLayout(group_layout)
        layout.addWidget(group)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Bin (s):"))
        self.fr_bin_spin = QDoubleSpinBox()
        self.fr_bin_spin.setRange(0.001, 1.0)
        self.fr_bin_spin.setValue(self.app_state.fr_bin_size)
        self.fr_bin_spin.setSingleStep(0.005)
        self.fr_bin_spin.setDecimals(3)
        self.fr_bin_spin.setToolTip("Time bin width in seconds for spike counting")
        self.fr_bin_spin.valueChanged.connect(self._on_fr_param_changed)
        row2.addWidget(self.fr_bin_spin)
        row2.addWidget(QLabel("σ (bins):"))
        self.fr_sigma_spin = QDoubleSpinBox()
        self.fr_sigma_spin.setRange(0.0, 50.0)
        self.fr_sigma_spin.setValue(self.app_state.fr_sigma)
        self.fr_sigma_spin.setSingleStep(0.5)
        self.fr_sigma_spin.setDecimals(1)
        self.fr_sigma_spin.setToolTip("Gaussian smoothing width in bins (0 = no smoothing)")
        self.fr_sigma_spin.valueChanged.connect(self._on_fr_param_changed)
        row2.addWidget(self.fr_sigma_spin)
        group_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.fr_compute_btn = QPushButton("Compute")
        self.fr_compute_btn.setToolTip("Compute firing rates for the current trial")
        self.fr_compute_btn.clicked.connect(self._on_fr_compute_clicked)
        row3.addWidget(self.fr_compute_btn)
        self.fr_status_label = QLabel("")
        row3.addWidget(self.fr_status_label)
        row3.addStretch()
        group_layout.addLayout(row3)

        main_layout.addWidget(self.firing_rate_panel)

    def _on_fr_param_changed(self):
        self.app_state.fr_bin_size = self.fr_bin_spin.value()
        self.app_state.fr_sigma = self.fr_sigma_spin.value()
        self._refresh_firing_rates()

    def _trial_span_spike_clock(self) -> tuple[float, float] | None:
        """The current trial's start and stop on the spike clock, whatever the display basis."""
        bounds = self.app_state.trial_bounds
        if bounds is None:
            return None
        start = self._trial_start_session() + float(getattr(self.app_state, "ephys_offset", 0.0) or 0.0)
        return start, start + bounds.duration

    def _compute_firing_rates(self, force: bool = False) -> bool:
        """Bin the filtered units' spikes over the current trial into the ``firing_rate`` feature.

        One column per unit, in the raster's row order, on the display clock.
        The feature is a snapshot of this trial; ``on_trial_changed`` bins the
        next one. Returns whether the feature now holds this trial's rates.
        """
        loader = derived_loader_for(self.app_state)
        span = self._trial_span_spike_clock()
        if self._tsgroup is None or not self.plot_container or loader is None or span is None:
            return False

        unit_ids = self.ordered_unit_ids()
        if not unit_ids:
<<<<<<< HEAD
            notify("No units pass the neuron table's filters.", "warning")
=======
            notify("No units pass the cluster table's filters.", "warning")
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
            return False
        bin_size = self.fr_bin_spin.value()
        sigma = self.fr_sigma_spin.value()

        cache_key = (self.app_state.trials_sel, bin_size, sigma, tuple(unit_ids), self.app_state.display_basis)
        if not force and self._fr_cache_key == cache_key and loader.is_derived(FIRING_RATE_FEATURE):
            return True

        t_start, t_stop = span
        n_cells = int((t_stop - t_start) / bin_size) * len(unit_ids)
        if n_cells > _FIRING_RATE_MAX_CELLS:
            notify(
                f"Firing rates of {len(unit_ids)} units in {bin_size:g} s bins over this trial would be "
<<<<<<< HEAD
                f"{n_cells:,} values — raise the bin size or filter the neuron table.",
=======
                f"{n_cells:,} values — raise the bin size or filter the cluster table.",
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
                "warning",
            )
            return False

        rates, centers = self._bin_rates(unit_ids, t_start, t_stop)

        loader.register(
            DerivedFeature(
                FIRING_RATE_FEATURE,
                time=centers - self._ephys_offset(),
                values=rates,
                dim_labels=[str(cid) for cid in unit_ids],
                n_columns=len(unit_ids),
            )
        )
        self._fr_cache_key = cache_key
<<<<<<< HEAD
        self.fr_status_label.setText(f"{len(unit_ids)} units (neuron table filter)")
=======
        self.fr_status_label.setText(f"{len(unit_ids)} units (cluster table filter)")
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        if self.data_widget:
            self.data_widget.refresh_feature_choices()
        return True

    def _bin_rates(self, unit_ids: list[int], t_start: float, t_stop: float) -> tuple[np.ndarray, np.ndarray]:
        """Smoothed rates of *unit_ids* over ``[t_start, t_stop]`` (spike clock), with the panel's bin and σ.

        Shape ``(n_bins, n_units)`` and the bin centres, on the spike clock.
        """
        bin_size = self.fr_bin_spin.value()
        sigma = self.fr_sigma_spin.value()
        counts, centers = self._spike_table().count(unit_ids, t_start, t_stop, bin_size)
        rates = (counts / bin_size).astype(np.float32)
        if sigma > 0:
            rates = gaussian_filter1d(rates, sigma, axis=0)
        return rates, centers

    def firing_rate_units(self) -> list[str]:
        """The units a firing rate can be read for, as the ``firing_rate`` feature lists them."""
        return [str(cid) for cid in self.ordered_unit_ids()]

<<<<<<< HEAD
    def unit_class_rows(self, mappings: dict[int, dict]) -> dict[int, dict] | None:
        """The classes of a per-unit label vocabulary that have a row: the filtered units', top row first.

        A set made along the units (``unit_<id>`` classes) is shown the way
        the raster is — the units the neuron table's filters let through, in
        the raster's row order — so a row of one is the same unit as the row
        of the other. ``None`` for a vocabulary that is not per unit, or with
        no units loaded.
        """
        classes = classes_along(mappings, FIRING_RATE_UNIT_DIM)
        if not classes or self._tsgroup is None:
            return None
        return {classes[unit]: mappings[classes[unit]] for unit in self.firing_rate_units() if unit in classes}

    def selected_unit_classes(self, mappings: dict[int, dict]) -> dict[int, tuple]:
        """The classes of a per-unit label vocabulary whose units are selected in the neuron table.

        Each with the colour its unit is drawn in, so every panel marks a
        selected unit the same way.
        """
        classes = classes_along(mappings, FIRING_RATE_UNIT_DIM)
        return {
            classes[str(unit)]: self._multi_cluster_colors.get(unit, _CLUSTER_COLORS[0])
            for unit in self.selected_unit_ids()
            if str(unit) in classes
        }

    def spike_trains(self) -> dict[str, nap.Ts | nap.Tsd]:
        """Each filtered unit's spike train, keyed and ordered as :meth:`firing_rate_units` lists them."""
        if self._tsgroup is None:
            return {}
        return {str(cid): self._tsgroup[cid] for cid in self.ordered_unit_ids()}

    def firing_rate_windows(self, trials: list) -> Iterator[tuple[object, FiringRateReader, float, float, float]]:
        """Yield ``(trial, reader, t0, t1, shift)`` per trial: its firing rates, read on the spike clock.

=======
    def firing_rate_windows(self, trials: list) -> Iterator[tuple[object, FiringRateReader, float, float, float]]:
        """Yield ``(trial, reader, t0, t1, shift)`` per trial: its firing rates, read on the spike clock.

>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        The ``firing_rate`` feature is a snapshot of the trial on screen;
        this is the same reading for any trial, so an operation over
        ``app_state.trials`` can use the rates too. *shift* subtracted from
        the reader's times makes them trial-relative. A trial the session
        has no timing for is left out.
        """
        collection = getattr(self.app_state, "source_collection", None)
        align = getattr(self.app_state, "nwb_alignment", None)
        if self._tsgroup is None or collection is None or align is None:
            return
        reader = FiringRateReader(self, self.ordered_unit_ids())
        ephys_offset = float(getattr(self.app_state, "ephys_offset", 0.0) or 0.0)
        for trial in trials:
            idx = collection.trial_index(trial)
            if idx is None:
                continue
            start = float(align.start_time(trial) or 0.0) + ephys_offset
            yield trial, reader, start, start + collection.trial_range(idx).duration, start

    def _refresh_firing_rates(self) -> None:
<<<<<<< HEAD
        """Re-bin the firing rates if they are in use: the trial, the units or the bin changed.

        In use means they are a feature already, or a panel names them — a
        restored layout does, before anything has binned them.
        """
        loader = derived_loader_for(self.app_state)
        if loader is None or not self.plot_container:
            return
        shown = [
            plot
            for plot in [*self.plot_container.line_plots, *self.plot_container.heatmap_plots]
            if plot._effective_feature() == FIRING_RATE_FEATURE
        ]
        if not shown and not loader.is_derived(FIRING_RATE_FEATURE):
            return
=======
        """Re-bin the firing rates if they are a feature: the trial, the units or the bin changed."""
        loader = derived_loader_for(self.app_state)
        if loader is None or not loader.is_derived(FIRING_RATE_FEATURE):
            return
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        if not self._compute_firing_rates(force=True):
            # Nothing to bin here: a stale snapshot would show another trial's rates as this one's.
            loader.unregister(FIRING_RATE_FEATURE)
            if self.data_widget:
                self.data_widget.refresh_feature_choices()
            return
<<<<<<< HEAD
        for plot in shown:
            plot.resync_selections()
            plot.invalidate_data()
            plot.update_plot()
        self.sync_firing_rate_panels()

    def sync_firing_rate_panels(self) -> None:
        """Make every firing-rate heatmap a neuron panel: rows in the unit order, selected units outlined."""
        if not self.plot_container:
            return
        colors = {str(cid): color for cid, color in self._multi_cluster_colors.items()}
        for plot in self.plot_container.heatmap_plots:
            shows_rates = plot._effective_feature() == FIRING_RATE_FEATURE
            plot.keeps_source_order = shows_rates
            plot.set_row_highlights(colors if shows_rates else {})

=======
        for plot in [*self.plot_container.line_plots, *self.plot_container.heatmap_plots]:
            if plot._effective_feature() == FIRING_RATE_FEATURE:
                plot.resync_selections()
                plot.invalidate_data()
                plot.update_plot()

>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
    def _on_fr_compute_clicked(self) -> None:
        """Compute the firing rates and show them in the active feature panel."""
        if not self._compute_firing_rates(force=True) or not self.app_state.ready:
            return
        features_combo = self.data_widget.combos.get("features")
        if features_combo is not None:
            set_combo_to_value(features_combo, FIRING_RATE_FEATURE)
            self.data_widget.apply_panel_control("features", get_combo_value(features_combo))
        self.data_widget.update_main_plot()

    def _get_any_ephys_loader(self):
        if self._phy_reader is not None:
            return self._phy_reader

        ephys_path, stream_id, _ = self.app_state.get_ephys_source()
        if not ephys_path:
            return None
        try:
            return load_ephys(ephys_path, stream_id)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Bidirectional cluster_id sync
    # ------------------------------------------------------------------

    def _sync_cluster_id_to_combo(self, cluster_id: int):
        if not self.data_widget:
            return
        combo = self.data_widget.combos.get("cluster_id")
        if combo is None:
            return
        combo.blockSignals(True)
        idx = find_combo_index(combo, str(cluster_id))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)
        self.app_state.set_key_sel("cluster_id", str(cluster_id))

    def select_cluster_in_table(self, cluster_id: int):
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return
        sel_model = self.cluster_table.selectionModel()
        sel_model.blockSignals(True)
        for proxy_row in range(self._cluster_proxy.rowCount()):
            item = self._source_item(proxy_row, cid_col)
            if item and item.text() == str(cluster_id):
                self.cluster_table.selectRow(proxy_row)
                sel_model.blockSignals(False)
                return
        sel_model.blockSignals(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container
        plot_container.plot_changed.connect(self._on_plot_changed)
        plot_container.raster_plot.plot_clicked.connect(self._on_unit_row_clicked)
        for plot in plot_container.heatmap_plots:
            self._follow_panel(plot)
        plot_container.panel_added.connect(self._follow_panel)
        ephys_plot = plot_container.ephys_trace_plot
        if ephys_plot is not None:
            ephys_plot.gain_scroll_requested.connect(self._on_gain_scroll)
            ephys_plot.visible_channels_changed.connect(self._on_visible_channels_changed)

    def _follow_panel(self, plot) -> None:
        """A Ctrl+click on a unit's row in a new panel selects that unit, as in the raster."""
        plot.plot_clicked.connect(self._on_unit_row_clicked)

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def set_data_widget(self, data_widget):
        self.data_widget = data_widget

    def _on_gain_scroll(self, delta: int):
        spin = self.ephys_gain_spin
        new_val = round(spin.value() + delta * 0.1, 1)
        spin.setValue(max(spin.minimum(), min(new_val, spin.maximum())))

    def on_trial_changed(self):
        if not self.data_widget:
            return
<<<<<<< HEAD
        if self._row_order() == "peak_trial":
            self._sort_by_peak()
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        # The firing rates are a snapshot of one trial: bin the new one in their place.
        self._refresh_firing_rates()

        self.configure_ephys_trace_plot()
        self.refresh_raster()
        if self.data_widget:
            self.data_widget.refresh_neo_panels()

    def _redraw_selected_clusters(self):
        if not self._multi_cluster_colors:
            return
        cid_col = self._find_col_by_header("", exact="id")
        if cid_col is None:
            return

        target_ids = set(self._multi_cluster_colors.keys())
        proxy = self._cluster_proxy
        sel_model = self.cluster_table.selectionModel()
        sel_model.blockSignals(True)
        sel_model.clearSelection()
        for row in range(proxy.rowCount()):
            val = proxy.data(proxy.index(row, cid_col))
            try:
                if int(val) in target_ids:
                    sel_model.select(
                        proxy.index(row, 0),
                        QItemSelectionModel.Select | QItemSelectionModel.Rows,
                    )
            except (ValueError, TypeError):
                pass
        sel_model.blockSignals(False)
        self._on_cluster_row_selected()

    def _on_plot_changed(self, plot_type: str):
        if plot_type == "ephystrace":
            self.apply_probe_order()
