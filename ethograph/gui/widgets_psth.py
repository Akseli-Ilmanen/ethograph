"""PSTH popup dialog — integrates with Ethograph's EphysWidget + LabelsWidget.

Opens when the user clicks "Open PSTH" in EphysWidget's traceview panel.
Cluster selection is driven by the cluster table in EphysWidget (not duplicated
here); the dialog listens to EphysWidget.cluster_selected signal.

Label-aligned PSTH
------------------
For each trial the dialog:
1. Gets the session-absolute start via ``nwb_alignment.start_time(trial)``
   (falls back to 0 if no session timing is available).
2. Reads label intervals from ``app_state.get_trial_intervals(trial)`` and
   finds the onset/offset of the chosen label class in trial-relative time.
3. Converts to session-absolute: ``ref_abs = trial_start_abs + local_t``.
4. Calls ``nap.compute_perievent`` on the EphysWidget's ``_tsgroup``.

Trial condition filtering reuses ``dt.filter_by_attr`` (same method that
NavigationWidget uses) — no logic is duplicated.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np  # noqa: E402
import pynapple as nap  # noqa: E402
from qtpy.QtCore import Qt, Signal  # noqa: E402
from qtpy.QtGui import QPixmap  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.notify import notify_dialog  # noqa: E402
from ethograph.utils.qt import add_combo_separator, color_icon, gray_icon  # noqa: E402

from .plots_psth import PSTHPlot, sort_trials  # noqa: E402

# ---------------------------------------------------------------------------
# Loading overlay
# ---------------------------------------------------------------------------


class _LoadingOverlay(QLabel):
    """Semi-transparent overlay shown over the plot during computation.

    Positioned as a child of the target widget so it covers it completely.
    Call ``activate()`` to show (includes a processEvents flush so it renders
    before the blocking computation begins), and ``hide()`` when done.
    """

    def __init__(self, parent: QWidget):
        super().__init__("Computing…", parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(10, 15, 25, 210);
                color: #88ccff;
                font-size: 22px;
                font-weight: bold;
                border-radius: 10px;
            }
        """)
        self.hide()

    def activate(self):
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()
        QApplication.processEvents()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class PSTHDialog(QDialog):
    """Peri-Stimulus Time Histogram popup.

    Parameters
    ----------
    app_state : ObservableAppState
    ephys_widget : EphysWidget — provides _tsgroup + cluster_selected signal
    labels_widget : LabelsWidget — provides _mappings for label colors
    navigation_widget : NavigationWidget — provides type_vars_dict for conditions
    """

    trial_jump_requested = Signal(str)  # trial_id → main GUI should navigate

    def __init__(self, app_state, ephys_widget, labels_widget, navigation_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSTH — Peri-Stimulus Time Histogram")
        self.resize(1200, 740)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self.app_state = app_state
        self.ephys_widget = ephys_widget
        self.labels_widget = labels_widget
        self.navigation_widget = navigation_widget

        self._current_cluster_id: int | None = None
        self._perievent: dict[int, np.ndarray] = {}
        self._current_trials: list[str] = []  # trial IDs in current PSTH
        self._ref_times_abs_map: dict[str, float] = {}  # trial_id → session-abs ref time
        self._start_map: dict[str, float] = {}  # trial_id → session-abs start

        self._build_ui()
        self._populate_align_combo()
        self._populate_condition_combo()
        self._populate_cluster_combo()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ---- title ----
        self._title_label = QLabel("Select a cluster in the TraceView table →")
        self._title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #88ccff; padding: 4px;")
        root.addWidget(self._title_label)

        # ---- body: sidebar + plot ----
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setSpacing(8)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self._build_sidebar())

        self._plot = PSTHPlot([])  # populated on first cluster select
        self._plot.trial_selected.connect(self._on_trial_selected)
        self._plot.trial_hovered.connect(self._on_trial_hovered)
        self._plot.hover_info.connect(self._on_hover_info)
        self._plot.trial_time_requested.connect(self._on_trial_time_requested)
        self._loading_overlay = _LoadingOverlay(self._plot)
        body_layout.addWidget(self._plot)
        root.addWidget(body)

        # ---- status bar ----
        self._status = QLabel("Select a cluster in EphysWidget to start.")
        self._status.setStyleSheet("color: #aaa; font-size: 11px; padding: 2px;")
        root.addWidget(self._status)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(240)
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)

        # Helper image
        _img_path = Path(__file__).parent.parent.parent / "docs" / "media" / "psth_helper.png"
        if _img_path.exists():
            pix = QPixmap(str(_img_path))
            img_label = QLabel()
            img_label.setPixmap(pix.scaledToWidth(220, Qt.SmoothTransformation))
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)

        # Cluster selection
        g = QGroupBox("Cluster")
        gl = QVBoxLayout(g)
        self._cluster_combo = QComboBox()
        self._cluster_combo.setToolTip("Select cluster to compute PSTH for")
        self._cluster_combo.currentIndexChanged.connect(self._on_cluster_combo_changed)
        gl.addLayout(self._make_stepper_row(self._cluster_combo, self._step_cluster))
        filter_row = QHBoxLayout()
        self._filter_visible_cb = QCheckBox("Filter to visible in Cluster Table")
        self._filter_visible_cb.setChecked(True)
        self._filter_visible_cb.toggled.connect(self._populate_cluster_combo)
        filter_row.addWidget(self._filter_visible_cb)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Refresh cluster list from Cluster Table")
        refresh_btn.clicked.connect(self._populate_cluster_combo)
        filter_row.addWidget(refresh_btn)
        gl.addLayout(filter_row)
        layout.addWidget(g)

        # Align to
        g = QGroupBox("Align to event")
        gl = QVBoxLayout(g)
        self._align_combo = QComboBox()
        self._align_combo.setToolTip(
            "Choose what each trial is aligned to.\nLabels show the first occurrence of that label per trial."
        )
        self._align_combo.currentIndexChanged.connect(self._on_align_changed)
        gl.addLayout(self._make_stepper_row(self._align_combo, self._step_align))

        # Onset / Offset radio (only relevant for labels)
        boundary_row = QHBoxLayout()
        self._onset_radio = QRadioButton("Onset")
        self._offset_radio = QRadioButton("Offset")
        self._onset_radio.setChecked(True)
        self._onset_radio.toggled.connect(self._recompute)
        boundary_row.addWidget(self._onset_radio)
        boundary_row.addWidget(self._offset_radio)
        self._boundary_widget = QWidget()
        self._boundary_widget.setLayout(boundary_row)
        self._boundary_widget.setVisible(False)
        gl.addWidget(self._boundary_widget)
        layout.addWidget(g)

        # Condition grouping
        g = QGroupBox("Color / group by condition")
        gl = QVBoxLayout(g)
        self._cond_key_combo = QComboBox()
        self._cond_key_combo.currentIndexChanged.connect(self._on_condition_key_changed)
        gl.addWidget(self._cond_key_combo)
        layout.addWidget(g)

        # Window
        g = QGroupBox("Window (s)")
        gl = QVBoxLayout(g)
        for lbl, attr, default in [
            ("Pre:", "_pre_spin", 0.5),
            ("Post:", "_post_spin", 1.5),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl))
            spin = QDoubleSpinBox()
            spin.setRange(0.01, 30.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(default)
            row.addWidget(spin)
            setattr(self, attr, spin)
            gl.addLayout(row)
        update_btn = QPushButton("Update window size")
        update_btn.clicked.connect(self._recompute)
        gl.addWidget(update_btn)
        layout.addWidget(g)

        # Bin + sort
        g = QGroupBox("PSTH bin (s)")
        gl = QVBoxLayout(g)
        self._bin_spin = QDoubleSpinBox()
        self._bin_spin.setRange(0.001, 2.0)
        self._bin_spin.setSingleStep(0.005)
        self._bin_spin.setDecimals(3)
        self._bin_spin.setValue(0.025)
        self._bin_spin.valueChanged.connect(self._replot)
        gl.addWidget(self._bin_spin)
        layout.addWidget(g)

        g = QGroupBox("Spike width (px)")
        gl = QVBoxLayout(g)
        self._spike_width_spin = QDoubleSpinBox()
        self._spike_width_spin.setRange(0.5, 8.0)
        self._spike_width_spin.setSingleStep(0.5)
        self._spike_width_spin.setDecimals(1)
        self._spike_width_spin.setValue(1.0)
        self._spike_width_spin.valueChanged.connect(self._on_spike_width_changed)
        gl.addWidget(self._spike_width_spin)
        layout.addWidget(g)

        g = QGroupBox("Sort trials by")
        gl = QVBoxLayout(g)
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["index", "spike_count", "rate (0→post)"])
        self._sort_combo.currentIndexChanged.connect(self._replot)
        gl.addWidget(self._sort_combo)
        layout.addWidget(g)

        layout.addStretch()

        # Selected trial info
        self._sel_label = QLabel("Hover or click a row")
        self._sel_label.setWordWrap(True)
        self._sel_label.setStyleSheet("color: #7ff; font-size: 12px; font-weight: bold; padding: 4px;")
        layout.addWidget(self._sel_label)

        # Jump to trial
        g = QGroupBox("Jump to trial #")
        gl = QHBoxLayout(g)
        self._jump_spin = QSpinBox()
        self._jump_spin.setRange(0, 0)
        self._jump_spin.setToolTip("Change to highlight that trial row")
        self._jump_spin.valueChanged.connect(self._on_jump_clicked)
        gl.addWidget(self._jump_spin)
        layout.addWidget(g)

        nav_btn = QPushButton("Navigate to trial →")
        nav_btn.setStyleSheet("font-weight: bold;")
        nav_btn.setToolTip("Seeks the main GUI timeline to the selected trial")
        nav_btn.clicked.connect(self._on_navigate_clicked)
        layout.addWidget(nav_btn)

        return sidebar

    # ------------------------------------------------------------------
    # Stepper controls (click through options without opening the dropdown)
    # ------------------------------------------------------------------

    def _make_stepper_row(self, combo: QComboBox, step_fn) -> QHBoxLayout:
        """Wrap ``combo`` between ◀/▶ buttons that step through its entries."""
        row = QHBoxLayout()
        prev_btn = QPushButton("◀")
        prev_btn.setFixedWidth(28)
        prev_btn.setToolTip("Previous")
        prev_btn.clicked.connect(lambda: step_fn(-1))
        row.addWidget(prev_btn)
        row.addWidget(combo, 1)
        next_btn = QPushButton("▶")
        next_btn.setFixedWidth(28)
        next_btn.setToolTip("Next")
        next_btn.clicked.connect(lambda: step_fn(1))
        row.addWidget(next_btn)
        return row

    def _step_combo(self, combo: QComboBox, delta: int, is_valid=None):
        """Advance ``combo`` by ``delta``, wrapping around and skipping separators.

        ``is_valid`` optionally further restricts which entries are valid stops;
        it receives each candidate's ``itemData`` and returns a bool.
        """
        n = combo.count()
        if n == 0:
            return
        idx = combo.currentIndex()
        for _ in range(n):
            idx = (idx + delta) % n
            data = combo.itemData(idx)
            if data is None:  # separators carry no data
                continue
            if is_valid is not None and not is_valid(data):
                continue
            combo.setCurrentIndex(idx)
            return

    def _step_cluster(self, delta: int):
        self._step_combo(self._cluster_combo, delta)

    def _step_align(self, delta: int):
        present = self._labels_present_in_session()
        # Trial start/end (str data) always valid; labels (int) only if they occur.
        self._step_combo(
            self._align_combo,
            delta,
            is_valid=lambda data: not isinstance(data, int) or data in present,
        )

    def _labels_present_in_session(self) -> set[int]:
        """Label ids that occur in at least one trial of the current session."""
        present: set[int] = set()
        for trial_id in self._build_trial_list():
            df = self.app_state.get_trial_intervals(trial_id)
            if df is not None and len(df):
                present.update(int(v) for v in df["labels"].unique())
        return present

    # ------------------------------------------------------------------
    # Combo population
    # ------------------------------------------------------------------

    def _get_visible_cluster_ids(self) -> list[int]:
        proxy = self.ephys_widget._cluster_proxy
        model = self.ephys_widget._cluster_model
        cluster_col = 0
        for col in range(model.columnCount()):
            h = model.horizontalHeaderItem(col)
            if h and h.text() == "cluster_id":
                cluster_col = col
                break
        ids = []
        for row in range(proxy.rowCount()):
            val = proxy.data(proxy.index(row, cluster_col))
            try:
                ids.append(int(val))
            except (ValueError, TypeError):
                pass
        return ids

    def _populate_cluster_combo(self):
        prev_id = self._cluster_combo.currentData()
        self._cluster_combo.blockSignals(True)
        self._cluster_combo.clear()

        if self._filter_visible_cb.isChecked():
            cluster_ids = self._get_visible_cluster_ids()
        else:
            tsgroup = getattr(self.ephys_widget, "_tsgroup", None)
            cluster_ids = sorted(tsgroup.keys()) if tsgroup is not None else []

        df = getattr(self.ephys_widget, "_cluster_df", None)
        for cid in cluster_ids:
            if df is not None and "cluster_id" in df.columns:
                rows = df[df["cluster_id"] == cid]
                if not rows.empty:
                    r = rows.iloc[0]
                    grp = str(r["group"]) if "group" in r.index else "?"
                    ch = int(r["ch"]) if "ch" in r.index else "?"
                    fr = f"{r['fr']:.1f}" if "fr" in r.index else "?"
                    label = f"[{grp}] #{cid}  ch={ch}  {fr}Hz"
                else:
                    label = f"#{cid}"
            else:
                label = f"#{cid}"
            self._cluster_combo.addItem(label, userData=cid)

        # Restore previous selection if still present
        restored = False
        if prev_id is not None:
            for i in range(self._cluster_combo.count()):
                if self._cluster_combo.itemData(i) == prev_id:
                    self._cluster_combo.setCurrentIndex(i)
                    restored = True
                    break

        self._cluster_combo.blockSignals(False)

        if not restored and self._cluster_combo.count() > 0:
            self._cluster_combo.setCurrentIndex(0)
            self._on_cluster_combo_changed(0)
        elif restored:
            # cluster unchanged — title still valid, no recompute needed
            self._update_title()

    def _on_cluster_combo_changed(self, index: int):
        if index < 0:
            return
        cid = self._cluster_combo.currentData()
        if cid is not None:
            self._on_cluster_selected(cid)

    def _populate_align_combo(self):
        self._align_combo.blockSignals(True)
        self._align_combo.clear()
        self._align_combo.addItem(gray_icon(), "Trial start", "trial_start")
        self._align_combo.addItem(gray_icon(), "Trial end", "trial_end")
        add_combo_separator(self._align_combo)

        if self.labels_widget:
            for label_id, data in self.labels_widget._mappings.items():
                if label_id == 0:
                    continue
                icon = color_icon(data["color"])
                self._align_combo.addItem(icon, data["name"], label_id)

        self._align_combo.blockSignals(False)

    def _populate_condition_combo(self):
        self._cond_key_combo.blockSignals(True)
        self._cond_key_combo.clear()
        self._cond_key_combo.addItem("None", None)

        catalog = getattr(self.navigation_widget, "catalog", None)
        for condition in catalog.trial_conditions if catalog else []:
            self._cond_key_combo.addItem(str(condition), condition)

        self._cond_key_combo.blockSignals(False)

    def _on_condition_key_changed(self):
        self._replot()

    def _get_condition_groups(self) -> tuple[dict[int, int] | None, list[str] | None]:
        key = self._cond_key_combo.currentData()
        if not key or not hasattr(self.app_state, "dt") or self.app_state.dt is None:
            return None, None

        dt = self.app_state.dt
        if dt is None:
            return None, None
        all_vals: list = sorted(
            {ds.attrs[key] for _, ds in dt.trial_items() if key in ds.attrs},
            key=str,
        )
        if not all_vals:
            return None, None

        val_to_idx = {v: i for i, v in enumerate(all_vals)}
        group: dict[int, int] = {}
        for trial_i, trial_id in enumerate(self._current_trials):
            try:
                ds = dt.trial(trial_id)
                val = ds.attrs.get(key)
                group[trial_i] = val_to_idx.get(val, 0)
            except (KeyError, AttributeError):
                group[trial_i] = 0

        return group, [str(v) for v in all_vals]

    def _on_align_changed(self):
        is_label = isinstance(self._align_combo.currentData(), int)
        self._boundary_widget.setVisible(is_label)
        self._recompute()

    # ------------------------------------------------------------------
    # Cluster selection (driven by EphysWidget signal)
    # ------------------------------------------------------------------

    def _on_cluster_selected(self, cluster_id: int):
        self._current_cluster_id = cluster_id
        self._update_title()
        self._select_cluster_in_table(cluster_id)
        self._recompute()

    def _select_cluster_in_table(self, cluster_id: int):
        cid_col = self.ephys_widget._find_col_by_header("", exact="id")
        if cid_col is None:
            return
        proxy = self.ephys_widget._cluster_proxy
        for row in range(proxy.rowCount()):
            val = proxy.data(proxy.index(row, cid_col))
            try:
                if int(val) == cluster_id:
                    table = self.ephys_widget.cluster_table
                    table.selectRow(row)
                    table.scrollTo(proxy.index(row, 0))
                    return
            except (ValueError, TypeError):
                pass

    def _update_title(self):
        cid = self._current_cluster_id
        align = self._align_combo.currentText() if self._align_combo.count() else "—"
        if cid is None:
            self._title_label.setText("Select a cluster in the TraceView table →")
        else:
            self._title_label.setText(f"Raster of Cluster {cid}  ·  aligned to '{align}'")

    # ------------------------------------------------------------------
    # Trial list (with condition filter)
    # ------------------------------------------------------------------

    def _build_trial_list(self) -> list[str]:
        return list(getattr(self.app_state, "trials", None) or [])

    # ------------------------------------------------------------------
    # Alignment time resolution
    # ------------------------------------------------------------------

    def _trial_abs_start(self, trial_id) -> float:
        nwb_alignment = getattr(self.app_state, "nwb_alignment", None)
        if nwb_alignment is None:
            nwb_alignment = getattr(self.app_state.dt, "nwb_alignment", None)
        if nwb_alignment is not None:
            return nwb_alignment.start_time(trial_id)
        return 0.0  # per-trial files: local == session-absolute

    def _trial_abs_end(self, trial_id) -> float:
        nwb_alignment = getattr(self.app_state, "nwb_alignment", None)
        if nwb_alignment is None:
            nwb_alignment = getattr(self.app_state.dt, "nwb_alignment", None)
        if nwb_alignment is not None:
            stop = nwb_alignment.stop_time(trial_id)
            if stop is not None:
                return stop
        else:
            return None

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _recompute(self):
        if self._current_cluster_id is None:
            return
        tsgroup = getattr(self.ephys_widget, "_tsgroup", None)
        if tsgroup is None or self._current_cluster_id not in tsgroup:
            self._status.setText("No spike data loaded.")
            return

        self._update_title()
        trials = self._build_trial_list()
        if not trials:
            self._status.setText("No trials available.")
            return

        align_data = self._align_combo.currentData()
        pre_s = self._pre_spin.value()
        post_s = self._post_spin.value()
        use_offset = self._offset_radio.isChecked()

        self._loading_overlay.activate()
        try:
            ref_times_abs: list[float] = []
            valid_display: list[int] = []

            # Precompute per-trial start times in one pass (avoids O(N²) _trial_idx scans)
            nwb_alignment = getattr(self.app_state, "nwb_alignment", None)
            if nwb_alignment is None:
                nwb_alignment = getattr(self.app_state.dt, "nwb_alignment", None)
            if nwb_alignment is not None:
                start_map = {t: nwb_alignment.start_time(t) for t in trials}
                end_map = {t: nwb_alignment.stop_time(t) for t in trials} if align_data == "trial_end" else {}
            else:
                start_map = {t: 0.0 for t in trials}
                end_map = {}

            # Precompute label times if needed (avoids repeated DataTree lookups)
            label_map: dict = {}
            if isinstance(align_data, int):
                col = "offset_s" if use_offset else "onset_s"
                for trial_id in trials:
                    df = self.app_state.get_trial_intervals(trial_id)
                    if df is not None and len(df):
                        match = df[df["labels"] == align_data]
                        label_map[trial_id] = float(match[col].iloc[0]) if len(match) else None
                    else:
                        label_map[trial_id] = None

            for i, trial_id in enumerate(trials):
                t_abs = start_map[trial_id]
                if align_data == "trial_start":
                    ref_times_abs.append(t_abs)
                    valid_display.append(i)
                elif align_data == "trial_end":
                    stop = end_map.get(trial_id)
                    if stop is None:
                        stop = self._trial_abs_end(trial_id)
                    ref_times_abs.append(stop)
                    valid_display.append(i)
                elif isinstance(align_data, int):
                    local_t = label_map.get(trial_id)
                    if local_t is not None:
                        ref_times_abs.append(t_abs + local_t)
                        valid_display.append(i)

            if not ref_times_abs:
                label_name = self._align_combo.currentText()
                notify_dialog(
                    f"Label '{label_name}' has no occurrences in any trial.\nFalling back to Trial start alignment.",
                    "warning",
                    "No events found",
                    self,
                )
                self._align_combo.blockSignals(True)
                self._align_combo.setCurrentIndex(0)  # "Trial start"
                self._align_combo.blockSignals(False)
                self._boundary_widget.setVisible(False)
                self._update_title()
                # restart with trial_start — guaranteed to have events
                align_data = self._align_combo.currentData()
                for i, trial_id in enumerate(trials):
                    ref_times_abs.append(self._trial_abs_start(trial_id))
                    valid_display.append(i)

            # Refs are on the alignment clock; the spike TsGroup runs on the
            # recording clock = alignment + the user's scalar ephys_offset.
            # Shift only the refs handed to pynapple — the stored maps stay on
            # the alignment clock for navigation. Without this the PSTH is
            # displaced from the raster/trace by exactly ephys_offset.
            eph_scalar = float(getattr(self.app_state, "ephys_offset", 0.0) or 0.0)
            ref_arr = np.array(ref_times_abs) + eph_scalar
            ref_ts = nap.Ts(t=ref_arr)
            peri_nap = nap.compute_perievent(tsgroup[self._current_cluster_id], ref_ts, window=(-pre_s, post_s))

            self._perievent = {
                valid_display[k]: (peri_nap[k].t if k in peri_nap else np.array([], dtype=np.float64))
                for k in range(len(ref_times_abs))
            }
            for i in range(len(trials)):
                if i not in self._perievent:
                    self._perievent[i] = np.array([], dtype=np.float64)

            # Store maps needed for double-click seek
            self._start_map = start_map
            self._ref_times_abs_map = {trials[valid_display[k]]: ref_times_abs[k] for k in range(len(ref_times_abs))}

            self._current_trials = trials
            try:
                trial_ints = [int(t) for t in trials]
                self._jump_spin.setRange(min(trial_ints), max(trial_ints))
            except (ValueError, TypeError):
                self._jump_spin.setRange(0, max(0, len(trials) - 1))

            trial_ids = [str(t) for t in trials]
            self._plot._trial_ids = trial_ids
            self._plot._n_trials = len(trial_ids)
            self._plot._sort_order = list(range(len(trial_ids)))

            self._replot()
        finally:
            self._loading_overlay.hide()

    def _replot(self):
        if not self._perievent or not self._current_trials:
            return
        post_s = self._post_spin.value()
        condition_group, condition_labels = self._get_condition_groups()
        order = sort_trials(self._perievent, self._sort_combo.currentText(), post_s, condition_group)
        self._plot.set_data(
            self._perievent,
            order,
            self._pre_spin.value(),
            post_s,
            self._bin_spin.value(),
            condition_group=condition_group,
            condition_labels=condition_labels,
        )
        n = len(self._current_trials)
        n_with_spikes = sum(1 for v in self._perievent.values() if len(v) > 0)
        self._status.setText(
            f"Cluster #{self._current_cluster_id} · {n} trials "
            f"({n_with_spikes} with spikes in window) · "
            "click row to select · scroll Y to zoom"
        )

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------

    def _on_spike_width_changed(self, w: float):
        self._plot._spike_width = w
        self._replot()

    def _on_trial_hovered(self, trial_idx: int):
        if trial_idx < 0 or trial_idx >= len(self._current_trials):
            return
        self._sel_label.setText(f"Hover: trial {self._current_trials[trial_idx]}")

    def _on_hover_info(self, trial_idx: int, rel_time: float):
        if trial_idx < 0 or trial_idx >= len(self._current_trials):
            return
        trial_id = self._current_trials[trial_idx]
        sign = "+" if rel_time >= 0 else ""
        self._sel_label.setText(f"Trial {trial_id}  ·  t = {sign}{rel_time:.3f} s  (double-click to navigate)")

    def _on_trial_time_requested(self, trial_idx: int, rel_time: float):
        """Double-click: navigate to trial and seek to the clicked time point."""
        if trial_idx < 0 or trial_idx >= len(self._current_trials):
            return
        trial_id = self._current_trials[trial_idx]

        # Navigate to trial
        if self.navigation_widget is not None:
            self.app_state._preserve_x_range_next = True
            self.navigation_widget.navigate_to_trial(trial_id)

        # Convert rel_time (relative to event) → trial-relative time
        trial_start_abs = self._start_map.get(trial_id, 0.0)
        ref_abs = self._ref_times_abs_map.get(trial_id)
        if ref_abs is None:
            return
        trial_relative_time = (ref_abs - trial_start_abs) + rel_time

        video = getattr(self.app_state, "video", None)
        if video is not None:
            display_t = self.app_state.to_display(trial_id, trial_relative_time)
            video.seek_to_frame(video.time_to_frame(display_t))

        sign = "+" if rel_time >= 0 else ""
        self._status.setText(
            f"Navigated → trial {trial_id}  ·  t = {sign}{rel_time:.3f} s from event  "
            f"(trial-relative: {trial_relative_time:.3f} s)"
        )

    def _on_trial_selected(self, trial_idx: int):
        if trial_idx >= len(self._current_trials):
            return
        trial_id = self._current_trials[trial_idx]
        self._sel_label.setText(f"Selected: {trial_id}")
        self._jump_spin.blockSignals(True)
        try:
            self._jump_spin.setValue(int(trial_id))
        except (ValueError, TypeError):
            self._jump_spin.setValue(self._plot._selected)
        self._jump_spin.blockSignals(False)
        self._status.setText(f"Selected {trial_id} — press 'Navigate to trial →' to seek main timeline")

    def _on_jump_clicked(self):
        target = str(self._jump_spin.value())
        for display_row, peri_idx in enumerate(self._plot._sort_order):
            if peri_idx < len(self._current_trials) and str(self._current_trials[peri_idx]) == target:
                self._plot.select_row(display_row)
                return

    def _on_navigate_clicked(self):
        if self._plot._selected < 0 or not self._current_trials:
            self._status.setText("Select a trial first (click a row or use Jump).")
            return
        trial_idx = self._plot._sort_order[self._plot._selected]
        if trial_idx >= len(self._current_trials):
            return
        trial_id = str(self._current_trials[trial_idx])
        if self.navigation_widget is not None:
            self.app_state._preserve_x_range_next = True
            self.navigation_widget.navigate_to_trial(trial_id)
        self._status.setText(f"Navigated → {trial_id}")

    # ------------------------------------------------------------------
    # Public: update after data reload
    # ------------------------------------------------------------------
