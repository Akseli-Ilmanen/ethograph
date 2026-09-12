"""Combined plot settings widget with LinePlot / Spectrogram / HeatMap tabs."""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .app_state import AppStateSpec

HEATMAP_COLORMAPS = [
    "RdBu_r",
    "viridis",
    "inferno",
    "coolwarm",
    "plasma",
    "magma",
    "cividis",
]

_NORM_DISPLAY_TO_KEY = {
    "No normalization": "none",
    "Per-channel z-normalization": "per_channel",
    "Global z-normalization": "global",
}
_NORM_KEY_TO_DISPLAY = {v: k for k, v in _NORM_DISPLAY_TO_KEY.items()}


class PlotSettingsWidget(QWidget):
    """Combined plot settings with toggle-button tabs: LinePlot | SpacePlot | Spectrogram | HeatMap."""

    def __init__(self, shell, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.shell = shell
        self.plot_container = None
        self.meta_widget = None
        self._needs_auto_levels = True
        #: Source (spectrogram buffer) identity last auto-leveled. Mirrors
        #: audian's ``BufferedSpectrogram.init`` flag: the noise-level estimate
        #: is computed once per opened source, not on every pan/zoom buffer
        #: refresh — a source with a few loud transients amid a quiet baseline
        #: (real animal-sound dynamic range) would otherwise flicker between
        #: whatever transient happens to be in the current buffer window.
        self._auto_leveled_identity: str | None = None

        self.setAttribute(Qt.WA_AlwaysShowToolTips)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_toggle_buttons(main_layout)
        self._create_lineplot_panel(main_layout)
        self._create_spaceplot_panel(main_layout)
        self._create_radialplot_panel(main_layout)
        self._create_spectrogram_panel(main_layout)
        self._create_heatmap_panel(main_layout)
        self._create_audio_channel_group(main_layout)
        self._create_neo_controls_group(main_layout)
        self._create_shared_controls(main_layout)

        self._restore_lineplot_defaults()
        self._restore_spaceplot_defaults()
        self._restore_spectrogram_defaults()
        self._restore_heatmap_defaults()

        self._show_panel("lineplot")

    # ------------------------------------------------------------------
    # Toggle buttons
    # ------------------------------------------------------------------

    def _create_toggle_buttons(self, main_layout):
        toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_widget.setLayout(toggle_layout)

        toggle_defs = [
            ("lineplot_toggle", "LinePlot", self._toggle_lineplot),
            ("spaceplot_toggle", "SpacePlot", self._toggle_spaceplot),
            ("spectrogram_toggle", "Spectrogram", self._toggle_spectrogram),
            ("heatmap_toggle", "HeatMap", self._toggle_heatmap),
        ]
        for attr, label, callback in toggle_defs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(callback)
            toggle_layout.addWidget(btn)
            setattr(self, attr, btn)

        main_layout.addWidget(toggle_widget)

    def _show_panel(self, panel_name: str):
        panels = {
            "lineplot": (self.lineplot_panel, self.lineplot_toggle),
            "spaceplot": (self.spaceplot_panel, self.spaceplot_toggle),
            "spectrogram": (self.spectrogram_panel, self.spectrogram_toggle),
            "heatmap": (self.heatmap_panel, self.heatmap_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)
        self._refresh_layout()

    def _toggle_lineplot(self):
        self._show_panel("lineplot" if self.lineplot_toggle.isChecked() else "spaceplot")

    def _toggle_spaceplot(self):
        self._show_panel("spaceplot" if self.spaceplot_toggle.isChecked() else "lineplot")

    def _toggle_spectrogram(self):
        self._show_panel("spectrogram" if self.spectrogram_toggle.isChecked() else "lineplot")

    def _toggle_heatmap(self):
        self._show_panel("heatmap" if self.heatmap_toggle.isChecked() else "lineplot")

    def _refresh_layout(self):
        if self.meta_widget:
            self.meta_widget.refresh_widget_layout(self)

    # ------------------------------------------------------------------
    # LinePlot panel
    # ------------------------------------------------------------------

    def _create_lineplot_panel(self, main_layout):
        self.lineplot_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.lineplot_panel.setLayout(layout)

        group_box = QGroupBox("Axes Controls")
        group_layout = QGridLayout()
        group_layout.setSpacing(2)
        group_layout.setContentsMargins(2, 2, 2, 2)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        self.ymin_edit = QLineEdit()
        self.ymax_edit = QLineEdit()

        self.percentile_ylim_edit = QLineEdit()
        validator = QDoubleValidator(95.0, 100, 2, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.percentile_ylim_edit.setValidator(validator)

        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")

        row = 0
        group_layout.addWidget(QLabel("Y min:"), row, 0)
        group_layout.addWidget(self.ymin_edit, row, 1)
        group_layout.addWidget(QLabel("Y max:"), row, 2)
        group_layout.addWidget(self.ymax_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("Percentile Y-lim:"), row, 0)
        group_layout.addWidget(self.percentile_ylim_edit, row, 1)
        group_layout.addWidget(self.apply_button, row, 2)
        group_layout.addWidget(self.reset_button, row, 3)

        row += 1
        self.hline_value_edit = QLineEdit()
        self.hline_value_edit.setPlaceholderText("y value")
        self.hline_value_edit.setToolTip("Y value for a new horizontal reference line on the active line plot")
        self.add_hline_button = QPushButton("Add h-line")
        self.add_hline_button.setToolTip("Draw a horizontal line at this value; it stays across trial changes")
        self.clear_hlines_button = QPushButton("Clear h-lines")
        self.clear_hlines_button.setToolTip("Remove every horizontal line from the active line plot")

        group_layout.addWidget(QLabel("Horizontal line:"), row, 0)
        group_layout.addWidget(self.hline_value_edit, row, 1)
        group_layout.addWidget(self.add_hline_button, row, 2)
        group_layout.addWidget(self.clear_hlines_button, row, 3)

        row += 1
        self.hline_list_label = QLabel("")
        self.hline_list_label.setWordWrap(True)
        group_layout.addWidget(self.hline_list_label, row, 0, 1, 4)

        self.hline_value_edit.returnPressed.connect(self._on_add_hline)
        self.add_hline_button.clicked.connect(self._on_add_hline)
        self.clear_hlines_button.clicked.connect(self._on_clear_hlines)

        self.ymin_edit.editingFinished.connect(self._on_axes_edited)
        self.ymax_edit.editingFinished.connect(self._on_axes_edited)
        self.percentile_ylim_edit.editingFinished.connect(self._on_axes_edited)

        self.apply_button.clicked.connect(self._on_axes_edited)
        self.reset_button.clicked.connect(self._reset_axes_to_defaults)

        # Live-apply: values apply on editingFinished, so the explicit Apply /
        # Reset buttons are redundant and hidden (kept alive for wiring).
        self.apply_button.setVisible(False)
        self.reset_button.setVisible(False)

        main_layout.addWidget(self.lineplot_panel)

    def _restore_lineplot_defaults(self):
        for attr, edit in [
            ("ymin", self.ymin_edit),
            ("ymax", self.ymax_edit),
            ("percentile_ylim", self.percentile_ylim_edit),
        ]:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = self.app_state.get_with_default(attr)
                setattr(self.app_state, attr, value)
            edit.setText("" if value is None else str(value))

        lock_axes = self.app_state.get_with_default("lock_axes")
        self.lock_axes_checkbox.setChecked(lock_axes)

        self._sync_hlines_to_active_plot()

    def _autoscale_y_toggle(self, checked: bool):
        if not self.plot_container:
            return

        target = self.plot_container.get_hovered_plot()
        if target is None:
            target = self.plot_container.get_current_plot()

        if checked:
            target.vb.enableAutoRange(x=False, y=True)
            target._apply_y_constraints()
            self.lock_axes_checkbox.setChecked(False)
        else:
            target.vb.disableAutoRange()
            target._apply_y_constraints()

    def _on_lock_axes_toggled(self, checked: bool):
        self.app_state.lock_axes = checked
        if self.plot_container:
            self.plot_container.toggle_axes_lock()
        if checked:
            self.autoscale_checkbox.setChecked(False)

    def sync_axes_to_active_plot(self):
        """Populate the Y min/max/percentile fields from the active plot's own
        per-panel state (every line plot is independent)."""
        plot = getattr(self.plot_container, "active_feature_plot", None) if self.plot_container else None
        ps = getattr(plot, "panel_state", {}) if plot is not None else {}
        ymin, ymax = ps.get("ymin"), ps.get("ymax")
        pct = ps.get("percentile")
        if pct is None:
            pct = self.app_state.get_with_default("percentile_ylim")

        def _fmt(v):
            return "" if v is None else str(v)

        for edit, val in (
            (self.ymin_edit, ymin),
            (self.ymax_edit, ymax),
            (self.percentile_ylim_edit, pct),
        ):
            edit.blockSignals(True)
            edit.setText(_fmt(val))
            edit.blockSignals(False)

        self._sync_hlines_to_active_plot()

    # ------------------------------------------------------------------
    # Horizontal reference lines (per line-plot panel, session-lived)
    # ------------------------------------------------------------------

    def _active_line_plot(self):
        """The active feature panel when it is a line plot (the heatmap has no
        h-lines), else ``None``."""
        pc = self.plot_container
        if pc is None:
            return None
        plot = getattr(pc, "active_feature_plot", None)
        return plot if plot in pc.line_plots else None

    def _on_add_hline(self):
        plot = self._active_line_plot()
        value = self._parse_float(self.hline_value_edit.text())
        if plot is None or value is None:
            return
        plot.add_hline(value)
        self.hline_value_edit.clear()
        self._sync_hlines_to_active_plot()

    def _on_clear_hlines(self):
        plot = self._active_line_plot()
        if plot is None:
            return
        plot.clear_hlines()
        self._sync_hlines_to_active_plot()

    def _sync_hlines_to_active_plot(self):
        """Mirror the active line plot's h-lines into the sidebar."""
        plot = self._active_line_plot()
        values = plot.hline_values() if plot is not None else []
        joined = ", ".join(f"{v:g}" for v in values)
        self.hline_list_label.setText(f"Lines at: {joined}" if values else "")
        self.clear_hlines_button.setEnabled(bool(values))

    def _on_axes_edited(self):
        if not self.plot_container:
            return

        ymin = self._parse_float(self.ymin_edit.text())
        ymax = self._parse_float(self.ymax_edit.text())
        pct = self._parse_float(self.percentile_ylim_edit.text())
        if pct is None:
            pct = self.app_state.get_with_default("percentile_ylim")
        user_set_yrange = ymin is not None or ymax is not None

        if self.plot_container.is_spectrogram():
            # Spectrogram y-limits are global (there is only one spectrogram).
            for attr, val in (("ymin", ymin), ("ymax", ymax), ("percentile_ylim", pct)):
                setattr(self.app_state, attr, val)
        else:
            # Every line plot has its own y-viewbox: store axes on the active plot
            # and apply only to it (no canonical vs. extra split).
            active = getattr(self.plot_container, "active_feature_plot", None)
            if active is not None and hasattr(active, "panel_state"):
                active.panel_state["ymin"] = ymin
                active.panel_state["ymax"] = ymax
                active.panel_state["percentile"] = pct
                if not self.autoscale_checkbox.isChecked():
                    if user_set_yrange:
                        if hasattr(active, "vb"):
                            active.vb.setLimits(yMin=None, yMax=None, minYRange=None, maxYRange=None)
                        active.apply_y_range(ymin, ymax)
                    elif hasattr(active, "_apply_y_constraints"):
                        active._apply_y_constraints()

        new_xmin, new_xmax = self._calculate_new_window_size()
        if new_xmin is not None and new_xmax is not None:
            self.plot_container.set_x_range(mode="preserve", curr_xlim=(new_xmin, new_xmax))

    def _calculate_new_window_size(self):
        if not self.plot_container:
            return None, None
        if not hasattr(self.app_state, "ds") or self.app_state.ds is None:
            return None, None
        video = getattr(self.app_state, "video", None)
        current_time = (
            video.frame_to_time(self.app_state.current_frame)
            if video
            else self.app_state.current_frame / self.app_state.video_fps
        )
        before = self.app_state.before_s
        after = self.app_state.after_s
        half_window = (before + after) / 2
        return current_time - half_window, current_time + half_window

    def _reset_axes_to_defaults(self):
        for attr, edit in [
            ("ymin", self.ymin_edit),
            ("ymax", self.ymax_edit),
            ("percentile_ylim", self.percentile_ylim_edit),
        ]:
            value = self.app_state.get_with_default(attr)
            edit.setText("" if value is None else str(value))
            setattr(self.app_state, attr, value)

        self.lock_axes_checkbox.setChecked(False)
        self.app_state.lock_axes = False
        self._on_axes_edited()

    # ------------------------------------------------------------------
    # SpacePlot panel
    # ------------------------------------------------------------------

    def _create_radialplot_panel(self, main_layout):
        """Host for the active radial plot's own controls.

        Deliberately empty: everything a radial plot shows (feature, which
        value points up) is per instance, so the instance's ``controls_widget``
        is inserted here by ``DataWidget.add_radial_plot`` — there are no
        global radial settings to put alongside it.
        """
        self.radialplot_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.radialplot_panel.setLayout(layout)

        group_box = QGroupBox("Radial Plot Controls")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        main_layout.addWidget(self.radialplot_panel)
        self.radialplot_panel.hide()

    def _create_spaceplot_panel(self, main_layout):
        self.spaceplot_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spaceplot_panel.setLayout(layout)

        group_box = QGroupBox("Space Plot Controls")
        group_layout = QGridLayout()
        group_layout.setSpacing(2)
        group_layout.setContentsMargins(2, 2, 2, 2)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        row = 0
        group_layout.addWidget(QLabel("Percentile XYZ-lim:"), row, 0)
        self.space_percentile_spin = QDoubleSpinBox()
        self.space_percentile_spin.setRange(50.0, 100.0)
        self.space_percentile_spin.setSingleStep(0.5)
        self.space_percentile_spin.setDecimals(1)
        self.space_percentile_spin.setToolTip("Percentile for per-axis range limits (100 = show all data)")
        self.space_percentile_spin.valueChanged.connect(self._on_space_percentile_changed)
        group_layout.addWidget(self.space_percentile_spin, row, 1)

        self.space_marker_checkbox = QCheckBox("Marker")
        self.space_marker_checkbox.toggled.connect(self._on_space_marker_toggled)
        group_layout.addWidget(self.space_marker_checkbox, row, 2)

        self.space_limit_window_checkbox = QCheckBox("Limit to window")
        self.space_limit_window_checkbox.setToolTip("Only draw trajectory for the time range visible in the line plot")
        self.space_limit_window_checkbox.toggled.connect(self._on_space_limit_window_toggled)
        group_layout.addWidget(self.space_limit_window_checkbox, row, 3)

        row += 1
        self.space_lock_axes_checkbox = QCheckBox("Lock axes (Space)")
        self.space_lock_axes_checkbox.setToolTip("Keep the current axis ranges when switching trials")
        self.space_lock_axes_checkbox.toggled.connect(self._on_space_lock_axes_toggled)
        group_layout.addWidget(self.space_lock_axes_checkbox, row, 0, 1, 2)

        self.space_hide_zeros_checkbox = QCheckBox("Hide zeros")
        self.space_hide_zeros_checkbox.setToolTip("Hide points where all dimensions are exactly zero")
        self.space_hide_zeros_checkbox.toggled.connect(self._on_space_hide_zeros_toggled)
        group_layout.addWidget(self.space_hide_zeros_checkbox, row, 2, 1, 2)

        row += 1
        self.space_sync_views_checkbox = QCheckBox("Sync views across space plots")
        self.space_sync_views_checkbox.setToolTip(
            "Mirror zoom/pan (2D) and camera angle (3D) across all open space plots of the same type"
        )
        self.space_sync_views_checkbox.toggled.connect(self._on_space_sync_views_toggled)
        group_layout.addWidget(self.space_sync_views_checkbox, row, 0, 1, 4)

        row += 1
        self.space_show_references_checkbox = QCheckBox("Show reference geometry")
        self.space_show_references_checkbox.setToolTip("Draw the selected library reference geometry")
        self.space_show_references_checkbox.toggled.connect(self._on_space_show_references_toggled)
        group_layout.addWidget(self.space_show_references_checkbox, row, 0, 1, 4)

        row += 1
        group_layout.addWidget(QLabel("Library geometry:"), row, 0)
        self.space_library_combo = QComboBox()
        self.space_library_combo.setToolTip(
            "Reference geometry drawn behind the trajectory — one entry per "
            "YAML file in the geometry library (~/.ethograph/defaults/config/space/*.yaml)"
        )
        self.space_library_combo.currentTextChanged.connect(self._on_space_library_changed)
        # Re-sync when set externally (e.g. a template's library_geometry default)
        self.app_state.space_library_geometry_changed.connect(lambda *_: self._populate_space_library_combo())
        group_layout.addWidget(self.space_library_combo, row, 1, 1, 3)

        main_layout.addWidget(self.spaceplot_panel)

    def _restore_spaceplot_defaults(self):
        self.space_percentile_spin.setValue(self.app_state.get_with_default("space_percentile_xyzlim"))
        self.space_marker_checkbox.setChecked(self.app_state.get_with_default("space_marker_visible"))

        self.space_limit_window_checkbox.setChecked(self.app_state.get_with_default("space_limit_to_window"))

        self.space_lock_axes_checkbox.setChecked(self.app_state.get_with_default("space_lock_axes"))

        self.space_hide_zeros_checkbox.setChecked(self.app_state.get_with_default("space_hide_zeros"))

        self.space_sync_views_checkbox.setChecked(self.app_state.get_with_default("space_sync_views"))

        self.space_show_references_checkbox.setChecked(self.app_state.get_with_default("space_show_references"))

        self._populate_space_library_combo()

    def _populate_space_library_combo(self):
        """Re-scan the global geometry library and restore the saved selection."""
        from ethograph.gui.plots_space import load_library_geometries

        combo = self.space_library_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None")
        combo.addItems(sorted(load_library_geometries()))
        saved = self.app_state.get_with_default("space_library_geometry")
        if saved:
            idx = combo.findText(saved)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _on_space_percentile_changed(self, value: float):
        self.app_state.space_percentile_xyzlim = value

    def _on_space_marker_toggled(self, checked: bool):
        self.app_state.space_marker_visible = checked

    def _on_space_limit_window_toggled(self, checked: bool):
        self.app_state.space_limit_to_window = checked

    def _on_space_lock_axes_toggled(self, checked: bool):
        self.app_state.space_lock_axes = checked

    def _on_space_hide_zeros_toggled(self, checked: bool):
        self.app_state.space_hide_zeros = checked

    def _on_space_sync_views_toggled(self, checked: bool):
        self.app_state.space_sync_views = checked

    def _on_space_show_references_toggled(self, checked: bool):
        self.app_state.space_show_references = checked

    def _on_space_library_changed(self, text: str):
        self.app_state.space_library_geometry = text if text and text != "None" else None

    # ------------------------------------------------------------------
    # Shared controls (apply to all plot types)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Audio channel group (context sidebar for audio trace / spectrogram)
    # ------------------------------------------------------------------

    def _create_audio_channel_group(self, main_layout):
        """Per-panel channel selector shown in the audiotrace / spectrogram
        sidebar contexts. Edits the *active* audio panel's pinned channel."""
        self._active_audio_plot = None
        self.audio_channel_group = QGroupBox("Audio source")
        layout = QHBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.audio_channel_group.setLayout(layout)
        layout.addWidget(QLabel("Channel:"))
        self.audio_channel_combo = QComboBox()
        self.audio_channel_combo.setObjectName("audio_channel_combo")
        self.audio_channel_combo.currentIndexChanged.connect(self._on_audio_channel_changed)
        layout.addWidget(self.audio_channel_combo, 1)
        main_layout.addWidget(self.audio_channel_group)

    def set_active_audio_plot(self, plot):
        """Point the Channel combo at *plot* (audio trace or spectrogram
        panel) and mirror its pinned mic/channel."""
        self._active_audio_plot = plot
        self._sync_audio_channel_combo()

    def _channel_keys_for_plot(self, plot) -> list[str]:
        """All ``audio_source_map`` keys for the file the panel is showing
        (its pinned key, or the global mic selection when unpinned)."""
        state = self.app_state
        key = plot.mic_name or getattr(state, "mics_sel", None)
        groups = getattr(state, "audio_mic_channels", None) or {}
        for keys in groups.values():
            if key in keys:
                return list(keys)
        source_map = getattr(state, "audio_source_map", None) or {}
        if key in source_map:
            return [key]
        return []

    def _sync_audio_channel_combo(self):
        combo = self.audio_channel_combo
        plot = self._active_audio_plot
        keys = self._channel_keys_for_plot(plot) if plot is not None else []
        source_map = getattr(self.app_state, "audio_source_map", None) or {}
        combo.blockSignals(True)
        combo.clear()
        for key in keys:
            _, ch = source_map.get(key, (key, 0))
            combo.addItem(f"Channel {ch + 1}", key)
        current = plot.mic_name if plot is not None else None
        if current is None:
            current = getattr(self.app_state, "mics_sel", None)
        if current in keys:
            combo.setCurrentIndex(keys.index(current))
        combo.blockSignals(False)
        combo.setEnabled(len(keys) > 1)

    def _on_audio_channel_changed(self, index):
        plot = self._active_audio_plot
        pc = self.plot_container
        if plot is None or pc is None or index < 0:
            return
        if plot not in pc.audio_trace_plots and plot not in pc.spectrogram_plots:
            return
        key = self.audio_channel_combo.itemData(index)
        if not key or key == plot.mic_name:
            return
        pc.set_audio_panel_mic(plot, key)
        # Re-pinning the active panel's channel also redirects playback to it.
        self.app_state.playback_mic_key = key

    # ------------------------------------------------------------------
    # Neo trace controls (context sidebar for the Neo viewer)
    # ------------------------------------------------------------------

    def _create_neo_controls_group(self, main_layout):
        """Per-panel gain + channel-spacing controls shown in the neo sidebar
        context. Edits the *active* Neo trace panel (each is one modality)."""
        self._active_neo_plot = None
        self.neo_controls_group = QGroupBox("Neo trace controls")
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.neo_controls_group.setLayout(layout)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Gain:"))
        self.neo_gain_spin = QDoubleSpinBox()
        self.neo_gain_spin.setRange(-100.0, 100.0)
        self.neo_gain_spin.setSingleStep(0.1)
        self.neo_gain_spin.setDecimals(1)
        self.neo_gain_spin.setToolTip("Display gain: negative = amplify, positive = attenuate (Ctrl+Wheel on the plot)")
        self.neo_gain_spin.valueChanged.connect(self._on_neo_gain_changed)
        gain_row.addWidget(self.neo_gain_spin)
        self.neo_auto_gain_cb = QCheckBox("Auto gain")
        self.neo_auto_gain_cb.setChecked(True)
        self.neo_auto_gain_cb.setToolTip("Quantile-based auto-scaling (Phy method)")
        self.neo_auto_gain_cb.toggled.connect(self._on_neo_auto_gain_toggled)
        gain_row.addWidget(self.neo_auto_gain_cb)
        gain_row.addStretch()
        layout.addLayout(gain_row)

        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Channel spacing:"))
        self.neo_spacing_spin = QDoubleSpinBox()
        self.neo_spacing_spin.setRange(0.1, 1000.0)
        self.neo_spacing_spin.setSingleStep(0.5)
        self.neo_spacing_spin.setDecimals(2)
        self.neo_spacing_spin.setToolTip("Vertical offset between consecutive channels")
        self.neo_spacing_spin.valueChanged.connect(self._on_neo_spacing_changed)
        spacing_row.addWidget(self.neo_spacing_spin)
        self.neo_auto_spacing_btn = QPushButton("Auto")
        self.neo_auto_spacing_btn.setToolTip("Recompute spacing to fit the channels")
        self.neo_auto_spacing_btn.clicked.connect(self._on_neo_auto_spacing_clicked)
        spacing_row.addWidget(self.neo_auto_spacing_btn)
        spacing_row.addStretch()
        layout.addLayout(spacing_row)

        main_layout.addWidget(self.neo_controls_group)

    def set_active_neo_plot(self, plot):
        """Point the Neo trace controls at *plot* and mirror its gain/spacing."""
        self._active_neo_plot = plot
        self._sync_neo_controls()

    def _sync_neo_controls(self):
        plot = self._active_neo_plot
        if plot is None:
            return
        self.neo_gain_spin.blockSignals(True)
        self.neo_spacing_spin.blockSignals(True)
        self.neo_gain_spin.setValue(float(getattr(plot.buffer, "display_gain", 0.0)))
        self.neo_spacing_spin.setValue(float(getattr(plot.buffer, "channel_spacing", 3.0)))
        self.neo_gain_spin.blockSignals(False)
        self.neo_spacing_spin.blockSignals(False)

    def _neo_plot_active(self):
        plot = self._active_neo_plot
        pc = self.plot_container
        if plot is None or pc is None or plot not in pc.neo_trace_plots:
            return None
        return plot

    def _rerender_neo(self, plot):
        pc = self.plot_container
        xmin, xmax = pc.get_current_xlim()
        plot.update_plot_content(xmin, xmax)

    def _on_neo_gain_changed(self, value: float):
        plot = self._neo_plot_active()
        if plot is None:
            return
        if self.neo_auto_gain_cb.isChecked():
            self.neo_auto_gain_cb.blockSignals(True)
            self.neo_auto_gain_cb.setChecked(False)
            self.neo_auto_gain_cb.blockSignals(False)
        plot.buffer.display_gain = value
        self._rerender_neo(plot)

    def _on_neo_auto_gain_toggled(self, checked: bool):
        plot = self._neo_plot_active()
        if plot is None or not checked:
            return
        new_gain = plot.auto_gain()
        self.neo_gain_spin.blockSignals(True)
        self.neo_gain_spin.setValue(new_gain)
        self.neo_gain_spin.blockSignals(False)
        self._rerender_neo(plot)

    def _on_neo_spacing_changed(self, value: float):
        plot = self._neo_plot_active()
        if plot is None:
            return
        plot.buffer.channel_spacing = value
        plot._setup_global_y_space()
        self._rerender_neo(plot)
        plot.autoscale()

    def _on_neo_auto_spacing_clicked(self):
        plot = self._neo_plot_active()
        if plot is None:
            return
        plot.auto_channel_spacing()
        self._sync_neo_controls()
        self._rerender_neo(plot)
        plot.autoscale()

    def _create_shared_controls(self, main_layout):
        shared_widget = QWidget()
        self.shared_widget = shared_widget  # exposed for the context-sensitive sidebar
        shared_layout = QHBoxLayout()
        shared_layout.setSpacing(6)
        shared_layout.setContentsMargins(0, 0, 0, 0)
        shared_widget.setLayout(shared_layout)

        self.autoscale_checkbox = QCheckBox("Autoscale Y")
        self.lock_axes_checkbox = QCheckBox("Lock Axes")

        shared_layout.addWidget(self.autoscale_checkbox)
        shared_layout.addWidget(self.lock_axes_checkbox)
        shared_layout.addStretch()

        self.autoscale_checkbox.toggled.connect(self._autoscale_y_toggle)
        self.lock_axes_checkbox.toggled.connect(self._on_lock_axes_toggled)

        main_layout.addWidget(shared_widget)

    # ------------------------------------------------------------------
    # Spectrogram panel
    # ------------------------------------------------------------------

    def _create_spectrogram_panel(self, main_layout):
        self.spectrogram_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spectrogram_panel.setLayout(layout)

        group_box = QGroupBox("Spectrogram Controls")
        group_layout = QGridLayout()
        group_layout.setSpacing(2)
        group_layout.setContentsMargins(2, 2, 2, 2)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        self.spec_ymin_edit = QLineEdit()
        self.spec_ymax_edit = QLineEdit()
        self.vmin_db_edit = QLineEdit()
        self.vmax_db_edit = QLineEdit()
        self.nfft_edit = QLineEdit()
        self.hop_frac_edit = QLineEdit()

        self.colormap_combo = QComboBox()
        self.colormap_display = {
            "CET-R4": "jet",
            "CET-L8": "blue-pink-yellow",
            "CET-L16": "black-blue-green-white",
            "CET-CBL2": "black-blue-yellow-white",
            "CET-L1": "black-white",
            "CET-L3": "inferno",
        }
        self.colormaps = list(self.colormap_display.keys())
        self.colormap_combo.addItems(self.colormap_display.values())

        self.levels_mode_combo = QComboBox()
        self.levels_mode_combo.addItems(["Always auto dB levels", "Remember dB levels"])

        self.auto_levels_button = QPushButton("Auto dB levels")
        self.spec_apply_button = QPushButton("Apply settings")

        row = 0
        group_layout.addWidget(QLabel("Freq min (kHz):"), row, 0)
        group_layout.addWidget(self.spec_ymin_edit, row, 1)
        group_layout.addWidget(QLabel("Freq max (kHz):"), row, 2)
        group_layout.addWidget(self.spec_ymax_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("NFFT:"), row, 0)
        group_layout.addWidget(self.nfft_edit, row, 1)
        group_layout.addWidget(QLabel("Hop fraction:"), row, 2)
        group_layout.addWidget(self.hop_frac_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("dB min:"), row, 0)
        group_layout.addWidget(self.vmin_db_edit, row, 1)
        group_layout.addWidget(QLabel("dB max:"), row, 2)
        group_layout.addWidget(self.vmax_db_edit, row, 3)

        row += 1
        group_layout.addWidget(QLabel("Colormap:"), row, 0)
        group_layout.addWidget(self.colormap_combo, row, 1)
        group_layout.addWidget(QLabel("Levels:"), row, 2)
        group_layout.addWidget(self.levels_mode_combo, row, 3)

        row += 1
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.auto_levels_button)
        button_layout.addWidget(self.spec_apply_button)
        button_widget.setLayout(button_layout)
        group_layout.addWidget(button_widget, row, 0, 1, 4)

        self.spec_ymin_edit.editingFinished.connect(self._on_spec_edited)
        self.spec_ymax_edit.editingFinished.connect(self._on_spec_edited)
        self.vmin_db_edit.editingFinished.connect(self._on_spec_edited)
        self.vmax_db_edit.editingFinished.connect(self._on_spec_edited)
        self.nfft_edit.editingFinished.connect(self._on_spec_edited)
        self.hop_frac_edit.editingFinished.connect(self._on_spec_edited)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.levels_mode_combo.currentIndexChanged.connect(self._on_levels_mode_changed)
        self.auto_levels_button.clicked.connect(self._auto_levels)
        self.spec_apply_button.clicked.connect(self._on_spec_edited)

        main_layout.addWidget(self.spectrogram_panel)

    def _restore_spectrogram_defaults(self):
        for attr, edit in [
            ("vmin_db", self.vmin_db_edit),
            ("vmax_db", self.vmax_db_edit),
            ("nfft", self.nfft_edit),
            ("hop_frac", self.hop_frac_edit),
        ]:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = self.app_state.get_with_default(attr)
                setattr(self.app_state, attr, value)
            edit.setText("" if value is None else str(value))

        default_vmin = AppStateSpec.get_default("vmin_db")
        default_vmax = AppStateSpec.get_default("vmax_db")
        if (
            getattr(self.app_state, "vmin_db", default_vmin) != default_vmin
            or getattr(self.app_state, "vmax_db", default_vmax) != default_vmax
        ):
            self._needs_auto_levels = False

        for attr, edit in [
            ("spec_ymin", self.spec_ymin_edit),
            ("spec_ymax", self.spec_ymax_edit),
        ]:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = self.app_state.get_with_default(attr)
                setattr(self.app_state, attr, value)
            display_val = value / 1000 if value is not None else None
            edit.setText("" if display_val is None else str(display_val))

        colormap = self.app_state.get_with_default("spec_colormap")
        if colormap in self.colormap_display:
            self.colormap_combo.setCurrentText(self.colormap_display[colormap])

        levels_mode = getattr(self.app_state, "spec_levels_mode", None)
        if levels_mode is None:
            levels_mode = self.app_state.get_with_default("spec_levels_mode")
            self.app_state.spec_levels_mode = levels_mode
        self.levels_mode_combo.setCurrentIndex(0 if levels_mode == "auto" else 1)
        if levels_mode == "remember":
            self._needs_auto_levels = False

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container
        plot_container.plot_changed.connect(self._on_plot_changed)
        plot_container.spectrogram_overlay_shown.connect(self._on_overlay_shown)
        # Relays bufferUpdated from every spectrogram instance (they are
        # dynamic panels — duplicates allowed).
        plot_container.spectrogram_buffer_updated.connect(self._on_buffer_updated)

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def set_enabled_state(self):
        self.setEnabled(True)

    def _is_auto_levels_mode(self) -> bool:
        return getattr(self.app_state, "spec_levels_mode", "auto") == "auto"

    def _on_plot_changed(self, plot_type: str):
        if plot_type == "spectrogram" and self._needs_auto_levels:
            QTimer.singleShot(500, self._try_initial_auto_levels)

    def _try_initial_auto_levels(self):
        if not self._needs_auto_levels:
            return
        if not self.plot_container or not self.plot_container.is_spectrogram():
            return
        current_plot = self.plot_container.get_current_plot()
        if not hasattr(current_plot, "buffer") or current_plot.buffer.Sxx_db is None:
            return
        self._needs_auto_levels = False
        self._auto_levels()

    def _on_buffer_updated(self):
        if not self._is_auto_levels_mode():
            return
        spec_plot = next(
            (p for p in self.plot_container.spectrogram_plots if p.isVisible() and p.buffer.Sxx_db is not None),
            None,
        )
        if spec_plot is None:
            return
        # One estimate per opened source (mirrors audian's ``init`` flag):
        # a source with a quiet baseline and a few loud transients would
        # otherwise re-level on every pan/zoom to whatever transient the
        # current buffer window happens to contain.
        if spec_plot.buffer.current_identity == self._auto_leveled_identity:
            return
        self._auto_levels()

    def _on_overlay_shown(self):
        colormap_name = self.app_state.get_with_default("spec_colormap")
        self.plot_container.apply_overlay_colormap(colormap_name)
        if self._is_auto_levels_mode():
            QTimer.singleShot(200, self._auto_levels)
        else:
            self._apply_remembered_levels()

    def _on_levels_mode_changed(self, index: int):
        mode = "auto" if index == 0 else "remember"
        self.app_state.spec_levels_mode = mode
        if mode == "auto":
            self._auto_levels()
        else:
            self._apply_remembered_levels()

    def _on_colormap_changed(self, display_name: str):
        display_to_internal = {v: k for k, v in self.colormap_display.items()}
        colormap_name = display_to_internal.get(display_name, display_name)
        self.app_state.spec_colormap = colormap_name
        if self.plot_container:
            for spec_plot in self.plot_container.spectrogram_plots:
                spec_plot.update_colormap(colormap_name)
            if self.plot_container.has_spectrogram_overlay():
                self.plot_container.apply_overlay_colormap(colormap_name)

    def _auto_levels(self):
        if not self.plot_container:
            return

        # Levels are computed from the first spectrogram instance holding
        # data and applied to every instance.
        spec_plot = next(
            (p for p in self.plot_container.spectrogram_plots if p.isVisible() and p.buffer.Sxx_db is not None),
            None,
        )
        has_overlay = self.plot_container.has_spectrogram_overlay()

        if spec_plot is None and not has_overlay:
            return

        if spec_plot is None or spec_plot.buffer.Sxx_db is None:
            return

        Sxx_db = spec_plot.buffer.Sxx_db
        if Sxx_db.size == 0:
            return

        nf = max(1, Sxx_db.shape[0] // 16)

        with np.errstate(all="ignore"):
            zmin = np.percentile(Sxx_db[-nf:, :], 95)
            zmax = np.max(Sxx_db)

        if not np.isfinite(zmin) or not np.isfinite(zmax):
            return

        zmax = zmin + 0.95 * (zmax - zmin)

        if zmax - zmin < 20:
            zmax = zmin + 20
        if zmax - zmin > 80:
            zmin = zmax - 80

        zmin = round(zmin, 1)
        zmax = round(zmax, 1)

        self.vmin_db_edit.setText(str(zmin))
        self.vmax_db_edit.setText(str(zmax))
        self._auto_leveled_identity = spec_plot.buffer.current_identity

        self.app_state.vmin_db = zmin
        self.app_state.vmax_db = zmax

        for plot in self.plot_container.spectrogram_plots:
            plot.update_levels(zmin, zmax)

        if has_overlay:
            self.plot_container.apply_overlay_levels(zmin, zmax)

    def _apply_remembered_levels(self):
        vmin = self._parse_float(self.vmin_db_edit.text())
        vmax = self._parse_float(self.vmax_db_edit.text())
        if vmin is None:
            vmin = self.app_state.get_with_default("vmin_db")
        if vmax is None:
            vmax = self.app_state.get_with_default("vmax_db")
        self.app_state.vmin_db = vmin
        self.app_state.vmax_db = vmax
        if self.plot_container:
            for spec_plot in self.plot_container.spectrogram_plots:
                spec_plot.update_levels(vmin, vmax)
            if self.plot_container.has_spectrogram_overlay():
                self.plot_container.apply_overlay_levels(vmin, vmax)

    def _on_spec_edited(self):
        if not self.plot_container:
            return

        float_edits = {
            "vmin_db": self.vmin_db_edit,
            "vmax_db": self.vmax_db_edit,
            "hop_frac": self.hop_frac_edit,
        }

        khz_edits = {
            "spec_ymin": self.spec_ymin_edit,
            "spec_ymax": self.spec_ymax_edit,
        }

        int_edits = {
            "nfft": self.nfft_edit,
        }

        values = {}
        for attr, edit in float_edits.items():
            val = self._parse_float(edit.text())
            if val is None:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            setattr(self.app_state, attr, val)

        for attr, edit in khz_edits.items():
            val = self._parse_float(edit.text())
            if val is not None:
                val = val * 1000
            else:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            setattr(self.app_state, attr, val)

        for attr, edit in int_edits.items():
            val = self._parse_int(edit.text())
            if val is None:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            setattr(self.app_state, attr, val)

        for spec_plot in self.plot_container.spectrogram_plots:
            spec_plot.update_buffer_settings()
            spec_plot.update_levels(values["vmin_db"], values["vmax_db"])
            spec_plot.apply_y_range(values["spec_ymin"], values["spec_ymax"])
            spec_plot.update_plot_content()

        if self.plot_container.has_spectrogram_overlay():
            self.plot_container.apply_overlay_levels(values["vmin_db"], values["vmax_db"])
            for spec_plot in self.plot_container.spectrogram_plots:
                spec_plot.buffer._clear_buffer()
            self.plot_container.update_audio_overlay()

    # ------------------------------------------------------------------
    # HeatMap panel
    # ------------------------------------------------------------------

    def _create_heatmap_panel(self, main_layout):
        self.heatmap_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.heatmap_panel.setLayout(layout)

        hm_group = QGroupBox("Heatmap Display")
        hm_layout = QGridLayout()
        hm_layout.setSpacing(2)
        hm_layout.setContentsMargins(2, 2, 2, 2)
        hm_group.setLayout(hm_layout)
        layout.addWidget(hm_group)

        hm_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.heatmap_colormap_combo = QComboBox()
        self.heatmap_colormap_combo.addItems(HEATMAP_COLORMAPS)
        self.heatmap_colormap_combo.currentTextChanged.connect(self._on_heatmap_colormap_changed)
        hm_layout.addWidget(self.heatmap_colormap_combo, 0, 1)

        hm_layout.addWidget(QLabel("Excl. percentile:"), 0, 2)
        self.heatmap_percentile_spin = QDoubleSpinBox()
        self.heatmap_percentile_spin.setRange(50.0, 100.0)
        self.heatmap_percentile_spin.setSingleStep(1.0)
        self.heatmap_percentile_spin.setDecimals(1)
        self.heatmap_percentile_spin.setToolTip("Percentile of abs(z-scores) for symmetric color range")
        self.heatmap_percentile_spin.valueChanged.connect(self._on_heatmap_percentile_changed)
        hm_layout.addWidget(self.heatmap_percentile_spin, 0, 3)

        hm_layout.addWidget(QLabel("Normalization:"), 1, 0)
        self.heatmap_norm_combo = QComboBox()
        self.heatmap_norm_combo.addItems(list(_NORM_DISPLAY_TO_KEY.keys()))
        self.heatmap_norm_combo.currentTextChanged.connect(self._on_heatmap_normalization_changed)
        hm_layout.addWidget(self.heatmap_norm_combo, 1, 1)

        main_layout.addWidget(self.heatmap_panel)

    def _restore_heatmap_defaults(self):
        cmap = self.app_state.get_with_default("heatmap_colormap")
        if cmap in HEATMAP_COLORMAPS:
            self.heatmap_colormap_combo.setCurrentText(cmap)

        self.heatmap_percentile_spin.setValue(self.app_state.get_with_default("heatmap_exclusion_percentile"))

        norm_key = self.app_state.get_with_default("heatmap_normalization")
        display = _NORM_KEY_TO_DISPLAY.get(norm_key, "Per-channel")
        self.heatmap_norm_combo.setCurrentText(display)

    def _on_heatmap_colormap_changed(self, colormap_name: str):
        self.app_state.heatmap_colormap = colormap_name
        if self.plot_container:
            for heatmap in self.plot_container.heatmap_plots:
                heatmap.update_colormap(colormap_name)
                heatmap._clear_buffer()
                heatmap.update_plot_content()

    def _on_heatmap_percentile_changed(self, value: float):
        self.app_state.heatmap_exclusion_percentile = value
        if self.plot_container:
            for heatmap in self.plot_container.heatmap_plots:
                heatmap._clear_buffer()
                heatmap.update_plot_content()

    def _on_heatmap_normalization_changed(self, display_name: str):
        self.app_state.heatmap_normalization = _NORM_DISPLAY_TO_KEY.get(display_name, "per_channel")
        if self.plot_container:
            for heatmap in self.plot_container.heatmap_plots:
                heatmap._clear_buffer()
                heatmap.update_plot_content()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _parse_float(self, text: str) -> Optional[float]:
        s = (text or "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _parse_int(self, text: str) -> Optional[int]:
        s = (text or "").strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
