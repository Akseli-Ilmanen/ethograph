"""Changepoints widget - dataset changepoints and audio changepoint detection."""

import logging

import numpy as np
import pandas as pd
import ruptures as rpt
import xarray as xr
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import ethograph as eto
from ethograph.features.changepoints import (
    correct_changepoints,
    correct_changepoints_automatic,
    dataset_changepoint_times,
)
from ethograph.gui.notify import notify
from ethograph.io import schema

from .dialog_function_params import open_function_params_dialog
from .make_pretty import styled_link

logger = logging.getLogger(__name__)

# Maps UI combo text → registry key
_AUDIO_CP_REGISTRY_MAP = {
    "VocalPy meansquared": "meansquared_cp",
    "VocalPy ava": "ava_cp",
    "VocalSeg dynamic thresholding": "vocalseg_cp",
    "VocalSeg continuity filtering": "continuity_cp",
}

_RUPTURES_REGISTRY_MAP = {
    "Pelt": "ruptures_pelt",
    "Binseg": "ruptures_binseg",
    "BottomUp": "ruptures_bottomup",
    "Window": "ruptures_window",
    "Dynp": "ruptures_dynp",
}

_KINEMATIC_REGISTRY_MAP = {
    "troughs": "find_troughs",
    "turning_points": "find_turning_points",
}


def _run_ruptures_in_process(
    signal: np.ndarray,
    method: str,
    params: dict,
) -> tuple[list[int] | None, str | None]:
    try:
        model = params.get("model", "l2")
        min_size = params.get("min_size", 2)
        jump = params.get("jump", 5)

        algo_map = {
            "Pelt": lambda: rpt.Pelt(model=model, min_size=min_size, jump=jump),
            "Binseg": lambda: rpt.Binseg(model=model, min_size=min_size, jump=jump),
            "BottomUp": lambda: rpt.BottomUp(model=model, min_size=min_size, jump=jump),
            "Window": lambda: rpt.Window(
                width=params.get("width", 100),
                model=model,
                min_size=min_size,
                jump=jump,
            ),
            "Dynp": lambda: rpt.Dynp(model=model, min_size=min_size, jump=jump),
        }

        if method not in algo_map:
            return (None, f"Unknown method: {method}")

        algo = algo_map[method]().fit(signal)

        if method == "Pelt":
            bkps = algo.predict(pen=params.get("pen", 1.0))
        elif method == "Binseg":
            pen = params.get("pen")
            if pen is not None:
                bkps = algo.predict(pen=pen)
            else:
                bkps = algo.predict(n_bkps=params.get("n_bkps", 5))
        else:
            bkps = algo.predict(n_bkps=params.get("n_bkps", 5))

        return (bkps, None)

    except (ValueError, RuntimeError) as e:
        return (None, str(e))


class ChangepointsWidget(QWidget):
    """Changepoints controls - dataset changepoints and audio changepoint detection."""

    request_plot_update = Signal()

    def __init__(self, shell, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.shell = shell
        self.plot_container = None
        self.meta_widget = None
        self.setAttribute(Qt.WA_AlwaysShowToolTips)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._main_layout = main_layout
        self._create_shared_controls(main_layout)

        # Detection and correction are two steps, so they live in two hosts the
        # top bar pops up separately (``detection_host`` / ``correction_host``);
        # ``restore_host`` puts a borrowed host back where it was.
        self.detection_host = QWidget()
        detection_layout = QVBoxLayout(self.detection_host)
        detection_layout.setContentsMargins(0, 0, 0, 0)
        detection_layout.setSpacing(2)
        self._create_toggle_buttons(detection_layout)
        self._create_changepoints_panel()
        self._create_ruptures_panel()
        self._create_audio_cp_panel()
        detection_layout.addWidget(self.changepoints_panel)
        detection_layout.addWidget(self.ruptures_panel)
        detection_layout.addWidget(self.audio_cp_panel)

        self.correction_host = QWidget()
        correction_layout = QVBoxLayout(self.correction_host)
        correction_layout.setContentsMargins(0, 0, 0, 0)
        correction_layout.setSpacing(2)
        self._create_correction_params_panel()
        correction_layout.addWidget(self.correction_params_panel)

        self._hosts = (self.detection_host, self.correction_host)
        for host in self._hosts:
            main_layout.addWidget(host)
        main_layout.addStretch()

        self._show_panel("kinematic")

        self._restore_or_set_defaults()
        self.setEnabled(False)

    def _update_trial_dataset(self, new_ds: xr.Dataset):
        trial = self.app_state.trials_sel
        if self.app_state.dt is not None:
            self.app_state.dt.update_trial(trial, lambda _: new_ds)
            self.app_state.ds = self.app_state.dt.trial(trial)
        else:
            self.app_state.ds = new_ds
        store = getattr(self.app_state, "data_loader", None)
        if store is not None and hasattr(store, "update_ds"):
            store.update_ds(self.app_state.ds)

    def _ensure_changepoints_visible(self):
        self.show_cp_checkbox.blockSignals(True)
        self.show_cp_checkbox.setChecked(True)
        self.show_cp_checkbox.blockSignals(False)
        self.app_state.show_changepoints = True
        self.request_plot_update.emit()

    def _store_audio_cps_to_ds(self, onsets: np.ndarray, offsets: np.ndarray, target_feature: str, method: str):
        ds = self.app_state.ds
        if ds is None:
            return

        new_ds = ds.copy()
        for var in ("audio_cp_onsets", "audio_cp_offsets"):
            if var in new_ds.data_vars:
                new_ds = new_ds.drop_vars(var)

        # Onset/offset *times*, not a per-frame mask — read back by name, so
        # they carry no schema attrs.
        attrs = {
            "target_feature": target_feature,
            "method": method,
        }
        new_ds["audio_cp_onsets"] = xr.DataArray(onsets, dims=["audio_cp"], attrs=attrs)
        new_ds["audio_cp_offsets"] = xr.DataArray(offsets, dims=["audio_cp"], attrs=attrs)
        self._update_trial_dataset(new_ds)

    def _get_audio_cps_from_ds(self) -> tuple[np.ndarray, np.ndarray] | None:
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            return None
        if "audio_cp_onsets" not in ds.data_vars or "audio_cp_offsets" not in ds.data_vars:
            return None
        return ds["audio_cp_onsets"].values, ds["audio_cp_offsets"].values

    # =========================================================================
    # Shared controls / toggle buttons
    # =========================================================================

    def _create_shared_controls(self, main_layout):
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)

        self.show_cp_checkbox = QCheckBox("Show changepoints (CPs)")
        self.show_cp_checkbox.setToolTip("Display changepoints on plot")
        self.show_cp_checkbox.setChecked(True)
        self.show_cp_checkbox.stateChanged.connect(self._on_show_changepoints_changed)
        row1_layout.addWidget(self.show_cp_checkbox)

        self.changepoint_correction_checkbox = QCheckBox("Changepoint correction")
        self.changepoint_correction_checkbox.setChecked(self.app_state.apply_changepoint_correction)
        self.changepoint_correction_checkbox.setToolTip(
            "Snap label boundaries to nearest changepoint when creating labels.\n"
            "When enabled, uses full correction parameters.\n"
        )
        self.changepoint_correction_checkbox.stateChanged.connect(self._on_changepoint_correction_changed)
        row1_layout.addWidget(self.changepoint_correction_checkbox)

        row1_layout.addStretch()
        main_layout.addLayout(row1_layout)

    def _create_toggle_buttons(self, main_layout):
        self.toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.toggle_widget.setLayout(toggle_layout)

        toggle_defs = [
            ("cp_toggle", "Kinematic CPs", True, self._toggle_changepoints),
            ("ruptures_toggle", "Ruptures", False, self._toggle_ruptures),
            ("audio_cp_toggle", "Audio CPs", False, self._toggle_audio_cps),
        ]
        for attr, label, checked, callback in toggle_defs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.clicked.connect(callback)
            toggle_layout.addWidget(btn)
            setattr(self, attr, btn)

        main_layout.addWidget(self.toggle_widget)

    def restore_host(self, host: QWidget) -> None:
        """Put a host a pop-up borrowed back in its slot below the shared controls."""
        index = 1 + self._hosts.index(host)
        self._main_layout.insertWidget(index, host)
        host.setVisible(True)

    def show_panel(self, panel_name: str) -> None:
        """Bring one detection panel to the front: ``kinematic``, ``ruptures`` or ``audio_cps``."""
        self._show_panel(panel_name)

    def _show_panel(self, panel_name: str):
        panels = {
            "kinematic": (self.changepoints_panel, self.cp_toggle, "Kinematic CPs"),
            "ruptures": (self.ruptures_panel, self.ruptures_toggle, "Ruptures"),
            "audio_cps": (self.audio_cp_panel, self.audio_cp_toggle, "Audio CPs"),
        }
        for name, (panel, toggle, label) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)

        self._refresh_layout()

    # The toggle row is radio-like: the clicked panel comes to the front and
    # clicking the front one again keeps it (there is no "nothing" state).
    def _toggle_changepoints(self):
        self._show_panel("kinematic")

    def _toggle_ruptures(self):
        self._show_panel("ruptures")

    def _toggle_audio_cps(self):
        self._show_panel("audio_cps")

    def _refresh_layout(self):
        if self.meta_widget:
            self.meta_widget.refresh_widget_layout(self)

    # =========================================================================
    # Panel creation — simplified with "Configure..." buttons
    # =========================================================================

    def _create_changepoints_panel(self):
        self.changepoints_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 0)
        self.changepoints_panel.setLayout(layout)

        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.setToolTip(
            "Troughs: local minima\nTurning points: points where gradient is near zero around peaks"
        )
        self.method_combo.addItems(["troughs", "turning_points"])
        row_layout.addWidget(self.method_combo)

        self.kinematic_configure_btn = QPushButton("Configure...")
        self.kinematic_configure_btn.setToolTip("Open parameter editor for selected method")
        self.kinematic_configure_btn.clicked.connect(self._open_kinematic_params)
        row_layout.addWidget(self.kinematic_configure_btn)
        layout.addLayout(row_layout)

        button_layout = QHBoxLayout()

        self.compute_ds_cp_button = QPushButton("Detect")
        self.compute_ds_cp_button.setToolTip("Detect changepoints for current feature and add to dataset")
        self.compute_ds_cp_button.clicked.connect(self._compute_dataset_changepoints)
        button_layout.addWidget(self.compute_ds_cp_button)

        self.clear_ds_cp_button = QPushButton("Clear")
        self.clear_ds_cp_button.setToolTip("Remove all changepoints for current feature")
        self.clear_ds_cp_button.clicked.connect(self._clear_current_feature_changepoints)
        button_layout.addWidget(self.clear_ds_cp_button)

        self.ds_cp_count_label = QLabel("")
        button_layout.addWidget(self.ds_cp_count_label)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _create_audio_cp_panel(self):
        self.audio_cp_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.audio_cp_panel.setLayout(layout)

        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Method:"))
        self.audio_cp_method_combo = QComboBox()
        self.audio_cp_method_combo.addItems(
            [
                "VocalPy meansquared",
                "VocalPy ava",
                "VocalSeg dynamic thresholding",
                "VocalSeg continuity filtering",
            ]
        )
        self.audio_cp_method_combo.currentTextChanged.connect(self._on_audio_cp_method_changed)
        row_layout.addWidget(self.audio_cp_method_combo)

        self.audio_cp_configure_btn = QPushButton("Configure...")
        self.audio_cp_configure_btn.setToolTip("Open parameter editor for selected method")
        self.audio_cp_configure_btn.clicked.connect(self._open_audio_cp_params)
        row_layout.addWidget(self.audio_cp_configure_btn)
        layout.addLayout(row_layout)

        button_layout = QHBoxLayout()

        self.compute_audio_cp_button = QPushButton("Detect")
        self.compute_audio_cp_button.setToolTip("Detect onset/offset candidates using selected method")
        self.compute_audio_cp_button.clicked.connect(self._compute_audio_changepoints)
        button_layout.addWidget(self.compute_audio_cp_button)

        self.clear_audio_cp_button = QPushButton("Clear")
        self.clear_audio_cp_button.setToolTip("Remove all audio changepoints from the plot")
        self.clear_audio_cp_button.clicked.connect(self._clear_spectral_changepoints)
        button_layout.addWidget(self.clear_audio_cp_button)

        self.audio_cp_count_label = QLabel("")
        button_layout.addWidget(self.audio_cp_count_label)

        button_layout.addStretch()

        self.audio_cp_ref_label = QLabel()
        self.audio_cp_ref_label.setOpenExternalLinks(True)
        button_layout.addWidget(self.audio_cp_ref_label)

        layout.addLayout(button_layout)

        self._on_audio_cp_method_changed(self.audio_cp_method_combo.currentText())

    def _create_ruptures_panel(self):
        self.ruptures_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.ruptures_panel.setLayout(layout)

        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Method:"))
        self.ruptures_method_combo = QComboBox()
        self.ruptures_method_combo.setToolTip(
            "Pelt: Fast, penalty-based (unknown # of changepoints)\n"
            "Binseg: Binary segmentation (fast)\n"
            "BottomUp: Bottom-up segmentation\n"
            "Window: Sliding window method\n"
            "Dynp: Dynamic programming (optimal but slow)"
        )
        self.ruptures_method_combo.addItems(["Pelt", "Binseg", "BottomUp", "Window", "Dynp"])
        row_layout.addWidget(self.ruptures_method_combo)

        self.ruptures_configure_btn = QPushButton("Configure...")
        self.ruptures_configure_btn.setToolTip("Open parameter editor for selected method")
        self.ruptures_configure_btn.clicked.connect(self._open_ruptures_params)
        row_layout.addWidget(self.ruptures_configure_btn)
        layout.addLayout(row_layout)

        button_layout = QHBoxLayout()

        self.compute_ruptures_button = QPushButton("Detect")
        self.compute_ruptures_button.setToolTip("Detect changepoints for current feature using ruptures library")
        self.compute_ruptures_button.clicked.connect(self._compute_ruptures_changepoints)
        button_layout.addWidget(self.compute_ruptures_button)

        self.ruptures_count_label = QLabel("")
        button_layout.addWidget(self.ruptures_count_label)

        button_layout.addStretch()

        ref_label = QLabel(
            styled_link(
                "https://centre-borelli.github.io/ruptures-docs",
                "Ruptures (Truong et al., 2020)",
            )
        )
        ref_label.setOpenExternalLinks(True)
        ref_label.setToolTip("Open ruptures documentation")
        button_layout.addWidget(ref_label)

        layout.addLayout(button_layout)

    # =========================================================================
    # Configure... dialog openers
    # =========================================================================

    def _open_kinematic_params(self):
        method = self.method_combo.currentText()
        key = _KINEMATIC_REGISTRY_MAP.get(method)
        if key and open_function_params_dialog(key, self.app_state, parent=self) is not None:
            self._compute_dataset_changepoints()

    def _open_audio_cp_params(self):
        method = self.audio_cp_method_combo.currentText()
        key = _AUDIO_CP_REGISTRY_MAP.get(method)
        if key and open_function_params_dialog(key, self.app_state, parent=self) is not None:
            self._compute_audio_changepoints()

    def _open_ruptures_params(self):
        method = self.ruptures_method_combo.currentText()
        key = _RUPTURES_REGISTRY_MAP.get(method)
        if key and open_function_params_dialog(key, self.app_state, parent=self) is not None:
            self._compute_ruptures_changepoints()

    # =========================================================================
    # Reference label update
    # =========================================================================

    def _on_audio_cp_method_changed(self, method: str):
        if method.startswith("VocalSeg"):
            self.audio_cp_ref_label.setText(
                styled_link(
                    "https://github.com/timsainb/vocalization-segmentation",
                    "VocalSeg (Sainburg et al., 2020)",
                )
            )
            self.audio_cp_ref_label.setToolTip("Open vocalseg GitHub repository")
        else:
            self.audio_cp_ref_label.setText(
                styled_link(
                    "https://vocalpy.readthedocs.io/",
                    "VocalPy (Nicholson et al.)",
                )
            )
            self.audio_cp_ref_label.setToolTip("Open VocalPy documentation")

    # =========================================================================
    # Setters / state
    # =========================================================================

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    # =========================================================================
    # Defaults / parameter persistence
    # =========================================================================

    def _restore_or_set_defaults(self):
        show_cp = getattr(self.app_state, "show_changepoints", False)
        self.show_cp_checkbox.setChecked(show_cp)
        n_custom = len(self.app_state.cp_label_thresholds)
        if n_custom:
            self.per_label_btn.setText(f"Per-label thresholds ({n_custom})...")

    # =========================================================================
    # Parameter extraction from cache
    # =========================================================================

    def _get_cached_params(self, registry_key: str) -> dict:
        cache = getattr(self.app_state, "function_params_cache", None) or {}
        return dict(cache.get(registry_key, {}))

    def _get_audio_cp_params(self) -> dict:
        method = self.audio_cp_method_combo.currentText()
        key = _AUDIO_CP_REGISTRY_MAP.get(method)
        params = self._get_cached_params(key) if key else {}

        if method == "VocalSeg dynamic thresholding":
            params["method"] = "vocalseg"
        elif method == "VocalSeg continuity filtering":
            params["method"] = "continuity"
        elif method == "VocalPy ava":
            params["method"] = "ava"
            nperseg = params.get("nperseg", 1024)
            params["noverlap"] = nperseg // 2
        else:
            params["method"] = "meansquared"

        return params

    def _get_kinematic_params(self) -> dict:
        method = self.method_combo.currentText()
        key = _KINEMATIC_REGISTRY_MAP.get(method)
        return self._get_cached_params(key) if key else {}

    def _get_ruptures_params(self) -> dict:
        method = self.ruptures_method_combo.currentText()
        key = _RUPTURES_REGISTRY_MAP.get(method)
        return self._get_cached_params(key) if key else {}

    # =========================================================================
    # Show / clear changepoints on plot
    # =========================================================================

    def _on_show_changepoints_changed(self, state):
        show = Qt.CheckState(state) == Qt.Checked
        self.app_state.show_changepoints = show

        if not show:
            self.changepoint_correction_checkbox.setChecked(False)

        # Feature panels draw their own changepoints from PlotData on the
        # plot update below; only the audio lines are drawn from here.
        if self.plot_container:
            if show:
                result = self._get_audio_cps_from_ds()
                if result is not None:
                    self.plot_container.draw_audio_changepoints(*result)
            else:
                self.plot_container.clear_audio_changepoints()

        self.request_plot_update.emit()

    def _clear_spectral_changepoints(self):
        ds = getattr(self.app_state, "ds", None)
        if ds is not None:
            vars_to_drop = [v for v in ("audio_cp_onsets", "audio_cp_offsets") if v in ds.data_vars]
            if vars_to_drop:
                self._update_trial_dataset(ds.drop_vars(vars_to_drop))

        self.audio_cp_count_label.setText("")

        if self.plot_container:
            self.plot_container.clear_audio_changepoints()

        self.request_plot_update.emit()

    # =========================================================================
    # Audio CP detection (meansquared / ava / vocalseg)
    # =========================================================================

    def _compute_audio_changepoints(self):
        from .dialog_busy_progress import BusyProgressDialog

        audio_path, channel_idx = self.app_state.get_audio_source()
        if not audio_path:
            notify("No audio data loaded. Audio CPs require an audio file.", "warning")
            return
        import audioio as aio

        from ..io.audio_extract import resolve_audio_path

        data, sample_rate = aio.load_audio(resolve_audio_path(audio_path))
        sample_rate = float(sample_rate)
        if data.ndim > 1:
            data = data[:, channel_idx]

        params = self._get_audio_cp_params()
        method = params.pop("method")
        signal_array = np.asarray(data, dtype=np.float64)

        if method in ("vocalseg", "continuity"):
            n_fft = params.get("n_fft", 1024)
            min_n_fft = int(np.ceil(0.005 * sample_rate))
            if n_fft < min_n_fft:
                params["n_fft"] = min_n_fft
                notify(f"n_fft raised to {min_n_fft} (minimum for sample rate {sample_rate:.0f} Hz)")
        elif method == "ava":
            nperseg = params.get("nperseg", 1024)
            max_nperseg = max(4, len(signal_array) // 4)
            if nperseg > max_nperseg:
                nperseg = max_nperseg
                params["nperseg"] = nperseg
            params["noverlap"] = nperseg // 2

        def _run():
            from ethograph.features.audio_changepoints import (
                get_audio_changepoints,
            )

            return get_audio_changepoints(
                method=method,
                signal=signal_array,
                sr=sample_rate,
                **params,
            )

        dialog = BusyProgressDialog(f"Detecting audio changepoints ({method})...", parent=self)
        result, error = dialog.execute(_run)

        if dialog.was_cancelled:
            return
        if error:
            notify(f"Error detecting changepoints: {error}", "warning")
            return

        (onsets, offsets), env_time, envelope = result

        if method == "meansquared" and self.plot_container:
            threshold = params.get("threshold", 5000)
            self.plot_container.draw_amplitude_envelope(env_time, envelope, threshold)
        elif method == "ava" and self.plot_container:
            self.plot_container.draw_amplitude_envelope(
                env_time,
                envelope,
                (
                    params.get("thresh_lowest", 0.1),
                    params.get("thresh_min", 0.2),
                    params.get("thresh_max", 0.3),
                ),
            )

        if len(onsets) == 0 and len(offsets) == 0:
            notify("No changepoints detected. Try adjusting parameters.")
            return

        self._store_audio_cps_to_ds(onsets, offsets, "Audio Waveform", method)
        self.audio_cp_count_label.setText(f"{len(onsets)}+{len(offsets)}")
        notify(f"Detected {len(onsets)} onsets, {len(offsets)} offsets")

        if self.plot_container:
            self.plot_container.draw_audio_changepoints(onsets, offsets)

        self._ensure_changepoints_visible()

    # =========================================================================
    # Kinematic (dataset) changepoint detection
    # =========================================================================

    def _compute_dataset_changepoints(self):
        from ethograph.features.changepoints import (
            find_nearest_turning_points_binary,
            find_troughs_binary,
        )

        from .dialog_busy_progress import BusyProgressDialog

        method = self.method_combo.currentText()
        func_kwargs = self._get_kinematic_params()

        if method == "troughs":
            changepoint_func = find_troughs_binary
            changepoint_name = "troughs"
        else:
            changepoint_func = find_nearest_turning_points_binary
            changepoint_name = "turning_points"

        ds_copy = self.app_state.ds.copy()
        feature = self.app_state.features_sel

        def _run():
            return eto.add_changepoints_to_ds(
                ds=ds_copy,
                target_feature=feature,
                changepoint_name=changepoint_name,
                changepoint_func=changepoint_func,
                **func_kwargs,
            )

        dialog = BusyProgressDialog(f"Detecting {changepoint_name}...", parent=self)
        new_ds, error = dialog.execute(_run)

        if dialog.was_cancelled:
            return
        if error:
            notify(f"Error computing changepoints: {error}", "warning")
            return

        cp_var_name = f"{feature}_{changepoint_name}"
        self._update_trial_dataset(new_ds)

        n_changepoints = np.sum(new_ds[cp_var_name].values > 0)
        self.ds_cp_count_label.setText(f"{n_changepoints} changepoints")
        notify(f"Added '{cp_var_name}' with {n_changepoints} changepoints")

        self._ensure_changepoints_visible()

    def _clear_current_feature_changepoints(self):
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            notify("No dataset loaded", "warning")
            return

        feature = getattr(self.app_state, "features_sel", None)
        if not feature:
            notify("No feature selected in Data Controls", "warning")
            return

        n_removed = self._clear_all_changepoints_for_feature(feature)

        if n_removed == 0:
            notify(f"No changepoints found for '{feature}'")
            return

        self.ds_cp_count_label.setText("")
        self.ruptures_count_label.setText("")
        notify(f"Removed {n_removed} changepoint variable(s) for '{feature}'")

        self.request_plot_update.emit()

    def _clear_all_changepoints_for_feature(self, feature: str) -> int:
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            return 0

        cp_suffixes = ["_peaks", "_troughs", "_turning_points", "_ruptures"]
        vars_to_remove = [f"{feature}{suffix}" for suffix in cp_suffixes if f"{feature}{suffix}" in ds.data_vars]

        if not vars_to_remove:
            return 0

        self._update_trial_dataset(ds.drop_vars(vars_to_remove))
        return len(vars_to_remove)

    # =========================================================================
    # Ruptures detection (via BusyProgressDialog + ProcessPoolExecutor)
    # =========================================================================

    def _compute_ruptures_changepoints(self):
        from .dialog_busy_progress import BusyProgressDialog

        features_sel = self.app_state.features_sel
        ds_kwargs = self.app_state.get_ds_kwargs()
        if features_sel == "Audio Waveform":
            notify(
                "Raw audio is too large for ruptures. Select a derived feature or use Audio CPs instead.",
                "warning",
            )
            return

        data, _ = eto.sel_valid(self.app_state.ds[features_sel], ds_kwargs)

        signal = np.asarray(data).reshape(-1, 1)
        method = self.ruptures_method_combo.currentText()
        params = self._get_ruptures_params()

        dialog = BusyProgressDialog(
            f"Detecting ruptures ({method})...",
            parent=self,
            use_process=True,
        )
        result, error = dialog.execute(
            _run_ruptures_in_process,
            signal,
            method,
            params,
        )

        if dialog.was_cancelled:
            self.ruptures_count_label.setText("Cancelled")
            return
        if error:
            notify(f"Error computing ruptures changepoints: {error}", "warning")
            return

        bkps, error_msg = result
        if error_msg:
            notify(f"Error computing ruptures changepoints: {error_msg}", "warning")
            return
        if bkps is None:
            return

        signal_len = len(signal)
        if bkps and bkps[-1] == signal_len:
            bkps = bkps[:-1]

        cp_array = np.zeros(signal_len, dtype=np.int8)
        for bkp in bkps:
            if 0 <= bkp < signal_len:
                cp_array[bkp] = 1

        time_coord = self.app_state.time_coord

        cp_var_name = f"{features_sel}_ruptures"

        new_ds = self.app_state.ds.copy()
        if cp_var_name in new_ds.data_vars:
            new_ds = new_ds.drop_vars(cp_var_name)

        model = params.get("model", "l2")
        new_ds[cp_var_name] = xr.Variable(
            dims=[time_coord.name],
            data=cp_array,
            attrs=schema.changepoint_attrs(
                target_feature=features_sel,
                method=f"ruptures_{method}",
                model=model,
            ),
        )

        self._update_trial_dataset(new_ds)

        n_changepoints = len(bkps)
        self.ruptures_count_label.setText(f"{n_changepoints} changepoints")
        notify(f"Added '{cp_var_name}' with {n_changepoints} changepoints")

        self._ensure_changepoints_visible()

    # =========================================================================
    # Correction Parameters Panel (unchanged)
    # =========================================================================

    def _create_correction_params_panel(self):
        self.correction_params_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.correction_params_panel.setLayout(layout)

        self._motif_mappings = {}
        self._correction_snapshot = None

        self._automatic_correction_group = QGroupBox("Changepoint correction (automatic during labelling)")
        automatic_layout = QGridLayout()
        self._automatic_correction_group.setLayout(automatic_layout)
        layout.addWidget(self._automatic_correction_group)

        self._manual_correction_group = QGroupBox("Changepoint correction (manual) - in development")
        manual_layout = QVBoxLayout()
        self._manual_correction_group.setLayout(manual_layout)
        layout.addWidget(self._manual_correction_group)

        self.max_expansion_spin = QDoubleSpinBox()
        self.max_expansion_spin.setRange(0, 100000)
        self.max_expansion_spin.setDecimals(3)
        self.max_expansion_spin.setToolTip("Max expansion of label boundaries at changepoints")
        self.max_expansion_spin.setValue(self.app_state.cp_max_expansion_s)
        self.max_expansion_spin.valueChanged.connect(lambda v: setattr(self.app_state, "cp_max_expansion_s", float(v)))

        self.max_shrink_spin = QDoubleSpinBox()
        self.max_shrink_spin.setRange(0, 100000)
        self.max_shrink_spin.setDecimals(3)
        self.max_shrink_spin.setToolTip("Max shrinkage of label boundaries at changepoints")
        self.max_shrink_spin.setValue(self.app_state.cp_max_shrink_s)
        self.max_shrink_spin.valueChanged.connect(lambda v: setattr(self.app_state, "cp_max_shrink_s", float(v)))

        self.manual_min_label_length_spin = QDoubleSpinBox()
        self.manual_min_label_length_spin.setRange(0.001, 100000)
        self.manual_min_label_length_spin.setDecimals(3)
        self.manual_min_label_length_spin.setToolTip("Minimum label length used by manual changepoint correction.")
        self.manual_min_label_length_spin.setValue(self.app_state.cp_min_label_length_s)
        self.manual_min_label_length_spin.valueChanged.connect(
            lambda v: setattr(self.app_state, "cp_min_label_length_s", float(v))
        )

        self.manual_stitch_gap_spin = QDoubleSpinBox()
        self.manual_stitch_gap_spin.setRange(0, 100000)
        self.manual_stitch_gap_spin.setDecimals(3)
        self.manual_stitch_gap_spin.setToolTip("Gap threshold used by manual changepoint correction.")
        self.manual_stitch_gap_spin.setValue(self.app_state.cp_stitch_gap_len_s)
        self.manual_stitch_gap_spin.valueChanged.connect(
            lambda v: setattr(self.app_state, "cp_stitch_gap_len_s", float(v))
        )

        self.automatic_min_label_length_spin = QDoubleSpinBox()
        self.automatic_min_label_length_spin.setRange(0.001, 100000)
        self.automatic_min_label_length_spin.setDecimals(3)
        self.automatic_min_label_length_spin.setToolTip(
            "Minimum label length used in the automatic cleanup after applying a label."
        )
        self.automatic_min_label_length_spin.setValue(getattr(self.app_state, "automatic_min_label_length_s", 1e-3))
        self.automatic_min_label_length_spin.valueChanged.connect(
            lambda value: setattr(self.app_state, "automatic_min_label_length_s", float(value))
        )

        self.automatic_stitch_gap_spin = QDoubleSpinBox()
        self.automatic_stitch_gap_spin.setRange(0, 100000)
        self.automatic_stitch_gap_spin.setDecimals(3)
        self.automatic_stitch_gap_spin.setToolTip("Gap threshold used in the automatic cleanup after applying a label.")
        self.automatic_stitch_gap_spin.setValue(getattr(self.app_state, "automatic_stitch_gap_s", 0.0))
        self.automatic_stitch_gap_spin.valueChanged.connect(
            lambda value: setattr(self.app_state, "automatic_stitch_gap_s", float(value))
        )

        for spin in (
            self.max_expansion_spin,
            self.max_shrink_spin,
            self.manual_min_label_length_spin,
            self.manual_stitch_gap_spin,
            self.automatic_min_label_length_spin,
            self.automatic_stitch_gap_spin,
        ):
            spin.setDecimals(3)
            spin.setSuffix(" s")

        automatic_layout.addWidget(QLabel("Min label length (s):"), 0, 0)
        automatic_layout.addWidget(self.automatic_min_label_length_spin, 0, 1)
        automatic_layout.addWidget(QLabel("Stitch gap (s):"), 0, 2)
        automatic_layout.addWidget(self.automatic_stitch_gap_spin, 0, 3)

        manual_grid = QGridLayout()
        manual_grid.addWidget(QLabel("Min label length (s):"), 0, 0)
        manual_grid.addWidget(self.manual_min_label_length_spin, 0, 1)
        manual_grid.addWidget(QLabel("Stitch gap (s):"), 0, 2)
        manual_grid.addWidget(self.manual_stitch_gap_spin, 0, 3)
        manual_grid.addWidget(QLabel("Max expansion (s):"), 1, 0)
        manual_grid.addWidget(self.max_expansion_spin, 1, 1)
        manual_grid.addWidget(QLabel("Max shrink (s):"), 1, 2)
        manual_grid.addWidget(self.max_shrink_spin, 1, 3)
        manual_note = QLabel("Manual correction is for testing parameters of model correction.")
        manual_note.setWordWrap(True)
        manual_layout.addWidget(manual_note)
        manual_layout.addLayout(manual_grid)

        steps_layout = QHBoxLayout()
        self.cp_step_purge_cb = QCheckBox("1. Purge")
        self.cp_step_purge_cb.setChecked(self.app_state.cp_step_purge)
        self.cp_step_purge_cb.setToolTip("Remove intervals shorter than min label length")
        self.cp_step_stitch_cb = QCheckBox("2. Stitch")
        self.cp_step_stitch_cb.setChecked(self.app_state.cp_step_stitch)
        self.cp_step_stitch_cb.setToolTip("Merge same-label intervals across small gaps")
        self.cp_step_snap_cb = QCheckBox("3. Snap to changepoints")
        self.cp_step_snap_cb.setChecked(self.app_state.cp_step_snap)
        self.cp_step_snap_cb.setToolTip("Snap interval boundaries to nearest changepoint")
        self.cp_step_purge_after_cb = QCheckBox("4. Purge again")
        self.cp_step_purge_after_cb.setChecked(self.app_state.cp_step_purge_after)
        self.cp_step_purge_after_cb.setToolTip("Remove short intervals created by snapping")
        self.cp_step_purge_cb.stateChanged.connect(lambda v: setattr(self.app_state, "cp_step_purge", bool(v)))
        self.cp_step_stitch_cb.stateChanged.connect(lambda v: setattr(self.app_state, "cp_step_stitch", bool(v)))
        self.cp_step_snap_cb.stateChanged.connect(lambda v: setattr(self.app_state, "cp_step_snap", bool(v)))
        self.cp_step_purge_after_cb.stateChanged.connect(
            lambda v: setattr(self.app_state, "cp_step_purge_after", bool(v))
        )
        for cb in (
            self.cp_step_purge_cb,
            self.cp_step_stitch_cb,
            self.cp_step_snap_cb,
            self.cp_step_purge_after_cb,
        ):
            steps_layout.addWidget(cb)
        steps_layout.addStretch()
        manual_layout.addLayout(steps_layout)

        button_layout = QHBoxLayout()

        self.per_label_btn = QPushButton("Per-label thresholds...")
        self.per_label_btn.setToolTip("Override min label length for individual labels")
        self.per_label_btn.clicked.connect(self._open_label_thresholds_dialog)
        button_layout.addWidget(self.per_label_btn)

        button_layout.addStretch()
        manual_layout.addLayout(button_layout)

        correction_layout = QHBoxLayout()

        cp_label = QLabel("Apply manual correction to:")
        correction_layout.addWidget(cp_label)

        self.cp_correction_trial_btn = QPushButton("Single Trial")
        self.cp_correction_trial_btn.clicked.connect(lambda: self._cp_correction("single_trial"))
        correction_layout.addWidget(self.cp_correction_trial_btn)

        self.cp_correction_all_trials_btn = QPushButton("All Trials (Filtered only)")
        self.cp_correction_all_trials_btn.clicked.connect(lambda: self._cp_correction("all_trials"))
        correction_layout.addWidget(self.cp_correction_all_trials_btn)

        self.cp_undo_btn = QPushButton("\u21bb")
        self.cp_undo_btn.setToolTip("Undo last manual correction")
        self.cp_undo_btn.setFixedWidth(30)
        self.cp_undo_btn.setEnabled(False)
        self.cp_undo_btn.clicked.connect(self._undo_correction)
        correction_layout.addWidget(self.cp_undo_btn)

        correction_layout.addStretch()
        manual_layout.addLayout(correction_layout)

        apply_cp = self.changepoint_correction_checkbox.isChecked()
        self._automatic_correction_group.setEnabled(apply_cp)
        self._manual_correction_group.setEnabled(apply_cp)

    def set_motif_mappings(self, mappings: dict):
        self._motif_mappings = mappings

    def _open_label_thresholds_dialog(self):
        if not self._motif_mappings:
            notify("No label mappings loaded yet", "warning")
            return

        dialog = LabelThresholdsDialog(
            self._motif_mappings,
            self.app_state.cp_label_thresholds,
            self.manual_min_label_length_spin.value(),
            parent=self,
        )
        if dialog.exec_():
            self.app_state.cp_label_thresholds = dialog.get_custom_thresholds()
            n_custom = len(self.app_state.cp_label_thresholds)
            if n_custom:
                self.per_label_btn.setText(f"Per-label thresholds ({n_custom})...")
            else:
                self.per_label_btn.setText("Per-label thresholds...")

    def is_changepoint_correction_enabled(self) -> bool:
        return self.changepoint_correction_checkbox.isChecked()

    def _on_changepoint_correction_changed(self, state):
        enabled = Qt.CheckState(state) == Qt.Checked
        self.app_state.apply_changepoint_correction = enabled
        if hasattr(self, "_automatic_correction_group"):
            self._automatic_correction_group.setEnabled(enabled)
        if hasattr(self, "_manual_correction_group"):
            self._manual_correction_group.setEnabled(enabled)

    def _save_correction_snapshot(self, mode):
        snapshot = {"mode": mode}
        if mode == "single_trial":
            trial = self.app_state.trials_sel
            snapshot["trial"] = trial
            snapshot["intervals_df"] = self.app_state.get_trial_intervals(trial).copy()
        elif mode == "all_trials":
            snapshot["trials"] = {}
            for trial in self.app_state.trials:
                snapshot["trials"][trial] = self.app_state.get_trial_intervals(trial).copy()
            snapshot["old_cp_corrected"] = self.app_state.get_global_meta_attr("changepoint_corrected", 0)
        self._correction_snapshot = snapshot
        self.cp_undo_btn.setEnabled(True)

    def _undo_correction(self):
        if self._correction_snapshot is None:
            return
        snapshot = self._correction_snapshot
        mode = snapshot["mode"]

        if mode == "single_trial":
            trial = snapshot["trial"]
            self.app_state.set_trial_intervals(trial, snapshot["intervals_df"])
            if trial == self.app_state.trials_sel:
                self.app_state.label_intervals = snapshot["intervals_df"]
        elif mode == "all_trials":
            for trial, df in snapshot["trials"].items():
                self.app_state.set_trial_intervals(trial, df)
            self.app_state.set_global_meta_attr("changepoint_corrected", snapshot["old_cp_corrected"])
            self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
            self._update_cp_status()

        self._correction_snapshot = None
        self.cp_undo_btn.setEnabled(False)
        if self.data_widget:
            self.data_widget.update_main_plot()
        notify("Reverted correction")

    def _extract_cp_times(self, ds, ds_kwargs):
        """Changepoint times (trial clock) the correction snaps to, from either backend.

        The current feature's masks at the sidebar's selections plus the
        trial's audio changepoints. *ds* is the trial being corrected, so an
        all-trials run reads each trial's own masks; a pynapple session
        (no *ds*) asks the loader, whose changepoints are event times.
        """
        feature = getattr(self.app_state, "features_sel", None)
        if ds is None:
            wb = self.app_state.window_bounds
            t0, t1 = (wb.start_s, wb.end_s) if wb is not None else (None, None)
            return self.app_state.data_loader.get_cp_times(feature, ds_kwargs, t0=t0, t1=t1)

        cp_times = dataset_changepoint_times(ds, feature, ds_kwargs)
        if "audio_cp_onsets" in ds.data_vars and "audio_cp_offsets" in ds.data_vars:
            onsets = ds["audio_cp_onsets"].values.astype(np.float64)
            offsets = ds["audio_cp_offsets"].values.astype(np.float64)
            cp_times = np.unique(np.concatenate([cp_times, onsets, offsets]))
        return cp_times

    def _correct_trial_intervals(self, trial, ds, all_params, ds_kwargs) -> tuple[pd.DataFrame, bool]:
        """Interval-native correction: purge -> stitch -> snap -> purge.

        Returns the corrected labels and whether anything was *snapped* —
        False when the trial has no changepoints (or the snap step is off),
        so the caller does not record a correction that never happened.
        """
        intervals_df = self.app_state.get_trial_intervals(trial)

        cp_kwargs = all_params.get("cp_kwargs", ds_kwargs)
        cp_times = self._extract_cp_times(ds, cp_kwargs)
        snapped = self.cp_step_snap_cb.isChecked() and len(cp_times) > 0

        min_duration_s = all_params.get("min_label_length_s", 0)
        label_thresholds_raw = all_params.get("label_thresholds", {})
        stitch_gap_s = all_params.get("stitch_gap_len_s", 0)
        cp_params = all_params.get("changepoint_params", {})
        max_expansion_s = cp_params.get("max_expansion_s", np.inf)
        max_shrink_s = cp_params.get("max_shrink_s", np.inf)

        label_thresholds_s = {int(k): v for k, v in label_thresholds_raw.items()}

        corrected = correct_changepoints(
            intervals_df,
            cp_times,
            min_duration_s=min_duration_s,
            stitch_gap_s=stitch_gap_s,
            max_expansion_s=max_expansion_s,
            max_shrink_s=max_shrink_s,
            label_thresholds_s=label_thresholds_s or None,
            do_purge=self.cp_step_purge_cb.isChecked(),
            do_stitch=self.cp_step_stitch_cb.isChecked(),
            do_snap=snapped,
            do_purge_after=self.cp_step_purge_after_cb.isChecked(),
        )
        return corrected, snapped

    def _current_feature_has_changepoints(self) -> bool:
        return getattr(self.app_state, "plot_has_changepoints", False)

    def cp_correction_from_labelling(self):
        if not self.app_state.apply_changepoint_correction:
            return
        if not self._current_feature_has_changepoints():
            return

        min_duration_s, stitch_gap_s = self.get_apply_label_cleanup_params()

        trial = self.app_state.trials_sel
        corrected_df = correct_changepoints_automatic(
            self.app_state.get_trial_intervals(trial),
            min_duration_s=min_duration_s,
            stitch_gap_s=stitch_gap_s,
        )
        self.app_state.set_trial_intervals(trial, corrected_df)
        self.app_state.label_intervals = corrected_df

    def _cp_correction(self, mode):
        all_params = self.get_correction_params()
        ds_kwargs = self.app_state.get_ds_kwargs()
        all_params["cp_kwargs"] = ds_kwargs

        try:
            # A trial is stamped corrected only when something was snapped:
            # with no changepoints (or the snap step off) purge/stitch still
            # run, but no correction happened and the button must not go green.
            if mode == "single_trial":
                self._save_correction_snapshot(mode)
                trial = self.app_state.trials_sel
                corrected_df, snapped = self._correct_trial_intervals(trial, self.app_state.ds, all_params, ds_kwargs)
                self.app_state.set_trial_intervals(trial, corrected_df)
                self.app_state.label_intervals = corrected_df
                if snapped:
                    self.app_state.set_trial_meta_attr(trial, "changepoint_corrected", 1)
                else:
                    notify(f"Trial {trial}: no changepoints to snap to — labels purged/stitched only", "warning")
                self._update_cp_status()

            if mode == "all_trials":
                if self.app_state.get_global_meta_attr("changepoint_corrected", 0) == 1:
                    notify("Note: Changepoint correction was previously applied to all trials. Re-applying.")

                # TODO: Mention in documentation, only Ctrl+Z functionality of the GUI.
                self._save_correction_snapshot(mode)
                unsnapped: list = []
                for trial in self.app_state.trials:
                    ds = self.app_state.dt.trial(trial) if self.app_state.dt is not None else self.app_state.ds
                    corrected_df, snapped = self._correct_trial_intervals(trial, ds, all_params, ds_kwargs)
                    self.app_state.set_trial_intervals(trial, corrected_df)
                    if snapped:
                        self.app_state.set_trial_meta_attr(trial, "changepoint_corrected", 1)
                    else:
                        unsnapped.append(trial)
                if len(unsnapped) < len(self.app_state.trials):
                    self.app_state.set_global_meta_attr("changepoint_corrected", 1)
                if unsnapped:
                    notify(
                        f"{len(unsnapped)} of {len(self.app_state.trials)} trials had no changepoints to snap to "
                        f"(e.g. trial {unsnapped[0]}) — those were purged/stitched only",
                        "warning",
                    )
                self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
                self._update_cp_status()

            if self.data_widget:
                self.data_widget.update_main_plot()

        except (ValueError, IndexError, RuntimeError) as e:
            logger.exception("Changepoint correction failed")
            notify(f"Changepoint correction failed: {e}", "warning")

    def _update_cp_status(self):
        default_style = ""
        corrected_style = "background-color: green; color: white;"

        if self.app_state.trials_sel is None:
            self.cp_correction_trial_btn.setStyleSheet(default_style)
            self.cp_correction_all_trials_btn.setStyleSheet(default_style)
            return

        apply_cp = self.changepoint_correction_checkbox.isChecked()
        if hasattr(self, "_manual_correction_group"):
            self._manual_correction_group.setEnabled(apply_cp)
        self.cp_correction_all_trials_btn.setToolTip("")

        trial_corrected = self.app_state.get_trial_meta(self.app_state.trials_sel).get("changepoint_corrected", 0)
        self.cp_correction_trial_btn.setStyleSheet(corrected_style if trial_corrected else default_style)

        global_corrected = self.app_state.get_global_meta_attr("changepoint_corrected", 0)
        self.cp_correction_all_trials_btn.setStyleSheet(corrected_style if global_corrected else default_style)

    def get_correction_params(self) -> dict:
        return {
            "min_label_length_s": self.app_state.cp_min_label_length_s,
            "label_thresholds": {str(k): v for k, v in self.app_state.cp_label_thresholds.items()},
            "stitch_gap_len_s": self.app_state.cp_stitch_gap_len_s,
            "changepoint_params": {
                "max_expansion_s": self.app_state.cp_max_expansion_s,
                "max_shrink_s": self.app_state.cp_max_shrink_s,
            },
        }

    def get_apply_label_cleanup_params(self) -> tuple[float, float]:
        return (
            self.automatic_min_label_length_spin.value(),
            self.automatic_stitch_gap_spin.value(),
        )

    # =========================================================================
    # Changepoint navigation (jump forward/backward between CPs)
    # =========================================================================

    def jump_changepoint(self, direction: int):
        """Jump to the next (direction=+1) or previous (direction=-1) changepoint.

        Panel context:
        - audio/spectrogram panel last clicked → audio changepoints
        - feature/ephys/raster panel last clicked → dataset kinematic changepoints
        """
        if self.plot_container is None:
            return

        current_time = self._get_current_time()
        cp_times = self._get_jump_cp_times()
        if cp_times is None or len(cp_times) == 0:
            notify("No changepoints available. Detect changepoints first.")
            return

        target = self._find_adjacent_cp(cp_times, current_time, direction)
        if target is None:
            return

        self._seek_to_time(target)

    def _get_current_time(self) -> float:
        video = getattr(self.app_state, "video", None)
        if video:
            return video.frame_to_time(self.app_state.current_frame)

    def _get_jump_cp_times(self) -> np.ndarray | None:
        last_panel = getattr(self.plot_container, "_last_clicked_panel", "feature")
        if last_panel in ("audio", "spectrogram"):
            result = self._get_audio_cps_from_ds()
            if result is None:
                return None
            onsets, offsets = result
            return np.unique(np.concatenate([onsets, offsets]))
        else:
            # The marks the active feature panel draws: its feature at its
            # own selections — the set a click on it snaps to.
            loader = getattr(self.app_state, "data_loader", None)
            plot = self.plot_container.get_current_plot()
            feature = plot._effective_feature()
            if loader is None or not feature:
                return None
            cp_times = loader.get_cp_times(feature, plot._effective_selections())
            return cp_times if len(cp_times) else None

    def _find_adjacent_cp(self, cp_times: np.ndarray, current_time: float, direction: int) -> float | None:
        if direction > 0:
            candidates = cp_times[cp_times > current_time + 1e-3]
            return float(candidates[0]) if len(candidates) > 0 else None
        else:
            candidates = cp_times[cp_times < current_time - 1e-3]
            return float(candidates[-1]) if len(candidates) > 0 else None

    def _seek_to_time(self, time_s: float):
        video = getattr(self.app_state, "video", None)
        if video:
            new_frame = video.time_to_frame(time_s)
            self.app_state.current_frame = new_frame
            video.blockSignals(True)
            video.seek_to_frame(new_frame)
            video.blockSignals(False)

        self.plot_container.update_time_marker_by_time(time_s)

        xlim = self.plot_container.get_current_xlim()
        if time_s < xlim[0] or time_s > xlim[1]:
            half = self.app_state.view_span / 2.0
            master = self.plot_container._xlink_master or self.plot_container._feature_plot
            master.vb.setXRange(time_s - half, time_s + half, padding=0)

    def closeEvent(self, event):
        super().closeEvent(event)


class LabelThresholdsDialog(QDialog):
    def __init__(
        self,
        motif_mappings: dict,
        custom_thresholds: dict,
        global_min: float,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Per-label min length")
        self.setMinimumWidth(350)

        self._global_min = global_min
        self._custom_thresholds = dict(custom_thresholds)

        layout = QVBoxLayout(self)

        info = QLabel(f"Global min label length: {global_min} s")
        layout.addWidget(info)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["ID", "Name", "Min Length"])
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(24)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(0, 35)
        self._table.setColumnWidth(2, 90)

        items = [(k, v) for k, v in motif_mappings.items() if k != 0]
        self._table.setRowCount(len(items))
        self._spins: dict[int, QDoubleSpinBox] = {}

        for row_idx, (motif_id, data) in enumerate(items):
            id_item = QTableWidgetItem(str(motif_id))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row_idx, 0, id_item)

            name_item = QTableWidgetItem(data["name"])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row_idx, 1, name_item)

            spin = QDoubleSpinBox()
            spin.setRange(0.001, 100000)
            spin.setDecimals(3)
            spin.setSuffix(" s")
            spin.setValue(self._custom_thresholds.get(motif_id, global_min))
            self._spins[motif_id] = spin
            self._table.setCellWidget(row_idx, 2, spin)

        layout.addWidget(self._table)

        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset all to global")
        reset_btn.clicked.connect(self._reset_all)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _reset_all(self):
        for spin in self._spins.values():
            spin.setValue(self._global_min)

    def get_custom_thresholds(self) -> dict[int, float]:
        result = {}
        for motif_id, spin in self._spins.items():
            val = spin.value()
            if val != self._global_min:
                result[motif_id] = val
        return result
