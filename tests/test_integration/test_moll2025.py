"""Moll2025 integration tests: template loading, labelling with/without
changepoint correction, space plot 2D/3D, lineplot, pynapple loading."""

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from ethograph.datasets import dataset_dir
from ethograph.labels.intervals import find_interval_at

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _navigate_to_trial(meta, trial_id):
    meta.navigation_widget.scope_combo.setCurrentText("Trial start → Trial end")
    QApplication.processEvents()
    meta.navigation_widget.trials_combo.setCurrentText(str(trial_id))
    QApplication.processEvents()
    assert meta.app_state.trials_sel == trial_id, (
        f"Expected trial {trial_id}, got {meta.app_state.trials_sel}. Available: {meta.app_state.trials}"
    )


def _delete_all_labels(meta):
    from ethograph.labels.intervals import empty_intervals

    meta.app_state.label_intervals = empty_intervals()
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# Constants — manually validated on the Moll2025 dataset, trial 41
# ---------------------------------------------------------------------------

TRIAL = 41
LABEL_ID = 1
T_CLICK_START = 1.30
T_CLICK_END = 1.37
T_SNAPPED_ONSET = 1.27
T_SNAPPED_OFFSET = 1.375


# ===================================================================
# Template loading
# ===================================================================


class TestMollComboInteractions:
    def test_all_checkbox_sets_sel_to_none(self, moll2025_gui):
        _, meta = moll2025_gui
        if not meta.data_widget.all_checkboxes:
            pytest.skip("No 'All' checkboxes available")
        key = next(iter(meta.data_widget.all_checkboxes))
        checkbox = meta.data_widget.all_checkboxes[key]
        checkbox.setChecked(True)
        QApplication.processEvents()
        assert meta.app_state.get_key_sel(key) is None

    def test_uncheck_all_restores_combo_value(self, moll2025_gui):
        _, meta = moll2025_gui
        if not meta.data_widget.all_checkboxes:
            pytest.skip("No 'All' checkboxes available")
        key = next(iter(meta.data_widget.all_checkboxes))
        combo = meta.data_widget.combos[key]
        checkbox = meta.data_widget.all_checkboxes[key]
        original_text = combo.currentText()
        checkbox.setChecked(True)
        QApplication.processEvents()
        checkbox.setChecked(False)
        QApplication.processEvents()
        restored = meta.app_state.get_key_sel(key)
        assert restored == original_text


class TestMollActivateAndClickLabel:
    def test_activate_label(self, moll2025_gui):
        _, meta = moll2025_gui
        meta.labels_widget.activate_label(1)
        assert meta.labels_widget.ready_for_label_click is True
        assert meta.labels_widget.selected_labels == 1

    def test_label_creation_via_two_clicks(self, moll2025_gui):
        _, meta = moll2025_gui

        labels = 1
        t_start = 1.0
        t_end = 2.0

        # Off, or the click snaps to the nearest changepoint the panel draws.
        meta.changepoints_widget.changepoint_correction_checkbox.setChecked(False)
        meta.labels_widget.activate_label(labels)
        meta.labels_widget._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
        assert meta.labels_widget.first_click == pytest.approx(t_start)

        meta.labels_widget._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty, "No intervals after label creation"

        individual = meta.labels_widget._current_individual()
        idx = find_interval_at(df, (t_start + t_end) / 2, individual)
        assert idx is not None, "Interval not found at midpoint"

        row = df.loc[idx]
        assert row["labels"] == labels
        assert row["onset_s"] == pytest.approx(t_start, abs=0.01)
        assert row["offset_s"] == pytest.approx(t_end, abs=0.01)


class TestMoll2025Loading:
    def test_state_after_load(self, moll2025_gui):
        _, meta = moll2025_gui
        s = meta.app_state
        assert s.ready is True
        assert s.dt is not None
        assert s.ds is not None

    def test_two_trials(self, moll2025_gui):
        _, meta = moll2025_gui
        assert len(meta.app_state.trials) == 2

    def test_features_rich(self, moll2025_gui):
        _, meta = moll2025_gui
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "speed" in items
        assert "position" in items

    def test_navigate_to_second_trial(self, moll2025_gui):
        _, meta = moll2025_gui
        meta.navigation_widget.scope_combo.setCurrentText("Trial start → Trial end")
        QApplication.processEvents()
        trials = meta.app_state.trials
        if len(trials) < 2:
            pytest.skip("Need 2+ trials")
        meta.navigation_widget.trials_combo.setCurrentText(str(trials[0]))
        QApplication.processEvents()
        first = meta.app_state.trials_sel
        meta.navigation_widget.trials_combo.setCurrentText(str(trials[1]))
        QApplication.processEvents()
        second = meta.app_state.trials_sel
        assert second != first
        assert meta.app_state.ds is not None

    def test_labels_imported(self, moll2025_gui):
        _, meta = moll2025_gui
        df = meta.app_state._all_labels_df
        assert df is not None
        assert not df.empty

    def test_label_creation_and_verify(self, moll2025_gui):
        _, meta = moll2025_gui
        lw = meta.labels_widget
        if not lw._mappings or 1 not in lw._mappings:
            pytest.skip("No label mapping 1")

        lw.activate_label(1)
        assert lw.ready_for_label_click is True

        time = meta.app_state.time_coord
        t_start = float(time[len(time) // 4])
        t_end = float(time[len(time) // 4 + 10])
        lw._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
        lw._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty

    def test_save_labels_tsv(self, moll2025_gui, tmp_path):
        _, meta = moll2025_gui

        lw = meta.labels_widget
        if lw._mappings and 1 in lw._mappings:
            lw.activate_label(1)
            time = meta.app_state.time_coord
            t_start = float(time[len(time) // 4])
            t_end = float(time[len(time) // 4 + 10])
            lw._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
            lw._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
            QApplication.processEvents()

        from ethograph.labels.tsv_store import load_labels_tsv, save_labels_tsv

        tsv_out = tmp_path / "test_labels.tsv"
        df = meta.app_state._all_labels_df
        assert df is not None
        save_labels_tsv(tsv_out, df)

        assert tsv_out.exists()
        loaded = load_labels_tsv(tsv_out)
        assert len(loaded) == len(df)


# ===================================================================
# Pynapple .npz loading
# ===================================================================


class TestMoll2025Pynapple:
    @pytest.fixture(autouse=True)
    def _check_npz(self):
        dest = dataset_dir("moll2025")
        self._speed_npz = dest / "beakTip_speed.npz"
        if not self._speed_npz.exists():
            pytest.skip("Moll2025 pynapple .npz not downloaded")

    def test_load_npz_into_gui(self, moll2025_gui):
        _, meta = moll2025_gui
        io = meta.io_widget
        io._clear_all_line_edits()
        io.nc_file_path_edit.setText(str(self._speed_npz))
        meta.app_state.nc_file_path = str(self._speed_npz)

        meta.data_widget.on_load_clicked()
        QApplication.processEvents()

        assert meta.app_state.ready is True
        # Pynapple loads build no TrialTree — the session lives in ds/loader.
        assert meta.app_state.dt is None
        assert meta.app_state.ds is not None
        assert "beakTip_speed" in meta.app_state.data_loader.catalog.feature_choices()

    def test_npz_type_vars_structure(self):
        from ethograph.io.catalog import catalog_from_pynapple
        from ethograph.io.pynapple import load_nap_data

        data, _trials_ep = load_nap_data(str(self._speed_npz))
        cat = catalog_from_pynapple(data)
        tv = cat.to_type_vars_dict()
        assert isinstance(tv, dict)
        assert "features" in tv or "individuals" in tv or len(tv) > 0


# ===================================================================
# Labelling without changepoints
# ===================================================================


class TestMollLabellingWithoutChangepoints:
    def test_label_at_exact_click_times(self, moll2025_gui):
        _, meta = moll2025_gui
        _navigate_to_trial(meta, TRIAL)
        _delete_all_labels(meta)

        lw = meta.labels_widget
        if not lw._mappings or LABEL_ID not in lw._mappings:
            pytest.skip(f"No label mapping {LABEL_ID}")

        meta.changepoints_widget.changepoint_correction_checkbox.setChecked(False)
        QApplication.processEvents()
        assert not meta.changepoints_widget.is_changepoint_correction_enabled()

        lw.activate_label(LABEL_ID)
        lw._on_plot_clicked({"x": T_CLICK_START, "button": Qt.LeftButton})
        assert lw.first_click == pytest.approx(T_CLICK_START)

        lw._on_plot_clicked({"x": T_CLICK_END, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty, "No label created"

        individual = lw._current_individual()
        idx = find_interval_at(df, (T_CLICK_START + T_CLICK_END) / 2, individual)
        assert idx is not None, "Label not found at midpoint"

        row = df.loc[idx]
        assert row["labels"] == LABEL_ID
        assert row["onset_s"] == pytest.approx(T_CLICK_START, abs=0.01)
        assert row["offset_s"] == pytest.approx(T_CLICK_END, abs=0.01)


# ===================================================================
# Labelling with changepoints
# ===================================================================


class TestMollLabellingWithChangepoints:
    def test_label_snaps_to_changepoints(self, moll2025_gui):
        _, meta = moll2025_gui
        _navigate_to_trial(meta, TRIAL)
        _delete_all_labels(meta)

        from ethograph.utils.qt import set_combo_to_value

        set_combo_to_value(meta.data_widget.combos["features"], "speed")
        QApplication.processEvents()

        lw = meta.labels_widget
        if not lw._mappings or LABEL_ID not in lw._mappings:
            pytest.skip(f"No label mapping {LABEL_ID}")

        meta.changepoints_widget.changepoint_correction_checkbox.setChecked(True)
        QApplication.processEvents()
        assert meta.changepoints_widget.is_changepoint_correction_enabled()

        lw.activate_label(LABEL_ID)
        lw._on_plot_clicked({"x": T_CLICK_START, "button": Qt.LeftButton})
        assert lw.first_click == pytest.approx(T_SNAPPED_ONSET, abs=0.02), (
            f"First click should snap to ~{T_SNAPPED_ONSET}, got {lw.first_click}"
        )

        lw._on_plot_clicked({"x": T_CLICK_END, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty, "No label created"

        individual = lw._current_individual()
        mid = (T_SNAPPED_ONSET + T_SNAPPED_OFFSET) / 2
        idx = find_interval_at(df, mid, individual)
        assert idx is not None, "Snapped label not found at expected midpoint"

        row = df.loc[idx]
        assert row["labels"] == LABEL_ID
        assert row["onset_s"] == pytest.approx(T_SNAPPED_ONSET, abs=0.02)
        assert row["offset_s"] == pytest.approx(T_SNAPPED_OFFSET, abs=0.02)

        meta.changepoints_widget.changepoint_correction_checkbox.setChecked(False)
        QApplication.processEvents()


# ===================================================================
# Space plot
# ===================================================================


class TestMollSpacePlot:
    def test_space_2d_has_data(self, moll2025_gui):
        _, meta = moll2025_gui
        _navigate_to_trial(meta, TRIAL)

        meta.app_state.space_plot_type = "Space Plot"
        if hasattr(meta.data_widget, "space_view_combo"):
            meta.data_widget.space_view_combo.setCurrentText("Space Plot")
            QApplication.processEvents()

        meta.data_widget.update_space_plot()
        QApplication.processEvents()

        sp = meta.data_widget.space_plot
        assert sp is not None, "SpacePlot not created"

        sp.cb_3d.setChecked(False)
        QApplication.processEvents()
        sp.refresh()
        QApplication.processEvents()

        assert sp._trajectory_pos is not None, "No trajectory data"
        X, Y, Z = sp._trajectory_pos
        assert X is not None and len(X) > 0, "No X data in space plot"
        assert Y is not None and len(Y) > 0, "No Y data in space plot"
        assert np.any(np.isfinite(X)), "X data is all NaN"
        assert np.any(np.isfinite(Y)), "Y data is all NaN"
        assert sp._trajectory_times is not None and len(sp._trajectory_times) > 0

        # Verify actual plot items are rendered on the 2D widget
        import pyqtgraph as pg

        plot_item = sp.space_widget.getPlotItem()
        trajectory_items = [
            item
            for item in plot_item.items
            if isinstance(item, (pg.PlotCurveItem, pg.PlotDataItem)) or hasattr(item, "_is_trajectory")
        ]
        assert len(trajectory_items) > 0, (
            f"No trajectory items rendered on 2D space plot. Items: {[type(i).__name__ for i in plot_item.items]}"
        )

    def test_space_3d_has_data(self, moll2025_gui):
        _, meta = moll2025_gui
        _navigate_to_trial(meta, TRIAL)

        meta.app_state.space_plot_type = "Space Plot"
        meta.app_state.space_show_references = True
        if hasattr(meta.data_widget, "space_view_combo"):
            meta.data_widget.space_view_combo.setCurrentText("Space Plot")
            QApplication.processEvents()

        meta.data_widget.update_space_plot()
        QApplication.processEvents()

        sp = meta.data_widget.space_plot
        assert sp is not None, "SpacePlot not created"

        if sp.z_combo.count() == 0:
            pytest.skip("No z-axis options available for 3D")

        sp.cb_3d.setChecked(True)
        QApplication.processEvents()
        sp.refresh()
        QApplication.processEvents()

        assert sp._trajectory_pos is not None, "No trajectory data in 3D"
        X, Y, Z = sp._trajectory_pos
        assert X is not None and len(X) > 0, "No X data"
        assert Y is not None and len(Y) > 0, "No Y data"
        assert Z is not None and len(Z) > 0, "No Z data in 3D mode"
        assert np.any(np.isfinite(X))
        assert np.any(np.isfinite(Y))
        assert np.any(np.isfinite(Z)), "Z data is all NaN"
        assert sp.is_3d is True

        # Verify the GL widget and its container have Expanding policy
        # so the layout doesn't collapse the plot to zero pixels.
        from qtpy.QtWidgets import QSizePolicy

        w = sp.space_widget
        assert w.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding, (
            f"GLViewWidget horizontal policy is {w.sizePolicy().horizontalPolicy()}, "
            f"expected Expanding ({QSizePolicy.Expanding})"
        )
        holder = sp._plot_holder
        assert holder.sizePolicy().verticalPolicy() == QSizePolicy.Expanding, (
            "Plot holder vertical policy must be Expanding"
        )

        # Verify actual GL items are rendered on the 3D widget
        import pyqtgraph.opengl as gl

        gl_items = [
            item for item in sp.space_widget.items if isinstance(item, (gl.GLLinePlotItem, gl.GLScatterPlotItem))
        ]
        assert len(gl_items) > 0, (
            f"No GL items rendered on 3D space plot. Items: {[type(i).__name__ for i in sp.space_widget.items]}"
        )
        # Verify the trajectory line has valid (non-empty) position data
        trajectory_lines = [
            item for item in gl_items if isinstance(item, gl.GLLinePlotItem) and getattr(item, "_is_trajectory", False)
        ]
        assert len(trajectory_lines) > 0, "No trajectory GLLinePlotItem found"

        # Verify moll2025.yaml reference geometry is rendered (wireframe)
        non_trajectory = [
            item
            for item in sp.space_widget.items
            if isinstance(item, gl.GLLinePlotItem) and not getattr(item, "_is_trajectory", False)
        ]
        refs = sp._load_references()
        if refs:
            assert len(non_trajectory) > 0, (
                f"library geometry has {len(refs)} references but none rendered. "
                f"GL items: {[type(i).__name__ for i in sp.space_widget.items]}"
            )


# ===================================================================
# Lineplot
# ===================================================================


class TestMollLinePlot:
    def test_lineplot_has_data_on_trial(self, moll2025_gui):
        _, meta = moll2025_gui
        _navigate_to_trial(meta, TRIAL)

        from ethograph.utils.qt import set_combo_to_value

        set_combo_to_value(meta.data_widget.combos["features"], "speed")
        QApplication.processEvents()
        meta.data_widget.update_main_plot()
        QApplication.processEvents()

        lp = meta.plot_container.line_plots[0]
        assert len(lp.plot_items) > 0, "LinePlot has no plot items"

        import pyqtgraph as pg

        found = False
        for item in lp.plot_items:
            if isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
                x, y = item.getData()
                assert x is not None and len(x) > 0, "Curve has no x data"
                assert y is not None and len(y) > 0, "Curve has no y data"
                found = True
                break
            elif hasattr(item, "x") and hasattr(item, "y"):
                assert len(item.x) > 0, "Line has no x data"
                assert len(item.y) > 0, "Line has no y data"
                found = True
                break
        assert found, f"No recognizable plot item found. Types: {[type(i).__name__ for i in lp.plot_items]}"


# ===================================================================
# Pynapple-loaded Moll2025 — same tests on different backend
# ===================================================================


class TestMollPynappleLoading:
    def test_state_after_load(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui
        s = meta.app_state
        assert s.ready is True
        assert s.ds is not None

    def test_two_trials(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui
        assert len(meta.app_state.trials) == 2

    def test_features_available(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "beakTip_speed" in items
        assert "beakTip_position" in items

    def test_labels_df_exists(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui
        df = meta.app_state._all_labels_df
        assert df is not None


class TestMollPynappleLinePlot:
    def test_lineplot_has_data(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui

        from ethograph.utils.qt import set_combo_to_value

        set_combo_to_value(meta.data_widget.combos["features"], "beakTip_speed")
        QApplication.processEvents()
        meta.data_widget.update_main_plot()
        QApplication.processEvents()

        lp = meta.plot_container.line_plots[0]
        assert len(lp.plot_items) > 0, "LinePlot has no plot items"

        import pyqtgraph as pg

        for item in lp.plot_items:
            if isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
                x, y = item.getData()
                assert x is not None and len(x) > 0
                assert y is not None and len(y) > 0
                break

    def test_cycle_all_features(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui
        features_combo = meta.data_widget.combos["features"]
        lp = meta.plot_container.line_plots[0]
        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            assert len(lp.plot_items) > 0, f"Feature '{text}' produced no plot items"

    def test_all_checkbox_frees_the_column_dim(self, moll2025_pynapple_gui):
        """'All' on a multi-column pynapple feature draws one curve per column,
        on the panel already open — the xarray behaviour, same dim name."""
        import pyqtgraph as pg

        from ethograph.utils.qt import set_combo_to_value

        _, meta = moll2025_pynapple_gui
        dw = meta.data_widget
        lp = meta.plot_container.line_plots[0]

        set_combo_to_value(dw.combos["features"], "beakTip_position")
        QApplication.processEvents()

        def curves():
            return sum(1 for it in lp.plot_items if isinstance(it, (pg.PlotDataItem, pg.PlotCurveItem)))

        assert dw.all_checkboxes["space"].isChecked() is False
        assert curves() == 1

        dw.all_checkboxes["space"].setChecked(True)
        QApplication.processEvents()
        assert curves() == 3, "'All' did not free the column dim"
        assert "space" not in lp._effective_selections()


class TestMollPynappleSpacePlot:
    def test_space_2d_has_data(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui

        meta.app_state.space_plot_type = "Space Plot"
        if hasattr(meta.data_widget, "space_view_combo"):
            meta.data_widget.space_view_combo.setCurrentText("Space Plot")
            QApplication.processEvents()

        meta.data_widget.update_space_plot()
        QApplication.processEvents()

        sp = meta.data_widget.space_plot
        assert sp is not None, "SpacePlot not created"

        sp.cb_3d.setChecked(False)
        QApplication.processEvents()
        sp.refresh()
        QApplication.processEvents()

        assert sp._trajectory_pos is not None, "No trajectory data"
        X, Y, Z = sp._trajectory_pos
        assert X is not None and len(X) > 0
        assert Y is not None and len(Y) > 0
        assert np.any(np.isfinite(X))
        assert np.any(np.isfinite(Y))

        # Verify actual plot items are rendered
        import pyqtgraph as pg

        plot_item = sp.space_widget.getPlotItem()
        trajectory_items = [
            item
            for item in plot_item.items
            if isinstance(item, (pg.PlotCurveItem, pg.PlotDataItem)) or hasattr(item, "_is_trajectory")
        ]
        assert len(trajectory_items) > 0, "No trajectory items rendered on 2D space plot"

    def test_popup_offers_space_for_multicolumn_pynapple_feature(self, moll2025_pynapple_gui):
        """The add-panel popup must gate on the loader's shape, not app_state.ds
        — reading only the ds answered N=1 for every pynapple feature, hiding
        Heatmap/Space exactly there."""
        from ethograph.gui.source_popup import allowed_plot_types

        _, meta = moll2025_pynapple_gui
        options = allowed_plot_types("feature", "beakTip_position", meta.app_state)
        assert "Heatmap" in options
        assert "Space (2D)" in options
        assert "Space (3D)" in options  # (N, 3) → all three axes available
        # A 1-D feature still only offers a line plot.
        assert allowed_plot_types("feature", "beakTip_speed", meta.app_state) == ["Lineplot"]

    def test_dropping_a_pynapple_feature_creates_a_space_panel(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui

        n_before = len(meta.data_widget.space_plots)
        meta._create_panel_for_source("feature", "beakTip_position", "Space (2D)")
        QApplication.processEvents()

        assert len(meta.data_widget.space_plots) == n_before + 1
        sp = meta.data_widget.space_plots[-1]
        assert sp.feature_combo.currentText() == "beakTip_position"
        assert sp._trajectory_pos is not None, "No trajectory data"
        X, Y, _ = sp._trajectory_pos
        assert np.any(np.isfinite(X)) and np.any(np.isfinite(Y))


class TestMollPynappleLabellingWithoutChangepoints:
    def test_label_at_exact_click_times(self, moll2025_pynapple_gui):
        _, meta = moll2025_pynapple_gui

        lw = meta.labels_widget
        if not lw._mappings or 1 not in lw._mappings:
            pytest.skip("No label mapping 1")

        meta.changepoints_widget.changepoint_correction_checkbox.setChecked(False)
        QApplication.processEvents()

        t_start = 1.0
        t_end = 2.0
        lw.activate_label(1)
        lw._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
        assert lw.first_click == pytest.approx(t_start)

        lw._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty, "No label created"

        individual = lw._current_individual()
        idx = find_interval_at(df, (t_start + t_end) / 2, individual)
        assert idx is not None, "Label not found at midpoint"
        row = df.loc[idx]
        assert row["labels"] == 1
        assert row["onset_s"] == pytest.approx(t_start, abs=0.01)
        assert row["offset_s"] == pytest.approx(t_end, abs=0.01)


class TestMollPoseOverlay:
    """The pose stream is named for what it holds (``pose_2d``), not for its
    camera (``cam-1``). Asking only for the camera's own name made the whole
    dataset look pose-less: no overlay, no keypoints table, no skeleton."""

    def test_pose_device_pairs_by_index_on_name_mismatch(self, moll2025_gui):
        _, meta = moll2025_gui
        sio = meta.app_state.nwb_alignment
        # Precondition: the pose device is NOT the camera name.
        assert sio.cameras == ["cam-1"]
        assert "cam-1" not in sio.pose_keys

        pm = meta.data_widget.pose_mgr
        assert pm is not None
        assert pm._pose_device(0) == sio.pose_keys[0]

    def test_pose_overlay_renders(self, moll2025_gui):
        from qtpy.QtWidgets import QApplication

        _, meta = moll2025_gui
        pm = meta.data_widget.pose_mgr

        pr = pm._load_pose_for_camera(0)
        assert pr is not None and len(pr.data) > 0

        meta.data_widget.update_pose()
        QApplication.processEvents()
        assert meta.app_state.keypoints, "keypoints table never populated"
        assert meta.shell.video_area.primary._overlay is not None
