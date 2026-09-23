"""BirdPark integration tests: GUI lifecycle, combos, trial navigation,
labels, plot content, heatmap, pan/zoom, hidden panels, downsampling,
from_continuous slicing, and panel container basics."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pynwb
import pyqtgraph as pg
import pytest
from qtpy.QtWidgets import QApplication

from ethograph.io.trialtree import TrialTree

# ---------------------------------------------------------------------------
# Plot content helpers
# ---------------------------------------------------------------------------


def _get_curve_data(plot_items):
    for item in plot_items:
        if isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
            return item.xData, item.yData
    return None, None


def _assert_lineplot_has_data(line_plot):
    assert len(line_plot.plot_items) > 0, "LinePlot.plot_items is empty"
    x, y = _get_curve_data(line_plot.plot_items)
    assert x is not None and len(x) > 0, "LinePlot curve has no x data"
    assert y is not None and len(y) > 0, "LinePlot curve has no y data"


def _assert_audio_has_data(audio_plot):
    assert audio_plot.trace_item.xData is not None, "AudioTrace xData is None"
    assert len(audio_plot.trace_item.xData) > 0, "AudioTrace xData is empty"
    assert audio_plot.trace_item.yData is not None, "AudioTrace yData is None"
    assert len(audio_plot.trace_item.yData) > 0, "AudioTrace yData is empty"


def _assert_spectrogram_has_data(spec_plot):
    assert spec_plot.spec_item.image is not None, "Spectrogram image is None"
    assert spec_plot.spec_item.image.size > 0, "Spectrogram image is empty"


# ===================================================================
# Widget creation (no data loaded)
# ===================================================================


class TestMetaWidgetCreation:
    def test_widget_initialization(self, gui):
        _, meta = gui
        for attr in (
            "app_state",
            "plot_container",
            "data_widget",
            "io_widget",
            "labels_widget",
            "navigation_widget",
            "changepoints_widget",
            "plot_settings_widget",
            "ephys_widget",
        ):
            assert getattr(meta, attr) is not None

        assert meta.app_state.ready is False
        assert meta.app_state.trial_changed is not None


# ===================================================================
# Template loading basics
# ===================================================================


class TestBirdParkLoading:
    def test_state_after_load(self, birdpark_gui):
        _, meta = birdpark_gui
        state = meta.app_state
        assert state.ready is True
        assert state.dt is not None
        assert state.ds is not None
        assert len(state.trials) > 0
        assert state.trials_sel in state.trials
        assert state.time_coord is not None
        assert len(state.time_coord) > 0
        cat = meta.data_widget.catalog
        assert cat is not None
        assert len(cat.features) > 0

    def test_combos_populated_after_load(self, birdpark_gui):
        _, meta = birdpark_gui
        features_combo = meta.data_widget.combos.get("features")
        assert features_combo is not None
        assert features_combo.count() > 0
        ind_combo = meta.data_widget.combos.get("individuals")
        if ind_combo is None:
            ind_combo = meta.io_widget.combos.get("individuals")
        assert ind_combo is not None
        assert ind_combo.count() > 0
        assert meta.navigation_widget.trials_combo.count() > 0

    def test_audio_panels_visible(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        assert pc.audio_trace_plots or pc.spectrogram_plots

    def test_spectrogram_panel_exists(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        assert pc.spectrogram_plot is not None
        assert pc.spectrogram_plot.parent() is not None

    def test_audio_trace_panel_exists(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        assert pc.audio_trace_plot is not None
        assert pc.audio_trace_plot.parent() is not None

    def test_select_accelerometer(self, birdpark_gui):
        _, meta = birdpark_gui
        combo = meta.data_widget.combos["features"]
        idx = next(i for i in range(combo.count()) if combo.itemText(i) == "vibration")
        combo.setCurrentIndex(idx)
        QApplication.processEvents()
        assert meta.app_state.features_sel == "vibration"

    def test_has_audio_flag(self, birdpark_gui):
        _, meta = birdpark_gui
        assert meta.app_state.has_audio is True

    def test_feature_plot_defaults_to_lineplot(self, birdpark_gui):
        _, meta = birdpark_gui
        assert meta.plot_container.is_lineplot()


# ===================================================================
# Trial navigation
# ===================================================================


class TestTrialNavigation:
    def test_prev_trial_at_start_stays(self, birdpark_gui):
        _, meta = birdpark_gui
        meta.navigation_widget.scope_combo.setCurrentText("Trial start → Trial end")
        QApplication.processEvents()
        first_trial = meta.app_state.trials[0]
        meta.navigation_widget.trials_combo.setCurrentText(str(first_trial))
        QApplication.processEvents()
        meta.navigation_widget.prev_trial()
        QApplication.processEvents()
        assert meta.app_state.trials_sel == first_trial


# ===================================================================
# Labels (basic — no mappings needed)
# ===================================================================


class TestLabelsWidget:
    def test_changes_saved_initially_true(self, birdpark_gui):
        _, meta = birdpark_gui
        assert meta.app_state.changes_saved is True

    def test_curate_trial_leaves_no_automated_label(self, birdpark_gui):
        _, meta = birdpark_gui
        meta.labels_widget.curation_panel.curate_current_trial()
        QApplication.processEvents()
        df = meta.app_state._all_labels_df
        trial = meta.app_state.trials_sel
        if df is not None and not df.empty:
            trial_rows = df[df["trial"] == trial]
            assert (trial_rows["labeling_method"] != "automated").all()
            assert meta.app_state.trial_is_curated(trial)

    def test_label_creation_via_two_clicks(self, birdpark_gui):
        """Two-click state labelling on the xarray backend.

        test_moll2025 covers the same flow on pynapple only; the click →
        ``from_display`` → ``_apply_label`` chain is backend-sensitive, so
        both backends need the regression pinned.
        """
        from qtpy.QtCore import Qt

        from ethograph.labels.intervals import find_interval_at

        _, meta = birdpark_gui
        lw = meta.labels_widget
        t_start, t_end = 0.5, 1.0

        lw.activate_label(1)
        assert lw.ready_for_label_click is True

        lw._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
        assert lw.first_click == pytest.approx(t_start)

        lw._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty, "No intervals after label creation"
        idx = find_interval_at(df, (t_start + t_end) / 2, lw._current_individual())
        assert idx is not None, "Interval not found at midpoint"
        row = df.loc[idx]
        assert row["labels"] == 1
        assert row["onset_s"] == pytest.approx(t_start, abs=0.01)
        assert row["offset_s"] == pytest.approx(t_end, abs=0.01)


# ===================================================================
# Downsampled data
# ===================================================================


class TestDownsampledData:
    DOWNSAMPLE_FACTOR = 100

    def test_downsample_by_100(self, birdpark_gui_downsampled):
        _, meta = birdpark_gui_downsampled
        assert meta.app_state.ready is True
        assert meta.app_state.downsample_factor_used == self.DOWNSAMPLE_FACTOR
        assert meta.app_state.dt is not None

        attrs = meta.app_state.ds.attrs
        assert attrs["downsample_factor"] == self.DOWNSAMPLE_FACTOR

        n_time = len(meta.app_state.time_coord.values)
        original_approx = (n_time // 2) * self.DOWNSAMPLE_FACTOR
        assert n_time < original_approx

        assert not meta.io_widget.downsample_checkbox.isEnabled()
        assert not meta.io_widget.downsample_spin.isEnabled()

        features_combo = meta.data_widget.combos.get("features")
        assert features_combo is not None
        assert features_combo.count() > 0

        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            assert meta.app_state.features_sel == text

        assert meta.plot_container.is_lineplot()
        xlim = meta.plot_container.get_current_xlim()
        assert xlim[0] < xlim[1]


# ===================================================================
# Plot content after load
# ===================================================================


class TestPlotPopulatedAfterLoad:
    def test_lineplot_has_data(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        assert pc._feature_type == "lineplot"
        _assert_lineplot_has_data(pc.line_plots[0])

    def test_lineplot_x_within_reasonable_range(self, birdpark_gui):
        _, meta = birdpark_gui
        lp = meta.plot_container.line_plots[0]
        x, _ = _get_curve_data(lp.plot_items)
        xlim = lp.get_current_xlim()
        assert x[0] <= xlim[1], "Curve starts after visible range"
        assert x[-1] >= xlim[0], "Curve ends before visible range"

    def test_lineplot_y_is_finite(self, birdpark_gui):
        _, meta = birdpark_gui
        lp = meta.plot_container.line_plots[0]
        _, y = _get_curve_data(lp.plot_items)
        assert np.all(np.isfinite(y)), "LinePlot y contains inf/nan"

    def test_audio_trace_has_data_if_visible(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        if not pc.audio_trace_plots:
            pytest.skip("Audio panel not visible for this dataset")
        _assert_audio_has_data(pc.audio_trace_plot)

    def test_spectrogram_has_data_if_visible(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        if not pc.spectrogram_plots:
            pytest.skip("Spectrogram panel not visible for this dataset")
        _assert_spectrogram_has_data(pc.spectrogram_plot)


# ===================================================================
# Trial switch updates plot
# ===================================================================


class TestTrialSwitchUpdatesPlot:
    def test_lineplot_data_changes_on_trial_switch(self, moll2025_gui):
        _, meta = moll2025_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need 2+ trials")
        lp = meta.plot_container.line_plots[0]
        x1, y1 = _get_curve_data(lp.plot_items)
        x1, y1 = x1.copy(), y1.copy()
        meta.navigation_widget.next_trial()
        QApplication.processEvents()
        x2, y2 = _get_curve_data(lp.plot_items)
        assert x2 is not None and len(x2) > 0, "LinePlot empty after trial switch"
        changed = (not np.array_equal(x1, x2)) or (not np.array_equal(y1, y2))
        assert changed, "LinePlot data identical after trial switch"

    def test_lineplot_has_data_every_trial(self, moll2025_gui):
        _, meta = moll2025_gui
        lp = meta.plot_container.line_plots[0]
        for trial in meta.app_state.trials:
            meta.navigation_widget.trials_combo.setCurrentText(str(trial))
            QApplication.processEvents()
            assert len(lp.plot_items) > 0, f"LinePlot empty on trial {trial}"
            x, y = _get_curve_data(lp.plot_items)
            assert x is not None and len(x) > 0, f"No curve data on trial {trial}"

    def test_xlim_updates_on_trial_switch(self, moll2025_gui):
        _, meta = moll2025_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need 2+ trials")
        meta.navigation_widget.next_trial()
        QApplication.processEvents()
        xlim2 = meta.plot_container.get_current_xlim()
        assert xlim2[0] < xlim2[1], "Invalid xlim after trial switch"


# ===================================================================
# Feature switch updates plot
# ===================================================================


class TestFeatureSwitchUpdatesPlot:
    def test_switching_feature_changes_plot_data(self, moll2025_gui):
        _, meta = moll2025_gui
        features_combo = meta.data_widget.combos["features"]
        if features_combo.count() < 2:
            pytest.skip("Need 2+ features")
        lp = meta.plot_container.line_plots[0]
        features_combo.setCurrentIndex(0)
        QApplication.processEvents()
        if features_combo.currentText() in ("Spectrogram", "Waveform"):
            features_combo.setCurrentIndex(1)
            QApplication.processEvents()
        _, y1 = _get_curve_data(lp.plot_items)
        if y1 is not None:
            y1 = y1.copy()
        switched = False
        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            if i == features_combo.currentIndex():
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            switched = True
            break
        if not switched:
            pytest.skip("Could not find a second plottable feature")
        _, y2 = _get_curve_data(lp.plot_items)
        assert y2 is not None and len(y2) > 0, "LinePlot empty after feature switch"

    def test_every_feature_renders_data(self, birdpark_gui):
        _, meta = birdpark_gui
        features_combo = meta.data_widget.combos["features"]
        lp = meta.plot_container.line_plots[0]
        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            assert len(lp.plot_items) > 0, f"Feature '{text}' produced no plot items"
            x, y = _get_curve_data(lp.plot_items)
            assert x is not None and len(x) > 0, f"Feature '{text}' has empty curve"


# ===================================================================
# Heatmap mode
# ===================================================================


class TestHeatmapMode:
    def test_switch_to_heatmap_renders_image(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        pc.set_feature_view("heatmap")
        QApplication.processEvents()
        meta.data_widget.update_main_plot()
        QApplication.processEvents()
        assert pc._feature_type == "heatmap"
        hm = pc.heatmap_plot
        assert hm.image_item.image is not None, "Heatmap image is None after switch"
        assert hm.image_item.image.size > 0, "Heatmap image is empty"

    def test_switch_back_to_lineplot(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        pc.set_feature_view("heatmap")
        QApplication.processEvents()
        meta.data_widget.update_main_plot()
        QApplication.processEvents()
        pc.set_feature_view("lineplot")
        QApplication.processEvents()
        meta.data_widget.update_main_plot()
        QApplication.processEvents()
        assert pc._feature_type == "lineplot"
        _assert_lineplot_has_data(pc.line_plots[0])

    def test_switch_to_heatmap_and_back_via_container(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        pc.switch_to_heatmap()
        QApplication.processEvents()
        assert pc.is_heatmap()
        pc.switch_to_lineplot()
        QApplication.processEvents()
        assert pc.is_lineplot()


# ===================================================================
# Pan / zoom — buffer reload
# ===================================================================


class TestPanZoom:
    def test_zoom_in_preserves_data(self, birdpark_gui):
        _, meta = birdpark_gui
        lp = meta.plot_container.line_plots[0]
        xlim = lp.get_current_xlim()
        mid = (xlim[0] + xlim[1]) / 2
        quarter = (xlim[1] - xlim[0]) / 4
        lp.plot_item.setXRange(mid - quarter, mid + quarter)
        QApplication.processEvents()
        new_xlim = lp.get_current_xlim()
        assert new_xlim[1] - new_xlim[0] < xlim[1] - xlim[0], "Zoom didn't narrow range"
        assert len(lp.plot_items) > 0, "Plot items gone after zoom"

    def test_zoom_out_preserves_data(self, birdpark_gui):
        _, meta = birdpark_gui
        lp = meta.plot_container.line_plots[0]
        xlim = lp.get_current_xlim()
        span = xlim[1] - xlim[0]
        mid = (xlim[0] + xlim[1]) / 2
        lp.plot_item.setXRange(mid - span, mid + span)
        QApplication.processEvents()
        assert len(lp.plot_items) > 0, "Plot items gone after zoom out"

    def test_pan_right_preserves_data(self, birdpark_gui):
        _, meta = birdpark_gui
        lp = meta.plot_container.line_plots[0]
        xlim = lp.get_current_xlim()
        span = xlim[1] - xlim[0]
        shift = span * 0.3
        lp.plot_item.setXRange(xlim[0] + shift, xlim[1] + shift)
        QApplication.processEvents()
        assert len(lp.plot_items) > 0, "Plot items gone after pan"

    def test_pan_triggers_buffer_update(self, birdpark_gui):
        _, meta = birdpark_gui
        lp = meta.plot_container.line_plots[0]
        xlim = lp.get_current_xlim()
        trial_dur = xlim[1] - xlim[0]
        if trial_dur < 0.5:
            pytest.skip("Trial too short to test buffer reload")
        t_end = xlim[1]
        t_start = t_end - trial_dur * 0.2
        lp.plot_item.setXRange(t_start, t_end)
        QApplication.processEvents()
        lp.update_plot_content(t_start, t_end)
        QApplication.processEvents()
        x, y = _get_curve_data(lp.plot_items)
        assert x is not None and len(x) > 0, "No data after panning to end of trial"
        assert x[-1] >= t_start, "Data doesn't cover panned region"


# ===================================================================
# Time axes alignment
# ===================================================================


class TestTimeAxesAlignment:
    def test_time_axes_aligned(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        feature_xlim = pc._feature_plot.get_current_xlim()
        assert feature_xlim[0] < feature_xlim[1]
        if pc.audio_trace_plots:
            audio_xlim = pc.audio_trace_plot.get_current_xlim()
            assert abs(audio_xlim[0] - feature_xlim[0]) < 0.5
            assert abs(audio_xlim[1] - feature_xlim[1]) < 0.5
        if pc.spectrogram_plots:
            spec_xlim = pc.spectrogram_plot.get_current_xlim()
            assert abs(spec_xlim[0] - feature_xlim[0]) < 0.5
            assert abs(spec_xlim[1] - feature_xlim[1]) < 0.5

    def test_time_marker_syncs_across_panels(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        t = 1.0
        for plot in [pc._feature_plot, pc.audio_trace_plot, pc.spectrogram_plot]:
            if plot is not None:
                plot.update_time_marker(t)
        if pc.audio_trace_plots:
            assert pc.audio_trace_plot.time_marker.value() == pytest.approx(t)
        if pc.spectrogram_plots:
            assert pc.spectrogram_plot.time_marker.value() == pytest.approx(t)
        assert pc._feature_plot.time_marker.value() == pytest.approx(t)


# ===================================================================
# Hidden panels
# ===================================================================


class TestHiddenPanelsNoData:
    def _toggle_panel(self, meta, name, checked):
        # Panels are layout instances: "off" removes every instance (each
        # panel's ✕), "on" recreates one (add-panel popup drop) and re-wires
        # the audio source via update_audio_panels.
        pc = meta.plot_container
        setter = {
            "audiotrace": pc.set_audiotrace_visible,
            "spectrogram": pc.set_spectrogram_visible,
        }[name]
        setter(checked)
        if checked:
            pc.update_audio_panels()
        QApplication.processEvents()

    def test_audiotrace_hidden_removes_instances(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        assert pc.audio_trace_plots
        assert pc.audio_trace_plot.source is not None
        self._toggle_panel(meta, "audiotrace", False)
        assert pc.audio_trace_plots == []
        assert pc.audio_trace_plot is None

    def test_audiotrace_show_restores_source(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        self._toggle_panel(meta, "audiotrace", False)
        assert pc.audio_trace_plot is None
        self._toggle_panel(meta, "audiotrace", True)
        assert pc.audio_trace_plot.source is not None

    def test_spectrogram_hidden_removes_instances(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        assert pc.spectrogram_plots
        assert pc.spectrogram_plot.source is not None
        self._toggle_panel(meta, "spectrogram", False)
        assert pc.spectrogram_plots == []
        assert pc.spectrogram_plot is None

    def test_spectrogram_show_restores_source(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        self._toggle_panel(meta, "spectrogram", False)
        assert pc.spectrogram_plot is None
        self._toggle_panel(meta, "spectrogram", True)
        assert pc.spectrogram_plot.source is not None

    def test_duplicate_audio_panels_allowed(self, no_video_gui):
        # Duplicates are never prevented — every add creates a new instance,
        # even for the same mic/channel.
        _, meta = no_video_gui
        pc = meta.plot_container
        n_spec = len(pc.spectrogram_plots)
        n_audio = len(pc.audio_trace_plots)
        pc.add_audio_panel("spectrogram")
        pc.add_audio_panel("spectrogram")
        pc.add_audio_panel("audiotrace")
        QApplication.processEvents()
        assert len(pc.spectrogram_plots) == n_spec + 2
        assert len(pc.audio_trace_plots) == n_audio + 1
        for plot in pc.spectrogram_plots + pc.audio_trace_plots:
            assert plot.source is not None
        for plot in pc.spectrogram_plots[n_spec:] + pc.audio_trace_plots[n_audio:]:
            pc.remove_audio_panel(plot)
        QApplication.processEvents()
        assert len(pc.spectrogram_plots) == n_spec
        assert len(pc.audio_trace_plots) == n_audio

    def test_neo_panel_removed_clears_source(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        plot = pc.add_panel("neo", stream_name="fake-stream", channels=None)
        assert plot in pc.neo_trace_plots
        fake_loader = MagicMock()
        fake_loader.rate = 30000.0
        fake_loader.n_channels = 4
        fake_loader.__len__ = lambda self: 1000
        plot.buffer.loader = fake_loader
        plot.set_source(MagicMock())
        pc.remove_panel(plot)
        assert plot not in pc.neo_trace_plots
        assert plot._source is None

    def test_neo_panel_no_update_on_xrange(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        plot = pc.add_panel("neo", stream_name="fake-stream")
        plot.hide()
        plot._on_view_range_changed()
        pc.remove_panel(plot)

    def test_ephys_hidden_clears_loader(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        fake_loader = MagicMock()
        fake_loader.rate = 30000.0
        fake_loader.n_channels = 4
        fake_loader.__len__ = lambda self: 1000
        pc.ephys_trace_plot.buffer.loader = fake_loader
        pc._panel_visible["ephys"] = True
        pc.set_ephys_visible(False)
        assert not pc._panel_visible["ephys"]
        assert pc.ephys_trace_plot.buffer.loader is None
        assert pc.ephys_trace_plot._source is None

    def test_all_lineplots_removable(self, birdpark_gui):
        _, meta = birdpark_gui
        pc = meta.plot_container
        for plot in list(pc.line_plots):
            pc.remove_lineplot(plot)
        QApplication.processEvents()
        assert pc.line_plots == []

    def test_update_audio_panels_does_not_recreate(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        self._toggle_panel(meta, "audiotrace", False)
        self._toggle_panel(meta, "spectrogram", False)
        pc.update_audio_panels()
        assert pc.audio_trace_plots == []
        assert pc.spectrogram_plots == []


# ===================================================================
# Panel container basics
# ===================================================================


class TestPanelContainerBasics:
    def test_container_is_unified(self, birdpark_gui):
        from ethograph.gui.plots_container import UnifiedPanelContainer

        _, meta = birdpark_gui
        assert isinstance(meta.plot_container, UnifiedPanelContainer)

    def test_valid_x_range(self, birdpark_gui):
        _, meta = birdpark_gui
        xlim = meta.plot_container.get_current_xlim()
        assert xlim[0] < xlim[1]
        assert xlim[0] >= -1.0
        assert xlim[1] > 0.1

    def test_default_lineplot_with_valid_range(self, birdpark_gui):
        _, meta = birdpark_gui
        assert meta.plot_container.is_lineplot()
        xlim = meta.plot_container.get_current_xlim()
        assert len(xlim) == 2
        assert xlim[0] < xlim[1]

    def test_get_current_plot_returns_feature(self, birdpark_gui):
        _, meta = birdpark_gui
        plot = meta.plot_container.get_current_plot()
        assert plot is not None
        assert plot is meta.plot_container._feature_plot

    def test_toggle_axes_lock(self, birdpark_gui):
        _, meta = birdpark_gui
        meta.plot_container.toggle_axes_lock()
        QApplication.processEvents()
        meta.plot_container.toggle_axes_lock()
        QApplication.processEvents()


# ===================================================================
# from_continuous slicing (real BirdPark data)
# ===================================================================

_VIDEO_NAME = "BP_2021-05-25_08-12-51_655154_0380000.mp4"
_AUDIO_NAME = "BP_2021-05-25_08-12-51_655154_0380000.wav"
_FPS = 47.68


def _make_epochs(n_trials: int = 3, chunk: float = 20.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": list(range(1, n_trials + 1)),
            "start_time": [i * chunk for i in range(n_trials)],
            "stop_time": [(i + 1) * chunk for i in range(n_trials)],
        }
    )


def _make_alignment_nwb(epochs: pd.DataFrame, output_dir: Path) -> Path:
    """Build an alignment NWB for session-wide continuous media.

    One video/audio file spans the whole session.  Each trial's
    ``starting_frame`` points into that single continuous stream so
    ``stream_offset_for_trial`` returns a negative offset for trials
    after the first.
    """
    from datetime import datetime
    from uuid import uuid4

    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO
    from pynwb.image import ImageSeries

    nwbfile = pynwb.NWBFile(
        session_description="NWB file for media alignment (ethograph generated).",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    nwbfile.add_trial_column(name="trial", description="Trial number")
    nwbfile.add_trial_column(name="video_cam-1", description="video filename")
    nwbfile.add_trial_column(name="audio_mic-1", description="audio filename")
    for _, row in epochs.iterrows():
        nwbfile.add_trial(
            start_time=float(row["start_time"]),
            stop_time=float(row["stop_time"]),
            trial=row["trial"],
            **{"video_cam-1": _VIDEO_NAME, "audio_mic-1": _AUDIO_NAME},
        )

    session_end = float(epochs["stop_time"].max())
    n_video_frames = int(session_end * _FPS)
    video_ts = np.arange(n_video_frames) / _FPS

    nwbfile.create_device(name="cam-1", description="video device cam-1")
    nwbfile.add_acquisition(
        ImageSeries(
            name="video_cam-1",
            description="video from cam-1",
            external_file=[_VIDEO_NAME],
            format="external",
            starting_frame=np.array([0], dtype=np.int32),
            timestamps=video_ts,
        )
    )

    nwb_path = output_dir / ".ethograph" / "alignment.nwb"
    nwb_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)
    return nwb_path


class TestFromContinuous:
    def test_continuous_trial_slicing(self, birdpark_gui):
        _, meta = birdpark_gui
        ds = meta.app_state.dt.itrial(0)
        epochs = _make_epochs()
        dt = TrialTree.from_continuous(ds, epochs)
        assert dt._is_continuous
        assert dt.trials == [1, 2, 3]
        ds1, ds2, ds3 = dt.trial(1), dt.trial(2), dt.trial(3)
        for i, trial_ds in enumerate([ds1, ds2, ds3], 1):
            t0 = float(trial_ds.time.values[0])
            assert abs(t0) < 0.02, f"Trial {i} time should start near 0, got {t0}"
            assert trial_ds.attrs["trial"] == i
        for trial_ds in [ds1, ds2, ds3]:
            duration = float(trial_ds.time.values[-1])
            assert 19.0 < duration < 20.5, f"Expected ~20s, got {duration:.1f}s"
        assert not np.allclose(ds1["vibration"].values[:10], ds2["vibration"].values[:10])

    def test_nwb_alignment_with_continuous(self, birdpark_gui, tmp_path):
        _, meta = birdpark_gui
        ds = meta.app_state.dt.itrial(0)
        epochs = _make_epochs()
        dt = TrialTree.from_continuous(ds, epochs)
        nwb_path = _make_alignment_nwb(epochs, tmp_path)
        from ethograph.io.nwb_alignment import make_nwb_alignment

        dt.nwb_alignment = make_nwb_alignment(nwb_path)
        sio = dt.nwb_alignment
        assert sio.cameras == ["cam-1"]
        assert sio.mics == ["mic-1"]
        assert sio.start_time(1) == 0.0
        assert sio.start_time(2) == 20.0
        assert sio.start_time(3) == 40.0
        assert sio.stop_time(1) == 20.0
        assert sio.get_media(1, "video", "cam-1") == _VIDEO_NAME
        assert sio.get_media(2, "audio", "mic-1") == _AUDIO_NAME
        offset_t2 = sio.stream_offset_for_trial(2, "video", "cam-1")
        assert abs(offset_t2 - (-20.0)) < 0.1
        detected_fps = sio.get_stream_rate("video", "cam-1")
        assert detected_fps is not None
        assert abs(detected_fps - _FPS) < 0.01

    def test_itrial_and_trial_items(self, birdpark_gui):
        _, meta = birdpark_gui
        ds = meta.app_state.dt.itrial(0)
        epochs = _make_epochs()
        dt = TrialTree.from_continuous(ds, epochs)
        ds_i = dt.itrial(0)
        ds_t = dt.trial(1)
        np.testing.assert_allclose(ds_i.time.values, ds_t.time.values)
        items = list(dt.trial_items())
        assert len(items) == 3
        assert items[0][0] == 1
        assert items[2][0] == 3

    def test_pynapple_epochs(self, birdpark_gui):
        import pynapple as nap

        _, meta = birdpark_gui
        ds = meta.app_state.dt.itrial(0)
        ep = nap.IntervalSet(start=[0.01, 20.01, 40.01], end=[19.99, 39.99, 59.99])
        dt = TrialTree.from_continuous(ds, ep)
        assert dt.trials == [1, 2, 3]
        ds2 = dt.trial(2)
        assert abs(float(ds2.time.values[0])) < 0.02

    def test_update_trial_raises(self, birdpark_gui):
        _, meta = birdpark_gui
        ds = meta.app_state.dt.itrial(0)
        epochs = _make_epochs()
        dt = TrialTree.from_continuous(ds, epochs)
        with pytest.raises(TypeError, match="continuous"):
            dt.update_trial(1, lambda d: d)

    def test_continuous_xarray_and_nwb_together(self, birdpark_gui, tmp_path):
        _, meta = birdpark_gui
        ds = meta.app_state.dt.itrial(0)
        epochs = _make_epochs()
        dt = TrialTree.from_continuous(ds, epochs)
        nwb_path = _make_alignment_nwb(epochs, tmp_path)
        from ethograph.io.nwb_alignment import make_nwb_alignment

        dt.nwb_alignment = make_nwb_alignment(nwb_path)
        for trial_id in [1, 2, 3]:
            trial_ds = dt.trial(trial_id)
            t_start = float(trial_ds.time.values[0])
            t_end = float(trial_ds.time.values[-1])
            assert abs(t_start) < 0.02
            assert 19.0 < t_end < 20.5
            nwb_start = dt.nwb_alignment.start_time(trial_id)
            nwb_stop = dt.nwb_alignment.stop_time(trial_id)
            assert abs(nwb_stop - nwb_start - 20.0) < 0.01
            v_offset = dt.nwb_alignment.stream_offset_for_trial(trial_id, "video", "cam-1")
            expected_offset = 0.0 - nwb_start
            assert abs(v_offset - expected_offset) < 0.1
        ds2 = dt.trial(2)
        orig_slice = ds.sel(time=slice(20.0, 40.0))
        np.testing.assert_allclose(ds2["vibration"].values, orig_slice["vibration"].values)
