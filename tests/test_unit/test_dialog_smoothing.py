"""The smoothing dialog: its snippet reproduces the preview, stages run in
order, and the preview lives exactly as long as the dialog."""

from types import SimpleNamespace

import numpy as np
import pytest

from ethograph.features.preprocessing import bridge_gaps, gaussian, nan_aware
from ethograph.gui.dialog_smoothing import (
    STEPS,
    Companions,
    Pipeline,
    SmoothingDialog,
    Step,
    console_line,
    smoothing_snippet,
)
from ethograph.gui.plots_lineplot import LinePlot

GAUSS = Step("gaussian", {"sigma_s": 0.05})
BRIDGE = Step("bridge_gaps", {"which": "gaps up to max_gap_s", "max_gap_s": 0.2})
CONF = Step("filter_by_confidence", {"threshold": 0.6})


def test_every_step_links_its_documentation():
    for kind, spec in STEPS.items():
        assert spec.doc_url.startswith("https://"), kind
        assert spec.source is not None and spec.source().endswith(".py"), kind


def test_stages_run_in_order_whatever_order_steps_were_added():
    pipeline = Pipeline((GAUSS, BRIDGE, CONF))
    assert [s.kind for s in pipeline.steps] == ["filter_by_confidence", "bridge_gaps", "gaussian"]

    x = np.sin(np.arange(300) / 10.0)
    x[100:103] = np.nan
    conf = np.ones(300)
    conf[200:210] = 0.1
    stages = pipeline.run_stages(x, 30.0, Companions(confidence=conf))
    assert [label for label, _ in stages] == [STEPS[s.kind].label for s in pipeline.steps]
    filtered, bridged, smoothed = (arr for _, arr in stages)
    assert np.isnan(filtered[200:210]).all() and np.isfinite(filtered[:100]).all()
    assert np.isfinite(bridged[100:103]).all(), "the 0.1 s gap is bridged"
    assert np.isnan(bridged[200:210]).all(), "the 0.33 s gap stays"
    assert np.isnan(smoothed[200:210]).all(), "a kernel never fills a gap"
    np.testing.assert_array_equal(
        smoothed,
        nan_aware(gaussian)(bridge_gaps(filtered, 30.0, max_gap_s=0.2, axis=0), 30.0, axis=0, sigma_s=0.05),
    )


def test_snippet_spells_the_stages_as_the_preview_ran_them():
    text = smoothing_snippet(Pipeline((GAUSS, BRIDGE, CONF)), "position", {"keypoint": "beak"}, "xarray")
    lines = text.splitlines()
    assert "from movement.filtering import filter_by_confidence" in lines
    assert "from ethograph.features.preprocessing import bridge_gaps, gaussian, nan_aware" in lines
    body = [line for line in lines if line.startswith("x = ")]
    assert body == [
        'x = ds["position"]',
        'x = filter_by_confidence(x, ds["confidence"], threshold=0.6)',
        "x = eto.apply_along_time(x, bridge_gaps, max_gap_s=0.2)",
        "x = eto.apply_along_time(x, nan_aware(gaussian), sigma_s=0.05)",
    ]
    assert lines[-1] == 'ds["position_clean"] = x'
    assert "fs = " not in text, "no step needed a sample count"


def test_snippet_derives_sample_counts_from_the_rate_and_comments_movement_out_on_pynapple():
    rolling = Step("rolling_filter", {"window_s": 0.2, "statistic": "median"})
    text = smoothing_snippet(Pipeline((rolling, GAUSS)), "speed", {}, "pynapple")
    assert 'data, _ = eto.load_nap_data("session.nwb")' in text
    assert "fs = x.rate" in text
    assert "# x = rolling_filter(x, window=round(0.2 * fs), statistic='median')" in text
    assert "x = eto.apply_along_time(x, nan_aware(gaussian), sigma_s=0.05)" in text
    xr_text = smoothing_snippet(Pipeline((rolling,)), "speed", {}, "xarray")
    assert "fs = eto.sampling_rate(eto.get_time_coord(x).values)" in xr_text
    assert "x = rolling_filter(x, window=round(0.2 * fs), statistic='median')" in xr_text


def test_console_line_chains_our_steps_and_refuses_movement_ones():
    assert console_line(Pipeline((BRIDGE, GAUSS)), "speed") == (
        "speed_clean = apply_along_time(apply_along_time(speed, bridge_gaps, max_gap_s=0.2, fs=sampling_rate(t)), "
        "nan_aware(gaussian), sigma_s=0.05, fs=sampling_rate(t))"
    )
    assert console_line(Pipeline((CONF, GAUSS)), "speed") is None


def test_cache_round_trip_restores_the_pipeline_and_ignores_junk():
    pipeline = Pipeline((CONF, GAUSS))
    assert Pipeline.from_cache(pipeline.to_cache()) == pipeline
    assert Pipeline.from_cache({"steps": [{"kind": "nope"}]}).steps == ()
    assert Pipeline.from_cache(None) == Pipeline()


@pytest.fixture
def plot_and_dialog(qtbot, app_state):
    plot = LinePlot(app_state)
    qtbot.addWidget(plot)
    plot.panel_state["feature"] = "speed"
    dlg = SmoothingDialog(SimpleNamespace(active_panels=None, line_plots=[plot]), app_state)
    qtbot.addWidget(dlg)
    return plot, dlg


def test_preview_is_set_while_open_and_cleared_on_close(plot_and_dialog, app_state):
    plot, dlg = plot_and_dialog
    assert plot.preview is not None and plot.preview.pad_s > 0
    assert 'ds["speed_clean"]' in dlg.code_edit.toPlainText()
    assert app_state.function_params_cache["smoothing"]["steps"][0]["kind"] == "gaussian"
    dlg.close()
    assert plot.preview is None


def test_adding_and_removing_steps_updates_pipeline_lists_and_code(plot_and_dialog):
    plot, dlg = plot_and_dialog
    dlg._add_step("bridge_gaps")
    dlg._add_step("rolling_filter")
    assert [s.kind for s in dlg.pipeline.steps] == ["rolling_filter", "bridge_gaps", "gaussian"]
    assert dlg.boxes["filter"].list.count() == 1 and dlg.boxes["gaps"].list.count() == 1
    assert dlg.selected_step.kind == "rolling_filter"
    assert not dlg.console_btn.isEnabled(), "movement steps cannot go to the console"
    assert "rolling_filter(" in dlg.code_edit.toPlainText()

    stages = plot.preview.apply(np.random.default_rng(0).normal(size=300), np.arange(300) / 30.0)
    assert [label for label, _ in stages] == [STEPS[s.kind].label for s in dlg.pipeline.steps]

    dlg._remove_step(dlg.boxes["filter"])
    assert [s.kind for s in dlg.pipeline.steps] == ["bridge_gaps", "gaussian"]
    assert dlg.console_btn.isEnabled()
    dlg._remove_step(dlg.boxes["gaps"])
    dlg._remove_step(dlg.boxes["smooth"])
    assert dlg.pipeline.steps == () and plot.preview is None


def test_a_kernel_too_short_for_the_panel_reports_instead_of_drawing(plot_and_dialog):
    plot, dlg = plot_and_dialog
    dlg._set_param("sigma_s", 0.002)  # 0.06 samples at 30 Hz
    assert plot.preview.apply(np.zeros(100), np.arange(100) / 30.0) is None
    assert "30 Hz" in dlg.status_label.text()
    stages = plot.preview.apply(np.zeros(100), np.arange(100) / 1000.0)
    assert len(stages) == 1 and stages[0][1].shape == (100,)
