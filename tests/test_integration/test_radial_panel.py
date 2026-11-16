"""The radial plot as a panel: offered only for headings, arrow follows time.

Exercised through a console-derived heading, which is the realistic route —
raw datasets rarely ship a full-turn variable, and it also proves the gate
reads the data rather than the variable's name or metadata.
"""

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from ethograph.gui.source_popup import allowed_plot_types
from ethograph.io.time_model import TimeRange


def _sampled(rp, n: int = 40):
    """(time, heading) across the window — real data starts with NaN gaps, so
    a test must look for a finite sample rather than assume one at t0."""
    bounds = rp.app_state.window_bounds
    for t in np.linspace(bounds.start_s, bounds.end_s, n):
        rp.set_time(float(t))
        yield float(t), rp.current_value()


def _heading(meta) -> str:
    """Derive a full-turn variable in the console and return its name.

    The sweep is built from the panel's own time vector rather than from its
    feature: a feature can be constant over the window (its span is then 0 and
    the mapping is all-NaN), and which feature a panel opens on is a property
    of the dataset, not of what these tests are about. Time always sweeps.
    """
    plot = meta.plot_container.line_plots[0]
    meta.plot_container.active_feature_plot = plot
    meta._add_console_panel()
    console = meta.plot_container.console_panel
    console._on_submitted("lo = np.nanmin(t) * np.ones_like(t)")
    console._on_submitted("span = (np.nanmax(t) - np.nanmin(t)) * np.ones_like(t)")
    console._on_submitted("hd = (t - lo) / span * 360.0")
    QApplication.processEvents()
    assert meta.app_state.data_loader.is_derived("hd"), console.output.toPlainText()
    return "hd"


def test_radial_is_offered_when_the_window_holds_only_part_of_the_turn(moll2025_gui, monkeypatch):
    """Whether a variable is a heading is a property of the variable, not of
    the viewport — a window holding a quarter turn must not hide the option."""
    _, meta = moll2025_gui
    heading = _heading(meta)
    assert "Radial" in allowed_plot_types("feature", heading, meta.app_state)

    bounds = meta.app_state.window_bounds
    quarter = TimeRange(bounds.start_s, bounds.start_s + (bounds.end_s - bounds.start_s) * 0.25)
    monkeypatch.setattr(type(meta.app_state), "window_bounds", property(lambda _self: quarter))
    assert meta.app_state.window_bounds is quarter

    assert "Radial" in allowed_plot_types("feature", heading, meta.app_state)


def test_the_arrow_survives_zooming_into_part_of_the_turn(moll2025_gui, monkeypatch):
    _, meta = moll2025_gui
    heading = _heading(meta)
    meta._create_panel_for_source("feature", heading, "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]
    assert rp._unit == "deg"

    bounds = meta.app_state.window_bounds
    quarter = TimeRange(bounds.start_s, bounds.start_s + (bounds.end_s - bounds.start_s) * 0.25)
    monkeypatch.setattr(type(meta.app_state), "window_bounds", property(lambda _self: quarter))
    rp._invalidate()
    rp.refresh()

    assert rp._unit == "deg", "zooming in made the plot forget it was angular"
    assert any(v is not None for _, v in _sampled(rp)), "no heading anywhere after zoom"


def test_radial_is_offered_only_for_a_full_turn_variable(moll2025_gui):
    _, meta = moll2025_gui
    heading = _heading(meta)
    plain = meta.plot_container.line_plots[0]._effective_feature()

    assert "Radial" in allowed_plot_types("feature", heading, meta.app_state)
    assert "Radial" not in allowed_plot_types("feature", plain, meta.app_state)


def test_dropping_a_heading_creates_a_radial_panel(moll2025_gui):
    _, meta = moll2025_gui
    heading = _heading(meta)

    n_before = len(meta.data_widget.radial_plots)
    meta._create_panel_for_source("feature", heading, "Radial")
    QApplication.processEvents()

    assert len(meta.data_widget.radial_plots) == n_before + 1
    rp = meta.data_widget.radial_plots[-1]
    assert rp.feature_combo.currentText() == heading
    assert rp._unit == "deg"
    assert rp.dock_widget is not None


def test_the_arrow_follows_the_time_marker(moll2025_gui):
    _, meta = moll2025_gui
    heading = _heading(meta)
    meta._create_panel_for_source("feature", heading, "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]

    values = [(t, v) for t, v in _sampled(rp) if v is not None]
    assert len(values) >= 2, "no finite heading anywhere in the window"
    assert values[0][1] != values[-1][1], "the heading never changes"
    rp.set_time(values[-1][0])
    assert rp._arrow_items, "no arrow drawn"


def test_up_decides_which_value_points_up(moll2025_gui):
    _, meta = moll2025_gui
    heading = _heading(meta)
    meta._create_panel_for_source("feature", heading, "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]

    value = next(v for _, v in _sampled(rp) if v is not None)

    rp.up_spin.setValue(value)
    QApplication.processEvents()
    # abs=0.1: the Up spin box holds one decimal, so it cannot store the
    # sample exactly — the arrow lands within a tenth of a degree of up.
    assert rp._screen_angle(value) == pytest.approx(90.0, abs=0.1), "the value set as Up is not pointing up"


def test_the_sidebar_shows_the_radial_context(moll2025_gui):
    from ethograph.gui.right_context import _CONTEXT_MAP

    _, meta = moll2025_gui
    heading = _heading(meta)
    meta._create_panel_for_source("feature", heading, "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]

    assert _CONTEXT_MAP["radial"] == ["radialplot"]
    assert meta.context_panel.current_context() == "radial"
    # The instance's own controls are the ones on show.
    assert rp.controls_widget.isVisibleTo(meta.context_panel)


def test_settings_round_trip_through_the_saved_layout(moll2025_gui):
    _, meta = moll2025_gui
    heading = _heading(meta)
    meta._create_panel_for_source("feature", heading, "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]
    rp.up_spin.setValue(45.0)
    rp.cw_check.setChecked(True)

    state = meta.data_widget.radial_layout_state()
    assert state[-1] == {"feature": heading, "selections": {}, "up": 45.0, "clockwise": True}

    meta.data_widget.apply_radial_layout_state(state)
    QApplication.processEvents()
    restored = meta.data_widget.radial_plots[-1]
    assert restored.up_spin.value() == 45.0
    assert restored.cw_check.isChecked()
    assert restored.feature_combo.currentText() == heading


def test_a_stored_heading_with_a_keypoint_dim_is_offered(moll2025_gui):
    """The dataset's own ``angles`` carries keypoint/individual dims, like any
    other feature. Gating on the raw column count hid the compass from exactly
    the variables that have a direction."""
    _, meta = moll2025_gui
    if "angles" not in meta.app_state.data_loader.catalog.feature_choices():
        pytest.skip("this dataset has no stored heading")

    assert "Radial" in allowed_plot_types("feature", "angles", meta.app_state)
    meta._create_panel_for_source("feature", "angles", "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]

    assert rp._unit == "deg"
    # A new panel starts with its first multi-value dim on "All".
    assert "keypoint" in rp._dim_combos
    assert "keypoint" not in rp.selections()

    rp._dim_all_checks["keypoint"].setChecked(False)
    other = rp._dim_combos["keypoint"].itemText(1)
    rp._dim_combos["keypoint"].setCurrentText(other)
    QApplication.processEvents()
    assert rp.selections()["keypoint"] == other
    assert rp._unit == "deg"


def test_all_draws_one_arrow_per_value_with_a_legend(moll2025_gui):
    """ "All" on a dim frees it into one colour-coded arrow per value — how you
    see whether two keypoints (or two individuals) point the same way."""
    _, meta = moll2025_gui
    if "angles" not in meta.app_state.data_loader.catalog.feature_choices():
        pytest.skip("this dataset has no stored heading")
    meta._create_panel_for_source("feature", "angles", "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]

    n_values = rp._dim_combos["keypoint"].count()
    assert n_values >= 2
    assert rp._dim_all_checks["keypoint"].isChecked(), "a new panel shows every arrow"
    rp._dim_all_checks["keypoint"].setChecked(False)
    QApplication.processEvents()

    t = next(t for t, v in _sampled(rp) if v is not None)
    assert len(rp.current_values()) == 1, "a pinned dim must draw one arrow"
    assert not rp._legend.isVisible(), "one arrow needs no legend"

    rp._dim_all_checks["keypoint"].setChecked(True)
    QApplication.processEvents()
    assert "keypoint" not in rp.selections(), "a free dim must drop out of the selections"
    assert not rp._dim_combos["keypoint"].isEnabled()

    rp.set_time(t)
    drawn = rp.current_values()
    assert len(drawn) == n_values
    assert [label for label, _ in drawn] == [rp._dim_combos["keypoint"].itemText(i) for i in range(n_values)], (
        "legend labels must name the dim values"
    )
    assert rp._legend.isVisible()
    assert len(rp._legend.items) == n_values
    assert rp._unit == "deg", "freeing a dim must not lose the unit"


def test_a_free_dim_survives_a_data_refresh(moll2025_gui):
    """Trial changes re-enter refresh_features(); rebuilding from scratch would
    silently re-pin a dim the user had freed."""
    _, meta = moll2025_gui
    if "angles" not in meta.app_state.data_loader.catalog.feature_choices():
        pytest.skip("this dataset has no stored heading")
    meta._create_panel_for_source("feature", "angles", "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]
    rp._dim_all_checks["keypoint"].setChecked(True)

    meta.data_widget.refresh_radial_plots()
    QApplication.processEvents()

    assert rp._dim_all_checks["keypoint"].isChecked()
    assert "keypoint" not in rp.selections()


def test_only_one_dim_can_be_free(moll2025_gui):
    """Two free dims have no single column to draw — the same invariant the
    feature panels keep."""
    _, meta = moll2025_gui
    if "angles" not in meta.app_state.data_loader.catalog.feature_choices():
        pytest.skip("this dataset has no stored heading")
    meta._create_panel_for_source("feature", "angles", "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]
    if len(rp._dim_all_checks) < 2:
        pytest.skip("this heading has only one dim")

    first, second = list(rp._dim_all_checks)[:2]
    rp._dim_all_checks[first].setChecked(True)
    rp._dim_all_checks[second].setChecked(True)
    QApplication.processEvents()

    assert not rp._dim_all_checks[first].isChecked()
    assert rp._dim_combos[first].isEnabled()
    assert rp.selections().keys() == {first}


def test_a_free_dim_round_trips_through_the_saved_layout(moll2025_gui):
    _, meta = moll2025_gui
    if "angles" not in meta.app_state.data_loader.catalog.feature_choices():
        pytest.skip("this dataset has no stored heading")
    meta._create_panel_for_source("feature", "angles", "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]
    rp._dim_all_checks["keypoint"].setChecked(True)

    state = meta.data_widget.radial_layout_state()
    assert "keypoint" not in state[-1]["selections"]

    meta.data_widget.apply_radial_layout_state(state)
    QApplication.processEvents()
    restored = meta.data_widget.radial_plots[-1]
    assert restored._dim_all_checks["keypoint"].isChecked()
    drawn = [vals for _, vals in ((t, restored.current_values()) for t, _ in _sampled(restored)) if vals]
    assert drawn, "no heading anywhere in the window"
    assert max(len(vals) for vals in drawn) == restored._dim_combos["keypoint"].count()


def test_a_radial_panel_dies_with_its_derived_feature(moll2025_gui):
    _, meta = moll2025_gui
    heading = _heading(meta)
    meta._create_panel_for_source("feature", heading, "Radial")
    QApplication.processEvents()
    assert meta.data_widget.radial_plots

    meta.plot_container.console_panel._on_submitted(f"forget('{heading}')")
    QApplication.processEvents()

    assert not [rp for rp in meta.data_widget.radial_plots if rp.feature_combo.currentText() == heading]


def test_radial_values_survive_a_nan_window(moll2025_gui):
    """current_value() must report "unknown" rather than draw a NaN arrow."""
    _, meta = moll2025_gui
    heading = _heading(meta)
    meta._create_panel_for_source("feature", heading, "Radial")
    QApplication.processEvents()
    rp = meta.data_widget.radial_plots[-1]

    rp._ensure_data = lambda: True  # freeze the cache we are about to fake
    rp._time = np.array([0.0, 1.0])
    rp._values = np.array([np.nan, np.nan])
    rp.set_time(0.5)
    assert rp.current_value() is None
    assert not rp._arrow_items
