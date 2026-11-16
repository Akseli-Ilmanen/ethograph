"""Angular detection for the radial (compass) plot — the gate that decides
which variables the add-panel popup offers "Radial" for."""

import numpy as np
import pyqtgraph as pg
import pytest

from ethograph.gui.plots_radial import FULL_TURN_TOLERANCE, angular_unit


def test_degrees_are_recognised():
    assert angular_unit(np.linspace(0.0, 360.0, 500)) == "deg"
    assert angular_unit(np.linspace(-180.0, 180.0, 500)) == "deg"


def test_radians_are_recognised():
    assert angular_unit(np.linspace(0.0, 2 * np.pi, 500)) == "rad"
    assert angular_unit(np.linspace(-np.pi, np.pi, 500)) == "rad"


def test_a_partial_turn_within_tolerance_still_counts():
    """Real headings rarely sweep exactly a full turn."""
    span = 360.0 * (1 - FULL_TURN_TOLERANCE * 0.5)
    assert angular_unit(np.linspace(0.0, span, 500)) == "deg"


def test_non_angular_signals_are_rejected():
    assert angular_unit(np.linspace(0.0, 1.0, 500)) is None  # normalised
    assert angular_unit(np.linspace(0.0, 100.0, 500)) is None  # speed
    assert angular_unit(np.linspace(0.0, 180.0, 500)) is None  # half turn only
    assert angular_unit(np.array([1.0])) is None  # too few samples
    assert angular_unit(np.full(100, np.nan)) is None


def test_nans_do_not_break_detection():
    values = np.linspace(0.0, 360.0, 500)
    values[::7] = np.nan
    assert angular_unit(values) == "deg"


@pytest.fixture
def state():
    """A minimal app_state over a real loader, so the gate is exercised end to
    end (feature_ncols + select + span), not just the span maths."""
    import xarray as xr

    import ethograph as eto
    from ethograph.io.catalog import XarrayLoader, catalog_from_xarray
    from ethograph.io.derived import DerivedLoader
    from ethograph.io.time_model import TimeRange

    time = np.linspace(0.0, 10.0, 1001)
    heading = np.linspace(0.0, 359.0, 1001)
    scattered = heading.copy()
    scattered[::5] = np.nan
    partial = heading.copy()
    partial[:300] = np.nan  # NaNs that swallow a third of the sweep

    ds = xr.Dataset(
        {
            "hd": (("time", "individuals"), heading[:, None]),
            "hd_nan": (("time", "individuals"), scattered[:, None]),
            "hd_partial": (("time", "individuals"), partial[:, None]),
            "hd_two": (("time", "pair"), np.column_stack([heading, heading])),
            "speed": (("time", "individuals"), np.linspace(0.0, 50.0, 1001)[:, None]),
        },
        coords={"time": time, "individuals": ["a"], "pair": ["a", "b"]},
        attrs={"trial": 1},
    )
    catalog = catalog_from_xarray(ds, eto.from_datasets([ds]))

    class _State:
        data_loader = DerivedLoader(XarrayLoader(ds, catalog))
        window_bounds = TimeRange(0.0, 10.0)
        source_collection = None

    state = _State()
    state.ds = ds
    return state


def test_the_gate_does_not_depend_on_app_state_ds(state):
    """Pynapple sessions have no ``app_state.ds``; the dims come from the
    loader, so the answer must be the same with or without one."""
    from ethograph.gui.plots_radial import feature_angular_unit

    state.ds = None
    assert feature_angular_unit(state, "hd") == "deg"
    assert feature_angular_unit(state, "speed") is None


def test_scattered_nans_do_not_hide_the_radial_option(state):
    from ethograph.gui.plots_radial import feature_angular_unit
    from ethograph.gui.source_popup import allowed_plot_types

    assert feature_angular_unit(state, "hd") == "deg"
    assert feature_angular_unit(state, "hd_nan") == "deg"
    assert "Radial" in allowed_plot_types("feature", "hd_nan", state)


def test_a_heading_carrying_a_dim_is_still_offered(state):
    """A compass shows one heading — one *column*, not one variable. Headings
    normally come with a keypoint/individual dim, so the gate pins the dims and
    judges what comes out; gating on the raw column count hid the option from
    exactly the datasets that have headings."""
    from ethograph.gui.plots_radial import default_selections, feature_angular_unit
    from ethograph.gui.source_popup import allowed_plot_types

    assert default_selections(state, "hd_two") == {"pair": "a"}
    assert feature_angular_unit(state, "hd_two") == "deg"
    assert feature_angular_unit(state, "hd_two", {"pair": "b"}) == "deg"
    assert "Radial" in allowed_plot_types("feature", "hd_two", state)


def test_a_non_angular_signal_is_never_offered(state):
    from ethograph.gui.plots_radial import feature_angular_unit
    from ethograph.gui.source_popup import allowed_plot_types

    assert feature_angular_unit(state, "speed") is None
    assert "Radial" not in allowed_plot_types("feature", "speed", state)


def test_nans_that_swallow_part_of_the_sweep_are_reported_honestly(state):
    """The finite data really does span only ~250°, so it is not a full turn."""
    from ethograph.gui.plots_radial import feature_angular_unit

    assert feature_angular_unit(state, "hd_partial") is None


def test_session_bounds_in_another_time_frame_do_not_hide_the_option(state):
    """Session ranges are session-absolute while xarray slices trial-relative
    time — on a trial carrying an offset they select nothing, and used alone
    they wrongly answered "not a heading"."""
    from ethograph.gui.plots_radial import feature_angular_unit
    from ethograph.io.time_model import TimeRange

    class _Collection:
        session_range = TimeRange(5000.0, 5010.0)  # selects nothing from this trial
        union_range = TimeRange(5000.0, 5010.0)

    state.source_collection = _Collection()
    assert feature_angular_unit(state, "hd") == "deg"


def _pen_color(item) -> str:
    return item.opts["pen"].color().name().lower()


def _compass_with_arrows(qtbot, n_columns: int):
    """A radial plot holding *n_columns* finite headings at t = 0."""
    from ethograph.gui.plots_radial import RadialPlot

    plot = RadialPlot(shell=None, app_state=None)
    qtbot.addWidget(plot)
    plot._ensure_data = lambda: True
    plot._unit = "deg"
    plot._time = np.array([0.0, 1.0])
    plot._values = np.tile(np.linspace(0.0, 300.0, n_columns), (2, 1))
    plot._labels = [f"kp{i}" for i in range(n_columns)]
    plot.set_time(0.0)
    return plot


def test_each_arrow_gets_its_own_colour_and_legend_entry(qtbot):
    from ethograph.gui.app_constants import MULTIDIM_COLORS
    from ethograph.gui.plots_radial import RadialPlot

    plot = _compass_with_arrows(qtbot, 3)

    assert len(plot.current_values()) == 3
    assert len(plot._arrow_items) == 6  # shaft + head per arrow
    assert len(plot._legend.items) == 3
    names = [_pen_color(sample.item) for sample, _ in plot._legend.items]
    assert len(set(names)) == 3
    assert names[0] == MULTIDIM_COLORS[0].lower()
    assert isinstance(plot, RadialPlot)


def test_more_arrows_than_colours_share_one_colour(qtbot):
    """Recycling hues would claim the 1st and 11th individual are the same."""
    from ethograph.gui.app_constants import MULTIDIM_COLORS

    n = len(MULTIDIM_COLORS) + 3
    plot = _compass_with_arrows(qtbot, n)

    assert len(plot.current_values()) == n
    assert {_pen_color(item) for item in plot._arrow_items if isinstance(item, pg.PlotCurveItem)} == {
        MULTIDIM_COLORS[0].lower()
    }
    # One entry saying how many, rather than a colour key that means nothing.
    assert len(plot._legend.items) == 1
    assert str(n) in plot._legend.items[0][1].text


@pytest.mark.parametrize(
    ("up", "clockwise", "value", "expected"),
    [
        # The value set as "up" always points up (screen 90°), whatever it is.
        (0.0, False, 0.0, 90.0),
        (90.0, False, 90.0, 90.0),
        (-45.0, True, -45.0, 90.0),
        # Handedness: +90 from up is left (CCW) or right (CW).
        (0.0, False, 90.0, 180.0),
        (0.0, True, 90.0, 0.0),
    ],
)
def test_screen_angle_places_up_where_the_user_asked(qtbot, up, clockwise, value, expected):
    from ethograph.gui.plots_radial import RadialPlot

    plot = RadialPlot(shell=None, app_state=None)
    qtbot.addWidget(plot)
    plot.up_spin.setValue(up)
    plot.cw_check.setChecked(clockwise)
    assert plot._screen_angle(value) == pytest.approx(expected)


def test_a_heading_only_in_a_later_dim_value_is_still_offered():
    """Every value of a dim is judged, not just the first."""
    import xarray as xr

    import ethograph as eto
    from ethograph.gui.plots_radial import feature_angular_unit
    from ethograph.io.catalog import XarrayLoader, catalog_from_xarray
    from ethograph.io.derived import DerivedLoader
    from ethograph.io.time_model import TimeRange

    time = np.linspace(0.0, 10.0, 1001)
    still = np.full(1001, 40.0)
    turning = np.linspace(0.0, 359.0, 1001)
    ds = xr.Dataset(
        {"hd": (("time", "keypoints", "individuals"), np.column_stack([still, turning])[:, :, None])},
        coords={"time": time, "keypoints": ["a", "b"], "individuals": ["x"]},
        attrs={"trial": 1},
    )
    catalog = catalog_from_xarray(ds, eto.from_datasets([ds]))

    class _State:
        data_loader = DerivedLoader(XarrayLoader(ds, catalog))
        window_bounds = TimeRange(0.0, 10.0)
        source_collection = None

    assert feature_angular_unit(_State(), "hd") == "deg"
    assert feature_angular_unit(_State(), "hd", {"keypoints": "a", "individuals": "x"}) is None
