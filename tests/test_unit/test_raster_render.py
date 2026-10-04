"""What the spike raster draws: ticks or a density image, and which spike goes where."""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui.raster_render import (
    DENSITY_OFF,
    DENSITY_ON,
    MAX_TICKS,
    auto_tick_width,
    choose_render,
    density_image,
    group_by_color,
    order_units,
    tick_segments,
)

GREY = (180, 180, 180, 200)
RED = (228, 26, 28)
BLUE = (55, 126, 184)


def test_auto_switches_on_crowding_with_a_band_that_holds_the_current_rendering():
    n_cells = 1000
    between = int((DENSITY_ON + DENSITY_OFF) / 2 * n_cells)

    assert choose_render("auto", "ticks", int(DENSITY_ON * n_cells) + 1, n_cells) == "density"
    assert choose_render("auto", "density", int(DENSITY_OFF * n_cells) - 1, n_cells) == "ticks"
    # Inside the band, whatever is on screen stays.
    assert choose_render("auto", "ticks", between, n_cells) == "ticks"
    assert choose_render("auto", "density", between, n_cells) == "density"


def test_a_forced_mode_wins_except_ticks_past_the_paint_limit():
    assert choose_render("density", "ticks", 1, 10**6) == "density"
    assert choose_render("ticks", "density", 10**5, 10) == "ticks"
    assert choose_render("ticks", "ticks", MAX_TICKS + 1, 10**9) == "density"


def test_spikes_are_grouped_by_colour_with_unfiltered_units_dropped():
    times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    units = np.array([7, 42, 105, 7, 9])
    rows = {7: 0, 105: 1, 9: 2}  # 42 does not pass the filter

    entries = group_by_color(times, units, rows, {7: RED, 9: RED, 42: BLUE}, GREY)

    by_color = {color: (t.tolist(), r.tolist()) for t, r, color in entries}
    assert by_color == {GREY: ([0.3], [1]), RED: ([0.1, 0.4, 0.5], [0, 0, 2])}
    assert entries[0][2] == GREY, "the unselected group is drawn first, underneath"


def test_ticks_are_endpoint_pairs_centred_on_the_row():
    x, y = tick_segments(np.array([1.0, 2.0]), np.array([10.0, 20.0]), 0.5)
    assert x.tolist() == [1.0, 1.0, 2.0, 2.0]
    assert y.tolist() == [9.5, 10.5, 19.5, 20.5]


def test_auto_tick_width_is_bold_when_sparse_thin_when_crowded_and_never_a_blob():
    tall = 30.0
    assert auto_tick_width(n_spikes=100, n_cells=10_000, tick_height_px=tall) == 3
    assert auto_tick_width(n_spikes=1_200, n_cells=10_000, tick_height_px=tall) == 2
    assert auto_tick_width(n_spikes=5_000, n_cells=10_000, tick_height_px=tall) == 1
    # A 3 px tall tick in a thin row stays 1 px wide however sparse the view is.
    assert auto_tick_width(n_spikes=1, n_cells=10_000, tick_height_px=3.0) == 1


def test_density_counts_every_spike_once_and_mixes_colours_by_count():
    red = (np.array([0.5, 0.5, 0.5, 2.5]), np.array([0.0, 0.0, 0.0, 1.0]), RED)
    blue = (np.array([0.5, 9.0]), np.array([0.0, 0.0]), BLUE)  # 9.0 lies outside the grid

    image = density_image([red, blue], t0=0.0, t1=4.0, n_x=4, y0=-0.5, cell_height=1.0, n_y=2)

    assert image.shape == (4, 2, 4)
    filled = image[..., 3] > 0
    assert filled.tolist() == [[True, False], [False, False], [False, True], [False, False]]
    # Three red spikes and one blue share a cell: its colour is their weighted mean.
    expected = np.round((3 * np.array(RED) + np.array(BLUE)) / 4)
    assert image[0, 0, :3].tolist() == expected.tolist()
    assert image[2, 1, :3].tolist() == list(RED)
    assert image[0, 0, 3] > image[2, 1, 3], "more spikes, brighter"


def test_density_refuses_a_grid_without_extent():
    with pytest.raises(ValueError, match="positive size"):
        density_image([], t0=1.0, t1=1.0, n_x=4, y0=0.0, cell_height=1.0, n_y=2)


def test_a_fitted_order_applies_to_the_units_shown_and_appends_the_ones_it_never_saw():
    assert order_units([1, 2, 3, 4], [3, 9, 1]) == [3, 1, 2, 4]
    assert order_units([1, 2], None) == [1, 2]
