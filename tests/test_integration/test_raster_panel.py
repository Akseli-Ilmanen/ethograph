"""The spike raster as a panel of its own, drawing what the cluster table's filters let through."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
nap = pytest.importorskip("pynapple")

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtGui import QPainter, QPixmap  # noqa: E402
from qtpy.QtWidgets import QStyle, QStyleOptionViewItem  # noqa: E402

from ethograph.gui.widgets_ephys import _COLOR_ROLE  # noqa: E402

GOOD_A, MUA, GOOD_B = 7, 42, 105
RED = (228, 26, 28)


@pytest.fixture
def units_widget(gui, monkeypatch):
    """The ephys widget holding three pynapple units, two of them ``good``."""
    _, meta = gui
    ew = meta.ephys_widget
    group = nap.TsGroup(
        {
            GOOD_A: nap.Ts(t=np.array([1.0, 2.0, 30.0])),
            MUA: nap.Ts(t=np.array([3.0])),
            GOOD_B: nap.Ts(t=np.array([4.0])),
        }
    )
    group.set_info(group=np.array(["good", "mua", "good"]))
    ew._tsgroup = group
    ew._neurons_source = "pynapple"
    ew._cluster_df = ew._build_cluster_df_from_tsgroup(group)
    ew._populate_cluster_table(ew._cluster_df)
    monkeypatch.setattr(ew, "_trial_ep", lambda: nap.IntervalSet(0.0, 10.0))
    monkeypatch.setattr(ew, "_ephys_offset", lambda: 0.0)
    return ew, meta.plot_container.raster_plot


def _times_by_color(raster) -> dict[tuple, list[float]]:
    return {tuple(color): sorted(times.tolist()) for times, _, color in raster._multi_entries}


def test_a_saved_layout_restores_raster_and_trace_independently(gui):
    _, meta = gui
    pc = meta.plot_container

    pc.apply_layout_state({"panels": [{"type": "raster"}, {"type": "ephys"}]})
    assert pc._panel_visible == {"ephys": True, "raster": True}

    pc.apply_layout_state({"panels": [{"type": "raster"}]})
    assert pc._panel_visible == {"ephys": False, "raster": True}


def test_the_raster_draws_the_units_the_cluster_table_lets_through(units_widget, qtbot):
    ew, raster = units_widget
    assert ew.filtered_unit_ids() == [GOOD_A, GOOD_B]

    ew.refresh_raster()
    (drawn,) = _times_by_color(raster).values()
    assert drawn == [1.0, 2.0, 4.0]

    human_col = ew._find_col_by_header("", exact="Human")
    with qtbot.waitSignal(ew.unit_filter_changed):
        ew._cluster_proxy.set_cat_filter(human_col, set())
        ew._on_unit_filter_changed()
    (drawn,) = _times_by_color(raster).values()
    assert drawn == [1.0, 2.0, 3.0, 4.0]


def test_ctrl_click_on_a_raster_row_toggles_its_unit_in_the_table(units_widget):
    ew, raster = units_widget
    ew.refresh_raster()

    def click(unit: int, modifiers=Qt.ControlModifier) -> None:
        (row,) = (key for key, units in ew._raster_row_units.items() if units == [unit])
        y = raster._hw_to_global_y[row]
        raster.plot_clicked.emit({"x": 1.0, "y": y, "button": Qt.LeftButton, "modifiers": modifiers, "plot": raster})

    click(GOOD_B)
    click(GOOD_A)
    assert ew.selected_unit_ids() == [GOOD_A, GOOD_B]

    click(GOOD_B)
    assert ew.selected_unit_ids() == [GOOD_A]

    # A plain click is the labels' and the playhead's, never a selection.
    click(GOOD_B, modifiers=Qt.NoModifier)
    assert ew.selected_unit_ids() == [GOOD_A]


def test_a_selected_unit_is_highlighted_among_the_filtered_ones(units_widget):
    ew, raster = units_widget
    # A highlight on a unit the filter hides draws nothing.
    ew._multi_cluster_colors = {GOOD_A: RED, MUA: RED}

    ew._draw_raster()

    by_color = _times_by_color(raster)
    assert by_color.pop(RED) == [1.0, 2.0]
    (unselected,) = by_color.values()
    assert unselected == [4.0]


def test_table_order_gives_each_unit_a_row_in_the_tables_order(units_widget):
    ew, raster = units_widget
    id_col = ew._find_col_by_header("", exact="id")

    ew.cluster_table.sortByColumn(id_col, Qt.DescendingOrder)

    assert ew.ordered_unit_ids() == [GOOD_B, GOOD_A]
    assert not raster.follows_trace_y, "unit rows are not the trace's channels"
    rows_of_spikes = {float(t): int(r) for times, rows, _ in raster._multi_entries for t, r in zip(times, rows)}
    assert rows_of_spikes == {4.0: 0, 1.0: 1, 2.0: 1}


def _units_in(meta, n_units: int = 12, t_end: float = 400.0) -> None:
    """Load ``n_units`` pynapple units spiking all through the session into the ephys widget."""
    ew = meta.ephys_widget
    rng = np.random.default_rng(0)
    group = nap.TsGroup({uid: nap.Ts(t=np.sort(rng.uniform(0.0, t_end, 2000))) for uid in range(n_units)})
    ew._tsgroup = group
    ew._neurons_source = "pynapple"
    ew._cluster_df = ew._build_cluster_df_from_tsgroup(group)
    ew._populate_cluster_table(ew._cluster_df)
    ew._sync_row_order_combo()


def _switch_to_other_trial(state) -> None:
    state.trials_sel = next(t for t in state.trials if t != state.trials_sel)
    state.trial_changed.emit()


def test_firing_rate_opens_as_a_heatmap_with_a_row_per_filtered_unit(moll2025_gui):
    _, meta = moll2025_gui
    _units_in(meta)
    pc = meta.plot_container
    n_before = len(pc.heatmap_plots)

    meta._create_panel_for_source("firing_rate", "firing_rate", "Heatmap")

    assert len(pc.heatmap_plots) == n_before + 1
    heatmap = pc.heatmap_plots[-1]
    assert heatmap._effective_feature() == "firing_rate"
    assert heatmap._buffered_data.shape[1] == 12
    assert "firing_rate" in pc._available_features()


def test_a_restored_firing_rate_heatmap_gets_its_rates_binned(moll2025_gui):
    """A saved layout names the firing rates before anything has computed them."""
    _, meta = moll2025_gui
    _units_in(meta)
    pc = meta.plot_container
    assert "firing_rate" not in pc._available_features()

    pc.apply_layout_state({"panels": [{"type": "heatmap", "feature": "firing_rate", "selections": {}}]})
    heatmap = pc.heatmap_plots[-1]
    # Not the feature the sidebar happens to have selected, drawn under the firing rates' name.
    assert heatmap.image_item.image is None

    _switch_to_other_trial(meta.app_state)

    assert heatmap._buffered_data.shape[1] == 12


def test_the_raster_and_the_firing_rates_share_one_row_order(moll2025_gui):
    _, meta = moll2025_gui
    _units_in(meta)
    ew = meta.ephys_widget
    # The heatmaps' own sort is on: it must not reorder a heatmap of units.
    meta.app_state.heatmap_sort_mode = "trial"
    meta._create_panel_for_source("firing_rate", "firing_rate", "Heatmap")
    heatmap = meta.plot_container.heatmap_plots[-1]

    ew.raster_row_order_combo.setCurrentIndex(ew.raster_row_order_combo.findData("peak_trial"))

    order = ew.ordered_unit_ids()
    assert order == ew._peak_units
    assert sorted(order) == list(range(12)) and order != list(range(12))
    assert [int(label) for label in heatmap._last_visible_labels] == order
    assert [units for _, units in sorted(ew._raster_row_units.items())] == [[unit] for unit in order]


def test_a_firing_rate_heatmap_shows_the_neuron_table_not_the_coords(moll2025_gui):
    _, meta = moll2025_gui
    _units_in(meta)
    meta._create_panel_for_source("firing_rate", "firing_rate", "Heatmap")
    meta.plot_container.active_feature_plot = meta.plot_container.heatmap_plots[-1]

    meta._on_plot_focus("heatmap")

    assert meta.context_panel.current_context() == "firing_rate"


def test_the_firing_rate_panel_follows_the_trial_with_a_console_open(moll2025_gui):
    """The rates are re-binned per trial; the console's per-trial reset must not take them along."""
    _, meta = moll2025_gui
    _units_in(meta)
    meta._add_console_panel()
    meta._create_panel_for_source("firing_rate", "firing_rate", "Heatmap")
    heatmap = meta.plot_container.heatmap_plots[-1]
    loader = meta.app_state.data_loader
    before = loader.derived["firing_rate"].values.copy()

    _switch_to_other_trial(meta.app_state)

    assert heatmap in meta.plot_container.heatmap_plots
    after = loader.derived["firing_rate"].values
    assert after.shape != before.shape or not np.array_equal(after, before)


def test_the_firing_rate_rows_follow_the_cluster_tables_filter(moll2025_gui):
    _, meta = moll2025_gui
    _units_in(meta)
    ew = meta.ephys_widget
    meta._create_panel_for_source("firing_rate", "firing_rate", "Heatmap")
    heatmap = meta.plot_container.heatmap_plots[-1]

    id_col = ew._find_col_by_header("", exact="id")
    ew._cluster_proxy.set_numeric_filter(id_col, "<=", 4.0)
    ew._on_unit_filter_changed()

    assert ew.filtered_unit_ids() == [0, 1, 2, 3, 4]
    assert heatmap._buffered_data.shape[1] == 5


def test_a_selected_units_coloured_id_cell_paints(units_widget):
    """The delegate reads the selected state off Qt's flag enum, which no longer mixes with a bare int."""
    ew, _raster = units_widget
    ew.cluster_table.selectRow(0)
    index = ew._cluster_proxy.index(0, ew._find_col_by_header("", exact="id"))
    assert index.data(_COLOR_ROLE) is not None, "selecting a unit colours its id cell"

    pixmap = QPixmap(60, 20)
    option = QStyleOptionViewItem()
    option.rect = pixmap.rect()
    option.state |= QStyle.StateFlag.State_Selected
    painter = QPainter(pixmap)
    try:
        ew._cluster_id_delegate.paint(painter, option, index)
    finally:
        painter.end()
