"""A state label being drawn shows where it started.

The first of the two clicks used to change nothing on screen, so the end time
was picked blind. ``show_pending_label`` puts a dashed anchor plus a faint
cursor-tracking region on every panel, and ``clear_pending_label`` must take
both off again on commit, cancel and trial change — a stale anchor would claim
an interval is still being drawn when none is.
"""

import pyqtgraph as pg
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.label_drawing_mixin import LabelDrawingMixin  # noqa: E402


class _Container(LabelDrawingMixin):
    """Minimal host: the mixin only needs the plot list and its own state."""

    def __init__(self, plots):
        self._plots = plots
        self.label_mappings = {}
        self._pending_label_items: list = []
        self._pending_label_regions: list = []
        self._pending_hover_conns: list = []
        self._pending_label_anchor = None

    def _get_all_plots(self) -> list:
        return list(self._plots)


@pytest.fixture
def container(qtbot):
    plots = []
    for _ in range(2):
        widget = pg.PlotWidget()
        qtbot.addWidget(widget)
        widget.plot_item = widget.getPlotItem()
        plots.append(widget)
    return _Container(plots)


def _items(plot):
    return set(plot.plot_item.items)


def test_anchor_and_preview_land_on_every_panel(container):
    before = [_items(p) for p in container._plots]

    container.show_pending_label(3.5, (200, 100, 50))

    for plot, prior in zip(container._plots, before):
        added = _items(plot) - prior
        assert len(added) == 2, "each panel gets an anchor line and a preview region"
        assert any(isinstance(i, pg.InfiniteLine) for i in added)
        assert any(isinstance(i, pg.LinearRegionItem) for i in added)
    assert container._pending_label_anchor == 3.5


def test_clear_removes_everything_it_added(container):
    before = [_items(p) for p in container._plots]

    container.show_pending_label(3.5, (200, 100, 50))
    container.clear_pending_label()

    for plot, prior in zip(container._plots, before):
        assert _items(plot) == prior
    assert container._pending_label_anchor is None
    assert not container._pending_hover_conns


def test_showing_twice_does_not_stack_anchors(container):
    """Re-arming mid-draw must replace the anchor, not leave the old one behind."""
    before = [_items(p) for p in container._plots]

    container.show_pending_label(3.5, (200, 100, 50))
    container.show_pending_label(9.0, (10, 20, 30))

    for plot, prior in zip(container._plots, before):
        assert len(_items(plot) - prior) == 2
    assert container._pending_label_anchor == 9.0


def test_hover_stretches_the_preview_on_all_panels(container):
    container.show_pending_label(2.0, (200, 100, 50))
    hovered = container._plots[0]

    scene_pos = hovered.plot_item.vb.mapViewToScene(pg.Point(7.0, 0.0))
    container._on_pending_hover(hovered, scene_pos)

    for region in container._pending_label_regions:
        lo, hi = sorted(region.getRegion())
        assert lo == pytest.approx(2.0, abs=1e-6)
        assert hi == pytest.approx(7.0, abs=1e-6)


def test_hover_after_clear_is_inert(container):
    """A queued mouse-move must not resurrect a preview that was just cleared."""
    container.show_pending_label(2.0, (200, 100, 50))
    hovered = container._plots[0]
    scene_pos = hovered.plot_item.vb.mapViewToScene(pg.Point(7.0, 0.0))
    container.clear_pending_label()

    container._on_pending_hover(hovered, scene_pos)  # must not raise

    assert container._pending_label_anchor is None


def test_hover_reports_the_cursor_time(container):
    """The host's hook sees the hovered time so the video can follow it."""
    seen: list[float] = []
    container._pending_hover_moved = seen.append
    container.show_pending_label(2.0, (200, 100, 50))
    hovered = container._plots[0]

    scene_pos = hovered.plot_item.vb.mapViewToScene(pg.Point(7.0, 0.0))
    container._on_pending_hover(hovered, scene_pos)

    assert seen == [pytest.approx(7.0, abs=1e-6)]
