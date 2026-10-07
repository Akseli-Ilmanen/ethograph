"""A NaN sample on the space plot is a gap: no line through it, no marker on it."""

import numpy as np
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.plots_lineplot import MultiColoredLineItem  # noqa: E402
from ethograph.gui.plots_space import SpacePlot, gl_line_segments  # noqa: E402


def test_gl_segments_skip_every_pair_touching_a_nan():
    xyz = np.array([[0, 0, 0], [1, 1, 1], [np.nan, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=float)
    colors = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 4))
    pos, color = gl_line_segments(xyz, colors)
    assert pos.tolist() == [[0, 0, 0], [1, 1, 1], [3, 3, 3], [4, 4, 4]]
    assert color[:, 0].tolist() == [0, 0, 3, 3], "a segment carries its first sample's colour"
    assert gl_line_segments(xyz[:1])[0].shape == (0, 3)


def test_coloured_2d_line_leaves_a_gap_at_a_nan(qapp):
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    colors = np.ones((4, 3))
    item = MultiColoredLineItem(x=x, y=x.copy(), colors=colors)
    assert sum(path.elementCount() for _pen, path in item._paths) == 4, "two drawable segments, two points each"


def test_marker_disappears_on_a_nan_sample(qapp, app_state):
    plot = SpacePlot.__new__(SpacePlot)
    plot.app_state = app_state
    plot._trajectory_times = np.array([0.0, 1.0, 2.0])
    plot._trajectory_pos = (np.array([0.0, np.nan, 2.0]), np.array([0.0, 1.0, 2.0]), None)
    plot._time_marker_item = object()
    plot.space_widget = object()
    removed = []
    plot._remove_time_marker = lambda: removed.append(True)
    plot._maybe_slide_window = lambda _t: None
    plot.update_time_marker(1.5)
    assert removed, "the marker stayed at a position the data does not have"
