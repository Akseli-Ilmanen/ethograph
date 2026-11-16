"""A heading arrow in the space plot points where the angle says, in the data's x-y plane."""

import numpy as np

from ethograph.gui.plots_space import heading_arrow


def test_arrow_points_counter_clockwise_from_x():
    segments = heading_arrow(np.array([1.0, 2.0, 3.0]), 90.0, 2.0)
    origin, tip = segments[0], segments[1]
    np.testing.assert_allclose(tip - origin, [0.0, 2.0, 0.0], atol=1e-6)
    # Barbs start at the tip and fall back behind it.
    assert (segments[3][1] < tip[1]) and (segments[5][1] < tip[1])
