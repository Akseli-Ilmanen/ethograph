"""The pose overlay survives a proxy swap and lands on the proxy's pixels.

Proxy playback rebuilds the view's plot at a lower resolution. The overlay
belongs to the old scene, so it must go with it and be drawn again; and a
pose file speaks source pixels, so it is scaled to the texture it is drawn
on. Regression: boxes vanished when the proxy was toggled, and would have
sat at full-resolution coordinates on a 480 px proxy had they come back.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygfx")

import pygfx as gfx  # noqa: E402

from ethograph.gui.pose_overlay import OverlayStyle, PoseOverlay  # noqa: E402


def test_pose_pixels_are_scaled_to_the_texture_then_flipped():
    overlay = PoseOverlay(gfx.Scene())
    overlay.set_data(None, OverlayStyle(), img_height=480.0, scale=0.25)

    world = overlay._to_world(np.array([[400.0, 800.0]], dtype=np.float32))
    assert np.allclose(world, [[100.0, 480.0 - 200.0]])

    overlay.set_data(None, OverlayStyle(), img_height=1920.0)
    assert np.allclose(overlay._to_world(np.array([[400.0, 800.0]])), [[400.0, 1120.0]]), "source: scale 1"


def test_clearing_the_view_drops_the_overlay_with_its_scene(qapp):
    from ethograph.gui.pygfx_video import CameraView

    view = CameraView()
    assert view.overlay_scale() == 1.0, "no texture, nothing to scale to"
    view._overlay = PoseOverlay(gfx.Scene())
    view.clear()
    assert view.overlay is None
