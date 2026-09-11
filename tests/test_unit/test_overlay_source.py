"""The video overlays any feature in the camera's pixels; `position` is the default."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ethograph.gui.pose_convert import poses_ds_to_points
from ethograph.io.derived import DerivedFeature
from ethograph.io.overlay_source import (
    companions,
    fits_frame,
    frames_for_times,
    overlay_candidates,
    overlay_dataset,
)

T = 5
TIME = np.arange(T) / 10.0
W, H = 640, 480


def _pos(x: float, y: float, dims=("time", "space", "individuals")) -> xr.DataArray:
    data = np.full((T, 2, 2), np.nan)
    data[:, 0, :] = x
    data[:, 1, :] = y
    return xr.DataArray(data, dims=dims, coords={"time": TIME, "space": ["x", "y"], dims[-1]: ["m", "f"]})


def _session(**extra: xr.DataArray) -> xr.Dataset:
    ds = xr.Dataset({"position": _pos(100.0, 200.0)})
    for k, v in extra.items():
        ds[k] = v
    return ds


class TestCandidates:
    def test_position_first_then_anything_else_in_pixels(self):
        ds = _session(position_smooth=_pos(101.0, 199.0), speed=xr.DataArray(np.ones(T), dims=["time"]))
        names = [c.name for c in overlay_candidates(ds, None, None, (W, H))]
        assert names == ["position", "position_smooth"]

    def test_a_metric_variable_is_not_drawn(self):
        ds = _session(position_mm=_pos(5000.0, 5.0))
        assert [c.name for c in overlay_candidates(ds, None, None, (W, H))] == ["position"]

    def test_declared_pixels_are_trusted_and_millimetres_refused(self):
        far = _pos(5000.0, 5.0).assign_attrs(space_unit="px")
        ds = _session(far=far, mm=_pos(1.0, 1.0).assign_attrs(space_unit="mm"))
        assert [c.name for c in overlay_candidates(ds, None, None, (W, H))] == ["position", "far"]

    def test_an_unknown_extra_dim_disqualifies(self):
        odd = xr.DataArray(
            np.full((T, 2, 3), 10.0), dims=["time", "space", "sensor"], coords={"time": TIME, "space": ["x", "y"]}
        )
        assert [c.name for c in overlay_candidates(_session(odd=odd), None, None, (W, H))] == ["position"]

    def test_companions_are_found_by_name(self):
        shape = _pos(20.0, 20.0)
        ds = _session(shape=shape, confidence=xr.DataArray(np.ones((T, 2)), dims=["time", "individuals"]))
        (c,) = overlay_candidates(ds, None, None, (W, H))
        assert c.name == "position" and c.has_shape and c.has_confidence, "shape is a companion, not a feature"
        assert companions("position_smooth") == ("position_smooth_confidence", "position_smooth_shape")

    def test_camera_dim_is_selected_away_per_camera(self):
        cams = xr.DataArray(["cam-1", "cam-2"], dims="camera")
        stacked = xr.concat([_pos(100.0, 200.0), _pos(9000.0, 200.0)], dim=cams)
        ds = xr.Dataset({"position": stacked})
        assert [c.name for c in overlay_candidates(ds, None, "cam-1", (W, H))] == ["position"]
        assert overlay_candidates(ds, None, "cam-2", (W, H)) == []
        assert overlay_candidates(ds, None, None, (W, H)) == [], "a stacked dataset needs a camera"

    def test_a_console_snapshot_with_x_and_y_qualifies(self):
        derived = {"track": DerivedFeature("track", time=TIME, values=np.full((T, 2), 50.0), dim_labels=["x", "y"])}
        assert [c.name for c in overlay_candidates(None, derived, None, (W, H))] == ["track"]
        one_col = {"speed": DerivedFeature("speed", time=TIME, values=np.ones(T))}
        assert overlay_candidates(None, one_col, None, (W, H)) == []

    def test_unknown_frame_size_skips_the_pixel_test(self):
        ds = _session(position_mm=_pos(5000.0, 5.0))
        assert [c.name for c in overlay_candidates(ds, None, None, None)] == ["position", "position_mm"]


class TestOverlayDataset:
    def test_a_console_result_becomes_a_movement_dataset_the_converter_accepts(self):
        derived = {"track": DerivedFeature("track", time=TIME, values=np.full((T, 2), 50.0), dim_labels=["x", "y"])}
        ds = overlay_dataset(None, derived, "track", None)
        assert ds["position"].dims == ("time", "space", "individuals")
        points, boxes, props = poses_ds_to_points(ds)
        assert boxes is None and len(points) == T and (props["confidence"] == 1).all()

    def test_shape_companion_yields_boxes_and_axes_are_ordered_for_the_converter(self):
        ds = _session(shape=_pos(20.0, 20.0))
        ds = ds.transpose("individuals", "space", "time")  # a hostile order
        out = overlay_dataset(ds, None, "position", None)
        assert out["position"].dims == ("time", "space", "individuals")
        points, boxes, _ = poses_ds_to_points(out)
        assert boxes is not None and boxes.shape == (2 * T, 4, 4)

    def test_camera_is_required_when_the_dim_exists(self):
        stacked = xr.concat([_pos(1.0, 1.0), _pos(2.0, 2.0)], dim=xr.DataArray(["a", "b"], dims="camera"))
        with pytest.raises(KeyError):
            overlay_dataset(xr.Dataset({"position": stacked}), None, "position", None)


class TestFrames:
    def test_frames_follow_the_video_clock(self):
        # Trial starts 10 s into a session-long 30 fps video: offset -10.
        assert list(frames_for_times(np.array([0.0, 0.5]), 30.0, -10.0)) == [300, 315]
        assert list(frames_for_times(np.array([0.0, 0.5]), 30.0, 0.0)) == [0, 15]

    def test_fits_frame_includes_box_edges(self):
        pos = _pos(630.0, 100.0)
        assert fits_frame(pos, None, W, H)
        assert not fits_frame(pos, _pos(40.0, 40.0), W, H), "a 40 px box at x=630 crosses the right edge"
