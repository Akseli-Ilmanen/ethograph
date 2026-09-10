"""A dropped ``.nc`` is always features; it is a pose overlay only in its video's pixels.

Several ``.nc`` files dropped together become one session on a ``camera``
dim, so every one of them reaches the add-panel popup. A movement dataset is
drawn on the paired video only when its positions plausibly are that
video's pixels — millimetres, normalised or negative coordinates are still
features, never boxes on the wrong frame.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ethograph.io.nc_drop import concat_on_camera, positions_fit_frame

N = 10


def _bboxes(x: float = 100.0, y: float = 200.0, unit: str | None = None, individuals=("m", "f")) -> xr.Dataset:
    n_ind = len(individuals)
    position = np.zeros((N, 2, n_ind))
    position[:, 0, :] = x
    position[:, 1, :] = y
    attrs = {"ds_type": "bboxes", "fps": 25.0}
    if unit is not None:
        attrs["space_unit"] = unit
    return xr.Dataset(
        {
            "position": (("time", "space", "individual"), position),
            "shape": (("time", "space", "individual"), np.full((N, 2, n_ind), 40.0)),
            "confidence": (("time", "individual"), np.full((N, n_ind), 0.9)),
        },
        coords={"time": np.arange(N) / 25.0, "space": ["x", "y"], "individual": list(individuals)},
        attrs=attrs,
    )


# ----------------------------------------------------------------------
# Pixel plausibility
# ----------------------------------------------------------------------


def test_positions_inside_the_frame_fit():
    assert positions_fit_frame(_bboxes(), 640, 480)


def test_a_box_edge_past_the_frame_does_not_fit():
    # centre 630 with a 40 px box reaches 650 > 640
    assert not positions_fit_frame(_bboxes(x=630.0), 640, 480)
    assert not positions_fit_frame(_bboxes(x=-5.0), 640, 480)


def test_a_space_unit_attr_settles_it():
    assert positions_fit_frame(_bboxes(x=5000.0, unit="px"), 640, 480), "declared pixels: trusted"
    assert not positions_fit_frame(_bboxes(unit="mm"), 640, 480), "declared millimetres: never drawn"


def test_nothing_finite_or_no_frame_size_is_refused():
    ds = _bboxes()
    ds["position"][:] = np.nan
    assert not positions_fit_frame(ds, 640, 480)
    assert not positions_fit_frame(_bboxes(), 0, 0)
    assert not positions_fit_frame(xr.Dataset({"speed": ("time", np.zeros(N))}), 640, 480)


# ----------------------------------------------------------------------
# Concatenation on a camera dim
# ----------------------------------------------------------------------


def test_one_file_is_left_as_it_is():
    ds = concat_on_camera([_bboxes()], ["cam-1"])
    assert "camera" not in ds.dims
    assert ds.attrs == {"fps": 25.0}


def test_several_files_stack_on_camera_with_an_outer_join():
    a = _bboxes(individuals=("m", "f"))
    b = _bboxes(individuals=("m",))
    ds = concat_on_camera([a, b], ["cam-1", "cam-2"])

    assert list(ds.coords["camera"].values) == ["cam-1", "cam-2"]
    assert list(ds.coords["individual"].values) == ["m", "f"]
    assert np.isnan(ds["position"].sel(camera="cam-2", individual="f")).all(), "padded, not invented"
    assert np.isfinite(ds["position"].sel(camera="cam-2", individual="m")).all()
    assert ds.attrs["fps"] == 25.0


def test_files_disagreeing_on_fps_are_refused():
    b = _bboxes()
    b.attrs["fps"] = 30.0
    with pytest.raises(ValueError, match="fps"):
        concat_on_camera([_bboxes(), b], ["cam-1", "cam-2"])
    with pytest.raises(ValueError, match="camera names"):
        concat_on_camera([_bboxes()], ["cam-1", "cam-2"])


def test_a_plain_features_file_joins_the_stack():
    feats = xr.Dataset({"speed": ("time", np.ones(N))}, coords={"time": np.arange(N) / 25.0}, attrs={"fps": 25.0})
    ds = concat_on_camera([_bboxes(), feats], ["cam-1", "arena"])
    assert {"position", "speed"} <= set(ds.data_vars)
    assert np.isnan(ds["speed"].sel(camera="cam-1")).all()
    assert (ds["speed"].sel(camera="arena") == 1).all()
