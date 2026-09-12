"""Napari-free conversion of movement pose datasets to point/bbox arrays.

Replaces ``movement.napari.convert.ds_to_napari_layers`` and the colormap
sampling from ``movement.napari.layer_styles`` so the GUI has no napari
dependency. Output formats are kept identical so the rest of the pose
pipeline (``PoseRenderData``, filtering, overlays) is unchanged:

- points: ``(N, 4)`` array of ``(track_id, frame_idx, y, x)``
- bboxes: ``(N, 4, 4)`` array — 4 corner rows of ``(track_id, frame, y, x)``
- properties: DataFrame with ``individual``/``keypoint``/``time``/``confidence``
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

#: Which axis of the pose hierarchy colour encodes — the value of
#: ``app_state.pose_color_by``. Kept here, with the other colour helpers,
#: because both surfaces that draw a keypoint read it: the pose display
#: (:mod:`~ethograph.gui.pose_render`) and the labelling canvas
#: (:mod:`~ethograph.gui.pose_edit_mixin`).
COLOR_BY_KEYPOINT = "keypoint"
COLOR_BY_INDIVIDUAL = "individual"
COLOR_BY_MODES = (COLOR_BY_KEYPOINT, COLOR_BY_INDIVIDUAL)


def _dim(ds: xr.Dataset, singular: str) -> str | None:
    """Return the dataset's dim name for *singular*, accepting plural too."""
    for name in (singular, singular + "s"):
        if name in ds.dims or name in ds.coords:
            return name
    return None


def _construct_properties_dataframe(ds_stacked: xr.Dataset, kp_dim: str | None, ind_dim: str) -> pd.DataFrame:
    n = len(ds_stacked.coords[ind_dim].values)
    confidence = ds_stacked["confidence"].values.flatten() if "confidence" in ds_stacked else np.ones(n)
    data = {
        "individual": ds_stacked.coords[ind_dim].values,
        "time": ds_stacked.coords["time"].values,
        "confidence": confidence,
    }
    desired_order = list(data.keys())
    if kp_dim is not None:
        data["keypoint"] = ds_stacked.coords[kp_dim].values
        desired_order.insert(1, "keypoint")
    return pd.DataFrame(data).reindex(columns=desired_order)


def poses_ds_to_points(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Convert a movement dataset to tracks-format points, bboxes, properties.

    Mirrors ``ds_to_napari_layers`` (movement) exactly, including corner
    ordering for bounding boxes, but without importing napari.
    """
    ind_dim = _dim(ds, "individual")
    kp_dim = _dim(ds, "keypoint")
    if ind_dim is None:
        raise ValueError("Dataset has no individual(s) dimension.")

    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes[ind_dim]
    n_keypoints = ds.sizes.get(kp_dim, 1) if kp_dim else 1
    n_tracks = n_individuals * n_keypoints

    track_id_col = np.repeat(np.arange(n_tracks), n_frames).reshape(-1, 1)
    time_col = np.tile(np.arange(n_frames), n_tracks).reshape(-1, 1)

    # position dims: (time, space, [keypoint,] individual) -> (individual, [keypoint,] time, space)
    axes_reordering: tuple[int, ...] = (2, 0, 1)
    if kp_dim:
        axes_reordering = (3,) + axes_reordering
    yx_cols = np.transpose(ds.position.values, axes_reordering).reshape(-1, 2)[:, [1, 0]]

    points = np.hstack((track_id_col, time_col, yx_cols))
    bboxes = None

    if "shape" in ds:
        xmin_ymin = ds.position - (ds["shape"] / 2)
        xmax_ymax = ds.position + (ds["shape"] / 2)
        xmax_ymin = xmin_ymin.copy()
        xmax_ymin.loc[{"space": "x"}] = xmax_ymax.loc[{"space": "x"}]
        xmin_ymax = xmin_ymin.copy()
        xmin_ymax.loc[{"space": "y"}] = xmax_ymax.loc[{"space": "y"}]

        corner_arrays = [
            np.c_[
                track_id_col,
                time_col,
                np.transpose(corner.values, axes_reordering).reshape(-1, 2),
            ]
            for corner in [xmin_ymin, xmin_ymax, xmax_ymax, xmax_ymin]
        ]
        corners = np.concatenate(corner_arrays, axis=1).reshape(-1, 4, 4)
        bboxes = corners[:, :, [0, 1, 3, 2]]  # (track_id, time, x, y) -> (..., y, x)

    dims_to_stack: tuple[str, ...] = (ind_dim, "time")
    if kp_dim:
        dims_to_stack += (kp_dim,)
    ds_stacked = ds.stack(tracks=sorted(dims_to_stack))
    properties = _construct_properties_dataframe(ds_stacked, kp_dim, ind_dim)

    return points, bboxes, properties


#: Bright, saturated, far apart in hue — an animal must be told from its
#: neighbour at a glance on top of video. Indexed by the individual's position
#: in the dataset's list; cycles past eight.
INDIVIDUAL_PALETTE: tuple[str, ...] = (
    "#FF3B30",  # red
    "#0A84FF",  # blue
    "#34C759",  # green
    "#FFD60A",  # yellow
    "#FF2DE5",  # magenta
    "#00E5FF",  # cyan
    "#FF9F0A",  # orange
    "#BF5AF2",  # purple
)


def individual_color_map(individuals: list[str], overrides: dict[str, str] | None = None) -> dict[str, tuple]:
    """One colour per individual, by *name*, the same wherever that name is drawn.

    The palette is indexed over the dataset's full individual list, so an
    animal keeps its colour whether it is drawn alone, with the others, or on
    a camera that never sees the rest. *overrides* (``name -> "#RRGGBB"``)
    are the user's own picks and win over the palette.
    """
    from ethograph.skeleton.config import hex_to_rgba

    colors = {
        name: tuple(hex_to_rgba(INDIVIDUAL_PALETTE[i % len(INDIVIDUAL_PALETTE)])) for i, name in enumerate(individuals)
    }
    for name, hex_color in (overrides or {}).items():
        if name in colors and hex_color:
            colors[name] = tuple(hex_to_rgba(hex_color))
    return colors


def sample_colormap(n: int, cmap_name: str = "turbo") -> list[tuple]:
    """Sample n equally-spaced RGBA tuples (0-1 floats) from a colormap."""
    cmap = matplotlib.colormaps[cmap_name]
    if n <= 1:
        return [tuple(cmap(0.0))]
    return [tuple(cmap(i / (n - 1))) for i in range(n)]
