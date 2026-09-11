"""What a dropped ``.nc`` is allowed to be: features always, a pose overlay only
when it is plausibly in the paired video's pixels.

A movement dataset (``ds_type`` poses/bboxes with ``position``) is drawn on
its camera *iff* :func:`positions_fit_frame` — ``position`` in millimetres,
normalised or negative is still a perfectly good feature, it just has no
business on the video. Several ``.nc`` files dropped together become one
session dataset stacked on a ``camera`` dim (:func:`concat_on_camera`), so
every one of them reaches the add-panel popup. Qt-free.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

from ethograph.io.overlay_source import PIXEL_UNITS, fits_frame

__all__ = ["PIXEL_UNITS", "concat_on_camera", "positions_fit_frame"]
#: Coordinates whose order is meaningful (a palette is indexed by it).
_NAMED_DIMS = ("individual", "individuals", "keypoint", "keypoints")


def positions_fit_frame(ds: xr.Dataset, width: int, height: int) -> bool:
    """Whether ``position`` plausibly lives in a *width* × *height* pixel frame.

    A ``space_unit`` attr settles it when present (pixels: yes, anything
    else: no). Otherwise every finite x must lie in ``[0, width]`` and every
    finite y in ``[0, height]``, box edges (``position ± shape/2``) included;
    a dataset with no finite position at all cannot be placed and is refused.
    """
    if "position" not in ds or "space" not in ds["position"].dims:
        return False
    pos = ds["position"]
    if "space_unit" in ds.attrs and "space_unit" not in pos.attrs:
        pos = pos.assign_attrs(space_unit=ds.attrs["space_unit"])
    return fits_frame(pos, ds["shape"] if "shape" in ds else None, width, height)


def concat_on_camera(datasets: list[xr.Dataset], names: list[str]) -> xr.Dataset:
    """One session dataset from the ``.nc`` files of one drop.

    Two or more stack on a new ``camera`` dim named by *names* (outer join,
    so files with different individuals or keypoints pad with NaN); one is
    returned as it is. The only attrs kept are ``fps`` — which every file
    that has one must agree on — and ``source_software`` when they all
    agree, since NetCDF cannot store the ``None`` values movement leaves.
    """
    if len(datasets) != len(names):
        raise ValueError(f"{len(datasets)} datasets but {len(names)} camera names")
    if not datasets:
        raise ValueError("No datasets to combine")
    rates = {float(ds.attrs["fps"]) for ds in datasets if ds.attrs.get("fps")}
    if len(rates) > 1:
        raise ValueError(f"The dropped .nc files disagree on fps: {sorted(rates)}")
    softwares = {str(ds.attrs["source_software"]) for ds in datasets if ds.attrs.get("source_software")}

    if len(datasets) == 1:
        ds = datasets[0].copy()
    else:
        cams = pd.Index(names, name="camera")
        ds = xr.concat(datasets, dim=cams, join="outer", combine_attrs="drop")
        # The outer join sorts every coordinate; keep the names in the order
        # the files list them, so an individual's slot does not move.
        for dim in _NAMED_DIMS:
            if dim in ds.coords:
                order = list(dict.fromkeys(str(v) for d in datasets if dim in d.coords for v in d.coords[dim].values))
                ds = ds.reindex({dim: order})
    ds.attrs = {}
    if rates:
        ds.attrs["fps"] = rates.pop()
    if len(softwares) == 1:
        ds.attrs["source_software"] = softwares.pop()
    return ds
