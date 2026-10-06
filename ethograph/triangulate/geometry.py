"""Triangulated landmarks as a Space plot geometry: ``{project}/space/{name}.yaml``.

Static keypoints — the corners of an arena, a fixed landmark — have one 3D
position. Written as the geometry library's ``references`` (vertices and
indexed edges), they are the room the animal moves in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from ethograph.io.catalog import KEYPOINT_DIMS
from ethograph.io.data_loader import load_features_dataset
from ethograph.skeleton.library import resolve_skeleton
from ethograph.triangulate.session import POSITION_3D
from ethograph.utils.paths import global_setting

#: The geometry library's folder inside a project (``plots_space.GEOMETRY_DIRNAME``).
GEOMETRY_DIRNAME = "space"


def landmark_positions(datasets: Sequence[xr.Dataset], keypoints: Sequence[str]) -> dict[str, np.ndarray]:
    """Each keypoint's median ``position_3d`` across *datasets*, time and individuals; unseen ones are left out."""
    found: dict[str, np.ndarray] = {}
    for name in keypoints:
        seen = []
        for ds in datasets:
            da = ds[POSITION_3D]
            kp_dim = next(d for d in KEYPOINT_DIMS if d in da.dims)
            other = [d for d in da.dims if d != "space" and d != kp_dim]
            values = da.sel({kp_dim: name}).stack(sample=other).transpose("sample", "space").values
            seen.append(values[np.isfinite(values).all(axis=1)])
        stacked = np.concatenate(seen)
        if len(stacked):
            found[name] = np.median(stacked, axis=0)
    return found


def write_geometry(
    path: Path | str,
    points: Mapping[str, np.ndarray],
    edges: Sequence[tuple[str, str]],
    *,
    color: str = "black",
) -> Path:
    """Write *points* (name → xyz) and the *edges* between them as one reference geometry.

    An edge naming a point that is not in *points* is dropped: a landmark
    one camera never saw must not cost the rest of the room.
    """
    path = Path(path)
    names = list(points)
    cfg = {
        "references": [
            {
                "name": path.stem,
                "vertices": [[round(float(v), 6) for v in points[name]] for name in names],
                "edges": [[names.index(a), names.index(b)] for a, b in edges if a in points and b in points],
                "color": color,
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Triangulated landmarks, in vertex order: {', '.join(names)}.\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    return path


def skeleton_edges(project: Path | str, skeleton: str | None = None) -> list[tuple[str, str]]:
    """The connections of the project's skeleton, as keypoint-name pairs (``[]`` when it has none to choose)."""
    config = resolve_skeleton(skeleton, project)
    if config is None:
        return []
    return [(str(c["start"]), str(c["end"])) for c in config["connections"]]


def export_geometry(
    source: Path | str,
    project: Path | str,
    name: str,
    *,
    keypoints: Sequence[str],
    edges: Sequence[tuple[str, str]] | None = None,
    skeleton: str | None = None,
) -> Path:
    """Write a triangulated session's static *keypoints* as ``{project}/space/{name}.yaml``.

    *edges* default to the project skeleton's connections between those
    keypoints. The session must have been triangulated first.
    """
    result = load_features_dataset(str(source), ignore=tuple(global_setting("ignore_files", [])))
    if result.dt is None:
        raise ValueError(f"{source}: geometry is exported from an xarray (.nc) session.")
    datasets = [ds for _, ds in result.dt.trial_items() if POSITION_3D in ds.data_vars]
    if not datasets:
        raise ValueError(f"{source} has no {POSITION_3D!r}; triangulate it first.")
    points = landmark_positions(datasets, keypoints)
    missing = [k for k in keypoints if k not in points]
    if missing:
        raise ValueError(f"No 3D position for {missing}; they were never seen by two cameras.")
    if edges is None:
        edges = skeleton_edges(project, skeleton)
    return write_geometry(Path(project) / GEOMETRY_DIRNAME / f"{name}.yaml", points, edges)
