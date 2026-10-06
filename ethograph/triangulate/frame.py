"""The world frame: ``{project}/calibration/{name}.frame.yaml``, beside its calibration.

Triangulation speaks the calibration's frame — a camera's, tilted against the
room, in the calibration board's units. A world frame names two axes and an
origin by keypoint, in Anipose's ``config.toml`` spelling, plus one known
length for the scale::

    axes:
      - [x, box1, box2]
      - [z, box1, box5]
    reference_point: box1
    scale: {from: box1, to: box2, length: 0.30}   # optional
    unit: m                                       # optional

The first fit writes the resolved ``rotation`` / ``origin`` / ``scale_factor``
back into the file, so every session of the rig lands in the same frame
whether or not it sees the landmarks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_AXES = "xyz"


class WorldFrameError(ValueError):
    """A world frame whose definition is malformed or whose landmarks are missing."""


@dataclass(frozen=True)
class WorldTransform:
    """``world = (calibration @ rotation.T - origin) * scale_factor``."""

    rotation: np.ndarray  # (3, 3), orthonormal
    origin: np.ndarray  # (3,), in rotated calibration units
    scale_factor: float = 1.0

    def apply(self, points: np.ndarray) -> np.ndarray:
        """*points* ``(..., 3)`` in the world frame."""
        return (points @ self.rotation.T - self.origin) * self.scale_factor


@dataclass(frozen=True)
class WorldFrame:
    axes: tuple[tuple[str, str, str], tuple[str, str, str]]
    reference_point: str
    scale: tuple[str, str, float] | None = None
    unit: str | None = None
    transform: WorldTransform | None = None

    @property
    def landmarks(self) -> list[str]:
        names = [n for _, a, b in self.axes for n in (a, b)] + [self.reference_point]
        if self.scale is not None:
            names += list(self.scale[:2])
        return list(dict.fromkeys(names))

    def fit(self, points: Mapping[str, np.ndarray]) -> WorldTransform:
        """The transform that puts the landmarks' calibration-frame *points* on the named axes.

        The first axis is taken as given; the second is made perpendicular to
        it, so the result is rigid even when the landmarks are not square.
        """
        missing = [n for n in self.landmarks if n not in points or not np.all(np.isfinite(points[n]))]
        if missing:
            raise WorldFrameError(f"World frame landmarks without a 3D position: {missing}.")

        (axis_a, a0, a1), (axis_b, b0, b1) = self.axes
        ia, ib = _AXES.index(axis_a), _AXES.index(axis_b)
        ic = ({0, 1, 2} - {ia, ib}).pop()
        va = _unit(points[a1] - points[a0])
        vb = points[b1] - points[b0]
        vb = _unit(vb - (vb @ va) * va)
        rotation = np.zeros((3, 3))
        rotation[ia], rotation[ib] = va, vb
        rotation[ic] = np.cross(va, vb) if (ia, ib) in {(0, 1), (1, 2), (2, 0)} else np.cross(vb, va)

        scale_factor = 1.0
        if self.scale is not None:
            s0, s1, length = self.scale
            scale_factor = length / float(np.linalg.norm(points[s1] - points[s0]))
        return WorldTransform(rotation, rotation @ points[self.reference_point], scale_factor)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise WorldFrameError("A world frame axis runs between two landmarks at the same position.")
    return v / norm


def frame_path(calibration_path: Path | str) -> Path:
    """The world-frame file belonging to a calibration file."""
    return Path(calibration_path).with_suffix(".frame.yaml")


def load_frame(calibration_path: Path | str) -> WorldFrame | None:
    """The world frame beside *calibration_path*; ``None`` when the rig has none."""
    path = frame_path(calibration_path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    axes = [tuple(str(v) for v in entry) for entry in cfg["axes"]]
    if len(axes) != 2 or any(len(e) != 3 or e[0] not in _AXES for e in axes) or axes[0][0] == axes[1][0]:
        raise WorldFrameError(f"{path}: `axes` is two entries [axis, from, to] naming two different axes of x/y/z.")
    scale = cfg.get("scale")
    transform = None
    if "rotation" in cfg:
        transform = WorldTransform(
            np.array(cfg["rotation"], dtype=np.float64),
            np.array(cfg["origin"], dtype=np.float64),
            float(cfg["scale_factor"]),
        )
    return WorldFrame(
        axes=(axes[0], axes[1]),  # type: ignore[arg-type]
        reference_point=str(cfg["reference_point"]),
        scale=(str(scale["from"]), str(scale["to"]), float(scale["length"])) if scale else None,
        unit=cfg.get("unit"),
        transform=transform,
    )


def save_frame(calibration_path: Path | str, frame: WorldFrame) -> Path:
    cfg: dict = {"axes": [list(entry) for entry in frame.axes], "reference_point": frame.reference_point}
    if frame.scale is not None:
        cfg["scale"] = {"from": frame.scale[0], "to": frame.scale[1], "length": frame.scale[2]}
    if frame.unit is not None:
        cfg["unit"] = frame.unit
    if frame.transform is not None:
        cfg["rotation"] = frame.transform.rotation.tolist()
        cfg["origin"] = frame.transform.origin.tolist()
        cfg["scale_factor"] = frame.transform.scale_factor
    path = frame_path(calibration_path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    return path
