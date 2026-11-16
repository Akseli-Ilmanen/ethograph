"""Rigid body parts drawn on a pose — e.g. a skull or a stick instead of the bones between their keypoints.

A mesh lives in ``{project}/meshes/`` as a binary PLY beside a
``{stem}.landmarks.yaml`` that names where keypoints sit in the mesh's own
coordinates::

    landmarks:
      beakTip: [32.0, 0.4, -9.2]
      headPost: [-38.0, 0.0, 6.5]

A mesh is rigid: each frame it is moved by the rotation and shift that best
map its visible landmarks onto the keypoints in the least-squares sense
(Umeyama 1991) — every landmark pulls equally, none is privileged. Its scale is
one number per animal, the median of the per-frame best scales
(:func:`fit_scale`), so the mesh's units need not match the data's and labelling
noise does not make it breathe. A mesh with only two landmarks is taken as
symmetric about the line through them (a stick), so its roll is not needed.
Only a mesh with a landmarks file is offered: without one it cannot be placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

MESH_DIR = "meshes"
LANDMARKS_SUFFIX = ".landmarks.yaml"
#: Fewer matched landmarks than this leave the rotation undetermined
#: (two suffice for a mesh symmetric about its landmark line).
MIN_LANDMARKS = 3

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
    "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}  # fmt: skip


@dataclass(frozen=True)
class MeshModel:
    name: str
    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int32
    landmarks: dict[str, np.ndarray]
    #: ``(F, 4)`` RGBA in 0..1 from the file's per-face colours; ``None`` draws the default colour.
    face_colors: np.ndarray | None = None

    @property
    def vertex_colors(self) -> np.ndarray | None:
        """``(V, 4)`` RGBA: each vertex takes the colour of a face it belongs to.

        Exact when no vertex is shared by faces of different colours — a coloured
        mesh keeps its colour bands on vertices of their own, so the edges stay sharp.
        """
        if self.face_colors is None:
            return None
        colors = np.ones((len(self.vertices), 4), dtype=np.float32)
        colors[self.faces.ravel()] = np.repeat(self.face_colors, 3, axis=0)
        return colors

    @property
    def min_landmarks(self) -> int:
        return min(MIN_LANDMARKS, len(self.landmarks))


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Vertices ``(V, 3)``, triangles ``(F, 3)`` and per-face RGBA ``(F, 4)`` (``None``
    when the faces carry no ``red``/``green``/``blue``) of a binary little-endian PLY."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        elements: list[tuple[str, int, list[tuple[str, ...]]]] = []
        while (line := fh.readline().decode("ascii").strip()) != "end_header":
            words = line.split()
            if words[0] == "format" and words[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary_little_endian PLY is supported, not {words[1]}")
            if words[0] == "element":
                elements.append((words[1], int(words[2]), []))
            elif words[0] == "property":
                elements[-1][2].append(tuple(words[1:]))
        body = fh.read()

    vertices = faces = colors = None
    offset = 0
    for name, count, props in elements:
        if props and props[0][0] == "list":
            _, count_type, index_type, _ = props[0]
            n_type, i_type = np.dtype("<" + _PLY_TYPES[count_type]), np.dtype("<" + _PLY_TYPES[index_type])
            extra = [(q[1], "<" + _PLY_TYPES[q[0]]) for q in props[1:]]
            row = np.dtype([("n", n_type), ("i", i_type, 3), *extra])
            data = np.frombuffer(body, dtype=row, count=count, offset=offset)
            if (data["n"] != 3).any():
                raise ValueError(f"{path}: only triangle meshes are supported")
            faces = data["i"].astype(np.int32)
            if {"red", "green", "blue"} <= set(data.dtype.names or ()):
                rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1) / 255.0
                colors = np.hstack([rgb, np.ones((count, 1))]).astype(np.float32)
        else:
            row = np.dtype([(p[1], "<" + _PLY_TYPES[p[0]]) for p in props])
            data = np.frombuffer(body, dtype=row, count=count, offset=offset)
            if name == "vertex":
                vertices = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        offset += row.itemsize * count
    if vertices is None or faces is None:
        raise ValueError(f"{path} has no vertex or face element")
    return vertices, faces, colors


def available_meshes(project: Path | None) -> list[Path]:
    """The project's placeable meshes: PLY files with a landmarks file beside them."""
    if project is None or not (project / MESH_DIR).is_dir():
        return []
    return sorted(p for p in (project / MESH_DIR).glob("*.ply") if landmarks_path(p).is_file())


def landmarks_path(ply: Path) -> Path:
    return ply.with_name(ply.stem + LANDMARKS_SUFFIX)


def load_mesh(ply: Path) -> MeshModel:
    vertices, faces, colors = read_ply(ply)
    with open(landmarks_path(ply)) as fh:
        raw = yaml.safe_load(fh)["landmarks"]
    landmarks = {str(k): np.asarray(v, dtype=np.float64) for k, v in raw.items()}
    return MeshModel(ply.stem, vertices, faces, landmarks, colors)


def similarity_fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """``(s, R, t)`` minimising ``|| s R src + t - dst ||`` over ``(N, 3)`` point pairs."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    u, sigma, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    var_s = (a**2).sum() / len(src)
    scale = float((sigma * d).sum() / var_s) if var_s > 0 else 1.0
    return scale, rotation, mu_d - scale * rotation @ mu_s


def _pairs(
    model: MeshModel, keypoints: list[str] | tuple[str, ...], xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = [
        (model.landmarks[k], xyz[i])
        for i, k in enumerate(keypoints)
        if k in model.landmarks and np.isfinite(xyz[i]).all()
    ]
    if len(pairs) < model.min_landmarks:
        return None
    src, dst = zip(*pairs)
    return np.array(src), np.array(dst)


def fit_scale(model: MeshModel, keypoints: list[str] | tuple[str, ...], coords: np.ndarray) -> float | None:
    """The animal's mesh scale: the median best scale over frames ``coords`` ``(T, K, 3)``."""
    scales = []
    for xyz in coords:
        pairs = _pairs(model, keypoints, xyz)
        if pairs is not None:
            scales.append(similarity_fit(*pairs)[0])
    return float(np.median(scales)) if scales else None


def place_mesh(
    model: MeshModel, keypoints: list[str] | tuple[str, ...], xyz: np.ndarray, scale: float | None = None
) -> np.ndarray | None:
    """The mesh's vertices on one animal's pose ``xyz`` ``(K, 3)`` at *scale*
    (this frame's best when ``None``), or ``None`` when too few landmarks are visible."""
    pairs = _pairs(model, keypoints, xyz)
    if pairs is None:
        return None
    src, dst = pairs
    best, rotation, _ = similarity_fit(src, dst)
    s = best if scale is None else scale
    shift = dst.mean(axis=0) - s * rotation @ src.mean(axis=0)
    return (s * model.vertices @ rotation.T + shift).astype(np.float32)
