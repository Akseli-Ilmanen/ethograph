"""A mesh is placed on a pose by its landmarks, in whatever units and pose the data has."""

from pathlib import Path

import numpy as np
import pytest

from ethograph.skeleton.mesh import (
    MeshModel,
    fit_scale,
    place_mesh,
    read_ply,
)


def _tetra() -> MeshModel:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    landmarks = {"a": vertices[1].astype(float), "b": vertices[2].astype(float), "c": vertices[3].astype(float)}
    return MeshModel("tetra", vertices, faces, landmarks)


def test_place_mesh_recovers_rotation_scale_and_shift() -> None:
    model = _tetra()
    rotation = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    rotation *= np.linalg.det(rotation)  # a proper rotation, never a mirror
    keypoints = ["x", "a", "b", "c"]
    pose = np.array([[np.nan] * 3] + [12.0 * rotation @ model.landmarks[k] + 3.0 for k in keypoints[1:]])

    placed = place_mesh(model, keypoints, pose)

    assert placed is not None
    np.testing.assert_allclose(placed, 12.0 * model.vertices @ rotation.T + 3.0, atol=1e-4)


def test_place_mesh_needs_three_visible_landmarks() -> None:
    model = _tetra()
    pose = np.array([[1.0, 0, 0], [0, 1.0, 0], [np.nan] * 3])
    assert place_mesh(model, ["a", "b", "c"], pose) is None


def test_read_ply_reads_binary_triangles(tmp_path: Path) -> None:
    model = _tetra()
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\nproperty float nx\n"
        "element face 4\nproperty list uchar int vertex_indices\nend_header\n"
    )
    vertex_rows = np.hstack([model.vertices, np.zeros((4, 1), np.float32)]).astype("<f4").tobytes()
    face_rows = b"".join(b"\x03" + f.astype("<i4").tobytes() for f in model.faces)
    path = tmp_path / "tetra.ply"
    path.write_bytes(header.encode() + vertex_rows + face_rows)

    vertices, faces, colors = read_ply(path)

    np.testing.assert_array_equal(vertices, model.vertices)
    np.testing.assert_array_equal(faces, model.faces)
    assert colors is None


def test_two_landmarks_place_an_axisymmetric_mesh() -> None:
    """A stick needs only its two points: the line through them fixes it up to a roll it does not have."""
    vertices = np.array([[0, 0, 0], [10, 0, 0], [5, 1, 0]], dtype=np.float32)
    stick = MeshModel(
        "stick", vertices, np.array([[0, 1, 2]], np.int32), {"tip": vertices[0] * 1.0, "prox": vertices[1] * 1.0}
    )
    placed = place_mesh(stick, ["tip", "prox"], np.array([[1.0, 1, 1], [1, 1, 21]]))

    assert placed is not None
    np.testing.assert_allclose(placed[:2], [[1, 1, 1], [1, 1, 21]], atol=1e-5)


def test_fit_scale_is_the_median_so_one_bad_frame_does_not_resize_the_mesh() -> None:
    model = _tetra()
    keypoints = ["a", "b", "c"]
    frames = np.array([[s * model.landmarks[k] for k in keypoints] for s in (2.0, 2.0, 2.0, 9.0)])

    assert fit_scale(model, keypoints, frames) == pytest.approx(2.0)


def test_vertex_colours_follow_their_faces() -> None:
    """pyqtgraph's smooth shading needs colours per vertex; a band's vertices carry the band's colour."""
    model = _tetra()
    faces = np.array([[0, 1, 2], [1, 2, 3]], np.int32)
    red, black = [1.0, 0, 0, 1], [0.0, 0, 0, 1]
    coloured = MeshModel("m", model.vertices, faces, model.landmarks, np.array([red, black], np.float32))

    colors = coloured.vertex_colors

    assert colors is not None
    np.testing.assert_array_equal(colors[[0, 3]], [red, black])
