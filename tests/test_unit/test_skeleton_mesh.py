"""A mesh is placed on a pose by its landmarks, in whatever units and pose the data has."""

from pathlib import Path

import numpy as np
import pytest

from ethograph.skeleton.mesh import (
    MeshModel,
    fit_pose,
    fit_scale,
    landmarks_path,
    load_mesh,
    place_mesh,
    read_ply,
    similarity_fit,
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


def _rotation(seed: int) -> np.ndarray:
    rotation = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))[0]
    return rotation * np.linalg.det(rotation)  # a proper rotation, never a mirror


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


#: A skull in mm: beak along +x, left +y, dorsal +z, mirror-symmetric about y = 0.
_SKULL = {
    "beakTip": [30.0, 0.0, -9.0],
    "beakBase": [-14.0, 0.0, 1.0],
    "leftNostril": [-4.0, 5.0, -4.0],
    "rightNostril": [-4.0, -5.0, -4.0],
    "leye": [-25.0, 11.0, -9.0],
    "reye": [-25.0, -11.0, -9.0],
    "headPost": [-36.0, 0.0, 8.0],
}


def _skull(weights: dict[str, float] | None = None, outlier: float | None = None) -> MeshModel:
    landmarks = {k: np.array(v) for k, v in _SKULL.items()}
    vertices = np.array(list(_SKULL.values()), dtype=np.float32)
    return MeshModel("skull", vertices, np.array([[0, 1, 2]], np.int32), landmarks, None, weights or {}, outlier)


def _pose(model: MeshModel, rotation: np.ndarray, scale: float = 0.1) -> tuple[list[str], np.ndarray]:
    """The model's landmarks as a pose in cm (mesh mm × 0.1), rotated and shifted."""
    keypoints = list(model.landmarks)
    return keypoints, np.array([scale * rotation @ model.landmarks[k] + [5.0, -2.0, 30.0] for k in keypoints])


def test_a_weighted_fit_of_exact_data_recovers_rotation_shift_and_scale() -> None:
    rng = np.random.default_rng(1)
    src, rotation, shift = rng.normal(size=(6, 3)), _rotation(2), np.array([1.0, -2.0, 3.0])

    scale, fitted, moved = similarity_fit(src, 4.0 * src @ rotation.T + shift, rng.uniform(0.1, 2.0, 6))

    assert scale == pytest.approx(4.0)
    np.testing.assert_allclose(fitted, rotation, atol=1e-9)
    np.testing.assert_allclose(moved, shift, atol=1e-9)


def test_a_displaced_landmark_is_dropped_instead_of_tilting_the_mesh() -> None:
    rotation = _rotation(3)
    keypoints, pose = _pose(_skull(), rotation)
    pose[keypoints.index("leye")] += [0.0, 0.0, 5.0]  # 50 mm, in the data's cm

    _, tilted, _ = fit_pose(_skull(), keypoints, pose, scale=0.1)
    _, robust, _ = fit_pose(_skull(outlier=10.0), keypoints, pose, scale=0.1)

    assert _angle_deg(tilted, rotation) > 5.0
    assert _angle_deg(robust, rotation) < 1.0


def test_a_left_eye_guessed_onto_the_right_side_does_not_roll_the_skull() -> None:
    """One mirrored landmark is found and dropped. Two are not: the first drop is then a true one."""
    rotation = _rotation(4)
    weights = {"leye": 0.2, "reye": 0.2}
    keypoints, pose = _pose(_skull(), rotation)
    pose[keypoints.index("leye")] = pose[keypoints.index("reye")]

    _, rolled, _ = fit_pose(_skull(weights), keypoints, pose, scale=0.1)
    _, robust, _ = fit_pose(_skull(weights, outlier=10.0), keypoints, pose, scale=0.1)

    assert _angle_deg(rolled, rotation) > 10.0
    assert _angle_deg(robust, rotation) < 1.0


def test_a_mesh_without_weights_or_outlier_is_placed_as_before() -> None:
    """The unweighted Umeyama fit this module had, on noisy data: old landmarks files change nothing."""
    rng = np.random.default_rng(5)
    model = _skull()
    keypoints, pose = _pose(model, _rotation(6))
    pose += rng.normal(scale=0.3, size=pose.shape)
    src = np.array([model.landmarks[k] for k in keypoints])
    a, b = src - src.mean(axis=0), pose - pose.mean(axis=0)
    u, sigma, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.array([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ np.diag(d) @ vt
    scale = (sigma * d).sum() / ((a**2).sum() / len(src))
    before = scale * model.vertices @ rotation.T + pose.mean(axis=0) - scale * rotation @ src.mean(axis=0)

    np.testing.assert_allclose(place_mesh(model, keypoints, pose), before, atol=1e-5)


def test_landmarks_on_one_line_leave_the_roll_open_so_nothing_is_placed() -> None:
    model = _skull()
    line = ["beakTip", "beakBase", "headPost"]
    on_line = {k: np.array([x, 0.0, 0.0]) for k, x in zip(line, (30.0, -14.0, -36.0))}
    straight = MeshModel("line", model.vertices, model.faces, on_line)

    assert place_mesh(straight, line, np.array([on_line[k] for k in line])) is None


def test_weights_and_outlier_are_read_from_the_landmarks_file_and_optional(tmp_path: Path) -> None:
    model = _tetra()
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 4\nproperty list uchar int vertex_indices\nend_header\n"
    )
    faces = b"".join(b"\x03" + f.astype("<i4").tobytes() for f in model.faces)
    ply = tmp_path / "tetra.ply"
    ply.write_bytes(header.encode() + model.vertices.astype("<f4").tobytes() + faces)
    landmarks = "landmarks: {a: [1, 0, 0], b: [0, 1, 0], c: [0, 0, 1]}\n"

    landmarks_path(ply).write_text(landmarks, encoding="utf-8")
    plain = load_mesh(ply)
    assert (plain.weights, plain.outlier) == ({}, None)

    landmarks_path(ply).write_text(landmarks + "weights: {a: 0.2}\noutlier: 10\n", encoding="utf-8")
    robust = load_mesh(ply)
    assert (robust.weights, robust.outlier) == ({"a": 0.2}, 10.0)

    landmarks_path(ply).write_text(landmarks + "weights: {nose: 0.2}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not landmarks"):
        load_mesh(ply)
