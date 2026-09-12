"""The skeleton-graph architectures (SpecScalpel, LaDy) honour the registry contract.

``model(x: (B, F, T), mask: (B, 1, T)) -> (S, B, C, T)`` with S >= 1, finite
values and zeros on padded frames — the same contract the DLC2Action adapters
meet, plus the joint-layout wiring these two add (keypoints → V axis, skeleton →
adjacency, and for LaDy a rooted kinematic tree).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ethograph.segment.models import ARCHITECTURES, as_output, build_model  # noqa: E402
from ethograph.segment.models.skeleton_graph import (  # noqa: E402
    _lady_dof,
    _normalised_adjacency,
    _parents_list,
    pose_to_generalised_coordinates,
    resolve_skeleton,
)
from ethograph.skeleton.graph import Skeleton  # noqa: E402

SKELETON_GRAPH = ["specscalpel", "lady"]

#: A small quadruped-ish tree: a spine (tail→neck→nose) with two hip→knee legs.
KEYPOINTS = ["nose", "neck", "hipL", "hipR", "kneeL", "kneeR", "tail"]
EDGES = [
    ["nose", "neck"],
    ["neck", "hipL"],
    ["neck", "hipR"],
    ["hipL", "kneeL"],
    ["hipR", "kneeR"],
    ["neck", "tail"],
]
LADY_LANDMARKS = {"root": "tail", "spine": "neck", "left": "hipL", "right": "hipR"}

B = 2
T = 512
N_CLASSES = 4
N_PADDED = 50


def _params(name: str, *, skeleton: Any = EDGES) -> dict[str, Any]:
    params: dict[str, Any] = {"keypoints": KEYPOINTS}
    if skeleton is not None:
        params["skeleton"] = skeleton
    if name == "lady":
        params.update(LADY_LANDMARKS)
    return params


def _n_features(channels: int = 3) -> int:
    return len(KEYPOINTS) * channels


def _inputs(n_frames: int = T, channels: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(B, _n_features(channels), n_frames)
    mask = torch.ones(B, 1, n_frames)
    mask[1, :, -N_PADDED:] = 0
    return x, mask


def test_registered() -> None:
    assert set(SKELETON_GRAPH) <= set(ARCHITECTURES)


@pytest.mark.parametrize("name", SKELETON_GRAPH)
def test_forward_contract(name: str) -> None:
    model = build_model(name, _params(name), _n_features(), N_CLASSES)
    x, mask = _inputs()
    model.train()
    out = as_output(model(x, mask)).logits
    assert out.dim() == 4
    assert out.shape[0] >= 1
    assert tuple(out.shape[1:]) == (B, N_CLASSES, T)
    assert torch.isfinite(out).all()
    assert (out[:, 1, :, -N_PADDED:] == 0).all(), "padded frames must be zero"
    assert (out[:, :, :, : T - N_PADDED] != 0).any()
    out.mean().backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", SKELETON_GRAPH)
def test_eval_is_deterministic(name: str) -> None:
    model = build_model(name, _params(name), _n_features(), N_CLASSES).eval()
    x, mask = _inputs()
    with torch.no_grad():
        first = as_output(model(x, mask)).logits
        second = as_output(model(x, mask)).logits
    assert torch.equal(first, second)


def test_lady_runs_on_2d_pose() -> None:
    """The dynamics coordinates have a 2D path (heading + signed joint angles)."""
    model = build_model(
        "lady", {"keypoints": KEYPOINTS, "skeleton": EDGES, "root": "tail", "spine": "neck"}, _n_features(2), N_CLASSES
    )
    x, mask = _inputs(channels=2)
    out = as_output(model(x, mask)).logits
    assert tuple(out.shape[1:]) == (B, N_CLASSES, T)


def test_specscalpel_runs_without_a_skeleton() -> None:
    """No edges is allowed: the adjacency is the identity, the residual learns."""
    model = build_model("specscalpel", _params("specscalpel", skeleton=None), _n_features(), N_CLASSES)
    x, mask = _inputs()
    assert torch.isfinite(as_output(model(x, mask)).logits).all()


@pytest.mark.parametrize("name", SKELETON_GRAPH)
def test_keypoints_required(name: str) -> None:
    with pytest.raises(ValueError, match="model.params.keypoints"):
        build_model(name, {}, _n_features(), N_CLASSES)


@pytest.mark.parametrize("name", SKELETON_GRAPH)
def test_keypoints_must_divide_the_feature_count(name: str) -> None:
    with pytest.raises(ValueError, match="does not divide"):
        build_model(name, _params(name), _n_features() + 1, N_CLASSES)


def test_lady_refuses_non_positional_channels() -> None:
    """`q` is a kinematic quantity, so lady needs 2 or 3 channels per keypoint."""
    with pytest.raises(ValueError, match="2 or 3 channels"):
        build_model("lady", _params("lady"), _n_features(5), N_CLASSES)


def test_lady_requires_its_landmarks() -> None:
    with pytest.raises(ValueError, match="model.params.spine"):
        build_model("lady", {"keypoints": KEYPOINTS, "skeleton": EDGES, "root": "tail"}, _n_features(), N_CLASSES)


def test_lady_refuses_a_skeleton_with_no_edges() -> None:
    with pytest.raises(ValueError, match="needs a kinematic tree"):
        build_model("lady", {"keypoints": KEYPOINTS, "root": "tail", "spine": "neck"}, _n_features(2), N_CLASSES)


def test_lady_refuses_a_skeleton_with_no_articulated_joint() -> None:
    # A depth-1 star from the root: every joint's parent is the root, so none
    # has a grandparent and the dynamics stream has no articulated joint.
    star = [["tail", k] for k in KEYPOINTS if k != "tail"]
    with pytest.raises(ValueError, match="no articulated joint"):
        build_model(
            "lady",
            {"keypoints": KEYPOINTS, "skeleton": star, "root": "tail", "spine": "neck"},
            _n_features(2),
            N_CLASSES,
        )


@pytest.mark.parametrize("name", SKELETON_GRAPH)
def test_unknown_hyperparameter_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="nonsense"):
        build_model(name, {**_params(name), "nonsense": 1}, _n_features(), N_CLASSES)


@pytest.mark.parametrize("name", SKELETON_GRAPH)
def test_a_hyperparameter_override_reaches_the_network(name: str) -> None:
    """`n_stages_asb` sets the number of segmentation stages, so it changes S."""
    model = build_model(name, {**_params(name), "n_stages_asb": 1}, _n_features(), N_CLASSES)
    x, mask = _inputs(n_frames=256)
    assert as_output(model(x, mask)).logits.shape[0] == 1


def test_skeleton_accepts_name_pairs_index_pairs_and_a_config_dict() -> None:
    """Every spelling of model.params.skeleton resolves to the same adjacency."""
    idx = [[KEYPOINTS.index(a), KEYPOINTS.index(b)] for a, b in EDGES]
    config = {"connections": [{"start": a, "end": b} for a, b in EDGES]}
    a_names = resolve_skeleton(EDGES, KEYPOINTS).adjacency(KEYPOINTS)
    a_idx = resolve_skeleton(idx, KEYPOINTS).adjacency(KEYPOINTS)
    a_config = resolve_skeleton(config, KEYPOINTS).adjacency(KEYPOINTS)
    assert np.array_equal(a_names, a_idx)
    assert np.array_equal(a_names, a_config)


def test_parents_orient_the_tree_away_from_the_root() -> None:
    chain = ["a", "b", "c"]
    skeleton = Skeleton(tuple(chain), (("a", "b"), ("b", "c")))
    assert _parents_list(skeleton, chain, "a") == [-1, 0, 1]


def test_lady_dof_counts_root_plus_grandchild_joints() -> None:
    parents = _parents_list(resolve_skeleton(EDGES, KEYPOINTS), KEYPOINTS, "tail")
    # 5 joints have both a parent and a grandparent; root orientation is the rest.
    assert _lady_dof(parents, ndim=3) == 3 + 5 * 3
    assert _lady_dof(parents, ndim=2) == 1 + 5 * 1


def test_normalised_adjacency_is_symmetric_and_finite_with_isolated_nodes() -> None:
    a = Skeleton(("a", "b", "c"), (("a", "b"),)).adjacency(["a", "b", "c"])  # node c isolated
    norm = _normalised_adjacency(a)
    assert np.isfinite(norm).all()
    assert np.allclose(norm, norm.T)


def test_generalised_coordinates_are_invariant_to_a_global_translation() -> None:
    """`q` is built from root-relative positions, so shifting the whole animal leaves it unchanged."""
    parents = _parents_list(resolve_skeleton(EDGES, KEYPOINTS), KEYPOINTS, "tail")
    torch.manual_seed(1)
    pos = torch.randn(1, 3, len(KEYPOINTS), 8)
    shifted = pos + torch.tensor([1.5, -2.0, 0.7]).view(1, 3, 1, 1)
    args = (
        parents,
        KEYPOINTS.index("tail"),
        KEYPOINTS.index("neck"),
        KEYPOINTS.index("hipL"),
        KEYPOINTS.index("hipR"),
        3,
    )
    q0 = pose_to_generalised_coordinates(pos, *args)
    q1 = pose_to_generalised_coordinates(shifted, *args)
    assert torch.allclose(q0, q1, atol=1e-5)
