"""Skeleton-graph architectures: SpecScalpel and LaDy under the registry contract.

Both come from Haoyu Ji's skeleton-based action-segmentation line (MIT), vendored
in ``ethograph/segment/{specscalpel,lady}/`` — see each directory's ``NOTICE.md``.
They are **pose-native**: unlike the DLC2Action models (which read one flat
``(F, T)`` feature block), a graph model reads a *joint layout* ``(C, V, T)`` — a
skeleton of ``V`` keypoints, each carrying ``C`` channels — and a **skeleton
adjacency** over those keypoints. This module is the one place that layout and
that adjacency are built, so the rest of the pipeline still speaks the flat
``(B, F, T)`` registry contract and nothing downstream changes.

The bridge is the same one MotionBERT already relies on: the model's feature
columns are ``V`` contiguous blocks of ``C`` channels, so a pose feature listed
with its **keypoint dim first** (``position: {keypoint: [...], space: [x, y]}``)
materialises keypoint-major and folds straight back to ``(C, V, T)``. The
keypoint order and the skeleton are named in ``model.params`` because the
registry only hands a builder the feature *count*, never the column identities:

* ``keypoints`` — the ordered keypoint names, the ``V`` axis. ``len(keypoints)``
  must divide ``n_features``; ``C = n_features // len(keypoints)``.
* ``skeleton`` — the connectivity. Resolved through
  :class:`ethograph.skeleton.graph.Skeleton`, so it may be a path to an
  ``.nwb`` (an ndx-pose ``Skeleton`` — the interchange standard), a path to a
  YAML (this project's skeleton config or a ``{nodes, edges}`` mapping), an
  inline config/``{nodes, edges}`` dict, or a list of ``[a, b]`` name **or**
  index pairs. Absent or edgeless is allowed for SpecScalpel: the adjacency is
  then the identity and the model's learnable residual (``A_res``) discovers
  structure on its own — the ndx-pose "empty edges = unknown, not none"
  reading. LaDy needs real edges, since its dynamics stream needs a tree.

``as_output`` normalises what these return, exactly as for the DLC2Action
adapters. Defaults come from ``{model}/config/defaults.yaml`` (upstream's own
architecture hyperparameters, dataset specifics stripped) and are overridden
key by key by ``model.params``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from ethograph.segment.lady.generalized_coordinates import get_derivatives
from ethograph.segment.lady.models.LaDy import Model as LaDyNet
from ethograph.segment.models import register_architecture
from ethograph.segment.specscalpel.models.SpecScalpel import Model as SpecScalpelNet
from ethograph.skeleton.graph import Skeleton

SPECSCALPEL_DEFAULTS = Path(__file__).parent.parent / "specscalpel" / "config" / "defaults.yaml"
LADY_DEFAULTS = Path(__file__).parent.parent / "lady" / "config" / "defaults.yaml"

#: ``model.params`` keys this module consumes itself rather than passing to the
#: upstream constructor — the joint layout and skeleton, plus LaDy's landmarks.
_LAYOUT_KEYS = frozenset({"keypoints", "skeleton", "root", "spine", "left", "right"})

#: Registry name → its architecture-defaults YAML, for :func:`tunable_params`.
_DEFAULTS_FILE = {"specscalpel": SPECSCALPEL_DEFAULTS, "lady": LADY_DEFAULTS}


def tunable_params(architecture: str) -> dict[str, Any]:
    """Hyperparameters ``model.params`` accepts for a skeleton-graph *architecture*.

    The joint-layout keys (``keypoints``, ``skeleton``, and LaDy's landmarks)
    are structural inputs, not a search space, so they are not listed here — a
    sweep tunes only the architecture numbers.
    """
    path = _DEFAULTS_FILE[architecture]
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# Joint layout + skeleton adjacency
# ---------------------------------------------------------------------------


def _keypoints(params: dict[str, Any], n_features: int) -> list[str]:
    keypoints = params.get("keypoints")
    if not keypoints:
        raise ValueError(
            "A skeleton-graph architecture needs model.params.keypoints: the ordered keypoint "
            "names of the pose feature (the V axis). List the pose feature with its keypoint dim "
            "first so its columns materialise keypoint-major, e.g. "
            "position: {keypoint: [nose, ear_l, ...], space: [x, y]}."
        )
    keypoints = [str(k) for k in keypoints]
    if n_features % len(keypoints):
        raise ValueError(
            f"model.params.keypoints has {len(keypoints)} keypoints, which does not divide the "
            f"{n_features} feature columns of this dataset. Every keypoint must carry the same "
            "number of channels."
        )
    return keypoints


def resolve_skeleton(spec: Any, keypoints: list[str]) -> Skeleton:
    """A ``model.params.skeleton`` value → a :class:`Skeleton` on *keypoints*.

    Accepts a path (``.nwb`` ndx-pose or a YAML), an inline config /
    ``{nodes, edges}`` dict, a list of ``[a, b]`` name-or-index pairs, or
    ``None`` / empty (an edgeless skeleton). The result is restricted to
    *keypoints*, so a skeleton authored over a superset (a full ndx-pose file)
    lines up with the columns this run actually selected.
    """
    if spec is None or (isinstance(spec, (list, tuple)) and len(spec) == 0):
        return Skeleton(tuple(keypoints), ())
    if isinstance(spec, (str, Path)):
        skeleton = Skeleton.load(spec)
    elif isinstance(spec, dict):
        skeleton = Skeleton.from_config(spec) if "connections" in spec else Skeleton.from_dict(spec)
    else:
        index = {name: i for i, name in enumerate(keypoints)}
        pairs = []
        for a, b in spec:
            na = str(a) if str(a) in index else keypoints[int(a)]
            nb = str(b) if str(b) in index else keypoints[int(b)]
            pairs.append((na, nb))
        skeleton = Skeleton(tuple(keypoints), tuple(pairs))
    missing = [k for k in keypoints if k not in skeleton.nodes]
    if missing:
        raise ValueError(
            f"model.params.skeleton does not describe keypoint(s) {missing}; it names {list(skeleton.nodes)}."
        )
    return skeleton.restricted(keypoints)


def _normalised_adjacency(a_binary: np.ndarray) -> np.ndarray:
    """``D^-1/2 (A + I) D^-1/2`` — the graph passed to the models' CTR-GC blocks.

    Stands in for upstream's text-similarity ``joint_graph``: a structural prior
    the attention is added to, so an isolated node (identity only) is simply
    left to the learnable residual.
    """
    a = np.asarray(a_binary, dtype=np.float32) + np.eye(len(a_binary), dtype=np.float32)
    deg = a.sum(1)
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
    return (inv_sqrt[:, None] * a * inv_sqrt[None, :]).astype(np.float32)


def _to_joint_layout(x: torch.Tensor, num_keypoints: int) -> torch.Tensor:
    """``(B, F, T)`` keypoint-major → ``(B, C, V, T)`` with ``C = F // V``."""
    b, f, t = x.shape
    c = f // num_keypoints
    return x.reshape(b, num_keypoints, c, t).permute(0, 2, 1, 3).contiguous()


def _stack_stages(result: Any, mask: torch.Tensor) -> torch.Tensor:
    """A model's train/eval output → ``(S, B, C, T)`` with padded frames zeroed.

    Training returns ``(outputs_cls, ...)`` where ``outputs_cls`` is a list of
    per-stage ``(B, C, T)`` logits; evaluation returns ``(out_cls, out_bound)``
    with a single ``(B, C, T)``. The registry reads ``logits[-1]``, and these
    models already order their stages coarse→fine, so no flip is needed.
    """
    out_cls = result[0]
    stages = out_cls if isinstance(out_cls, (list, tuple)) else [out_cls]
    logits = torch.stack(list(stages), dim=0)
    return logits * mask.unsqueeze(0)


class SkeletonGraphModel(nn.Module):
    """Adapt a skeleton-graph ``Model`` to ``forward(x, mask) -> (S, B, C, T)``.

    Holds the joint count and the normalised skeleton adjacency (a buffer, so it
    follows the model's device), reshapes the flat batch into ``(B, C, V, T)``
    and stacks the model's per-stage logits back to the contract shape.
    """

    def __init__(self, inner: nn.Module, num_keypoints: int, adjacency: np.ndarray) -> None:
        super().__init__()
        self.inner = inner
        self.num_keypoints = int(num_keypoints)
        self.register_buffer("joint_graph", torch.from_numpy(_normalised_adjacency(adjacency)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = _to_joint_layout(x, self.num_keypoints)
        result = self.inner(pos, mask, self.joint_graph)
        return _stack_stages(result, mask)


def _defaults(path: Path, params: dict[str, Any], **supplied: Any) -> dict[str, Any]:
    """Upstream architecture defaults, overridden by *params* (layout keys removed).

    A ``model.params`` key that is neither a layout key nor an architecture
    hyperparameter is refused here, naming the ones that exist — the same
    contract the DLC2Action builders give.
    """
    defaults = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    hyper = {k: v for k, v in params.items() if k not in _LAYOUT_KEYS}
    unknown = set(hyper) - set(defaults) - set(supplied)
    if unknown:
        raise ValueError(
            f"model.params {sorted(unknown)} are neither a joint-layout key "
            f"({sorted(_LAYOUT_KEYS)}) nor a hyperparameter of this architecture "
            f"({sorted(set(defaults) | set(supplied))}). See {path}."
        )
    return {**defaults, **supplied, **hyper}


@register_architecture("specscalpel")
def build_specscalpel(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """SpecScalpel: multi-scale graph conv → frequency-selective filtering → temporal encoder.

    Defaults from ``specscalpel/config/defaults.yaml``. Reads a pose feature as
    ``V = len(model.params.keypoints)`` keypoints of ``C = n_features // V``
    channels each; ``model.params.skeleton`` supplies the connectivity. Returns
    ``S = n_stages_asb`` stages, ``logits[-1]`` the prediction.
    """
    keypoints = _keypoints(params, n_features)
    v = len(keypoints)
    adjacency = resolve_skeleton(params.get("skeleton"), keypoints).adjacency(keypoints)
    kwargs = _defaults(SPECSCALPEL_DEFAULTS, params)
    net = SpecScalpelNet(
        in_channel=n_features // v,
        n_classes=n_classes,
        node=v,
        A_binary=adjacency,
        **kwargs,
    )
    return SkeletonGraphModel(net, v, adjacency)


# ---------------------------------------------------------------------------
# LaDy — the dynamics stream needs generalised coordinates
# ---------------------------------------------------------------------------


def _parents_list(skeleton: Skeleton, keypoints: list[str], root: str) -> list[int]:
    """:meth:`Skeleton.parents` as a ``child → parent`` list, root and unreached ``-1``."""
    parents = [-1] * len(keypoints)
    for child, parent in skeleton.parents(root, keypoints).items():
        parents[child] = parent
    return parents


def _lady_dof(parents: list[int], ndim: int) -> int:
    """Generalised-coordinate count: root orientation + one rotation per grandchild joint."""
    per = 3 if ndim == 3 else 1
    joints = sum(1 for j, p in enumerate(parents) if p != -1 and parents[p] != -1)
    return per + per * joints


def _landmark(params: dict[str, Any], key: str, keypoints: list[str]) -> int:
    name = params.get(key)
    if name is None:
        raise ValueError(
            f"lady needs model.params.{key}: a keypoint naming the {key} of the root frame. "
            f"Available keypoints: {keypoints}."
        )
    if str(name) not in keypoints:
        raise ValueError(f"model.params.{key}={name!r} is not one of the keypoints {keypoints}.")
    return keypoints.index(str(name))


def _axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(..., 3, 3)`` → axis-angle vectors ``(..., 3)``."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = torch.acos(cos)
    axis = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin = torch.sin(angle).unsqueeze(-1)
    axis = axis / (2 * sin + 1e-8)
    return axis * angle.unsqueeze(-1)


def _root_orientation(rel: torch.Tensor, spine: int, left: int, right: int, ndim: int) -> torch.Tensor:
    """Root-frame orientation of every frame from named landmarks.

    *rel* is ``(N, T, V, C)`` root-relative positions. In 3D the spine axis and
    the left→right axis are orthogonalised (Gram-Schmidt) into a rotation, read
    out as axis-angle; in 2D the heading is the spine direction's ``atan2``.
    """
    if ndim == 2:
        body = rel[:, :, spine, :]
        return torch.atan2(body[..., 1], body[..., 0]).unsqueeze(-1)
    y = torch.nn.functional.normalize(rel[:, :, spine, :], dim=-1)
    x_raw = rel[:, :, right, :] - rel[:, :, left, :]
    z = torch.nn.functional.normalize(torch.cross(x_raw, y, dim=-1), dim=-1)
    x = torch.nn.functional.normalize(torch.cross(y, z, dim=-1), dim=-1)
    rotation = torch.stack([x, y, z], dim=-1)
    return _axis_angle(rotation)


def _joint_rotations(rel: torch.Tensor, parents: list[int], ndim: int) -> torch.Tensor:
    """Per-joint rotation of the child limb relative to the parent limb.

    A joint ``j`` with parent ``p`` and grandparent ``g`` contributes the
    rotation from ``g→p`` to ``p→j`` — axis-angle in 3D, a signed angle in 2D.
    Order follows node index, so the coordinate vector is deterministic.
    """
    out = []
    for j, p in enumerate(parents):
        if p == -1 or parents[p] == -1:
            continue
        g = parents[p]
        vparent = torch.nn.functional.normalize(rel[:, :, p, :] - rel[:, :, g, :], dim=-1)
        vchild = torch.nn.functional.normalize(rel[:, :, j, :] - rel[:, :, p, :], dim=-1)
        if ndim == 2:
            dot = vparent[..., 0] * vchild[..., 0] + vparent[..., 1] * vchild[..., 1]
            cross = vparent[..., 0] * vchild[..., 1] - vparent[..., 1] * vchild[..., 0]
            out.append(torch.atan2(cross, dot).unsqueeze(-1))
        else:
            axis = torch.nn.functional.normalize(torch.cross(vparent, vchild, dim=-1), dim=-1)
            angle = torch.acos(torch.clamp((vparent * vchild).sum(-1), -1.0, 1.0)).unsqueeze(-1)
            out.append(angle * axis)
    if not out:
        return rel.new_zeros((rel.shape[0], rel.shape[1], 0))
    return torch.cat(out, dim=-1)


def pose_to_generalised_coordinates(
    pos: torch.Tensor, parents: list[int], root: int, spine: int, left: int, right: int, ndim: int
) -> torch.Tensor:
    """Positions ``(N, C, V, T)`` → generalised coordinates ``(N, T, dof)``.

    Root-relative each frame, then the root-frame orientation followed by every
    grandchild joint's rotation — the species-agnostic rewrite of upstream's
    per-dataset ``generate_generalized_coordinates``, driven by an explicit
    rooted tree (:meth:`Skeleton.parents`) and named landmarks instead of
    hardcoded human joint maps.
    """
    p = pos.permute(0, 3, 2, 1)  # (N, T, V, C)
    rel = p - p[:, :, root : root + 1, :]
    root_q = _root_orientation(rel, spine, left, right, ndim)
    joint_q = _joint_rotations(rel, parents, ndim)
    return torch.cat([root_q, joint_q], dim=-1)


class LaDyModel(SkeletonGraphModel):
    """SpecScalpel's chassis plus LaDy's Lagrangian-dynamics stream.

    The dynamics stream reads generalised coordinates ``q``; upstream computes
    them in its training loop, but the registry hands us only ``(B, F, T)``, so
    the coordinates are rebuilt inside ``forward`` from the same positions the
    graph stream sees. The pose feature must therefore be raw positions —
    ``C ∈ {2, 3}`` channels per keypoint — because ``q`` is a kinematic quantity
    of the skeleton, not of arbitrary per-keypoint features.
    """

    def __init__(
        self,
        inner: nn.Module,
        num_keypoints: int,
        adjacency: np.ndarray,
        parents: list[int],
        root: int,
        spine: int,
        left: int,
        right: int,
        ndim: int,
    ) -> None:
        super().__init__(inner, num_keypoints, adjacency)
        self.parents = parents
        self.root = root
        self.spine = spine
        self.left = left
        self.right = right
        self.ndim = ndim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = _to_joint_layout(x, self.num_keypoints)
        q = pose_to_generalised_coordinates(pos, self.parents, self.root, self.spine, self.left, self.right, self.ndim)
        result = self.inner(pos, q, mask, self.joint_graph)
        return _stack_stages(result, mask)

    # ``inner`` reads ``q`` via ``get_derivatives`` (imported for provenance so
    # the reference stays with the vendored module, not duplicated here).
    _get_derivatives = staticmethod(get_derivatives)


@register_architecture("lady")
def build_lady(params: dict[str, Any], n_features: int, n_classes: int) -> nn.Module:
    """LaDy: SpecScalpel's spatial-temporal chassis modulated by a learned Lagrangian stream.

    Defaults from ``lady/config/defaults.yaml``. Needs, beyond the shared
    ``keypoints``/``skeleton``, the root frame's landmarks — ``root`` and
    ``spine`` (and ``left``/``right`` in 3D) — raw-position columns
    (``C ∈ {2, 3}``), and a skeleton whose edges form a tree deep enough to
    have articulated joints. Returns ``S = n_stages_asb`` stages.
    """
    keypoints = _keypoints(params, n_features)
    v = len(keypoints)
    ndim = n_features // v
    if ndim not in (2, 3):
        raise ValueError(
            f"lady derives generalised coordinates from raw positions, so each keypoint must carry "
            f"2 or 3 channels; this dataset has {n_features} columns over {v} keypoints ({ndim} each). "
            "List only the position feature (space: [x, y] or [x, y, z]) for lady."
        )
    if params.get("root") is None:
        raise ValueError(
            f"lady needs model.params.root: a keypoint naming the root of the tree. Available: {keypoints}."
        )
    root_name = str(params["root"])
    root = _landmark(params, "root", keypoints)
    spine = _landmark(params, "spine", keypoints)
    left = _landmark(params, "left", keypoints) if ndim == 3 else -1
    right = _landmark(params, "right", keypoints) if ndim == 3 else -1

    skeleton = resolve_skeleton(params.get("skeleton"), keypoints)
    if not skeleton.edges:
        raise ValueError(
            "lady's dynamics stream needs a kinematic tree, but model.params.skeleton has no edges. "
            "Give the skeleton connectivity (a config, an ndx-pose .nwb, or [a, b] pairs), or use "
            "specscalpel, which runs without a tree."
        )
    parents = _parents_list(skeleton, keypoints, str(root_name))
    adjacency = skeleton.adjacency(keypoints)
    dof = _lady_dof(parents, ndim)
    if dof <= (3 if ndim == 3 else 1):
        raise ValueError(
            "lady's dynamics stream found no articulated joint: the skeleton must be a tree at least "
            "three keypoints deep from model.params.root. Add the missing edges, or use specscalpel."
        )
    kwargs = _defaults(LADY_DEFAULTS, params, num_people=1)
    net = LaDyNet(
        in_channel=ndim,
        n_classes=n_classes,
        node=v,
        A_binary=adjacency,
        dof=dof,
        **kwargs,
    )
    return LaDyModel(net, v, adjacency, parents, root, spine, left, right, ndim)
