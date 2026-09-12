"""The joint layout: a sample's columns read as ``(channel, keypoint)`` pairs.

The materialised dataset is flat — ``(F, T)``, one column per pinned feature
(:mod:`ethograph.features.columns`). A skeleton architecture wants the same
numbers as ``(C, V, T)``: *V* keypoints, each carrying the same *C* channels
(``position`` x and y, a velocity, the other individual's copy, …). Nothing
new is computed here; :class:`JointLayout` only says which flat column sits at
which ``(channel, keypoint)`` slot, derived from the column names alone, so it
holds for any materialised dataset and any ablation of one.

A column with no keypoint dim has no slot, and is refused: a skeleton model
reads keypoints and nothing else, and silently dropping a configured feature
is the one thing this module must never do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ethograph.features.columns import column_name, parse_column_name
from ethograph.io.catalog import INDIVIDUAL_DIMS, KEYPOINT_DIMS, SPACE_DIM

SELF_TOKEN = "self"
"""The sample's own individual, as the layout spells it (``samples.SELF_TOKEN``)."""

#: The order a coordinate group lists its space channels in.
SPACE_ORDER = ("x", "y", "z")


@dataclass(frozen=True)
class ChannelKey:
    """What one channel is: a feature, its non-keypoint selections, and whether it is a derivative."""

    feature: str
    selections: tuple[tuple[str, str], ...]
    derivative: bool = False
    circular: str | None = None

    @property
    def label(self) -> str:
        return column_name(self.feature, dict(self.selections), self.derivative, self.circular)

    def value(self, dim: str) -> str | None:
        return dict(self.selections).get(dim)

    @property
    def individual(self) -> str | None:
        """The individual token (``self``, ``other1``, …) this channel belongs to, if any."""
        for dim in INDIVIDUAL_DIMS:
            token = self.value(dim)
            if token is not None:
                return token
        return None


@dataclass(frozen=True)
class CoordinateGroup:
    """The channels holding one individual's coordinates, in space order (x, y[, z])."""

    individual: str | None
    feature: str
    channels: tuple[int, ...]
    space: tuple[str, ...]

    @property
    def n_dims(self) -> int:
        return len(self.channels)


@dataclass(frozen=True)
class JointLayout:
    """``(V, C)`` column indices into a flat layout, with what each axis means."""

    keypoints: tuple[str, ...]
    channels: tuple[ChannelKey, ...]
    #: ``index[v, c]`` is the flat column of keypoint *v*'s channel *c*.
    index: np.ndarray

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoints)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @classmethod
    def from_names(cls, names: Sequence[str]) -> JointLayout:
        """Group the flat columns by keypoint; every keypoint must carry the same channels."""
        keypoints: list[str] = []
        per_keypoint: dict[str, dict[ChannelKey, int]] = {}
        flat: list[str] = []
        for i, name in enumerate(names):
            col = parse_column_name(name)
            kp_dim = next((d for d in KEYPOINT_DIMS if d in col.selections), None)
            if kp_dim is None:
                flat.append(name)
                continue
            keypoint = col.selections[kp_dim]
            key = ChannelKey(
                col.feature,
                tuple(sorted((d, v) for d, v in col.selections.items() if d != kp_dim)),
                col.derivative,
                col.circular,
            )
            if keypoint not in per_keypoint:
                keypoints.append(keypoint)
                per_keypoint[keypoint] = {}
            if key in per_keypoint[keypoint]:
                raise ValueError(f"Column {name!r} appears twice in the layout")
            per_keypoint[keypoint][key] = i
        if flat:
            raise ValueError(
                f"A skeleton architecture reads keypoint columns only, but the layout has {len(flat)} "
                f"column(s) with no keypoint dim: {flat[:5]}{'…' if len(flat) > 5 else ''}. "
                "Drop those features from features.columns, or pick an architecture that reads flat columns."
            )
        if not keypoints:
            raise ValueError("The layout has no keypoint columns — nothing for a skeleton architecture to read.")
        channels = tuple(per_keypoint[keypoints[0]])
        for kp in keypoints[1:]:
            have = tuple(per_keypoint[kp])
            if set(have) != set(channels):
                missing = [k.label for k in channels if k not in per_keypoint[kp]]
                extra = [k.label for k in have if k not in per_keypoint[keypoints[0]]]
                raise ValueError(
                    f"Keypoint {kp!r} does not carry the same channels as {keypoints[0]!r}: "
                    f"missing {missing}, extra {extra}. Every keypoint must select the same features and dims."
                )
        index = np.array([[per_keypoint[kp][ch] for ch in channels] for kp in keypoints], dtype=np.int64)
        return cls(tuple(keypoints), channels, index)

    def coordinate_groups(self) -> tuple[CoordinateGroup, ...]:
        """The raw coordinate channels, one group per (individual, feature), each in x, y[, z] order.

        A derivative or a sin/cos component is not a coordinate, and is left out.
        """
        grouped: dict[tuple[str | None, str, tuple[tuple[str, str], ...]], dict[str, int]] = {}
        for c, key in enumerate(self.channels):
            space = key.value(SPACE_DIM)
            if space is None or key.derivative or key.circular is not None:
                continue
            if space not in SPACE_ORDER:
                raise ValueError(f"Channel {key.label!r} has space={space!r}; a coordinate is one of {SPACE_ORDER}")
            rest = tuple((d, v) for d, v in key.selections if d != SPACE_DIM)
            grouped.setdefault((key.individual, key.feature, rest), {})[space] = c
        groups = []
        for (individual, feature, _), by_space in grouped.items():
            space = tuple(s for s in SPACE_ORDER if s in by_space)
            groups.append(CoordinateGroup(individual, feature, tuple(by_space[s] for s in space), space))
        return tuple(groups)
