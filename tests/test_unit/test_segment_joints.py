"""The joint layout reads (channel, keypoint) slots off materialised column names alone."""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.features.columns import column_name, parse_column_name
from ethograph.segment.joints import JointLayout


def _names(order: str) -> list[str]:
    keypoints, space = ["nose", "tail"], ["x", "y"]
    if order == "keypoint_major":
        pairs = [(k, s) for k in keypoints for s in space]
    else:
        pairs = [(k, s) for s in space for k in keypoints]
    return [column_name("position", {"individual": "self", "keypoint": k, "space": s}) for k, s in pairs]


@pytest.mark.parametrize(
    "name",
    [
        column_name("speed", {}),
        column_name("position", {"keypoint": "nose", "space": "x"}, derivative=True),
        column_name("heading", {"individual": "other1"}, circular="sin"),
        column_name("heading", {"individual": "self"}, derivative=True, circular="cos"),
    ],
)
def test_parse_column_name_inverts_column_name(name: str) -> None:
    col = parse_column_name(name)
    assert column_name(col.feature, col.selections, col.derivative, col.circular) == name


@pytest.mark.parametrize("order", ["keypoint_major", "feature_major"])
def test_index_points_every_slot_at_its_own_column(order: str) -> None:
    names = _names(order)
    layout = JointLayout.from_names(names)
    for v, kp in enumerate(layout.keypoints):
        for c, key in enumerate(layout.channels):
            col = parse_column_name(names[layout.index[v, c]])
            assert col.selections["keypoint"] == kp
            assert col.selections["space"] == key.value("space")


def test_sin_and_cos_are_separate_channels() -> None:
    names = [column_name("angle", {"keypoint": k}, circular=c) for k in ("nose", "tail") for c in ("sin", "cos")]
    assert JointLayout.from_names(names).n_channels == 2


def test_column_without_keypoint_is_refused() -> None:
    with pytest.raises(ValueError, match="no keypoint dim"):
        JointLayout.from_names([*_names("keypoint_major"), column_name("speed", {})])


def test_keypoints_with_different_channels_are_refused() -> None:
    names = _names("keypoint_major")[:-1]
    with pytest.raises(ValueError, match="same channels"):
        JointLayout.from_names(names)


def test_coordinate_groups_split_individuals_and_order_space() -> None:
    names = [
        column_name("position", {"individual": ind, "keypoint": k, "space": s})
        for k in ("nose", "tail")
        for ind in ("self", "other1")
        for s in ("y", "x")
    ]
    names += [
        column_name("position", {"individual": "self", "keypoint": k, "space": "x"}, derivative=True)
        for k in ("nose", "tail")
    ]
    layout = JointLayout.from_names(names)
    groups = {g.individual: g for g in layout.coordinate_groups()}
    assert set(groups) == {"self", "other1"}
    for ind, group in groups.items():
        assert group.space == ("x", "y")
        labels = [layout.channels[c] for c in group.channels]
        assert [k.value("space") for k in labels] == ["x", "y"]
        assert all(k.individual == ind and not k.derivative for k in labels)
    assert np.unique(layout.index).size == len(names)
