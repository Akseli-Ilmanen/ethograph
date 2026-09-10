"""How the pose overlay spends colour and text on a multi-animal dataset.

``_build_overlay_style`` is pure enough to drive with a stub data widget: it
reads the properties table, the ``pose_color_by`` setting and the sidebar's
spin boxes, and returns the style the pygfx overlay is built from.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pygfx")

from ethograph.gui.pose_convert import COLOR_BY_INDIVIDUAL, COLOR_BY_KEYPOINT  # noqa: E402
from ethograph.gui.pose_render import PoseDisplayManager  # noqa: E402

KEYPOINTS = ["beak", "tail"]
INDIVIDUALS = ["crow_a", "crow_b"]


class _Value:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def isChecked(self):
        return bool(self._value)


class _StubDataWidget:
    pose_point_size_spin = _Value(10.0)
    pose_text_size_spin = _Value(12.0)
    pose_show_text_checkbox = _Value(True)
    pose_show_keypoints_checkbox = _Value(True)
    pose_skeleton_width_spin = _Value(2.0)


class _StubState:
    def __init__(self, color_by=COLOR_BY_KEYPOINT):
        self.pose_color_by = color_by
        self.pose_points_use_base = False
        self.pose_points_base_color = "#FF3333"
        self.pose_individual_colors = {}

    def label_individuals(self) -> list[str]:
        return list(INDIVIDUALS)


def _manager(color_by=COLOR_BY_KEYPOINT) -> PoseDisplayManager:
    manager = PoseDisplayManager(None, _StubState(color_by), None, _StubDataWidget())
    manager._camera_keypoints = {"cam-1": list(KEYPOINTS)}
    return manager


def _properties(individuals=INDIVIDUALS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"individual": individual, "keypoint": keypoint, "confidence": 1.0}
            for individual in individuals
            for keypoint in KEYPOINTS
        ]
    )


def test_colour_by_keypoint_holds_with_several_individuals():
    """The beak is one colour on every animal — SLEAP's default."""
    style = _manager(COLOR_BY_KEYPOINT)._build_overlay_style(_properties())

    assert style.color_prop == "keypoint"
    assert set(style.color_map) == set(KEYPOINTS)


def test_colour_by_individual_gives_each_animal_one_colour():
    style = _manager(COLOR_BY_INDIVIDUAL)._build_overlay_style(_properties())

    assert style.color_prop == "individual"
    assert set(style.color_map) == set(INDIVIDUALS)


def test_text_carries_whichever_axis_the_colours_do_not():
    """Turning labels on must add information, not repeat the colours."""
    assert _manager(COLOR_BY_KEYPOINT)._build_overlay_style(_properties()).text_prop == "individual"
    assert _manager(COLOR_BY_INDIVIDUAL)._build_overlay_style(_properties()).text_prop == "keypoint"


def test_text_falls_back_when_the_other_axis_has_nothing_to_say():
    """One individual: naming it on every marker is noise, so name the keypoint."""
    style = _manager(COLOR_BY_KEYPOINT)._build_overlay_style(_properties(["crow_a"]))

    assert style.color_prop == "keypoint"
    assert style.text_prop == "keypoint"


def test_bounding_boxes_have_only_one_axis_to_colour():
    """A dataset without keypoints cannot honour colour-by-keypoint."""
    properties = pd.DataFrame([{"individual": name, "confidence": 1.0} for name in INDIVIDUALS])

    style = _manager(COLOR_BY_KEYPOINT)._build_overlay_style(properties)

    assert style.color_prop == "individual"
    assert style.text_prop == "individual"


def test_the_base_colour_still_overrides_every_point():
    manager = _manager(COLOR_BY_INDIVIDUAL)
    manager.app_state.pose_points_use_base = True
    manager.app_state.pose_points_base_color = "#00FF00"

    style = manager._build_overlay_style(_properties())

    assert len(set(style.color_map.values())) == 1
