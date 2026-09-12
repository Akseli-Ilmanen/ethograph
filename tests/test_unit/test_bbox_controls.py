"""Bounding boxes get their own sidebar section, and an individual keeps its colour.

A box has no skeleton and no keypoints: the video's controls for a bboxes
pose file are a confidence filter, text, line width and one colour per
individual. The colour is a property of the *name*, sampled over the
dataset's whole individual list, so the same animal is the same colour on a
camera that sees it alone as on one that sees everybody.
"""

from __future__ import annotations

from qtpy.QtWidgets import QWidget

from ethograph.gui.pose_convert import individual_color_map
from ethograph.gui.right_context import _CONTEXT_MAP, RightContextPanel


def test_an_individual_keeps_its_colour_regardless_of_who_else_is_drawn():
    everyone = ["bird_female", "bird_male", "bird_juv"]
    alone = individual_color_map(everyone)["bird_male"]
    assert individual_color_map(everyone)["bird_male"] == alone
    # The palette is over the full list: a partial or reordered list would
    # hand bird_male another slot, which is exactly what the caller must not do.
    assert individual_color_map(["bird_male"])["bird_male"] != alone
    assert len({tuple(c) for c in individual_color_map(everyone).values()}) == 3


def test_individuals_get_bright_contrasting_colours():
    from ethograph.gui.pose_convert import INDIVIDUAL_PALETTE

    colors = individual_color_map(["a", "b"])
    assert colors["a"][:3] == (1.0, 59 / 255, 48 / 255), "first individual is red"
    assert colors["b"][:3] == (10 / 255, 132 / 255, 1.0), "second is blue"
    many = individual_color_map([str(i) for i in range(len(INDIVIDUAL_PALETTE) + 1)])
    assert many["0"] == many[str(len(INDIVIDUAL_PALETTE))], "the palette cycles"


def test_a_users_pick_wins_over_the_palette():
    colors = individual_color_map(["a", "b"], {"b": "#FF0000", "ghost": "#00FF00"})
    assert colors["b"][:3] == (1.0, 0.0, 0.0)
    assert "ghost" not in colors, "an override for a name not in the dataset is ignored"


def test_video_context_shows_bbox_controls_instead_of_pose_for_boxes(qapp):
    sections = {name: QWidget() for name in ("videocrop", "videolabel", "overlay", "pose", "bbox")}
    panel = RightContextPanel(sections)
    assert {"pose", "bbox"} <= set(_CONTEXT_MAP["video"])

    assert panel.set_context("video", has_pose=True, pose_kind="bboxes")
    assert not sections["pose"].isVisibleTo(panel) and sections["bbox"].isVisibleTo(panel)
    assert sections["overlay"].isVisibleTo(panel), "the overlay chooser sits above either section"

    # Same plot type, other kind of pose file: still a change.
    assert panel.set_context("video", has_pose=True, pose_kind="poses")
    assert sections["pose"].isVisibleTo(panel) and not sections["bbox"].isVisibleTo(panel)
    assert not panel.set_context("video", has_pose=True, pose_kind="poses")
    assert panel.current_context() == "video"

    # No pose at all: neither, whatever the kind claims.
    panel.set_context("video", has_pose=False, pose_kind="bboxes")
    assert not sections["pose"].isVisibleTo(panel) and not sections["bbox"].isVisibleTo(panel)
    assert not sections["overlay"].isVisibleTo(panel), "nothing to choose from without pose data"
