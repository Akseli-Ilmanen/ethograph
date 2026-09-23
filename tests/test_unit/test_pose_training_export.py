"""Training-label export: which frames, which points, and additive merging.

Qt-free — the engine that decides what lands in a DeepLabCut or COCO folder.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ethograph.gui.pose_annotate import KeypointStore
from ethograph.gui.pose_training_export import (
    coco_annotation,
    coco_category,
    dlc_rows,
    dlc_video_folder,
    export_coco,
    export_dlc,
    frame_digits,
    merge_coco,
    merge_collected_data,
    read_collected_data,
    training_frames,
)


def _store(n_frames: int = 6, individuals: list[str] | None = None) -> KeypointStore:
    return KeypointStore(keypoint_names=["beak", "tail"], n_frames=n_frames, individual_names=individuals or ["bird"])


def _decode(video_frame: int) -> np.ndarray:
    """A 4x6 frame whose red channel is the frame index — tells frames apart."""
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = video_frame
    return rgb


# ----------------------------------------------------------------------
# Which frames, which points
# ----------------------------------------------------------------------


def test_training_frames_are_the_clicked_ones_with_file_points_and_never_fill():
    store = _store()
    # The file saw both points on frames 0 and 2; the user corrected the beak on 2.
    store.set_detections(
        {0: np.array([[[1.0, 1.0], [2.0, 2.0]]]), 2: np.array([[[5.0, 5.0], [6.0, 6.0]]])},
        {0: np.array([[0.9, 0.9]]), 2: np.array([[0.9, 0.9]])},
    )
    store.set_point(2, "beak", (50.0, 50.0))
    store.set_point(4, "tail", (70.0, 70.0))
    filled = np.full((6, 1, 2, 2), 99.0)
    store.set_fill(filled, np.ones((6, 1, 2)))

    frames = training_frames(store, window_start=100)
    assert [f.video_frame for f in frames] == [102, 104]
    np.testing.assert_allclose(frames[0].positions[0, 0], [50.0, 50.0])  # the click wins
    np.testing.assert_allclose(frames[0].positions[0, 1], [6.0, 6.0])  # the file's own point stays
    assert np.isnan(frames[1].positions[0, 0]).all()  # the fill is not a label
    np.testing.assert_allclose(frames[1].positions[0, 1], [70.0, 70.0])


def test_frame_digits_follow_the_folder_then_the_video():
    assert frame_digits(["img0042.png"], 10) == 4
    assert frame_digits([], 10) == 1
    assert frame_digits([], 100_000) == 5
    assert frame_digits(["CollectedData_x.csv"], 1000) == 3


# ----------------------------------------------------------------------
# DeepLabCut
# ----------------------------------------------------------------------


def test_dlc_video_folder_lands_under_a_projects_labeled_data(tmp_path):
    assert dlc_video_folder(tmp_path, "cam1") == tmp_path / "cam1"
    (tmp_path / "config.yaml").write_text("scorer: me\n")
    assert dlc_video_folder(tmp_path, "cam1") == tmp_path / "labeled-data" / "cam1"


def test_single_animal_table_has_three_header_levels_and_multi_four():
    store = _store()
    frames = training_frames(store)  # empty is fine for the header shape
    single = dlc_rows(frames, ["beak"], ["bird"], "me", "cam1", 3, multi_animal=False)
    assert single.columns.names == ["scorer", "bodyparts", "coords"]
    multi = dlc_rows(frames, ["beak"], ["a", "b"], "me", "cam1", 3, multi_animal=True)
    assert list(multi.columns) == [
        ("me", "a", "beak", "x"),
        ("me", "a", "beak", "y"),
        ("me", "b", "beak", "x"),
        ("me", "b", "beak", "y"),
    ]
    with pytest.raises(ValueError, match="multi-animal"):
        dlc_rows(frames, ["beak"], ["a", "b"], "me", "cam1", 3, multi_animal=False)


def test_merge_keeps_existing_frames_and_replaces_the_same_one():
    store = _store()
    store.set_point(1, "beak", (1.0, 1.0))
    store.set_point(3, "beak", (3.0, 3.0))
    first = dlc_rows(training_frames(store), ["beak", "tail"], ["bird"], "me", "cam1", 2, False)

    store.anchors.clear()
    store.set_point(3, "beak", (30.0, 30.0))
    store.set_point(5, "tail", (5.0, 5.0))
    second = dlc_rows(training_frames(store), ["beak", "tail"], ["bird"], "me", "cam1", 2, False)

    merged = merge_collected_data(first, second)
    assert [row[2] for row in merged.index] == ["img01.png", "img03.png", "img05.png"]
    assert merged.loc[("labeled-data", "cam1", "img03.png"), ("me", "beak", "x")] == 30.0
    assert merged.loc[("labeled-data", "cam1", "img01.png"), ("me", "beak", "x")] == 1.0


def test_merge_refuses_another_scorer_or_project_type():
    store = _store()
    store.set_point(1, "beak", (1.0, 1.0))
    frames = training_frames(store)
    keypoints = ["beak", "tail"]
    mine = dlc_rows(frames, keypoints, ["bird"], "me", "cam1", 2, False)
    with pytest.raises(ValueError, match="scored by"):
        merge_collected_data(mine, dlc_rows(frames, keypoints, ["bird"], "you", "cam1", 2, False))
    with pytest.raises(ValueError, match="single vs multi"):
        merge_collected_data(mine, dlc_rows(frames, keypoints, ["bird"], "me", "cam1", 2, True))


def test_export_dlc_twice_grows_the_collected_data(tmp_path):
    (tmp_path / "config.yaml").write_text("scorer: me\nmultianimalproject: false\n")
    store = _store(n_frames=10)
    store.set_point(2, "beak", (1.0, 2.0))
    outcome = export_dlc(training_frames(store), store, _decode, tmp_path, "cam1", "me", n_video_frames=1000)
    folder = tmp_path / "labeled-data" / "cam1"
    assert outcome.n_frames == 1
    assert (folder / "img002.png").exists()
    assert (folder / "CollectedData_me.csv").exists()

    store.anchors.clear()
    store.set_point(7, "tail", (3.0, 4.0))
    export_dlc(training_frames(store), store, _decode, tmp_path, "cam1", "me", n_video_frames=1000)

    table = read_collected_data(folder / "CollectedData_me.csv")
    assert [row[2] for row in table.index] == ["img002.png", "img007.png"]
    assert table.loc[("labeled-data", "cam1", "img002.png"), ("me", "beak", "x")] == 1.0
    assert np.isnan(table.loc[("labeled-data", "cam1", "img007.png"), ("me", "beak", "x")])
    assert table.loc[("labeled-data", "cam1", "img007.png"), ("me", "tail", "y")] == 4.0
    # The csv is what DeepLabCut's own reader sees: the header depth round-trips.
    raw = pd.read_csv(folder / "CollectedData_me.csv", header=[0, 1, 2], index_col=[0, 1, 2])
    assert raw.shape == (2, 4)


def test_export_dlc_refuses_a_folder_scored_by_someone_else(tmp_path):
    store = _store()
    store.set_point(1, "beak", (1.0, 2.0))
    export_dlc(training_frames(store), store, _decode, tmp_path, "cam1", "me", n_video_frames=10)
    with pytest.raises(ValueError, match="scorer 'me'"):
        export_dlc(training_frames(store), store, _decode, tmp_path, "cam1", "you", n_video_frames=10)


# ----------------------------------------------------------------------
# COCO
# ----------------------------------------------------------------------


def test_coco_annotation_flags_visibility_and_boxes_the_visible_points():
    body = coco_annotation(np.array([[10.0, 20.0], [np.nan, np.nan], [30.0, 60.0]]))
    assert body is not None
    assert body["keypoints"] == [10.0, 20.0, 2, 0.0, 0.0, 0, 30.0, 60.0, 2]
    assert body["num_keypoints"] == 2
    assert body["bbox"] == [10.0, 20.0, 20.0, 40.0]
    assert coco_annotation(np.full((2, 2), np.nan)) is None


def test_merge_coco_replaces_the_same_image_and_continues_ids():
    category = coco_category(None, ["beak"])
    first = merge_coco(None, [{"file_name": "a.png", "width": 6, "height": 4}], [[{"keypoints": [1, 1, 2]}]], category)
    assert [img["id"] for img in first["images"]] == [1]
    second = merge_coco(
        first,
        [{"file_name": "a.png", "width": 6, "height": 4}, {"file_name": "b.png", "width": 6, "height": 4}],
        [[{"keypoints": [9, 9, 2]}], [{"keypoints": [2, 2, 2]}, {"keypoints": [3, 3, 2]}]],
        category,
    )
    assert [img["file_name"] for img in second["images"]] == ["a.png", "b.png"]
    assert len(second["annotations"]) == 3
    assert len({a["id"] for a in second["annotations"]}) == 3
    replaced = next(a for a in second["annotations"] if a["image_id"] == second["images"][0]["id"])
    assert replaced["keypoints"] == [9, 9, 2]


def test_coco_category_must_agree_with_the_existing_file():
    existing = {"categories": [{"id": 1, "keypoints": ["beak"]}]}
    assert coco_category(existing, ["beak"])["id"] == 1
    with pytest.raises(ValueError, match="already describes"):
        coco_category(existing, ["beak", "tail"])


def test_export_coco_writes_images_and_one_annotation_per_individual(tmp_path):
    store = _store(individuals=["a", "b"])
    store.set_point(1, "beak", (1.0, 2.0), individual="a")
    store.set_point(1, "tail", (3.0, 4.0), individual="b")
    export_coco(training_frames(store), store, _decode, tmp_path, "cam1", n_video_frames=10)
    payload = json.loads((tmp_path / "annotations.json").read_text())
    assert (tmp_path / "images" / "cam1_img1.png").exists()
    assert payload["images"][0]["width"] == 6 and payload["images"][0]["height"] == 4
    assert len(payload["annotations"]) == 2
    assert payload["categories"][0]["keypoints"] == ["beak", "tail"]
