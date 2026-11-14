"""A LightningPose multiview project: every view's frames and rows extracted as pairs (labels/pose_project.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from ethograph.gui.pose_fill import VideoFrameSource
from ethograph.labels import pose_project as pp
from tests.conftest import POSE_FPS as FPS
from tests.conftest import POSE_KEYPOINTS as KEYPOINTS
from tests.conftest import POSE_N_FRAMES as N_FRAMES
from tests.conftest import write_dlc_predictions, write_test_video

SESSION = "2024-02-06_125"
VIEWS = ["cam-1", "cam-2"]
#: Each view's frame size: an image written from the wrong view has the wrong shape.
SIZES = {"cam-1": 32, "cam-2": 48}


@pytest.fixture
def multiview_root(tmp_path):
    root = tmp_path / "lp_multiview"
    (root / pp.VIDEOS_DIR).mkdir(parents=True)
    (root / pp.LABELED_DATA_DIR).mkdir()
    (root / pp.LP_PROJECT_FILE).write_text(
        yaml.safe_dump({"view_names": VIEWS, "keypoint_names": KEYPOINTS, "schema_version": 0}), encoding="utf-8"
    )
    for view in VIEWS:
        # Frame i of either video is a uniform gray of i * 10: the image says which frame it is.
        write_test_video(root / pp.VIDEOS_DIR / f"{SESSION}_{view}.mp4", size=SIZES[view])
    write_dlc_predictions(root / pp.VIDEOS_DIR / f"{SESSION}_cam-1DLC_resnet50_testMar1shuffle1_500.h5")
    return root


def _extract(project: pp.PoseProject, stem: str, frames: list[int]) -> pp.MultiviewExtractOutcome:
    videos = project.view_videos(stem)
    decoders = {view: VideoFrameSource(v.video, FPS, N_FRAMES).__getitem__ for view, v in videos.items()}
    predictions = {
        view: pp.predictions_by_frame(pp.tracking_dataset(v, FPS), project.keypoints, []) if v.tracking else None
        for view, v in videos.items()
    }
    return pp.extract_multiview_frames(project, videos, frames, decoders, N_FRAMES, predictions)


def _row_paths(path) -> list[str]:
    return [str(line.split(",")[0]) for line in path.read_text(encoding="utf-8").splitlines()[3:]]


def test_extracting_from_one_view_writes_every_view_in_the_same_row_order(multiview_root):
    project = pp.PoseProject.open(multiview_root, scorer="bob")
    assert project.views == VIEWS and project.layout == pp.LAYOUT_LIGHTNINGPOSE

    outcome = _extract(project, f"{SESSION}_cam-2", [12, 3, 7])
    assert outcome.frames == [12, 3, 7]
    rows = {view: _row_paths(project.view_table_path(view)) for view in VIEWS}
    assert rows["cam-1"] == [f"labeled-data/{SESSION}_cam-1/img{f:04d}.png" for f in (12, 3, 7)]
    assert [r.replace("_cam-1/", "_cam-2/") for r in rows["cam-1"]] == rows["cam-2"]

    for view in VIEWS:
        source = VideoFrameSource(project.videos_dir / f"{SESSION}_{view}.mp4", FPS, N_FRAMES)
        image = cv2.cvtColor(cv2.imread(str(outcome.folders[view] / "img0007.png")), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(image, source[7])
        assert image.shape[:2] == (SIZES[view], SIZES[view])
        assert abs(float(image.mean()) - 70) < 3

    # cam-1 has predictions (keypoint k of frame t at (t, 10 k)); cam-2 has none, so its row is empty.
    cam1 = project.labels_table(f"{SESSION}_cam-1").read()
    np.testing.assert_allclose(cam1["img0007.png"][0, 1], (7, 10))
    assert np.isnan(project.labels_table(f"{SESSION}_cam-2").read()["img0007.png"]).all()


def test_re_extracting_leaves_existing_rows_byte_for_byte(multiview_root):
    project = pp.PoseProject.open(multiview_root, scorer="bob")
    _extract(project, f"{SESSION}_cam-1", [3, 7])
    before = {view: project.view_table_path(view).read_bytes() for view in VIEWS}

    assert _extract(project, f"{SESSION}_cam-2", [7, 3]).frames == []
    assert {view: project.view_table_path(view).read_bytes() for view in VIEWS} == before

    _extract(project, f"{SESSION}_cam-2", [3, 15])
    for view in VIEWS:
        after = project.view_table_path(view).read_bytes()
        assert after.startswith(before[view])
        assert _row_paths(project.view_table_path(view))[-1].endswith("img0015.png")


def test_refine_edits_keep_every_row_in_place(multiview_root):
    """A save in Refine Pose rewrites one view's table; its rows stay paired with the other views'."""
    project = pp.PoseProject.open(multiview_root, scorer="bob")
    _extract(project, f"{SESSION}_cam-1", [12, 3])
    table = project.labels_table(f"{SESSION}_cam-2")
    frames = table.read()
    frames["img0012.png"][0, 0] = (1.5, 2.5)
    table.write(frames)
    assert _row_paths(project.view_table_path("cam-2"))[0].endswith("img0012.png")
    np.testing.assert_allclose(table.read()["img0012.png"][0, 0], (1.5, 2.5))


def test_a_table_whose_first_row_is_empty_reads_every_row(tmp_path):
    path = tmp_path / "CollectedData_cam-1.csv"
    path.write_text(
        "scorer,bob,bob\nbodyparts,nose,nose\ncoords,x,y\nlabeled-data/s_cam-1/img0001.png,,\n"
        "labeled-data/s_cam-1/img0002.png,4.0,5.0\n",
        encoding="utf-8",
    )
    df = pp.read_labels_table(path)
    assert [row[2] for row in df.index] == ["img0001.png", "img0002.png"]
    assert np.isnan(df.iloc[0]).all() and df.iloc[1].tolist() == [4.0, 5.0]


def test_views_come_from_table_names_without_project_yaml(multiview_root):
    project = pp.PoseProject.open(multiview_root, scorer="bob")
    _extract(project, f"{SESSION}_cam-1", [2])
    (multiview_root / pp.LP_PROJECT_FILE).unlink()
    again = pp.PoseProject.open(multiview_root, scorer="bob")
    assert again.views == VIEWS and again.keypoints == KEYPOINTS


def test_lightning_pose_reads_the_paired_tables(multiview_root):
    """LightningPose's own multiview reader takes the tables as they are written."""
    pytest.importorskip("lightning_pose")
    project = pp.PoseProject.open(multiview_root, scorer="bob")
    _extract(project, f"{SESSION}_cam-1", [3, 7, 12])
    tables = [pd.read_csv(project.view_table_path(view), header=[0, 1, 2], index_col=0) for view in VIEWS]
    assert [t.shape for t in tables] == [(3, 2 * len(KEYPOINTS))] * 2
