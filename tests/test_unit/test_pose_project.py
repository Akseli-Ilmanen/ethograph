"""The DeepLabCut / LightningPose project folder as two ethograph sessions (labels/pose_project.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ethograph.io.data_loader import load_features_dataset
from ethograph.io.image_sequence import IMAGE_SEQUENCE_RATE, ImageSequence, image_files, is_image_folder
from ethograph.io.video_probe import probe_video
from ethograph.labels import pose_project as pp
from tests.conftest import POSE_FPS as FPS
from tests.conftest import POSE_KEYPOINTS as KEYPOINTS
from tests.conftest import POSE_N_FRAMES as N_FRAMES


class TestOpen:
    def test_refuses_folder_without_videos_or_labeled_data(self, tmp_path):
        (tmp_path / "videos").mkdir()
        with pytest.raises(pp.PoseProjectError, match="labeled-data"):
            pp.PoseProject.open(tmp_path)

    def test_reads_schema_from_config(self, pose_project_root):
        project = pp.PoseProject.open(pose_project_root)
        assert project.layout == pp.LAYOUT_DEEPLABCUT
        assert project.scorer == "alice"
        assert project.keypoints == KEYPOINTS
        assert not project.multi_animal

    def test_schema_falls_back_to_predictions_and_scorer_must_be_named(self, pose_project_root):
        (pose_project_root / pp.CONFIG_FILE).unlink()
        assert pp.PoseProject.scorer_of(pose_project_root) is None
        with pytest.raises(pp.PoseProjectError, match="scorer"):
            pp.PoseProject.open(pose_project_root)
        project = pp.PoseProject.open(pose_project_root, scorer="bob")
        assert project.keypoints == KEYPOINTS
        assert (project.scorer, project.scorer_from_config) == ("bob", False)

    def test_config_scorer_wins_over_the_name_given(self, pose_project_root):
        project = pp.PoseProject.open(pose_project_root, scorer="bob")
        assert (project.scorer, project.scorer_from_config) == ("alice", True)
        assert pp.PoseProject.scorer_of(pose_project_root) == "alice"

    def test_videos_pair_with_their_predictions_by_name(self, pose_project_root):
        videos = pp.PoseProject.open(pose_project_root).videos()
        assert [v.stem for v in videos] == ["2024-02-05_33_cam-1", "2024-02-06_34_cam-1"]
        assert videos[0].tracking is not None and videos[0].software == "DeepLabCut"
        assert videos[1].tracking is None


class TestExtractSession:
    def test_one_trial_per_video_with_curves(self, pose_project_root):
        project = pp.PoseProject.open(pose_project_root)
        outcome = pp.build_extract_session(project)
        assert outcome.trials == ["2024-02-05_33_cam-1", "2024-02-06_34_cam-1"]
        assert outcome.untracked == ["2024-02-06_34_cam-1"]
        assert (outcome.session_dir / ".ethograph" / "mapping.txt").read_text() == pp.MAPPING_TEXT

        result = load_features_dataset(str(outcome.session_dir))
        assert [str(t) for t in result.trial_ids] == outcome.trials
        ds = result.dt.trial("2024-02-05_33_cam-1")
        assert {"position", "confidence", "velocity", "speed"} <= set(ds.data_vars)
        # The untracked video is a trial too, with nothing in it: empty panels, not the last trial's.
        blank = result.dt.trial("2024-02-06_34_cam-1")
        assert set(blank.data_vars) == set(ds.data_vars)
        assert blank.sizes["time"] == N_FRAMES and np.isnan(blank["speed"].values).all()
        video = result.nwb_alignment.resolve_media_path("2024-02-05_33_cam-1", "video", device=pp.CAMERA)
        assert Path(video).name == "2024-02-05_33_cam-1.mp4"

    def test_opens_with_position_speed_and_confidence_panels(self, pose_project_root):
        import yaml

        project = pp.PoseProject.open(pose_project_root)
        session_dir = pp.build_extract_session(project).session_dir
        settings = session_dir / ".ethograph" / "local_settings.yaml"
        layout = yaml.safe_load(settings.read_text())["panel_layout"]
        assert [(p["type"], p["feature"]) for p in layout["panels"]] == [
            ("lineplot", "position"),
            ("lineplot", "speed"),
            ("heatmap", "confidence"),
        ]
        for panel in layout["panels"][:2]:
            assert panel["selections"]["keypoint"] == "nose"
            assert set(panel["selections"]) == {"individual", "keypoint"}  # space stays free
        assert "keypoint" not in layout["panels"][2]["selections"]
        # The user's layout, once saved there, is never overwritten.
        settings.write_text("panel_layout: {panels: []}\n")
        pp.build_extract_session(project)
        assert yaml.safe_load(settings.read_text()) == {"panel_layout": {"panels": []}}

    def test_a_dataset_missing_a_video_is_rebuilt(self, pose_project_root):
        """A session.nc from before untracked videos were trials is stale even with the same sources."""
        project = pp.PoseProject.open(pose_project_root)
        session_dir = pp.build_extract_session(project).session_dir
        nc_path = session_dir / "session.nc"
        tracked = pp.tracking_dataset(project.videos()[0], FPS)
        tracked.attrs["trial"] = "2024-02-05_33_cam-1"
        from ethograph.io.netcdf import netcdf_engine
        from ethograph.io.trialtree import TrialTree

        nc_path.unlink()
        TrialTree.from_datasets([tracked]).to_netcdf(nc_path, engine=netcdf_engine(nc_path))
        pp.build_extract_session(project)
        result = load_features_dataset(str(session_dir))
        assert "2024-02-06_34_cam-1" in [str(t) for t in result.dt.trials]

    def test_rebuilding_is_idempotent(self, pose_project_root):
        project = pp.PoseProject.open(pose_project_root)
        first = pp.build_extract_session(project)
        second = pp.build_extract_session(project)
        assert first.trials == second.trials


class TestPlanFrames:
    def test_points_all_taken_and_segments_covered(self):
        frames = pp.plan_frames(100, segments=[(10, 19)], points=[3, 50], method=pp.METHOD_UNIFORM, coverage=0.5)
        assert {3, 50} <= set(frames)
        segment = [f for f in frames if 10 <= f <= 19]
        assert len(segment) == 5

    def test_excluded_frames_never_picked_again(self):
        frames = pp.plan_frames(
            30, segments=[(0, 9)], points=[5], method=pp.METHOD_UNIFORM, coverage=1.0, exclude={5, 6}
        )
        assert 5 not in frames and 6 not in frames
        assert len(frames) == 8

    def test_diverse_needs_frames(self):
        with pytest.raises(ValueError, match="frames"):
            pp.plan_frames(30, segments=[(0, 9)], points=[], method=pp.METHOD_DIVERSE, coverage=0.5)


class TestExtractFrames:
    def test_writes_images_and_prefilled_table(self, pose_project_root):
        project = pp.PoseProject.open(pose_project_root)
        video = project.videos()[0]
        ds = pp.tracking_dataset(video, FPS)
        predictions = pp.predictions_by_frame(ds, project.keypoints, project.individuals)
        outcome = pp.extract_frames(
            project,
            video,
            [2, 7],
            decode=lambda f: np.zeros((32, 32, 3), np.uint8),
            n_video_frames=N_FRAMES,
            predictions=predictions,
        )
        assert outcome.folder == pose_project_root / pp.LABELED_DATA_DIR / video.stem
        assert [p.name for p in outcome.images] == ["img02.png", "img07.png"]
        assert outcome.table.name == "CollectedData_alice.csv"
        assert outcome.table.with_suffix(".h5").is_file()

        table = project.labels_table(video.stem)
        frames = table.read()
        assert set(frames) == {"img02.png", "img07.png"}
        np.testing.assert_allclose(frames["img07.png"][0, 1], (7, 10))

    def test_extracting_again_keeps_edited_rows(self, pose_project_root):
        project = pp.PoseProject.open(pose_project_root)
        video = project.videos()[0]
        decode = lambda f: np.zeros((32, 32, 3), np.uint8)  # noqa: E731
        pp.extract_frames(project, video, [2], decode, N_FRAMES, predictions=None)
        table = project.labels_table(video.stem)
        edited = table.read()
        edited["img02.png"][0, 0] = (1.5, 2.5)
        table.write(edited)
        pp.extract_frames(project, video, [2, 9], decode, N_FRAMES, predictions=None)
        frames = table.read()
        np.testing.assert_allclose(frames["img02.png"][0, 0], (1.5, 2.5))
        assert np.isnan(frames["img09.png"]).all()

    def test_table_survives_a_dlc_round_trip(self, pose_project_root):
        """What we write is what DeepLabCut's own reader (pandas h5) gives back."""
        project = pp.PoseProject.open(pose_project_root)
        video = project.videos()[0]
        pp.extract_frames(project, video, [4], lambda f: np.zeros((32, 32, 3), np.uint8), N_FRAMES)
        path = project.labels_table(video.stem).path
        df = pd.read_hdf(path.with_suffix(".h5"), key=pp.DLC_H5_KEY)
        assert list(df.columns.names) == ["scorer", "bodyparts", "coords"]
        assert df.index[0] == ("labeled-data", video.stem, "img04.png")


class TestRefineSession:
    def test_each_folder_is_a_trial_whose_video_is_the_folder(self, pose_project_root):
        project = pp.PoseProject.open(pose_project_root)
        video = project.videos()[0]
        pp.extract_frames(project, video, [1, 5, 9], lambda f: np.full((32, 32, 3), f, np.uint8), N_FRAMES)
        session_dir = pp.build_refine_session(project)

        result = load_features_dataset(str(session_dir))
        assert [str(t) for t in result.trial_ids] == [video.stem]
        folder = result.nwb_alignment.resolve_media_path(video.stem, "video", device=pp.CAMERA)
        assert is_image_folder(folder)
        probe = probe_video(folder)
        assert (probe.fps, probe.nframes, probe.width) == (IMAGE_SEQUENCE_RATE, 3, 32)
        assert pp.read_curated(session_dir) == {video.stem: False}

    def test_curated_verdict_survives_a_rebuild(self, pose_project_root):
        project = pp.PoseProject.open(pose_project_root)
        video = project.videos()[0]
        pp.extract_frames(project, video, [1], lambda f: np.zeros((32, 32, 3), np.uint8), N_FRAMES)
        session_dir = pp.build_refine_session(project)
        pp.write_curated(session_dir, [video.stem], {video.stem: True})
        pp.build_refine_session(project)
        assert pp.read_curated(session_dir) == {video.stem: True}

    def test_refuses_without_any_labeled_folder(self, pose_project_root):
        with pytest.raises(pp.PoseProjectError, match="extract frames first"):
            pp.build_refine_session(pp.PoseProject.open(pose_project_root))


class TestImageSequence:
    def test_natural_order_and_lazy_frames(self, tmp_path):
        import imageio.v3 as iio

        for name, value in (("img10.png", 10), ("img2.png", 2), ("notes.txt", 0)):
            if name.endswith(".png"):
                iio.imwrite(tmp_path / name, np.full((4, 6, 3), value, np.uint8))
            else:
                (tmp_path / name).write_text("x")
        seq = ImageSequence(tmp_path)
        assert [p.name for p in image_files(tmp_path)] == ["img2.png", "img10.png"]
        assert seq.size == (6, 4)
        assert seq[1][0, 0, 0] == 10
        assert seq.index_of("img10.png") == 1 and seq.name_of(0) == "img2.png"


class TestLightningPoseLayout:
    def test_root_table_holds_every_video(self, tmp_path):
        root = tmp_path / "lp"
        (root / pp.VIDEOS_DIR).mkdir(parents=True)
        (root / pp.LABELED_DATA_DIR).mkdir()
        header = "scorer,bob,bob,bob,bob\nbodyparts,nose,nose,tail,tail\ncoords,x,y,x,y\n"
        (root / pp.LP_COLLECTED_DATA).write_text(header + "labeled-data/vidA/img0.png,1,2,3,4\n", encoding="utf-8")
        project = pp.PoseProject.open(root)
        assert project.layout == pp.LAYOUT_LIGHTNINGPOSE
        assert project.scorer == "bob" and project.keypoints == ["nose", "tail"]

        table = project.labels_table("vidB")
        table.write({"img5.png": np.array([[[9.0, 8.0], [7.0, 6.0]]])})
        assert project.labels_table("vidA").read()["img0.png"][0, 1].tolist() == [3.0, 4.0]
        again = pd.read_csv(root / pp.LP_COLLECTED_DATA, header=[0, 1, 2], index_col=0)
        assert list(again.index) == ["labeled-data/vidA/img0.png", "labeled-data/vidB/img5.png"]


class TestCheckLabels:
    def test_draws_every_frame_into_a_labeled_folder_that_is_never_a_trial(self, pose_project_root):
        from ethograph.labels.check_labels import check_labels, labeled_output_dir

        project = pp.PoseProject.open(pose_project_root)
        video = project.videos()[0]
        predictions = pp.predictions_by_frame(pp.tracking_dataset(video, FPS), project.keypoints, [])
        pp.extract_frames(project, video, [3, 8], lambda f: np.zeros((32, 32, 3), np.uint8), N_FRAMES, predictions)
        folder = project.labeled_dir / video.stem
        # One frame nobody labelled: written untouched, still part of the folder.
        import imageio.v3 as iio

        iio.imwrite(folder / "img15.png", np.zeros((32, 32, 3), np.uint8))

        output = check_labels(project, video.stem)
        assert output == labeled_output_dir(folder) and output.name == f"{video.stem}_labeled"
        assert sorted(p.name for p in output.iterdir()) == ["img03.png", "img08.png", "img15.png"]
        drawn = iio.imread(output / "img08.png")
        assert drawn.max() > 0  # a dot landed
        assert iio.imread(output / "img15.png").max() == 0
        assert [f.name for f in project.labeled_folders()] == [video.stem]
