"""The pose-project mode in the GUI: two stages over one DeepLabCut folder (gui/pose_project_mode.py)."""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication

from ethograph.labels import pose_project as pp

VIDEO_A = "2024-02-05_33_cam-1"
VIDEO_B = "2024-02-06_34_cam-1"


@pytest.fixture
def pose_mode_gui(gui, pose_project_root):
    shell, meta = gui
    mode = meta.pose_project
    # The config names the scorer, so nothing is asked.
    assert mode.enter(pose_project_root, parent=None)
    QApplication.processEvents()
    return shell, meta, mode


def _menu_titles(shell) -> list[str]:
    return [a.text().replace("&", "").lstrip("▶ ") for a in shell.menuBar().actions()]


def _seek(meta, frame: int) -> None:
    meta.app_state.video.seek_to_frame(frame)
    meta.app_state.current_frame = frame
    QApplication.processEvents()


class TestExtractStage:
    def test_videos_are_trials_and_the_bar_is_reduced(self, pose_mode_gui):
        shell, meta, mode = pose_mode_gui
        assert mode.stage == pp.STAGE_EXTRACT
        assert [str(t) for t in meta.app_state.trials] == [VIDEO_A, VIDEO_B]
        titles = _menu_titles(shell)
        assert titles == ["Extract Frames", "Refine Pose", "Docs", "Help"]
        assert type(meta.labels_widget._mode_widget).__name__ == "FrameExtractPanel"
        assert not meta.labels_widget._body.isVisible()
        assert meta.collapsible_widgets[1].isExpanded() or meta._active == 1

    def test_video_gets_half_the_window_unless_the_user_arranged_it(self, gui, pose_project_root, monkeypatch):
        shell, meta = gui
        shell.resize(800, 600)
        calls = []
        monkeypatch.setattr(shell, "resizeDocks", lambda docks, sizes, orientation: calls.append((docks, sizes)))
        assert meta.pose_project.enter(pose_project_root, parent=None)
        QApplication.processEvents()
        assert calls and calls[-1] == ([shell._video_dock], [300])

        calls.clear()
        meta.app_state.panel_layout = {"panels": [], "shell_dock_state_b64": "AAAA"}
        meta.pose_project._split_video_and_plots()
        QApplication.processEvents()
        assert calls == []

    def test_curves_come_from_the_predictions(self, pose_mode_gui):
        _shell, meta, _mode = pose_mode_gui
        features = set(meta.app_state.catalog.feature_choices()) if hasattr(meta.app_state, "catalog") else set()
        ds = meta.app_state.ds
        assert {"position", "confidence", "velocity", "speed"} <= set(ds.data_vars) | features

    def test_untracked_video_shows_empty_panels(self, pose_mode_gui):
        import numpy as np

        _shell, meta, _mode = pose_mode_gui
        meta.navigation_widget.navigate_to_trial(VIDEO_B)
        QApplication.processEvents()
        assert meta.app_state.trials_sel == VIDEO_B
        assert meta.app_state.ds.attrs["trial"] == VIDEO_B
        assert np.isnan(meta.app_state.ds["speed"].values).all()
        assert meta.app_state.video_path.endswith(f"{VIDEO_B}.mp4")

    def test_marking_and_extracting(self, pose_mode_gui):
        _shell, meta, mode = pose_mode_gui
        labels = meta.labels_widget
        _seek(meta, 5)
        labels.activate_label(2)  # the "2" key: a point event lands on the frame at once
        assert mode.marked_frames() == ([], [5])
        _seek(meta, 2)
        labels.place_label_now(1)
        _seek(meta, 9)
        labels.place_label_now(1)
        assert mode.marked_frames() == ([(2, 9)], [5])

        meta.app_state.pose_extract_method = pp.METHOD_UNIFORM
        meta.app_state.pose_extract_coverage = 50.0
        mode.extract_current_video()
        folder = mode.project.labeled_dir / VIDEO_A
        table = mode.project.labels_table(VIDEO_A)
        frames = table.read()
        assert {pp.frame_index_of(n) for n in frames} >= {5}
        assert len(frames) == 1 + 4  # the point + half of the 8-frame segment
        assert (folder / "CollectedData_alice.csv").is_file()
        # The refine stage's alignment already lists the new folder as a trial.
        from ethograph.io.nwb_alignment import make_nwb_alignment
        from ethograph.io.session_layout import alignment_path

        refine = make_nwb_alignment(alignment_path(mode.project.session_dir(pp.STAGE_REFINE)))
        assert [str(t) for t in refine.trials_df["trial"]] == [VIDEO_A]
        refine.close()


class TestRefineStage:
    @pytest.fixture
    def refined(self, pose_mode_gui):
        shell, meta, mode = pose_mode_gui
        video = mode.project.videos()[0]
        import numpy as np

        predictions = pp.predictions_by_frame(pp.tracking_dataset(video, 10), mode.project.keypoints, [])
        pp.extract_frames(mode.project, video, [1, 4, 7], lambda f: np.full((32, 32, 3), f, np.uint8), 20, predictions)
        assert mode.switch(pp.STAGE_REFINE, parent=None)
        QApplication.processEvents()
        return shell, meta, mode

    def test_folder_is_the_trial_and_its_images_the_video(self, refined):
        shell, meta, mode = refined
        assert [str(t) for t in meta.app_state.trials] == [VIDEO_A]
        assert type(shell.video_area.primary.plot).__name__ == "ImageSequencePlot"
        assert meta.app_state.num_frames == 3
        assert _menu_titles(shell)[:2] == ["Extract Frames", "Refine Pose"]

    def test_frame_folder_passes_the_media_check(self, refined):
        """A saved video_folder pointing at labeled-data/ must not cancel the load: the trial's
        video is a folder, and 'isfile' on it was what refused every refine session."""
        _shell, meta, mode = refined
        missing = meta.data_widget._validate_media_files(
            nwb_alignment=meta.app_state.nwb_alignment,
            first_trial=VIDEO_A,
            video_folder=str(mode.project.labeled_dir),
        )
        assert missing == []

    def test_table_rows_are_draggable_labels(self, refined):
        _shell, meta, mode = refined
        dialog = mode.refine_dialog
        assert dialog is meta.labels_widget._mode_widget
        assert dialog.store.anchor_frames() == [0, 1, 2]
        assert dialog.interaction_mode == "sequential"
        # Unwrapped from the tab widget, the page must not read as "another tab": clicks label.
        assert dialog._lock_wanted() is False
        assert dialog.static_group._dialog is dialog
        assert dialog.static_group.parentWidget() is not None  # inside the Refine Pose panel

    def test_edit_is_written_back_to_the_table(self, refined):
        _shell, meta, mode = refined
        dialog = mode.refine_dialog
        dialog.store.set_point(1, "nose", (3.0, 4.0))
        dialog._on_store_changed()
        dialog.save_now()
        frames = mode.project.labels_table(VIDEO_A).read()
        assert frames["img04.png"][0, 0].tolist() == [3.0, 4.0]

    def test_static_landmark_propagates_to_every_frame(self, refined):
        _shell, meta, mode = refined
        dialog = mode.refine_dialog
        _seek(meta, 0)
        dialog.store.set_point(0, "tail", (11.0, 12.0))
        dialog.store.set_static("tail", True)
        assert dialog.propagate_static_from_current_frame() == 1
        dialog.save_now()
        frames = mode.project.labels_table(VIDEO_A).read()
        for image in ("img01.png", "img04.png", "img07.png"):
            assert frames[image][0, 1].tolist() == [11.0, 12.0]

    def test_check_labels_button_renders_the_folder(self, refined):
        _shell, meta, mode = refined
        dialog = mode.refine_dialog
        dialog.store.set_point(0, "nose", (5.0, 6.0))
        dialog._on_store_changed()
        output = dialog.check_labels()
        assert output is not None and output.name == f"{VIDEO_A}_labeled"
        assert sorted(p.name for p in output.iterdir()) == ["img01.png", "img04.png", "img07.png"]
        assert not dialog._dirty  # the table was written first

    def test_folder_starts_uncurated_and_ctrl_c_curates_it(self, refined):
        _shell, meta, mode = refined
        assert meta.app_state.trial_is_curated(VIDEO_A) is False
        meta.curate_current_trial()
        assert meta.app_state.trial_is_curated(VIDEO_A) is True
        meta.trials_widget.flush_metadata()
        assert pp.read_curated(mode.project.session_dir(pp.STAGE_REFINE)) == {VIDEO_A: True}

    def test_switching_back_keeps_both_stages(self, refined):
        shell, meta, mode = refined
        assert mode.switch(pp.STAGE_EXTRACT, parent=None)
        assert [str(t) for t in meta.app_state.trials] == [VIDEO_A, VIDEO_B]
        assert mode.refine_dialog is None
        assert type(shell.video_area.primary.plot).__name__ == "PlotVideo"

    def test_a_second_load_does_not_duplicate_the_sidebar_combos(self, refined):
        """Every stage switch is a session load; the Data tab must still show each dim once."""
        from qtpy.QtWidgets import QComboBox

        _shell, meta, mode = refined
        assert mode.switch(pp.STAGE_EXTRACT, parent=None)
        QApplication.processEvents()
        dw = meta.data_widget
        names = [c.objectName() for c in dw.coords_groupbox.findChildren(QComboBox) if not c.isHidden()]
        assert names.count("keypoint_combo") == 1 and names.count("space_combo") == 1
        individuals = [
            c for c in dw.individual_groupbox.findChildren(QComboBox) if c.objectName() == "individual_combo"
        ]
        assert len(individuals) == 1
