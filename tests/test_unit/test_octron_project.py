"""The OCTRON project layer writes what OCTRON reads, and the config round-trips.

Two things must agree and nothing forces them to: the organizer JSON EthoGraph
writes and the schema OCTRON's own ``restore_object_organizer`` produces /
``collect_labels`` consumes; and the ``octron.yaml`` keys and the CLI flags
they become.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ethograph.labels.octron_project import (
    CONFIG_FILENAME,
    ObjectEntry,
    OctronConfig,
    OctronProject,
    VideoEntry,
    mask_bbox,
    mask_zarr_path,
    read_tracks,
    trackers,
    yolo_models,
)

pytest.importorskip("octron")


def _entry(tmp_path: Path) -> VideoEntry:
    folder = tmp_path / "octron" / "abcd1234"
    folder.mkdir(parents=True)
    video = tmp_path / "cam-top.mp4"
    video.write_bytes(b"not a real video")
    return VideoEntry(path=video, hash8="abcd1234", folder=folder, width=64, height=48, fps=25.0, num_frames=10)


class TestConfig:
    def test_round_trip_and_unknown_key(self, tmp_path: Path):
        cfg = OctronConfig(model="YOLO26l", imgsz=640, tracker="botsort", label_scheme="suffix")
        path = tmp_path / CONFIG_FILENAME
        cfg.save(path)
        assert OctronConfig.load(path) == cfg
        path.write_text(path.read_text() + "augment: {degrees: 15}\n")
        with pytest.raises(ValueError, match="unknown keys"):
            OctronConfig.load(path)

    def test_suggestion_knobs_are_bounded(self):
        with pytest.raises(ValueError):
            OctronConfig(suggest_motion_share=1.5)
        assert OctronConfig().suggest_motion_share == 0.3

    def test_split_must_leave_a_test_share(self):
        with pytest.raises(ValueError):
            OctronConfig(train_fraction=0.9, val_fraction=0.1)

    def test_label_scheme(self):
        assert OctronConfig().label_scheme == "suffix"  # animals of the same type, by default
        per = OctronConfig(label_scheme="per_individual")
        assert per.label_for("Female") == ("Female", "")
        suf = OctronConfig(label_scheme="suffix", class_name="bird")
        assert suf.label_for("Female") == ("bird", "female")
        with pytest.raises(ValueError):
            OctronConfig(label_scheme="classes")

    def test_shipped_default_loads(self):
        from ethograph.utils.paths import BUNDLED_DEFAULTS_DIR

        cfg = OctronConfig.load(BUNDLED_DEFAULTS_DIR / "config" / "octron.yaml")
        assert cfg.model == "YOLO26m"


class TestCommands:
    def test_train_and_predict_speak_the_octron_cli(self, tmp_path: Path):
        project = OctronProject(tmp_path / "octron")
        cfg = OctronConfig(model="yolo26m", imgsz=1024, epochs=3, tracker="bytetrack")
        train = project.train_command(cfg, overwrite=True)
        assert train[:3] == ["octron", "train", str(project.root)]
        assert "--mode" in train and train[train.index("--mode") + 1] == "detect"
        assert "--overwrite" in train and "--resume" not in train
        assert "degrees" not in " ".join(train)
        predict = project.predict_command(cfg, [tmp_path / "a.mp4", tmp_path / "b.mp4"])
        assert predict[:4] == ["octron", "predict", str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")]
        assert predict[predict.index("--model-path") + 1] == str(project.weights_path)
        assert predict[predict.index("--output-dir") + 1] == str(project.predictions_dir)
        assert project.prediction_folder(tmp_path / "a.mp4", "bytetrack") == (
            project.predictions_dir / "octron_predictions" / "a_bytetrack"
        )


class TestCatalogues:
    def test_choices_are_the_cli_s_own(self):
        """What the dialog offers must be a value OCTRON's CLI accepts."""
        from octron.cli import TrackerName, YOLOModel

        assert set(yolo_models()) == {m.value for m in YOLOModel}
        assert set(trackers()) == {t.value for t in TrackerName}


class TestOrganizer:
    def test_objects_get_octron_ids_colours_and_a_readable_json(self, tmp_path: Path):
        project = OctronProject(tmp_path / "octron")
        entry = _entry(tmp_path)

        female = project.ensure_object(entry, "female", "")
        male = project.ensure_object(entry, "male", "")
        again = project.ensure_object(entry, "female", "")

        assert again.obj_id == female.obj_id and again.label_id == female.label_id
        assert female.label_id != male.label_id
        assert len(female.color) == 4 and female.color != male.color
        assert mask_zarr_path(entry.folder, "female", "").exists()

        raw = json.loads(entry.organizer_path.read_text())
        e = raw["entries"][str(female.obj_id)]
        meta = e["prediction_layer_metadata"]
        assert e["label"] == "female" and e["label_id"] == female.label_id
        assert meta["name"] == "female masks" and meta["type"] == "Labels"
        assert meta["data_shape"] == [10, 48, 64]
        assert meta["zarr_path"] == "abcd1234/female masks.zarr"
        assert meta["video_hash"] == "abcd1234"
        assert not Path(meta["zarr_path"]).is_absolute()
        assert project.load_objects(entry) == [female, male]

    def test_suffix_objects_share_a_label_id(self, tmp_path: Path):
        project = OctronProject(tmp_path / "octron")
        entry = _entry(tmp_path)
        a = project.ensure_object(entry, "animal", "female")
        b = project.ensure_object(entry, "animal", "male")
        assert a.label_id == b.label_id and a.obj_id != b.obj_id
        assert a.layer_name == "animal female"

    def test_remove_object_deletes_store_and_keeps_the_others(self, tmp_path: Path):
        project = OctronProject(tmp_path / "octron")
        entry = _entry(tmp_path)
        female = project.ensure_object(entry, "female", "")
        male = project.ensure_object(entry, "male", "")
        assert project.remove_object(entry, "female", "")
        assert not female.mask_path.exists() and male.mask_path.exists()
        assert project.load_objects(entry) == [male]
        assert not project.remove_object(entry, "female", "")
        assert project.remove_object(entry, "male", "")
        assert not entry.organizer_path.exists()

    def test_a_given_colour_is_stored_and_updated(self, tmp_path: Path):
        """One colour per individual across cameras: the caller's colour wins over OCTRON's palette."""
        project = OctronProject(tmp_path / "octron")
        entry = _entry(tmp_path)
        red = [0.95, 0.30, 0.30, 1.0]
        obj = project.ensure_object(entry, "male", "", color=red)
        assert obj.color == red
        assert project.load_objects(entry)[0].color == red
        green = [0.24, 0.86, 0.52, 1.0]
        same = project.ensure_object(entry, "male", "", color=green)
        assert same.obj_id == obj.obj_id and same.color == green
        assert project.load_objects(entry)[0].color == green

    def test_annotated_frame_counts(self, tmp_path: Path):
        from octron.sam_octron.helpers.sam_zarr import mark_frames_annotated

        project = OctronProject(tmp_path / "octron")
        entry = _entry(tmp_path)
        obj = project.ensure_object(entry, "female", "")
        array = project.open_mask(entry, "female", "")
        array[3] = np.ones((48, 64), dtype=np.int16)
        mark_frames_annotated(array, [3])
        assert project.label_counts(entry) == {"female": 1}
        project.save_organizer(entry, [obj], sam_model="sam2_large")
        raw = json.loads(entry.organizer_path.read_text())
        assert raw["entries"][str(obj.obj_id)]["prediction_layer_metadata"]["num_predicted_indices"] == 1
        assert raw["settings"]["model_name"] == "sam2_large"

    def test_matches_octrons_own_reconstruction(self, tmp_path: Path):
        """OCTRON rebuilds the organizer from the zarr stores; ours must say the same."""
        from octron.sam_octron.restore_object_organizer import reconstruct

        project = OctronProject(tmp_path / "octron")
        entry = _entry(tmp_path)
        project.ensure_object(entry, "female", "")
        project.ensure_object(entry, "male", "")
        ours = json.loads(entry.organizer_path.read_text())["entries"]
        theirs, _warnings = reconstruct(entry.folder, project_root=project.root, video_path=str(entry.path))
        by_label = {e["label"]: e for e in theirs["entries"].values()}
        for e in ours.values():
            t = by_label[e["label"]]
            assert t["label_id"] == e["label_id"]
            assert t["prediction_layer_metadata"]["zarr_path"] == e["prediction_layer_metadata"]["zarr_path"]
            assert t["prediction_layer_metadata"]["data_shape"] == e["prediction_layer_metadata"]["data_shape"]


class TestTracks:
    def test_read_octron_csvs(self, tmp_path: Path):
        folder = tmp_path / "cam_bytetrack"
        folder.mkdir()
        header = (
            "video_name: x\nframe_count: 3\nframe_count_analyzed: 3\n"
            "video_height: 48\nvideo_width: 64\ncreated_at: t\n\n"
        )
        body = (
            "frame_counter,frame_idx,track_id,label,confidence,pos_x,pos_y,bbox_area,bbox_aspect_ratio,"
            "bbox_x_min,bbox_x_max,bbox_y_min,bbox_y_max\n"
            "0,0,1,female,0.9,10.0,20.0,100,1.0,5,15,15,25\n"
            "1,1,1,female,0.8,11.0,21.0,100,1.0,6,16,16,26\n"
        )
        (folder / "female_track_1.csv").write_text(header + body)
        (folder / "prediction_metadata.json").write_text("{}")
        tracks = read_tracks(folder)
        assert len(tracks) == 1
        t = tracks[0]
        assert (t.label, t.track_id) == ("female", 1)
        assert t.frame_idx.tolist() == [0, 1]
        assert t.pos_x.tolist() == [10.0, 11.0]
        assert t.bbox.shape == (2, 4)

    def test_mask_bbox(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert mask_bbox(mask) is None
        mask[2:5, 3:8] = 1
        assert mask_bbox(mask) == (3, 2, 8, 5)


def test_object_entry_layer_name():
    obj = ObjectEntry(obj_id=0, label="animal", suffix="", label_id=0, color=[0, 0, 0, 1], mask_path=Path("x"))
    assert obj.layer_name == "animal"


def test_exclusive_masks_takes_the_clicked_pixels_from_the_others():
    """Two individuals never share pixels in one camera: the newer click wins."""
    from ethograph.gui.box_annotate import exclusive_masks

    a = np.zeros((4, 4), dtype=np.uint8)
    a[0:3, 0:3] = 1
    b = np.zeros((4, 4), dtype=np.uint8)
    b[1:4, 1:4] = 1
    changed = exclusive_masks({0: a, 1: b, 2: None}, winner=1)
    assert list(changed) == [0]
    assert changed[0][1:3, 1:3].sum() == 0  # the overlap left A
    assert changed[0].sum() == a.sum() - 4  # and nothing else did
    assert exclusive_masks({0: a, 1: b}, winner=5) == {}
    # a loser that would keep only a sliver was the same animal: it is emptied
    c = np.zeros((4, 4), dtype=np.uint8)
    c[0:3, 0:4] = 1  # covers most of A
    changed = exclusive_masks({0: a, 1: c}, winner=1)
    assert changed[0].sum() == 0


class TestTrackState:
    """Predict is constrained to continue the stretch SAM's memory was built on."""

    def test_empty_never_predicts(self):
        from ethograph.gui.box_annotate import TrackState

        track = TrackState()
        assert track.predict_start(0) is None
        assert not track.click_resets(7)
        assert not track.seek_resets(500, chunk=15)

    def test_predict_runs_from_the_seed_wherever_the_playhead_is(self):
        from ethograph.gui.box_annotate import TrackState

        track = TrackState()
        track.note_click(100)
        assert track.predict_start(100) == 100
        assert track.predict_start(104) == 100  # guard 1: always from the seed

    def test_after_predicting_only_the_last_frame_continues(self):
        from ethograph.gui.box_annotate import TrackState

        track = TrackState()
        track.note_click(100)
        track.note_predicted(100, 114)
        assert track.span == (100, 114)
        assert track.predict_start(114) == 114
        assert track.predict_start(105) is None
        track.note_predicted(114, 128)
        assert track.span == (100, 128)

    def test_a_click_inside_the_span_corrects_and_outside_starts_afresh(self):
        from ethograph.gui.box_annotate import TrackState

        track = TrackState()
        track.note_click(100)
        track.note_predicted(100, 114)
        assert not track.click_resets(107)
        track.note_click(107)
        assert track.predict_start(120) == 107
        assert track.click_resets(300)
        track.note_click(300)  # after the dialog reset the memory
        assert track.span is None and track.seed == 300

    def test_seeking_farther_than_a_chunk_drops_the_memory(self):
        from ethograph.gui.box_annotate import TrackState

        track = TrackState()
        track.note_click(100)
        track.note_predicted(100, 114)
        assert not track.seek_resets(129, chunk=15)
        assert not track.seek_resets(85, chunk=15)
        assert track.seek_resets(130, chunk=15)
        assert track.seek_resets(84, chunk=15)
        track.reset()
        assert track.state == TrackState.EMPTY and track.predict_start(114) is None
