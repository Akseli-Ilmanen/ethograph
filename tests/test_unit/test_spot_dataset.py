"""Stage 1 of the pixel pipeline: the branching that has an edge case.

What earns a test here is whether a folder of frames on disk can be trusted
for the current config (`export_is_current`'s crop/size disagreement branch)
and the crop's effect on `plan_session`'s pre-resize geometry -- not the
`TrialRecord` fields themselves, which only restate what the code assigns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ethograph.spot import dataset as spot_dataset
from ethograph.spot.config import CropConfig, config_from_dict
from ethograph.spot.dataset import TrialRecord, export_is_current, plan_session


def _record(tmp_path: Path, num_frames=3, width=640, height=480, crop=None) -> TrialRecord:
    return TrialRecord(
        video_id="v",
        source=tmp_path / "ses.nc",
        trial=1,
        video_path=tmp_path / "v.mp4",
        num_frames=num_frames,
        fps=200.0,
        width=width,
        height=height,
        events={},
        crop=crop,
    )


def _write_frames(out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (out_dir / f"{i:06d}.jpg").write_bytes(b"")


class TestCropConfigChecks:
    """`check_fits`/`validate` are the arithmetic with an edge case; the
    dataclass's fields themselves are not tested."""

    def test_fits_inside_the_frame(self):
        CropConfig(x0=0, y0=0, x1=100, y1=100).check_fits(200, 200, "trial 1")

    def test_x_overflow_names_the_frame_size(self):
        with pytest.raises(ValueError, match="200x100"):
            CropConfig(x0=0, y0=0, x1=250, y1=50).check_fits(200, 100, "trial 1")

    def test_y_overflow_names_the_frame_size(self):
        with pytest.raises(ValueError, match="200x100"):
            CropConfig(x0=0, y0=0, x1=100, y1=150).check_fits(200, 100, "trial 1")

    def test_negative_corner_is_refused(self):
        with pytest.raises(ValueError, match="outside the frame"):
            CropConfig(x0=-1, y0=0, x1=100, y1=100).validate()

    def test_an_inverted_box_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            CropConfig(x0=10, y0=10, x1=10, y1=50).validate()
        with pytest.raises(ValueError, match="empty"):
            CropConfig(x0=10, y0=50, x1=100, y1=50).validate()

    def test_a_normal_box_passes(self):
        CropConfig(x0=0, y0=0, x1=10, y1=10).validate()


class TestExportIsCurrent:
    def test_missing_folder_is_not_current(self, tmp_path):
        assert export_is_current(tmp_path / "missing", _record(tmp_path)) is False

    def test_wrong_frame_count_is_not_current(self, tmp_path):
        out_dir = tmp_path / "v"
        _write_frames(out_dir, 2)
        assert export_is_current(out_dir, _record(tmp_path, num_frames=3)) is False

    def test_legacy_full_frame_export_with_no_crop_wanted_is_current(self, tmp_path):
        """A folder decoded before crops existed carries no `export.json`."""
        out_dir = tmp_path / "v"
        _write_frames(out_dir, 3)
        assert export_is_current(out_dir, _record(tmp_path, num_frames=3, crop=None)) is True

    def test_legacy_export_cannot_be_trusted_for_a_wanted_crop(self, tmp_path):
        """No `export.json` and the config now wants a crop -- re-decode."""
        out_dir = tmp_path / "v"
        _write_frames(out_dir, 3)
        record = _record(tmp_path, num_frames=3, crop=(0, 0, 100, 100))
        assert export_is_current(out_dir, record) is False

    def test_matching_export_json_is_current(self, tmp_path):
        out_dir = tmp_path / "v"
        _write_frames(out_dir, 3)
        record = _record(tmp_path, num_frames=3, width=200, height=200, crop=(0, 0, 100, 100))
        (out_dir / spot_dataset.EXPORT_FILE).write_text(json.dumps(record.export_spec()), encoding="utf-8")
        assert export_is_current(out_dir, record) is True

    def test_mismatched_export_json_is_not_current(self, tmp_path):
        out_dir = tmp_path / "v"
        _write_frames(out_dir, 3)
        stored = _record(tmp_path, num_frames=3, width=200, height=200, crop=(0, 0, 50, 50))
        (out_dir / spot_dataset.EXPORT_FILE).write_text(json.dumps(stored.export_spec()), encoding="utf-8")
        wanted = _record(tmp_path, num_frames=3, width=200, height=200, crop=(0, 0, 100, 100))
        assert export_is_current(out_dir, wanted) is False


class _FakeAlignment:
    def stream_offset_for_trial(self, trial, stream, device=None):
        return 0.0


class _FakeResult:
    def __init__(self, labels_df: pd.DataFrame):
        self.all_labels_df = labels_df
        self.nwb_alignment = _FakeAlignment()


class _FakeSpec:
    label = "ses"


class _FakeSession:
    """The handful of `Session` attributes/methods `plan_session` reads."""

    def __init__(self, tmp_path: Path, labels_df: pd.DataFrame, trial_ids=(1,)):
        self.spec = _FakeSpec()
        self.source = tmp_path / "ses.nc"
        self.trial_ids = list(trial_ids)
        self.result = _FakeResult(labels_df)
        self._video_path = tmp_path / "cam-1.mp4"

    def media_path(self, trial, stream: str = "video", device: str | None = None):
        return self._video_path

    def video_device(self, camera):
        return camera


class TestVideoDevice:
    """`labels.camera` is the config's name; the alignment may number its cameras and still point at `…-cam-1.mp4`."""

    def _session(self, tmp_path, cameras, media):
        from types import SimpleNamespace

        from ethograph.segment.config import SessionSpec
        from ethograph.segment.sessions import Session

        alignment = SimpleNamespace(cameras=cameras, get_media=lambda trial, stream, device: media.get(device))
        result = SimpleNamespace(nwb_alignment=alignment, trial_ids=[1, 2])
        return Session(spec=SessionSpec(source=tmp_path / "s.nc"), id="s", result=result)

    def test_own_name_numbered_camera_and_no_match(self, tmp_path):
        named = self._session(tmp_path, ["cam-1", "cam-2"], {"cam-1": "x-cam-1.mp4"})
        assert named.video_device("cam-1") == "cam-1"
        assert named.video_device(None) is None
        files = {"0": "2025-05-12_002_Ivy-cam-1.mp4", "1": "2025-05-12_002_Ivy-cam-2.mp4"}
        numbered = self._session(tmp_path, ["0", "1"], files)
        assert numbered.video_device("cam-1") == "0"
        assert numbered.video_device("cam-2") == "1"
        with pytest.raises(ValueError, match=r"no camera 'cam-3'.*\['0', '1'\]"):
            numbered.video_device("cam-3")


def _labels_df(onset_s: float = 1.0, label: int = 31) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": [1],
            "individual": [""],
            "individual_rec": [""],
            "labels": [label],
            "onset_s": [onset_s],
            "offset_s": [onset_s],
            "event_type": ["point"],
            "confidence": [1.0],
            "labeling_method": ["manual"],
        }
    )


class TestPlanSessionCrop:
    """`labels.crop` narrows the pre-resize size `plan_session` scales from."""

    def _config(self, tmp_path, crop: dict | None, frame_height: int = 100):
        source = tmp_path / "ses.nc"
        source.touch()
        labels: dict = {"classes": [31], "camera": "cam-1", "frame_height": frame_height}
        if crop is not None:
            labels["crop"] = crop
        return config_from_dict({"sessions": [str(source)], "labels": labels}, tmp_path)

    def test_crop_narrows_the_pre_resize_size_before_the_scale(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spot_dataset, "probe_video", lambda video: (200.0, 1000, 640, 480))
        config = self._config(tmp_path, {"x0": 40, "y0": 40, "x1": 240, "y1": 240})
        session = _FakeSession(tmp_path, _labels_df())
        records = plan_session(session, config)
        assert len(records) == 1
        record = records[0]
        assert record.crop == (40, 40, 240, 240)
        # crop is 200x200 -> scale = frame_height / crop.height = 100/200 = 0.5
        assert record.height == 100
        assert record.width == 100

    def test_no_crop_scales_the_whole_frame(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spot_dataset, "probe_video", lambda video: (200.0, 1000, 640, 480))
        config = self._config(tmp_path, None)
        session = _FakeSession(tmp_path, _labels_df())
        record = plan_session(session, config)[0]
        assert record.crop is None
        # 640x480 -> scale = 100/480; width = round(640 * 100/480) = 133
        assert record.height == 100
        assert record.width == 133

    def test_a_crop_reaching_outside_the_video_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spot_dataset, "probe_video", lambda video: (200.0, 1000, 640, 480))
        config = self._config(tmp_path, {"x0": 0, "y0": 0, "x1": 700, "y1": 100})
        session = _FakeSession(tmp_path, _labels_df())
        with pytest.raises(ValueError, match="reaches outside"):
            plan_session(session, config)


class TestSeveralEventsPerTrial:
    """A trial labelled with one class twice trains on both -- or is refused
    when inference could never return both."""

    def _config(self, tmp_path, cap):
        source = tmp_path / "ses.nc"
        source.touch()
        infer = {"max_events_per_trial": cap, "confidence": "focus"}
        return config_from_dict({"sessions": [str(source)], "labels": {"classes": [31]}, "infer": infer}, tmp_path)

    def _twice(self, tmp_path):
        df = pd.concat([_labels_df(onset_s=2.0), _labels_df(onset_s=1.0)], ignore_index=True)
        return _FakeSession(tmp_path, df)

    def test_refused_at_the_default_cap_naming_the_fix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spot_dataset, "probe_video", lambda video: (200.0, 1000, 640, 480))
        with pytest.raises(ValueError, match=r"trial 1.*2.*max_events_per_trial=1.*cut the trial"):
            plan_session(self._twice(tmp_path), self._config(tmp_path, cap=1))

    def test_both_events_are_kept_in_time_order_above_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spot_dataset, "probe_video", lambda video: (200.0, 1000, 640, 480))
        record = plan_session(self._twice(tmp_path), self._config(tmp_path, cap=2))[0]
        assert record.events == {"label_31": [200, 400]} and record.n_events == 2
        assert [e["frame"] for e in record.to_json()["events"]] == [200, 400]

    def test_inference_planning_never_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spot_dataset, "probe_video", lambda video: (200.0, 1000, 640, 480))
        assert len(plan_session(self._twice(tmp_path), self._config(tmp_path, cap=1), require_events=False)) == 1
