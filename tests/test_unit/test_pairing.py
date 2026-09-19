"""Tests for ethograph.io.pairing: discover_media + pair_media."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dateutil.tz import tzlocal

from ethograph.io.nwb_alignment import NWBAlignment
from ethograph.io.pairing import SourceSpec, discover_media, pair_media


class TestDiscoverMedia:
    def test_natsort_pairing_of_two_sources(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        pose_dir = tmp_path / "pose"
        pose_dir.mkdir()
        for name in ("b2.mp4", "a1.mp4", "c3.mp4"):
            (video_dir / name).touch()
        for name in ("b2.h5", "a1.h5", "c3.h5"):
            (pose_dir / name).touch()

        df = discover_media(
            [
                SourceSpec(stream="video", folder=str(video_dir)),
                SourceSpec(stream="pose", folder=str(pose_dir)),
            ],
        )

        assert list(df["trial"]) == [1, 2, 3]
        assert df["video_cam-1"].tolist() == [str(video_dir / n) for n in ("a1.mp4", "b2.mp4", "c3.mp4")]
        assert df["pose_cam-1"].tolist() == [str(pose_dir / n) for n in ("a1.h5", "b2.h5", "c3.h5")]

    def test_patternless_count_mismatch_raises(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for i in range(3):
            (video_dir / f"t{i}.mp4").touch()
        for i in range(2):
            (audio_dir / f"t{i}.wav").touch()

        with pytest.raises(ValueError, match="3|2"):
            discover_media(
                [
                    SourceSpec(stream="video", folder=str(video_dir)),
                    SourceSpec(stream="audio", folder=str(audio_dir)),
                ],
            )

    def test_pattern_with_camera_group_makes_two_columns(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        for trial in (1, 2):
            for cam in ("camA", "camB"):
                (video_dir / f"trial{trial}_{cam}.mp4").touch()

        source = SourceSpec(
            stream="video",
            folder=str(video_dir),
            pattern=r"trial(?P<trial>\d+)_(?P<camera>\w+)",
        )
        df = discover_media([source])

        assert set(df.columns) == {"trial", "video_camA", "video_camB"}
        assert sorted(df["trial"].tolist()) == [1, 2]
        row1 = df[df["trial"] == 1].iloc[0]
        assert row1["video_camA"] == str(video_dir / "trial1_camA.mp4")
        assert row1["video_camB"] == str(video_dir / "trial1_camB.mp4")

    def test_non_matching_file_is_skipped(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        (video_dir / "trial1_camA.mp4").touch()
        (video_dir / "nomatch.mp4").touch()

        source = SourceSpec(
            stream="video",
            folder=str(video_dir),
            pattern=r"trial(?P<trial>\d+)_(?P<camera>\w+)",
        )
        table = discover_media([source])
        assert list(table["trial"]) == [1]
        assert table.iloc[0]["video_camA"] == str(video_dir / "trial1_camA.mp4")

    def test_pattern_matching_nothing_raises(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        (video_dir / "nomatch.mp4").touch()

        source = SourceSpec(stream="video", folder=str(video_dir), pattern=r"trial(?P<trial>\d+)")
        with pytest.raises(ValueError, match="matches none"):
            discover_media([source])

    def test_extension_narrows_a_mixed_folder(self, tmp_path: Path):
        """A DLC folder holds .h5 and .csv twins of every file; one extension keeps one twin."""
        pose_dir = tmp_path / "pose"
        pose_dir.mkdir()
        for name in ("trial1_camA.h5", "trial1_camA.csv", "trial2_camA.h5", "trial2_camA.csv"):
            (pose_dir / name).touch()
        pattern = r"trial(?P<trial>\d+)_(?P<camera>\w+)"

        with pytest.raises(ValueError, match="trial1_camA.csv, trial1_camA.h5"):
            discover_media([SourceSpec(stream="pose", folder=str(pose_dir), pattern=pattern)])

        df = discover_media([SourceSpec(stream="pose", folder=str(pose_dir), pattern=pattern, extension=".h5")])
        assert list(df["pose_camA"]) == [str(pose_dir / "trial1_camA.h5"), str(pose_dir / "trial2_camA.h5")]

    def test_extension_must_belong_to_the_stream(self, tmp_path: Path):
        pose_dir = tmp_path / "pose"
        pose_dir.mkdir()
        (pose_dir / "a.h5").touch()
        with pytest.raises(ValueError, match="not a pose extension"):
            discover_media([SourceSpec(stream="pose", folder=str(pose_dir), extension=".mp4")])

    def test_empty_folder_raises(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        with pytest.raises(ValueError):
            discover_media([SourceSpec(stream="video", folder=str(video_dir))])


class TestPairMedia:
    def test_full_paths_become_basenames_in_trials_and_paths_in_external_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The table says where the files are; the NWB stores names per trial and paths per stream."""
        from ethograph.utils import stream_durations

        cam1 = tmp_path / "rig" / "cam1"
        cam2 = tmp_path / "elsewhere" / "cam2"
        cam1.mkdir(parents=True)
        cam2.mkdir(parents=True)
        for d in (cam1, cam2):
            for i in (1, 2):
                (d / f"t{i}.mp4").touch()
        monkeypatch.setattr(stream_durations, "get_video_duration", lambda path: 4.0)

        table = pd.DataFrame(
            {
                "trial": [1, 2],
                "video_cam-1": [str(cam1 / "t1.mp4"), str(cam1 / "t2.mp4")],
                "video_cam-2": [str(cam2 / "t1.mp4"), str(cam2 / "t2.mp4")],
            }
        )
        out = tmp_path / "session" / ".ethograph" / "alignment.nwb"
        pair_media(table, stream_rates={"video": 30.0}, output_path=out)

        align = NWBAlignment(out)
        assert align.get_media(2, "video", "cam-2") == "t2.mp4"
        assert align.resolve_media_path(2, "video", "cam-2") == str(cam2 / "t2.mp4")
        assert align.resolve_media_path(1, "video", "cam-1") == str(cam1 / "t1.mp4")
        assert align.start_time(2) == pytest.approx(4.0)

    def test_writes_sidecar_readable_by_nwb_alignment(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        for i in (1, 2, 3):
            (video_dir / f"t{i}.mp4").touch()
        audio_file = tmp_path / "session.wav"
        audio_file.touch()

        trial_table = pd.DataFrame(
            {
                "trial": [1, 2, 3],
                "start_time": [0.0, 10.0, 20.0],
                "stop_time": [8.0, 18.0, 28.0],
                "video_cam-1": ["t1.mp4", "t2.mp4", "t3.mp4"],
            }
        )
        out = tmp_path / ".ethograph" / "alignment.nwb"

        pair_media(
            trial_table,
            stream_rates={"video": 30.0},
            session_wide={"audio_mic-1": (str(audio_file), 44100.0, -2.0)},
            output_path=out,
        )

        align = NWBAlignment(out)
        assert align.cameras == ["cam-1"]
        assert align.mics == ["mic-1"]

        resolved = align.resolve_media_path(1, "video", "cam-1", fallback_folder=str(video_dir))
        assert resolved is not None and Path(resolved).name == "t1.mp4"

        # starting_time - trial start, for trial 1 (start_time == 0.0)
        assert align.stream_offset_for_trial(1, "audio", "mic-1") == pytest.approx(-2.0)

    def test_pair_into_existing_neuroconv_nwb_adds_audio_leaves_video_intact(self, tmp_path: Path):
        import pynwb
        from pynwb import NWBHDF5IO
        from pynwb.image import ImageSeries

        nwbfile = pynwb.NWBFile(
            session_description="neuroconv-shaped",
            identifier="x",
            session_start_time=datetime(2026, 1, 1, tzinfo=tzlocal()),
        )
        nwbfile.add_acquisition(
            ImageSeries(
                name="Video session",
                description="",
                unit="Frames",
                format="external",
                external_file=["session.mp4"],
                starting_frame=[0],
                rate=30.0,
                starting_time=0.0,
                num_samples=np.uint64(300),
            )
        )
        nwbfile.add_trial(start_time=0.0, stop_time=3.0)
        nwbfile.add_trial(start_time=5.0, stop_time=8.0)

        path = tmp_path / "alignment.nwb"
        with NWBHDF5IO(str(path), "w") as io:
            io.write(nwbfile)

        audio_file = tmp_path / "session.wav"
        audio_file.touch()

        trial_table = pd.DataFrame({"trial": [1, 2]})
        pair_media(
            trial_table,
            session_wide={"audio_mic-1": (str(audio_file), 44100.0, 0.0)},
            output_path=path,
        )

        align = NWBAlignment(path)
        assert align.mics == ["mic-1"]
        assert "Video session" in align.nwb.acquisition
        video = align.nwb.acquisition["Video session"]
        assert list(video.external_file) == ["session.mp4"]
        assert float(video.rate) == pytest.approx(30.0)

    def test_row_count_mismatch_against_existing_nwb_raises(self, tmp_path: Path):
        import pynwb
        from pynwb import NWBHDF5IO

        nwbfile = pynwb.NWBFile(
            session_description="neuroconv-shaped",
            identifier="x",
            session_start_time=datetime(2026, 1, 1, tzinfo=tzlocal()),
        )
        nwbfile.add_trial(start_time=0.0, stop_time=3.0)
        nwbfile.add_trial(start_time=5.0, stop_time=8.0)
        path = tmp_path / "alignment.nwb"
        with NWBHDF5IO(str(path), "w") as io:
            io.write(nwbfile)

        trial_table = pd.DataFrame({"trial": [1, 2, 3]})
        with pytest.raises(ValueError, match="rows"):
            pair_media(trial_table, output_path=path)


class TestPerDeviceRate:
    def test_device_key_overrides_stream_rate(self, tmp_path: Path):
        table = pd.DataFrame(
            {
                "trial": [1],
                "start_time": [0.0],
                "stop_time": [2.0],
                "video_cam-1": ["a.mp4"],
                "video_cam-2": ["b.mp4"],
            }
        )
        out = tmp_path / "out.nwb"
        pair_media(table, stream_rates={"video": 30.0, "video_cam-2": 60.0}, output_path=out)

        alignment = NWBAlignment(out)
        assert alignment.get_stream_rate("video", "cam-1") == 30.0
        assert alignment.get_stream_rate("video", "cam-2") == 60.0
        alignment.close()


def test_media_folders_come_from_the_recorded_paths(tmp_path: Path):
    """The alignment tells the GUI where each stream's files are, per stream, when they exist here."""
    cam = tmp_path / "cams"
    cam.mkdir()
    (cam / "t1.mp4").touch()
    table = pd.DataFrame({"trial": [1], "start_time": [0.0], "stop_time": [1.0], "video_cam-1": [str(cam / "t1.mp4")]})
    out = tmp_path / ".ethograph" / "alignment.nwb"
    pair_media(table, stream_rates={"video": 30.0}, output_path=out)
    assert NWBAlignment(out).media_folders() == {"video": cam}


class TestOnExisting:
    """What an existing file means is the caller's decision, never the filesystem's.

    Pairing used to pick between "add to this NWB" and "write a new one" by looking
    at the disk, so re-dropping a folder hit the additive path and died on
    ``column 'video_cam-1' already exists``.
    """

    def _table(self, tmp_path: Path, monkeypatch) -> pd.DataFrame:
        from ethograph.utils import stream_durations

        cam = tmp_path / "video"
        cam.mkdir(exist_ok=True)
        for i in (1, 2):
            (cam / f"t{i}.mp4").touch()
        monkeypatch.setattr(stream_durations, "get_video_duration", lambda path: 4.0)
        return pd.DataFrame({"trial": [1, 2], "video_cam-1": [str(cam / "t1.mp4"), str(cam / "t2.mp4")]})

    def test_replace_is_idempotent(self, tmp_path: Path, monkeypatch):
        """Pairing the same folder twice lands where pairing it once did."""
        table = self._table(tmp_path, monkeypatch)
        out = tmp_path / "session" / ".ethograph" / "alignment.nwb"

        pair_media(table, stream_rates={"video": 30.0}, output_path=out, on_existing="replace")
        pair_media(table, stream_rates={"video": 30.0}, output_path=out, on_existing="replace")

        align = NWBAlignment(out)
        assert align.get_media(2, "video", "cam-1") == "t2.mp4"
        assert align.cameras == ["cam-1"], "one camera, not one per run"

    def test_extend_refuses_media_it_already_pairs_and_says_what_to_pass(self, tmp_path: Path, monkeypatch):
        table = self._table(tmp_path, monkeypatch)
        out = tmp_path / "session" / ".ethograph" / "alignment.nwb"
        pair_media(table, stream_rates={"video": 30.0}, output_path=out)

        with pytest.raises(ValueError, match=r"already pairs video_cam-1.*on_existing='replace'"):
            pair_media(table, stream_rates={"video": 30.0}, output_path=out)

    def test_an_unknown_mode_is_refused_by_name(self, tmp_path: Path, monkeypatch):
        table = self._table(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="on_existing must be"):
            pair_media(table, output_path=tmp_path / "a.nwb", on_existing="overwrite")
