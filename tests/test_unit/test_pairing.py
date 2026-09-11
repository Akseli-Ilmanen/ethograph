"""Tests for ethograph.io.pairing: discover_media + pair_media."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dateutil.tz import tzlocal

from ethograph.io.nwb_alignment import NWBAlignment, align_media_from_streams
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
            tmp_path,
            [
                SourceSpec(stream="video", folder="video"),
                SourceSpec(stream="pose", folder="pose"),
            ],
        )

        assert list(df["trial"]) == [1, 2, 3]
        assert df["video_cam-1"].tolist() == ["a1.mp4", "b2.mp4", "c3.mp4"]
        assert df["pose_cam-1"].tolist() == ["a1.h5", "b2.h5", "c3.h5"]

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
                tmp_path,
                [
                    SourceSpec(stream="video", folder="video"),
                    SourceSpec(stream="audio", folder="audio"),
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
            folder="video",
            pattern=r"trial(?P<trial>\d+)_(?P<camera>\w+)",
        )
        df = discover_media(tmp_path, [source])

        assert set(df.columns) == {"trial", "video_camA", "video_camB"}
        assert sorted(df["trial"].tolist()) == [1, 2]
        row1 = df[df["trial"] == 1].iloc[0]
        assert row1["video_camA"] == "trial1_camA.mp4"
        assert row1["video_camB"] == "trial1_camB.mp4"

    def test_non_matching_file_raises(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        (video_dir / "trial1_camA.mp4").touch()
        (video_dir / "nomatch.mp4").touch()

        source = SourceSpec(
            stream="video",
            folder="video",
            pattern=r"trial(?P<trial>\d+)_(?P<camera>\w+)",
        )
        with pytest.raises(ValueError, match="nomatch"):
            discover_media(tmp_path, [source])

    def test_empty_folder_raises(self, tmp_path: Path):
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        with pytest.raises(ValueError):
            discover_media(tmp_path, [SourceSpec(stream="video", folder="video")])


class TestPairMedia:
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


class TestAlignMediaFromStreamsTimestampsRefused:
    def test_timestamps_key_raises(self, tmp_path: Path):
        trials = pd.DataFrame({"trial": [1], "start_time": [0.0], "stop_time": [1.0]})
        streams = [{"name": "ephys_probe-1", "files": ["x.dat"], "timestamps": np.array([0.0, 0.1])}]
        with pytest.raises(ValueError, match="neuroconv"):
            align_media_from_streams(trials, streams, tmp_path / "out.nwb")
