"""A trial finds its video file by time, never by position.

Fixtures are written in the shape neuroconv's ``ExternalVideoInterface``
produces: one ``ImageSeries`` in ``acquisition`` (default name ``Video {stem}``),
``format="external"``, cumulative ``starting_frame`` per file, and either
``rate`` + ``starting_time`` or dense ``timestamps``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pynwb
import pytest
from dateutil.tz import tzlocal
from pynwb.image import ImageSeries

from ethograph.io.nwb_alignment import NWBAlignment

FPS = 30.0
NAME = "Video session"


def _nwbfile() -> pynwb.NWBFile:
    return pynwb.NWBFile(
        session_description="neuroconv-shaped",
        identifier="x",
        session_start_time=datetime(2026, 1, 1, tzinfo=tzlocal()),
    )


def _write(nwbfile: pynwb.NWBFile, path: Path) -> NWBAlignment:
    """Write the NWB and touch its external files beside it (paths resolve only on disk)."""
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    for series in nwbfile.acquisition.values():
        for name in series.external_file:
            (path.parent / name).touch()
    return NWBAlignment(path)


def _add_trials(nwbfile: pynwb.NWBFile, bounds: list[tuple[float, float]]) -> None:
    for start, stop in bounds:
        nwbfile.add_trial(start_time=start, stop_time=stop)


@pytest.fixture
def triggered(tmp_path: Path) -> NWBAlignment:
    """One file per trial, frames in bursts with real gaps (dense timestamps)."""
    nwbfile = _nwbfile()
    onsets = [1.0, 20.0, 45.0]
    n = 300
    ts = np.concatenate([o + np.arange(n) / FPS for o in onsets])
    nwbfile.add_acquisition(
        ImageSeries(
            name=NAME,
            description="",
            unit="Frames",
            format="external",
            external_file=["t1.mp4", "t2.mp4", "t3.mp4"],
            starting_frame=[0, n, 2 * n],
            timestamps=ts,
        )
    )
    _add_trials(nwbfile, [(o, o + n / FPS) for o in onsets])
    return _write(nwbfile, tmp_path / "triggered.nwb")


@pytest.fixture
def free_running(tmp_path: Path) -> NWBAlignment:
    """One file for the whole session, rate mode."""
    nwbfile = _nwbfile()
    nwbfile.add_acquisition(
        ImageSeries(
            name=NAME,
            description="",
            unit="Frames",
            format="external",
            external_file=["session.mp4"],
            starting_frame=[0],
            rate=FPS,
            starting_time=0.0,
            num_samples=np.uint64(60 * FPS),
        )
    )
    _add_trials(nwbfile, [(10.0, 20.0), (30.0, 40.0), (50.0, 55.0)])
    return _write(nwbfile, tmp_path / "free.nwb")


@pytest.fixture
def split_files(tmp_path: Path) -> NWBAlignment:
    """Free-running camera split by the recorder into three contiguous files."""
    nwbfile = _nwbfile()
    n = 300  # 10 s each
    nwbfile.add_acquisition(
        ImageSeries(
            name=NAME,
            description="",
            unit="Frames",
            format="external",
            external_file=["part1.mp4", "part2.mp4", "part3.mp4"],
            starting_frame=[0, n, 2 * n],
            rate=FPS,
            starting_time=0.0,
            num_samples=np.uint64(3 * n),
        )
    )
    # Five trials over three files: trial 2 sits inside part2, trial 4 inside part3.
    _add_trials(nwbfile, [(1.0, 4.0), (12.0, 15.0), (18.0, 19.0), (22.0, 27.0), (28.0, 29.5)])
    return _write(nwbfile, tmp_path / "split.nwb")


@pytest.fixture
def trialless(tmp_path: Path) -> NWBAlignment:
    """Free-running camera and no trials table at all."""
    nwbfile = _nwbfile()
    nwbfile.add_acquisition(
        ImageSeries(
            name=NAME,
            description="",
            unit="Frames",
            format="external",
            external_file=["session.mp4"],
            starting_frame=[0],
            rate=FPS,
            starting_time=2.5,
            num_samples=np.uint64(100),
        )
    )
    return _write(nwbfile, tmp_path / "trialless.nwb")


class TestDiscovery:
    def test_default_neuroconv_name_is_a_camera(self, free_running: NWBAlignment):
        assert free_running.cameras == [NAME]
        assert free_running.get_stream_rate("video", NAME) == FPS


class TestTriggered:
    def test_each_trial_takes_its_own_file(self, triggered: NWBAlignment):
        for trial, name in zip((1, 2, 3), ("t1.mp4", "t2.mp4", "t3.mp4")):
            path = triggered.resolve_media_path(trial, "video", NAME, fallback_folder=None)
            assert path is not None and Path(path).name == name
            assert triggered.stream_offset_for_trial(trial, "video", NAME) == pytest.approx(0.0)

    def test_trial_starting_in_the_gap_before_its_file(self, triggered: NWBAlignment):
        # Trial 2 is retimed to start 0.2 s before its burst (trigger before first exposure).
        df = triggered.trials_df
        assert df.loc[1, "start_time"] == pytest.approx(20.0)
        triggered._trials_df_cache = df.assign(start_time=[1.0, 19.8, 45.0])
        assert Path(triggered.resolve_media_path(2, "video", NAME)).name == "t2.mp4"
        assert triggered.stream_offset_for_trial(2, "video", NAME) == pytest.approx(0.2)


class TestFreeRunning:
    def test_every_trial_reads_the_one_file(self, free_running: NWBAlignment):
        for trial in (1, 2, 3):
            assert Path(free_running.resolve_media_path(trial, "video", NAME)).name == "session.mp4"

    def test_offset_is_file_start_minus_trial_start(self, free_running: NWBAlignment):
        assert free_running.stream_offset_for_trial(1, "video", NAME) == pytest.approx(-10.0)
        assert free_running.stream_offset_for_trial(2, "video", NAME) == pytest.approx(-30.0)
        assert free_running.stream_offset_for_trial(3, "video", NAME) == pytest.approx(-50.0)


class TestSplitFiles:
    @pytest.mark.parametrize(
        ("trial", "name", "offset"),
        [
            (1, "part1.mp4", -1.0),
            (2, "part2.mp4", -2.0),
            (3, "part2.mp4", -8.0),
            (4, "part3.mp4", -2.0),
            (5, "part3.mp4", -8.0),
        ],
    )
    def test_file_and_offset_follow_time(self, split_files: NWBAlignment, trial, name, offset):
        assert Path(split_files.resolve_media_path(trial, "video", NAME)).name == name
        assert split_files.stream_offset_for_trial(trial, "video", NAME) == pytest.approx(offset)

    def test_spans_are_contiguous(self, split_files: NWBAlignment):
        spans = split_files.file_time_spans("video", NAME)
        assert [(Path(p).name, a, b) for p, a, b in spans] == [
            ("part1.mp4", 0.0, 10.0),
            ("part2.mp4", 10.0, 20.0),
            ("part3.mp4", 20.0, 30.0),
        ]


class TestTrialless:
    def test_one_file_no_trials(self, trialless: NWBAlignment):
        assert trialless.trials_df.empty
        assert Path(trialless.resolve_media_path(1, "video", NAME)).name == "session.mp4"
        assert trialless.stream_offset_for_trial(1, "video", NAME) == pytest.approx(2.5)
