"""Tests for ethograph.io.nwb_alignment and ethograph.io.time_model navigation."""

import numpy as np
import pandas as pd

from ethograph.io.nwb_alignment import (
    EmpytAlignment,
    TableAlignment,
    _build_trials_ep,
)
from ethograph.io.time_model import (
    infer_slider_range,
)

# ---------------------------------------------------------------------------
# trials_ep must exist when alignment has start/stop times
# ---------------------------------------------------------------------------


class TestTrialsEp:
    def _make_table_alignment(self, n_trials: int = 5, with_trial_col: bool = True) -> TableAlignment:
        rng = np.random.default_rng(42)
        gaps = rng.uniform(2.0, 5.0, n_trials)
        starts = np.cumsum(gaps)
        durations = rng.uniform(0.5, 1.5, n_trials)
        stops = starts + durations
        df = pd.DataFrame({"start_time": starts, "stop_time": stops})
        if with_trial_col:
            df["trial"] = list(range(1, n_trials + 1))
        return TableAlignment(df)

    def test_table_alignment_trials_ep(self):
        alignment = self._make_table_alignment()
        ep = alignment.trials_ep
        assert ep is not None
        assert len(ep) == 5

    def test_empty_alignment_trials_ep_none(self):
        assert EmpytAlignment().trials_ep is None

    def test_trial_metadata_ids(self):
        alignment = self._make_table_alignment()
        ep = alignment.trials_ep
        assert list(ep.metadata["trial"]) == [1, 2, 3, 4, 5]

    def test_no_trial_column(self):
        alignment = self._make_table_alignment(n_trials=3, with_trial_col=False)
        ep = alignment.trials_ep
        assert ep is not None
        assert len(ep) == 3
        assert list(ep.metadata["trial"]) == [1, 2, 3]

    def test_consistency_start_stop_and_ep(self):
        alignment = self._make_table_alignment(n_trials=10)
        ep = alignment.trials_ep
        assert ep is not None
        for i, trial_id in enumerate(range(1, 11)):
            start = alignment.start_time(trial_id)
            stop = alignment.stop_time(trial_id)
            if stop is not None and stop > start:
                assert abs(float(ep.start[i]) - start) < 1e-6
                assert abs(float(ep.end[i]) - stop) < 1e-6


# ---------------------------------------------------------------------------
# _build_trials_ep with session_range fallback
# ---------------------------------------------------------------------------


class TestBuildTrialsEpCascade:
    def test_session_end_fallback_for_last_trial(self):
        """Last trial with no stop_time should use session_end."""
        df = pd.DataFrame(
            {
                "trial": [1, 2, 3],
                "start_time": [0.0, 10.0, 20.0],
                "stop_time": [5.0, 15.0, np.nan],
            }
        )
        ep = _build_trials_ep(df, session_end=30.0)
        assert ep is not None
        assert len(ep) == 3
        assert float(ep.end[2]) == 30.0

    def test_no_session_end_drops_last_trial(self):
        """Without session_end, last trial with no stop should be dropped."""
        df = pd.DataFrame(
            {
                "trial": [1, 2, 3],
                "start_time": [0.0, 10.0, 20.0],
                "stop_time": [5.0, 15.0, np.nan],
            }
        )
        ep = _build_trials_ep(df, session_end=None)
        assert ep is not None
        assert len(ep) == 2

    def test_all_stops_present_ignores_session_end(self):
        df = pd.DataFrame(
            {
                "trial": [1, 2],
                "start_time": [0.0, 10.0],
                "stop_time": [5.0, 15.0],
            }
        )
        ep = _build_trials_ep(df, session_end=100.0)
        assert ep is not None
        assert len(ep) == 2
        assert float(ep.end[1]) == 15.0  # session_end not used


# ---------------------------------------------------------------------------
# infer_slider_range cascade
# ---------------------------------------------------------------------------


class TestInferSliderRange:
    def test_trial_with_stop(self):
        """Trial with start+stop → scope = 'trial'."""
        df = pd.DataFrame(
            {
                "trial": [1, 2],
                "start_time": [0.0, 10.0],
                "stop_time": [5.0, 15.0],
            }
        )
        alignment = TableAlignment(df)
        scope, tr = infer_slider_range(alignment, 1)
        assert scope == "trial"
        assert tr is not None
        assert abs(tr.duration - 5.0) < 1e-6

    def test_start_only_extends_to_next(self):
        """Trial with start only → scope = 'trial_start', extends to next start."""
        df = pd.DataFrame(
            {
                "trial": [1, 2, 3],
                "start_time": [0.0, 10.0, 20.0],
            }
        )
        alignment = TableAlignment(df)
        scope, tr = infer_slider_range(alignment, 1)
        assert scope == "trial_start"
        assert tr is not None
        assert abs(tr.end_s - 10.0) < 1e-6

    def test_last_trial_no_stop_uses_session(self):
        """Last trial with no stop → uses session extent."""
        from ethograph.io.time_model import SourceCollection

        df = pd.DataFrame(
            {
                "trial": [1, 2],
                "start_time": [0.0, 10.0],
            }
        )
        alignment = TableAlignment(df)
        sc = SourceCollection()
        sc.set_trials([1, 2], [0.0, 10.0], [5.0, 25.0])
        scope, tr = infer_slider_range(alignment, 2, sc)
        assert scope == "session"
        assert tr is not None
        assert abs(tr.end_s - 15.0) < 1e-6  # 25.0 - 10.0

    def test_empty_alignment(self):
        alignment = EmpytAlignment()
        scope, tr = infer_slider_range(alignment, 1)
        assert scope == "session"
        assert tr is None


# ---------------------------------------------------------------------------
# Media filenames (the trials table's rightmost columns)
# ---------------------------------------------------------------------------


class TestMediaFilename:
    """A per-trial stream names one file per trial, a session-wide one names
    the same file for every trial — and neither needs the media on disk."""

    def _alignment(self, tmp_path):
        from ethograph.io.nwb_alignment import align_media_from_streams, make_nwb_alignment

        trials = pd.DataFrame({"trial": [1, 2, 3], "start_time": [0.0, 10.5, 22.3], "stop_time": [8.2, 19.1, 30.0]})
        streams = [
            {"name": "video_cam-1", "files": ["/media/t1.mp4", "/media/t2.mp4", "/media/t3.mp4"], "rate": 30.0},
            {"name": "audio_mic-1", "files": ["/media/session.wav"], "rate": 48000.0, "starting_time": 0.0},
        ]
        return make_nwb_alignment(align_media_from_streams(trials, streams, tmp_path / "alignment.nwb"))

    def test_per_trial_stream(self, tmp_path):
        alignment = self._alignment(tmp_path)
        assert [alignment.media_filename(t, "video", "cam-1") for t in (1, 2, 3)] == [
            "t1.mp4",
            "t2.mp4",
            "t3.mp4",
        ]

    def test_session_wide_stream(self, tmp_path):
        alignment = self._alignment(tmp_path)
        assert [alignment.media_filename(t, "audio", "mic-1") for t in (1, 3)] == [
            "session.wav",
            "session.wav",
        ]

    def test_filenames_stored_as_trial_columns(self, tmp_path):
        """The other spelling: a drop paired per trial writes filename columns,
        and the table must name the file, not the authoring machine's path."""
        from ethograph.io.metadata_table import load_metadata_df
        from ethograph.io.nwb_alignment import make_nwb_alignment
        from ethograph.io.pairing import pair_media

        table = pd.DataFrame(
            {
                "trial": [1, 2],
                "video_cam-1": ["/media/clip_1.mp4", "/media/clip_2.mp4"],
                "start_time": [0.0, 5.0],
                "stop_time": [5.0, 10.0],
            }
        )
        nwb = tmp_path / "alignment.nwb"
        pair_media(trial_table=table, stream_rates={"video": 30.0}, output_path=nwb)

        df, _ = load_metadata_df(nwb_alignment=make_nwb_alignment(nwb), trial_ids=[1, 2])
        assert list(df["video_cam-1"]) == ["clip_1.mp4", "clip_2.mp4"]

    def test_metadata_table_carries_them(self, tmp_path):
        from ethograph.io.metadata_table import load_metadata_df

        df, _ = load_metadata_df(nwb_alignment=self._alignment(tmp_path), trial_ids=[1, 2, 3])
        assert list(df.columns) == ["trial", "video_cam-1", "audio_mic-1"]
