"""Tests for the metadata table system."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ethograph.io.metadata_table import (
    attach_media_columns,
    condition_columns,
    empty_metadata_df,
    load_metadata_df,
    load_metadata_tsv,
    metadata_from_intervalset,
    metadata_from_nwb_trials,
    metadata_tsv_path,
    order_metadata_columns,
    save_metadata_tsv,
)


def test_condition_columns():
    df = pd.DataFrame(
        {
            "trial": [1, 2],
            "genotype": ["WT", "KO"],
            "start_time": [0.0, 1.0],
            "video_cam-1": ["a.mp4", "b.mp4"],
        }
    )
    cols = condition_columns(df)
    assert cols == ["genotype"]
    assert "trial" not in cols


def test_save_load_roundtrip():
    df = pd.DataFrame(
        {
            "trial": [1, 2, 3],
            "genotype": ["WT", "KO", "WT"],
            "dose_mg": [0.0, 5.0, 10.0],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_metadata.tsv"
        save_metadata_tsv(path, df)
        loaded = load_metadata_tsv(path)
        assert list(loaded.columns) == ["trial", "genotype", "dose_mg"]
        assert len(loaded) == 3
        assert loaded.loc[1, "genotype"] == "KO"


def test_metadata_tsv_path():
    p = metadata_tsv_path("/data/experiment.nc")
    assert p.name == "experiment_metadata.tsv"


def test_empty_metadata_df():
    mdf = empty_metadata_df([1])
    assert "trial" in mdf.columns


def test_load_metadata_df_empty_fallback():
    mdf, source = load_metadata_df(trial_ids=[1, 2])
    assert source is None
    assert list(mdf["trial"]) == [1, 2]


def test_metadata_from_nwb_trials_drops_timing_keeps_filenames():
    trials_df = pd.DataFrame(
        {
            "trial": [1, 2],
            "start_time": [0.0, 1.0],
            "stop_time": [0.5, 1.5],
            "video_cam-1": ["a.mp4", "b.mp4"],
            "genotype": ["WT", "KO"],
        }
    )
    mdf = metadata_from_nwb_trials(trials_df)
    assert list(mdf.columns) == ["trial", "genotype", "video_cam-1"]
    assert list(mdf["video_cam-1"]) == ["a.mp4", "b.mp4"]


def test_order_metadata_columns_puts_filenames_last():
    df = pd.DataFrame(
        {
            "audio_mic-1": ["a.wav", "b.wav"],
            "video_cam-1": ["a.mp4", "b.mp4"],
            "trial": [1, 2],
            "genotype": ["WT", "KO"],
        }
    )
    assert list(order_metadata_columns(df).columns) == [
        "trial",
        "genotype",
        "video_cam-1",
        "audio_mic-1",
    ]


class _FakeAlignment:
    """An alignment that names one video file per trial."""

    def devices(self, stream):
        return ["cam-1"] if stream == "video" else []

    def media_filename(self, trial, stream, device=None):
        return f"trial{trial}.mp4"


def test_attach_media_columns_from_alignment():
    df = pd.DataFrame({"trial": [1, 2], "genotype": ["WT", "KO"]})
    out = attach_media_columns(df, _FakeAlignment())
    assert list(out.columns) == ["trial", "genotype", "video_cam-1"]
    assert list(out["video_cam-1"]) == ["trial1.mp4", "trial2.mp4"]


def test_attach_media_columns_alignment_wins_over_stale_copy():
    df = pd.DataFrame({"trial": [1, 2], "video_cam-1": ["old1.mp4", "old2.mp4"]})
    out = attach_media_columns(df, _FakeAlignment())
    assert list(out["video_cam-1"]) == ["trial1.mp4", "trial2.mp4"]


def test_media_columns_are_not_editable_conditions():
    df = pd.DataFrame({"trial": [1], "genotype": ["WT"], "video_cam-1": ["a.mp4"]})
    assert condition_columns(df) == ["genotype"]


def test_metadata_from_intervalset():
    nap = pytest.importorskip("pynapple")

    intervals = nap.IntervalSet(
        start=[0.0, 1.0],
        end=[0.5, 1.5],
        metadata=pd.DataFrame(
            {
                "trial": [1, 2],
                "condition": ["A", "B"],
                "start_time": [0.0, 1.0],
            }
        ),
    )
    mdf = metadata_from_intervalset(intervals)
    assert list(mdf.columns) == ["trial", "condition"]
    assert list(mdf["condition"]) == ["A", "B"]


def test_load_metadata_df_uses_nwb_source_when_trials_present(monkeypatch, tmp_path):
    nwb_path = tmp_path / "session.nwb"
    nwb_path.write_text("placeholder", encoding="utf-8")

    class _FakeAlignment:
        def __init__(self):
            self.trials_df = pd.DataFrame(
                {
                    "trial": [1, 2],
                    "genotype": ["WT", "KO"],
                    "start_time": [0.0, 1.0],
                    "stop_time": [0.5, 1.5],
                }
            )

        def devices(self, stream):
            return []

    monkeypatch.setattr("ethograph.io.metadata_table.make_nwb_alignment", lambda _: _FakeAlignment())

    mdf, source = load_metadata_df(source_path=nwb_path)

    assert source == str(nwb_path)
    assert list(mdf.columns) == ["trial", "genotype"]
    assert list(mdf["trial"]) == [1, 2]


# ---------------------------------------------------------------------------
# Trials from a pynapple IntervalSet: explicit conversion to alignment.nwb
# ---------------------------------------------------------------------------


def _alignment_trials_df():
    return pd.DataFrame(
        {
            "start_time": [20.9, 33.6, 46.5],
            "stop_time": [25.5, 38.4, 50.7],
            "trial": [1, 2, 4],
            "video_cam-1": ["a.mp4", "b.mp4", "c.mp4"],
        }
    )


def test_trial_ids_from_alignment_trials_ep():
    """IntervalSet built from a trials table carries the real trial ids."""
    pytest.importorskip("pynapple")
    from ethograph.io.data_loader import _trial_ids_from_ep
    from ethograph.io.nwb_alignment import _build_trials_ep

    ep = _build_trials_ep(_alignment_trials_df())
    assert ep is not None
    assert _trial_ids_from_ep(ep) == [1, 2, 4]


def test_alignment_from_trials_ep_roundtrip(tmp_path):
    """A trials IntervalSet converts to an alignment.nwb whose trials table
    carries the timing AND the IntervalSet's metadata columns."""
    nap = pytest.importorskip("pynapple")
    from ethograph.io.nwb_alignment import NWBAlignment, alignment_from_trials_ep

    ep = nap.IntervalSet(
        start=[21.5, 47.0, 59.3],
        end=[24.7, 50.4, 62.3],
        metadata={"condition": ["right", "left", "right"], "region": ["AId", "AId", "MSt"]},
    )
    out = tmp_path / ".ethograph" / "alignment.nwb"
    alignment_from_trials_ep(ep, out)
    assert out.exists()

    alignment = NWBAlignment(out)
    df = alignment.trials_df
    assert list(df["trial"]) == [1, 2, 3]
    assert df["start_time"].iloc[0] == pytest.approx(21.5)
    assert df["stop_time"].iloc[-1] == pytest.approx(62.3)
    assert list(df["condition"]) == ["right", "left", "right"]
    assert list(df["region"]) == ["AId", "AId", "MSt"]

    trials_ep = alignment.trials_ep
    assert trials_ep is not None and len(trials_ep) == 3
    alignment.close()


def test_find_trials_intervalset(tmp_path):
    nap = pytest.importorskip("pynapple")
    import numpy as np

    from ethograph.io.pynapple import find_trials_intervalset

    # A non-IntervalSet npz must be skipped cheaply.
    np.savez(tmp_path / "speed.npz", t=np.arange(5.0), d=np.zeros(5), type="Tsd")
    assert find_trials_intervalset(tmp_path) is None

    ep = nap.IntervalSet(start=[1.0, 5.0], end=[2.0, 6.0])
    ep.save(str(tmp_path / "trials.npz"))
    found = find_trials_intervalset(tmp_path)
    assert found is not None and len(found) == 2
