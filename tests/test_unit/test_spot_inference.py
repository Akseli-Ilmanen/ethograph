"""Choosing the epoch and staging it for upstream's CLI.

The epoch choice is the one judgement inference makes, and it must be the
ladder's (misses first, then hits), not ``val_mAP``'s; the staging exists
because ``test_e2e.py`` loads whichever checkpoint is *last* in the folder.
"""

from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import pytest

from ethograph.spot.config import config_from_dict
from ethograph.spot.inference import (
    best_epoch,
    prediction_individual,
    resolve_run_dir,
    run_clip,
    run_config_file,
    stage_checkpoint,
)


def _config(tmp_path):
    source = tmp_path / "ses-01.nc"
    source.touch()
    return config_from_dict({"sessions": [str(source)], "labels": {"classes": [31]}, "root": str(tmp_path)}, tmp_path)


def _run(tmp_path, stride=2, epochs=(0, 1, 2)):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"stride": stride, "clip_len": 100, "dilate_len": 1}))
    for e in epochs:
        (run_dir / f"checkpoint_{e:03d}.pt").write_bytes(b"x")
    return run_dir


def _val(config, events):
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    truth = [
        {"video": "v", "fps": 200.0, "num_frames": 1000, "events": [{"label": "label_31", "frame": f} for f in events]}
    ]
    (config.dataset_dir / "val.json").write_text(json.dumps(truth))


def _pred(run_dir, epoch, frame, score=0.9):
    events = [] if frame is None else [{"label": "label_31", "frame": frame, "score": score}]
    with gzip.open(run_dir / f"pred-val.{epoch}.recall.json.gz", "wt") as fh:
        json.dump([{"video": "v", "fps": 100.0, "num_frames": 500, "events": events}], fh)


class TestRunConfigFile:
    """The config copied beside a run's predictions: ours when the run has one, else upstream's."""

    def test_own_then_upstreams(self, tmp_path):
        run_dir = _run(tmp_path)
        assert run_config_file(run_dir) == run_dir / "config.json"  # a run trained before configs were written
        (run_dir / "config.yaml").write_text("x: 1\n")
        assert run_config_file(run_dir) == run_dir / "config.yaml"


class TestToLabelsFrame:
    """`config.individual` is stamped into every exported row's `individual` column."""

    def _event(self):
        from ethograph.spot.confidence import CurveStats
        from ethograph.spot.prediction_sets import SpottedEvent

        return SpottedEvent(video_id="v", label=31, frame=10.0, video_s=0.1, stats=CurveStats(0, 0.9, 0.9, 0.9))

    def test_configured_individual_is_stamped_on_every_row(self):
        from ethograph.spot.prediction_sets import to_labels_frame

        df = to_labels_frame([self._event()], {"v": (1, 0.0)}, source="s", individual="A")
        assert df["individual"].tolist() == ["A"]


class TestBestEpoch:
    def test_fewest_misses_wins_over_a_closer_hit(self, tmp_path):
        config = _config(tmp_path)
        run_dir = _run(tmp_path)
        _val(config, events=[400])
        _pred(run_dir, 0, frame=None)  # a miss
        _pred(run_dir, 1, frame=199)  # bin 199 at stride 2 -> frame 398.5, within 4
        _pred(run_dir, 2, frame=None)
        assert best_epoch(run_dir, config) == 1

    def test_reads_predictions_back_on_the_full_rate_clock(self, tmp_path):
        """A strided bin is a hit only once it is mapped back to the video's frames."""
        config = _config(tmp_path)
        run_dir = _run(tmp_path, stride=4)
        _val(config, events=[400])
        _pred(run_dir, 0, frame=100)  # bin 100 at stride 4 -> centre 401.5: a hit
        _pred(run_dir, 1, frame=110)  # -> 441.5: a miss by the tolerance
        assert best_epoch(run_dir, config) == 0

    def test_no_val_predictions_falls_back_to_the_last_checkpoint(self, tmp_path):
        config = _config(tmp_path)
        run_dir = _run(tmp_path, epochs=(0, 3, 7))
        assert best_epoch(run_dir, config) == 7


class TestStaging:
    def test_only_the_chosen_epoch_is_visible_as_the_last_one(self, tmp_path):
        run_dir = _run(tmp_path)
        staged = stage_checkpoint(run_dir, 1, tmp_path / "staging")
        assert sorted(p.name for p in staged.iterdir()) == ["checkpoint_000.pt", "config.json", "optim_000.pt"]

    def test_a_missing_epoch_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="epoch 9"):
            stage_checkpoint(_run(tmp_path), 9, tmp_path / "staging")


class TestResolveRun:
    def test_by_name_by_path_and_newest(self, tmp_path):
        config = _config(tmp_path)
        run_dir = _run(tmp_path)
        assert resolve_run_dir(config, "r") == run_dir
        assert resolve_run_dir(config, run_dir) == run_dir.resolve()
        assert resolve_run_dir(config, None) == run_dir

    def test_unknown_run_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope"):
            resolve_run_dir(_config(tmp_path), "nope")

    def test_run_clip_reads_the_stored_stride(self, tmp_path):
        clip = run_clip(_run(tmp_path, stride=4), fps=200.0)
        assert (clip.stride, clip.clip_len, clip.fps) == (4, 100, 200.0)


class TestValTruth:
    def test_a_run_trained_elsewhere_brings_its_own_val(self, tmp_path):
        """A run's config.json names its dataset; an absolute one is honoured."""
        config = _config(tmp_path)
        elsewhere = tmp_path / "old_data"
        elsewhere.mkdir()
        run_dir = _run(tmp_path)
        stored = json.loads((run_dir / "config.json").read_text())
        stored["dataset"] = str(elsewhere)
        (run_dir / "config.json").write_text(json.dumps(stored))
        truth = [{"video": "v", "fps": 200.0, "num_frames": 1000, "events": [{"label": "label_31", "frame": 400}]}]
        (elsewhere / "val.json").write_text(json.dumps(truth))
        _pred(run_dir, 0, frame=None)
        _pred(run_dir, 1, frame=199)
        _pred(run_dir, 2, frame=None)
        assert best_epoch(run_dir, config) == 1  # not the last checkpoint


class TestCurveLength:
    def test_every_class_spans_the_whole_trial(self, tmp_path):
        """Upstream's recall entries state no length; the trial's own is used."""
        from ethograph.spot.prediction_sets import spot_entry

        config = config_from_dict(
            {"sessions": [str(tmp_path / "s.nc")], "labels": {"classes": [31, 32]}, "root": str(tmp_path)}, tmp_path
        )
        clip = run_clip(_run(tmp_path, stride=2), fps=200.0)
        entry = {
            "video": "v",
            "fps": 100.0,
            "events": [
                {"label": "label_31", "frame": 50, "score": 0.9},
                {"label": "label_32", "frame": 300, "score": 0.8},
            ],
        }
        _, curves = spot_entry(entry, config, clip, num_frames=1001)
        assert {k: v.shape for k, v in curves.items()} == {31: (500,), 32: (500,)}
        _, bare = spot_entry(entry, config, clip)
        assert bare[31].shape == (51,)  # without a length there is nothing better than the last candidate


class TestPredictionIndividual:
    """A predicted row always names an individual the session itself uses."""

    def _session(self, named):
        return SimpleNamespace(label_individuals=lambda: named, spec=SimpleNamespace(label="ses-01"))

    def test_config_wins(self):
        assert prediction_individual(SimpleNamespace(individual="A"), self._session(["B", "C"])) == "A"

    def test_unset_reads_the_sessions_one_individual(self):
        assert prediction_individual(SimpleNamespace(individual=None), self._session(["crow"])) == "crow"

    @pytest.mark.parametrize("named", [[], ["A", "B"]])
    def test_unset_and_ambiguous_is_refused(self, named):
        with pytest.raises(ValueError, match="individual"):
            prediction_individual(SimpleNamespace(individual=None), self._session(named))

    def test_exported_rows_load_as_a_labels_tsv(self, tmp_path):
        from ethograph.labels.tsv_store import load_labels_tsv, save_labels_tsv
        from ethograph.spot.prediction_sets import to_labels_frame

        event = TestToLabelsFrame()._event()
        path = tmp_path / "ses_predictions.tsv"
        save_labels_tsv(path, to_labels_frame([event], {"v": (1, 0.0)}, source="s", individual="crow"))
        assert load_labels_tsv(path)["individual"].tolist() == ["crow"]
