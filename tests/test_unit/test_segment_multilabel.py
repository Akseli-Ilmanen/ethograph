"""Multi-label targets: several branches, several subjects, one model.

Two individuals; branch 0 (``flap``) and branch 1 (``peck``). A flaps and
pecks at once, B pecks while A flaps — so labels overlap across branches and
across animals, and never within one (subject, branch) track. Covers the
channel table and its tracks, the ``(C, T)`` targets, the materialised
layout, the collate/loss shapes, tie-breaking within a track, and a short
train + inference writing a prediction set the GUI reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import ethograph as eto
from ethograph.labels.intervals import LABELING_MANUAL
from ethograph.labels.tsv_store import load_labels_tsv, save_labels_tsv
from ethograph.segment.config import load_config
from ethograph.segment.dataset import PAD_TARGET, collate
from ethograph.segment.losses import build_objective
from ethograph.segment.materialise import load_sample, materialise, read_classes, read_index
from ethograph.segment.metrics import flatten_channels
from ethograph.segment.models import ModelOutput
from ethograph.segment.postprocess import postprocess_channels
from ethograph.segment.samples import (
    ChannelTable,
    ClassTable,
    channel_table,
    channels_to_track,
    dense_channel_targets,
    subject_tokens,
    target_table,
)

torch = pytest.importorskip("torch")

FS = 50.0
DURATION = 6.0
INDIVIDUALS = ["A", "B"]
FLAP, PECK = 3, 4


def _trial_ds(trial: int, seed: int) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    pos = rng.normal(0, 0.2, size=(t.size, 2, 2, len(INDIVIDUALS)))
    m = (t >= 1.0) & (t <= 2.0)
    pos[m, :, :, 0] += np.sin(t[m, None, None] * 40) * 3.0
    return xr.Dataset(
        {"position": (("time", "space", "keypoint", "individual"), pos)},
        coords={"time": t, "space": ["x", "y"], "keypoint": ["beak", "tail"], "individual": INDIVIDUALS},
        attrs={"trial": trial, "fps": FS},
    )


def _row(trial: int, individual: str, label: int, on: float, off: float) -> dict:
    return {
        "trial": trial,
        "individual": individual,
        "individual_rec": "",
        "labels": label,
        "onset_s": on,
        "offset_s": off,
        "event_type": "state",
        "confidence": 1.0,
        "labeling_method": LABELING_MANUAL,
        "changepoint_corrected": 0,
        "prediction_source": "",
        "n_samples": int(DURATION * FS),
    }


def _labels(trials: list[int]) -> pd.DataFrame:
    rows = []
    for trial in trials:
        rows.append(_row(trial, "A", FLAP, 1.0, 2.0))  # A flaps …
        rows.append(_row(trial, "A", PECK, 1.5, 2.5))  # … and pecks over the flap (another branch)
        rows.append(_row(trial, "B", PECK, 1.2, 1.8))  # B pecks while A flaps
    return pd.DataFrame(rows)


def _make_session(folder: Path, name: str, trials: list[int], seed: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    dt = eto.from_datasets([_trial_ds(t, seed + i) for i, t in enumerate(trials)])
    nc_path = folder / f"{name}.nc"
    dt.save(str(nc_path))
    save_labels_tsv(folder / f"{name}_labels.tsv", _labels(trials))
    return nc_path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "mapping.txt").write_text("0 background\n3 flap 0\n4 peck 1\n5 call 0 point\n", encoding="utf-8")
    s1 = _make_session(tmp_path / "sessions" / "s1", "s1", [1, 2, 3], seed=0)
    s2 = _make_session(tmp_path / "sessions" / "s2", "s2", [1, 2], seed=10)
    config = {
        "sessions": [{"source": str(s1)}, {"source": str(s2)}],
        "features": {
            "name": "kin",
            "columns": {"position": {"space": ["x", "y"], "keypoint": ["beak"]}},
            "labels": {"mapping": "mapping.txt", "branches": [0, 1], "subjects": "all"},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
        "train": {
            "epochs": 2,
            "eval_every": 1,
            "batch_size": 2,
            "run_name": "multi",
            "split": {"train_fraction": 0.6, "val_fraction": 0.2, "test_fraction": 0.2},
        },
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Table + targets
# ---------------------------------------------------------------------------


def test_channel_table_is_subject_by_class_and_tracks_are_subject_by_branch(project: Path):
    cfg = load_config(project / "config.yaml")
    table = channel_table(cfg, n_individuals=2)
    assert [c.name for c in table.channels] == ["self:flap", "self:peck", "other1:flap", "other1:peck"]
    assert table.n_outputs == 4 and table.n_classes == 5
    tracks = table.tracks()
    assert [(t.subject, t.branch) for t in tracks] == [("self", 0), ("self", 1), ("other1", 0), ("other1", 1)]
    assert tracks[0].classes.label_ids == [0, FLAP]
    assert [t.channels for t in table.tracks(subject="self")] == [(0,), (1,)]


def test_a_config_with_one_branch_stays_exclusive(project: Path):
    cfg = load_config(project / "config.yaml", ["features.labels.branches=null", "features.labels.subjects=self"])
    assert isinstance(target_table(cfg, 2), ClassTable)
    assert not cfg.features.labels.multilabel


def test_branch_and_branches_together_are_refused(project: Path):
    with pytest.raises(ValueError, match="both branch"):
        load_config(project / "config.yaml", ["features.labels.branch=1"])


def test_dense_channel_targets_overlap_across_branches_and_animals(project: Path):
    cfg = load_config(project / "config.yaml")
    table = channel_table(cfg, 2)
    time = np.arange(0.0, DURATION, 1.0 / FS)
    y, n = dense_channel_targets(_labels([1]), time, subject_tokens("A", ["B"]), table)
    assert y.shape == (4, time.size) and n == 3
    at = lambda s: int(np.searchsorted(time, s))  # noqa: E731
    assert y[0, at(1.6)] == 1 and y[1, at(1.6)] == 1, "A's flap and peck coexist — different branches"
    assert y[3, at(1.5)] == 1 and y[2].sum() == 0, "B's peck lands on other1:peck; B never flaps"
    assert y[0, at(2.2)] == 0 and y[1, at(2.2)] == 1


def test_channels_to_track_is_exclusive_and_breaks_ties_by_probability():
    table = ChannelTable.from_dict(
        {
            "target": "multilabel",
            "channels": [
                {"subject": "self", "label_id": 3, "branch": 0, "name": "self:a"},
                {"subject": "self", "label_id": 6, "branch": 0, "name": "self:b"},
                {"subject": "self", "label_id": 4, "branch": 1, "name": "self:c"},
            ],
        }
    )
    on = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
    probs = np.array([[0.6, 0.9, 0.1], [0.8, 0.2, 0.0], [0.7, 0.7, 0.7]])
    same_branch, other_branch = table.tracks()
    assert channels_to_track(on, probs, same_branch).tolist() == [2, 1, 0], "b wins frame 0 at 0.8 > 0.6"
    assert channels_to_track(on, probs, other_branch).tolist() == [1, 1, 1]
    assert channels_to_track(on, None, same_branch).tolist() == [1, 1, 0], "no probabilities: the first wins"


def test_postprocess_channels_never_leaves_two_channels_of_a_track_on(project: Path):
    from ethograph.segment.config import PostprocessConfig

    cfg = load_config(project / "config.yaml")
    table = channel_table(cfg, 2)
    on = np.zeros((4, 100), dtype=np.int64)
    on[0, 10:40] = 1  # self:flap
    on[1, 30:60] = 1  # self:peck — another branch, may overlap
    out = postprocess_channels(on, FS, table, PostprocessConfig(min_duration_s=0.0, stitch_gap_s=0.0))
    assert out[0, 10:40].all() and out[1, 30:60].all(), "cross-branch overlap survives"
    for track in table.tracks():
        assert (out[list(track.channels)].sum(axis=0) <= 1).all()


def test_flatten_channels_gives_each_channel_its_own_class():
    flat = flatten_channels({"k": np.array([[1, 0, 0], [0, 1, 1]])})
    assert flat["k#0"].tolist() == [1, 0, 0] and flat["k#1"].tolist() == [0, 2, 2]


# ---------------------------------------------------------------------------
# Materialise, batch, loss
# ---------------------------------------------------------------------------


def test_materialise_writes_channel_targets(project: Path):
    cfg = load_config(project / "config.yaml")
    data_dir = materialise(cfg)
    classes = read_classes(data_dir)
    assert isinstance(classes, ChannelTable)
    assert yaml.safe_load((data_dir / "classes.yaml").read_text(encoding="utf-8"))["target"] == "multilabel"
    assert (data_dir / "mapping.txt").read_text(encoding="utf-8").splitlines()[1] == "1 self:flap"
    index = read_index(data_dir)
    key = index.loc[index["session_id"].str.startswith("s1") & (index["individual"] == "A"), "key"].iloc[0]
    x, y = load_sample(data_dir, key, classes)
    assert y.shape == (4, x.shape[1]) and set(np.unique(y)) <= {0, 1}
    assert (data_dir / "groundTruth" / f"{key}.npy").is_file()
    assert int(index.loc[index["key"] == key, "n_labelled"].iloc[0]) == 3


def test_collate_pads_channel_targets():
    x1, y1 = torch.zeros(2, 5), torch.ones(3, 5, dtype=torch.long)
    x2, y2 = torch.zeros(2, 3), torch.zeros(3, 3, dtype=torch.long)
    x, y, mask, cand, keys = collate(
        [(x1, y1, torch.zeros(5, dtype=torch.bool), "a"), (x2, y2, torch.zeros(3, dtype=torch.bool), "b")]
    )
    assert y.shape == (2, 3, 5) and mask.shape == (2, 1, 5)
    assert (y[1, :, 3:] == PAD_TARGET).all() and (y[0] == 1).all()


def test_loss_follows_the_target(project: Path):
    cfg = load_config(project / "config.yaml")
    objective, settings = build_objective(cfg, 4, exclusive=False)
    assert settings["frame"]["exclusive"] is False
    logits = torch.randn(2, 3, 4, 20, requires_grad=True)  # (S, B, C, T)
    y = torch.randint(0, 2, (3, 4, 20))
    y[:, :, 15:] = PAD_TARGET
    mask = torch.ones(3, 1, 20)
    mask[:, :, 15:] = 0
    total, parts = objective(ModelOutput(logits=logits), y, torch.zeros(3, 20, dtype=torch.bool))
    assert torch.isfinite(total) and "frame" in parts
    total.backward()
    assert torch.isfinite(logits.grad).all()


def test_spelling_exclusive_against_the_target_is_refused(project: Path):
    cfg = load_config(project / "config.yaml", ["train.loss.exclusive=true"])
    with pytest.raises(ValueError, match="contradicts the target"):
        build_objective(cfg, 4, exclusive=False)


# ---------------------------------------------------------------------------
# Train + infer
# ---------------------------------------------------------------------------


def test_train_and_infer_write_one_track_per_branch(project: Path):
    from ethograph.segment.project import Project

    proj = Project(project / "config.yaml")
    result = proj.train()
    metrics = yaml.safe_load((result.run_dir / "test_metrics.yaml").read_text(encoding="utf-8"))
    assert set(metrics["raw"]["classwise"]) <= {1, 2, 3, 4}, "one class per channel in the flattened space"

    written = proj.inference(sessions=["s1"])
    df = load_labels_tsv(written[0])
    assert set(df["labels"].unique()) <= {FLAP, PECK}
    assert set(df["individual"].unique()) <= set(INDIVIDUALS), "every animal's labels come from its own sample"
    branch_of = {FLAP: 0, PECK: 1}
    for (_trial, individual, _branch), rows in df.assign(branch=df["labels"].map(branch_of)).groupby(
        ["trial", "individual", "branch"]
    ):
        rows = rows.sort_values("onset_s")
        assert (rows["onset_s"].to_numpy()[1:] >= rows["offset_s"].to_numpy()[:-1]).all(), (
            f"{individual}: two labels of one branch overlap"
        )
    probs = np.load(written[0].with_name("s1_probs.npz"))
    key = next(k for k in probs.files if not k.endswith("_time"))
    assert probs[key].shape[1] == 4
