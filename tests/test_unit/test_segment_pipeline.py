"""End-to-end tests of the segmentation pipeline on synthetic sessions.

Two xarray sessions, two individuals, three keypoints; one label class whose
frames are where keypoint speed is high. Covers: config loading + overrides,
materialisation layout, role assignment, a short training run, inference
writing a prediction set the GUI's TSV reader accepts, and the compare table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import ethograph as eto
from ethograph.labels.intervals import LABELING_AUTOMATED, LABELING_MANUAL
from ethograph.labels.tsv_store import load_labels_tsv, save_labels_tsv
from ethograph.segment.config import apply_overrides, load_config
from ethograph.segment.materialise import materialise, read_index, read_layout
from ethograph.segment.samples import class_table, dense_targets
from ethograph.utils.paths import session_id

torch = pytest.importorskip("torch")

FS = 50.0
DURATION = 8.0
INDIVIDUALS = ["A", "B"]
KEYPOINTS = ["beak", "tail", "wing"]


def _trial_ds(trial: int, seed: int) -> tuple[xr.Dataset, list[tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    pos = rng.normal(0, 0.2, size=(t.size, 2, len(KEYPOINTS), len(INDIVIDUALS)))
    bouts = [(1.0 + 0.3 * (trial % 3), 2.0 + 0.3 * (trial % 3)), (4.5, 5.5)]
    for on, off in bouts:
        m = (t >= on) & (t <= off)
        pos[m, :, :, 0] += np.sin(t[m, None, None] * 40)[:, :, :] * 3.0
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * FS
    ds = xr.Dataset(
        {
            "position": (("time", "space", "keypoint", "individual"), pos),
            "speed": (("time", "keypoint", "individual"), speed),
        },
        coords={"time": t, "space": ["x", "y"], "keypoint": KEYPOINTS, "individual": INDIVIDUALS},
        attrs={"trial": trial, "fps": FS},
    )
    return ds, bouts


def _make_session(folder: Path, name: str, trials: list[int], seed: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    datasets, rows = [], []
    for i, trial in enumerate(trials):
        ds, bouts = _trial_ds(trial, seed + i)
        datasets.append(ds)
        for on, off in bouts:
            rows.append(
                {
                    "trial": trial,
                    "individual": "A",
                    "individual_rec": "",
                    "labels": 3,
                    "onset_s": on,
                    "offset_s": off,
                    "event_type": "state",
                    "confidence": 1.0,
                    "labeling_method": LABELING_MANUAL,
                    "changepoint_corrected": 0,
                    "prediction_source": "",
                    "n_samples": int(DURATION * FS),
                }
            )
        # an automated label that must never become training data
        automated = {**rows[-1], "onset_s": 6.5, "offset_s": 7.0, "labeling_method": LABELING_AUTOMATED}
        rows.append({**automated, "confidence": 0.3})
    dt = eto.from_datasets(datasets)
    nc_path = folder / f"{name}.nc"
    dt.save(str(nc_path))
    save_labels_tsv(folder / f"{name}_labels.tsv", pd.DataFrame(rows))
    pd.DataFrame({"trial": trials, "condition": ["odd" if t % 2 else "even" for t in trials]}).to_csv(
        folder / f"{name}_metadata.tsv", sep="\t", index=False
    )
    return nc_path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "mapping.txt").write_text("0 background\n3 flap\n4 peck 1\n5 call 0 point\n", encoding="utf-8")
    s1 = _make_session(tmp_path / "sessions" / "s1", "s1", [1, 2, 3, 4], seed=0)
    s2 = _make_session(tmp_path / "sessions" / "s2", "s2", [1, 2], seed=10)
    config = {
        "sessions": [
            {"source": str(s1), "labels_path": str(s1.with_name("s1_labels.tsv"))},
            {"source": str(s2), "labels_path": str(s2.with_name("s2_labels.tsv"))},
        ],
        "features": {
            "name": "kin",
            "columns": {
                "position": {"space": ["x", "y"], "keypoint": ["beak", "tail"]},
                "speed": {"keypoint": ["beak"]},
            },
            "labels": {"mapping": "mapping.txt", "branch": 0},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [32]}},
        "train": {
            "epochs": 2,
            "eval_every": 1,
            "batch_size": 2,
            "run_name": "smoke",
            # 6 trials in all (s1: 4, s2: 2) -> 1 test, 2 val, 3 train
            "split": {"train_fraction": 0.5, "val_fraction": 0.3, "test_fraction": 0.2},
        },
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_loads_with_base_and_overrides(project: Path):
    (project / "child.yaml").write_text(
        yaml.safe_dump({"base": "config.yaml", "train": {"epochs": 7}}), encoding="utf-8"
    )
    cfg = load_config(project / "child.yaml", ["model.architecture=asformer", "train.loss.gamma=3"])
    assert cfg.train.epochs == 7
    assert cfg.model.architecture == "asformer"
    # train.loss is a plain override dict over DLC2Action's own losses.yaml
    assert cfg.train.loss == {"gamma": 3}
    assert cfg.root == project
    assert cfg.features.labels.mapping == project / "mapping.txt"
    assert cfg.data_dir == project / "data" / "kin"


def test_config_rejects_unknown_keys(project: Path):
    with pytest.raises(ValueError, match="unknown key"):
        load_config(project / "config.yaml", ["train.epoch=3"])


def test_missing_labels_path_defaults_and_is_never_created(project: Path):
    """A session named with no labels_path is fine for inference-only use: the
    path defaults to {stem}_labels.tsv beside source, but nothing ever creates
    that file — it is the user's curated labels, and a session with none yet
    just opens with an empty labels table, same as the GUI would treat it."""
    from ethograph.segment.config import config_from_dict
    from ethograph.segment.sessions import open_session

    s3 = _make_session(project.parent / "sessions" / "s3", "s3", [1], seed=20)
    s3_labels = s3.with_name("s3_labels.tsv")
    s3_labels.unlink()  # a session that was never curated has no labels file yet

    cfg = config_from_dict(
        {
            "sessions": [{"source": str(s3)}],
            "features": {"columns": {"speed": {}}, "labels": {"mapping": "mapping.txt"}},
        },
        project,
    )
    assert cfg.sessions[0].labels_path == s3_labels
    assert not s3_labels.exists()

    session = open_session(cfg.sessions[0], cfg)
    assert not s3_labels.exists()
    assert session.result.all_labels_df.empty


def test_top_level_individual_fills_features_individuals(project: Path):
    """`config.individual` is the single-animal spelling of `features.individuals`."""
    from ethograph.segment.config import config_from_dict

    cfg = config_from_dict(
        {
            "sessions": [{"source": "s.nc", "labels_path": "s_labels.tsv"}],
            "individual": "A",
            "features": {"columns": {"speed": {}}, "labels": {"mapping": "mapping.txt"}},
        },
        project,
    )
    assert cfg.features.individuals == ["A"]


def test_top_level_individual_conflicting_with_features_individuals_is_refused(project: Path):
    from ethograph.segment.config import config_from_dict

    with pytest.raises(ValueError, match="conflicts"):
        config_from_dict(
            {
                "sessions": [{"source": "s.nc", "labels_path": "s_labels.tsv"}],
                "individual": "A",
                "features": {
                    "columns": {"speed": {}},
                    "labels": {"mapping": "mapping.txt"},
                    "individuals": ["B"],
                },
            },
            project,
        )


def test_training_defaults_come_from_the_vendored_training_yaml(project: Path):
    """`epochs`, `learning_rate` and `weight_decay` are DLC2Action's, not ours.

    A config with no ``train`` block at all, so nothing masks the defaults.
    """
    from ethograph.segment.config import config_from_dict, upstream_training_defaults

    upstream = upstream_training_defaults()
    cfg = config_from_dict(
        {
            "sessions": [{"source": "s.nc", "labels_path": "s_labels.tsv"}],
            "features": {"columns": {"speed": {}}, "labels": {"mapping": "mapping.txt"}},
        },
        project,
    )
    assert cfg.train.epochs == int(upstream["num_epochs"])
    assert cfg.train.learning_rate == float(upstream["lr"])
    assert cfg.train.weight_decay == float(upstream["weight_decay"])
    # ours on purpose: a sample here is a whole trial, not a 128-frame window
    assert cfg.train.batch_size == 1
    # and so is the split — 60/20/20, the ratio a search is tuned against
    assert (cfg.train.split.train_fraction, cfg.train.split.val_fraction, cfg.train.split.test_fraction) == (
        0.6,
        0.2,
        0.2,
    )


def test_a_number_written_in_yaml_1_1_exponent_form_is_still_a_number(project: Path):
    """`learning_rate: 1e-4` has no dot and no sign, so YAML 1.1 reads it as a
    *string*. Unconverted it would reach the optimizer as text."""
    (project / "lr.yaml").write_text(
        yaml.safe_dump({"base": "config.yaml", "train": {"learning_rate": "1e-4", "epochs": "12"}}),
        encoding="utf-8",
    )
    cfg = load_config(project / "lr.yaml")
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    assert isinstance(cfg.train.learning_rate, float)
    assert cfg.train.epochs == 12
    assert isinstance(cfg.train.epochs, int)

    # and the override path, which is where a stray string is likeliest
    cfg = load_config(project / "config.yaml", ["train.learning_rate=5e-5"])
    assert cfg.train.learning_rate == pytest.approx(5e-5)


def test_a_whole_number_field_refuses_a_fraction(project: Path):
    with pytest.raises(ValueError, match="whole number"):
        load_config(project / "config.yaml", ["train.epochs=1.5"])


def test_a_number_field_refuses_nonsense(project: Path):
    with pytest.raises(ValueError, match="expected a number"):
        load_config(project / "config.yaml", ["train.learning_rate=fast"])


def test_a_key_written_with_no_value_keeps_its_default(project: Path):
    """``params:`` followed by only a comment is YAML null, and means "default"."""
    (project / "empty.yaml").write_text(
        """
base: config.yaml
model:
  architecture: mlp
  params:
  # upstream's defaults
train:
  loss:
""",
        encoding="utf-8",
    )
    cfg = load_config(project / "empty.yaml")
    assert cfg.model.params == {}
    assert cfg.train.loss == {}
    assert cfg.train.split.train_fraction == 0.5  # the base's, not clobbered


def test_apply_overrides_parses_yaml_values():
    out = apply_overrides({"a": {"b": 1}}, ["a.b=[1, 2]", "a.c=null", "d=true"])
    assert out == {"a": {"b": [1, 2], "c": None}, "d": True}


def test_apply_overrides_descends_into_an_empty_section():
    """``params:`` with nothing after it is YAML null, which reads as the default — an override may fill it."""
    out = apply_overrides({"model": {"architecture": "mlp", "params": None}}, ["model.params.f_maps_list=[8]"])
    assert out == {"model": {"architecture": "mlp", "params": {"f_maps_list": [8]}}}
    with pytest.raises(ValueError, match="not a mapping"):
        apply_overrides({"model": {"architecture": "mlp"}}, ["model.architecture.x=1"])


def test_class_table_is_one_branch_state_classes_only(project: Path):
    cfg = load_config(project / "config.yaml")
    classes = class_table(cfg)
    assert classes.label_ids == [0, 3]
    assert classes.names == ["background", "flap"]


# ---------------------------------------------------------------------------
# Materialise
# ---------------------------------------------------------------------------


def test_materialise_writes_literature_layout(project: Path):
    cfg = load_config(project / "config.yaml")
    data_dir = materialise(cfg)
    index = read_index(data_dir)
    assert len(index) == (4 + 2) * len(INDIVIDUALS)
    assert set(index["individual"]) == set(INDIVIDUALS)
    layout = read_layout(data_dir)
    # config order, first dim slowest (the lightgbm model's convention)
    assert layout.names == [
        "position|space=x,keypoint=beak,individual=self",
        "position|space=x,keypoint=tail,individual=self",
        "position|space=y,keypoint=beak,individual=self",
        "position|space=y,keypoint=tail,individual=self",
        "speed|keypoint=beak,individual=self",
    ]
    assert layout.vector_groups == [[0, 2], [1, 3]]
    assert layout.fs == pytest.approx(FS)
    key = index["key"][0]
    x = np.load(data_dir / "features" / f"{key}.npy")
    assert x.shape == (5, int(DURATION * FS))
    names = (data_dir / "groundTruth" / f"{key}.txt").read_text().splitlines()
    assert len(names) == x.shape[1]
    assert (data_dir / "mapping.txt").read_text() == "0 background\n1 flap\n"
    # individual B carries no labels → all background; A has two bouts
    assert index.loc[index["individual"] == "B", "n_labelled"].eq(0).all()
    assert index.loc[index["individual"] == "A", "n_labelled"].eq(2).all()


def test_dense_targets_ignore_other_individuals():
    from ethograph.segment.samples import ClassTable

    table = ClassTable([0, 3], ["background", "flap"])
    t = np.arange(0, 2, 0.1)
    df = pd.DataFrame(
        {
            "onset_s": [0.5, 1.0],
            "offset_s": [0.7, 1.5],
            "labels": [3, 3],
            "individual": ["A", "B"],
            "event_type": ["state", "state"],
        }
    )
    y, n = dense_targets(df, t, "A", table)
    assert n == 1
    assert y[5:8].tolist() == [1, 1, 1]
    assert y[8:].sum() == 0


def test_trials_filter_uses_metadata(project: Path):
    cfg = load_config(project / "config.yaml", ["trials.where={condition: [odd]}"])
    data_dir = materialise(cfg)
    index = read_index(data_dir)
    assert set(index["trial"]) == {1, 3}


# ---------------------------------------------------------------------------
# Train + infer + compare
# ---------------------------------------------------------------------------


def test_train_infer_compare_roundtrip(project: Path):
    from ethograph.segment.inference import inference
    from ethograph.segment.train import compare_runs, train

    cfg = load_config(project / "config.yaml", ["train.device=cpu"])
    result = train(cfg)
    run_dir = result.run_dir
    assert run_dir.parent == project / "runs"
    assert run_dir.name.startswith("smoke_")  # base run name + creation timestamp, never overwritten
    for name in ("config.yaml", "columns.yaml", "classes.yaml", "stats.npz", "best.pt", "last.pt", "metrics.tsv"):
        assert (run_dir / name).is_file(), name
    metrics = pd.read_csv(run_dir / "metrics.tsv", sep="\t")
    assert list(metrics["epoch"]) == [1, 2]
    assert {"f1@50", "acc", "edit", "frame_f1"} <= set(metrics.columns)
    bundles = {p.stem: p.read_text().split() for p in (run_dir / "splits").glob("*.bundle")}
    # the three fractions divide the 6 trials by whole trial (× 2 individuals = 12 samples)
    assert len(bundles["train"]) == 6 and len(bundles["val"]) == 4 and len(bundles["test"]) == 2
    assert set(bundles["train"]).isdisjoint(bundles["val"] + bundles["test"])
    assert (run_dir / "test_metrics.yaml").is_file()  # test_fraction > 0
    assert (run_dir / "test_eval.npz").is_file()
    test_metrics = yaml.safe_load((run_dir / "test_metrics.yaml").read_text(encoding="utf-8"))
    assert test_metrics["thresholds"] == list(cfg.train.f1_thresholds)
    assert {"tp", "fp", "fn"} <= set(test_metrics["raw"])

    # inference runs over every session of the config
    written = inference(cfg, run="smoke")
    assert len(written) == 2
    tsv = next(p for p in written if p.name == "s2_predictions.tsv")
    assert tsv.parent.parent == project.parent / "sessions" / "s2" / "labels"
    assert tsv.parent.name.startswith(f"predictions_{run_dir.name}_")
    # the folder says what wrote it: the run's own config, and how it was applied
    assert (tsv.parent / "config.yaml").read_text(encoding="utf-8") == (run_dir / "config.yaml").read_text(
        encoding="utf-8"
    )
    applied = yaml.safe_load((tsv.parent / "inference.yaml").read_text(encoding="utf-8"))
    assert applied["run"] == run_dir.name and "postprocess" in applied["infer"]
    df = load_labels_tsv(tsv)
    if not df.empty:
        assert set(df["labeling_method"]) == {LABELING_AUTOMATED}
        assert set(df["prediction_source"]) == {run_dir.name}
        assert set(df["labels"]) <= {3}
        assert (df["confidence"] <= 1.0).all()
    probs = np.load(tsv.with_name("s2_probs.npz"))
    keys = [k for k in probs.files if not k.endswith("_time")]
    assert len(keys) == 2 * len(INDIVIDUALS)
    assert probs[keys[0]].shape == (int(DURATION * FS), 2)

    # a second run holding a whole session out — one cross-validation fold
    cfg2 = load_config(
        project / "config.yaml",
        [
            "train.device=cpu",
            "train.run_name=smoke2",
            "train.split.holdout_sessions=['%s']" % str(cfg.sessions[1].source).replace("\\", "/"),
        ],
    )
    result2 = train(cfg2)
    assert (result2.run_dir / "test_metrics.yaml").is_file()
    assert (result2.run_dir / "eval.pdf").is_file()
    # every trial of the held-out session is test, whatever the fractions say
    fold = {p.stem: p.read_text().split() for p in (result2.run_dir / "splits").glob("*.bundle")}
    assert len(fold["test"]) == 2 * len(INDIVIDUALS)  # s2 has 2 trials
    assert all(k.startswith("s2-") for k in fold["test"])  # {session_id}_trial{n}_{individual}
    assert all(k.startswith("s1-") for k in fold["train"] + fold["val"])
    table = compare_runs(cfg.runs_dir)
    assert set(table["run"].str.split("_").str[0]) == {"smoke", "smoke2"}
    assert "postprocessed.f1@50" in table.columns
    # both runs wrote test_eval.npz, so compare_runs also drew the comparison figure
    assert (cfg.runs_dir / "compare.pdf").is_file()


# ---------------------------------------------------------------------------
# Eval arrays + cross-run comparison figure
# ---------------------------------------------------------------------------


def test_evaluate_reports_tp_fp_fn():
    from ethograph.segment.metrics import evaluate

    gt = {"a": np.array([0, 0, 1, 1, 1, 0, 0])}
    pred = {"a": np.array([0, 0, 1, 1, 1, 0, 0])}
    out = evaluate(gt, pred, [0.5], fs=10.0)
    assert out["tp"] == 1.0 and out["fp"] == 0.0 and out["fn"] == 0.0


def test_eval_arrays_roundtrip(tmp_path: Path):
    from ethograph.segment.metrics import evaluate, load_eval_arrays, save_eval_arrays

    gt = {"a": np.array([0, 1, 1, 0]), "b": np.array([0, 0, 1, 1])}
    pred = {"a": np.array([0, 1, 1, 0]), "b": np.array([0, 1, 1, 0])}
    raw = evaluate(gt, pred, [0.5], fs=10.0)
    path = tmp_path / "test_eval.npz"
    save_eval_arrays(path, raw, raw)
    arrays = load_eval_arrays(path)
    np.testing.assert_array_equal(arrays["raw_ious"], raw["ious"])
    np.testing.assert_array_equal(arrays["post_start_deltas_s"], raw["start_deltas_s"])


def _fake_run_eval(name: str, seed: int):
    from ethograph.segment.plotting import RunEval

    rng = np.random.default_rng(seed)
    classwise = {1: {"f1@50": float(rng.uniform(40, 90))}, 2: {"f1@50": float(rng.uniform(40, 90))}}
    stage = lambda: {  # noqa: E731
        "acc": float(rng.uniform(50, 95)),
        "edit": float(rng.uniform(50, 95)),
        "frame_f1": float(rng.uniform(50, 95)),
        "f1@50": float(rng.uniform(50, 95)),
        "tp": float(rng.integers(5, 20)),
        "fp": float(rng.integers(0, 5)),
        "fn": float(rng.integers(0, 5)),
        "classwise": classwise,
    }
    return RunEval(
        name=name,
        thresholds=[0.5],
        raw=stage(),
        processed=stage(),
        raw_ious=rng.uniform(0, 1, size=20),
        processed_ious=rng.uniform(0, 1, size=20),
        raw_deltas_s=rng.uniform(0, 0.5, size=20),
        processed_deltas_s=rng.uniform(0, 0.5, size=20),
    )


def test_write_comparison_pdf_needs_two_runs():
    from ethograph.segment.plotting import write_comparison_pdf
    from ethograph.segment.samples import ClassTable

    classes = ClassTable([0, 1, 2], ["background", "a", "b"])
    with pytest.raises(ValueError):
        write_comparison_pdf(Path("unused.pdf"), [_fake_run_eval("only", 0)], classes)


def test_write_model_report_pdf_single_run(tmp_path: Path):
    """The per-run report draws one run happily — the cross-run figure refuses to."""
    from ethograph.segment.plotting import write_model_report_pdf
    from ethograph.segment.samples import ClassTable

    classes = ClassTable([0, 1, 2], ["background", "a", "b"])
    path = write_model_report_pdf(tmp_path / "report.pdf", [_fake_run_eval("only", 0)], classes, stamp="stamp")
    assert path.is_file() and path.stat().st_size > 0


def test_write_comparison_pdf(tmp_path: Path):
    from ethograph.segment.plotting import write_comparison_pdf
    from ethograph.segment.samples import ClassTable

    classes = ClassTable([0, 1, 2], ["background", "a", "b"])
    evals = [_fake_run_eval(f"run{i}", i) for i in range(3)]
    path = write_comparison_pdf(tmp_path / "comparison.pdf", evals, classes, title="fake runs")
    assert path.is_file() and path.stat().st_size > 0


def test_write_factorial_pdf_uneven_grid(tmp_path: Path):
    """An interrupted bench leaves holes: a cell with one fold, an architecture one group never trained."""
    from ethograph.segment.plotting import FactorCell, write_factorial_pdf
    from ethograph.segment.samples import ClassTable

    classes = ClassTable([0, 1, 2], ["background", "a", "b"])
    cells = [
        FactorCell("g1", "mlp", "all", [_fake_run_eval("f1", 1), _fake_run_eval("f2", 2)]),
        FactorCell("g1", "mlp", "no_circle", [_fake_run_eval("f1", 3)]),
        FactorCell("g2", "mlp", "all", [_fake_run_eval("f1", 4), _fake_run_eval("f2", 5)]),
        FactorCell("g2", "mstcn", "no_circle", [_fake_run_eval("f1", 6), _fake_run_eval("f2", 7)]),
    ]
    path = write_factorial_pdf(tmp_path / "factorial.pdf", cells, classes, title="fake", stamp="stamp")
    assert path.is_file() and path.stat().st_size > 0
    with pytest.raises(ValueError, match="at least one fold"):
        write_factorial_pdf(tmp_path / "empty.pdf", [FactorCell("g1", "mlp", "all", [])], classes)


def test_layout_mismatch_is_an_error(project: Path):
    from ethograph.segment.inference import inference
    from ethograph.segment.train import train

    cfg = load_config(project / "config.yaml", ["train.device=cpu"])
    train(cfg)
    changed = load_config(project / "config.yaml", ["features.columns.speed.keypoint=[tail]"])
    with pytest.raises(ValueError, match="column layout differs"):
        inference(changed, run="smoke")


# ---------------------------------------------------------------------------
# The split: three ratios, drawn by whole trial
# ---------------------------------------------------------------------------


class TestSplit:
    """``train.split`` is three fractions summing to 1, cut by whole trial."""

    def test_a_session_may_no_longer_declare_a_role(self, project: Path):
        with pytest.raises(ValueError, match="no longer carry a 'role'"):
            load_config(project / "config.yaml", ["sessions=[{source: s.nc, labels_path: l.tsv, role: train}]"])

    def test_fractions_must_sum_to_one(self, project: Path):
        with pytest.raises(ValueError, match="must sum to 1"):
            load_config(project / "config.yaml", ["train.split.val_fraction=0.5"])

    def test_holdout_must_name_a_listed_session(self, project: Path):
        with pytest.raises(ValueError, match="config.sessions does not list"):
            load_config(project / "config.yaml", ["train.split.holdout_sessions=[nowhere.nc]"])

    def test_no_trial_is_in_two_roles_and_the_seed_pins_the_draw(self, project: Path):
        from ethograph.segment.materialise import materialise, read_index
        from ethograph.segment.train import assign_roles

        cfg = load_config(project / "config.yaml")
        index = read_index(materialise(cfg))
        roles = assign_roles(cfg, index)
        by_trial: dict[tuple, set[str]] = {}
        for row in index.itertuples():
            by_trial.setdefault((row.source, row.trial), set()).add(roles[row.key])
        assert all(len(r) == 1 for r in by_trial.values()), "a trial was split across roles"
        assert roles == assign_roles(cfg, index)
        redrawn = assign_roles(load_config(project / "config.yaml", ["train.split.seed=7"]), index)
        assert redrawn != roles


# ---------------------------------------------------------------------------
# Stage 1: hyperparameter search.  Stage 2: cross-validation
# ---------------------------------------------------------------------------


class TestSearchAndCrossValidation:
    def test_search_writes_a_config_that_inherits_the_one_it_searched(self, project: Path):
        pytest.importorskip("optuna")
        from ethograph.segment.search import search

        cfg = load_config(
            project / "config.yaml",
            [
                "train.device=cpu",
                "search.n_trials=2",
                "search.prune=false",
                "search.params={train.learning_rate: {type: float, low: 1.0e-4, high: 1.0e-2, log: true}}",
            ],
        )
        result = search(cfg)
        assert set(result.best_params) == {"train.learning_rate"}
        assert len(result.trials) == 2
        assert (result.search_dir / "trials.tsv").is_file()
        assert result.best_run_dir.is_dir()
        # both trials finished with test_fraction > 0, so the comparison figure was drawn
        assert (result.search_dir / "eval_comparison.pdf").is_file()

        # best.yaml is a config: `base:` the one we searched, plus the winner
        best = load_config(result.config_path)
        assert best.train.learning_rate == pytest.approx(result.best_params["train.learning_rate"])
        assert best.model.architecture == cfg.model.architecture  # inherited
        # only the winning trial keeps its weights (search.keep_weights is off)
        losers = [d for d in result.best_run_dir.parent.iterdir() if d != result.best_run_dir]
        assert losers and not any((d / "best.pt").exists() for d in losers)

    def test_search_refuses_without_validation_trials(self, project: Path):
        pytest.importorskip("optuna")
        from ethograph.segment.search import search

        cfg = load_config(
            project / "config.yaml",
            [
                "train.split.train_fraction=0.8",
                "train.split.val_fraction=0.0",
                "train.split.test_fraction=0.2",
                "search.params={train.learning_rate: {type: float, low: 1.0e-4, high: 1.0e-2}}",
            ],
        )
        with pytest.raises(ValueError, match="scored on the validation trials"):
            search(cfg)

    def test_cross_validation_predicts_the_session_it_held_out(self, project: Path):
        from ethograph.labels.tsv_store import load_labels_tsv
        from ethograph.segment.crossval import cross_validate

        cfg = load_config(project / "config.yaml", ["train.device=cpu"])
        table = cross_validate(cfg, folds=["s2"])
        # the session *id*, not its stem: a project's sessions are routinely
        # files of the same name in different session folders
        assert [str(v).split("-")[0] for v in table["session"]] == ["s2"]
        assert list(table["session"]) == [session_id(cfg.sessions[1].source)]

        prediction = Path(table["predictions"].iloc[0])
        assert prediction.name == "s2_predictions.tsv"
        assert prediction.parent.parent.parent == project.parent / "sessions" / "s2"
        assert set(load_labels_tsv(prediction)["labeling_method"]) <= {LABELING_AUTOMATED}

        # the fold's own run, nested under runs/{cv name}/ so it does not bury the hand-trained runs
        run_dir = Path(table["run_dir"].iloc[0])
        assert run_dir.parent.parent == cfg.runs_dir and run_dir.parent.name.startswith("cv_")
        assert run_dir.name.startswith(f"fold-{session_id(cfg.sessions[1].source)}_")
        bundles = {p.stem: p.read_text().split() for p in (run_dir / "splits").glob("*.bundle")}
        assert all(k.startswith("s2-") for k in bundles["test"])
        assert all(k.startswith("s1-") for k in bundles["train"])
        assert bundles["val"] == []  # val_fraction defaults to 0 — the parameters are settled

    def test_cross_validation_refuses_a_config_that_already_pins_a_fold(self, project: Path):
        from ethograph.segment.crossval import cross_validate

        source = str(load_config(project / "config.yaml").sessions[1].source).replace("\\", "/")
        cfg = load_config(project / "config.yaml", ["train.split.holdout_sessions=['%s']" % source])
        with pytest.raises(ValueError, match="already set"):
            cross_validate(cfg)


# ---------------------------------------------------------------------------
# Ablation by feature kind
# ---------------------------------------------------------------------------


def _kinded_project(project: Path) -> Path:
    """Re-save the sessions with `kind` on their variables, and add a video feature."""
    from ethograph.io.schema import KINEMATIC_FEATURE, VIDEO_FEATURE, describe

    for source in sorted((project.parent / "sessions").rglob("*.nc")):
        dt = eto.open(str(source))
        for trial in dt.trials:
            ds = dt.trial(trial).copy()
            describe(ds["position"], KINEMATIC_FEATURE, is_egocentric=False)
            describe(ds["speed"], KINEMATIC_FEATURE, is_egocentric=False)
            n = ds.sizes["time"]
            ds["vid"] = (("time", "vid_dims"), np.zeros((n, 3), dtype=np.float32))
            describe(ds["vid"], VIDEO_FEATURE, is_egocentric=False)
            dt.update_trial(trial, lambda _ds, new=ds: new)
        dt.save(str(source))
    return project


def test_layout_records_each_column_kind(project: Path):
    from ethograph.io.schema import KINEMATIC_FEATURE, VIDEO_FEATURE

    _kinded_project(project)
    cfg = load_config(
        project / "config.yaml",
        ["features.columns.vid={vid_dims: ['0', '1', '2']}"],
    )
    layout = read_layout(materialise(cfg))
    by_name = dict(zip(layout.names, layout.kinds))
    assert by_name["speed|keypoint=beak,individual=self"] == KINEMATIC_FEATURE
    # `vid` carries no individual dim, so it gets no `self` token.
    assert by_name["vid|vid_dims=0"] == VIDEO_FEATURE
    assert layout.keep_mask([VIDEO_FEATURE]).sum() == layout.n_features - 3


def test_undeclared_columns_are_never_dropped(project: Path):
    """The advisory rule: no `kind` means the ablation leaves it alone."""
    cfg = load_config(project / "config.yaml")
    layout = read_layout(materialise(cfg))
    assert all(k is None for k in layout.kinds)
    assert layout.keep_mask(["video_feature"]).all()


def test_drop_kinds_trains_a_narrower_model(project: Path):
    """`drop_kinds` ablates at train time — same materialised dataset, fewer columns."""
    import torch

    from ethograph.io.schema import VIDEO_FEATURE
    from ethograph.segment.inference import load_run
    from ethograph.segment.train import train

    _kinded_project(project)
    base = ["train.device=cpu", "features.columns.vid={vid_dims: ['0', '1', '2']}"]
    full = train(load_config(project / "config.yaml", [*base, "train.run_name=full"]))
    ablated = train(
        load_config(project / "config.yaml", [*base, "train.run_name=no_video", f"train.drop_kinds=[{VIDEO_FEATURE}]"])
    )
    # One materialisation served both runs.
    assert len(list((project / "data").iterdir())) == 1

    def _input_width(run_dir: Path) -> int:
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        return next(t for t in state.values() if t.ndim >= 2).shape[1]

    assert _input_width(full.run_dir) - _input_width(ablated.run_dir) == 3
    # The run reloads with the same ablation it was trained with.
    assert load_run(ablated.run_dir).keep.sum() == _input_width(ablated.run_dir)
    assert load_run(full.run_dir).keep is None


def test_dropping_every_column_is_refused(project: Path):
    from ethograph.segment.train import train

    _kinded_project(project)
    cfg = load_config(
        project / "config.yaml",
        ["train.device=cpu", "train.drop_kinds=[kinematic_feature]", "train.run_name=empty"],
    )
    with pytest.raises(ValueError, match="drops every column"):
        train(cfg)


# ---------------------------------------------------------------------------
# The Project object — the one way to drive the pipeline
# ---------------------------------------------------------------------------


def test_project_runs_every_stage(project: Path):
    from ethograph.segment import Project

    p = Project(project / "config.yaml", "train.device=cpu")
    assert p.root == project
    assert "smoke" in repr(p) or "config.yaml" in repr(p)
    assert p.runs() == []

    assert p.materialise() == project / "data" / "kin"
    result = p.train()
    assert result.run_dir.parent == project / "runs"
    assert result.run_dir.name.startswith("smoke_")
    assert p.runs() == [result.run_dir.name]

    written = p.inference()
    assert len(written) == 2
    assert all(path.name.endswith("_predictions.tsv") for path in written)


def test_project_update_accumulates_overrides(project: Path):
    from ethograph.segment import Project

    p = Project(project / "config.yaml")
    assert p.config.model.architecture == "mlp"
    p.update("model.architecture=mstcn").update("train.epochs=9")
    assert p.config.model.architecture == "mstcn"
    assert p.config.train.epochs == 9
    # A later override of the same key wins.
    p.update("train.epochs=3")
    assert p.config.train.epochs == 3


def test_project_rejects_a_bad_override_immediately(project: Path):
    from ethograph.segment import Project

    with pytest.raises(ValueError, match="unknown key"):
        Project(project / "config.yaml", "train.epoch=3")
    p = Project(project / "config.yaml")
    with pytest.raises(ValueError, match="unknown key"):
        p.update("model.architektur=mlp")


def test_project_ranks_video_features_from_the_materialised_columns(project: Path):
    """`kind="video_feature"` is what tells the video columns apart."""
    from ethograph.segment import Project

    _kinded_project(project)
    p = Project(project / "config.yaml", "features.columns.vid={vid_dims: ['0', '1', '2']}")
    p.materialise()
    ranking, names = p.rank_video_features()
    assert names == ["vid|vid_dims=0", "vid|vid_dims=1", "vid|vid_dims=2"]
    assert ranking.n_features == 3
    assert len(ranking.top(2)) == 2


def test_ranking_needs_declared_video_features(project: Path):
    from ethograph.segment import Project

    p = Project(project / "config.yaml")
    p.materialise()
    with pytest.raises(ValueError, match="video_feature"):
        p.rank_video_features()


class TestSubsample:
    """``train.subsample`` — the temporal-resolution axis.

    Run-level like ``drop_kinds``: one materialised dataset serves every rate,
    and everything downstream of the store reads the rate the store reports
    rather than the rate on disk.
    """

    def test_the_store_strides_and_says_so(self, project: Path):
        from ethograph.segment.dataset import MaterialisedStore

        cfg = load_config(project / "config.yaml")
        materialise(cfg)
        full = MaterialisedStore.open(cfg.data_dir)
        half = MaterialisedStore.open(cfg.data_dir, 2)
        assert half.layout.fs == full.layout.fs / 2

        key = full.keys[0]
        x, y = full.load(key)
        xh, yh = half.load(key)
        assert xh.shape == (x.shape[0], (x.shape[1] + 1) // 2)
        assert len(yh) == len(xh[0])
        assert np.array_equal(xh, x[:, ::2]) and np.array_equal(yh, y[::2])

    def test_a_stride_below_one_is_refused(self, project: Path):
        from ethograph.segment.dataset import MaterialisedStore

        cfg = load_config(project / "config.yaml")
        materialise(cfg)
        with pytest.raises(ValueError, match="frame stride"):
            MaterialisedStore.open(cfg.data_dir, 0)

    def test_a_run_trains_and_predicts_at_its_own_rate(self, project: Path):
        """The model sees half the frames; the labels still span the same seconds."""
        from ethograph.segment.inference import inference
        from ethograph.segment.train import train

        cfg = load_config(project / "config.yaml", ["train.device=cpu", "train.run_name=half", "train.subsample=2"])
        result = train(cfg)
        assert (result.run_dir / "test_metrics.yaml").is_file()

        written = inference(cfg, run=result.run_dir)
        tsv = next(p for p in written if p.name == "s2_predictions.tsv")
        probs = np.load(tsv.with_name("s2_probs.npz"))
        key = next(k for k in probs.files if not k.endswith("_time"))
        assert probs[key].shape[0] == int(DURATION * FS) // 2
        time = probs[f"{key}_time"]
        # half as many frames over the same trial, so the spacing doubled
        assert np.isclose(np.median(np.diff(time)), 2.0 / FS)
        df = load_labels_tsv(tsv)
        if not df.empty:
            assert (df["offset_s"] <= DURATION).all()
