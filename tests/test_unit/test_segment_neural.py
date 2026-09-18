"""Spike trains as segmentation input (``features.neural``).

A pynapple session's units are a ``TsGroup`` — event times, not a feature —
so the config spells how they are binned, the session applies it at open,
and materialise reads the unit columns off the result. These tests pin that
contract: the transform runs as written and ends in a frame; the columns are
resolved once, recorded, and read back by later stages; the alignment is
found one folder up from the spikes; and the wrong backend is refused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ethograph.io import schema
from ethograph.labels.intervals import LABELING_MANUAL
from ethograph.labels.tsv_store import save_labels_tsv

nap = pytest.importorskip("pynapple")
torch = pytest.importorskip("torch")

N_TRIALS = 5
DURATION = 4.0
UNITS = (7, 42, 105)
BIN_S = 0.02
FS = 1.0 / BIN_S
ACTOR = "A"


def _write_session(root: Path) -> Path:
    """``root/behav/pynapple/units.npz`` with the alignment in ``root/behav/.ethograph/``."""
    from ethograph.io.nwb_alignment import alignment_from_trials_ep

    behav = root / "behav"
    folder = behav / "pynapple"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    total = N_TRIALS * DURATION
    spikes = {
        uid: nap.Ts(t=np.sort(rng.uniform(0.0, total, size=int(total * rate)))) for uid, rate in zip(UNITS, (20, 5, 50))
    }
    nap.TsGroup(spikes).save(str(folder / "units.npz"))
    t = np.arange(0, total, BIN_S)
    nap.Tsd(t=t, d=np.abs(np.sin(t))).save(str(folder / "speed.npz"))

    starts = np.arange(N_TRIALS) * DURATION
    ep = nap.IntervalSet(start=starts, end=starts + DURATION - BIN_S)
    ep.set_info(trial=np.arange(1, N_TRIALS + 1))
    alignment_from_trials_ep(ep, behav / ".ethograph" / "alignment.nwb")

    rows = [
        {
            "trial": trial,
            "individual": ACTOR,
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
        for trial in range(1, N_TRIALS + 1)
        for on, off in ((0.5, 1.5), (2.5, 3.0))
    ]
    save_labels_tsv(behav / "session_labels.tsv", pd.DataFrame(rows))
    return folder / "units.npz"


def _write_config(root: Path, source: Path, **features) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    config = {
        "sessions": [{"source": str(source), "labels_path": str(source.parent.parent / "session_labels.tsv")}],
        "individual": ACTOR,
        "features": {
            "name": "rate",
            "neural": {
                "units": "units",
                "name": "rate",
                "transform": [f"x.count({BIN_S}) / {BIN_S}", "sliding_window(x, window_size=0.1)"],
            },
            "preprocess": {"clip_percentiles": None},
            "labels": {"mapping": "mapping.txt", "branch": 0},
            **features,
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [8]}},
        "train": {
            "epochs": 1,
            "eval_every": 1,
            "device": "cpu",
            "run_name": "rate",
            "split": {"train_fraction": 0.8, "val_fraction": 0.0, "test_fraction": 0.2},
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return _write_session(tmp_path / "session")


@pytest.fixture
def project(tmp_path: Path, source: Path):
    import ethograph as eto

    return eto.segment.Project(_write_config(tmp_path / "project", source))


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


class TestTransform:
    def test_steps_run_in_order_on_x(self):
        from ethograph.features.neural import transform_units

        group = _two_units()
        frame = transform_units(group, ["x.count(0.1)", "x * 10"])
        assert isinstance(frame, nap.TsdFrame)
        assert list(frame.columns) == [1, 2]
        assert frame.shape[0] > 1
        assert float(frame.values.max()) == 20.0  # two spikes in one 0.1 s bin, × 10

    def test_sliding_window_reads_the_bin_size_off_the_frame(self):
        from ethograph.features.neural import sliding_window

        t = np.arange(0, 1.0, 0.01)
        frame = nap.TsdFrame(t=t, d=np.zeros((t.size, 1)), columns=[1])
        frame[50, 0] = 1.0
        mean = sliding_window(frame, window_size=0.05)
        # a 5-bin boxcar mean spreads the spike over 5 bins at 1/5 each
        np.testing.assert_allclose(mean.values.sum(), 1.0)
        assert np.isclose(mean.values.max(), 0.2)
        total = sliding_window(frame, window_size=0.05, reduction="sum")
        assert np.isclose(total.values.max(), 1.0)
        assert sliding_window(frame, window_size=0.05, step_size=0.02).shape[0] == t.size // 2

    def test_sliding_window_refuses_spike_times(self):
        from ethograph.features.neural import sliding_window

        with pytest.raises(TypeError, match="count"):
            sliding_window(_two_units(), window_size=0.1)

    @pytest.mark.parametrize(
        "steps, match",
        [
            ([], "empty"),
            (["x.count(0.1"], "not a Python expression"),
            (["x.nonsense()"], r"\[0\].*failed"),
            (["x.count(0.1)", "x[:, 0]"], "TsdFrame"),
            (["x"], "TsdFrame"),
        ],
    )
    def test_a_bad_chain_names_its_step(self, steps, match):
        from ethograph.features.neural import transform_units

        with pytest.raises(ValueError, match=match):
            transform_units(_two_units(), steps)


def _two_units():
    """Two units over one second — the support is spelled, so a lone spike is not a zero-length epoch."""
    return nap.TsGroup(
        {1: nap.Ts(t=np.array([0.1, 0.3, 0.35])), 2: nap.Ts(t=np.array([0.6, 0.7]))},
        time_support=nap.IntervalSet(0.0, 1.0),
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_neural_alone_satisfies_the_columns_rule(self, project):
        assert project.config.features.columns == {}
        assert project.config.features.neural is not None
        assert project.config.features.neural.transform[0].startswith("x.count")

    def test_neural_needs_a_transform(self, tmp_path: Path, source: Path):
        import ethograph as eto

        path = _write_config(tmp_path / "project", source, neural={"units": "units", "name": "rate", "transform": []})
        with pytest.raises(ValueError, match="transform"):
            eto.segment.Project(path)

    def test_name_may_not_be_the_group_itself(self, tmp_path: Path, source: Path):
        import ethograph as eto

        path = _write_config(
            tmp_path / "project", source, neural={"units": "units", "name": "units", "transform": ["x.count(0.1)"]}
        )
        with pytest.raises(ValueError, match="pick another name"):
            eto.segment.Project(path)


# ---------------------------------------------------------------------------
# Opening a session
# ---------------------------------------------------------------------------


class TestSession:
    def test_alignment_one_folder_up_gives_the_trials(self, project):
        from ethograph.segment.sessions import open_session

        session = open_session(project.config.sessions[0], project.config)
        assert session.trial_ids == list(range(1, N_TRIALS + 1))

    def test_the_frame_becomes_a_declared_feature(self, project):
        from ethograph.segment.sessions import feature_sampling_rates, neural_columns, open_session

        session = open_session(project.config.sessions[0], project.config)
        assert "rate" in session.result.catalog.features
        assert session.variable_attrs("rate") == {schema.KIND: schema.NEURAL_FEATURE, schema.NORMALISE: 1}
        assert session.declares_schema()
        assert neural_columns(session, project.config.features.neural) == {"rate_columns": [str(u) for u in UNITS]}
        assert feature_sampling_rates(session)["rate"] == pytest.approx(FS, rel=1e-3)

    def test_expansion_is_idempotent(self, project):
        from ethograph.segment.sessions import expand_neural_features, open_session

        session = open_session(project.config.sessions[0], project.config)
        loader = session.result.data_loader
        expand_neural_features(session, project.config)
        assert session.result.data_loader is loader

    def test_a_missing_group_names_the_ones_there(self, tmp_path: Path, source: Path):
        import ethograph as eto
        from ethograph.segment.sessions import open_session

        path = _write_config(
            tmp_path / "project", source, neural={"units": "cells", "name": "rate", "transform": ["x.count(0.1)"]}
        )
        project = eto.segment.Project(path)
        with pytest.raises(ValueError, match=r"'cells'.*\['units'\]"):
            open_session(project.config.sessions[0], project.config)

    def test_a_name_the_session_uses_is_refused(self, tmp_path: Path, source: Path):
        import ethograph as eto
        from ethograph.segment.sessions import open_session

        path = _write_config(
            tmp_path / "project", source, neural={"units": "units", "name": "speed", "transform": ["x.count(0.1)"]}
        )
        project = eto.segment.Project(path)
        with pytest.raises(ValueError, match="already has a variable called 'speed'"):
            open_session(project.config.sessions[0], project.config)

    def test_an_xarray_session_is_refused(self, tmp_path: Path):
        import ethograph as eto
        from ethograph.segment.config import NeuralFeaturesConfig, config_from_dict
        from ethograph.segment.sessions import open_session

        nc = tmp_path / "Trial_data.nc"
        eto.from_datasets([_xarray_trial(1), _xarray_trial(2)]).to_netcdf(nc)
        (tmp_path / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
        config = config_from_dict(
            {
                "sessions": [str(nc)],
                "features": {"columns": {"speed": {}}, "labels": {"mapping": "mapping.txt"}},
            },
            tmp_path,
        )
        config.features.neural = NeuralFeaturesConfig(transform=["x.count(0.1)"])
        with pytest.raises(ValueError, match="no pynapple backend"):
            open_session(config.sessions[0], config)


def _xarray_trial(trial: int):
    import xarray as xr

    t = np.arange(0, 2.0, 0.1)
    ds = xr.Dataset(
        {"speed": (("individuals", "time"), np.abs(np.sin(t))[None, :])},
        coords={"individuals": [ACTOR], "time": t},
    )
    ds.attrs["trial"] = trial
    return ds


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


class TestStages:
    def test_materialise_resolves_and_records_the_unit_columns(self, project):
        from ethograph.segment.materialise import read_index, read_layout

        data_dir = project.materialise()
        layout = read_layout(data_dir)
        assert layout.names == [f"rate|rate_columns={u}" for u in UNITS]
        assert layout.kinds == [schema.NEURAL_FEATURE] * len(UNITS)
        assert all(layout.normalise)
        assert layout.fs == pytest.approx(FS, rel=1e-3)
        assert layout.neural_columns == {"rate": {"rate_columns": [str(u) for u in UNITS]}}
        index = read_index(data_dir)
        assert len(index) == N_TRIALS
        assert list(index["n_labelled"]) == [2] * N_TRIALS
        # the project's own config is untouched: the columns live in the dataset
        assert project.config.features.columns == {}

    def test_a_spelled_subset_is_kept(self, project):
        from ethograph.segment.materialise import read_layout

        project.update("features.columns.rate.rate_columns=[42, 105]")
        layout = read_layout(project.materialise())
        assert layout.names == ["rate|rate_columns=42", "rate|rate_columns=105"]

    def test_train_reads_the_columns_back_and_infers(self, project):
        from ethograph.labels.tsv_store import load_labels_tsv
        from ethograph.segment.config import load_config
        from ethograph.segment.materialise import read_index

        project.materialise()
        result = project.train()
        saved = load_config(result.run_dir / "config.yaml")
        assert saved.features.columns == {"rate": {"rate_columns": [str(u) for u in UNITS]}}
        assert saved.features.neural is not None and saved.features.neural.transform == (
            project.config.features.neural.transform
        )
        roles = (result.run_dir / "splits" / "val.bundle").read_text(encoding="utf-8").split()
        assert roles == []
        assert len(read_index(project.config.data_dir)) == N_TRIALS

        written = project.inference()
        df = load_labels_tsv(written[0])
        assert set(df["labeling_method"]) <= {"automated"}
        assert set(df["individual"]) <= {ACTOR}

    def test_inference_needs_only_the_run(self, project):
        """The run's config carries the unit columns, so a project with no data/ still predicts."""
        import shutil

        from ethograph.labels.tsv_store import load_labels_tsv

        project.materialise()
        project.train()
        shutil.rmtree(project.config.data_dir)
        written = project.inference()
        assert not load_labels_tsv(written[0]).empty

    def test_a_run_without_the_feature_is_refused(self, project):
        from ethograph.segment.config import NeuralFeaturesConfig, with_overrides
        from ethograph.segment.inference import inherit_neural_columns

        run_config = with_overrides(project.config, features=project.config.features)
        run_config.features.neural = None
        run_config.features.columns = {"speed": {}}
        asking = with_overrides(project.config, features=project.config.features)
        asking.features.neural = NeuralFeaturesConfig(transform=["x.count(0.1)"])
        asking.features.columns = {}
        with pytest.raises(ValueError, match="trained without it"):
            inherit_neural_columns(asking, run_config)

    def test_train_without_a_materialised_dataset_says_so(self, project):
        from ethograph.segment.materialise import resolved_config

        with pytest.raises(ValueError, match="materialise"):
            resolved_config(project.config)

    def test_a_dataset_without_neural_columns_is_refused(self, project, tmp_path: Path):
        from ethograph.segment.materialise import COLUMNS_FILE, read_layout, resolved_config

        data_dir = project.materialise()
        layout = read_layout(data_dir)
        layout.neural_columns = None
        (data_dir / COLUMNS_FILE).write_text(yaml.safe_dump(layout.to_dict(), sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="records no unit columns"):
            resolved_config(project.config)

    def test_the_ablation_axis_sees_the_kind(self, project):
        from ethograph.segment.materialise import read_layout

        layout = read_layout(project.materialise())
        assert not layout.keep_mask([schema.NEURAL_FEATURE]).any()


# ---------------------------------------------------------------------------
# Windows over an epoch with no trials (sleep)
# ---------------------------------------------------------------------------


class TestWindows:
    def test_tiling_covers_the_epoch_without_a_gap(self):
        from ethograph.segment.windows import STATE_COLUMN, tile_windows

        ep = tile_windows(10.0, 25.0, 6.0)
        np.testing.assert_allclose(ep.start, [10.0, 16.0, 22.0])
        np.testing.assert_allclose(ep.end, [16.0, 22.0, 25.0], atol=1e-3)  # the remainder stays: no gap
        assert list(ep.metadata["trial"]) == [1, 2, 3]
        assert set(ep.metadata[STATE_COLUMN]) == {"sleep"}

    def test_a_sub_second_remainder_is_dropped(self):
        from ethograph.segment.windows import tile_windows

        ep = tile_windows(0.0, 12.5, 6.0)
        np.testing.assert_allclose(ep.end, [6.0, 12.0], atol=1e-3)

    @pytest.mark.parametrize("args, match", [((0.0, 10.0, 0.0), "positive"), ((10.0, 10.0, 6.0), "after")])
    def test_bad_bounds_are_refused(self, args, match):
        from ethograph.segment.windows import tile_windows

        with pytest.raises(ValueError, match=match):
            tile_windows(*args)

    def test_an_alignment_of_windows_is_another_session_of_the_same_file(self, project, source: Path, tmp_path: Path):
        """Train on the wake trials, then predict windows tiled over the same recording."""
        import ethograph as eto
        from ethograph.segment.sessions import open_session
        from ethograph.segment.windows import write_windows_alignment

        project.materialise()
        result = project.train()

        alignment = source.parent.parent / ".ethograph" / "sleep_alignment.nwb"
        write_windows_alignment(alignment, 2.0, N_TRIALS * DURATION, 6.0)
        sleep_yaml = tmp_path / "project" / "sleep.yaml"
        sleep_yaml.write_text(
            yaml.safe_dump(
                {
                    "base": "config.yaml",
                    "sessions": [
                        {
                            "source": str(source),
                            "alignment": str(alignment),
                            "labels_path": str(tmp_path / "nowhere_labels.tsv"),
                            "name": "sleep",
                        }
                    ],
                    "trials": {"where": {"state": ["sleep"]}},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        sleep = eto.segment.Project(sleep_yaml)
        session = open_session(sleep.config.sessions[0], sleep.config)
        assert session.trial_ids == [1, 2, 3]  # 18 s / 6 s
        assert session.result.all_labels_df.empty

        written = sleep.inference(run=result.run_dir)
        predicted, window = _predicted_trials(written[0].with_name("units_probs.npz"))
        assert predicted == {"1", "2", "3"}
        assert window[0] == pytest.approx(0.0, abs=BIN_S) and window[-1] == pytest.approx(6.0, abs=2 * BIN_S)

    def test_another_recordings_units_are_refused_by_name(self, project, tmp_path: Path):
        """A decoder only runs on the recording it was trained on; the error says which units are missing."""
        import ethograph as eto
        from ethograph.segment.inference import inference

        project.materialise()
        result = project.train()
        other = _write_session(tmp_path / "other")
        # the other recording: same folder layout, different unit ids
        group = nap.load_file(str(other))
        renamed = nap.TsGroup({uid + 1000: group[uid] for uid in group.index}, time_support=group.time_support)
        renamed.save(str(other))
        elsewhere = eto.segment.Project(_write_config(tmp_path / "other_project", other))
        with pytest.raises(ValueError, match=r"missing \{'rate_columns': \['105', '42', '7'\]\}"):
            inference(elsewhere.config, run=result.run_dir)

    def test_a_missing_alignment_is_refused(self, project, source: Path, tmp_path: Path):
        from ethograph.segment.config import SessionSpec
        from ethograph.segment.sessions import open_session

        spec = SessionSpec(source=source, alignment=tmp_path / "missing.nwb", name="sleep")
        with pytest.raises(FileNotFoundError, match="missing.nwb"):
            open_session(spec, project.config)


# ---------------------------------------------------------------------------
# Cross-validation by trial
# ---------------------------------------------------------------------------


class TestTrialFolds:
    """One session cannot be held out, so the folds are groups of trials."""

    def test_dealing_is_a_partition_and_deterministic(self):
        from ethograph.segment.crossval import trial_folds

        folds = trial_folds(range(1, 11), 3, seed=0)
        assert len(folds) == 3
        assert sorted(sum(folds, []), key=int) == [str(i) for i in range(1, 11)]
        assert {len(f) for f in folds} <= {3, 4}
        assert folds == trial_folds(range(1, 11), 3, seed=0)
        assert folds != trial_folds(range(1, 11), 3, seed=1)

    @pytest.mark.parametrize("n_folds, match", [(1, "at least 2"), (6, "only 5 trials")])
    def test_dealing_refuses_a_bad_k(self, n_folds, match):
        from ethograph.segment.crossval import trial_folds

        with pytest.raises(ValueError, match=match):
            trial_folds(range(1, 6), n_folds, seed=0)

    def test_holdout_trials_become_test(self, project):
        from ethograph.segment.materialise import read_index
        from ethograph.segment.roles import assign_roles

        index = read_index(project.materialise())
        project.update(
            "train.split.holdout_trials=[2, 4]", "train.split.test_fraction=0", "train.split.train_fraction=1"
        )
        roles = assign_roles(project.config, index)
        by_trial = {str(index.set_index("key").loc[k, "trial"]): r for k, r in roles.items()}
        assert by_trial == {"1": "train", "2": "test", "3": "train", "4": "test", "5": "train"}

    def test_a_holdout_trial_nobody_has_is_refused(self, project):
        from ethograph.segment.materialise import read_index
        from ethograph.segment.roles import assign_roles

        index = read_index(project.materialise())
        project.update("train.split.holdout_trials=[99]")
        with pytest.raises(ValueError, match=r"\['99'\]"):
            assign_roles(project.config, index)

    def test_both_holdouts_are_refused(self, project, source: Path):
        with pytest.raises(ValueError, match="never both"):
            project.update(
                f"train.split.holdout_sessions=['{str(source).replace(chr(92), '/')}']",
                "train.split.holdout_trials=[1]",
            )

    def test_one_session_needs_n_folds(self, project):
        with pytest.raises(ValueError, match="n_folds=5"):
            project.cross_validate()

    def test_folds_and_n_folds_are_exclusive(self, project):
        with pytest.raises(ValueError, match="not both"):
            project.cross_validate(folds=["units"], n_folds=2)

    def test_every_trial_is_predicted_once_by_a_model_that_never_saw_it(self, project):
        from ethograph.labels.tsv_store import load_labels_tsv
        from ethograph.segment.config import load_config
        from ethograph.segment.crossval import FOLDS_FILE, MERGED_FILE, cross_validation_name_for

        table = project.cross_validate(n_folds=3)
        assert len(table) == 3
        out_dir = project.config.cross_validation_dir / cross_validation_name_for(project.config)
        assert (out_dir / FOLDS_FILE).is_file()

        held_out: list[str] = []
        for _, row in table.iterrows():
            run_dir = Path(row["run_dir"])
            fold_trials = row["trials"].split(",")
            saved = load_config(run_dir / "config.yaml")
            assert saved.train.split.holdout_trials == fold_trials
            test_keys = (run_dir / "splits" / "test.bundle").read_text(encoding="utf-8").split()
            assert sorted(k.rsplit("_trial", 1)[1].split("_")[0] for k in test_keys) == sorted(fold_trials)
            train_keys = (run_dir / "splits" / "train.bundle").read_text(encoding="utf-8").split()
            assert not {k.rsplit("_trial", 1)[1].split("_")[0] for k in train_keys} & set(fold_trials)
            # the fold's own prediction set covers only its held-out trials
            predicted, _ = _predicted_trials(Path(row["predictions"]).with_name("units_probs.npz"))
            assert predicted == set(fold_trials)
            held_out += fold_trials
        assert sorted(held_out, key=int) == [str(t) for t in range(1, N_TRIALS + 1)]

        merged = pd.read_csv(out_dir / MERGED_FILE, sep="\t")
        assert len(merged) == 1
        merged_tsv = Path(merged.loc[0, "predictions"])
        assert merged_tsv.name == "units_predictions.tsv"
        assert "predictions_cv_rate_" in merged_tsv.parent.name
        predicted, _ = _predicted_trials(merged_tsv.with_name("units_probs.npz"))
        assert predicted == {str(t) for t in range(1, N_TRIALS + 1)}
        df = load_labels_tsv(merged_tsv)
        assert set(df["prediction_source"]) <= {Path(r).name for r in table["run_dir"]}


def _predicted_trials(npz_path: Path) -> tuple[set[str], np.ndarray]:
    """The trial ids a ``_probs.npz`` holds predictions for, and the first sample's time axis."""
    with np.load(npz_path) as probs:
        keys = [k for k in probs.files if not k.endswith("_time")]
        time = probs[f"{keys[0]}_time"]
        return {k.rsplit("_trial", 1)[1].split("_")[0] for k in keys}, time
