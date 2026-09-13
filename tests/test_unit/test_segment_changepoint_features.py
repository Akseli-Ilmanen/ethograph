"""``features.changepoint_features`` — expanding named raw changepoint masks
at ``open_session`` time so ``more_changepoint_features``'s proximity/
offset/length columns are auto-merged into ``features.columns`` without the
user spelling out every derived name, and the temporal scales are read off
the labels at ``materialise`` unless spelled. Xarray-only; pynapple sessions
must refuse it clearly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

import ethograph as eto
from ethograph.io import schema
from ethograph.labels.tsv_store import save_labels_tsv
from ethograph.segment.config import load_config
from ethograph.segment.materialise import materialise, read_index, read_layout
from ethograph.segment.sessions import open_session

FS = 50.0
DURATION = 4.0


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trial",
            "individual",
            "individual_rec",
            "labels",
            "onset_s",
            "offset_s",
            "event_type",
            "confidence",
            "labeling_method",
            "changepoint_corrected",
            "prediction_source",
            "n_samples",
        ]
    )


def _session_with_changepoints(folder: Path, *, legacy: bool = False) -> Path:
    """One trial with ``speed`` + its ``speed_troughs`` mask.

    ``legacy=True`` writes the pre-schema ``attrs["type"] = "changepoints"``
    spelling most real session files still carry.
    """
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    speed = np.abs(np.sin(t * 2.0))[:, None]
    mask = np.zeros(t.size, dtype=np.int8)[:, None]
    mask[[40, 120, 160], 0] = 1
    ds = xr.Dataset(
        {"speed": (("time", "individual"), speed), "speed_troughs": (("time", "individual"), mask)},
        coords={"time": t, "individual": ["A"]},
        attrs={"trial": 1, "fps": FS},
    )
    ds["speed_troughs"].attrs = (
        {"type": "changepoints", "target_feature": "speed"}
        if legacy
        else schema.changepoint_attrs(target_feature="speed")
    )
    dt = eto.from_datasets([ds])
    nc_path = folder / "cp.nc"
    dt.save(str(nc_path))
    save_labels_tsv(folder / "cp_labels.tsv", _empty_labels())
    return nc_path


def _session_with_two_changepoint_masks(folder: Path) -> Path:
    """``speed`` plus two masks over it, firing on different frames."""
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    speed = np.abs(np.sin(t * 2.0))[:, None]
    troughs = np.zeros(t.size, dtype=np.int8)[:, None]
    troughs[[40, 160], 0] = 1
    peaks = np.zeros(t.size, dtype=np.int8)[:, None]
    peaks[[90, 120], 0] = 1
    ds = xr.Dataset(
        {
            "speed": (("time", "individual"), speed),
            "speed_troughs": (("time", "individual"), troughs),
            "speed_peaks": (("time", "individual"), peaks),
        },
        coords={"time": t, "individual": ["A"]},
        attrs={"trial": 1, "fps": FS},
    )
    for var in ("speed_troughs", "speed_peaks"):
        ds[var].attrs = schema.changepoint_attrs(target_feature="speed")
    dt = eto.from_datasets([ds])
    nc_path = folder / "cp2.nc"
    dt.save(str(nc_path))
    save_labels_tsv(folder / "cp_labels.tsv", _empty_labels())  # the name _write_config expects
    return nc_path


def _write_config(
    root: Path,
    nc_path: Path,
    *,
    columns: dict | None = None,
    changepoint_features: dict | None = None,
) -> Path:
    (root / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    cp_cfg = (
        {"sigmas": [2.0], "inputs": {"speed_troughs": {}}} if changepoint_features is None else changepoint_features
    )
    # The fixtures carry no labels, so the scales cannot be derived: pin them
    # unless a test spells its own (``None`` = ask for the derivation).
    cp_cfg = {"horizon": 8, "max_length": 64, **cp_cfg}
    config = {
        "sessions": [{"source": str(nc_path), "labels_path": str(nc_path.with_name("cp_labels.tsv"))}],
        "features": {
            "name": "cp",
            "columns": {"speed": {}} if columns is None else columns,
            "labels": {"mapping": "mapping.txt", "branch": 0},
            "changepoint_features": cp_cfg,
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_open_session_migrates_legacy_attrs_before_expanding(tmp_path: Path):
    """Real session files predate the variable-schema convention and still
    spell a changepoint mask ``attrs["type"] = "changepoints"``; nothing
    migrates that automatically on load, so the expansion must do it itself."""
    nc_path = _session_with_changepoints(tmp_path / "session", legacy=True)
    config_path = _write_config(tmp_path, nc_path)
    cfg = load_config(config_path)
    session = open_session(cfg.sessions[0], cfg)
    trial_ds = session.trial_dataset(1)
    assert "speed_troughs_cp_prox0" in trial_ds.data_vars
    assert trial_ds["speed_troughs_cp_prox0"].attrs["normalise"] == 0


def test_open_session_expands_configured_changepoints(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(tmp_path, nc_path)
    cfg = load_config(config_path)
    session = open_session(cfg.sessions[0], cfg)
    trial_ds = session.trial_dataset(1)
    assert "speed_troughs_cp_prox0" in trial_ds.data_vars
    assert "speed_troughs_cp_binary" in trial_ds.data_vars
    assert "speed_troughs_cp_since" in trial_ds.data_vars
    assert "speed_troughs_cp_until" in trial_ds.data_vars
    assert "speed_troughs_cp_length" in trial_ds.data_vars
    assert trial_ds["speed_troughs_cp_prox0"].attrs["normalise"] == 0
    assert trial_ds["speed_troughs_cp_prox0"].attrs[schema.KIND] == schema.CHANGEPOINT_FEATURE
    # the raw mask stays the only changepoint variable — expansions are ordinary features
    assert schema.changepoint_vars(trial_ds) == ["speed_troughs"]


def test_inputs_generate_columns_without_spelling_them_out(tmp_path: Path):
    """``inputs`` + ``transforms`` are enough — no derived name in ``features.columns``."""
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={},
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["proximity"]},
    )
    cfg = load_config(config_path)
    assert cfg.features.columns == {"speed_troughs_cp_prox0": {}}
    data_dir = materialise(cfg)
    layout = read_layout(data_dir)
    assert layout.names == ["speed_troughs_cp_prox0|individual=self"]
    assert layout.normalise == [False]


def test_normalise_zero_columns_are_not_percentile_clipped(tmp_path: Path):
    """A sparse mask clipped to its 2nd/98th percentile would become a constant.

    ``normalise=0`` says the column's values already mean what they say, so it
    gates clipping as well as z-scoring: the mask keeps its ones and the
    proximity feature keeps its peak, while an ordinary column is still clipped.
    """
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={"speed": {}},
        changepoint_features={
            "sigmas": [2.0],
            "inputs": {"speed_troughs": {}},
            "transforms": ["binary", "proximity"],
        },
    )
    cfg = load_config(config_path)
    data_dir = materialise(cfg)
    layout = read_layout(data_dir)
    key = read_index(data_dir)["key"].iloc[0]
    x = np.load(data_dir / "features" / f"{key}.npy")
    column = {name.split("|")[0]: x[i] for i, name in enumerate(layout.names)}

    assert column["speed_troughs_cp_binary"].max() == pytest.approx(1.0)
    assert column["speed_troughs_cp_binary"].sum() == pytest.approx(3.0)
    assert column["speed_troughs_cp_prox0"].max() == pytest.approx(1.0)

    raw_speed = eto.open(str(nc_path)).trial(1)["speed"].sel(individual="A").values
    assert column["speed"].max() < raw_speed.max()


def test_merge_expands_one_block_for_all_masks(tmp_path: Path):
    """``merge: true`` ORs the named masks into one and expands that once."""
    nc_path = _session_with_two_changepoint_masks(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={},
        changepoint_features={
            "sigmas": [2.0],
            "inputs": {"speed_troughs": {}, "speed_peaks": {}},
            "transforms": ["binary"],
            "merge": True,
        },
    )
    cfg = load_config(config_path)
    assert cfg.features.columns == {"changepoints_cp_binary": {}}

    data_dir = materialise(cfg)
    layout = read_layout(data_dir)
    # the merge collapses the keypoint-like dims but leaves the individual
    # standing, so the sample still pins it to itself
    assert layout.names == ["changepoints_cp_binary|individual=self"]
    key = read_index(data_dir)["key"].iloc[0]
    merged = np.load(data_dir / "features" / f"{key}.npy")[0]
    # the union of both masks' frames, each mask contributing its own
    assert np.flatnonzero(merged).tolist() == [40, 90, 120, 160]


def test_merge_refuses_a_variable_that_is_not_a_mask(tmp_path: Path):
    nc_path = _session_with_two_changepoint_masks(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        changepoint_features={"sigmas": [2.0], "inputs": {"speed": {}}, "merge": True},
    )
    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="not changepoint masks"):
        open_session(cfg.sessions[0], cfg)


def test_transforms_filters_which_columns_are_generated(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={},
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["offset"]},
    )
    cfg = load_config(config_path)
    assert cfg.features.columns == {"speed_troughs_cp_since": {}, "speed_troughs_cp_until": {}}
    session = open_session(cfg.sessions[0], cfg)
    trial_ds = session.trial_dataset(1)
    assert "speed_troughs_cp_since" in trial_ds.data_vars
    assert "speed_troughs_cp_binary" not in trial_ds.data_vars
    assert "speed_troughs_cp_prox0" not in trial_ds.data_vars


def test_horizon_and_scale_by_reach_the_expansion(tmp_path: Path):
    """Both knobs travel from the YAML to ``add_changepoint_features`` unchanged."""
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={},
        changepoint_features={
            "sigmas": [2.0],
            "inputs": {"speed_troughs": {}},
            "transforms": ["proximity", "offset"],
            "horizon": 3,
            "scale_by": "speed",
            "max_length": 4,
        },
    )
    cfg = load_config(config_path)
    assert cfg.features.changepoint_features.horizon == 3.0
    assert cfg.features.changepoint_features.max_length == 4.0
    assert "speed_troughs_cp_length" not in cfg.features.columns  # ``length`` is its own transform group
    session = open_session(cfg.sessions[0], cfg)
    trial_ds = session.trial_dataset(1)
    since = trial_ds["speed_troughs_cp_since"].values
    # a horizon of 3 samples leaves exactly four levels
    assert set(np.round(np.unique(since), 6)) <= {0.0, round(1 / 3, 6), round(2 / 3, 6), 1.0}
    assert "scaled by speed" in trial_ds["speed_troughs_cp_prox0"].attrs["description"]


def _labels(n: int, *, seed: int = 0) -> pd.DataFrame:
    """*n* curated ``flap`` (id 3) state events on trial 1, durations 0.2–1.0 s."""
    rng = np.random.default_rng(seed)
    onsets = np.sort(rng.uniform(0.0, DURATION - 1.2, size=n))
    durations = rng.uniform(0.2, 1.0, size=n)
    return pd.DataFrame(
        {
            "trial": 1,
            "individual": "A",
            "individual_rec": "",
            "labels": 3,
            "onset_s": onsets,
            "offset_s": onsets + durations,
            "event_type": "state",
            "confidence": 1.0,
            "labeling_method": "manual",
            "changepoint_corrected": 0,
            "prediction_source": "",
            "n_samples": 0,
        }
    )


def test_scales_are_derived_from_the_labels_at_materialise(tmp_path: Path):
    """Leave sigmas/horizon/max_length out and materialise reads them off the curated durations.

    The numbers land in ``columns.yaml`` with a note saying what happened,
    and a session opened afterwards is expanded at exactly those scales.
    """
    nc_path = _session_with_changepoints(tmp_path / "session")
    labels = _labels(40)
    save_labels_tsv(nc_path.with_name("cp_labels.tsv"), labels)
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={"speed": {}},
        changepoint_features={"inputs": {"speed_troughs": {}}, "horizon": None, "max_length": None},
    )
    cfg = load_config(config_path)
    cpf = cfg.features.changepoint_features
    assert cpf.unresolved and cpf.sigmas is None
    # the column names are known before the values are
    assert [c for c in cfg.features.columns if "_cp_prox" in c] == [
        "speed_troughs_cp_prox0",
        "speed_troughs_cp_prox1",
        "speed_troughs_cp_prox2",
    ]
    # before materialise nothing can expand a session
    with pytest.raises(ValueError, match="materialise"):
        open_session(cfg.sessions[0], cfg)

    data_dir = materialise(cfg)
    recorded = read_layout(data_dir).changepoint_features
    durations = (labels["offset_s"] - labels["onset_s"]).to_numpy()
    assert recorded["horizon"] == pytest.approx(0.5 * np.percentile(durations, 5) * FS, abs=1e-4)
    assert recorded["max_length"] == pytest.approx(np.percentile(durations, 95) * FS, abs=1e-4)
    assert recorded["sigmas"] == pytest.approx([recorded["horizon"] / k for k in (16, 8, 4)], abs=1e-4)
    assert "Derived at materialise from 40 manual/curated state events of 1 session(s)" in recorded["note"]
    assert "horizon = 0.5 x p5(duration)" in recorded["note"]

    # every later stage reads them back: an unresolved config now opens fine, at the recorded scale
    session = open_session(cfg.sessions[0], cfg)
    since = session.trial_dataset(1)["speed_troughs_cp_since"].values.ravel()
    step = 1.0 / recorded["horizon"]
    assert since[41] == pytest.approx(step) and since[40] == 0.0
    desc = session.trial_dataset(1)["speed_troughs_cp_prox2"].attrs["description"]
    assert f"sigma {recorded['sigmas'][2]:g} samples" in desc


def test_derivation_needs_labels(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")  # empty labels
    config_path = _write_config(
        tmp_path, nc_path, changepoint_features={"inputs": {"speed_troughs": {}}, "horizon": None, "max_length": None}
    )
    with pytest.raises(ValueError, match="at least two positive label durations"):
        materialise(load_config(config_path))


def test_spelled_scales_are_kept_and_carry_no_note(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    save_labels_tsv(nc_path.with_name("cp_labels.tsv"), _labels(10))
    config_path = _write_config(
        tmp_path,
        nc_path,
        changepoint_features={"sigmas": [2.0, 4.0], "inputs": {"speed_troughs": {}}, "horizon": 6, "max_length": 30},
    )
    cfg = load_config(config_path)
    assert not cfg.features.changepoint_features.unresolved
    recorded = read_layout(materialise(cfg)).changepoint_features
    assert recorded == {"sigmas": [2.0, 4.0], "horizon": 6.0, "max_length": 30.0, "note": None}


def test_a_trained_run_records_the_derived_scales(tmp_path: Path):
    """``train`` saves the resolved config, so inference never re-derives."""
    from ethograph.segment.config import load_config as _load
    from ethograph.segment.train import train

    nc_path = _session_with_changepoints(tmp_path / "session")
    save_labels_tsv(nc_path.with_name("cp_labels.tsv"), _labels(12))
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={"speed": {}},
        changepoint_features={"inputs": {"speed_troughs": {}}, "horizon": None, "max_length": None},
    )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["train"] = {
        "epochs": 1,
        "eval_every": 1,
        "split": {"train_fraction": 1.0, "val_fraction": 0.0, "test_fraction": 0.0},
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = train(_load(config_path))
    saved = _load(result.run_dir / "config.yaml").features.changepoint_features
    recorded = read_layout(_load(config_path).data_dir).changepoint_features
    assert not saved.unresolved
    assert saved.horizon == recorded["horizon"] and saved.sigmas == pytest.approx(recorded["sigmas"])
    assert saved.note == recorded["note"]


def test_horizon_must_be_positive(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "horizon": -1},
    )
    with pytest.raises(ValueError, match="horizon must be positive"):
        load_config(config_path)


def test_a_saved_config_reloads(tmp_path: Path):
    """`save_config` -> `load_config` is what a run directory does on every infer.

    The generated columns must not be dumped as explicit entries, or reloading
    would read them as colliding with the expansion that produced them.
    """
    from ethograph.segment.config import save_config

    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={"speed": {}},
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["proximity"]},
    )
    cfg = load_config(config_path)
    reloaded = load_config(save_config(cfg, tmp_path / "run" / "config.yaml"))
    assert reloaded.features.columns == cfg.features.columns
    assert "speed_troughs_cp_prox0" in reloaded.features.columns


def test_explicit_column_colliding_with_generated_is_an_error(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        columns={"speed_troughs_cp_prox0": {}},
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["proximity"]},
    )
    with pytest.raises(ValueError, match="also generates"):
        load_config(config_path)


def test_changepoint_features_requires_inputs(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(tmp_path, nc_path, changepoint_features={"sigmas": [2.0]})
    with pytest.raises(ValueError, match="must name at least one changepoint variable"):
        load_config(config_path)


def test_changepoint_features_rejects_unknown_transform(tmp_path: Path):
    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(
        tmp_path,
        nc_path,
        changepoint_features={"sigmas": [2.0], "inputs": {"speed_troughs": {}}, "transforms": ["bogus"]},
    )
    with pytest.raises(ValueError, match="must be a subset of"):
        load_config(config_path)


def test_expansion_is_required_to_be_configured(tmp_path: Path):
    """Without ``features.changepoint_features`` the expanded column simply doesn't exist."""
    nc_path = _session_with_changepoints(tmp_path / "session")
    (tmp_path / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    config = {
        "sessions": [{"source": str(nc_path), "labels_path": str(nc_path.with_name("cp_labels.tsv"))}],
        "features": {
            "name": "cp",
            "columns": {"speed_troughs_cp_prox0": {}},
            "labels": {"mapping": "mapping.txt", "branch": 0},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="not available in this session"):
        materialise(cfg)


def test_pynapple_session_refuses_changepoint_expansion(tmp_path: Path):
    nap = pytest.importorskip("pynapple")
    from ethograph.io.nwb_alignment import alignment_from_trials_ep

    folder = tmp_path / "pynapple_session"
    folder.mkdir(parents=True)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    nap.Tsd(t=t, d=np.abs(np.sin(t * 3))).save(str(folder / "speed.npz"))
    group = nap.TsGroup({0: nap.Ts(t=np.array([1.0, 2.0]))})
    group.set_info(source_label=["speed"], **schema.changepoint_metadata(1, target_feature="speed"))
    group.save(str(folder / "speed_troughs.npz"))
    ep = nap.IntervalSet(start=[0.0], end=[DURATION - 1 / FS])
    ep.set_info(trial=[1])
    alignment_from_trials_ep(ep, folder / ".ethograph" / "alignment.nwb")
    save_labels_tsv(folder / "labels.tsv", _empty_labels())

    (tmp_path / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    config = {
        "sessions": [{"source": str(folder), "labels_path": str(folder / "labels.tsv")}],
        "features": {
            "name": "cp",
            "columns": {"speed": {}},
            "labels": {"mapping": "mapping.txt", "branch": 0},
            "changepoint_features": {"sigmas": [2.0], "inputs": {"speed_troughs": {}}},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="only implemented for xarray sessions"):
        open_session(cfg.sessions[0], cfg)


def test_samples_carry_their_candidate_frames_even_when_ablated(tmp_path: Path):
    """The loss is told where the masks fire off the full sample; ``drop_kinds`` only changes what the model sees."""
    from ethograph.io.schema import CHANGEPOINT_FEATURE
    from ethograph.segment.dataset import MaterialisedStore, SampleDataset, collate
    from ethograph.segment.preprocess import NormStats

    nc_path = _session_with_changepoints(tmp_path / "session")
    config_path = _write_config(tmp_path, nc_path, columns={"speed": {}})
    cfg = load_config(config_path)
    data_dir = materialise(cfg)
    store = MaterialisedStore.open(data_dir)
    assert list(store.layout.candidate_columns()) == [store.layout.features.index("speed_troughs_cp_binary")]
    keep = store.layout.keep_mask([CHANGEPOINT_FEATURE])
    ablated = store.layout.subset(keep)
    ds = SampleDataset(store, store.keys, NormStats.identity(int(keep.sum())), keep=keep, layout=ablated)
    x, y, candidates, key = ds[0]
    assert x.shape[0] == int(keep.sum()) and "cp_" not in " ".join(ablated.features)
    assert candidates.dtype == torch.bool and list(np.flatnonzero(candidates.numpy())) == [40, 120, 160]
    xb, yb, mask, cb, keys = collate([ds[0]])
    assert cb.shape == (1, x.shape[1]) and bool(cb[0, 120]) and not bool(cb[0, 121])
