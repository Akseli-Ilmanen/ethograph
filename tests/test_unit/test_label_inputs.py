"""Existing labels as model inputs (``features/label_inputs.py``).

What earns a test: the two renderings (a state's indicator, a point's
Laplacian bump), the rule that an input branch is never a target branch —
in both pipelines' configs — and the session-open expansion that every
stage reads through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import ethograph as eto
from ethograph.features.label_inputs import (
    DIM,
    VARIABLE,
    LabelInputClass,
    add_label_inputs,
    laplacian_bump,
    render,
    state_indicator,
)
from ethograph.io import schema
from ethograph.labels.tsv_store import save_labels_tsv
from ethograph.segment.config import config_from_dict as segment_config
from ethograph.segment.config import config_to_dict as segment_to_dict
from ethograph.segment.config import load_config
from ethograph.segment.materialise import materialise, read_layout
from ethograph.segment.sessions import open_session
from ethograph.spot.config import config_from_dict as spot_config
from ethograph.spot.config import config_to_dict as spot_to_dict

WALK = LabelInputClass(1, "walk", 0, "state")
PECK = LabelInputClass(2, "peck", 0, "point")
MAPPING = "0 background\n1 walk 0\n2 peck 0 point\n3 groom 1\n"
LABEL_COLUMNS = [
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


def _row(trial: int, individual: str, label: int, onset: float, offset: float, method: str = "manual") -> dict:
    return {
        "trial": trial,
        "individual": individual,
        "individual_rec": "",
        "labels": label,
        "onset_s": onset,
        "offset_s": offset,
        "event_type": "point" if label == PECK.label else "state",
        "confidence": 1.0,
        "labeling_method": method,
        "changepoint_corrected": 0,
        "prediction_source": "",
        "n_samples": 0,
    }


def _labels(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=LABEL_COLUMNS)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRender:
    time = np.arange(0.0, 4.0, 0.1)

    def test_a_state_is_its_indicator(self):
        out = state_indicator(self.time, np.array([1.0]), np.array([2.0]))
        assert out[(self.time >= 1.0) & (self.time <= 2.0)].all()
        assert not out[(self.time < 1.0) | (self.time > 2.0)].any()

    def test_a_point_is_a_laplacian_bump_bounded_by_one(self):
        out = laplacian_bump(self.time, np.array([2.0]), 0.5)
        assert out[20] == pytest.approx(1.0)
        assert out[25] == pytest.approx(np.exp(-1.0))
        # Two events close together read as one strong moment, never as more than 1.
        two = laplacian_bump(self.time, np.array([2.0, 2.1]), 0.5)
        assert two.max() == pytest.approx(1.0)
        assert laplacian_bump(self.time, np.array([]), 0.5).sum() == 0

    def test_columns_follow_the_classes_and_the_widths(self):
        df = _labels(_row(1, "A", WALK.label, 0.5, 1.0), _row(1, "B", PECK.label, 2.0, 2.0))
        out = render(df, self.time, [WALK, PECK], [0.1, 1.0])
        assert out.shape == (self.time.size, 3)
        assert out[7, 0] == 1.0 and out[20, 0] == 0.0
        assert out[20, 1] == pytest.approx(1.0) and out[20, 2] == pytest.approx(1.0)
        assert out[30, 1] < out[30, 2]  # the wide bump reaches further

    def test_an_actor_reads_only_its_own_rows_and_an_absent_class_is_zeros(self):
        df = _labels(_row(1, "A", WALK.label, 0.5, 1.0), _row(1, "B", PECK.label, 2.0, 2.0))
        a = render(df, self.time, [WALK, PECK], [0.5], actor="A")
        assert a[:, 0].sum() > 0 and a[:, 1].sum() == 0
        assert render(_labels(), self.time, [WALK, PECK], [0.5]).sum() == 0


class TestAddLabelInputs:
    def _ds(self, individuals: list[str] | None) -> xr.Dataset:
        t = np.arange(0.0, 4.0, 0.1)
        if individuals is None:
            return xr.Dataset({"speed": (("time",), np.ones(t.size))}, coords={"time": t}, attrs={"trial": 1})
        return xr.Dataset(
            {"speed": (("time", "individual"), np.ones((t.size, len(individuals))))},
            coords={"time": t, "individual": individuals},
            attrs={"trial": 1},
        )

    def test_the_dataset_individual_dim_is_carried_one_slice_per_animal(self):
        ds = self._ds(["A", "B"])
        df = _labels(_row(1, "A", WALK.label, 0.5, 1.0), _row(1, "B", PECK.label, 2.0, 2.0))
        out = add_label_inputs(ds, df, [WALK, PECK], [0.5], "speed", individuals=["A", "B"])
        da = out[VARIABLE]
        assert da.dims == ("time", "individual", DIM)
        assert list(da.coords[DIM].values) == ["walk", "peck@0.5s"]
        assert float(da.sel(individual="A", label_input="walk")[7]) == 1.0
        assert float(da.sel(individual="B", label_input="walk").sum()) == 0.0
        assert float(da.sel(individual="B", label_input="peck@0.5s")[20]) == pytest.approx(1.0)
        assert schema.kind_of(da) == schema.LABEL_INPUT and not schema.is_normalise(da)

    def test_without_a_dataset_dim_the_individuals_become_one(self):
        out = add_label_inputs(self._ds(None), _labels(), [WALK], [0.5], "speed", individuals=["A"])
        assert out[VARIABLE].dims == ("time", "individual", DIM)

    def test_flat_for_a_single_event_stream(self):
        df = _labels(_row(1, "A", WALK.label, 0.5, 1.0), _row(1, "B", WALK.label, 2.0, 3.0))
        out = add_label_inputs(self._ds(None), df, [WALK], [0.5], "speed", actor="A")
        da = out[VARIABLE]
        assert da.dims == ("time", DIM)
        assert float(da[7, 0]) == 1.0 and float(da[25, 0]) == 0.0

    def test_the_clock_must_exist(self):
        with pytest.raises(ValueError, match="clock"):
            add_label_inputs(self._ds(None), _labels(), [WALK], [0.5], "nothing")


# ---------------------------------------------------------------------------
# The rule, in both configs
# ---------------------------------------------------------------------------


@pytest.fixture
def mapping(tmp_path: Path) -> Path:
    path = tmp_path / "mapping.txt"
    path.write_text(MAPPING, encoding="utf-8")
    return path


def _segment_dict(mapping: Path, **label_inputs) -> dict:
    return {
        "sessions": [{"source": "x.nc"}],
        "features": {
            "columns": {"speed": {}},
            "labels": {"mapping": str(mapping), "branch": 1},
            "label_inputs": {"branches": [0], **label_inputs},
        },
    }


class TestSegmentConfig:
    def test_an_input_branch_that_is_a_target_branch_is_refused(self, mapping: Path, tmp_path: Path):
        data = _segment_dict(mapping)
        data["features"]["label_inputs"]["branches"] = [0, 1]
        with pytest.raises(ValueError, match="learns to copy"):
            segment_config(data, tmp_path)

    def test_the_generated_entry_joins_the_columns_and_round_trips(self, mapping: Path, tmp_path: Path):
        cfg = segment_config(_segment_dict(mapping, point_sigmas_s=[0.2]), tmp_path)
        assert cfg.features.columns[VARIABLE] == {DIM: ["walk", "peck@0.2s"]}
        assert cfg.features.label_inputs is not None
        assert cfg.features.label_inputs.clock == "speed"
        assert cfg.features.label_inputs.mapping == mapping
        plain = segment_to_dict(cfg)
        assert VARIABLE not in plain["features"]["columns"]
        again = segment_config(plain, tmp_path)
        assert again.features.columns == cfg.features.columns

    def test_a_spelled_duplicate_and_a_missing_clock_are_refused(self, mapping: Path, tmp_path: Path):
        data = _segment_dict(mapping)
        data["features"]["columns"][VARIABLE] = {DIM: ["walk"]}
        with pytest.raises(ValueError, match="label_inputs generates"):
            segment_config(data, tmp_path)
        data = _segment_dict(mapping)
        data["features"]["columns"] = {}
        with pytest.raises(ValueError, match="clock is unset"):
            segment_config(data, tmp_path)


class TestSpotConfig:
    def _dict(self, mapping: Path, classes: list[int], features: dict | None = None) -> dict:
        return {
            "sessions": [{"source": "x.nc"}],
            "labels": {"classes": classes},
            "features": {"speed": {}} if features is None else features,
            "label_inputs": {"branches": [0], "mapping": str(mapping), "point_sigmas_s": [0.1]},
        }

    def test_a_target_class_inside_an_input_branch_is_refused(self, mapping: Path, tmp_path: Path):
        with pytest.raises(ValueError, match="learns to copy"):
            spot_config(self._dict(mapping, classes=[2]), tmp_path)

    def test_the_columns_ride_into_features_and_round_trip(self, mapping: Path, tmp_path: Path):
        cfg = spot_config(self._dict(mapping, classes=[3]), tmp_path)
        assert list(cfg.features) == ["speed", VARIABLE]
        assert cfg.fusing
        plain = spot_to_dict(cfg)
        assert VARIABLE not in plain["features"]
        assert spot_config(plain, tmp_path).features == cfg.features

    def test_pixels_alone_need_a_named_clock(self, mapping: Path, tmp_path: Path):
        with pytest.raises(ValueError, match="clock is unset"):
            spot_config(self._dict(mapping, classes=[3], features={}), tmp_path)


# ---------------------------------------------------------------------------
# Session open → materialise
# ---------------------------------------------------------------------------

FS = 50.0


def _session(folder: Path) -> Path:
    """Two trials, two individuals, ``speed`` at 50 Hz, labels of both branches."""
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, 4.0, 1.0 / FS)
    datasets = [
        xr.Dataset(
            {"speed": (("time", "individual"), np.random.default_rng(trial).random((t.size, 2)))},
            coords={"time": t, "individual": ["A", "B"]},
            attrs={"trial": trial, "fps": FS},
        )
        for trial in (1, 2)
    ]
    nc_path = folder / "li.nc"
    eto.from_datasets(datasets).save(str(nc_path))
    save_labels_tsv(
        folder / "li_labels.tsv",
        _labels(
            _row(1, "A", WALK.label, 0.5, 1.0),
            _row(1, "B", PECK.label, 2.0, 2.0),
            _row(1, "A", 3, 1.5, 2.5),  # the target class: never an input
            _row(2, "A", WALK.label, 1.0, 2.0, method="automated"),
        ),
    )
    return nc_path


def _write_config(root: Path, nc_path: Path, **label_inputs) -> Path:
    (root / "mapping.txt").write_text(MAPPING, encoding="utf-8")
    config = {
        "sessions": [{"source": str(nc_path), "labels_path": str(nc_path.with_name("li_labels.tsv"))}],
        "features": {
            "name": "li",
            "columns": {"speed": {}},
            "labels": {"mapping": "mapping.txt", "branch": 1},
            "label_inputs": {"branches": [0], "point_sigmas_s": [0.5], **label_inputs},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_open_session_renders_each_trial_from_its_own_curated_labels(tmp_path: Path):
    nc_path = _session(tmp_path / "session")
    cfg = load_config(_write_config(tmp_path, nc_path))
    session = open_session(cfg.sessions[0], cfg)
    first = session.trial_dataset(1)[VARIABLE]
    assert first.dims == ("time", "individual", DIM)
    assert float(first.sel(individual="A", label_input="walk").sel(time=0.75, method="nearest")) == 1.0
    assert float(first.sel(individual="B", label_input="peck@0.5s").sel(time=2.0, method="nearest")) == 1.0
    assert float(first.sel(individual="A", label_input="peck@0.5s").sum()) == 0.0
    # Trial 2 only has an automated row: not evidence, so it renders as zeros.
    assert float(session.trial_dataset(2)[VARIABLE].sum()) == 0.0
    cfg = load_config(_write_config(tmp_path, nc_path, include_automated=True))
    session = open_session(cfg.sessions[0], cfg)
    assert float(session.trial_dataset(2)[VARIABLE].sel(individual="A", label_input="walk").sum()) > 0


def test_materialised_columns_carry_the_kind_and_are_never_zscored(tmp_path: Path):
    nc_path = _session(tmp_path / "session")
    cfg = load_config(_write_config(tmp_path, nc_path))
    data_dir = materialise(cfg)
    layout = read_layout(data_dir)
    ours = [i for i, f in enumerate(layout.features) if f == VARIABLE]
    assert [layout.names[i] for i in ours] == [
        f"{VARIABLE}|label_input=walk,individual=self",
        f"{VARIABLE}|label_input=peck@0.5s,individual=self",
    ]
    assert all(layout.kinds[i] == schema.LABEL_INPUT for i in ours)
    assert not any(layout.normalise[i] for i in ours)
    assert layout.keep_mask([schema.LABEL_INPUT]).sum() == len(layout.names) - len(ours)
