"""The session record owns the individuals; a dataset may name a subset, never a stranger."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ethograph as eto
from ethograph.io.data_loader import load_features_dataset
from ethograph.io.nwb_alignment import NWBAlignment, set_individuals
from ethograph.io.pairing import pair_media


def _session(tmp_path: Path, individuals: list[str] | None, dim: list[str]) -> Path:
    folder = tmp_path / "sess"
    folder.mkdir()
    n = 10
    ds = xr.Dataset(
        {"speed": (("time", "individual"), np.ones((n, len(dim))))},
        coords={"time": np.arange(n) / 10.0, "individual": dim},
        attrs={"trial": 1, "fps": 10.0},
    )
    eto.from_datasets([ds]).save(folder / "session.nc")
    table = pd.DataFrame({"trial": [1], "start_time": [0.0], "stop_time": [1.0]})
    pair_media(table, output_path=folder / ".ethograph" / "alignment.nwb", individuals=individuals)
    return folder


def test_pair_media_records_individuals_and_they_can_be_edited(tmp_path: Path):
    out = tmp_path / ".ethograph" / "alignment.nwb"
    pair_media(pd.DataFrame({"trial": [1], "start_time": [0.0], "stop_time": [1.0]}), output_path=out)
    assert NWBAlignment(out).individuals == []  # nothing declared: unknown, not empty-by-fiat
    set_individuals(out, ["crow1", "crow2"])
    assert NWBAlignment(out).individuals == ["crow1", "crow2"]
    with pytest.raises(ValueError, match="unique"):
        set_individuals(out, ["crow1", "crow1"])


def test_a_dataset_naming_a_subset_loads(tmp_path: Path):
    folder = _session(tmp_path, ["crow1", "crow2"], dim=["crow2"])
    result = load_features_dataset(str(folder))
    assert result.nwb_alignment.individuals == ["crow1", "crow2"]


def test_a_dataset_naming_a_stranger_is_refused(tmp_path: Path):
    folder = _session(tmp_path, ["crow1", "crow2"], dim=["crow1", "crwo2"])
    with pytest.raises(ValueError, match=r"crwo2.*does not declare"):
        load_features_dataset(str(folder))


def test_no_declared_individuals_means_no_check(tmp_path: Path):
    folder = _session(tmp_path, None, dim=["anyone"])
    assert load_features_dataset(str(folder)).trial_ids == [1]


def test_label_individuals_prefers_the_session_record(app_state, tmp_path: Path):
    class _Alignment:
        individuals = ["crow1", "crow2"]

    app_state.nwb_alignment = _Alignment()
    assert app_state.label_individuals() == ["crow1", "crow2"]
