"""Opening a session folder: its .nc layers, its alignment alone, one backend per folder."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ethograph as eto
from ethograph.io.data_loader import AmbiguousSessionError, load_features_dataset
from ethograph.io.nwb_alignment import NWBAlignment, alignment_from_trialtree
from ethograph.io.pairing import pair_media


def _tree(trials: list[int], var: str, n: int = 10, fps: float = 10.0):
    datasets = []
    for t in trials:
        ds = xr.Dataset(
            {var: (("time", "individual"), (np.arange(n, dtype=float) + t)[:, None])},
            coords={"time": np.arange(n) / fps, "individual": ["A"]},
            attrs={"trial": t, "fps": fps},
        )
        datasets.append(ds)
    return eto.from_datasets(datasets)


@pytest.fixture
def session(tmp_path: Path) -> Path:
    folder = tmp_path / "sess"
    folder.mkdir()
    _tree([1, 2], "speed").save(folder / "speed.nc")
    return folder


def test_a_folder_with_one_nc_loads_like_the_file(session: Path):
    by_folder = load_features_dataset(str(session))
    by_file = load_features_dataset(str(session / "speed.nc"))
    assert by_folder.trial_ids == by_file.trial_ids == [1, 2]
    assert "speed" in by_folder.dt.trial(1)


def test_a_second_nc_is_an_old_version_and_refused_by_name(session: Path):
    _tree([1, 2], "speed").save(session / "speed_v2.nc")
    with pytest.raises(AmbiguousSessionError, match=r"speed.nc, speed_v2.nc.*ignore: \[speed.nc\]") as info:
        load_features_dataset(str(session))
    assert sorted(p.name for p in info.value.candidates) == ["speed.nc", "speed_v2.nc"]


def test_the_ignore_list_resolves_it_and_a_named_file_always_loads(session: Path):
    _tree([1, 2], "speed").save(session / "speed_v2.nc")
    assert load_features_dataset(str(session), ignore=["speed.nc"]).trial_ids == [1, 2]
    assert load_features_dataset(str(session), ignore=["*_v2.nc"]).trial_ids == [1, 2]
    assert load_features_dataset(str(session / "speed.nc"), ignore=["speed.nc"]).trial_ids == [1, 2]


def test_a_root_nwb_beside_nc_is_refused(session: Path):
    (session / "session.nwb").touch()
    with pytest.raises(ValueError, match="one backend"):
        load_features_dataset(str(session))


def test_alignment_only_folder_is_a_session_with_trials_and_no_features(tmp_path: Path):
    folder = tmp_path / "videos"
    folder.mkdir()
    table = pd.DataFrame({"trial": [1, 2], "start_time": [0.0, 5.0], "stop_time": [4.0, 9.0]})
    pair_media(table, output_path=folder / ".ethograph" / "alignment.nwb")
    result = load_features_dataset(str(folder))
    assert result.dt is None
    assert result.trial_ids == [1, 2]
    assert result.nwb_alignment.stop_time(2) == pytest.approx(9.0)
    assert result.catalog.feature_choices() == []


def test_a_folder_with_nothing_is_not_a_session(tmp_path: Path):
    with pytest.raises(ValueError, match="not a session"):
        load_features_dataset(str(tmp_path))


def test_legacy_labels_are_adopted_on_load(session: Path):
    (session / "speed_labels.tsv").write_text("onset_s\toffset_s\tlabels\tindividual\ttrial\n0.1\t0.4\t1\tdefault\t1\n")
    result = load_features_dataset(str(session))
    assert Path(result.labels_file_path).name == "labels.tsv"
    assert not (session / "speed_labels.tsv").exists()
    assert len(result.all_labels_df) == 1


def test_alignment_from_trialtree_lays_trials_end_to_end(tmp_path: Path):
    out = alignment_from_trialtree(_tree([1, 2], "speed", n=10, fps=10.0), tmp_path / ".ethograph" / "alignment.nwb")
    align = NWBAlignment(out)
    assert align.start_time(1) == pytest.approx(0.0)
    assert align.stop_time(1) == pytest.approx(1.0)  # 10 samples at 10 Hz: 0.9 s extent + one sample
    assert align.start_time(2) == pytest.approx(1.0)


def test_segment_session_opens_from_a_folder(session: Path):
    """A pipeline config names the session folder; the stem it logs and matches by is the folder's name."""
    from ethograph.segment.config import SessionSpec
    from ethograph.segment.sessions import open_session

    opened = open_session(SessionSpec(source=session))
    assert opened.trial_ids == [1, 2]
    assert opened.stem == "sess"


def test_a_pipeline_reads_the_ignore_list_from_the_nearest_project_yaml(session: Path, tmp_path: Path):
    """segment/spot configs find project.yaml by walking up from the config file."""
    from ethograph.segment.config import project_ignore

    _tree([1, 2], "speed").save(session / "speed_v2.nc")
    (tmp_path / "project.yaml").write_text("ignore: [speed.nc]\n", encoding="utf-8")
    config_path = tmp_path / "config" / "segment.yaml"
    config_path.parent.mkdir()
    assert project_ignore(config_path) == ("speed.nc",)
    assert project_ignore(tmp_path / "elsewhere") == ("speed.nc",)  # walk-up stops at the first project.yaml
    assert project_ignore(Path(tmp_path.anchor)) == ()
    (tmp_path / "project.yaml").write_text("sessions: [a]\nfeatures: {}\n", encoding="utf-8")  # a config, misnamed
    assert project_ignore(config_path) == ()
