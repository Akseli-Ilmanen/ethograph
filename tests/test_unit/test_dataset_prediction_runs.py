"""A template's shipped prediction runs must be folders ``PredictionsStore`` can open.

``datasets.get_prediction_run_assets`` names the files from the dataset's
metadata; ``PredictionsStore`` finds them by glob. Nothing else ties the two
spellings together.
"""

from pathlib import Path

import pytest

from ethograph.datasets import DATASETS, get_prediction_run_assets
from ethograph.labels.onset_curves import RUN_PREFIX
from ethograph.labels.predictions import PredictionsStore


@pytest.mark.parametrize("key", [k for k, ds in DATASETS.items() if ds.get("prediction_runs")])
def test_shipped_runs_open_as_prediction_folders(key: str, tmp_path: Path) -> None:
    runs = get_prediction_run_assets(key)
    assert runs
    for folder, files in runs.items():
        assert folder.startswith(RUN_PREFIX)
        run_dir = tmp_path / folder
        run_dir.mkdir()
        for local in files.values():
            (run_dir / local).touch()
        store = PredictionsStore(run_dir)
        assert store.tsv_path.name in files.values()
        assert store.npz_path is not None and store.npz_path.name in files.values()


def test_dataset_without_runs_has_none() -> None:
    assert get_prediction_run_assets("birdpark") == {}
