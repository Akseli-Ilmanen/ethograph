"""How the config reader treats a nested block written empty, or pinned to ``null``.

One synthetic session, the same shape as ``test_segment_pipeline``'s fixture,
used only to give ``load_config`` a real project to read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

torch = pytest.importorskip("torch")

import ethograph as eto  # noqa: E402
from ethograph.labels.intervals import LABELING_MANUAL  # noqa: E402
from ethograph.labels.tsv_store import save_labels_tsv  # noqa: E402
from ethograph.segment.config import load_config  # noqa: E402

FS = 50.0
DURATION = 8.0
TRIALS = [1, 2, 3, 4, 5, 6]
KEYPOINTS = ["beak", "tail"]


def _trial_ds(trial: int) -> tuple[xr.Dataset, list[tuple[float, float]]]:
    rng = np.random.default_rng(trial)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    pos = rng.normal(0, 0.2, size=(t.size, 2, len(KEYPOINTS), 1))
    bouts = [(1.0 + 0.2 * (trial % 3), 2.0 + 0.2 * (trial % 3)), (4.5, 5.5)]
    for on, off in bouts:
        m = (t >= on) & (t <= off)
        pos[m] += np.sin(t[m, None, None, None] * 40) * 3.0
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * FS
    ds = xr.Dataset(
        {
            "position": (("time", "space", "keypoint", "individual"), pos),
            "speed": (("time", "keypoint", "individual"), speed),
        },
        coords={"time": t, "space": ["x", "y"], "keypoint": KEYPOINTS, "individual": ["A"]},
        attrs={"trial": trial, "fps": FS},
    )
    return ds, bouts


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A minimal one-session project, ready to train any architecture."""
    folder = tmp_path / "sessions" / "s1"
    folder.mkdir(parents=True)
    datasets, rows = [], []
    for trial in TRIALS:
        ds, bouts = _trial_ds(trial)
        datasets.append(ds)
        rows += [
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
            for on, off in bouts
        ]
    nc_path = folder / "s1.nc"
    eto.from_datasets(datasets).save(str(nc_path))
    save_labels_tsv(folder / "s1_labels.tsv", pd.DataFrame(rows))

    root = tmp_path / "project"
    root.mkdir()
    (root / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "sessions": [{"source": str(nc_path), "labels_path": str(folder / "s1_labels.tsv")}],
                "features": {
                    "name": "kin",
                    "columns": {
                        "position": {"space": ["x", "y"], "keypoint": KEYPOINTS},
                        "speed": {"keypoint": ["beak"]},
                    },
                    "labels": {"mapping": "mapping.txt", "branch": 0},
                },
                "train": {
                    "epochs": 2,
                    "eval_every": 1,
                    "split": {"train_fraction": 0.5, "val_fraction": 0.25, "test_fraction": 0.25},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


def _run(project_dir: Path, overrides: list[str], name: str):
    return eto.segment.Project(project_dir / "config.yaml", *overrides, f"train.run_name={name}").train()


class TestEmptyBlocks:
    """A block with nothing after it, and an explicit ``null`` that means something.

    A nested block written empty is "use the defaults" — but
    ``clip_percentiles: null`` is not the same statement: ``None`` is the
    *value* that says "do not clip at all". A config reader that treated every
    null as "absent" would silently restore the default percentiles.
    """

    def _written(self, project: Path, block: str) -> str:
        (project / "empty.yaml").write_text(f"base: config.yaml\nfeatures:\n{block}", encoding="utf-8")
        return str(project / "empty.yaml")

    def test_an_empty_preprocess_block_takes_the_defaults(self, project: Path) -> None:
        cfg = load_config(self._written(project, "  preprocess:\n  # as it ships\n"))
        assert cfg.features.preprocess.clip_percentiles == (2.0, 98.0)
        assert cfg.features.preprocess.zscore is True

    def test_an_explicit_null_clip_percentiles_survives(self, project: Path) -> None:
        cfg = load_config(self._written(project, "  preprocess:\n    clip_percentiles: null\n"))
        assert cfg.features.preprocess.clip_percentiles is None

    def test_pinned_percentiles_are_read(self, project: Path) -> None:
        cfg = load_config(self._written(project, "  preprocess:\n    clip_percentiles: [5, 95]\n"))
        assert cfg.features.preprocess.clip_percentiles == (5.0, 95.0)
