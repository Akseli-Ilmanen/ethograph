"""A session in a folder with a non-ASCII name loads and saves.

netCDF-C on Windows cannot reach such a path ("No such file or directory" for
a file that is there), so every ``.nc`` open and write picks its engine from
the path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from ethograph.gui.cover_page import _is_trial_tree
from ethograph.io.trialtree import TrialTree
from ethograph.io.validation import movement_dataset_info


def _trial(trial: int) -> xr.Dataset:
    return xr.Dataset(
        {"speed": (("time", "individual"), np.arange(5, dtype=float)[:, None])},
        coords={"time": np.arange(5) / 10, "individual": ["A"]},
        attrs={"trial": trial},
    )


def test_trialtree_round_trips_in_a_non_ascii_folder(tmp_path: Path) -> None:
    folder = tmp_path / "präsi 发表"
    folder.mkdir()
    source = folder / "session.nc"

    TrialTree.from_datasets([_trial(1), _trial(2)]).save(source)
    assert _is_trial_tree(str(source)), "the drop must recognise it as a session, not swallow an OSError"

    dt = TrialTree.open(str(source))
    dt.update_trial(2, lambda ds: ds.assign(speed=ds["speed"] * 2))
    dt.save(source)  # the in-place path: append to a group, then re-open

    reopened = TrialTree.open(str(source))
    np.testing.assert_array_equal(reopened.trial(2)["speed"].values[:, 0], np.arange(5) * 2.0)
    np.testing.assert_array_equal(reopened.trial(1)["speed"].values[:, 0], np.arange(5, dtype=float))


def test_movement_file_is_recognised_in_a_non_ascii_folder(tmp_path: Path) -> None:
    folder = tmp_path / "präsi"
    folder.mkdir()
    pose = folder / "poses.nc"
    position = np.zeros((3, 2, 1, 1))
    xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), position)},
        attrs={"ds_type": "poses", "fps": 30.0},
    ).to_netcdf(pose, engine="h5netcdf")

    assert movement_dataset_info(pose) == ("poses", 30.0)
