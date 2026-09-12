"""A dataset whose only metadata is the alignment's trials table keeps it.

Dropping a folder of clips writes an alignment NWB with one ``video_cam-1``
filename per trial and no metadata file at all — so the load result carries a
table with ``metadata_path`` None. Committing that load must not read the
missing path as "this dataset has no metadata".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ethograph.gui.widgets_data import _LoadContext
from ethograph.io.data_loader import load_features_dataset
from ethograph.io.pairing import pair_media


@pytest.fixture
def session_with_alignment(tmp_path):
    """A two-trial ``.nc`` beside an alignment NWB naming one clip per trial."""
    clips = [str(tmp_path / f"clip_{i}.mp4") for i in (1, 2)]
    pair_media(
        trial_table=pd.DataFrame(
            {
                "trial": [1, 2],
                "video_cam-1": clips,
                "start_time": [0.0, 2.0],
                "stop_time": [2.0, 4.0],
            }
        ),
        stream_rates={"video": 30.0},
        output_path=tmp_path / ".ethograph" / "alignment.nwb",
    )

    datasets = []
    for trial, t0 in ((1, 0.0), (2, 2.0)):
        time = np.arange(t0, t0 + 2.0, 0.1)
        ds = xr.Dataset(
            {"motion": (("time", "individuals"), np.zeros((len(time), 1)))},
            coords={"time": time, "individuals": ["individual_1"]},
        )
        ds.attrs["trial"] = trial
        datasets.append(ds)

    import ethograph as eto

    nc_path = tmp_path / "session.nc"
    eto.from_datasets(datasets).to_netcdf(nc_path)
    return str(nc_path)


def test_alignment_filenames_survive_the_commit(gui, session_with_alignment):
    _shell, meta = gui
    data_widget = meta.data_widget

    result = load_features_dataset(session_with_alignment)
    assert result.metadata_path is None
    assert list(result.metadata_df["video_cam-1"]) == ["clip_1.mp4", "clip_2.mp4"]

    data_widget._apply_to_state(
        _LoadContext(
            result=result,
            nc_file_path=session_with_alignment,
            catalog=result.catalog,
            dt=result.dt,
            ds=result.dt.itrial(0),
            trials=result.trial_ids,
            all_labels_df=result.all_labels_df,
            data_loader=result.data_loader,
        )
    )

    metadata_df = meta.app_state.metadata_df
    assert metadata_df is not None
    assert list(metadata_df["video_cam-1"]) == ["clip_1.mp4", "clip_2.mp4"]
