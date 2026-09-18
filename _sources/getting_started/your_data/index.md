(target-data-requirements)=
(target-multi-trial)=
# Bring your own data

{ref}`Drag & drop <target-data-loading>` handles one recording session whose
files all start together. Everything else — trials split across files, media on
separate clocks — needs a **session file** plus an **alignment file**, built
in the Data wizard or from a short Python
script.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fas}`table` 1. Dataset schema
:link: dataset_schema
:link-type: doc

The variables, attributes and dimensions Ethograph expects, per backend.
:::

:::{grid-item-card} {fas}`layer-group` 2. Trials
:link: trials
:link-type: doc

Grouping datasets into a trial structure, or splitting a continuous recording.
:::

:::{grid-item-card} {fas}`clock` 3. Media alignment (w. neuroconv)
:link: media_alignment
:link-type: doc

Pair media files, or align video to a recording system with neuroconv.
:::

:::{grid-item-card} {fas}`bolt` 4. Ephys recordings
:link: ephys
:link-type: doc

Supported formats, Kilosort output, and the two trace viewers.
:::

:::{grid-item-card} {fas}`folder-tree` 5. Folder layout
:link: folder_layout
:link-type: doc

Session, project and home folders: what lives where and who writes it.
:::

::::

Ethograph supports three backends. Pick the one matching your workflow; every
page has a tab per backend where they differ.

| Backend | Best for | Core object |
|---------|----------|-------------|
| **xarray** | Custom datasets, pose estimation, multi-dim arrays | {class}`xarray.Dataset` / {class}`~ethograph.io.trialtree.TrialTree` |
| **Pynapple** | Neuroscience time-series, NWB interop | {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` / {class}`~pynapple.TsGroup` |
| **NWB** | Standardised neurodata, DANDI archives | `.nwb` file (loaded via pynapple) |

```{note}
**NWB needs almost none of this.** An `.nwb` file already stores trials, media
references and features together, so it loads directly. The NWB tabs only
note where behaviour differs.
```

---

## Putting it together

A complete two-camera, ten-trial setup: {doc}`dataset_schema` and {doc}`trials`
for step 1, {doc}`media_alignment` for step 2.

::::{tab-set}

:::{tab-item} Xarray

```python
import numpy as np
import xarray as xr
import ethograph as eto

# 1) One Dataset per trial
datasets = []
for trial_id in range(1, 11):
    n_time = 9000  # 5 minutes at 30 fps
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                np.random.randn(n_time, 2, 4, 2),
                dims=["time", "space", "keypoint", "individual"],
            ),
            "speed": xr.DataArray(
                np.abs(np.random.randn(n_time, 4, 2)),
                dims=["time", "keypoint", "individual"],
            ),
        },
        coords={
            "time": np.arange(n_time) / 30.0,
            "space": ["x", "y"],
            "keypoint": ["nose", "left_ear", "right_ear", "tail"],
            "individual": ["mouse1", "mouse2"],
        },
    )
    ds.attrs["trial"] = trial_id
    ds.attrs["fps"] = 30.0
    datasets.append(ds)

dt = eto.from_datasets(datasets)
dt.save("session.nc")

# 2) Pairing: media files + trial timing
sources = [  # media folders are absolute; they need not be near the session folder
    eto.SourceSpec("video", device="cam-1", folder="/data/rig/video/cam1"),
    eto.SourceSpec("video", device="cam-2", folder="/data/rig/video/cam2"),
    eto.SourceSpec("pose", device="cam-1", folder="/data/rig/pose/cam1"),
    eto.SourceSpec("pose", device="cam-2", folder="/data/rig/pose/cam2", extension=".h5"),  # DLC also writes .csv twins
]
trial_table = eto.discover_media(sources)  # 10 rows, natural sort order, full paths
trial_table["start_time"] = [i * 300.0 for i in range(10)]
trial_table["stop_time"] = [(i + 1) * 300.0 - 0.5 for i in range(10)]

eto.pair_media(
    trial_table,
    stream_rates={"video": 30.0, "pose": 30.0},
    output_path=".ethograph/alignment.nwb",
)
```

Then launch Ethograph, select `session.nc` in the **Custom set-up** card, and
point the media folders at your video and pose directories.
:::

:::{tab-item} NWB

NWB stores trials, media references and features together, so there is no
separate alignment step. For DANDI datasets, select the `.nwb` URL or
downloaded file in the GUI and click **Load**. To build one yourself with
pynwb:

```python
from datetime import datetime

import pynwb
from dateutil.tz import tzlocal
from pynwb.behavior import BehavioralTimeSeries

nwbfile = pynwb.NWBFile(
    session_description="My experiment",
    identifier="session-001",
    session_start_time=datetime.now(tzlocal()),
)

nwbfile.add_trial_column(name="stimulus", description="Stimulus type")
for i in range(10):
    nwbfile.add_trial(
        start_time=i * 300.0,
        stop_time=(i + 1) * 300.0 - 0.5,
        stimulus="tone_A" if i % 2 else "tone_B",
    )

behavior_mod = nwbfile.create_processing_module("behavior", "Behavioral data")
behavior_ts = BehavioralTimeSeries(name="BehavioralTimeSeries")
behavior_ts.create_timeseries(name="speed", data=speed_array, rate=30.0, unit="cm/s")
behavior_mod.add(behavior_ts)

with pynwb.NWBHDF5IO("session.nwb", "w") as io:
    io.write(nwbfile)
```
:::

::::

---

## References

- {doc}`../../api/trialtree` — `from_datasets()`, `from_continuous()`, timing, iteration
- {class}`~ethograph.io.nwb_alignment.NWBAlignment` — alignment reader API
- {func}`~ethograph.discover_media` — the pairing table from folders or a filename pattern
- {func}`~ethograph.pair_media` — writes the alignment file, or adds streams to an existing `.nwb`

```{toctree}
:hidden:

dataset_schema
trials
media_alignment
ephys
folder_layout
```
