(target-data-requirements)=
(target-multi-trial)=
# Preparing your own data

{ref}`Drag & drop <target-data-loading>` handles one recording session whose
files all start together. Everything else — trials split across files, media on
separate clocks — needs a **session file** plus an **alignment file**, built
from a short Python script.

This page covers all three pieces:

1. **[Your dataset](#your-dataset)** — the schema EthoGraph expects.
2. **[Trials](#trials)** — grouping datasets into a trial structure.
3. **[Alignment](#alignment)** — tying media files and timing to those trials.

EthoGraph supports three backends. Pick the one matching your workflow; every
section below has a tab per backend.

| Backend | Best for | Core object |
|---------|----------|-------------|
| **xarray** | Custom datasets, pose estimation, multi-dim arrays | {class}`xarray.Dataset` / {class}`~ethograph.io.trialtree.TrialTree` |
| **Pynapple** | Neuroscience time-series, NWB interop | {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` / {class}`~pynapple.TsGroup` |
| **NWB** | Standardised neurodata, DANDI archives | `.nwb` file (loaded via pynapple) |

```{note}
**NWB needs almost none of this.** An `.nwb` file already stores trials, media
references and features together, so it loads directly — no session file to
build, no alignment step. Trials come from `nwb.trials`, media from
{class}`~pynwb.image.ImageSeries` in `nwb.acquisition`. If `nwb.trials` is
absent the recording is one trial; if a DANDI file lacks local media paths, the
GUI writes `.ethograph/alignment.nwb` on first load. The NWB tabs below only
note where behaviour differs.
```

---

## Your dataset

### Minimal working example

::::{tab-set}

:::{tab-item} Xarray
```python
import numpy as np
import xarray as xr
import ethograph as eto

ds = xr.Dataset(
    data_vars={
        "speed": xr.DataArray(
            np.random.randn(9000),
            dims=["time"],
            coords={"time": np.arange(9000) / 30.0},
        ),
    },
    coords={"individual": ["mouse1"]},
)
ds.attrs["trial"] = 1
ds.attrs["fps"] = 30.0

dt = eto.from_datasets([ds])
dt.save("session.nc")
```
:::

:::{tab-item} Pynapple
```python
import numpy as np
import pynapple as nap

speed = nap.Tsd(
    t=np.arange(9000) / 30.0,
    d=np.random.randn(9000),
)
nap.save_file({"speed": speed}, "session")
# Load in GUI: select the session.npz file
```
:::

:::{tab-item} NWB
```python
# Loaded directly — no conversion needed. In the GUI: select the .nwb file
# in the I/O widget and click Load.
#
# To create an NWB file programmatically, see the pynwb documentation:
# https://pynwb.readthedocs.io/en/stable/tutorials/general/plot_file.html
```
:::

::::

### Required attributes

::::{tab-set}

:::{tab-item} Xarray

Every trial's {class}`xarray.Dataset` **must** have:

| Attribute | Type | Description |
|-----------|------|-------------|
| `attrs["trial"]` | `int`, `str` | Trial identifier (1, 2, 3, ...). Must be unique across trials. |
| `attrs["fps"]` | `float` | Frame rate of the primary video. Not required for audio-only datasets. |

```python
ds.attrs["trial"] = 1
ds.attrs["fps"] = 30.0
```
:::

:::{tab-item} Pynapple

Pynapple objects carry timestamps natively — no `fps` or `trial` attribute is needed.

- **Timestamps**: every {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` / {class}`~pynapple.TsdTensor` stores its own time axis.
- **Trials**: defined by an {class}`~pynapple.IntervalSet` (either from NWB trials or created manually).

```python
import pynapple as nap
import numpy as np

trials = nap.IntervalSet(
    start=[0.0, 300.0, 600.0],
    end=[299.5, 599.5, 899.5],
)
speed = nap.Tsd(t=np.arange(27000) / 30.0, d=np.random.randn(27000))
```
:::

:::{tab-item} NWB

NWB files follow the [NWB standard](https://www.nwb.org/). EthoGraph reads:

- **Trials**: `nwb.trials` table (`start_time`, `stop_time`, plus custom columns).
- **Behavioural data**: {class}`~pynwb.TimeSeries` in `nwb.processing` modules.
- **Electrophysiology**: {class}`~pynwb.ecephys.ElectricalSeries` in `nwb.acquisition`.
- **Pose estimation**: `PoseEstimation` containers (ndx-pose extension).
:::

::::

### Features (plottable variables)

::::{tab-set}

:::{tab-item} Xarray

Any `data_var` with at least one dimension whose name contains `"time"` appears
in the GUI's **Feature** dropdown:

```python
ds["speed"] = xr.DataArray(
    speed_values,
    dims=["time", "keypoint", "individual"],
)
```

Different features can use different time coordinates with different sampling
rates (e.g. `time`, `time_accelerometer`, `time_video`).
:::

:::{tab-item} Pynapple

All {class}`~pynapple.Tsd`, {class}`~pynapple.TsdFrame`, and
{class}`~pynapple.TsdTensor` objects in the loaded data dict are detected
automatically. Column names become selectable dimensions.

```python
speed = nap.Tsd(t=time_s, d=speed_values)

position = nap.TsdFrame(
    t=time_s,
    d=pos_array,                        # shape: (n_time, 3)
    columns=["x", "y", "z"],
)
```
:::

:::{tab-item} NWB

Features are discovered automatically from NWB processing modules (excluding
`ecephys`, `ophys`, `ogen`). NWB data is loaded via pynapple, so all
{class}`~pynwb.TimeSeries` become pynapple objects internally.
:::

::::

### Specifying individuals

::::{tab-set}

:::{tab-item} Xarray

Individuals are stored as a **coordinate**, not an attribute. With multi-animal
data, this allows the GUI to store separate labels and feature data for
different individuals.

```python
ds = xr.Dataset(
    data_vars={
        "speed": xr.DataArray(
            speed_array,                # shape: (time, individual)
            dims=["time", "individual"],
        ),
    },
    coords={
        "time": time_values,
        "individual": ["mouse1", "mouse2", "mouse3"],
    },
)
```

When labelling, the selected individual filters which labels are shown and
created.
:::

:::{tab-item} Pynapple

Pynapple has no built-in concept of "individuals". Multi-subject data is
typically stored as separate objects:

```python
data = {
    "speed_mouse1": nap.Tsd(t=time_s, d=speed_mouse1),
    "speed_mouse2": nap.Tsd(t=time_s, d=speed_mouse2),
}
```

Each object appears as a separate feature in the GUI. Individual selection is
not available for pynapple backends.
:::

:::{tab-item} NWB

NWB files represent a single subject per file —
{attr}`~pynwb.file.NWBFile.subject` is a singular
{class}`~pynwb.file.Subject` object
([PyNWB docs](https://pynwb.readthedocs.io/en/stable/pynwb.file.html)).
Multi-subject experiments use separate `.nwb` files per subject. Individual
selection is not available when loading a single NWB file.
:::

::::

### Optional: custom dimensions

Any dimension that co-occurs with a time dimension in at least one feature
variable is automatically discovered and gets a selection
[combo box](https://www.pythonguis.com/docs/qcombobox/) in the GUI.

::::{tab-set}

:::{tab-item} Xarray

```python
ds["emg"] = xr.DataArray(
    emg_data,                            # shape: (time, channels)
    dims=["time", "channels"],
    coords={"channels": ["biceps", "triceps"]},
)
```

Dimensions **do not need to match across features**. For example, `position`
may have `(time, keypoint, space, individual)` while `speed` only has
`(time, keypoint, individual)`. The GUI creates combo boxes for the union of
all discovered dimensions. When a feature doesn't have a selected dimension,
that selection is silently ignored via
{func}`~ethograph.utils.xr_utils.sel_valid`:

```python
import ethograph as eto

# "keypoint" and "individual" are applied; "space" is silently dropped
data, used_kwargs = eto.sel_valid(
    ds["speed"],
    {"keypoint": "nose", "space": "x", "individual": "mouse1"},
)
```
:::

:::{tab-item} Pynapple

Column names in a {class}`~pynapple.TsdFrame` become a selectable dimension.
Objects with identical column names share a single combo in the GUI.

```python
position = nap.TsdFrame(t=time_s, d=pos, columns=["x", "y", "z"])
velocity = nap.TsdFrame(t=time_s, d=vel, columns=["x", "y", "z"])
```
:::

::::

### Optional: color variables

Color variables are identified by **name**: any feature with `"rgb"` in its
name (case-insensitive) is automatically offered in the GUI's **Colors** combo.
Values should lie in `[0, 1]` (float) or `[0, 255]` (int).

::::{tab-set}

:::{tab-item} Xarray

The variable should have an `RGB` dimension of size 3:

```python
ds["angle_rgb"] = xr.DataArray(
    rgb_values,                          # shape: (time, keypoint, individual, 3)
    dims=["time", "keypoint", "individual", "RGB"],
)
```

To compute angle-based RGB automatically from pose data, use
{func}`~ethograph.io.dataset.add_angle_rgb_to_ds`:

```python
import ethograph as eto

ds = eto.add_angle_rgb_to_ds(ds, smoothing_params={"sigma": 3})
```
:::

:::{tab-item} Pynapple

Store RGB as a {class}`~pynapple.TsdFrame` with columns `["R", "G", "B"]` (or
any 3-column frame whose name contains `"rgb"`):

```python
import pynapple as nap

angle_rgb = nap.TsdFrame(
    t=time_s,
    d=rgb_values,                        # shape: (n_time, 3)
    columns=["R", "G", "B"],
)
data = {"angle_rgb": angle_rgb}
```

To compute angle-based RGB automatically from a position
{class}`~pynapple.TsdFrame`, use
{func}`~ethograph.io.pynapple.add_angle_rgb_to_nap`:

```python
from ethograph.io.pynapple import add_angle_rgb_to_nap

angle_rgb = add_angle_rgb_to_nap(position, smoothing_params={"sigma": 3})
data["angle_rgb"] = angle_rgb
```
:::

::::

### Summary

The whole section at a glance:

| | xarray | Pynapple | NWB |
|--|--------|----------|-----|
| **File format** | `.nc` via {class}`~ethograph.io.trialtree.TrialTree` | `.npz` or folder | `.nwb` |
| **Required attrs** | `trial`, `fps` | *(none)* | *(NWB standard)* |
| **Features** | Any `data_var` with a time dim | Any {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` | {class}`~pynwb.TimeSeries` in processing |
| **Individuals** | `coords["individual"]` | Separate objects | One subject per file |
| **Trials** | One `Dataset` per trial | {class}`~pynapple.IntervalSet` | `nwb.trials` table |

---

## Trials

### One dataset per trial

Build a {class}`xarray.Dataset` per trial, then combine them with
{func}`eto.from_datasets() <ethograph.from_datasets>`:

```python
import numpy as np
import xarray as xr
import ethograph as eto

datasets = []
for trial_id in range(1, 6):
    n_time = 9000                                   # 5 min at 30 fps
    ds = xr.Dataset(
        {"speed": xr.DataArray(
            np.random.randn(n_time),
            dims=["time"],
            coords={"time": np.arange(n_time) / 30.0},
        )},
    )
    ds.attrs["trial"] = trial_id
    ds.attrs["fps"] = 30.0
    datasets.append(ds)

dt = eto.from_datasets(datasets)
dt.save("session.nc")
```

Extra attributes such as `stimulus` become per-trial metadata and flow through
to label TSV exports.

For pynapple, trials are an {class}`~pynapple.IntervalSet` saved alongside the
features rather than separate objects:

```python
import pynapple as nap

trials = nap.IntervalSet(
    start=[i * 300.0 for i in range(5)],
    end=[(i + 1) * 300.0 - 0.5 for i in range(5)],
)
nap.save_file({"speed": speed, "trials": trials}, "session")
```

(target-from-continuous)=
### Splitting a continuous recording into trials

If you have a single session-long `xr.Dataset` and want to parcelate it into a
trial structure, use {func}`eto.from_continuous() <ethograph.from_continuous>`:

```python
import numpy as np
import pandas as pd
import xarray as xr
import ethograph as eto

# A continuous 10-minute recording at 30 fps
n_samples = 18000
time = np.arange(n_samples) / 30.0

ds = xr.Dataset({
    "speed": xr.DataArray(np.random.randn(n_samples), dims=["time"],
                          coords={"time": time}),
})

# Define trial boundaries (seconds)
trials = pd.DataFrame({
    "trial": [1, 2, 3],
    "start_time": [0.0, 120.0, 300.0],
    "stop_time": [100.0, 250.0, 500.0],
})

dt = eto.from_continuous(ds, trials)
dt.save("session.nc")

dt.trial(2)  # returns the 120–250 s slice, time shifted to start at 0
```

`from_continuous` slices the dataset on demand and shifts time coordinates to 0
for each trial.

---

(target-nwb-alignment)=
## Alignment

Media filenames, trial timing and stream offsets live in an **NWB alignment
file**, not inside the data file. This keeps data portable — filenames are
stored as basename only — and lets you move media without re-exporting
features.

For `.nwb` sources the source file is used directly and edits go back into it.
Every other format (`.nc`, `.npz`, pynapple folders) gets a sidecar at
**`.ethograph/alignment.nwb`** next to the data file.

For single-trial recordings you never write this yourself —
{ref}`drag & drop <target-data-loading>` builds it for you. The rest of this
section is for multi-trial, multi-camera or session-wide media.

### What it contains

| Concept | Stored as | Read via |
|---------|-----------|----------|
| **Trial timing** | `nwb.trials` table with `start_time`, `stop_time`, and custom columns | `alignment.trials_df`, `alignment.start_time(trial)`, `alignment.stop_time(trial)` |
| **Media files** | {class}`~pynwb.image.ImageSeries` in `nwb.acquisition` per stream/device | `alignment.resolve_media_path(trial, stream, device)` |
| **Stream rates** | `rate` field on each ImageSeries | `alignment.get_stream_rate(stream, device)` |
| **Stream offsets** | `starting_time` on ImageSeries — when sample 0 occurs in session time | `alignment.stream_offset_for_trial(trial, stream, device)` |
| **Cameras / mics** | Device names parsed from ImageSeries names | `alignment.cameras`, `alignment.mics` |

Streams are named `{stream}_{device}` throughout — `video_cam-1`, `audio_mic-1`,
`pose_cam-1`, `ephys_probe-1`.

### Trial-relative vs session-absolute time

EthoGraph uses two time conventions, and the alignment file connects them:

- **Trial-relative** (`onset_s`, `offset_s`, internal feature time): each trial
  starts at `0.0`. This matches pose trackers, video files, and per-trial audio.
- **Session-absolute** (`onset_global`, `offset_global`, ephys timestamps):
  measured from the start of the recording session.

Conversion uses the trials table:
`onset_global = alignment.start_time(trial) + onset_s`.

### Choosing a builder

| Function | When to use |
|----------|-------------|
| {func}`~ethograph.io.nwb_alignment.align_media_per_trial` | Media files map 1:1 to trials |
| {func}`~ethograph.io.nwb_alignment.align_media_from_streams` | Session-wide files, mixed per-trial + continuous, or explicit timestamps |

### `align_media_per_trial` — one file per trial

One column per stream, one row per trial. This example covers two cameras, a
pose file per camera, two microphones and explicit trial timing:

```python
import pandas as pd
import ethograph as eto

trial_table = pd.DataFrame({
    "trial": [1, 2],
    "start_time": [0.0, 300.0],
    "stop_time":  [299.5, 599.5],
    "video_cam-1": ["cam1_t1.mp4", "cam1_t2.mp4"],
    "video_cam-2": ["cam2_t1.mp4", "cam2_t2.mp4"],
    "pose_cam-1":  ["dlc_cam1_t1.h5", "dlc_cam1_t2.h5"],
    "pose_cam-2":  ["dlc_cam2_t1.h5", "dlc_cam2_t2.h5"],
    "audio_mic-1": ["mic1_t1.wav", "mic1_t2.wav"],
    "audio_mic-2": ["mic2_t1.wav", "mic2_t2.wav"],
    "stimulus":    ["tone_A", "tone_B"],
})

eto.align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0, "pose": 30.0, "audio": 48000.0},
    output_path=".ethograph/alignment.nwb",
)
```

{func}`~ethograph.io.nwb_alignment.align_media_per_trial`,
{func}`~ethograph.io.nwb_alignment.align_media_from_streams` and
{class}`~ethograph.io.nwb_alignment.NWBAlignment` are all re-exported at the top
level, so `eto.align_media_per_trial` and
`from ethograph.io.nwb_alignment import align_media_per_trial` are the same
function — use whichever you prefer.

Notes:

- **Files are paired by row, not by name.** Nothing scans a folder: row order
  is trial order and each `{stream}_{device}` column is that stream's file for
  that trial. Only the basename is stored; filename matching happens later, at
  load time, when the GUI resolves each basename against the media folder you
  select.
- **Camera index pairs video with pose**: device `cam-1` overlays `pose_cam-1`.
- **Extra columns** (`stimulus`, `condition`, …) become trial attributes and
  flow through to label TSV exports.
- **Multi-camera NWB files** need no table: each camera is already a separate
  {class}`~pynwb.image.ImageSeries` in `nwb.acquisition`, discovered
  automatically.

(target-omitting-trial-times)=
#### Omitting `start_time` / `stop_time`

Both columns are optional. Leave them out and each trial's duration is probed
from its own media file (video first, then audio, then pose), with trials laid
**end to end** starting at `0.0`:

```python
eto.align_media_per_trial(
    trial_table.drop(columns=["start_time", "stop_time"]),
    stream_rates={"video": 30.0, "pose": 30.0, "audio": 48000.0},
    output_path="session_01/.ethograph/alignment.nwb",
    media_root="session_01/video",     # needed to open the files for probing
)
```

Three things to know before relying on this:

- **The files must be openable at build time.** Probing opens the media, so
  either give absolute paths in the table or pass `media_root`; a filename that
  doesn't resolve to an existing file raises `ValueError`. This is the one place
  the builder needs the actual media rather than just its name.
- **Inferred trials are contiguous.** Trial 2 starts exactly where trial 1
  ends, so the inter-trial gaps of the real recording are erased. Trial-relative
  time (labels, features, per-trial video) is unaffected, but session-absolute
  time is fiction — pass the real `start_time` / `stop_time` whenever you have
  ephys on a session clock, session-wide media, or session-mode navigation to
  support.
- **Pose-only tables need a rate.** A pose file has no intrinsic duration, so
  probing it also requires `pose_fps=`. Tables with no `video_*`, `audio_*` or
  `pose_*` column at all cannot infer anything and raise.

When present, `start_time` / `stop_time` also enable session-mode navigation and
let the GUI restrict neural data to trial windows.

(target-session-wide-streams)=
### `align_media_from_streams` — session-wide or mixed

Use this when media doesn't map 1:1 to trials: one continuous audio file across
all trials, ephys on a separate clock, or explicit DAQ timestamps.

```python
import pandas as pd
import ethograph as eto

trials = pd.DataFrame({
    "trial": [1, 2, 3],
    "start_time": [0.0, 300.0, 600.0],
    "stop_time": [299.5, 599.5, 899.5],
})

streams = [
    # Per-trial: one file per trial, as above
    {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4", "t3.mp4"], "rate": 30.0},
    # Session-wide: one file; starting_time marks when it begins in session time
    {"name": "audio_mic-1", "files": ["session_ch1.wav"], "rate": 48000.0, "starting_time": 0.0},
    # Ephys on its own clock, starting 0.5 s after the behavioural reference
    {"name": "ephys_probe-1", "files": ["session.dat"], "rate": 30000.0, "starting_time": 0.5},
]

eto.align_media_from_streams(trials, streams, ".ethograph/alignment.nwb")
```

Each entry in `streams` accepts:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Stream identifier: `{stream}_{device}` (e.g. `video_cam-1`) |
| `files` | `list[str]` | One file (session-wide) or one per trial |
| `rate` | `float` | Sampling rate in Hz |
| `starting_time` | `float` | When the stream begins in session time. Default 0.0 |
| `timestamps` | `ndarray` | Explicit per-sample timestamps. Overrides `rate` when clocks drift |

Ephys is always session-wide: select the file in the GUI rather than embedding
it in the dataset. See {doc}`loading_ephys` for supported formats, Kilosort
folder setup and channel mapping.

### Reading an existing alignment file

```python
from ethograph.io.nwb_alignment import NWBAlignment

alignment = NWBAlignment("session_01/.ethograph/alignment.nwb")
print(alignment.trials_df)
print(alignment.cameras)          # ["cam-1", "cam-2"]
print(alignment.mics)             # ["mic-1"]
print(alignment.start_time(1))    # 0.0
alignment.close()
```

The same interface is exposed via `dt.nwb_alignment` on a loaded TrialTree and
via `app_state.nwb_alignment` inside the GUI. See
{class}`~ethograph.io.nwb_alignment.NWBAlignment` for the full API.

---

## Putting it together

A complete two-camera, ten-trial setup:

::::{tab-set}

:::{tab-item} Xarray

```python
import numpy as np
import pandas as pd
import xarray as xr
import ethograph as eto

# 1) One Dataset per trial
datasets = []
for trial_id in range(1, 11):
    n_time = 9000                                   # 5 minutes at 30 fps
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

# 2) Alignment: media files + trial timing
trial_table = pd.DataFrame({
    "trial": list(range(1, 11)),
    "start_time": [i * 300.0 for i in range(10)],
    "stop_time": [(i + 1) * 300.0 - 0.5 for i in range(10)],
    "video_cam-1": [f"cam1_trial{tid:03d}.mp4" for tid in range(1, 11)],
    "video_cam-2": [f"cam2_trial{tid:03d}.mp4" for tid in range(1, 11)],
    "pose_cam-1":  [f"dlc_cam1_trial{tid:03d}.h5" for tid in range(1, 11)],
    "pose_cam-2":  [f"dlc_cam2_trial{tid:03d}.h5" for tid in range(1, 11)],
})

eto.align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0, "pose": 30.0},
    output_path=".ethograph/alignment.nwb",
)
```

Then launch EthoGraph, select `session.nc` in the **Custom set-up** card, and
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

## Folder structure

Three folders play a part, and only one of them is yours to arrange:

| Folder | Who writes it | What lives there |
|---|---|---|
| **Session folder** — one per recording | You: the session file and the media. The GUI: labels, alignment, layout. | The data. Any location; media folders are selected in the GUI and can live elsewhere. |
| **Project folder** — one per study | You, on the start page. | Everything that spans sessions: the label vocabulary, pipeline configs, trained models, curation workflows, kept drag & drops. **Never the data** — nothing is copied into it. |
| `~/.ethograph/` | The GUI. | Your settings, caches, and a starter project used while no project folder is chosen. |

### The project folder

Chosen once on the start page (**Project folder**) and remembered across
restarts. Sessions are *listed* from wherever they are; the folder holds what
you build on top of them:

```
my_study/                              # chosen on the start page
    ├── mapping.txt                    # the study's label_id → name vocabulary
    ├── config/
    │   ├── segment.yaml               # action-segmentation config (copy from ~/.ethograph/defaults/config/)
    │   ├── spot.yaml                  # pixel event-spotting config
    │   └── space/                     # reference geometries for the Space plot
    ├── runs/
    │   └── lightgbm/                  # onset models trained from the Model menu
    ├── workflows/                     # curation workflows
    ├── wizard/                        # alignment-wizard notebooks
    └── sessions/                      # drag & drops made with this project set
        └── 2026-09-06_21-47-12/       # one timestamped folder per drop, reopenable
```

Without a project folder, `~/.ethograph/defaults/` stands in — it has the same
shape and ships with a default `mapping.txt`, example configs and geometries.
See {ref}`target-label-mapping` for how the mapping is resolved between the
session, the project and that backup.

### The session folder

One per recording, per backend:

::::{tab-set}

:::{tab-item} xarray (.nc)

```
session_01/
    ├── session.nc                     # Behavioural dataset (TrialTree or plain Dataset)
    ├── session_labels.tsv             # Session labels
    ├── session_metadata.tsv           # Trial-level metadata
    ├── .ethograph/
    │   ├── alignment.nwb              # Media paths, trial timing, stream offsets
    │   ├── local_settings.yaml        # Session-specific GUI state
    │   └── mapping.txt                # Optional: overrides the project's for this session
    │
    ├── labels/
    │   ├── backups/
    │   │  └── session_labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215/
    │       ├── trial1.npy
    │       └── trial2.npy
    │
    ├── video/
    │   ├── camera1_trial001.mp4
    │   └── camera2_trial001.mp4
    ├── pose/                          # External pose files (DLC, SLEAP, ...)
    │   ├── trial001_pose.h5
    │   └── ...
    ├── audio/
    │   ├── mic1_trial001.wav
    │   └── ...
    └── ephys/
        ├── recording.rhd
        └── kilosort4/
            ├── params.py
            ├── spike_times.npy
            ├── spike_clusters.npy
            ├── channel_positions.npy
            ├── channel_map.npy
            ├── templates.npy
            └── cluster_info.tsv
```
:::

:::{tab-item} NWB (.nwb)

```
session_01/
    ├── session.nwb                    # Self-contained: trials, time series,
    │                                  # pose (PoseEstimationSeries), video
    │                                  # refs (ImageSeries.external_file)
    │
    ├── session_labels.tsv             # Session labels
    ├── session_metadata.tsv           # Trial-level metadata
    │
    ├── .ethograph/
    │   ├── alignment.nwb              # inherit/overwrite alignment in session.nwb
    │   ├── local_settings.yaml        # Session-specific GUI state
    │   └── mapping.txt                # Optional: overrides the project's for this session
    │
    ├── labels/
    │   ├── backups/
    │   │   └── session_labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215/
    │       ├── trial1.npy
    │       └── trial2.npy
    │
    └── video/                         # Only needed if ImageSeries paths
        ├── camera1_trial001.mp4       # no longer point to the right location
        └── camera2_trial001.mp4
```

Pose estimation and other behavioural time series are stored inside the `.nwb`
file — no external tracking folder needed.
:::

:::{tab-item} Pynapple (.npz / folder)

```
session_01/
    ├── position.npz                   # Pynapple Tsd/TsdFrame objects
    ├── speed.npz
    ├── units.npz                      # TsGroup of spike times
    │
    ├── labels.tsv                     # Session labels
    ├── metadata.tsv                   # Trial-level metadata
    │
    ├── .ethograph/
    │   ├── alignment.nwb              # Media paths, trial timing, stream offsets
    │   ├── local_settings.yaml        # Session-specific GUI state
    │   └── mapping.txt                # Optional: overrides the project's for this session
    │
    ├── labels/
    │   ├── backups/
    │   │   └── labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215/
    │       ├── trial1.npy
    │       └── trial2.npy
    │
    ├── video/
    │   ├── camera1_trial001.mp4
    │   └── ...
    └── audio/
        ├── mic1_trial001.wav
        └── ...
```
:::

::::

You write the session file and the media folders. Everything under
`.ethograph/` and `labels/backups/` is created by the GUI on first load and
first save.

### The home folder

```
~/.ethograph/
    ├── gui_settings.yaml              # your layout, playback and dialog folders
    ├── logs/
    ├── cache/                         # video proxies, extracted audio, example data — safe to delete
    └── defaults/                      # the starter project: mapping.txt, config/, …
```

---

## Operations across backends

Once the data exists, these are the equivalent calls for working with it in
code:

| Operation | xarray / TrialTree | Pynapple | NWB |
|-----------|-------------------|----------|-----|
| **Load** | {func}`~ethograph.open` | {func}`~ethograph.load_nap_data` | {func}`~ethograph.load_nap_data` |
| **Restrict** | {meth}`~xarray.Dataset.sel` | {meth}`~pynapple.Tsd.restrict` | Via {mod}`pynapple` |
| **Select dims** | {func}`~ethograph.sel_valid` | Column indexing on {class}`~pynapple.TsdFrame` | Via {mod}`pynapple` |
| **Build** | {func}`~ethograph.from_datasets` | {func}`~pynapple.save_file` | {class}`~pynwb.NWBHDF5IO` |
| **Alignment** | `.ethograph/alignment.nwb` | `.ethograph/alignment.nwb` | In source `.nwb` |
| **Iterate trials** | {meth}`~ethograph.io.trialtree.TrialTree.trial_items` | {meth}`~pynapple.Tsd.restrict` | Via {mod}`pynapple` |

---

## References

- {doc}`../api/trialtree` — `from_datasets()`, `from_continuous()`, timing, iteration
- {doc}`loading_ephys` — ephys formats, Kilosort, channel mapping
- {class}`~ethograph.io.nwb_alignment.NWBAlignment` — alignment reader API
- {func}`~ethograph.io.nwb_alignment.align_media_per_trial` — per-trial builder
- {func}`~ethograph.io.nwb_alignment.align_media_from_streams` — session-wide builder
