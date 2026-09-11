(target-data-requirements)=
(target-multi-trial)=
# Preparing your own data

{ref}`Drag & drop <target-data-loading>` handles one recording session whose
files all start together. Everything else — trials split across files, media on
separate clocks — needs a **session file** plus an **alignment file**, built
in the {doc}`Data wizard <../advanced/data_wizard>` or from a short Python
script.

This page covers all three pieces:

1. **[Your dataset](#your-dataset)** — the schema EthoGraph expects.
2. **[Trials](#trials)** — grouping datasets into a trial structure.
3. **[Pairing and alignment](#pairing-and-alignment)** — tying media files and
   timing to those trials.

EthoGraph supports three backends. Pick the one matching your workflow; every
section below has a tab per backend.

| Backend | Best for | Core object |
|---------|----------|-------------|
| **xarray** | Custom datasets, pose estimation, multi-dim arrays | {class}`xarray.Dataset` / {class}`~ethograph.io.trialtree.TrialTree` |
| **Pynapple** | Neuroscience time-series, NWB interop | {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` / {class}`~pynapple.TsGroup` |
| **NWB** | Standardised neurodata, DANDI archives | `.nwb` file (loaded via pynapple) |

```{note}
**NWB needs almost none of this.** An `.nwb` file already stores trials, media
references and features together, so it loads directly. The NWB tabs below only
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
## Pairing and alignment

You have media files (video, audio, pose), possibly from several cameras or
microphones. If they are already aligned in time, pair them. Otherwise align
them to a recording system's clock with neuroconv.

| Mode | What you have | Where it happens | What you get |
|---|---|---|---|
| **1 Pair my media files** | Files that already share a clock: one file per trial, or session-wide files whose start you know | The {doc}`Data wizard <../advanced/data_wizard>`, or {func}`~ethograph.discover_media` + {func}`~ethograph.pair_media` in Python | Trial time exact. Session time laid end to end from the media durations, or as good as your number for `session_wide` offsets |
| **2 Align media to a recording system, free-running camera** | A camera that ran for the whole session, and a recording system (Intan, Open Ephys, ...) that logged its frames or its start | A notebook the wizard writes; you run it with neuroconv | Every frame on the recording clock |
| **3 Align media to a recording system, triggered camera** | One file per trial, each started by a pulse from the recording system | Same as mode 2 | Every frame on the recording clock |

Modes 2 and 3 produce a `session.nwb`. Mode 1 and the drag & drop produce a
sidecar `.ethograph/alignment.nwb` next to the data file. An `.nwb` source is
read and edited directly and needs no sidecar; if `nwb.trials` is absent the
recording is one trial, and a DANDI file without local media paths gets a
sidecar on first load.

### Pair my media files

On the start page, click **Data wizard, prepare my data** and choose
**1 Pair my media files**. Tell it how many cameras and microphones you have
and where the files are. It builds the pairing table, writes
`.ethograph/alignment.nwb`, and saves a notebook with the same calls under
`wizard/` in the project folder. Every page of the wizard is described in
{doc}`../advanced/data_wizard`.

The same thing in Python is two calls. {func}`~ethograph.discover_media` builds
the pairing table from folders (files in natural sort order) or from a filename
pattern with named groups `trial`, `camera` and `mic`.
{func}`~ethograph.pair_media` writes the alignment file:

```python
import ethograph as eto

session_dir = "session_01"

sources = [
    eto.SourceSpec("video", device="cam-1", folder="video/cam1"),
    eto.SourceSpec("video", device="cam-2", folder="video/cam2"),
    eto.SourceSpec("pose", device="cam-1", folder="pose/cam1"),
    eto.SourceSpec("pose", device="cam-2", folder="pose/cam2"),
    eto.SourceSpec("audio", device="mic-1", folder="audio"),
]
trial_table = eto.discover_media(session_dir, sources)
print(trial_table)
#    trial  video_cam-1  video_cam-2      pose_cam-1  ...  audio_mic-1
# 0      1  cam1_t1.mp4  cam2_t1.mp4  dlc_cam1_t1.h5  ...  mic1_t1.wav
# 1      2  cam1_t2.mp4  cam2_t2.mp4  dlc_cam1_t2.h5  ...  mic1_t2.wav

trial_table["stimulus"] = ["tone_A", "tone_B"]

eto.pair_media(
    trial_table,
    stream_rates={"video": 30.0, "pose": 30.0, "audio": 48000.0},
    output_path=f"{session_dir}/.ethograph/alignment.nwb",
    media_root=session_dir,
)
```

When one folder holds every camera, give a pattern instead of a folder:
`eto.SourceSpec("video", pattern=r"cam(?P<camera>\d+)_t(?P<trial>\d+)\.mp4")`.
The `camera` group becomes the device (`cam-1`, `cam-2`) and the `trial` group
the row.

The pairing table is a plain {class}`~pandas.DataFrame` with a `trial` column
and one `{stream}_{device}` column per source. You can build it by hand or edit
it before writing.

- **Files are paired by row, not by name.** Row order is trial order and each
  `{stream}_{device}` column is that stream's file for that trial. Only the
  basename is stored; it is resolved against the media folder you select in the
  GUI at load time.
- **Camera index pairs video with pose**: device `cam-1` overlays `pose_cam-1`.
- **Extra columns** (`stimulus`, `condition`, ...) become trial attributes and
  flow through to label TSV exports.
- **Multi-camera NWB files** need no table: each camera is already its own
  {class}`~pynwb.image.ImageSeries` in `nwb.acquisition`.

(target-omitting-trial-times)=
#### Trial times

`start_time` and `stop_time` columns are optional. Without them each trial's
duration is probed from its own media (video first, then audio, then pose) and
the trials are laid **end to end** from `0.0`. Three things follow:

- **The files must be openable at build time.** Pass `media_root` or absolute
  paths; a name that does not resolve raises `ValueError`.
- **Inferred trials are contiguous.** The inter-trial gaps of the real
  recording are erased. Trial-relative time (labels, features, per-trial video)
  is unaffected, but session time is fiction. Pass real times whenever you have
  ephys, session-wide media or session-mode navigation to support.
- **Pose-only tables need `pose_fps=`.** A pose file has no intrinsic duration.

With real `start_time` / `stop_time` the GUI can also navigate in session mode
and restrict neural data to trial windows.

(target-session-wide-streams)=
#### Session-wide streams

A file that spans every trial (one continuous audio recording, a probe on its
own clock) goes in `session_wide`, keyed by stream name, with its rate and the
session time at which its first sample was taken:

```python
eto.pair_media(
    trial_table,
    stream_rates={"video": 30.0},
    session_wide={
        "audio_mic-1": ("session_ch1.wav", 48000.0, 0.0),
        "ephys_probe-1": ("session.dat", 30000.0, 0.5),
    },
    output_path=f"{session_dir}/.ethograph/alignment.nwb",
)
```

Session time is then as good as the offset you typed. If you measured the
offset from a sync line, use mode 2 or 3 instead and let neuroconv place every
frame.

If `output_path` is an existing `.nwb` with a trials table, such as a file
neuroconv wrote, the streams are added to it in place. That is how the pose
and audio of modes 2 and 3 join the video.

{func}`~ethograph.io.nwb_alignment.align_media_per_trial` and
{func}`~ethograph.io.nwb_alignment.align_media_from_streams` remain as
compatibility aliases of {func}`~ethograph.pair_media`. The old `timestamps`
key is gone: per-sample timestamps are neuroconv's job.

Ephys is always session-wide: select the file in the GUI rather than listing it
in the table. See {doc}`loading_ephys` for supported formats, Kilosort folder
setup and channel mapping.

### Align to a recording system

When a recording system (Intan, Open Ephys, SpikeGLX, ...) logged the camera,
its clock is the session clock and the video should sit on it frame by frame.
That is what [neuroconv](https://neuroconv.readthedocs.io) does. A
free-running camera starts with the session and stops with it, in one file or
split into several:

```{figure} ../_static/neuroconv/video_setup_free_running.png
:alt: A free-running camera, as one file or split into several
:width: 90%

Free-running camera. Figure from neuroconv's how-to (BSD-3-Clause).
```

A triggered camera receives a pulse at each trial onset and writes one file per
trial, with gaps between them:

```{figure} ../_static/neuroconv/video_setup_triggered.png
:alt: A triggered camera, one file per trial
:width: 90%

Triggered camera. Figure from neuroconv's how-to (BSD-3-Clause).
```

Choose **2** or **3** in the Data wizard. It asks how the camera is wired (a
known offset, a pulse per frame, or a pulse per trial), writes
`wizard/{rig_name}.ipynb` in the project folder and stops. Run the notebook; it
writes `session.nwb`, which you then open on the start page. The recipes are
neuroconv's own, from its
[how-to on aligning external video](https://neuroconv--2037.org.readthedocs.build/en/2037/how_to/align_external_video.html);
{doc}`video_and_ephys` walks one rig through end to end. Then pair the rest
(pose, audio) over the `.nwb` with {func}`~ethograph.pair_media`, as above.

### What the alignment file contains

| Concept | Stored as | Read via |
|---------|-----------|----------|
| **Trial timing** | `nwb.trials` with `start_time`, `stop_time` and custom columns | `alignment.trials_df`, `alignment.start_time(trial)`, `alignment.stop_time(trial)` |
| **Media files** | one {class}`~pynwb.image.ImageSeries` per `{stream}_{device}` in `nwb.acquisition` | `alignment.resolve_media_path(trial, stream, device)` |
| **Stream rates** | `rate` on each ImageSeries | `alignment.get_stream_rate(stream, device)` |
| **Stream offsets** | `starting_time` on each ImageSeries: when sample 0 occurs in session time | `alignment.stream_offset_for_trial(trial, stream, device)` |
| **Cameras / mics** | device names parsed from the ImageSeries names | `alignment.cameras`, `alignment.mics` |

Streams are named `{stream}_{device}` throughout: `video_cam-1`, `audio_mic-1`,
`pose_cam-1`, `ephys_probe-1`. Only basenames are stored, so media can move
without re-exporting features.

Two time conventions meet here. **Trial-relative** time (`onset_s`, `offset_s`,
feature time) starts at `0.0` in every trial, like pose trackers and per-trial
video. **Session** time (`onset_global`, ephys timestamps) is measured from the
start of the recording. The trials table converts between them:
`onset_global = alignment.start_time(trial) + onset_s`.

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

# 2) Pairing: media files + trial timing
sources = [
    eto.SourceSpec("video", device="cam-1", folder="video/cam1"),
    eto.SourceSpec("video", device="cam-2", folder="video/cam2"),
    eto.SourceSpec("pose", device="cam-1", folder="pose/cam1"),
    eto.SourceSpec("pose", device="cam-2", folder="pose/cam2"),
]
trial_table = eto.discover_media(".", sources)       # 10 rows, natural sort order
trial_table["start_time"] = [i * 300.0 for i in range(10)]
trial_table["stop_time"] = [(i + 1) * 300.0 - 0.5 for i in range(10)]

eto.pair_media(
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
    ├── wizard/                        # Data wizard notebooks, one per rig
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
- {doc}`video_and_ephys` — video on a recorder's clock through neuroconv
- {doc}`../advanced/data_wizard` — the Data wizard, page by page
- {class}`~ethograph.io.nwb_alignment.NWBAlignment` — alignment reader API
- {func}`~ethograph.discover_media` — the pairing table from folders or a filename pattern
- {func}`~ethograph.pair_media` — writes the alignment file (`align_media_per_trial` / `align_media_from_streams` are aliases)
