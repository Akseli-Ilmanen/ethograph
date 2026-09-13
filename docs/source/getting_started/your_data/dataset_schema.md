# Dataset schema

## Minimal working example

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

## Required attributes

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

## Features (plottable variables)

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

## Specifying individuals

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

## Optional: custom dimensions

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

## Optional: color variables

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

## Summary

The whole page at a glance:

| | xarray | Pynapple | NWB |
|--|--------|----------|-----|
| **File format** | `.nc` via {class}`~ethograph.io.trialtree.TrialTree` | `.npz` or folder | `.nwb` |
| **Required attrs** | `trial`, `fps` | *(none)* | *(NWB standard)* |
| **Features** | Any `data_var` with a time dim | Any {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` | {class}`~pynwb.TimeSeries` in processing |
| **Individuals** | `coords["individual"]` | Separate objects | One subject per file |
| **Trials** | One `Dataset` per trial | {class}`~pynapple.IntervalSet` | `nwb.trials` table |

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
