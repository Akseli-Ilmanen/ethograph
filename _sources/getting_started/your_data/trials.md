# Trials

## One dataset per trial

Build a {class}`xarray.Dataset` per trial, then combine them with
{func}`eto.from_datasets() <ethograph.from_datasets>`:

```python
import numpy as np
import xarray as xr
import ethograph as eto

datasets = []
for trial_id in range(1, 6):
    n_time = 9000  # 5 min at 30 fps
    ds = xr.Dataset(
        {
            "speed": xr.DataArray(
                np.random.randn(n_time),
                dims=["time"],
                coords={"time": np.arange(n_time) / 30.0},
            )
        },
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
## Splitting a continuous recording into trials

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

ds = xr.Dataset(
    {
        "speed": xr.DataArray(np.random.randn(n_samples), dims=["time"], coords={"time": time}),
    }
)

# Define trial boundaries (seconds)
trials = pd.DataFrame(
    {
        "trial": [1, 2, 3],
        "start_time": [0.0, 120.0, 300.0],
        "stop_time": [100.0, 250.0, 500.0],
    }
)

dt = eto.from_continuous(ds, trials)
dt.save("session.nc")

dt.trial(2)  # returns the 120–250 s slice, time shifted to start at 0
```

`from_continuous` slices the dataset on demand and shifts time coordinates to 0
for each trial.

See {doc}`../../api/trialtree` for the full TrialTree API.

Training a model on your trials? See {doc}`../../models/trial_windows` for how
long to make them and why they don't overlap.
