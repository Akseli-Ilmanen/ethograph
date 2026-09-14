(target-kinematic-changepoints)=
# Kinematic changepoints

One example of changepoints occurring at behavioural boundaries are speed
minima. In point-to-point reaching tasks, the hand speed profile is
characterised by a unimodal bell-shaped curve with two speed minima
{cite:p}`torricelli2023motor`, where the minima correspond to the onset and
offset of the movement, and the peak of the speed bump marks the point where
the hand starts decelerating. These minima can even be used to identify
sub-movements, such as the small second corrective speed bump in the figure
below {cite:p}`meyer1988optimality`.

![Kinematic changepoints](../../_static/media/changepoints1.png)

---

## Troughs (local minima)

Finds local minima in the signal using `scipy.signal.find_peaks` applied to
the negated signal. Troughs in speed often correspond to moments where an
animal pauses or reverses direction.

Parameters are passed directly to {func}`scipy.signal.find_peaks`: `height`,
`distance`, `prominence`, `width`, etc.

---

## Turning points

A custom algorithm that identifies the boundaries of peak regions rather than
the peaks themselves. The idea is that behaviour transitions happen not at
the peak of a movement, but where the animal starts or stops accelerating.

![Turning points](../../_static/media/changepoints2.png)

The algorithm works in four steps:

1. **Compute the gradient** of the signal and find all indices where `|gradient| < threshold` — these are candidate turning points (near-stationary regions).
2. **Find peaks** in the original signal using `scipy.signal.find_peaks` with `prominence` and `width` parameters.
3. **For each peak**, select the closest candidate turning point to its left and right. These define the boundaries of the peak region.
4. **Filter by `max_value`**: any candidate turning point where the signal exceeds `max_value` is discarded. This prevents selecting turning points on high speed plateaus.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `threshold` | Maximum absolute gradient to qualify as a turning point. Lower = only very flat regions. |
| `max_value` | Discard turning points where signal exceeds this value. |
| `prominence` | Minimum peak prominence (passed to `find_peaks`). |
| `distance` | Minimum distance between peaks (passed to `find_peaks`). |

All kinematic changepoints also include NaN-boundary markers — transitions
between valid data and NaN gaps are automatically added as changepoints.

---

## Usage

1. Select a feature in the Data Controls (e.g. `speed`).
2. Open the **Kinematic CPs** panel.
3. Choose a method (`troughs` or `turning_points`).
4. Click **Configure...** to adjust parameters.
5. Click **Detect**.

The changepoints are stored in the dataset as `{feature}_{method}` (e.g.
`speed_troughs`) and persist when you save.

---

## Data format

::::{tab-set}

:::{tab-item} Xarray

Changepoint arrays are binary (`0` or `1`) integer arrays that share the
same time dimension as their target feature. They require:

- `attrs["kind"] = "changepoint_feature"` and `attrs["changepoint_mask"] = 1` —
  the label and the marker, both written by
  {func}`~ethograph.io.schema.changepoint_attrs`.
  A file predating this needs {func}`~ethograph.io.schema.migrate_legacy_attrs`.
- `attrs["target_feature"]` — name of the feature variable they annotate

```python
from ethograph.io import schema

ds["speed_troughs"] = xr.DataArray(
    cp_binary,                           # shape: (time, keypoint, individual), values 0 or 1
    dims=["time", "keypoint", "individual"],
    attrs=schema.changepoint_attrs(target_feature="speed"),
)
```

To compute changepoints programmatically, use
{func}`~ethograph.io.dataset.add_changepoints_to_ds`:

```python
import ethograph as eto
from ethograph.features.changepoints import find_troughs_binary

ds = eto.add_changepoints_to_ds(
    ds,
    target_feature="speed",
    changepoint_name="troughs",
    changepoint_func=find_troughs_binary,
)
```

{func}`~ethograph.io.dataset.add_changepoints_to_ds` uses
{func}`xarray.apply_ufunc` with `vectorize=True`, so your detection function
only needs to handle a 1-D signal.
:::

:::{tab-item} Pynapple

For pynapple data, changepoints are stored as a
{class}`~pynapple.TsGroup` — one timestamp series per unit (keypoint,
channel, neuron, ...) with metadata columns marking it as a changepoint set.

Use {func}`~ethograph.io.pynapple.add_changepoints_to_nap`:

```python
import pynapple as nap
from ethograph.io.pynapple import add_changepoints_to_nap
from ethograph.features.changepoints import find_troughs_binary

speed = nap.TsdFrame(
    t=time_s,
    d=speed_array,                       # shape: (n_time, n_keypoints)
    columns=["nose", "left_ear", "right_ear", "tail"],
)

cps = add_changepoints_to_nap(
    speed,
    target_feature="speed",
    changepoint_func=find_troughs_binary,
    prominence=0.1,
)
# cps is a TsGroup — one Ts per column with:
#   metadata.source_label = column name (e.g. "nose")
#   metadata.target_feature = "speed"
#   metadata.kind = "changepoint_feature"   # the same names the xarray side
#   metadata.changepoint_mask = 1           # writes as attrs
```

Accepts {class}`~pynapple.Tsd`, {class}`~pynapple.TsdFrame`, or
{class}`~pynapple.TsGroup` as input. Existing metadata columns are carried
through when the input is a `TsGroup`.
:::

::::

---
