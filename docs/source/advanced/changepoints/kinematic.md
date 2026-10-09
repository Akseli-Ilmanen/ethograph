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

Detection lives in the top bar: **Changepoints ▸ Detect changepoints…**.
Correction is the next entry, **Run changepoint correction…**, and opens a
pop-up of its own, so the two steps never share a window.

1. Click a line plot and pick the feature in the sidebar's **Data** section (e.g. `speed`).
2. Open **Changepoints ▸ Detect changepoints…** in the top bar; the **Kinematic CPs** panel is in front.
3. Choose a method (`troughs` or `turning_points`).
4. Click **Configure...** to adjust parameters.
5. Click **Detect**.

The changepoints are stored in the dataset as `{feature}_{method}` (e.g.
`speed_troughs`) and persist when you save.

```{tip}
**Play in the GUI, then script it.** The pop-up is for finding a threshold
interactively: change a parameter, click **Detect**, look at the marks on the
plot. Once they look right, click **Copy code to clipboard** in the
**Configure...** dialog: it copies the Python call with the parameters you
chose. Put that call in the script that builds your dataset, so every session
is detected the same way and nobody has to redo the clicks.
```

---

## Data format

::::{tab-set}

:::{tab-item} Xarray

Compute changepoints with
{func}`~ethograph.io.dataset.add_changepoints_to_ds`. It runs the detector
over every non-time dimension through {func}`xarray.apply_ufunc` with
`vectorize=True`, so the detector only has to handle a 1-D signal, and it
stores the result as `ds["{target_feature}_{changepoint_name}"]` with the
right attrs:

```python
import ethograph as eto
from ethograph.features.changepoints import find_troughs_binary

ds = eto.add_changepoints_to_ds(
    ds,
    target_feature="speed",
    changepoint_name="troughs",
    changepoint_func=find_troughs_binary,
    prominence=0.5,  # ignore dips shallower than 0.5 (feature units)
    distance=10,  # at most one trough per 10 samples
)
```

The stored variable is a binary (`0` or `1`) integer array on the same time
dimension as its target feature, carrying `attrs["target_feature"]` (the
feature it annotates) plus the schema stamp written by
{func}`~ethograph.io.schema.changepoint_attrs` (`kind="changepoint_feature"`,
`changepoint_mask=1`). A mask you computed elsewhere only needs those attrs
to be recognised.
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
    d=speed_array,  # shape: (n_time, n_keypoints)
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
