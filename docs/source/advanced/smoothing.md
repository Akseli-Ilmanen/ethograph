(target-smoothing)=
# Smoothing and filtering

Pose and other kinematic traces are noisy: tracking jitter, single-frame
outliers, gaps where a keypoint was lost. Smoothing them is a scripting job,
but picking the parameters by hand in a notebook means guessing, running,
plotting, repeating. **Tools ▸ Data: Find smoothing parameters…** does the
guessing on screen: it previews a pipeline over the line plot you clicked
while you move the sliders, and hands you the Python that reproduces it.

The GUI itself never transforms your data. The preview is drawn on top of the
raw trace and disappears when the dialog closes; labels and changepoint
snapping keep using the raw values throughout.

## The dialog

Click a line plot (it gets a green edge), then open the dialog from the
**Tools** menu. It previews on the clicked plot and follows your clicks, so
you can compare keypoints or individuals without reopening it.

```{figure} ../_static/media/smoothing_dialog_annotated.png
:alt: The Find smoothing parameters dialog: three stage boxes (Filters, Gaps, Smoothing) across the top, the selected Gaussian step's sigma slider and the generated Python script below, the Copy code and Send to console buttons at the bottom right.
:width: 100%

A bridge-gaps step followed by a Gaussian kernel, previewed on the speed of one
keypoint. The script on the right is rewritten on every change.
```

A pipeline is three stages, run left to right:

1. **Filters** — drop what you do not trust. `filter by confidence` blanks
   samples whose pose confidence is below a threshold; `rolling filter` is a
   running median/mean/max/min. Both are
   [movement](https://movement.neuroinformatics.dev)'s functions.
2. **Gaps** — what to do with NaNs, whether from the tracker or from the
   filter above. `bridge gaps` interpolates across gaps up to a given length
   (or every gap); `interpolate over time` is movement's equivalent.
3. **Smoothing** — a kernel: Gaussian, Savitzky-Golay, median, moving average
   or a Butterworth low-pass. Kernels are zero-phase (no lag) and never create
   or remove a gap.

Each stage holds any number of steps (**Add**, **Remove**, drag to reorder).
Click a step to edit its parameters; **Documentation** and **Open source
code** go to what it calls.

**Every parameter is a duration in seconds or a frequency in Hz, never a
sample count.** The dialog reads the plotted data's own rate (shown in the
status line) and converts; the same `sigma_s=0.008` means the same thing on a
200 Hz pose trace and a 30 Hz one. A window too short for the rate says so in
the status line instead of drawing.

## Reading the preview

```{figure} ../_static/media/smoothing_preview.png
:alt: A speed line plot with the raw trace faded, the bridge-gaps stage as a thin blue line and the Gaussian result as a thick orange line.
:width: 100%

The raw trace draws faded, each stage thin in its own colour, the final result
thick in orange. Here a Gaussian with `sigma_s=0.008` removes the tracking
jitter without flattening the speed peaks.
```

Too wide a kernel rounds off the peaks and shifts the troughs that
{doc}`changepoint detection <changepoints/kinematic>` relies on; too narrow a
one leaves the noise. Watch the orange line and stop when the small wiggles
are gone and the bumps still have their height.

## Taking the result with you

- **Copy code** copies the script on the right. It opens the session, runs the
  same steps over the *whole* variable (every keypoint and individual, not just
  the one previewed) and stores the result as a new variable, e.g.
  `speed_clean`. Save and **File ▸ Reload data from disk** to see it in the
  GUI; see {ref}`target-reload-data`.
- **Send to console** runs the same chain in the
  {ref}`console panel <target-reload-data>` on the variable bound to the
  clicked plot, so the smoothed trace is a feature for the current trial
  without writing anything. Steps from movement take an xarray object, so
  they are left out of the console line and marked in the script.

The copied script uses `eto.apply_along_time`, which applies a
kernel along the time axis of an {class}`xarray.DataArray`, a pynapple
`Tsd`/`TsdFrame` or a plain array, reading the rate off the time coordinate:

```python
import ethograph as eto
from ethograph.features.preprocessing import bridge_gaps, gaussian, nan_aware

ds = eto.open("data.nc")
x = ds["speed"]
x = eto.apply_along_time(x, bridge_gaps, max_gap_s=0.2)
x = eto.apply_along_time(x, nan_aware(gaussian), sigma_s=0.008)
ds["speed_clean"] = x
```

`nan_aware` lets a kernel run across NaNs and puts the gaps back afterwards,
so a lost keypoint stays lost rather than being smeared into its neighbours.

The dialog remembers the last pipeline, so reopening it on another session
starts from the values you settled on.
