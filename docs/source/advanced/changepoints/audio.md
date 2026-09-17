(target-audio-changepoints)=
# Audio changepoints

For audio data or high-sample-rate periodic signals (loaded as `.wav` file).

Four methods are available, drawn from two libraries.

---

## VocalPy methods

Reference: [VocalPy documentation](https://vocalpy.readthedocs.io/) {cite:p}`nicholson2023vocalpy`

**Mean-squared energy** (`meansquared`): Computes a smoothed energy envelope
via mean-squared amplitude, then thresholds to find vocal segments. Simple
and fast.

**AVA** (`ava`): The segmentation method from the Animal Vocalization
Analysis pipeline. Uses a spectrogram-based approach with multiple threshold
levels.

---

## VocalSeg methods

Reference: [VocalSeg](https://github.com/timsainb/vocalization-segmentation) {cite:p}`sainburg2020finding`

**Dynamic thresholding** (`vocalseg`): Adaptive threshold segmentation that
adjusts to local spectral energy. Good for signals with varying background
noise.

**Continuity filtering** (`continuity`): Extends dynamic thresholding with
temporal continuity constraints to merge fragmented detections.

---

## Usage

1. Open the **Audio CPs** panel.
2. Choose a method and click **Configure...** to adjust parameters.
3. Click **Detect**.

Detected onsets and offsets are drawn as vertical lines on the plot and
stored in the dataset as `audio_cp_onsets` / `audio_cp_offsets`.

---

## Data format

```{note}
Audio changepoints format is subject to change, still in development.
```

Audio changepoints use a different storage format because dense binary arrays
at audio sample rates (e.g. 44 kHz) would be prohibitively large. They are
stored as onset/offset time pairs in seconds:

```python
ds["audio_cp_onsets"] = xr.DataArray(
    onset_times_s,
    dims=["audio_cp"],
    attrs={"target_feature": "audio", "method": method},
)
ds["audio_cp_offsets"] = xr.DataArray(
    offset_times_s,
    dims=["audio_cp"],
    attrs={"target_feature": "audio", "method": method},
)
```

---
