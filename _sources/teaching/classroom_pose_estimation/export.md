# 7. Visualizing & exporting

- **Load into the GUI** — the keypoints become the session's features, saved
  beside your video as `<video>.keypoints.nc`. Pick `position`, `velocity`,
  `speed` or `acceleration` in the right sidebar; the **Keypoint** and
  **Individual** dropdowns choose which one the plot shows. If a panel looks
  empty, check those dropdowns first.

- **Head direction (from marker orientation)** — with tags, tick this to get
  each animal's heading as a feature too.

- **Export poses (NetCDF)…** — a [movement](https://movement.neuroinformatics.dev)-compatible
  file covering every frame of the video; frames outside the filled span are
  `NaN`.

```{note}
Velocity, speed and acceleration are computed between the frames a point was
seen on. Run **Fill** first if you want them frame by frame.
```
