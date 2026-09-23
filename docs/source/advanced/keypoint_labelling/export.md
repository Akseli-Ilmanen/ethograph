# 7. Visualizing & exporting

- **Load into the GUI** — the keypoints become the session's features, and a
  copy is saved beside your video as `<video>.keypoints.nc`.

  Nothing is grafted onto the dataset you had open: the keypoints *are* a
  dataset, so they replace what serves features, and `keypoint` and `individual`
  arrive as ordinary dimensions you pick in the right sidebar next to `space`
  (x/y). Your video, audio, panels and layout are untouched — only the data
  under them changes.


  A feature plot draws one line per value of *one* dimension, so picking a
  keypoint feature pins the rest automatically; the **Space / Keypoint /
  Individual** dropdowns change which one is free. If a panel looks empty, that
  is the first thing to check.


- **Head direction (from marker orientation)** — TODO replace with visual example.


- **Export poses (NetCDF)…** — a [movement](https://movement.neuroinformatics.dev)-compatible
  poses dataset covering every frame of the video; frames outside the filled
  span carry `NaN` position and `NaN` confidence, which is how movement
  represents a missing point (see {ref}`target-labelled-span`). Head direction,
  if ticked, is written alongside `position` as `head_direction` and `heading`.


```{note}
Velocity, speed and acceleration are measured between the frames a point was
actually *seen* on — labelled or filled. With a handful of labels and no fill
that means one value per labelled frame (the average velocity across each gap)
and `NaN` in between, rather than nothing at all. Run **Fill** first if you want
them frame by frame.
```
