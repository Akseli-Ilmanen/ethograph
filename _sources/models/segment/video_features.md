# Video features

```{note}
Still in development — the API below may change.
```

A pretrained network turns each frame of a video into a vector: what the
animal *looks like it is doing*, which pose keypoints alone do not capture.
For a worked example from a session to a ranked subset of S3D dimensions, see
{doc}`../../examples/segement_s3d_features`.

| `extractor` | Kind | What one frame's feature is | Docs |
|---|---|---|---|
| `s3d` (default) | clip-wise | the `stack_s` window of frames centred on it, embedded by S3D {cite:p}`xie2018s3d` (Kinetics-400 {cite:p}`kay2017kinetics`) — motion is in the feature | [video_features › S3D](https://v-iashin.github.io/video_features/models/s3d/) |
| `timm` | frame-wise | the frame on its own, embedded by an image backbone — DINOv2 {cite:p}`oquab2024dinov2` ViT-B/14 unless `model_name` says otherwise; needs `pip install 'ethograph[timm]'` | [video_features › timm](https://v-iashin.github.io/video_features/models/timm/) |

Both follow the recipes of
[v-iashin/video_features](https://v-iashin.github.io/video_features/) {cite:p}`iashin2020videofeatures`.
Extraction is a forward pass per frame, so it runs once into a sidecar file per
video, which is then merged into your sessions.

```{important}
Every temporal setting is in **seconds**, resolved against each video's own
rate. `analysis_fps` is how many frames per second the network sees (frames
are skipped, never interpolated) and is the main cost lever. S3D's `stack_s`
needs at least **13 frames** at that rate; the error names the shortest window
that works.
```

## Extracting from a folder of videos

```python
import ethograph as eto

eto.segment.extract_videos(["/data/videos"], "/data/features", stack_s=0.5)  # S3D
eto.segment.extract_videos(["/data/videos"], "/data/features", extractor="timm", analysis_fps=25)
```

Each video becomes `/data/features/{video stem}_{extractor}.nc`, a
`(time_video, {extractor}_dims)` array on the video's own clock. Videos that
already have a sidecar are skipped unless `overwrite=True`.

| Parameter | Meaning |
|---|---|
| `extractor` | `s3d` (default) or `timm`. |
| `model_name` | `timm` only: any timm model with pretrained weights {cite:p}`wightman2019timm`. `None` = DINOv2 ViT-B/14. |
| `stack_s` | `s3d` only: window length in seconds. |
| `analysis_fps` | Rate the network sees; `None` = every frame. |
| `crop` | Pixel box `{x0, y0, x1, y1}` cut from every frame, as reported by *Tools ▸ Video: Pick a crop for a config…*. Draw it square: the network takes a square input, and a non-square box loses its long side. |
| `include` | Regular expressions matched against the path, e.g. `["cam-1"]` to extract one camera only. |
| `overwrite` | Re-extract videos that already have a sidecar. |

A setting belonging to the other extractor is refused by name, not ignored.

## Extracting and merging from a config

With an alignment naming each trial's video, name each session's `video_dir`:

```yaml
sessions:
  - source: ../sub-01/ses-01/behav
    video_dir: /data/videos

video_features:
  extractor: s3d                  # or timm
  analysis_fps: 25
  camera: cam-1
```

```python
project = eto.segment.Project("project.yaml")
project.video_features(merge=True)
# → /data/sub-01/ses-01/behav/Trial_data_s3d.nc
```

Sidecars go to `{root}/video_features/`. Merging samples each sidecar onto its
trial's time axis (applying the trial's video offset) and writes a sibling
`{stem}_{extractor}.nc` carrying an `s3d` or `timm` variable — your session
file is never overwritten. Point `sessions:` at the new file and name the
feature like any other:

```yaml
features:
  columns:
    s3d: {s3d_dims: [0, 1, 2, 3]}      # or a shortlist you selected
```

Merging is xarray-only; for pynapple and NWB sessions carry the sidecar in
yourself.

## Features from a folder of `.npy`

A network run outside ethograph (FERAL, DINO, …) that writes one
`(frames, D)` array per video as `{video stem}.npy` needs no merging.

**In the GUI**, pick **Video features — browse a folder of .npy…** in the
add-panel popup (➕ / Shift+N), or drop the folder on the window. Files are
matched to the session's videos by name; a video without a file reads NaN.

**In a config**, name the folder per session:

```yaml
sessions:
  - source: D:\data\ses-01\behav
    video_feature_folders:
      feral: D:\emb\ses-01
features:
  columns:
    feral: {feral_dims: 0..767}
```

The arrays are attached in memory, never written back to the session file.
FERAL is handled end to end by the project — {doc}`feral`.

## Choosing which dimensions to keep

1024 (S3D) or 768 (DINOv2) columns dominate a handful of kinematic ones, and a
well-chosen subset often does better. To check whether the video columns help
at all, train with and without them and compare:

```python
eto.segment.Project("project.yaml", "train.run_name=full").train()
eto.segment.Project("project.yaml", "train.run_name=no_video", "train.drop_kinds=[video_feature]").train()
print(eto.segment.Project("project.yaml").compare())
```

To pick individual dimensions, rank them by Cohen's d against your curated
labels — {meth}`~ethograph.segment.project.Project.rank_video_features` on a
materialised project, or {func}`~ethograph.video_features.rank_features` on
your own trials. The {doc}`example notebook <../../examples/segement_s3d_features>`
walks through it, including a classes × dimensions heatmap. Rank on training
sessions only, or the selection leaks the test set.

## Adding an extractor

An extractor is an entry in `ethograph.video_features.EXTRACTORS` — a class
with `name`, `plan(video_fps)` and `extract(video)` returning
`to_dataarray(...)`. Its package is pip-installed, never vendored.
