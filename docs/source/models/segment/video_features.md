# Video features

A pretrained network turns each frame of a video into a vector: what the
animal *looks like it is doing*, which pose keypoints alone do not capture.
The network is an **extractor**, chosen by name; every extractor writes the
same kind of file, and downstream nothing cares which one made it.

| `extractor` | Kind | What one frame's feature is | Docs |
|---|---|---|---|
| `s3d` (default) | clip-wise | the `stack_s` window of frames centred on it, embedded by S3D {cite:p}`xie2018s3d` (Kinetics-400 {cite:p}`kay2017kinetics`) — motion is in the feature | [video_features › S3D](https://v-iashin.github.io/video_features/models/s3d/) |
| `timm` | frame-wise | the frame on its own, embedded by an image backbone — DINOv2 {cite:p}`oquab2024dinov2` ViT-B/14 (`vit_base_patch14_reg4_dinov2.lvd142m`) unless `model_name` says otherwise; needs `pip install 'ethograph[timm]'` | [video_features › timm](https://v-iashin.github.io/video_features/models/timm/) |

Both follow the recipes of
[v-iashin/video_features](https://v-iashin.github.io/video_features/) {cite:p}`iashin2020videofeatures`, which
documents each network and links on to its weights; our config exposes only
what changes the *features*, so read there for the rest.

A frame-wise feature carries no motion; the temporal model downstream reads
it off the sequence. A clip-wise feature carries the window's motion but
needs the window to be long enough for the network. Which is better for a
given behaviour is an empirical question — the point of one registry is that
the comparison is a config line and a `compare_runs`.

They are expensive — a forward pass per frame — so they are computed once
into a sidecar file per video and merged into your sessions afterwards.

```{important}
Every temporal setting is in **seconds** and resolved against each video's own
rate. `analysis_fps` says how many frames per second the network sees (frames
are skipped, never interpolated up); it is the one cost lever, and at 200 fps
you will want it. S3D's `stack_s` needs at least **13 frames** at that rate,
so its 0.5 s default works down to 26 fps but *fails* below it — the error
names the shortest window that works.
```

## Starting from a folder of videos

No config, no session, no alignment — just videos in, sidecars out:

```python
import ethograph as eto

eto.segment.extract_videos(["/data/videos"], "/data/features", stack_s=0.5)  # S3D (default)
eto.segment.extract_videos(["/data/videos"], "/data/features", extractor="timm", analysis_fps=25)  # timm / DINOv2
```

The first argument takes any mix of files, folders (searched recursively) and
globs. Each video becomes `/data/features/{video stem}_{extractor}.nc`,
holding a `(time_video, {extractor}_dims)` array **on the video's own clock**
(frame 0 at t=0), with the resolved plan and the model in its attrs. The
written paths are returned. Videos that already have a sidecar are skipped
unless you pass `overwrite=True`, so re-running after adding footage is cheap.

| Parameter | Meaning |
|---|---|
| `extractor` | `s3d` (default) or `timm`. |
| `model_name` | `timm` only: any timm model with pretrained weights {cite:p}`wightman2019timm` — valid names are the [timm model list](https://huggingface.co/docs/timm/models) (or `timm.list_models(pretrained=True)`). `None` = DINOv2 ViT-B/14. |
| `stack_s` | `s3d` only: window length in seconds. Must be ≥ 13 frames at the effective rate. |
| `analysis_fps` | Rate the network sees; `None` = every frame. |
| `crop` | A pixel box cut from every frame before the network sees it (`{x0, y0, x1, y1}`, the crop tool's numbers). |
| `include` | Regular expressions; keep only videos whose path matches one. |
| `overwrite` | Re-extract videos that already have a sidecar (default `False`). |

A setting that belongs to the other extractor is refused by name — `stack_s`
means nothing to a frame-wise model, `model_name` nothing to S3D — rather than
silently ignored. Those are the only settings that change the features. Batch
size, decode chunk, `fp16` and the device live on the extractor's own config
({class}`~ethograph.video_features.S3DConfig`,
{class}`~ethograph.video_features.timm_extract.TimmConfig`).

### Cropping to the animal

Resolution on the animal matters more than resolution on the arena. `crop`
cuts one rectangle from every decoded frame before the resize, in the same
numbers the GUI's crop tool reports (*Tools ▸ Video: Pick a crop for a config…*), so they
copy straight across:

```yaml
video_features:
  crop: {x0: 240, y0: 80, x1: 880, y1: 720}
```

**The network takes a square, and you choose which one.** Every extractor
has a fixed square input — 224×224 for S3D, the model's own side for a timm
backbone (518 for DINOv2) — and gets there by scaling the box's shorter side
to it and taking the centre square, the Kinetics / ImageNet evaluation
transform. So:

- **No `crop`**: the centre square of the full frame, scaled. The default,
  and fine when the animal sits mid-frame.
- **A square `crop`**: exactly what you drew, only scaled. The crop tool's
  *Square* option (ticked by default) grows the dragged rectangle to a square
  about its centre, so the numbers it prints are the numbers the network
  sees. Draw it 224×224 for S3D and there is no resampling at all.
- **A non-square `crop`**: the long side is cut off — a 203×164 box loses
  19 % of its width. The extraction logs a warning saying how much; it does
  not refuse.

One box per video, applied to every trial's video alike. A box that follows an
individual — the per-individual feature a multi-animal recording needs — is
the planned extension; the sidecar already records the crop in its attrs so
that step needs no new format.

### Taking one camera out of a folder

Two cameras pointed at the same arena give nearly identical features, so
extracting both is an hour of GPU time for nothing. `include` narrows what is
found:

```python
eto.segment.extract_videos(["/data/videos"], "/data/features", include=["cam-1"])
```

Each pattern is a regular expression matched against the **whole path** with
`re.search`, so a plain substring works, and it finds the camera wherever it
sits in the layout — `cam-1/trial003.mp4` and `trial003_cam-1.mp4` alike.
Several patterns are a union (`["cam-1", "cam-3"]`).

```{note}
A filter that matches nothing raises rather than extracting zero videos —
silently doing nothing looks too much like success. The message says how many
videos were found and shows a few, so you can see what to match against.
```

Working from a config instead? Use `video_features.camera` — the alignment
already knows which stream is which, so there is nothing to pattern-match.

## Starting from a config

If your sessions already have an alignment naming each trial's video, name
each session's `video_dir` and let the alignment resolve the file within it:

```yaml
sessions:
  - source: ../sub-01/ses-01/behav
    labels_path: ../sub-01/ses-01/behav/Trial_data_labels.tsv
    video_dir: /data/videos

video_features:
  extractor: s3d                  # or timm
  analysis_fps: 25
  camera: cam-1                   # regex pattern so video features only for camera 1
```

```python
project = eto.segment.Project("project.yaml")

project.video_features()  # extract only
project.video_features(merge=True)  # extract, then merge into the sessions
```

Sidecars go to `{root}/video_features/`; the paths written are returned, and
`overwrite=True` re-extracts. Any setting can also be overridden without
editing the file:
`eto.segment.Project("project.yaml", "video_features.extractor=timm", "video_features.analysis_fps=25")`.

## Merging into a session

Extraction alone does not make the embedding a *feature* — the sidecar is on
the video's clock, and the pipeline reads features from the session. Merging
samples the sidecar onto each trial's own time axis (nearest neighbour,
applying that trial's video offset) and writes a session copy carrying a
variable named after the extractor, `s3d (time, s3d_dims)` or
`timm (time, timm_dims)`:

```python
project.video_features(merge=True)
# → /data/sub-01/ses-01/behav/Trial_data_s3d.nc
```

The merge **never overwrites your session file**: it writes a sibling
`{stem}_{extractor}.nc` and logs the path — point your config's `sessions:` at
it (or call {func}`~ethograph.segment.video_features.merge_video_features`
yourself with `in_place=True` if you would rather overwrite). Merging is
xarray-only; for pynapple and NWB sessions the sidecar exists but you carry it
in yourself. Sidecars written before the registry existed (`time_s3d`) still
merge: the time dim is found by name.

Then name it like any other feature:

```yaml
features:
  columns:
    s3d: {s3d_dims: [0, 1, 2, 3]}      # or a shortlist you selected
    speed: {keypoint: [beakTip]}
```

## Video features from files: a folder of `.npy`

A network run outside ethograph — FERAL, DINO, anything that writes one
`(frames, D)` array per video — leaves a folder of `{video stem}.npy`.
Those are video features like any other, and they need no merging.

**In the GUI**, open the add-panel popup (➕ or Shift+N) and pick
**Video features — browse a folder of .npy…**, or drop the folder (or a
selection of `.npy` files) anywhere on the window. Each folder becomes one
feature named after it and opens as a heatmap; loose files together become
one feature named after their folder. Files are matched to the session's
videos by name, the camera whose names they follow is detected, and the log
says how many videos matched:

```
Added video features 'feral_cfgA' (768 columns) from D:\emb\cfgA: 120 of 136 videos matched (88%)
```

A video without a file is fine — features computed on a subset are still
features; its trial reads NaN. Files matching *no* video are an error. Two
exports of one model with different settings live side by side: the
add-panel popup lists them under **Video features**, each with the path it
came from, so `feral_cfgA (D:\emb\cfgA)` and `feral_cfgB (D:\emb\cfgB)` are
never confused.

**In a config**, a session names its folders, and `open_session` attaches
them the same way, so `materialise` and `inference` see the variable:

```yaml
sessions:
  - source: D:\data\ses-01ehav
    video_feature_folders:
      feral: D:\emb\ses-01          # one {video stem}.npy per trial's camera file
features:
  columns:
    feral: {feral_dims: 0..767}
```

Either way the arrays are memory-mapped and sampled onto the trial clock
exactly as a merged sidecar is (trial = video + offset, nearest frame, the
video's own rate). Nothing is copied or declared beside the session: the
file stays as it was, and {meth}`~ethograph.io.trialtree.TrialTree.save`
never writes the variable back (it carries `attrs["attached_from"]`).

A script attaches the same way, on the tree the GUI would open:

```python
from ethograph.io.video_feature_files import VideoFeatureFiles, attach_video_features

attach_video_features(dt, alignment, VideoFeatureFiles.from_paths("feral_cfgA", ["D:/emb/cfgA"]))
```

Merging stays the path for the extractors ethograph runs itself, whose
sidecars carry the plan and model in their attrs.

FERAL is the one such network the project handles end to end: `extractor:
feral` writes its inputs from the sessions' curated labels, and its
embeddings attach from `{root}/feral/embeddings/` without listing a folder
per session — {doc}`feral`.

## Choosing which dimensions to keep

1024 (S3D) or 768 (DINOv2 ViT-B) columns is a lot next to a handful of
kinematic ones, and they dominate the input — in practice a small, well-chosen
subset does *better* than all of them. Two tools, both leaning on
`kind="video_feature"`, which every extractor
stamps for you.

**Is the whole group pulling its weight?** That is a question about the
trained model, so it takes two runs — the same materialised dataset, one
extra fit:

```python
eto.segment.Project("project.yaml", "train.run_name=full").train()
eto.segment.Project(
    "project.yaml",
    "train.run_name=no_video",
    "train.drop_kinds=[video_feature]",
).train()

print(eto.segment.Project("project.yaml").compare())
```

```{important}
`compare()` reads each run's `test_metrics.yaml`, which is written only when
the split leaves something in `test`. With `train.split.test_fraction: 0` the
table comes back empty — and an ablation judged on a random trial split is
flattered anyway, so the honest version of this benchmark is
`cross_validate()` with and without the video columns.
```

**Which extractor?** The same two-run shape: merge both sidecars into the
session, list one or the other under `features.columns`, compare.

**Which individual features are stereotypic?** Rank them by Cohen's d: for
each feature and each behaviour class, how far apart the feature's
distribution is *during* that behaviour versus outside it, in pooled standard
deviations. A feature with a large d for some class is one the model can act
on; a feature with a small d everywhere is noise the model has to learn to
ignore.

```python
ranking, names = project.rank_video_features()
# the 20 most discriminating dims
print([names[i] for i in ranking.top(20)])
```

This reads the **materialised dataset**, so it ranks exactly the columns a
model would see and costs no re-extraction. It picks them out by
`kind="video_feature"`, and raises if no column declares it — describe your
features when you build the session ({func}`~ethograph.io.schema.describe`),
then materialise again. `min_frames` drops a class that barely occurs in a
trial.

The library call underneath takes the trials directly, for a bank that is not
in a project yet:

```python
from ethograph.video_features import rank_features

ranking = rank_features(trials)  # [(values (T, F), labels (T,)), ...]
```

`labels` are dense per-frame class ids with 0 = background; background is
excluded from the comparison by default (`background=0`). Scores are averaged
over trials.

In the GUI the same ranking is a heatmap — **Model ▸ Video features: rank by
Cohen's d…** — of classes against the top-k features, so you can see which
behaviours each feature actually separates, and copy the top-k straight into
your config as a `features.columns` line, keyed by the feature and its own
dim:

```yaml
features:
  columns:
    s3d: {s3d_dims: [492, 734, 671, 640, 585, ...]}
```


## One video, no ethograph at all

```python
from ethograph.video_features import S3DConfig, build_extractor, extract_s3d

da = extract_s3d("clip.mp4", S3DConfig(stack_s=0.5))  # (time_video, s3d_dims)
da.to_netcdf("clip_s3d.nc")

da = build_extractor("timm").extract("clip.mp4")  # (time_video, timm_dims)
```

## Adding an extractor

An extractor is an entry in `ethograph.video_features.EXTRACTORS` — a name
mapped to a class with `name`, `plan(video_fps)` and `extract(video)` — whose
`extract` returns `to_dataarray(...)`. Its package is pip-installed, never
copied into the tree; if the package cannot share the GUI
environment, the extractor runs it by subprocess and reads the file back.
