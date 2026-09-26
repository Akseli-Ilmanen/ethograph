# Config reference

One YAML file per project. Relative paths resolve against the file's own
folder. `base: other.yaml` deep-merges this file over another. Every stage
is a method on `eto.spot.Project`; overrides use the same dotted spelling
the YAML has, passed to `Project(...)`:

```python
import ethograph as eto

project = eto.spot.Project("project.yaml", "clip.context_s=4", "train.epochs=12")
```

```{important}
**Every temporal setting is a duration.** Seconds and milliseconds in the
file; the frame counts the vendored trainer wants are derived from each
video's own rate at run time, so a config moved between a 200 fps rig and a
60 fps one keeps meaning the same thing. See {ref}`target-spot-seconds`.
```

## Top level

| Key | Default | Meaning |
|---|---|---|
| `root` | the config's folder | Project directory: `dataset/`, `features/`, `runs/` and `cross_validation/` live here. |
| `sessions` | required | List of sessions — see below. |
| `frames` | `{root}/frames` | Where decoded frames go, or a folder another project decoded. One folder whatever the crop: each trial's `export.json` records the size and crop it was decoded at, and a trial whose record disagrees with the config is decoded again in place. |
| `individual` | `null` = the session's own | Whose events these are, stamped into every predicted row's `individual` column. The GUI draws a label only on the panels of the individual it names, so it must match a name the session uses. Unset, it is read from the session: its declared individuals, else the dataset's individual dim, else the actor its labels name. A session naming several (or none) is refused until you set it. |
| `features` | `{}` | The pose side, optional — see {ref}`below <spot-config-features>`. |

### `sessions`

| Key | Default | Meaning |
|---|---|---|
| `source` | required | The session file (`.nc`, `.nwb`, a pynapple folder). |
| `name` | the file's stem | The session id in every output (`frames/{name}_trial{n}`, fold names, prediction folders). Quote digit-only names — YAML reads `20260307_01` as a number. Two sessions may not share one. |
| `labels_path` | `{stem}_labels.tsv` beside the source | The curated labels TSV — where the point events to learn come from. |
| `video_dir` | the alignment's | The folder holding this session's videos. |

The sessions layer is the segmentation pipeline's, imported unchanged: a
session has no role; `train.split` gives every trial one.

### `trials`

| Key | Default | Meaning |
|---|---|---|
| `where` | `{}` | Metadata column → allowed values. The one trial filter, applied in every stage. |
| `limit` | `null` | Keep only the first N trials per session — a smoke run. |

## `labels`

| Key | Default | Meaning |
|---|---|---|
| `classes` | required | The point-event label ids to spot, **at most `infer.max_events_per_trial` of each per trial** — one by default: the class's prediction is the tallest peak of its curve. A trial labelled more often than that is refused at `materialise()`; either raise the cap or cut the trial in the trials table. Their order is the order the events happen in (first contact before last contact): `infer.flag_out_of_order` reads it. |
| `camera` | the alignment's default | Which camera's video the model reads, one per project. Only matters with several cameras: give the camera's name as the alignment spells it (`side`, `top`, `0`…), or a piece of the video filename that exactly one camera's files carry (`cam-1` picks `…-cam-1.mp4`). A name no camera matches, or one several match, is refused naming the cameras the alignment has. Overlapping cameras: pick one; non-overlapping: a mosaic — see {ref}`one camera <target-spot-one-camera>`. |
| `crop` | `null` | `{x0, y0, x1, y1}` in source pixels, cut from the decoded frame **before** the resize, so a tight crop spends the model's pixels on less scene. Must fit inside every trial's video for the camera — checked at materialise time. The GUI writes it: Tools ▸ *Video: Pick a crop for a config…*. |
| `frame_height` | `224` | Height the (cropped) frame is resized to; width follows the aspect ratio. E2E-Spot's own {cite:p}`hong2022e2espot`. |

(spot-config-clip)=
## `clip`

The three durations everything else is derived from
(`ClipConfig.resolve(fps)`):

```
stride     = round(resolution_ms / 1000 * fps)
clip_len   = round(context_s * fps / stride)
dilate_len = round(positive_window_ms / 1000 * fps / stride)
```

| Key | Default | Meaning |
|---|---|---|
| `context_s` | `2.0` | Seconds of video the model sees at once. Below about 2 s the model misses events outright. |
| `resolution_ms` | `null` = as fine as fits | Milliseconds one model frame spans — the grid a label can land on. Unset, the stride is the smallest that fits `context_s` into the card's frame budget: every frame on a 25 fps video, every second frame for 2 s of 200 fps on a 10 GB card. Spell it to pin the grid across machines — a bigger card would otherwise pick a finer one. Buying context by coarsening it stopped paying at about 10 ms on the rig this was measured on. |
| `positive_window_ms` | `10` | ± this around the labelled event counts as positive during training. A duration, so dilation is not confounded with resolution when the latter changes. |


## `model`

| Key | Default | Meaning |
|---|---|---|
| `architecture` | `rny008_gsm` | Upstream's `--feature_arch`: a backbone plus a temporal module. `rny008_gsm` is E2E-Spot's own {cite:p}`hong2022e2espot,radosavovic2020regnet,sudhakaran2020gsn`; `rny008_msagsm` swaps the Gate Shift Module for the multi-scale one {cite:p}`msagsm2025`. `eto.spot.architectures()` lists the names, `describe_architecture(name)` says what each is; an unknown name is refused before any frame is decoded. |
| `head` | `gru` | Upstream's `--temporal_arch`. |
| `shift_scales_ms` | `[40, 80, 120]` | MSAGSM only: how far each gated-shift branch reaches, in milliseconds, resolved against the *strided* clock. The paper's `{1, 2, 3}` frames at 25 fps. |
| `attention_groups` | `2` | MSAGSM only: channel groups of its spatial attention (the paper's 2). |

## `train`

| Key | Default | Meaning |
|---|---|---|
| `run_name` | `ctx{context_s}s_res{resolution_ms}ms` | The run's folder under `runs/`; `_features` is appended when the features are fed in. Two runs with one name overwrite each other — name a second run of the same clip. |
| `epochs` | `8` | Every run trains its full budget; the epoch used afterwards is the one the sweep ranks first on the run's own validation predictions (fewest misses, then most within 4 frames, 20 ms at 200 fps), never the last and never `val_mAP`. |
| `epoch_frames` | `250000` | Frames per epoch. What an epoch costs, whatever the trial count (~6.5 min per 250 k at 3.2 it/s on one RTX 3080). |
| `learning_rate` | `1e-3` | Linear warm-up then cosine, upstream's schedule. |
| `warm_up_epochs` | `1` | Epochs of warm-up; must be fewer than `epochs`, or the cosine has nothing left. |
| `start_val_epoch` | `1` | First epoch that writes validation predictions (`pred-val.{epoch}.recall.json.gz`), which the epoch choice reads. |
| `batch_size`, `acc_grad` | `4`, `4` | Clips per optimiser step and gradient-accumulation steps: `batch_size / acc_grad` clips per loader batch, i.e. `clip_len × batch_size / acc_grad` frames — the number `MAX_FRAMES_PER_BATCH` caps. |
| `retries` | `2` | Resume-and-retry a training that crashed (a GPU hiccup), from its last checkpoint. |
| `seed`, `device` | `0`, auto | `device` = `cuda`, `mps`, `cpu`; auto picks the best available. |
| `features_dropout` | `0` | Share of training clips whose feature block is zeroed (modality dropout). Off by default; set it (e.g. `0.3`) when some trials will have no pose, or to make `evaluate(zero_features=True)` a fair ablation. |

### `train.split`

The segmentation pipeline's: three ratios drawn by whole trial over every
trial of every session, and `holdout_sessions` for a cross-validation fold
(written per fold by `project.cross_validate()`, not by hand).

| Key | Default | Meaning |
|---|---|---|
| `train_fraction` | `0.6` | Fraction of trials the model learns from. |
| `val_fraction` | `0.2` | Trials that choose the epoch. |
| `test_fraction` | `0.2` | Trials `evaluate()` scores, once. |
| `seed` | `0` | Change it to re-draw the split. |
| `holdout_sessions` | `[]` | Sessions held out whole as `test` — one fold. |

The three fractions must sum to 1. Adding a session reshuffles the existing
trials, same seed or not; a fold pins the split instead.

(spot-config-features)=
## `features`

The pose side, **optional**, and the whole of it: session variables in the
segmentation pipeline's *column* spelling — `feature → dim → values`, every
dim pinned except the individual's — on the pose's own rate.

```yaml
features:
  velocity: {space: [x, y], keypoint: [stickTip, pellet]}
  pellet_stickClosest_dist: {}
```

Every entry is a variable in the session file you built and can plot; there
is no graph, no adjacency, no learned geometry. Listed, the columns are
written once per trial to `features/{video_id}.npz` at `materialise()` and
fed to the pixel model beside the frames (z-scored on the training split
under `features/block/`). Absent,
the model is E2E-Spot on pixels alone. See {doc}`multimodal`.

(spot-config-label-inputs)=
## `label_inputs`

Optional. The **labels of other branches** as input columns, appended to
`features:` so they ride into the GRU beside the pose — the segmentation
pipeline's {ref}`features.label_inputs <segment-config-label-inputs>`, same
keys, same renderings (a state's indicator, a point's Laplacian bump per
width in `point_sigmas_s`). Two differences:

| Key | Default | Meaning |
|---|---|---|
| `mapping` | `~/.ethograph/defaults/mapping.txt` | This config names no mapping of its own, so say where the branches are read from when it is not the default one. |
| `clock` | the first of `features:` | With no pose listed there is no feature to take the clock from — name the session variable whose time coordinate the columns should be rendered on. |

```yaml
labels:
  classes: [7]                        # first contact, branch 1 of the mapping
features:
  pellet_stickClosest_dist: {}
label_inputs:
  branches: [0]                       # the earlier question's labels, branch 0
```

**A branch holding any of `labels.classes` is refused**: the model would
learn to copy its input. The columns are rendered for `individual` (the one
event stream this pipeline predicts) and a trial with none of these labels
renders zeros. With `label_inputs` set the run is a `_features` run, `train`
z-scores the block on the training split as for the pose, and
`evaluate(zero_features=True)` measures what the block — pose and old labels
together — contributes.

## `infer`

| Key | Default | Meaning |
|---|---|---|
| `focus_window_ms` | `100` | ± this around the tallest peak counts as the same event when reading `focus`/`ratio` off a curve — twice the precision you believe your labels to (the lightgbm model takes it from its `tolerance_s`). See the confidence page. |
| `max_events_per_trial` | `1` | How many events of one class a trial may hold. `1`: the tallest peak of the class's curve is the event, no threshold needed — the best candidate wins. Above 1: every peak at least `min_event_gap_s` from a taller one is an event, up to this many, and the model now returns spurious ones too, so calibrate the confidence you flag below on the grid's histogram before reviewing. `ratio` and the rules built on it (`product`, `custom`) are refused — a second peak is another event, not a rival — so set `confidence: focus` or `peak`. `flag_out_of_order` is refused, since order between classes is undefined. Training refuses a trial labelled more often than this. |
| `min_event_gap_s` | `0.5` | With several events per trial, two peaks closer than this are one event, the taller. Read only when `max_events_per_trial > 1`. |
| `flag_out_of_order` | `false` | A trial whose predicted events are not in `labels.classes` order has every event's confidence set to 0 — flagged, never reordered or dropped. One event per class only. |
| `source` | `spot:{run}@{epoch}` | Currently unused: every predicted row's `prediction_source` is always `spot:{run}@{epoch}`. |
| `flag_confidence_below` | `0.01` | Events below this confidence are logged as flagged, never dropped. |
| `jpeg_roundtrip` | `true` | Inference decodes the video straight into the model; each frame passes through JPEG in memory first, so the model sees what training saw (the export writes JPEGs). Off = an ablation. |
| `confidence` | `product` | Which reading of a prediction's curve is written as its `confidence`: `product` (focus × ratio), `ratio` (one candidate or two), `focus` (sharp or smeared), `peak`, or `custom` = `ratio × (α + (1 − α)·focus)`. The grids' histogram popup previews these on a session's curves and its **Copy for project.yaml** button hands you these lines. |
| `confidence_alpha` | `0.5` | α of the `custom` rule; ignored otherwise. |

Inference never exports frames: `inference()` decodes each trial front to
back into the model (`spot/stream.py`), with a rolling one-window buffer.
Frames on disk are training's alone.

## What a run writes

```
dataset/
  train.json / val.json / test.json   E2E-Spot's own index, one entry per trial with its events
  class.txt, index.tsv                class names; every materialised trial with its rate and frame count
{frames}/{video_id}/%06d.jpg          decoded frames, plus export.json (size + crop)
features/                             with features: listed
  {video_id}.npz, features.json       the listed columns per trial; their names in order
  block/                              the same, z-scored on the training split (stats.npz, block.json)
runs/{run}/
  config.yaml, config.json            the resolved config; upstream's own record (stride, clip_len, …)
  checkpoint_{epoch}.pt, loss.json    weights per epoch; train/val loss and val_mAP per epoch
  pred-val.{epoch}.recall.json.gz     validation predictions per epoch — what the epoch choice reads
  pred-test.{epoch}.recall.json.gz    test predictions of the chosen epoch, written by evaluate()
  test_metrics.yaml                   per class: misses, spurious, error in ms, hit rate per tolerance
  test_metrics_nofeatures.yaml        the same with the feature block zeroed (features runs)
  train.log, evaluate.log             everything logged
runs/compare.tsv                      written by project.compare(): every scored run side by side
cross_validation/{session}/           one project per fold, its own dataset/ and runs/fold_{session}
```

Predictions land beside each session, never under `root`:
`{session}/labels/predictions_spot_{run}_{timestamp}/` holds the labels TSV
(`{stem}_predictions.tsv`) the GUI imports and `onset_curves.npz`, the curves frame-by-frame review
draws.
