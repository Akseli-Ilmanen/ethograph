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
An unknown key is an error, in the file and in an override alike — a typo
must not silently become a default. So is a section this pipeline retired:
`graph:`, `fuse:` and `teacher.features` are refused by name with the
replacement, and a `columns:` key under `features:` (the segmentation
pipeline's section shape) with the difference.
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
| `root` | the config's folder | Project directory: `dataset/`, `features/`, `teacher/`, `runs/` and `cross_validation/` live here. |
| `sessions` | required | List of sessions — see below. |
| `frames` | `{root}/frames` | Where decoded frames go, or a folder another project decoded. One folder whatever the crop: each trial's `export.json` records the size and crop it was decoded at, and a trial whose record disagrees with the config is decoded again in place. |
| `individual` | `null` | Stamped into every predicted row's `individual` column; `null` writes the empty recipient, as before. |
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
| `classes` | required | The point-event label ids to spot. Their order is the order the events happen in (first contact before last contact): `infer.flag_out_of_order` reads it. |
| `camera` | the alignment's default | Which camera's video, one per project. |
| `crop` | `null` | `{x0, y0, x1, y1}` in source pixels, cut from the decoded frame **before** the resize, so a tight crop spends the model's pixels on less scene. Must fit inside every trial's video for the camera — checked at materialise time. The GUI writes it: Tools ▸ *Video: Pick a crop for a config…*. |
| `frame_height` | `224` | Height the (cropped) frame is resized to; width follows the aspect ratio. E2E-Spot's own {cite:p}`hong2022e2espot`. |

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

The **frame budget** — frames per loader batch the card holds — scales
with the card present: 200 was measured on a 10 GB card (`MAX_FRAMES_PER_BATCH`,
the one measured point), so a 24 GB card gets ~480 and an 8 GB one ~160
(`frame_budget()`); with no CUDA device the measured card is assumed, so a
config resolves the same on every machine without one. A spelled
`resolution_ms` whose clip would exceed the budget is refused, naming the
durations to change rather than the frame counts you never wrote; a clip
shorter than `MIN_CLIP_LEN` (8 model frames — below that the GRU has nothing
to integrate over) likewise. A trained run records its stride in
`config.json` and is read back from there, so a session predicted on another
card uses the run's stride. A strided prediction is read back at the
**centre** of its bin, `bin * k + (k - 1) / 2`.

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
| `features_as_input` | `true` | With `features:` listed: hand them to the pixel model beside the CNN features, before the GRU (option 3). `false` keeps the list for the pose teacher only (option 4). |
| `features_dropout` | `0.3` | Share of training clips whose feature block is zeroed (modality dropout), so the pixels are trained to carry the event on their own too — which is what keeps `evaluate(zero_features=True)` meaningful. |

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
serve two models: fed to the pixel model beside the frames
(`train.features_as_input`, z-scored on the training split under
`features/block/`), and read by the pose teacher (`train_teacher()`). Absent,
the model is E2E-Spot on pixels alone. See {doc}`multimodal`.

## `teacher`

The pose-only teacher (`pose_model.PoseSpotter`): the listed features → a
linear embedding → `depth` blocks of a parameter-free multi-scale temporal
shift → a bi-GRU → a `K + 1` softmax. Trained by `train_teacher()`, minutes on
a GPU; distilled into the pixel model by `distil()`.

| Key | Default | Meaning |
|---|---|---|
| `shift_scales_ms` | `[40, 80, 160]` | Temporal shift scales of the blocks, in ms, resolved against the features' own rate (UMEG-Net's `{1, 2, 4}` frames at 25 fps {cite:p}`umegnet2026`). |
| `hidden` | `64` | Width of every block. |
| `depth` | `4` | Stacked blocks. |
| `shift_fraction` | `0.125` | Channels shifted forward and backward, as a fraction of `hidden`. |
| `head_hidden` | `128` | Bi-GRU width. |
| `epochs` | `30` | Training budget; the epoch is chosen by the same sweep as a pixel run's. |
| `learning_rate`, `weight_decay` | `1e-3`, `1e-2` | AdamW. |
| `batch_size` | `8` | Clips per step. |
| `fg_weight` | `5` | Foreground class weight in the per-frame cross-entropy — E2E-Spot's own. |
| `seed` | `0` | |

The `features` and `teacher` sections are fingerprinted into the teacher's
folder (`teacher/{clip}_{fingerprint}`) and the distilled student's, so an
edited list lands beside the earlier result, never on top of it.

## `distil`

The two distillation steps, inside the vendored trainer: (2) the baseline's
trunk + GRU learn to reproduce the teacher's per-frame embedding on every
clip that has pose — no labels; (3) the CNN is frozen and the head learns the
labels. Both are ordinary runs under `runs/{baseline}_distil_{fingerprint}/`.

| Key | Default | Meaning |
|---|---|---|
| `teacher_run` | the one whose embeddings are under `features/embeddings/` | The teacher run under `teacher/` to distil from. A mismatch is refused. |
| `init_run` | the newest trained run (refused if it is a distilled student) | The baseline the student starts from; must agree about `features_as_input`. |
| `epochs`, `epoch_frames`, `learning_rate` | `6`, `250000`, `1e-4` | The embedding-matching step. |
| `head_epochs`, `head_learning_rate` | `4`, `1e-4` | The head step. |
| `retries` | `2` | As `train.retries`. |

## `infer`

| Key | Default | Meaning |
|---|---|---|
| `focus_window_ms` | `100` | ± this around the tallest peak counts as the same event when reading `focus`/`ratio` off a curve — twice the precision you believe your labels to (the lightgbm model takes it from its `tolerance_s`). See the confidence page. |
| `flag_out_of_order` | `false` | A trial whose predicted events are not in `labels.classes` order has every event's confidence set to 0 — flagged, never reordered or dropped. |
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
  embeddings/                         the teacher's per-clip embeddings (teacher.json names the teacher)
teacher/{clip}_{fingerprint}/         a teacher run: checkpoints, pred-val.*.recall.json.gz, loss.json, stats.npz
runs/{run}/
  config.yaml, config.json            the resolved config; upstream's own record (stride, clip_len, …)
  checkpoint_{epoch}.pt, loss.json    weights per epoch; train/val loss and val_mAP per epoch
  pred-val.{epoch}.recall.json.gz     validation predictions per epoch — what the epoch choice reads
  pred-test.{epoch}.recall.json.gz    test predictions of the chosen epoch, written by evaluate()
  test_metrics.yaml                   per class: misses, spurious, error in ms, hit rate per tolerance
  test_metrics_nofeatures.yaml        the same with the feature block zeroed (features runs)
  train.log, evaluate.log             everything logged
runs/{baseline}_distil_{fingerprint}/stage2/, stage3/    the distilled student, two ordinary runs
runs/compare.tsv                      written by project.compare(): every scored run side by side
cross_validation/{session}/           one project per fold, its own dataset/ and runs/fold_{session}
```

Predictions land beside each session, never under `root`:
`{session}/labels/predictions_spot_{run}_{timestamp}/` holds the labels TSV
(`{stem}_predictions.tsv`) the GUI imports and `onset_curves.npz`, the curves frame-by-frame review
draws.
