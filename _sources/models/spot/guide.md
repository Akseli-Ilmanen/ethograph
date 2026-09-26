(target-spot-guide)=
# Full guide

Every stage of pixel event spotting with the choices put back in: the three
ways to spot a point event, what this shares with the segmentation pipeline,
the config you write, why every temporal setting is a duration, the stages
and their cost, how confidence is read from a curve, and MSAGSM.
{doc}`quickstart` is this page with every choice made for you; {doc}`config`
lists every key.

```python
import ethograph as eto

project = eto.spot.Project("spot.yaml")
project.materialise()  # sessions -> frames (+ the listed features), and the model's own index
project.train()  # one run under runs/
project.evaluate()  # per class: misses, error in ms, hit rate per tolerance -> test_metrics.yaml
project.compare()  # every scored run side by side -> runs/compare.tsv
project.inference()  # a run's predictions into each session's labels/ folder
project.cross_validate()  # one fold per session: train on the rest, predict the held-out one
```

The model is **E2E-Spot** {cite:p}`hong2022e2espot`: a RegNetY-008 backbone {cite:p}`radosavovic2020regnet`
with Gate Shift Modules {cite:p}`sudhakaran2020gsn` for temporal mixing and a bi-GRU head emitting a
per-frame softmax over `K + 1` classes. Upstream's
code ships inside Ethograph (`ethograph/spot/e2espot/`), in its own layout.

## Three ways to spot a point event

Which one to use is decided by **what is available when the model runs**.

```{list-table}
:header-rows: 1
:widths: 4 30 22 22 22

* -
  - Model
  - Trains on
  - Reads at inference
  - Where
* - 1
  - **LightGBM model** {cite:p}`ke2017lightgbm` — a boosted-tree classifier on a window of the
    features you tick
  - pose features
  - pose features
  - the GUI, `Model ▸ LightGBM: Train…` ({doc}`../onset_model`)
* - 2
  - **E2E-Spot** {cite:p}`hong2022e2espot` — pixels only; `rny008_msagsm` {cite:p}`msagsm2025` widens its temporal aperture
  - video
  - video
  - `eto.spot`, no `features:`
* - 3
  - **E2E-Spot + features** — the listed pose features ride into the GRU
    beside the CNN features
  - video + pose
  - video + pose
  - `eto.spot`, `features:` listed
```

- **Pose exists for every trial** → start with **1**: minutes on a CPU, and
  every input is a variable you can plot. If a class stays weak, **3** adds
  the pixels to the same features; `evaluate(zero_features=True)` scores that
  run with the features zeroed, so what the pose contributes is a number
  rather than an assumption.
- **Video only at inference** → **2**.

### Which stages to run

`scripts/spot.py` runs the stages you name, in order.

**Pose in every session, now and later** (option 3). The features ride into
the GRU:

```bash
python scripts/spot.py materialise baseline evaluate
python scripts/spot.py inference --run ctx2s_res10ms_features --sessions 20260308_01
```

**Video only, no pose anywhere** (option 2). The same two lines with no
`features:` listed; the run is named `ctx2s_res10ms`.

**Every pose input is a variable in your session file**, spelled the way the
segmentation pipeline spells feature columns — `velocity: {space: [x, y],
keypoint: [stickTip]}`, `pellet_stickClosest_dist: {}`. Build it with
`movement.kinematics` or `features/geometry.py` or your own code,
plot it in the GUI, list it. The model gets exactly that; there is no graph, no adjacency, no learned
geometry to reason about. Option 3 is described in {doc}`multimodal`.

## Why this shares the segmentation pipeline's workflow

The two pipelines learn different things from different inputs with different
models. What they share is everything *around* the model:

```{list-table}
:header-rows: 1
:widths: 26 37 37

* -
  - **`eto.segment`**
  - **`eto.spot`**
* - Learns
  - state events (spans)
  - Point events (moments)
* - Reads
  - Feature columns you choose
  - Video frames (+ optional pose features)
* - Model
  - Vendored DLC2Action architectures
  - Vendored E2E-Spot
* - Stage 1
  - `materialise()` — features to `(F, T)` arrays
  - `materialise()` — video to frames + index
* - Sessions, trial filter, split
  - `SessionSpec` / `TrialsConfig` / `SplitConfig`
  - **the same three, imported**
* - Holding a session out
  - `cross_validate()`
  - **the same call**
* - Predictions
  - The GUI's labels TSV
  - The GUI's labels TSV
```

Combining sessions into one training set, filtering trials by a metadata
column, drawing a split by whole trial, and holding one whole session out per
fold are properties of how the data is organised, not of either model. So
`eto.spot` imports `SessionSpec`, `TrialsConfig`, `SplitConfig`,
`open_session`, `filter_trials` and `assign_roles` unchanged, and defines only
what is its own: how video becomes clips, and which model reads them. The
vocabulary is the same on purpose — `eto.segment.Project` and
`eto.spot.Project`, each built from one YAML with `base:` and dotted overrides.

```{important}
What it does **not** share is the segmentation pipeline's `features:`
*section* (`columns:`, `drop_kinds`, `subsample`, preprocessing). `features:`
here is a flat list of variables in that pipeline's *column* spelling; a
`columns:` key under it is an error naming the difference.
```

## What you write

Everything a session needs is already known from the alignment — the video
path per trial, its frame rate, its offset — so a session is one line:

```yaml
sessions:
  - source: /data/derivatives/ses-01/behav        # the session folder
    name: '20260307_01'             # optional — see "The session lines" below
  - source: /data/derivatives/ses-02/behav
    name: '20260309_01'

frames: ../shared_frames            # optional: reuse frames another project decoded

trials:
  where: {num_pellets: [1, 2]}      # the trials-table filter, by column name

labels:
  classes: [31, 32]                # the point classes to spot
  camera: cam-1                    # which camera's video (one per project)
  crop: {x0: 120, y0: 40, x1: 500, y1: 380}   # optional: train on this part of the frame only

clip:
  context_s: 2.0                   # how much video the model sees at once
  resolution_ms: 10                # how finely a label may be placed; unset = as fine as the card allows
  positive_window_ms: 10           # +- this counts as the event during training

model:
  architecture: rny008_gsm         # eto.spot.architectures() lists the choices

train:
  epochs: 8
  epoch_frames: 250000
  split: {train_fraction: 0.6, val_fraction: 0.2, test_fraction: 0.2}

features:                          # optional — option 3, see multimodal
  velocity: {space: [x, y], keypoint: [stickTip, pellet]}
  pellet_stickClosest_dist: {}
```

There is no preprocessing and no individuals. A point event's subject comes
from the labels; the pixels are whatever the camera saw. **By default each
class happens at most once per trial**: the prediction is the tallest peak
of the class's curve, and its confidence reads a second peak as doubt, not
as a second event. A recording in which the event repeats is either cut into
trials at the alignment, or spotted with `infer.max_events_per_trial` raised
— then peaks at least `infer.min_event_gap_s` apart are separate events, the
confidence rule has to be `focus` or `peak`, and the threshold you flag below
needs calibrating, since the model now returns spurious events as well as
real ones (see {doc}`config`).

(target-spot-session-lines)=
### The session lines

`sessions:` is read the same way by both pipelines (`SessionSpec`, shared
with {doc}`the segmentation pipeline <../segment/index>`): a session is a
`source` plus whatever the stage you run needs from it.

- **`source`** — the session file. Always.
- **`labels_path`** — its curated labels TSV. Unset, it is `{stem}_labels.tsv`
  beside `source` (the GUI's own convention; the log says what was assumed).
  Training and `evaluate()` read it. A session you only
  **predict into** needs none — a `labels_path` naming a file that does not
  exist simply means the session has no labels, and it contributes nothing to
  a training set.
- **`video_dir`** — the folder searched for a trial's video when the
  alignment does not already resolve it. Optional when the alignment carries
  the paths; either way, a trial without a video for `labels.camera` can be
  neither trained on nor predicted, so for this pipeline the video has to be
  findable.
- **`name`** — the session id in every output (run folders, prediction
  sources, fold names). Optional: unset, it is the file's stem, and sessions
  whose stems collide — `Trial_data3.nc` in every `behav/` folder — are named
  by the nearest folder that tells them apart,
  `ses-000_date-20250309_01_Trial_data3`, listed in the log. Quote a name
  that is all digits, or YAML reads it as a number.

So a session to train on and one to predict into sit side by side:

```yaml
sessions:
  - source: C:/data/derivatives/sub-01/ses-000_date-20250503_02/behav
    labels_path: C:/data/derivatives/sub-01/ses-000_date-20250503_02/behav/Trial_data_labels.tsv
    video_dir: C:/VidData/20250503_02_Ivy
  - source: C:/data/derivatives/sub-01/ses-000_date-20250506_02/behav   # no labels: predict into it
    video_dir: C:/VidData/20250506_02_Ivy
```

`inference()` covers every listed session unless `sessions=` narrows it, so
that is the stage an unlabelled session is listed for. `cross_validate()` is
for labelled sessions only: each fold scores the session it held out, which
takes labels to score against.

`labels.crop` is cut from the decoded frame before the resize, so a tight
crop spends the model's pixels on less scene rather than shrinking the frame.
It must fit inside every trial's video for the named camera — checked at
materialise time, naming the trial it fails on. The GUI produces it (Tools ▸
Video: *Pick a crop for a config…* — drag a rectangle, get back the box in
this spelling).

(target-spot-one-camera)=
`labels.camera` names **one** camera: the model reads a single video per
prediction. With several cameras filming the same arena with large overlap,
pick the one that shows the event most clearly. With cameras covering
**non-overlapping** parts of the arena, tile them into one mosaic video as in
the birdpark recording system {cite:p}`ruttimann2025birdpark`, register the
mosaic in the alignment as a camera of its own, and name that. The cameras
have to be frame-synchronous, and `frame_height` is then shared between the
views, so each gets a fraction of the pixels.

`model.architecture` is a backbone plus a temporal module, in upstream's own
spelling — `rny008_gsm` is E2E-Spot's default, `rny008_msagsm` the multi-scale
variant below. The list is read off the vendored trainer's CLI, so it cannot
drift from what can actually be trained:

```python
for name in eto.spot.architectures():
    print(name, "-", eto.spot.describe_architecture(name))
```

(target-spot-seconds)=
## Every temporal setting is a duration

Upstream expresses every temporal hyperparameter in **frames**, tuned at
25 fps. The same numbers at 200 fps give the model an eighth of the context
in real time. So this config takes **`context_s`, `resolution_ms` and
`positive_window_ms`**, and derives upstream's frame counts from the video's
own rate:

```
stride     = round(resolution_ms / 1000 * fps)     # unset: the smallest stride that fits context_s in the card's frame budget
clip_len   = round(context_s * fps / stride)
dilate_len = round(positive_window_ms / 1000 * fps / stride)
```

`stride` is the label grid: at stride *k* the model sees every *k*-th frame,
so an event can only be placed to ±*k*/2 frames. `clip_len` is the number of
strided frames per clip. A config moved between a 200 fps rig and a 60 fps
one keeps meaning the same thing. Context and resolution pull against each
other — more context per clip means a coarser grid at a fixed memory budget —
so they are the two axes to tune. Left unset, `resolution_ms` is as fine as
the card's frame budget allows (every frame when it fits); spelled, a
combination whose loader batch would exceed the budget is refused, naming
the durations to change. See {doc}`config`.

```{note}
The recovered full-rate frame of a strided prediction is the **centre** of its
bin, `bin * k + (k - 1) / 2`. Reading it as `bin * k` makes every strided run
look systematically early by half a stride.
```

## The stages

```{list-table}
:header-rows: 1
:widths: 18 50 32

* - Stage
  - What it does
  - Cost
* - `materialise()`
  - Every trial's frames — only those between its start and stop when the
    video is longer than the trial — to `{video_id}/%06d.jpg` at the model's input height,
    plus `{split}.json` and `class.txt` in E2E-Spot's own schema; with
    `features:`, the listed columns per trial under `features/`. Resumable.
  - Minutes per session, once.
* - `train()`
  - Upstream's training loop, driven by the resolved config. One run under
    `runs/` — `{clip}` for pixels only, `{clip}_features` with the features
    fed in.
  - The expensive one.
* - `evaluate()`
  - The run's sweep-chosen epoch scored on the test split: per class, the
    labelled events, misses, spurious predictions, mean and median error in
    ms, hit rate at 10/20/50/100 ms → `test_metrics.yaml`.
    `zero_features=True` scores a features run with them zeroed.
  - Seconds, once predictions exist.
* - `compare()`
  - Every scored run as one table, `runs/compare.tsv`.
  - Seconds.
* - `inference()`
  - A run's predictions for chosen sessions — every trial with video,
    labelled or not — as the GUI's labels TSV (`labeling_method=automated`)
    plus the curves beside them, under `labels/predictions_spot_{run}_{time}/`.
    The epoch is the one the sweep ranks first on the run's own validation
    predictions. A features run exports the sessions' features first.
    Decodes the video directly — no frames are exported for a predicted session.
    The folder also holds `config.yaml` (the run's, as trained) and
    `inference.yaml` (run, epoch, the `infer:` settings), so it can be
    reconstructed without the project.
  - Minutes.
* - `cross_validate()`
  - One fold per session: hold it out, train on the rest, score it, predict
    into it. Each session ends up with predictions from a model that never
    saw it.
  - `n_sessions` × `train()`.
```

`materialise()` is role-agnostic, so one export serves every run and every
fold. Roles live in the run, exactly as `splits/*.bundle` do for the
segmentation pipeline.

### Frames on disk are for training only

`materialise()` writes every training trial's frames to a folder of JPEGs
because training reads them at random — a different clip from a different
trial at every step, over and over — and a video file cannot be read like
that. Inference reads each trial once, start to finish, so it never needs
individual frames on disk: `inference()` decodes the video directly into the
model. Nothing to export for a new session; nothing to clean up afterwards.

The frames are prepared the same way in both cases (crop, resize, and a JPEG
pass — in memory at inference), so the model sees the same pixels it trained
on. `infer.jpeg_roundtrip: false` skips that pass; leave it on.

## Confidence

A predicted event carries a `confidence` in the labels TSV, and the review
tools threshold on it. For this model the number is by default **`focus × ratio`** (`infer.confidence`): how much of the class's curve sits within the window of its peak,
times one minus the tallest *rival* peak over it. A lone sharp bump reads
near 1; a second candidate or a smeared bump pulls it down. Why not the
peak's height, the equations, and how this compares with the segmentation
pipeline's entropy confidence are on {doc}`the confidence page <../curation/confidence>`.

Curves are written through `ethograph.labels.onset_curves` — `(time, {label:
curve})`, numpy only, model-agnostic — so frame-by-frame review draws them
with no new GUI code. Every model writes under one convention,
`predictions_{model}_{timestamp}`, and runs are ordered by their timestamp.

## MSAGSM

MSAGSM {cite:p}`msagsm2025` is a drop-in replacement for the
Gate Shift Module {cite:p}`sudhakaran2020gsn`: the same gated split-and-shift, applied at **several
temporal dilations at once** and preceded by channel-grouped spatial
attention. GSM shifts by ±1 frame — at a high frame rate, a very narrow
aperture in real time. Widening it with `stride` costs label resolution;
MSAGSM widens the backbone's aperture without touching the label grid.

```yaml
model:
  architecture: rny008_msagsm       # E2E-Spot with MSAGSM in place of GSM
  shift_scales_ms: [40, 80, 120]    # the paper's {1, 2, 3} frames, at 25 fps
  attention_groups: 2
```

`ethograph/spot/msagsm.py` is written from the paper on top of the BSD-2 GSM
E2E-Spot already vendors (the reference MSAGSM repository carries no licence).
The vendored `model/shift.py` accepts any module with GSM's `(channels,
n_segment)` constructor, and `rny008_msagsm` hands it this one; the rest of
the network is untouched, so a GSM run and an MSAGSM run differ in exactly
one module. `shift_scales_ms` are durations resolved against the **strided**
clock. Three choices the paper leaves open are made deliberately: the branch
weights are a softmax; the module starts as the identity up to a uniform
scale, so a pretrained backbone is not perturbed at step 0; and the defaults
are the paper's (`{1, 2, 3}`, 2 groups).

