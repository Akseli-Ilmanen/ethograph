(target-segment)=
# Segmentation pipeline

Learn the **state events** you curated in the GUI from trial-structured
sessions, and predict them back into the GUI. A code-first pipeline — one
YAML config, one object with a method per stage — that reads every backend
the GUI reads (`.nc`, pynapple, NWB) through the same loaders and writes its
predictions in the GUI's own labels format.

```python
import ethograph as eto                   # after installing PyTorch, then `uv pip install "ethograph[model]"`

project = eto.segment.Project("project.yaml")
project.materialise()   # feature engineering → materialised dataset

# Stage 1 — find good settings on a 60/20/20 split of the trials
best = project.search()

# Stage 2 — cross-validate them, one fold per session, and review the
# predictions in the GUI
eto.segment.Project(best.config_path).cross_validate()
```

```{tip}
New here? {doc}`quickstart` is this page cut down to one architecture, three
kinematic features and two sessions — a config you can copy and a model
trained in four lines. Come back when you want the choices back.
```

## The two stages of a workflow

The pipeline has one set of stages you *run* (materialise, train, infer) and
two ways to *use* them. Which one you are in decides how the trials are
divided, and that is the only thing that changes between them.

```{list-table}
:header-rows: 1
:widths: 14 43 43

* -
  - **Stage 1 — search**
  - **Stage 2 — cross-validate**
* - Question
  - *What settings work?*
  - *Where is the model still wrong?*
* - Split
  - The trials of every session, pooled and cut **60/20/20** by
    `train.split`.
  - One whole **session** held out per fold; every other session trains.
* - Chooses on
  - The **validation** trials — that is all validation is for.
  - Nothing. The settings are already fixed.
* - Call
  - `project.search()`
  - `project.cross_validate()`
* - Gives you
  - `searches/{name}/best.yaml` — a config inheriting yours with the winning
    parameters pinned.
  - A prediction set beside **every** session, each written by a model that
    never saw it. Load them in the GUI.
```
In stage 2 (cross-validation), we recommend taking an entire behavioural session
as the held-out test set. This is convenient as you can use the GUI to open any trial
in that session and compare your ground truth labels with predictions visually.

### The stages themselves

```{list-table}
:header-rows: 1
:widths: 18 42 40

* - Stage
  - What it does
  - Writes
* - **Feature engineering**

    `project.materialise()`
  - Selects every configured *feature column* of every *sample* (one
    trial × one individual), applies the fixed preprocessing chain, encodes
    the branch's curated labels per frame.
  - `{root}/data/{name}/`

    the materialised dataset
* - **Train**

    `project.train()`
  - Fits an *architecture* on the training samples, validates every few
    epochs, keeps the best checkpoint, evaluates the test samples once (raw
    and post-processed). Materialises first if needed. Returns a `RunResult`
    (`run_dir`, `best_epoch`, `best_score`, `test_metrics`).
  - `{root}/runs/{run}/`

    config, layout, stats, weights, metrics
* - **Search**

    `project.search()`
  - Optuna over `search.params`: each trial is a training run, scored by
    `train.select_on` **on the validation trials**. Resumable — the study
    lives in a SQLite file. Returns a `SearchResult`.
  - `{root}/searches/{name}/`

    `study.db`, `trials.tsv`, `best.yaml`
* - **Cross-validate**

    `project.cross_validate()`
  - One fold per session: train on the rest, predict the held-out one.
    `folds=` runs only some of them. Returns one DataFrame row per fold.
  - `{root}/cross_validation/{name}/folds.tsv`

    plus a prediction set per held-out session
* - **Inference**

    `project.inference()`
  - Runs a run over the sessions and post-processes the predictions (purge →
    stitch → snap to changepoints → purge). Returns the prediction paths;
    `run=` picks another run, `sessions=` narrows to a few.
  - `{session folder}/labels/predictions_{run}_{timestamp}/{stem}_predictions.tsv`

    plus `_probs.npz`
```

`project.config` is the resolved config and `project.root` the project
directory; `project.sessions()` opens every session, `project.runs()` names
the runs already trained, and `project.load_run()` returns a trained run
ready to predict with.

The vocabulary — session, trial, sample, feature column, branch, curated
label, materialised dataset, architecture, run, role, prediction set — is
pinned in the repository's `CONTEXT.md`.

## Features are built with the session, not by the pipeline

The pipeline never invents features: it **selects** variables that already
exist in the session file and pins their dims. Anything you want a model to
see — egocentric coordinates, pairwise distances, headings, changepoint
proximity, video embeddings — is a data variable you add when you build
the `.nc` (or pynapple / NWB file), which also makes it plottable in the GUI
so you can review what the model will read.

```python
import ethograph as eto
from ethograph.features.changepoints import add_changepoint_features
from ethograph.features.geometry import egocentric_position, heading, intra_distances

ds["position_ego"] = egocentric_position(ds["position"], "body", heading_keypoint="head")
ds["intra"] = intra_distances(ds["position"])
ds["heading"] = heading(ds["position"], "body", "head")   # unit vector: attrs normalise=0
ds = add_changepoint_features(ds, sigmas=[2, 3, 5])       # *_cp_binary, *_cp_sigma3, …
```

Every function in {mod}`ethograph.features.geometry` takes and returns
xarray (the `movement` convention: `position (time, space, keypoint,
individual)`), works in 2-D and 3-D, and keeps the `individual` dim. Outputs
that must never be z-scored (unit vectors, angles, binary flags, segment ids)
carry `attrs["normalise"] = 0`; the pipeline honours it.

Spike trains are the exception in principle: a pynapple session's units are
a `TsGroup` of event times, which no loader can read as a feature, and how
they are binned is a modelling choice worth sweeping. `features.neural` spells
that binning as pynapple expressions and applies it at every session open —
single-trial neural decoding with the same models and prediction sets. See
{doc}`config` (`features.neural`).

Video features are the one exception in mechanics, not in principle — a
pretrained network (S3D {cite:p}`xie2018s3d` by default, a timm {cite:p}`wightman2019timm` backbone such as DINOv2 {cite:p}`oquab2024dinov2` by name) is expensive enough to
run once per video and cache. See {doc}`video_features`; the short version is

```python
# a folder of videos, before any session exists
eto.segment.extract_videos(["/data/videos"], "/data/features", stack_s=0.5)

# or: the videos this config's sessions already name, then merge them in
project.video_features(merge=True)
```

which leaves an ordinary `s3d (time, s3d_dims)` variable on each trial —
plottable in the GUI, and named in `features.columns` like anything else.

## A config

```yaml
root: .                       # data/ and runs/ live here (default: this file's folder)

sessions:                     # every path explicit — no sidecar/folder guessing
  - source: ../sub-01/ses-01/behav/Trial_data.nc
    labels_path: ../sub-01/ses-01/behav/Trial_data_labels.tsv
    video_dir: ../videos
  - source: ../sub-01/ses-02/behav/Trial_data.nc
    labels_path: ../sub-01/ses-02/behav/Trial_data_labels.tsv
    video_dir: ../videos
  - source: ../sub-01/ses-03/behav/Trial_data.nc
    labels_path: ../sub-01/ses-03/behav/Trial_data_labels.tsv
    video_dir: ../videos
  - source: ../sub-02/ses-01/behav/Trial_data.nc
    labels_path: ../sub-02/ses-01/behav/Trial_data_labels.tsv
    video_dir: ../videos

trials:
  where: {num_pellets: [1, 2]}     # the metadata-table filter, every stage

features:
  name: kin_cp                     # → data/kin_cp/
  columns:                         # feature → dim → values; the individual dim is never listed
    position_ego: {space: [x, y, z], keypoint: [beakTip, stickTip]}
    speed:        {keypoint: [beakTip]}
    speed_cp_sigma3: {keypoint: [beakTip]}
    inter_beak:   {other: "*"}     # a second individual dim: the others, in order
  preprocess:
    likelihood_threshold: 0.6      # needs a `confidence` feature
    clip_percentiles: [2, 98]
    zscore: true                   # statistics from the training samples only
  labels:
    mapping: mapping.txt
    branch: 0                      # one model per branch

model:
  architecture: c2f_tcn            # eto.segment.architectures()
  params: {num_f_maps: 128}

train:
  run_name: c2f_kin_cp
  epochs: 100
  eval_every: 5
  select_on: f1@50               # what val is scored on, and what a search maximises
  loss: {alpha: 0.01}            # only the keys you want to change
  augment: {noise_std: 0.05, stretch: [0.8, 1.2], mirror: false, rotate_deg: 0}
  # the three ratios, drawn by whole trial across every session; they must sum to 1
  split: {train_fraction: 0.6, val_fraction: 0.2, test_fraction: 0.2}

search:                          # stage 1 — keys are the same dotted paths an override uses
  n_trials: 30
  params:
    train.learning_rate: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
    train.loss.alpha: {type: float, low: 0.0, high: 0.5}
    model.params.num_f_maps: {type: categorical, choices: [64, 128, 256]}

infer:
  postprocess:
    min_duration_s: 0.05
    stitch_gap_s: 0.015
    changepoint_correction: true
    changepoints: {keypoint: beakTip}
    max_expansion_s: 0.05
    max_shrink_s: 0.05
```

Every key is documented in {doc}`config`. Two conveniences: `base: other.yaml`
merges a file over another, and any key can be overridden without editing the
file, using the same dotted spelling the YAML has — passed to the constructor
or accumulated with `update()`:

```python
project = eto.segment.Project("project.yaml", "model.architecture=mstcn")
project.update("train.run_name=mstcn", "train.loss.gamma=2")
```

Values are parsed as YAML, and the config is rebuilt from the file each time,
so a typo is caught there rather than half-way through a run. That is how a
benchmark is written — one base file, a loop over overrides, then `compare()`:

```python
for architecture in ("c2f_tcn", "mstcn", "mlp"):
    eto.segment.Project(
        "project.yaml",
        f"model.architecture={architecture}",
        f"train.run_name={architecture}",
    ).train()

print(eto.segment.Project("project.yaml").compare())
```

(target-segment-session-lines)=
### The session lines

`sessions:` is read the same way by both pipelines (`SessionSpec`, shared
with {doc}`pixel event spotting <../spot/index>`): a session is a `source`
plus whatever the stage you run needs from it.

- **`source`** — the session file. Always.
- **`labels_path`** — its curated labels TSV. Unset, it is `{stem}_labels.tsv`
  beside `source` (the GUI's own convention; the log says what was assumed).
  Training and scoring read it. A session you only **predict into** needs
  none — a `labels_path` naming a file that does not exist simply means the
  session has no labels, and it contributes nothing to a training set.
- **`video_dir`** — the folder searched for a trial's video when the
  alignment does not already resolve it. Optional here: this pipeline reads
  feature columns, and only the video features
  ({doc}`video_features`) ever open a video.
- **`name`** — the session id in every output (fold names, prediction
  sources, log lines). Optional: unset, it is the file's stem, and sessions
  whose stems collide — `Trial_data3.nc` in every `behav/` folder — are named
  by the nearest folder that tells them apart,
  `ses-000_date-20250309_01_Trial_data3`, listed in the log. Quote a name
  that is all digits, or YAML reads it as a number.

So a session to train on and one to predict into sit side by side:

```yaml
sessions:
  - source: ../sub-01/ses-000_date-20250503_02/behav/Trial_data3.nc
    labels_path: ../sub-01/ses-000_date-20250503_02/behav/Trial_data_labels.tsv
  - source: ../sub-01/ses-000_date-20250506_02/behav/Trial_data3.nc     # no labels: predict into it
```

`inference()` covers every listed session unless `sessions=` narrows it, so
that is the stage an unlabelled session is listed for. `cross_validate()` is
for labelled sessions only: each fold scores the session it held out, which
takes labels to score against.

## What a sample is

One **(trial, individual)**. Its columns are the configured features with
the individual dim pinned to that individual (`individual=self` in the
layout) and, for pair features, the `other` dim enumerating the remaining
individuals in dataset order (`other1`, `other2`, …). Its target is that
individual's labels *as actor*. So a trial with two individuals is two
samples, a dataset with one individual is unaffected, and every session must
carry the same number of individuals.

Only **`manual` and `curated` labels** ever become training targets; an
`automated` label — the output of any model — never does. Point events are
skipped (the lightgbm model owns them). One branch per model is the exclusive
target; `features.labels.branches` lists several and the target becomes
multi-label — one binary channel per (subject, class), decoded one *track*
(subject, branch) at a time, so labels of different branches overlap and
labels of one branch never do (see the config reference).

## The materialised dataset

`{root}/data/{features.name}/` uses the layout of the [action-segmentation literature](https://github.com/nus-cvml/awesome-temporal-action-segmentation) for interopability with newer architectures.

```
features/{key}.npy       (F, T) float32, session-level preprocessed
groundTruth/{key}.txt    one class name per frame
mapping.txt              "{index} {name}", contiguous, 0 = background
index.tsv                key → session, source, trial, individual, n_frames, fs, n_labelled
columns.yaml             the input layout: names, normalise flags, vector groups
classes.yaml             class index ↔ label id
```

`key` is `{session_id}_trial{trial}_{individual}`; `session_id` is the
source's stem plus a path hash, so two `Trial_data.nc` never collide. Roles
and normalisation statistics are *not* part of the dataset — they belong to
a run (`runs/{run}/splits/*.bundle`, `stats.npz`).

## Architectures

Ten networks are available; `eto.segment.architectures()` lists them.
Switching between them is a one-line change, and `project.compare()` puts the
runs side by side, so trying two or three is cheap.

| Name | Shape | When to reach for it |
|---|---|---|
| `c2f_tcn` {cite:p}`singhania2021c2ftcn` | U-Net over time | The default. Fast, and sees long-range context cheaply. Needs trials of at least 384 frames. |
| `c2f_transformer` {cite:p}`kozlova2025dlc2action` | `c2f_tcn` + attention | Same size limit; worth a run when `c2f_tcn` misses long-range structure. |
| `mstcn` (MS-TCN3) {cite:p}`kozlova2025dlc2action` | Dilated TCN, refined in stages | Works at any trial length. The usual baseline. |
| `asformer` {cite:p}`yi2021asformer` | Sliding-window attention + decoders | Strongest context modelling, several times slower per epoch. |
| `edtcn` {cite:p}`lea2017edtcn` | Encoder–decoder, wide kernels | Small and quick. |
| `rnn` | Bidirectional GRU or LSTM over the trial | Ours: a recurrent baseline, any trial length. Defaults in `ethograph/segment/models/config/rnn.yaml`. |
| `mlp` {cite:p}`kozlova2025dlc2action` | Per-frame, no temporal context | A floor to compare against: how much is temporal context buying you? |
| `motionbert` {cite:p}`zhu2023motionbert` | Attention across joints, then across time | For pose columns that factor into joints: set `model.params.num_joints` (it has no default and must divide the column count). Reads a fixed 128-frame window at a time. |
| `specscalpel` {cite:p}`ji2026specscalpel` | Skeleton graph + frequency-selective filtering | **Pose-native.** Reads a skeleton of keypoints and sharpens the boundaries between adjacent behaviours in the frequency domain. Needs `model.params.keypoints` and (optionally) `skeleton`. |
| `lady` {cite:p}`ji2026lady` | `specscalpel` + a learned Lagrangian-dynamics stream | **Pose-native.** Adds a physics-informed stream (torque, power, energy) over the skeleton's generalised coordinates. Needs `keypoints`, a `skeleton` with edges, and the root-frame landmarks `root`/`spine` (+ `left`/`right` in 3D); reads raw positions only. |

`eto.segment.tunable_params(name)` lists each one's hyperparameters; set only
the ones you want to change under `model.params`. For the vendored models they
come, with a comment on each and its default, from
`ethograph/segment/dlc2action/config/model/{file}.yaml` (`mstcn` reads
`ms_tcn3.yaml`); `specscalpel` and `lady` read their skeleton-graph defaults, and
`rnn` reads `ethograph/segment/models/config/rnn.yaml`. The loss is configured the same
way under `train.loss`, from `config/losses.yaml`. See {doc}`config`.

```{note}
The networks and the loss come from
[DLC2Action](https://github.com/amathislab/DLC2Action) (AGPLv3, compatible
with this project's GPLv3 — see `ethograph/segment/dlc2action/NOTICE.md`). To
plug in your own, register a builder with `@register_architecture("name")`, or
ship one from another package through the `ethograph.segment.architectures`
entry-point group. It takes `(x (B,F,T), mask (B,1,T))` and returns
`logits (S,B,C,T)`, finest stage last — or a `ModelOutput` carrying those
logits.
```

## Stage 1: find the settings

`project.search()` runs an Optuna study over
`search.params`. Every trial is a full training run, and its score is
`train.select_on` measured on the **validation** trials — the one thing
validation is for. `test` is never read, so it is still an honest number at
the end.

```python
result = project.search()               # or search(n_trials=50)
print(result.best_params, result.best_score)
print(result.trials)                    # one row per trial
```

A parameter is keyed by the same dotted path an override uses, so there is one
spelling for "learning rate" and it works in the file, in an override and in a
search space alike. Three kinds of space, mirroring Optuna's three suggest
calls:

```yaml
search:
  n_trials: 30
  params:
    train.learning_rate:     {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
    model.params.num_f_maps: {type: int, low: 32, high: 256, step: 32}
    train.augment.mirror:    {type: categorical, choices: [true, false]}
```

The winner is written to `searches/{name}/best.yaml` — a config that inherits
yours and pins the parameters that won, which is what stage 2 reads:

```yaml
base: ../../project.yaml
train:
  learning_rate: 0.00043
  loss: {alpha: 0.087}
```

The study itself lives in `searches/{name}/study.db`, so calling `search()`
again **adds** trials rather than starting over — stop a study, look at
`trials.tsv`, continue it. Trials that fall behind the running median are
abandoned early (`search.prune`), and only the winning trial keeps its weights
(`search.keep_weights`); every trial's config, split and metrics are kept
either way.

### Sweeping several architectures

Searching more than one architecture is a loop of searches, not one search
with `model.architecture` in the space — because the architectures share
almost no hyperparameter names (`mlp` takes `f_maps_list`, `mstcn` takes
`num_f_maps`), so each needs its own space. `eto.segment.tunable_params(name)`
lists what each one accepts.

Give each its own `train.run_name`: that names the study as well, and without
it every architecture would pool incomparable trials into one `study.db`.

```python
import ethograph as eto

SHARED = {   # keys that mean the same thing to every architecture
    "train.learning_rate": {"type": "float", "low": 1.0e-5, "high": 1.0e-2, "log": True},
    "train.loss.alpha": {"type": "float", "low": 0.0, "high": 0.5},
}
VARIANTS = {
    "asformer_enc": {                       # ASFormer, encoder only
        "architecture": "asformer",
        "params": {"num_decoders": 0},      # pinned
        "space": {"model.params.num_f_maps": {"type": "categorical", "choices": [64, 128, 256]}},
    },
    "asformer_dec": {                       # ASFormer as published: does refinement pay?
        "architecture": "asformer",
        "params": {},
        "space": {"model.params.num_decoders": {"type": "int", "low": 1, "high": 3}},
    },
    "mstcn": {
        "architecture": "mstcn",
        "params": {},
        "space": {"model.params.num_R": {"type": "int", "low": 1, "high": 3}},
    },
}

eto.segment.Project("project.yaml").materialise()      # once, for every variant

results = []
for variant, spec in VARIANTS.items():
    overrides = eto.segment.as_overrides({
        "model.architecture": spec["architecture"],
        "train.run_name": f"{variant}_kin",            # → searches/search_{variant}_kin/
        "model.params": spec["params"],
        "search.params": {**SHARED, **spec["space"]},
    })
    result = eto.segment.Project("project.yaml", *overrides).search()
    results.append((result.best_score, variant, result.config_path))

best_score, best_variant, best_config = max(results)
eto.segment.Project(best_config).cross_validate()      # stage 2 on the winner only
```

Two entries may share an architecture and differ only in what is pinned versus
searched — the two ASFormers above ask "does refinement earn its cost here?",
which one study cannot answer cleanly because a pinned `num_decoders: 0` and a
searched `1..3` are different questions.

`eto.segment.as_overrides({...})` turns a dict into the dotted `key=value`
strings `Project` takes, through YAML — so a nested dict or a float in
exponent form survives, where an f-string would hand over Python's `repr`.

Cross-validate the **winner only**: a fold is a training run per session, so
it is not something to spend on the variants that already lost.

## Stage 2: cross-validate, and look at the mistakes

With the settings settled, hold out a whole **session** per fold:

```python
best = eto.segment.Project(result.config_path)
folds = best.cross_validate()           # one fold per session
```

Fold *i* trains on every session but the *i*-th and then predicts that one, so
each session ends up with a prediction set from a model that never saw a frame
of it. `folds` is one row per fold — its run, its metrics on the held-out
session, and the path of the prediction set:

| session | run | postprocessed.f1@50 | predictions |
|---|---|---|---|
| ses-01 | fold-ses-01_… | 0.71 | …/ses-01/behav/labels/predictions_fold-ses-01_…_{timestamp}/Trial_data_predictions.tsv |

Folds are independent, so you can run some of them:

```python
best.cross_validate(folds=["ses-01", "ses-02"])     # two folds, not all four
```

which is how you compare two parameter sets at a fraction of the cost. When
you actually want to *inspect* a session in the GUI, run its own fold — a
model that trained on the session it is predicting tells you nothing.

### One session: fold by trial

A session cannot be held out when it is the only one — which is every
neural decoding project (`features.neural`), since units exist in one
recording only. Fold by **trial** instead:

```python
folds = project.cross_validate(n_folds=5)
```

Every trial is dealt into exactly one of the five folds (seeded by
`train.split.seed`, so the folds are the same for every transform you
compare); fold *k* trains on the other four and predicts its own, through
`train.split.holdout_trials`; and the five prediction sets are merged into
**one** per session, under
`labels/predictions_cv_{run_name}_{timestamp}/`, so the whole session opens
in the GUI with every trial predicted once by a model that never saw it —
`cross_validation/{name}/predictions.tsv` lists it. `folds.tsv` still has one
row per fold, with the trials it held out and its metrics on them; each
row's `prediction_source` says which fold wrote it.

Comparing binnings is then a loop over configs that share this one
(`base:`), one materialised dataset each:

```python
from ethograph.segment import as_overrides

for name, steps in {
    "rate_5ms_boxcar25ms": ["x.count(0.005) / 0.005", "sliding_window(x, window_size=0.025)"],
    "sqrt_count_10ms": ["x.count(0.01)", "np.sqrt(x)"],
}.items():
    eto.segment.Project(
        "decoding.yaml",
        f"features.name={name}",
        f"train.run_name={name}",
        *as_overrides({"features.neural.transform": steps}),
    ).cross_validate(n_folds=5)
```

giving `cross_validation/cv_{name}/folds.tsv` per transform — the same
folds, so the numbers are paired.

## Reviewing predictions in the GUI

A prediction set is a labels TSV in the GUI's own format, every row
`labeling_method = automated` with the model's confidence. Load it with
**File ▸ Import labels…** and it enters the {doc}`curation <../curation>`
workflow: automated labels draw dotted, the grid views rank them by
confidence, and every label you confirm becomes `curated`. Loading several
runs side by side for comparison is noted in {doc}`later`.

This is what makes stage 2 worth its cost: the fold's predictions and the
labels you drew are the same kind of object on the same axis, so "60% F1"
becomes *which* class, *which* trials and *how far off* the boundaries are.


```{toctree}
:maxdepth: 1
:hidden:

quickstart
config
video_features
later
```
