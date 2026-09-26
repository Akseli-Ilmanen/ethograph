# Config reference

One YAML file per project. Relative paths resolve against the file's own
folder. `base: other.yaml` deep-merges this file over another.

Any key can be overridden without editing the file, with the same dotted
spelling the YAML has — passed to `Project(...)` or accumulated with
`update()`. Values are parsed as YAML, so `train.augment.stretch=[0.8,1.2]`
works:

```python
import ethograph as eto

project = eto.segment.Project("project.yaml", "model.architecture=mstcn")
project.update("train.run_name=mstcn", "train.augment.stretch=[0.8,1.2]")
```


## Top level

| Key | Default | Meaning |
|---|---|---|
| `root` | the config's folder | Project directory: `data/` and `runs/` live here. |
| `sessions` | required | List of sessions: `{source, labels_path, video_dir, alignment, name, video_feature_folders}`. `video_feature_folders` is `{variable: folder}` — one `{video stem}.npy` per trial's camera file, attached in memory at open ({doc}`video_features`). `alignment` reads the trials from that NWB instead of the source's own sidecar — the same file listed twice, once with its behaviour trials and once with windows tiled over a sleep epoch ({func}`ethograph.segment.windows.write_windows_alignment`), is two sessions of one recording; give the second a `name`. |
| `individual` | `null` | The one individual a single-animal project's samples belong to, stamped into every exported label's `individual`. Equivalent to `features.individuals: [name]`; set only one of them. |
| `trials.where` | `{}` | Metadata column → allowed values. The one trial filter; applied in every stage. |
| `trials.limit` | `null` | Keep only the first N trials that pass `where`, in session order — a smoke run before a long one. `null` = all. |



## `features`

| Key | Default | Meaning |
|---|---|---|
| `name` | `default` | Materialised dataset name → `{root}/data/{name}`. |
| `columns` | required | `feature → dim → values`. Every dim of a feature must be pinned except the individual dim (pinned per sample). A second individual dim is spelled `other: "*"` — the remaining individuals in dataset order. |
| `sin_cos` | `[]` | Features in `columns` that are **angles** — see below. |
| `individuals` | dataset's individual coord | Which individuals become samples. Required when the dataset has no individual dim but the labels name individuals. |
| `labels.mapping` | required | `mapping.txt` (`id name [branch] [event_type]`). |
| `labels.branch` | `0` | The one branch this model predicts — the exclusive target: one class per frame, a softmax. |
| `labels.branches` | unset | Several branches at once — a **multi-label** target: one binary channel per class, a sigmoid each, so a label of branch 0 and one of branch 1 coexist on a frame. Within one branch the classes stay exclusive (a *track*, see below). Spell `branch` or `branches`, never both. |
| `labels.subjects` | `self` | Whose labels are targets. `self`: the sample's own individual. `all`: also `other1`, `other2`, … as further channels (multi-label) — the model learns what the *other* animals are doing while it watches this one. At inference every animal's labels still come from its own sample; the other-channels are training signal only. |
| `labels.classes` | all state classes of the branch(es) | Subset of label ids to predict. |

A multi-label target is decoded one **track** at a time — a track is one
(subject, branch), the unit inside which the GUI never lets two labels
overlap. A channel is on where its sigmoid exceeds `infer.threshold`; where
two channels of one track are on at once the more probable wins; each
track's on/off sequence then goes through the same post-processing as an
exclusive run. So a prediction set can hold A's `flap` (branch 0) under A's
`peck` (branch 1), and never two branch-0 labels of A at once.

### `features.sin_cos`

An angle read as a plain number lies about its own geometry: 359° and 1° are
two degrees apart and the column says they are the furthest apart it ever
gets, and no amount of z-scoring repairs that jump. Name the feature here and
each of its columns is replaced by the two components of its angle:

```yaml
features:
  columns:
    angles: {keypoint: [beakTip, stickTip]}
  sin_cos: [angles]
```

gives `angles|keypoint=beakTip|sin`, `angles|keypoint=beakTip|cos`, and the
same pair for `stickTip` — the raw column is gone, not supplemented. The
units are the variable's own `units` attr (`rad` / `deg`, either spelling,
which is what {mod}`ethograph.features.geometry` writes); a variable that
declares none has them read off its values, logged at INFO, since a full turn
is 6.28 one way and 360 the other. A `units` that is not angular at all is an
error — it says the feature is not an angle.

The components live in `[-1, 1]` and mean what they say there, so they are
never z-scored or percentile-clipped, exactly like a column carrying
`attrs["normalise"] = 0`. Naming a feature that `columns` does not select is
an error.

### `features.changepoint_features`

Optional. Expands named raw changepoint masks into
{func}`~ethograph.features.changepoints.more_changepoint_features` once per
session, at `materialise`/`infer` time, and **merges the generated columns
straight into `features.columns`** — you never spell out a name like
`speed_troughs_cp_since` yourself. See {doc}`../../api/changepoints`
and `examples/segment_changepoint_features.ipynb` for what each output column
looks like. This is the one exception to "features are built with the
session, never by the pipeline": it is a deterministic expansion of a mask
already in the file, not a new modelling choice.

| Key | Default | Meaning |
|---|---|---|
| `sigmas` | derived | Kernel widths (in samples) of the proximity columns, e.g. `[2.0, 3.0, 5.0]`; the columns are named by rank (`_cp_prox0` is the narrowest). Left out, the ladder `horizon / (16, 8, 4)` is derived at materialise. |
| `distribution` | `laplacian` | `laplacian` or `gaussian`. |
| `inputs` | required | `feature → dim → values` — which raw changepoint masks to expand, dims pinned the same way as `features.columns` (the individual dim is still pinned per sample). |
| `transforms` | all six | Subset of `binary`, `proximity`, `offset`, `length`, `prominence`, `asymmetry` — which of `more_changepoint_features`'s column groups to keep. `binary` duplicates the raw mask (just marked `normalise=0`), so drop it if you already select the mask itself; `offset` is two columns, samples *since* the previous and *until* the next candidate, so a frame knows which side of its nearest candidate it is on; `length` is the log length of the candidate segment the frame sits in, the long range the offsets saturate on. `prominence` and `asymmetry` describe the *shape* of the signal the mask was detected on (its `target_feature`) at the nearest candidate, one column per window: prominence is how far the candidate sits below the lower of the two hills beside it, so a through between two movements scores high and a dip next to a flat stretch scores low; asymmetry is the mean after minus the mean before, positive where a movement starts, negative where one ends. Together they say which of several nearby candidates is a real boundary, which the four positional groups cannot. |
| `horizon` | derived | Where the `offset` columns saturate, in samples. Left out, materialise reads it off the curated labels: half the 5th-percentile duration, the largest radius that keeps both edges of the shortest behaviour apart. |
| `scale_by` | unscaled | A feature whose values scale the proximity columns by `exp(−x / mean x)`, emphasising candidates where it is low — `speed` favours the troughs at rest over a dip inside a movement; an amplitude envelope favours the silences between calls. Pinned like the mask (it may carry no dim the mask lacks). |
| `max_length` | derived | Where the `length` column saturates, in samples. Left out, materialise reads it off the curated labels: the 95th-percentile duration. |
| `windows` | derived | Half-widths of the `prominence` / `asymmetry` columns, in samples, one column each per window. Left out, `horizon × (1, 2, 4)`. |
| `note` | written | Filled in by materialise when it derived any of the scales — what was read off which labels — and carried into the run's `config.yaml`. Never set it by hand. |
| `merge` | `false` | OR every mask named in `inputs` into one `changepoints` mask before expanding, so the whole section costs **one** block of columns instead of one per mask × pinned dim. All merged masks must share a `target_feature`. |

Use `merge: true` when what matters is *that* something changed, not which
detector noticed. Without it you get one block of columns per mask per
keypoint, which can easily outnumber your kinematic features; with it you get
a single block named `changepoints_cp_*`. Each animal keeps its own
changepoints either way.

Xarray sessions only — pynapple changepoints are event times, not a dense
mask, so a pynapple session with this set raises immediately.

**The scales are read off the labels.** With `sigmas`, `horizon` and
`max_length` left out, `materialise` derives them from the durations of the
curated state events of the branch's classes (over the trials the config
selects, at the rate of the first mask), writes them into the dataset's
`columns.yaml` under `changepoint_features` together with a `note` saying
exactly what was read off what, and every later stage reads them back from
there: `train` saves them into the run's `config.yaml`, and `inference`
expands each session at the run's scales, labelled or not. Until then the
config is *unresolved*, and opening a session through it is an error that
says to materialise first. Spell any of the three (in samples) to pin it;
the note then lists only what was derived.

```yaml
features:
  changepoint_features:
    transforms: [proximity, offset, length]
    scale_by: speed
    inputs:
      speed_troughs: {keypoint: [beakTip, stickTip]}
      speed_turning_points: {keypoint: [beakTip, stickTip]}
```

This generates every `speed_troughs_cp_prox*`/`speed_turning_points_cp_prox*`
(etc.) column for both keypoints and merges them into `features.columns` —
naming `speed_troughs`/`speed_turning_points` there too, or under `inputs`
again elsewhere, is a config error (`config.features.columns already names
[...], which config.features.changepoint_features also generates`).

The generated columns are already in `[0, 1]`, so `preprocess` leaves them
alone — you do not need a `zscore_exclude` entry for any of them. The one
exception is the raw mask itself (`speed_troughs`): if you select it directly
in `features.columns`, rather than taking its `_cp_binary` twin from here, add
it to `zscore_exclude`.

### `features.neural`

Optional. Bins a pynapple session's **spike trains** into one dense feature
at session-open time — single-trial neural decoding with the same models,
split, metrics and prediction sets as behaviour. A session's units arrive as
a `TsGroup` (spike *times*, which no loader can read as a feature), and how
they are binned — bin size, smoothing, a rate versus a count — is a
modelling choice worth sweeping. So it is spelled here as pynapple
expressions, run at every open, and never written out as a feature file;
another binning is another config (`base:` this one, change `features.name`
and `transform`) and another materialised dataset.

| Key | Default | Meaning |
|---|---|---|
| `units` | `units` | The `TsGroup`'s key in the session (`units.npz` → `units`). |
| `name` | `rate` | The feature the transform produces, declared `kind: neural_feature`. |
| `transform` | required | pynapple expressions applied in order. `x` is the previous result (the `TsGroup` for the first); `nap`, `np` and `sliding_window` are in scope. The last must leave a `TsdFrame`, one column per unit. |

```yaml
sessions:
  - source: .../behav/pynapple/units.npz          # trials from .ethograph/alignment.nwb beside it or one folder up
    labels_path: .../behav/Trial_data_labels.tsv
individual: A
features:
  name: rate_5ms_boxcar25ms
  neural:
    units: units
    name: rate
    transform:
      - x.count(0.005) / 0.005                    # spikes / s in 5 ms bins → 200 Hz
      - sliding_window(x, window_size=0.025)      # 25 ms boxcar; reduction="sum" for counts per window
  columns: {}                                     # the neural feature alone; kinematics at the same rate may join it
  preprocess: {clip_percentiles: null}            # a rate's tail is signal, not an outlier
```

`sliding_window(x, window_size, step_size=None, reduction="mean")`
({func}`ethograph.features.neural.sliding_window`) reads the bin size off
the frame, so the window is in seconds whatever `count` was given; spike
times themselves cannot be windowed — bin first.

**The unit columns are read off the session, not written in the YAML.** The
feature's columns are the session's own unit ids, so `features.columns`
does not spell them (it may be empty when this section is set): `materialise`
resolves them from the opened session, records them in the dataset's
`columns.yaml` under `neural_columns`, `train` reads them back into the run's
`config.yaml`, and `inference` takes them from the run — a project with no
`data/` left still predicts. Spell `features.columns.{name}: {{name}_columns:
[ids]}` yourself to pin a subset (`{name}_columns` is the dim a lone
`TsdFrame` is selected on; a session where another frame shares the same
column labels calls it `columns`). Which is also why a neural project is
**one session**: units are only consistent within a recording, and two
sessions with different unit lists are refused by name. An xarray session
with this section set is refused too — it has no spike trains to bin.

The feature is z-scored per run like any kinematic column (`normalise: 1`),
and `train.drop_kinds: [neural_feature]` is its ablation. The predictions
land in the session's own `labels/predictions_{run}_{timestamp}/` like any
other run's, so they open in the GUI against the curated labels.

(segment-config-label-inputs)=
### `features.label_inputs`

Optional. Feeds the **labels of other branches** to the model as input
columns. Labelling is expensive, and the labels of an earlier question often
say a great deal about *when* the answer to a new one can happen — a peck
rarely comes before the head has turned. Every class of the named branches
becomes one column of a variable `label_inputs`, rendered per trial at
session-open time on the clock of `clock` and **merged straight into
`features.columns`** like `changepoint_features` — you never spell the entry
yourself.

| Key | Default | Meaning |
|---|---|---|
| `branches` | required | Branches of the mapping whose classes become inputs. Never a branch `features.labels` predicts. |
| `mapping` | `features.labels.mapping` | The `mapping.txt` the branches are read from. |
| `classes` | every class of the branches | Label ids to feed; an id outside the branches is refused. |
| `point_sigmas_s` | `[0.1, 1.0]` | Widths (seconds) of the Laplacian bump a **point** event is rendered at, one column each — a sharp one for timing, a wide one for reach. |
| `clock` | the first of `features.columns` | The feature whose time coordinate the columns are rendered onto, so they share its rate exactly. |
| `include_automated` | `false` | Read `automated` rows too — another model's predictions as input. Off, only `manual`/`curated` labels count, exactly as for targets. |

A **state** class is its on/off indicator: `1` inside every interval, `0`
outside. A **point** class is `max_i exp(−|t − t_i| / σ)` per width in
`point_sigmas_s` — the kernel `changepoint_features` draws around a
changepoint, for the same reason: the narrow peak points at the moment, the
long tails stay readable from far away. The columns are named by class:
`label_inputs|label_input=walk,individual=self`,
`label_inputs|label_input=peck@0.1s,individual=self`. They live in `[0, 1]` and are never z-scored
(`normalise: 0`); they are declared `kind: label_input`, so
`train.drop_kinds: [label_input]` trains the same model without them — the
ablation that says what the old labels are worth.

```yaml
features:
  columns:
    speed: {keypoint: [beakTip]}
  labels: {branch: 1}                 # the new question: branch 1's classes
  label_inputs:
    branches: [0]                     # the old answers: branch 0's classes, as inputs
    point_sigmas_s: [0.1, 1.0]
```

**An input branch is never a target branch.** A config whose `label_inputs.branches`
overlaps `features.labels.branch`/`branches` is refused at load: a model fed
the labels it is asked to predict learns to copy them, scores perfectly under
cross-validation, and has learned nothing about its other inputs. Putting the
two questions on two branches of the mapping is what makes the rule
checkable.

Each sample reads its **own animal's** labels: the variable carries the
individual dim (the dataset's, or one made from the config's individuals) and
is pinned per sample like every other feature. A trial with none of these
labels renders zeros — which is also what a session predicted later, before
anyone has labelled its old branch, reads as. The log says how many trials
carried them. Xarray sessions only.

### `features.preprocess`

All five keys live under `features.preprocess` in the YAML, but they run at
two different stages — the first four bake into the materialised `.npy`
files at `materialise` time; `zscore`/`zscore_exclude` are deferred to
`train`/`infer`, because mean/std can only be computed once a train split
exists. **If you inspect `data/{name}/features/*.npy` directly, expect it to
be un-z-scored** — that normalisation happens later, per run, and lands in
`runs/{run}/stats.npz` (see below).

Baked in at `materialise` (session-level, order below):

| Key | Default | Meaning |
|---|---|---|
| `likelihood_threshold` | `null` | Masks low-confidence tracking frame by frame; no column is ever dropped. A column holds exactly one keypoint, and the frames where that keypoint's `likelihood_feature` (of the sample's individual) is below this become NaN, and `interpolate` then bridges them. Columns without a keypoint are untouched; set without the feature in the session, it is an error. `null` = off. |
| `likelihood_feature` | `confidence` | The per-keypoint confidence feature. |
| `interpolate` | `true` | Linear interpolation over NaNs. |
| `clip_percentiles` | `[2, 98]` | Pull outliers in to this percentile range (`null` = off). Columns already on a fixed scale — unit vectors, angles, flags, changepoint features — are left alone. |

Applied at `train`/`infer` (run-level, not materialise):

| Key | Default | Meaning |
|---|---|---|
| `zscore` | `true` | Z-score each column, using statistics from the training trials only. The same statistics are reused at inference (`runs/{run}/stats.npz`). Columns already on a fixed scale are left alone. |
| `zscore_exclude` | `[]` | Extra feature names to leave un-z-scored, on top of those detected automatically. |

## `model`

| Key | Default | Meaning |
|---|---|---|
| `architecture` | `c2f_tcn` | Which network to train (the default is {cite:t}`singhania2021c2ftcn`). `eto.segment.architectures()` lists the names. |
| `params` | `{}` | Change individual hyperparameters of that network. Keys you leave out keep their default. |

```yaml
model:
  architecture: mstcn
  params: {num_f_maps: 64}     # only this one changes; the rest keep their defaults
```

(segment-config-model-params)=
### What you can put in `params`

Anything the architecture accepts. To see the full list of keys for a given
architecture, with a comment on each and its default value, open its file:

```
ethograph/segment/dlc2action/config/model/{architecture}.yaml
```

One name differs from its file: `mstcn` reads `ms_tcn3.yaml`. Every other
architecture matches.

The two skeleton-graph architectures (`specscalpel` {cite:p}`ji2026specscalpel`, `lady` {cite:p}`ji2026lady`) are the
exception to "params are architecture hyperparameters only": their `params`
also carry the **joint layout** — `keypoints` (the ordered keypoint names) and
`skeleton` (a skeleton-config YAML, an ndx-pose `.nwb`, or `[a, b]` pairs) — and,
for `lady`, the root-frame landmarks `root`/`spine`/`left`/`right`. These are
structural, not tunable, so `eto.segment.tunable_params(name)` lists only the
network numbers. Their defaults live in
`ethograph/segment/{specscalpel,lady}/config/defaults.yaml`.

An unknown key is an error naming the valid ones, so a typo cannot silently do
nothing — and it is raised *before* training starts, not by the constructor
half-way into a search.

Or ask, which is what a script sweeping several architectures wants:

```python
for name in eto.segment.architectures():
    print(name, eto.segment.tunable_params(name))
```

```{important}
The architectures share almost no hyperparameter names — `mlp` takes
`f_maps_list`, `mstcn` takes `num_f_maps`, `edtcn` takes `kernel_size`. So
`model.params` and any `search.params` entry under `model.params.*` are
**per architecture**: a sweep needs one search space each, not one shared
space. See {doc}`guide` for the loop.
```

For what each architecture is good at, see {doc}`guide`.

## `train`

| Key | Default | Meaning |
|---|---|---|
| `run_name` | `{architecture}_{features.name}` | Base run name. Each `train()` call gets its own directory, `runs/{run_name}_{YYYYmmdd-HHMM}/` — never overwrites a previous run. |
| `epochs` | `50` | How long to train. Every run trains its full budget. |
| `batch_size` | `1` | Trials per step. One whole trial at a time is the tested setting; raising it pads every trial in the batch out to the longest, which costs memory and changes what the C2F models' BatchNorm sees. |
| `learning_rate` | `1e-3` | Adam, held constant for the whole run. |
| `weight_decay` | `0` | Adam weight decay. |
| `grad_clip` | `1.0` | Clip the gradient norm to this. `0` turns clipping off. |
| `eval_every` | `5` | Score the validation trials every N epochs. Lower it to place the best checkpoint more precisely, at the cost of time. |
| `select_on` | `f1@50` | Which validation metric decides the kept checkpoint: `acc`, `edit`, `frame_f1`, `f1@50`, `f1@75`, `f1@90`. Pick the one that matches what you need from the model — `f1@50` for "did it find the behaviour", `f1@90` for "are the boundaries right", `edit` for "is the sequence of behaviours right". |
| `f1_thresholds` | `[0.5, 0.75, 0.9]` | IoU thresholds of the segmental F1 scores. |
| `seed`, `device` | `0`, auto | `device` = `cuda`, `mps`, `cpu`; auto picks the best available. |
| `drop_kinds` | `[]` | Feature categories to leave out of this run — the ablation axis. `[video_feature]` trains the same model without the video features. Applied to the materialised dataset's columns, so an ablation costs a run rather than a re-materialisation; columns whose `kind` is undeclared are always kept. A feature a changepoint mask was computed from (its `target_feature`) also counts as `changepoint_feature`, so `[changepoint_feature]` drops the expansions but keeps the signal they were found in, and `[kinematic_feature]` keeps that one kinematic column for the changepoints to read against. |
| `frame_weight` | `1.0` | Weight of `train.loss` in the total. `0` is a `ValueError` — there would be nothing to train on. |
| `oversample` | `{column: null, weights: {}}` | Draw some trials more often, by a metadata column: `{column: difficulty, weights: {hard: 3.0}}` draws a trial the GUI flagged hard during curation (`Ctrl+T`, or the post-curation review's threshold — see {doc}`../curation/difficulty`) three times as often. The column is read from each session's metadata table at training time, so flagging more trials never needs a re-materialisation; a value the mapping does not name keeps weight `1`. An epoch is still one draw per training sample, and validation and test are never reweighted. |
| `subsample` | `1` | Train and predict at `fs / subsample` — the temporal-resolution axis, run-level like `drop_kinds`, so one materialised dataset serves every rate. Every frame count the run reports (its metrics, its `_probs.npz`) is then in *its* frames, so runs at different rates are only comparable once their predictions are scored back on one grid. Striding, with no anti-alias filter. |

### Losses

The objective is the frame term, weighted by `train.frame_weight` —
`Objective` in {mod}`ethograph.segment.losses` computes and itemises it, and
its value lands in `metrics.tsv`/the console log, so a loss that stops moving
can be traced to the term that stopped moving.

| Term | Weight key | Default | Config section | Needs |
|---|---|---|---|---|
| frame (CE + consistency) | `train.frame_weight` | `1.0` | `train.loss` | any architecture |

`frame_weight: 0` is a `ValueError` ("nothing to train on").

(segment-config-train-loss)=
### `train.loss`

Cross-entropy per frame, plus a consistency term that penalises the
prediction changing from one frame to the next — that second term is what
stops the output flickering between classes mid-behaviour.

| Key | Default | Meaning |
|---|---|---|
| `alpha` | `0.001` | Weight of the consistency term. Raise it if predictions flicker; lower it if short behaviours are being swallowed by their neighbours. The default is DLC2Action's {cite:p}`kozlova2025dlc2action`, at which the term barely registers; MS-TCN's {cite:p}`abufarha2019mstcn` published value is `0.15`, which is what the earlier CETNet training script used and what `scripts/bench.py` pins when it asks whether the term helps at all. |
| `candidate_gate` | `null` | Do not smooth across changepoint candidates. The consistency term normally penalises every frame-to-frame change in the prediction, including the ones at real boundaries; with the gate on it skips the change into and out of each candidate frame, so a boundary that sits on a candidate is free and everything else is smoothed as before. `null` = on when the inputs include a `{var}_cp_binary` column, off otherwise; `true` / `false` force it. |
| `tau` | `4` | How large a frame-to-frame jump in log-probability that term still penalises; beyond `tau` it is truncated, so a genuine class change is not punished without limit. Ours, not a key of a config file upstream: DLC2Action writes MS-TCN's `tau` of 4 into the arithmetic as `clamp(..., max=16)`. Both it and `alpha` were tuned in the literature at 15–30 fps, so at a high sampling rate they are worth re-tuning together. |
| `focal` | `true` | Focus the loss on frames the model still gets wrong, instead of ones it already has right. |
| `gamma` | `2` | How sharply `focal` does that. Higher = more focus on hard frames. No effect when `focal: false`. |
| `weights` | `null` | Per-class multipliers on the cross-entropy. `null` treats every class alike. `dataset_inverse_weights` counts them from the run's training samples by DLC2Action's formula, `n_samples / n_frames` of each class, so a rare class counts for more; `dataset_proportional_weights` is the same relative to the most common class, whose weight is then 1. Or an explicit list, one entry per class (background first). The resolved numbers are logged and saved with the run. DLC2Action's own default is `dataset_inverse_weights`; here it is opt-in. |
| `hard_negative_weight` | `1` | Multi-label only (upstream's weight on frames marked as hard negatives — nothing here marks any, so it has no effect). |
| `exclusive` | follows the target | `true` (softmax cross-entropy) for a `labels.branch` target, `false` (a sigmoid per channel, `BCEWithLogits`) for `labels.branches` / `subjects: all`. Not a knob: spelling it against the target is refused. |

```yaml
train:
  loss: {alpha: 0.01}          # only this one changes; the rest keep their defaults
```

An unknown key is an error naming the valid ones. The defaults, with a comment
on each, live in `ethograph/segment/dlc2action/config/losses.yaml`.


**TODO**: See if inverse_frequency weights loss is detrimental

## `search`

Stage 1 of the workflow: Optuna {cite:p}`akiba2019optuna` over the config, every
trial a full training run scored by `train.select_on` on the **validation**
trials. Run it with `project.search()`.

| Key | Default | Meaning |
|---|---|---|
| `params` | `{}` | dotted config key → search space. Required to search. |
| `n_trials` | `20` | How many configurations to try. `project.search(n_trials=...)` overrides it for one call. |
| `timeout` | `null` | Stop the study after this many seconds, however many trials are left. |
| `name` | derived from the run name | Study name → `{root}/searches/{name}`, and `runs/{name}/trial000_…`. |
| `seed` | `0` | The sampler's seed. |
| `prune` | `true` | Abandon a trial whose validation curve is behind the running median at the same epoch. |
| `keep_weights` | `false` | Keep every trial's `best.pt`/`last.pt`. Off by default — a study is dozens of runs, and only the winner's weights are worth the disk. Each trial's config, split and metrics are kept either way. |

A **search space** is one entry of `params`, keyed by the same dotted path an
override uses, so there is exactly one spelling for a setting:

| Key | Applies to | Meaning |
|---|---|---|
| `type` | all | `float`, `int` or `categorical`. |
| `low`, `high` | float, int | The range, inclusive. |
| `step` | float, int | Quantise the range. |
| `log` | float, int | Sample on a log scale (needs `low > 0`) — the right choice for a learning rate. |
| `choices` | categorical | The list to pick from. |

```yaml
search:
  n_trials: 30
  params:
    train.learning_rate:     {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
    train.loss.alpha:        {type: float, low: 0.0, high: 0.5}
    model.params.num_f_maps: {type: int, low: 32, high: 256, step: 32}
    train.augment.mirror:    {type: categorical, choices: [true, false]}
```

The study is stored in `searches/{name}/study.db`, so calling `search()` again
**adds** trials to it rather than starting over. The winning draw is written to
`searches/{name}/best.yaml` as a config that inherits yours:

```yaml
base: ../../project.yaml
train:
  learning_rate: 0.00043
```

which is what stage 2 reads: `eto.segment.Project(result.config_path)`.

```{important}
A search tunes the **model**, not the feature engineering. The materialised
dataset is built once, before the study starts, and every trial reads it — so
`features.*` keys have no business in `search.params`.
```

## `video_features`

Which extractor, the settings that change its *features*, and the camera. In
seconds — frame counts come from each video's own rate. See {doc}`video_features`.

| Key | Default | Meaning |
|---|---|---|
| `extractor` | `s3d` | The network, by registry name: `s3d` (clip-wise; Kinetics-400 {cite:p}`kay2017kinetics` S3D {cite:p}`xie2018s3d`), `timm` (frame-wise; any timm {cite:p}`wightman2019timm` image backbone; `pip install 'ethograph[timm]'`) or `feral` (FERAL {cite:p}`skovorodnikov2025feral`, fine-tuned in its own environment from an export the project writes; {doc}`feral`). Names the sidecar suffix and the merged variable. |
| `model_name` | `null` | `timm` only: the backbone. `null` = `vit_base_patch14_reg4_dinov2.lvd142m` (DINOv2 ViT-B/14). |
| `stack_s` | `null` | `s3d` only: temporal extent of one window — how much motion context each frame's feature sees. `null` = 0.5 s. |
| `context_s` | `null` | `feral` only: seconds one FERAL chunk (64 frames) spans, resolved against each video's rate into FERAL's `chunk_step`; its chunk shifts scale with it. `null` = every frame, FERAL's own chunk. |
| `preset` | `null` | `feral` only: FERAL's `--mode` recipe laid over its default config — `lite`, `max` or `rare`. `null` = the default recipe. |
| `analysis_fps` | `null` | Rate the network sees; frames are skipped to reach it, never interpolated up, so halving this roughly halves the cost. `null` = every frame. |
| `camera` | `null` | Which camera's video to take, when the alignment holds several. |
| `crop` | `null` | `{x0, y0, x1, y1}`: one pixel box cut from every frame before the network sees it, in the GUI crop tool's numbers. |

A key that belongs to another extractor (`stack_s` with `timm`, `model_name`
with `s3d`, `context_s` with either) is an error naming the mismatch, never ignored. `stack_s` must be
at least 13 frames at the effective rate: the 0.5 s default works down to
26 fps; if it does not, the error names the shortest window that does.

```{note}
Everything else about the extraction — batch size, decode chunk, `fp16`,
device, S3D's `dense` ablation mode — is a performance detail with one
sensible answer, so it is not a project setting and naming it here is an
error. Build the extractor's own config
({class}`~ethograph.video_features.S3DConfig`,
{class}`~ethograph.video_features.timm_extract.TimmConfig`) yourself in the
rare case you need one.
```

Sidecars go to `{root}/video_features/`; FERAL's export, checkpoints and
embeddings to `{root}/feral/`.

## `infer`

| Key | Default | Meaning |
|---|---|---|
| `run` | `train.run_name` | Run name (exact or base) or a run directory under `runs/`; a base name resolves to its most recently trained timestamped run (`project.inference(run=…)` overrides it for one call). |
| `threshold` | `0.5` | Multi-label targets: a channel is on where its sigmoid exceeds this. Exclusive targets argmax and never read it. |

### `infer.postprocess`

Purge → stitch → snap → purge, through the same functions as the GUI's
changepoint correction. Also used for the *post-processed* numbers in
`test_metrics.yaml`.

The interval steps are the GUI's *CP Correction* section under other names,
and the default way to fill them is to **take the GUI's numbers**:

```yaml
infer:
  postprocess:
    gui_settings: true          # ~/.ethograph/gui_settings.yaml (or a path)
    max_shrink_s: 0.1           # anything spelled beside it still wins
```

`gui_settings` reads the file every time the config is loaded, so the
pipeline stays in step with what you tune in the GUI; a saved run config
carries the resolved values explicitly (plus the path they came from), so a
finished run does not change when the GUI does. The GUI's step checkboxes
read as zeroed parameters (purge off → `min_duration_s: 0`, stitch off →
`stitch_gap_s: 0`, snap off → `changepoint_correction: false`). Spell the
values instead when one project needs settings the GUI does not hold.
`changepoints` has no GUI counterpart and is always the config's.

| Key | Default | Meaning |
|---|---|---|
| `gui_settings` | `null` | `true` or a path: read the interval-step values below from the GUI's `gui_settings.yaml`; explicit keys override. |
| `min_duration_s` | `0` | Drop predicted labels shorter than this (`0` = off). |
| `label_thresholds` | `{}` | Per-label-id minimum durations overriding `min_duration_s`. |
| `stitch_gap_s` | `0` | Merge same-label predictions separated by less than this. |
| `changepoint_correction` | `false` | Snap onsets/offsets to the session's changepoint masks (xarray sessions only). |
| `changepoints` | `{}` | Selections pinning those variables (e.g. `{keypoint: beakTip}`); the individual is pinned per sample. |
| `max_expansion_s`, `max_shrink_s` | `0.05`, `0.05` | How far an interval edge may move outwards / inwards when snapping. |

## What a run writes

```
runs/{run}/
  config.yaml        the resolved config (absolute paths) this run was trained with
  columns.yaml       input layout, copied from the materialised dataset
  classes.yaml       class index ↔ label id
  stats.npz          normalisation statistics of the training samples
  splits/            train.bundle / val.bundle / test.bundle — sample keys per role
  metrics.tsv        one row per validation: epoch, loss, val metrics, + test_raw_*/test_post_* diagnostic
  best.pt / last.pt  weights selected on validation / at the end
  test_metrics.yaml  test evaluation of best.pt: raw and post-processed, overall and class-wise
  eval.pdf           overall + class-wise F1, onset/offset |Δ| histograms
  train.log          everything logged during this run (always written, on top of the console)
  infer.log          everything logged by every `project.inference()` call against this run (appended)
runs/compare.tsv     written by `project.compare()`, which also returns it as a DataFrame
```

A search's trials and a cross-validation's folds are ordinary runs, nested one
level deeper (`runs/{search or cv name}/trial000_…`, `runs/{cv name}/fold-…`)
so they do not bury the runs you trained by hand — and so `project.compare()`,
which reads only the top level, keeps showing those.

## What a search and a cross-validation write

```
searches/{name}/
  study.db           the Optuna storage — calling search() again resumes it
  trials.tsv         one row per trial: number, state, value, best_epoch, run_dir, parameters
  best.yaml          `base:` your config + the winning parameters — what stage 2 reads
  search.log         everything logged during the study
cross_validation/{name}/
  folds.tsv          one row per fold: session, run, run_dir, best_epoch, held-out metrics, predictions
  crossval.log       everything logged during the folds
```
