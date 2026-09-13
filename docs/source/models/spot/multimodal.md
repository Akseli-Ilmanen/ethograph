(target-spot-multimodal)=
# Pixels + Pose (extra features/ model distillation)

`eto.spot` reads video. Where pose exists too, there are two ways to use it,
and both start from the same thing — a flat list of variables in your session
file, spelled the way the segmentation pipeline spells feature columns:

```yaml
features:
  velocity: {space: [x, y], keypoint: [stickTip, pellet]}
  pellet_stickClosest_dist: {}          # a variable you computed into the .nc
```

Every entry is something you built — with `features/geometry.py` or your own
code — and can plot in the GUI before a model sees it. There is no graph, no
adjacency and no learned geometry: if you want "the distance from the middle
of the stick to the pellet", compute that variable, look at it, list it. A
session with several individuals names one per feature
(`{individual: [name]}`) or reads them all.

`materialise()` writes the listed columns once per trial under `features/`
(`{video_id}.npz`: `time`, `x (T, F)`, the events on that clock, `fps`) at
the pose's own rate, and every temporal setting is then resolved against that
rate exactly as the clip is against the video's. The columns are z-scored on
the training split; the statistics are saved, so a session predicted later is
put on the training scale rather than its own.

## Option 3 — the features ride into the GRU

With `features:` listed, `train()` hands the pixel model the columns as a
**second input**, concatenated to the CNN features before the bi-GRU. The run
is named `{clip}_features`, beside a video-only baseline. Every predicted
trial then needs its pose — `inference()` exports and scales a session's
features before predicting it — which is the price of using it.

```yaml
train:
  features_as_input: true     # the default when features: is listed
  features_dropout: 0.3       # share of training clips that see zeros in place of the features
```

Nothing makes a network use an input, so the contribution is measured, not
assumed: `evaluate(run, zero_features=True)` scores the trained model with
zeros in place of the block (`test_metrics_nofeatures.yaml`), and the
difference to `test_metrics.yaml` is what the pose adds, per class and
tolerance. `features_dropout` is what keeps that ablation meaningful — a share
of training clips see no features, so the pixels are trained to carry the
event on their own too.

Adding, removing or renaming a column afterwards means a new run: the block's
width and column order are part of the trained model.

## Option 4 — a pose teacher, distilled; video only at inference

For the case where pose exists only for the training sessions — label a few
frames in the GUI, `fill_poses()` → dense pose — and the deployed model must
read video alone:

```yaml
train:
  features_as_input: false     # the list is for the teacher only
teacher:
  shift_scales_ms: [40, 80, 160]
  hidden: 64
  depth: 4
  epochs: 30
```

`train_teacher()` fits the **pose teacher** (`pose_model.PoseSpotter`): the
listed features → a linear embedding → `depth` blocks of a parameter-free
multi-scale temporal shift (UMEG-Net's {cite:p}`umegnet2026`: a slice of channels copied from
`t ± k` for each scale, so a block sees several ranges of context at no
parameter cost) → a bi-GRU → a `K + 1` softmax, E2E-Spot's {cite:p}`hong2022e2espot` own output
contract. Minutes on a GPU. Every epoch writes val predictions in E2E-Spot's
schema, so `evaluate()` scores the teacher exactly as it scores a pixel run,
and the sweep-chosen epoch writes its per-clip embeddings under
`features/embeddings/`.

`distil()` then runs the two distillation steps inside the vendored trainer
(ADR 0007): the baseline's trunk + GRU learn to reproduce the teacher's
embedding on every clip that has pose (no labels), then the CNN is frozen and
the head learns the labels. The student reads video only. Runs land under
`runs/{baseline}_distil_{fingerprint}/`, where the fingerprint names the
feature list + teacher settings, so an edited list lands beside the earlier
result and `distil()` refuses embeddings another list's teacher wrote.

**The gate.** Score the teacher on the same test split as the baseline before
distilling, and distil only from a teacher that beats it. A student cannot
learn from a model that knows less than it does.

## Choosing between them

- Pose available at inference → option 3 (after the LightGBM lightgbm model {cite:p}`ke2017lightgbm`,
  which is the cheaper first try on the same features).
- Pose available only for the labelled sessions → option 4, gated as above.
- No pose → neither; E2E-Spot alone.

