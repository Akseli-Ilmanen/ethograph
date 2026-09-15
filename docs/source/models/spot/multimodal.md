(target-spot-multimodal)=
# Pixels + Pose (extra features)

`eto.spot` reads video. Where pose exists too, it can read that as well — a
flat list of variables in your session file, spelled the way the
segmentation pipeline spells feature columns:

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

## The features ride into the GRU

With `features:` listed, `train()` hands the pixel model the columns as a
**second input**, concatenated to the CNN features before the bi-GRU. The run
is named `{clip}_features`, beside a video-only baseline. Every predicted
trial then needs its pose — `inference()` exports and scales a session's
features before predicting it — which is the price of using it.

```yaml
train:
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

## Choosing

- Pose available at inference → this (after the LightGBM model {cite:p}`ke2017lightgbm`,
  which is the cheaper first try on the same features).
- No pose at inference → E2E-Spot alone {cite:p}`hong2022e2espot`, no `features:`.
