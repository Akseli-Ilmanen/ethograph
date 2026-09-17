# ADR 0013 — No learned video boundary model while pose carries the effector

**Status:** rejected (2026-09-16). Records why a "spot the boundaries,
segment between them" pipeline was considered and not built, and the
conditions under which it would be reopened.

## Context

The question: when a session has no time-series feature, or the C2F-TCN's
boundaries are blunt, could a small E2E-Spot be trained on the *edges* of
the state labels (class-agnostic onset / offset point events) so that its
per-frame curves become changepoint features for the segmentation model,
and its peaks a mask to snap to?

Everything but the glue exists. `spot` writes dense per-frame curves
through `labels/onset_curves.py`; `segment.ChangepointFeaturesConfig`
expands any mask into proximity / offset / length / shape columns;
`postprocess.changepoint_correction` snaps to a mask; and
`spot.cross_validate()` already writes each held-out session's predictions
from the fold that never saw it, which is exactly the out-of-fold curve a
stacked model needs. Missing: deriving onset/offset point labels into their
own branch, attaching a run's curves as a session variable (the
`merge_video_features` / `attach_feral_embeddings` shape), and raising
`infer.max_events_per_trial` with `confidence: focus`.

What the literature says about the shape of the idea:

- **ASRF** (Ishikawa et al., WACV 2021) is the one-network version: a
  class-agnostic boundary branch (weighted BCE on transition frames, same
  dilated TCN as the classifier, same I3D features at 15 fps), local maxima
  above 0.5 become cuts, and the classifier's frames are majority-voted
  inside each cut. It is evaluated on frame accuracy, edit score and
  F1@{10,25,50}, none of which measures boundary distance. Its gain is
  fewer spurious segments, not sharper edges, and it cannot be sharper
  than the classifier because it reads the same features. The 2026
  "boundary supervision + CDF segment regularisation" losses (arXiv
  2604.01859, no code) sit in the same regime.
- **TRACE** (bioRxiv 2026) does animal behaviour end to end from pixels,
  but with a frozen VideoMAE ViT, adapters, and an ActionFormer-style
  trident head regressing start/end offsets per 16-frame token. Not a
  GSM lineage and not a per-frame boundary curve.
- Nothing found trains a segmentation model on E2E-Spot's or GSM's
  output. The closest is the SoccerNet habit of fine-tuning a spotting
  backbone and feeding a temporal detector.

So the proposal is ASRF's refinement step with the boundary curve coming
from a model that can actually be sharper than the classifier's inputs: a
pixel model at the video's rate with a positive window in the tens of
milliseconds. That is a real difference, and it is why the idea was not
simply a rediscovery.

## Decision

- **Not built.** Where the label is defined on an effector's motion and
  that effector is tracked, pose plus a speed-trough changepoint mask *is*
  the boundary signal, measured directly. A video model can only recover
  it from pixels at more cost and more noise. ADR 0008's table says the
  same for point events: hand-picked distances beat E2E-Spot on the
  contact the effector defines, and lost only on the release, where the
  boundary is not purely kinematic.
- **The decision rule for reopening.** Take the boundary-error histogram of
  snapped predictions against relabelled trials. If it sits near the
  labeller's own jitter, the effector signal is saturated and no video
  route has a job. If it has a fat tail, the tail must share one of three
  causes before any video work starts: the effector is not tracked well
  at the boundary (occlusion, jitter, wrong keypoint); the boundary is not
  kinematic (contact, mouth, texture); or there is no pose at all.
- **If reopened, cheapest first, each step allowed to stop the next:**
  1. Motion energy in the animal crop, or the frame-to-frame distance of
     an already-extracted frozen embedding (`timm`, frame-wise), as a
     changepoint mask with `changepoint_attrs(target_feature=...)`. No
     training; the expansion and the snap work unchanged.
  2. A frozen frame embedding plus lag-differences (the hand-made GSM) and
     a tiny temporal head predicting {background, onset, offset} with
     dilated positives — E2E-Spot minus the backbone fine-tuning. Trains
     in seconds, so out-of-fold curves are free. Ideally a segment run on
     a derived boundary branch, so sessions, splits and `cross_validate()`
     are reused.
  3. One real spot model on class-agnostic onset/offset, with
     `train.split.holdout_sessions` covering the sessions segment tests on
     (the FERAL guard, ADR 0012), a tight crop, ~3 epochs. Never one per
     fold unless the numbers from step 2 demand it.
- **Any refinement keeps the snap budget.** ASRF's cut-and-vote has no
  `max_expansion_s` / `max_shrink_s`; a spurious peak inside a long bout
  would split it. A cut step, if ever added, is bounded the way
  `correct_changepoints` is.
- **Judged by boundary distance**, median and 90th percentile of onset and
  offset error, never by F1@k or edit score, which forgive exactly the
  error this is about.

## Consequences

- No new config section, no spot-to-segment glue, no boundary branch in
  `mapping.txt` conventions.
- A boundary-distance metric in `segment/metrics.py` is the one cheap
  piece worth having regardless, because it is what decides whether this
  ADR is ever revisited.
- Spot's cap of one event per class per trial stands; the multi-event
  path (`max_events_per_trial > 1`) remains for genuine repeated point
  events, not for boundaries.
