# ADR 0009 — Spike trains enter the segment pipeline as a config transform, per session, in memory

**Status:** accepted (2026-09-02).

## Context

The segmentation pipeline learns state events from features that already
exist in the session file: "features are built with the session, never by
the pipeline" (ADR 0008 for the pose side). Neural decoding wants the same
models, split, metrics and prediction sets, but its input is a pynapple
`TsGroup` — spike *times* — which no loader can read as a feature, and how
the spikes become a dense signal (bin size, smoothing, rate versus count,
a square root) is exactly the kind of choice one wants to sweep.

Two further facts shape the design. Units are only consistent within one
recording, so a neural project is one session and leave-one-session-out
cross-validation cannot apply. And the same recording may hold an epoch
with no trials at all (sleep), over which the wake-trained decoder should
run.

## Decision

1. **The binning is spelled in the config, as pynapple expressions, and
   runs at every session open.** `features.neural` names the `TsGroup`, the
   feature it becomes, and a `transform` list evaluated in order on `x`
   with `nap`, `np` and `sliding_window` in scope
   (`ethograph/features/neural.py`). The result is a `TsdFrame` added to
   the session's pynapple objects and declared `kind: neural_feature`. It
   is never written to disk: a different binning is a different
   `features.name`, one materialised dataset per idea, so the transform is
   swept like any other run-level knob without ever saving raw spikes as
   a feature.
2. **The unit columns are read off the session, not written in the YAML.**
   `features.columns` never spells them; materialise resolves them from
   the opened session, records them in `columns.yaml` (`neural_columns`),
   train reads them back into the run config, and inference inherits them
   from the run. Two sessions with different units are refused by name, as
   is inference on a recording missing units the run was trained on.
3. **Cross-validation folds by trial when there is one session.**
   `train.split.holdout_trials` is the trial-level twin of
   `holdout_sessions`; `cross_validate(n_folds=k)` deals every trial into
   exactly one fold, predicts each fold's own trials, and merges the folds
   into one prediction set per session so the whole session still opens in
   the GUI predicted once by a model that never saw it.
4. **A trial-less epoch is a second alignment on the same file.**
   `sessions[].alignment` reads the trials from another NWB;
   `segment/windows.py` tiles an epoch into contiguous windows and writes
   them as that alignment's trials, with a `state` column for
   `trials.where`. Windows never overlap — pynapple merges overlaps and the
   alignment loader clips them — so overlap is a second, phase-shifted
   tiling.

## Consequences

- One session per neural project; the sessions layer, the split and the
  models are untouched, and the transform sweep costs a materialisation per
  variant and nothing else.
- The transform is evaluated Python, trusted like the script that runs the
  pipeline. An error names the step.
- Long windows are only sound for an architecture whose receptive field
  the training trials filled. MS-TCN's default receptive field is minutes
  at 200 Hz while a wake trial is seconds, so its layers must be cut to the
  trial (`num_layers_PG` / `num_layers_R` of 10) before a one-minute window
  means anything; C2F-TCN pools over its whole input and has no version of
  that fix.
- Time compression of replay is handled by stretching the sleep clock (spike
  times and window bounds by a factor k) rather than by any pipeline
  setting; the wake-trained transform then sees replay at wake speed.

## Alternatives considered

- **Save the binned rates as a feature `.npz` beside the session.** Rejected:
  it fixes one binning into the data and hides the sweep.
- **`"*"` for "all units" in `features.columns`.** Rejected: "explicit
  values, never all" is what keeps layouts identical across sessions; the
  resolved list recorded at materialise is the explicit spelling, derived
  once.
- **Overlapping windows in one alignment.** Impossible with pynapple's
  `IntervalSet`, and the loader would clip them silently; phased tilings or
  a future `infer.windows` that bypasses the trials table are the routes.
