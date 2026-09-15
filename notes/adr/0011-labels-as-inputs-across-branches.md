# ADR 0011 — Existing labels are inputs, from one branch to another only

**Status:** accepted (2026-09-15).

## Context

A project rarely asks one question of a dataset. Labels from an earlier
question — every bout of a behaviour, every contact — are expensive and carry
information about *when* the answer to the next question can happen: a peck
rarely comes before the head has turned, a landing never before the
approach. Until now none of it reached a model. Both pipelines only read
labels as targets, and the one earlier attempt at this (a `label_inputs`
module feeding the GUI's LightGBM model) was deleted with the graph code it
sat beside, unused by either pipeline.

The obvious hazard is leakage. If the labels a model reads as input include
the labels it predicts, it learns to copy them; cross-validation is blind to
it, because the same labels are on both sides of every fold; and the model
learns nothing about the other inputs — the exact opposite of what was
wanted.

## Decision

- **Labels become inputs through one module**, `features/label_inputs.py`,
  rendered per trial onto the clock of a named feature at session-open time
  and merged into the column list like the changepoint expansion. A state is
  its on/off indicator; a point event is a Laplacian bump per configured
  width, `max` over events so the column stays in `[0, 1]`. The event type is
  the mapping's, never the row's, so the layout cannot change with what a
  trial contains.
- **An input branch is never a target branch.** The `mapping.txt` branch is
  the unit of the rule: `segment` refuses `label_inputs.branches` overlapping
  `features.labels.branch(es)`; `spot` refuses a branch holding any of
  `labels.classes`. Nothing finer (per class, per trial) is offered — the
  branch is what the GUI already keeps exclusive, so it is the boundary a
  user can see.
- **Absence renders zeros.** A trial (or a whole session predicted later)
  without these labels reads as "none here", logged, never refused. Only
  `manual`/`curated` rows count unless `include_automated` says otherwise —
  the same evidence rule as for targets.
- **Declared `kind: label_input`, `normalise: 0`.** The ablation is
  `train.drop_kinds: [label_input]` in `segment`, `zero_features=True` in
  `spot`; measured, never assumed.

## Consequences

- A second question on a labelled dataset can exploit the first without
  re-materialising anything by hand: two branches in the mapping, one line
  in the config.
- Predicting a new session then needs the old branch labelled on it too, or
  the model runs on zeros where it trained on evidence. The docs say so; the
  pipeline does not guard against it beyond the log line.
- Xarray sessions only, like the changepoint expansion; a pynapple session
  with the section set is refused by name.
