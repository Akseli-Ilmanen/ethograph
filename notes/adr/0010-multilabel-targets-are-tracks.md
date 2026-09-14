# ADR 0010 — A multi-label target is decoded one track at a time

**Status:** accepted (2026-09-06).

## Context

The segment pipeline predicted one branch of `mapping.txt` for the sample's
own individual: one class per frame, a softmax, DLC2Action's loss with
`exclusive=True` pinned in our adapter. Two things it could not express:

- **Branches coexisting.** The GUI already draws up to three branches as
  stacked lanes of one animal, and a label of branch 0 routinely overlaps a
  label of branch 1 (a posture under a vocalisation). Predicting both meant
  two projects and two prediction sets.
- **The other animals as signal.** A sample carries `other1`, `other2`, …
  feature columns, but the model was never told what those animals were
  doing, though their labels are in the same TSV.

feral calls this `is_multilabel` and infers it from the label array's shape;
DLC2Action calls it `exclusive=False`, builds `(C, T)` targets in its
annotation store and switches the loss to a sigmoid per class with
`BCEWithLogits`. The loss we vendor already has that branch. What was
missing was our layer: how targets are laid out, and how a per-class sigmoid
becomes rows the GUI accepts.

## Decision

`features.labels.branches: [..]` (several branches) or
`features.labels.subjects: all` (the other animals too) makes the target
**multi-label**: a `ChannelTable` of one binary channel per (subject, class)
in place of the `ClassTable`. The loss reads `exclusive` off the table and
refuses a `train.loss.exclusive` that contradicts it.

Decoding is **per track**, a track being one (subject, branch). Inside a
track a channel's sigmoid above `infer.threshold` is on, the most probable
channel wins a frame where several are on, and the track's on/off sequence
runs through the same `postprocess_dense` as an exclusive run. Between
tracks nothing is imposed. So the rule the prediction set obeys is exactly
the GUI's: labels of one animal in one branch never overlap, everything else
may.

The other animals' channels are **training targets only**. Every animal is
its own sample, so at inference each animal's labels come from its own
`self` channels; writing `other1`'s rows too would say the same thing twice
from two samples that may disagree.

Metrics flatten each channel into its own exclusive sample
(`"{key}#{c}"`, class `c + 1`), so `evaluate` is unchanged and
`classwise[c + 1]` is channel `c`'s F1. The circle loss, which pairs frames
by their one label, is refused for a multi-label run.

> **Amended 2026-09-14**: the circle loss has since been removed from the
> codebase (it earned nothing in `scripts/bench.py`'s ablation), so the last
> sentence no longer describes any code. `train.circle` is a retired config
> key — see `RETIRED_KEYS` in `ethograph/segment/config.py`.

## Consequences

- One config key decides the target; the exclusive path is byte-for-byte
  what it was (`branch: 0`, `subjects: self`).
- `classes.yaml` carries `target: multilabel` and the channel list; every
  reader goes through `read_target_table`, never `ClassTable.from_dict`.
- `groundTruth/{key}.npy` `(C, T)` replaces the literature's `.txt` for a
  multi-label dataset — a third-party model reading the text layout gets
  the exclusive dataset only, which is the only one it could use anyway.
- The GUI needs nothing new to open a multi-label prediction set: it is
  rows of several branches for one animal, which the branch lanes already
  draw. Reviewing it is where the branch cap (three lanes) and a
  branch-aware review queue become the next questions.
