# 7. Distillation is a stage flag in the vendored E2E-Spot trainer; the teacher stays UMEG-Net's

Date: 2026-08-26

## Status

Accepted

## Context

The pixel spotter (`ethograph.spot`) collapses under data starvation: two
events per ~1200 frames, and every run comes apart after a few passes over the
labelled trials. UMEG-Net's answer is distillation — a keypoint teacher whose
per-frame representation a video student is trained to reproduce on every
clip that has pose, labels or not, before its head learns the labels.

Two places that student could be trained: our own PyTorch loop importing the
vendored `E2EModel` in-process, or the vendored `train_e2e.py` itself, given a
stage flag. And two shapes the teacher could take: UMEG-Net's, or the
graph-critique variant (box-frame coordinates, explicit distances, an MLP
instead of a GCN, a validity channel) that a literature pass argued for.

## Decision

**Distillation lives in the vendored trainer.** `train_e2e.py` gains
`--stage {1,2,3}`, `--teacher_dir`, `--distil_dim`, `--init_from`; the
dataset gains one branch that loads a clip's teacher embedding; the model
gains one projection layer and a `return_embedding` path. `Project.distil()`
spells the two commands and nothing else. Every stage is an ordinary run that
`test_e2e.py`, the epoch choice, `inference()` and the ladder comparison read
unchanged.

**The teacher is UMEG-Net's, with one addition.** Static keypoints are nodes
(as the paper's court corners are), edges are explicit or `all`, the trunk is
the paper's GCN + multi-scale shift, the input is `(x, y)` with missing
coordinates written as 0. The one addition is `teacher.extra_features`:
session variables in the segmentation pipeline's column spelling, concatenated
before the GRU. The four departures the critique argued for exist as config
switches, **off by default**.

## Consequences

- One training loop, one clip sampler, one set of augmentations, one place a
  Windows `spawn` guard has to be right. The cost is that our changes to the
  clone grow (`scripts/spot_windows_compat.patch` is the record) and a
  student is only as flexible as upstream's CLI.
- A GSM run and a distilled run differ in exactly the flags on the command
  line, so a comparison is a comparison of the recipe, not of two codebases.
- The teacher's departures from the paper are an experiment, not a design:
  they are one ladder away from being the default or being deleted, and
  nothing else depends on which. If the teacher is later replaced (a fuller
  skeleton, a different architecture), the student, the embeddings file and
  the two stages do not change.
- `teacher.extra_features` couples the teacher to the session's variables the
  way the lightgbm model already is. That is deliberate: a user who knows a
  distance matters can say so without touching the model.
