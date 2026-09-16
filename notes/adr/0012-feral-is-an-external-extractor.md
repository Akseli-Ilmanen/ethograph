# ADR 0012 — FERAL is an extractor that runs somewhere else

**Status:** accepted (2026-09-15). Settles the question ADR 0009 left open
("FERAL enters through this registry when the interop discussion settles
the environment question").

## Context

FERAL fine-tunes V-JEPA 2 to a class per frame from pixels alone. On the
crow tool-use data it recognises *what* is happening as well as the C2F-TCN
on hand-crafted features, but places boundaries poorly (F1@90 of 10 with a
2 s chunk); a C2F-TCN reading FERAL's per-frame embedding *plus* the
changepoint features doubles F1@90 over the embedding alone
(github.com/Skovorp/feral/issues/21). So FERAL is wanted as a video
feature, next to S3D and DINOv2.

Two things stop it being an extractor like those. `feral` 1.0.1 pins exact
versions of packages the GUI depends on (`timm`, `pandas`, `transformers`)
and cannot be installed into the ethograph environment. And it is not a
frozen network: it is *trained on the labels*, so its features on a session
it trained on are better than on one it did not, and any model trained
downstream on those features scores that session too well.

Before this ADR the interop lived in `projects/paper/feral/`: an export
script that read a materialised segment run, a driver merging FERAL's
defaults in the FERAL environment, and `video_feature_folders` per session
to get the embeddings back.

## Decision

- **`feral` is a name in the extractor registry that nothing here can
  build** (`EXTERNAL` in `video_features/base.py`). A config selects it
  like `s3d`; its variable is spelled `feral` / `feral_dims` like every
  extractor's; `extractor_module("feral")` raises naming where the
  features come from.
- **`project.video_features()` is the export** (`segment/feral.py`):
  `{root}/feral/labels.json` and a *complete* `config.yaml` for
  `feral train-config`, built from a verbatim copy of FERAL's
  `default_config.yaml` and its presets (`segment/feral_defaults/`,
  MIT, versioned in its NOTICE). FERAL reads its config verbatim, no
  merge, so a sparse overrides file cannot be pointed at it — and FERAL
  cannot be imported here to merge one.
- **The one temporal setting is `video_features.context_s`**, a duration
  resolved against the video's rate into upstream's `chunk_step`, with
  the chunk shifts scaled by the same stride. `preset` is FERAL's `--mode`.
  Nothing else of FERAL's is a project setting; the written `config.yaml`
  is FERAL's file to edit.
- **Labels are rasterised on the video's frame clock through the
  alignment**, trial = video + offset, one video per (trial, individual).
  A second individual is refused; a video with more than half its frames
  outside its trial is refused. No materialised run is needed: the export
  reads the sessions' curated labels directly.
- **`train.split.holdout_sessions` does double duty.** Those sessions are
  FERAL's `test` and `inference` only, never `train`/`val`, so their
  embeddings come from a model that never saw their labels; downstream
  they are the test set as before. `cross_validate()` over `feral`
  columns warns that its folds are inflated. Cross-fitting (one FERAL per
  fold) is not done for the user.
- **Embeddings attach from `{root}/feral/embeddings/`** when a session
  opens under a config whose extractor is `feral`
  (`sessions.attach_feral_embeddings`): this session's files are the ones
  named after its own camera files, attached exactly as
  `video_feature_folders` would, without a folder listed per session. A
  missing folder is an error only once `features.columns` selects `feral`.
- **FERAL is never driven from here.** No subprocess, no environment
  detection: the export logs the two commands and the user runs them in
  the FERAL environment. Its `--save_embeddings` is a pull request against
  FERAL until merged.

## Consequences

- A pure-FERAL user (no segmentation model, no changepoints) gets
  `labels.json` from GUI labels through `eto.segment.export_feral` and a
  minimal config; the split, class names and chunking are not done by hand.
- The paper scripts under `projects/paper/feral/` are superseded by the
  package; `videos.tsv` and `export.yaml` keep what they recorded
  (which trial a video is, FERAL class → label id).
- FERAL's predictions (`feral/answers/*.json`) are not read back as a
  prediction set. That is the next piece if FERAL-only curation in the GUI
  is wanted: `labeling_method=automated`, `prediction_source=feral`, under
  `labels/predictions_feral_{timestamp}/`.
- When FERAL's defaults change, `feral_defaults/` is re-copied and its
  NOTICE bumped; the config header names the version it was built from.
