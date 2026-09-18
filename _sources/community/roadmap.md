# Roadmap

What Ethograph already does, what is in progress, and what is still open. Grouped by theme; within each theme items run roughly in the order they were (or will be) tackled.

**Legend:** ✅ done · 🚧 in progress · ⬜ not started

## Milestones

| When | Milestone |
|---|---|
| April 2026 | Movement community call demo ([slides](https://neuroinformatics.zulipchat.com/user_uploads/58792/jV4lyzfLheHU4Gj4qCQkKzwy/2026_04-Ethograph-demo.pptx)) |
| 2026 | Segmentation pipeline, lightgbm models, pixel event spotting, curation workflows |
| Next | Shared feature schema with `movement`, NWB video alignment upstream |

---

## 1. Interop with segmentation models

- [x] Import predictions from action segmentation models (DLC2Action, ASFormer, MS-TCN): **File → Import predictions…** turns per-trial `(T, n_classes)` or `(T,)` arrays into labels with a confidence overlay (1 − normalised entropy)
- [x] Scripted segmentation pipeline (`ethograph.segment`): materialise → search → cross-validate, predictions written as GUI label files
- [x] lightgbm models for point events, with confidence read off the curve
- [x] Pixel event spotting from video (`ethograph.spot`)
- [x] Curation of model output: the Curation section, grids and saved workflows
- [ ] 🚧 A shared schema for segmentation feature data with `movement` ([movement#978](https://github.com/neuroinformatics-unit/movement/issues/978))

## 2. Aligning video and data streams via `.nwb`

- [x] Read and edit alignment directly in `.nwb` sources; `.ethograph/alignment.nwb` sidecars for everything else
- [ ] 🚧 Video alignment in NWB tooling upstream ([nwb-video-widgets#34](https://github.com/catalystneuro/nwb-video-widgets/issues/34), [nwb-schema#677](https://github.com/NeurodataWithoutBorders/nwb-schema/issues/677))

## 3. Changepoints

- [x] Fast changepoint detection (gradient-, RMS-based, …) and changepoint correction of label boundaries
- [x] Changepoint features ({func}`~ethograph.features.changepoints.more_changepoint_features`), which massively improved fine-grained accuracy for [ASFormer](https://github.com/ChinaYi/ASFormer)
- [x] Changepoint features available to every segmentation model (`features.changepoint_features` in the segment pipeline)
- [ ] Changepoint features for pynapple / NWB sessions (currently `.nc` only), e.g. to segment extracellular recordings (sharp-wave ripple detection) or other neural time series
- [ ] ML-based changepoint detection. It must stay reproducible so it is a reliable feature. Note: post-model changepoint correction sometimes makes things worse, since the transformer learns a better representation than simple gradient-based methods.
- [ ] 🚧 Audio changepoints (`ethograph.features.audio_changepoints`)

## 4. Neural data

- [ ] 🚧 Interactive PSTH (`ethograph.gui.widgets_psth`)
- [ ] Single-trial neural dimensionality reduction; visualise label segments in latent space
