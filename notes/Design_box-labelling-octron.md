# Box labelling on OCTRON — design as agreed 2026-09-07

Grilled and settled with the UI prototype `tests/_prototype_box_labelling_ui.html`
(variant E is the chosen one: OCTRON's dock copied 1:1 — 410 px, collapsible pages Manage project / Generate annotation data / Train model / Analyze videos — with only the label manager swapped for the individual × camera approach; A–D are the rejected alternatives, kept as primary source).
First slice implemented 2026-09-07: `gui/dialog_box_labelling.py`, `gui/box_annotate.py`, `gui/box_overlay.py`, `labels/octron_project.py`, `defaults/config/octron.yaml`; tests `tests/test_unit/test_octron_project.py` + `tests/test_integration/test_box_labelling.py`. Later the same day: Remove individual (OCTRON's Remove semantics), one fixed colour per individual written into the organizer, the 'missing by default' rule (no absence bookkeeping — a cell is ✓ mask or missing, Prune off), B/N over suggestions, first individual re-selected on every frame change, `motion_diverse` (motion gate at the median, then k-means), median-filtered motion traces (`features/movement.smooth_motion`, default 3 frames, also on the cover page's pixel change) scored by windowed area (`motion_area`, 0.5 s), the trace cached per camera as `{hash8}/ethograph_motion.npy` and shown as a derived line plot when 'Show pixel motion' is ticked, and sequential frame streaming for suggestions (`VideoFrameSource.iter_frames`: 17 s instead of minutes on 20000 frames). Not yet: suggestion ticks on the time slider, the identity Assign step.

## The one-paragraph version

EthoGraph gets a **Tools ▸ Box labelling…** dialog, a sibling of keypoint labelling,
that drives OCTRON's headless core (the fork at `C:\Users\aksel\Documents\Code\OCTRON-GUI`,
installed editable into the `ethograph` env; upstream issue #87) for everything below the
canvas: SAM2/SAM3 click-to-mask with propagation, OCTRON-native storage, OCTRON's detect-mode
training-data generation, YOLO training and BoxMOT tracking. EthoGraph adds only what OCTRON
lacks: labelling one individual across several cameras at one time index, motion-driven
frame suggestion, a per-camera balance table, training run outside the GUI from a config,
and tracks landing in the session as time series. Segmentation is out of scope: the persisted
label is the box derived from the mask, exactly as OCTRON's detect mode does.

## Decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Deliverable | Suggest → label → train → predict → import, all inside EthoGraph | Tracks in the time-series plots is the value; stopping earlier leaves the OCTRON GUI in the loop |
| 2 | Code reuse | Import the OCTRON fork; **no rewriting** of SAM wrappers, training, tracking | Stay compatible with upstream via issue #87 |
| 3a | Dock layout | OCTRON's toolbox 1:1 (collapsible pages, same group titles, same control names) so OCTRON users are at home; config file named `octron.yaml` | Familiarity; compatibility |
| 3 | Canvas interaction | Copy OCTRON 1:1: left click = positive point, right click = negative, mask overlay redraws, ▶ / 15 frames / Skip; no manual Reset — SAM's memory is per camera and dropped automatically (`TrackState`: Predict runs only from the seeded frame or on from the last predicted one, seeking back first; a jump farther than one chunk from the predicted span, a click outside it, or training resets it) | The workflow is good; only the multi-camera layer differs |
| 4 | Mask rendering | RGBA overlay on the pygfx camera view, same cost as a box | No napari layer stack needed |
| 5 | Cameras | One camera = one video file (the cuts in `octron/videos/`) = one OCTRON hash folder | OCTRON-native; multi-camera is "several videos" |
| 6 | Click flow | Individual-major: pick A, click it in every camera where visible, **X** marks not-visible per camera, Tab → next individual | Matches the described workflow; identity is stated at labelling time |
| 7 | Suggestion | One time index for all cameras; motion (max over cameras) + coverage; ticks on the time slider under the cameras; `gui/pose_suggest.py` methods reused | Runs of consecutive frames are what broke the first model |
| 8 | Propagation | OCTRON's batch predict, run in every camera at once; a verification filmstrip in the dialog | Convenience of one click, but not a training-set strategy |
| 9 | Training set | OCTRON-native: every annotated frame in every hash folder is pooled by Generate training data; EthoGraph shows a balance table (frames / boxes per camera and class) | Keep compatibility; diversity is handled upstream by suggestion |
| 10 | OCTRON project folder | `my_study/octron/` inside the EthoGraph project (hash folders + `model/`) | Pools across sessions; one place both tools agree on |
| 11 | Training | Outside the GUI, as a separate process, from `octron.yaml`: model size, imgsz, epochs, save period, split fractions + seed, batch override + device. No augmentation block. Train tab writes the config, starts the process, streams the log, offers the command to copy | GUI + SAM hold VRAM; a large model at batch 1 is what failed |
| 12 | Predict | Per-camera tracklets (position, box, tracklet id) written into the session as variables | Identity is a later Assign step, seeded by labelled frames + rig geometry (mirror axis) |
| 13 | SAM default | SAM2 L, OCTRON's model dropdown kept (B+, L, L HQ, SAM3, SAM3 multi) | Per OCTRON docs; SAM3 for occlusion cases, SAM3 multi for swarms |

## Open — explore before building

- **Individuals vs OCTRON labels (class) — decided 2026-09-07.** Default: *Tracked individuals
  are of the same type* (a checkbox in the label manager, ticked): one OCTRON label = one YOLO
  class (`class_name`, default `animal`), the individual as OCTRON's suffix — the docs' 'LED 1 /
  LED 2' case. Unticked: one label per individual, for animals the detector can tell apart.
  No naming convention is imposed on the individuals. Still worth confirming with Horst that
  the suffix path is the intended one for identity-by-tracking.
- Identity assignment across tracklets and cameras (the Assign step) is designed only in
  outline: labelled frames as seeds, mirror geometry per rig in `project.yaml`, never in the tool.

## Lessons carried from the first model (2026-09-07, `octron/videos/model`)

- AutoBatch put YOLO26-L at 1024 px on a 10 GB card at **batch 1**; the run never learned
  (best mAP50 0.20 at epoch 49, then collapse). Prefer medium at a real batch; warn when a
  size forces batch 1.
- 265 training frames were mostly consecutive propagated frames; effective variety ≈ 30 frames,
  and val/test shared the same runs. Suggest spread-out frames; keep one held-out stretch.
- Back-mirror birds are ~40 px at 1024; the smallest views may need a larger image size.
