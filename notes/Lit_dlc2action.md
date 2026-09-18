# DLC2Action vs. the Ethograph segmentation pipeline

A comparison of `ethograph/model` (+ `ethograph/features`, `ethograph/video_features`,
`scripts/model`) against [DLC2Action](https://github.com/amathislab/DLC2Action), written
to decide what, if anything, is worth adopting. Date: 2026-08-22.

## Framing

The two are not the same kind of thing. DLC2Action is a general multi-animal toolbox:
7 input formats, 8 model architectures, 11 self-supervised tasks, Optuna search, active
learning, ~350 YAML config keys across 29 files, and a 6,864-line `project.py`. The
Ethograph pipeline is one bespoke path — `.nc` → features → a single sliding-window
ASFormer encoder → per-trial predictions the GUI loads — tuned for **fine-grained
boundaries** on trial-structured data. Most of DLC2Action's breadth is not a gap for
Ethograph; a handful of specific pieces are.

## Side by side

| Area | Ethograph | DLC2Action | Verdict |
|---|---|---|---|
| **Video features** | S3D (Kinetics-400), a 0.1 s stack (21 frames at 200 fps) centred on every frame, 1024-d, motion-aware (`video_features/`) | DINOv3 ConvNeXt-tiny CLS token per single frame, whole image, unbatched (`preprocessing/visual_encoders.py`) — the only encoder | **Ethograph ahead.** Theirs carries no motion. Both are whole-frame (no crop around the animal). (The stack loop used to run one window per forward pass; it is now batched and streamed — see `video_features/extract.py`.) |
| **Kinematic features** | Smoothed `position` / `velocity` / `speed` / `acceleration` of the keypoints picked by `feat_kwargs` | Egocentric coords (minus body centre), all pairwise intra-distances, inter-individual distances, speed *direction* unit vectors, angles, polygon areas, zone bools, likelihood (`feature_extraction/__init__.py`) | **Behind.** Theirs is a much richer, mostly cheap, relative-geometry set. |
| **Changepoint features** | Multi-σ Laplacian proximity + segment IDs + speed-weighted variants (`features/changepoints.py: more_changepoint_features`) | Nothing comparable | **Ahead** — the most original piece, and the one the roadmap credits for boundary accuracy. |
| **Preprocessing / normalisation** | Per-trial NaN interpolation → percentile clip → **per-trial** z-score (`model/dataset.py`) | Likelihood threshold → interpolate → fill from visible-keypoint mean; **train-set-only** mean/std, scale-free keys skipped (`data/dataset.py: get_normalization_stats`) | **Behind.** Per-trial stats give the same feature value different meanings in different trials and are recomputed on test data. (Their missing-keypoint-as-0 sentinel is a real wart, though.) |
| **Augmentation** | None (time-warp stubbed out); ASFormer channel dropout 0.3 | mirror / shift / noise / rotate / zoom / joint-mask / `switch` individuals, missing-mask preserved (`transformer/kinematic.py`) | **Behind**, but most of theirs do not apply: fixed camera, and S3D channels cannot be geometrically co-transformed. |
| **Model** | One ASFormer encoder (decoders removed), sliding-window attention, whole-trial input at batch 1 (`model/cetnet_encoder.py`) | MS-TCN++, ASFormer enc+dec, C2F-TCN, C2F-Transformer, EDTCN, MotionBERT, MLP; a feature-extractor / predictor split so SSL heads attach to an intermediate tensor (`model/base_model.py`) | Parity. One good model beats eight. Whole-trial sequences also sidestep their segment cut / stitch / overlap-leakage machinery. An MLP no-temporal baseline is the one worth having. |
| **Loss** | CE + scheduled boundary weighting + smoothing MSE @ 0.15 + circle loss on features | Weighted CE / BCE, inverse-frequency class weights, focal γ=2, smoothing @ 0.001 (effectively off), hard negatives, `-100` unknown (`loss/ms_tcn.py`) | Mixed. Ethograph's boundary terms fit its goal; it has **no class weighting or focal term** at all. |
| **Training loop** | Adam + weight decay, ReduceLROnPlateau, gradient clipping | Adam, constant LR, no AMP, no clipping, no scheduler (`task/universal_task.py`) | **Ahead.** |
| **Checkpoint selection** | Save every `log_freq`, hand-pick `epoch-100` | Periodic save, crude autostop, no best-weights restore | Both weak. |
| **Labels** | Exclusive per-frame, one individual; actor/recipient pairs exist in the TSV but `intervals_to_dense(..., [individual])` collapses to one | Exclusive *or* multi-label, unknown / hard-negative / visibility-filtered bouts, `filter_background` (only frames near annotations become background) | Behind for multi-animal; irrelevant until both individuals are modelled at once. |
| **Metrics** | Acc, edit score, F1@k, frame-F1, class-wise, IoU + start/end delta histograms (`model/eval_metrics.py`, `eval_plotting.py`) | Same set + mAP + semi-segmental F1 (length-dependent IoU threshold) + threshold sweeps (`metric/metrics.py`) | Parity; the boundary-delta histograms are the thing theirs lacks. Not worth porting. |
| **Post-processing** | Changepoint snapping, purge / stitch (`labels/ml.py`) | Min-interval smoothing (and `Task._smooth` is dead code) | **Ahead.** |
| **Config / reproducibility** | ~25-key JSON + argparse + per-person Python scripts with hardcoded paths (`scripts/model/model_config_*.py`) | 29 YAML files, ~350 keys, `???` blanks, string sentinels resolved at runtime | Ethograph's is simpler but not portable; theirs is the thing not wanted. |
| **Uncertainty / active learning** | Entropy confidence → grid-view threshold | Least-confidence, entropy, MC-dropout BALD → suggested intervals (`Task.generate_bald_score`) | Slightly behind; BALD is ~30 lines if the grid view ever needs better scores. |
| **SSL** | None | 11 tasks (masked features / kinematics / frames, contrastive, pairwise, segment order, reverse, TCC), run **multi-task** (not pretraining), plumbing spread across dataset, transformer, model dispatch, task and dispatcher | Skip. |

## What to take — reimplement the idea, do not copy code

DLC2Action is **AGPLv3**; copying code verbatim would pull Ethograph under AGPL. Each item
below is small enough to write fresh, ranked by value ÷ effort:

1. **Relative kinematic features** (largest expected gain, ~100 lines in
   `features/movement.py`, computed when the `.nc` is built): the vector between the two
   keypoints of interest in an egocentric frame, pairwise keypoint distances, speed
   direction as a unit vector separate from speed magnitude, the angle between keypoints.
   No config — more `type="features"` variables.
2. **Train-set normalisation stats** (~30 lines in `model/dataset.py`): compute mean/std
   once over the training trials, save next to the model, apply at eval / inference. Skip
   the binary and segment-ID changepoint columns.
3. **Class weights + optional focal term** (~10 lines in `Trainer`): inverse-frequency
   weights from the training labels into `nn.CrossEntropyLoss(weight=…)`; `gamma` as one
   number. Keep the boundary schedule on top.
4. **Best-checkpoint-on-validation + patience** (~15 lines): track F1@50 (or frame-F1) on
   the test bundle already evaluated every `log_freq`, keep `best.model`, stop after N
   non-improving evaluations. Replaces the hand-picked `epoch-100`.
5. **Two augmentations in `BatchGenerator.next_batch`**: Gaussian noise on the kinematic
   channels, random temporal stretch 0.8–1.2× (nearest-neighbour labels) — the warp that
   is stubbed out. Nothing geometric.
6. **MLP baseline** (a `num_layers=0` switch): confirms temporal context is doing the work.

## What not to take

- **SSL** — multi-task through five modules; the one cheap piece (`ReverseSSL`, ~40 lines)
  has unproven benefit on data of this kind.
- `project.py`, the YAML / blank config DSL, Optuna, the heatmap pipeline, the model zoo,
  DINOv3 (S3D is strictly more informative for motion; per-frame appearance could
  complement later).

## Improvements that come from neither

- **S3D cost**: batch the sliding windows and crop around the animal using Ethograph's own
  keypoints — both bigger wins than anything in DLC2Action's visual path.
- **Pipeline debt worth a cleanup pass first**: `from cetnet_encoder import *`;
  module-level `device = cuda` bypasses `resolve_device()`; `"fps": 200` hardcoded in the
  config; `"corrected"` and `"uncorrected"` evaluate the **same** dict
  (`cetnet_encoder.py`, `# SAME SAME`); the `no_circle_loss` flag is inverted (`True`
  enables circle loss); a keypoint name hardcoded in `get_trial_dict`'s inference filter;
  the MS-TCN legacy disk format (`groundTruth/*.txt` + `.bundle` + hash keys) could be
  replaced by a torch `Dataset` reading `.nc` + TSV directly.

## Recap

Ethograph is ahead on video features, changepoint features, boundary-focused losses, the
optimiser loop and post-processing; behind on relative kinematic feature engineering,
normalisation discipline, class weighting, augmentation and checkpoint selection. Items
1–4 above are each under 100 lines, add no new config surface, and are where to start.
