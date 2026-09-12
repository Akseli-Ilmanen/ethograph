# EthoGraph × FERAL — labelling, intermediate outputs, curation


## 0. The one-paragraph version

EthoGraph is a labelling and curation environment over multimodal time series; FERAL is a
video-to-ethogram model with no labelling story at all. Three things came out of the discussion,
in increasing order of how much they change the product:

1. **A BORIS-style labelling mode is cheap** and turns EthoGraph into a tool a user with nothing
   but video can start with. The mode itself is a flag; the real work is the supporting paths.
2. **Every per-frame quantity FERAL computes is a time series, and almost none of it is visible
   to its users.** Exposing embeddings, probabilities, overlap disagreement and attention costs
   EthoGraph nearly no new GUI code, because the panels already exist.
3. **Curation is the moat.** A rough human first pass and a model's predictions are the same
   object, so the review machinery already built for models also serves cold-start labelling.

---

## 1. Labelling from scratch — the BORIS-style mode

### The claim

Claude proposed *pick a class → click start → play → click end*.
Better (do this instead): Use shortcut to set a class, e.g. letters 4 or w as before. but here the clciking of "4" already place the point event at the current frame. If 4 or w belogns to a state event, then the user can click 4 will show start point (label onset) at the current frame and then they can naivgate the gui with playback (click space) or timeline or manual frame-by-frame navigation with <> buttons. Second time they click 4 they confirmtm the label offset.

in Labels tab at the top there should be a combo box where the user can controlw aht labelling mode they are in Label on time series (Ethograph-style) or Label at current frame (Classic-Style) - maybe reprhase this



The two-click state placement already exists (`show_pending_label` → second click →
`add_interval`, with `_reset_label_clicks` on every abandon path). The only difference is where
the time comes from:

- today: the mouse x-position on a plot panel
- BORIS mode: `video.frame_to_time(current_frame)` → `from_display`

That second line is verbatim what frame-mode review already commits with. Everything downstream —
undo snapshot, overlap resolution, subject pair, the TSV — is untouched.

Day one it is already better than BORIS, because the labels inherit undo, the label table, the
grids, export, and the model loop.

### The two parts that are harder than the mode

**A bare timeline panel.** The label overlay is a mixin on plot panels; `label_overlay_modes` maps
plot *type* → render mode and `schedule_labels_redraw` runs over panels. A user with only a video
has no panels, so no visible labels, so no way to see or re-click what they just placed. The real
deliverable is a `BasePlot` with an empty y-axis — a label ribbon over the trial — shown by default
when a dataset has no features. Useful beyond this case as a label-only overview.
- this empty plot can be the first option always visible in add panels button, and maybe we next to the Label current frame checkbox have a checkbox, create empty plot checkbox ticked, which loads this empty plot (if no panel currently exists).

**Opening a folder of videos.** Loading assumes NetCDF/NWB/pynapple plus `.ethograph/alignment.nwb`
for timing. A user with `videos/*.mp4` has no dataset, no trials, no clock. Something must
synthesise a session: one trial per video, fps from `io/video_probe.py`, a minimal alignment NWB.
`classify_files()` is the entry point, and the cover page's consented pynapple→alignment conversion
is the precedent. **This is where the surprises will be, not in the keys.**


---

## 3. Manufacturing a time series when the user has none

The recurring wall: a video-only user loses the superpower above. Two ways to give them one.

### 3a. ROI motion energy — cheapest, do this first

Draw boxes on a frame; compute mean absolute frame difference (greyscale, light blur, low
resolution) inside each; get `(time, roi)` traces. No model, no GPU, no labels, no dependency on
anyone's roadmap. Well-precedented (Facemap ROIs, motion-energy regressors; BEAST uses it for frame
selection).

Where it lands — almost nowhere new:

- **Clock**: same pattern as `{stem}_s3d.nc` — on the video clock, frame 0 at t=0, merged via
  `stream_offset_for_trial`. The clock question is already answered.
- **Panel**: a lineplot. Tag `kind="changepoint_feature"` and it flows into
  `features/changepoints.py`, so **click-to-snap works on it** — pass 2 becomes available to
  someone with only video.
- **Model**: `features/columns.py` picks it up, so the LightGBM onset model and the segment
  pipeline can train on it. A video-only user can train a real model with no pose and no CUDA.

Design decisions:

- **Crop is currently display state; ROIs need to be data.** Today: one crop per camera,
  session-scoped, display-only. This wants N *named*, persisted regions per camera — the trace's
  identity is the region's name and must survive a reload and mean the same thing in a config.
  This is the one real modelling change; do it deliberately rather than overloading the crop.
- Compute at ~64 px. The cost is decode, not arithmetic — so compute every ROI in one pass.
- **Free bonus: mean intensity, not just difference.** Same pass, same machinery → **sync-LED
  detection**, a square wave to align other streams to. Arguably as valuable as the motion trace
  for someone assembling a session from raw video.
- Cache as a sidecar, not a console-derived feature (those live for one trial by design).

Honest caveat: motion energy is confounded by camera shake, lighting, an experimenter's hand, a
second animal. It is a *navigation and search* signal that happens to be a decent model input, not
a measurement of the animal. Say so in the docs — the failure mode is treating a peak as evidence
rather than as a place to look.

### 3b. Frozen-backbone embedding traces — the same idea one level up

ImageNet ViT-MAE CLS (CPU-tolerant, no CUDA, no 24 GB) → per-frame embeddings → three traces:

- **Δ-embedding** (adjacent-frame cosine distance): a behavioural novelty curve; as a changepoint
  feature it gets click-to-snap.
- **PCA to 3, as lineplot traces**: a bout is a plateau, a transition is a ramp. Far more legible
  than a 768-row heatmap for finding a boundary by eye.
- **Similarity-to-exemplar**: user labels one clean instance, plot cosine similarity of every frame
  to it; peaks are candidates. Turns the first label into a search over the session. For cold start
  this is the difference between labelling 200 events and labelling 20. Needs a small piece of UI
  ("use this label as exemplar").

Build these on ViT-MAE **first**. They work on every machine, and if they help then a FERAL or
BEAST backbone is later a name in a registry. Building 3a first also tells you whether the *shape*
of the feature — a change trace you snap to — is useful before paying for a backbone.

### On SAM

Asked and answered: **no, for now.** Apply the earlier note's own rule 5 — does a mask beat a
rectangle for motion energy? For a static ROI the gain is below the noise floor of the confounds.
For a moving animal you do not want an ROI at all — you want pose, or a **pose-anchored box**
("80×80 px centred on the head keypoint"), which is free, already available, and *better* than a
mask because it is identity-aware.

SAM2 video propagation is a tracker: GPU, mask drift, re-prompting, and per-frame cost that dwarfs
the feature it feeds. It also needs a third mask-editing surface alongside keypoint labelling and
skeleton editing — that is where it gets out of hand.

Sharper scope test than the earlier one: **does the fancier version change the answer, or only the
picture?** SAM changes the picture. Pose-anchoring changes the answer. Motion energy at all changes
the answer enormously for someone who previously had no trace.

Only shape worth entertaining later: *single-frame* MobileSAM/EfficientSAM as a "help me draw a
better static region" affordance — no propagation, no tracking, a drawing convenience.

---

## 4. FERAL intermediate outputs in EthoGraph — the cool part

### What the code actually does

Read from `feral/model.py` + `backbones.py` + `default_config.yaml`:

```
backbone (B,T,C,H,W) → (B, N, D)            patch tokens
clip_projector        → (B*Q, D)             Q = predict_per_item
fc_norm → head        → (B*Q, num_classes)
```

Three corrections to earlier assumptions:

- **The 64 queries *are* the 64 frames.** `predict_per_item: 64` = `chunk_length: 64`, one learned
  query per output frame. There is no extra "query" axis. `clip_projector` output *is* the
  per-frame time series: `(64, 1024)` per chunk.
- Hidden dim is **1024** (ViT-L `vjepa2_vitl_diving48` default), not 768.
- **Pre-pooling has 32 temporal positions, not 64** — V-JEPA2 uses 2-frame tubelets, so for 64
  frames at 256 px, patch 16: N ≈ 32 temporal × 16×16 spatial = 8192 tokens. Attention pooling is a
  32 → 64 temporal *upsample*. **The ±1-frame ambiguity is structural, and this is where it lives.**

### Before or after pooling?

**For time series, after is strictly better** — already frame-rate, one vector per frame, no
reshape. Pre-pooling is half-rate and mixes 8192 spatial tokens per slot; as a curve it is a
blurrier version of the same signal.

**Pre-pooling's unique content is spatial, not temporal** — a video overlay, not a trace.

### The mapping — how little is new

| FERAL output | Shape | EthoGraph home | New code? |
|---|---|---|---|
| per-frame class probabilities | (T, K) | onset-curve overlay / lineplot | none |
| overlap disagreement | (T,) | lineplot, or review-queue order | none |
| per-frame embedding (`clip_projector`) | (T, 1024) | heatmap + derived lineplots (§3b) | none |
| attention weights | (T, 32, 16, 16) | video overlay | **new** |

**The integration is a file format, not a UI.** FERAL does not need a viewer, it needs to write
time series.

### The two things to ask them for

**1. `need_weights=True`.** `clip_projector` currently does
`self.attn(x_q, x_kv, x_kv, need_weights=False)` — it **discards** attention weights that are
`(64 frames, 8192 tokens)`. Reshaped to `(64, 32, 16, 16)` these give:

- a **spatial attention heatmap per frame**, overlaid on the video: *where the model looked when it
  called this frame behaviour X*. If it is on the wrong animal, a shadow, or the experimenter's
  hand, you know the prediction is worthless without reading a single number. No confidence
  statistic communicates that. **This is the demo.**
- a **temporal attention profile** per frame (marginalise over space): how much frame *t* drew on
  each of the 32 slots. Concentrated = local evidence; spread = leaning on distant context.
  Plausibly a boundary indicator, and it is a `(time, 32)` heatmap — free.

One-line change; a good PR to offer.

**2. The unaveraged per-chunk predictions.** `chunk_shift: 32` = 50 % overlap in training, and
`eval_chunk_shift` goes denser (the `max` preset evaluates at 80 %). Every frame is predicted 2–5×
from different windows and `save_inference_results` **ensembles them, throwing the spread away**.
That spread is a free uncertainty estimate **independent of the softmax** — which is famously
miscalibrated. Cheap for them; they have probably never looked at it.

Also note `eval_smoothing_window` (moving average over per-frame probabilities) will bias onset
estimates — **turn it off for point events**.

### Design notes on the display

- **The attention overlay is a diagnostic, not a routine display.** Off by default, toggled when a
  prediction looks wrong. Only meaningful after fine-tuning. Lives in video-panel territory
  alongside pose rendering, not in a plot panel.
- **Embeddings: do not heatmap all 1024 rows.** Sort/select — top-30 dims by Cohen's d against the
  labels that exist, or by variance when none do.
- **Uncertainty: the best use of a new uncertainty signal is usually not to draw it.** The grids
  already sort by confidence and flag below a threshold; overlap disagreement is simply a better
  number feeding machinery that exists. The visualization is the queue order. Resist putting it on
  the label's colour — that channel already means `labeling_method`, and pose rendering holds the
  same rule (colour encodes one axis).
- **States vs points.** FERAL is state-oriented; onset curves were built for point events. For a
  state class the curve is a plateau, so `tallest_peak` and focus/ratio are the wrong readings. A
  state-flavoured confidence (mean probability over the segment, or margin at the two edges)
  belongs in `labels/rescore.py`'s `RULES`, where the histogram popup already lets a user compare
  statistics by looking. Small, and the same shape as work done twice already.

---

## 5. Better curation — where the moat is

Curation asks *"is this boundary right?"*; labelling asks *"is anything happening, and where does
it start?"* Almost every ergonomic decision differs — but the **object is the same**, which is why
the curation stack generalises.

What no other tool in this space has: `labeling_method` (manual / automated / curated), per-label
confidence, the review queue, the two grids, frame-by-frame review, confidence rules with a
histogram you threshold *by looking*, and recorded curation workflows. FERAL's paper implicitly
assumes labels arrived from somewhere already correct.

Concrete improvements that came out of the discussion:

- **A state-aware confidence rule** (above) — needed the moment any state-oriented model's output
  is curated.
- **Overlap disagreement as a queue key** — a better uncertainty signal than softmax, free from
  FERAL, no new surface.
- **Provisional/rough labels in the queue** — the `HUMAN_CONFIDENCE` tension in §2. Resolving it is
  what makes the sweep→snap→refine loop work without a new subsystem.
- **Similarity-to-exemplar as a review aid**, not only a labelling aid: after curating one
  instance, the trace shows every frame that looks like it.

---

## 6. Multi-animal — the strongest idea in the thread

### Framing

CalMS21, FERAL's headline benchmark, is resident-intruder with a **black mouse and a white mouse**.
The identity problem is solved for the model by the animals being visually distinct, and nobody
says so. The proposal — track individuals and colour them — is *"synthesize the CalMS21 condition
for any dataset"*. Worth saying out loud to them.

### The sharper version: focal encoding, not N colours

FERAL's head is `nn.Linear(d, num_classes)` → `(B*Q, num_classes)`. **There is no individual axis.**
One label track per video. So colouring animals red and blue still forces pair-classes
("A grooms B", "B grooms A"), which explodes as N² and shares no statistical strength.

The version that works:

> **Highlight one focal animal, neutralise the others, run the model once per individual.**

Output is per-individual by construction, `num_classes` stays flat, and every individual's data
trains the same head. This is **exactly the existing invariant**: in the segment pipeline a sample
is one (trial, individual), pinned, with others as `other: "*"`. The video version is the same rule
in pixels — a strong signal it is the right factorisation. Cost is N× decode and N× inference; fine
for 2–3 animals, not for 10.

### The risk

Colouring **bakes tracker errors into pixels, irreversibly**. An ID swap becomes a wrong colour
becomes a confident wrong attribution that looks exactly like a correct prediction about the other
animal.

The GUI has an answer a pipeline does not: **keep tracker output as a first-class stream, not just
a rendering input.** Per-individual centroid traces, plotted — a swap is two lines crossing and
exchanging, far more obvious than in video. The render is derived from a stream the user can see
and correct, and regenerated. This is a genuine argument for doing it inside EthoGraph.

### Rendering

Do **not** tint the animal — a saturated overlay destroys the texture the model needs, and
V-JEPA2's pretraining distribution contains no magenta mice. Lighter options, in order: a soft
coloured halo/disc outside the body; a low-alpha hue shift preserving luminance; the skeleton drawn
in the focal colour; desaturating the non-focal animals instead. Which works is empirical — one
model, four input encodings, one test split. Paper-shaped for them, so an easy collaboration ask.

### Do we need a new tracker? Mostly no

- **With pose** (DLC/SLEAP import, PosePAL): a centroid is the mean of keypoints. Free.
- **With AprilTags**: `pose_detect.py` already does tag36h11 with assignment learning —
  unambiguous, drift-free identity, with the tag-sheet printing tools to make it practical. For a
  user willing to tag animals, **identity is already solved in the repo today.**
- **Without either**: background subtraction → blob → Hungarian fails exactly when animals touch,
  which is when the interesting behaviours happen. Do not build it.

So the tracking half is mostly *exposing what exists* as a centroid stream.

### The time-series half — independently valuable, cheaper

Per-individual centroids give, through `features/geometry.py` which already exists: inter-animal
distance, approach/retreat velocity, relative heading, contact — the traces on which social
behaviours are actually *defined*.

This directly addresses the **consistency problem**. "Approach" labelled by eye drifts between
trials and observers; "approach" labelled where the distance trace starts falling is reproducible,
and Tools ▸ Find label inconsistencies… can then genuinely check it. It also makes the label
*defensible*, which a video-only label never is.

Third appearance of the same fact: a pure-video model cannot see inter-animal distance directly; a
labeller with the trace can.

---
