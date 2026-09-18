# Precise event spotting — takeaways (2026-08-26)

What survives from a literature pass over PES methods (F3Set/F3ED, UMEG-Net,
T-DEED, ASTRM, AdaSpot, MSAGSM, the Xu et al. survey), checked against what
this project has actually measured: the stride ladder
(`Z_conclusions.md`), the confidence probe on the held-out session, and the
pose-dropout measurement. Where a paper's advice and our numbers disagree,
the numbers win and the disagreement is recorded.

## Ground truth this note is checked against

- **Target is 3–4 frames = 15–20 ms at 200 fps**, not the literature's
  "1–2 frames" (40–80 ms at 25 fps) and not a 50 ms criterion.
- Ladder (val, 54 events): 2.0 s context / 10 ms grid wins — 0 misses, median
  3 frames, 57 % within 20 ms. Context fixes recall; coarsening the grid to buy
  more context loses precision. **The remaining error is quantisation-bound.**
- Held-out test (72 events): stick 72 % within 20 ms, pellet 47 %.
- Confidence: peak height is near chance out of session (AUC 0.58 for a
  >50 ms error); curve shape (`focus × ratio`) 0.82. For the LightGBM onset
  model peak height is fine (0.67 / 0.75) — its curve is shape-constrained.
- Pose is **most** reliable at the event: `stickTip` NaN 0 % within ±100 ms
  of an event, 55 % elsewhere. "Stick is tracked" ≈ "stick is in play".

## Decisions

### Model
1. **Point codec on pixels stays the default for this task.** The "state
   codec by default" argument rests on a 50 ms tolerance we do not use. At
   20 ms the ladder shows quantisation as the limiting error, which is the
   case the displacement head exists for. One cheap arm is still worth
   running: first/last contact are the two edges of one interval, so
   `eto.segment` on the existing features yields both events for free —
   score it on the same 20 ms sweep and let the point/state question be
   empirical.
2. **Displacement head stays on the list** — the next lever after MSAGSM,
   for the reason above.
3. **Stride 4 is not free at this target.** A3 (20 ms grid) fell from 57 %
   to 37 % within 20 ms. The literature's "stride 4–8 collapses" was at
   25 fps; ours collapses at a coarser grid for the same reason.
4. **Static crop of the box** before anything else. Fixed camera, fixed
   apparatus, contact region a small fraction of a 224 px frame: most of
   AdaSpot's learned-RoI benefit at zero cost.
5. **Audio.** If the rig records it, a stick–pellet contact may be audible
   and `features/audio_changepoints.py` already exists. The cheapest possible
   high-value experiment; unexplained in the source notes, worth an
   afternoon before any more GPU.
6. Not pursuing: T-DEED (worse at the tightest tolerance in AdaSpot's table),
   ASTRM (no public code), AdaSpot's RoI selector (static crop instead),
   F3ED's CTX and multi-label head (built for ~1 000 event types; a no-op for
   two — revisit for a multi-syllable ethogram, wired to *flag* its edits
   rather than repair silently).

### Graph teacher (`ethograph/spot/graph_model.py`) — corrections
The GCN as built is the wrong arm, for four reasons that are each a small fix:
- **`ReLU(A H W)` cannot compute a distance** — linear in coordinates before
  the nonlinearity. Both events are distance crossings, which is why LightGBM
  with explicit `pellet_beakTip_dist` is hard to beat. Concatenate edge
  distances and their derivatives to the embedding before the GRU.
- **Static landmarks as nodes are a per-node bias** after z-scoring. Use the
  corners to define the coordinate frame (origin at box centroid, axes along
  edges, scale by width) and drop them as nodes. This also removes the
  "coordinates must be in `position`'s frame" trap in the YAML.
- **Mean-imputing NaN erases the strongest cue** (see ground truth). Add a
  validity channel per node.
- **|V| = 7 heterogeneous nodes is not a weight-sharing regime; don't
  mean-pool.** Concatenate the fixed, ordered node set.

**Decisive ablation, before stages 2–3:** same shift module, same bi-GRU,
same head; three trunks — dense-learnable-`A` GCN / topology GCN /
concat+MLP — all with the node-feature fixes. If they tie, the trunk is not
where the signal is. Prior: concat+MLP with distance channels wins. The graph
path is kept regardless because a fuller skeleton (head, body, tail, feet)
and multi-animal arenas are the regime it was designed for.

### Distillation — the bar changes
The rule "if the teacher can't beat LightGBM, distil from LightGBM's curves"
is wrong. A tree ensemble has no `F_teacher(t)`: prediction distillation
works, representation distillation does not, and representation matching is
the richer signal (it constrains the student at every frame, including the
99.8 % with no event). **A neural teacher that ties LightGBM is still the
better teacher.** The bar is which teacher produces the better *student* —
one more ladder arm, and the one that decides. Prediction to check: if the
teacher wins anywhere it is on the offset event (defined by what follows),
where the bi-GRU has the structural advantage — and pellet (32) is already
the harder class for pixels. Read "From Skeletons to Pixels" (Yeoh & Jiang
2026) before building stage 2; it may change the loss rather than add to it.
Never put S3D features in the teacher.

### Confidence and curation
The stack, cheapest and sharpest first; we have item 3 and neither of the
two above it:
1. **Structural rules** — exactly one 31 and one 32 per trial, 31 before 32,
   plausible gap. Free, no calibration, near-perfect precision when they
   fire. Enforce as rules, never learn them.
2. **Window disagreement in ms** — with 50 % overlapping clips every frame is
   predicted twice at different clip positions; decode each window
   independently and take the spread of decoded event times. Same units as
   the tolerance; a second free signal is how many windows fire at all.
   Windows are not exchangeable (centre positions are better): spread as
   uncertainty, position-weighted estimate as the point.
3. **Curve shape** (`focus × ratio`) — built, measured, in the TSV.
4. Calibrated probability only after temperature scaling on a held-out fold;
   raw thresholds do not transfer between sessions (measured).
5. Teacher–student disagreement and embedding distance, where both exist.
6. Ensembles / MC dropout — rarely worth it once 1–3 are in.

**Combine by rank-normalising within a session and taking the max**, never a
sum. **Validate as a risk–coverage curve** (fraction of errors caught vs.
fraction reviewed) — the defensible form of any review-time claim, and
model-agnostic, so it is a natural Ethograph output for anything in the
codec slot. Keep a small randomly sampled audit set outside any active-learning
loop or the savings number is unfalsifiable.

## Next steps, in order
1. Static crop; find out whether the rig has audio.
2. Three-trunk teacher ablation with the node-feature fixes above.
3. Confidence: structural rules + window disagreement, scored as a
   risk–coverage curve against `focus × ratio` on the held-out session.
4. Distillation stages 2–3, judged by the student, after the Yeoh & Jiang
   read.
5. Displacement head on the pixel model; then the state-codec arm via
   `eto.segment` on the same sweep.

## Open questions
- Does the teacher win on the offset event specifically?
- Does the changepoint σ-stack already supply what the multi-scale shift
  provides — is `shift_scales_ms` near-inert? (A reportable negative.)
- Is ε = 1/8 right at |V| = 7? Tuned for a 39-node graph; sweep ε = 1/4.
- Licence on `arturxe2/AdaSpot` before planning around its ASTRM
  reimplementation.
