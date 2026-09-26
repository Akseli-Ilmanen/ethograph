(target-confidence)=
# Model confidence

Every predicted label carries a `confidence` in the labels TSV (a hand-placed
label is `1.0`). The review tools threshold on it: the label grid outlines
tiles below **Flag confidence below**, **Histogram…** shows where the scores
sit per class, and **Tag low-confidence** tags exactly those tiles. The
number is computed by the model that made the prediction, and *how* depends
on the question that model answered — *which class is this frame?* for a
state event, *when did it happen?* for a point event.

## State events: which class?

```{figure} ../../_static/media/curation_stateevents.png
:alt: The segmentation model's per-frame softmax is turned into a confidence curve by normalised entropy; predictions are purged and stitched, then reviewed in bulk in the video grid, and the low-confidence trials inspected in depth.
:width: 100%

The {doc}`segmentation pipeline <../segment/index>` predicts a distribution
over the classes at every frame. Its confidence curve dips wherever the model
is torn between two classes; a segment's own confidence is the mean of its
class's probability over its span.
```

The segmentation pipeline (`eto.segment`) outputs one probability
distribution over the $C$ classes at every frame, $p_1(t), \dots, p_C(t)$
with $\sum_c p_c(t) = 1$, and the label at a frame is the largest. Two numbers
come out of it:

- **A segment's `confidence`** in the TSV is the mean probability of its own
  class $c$ over its frames $S$:

  $$
  \text{confidence} = \frac{1}{|S|} \sum_{t \in S} p_{c}(t).
  $$

- **The confidence curve** drawn under the labels during review is
  per-frame: one minus the normalised entropy of the distribution.

  $$
  \text{confidence}(t) = 1 - \frac{H(t)}{\log C}, \qquad
  H(t) = -\sum_{c=1}^{C} p_c(t)\,\log p_c(t).
  $$

  `1` means all the mass sits on one class, `0` that every class is equally
  likely. Its mean over a trial is the `model_confidence` column
  **Confidence curves…** writes to the metadata table.

## Point events: when?

```{figure} ../../_static/media/curation_pointevents.png
:alt: E2E-Spot's output layer gives one probability curve per class; the tallest peak is the point event, and its confidence is the curve's ratio times its focus; low-confidence events are reviewed frame by frame.
:width: 100%

The {doc}`LightGBM <../onset_model>` {cite:p}`ke2017lightgbm` and
{doc}`E2E-Spot <../spot/index>` {cite:p}`hong2022e2espot` models produce one
probability curve per class, and the point event sits on its tallest peak.
The confidence is the **shape** of that curve, not the height of the peak.
```

For each class the model produces a curve $p_k(t)$, the per-frame belief that
class $k$'s event is *here*; the prediction is its tallest peak
$t^\ast = \arg\max_t p_k(t)$ (a local maximum — a curve still climbing at the
trial's edge is not a peak). Entropy across classes says nothing useful here:
the alternatives are other *moments*, not other classes. So the confidence is
read off the curve's shape around its peak, within a window $w$:

$$
\text{focus} = \frac{\sum_{|t - t^\ast| \le w} p(t)}{\sum_t p(t)}, \qquad
\text{ratio} = 1 - \frac{\max_{t' \in \text{peaks},\; |t' - t^\ast| > w} p(t')}{p(t^\ast)}, \qquad
\text{confidence} = \text{focus} \times \text{ratio}.
$$

- **focus** — the share of the curve's mass within $w$ of the peak: `1` is one
  clean bump, lower a broad bump or belief spread elsewhere.
- **ratio** — one minus the tallest *rival* over the peak, a rival being
  another local maximum outside $w$ (never the peak's own shoulder): `1` is
  no rival, `0` a second candidate as tall as the first. Blind to width by
  design — width is `focus`'s job.
- **peak** — the model's own score $p(t^\ast)$, kept as a candidate.

```{figure} ../../_static/media/confidence_curve_stats.png
:alt: peak, focus and ratio on a sharp bump, a broad bump, and a curve with a rival
:width: 100%

A lone sharp bump reads near `1`; a smeared bump pulls `focus` down, a rival
elsewhere in the trial pulls `ratio` down.
```

**The window is your timescale.** $w = 2 \times$ the tolerance the labels are
believed to: the LightGBM model takes it from its own `tolerance_s`, the pixel
model from `infer.focus_window_ms`. A bump wider than that is smeared by your
own definition; a peak further away than that is a rival.

**Two curves read `0` whatever their shape**: one that is nearly nothing
everywhere (tallest peak below 0.05, else a single surviving blip would be
the cleanest bump imaginable), and one with no interior peak, or higher at an
edge than at any peak inside (the event may lie past the trial's end). Such a
label is flagged for review, never dropped.

**Several events per trial change what a second peak means.** When the pixel
model reads a curve as up to `infer.max_events_per_trial` events, a second
peak is another event, not a rival: `ratio` is refused and `focus` (over each
event's own stretch of the curve) or `peak` is written instead. The model
then also returns spurious peaks, so calibrate **Flag confidence below** on
the histogram before review rather than leaving it at its default.

**Which statistic is written is measured per model, not assumed.** On the
same held-out trials, the candidate that best separates the model's hits from
its misses (AUC) wins:

- The **LightGBM model** ranks every candidate per class when it trains and
  writes `peak` unless `focus`, `ratio` or their product wins by a clear
  margin. Its curve is shape-constrained by construction (a Gaussian-weighted
  target smoothed with the matching kernel), so its bumps all look alike and
  height is what varies. The training message says which was chosen.
- The **pixel model** writes `focus × ratio`. E2E-Spot's softmax normalises
  across classes and nothing normalises across time, so a class can sit
  moderately high for a long stretch and its peak still read as confident:
  height was near chance (AUC 0.58) where `focus`, `ratio` and their product
  reached ~0.8. The two halves look different in a histogram — `ratio` is
  bimodal (one candidate or two), `focus` sits in a middle band — so how much
  each counts is set in the GUI, below.

Frame-by-frame review draws every curve in scope under the label on a fixed
0–1 axis, so the peak, the rival and the smear behind a score are all in
view. **How often a model is right is a verdict on the model, not on a
label**: training reports the held-out hit rate per class (*peck: 6/8 within
0.05 s*) and never folds it into a label's confidence.

(target-confidence-rule)=
## Changing the rule in the GUI

Which reading is the confidence is a review preference, so it is set where
its effect is seen: the **Histogram…** popup of the label grid and the video
grid carries a *Confidence rule* panel above the bars whenever the session
has a prediction run with curves beside it (every run merged, newest per
class).

- **Rule** — `focus × ratio` (the default), `ratio` alone (one candidate or
  two), `focus` alone (sharp or smeared), `peak`, or **custom**:
  `ratio × (α + (1 − α)·focus)` with one slider — α = 1 is `ratio`, α = 0 the
  product.
- **Same event within** — the window $w$ in ms.

Every change redraws the histogram and restyles the grid's tiles at once, with
the threshold line in the same picture. **Apply to labels** writes the values
into the labels — only automated labels that have a curve; manual and curated
ones are a human's word and never change — as one undo step per trial
(`Ctrl+Z` takes it back). Closing without applying puts the original values
back. The rule, α and window are remembered across sessions.

**Copy for project.yaml** puts the same choice on the clipboard as the
`infer:` lines of a spot project config, so the next `inference()` writes
confidence the way the review settled on and the grid and the pipeline never
disagree about what the number means (`infer.confidence`,
`infer.confidence_alpha` for the custom rule):

```yaml
infer:
  confidence: ratio
  focus_window_ms: 100
```

## Reviewing by confidence

How the grids sort, flag and pre-click on this number is on the
{doc}`review grids page <grids>`, under *Reviewing by confidence*.
