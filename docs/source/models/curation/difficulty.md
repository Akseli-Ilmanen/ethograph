(target-curation-difficulty)=
# Curator feedback

Where you overruled the model, it should pay extra attention next time. That
is the whole idea of the **Curator feedback** block at the bottom of the
Curation section: a trial you had to correct is flagged `hard` in the
metadata table's `difficulty` column, and the next training run draws it
more often. The column is a checklist in the trials table's filter like
`curated`, so "show me the trials the model got wrong" is one click, and the
segmentation pipeline reads it back:
`train.oversample: {column: difficulty, weights: {hard: 3.0}}` draws a hard
trial three times as often ({doc}`../segment/config`). Auto-flagging only
ever adds: a trial you flagged by hand stays hard whatever it scored.

```{admonition} This is not confidence
:class: important

A low **confidence** tells *you* where to look ({doc}`confidence`, the grids'
**Flag confidence below**). It never flags a trial hard. The model already
works on the classes it is unsure of; what it cannot know is where it was
**confident and wrong**, and only a curator can tell it. Keep the two apart:
confidence guides the review, disagreement guides the training.
```

## By hand

`Ctrl+T` (or the **Hard trial** box) flags the current trial, and again puts
it back to `normal`. Use it for a trial that is simply difficult — an odd
posture, an occlusion, a rare variant — whether or not the model got it right
this time.

## By score, once, by your hand

Which trials the model got wrong is measured, not guessed — but which of
them the next run should see more often is *decided*, not thresholded by
default. Every prediction run keeps the labels it wrote in its folder under
`labels/` (`{stem}_predictions.tsv`, beside `onset_curves.npz`), and each
trial's final, curated labels are scored against what its run said — an F1
per trial and event type:

- **State labels** match at IoU ≥ 0.5, the `f1@50` the pipeline selects its
  checkpoints on.
- **Point labels** match within the run's own **tolerance** — the precision the
  model was trained to, read from the run folder (the onset model's
  `tolerance_s`, half of spot's `focus_window_ms`). Nothing to pick; set the
  block's **Tolerance** only to judge every run at one number, or for an old
  run folder that carries none.

The scores land in the metadata table as `review_f1_state` and
`review_f1_point` (blank where a trial had nothing of that type) and nothing
else happens. A trial no run predicted into is not scored. The comparison
runs by itself, once, the moment the last trial the table shows becomes
curated; **Score now** runs it without waiting, and a
{doc}`workflow <workflows>` step (**Score trials against the model**) presses
it.

Then **Histogram…** shows the scores as a distribution, one histogram per
event type. What you are looking for is a *split*: a cluster of trials the
model handled and a tail, or a second bump, it did not. Put the threshold in
the gap — the line and the red bars follow it — and press **Flag N trials
hard**. That is the one moment the scores touch the `difficulty` column, and
it is your press, remembered for next time but never repeated for you. There
is no default cut-off because 0.5 means nothing in general: a clean
behaviour on a good camera may split at 0.8, a hard one at 0.3.

**Flag few.** The count under the histogram turns red past a quarter of the
scored trials. With `train.oversample` on, that many hard trials would be
most of what the model sees, and a flag most trials carry says nothing. If
the whole distribution is low, the answer is a better model or more
training data, not more flags.

```{admonition} The scores are anchored to the model
:class: note

A curated label is a prediction a human accepted, so a session confirmed
with `N` fifty times in a row scores near 1 whatever the model did. The F1
is honest in proportion to how much you actually moved or deleted. Read it
as "how much the curator had to change", not as an independent benchmark —
the pipeline's own evaluation on a held-out split is that.
```

