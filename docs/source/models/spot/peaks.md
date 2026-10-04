(target-spot-peaks)=
# From probabilities to events

A point-event model never outputs an event. For every class it outputs a
**curve**: the per-frame probability that the class's event is *here*, over
the whole trial. E2E-Spot gets it from one softmax per frame across the
classes and background; the {doc}`LightGBM model <../onset_model>` from a
per-class classifier smoothed with its tolerance kernel. Turning a curve
into a label is a rule, and the rule rests on one assumption about your
trials. This page is that rule.

## One event per class per trial

The {doc}`trial <../trial_windows>` is the unit a model reads, and by
default **each class happens at most once in it**. The prediction is the
tallest interior peak of the class's curve, with no threshold: the best
candidate wins, however low. A threshold would have to be calibrated per
class and per model before you could see a single prediction; sorting the
{ref}`review grids <target-curation-grids>` by confidence does the same job
with the picture in view.

The assumption is enforced where it can be: `materialise()` refuses a trial
labelled twice for a class, so a recording in which the event repeats is cut
into trials at the alignment ({ref}`target-trial-windows`) or spotted with
the cap raised (below). It is also what gives the confidence its meaning: a
second peak in the curve is *doubt about when*, not a second event, and
`ratio` reads it that way. With one event per class the order of
`labels.classes` is the order the events happen in, and
`infer.flag_out_of_order` sets every event of a trial that breaks it to
confidence `0`.

## When the class is absent

Every trial is assumed to hold the event, so a curve that is nearly flat
still has a tallest sample. Below **`infer.min_peak`** (default `0.05`) the
class is **absent** from the trial and no label is written. The floor is set
deliberately low, for three reasons:

- a false positive costs one `Backspace` in
  {ref}`frame-by-frame review <target-curation-frame>`; a missed event is
  invisible, and a missing label cannot be reviewed;
- a curve that low scores a confidence near `0` anyway, so it lands on the
  first screens of a grid sorted lowest first;
- the model's own scores decide, not a number chosen before seeing them.

Raise it for a class that is often absent, once the histogram shows where
the absent trials sit; set it to `0` to write every curve. The LightGBM
dialog's **Min confidence** does the same for that model and is near `0` by
default.

Every other curve on which nothing was found is written at confidence `0`
and flagged: one with no interior peak, or higher at the trial's edge than
at any peak inside it, is the model saying *maybe after the end*, which is
worth a look rather than a deletion.

(target-spot-several-events)=
## Several events per trial

`infer.max_events_per_trial` lifts the cap. Every interior peak at least
`infer.min_event_gap_s` from a taller one is then an event, tallest first,
up to the cap, and each is scored over its own stretch of the curve. Three
things follow:

- a second peak is another event, not a rival, so `ratio` and the rules
  built on it (`product`, `custom`) are refused: set `infer.confidence` to
  `focus` or `peak`;
- the model now returns spurious peaks beside real ones, so **Flag
  confidence below** has to be calibrated on the grid's histogram before
  review rather than left at its default;
- `infer.flag_out_of_order` is refused, since the order between classes is
  undefined.

Prefer cutting the trials when you can: the one-event rule needs no
threshold at all.

## Then the confidence

What is written beside the event, why it is the curve's *shape* rather than
its height, and how to change the rule from the histogram, are on the
{doc}`confidence page <../curation/confidence>`.
