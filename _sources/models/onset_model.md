(target-onset-model)=
# LightGBM for point events

EthoGraph includes a lightweight model for predicting **point events**, built
on scikit-learn's {cite:p}`pedregosa2011sklearn` histogram-based gradient boosting (`HistGradientBoostingClassifier`), inspired by
Microsoft's LightGBM {cite:p}`ke2017lightgbm`. It is a small classifier that detects the onset of an
event from a local window of hand-crafted features: the first time a mouse
touches a lever, the frame a bird lands, the moment a beak opens. You label
the moment in a handful of trials, tick the features it should look at, and it
fills in the rest.

**Model ▸ LightGBM: Train…** and **Model ▸ LightGBM: Predict…**

---

## How it works

For each frame, the model sees a **window** of your chosen features centred on
that frame and answers one question: *is the event here?*

* **Targets.** Frames within `tolerance_s` of your labelled event count as
  positive, weighted by a Gaussian bump peaking at the event, so a frame one
  tick off counts less than the exact frame. Far negatives are subsampled.
* **Inference** smooths the per-frame probability with the same tolerance — a
  plateau of near-hits beats one spurious spike — and takes the tallest peak.

Each class gets **its own binary classifier** over the same features, window
and tolerance. The design matrix is built once per trial and reused, so
predicting five classes costs barely more than predicting one.

---

## Training

1. **Name the model** — leave the combo on *New model…* and type a name, or
   pick an existing model to add more training data to it.
2. **Tick the point events to predict.** Only classes marked as point events
   in {doc}`mapping.txt <../advanced/labels/mapping>` are listed; tick as many as you like.
3. **Tick the features.** Ticking `speed ▸ keypoints ▸ beak, head` gives two
   input columns. **Every dim has to be pinned to explicit values** — that
   frozen list *is* the model's input layout, which is what lets the model run
   on another session.

   Ticking a feature's **d/dt** box adds its rate of change beside every column
   it produces (`np.gradient`: central differences, centred on the frame, so a
   turn in the signal shows up at the frame it happened). The classifier sees
   each tap of the window on its own and cannot difference them, so *how fast
   is this changing* has to be handed to it as its own input — worth a tick
   when the event is a change of speed or direction rather than a level.
4. **Tick the existing labels to read as inputs** (optional) — see
   {ref}`below <target-onset-model-label-inputs>`.
5. **Set the parameters.** `Window size` is how much context the classifier
   sees around each frame; `Tolerance` is how precisely you believe your own
   labels.
6. **Add current session's events**, then **Train**.

Only trials **visible in the trials table** contribute, so the table's filters
double as a training-set selector. A trial carrying none of the ticked events
is skipped, and one carrying only some contributes only to those — an
unlabelled trial is not evidence that the event never happened, so it is never
used as a negative example for that class.

Once a model exists its targets, features, label inputs, window and tolerance
are **read-only**: they define the classifier's input columns, so editing them
would invalidate every training trial already stored. To change them, make a
new model. To add more sessions, open the dialog there, pick the model, and
press **Add current session's events**.


(target-onset-model-label-inputs)=
### Existing labels as inputs

You can also pass existing labels as features to the model.

* a **state** class becomes its **on/off indicator** — `1` inside every
  interval of that class, `0` outside. That is the whole of what a state says.
* a **point** class becomes a **Laplacian bump** centred on the event, at two
  hard-coded widths (0.1 s and 1 s), one column each — the same kernel
  EthoGraph puts on {doc}`changepoints <../advanced/changepoints/index>`, for the same reason:
  the narrow peak points straight at the moment while the long tails stay
  readable from far away, so one column carries both *it is here* and *it was
  a while ago*.
---

## Predicting

Pick a trained model, choose the **individual** — whose data is read *and*
whose labels these are — and press **Predict missing onsets**. Two things are
never touched:

* **Trials that already carry a class** keep what they have — the model fills
  gaps, it never overrides. A trial that already has *one* class can still
  receive the others.
* **Trials the trials table hides.** Training and prediction both run over
  exactly the trials the {doc}`trials table <../advanced/metadata>` shows — its filters
  are the one trial filter in EthoGraph, so filtering `genotype = wt` there
  trains on and predicts into wild-type trials only. The dialog has no
  filters of its own; it says how many trials it will run over, read off the
  table.

Predictions land in memory like any other label, stamped
`labeling_method = automated` — they draw dotted on the plots until you
{doc}`curate <curation>` them, and a trial holding one stays red in the trial
list. **Review predictions…** at the bottom of the dialog opens the
label grid view on exactly what the run
just wrote — those classes, those trials — so you can check the video frame at
each one and either click through to fix it or mark it curated.

---

(target-onset-model-confidence)=
## Confidence

A predicted label's **`confidence`** is a statistic of that class's
probability curve around its tallest peak — its height, or how concentrated
the curve is there and whether a rival peak stands elsewhere — chosen per
class on the trials the model did not see, and named in the training
message. The curve is the **dashed line** frame-by-frame review draws under
the label, so the number is always something you can see. The statistics,
the equations and how they compare with the segmentation pipeline's entropy
are in {doc}`the confidence page <confidence>`, together with reviewing
by confidence in the label grid.
