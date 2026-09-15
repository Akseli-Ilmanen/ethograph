(target-onset-model)=
# LightGBM for point events

A small classifier that predicts **point events** — the first time a mouse
touches a lever, the frame a bird lands, the moment a beak opens — from a
local window of your existing features. Label the moment in a handful of
trials, tick the features it should look at, and it fills in the rest.

Unlike {doc}`action segmentation <segment/index>` and
{doc}`pixel event spotting <spot/index>`, this model is **entirely in the
GUI**: no Python scripting, no config files, no GPU. It trains and predicts on
an ordinary CPU in seconds, so it is the right starting point if you are not
comfortable with code.

**Model ▸ LightGBM: Train…** and **Model ▸ LightGBM: Predict…**

Under the hood it is scikit-learn's {cite:p}`pedregosa2011sklearn`
`HistGradientBoostingClassifier`, inspired by LightGBM {cite:p}`ke2017lightgbm`.

---

## How it works

For each frame, the model sees a **window** of your chosen features centred on
that frame and answers one question: *is the event here?* Frames within
`tolerance_s` of a labelled event count as positive, weighted so the exact
frame counts most. At inference the per-frame probability is smoothed with the
same tolerance and the label goes on the tallest peak. Each class gets its own
binary classifier over the same inputs.

---

## Training

1. **Name the model** — leave the combo on *New model…* and type a name, or
   pick an existing model to add more training data to it.
2. **Tick the point events to predict.** Only classes marked as point events
   in {doc}`mapping.txt <../advanced/labels/mapping>` are listed.
3. **Tick the features.** Ticking `speed ▸ keypoints ▸ beak, head` gives two
   input columns. Every dim has to be pinned to explicit values — that frozen
   list *is* the model's input layout, which is what lets it run on another
   session. A feature's **d/dt** box adds its rate of change as an extra
   column, worth ticking when the event is a change of speed or direction
   rather than a level.
4. **Tick the existing labels to read as inputs** (optional) — see
   {ref}`below <target-onset-model-label-inputs>`.
5. **Set the parameters.** `Window size` is how much context the classifier
   sees around each frame; `Tolerance` is how precisely you believe your own
   labels.
6. **Add current session's events**, then **Train**.

Only trials **visible in the trials table** contribute, so the table's filters
double as a training-set selector. A trial carrying none of the ticked events
is skipped; one carrying only some contributes only to those, and is never
used as a negative example for the classes it lacks.

Once a model exists its targets, features, label inputs, window and tolerance
are **read-only**, because they define the input columns of every training
trial already stored. To change them, make a new model. To add more sessions,
open the dialog there, pick the model, and press **Add current session's
events**.

(target-onset-model-label-inputs)=
### Existing labels as inputs

Labels of another class can be inputs too:

* a **state** class becomes its on/off indicator — `1` inside every interval,
  `0` outside;
* a **point** class becomes a Laplacian bump centred on the event, at two
  widths (0.1 s and 1 s) — the same kernel EthoGraph puts on
  {doc}`changepoints <../advanced/changepoints/index>`, so one column says both
  *it is here* and *it was a while ago*.

---

## Predicting

Pick a trained model, choose the **individual** — whose data is read *and*
whose labels these are — and press **Predict missing onsets**. Two things are
never touched:

* **Trials that already carry a class** keep what they have — the model fills
  gaps, it never overrides. A trial with *one* class can still receive the
  others.
* **Trials the trials table hides.** Training and prediction both run over
  exactly the trials the {doc}`trials table <../advanced/metadata>` shows; the
  dialog has no filters of its own and says how many trials it will run over.

Predictions land in memory like any other label, stamped
`labeling_method = automated`: dotted on the plots until you
{doc}`curate <curation>` them, red in the trial list. **Review predictions…**
at the bottom of the dialog opens the label grid view on exactly what the run
just wrote, so you can check the video frame at each one and fix it or mark it
curated. Each prediction also carries a `confidence`, explained on
{doc}`the confidence page <confidence>`.
