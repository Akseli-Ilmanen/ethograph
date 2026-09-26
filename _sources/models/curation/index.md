(target-curation)=
# Curating labels

A model's predictions come back into the GUI as labels nobody has looked at
yet. **Curation** is looking at them, and Ethograph makes that fast by
telling you *where* to look: every predicted label carries a **model
confidence**, and the review tools sort, flag and threshold on it, so the
doubtful labels are on the first screen instead of scattered through the
trials. How the confidence is computed depends on the kind of label, and so
does the routine that suits it. Both are below.

Everything about curation lives in one place: the **Curation** section at the
bottom of the **Labels** tab.

## What curation means

Every label carries a `labeling_method` — the vocabulary of
[ndx-ethogram](https://github.com/catalystneuro/ndx-ethogram):

| Value | Meaning | Drawn as |
|-------|---------|----------|
| `manual` | A human placed or last edited it | solid outline |
| `automated` | A model produced it and nobody has looked at it | **dotted** outline |
| `curated` | Automated output a human looked at and let stand | solid outline |

Curation is the move *automated → curated*. It never touches a manual label
(a human already vouched for it), and it never runs backwards: editing a
label makes it manual, re-running a model over a trial can add new automated
labels, and nothing else changes a method.

A **trial is curated** when none of its labels is still automated. That
verdict is everywhere you navigate — the trial combo in the Navigation section
and the `Trial 12 (12/173)` counter in the bottom bar are green for a curated
trial and red for one with automated labels left — and it is written to the
metadata table's `curated` column (`yes` / `no`), refreshed every few seconds
while you work (see {doc}`../../advanced/metadata`). Predicting new labels into
a curated trial turns it red again until those are curated too.

### Where the verdict is saved

Nothing is written until you start curating. Opening a dataset arms nothing:
curation becomes **active** the moment you drop label classes into the scope
area or curate anything, and only then does the verdict start being saved
(a line in the terminal says so). Loading another dataset disarms it again.

Arming is also the one moment a metadata file appears. The `curated` column is
Ethograph's own bookkeeping, so it is never written into a recording or into
`.ethograph/alignment.nwb`: that write happens in place, and for a non-NWB
dataset the alignment NWB is the only holder of your trial timing. Instead the
metadata table you have loaded is copied to a sidecar `metadata.tsv`
next to the data, and that file is the metadata table from then on — it is
what the next load reads, and where later edits to trial metadata go. An
existing metadata file is used as it stands, never overwritten.

## Two kinds of label, two routines

A model that predicts **point events** answers *when*; one that predicts
**state events** answers *which class is this frame*. The confidence each
one writes is built to answer the same question, and it decides which review
surface to open first.

### Point events: when?

```{figure} ../../_static/media/curation_pointevents.png
:alt: E2E-Spot's output layer gives one probability curve per class; the tallest peak is the point event, and its confidence is the curve's ratio times its focus; low-confidence events are reviewed frame by frame.
:width: 100%

The {doc}`LightGBM <../onset_model>` and {doc}`E2E-Spot <../spot/index>`
models produce one probability curve per class, and the point event sits on
its tallest peak. The confidence is the **shape** of that curve: `ratio`
(how much taller the peak is than its best rival) times `focus` (how much of
the curve sits close to the peak). One clean bump reads near 1; a second
candidate elsewhere in the trial, or a smeared bump, pulls it down.
```

The routine: drag the class into the scope, pick **Frame-by-frame review**,
and walk the events one at a time. Each event is centred in a short window
with its curve drawn underneath, so a low score explains itself before you
press a key — `Enter` confirms the frame on screen, `←`/`→` moves it,
`Backspace` deletes an event that never happened. To start with the doubtful
ones, open the {ref}`label grid <target-curation-grids>` sorted by
confidence and double-click a tile: the review opens at that boundary. See
{ref}`target-curation-frame`.

### State events: which class?

```{figure} ../../_static/media/curation_stateevents.png
:alt: The segmentation model's per-frame softmax is turned into a confidence curve by normalised entropy; predictions are purged and stitched, then reviewed in bulk in the video grid, and the low-confidence trials inspected in depth.
:width: 100%

The {doc}`segmentation pipeline <../segment/index>` predicts a distribution
over the classes at every frame. Its confidence curve is one minus the
normalised entropy of that distribution: 1 where all the mass sits on one
class, dropping wherever the model is torn between two. A segment's own
`confidence` is the mean of its class's probability over its span.
```

The routine has three steps. **Automatic refinement** — purging short
segments and stitching same-class neighbours — happens in the pipeline's own
post-processing before the labels reach the GUI ({doc}`../segment/config`),
with **Purge short labels** available again as a
{doc}`workflow step <workflows>`. **Bulk review** opens the
{ref}`video grid <target-curation-grids>` on one class at a time: clips of
similar length play side by side, and a click is a verdict. **In-depth
review** is for the trials the grid flagged: a trial is flagged as soon as
*one* of its labels falls below **Flag confidence below** (there is no
separate trial threshold), a double-click jumps the main GUI there, and the
confidence curve under the labels shows exactly which boundary to inspect.

## In this section

- {doc}`modes` — the scope (which classes) and the four ways a label gets
  curated: by hand, in bulk, by inspection, or frame by frame.
- {doc}`grids` — the label grid and the video grid: what every control does,
  and what a click means.
- {doc}`confidence` — how each model computes the number, and how to change
  the rule from the histogram.
- {doc}`difficulty` — flagging hard trials, and scoring how the model did per
  trial once a session is curated.
- {doc}`workflows` — recording the whole routine and replaying it in one press.

```{toctree}
:maxdepth: 1
:hidden:

Ways to curate <modes>
Review grids <grids>
Model confidence <confidence>
Hard trials and review F1 <difficulty>
Curation workflows <workflows>
```
