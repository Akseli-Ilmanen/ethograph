(target-curation)=
# Curating labels

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
while you work (see {doc}`../advanced/metadata`). Predicting new labels into a curated
trial turns it red again until those are curated too.

Everything about curation lives in one place: the **Curation** section at the
bottom of the **Labels** tab.

## Where the verdict is saved

Nothing is written until you start curating. Opening a dataset arms nothing:
curation becomes **active** the moment you drop label classes into the scope
area or curate anything, and only then does the verdict start being saved
(a line in the terminal says so). Loading another dataset disarms it again.

Arming is also the one moment a metadata file appears. The `curated` column is
EthoGraph's own bookkeeping, so it is never written into a recording or into
`.ethograph/alignment.nwb`: that write happens in place, and for a non-NWB
dataset the alignment NWB is the only holder of your trial timing. Instead the
metadata table you have loaded is copied to a sidecar `{stem}_metadata.tsv`
next to the data, and that file is the metadata table from then on — it is
what the next load reads, and where later edits to trial metadata go. An
existing metadata file is used as it stands, never overwritten.

## Scope: which labels

Curation acts on a *scope* of label classes. Drag rows out of the label tables
above into the drop area — a multi-selection drags as one — and their ids are
listed there. Empty (or **All**) means every class; **Reset** empties the area
so other labels can be dragged in. The scope is remembered per dataset.

## Modes: how a label gets curated

**Manual (trial level)** — the default. Placing, moving or deleting a label
makes it manual, as always. Press `Ctrl+C` (or use **Tools ▸ Labels: Bulk editing…**) and every
automated label in scope of the current trial becomes curated; manual labels
stay manual.

**Curate…** in **Tools ▸ Labels: Bulk editing…** does the same across the trials
and label classes you pick there. It asks first, because one click here says a human
approved labels nobody looked at, and **curating cannot be undone** — `Ctrl+Z`
takes back label edits, not curations. (Nothing reaches disk until you save,
so closing without saving still discards it.) Reach for it when a review left
some unjudged — a grid browsed without curating, a review stopped partway —
not as a way to skip looking.

**Inspect is enough (trial level)** — merely opening a trial curates its
automated labels in scope. Use it when a model is good enough that looking at
the trial is the review. The mode is per dataset, so it never follows you
silently into another one.

(target-curation-frame)=
**Frame-by-frame review** — the labels in scope become a queue of boundaries
(one per point event, a start then an end per state event, in time order)
walked one at a time, each centred in a small **View window**
(untick **Locked around label** to pan the whole trial). The boundary being
reviewed is named in large coloured text; the keys are drawn in the section,
and **Shortcuts…** spells them out. By default the queue holds only automated
boundaries — a human already vouched for manual and curated ones — untick
**Show automated only** to walk those too. The **Order** combo walks the queue
*Trial-by-trial* (every boundary of a trial, then the next trial) or
*Label-by-label* (one class across every trial, then the next class), and with
**Jump to next after Enter/Backspace** ticked (the default) confirming or
deleting moves straight on to the next boundary:

| Key | Action |
|-----|--------|
| `←` / `→` | Step the video one frame |
| `Enter` | Confirm: the frame on screen becomes the boundary. A boundary that moved makes the label **manual** (`confidence = 1.0`); one confirmed where it stood becomes **curated** |
| `Backspace` / `Delete` | The event should not exist — delete it (both boundaries of a state event) and move on |
| `N` | Next boundary. With **Click N curates current** ticked (the default) the boundary you leave is curated |
| `B` | Back to the previous boundary |
| `Space` | Play / pause |

Navigating trials the normal way (trial combo, `Up`/`Down`) pulls the review
along to that trial's first boundary. Nothing reaches disk until you save with
`Ctrl+S`.

Reviewing what an lightgbm model (LightGBM {cite:p}`ke2017lightgbm`) predicted, you also get the
**curve it predicted from**: a dashed line per label class, in the class's own
colour, on a 0–1 right-hand axis. Only the classes **in scope** are drawn, so
dragging in one class shows that class's belief and nothing else. A low
confidence then explains itself — a second peak elsewhere in the trial means
the model was torn, a flat line means it never found anything. Labels placed
by hand have no curve and none is drawn.

Seeds don't have to be hand-placed. Because the queue is built from the labels
TSV, you can generate first-guess labels **programmatically from a time-series
criterion** — say, the first frame where beak opening exceeds a threshold
width — write them into the `{name}_labels.tsv` file with
`labeling_method = automated` (see {ref}`the column reference
<target-exporting-labels>`), load it into the GUI, and walk the guesses here.
A rough automatic pass plus a fast frame-accurate review is often far quicker
than either alone.

(target-curation-grids)=
## The grids

Two buttons in the section open review grids on the scope; both come with the
same **mode** combo and a **Done** button. Their *Setup* tab lists the labels
in scope for clarity but cannot change them — the scope area is the one place
labels are chosen, so close the grid, drag other rows in, and open it again.

Setup's **Labeling method** combo picks which labels of those classes the grid
is about: *All labels*, *Manual only*, *Curated only*, *Manual or curated*, or
*Automated only* — a model's output that nobody has looked at, which is what a
prediction review is for. *Manual or curated* is there for checking your own
work: both mean a human vouched for the label, and which of the two it is says
only how it got there — *Manual only* and *Curated only* are available
alongside it when you want to isolate one. Like the rest of the grid setup the choice is remembered across
sessions, and a {doc}`workflow <../advanced/labels/workflows>` step sets it per grid.

Both grids take a **Sort**: by trial (the default in the label grid) or by
**confidence**, lowest or highest first. Sorting by confidence is the point of
having it — it puts every doubtful label on the first screens instead of
scattering them through the trials, so a model review starts where it should.
The video grid adds **duration** (its default), which keeps clips of a similar
length together so they end around the same time when they play. The choice is
remembered, and reordering never moves a verdict: clicks are keyed by label,
not by position.

**Label grid view…** shows the video frame at every boundary in scope — one
tile per point event, a start and an end tile per state event, per camera —
titled with label, trial, time, confidence and method, with **Flag confidence
below** outlining doubtful tiles in red and **Histogram…** showing where the
scores pile up. The grid exports to
a paginated PDF.

A **double click** always jumps the main GUI to that trial and time — in
frame-by-frame mode, straight into the review at that boundary — whichever
mode the grid is in, and it leaves the verdicts exactly as they were. So
judging a batch and going to look at one of its labels are not two modes to
switch between: click to judge, double-click to go and see.

A **single** click is a verdict, and the mode says which:

* *Click = curated* — click the tiles that are right (green); **Done** curates
  those labels.
* *Click = uncurated, rest = curated* — for a batch that is mostly right:
  click only the bad ones (orange), **Mark low-confidence as uncurated**
  pre-clicks what the threshold outlines (it exists only in this mode — a low
  score is a reason to doubt a label, never to approve it), and **Done**
  curates every other label.

When the scope holds more than one label class the grid gets a **Label**
combo, which narrows it to one class at a time (each choice says how many
tiles it has). It narrows the *operations* too, which is the point of it:
**Mark low-confidence as uncurated**, **Done** and the PDF apply to the class
on screen and to no other — so "rest = curated" means the rest of *that*
class, and a scope of several classes is curated one class at a time without
reopening the dialog. Clicks on a class you have filtered away are simply out
of **Done**'s reach until you show it again.

**Video grid…** plays the labels instead of freezing them — a state event's
whole span, a point event's window (**Window around point events**) with a
red marker in the corner on the frame the event falls on. It is built for
comparison: only clips of **one label class** are on screen at a time
(**Previous / Next label** switch the class, greyed out when there is only
one), they are **sorted by duration** so clips of similar length share a
screen (**Clips on screen** sets how many; **Previous / Next clips** step
through the rest), and the view never scrolls — one **Play** button and one
slider spanning the longest clip on screen drive every tile at once, played
once and stopped, shorter clips holding their last frame; **←/→** pause and
step every tile one frame back or forward. The **speed** field
next to Play opens at the speed last used in the grid (100 % the first time),
independent of the GUI's playback speed, as a percentage of real time. The layout choices —
window around point events (0.5 s by default), clips on screen, columns —
are remembered across sessions and datasets, like the label grid's column
count. Each tile's caption says where in the
trial the label sits (`at 0.03 s` for a point event, `1.20–1.85 s` for a
state) and whether the clip had to be cut at the video's start or end — so a
point event that seems to show "the start of the trial" can be told apart
from one whose window was clipped. Clips decode a screenful at a time at a
reduced size, so opening long events takes a moment; while a screenful is
showing, the next one is already decoding in the background, so stepping on
is quick. Clicks mean the same as in the label grid.

## Doing all of that again next session

Scope, mode, grid layout and review window are settings you will set the same
way every time you review the same behaviour. **Model ▸ Curation workflows…**
records that whole routine — filters, prediction, scope, grid,
review, save — and replays it in one press. See
{doc}`../advanced/labels/workflows`.
