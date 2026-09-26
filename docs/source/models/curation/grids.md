(target-curation-grids)=
# Review grids

Two buttons in the Curation section open review grids on the
{ref}`scope <target-curation-scope>`: **Label grid view…** freezes every
boundary as a video frame, **Video grid…** plays the labels as clips. Both
come with the same **mode** combo and a **Done** button, and a click means
the same in both. Their *Setup* tab lists the labels in scope for clarity but
cannot change them — the scope area is the one place labels are chosen.

The controls fall into five families, coloured the same way in both
screenshots below:

| Colour | Family | What it decides |
|--------|--------|-----------------|
| blue | **What is on screen** | which class, in which order |
| orange | **Confidence threshold** | which tiles are outlined as doubtful |
| green | **Verdicts** | what a click means, and **Done** |
| pink | **What a tile says** | the label, where it sits, its confidence and method |
| purple | **Playback, navigation, export** | stepping through the rest, and the PDF |

## Setup: which labels, in which order

Setup's **Labeling method** combo picks which labels of those classes the grid
is about: *All labels*, *Manual only*, *Curated only*, *Manual or curated*, or
*Automated only* — a model's output that nobody has looked at, which is what a
prediction review is for. *Manual or curated* is there for checking your own
work: both mean a human vouched for the label, and which of the two it is says
only how it got there — *Manual only* and *Curated only* are available
alongside it when you want to isolate one. Like the rest of the grid setup the
choice is remembered across sessions, and a {doc}`workflow <workflows>` step
sets it per grid.

Both grids take a **Sort**: by trial (the default in the label grid) or by
**confidence**, lowest or highest first. Sorting by confidence is the point of
having it — it puts every doubtful label on the first screens instead of
scattering them through the trials, so a model review starts where it should.
The video grid adds **duration** (its default), which keeps clips of a similar
length together so they end around the same time when they play. The choice is
remembered, and reordering never moves a verdict: clicks are keyed by label,
not by position.

## Label grid view

```{figure} ../../_static/media/curation_framegrid_annotated.png
:alt: The label grid with its controls numbered: label combo, sort, confidence threshold, mode row, Done, a tile's title, and the frame counter with Export PDF.
:width: 100%

One tile per boundary, per camera, titled with what it is and how sure the
model was.
```

1. **Label** — when the scope holds more than one class, narrows the grid to
   one class at a time; each choice says how many tiles it has. It narrows
   the *operations* too: **Mark low-confidence as uncurated**, **Done** and
   the PDF apply to the class on screen and to no other, so a scope of
   several classes is curated one class at a time without reopening the
   dialog. Clicks on a class you have filtered away are out of **Done**'s
   reach until you show it again.
2. **Sort** — by trial then time, or by confidence, lowest or highest first.
3. **Flag confidence below** and **Histogram…** — the threshold. Every tile
   below it is outlined red, in the grid and in the PDF. The number is typed
   in full rather than stepped, so a model whose scores sit at the bottom of
   the range can be flagged at `0.0002` as easily as at `0.6`; **Histogram…**
   shows where the scores actually sit, per class, before you commit — with a
   bimodal statistic such as `ratio` the gap is where the threshold goes. The
   popup also holds the {ref}`confidence rule <target-confidence-rule>`.
4. **Mode**, **Mark low-confidence as uncurated**, **Clear** — what a click
   means (below). **Mark low-confidence as uncurated** pre-clicks exactly the
   outlined tiles, and exists only in the *Click = uncurated* mode: a low
   score is a reason to doubt a label, never to approve it. **Clear** forgets
   every click.
5. **Done** — applies the verdicts to the labels on screen and closes the
   grid. Curating is not undoable; nothing reaches disk until you save.
6. **A tile's title** — the label's name and id, then trial, camera,
   individual, the boundary's time and its `labeling_method`; the confidence
   sits at the right. A state event has a start tile and an end tile.
7. **The counter** and **Export PDF…** — how many tiles the class has, and a
   paginated PDF of the grid as it stands, red outlines included.

## Video grid

```{figure} ../../_static/media/curation_videogrid_annotated.png
:alt: The video grid with its controls numbered: class header, sort, confidence threshold, mode row with Done, a tile, the playback bar, and the previous and next buttons.
:width: 100%

Clips of one class, sorted by duration, driven by one Play button and one
slider.
```

1. **The class header** — the class on screen, its event type and how many
   clips it has. Only clips of **one label class** are on screen at a time,
   so what you compare is like with like.
2. **Sort** — by duration (the default), so clips of similar length share a
   screen and end around the same time, or by trial or confidence.
3. **Flag confidence below** and **Histogram…** — the same threshold as the
   label grid.
4. **Mode**, **Mark low-confidence as uncurated**, **Clear**, **Done** — the
   same verdicts as the label grid.
5. **A tile** — its title names the class and trial, its confidence sits at
   the right, and the caption says camera, individual, where in the trial the
   label sits (`1.77–1.82 s` for a state event, `at 0.03 s` for a point
   event), its duration and its `labeling_method`. The caption also says
   whether the clip had to be cut at the video's start or end — so a point
   event that seems to show "the start of the trial" can be told apart from
   one whose window was clipped. A point event plays its window (**Window
   around point events** on the Setup tab, 0.5 s by default) with a red
   marker in the corner on the frame the event falls on.
6. **Play**, **speed**, the **slider** and the clock — one slider spanning the
   longest clip on screen drives every tile at once, played once and stopped,
   shorter clips holding their last frame; **←/→** pause and step every tile
   one frame back or forward. The speed opens at the value last used in the
   grid (100 % the first time), independent of the GUI's playback speed, as a
   percentage of real time. The view never scrolls.
7. **Previous label** and **Previous clips** — step back a class, or back a
   screenful within the class.
8. **Next clips** and **Next label** — step on a screenful (**Clips on
   screen** on the Setup tab sets how many), or on to the next class; the
   label buttons are greyed out when the scope holds one class.

Clips decode a screenful at a time at a reduced size, so opening long events
takes a moment; while a screenful is showing, the next one is already
decoding in the background, so stepping on is quick. The layout choices —
window around point events, clips on screen, columns — are remembered across
sessions and datasets, like the label grid's column count.

## What a click means

A **double click** always jumps the main GUI to that trial and time — in
frame-by-frame mode, straight into the review at that boundary — whichever
mode the grid is in, and it leaves the verdicts exactly as they were. So
judging a batch and going to look at one of its labels are not two modes to
switch between: **click to judge, double-click to go and see.**

A **single** click is a verdict, and the mode says which:

* *Click = curated* — click the tiles that are right (green); **Done** curates
  those labels.
* *Click = uncurated, rest = curated* — for a batch that is mostly right:
  click only the bad ones (orange), **Mark low-confidence as uncurated**
  pre-clicks what the threshold outlines, and **Done** curates every other
  label. With a **Label** filter active, "rest" means the rest of *that*
  class.

## Reviewing by confidence

Sort by confidence, lowest first, set **Flag confidence below** with the
histogram in view, and the first screens hold every label the model doubted.
In the *Click = uncurated, rest = curated* mode, **Mark low-confidence as
uncurated** pre-clicks exactly the outlined tiles; click any other tile that
looks wrong, and **Done** curates everything else in one go. With the
Curation section in frame-by-frame review, a double-click drops straight into
that boundary instead: `Enter` moves the event onto the right frame,
`Backspace` deletes one that never happened, `N` marks it curated (with
**Click N curates current** ticked).

Judge a cutoff by what it buys: on a session with curated labels, "reviewing
everything below *t* catches what share of the errors?" is the question the
confidence exists to answer, and it is a better guide than how the histogram
looks. What the number means for each model, and how to change the rule it is
computed by, is on the {doc}`confidence page <confidence>`.
