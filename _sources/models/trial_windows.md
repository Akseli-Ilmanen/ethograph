(target-trial-windows)=
# Trial windows

Every model in EthoGraph learns from and predicts over **trials**. A trial is
one window of the session: the stretch of time you navigate to, curate, and
that a model reads as one input. How to store and align trials is covered in
{doc}`../getting_started/your_data/trials` and
{doc}`../getting_started/your_data/media_alignment`; this page is about
**where to draw the windows** so they suit a model.

## Natural trials

If your behaviour already comes in trials — a pellet is dispensed, a stimulus
plays, an animal enters an arena — or your recording system is **triggered**,
writing one file per trial, the trial is already there. Carve your trials out
this way in `alignment.nwb` ({ref}`target-nwb-alignment`).

::::{grid} 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/neuroconv/video_setup_free_running.png
:alt: A free-running camera, as one file or split into several

Free-running camera. Figure from neuroconv's how-to (BSD-3-Clause).
```
:::

:::{grid-item}
```{figure} ../_static/neuroconv/video_setup_triggered.png
:alt: A triggered camera, one file per trial

Triggered camera. Figure from neuroconv's how-to (BSD-3-Clause).
```
:::
::::

Passing these trials to a model as its samples is a natural fit, both for
{doc}`action segmentation <segment/index>`, where a sample is one
(trial, individual) pair, and for {doc}`event spotting <spot/index>`, which
shares the same sessions, trial filter and train/val/test split drawn by
whole trial. (E2E-Spot then draws its clips of `context_s` inside each trial;
see {ref}`target-spot-seconds`.)

Both read a trial as **its window of the video**, not the video file: a trial
carved out of a longer video gives the model only the frames between its start
and stop, and a video file that already matches its trial is read whole.

A free-running system that writes **several files at fixed intervals** (the
third row of the free-running figure, `part_01.mp4, part_02.mp4, …`) but has
no defined trials may also already give you a natural split: one file, one
trial.

## One long recording

A single recording lasting an hour, or a whole day, is not a good sample. It
may not fit in memory for the model, and with one sample per recording the
gradient updates come rarely. Divide the recording into trial windows instead,
and each window is passed to the model on its own.

The right length depends on the model, but we recommend windows of around
**1024, 2048 or 4096 frames**. In seconds this differs a lot between datasets:
4096 frames is over two minutes of a slow behaviour filmed at 30 fps, but only
about 20 s of a crow's tool use filmed at 200 fps. Build the windows with
{ref}`from_continuous() <target-from-continuous>` or as the trials table of
the alignment.

## Why non-overlapping windows

Tools such as DLC2Action split a recording into much shorter segments (e.g.
256 frames; {cite:alp}`kozlova2025dlc2action`, Figure S5) that **overlap**. The clear
advantage is that a behaviour cut at the edge of one segment is seen whole in
the next.

We take a simpler approach: **larger windows with no overlap**, following the
convention of the temporal action segmentation literature
{cite:p}`ding2023tasreview` (see also
[awesome-temporal-action-segmentation](https://github.com/nus-cvml/awesome-temporal-action-segmentation)).
This

- keeps new architectures easy to plug in, since they expect exactly this
  input;
- avoids majority voting or averaging over overlapping frames;
- makes the context a model sees the same as the trial a user curates.

The cost is that an action can be bisected by a window boundary. Don't make
trial windows too small (e.g. 256 frames): the shorter the window, the more
often an action is cut in two.
