(target-refine-project)=
# Refining a DeepLabCut / LightningPose project

**Start page ▸ Refine DLC / LightningPose training data…** opens a pose
project *folder* — the one with `videos/` and `labeled-data/` — and turns it
into two views of the same training set. Where {doc}`refine_imported`
corrects a pose file inside a session, this works on the project the way the
pose tool itself lays it out, and writes back only what the tool reads:
PNGs under `labeled-data/<video>/` and the labels table beside them.

```
my_project/
├── config.yaml                  DeepLabCut: scorer, bodyparts, individuals
├── CollectedData.csv            LightningPose: one labels table for every video
├── labeled-data/
│   └── <video>/                 the training frames of one video
│       ├── img00012.png
│       └── CollectedData_<scorer>.csv / .h5      DeepLabCut: one table per video
└── videos/
    ├── <video>.mp4
    └── <video>DLC_<model>.h5    the model's predictions on the whole video
```

The folder must hold both `videos/` and `labeled-data/`; anything else is
refused with the missing name. A DeepLabCut config fixes the **scorer** (the
table has to be `CollectedData_<scorer>` for `create_training_dataset` to
find it); a project without one — LightningPose — asks who is labelling, and
remembers the name.

The top bar is reduced to the two stages, **Docs** and **Help**. Everything
about sessions and models is gone because there is no session here: each
stage is its own folder under `<project>/.ethograph/`, with its own
alignment, labels and layout, so navigation, the trials table and the
green/red curated colouring work exactly as in any session.

## Extract Frames

One trial per video in `videos/`, named after the file. A video with
predictions beside it (`<video>DLC_*.h5`, or `video_preds/<video>.csv`)
brings them in as features — `position`, `confidence`, `velocity`, `speed` —
and opens with one keypoint's position (x and y) and speed as line plots
and every keypoint's confidence as a heatmap. A video without predictions is still a trial —
its panels are simply empty.

The point is to choose frames off those curves rather than by scrubbing.
The Labels section holds two classes:

| Key | Class | What it marks |
|---|---|---|
| `1` | **ExtractSegment** (state) | a stretch worth sampling from: two clicks on a plot |
| `2` | **ExtractFrame** (point) | the frame on screen, taken as it is |

Below them, how a segment is sampled — **Uniform** (evenly spaced) or
**Diverse** (one frame per k-means cluster of thumbnails, DeepLabCut's
method) — and the **coverage**, the share of a segment's frames to take.
Point events are always taken.

**Extract frames from this video** writes them as `img<frame>.png` into
`labeled-data/<video>/`, numbered by their index on the video so a second
extraction from the same video adds to the folder, and adds a row per frame
to the labels table **prefilled with the model's prediction** — the frame is
reviewed, not labelled from nothing. Rows already in the table are kept.

## Refine Pose

One trial per folder in `labeled-data/`, the folder standing in for a video:
its images in natural order, one frame per image, on an image-sequence clock
(one image per second, so the time axis reads as the image index). No
features are shown — the selection was made in the other stage, and every
frame here is to be reviewed.

The Labels section is the {doc}`Label & Edit tab <labelling>` of the
keypoint labelling dialog with Sequential already armed: every row of the
table is a solid, draggable label; `Tab` cycles keypoints, `1`–`9` pick the
individual, `Backspace` deletes, `Ctrl+Z` undoes. There is no fill, no
detect and no export tab — the table *is* the export.

**Static keypoints** sit in the panel between the mode buttons and the
table. Tick a
landmark that never moves (a table corner, a fixed marker); it turns bold
and pinned in the list. Label every static keypoint once on any frame, then
**Propagate static landmarks from this frame to all frames** copies their
positions on the frame on screen to every frame of the folder, so only the
moving keypoints are left to place elsewhere. Static keypoints are
remembered per project: the next folder starts with them in place.

### Saving

Edits go back to `CollectedData_<scorer>.csv` and its `.h5` twin (or the
root `CollectedData.csv` of a LightningPose project) a few seconds after the
last change, when you move to another folder, on **Save labels now**, when
you switch stage and when the window closes — never per drag, since the
`.h5` rewrite is not free and a half-written table is worse than a stale one.
The line under the button says whether the table holds unsaved edits.

### Checking a folder

**Check labels** writes every frame of the folder with its labels drawn on
— one colour per keypoint, the config's skeleton and dot size — into
`labeled-data/<video>_labeled/`, the folder DeepLabCut's own `check_labels`
produces, so the reviewed poses can be looked over in any image viewer. A
`_labeled` folder is never a trial.

### Curation

A folder is a trial, and a trial is curated when every frame of it has been
reviewed. **Going to the next folder marks this one curated** is on by
default; `Ctrl+C` curates by hand. The verdict lives in the stage's
`metadata.tsv` `curated` column and colours the trial green in the
navigation, exactly like a curated session trial.
