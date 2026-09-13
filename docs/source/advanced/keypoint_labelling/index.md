# Keypoint labelling

**Tools ▸ Pose tracking (from scratch)…** (or **Label keypoints…** in the Pose sidebar) opens a dialog where you
label a few frames by clicking the video and let a point tracker fill in the
rest. No training, no annotated dataset, no GPU for the default backend.

```{important}
**This is a tracker, not a pose detector.** It follows points you have already
placed; nothing in it *finds* the animal in a video it has not been pointed at.
Everything on these pages — including the model choice — follows from that. See
{ref}`target-tracking-vs-training` for when this is the right tool and when
DeepLabCut or SLEAP is.

The one exception is {ref}`target-detect`, for animals wearing printed tags: a
tag *can* be found on a frame nobody has labelled. It produces the same kind of
thing a click does, so it slots in ahead of the tracker rather than replacing it.
```

(target-one-video)=
## One video at a time

The dialog works on **a single camera and a single trial** — one continuous
video, which is what a {doc}`drag & drop <../../getting_started/your_data/index>` of a video file gives
you. It always follows the primary camera view's current video: labels are keyed
by frame index on that video's own frame grid, and they are saved to a sidecar
next to it (`<video>.keypoints.json`).

There is no trial or camera axis in the model, so a **multi-trial dataset
(`TrialTree`) is not supported**, and neither is labelling a second camera view
of the same scene. Nothing stops you from opening the dialog with such a dataset
loaded — but you are labelling whichever single video the primary view is
showing, and switching to another video starts from that video's own sidecar.
Multi-trial, multi-camera pose data is something you *import* (from NWB or from
DeepLabCut/SLEAP output); see {doc}`export` for going the other way.

(target-tracking-vs-training)=
## Tracking vs. training: which tool for which problem

Established pose tools (DeepLabCut, SLEAP) **train a detector**: you label
several hundred frames, train a network, and it then finds the keypoints in
videos it has never seen. The cost is paid once and amortised over the whole
project — provided your videos are similar enough that one model covers them.

EthoGraph's keypoint labelling **tracks instead of trains**. You label a handful
of frames in *one* video and the tracker propagates them through *that* video.
Nothing is learned that transfers to the next recording.

| | Point tracking (this dialog) | Detector training (DLC / SLEAP) |
|---|---|---|
| Frames to label | roughly every 10th, per video | several hundred, once |
| Setup cost | none — works immediately | training run, tuning, GPU time |
| Generalises to new videos | **no** | yes, within its domain |
| Best when | few videos, or every video looks different | many videos that look alike |
| Fails when | you have 500 similar videos | your videos are all different |

**This is the tool for heterogeneous footage.** If every recording has a
different animal, camera angle, background or lighting, a trained detector
generalises worst exactly where your data varies most. Tracking never had to
generalise — it is only ever asked about the video in front of it {cite:p}`pan2025posepal`.

## In the classroom

Ethographs keypoint labelling mode can also be used in an animal behaviour class. Students meet in groups, use
their phones to film a short video of one or multiple animals doing a behaviour. Using their laptops (no GPU required), they
can label a a few keypoints and get started with pose-estimation within an hour.

Once a few frames are labelled and the rest filled in, the video can be loaded
into the GUI with velocity, speed and acceleration ticked. The traces sit on the
same time axis as everything else in the recording, so students can read them
against the video, the audio waveform or the spectrogram and check they match
what the animal was actually doing. {doc}`../../examples/create_dataset_cricket`
does exactly this, putting leg kinematics next to the sound they produce.

## The workflow

The dialog has one tab per stage (Fill and export share one); these pages
follow the same order, with correction and refining imported poses at the end.

```{toctree}
:maxdepth: 1

labelling
detect
calibration
fill
correction
export
refine_imported
```

