# 1-2. Define keypoints and label frames

The first two tabs of the dialog: say what you are labelling, then label it.

## 1. Define keypoints

Keypoints are defined as one shared list of names instantiated by one or more individuals, shown as a tree with a branch per individual and a leaf per keypoint carrying a per-frame mark and a labelled/total count — unticking the shared-keypoints option lets each individual carry its own subset, and names outside that subset stay permanently empty.

**Colour by** picks what colour tells apart, as in SLEAP — and it is the same setting the pose overlay uses (Pose overlay ▸ Design ▸ Colour by), so the canvas and the overlay never disagree:

| Colour by | What it means | Use it |
|---|---|---|
| **Keypoint** (default) | one colour per body part, the same on every individual | labelling — a click answers "which body part is this?" |
| **Individual** | one colour per individual, shared by all its keypoints | telling two animals apart when they overlap |

Every marker is a circle either way; the individual you are labelling is drawn at full opacity, the others dimmed, and each carries its name at the centre of its points. Colours are auto-assigned from a spread palette, overridable via **Colour…** — which edits whichever palette is being drawn, the selected keypoint's or the selected individual's — applied consistently across overlay, tree and points table, resettable wholesale, and persisted in `<video>.keypoints.json`.


### Static keypoints

Tick **Static** beside a keypoint that does not move — the corners of the
arena, a fixed landmark. Label it **once**, on any frame, and it is there on
every frame: the canvas draws it everywhere as a label, the fill leaves it
alone, and the export writes it on every frame. Placing it again moves it
(everywhere); deleting it removes it (everywhere). It is saved in the sidecar,
and the next clip you open from the same camera starts with the static
keypoints already placed where they were in the last clip you saved — so with
many short clips of one rig you click the corners once, not once per clip.

## 2. Label & Edit

Arm **Sequential** or **Loop** and click the video:

- **Sequential** — label every keypoint on one frame. Each click places the
  active keypoint and advances to the next one this individual still lacks.
  It never navigates.
- **Loop** — sweep one keypoint across frames. Each click places it, then does
  whatever **Then go to** says: step one frame, jump to the next suggested
  frame, or stay put. The same dropdown says where `Shift+H` lands.

`Tab` cycles keypoints, `1`–`9` pick the individual, `Backspace` deletes the
active point, `Ctrl+Z` undoes, `Shift+H` approves this frame's predictions (see
{ref}`target-correction-loop`). Clicking an existing point always selects and
drags it — correcting never requires switching mode first.

The overlay shows **everything** — your labels, anything a detector found, and
the fill's predictions — so you can judge a prediction before accepting it: a
label is a **solid** marker, anything else is the same colour drawn
**hollow**, left empty so you can see the pixels underneath. A hollow marker
with a **dot in it** was read off *this* frame by a detector (see
{ref}`target-detect`); an empty one was interpolated between other frames.
Clicking a hollow marker pins it as a label and it turns solid; dragging one
corrects it first.

While a mode is armed, left-drag labels and panning moves to `Shift`+left-drag.
Tick **Lock** to look around instead: left-drag pans again and clicks no longer
place, move or pin anything. The labels stay on screen and the active keypoint
is kept, so unticking carries on where you were — unlike stopping the mode,
which takes the anchor overlay with it.

**Switching to any other tab locks the pointer on its own.** Defining keypoints,
detecting and filling are all things you do while *looking* at the video —
scrubbing to judge a detection, checking a fill — and a stray click there would
drop a point you never meant to place, silently. The mode itself keeps running,
so returning to this tab carries straight on. Your own **Lock** tick is left
exactly as you set it and applies again the moment you come back. The one
exception is {ref}`Calibrate <target-calibration>`, which takes the pointer for
its own landmark clicks — leaving it hands the canvas back to labelling, active
keypoint and all.

### Which frames to label

Labelling consecutive frames is close to wasted effort: neighbouring frames look
almost identical, so the second one tells the tracker nothing the first did not.
The **Which frames to label** group proposes a spread instead. Ask for a *share*
of the video — the default is **10%, roughly every 10th frame**, the density
CoTracker3 is evaluated at for this task (6 labelled frames of 60 {cite:p}`pan2025posepal`).

| Method | What it picks | Use it |
|---|---|---|
| **Evenly spaced** | equally spaced frames, no video scan | first pass; hard to beat on a short clip of one behaviour |
| **Biggest pixel change** | frames that differ most from their predecessor | fast motion, where a spline cuts the corner |
| **Most different frames** | k-means over frame thumbnails, one per cluster (DeepLabCut's method) | long, varied recordings |
| **Where the detector saw nothing** | frames furthest from any detection | **after Detect** — the marker was occluded, blurred or facing away |
| **Lowest fill confidence** | frames whose *worst keypoint* the last fill scored lowest, within the span it covered | **after a fill** — the correction loop |

`N` — or **Next suggested frame** — jumps to the next suggestion, wrapping at
the end; plain `←` / `→` still move one frame at a time. There is no key for the
*previous* suggestion: the list is a queue to work down, and clicking a row of
the points table seeks to any frame, suggested or not.

