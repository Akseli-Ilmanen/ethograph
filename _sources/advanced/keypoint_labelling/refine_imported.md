(target-refine-imported)=
# Refining imported poses

**Tools ▸ Pose correction (DLC, SLEAP, …)…** corrects a pose file another tool produced —
DeepLabCut, SLEAP, LightningPose — on the video, and writes the result back
**in the source format** as `{stem}_refined{ext}` beside the original
(overwritten on every save — a refined file only gets more refined). Where the
keypoint labelling dialog starts from nothing on a single video, this one
starts from your files and follows the loaded session: **multi-trial**, with
each trial resolving its pose file through the alignment exactly as the pose
overlay does.

The dialog **is** the labelling dialog with the Define keypoints and Detect
tabs removed: the whole {doc}`Label & Edit tab <labelling>` — modes,
the points table with its funnel filters, frame suggestions, `Shift+H`
approval, `Tab`/`1`–`9`/`Backspace`/`Ctrl+Z` — works unchanged. The file's
keypoints and individuals *are* the schema; its points appear as machine
observations (hollow with a pip, `Detected` in the table's Source column),
your clicks are labels (solid), and clicking a file point pins your correction
over it exactly as in the correction loop.

## Cameras

Every **open camera view** (primary or extra) is a live editing context with
its own file, its own labels and its own `_refined` copy — all held in memory
for the whole trial. The **Camera** combo at the top only picks which one the
canvas edits: switching discards nothing, so labels can never be lost to a
camera switch, and the context line names the camera being edited plus any
others carrying unsaved edits. The points table shows the selected camera's
file; clicks land on that camera's view. (While an *extra* view is being
edited, its file points are drawn by both its pose overlay and the editing
overlay — a known cosmetic double.)

## Fill

**Fill runs for every open camera**, each from its own labels and its own
file — the same stretch is usually wrong on both views. The scope choice:

- **my labels only** — interpolate between *your* clicks alone; inside the
  filled span the file's points are replaced by the fill (for stretches where
  the file is plain wrong), outside they are kept. A camera without clicks of
  yours is skipped — there is nothing to interpolate between.
- **my labels + the file's points** — the file's points are trusted
  observations too; the fill only bridges the gaps between them and your
  clicks, and never replaces a file point.

## Saving

There is no export step — the last tab is **Fill and save** — and **Save always means every edited camera**. A
camera's `_refined` copy is created the moment it is first edited, and
rewritten on every trial switch, on **Save refined now**, and on close — an
untouched camera writes nothing, so the output folder records exactly what
was reviewed. A `.slp` source refines to an analysis `.h5` (movement has no
SLEAP project writer). Editing one trial of a session-wide pose file rewrites
only that trial's window; the rest of the file round-trips verbatim.

Your own clicks are additionally kept in `{pose file}.refine.json` beside the
source — written **before** every refined-file write, so a failed write can
never take the labels with it. Reopening loads the `_refined` copy when one
exists plus the sidecar, so the work resumes with your points still marked as
yours.
