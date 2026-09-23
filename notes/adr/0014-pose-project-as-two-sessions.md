# 0014 — A pose project is two sessions, an image folder is a video

## Context

Refining DeepLabCut / LightningPose training data used to mean **Tools ▸ Pose
refinement** over a loaded session: a pose file corrected on the video, written
back as `_refined`, training frames exported on request. It never touched the
tool's own project folder, and choosing *which* frames to label was left to the
tool's own extractor (k-means over the whole video) or to scrubbing by eye.

The project folder is the thing the user actually has: `videos/` with the
model's predictions beside each video, `labeled-data/<video>/` with the
training frames as PNGs and a labels table. The predictions carry exactly the
signal that says where labelling is worth it — confidence and speed — and the
GUI already plots those better than any pose tool does.

## Decision

1. **The project is opened as two ordinary sessions, not as a new data
   backend.** `<project>/.ethograph/extract_frames/` holds one trial per video
   with the predictions as features (`session.nc`); `<project>/.ethograph/
   refine_pose/` holds one trial per `labeled-data/` folder. Each has its own
   `alignment.nwb`, `labels.tsv`, `metadata.tsv` and `local_settings.yaml`.
   Nothing in the loader, the navigation or the curation colouring learns
   about pose projects; the mode (`gui/pose_project_mode.py`) only decides
   which of the two folders is loaded and which panel the Labels section shows.
   The alternative — one folder with two alignment files — would have broken
   *a session is a folder* for every reader of `session_layout.py`.

2. **A folder of images is a video.** `io/image_sequence.py` gives it a probe,
   a lazy frame source and a declared clock (`IMAGE_SEQUENCE_RATE`, one image
   per second, so time reads as image index — a definition of the sequence's
   clock, not a guessed recording rate). Every "is this media on disk" check
   goes through `media_exists`, never a bare `isfile`, and `CameraView` renders
   the folder through pynaviz's in-memory tensor plot with lazily read frames.
   The frame index of the refine stage is the image's position in the folder;
   the labels table keys rows by image name, and `ImageSequence.name_of` /
   `index_of` are the one bridge.

3. **Frame selection is labelling.** `ExtractSegment` (state) and
   `ExtractFrame` (point) are two classes in the extract session's
   `mapping.txt`; the marked stretches are sampled (uniform / k-means) at a
   coverage and written as PNGs plus prefilled table rows. Nothing new is
   drawn, undone or saved — the labels machinery already does all of it.

4. **The refine stage edits the tool's labels table in place**, the labels
   dialog's store loaded from it and written back to it — csv and h5 for
   DeepLabCut, the root csv for LightningPose — never a sidecar of our own.
   Writes are debounced (seconds after the last edit, on folder switch, on
   stage switch, on close), not per drag: the h5 rewrite costs, and a table
   caught mid-write is worse than one a few seconds stale.

5. **The scorer is the config's when the config names one**, because
   DeepLabCut's `create_training_dataset` looks for `CollectedData_<scorer>`;
   otherwise it is asked on entry and remembered. There is no default name.

6. **A folder's curated verdict is its own**, not derived from labels (a
   folder has none): `app_state.curation_status_override` feeds the same
   `trial_curation_status` every colouring reads, and the verdict lives in the
   stage's `metadata.tsv` `curated` column like any other.

## Consequences

- `Tools ▸ Pose refinement (DLC, SLEAP, …)` stays for pose files inside a
  session; the project mode is the path for a project folder.
- A project without predictions still opens: its videos are trials without
  curves, and extraction prefills nothing.
- `session.nc` of the extract stage is rebuilt only when a video or prediction
  file changed (`sources.json`), because Windows will not let the file the GUI
  holds open be rewritten, and re-reading every prediction file per stage
  switch is a cost the switch should not carry.
