# 0016 — aniposelib does the geometry; movement reads the files; a second view is a second store

## Context

Sessions filmed with two or more cameras had no way to 3D inside Ethograph:
users ran DeepLabCut 3D outside it and loaded the result, and a hand-tuned
`Position3DCalibration` (`features/movement.py`) turned one dataset's DeepLabCut
frame into room coordinates. `aniposelib` is now a dependency. Three things
were wanted: triangulating existing 2D pose files, seeing hand labels from two
camera views as a 3D point while labelling, and turning triangulated landmarks
into a Space plot geometry.

## Decision

- **Every calibration, triangulation and reprojection is aniposelib's.**
  `triangulate/points.py` is the one place `CameraGroup.triangulate*` is called;
  batch runs, the dialog's live point and the geometry export all go through it.
  Nothing in Ethograph reimplements multi-view geometry.
- **Arrays in, not files.** aniposelib's native interface is the
  `(cameras, points, 2)` array. Its `load_pose2d_fnames` reads DeepLabCut `.h5`
  only, so it is not used: 2D points are read the way the video overlay reads
  them (a `camera`-dim variable, else pose files through movement), which covers
  every tracker movement reads. A round trip through DeepLabCut files to reach
  that loader was considered and dropped — it writes temporary files to
  reproduce an array already in hand, and the live paths have no files at all.
- **The calibration is the project's**: `{project}/calibration/{name}.toml`,
  aniposelib's own TOML. Never per session (a re-calibrated rig is a second
  file, named by the dataset's `calibration_name`), never in a source `.nwb`.
- **A DeepLabCut 3D calibration is converted, not re-estimated.** The stereo
  pickle holds everything a `Camera` takes; the import copies it and keeps
  DeepLabCut's world frame (camera 1, rectified), so output matches existing
  `_DLC_3D.h5` files. Re-calibrating from the kept board images was the
  alternative: pure aniposelib, but different numbers and it needs the square
  size. New rigs calibrate with aniposelib directly; there is no wrapper.
- **The world frame is ours**, because aniposelib has none (Anipose's lives in
  the `anipose` pipeline package). It is a small YAML beside the calibration in
  Anipose's `config.toml` spelling plus a scale, resolved once per rig and
  written back, so every session shares it. `Position3DCalibration` stays for
  the Moll25 example only.
- **A second camera view is a second store.** The label model keeps no camera
  axis: the second view has its own `KeypointStore`, sidecar and label mode, and
  the pair is made by the trial clock and a shared schema
  (`gui/pose_two_view.py`, `gui/pose_second_view.py`). Fill, detection and
  correction are untouched.
- **No epipolar guide.** aniposelib has no function for it; the feedback is the
  triangulated point projected back (`CameraGroup.project`) and its error.

## Consequences

- `position_3d` shares `space` with any 2D variable in the same dataset, so a
  dataset with `space = [x, y]` is widened to `[x, y, z]` (2D variables carry
  `NaN` z). The overlay selects x and y by name and is unaffected.
- The fill is still one camera at a time; a fill is kept in memory across a
  camera swap (`_fill_stash`) because sidecars never hold one.
- DeepLabCut 3D's pair limit carries over to imported calibrations.

## Would reopen this

- aniposelib gaining a general pose-file loader (then use it).
- A need for true multi-view labelling aids (epipolar lines, a reprojected
  guess before the second click) — that changes the labelling design rules and
  needs geometry aniposelib does not offer.
