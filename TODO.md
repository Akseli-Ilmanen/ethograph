

0) IMPORTANT: go through references.md and make sure all are correct.


1) Create small gifs (low resolution).For github readme and maybe other place holders, as short demo.
2) Discussion with Heberto. You can load in VAME/others? predictions as nwb, pynapple extracts intervalset, and load those as predictions. You don't get confidence,
and they have to save as nwb.
3) DONE (2026-09-16, ADR 0013): a session is a folder; a video-only drop or wizard run gets only `.ethograph/alignment.nwb` and individuals live in that record (with `project.yaml` as the study default). Was: Video-only sessions (user picks just a video folder) still get a synthesised .nc file. The GUI reads it only to work out the multi-animal situation (individual dim / names); it carries no feature data otherwise. Think this through: either drop the file and read individuals from somewhere else (alignment NWB, a settings entry), or make the synthesised dataset carry something useful (per-video trial, fps, ROI/motion traces later). Related: opening a folder of videos with no dataset at all (notes/feral_notes.md §1).
4) Boris IO, the following may be helpful
https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/behavior/boris.html
https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/behavior/moseq_keypoints.html
5) DONE (2026-09-10, `io/nc_drop.py`): every dropped `.nc` is a feature source, several concatenate on a `camera` dim, and a movement `.nc` is also drawn on its video only when `positions_fit_frame`. History: a movement `.nc` (poses/bboxes) took one of two separate paths on the cover page, decided once by `classify_files`: session bucket → features in the sidebar/plots, never the overlay; pose bucket → `pose_cam-N` in the alignment, drawn as points/boxes, but no features (the session is the empty alignment NWB unless something else was dropped). Only the standalone case (poses, no video) stacks the files into a features `.nc`. Unify: one dropped movement dataset should feed both — overlay on its camera and `position`/`speed`/`confidence` per camera as features — rather than a bucket picking one. Plan: (a) every dropped `.nc` is accessible as features — several `.nc` files dropped together are concatenated into one session dataset on a new `camera` dim (named after the paired video / `cam-N`, `xr.concat(..., join="outer")` so files with different individuals/keypoints still fit; `_build_pose_features_nc` already does this for the standalone case) and show up in the add-panel popup; (b) a file is *also* a pose overlay for its camera when it is plausibly in that video's pixel coordinates: `ds_type` poses/bboxes with `position`, and the finite x/y range falls inside the paired video's frame size (from `probe_video`) — an `.nc` whose positions are in mm/normalised/negative is features only, never drawn on the video.
6) user manual in trial vs sesison explain how slide scope works trial, session vs fiexed windows approach OR SHOULD i put this in bring your own data -> Trials?
7) update video in branches.md in docs
8) variable schema, formalize for comparisons, and adjust docs and mentions of it in some files in examples/
9) Comparing runs in the GUI, after cross_validate model output, find easy way to show these different runs in the GUI to see which params lead to which outcomes.
10) ### Miscellaneous
If your `NWB` files contains pose or spike times, you can visualize them for free in the GUI.
- **Pose.** neuroconv's `DeepLabCutInterface` (and the SLEAP and LightningPose
  ones) write ndx-pose into the same file. Give it the video as `source_video`
  and EthoGraph pairs the pose to that camera. Or leave the pose file beside the
  `.nwb` and pair it with {func}`~ethograph.pair_media`, as in step 6; both
  end up on the same overlay.
- **Spike sorting.** `KiloSortSortingInterface` writes the units, which gives
  you the raster. The {ref}`Phy-like viewer <target-ephys-viewers>` reads
  waveforms straight off the raw binary and still needs the Kilosort folder
  selected in the GUI.
-> VERIFY THIS WORKS
11) in docs/segment we mention that we take dataset structure where mapping.txt is single class per frame, put for multi label, mult individual this breaks down. Inestigate how claude solved this.
12) project.yaml: move the remote label backup (path, mode, depth) from gui_settings.yaml to the project, since a backup target is per study; paths in project.yaml must be checked on load like every other path. Also still open from ADR 0013: rename `.ethograph/alignment.nwb` to `session.nwb` (52 files, read the old name as a fallback).
