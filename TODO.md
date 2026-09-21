

0) IMPORTANT: go through references.md and make sure all are correct.


1) Create small gifs (low resolution).For github readme and maybe other place holders, as short demo.
2) Discussion with Heberto. You can load in VAME/others? predictions as nwb, pynapple extracts intervalset, and load those as predictions. You don't get confidence,
and they have to save as nwb.
4) Boris IO, the following may be helpful
https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/behavior/boris.html
https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/behavior/moseq_keypoints.html
7) update video in branches.md in docs
8) variable schema, formalize for comparisons, and adjust docs and mentions of it in some files in examples/
10) ### Miscellaneous
If your `NWB` files contains pose or spike times, you can visualize them for free in the GUI.
- **Pose.** neuroconv's `DeepLabCutInterface` (and the SLEAP and LightningPose
  ones) write ndx-pose into the same file. Give it the video as `source_video`
  and Ethograph pairs the pose to that camera. Or leave the pose file beside the
  `.nwb` and pair it with {func}`~ethograph.pair_media`, as in step 6; both
  end up on the same overlay.
- **Spike sorting.** `KiloSortSortingInterface` writes the units, which gives
  you the raster. The {ref}`Phy-like viewer <target-ephys-viewers>` reads
  waveforms straight off the raw binary and still needs the Kilosort folder
  selected in the GUI.
-> VERIFY THIS WORKS
11) in docs/segment we mention that we take dataset structure where mapping.txt is single class per frame, put for multi label, mult individual this breaks down. Inestigate how claude solved this.
12) project.yaml: move the remote label backup (path, mode, depth) from gui_settings.yaml to the project, since a backup target is per study; paths in project.yaml must be checked on load like every other path. Also still open from ADR 0013: rename `.ethograph/alignment.nwb` to `session.nwb` (52 files, read the old name as a fallback).
