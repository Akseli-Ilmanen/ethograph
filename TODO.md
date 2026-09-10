

1) Create small gifs (low resolution).For github readme and maybe other place holders, as short demo.
2) Discussion with Heberto. You can load in VAME/others? predictions as nwb, pynapple extracts intervalset, and load those as predictions. You don't get confidence,
and they have to save as nwb.
3) Video-only sessions (user picks just a video folder) still get a synthesised .nc file. The GUI reads it only to work out the multi-animal situation (individual dim / names); it carries no feature data otherwise. Think this through: either drop the file and read individuals from somewhere else (alignment NWB, a settings entry), or make the synthesised dataset carry something useful (per-video trial, fps, ROI/motion traces later). Related: opening a folder of videos with no dataset at all (notes/feral_notes.md §1).
4) Boris IO, the following may be helpful  
5) A movement `.nc` (poses/bboxes) takes one of two separate paths on the cover page, decided once by `classify_files`: session bucket → features in the sidebar/plots, never the overlay; pose bucket → `pose_cam-N` in the alignment, drawn as points/boxes, but no features (the session is the empty alignment NWB unless something else was dropped). Only the standalone case (poses, no video) stacks the files into a features `.nc`. Unify: one dropped movement dataset should feed both — overlay on its camera and `position`/`speed`/`confidence` per camera as features — rather than a bucket picking one.
https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/behavior/boris.html
https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/behavior/moseq_keypoints.html