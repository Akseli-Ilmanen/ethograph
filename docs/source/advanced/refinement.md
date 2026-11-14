(target-refinement)=
# Refinement

You end up here in one of two situations: you want a **better DeepLabCut or
LightningPose model**, or you have a pose file and want to **edit it**.

## A better model: label more frames

Often the best refinement is not editing predictions frame by frame but
training a better model on more labelled frames. For that we recommend the
[LightningPose labelling GUI](https://lightning-pose.readthedocs.io/en/latest/source/user_guide_app/index.html):
we find it very convenient to use, especially for multi-camera setups, where
it extracts the same frame from every view and keeps each view's labels
paired.

## Editing a pose file

Ethograph does not carry an editor of its own — use the movement GUI, which
does it well:

- the [movement napari GUI](https://movement.neuroinformatics.dev/latest/user_guide/gui.html)
  — load a pose file, inspect and move points on the video;
- the [edit widget](https://neuroinformatics.dev/blog/gsoc_2026_movement_edit_widget.html)
  built for it, with drag-to-correct and per-keypoint editing;
- [this fork](https://github.com/Akseli-Ilmanen/movement/tree/napari-interpolate-edit)
  adds interpolation and bulk editing across a stretch of frames.
