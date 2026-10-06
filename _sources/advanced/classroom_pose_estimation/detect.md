(target-detect)=
# 3. Detect — optional, if your animals wear tags

If your animals carry printed AprilTags, a detector can read the tag position
off every frame on its own. A detection counts exactly like a click, so the
fill interpolates between both. Skip this page if there are no tags.

## Print the tags

On the start page, open **🛠 More tools ▸ Print tag sheet…**, keep the default
family `tag36h11`, set the tag size in millimetres and print the PDF.

```{figure} ../../_static/media/apriltags.png
:alt: Two example tags each from the tag16h5, tag25h9 and tag36h11 AprilTag families
:width: 100%

AprilTag families. `tag36h11` is the default; `tag16h5` has a coarser grid for
very small animals or low-resolution video.
```

:::{admonition} Printing — all four of these matter
:class: important

- **Do not cut off the white border.** Cut in the white, never on the black edge.
- **Print at 100 % / actual size.** Each sheet has a 50 mm rule: measure it.
- **Matte paper on a laser printer**, black cartridge only.
- **Glue the tag to card** so it stays flat.
:::

## Run the detector

1. **Family** must match what you printed.
2. Leave **Downscale** at `1.0`; raise **Sharpening** if the video is blurry.
3. Choose a frame range and press **Run detector**.

Detections appear as hollow markers with a dot. The first time, Ethograph asks
which tag ID is which keypoint and individual; correct any wrong guess in the
table. A detection never overwrites your own label.[^apriltag]

[^apriltag]: Detection runs through [pupil-apriltags](https://github.com/pupil-labs/apriltags), a binding to the AprilTag 3 library {cite:p}`krogius2019apriltag`.
