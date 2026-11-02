(target-detect)=
# 3. Detect — optional, if your animals wear tags

If your animals carry printed AprilTags, a detector can read the tag position
off every frame on its own. A detection counts exactly like a click, so the
fill interpolates between both. Skip this page if there are no tags.

With **one tag per animal** you do not need to label a single frame: define
the individuals and one keypoint, press **Run detector**, and the tags found in
the video are paired with your individuals in order.

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

Detections appear as hollow markers with a dot. A detection never overwrites
your own label.[^apriltag]

## Which tag is which animal

The table **What the detector's labels mean** says which tag ID lands on
which individual and keypoint. It is filled in one of two ways, and any row
can be changed by hand; a row you edit is never overwritten.

- **Find tags in video** (also run for you by **Run detector** when the table
  is empty) scans a few dozen frames and pairs the tags it sees with your
  individuals in order: the lowest tag ID becomes the first individual. A
  tag's parts go to that individual's keypoints in order, so a single keypoint
  gets the tag's centre, and a keypoint named `corner_0` gets that corner.
  With one tag per animal this is all you need.
- **Learn from labels** is for when order says nothing — several tags on one
  animal, or a tag on a keypoint whose name does not say which. Label a few
  frames first; each decoded tag is matched to the nearest labelled point, and
  a row is proposed once two frames agree.

[^apriltag]: Detection runs through [pupil-apriltags](https://github.com/pupil-labs/apriltags), a binding to the AprilTag 3 library.
