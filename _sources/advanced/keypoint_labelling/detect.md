(target-detect)=
# 3. Detect — optional, if your animals wear tags

A click and a tag detector produce the same kind of thing: a position read off
the pixels of one specific frame. Ethograph calls both **observations**, and the
fill interpolates between them — so **any detector composes with any fill
backend**. Five clicked frames becoming several hundred detected ones is what
stops optical flow drifting and gives PosePAL far more to fit to.

Through trial and error I settled on **AprilTag `tag36h11`**, which works very
well on moving 5mm and 10mm wide tags.[^apriltag] Two smaller families are
offered alongside it, and Ethograph **prints exactly what it can read** — a
sheet of tags the detector cannot decode would be a trap.

```{figure} ../../_static/media/apriltags.png
:alt: Two example tags each from the tag16h5, tag25h9 and tag36h11 AprilTag families
:width: 100%

Three AprilTag families. The first number is how many data bits a tag carries
(a 4×4, 5×5 or 6×6 grid of modules); the number after the `h` is the smallest
Hamming distance between any two valid codes — how many bits have to be misread
before one ID turns into another *valid* ID.
```

| Family | Hamming | Grid | Data bits | Unique tags | Modules of paper |
|---|---|---|---|---|---|
| **`tag36h11`** (default) | 11 | 6×6 | 36 | 587 | 8 |
| `tag25h9` | 9 | 5×5 | 25 | 35 | 7 |
| `tag16h5` | 5 | 4×4 | 16 | 30 | 6 |

If your animal is very small and your pixel resolution limited, it may be helpful to take a family with a smaller grid `tag16h5`.
On the other hand, if those are not limiting factor, then `tag36h11` will give you an orders of magnitude lower false-positive rate and
a few more distinguishable IDs.

## Print the tags first

The sheet is made on the **start page**, under **🛠 Pre-recording tools ▸ Print
tag sheet…** — by the time you are tuning a detector the tags are already on
the animals. It produces a print-ready vector PDF at a size given in millimetres.

A sheet is a table of **rows**, so one page can mix sizes and families — a
handful of big tags and many small ones is the usual rig.

| Family | First ID | Count | Tag mm | Min mm |
|---|---|---|---|---|
| `tag36h11` | 0 | 24 | 4.0 | 4.2 |
| `tag16h5` | 0 | 6 | 4.0 | 3.1 |

- **Tag mm** is the printed tag itself, black border included — *not* the white
  margin around it, which is set for you.
- **Min mm** is a readout, not something you fill in. Give it your **Camera**
  resolution (seeded from the loaded video, and editable — the sheet is often for
  a rig that has not recorded yet) and the **Scene width in view**: how wide the
  frame is in millimetres at the animal's distance. A tag under its minimum is flagged in red.

:::{admonition} Printing — all four of these decide whether it works
:class: important

- **IMPORTANT: Do not crop the white border.** It is not decoration. The detector finds a
  tag by its black border against a *lighter surround*, so the printed white
  margin is what lets the same tag work on dark fur or paint as well as on pale
  skin. **Cut in the white, never on the black edge.**
- **Print at Actual size / 100%** — never "fit to printable area" or "shrink
  oversized pages". Every sheet carries a 50 mm rule: measure it with calipers,
  and if it does not read 50 mm, reprint rather than adjusting the size to
  compensate.
- **Matte paper, laser, black cartridge only**, toner-save and edge smoothing
  off. Gloss reflects the light source across the border, inkjet wicks, and
  composite black fringes every module edge.
- **Glue to card** so the tag stays flat. A curled tag is not a quad, and the
  detector has nothing to unwarp.
:::

If a small size is important for your rig, measure it rather than trusting the readout:
print one ID at 2.0, 2.5, 3.0, 4.0 and 5.0 mm, photograph the sheet under the
rig's own lighting, and adopt the smallest size that detects on every frame plus
one step of margin.


## Tuning

**Family** must match what you printed — e.g. a `tag16h5` tag will never decode as
`tag36h11`.

**Downscale** (`quad_decimate`) is how far the frame is shrunk before tags are
looked for — the real speed-against-size trade. `2.0` runs several times faster
and needs tags **twice as big**, which is why Ethograph defaults it to `1.0`
while the underlying library ships `2.0`.

**Sharpening** (`decode_sharpening`, default `0.25`) is applied to the sampled
bit pattern before it is read. Raise it for motion-blurred or slightly
out-of-focus tags.

**Detect the four corners too** emits each corner as its own keypoint, on top of
the tag's centre. It is for when you want the corner *positions* themselves —
you do **not** need it for head direction. Orientation comes free with every
decode, because a tag is a square: see
{doc}`head direction <export>`.

**Quality threshold**, on the Run panel, is the decode margin as a fraction of a
good read. A real tag scores about `1.0` and spurious reads on noise score under
`0.15`, so the default of `0.3` sits in the gap between them. It is applied as
results are stored, so retuning it costs nothing within a session.

## Running it

Choose a range and press **Run detector**. The result joins your labels as
observations; a detection never overwrites a label, and correcting one is just
clicking it. Runs are cached next to the video as `<video>.detections.npz`, which
is safe to delete.

## Notes

[^apriltag]: Detection runs through [pupil-apriltags](https://github.com/pupil-labs/apriltags), a maintained binding to the reference AprilTag 3 C library {cite:p}`krogius2019apriltag`. OpenCV renders the printed tags, since `pupil-apriltags` has no generator.
