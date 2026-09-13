# 5. Fill

Pick a backend, press **Fill frames between labels**, and every frame from your
first observation to your last gets a position. Your labelled frames always come
back exactly as you placed them — a fill never overwrites a label or a
detection, and never feeds on the previous fill. Frames outside that span are left empty.



(target-labelled-span)=
## Everything happens between your labels

The span between your outermost **observations** — labels, plus anything a
detector found — is the boundary for the whole workflow, not just for the fill:

- **Filling** runs in the gaps *between* labelled frames and nowhere else.

This is what makes partial labelling well-defined. If only part of a recording
interests you — one bout, one trial, the minute the animal is actually in frame
— label its start and its end and work inside it; the rest of the video stays
`NaN` rather than being padded with extrapolated positions you would then have
to detect and discard. The trade is the obvious one: you get results exactly
where you put labels, so a stretch you never bracketed produces nothing at all,
however long you spend filling.

## Choosing a fill backend

| Backend | How it works | Needs | Cost |
|---|---|---|---|
| **Spline** (default) | Monotone cubic (PCHIP) interpolation per keypoint over its own labelled frames[^pchip] — geometry only, the pixels are never read | nothing extra | instant |
| **Optical flow** | Pyramidal Lucas–Kanade, tracked forwards and backwards across each gap[^lk] | nothing extra | ~video speed, CPU |
| **PosePAL (CoTracker3 + refinement)** | A transformer point tracker[^cotracker] whose per-keypoint appearance features are first fitted to *your* labels on *this* video {cite:p}`pan2025posepal`, then tracked forwards and backwards across each gap | torch + cotracker, **GPU** | a few minutes to fit, reused by every fill made from the same labels |

Only PosePAL needs installing; see {ref}`PosePAL install <target-keypoint-fill>`.

**Without a GPU**, start with the spline — it costs nothing and is hard to beat when motion is smooth and your labels are dense — and switch to optical flow when the path between two labels isn't a smooth curve, like a fast turn or a wingbeat.

**With a GPU**, use PosePAL {cite:p}`pan2025posepal`. It handles longer gaps and larger
displacements than either of the others, and because it learns what *your*
keypoints look like in *this* recording, it stays on the right leg and the right
animal.

## Fit and track with PosePAL

PosePAL fills in two phases, and it is worth knowing which one you are paying
for:

- **Fit** — optimise CoTracker3's per-keypoint appearance features against the
  frames you labelled. Minutes on a GPU. It depends on nothing but your labels
  and the video.
- **Track** — run the fitted tracker forwards and backwards across each gap.
  This is what produces the filled frames, and it is fast.

**Fill** always does both, but it skips the fit while the fit it already has was
made from exactly the labels you have now — held in memory and cached beside the
video as `<video>.posepal.pt`. Correct a point, approve a frame, edit the schema,
and that no longer holds: the next **Fill** refits by itself before tracking.
There is no fit button, because there is no decision to make; the Fill and export tab simply
says which phases the next fill will pay for, so a three-minute wait is never a
surprise. Cancelling a fill leaves your labels, your current fill and the current
fit exactly as they were.

So there is no "carry on from the old fit" anywhere: **a refit is a fresh fit**,
started from scratch on all your labels — the reference implementation
re-optimises on every press for the same reason. The only thing the cache buys
you is not repeating a fit that would come out the same, which is what keeps
{ref}`the correction loop <target-correction-loop>` quick.

## Notes

[^pchip]: {cite:t}`fritsch1980pchip`. Implemented by [`scipy.interpolate.PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

[^lk]: {cite:t}`lucas1981lk`. The pyramidal form used here is {cite:t}`bouguet2001pyramidal`, via [`cv2.calcOpticalFlowPyrLK`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323).

[^cotracker]: {cite:t}`karaev2024cotracker3`. [Project page](https://cotracker3.github.io/) · [GitHub](https://github.com/facebookresearch/co-tracker)
