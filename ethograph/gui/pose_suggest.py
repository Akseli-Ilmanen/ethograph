"""Which frames to label: pick diverse or high-motion frames, not neighbours.

Labelling frames 100, 101, 102… is close to wasted effort — consecutive frames
are near-identical, so they teach a tracker almost nothing new. Both established
labelling GUIs solve this the same way, and this module follows them:

- **DeepLabCut** (``frameselectiontools.KmeansbasedFrameselection``) downsamples
  the video to ~30 px wide, treats each frame as a vector, mean-centres, runs
  MiniBatchKMeans with ``n_clusters = numframes2pick``, and takes **one frame per
  cluster** — so the chosen frames "look different, i.e. different postures".
- **SLEAP** (``FeatureSuggestionPipeline``) does image features (raw / HOG /
  BRISK bag-of-features) → PCA → k-means → sample per cluster, and separately
  offers motion-driven methods (``velocity``, ``max_point_displacement``) that
  threshold per-frame displacement to find frames worth proofreading.

Three methods are offered here:

``uniform``
    Evenly spaced. The honest baseline — for a short clip of one behaviour it is
    hard to beat, and it needs no decoding.
``diverse``
    DeepLabCut's k-means on downscaled grayscale frames: one frame per cluster,
    so distinct postures are covered rather than whatever the animal did most.
``motion``
    Frames with the largest change from the previous frame — where the action
    is. With a trace from ``extract_packet_motion`` (bytes per compressed frame)
    no decoding is needed; otherwise thumbnails are differenced. If the clip
    holds fewer *distinct* moving moments
    than you asked for, the remaining slots are filled with the next
    best-scoring frames, so the requested count is always returned; the highest-
    motion moments are simply taken first.
``uncertain``
    Frames whose *worst* point the last fill was least sure about. Available only
    after a fill, and the method that actually matches a tracker-based workflow —
    see below.
``detection_gaps``
    Frames furthest from any *detection*. Available once a detector has run
    (:mod:`ethograph.gui.pose_detect`), and the natural partner to it: a marker
    detector either sees the marker or it does not, so the frames it missed —
    occlusion, blur, the animal facing away — are exactly the ones worth
    labelling by hand. Ranking by distance from the nearest detection puts the
    middle of the longest blind stretch first.

Matching the method to the backend
----------------------------------
The four methods are **not interchangeable**, and the fill backend decides which
one helps:

===============  ==========================================  ====================
Fill backend     Fails when                                  Suggest with
=================  ========================================  ====================
``spline``         the trajectory turns sharply between      ``motion`` (a proxy
                   anchors — it never looks at pixels        for curvature), or
                                                             ``uniform``
``flow``           displacement exceeds Lucas-Kanade's       ``motion``, then
                   pyramid capture range; occlusion          ``uncertain``
``posepal``        occlusion, target leaves frame            ``uncertain``, then
                                                             ``diverse``
any, after Detect  the marker is occluded, blurred, or       ``detection_gaps``
                   facing away
=================  ========================================  ====================

``diverse`` suits neither ``spline`` nor ``flow``, which is worth stating plainly
because it is DeepLabCut's default and the obvious thing to copy. Its premise is
that labels are *training data*, so the model needs varied appearance to
generalise from. Those two are frozen or purely geometric — neither learns from
your labels — so a visually distinct frame where tracking already succeeded buys
nothing. It earns its place for ``posepal``
(:mod:`ethograph.gui.pose_refine`), the one backend that *is* fitted to the
labels: there the appearance embedding does generalise across the video, and
frames covering distinct poses and lighting are what it needs.

Why ``uncertain`` is the right criterion for tracker fill
---------------------------------------------------------
DeepLabCut and SLEAP select frames to *train* a pose model, so redundancy is the
enemy and visual diversity is the goal. No detector is trained here: CoTracker3
takes queries of ``(t, x, y)`` — **one query frame per point** — and propagates
them, so extra labelled frames mostly serve to reset accumulated drift (PosePAL
additionally fits the query embedding to them, which is why ``diverse`` earns a
place there). The frames worth labelling are therefore the ones where tracking
*fails* (occlusion, motion blur, the animal leaving frame), which is not the
same set as the visually diverse ones. ``uncertain`` ranks by the fill's own
confidence — forward/backward disagreement and visibility — closing the label → fill →
correct-the-worst → fill loop. It is the analogue of SLEAP's ``prediction_score``.

One thing neither GUI enforces, and which matters for the stated goal: SLEAP's
velocity method returns *every* frame over the threshold, so a single fast bout
can supply a run of consecutive frames. Every method here passes through
:func:`enforce_min_gap`, which keeps the best-scoring frame in any neighbourhood
and drops the rest — so a burst of motion contributes one frame, not thirty.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

#: Frames decoded to score a video. Above this the candidate grid is strided —
#: clustering 2000 thumbnails is plenty to characterise any clip.
MAX_CANDIDATES = 2000

#: Longest side of the thumbnails used for scoring. DeepLabCut uses ~30 px wide;
#: a little more detail costs nothing at this frame count.
FEATURE_MAX_SIDE = 64

#: Fraction of the mean suggestion spacing enforced as a minimum gap. At 1/4,
#: asking for 20 frames over 2000 keeps suggestions at least 25 frames apart.
MIN_GAP_FRACTION = 0.25

METHODS = ("uniform", "diverse", "motion", "mixed", "uncertain", "detection_gaps")
#: ``mixed``: a share of the picks are the strongest movements (motion ranking,
#: min-gap applied), the rest one k-means pick per cluster over the candidates
#: whose motion is above the gate quantile and that are not within the gap of a
#: motion pick. Motion says what is worth looking at, k-means that the picks
#: do not all show one event; the share sets how many strong events are kept
#: on purpose rather than left to the clustering.
MOTION_GATE_QUANTILE = 0.5
MOTION_SHARE = 0.3


def motion_area(motion: np.ndarray, window_frames: int) -> np.ndarray:
    """Motion summed over a centred window — the area under the curve around each frame.

    A brief spike scores little; movement that lasts scores in proportion to how
    long it lasts, which is what 'a moment worth labelling' means.
    """
    motion = np.asarray(motion, dtype=np.float32)
    window = max(1, int(window_frames)) | 1  # odd, so the window is centred on the frame
    if window == 1 or len(motion) == 0:
        return motion
    return np.convolve(motion, np.ones(window, dtype=np.float32), mode="same")


def default_min_gap(n_frames: int, count: int) -> int:
    """Minimum spacing between suggestions, derived from the request.

    Scales with the video and the number of frames asked for rather than being
    a fixed number of frames or seconds — a fixed gap would be meaningless
    across clips of different length and frame rate.
    """
    if count <= 0:
        return 1
    return max(1, int(n_frames / count * MIN_GAP_FRACTION))


def enforce_min_gap(frames: Sequence[int], min_gap: int, count: int) -> list[int]:
    """Greedily keep frames in priority order, dropping any within *min_gap*.

    *frames* must already be ordered best-first; the result is sorted by frame
    index. This is what stops one burst of motion from filling the whole budget.
    """
    kept: list[int] = []
    for frame in frames:
        if len(kept) >= count:
            break
        if all(abs(frame - k) >= min_gap for k in kept):
            kept.append(frame)
    return sorted(kept)


def _candidate_indices(n_frames: int, exclude: set[int]) -> np.ndarray:
    """Frame indices to score, strided so long videos stay tractable."""
    step = max(1, int(np.ceil(n_frames / MAX_CANDIDATES)))
    candidates = np.arange(0, n_frames, step)
    if exclude:
        candidates = np.array([c for c in candidates if int(c) not in exclude], dtype=int)
    return candidates


def _thumbnails(frames, indices: np.ndarray, progress: Callable[[float], bool] | None) -> np.ndarray:
    """``(n, d)`` mean-centred grayscale vectors, one row per candidate frame.

    Grayscale by averaging channels — DeepLabCut's choice, and colour rarely
    distinguishes postures.
    """
    rows: list[np.ndarray] = []
    if hasattr(frames, "iter_frames"):
        # One sequential decode for every candidate. A seek per frame is what
        # made this scan take minutes on a long recording; *indices* is sorted,
        # so an early stop leaves rows for its prefix, which the caller trims to.
        for _index, image in frames.iter_frames(indices, gray=True, progress=progress):
            rows.append(np.asarray(image, dtype=np.float32).reshape(-1))
    else:
        total = max(len(indices), 1)
        for position, index in enumerate(indices):
            if progress is not None and not progress(position / total):
                break
            image = np.asarray(frames[int(index)], dtype=np.float32)
            if image.ndim == 3:
                image = image.mean(axis=2)
            rows.append(image.reshape(-1))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    data = np.asarray(rows, dtype=np.float32)
    return data - data.mean(axis=0)


def suggest_uniform(count: int, n_frames: int, exclude: set[int] | None = None) -> list[int]:
    """Evenly spaced frames — no decoding, no dependencies."""
    exclude = exclude or set()
    available = [f for f in range(n_frames) if f not in exclude]
    if not available or count <= 0:
        return []
    if count >= len(available):
        return available
    picks = np.linspace(0, len(available) - 1, count).round().astype(int)
    return sorted({available[p] for p in picks})


def _suggest_diverse(features: np.ndarray, indices: np.ndarray, count: int) -> list[int]:
    """One frame per k-means cluster (DeepLabCut's strategy).

    ``n_clusters = count`` so each cluster contributes exactly one frame; the
    frame nearest its centroid is chosen rather than a random member, which
    makes the result deterministic and picks the most representative posture.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = min(count, len(indices))
    if n_clusters < 2:
        return [int(indices[0])] if len(indices) else []

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=0)
    labels = kmeans.fit_predict(features)

    picks: list[int] = []
    for cluster in range(n_clusters):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            continue
        distances = np.linalg.norm(features[members] - kmeans.cluster_centers_[cluster], axis=1)
        picks.append(int(indices[members[int(np.argmin(distances))]]))
    return sorted(set(picks))


def _motion_scores(features: np.ndarray) -> np.ndarray:
    """Mean absolute change from the previous candidate frame."""
    if len(features) < 2:
        return np.zeros(len(features), dtype=np.float32)
    difference = np.abs(np.diff(features, axis=0)).mean(axis=1)
    # Align to the later frame of each pair: motion is attributed to where the
    # animal has moved TO, and the first candidate gets the first score.
    return np.concatenate([difference[:1], difference])


def frame_confidence(confidence: np.ndarray) -> np.ndarray:
    """Reduce a fill's confidence array to one score per frame — its worst point.

    The minimum over the trailing (point) axes, skipping ``NaN``. A frame is
    only as good as the keypoint the tracker lost, and that one point is the
    entire reason to go back to the frame; averaging hid it, letting nine
    well-tracked points pull a frame with one lost point into the middle of the
    ranking. Points a schema leaves out are ``NaN`` rather than zero (``set_fill``
    blanks them), so they are skipped instead of pinning every frame of that
    individual to the same score.

    A frame the fill did not cover at all scores ``NaN``, computed without
    ``nanmin``'s empty-slice warning since a whole video's worth is normal.
    """
    array = np.asarray(confidence, dtype=np.float64)
    if array.ndim == 1:
        return array
    flat = array.reshape(len(array), -1)
    scored = ~np.isnan(flat)
    return np.where(scored.any(axis=1), np.where(scored, flat, np.inf).min(axis=1), np.nan)


def suggest_uncertain(
    confidence: np.ndarray,
    count: int,
    exclude: set[int] | None = None,
    min_gap: int | None = None,
) -> list[int]:
    """Frames the fill was least confident about, worst first.

    A frame is scored by its worst point (:func:`frame_confidence`), so a single
    keypoint the tracker lost is enough to bring the frame to the front — which
    is exactly the correction the next fill needs.

    Only frames the fill actually covered are offered. A fill spans the labelled
    frames and nothing beyond them, so a ``NaN`` score means the frame lies
    outside that span: there is no prediction there to doubt, and nothing to
    correct. This method points at the *gaps between* labels — where the fill ran
    and did badly — not at the unlabelled tail, which would otherwise dominate
    the ranking simply by being empty.
    """
    exclude = set(exclude or ())
    scores = frame_confidence(confidence)
    n_frames = len(scores)
    if count <= 0 or not n_frames:
        return []
    covered = np.flatnonzero(~np.isnan(scores))
    order = covered[np.argsort(scores[covered], kind="stable")]
    ranked = [int(f) for f in order if int(f) not in exclude]
    gap = default_min_gap(n_frames, count) if min_gap is None else int(min_gap)
    return enforce_min_gap(ranked, gap, count)


def suggest_detection_gaps(
    detected: Sequence[int],
    count: int,
    n_frames: int,
    exclude: set[int] | None = None,
    min_gap: int | None = None,
) -> list[int]:
    """Frames furthest from any detection — where the detector went blind.

    The complement of a detector run, and the reason detection composes with
    hand labelling rather than replacing it: a marker detector is not uncertain,
    it is *absent*, so its failures are a set of frames rather than a low score.
    Ranking by distance from the nearest detection puts the middle of the
    longest blind stretch first, which is the frame a fill has least to go on.

    With no detections at all every frame is equally blind, so this falls back
    to even spacing rather than returning the video in index order.
    """
    exclude = set(exclude or ())
    if count <= 0 or n_frames <= 0:
        return []
    detected = np.asarray(sorted({int(f) for f in detected if 0 <= int(f) < n_frames}), dtype=np.int64)
    if not len(detected):
        return suggest_uniform(count, n_frames, exclude)
    grid = np.arange(n_frames)
    distance = np.min(np.abs(grid[:, None] - detected[None, :]), axis=1)
    # Stable sort on the negated distance: ties (the two sides of a gap) keep
    # frame order, so the result does not depend on numpy's sort internals.
    ranked = [int(f) for f in np.argsort(-distance, kind="stable") if distance[f] > 0 and int(f) not in exclude]
    gap = default_min_gap(n_frames, count) if min_gap is None else int(min_gap)
    return enforce_min_gap(ranked, gap, count)


def suggest_frames(
    method: str,
    count: int,
    n_frames: int,
    frames=None,
    exclude: set[int] | None = None,
    min_gap: int | None = None,
    progress: Callable[[float], bool] | None = None,
    confidence: np.ndarray | None = None,
    detected: Sequence[int] | None = None,
    motion: np.ndarray | None = None,
    motion_window: int = 1,
    motion_share: float = MOTION_SHARE,
    motion_gate: float = MOTION_GATE_QUANTILE,
) -> list[int]:
    """Suggest *count* frames to label.

    Parameters
    ----------
    method
        One of :data:`METHODS`. ``diverse`` and ``motion`` need *frames*.
    frames
        Frame source indexable by frame index (see
        :class:`~ethograph.gui.pose_fill.VideoFrameSource`). Open it with a
        small ``max_side`` — only thumbnails are needed.
    exclude
        Frames already labelled; never suggested again (SLEAP's
        ``filter_unique_suggestions``).
    min_gap
        Minimum spacing; defaults to :func:`default_min_gap`.
    detected
        Frames a detector found a marker on; ``detection_gaps`` needs it.
    motion
        A per-frame motion trace over all *n_frames* (``extract_packet_motion``).
        When given, ``motion`` and ``mixed``
        score candidates by the area under it over *motion_window* frames —
        sustained movement, never a one-frame spike — and ``motion`` then
        needs no decoding at all.
    motion_window
        Frames the area is summed over (:func:`motion_area`).
    motion_share
        ``mixed`` only: fraction of *count* taken from the top of the motion
        ranking; the rest is filled by k-means.
    motion_gate
        ``mixed`` only: candidates below this motion quantile are not clustered.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown suggestion method {method!r}; expected one of {METHODS}")
    exclude = set(exclude or ())
    if count <= 0 or n_frames <= 0:
        return []
    if method == "uniform":
        return suggest_uniform(count, n_frames, exclude)
    if method == "uncertain":
        if confidence is None:
            raise ValueError("The 'uncertain' method needs a fill to have run first.")
        return suggest_uncertain(confidence, count, exclude, min_gap)
    if method == "detection_gaps":
        if detected is None:
            raise ValueError("The 'detection_gaps' method needs a detector to have run first.")
        return suggest_detection_gaps(detected, count, n_frames, exclude, min_gap)
    indices = _candidate_indices(n_frames, exclude)
    if not len(indices):
        return []
    gap = default_min_gap(n_frames, count) if min_gap is None else int(min_gap)

    trace = None
    if motion is not None:
        motion = np.asarray(motion, dtype=np.float32)
        if len(motion) < n_frames:
            raise ValueError(f"motion trace has {len(motion)} frames, the video {n_frames}")
        trace = motion_area(motion[:n_frames], motion_window)
        if method == "motion":
            scores = trace[indices]
            ranked = [int(indices[i]) for i in np.argsort(scores)[::-1]]
            return enforce_min_gap(ranked, gap, count)

    if frames is None:
        raise ValueError(f"The {method!r} method needs video frames.")
    features = _thumbnails(frames, indices, progress)
    if not len(features):
        return []
    indices = indices[: len(features)]

    if method == "diverse":
        # Cluster picks are already spread across posture space; the gap only
        # breaks ties between near-identical neighbours.
        return enforce_min_gap(_suggest_diverse(features, indices, count), gap, count)

    scores = trace[indices] if trace is not None else _motion_scores(features)
    ranked = [int(indices[i]) for i in np.argsort(scores)[::-1]]
    if method == "motion":
        return enforce_min_gap(ranked, gap, count)

    # mixed: strong events first, then diverse postures among the moving rest
    if not 0.0 <= motion_share <= 1.0:
        raise ValueError(f"motion_share must be within [0, 1], got {motion_share}")
    n_motion = int(round(count * motion_share))
    strong = enforce_min_gap(ranked, gap, n_motion) if n_motion else []
    keep = np.flatnonzero(scores >= np.quantile(scores, motion_gate))
    if strong:
        near = np.array([min(abs(int(i) - s) for s in strong) < gap for i in indices[keep]])
        keep = keep[~near]
    if len(keep) < 2:
        keep = np.arange(len(scores))
    diverse = _suggest_diverse(features[keep], indices[keep], count - len(strong))
    return sorted(set(strong) | set(enforce_min_gap(diverse, gap, count - len(strong))))
