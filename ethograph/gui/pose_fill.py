"""Fill backends: turn a handful of labelled frames into every frame.

One protocol, two implementations, chosen in the labelling dialog:

- :class:`SplineBackend` — monotone cubic interpolation, no new dependencies.
  Ignores pixels entirely and is the yardstick the other must beat.
- :class:`OpticalFlowBackend` — Lucas-Kanade forward/backward (``opencv-contrib-python-headless``).

Neither needs a GPU, which is what makes the dialog usable on a laptop.

Both share the same invariant, asserted by the tests: **anchor frames come
back exactly as they were labelled.** Pixel-based backends seed missing points
from a spline pre-pass, so partially labelled anchors (beak on some frames, tail
on others) work without a shared frame list.

Backends track independent *points* and know nothing about the individual /
keypoint hierarchy above them: :meth:`~ethograph.gui.pose_annotate.KeypointStore.flat_anchors`
flattens one row per ``(individual, keypoint)`` pair before the fill, and
``set_fill_from_flat`` restores the shape afterwards. Multi-individual labelling
therefore needs nothing from this module beyond more rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator

#: ``progress(fraction) -> keep_going``; backends bail out when it returns False.
Progress = Callable[[float], bool]

#: Frames over which spline confidence decays to 1/e away from an anchor.
CONFIDENCE_DECAY_FRAMES = 10.0

#: Pixels of forward/backward disagreement that costs a factor 1/e of confidence.
DISAGREEMENT_SCALE = 10.0


@runtime_checkable
class FillBackend(Protocol):
    """Fills every frame from a sparse set of labelled ones."""

    name: str
    requires_video: bool

    def fill(
        self,
        anchors: dict[int, np.ndarray],
        n_frames: int,
        frames: object | None,
        progress: Progress,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions, confidence)`` for every frame of the video.

        Only the gaps *between* labels are filled: frames before the first
        labelled one and after the last come back as ``NaN`` (see
        :func:`anchor_span`). ``anchors`` maps frame index to an
        ``(n_points, 2)`` array of ``(x, y)`` with ``NaN`` for unlabelled
        points. ``frames`` is a frame source indexable by frame index and slice
        (see :class:`VideoFrameSource`); it is ``None`` for backends with
        ``requires_video = False``.
        """


def no_progress(_fraction: float) -> bool:
    return True


def _n_points(anchors: dict[int, np.ndarray]) -> int:
    return len(next(iter(anchors.values())))


def _apply_anchors(filled: np.ndarray, confidence: np.ndarray, anchors: dict[int, np.ndarray]) -> None:
    """Copy anchors through verbatim — the invariant every backend must hold."""
    n_frames = filled.shape[0]
    for frame, points in anchors.items():
        if not 0 <= frame < n_frames:
            continue
        labelled = ~np.isnan(points[:, 0])
        filled[frame][labelled] = points[labelled]
        confidence[frame][labelled] = 1.0


def _empty(n_frames: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((n_frames, n_points, 2), np.nan, dtype=np.float64),
        np.zeros((n_frames, n_points), dtype=np.float64),
    )


def anchor_span(anchors: dict[int, np.ndarray], n_frames: int) -> tuple[int, int] | None:
    """First and last labelled frame, clipped to the video; ``None`` if unlabelled.

    This is the span a fill covers. Outside it there is no second label to
    interpolate towards and nothing to track between, so whatever a backend
    produced there would be an extrapolation of a single endpoint — asserted
    with the same confidence as a genuinely bracketed frame, which is what made
    it worth suppressing. A user who labels frames 100 to 500 of a 1000-frame
    video gets exactly those 401 frames.
    """
    labelled = sorted(
        frame for frame, points in anchors.items() if 0 <= frame < n_frames and np.isfinite(points[:, 0]).any()
    )
    return (labelled[0], labelled[-1]) if labelled else None


def _restrict_to_span(filled: np.ndarray, confidence: np.ndarray, span: tuple[int, int] | None) -> None:
    """Blank every frame outside *span* — no position, and no score either."""
    n_frames = filled.shape[0]
    first, last = span if span is not None else (n_frames, n_frames - 1)
    for array in (filled, confidence):
        array[:first] = np.nan
        array[last + 1 :] = np.nan


# ----------------------------------------------------------------------
# Spline
# ----------------------------------------------------------------------


class SplineBackend:
    """Per-point monotone cubic (PCHIP) interpolation over its own anchors.

    Only the labelled span is filled (see :func:`anchor_span`); a cubic run past
    its data produces confident nonsense. Within that span a point labelled on
    only some of the frames holds its nearest value rather than extrapolating,
    so a keypoint labelled once is still available to the gap backends as a seed.
    Confidence decays exponentially with distance from the nearest anchor.
    """

    name = "Spline"
    requires_video = False

    def __init__(self, decay_frames: float = CONFIDENCE_DECAY_FRAMES):
        self._decay = float(decay_frames)

    def fill(self, anchors, n_frames, frames=None, progress: Progress = no_progress):
        if not anchors:
            raise ValueError("No labelled frames — label at least one frame before filling.")
        n_points = _n_points(anchors)
        filled, confidence = _empty(n_frames, n_points)
        grid = np.arange(n_frames, dtype=np.float64)

        for k in range(n_points):
            if not progress(k / n_points):
                break
            labelled = sorted(f for f, points in anchors.items() if not np.isnan(points[k, 0]))
            labelled = [f for f in labelled if 0 <= f < n_frames]
            if not labelled:
                continue
            xy = np.array([anchors[f][k] for f in labelled], dtype=np.float64)
            if len(labelled) == 1:
                filled[:, k, :] = xy[0]
            else:
                knots = np.asarray(labelled, dtype=np.float64)
                clamped = np.clip(grid, knots[0], knots[-1])
                filled[:, k, 0] = PchipInterpolator(knots, xy[:, 0])(clamped)
                filled[:, k, 1] = PchipInterpolator(knots, xy[:, 1])(clamped)
            distance = np.min(np.abs(grid[:, None] - np.asarray(labelled)[None, :]), axis=1)
            confidence[:, k] = np.exp(-distance / self._decay)

        _restrict_to_span(filled, confidence, anchor_span(anchors, n_frames))
        _apply_anchors(filled, confidence, anchors)
        return filled, confidence


# ----------------------------------------------------------------------
# Shared gap machinery for the pixel-based backends
# ----------------------------------------------------------------------


def _seeded_endpoints(anchors: dict[int, np.ndarray], seed: np.ndarray, frame: int) -> np.ndarray:
    """Anchor row at *frame*, with unlabelled points taken from the spline seed."""
    points = seed[frame].copy()
    labelled = ~np.isnan(anchors[frame][:, 0])
    points[labelled] = anchors[frame][labelled]
    return points


def _blend(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    """Linear crossfade from the left track to the right one across a gap."""
    weight = np.linspace(0.0, 1.0, forward.shape[0])[:, None, None]
    return forward * (1.0 - weight) + backward * weight


class _GapBackend:
    """Base for backends that track each gap between consecutive anchor frames.

    A spline pre-pass provides positions for points that are missing on a gap's
    endpoints. Nothing outside the labelled span is produced — there is no gap
    there to track across (see :func:`anchor_span`).

    *disagreement_px* is the confidence scale: pixels of forward/backward
    disagreement that cost a factor 1/e. It is a constructor argument because
    the right value depends on the footage — the same 10 px is a fifth of a
    small animal on one recording and a rounding error on a 4K one.
    """

    name = "gap"
    requires_video = True

    def __init__(self, disagreement_px: float = DISAGREEMENT_SCALE):
        self.disagreement_px = disagreement_px

    @property
    def disagreement_px(self) -> float:
        return self._disagreement

    @disagreement_px.setter
    def disagreement_px(self, value: float) -> None:
        # Settable because a backend can outlive the spin box that set it: the
        # refined backend is kept across fills so its fit is not repaid.
        if value <= 0:
            raise ValueError("disagreement_px must be positive — it is the scale of an exponential.")
        self._disagreement = float(value)

    def fill(self, anchors, n_frames, frames=None, progress: Progress = no_progress):
        if not anchors:
            raise ValueError("No labelled frames — label at least one frame before filling.")
        if frames is None:
            raise ValueError(f"The {self.name} backend needs video frames.")

        seed, seed_confidence = SplineBackend().fill(anchors, n_frames, None, no_progress)
        filled, confidence = seed.copy(), seed_confidence.copy()

        anchor_frames = [f for f in sorted(anchors) if 0 <= f < n_frames]
        gaps = list(zip(anchor_frames, anchor_frames[1:]))
        for done, (start, end) in enumerate(gaps):
            if not progress(done / len(gaps)):
                break
            if end - start < 2:
                continue
            clip = np.asarray(frames[start : end + 1])
            scale = float(getattr(frames, "scale", 1.0))
            left = _seeded_endpoints(anchors, seed, start) / scale
            right = _seeded_endpoints(anchors, seed, end) / scale
            # A point labelled nowhere in the video has no spline seed either, so
            # its endpoints stay NaN. Such a row must never reach the tracker:
            # a tracker that attends jointly across points answers ONE NaN query
            # with NaN for every point in the gap — blanking the whole span
            # while the untracked head and tail keep their seed. Track what is
            # seeded and leave the rest at the seed.
            trackable = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
            if not trackable.any():
                continue

            forward, visible_forward = self._track(clip, left[trackable], 0)
            backward, visible_backward = self._track(clip, right[trackable], end - start)
            forward, backward = forward * scale, backward * scale

            filled[start : end + 1, trackable] = _blend(forward, backward)
            disagreement = np.linalg.norm(forward - backward, axis=-1)
            confidence[start : end + 1, trackable] = np.minimum(visible_forward, visible_backward) * np.exp(
                -disagreement / self._disagreement
            )

        _apply_anchors(filled, confidence, anchors)
        return filled, confidence

    def _track(self, clip: np.ndarray, points: np.ndarray, query_frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Track *points* (given at *query_frame*) across every frame of *clip*.

        Returns ``(positions (T, N, 2), visibility (T, N))`` in clip pixels.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# Optical flow
# ----------------------------------------------------------------------


class OpticalFlowBackend(_GapBackend):
    """Lucas-Kanade pyramidal flow, tracked forward and backward per gap.

    Real-time on CPU and a useful fallback where torch cannot be installed.
    Requires ``opencv-contrib-python-headless`` — plain ``opencv-python`` ships
    Qt plugins that conflict with PyQt6.
    """

    name = "Optical flow"
    requires_video = True

    def __init__(self, window: int = 21, levels: int = 3, disagreement_px: float = DISAGREEMENT_SCALE):
        super().__init__(disagreement_px)
        self._window = int(window)
        self._levels = int(levels)

    def _track(self, clip, points, query_frame):
        import cv2

        n = len(points)
        positions = np.full((len(clip), n, 2), np.nan, dtype=np.float32)
        visibility = np.zeros((len(clip), n), dtype=np.float64)
        positions[query_frame] = points
        visibility[query_frame] = 1.0

        gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in clip]
        params = dict(
            winSize=(self._window, self._window),
            maxLevel=self._levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        for direction in (1, -1):
            current = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            alive = np.ones(n, dtype=bool)
            index = query_frame
            while 0 <= index + direction < len(clip):
                nxt = index + direction
                moved, status, _ = cv2.calcOpticalFlowPyrLK(gray[index], gray[nxt], current, None, **params)
                status = status.reshape(-1).astype(bool) & ~np.any(np.isnan(moved.reshape(-1, 2)), axis=1)
                alive &= status
                current = np.where(status[:, None, None], moved, current)
                positions[nxt] = current.reshape(-1, 2)
                visibility[nxt] = alive.astype(np.float64)
                index = nxt
        return positions.astype(np.float64), visibility


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------


@dataclass
class BackendInfo:
    key: str
    label: str
    available: bool
    hint: str = ""


def _module_available(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> list[BackendInfo]:
    """Describe every backend so the dialog can grey out the missing ones."""
    return [
        BackendInfo("spline", "Spline (no extra dependencies)", True),
        BackendInfo(
            "flow",
            "Optical flow (OpenCV)",
            _module_available("cv2"),
            "pip install opencv-contrib-python-headless",
        ),
    ]


def build_backend(key: str, disagreement_px: float = DISAGREEMENT_SCALE) -> FillBackend:
    """Instantiate a backend by key, importing heavy dependencies only now.

    ``disagreement_px`` tunes the confidence of the tracking backend only —
    the spline scores by distance from the nearest anchor instead.
    """
    if key == "spline":
        return SplineBackend()
    if key == "flow":
        return OpticalFlowBackend(disagreement_px=disagreement_px)
    raise ValueError(f"Unknown fill backend {key!r}")


# ----------------------------------------------------------------------
# Frame source
# ----------------------------------------------------------------------


def video_size(path: str | Path) -> tuple[int, int] | None:
    """``(width, height)`` in the video's **own** pixels, or ``None``.

    Read from the stream header, so it costs an open and decodes nothing.

    It is deliberately the *file's* size rather than anything on screen: what
    is displayed may be a low-resolution proxy (see
    :mod:`~ethograph.io.video_proxy`) or a downscaled decode, and a number
    taken from there is wrong by the proxy's own scale factor without
    announcing it. Returns ``None`` when the file cannot be read, since every
    caller is asking in order to *offer* a value.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            return int(stream.codec_context.width), int(stream.codec_context.height)
    except (OSError, ValueError, IndexError, StopIteration):
        return None


class VideoFrameSource:
    """Lazily decoded RGB frames, indexable by frame index and slice.

    Decoding the whole video into memory is not viable for the videos this GUI
    targets, so frames are decoded on demand with PyAV. Access is assumed to be
    broadly forward (gaps are visited in order); a backward request re-seeks.

    ``max_side`` downscales during decode — the single biggest CPU speedup for
    the tracking backends, and near-free in accuracy at this anchor density.
    :attr:`scale` converts decoded pixels back to source pixels.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        n_frames: int,
        max_side: int | None = None,
        start_frame: int = 0,
    ):
        import av

        if fps <= 0:
            raise ValueError("fps must be positive — read it from the video, do not default it.")
        self._path = str(path)
        self._fps = float(fps)
        self._n_frames = int(n_frames)
        #: Video frame that index 0 of this source maps to — lets callers work
        #: in trial frames while decoding stays in video frames.
        self._start_frame = int(start_frame)
        self._container = av.open(self._path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

        width, height = self._stream.codec_context.width, self._stream.codec_context.height
        longest = max(width, height)
        if max_side and longest > max_side:
            self.scale = longest / float(max_side)
            self._size = (
                max(2, round(width / self.scale / 2) * 2),
                max(2, round(height / self.scale / 2) * 2),
            )
        else:
            self.scale = 1.0
            self._size = (width, height)

    def __len__(self) -> int:
        return self._n_frames

    @property
    def video_frames(self) -> int:
        """Frames in the whole video file, from the stream header (0 if unknown)."""
        return int(self._stream.frames or 0)

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` frames are **decoded** at, after ``max_side``.

        Callers that care about pixels rather than positions need this: a tag
        decoder's whole signal is resolution, and a memory budget is counted in
        decoded pixels, not source ones.
        """
        return self._size

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._n_frames)
            if step != 1:
                raise ValueError("VideoFrameSource only supports contiguous slices.")
            return self._decode(start, stop)
        return self._decode(int(key), int(key) + 1)[0]

    def _decode(self, start: int, stop: int) -> np.ndarray:
        start, stop = start + self._start_frame, stop + self._start_frame
        decoded: dict[int, np.ndarray] = {}
        self._seek(start)
        for frame in self._container.decode(self._stream):
            index = self._frame_index(frame)
            if index >= stop:
                break
            if index >= start:
                decoded[index] = frame.reformat(width=self._size[0], height=self._size[1], format="rgb24").to_ndarray()
        if not decoded:
            raise ValueError(f"No frames decoded for [{start}, {stop}) in {self._path}")

        # Timestamp rounding can skip an index; hold the previous frame so the
        # returned clip always has one entry per requested frame.
        out, previous = [], decoded[min(decoded)]
        for index in range(start, stop):
            previous = decoded.get(index, previous)
            out.append(previous)
        return np.stack(out)

    def iter_frames(self, indices, gray: bool = False, progress=None):
        """Yield ``(index, frame)`` for *indices* in one forward pass — no seek per frame.

        Random access costs a keyframe seek plus a partial-GOP decode per
        frame, which is what made scanning thousands of candidate frames slow.
        Here the container is sought once to the first wanted frame and decoded
        forward, yielding only the wanted indices, the way the video motion
        pass in :mod:`ethograph.features.movement` streams. ``gray`` reformats
        to one channel during decode (``(H, W)`` uint8); otherwise frames are
        ``(H, W, 3)`` RGB. Indices are delivered in ascending order; *progress*
        gets the fraction delivered and may return ``False`` to stop early.
        """
        wanted = sorted({int(i) for i in indices if 0 <= int(i) < self._n_frames})
        if not wanted:
            return
        remaining = set(wanted)
        last = wanted[-1]
        total = len(wanted)
        done = 0
        fmt = "gray" if gray else "rgb24"
        self._seek(wanted[0] + self._start_frame)
        previous_index: int | None = None
        for frame in self._container.decode(self._stream):
            index = self._frame_index(frame) - self._start_frame
            if index > last:
                break
            # Timestamp rounding can skip an index: a wanted index that fell in
            # a gap is served by the frame that follows it, as ``_decode`` does.
            first = index if previous_index is None else previous_index + 1
            hits = [i for i in range(min(first, index), index + 1) if i in remaining]
            previous_index = index
            if not hits:
                continue
            image = frame.reformat(width=self._size[0], height=self._size[1], format=fmt).to_ndarray()
            for i in hits:
                remaining.discard(i)
                done += 1
                yield i, image
                if progress is not None and not progress(done / total):
                    return

    def _seek(self, frame_index: int) -> None:
        target = int(frame_index / self._fps / float(self._stream.time_base))
        offset = int(self._stream.start_time or 0)
        self._container.seek(max(0, target + offset), stream=self._stream, backward=True)

    def _frame_index(self, frame) -> int:
        pts = frame.pts if frame.pts is not None else 0
        pts -= int(self._stream.start_time or 0)
        return int(round(float(pts * self._stream.time_base) * self._fps))

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> VideoFrameSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
