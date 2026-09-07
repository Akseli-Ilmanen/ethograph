"""Fill backends: turn a handful of labelled frames into every frame.

One protocol, three implementations, chosen in the labelling dialog:

- :class:`SplineBackend` — monotone cubic interpolation, no new dependencies.
  Ignores pixels entirely and is the yardstick the others must beat.
- :class:`OpticalFlowBackend` — Lucas-Kanade forward/backward (``opencv-contrib-python-headless``).
- ``PosePALBackend`` (:mod:`ethograph.gui.pose_refine`) — CoTracker3 point
  tracking with its query features fitted to the user's labels. GPU only, and
  imported lazily so nothing here depends on torch.

:class:`_CoTrackerTracking` holds the plain CoTracker3 gap tracking that PosePAL
builds on. It is **not** offered as a backend of its own: unrefined tracking
follows the appearance a point had on one frame and drifts onto the wrong leg or
the other animal, which the refinement exists to fix, so choosing between them
was a choice between a method and a worse version of the same method.

All of them share the same invariant, asserted by the tests: **anchor frames come
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

from ethograph.utils.device import resolve_device as _resolve_device

#: ``progress(fraction) -> keep_going``; backends bail out when it returns False.
Progress = Callable[[float], bool]

#: Frames over which spline confidence decays to 1/e away from an anchor.
CONFIDENCE_DECAY_FRAMES = 10.0

#: Pixels of forward/backward disagreement that costs a factor 1/e of confidence.
DISAGREEMENT_SCALE = 10.0

#: The learned backend: CoTracker3 plus the query-feature refinement of Pan et
#: al. 2025 (:mod:`ethograph.gui.pose_refine`). Named after the paper's reference
#: implementation, since the tracker alone is not something the user can pick.
POSEPAL_BACKEND = "posepal"
POSEPAL_LABEL = "PosePAL (CoTracker3 + refinement)"


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
            # CoTracker attends jointly across points, so ONE NaN query comes
            # back as NaN for every point in the gap — blanking the whole span
            # while the untracked head and tail keep their seed. Track what is
            # seeded and leave the rest at the seed.
            trackable = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
            if not trackable.any():
                continue

            self._on_rows(np.flatnonzero(trackable))
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

    def _on_rows(self, rows: np.ndarray) -> None:
        """Which flat point rows the next :meth:`_track` calls are about.

        Rows with no seed are dropped above, so the query list a tracker sees is
        *compressed* — query ``i`` is point ``rows[i]``, not point ``i``. Only
        matters to a backend holding per-row state (PosePAL's learned query
        features, one per point); tracking a point is otherwise independent of
        which point it is, so the default ignores it.
        """

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
# CoTracker3
# ----------------------------------------------------------------------


class _CoTrackerTracking(_GapBackend):
    """CoTracker3 point tracking, forward from the left anchor and backward
    from the right, blended linearly across the gap.

    The tracking half of ``PosePALBackend``, which is the only thing that
    instantiates it — see this module's docstring for why plain CoTracker3 is
    not a backend the user picks. Cost is dominated by frame feature extraction
    rather than point count, so tracking 20 keypoints costs about what 3 do.
    """

    name = "CoTracker3"
    requires_video = True

    def __init__(self, predictor, device: str | None = None, disagreement_px: float = DISAGREEMENT_SCALE):
        super().__init__(disagreement_px)
        self._predictor = predictor
        self._device = device or resolve_device()

    def _track(self, clip, points, query_frame):
        import torch

        video = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2).float()[None]
        video = video.to(self._device)
        queries = np.column_stack([np.full(len(points), query_frame, dtype=np.float32), points.astype(np.float32)])
        queries = torch.from_numpy(queries)[None].to(self._device)

        with torch.no_grad():
            tracks, visibility = self._predictor(
                video,
                queries=queries,
                backward_tracking=query_frame > 0,
            )
        return (
            tracks[0].cpu().numpy().astype(np.float64),
            visibility[0].cpu().numpy().astype(np.float64),
        )


def resolve_device(preferred: str | None = None) -> str:
    """Best available torch device (CUDA → MPS → CPU); see :mod:`ethograph.utils.device`."""
    return _resolve_device(preferred)


#: The *default* CoTracker3 offline weights (~97 MB). Fetched once into the
#: checkpoint dir so installing the backend stays a single pip command — the
#: model itself has no PyPI release and cannot ship weights through the
#: dependency resolver.
#:
#: Pinned rather than "latest" on purpose: a state dict only loads into the
#: architecture it was trained against, so the weights and :data:`COTRACKER_COMMIT`
#: move together. Better weights — a variant fine-tuned on animal footage, say —
#: are a drop-in state dict for the *same* architecture, and are selected by
#: passing ``checkpoint=`` to :func:`build_backend` (the dialog's "Model weights"
#: row, ``app_state.labelling_cotracker_checkpoint``), never by editing this URL.
#: A genuinely different architecture would be a new backend, not a new URL.
COTRACKER_CHECKPOINT_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
COTRACKER_CHECKPOINT_NAME = "scaled_offline.pth"

#: Pinned so the install is reproducible — the repo has no PyPI release, and an
#: unpinned branch is exactly the moving target we avoid ``torch.hub`` for.
COTRACKER_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"

#: The single install command. Kept here so the GUI hint, the docs and the error
#: messages cannot drift apart.
#: One explicit command — there is no ``[co-tracker]`` extra. cotracker has no
#: PyPI release and declares no dependencies (not even torch), so both halves
#: have to be named anyway; ``--torch-backend=auto`` picks up a GPU, which the
#: CPU-only Windows wheels on PyPI would otherwise silently ignore.
COTRACKER_INSTALL_HINT = (
    "uv pip install --torch-backend=auto torch "
    f'"cotracker @ git+https://github.com/facebookresearch/co-tracker.git@{COTRACKER_COMMIT}"'
)


def cotracker_checkpoint_dir() -> Path:
    """Where CoTracker3 weights are expected: ``~/.ethograph/cache/weights/cotracker``."""
    from ethograph.utils.paths import cache_dir

    return cache_dir("weights") / "cotracker"


def download_cotracker_checkpoint(progress: Progress | None = None) -> Path:
    """Fetch the CoTracker3 weights into the checkpoint dir.

    A plain HTTPS download — deliberately *not* ``torch.hub.load``, which needs
    GitHub reachable, can prompt interactively (hanging the Qt event loop) and
    tracks a moving branch. Downloads to a ``.part`` file and renames only on
    success, so an interrupted fetch never leaves weights that load as garbage.
    """
    import urllib.request

    directory = cotracker_checkpoint_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / COTRACKER_CHECKPOINT_NAME
    partial = target.with_suffix(target.suffix + ".part")

    with urllib.request.urlopen(COTRACKER_CHECKPOINT_URL) as response:  # noqa: S310 - fixed https URL
        total = int(response.headers.get("content-length") or 0)
        done = 0
        with open(partial, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if progress is not None and not progress(done / total if total else 0.0):
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("Checkpoint download cancelled.")
    partial.replace(target)
    return target


def find_cotracker_checkpoint(explicit: str | Path | None = None) -> Path | None:
    """Locate CoTracker3 weights, or ``None`` if they are not downloaded."""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    directory = cotracker_checkpoint_dir()
    if not directory.is_dir():
        return None
    # Prefer the offline (whole-clip) model — gaps here are short clips.
    for pattern in ("scaled_offline.pth", "*offline*.pth", "*.pth"):
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_cotracker_predictor(
    checkpoint: str | Path | None = None,
    device: str | None = None,
    progress: Progress | None = None,
) -> object:
    """Construct a ``CoTrackerPredictor`` on the best available device.

    Weights are downloaded on first use if absent, so installing the backend
    stays one pip command. Never ``torch.hub.load``: hub needs GitHub reachable,
    can prompt interactively (hanging the Qt event loop) and tracks a moving
    branch.

    Never passes ``checkpoint=None`` through to ``CoTrackerPredictor`` — that
    builds an *unloaded* network which returns confident nonsense with no error.
    """
    from cotracker.predictor import CoTrackerPredictor

    resolved = find_cotracker_checkpoint(checkpoint)
    if resolved is None:
        if checkpoint is not None:
            raise FileNotFoundError(f"No CoTracker3 checkpoint at {checkpoint}")
        resolved = download_cotracker_checkpoint(progress)

    predictor = CoTrackerPredictor(checkpoint=str(resolved))
    return predictor.to(resolve_device(device))


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
    # torch and cotracker install together but resolve separately (cotracker has
    # no PyPI release), so a missing either way reports the same one command.
    # Weights are NOT a precondition — they download on first use.
    installed = _module_available("torch") and _module_available("cotracker")
    # PosePAL is 500 optimisation steps, not a forward pass: on CPU it is not
    # slow but unusable, so it is offered only with a GPU.
    device = resolve_device() if installed else "cpu"
    on_gpu = installed and device != "cpu"
    label = POSEPAL_LABEL
    if on_gpu:
        # Naming the resolved device confirms the GPU was picked up — PyPI's
        # Windows torch wheels are CPU-only unless --torch-backend=auto was used.
        label = f"{POSEPAL_LABEL} ({device})"
        hint = "" if find_cotracker_checkpoint() else "~97 MB of weights will download on first use"
    elif installed:
        hint = "Needs a CUDA or Apple Silicon GPU — it fits a model to your labels."
    else:
        hint = COTRACKER_INSTALL_HINT

    return [
        BackendInfo("spline", "Spline (no extra dependencies)", True),
        BackendInfo(
            "flow",
            "Optical flow (OpenCV)",
            _module_available("cv2"),
            "pip install opencv-contrib-python-headless",
        ),
        BackendInfo(POSEPAL_BACKEND, label, on_gpu, hint),
    ]


def build_backend(
    key: str,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    progress: Progress | None = None,
    disagreement_px: float = DISAGREEMENT_SCALE,
    n_points: int | None = None,
) -> FillBackend:
    """Instantiate a backend by key, importing heavy dependencies only now.

    ``device=None`` auto-detects (CUDA → MPS → CPU) via :func:`resolve_device`.
    ``progress`` reports the one-time CoTracker weight download; call this from
    inside the progress dialog so a ~97 MB fetch is visible and cancellable.
    ``disagreement_px`` tunes the confidence of the tracking backends only —
    the spline scores by distance from the nearest anchor instead. ``n_points``
    is the flat ``(individual, keypoint)`` row count, needed only by PosePAL,
    which learns one feature per row.
    """
    if key == "spline":
        return SplineBackend()
    if key == "flow":
        return OpticalFlowBackend(disagreement_px=disagreement_px)
    if key == POSEPAL_BACKEND:
        if not n_points:
            raise ValueError("PosePAL fits one feature per point — pass n_points.")
        # Imported here and nowhere else: this module must stay importable
        # without torch, and pose_refine imports both torch and cotracker.
        from ethograph.gui.pose_refine import PosePALBackend, QueryFeatureRefinement

        resolved = resolve_device(device)
        predictor = load_cotracker_predictor(checkpoint, resolved, progress)
        refinement = QueryFeatureRefinement(predictor, n_points, device=resolved)
        return PosePALBackend(
            predictor,
            refinement,
            device=resolved,
            disagreement_px=disagreement_px,
        )
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
