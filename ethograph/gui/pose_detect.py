"""Point detection: read a position off the pixels of one frame.

A detector produces the same *kind* of thing a click does — a position asserted
from one specific frame — which is why detections join the anchors as
observations rather than competing with the fill (see
:mod:`~ethograph.gui.pose_annotate`). Filling is the other kind: inference about
the frames where no such evidence exists. Keeping the two apart is what lets any
detector compose with any fill backend, and it is why nothing in
:mod:`~ethograph.gui.pose_fill` changes to accommodate this module::

    Observations (frame-local evidence)      Inference (between observations)
      ├── manual    — the user clicks          ├── spline
      └── detected  — this module             ├── optical flow
                                              └── PosePAL
            └────────────── feeds ───────────────┘

Two consequences are the point of the exercise. **PosePAL improves**: its
refinement fits query embeddings against the labelled frames, and five
hand-labelled frames becoming several hundred detected ones is a far
better-posed optimisation. **Optical flow stops drifting**: every detected frame
is a fresh anchor, so Lucas-Kanade never integrates error across more than one
gap.

What a detector does *not* know
-------------------------------
A tag decodes to the integer ``7``. It does not know it is the *beak*, or
*bee 12*. Closing that gap is the
:class:`~ethograph.gui.pose_annotate.AssignmentTable`, which is explicit,
editable and persisted rather than an implementation detail of a run:
:func:`learn_assignment` proposes it by matching detections to the frames the
user labelled, and the dialog lets any row be corrected by hand.

:func:`run_detector` is the whole stage, and it is a free function rather than a
class because it holds no state between frames. That independence is what makes
detection restartable, cacheable and trivially parallel later.

Three families, one detector, and why
-------------------------------------
AprilTag decoded by **pupil-apriltags** (AprilTag 3, Krogius et al. 2019) is the
only detector here, over an **allowlist of three families**: ``tag36h11`` (the
default), ``tag25h9`` and ``tag16h5``. Everything below was measured rather than
assumed.

**Why an allowlist rather than a free string.** Constructing a ``Detector`` for
``tagCircle49h12``, ``tagCustom48h12`` or ``tagStandard52h13`` **aborts the
process** with "failed to allocate hamming decode table". An abort is not an
exception — it takes the GUI and any unsaved labels with it — so the check has to
happen before the call, which is :func:`check_family`. The three listed are each
verified to construct, to round-trip through OpenCV's renderer, and to produce no
``hamming == 0`` detection on blurred noise.

**Why ``tag36h11`` is the default.** A family is a set of code words, and how
sparse that set is decides how often noise looks like a tag. ``tag36h11`` accepts
~1.5% of its 36-bit space with minimum Hamming distance 11, and holds 587 IDs;
``tag16h5`` accepts roughly a quarter of all 16-bit patterns, at distance 5, and
holds 30.

**Why the smaller two are still offered.** The failure that matters is a *wrong
ID accepted silently*, not a missed tag — and the ``hamming > 0`` rule already
catches it. Measured over 20 blurred-noise frames with no tag present:
``tag16h5`` proposed 50 reads and **every one needed a bit correction**, so none
survived; ``tag25h9`` and ``tag36h11`` proposed none at all. What ``tag16h5``
buys is paper: 6 modules against 8, so ~25% less printed side for the same pixels
per module, which is the difference between a tag that fits an animal and one
that does not. The trade is real but bounded, and it is the user's to make — with
the caveat in :func:`family_note` that a genuinely corrupted read has less room
to be caught.

That is *not* the same as re-admitting ArUco's ``DICT_4X4_*``. Those were
detected by OpenCV's own pipeline, which reports no hamming distance at all, so
the rule that rescues ``tag16h5`` here has nothing to act on.

**The detector.** The previous implementation ran ``cv2.aruco``, which offers the
AprilTag *code words* decoded by OpenCV's own ArUco pipeline — not AprilTag's
detector. Throughput is not a reason to stay with it; measured on synthetic
scenes with four 50 px tags, ``pupil-apriltags`` at ``quad_decimate=1.0`` takes
18.0 ms at 1080p against ``cv2.aruco``'s 21.8 ms (at 4K, 102.5 ms against
58.1 ms, and 19.3 ms at ``quad_decimate=2.0``). Either way the cost is minutes
per hour of video, and :func:`run_detector` already chunks frames. ``nthreads``
barely helps — parallelise **across** frames, not within them.

OpenCV is still required, but only to *render* tags: ``pupil-apriltags`` exports
no generator, so :mod:`~ethograph.gui.pose_tagsheet` and :meth:`AprilTagDetector.preview`
draw through ``cv2.aruco.generateImageMarker`` with the matching
``DICT_APRILTAG_*`` (:data:`TAG_DICTIONARIES`). That round trip is verified per
family in the tests, and holds down to ~6 px per module.

:class:`PointDetector` stays a Protocol and :func:`available_detectors` stays a
list. Colour still wins where a tag physically cannot go — curved or tiny
bodies, steep viewing angles, retroreflective markers on a mono sensor — and a
learned detector is a plausible second entry.

Printing the tags
-----------------
- **Pixel budget:** :data:`PX_PER_MODULE` px per module including the mandatory
  black border, so ~40 px per side for ``tag36h11``'s 8 modules and ~30 for
  ``tag16h5``'s 6. :attr:`AprilTagDetector.min_side_px` states it for the
  settings in force, ``quad_decimate`` included — that is the number to compare
  against the tag on screen when nothing is being found.
- **Traps:** ``quad_decimate`` defaults to **2.0** in ``pupil-apriltags`` itself,
  halving the effective tag size before the quad finder ever runs; this module
  defaults it to 1.0. The white quiet zone is not decoration and is easy to crop
  away.
- **Materials:** matte paper only — gloss reflects the light source and kills
  quad detection. Print without interpolation and glue to card so the tag stays
  planar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ethograph.gui.pose_annotate import LEARNED, Assignment, KeypointStore
from ethograph.gui.pose_fill import Progress, no_progress

#: Most frames decoded per read. One frame at a time would re-seek per frame; a
#: whole video at once does not fit in memory.
CHUNK_FRAMES = 64

#: Memory a chunk of decoded frames may take. Detection runs at **full
#: resolution** (see :func:`run_detector`), where 64 frames of 4K is 1.6 GB, so
#: the frame count is derived from this rather than fixed.
CHUNK_BYTES = 64 * 1024 * 1024

#: Detector keys, as :func:`build_detector` takes them.
APRILTAG_DETECTOR = "apriltag"

#: Tag families offered, best first — and an **allowlist, not a suggestion**.
#: Only these three are ever handed to ``pupil_apriltags.Detector``, because
#: constructing one for ``tagCircle49h12``, ``tagCustom48h12`` or
#: ``tagStandard52h13`` aborts the **process** with "failed to allocate hamming
#: decode table", taking the GUI and any unsaved labels with it. An abort is not
#: an exception, so the guard has to be *before* the call:
#: :class:`AprilTagDetector` raises on anything outside this tuple and a family
#: string never reaches the library unchecked. **Never widen this without
#: constructing the family in a throwaway process first.**
#:
#: All three are verified in the tests to construct, to round-trip through
#: OpenCV's renderer, and to yield no ``hamming == 0`` detection on blurred
#: noise.
TAG_FAMILIES = ("tag36h11", "tag25h9", "tag16h5")

#: What a new detector starts on. ``tag36h11`` is the default because it has the
#: most margin and the most IDs — see :func:`family_note` for what the smaller
#: two trade away for paper.
DEFAULT_TAG_FAMILY = "tag36h11"

#: The OpenCV dictionaries holding the same code words, for *rendering* only —
#: ``pupil-apriltags`` has no generator. Verified in the tests to decode back to
#: the same IDs, per family.
TAG_DICTIONARIES = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
}

#: Pixels per tag module needed for a reliable decode, border included. Below
#: ~3 px per module detection falls off a cliff; 5 is the design figure. It
#: lives here rather than in `pose_tagsheet` because it is a fact about the
#: *decoder*, and the printing advice is derived from it — one number, two uses.
PX_PER_MODULE = 5.0

#: Border modules the renderer puts around the data grid, one each side.
BORDER_MODULES = 1

#: The decode margin a clean, well-lit tag scores — measured at ~119 on a
#: rendered tag at 8 px per module. :class:`Detection` quality is this margin
#: divided by this figure and clipped to ``[0, 1]``, so a good read saturates at
#: 1.0 and the dialog's one quality threshold stays a plain fraction that
#: composes with fill confidence.
#:
#: The scale is not cosmetic. Spurious detections on noise scored 2–13 in
#: testing against ~119 for a real tag, so the default threshold of 0.3 (margin
#: 30) sits in an empty band between the two populations rather than on a guess.
GOOD_DECISION_MARGIN = 100.0

#: ``pupil-apriltags`` reports corners as TR, TL, BL, BR; OpenCV reports TL, TR,
#: BR, BL, and ``TAG_PARTS`` is written in OpenCV's order. This maps the former
#: onto the latter — measured, and stable under tag rotation. It is applied once,
#: in :meth:`AprilTagDetector._decode`, so nothing downstream knows about it; a
#: regression test pins it, because getting it wrong silently shifts every saved
#: corner assignment by one physical corner.
_CORNER_ORDER = (1, 0, 3, 2)

#: Match radius for :func:`learn_assignment`, as a share of the longest image
#: side. A fixed pixel count would mean something different on every recording;
#: this is "a fiftieth of the frame", which is generous for a marker the user
#: clicked on and still far short of the next keypoint along.
MATCH_RADIUS_FRACTION = 0.02

#: Labelled frames that must agree before an assignment is proposed. One frame
#: is a coincidence — the nearest detection to a click is *always* something.
MIN_AGREEING_FRAMES = 2

#: Parts of a tag that can be detected, in a fixed order. The label a detection
#: carries is ``tag_id * TAG_PART_STRIDE + index into this tuple``, so a saved
#: assignment keeps its meaning when the enabled parts change.
TAG_PARTS = ("centre", "corner_0", "corner_1", "corner_2", "corner_3")
TAG_PART_STRIDE = 8

#: The edge of a printed tag taken as its FRONT, as indices into the quad
#: (which is in OpenCV's order: TL, TR, BR, BL). A tag is worn with its printed
#: top edge facing forwards, so that edge runs TL → TR and the forward direction
#: is perpendicular to it, pointing away from the bottom two corners.
FRONT_EDGE = (0, 1)


class PointDetectorError(Exception):
    """Raised when a detector cannot be built or run."""


def check_family(family: str) -> str:
    """Return *family* if it is on the allowlist, else raise.

    The single gate in front of the C library. It exists because an unlisted
    family does not raise — it *aborts the process* (see :data:`TAG_FAMILIES`),
    so validation cannot be left to ``Detector`` itself.
    """
    if family not in TAG_FAMILIES:
        raise PointDetectorError(f"Unknown tag family {family!r}; expected one of {list(TAG_FAMILIES)}")
    return family


def family_modules(family: str) -> int:
    """Cells per side of a printed tag, black border included.

    Derived from the family's own name rather than tabulated: ``tag36h11`` is a
    36-bit code, so a 6×6 data grid, so 8 modules with the border. A test pins
    this against OpenCV's ``markerSize``, which is the other place the same fact
    lives — so the two can never quietly disagree.
    """
    from math import isqrt

    bits = int("".join(ch for ch in check_family(family).removeprefix("tag").split("h")[0] if ch.isdigit()))
    side = isqrt(bits)
    if side * side != bits:
        raise PointDetectorError(f"Cannot derive a module count from family {family!r}")
    return side + 2 * BORDER_MODULES


def family_note(family: str) -> str:
    """One line on what this family costs and buys, for the dialog's tooltip.

    Kept here rather than in the GUI because it is a fact about the code space,
    and the tag sheet wants to say the same thing as the Detect tab.
    """
    info = {
        "tag36h11": "587 IDs, the widest margin — the default, and the right answer unless paper is tight.",
        "tag25h9": "35 IDs, 7 modules — one module less paper than tag36h11 for the same pixels per module.",
        "tag16h5": (
            "30 IDs, 6 modules — the smallest printable tag, and the thinnest margin. "
            "It proposes far more bad reads than the others; Ethograph drops every one that "
            "needed a bit correction, so what survives is trustworthy, but a genuinely "
            "corrupted read has less room to be caught. Check a run before relying on it."
        ),
    }
    return info.get(check_family(family), "")


@dataclass(frozen=True)
class Detection:
    """One marker found on one frame.

    ``xy`` is in the pixels of the frame it was found in — :func:`run_detector`
    scales it back to source pixels, since :class:`~ethograph.gui.pose_fill.VideoFrameSource`
    may be decoding smaller.

    ``quality`` is in ``[0, 1]`` so it composes with fill confidence; for tags it
    is the decode margin over :data:`GOOD_DECISION_MARGIN`.

    ``orientation`` is the marker's own forward direction as a **unit vector in
    image coordinates**, or ``None`` for a marker that has none. A decoded tag
    is not a point — it is a square whose corners say which way it is facing —
    and that is the whole of what a head direction needs. Carrying it on the
    detection keeps it attached to the point it belongs to, so nothing
    downstream has to reconstruct an orientation from separate keypoints that
    happen to sit on the same marker.
    """

    xy: np.ndarray
    label: int
    quality: float = 1.0
    orientation: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "xy", np.asarray(self.xy, dtype=np.float64).reshape(2))
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "quality", float(self.quality))
        if self.orientation is not None:
            object.__setattr__(self, "orientation", np.asarray(self.orientation, dtype=np.float64).reshape(2))


@dataclass(frozen=True)
class PreviewShape:
    """One thing the detector saw on a frame, kept or not.

    ``accepted`` is the whole point: a shape the detector *found* and then
    discarded is a different problem from one it never found, and *reason* says
    which threshold discarded it.
    """

    xy: np.ndarray
    label: int | None
    accepted: bool
    reason: str = ""
    outline: np.ndarray | None = None
    quality: float = 1.0


@dataclass(frozen=True)
class DetectionPreview:
    """What one frame looks like to a detector, for the tuning preview.

    *size* is ``(width, height)`` of the frame the shapes are measured in, which
    is the **decoded** size rather than the source video's — previewing over a
    full-resolution frame would make a tag look readable that the detector will
    never see at all.
    """

    shapes: list[PreviewShape] = field(default_factory=list)
    size: tuple[int, int] = (0, 0)

    @property
    def accepted(self) -> list[PreviewShape]:
        return [shape for shape in self.shapes if shape.accepted]

    @property
    def rejected(self) -> list[PreviewShape]:
        return [shape for shape in self.shapes if not shape.accepted]


@runtime_checkable
class PointDetector(Protocol):
    """Finds labelled markers in a single frame, statelessly."""

    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Every marker visible in *frame* (an ``(H, W, 3)`` RGB array)."""


def label_name(detector: object, label: int) -> str:
    """Human-readable name for a detector label, for the assignment table."""
    namer = getattr(detector, "label_name", None)
    return str(namer(label)) if callable(namer) else f"label {int(label)}"


def label_preview(detector: object, label: int) -> np.ndarray | None:
    """Small RGB image standing for a label — a swatch, or the tag itself."""
    preview = getattr(detector, "preview", None)
    return preview(label) if callable(preview) else None


def diagnose_frame(detector: object, frame: np.ndarray) -> DetectionPreview:
    """What *detector* sees on *frame*, near misses included.

    Optional, like :func:`label_name` and :func:`label_preview`: the
    :class:`PointDetector` protocol stays ``name`` + ``detect``, so a detector
    written elsewhere never pays for this dialog's needs. A detector without it
    still previews — as its accepted detections and nothing else.
    """
    diagnose = getattr(detector, "diagnose", None)
    if callable(diagnose):
        return diagnose(frame)
    image = _as_rgb(frame)
    shapes = [
        PreviewShape(xy=found.xy, label=found.label, accepted=True, quality=found.quality)
        for found in detector.detect(image)
    ]
    return DetectionPreview(shapes=shapes, size=(image.shape[1], image.shape[0]))


def _quad_side(quad: np.ndarray) -> float:
    """Mean side length of a 4-point quad, in the pixels it was measured in."""
    corners = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    if len(corners) < 2:
        return 0.0
    return float(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1).mean())


def quad_forward_vector(quad: np.ndarray) -> np.ndarray | None:
    """Unit vector along a tag's forward direction, in **image coordinates**.

    A tag is a square, so one decode fixes its orientation completely — no
    second marker and no second keypoint are needed. The front is the printed
    top edge (:data:`FRONT_EDGE`, TL → TR in the normalised corner order), and
    forward is perpendicular to it, pointing away from the bottom two corners.

    Concretely, with y growing **downward** as it does in image coordinates, the
    left-hand normal of TL → TR points towards the top of the frame — which is
    where a tag printed the right way up is facing. ``None`` when the edge is
    degenerate (a tag seen exactly edge-on), which is a missing measurement
    rather than a zero direction.
    """
    corners = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    left, right = corners[FRONT_EDGE[0]], corners[FRONT_EDGE[1]]
    edge = right - left
    length = float(np.hypot(*edge))
    if not length:
        return None
    # Left-hand normal of the edge: (dx, dy) -> (dy, -dx).
    return np.array([edge[1], -edge[0]]) / length


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    """Contiguous ``(H, W, 3)`` uint8, whatever the source handed over.

    Mono footage arrives as ``(H, W)`` — OpenCV's colour conversions reject it
    rather than broadcasting, and a detector that crashes on grayscale video is
    a detector nobody with an infrared camera can use.
    """
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


# ----------------------------------------------------------------------
# Fiducial tags
# ----------------------------------------------------------------------

#: Downscale applied before the quad finder. ``pupil-apriltags`` ships **2.0**,
#: which halves the effective tag size — the single most expensive default in
#: the library for animal-borne tags, and the reason this module overrides it.
#: Raising it is the real speed/size trade, so it stays a control.
DEFAULT_QUAD_DECIMATE = 1.0

#: Sharpening applied to the sampled bit pattern before decoding. The library's
#: own default, kept because it is the knob that recovers a motion-blurred tag.
DEFAULT_DECODE_SHARPENING = 0.25


@lru_cache(maxsize=len(TAG_FAMILIES))
def _shared_detector(family: str):
    """One ``pupil_apriltags.Detector`` per family, for the life of the process.

    Keyed by family and **only** by family. A ``tag36h11`` detector allocates a
    ~36 MB hamming decode table (``tag16h5`` 4.5 MB, ``tag25h9`` 1.1 MB), so one
    per *parameter* combination is not an option — a dragged spin box would be
    hundreds of megabytes. It does not need to be: ``quad_decimate`` and
    ``decode_sharpening`` are plain fields of the C struct
    (:func:`_apply_params`), so one instance per family serves every setting.
    All three together cost ~41 MB, which is why the cache holds all of them
    rather than evicting.

    They are also deliberately **never released**. ``Detector.__del__`` calls
    ``apriltag_detector_destroy``, and a ``malloc(): mismatching next->prev_size``
    abort has been seen on that path after a successful detection. An abort is
    not an exception — it would take the GUI and any unsaved labels with it — so
    the safest number of destructions is zero, and ``maxsize`` is set so the
    cache can never evict one either.

    A family that is not on the allowlist must never get this far; see
    :func:`check_family`.
    """
    try:
        from pupil_apriltags import Detector
    except ImportError as e:
        raise PointDetectorError(f"AprilTag detection needs pupil-apriltags: {APRILTAG_INSTALL_HINT}") from e
    return Detector(families=check_family(family))


def _apply_params(detector, quad_decimate: float, decode_sharpening: float) -> None:
    """Stamp this detector's settings onto the shared C struct before a decode.

    Written directly rather than passed to the constructor for two reasons. It
    is what makes one shared instance possible at all, and the constructor is
    **wrong**: ``pupil_apriltags`` casts ``decode_sharpening`` with ``int()``
    into a ``c_double`` field, so its own documented default of 0.25 arrives as
    0.0 and every value under 1.0 silently does nothing. The struct takes the
    float unharmed.
    """
    params = detector.tag_detector_ptr.contents
    params.quad_decimate = float(quad_decimate)
    params.decode_sharpening = float(decode_sharpening)


class AprilTagDetector:
    """AprilTag markers, via ``pupil-apriltags`` (AprilTag 3).

    One family per detector — mixing them in a single ``Detector`` would put
    ``tag16h5``'s id 7 and ``tag36h11``'s id 7 on the same label, and the label
    is what an assignment is keyed by. Two families in one video therefore mean
    two runs, exactly as two dictionaries used to.

    Each tag yields its centre by default; ``parts`` can add the four corners,
    which is how one tag becomes four keypoints. The label a part carries is
    ``tag_id * TAG_PART_STRIDE + part index``, fixed regardless of which parts
    are enabled, so an assignment saved today still means the same tag corner
    tomorrow — **within the same family**. Corners are reported in OpenCV's
    order (TL, TR, BR, BL) whatever the underlying library does — see
    :data:`_CORNER_ORDER`.

    **A corrected read is not a read.** Detections with ``hamming > 0`` are
    dropped outright rather than scored down: in testing every false positive
    needed bit corrections and every true one needed none, so the boundary is
    exactly there. That rule is what makes the smaller families usable at all —
    ``tag16h5`` proposed 50 reads across 20 blurred-noise frames and every one
    of them needed a correction, so none survived. Quality is then the decode
    margin (see :data:`GOOD_DECISION_MARGIN`) — a real measurement of how
    confidently the bits separated.
    """

    requires_video = True

    def __init__(
        self,
        family: str = DEFAULT_TAG_FAMILY,
        quad_decimate: float = DEFAULT_QUAD_DECIMATE,
        decode_sharpening: float = DEFAULT_DECODE_SHARPENING,
        parts: Sequence[str] = ("centre",),
    ):
        unknown = set(parts) - set(TAG_PARTS)
        if unknown:
            raise PointDetectorError(f"Unknown tag part(s) {sorted(unknown)}; expected {list(TAG_PARTS)}")
        if quad_decimate < 1.0:
            raise PointDetectorError(f"quad_decimate is a downscale and cannot go below 1.0; got {quad_decimate}")
        self.family = check_family(family)
        #: Cells per side including the black border — what a decode samples.
        self.modules = family_modules(self.family)
        self.quad_decimate = float(quad_decimate)
        self.decode_sharpening = float(decode_sharpening)
        self.parts = list(parts)
        self._detector = _shared_detector(self.family)

    @property
    def name(self) -> str:
        return f"AprilTag ({self.family})"

    def __repr__(self) -> str:
        return f"AprilTagDetector({self.family}, quad_decimate={self.quad_decimate:g}, parts={self.parts})"

    @property
    def min_side_px(self) -> float:
        """Pixels per tag side needed for a reliable decode, in decoded pixels.

        The single most useful number when nothing is being found: compare it
        with how big the tag actually is on screen. ``quad_decimate`` is part of
        it because the quad finder never sees the full-resolution frame — a
        decimation of 2.0 genuinely doubles the tag size needed.
        """
        return self.modules * PX_PER_MODULE * self.quad_decimate

    @staticmethod
    def label_for(tag_id: int, part: str = "centre") -> int:
        return int(tag_id) * TAG_PART_STRIDE + TAG_PARTS.index(part)

    @staticmethod
    def decode_label(label: int) -> tuple[int, str]:
        tag_id, index = divmod(int(label), TAG_PART_STRIDE)
        return tag_id, TAG_PARTS[index] if index < len(TAG_PARTS) else f"part_{index}"

    def label_name(self, label: int) -> str:
        tag_id, part = self.decode_label(label)
        return f"tag {tag_id}" if part == "centre" else f"tag {tag_id} · {part.replace('_', ' ')}"

    def preview(self, label: int) -> np.ndarray | None:
        """The tag itself, rendered — a thumbnail nobody has to decode by eye.

        Rendered by OpenCV, since ``pupil-apriltags`` has no generator; the two
        libraries agree on IDs, which the tests pin.
        """
        import cv2

        tag_id, _part = self.decode_label(label)
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_DICTIONARIES[self.family]))
        try:
            image = cv2.aruco.generateImageMarker(dictionary, tag_id, 32)
        except cv2.error:
            return None
        return np.repeat(image[:, :, None], 3, axis=2)

    def _decode(self, frame: np.ndarray) -> tuple[np.ndarray, list[tuple[int, np.ndarray, float, int]]]:
        """``(rgb image, [(tag_id, quad, quality, hamming)])``, corners in cv2 order.

        The single boundary with ``pupil-apriltags``: grayscale conversion, the
        corner reordering and the quality scale all happen here and nowhere
        else, so :meth:`detect` and :meth:`diagnose` cannot drift apart. The
        settings are stamped on **per call** rather than at construction,
        because the detector object is shared between every
        :class:`AprilTagDetector` — see :func:`_shared_detector`.
        """
        import cv2

        image = _as_rgb(frame)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _apply_params(self._detector, self.quad_decimate, self.decode_sharpening)
        found = []
        # `detect` asserts ndim == 2 and dtype == uint8 with a bare, message-less
        # AssertionError, which is why the conversion above is not optional.
        for tag in self._detector.detect(grey):
            quad = np.asarray(tag.corners, dtype=np.float64).reshape(4, 2)[list(_CORNER_ORDER)]
            quality = float(np.clip(tag.decision_margin / GOOD_DECISION_MARGIN, 0.0, 1.0))
            found.append((int(tag.tag_id), quad, quality, int(tag.hamming)))
        return image, found

    def detect(self, frame: np.ndarray) -> list[Detection]:
        found: list[Detection] = []
        for tag_id, quad, quality, hamming in self._decode(frame)[1]:
            if hamming:
                continue
            # One orientation per tag, carried by every part it emits: the
            # square is what is facing a direction, not any one of its corners.
            orientation = quad_forward_vector(quad)
            for part in self.parts:
                xy = quad.mean(axis=0) if part == "centre" else quad[TAG_PARTS.index(part) - 1]
                found.append(Detection(xy, self.label_for(tag_id, part), quality, orientation))
        return found

    def diagnose(self, frame: np.ndarray) -> DetectionPreview:
        """Decoded tags, plus the reads that were thrown away and why.

        The two failures worth telling apart are "there is a tag here that could
        not be trusted" and "there is no tag here". The first shows up as a
        rejected shape drawn in place, captioned with the measured side alone —
        the dashed red outline already says it was thrown away, and the size is
        the only number worth acting on; the second is an empty preview, where
        the frame and :attr:`min_side_px` are what settle it.
        """
        image, found = self._decode(frame)
        shapes: list[PreviewShape] = []
        for tag_id, quad, quality, hamming in found:
            side = _quad_side(quad)
            reason = ""
            if hamming:
                # A corrected read is the one failure mode that survives every
                # later stage, so it is shown as a *rejection*, not a warning.
                reason = f"{side:.0f} px"
            shapes.append(
                PreviewShape(
                    xy=quad.mean(axis=0),
                    label=None if hamming else self.label_for(tag_id, "centre"),
                    accepted=not hamming,
                    reason=reason,
                    outline=quad,
                    quality=quality,
                )
            )
        return DetectionPreview(shapes=shapes, size=(image.shape[1], image.shape[0]))


# ----------------------------------------------------------------------
# Running a detector
# ----------------------------------------------------------------------


def run_detector(
    detector: PointDetector,
    frames,
    rows: Mapping[int, int],
    n_points: int,
    span: tuple[int, int] | None = None,
    progress: Progress = no_progress,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Detect over *span*, mapped onto flat point rows by *rows*.

    *rows* is ``detector label -> flat point row`` (see
    :meth:`~ethograph.gui.pose_annotate.KeypointStore.assignment_rows`) — the
    whole of what a run needs to know about the individual/keypoint hierarchy.
    A label with no assignment is **dropped rather than guessed at**, and a
    label found twice in one frame is dropped as well: two markers claiming the
    same point is an ambiguity, and picking the bigger one would hide a set-up
    that does not separate two markers.

    *frames* must decode at the video's **full resolution**. Every other stage
    here downscales — the tracking backends lose almost nothing at 512 px — but
    resolution is the entire signal a tag decoder has: an 8-module tag needs
    ~5 px per module including its border, so a 3.75× downscale turns a
    comfortable 45 px tag into 12 px, which the quad finder does not even
    propose as a candidate. Chunks are therefore sized by
    :data:`CHUNK_BYTES` rather than by a fixed frame count.

    Returns ``(positions, quality, orientation)`` as sparse ``frame ->
    (n_points, 2)``, ``frame -> (n_points,)`` and ``frame -> (n_points, 2)``,
    holding only the frames where something was found, in **source-video
    pixels** — a frame source decoding smaller is scaled back here, so every
    pixel threshold in this module applies to the decoded frame and everything
    leaving it is in the coordinates the store keeps. *orientation* is a unit
    vector and so is **not** scaled; a row whose marker has no orientation, and
    every row of a detector that reports none at all, stays ``NaN``.

    Cancelling returns what has been found so far; nothing here depends on
    having seen the whole video, which is what makes a run restartable.
    """
    total_frames = len(frames)
    first, last = span if span is not None else (0, total_frames - 1)
    first, last = max(int(first), 0), min(int(last), total_frames - 1)
    if last < first:
        return {}, {}, {}
    scale = float(getattr(frames, "scale", 1.0))

    positions: dict[int, np.ndarray] = {}
    quality: dict[int, np.ndarray] = {}
    orientation: dict[int, np.ndarray] = {}
    span_length = last - first + 1
    chunk = chunk_frames(frames)
    for start in range(first, last + 1, chunk):
        if not progress((start - first) / span_length):
            break
        stop = min(start + chunk, last + 1)
        clip = np.asarray(frames[start:stop])
        for offset, image in enumerate(clip):
            found = _unambiguous(detector.detect(image), rows)
            if not found:
                continue
            frame_positions = np.full((n_points, 2), np.nan, dtype=np.float64)
            frame_quality = np.full(n_points, np.nan, dtype=np.float64)
            frame_orientation = np.full((n_points, 2), np.nan, dtype=np.float64)
            for label, detection in found.items():
                row = rows[label]
                frame_positions[row] = detection.xy * scale
                frame_quality[row] = detection.quality
                if detection.orientation is not None:
                    # A unit vector is scale-free, so the frame source's
                    # downscale must NOT be applied to it.
                    frame_orientation[row] = detection.orientation
            positions[start + offset] = frame_positions
            quality[start + offset] = frame_quality
            orientation[start + offset] = frame_orientation
    return positions, quality, orientation


def chunk_frames(frames) -> int:
    """How many full-resolution frames to hold at once, within :data:`CHUNK_BYTES`.

    A frame source that does not report its size gets :data:`CHUNK_FRAMES`, the
    old fixed count — the budget is a guard against 4K, not a contract.
    """
    size = getattr(frames, "size", None)
    if not size:
        return CHUNK_FRAMES
    width, height = size
    per_frame = max(int(width) * int(height) * 3, 1)
    return int(np.clip(CHUNK_BYTES // per_frame, 1, CHUNK_FRAMES))


def _unambiguous(detections: Sequence[Detection], rows: Mapping[int, int] | None) -> dict[int, Detection]:
    """One detection per label, dropping labels seen twice or not assigned."""
    seen: dict[int, Detection | None] = {}
    for detection in detections:
        if rows is not None and detection.label not in rows:
            continue
        seen[detection.label] = None if detection.label in seen else detection
    return {label: detection for label, detection in seen.items() if detection is not None}


# ----------------------------------------------------------------------
# Learning the assignment
# ----------------------------------------------------------------------


@dataclass
class LearnedAssignments:
    """What :func:`learn_assignment` found, including what it could not."""

    proposals: list[Assignment] = field(default_factory=list)
    #: Labels seen but matched to no labelled point often enough to propose.
    unmatched_labels: list[int] = field(default_factory=list)
    #: ``(individual, keypoint)`` pairs the user labelled that no detector label
    #: claimed — the warning that stops a run from quietly producing nothing.
    unmatched_targets: list[tuple[str, str]] = field(default_factory=list)
    #: Labelled frames actually scanned.
    frames_scanned: int = 0


def learn_assignment(
    detector: PointDetector,
    frames,
    store: KeypointStore,
    radius_px: float | None = None,
    min_frames: int = MIN_AGREEING_FRAMES,
    progress: Progress = no_progress,
) -> LearnedAssignments:
    """Propose what each detector label means, by matching it to the labels.

    On every labelled frame the detector is run and each detection matched to
    the nearest labelled point. A label is proposed for the target that won on
    the most frames, subject to three guard rails: at least *min_frames* frames
    must agree, no match beyond *radius_px* counts, and a label whose target is
    already claimed by a better-supported label is left unassigned rather than
    guessed at.

    *radius_px* is in **source pixels** and defaults to
    :data:`MATCH_RADIUS_FRACTION` of the longest image side — a fixed pixel
    count would mean something different on every recording.
    """
    labelled = store.anchor_frames()
    result = LearnedAssignments()
    if not labelled:
        return result
    scale = float(getattr(frames, "scale", 1.0))

    votes: dict[int, dict[tuple[str | None, str], int]] = {}
    radius = radius_px
    for done, frame_index in enumerate(labelled):
        if not progress(done / len(labelled)):
            break
        image = np.asarray(frames[int(frame_index)])
        if radius is None:
            radius = MATCH_RADIUS_FRACTION * float(max(image.shape[:2])) * scale
        result.frames_scanned += 1
        points = store.anchor_positions(frame_index)
        for label, detection in _unambiguous(detector.detect(image), None).items():
            xy = detection.xy * scale
            target = _nearest_label(points, store, xy, radius)
            if target is None:
                continue
            tally = votes.setdefault(label, {})
            tally[target] = tally.get(target, 0) + 1

    claimed: dict[tuple[str | None, str], int] = {}
    ranked = sorted(votes.items(), key=lambda item: -max(item[1].values()))
    for label, tally in ranked:
        target, count = max(tally.items(), key=lambda item: item[1])
        if count < min_frames or target in claimed:
            result.unmatched_labels.append(label)
            continue
        claimed[target] = label
        individual, keypoint = target
        result.proposals.append(Assignment(label, individual, keypoint, LEARNED, matched_frames=count))
    result.proposals.sort(key=lambda a: a.label)
    result.unmatched_labels.sort()
    result.unmatched_targets = sorted(
        (individual, keypoint)
        for individual in store.individual_names
        for keypoint in store.keypoints_for(individual)
        if (individual, keypoint) not in claimed and store.anchor_frames_for(keypoint, individual)
    )
    return result


def _nearest_label(
    points: np.ndarray,
    store: KeypointStore,
    xy: np.ndarray,
    radius: float,
) -> tuple[str, str] | None:
    """The labelled ``(individual, keypoint)`` closest to *xy*, within *radius*."""
    distances = np.hypot(points[:, :, 0] - xy[0], points[:, :, 1] - xy[1])
    distances[np.isnan(distances)] = np.inf
    if not distances.size:
        return None
    i, k = np.unravel_index(int(np.argmin(distances)), distances.shape)
    if distances[i, k] > radius:
        return None
    return store.individual_names[i], store.keypoint_names[k]


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------


@dataclass
class DetectorInfo:
    key: str
    label: str
    available: bool
    hint: str = ""


#: Detection itself. A wheel exists for every platform this GUI runs on.
APRILTAG_INSTALL_HINT = "pip install pupil-apriltags"

#: What to install when cv2 is missing. Mirrors the optical-flow backend's
#: requirement: plain ``opencv-python`` ships Qt plugins that conflict with
#: PyQt6. Needed here only to *render* tags — detection does not touch it.
OPENCV_INSTALL_HINT = "pip install opencv-contrib-python-headless"


def available_detectors() -> list[DetectorInfo]:
    """Describe every detector so the dialog can grey out the missing ones.

    One entry today, covering all of :data:`TAG_FAMILIES` — the family is a
    *parameter* of this detector, not a detector of its own, since one class
    reads all three. The list, the key and the combo that shows it are the seam
    a genuinely different detector arrives through.
    """
    from importlib.util import find_spec

    missing = []
    for module, hint in (("pupil_apriltags", APRILTAG_INSTALL_HINT), ("cv2", OPENCV_INSTALL_HINT)):
        try:
            present = find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(hint)
    return [
        DetectorInfo(
            APRILTAG_DETECTOR,
            "AprilTag (pupil-apriltags)",
            not missing,
            "; ".join(missing),
        ),
    ]


def build_detector(key: str, **params) -> PointDetector:
    """Instantiate a detector by key, importing the tag library only now."""
    if key == APRILTAG_DETECTOR:
        return AprilTagDetector(**params)
    raise PointDetectorError(f"Unknown point detector {key!r}")
