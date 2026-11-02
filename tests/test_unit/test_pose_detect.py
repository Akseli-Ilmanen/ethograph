"""Point detection: the detector, the assignment table, and detections as observations.

Everything here is synthetic — tags rendered by ``cv2.aruco.generateImageMarker``
and read back by ``pupil-apriltags`` — so recovery can be checked against ground
truth with no video fixture and no downloaded weights. **This feature will never
run on the crow data**, which is untagged, so these tests are the only regression
safety net it has.

Two of them are load-bearing beyond the usual. The **round trip** is the contract
between the two libraries: OpenCV draws the tags, AprilTag reads them, nothing
checks that agreement at runtime, and a version bump on either side is exactly
what would break it. The **corner order** is the other: the libraries disagree
about which corner comes first, and getting the remap wrong shifts every saved
corner assignment by one physical corner — silently, with no error anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui.pose_annotate import (
    LEARNED,
    MANUAL,
    SCANNED,
    Assignment,
    AssignmentError,
    AssignmentTable,
    KeypointStore,
    detections_path,
)
from ethograph.gui.pose_detect import (
    APRILTAG_DETECTOR,
    CHUNK_BYTES,
    CHUNK_FRAMES,
    DEFAULT_TAG_FAMILY,
    GOOD_DECISION_MARGIN,
    PX_PER_MODULE,
    TAG_DICTIONARIES,
    TAG_FAMILIES,
    TAG_PART_STRIDE,
    TAG_PARTS,
    AprilTagDetector,
    Detection,
    PointDetector,
    PointDetectorError,
    ScannedLabels,
    available_detectors,
    build_detector,
    check_family,
    chunk_frames,
    diagnose_frame,
    family_modules,
    family_note,
    label_name,
    label_preview,
    learn_assignment,
    marker_part,
    part_named,
    propose_assignment,
    quad_forward_vector,
    run_detector,
    scan_labels,
)

cv2 = pytest.importorskip("cv2")
pytest.importorskip("pupil_apriltags")

NAMES = ["beak", "tail"]
N_FRAMES = 12
SIZE = (240, 320)

#: Printed side of a synthetic tag, in pixels. tag36h11 is 8 modules, so this is
#: 8 px per module — comfortably above the 5 the module advertises.
TAG_SIDE = 64


# ----------------------------------------------------------------------
# Fixtures: a fake frame source and synthetic footage
# ----------------------------------------------------------------------


class FakeFrames:
    """A list of RGB frames with the indexing contract of ``VideoFrameSource``."""

    def __init__(self, frames: list[np.ndarray], scale: float = 1.0):
        self._frames = frames
        self.scale = scale

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return np.stack(self._frames[key])
        return self._frames[key]


def _dictionary(family: str = DEFAULT_TAG_FAMILY):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_DICTIONARIES[family]))


def _noise(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(40, 90, size=(*SIZE, 3), dtype=np.uint8)


def _tag_image(rng, placements: dict[int, tuple[int, int]], side: int = TAG_SIDE) -> np.ndarray:
    """Tags pasted onto noise, each with a white quiet zone the decoder needs."""
    image = _noise(rng)
    quiet = max(side // 8, 3)
    dictionary = _dictionary()
    for tag_id, (x, y) in placements.items():
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, side)
        image[y - quiet : y + side + quiet, x - quiet : x + side + quiet] = 255
        image[y : y + side, x : x + side] = np.repeat(marker[:, :, None], 3, axis=2)
    return image


def _tag_video(n_frames: int = 6) -> tuple[FakeFrames, dict[int, dict[int, tuple[float, float]]]]:
    """Tags 3 and 7 marching left to right; truth is each tag's centre."""
    rng = np.random.default_rng(1)
    frames, truth = [], {}
    for index in range(n_frames):
        placements = {3: (20 + 6 * index, 30), 7: (170 + 4 * index, 140)}
        frames.append(_tag_image(rng, placements))
        truth[index] = {tag: (x + TAG_SIDE / 2 - 0.5, y + TAG_SIDE / 2 - 0.5) for tag, (x, y) in placements.items()}
    return FakeFrames(frames), truth


@pytest.fixture
def store() -> KeypointStore:
    return KeypointStore(keypoint_names=list(NAMES), n_frames=N_FRAMES)


# ----------------------------------------------------------------------
# The contract between the two libraries
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", TAG_FAMILIES)
@pytest.mark.parametrize("marker_id", [0, 1, 7, 29])
def test_an_opencv_rendered_tag_decodes_to_the_same_id(family, marker_id):
    """OpenCV draws the sheet, AprilTag reads it — and nobody checks at runtime.

    ``pupil-apriltags`` exports no generator, so the printed tags come from
    ``cv2.aruco``'s copy of the same code words. Nothing verifies that agreement
    when the GUI runs; a version bump on either library is precisely what would
    break it, and it would show up as a rig where no tag has ever decoded.

    Checked per family, because each is a separate pair of code tables.
    """
    dictionary = _dictionary(family)
    if marker_id >= int(dictionary.bytesList.shape[0]):
        pytest.skip(f"{family} holds fewer than {marker_id + 1} IDs")
    side = family_modules(family) * 12
    pad = 20
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    image = np.full((side + 2 * pad, side + 2 * pad, 3), 255, np.uint8)
    image[pad : pad + side, pad : pad + side] = np.repeat(marker[:, :, None], 3, axis=2)

    found = AprilTagDetector(family=family).detect(image)

    assert len(found) == 1
    assert AprilTagDetector.decode_label(found[0].label) == (marker_id, "centre")


@pytest.mark.parametrize("family", TAG_FAMILIES)
def test_the_module_count_is_derived_and_matches_opencv(family):
    """Two places know a family's grid size; they must never disagree.

    ``family_modules`` parses the family's name (36 bits → 6×6 → 8 with the
    border) rather than tabulating it, while OpenCV reports ``markerSize``. The
    paper size comes from the first and the rendering from the second.
    """
    assert family_modules(family) == int(_dictionary(family).markerSize) + 2


def test_only_allowlisted_families_reach_the_library():
    """An unlisted family **aborts the process** — it does not raise.

    So the guard has to sit in front of the call, not around it. ``tag36h10`` is
    the case that matters in practice: OpenCV renders it happily (2320 IDs), but
    AprilTag 3 dropped the family, so offering it would print tags that nothing
    can ever read.
    """
    for bad in ("tag36h10", "tagStandard52h13", "tagCircle49h12", "", "DICT_4X4_50"):
        with pytest.raises(PointDetectorError):
            check_family(bad)
        with pytest.raises(PointDetectorError):
            AprilTagDetector(family=bad)
    assert "tag36h10" not in TAG_FAMILIES
    assert set(TAG_DICTIONARIES) == set(TAG_FAMILIES)


@pytest.mark.parametrize("family", TAG_FAMILIES)
def test_every_offered_family_has_a_note(family):
    """The dialog states the trade per family, so every entry must have one."""
    assert family_note(family)


@pytest.mark.parametrize("family", TAG_FAMILIES)
def test_no_family_hallucinates_a_clean_read_on_noise(family):
    """The rule that makes the small families offerable at all.

    ``tag16h5`` really does propose tags on noise — 50 across 20 frames in
    testing — but every one of them needed a bit correction, and ``hamming > 0``
    is rejected outright. What survives the rule is what matters.
    """
    rng = np.random.default_rng(11)
    detector = AprilTagDetector(family=family)
    for _trial in range(10):
        blurred = cv2.GaussianBlur(_noise(rng), (5, 5), 0)
        assert detector.detect(blurred) == []


def test_the_corner_order_is_normalised_to_opencvs():
    """``corner_0`` must be the same physical corner it always was.

    ``pupil-apriltags`` reports TR, TL, BL, BR where OpenCV reports TL, TR, BR,
    BL. Getting the remap wrong shifts every saved corner assignment by one
    corner, with no error raised anywhere — so the physical position is pinned
    rather than the index.
    """
    side = 80
    pad = 20
    marker = cv2.aruco.generateImageMarker(_dictionary(), 7, side)
    image = np.full((side + 2 * pad, side + 2 * pad, 3), 255, np.uint8)
    image[pad : pad + side, pad : pad + side] = np.repeat(marker[:, :, None], 3, axis=2)

    detector = AprilTagDetector(parts=("corner_0", "corner_1", "corner_2", "corner_3"))
    corners = {AprilTagDetector.decode_label(d.label)[1]: d.xy for d in detector.detect(image)}

    low, high = pad, pad + side
    for part, expected in (
        ("corner_0", (low, low)),  # top-left
        ("corner_1", (high, low)),  # top-right
        ("corner_2", (high, high)),  # bottom-right
        ("corner_3", (low, high)),  # bottom-left
    ):
        np.testing.assert_allclose(corners[part], expected, atol=2.0, err_msg=part)


def test_the_corner_order_survives_a_rotated_tag():
    """The order is a property of the tag's own frame, not the image's."""
    side = 80
    pad = 20
    marker = cv2.aruco.generateImageMarker(_dictionary(), 7, side)
    image = np.full((side + 2 * pad, side + 2 * pad, 3), 255, np.uint8)
    image[pad : pad + side, pad : pad + side] = np.repeat(marker[:, :, None], 3, axis=2)
    turned = np.rot90(image).copy()

    detector = AprilTagDetector(parts=("corner_0", "corner_1"))
    upright = {AprilTagDetector.decode_label(d.label)[1]: d.xy for d in detector.detect(image)}
    rotated = {AprilTagDetector.decode_label(d.label)[1]: d.xy for d in detector.detect(turned)}

    # A 90° CCW image rotation maps (x, y) -> (y, width - 1 - x); the tag's own
    # corner_0 must follow the tag, not stay in the image's top-left.
    width = image.shape[1]
    expected = (upright["corner_0"][1], width - 1 - upright["corner_0"][0])
    np.testing.assert_allclose(rotated["corner_0"], expected, atol=2.0)


# ----------------------------------------------------------------------
# Quality is a real measurement now
# ----------------------------------------------------------------------


def test_a_clean_tag_scores_far_above_the_default_threshold():
    """The 0.3 default has to sit in a gap, not on a guess."""
    frames, _truth = _tag_video()
    found = AprilTagDetector().detect(frames[0])
    assert found
    assert all(detection.quality > 0.8 for detection in found)


def test_quality_is_the_decode_margin_on_the_stores_zero_to_one_scale():
    """It composes with fill confidence, so it cannot be a raw margin."""
    frames, _truth = _tag_video()
    assert all(0.0 <= d.quality <= 1.0 for d in AprilTagDetector().detect(frames[0]))
    assert GOOD_DECISION_MARGIN > 0


def test_the_downscale_is_part_of_the_size_a_tag_must_be():
    """quad_decimate shrinks the frame *before* anything is looked for."""
    assert AprilTagDetector().min_side_px == family_modules(DEFAULT_TAG_FAMILY) * PX_PER_MODULE
    assert AprilTagDetector(quad_decimate=2.0).min_side_px == 2 * AprilTagDetector().min_side_px


def test_a_smaller_family_needs_less_paper():
    """The entire reason to offer tag16h5: 6 modules against tag36h11's 8."""
    assert AprilTagDetector(family="tag16h5").min_side_px < AprilTagDetector(family="tag36h11").min_side_px


def test_a_decimating_detector_still_reads_a_big_enough_tag():
    frames, _truth = _tag_video()
    assert len(AprilTagDetector(quad_decimate=2.0).detect(frames[0])) == 2


def test_a_downscale_below_one_is_rejected():
    with pytest.raises(PointDetectorError):
        AprilTagDetector(quad_decimate=0.5)


def test_an_unknown_tag_part_is_rejected():
    with pytest.raises(PointDetectorError):
        AprilTagDetector(parts=("centre", "middle"))


def test_one_library_detector_serves_every_setting():
    """A tag36h11 detector costs ~36 MB, so there is one per FAMILY and no more.

    One per parameter combination would put hundreds of megabytes behind a
    dragged spin box, and constructing (and destroying) them is what risks the
    process abort. The settings are struct fields instead; the family is not,
    so it is the one thing that keys the cache.
    """
    assert AprilTagDetector()._detector is AprilTagDetector(quad_decimate=2.0)._detector
    assert AprilTagDetector(family="tag16h5")._detector is AprilTagDetector(family="tag16h5")._detector
    assert AprilTagDetector(family="tag16h5")._detector is not AprilTagDetector(family="tag36h11")._detector


def test_the_settings_reach_the_shared_detector():
    """The corollary: a shared object must still honour each detector's params.

    Also a regression against the library's own constructor, which casts
    ``decode_sharpening`` with ``int()`` — so 0.25 would arrive as 0.0 and the
    Sharpening control would silently do nothing below 1.0.
    """
    rng = np.random.default_rng(21)
    image = _tag_image(rng, {7: (100, 90)}, side=26)

    fine = AprilTagDetector(quad_decimate=1.0, decode_sharpening=0.25)
    coarse = AprilTagDetector(quad_decimate=4.0)

    assert len(fine.detect(image)) == 1
    assert coarse.detect(image) == [], "a 4× downscale must genuinely lose a 26 px tag"
    # And back again on the same shared object, in the same process.
    assert len(fine.detect(image)) == 1
    assert fine._detector.tag_detector_ptr.contents.decode_sharpening == pytest.approx(0.25)


# ----------------------------------------------------------------------
# The store: a third provenance, not a rewrite
# ----------------------------------------------------------------------


def _detection_frame(store: KeypointStore, *points: tuple[str, tuple[float, float]]) -> np.ndarray:
    array = np.full((store.n_individuals, store.n_keypoints, 2), np.nan)
    for keypoint, xy in points:
        array[0, store.keypoint_index(keypoint)] = xy
    return array


def test_detections_are_observations_but_not_labels(store):
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)))})
    assert store.detection_frames() == [4]
    assert store.anchor_frames() == []
    np.testing.assert_allclose(store.positions(4)[0, 0], [5.0, 6.0])
    assert store.detected_mask(4)[0, 0]
    assert not store.human_mask(4)[0, 0]
    assert store.is_detected(4)
    assert not store.is_human(4)


def test_manual_wins_over_detection_on_the_same_point(store):
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)))})
    store.set_point(4, "beak", (99.0, 98.0))
    np.testing.assert_allclose(store.positions(4)[0, 0], [99.0, 98.0])
    np.testing.assert_allclose(store.observations()[4][0, 0], [99.0, 98.0])
    # Still a detection in the raw dict — only the *display* precedence changed.
    assert not store.detected_mask(4)[0, 0]
    assert store.is_human(4)


def test_observations_merge_per_point_not_per_frame(store):
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)), ("tail", (7.0, 8.0)))})
    store.set_point(4, "beak", (1.0, 2.0))
    merged = store.observations()[4]
    np.testing.assert_allclose(merged[0, 0], [1.0, 2.0])
    np.testing.assert_allclose(merged[0, 1], [7.0, 8.0])


def test_flat_observations_feed_the_fill_backends(store):
    from ethograph.gui.pose_fill import SplineBackend

    store.set_detections(
        {
            0: _detection_frame(store, ("beak", (0.0, 0.0))),
            8: _detection_frame(store, ("beak", (80.0, 0.0))),
        }
    )
    filled, confidence = SplineBackend().fill(store.flat_observations(), store.n_frames)
    store.set_fill_from_flat(filled, confidence)
    # The span comes from the detections alone: no frame was labelled by hand.
    assert store.fill_range == (0, 8)
    np.testing.assert_allclose(store.positions(4)[0, 0], [40.0, 0.0], atol=1e-6)


def test_detections_widen_the_fill_span_past_the_manual_anchors(store):
    from ethograph.gui.pose_fill import anchor_span

    store.set_point(3, "beak", (10.0, 10.0))
    store.set_point(5, "beak", (20.0, 10.0))
    assert anchor_span(store.flat_anchors(), store.n_frames) == (3, 5)
    store.set_detections({1: _detection_frame(store, ("beak", (0.0, 0.0)))})
    assert anchor_span(store.flat_observations(), store.n_frames) == (1, 5)


def test_predicted_mask_excludes_detections(store):
    from ethograph.gui.pose_fill import SplineBackend

    store.set_point(0, "beak", (0.0, 0.0))
    store.set_point(8, "beak", (80.0, 0.0))
    store.set_detections({4: _detection_frame(store, ("beak", (44.0, 0.0)))})
    filled, confidence = SplineBackend().fill(store.flat_observations(), store.n_frames)
    store.set_fill_from_flat(filled, confidence)
    assert store.detected_mask(4)[0, 0]
    assert not store.predicted_mask(4)[0, 0]
    assert store.predicted_mask(2)[0, 0]
    # A detection survives the fill verbatim, exactly as an anchor does.
    np.testing.assert_allclose(store.positions(4)[0, 0], [44.0, 0.0])


def test_fill_keeps_the_detectors_own_confidence(store):
    from ethograph.gui.pose_fill import SplineBackend

    store.set_point(0, "beak", (0.0, 0.0))
    store.set_point(8, "beak", (80.0, 0.0))
    quality = np.full((1, 2), np.nan)
    quality[0, 0] = 0.4
    store.set_detections({4: _detection_frame(store, ("beak", (44.0, 0.0)))}, {4: quality})
    filled, confidence = SplineBackend().fill(store.flat_observations(), store.n_frames)
    store.set_fill_from_flat(filled, confidence)
    assert store.confidence[4, 0, 0] == pytest.approx(0.4)
    assert store.confidence_at(0)[0, 0] == 1.0
    assert store.confidence_at(4)[0, 0] == pytest.approx(0.4)


def test_quality_threshold_is_applied_on_write(store):
    quality = np.array([[0.9, 0.2]])
    kept = store.set_detections(
        {4: _detection_frame(store, ("beak", (5.0, 6.0)), ("tail", (7.0, 8.0)))},
        {4: quality},
        quality_min=0.5,
    )
    assert kept == 1
    assert store.detected_mask(4).tolist() == [[True, False]]


def test_clearing_one_frames_detections_keeps_the_labels(store):
    store.set_point(4, "tail", (1.0, 1.0))
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)))})
    assert store.clear_detections_for(4) == 1
    assert store.detection_frames() == []
    assert store.is_anchor(4, "tail")


def test_promote_detections_makes_them_labels(store):
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)), ("tail", (7.0, 8.0)))})
    assert store.promote_detections(4) == 2
    assert store.is_anchor(4, "beak") and store.is_anchor(4, "tail")
    # One undo step each, as if they had been clicked.
    store.undo()
    assert store.labelled_count(4) == 1


def test_promote_all_detections_takes_the_whole_run(store):
    """The bulk verdict: a run that looks right becomes ground truth in one go."""
    for frame in (2, 4, 6):
        merged = dict(store.detections)
        merged[frame] = _detection_frame(store, ("beak", (float(frame), 1.0)), ("tail", (float(frame), 9.0)))
        store.set_detections(merged)
    # One point is already the user's, and a human position always wins.
    store.set_point(4, "beak", (99.0, 99.0))

    assert store.promote_all_detections() == 5
    assert store.anchor_frames() == [2, 4, 6]
    np.testing.assert_allclose(store.anchor_positions(6)[0, 0], [6.0, 1.0])
    np.testing.assert_allclose(store.anchor_positions(4)[0, 0], [99.0, 99.0], err_msg="the label was overwritten")
    assert store.is_human(2) and store.is_human(6)


def test_promote_all_fill_covers_the_whole_span(store):
    from ethograph.gui.pose_fill import SplineBackend

    store.set_point(0, "beak", (0.0, 0.0))
    store.set_point(8, "beak", (80.0, 0.0))
    filled, confidence = SplineBackend().fill(store.flat_anchors(), store.n_frames)
    store.set_fill_from_flat(filled, confidence)
    assert store.fill_range == (0, 8)

    promoted = store.promote_all_fill()

    # The seven interpolated frames between the two labels, and nothing outside.
    assert promoted == 7
    assert store.anchor_frames() == list(range(9))
    assert not store.has_predictions(4)
    np.testing.assert_allclose(store.anchor_positions(4)[0, 0], [40.0, 0.0], atol=1e-6)


def test_a_bulk_approval_is_not_undoable(store):
    """It discards the history rather than leaving a stack nobody can walk back."""
    store.set_point(1, "beak", (1.0, 2.0))
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)))})

    assert store.promote_all_detections() == 1
    assert store.undo() is None
    assert store.is_anchor(4, "beak"), "the promotion itself stands"


def test_a_bulk_approval_with_nothing_to_take_reports_zero(store):
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)))})
    store.promote_all_detections()

    assert store.promote_all_detections() == 0, "already labelled, so nothing left to promote"
    assert store.promote_all_fill() == 0


def test_promote_fill_accepts_a_detection_with_no_fill_loaded(store):
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)))})
    assert store.has_predictions(4)
    assert store.promote_fill(4) == 1
    assert store.is_anchor(4, "beak")


def test_a_detector_run_discards_the_stale_fill(store):
    from ethograph.gui.pose_fill import SplineBackend

    store.set_point(0, "beak", (0.0, 0.0))
    store.set_point(8, "beak", (80.0, 0.0))
    filled, confidence = SplineBackend().fill(store.flat_anchors(), store.n_frames)
    store.set_fill_from_flat(filled, confidence)
    store.set_detections({4: _detection_frame(store, ("beak", (44.0, 0.0)))})
    assert store.filled is None
    assert store.fill_range is None


def test_schema_change_carries_detections_over_by_name(store):
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)), ("tail", (7.0, 8.0)))})
    store.set_keypoint_names(["tail", "beak", "eye"])
    np.testing.assert_allclose(store.positions(4)[0, store.keypoint_index("beak")], [5.0, 6.0])
    np.testing.assert_allclose(store.positions(4)[0, store.keypoint_index("tail")], [7.0, 8.0])


def test_detections_outside_an_asymmetric_schema_are_dropped():
    store = KeypointStore(
        keypoint_names=list(NAMES),
        n_frames=N_FRAMES,
        individual_names=["a", "b"],
        shared_keypoints=False,
        keypoint_sets={"a": ["beak"], "b": ["beak", "tail"]},
    )
    points = np.full((2, 2, 2), np.nan)
    points[0, store.keypoint_index("tail")] = (1.0, 2.0)  # 'a' has no tail
    points[1, store.keypoint_index("tail")] = (3.0, 4.0)
    store.set_detections({2: points})
    assert store.detected_mask(2).tolist() == [[False, False], [False, True]]


def test_detection_cache_round_trip(tmp_path, store):
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)))}, {4: np.array([[0.7, np.nan]])})
    path = detections_path(tmp_path / "clip.mp4")
    assert path.name == "clip.mp4.detections.npz"
    store.save_detections(path, "blob/v1")

    reloaded = KeypointStore(keypoint_names=list(NAMES), n_frames=N_FRAMES)
    assert reloaded.load_detections(path, "blob/v1")
    np.testing.assert_allclose(reloaded.positions(4)[0, 0], [5.0, 6.0])
    assert reloaded.detection_confidence[4][0, 0] == pytest.approx(0.7)
    # Another detector, or another schema, is a miss rather than a lie.
    assert not reloaded.load_detections(path, "apriltag/v1")
    assert not KeypointStore(keypoint_names=["wing"], n_frames=N_FRAMES).load_detections(path, "blob/v1")
    assert not reloaded.load_detections(tmp_path / "nothing.npz", "blob/v1")


def test_detections_stay_out_of_the_json_sidecar(store):
    store.set_point(1, "beak", (1.0, 2.0))
    store.set_detections({4: _detection_frame(store, ("beak", (5.0, 6.0)))})
    payload = store.to_dict()
    assert "detections" not in payload
    assert list(payload["anchors"]) == ["1"]


# ----------------------------------------------------------------------
# Assignment
# ----------------------------------------------------------------------


def test_assignment_round_trips_through_the_sidecar(store):
    store.assignment.set(7, "individual_0", "beak", MANUAL)
    store.assignment.set(2, None, "tail", LEARNED, matched_frames=3)
    reloaded = KeypointStore.from_dict(store.to_dict())
    assert reloaded.assignment.target(7) == ("individual_0", "beak")
    assert reloaded.assignment.get(2).matched_frames == 3
    assert reloaded.assignment.get(2).source == LEARNED


def test_a_sidecar_without_assignments_still_loads(store):
    payload = store.to_dict()
    assert "assignment" not in payload
    assert len(KeypointStore.from_dict(payload).assignment) == 0


def test_two_labels_may_not_share_one_point(store):
    store.assignment.set(1, None, "beak")
    with pytest.raises(AssignmentError):
        store.assignment.set(2, None, "beak")


def test_a_manual_assignment_survives_a_relearn(store):
    store.assignment.set(1, None, "beak", MANUAL)
    store.assignment.set(2, None, "tail", LEARNED)
    taken = store.assignment.learn([Assignment(1, None, "tail"), Assignment(2, None, "beak")])
    # Label 1 is the user's; label 2's proposal collides with it, so neither lands.
    assert taken == 0
    assert store.assignment.target(1) == (None, "beak")
    assert store.assignment.target(2) == (None, "tail")


def test_a_learned_assignment_is_replaced_by_a_relearn(store):
    store.assignment.set(1, None, "beak", LEARNED)
    assert store.assignment.learn([Assignment(1, None, "tail", matched_frames=5)]) == 1
    assert store.assignment.target(1) == (None, "tail")


def test_an_assignment_to_a_deleted_keypoint_is_invalid_not_fatal(store):
    store.assignment.set(1, None, "beak")
    store.assignment.set(2, None, "tail")
    store.set_keypoint_names(["tail"])
    assert store.assignment.invalid_labels(store) == {1}
    # Kept, so the dialog can show it in red — and simply never written to.
    assert store.assignment.target(1) == (None, "beak")
    assert store.assignment_rows() == {2: store.keypoint_index("tail")}


def test_assignment_rows_index_the_flat_point_grid():
    store = KeypointStore(keypoint_names=list(NAMES), n_frames=N_FRAMES, individual_names=["a", "b"])
    store.assignment.set(10, "b", "tail")
    assert store.assignment_rows() == {10: 1 * len(NAMES) + 1}


def test_an_assignment_to_a_deleted_individual_is_invalid():
    store = KeypointStore(keypoint_names=list(NAMES), n_frames=N_FRAMES, individual_names=["a", "b"])
    store.assignment.set(1, "b", "beak")
    store.set_individual_names(["a"])
    assert store.assignment.invalid_labels(store) == {1}
    assert store.assignment_rows() == {}


# ----------------------------------------------------------------------
# Head direction from tag corners
# ----------------------------------------------------------------------


#: An upright tag in image coordinates (y DOWN), in cv2 corner order:
#: TL, TR, BR, BL. Its printed top edge faces the top of the frame.
_UPRIGHT_QUAD = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])


def _rotated(quad: np.ndarray, degrees: float) -> np.ndarray:
    """*quad* turned about its own centre, corner order preserved."""
    theta = np.radians(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centre = quad.mean(axis=0)
    return (quad - centre) @ rotation.T + centre


def test_an_upright_tag_faces_the_top_of_the_frame():
    """The printed top edge is the front, and y grows downward in an image."""
    forward = quad_forward_vector(_UPRIGHT_QUAD)
    np.testing.assert_allclose(forward, [0.0, -1.0], atol=1e-9)


def test_the_forward_vector_is_a_unit_vector():
    for degrees in (0.0, 17.0, 90.0, 213.0):
        forward = quad_forward_vector(_rotated(_UPRIGHT_QUAD, degrees))
        np.testing.assert_allclose(np.hypot(*forward), 1.0)


@pytest.mark.parametrize("degrees", [0.0, 30.0, 90.0, 180.0, 270.0])
def test_the_forward_vector_turns_with_the_tag(degrees):
    """A tag is a rigid square, so its heading is its rotation — no more, no less."""
    forward = quad_forward_vector(_rotated(_UPRIGHT_QUAD, degrees))
    # Image coordinates are left-handed (y down), so a positive rotation there
    # is a clockwise turn of the vector in the same frame.
    theta = np.radians(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    np.testing.assert_allclose(forward, rotation @ np.array([0.0, -1.0]), atol=1e-9)


def test_the_forward_vector_is_perpendicular_to_the_front_edge():
    quad = _rotated(_UPRIGHT_QUAD, 37.0)
    edge = quad[1] - quad[0]
    np.testing.assert_allclose(np.dot(quad_forward_vector(quad), edge), 0.0, atol=1e-9)


def test_a_degenerate_quad_has_no_orientation():
    """A missing measurement, not a zero direction."""
    collapsed = np.zeros((4, 2))
    assert quad_forward_vector(collapsed) is None


def test_a_decoded_tag_carries_its_orientation():
    """Every part of a tag reports the SAME heading: the square is what faces a
    direction, not any one of its corners. This is the whole reason a head
    direction needs one keypoint rather than a nominated pair of them."""
    image = _tag_image(np.random.default_rng(0), {7: (60, 60)})
    detections = AprilTagDetector(parts=("centre", "corner_0", "corner_1")).detect(image)
    assert detections
    headings = [d.orientation for d in detections]
    assert all(h is not None for h in headings)
    for heading in headings[1:]:
        np.testing.assert_allclose(heading, headings[0], atol=1e-9)
    # Rendered upright, so it faces the top of the frame.
    np.testing.assert_allclose(headings[0], [0.0, -1.0], atol=0.02)


# ----------------------------------------------------------------------
# run_detector
# ----------------------------------------------------------------------


class ConstantDetector:
    """Emits a fixed set of detections on every frame."""

    name = "constant"

    def __init__(self, detections: list[Detection]):
        self._detections = detections

    def detect(self, frame):
        return list(self._detections)


def test_scan_counts_each_label_once_per_frame_and_samples_the_span():
    """A tag emitting its centre and corners on one frame is seen once, not five times."""
    frames = FakeFrames([np.zeros((10, 10, 3), np.uint8)] * 100)
    detector = ConstantDetector([Detection((1.0, 1.0), 8), Detection((2.0, 2.0), 8), Detection((3.0, 3.0), 9)])
    scanned = scan_labels(detector, frames, span=(10, 29), n_samples=5)
    assert scanned.frames_scanned == 5
    assert scanned.counts == {8: 5, 9: 5}

    cancelled = scan_labels(detector, frames, n_samples=5, progress=lambda _f: False)
    assert cancelled.frames_scanned == 0 and cancelled.seen() == []


def test_scan_streams_when_the_source_can():
    """The forward pass is the point: one seek, not one per sampled frame."""

    class Streaming(FakeFrames):
        def __init__(self, frames):
            super().__init__(frames)
            self.asked: list[int] = []

        def iter_frames(self, indices, gray=False, progress=None):
            for i in indices:
                self.asked.append(int(i))
                yield int(i), self._frames[int(i)]

        def __getitem__(self, key):
            raise AssertionError("random access while a stream was offered")

    frames = Streaming([np.zeros((4, 4, 3), np.uint8)] * 50)
    scan_labels(ConstantDetector([]), frames, n_samples=4)
    assert frames.asked == [0, 16, 33, 49]


def test_a_single_read_is_a_misread_until_a_second_frame_agrees():
    scanned = ScannedLabels(counts={8: 1, 16: 7}, frames_scanned=10)
    assert scanned.seen() == [16]


def test_one_tag_per_animal_needs_no_label():
    """Two individuals, one keypoint each, two tags seen: paired lowest-first, by order."""
    store = KeypointStore(keypoint_names=["tag"], individual_names=["a", "b"], n_frames=10)
    detector = AprilTagDetector()
    scanned = ScannedLabels(counts={detector.label_for(7): 30, detector.label_for(3): 28}, frames_scanned=40)
    found = propose_assignment(detector, scanned, store)
    assert [(a.label, a.individual, a.keypoint, a.source) for a in found.proposals] == [
        (detector.label_for(3), "a", "tag", SCANNED),
        (detector.label_for(7), "b", "tag", SCANNED),
    ]
    assert found.unmatched_labels == [] and found.unmatched_targets == []
    assert found.proposals[0].matched_frames == 28


def test_the_tags_seen_most_win_when_there_are_more_tags_than_animals():
    """A spurious decode on two frames must not take an individual from a real tag."""
    store = KeypointStore(keypoint_names=["tag"], individual_names=["a"], n_frames=10)
    detector = AprilTagDetector()
    scanned = ScannedLabels(counts={detector.label_for(1): 2, detector.label_for(9): 35}, frames_scanned=40)
    found = propose_assignment(detector, scanned, store)
    assert [a.label for a in found.proposals] == [detector.label_for(9)]
    assert found.unmatched_labels == [detector.label_for(1)]


def test_corners_go_to_keypoints_named_after_them_and_the_rest_by_order():
    store = KeypointStore(keypoint_names=["Corner 2", "body", "center"], individual_names=["a"], n_frames=10)
    detector = AprilTagDetector(parts=TAG_PARTS)
    counts = {detector.label_for(5, part): 20 for part in TAG_PARTS}
    found = propose_assignment(detector, ScannedLabels(counts=counts, frames_scanned=20), store)
    by_keypoint = {a.keypoint: detector.decode_label(a.label)[1] for a in found.proposals}
    # Named parts first, whatever their position; the nameless keypoint takes
    # the lowest part still free.
    assert by_keypoint == {"Corner 2": "corner_2", "center": "centre", "body": "corner_0"}
    assert [detector.decode_label(label)[1] for label in found.unmatched_labels] == ["corner_1", "corner_3"]


def test_fewer_tags_than_animals_names_who_is_missing():
    store = KeypointStore(keypoint_names=["tag"], individual_names=["a", "b", "c"], n_frames=10)
    detector = AprilTagDetector()
    found = propose_assignment(detector, ScannedLabels(counts={detector.label_for(4): 9}, frames_scanned=9), store)
    assert [(a.individual) for a in found.proposals] == ["a"]
    assert found.unmatched_targets == [("b", "tag"), ("c", "tag")]


def test_a_detector_without_parts_pairs_whole_labels():
    store = KeypointStore(keypoint_names=["tag"], individual_names=["a", "b"], n_frames=10)
    detector = ConstantDetector([])
    assert marker_part(detector, 42) == (42, 0)
    assert part_named(detector, "centre") is None
    found = propose_assignment(detector, ScannedLabels(counts={42: 5, 41: 5}, frames_scanned=5), store)
    assert [(a.label, a.individual) for a in found.proposals] == [(41, "a"), (42, "b")]


def test_a_scanned_row_is_replaced_by_a_learn_and_never_a_manual_one(store):
    store.assignment.learn([Assignment(8, None, NAMES[0], SCANNED)])
    assert store.assignment.get(8).source == SCANNED
    store.assignment.learn([Assignment(8, None, NAMES[1], LEARNED)])
    assert store.assignment.get(8).keypoint == NAMES[1]
    store.assignment.set(8, None, NAMES[0], MANUAL)
    store.assignment.learn([Assignment(8, None, NAMES[1], SCANNED)])
    assert store.assignment.get(8).keypoint == NAMES[0]
    assert Assignment.from_dict(store.assignment.get(8).to_dict()) == store.assignment.get(8)


def test_run_detector_writes_flat_rows_and_skips_unassigned_labels():
    frames = FakeFrames([np.zeros((10, 10, 3), np.uint8)] * 4)
    detector = ConstantDetector([Detection((3.0, 4.0), 1, 0.8), Detection((5.0, 6.0), 99)])
    positions, quality, _orientation = run_detector(detector, frames, {1: 2}, n_points=4)
    assert sorted(positions) == [0, 1, 2, 3]
    np.testing.assert_allclose(positions[0][2], [3.0, 4.0])
    assert np.isnan(positions[0][[0, 1, 3]]).all()
    assert quality[0][2] == pytest.approx(0.8)


def test_run_detector_drops_a_label_found_twice_in_one_frame():
    frames = FakeFrames([np.zeros((10, 10, 3), np.uint8)] * 2)
    detector = ConstantDetector([Detection((3.0, 4.0), 1), Detection((7.0, 8.0), 1)])
    positions, _quality, _orientation = run_detector(detector, frames, {1: 0}, n_points=1)
    assert positions == {}


def test_run_detector_scales_back_to_source_pixels():
    frames = FakeFrames([np.zeros((10, 10, 3), np.uint8)] * 2, scale=4.0)
    detector = ConstantDetector([Detection((3.0, 4.0), 1)])
    positions, _quality, _orientation = run_detector(detector, frames, {1: 0}, n_points=1)
    np.testing.assert_allclose(positions[0][0], [12.0, 16.0])


def test_run_detector_honours_the_span_and_cancellation():
    frames = FakeFrames([np.zeros((10, 10, 3), np.uint8)] * 200)
    detector = ConstantDetector([Detection((1.0, 1.0), 1)])
    positions, _quality, _orientation = run_detector(detector, frames, {1: 0}, n_points=1, span=(10, 19))
    assert sorted(positions) == list(range(10, 20))

    positions, _quality, _orientation = run_detector(detector, frames, {1: 0}, n_points=1, progress=lambda _f: False)
    assert positions == {}


def test_the_chunk_size_follows_the_frame_size():
    """Detection decodes at full resolution, so 64 frames of 4K is 1.6 GB."""
    small = FakeFrames([np.zeros((4, 4, 3), np.uint8)])
    small.size = (320, 240)
    assert chunk_frames(small) == CHUNK_FRAMES, "a small frame keeps the old fixed count"

    uhd = FakeFrames([np.zeros((4, 4, 3), np.uint8)])
    uhd.size = (3840, 2160)
    assert 1 <= chunk_frames(uhd) < CHUNK_FRAMES
    assert chunk_frames(uhd) * 3840 * 2160 * 3 <= CHUNK_BYTES

    enormous = FakeFrames([np.zeros((4, 4, 3), np.uint8)])
    enormous.size = (20000, 20000)
    assert chunk_frames(enormous) == 1, "never zero, whatever the budget says"


def test_a_frame_source_without_a_size_still_chunks():
    """`size` is a courtesy of VideoFrameSource, not part of the contract."""
    assert chunk_frames(FakeFrames([np.zeros((4, 4, 3), np.uint8)])) == CHUNK_FRAMES


@pytest.mark.parametrize("family", TAG_FAMILIES)
def test_the_pixel_budget_is_the_decoders_own_figure(family):
    """The printed-size advice must not drift from what decoding needs."""
    from ethograph.gui.pose_tagsheet import MIN_PX_PER_MODULE, dictionary_info

    assert MIN_PX_PER_MODULE == PX_PER_MODULE
    assert dictionary_info(family).modules == family_modules(family) == AprilTagDetector(family=family).modules


def test_run_detector_with_no_detections_anywhere():
    frames = FakeFrames([np.zeros((10, 10, 3), np.uint8)] * 3)
    positions, quality, _orientation = run_detector(ConstantDetector([]), frames, {1: 0}, n_points=1)
    assert positions == {} and quality == {}


# ----------------------------------------------------------------------
# Learning what a tag means
# ----------------------------------------------------------------------


def _tag_store(frames, truth, label_frames=(0, 2, 4)) -> KeypointStore:
    """A store with tag 3 labelled as bee_03's thorax on *label_frames*."""
    store = KeypointStore(keypoint_names=["thorax"], n_frames=len(frames), individual_names=["bee_03"])
    for frame in label_frames:
        store.set_point(frame, "thorax", truth[frame][3])
    return store


def test_learning_needs_more_than_one_agreeing_frame():
    """The nearest detection to a single click is *always* something."""
    frames, truth = _tag_video()
    store = _tag_store(frames, truth, label_frames=(0,))

    learned = learn_assignment(AprilTagDetector(), frames, store)

    assert learned.proposals == []
    assert AprilTagDetector.label_for(3) in learned.unmatched_labels
    assert learned.unmatched_targets == [("bee_03", "thorax")]


def test_learning_rejects_a_match_beyond_the_radius():
    frames, truth = _tag_video()
    store = KeypointStore(keypoint_names=["thorax"], n_frames=len(frames), individual_names=["bee_03"])
    for frame in (0, 2, 4):
        store.set_point(frame, "thorax", (truth[frame][3][0] + 90, truth[frame][3][1]))

    assert learn_assignment(AprilTagDetector(), frames, store, radius_px=5.0).proposals == []


def test_no_labels_means_no_assignment_can_be_learned():
    frames, _truth = _tag_video()
    store = KeypointStore(keypoint_names=["thorax"], n_frames=len(frames))

    learned = learn_assignment(AprilTagDetector(), frames, store)

    assert learned.proposals == [] and learned.frames_scanned == 0


def test_a_detector_survives_single_channel_footage():
    """Infrared and machine-vision cameras hand over ``(H, W)``, not RGB.

    ``pupil-apriltags`` asserts ``ndim == 2`` with a bare, message-less
    ``AssertionError``, so both the mono and the colour path have to be
    normalised before it ever sees a frame.
    """
    rng = np.random.default_rng(3)
    mono = _noise(rng)[:, :, 0]
    rgba = np.dstack([_noise(rng), np.full(SIZE, 255, np.uint8)])

    assert AprilTagDetector().detect(mono) == []
    assert AprilTagDetector().detect(rgba) == []

    frames, _truth = _tag_video()
    assert AprilTagDetector().detect(frames[0][:, :, 0]), "a mono frame with a tag still decodes"


# ----------------------------------------------------------------------
# Tags
# ----------------------------------------------------------------------


def test_the_detector_recovers_tag_centres():
    frames, truth = _tag_video()
    detector = AprilTagDetector()
    found = {d.label: d for d in detector.detect(frames[2])}
    assert set(found) == {AprilTagDetector.label_for(3), AprilTagDetector.label_for(7)}
    for tag_id, centre in truth[2].items():
        np.testing.assert_allclose(found[AprilTagDetector.label_for(tag_id)].xy, centre, atol=1.5)


def test_tag_labels_encode_the_part_stably():
    assert AprilTagDetector.label_for(7) == 7 * TAG_PART_STRIDE
    assert AprilTagDetector.decode_label(AprilTagDetector.label_for(7, "corner_2")) == (7, "corner_2")
    # The encoding does not shift when other parts are enabled.
    assert AprilTagDetector(parts=("centre", "corner_0")).label_for(7) == 7 * TAG_PART_STRIDE


def test_corners_become_four_keypoints():
    frames, _truth = _tag_video()
    detector = AprilTagDetector(parts=("corner_0", "corner_1", "corner_2", "corner_3"))
    found = [d for d in detector.detect(frames[0]) if AprilTagDetector.decode_label(d.label)[0] == 3]
    assert len(found) == 4
    quad = np.array([d.xy for d in found])
    assert np.ptp(quad[:, 0]) > 30 and np.ptp(quad[:, 1]) > 30


def test_detection_survives_a_homography():
    """Warping by a known homography gives ground truth for sub-pixel recovery."""
    rng = np.random.default_rng(4)
    image = _tag_image(rng, {5: (80, 80)})
    homography = np.array([[1.0, 0.12, 5.0], [0.05, 1.0, -3.0], [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(image, homography, (SIZE[1], SIZE[0]))

    centre = np.array([80 + TAG_SIDE / 2 - 0.5, 80 + TAG_SIDE / 2 - 0.5, 1.0])
    expected = homography @ centre
    expected = expected[:2] / expected[2]

    found = AprilTagDetector().detect(warped)
    assert len(found) == 1
    np.testing.assert_allclose(found[0].xy, expected, atol=1.5)


def test_one_individual_is_learned_per_tag():
    frames, truth = _tag_video()
    store = KeypointStore(keypoint_names=["thorax"], n_frames=len(frames), individual_names=["bee_03", "bee_07"])
    for frame in (0, 2, 4):
        store.set_point(frame, "thorax", truth[frame][3], "bee_03")
        store.set_point(frame, "thorax", truth[frame][7], "bee_07")

    detector = AprilTagDetector()
    learned = learn_assignment(detector, frames, store)
    assert {(a.label, a.individual) for a in learned.proposals} == {
        (AprilTagDetector.label_for(3), "bee_03"),
        (AprilTagDetector.label_for(7), "bee_07"),
    }
    store.assignment.learn(learned.proposals)
    positions, quality, _orientation = run_detector(detector, frames, store.assignment_rows(), store.n_points)
    store.set_detections_from_flat(positions, quality)
    np.testing.assert_allclose(store.positions(5)[1, 0], truth[5][7], atol=1.5)


# ----------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------


def test_build_detector_and_availability():
    """One detector today; the list is the seam a second arrives through."""
    assert isinstance(build_detector(APRILTAG_DETECTOR), PointDetector)
    with pytest.raises(PointDetectorError):
        build_detector("magic")
    assert [info.key for info in available_detectors()] == [APRILTAG_DETECTOR]
    assert all(info.available for info in available_detectors())


def test_a_family_is_named_in_the_detectors_own_name():
    """The combo shows one detector; its name has to say which family it is on."""
    assert "tag16h5" in AprilTagDetector(family="tag16h5").name
    assert "tag36h11" in AprilTagDetector().name


def test_labels_describe_themselves_for_the_dialog():
    tags = AprilTagDetector()
    assert label_name(tags, AprilTagDetector.label_for(7)) == "tag 7"
    assert label_preview(tags, AprilTagDetector.label_for(7)).shape == (32, 32, 3)
    assert label_name(ConstantDetector([]), 4) == "label 4"
    assert label_preview(ConstantDetector([]), 4) is None


# ----------------------------------------------------------------------
# The tuning preview
# ----------------------------------------------------------------------


def test_diagnose_agrees_with_detect_on_what_was_kept():
    """A preview that can disagree with the detector would be worse than none."""
    frames, _truth = _tag_video()
    detector = AprilTagDetector()
    preview = detector.diagnose(frames[2])

    kept = sorted((shape.label, tuple(np.round(shape.xy, 6))) for shape in preview.accepted)
    assert kept == sorted((found.label, tuple(np.round(found.xy, 6))) for found in detector.detect(frames[2]))
    assert preview.size == (SIZE[1], SIZE[0])
    assert all(shape.outline is not None for shape in preview.shapes)


def test_a_corrected_read_is_rejected_and_says_so(monkeypatch):
    """A wrong ID is the one failure that survives every later stage.

    Real footage producing a hamming > 0 read on demand is not something a unit
    test can stage, so the boundary is checked at the seam: whatever the library
    reports, a non-zero correction count never becomes a detection, and the
    preview still draws it — captioned with the measured side, which is the one
    number worth acting on — rather than dropping it silently.
    """
    frames, _truth = _tag_video()
    detector = AprilTagDetector()
    real = detector._decode

    def _one_corrected(frame):
        image, found = real(frame)
        return image, [(tag_id, quad, quality, 2) for tag_id, quad, quality, _h in found]

    monkeypatch.setattr(detector, "_decode", _one_corrected)

    assert detector.detect(frames[0]) == []
    preview = detector.diagnose(frames[0])
    assert preview.accepted == []
    assert preview.rejected
    assert all(shape.reason.endswith(" px") for shape in preview.rejected)
    assert all(shape.label is None for shape in preview.rejected)


def test_diagnose_shows_nothing_when_there_is_no_tag():
    """The other failure: no shape at all, so the size budget is the diagnosis."""
    rng = np.random.default_rng(7)
    assert AprilTagDetector().diagnose(_noise(rng)).shapes == []


def test_diagnose_falls_back_to_plain_detections():
    """A detector with no diagnose() still previews — as what it accepted."""
    preview = diagnose_frame(ConstantDetector([Detection((3.0, 4.0), 1, 0.5)]), np.zeros((8, 8, 3), np.uint8))
    assert [shape.label for shape in preview.accepted] == [1]
    assert preview.rejected == []


def test_assignment_table_equality_and_ordering():
    table = AssignmentTable([Assignment(3, None, "tail"), Assignment(1, None, "beak")])
    assert [entry.label for entry in table] == [1, 3]
    assert table == AssignmentTable.from_list(table.to_list())
    assert table.remove(1) and not table.remove(1)
    table.clear()
    assert len(table) == 0
