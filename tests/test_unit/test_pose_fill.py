"""Fill backend protocol conformance and the anchor-preservation invariant.

Backend tests use a stand-in gap tracker, so nothing heavy is needed in CI.
The assertion that matters for every backend is the same:
**anchor frames come back exactly as they were labelled.**
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui import pose_fill
from ethograph.gui.pose_fill import (
    FillBackend,
    OpticalFlowBackend,
    SplineBackend,
    _GapBackend,
    available_backends,
    build_backend,
    video_size,
)

N_FRAMES = 21
N_KEYPOINTS = 2


def _anchors() -> dict[int, np.ndarray]:
    return {
        0: np.array([[0.0, 0.0], [10.0, 10.0]]),
        10: np.array([[10.0, 5.0], [20.0, 15.0]]),
        20: np.array([[20.0, 0.0], [30.0, 10.0]]),
    }


def _partial_anchors() -> dict[int, np.ndarray]:
    """Beak labelled on some frames, tail on others — the normal case."""
    return {
        0: np.array([[0.0, 0.0], [10.0, 10.0]]),
        7: np.array([[7.0, 3.0], [np.nan, np.nan]]),
        14: np.array([[np.nan, np.nan], [24.0, 12.0]]),
        20: np.array([[20.0, 0.0], [30.0, 10.0]]),
    }


class _HoldBackend(_GapBackend):
    """Exercises the shared gap machinery without torch or OpenCV.

    ``_track`` holds every query point still, so blending, scaling and endpoint
    seeding are observable in isolation from any tracker.
    """

    name = "hold"

    def _track(self, clip, points, query_frame):
        positions = np.repeat(np.asarray(points)[None], len(clip), axis=0)
        return positions.astype(np.float64), np.ones((len(clip), len(points)))


class _ScaledFrames:
    """Frame source decoded at half resolution (``scale`` back to source px)."""

    scale = 2.0

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


def _frames(n: int = N_FRAMES, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(n, size, size, 3), dtype=np.uint8)


# ----------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------


def test_backends_satisfy_the_protocol():
    for backend in (SplineBackend(), OpticalFlowBackend()):
        assert isinstance(backend, FillBackend)
        assert isinstance(backend.name, str)
        assert isinstance(backend.requires_video, bool)


def test_spline_needs_no_video():
    assert SplineBackend().requires_video is False
    assert OpticalFlowBackend().requires_video is True


def test_available_backends_always_offers_spline():
    infos = {info.key: info for info in available_backends()}
    assert infos["spline"].available is True
    assert infos["flow"].available or infos["flow"].hint


def test_build_backend_rejects_unknown_key():
    with pytest.raises(ValueError):
        build_backend("nope")


# ----------------------------------------------------------------------
# The shared invariant
# ----------------------------------------------------------------------


def _spline_backend():
    return SplineBackend(), None


def _gap_backend():
    return _HoldBackend(), _frames()


def _optical_flow_backend():
    pytest.importorskip("cv2")
    return OpticalFlowBackend(), _frames()


#: Every backend, built the cheapest way that still runs its real fill path.
BACKENDS = [
    pytest.param(_spline_backend, id="spline"),
    pytest.param(_gap_backend, id="gap"),
    pytest.param(_optical_flow_backend, id="optical-flow"),
]


@pytest.mark.parametrize("make_backend", BACKENDS)
def test_every_backend_returns_anchor_frames_exactly_as_labelled(make_backend):
    """The one assertion that matters for all of them — see the module docstring.

    Parametrized rather than repeated per backend: it is a property of the
    *protocol*, so a backend added later inherits the check by joining the list
    instead of by someone remembering to copy the test.
    """
    backend, frames = make_backend()
    anchors = _anchors()
    filled, confidence = backend.fill(anchors, N_FRAMES, frames)

    assert filled.shape == (N_FRAMES, N_KEYPOINTS, 2)
    assert confidence.shape == (N_FRAMES, N_KEYPOINTS)
    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
        np.testing.assert_allclose(confidence[frame], 1.0)


# ----------------------------------------------------------------------
# Spline
# ----------------------------------------------------------------------


def test_spline_fills_every_frame():
    filled, _ = SplineBackend().fill(_anchors(), N_FRAMES, None)
    assert not np.any(np.isnan(filled))


def test_spline_interpolates_between_anchors():
    filled, _ = SplineBackend().fill(_anchors(), N_FRAMES, None)
    # x moves monotonically 0 -> 20, so the midpoint sits strictly between.
    assert 0.0 < filled[5, 0, 0] < 10.0


def test_spline_confidence_decays_away_from_anchors():
    _, confidence = SplineBackend().fill(_anchors(), N_FRAMES, None)
    assert confidence[5, 0] < confidence[1, 0] < 1.0
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))


def test_spline_handles_partial_anchors():
    anchors = _partial_anchors()
    filled, confidence = SplineBackend().fill(anchors, N_FRAMES, None)

    for frame, points in anchors.items():
        labelled = ~np.isnan(points[:, 0])
        np.testing.assert_allclose(filled[frame][labelled], points[labelled])
        np.testing.assert_allclose(confidence[frame][labelled], 1.0)
    assert not np.any(np.isnan(filled))


def test_spline_leaves_frames_outside_the_anchored_span_empty():
    """Only the gaps between labels are filled — never past the outermost one."""
    anchors = {5: np.array([[5.0, 5.0], [1.0, 1.0]]), 15: np.array([[15.0, 5.0], [2.0, 1.0]])}
    filled, confidence = SplineBackend().fill(anchors, N_FRAMES, None)

    assert np.all(np.isnan(filled[:5]))
    assert np.all(np.isnan(filled[16:]))
    assert np.all(np.isnan(confidence[:5]))
    assert np.all(np.isnan(confidence[16:]))
    assert not np.any(np.isnan(filled[5:16]))


def test_spline_with_a_single_anchor_fills_only_that_frame():
    """One label brackets nothing, so there is no gap to interpolate across."""
    anchors = {4: np.array([[3.0, 4.0], [5.0, 6.0]])}
    filled, _ = SplineBackend().fill(anchors, N_FRAMES, None)
    np.testing.assert_allclose(filled[4], anchors[4])
    assert np.all(np.isnan(np.delete(filled, 4, axis=0)))


def test_spline_holds_a_keypoint_labelled_once_across_the_span():
    """Within the span a point with a single label is held, not dropped.

    It is what seeds the gap backends' endpoints for that keypoint.
    """
    anchors = {
        0: np.array([[0.0, 0.0], [5.0, 6.0]]),
        20: np.array([[20.0, 0.0], [np.nan, np.nan]]),
    }
    filled, _ = SplineBackend().fill(anchors, N_FRAMES, None)
    np.testing.assert_allclose(filled[:, 1], np.tile([5.0, 6.0], (N_FRAMES, 1)))


def test_unlabelled_keypoint_stays_nan():
    anchors = {
        0: np.array([[1.0, 1.0], [np.nan, np.nan]]),
        20: np.array([[2.0, 1.0], [np.nan, np.nan]]),
    }
    filled, confidence = SplineBackend().fill(anchors, N_FRAMES, None)
    assert np.all(np.isnan(filled[:, 1]))
    assert np.all(confidence[:, 1] == 0.0)


def test_anchor_span_reports_the_outermost_labelled_frames():
    assert pose_fill.anchor_span(_anchors(), N_FRAMES) == (0, 20)
    assert pose_fill.anchor_span({7: np.array([[1.0, 1.0]])}, N_FRAMES) == (7, 7)
    # Frames past the end of the video, and rows carrying no point, are not labels.
    assert pose_fill.anchor_span({99: np.array([[1.0, 1.0]])}, N_FRAMES) is None
    assert pose_fill.anchor_span({3: np.array([[np.nan, np.nan]])}, N_FRAMES) is None
    assert pose_fill.anchor_span({}, N_FRAMES) is None


def test_fill_without_anchors_raises():
    with pytest.raises(ValueError):
        SplineBackend().fill({}, N_FRAMES, None)


def test_progress_cancellation_stops_early():
    calls = []

    def progress(fraction):
        calls.append(fraction)
        return False

    SplineBackend().fill(_anchors(), N_FRAMES, None, progress)
    assert len(calls) == 1


# ----------------------------------------------------------------------
# Gap backends
# ----------------------------------------------------------------------


def test_gap_backend_requires_frames():
    with pytest.raises(ValueError):
        OpticalFlowBackend().fill(_anchors(), N_FRAMES, None)


def test_gap_backend_blends_linearly_across_the_gap():
    anchors = {0: np.array([[0.0, 0.0]]), 10: np.array([[10.0, 0.0]])}
    filled, _ = _HoldBackend().fill(anchors, 11, _frames(11))
    np.testing.assert_allclose(filled[5, 0], [5.0, 0.0])
    np.testing.assert_allclose(filled[2, 0], [2.0, 0.0])


def test_gap_backend_seeds_missing_endpoints_from_the_spline():
    """A keypoint unlabelled on a gap endpoint is still filled, not left NaN."""
    anchors = _partial_anchors()
    filled, confidence = _HoldBackend().fill(anchors, N_FRAMES, _frames())

    for frame, points in anchors.items():
        labelled = ~np.isnan(points[:, 0])
        np.testing.assert_allclose(filled[frame][labelled], points[labelled])
        np.testing.assert_allclose(confidence[frame][labelled], 1.0)
    assert not np.any(np.isnan(filled))


def test_gap_backend_honours_frame_source_scale():
    """A downscaled frame source must not shrink the returned coordinates."""
    anchors = _anchors()
    filled, _ = _HoldBackend().fill(anchors, N_FRAMES, _ScaledFrames(_frames()))

    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)
    # Midpoint of a held track between (0, 0) and (10, 5), in source pixels.
    np.testing.assert_allclose(filled[5, 0], [5.0, 2.5])


def test_gap_backend_leaves_frames_outside_the_anchored_span_empty():
    """The same rule as the spline: a fill stops at the outermost labels."""
    anchors = {5: np.array([[5.0, 5.0]]), 15: np.array([[15.0, 5.0]])}
    filled, confidence = _HoldBackend().fill(anchors, N_FRAMES, _frames())

    assert not np.any(np.isnan(filled[5:16]))
    assert np.all(np.isnan(filled[:5]))
    assert np.all(np.isnan(filled[16:]))
    assert np.all(np.isnan(confidence[16:]))


class _NaNIntolerantBackend(_GapBackend):
    """A tracker that cannot be handed a NaN query.

    A transformer tracker attends jointly across points, so one NaN query row
    returns NaN for *every* point. Reproduced here as an assertion rather than by spreading
    NaN, so the test names the contract it is protecting.
    """

    name = "nan-intolerant"

    def _track(self, clip, points, query_frame):
        assert np.isfinite(points).all(), "a NaN query reached the tracker"
        positions = np.repeat(np.asarray(points)[None], len(clip), axis=0)
        return positions.astype(np.float64), np.ones((len(clip), len(points)))


def test_a_point_labelled_nowhere_never_reaches_the_tracker():
    """It has no spline seed either, so its endpoints are NaN.

    Passing it through blanked every *other* point across every tracked gap,
    leaving only the untracked head and tail filled.
    """
    anchors = {
        0: np.array([[0.0, 0.0], [np.nan, np.nan]]),
        10: np.array([[10.0, 5.0], [np.nan, np.nan]]),
    }

    filled, confidence = _NaNIntolerantBackend().fill(anchors, N_FRAMES, _frames())

    # The labelled point is tracked across the gap it was seeded on...
    assert not np.any(np.isnan(filled[:11, 0]))
    np.testing.assert_allclose(filled[5, 0], [5.0, 2.5])
    assert confidence[5, 0] > 0.0
    # ...and the one labelled nowhere stays empty rather than inventing a track.
    assert np.all(np.isnan(filled[:, 1]))
    np.testing.assert_allclose(confidence[:11, 1], 0.0)


def test_a_gap_with_nothing_trackable_keeps_the_seed():
    """Every point unlabelled on both endpoints: the gap is skipped, not crashed."""
    anchors = {0: np.array([[np.nan, np.nan]]), 10: np.array([[np.nan, np.nan]])}

    filled, _ = _NaNIntolerantBackend().fill(anchors, N_FRAMES, _frames())
    assert np.all(np.isnan(filled))


class _DriftBackend(_GapBackend):
    """Forward and backward tracks that disagree by a known number of pixels."""

    name = "drift"
    DRIFT = 10.0

    def _track(self, clip, points, query_frame):
        offset = 0.0 if query_frame == 0 else self.DRIFT
        positions = np.repeat((np.asarray(points) + offset)[None], len(clip), axis=0)
        return positions.astype(np.float64), np.ones((len(clip), len(points)))


def test_disagreement_tolerance_scales_the_confidence():
    """The confidence knob: how far the two tracks may drift before it counts."""
    anchors = {0: np.array([[0.0, 0.0]]), 10: np.array([[0.0, 0.0]])}

    strict, strict_confidence = _DriftBackend(disagreement_px=1.0).fill(anchors, 11, _frames(11))
    lenient, lenient_confidence = _DriftBackend(disagreement_px=100.0).fill(anchors, 11, _frames(11))

    # Same positions either way — the tolerance only scores them.
    np.testing.assert_allclose(strict, lenient)
    drift = np.hypot(_DriftBackend.DRIFT, _DriftBackend.DRIFT)
    assert strict_confidence[5, 0] == pytest.approx(np.exp(-drift / 1.0))
    assert lenient_confidence[5, 0] == pytest.approx(np.exp(-drift / 100.0))


def test_disagreement_tolerance_must_be_positive():
    with pytest.raises(ValueError):
        OpticalFlowBackend(disagreement_px=0.0)


def test_build_backend_passes_the_tolerance_through():
    assert build_backend("flow", disagreement_px=42.0)._disagreement == 42.0


def test_gap_backend_cancellation_stops_early():
    anchors = _anchors()
    filled, _ = _HoldBackend().fill(anchors, N_FRAMES, _frames(), lambda _f: False)
    # Anchors still hold even when the user cancels mid-fill.
    for frame, points in anchors.items():
        np.testing.assert_allclose(filled[frame], points)


def test_video_size_reads_the_files_own_pixels(tmp_path):
    """The tag sheet sizes tags from this, so it must be the SOURCE resolution.

    What is on screen can be a low-resolution proxy or a downscaled decode;
    a minimum tag size derived from either is wrong by that scale factor and
    says nothing about it.
    """
    av = pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())

    assert video_size(path) == (320, 240)


def test_video_size_is_none_for_something_that_is_not_a_video(tmp_path):
    """Callers ask in order to OFFER a value, so a bad path is not an error."""
    path = tmp_path / "notes.txt"
    path.write_text("no video here", encoding="utf-8")
    assert video_size(path) is None
    assert video_size(tmp_path / "missing.mp4") is None
