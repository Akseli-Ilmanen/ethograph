"""Frame-suggestion strategies: spacing, diversity and motion ranking.

Uses synthetic frame stacks rather than a video, so the selection logic is
tested independently of decoding.
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui import pose_suggest
from ethograph.gui.pose_suggest import (
    METHODS,
    default_min_gap,
    enforce_min_gap,
    suggest_frames,
    suggest_uniform,
)

N_FRAMES = 200


class _FakeFrames:
    """Indexable stack of tiny frames, standing in for VideoFrameSource."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


def _static_video(n: int = N_FRAMES, size: int = 8) -> _FakeFrames:
    """Three distinct scenes, each held for a third of the clip."""
    data = np.zeros((n, size, size, 3), dtype=np.uint8)
    data[n // 3 : 2 * n // 3] = 128
    data[2 * n // 3 :] = 255
    return _FakeFrames(data)


def _burst_video(n: int = N_FRAMES, size: int = 8, burst=(100, 110)) -> _FakeFrames:
    """Still, except for a short burst of rapid change."""
    rng = np.random.default_rng(0)
    data = np.zeros((n, size, size, 3), dtype=np.uint8)
    for f in range(*burst):
        data[f] = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return _FakeFrames(data)


# ----------------------------------------------------------------------
# Spacing
# ----------------------------------------------------------------------


def test_min_gap_scales_with_the_request():
    assert default_min_gap(2000, 20) == 25
    # More frames requested from the same video -> tighter spacing.
    assert default_min_gap(2000, 100) < default_min_gap(2000, 20)


def test_min_gap_never_returns_zero():
    assert default_min_gap(10, 100) >= 1
    assert default_min_gap(0, 0) >= 1


def test_enforce_min_gap_keeps_priority_order_winners():
    kept = enforce_min_gap([100, 101, 102, 150], min_gap=10, count=4)
    assert kept == [100, 150]


def test_enforce_min_gap_respects_the_count():
    kept = enforce_min_gap(list(range(0, 200, 20)), min_gap=5, count=3)
    assert len(kept) == 3


def test_enforce_min_gap_returns_sorted_frames():
    kept = enforce_min_gap([150, 50, 100], min_gap=10, count=3)
    assert kept == sorted(kept)


# ----------------------------------------------------------------------
# Uniform
# ----------------------------------------------------------------------


def test_uniform_spans_the_video():
    picks = suggest_uniform(5, N_FRAMES)
    assert len(picks) == 5
    assert picks[0] == 0
    assert picks[-1] == N_FRAMES - 1


def test_uniform_needs_no_frames():
    assert suggest_frames("uniform", 4, N_FRAMES) == suggest_uniform(4, N_FRAMES)


def test_uniform_caps_at_what_is_available():
    assert suggest_uniform(50, 5) == [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------
# Diverse (k-means)
# ----------------------------------------------------------------------


def test_diverse_covers_distinct_scenes():
    """Three visually distinct thirds -> one frame from each."""
    picks = suggest_frames("diverse", 3, N_FRAMES, _static_video())
    thirds = {min(2, p // (N_FRAMES // 3)) for p in picks}
    assert thirds == {0, 1, 2}


def test_diverse_is_deterministic():
    video = _static_video()
    assert suggest_frames("diverse", 3, N_FRAMES, video) == suggest_frames("diverse", 3, N_FRAMES, video)


def test_diverse_never_exceeds_the_count():
    picks = suggest_frames("diverse", 4, N_FRAMES, _static_video())
    assert len(picks) <= 4


# ----------------------------------------------------------------------
# Motion
# ----------------------------------------------------------------------


def test_motion_picks_the_active_stretch_first():
    """Asked for one frame, it must be the moving one."""
    picks = suggest_frames("motion", 1, N_FRAMES, _burst_video())
    assert len(picks) == 1
    assert 95 <= picks[0] <= 115, picks


def test_motion_always_covers_the_burst():
    """Min-gap spacing means one short burst yields one frame, not a run — but
    that frame is always present, and the rest pad out the requested count."""
    picks = suggest_frames("motion", 3, N_FRAMES, _burst_video())
    assert len(picks) == 3
    assert any(95 <= p <= 115 for p in picks), picks


def test_motion_does_not_return_a_run_of_neighbours():
    """The point of the min-gap pass: one burst must not fill the budget."""
    picks = suggest_frames("motion", 5, N_FRAMES, _burst_video(), min_gap=4)
    gaps = np.diff(picks)
    assert all(g >= 4 for g in gaps), picks


# ----------------------------------------------------------------------
# Uncertain (post-fill)
# ----------------------------------------------------------------------


def _confidence(low_at: range, n: int = N_FRAMES, n_points: int = 3) -> np.ndarray:
    conf = np.ones((n, n_points), dtype=float)
    conf[low_at] = 0.05
    return conf


def test_uncertain_finds_the_low_confidence_stretch():
    picks = suggest_frames("uncertain", 1, N_FRAMES, confidence=_confidence(range(60, 70)))
    assert len(picks) == 1
    assert 60 <= picks[0] < 70, picks


def test_uncertain_needs_no_video():
    picks = suggest_frames("uncertain", 3, N_FRAMES, frames=None, confidence=_confidence(range(60, 70)))
    assert len(picks) == 3


def test_uncertain_without_a_fill_raises():
    with pytest.raises(ValueError, match="fill to have run first"):
        suggest_frames("uncertain", 3, N_FRAMES)


def test_uncertain_spreads_across_one_bad_stretch():
    """A single long failure must not consume the whole budget as neighbours."""
    picks = suggest_frames("uncertain", 4, N_FRAMES, confidence=_confidence(range(20, 90)), min_gap=10)
    assert all(g >= 10 for g in np.diff(picks)), picks


def test_uncertain_ignores_the_frames_the_fill_never_reached():
    """A fill stops at the outermost labels; past them there is no prediction.

    Those frames score NaN, and a NaN is not a bad prediction — it is no
    prediction. Suggesting them would send the user past their last label
    instead of into the gaps between labels, which is what needs correcting.
    """
    conf = _confidence(range(60, 70))
    conf[80:] = np.nan  # nothing labelled out here, so nothing was filled either

    picks = suggest_frames("uncertain", 3, N_FRAMES, confidence=conf, min_gap=5)

    assert picks, "the filled stretch still has frames worth labelling"
    assert all(pick < 80 for pick in picks), picks


def test_uncertain_stays_inside_the_filled_span():
    """The span is bounded on both sides: nothing before the first label either."""
    conf = _confidence(range(60, 70))
    conf[:40] = np.nan
    conf[150:] = np.nan

    picks = suggest_frames("uncertain", 5, N_FRAMES, confidence=conf, min_gap=5)

    assert all(40 <= pick < 150 for pick in picks), picks
    assert any(60 <= pick < 70 for pick in picks), picks


def test_uncertain_returns_nothing_when_no_frame_was_filled():
    conf = np.full((N_FRAMES, 3), np.nan)
    assert suggest_frames("uncertain", 3, N_FRAMES, confidence=conf) == []


def test_frame_confidence_is_nan_where_the_fill_did_not_reach():
    conf = np.array([[1.0, 0.0], [np.nan, np.nan]])
    scores = pose_suggest.frame_confidence(conf)
    assert scores[0] == 0.0
    assert np.isnan(scores[1])


def test_frame_confidence_takes_the_worst_point():
    """One lost keypoint is the whole reason to revisit a frame — an average
    let the other points vote it back down the ranking."""
    conf = np.array([[1.0, 0.0], [0.5, 0.5]])
    np.testing.assert_allclose(pose_suggest.frame_confidence(conf), [0.0, 0.5])


def test_frame_confidence_skips_points_the_schema_leaves_out():
    """An asymmetric schema blanks those to NaN; taking them as the minimum
    would pin every frame of that individual to the same score."""
    conf = np.array([[0.8, np.nan], [0.4, np.nan]])
    np.testing.assert_allclose(pose_suggest.frame_confidence(conf), [0.8, 0.4])


def test_a_single_bad_keypoint_outranks_a_uniformly_mediocre_frame():
    conf = np.full((N_FRAMES, 3), 0.5)
    conf[7] = [0.9, 0.9, 0.1]  # better on average, worse where it matters

    assert suggest_frames("uncertain", 1, N_FRAMES, confidence=conf) == [7]


def test_frame_confidence_passes_through_1d():
    conf = np.array([0.2, 0.8])
    np.testing.assert_allclose(pose_suggest.frame_confidence(conf), conf)


def test_frame_confidence_handles_the_store_shape():
    """Store confidence is (frames, individuals, keypoints)."""
    conf = np.ones((N_FRAMES, 2, 3))
    conf[5] = 0.0
    scores = pose_suggest.frame_confidence(conf)
    assert scores.shape == (N_FRAMES,)
    assert scores[5] == 0.0


# ----------------------------------------------------------------------
# Shared contract
# ----------------------------------------------------------------------


def _kwargs_for(method):
    """Whatever that method needs: pixels, a fill's confidence, or nothing."""
    if method == "uncertain":
        return {"confidence": _confidence(range(60, 70))}
    if method == "uniform":
        return {}
    return {"frames": _burst_video()}


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_sorted_unique_frames_in_range(method):
    picks = suggest_frames(method, 6, N_FRAMES, **_kwargs_for(method))
    assert picks == sorted(set(picks))
    assert all(0 <= p < N_FRAMES for p in picks)


@pytest.mark.parametrize("method", METHODS)
def test_zero_count_returns_nothing(method):
    assert suggest_frames(method, 0, N_FRAMES, **_kwargs_for(method)) == []


@pytest.mark.parametrize("method", METHODS)
def test_every_method_skips_already_labelled_frames(method):
    """There is nothing to suggest about a frame that is already labelled, and
    no method is exempt — including `diverse`, which a per-method test missed."""
    exclude = set(range(60, 70))
    picks = suggest_frames(method, 3, N_FRAMES, exclude=exclude, **_kwargs_for(method))
    assert not (set(picks) & exclude)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown suggestion method"):
        suggest_frames("magic", 3, N_FRAMES)


@pytest.mark.parametrize("method", ["diverse", "motion"])
def test_pixel_methods_require_frames(method):
    with pytest.raises(ValueError, match="needs video frames"):
        suggest_frames(method, 3, N_FRAMES, frames=None)


def test_empty_video_returns_nothing():
    assert suggest_frames("uniform", 5, 0) == []


def test_long_videos_are_strided_not_fully_decoded(monkeypatch):
    """A 100k-frame video must not decode every frame to make suggestions."""
    monkeypatch.setattr(pose_suggest, "MAX_CANDIDATES", 100)
    reads = []

    class _Counting(_FakeFrames):
        def __getitem__(self, key):
            reads.append(key)
            return np.zeros((4, 4, 3), dtype=np.uint8)

    suggest_frames("motion", 5, 100_000, _Counting(None))
    assert len(reads) <= 100


def test_progress_cancellation_stops_decoding():
    calls = []

    def progress(fraction):
        calls.append(fraction)
        return False

    suggest_frames("motion", 3, N_FRAMES, _burst_video(), progress=progress)
    assert len(calls) == 1
