"""A microphone's offset against a camera soundtrack is found, scored and paired."""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.io.audio_align import (
    ANALYSIS_RATE,
    LOW_SCORE,
    AlignedPair,
    OffsetEstimate,
    estimate_offset,
    loudest_time,
    propose_pairs,
    score_matrix,
)

RATE = ANALYSIS_RATE


def _scene(seconds: float, seed: int) -> np.ndarray:
    """A room's worth of events: short filtered-noise bursts at random moments, silence between."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * RATE), dtype=np.float32)
    for _ in range(int(seconds * 2)):
        start = rng.integers(0, len(out) - RATE // 4)
        burst = rng.standard_normal(RATE // 4).astype(np.float32) * np.hanning(RATE // 4)
        out[start : start + RATE // 4] += burst * rng.uniform(0.2, 1.0)
    return out


def _as_camera(scene: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """What a camera mic hears: low-passed, quieter, with its own noise floor."""
    kernel = np.ones(8, dtype=np.float32) / 8
    return (np.convolve(scene, kernel, mode="same") * 0.3 + rng.standard_normal(len(scene)) * 0.01).astype(np.float32)


def _shifted(scene: np.ndarray, start_s: float, length_s: float) -> np.ndarray:
    """The scene as recorded by a device that started at *start_s* on the scene's clock."""
    out = np.zeros(int(length_s * RATE), dtype=np.float32)
    i0 = int(round(start_s * RATE))
    src_lo, src_hi = max(i0, 0), min(i0 + len(out), len(scene))
    out[src_lo - i0 : src_hi - i0] = scene[src_lo:src_hi]
    return out


@pytest.mark.parametrize("mic_start_s", [-5.0, 3.37])
def test_offset_is_where_the_microphone_started_on_the_camera_clock(mic_start_s):
    rng = np.random.default_rng(0)
    scene = _scene(40.0, seed=1)
    camera = _as_camera(_shifted(scene, 10.0, 25.0), rng)  # camera runs 10 s → 35 s of the scene
    mic = _shifted(scene, 10.0 + mic_start_s, 25.0) + rng.standard_normal(int(25.0 * RATE)).astype(np.float32) * 0.02

    est = estimate_offset(camera, mic)

    assert est.offset_s == pytest.approx(mic_start_s, abs=0.02)
    assert est.score > LOW_SCORE
    assert est.drift_ppm is None  # 25 s cannot be split into three 60 s windows


def test_unrelated_recordings_score_low():
    rng = np.random.default_rng(2)
    est = estimate_offset(_as_camera(_scene(20.0, seed=3), rng), _scene(20.0, seed=4))
    assert not est.reliable


def test_windows_expose_a_clock_drift():
    rng = np.random.default_rng(5)
    scene = _scene(400.0, seed=6)
    camera = _as_camera(scene, rng)
    # The mic's clock runs 0.1 % fast (more mic seconds per camera second): its
    # file holds the scene stretched, so each later window lands earlier on the
    # camera clock than the first did.
    stretched = np.interp(np.arange(len(scene)) / 1.001, np.arange(len(scene)), scene).astype(np.float32)

    est = estimate_offset(camera, stretched, window_s=60.0)

    assert est.drift_ppm is not None
    assert est.drift_ppm == pytest.approx(-1000.0, rel=0.1)


def test_pairs_are_proposed_by_content_not_order():
    rng = np.random.default_rng(7)
    scenes = [_scene(20.0, seed=s) for s in (10, 11, 12)]
    cameras = [_as_camera(s, rng) for s in scenes]
    mics = [_shifted(scenes[2], -1.0, 20.0), _shifted(scenes[0], 2.0, 20.0), _shifted(scenes[1], 0.5, 20.0)]

    matrix = score_matrix(cameras, mics)

    assert propose_pairs(matrix) == [1, 2, 0]
    assert matrix[0][1].offset_s == pytest.approx(2.0, abs=0.02)


def test_pairing_refuses_unequal_counts():
    with pytest.raises(ValueError, match="as many"):
        propose_pairs([[OffsetEstimate(0.0, 1.0), OffsetEstimate(0.0, 1.0)]])


def test_loudest_time_finds_the_burst():
    signal = np.zeros(10 * RATE, dtype=np.float32)
    signal[6 * RATE : 6 * RATE + RATE // 2] = 1.0
    assert loudest_time(signal, RATE, 0.0, 10.0) == pytest.approx(6.25, abs=0.3)


def test_offset_sentence_names_direction():
    assert AlignedPair("v", "a", -5.0, 20.0).describe_offset() == "audio starts 5.00 s before video"
    assert AlignedPair("v", "a", 0.25, 20.0).describe_offset() == "audio starts 0.25 s after video"
    assert AlignedPair("v", "a", 0.0, 20.0).describe_offset() == "audio and video start together"
