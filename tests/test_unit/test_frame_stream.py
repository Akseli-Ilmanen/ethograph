"""A sparse set of frames is decoded in one forward pass, and matches random access.

Regression guard: the frame suggester used to fetch each candidate with its
own seek, which on a long recording took minutes; ``iter_frames`` must return
exactly the requested indices, in order, with the same pixels ``__getitem__``
returns.
"""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from ethograph.gui.pose_fill import VideoFrameSource
from ethograph.gui.pose_suggest import suggest_frames

FPS = 25
N_FRAMES = 60


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    path = tmp_path_factory.mktemp("clip") / "testsrc.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x48:rate={FPS}:duration={N_FRAMES / FPS}",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "12",
            str(path),
        ],
        check=True,
    )
    return path


def test_iter_frames_matches_random_access(clip: Path):
    wanted = [3, 4, 17, 30, 31, 55]
    with VideoFrameSource(clip, fps=FPS, n_frames=N_FRAMES) as src:
        streamed = dict(src.iter_frames(wanted))
        assert list(streamed) == wanted
        for i in wanted:
            assert np.array_equal(streamed[i], src[i]), i
        grey = dict(src.iter_frames([10, 11], gray=True))
        assert grey[10].ndim == 2 and grey[10].shape == src[10].shape[:2]


def test_iter_frames_stops_when_progress_says_so(clip: Path):
    seen = []

    def progress(fraction: float) -> bool:
        seen.append(fraction)
        return fraction < 0.5

    with VideoFrameSource(clip, fps=FPS, n_frames=N_FRAMES) as src:
        delivered = [i for i, _ in src.iter_frames(range(0, 40, 4), progress=progress)]
    assert delivered == [0, 4, 8, 12, 16]
    assert seen[-1] == 0.5


def test_suggest_uses_the_stream(clip: Path, monkeypatch):
    with VideoFrameSource(clip, fps=FPS, n_frames=N_FRAMES, max_side=32) as src:
        calls = {"getitem": 0}
        original = src.__getitem__

        def counting(key):
            calls["getitem"] += 1
            return original(key)

        monkeypatch.setattr(VideoFrameSource, "__getitem__", lambda self, key: counting(key))
        picks = suggest_frames("diverse", 4, N_FRAMES, frames=src)
    assert len(picks) == 4 and all(0 <= p < N_FRAMES for p in picks)
    assert calls["getitem"] == 0


def test_mixed_takes_its_share_from_the_strongest_movements(clip: Path):
    """Half the picks are the top of the motion ranking, the rest k-means over the moving frames."""
    trace = np.zeros(N_FRAMES, dtype=np.float32)
    trace[20:26] = 5.0  # the one real movement
    with VideoFrameSource(clip, fps=FPS, n_frames=N_FRAMES, max_side=32) as src:
        picks = suggest_frames("mixed", 4, N_FRAMES, frames=src, motion=trace, motion_window=5, motion_share=0.5)
    assert any(18 <= p <= 27 for p in picks)  # a motion pick sits on the movement
    with VideoFrameSource(clip, fps=FPS, n_frames=N_FRAMES, max_side=32) as src:
        picks = suggest_frames("mixed", 4, N_FRAMES, frames=src)
    assert len(picks) == 4
    assert all(0 <= p < N_FRAMES for p in picks)
    assert min(np.diff(sorted(picks))) >= 2  # never consecutive frames of one hop


def test_motion_trace_ranks_a_bout_over_a_spike_without_decoding():
    """With a per-frame trace, 'motion' needs no frames and prefers sustained movement."""
    trace = np.zeros(N_FRAMES, dtype=np.float32)
    trace[10] = 20.0  # one-frame flicker, larger than any single frame of the bout
    trace[40:48] = 4.0  # a real movement
    picks = suggest_frames("motion", 1, N_FRAMES, motion=trace, motion_window=8)
    assert 40 <= picks[0] < 48


def test_an_index_a_phone_skipped_is_served_by_the_next_frame(tmp_path: Path):
    """Variable frame timing can skip an index outright (392 → 394).

    Regression guard: asking for the missing one raised "No frames decoded",
    which took the detector's learn step down on a Pixel recording. The gap
    is served by the frame that follows it, as ``iter_frames`` already did.
    """
    av = pytest.importorskip("av")
    path = tmp_path / "vfr.mp4"
    skipped = 20
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        for index in range(N_FRAMES):
            if index == skipped:
                continue
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 4, np.uint8), format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with VideoFrameSource(path, fps=FPS, n_frames=N_FRAMES) as src:
        assert np.array_equal(src[skipped], src[skipped + 1])
        clip = src[skipped - 1 : skipped + 2]
        assert clip.shape[0] == 3
        assert np.array_equal(clip[1], clip[0]), "inside a clip the gap holds the previous frame"
