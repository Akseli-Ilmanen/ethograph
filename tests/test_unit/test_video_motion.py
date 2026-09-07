"""Video motion from the compressed stream: bytes per frame, no decoding.

``extract_packet_motion`` reads packet sizes in display order, takes them
relative to their GOP-offset profile and masks keyframes. These tests
synthesise a short clip with PyAV and check the shape, that a moving segment
reads as higher motion than a static one, and that the encoder's frame-role
pattern is removed.
"""

import numpy as np
import pytest

av = pytest.importorskip("av")

from ethograph.features.movement import (  # noqa: E402
    extract_packet_motion,
    gop_relative_sizes,
    mask_keyframes,
)
from ethograph.io.validation import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS  # noqa: E402


def _make_clip(path, n_static, n_moving, width=64, height=48, fps=30):
    """Write a clip: ``n_static`` still frames then ``n_moving`` with motion."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for i in range(n_static + n_moving):
            img = np.full((height, width, 3), 40, dtype=np.uint8)
            if i >= n_static:
                # Bright square sweeping left→right = large luma difference.
                x = (i - n_static) * 6 % (width - 12)
                img[10:30, x : x + 12] = 230
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def test_audio_extensions_exclude_video_containers():
    # Regression guard for the deleted MP4 audio branch: re-adding a video
    # container extension here would resurrect dead code paths.
    assert AUDIO_EXTENSIONS.isdisjoint(VIDEO_EXTENSIONS)
    assert ".mp4" not in AUDIO_EXTENSIONS
    assert ".mov" not in AUDIO_EXTENSIONS
    assert ".avi" not in AUDIO_EXTENSIONS


def test_mask_keyframes_interpolates_the_spike_and_its_halo():
    trace = np.array([1, 1, 1, 9, 9, 9, 1, 1], dtype=np.float32)
    out = mask_keyframes(trace, np.array([4]), halo=1)
    assert out[3:6].tolist() == [1.0, 1.0, 1.0]
    assert mask_keyframes(trace, np.array([], dtype=int)).tolist() == trace.tolist()


def test_gop_relative_sizes_removes_the_frame_role_pattern():
    """I- and reference frames cost bytes by role; relative to the offset profile they read as 1."""
    gop = np.array([100.0, 5.0, 20.0, 5.0], dtype=np.float32)
    sizes = np.tile(gop, 4)
    keyframes = np.arange(0, 16, 4)
    rel = gop_relative_sizes(sizes, keyframes)
    np.testing.assert_allclose(rel, 1.0)
    sizes[9] = 40.0  # a real change on a frame that is usually cheap
    rel = gop_relative_sizes(sizes, keyframes)
    assert rel[9] > 2 and abs(rel[8] - 1.0) < 1e-6


def test_packet_motion_shape_and_movement(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, n_static=15, n_moving=15)
    da = extract_packet_motion(clip, fps=30.0)
    assert da.ndim == 1 and len(da) == 30
    np.testing.assert_allclose(da["time"].values, np.arange(30) / 30.0)
    motion = da.values
    # the moving half costs the encoder more bytes than the static half
    assert motion[16:].mean() > motion[1:15].mean()
