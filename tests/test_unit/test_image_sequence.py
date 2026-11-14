"""A folder of images as a video (io/image_sequence.py)."""

from __future__ import annotations

import imageio.v3 as iio
import numpy as np

from ethograph.io.image_sequence import ImageSequence, image_files


def test_natural_order_and_lazy_frames(tmp_path):
    for name, value in (("img10.png", 10), ("img2.png", 2)):
        iio.imwrite(tmp_path / name, np.full((4, 6, 3), value, np.uint8))
    (tmp_path / "notes.txt").write_text("x")
    seq = ImageSequence(tmp_path)
    assert [p.name for p in image_files(tmp_path)] == ["img2.png", "img10.png"]
    assert seq.size == (6, 4)
    assert seq[1][0, 0, 0] == 10
    assert seq.index_of("img10.png") == 1 and seq.name_of(0) == "img2.png"
