"""Export a stretch of a video as cropped PNG frames, one per frame, named by their time in the video.

Edit the constants and run it::

    python scripts/export_frames.py

Frames from :data:`T_START` to :data:`T_END` (inclusive, seconds on the video's own clock) are
decoded with the repo's decoder, cropped to :data:`CROP` (pixels of the encoded frame, ``x1``/``y1``
exclusive, the GUI's crop spelling) and written as ``{stem}_{time}.png`` into :data:`OUT_DIR`,
where ``time`` is the frame's time in seconds to the video's own frame precision. The rate comes
from the file; nothing here assumes one.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from ethograph.io.video_decode import RGBConverter, decode_frames
from ethograph.io.video_probe import probe_video

VIDEO = Path(r"C:\Users\aksel\Documents\VidData\20250512_01_Ivy\2025-05-12_052_Ivy-cam-1.mp4")
OUT_DIR = Path(r"C:\Users\aksel\Desktop\ethograph\frames")
#: Seconds on the video's clock, both ends inclusive.
T_START = 3.7
T_END = 4.1
#: Pixels of the encoded frame; ``x1`` and ``y1`` are exclusive.
CROP = {"x0": 129, "y0": 2, "x1": 409, "y1": 282}


def export(video: Path, out_dir: Path, t_start: float, t_end: float, crop: dict[str, int]) -> list[Path]:
    """Write every frame of *video* between *t_start* and *t_end* cropped to *crop*; returns the files."""
    probe = probe_video(str(video))
    if not (0 <= crop["x0"] < crop["x1"] <= probe.width and 0 <= crop["y0"] < crop["y1"] <= probe.height):
        raise ValueError(f"crop {crop} does not fit the {probe.width}×{probe.height} frame")
    # A hair of slack so 4.1 s at 200 fps is frame 820, not 819.
    first, last = math.ceil(t_start * probe.fps - 1e-6), math.floor(t_end * probe.fps + 1e-6)
    if last < first:
        raise ValueError(f"no frame of {video.name} lies between {t_start} s and {t_end} s at {probe.fps} fps")
    # Digits that tell neighbouring frames apart in the file name.
    decimals = max(1, math.ceil(math.log10(probe.fps)))

    out_dir.mkdir(parents=True, exist_ok=True)
    to_rgb = RGBConverter()
    written = []
    for index, frame in zip(range(first, last + 1), decode_frames(video, start=first), strict=False):
        pixels = to_rgb(frame)[crop["y0"] : crop["y1"], crop["x0"] : crop["x1"]]
        path = out_dir / f"{video.stem}_{index / probe.fps:.{decimals}f}s.png"
        Image.fromarray(pixels).save(path)
        written.append(path)
    if len(written) != last - first + 1:
        raise ValueError(f"{video.name} ended after {len(written)} of {last - first + 1} frames")
    return written


def main() -> None:
    written = export(VIDEO, OUT_DIR, T_START, T_END, CROP)
    print(f"Wrote {len(written)} frames to {OUT_DIR}: {written[0].name} … {written[-1].name}")


if __name__ == "__main__":
    main()
