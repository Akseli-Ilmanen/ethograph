"""Cheap metadata probe of a video file: frame rate, frame count and size.

Qt-free, so both the GUI and the feature extractors read a video's rate from
the same place — and nothing ever hardcodes one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av

from ethograph.io.image_sequence import IMAGE_SEQUENCE_RATE, ImageSequence


@dataclass
class VideoProbe:
    """What a video reports about itself (PyAV), before any frame is decoded."""

    path: str
    fps: float
    nframes: int
    #: Frame size in pixels, as encoded — the pixels a crop is spelled in.
    width: int = 0
    height: int = 0


def probe_video(video_path: str) -> VideoProbe:
    """What *video_path* reports about itself; an image folder reports one frame per image."""
    if Path(video_path).is_dir():
        return probe_image_folder(video_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None:
            raise ValueError(f"Cannot determine frame rate of {video_path}")
        fps = float(rate)
        width, height = int(stream.codec_context.width), int(stream.codec_context.height)
        nframes = stream.frames
        if not nframes and stream.duration and stream.time_base:
            nframes = int(float(stream.duration * stream.time_base) * fps)
        if not nframes and container.duration:
            nframes = int(container.duration / av.time_base * fps)
    return VideoProbe(path=str(video_path), fps=fps, nframes=int(nframes), width=width, height=height)


def probe_image_folder(folder: str) -> VideoProbe:
    """An image folder as a video: its images in natural order, on the image-sequence clock."""
    sequence = ImageSequence(folder)
    return VideoProbe(
        path=str(folder),
        fps=IMAGE_SEQUENCE_RATE,
        nframes=len(sequence),
        width=sequence.width,
        height=sequence.height,
    )
