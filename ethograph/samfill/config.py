"""The settings of a fill and the lookup table that scopes it."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PixelExclusion:
    """Pixels where the keypoint cannot truly be: a detection there is wrong and is dropped.

    A frame whose keypoint falls inside any of ``boxes`` (``x0, y0, x1, y1``,
    inclusive) or below ``y_max`` (image y grows downwards, so "below" is a
    larger y) is set to NaN, together with ``pad_frames`` neighbours either side.
    NaN counts as low, so SAM fills those frames, and never as a seed.
    """

    boxes: tuple[tuple[float, float, float, float], ...] = ()
    y_max: float | None = None
    pad_frames: int = 0

    def __post_init__(self) -> None:
        for box in self.boxes:
            if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
                raise ValueError(f"box must be (x0, y0, x1, y1) with x0 <= x1 and y0 <= y1, got {box}")
        if self.pad_frames < 0:
            raise ValueError("pad_frames must be >= 0")


@dataclass
class FillConfig:
    """Everything a fill needs besides the files.

    Confidence thresholds are the tracker's own (DeepLabCut likelihoods in
    ``[0, 1]``). A frame is *low* below ``low_confidence`` and *confident* at
    or above ``seed_confidence``; the band in between is neither seeded from
    nor overwritten.
    """

    #: Keypoints to fill; every other keypoint is copied through untouched.
    keypoints: list[str]
    #: Frames below this are filled (no default: it depends on the tracker and the data).
    low_confidence: float
    #: A seed, and the frames that end a run, must reach this.
    seed_confidence: float
    #: Individuals to fill; ``None`` means all of them.
    individuals: list[str] | None = None
    #: Key into :data:`ethograph.samfill.sam.SAM_MODELS`.
    model: str = "sam2_small"
    #: SAMURAI's motion-aware mask selection and memory on top of the model; off is plain SAM 2.
    samurai: bool = True
    #: Safety margin in frames on both sides of a gap: the seed is the nearest confident
    #: frame at least this far before the drop, and a run ends once confidence has held
    #: for this many frames.
    margin: int = 5
    #: ``"gap"`` seeds every gap from a confident frame just before it; ``"anchor"`` seeds
    #: once per segment, at its first confident frame, and tracks the whole segment.
    seeding: Literal["gap", "anchor"] = "gap"
    #: Side of the box prompt drawn round the keypoint at the seed, in pixels. A bare click
    #: makes SAM segment the whole object the keypoint sits on; a box the size of the part
    #: keeps it to the part. ``None`` is a bare click.
    seed_box_px: int | None = 24
    #: A seed whose mask is less round than this (1 is a disc) is rejected and the next
    #: confident frame tried; a rod or a half-hidden keypoint scores well below it.
    seed_min_circularity: float = 0.75
    #: How many confident frames to try as the seed of one run before giving it up.
    seed_tries: int = 10
    #: ``"netcdf"`` writes a movement NetCDF carrying its ``edited`` flag; ``"source"`` writes the
    #: tracker's own format back (DeepLabCut ``.h5``, SLEAP analysis ``.h5``, LightningPose ``.csv``).
    output: Literal["netcdf", "source"] = "netcdf"
    #: movement's ``source_software`` for the pose files.
    source_software: str = "DeepLabCut"
    #: Extensions recognised as pose files.
    pose_extensions: tuple[str, ...] = (".h5", ".csv", ".slp")
    #: Extensions recognised as videos.
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    #: Pixel exclusions keyed by a substring of the video name (``"cam-1"``); the longest
    #: key found in the name applies, ``""`` applies to every video.
    exclusions: dict[str, PixelExclusion] = field(default_factory=dict)

    def exclusion_for(self, video_name: str) -> PixelExclusion | None:
        keys = [k for k in self.exclusions if k in video_name]
        return self.exclusions[max(keys, key=len)] if keys else None

    def __post_init__(self) -> None:
        if not self.keypoints:
            raise ValueError("FillConfig.keypoints must name at least one keypoint")
        if not 0.0 <= self.low_confidence <= self.seed_confidence <= 1.0:
            raise ValueError("need 0 <= low_confidence <= seed_confidence <= 1")
        if self.margin < 1:
            raise ValueError(f"margin must be >= 1 frame, got {self.margin}")
        if not 0.0 <= self.seed_min_circularity <= 1.0 or self.seed_tries < 1:
            raise ValueError("seed_min_circularity must be in [0, 1] and seed_tries >= 1")
        if self.seed_box_px is not None and self.seed_box_px < 2:
            raise ValueError(f"seed_box_px must be >= 2 pixels or None, got {self.seed_box_px}")


@dataclass(frozen=True)
class Segment:
    """One row of the lookup table: a stretch of one video, in seconds.

    ``offset_s`` of ``None`` runs to the end of the video; with ``onset_s`` 0 that
    is the whole video.
    """

    file: str
    onset_s: float = 0.0
    offset_s: float | None = None

    def __post_init__(self) -> None:
        if self.offset_s is not None and self.offset_s < self.onset_s:
            raise ValueError(f"segment of {self.file!r} ends ({self.offset_s}) before it starts ({self.onset_s})")


SEGMENT_COLUMNS = ("file", "onset_s", "offset_s")


def read_segments_tsv(path: str | Path) -> list[Segment]:
    """Read the lookup table: tab-separated, columns ``file``, ``onset_s``, ``offset_s``.

    A header row is optional; when present it must name those columns. The
    ``file`` cell is a file name or a stem, matched to the video either way.
    An empty ``offset_s`` means "to the end of the video"; an empty ``onset_s``
    means 0. A row holding only the video name therefore fills the whole video.
    """
    path = Path(path)
    rows: list[Segment] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for n, raw in enumerate(reader, start=1):
            cells = [c.strip() for c in raw]
            if not any(cells) or cells[0].startswith("#"):
                continue
            if n == 1 and cells[0].lower() == "file":
                if tuple(c.lower() for c in cells[:3]) != SEGMENT_COLUMNS:
                    raise ValueError(f"{path}: header must be {SEGMENT_COLUMNS}, got {tuple(cells[:3])}")
                continue
            if len(cells) > 3:
                raise ValueError(f"{path}:{n}: expected at most 3 columns {SEGMENT_COLUMNS}, got {cells}")
            cells += [""] * (3 - len(cells))
            start = float(cells[1]) if cells[1] else 0.0
            end = float(cells[2]) if cells[2] else None
            rows.append(Segment(cells[0], start, end))
    return rows


def segments_to_frames(segments: list[Segment], video_name: str, fps: float, n_frames: int) -> list[range]:
    """The frame windows of *video_name*, rounded to the nearest frame, clipped and merged.

    A window is inclusive at both ends in the table (``10.0`` to ``12.0`` s
    covers the frame nearest ``12.0``), so the returned ranges end one past the
    rounded end frame. No row for the video means no window: the video is skipped.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    stem = Path(video_name).stem
    windows: list[tuple[int, int]] = []
    for seg in segments:
        if Path(seg.file).stem != stem:
            continue
        start = max(0, int(round(seg.onset_s * fps)))
        end = n_frames if seg.offset_s is None else min(n_frames, int(round(seg.offset_s * fps)) + 1)
        if end > start:
            windows.append((start, end))
    windows.sort()
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [range(s, e) for s, e in merged]
