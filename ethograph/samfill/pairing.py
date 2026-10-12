"""Pair pose files with the videos they were tracked on, by name."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: What DeepLabCut (``DLC``, ``DeepCut``) and SLEAP (``.predictions``, ``.analysis``) append to the video stem.
DLC_MARKERS = ("DLC", "DeepCut", ".predictions", ".analysis")


@dataclass(frozen=True)
class Pair:
    pose: Path
    video: Path

    @property
    def name(self) -> str:
        """The video stem: how the pair is reported and plotted."""
        return self.video.stem

    @property
    def output_stem(self) -> str:
        """The pose file's stem: what the fill's output is named after, so it sits beside its input."""
        return self.pose.stem


#: Characters that differ between a video name and what DeepLabCut makes of it.
SEPARATORS = "_-. "


def normalise(stem: str) -> str:
    """Case-folded stem with separators removed: ``Ivy_-cam-1`` and ``Ivy-cam-1`` agree."""
    return stem.casefold().translate(str.maketrans("", "", SEPARATORS))


def video_key(pose_stem: str) -> str:
    """The video a DeepLabCut file names: everything before ``DLC``/``DeepCut``, normalised."""
    cut = min((i for m in DLC_MARKERS if (i := pose_stem.find(m)) >= 0), default=len(pose_stem))
    return normalise(pose_stem[:cut])


def pose_matches_video(pose_stem: str, video_stem: str) -> bool:
    """The pose stem, up to its DeepLabCut suffix, is the video stem up to separators and case.

    Without a DLC suffix, ``video`` or ``video_<anything>`` also matches.
    """
    if video_key(pose_stem) == normalise(video_stem):
        return True
    if any(m in pose_stem for m in DLC_MARKERS) or not pose_stem.startswith(video_stem):
        return False
    return pose_stem[len(video_stem) :].startswith(tuple(SEPARATORS))


def _files(folder: Path, extensions: Iterable[str]) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    exts = {e.lower() for e in extensions}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def pair_folders(
    pose_dir: str | Path,
    video_dir: str | Path,
    pose_extensions: Iterable[str] = (".h5", ".csv", ".slp"),
    video_extensions: Iterable[str] = (".mp4", ".avi", ".mov", ".mkv"),
) -> tuple[list[Pair], list[Path], list[Path]]:
    """``(pairs, unmatched_poses, unmatched_videos)``.

    Each video takes the pose file with its name; when several pose files
    match, the one with the shortest suffix after the video stem wins, so
    ``trial1.mp4`` never claims ``trial10DLC....h5``. A pose file matching
    several videos goes to the longest video stem.
    """
    poses = _files(Path(pose_dir), pose_extensions)
    videos = _files(Path(video_dir), video_extensions)
    claimed: dict[Path, Path] = {}
    for pose in poses:
        candidates = [v for v in videos if pose_matches_video(pose.stem, v.stem)]
        if not candidates:
            continue
        video = max(candidates, key=lambda v: len(v.stem))
        current = claimed.get(video)
        if current is None or len(pose.stem) < len(current.stem):
            claimed[video] = pose
    pairs = sorted((Pair(p, v) for v, p in claimed.items()), key=lambda pr: pr.video.name)
    matched_poses = {pr.pose for pr in pairs}
    return (
        pairs,
        [p for p in poses if p not in matched_poses],
        [v for v in videos if v not in claimed],
    )
