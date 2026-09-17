"""Which folder receives a session's files (alignment, labels, settings) — Qt-free.

A session is a folder holding ``.ethograph/``. When files come from one folder, that
folder is the session. When they come from several, one of them (or another folder
entirely) is chosen, and the *kind* of folder chosen — the folder with the ``.nc``,
the one with the videos, the one with the audio — is what is worth remembering:
a path is one rig on one machine, a kind preference follows the user.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: Which kind of folder makes the best session folder, most preferred first.
#: Data files before media: the folder with the features is the natural home.
KIND_ORDER: tuple[str, ...] = ("nc", "nwb", "npz", "npy", "video", "pose", "audio", "ephys", "image", "labels")

#: The ``classify_files`` bucket a kind is read from; ``session`` splits by suffix.
_SESSION_SUFFIX_KIND = {".nc": "nc", ".nwb": "nwb", ".npz": "npz"}


@dataclass(frozen=True)
class SourceFolder:
    """One folder the dropped files came from, with how many of each kind."""

    folder: Path
    counts: Mapping[str, int]

    @property
    def best_kind(self) -> str:
        return min(self.counts, key=_rank)

    def label(self) -> str:
        parts = ", ".join(f"{kind} ×{n}" for kind, n in sorted(self.counts.items(), key=lambda kv: _rank(kv[0])))
        return f"{self.folder}  ({parts})"


def _rank(kind: str) -> int:
    return KIND_ORDER.index(kind) if kind in KIND_ORDER else len(KIND_ORDER)


def kind_of(path: str | Path, bucket: str) -> str | None:
    """The kind a dropped file counts as, from its ``classify_files`` bucket.

    Folders in the ``session`` bucket (pynapple, Kilosort) have no kind here: they are
    sessions or overlays in their own right, never a source folder to choose between.
    """
    p = Path(path)
    if bucket == "session":
        return None if p.is_dir() else _SESSION_SUFFIX_KIND.get(p.suffix.lower())
    if bucket in KIND_ORDER:
        return bucket
    return None


def source_folders(buckets: Mapping[str, Iterable[str]]) -> list[SourceFolder]:
    """Every folder the dropped files came from, best-kind first."""
    counts: dict[Path, Counter[str]] = {}
    for bucket, paths in buckets.items():
        for path in paths:
            kind = kind_of(path, bucket)
            if kind is None:
                continue
            counts.setdefault(Path(path).resolve().parent, Counter())[kind] += 1
    folders = [SourceFolder(folder, dict(c)) for folder, c in counts.items()]
    folders.sort(key=lambda f: (_rank(f.best_kind), -f.counts[f.best_kind], str(f.folder).lower()))
    return folders


def ranked_kinds(preferred: Sequence[str] = ()) -> tuple[str, ...]:
    """:data:`KIND_ORDER` with the remembered kinds moved to the front, most recent first."""
    front = [k for k in dict.fromkeys(preferred) if k in KIND_ORDER]
    return tuple(front) + tuple(k for k in KIND_ORDER if k not in front)


def propose(folders: Sequence[SourceFolder], preferred: Sequence[str] = ()) -> SourceFolder | None:
    """The folder to preselect: the one holding the best-ranked kind present.

    Ties between folders holding that kind go to the one with more of its files, then
    to path order, so the proposal is stable across drops.
    """
    if not folders:
        return None
    for kind in ranked_kinds(preferred):
        holders = [f for f in folders if kind in f.counts]
        if holders:
            return max(holders, key=lambda f: (f.counts[kind], -folders.index(f)))
    return folders[0]


def remember(preferred: Sequence[str], chosen: str) -> list[str]:
    """*preferred* with *chosen* promoted to the front (the setting to store)."""
    return [chosen] + [k for k in preferred if k != chosen]
