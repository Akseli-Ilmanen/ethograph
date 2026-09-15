"""Video features from files: one ``(frames, D)`` array per video, matched by name.

A network run outside ethograph — FERAL, DINO, anything that writes one
array per video — leaves a folder of ``{video stem}.npy``. Those arrays are
*video features*, whatever the network: attached to the open trial tree in
memory, never copied into the session file, never declared beside it.

* **GUI**: drop folders or ``.npy`` files onto the window. Each folder is one
  feature named after it; loose files together are one feature named after
  their folder. Every one opens as a heatmap and is listed under *Video
  features* in the add-panel popup with the path it came from, so two
  exports of one model with different settings stay tellable apart.
* **Pipeline**: a session lists them in its config —
  ``video_feature_folders: {feral: D:/emb/session_01}`` — and
  :func:`ethograph.segment.sessions.open_session` attaches them, so
  ``features.columns`` selects ``feral: {feral_dims: 0..767}`` like any
  variable of the file.

The only contract is the file name: ``{video stem}.npy`` for the video the
alignment names for that trial. Which camera's names the files follow is
found by looking (:func:`detect_device`). A video without a file is fine —
features computed on a subset are still features — its trial reads NaN and
the log says how many videos matched; files matching *no* video are an
error, since that is the wrong folder, not a gap.

Each array is memory-mapped and sampled onto the trial clock the way the
S3D merge does (trial = video + offset, nearest frame, the video's own
rate). The variable carries ``attrs["attached_from"]`` — the folder or
file list it came from — which is what marks it as in-memory:
:meth:`TrialTree.save` drops every such variable before writing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import xarray as xr

from ethograph.io import schema
from ethograph.utils.xr_utils import get_time_coord

if TYPE_CHECKING:
    from ethograph.io.trialtree import TrialTree

logger = logging.getLogger(__name__)

#: The attr naming where an in-memory video feature came from; its presence
#: is what keeps the variable out of the session file.
ATTACHED_FROM = "attached_from"
ARRAY_SUFFIX = ".npy"

__all__ = [
    "ARRAY_SUFFIX",
    "ATTACHED_FROM",
    "AttachResult",
    "VideoFeatureFiles",
    "attach_video_features",
    "detect_device",
    "feature_name_for",
    "sample_on_trial",
    "video_feature_sources",
    "video_feature_vars",
]


@dataclass(frozen=True)
class VideoFeatureFiles:
    """One video feature: the candidate arrays and which media names them."""

    name: str
    files: tuple[Path, ...]
    #: What the user pointed at, for the popup and the attrs: a folder, or ``"N files in <folder>"``.
    source: str
    stream: str = "video"
    device: str | None = None
    kind: str = schema.VIDEO_FEATURE
    dim: str | None = None

    @classmethod
    def from_paths(
        cls,
        name: str,
        paths: Iterable[str | Path],
        device: str | None = None,
        kind: str = schema.VIDEO_FEATURE,
        dim: str | None = None,
    ) -> VideoFeatureFiles:
        """Folders expand to the ``.npy`` files directly inside them; files are kept as given.

        A path that is neither an existing folder nor a ``.npy`` file is an
        error naming it.
        """
        files: list[Path] = []
        folders: list[Path] = []
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                folders.append(p)
                files += sorted(q for q in p.iterdir() if q.suffix.lower() == ARRAY_SUFFIX and q.is_file())
            elif p.is_file() and p.suffix.lower() == ARRAY_SUFFIX:
                files.append(p)
            else:
                raise FileNotFoundError(f"{p}: not a folder and not a {ARRAY_SUFFIX} file")
        if not files:
            raise FileNotFoundError(f"No {ARRAY_SUFFIX} files under {[str(Path(p)) for p in paths]}")
        loose = [p for p in files if p.parent not in folders]
        if folders and not loose:
            source = str(folders[0]) if len(folders) == 1 else ", ".join(str(f) for f in folders)
        else:
            parents = sorted({str(p.parent) for p in loose})
            source = f"{len(files)} files in {', '.join(parents)}"
        return cls(name, tuple(files), source, device=device, kind=kind, dim=dim)

    @property
    def feature_dim(self) -> str:
        return self.dim or f"{self.name}_dims"

    @property
    def by_stem(self) -> dict[str, Path]:
        return {p.stem: p for p in self.files}

    def file_for(self, media_filename: str) -> Path | None:
        return self.by_stem.get(Path(media_filename).stem)


@dataclass(frozen=True)
class AttachResult:
    """What :func:`attach_video_features` did."""

    name: str
    source: str
    device: str | None
    width: int
    n_trials: int
    matched: int
    missing: tuple[Any, ...]

    @property
    def coverage(self) -> float:
        return self.matched / self.n_trials if self.n_trials else 0.0


def feature_name_for(path: str | Path, taken: Iterable[str] = ()) -> str:
    """A variable name from a folder name: ``D:/emb/feral cfg-A`` → ``feral_cfg_A``.

    Made unique against *taken* with a numeric suffix.
    """
    stem = re.sub(r"\W+", "_", Path(path).name).strip("_") or "video_features"
    if stem[0].isdigit():
        stem = f"f_{stem}"
    taken_set = set(taken)
    name, n = stem, 2
    while name in taken_set:
        name, n = f"{stem}_{n}", n + 1
    return name


def video_feature_vars(ds: xr.Dataset) -> list[str]:
    """The data_vars of *ds* attached in memory from files (never to be saved)."""
    return [str(name) for name, var in ds.data_vars.items() if var.attrs.get(ATTACHED_FROM)]


def video_feature_sources(ds: xr.Dataset) -> dict[str, str]:
    """``{variable: where it came from}`` for every attached variable of *ds*."""
    return {name: str(ds[name].attrs[ATTACHED_FROM]) for name in video_feature_vars(ds)}


def sample_on_trial(array: np.ndarray, time: np.ndarray, rate: float, offset: float) -> np.ndarray:
    """*array* ``(frames, D)`` on the video clock, read at each trial time.

    Frame 0 of the file sits at trial time *offset* (``VideoSync``: trial =
    video + offset), so trial time ``t`` reads frame ``round((t - offset) *
    rate)``, clipped to the file. When that is the identity — same rate, no
    offset, same length — the array itself is returned, so a memory-mapped
    file stays memory-mapped.
    """
    index = np.rint((np.asarray(time, dtype=float) - offset) * rate).astype(int)
    index = np.clip(index, 0, array.shape[0] - 1)
    if array.shape[0] == len(index) and np.array_equal(index, np.arange(len(index))):
        return array
    return np.asarray(array[index])


def _files_by_trial(
    dt: TrialTree, alignment: Any, spec: VideoFeatureFiles, device: str | None
) -> dict[Any, Path | None]:
    out: dict[Any, Path | None] = {}
    for trial in dt.trials:
        filename = alignment.media_filename(trial, spec.stream, device)
        out[trial] = spec.file_for(filename) if filename else None
    return out


def detect_device(dt: TrialTree, alignment: Any, spec: VideoFeatureFiles) -> str | None:
    """The device whose media names *spec*'s files, or ``None`` for the alignment's unnamed default.

    Every device is tried; the one with the most files matched wins. A tie
    between two devices that both match is an error naming them, since the
    files cannot be told apart. No match at all falls back to the first
    device, so :func:`attach_video_features` can name the file it expected.
    """
    devices: list[str | None] = list(alignment.devices(spec.stream)) or [None]
    counts = {
        device: sum(p is not None for p in _files_by_trial(dt, alignment, spec, device).values()) for device in devices
    }
    best = max(counts.values())
    if best == 0:
        return devices[0]
    winners = [d for d, n in counts.items() if n == best]
    if len(winners) > 1:
        raise ValueError(
            f"{spec.source}: the files match {len(winners)} cameras equally ({winners}), so none can be picked"
        )
    return winners[0]


def attach_video_features(dt: TrialTree, alignment: Any, spec: VideoFeatureFiles) -> AttachResult:
    """Attach *spec* to every trial of *dt* as the variable ``spec.name``, replacing one of that name.

    ``spec.device`` left ``None`` is detected. Nodes are replaced without
    marking them dirty: nothing here is a change to the session file.
    """
    if dt._is_continuous:
        raise ValueError(f"{spec.source}: video features need a trial tree, not a continuous dataset")
    device = spec.device if spec.device is not None else detect_device(dt, alignment, spec)
    rate = alignment.get_stream_rate(spec.stream, device)
    if rate is None:
        raise ValueError(
            f"{spec.source}: {spec.name!r} is on the {spec.stream} clock, but the alignment "
            f"declares no rate for that stream (device {device!r})"
        )
    files = _files_by_trial(dt, alignment, spec, device)
    found = [p for p in files.values() if p is not None]
    if not found:
        named = next((n for n in (alignment.media_filename(t, spec.stream, device) for t in dt.trials) if n), None)
        raise FileNotFoundError(
            f"{spec.source}: none of its {len(spec.files)} files is named after a video of this session"
            + (f" (expected e.g. {Path(named).stem}{ARRAY_SUFFIX})" if named else " — the alignment names no video")
        )
    width = int(np.load(found[0], mmap_mode="r").shape[1])
    missing = []
    for trial, path in files.items():
        ds = dt.trial(trial)
        time_coord = get_time_coord(next(iter(ds.data_vars.values())))
        if time_coord is None:
            raise ValueError(f"trial {trial!r} has no time coord to sample {spec.name!r} onto")
        time = np.asarray(time_coord.values)
        if path is None:
            values = np.full((len(time), width), np.nan, dtype=np.float32)
            missing.append(trial)
        else:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2:
                raise ValueError(f"{path}: expected (frames, D), got shape {array.shape}")
            if array.shape[1] != width:
                raise ValueError(f"{path}: {array.shape[1]} columns, but {found[0].name} has {width}")
            offset = float(alignment.stream_offset_for_trial(trial, spec.stream, device))
            values = sample_on_trial(array, time, float(rate), offset)
        da = xr.DataArray(
            values,
            dims=(time_coord.name, spec.feature_dim),
            coords={time_coord.name: time, spec.feature_dim: np.arange(width)},
            name=spec.name,
        )
        schema.describe(da, spec.kind, is_egocentric=False, **{ATTACHED_FROM: spec.source})
        ds = ds.drop_vars([spec.name, spec.feature_dim], errors="ignore")
        dt[dt._trial_node_name(trial)] = xr.DataTree(ds.assign({spec.name: da}))
    result = AttachResult(spec.name, spec.source, device, width, len(files), len(found), tuple(missing))
    logger.info(
        "Added video features %r (%d columns) from %s: %d of %d videos matched (%.0f%%)",
        spec.name,
        width,
        spec.source,
        result.matched,
        result.n_trials,
        100 * result.coverage,
    )
    if missing:
        logger.warning("%r: no file for %d trials, which read NaN: %s", spec.name, len(missing), missing)
    return result
