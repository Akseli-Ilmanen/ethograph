"""Per-video feature files attached to an open session, matched by video name.

A feature computed outside ethograph — a FERAL embedding, a DINO feature,
anything that writes one ``(frames, D)`` array per video — is attached to
the loaded trial tree in memory, never copied into the session file and
never declared beside it. The GUI does it from **Model ▸ Video embeddings:
open folder(s) as heatmap…** (any number of folders, each one feature named
after its folder); a script does it with :func:`attach_external_feature`.

The only contract is the file name: ``{video stem}.npy`` for the video the
alignment names for that trial. Which camera's names a folder follows is
found by looking (:func:`detect_device`), so the user picks a folder and
nothing else. Each array is memory-mapped and sampled onto the trial's
clock the way ``with_video_feature`` merges an S3D sidecar (trial = video +
offset, nearest frame, the video's own rate). Downstream the variable is a
plain feature carrying ``attrs["external"] = 1`` and the folder it came
from in ``attrs["external_dir"]``; :meth:`TrialTree.save` drops it before
writing. A trial whose file is missing reads NaN and is named in the log; a
folder with no file for *any* trial is an error, since that is a wrong
folder, not a gap.
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

#: The attr marking a variable that was attached in memory and must never be saved.
EXTERNAL = "external"
#: The attr naming the folder the variable's files came from.
EXTERNAL_DIR = "external_dir"
EXTERNAL_FILE = "external_file"

__all__ = [
    "EXTERNAL",
    "EXTERNAL_DIR",
    "AttachResult",
    "ExternalFeature",
    "attach_external_feature",
    "detect_device",
    "external_dirs",
    "external_vars",
    "feature_name_for",
    "sample_on_trial",
]


@dataclass(frozen=True)
class ExternalFeature:
    """One feature: where its files are and which media file names them."""

    name: str
    dir: Path
    stream: str = "video"
    device: str | None = None
    pattern: str = "{stem}.npy"
    kind: str = schema.VIDEO_FEATURE
    dim: str | None = None

    @property
    def feature_dim(self) -> str:
        return self.dim or f"{self.name}_dims"

    def file_for(self, media_filename: str) -> Path:
        return self.dir / self.pattern.format(stem=Path(media_filename).stem)


@dataclass(frozen=True)
class AttachResult:
    """What :func:`attach_external_feature` did."""

    name: str
    dir: Path
    device: str | None
    width: int
    n_trials: int
    missing: tuple[Any, ...]


def feature_name_for(folder: str | Path, taken: Iterable[str] = ()) -> str:
    """A variable name from a folder name: ``D:/emb/feral cfg-A`` → ``feral_cfg_A``.

    Made unique against *taken* with a numeric suffix.
    """
    stem = re.sub(r"\W+", "_", Path(folder).name).strip("_") or "video_features"
    if stem[0].isdigit():
        stem = f"f_{stem}"
    taken_set = set(taken)
    name, n = stem, 2
    while name in taken_set:
        name, n = f"{stem}_{n}", n + 1
    return name


def external_vars(ds: xr.Dataset) -> list[str]:
    """The data_vars of *ds* that were attached in memory (never to be saved)."""
    return [str(name) for name, var in ds.data_vars.items() if var.attrs.get(EXTERNAL)]


def external_dirs(ds: xr.Dataset) -> dict[str, str]:
    """``{variable: folder}`` for every attached variable of *ds*."""
    return {name: str(ds[name].attrs.get(EXTERNAL_DIR, "")) for name in external_vars(ds)}


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


def _files_for(dt: TrialTree, alignment: Any, feature: ExternalFeature) -> dict[Any, Path | None]:
    files: dict[Any, Path | None] = {}
    for trial in dt.trials:
        filename = alignment.media_filename(trial, feature.stream, feature.device)
        path = feature.file_for(filename) if filename else None
        files[trial] = path if path is not None and path.is_file() else None
    return files


def detect_device(dt: TrialTree, alignment: Any, folder: str | Path, stream: str = "video") -> str | None:
    """The device whose media names the files in *folder*, or ``None`` for the
    alignment's unnamed default.

    Every device is tried; the one with the most files present wins. A tie
    between two devices that both match is an error naming them, since the
    folder cannot be told apart.
    """
    folder = Path(folder)
    devices: list[str | None] = list(alignment.devices(stream)) or [None]
    counts = {}
    for device in devices:
        files = _files_for(dt, alignment, ExternalFeature("_", folder, stream, device))
        counts[device] = sum(p is not None for p in files.values())
    best = max(counts.values())
    if best == 0:
        return None if None in counts else devices[0]
    winners = [d for d, n in counts.items() if n == best]
    if len(winners) > 1:
        raise ValueError(f"{folder}: its files match {len(winners)} cameras equally ({winners}), so none can be picked")
    return winners[0]


def attach_external_feature(dt: TrialTree, alignment: Any, feature: ExternalFeature) -> AttachResult:
    """Attach *feature* to every trial of *dt*, replacing a variable of that name.

    Nodes are replaced without marking them dirty: nothing here is a change
    to the session file.
    """
    if dt._is_continuous:
        raise ValueError(f"{feature.dir}: external features need a trial tree, not a continuous dataset")
    rate = alignment.get_stream_rate(feature.stream, feature.device)
    if rate is None:
        raise ValueError(
            f"{feature.dir}: {feature.name!r} is on the {feature.stream} clock, but the alignment "
            f"declares no rate for that stream (device {feature.device!r})"
        )
    files = _files_for(dt, alignment, feature)
    found = [p for p in files.values() if p is not None]
    if not found:
        wanted = next(
            (
                feature.file_for(n)
                for n in (alignment.media_filename(t, feature.stream, feature.device) for t in dt.trials)
                if n
            ),
            None,
        )
        raise FileNotFoundError(
            f"{feature.dir}: no file for any trial"
            + (f" (expected e.g. {wanted.name})" if wanted else " — the alignment names no media file for this stream")
        )
    width = int(np.load(found[0], mmap_mode="r").shape[1])
    missing = []
    for trial, path in files.items():
        ds = dt.trial(trial)
        time_coord = get_time_coord(next(iter(ds.data_vars.values())))
        if time_coord is None:
            raise ValueError(f"trial {trial!r} has no time coord to sample {feature.name!r} onto")
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
            offset = float(alignment.stream_offset_for_trial(trial, feature.stream, feature.device))
            values = sample_on_trial(array, time, float(rate), offset)
        da = xr.DataArray(
            values,
            dims=(time_coord.name, feature.feature_dim),
            coords={time_coord.name: time, feature.feature_dim: np.arange(width)},
            name=feature.name,
        )
        schema.describe(
            da,
            feature.kind,
            is_egocentric=False,
            **{EXTERNAL: 1, EXTERNAL_DIR: str(feature.dir), EXTERNAL_FILE: str(path or "")},
        )
        ds = ds.drop_vars([feature.name, feature.feature_dim], errors="ignore")
        dt[dt._trial_node_name(trial)] = xr.DataTree(ds.assign({feature.name: da}))
    logger.info("attached %r (%d columns) to %d trials from %s", feature.name, width, len(files), feature.dir)
    if missing:
        logger.warning("%r has no file for %d trials, which read NaN: %s", feature.name, len(missing), missing)
    return AttachResult(feature.name, feature.dir, feature.device, width, len(files), tuple(missing))
