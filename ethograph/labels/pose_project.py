"""A DeepLabCut / LightningPose project folder, in its own layout.

The folder the two tools share::

    my_project/
    ├── config.yaml                  (DeepLabCut: scorer, bodyparts, individuals)
    ├── CollectedData.csv            (LightningPose: one labels table for every video)
    ├── labeled-data/
    │   └── <video>/                 (one folder per video: the training frames)
    │       ├── img00012.png
    │       └── CollectedData_<scorer>.csv / .h5   (DeepLabCut: one table per video)
    └── videos/
        ├── <video>.mp4
        └── <video>DLC_<model>.h5    (the model's predictions on the whole video)

Ethograph refines such a project in two stages, each an ordinary session
folder under ``<root>/.ethograph/`` (:func:`session_dir`), so nothing in the
tool's own layout is touched except ``labeled-data/`` and its labels tables:

- **Extract frames** (:func:`build_extract_session`): one trial per video in
  ``videos/``, the model's predictions as features (position, confidence,
  velocity, speed), so the frames worth labelling are picked off the curves
  rather than by eye. Segments and single frames are ordinary labels
  (:data:`EXTRACT_SEGMENT_LABEL`, :data:`EXTRACT_FRAME_LABEL`) and
  :func:`extract_frames` turns them into ``labeled-data/<video>/`` PNGs plus a
  labels table prefilled with the predictions.
- **Refine pose** (:func:`build_refine_session`): one trial per folder in
  ``labeled-data/``, the folder standing in for a video
  (:mod:`ethograph.io.image_sequence`), and its labels table edited in place.
  A trial is curated when every frame has been reviewed; the verdict lives in
  the session's ``metadata.tsv`` like any other curated trial.

Everything here is Qt-free; the GUI mode lives in ``gui/pose_project_mode.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import natsort
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from ethograph.io import schema
from ethograph.io.image_sequence import IMAGE_SEQUENCE_RATE, image_files, is_image_folder
from ethograph.io.nc_drop import add_speed
from ethograph.io.netcdf import netcdf_engine
from ethograph.io.pairing import pair_media
from ethograph.io.session_layout import alignment_path, metadata_path, settings_dir
from ethograph.io.trialtree import TrialTree
from ethograph.io.validation import VIDEO_EXTENSIONS
from ethograph.io.video_probe import probe_video
from ethograph.labels.curation import CURATED_COLUMN, CURATED_NO, CURATED_YES

logger = logging.getLogger(__name__)

STAGE_EXTRACT = "extract"
STAGE_REFINE = "refine"
STAGES = (STAGE_EXTRACT, STAGE_REFINE)
_STAGE_DIRS = {STAGE_EXTRACT: "extract_frames", STAGE_REFINE: "refine_pose"}

LAYOUT_DEEPLABCUT = "deeplabcut"
LAYOUT_LIGHTNINGPOSE = "lightningpose"

VIDEOS_DIR = "videos"
LABELED_DATA_DIR = "labeled-data"
CONFIG_FILE = "config.yaml"
#: ``labeled-data/<video>_labeled/`` is a rendering of the labels (``check_labels``), never a trial.
LABELED_SUFFIX = "_labeled"
LP_COLLECTED_DATA = "CollectedData.csv"
DLC_H5_KEY = "df_with_missing"

#: The two label classes of the extract stage, in the session's ``mapping.txt``.
EXTRACT_SEGMENT_LABEL = 1
EXTRACT_FRAME_LABEL = 2
MAPPING_TEXT = f"{EXTRACT_SEGMENT_LABEL} ExtractSegment 0 state\n{EXTRACT_FRAME_LABEL} ExtractFrame 0 point\n"

METHOD_UNIFORM = "uniform"
METHOD_DIVERSE = "diverse"
EXTRACT_METHODS = (METHOD_UNIFORM, METHOD_DIVERSE)

#: The camera every project video is filmed by: the layout names no camera.
CAMERA = "cam-1"
#: movement's name for the one individual of a single-animal file.
SINGLE_INDIVIDUAL = "individual_0"


class PoseProjectError(ValueError):
    """The folder is not a DeepLabCut / LightningPose project."""


# ----------------------------------------------------------------------
# The project
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedVideo:
    """A video of ``videos/`` and the model's predictions on it, if any."""

    video: Path
    tracking: Path | None
    #: movement's ``source_software`` for *tracking*.
    software: str | None

    @property
    def stem(self) -> str:
        return self.video.stem


@dataclass
class PoseProject:
    root: Path
    layout: str
    scorer: str
    keypoints: list[str]
    #: The named individuals of a multi-animal table; ``[]`` for a single-animal one.
    individuals: list[str]

    @property
    def multi_animal(self) -> bool:
        return bool(self.individuals)

    @property
    def individual_names(self) -> list[str]:
        """The individuals as a labels table spells them: one placeholder when single-animal."""
        return list(self.individuals) if self.individuals else [SINGLE_INDIVIDUAL]

    @property
    def videos_dir(self) -> Path:
        return self.root / VIDEOS_DIR

    @property
    def labeled_dir(self) -> Path:
        return self.root / LABELED_DATA_DIR

    #: Whether the scorer came from ``config.yaml`` (DeepLabCut fixes it there:
    #: the table has to be ``CollectedData_<scorer>`` for training to find it).
    scorer_from_config: bool = False

    @classmethod
    def scorer_of(cls, root: str | Path) -> str | None:
        """The scorer *root* already names — its config's, else its labels table's — or ``None``."""
        root = Path(root)
        config = _read_config(root / CONFIG_FILE)
        scorer, _keypoints, _individuals = _schema_from_config(config)
        if scorer:
            return scorer
        layout = LAYOUT_LIGHTNINGPOSE if (root / LP_COLLECTED_DATA).is_file() else _layout_of_config(config)
        table = _first_labels_table(root, layout)
        return table_scorer(table) or None if table is not None else None

    @classmethod
    def open(cls, root: str | Path, scorer: str | None = None) -> PoseProject:
        """Read the project at *root*; refused unless it has ``videos/`` and ``labeled-data/``.

        *scorer* names who labels when the project itself does not (no
        DeepLabCut config): a LightningPose table's scorer column is the
        labeller's name, and a lab has several. A config's scorer always wins.
        """
        root = Path(root).resolve()
        if not root.is_dir():
            raise PoseProjectError(f"{root} is not a folder")
        missing = [name for name in (VIDEOS_DIR, LABELED_DATA_DIR) if not (root / name).is_dir()]
        if missing:
            raise PoseProjectError(
                f"{root} is not a DeepLabCut / LightningPose project: it has no {' or '.join(missing)} folder"
            )
        config = _read_config(root / CONFIG_FILE)
        layout = LAYOUT_LIGHTNINGPOSE if (root / LP_COLLECTED_DATA).is_file() else _layout_of_config(config)
        scorer_named, keypoints, individuals = _schema_from_config(config)
        if not keypoints:
            table = _first_labels_table(root, layout)
            if table is not None:
                scorer_named = scorer_named or table_scorer(table)
                keypoints, individuals = table_schema(table)
        if not keypoints:
            tracked = [v for v in _videos_of(root, layout) if v.tracking is not None]
            if tracked:
                keypoints, individuals = _schema_from_tracking(tracked[0])
        if not keypoints:
            raise PoseProjectError(
                f"{root} names no keypoints: no config.yaml with bodyparts, no labels table and no predictions"
            )
        from_config = bool(_schema_from_config(config)[0])
        scorer = scorer if (scorer and not from_config) else scorer_named
        if not scorer:
            raise PoseProjectError(f"{root} names no scorer: pass the labeller's name")
        return cls(
            root=root,
            layout=layout,
            scorer=scorer,
            keypoints=keypoints,
            individuals=individuals,
            scorer_from_config=from_config,
        )

    def videos(self) -> list[TrackedVideo]:
        """Every video of ``videos/`` in natural order, paired with its predictions by name."""
        return _videos_of(self.root, self.layout)

    def labeled_folders(self) -> list[Path]:
        """Every ``labeled-data/<video>/`` holding at least one image, in natural order."""
        folders = (
            [p for p in self.labeled_dir.iterdir() if is_image_folder(p) and not p.name.endswith(LABELED_SUFFIX)]
            if self.labeled_dir.is_dir()
            else []
        )
        return natsort.natsorted(folders, key=lambda p: p.name)

    def session_dir(self, stage: str) -> Path:
        """The session folder of *stage*: ``<root>/.ethograph/<stage>/``."""
        if stage not in _STAGE_DIRS:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        return self.root / ".ethograph" / _STAGE_DIRS[stage]

    def labels_table(self, video_stem: str) -> LabelsTable:
        return LabelsTable(self, video_stem)


def _read_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _layout_of_config(config: dict) -> str:
    """LightningPose configs nest their keypoints under ``data``; DeepLabCut's are flat."""
    data = config.get("data")
    if isinstance(data, dict) and data.get("keypoint_names"):
        return LAYOUT_LIGHTNINGPOSE
    return LAYOUT_DEEPLABCUT


def _schema_from_config(config: dict) -> tuple[str, list[str], list[str]]:
    """``(scorer, keypoints, individuals)`` from a DeepLabCut or LightningPose config."""
    scorer = str(config.get("scorer") or "")
    data = config.get("data")
    if isinstance(data, dict) and data.get("keypoint_names"):
        return scorer, [str(k) for k in data["keypoint_names"]], []
    if config.get("multianimalproject"):
        keypoints = [str(k) for k in (config.get("multianimalbodyparts") or [])]
        keypoints += [str(k) for k in (config.get("uniquebodyparts") or []) if str(k) not in keypoints]
        individuals = [str(i) for i in (config.get("individuals") or [])]
        return scorer, keypoints, individuals
    return scorer, [str(k) for k in (config.get("bodyparts") or [])], []


def _schema_from_tracking(video: TrackedVideo) -> tuple[list[str], list[str]]:
    from movement.io import load_dataset

    ds = load_dataset(str(video.tracking), source_software=video.software)
    keypoints = [str(k) for k in ds.coords["keypoint"].values]
    individuals = [str(i) for i in ds.coords["individual"].values]
    if individuals == [SINGLE_INDIVIDUAL]:
        individuals = []
    return keypoints, individuals


def _first_labels_table(root: Path, layout: str) -> pd.DataFrame | None:
    if layout == LAYOUT_LIGHTNINGPOSE:
        path = root / LP_COLLECTED_DATA
        return read_labels_table(path) if path.is_file() else None
    labeled = root / LABELED_DATA_DIR
    for folder in natsort.natsorted(labeled.iterdir(), key=lambda p: p.name) if labeled.is_dir() else []:
        found = find_dlc_table(folder)
        if found is not None:
            return read_labels_table(found)
    return None


def _videos_of(root: Path, layout: str) -> list[TrackedVideo]:
    videos_dir = root / VIDEOS_DIR
    if not videos_dir.is_dir():
        return []
    videos = natsort.natsorted(
        [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.name,
    )
    return [TrackedVideo(video, *find_tracking(root, video.stem, layout)) for video in videos]


def find_tracking(root: Path, stem: str, layout: str) -> tuple[Path | None, str | None]:
    """The model's predictions for video *stem*: ``(file, source_software)``.

    DeepLabCut writes ``videos/<stem>DLC_<model>.h5`` (a ``.csv`` twin, and a
    ``_filtered`` copy); LightningPose writes ``video_preds/<stem>.csv`` under
    a model folder. The unfiltered ``.h5`` wins, then the csv, and a file
    named exactly ``<stem>.csv`` beside the video is LightningPose's.
    """
    videos_dir = root / VIDEOS_DIR
    dlc = [
        p
        for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".h5", ".csv") and p.stem.startswith(stem) and "DLC" in p.stem
    ]
    if dlc:
        dlc.sort(key=lambda p: ("_filtered" in p.stem, p.suffix.lower() != ".h5", p.name))
        return dlc[0], "DeepLabCut"
    beside = videos_dir / f"{stem}.csv"
    if beside.is_file():
        return beside, "LightningPose"
    for candidate in natsort.natsorted(root.rglob(f"video_preds/{stem}.csv"), key=lambda p: str(p)):
        return candidate, "LightningPose"
    if layout == LAYOUT_DEEPLABCUT:
        loose = [
            p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".h5" and p.stem.startswith(stem)
        ]
        if loose:
            return natsort.natsorted(loose, key=lambda p: p.name)[0], "DeepLabCut"
    return None, None


# ----------------------------------------------------------------------
# Labels tables (CollectedData)
# ----------------------------------------------------------------------


def find_dlc_table(folder: Path) -> Path | None:
    """The one ``CollectedData_<scorer>`` of a DeepLabCut video folder (csv first, else h5)."""
    for suffix in (".csv", ".h5"):
        found = sorted(folder.glob(f"CollectedData_*{suffix}"))
        if len(found) > 1:
            raise PoseProjectError(f"{folder} holds several labels tables: {', '.join(p.name for p in found)}")
        if found:
            return found[0]
    return None


def read_labels_table(path: Path) -> pd.DataFrame:
    """A labels table with a ``(labeled-data, video, image)`` row index, whichever file it came from.

    DeepLabCut's csv and h5 carry the three-level path index; LightningPose's
    single csv keys rows by the relative path string, split here so both read
    the same. The csv is read without an index column and split afterwards:
    pandas misreads a row whose every value is empty (a frame not yet
    labelled) when asked for the index up front.
    """
    path = Path(path)
    if path.suffix.lower() == ".h5":
        df = pd.read_hdf(path, key=DLC_H5_KEY)
        rows = [tuple(str(x) for x in row) if isinstance(row, tuple) else _split_image_path(row) for row in df.index]
    else:
        header_rows = 4 if _is_multi_animal_csv(path) else 3
        raw = pd.read_csv(path, header=list(range(header_rows)))
        is_index = [str(col[-1]) not in ("x", "y", "likelihood") for col in raw.columns]
        index_cols = [col for col, flag in zip(raw.columns, is_index) if flag]
        rows = [
            tuple(str(x) for x in row) if len(index_cols) == 3 else _split_image_path(row[0])
            for row in raw[index_cols].itertuples(index=False, name=None)
        ]
        df = raw[[col for col, flag in zip(raw.columns, is_index) if not flag]]
        # The level names sit in the first header cells, which pandas read as a column.
        names = (
            ["scorer", "individuals", "bodyparts", "coords"] if header_rows == 4 else ["scorer", "bodyparts", "coords"]
        )
        df.columns = pd.MultiIndex.from_tuples(list(df.columns), names=names)
    df = df.copy()
    df.index = pd.MultiIndex.from_tuples(rows) if rows else pd.MultiIndex(levels=[[], [], []], codes=[[], [], []])
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(str(x) for x in col) for col in df.columns], names=list(df.columns.names)
    )
    return df.astype(np.float64)


def _split_image_path(value) -> tuple[str, str, str]:
    parts = str(value).replace("\\", "/").split("/")
    if len(parts) < 3:
        raise PoseProjectError(f"labels row {value!r} is not a labeled-data/<video>/<image> path")
    return parts[-3], parts[-2], parts[-1]


def _is_multi_animal_csv(path: Path) -> bool:
    with open(path, encoding="utf-8") as fh:
        lines = [fh.readline() for _ in range(4)]
    return lines[1].startswith("individuals")


def table_scorer(df: pd.DataFrame) -> str:
    return str(df.columns[0][0]) if len(df.columns) else ""


def table_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """``(keypoints, individuals)`` of a labels table; ``[]`` individuals when single-animal."""
    keypoints = list(dict.fromkeys(str(k) for k in df.columns.get_level_values("bodyparts")))
    if "individuals" in (df.columns.names or []):
        individuals = list(dict.fromkeys(str(i) for i in df.columns.get_level_values("individuals")))
    else:
        individuals = []
    return keypoints, individuals


def labels_rows(
    frames: dict[str, np.ndarray],
    video_stem: str,
    keypoints: Sequence[str],
    individuals: Sequence[str],
    scorer: str,
) -> pd.DataFrame:
    """*frames* (``image name -> (n_individuals, n_keypoints, 2)``) as labels-table rows."""
    multi = bool(individuals)
    names = list(individuals) if multi else [SINGLE_INDIVIDUAL]
    columns: list[tuple[str, ...]] = []
    for individual in names:
        for keypoint in keypoints:
            for coord in ("x", "y"):
                columns.append((scorer, individual, keypoint, coord) if multi else (scorer, keypoint, coord))
    level_names = ["scorer", "individuals", "bodyparts", "coords"] if multi else ["scorer", "bodyparts", "coords"]
    images = natsort.natsorted(frames)
    index = pd.MultiIndex.from_tuples([(LABELED_DATA_DIR, video_stem, image) for image in images])
    if images:
        values = np.stack([np.asarray(frames[image], dtype=np.float64).reshape(-1) for image in images])
    else:
        values = np.empty((0, len(columns)))
    return pd.DataFrame(values, index=index, columns=pd.MultiIndex.from_tuples(columns, names=level_names))


class LabelsTable:
    """The labels of one video, wherever the layout keeps them.

    DeepLabCut: ``labeled-data/<video>/CollectedData_<scorer>.csv`` and its
    ``.h5`` twin, both rewritten on every save. LightningPose: the rows of
    this video inside the project's one ``CollectedData.csv``; other videos'
    rows are kept verbatim.
    """

    def __init__(self, project: PoseProject, video_stem: str):
        self.project = project
        self.video_stem = video_stem

    @property
    def folder(self) -> Path:
        return self.project.labeled_dir / self.video_stem

    @property
    def path(self) -> Path:
        if self.project.layout == LAYOUT_LIGHTNINGPOSE:
            return self.project.root / LP_COLLECTED_DATA
        existing = find_dlc_table(self.folder) if self.folder.is_dir() else None
        if existing is not None:
            return existing.with_suffix(".csv")
        return self.folder / f"CollectedData_{self.project.scorer}.csv"

    def exists(self) -> bool:
        return self.path.is_file() or self.path.with_suffix(".h5").is_file()

    def _source(self) -> Path | None:
        if self.path.is_file():
            return self.path
        h5 = self.path.with_suffix(".h5")
        return h5 if h5.is_file() else None

    def read_table(self) -> pd.DataFrame | None:
        source = self._source()
        return read_labels_table(source) if source is not None else None

    def schema(self) -> tuple[list[str], list[str], str]:
        """``(keypoints, individuals, scorer)``: the table's own when it exists, else the project's."""
        table = self.read_table()
        if table is not None and len(table.columns):
            keypoints, individuals = table_schema(table)
            return keypoints, individuals, table_scorer(table)
        return list(self.project.keypoints), list(self.project.individuals), self.project.scorer

    def read(self) -> dict[str, np.ndarray]:
        """``image name -> (n_individuals, n_keypoints, 2)`` for this video, in :meth:`schema` order."""
        table = self.read_table()
        if table is None or table.empty:
            return {}
        keypoints, individuals, scorer = self.schema()
        names = individuals or [SINGLE_INDIVIDUAL]
        multi = bool(individuals)
        mine = table[[row[1] == self.video_stem for row in table.index]]
        frames: dict[str, np.ndarray] = {}
        for row in mine.index:
            positions = np.full((len(names), len(keypoints), 2), np.nan)
            for i, individual in enumerate(names):
                for k, keypoint in enumerate(keypoints):
                    for c, coord in enumerate(("x", "y")):
                        key = (scorer, individual, keypoint, coord) if multi else (scorer, keypoint, coord)
                        if key in mine.columns:
                            positions[i, k, c] = float(mine.at[row, key])
            frames[str(row[2])] = positions
        return frames

    def write(self, frames: dict[str, np.ndarray]) -> list[Path]:
        """Replace this video's rows with *frames* (in :meth:`schema` order); returns what was written."""
        keypoints, individuals, scorer = self.schema()
        new = labels_rows(frames, self.video_stem, keypoints, individuals, scorer)
        existing = self.read_table()
        if existing is not None and len(existing.columns):
            others = existing[[row[1] != self.video_stem for row in existing.index]]
            columns = list(existing.columns) + [c for c in new.columns if c not in existing.columns]
            merged = pd.concat([others, new]).reindex(
                columns=pd.MultiIndex.from_tuples(columns, names=list(new.columns.names))
            )
        else:
            merged = new
        merged = merged.loc[natsort.natsorted(merged.index, key=lambda row: "/".join(row))]
        return write_labels_table(merged, self.path, self.project.layout)


def write_labels_table(df: pd.DataFrame, path: Path, layout: str) -> list[Path]:
    """Write a labels table the way its layout reads it back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout == LAYOUT_LIGHTNINGPOSE:
        flat = df.copy()
        flat.index = pd.Index(["/".join(row) for row in df.index])
        flat.to_csv(path)
        return [path]
    df.to_csv(path)
    written = [path]
    h5 = path.with_suffix(".h5")
    try:
        df.to_hdf(h5, key=DLC_H5_KEY, mode="w")
    except ImportError:
        # pytables is optional; DeepLabCut's convertcsv2h5 rebuilds the twin.
        return written
    written.append(h5)
    return written


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------


Progress = Callable[[float, str], None]


@dataclass
class ExtractSessionOutcome:
    session_dir: Path
    trials: list[str]
    #: Videos that entered the session without predictions: video only, no curves.
    untracked: list[str]


def tracking_dataset(video: TrackedVideo, fps: float) -> xr.Dataset:
    """The model's predictions on *video* as a movement dataset with velocity and speed."""
    from movement.io import load_dataset
    from movement.kinematics import compute_velocity

    if video.tracking is None:
        raise ValueError(f"{video.video.name} has no predictions")
    ds = load_dataset(str(video.tracking), source_software=video.software, fps=fps)
    velocity = schema.describe(compute_velocity(ds["position"]), schema.KINEMATIC_FEATURE, is_egocentric=False)
    ds = add_speed(ds.assign(velocity=velocity))
    ds.attrs = {k: v for k, v in ds.attrs.items() if v is not None}
    ds.attrs["fps"] = float(fps)
    ds.attrs["source_software"] = video.software
    return ds


def _sources_signature(videos: Sequence[TrackedVideo]) -> list[list]:
    """What ``session.nc`` was built from: every video and prediction file with its mtime."""
    signature: list[list] = []
    for video in videos:
        for path in (video.video, video.tracking):
            if path is not None:
                signature.append([str(path), int(path.stat().st_mtime_ns)])
    return signature


def build_extract_session(project: PoseProject, progress: Progress | None = None) -> ExtractSessionOutcome:
    """The extract stage's session: one trial per video, predictions as features.

    The alignment is rewritten every time (derived data, cheap); ``session.nc``
    only when a video or prediction file changed since it was built — it is
    the file the GUI holds open, and reading every prediction file again is
    what a stage switch should not cost. ``labels.tsv`` — the segments and
    frames already chosen — is kept.
    """
    videos = project.videos()
    if not videos:
        raise PoseProjectError(f"{project.videos_dir} holds no video")
    session_dir = project.session_dir(STAGE_EXTRACT)
    session_dir.mkdir(parents=True, exist_ok=True)
    signature_path = settings_dir(session_dir) / "sources.json"
    signature = _sources_signature(videos)
    nc_path = session_dir / "session.nc"
    unchanged = (
        nc_path.is_file()
        and signature_path.is_file()
        and json.loads(signature_path.read_text(encoding="utf-8")) == signature
    )
    existing_trials: list[str] = []
    individuals: list[str] = []
    if unchanged:
        existing_trials, individuals = _nc_summary(nc_path)
        # A file from before every video was a trial (blank ones included) is stale too.
        unchanged = set(existing_trials) == {v.stem for v in videos}
    rows: list[dict] = []
    datasets: list[xr.Dataset] = []
    untracked: list[str] = []
    clock = 0.0
    blanks: list[tuple[str, int, float]] = []
    for i, video in enumerate(videos):
        if progress is not None:
            progress(i / len(videos), video.video.name)
        probe = probe_video(str(video.video))
        duration = probe.nframes / probe.fps
        row = {
            "trial": video.stem,
            "start_time": clock,
            "stop_time": clock + duration,
            f"video_{CAMERA}": str(video.video),
        }
        if video.tracking is None:
            untracked.append(video.stem)
            blanks.append((video.stem, probe.nframes, probe.fps))
        elif not unchanged:
            ds = tracking_dataset(video, probe.fps)
            ds.attrs["trial"] = video.stem
            datasets.append(ds)
            for name in ds.coords["individual"].values:
                if str(name) not in individuals:
                    individuals.append(str(name))
        rows.append(row)
        clock += duration
    table = pd.DataFrame(rows)
    rates = {"video": float(probe_video(str(videos[0].video)).fps)}
    if unchanged:
        pass
    elif datasets:
        # A video without predictions is still a trial: an all-NaN copy of the
        # first tracked video's variables on its own clock, so its panels are
        # empty rather than still showing the trial before it.
        datasets += [blank_like(datasets[0], stem, n_frames, fps) for stem, n_frames, fps in blanks]
        tree = TrialTree.from_datasets(datasets, validate=True)
        tree.to_netcdf(nc_path, engine=netcdf_engine(nc_path))
    elif nc_path.exists():
        nc_path.unlink()
    settings_dir(session_dir).mkdir(parents=True, exist_ok=True)
    signature_path.write_text(json.dumps(signature), encoding="utf-8")
    if datasets:
        seed_extract_layout(session_dir, datasets[0])
    pair_media(
        trial_table=table,
        stream_rates=rates,
        output_path=alignment_path(session_dir),
        individuals=individuals or project.individual_names,
        on_existing="replace",
    )
    mapping = settings_dir(session_dir) / "mapping.txt"
    mapping.write_text(MAPPING_TEXT, encoding="utf-8")
    return ExtractSessionOutcome(session_dir=session_dir, trials=[v.stem for v in videos], untracked=untracked)


def blank_like(template: xr.Dataset, trial: str, n_frames: int, fps: float) -> xr.Dataset:
    """*template*'s variables, all NaN, over *n_frames* at *fps* — a trial with nothing tracked."""
    time = np.arange(int(n_frames)) / float(fps)
    blank = xr.full_like(template, np.nan).reindex(time=time)
    blank.attrs = dict(template.attrs)
    blank.attrs["trial"] = trial
    blank.attrs["fps"] = float(fps)
    return blank


LOCAL_SETTINGS = "local_settings.yaml"


def default_extract_layout(ds: xr.Dataset) -> dict:
    """The panels the extract stage opens with.

    One keypoint's position (both space coordinates as lines) and speed, and
    every keypoint's confidence as a heatmap: where the animal is, how fast
    it moves and where the model was unsure are what say which frames are
    worth labelling. Each panel leaves exactly one dim free — ``space`` for
    the position, ``keypoint`` for the heatmap.
    """
    individual = str(ds.coords["individual"].values[0])
    keypoint = str(ds.coords["keypoint"].values[0])
    pinned = {"individual": individual, "keypoint": keypoint}
    return {
        "panels": [
            {"type": "lineplot", "feature": "position", "selections": dict(pinned)},
            {"type": "lineplot", "feature": "speed", "selections": dict(pinned)},
            {"type": "heatmap", "feature": "confidence", "selections": {"individual": individual}},
        ]
    }


def seed_extract_layout(session_dir: Path, ds: xr.Dataset) -> Path | None:
    """Write the default panel layout into the session's ``local_settings.yaml`` — once.

    A file that exists is the user's (the GUI saves the layout there), so it
    is never touched; returns the path written, or ``None``.
    """
    path = settings_dir(session_dir) / LOCAL_SETTINGS
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"panel_layout": default_extract_layout(ds)}), encoding="utf-8")
    return path


def _nc_summary(nc_path: Path) -> tuple[list[str], list[str]]:
    """``(trials, individuals)`` a written ``session.nc`` names."""
    from ethograph.io.data_loader import _open_trialtree

    tree = _open_trialtree(str(nc_path))
    try:
        trials: list[str] = []
        names: list[str] = []
        for trial, ds in tree.trial_items():
            trials.append(str(trial))
            for name in ds.coords["individual"].values if "individual" in ds.coords else []:
                if str(name) not in names:
                    names.append(str(name))
        return trials, names
    finally:
        tree.close()


def build_refine_session(project: PoseProject) -> Path:
    """The refine stage's session: one trial per ``labeled-data/<video>/`` folder.

    The folder is the trial's video; its rate is the image-sequence clock.
    ``metadata.tsv`` keeps each folder's curated verdict, and a folder seen
    for the first time starts uncurated.
    """
    folders = project.labeled_folders()
    session_dir = project.session_dir(STAGE_REFINE)
    session_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    clock = 0.0
    for folder in folders:
        duration = len(image_files(folder)) / IMAGE_SEQUENCE_RATE
        rows.append(
            {"trial": folder.name, "start_time": clock, "stop_time": clock + duration, f"video_{CAMERA}": str(folder)}
        )
        clock += duration
    if not rows:
        raise PoseProjectError(f"{project.labeled_dir} holds no folder with images yet — extract frames first")
    pair_media(
        trial_table=pd.DataFrame(rows),
        stream_rates={"video": IMAGE_SEQUENCE_RATE},
        output_path=alignment_path(session_dir),
        individuals=project.individual_names,
        on_existing="replace",
    )
    write_curated(session_dir, [f.name for f in folders], read_curated(session_dir))
    return session_dir


def read_curated(session_dir: Path) -> dict[str, bool]:
    """``{folder: curated?}`` from the refine session's ``metadata.tsv``; empty when there is none."""
    path = metadata_path(session_dir)
    if not path.is_file():
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "trial" not in df.columns or CURATED_COLUMN not in df.columns:
        return {}
    return {str(t): str(v).strip().lower() == CURATED_YES for t, v in zip(df["trial"], df[CURATED_COLUMN])}


def write_curated(session_dir: Path, trials: Sequence[str], status: dict[str, bool]) -> Path:
    """``metadata.tsv`` with one row per trial and its verdict (missing = not curated)."""
    path = metadata_path(session_dir)
    df = pd.DataFrame(
        {
            "trial": list(trials),
            CURATED_COLUMN: [CURATED_YES if status.get(str(t), False) else CURATED_NO for t in trials],
        }
    )
    df.to_csv(path, sep="\t", index=False)
    return path


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------


def plan_frames(
    n_frames: int,
    segments: Sequence[tuple[int, int]],
    points: Sequence[int],
    method: str,
    coverage: float,
    frames=None,
    exclude: set[int] | None = None,
) -> list[int]:
    """Which frames of a video to extract.

    Every *point* is taken. Each *segment* ``(first, last)`` (inclusive)
    contributes ``coverage`` of its frames: evenly spaced (``uniform``) or one
    per k-means cluster of frame thumbnails (``diverse``, DeepLabCut's
    method, which needs *frames*). Frames in *exclude* — already in the
    folder — are never picked twice.
    """
    from ethograph.gui.pose_suggest import suggest_within

    if method not in EXTRACT_METHODS:
        raise ValueError(f"method must be one of {EXTRACT_METHODS}, got {method!r}")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    exclude = set(exclude or ())
    chosen: set[int] = {int(p) for p in points if 0 <= int(p) < n_frames and int(p) not in exclude}
    for first, last in segments:
        first, last = max(0, int(first)), min(n_frames - 1, int(last))
        if last < first:
            continue
        candidates = [f for f in range(first, last + 1) if f not in exclude and f not in chosen]
        if not candidates:
            continue
        count = max(1, int(round(coverage * (last - first + 1))))
        chosen.update(suggest_within(method, count, candidates, frames))
    return sorted(chosen)


@dataclass
class ExtractOutcome:
    folder: Path
    images: list[Path]
    table: Path


def image_digits(folder: Path, n_video_frames: int) -> int:
    """Zero-padding of ``img{N}.png``: what the folder already uses, else what the video needs."""
    from ethograph.gui.pose_training_export import frame_digits

    return frame_digits([p.name for p in image_files(folder)], n_video_frames)


def frame_index_of(name: str) -> int | None:
    """The video frame an ``img{N}.png`` names, or ``None`` for any other file name."""
    stem = Path(name).stem
    if not stem.startswith("img") or not stem[3:].isdigit():
        return None
    return int(stem[3:])


def extract_frames(
    project: PoseProject,
    video: TrackedVideo,
    frames: Sequence[int],
    decode: Callable[[int], np.ndarray],
    n_video_frames: int,
    predictions: dict[int, np.ndarray] | None = None,
    progress: Callable[[float], bool] | None = None,
) -> ExtractOutcome:
    """Write *frames* of *video* into ``labeled-data/<video>/`` with their predicted pose.

    *decode* returns the RGB image of a video frame; *predictions* maps a
    frame to its ``(n_individuals, n_keypoints, 2)`` pose in the table's
    schema order (missing = NaN row, to be labelled from scratch). Frames
    already in the folder keep their rows; new ones are added.
    """
    from ethograph.gui.pose_training_export import image_name, write_frame_image

    table = project.labels_table(video.stem)
    folder = table.folder
    folder.mkdir(parents=True, exist_ok=True)
    digits = image_digits(folder, n_video_frames)
    keypoints, individuals, _scorer = table.schema()
    n_ind = len(individuals) if individuals else 1
    existing = table.read()
    images: list[Path] = []
    for i, frame in enumerate(frames):
        name = image_name(int(frame), digits)
        path = folder / name
        if not path.is_file():
            write_frame_image(decode(int(frame)), path)
        images.append(path)
        if name not in existing:
            pose = predictions.get(int(frame)) if predictions else None
            existing[name] = (
                np.asarray(pose, dtype=np.float64) if pose is not None else np.full((n_ind, len(keypoints), 2), np.nan)
            )
        if progress is not None and not progress((i + 1) / max(1, len(frames))):
            raise InterruptedError("Extraction cancelled")
    written = table.write(existing)
    return ExtractOutcome(folder=folder, images=images, table=written[0])


def predictions_by_frame(ds: xr.Dataset, keypoints: Sequence[str], individuals: Sequence[str]) -> dict[int, np.ndarray]:
    """``frame -> (n_individuals, n_keypoints, 2)`` from a movement dataset, in the table's order.

    A keypoint or individual the predictions lack is NaN; the frame index is
    the row's position, which is how a DeepLabCut prediction file is laid out.
    """
    names = list(individuals) if individuals else [str(i) for i in ds.coords["individual"].values][:1]
    pos = ds["position"].transpose("time", "individual", "keypoint", "space")
    out = np.full((pos.sizes["time"], len(names), len(keypoints), 2), np.nan)
    kp_index = {str(k): i for i, k in enumerate(pos.coords["keypoint"].values)}
    ind_index = {str(k): i for i, k in enumerate(pos.coords["individual"].values)}
    values = pos.values
    for i, individual in enumerate(names):
        if individual not in ind_index:
            continue
        for k, keypoint in enumerate(keypoints):
            if keypoint not in kp_index:
                continue
            out[:, i, k, :] = values[:, ind_index[individual], kp_index[keypoint], :2]
    return {frame: out[frame] for frame in range(out.shape[0])}
