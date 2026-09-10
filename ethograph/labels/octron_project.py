"""An OCTRON project on disk, spoken in OCTRON's own layout (Qt-free, torch-free).

EthoGraph's box labelling drives the OCTRON fork headless. This module owns
the *files* OCTRON expects, so anything written here opens unchanged in the
OCTRON GUI and trains with OCTRON's own ``octron split`` / ``octron train``::

    {project}/octron/                       # the OCTRON project root (OCTRON_DIRNAME)
    ├── octron.yaml                         # EthoGraph's training/prediction config
    ├── {hash8}/                            # one folder per camera video
    │   ├── video data.zarr                 # SAM's resized frame cache (written by the SAM session)
    │   ├── video_info.txt                  # informational; read by OCTRON's organizer recovery
    │   ├── {label} {suffix} masks.zarr     # one per tracked object, full video resolution
    │   └── object_organizer.json           # the index OCTRON's training reads
    ├── model/                              # octron split / train output
    └── predictions/                        # octron predict output (per video, per tracker)

The organizer JSON is written in the schema OCTRON's ``restore_object_organizer``
reconstructs, so the two agree by construction. ``label_id`` comes from
:class:`ObjectOrganizer` (OCTRON's own class), and so does the colour.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import zarr

OCTRON_DIRNAME = "octron"
CONFIG_FILENAME = "octron.yaml"
ORGANIZER_FILENAME = "object_organizer.json"
VIDEO_INFO_FILENAME = "video_info.txt"
VIDEO_ZARR = "video data.zarr"
MASK_SUFFIX = " masks.zarr"
PREDICTIONS_DIRNAME = "predictions"
MASK_OPACITY = 0.4

#: How an EthoGraph individual becomes an OCTRON label (= YOLO class).
#: ``suffix`` (the default, OCTRON's 'LED 1 / LED 2' case) trains one class and
#: keeps the individual as OCTRON's suffix — right for animals that look alike,
#: where identity is the tracker's job; ``per_individual`` gives each
#: individual its own class, for animals a detector can tell apart.
LABEL_SCHEMES = ("per_individual", "suffix")
DEFAULT_SAM_MODEL = "sam2_large"


# ---------------------------------------------------------------------------
# OCTRON's own catalogues
# ---------------------------------------------------------------------------


def _catalogue(*parts: str) -> dict[str, dict]:
    import octron

    path = Path(octron.__file__).resolve().parent.joinpath(*parts)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def yolo_models() -> tuple[str, ...]:
    """The YOLO base models OCTRON ships, in its CLI's spelling (``yolo_models.yaml``)."""
    return tuple(str(key).lower() for key in _catalogue("yolo_octron", "yolo_models.yaml"))


def trackers() -> tuple[str, ...]:
    """The trackers OCTRON marks available, in its CLI's spelling (``boxmot_trackers.yaml``)."""
    entries = _catalogue("tracking", "boxmot_trackers.yaml")
    return tuple(str(key).lower() for key, info in entries.items() if info.get("available", False))


# ---------------------------------------------------------------------------
# Config (octron.yaml)
# ---------------------------------------------------------------------------


@dataclass
class OctronConfig:
    """The knobs the Train / Analyze pages expose — nothing OCTRON's CLI cannot take."""

    model: str = "YOLO26m"  # a key of OCTRON's yolo_models.yaml
    imgsz: int = 1024
    epochs: int = 250
    save_period: int = 50
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    seed: int = 88
    batch: str = "auto"  # "auto" or an int, as text
    device: str = "auto"
    prune: bool = False
    sam_model: str = DEFAULT_SAM_MODEL
    label_scheme: str = "suffix"
    class_name: str = "animal"  # the one class under the ``suffix`` scheme
    tracker: str = "bytetrack"
    conf_thresh: float = 0.5
    iou_thresh: float = 0.7
    skip_frames: int = 0
    one_object_per_label: bool = False
    # Frame suggestion (EthoGraph's side, not OCTRON's)
    suggest_motion_share: float = 0.3  # mixed: share of picks from the strongest movements
    suggest_motion_gate: float = 0.5  # mixed: candidates below this motion quantile are not clustered

    def __post_init__(self) -> None:
        if self.label_scheme not in LABEL_SCHEMES:
            raise ValueError(f"label_scheme must be one of {LABEL_SCHEMES}, got {self.label_scheme!r}")
        if not 0 < self.train_fraction < 1 or not 0 <= self.val_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1) and val_fraction in [0, 1)")
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("train_fraction + val_fraction must leave room for a test split")
        if not 0 <= self.suggest_motion_share <= 1 or not 0 <= self.suggest_motion_gate < 1:
            raise ValueError("suggest_motion_share must be in [0, 1] and suggest_motion_gate in [0, 1)")

    @classmethod
    def load(cls, path: Path) -> OctronConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        return cls(**raw)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "# EthoGraph box labelling — OCTRON training/prediction settings (octron.yaml)\n"
        path.write_text(header + yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    def label_for(self, individual: str) -> tuple[str, str]:
        """``(label, suffix)`` OCTRON stores for *individual* under this scheme."""
        name = str(individual).strip()
        if self.label_scheme == "suffix":
            return self.class_name, name.lower()
        return name, ""

    def individual_for(self, label: str, suffix: str) -> str:
        """The individual an OCTRON ``(label, suffix)`` stands for — the inverse of ``label_for``."""
        return suffix if self.label_scheme == "suffix" else label


# ---------------------------------------------------------------------------
# Project + videos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoEntry:
    """One camera video registered in the project."""

    path: Path
    hash8: str
    folder: Path
    width: int
    height: int
    fps: float
    num_frames: int

    @property
    def organizer_path(self) -> Path:
        return self.folder / ORGANIZER_FILENAME


@dataclass
class ObjectEntry:
    """One tracked object: what OCTRON's organizer keeps per ``obj_id``."""

    obj_id: int
    label: str
    suffix: str
    label_id: int
    color: list[float]
    mask_path: Path

    @property
    def layer_name(self) -> str:
        return layer_name(self.label, self.suffix)


def octron_root(project_dir: Path) -> Path:
    return Path(project_dir) / OCTRON_DIRNAME


def hash8_of(video_path: Path) -> str:
    """OCTRON's folder name for a video: the last 8 hex chars of its blake2b."""
    from octron.sam_octron.helpers.video_loader import get_vfile_hash

    return get_vfile_hash(str(video_path))[-8:]


def probe(video_path: Path) -> dict:
    from octron.sam_octron.helpers.video_loader import probe_video

    return probe_video(str(video_path), verbose=False)


def layer_name(label: str, suffix: str) -> str:
    """What OCTRON calls the object's layer — the stem of its mask store."""
    return f"{label} {suffix}".strip()


def mask_zarr_path(folder: Path, label: str, suffix: str) -> Path:
    return folder / (layer_name(label, suffix) + MASK_SUFFIX)


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


class OctronProject:
    """The OCTRON project folder of an EthoGraph project."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def for_project(cls, project_dir: Path) -> OctronProject:
        return cls(octron_root(project_dir))

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def weights_path(self) -> Path:
        return self.model_dir / "training" / "weights" / "best.pt"

    @property
    def training_data_ready(self) -> bool:
        """``octron split`` has exported frames + labels for this project."""
        return (self.model_dir / "training_data" / "yolo_config.yaml").is_file()

    @property
    def predictions_dir(self) -> Path:
        return self.root / PREDICTIONS_DIRNAME

    def load_config(self) -> OctronConfig:
        if self.config_path.is_file():
            return OctronConfig.load(self.config_path)
        return OctronConfig()

    # -- videos -------------------------------------------------------------

    def register_video(self, video_path: Path) -> VideoEntry:
        """Create (or reopen) the video's hash folder and write ``video_info.txt``."""
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        info = probe(video_path)
        h8 = hash8_of(video_path)
        folder = self.root / h8
        folder.mkdir(parents=True, exist_ok=True)
        entry = VideoEntry(
            path=video_path,
            hash8=h8,
            folder=folder,
            width=int(info["width"]),
            height=int(info["height"]),
            fps=float(info["fps"]),
            num_frames=int(info["num_frames"]),
        )
        self._write_video_info(entry)
        return entry

    def _write_video_info(self, entry: VideoEntry) -> None:
        # The layout OCTRON's restore_object_organizer parses ("Video path:",
        # "Video abbreviated hash:"); everything else is for a human.
        lines = [
            "# Written by EthoGraph — informational, not read by training",
            f"Video path: {entry.path.as_posix()}",
            f"Video abbreviated hash: {entry.hash8}",
            f"Width: {entry.width}",
            f"Height: {entry.height}",
            f"FPS: {entry.fps}",
            f"Number of frames: {entry.num_frames}",
            f"Registered: {datetime.now().isoformat(timespec='seconds')}",
        ]
        (entry.folder / VIDEO_INFO_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def video_folders(self) -> list[Path]:
        """Every hash folder holding an organizer, sorted by name."""
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.iterdir() if p.is_dir() and (p / ORGANIZER_FILENAME).is_file())

    # -- objects --------------------------------------------------------------

    def open_mask(self, entry: VideoEntry, label: str, suffix: str) -> zarr.Array:
        """The object's mask store: full video resolution, one frame per chunk, -1 = untouched."""
        from octron.sam_octron.helpers.sam_zarr import create_image_zarr, load_image_zarr

        path = mask_zarr_path(entry.folder, label, suffix)
        if path.exists():
            array, ok = load_image_zarr(
                path,
                num_frames=entry.num_frames,
                image_height=entry.height,
                image_width=entry.width,
                chunk_size=1,
                video_hash_abrrev=entry.hash8,
                verbose=False,
            )
            if ok:
                return array
            shutil.rmtree(path, ignore_errors=True)
        return create_image_zarr(
            path,
            num_frames=entry.num_frames,
            image_height=entry.height,
            image_width=entry.width,
            chunk_size=1,
            fill_value=-1,
            dtype="int16",
            video_hash_abbrev=entry.hash8,
            verbose=False,
        )

    def load_objects(self, entry: VideoEntry) -> list[ObjectEntry]:
        """The objects recorded in the video's organizer, in ``obj_id`` order."""
        if not entry.organizer_path.is_file():
            return []
        raw = json.loads(entry.organizer_path.read_text(encoding="utf-8"))
        objects: list[ObjectEntry] = []
        for obj_id, e in sorted(raw.get("entries", {}).items(), key=lambda kv: int(kv[0])):
            meta = e.get("prediction_layer_metadata", {})
            zarr_rel = meta.get("zarr_path")
            mask_path = Path(zarr_rel or "")
            if zarr_rel and not mask_path.is_absolute():
                mask_path = self.root / zarr_rel
            objects.append(
                ObjectEntry(
                    obj_id=int(obj_id),
                    label=str(e["label"]),
                    suffix=str(e.get("suffix", "")),
                    label_id=int(e["label_id"]),
                    color=[float(c) for c in e["color"]],
                    mask_path=mask_path,
                )
            )
        return objects

    def stored_individuals(self, cfg: OctronConfig, entries: list[VideoEntry]) -> list[str]:
        """The individuals the videos' organizers already hold, in first-seen order.

        What a reopened project starts its individual list from, so nothing
        labelled earlier is invisible.
        """
        names: list[str] = []
        for entry in entries:
            for obj in self.load_objects(entry):
                name = cfg.individual_for(obj.label, obj.suffix)
                if name and name not in names:
                    names.append(name)
        return names

    def ensure_object(
        self, entry: VideoEntry, label: str, suffix: str, color: list[float] | None = None
    ) -> ObjectEntry:
        """The object for ``(label, suffix)``, created with OCTRON's own id + colour if new.

        Ids and colours come from ``restore_object_organizer._compute_colors``,
        OCTRON's reference for rebuilding an organizer: label ids by first
        appearance, colours from its palette with its own fallback.
        """
        from octron.sam_octron.restore_object_organizer import _compute_colors

        suffix = suffix.strip().lower()
        existing = self.load_objects(entry)
        for obj in existing:
            if obj.label == label and obj.suffix == suffix:
                if color is not None and [round(c, 4) for c in obj.color] != [round(c, 4) for c in color]:
                    obj.color = [float(c) for c in color]
                    self.save_organizer(entry, existing, sam_model=None)
                return obj
        pairs = [(obj.label, obj.suffix) for obj in existing] + [(label, suffix)]
        colors, label_id_map = _compute_colors(pairs)
        if color is not None:  # the caller's colour wins (EthoGraph keeps one colour per individual)
            colors[-1] = [float(c) for c in color]
        used = {obj.obj_id for obj in existing}
        obj_id = next(i for i in range(len(existing) + 1) if i not in used)
        new = ObjectEntry(
            obj_id=obj_id,
            label=label,
            suffix=suffix,
            label_id=int(label_id_map[label]),
            color=[float(c) for c in colors[-1]],
            mask_path=mask_zarr_path(entry.folder, label, suffix),
        )
        self.open_mask(entry, label, suffix)
        self.save_organizer(entry, [*existing, new], sam_model=None)
        self.sync_label_ids()  # the label may already have an index in another camera
        return next(o for o in self.load_objects(entry) if o.label == label and o.suffix == suffix)

    def save_organizer(self, entry: VideoEntry, objects: list[ObjectEntry], sam_model: str | None) -> Path:
        """Write ``object_organizer.json`` in the schema OCTRON's training reads."""
        from octron.sam_octron.helpers.sam_zarr import get_annotated_frames

        entries: dict[str, dict] = {}
        for obj in objects:
            n_annotated = 0
            if obj.mask_path.exists():
                n_annotated = int(len(get_annotated_frames(zarr.open(obj.mask_path, mode="r")["masks"])))
            entries[str(obj.obj_id)] = {
                "label": obj.label,
                "suffix": obj.suffix,
                "label_id": obj.label_id,
                "color": obj.color,
                "prediction_layer_metadata": {
                    "name": f"{obj.layer_name} masks",
                    "type": "Labels",
                    "num_predicted_indices": n_annotated,
                    "data_shape": [entry.num_frames, entry.height, entry.width],
                    "ndim": 3,
                    "visible": True,
                    "opacity": MASK_OPACITY,
                    "zarr_path": _relative_posix(obj.mask_path, self.root),
                    "video_file_path": _relative_posix(entry.path, self.root),
                    "video_hash": entry.hash8,
                },
            }
        settings = {"model_name": sam_model} if sam_model else {}
        if entry.organizer_path.is_file():
            old = json.loads(entry.organizer_path.read_text(encoding="utf-8"))
            settings = {**old.get("settings", {}), **settings}
        data = {"entries": entries, "settings": settings, "time_last_changed": datetime.now().isoformat()}
        entry.organizer_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return entry.organizer_path

    def migrate_label_scheme(self, old: OctronConfig, new: OctronConfig) -> int:
        """Rename every stored object from *old*'s ``(label, suffix)`` naming to *new*'s.

        The scheme decides what an individual is called on disk, so changing it
        without moving the masks leaves them answering to a name nothing asks
        for any more — the annotations become invisible and the next click
        starts an empty store beside them. Runs over every registered video,
        not only the open ones. Returns the number of objects renamed.
        """
        renamed = 0
        for folder in self.video_folders():
            organizer = folder / ORGANIZER_FILENAME
            raw = json.loads(organizer.read_text(encoding="utf-8"))
            entries = raw.get("entries", {})
            if not entries:
                continue
            for entry in entries.values():
                meta = entry["prediction_layer_metadata"]
                label, suffix = new.label_for(old.individual_for(str(entry["label"]), str(entry.get("suffix", ""))))
                source = self.root / str(meta["zarr_path"])
                target = mask_zarr_path(folder, label, suffix)
                if source != target and source.exists():
                    shutil.rmtree(target, ignore_errors=True)
                    source.rename(target)
                    renamed += 1
                entry["label"], entry["suffix"] = label, suffix
                meta["name"] = f"{layer_name(label, suffix)} masks"
                meta["zarr_path"] = _relative_posix(target, self.root)
            raw["time_last_changed"] = datetime.now().isoformat()
            organizer.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        self.sync_label_ids()
        return renamed

    def label_ids(self) -> dict[str, int]:
        """The project-wide label → YOLO class index, by first appearance.

        Class indices cannot be decided per video: OCTRON's training asserts
        one label per index across all of them, so a camera whose first object
        is the other individual would abort the run.
        """
        from octron.sam_octron.restore_object_organizer import _compute_colors

        pairs: list[tuple[str, str]] = []
        for folder in self.video_folders():
            raw = json.loads((folder / ORGANIZER_FILENAME).read_text(encoding="utf-8"))
            for _, entry in sorted(raw.get("entries", {}).items(), key=lambda kv: int(kv[0])):
                pair = (str(entry["label"]), str(entry.get("suffix", "")))
                if pair not in pairs:
                    pairs.append(pair)
        if not pairs:
            return {}
        _, label_id_map = _compute_colors(pairs)
        return {str(label): int(index) for label, index in label_id_map.items()}

    def sync_label_ids(self) -> int:
        """Rewrite every organizer's ``label_id`` from :meth:`label_ids`. Returns how many changed."""
        ids = self.label_ids()
        changed = 0
        for folder in self.video_folders():
            organizer = folder / ORGANIZER_FILENAME
            raw = json.loads(organizer.read_text(encoding="utf-8"))
            entries = raw.get("entries", {})
            dirty = False
            for entry in entries.values():
                want = ids.get(str(entry["label"]))
                if want is not None and int(entry["label_id"]) != want:
                    entry["label_id"] = want
                    dirty = True
                    changed += 1
            if dirty:
                raw["time_last_changed"] = datetime.now().isoformat()
                organizer.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return changed

    def remove_object(self, entry: VideoEntry, label: str, suffix: str) -> bool:
        """Delete the object's mask store and organizer entry — OCTRON's Remove label.

        Returns whether anything was removed. Other objects keep their ids and
        colours, as they do in OCTRON, so a remaining object never changes class.
        """
        suffix = suffix.strip().lower()
        objects = self.load_objects(entry)
        keep = [o for o in objects if not (o.label == label and o.suffix == suffix)]
        if len(keep) == len(objects):
            return False
        for obj in objects:
            if obj not in keep and obj.mask_path.exists():
                shutil.rmtree(obj.mask_path, ignore_errors=True)
        if keep:
            self.save_organizer(entry, keep, sam_model=None)
        elif entry.organizer_path.is_file():
            entry.organizer_path.unlink()
        return True

    # -- what a balance table shows ------------------------------------------

    def label_counts(self, entry: VideoEntry) -> dict[str, int]:
        """Annotated frames per object layer name for one video."""
        from octron.sam_octron.helpers.sam_zarr import get_annotated_frames

        counts: dict[str, int] = {}
        for obj in self.load_objects(entry):
            if obj.mask_path.exists():
                counts[obj.layer_name] = int(len(get_annotated_frames(zarr.open(obj.mask_path, mode="r")["masks"])))
            else:
                counts[obj.layer_name] = 0
        return counts

    # -- the commands the Train / Analyze pages run --------------------------

    def split_command(self, cfg: OctronConfig) -> list[str]:
        return [
            "octron",
            "split",
            str(self.root),
            "--mode",
            "detect",
            "--train",
            str(cfg.train_fraction),
            "--val",
            str(cfg.val_fraction),
            "--seed",
            str(cfg.seed),
            "--prune" if cfg.prune else "--no-prune",
        ]

    def train_command(
        self, cfg: OctronConfig, *, overwrite: bool = False, resume: bool = False, regenerate: bool = True
    ) -> list[str]:
        """``octron train``; *regenerate* False passes ``--no-split`` so the exported frames are reused."""
        cmd = [
            "octron",
            "train",
            str(self.root),
            "--model",
            cfg.model,
            "--mode",
            "detect",
            "--device",
            cfg.device,
            "--epochs",
            str(cfg.epochs),
            "--imagesz",
            str(cfg.imgsz),
            "--save-period",
            str(cfg.save_period),
            "--train",
            str(cfg.train_fraction),
            "--val",
            str(cfg.val_fraction),
            "--seed",
            str(cfg.seed),
            "--prune" if cfg.prune else "--no-prune",
        ]
        if overwrite:
            cmd.append("--overwrite")
        if not regenerate:
            cmd.append("--no-split")
        if resume:
            cmd.append("--resume")
        return cmd

    def predict_command(self, cfg: OctronConfig, videos: list[Path], *, overwrite: bool = False) -> list[str]:
        cmd = [
            "octron",
            "predict",
            *[str(v) for v in videos],
            "--model",
            str(self.weights_path),
            "--tracker",
            cfg.tracker,
            "--device",
            cfg.device,
            "--conf-thresh",
            str(cfg.conf_thresh),
            "--iou-thresh",
            str(cfg.iou_thresh),
            "--skip-frames",
            str(cfg.skip_frames),
            "--output-dir",
            str(self.predictions_dir),
        ]
        if cfg.one_object_per_label:
            cmd.append("--one-object-per-label")
        if overwrite:
            cmd.append("--overwrite")
        return cmd

    def prediction_folder(self, video: Path, tracker: str) -> Path:
        """Where ``octron predict`` writes one video's tracks (OCTRON's naming)."""
        return self.predictions_dir / "octron_predictions" / f"{Path(video).stem}_{tracker}"


# ---------------------------------------------------------------------------
# Reading predictions back
# ---------------------------------------------------------------------------


@dataclass
class Track:
    label: str
    track_id: int
    frame_idx: np.ndarray  # (N,) int
    pos_x: np.ndarray  # (N,) float
    pos_y: np.ndarray
    confidence: np.ndarray
    bbox: np.ndarray  # (N, 4) x_min, x_max, y_min, y_max


def read_tracks(folder: Path) -> list[Track]:
    """Every ``{label}_track_{id}.csv`` OCTRON wrote for one video."""
    import pandas as pd

    tracks: list[Track] = []
    for csv in sorted(Path(folder).glob("*_track_*.csv")):
        with open(csv, encoding="utf-8") as f:
            lines = f.readlines()
        # OCTRON writes a metadata header, a blank line, then the CSV.
        try:
            start = lines.index("\n") + 1
        except ValueError:
            start = 0
        from io import StringIO

        df = pd.read_csv(StringIO("".join(lines[start:])))
        if df.empty:
            continue
        stem = csv.stem
        label, _, tid = stem.rpartition("_track_")
        tracks.append(
            Track(
                label=label,
                track_id=int(tid),
                frame_idx=df["frame_idx"].to_numpy(dtype=int),
                pos_x=df["pos_x"].to_numpy(dtype=float),
                pos_y=df["pos_y"].to_numpy(dtype=float),
                confidence=df["confidence"].to_numpy(dtype=float),
                bbox=df[["bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max"]].to_numpy(dtype=float),
            )
        )
    return tracks


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` of the mask's foreground, or ``None`` when empty."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
