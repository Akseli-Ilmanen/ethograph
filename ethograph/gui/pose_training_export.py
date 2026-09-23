"""Export corrected frames as pose-estimation training labels.

The refinement dialog's other output is a corrected pose *file* — every frame,
for downstream analysis. This module is the second purpose: the frames the
user actually looked at, written the way a pose-estimation trainer reads them,
so corrections made in ethograph become training data for the next model.

Two layouts, both **additive**: an export into a folder that already holds
labels keeps every existing frame and only replaces the frames it exports
again, so a project grows over sessions rather than being rewritten.

- **DeepLabCut** — ``labeled-data/{video}/CollectedData_{scorer}.csv`` (+ the
  ``.h5`` twin DeepLabCut trains from) beside ``img{frame}.png`` files, the
  layout ``deeplabcut.extract_frames`` + ``label_frames`` produce. Pointed at a
  project folder (one holding ``config.yaml``) the labels land under its
  ``labeled-data``; pointed anywhere else, the video's folder is made directly
  inside.
- **COCO** — ``images/`` + one ``annotations.json`` with a single category
  whose ``keypoints`` are the file's keypoints; one annotation per
  (image, individual).

What is a training frame: **a frame carrying at least one click of the
user's** (:func:`training_frames`). Its pose is the user's points over the
file's own points — the frame the user reviewed and corrected — and never a
fill: a filled point is an interpolation, and a detector trained on it learns
the interpolation's mistakes as ground truth. That is the whole reason the
dialog marks filling "not recommended" while this purpose is selected.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethograph.gui.pose_annotate import KeypointStore

#: The two layouts, by the key the format combo carries.
FORMAT_DLC = "dlc"
FORMAT_COCO = "coco"

#: DeepLabCut's project file — its presence makes a folder a project.
DLC_CONFIG = "config.yaml"
DLC_LABELED_DATA = "labeled-data"
#: DeepLabCut's fixed HDF key for a labels table.
DLC_H5_KEY = "df_with_missing"

COCO_ANNOTATIONS = "annotations.json"
COCO_IMAGES = "images"
#: COCO's visibility flag for a labelled, visible keypoint.
COCO_VISIBLE = 2

_IMG_RE = re.compile(r"^img(\d+)\.png$")

Progress = Callable[[float], bool]


@dataclass(frozen=True)
class TrainingFrame:
    """One exported frame: its index on the **video** and its reviewed pose."""

    video_frame: int
    #: ``(n_individuals, n_keypoints, 2)`` — NaN where nothing was observed.
    positions: np.ndarray


def training_frames(store: KeypointStore, window_start: int = 0) -> list[TrainingFrame]:
    """The frames worth training on: every frame the user clicked at least once.

    The pose is :meth:`KeypointStore.observation_positions` — the user's points
    over the file's — and the frame index is shifted by *window_start* from the
    store's trial-local grid onto the video's own. Fill is never read.
    """
    return [
        TrainingFrame(video_frame=int(frame) + int(window_start), positions=store.observation_positions(frame))
        for frame in store.anchor_frames()
    ]


# ----------------------------------------------------------------------
# Frame images
# ----------------------------------------------------------------------


def frame_digits(existing_names: Sequence[str], n_video_frames: int) -> int:
    """Zero-padding width of ``img{N}.png`` names.

    A folder that already holds frames dictates it — a second width would put
    the same frame under two names — otherwise as wide as the video needs.
    """
    for name in existing_names:
        match = _IMG_RE.match(Path(name).name)
        if match:
            return len(match.group(1))
    return max(1, len(str(max(0, int(n_video_frames) - 1))))


def image_name(video_frame: int, digits: int) -> str:
    return f"img{int(video_frame):0{digits}d}.png"


def write_frame_image(rgb: np.ndarray, path: Path) -> None:
    """Write one decoded RGB frame as PNG."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write {path}")


# ----------------------------------------------------------------------
# DeepLabCut: CollectedData_{scorer}.csv / .h5 per video folder
# ----------------------------------------------------------------------


def dlc_project_settings(folder: Path) -> tuple[str | None, bool | None]:
    """``(scorer, multianimal)`` from a project's ``config.yaml``, or ``None``s.

    A folder without the file is not a project; a project without the keys is
    an old or hand-edited one and simply does not answer.
    """
    config = Path(folder) / DLC_CONFIG
    if not config.is_file():
        return None, None
    with open(config, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    scorer = payload.get("scorer")
    multi = payload.get("multianimalproject")
    return (str(scorer) if scorer else None), (bool(multi) if multi is not None else None)


def dlc_video_folder(target: Path, video_stem: str) -> Path:
    """Where one video's frames and labels go under *target*.

    A DeepLabCut project (``config.yaml`` present) keeps them under its
    ``labeled-data``; a ``labeled-data`` folder itself, or any other folder,
    takes the video folder directly.
    """
    target = Path(target)
    if (target / DLC_CONFIG).is_file():
        return target / DLC_LABELED_DATA / video_stem
    return target / video_stem


def collected_data_path(video_folder: Path, scorer: str) -> Path:
    return Path(video_folder) / f"CollectedData_{scorer}.csv"


def find_collected_data(video_folder: Path) -> Path | None:
    """The one ``CollectedData_*.csv`` already in *video_folder*, if any."""
    found = sorted(Path(video_folder).glob("CollectedData_*.csv"))
    if len(found) > 1:
        raise ValueError(f"{video_folder} holds several CollectedData files: {', '.join(p.name for p in found)}")
    return found[0] if found else None


def dlc_scorer_of(path: Path) -> str:
    return Path(path).stem.removeprefix("CollectedData_")


def dlc_rows(
    frames: Sequence[TrainingFrame],
    keypoints: Sequence[str],
    individuals: Sequence[str],
    scorer: str,
    video_stem: str,
    digits: int,
    multi_animal: bool,
) -> pd.DataFrame:
    """The frames as a DeepLabCut labels table.

    Rows are DeepLabCut's three-level path index ``(labeled-data, video,
    image)``; columns are ``(scorer, bodyparts, coords)`` for a single-animal
    project and ``(scorer, individuals, bodyparts, coords)`` for a
    multi-animal one. A missing point is NaN, as DeepLabCut writes it.
    """
    if not multi_animal and len(individuals) != 1:
        raise ValueError(
            f"A single-animal DeepLabCut table holds one individual, the file has {len(individuals)} — "
            "export into a multi-animal project."
        )
    columns: list[tuple[str, ...]] = []
    for individual in individuals:
        for keypoint in keypoints:
            for coord in ("x", "y"):
                columns.append((scorer, individual, keypoint, coord) if multi_animal else (scorer, keypoint, coord))
    names = ["scorer", "individuals", "bodyparts", "coords"] if multi_animal else ["scorer", "bodyparts", "coords"]
    rows = [(DLC_LABELED_DATA, video_stem, image_name(f.video_frame, digits)) for f in frames]
    index = pd.MultiIndex.from_tuples(rows) if rows else pd.MultiIndex(levels=[[], [], []], codes=[[], [], []])
    values = np.empty((0, len(columns)))
    if frames:
        values = np.stack([f.positions.reshape(-1) for f in frames]).astype(np.float64)
    return pd.DataFrame(values, index=index, columns=pd.MultiIndex.from_tuples(columns, names=names))


def read_collected_data(path: Path) -> pd.DataFrame:
    """A DeepLabCut labels csv, either header depth, with its path index."""
    path = Path(path)
    header_rows = 4 if _is_multi_animal_csv(path) else 3
    df = pd.read_csv(path, header=list(range(header_rows)), index_col=[0, 1, 2])
    df.index = pd.MultiIndex.from_tuples([tuple(str(x) for x in row) for row in df.index])
    return df.astype(np.float64)


def _is_multi_animal_csv(path: Path) -> bool:
    with open(path, encoding="utf-8") as fh:
        lines = [fh.readline() for _ in range(4)]
    return lines[1].startswith("individuals")


def is_multi_animal_table(df: pd.DataFrame) -> bool:
    return df.columns.nlevels == 4


def merge_collected_data(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """*new* over *existing*: rows keyed by image replace, columns union.

    Two scorers or two header depths cannot share a table — DeepLabCut reads
    one scorer per file — so those are refused rather than merged.
    """
    if existing is None or existing.empty:
        return new.sort_index()
    if existing.columns.nlevels != new.columns.nlevels:
        raise ValueError("The existing CollectedData is a different project type (single vs multi-animal).")
    old_scorer, new_scorer = existing.columns[0][0], new.columns[0][0]
    if old_scorer != new_scorer:
        raise ValueError(f"The existing CollectedData is scored by {old_scorer!r}, this export by {new_scorer!r}.")
    kept = existing.drop(index=[row for row in new.index if row in existing.index])
    columns = list(existing.columns) + [c for c in new.columns if c not in existing.columns]
    merged = pd.concat([kept, new]).reindex(columns=pd.MultiIndex.from_tuples(columns, names=new.columns.names))
    return merged.sort_index()


def write_collected_data(df: pd.DataFrame, path: Path) -> list[Path]:
    """Write the csv and, when pandas can, the ``.h5`` twin; returns what was written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
# COCO: images/ + annotations.json
# ----------------------------------------------------------------------


def coco_image_name(video_stem: str, video_frame: int, digits: int) -> str:
    """One flat images folder holds several videos, so the stem is the prefix."""
    return f"{video_stem}_{image_name(video_frame, digits)}"


def read_coco(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    for key in ("images", "annotations", "categories"):
        payload.setdefault(key, [])
    return payload


def coco_category(existing: dict | None, keypoints: Sequence[str], name: str = "animal") -> dict:
    """The one category the export writes; an existing one must agree on keypoints."""
    if existing is not None and existing["categories"]:
        category = existing["categories"][0]
        if list(category.get("keypoints", [])) != list(keypoints):
            raise ValueError(
                f"{COCO_ANNOTATIONS} already describes keypoints {category.get('keypoints')}, "
                f"this file has {list(keypoints)}."
            )
        return category
    return {"id": 1, "name": name, "supercategory": name, "keypoints": list(keypoints), "skeleton": []}


def coco_annotation(positions: np.ndarray) -> dict | None:
    """One individual's ``(n_keypoints, 2)`` as a COCO annotation body, or ``None`` if unlabelled."""
    visible = np.isfinite(positions[:, 0])
    if not visible.any():
        return None
    flat: list[float] = []
    for (x, y), seen in zip(positions, visible, strict=True):
        flat.extend([float(x), float(y), COCO_VISIBLE] if seen else [0.0, 0.0, 0])
    xs, ys = positions[visible, 0], positions[visible, 1]
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return {
        "keypoints": flat,
        "num_keypoints": int(visible.sum()),
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "area": (x1 - x0) * (y1 - y0),
        "iscrowd": 0,
    }


def merge_coco(
    existing: dict | None,
    images: Sequence[dict],
    annotations_by_image: Sequence[Sequence[dict]],
    category: dict,
) -> dict:
    """*images* into *existing*: same ``file_name`` replaces, ids continue.

    Each entry of *annotations_by_image* holds the annotation bodies for the
    image at the same position; ids and ``image_id`` are assigned here so the
    caller never has to know what is already in the file.
    """
    payload = {"images": [], "annotations": [], "categories": [category]}
    if existing is not None:
        replaced = {img["file_name"] for img in images}
        dropped_ids = {img["id"] for img in existing["images"] if img["file_name"] in replaced}
        payload["images"] = [img for img in existing["images"] if img["file_name"] not in replaced]
        payload["annotations"] = [a for a in existing["annotations"] if a["image_id"] not in dropped_ids]
    next_image = max((img["id"] for img in payload["images"]), default=0) + 1
    next_ann = max((a["id"] for a in payload["annotations"]), default=0) + 1
    for image, bodies in zip(images, annotations_by_image, strict=True):
        image = {**image, "id": next_image}
        payload["images"].append(image)
        for body in bodies:
            payload["annotations"].append(
                {**body, "id": next_ann, "image_id": next_image, "category_id": category["id"]}
            )
            next_ann += 1
        next_image += 1
    return payload


def write_coco(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


# ----------------------------------------------------------------------
# The two exports, one video each
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExportOutcome:
    n_frames: int
    written: list[Path]


def export_dlc(
    frames: Sequence[TrainingFrame],
    store: KeypointStore,
    decode: Callable[[int], np.ndarray],
    target: Path,
    video_stem: str,
    scorer: str,
    n_video_frames: int,
    multi_animal: bool | None = None,
    progress: Progress | None = None,
) -> ExportOutcome:
    """Add *frames* of one video to a DeepLabCut labels folder under *target*.

    *decode* returns the RGB image of a **video** frame index. *multi_animal*
    defaults to what the folder already holds, else to the project's config,
    else to whether the file has several individuals.
    """
    folder = dlc_video_folder(target, video_stem)
    existing_csv = find_collected_data(folder)
    existing = read_collected_data(existing_csv) if existing_csv else None
    if existing_csv is not None and dlc_scorer_of(existing_csv) != scorer:
        raise ValueError(
            f"{folder} already holds {existing_csv.name}; "
            f"export with scorer {dlc_scorer_of(existing_csv)!r} to add to it."
        )
    if multi_animal is None:
        if existing is not None:
            multi_animal = is_multi_animal_table(existing)
        else:
            _, project_multi = dlc_project_settings(target)
            multi_animal = project_multi if project_multi is not None else store.n_individuals > 1
    existing_names = [row[2] for row in existing.index] if existing is not None else []
    digits = frame_digits(existing_names or [p.name for p in folder.glob("img*.png")], n_video_frames)

    written: list[Path] = []
    for i, frame in enumerate(frames):
        image = folder / image_name(frame.video_frame, digits)
        write_frame_image(decode(frame.video_frame), image)
        written.append(image)
        if progress is not None and not progress((i + 1) / max(1, len(frames))):
            raise InterruptedError("Export cancelled")
    table = dlc_rows(
        frames, store.keypoint_names, store.individual_names, scorer, video_stem, digits, bool(multi_animal)
    )
    merged = merge_collected_data(existing, table)
    written.extend(write_collected_data(merged, collected_data_path(folder, scorer)))
    return ExportOutcome(n_frames=len(frames), written=written)


def export_coco(
    frames: Sequence[TrainingFrame],
    store: KeypointStore,
    decode: Callable[[int], np.ndarray],
    target: Path,
    video_stem: str,
    n_video_frames: int,
    progress: Progress | None = None,
) -> ExportOutcome:
    """Add *frames* of one video to a COCO folder at *target*."""
    target = Path(target)
    annotations_path = target / COCO_ANNOTATIONS
    existing = read_coco(annotations_path) if annotations_path.is_file() else None
    category = coco_category(existing, store.keypoint_names)
    existing_names = [img["file_name"] for img in existing["images"]] if existing else []
    digits = frame_digits([n.split("_")[-1] for n in existing_names], n_video_frames)

    written: list[Path] = []
    images: list[dict] = []
    bodies: list[list[dict]] = []
    for i, frame in enumerate(frames):
        name = coco_image_name(video_stem, frame.video_frame, digits)
        rgb = decode(frame.video_frame)
        write_frame_image(rgb, target / COCO_IMAGES / name)
        written.append(target / COCO_IMAGES / name)
        images.append({"file_name": name, "width": int(rgb.shape[1]), "height": int(rgb.shape[0])})
        bodies.append([b for b in (coco_annotation(p) for p in frame.positions) if b is not None])
        if progress is not None and not progress((i + 1) / max(1, len(frames))):
            raise InterruptedError("Export cancelled")
    write_coco(merge_coco(existing, images, bodies, category), annotations_path)
    written.append(annotations_path)
    return ExportOutcome(n_frames=len(frames), written=written)
