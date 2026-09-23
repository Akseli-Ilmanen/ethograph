"""Draw a folder's labels onto its frames — DeepLabCut's ``check_labels``, in OpenCV.

After a ``labeled-data/<video>/`` folder has been reviewed, the one honest
check is to look at the frames with the labels burnt in. This writes every
image of the folder with its points (one colour per keypoint) and the
project's skeleton into ``labeled-data/<video>_labeled/`` — the name
DeepLabCut's own ``check_labels`` uses — so the output opens in any image
viewer and the tool's ``create_training_dataset`` never mistakes it for
training data. Reads the labels through :class:`~ethograph.labels.pose_project.LabelsTable`,
so it works for both layouts and either table format. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml
from matplotlib import colormaps

from ethograph.io.image_sequence import image_files, read_image
from ethograph.labels.pose_project import CONFIG_FILE, LABELED_SUFFIX, PoseProject

BGR = tuple[int, int, int]
SKELETON_COLOR: BGR = (180, 180, 180)
DEFAULT_COLORMAP = "rainbow"
DEFAULT_RADIUS = 5


def keypoint_colors(keypoints: Sequence[str], colormap: str = DEFAULT_COLORMAP) -> dict[str, BGR]:
    """One BGR colour per keypoint, spread over *colormap*."""
    cmap = colormaps[colormap].resampled(max(1, len(keypoints)))
    return {name: tuple(int(255 * c) for c in cmap(i)[2::-1]) for i, name in enumerate(keypoints)}  # type: ignore[misc]


def project_drawing_settings(project: PoseProject) -> tuple[list[list[str]], int, str]:
    """``(skeleton, dot radius, colormap)`` from a DeepLabCut config, else the defaults."""
    path = project.root / CONFIG_FILE
    if not path.is_file():
        return [], DEFAULT_RADIUS, DEFAULT_COLORMAP
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    skeleton = [[str(a), str(b)] for a, b in (cfg.get("skeleton") or [])]
    radius = max(1, int(cfg.get("dotsize", DEFAULT_RADIUS)))
    colormap = str(cfg.get("colormap") or DEFAULT_COLORMAP)
    if colormap not in colormaps:
        colormap = DEFAULT_COLORMAP
    return skeleton, radius, colormap


def draw_pose(
    image: np.ndarray,
    positions: np.ndarray,
    keypoints: Sequence[str],
    colors: dict[str, BGR],
    skeleton: Sequence[Sequence[str]],
    radius: int,
) -> None:
    """Burn ``(n_individuals, n_keypoints, 2)`` *positions* into *image* (BGR, in place)."""
    for individual in positions:
        placed = {
            name: (int(round(float(xy[0]))), int(round(float(xy[1]))))
            for name, xy in zip(keypoints, individual)
            if np.all(np.isfinite(xy))
        }
        for start, end in skeleton:
            if start in placed and end in placed:
                cv2.line(image, placed[start], placed[end], SKELETON_COLOR, 1, cv2.LINE_AA)
        for name, point in placed.items():
            cv2.circle(image, point, radius, colors[name], -1, cv2.LINE_AA)


def labeled_output_dir(folder: Path) -> Path:
    """``labeled-data/<video>_labeled/`` beside *folder*."""
    return folder.with_name(f"{folder.name}{LABELED_SUFFIX}")


def check_labels(project: PoseProject, video_stem: str) -> Path:
    """Write every frame of ``labeled-data/<video_stem>/`` with its labels drawn; returns the folder.

    A frame the table has no row for is written as it is, so the output
    always holds the whole folder and a missing pose is visible as such.
    """
    table = project.labels_table(video_stem)
    folder = table.folder
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} does not exist")
    keypoints, _individuals, _scorer = table.schema()
    frames = table.read()
    skeleton, radius, colormap = project_drawing_settings(project)
    colors = keypoint_colors(keypoints, colormap)
    output = labeled_output_dir(folder)
    output.mkdir(exist_ok=True)
    for path in image_files(folder):
        image = cv2.cvtColor(read_image(path), cv2.COLOR_RGB2BGR)
        positions = frames.get(path.name)
        if positions is not None:
            draw_pose(image, positions, keypoints, colors, skeleton, radius)
        ok, encoded = cv2.imencode(path.suffix, image)
        if not ok:
            raise OSError(f"Could not encode {path.name}")
        encoded.tofile(str(output / path.name))
    return output
