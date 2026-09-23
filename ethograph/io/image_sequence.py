"""A folder of still images treated as a video.

DeepLabCut and LightningPose keep their training frames as loose PNGs under
``labeled-data/<video>/``; refining those labels means navigating the folder
frame by frame, exactly like a video. Everywhere the GUI accepts a video path
it also accepts such a folder: the probe reports one frame per image and the
image-sequence clock, and :class:`ImageSequence` decodes on demand.

The images carry no rate, so the sequence declares its own clock:
:data:`IMAGE_SEQUENCE_RATE` is one image per second, which makes the time axis
read as the image index. It is a definition of the sequence's clock, not a
guess at a recording rate — nothing here pretends the frames were filmed at
this rate, and every frame ↔ time conversion still goes through the alignment.
"""

from __future__ import annotations

from pathlib import Path

import natsort
import numpy as np

from ethograph.io.validation import IMAGE_EXTENSIONS

#: Images per second on the image-sequence clock: time == image index.
IMAGE_SEQUENCE_RATE = 1.0


def image_files(folder: str | Path) -> list[Path]:
    """The images of *folder* in natural order (``img0002`` before ``img0010``)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return natsort.natsorted(files, key=lambda p: p.name)


def is_image_folder(path: str | Path | None) -> bool:
    """A directory holding at least one image — the shape of ``labeled-data/<video>/``."""
    if not path:
        return False
    folder = Path(path)
    return folder.is_dir() and bool(image_files(folder))


def media_exists(path: str | Path | None) -> bool:
    """A media path is on disk: a file, or an image folder standing in for a video."""
    if not path:
        return False
    return Path(path).is_file() or is_image_folder(path)


def read_image(path: str | Path) -> np.ndarray:
    """One image as ``(H, W, 3)`` uint8 RGB — greyscale and alpha are normalised away."""
    import imageio.v3 as iio

    data = np.asarray(iio.imread(path))
    if data.ndim == 2:
        data = np.repeat(data[:, :, None], 3, axis=2)
    if data.shape[2] > 3:
        data = data[:, :, :3]
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(data)


class ImageSequence:
    """Lazily read RGB frames of an image folder, indexable like a decoded video.

    Frames are read from disk on every access — a labelled folder holds tens
    of frames, and holding them all as float textures is what this avoids.
    Every frame is returned at the size of the first image, so a stray odd-sized
    frame cannot break a texture of fixed shape.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.files = image_files(self.folder)
        if not self.files:
            raise ValueError(f"{self.folder} holds no images")
        first = read_image(self.files[0])
        self.height, self.width = int(first.shape[0]), int(first.shape[1])

    def __len__(self) -> int:
        return len(self.files)

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` every frame is returned at."""
        return self.width, self.height

    def __getitem__(self, key):
        if isinstance(key, slice):
            return np.stack([self[i] for i in range(*key.indices(len(self)))])
        index = int(key)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"frame {key} outside 0..{len(self) - 1}")
        image = read_image(self.files[index])
        if image.shape[:2] != (self.height, self.width):
            import cv2

            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return image

    def name_of(self, index: int) -> str:
        """The file name behind frame *index* — what a labels table keys its rows by."""
        return self.files[int(index)].name

    def index_of(self, name: str) -> int | None:
        """The frame index of image *name*, or ``None`` when the folder has no such image."""
        for i, path in enumerate(self.files):
            if path.name == name:
                return i
        return None
