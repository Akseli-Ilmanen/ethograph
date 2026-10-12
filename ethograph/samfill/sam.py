"""SAM2 as a point tracker: seed one click, propagate, read back mask centroids.

Built on OCTRON's predictor (``octron.sam_octron``), the same one the box
labelling dialog drives, because it streams frames lazily instead of loading
the whole video. Its zarr frame cache is replaced by :class:`FrameStore`, an
in-memory store that forgets frames the tracker has moved past, so a run's
memory stays flat and nothing is written to disk. Hiera-Tiny is the default;
every SAM 2.1 size OCTRON ships a config for is offered. OCTRON picks the
torch device itself (CUDA, then MPS, then CPU).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ethograph.samfill.samurai import SamuraiParams

logger = logging.getLogger(__name__)

SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

#: Frames handed to SAM per propagation call; OCTRON's own batch size.
CHUNK = 15

#: Resized frames kept behind the tracker; SAM2 attends to far fewer than this.
KEEP_BEHIND = 32

#: SAM's own object id; one point is tracked at a time.
OBJ_ID = 1


@dataclass(frozen=True)
class SamModel:
    name: str
    config: str
    checkpoint: str

    @property
    def url(self) -> str:
        return f"{SAM2_BASE_URL}/{self.checkpoint}"


SAM_MODELS: dict[str, SamModel] = {
    "sam2_tiny": SamModel("SAM 2.1 Hiera-Tiny", "sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "sam2_small": SamModel("SAM 2.1 Hiera-Small", "sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "sam2_base_plus": SamModel("SAM 2.1 Hiera-Base+", "sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "sam2_large": SamModel("SAM 2.1 Hiera-Large", "sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def ensure_checkpoint(model_key: str) -> tuple[str, Path]:
    """``(config name, checkpoint path)`` for *model_key*, downloading the checkpoint once."""
    import requests
    from octron.config import get_sam_checkpoints_dir

    if model_key not in SAM_MODELS:
        raise KeyError(f"Unknown SAM model {model_key!r}; choose from {sorted(SAM_MODELS)}")
    model = SAM_MODELS[model_key]
    ckpt = get_sam_checkpoints_dir() / model.checkpoint
    if not ckpt.is_file():
        logger.info("Downloading %s from %s", model.name, model.url)
        part = ckpt.with_suffix(ckpt.suffix + ".part")
        with requests.get(model.url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with part.open("wb") as fh:
                for chunk in response.iter_content(1 << 20):
                    fh.write(chunk)
        part.replace(ckpt)
    return model.config, ckpt


class FrameStore:
    """The array OCTRON's ``OctoZarr`` writes resized frames into, kept in memory.

    Shaped ``(n_frames, 3, size, size)`` like the zarr it stands in for, but
    holding only the frames written since the last :meth:`evict_before`. A
    read of a forgotten frame raises, because SAM2 never re-reads a frame it
    has tracked past; if it did, silently re-decoding would hide a bug.
    """

    def __init__(self, n_frames: int, size: int):
        self.shape = (n_frames, 3, size, size)
        self._frames: dict[int, np.ndarray] = {}

    @staticmethod
    def _indices(key: object) -> list[int]:
        if isinstance(key, tuple):
            key = key[0]
        if isinstance(key, (int, np.integer)):
            return [int(key)]
        if isinstance(key, slice):
            return list(range(*key.indices(10**12)))
        return [int(k) for k in np.asarray(key).ravel()]

    def __getitem__(self, key: object) -> np.ndarray:
        indices = self._indices(key)
        try:
            frames = [self._frames[i] for i in indices]
        except KeyError as exc:
            raise KeyError(f"frame {exc.args[0]} was evicted from the FrameStore") from None
        if isinstance(key, (int, np.integer)):
            return frames[0]
        return np.stack(frames)

    def __setitem__(self, key: object, value: np.ndarray) -> None:
        indices = self._indices(key)
        value = np.asarray(value, dtype=np.float16)
        if value.ndim == 3:
            value = value[None]
        if len(indices) != len(value):
            raise ValueError(f"{len(indices)} indices for {len(value)} frames")
        for i, frame in zip(indices, value):
            self._frames[i] = frame

    def evict_before(self, frame: int) -> None:
        for i in [i for i in self._frames if i < frame]:
            del self._frames[i]

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def __repr__(self) -> str:
        return f"FrameStore(shape={self.shape}, held={len(self._frames)})"


@dataclass(frozen=True)
class SeedShape:
    """What the seed mask looks like, for deciding whether SAM was shown the right thing."""

    area: int
    #: ``4πA / P²``: 1 for a disc, towards 0 for a rod or a sliver.
    circularity: float


class BadSeedError(ValueError):
    """The seed mask is not the compact blob a keypoint should segment to."""

    def __init__(self, frame: int, shape: SeedShape | None, reason: str):
        super().__init__(f"seed frame {frame}: {reason}")
        self.frame, self.shape = frame, shape


def seed_shape(mask: np.ndarray) -> SeedShape | None:
    """Area and circularity of a binary mask, or ``None`` when it is empty.

    The perimeter is the count of mask pixels with a 4-neighbour outside the
    mask, so circularity can slightly exceed 1 for small discs; the test is
    against a threshold well below that.
    """
    from scipy.ndimage import binary_erosion

    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return None
    perimeter = int((mask & ~binary_erosion(mask, border_value=0)).sum())
    return SeedShape(area, float(4 * np.pi * area / max(perimeter, 1) ** 2))


def mask_centroid(mask: np.ndarray) -> tuple[float, float, int] | None:
    """``(x, y, area)`` of a binary mask, or ``None`` when it is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()), int(len(xs))


@dataclass(frozen=True)
class TrackedPoint:
    frame: int
    x: float
    y: float
    #: Mask area relative to the seed's, folded to ``[0, 1]``: 1 means the same size.
    confidence: float


def build_predictor(model_key: str, samurai: SamuraiParams | None) -> tuple[Any, Any]:
    """``(predictor, device)`` for *model_key*: OCTRON's SAM 2, or SAMURAI on top of it.

    Build it once and hand it to every :class:`SamPointTracker` of a batch: a
    model holds a few GB of VRAM and loading it per video is what a batch
    spends most of its time on.
    """
    from octron.sam_octron.helpers.build_sam2_octron import build_sam2_octron

    config, ckpt = ensure_checkpoint(model_key)
    if samurai is not None:
        from ethograph.samfill.samurai import build_samurai

        return build_samurai(config, ckpt, samurai)
    return build_sam2_octron(config_file_path=config, ckpt_path=str(ckpt))


class SamPointTracker:
    """One video, one SAM2 model; :meth:`track` runs one seed through a frame range.

    With *samurai* set, SAMURAI's motion-aware selection and memory
    (:mod:`ethograph.samfill.samurai`) run on top of the same checkpoint;
    ``None`` is plain SAM 2.
    """

    def __init__(
        self,
        video: str | Path,
        model_key: str = "sam2_tiny",
        samurai: SamuraiParams | None = None,
        predictor: Any = None,
    ):
        self.video = Path(video)
        self.model_key = model_key
        self.samurai = samurai
        #: A model built by :func:`build_predictor`, shared across videos; built here when ``None``.
        self.predictor: Any = predictor
        self._shared = predictor is not None
        self.device: Any = None
        self._reader = None
        self._frames: FrameStore | None = None
        self.fps: float | None = None
        self.n_frames: int | None = None

    def __enter__(self) -> SamPointTracker:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        from napari_pyav._reader import FastVideoReader
        from octron.sam_octron.helpers.video_loader import probe_video

        if not self.video.is_file():
            raise FileNotFoundError(self.video)
        info = probe_video(str(self.video), verbose=False)
        self.fps = float(info["fps"])
        self.n_frames = int(info["num_frames"])
        if self._shared:
            predictor = self.predictor
            if getattr(predictor, "is_initialized", False):
                predictor.reset_state()  # forget the previous video before init_state binds this one
            device = next(predictor.parameters()).device
        else:
            predictor, device = build_predictor(self.model_key, self.samurai)
        self._reader = FastVideoReader(str(self.video), read_format="rgb24")
        self._frames = FrameStore(self.n_frames, int(predictor.image_size))
        predictor.init_state(video_data=self._reader, zarr_store=self._frames)
        predictor.is_initialized = True
        self.predictor, self.device = predictor, device
        flavour = "SAMURAI" if self.samurai is not None else "SAM 2"
        logger.info("%s on %s ready for %s on %s", flavour, SAM_MODELS[self.model_key].name, self.video.name, device)

    def close(self) -> None:
        """Release the video and this run's state; a shared model stays loaded for the next video."""
        if self.predictor is not None:
            if getattr(self.predictor, "is_initialized", False):
                self.predictor.reset_state()
            if not self._shared:
                self.predictor = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._frames is not None:
            self._frames.clear()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _fresh_state(self) -> None:
        """Forget every prompt, memory and cached frame, as OCTRON's Reset does."""
        if self.predictor is None or self._frames is None:
            raise RuntimeError("tracker not open")
        self._frames.clear()
        self.predictor.reset_state()
        if self.samurai is not None:
            self.predictor.reset_motion()
        self.predictor.init_state(video_data=self._reader, zarr_store=self._frames)

    def track(
        self,
        seed_frame: int,
        x: float,
        y: float,
        frames: range,
        box_px: int | None = None,
        min_circularity: float = 0.0,
    ) -> Iterator[TrackedPoint]:
        """Seed SAM at ``(x, y)`` on *seed_frame*, yield the centroid on each of *frames*.

        The seed is a positive click, inside a ``box_px``-wide box when given:
        the click alone makes SAM pick the whole object the point sits on, the
        box tells it which part. The seed mask must be a blob at least
        ``min_circularity`` round (1 is a disc) and, with a box, no larger than
        the box; otherwise :class:`BadSeedError` is raised before any frame is
        tracked, so the caller can try another seed. *frames* must lie after
        the seed and inside the video. A frame whose mask comes back empty is
        skipped, so the caller sees a gap rather than a made-up point. The
        confidence is the mask area relative to the seed's, folded into
        ``[0, 1]``; a mask that balloons or collapses scores low.
        """
        if self.n_frames is None or self.predictor is None or self._frames is None:
            raise RuntimeError("tracker not open")
        if not 0 <= seed_frame < self.n_frames:
            raise ValueError(f"seed frame {seed_frame} outside video of {self.n_frames} frames")
        if frames and (frames.start <= seed_frame or frames.stop > self.n_frames or frames.step != 1):
            raise ValueError(f"frames {frames} must follow seed {seed_frame} contiguously inside the video")
        self._fresh_state()
        half = None if box_px is None else box_px / 2
        _, obj_ids, logits = self.predictor.add_new_points_or_box(
            frame_idx=int(seed_frame),
            obj_id=OBJ_ID,
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            box=None if half is None else [x - half, y - half, x + half, y + half],
            clear_old_points=True,
            normalize_coords=True,
        )
        seed_mask = (logits[list(obj_ids).index(OBJ_ID)] > 0).cpu().numpy().squeeze()
        shape = seed_shape(seed_mask)
        if shape is None:
            raise BadSeedError(seed_frame, None, f"empty mask at ({x:.0f}, {y:.0f})")
        if shape.circularity < min_circularity:
            raise BadSeedError(seed_frame, shape, f"circularity {shape.circularity:.2f} < {min_circularity}")
        if box_px is not None and shape.area > box_px * box_px:
            raise BadSeedError(seed_frame, shape, f"mask of {shape.area} px overflows the {box_px} px box")
        seed_area = shape.area
        order = list(frames)
        for chunk_start in range(0, len(order), CHUNK):
            chunk = order[chunk_start : chunk_start + CHUNK]
            self.predictor.images[chunk]  # decode and resize the chunk in one go, as OCTRON does
            for frame_idx, obj_ids, logits in self.predictor.propagate_in_video(processing_order=chunk):
                masks = (logits > 0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids):
                    if int(obj_id) != OBJ_ID:
                        continue
                    found = mask_centroid(masks[i].squeeze())
                    if found is None:
                        continue
                    cx, cy, area = found
                    ratio = min(area, seed_area) / max(area, seed_area)
                    yield TrackedPoint(int(frame_idx), cx, cy, float(ratio))
            self._frames.evict_before(chunk[0] - KEEP_BEHIND)
