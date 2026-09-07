"""The SAM side of box labelling: OCTRON's predictor driven headless, one per camera video.

Everything here is a thin call into the OCTRON fork (``octron.sam_octron``):
building the predictor from OCTRON's model catalogue, the resized frame cache
(``video data.zarr``), point prompts → mask (``run_new_pred``), propagation
(``propagate_in_video``) and the per-object mask stores. The interaction and
the on-disk layout are OCTRON's; what is ours is only that several sessions —
one per camera — are held side by side. Qt-free; torch is imported lazily so the
GUI starts without it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr

from ethograph.labels.octron_project import VIDEO_ZARR, ObjectEntry, OctronProject, VideoEntry

logger = logging.getLogger(__name__)

#: Frames per propagation chunk, OCTRON's own default for SAM2.
CHUNK_SIZE = 15
#: OCTRON's chunking of the resized frame cache.
VIDEO_ZARR_CHUNK = 15

Progress = Callable[[float], bool]


class TrackState:
    """Where one camera's SAM memory stands, and what Predict may do with it.

    SAM2's memory is one forward pass over the frames it has seen: the frames a
    user clicked on (conditioning) plus what it predicted after them. It is only
    meaningful for the stretch it was built on, so Predict is constrained to
    continue that stretch, and the memory is dropped as soon as the user moves
    somewhere else.

    * ``empty`` — nothing in memory; Predict is disabled.
    * ``seeded`` — clicks landed on *seed*; Predict runs from that frame.
    * ``predicted`` — frames *span* are in memory; Predict continues from the
      last one, but only while the playhead sits there. A click inside the span
      re-seeds the track at that frame (a correction), a click outside it starts
      afresh, and seeking farther than one chunk away from the span drops the
      memory altogether (``seek_resets``).
    """

    EMPTY = "empty"
    SEEDED = "seeded"
    PREDICTED = "predicted"

    def __init__(self) -> None:
        self.state = self.EMPTY
        self.seed: int | None = None
        self.span: tuple[int, int] | None = None

    def reset(self) -> None:
        self.state = self.EMPTY
        self.seed = None
        self.span = None

    def _bounds(self) -> tuple[int, int] | None:
        if self.state == self.SEEDED:
            assert self.seed is not None
            return self.seed, self.seed
        if self.state == self.PREDICTED:
            return self.span
        return None

    def click_resets(self, frame: int) -> bool:
        """A click outside what the memory covers means a new track: reset first."""
        bounds = self._bounds()
        return bounds is not None and not bounds[0] <= frame <= bounds[1]

    def note_click(self, frame: int) -> None:
        self.state = self.SEEDED
        self.seed = int(frame)
        self.span = None

    def predict_start(self, current: int) -> int | None:
        """The frame Predict runs from, or ``None`` when it may not run now."""
        if self.state == self.SEEDED:
            return self.seed
        if self.state == self.PREDICTED:
            assert self.span is not None
            return self.span[1] if current == self.span[1] else None
        return None

    def note_predicted(self, start: int, last: int) -> None:
        first = start if self.span is None else min(self.span[0], start)
        self.state = self.PREDICTED
        self.span = (int(first), int(max(last, start)))
        self.seed = None

    def seek_resets(self, frame: int, chunk: int) -> bool:
        """Farther than one *chunk* from what the memory covers, the memory is stale."""
        bounds = self._bounds()
        return bounds is not None and not bounds[0] - chunk <= frame <= bounds[1] + chunk


#: An individual that keeps less than this share of its mask after another one
#: claimed the overlap was the same animal — the remnant is dropped, not kept.
MIN_SURVIVING_FRACTION = 0.5


def exclusive_masks(masks: dict[int, np.ndarray], winner: int) -> dict[int, np.ndarray]:
    """Take the *winner*'s pixels away from every other object's mask.

    Two individuals cannot occupy the same pixels of one camera on one frame;
    the object just clicked is the newer evidence, so it keeps the pixels. A
    loser that keeps less than :data:`MIN_SURVIVING_FRACTION` of itself was the
    same animal, not a neighbour, and is emptied rather than left as a sliver.
    Returns the masks that changed, by object id (an empty mask = removed).
    """
    own = masks.get(winner)
    changed: dict[int, np.ndarray] = {}
    if own is None:
        return changed
    claimed = own > 0
    for obj_id, mask in masks.items():
        if obj_id == winner or mask is None:
            continue
        overlap = (mask > 0) & claimed
        if not overlap.any():
            continue
        trimmed = mask.copy()
        trimmed[overlap] = 0
        if trimmed.sum() < MIN_SURVIVING_FRACTION * (mask > 0).sum():
            trimmed[:] = 0
        changed[obj_id] = trimmed
    return changed


def sam_models() -> dict[str, dict]:
    """OCTRON's SAM2 catalogue, key → {name, config_path, checkpoint_path, tooltip}."""
    import yaml

    return yaml.safe_load(_sam2_models_yaml().read_text(encoding="utf-8"))


def _sam2_models_yaml() -> Path:
    import octron.sam_octron as pkg

    return Path(pkg.__file__).resolve().parent / "sam2_models.yaml"


def ensure_checkpoint(model_key: str, progress: Progress | None = None) -> tuple[Path, Path]:
    """``(config, checkpoint)`` for *model_key*, downloading the checkpoint if missing."""
    from octron.config import get_sam_checkpoints_dir
    from octron.sam_octron.helpers.sam2_checks import check_sam2_models

    models = sam_models()
    if model_key not in models:
        raise KeyError(f"Unknown SAM model {model_key!r}; OCTRON offers {sorted(models)}")
    entry = models[model_key]
    ckpt = get_sam_checkpoints_dir() / Path(entry["checkpoint_path"]).name
    if not ckpt.is_file():
        if progress is not None:
            progress(0.0)
        check_sam2_models("", _sam2_models_yaml(), force_download=False)
    if not ckpt.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found after download: {ckpt}")
    return Path(entry["config_path"]), ckpt


@dataclass
class PromptState:
    """The clicks placed for one object on one frame, in source pixels."""

    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)


class SamSession:
    """OCTRON's predictor bound to one camera video.

    Built lazily on the first click into a camera: SAM2-L holds a few GB, so
    four cameras at once is a deliberate choice the user makes by clicking
    into four views. ``release()`` frees it.
    """

    def __init__(self, project: OctronProject, entry: VideoEntry, model_key: str):
        self.project = project
        self.entry = entry
        self.model_key = model_key
        self.predictor = None
        self.device = None
        self._reader = None
        self._video_zarr: zarr.Array | None = None
        self._masks: dict[int, zarr.Array] = {}
        self._objects: dict[int, ObjectEntry] = {}
        #: (obj_id, frame) → the prompts placed there; OCTRON re-sends the whole
        #: frame's points on every click, so the list is the source of truth.
        self.prompts: dict[tuple[int, int], PromptState] = {}
        #: Frames written but not yet flushed into the zarr's ``annotated_frames``.
        self._pending: dict[int, set[int]] = {}

    # -- lifecycle ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self.predictor is not None

    def build(self, progress: Progress | None = None) -> None:
        """Load the model, open the video, create/open the frame cache, init state."""
        from napari_pyav._reader import FastVideoReader
        from octron.sam_octron.helpers.build_sam2_octron import build_sam2_octron
        from octron.sam_octron.helpers.sam_zarr import create_image_zarr, load_image_zarr

        config, ckpt = ensure_checkpoint(self.model_key, progress)
        predictor, device = build_sam2_octron(config_file_path=str(config), ckpt_path=str(ckpt))
        # SAM2 tracks every object on its own; this makes it keep only the
        # highest-scoring object at each pixel, so two individuals never share
        # pixels in one camera (OCTRON leaves it off).
        predictor.non_overlap_masks = True
        size = int(predictor.image_size)
        self._reader = FastVideoReader(str(self.entry.path), read_format="rgb24")

        zarr_path = self.entry.folder / VIDEO_ZARR
        array = None
        if zarr_path.exists():
            array, ok = load_image_zarr(
                zarr_path,
                num_frames=self.entry.num_frames,
                image_height=size,
                image_width=size,
                chunk_size=VIDEO_ZARR_CHUNK,
                num_ch=3,
                video_hash_abrrev=self.entry.hash8,
                verbose=False,
            )
            if not ok:
                import shutil

                shutil.rmtree(zarr_path, ignore_errors=True)
                array = None
        if array is None:
            array = create_image_zarr(
                zarr_path,
                num_frames=self.entry.num_frames,
                image_height=size,
                image_width=size,
                chunk_size=VIDEO_ZARR_CHUNK,
                num_ch=3,
                video_hash_abbrev=self.entry.hash8,
                verbose=False,
            )
        self._video_zarr = array
        predictor.init_state(video_data=self._reader, zarr_store=array)
        predictor.is_initialized = True
        self.predictor, self.device = predictor, device
        for obj in self.project.load_objects(self.entry):
            self._objects[obj.obj_id] = obj
            self._masks[obj.obj_id] = self.project.open_mask(self.entry, obj.label, obj.suffix)
        logger.info("SAM session ready for %s on %s", self.entry.path.name, device)

    def release(self) -> None:
        self.flush()
        if self.predictor is not None:
            try:
                self.predictor.reset_state()
            except Exception:  # noqa: BLE001 - best effort on teardown
                logger.debug("reset_state failed on release", exc_info=True)
        self.predictor = None
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:  # noqa: BLE001
                pass
        self._reader = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def reset(self) -> None:
        """OCTRON's Reset: forget the tracking memory, keep every stored mask."""
        if self.predictor is not None:
            self.predictor.reset_state()
            self.predictor.init_state(video_data=self._reader, zarr_store=self._video_zarr)
        self.prompts.clear()

    # -- objects ----------------------------------------------------------------

    def object_for(self, label: str, suffix: str, color: list[float] | None = None) -> ObjectEntry:
        obj = self.project.ensure_object(self.entry, label, suffix, color=color)
        if obj.obj_id not in self._masks:
            self._masks[obj.obj_id] = self.project.open_mask(self.entry, obj.label, obj.suffix)
        self._objects[obj.obj_id] = obj
        return obj

    @property
    def objects(self) -> dict[int, ObjectEntry]:
        return dict(self._objects)

    def mask(self, obj_id: int, frame: int) -> np.ndarray | None:
        """The stored mask ``(H, W)`` for *frame*, or ``None`` if never annotated there."""
        array = self._masks.get(obj_id)
        if array is None:
            return None
        data = np.asarray(array[frame])
        if data.max() < 0:
            return None
        return (data > 0).astype(np.uint8)

    def annotated_frames(self, obj_id: int) -> np.ndarray:
        from octron.sam_octron.helpers.sam_zarr import get_annotated_frames

        array = self._masks.get(obj_id)
        return get_annotated_frames(array) if array is not None else np.zeros(0, dtype=int)

    def clear_frame(self, obj_id: int, frame: int) -> None:
        """Forget this frame's mask and prompts for one object (OCTRON's reset on a frame)."""
        from octron.sam_octron.helpers.sam_zarr import unmark_frames_annotated

        array = self._masks.get(obj_id)
        if array is not None:
            array[frame] = -1
            unmark_frames_annotated(array, frame)
        self.prompts.pop((obj_id, frame), None)

    def remove_object(self, obj_id: int) -> None:
        """Drop the object here and on disk; SAM's memory of it goes with the next reset."""
        obj = self._objects.pop(obj_id, None)
        self._masks.pop(obj_id, None)
        self._pending.pop(obj_id, None)
        for key in [k for k in self.prompts if k[0] == obj_id]:
            self.prompts.pop(key)
        if obj is not None:
            self.project.remove_object(self.entry, obj.label, obj.suffix)
            if self.predictor is not None:
                try:
                    self.predictor.remove_object(obj_id)
                except Exception:  # noqa: BLE001 - an object SAM never saw raises; nothing to undo
                    logger.debug("predictor.remove_object(%s) failed", obj_id, exc_info=True)

    # -- prompting ---------------------------------------------------------------

    def add_point(self, obj_id: int, frame: int, x: float, y: float, positive: bool) -> np.ndarray | None:
        """Add one click and re-run SAM on every click of this (object, frame), as OCTRON does."""
        from octron.sam_octron.helpers.sam2_octron import run_new_pred
        from octron.sam_octron.helpers.sam_zarr import mark_frames_annotated

        if self.predictor is None:
            raise RuntimeError("SAM session not built")
        state = self.prompts.setdefault((obj_id, frame), PromptState())
        state.points.append((float(x), float(y)))
        state.labels.append(1 if positive else 0)
        mask = run_new_pred(
            predictor=self.predictor,
            frame_idx=int(frame),
            obj_id=int(obj_id),
            labels=list(state.labels),
            points=[list(p) for p in state.points],
        )
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=np.uint8)
        array = self._masks[obj_id]
        array[frame] = mask
        mark_frames_annotated(array, int(frame))
        # Labels are exclusive: the pixels just claimed leave every other
        # object's stored mask on this frame.
        others = {oid: self.mask(oid, frame) for oid in self._masks if oid != obj_id}
        for oid, trimmed in exclusive_masks({**others, obj_id: mask}, obj_id).items():
            self._store_trimmed(oid, frame, trimmed)
        return mask

    def has_prompts(self) -> bool:
        st = self.predictor.inference_state if self.predictor is not None else None
        return bool(st and (st.get("point_inputs_per_obj") or st.get("mask_inputs_per_obj")))

    def propagate(self, start: int, n_frames: int, skip: int = 0) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        """OCTRON's batch predict: yield ``(frame, {obj_id: mask})`` for the next *n_frames*.

        *skip* frames are left out between predictions (OCTRON's Skip spin).
        Masks are written to the stores as they arrive; ``flush()`` records
        the annotated frames in one go afterwards, as OCTRON's finish handler does.
        """
        if self.predictor is None or not self.has_prompts():
            return
        stride = max(1, int(skip) + 1)
        end = min(self.entry.num_frames - 1, int(start) + int(n_frames) * stride)
        order = list(range(int(start), end, stride))
        if not order:
            return
        try:
            self.predictor.images[order]  # prefetch the resized frames, as OCTRON does
        except Exception:  # noqa: BLE001 - prefetch is an optimisation only
            logger.debug("frame prefetch failed", exc_info=True)
        for frame_idx, obj_ids, logits in self.predictor.propagate_in_video(processing_order=order):
            masks = (logits > 0).cpu().numpy().astype(np.uint8)
            out: dict[int, np.ndarray] = {}
            for i, obj_id in enumerate(obj_ids):
                obj_id = int(obj_id)
                mask = masks[i].squeeze()
                array = self._masks.get(obj_id)
                if array is None:
                    continue
                array[frame_idx] = mask
                self._pending.setdefault(obj_id, set()).add(int(frame_idx))
                out[obj_id] = mask
            # SAM2's non-overlap constraint only arbitrates between the objects
            # it is tracking right now. An object that is not part of this
            # propagation (no prompts since the last Reset, or labelled in an
            # earlier session) may still hold a stored mask on this frame — take
            # the newly claimed pixels away from it, as a click would.
            stale = {oid: self.mask(oid, int(frame_idx)) for oid in self._masks if oid not in out}
            stale = {oid: m for oid, m in stale.items() if m is not None}
            if stale and out:
                claimed = np.zeros_like(next(iter(out.values())), dtype=np.uint8)
                for mask in out.values():
                    claimed |= (mask > 0).astype(np.uint8)
                for oid, trimmed in exclusive_masks({**stale, -1: claimed}, -1).items():
                    self._store_trimmed(oid, int(frame_idx), trimmed)
            yield int(frame_idx), out

    def _store_trimmed(self, obj_id: int, frame: int, trimmed: np.ndarray) -> None:
        """Write a trimmed mask; an emptied one is cleared so the frame stops counting as annotated."""
        if trimmed.any():
            self._masks[obj_id][frame] = trimmed
        else:
            self.clear_frame(obj_id, frame)

    def flush(self) -> None:
        """Record propagated frames as annotated, batched like OCTRON's finish handler."""
        from octron.sam_octron.helpers.sam_zarr import mark_frames_annotated

        for obj_id, frames in self._pending.items():
            array = self._masks.get(obj_id)
            if array is not None and frames:
                mark_frames_annotated(array, sorted(frames))
        self._pending.clear()
        self.project.save_organizer(self.entry, list(self._objects.values()), sam_model=self.model_key)
