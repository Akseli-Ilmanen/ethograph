"""S3D features for one video, as a DataArray on its own time axis.

Two ways to run the network:

* **windows** — every frame's feature is the embedding of the ``stack``
  frames centred on it (zero frames beyond the video's ends). This is the
  sliding-stack scheme, done with a rolling buffer and batched forward
  passes instead of one pass per frame, and streaming decode instead of
  reading the file whole.
* **dense** — the convolutional trunk is translation-equivariant in time, so
  it runs once over the video (in overlapping chunks) and yields one
  position every ``stride`` frames, each centred where
  :data:`~ethograph.video_features.s3d.S3D_STAGES` says. Those are spatially
  averaged, interpolated to the frame grid and smoothed over the window
  length. Much cheaper, but every position sees the trunk's full receptive
  field (~99 frames), not the window — an ablation against *windows*.

The result carries ``time_s3d`` in seconds of the video's own clock at the
effective (possibly subsampled) rate; whoever builds a dataset interpolates
it onto the trial grid, and the alignment applies the stream offset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from ethograph.utils.device import resolve_device
from ethograph.video_features.base import TIME_DIM, CropBox, feature_dim, to_dataarray
from ethograph.video_features.frames import iter_frame_chunks, probe_video
from ethograph.video_features.plan import S3DConfig, S3DPlan, plan_s3d
from ethograph.video_features.s3d import FULL_STAGE, S3D, S3D_STAGES, S3DStage, truncated_base

#: Kinetics-400 weights shipped with the package.
CHECKPOINT = Path(__file__).resolve().parent / "checkpoint" / "S3D_kinetics400_torchified.pt"
#: Input side the network was trained at.
SIDE = 224
NAME = "s3d"
FEATURE_DIM = feature_dim(NAME)

__all__ = ["CHECKPOINT", "FEATURE_DIM", "NAME", "S3DExtractor", "TIME_DIM", "extract_s3d"]

Embed = Callable[[torch.Tensor], torch.Tensor]


def load_s3d(device: torch.device) -> S3D:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"S3D checkpoint not found at {CHECKPOINT}")
    model = S3D(num_class=400, ckpt_path=str(CHECKPOINT))
    model.eval()
    return model.to(device)


def preprocess(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """``(N, H, W, 3)`` uint8 → ``(N, 3, 224, 224)`` float32 in [0, 1].

    The shorter side is resized (bilinear) to 224 and the centre cropped —
    the Kinetics evaluation transform; the released weights expect no
    mean/std normalisation.
    """
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    h, w = x.shape[-2:]
    scale = SIDE / min(h, w)
    size = (max(SIDE, int(round(h * scale))), max(SIDE, int(round(w * scale))))
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    i = int(round((size[0] - SIDE) / 2.0))
    j = int(round((size[1] - SIDE) / 2.0))
    return x[..., i : i + SIDE, j : j + SIDE]


# ----------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------


def window_features(embed: Embed, chunks: Iterator[torch.Tensor], stack: int, batch: int) -> Iterator[torch.Tensor]:
    """Stream ``(N, 3, H, W)`` chunks in, yield ``(N, C)`` features out.

    Frame *t*'s feature is ``embed`` of the ``stack`` frames centred on it
    (``(B, 3, stack, H, W)`` in), zero frames standing in beyond the video's
    ends, so the output has exactly one row per input frame. A rolling
    carry of ``stack - 1`` frames joins consecutive chunks, and the windows
    are views (``unfold``) batched ``batch`` at a time.
    """
    if stack % 2 == 0 or stack < 1:
        raise ValueError(f"stack must be odd and positive, got {stack}")
    half = stack // 2
    carry: torch.Tensor | None = None

    def emit(buf: torch.Tensor) -> Iterator[torch.Tensor]:
        n = buf.shape[0] - stack + 1
        if n <= 0:
            return
        windows = buf.unfold(0, stack, 1).permute(0, 1, 4, 2, 3)  # (n, 3, stack, H, W)
        for i in range(0, n, max(1, batch)):
            yield embed(windows[i : i + batch].contiguous())

    for chunk in chunks:
        if carry is None:
            carry = chunk.new_zeros((half, *chunk.shape[1:]))
        buf = torch.cat([carry, chunk])
        yield from emit(buf)
        carry = buf[max(0, buf.shape[0] - stack + 1) :]
    if carry is None:
        return
    tail = torch.cat([carry, carry.new_zeros((half, *carry.shape[1:]))])
    yield from emit(tail)


# ----------------------------------------------------------------------
# Dense
# ----------------------------------------------------------------------


def dense_positions(
    trunk: Callable[[torch.Tensor], torch.Tensor],
    stage: S3DStage,
    chunks: Iterator[torch.Tensor],
    core: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the trunk over the video in cores of ``core`` frames, each padded
    with half a receptive field of context (zero frames beyond the ends).

    Returns the frame index each position is centred on and its spatially
    averaged feature, ``(P,)`` and ``(P, C)``, in time order.
    """
    context = stage.receptive_field // 2
    centres: list[int] = []
    feats: list[np.ndarray] = []

    def run(block: torch.Tensor, block_start: int, keep_from: int, keep_to: int) -> None:
        with torch.no_grad():
            out = trunk(block.permute(1, 0, 2, 3).unsqueeze(0))  # (1, C, L, h, w)
        out = out.float().mean(dim=(3, 4))[0].T.cpu().numpy()  # (L, C)
        for m in range(out.shape[0]):
            c = stage.stride * m + stage.offset
            if keep_from <= c < keep_to:
                centres.append(block_start + c)
                feats.append(out[m])

    buf: torch.Tensor | None = None
    start = -context  # frame index of buf[0]
    for chunk in chunks:
        if buf is None:
            buf = chunk.new_zeros((context, *chunk.shape[1:]))
        buf = torch.cat([buf, chunk])
        while buf.shape[0] >= 2 * context + core:
            block = buf[: 2 * context + core]
            run(block, start, context, context + core)
            buf = buf[core:]
            start += core
    if buf is None:
        return np.zeros(0, dtype=int), np.zeros((0, stage.channels), dtype=np.float32)
    tail = torch.cat([buf, buf.new_zeros((context, *buf.shape[1:]))])
    run(tail, start, context, tail.shape[0] - context)
    if not centres:
        raise ValueError("Video too short for dense mode — no trunk position fell inside it.")
    return np.asarray(centres), np.stack(feats)


def dense_to_frames(centres: np.ndarray, feats: np.ndarray, n_frames: int, stack: int) -> np.ndarray:
    """Interpolate stride-spaced positions onto every frame (flat beyond the
    first/last position) and smooth over ``stack`` frames."""
    grid = np.arange(n_frames)
    if len(centres) == 1:
        dense = np.repeat(feats, n_frames, axis=0)
    else:
        f = interp1d(centres, feats, axis=0, bounds_error=False, fill_value=(feats[0], feats[-1]))
        dense = f(grid)
    if stack > 1:
        dense = uniform_filter1d(dense, size=stack, axis=0, mode="nearest")
    return dense.astype(np.float32)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def _embedder(model: S3D, device: torch.device, precision: str) -> Embed:
    use_half = precision == "fp16" and device.type == "cuda"

    def embed(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_half):
            return model(x, features=True).float()

    return embed


def _trunk(model: S3D, stage: str, device: torch.device, precision: str) -> Callable[[torch.Tensor], torch.Tensor]:
    base = truncated_base(model, stage)
    use_half = precision == "fp16" and device.type == "cuda"

    def run(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_half):
            return base(x)

    return run


def extract_s3d(
    video_path: str | Path,
    cfg: S3DConfig = S3DConfig(),
    *,
    crop: CropBox | None = None,
    device: str | None = None,
    progress: Callable[[int], None] | None = None,
) -> xr.DataArray:
    """S3D features of *video_path* under *cfg* → ``(time_video, s3d_dims)``.

    The rate comes from the video; the plan (frames per window, step) is
    derived from it and recorded in ``attrs``. *crop* is cut from every
    decoded frame before the resize. *progress* is called with the number of
    frames consumed so far.
    """
    if cfg.mode not in ("windows", "dense"):
        raise ValueError(f"mode must be 'windows' or 'dense', got {cfg.mode!r}")
    if cfg.truncate_at is not None and cfg.mode != "dense":
        raise ValueError("truncate_at applies to dense mode only")
    stage_name = cfg.truncate_at or FULL_STAGE
    if stage_name not in S3D_STAGES:
        raise ValueError(f"Unknown S3D stage {stage_name!r}; choose from {sorted(S3D_STAGES)}")
    stage = S3D_STAGES[stage_name]

    info = probe_video(str(video_path))
    plan = plan_s3d(info.fps, cfg)
    if crop is not None:
        crop.validate(info.width, info.height, Path(video_path).name)
    dev = torch.device(resolve_device(device))
    model = load_s3d(dev)

    consumed = 0

    def chunks() -> Iterator[torch.Tensor]:
        nonlocal consumed
        for raw in iter_frame_chunks(video_path, step=plan.step, chunk=cfg.chunk):
            consumed += raw.shape[0]
            yield preprocess(raw if crop is None else crop.apply(raw), dev)
            if progress is not None:
                progress(consumed)

    if cfg.mode == "windows":
        windows = window_features(_embedder(model, dev, cfg.precision), chunks(), plan.stack_frames, cfg.batch)
        parts = [p.cpu() for p in windows]
        feats = torch.cat(parts).numpy() if parts else np.zeros((0, stage.channels), dtype=np.float32)
    else:
        centres, positions = dense_positions(_trunk(model, stage_name, dev, cfg.precision), stage, chunks(), cfg.chunk)
        feats = dense_to_frames(centres, positions, consumed, plan.stack_frames)
    if feats.shape[0] != consumed:
        raise RuntimeError(f"Decoded {consumed} frames but produced {feats.shape[0]} features")

    return _to_dataarray(feats, plan, info.path, cfg, stage_name, crop)


def _to_dataarray(
    feats: np.ndarray, plan: S3DPlan, path: str, cfg: S3DConfig, stage_name: str, crop: CropBox | None
) -> xr.DataArray:
    attrs: dict[str, object] = {
        "video_path": path,
        "stack_frames": plan.stack_frames,
        "stack_s": plan.stack_s,
        "mode": cfg.mode,
        "stage": stage_name,
        "precision": cfg.precision,
        "checkpoint": CHECKPOINT.name,
    }
    if crop is not None:
        attrs["crop"] = [crop.x0, crop.y0, crop.x1, crop.y1]
    return to_dataarray(feats, name=NAME, video_fps=plan.video_fps, step=plan.step, attrs=attrs)


@dataclass(frozen=True)
class S3DExtractor:
    """The ``s3d`` entry of the registry: :class:`S3DConfig` plus an optional crop."""

    config: S3DConfig = S3DConfig()
    crop: CropBox | None = None

    @property
    def name(self) -> str:
        return NAME

    def plan(self, video_fps: float) -> S3DPlan:
        return plan_s3d(video_fps, self.config)

    def extract(
        self,
        video: str | Path,
        *,
        device: str | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> xr.DataArray:
        return extract_s3d(video, self.config, crop=self.crop, device=device, progress=progress)
