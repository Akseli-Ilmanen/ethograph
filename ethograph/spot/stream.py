"""Predict a trial straight from its video — no frame folder in between.

Training needs the JPEG folder: the trainer samples random clips from random
trials all epoch long, and a video codec cannot be read at random offsets
cheaply. Inference does not: it reads each trial once, front to back, in
overlapping windows — sequential decode, which is what a codec is good at. So
for inference the folder is overhead — decode, encode JPEG, write, read,
decode JPEG, tensor — when it could be decode → tensor.

This module is that path. It mirrors the vendored ``test_e2e.py`` exactly:
the same window starts (``ActionSpotVideoDataset``), the same padding, the
same evaluation transform (centre crop + ImageNet normalisation), the same
score accumulation over overlapping windows, and the same high-recall entry
the rest of the pipeline reads (:func:`~ethograph.spot.predict.spot_entry`).
Frames are kept in a rolling buffer of one window, so memory is bounded
whatever the trial's length.

One thing is deliberate:

* **The pixels are the training pixels.** The model learned on frames that
  went through the export's resize *and* a JPEG at ``JPEG_QUALITY``; the
  blur and ringing that adds are part of its input distribution. So each
  decoded frame is resized the same way and, by default, round-tripped
  through JPEG in memory (``infer.jpeg_roundtrip``) before it reaches the
  model — cheap, and the input is then what training saw. Off, it is an
  ablation with a number, not an assumption.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from ethograph.io.video_decode import RGBConverter, decode_frames
from ethograph.spot.dataset import JPEG_QUALITY, TrialRecord
from ethograph.spot.e2espot.train_e2e import E2EModel

logger = logging.getLogger(__name__)

#: The vendored trainer's own (``dataset/frame.py``, ``train_e2e.py``).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_LEN = 5
INFERENCE_BATCH_FRAMES = 400
RECALL_THRESHOLD = 0.01
#: Decode chains in flight at once. Measured on eight 200 fps trials (RTX 3080, 24 cores,
#: graph replay, warm): 2 threads 1,045 fps, 3 → 1,051, 4 → 1,110, 6 → 1,065, 8 → 1,007.
#: Flat from two on: the limit is the Python that feeds the card, and more threads only
#: crowd it.
STREAM_WORKERS_MAX = 4


def load_run_model(run_dir: Path, epoch: int, n_classes: int, device: str):
    """The vendored ``E2EModel`` of *run_dir* at *epoch*, built the way ``test_e2e.py`` builds it."""
    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = E2EModel(
        n_classes + 1,
        stored["feature_arch"],
        stored["temporal_arch"],
        clip_len=int(stored["clip_len"]),
        modality=stored["modality"],
        device=device,
        multi_gpu=False,
        shift_dilations=stored.get("shift_dilations"),
        attention_groups=stored.get("attention_groups", 2),
        distil_dim=stored.get("distil_dim"),
        fuse_dim=stored.get("fuse_dim"),
    )
    checkpoint = run_dir / f"checkpoint_{epoch:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{run_dir} has no checkpoint for epoch {epoch}")
    model.load(torch.load(checkpoint, map_location=device))
    return model, stored


#: ``(seq (B, L, C, H, W) float on the model's device, fuse (B, L, F) | None) → scores (B, L, K + 1)`` float32:
#: the model's softmax per frame, the way the vendored ``predict`` returns it.
Forward = Callable[[torch.Tensor, torch.Tensor | None], np.ndarray]


def eager_forward(model) -> Forward:
    """The vendored ``predict``, op by op — the CPU path, and what :class:`GraphedForward` is checked against."""
    # mixed precision only where it exists; on CPU the cuda autocast is a warning, not a speed-up
    use_amp = str(model.device).startswith("cuda")

    def run(seq: torch.Tensor, fuse: torch.Tensor | None) -> np.ndarray:
        _, scores = model.predict(seq, use_amp=use_amp, fuse=fuse)
        return np.asarray(scores, dtype=np.float32)

    return run


@dataclass
class _Captured:
    graph: torch.cuda.CUDAGraph
    x: torch.Tensor
    fuse: torch.Tensor | None
    out: torch.Tensor


class GraphedForward:
    """The model's forward on one window, captured once as a CUDA graph and replayed.

    Eagerly, a window is hundreds of small kernels — the gated shifts most of
    all — each launched from Python under the GIL. Decode threads share that
    GIL, so the launching thread waits at every kernel and the card idles
    between them: measured 650 fps at two decode threads and *falling* with
    each thread added, while shortening ``sys.setswitchinterval`` alone bought
    18 % — the tell. A replay is one launch per window. The numbers are the
    eager ones to the bit (max |Δ| = 0 on a held window), a window runs
    121 → 111 ms, and the thread count is a decode question again.

    One graph per input shape, captured on first sight; every call runs under
    the caller's GPU lock.
    """

    def __init__(self, model) -> None:
        self._net = model._model.eval()
        self._device = model.device
        self._graphs: dict[tuple, _Captured] = {}

    def _capture(self, x: torch.Tensor, fuse: torch.Tensor | None) -> _Captured:
        static_x, static_fuse = x.clone(), None if fuse is None else fuse.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        # warm-up on a side stream first, as torch.cuda.graph documents
        with torch.cuda.stream(side), torch.no_grad(), torch.autocast("cuda", cache_enabled=False):
            for _ in range(3):
                self._net(static_x, fuse=static_fuse)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad(), torch.autocast("cuda", cache_enabled=False):
            out = self._net(static_x, fuse=static_fuse)
        if isinstance(out, tuple):  # the vendored predict's own two normalisations
            out = out[0]
        if out.dim() > 3:
            out = out[-1]
        return _Captured(graph, static_x, static_fuse, out)

    def __call__(self, seq: torch.Tensor, fuse: torch.Tensor | None) -> np.ndarray:
        scores = []
        for i in range(seq.shape[0]):
            x = seq[i : i + 1]
            f = None if fuse is None else fuse[i : i + 1].to(self._device)
            key = (tuple(x.shape), None if f is None else tuple(f.shape))
            captured = self._graphs.get(key)
            if captured is None:
                captured = self._graphs[key] = self._capture(x, f)
            captured.x.copy_(x)
            if f is not None:
                assert captured.fuse is not None
                captured.fuse.copy_(f)
            captured.graph.replay()
            # softmax in the output's own dtype, as predict() does it outside autocast
            scores.append(torch.softmax(captured.out, 2)[0].float().cpu().numpy())
        return np.stack(scores)


def prepare_frame(frame: np.ndarray, record: TrialRecord, jpeg_roundtrip: bool = True) -> np.ndarray:
    """One decoded frame as the export would have written it: cropped, resized, and through JPEG."""
    if record.crop is not None:
        x0, y0, x1, y1 = record.crop
        frame = frame[y0:y1, x0:x1]
    image = Image.fromarray(frame).resize((record.width, record.height), Image.BILINEAR)
    if jpeg_roundtrip:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return np.array(image)


def normalise(frames: torch.Tensor, crop_dim: int | None, modality: str) -> torch.Tensor:
    """``(L, H, W, C)`` uint8 → ``(L, C, crop_dim, crop_dim)``: the vendored eval transform.

    ``/255``, centre crop, normalise. Runs where *frames* live: on the GPU the float conversion and
    normalisation of a 200-frame window are a CPU-side cost it does for free.
    """
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    if crop_dim:
        from torchvision.transforms.functional import center_crop

        x = center_crop(x, [crop_dim, crop_dim])
    if modality == "rgb":
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std
    if modality == "bw":
        gray = 0.2989 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return (gray - 0.5) / 0.5
    raise NotImplementedError(f"modality {modality!r}: only rgb and bw are streamed")


def window_starts(num_frames: int, clip_len: int, stride: int, overlap: int) -> list[int]:
    """The clip starts ``ActionSpotVideoDataset`` would enumerate for a video of *num_frames*."""
    step = (clip_len - overlap) * stride
    starts = list(range(-PAD_LEN * stride, max(0, num_frames - overlap * stride), step))
    if not starts:
        raise ValueError(f"a video of {num_frames} frames yields no clip at clip_len {clip_len}, stride {stride}")
    return starts


def _side_clip(block: np.ndarray | None, start: int, clip_len: int, stride: int) -> np.ndarray | None:
    """The feature block's rows for one window (zeros past either end), as the trainer's loader cuts them."""
    if block is None:
        return None
    out = np.zeros((clip_len, block.shape[1]), np.float32)
    first = start // stride
    lo, hi = max(first, 0), min(first + clip_len, len(block))
    if hi > lo:
        out[lo - first : hi - first] = block[lo:hi]
    return out


@dataclass
class _Window:
    """One clip waiting for the card: its frames inside the video, and how many pad frames sit either side."""

    start: int
    #: ``(n_inside, H, W, C)`` uint8, prepared; in pinned host memory when the model is on a card.
    frames: torch.Tensor
    pad_start: int
    pad_end: int
    fuse: np.ndarray | None


def predict_trial(
    model,
    stored: dict,
    record: TrialRecord,
    *,
    block: np.ndarray | None = None,
    jpeg_roundtrip: bool = True,
    batch_frames: int = INFERENCE_BATCH_FRAMES,
    gpu_lock: threading.Lock | None = None,
    decode_threads: int | None = None,
    forward: Forward | None = None,
) -> tuple[np.ndarray, int]:
    """``(scores (T', K + 1), num_frames)`` for one trial, streamed from its video.

    Windows overlap by half a clip, as ``test_e2e.py`` runs them; each frame's
    score is the mean over the windows that saw it. A window is run as soon as
    its last frame is decoded, from a buffer that never holds more than one
    window, so a long trial costs no more memory than a short one. *gpu_lock*
    serialises the model when several trials are decoded in parallel threads;
    *forward* is how the model is run (default :func:`eager_forward`).

    Every window starts on the stride grid, so only every *stride*-th frame
    can ever enter one: the others are decoded (a codec reads sequentially)
    and dropped before the colour conversion, resize and JPEG. A frame that
    is kept is prepared **once**, as uint8 in host memory; the two windows
    that overlap on it slice it from there.

    A decode thread owns nothing on the card. Its windows wait as uint8 on
    the host and go to the device — upload, normalise, forward — only inside
    *gpu_lock*, so the card holds one batch and the model's own peak whatever
    the thread count. Float windows parked on the device per thread (120 MB
    each at 200 × 224²) beside a 2.4 GB forward paged the card and, on a
    10 GB card with a GUI open, four decode threads ran at 74 fps.
    """
    clip_len, stride = int(stored["clip_len"]), int(stored.get("stride", 1))
    crop_dim, modality = stored.get("crop_dim"), stored["modality"]
    overlap = clip_len // 2
    step = (clip_len - overlap) * stride
    span = clip_len * stride
    batch_size = max(1, batch_frames // clip_len)
    n_classes = model._num_classes
    device = str(model.device)
    on_card = device.startswith("cuda")
    run = forward if forward is not None else eager_forward(model)

    scores: list[np.ndarray] = []  # per window, (clip_len, K + 1)
    starts: list[int] = []
    pending: list[_Window] = []

    def to_device(w: _Window) -> torch.Tensor:
        x = normalise(w.frames.to(device, non_blocking=True), crop_dim, modality)
        if w.pad_start or w.pad_end:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, w.pad_start, w.pad_end))
        return x

    def flush() -> None:
        if not pending:
            return
        fuse = None if block is None else np.stack([w.fuse for w in pending if w.fuse is not None])
        with gpu_lock if gpu_lock is not None else threading.Lock():
            seq = torch.stack([to_device(w) for w in pending])
            batch_scores = run(seq, None if fuse is None else torch.from_numpy(fuse))
        for w, s in zip(pending, batch_scores):
            starts.append(w.start)
            scores.append(s)
        pending.clear()

    def window(start: int, buffer: list[np.ndarray], grid_first: int, total: int | None) -> _Window:
        """The clip at *start* from *buffer* (first frame = strided index *grid_first*), padded past either end."""
        if start % stride:
            raise RuntimeError(f"{record.video_id}: window start {start} is off the stride grid ({stride})")
        frames = []
        n_pad_start = n_pad_end = 0
        for frame_num in range(start, start + span, stride):
            if frame_num < 0:
                n_pad_start += 1
            elif total is not None and frame_num >= total:
                n_pad_end += 1
            else:
                frames.append(buffer[frame_num // stride - grid_first])
        # torch stacks with the GIL released (numpy may not), and pinned memory makes the upload asynchronous
        stacked = torch.stack([torch.from_numpy(f) for f in frames])
        if on_card:
            stacked = stacked.pin_memory()
        return _Window(start, stacked, n_pad_start, n_pad_end, _side_clip(block, start, clip_len, stride))

    buffer: list[np.ndarray] = []  # the frames on the stride grid, uint8 (H, W, C), prepared
    grid_first = 0  # strided index of buffer[0]
    next_start = -PAD_LEN * stride
    decoded = 0
    to_rgb = RGBConverter()
    for raw in decode_frames(record.video_path, threads=decode_threads):
        if decoded % stride == 0:
            buffer.append(prepare_frame(to_rgb(raw), record, jpeg_roundtrip))
        decoded += 1
        # every window whose last frame is now decoded
        while next_start + span <= decoded:
            pending.append(window(next_start, buffer, grid_first, None))
            if len(pending) >= batch_size:
                flush()
            next_start += step
        grid_keep = max(0, next_start) // stride
        if grid_keep > grid_first:
            del buffer[: grid_keep - grid_first]
            grid_first = grid_keep
    total = decoded
    # the windows the folder path would still run at the tail, padded past the end
    while next_start < max(0, total - overlap * stride):
        pending.append(window(next_start, buffer, grid_first, total))
        next_start += step
    flush()
    if not starts:
        raise ValueError(f"{record.video_id}: {total} frames yield no clip at clip_len {clip_len}, stride {stride}")

    n_strided = total // stride
    acc = np.zeros((n_strided, n_classes), np.float32)
    support = np.zeros(n_strided, np.int32)
    for start, s in zip(starts, scores):
        idx = start // stride
        if idx < 0:
            s = s[-idx:]
            idx = 0
        end = min(n_strided, idx + len(s))
        acc[idx:end] += s[: end - idx]
        support[idx:end] += 1
    if (support == 0).any():
        raise RuntimeError(f"{record.video_id}: a strided frame was covered by no window — the window walk is wrong")
    return acc / support[:, None], total


def recall_entry(video_id: str, scores: np.ndarray, fps_strided: float, class_names: list[str]) -> dict:
    """One video in E2E-Spot's high-recall schema — every (frame, class) at or above the threshold."""
    events = []
    for i, name in enumerate(class_names):
        column = scores[:, i + 1]
        for frame in np.flatnonzero(column >= RECALL_THRESHOLD):
            events.append({"label": name, "frame": int(frame), "score": float(column[frame])})
    return {"video": video_id, "fps": fps_strided, "num_frames": len(scores), "events": events}


def predict_records(
    run_dir: Path,
    epoch: int,
    records: list[TrialRecord],
    class_names: list[str],
    *,
    blocks: dict[str, np.ndarray] | None = None,
    jpeg_roundtrip: bool = True,
    device: str | None = None,
    workers: int | None = None,
    loaded: tuple[object, dict] | None = None,
    cuda_graph: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """Every record's recall entry, streamed; plus each trial's decoded frame count.

    Trials are decoded in parallel threads — decode, resize and JPEG all
    release the GIL — feeding the one model through a lock: the GPU is far
    faster than one thread's decode chain, so several chains keep it busy.
    On a card the model runs as a replayed graph (:class:`GraphedForward`);
    *cuda_graph* off is the eager path, for checking one against the other.

    *loaded* is a ``(model, stored config)`` pair from :func:`load_run_model`,
    so a caller predicting many sessions loads the checkpoint once.
    """
    from ethograph.spot.dataset import default_workers
    from ethograph.utils.device import resolve_device

    if loaded is None:
        loaded = load_run_model(run_dir, epoch, len(class_names), device or resolve_device())
    model, stored = loaded
    stride = int(stored.get("stride", 1))
    for record in records:
        if stored.get("fuse_dim") and (blocks or {}).get(record.video_id) is None:
            raise ValueError(f"{record.video_id}: the run reads a feature block and none was given")
    gpu_lock = threading.Lock()
    n_workers = min(workers or default_workers(), STREAM_WORKERS_MAX, max(1, len(records)))
    on_card = str(model.device).startswith("cuda")
    forward = GraphedForward(model) if on_card and cuda_graph else eager_forward(model)

    def one(record: TrialRecord) -> tuple[dict, int]:
        block = (blocks or {}).get(record.video_id)
        scores, total = predict_trial(
            model,
            stored,
            record,
            block=block,
            jpeg_roundtrip=jpeg_roundtrip,
            gpu_lock=gpu_lock,
            decode_threads=None,  # the codec's own threading measured faster than one thread per container
            forward=forward,
        )
        logger.info("%s: %d frames streamed", record.video_id, total)
        return recall_entry(record.video_id, scores, record.fps / stride, class_names), total

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(one, records))
    entries = [entry for entry, _ in results]
    lengths = {r.video_id: total for r, (_, total) in zip(records, results)}
    return entries, lengths
