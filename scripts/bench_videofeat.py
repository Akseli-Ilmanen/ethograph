"""Time S3D against DINOv2 (timm) on ten cam-1 clips — same frames, same crop.

Both extractors see the same input: every 4th frame (200 → 50 fps) of the
same crop. First timm call also downloads the weights; that is timed
separately as a warm-up so the numbers compare the extraction alone.
"""

import time
from pathlib import Path

import torch

import ethograph as eto
from ethograph.io.video_probe import probe_video
from ethograph.utils.device import resolve_device

SRC = Path(r"C:\Users\aksel\Documents\VidData\20250306_01_Ivy")
OUT = Path(r"C:\Users\aksel\Documents\Code\ethograph\data\features_bench")
N_VIDEOS = 10
ANALYSIS_FPS = 50
CROP = {"x0": 164, "y0": 0, "x1": 367, "y1": 164}  # data/spot/project.yaml, labels.crop

videos = sorted(SRC.glob("*cam-1*.mp4"))[:N_VIDEOS]
frames_seen = sum(probe_video(str(v)).nframes for v in videos) / (200 / ANALYSIS_FPS)
print(f"{len(videos)} videos, ~{frames_seen:.0f} frames at {ANALYSIS_FPS} fps, device {resolve_device()}")

# warm-up: weights download + CUDA init, on one short clip into a throwaway folder
for extractor in ("s3d", "timm"):
    eto.segment.extract_videos(
        [videos[0]], OUT / "_warmup", extractor=extractor, analysis_fps=ANALYSIS_FPS, crop=CROP, overwrite=True
    )

results = {}
for extractor in ("s3d", "timm"):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    eto.segment.extract_videos(
        videos, OUT / extractor, extractor=extractor, analysis_fps=ANALYSIS_FPS, crop=CROP, overwrite=True
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    results[extractor] = time.perf_counter() - t0

print()
print(f"{'extractor':<10}{'seconds':>10}{'frames/s':>12}{'per video':>12}")
for name, seconds in results.items():
    print(f"{name:<10}{seconds:>10.1f}{frames_seen / seconds:>12.0f}{seconds / len(videos):>12.1f}")
