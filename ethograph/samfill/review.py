"""Look at the confidence and trajectory of the configured keypoints inside the table's segments.

Run before a fill to see where the tracker loses the keypoint, and again with
``filled_dir`` to see what SAM wrote over it. Qt-free; plots with matplotlib.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np
import pandas as pd
import xarray as xr
from movement.io import load_dataset

from ethograph.samfill.config import FillConfig, Segment, read_segments_tsv, segments_to_frames
from ethograph.samfill.fill import apply_exclusions, output_file
from ethograph.samfill.pairing import pair_folders

logger = logging.getLogger(__name__)

TRACE_COLUMNS = ("file", "individual", "keypoint", "stage", "frame", "time_s", "confidence", "x", "y")
STAGES = ("before", "after")

#: Two series only: the tracker's own estimate, and the fill over it.
STAGE_COLORS = {"before": "#8c8c8c", "after": "#1f77b4"}


def probe_video(path: str | Path) -> tuple[float, int]:
    """``(fps, n_frames)`` read from the container; raises when either is missing."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate is None or stream.frames == 0:
            raise ValueError(f"{path}: the container does not state its frame rate and frame count")
        return float(rate), int(stream.frames)


def load_poses(path: Path, source_software: str, fps: float) -> xr.Dataset:
    """A movement poses dataset from a tracker file or a movement NetCDF."""
    if path.suffix.lower() == ".nc":
        with xr.open_dataset(path) as opened:
            return opened.load()
    return load_dataset(path, source_software=source_software, fps=fps)


def _dim(ds: xr.Dataset, spellings: tuple[str, ...]) -> str:
    for dim in spellings:
        if dim in ds.dims:
            return dim
    raise KeyError(f"dataset has none of {spellings}; dims are {tuple(ds.dims)}")


def traces_from_dataset(ds: xr.Dataset, windows: list[range], cfg: FillConfig, file: str, stage: str) -> pd.DataFrame:
    """One row per frame inside *windows* for every configured (individual, keypoint)."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    kp_dim = _dim(ds, ("keypoints", "keypoint"))
    ind_dim = _dim(ds, ("individuals", "individual"))
    individuals = [str(i) for i in ds[ind_dim].values] if cfg.individuals is None else cfg.individuals
    frames = (
        np.concatenate([np.arange(w.start, min(w.stop, ds.sizes["time"])) for w in windows])
        if windows
        else np.zeros(0, int)
    )
    time_s = ds["time"].values[frames]
    parts: list[pd.DataFrame] = []
    for ind in individuals:
        for kp in cfg.keypoints:
            sel = {kp_dim: kp, ind_dim: ind}
            pos = ds["position"].sel(sel).isel(time=frames)
            parts.append(
                pd.DataFrame(
                    {
                        "file": file,
                        "individual": ind,
                        "keypoint": kp,
                        "stage": stage,
                        "frame": frames,
                        "time_s": time_s,
                        "confidence": ds["confidence"].sel(sel).isel(time=frames).values,
                        "x": pos.sel(space="x").values,
                        "y": pos.sel(space="y").values,
                    }
                )
            )
    if not parts:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    return pd.concat(parts, ignore_index=True)[list(TRACE_COLUMNS)]


def gather_traces(
    pose_dir: str | Path,
    video_dir: str | Path,
    segments_tsv: str | Path,
    cfg: FillConfig,
    filled_dir: str | Path | None = None,
) -> pd.DataFrame:
    """The traces of every paired file inside its segments; with *filled_dir*, the fill's too (``stage="after"``)."""
    pairs, _, _ = pair_folders(pose_dir, video_dir, cfg.pose_extensions, cfg.video_extensions)
    segments: list[Segment] = read_segments_tsv(segments_tsv)
    frames: list[pd.DataFrame] = []
    for pair in pairs:
        fps, n_frames = probe_video(pair.video)
        windows = segments_to_frames(segments, pair.video.name, fps, n_frames)
        if not windows:
            continue
        before, _ = apply_exclusions(load_poses(pair.pose, cfg.source_software, fps), pair.video.name, cfg)
        frames.append(traces_from_dataset(before, windows, cfg, pair.name, "before"))
        if filled_dir is not None:
            path = output_file(Path(filled_dir), pair.output_stem)
            if path is None:
                logger.info("%s: no fill output in %s", pair.name, filled_dir)
                continue
            after = load_poses(path, cfg.source_software, fps)
            frames.append(traces_from_dataset(after, windows, cfg, pair.name, "after"))
    if not frames:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def plot_traces(traces: pd.DataFrame, cfg: FillConfig, out_dir: str | Path | None = None) -> list[Path]:
    """One figure per file: a confidence row and an x/y row per (individual, keypoint), before over after.

    Saved as ``{file}_traces.png`` into *out_dir* when given, else shown. Gaps in
    the table's segments break the lines, so nothing is drawn across them.
    """
    import matplotlib.pyplot as plt

    written: list[Path] = []
    for file, per_file in traces.groupby("file", sort=True):
        tracks = sorted(set(zip(per_file["individual"], per_file["keypoint"])))
        fig, axes = plt.subplots(2 * len(tracks), 1, figsize=(11, 3.2 * len(tracks)), sharex=True, squeeze=False)
        for i, (ind, kp) in enumerate(tracks):
            ax_conf, ax_pos = axes[2 * i, 0], axes[2 * i + 1, 0]
            track = per_file[(per_file["individual"] == ind) & (per_file["keypoint"] == kp)]
            for stage in STAGES:
                rows = track[track["stage"] == stage].sort_values("frame")
                if rows.empty:
                    continue
                t, conf, x, y = (
                    _break_gaps(rows["frame"].to_numpy(), rows[c].to_numpy())
                    for c in ("time_s", "confidence", "x", "y")
                )
                color = STAGE_COLORS[stage]
                ax_conf.plot(t, conf, color=color, lw=1.2, label=stage)
                ax_pos.plot(t, x, color=color, lw=1.2, label=f"{stage} x")
                ax_pos.plot(t, y, color=color, lw=1.2, ls="--", label=f"{stage} y")
            ax_conf.axhline(cfg.low_confidence, color="#c44e52", lw=0.8, ls=":", label=f"low < {cfg.low_confidence:g}")
            ax_conf.axhline(
                cfg.seed_confidence, color="#55a868", lw=0.8, ls=":", label=f"seed ≥ {cfg.seed_confidence:g}"
            )
            ax_conf.set_ylim(-0.02, 1.02)
            ax_conf.set_ylabel("confidence")
            ax_conf.set_title(f"{ind} · {kp}", loc="left", fontsize=10)
            ax_pos.set_ylabel("pixels")
            ax_conf.legend(loc="lower left", fontsize=8, ncol=4, frameon=False)
            ax_pos.legend(loc="upper left", fontsize=8, ncol=4, frameon=False)
            for ax in (ax_conf, ax_pos):
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        axes[-1, 0].set_xlabel("time (s)")
        fig.suptitle(str(file), fontsize=11)
        fig.tight_layout()
        if out_dir is None:
            plt.show()
        else:
            path = Path(out_dir) / f"{file}_traces.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=130)
            plt.close(fig)
            written.append(path)
    return written


def _break_gaps(frames: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Insert NaN where consecutive frames are not adjacent so matplotlib lifts the pen."""
    values = values.astype(float)
    gaps = np.flatnonzero(np.diff(frames) > 1)
    return np.insert(values, gaps + 1, np.nan)


def inspect_folders(
    pose_dir: str | Path,
    video_dir: str | Path,
    segments_tsv: str | Path,
    cfg: FillConfig,
    filled_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Gather and plot in one call; returns the traces so they can be looked at as numbers too."""
    traces = gather_traces(pose_dir, video_dir, segments_tsv, cfg, filled_dir)
    plot_traces(traces, cfg, out_dir)
    return traces
