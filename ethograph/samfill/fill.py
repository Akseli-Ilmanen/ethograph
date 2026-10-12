"""Run the fill over paired folders and write the results back as pose files."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from movement.io import load_dataset
from movement.io.save_poses import to_dlc_file, to_lp_file, to_sleap_analysis_file

from ethograph.samfill.config import FillConfig, Segment, read_segments_tsv, segments_to_frames
from ethograph.samfill.pairing import Pair, pair_folders
from ethograph.samfill.plan import FillRun, excluded_frames, plan_anchor_run, plan_runs
from ethograph.samfill.sam import BadSeedError, SamPointTracker, build_predictor

logger = logging.getLogger(__name__)

#: movement's own flag beside ``position``/``confidence``: True where a point was changed or removed.
#: Here that is every frame SAM wrote and every detection the pixel exclusion dropped, so the napari
#: edit timeline shows exactly what the fill touched. Absent when nothing was changed, as movement does.
EDITED = "edited"
REPORT_NAME = "samfill_report.tsv"
#: movement's suffix for a file holding edits of another, so its tools treat the output as one.
OUTPUT_SUFFIX = "_edited"


@dataclass(frozen=True)
class FillReport:
    """One SAM run as it went: planned frames versus the ones a mask came back for."""

    video: str
    individual: str
    keypoint: str
    seed: int
    first: int
    last: int
    n_low: int
    n_filled: int
    mean_confidence: float
    #: Confident frames whose seed mask was rejected before this seed was accepted.
    seeds_rejected: int = 0


def _name_of(ds: xr.Dataset, spellings: tuple[str, ...]) -> str:
    for dim in spellings:
        if dim in ds.dims:
            return dim
    raise KeyError(f"dataset has none of {spellings}; dims are {tuple(ds.dims)}")


def _check_names(wanted: list[str], available: list[str], what: str) -> None:
    missing = [w for w in wanted if w not in available]
    if missing:
        raise KeyError(f"{what} {missing} not in the pose file; it has {available}")


def apply_exclusions(ds: xr.Dataset, video_name: str, cfg: FillConfig) -> tuple[xr.Dataset, int]:
    """A copy of *ds* with the configured keypoints set to NaN where the video's exclusion says they cannot be.

    Returns the copy and the number of (frame, keypoint, individual) values dropped.
    """
    exclusion = cfg.exclusion_for(video_name)
    out = ds.copy(deep=True)
    if exclusion is None:
        return out, 0
    kp_dim = _name_of(ds, ("keypoints", "keypoint"))
    ind_dim = _name_of(ds, ("individuals", "individual"))
    individuals = [str(i) for i in ds[ind_dim].values] if cfg.individuals is None else cfg.individuals
    dropped = 0
    for ind in individuals:
        for kp in cfg.keypoints:
            sel = {kp_dim: kp, ind_dim: ind}
            pos = out["position"].sel(sel)
            bad = excluded_frames(pos.sel(space="x").values, pos.sel(space="y").values, exclusion)
            if not bad.any():
                continue
            times = out["time"].values[bad]
            out["position"].loc[{"time": times, **sel}] = np.nan
            out["confidence"].loc[{"time": times, **sel}] = np.nan
            dropped += int(bad.sum())
    if dropped:
        logger.info("%s: %d keypoint values inside the pixel exclusion dropped", video_name, dropped)
    return out, dropped


def fill_dataset(
    ds: xr.Dataset,
    tracker: SamPointTracker,
    windows: list[range],
    cfg: FillConfig,
) -> tuple[xr.Dataset, list[FillReport]]:
    """Fill *ds* (a movement poses dataset) in place of its low frames, inside *windows*.

    Returns a copy with ``position``/``confidence`` overwritten on filled frames
    and movement's boolean ``edited`` variable marking every changed or dropped
    point, plus one report row per SAM run. The video's pixel exclusion is
    applied first, so an excluded detection is neither kept nor used as a seed.
    """
    kp_dim = _name_of(ds, ("keypoints", "keypoint"))
    ind_dim = _name_of(ds, ("individuals", "individual"))
    keypoints = [str(k) for k in ds[kp_dim].values]
    individuals = [str(i) for i in ds[ind_dim].values]
    _check_names(cfg.keypoints, keypoints, "keypoints")
    chosen_inds = individuals if cfg.individuals is None else cfg.individuals
    _check_names(chosen_inds, individuals, "individuals")

    out, _ = apply_exclusions(ds, tracker.video.name, cfg)
    confidence = out["confidence"]
    # A dropped detection is an edit (movement counts a removed point as edited).
    flags = (confidence.isnull() & ds["confidence"].notnull()).astype(bool)
    n_pose = int(out.sizes["time"])
    if tracker.n_frames is None:
        raise RuntimeError("tracker not open")
    if tracker.n_frames != n_pose:
        logger.warning(
            "%s: pose file has %d frames, video %d; using the shorter", tracker.video.name, n_pose, tracker.n_frames
        )
    limit = min(n_pose, tracker.n_frames)

    reports: list[FillReport] = []
    for ind in chosen_inds:
        for kp in cfg.keypoints:
            sel = {kp_dim: kp, ind_dim: ind}
            conf_trace = confidence.sel(sel).values.astype(float)
            for window in windows:
                window = range(window.start, min(window.stop, limit))
                if cfg.seeding == "anchor":
                    runs = plan_anchor_run(
                        conf_trace, window, cfg.low_confidence, cfg.seed_confidence, cfg.seed_tries, cfg.margin
                    )
                else:
                    runs = plan_runs(
                        conf_trace, window, cfg.low_confidence, cfg.seed_confidence, cfg.margin, cfg.seed_tries
                    )
                for run in runs:
                    reports.append(_apply_run(out, flags, tracker, run, sel, ind, kp, window, cfg))
    if bool(flags.any()):
        out[EDITED] = flags
    out.attrs["samfill_model"] = cfg.model
    return out, reports


def _apply_run(
    out: xr.Dataset,
    flags: xr.DataArray,
    tracker: SamPointTracker,
    run: FillRun,
    sel: dict[str, str],
    individual: str,
    keypoint: str,
    window: range,
    cfg: FillConfig,
) -> FillReport:
    """Track one run from its seed, or the first alternate whose mask passes, and write the fills."""
    name = f"{tracker.video.name} {individual}/{keypoint}"
    end = run.frames.stop
    rejected = 0
    for seed in (run.seed, *run.alternates):
        seed_xy = out["position"].isel(time=seed).sel(sel)
        x, y = float(seed_xy.sel(space="x")), float(seed_xy.sel(space="y"))
        if np.isnan(x) or np.isnan(y):
            rejected += 1
            continue
        try:
            points = list(tracker.track(seed, x, y, range(seed + 1, end), cfg.seed_box_px, cfg.seed_min_circularity))
        except BadSeedError as bad:
            logger.info("%s: %s; trying the next confident frame", name, bad)
            rejected += 1
            continue
        break
    else:
        logger.warning(
            "%s: no usable seed among %d confident frames; frames %d-%d left as they are",
            name,
            rejected,
            run.frames.start,
            end - 1,
        )
        first, last, n_low = run.frames.start, end - 1, len(run.fill)
        return FillReport(
            tracker.video.name, individual, keypoint, run.seed, first, last, n_low, 0, float("nan"), rejected
        )

    wanted = set(run.fill)
    confidences: list[float] = []
    for point in points:
        if point.frame not in wanted or point.frame < window.start:
            continue
        loc = {"time": out["time"].values[point.frame], **sel}
        out["position"].loc[{**loc, "space": "x"}] = point.x
        out["position"].loc[{**loc, "space": "y"}] = point.y
        out["confidence"].loc[loc] = point.confidence
        flags.loc[loc] = True
        confidences.append(point.confidence)
    logger.info(
        "%s: seed %d (%d rejected), frames %d-%d, filled %d of %d",
        name,
        seed,
        rejected,
        seed + 1,
        end - 1,
        len(confidences),
        len(run.fill),
    )
    mean = float(np.mean(confidences)) if confidences else float("nan")
    n_low, n_filled = len(run.fill), len(confidences)
    return FillReport(
        tracker.video.name, individual, keypoint, seed, seed + 1, end - 1, n_low, n_filled, mean, rejected
    )


#: How each tracker's format is written back: ``source_software`` → (extension, writer).
SOURCE_WRITERS: dict[str, tuple[str, Callable[[xr.Dataset, Path], None]]] = {
    "DeepLabCut": (".h5", lambda ds, path: to_dlc_file(ds, path, split_individuals=False)),
    "SLEAP": (".h5", to_sleap_analysis_file),
    "LightningPose": (".csv", to_lp_file),
}


def write_output(ds: xr.Dataset, out_dir: Path, stem: str, output: str) -> Path:
    """Write ``{stem}_edited`` as movement NetCDF (``"netcdf"``) or in the tracker's own format (``"source"``).

    The NetCDF carries movement's ``edited`` flag; a tracker's format has no
    column for it, so there the changed frames are told apart by their values only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if output == "netcdf":
        path = out_dir / f"{stem}{OUTPUT_SUFFIX}.nc"
        ds.to_netcdf(path)
        return path
    if output != "source":
        raise ValueError(f"output must be 'source' or 'netcdf', got {output!r}")
    software = str(ds.attrs.get("source_software", ""))
    if software not in SOURCE_WRITERS:
        raise ValueError(
            f"cannot write {software!r} files back; movement writes {sorted(SOURCE_WRITERS)}, or use output='netcdf'"
        )
    extension, writer = SOURCE_WRITERS[software]
    path = out_dir / f"{stem}{OUTPUT_SUFFIX}{extension}"
    writer(ds[["position", "confidence"]], path)
    return path


def append_report(rows: list[FillReport], path: Path) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        if new:
            writer.writerow(FillReport.__dataclass_fields__)
        for row in rows:
            writer.writerow(asdict(row).values())


def output_file(out_dir: Path, stem: str) -> Path | None:
    """The fill's output for *stem* in *out_dir*, whichever format it was written in, or ``None``."""
    hits = [h for h in sorted(out_dir.glob(f"{stem}{OUTPUT_SUFFIX}.*")) if h.suffix.lower() in (".h5", ".csv", ".nc")]
    if len(hits) > 1:
        raise FileExistsError(f"several fill outputs for {stem!r} in {out_dir}: {[h.name for h in hits]}")
    return hits[0] if hits else None


def fill_pair(
    pair: Pair,
    segments: list[Segment],
    out_dir: str | Path,
    cfg: FillConfig,
    predictor: Any = None,
    skip_existing: bool = False,
) -> list[FillReport]:
    """Fill one pose/video pair inside its table segments; write the outputs; return the runs.

    *predictor* is a model from :func:`~ethograph.samfill.sam.build_predictor`
    shared across videos; without one the model is built for this video alone.
    With *skip_existing*, a pair whose output is already in *out_dir* is left as
    it is, so an interrupted batch resumes where it stopped.
    """
    from ethograph.samfill.samurai import SamuraiParams

    out_dir = Path(out_dir)
    if skip_existing and output_file(out_dir, pair.output_stem) is not None:
        logger.info("%s: already filled; skipped", pair.video.name)
        return []
    samurai = SamuraiParams() if cfg.samurai else None
    with SamPointTracker(pair.video, cfg.model, samurai, predictor) as tracker:
        assert tracker.fps is not None and tracker.n_frames is not None
        windows = segments_to_frames(segments, pair.video.name, tracker.fps, tracker.n_frames)
        if not windows:
            logger.info("%s: no segment in the table; skipped", pair.video.name)
            return []
        ds = load_dataset(pair.pose, source_software=cfg.source_software, fps=tracker.fps)
        filled, reports = fill_dataset(ds, tracker, windows, cfg)
    write_output(filled, out_dir, pair.output_stem, cfg.output)
    append_report(reports, out_dir / REPORT_NAME)
    return reports


def fill_folders(
    pose_dir: str | Path,
    video_dir: str | Path,
    segments_tsv: str | Path,
    out_dir: str | Path,
    cfg: FillConfig,
    skip_existing: bool = True,
) -> dict[str, list[FillReport]]:
    """Pair the folders, fill every pair the table names with one shared model, and return the runs per video.

    Pairs already written to *out_dir* are skipped unless *skip_existing* is off.
    """
    from ethograph.samfill.samurai import SamuraiParams

    pairs, lone_poses, lone_videos = pair_folders(pose_dir, video_dir, cfg.pose_extensions, cfg.video_extensions)
    for path in lone_poses:
        logger.info("no video for %s; ignored", path.name)
    for path in lone_videos:
        logger.info("no pose file for %s; ignored", path.name)
    if not pairs:
        raise FileNotFoundError(f"no pose/video pair between {pose_dir} and {video_dir}")
    segments = read_segments_tsv(segments_tsv)
    predictor, _ = build_predictor(cfg.model, SamuraiParams() if cfg.samurai else None)
    return {pair.name: fill_pair(pair, segments, out_dir, cfg, predictor, skip_existing) for pair in pairs}
