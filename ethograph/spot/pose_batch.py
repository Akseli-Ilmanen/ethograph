"""From labelled clips to a session with pose: fill, export, merge — headless.

The keypoint-labelling dialog works one clip at a time and saves a sidecar
beside each video (``<video>.keypoints.json``). A session cut into hundreds of
short clips is labelled clip by clip in the GUI — a few frames each, the
static keypoints carried along — and then everything else happens here,
without Qt, over every clip at once:

1. :func:`fill_and_export_video` — run a fill backend over one clip's sidecar
   and write the dense ``<video>.keypoints.nc`` the dialog's *Export* writes
   (:func:`~ethograph.gui.pose_annotate.store_to_dataset`, unchanged).
2. :func:`merge_keypoints` — sample every clip's keypoints onto its trial's
   time axis (``trial = video + offset``, the ``VideoSync`` convention) and
   write them into the session as an ordinary ``(time, space, keypoint,
   individual)`` variable the ``features:`` list reads (positions, and what you derive from them).

A tracker's output in movement format drops into the same place: anything
that writes ``<video>.keypoints.nc`` — DeepLabCut, SLEAP, the dialog — is
merged the same way, so the pipeline is not tied to how the pose was made.
Everything is 2-D, single camera, on purpose: the pose side's per-keypoint
channel count is just ``len(space)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ethograph.gui.pose_annotate import (
    KeypointStore,
    keypoints_dataset_path,
    sidecar_path,
    store_to_dataset,
)
from ethograph.gui.pose_fill import VideoFrameSource, build_backend, no_progress
from ethograph.io.netcdf import netcdf_engine
from ethograph.segment.sessions import Session, filter_trials
from ethograph.spot.config import SpotConfig
from ethograph.spot.dataset import probe_video
from ethograph.utils.xr_utils import get_time_coord

logger = logging.getLogger(__name__)

#: Longest side the tracking backends decode at — the dialog's own setting.
#: Near-free in accuracy at this anchor density, the single biggest CPU win.
MAX_SIDE = 640

#: The variable the merged keypoints land in, unless the session already has
#: one by that name (a 3-D pose, say) and the caller picks another.
POSE_VAR = "position"


def fill_and_export_video(
    video: Path,
    backend: str = "spline",
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Fill one clip's sidecar and write ``<video>.keypoints.nc``.

    Returns the path written, or ``None`` when the clip has no sidecar (it was
    never labelled — not an error, most clips of a pilot will not be). An
    existing export is reused unless *overwrite*.
    """
    video = Path(video)
    sidecar = sidecar_path(video)
    if not sidecar.is_file():
        return None
    out = keypoints_dataset_path(video)
    if out.is_file() and not overwrite:
        return out
    store = KeypointStore.load(sidecar)
    fps, n_frames, _w, _h = probe_video(video)
    store.n_frames = n_frames or store.n_frames
    if not store.observations():
        logger.warning("%s: sidecar has no labelled frames — skipped", video.name)
        return None
    engine = build_backend(backend, checkpoint=checkpoint, device=device, n_points=store.n_points)
    frames = VideoFrameSource(video, fps, store.n_frames, max_side=MAX_SIDE) if engine.requires_video else None
    try:
        filled, confidence = engine.fill(store.flat_observations(), store.n_frames, frames, no_progress)
    finally:
        if frames is not None:
            frames.close()
    if store.static_keypoints:
        filled, confidence = store.pin_static(filled, confidence)
    store.set_fill_from_flat(filled, confidence)
    ds = store_to_dataset(store, fps)
    ds.to_netcdf(out, engine=netcdf_engine(out))
    logger.info(
        "%s: filled %d keypoints over %d frames with %s -> %s",
        video.name,
        store.n_keypoints,
        n_frames,
        backend,
        out.name,
    )
    return out


def session_clips(session: Session, config: SpotConfig) -> dict[int | str, Path]:
    """Trial -> video, for every trial passing the filter that has one."""
    out: dict[int | str, Path] = {}
    camera = session.video_device(config.labels.camera)
    for trial in filter_trials(session, config.trials):
        video = session.media_path(trial, "video", device=camera)
        if video is not None:
            out[trial] = video
    return out


def fill_and_export_session(session: Session, config: SpotConfig, backend: str = "spline", **kw) -> list[Path]:
    """Every labelled clip of *session*, filled and exported. Returns what was written."""
    written: list[Path] = []
    clips = session_clips(session, config)
    for trial, video in clips.items():
        path = fill_and_export_video(video, backend, **kw)
        if path is not None:
            written.append(path)
    logger.info("%s: %d of %d clips carry labels and were exported", session.spec.label, len(written), len(clips))
    return written


def sample_onto_trial(ds: xr.Dataset, pose: xr.Dataset, offset: float, var: str = POSE_VAR) -> xr.Dataset:
    """*ds* with *pose*'s keypoints on its own time axis.

    *pose* is a movement dataset on the video's clock (frame 0 at 0 s). The
    stream offset places that frame 0 at trial time ``offset`` (``VideoSync``:
    trial = video + offset), so a trial time samples the video at
    ``t - offset``, nearest frame. Trial times outside the clip read ``NaN``.
    A variable already called *var* is refused: merging must never silently
    replace a pose the session already has.
    """
    if var in ds.data_vars:
        raise ValueError(f"the trial already has a variable {var!r} — merge under another name")
    reference = next(iter(ds.data_vars.values()))
    time_coord = get_time_coord(reference)
    if time_coord is None:
        raise ValueError("the trial has no time coord to sample the keypoints onto")
    time = np.asarray(time_coord.values, dtype=np.float64)
    video_time = time - float(offset)
    inside = (video_time >= float(pose.time.values[0])) & (video_time <= float(pose.time.values[-1]))
    out: dict[str, xr.DataArray] = {}
    for name in ("position", "confidence"):
        if name not in pose:
            continue
        da = pose[name].interp(time=video_time, method="nearest", kwargs={"fill_value": np.nan})
        da = da.rename({"time": time_coord.name}).assign_coords({time_coord.name: time})
        da = da.where(xr.DataArray(inside, dims=[time_coord.name]))
        da.attrs = {
            **pose[name].attrs,
            "description": f"{name} from <video>.keypoints.nc, sampled onto the trial clock",
        }
        out[var if name == "position" else f"{var}_confidence"] = da
    return ds.assign(out)


def merge_keypoints(
    session: Session,
    config: SpotConfig,
    var: str = POSE_VAR,
    out_path: Path | None = None,
    in_place: bool = False,
) -> Path:
    """Write a copy of *session* carrying every clip's keypoints as *var*.

    Trials without an export are left without the variable (``NaN`` is not
    written for them — a missing pose is missing, not zero). The result is a
    sibling ``{stem}_pose2d.nc`` unless *out_path* or *in_place* says
    otherwise, because the pipeline never overwrites a session file unasked.
    """
    dt = session.result.dt
    if dt is None:
        raise ValueError(f"{session.source}: merging needs an xarray (.nc) session")
    alignment = session.result.nwb_alignment
    camera = session.video_device(config.labels.camera)
    merged = 0
    for trial, video in session_clips(session, config).items():
        export = keypoints_dataset_path(video)
        if not export.is_file():
            continue
        pose = xr.load_dataset(export, engine=netcdf_engine(export))
        offset = float(alignment.stream_offset_for_trial(trial, "video", device=camera))
        dt.update_trial(trial, lambda ds, p=pose, o=offset: sample_onto_trial(ds, p, o, var))
        merged += 1
    if merged == 0:
        raise FileNotFoundError(f"{session.spec.label}: no clip has a <video>.keypoints.nc — fill and export first")
    if in_place:
        target = session.source
    elif out_path is not None:
        target = Path(out_path)
    else:
        target = session.source.with_name(f"{session.source.stem}_pose2d.nc")
    dt.save(str(target))
    logger.info("%s: merged keypoints into %d trials as %r -> %s", session.spec.label, merged, var, target)
    return target
