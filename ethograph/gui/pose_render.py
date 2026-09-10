"""Pose rendering pipeline: lazy loading + dynamic filtering.

Pure functions load pose keypoint data and return PoseRenderData.
PoseDisplayManager orchestrates display using keypoint table selections to
drive which keypoints are shown/hidden.

Two loading paths:
- File-based (DLC, SLEAP, etc.): via ``movement.io.load_dataset``
- NWB-based: direct HDF5 reads from ``PoseEstimationSeries`` with lazy slicing

Display is a pygfx overlay (:mod:`ethograph.gui.pose_overlay`) streamed per
frame on each camera view — no napari layers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from movement.io import load_dataset

from ethograph.gui.app_constants import POSE_SOFTWARES
from ethograph.gui.notify import notify
from ethograph.gui.pose_convert import (
    COLOR_BY_KEYPOINT,
    COLOR_BY_MODES,
    individual_color_map,
    poses_ds_to_points,
    sample_colormap,
)
from ethograph.gui.pose_overlay import OverlayStyle, PoseOverlayData
from ethograph.io.nwb_alignment import pose_keys_for_cameras, pose_video_links_from_nwb
from ethograph.io.nwb_import import _get_absolute_timestamps
from ethograph.io.time_model import trial_frame_window
from ethograph.skeleton import nwb_skeleton_to_config
from ethograph.skeleton.config import hex_to_rgba

logger = logging.getLogger(__name__)


@dataclass
class PoseRenderData:
    """Immutable result of the pose loading + filtering pipeline.

    data         : shape (N, 3) — [frame_idx, y, x] (or (N, 4) with track_id first)
    properties   : DataFrame with per-point metadata (keypoint, individual, confidence, ...)
    data_not_nan : bool mask shape (N,) — True for points that should be shown
    file_name    : label used as the display name base
    """

    data: np.ndarray
    properties: pd.DataFrame
    data_not_nan: np.ndarray
    file_name: str
    bbox_data: np.ndarray | None = None
    frame_path: str | None = None
    skeleton_config: dict | None = None

    @property
    def keypoints(self) -> list[str]:
        if "keypoint" not in self.properties.columns:
            return []
        return self.properties["keypoint"].unique().tolist()


def strip_common_prefix(names: list[str]) -> list[str]:
    """Remove the longest common prefix shared by all names."""
    if len(names) <= 1:
        return names
    prefix = os.path.commonprefix(names)
    if not prefix:
        return names
    return [n[len(prefix) :] for n in names]


def _strip_keypoint_prefix(properties: pd.DataFrame) -> pd.DataFrame:
    if "keypoint" not in properties.columns:
        return properties
    names = properties["keypoint"].tolist()
    prefix = os.path.commonprefix(names)
    if not prefix:
        return properties
    props = properties.copy()
    props["keypoint"] = props["keypoint"].str[len(prefix) :]
    return props


#: Extension of a poses dataset saved as NetCDF — what the keypoint labelling
#: dialog writes, and what ``movement`` datasets are usually stored as. It is a
#: pose file like any other, so it is read here rather than being a case the
#: caller has to know about; ``movement.io.load_dataset`` only takes the formats
#: of the *tracking tools*, and has nothing to convert from.
POSES_DATASET_SUFFIX = ".nc"


def ask_pose_source_software(pose_path: str | Path, parent=None) -> str | None:
    """One-time question: which software produced the pose files?

    xarray datasets carry ``source_software`` in their attrs, but pynapple
    sessions have no dataset to carry it — and ``movement.load_dataset``
    refuses ``source_software=None``, so without an answer the pose overlay
    can never load there. The answer is cached on
    ``app_state.source_software`` by the caller (persisted per dataset), so
    the question is asked once. Returns ``None`` on cancel / headless.
    """
    from qtpy.QtWidgets import QInputDialog

    from ethograph.gui import notify as _notify

    if _notify.SUPPRESS:
        return None
    item, ok = QInputDialog.getItem(
        parent,
        "Pose source software",
        f"Which software produced the pose files?\n({Path(pose_path).name})",
        POSE_SOFTWARES,
        0,
        False,
    )
    return item if ok and item else None


def load_pose_from_file(file_path: str, source_software: str, fps: float) -> PoseRenderData:
    """Load a pose file and return a PoseRenderData.

    A movement poses dataset saved as NetCDF is opened directly; everything else
    goes through ``movement.io.load_dataset`` to be converted from its tracking
    tool's own format first. Either way the result is one poses Dataset, and
    everything after this point is identical.
    """
    if Path(file_path).suffix.lower() == POSES_DATASET_SUFFIX:
        # Read into memory and close: an open NetCDF handle per pose load leaks
        # one for every camera, every trial change and every re-render.
        with xr.open_dataset(file_path) as opened:
            ds = opened.load()
    else:
        ds = load_dataset(file_path, source_software, fps)
    data, bbox_data, properties = poses_ds_to_points(ds)
    return PoseRenderData(
        data=data,
        properties=_strip_keypoint_prefix(properties),
        data_not_nan=~np.any(np.isnan(data), axis=1),
        file_name=Path(file_path).name,
        bbox_data=bbox_data,
        frame_path=ds.attrs.get("frame_path"),
    )


def slice_pose_to_frames(pr: PoseRenderData, start_frame: int, end_frame: int) -> PoseRenderData:
    """Slice pose data to ``[start_frame, end_frame)`` and reindex to 0.

    Mirrors the video trial clipping so that pose frame indices align with
    the trial-clipped video. Works for both points and bounding-box data.
    """
    frame_col = 1 if pr.data.shape[1] > 3 else 0
    frames = pr.data[:, frame_col]
    mask = (frames >= start_frame) & (frames < end_frame)

    if not np.any(mask):
        empty = np.empty((0, pr.data.shape[1]))
        return PoseRenderData(
            data=empty,
            properties=pr.properties.iloc[0:0].reset_index(drop=True),
            data_not_nan=np.empty(0, dtype=bool),
            file_name=pr.file_name,
            frame_path=pr.frame_path,
            skeleton_config=pr.skeleton_config,
        )

    data = pr.data[mask].copy()
    data[:, frame_col] -= start_frame

    bbox_data = None
    if pr.bbox_data is not None:
        bbox_data = pr.bbox_data[mask].copy()
        bbox_data[:, :, 1] -= start_frame  # frame column in each corner vertex

    return PoseRenderData(
        data=data,
        properties=pr.properties[mask].reset_index(drop=True),
        data_not_nan=pr.data_not_nan[mask],
        file_name=pr.file_name,
        bbox_data=bbox_data,
        frame_path=pr.frame_path,
        skeleton_config=pr.skeleton_config,
    )


def load_pose_from_nwb_direct(
    nwb_file: Any,
    pose_estimation_key: str,
    t_start: float | None = None,
    t_stop: float | None = None,
) -> PoseRenderData | None:
    """Load pose directly from NWB PoseEstimationSeries via lazy HDF5 slicing.

    Reads ``series.data`` and ``series.confidence`` using searchsorted for
    efficient time-based slicing. No xarray or movement conversion needed.

    Parameters
    ----------
    nwb_file
        Open pynwb.NWBFile object (provides access to processing modules).
    pose_estimation_key
        Container name (e.g., "pose_dlc", "pose_sleap").
    t_start, t_stop
        Optional time window [t_start, t_stop). If provided, slices data
        and makes time trial-relative (0-relative).

    Returns
    -------
    PoseRenderData | None
        Stacked keypoint data with per-point properties, or None if data
        is unavailable or empty after slicing.
    """
    # Find the pose estimation container
    proc_key = "pose_estimation"
    for mod_name, mod in nwb_file.processing.items():
        if pose_estimation_key in mod.data_interfaces:
            proc_key = mod_name
            break

    container = nwb_file.processing[proc_key][pose_estimation_key]
    raw_kp_names = list(container.pose_estimation_series.keys())
    stripped = strip_common_prefix(raw_kp_names)
    name_map = dict(zip(raw_kp_names, stripped))

    all_pts: list[np.ndarray] = []
    all_not_nan: list[np.ndarray] = []
    kp_col: list[str] = []
    ind_col: list[str] = []
    conf_col: list[float] = []

    for kp_name, series in container.pose_estimation_series.items():
        ts = _get_absolute_timestamps(series)
        n_frames = series.data.shape[0]

        # Determine time window index range
        if t_start is not None and t_stop is not None:
            idx = np.where((ts >= t_start) & (ts <= t_stop))[0]
            if len(idx) == 0:
                continue
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
        else:
            i0, i1 = 0, n_frames

        # Lazy slice from HDF5
        data = np.asarray(series.data[i0:i1], dtype=np.float64)
        n = len(data)
        frames = np.arange(n, dtype=np.float64)

        # NWB stores (x, y); points rows are (frame, y, x)
        pts = np.column_stack([frames, data[:, 1], data[:, 0]])
        not_nan = ~np.any(np.isnan(data[:, :2]), axis=1)

        # Optionally load confidence
        confidence: np.ndarray | None = None
        if hasattr(series, "confidence") and series.confidence is not None:
            try:
                confidence = np.asarray(series.confidence[i0:i1], dtype=np.float64)
            except Exception:
                confidence = None

        all_pts.append(pts)
        all_not_nan.append(not_nan)
        kp_col.extend([name_map[kp_name]] * n)
        ind_col.extend([pose_estimation_key] * n)
        conf_col.extend(confidence.tolist() if confidence is not None else [1.0] * n)

    if not all_pts:
        return None

    return PoseRenderData(
        data=np.vstack(all_pts),
        properties=pd.DataFrame(
            {
                "keypoint": kp_col,
                "individual": ind_col,
                "confidence": conf_col,
            }
        ),
        data_not_nan=np.concatenate(all_not_nan),
        file_name=f"NWB_pose_{pose_estimation_key}",
        skeleton_config=_read_skeleton_config(container, set(stripped)),
    )


def _read_skeleton_config(container: Any, keypoints: set[str]) -> dict | None:
    """Read an ndx-pose ``Skeleton`` and return a skeleton config dict.

    The skeleton's ``nodes``/``edges`` (node-index pairs) are converted to the
    config format consumed by ``ethograph.skeleton`` via ``nwb_skeleton_to_config``
    — this is the NWB *config layer*; the renderer is reused unchanged. Node
    names are reconciled with the rendered keypoint names (which may have had a
    common prefix stripped). Returns ``None`` if no skeleton is attached.
    """
    skel = getattr(container, "skeleton", None)
    if skel is None:
        return None

    nodes = [n.decode() if isinstance(n, bytes) else str(n) for n in np.asarray(skel.nodes[:]).ravel()]
    edges = np.asarray(skel.edges[:]).astype(int).reshape(-1, 2)

    name_map = _match_skeleton_nodes(nodes, keypoints)
    mapped = [name_map.get(n) for n in nodes]
    config = nwb_skeleton_to_config(mapped, edges)
    return config if config["connections"] else None


def _match_skeleton_nodes(nodes: list[str], keypoints: set[str]) -> dict[str, str]:
    """Map skeleton node names to rendered keypoint names.

    Tries an exact match first, then a common-prefix-stripped match so that
    skeletons authored against raw series names still line up with the
    prefix-stripped keypoints used by the points display.
    """
    mapping = {n: n for n in nodes if n in keypoints}
    if len(mapping) == len(nodes):
        return mapping
    stripped = dict(zip(nodes, strip_common_prefix(nodes)))
    for n in nodes:
        if n not in mapping and stripped[n] in keypoints:
            mapping[n] = stripped[n]
    return mapping


def _resolve_skeleton_colors(config: dict | None, base_color: str | None) -> dict | None:
    """Recolour every skeleton edge AND anchored shape with the uniform ``base_color``.

    Called only when the "use base colour" checkbox is on; otherwise the
    per-edge/per-shape colours from the editor / NWB config are used as-is.
    """
    if config is None or not base_color:
        return config
    connections = [{**c, "color": base_color} for c in config["connections"]]
    shapes = [{**s, "color": base_color} for s in config.get("shapes", [])]
    return {**config, "connections": connections, "shapes": shapes}


def pose_render_to_movement_ds(pr: PoseRenderData) -> xr.Dataset:
    """Rebuild a movement-format poses ``xr.Dataset`` from a ``PoseRenderData``.

    The skeleton editor consumes a movement poses dataset, so this adapter
    un-flattens the points back into a ``(time, space, keypoints, individuals)``
    ``position`` array. Points masked out by ``data_not_nan`` become NaN.
    """
    coords = pr.data[:, -3:]
    frames = coords[:, 0].astype(int)
    ys = coords[:, 1]
    xs = coords[:, 2]

    kp = pr.properties["keypoint"].to_numpy()
    if "individual" in pr.properties.columns:
        ind = pr.properties["individual"].to_numpy()
    else:
        ind = np.array(["ind_0"] * len(kp))
    conf = pr.properties["confidence"].to_numpy() if "confidence" in pr.properties.columns else np.ones(len(kp))

    keypoints = list(dict.fromkeys(kp))
    individuals = list(dict.fromkeys(ind))
    kp_idx = {n: i for i, n in enumerate(keypoints)}
    ind_idx = {n: i for i, n in enumerate(individuals)}
    n_t = int(frames.max()) + 1 if len(frames) else 0

    position = np.full((n_t, 2, len(keypoints), len(individuals)), np.nan)
    confidence = np.full((n_t, len(keypoints), len(individuals)), np.nan)
    for row, valid in enumerate(pr.data_not_nan):
        if not valid:
            continue
        t, k, i = frames[row], kp_idx[kp[row]], ind_idx[ind[row]]
        position[t, 0, k, i] = xs[row]
        position[t, 1, k, i] = ys[row]
        confidence[t, k, i] = conf[row]

    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(position, dims=["time", "space", "keypoints", "individuals"]),
            "confidence": xr.DataArray(confidence, dims=["time", "keypoints", "individuals"]),
        },
        coords={
            "time": np.arange(n_t),
            "space": ["x", "y"],
            "keypoints": keypoints,
            "individuals": individuals,
        },
        attrs={"ds_type": "poses"},
    )


def movement_ds_to_pose_render(ds: xr.Dataset, file_name: str) -> PoseRenderData:
    """Inverse of :func:`pose_render_to_movement_ds`.

    Lets an in-memory poses dataset (e.g. one built by the keypoint labelling
    dialog) render through exactly the same path as an imported DLC file.
    """
    data, bbox_data, properties = poses_ds_to_points(ds)
    return PoseRenderData(
        data=data,
        properties=properties,
        data_not_nan=~np.any(np.isnan(data), axis=1),
        file_name=file_name,
        bbox_data=bbox_data,
    )


def apply_confidence_filter(pr: PoseRenderData, threshold: float) -> PoseRenderData:
    """Mask out points below confidence threshold (UI-driven filtering)."""
    if threshold <= 0.0 or "confidence" not in pr.properties.columns:
        return pr
    mask = pr.data_not_nan.copy()
    mask[pr.properties["confidence"].values < threshold] = False
    return PoseRenderData(pr.data, pr.properties, mask, pr.file_name, pr.bbox_data, pr.frame_path, pr.skeleton_config)


def apply_keypoint_filter(pr: PoseRenderData, hidden: set[str]) -> PoseRenderData:
    """Mask out hidden keypoints from the keypoints table selection."""
    if not hidden or "keypoint" not in pr.properties.columns:
        return pr
    mask = pr.data_not_nan.copy()
    mask[pr.properties["keypoint"].isin(hidden).values] = False
    return PoseRenderData(pr.data, pr.properties, mask, pr.file_name, pr.bbox_data, pr.frame_path, pr.skeleton_config)


def apply_individual_filter(pr: PoseRenderData, keep: str) -> PoseRenderData:
    """Mask out every individual but *keep* — what a camera view pinned to one animal shows."""
    if "individual" not in pr.properties.columns:
        return pr
    mask = pr.data_not_nan.copy()
    mask[pr.properties["individual"].astype(str).values != str(keep)] = False
    return PoseRenderData(pr.data, pr.properties, mask, pr.file_name, pr.bbox_data, pr.frame_path, pr.skeleton_config)


class PoseDisplayManager:
    """Manages pose loading, filtering, and pygfx overlay display.

    Keypoint filtering is driven by the keypoints table in the UI:
    - Confidence threshold filters out low-confidence points
    - Hidden keypoints in the table prevent display

    Uses a single rendering path for all cameras (primary and extra): each
    camera view gets a :class:`PoseOverlay` fed a :class:`PoseOverlayData`
    cube; a masked (or hidden) keypoint becomes NaN, and the overlay drops
    any skeleton edge touching it on that frame automatically.
    """

    def __init__(self, video_area, app_state, video_manager, data_widget):
        self.video_area = video_area
        self.app_state = app_state
        self.video_mgr = video_manager
        self._data_widget = data_widget
        self._primary_file_name: str = ""
        self._primary_pr: PoseRenderData | None = None
        self._extra_pr: dict[str, PoseRenderData] = {}
        self._camera_keypoints: dict[str, list[str]] = {}
        #: In-memory pose shown on the primary camera instead of the file/NWB
        #: one — set by the keypoint labelling dialog so filled keypoints are
        #: indistinguishable from imported predictions.
        self._pose_override: PoseRenderData | None = None

    @property
    def all_keypoints(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for kps in self._camera_keypoints.values():
            for k in kps:
                if k not in seen:
                    seen.add(k)
                    result.append(k)
        return result

    def _camera_index(self, camera_name: str | None = None) -> int:
        return self.app_state.nwb_alignment.cameras.index(camera_name)

    def set_pose_override(self, pr: PoseRenderData | None) -> None:
        """Show *pr* on the primary camera instead of its loaded pose."""
        self._pose_override = pr

    def _primary_camera_index(self) -> int | None:
        name = self._primary_camera_name()
        cameras = self.app_state.nwb_alignment.cameras
        return cameras.index(name) if name in cameras else None

    def _camera_name_for_index(self, camera_idx: int) -> str:
        cameras = self.app_state.nwb_alignment.cameras
        return cameras[camera_idx] if camera_idx < len(cameras) else str(camera_idx)

    def _resolve_camera_fps(self, camera_idx: int) -> float:
        sio = self.app_state.nwb_alignment  # sio = session io (legacy name)
        cameras = sio.cameras
        if camera_idx < len(cameras):
            fps = sio.get_stream_rate("video", cameras[camera_idx])
            if fps is not None and fps > 0:
                return fps
        return self.app_state.video_fps

    def _get_nwb_file(self) -> Any | None:
        sio = self.app_state.nwb_alignment
        return getattr(sio, "nwb", None)

    def _pose_device(self, camera_idx: int) -> str | None:
        """Which pose stream feeds camera *camera_idx*.

        Usually the camera's own name (``pose_cam-1``), but a dataset may
        name the stream for what it holds (``pose_2d``) while its camera is
        ``cam-1`` — so pose streams pair with cameras by index whenever the
        camera's name isn't one of them. ``pose_keys`` sees both trials-table
        columns and acquisitions, so a stream missing there doesn't exist.
        """
        sio = self.app_state.nwb_alignment
        pose_devices = sio.pose_keys
        cameras = sio.cameras
        camera = cameras[camera_idx] if camera_idx < len(cameras) else None
        if camera in pose_devices:
            return camera
        return pose_devices[camera_idx] if camera_idx < len(pose_devices) else None

    def _load_pose_for_camera(self, camera_idx: int) -> PoseRenderData | None:
        if self._pose_override is not None and camera_idx == self._primary_camera_index():
            return self._pose_override

        trial_id = self.app_state.trials_sel
        sio = self.app_state.nwb_alignment
        cameras = sio.cameras

        if camera_idx < len(cameras):
            pose_path = sio.resolve_media_path(
                trial_id,
                "pose",
                device=self._pose_device(camera_idx),
                fallback_folder=self.app_state.pose_folder,
            )
            if not pose_path:
                return None
            try:
                source_software = self.app_state.source_software or getattr(self.app_state.ds, "source_software", None)
                if not source_software and Path(pose_path).suffix.lower() != POSES_DATASET_SUFFIX:
                    # A movement .nc is already a dataset; only a tracking
                    # tool's own format needs the converter told which tool.
                    if getattr(self, "_pose_software_declined", False):
                        return None
                    source_software = ask_pose_source_software(pose_path)
                    if not source_software:
                        # Declined: don't re-ask on every trial change/redraw.
                        self._pose_software_declined = True
                        return None
                    # Cache the answer (persisted per dataset) — pynapple
                    # sessions have no dataset attrs to carry it.
                    self.app_state.source_software = source_software
                pr = load_pose_from_file(
                    pose_path,
                    source_software,
                    self._resolve_camera_fps(camera_idx),
                )
            except (OSError, ValueError, KeyError) as e:
                notify(f"Failed to load pose for camera {camera_idx}: {e}", "warning")
                return None
            alignment = getattr(self.app_state, "trial_alignment", None)
            if alignment and alignment.trial_range:
                fps = self._resolve_camera_fps(camera_idx)
                time_offset = sio.stream_offset_for_trial(
                    trial_id,
                    "video",
                    cameras[camera_idx],
                )
                start_frame, end_frame = trial_frame_window(alignment.trial_range, fps, time_offset)
                pr = slice_pose_to_frames(pr, start_frame, end_frame)
            return pr

        nwb_file = self._get_nwb_file()

        # Pose→camera pairing priority:
        #   1. manual / saved mapping (app_state.nwb_pose_keys),
        #   2. native ndx-pose PoseEstimation.source_video links,
        #   3. device-name fallback (pose_cam-N ↔ video_cam-N).
        pose_keys = list(getattr(self.app_state, "nwb_pose_keys", None) or [])
        if not pose_keys and nwb_file is not None and sio:
            links = pose_video_links_from_nwb(nwb_file)
            if links:
                pose_keys = pose_keys_for_cameras(links, sio.cameras)
        if not pose_keys and sio:
            pose_keys = sio.pose_keys

        if pose_keys and camera_idx < len(pose_keys) and pose_keys[camera_idx]:
            if nwb_file is None:
                return None
            try:
                trial_id = self.app_state.trials_sel
                t_start = sio.start_time(trial_id) if trial_id else None
                t_stop = sio.stop_time(trial_id) if trial_id else None
                return load_pose_from_nwb_direct(
                    nwb_file,
                    pose_keys[camera_idx],
                    t_start=t_start,
                    t_stop=t_stop,
                )
            except (OSError, ValueError, KeyError) as e:
                notify(
                    f"Failed to load NWB pose for {pose_keys[camera_idx]}: {e}",
                    "warning",
                )
                return None
        return None

    def _prepare_pose(self, camera_idx: int, hidden_keypoints: set[str]) -> PoseRenderData | None:
        """Load pose and apply UI-driven filtering (confidence + keypoint table selection)."""
        pr = self._load_pose_for_camera(camera_idx)
        if pr is None:
            return None
        # Apply filters driven by UI keypoints table
        pr = apply_confidence_filter(pr, self.app_state.pose_hide_threshold)
        pr = apply_keypoint_filter(pr, hidden_keypoints)
        return pr if np.any(pr.data_not_nan) else None

    # ------------------------------------------------------------------
    # Per-camera keypoint tracking
    # ------------------------------------------------------------------

    def _register_keypoints(self, camera_name: str, keypoints: list[str] | None):
        if keypoints:
            self._camera_keypoints[camera_name] = keypoints
        self._sync_global_keypoints()

    def _sync_global_keypoints(self):
        merged = self.all_keypoints
        if merged != self.app_state.keypoints:
            self.app_state.keypoints = merged

    def on_camera_removed(self, camera_name: str):
        self._camera_keypoints.pop(camera_name, None)
        self._extra_pr.pop(camera_name, None)
        self._sync_global_keypoints()

    # ------------------------------------------------------------------
    # Unified display — same overlay path for primary and extra cameras
    # ------------------------------------------------------------------

    def _display_frame_background(self, pr: PoseRenderData) -> None:
        """Show a static frame image as background when no video is loaded.

        Checks (in order): user-provided frame path on the widget, then
        ``frame_path`` from the movement dataset attributes.
        """
        if self.app_state.video_path:
            return
        frame_path = getattr(self._data_widget, "pose_frame_path", None) or pr.frame_path
        if not frame_path:
            return
        frame_file = Path(frame_path)
        if not frame_file.exists():
            return
        import imageio.v3 as iio

        img = iio.imread(frame_file)
        self.video_area.primary.set_static_image(img)

    def _skeleton_enabled(self) -> bool:
        checkbox = getattr(self._data_widget, "pose_show_skeleton_checkbox", None)
        if checkbox is not None:
            return checkbox.isChecked()
        return bool(getattr(self.app_state, "pose_show_skeleton", False))

    def _resolved_skeleton_config(self, pr: PoseRenderData) -> dict | None:
        """Skeleton + shapes config after override/base-colour resolution."""
        if not self._skeleton_enabled():
            return None
        override = getattr(self.app_state, "skeleton_config_override", None)
        config = override if override is not None else pr.skeleton_config
        if getattr(self.app_state, "skeleton_use_base", True):
            config = _resolve_skeleton_colors(config, getattr(self.app_state, "skeleton_base_color", None))
        return config

    def _points_visible(self) -> bool:
        checkbox = getattr(self._data_widget, "pose_show_keypoints_checkbox", None)
        return checkbox.isChecked() if checkbox is not None else True

    def _color_by(self, properties: pd.DataFrame) -> str:
        """The property colour encodes — the user's choice, if the data has it.

        A poses dataset without a ``keypoint`` column (bounding boxes) has only
        one axis to colour by, so the setting cannot be honoured there.
        """
        color_by = getattr(self.app_state, "pose_color_by", COLOR_BY_KEYPOINT)
        if color_by not in COLOR_BY_MODES or color_by not in properties.columns:
            return "individual"
        return color_by

    def _build_overlay_style(self, properties: pd.DataFrame) -> OverlayStyle:
        # Colour encodes ONE axis, chosen by the user (SLEAP's toggle): keypoint
        # colours shared across individuals, or one colour per individual shared
        # across its keypoints. Text then carries the OTHER axis, so turning
        # labels on always adds what the colours are not already saying.
        color_prop = self._color_by(properties)
        text_prop = "individual" if color_prop == "keypoint" else "keypoint"
        if text_prop not in properties.columns or properties[text_prop].nunique() <= 1:
            text_prop = color_prop

        if color_prop == "keypoint" and self.all_keypoints:
            values = self.all_keypoints
        elif color_prop == "individual":
            # The dataset's own list, in its order, plus any name only the
            # pose file knows: the colour of an animal never depends on which
            # animals happen to be drawn beside it.
            values = self.app_state.label_individuals()
            values += [v for v in properties["individual"].astype(str).unique().tolist() if v not in values]
        else:
            values = properties[color_prop].unique().tolist()
        if getattr(self.app_state, "pose_points_use_base", False):
            base = getattr(self.app_state, "pose_points_base_color", None) or "#FF3333"
            rgba = hex_to_rgba(base)
            color_map = {v: rgba for v in values}
        elif color_prop == "individual":
            color_map = individual_color_map(values, getattr(self.app_state, "pose_individual_colors", None))
        else:
            cycle = sample_colormap(len(values), "turbo")
            color_map = dict(zip(values, cycle))

        return OverlayStyle(
            color_prop=color_prop,
            text_prop=text_prop,
            color_map=color_map,
            point_size=self._data_widget.pose_point_size_spin.value(),
            points_visible=self._points_visible(),
            text_size=self._data_widget.pose_text_size_spin.value(),
            text_visible=self._data_widget.pose_show_text_checkbox.isChecked(),
            edge_width=self._skeleton_width(),
        )

    def _display_pose_on_view(self, view, pr: PoseRenderData, camera_name: str | None) -> None:
        overlay = view.ensure_overlay()
        if overlay is None:
            return
        data = PoseOverlayData(pr)
        style = self._build_overlay_style(pr.properties)
        overlay.set_data(
            data,
            style,
            img_height=view.image_height(),
            skeleton_config=self._resolved_skeleton_config(pr),
        )
        overlay.set_frame(int(getattr(self.app_state, "current_frame", 0) or 0))
        view.request_draw()

    def primary_pose_for_editor(self) -> tuple[list[str], np.ndarray] | None:
        """Return ``(keypoints, positions)`` for the primary camera's pose.

        ``positions`` has shape ``(n_frames, n_keypoints, 2)`` in image-space
        ``(x, y)`` — the input format the skeleton editor dialog expects.
        """
        combo = getattr(self._data_widget, "primary_camera_combo", None)
        name = combo.currentText() if combo is not None else None
        if name is None:
            return None
        pr = self._load_pose_for_camera(self._camera_index(name))
        if pr is None:
            return None
        ds = pose_render_to_movement_ds(pr)
        positions = ds.position.isel(individuals=0).transpose("time", "keypoints", "space").values
        return list(ds.coords["keypoints"].values), positions

    def _skeleton_width(self) -> float:
        spin = getattr(self._data_widget, "pose_skeleton_width_spin", None)
        return spin.value() if spin is not None else 2.0

    def update_pose(self, hidden_keypoints: set[str]) -> None:
        """Update pose display for all cameras based on keypoints table selection.

        Parameters
        ----------
        hidden_keypoints
            Set of keypoint names to hide (from the UI keypoints table).
        """
        primary_name = self._primary_camera_name()
        if primary_name is not None:
            self._display_pose_on_primary(self._camera_index(primary_name), hidden_keypoints)

        for key, view in self.video_mgr.extra_widgets.items():
            if getattr(view, "static_image_path", None):
                self._display_pose_on_image(view, hidden_keypoints)
            else:
                self._display_pose_on_extra(getattr(view, "camera_name", key), hidden_keypoints, view)

    def _primary_camera_name(self) -> str | None:
        combo = getattr(self._data_widget, "primary_camera_combo", None)
        name = combo.currentText() if combo is not None else None
        return name or None

    def _display_pose_on_primary(self, camera_idx: int, hidden_keypoints: set[str]) -> None:
        """Render pose on the primary camera view (driven by keypoints table selection)."""
        view = self.video_area.primary
        pr = self._pinned_only(view, self._prepare_pose(camera_idx, hidden_keypoints))
        if pr is None:
            view.clear_overlay()
            self._primary_pr = None
            return
        camera_name = self._camera_name_for_index(camera_idx)
        self._register_keypoints(camera_name, pr.keypoints)
        self._primary_file_name = pr.file_name
        self._display_frame_background(pr)
        self._primary_pr = pr
        self._display_pose_on_view(view, pr, camera_name)
        if view.plot is None:
            # Primary shows a still image — the overlay has no frame clock,
            # so time-marker updates drive it via the pose's own fps.
            view.static_pose_fps = self._resolve_camera_fps(camera_idx)

    def _display_pose_on_image(self, view: Any, hidden_keypoints: set[str]) -> None:
        """Render the PRIMARY camera's pose on a static-image view.

        Images are timeless media; the overlay mirrors the primary camera's
        pose and animates with the time marker (``CameraView.set_overlay_time``).
        """
        primary_name = self._primary_camera_name()
        if primary_name is None or primary_name not in self.app_state.nwb_alignment.cameras:
            view.clear_overlay()
            return
        camera_idx = self._camera_index(primary_name)
        pr = self._prepare_pose(camera_idx, hidden_keypoints)
        if pr is None:
            view.clear_overlay()
            return
        self._display_pose_on_view(view, pr, None)
        view.static_pose_fps = self._resolve_camera_fps(camera_idx)

    def _display_pose_on_extra(
        self,
        camera_name: str,
        hidden_keypoints: set[str],
        view: Any,
    ) -> None:
        """Render pose on an extra camera view (driven by keypoints table selection)."""
        if not camera_name:
            view.clear_overlay()
            return
        pr = self._pinned_only(view, self._prepare_pose(self._camera_index(camera_name), hidden_keypoints))
        if pr is None:
            view.clear_overlay()
            return
        self._register_keypoints(camera_name, pr.keypoints)
        self._extra_pr[camera_name] = pr
        self._display_pose_on_view(view, pr, camera_name)

    def _pinned_only(self, view: Any, pr: PoseRenderData | None) -> PoseRenderData | None:
        """*pr* reduced to the individual *view* is pinned to (``None`` when nothing of it remains)."""
        if pr is None:
            return None
        individual = self.app_state.pinned_individual_of(view)
        if individual is None:
            return pr
        pr = apply_individual_filter(pr, individual)
        return pr if np.any(pr.data_not_nan) else None

    def update_extra_camera_pose(self, camera_name: str, hidden_keypoints: set[str]) -> None:
        for key, view in self.video_mgr.extra_widgets.items():
            if getattr(view, "camera_name", key) == camera_name:
                self._display_pose_on_extra(camera_name, hidden_keypoints, view)

    def _all_views(self) -> list:
        return [self.video_area.primary, *self.video_mgr.extra_widgets.values()]

    def pose_kind(self) -> str | None:
        """``"bboxes"`` when what is drawn is bounding boxes, ``"poses"`` for
        keypoints, ``None`` while nothing is loaded — decides which controls
        the sidebar shows for the video."""
        prs = [pr for pr in (self._primary_pr, *self._extra_pr.values()) if pr is not None]
        if not prs:
            return None
        return "bboxes" if all(pr.bbox_data is not None for pr in prs) else "poses"

    def clear_pose_display(self) -> None:
        """Remove pose overlays from all camera views."""
        for view in self._all_views():
            view.clear_overlay()
        self._primary_pr = None
        self._extra_pr.clear()

    def apply_pose_style(self) -> None:
        text_visible = self._data_widget.pose_show_text_checkbox.isChecked()
        size = self._data_widget.pose_point_size_spin.value()
        text_size = self._data_widget.pose_text_size_spin.value()
        points_visible = self._points_visible()
        for view in self._all_views():
            overlay = view.overlay
            if overlay is None:
                continue
            overlay.set_point_size(size)
            overlay.set_points_visible(points_visible)
            overlay.set_text_size(text_size)
            overlay.set_text_visible(text_visible)
            view.request_draw()

    def apply_skeleton_style(self) -> None:
        """Update skeleton edge width on existing overlays in place.

        Width is uniform, so it can be set without rebuilding — avoiding a full
        ``update_pose()`` reload.
        """
        width = self._skeleton_width()
        for view in self._all_views():
            if view.overlay is not None:
                view.overlay.set_edge_width(width)
                view.request_draw()

    def has_active_skeleton_overlay(self) -> bool:
        """Whether a skeleton is currently drawn on at least one camera view."""
        if not self._skeleton_enabled():
            return False
        prs = [self._primary_pr, *self._extra_pr.values()]
        return any(pr is not None and self._resolved_skeleton_config(pr) for pr in prs)

    def refresh_skeleton(self) -> None:
        """Rebuild overlays with the current skeleton config (colours changed).

        Reuses the last-rendered ``PoseRenderData`` so no pose reload happens.
        """
        combo = getattr(self._data_widget, "primary_camera_combo", None)
        primary_name = combo.currentText() if combo is not None else None
        if self._primary_pr is not None and primary_name is not None:
            self._display_pose_on_view(self.video_area.primary, self._primary_pr, primary_name)
        for key, view in self.video_mgr.extra_widgets.items():
            camera_name = getattr(view, "camera_name", key)
            if getattr(view, "static_image_path", None):
                # Image views mirror the primary camera's pose.
                if self._primary_pr is not None:
                    self._display_pose_on_view(view, self._primary_pr, None)
                continue
            pr = self._extra_pr.get(camera_name)
            if pr is not None:
                self._display_pose_on_view(view, pr, camera_name)

    def on_rotate_video_pose(self) -> None:
        """Rotate all camera views by 90° (camera-space rotation)."""
        self._rotation_count = (getattr(self, "_rotation_count", 0) + 1) % 4
        theta = np.radians(self._rotation_count * 90)
        for view in self._all_views():
            plot = getattr(view, "plot", None) or getattr(view, "_static", None)
            if plot is None:
                continue
            plot.camera.local.euler_z = theta
            view.request_draw()
