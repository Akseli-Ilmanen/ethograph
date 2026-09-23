"""Refine imported poses: correct existing pose files on the video, per trial.

Where the keypoint labelling dialog starts from nothing, this one starts from
pose files another tool produced (DeepLabCut, SLEAP, …): with a dataset
loaded, each **(trial, camera)** resolves its pose file through the alignment
exactly like the pose overlay does, the file's points appear on the canvas as
observations to correct, and the result is written back **in the source
format** as ``{stem}_refined{ext}`` beside the original. Multi-trial by
construction — the dialog follows the normal trial navigation, flushing every
refined file on the switch.

:class:`PoseRefinementDialog` **is** the labelling dialog with the schema and
Detect tabs removed: the whole Label & Edit tab — modes, the
points table, frame suggestions, ``Shift+H`` approval, the key handling — is
inherited rather than rebuilt, because correcting a file is the same work as
correcting a fill. What the subclass changes is where the store comes from and
where it goes: no keypoints are defined (the file's schema *is* the schema),
and there is no export page — saving is automatic (see below).

Cameras are **live contexts, not a reload**: every open camera view gets a
:class:`_CameraContext` holding its own file, store and dirty state, all kept
in memory for the whole trial. The Camera combo only chooses which context the
canvas edits — switching never discards anything, so labels cannot be lost to
a switch. **Fill and Save act on every open camera**, not the active one: the
same stretch is usually wrong on both views, and saving half the cameras is
how refined and unrefined files drift apart.

Provenance maps onto the keypoint store unchanged: the file's points are
**detections** (a DLC point is exactly that — read off one frame's pixels by a
machine), the user's clicks are **anchors**, and a fill is inference bridging
between observations. Each store lives on its view's **trial-local frame
grid**; the file's full arrays are kept aside and the edited window is merged
back into them at save time, so a session-wide pose file round-trips intact.

The Fill group gains a scope choice: **"my labels only"** interpolates between
the user's clicks alone and replaces the file's points inside the filled span
(for stretches where the file is plain wrong), while **"my labels + the
file's points"** treats the file as trusted observations and only bridges its
gaps. A camera with no clicks of its own is skipped by the first scope —
there is nothing to interpolate between.

Saving is not an export step. A camera's refined copy is created the moment it
is first edited, then rewritten on trial switch, the Save button and close —
an untouched camera writes nothing, so the output folder records exactly what
was reviewed. The user's own clicks are additionally kept in a
``{pose file}.refine.json`` sidecar (anchor format, on the file's own frame
grid) — and the sidecar is written **before** every refined-file write, so a
failed write (a locked file, a full disk) can never take the labels with it.
Without the sidecar, reopening would show every previous correction as
machine output again.

The Fill and save tab serves **two purposes**, chosen at its top. *Downstream
analysis* is the above: the refined file, every frame. *Pose-estimation
training* exports only the frames the user clicked on, as DeepLabCut
``CollectedData`` or COCO labels added to a folder that may already hold some
(:mod:`~ethograph.gui.pose_training_export`). Under that purpose the fill is
marked not recommended and asks before running: a filled point is an
interpolation, never exported, and a detector must not learn it as truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from movement.io import load_dataset as load_movement_dataset
from movement.io import save_poses
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_busy_progress import BusyProgressDialog
from ethograph.gui.dialog_pose_labelling import MAX_SIDE, PoseLabellingDialog
from ethograph.gui.file_dialogs import browse_open_dir
from ethograph.gui.notify import notify
from ethograph.gui.pose_annotate import (
    KeypointStore,
    KeypointStoreError,
    store_to_movement_ds,
)
from ethograph.gui.pose_fill import VideoFrameSource, build_backend
from ethograph.gui.pose_render import PoseRenderData, ask_pose_source_software
from ethograph.gui.pose_training_export import (
    FORMAT_COCO,
    FORMAT_DLC,
    ExportOutcome,
    dlc_project_settings,
    export_coco,
    export_dlc,
    training_frames,
)
from ethograph.io.netcdf import netcdf_engine

#: Suffix inserted before the extension of the written copy.
REFINED_SUFFIX = "_refined"

#: Sidecar holding the user's own clicks (anchor format) beside the pose file.
REFINE_SIDECAR_SUFFIX = ".refine.json"

#: The two fill scopes; see the module docstring.
SCOPE_MY_LABELS = "my_labels"
SCOPE_WITH_FILE = "with_file"

#: What the Fill and save tab is for — the refined file (analysis) or the
#: reviewed frames as a trainer's labels (training). See pose_training_export.
PURPOSE_ANALYSIS = "analysis"
PURPOSE_TRAINING = "training"

FILL_TITLE = "Fill"
FILL_TITLE_TRAINING = "Fill — not recommended for training data"


# ----------------------------------------------------------------------
# Pure helpers — the engine, testable without Qt
# ----------------------------------------------------------------------


def refined_pose_path(source: str | Path) -> Path:
    """Where the refined copy of *source* is written.

    Same format as the source, except ``.slp``: movement has no SLEAP project
    writer, only the analysis ``.h5``, so a ``.slp`` source refines to ``.h5``.
    """
    source = Path(source)
    suffix = ".h5" if source.suffix.lower() == ".slp" else source.suffix
    return source.with_name(f"{source.stem}{REFINED_SUFFIX}{suffix}")


def refine_sidecar_path(source: str | Path) -> Path:
    """Where the user's clicks for *source* are kept (``{file}.refine.json``)."""
    source = Path(source)
    return source.with_name(source.name + REFINE_SIDECAR_SUFFIX)


def pose_ds_to_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Unpack a movement poses dataset into store-shaped arrays.

    Returns ``(positions, confidence, keypoints, individuals)`` with positions
    ``(n_frames, n_individuals, n_keypoints, 2)`` and confidence
    ``(n_frames, n_individuals, n_keypoints)`` — the store's own axis order.
    A missing ``confidence`` variable becomes 1.0 wherever the position is
    finite, which is what a format without likelihoods means.
    """
    position = ds["position"].transpose("time", "individual", "keypoint", "space").values.astype(np.float64)
    if "confidence" in ds:
        confidence = ds["confidence"].transpose("time", "individual", "keypoint").values.astype(np.float64)
    else:
        confidence = np.where(np.isfinite(position[..., 0]), 1.0, np.nan)
    keypoints = [str(k) for k in ds.coords["keypoint"].values]
    individuals = [str(i) for i in ds.coords["individual"].values]
    return position, confidence, keypoints, individuals


def detections_from_file(
    positions: np.ndarray, confidence: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """File points as the store's sparse detection dicts, skipping empty frames."""
    frames = np.flatnonzero(np.isfinite(positions[..., 0]).any(axis=(1, 2)))
    pos = {int(f): positions[f].copy() for f in frames}
    conf = {int(f): np.nan_to_num(confidence[f], nan=0.0) for f in frames}
    return pos, conf


def outside_span(detections: dict[int, np.ndarray], span: tuple[int, int] | None) -> dict[int, np.ndarray]:
    """The entries of *detections* whose frame lies outside *span* (inclusive)."""
    if span is None:
        return dict(detections)
    first, last = span
    return {frame: points for frame, points in detections.items() if not first <= frame <= last}


def fill_span(filled: np.ndarray) -> tuple[int, int] | None:
    """First and last frame a backend result covers, or ``None`` for none."""
    finite = np.isfinite(np.asarray(filled)[..., 0]).reshape(len(filled), -1).any(axis=1)
    if not finite.any():
        return None
    return int(finite.argmax()), int(len(finite) - 1 - finite[::-1].argmax())


def full_refined_ds(
    file_positions: np.ndarray,
    file_confidence: np.ndarray,
    store: KeypointStore,
    start_frame: int,
    fps: float,
) -> xr.Dataset:
    """The whole file with the store's edited window merged back in.

    The store lives on the trial-local grid starting at *start_frame* of the
    file; everything outside that window keeps the file's own values verbatim,
    so refining one trial of a session-wide file never touches the others.
    """
    window = store_to_movement_ds(store, fps)
    merged_pos = file_positions.copy()
    merged_conf = file_confidence.copy()
    usable = min(store.n_frames, max(0, len(file_positions) - start_frame))
    window_pos = window["position"].transpose("time", "individual", "keypoint", "space").values
    window_conf = window["confidence"].transpose("time", "individual", "keypoint").values
    merged_pos[start_frame : start_frame + usable] = window_pos[:usable]
    merged_conf[start_frame : start_frame + usable] = window_conf[:usable]

    n_frames = len(merged_pos)
    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                merged_pos.transpose(0, 3, 2, 1), dims=["time", "space", "keypoint", "individual"]
            ),
            "confidence": xr.DataArray(merged_conf.transpose(0, 2, 1), dims=["time", "keypoint", "individual"]),
        },
        coords={
            "time": np.arange(n_frames) / fps,
            "space": ["x", "y"],
            "keypoint": list(store.keypoint_names),
            "individual": list(store.individual_names),
        },
        attrs={"ds_type": "poses", "fps": float(fps), "source_software": "ethograph"},
    )


def shift_anchors(anchors: dict[int, np.ndarray], offset: int) -> dict[int, np.ndarray]:
    """Anchors re-keyed by ``frame + offset``; negative results are dropped."""
    return {frame + offset: points.copy() for frame, points in anchors.items() if frame + offset >= 0}


def save_refined_ds(ds: xr.Dataset, path: str | Path, source_software: str) -> None:
    """Write *ds* at *path* in the source's own format, overwriting freely.

    DLC sources round-trip through ``to_dlc_file`` (``split_individuals=False``
    so exactly one file comes out), LightningPose through ``to_lp_file``, SLEAP
    through the analysis ``.h5`` writer, and ``.nc`` stays NetCDF. Formats
    movement cannot write raise rather than silently switching format.

    movement's writers refuse an existing target, while a refined file is
    rewritten on every flush — so the write goes to a temp name (same suffix,
    the writers validate it) and replaces the target atomically. A failed
    write leaves the previous refined copy untouched.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    software = (source_software or "").lower()
    temp = path.with_name(f"{path.stem}.writing{path.suffix}")
    temp.unlink(missing_ok=True)
    try:
        if suffix == ".nc":
            ds.to_netcdf(temp, engine=netcdf_engine(temp))
        elif software == "lightningpose" and suffix == ".csv":
            save_poses.to_lp_file(ds, temp)
        elif software == "sleap" and suffix == ".h5":
            save_poses.to_sleap_analysis_file(ds, temp)
        elif suffix in (".csv", ".h5"):
            save_poses.to_dlc_file(ds, temp, split_individuals=False)
        else:
            raise KeypointStoreError(
                f"movement cannot write {suffix} for {source_software!r} — refine a .csv/.h5/.nc source."
            )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


#: What the primary pose overlay shows while a refinement mode is armed:
#: nothing — the anchor overlay draws the file's points with provenance, and
#: the ordinary overlay showing the same file would double every marker.
def _empty_render() -> PoseRenderData:
    return PoseRenderData(
        data=np.empty((0, 3)),
        properties=pd.DataFrame({"individual": [], "keypoint": [], "time": [], "confidence": []}),
        data_not_nan=np.empty(0, dtype=bool),
        file_name="pose refinement",
    )


@dataclass
class _CameraContext:
    """One camera's live refinement state, held for the whole trial.

    Contexts are what makes camera switching lossless: the store, the file's
    arrays and the dirty flag stay in memory, and the Camera combo only picks
    which context the canvas edits.
    """

    camera: str
    source_path: Path
    software: str
    fps: float
    #: File frame that the store's frame 0 maps to (the view's trial clip).
    window_start: int
    file_positions: np.ndarray
    file_confidence: np.ndarray
    store: KeypointStore
    #: The file's points inside the trial window, trial-local — what the
    #: store's detections are rebuilt from after a scope-a fill.
    window_detections: dict[int, np.ndarray] = field(default_factory=dict)
    window_confidence: dict[int, np.ndarray] = field(default_factory=dict)
    dirty: bool = False
    refined_written: bool = False


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------


class PoseRefinementDialog(PoseLabellingDialog):
    """The labelling dialog re-rooted on imported pose files, one per camera.

    Inherits the whole Label & Edit tab and reuses the fill backends; overrides
    where stores are loaded from (each open camera's pose file, via the
    alignment) and where they are saved to (``_refined`` copies + click
    sidecars). The schema and Detect tabs are removed — the files define the
    schema, and detection belongs to from-scratch labelling.
    """

    def __init__(self, data_widget, parent=None):
        super().__init__(data_widget, parent)
        self.setWindowTitle("Refine imported poses")
        self._graft_refinement_ui()
        self.app_state.trial_changed.connect(self._on_trial_switched)
        self._refresh_context_label()

    # ------------------------------------------------------------------
    # Contexts: one live per open camera
    # ------------------------------------------------------------------

    @property
    def _context(self) -> _CameraContext | None:
        return self._contexts.get(self._active_camera)

    def _load_store(self) -> KeypointStore:
        """The active camera's store — reusing its live context when one exists.

        Called by the parent ``__init__`` before the UI exists and again on
        every context reload. A camera with no resolvable file gets an empty
        store — the dialog stays open and the context label says why nothing
        can be labelled.
        """
        # Cross-context state, created once (this is the first subclass hook
        # the parent __init__ reaches).
        self._contexts: dict[str, _CameraContext] = getattr(self, "_contexts", {})
        self._loading = getattr(self, "_loading", False)
        self._reload_pending = getattr(self, "_reload_pending", False)
        self._active_camera = getattr(self, "_active_camera", None) or self.app_state.primary_camera

        view = self._camera_view_for(self._active_camera)
        if view is not None:
            # Everything the parent reads off the view — fps, frame window,
            # scene, image height — must come from the camera being refined.
            self._view = view
        context = self._contexts.get(self._active_camera)
        if context is None:
            context = self._build_context(self._active_camera, self._view)
            if context is not None:
                self._contexts[self._active_camera] = context
        if context is None:
            return KeypointStore(keypoint_names=[], n_frames=self._n_frames(), individual_names=[])
        return context.store

    def _build_context(self, camera: str | None, view) -> _CameraContext | None:
        source = self._resolve_source(camera)
        if source is None or not source.exists():
            return None
        software = self._resolve_software(source)
        if not software:
            return None
        fps = getattr(view, "fps", None) or self.app_state.video_fps
        if not fps:
            return None

        refined = refined_pose_path(source)
        load_path = refined if refined.exists() else source
        load_software = software
        if refined.exists() and refined.suffix != source.suffix:
            load_software = "SLEAP"  # .slp refined to an analysis .h5
        try:
            if load_path.suffix.lower() == ".nc":
                ds = xr.open_dataset(load_path, engine=netcdf_engine(load_path)).load()
            else:
                ds = load_movement_dataset(str(load_path), load_software, float(fps))
        except (OSError, ValueError, KeyError) as e:
            notify(f"Could not read {load_path.name}: {e}", "warning")
            return None

        positions, confidence, keypoints, individuals = pose_ds_to_arrays(ds)
        start = int(getattr(view, "start_frame", 0) or 0)
        n_frames = int(getattr(view, "n_frames", 0) or 0) or max(0, len(positions) - start)
        store = KeypointStore(keypoint_names=keypoints, n_frames=n_frames, individual_names=individuals)

        window_pos = positions[start : start + n_frames]
        window_conf = confidence[start : start + n_frames]
        window_detections, window_confidence = detections_from_file(window_pos, window_conf)
        store.set_detections(dict(window_detections), dict(window_confidence))

        sidecar = refine_sidecar_path(source)
        if sidecar.exists():
            try:
                previous = KeypointStore.load(sidecar)
            except (KeypointStoreError, ValueError, KeyError, OSError) as e:
                notify(f"Ignoring unreadable {sidecar.name}: {e}", "warning")
            else:
                if previous.keypoint_names == keypoints and previous.individual_names == individuals:
                    # Sidecar frames are on the file's grid; the store's are
                    # trial-local. A resumed session re-runs its fill rather
                    # than trusting a stale one; only observations carry over.
                    store.anchors = {
                        frame: points
                        for frame, points in shift_anchors(previous.anchors, -start).items()
                        if frame < n_frames
                    }

        return _CameraContext(
            camera=str(camera),
            source_path=source,
            software=software,
            fps=float(fps),
            window_start=start,
            file_positions=positions,
            file_confidence=confidence,
            window_detections=window_detections,
            window_confidence=window_confidence,
            store=store,
            refined_written=refined.exists(),
        )

    def _camera_view_for(self, name: str | None):
        """The open camera view showing *name*, primary or extra, or ``None``."""
        primary = self._shell.video_area.primary
        if name is None or getattr(primary, "camera_name", None) == name:
            return primary
        for key, view in getattr(self._shell.video_area, "extras", {}).items():
            if getattr(view, "camera_name", key) == name:
                return view
        return None

    def _open_cameras(self) -> list[str]:
        """Every camera an open view shows, the active one first."""
        names: list[str] = []
        area = self._shell.video_area
        views = [area.primary, *getattr(area, "extras", {}).values()]
        for view in views:
            name = getattr(view, "camera_name", None) or self.app_state.primary_camera
            if name and name not in names:
                names.append(name)
        if self._active_camera in names:
            names.remove(self._active_camera)
            names.insert(0, self._active_camera)
        return names

    def _ensure_open_contexts(self) -> list[_CameraContext]:
        """A live context for every open camera that resolves a pose file."""
        contexts: list[_CameraContext] = []
        for name in self._open_cameras():
            context = self._contexts.get(name)
            if context is None:
                view = self._camera_view_for(name)
                if view is None:
                    continue
                context = self._build_context(name, view)
                if context is None:
                    continue
                self._contexts[name] = context
            contexts.append(context)
        return contexts

    def _video_path(self) -> str | None:
        """The ACTIVE view's video — the fill must decode the camera it refines."""
        if self._view is not self._shell.video_area.primary:
            return getattr(self._view, "source_video_path", None)
        return super()._video_path()

    def _resolve_source(self, camera: str | None = None) -> Path | None:
        """One (trial, camera)'s pose file, via the alignment's chain."""
        sio = self.app_state.nwb_alignment
        cameras = sio.cameras
        if not cameras:
            return None
        name = camera if camera is not None else self._active_camera
        camera_idx = cameras.index(name) if name in cameras else 0
        # The camera's own name when it is a pose stream, else index pairing —
        # the same rule as PoseDisplayManager._pose_device.
        pose_keys = sio.pose_keys
        if cameras[camera_idx] in pose_keys:
            device = cameras[camera_idx]
        else:
            device = pose_keys[camera_idx] if camera_idx < len(pose_keys) else None
        path = sio.resolve_media_path(
            self.app_state.trials_sel,
            "pose",
            device=device,
            fallback_folder=self.app_state.pose_folder,
        )
        return Path(path) if path else None

    def _resolve_software(self, pose_path: Path) -> str | None:
        software = self.app_state.source_software or getattr(self.app_state.ds, "source_software", None)
        if not software:
            software = ask_pose_source_software(pose_path, parent=self)
            if software:
                self.app_state.source_software = software
        return software

    # ------------------------------------------------------------------
    # Saving: sidecar first, refined copies for every camera
    # ------------------------------------------------------------------

    def _save_store(self) -> None:
        """The active camera's click sidecar — never the video sidecar."""
        context = self._context
        if context is not None and context.store is self.store:
            self._save_context_sidecar(context)

    def _save_context_sidecar(self, context: _CameraContext) -> None:
        """The user's clicks, on the file's own grid — the loss-proof copy."""
        shifted = KeypointStore(
            keypoint_names=list(context.store.keypoint_names),
            n_frames=len(context.file_positions),
            individual_names=list(context.store.individual_names),
        )
        shifted.anchors = shift_anchors(context.store.anchors, context.window_start)
        try:
            shifted.save(refine_sidecar_path(context.source_path))
        except OSError as e:
            notify(f"Could not write {refine_sidecar_path(context.source_path).name}: {e}", "warning")

    def _flush_context(self, context: _CameraContext) -> bool:
        """Write one camera's sidecar + refined copy; the sidecar comes FIRST.

        The sidecar holds the user's clicks and always writes (plain JSON), so
        a refined-file failure — a locked file, a full disk — costs a warning
        and a retry, never the labels.
        """
        self._save_context_sidecar(context)
        refined = refined_pose_path(context.source_path)
        try:
            ds = full_refined_ds(
                context.file_positions,
                context.file_confidence,
                context.store,
                context.window_start,
                context.fps,
            )
            save_refined_ds(ds, refined, context.software)
        except (KeypointStoreError, OSError, ValueError) as e:
            notify(f"Could not write {refined.name}: {e} — your clicks are safe in the sidecar.", "error")
            return False
        context.refined_written = True
        context.dirty = False
        return True

    def _flush_all(self) -> None:
        """Every edited camera's refined copy — Save always means all of them."""
        for context in list(self._contexts.values()):
            if context.dirty:
                self._flush_context(context)
        self._refresh_context_label()

    # The detector cache belongs to from-scratch labelling: our "detections"
    # are the pose files themselves, and writing them into
    # <video>.detections.npz would hand them to the labelling dialog as a
    # detector run.
    def _load_detections(self) -> None:
        pass

    def _save_detections(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Dirty tracking
    # ------------------------------------------------------------------

    def _on_store_changed(self, full: bool = False, frame: int | None = None) -> None:
        super()._on_store_changed(full=full, frame=frame)
        if not self._loading:
            self._mark_dirty()

    def _mark_dirty(self) -> None:
        context = self._context
        if context is None:
            return
        context.dirty = True
        if not context.refined_written:
            # "The moment you start labelling, a _refined file gets created."
            self._flush_context(context)
        self._refresh_context_label()

    # ------------------------------------------------------------------
    # Fill: every open camera, not just the one on screen
    # ------------------------------------------------------------------

    def _on_fill(self) -> None:
        if self._purpose() == PURPOSE_TRAINING and not self._confirm_fill_for_training():
            return
        contexts = self._ensure_open_contexts()
        if not contexts:
            notify("No pose file resolves for any open camera — nothing to fill.", "warning")
            return
        scope = self.fill_scope_combo.currentData()
        key = self.backend_combo.currentData()
        label = self.backend_combo.currentText()

        busy = BusyProgressDialog(f"Filling frames with {label}…", parent=self)
        cancelled = False

        def progress_for(camera: str):
            def progress(fraction: float) -> bool:
                nonlocal cancelled
                busy.setLabelText(f"Filling {camera} with {label}… {fraction:.0%}")
                busy.pump_events()
                cancelled = cancelled or busy.wasCanceled()
                return not cancelled

            return progress

        filled_cameras: list[str] = []
        skipped: list[str] = []
        for context in contexts:
            if scope == SCOPE_MY_LABELS and not context.store.anchor_frames():
                skipped.append(context.camera)
                continue
            outcome, error = busy.execute(self._fill_context, context, scope, key, progress_for(context.camera))
            if cancelled:
                notify("Fill cancelled — remaining cameras were left alone.", "info")
                break
            if error is None and outcome:
                filled_cameras.append(context.camera)
                context.dirty = True

        if self._mode is not None:
            self._mode.refresh()
        self._push_pose_override()
        self._refresh_point_table(full=True)
        self._refresh_context_label()
        if filled_cameras:
            notify(f"Filled {', '.join(filled_cameras)}.", "info")
        if skipped:
            notify(
                f"Skipped {', '.join(skipped)} — no clicks of yours there to interpolate between.",
                "info",
            )

    def _fill_context(self, context: _CameraContext, scope: str, key: str, progress) -> bool:
        """Run one camera's fill; returns whether a fill was applied."""
        store = context.store
        if scope == SCOPE_MY_LABELS:
            # Observations = the user's anchors alone: the file's points must
            # not anchor the interpolation they are being corrected against.
            store.set_detections({}, {})
        try:
            backend = build_backend(
                key,
                checkpoint=self.app_state.labelling_cotracker_checkpoint or None,
                progress=progress,
                disagreement_px=float(self.app_state.labelling_disagreement_px),
                n_points=store.n_points,
            )
            frames = None
            if backend.requires_video:
                view = self._camera_view_for(context.camera)
                video = getattr(view, "source_video_path", None) or self.app_state.video_path
                if not video:
                    raise ValueError(f"No video for {context.camera} — this backend needs to read frames.")
                frames = VideoFrameSource(
                    video,
                    fps=context.fps,
                    n_frames=store.n_frames,
                    max_side=MAX_SIDE,
                    start_frame=context.window_start,
                )
            try:
                filled, confidence = backend.fill(store.flat_observations(), store.n_frames, frames, progress)
            finally:
                if frames is not None:
                    frames.close()
        except Exception:  # noqa: BLE001 - restore the store, then let execute() report it
            if scope == SCOPE_MY_LABELS:
                store.set_detections(dict(context.window_detections), dict(context.window_confidence))
            raise

        span = fill_span(filled)
        if scope == SCOPE_MY_LABELS:
            # The file's points come back OUTSIDE the filled span only: inside
            # it the fill is the correction the user asked for. Restored BEFORE
            # set_fill — set_detections discards any fill it finds.
            store.set_detections(
                outside_span(context.window_detections, span),
                outside_span(context.window_confidence, span),
            )
        if span is None:
            return False
        store.set_fill_from_flat(filled, confidence)
        return True

    # ------------------------------------------------------------------
    # Display: never draw a file twice
    # ------------------------------------------------------------------

    def _push_pose_override(self) -> None:
        """While armed, the ordinary overlay shows NOTHING rather than the file.

        The parent clears its override while a mode is armed because in
        from-scratch labelling there is no loaded pose underneath. Here there
        is — the very file being corrected — and the anchor overlay already
        draws every one of its points with provenance, so the ordinary overlay
        must show an empty render, not the file again.

        The override reaches the PRIMARY view only, so it is used exactly when
        the primary is the camera being refined; refining an extra camera
        leaves the primary's own pose untouched (each camera keeps drawing its
        own file — the multi-camera visualization). The one gap: an extra
        view's pose overlay cannot be suppressed, so while an extra camera is
        being edited its file points are drawn by both overlays.
        """
        pose_mgr = self._data_widget.pose_mgr
        if pose_mgr is None:
            return
        primary_active = self._view is self._shell.video_area.primary
        if self._mode is not None:
            if primary_active:
                pose_mgr.set_pose_override(_empty_render())
                self._override_pushed = True
            elif self._override_pushed:
                pose_mgr.set_pose_override(None)
                self._override_pushed = False
            self._data_widget.update_pose()
            return
        if not primary_active:
            if self._override_pushed:
                pose_mgr.set_pose_override(None)
                self._override_pushed = False
                self._data_widget.update_pose()
            return
        super()._push_pose_override()

    # ------------------------------------------------------------------
    # Following the session
    # ------------------------------------------------------------------

    def _on_trial_switched(self) -> None:
        # The trial's files are done with: flush everything and drop the
        # contexts — the next trial resolves its own files.
        self._schedule_context_reload(drop_contexts=True)

    def _on_video_changed(self, _path=None) -> None:
        super()._on_video_changed(_path)
        self._schedule_context_reload(drop_contexts=True)

    def _schedule_context_reload(self, drop_contexts: bool = False) -> None:
        """Flush now, reload after the data widget's own cascade settles."""
        if self._loading or self._reload_pending:
            return
        self._flush_all()
        if drop_contexts:
            self._contexts.clear()
        self._reload_pending = True
        QTimer.singleShot(0, self._reload_context)

    def _reload_context(self) -> None:
        self._reload_pending = False
        if not self.isVisible():
            return
        self._loading = True
        try:
            mode = self.interaction_mode
            self._detach_mode()
            view = self._camera_view_for(self._active_camera)
            if view is not None:
                self._view = view
            self.store = self._load_store()
            self._suggestions = []
            self._suggestion_index = 0
            self._rebuild_tree()
            self._refresh_point_table(full=True)
            self._refresh_target_combos()
            self._push_pose_override()
            if mode and self.tabs.currentWidget() is self._label_page and self._can_label(quiet=True):
                self.set_interaction_mode(mode)
        finally:
            self._loading = False
        self._refresh_context_label()

    def closeEvent(self, event):
        self._flush_all()
        pose_mgr = self._data_widget.pose_mgr
        if pose_mgr is not None:
            pose_mgr.set_pose_override(None)
        try:
            self.app_state.trial_changed.disconnect(self._on_trial_switched)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)
        self._data_widget.update_pose()

    # ------------------------------------------------------------------
    # UI surgery: strip the tabs that do not apply, graft cameras + saving
    # ------------------------------------------------------------------

    def _graft_refinement_ui(self) -> None:
        # The schema page must survive tab removal — the Keypoints tree on it
        # is what the target selection and key handling read.
        self._schema_page = self.tabs.widget(0)
        for page in (self._schema_page, self._detect_page):
            self.tabs.removeTab(self.tabs.indexOf(page))

        output_index = self.tabs.count() - 1
        output_page = self.tabs.widget(output_index)
        self.tabs.setTabText(output_index, "Fill and save")

        # The export group retires whole: saving is automatic, not a step.
        self.invert_y_check.parentWidget().hide()

        fill_group = self.backend_combo.parentWidget()
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Fill between:"))
        self.fill_scope_combo = QComboBox()
        self.fill_scope_combo.addItem("my labels only (replaces the file inside the span)", SCOPE_MY_LABELS)
        self.fill_scope_combo.addItem("my labels + the file's points", SCOPE_WITH_FILE)
        self.fill_scope_combo.setToolTip(
            "Applies to EVERY open camera, each from its own labels and file.\n\n"
            "my labels only: interpolate between YOUR clicks alone; inside the\n"
            "filled span the file's points are replaced by the fill (for\n"
            "stretches where the file is wrong), outside they are kept. A\n"
            "camera without clicks of yours is skipped.\n\n"
            "my labels + the file's points: the file's points are trusted\n"
            "observations too — the fill only bridges the gaps between them\n"
            "and your clicks, and never replaces a file point."
        )
        scope_row.addWidget(self.fill_scope_combo)
        scope_row.addStretch()
        fill_group.layout().addLayout(scope_row)

        self.save_group = QGroupBox("Save")
        save_box = QVBoxLayout(self.save_group)
        note = QLabel(
            "Saved automatically for EVERY edited camera: refined copies are created "
            "on the first edit and rewritten on every trial switch and on close."
        )
        note.setWordWrap(True)
        save_box.addWidget(note)
        save_row = QHBoxLayout()
        save_row.addStretch()
        save_btn = QPushButton("Save refined now")
        save_btn.setToolTip("Write every edited camera's _refined copy now.")
        save_btn.clicked.connect(self._flush_all)
        save_row.addWidget(save_btn)
        save_box.addLayout(save_row)
        self.training_group = self._build_training_group()

        output_layout = output_page.layout()
        output_layout.insertWidget(output_layout.count() - 1, self.save_group)
        output_layout.insertWidget(output_layout.count() - 1, self.training_group)
        output_layout.insertWidget(0, self._build_purpose_row())
        self._apply_purpose()

        self.context_label = QLabel("")
        self.context_label.setWordWrap(True)
        self.layout().insertWidget(0, self.context_label)

        # Any OPEN camera view can be refined, not only the primary — each
        # camera keeps its own live context, and the combo only picks which
        # one the canvas edits. Switching discards nothing.
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip(
            "Which camera's pose file the canvas edits. The camera needs an\n"
            "open view (add one from the panel popup) — clicks land on that\n"
            "view. Fill and Save always cover every open camera."
        )
        self.camera_combo.currentTextChanged.connect(self._on_camera_picked)
        camera_row.addWidget(self.camera_combo)
        camera_row.addStretch()
        self.layout().insertLayout(1, camera_row)
        self._refresh_camera_combo()

    # ------------------------------------------------------------------
    # Purpose: the refined file, or the reviewed frames as training labels
    # ------------------------------------------------------------------

    def _build_purpose_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Purpose:"))
        self.purpose_combo = QComboBox()
        self.purpose_combo.addItem("Export for downstream analysis (refined pose file)", PURPOSE_ANALYSIS)
        self.purpose_combo.addItem("Export for pose-estimation training (DeepLabCut / COCO)", PURPOSE_TRAINING)
        self.purpose_combo.setToolTip(
            "Downstream analysis: the corrected pose file, every frame — filled\n"
            "frames included, which is what a fill is for.\n\n"
            "Pose-estimation training: only the frames you clicked on, written as\n"
            "training labels for the next model. Filled points are never exported\n"
            "— an interpolation is a guess, not an observation — so filling is\n"
            "not recommended under this purpose."
        )
        position = self.purpose_combo.findData(self.app_state.pose_refine_purpose)
        self.purpose_combo.setCurrentIndex(position if position >= 0 else 0)
        self.purpose_combo.currentIndexChanged.connect(self._on_purpose_changed)
        layout.addWidget(self.purpose_combo, stretch=1)
        return row

    def _build_training_group(self) -> QGroupBox:
        group = QGroupBox("Export training labels")
        box = QVBoxLayout(group)
        note = QLabel(
            "Every frame you clicked on, for EVERY open camera: your points over the "
            "file's own, never a filled point. Added to a folder that already holds "
            "labels — existing frames are kept, the same frame is replaced."
        )
        note.setWordWrap(True)
        box.addWidget(note)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.training_format_combo = QComboBox()
        self.training_format_combo.addItem("DeepLabCut CollectedData (labeled-data/<video>/)", FORMAT_DLC)
        self.training_format_combo.addItem("COCO (images/ + annotations.json)", FORMAT_COCO)
        self.training_format_combo.setToolTip(
            "DeepLabCut: pick the project folder (the one with config.yaml) and\n"
            "the frames land under its labeled-data/<video>/ as img<frame>.png +\n"
            "CollectedData_<scorer>.csv/.h5 — the layout label_frames produces,\n"
            "so create_training_dataset picks them up. Any other folder gets\n"
            "<video>/ directly.\n\n"
            "COCO: one images/ folder and one annotations.json with a single\n"
            "category whose keypoints are the file's."
        )
        position = self.training_format_combo.findData(self.app_state.pose_training_export_format)
        self.training_format_combo.setCurrentIndex(position if position >= 0 else 0)
        self.training_format_combo.currentIndexChanged.connect(self._on_training_format_changed)
        format_row.addWidget(self.training_format_combo, stretch=1)
        box.addLayout(format_row)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Folder:"))
        self.training_dir_edit = QLineEdit(self.app_state.pose_training_export_dir or "")
        self.training_dir_edit.setPlaceholderText("An existing DeepLabCut project / COCO folder, or a new one")
        self.training_dir_edit.editingFinished.connect(self._on_training_dir_edited)
        folder_row.addWidget(self.training_dir_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_training_dir)
        folder_row.addWidget(browse_btn)
        box.addLayout(folder_row)

        self.scorer_row = QWidget()
        scorer_layout = QHBoxLayout(self.scorer_row)
        scorer_layout.setContentsMargins(0, 0, 0, 0)
        scorer_layout.addWidget(QLabel("Scorer:"))
        self.scorer_edit = QLineEdit(self.app_state.pose_training_scorer)
        self.scorer_edit.setToolTip(
            "DeepLabCut names the labels file after who labelled: CollectedData_<scorer>.\n"
            "Read from the project's config.yaml when the folder is a project; a\n"
            "folder already holding a CollectedData file dictates it."
        )
        self.scorer_edit.editingFinished.connect(self._on_scorer_edited)
        scorer_layout.addWidget(self.scorer_edit, stretch=1)
        box.addWidget(self.scorer_row)

        self.training_status = QLabel("")
        self.training_status.setWordWrap(True)
        box.addWidget(self.training_status)

        export_row = QHBoxLayout()
        export_row.addStretch()
        export_btn = QPushButton("Export labelled frames…")
        export_btn.setToolTip("Write the clicked frames of every open camera into the folder above.")
        export_btn.clicked.connect(self._on_export_training)
        export_row.addWidget(export_btn)
        box.addLayout(export_row)
        self._refresh_training_rows()
        return group

    def _purpose(self) -> str:
        return str(self.purpose_combo.currentData())

    def _on_purpose_changed(self, _index: int) -> None:
        self.app_state.pose_refine_purpose = self._purpose()
        self._apply_purpose()

    def _apply_purpose(self) -> None:
        training = self._purpose() == PURPOSE_TRAINING
        fill_group = self.backend_combo.parentWidget()
        fill_group.setTitle(FILL_TITLE_TRAINING if training else FILL_TITLE)
        self.save_group.setVisible(not training)
        self.training_group.setVisible(training)
        if training:
            self._refresh_training_status()

    def _confirm_fill_for_training(self) -> bool:
        answer = QMessageBox.warning(
            self,
            "Fill under the training purpose?",
            "Filled frames are interpolated between your clicks, not observed. They are "
            "never exported as training labels — a detector trained on them would learn "
            "the interpolation's mistakes as ground truth — so a fill buys nothing here "
            "and can hide which points you actually checked.\n\n"
            "Switch the purpose to downstream analysis if the refined file is what you "
            "want. Fill anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _on_training_format_changed(self, _index: int) -> None:
        self.app_state.pose_training_export_format = str(self.training_format_combo.currentData())
        self._refresh_training_rows()

    def _refresh_training_rows(self) -> None:
        self.scorer_row.setVisible(self.training_format_combo.currentData() == FORMAT_DLC)

    def _on_browse_training_dir(self) -> None:
        path = browse_open_dir(
            self,
            self.app_state,
            "Training labels folder (existing or new)",
            preferred_dir=self.training_dir_edit.text() or None,
        )
        if path:
            self.training_dir_edit.setText(path)
            self._on_training_dir_edited()

    def _on_training_dir_edited(self) -> None:
        text = self.training_dir_edit.text().strip()
        self.app_state.pose_training_export_dir = text or None
        if text:
            scorer, _ = dlc_project_settings(Path(text))
            if scorer:
                # The project knows who labels; a typed scorer is only for
                # folders that are not projects.
                self.scorer_edit.setText(scorer)
                self._on_scorer_edited()
        self._refresh_training_status()

    def _on_scorer_edited(self) -> None:
        self.app_state.pose_training_scorer = self.scorer_edit.text().strip()

    def _refresh_training_status(self) -> None:
        counts = [
            f"{context.camera}: {len(context.store.anchor_frames())}"
            for context in self._contexts.values()
            if context.store.anchor_frames()
        ]
        self.training_status.setText(
            "Frames with your clicks — " + ", ".join(counts) if counts else "No frame carries a click of yours yet."
        )

    def _on_export_training(self) -> None:
        target_text = self.training_dir_edit.text().strip()
        if not target_text:
            notify("Pick a folder for the training labels first.", "warning")
            return
        target = Path(target_text)
        fmt = str(self.training_format_combo.currentData())
        scorer = self.scorer_edit.text().strip()
        if fmt == FORMAT_DLC and not scorer:
            notify("DeepLabCut needs a scorer name — the CollectedData file is named after it.", "warning")
            return
        contexts = [c for c in self._ensure_open_contexts() if c.store.anchor_frames()]
        if not contexts:
            notify("No open camera carries a click of yours — nothing to export for training.", "warning")
            return
        # The sidecars first: an export is a review milestone, and the
        # refined copies are what the analysis purpose already keeps current.
        self._flush_all()

        busy = BusyProgressDialog("Exporting labelled frames…", parent=self)
        cancelled = False

        def progress_for(camera: str):
            def progress(fraction: float) -> bool:
                nonlocal cancelled
                busy.setLabelText(f"Exporting {camera}… {fraction:.0%}")
                busy.pump_events()
                cancelled = cancelled or busy.wasCanceled()
                return not cancelled

            return progress

        exported: list[str] = []
        for context in contexts:
            outcome, error = busy.execute(
                self._export_context, context, fmt, target, scorer, progress_for(context.camera)
            )
            if cancelled:
                notify("Export cancelled — remaining cameras were left alone.", "info")
                break
            if error is None and outcome is not None:
                exported.append(f"{context.camera} ({outcome.n_frames} frames)")
        if exported:
            notify(f"Exported {', '.join(exported)} → {target}", "info")

    def _export_context(self, context: _CameraContext, fmt: str, target: Path, scorer: str, progress) -> ExportOutcome:
        view = self._camera_view_for(context.camera)
        video = getattr(view, "source_video_path", None) or self.app_state.video_path
        if not video:
            raise ValueError(f"No video for {context.camera} — the frames must be read from it.")
        frames = training_frames(context.store, context.window_start)
        source = VideoFrameSource(video, fps=context.fps, n_frames=context.window_start + context.store.n_frames)
        # Image names are padded to the whole video's length, so one width
        # serves every trial cut from it.
        n_video_frames = source.video_frames or len(source)

        def decode(video_frame: int) -> np.ndarray:
            return source[video_frame]

        try:
            if fmt == FORMAT_DLC:
                return export_dlc(
                    frames, context.store, decode, target, Path(video).stem, scorer, n_video_frames, progress=progress
                )
            return export_coco(
                frames, context.store, decode, target, Path(video).stem, n_video_frames, progress=progress
            )
        finally:
            source.close()

    def _refresh_camera_combo(self) -> None:
        combo = getattr(self, "camera_combo", None)
        if combo is None:
            return
        cameras = list(self.app_state.nwb_alignment.cameras)
        current = [combo.itemText(i) for i in range(combo.count())]
        combo.blockSignals(True)
        try:
            if cameras != current:
                combo.clear()
                combo.addItems(cameras)
            if self._active_camera in cameras:
                combo.setCurrentText(self._active_camera)
        finally:
            combo.blockSignals(False)

    def _on_camera_picked(self, name: str) -> None:
        if self._loading or not name or name == self._active_camera:
            return
        if self._camera_view_for(name) is None:
            notify(f"No open view shows {name} — add its camera view first (➕ popup).", "warning")
            self._refresh_camera_combo()
            return
        context = self._context
        if context is not None:
            # The outgoing camera's context stays live — only its sidecar is
            # written, so even a crash mid-session costs nothing.
            self._save_context_sidecar(context)
        self._active_camera = name
        self._schedule_context_reload()

    def _refresh_context_label(self) -> None:
        label = getattr(self, "context_label", None)
        if label is None:
            return
        self._refresh_camera_combo()
        context = self._context
        if context is None:
            label.setText(
                "No pose file resolves for this trial and camera — nothing to refine. "
                "Pose streams come from the alignment (or the Pose folder setting)."
            )
            return
        refined = refined_pose_path(context.source_path)
        state = "unsaved edits" if context.dirty else ("saved" if context.refined_written else "no edits yet")
        others = [c.camera for c in self._contexts.values() if c is not context and c.dirty]
        suffix = f"   ·   also edited: {', '.join(others)}" if others else ""
        label.setText(
            f"{context.camera} · trial {self.app_state.trials_sel} — "
            f"{context.source_path.name} → {refined.name} ({state}){suffix}"
        )
