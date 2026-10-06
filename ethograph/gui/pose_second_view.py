"""A second camera in the keypoint labelling dialog, and the 3D point two views make.

The label model stays one video per store: the second camera gets a
:class:`~ethograph.gui.pose_annotate.KeypointStore`, a sidecar and a
:class:`~ethograph.gui.pose_edit_mixin.KeypointLabelMode` of its own, on its
own open camera view. What the two share is the moment (the second view's
frame is the one showing when the first shows its own), the schema, and the
active keypoint — a click lands in whichever view it was made in, and
Sequential moves on once both views have the point.

With a calibration for the two cameras, every keypoint both views carry on
the current frame is triangulated (:mod:`ethograph.triangulate`): drawn in
the open Space plots, and projected back into each view as a ``+`` whose
distance from the label is the reprojection error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygfx as gfx
from qtpy.QtCore import QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QPushButton, QVBoxLayout

from ethograph.gui.notify import notify
from ethograph.gui.plots_space import GEOMETRY_DIRNAME
from ethograph.gui.pose_annotate import DEFAULT_INDIVIDUAL, KeypointStore, KeypointStoreError, sidecar_path
from ethograph.gui.pose_edit_mixin import SEQUENTIAL_MODE, KeypointLabelMode, keypoint_colors_for
from ethograph.gui.pose_two_view import (
    ViewMismatchError,
    ViewSource,
    frame_points,
    pair_datasets,
    paired_frames,
    poses_3d_path,
    restore_fill,
    stash_fill,
    views_dataset_path,
)
from ethograph.gui.project import project_dir_of
from ethograph.gui.triangulation import calibration_path_of
from ethograph.io.netcdf import netcdf_engine
from ethograph.triangulate.calibration import CalibrationError, load_calibration
from ethograph.triangulate.frame import WorldFrame, WorldFrameError, WorldTransform, load_frame, save_frame
from ethograph.triangulate.geometry import skeleton_edges, write_geometry
from ethograph.triangulate.points import reproject_points, triangulate_points
from ethograph.triangulate.session import fit_world_frame

_OFFSCREEN = -1.0e6
#: Above the anchors, below the active ring's text.
_Z_REPROJECTION = 4.3
_NO_VIEW = "Off"
#: A pair whose 3D point lands further than this from a label is worth a second look.
REPROJECTION_WARN_PX = 5.0


class ReprojectionOverlay:
    """``+`` markers where the triangulated points fall back into one camera view."""

    def __init__(self, scene: gfx.Scene, n_points: int, img_height: float, size: float):
        self._scene = scene
        self._img_height = float(img_height)
        self._points = gfx.Points(
            gfx.Geometry(positions=np.full((max(n_points, 1), 3), _OFFSCREEN, dtype=np.float32)),
            gfx.PointsMarkerMaterial(
                size=size, size_space="screen", marker="plus", color="#ffffff", edge_width=1.0, edge_color="#000000"
            ),
        )
        self._points.local.z = _Z_REPROJECTION
        scene.add(self._points)

    @property
    def scene(self) -> gfx.Scene:
        return self._scene

    def set_points(self, points: np.ndarray) -> None:
        """Draw ``(n, 2)`` image-space points; ``NaN`` hides one."""
        buffer = self._points.geometry.positions
        shown = np.isfinite(points).all(axis=1)
        buffer.data[:] = _OFFSCREEN
        buffer.data[: len(points), 0][shown] = points[shown, 0]
        buffer.data[: len(points), 1][shown] = self._img_height - points[shown, 1]
        buffer.data[:, 2] = 0.0
        buffer.update_full()

    def detach(self) -> None:
        self._scene.remove(self._points)


class SecondView(QGroupBox):
    """The dialog's "Second view" group: which camera, what the pair triangulates to, and its exports."""

    def __init__(self, dialog):
        super().__init__("Second view (3D)")
        self._dialog = dialog
        self.app_state = dialog.app_state
        self.view = None
        self.store: KeypointStore | None = None
        self.mode: KeypointLabelMode | None = None
        self._video: str | None = None
        #: The pair both views are asked for next, and whether each already had it.
        self._joint: tuple[str | None, str | None] = (None, None)
        self._joint_had = [False, False]
        #: True while the last edit was made in this view — where Backspace and Ctrl+Z then act.
        self._has_pointer = False
        self._calibration: tuple[Path, float, tuple[str, str]] | None = None
        self._cgroup = None
        self._transform: WorldTransform | None = None
        self._overlays: list[ReprojectionOverlay] = []

        box = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip(
            "Label the same frame in a second open camera view. Each camera keeps its\n"
            "own labels file; with a calibration in the project folder, a keypoint\n"
            "labelled in both views becomes a 3D point in the Space plot.\n\n"
            "Open the other camera as a panel first — it is listed here once it shows."
        )
        self.camera_combo.activated.connect(self._on_camera_picked)
        row.addWidget(self.camera_combo, stretch=1)
        box.addLayout(row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        box.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.load_check = QCheckBox("Load 3D into the GUI")
        self.load_check.setChecked(True)
        self.load_check.setToolTip("After exporting, serve the session's features from the 3D poses.")
        buttons.addWidget(self.load_check)
        buttons.addStretch(1)
        self.export_btn = QPushButton("Export 2D + 3D poses")
        self.export_btn.setToolTip(
            "Write two movement-compatible datasets beside the first camera's video:\n"
            "every camera's 2D keypoints (one camera dimension), and the triangulated\n"
            "3D keypoints with their reprojection error."
        )
        self.export_btn.setAutoDefault(False)
        self.export_btn.clicked.connect(self._on_export)
        buttons.addWidget(self.export_btn)
        self.geometry_btn = QPushButton("Export geometry…")
        self.geometry_btn.setToolTip(
            "Write the static keypoints' 3D positions — the corners of the arena — as\n"
            "a reference geometry in the project's space/ folder, joined by the\n"
            "skeleton's connections between them, and show it in the Space plot."
        )
        self.geometry_btn.setAutoDefault(False)
        self.geometry_btn.clicked.connect(self._on_export_geometry)
        buttons.addWidget(self.geometry_btn)
        box.addLayout(buttons)

        area = dialog._shell.video_area
        for name in ("camera_added", "camera_view_removed"):
            signal = getattr(area, name, None)
            if signal is not None:
                signal.connect(self._refresh_cameras_later)
        self.refresh_cameras()

    # ------------------------------------------------------------------
    # Which camera
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.view is not None and self.store is not None

    def _extras(self) -> dict:
        return dict(getattr(self._dialog._shell.video_area, "extras", {}) or {})

    def _refresh_cameras_later(self, *_args) -> None:
        QTimer.singleShot(0, self.refresh_cameras)

    def refresh_cameras(self) -> None:
        extras = self._extras()
        current = next((key for key, view in extras.items() if view is self.view), None)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem(_NO_VIEW, None)
        for key in extras:
            self.camera_combo.addItem(str(key), key)
        self.camera_combo.setCurrentIndex(max(self.camera_combo.findData(current), 0))
        self.camera_combo.blockSignals(False)
        if self.view is not None and current is None:
            self.deactivate(save=True)
        self._refresh_status()

    def _on_camera_picked(self, index: int) -> None:
        key = self.camera_combo.itemData(index)
        view = self._extras().get(key) if key is not None else None
        if view is self.view:
            return
        self.deactivate(save=True)
        if view is not None:
            self.activate(view)

    def activate(self, view) -> None:
        if view.scene() is None or not self._video_of(view):
            notify("That camera has no video loaded for this trial.", "warning")
            self.refresh_cameras()
            return
        if self._video_of(view) == self._dialog._video_path():
            notify("That panel shows the video you are already labelling — pick another camera.", "warning")
            self.refresh_cameras()
            return
        self.view = view
        self._video = self._video_of(view)
        self.store = self._load_store()
        # Asked once, here: with several calibrations in the project, picking a second view is the
        # moment the user says they want 3D.
        calibration_path_of(self.app_state, self, ask=True)
        if self._dialog._mode is not None:
            self.attach(self._dialog._mode.mode)
        self.update_3d()

    def deactivate(self, save: bool) -> None:
        if save:
            self.save()
        self.detach()
        self._clear_overlays()
        self._set_live_points(None)
        if self.store is not None:
            stash_fill(self._dialog._fill_stash, self._video, self.store)
        self.view, self.store, self._video = None, None, None
        self._has_pointer = False
        self._refresh_status()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    @staticmethod
    def _video_of(view) -> str | None:
        return getattr(view, "source_video_path", None)

    def _load_store(self) -> KeypointStore:
        primary = self._dialog.store
        n_frames = int(self.view.n_frames or 0)
        path = sidecar_path(self._video)
        store = None
        if path.exists():
            try:
                store = KeypointStore.load(path)
            except (KeypointStoreError, ValueError, KeyError, OSError) as e:
                notify(f"Could not read {path.name}: {e}. Starting the second view with no labels.", "warning")
        if store is None:
            store = KeypointStore(
                keypoint_names=list(primary.keypoint_names),
                n_frames=n_frames,
                individual_names=list(primary.individual_names or [DEFAULT_INDIVIDUAL]),
            )
        store.n_frames = n_frames or store.n_frames
        self._match_schema(store)
        restore_fill(self._dialog._fill_stash, self._video, store)
        return store

    def _match_schema(self, store: KeypointStore) -> None:
        """One schema for both views: the first view's, since that is the one the dialog edits."""
        primary = self._dialog.store
        if store.individual_names != primary.individual_names:
            store.set_individual_names(list(primary.individual_names))
        if store.keypoint_names != primary.keypoint_names:
            store.set_keypoint_names(list(primary.keypoint_names))
        for name in primary.keypoint_names:
            if store.is_static(name) != primary.is_static(name):
                store.set_static(name, primary.is_static(name))
        store.keypoint_color = dict(primary.keypoint_color)
        store.individual_color = dict(primary.individual_color)

    def save(self) -> None:
        if self.active and self._video:
            self.store.save(sidecar_path(self._video))

    def on_video_changed(self) -> None:
        """Another trial: the second camera shows another clip, with a sidecar of its own."""
        QTimer.singleShot(0, self._follow_video)

    def _follow_video(self) -> None:
        if not self.active:
            return
        video = self._video_of(self.view)
        if video == self._dialog._video_path():
            # The cameras were swapped: this video is the first view's now, and one file has one writer.
            self.deactivate(save=False)
            self.refresh_cameras()
            return
        if video == self._video:
            return
        if self._video:
            self.store.save(sidecar_path(self._video))
            stash_fill(self._dialog._fill_stash, self._video, self.store)
        if not video:
            self.deactivate(save=False)
            self.refresh_cameras()
            return
        self._video = video
        self.store = self._load_store()
        self._clear_overlays()
        if self.mode is not None:
            self.mode.store = self.store
            self.mode.set_frame(self._frame_for(self._dialog._current_frame()))
        self.update_3d()

    # ------------------------------------------------------------------
    # The canvas mode
    # ------------------------------------------------------------------

    def attach(self, mode: str) -> None:
        if not self.active or self.view.scene() is None:
            return
        self.detach()
        self._match_schema(self.store)
        self.mode = KeypointLabelMode(
            self.view,
            self.store,
            on_changed=self._on_changed,
            mode=mode,
            on_advance_frame=self._dialog._advance_after_click,
            point_size=float(self.app_state.labelling_point_size),
            on_released=self._on_changed,
            locked=self._dialog._lock_wanted(),
            color_by=self._dialog.color_by,
        )
        self.on_frame_changed(self._dialog._current_frame())
        self.mirror_active()

    def detach(self) -> None:
        if self.mode is not None:
            self.mode.detach()
            self.mode = None

    def set_mode(self, mode: str) -> None:
        if self.mode is not None:
            self.mode.set_mode(mode)

    def set_locked(self, locked: bool) -> None:
        if self.mode is not None:
            self.mode.set_locked(locked or self.mode.frame < 0)

    def set_point_size(self, size: float) -> None:
        if self.mode is not None:
            self.mode.set_point_size(size)

    def recolor(self, color_by: str | None = None) -> None:
        if self.store is not None:
            self.store.keypoint_color = dict(self._dialog.store.keypoint_color)
            self.store.individual_color = dict(self._dialog.store.individual_color)
        if self.mode is None:
            return
        if color_by is not None:
            self.mode.set_color_by(color_by)
        self.mode.refresh_colors()

    def pointer_mode(self) -> KeypointLabelMode | None:
        """This view's mode when the last edit was made here, else ``None``."""
        return self.mode if self._has_pointer else None

    def undo(self) -> None:
        self.store.undo()
        self.mode.refresh()
        self.update_3d()

    # ------------------------------------------------------------------
    # One moment, one active keypoint
    # ------------------------------------------------------------------

    def _source(self, primary: bool) -> ViewSource:
        dialog = self._dialog
        view = dialog._view if primary else self.view
        camera = (view.camera_name or self.app_state.primary_camera) if primary else view.camera_name
        return ViewSource(
            camera=str(camera),
            store=dialog.store if primary else self.store,
            fps=float(dialog._fps() if primary else view.fps),
            time_offset=float(view.time_offset or 0.0),
            scale=float(view.overlay_scale()),
        )

    def _frame_for(self, frame: int) -> int:
        """This view's frame at the moment the first view shows *frame*; ``-1`` outside its clip."""
        if not self._dialog._fps() or not self.view.fps:
            return int(frame)
        return int(paired_frames(np.array([frame]), self._source(True), self._source(False))[0])

    def on_frame_changed(self, frame: int) -> None:
        if not self.active:
            return
        if self.mode is not None:
            own = self._frame_for(frame)
            self.mode.set_frame(own)
            # Outside the second camera's clip there is no frame to put a label on.
            self.mode.set_locked(self._dialog._lock_wanted() or own < 0)
        self.update_3d()

    def _has(self, mode: KeypointLabelMode, individual: str, keypoint: str) -> bool:
        if mode.frame < 0:
            return True
        placed = mode.store.anchor_positions_for(mode.frame, individual)
        return not np.isnan(placed[mode.store.keypoint_index(keypoint), 0])

    def _set_joint(self, individual: str | None, keypoint: str | None) -> None:
        modes = (self._dialog._mode, self.mode)
        for mode in modes:
            if keypoint is not None and (mode.active_individual, mode.active_keypoint) != (individual, keypoint):
                mode.set_active(keypoint, individual)
        self._joint = (individual, keypoint)
        self._joint_had = [
            keypoint is not None and individual is not None and self._has(mode, individual, keypoint) for mode in modes
        ]

    def mirror_active(self) -> None:
        """The second view asks for whatever the first view asks for."""
        primary = self._dialog._mode
        if self.mode is None or primary is None:
            return
        pair = (primary.active_individual, primary.active_keypoint)
        if pair != self._joint or pair != (self.mode.active_individual, self.mode.active_keypoint):
            self._set_joint(*pair)

    def _follow(self, source: KeypointLabelMode, is_second: bool) -> None:
        """After an edit in *source*: stay on the pair until both views have it, then move on.

        A click that *placed* the pair both views were asked for leaves it
        active until the other view has it too, then moves to the next
        keypoint either view lacks — forward from here, so a keypoint skipped
        with ``Tab`` because one camera cannot see it is not asked for again
        until the rest are done. Any other edit (grabbing another point,
        deleting) makes that point the pair.
        """
        primary = self._dialog._mode
        individual, keypoint = self._joint
        placed = (
            keypoint is not None
            and individual is not None
            and not self._joint_had[int(is_second)]
            and self._has(source, individual, keypoint)
        )
        if source.mode != SEQUENTIAL_MODE or not placed:
            self._set_joint(source.active_individual, source.active_keypoint)
            return
        names = sorted(primary.store.keypoints_for(individual), key=primary.store.keypoint_index)
        start = names.index(keypoint)
        for name in names[start:] + names[:start]:
            if not (self._has(primary, individual, name) and self._has(self.mode, individual, name)):
                self._set_joint(individual, name)
                return
        self._set_joint(individual, keypoint)

    def after_primary_changed(self) -> None:
        """The first view's store changed — by a click there, or anything else the dialog does."""
        self._has_pointer = False
        primary = self._dialog._mode
        if self.mode is None or primary is None:
            if self.active:
                self.update_3d()
            return
        self._follow(primary, is_second=False)
        if not primary.dragging:
            self.update_3d()

    def _on_changed(self, *_args) -> None:
        """An edit in the second view."""
        self._has_pointer = True
        if self._dialog._mode is not None:
            self._follow(self.mode, is_second=True)
        self._dialog._sync_tree_selection()
        self._dialog._refresh_active_label()
        if not self.mode.dragging:
            self.update_3d()

    def pair_complete(self) -> bool:
        """Whether both views have the active pair on this moment — Loop only moves on then."""
        primary = self._dialog._mode
        if self.mode is None or primary is None:
            return True
        individual, keypoint = primary.active_individual, primary.active_keypoint
        if individual is None or keypoint is None:
            return True
        return self._has(primary, individual, keypoint) and self._has(self.mode, individual, keypoint)

    # ------------------------------------------------------------------
    # Calibration and the live 3D points
    # ------------------------------------------------------------------

    def _views(self) -> list[ViewSource]:
        return [self._source(True), self._source(False)]

    def _load_calibration(self) -> str | None:
        """Make ``_cgroup`` current for the two cameras; the reason there is none otherwise."""
        path = calibration_path_of(self.app_state)
        if path is None:
            self._calibration, self._cgroup, self._transform = None, None, None
            if project_dir_of(self.app_state) is None:
                return "no project folder to hold a calibration"
            return "the project folder has no calibration (or several, none chosen)"
        cameras = (self._source(True).camera, self._source(False).camera)
        frame_file = path.with_suffix(".frame.yaml")
        stamp = path.stat().st_mtime + (frame_file.stat().st_mtime if frame_file.is_file() else 0.0)
        key = (path, stamp, cameras)
        if key != self._calibration:
            self._calibration, self._cgroup, self._transform = None, None, None
            try:
                self._cgroup = load_calibration(path, list(cameras))
                frame = load_frame(path)
            except (CalibrationError, WorldFrameError) as err:
                return str(err)
            self._transform = None if frame is None else frame.transform
            self._calibration = key
        return None

    def update_3d(self) -> None:
        """Triangulate what both views carry on this moment; draw it in the Space plots and back in the views."""
        if not self.active:
            return
        reason = self._load_calibration()
        if reason is not None:
            self._clear_overlays()
            self._set_live_points(None)
            self._refresh_status(f"2D only — {reason}.")
            return
        views = self._views()
        try:
            points = frame_points(views, self._dialog._current_frame())
        except ViewMismatchError as err:
            self._refresh_status(str(err))
            return
        p3d, error = triangulate_points(self._cgroup, points)
        seen = np.isfinite(p3d[..., 0])

        self._draw_reprojections(reproject_points(self._cgroup, p3d), views)
        colors = keypoint_colors_for(self._dialog.store)
        shown = self._transform.apply(p3d) if self._transform is not None else p3d
        hexes = [QColor.fromRgbF(*[float(v) for v in colors[k][:3]]).name() for _, k in np.argwhere(seen)]
        self._set_live_points(shown[seen], hexes)

        if not seen.any():
            self._refresh_status("3D ready — label a keypoint in both views to see it in the Space plot.")
            return
        names = self._dialog.store.keypoint_names
        worst_ind, worst_kp = np.unravel_index(np.nanargmax(error), error.shape)
        worst = float(error[worst_ind, worst_kp])
        text = (
            f"3D: {int(seen.sum())} point(s) on this frame · reprojection error "
            f"median {float(np.nanmedian(error)):.1f} px, worst {names[worst_kp]} {worst:.1f} px"
        )
        if worst > REPROJECTION_WARN_PX:
            text = f'<span style="color:#e0a030;">{text} — check that label in both views</span>'
        self._refresh_status(text)

    def _draw_reprojections(self, pixels: np.ndarray, views: list[ViewSource]) -> None:
        canvases = (self._dialog._view, self.view)
        scenes = [canvas.scene() for canvas in canvases]
        n_points = pixels.shape[1] * pixels.shape[2]
        if [overlay.scene for overlay in self._overlays] != scenes or any(scene is None for scene in scenes):
            self._clear_overlays()
            if any(scene is None for scene in scenes):
                return
            size = 0.8 * float(self.app_state.labelling_point_size)
            self._overlays = [
                ReprojectionOverlay(scene, n_points, canvas.image_height(), size)
                for scene, canvas in zip(scenes, canvases)
            ]
        for overlay, view_pixels, view, canvas in zip(self._overlays, pixels, views, canvases):
            overlay.set_points(view_pixels.reshape(-1, 2) * view.scale)
            canvas.request_draw()

    def _clear_overlays(self) -> None:
        for overlay in self._overlays:
            overlay.detach()
        self._overlays = []

    def _set_live_points(self, points: np.ndarray | None, colors: list[str] | None = None) -> None:
        for plot in getattr(self._dialog._data_widget, "space_plots", []):
            plot.set_live_points(points, colors)

    def _refresh_status(self, text: str | None = None) -> None:
        if not self.active:
            text = (
                "Pick an open camera view to label the same frames in two views."
                if self._extras()
                else "Open a second camera panel to label the same frames in two views."
            )
        self.status_label.setText(text or "")
        self.export_btn.setEnabled(self.active)
        self.geometry_btn.setEnabled(self.active and self._cgroup is not None)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _fit_frame(self, path: Path, raw: np.ndarray) -> bool:
        """Resolve the rig's world frame from this clip's landmarks the first time it can be."""
        frame = load_frame(path)
        if frame is None or frame.transform is not None:
            return False
        try:
            transform = fit_world_frame(frame, [raw], list(self._dialog.store.keypoint_names))
        except WorldFrameError as err:
            notify(f"World frame not applied: {err}", "warning")
            return False
        save_frame(path, WorldFrame(frame.axes, frame.reference_point, frame.scale, frame.unit, transform))
        self._load_calibration()
        return True

    def _build(self):
        """``(2D dataset, 3D dataset or None)`` for the clip, or ``None`` with the reason notified."""
        self.save()
        reason = self._load_calibration()
        views = self._views()
        attrs = {} if self._calibration is None else {"calibration": self._calibration[0].stem}
        try:
            ds_2d, ds_3d, raw = pair_datasets(views, self._cgroup, self._transform, attrs)
            if ds_3d is not None and self._transform is None and self._fit_frame(self._calibration[0], raw):
                ds_2d, ds_3d, _ = pair_datasets(views, self._cgroup, self._transform, attrs)
        except (ViewMismatchError, KeypointStoreError, ValueError) as err:
            notify(f"Could not build the poses datasets: {err}", "error")
            return None
        if ds_3d is None:
            notify(f"Exporting 2D only — {reason}.", "warning")
        elif self._transform is not None:
            frame = load_frame(self._calibration[0])
            if frame is not None and frame.unit:
                ds_3d.attrs["space_unit"] = frame.unit
        return ds_2d, ds_3d

    def _on_export(self) -> None:
        video = self._dialog._video_path()
        if not self.active or not video:
            return
        built = self._build()
        if built is None:
            return
        ds_2d, ds_3d = built
        written = []
        for ds, path in ((ds_2d, views_dataset_path(video)), (ds_3d, poses_3d_path(video))):
            if ds is None:
                continue
            try:
                ds.to_netcdf(path, engine=netcdf_engine(path))
            except OSError as err:
                notify(f"Could not write {path.name}: {err}", "error")
                return
            written.append(path.name)
        notify(f"Wrote {' and '.join(written)} beside the video.", "info")
        if ds_3d is not None and self.load_check.isChecked():
            self._dialog._data_widget.load_keypoint_dataset(ds_3d)

    def _on_export_geometry(self) -> None:
        project = project_dir_of(self.app_state)
        statics = list(self._dialog.store.static_keypoints)
        if project is None or not statics:
            notify("Tick Static on the keypoints that make up the room, and label them in both views.", "warning")
            return
        points = frame_points(self._views(), self._dialog._current_frame())
        p3d, _ = triangulate_points(self._cgroup, points)
        if self._transform is not None:
            p3d = self._transform.apply(p3d)
        names = self._dialog.store.keypoint_names
        found = {}
        for name in statics:
            seen = p3d[:, names.index(name)]
            seen = seen[np.isfinite(seen).all(axis=1)]
            if len(seen):
                found[name] = np.median(seen, axis=0)
        missing = [name for name in statics if name not in found]
        if not found:
            notify("No static keypoint is labelled in both views yet.", "warning")
            return
        name, ok = QInputDialog.getText(self, "Export geometry", "Name of the geometry:", text="arena")
        if not ok or not name.strip():
            return
        edges = skeleton_edges(project, self.app_state.skeleton_name)
        path = write_geometry(Path(project) / GEOMETRY_DIRNAME / f"{name.strip()}.yaml", found, edges)
        self.app_state.space_library_geometry = path.stem
        skipped = f" ({', '.join(missing)} not in both views, left out)" if missing else ""
        notify(f"Wrote {path.name} to the project's space folder{skipped}.", "info")

    def close_view(self) -> None:
        self.deactivate(save=True)
