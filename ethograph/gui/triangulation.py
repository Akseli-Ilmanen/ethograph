"""The GUI's side of triangulation: which calibration, and the two Tools entries.

All geometry is :mod:`ethograph.triangulate`'s (aniposelib's); this module
only finds the project's calibration and runs the session-level call over the
live tree.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QInputDialog, QMessageBox, QWidget

from ethograph.gui.dialog_busy_progress import BusyProgressDialog
from ethograph.gui.file_dialogs import browse_open_dir
from ethograph.gui.notify import notify
from ethograph.gui.project import project_dir_of
from ethograph.triangulate.calibration import (
    CalibrationError,
    calibration_names,
    import_dlc_calibration,
    resolve_calibration,
)
from ethograph.triangulate.session import POSITION_3D, triangulate_tree


def calibration_path_of(app_state, parent: QWidget | None = None, *, ask: bool = False) -> Path | None:
    """The calibration this dataset was filmed with, or ``None`` when the project has none to offer.

    A project's only calibration needs no choosing. With several, the
    dataset's ``calibration_name`` decides; *ask* lets the user pick when it
    names none of them, and the answer is remembered for the dataset.
    """
    project = project_dir_of(app_state)
    if project is None:
        return None
    names = calibration_names(project)
    if not names:
        return None
    name = app_state.calibration_name if app_state.calibration_name in names else None
    if name is None and len(names) > 1:
        if not ask:
            return None
        name, ok = QInputDialog.getItem(
            parent, "Calibration", "Which calibration was this session filmed with?", names, 0, False
        )
        if not ok:
            return None
        app_state.calibration_name = name
    return resolve_calibration(project, name)


def import_dlc_calibration_into_project(app_state, parent: QWidget | None = None) -> Path | None:
    """Tools ▸ 3D: Import DeepLabCut calibration… — one DeepLabCut 3D folder becomes a project calibration."""
    project = project_dir_of(app_state)
    if project is None:
        QMessageBox.information(
            parent,
            "No project folder",
            "A calibration belongs to a rig, so it lives in the project folder.\n\n"
            "Choose a project folder on the start page first.",
        )
        return None
    folder = browse_open_dir(parent, app_state, "Choose a DeepLabCut 3D project folder")
    if not folder:
        return None
    try:
        path = import_dlc_calibration(folder, project)
    except CalibrationError as err:
        notify(str(err), "error")
        return None
    notify(f"Calibration imported as {path.name} in {path.parent}.", "info")
    return path


def triangulate_loaded_session(data_widget, parent: QWidget | None = None) -> bool:
    """Tools ▸ 3D: Triangulate poses… — ``position_3d`` for the trials the trials table shows."""
    app_state = data_widget.app_state
    dt = app_state.dt
    if dt is None:
        notify("Load a session first.", "warning")
        return False
    calibration = calibration_path_of(app_state, parent, ask=True)
    if calibration is None:
        project = project_dir_of(app_state)
        if project is None or not calibration_names(project):
            notify(
                "The project folder has no calibration. Import one with Tools ▸ 3D: Import DeepLabCut "
                "calibration…, or put an Anipose calibration.toml in its calibration/ folder.",
                "warning",
            )
        return False

    busy = BusyProgressDialog("Triangulating…", parent=parent)

    def progress(done: int, total: int) -> bool:
        busy.setLabelText(f"Triangulating… trial {done + 1} of {total}")
        busy.pump_events()
        return not busy.wasCanceled()

    done, error = busy.execute(
        triangulate_tree,
        dt,
        app_state.nwb_alignment,
        calibration,
        trials=list(app_state.trials),
        feature=app_state.pose_overlay_feature or "position",
        pose_folder=app_state.pose_folder,
        source_software=app_state.source_software,
        progress=progress,
    )
    if error is not None:
        return False

    saved = bool(getattr(dt, "_source_path", None))
    if saved:
        dt.save()
    data_widget.serve_current_tree()
    where = "saved to the session's dataset" if saved else "kept in memory (this session has no .nc to save to)"
    notify(f"{POSITION_3D} written for {len(done)} trials with {calibration.name}; {where}.", "info")
    return True
