"""Ad-hoc: drop one OCTRON video + its bboxes .nc and report what the overlay and sidebar see."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QApplication

OCTRON = Path("C:/Users/aksel/Documents/octron/OUTPUT PER VIDEO")


def test_repro(gui, qtbot):
    from ethograph.gui.cover_page import CoverPage, classify_files

    shell, meta = gui
    shell.show()
    paths = [str(OCTRON / "nestCam.mp4"), str(OCTRON / "nestCam.nc")]
    buckets = classify_files(paths)
    print("BUCKETS", {k: [Path(p).name for p in v] for k, v in buckets.items() if v})
    page = CoverPage(shell, meta.io_widget)
    page._populate_io_from_buckets(
        buckets,
        {"data_sr": None, "source_software": None, "pose_fps": None, "extract_audio": False, "audio_track_videos": []},
    )
    app = meta.app_state
    print("NC", app.nc_file_path)
    print("NWB", app.nwb_file_path)
    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    qtbot.wait(300)
    QApplication.processEvents()
    sio = app.nwb_alignment
    print("READY", app.ready, "| cameras", sio.cameras, "| pose devices", sio.devices("pose"), "| pose_keys", sio.pose_keys)
    print("resolve pose cam-1:", sio.resolve_media_path(1, "pose", "cam-1", fallback_folder=app.pose_folder))
    pm = meta.data_widget.pose_mgr
    pr = pm._primary_pr
    print("primary_pr:", None if pr is None else (pr.data.shape, pr.bbox_data is not None and pr.bbox_data.shape))
    print("pose_kind:", pm.pose_kind(), "| _pose_available:", meta._pose_available(), "| has_pose attr:", getattr(app, "has_pose", None))
    print("context current:", meta.context_panel._current)
    for name in ("pose", "bbox"):
        w = meta.context_panel._sections.get(name)
        print(name, "section visible:", None if w is None else w.isVisibleTo(meta.context_panel))
    meta.focus_video_context()
    print("after focus_video_context:", meta.context_panel._current)
    for name in ("pose", "bbox"):
        w = meta.context_panel._sections.get(name)
        print(name, "section visible:", None if w is None else w.isVisibleTo(meta.context_panel))
    view = shell.video_area.primary
    ov = getattr(view, "_overlay", None) or getattr(view, "overlay", None)
    print("overlay:", type(ov).__name__ if ov is not None else None, "| bbox lines:", getattr(ov, "_bbox_lines", "n/a") is not None if ov else None)
