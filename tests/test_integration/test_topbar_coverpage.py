"""Integration tests for the top menu bar and the IO cover page.

Exercises the two GUI layout-refactor deliverables against the real
``EthographMainWindow``/``MetaWidget`` (via the ``gui`` fixture) and against a
loaded template dataset (``birdpark_gui``), so the menu actions and the
drag&drop alignment builder are driven end-to-end rather than only imported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ethograph.io.validation import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS


def _menu(shell, title: str):
    for action in shell.menuBar().actions():
        if action.text().replace("&", "") == title:
            return action.menu()
    raise AssertionError(f"menu {title!r} not found")


def _find_action(menu, label: str):
    for action in menu.actions():
        if action.text().replace("&", "").startswith(label):
            return action
        if action.menu() is not None:
            for sub in action.menu().actions():
                if sub.text().replace("&", "").startswith(label):
                    return sub
    raise AssertionError(f"action {label!r} not found")


# ---------------------------------------------------------------------------
# Top bar (no dataset needed)
# ---------------------------------------------------------------------------


def test_video_area_survives_attach(gui):
    """setCentralWidget(plots) must not delete the video area / CameraView.

    Regression: the video used to be the central widget, and making the plots
    central deleted the CameraView C++ object (crash in _add_labels_shapes_layer).
    """
    from qtpy.QtWidgets import QLabel

    shell, meta = gui
    canvas = shell.canvas_widget()
    assert canvas is not None
    # Touch the CameraView and build a child QLabel on the canvas exactly like
    # the label overlay does — raises RuntimeError if the C++ object was deleted.
    assert shell.video_area.primary.objectName() is not None
    QLabel(canvas)


def test_every_panel_has_close_and_move_buttons(gui):
    from qtpy.QtWidgets import QPushButton

    shell, meta = gui
    pc = meta.plot_container
    # Every fixed panel is a dock with a slim title bar holding move + ✕
    # buttons (dynamic panels — incl. neo — get theirs per-instance in
    # add_panel). Fixed panels are the Phy trace + raster.
    assert set(pc._panel_docks) == {"ephys", "raster"}
    meta.app_state.has_audio = True
    plot = pc.add_audio_panel("audiotrace")
    dock = pc._dyn_docks[plot]
    assert not dock.isHidden()
    assert dock.titleBarWidget().findChild(QPushButton, "panel_move_btn") is not None
    close_btn = dock.titleBarWidget().findChild(QPushButton, "panel_close_btn")
    close_btn.click()
    assert plot not in pc.audio_trace_plots
    assert plot not in pc._dyn_docks


def test_tools_menu_screen_record_is_a_plain_action(gui, monkeypatch):
    """The Tools entry itself starts/stops recording — no nested button."""
    from qtpy.QtWidgets import QDialog, QWidgetAction

    from ethograph.gui import dialog_screen_recorder as dsr

    shell, meta = gui
    tools = _menu(shell, "Tools")
    assert not any(isinstance(a, QWidgetAction) for a in tools.actions())
    action = _find_action(tools, "Demo: Screen-record")
    builder = shell._top_bar

    # Triggering the entry goes straight to the recorder's settings dialog.
    opened = []

    class _StubDialog:
        def __init__(self, parent=None):
            opened.append(1)

        def exec_(self):
            return QDialog.Rejected

    monkeypatch.setattr(dsr, "RecordDialog", _StubDialog)
    action.trigger()
    assert opened == [1]
    assert builder._record_controller.state == "idle"  # cancelled → nothing started

    # State drives the entry's label (the only stop affordance besides Ctrl+Space).
    builder._on_record_state("recording")
    assert "Stop" in action.text()
    builder._on_record_state("idle")
    assert "Screen-record" in action.text()


def test_show_changepoints_menu_action_syncs_state(gui):
    shell, meta = gui
    cp_menu = _menu(shell, "Changepoints")
    action = _find_action(cp_menu, "Show changepoints")
    assert action.isCheckable()

    checkbox = meta.changepoints_widget.show_cp_checkbox
    start = checkbox.isChecked()
    action.trigger()  # simulate the user clicking the menu item
    assert checkbox.isChecked() != start
    assert meta.app_state.show_changepoints == checkbox.isChecked()


def test_changepoints_popup_borrows_and_returns_detached_widget(gui):
    shell, meta = gui
    cp = meta.changepoints_widget
    holder = meta._detached_holder
    assert cp.parent() is holder  # detached widget parked in the holder

    builder = shell._top_bar
    builder._popup_section("cp", "Changepoint correction", cp)
    dlg = builder._open_popups["cp"]
    assert cp.parent() is not holder  # borrowed into the dialog
    dlg.close()
    assert cp.parent() is holder  # returned to the holder


def test_io_subpanel_popups_are_separate(gui):
    """Import labels / Import predictions / Export labels each pop up alone.

    The popup hosts only its own sub-panel — none of the other sections —
    and the widget returns home on close.
    """
    shell, meta = gui
    io = meta.io_widget
    builder = shell._top_bar

    builder._popup_section("import_labels", "Import labels", io.labels_group)
    dlg = builder._open_popups["import_labels"]
    assert dlg.isAncestorOf(io.labels_group)
    assert not dlg.isAncestorOf(io.pred_group)  # predictions stay separate
    dlg.close()
    assert io.isAncestorOf(io.labels_group)

    builder._popup_section("import_predictions", "Import predictions", io.pred_group)
    dlg = builder._open_popups["import_predictions"]
    assert dlg.isAncestorOf(io.pred_group)
    assert not dlg.isAncestorOf(io.labels_group)
    dlg.close()
    assert io.isAncestorOf(io.pred_group)

    builder._popup_section("export_labels", "Export labels", io.export_panel)
    dlg = builder._open_popups["export_labels"]
    assert dlg.isAncestorOf(io.export_panel)
    dlg.close()
    assert io.isAncestorOf(io.export_panel)
    assert io.export_panel.isHidden()  # only shown while borrowed by its popup


def test_plot_click_shows_only_relevant_sections(birdpark_gui):
    """Clicking a plot shows only that plot's sections (minimal sidebar)."""
    meta = birdpark_gui[1]
    ctx = meta.context_panel
    ps = meta.plot_settings_widget
    dp = meta.data_panel

    # Audio trace → energy/envelope + shared, NOT spectrogram.
    meta._on_plot_focus("audiotrace")
    assert ctx.current_context() == "audiotrace"
    assert dp.energy_group.isVisibleTo(ctx)
    assert not ps.spectrogram_panel.isVisibleTo(ctx)
    assert not dp.coords_groupbox.isVisibleTo(ctx)

    # Spectrogram → only the spectrogram panel, NOT energy.
    meta._on_plot_focus("spectrogram")
    assert ctx.current_context() == "spectrogram"
    assert ps.spectrogram_panel.isVisibleTo(ctx)
    assert not dp.energy_group.isVisibleTo(ctx)

    # Feature (lineplot) → xarray coords (feature dims), NOT spectrogram.
    meta._on_plot_focus("feature")
    assert ctx.current_context() in ("lineplot", "heatmap")
    assert dp.coords_groupbox.isVisibleTo(ctx)
    assert not ps.spectrogram_panel.isVisibleTo(ctx)


def test_trials_table_hidden_without_metadata(gui):
    import pandas as pd

    shell, meta = gui
    trials = meta.trials_widget
    # Bare trial numbers → no metadata → table hidden. The editing controls
    # stay, since that is how the first column gets added.
    trials.setup(pd.DataFrame({"trial": [1, 2, 3]}))
    assert trials._table.isHidden()
    assert not trials._add_column_button.isHidden()
    # Real metadata columns → table shown.
    trials.setup(pd.DataFrame({"trial": [1, 2], "condition": ["a", "b"]}))
    assert not trials._table.isHidden()


def test_video_context_shows_pose_only_when_pose_exists(birdpark_gui):
    """Playback controls live in the bottom bar now — the video context holds
    only the pose section, hidden for a dataset with no pose data."""
    shell, meta = birdpark_gui
    meta.focus_video_context()
    assert meta.context_panel.current_context() == "video"
    pose_group = meta.data_panel.pose_groupbox
    assert meta.context_panel.isAncestorOf(pose_group)
    assert not pose_group.isVisibleTo(meta.context_panel)


def test_zen_mode_toggle(gui, qtbot):
    shell, meta = gui
    shell.show()
    qtbot.waitExposed(shell)
    shell.set_zen_mode(True)
    assert not shell._sidebar_dock.isVisible()
    assert meta.app_state.zen_mode is True
    shell.set_zen_mode(False)
    assert shell._sidebar_dock.isVisible()
    assert meta.app_state.zen_mode is False


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Add-panel popup (drag-drop / Enter panel creation)
# ---------------------------------------------------------------------------


def test_allowed_plot_types_gating():
    from ethograph.gui.source_popup import allowed_plot_types

    class _State:
        ds = None  # feature_ncols falls back to 1

    audio = allowed_plot_types("audio", "mic1", _State())
    assert audio == ["Audio Trace", "Spectrogram Trace"]
    feat = allowed_plot_types("feature", "speed", _State())
    assert feat == ["Lineplot"]  # N==1 -> no heatmap/space


def test_add_panel_button_wired(gui):
    """The bottom bar's ➕ button opens the popup (guarded before a load)."""
    shell, meta = gui
    btn = shell.bottom_bar.add_panel_btn
    assert not meta.app_state.ready
    btn.click()  # no dataset yet → warning notify, popup stays hidden
    assert not meta.source_popup.isVisible()


def test_source_popup_lists_filters_and_navigates(qtbot):
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent

    from ethograph.gui.source_popup import SourcePopup

    class _Catalog:
        @staticmethod
        def feature_choices():
            return ["speed", "position", "beak_angle"]

    class _State:
        nwb_alignment = None
        ds = None

    popup = SourcePopup(_State())
    qtbot.addWidget(popup)
    popup.refresh(catalog=_Catalog())
    lst = popup._list
    texts = [lst.item(i).text().strip() for i in range(lst.count())]
    assert "Media" in texts and "Features" in texts
    # At least one draggable data item exists below the headers.
    assert len(texts) > 2

    # Filtering hides non-matching entries (and empty group headers).
    rows = popup._visible_data_rows()
    assert rows, "expected at least one source entry"
    target = lst.item(rows[0]).text().strip()
    popup._filter.setText(target)
    visible = [lst.item(i).text().strip() for i in popup._visible_data_rows()]
    assert target in visible
    popup._filter.setText("zzz-no-such-feature")
    assert popup._visible_data_rows() == []
    popup._filter.clear()
    assert popup._visible_data_rows() == rows

    # ↑/↓ typed in the filter box move the list selection (wrapping).
    popup._select_first_visible()
    down = QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier)
    popup.eventFilter(popup._filter, down)
    assert lst.currentRow() == rows[1 % len(rows)]
    up = QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier)
    popup.eventFilter(popup._filter, up)
    assert lst.currentRow() == rows[0]


def test_drop_feature_creates_lineplot(birdpark_gui):
    shell, meta = birdpark_gui
    feats = meta.plot_container._available_features()
    assert feats, "expected at least one feature in birdpark"
    n_before = len(meta.plot_container.line_plots)
    meta._create_panel_for_source("feature", feats[0], "Lineplot")
    assert len(meta.plot_container.line_plots) == n_before + 1


def test_clicking_feature_plot_sets_active_and_context(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    p = pc.add_lineplot(feature=pc._available_features()[0])
    reg = meta.active_panels.registration_for(p)
    assert reg is not None  # every line plot auto-registers with the manager
    meta.active_panels.set_active(reg)
    assert pc.active_feature_plot is p
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")


def test_lineplots_have_independent_state(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    feats = pc._available_features()
    first = pc.line_plots[0]
    p = pc.add_lineplot(feature=feats[0])
    first_before = first._effective_feature()
    # Editing one plot's own state must not change any other line plot.
    pc.active_feature_plot = p
    target = feats[1] if len(feats) > 1 else feats[0]
    p.set_panel_control("features", target)
    p.set_panel_control("keypoint", "beakTip")
    assert p._effective_feature() == target
    assert p._effective_selections().get("keypoint") == "beakTip"
    assert first._effective_feature() == first_before  # peer untouched


def test_lineplot_axes_are_independent(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    p = pc.add_lineplot(feature=pc._available_features()[0])
    pc.active_feature_plot = p
    ps = meta.plot_settings_widget
    ps.ymin_edit.setText("1.0")
    ps.ymax_edit.setText("5.0")
    ps._on_axes_edited()
    assert p.panel_state.get("ymin") == 1.0
    assert p.panel_state.get("ymax") == 5.0
    # Global axes (which drive the main plot) were not touched.
    assert getattr(meta.app_state, "ymin", None) != 1.0


def test_active_panel_gets_green_edge(birdpark_gui):
    shell, meta = birdpark_gui
    mgr = meta.active_panels
    pc = meta.plot_container
    p = pc.add_lineplot(feature=pc._available_features()[0])
    p_reg = mgr.registration_for(p)
    mgr.set_active(p_reg)
    assert mgr.active is p_reg
    assert "2ecc71" in p.styleSheet().lower()
    # Activating another panel moves the green edge.
    first = pc.line_plots[0]
    mgr.set_active(mgr.registration_for(first))
    assert "2ecc71" not in p.styleSheet().lower()
    assert "2ecc71" in first.styleSheet().lower()


def test_video_click_activates_green_edge_and_pose_playback(birdpark_gui):
    shell, meta = birdpark_gui
    mgr = meta.active_panels
    primary = shell.video_area.primary
    reg = mgr.registration_for(primary)
    assert reg is not None
    primary.clicked.emit()  # simulate a click on the video
    assert mgr.active is reg
    assert "2ecc71" in primary.styleSheet().lower()  # green edge
    assert meta.context_panel.current_context() == "video"  # pose + playback


def test_extra_camera_views_are_individual_docks(gui, qtbot):
    """Each extra camera view is its own closable shell dock — closing one
    view's ✕ removes only that view, never its siblings (even duplicates of
    the same camera)."""
    from qtpy.QtWidgets import QDockWidget

    shell, meta = gui
    area = shell.video_area

    v1 = area.add_extra("cam")
    v2 = area.add_extra("cam")  # duplicate view of the same camera
    v3 = area.add_extra("other")
    assert len(area.extras) == 3

    docks = [v.dock_widget for v in (v1, v2, v3)]
    assert all(isinstance(d, QDockWidget) for d in docks)
    assert len({d.objectName() for d in docks}) == 3  # unique per instance

    mgr = meta.active_panels
    assert mgr.registration_for(v2) is not None

    # Close ONE duplicate — the other view of the same camera must survive.
    v2.dock_widget.close()
    qtbot.waitUntil(lambda: len(area.extras) == 2, timeout=2000)
    assert v1 in area.extras.values()
    assert v3 in area.extras.values()
    assert mgr.registration_for(v2) is None
    assert mgr.registration_for(v1) is not None

    # Programmatic removal by camera name still removes every view of it.
    meta.data_widget.video_mgr.remove_camera("cam")
    assert list(area.extras.values()) == [v3]


def test_space_plot_registers_and_activates(moll2025_gui):
    shell, meta = moll2025_gui
    meta.app_state.space_plot_type = "Space Plot"
    meta.data_widget.update_space_plot()
    sp = meta.data_widget.space_plot
    assert sp is not None
    mgr = meta.active_panels
    reg = mgr.registration_for(sp)
    assert reg is not None  # space plot registered with the manager
    sp.clicked.emit()  # simulate a click
    assert mgr.active is reg
    assert "2ecc71" in sp.styleSheet().lower()  # green edge
    assert meta.context_panel.current_context() == "space"


def test_all_panel_types_registered(birdpark_gui):
    shell, meta = birdpark_gui
    mgr = meta.active_panels
    pc = meta.plot_container
    # Video, audio trace, spectrogram, line plots all register.
    for w in (pc.audio_trace_plot, pc.spectrogram_plot, pc.line_plots[0]):
        assert mgr.registration_for(w) is not None
    assert mgr.registration_for(shell.video_area.primary) is not None
    # Heatmaps are dynamic instances like every other panel: they register
    # on creation and unregister on removal.
    hm = pc.add_panel("heatmap")
    assert mgr.registration_for(hm) is not None
    pc.remove_panel(hm)
    assert mgr.registration_for(hm) is None


def test_panel_control_only_affects_active_plot(birdpark_gui):
    shell, meta = birdpark_gui
    dw = meta.data_widget
    pc = meta.plot_container
    feats = pc._available_features()
    p1 = pc.add_lineplot(feature=feats[0])
    p2 = pc.add_lineplot(feature=feats[0])
    # Any data-panel control routes only to the active plot (generic path).
    pc.active_feature_plot = p1
    dw.apply_panel_control("keypoint", "beakTip")
    assert p1._effective_selections().get("keypoint") == "beakTip"
    assert p2._effective_selections().get("keypoint") != "beakTip"


def test_added_lineplot_behaves_like_any_lineplot(birdpark_gui):
    shell, meta = birdpark_gui
    pc = meta.plot_container
    feats = pc._available_features()
    plot = pc.add_lineplot(feature=feats[0])
    assert plot is not None
    # No per-plot feature dropdown (all line plots share the sidebar controls).
    assert not hasattr(plot, "_feature_combo")
    # Clicking it switches the sidebar to the lineplot context.
    meta.context_panel.set_context("audiotrace")
    meta.active_panels.set_active(meta.active_panels.registration_for(plot))
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")


def test_reclick_active_panel_resyncs_context(birdpark_gui):
    """A click is never a silent no-op: re-clicking the already-active panel
    re-announces it, so a stale sidebar recovers on the FIRST click — not
    after clicking another panel and coming back."""
    shell, meta = birdpark_gui
    pc = meta.plot_container
    plot = pc.add_lineplot(feature=pc._available_features()[0])
    reg = meta.active_panels.registration_for(plot)
    meta.active_panels.set_active(reg)
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")
    # The sidebar drifts while the panel stays active.
    meta.context_panel.set_context("audiotrace")
    # ONE re-click on the same, still-active panel restores its context.
    meta.active_panels.set_active(reg)
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")


def test_data_section_return_resyncs_context(birdpark_gui):
    """Clicks made while the Labels/Navigation section is open deliberately
    don't swap the sidebar; returning to the Data section applies them."""
    shell, meta = birdpark_gui
    pc = meta.plot_container
    plot = pc.add_lineplot(feature=pc._available_features()[0])
    meta._expand(1)  # Labels section open → context swaps suppressed
    before = meta.context_panel.current_context()
    meta.active_panels.set_active(meta.active_panels.registration_for(plot))
    assert meta.context_panel.current_context() == before  # suppressed
    meta._expand(0)  # back to Data → re-synced from the active panel
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")


def test_zen_mode_exit_resyncs_context(birdpark_gui):
    """Clicks made during zen mode don't swap the (hidden) sidebar; leaving
    zen mode re-syncs it to the active panel."""
    shell, meta = birdpark_gui
    pc = meta.plot_container
    plot = pc.add_lineplot(feature=pc._available_features()[0])
    meta.app_state.zen_mode = True
    before = meta.context_panel.current_context()
    meta.active_panels.set_active(meta.active_panels.registration_for(plot))
    assert meta.context_panel.current_context() == before  # suppressed
    meta.app_state.zen_mode = False
    assert meta.context_panel.current_context() in ("lineplot", "heatmap")


def test_cover_page_borrows_and_returns_load_panel(gui, qtbot):
    """The IO load panel (path fields + Load) is hosted on the cover page
    while it is shown, and handed back to the IO tab when it closes."""
    from ethograph.gui.cover_page import CoverPage

    shell, meta = gui
    io = meta.io_widget
    page = CoverPage(shell, io)
    page.show()
    qtbot.waitExposed(page)
    assert page._load_host.isAncestorOf(io.load_panel)  # borrowed
    assert io.load_buttons_row.isHidden()  # no duplicate wizard/template buttons
    page.close()
    assert io.isAncestorOf(io.load_panel)  # returned
    assert not io.load_buttons_row.isHidden()


def test_cover_page_custom_load_accepts_page(gui, qtbot, monkeypatch):
    """Custom set-up path: clicking the shared Load bar's button closes
    (accepts) the cover page once the dataset is loaded — otherwise the
    modal dialog stays open forever and the main window never appears."""
    from qtpy.QtWidgets import QDialog

    from ethograph.gui.cover_page import CoverPage

    shell, meta = gui
    io = meta.io_widget
    page = CoverPage(shell, io)
    page.show()
    qtbot.waitExposed(page)
    assert io.load_button.isHidden()  # replaced by the shared Load bar

    monkeypatch.setattr(
        io.data_widget,
        "on_load_clicked",
        lambda: setattr(meta.app_state, "ready", True),
    )
    page._shared_load_btn.click()

    assert page.result() == QDialog.Accepted
    assert not page.isVisible()
    assert io.isAncestorOf(io.load_panel)  # panel handed back on close


def test_cover_page_classify_files():
    from ethograph.gui.cover_page import classify_files

    buckets = classify_files(["a.mp4", "b.h5", "c.wav", "d.dat", "s.nc", "labels.tsv", "arena.png", "junk.xyz"])
    assert buckets["video"] == ["a.mp4"]
    assert buckets["pose"] == ["b.h5"]
    assert buckets["audio"] == ["c.wav"]
    assert buckets["ephys"] == ["d.dat"]
    assert buckets["session"] == ["s.nc"]
    assert buckets["labels"] == ["labels.tsv"]
    assert buckets["image"] == ["arena.png"]
    assert buckets["unknown"] == ["junk.xyz"]


def test_sidebar_can_be_widened_on_a_small_window(gui, qtbot):
    """The right sidebar must stay draggable on small screens.

    The playback bar packs ~900 px of controls; docked bare, its minimum width
    became the window's, so the sidebar separator would not move at all.
    """
    from qtpy.QtCore import Qt

    shell, meta = gui
    shell.show()
    qtbot.waitExposed(shell)
    shell.resize(1000, 640)
    qtbot.wait(200)
    # The bar is hosted in a scroll area, so it never dictates the window width.
    assert shell._bottom_bar_host.minimumSizeHint().width() < 300
    dock = shell._sidebar_dock
    shell.resizeDocks([dock], [600], Qt.Horizontal)
    qtbot.wait(200)
    assert dock.width() >= 500


def test_cover_page_shrinks_below_content_size(gui, qtbot):
    """Short screens: the page must be resizable smaller than its content.

    All cards live in a scroll area, so the dialog's minimum size stays well
    under a 1024x768 laptop screen instead of being pinned by the cards.
    """
    from ethograph.gui.cover_page import CoverPage

    shell, meta = gui
    page = CoverPage(shell, meta.io_widget)
    page.show()
    qtbot.waitExposed(page)
    assert page.minimumSizeHint().height() <= 560
    page.resize(760, 460)
    qtbot.waitUntil(lambda: page.height() <= 470, timeout=2000)
    assert page.width() <= 780
    page.close()


def test_cover_page_offers_the_tag_sheet_before_any_recording(gui):
    """Tags have to be printed before a video exists, so the entry point must
    live on the one screen shown before anything is loaded."""
    pytest.importorskip("cv2")
    from ethograph.gui.cover_page import CoverPage
    from ethograph.gui.dialog_tag_sheet import TagSheetDialog

    shell, meta = gui
    page = CoverPage(shell, meta.io_widget)
    actions = [action.text() for action in page._tools_button.menu().actions()]
    assert "Print tag sheet…" in actions

    page._open_tag_sheet()
    try:
        assert isinstance(page._tag_sheet, TagSheetDialog)
        assert page._tag_sheet._pages, "a sheet must lay out with no video and no dataset"
    finally:
        page._tag_sheet.close()
        page.close()


def test_cover_page_scales_with_screen_height(gui):
    """Fixed pixel sizes are multiplied by a screen-height-derived factor."""
    from ethograph.gui import cover_page as cp

    shell, meta = gui
    page = cp.CoverPage(shell, meta.io_widget)
    assert cp._MIN_SCALE <= page._scale <= 1.0
    assert page._px(100) == max(1, round(100 * page._scale))
    assert page._px(48) <= 48  # never grows beyond the tuned 1080 px sizes


def test_cover_page_image_only_drop_rejected(gui):
    """An image alone has no time axis — the drop must fail with guidance."""
    from ethograph.gui.cover_page import CoverPage, classify_files

    shell, meta = gui
    page = CoverPage(shell, meta.io_widget)
    buckets = classify_files(["arena.png"])
    with pytest.raises(RuntimeError, match="no time axis"):
        page._populate_io_from_buckets(buckets, {"data_sr": None, "source_software": None, "pose_fps": None})


def test_cover_page_pose_only_drop_loads_as_features(gui):
    """A pose file dropped with no video and no image loads on its own: the
    pose becomes a features .nc (position/confidence) with a pose-only
    alignment — plottable panels never require a camera."""
    from ethograph.datasets import dataset_dir, is_dataset_downloaded
    from ethograph.gui.cover_page import CoverPage, classify_files
    from ethograph.io.nwb_alignment import NWBAlignment

    if not is_dataset_downloaded("moll2025"):
        pytest.skip("moll2025 not downloaded")
    pose_csv = next(iter(sorted(dataset_dir("moll2025").glob("*DLC.csv"))), None)
    if pose_csv is None:
        pytest.skip("moll2025 has no DLC pose file")

    shell, meta = gui
    page = CoverPage(shell, meta.io_widget)
    buckets = classify_files([str(pose_csv)])
    assert buckets["pose"] == [str(pose_csv)]
    page._populate_io_from_buckets(buckets, {"data_sr": None, "source_software": "DeepLabCut", "pose_fps": 30.0})

    app_state = meta.app_state
    assert app_state.nc_file_path and app_state.nc_file_path.endswith(".nc")
    assert app_state.nwb_file_path and app_state.nwb_file_path.endswith(".tmp.nwb")
    align = NWBAlignment(app_state.nwb_file_path)
    assert align.cameras == []  # no camera view exists for a standalone pose
    assert align.get_stream_rate("pose", "cam-1") == 30.0

    import xarray as xr

    with xr.open_dataset(app_state.nc_file_path) as ds:
        assert "position" in ds.data_vars
        assert "time" in ds["position"].dims

    # The drop must load into a working session: position is a catalog
    # feature offered as a space plot, with no video anywhere.
    from qtpy.QtWidgets import QApplication

    from ethograph.gui.source_popup import allowed_plot_types

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    assert app_state.ready
    assert app_state.video is None

    assert "position" in meta.data_widget.catalog.feature_choices()
    assert "Space (2D)" in allowed_plot_types("feature", "position", app_state)
    meta._create_panel_for_source("feature", "position", "Space (2D)")
    QApplication.processEvents()
    assert meta.data_widget.space_plots
    assert meta.data_widget.space_plots[-1].dock_widget is not None


def test_cover_page_builds_single_trial_alignment(gui, birdpark_data_dir):
    """Drag&drop path: build a real alignment.tmp.nwb from birdpark media."""
    from ethograph.gui.cover_page import CoverPage
    from ethograph.io.nwb_alignment import NWBAlignment

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    videos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    audio_only = AUDIO_EXTENSIONS - VIDEO_EXTENSIONS
    audios = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in audio_only)
    if not videos:
        pytest.skip("birdpark has no video files to build an alignment from")

    page = CoverPage(shell, meta.io_widget)
    page._drop_tmp_dir = page._prepare_drop_dir()
    cam_map = [(str(videos[0]), None)]
    audio_files = [str(audios[0])] if audios else []
    nwb_path = page._build_tmp_alignment(cam_map, audio_files)

    assert nwb_path.exists()
    align = NWBAlignment(nwb_path)  # must be a readable alignment NWB
    rate = align.get_stream_rate("video", "cam-1")
    assert rate and rate > 0  # real fps, not a hardcoded fallback


def test_cover_page_session_plus_media_builds_alignment(gui, birdpark_data_dir):
    """Dropping a session .nc together with media synthesises a tmp alignment
    (nwb_file_path override) so the media is loadable — the session file's
    folder has no .ethograph sidecar describing the dropped files."""
    from ethograph.gui.cover_page import CoverPage, classify_files
    from ethograph.io.nwb_alignment import NWBAlignment

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    videos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    session = next(iter(sorted(data_dir.glob("*.nc"))), None)
    if not videos or session is None:
        pytest.skip("birdpark is missing a video or .nc session file")

    page = CoverPage(shell, meta.io_widget)
    buckets = classify_files([str(session), str(videos[0])])
    page._populate_io_from_buckets(buckets, {"data_sr": None, "source_software": None})

    app_state = meta.app_state
    assert app_state.nc_file_path == str(session)
    assert app_state.nwb_file_path and app_state.nwb_file_path.endswith(".tmp.nwb")
    align = NWBAlignment(app_state.nwb_file_path)
    assert align.cameras == ["cam-1"]
    assert app_state.video_folder == str(videos[0].parent)


def test_cover_page_audio_only_alignment(gui, birdpark_data_dir):
    """Audio-only drops build an alignment too (duration from the audio file)."""
    from ethograph.gui.cover_page import CoverPage
    from ethograph.io.nwb_alignment import NWBAlignment

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    audio_only = AUDIO_EXTENSIONS - VIDEO_EXTENSIONS
    audios = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in audio_only)
    if not audios:
        pytest.skip("birdpark has no audio-only files")

    page = CoverPage(shell, meta.io_widget)
    page._drop_tmp_dir = page._prepare_drop_dir()
    nwb_path = page._build_tmp_alignment([], [str(audios[0])])

    align = NWBAlignment(nwb_path)
    assert align.mics == ["mic-1"]
    assert align.cameras == []
    assert align.stop_time(1) and align.stop_time(1) > 0
    # Stream-based alignments have no trials-table filename columns —
    # the GUI must resolve audio via the ImageSeries external_file.
    assert align.get_media(1, "audio", "mic-1") is None
    resolved = align.resolve_media_path(1, "audio", device="mic-1")
    assert resolved and Path(resolved).name == audios[0].name


def test_cover_page_multi_trial_drop_pairs_by_natural_sort(gui):
    """"Several trials of one device" layout: 2+ video/pose files with no
    camera-matching dialog — natural-sort paired into a real multi-trial
    TrialTree, mirroring the Data wizard's single-camera 'Pair' route."""
    import ethograph as eto
    from ethograph.datasets import dataset_dir, is_dataset_downloaded
    from ethograph.gui.cover_page import CoverPage, classify_files

    if not is_dataset_downloaded("moll2025"):
        pytest.skip("moll2025 not downloaded")
    data_dir = dataset_dir("moll2025")
    videos = sorted(data_dir.glob("*.mp4"))
    poses = sorted(data_dir.glob("*DLC.csv"))
    if len(videos) < 2 or len(poses) < 2:
        pytest.skip("moll2025 does not have 2+ video/pose files to pair")

    shell, meta = gui
    page = CoverPage(shell, meta.io_widget)
    page._drop_layout_combo.setCurrentIndex(page._drop_layout_combo.findData("multi_trial"))
    buckets = classify_files([str(v) for v in videos] + [str(p) for p in poses])
    assert page._is_multi_trial_drop(buckets)

    page._populate_io_from_buckets(buckets, {"source_software": "DeepLabCut", "pose_fps": None})

    app_state = meta.app_state
    assert app_state.nc_file_path and app_state.nc_file_path.endswith("session.nc")
    dt = eto.open(app_state.nc_file_path)
    assert len(dt.trials) == len(videos)
    assert app_state.video_folder == str(videos[0].parent)
    assert app_state.pose_folder == str(poses[0].parent)

    align = app_state.nwb_alignment
    for trial in range(1, len(videos) + 1):
        resolved = align.resolve_media_path(trial, "video", device="cam-1")
        assert resolved and Path(resolved) in videos


def test_cover_page_labels_drop_loads_on_first_load(gui, birdpark_data_dir):
    """A dropped labels .tsv with a non-canonical name must still load.

    ``resolve_labels_tsv`` only auto-discovers ``{nc_stem}_labels.tsv``; a
    dropped tsv with a different name previously relied on the "Import
    labels" checkbox alone, which had no wiring into the load pipeline. The
    checkbox must set an explicit override that survives to Load, even
    though the labels format-combo/path-edit widgets it *normally* backs
    onto do not exist yet on a first load (they're created lazily, after
    load, by ``IOWidget.create_device_controls``).
    """
    from qtpy.QtWidgets import QApplication

    from ethograph.gui.cover_page import CoverPage, classify_files
    from ethograph.labels.tsv_store import save_labels_tsv

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    session = next(iter(sorted(data_dir.glob("*.nc"))), None)
    if session is None:
        pytest.skip("birdpark has no .nc session file")

    import pandas as pd

    # nc_file_path/labels_import_path are SCOPE_LOCAL and persist into the
    # real (shared) example dataset's .ethograph/local_settings.yaml — reset
    # before and after so this test never leaks state across runs.
    app_state = meta.app_state
    app_state.nc_file_path = str(session)
    app_state.labels_import_path = None

    labels_tsv = data_dir / "my_custom_annotations.tsv"
    rows = pd.DataFrame(
        {
            "onset_s": [0.1],
            "offset_s": [0.5],
            "labels": [1],
            "individual": ["individual 1"],
            "trial": [1],
            "human_verified": [False],
            "changepoint_corrected": [False],
            "prediction_source": ["manual"],
            "n_samples": [0],
        }
    )
    save_labels_tsv(labels_tsv, rows)

    try:
        page = CoverPage(shell, meta.io_widget)
        buckets = classify_files([str(session), str(labels_tsv)])
        assert buckets["labels"] == [str(labels_tsv)]
        page._populate_io_from_buckets(buckets, {"data_sr": None, "source_software": None})

        assert meta.io_widget.import_labels_checkbox.isChecked()
        assert app_state.labels_import_path == str(labels_tsv)
        assert meta.io_widget.get_import_labels_path() == str(labels_tsv)

        meta.data_widget.on_load_clicked()
        QApplication.processEvents()
        assert app_state.ready
        assert not app_state._all_labels_df.empty
        assert 1 in set(app_state._all_labels_df["labels"])
    finally:
        labels_tsv.unlink(missing_ok=True)
        app_state.labels_import_path = None


def test_import_labels_checkbox_seeds_and_persists_canonical_guess(gui, birdpark_data_dir):
    """First tick with no persisted override seeds+persists the canonical guess.

    Mirrors how a downloaded template (e.g. moll2025) ships its labels tsv
    named exactly ``{nc_stem}_labels.tsv`` next to the ``.nc`` — the checkbox
    must still resolve it on a completely fresh dataset (nothing in
    local_settings.yaml yet), and remember it afterwards so it is never
    re-guessed.
    """
    from ethograph.labels.tsv_store import labels_tsv_path, save_labels_tsv

    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    session = next(iter(sorted(data_dir.glob("*.nc"))), None)
    if session is None:
        pytest.skip("birdpark has no .nc session file")

    import pandas as pd

    canonical_tsv = labels_tsv_path(session)
    rows = pd.DataFrame(
        {
            "onset_s": [0.2],
            "offset_s": [0.6],
            "labels": [1],
            "individual": ["individual 1"],
            "trial": [1],
            "human_verified": [False],
            "changepoint_corrected": [False],
            "prediction_source": ["manual"],
            "n_samples": [0],
        }
    )
    save_labels_tsv(canonical_tsv, rows)
    app_state = meta.app_state
    io = meta.io_widget
    # `import_labels_nc_data` is SCOPE_GLOBAL — unlike the SCOPE_LOCAL fields
    # below it isn't keyed on this dataset, so it can carry over from a real
    # gui_settings.yaml on this machine. Force a real False→True transition
    # (a same-state setChecked() is a no-op and never fires the signal under
    # test) and restore the previous value afterwards.
    prior_checked = io.import_labels_checkbox.isChecked()
    io.import_labels_checkbox.setChecked(False)
    # SCOPE_LOCAL settings persist into the real (shared) example dataset's
    # .ethograph/local_settings.yaml — force the "nothing persisted yet"
    # starting condition explicitly and clean up after, rather than assuming
    # a previous test run left it untouched.
    app_state.nc_file_path = str(session)
    app_state.labels_import_path = None
    try:
        io.nc_file_path_edit.setText(str(session))
        io.import_labels_checkbox.setChecked(True)

        assert app_state.labels_import_path == str(canonical_tsv)
        assert io.get_import_labels_path() == str(canonical_tsv)
    finally:
        canonical_tsv.unlink(missing_ok=True)
        app_state.labels_import_path = None
        io.import_labels_checkbox.setChecked(prior_checked)


def test_import_labels_checkbox_missing_path_raises(gui, birdpark_data_dir):
    """A checked box pointing at a nonexistent file must fail loudly, not
    silently skip labels — the exact complaint that motivated the fix."""
    shell, meta = gui
    data_dir = Path(birdpark_data_dir)
    session = next(iter(sorted(data_dir.glob("*.nc"))), None)
    if session is None:
        pytest.skip("birdpark has no .nc session file")

    app_state = meta.app_state
    io = meta.io_widget
    io.nc_file_path_edit.setText(str(session))
    app_state.nc_file_path = str(session)
    try:
        app_state.labels_import_path = str(data_dir / "does_not_exist_labels.tsv")
        io.import_labels_checkbox.setChecked(True)

        with pytest.raises(Exception, match="Import labels"):
            meta.data_widget._phase_load_data(str(session))
    finally:
        app_state.labels_import_path = None


# ---------------------------------------------------------------------------
# End-to-end: load a template dataset, then drive a menu action
# ---------------------------------------------------------------------------


def test_birdpark_load_then_toggle_changepoints_via_menu(birdpark_gui):
    from qtpy.QtWidgets import QApplication

    shell, meta = birdpark_gui
    assert meta.app_state.ready

    cp_menu = _menu(shell, "Changepoints")
    action = _find_action(cp_menu, "Show changepoints")

    before = meta.app_state.show_changepoints
    action.trigger()
    QApplication.processEvents()
    assert meta.app_state.show_changepoints != before


def test_cover_page_project_drops_are_kept_and_reopenable(gui, qtbot, tmp_path):
    """With a project folder set, a drop's folder lands under the project's
    sessions/ and the reopen list puts its recorded state back into the IO
    fields; without one the drop stays throwaway and the list is hidden."""
    from ethograph.gui.cover_page import CoverPage
    from ethograph.gui.project import list_drops, record_drop
    from ethograph.utils.paths import is_throwaway_path

    shell, meta = gui
    page = CoverPage(shell, meta.io_widget)
    state = meta.app_state

    state.project_path = None
    page._refresh_project_ui()
    assert page._reopen_row.isHidden()
    assert is_throwaway_path(page._prepare_drop_dir())

    project = tmp_path / "study"
    project.mkdir()
    state.project_path = str(project)
    page._refresh_project_ui()
    assert not page._reopen_row.isHidden()
    assert not page._reopen_combo.isEnabled()  # nothing recorded yet

    drop_dir = page._prepare_drop_dir()
    assert drop_dir.parent == project / "sessions"
    state.video_folder = str(tmp_path)
    state.primary_camera = "cam-1"
    record_drop(drop_dir, [str(tmp_path / "clip.mp4")], state)
    page._refresh_project_ui()
    assert page._reopen_combo.count() == 2  # placeholder + the drop
    assert [f for f, _ in list_drops(project)] == [drop_dir]

    state.video_folder = None
    state.primary_camera = None
    page._reopen_combo.setCurrentIndex(1)
    page._on_reopen_drop(1)
    assert state.video_folder == str(tmp_path)
    assert state.primary_camera == "cam-1"
    assert page._reopened_drop == drop_dir
    assert page._drop.paths == []  # Load takes the restored fields, not a rebuild

    page.close()
