"""Sidebar screenshots for the user manual, taken from the GUI as ``ethograph launch`` builds it.

Builds the window exactly like :func:`ethograph.cli.launch` (theme, dialog
fitter, shell + MetaWidget), loads the tool-using crows template the way the
start page does, and captures through :mod:`screenshot_gui`::

    python scripts/docs_sidebar_screenshots.py

Writes into ``docs/source/_static/media``:

- ``add_panel.gif`` — from the camera alone, ➕ Add panel adds a 3D space plot
  and a speed line plot.
- ``sidebar_click_panel.gif`` — click a speed plot, a space plot, the video:
  the Data section follows the clicked panel.
- ``sidebar_individual_annotated.png`` — the Data section's top: the
  Individual group.
- ``sidebar_data_annotated.png`` — the Data section after clicking a line plot.
- ``sidebar_labels_annotated.png`` — the Labels section.
- ``cover_page.png`` — the start page, its cards boxed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from qtpy.QtCore import QElapsedTimer, QPoint, QRect, Qt
from qtpy.QtWidgets import QApplication, QWidget

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from screenshot_gui import capture_widget  # noqa: E402

from ethograph.datasets import resolve_dataset_paths  # noqa: E402
from ethograph.gui import notify, theme  # noqa: E402
from ethograph.gui.active_panel import PanelKind  # noqa: E402
from ethograph.gui.cover_page import CoverPage  # noqa: E402
from ethograph.gui.dialog_fit import install_dialog_fitter  # noqa: E402
from ethograph.gui.grid_section_container import GridSectionContainer  # noqa: E402
from ethograph.gui.main_window import EthographMainWindow  # noqa: E402
from ethograph.gui.widgets_meta import MetaWidget  # noqa: E402

OUT_DIR = ROOT / "docs" / "source" / "_static" / "media"
SCALE = 2
PAD = 4
WINDOW = (1800, 1100)
BLUE, ORANGE, GREEN, PURPLE, TEAL = "#29b6f6", "#ffa726", "#66bb6a", "#ab47bc", "#26c6da"
GIF_FRAME_MS = 2500
#: A few seconds of the trial read better than its whole extent.
GIF_XLIM_S = (1.0, 4.0)
#: The label V plays before the frames are taken, so the video shows a behaviour.
GIF_LABEL_NAME = "nodding"
#: Height of the camera + space plot dock in the GIF frames.
TOP_DOCK_HEIGHT_PX = 520


def wait(ms: int) -> None:
    timer = QElapsedTimer()
    timer.start()
    while timer.elapsed() < ms:
        QApplication.processEvents()


def rect_in(widget: QWidget, root: QWidget) -> QRect:
    return QRect(widget.mapTo(root, QPoint(0, 0)), widget.size())


def union(rects: list[QRect]) -> QRect:
    out = QRect(rects[0])
    for r in rects[1:]:
        out = out.united(r)
    return out


def font(size_pt: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size_pt * SCALE)
    except OSError:
        return ImageFont.load_default()


def annotate(img: Image.Image, boxes: list[tuple[QRect, str, str]]) -> Image.Image:
    """Coloured boxes by family plus a legend strip, as the curation grid figures."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for rect, colour, _ in boxes:
        x0, y0 = (rect.left() - PAD) * SCALE, (rect.top() - PAD) * SCALE
        x1, y1 = (rect.right() + PAD) * SCALE, (rect.bottom() + PAD) * SCALE
        draw.rounded_rectangle((x0, y0, x1, y1), radius=3 * SCALE, outline=colour, width=2 * SCALE)

    f = font(13)
    sw, gap, line_h = 12 * SCALE, 18 * SCALE, 24 * SCALE
    lines: list[list[tuple[str, str]]] = [[]]
    width = 8 * SCALE
    for _, colour, text in boxes:
        if not text:
            continue
        w = sw + 5 * SCALE + draw.textlength(text, font=f) + gap
        if lines[-1] and width + w > img.width:
            lines.append([])
            width = 8 * SCALE
        lines[-1].append((colour, text))
        width += w
    legend_h = line_h * len(lines) + 6 * SCALE
    out = Image.new("RGB", (img.width, img.height + legend_h), "#1e1e1e")
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    for row, line in enumerate(lines):
        x = 8 * SCALE
        y = img.height + 3 * SCALE + line_h * row + line_h // 2
        for colour, text in line:
            draw.rectangle((x, y - sw // 2, x + sw, y + sw // 2), fill=colour)
            x += sw + 5 * SCALE
            draw.text((x, y), text, fill="white", font=f, anchor="lm")
            x += draw.textlength(text, font=f) + gap
    return out


def draw_cursor(img: Image.Image, tip: tuple[int, int]) -> None:
    """A mouse-pointer arrow with its tip at *tip* (image pixels)."""
    x, y = tip
    s = 14 * SCALE
    pts = [
        (x, y),
        (x, y + s * 1.4),
        (x + s * 0.38, y + s * 1.05),
        (x + s * 0.62, y + s * 1.5),
        (x + s * 0.85, y + s * 1.38),
        (x + s * 0.6, y + s * 0.95),
        (x + s, y + s * 0.95),
    ]
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, fill="white", outline="black", width=2 * SCALE)


def caption(img: Image.Image, text: str) -> Image.Image:
    bar = 34 * SCALE
    out = Image.new("RGB", (img.width, img.height + bar), "#1e1e1e")
    out.paste(img.convert("RGB"), (0, 0))
    ImageDraw.Draw(out).text((12 * SCALE, img.height + bar // 2), text, fill="white", font=font(16), anchor="lm")
    return out


def qimage_to_pil(qimg) -> Image.Image:
    qimg = qimg.convertToFormat(qimg.Format_RGBA8888)
    ptr = qimg.constBits()
    ptr.setsize(qimg.sizeInBytes())
    return Image.frombuffer("RGBA", (qimg.width(), qimg.height()), bytes(ptr), "raw", "RGBA", 0, 1)


def launch_like_cli() -> tuple[QApplication, EthographMainWindow, MetaWidget]:
    # The app is returned so it outlives this call: dropping it would delete every widget.
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply_theme(app)
    install_dialog_fitter(app)
    shell = EthographMainWindow()
    meta = MetaWidget(shell)
    shell.attach_meta_widget(meta)
    return app, shell, meta


def load_template(meta: MetaWidget, key: str) -> None:
    """What the start page's template card does, without its dialogs."""
    t = resolve_dataset_paths(key)
    io = meta.io_widget
    io._clear_all_line_edits()
    for field, edit in (
        ("nc_file_path", io.nc_file_path_edit),
        ("video_folder", io.video_folder_edit),
        ("audio_folder", io.audio_folder_edit),
        ("pose_folder", io.pose_folder_edit),
    ):
        if t.get(field):
            edit.setText(t[field])
            setattr(meta.app_state, field, t[field])
    if t.get("import_labels"):
        io.import_labels_checkbox.setChecked(True)
    if t.get("library_geometry"):
        meta.app_state.space_library_geometry = t["library_geometry"]
    meta.data_widget.on_load_clicked()


def play_label(meta: MetaWidget, name: str) -> None:
    """Select the current trial's first label called *name* and press V."""
    lw = meta.labels_widget
    label_id = next(lid for lid, m in lw._mappings.items() if isinstance(lid, int) and m.get("name") == name)
    df = meta.app_state.label_intervals
    rows = df.index[df["labels"] == label_id]
    if len(rows) == 0:
        raise RuntimeError(f"no {name!r} label in trial {meta.app_state.trials_sel}")
    lw.current_labels = label_id
    lw.current_labels_pos = int(rows[0])
    lw.current_labels_is_prediction = False
    lw._play_segment()


def panel_to_click(meta: MetaWidget, kind: str, feature: str | None = None):
    matches = [
        reg
        for reg in meta.active_panels._regs
        if reg.kind == kind
        and (feature is None or (reg.plot is not None and reg.plot.panel_state.get("feature") == feature))
    ]
    # A freshly added panel can report isVisible() False for a moment; prefer a visible one.
    for reg in sorted(matches, key=lambda r: not r.widget.isVisible()):
        return reg
    raise RuntimeError(f"no visible {kind} panel" + (f" showing {feature}" if feature else ""))


def click_panel_gif(shell, meta, sections: GridSectionContainer) -> None:
    sections._expand(0)  # Data: the context-sensitive section
    wait(300)
    steps = [
        (PanelKind.LINEPLOT, "speed", "Click the speed plot → its feature and dimensions"),
        (PanelKind.SPACE, None, "Click the space plot → its axes and colouring"),
        (PanelKind.VIDEO, None, "Click the video → pose overlay and crop"),
    ]
    frames = []
    for kind, feature, text in steps:
        reg = panel_to_click(meta, kind, feature)
        meta.active_panels.set_active(reg)
        wait(600)
        img = qimage_to_pil(capture_widget(shell, SCALE))
        centre = rect_in(reg.widget, shell).center()
        draw_cursor(img, (centre.x() * SCALE, centre.y() * SCALE))
        frames.append(caption(img, text))
    frames[0].save(
        OUT_DIR / "sidebar_click_panel.gif",
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=False,
    )


def capture_with_popup(shell, popup: QWidget | None) -> Image.Image:
    """The window, with a Qt.Popup window (a separate top-level) composited in place."""
    img = qimage_to_pil(capture_widget(shell, SCALE))
    if popup is not None and popup.isVisible():
        shot = qimage_to_pil(capture_widget(popup, SCALE))
        origin = popup.mapToGlobal(QPoint(0, 0)) - shell.mapToGlobal(QPoint(0, 0))
        img.paste(shot, (origin.x() * SCALE, origin.y() * SCALE))
    return img


def select_popup_row(popup, name: str) -> None:
    """Highlight the popup row for source *name*, as the user would before Enter."""
    from ethograph.gui.source_popup import _ROLE_NAME

    rows = popup._list
    for i in range(rows.count()):
        if rows.item(i).data(_ROLE_NAME) == name:
            rows.setCurrentRow(i)
            return
    raise RuntimeError(f"no popup row named {name!r}")


def add_panel_gif(shell, meta) -> None:
    """Start from the camera alone and add a 3D space plot and a speed line plot."""
    pc = meta.plot_container
    dw = meta.data_widget
    for ribbon in pc.label_ribbons():
        pc.remove_panel(ribbon)
    for sp in list(dw.space_plots):
        dw.remove_space_plot(sp)
    for reg in list(meta.active_panels._regs):
        if reg.kind in PanelKind.FEATURE:
            pc.remove_panel(reg.widget)
    wait(1000)

    button = shell.bottom_bar.add_panel_btn
    popup = meta.source_popup
    frames = []

    def frame(text: str, cursor_on: QWidget | None = None, with_popup: bool = False) -> None:
        img = capture_with_popup(shell, popup if with_popup else None)
        if cursor_on is not None:
            centre = rect_in(cursor_on, shell).center()
            draw_cursor(img, (centre.x() * SCALE, centre.y() * SCALE))
        frames.append(caption(img, text))

    video = panel_to_click(meta, PanelKind.VIDEO)
    meta.active_panels.set_active(video)  # the sidebar shows the camera's settings
    wait(400)
    frame("Only the camera is open. Click Add panel, bottom-left (or Shift+N)…", cursor_on=button)

    meta.show_source_popup(button)
    wait(400)
    select_popup_row(popup, "position")
    wait(200)
    frame("…pick a source (position), press Enter, choose Space (3D)…", with_popup=True)
    popup.hide()
    meta._create_panel_for_source("feature", "position", "Space (3D)")
    wait(1500)
    frame("…and the space plot opens.")

    meta.show_source_popup(button)
    wait(400)
    select_popup_row(popup, "speed")
    wait(200)
    frame("Again for speed → Lineplot…", with_popup=True)
    popup.hide()
    meta._create_panel_for_source("feature", "speed", "Lineplot")
    wait(1000)
    for reg in meta.active_panels._regs:
        if reg.kind == PanelKind.LINEPLOT:
            reg.widget.vb.setXRange(*GIF_XLIM_S, padding=0)
    wait(800)
    frame("…a line plot. Each panel closes from the x in its title bar.")

    frames[0].save(
        OUT_DIR / "add_panel.gif",
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=False,
    )


def individual_png(shell, meta, sections: GridSectionContainer) -> None:
    dock = shell._sidebar_dock
    sections._expand(0)
    wait(300)
    buttons = union([rect_in(b, dock) for b in sections._buttons])
    group = rect_in(meta.data_panel.individual_groupbox, dock)
    img = qimage_to_pil(capture_widget(dock, SCALE))
    img = img.crop((0, 0, img.width, (group.bottom() + 12) * SCALE))
    annotate(
        img,
        [
            (buttons, BLUE, "Section buttons"),
            (group, ORANGE, "Individual: whose data is shown and whom new labels belong to"),
        ],
    ).save(OUT_DIR / "sidebar_individual_annotated.png")


def data_png(shell, meta, sections: GridSectionContainer) -> None:
    """The Data section after clicking the speed plot."""
    dock = shell._sidebar_dock
    sections._expand(0)
    meta.active_panels.set_active(panel_to_click(meta, PanelKind.LINEPLOT, "speed"))
    wait(500)
    dp = meta.data_panel
    buttons = union([rect_in(b, dock) for b in sections._buttons])
    group = rect_in(dp.individual_groupbox, dock)
    coords = rect_in(dp.coords_groupbox, dock)
    bottom = max(rect_in(w, dock).bottom() for w in meta.context_panel.findChildren(QWidget) if w.isVisible())
    img = qimage_to_pil(capture_widget(dock, SCALE))
    img = img.crop((0, 0, img.width, (bottom + 12) * SCALE))
    annotate(
        img,
        [
            (buttons, BLUE, "Section buttons"),
            (group, ORANGE, "Individual: always on top"),
            (coords, GREEN, "The clicked panel's feature and dimensions"),
            (rect_in(meta.plot_settings_widget.axes_groupbox, dock), TEAL, "The clicked panel's y-axis"),
        ],
    ).save(OUT_DIR / "sidebar_data_annotated.png")


def cover_png(shell, meta) -> None:
    """The start page as ``ethograph launch`` shows it, before anything is loaded."""
    page = CoverPage(shell, meta.io_widget)
    page.resize(*WINDOW)
    page.show()
    wait(1500)
    boxes = [
        (rect_in(page._template_card, page), BLUE, "1  Template datasets"),
        (rect_in(page._drop_card, page), ORANGE, "2  Drag & drop"),
        (rect_in(page._custom_card, page), GREEN, "3  Custom set-up"),
        (rect_in(page._load_bar, page), TEAL, "Load (drag & drop and custom)"),
        (rect_in(page._project_bar, page), PURPLE, "Project folder: where labels and drops are kept"),
    ]
    annotate(qimage_to_pil(capture_widget(page, SCALE)), boxes).save(OUT_DIR / "cover_page.png")
    page.close()


def labels_png(shell, meta, sections: GridSectionContainer) -> None:
    dock = shell._sidebar_dock
    sections._expand(1)
    wait(300)
    lw = meta.labels_widget
    buttons = union([rect_in(b, dock) for b in sections._buttons])
    mode_row = union([rect_in(lw.labelling_mode_combo, dock), rect_in(lw.label_overlay_combo, dock)])
    mode_row.setLeft(mode_row.left() - 44)
    headers = [
        union([rect_in(s["checkbox"], dock), rect_in(s["label"], dock)]) for _, s in sorted(lw._branch_sections.items())
    ]
    table0 = rect_in(lw._branch_sections[0]["table"], dock)
    img = qimage_to_pil(capture_widget(dock, SCALE))
    img = img.crop((0, 0, img.width, (rect_in(lw.curation_panel, dock).bottom() + 12) * SCALE))
    boxes = [
        (buttons, BLUE, "Section buttons"),
        (mode_row, ORANGE, "Mode and overlay"),
        (headers[0], PURPLE, "Branch: click the name to make it the one you edit"),
        *[(h, PURPLE, "") for h in headers[1:]],
        (table0, GREEN, "Classes and their keys"),
    ]
    annotate(img, boxes).save(OUT_DIR / "sidebar_labels_annotated.png")


def main() -> None:
    _app, shell, meta = launch_like_cli()
    notify.SUPPRESS = True  # no toast in a frame
    cover_png(shell, meta)
    # Never write this run's panels into the template's own local_settings.yaml,
    # and start from the data-availability defaults rather than a saved layout.
    meta.app_state._layout_snapshot_provider = None
    real_apply = meta.apply_saved_panel_layout
    meta.apply_saved_panel_layout = lambda: setattr(meta.app_state, "panel_layout", None) or real_apply()
    load_template(meta, "moll2025")
    shell.resize(*WINDOW)
    shell.show()
    wait(3000)
    meta.reset_panels()  # Ctrl+R: every panel and video rebuilt in place
    wait(2000)
    # The camera and the space plot get the taller share of the window.
    shell.resizeDocks([shell._video_dock], [TOP_DOCK_HEIGHT_PX], Qt.Vertical)
    # Exactly three panels: the camera, the speed line plot, the 3D space plot.
    pc = meta.plot_container
    for ribbon in pc.label_ribbons():
        pc.remove_panel(ribbon)
    has_speed = any(
        reg.kind == PanelKind.LINEPLOT and reg.plot is not None and reg.plot.panel_state.get("feature") == "speed"
        for reg in meta.active_panels._regs
    )
    if not has_speed:
        meta._create_panel_for_source("feature", "speed", "Lineplot")
    for reg in list(meta.active_panels._regs):
        if reg.kind == PanelKind.SPACE and reg.widget.isVisible():
            reg.widget.configure(view_3d=True)
    wait(500)
    # Re-apply the window: the removed ribbon may have been the x-link master.
    meta.navigation_widget._update_viewport_for_scope()
    for reg in meta.active_panels._regs:
        if reg.kind == PanelKind.LINEPLOT:
            reg.widget.vb.setXRange(*GIF_XLIM_S, padding=0)
    wait(500)
    play_label(meta, GIF_LABEL_NAME)  # V on the nodding label: the video shows the behaviour
    wait(3000)
    sections = shell.findChildren(GridSectionContainer)[0]

    add_panel_gif(shell, meta)
    click_panel_gif(shell, meta, sections)
    individual_png(shell, meta, sections)
    data_png(shell, meta, sections)
    labels_png(shell, meta, sections)
    print("wrote", OUT_DIR)
    shell.close()


if __name__ == "__main__":
    main()
