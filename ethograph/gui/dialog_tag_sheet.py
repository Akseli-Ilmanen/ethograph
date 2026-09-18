"""Print the tags the Detect stage reads.

Opened from the cover page's **Pre-recording tools**, and from **Tools ▸ Print
tag sheet…** once a dataset is loaded. Deliberately *not* from the Detect tab:
that tab needs a video, and a video means the tags were printed and glued to the
animals weeks earlier. The cover page is the only screen that exists before a
recording does, which is when this is actually needed.

The dialog is a table of *sheet rows* rather than one family and a count: a rig
usually wants a handful of big tags and many small ones, and printing one size
per page wastes most of the paper. The family column offers exactly what
:mod:`~ethograph.gui.pose_detect` can read, because a sheet of tags Ethograph
cannot detect is a trap rather than a feature. Mixing families on one sheet is a
**printing** convenience only — one detector reads one family, so two families in
one video mean two detection passes, and the dialog says so the moment a second
appears.

**Everything it can compute, it computes rather than asks**, which is what keeps
the dialog down to a handful of controls: modules per side and ID count come
from OpenCV, the quiet zone is always one module (floored at
:data:`~ethograph.gui.pose_tagsheet.MIN_QUIET_MM`) and never appears, the scale
bar is always printed, and the camera's pixel width is read off the loaded video
— or, with no video, **picked as a resolution** (``1920 × 1200``) rather than
typed as a bare width, since a resolution is what anyone knows about their
camera and a width is something they would have to work out. The one thing
nothing can derive — how many millimetres the frame covers — is the only
measurement asked for, and it is optional: it turns the advisory "Min mm" column
from "—" into a real size. Nothing here is a knob that needs calibrating before
a sheet is usable.

The preview, the PDF and the printer all go through
:func:`~ethograph.gui.pose_tagsheet.render_pages`, so what is on screen is what
comes out of the printer.
"""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QPainter, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ethograph.gui.notify import notify, notify_dialog
from ethograph.gui.pose_detect import TAG_FAMILIES, family_note
from ethograph.gui.pose_tagsheet import (
    MIN_PX_PER_MODULE,
    PAGE_SIZES_MM,
    SheetSpec,
    TagRow,
    TagSheetError,
    default_quiet_mm,
    dictionary_info,
    layout_sheet,
    min_tag_mm,
    print_sheet,
    render_pages,
    write_pdf,
)

#: Height the page preview is drawn at, in screen pixels.
PREVIEW_HEIGHT = 380

#: Supersampling of the preview. A 3 mm tag is ~4 px on a page-height preview,
#: so the modules only read at all if it is drawn big and scaled down.
PREVIEW_OVERSAMPLE = 3

#: Columns of the row table. "Min mm" is a readout, not an input — it is what
#: the camera figures below turn into, and the only number here that says
#: whether a size will actually work.
_COLUMNS = ("Family", "First ID", "Count", "Tag mm", "Min mm")

_COLUMN_TOOLTIPS = (
    "Which AprilTag family. Only the ones Ethograph can DETECT are offered —\n"
    "printing a family it cannot read would be a trap.\n\n"
    "tag36h11 is the default and has the most margin; the smaller families\n"
    "need less paper for the same pixels per module.",
    "The first ID printed in this row.",
    "How many consecutive IDs to print. Capped at what the family holds:\nthere is no wraparound.",
    "Printed side of the tag itself, black border included — NOT the white\n"
    "margin around it, which is one module wide and set for you.",
    f"Smallest size that still gives {MIN_PX_PER_MODULE:g} px per module in the camera\n"
    "below. '—' means the scene width has not been given.",
)

#: Camera resolutions offered when no video is loaded, as ``(label, width px)``.
#: Only the width is ever used; the full resolution is shown because that is how
#: a camera is described on its own datasheet.
CAMERA_PRESETS = (
    ("640 × 480", 640),
    ("1280 × 720", 1280),
    ("1280 × 1024", 1280),
    ("1920 × 1080", 1920),
    ("1920 × 1200", 1920),
    ("2048 × 1536", 2048),
    ("2592 × 1944", 2592),
    ("3840 × 2160", 3840),
)

#: Combo entry standing for "type the width yourself". Negative so it can never
#: collide with a real pixel width.
_CUSTOM_WIDTH = -1

#: Printing advice that decides whether a sheet works at all. Shown in the
#: dialog rather than in a manual: every one of these is a way to produce a
#: sheet that is the right size and still undetectable.
_MATERIALS = (
    "<b>Matte paper only</b> — gloss reflects the light source across the border "
    "and kills quad detection. <b>Laser, not inkjet</b> (inkjet wicks and softens "
    "0.5 mm module edges), <b>black cartridge only</b> (composite black from CMY "
    "fringes every edge), toner-save and edge smoothing <b>off</b>, and "
    "<b>glue the sheet to card</b> — the homography assumes a planar tag."
)

_VERIFY = (
    "Measure the printed 50 mm bar with calipers. If it does not read 50 mm the "
    "tags are wrong by the same ratio — reprint, never adjust the size to "
    "compensate."
)


@dataclass
class _RowWidgets:
    """The live widgets of one sheet row, so a change edits rather than rebuilds."""

    family: QComboBox
    first_id: QSpinBox
    count: QSpinBox
    tag_mm: QDoubleSpinBox
    minimum: QTableWidgetItem

    def to_row(self) -> TagRow:
        """The row, with its quiet zone computed rather than asked.

        One module of white is what the quad finder needs and what every
        reference sheet prints; there is no reading of the tag that a different
        value improves, so it is derived from the family and the tag size.
        """
        tag_mm = float(self.tag_mm.value())
        family = self.family.currentText()
        return TagRow(
            family=family,
            first_id=int(self.first_id.value()),
            count=int(self.count.value()),
            tag_mm=tag_mm,
            quiet_mm=default_quiet_mm(tag_mm, dictionary_info(family).modules),
        )


class TagSheetDialog(QDialog):
    """Compose a sheet of AprilTag fiducial tags and print or export it."""

    def __init__(self, app_state, family: str | None = None, image_width_px: float | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Print tag sheet")
        self.app_state = app_state
        #: Pixels across the loaded video **file** (never the on-screen texture,
        #: which may be a proxy) — it seeds the resolution picker rather than
        #: replacing it. See `_build_camera_group`.
        self._image_width_px = int(image_width_px or 0)
        self._rows: list[_RowWidgets] = []
        #: The last successful layout — empty while the settings describe no
        #: printable sheet, which is what disables the export buttons.
        self._pages: list[list] = []
        #: Suppresses the refresh while a row is being built widget by widget.
        self._loading = True

        self._build_ui()
        self._add_row(
            TagRow(
                family=family or app_state.detect_tag_family,
                tag_mm=float(app_state.tag_sheet_tag_mm),
            )
        )
        self._loading = False
        self._refresh()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_rows_group())
        layout.addWidget(self._build_page_group())
        layout.addWidget(self._build_camera_group())
        layout.addWidget(self._build_preview_group(), stretch=1)

        self.warnings = QLabel()
        self.warnings.setWordWrap(True)
        self.warnings.setTextFormat(Qt.RichText)
        layout.addWidget(self.warnings)

        materials = QLabel(_MATERIALS)
        materials.setWordWrap(True)
        materials.setTextFormat(Qt.RichText)
        materials.setStyleSheet("QLabel { font-size: 11px; }")
        materials.setToolTip(_VERIFY)
        layout.addWidget(materials)

        # Directly above the buttons, because it is a setting in the *system*
        # print dialog that no code here can set — and the single likeliest way
        # to end up with a sheet of subtly wrong tags. "Fit to printable area"
        # is on by default in most PDF viewers.
        scale_notice = QLabel(
            "⚠ Print at <b>Actual size / 100%</b> — never <b>Fit to printable area</b> or "
            "<b>Shrink oversized pages</b>. Then measure the printed 50 mm bar."
        )
        scale_notice.setWordWrap(True)
        scale_notice.setTextFormat(Qt.RichText)
        scale_notice.setToolTip(_VERIFY)
        layout.addWidget(scale_notice)

        buttons = QHBoxLayout()
        self.pdf_btn = QPushButton("Save PDF…")
        self.pdf_btn.setToolTip(
            "Vector PDF — the format to print from. In the viewer's print dialog\n"
            "choose Actual size / 100% (never 'Fit to printable area'), then\n"
            "check the scale bar with calipers."
        )
        self.pdf_btn.clicked.connect(self._on_save_pdf)
        buttons.addWidget(self.pdf_btn)

        self.print_btn = QPushButton("Print…")
        self.print_btn.setToolTip(
            "Send the sheet straight to a printer, through the system print\n"
            "dialog. Set the scale there to 100% / Actual size — page fitting\n"
            "is what silently resizes tags."
        )
        self.print_btn.clicked.connect(self._on_print)
        buttons.addWidget(self.print_btn)

        buttons.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        # Wide enough that the row table shows the "Min mm" readout without
        # scrolling — a warning nobody sees is not a warning.
        self.resize(720, 900)

    def _build_rows_group(self) -> QGroupBox:
        group = QGroupBox("Tags to print")
        group.setToolTip(
            "Only families Ethograph can DETECT are offered: a sheet of tags the\n"
            "Detect stage cannot read is a trap. (tag36h10 is deliberately absent —\n"
            "OpenCV renders it, but AprilTag 3 dropped the family, so it would\n"
            "print perfectly and never decode.)"
        )
        box = QVBoxLayout(group)

        self.row_table = QTableWidget(0, len(_COLUMNS))
        self.row_table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.row_table.verticalHeader().setVisible(False)
        self.row_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.row_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.row_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.row_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.row_table.setMaximumHeight(180)
        for column, tip in enumerate(_COLUMN_TOOLTIPS):
            self.row_table.horizontalHeaderItem(column).setToolTip(tip)
        box.addWidget(self.row_table)

        controls = QHBoxLayout()
        add_btn = QPushButton("Add row")
        add_btn.setToolTip("Another run of IDs at another size — a few big tags and many small ones on one sheet.")
        add_btn.clicked.connect(self._on_add_row)
        controls.addWidget(add_btn)
        remove_btn = QPushButton("Remove row")
        remove_btn.clicked.connect(self._on_remove_row)
        controls.addWidget(remove_btn)
        controls.addStretch()
        box.addLayout(controls)
        return group

    def _build_page_group(self) -> QGroupBox:
        group = QGroupBox("Page")
        first = QHBoxLayout(group)

        first.addWidget(QLabel("Size:"))
        self.page_combo = QComboBox()
        self.page_combo.addItems(list(PAGE_SIZES_MM))
        self.page_combo.setCurrentText(self.app_state.tag_sheet_page)
        self.page_combo.currentIndexChanged.connect(self._refresh)
        first.addWidget(self.page_combo)

        first.addWidget(QLabel("Margin:"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 50.0)
        self.margin_spin.setSingleStep(1.0)
        self.margin_spin.setDecimals(1)
        self.margin_spin.setSuffix(" mm")
        self.margin_spin.setValue(float(self.app_state.tag_sheet_margin_mm))
        self.margin_spin.setToolTip("Keep it at or above what your printer can actually reach.")
        self.margin_spin.valueChanged.connect(self._refresh)
        first.addWidget(self.margin_spin)

        self.labels_check = QCheckBox("Print IDs")
        self.labels_check.setChecked(bool(self.app_state.tag_sheet_labels))
        self.labels_check.setToolTip("Print each tag's ID beneath it, so a cut-out tag is still identifiable.")
        self.labels_check.toggled.connect(self._refresh)
        first.addWidget(self.labels_check)
        first.addStretch()
        return group

    def _build_camera_group(self) -> QGroupBox:
        """How big is big enough — a resolution, and a measurement nothing records.

        The resolution the sheet has to satisfy is the one that will **film the
        tags**, which is not necessarily anything currently open: a sheet is
        often printed before the recording exists, or for a second rig. So it
        is always an editable control, seeded from the loaded video when there
        is one.

        The seed is the video **file's** resolution (:func:`~ethograph.gui.pose_fill.video_size`),
        never what is on screen: with proxy playback the displayed texture is a
        480p re-encode, and seeding from it would inflate every minimum size by
        the proxy's scale factor without saying so.

        The scene width cannot be derived from anything — nothing in a video file
        says how many millimetres a pixel covers — so it is the one figure asked
        for, and it stays optional: it only fills in the advisory "Min mm" column.
        """
        group = QGroupBox("Will the tags be big enough? (optional)")
        group.setToolTip(
            f"min size = 8 modules × {MIN_PX_PER_MODULE:g} px × (scene width ÷ image width)\n\n"
            "Fills the 'Min mm' column above. Everything else works without it."
        )
        box = QHBoxLayout(group)

        box.addWidget(QLabel("Camera:"))
        box.addLayout(self._build_resolution_row())
        if self._image_width_px:
            seeded = QLabel("(from the video)")
            seeded.setStyleSheet("QLabel { font-size: 11px; }")
            seeded.setToolTip(
                "Taken from the loaded video file, not from what is on screen —\n"
                "proxy playback shows a 480p re-encode.\n\n"
                "Change it if these tags are for a different camera."
            )
            box.addWidget(seeded)

        box.addWidget(QLabel("Scene width in view:"))
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(0.0, 100000.0)
        self.fov_spin.setDecimals(1)
        self.fov_spin.setSuffix(" mm")
        self.fov_spin.setSpecialValueText("unknown")
        self.fov_spin.setValue(float(self.app_state.tag_sheet_fov_mm))
        self.fov_spin.setToolTip(
            "How wide the scene is, in millimetres, across the whole frame at the\n"
            "animal's distance — the arena, the cage, the dish. Nothing in a video\n"
            "file says this, so it is the one thing that has to be typed.\n\n"
            "Measure it once with a ruler in the frame."
        )
        self.fov_spin.valueChanged.connect(self._refresh)
        box.addWidget(self.fov_spin)
        box.addStretch()
        return group

    def _build_resolution_row(self) -> QHBoxLayout:
        """Pick a resolution, or type a width.

        A resolution is what anyone knows about their camera ("it's a 1920×1200
        camera"); a bare pixel *width* is a number they have to work out, and
        typing 1200 when they meant 1920 is a silent 60% error in the minimum
        size. So the presets are the normal path and the spin box appears only
        for **Custom**, on the same row.

        Only the width is used — a tag is judged across the frame — which the
        tooltip says, since otherwise 1920×1080 and 1920×1200 giving the same
        answer looks like a bug.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        # The video wins over the remembered setting, being the more specific
        # answer to "what is filming this"; either can still be overridden.
        seed = self._image_width_px or int(self.app_state.tag_sheet_camera_width_px)

        self.camera_combo = QComboBox()
        self.camera_combo.addItem("unknown", 0)
        for label, width in CAMERA_PRESETS:
            self.camera_combo.addItem(label, width)
        self.camera_combo.addItem("Custom…", _CUSTOM_WIDTH)
        self.camera_combo.setToolTip(
            "The resolution of the camera that will film these tags. Only the\n"
            "width counts — a tag is judged across the frame — so 1920×1080 and\n"
            "1920×1200 give the same answer.\n\n"
            "Seeded from the loaded video, and editable: the sheet may well be\n"
            "for a rig that has not recorded anything yet."
        )
        position = self.camera_combo.findData(seed)
        self.camera_combo.setCurrentIndex(position if position >= 0 else self.camera_combo.count() - 1)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        row.addWidget(self.camera_combo)

        self.camera_px_spin = QSpinBox()
        self.camera_px_spin.setRange(1, 100000)
        self.camera_px_spin.setSuffix(" px wide")
        self.camera_px_spin.setValue(max(seed, 1))
        self.camera_px_spin.setToolTip("Pixels across the frame.")
        self.camera_px_spin.valueChanged.connect(self._refresh)
        self.camera_px_spin.setVisible(self.camera_combo.currentData() == _CUSTOM_WIDTH)
        row.addWidget(self.camera_px_spin)
        return row

    def _on_camera_changed(self, _index: int) -> None:
        self.camera_px_spin.setVisible(self.camera_combo.currentData() == _CUSTOM_WIDTH)
        self._refresh()

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Preview")
        box = QVBoxLayout(group)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(PREVIEW_HEIGHT)
        self.preview.setToolTip("The page exactly as it will print — same painting code, same geometry.")
        box.addWidget(self.preview, stretch=1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Page:"))
        self.page_spin = QSpinBox()
        self.page_spin.setRange(1, 1)
        self.page_spin.valueChanged.connect(self._refresh_preview)
        controls.addWidget(self.page_spin)
        controls.addStretch()
        self.summary = QLabel()
        controls.addWidget(self.summary)
        box.addLayout(controls)
        return group

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _add_row(self, row: TagRow) -> None:
        info = dictionary_info(row.family)
        index = self.row_table.rowCount()
        self.row_table.insertRow(index)

        family = QComboBox()
        for name in TAG_FAMILIES:
            family.addItem(name, name)
            family.setItemData(family.count() - 1, family_note(name), Qt.ToolTipRole)
        family.setCurrentText(row.family)

        first_id = QSpinBox()
        first_id.setRange(0, info.n_ids - 1)
        first_id.setValue(row.first_id)

        count = QSpinBox()
        count.setRange(1, info.n_ids - row.first_id)
        count.setValue(min(row.count, info.n_ids - row.first_id))

        tag_mm = QDoubleSpinBox()
        tag_mm.setRange(0.5, 200.0)
        tag_mm.setDecimals(2)
        tag_mm.setSingleStep(0.5)
        tag_mm.setSuffix(" mm")
        tag_mm.setValue(row.tag_mm)

        minimum = QTableWidgetItem("—")
        minimum.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))

        widgets = _RowWidgets(family, first_id, count, tag_mm, minimum)
        self._rows.append(widgets)
        for column, widget in enumerate((family, first_id, count, tag_mm)):
            self.row_table.setCellWidget(index, column, widget)
        self.row_table.setItem(index, len(_COLUMNS) - 1, minimum)

        family.currentIndexChanged.connect(lambda _index, w=widgets: self._on_family_changed(w))
        first_id.valueChanged.connect(lambda _value, w=widgets: self._on_first_id_changed(w))
        tag_mm.valueChanged.connect(self._refresh)
        count.valueChanged.connect(self._refresh)

    def _on_add_row(self) -> None:
        last = self._rows[-1].to_row() if self._rows else TagRow()
        self._add_row(TagRow(family=last.family, tag_mm=last.tag_mm))
        self._refresh()

    def _on_remove_row(self) -> None:
        if len(self._rows) <= 1:
            notify("A sheet needs at least one row.", "warning")
            return
        index = self.row_table.currentRow()
        if index < 0:
            index = len(self._rows) - 1
        self.row_table.removeRow(index)
        del self._rows[index]
        self._refresh()

    def _on_family_changed(self, widgets: _RowWidgets) -> None:
        """A new family changes what IDs exist and how wide a module is."""
        info = dictionary_info(widgets.family.currentText())
        widgets.first_id.setMaximum(info.n_ids - 1)
        widgets.count.setMaximum(info.n_ids - widgets.first_id.value())
        self._refresh()

    def _on_first_id_changed(self, widgets: _RowWidgets) -> None:
        info = dictionary_info(widgets.family.currentText())
        widgets.count.setMaximum(info.n_ids - widgets.first_id.value())
        self._refresh()

    # ------------------------------------------------------------------
    # The sheet
    # ------------------------------------------------------------------

    def spec(self) -> SheetSpec:
        """What the widgets currently describe."""
        return SheetSpec(
            rows=[widgets.to_row() for widgets in self._rows],
            page=self.page_combo.currentText(),
            margin_mm=float(self.margin_spin.value()),
            labels=self.labels_check.isChecked(),
        )

    def _camera_width_px(self) -> float:
        """Pixels across the frame, from the picker.

        ``0`` means unknown, which is what makes the "Min mm" column read "—"
        rather than a size derived from a guessed camera.
        """
        chosen = int(self.camera_combo.currentData())
        return float(self.camera_px_spin.value() if chosen == _CUSTOM_WIDTH else chosen)

    def _store_settings(self) -> None:
        """Page setup and the rig persist; the rows themselves are per-sheet.

        Only ever called for a sheet that laid out, so a size typed on the way
        to a valid one is not what the next sheet opens on.
        """
        self.app_state.tag_sheet_page = self.page_combo.currentText()
        self.app_state.tag_sheet_margin_mm = float(self.margin_spin.value())
        self.app_state.tag_sheet_labels = self.labels_check.isChecked()
        self.app_state.tag_sheet_camera_width_px = int(self._camera_width_px())
        self.app_state.tag_sheet_fov_mm = float(self.fov_spin.value())
        if self._rows:
            self.app_state.tag_sheet_tag_mm = float(self._rows[0].tag_mm.value())
            # The sheet and the Detect tab share one family setting rather than
            # keeping two that can disagree: printing a family and then looking
            # for a different one is the mistake worth designing out.
            self.app_state.detect_tag_family = self._rows[0].family.currentText()

    def _refresh(self, *_args) -> None:
        """Re-lay out, redraw and re-warn — everything a change can affect."""
        if self._loading:
            return
        self._refresh_minimums()
        try:
            pages = layout_sheet(self.spec())
        except TagSheetError as e:
            self._pages = []
            self.preview.setPixmap(QPixmap())
            self.preview.setText(str(e))
            self.summary.setText("")
            self.page_spin.setEnabled(False)
            self._set_exports_enabled(False)
            self._refresh_warnings()
            return
        self._pages = pages
        self._store_settings()
        self.page_spin.setEnabled(len(pages) > 1)
        blocked = self.page_spin.blockSignals(True)
        self.page_spin.setMaximum(len(pages))
        self.page_spin.setValue(min(self.page_spin.value(), len(pages)))
        self.page_spin.blockSignals(blocked)
        self._set_exports_enabled(True)
        total = sum(len(page) for page in pages)
        self.summary.setText(f"{total} tag(s) over {len(pages)} page(s)")
        self._refresh_warnings()
        self._refresh_preview()

    def _set_exports_enabled(self, enabled: bool) -> None:
        for button in (self.pdf_btn, self.print_btn):
            button.setEnabled(enabled)

    def _minimum_for(self, widgets: _RowWidgets) -> float | None:
        info = dictionary_info(widgets.family.currentText())
        return min_tag_mm(info.modules, self._camera_width_px(), float(self.fov_spin.value()))

    def _refresh_minimums(self) -> None:
        """Fill the read-only minimum column, reddening anything below it."""
        for widgets in self._rows:
            minimum = self._minimum_for(widgets)
            if minimum is None:
                widgets.minimum.setText("—")
                widgets.minimum.setForeground(QBrush())
                continue
            widgets.minimum.setText(f"{minimum:.1f}")
            too_small = float(widgets.tag_mm.value()) < minimum
            widgets.minimum.setForeground(QBrush(QColor("#d05050")) if too_small else QBrush())

    def _refresh_warnings(self) -> None:
        """Say the things that decide whether the sheet works, and nothing else."""
        messages: list[str] = []
        families = self.spec().families
        if len(families) > 1:
            messages.append(
                "Mixing families is a <b>printing</b> convenience only: one detector reads one "
                f"family at a time, so tags from {len(families)} families in one video need "
                f"{len(families)} detection passes."
            )
        undersized = []
        for widgets in self._rows:
            minimum = self._minimum_for(widgets)
            if minimum is not None and float(widgets.tag_mm.value()) < minimum:
                undersized.append(
                    f"{widgets.family.currentText()} at {widgets.tag_mm.value():g} mm (needs {minimum:.1f} mm)"
                )
        if undersized:
            messages.append(
                f"Too small for the camera given: {', '.join(undersized)} — under {MIN_PX_PER_MODULE:g} px per module."
            )
        self.warnings.setText("<br>".join(f"⚠ {message}" for message in messages))
        self.warnings.setVisible(bool(messages))

    def _refresh_preview(self, *_args) -> None:
        """Draw one page into a pixmap with the routine that drives the print."""
        if not self._pages:
            return
        spec = self.spec()
        page_w, page_h = spec.page_mm()
        scale = PREVIEW_HEIGHT * PREVIEW_OVERSAMPLE / page_h
        pixmap = QPixmap(int(page_w * scale), int(page_h * scale))
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        index = min(self.page_spin.value(), len(self._pages)) - 1
        try:
            render_pages(painter, spec, self._pages[index : index + 1], scale)
        finally:
            painter.end()
        self.preview.setText("")
        self.preview.setPixmap(pixmap.scaledToHeight(PREVIEW_HEIGHT, Qt.SmoothTransformation))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _on_save_pdf(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Save tag sheet", "tag_sheet.pdf", "PDF (*.pdf)")
        if not path:
            return
        try:
            pages = write_pdf(self.spec(), path)
        except TagSheetError as e:
            notify_dialog(str(e), "error", "Tag sheet", self)
            return
        notify(f"Wrote {pages} page(s) to {path}. Print at 100% scale and check the scale bar.", "info")

    def _on_print(self) -> None:
        """Straight to a printer, through the system dialog.

        The enums are spelled out in full (``QPrinter.PrinterMode.HighResolution``):
        qtpy promotes the unscoped names for QtCore/QtGui/QtWidgets but **not**
        for QtPrintSupport, so the short spelling raises ``AttributeError`` on
        PyQt6 — inside a click handler, which shows up as the button doing
        nothing at all.
        """
        from qtpy.QtPrintSupport import QPrintDialog, QPrinter, QPrinterInfo

        if not QPrinterInfo.availablePrinters():
            # Qt's print dialog simply refuses to open without one, which would
            # otherwise be another silent dead button.
            notify_dialog(
                "No printer is installed. Save the PDF instead and print it from a "
                "PDF viewer — at 100% scale, with page fitting off.",
                "warning",
                "Tag sheet",
                self,
            )
            return

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        if QPrintDialog(printer, self).exec() != QDialog.DialogCode.Accepted:
            return
        try:
            pages = print_sheet(self.spec(), printer)
        except TagSheetError as e:
            notify_dialog(str(e), "error", "Tag sheet", self)
            return
        notify(f"Sent {pages} page(s) to the printer. Check the scale bar before using the tags.", "info")
