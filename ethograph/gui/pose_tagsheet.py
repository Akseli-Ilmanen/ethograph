"""Printing the tags the Detect stage reads.

A tag detector is only as good as the paper it reads, and the paper is made
outside the GUI today — a bitmap exported from somewhere, scaled by a print
dialog, measured by nobody. Two things go wrong, and both are silent:

**The print is rescaled.** "Fit to page", "shrink oversized pages" and driver
margins all resize a page by a few percent without saying so. At 3 mm per tag
nobody sees it, and the whole sheet is then the wrong size for the rig it was
designed for. The answer is a **printed scale bar**: a 50 mm rule on the sheet
itself, so a caliper says in one measurement whether the print is honest. It is
always printed — there is no option to leave it off, since the footer it sits in
is reserved either way — and the sheet carries its own settings as a caption, so
a sheet found on a bench a year later is still self-documenting.

**The tag is too small to decode.** ``modules × 5`` pixels is the design figure
(below ~3 px per module detection falls off a cliff), and modules is *not* the
data grid: ``tag36h11`` is a 6×6 grid plus a black border module each side, so
8 modules of paper, while ``tag16h5`` is 4×4 plus the border, so 6. That
difference — 25% less printed side for the same pixels per module — is the whole
reason to choose a smaller family. Give the dialog a camera width and a field of
view and it converts it into millimetres, per row.

Which families, and what OpenCV is still for
--------------------------------------------
The sheet prints exactly what :mod:`~ethograph.gui.pose_detect` can read —
:data:`~ethograph.gui.pose_detect.TAG_FAMILIES` — and nothing else, because a
sheet of tags Ethograph cannot detect is a trap rather than a feature. That rules
out ``tag36h10`` in particular: OpenCV renders it perfectly well (2320 IDs), but
AprilTag 3 dropped the family, so it would print beautifully and never decode.

Mixing families on one sheet is a **printing** convenience only. One detector
reads one family, so two families in one video mean two detection passes, and
the dialog says so as soon as a second appears.

The tags are *rendered* by ``cv2.aruco.generateImageMarker`` through the matching
``DICT_APRILTAG_*``, since ``pupil-apriltags`` exports no generator. The two
libraries agree on IDs — pinned by a per-family round-trip test, because that
agreement is the contract between the printer and the detector and it is what a
version bump would break. Every dictionary fact is **read from OpenCV**
(:func:`dictionary_info`) rather than tabulated: an ID count copied into Python
is a number that can drift from the library actually drawing the tags.

Geometry, not raster
--------------------
:func:`marker_grid` asks OpenCV for **one array element per printed module**
(``generateImageMarker`` at exactly ``modules`` pixels), and :func:`black_rects`
turns that into filled rectangles. Nothing is rasterised at any point, so
nothing can soften a module edge — the one thing a tag cannot survive.

Output is **PDF** (:func:`write_pdf`) and a printer (:func:`print_sheet`), and
deliberately nothing else. SVG was offered once and removed: Windows has no
native SVG print path and rasterises it on the way to the printer — giving back
exactly the soft edges the vector path exists to avoid — and every editor that
opens an SVG opens a vector PDF too, so it bought nothing but a button that
invites the wrong file to be printed. Both outputs and the on-screen preview go
through the **same** :func:`render_pages` routine: a preview that can disagree
with the print would be worse than none, exactly as in
:mod:`~ethograph.gui.pose_detect_preview`.

**The scale bar and caption are drawn on every page**, not just the last. Pages
are separated the moment they are printed, and reserving the footer
conditionally would make the packing depend on a page count that depends on the
packing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from qtpy.QtCore import QMarginsF, QRectF, Qt
from qtpy.QtGui import QBrush, QColor, QFont, QPageLayout, QPageSize, QPainter, QPdfWriter

from ethograph.gui.pose_detect import (
    DEFAULT_TAG_FAMILY,
    PX_PER_MODULE,
    TAG_DICTIONARIES,
    check_family,
)

#: Pixels per printed module the sizing advice is built on — the decoder's own
#: figure, re-exported rather than repeated. A printing rule that could drift
#: from what the detector actually needs would be worse than no rule.
MIN_PX_PER_MODULE = PX_PER_MODULE

#: Border modules OpenCV renders around the data grid. It is part of the tag —
#: the quiet zone is the *white* margin outside it — and cropping it away is one
#: of the classic ways to make a sheet undetectable.
BORDER_BITS = 1

#: Smallest white margin around a tag, **enforced** rather than merely
#: defaulted: the quad finder needs white to see the tag's black border against,
#: and a sub-millimetre gap does not survive printing and cutting. The default
#: is one module wide, but a module of a 2 mm tag is 0.25 mm, which is what this
#: floor is for.
MIN_QUIET_MM = 1.0

#: Height reserved under a tag for its printed ID.
LABEL_MM = 2.6

#: How far text may be shrunk to fit before it falls back to a shorter wording.
#: Deliberately close to 1: a label squeezed until it touches its neighbours is
#: no more readable than a clipped one.
MIN_LABEL_SHRINK = 0.9

#: The printed rule. 50 mm is long enough that a 2% rescale is a millimetre —
#: visible on any ruler, unmissable on calipers.
SCALE_BAR_MM = 50.0

#: Bottom strip kept clear on every page for the scale bar and caption. Two
#: lines: the rule, then the caption across the full width — the caption names
#: every row on the sheet and is the thing most likely to run long.
FOOTER_MM = 14.0

#: Left gutter kept clear on every page for each band's printed size. The
#: caption already names every size on the sheet, but a sheet gets **cut up**:
#: once a strip of tags is separated from the footer, the size printed beside it
#: is the only record of what was chosen.
ROW_LABEL_MM = 14.0

#: PDF device resolution. Device units are ``mm × resolution / 25.4``.
PDF_RESOLUTION = 1200

#: Page sizes offered, in millimetres, with the ``QPageSize`` id to print them.
PAGE_SIZES_MM: dict[str, tuple[float, float]] = {"A4": (210.0, 297.0), "Letter": (215.9, 279.4)}

#: Rounding slack for "does this band still fit", in mm. Cell heights are sums
#: of user-typed decimals, so an exact comparison drops a row that fits.
_EPS = 1e-9


class TagSheetError(Exception):
    """Raised when a sheet cannot be laid out or a row is impossible."""


# ----------------------------------------------------------------------
# The family
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DictionaryInfo:
    """What one family costs in paper and how many IDs it holds.

    Read from OpenCV, never tabulated: an ID count written down here is a number
    that can drift from the library that will actually draw the print.
    """

    #: The AprilTag family name, as :mod:`~ethograph.gui.pose_detect` spells it.
    family: str
    #: The OpenCV dictionary that renders it.
    name: str
    #: Data grid per side — ``6`` for ``tag36h11``, ``4`` for ``tag16h5``.
    marker_size: int
    n_ids: int

    @property
    def modules(self) -> int:
        """Printed cells per side, border included — what sets the paper size."""
        return self.marker_size + 2 * BORDER_BITS

    @property
    def short_name(self) -> str:
        return self.family


@lru_cache(maxsize=None)
def dictionary_info(family: str = DEFAULT_TAG_FAMILY) -> DictionaryInfo:
    """Describe a tag family, importing OpenCV only now."""
    import cv2

    check_family(family)
    name = TAG_DICTIONARIES[family]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    return DictionaryInfo(
        family=family,
        name=name,
        marker_size=int(dictionary.markerSize),
        n_ids=int(dictionary.bytesList.shape[0]),
    )


def marker_grid(family: str, marker_id: int) -> np.ndarray:
    """One array element per printed module: ``0`` black, ``255`` white.

    Asking ``generateImageMarker`` for exactly ``modules`` pixels is what keeps
    the whole pipeline free of raster — every element here becomes geometry.
    """
    import cv2

    info = dictionary_info(family)
    if not 0 <= marker_id < info.n_ids:
        raise TagSheetError(f"{family} holds IDs 0–{info.n_ids - 1}; {marker_id} is outside it.")
    return np.asarray(
        cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, info.name)),
            int(marker_id),
            info.modules,
            borderBits=BORDER_BITS,
        )
    )


def min_tag_mm(modules: int, image_width_px: float, fov_width_mm: float) -> float | None:
    """Smallest printed size that still gives :data:`MIN_PX_PER_MODULE`.

    Takes *modules* rather than a family because that is the only property of
    the family it uses, and the families differ: ``tag16h5`` needs 6 modules of
    paper where ``tag36h11`` needs 8, which is the whole reason to pick one.

    ``None`` when the rig is unknown — never a guessed field of view, since a
    minimum size invented from nothing is worse than no minimum at all.
    """
    if image_width_px <= 0 or fov_width_mm <= 0:
        return None
    return modules * MIN_PX_PER_MODULE * (fov_width_mm / image_width_px)


def default_quiet_mm(tag_mm: float, modules: int) -> float:
    """One module of white around the tag, but never under :data:`MIN_QUIET_MM`."""
    return max(tag_mm / modules, MIN_QUIET_MM)


# ----------------------------------------------------------------------
# What to print
# ----------------------------------------------------------------------


@dataclass
class TagRow:
    """A run of consecutive IDs at one size.

    A sheet is a list of these rather than a single count, because a rig usually
    wants a few big tags and many small ones, and reprinting one size at a time
    wastes most of a page.
    """

    family: str = DEFAULT_TAG_FAMILY
    first_id: int = 0
    count: int = 24
    tag_mm: float = 4.0
    quiet_mm: float = MIN_QUIET_MM

    def validate(self) -> DictionaryInfo:
        """Check the row against its family, returning what it describes."""
        info = dictionary_info(check_family(self.family))
        if self.count < 1:
            raise TagSheetError("A row must hold at least one tag.")
        if self.first_id < 0 or self.first_id + self.count > info.n_ids:
            raise TagSheetError(
                f"{self.family} holds {info.n_ids} IDs; {self.count} tags from ID {self.first_id} runs past the end."
            )
        if self.tag_mm <= 0:
            raise TagSheetError("Tag size must be positive.")
        if self.quiet_mm < MIN_QUIET_MM:
            raise TagSheetError(
                f"The quiet zone cannot go below {MIN_QUIET_MM:g} mm: the quad finder needs white "
                "around the tag's black border, and a sub-millimetre gap does not survive printing "
                "and cutting."
            )
        return info


@dataclass
class SheetSpec:
    """A whole sheet: the rows plus how the page is set up."""

    rows: list[TagRow] = field(default_factory=lambda: [TagRow()])
    page: str = "A4"
    margin_mm: float = 10.0
    labels: bool = True

    def page_mm(self) -> tuple[float, float]:
        if self.page not in PAGE_SIZES_MM:
            raise TagSheetError(f"Unknown page size {self.page!r}; expected one of {sorted(PAGE_SIZES_MM)}")
        return PAGE_SIZES_MM[self.page]

    @property
    def families(self) -> list[str]:
        """The families used, in order of first appearance.

        More than one is legal on paper and costs a **detection pass each**: one
        :class:`~ethograph.gui.pose_detect.AprilTagDetector` reads one family, so
        the dialog says so as soon as a second appears.
        """
        return list(dict.fromkeys(row.family for row in self.rows))


@dataclass(frozen=True)
class Placement:
    """One tag on one page, positioned by the **tag's** top-left corner in mm."""

    family: str
    marker_id: int
    x_mm: float
    y_mm: float
    tag_mm: float
    quiet_mm: float
    #: What is printed under the tag: the bare ID on a single-family sheet, and
    #: ``family·id`` once a sheet mixes families — a cut-out "7" from two
    #: families is two different tags.
    label: str


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def layout_sheet(spec: SheetSpec) -> list[list[Placement]]:
    """Place every tag, paginating as it goes.

    Each :class:`TagRow` starts a fresh band, since two rows sharing a band
    would have to share a cell size they do not have. A band that does not fit
    continues on a new page; a *tag* that cannot fit any page raises rather than
    emitting a blank one.
    """
    if not spec.rows:
        raise TagSheetError("Add at least one row of tags.")
    page_w, page_h = spec.page_mm()
    usable_w = page_w - 2 * spec.margin_mm - ROW_LABEL_MM
    usable_h = page_h - 2 * spec.margin_mm - FOOTER_MM
    if usable_w <= 0 or usable_h <= 0:
        raise TagSheetError(f"A {spec.margin_mm:g} mm margin leaves no room on {spec.page}.")

    mixed = len(spec.families) > 1
    pages: list[list[Placement]] = [[]]
    used_h = 0.0
    for row in spec.rows:
        info = row.validate()
        cell_w = row.tag_mm + 2 * row.quiet_mm
        cell_h = cell_w + (LABEL_MM if spec.labels else 0.0)
        columns = int(usable_w // cell_w)
        if columns < 1 or cell_h > usable_h + _EPS:
            raise TagSheetError(
                f"A {row.tag_mm:g} mm tag with a {row.quiet_mm:g} mm quiet zone does not fit on "
                f"{spec.page} at a {spec.margin_mm:g} mm margin."
            )
        ids = list(range(row.first_id, row.first_id + row.count))
        for start in range(0, len(ids), columns):
            if used_h + cell_h > usable_h + _EPS:
                pages.append([])
                used_h = 0.0
            for column, marker_id in enumerate(ids[start : start + columns]):
                pages[-1].append(
                    Placement(
                        family=row.family,
                        marker_id=marker_id,
                        x_mm=spec.margin_mm + ROW_LABEL_MM + column * cell_w + row.quiet_mm,
                        y_mm=spec.margin_mm + used_h + row.quiet_mm,
                        tag_mm=row.tag_mm,
                        quiet_mm=row.quiet_mm,
                        label=f"{info.short_name}·{marker_id}" if mixed else str(marker_id),
                    )
                )
            used_h += cell_h
    return pages


def black_rects(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover every black module with maximal ``(col, row, width, height)`` rects.

    Merging keeps solid black areas a handful of shapes rather than one per
    module: abutting rects can show antialiased seams wherever the output is
    rasterised, and a tag drawn as one rect per module bloats the PDF. Coverage
    is exact and the rects never overlap.
    """
    remaining = np.asarray(grid) == 0
    n_rows, n_cols = remaining.shape
    rects: list[tuple[int, int, int, int]] = []
    for row in range(n_rows):
        col = 0
        while col < n_cols:
            if not remaining[row, col]:
                col += 1
                continue
            width = 1
            while col + width < n_cols and remaining[row, col + width]:
                width += 1
            height = 1
            while row + height < n_rows and remaining[row + height, col : col + width].all():
                height += 1
            remaining[row : row + height, col : col + width] = False
            rects.append((col, row, width, height))
            col += width
    return rects


def caption(spec: SheetSpec) -> str:
    """What the sheet says about itself, printed under the scale bar."""
    parts = [f"{row.family} {row.tag_mm:g} mm ×{row.count}" for row in spec.rows]
    parts.append("print at 100% scale, no page fitting")
    return "  ·  ".join(parts)


def scale_bar_mm(spec: SheetSpec) -> tuple[float, float, float]:
    """``(x, y, length)`` of the printed rule, in page millimetres.

    Exposed because the one property worth testing about the bar is that it
    measures exactly :data:`SCALE_BAR_MM` on paper.
    """
    _page_w, page_h = spec.page_mm()
    return spec.margin_mm, page_h - spec.margin_mm - FOOTER_MM / 2, SCALE_BAR_MM


# ----------------------------------------------------------------------
# Painting — one routine behind the preview, the PDF and the printer
# ----------------------------------------------------------------------


def render_pages(
    painter: QPainter,
    spec: SheetSpec,
    pages: Sequence[Sequence[Placement]],
    scale: float,
    new_page: Callable[[], None] | None = None,
) -> int:
    """Paint *pages*; *scale* is device units per millimetre.

    ``new_page`` starts the next page on a paged device and may be ``None`` for
    a single-page device (the preview) — the remaining pages are then simply not
    drawn, and the count returned says how many were.
    """
    drawn = 0
    for number, page in enumerate(pages):
        if number:
            if new_page is None:
                break
            new_page()
        draw_page(painter, spec, page, scale, number + 1, len(pages))
        drawn += 1
    return drawn


def draw_page(
    painter: QPainter,
    spec: SheetSpec,
    page: Sequence[Placement],
    scale: float,
    page_number: int,
    page_count: int,
) -> None:
    """One page: its tags, then the footer that makes the print verifiable."""
    painter.save()
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(0, 0, 0)))
    for placement in page:
        _draw_tag(painter, placement, scale)
    painter.restore()
    if spec.labels:
        _draw_labels(painter, page, scale)
    _draw_band_sizes(painter, spec, page, scale)
    _draw_footer(painter, spec, scale, page_number, page_count)


def _draw_band_sizes(painter: QPainter, spec: SheetSpec, page: Sequence[Placement], scale: float) -> None:
    """The printed size beside each band of tags, in the left gutter.

    The footer caption already lists every size on the sheet, but the sheet does
    not stay whole: tags get cut out, and a strip separated from its footer is
    anonymous. This travels with the strip.

    Bands are recovered by grouping on ``y_mm`` rather than being carried
    through the layout — every tag in a band shares a row position and a size by
    construction, so there is nothing for a second representation to disagree
    with.
    """
    bands: dict[float, Placement] = {}
    for placement in page:
        bands.setdefault(round(placement.y_mm, 6), placement)
    painter.save()
    painter.setPen(QColor(0, 0, 0))
    base = _text_font(painter, scale, 2.0)
    width = (ROW_LABEL_MM - 2.0) * scale
    for placement in bands.values():
        size = placement.tag_mm
        text, font = _fitted_text(painter, base, (f"{size:g} × {size:g} mm", f"{size:g} mm"), width)
        painter.setFont(font)
        painter.drawText(
            QRectF(
                spec.margin_mm * scale,
                placement.y_mm * scale,
                width,
                placement.tag_mm * scale,
            ),
            Qt.AlignRight | Qt.AlignVCenter,
            text,
        )
    painter.restore()


def _draw_tag(painter: QPainter, placement: Placement, scale: float) -> None:
    """The modules themselves — no pen, so no half a pen width per rectangle."""
    grid = marker_grid(placement.family, placement.marker_id)
    module_mm = placement.tag_mm / grid.shape[0]
    for col, row, width, height in black_rects(grid):
        painter.drawRect(
            QRectF(
                (placement.x_mm + col * module_mm) * scale,
                (placement.y_mm + row * module_mm) * scale,
                width * module_mm * scale,
                height * module_mm * scale,
            )
        )


def _text_font(painter: QPainter, scale: float, height_mm: float) -> QFont:
    font = QFont(painter.font())
    font.setPixelSize(max(int(round(height_mm * scale)), 1))
    return font


def _draw_labels(painter: QPainter, page: Sequence[Placement], scale: float) -> None:
    """The printed ID under each tag, shrunk to fit rather than cut off.

    A mixed sheet spells the family into every label, which is far longer than a
    5 mm cell is wide. The label therefore degrades rather than being clipped:
    shrink to fit, and past :data:`MIN_LABEL_SHRINK` fall back to the bare ID —
    a small label is readable, "tag36h1" is a lie.
    """
    painter.save()
    painter.setPen(QColor(0, 0, 0))
    base = _text_font(painter, scale, LABEL_MM * 0.72)
    for placement in page:
        cell_w = (placement.tag_mm + 2 * placement.quiet_mm) * scale
        text, font = _fitted_text(painter, base, (placement.label, str(placement.marker_id)), cell_w)
        painter.setFont(font)
        rect = QRectF(
            (placement.x_mm - placement.quiet_mm) * scale,
            (placement.y_mm + placement.tag_mm + placement.quiet_mm) * scale,
            cell_w,
            LABEL_MM * scale,
        )
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter, text)
    painter.restore()


def _fitted_text(painter: QPainter, font: QFont, candidates: Sequence[str], width: float) -> tuple[str, QFont]:
    """The first of *candidates* that fits *width*, shrinking the font to help.

    Nothing printed here may be clipped: every string on this sheet is a fact
    about the tags beside it, and half a fact ("12.5 × 12.5 m") reads as a
    different one. So text shrinks down to :data:`MIN_LABEL_SHRINK` and then
    falls back to a shorter wording, with the last candidate accepted at that
    floor whatever its width.
    """
    last = len(candidates) - 1
    for index, text in enumerate(candidates):
        painter.setFont(font)
        advance = painter.fontMetrics().horizontalAdvance(text)
        if advance <= width or advance <= 0:
            return text, font
        ratio = width / advance
        if ratio >= MIN_LABEL_SHRINK or index == last:
            shrunk = QFont(font)
            shrunk.setPixelSize(max(int(font.pixelSize() * max(ratio, MIN_LABEL_SHRINK)), 1))
            return text, shrunk
    return candidates[last], font


def _draw_footer(painter: QPainter, spec: SheetSpec, scale: float, page_number: int, page_count: int) -> None:
    """The scale bar and the caption — how a print is checked and identified.

    Two lines, not one: the caption names every row on the sheet, so it is the
    string most likely to run long, and beside the bar it had barely half the
    page. It gets its own full-width line **and** shrinks to fit, because
    ``drawText`` silently clips to its rect — a caption cut off mid-size is
    worse than no caption, since what survives still looks authoritative.
    """
    page_w, _page_h = spec.page_mm()
    x_mm, y_mm, length_mm = scale_bar_mm(spec)
    painter.save()

    # Drawn as filled rects rather than strokes: a pen adds half its width to
    # each end, which is precisely the error the bar exists to detect.
    pen_mm = 0.25
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(0, 0, 0)))
    painter.drawRect(QRectF(x_mm * scale, (y_mm - pen_mm / 2) * scale, length_mm * scale, pen_mm * scale))
    for end in (x_mm, x_mm + length_mm):
        painter.drawRect(QRectF((end - pen_mm / 2) * scale, (y_mm - 1.5) * scale, pen_mm * scale, 3.0 * scale))
    painter.setPen(QColor(0, 0, 0))
    painter.setFont(_text_font(painter, scale, 2.4))
    painter.drawText(
        QRectF((x_mm + length_mm + 1.5) * scale, (y_mm - 2.0) * scale, 20.0 * scale, 4.0 * scale),
        Qt.AlignLeft | Qt.AlignVCenter,
        f"{length_mm:g} mm",
    )
    text = caption(spec)
    if page_count > 1:
        text = f"{text}  ·  page {page_number}/{page_count}"
    width = (page_w - 2 * spec.margin_mm) * scale
    _text, font = _fitted_text(painter, _text_font(painter, scale, 2.2), (text,), width)
    painter.setPen(QColor(0, 0, 0))
    painter.setFont(font)
    painter.drawText(
        QRectF(spec.margin_mm * scale, (y_mm + 2.0) * scale, width, 4.5 * scale),
        Qt.AlignLeft | Qt.AlignVCenter,
        text,
    )
    painter.restore()


# ----------------------------------------------------------------------
# Output devices
# ----------------------------------------------------------------------


def _page_size(spec: SheetSpec) -> QPageSize:
    return QPageSize(QPageSize.A4 if spec.page == "A4" else QPageSize.Letter)


def write_pdf(spec: SheetSpec, path: str | Path, resolution: int = PDF_RESOLUTION) -> int:
    """Write the sheet as vector PDF, returning the page count.

    Margins are set to zero and the page size to the real paper: every
    coordinate in this module is then a page-absolute millimetre, which is what
    makes the scale bar mean anything.
    """
    pages = layout_sheet(spec)
    writer = QPdfWriter(str(path))
    writer.setPageSize(_page_size(spec))
    writer.setPageMargins(QMarginsF(0.0, 0.0, 0.0, 0.0), QPageLayout.Millimeter)
    writer.setResolution(int(resolution))
    writer.setTitle("Ethograph tag sheet")
    painter = QPainter(writer)
    try:
        render_pages(painter, spec, pages, resolution / 25.4, writer.newPage)
    finally:
        painter.end()
    return len(pages)


def print_sheet(spec: SheetSpec, printer) -> int:
    """Paint the sheet onto an already-configured ``QPrinter``.

    The printer's own resolution drives the scale, since a driver is free to
    ignore a requested one — reading it back is the only way the millimetres
    stay millimetres.
    """
    pages = layout_sheet(spec)
    printer.setPageSize(_page_size(spec))
    printer.setPageMargins(QMarginsF(0.0, 0.0, 0.0, 0.0), QPageLayout.Millimeter)
    printer.setFullPage(True)
    painter = QPainter(printer)
    try:
        render_pages(painter, spec, pages, printer.resolution() / 25.4, printer.newPage)
    finally:
        painter.end()
    return len(pages)
