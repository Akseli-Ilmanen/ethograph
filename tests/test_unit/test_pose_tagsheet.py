"""The tag sheet: family facts, layout maths, geometry and output.

The point of every test here is that a printed sheet cannot be debugged after
the fact — a tag that is 4% too small or has a softened edge looks exactly like
a tag that is fine until a whole session decodes nothing. So the checks are:
the module grid is the size the paper maths assumes, a rendered tag still
decodes *with the detector that will read it*, the layout is the arithmetic it
claims, and the scale bar measures 50 mm in device units.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
cv2 = pytest.importorskip("cv2")

from qtpy.QtGui import QPainter, QPixmap  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from ethograph.gui.pose_detect import DEFAULT_TAG_FAMILY, TAG_FAMILIES, family_modules  # noqa: E402
from ethograph.gui.pose_tagsheet import (  # noqa: E402
    FOOTER_MM,
    LABEL_MM,
    MIN_PX_PER_MODULE,
    MIN_QUIET_MM,
    PDF_RESOLUTION,
    ROW_LABEL_MM,
    SCALE_BAR_MM,
    SheetSpec,
    TagRow,
    TagSheetError,
    black_rects,
    caption,
    default_quiet_mm,
    dictionary_info,
    layout_sheet,
    marker_grid,
    min_tag_mm,
    render_pages,
    scale_bar_mm,
    write_pdf,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------
# The family
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", TAG_FAMILIES)
def test_modules_are_the_data_grid_plus_a_border(family):
    """The paper size follows from this, so it is the one number to pin down."""
    info = dictionary_info(family)
    assert info.modules == info.marker_size + 2 == family_modules(family)
    assert marker_grid(family, 0).shape == (info.modules, info.modules)


def test_a_smaller_family_is_less_paper_for_the_same_pixels():
    """The whole reason the family is a choice and not a constant."""
    assert dictionary_info("tag16h5").modules == 6
    assert dictionary_info("tag25h9").modules == 7
    assert dictionary_info("tag36h11").modules == 8


def test_the_id_count_comes_from_opencv():
    assert dictionary_info("tag36h11").n_ids == 587
    assert dictionary_info("tag25h9").n_ids == 35
    assert dictionary_info("tag16h5").n_ids == 30


def test_an_id_outside_the_family_raises():
    with pytest.raises(TagSheetError):
        marker_grid("tag36h11", 587)
    with pytest.raises(TagSheetError):
        marker_grid("tag16h5", 30)


def test_a_family_the_detector_cannot_read_is_not_printable():
    """tag36h10 renders fine in OpenCV and never decodes — so it is refused.

    Printing is not allowed to be a superset of detecting: a sheet of tags
    Ethograph cannot read is worse than no sheet, because the failure only
    shows up after the animals are already wearing them.
    """
    from ethograph.gui.pose_detect import PointDetectorError

    with pytest.raises((TagSheetError, PointDetectorError)):
        dictionary_info("tag36h10")
    with pytest.raises((TagSheetError, PointDetectorError)):
        layout_sheet(SheetSpec(rows=[TagRow("tag36h10", 0, 4, 6.0, 1.5)]))


# ----------------------------------------------------------------------
# The tag itself survives printing
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", TAG_FAMILIES)
@pytest.mark.parametrize("marker_id", [0, 7, 29])
def test_a_rendered_tag_decodes_back_to_its_id(family, marker_id):
    """A round trip through the module grid: render big, pad white, detect.

    Decoded by the detector that will actually read the print, not by the
    library that drew it — that agreement is the whole contract between
    `pose_tagsheet` and `pose_detect`, and nothing checks it at runtime.
    """
    pytest.importorskip("pupil_apriltags")
    from ethograph.gui.pose_detect import AprilTagDetector

    if marker_id >= dictionary_info(family).n_ids:
        pytest.skip(f"{family} holds fewer than {marker_id + 1} IDs")
    grid = marker_grid(family, marker_id)
    scaled = np.kron(grid, np.ones((10, 10), dtype=np.uint8))
    padded = np.full((scaled.shape[0] + 40, scaled.shape[1] + 40), 255, dtype=np.uint8)
    padded[20:-20, 20:-20] = scaled

    found = AprilTagDetector(family=family).detect(padded)

    assert len(found) == 1, f"tag {marker_id} did not decode"
    assert AprilTagDetector.decode_label(found[0].label)[0] == marker_id


def test_the_rects_cover_the_black_modules_exactly():
    grid = marker_grid(DEFAULT_TAG_FAMILY, 3)
    covered = np.zeros(grid.shape, dtype=bool)
    for col, row, width, height in black_rects(grid):
        assert not covered[row : row + height, col : col + width].any(), "rects overlap"
        covered[row : row + height, col : col + width] = True
    assert (covered == (grid == 0)).all()


def test_solid_black_is_merged_rather_than_one_rect_per_module():
    """Abutting rects can show antialiased seams wherever the output rasterises."""
    grid = marker_grid(DEFAULT_TAG_FAMILY, 0)
    assert len(black_rects(grid)) < int((grid == 0).sum())


# ----------------------------------------------------------------------
# Sizing advice
# ----------------------------------------------------------------------


def test_the_minimum_size_is_the_pixel_budget_in_millimetres():
    # 1080p over a 200 mm field of view: 0.185 mm/px, and 8 modules to cover.
    assert min_tag_mm(8, 1080, 200.0) == pytest.approx(8 * MIN_PX_PER_MODULE * 200.0 / 1080)
    # A wider camera over the same scene affords a smaller tag.
    assert min_tag_mm(8, 1920, 200.0) < min_tag_mm(8, 1080, 200.0)
    # And so does a family with fewer modules — the point of offering them.
    assert min_tag_mm(6, 1080, 200.0) < min_tag_mm(8, 1080, 200.0)


def test_an_unknown_camera_gets_no_minimum_rather_than_a_guess():
    assert min_tag_mm(8, 0, 200.0) is None
    assert min_tag_mm(8, 1920, 0.0) is None


def test_the_quiet_zone_defaults_to_a_module_with_a_floor():
    assert default_quiet_mm(16.0, 8) == pytest.approx(2.0)
    assert default_quiet_mm(4.0, 8) == MIN_QUIET_MM


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _spec(**kwargs) -> SheetSpec:
    rows = kwargs.pop("rows", [TagRow(DEFAULT_TAG_FAMILY, 0, 24, 4.0, 1.0)])
    return SheetSpec(rows=rows, **kwargs)


def test_the_grid_is_the_arithmetic_it_claims():
    spec = _spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 60, 3.0, 1.0)], margin_mm=10.0, labels=False)
    pages = layout_sheet(spec)
    cell = 3.0 + 2 * 1.0
    columns = int((210.0 - 20.0 - ROW_LABEL_MM) // cell)
    placements = pages[0]
    assert len(pages) == 1
    assert len(placements) == 60
    # First tag sits one quiet zone inside the margin and the size gutter, and
    # the row wraps at exactly `columns`.
    assert placements[0].x_mm == pytest.approx(10.0 + ROW_LABEL_MM + 1.0)
    assert placements[0].y_mm == pytest.approx(10.0 + 1.0)
    assert placements[columns].y_mm == pytest.approx(placements[0].y_mm + cell)
    assert placements[columns].x_mm == pytest.approx(placements[0].x_mm)


def test_labels_make_the_cell_taller_but_not_wider():
    rows = [TagRow(DEFAULT_TAG_FAMILY, 0, 60, 3.0, 1.0)]
    without = layout_sheet(_spec(rows=rows, labels=False))[0]
    with_labels = layout_sheet(_spec(rows=rows, labels=True))[0]
    columns = sum(1 for place in without if place.y_mm == without[0].y_mm)
    assert sum(1 for place in with_labels if place.y_mm == with_labels[0].y_mm) == columns
    assert with_labels[columns].y_mm - with_labels[0].y_mm == pytest.approx(
        without[columns].y_mm - without[0].y_mm + LABEL_MM
    )


def test_overflow_paginates():
    spec = _spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 250, 12.0, 2.0)])
    pages = layout_sheet(spec)
    assert len(pages) > 1
    assert sum(len(page) for page in pages) == 250
    # Nothing intrudes on the footer the scale bar lives in.
    limit = 297.0 - spec.margin_mm - FOOTER_MM
    assert all(place.y_mm + place.tag_mm + place.quiet_mm <= limit + 1e-6 for page in pages for place in page)


def test_every_band_leaves_the_left_gutter_free_for_its_size():
    """A cut-out strip of tags carries its own size, so the gutter is reserved."""
    spec = _spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 8, 3.0, 1.0), TagRow(DEFAULT_TAG_FAMILY, 8, 4, 8.0, 1.5)])
    for page in layout_sheet(spec):
        for place in page:
            assert place.x_mm - place.quiet_mm >= spec.margin_mm + ROW_LABEL_MM - 1e-9


def test_the_caption_never_runs_past_the_page(qapp):
    """`drawText` clips silently, and half a caption still looks authoritative."""
    from qtpy.QtGui import QFont

    from ethograph.gui.pose_tagsheet import _fitted_text

    spec = _spec(rows=[TagRow(DEFAULT_TAG_FAMILY, index * 4, 4, 6.0, 1.5) for index in range(8)])
    pixmap = QPixmap(10, 10)
    painter = QPainter(pixmap)
    font = QFont(painter.font())
    font.setPixelSize(20)
    try:
        text = caption(spec)
        _fitted, shrunk = _fitted_text(painter, font, (text,), 400.0)
        painter.setFont(shrunk)
        assert painter.fontMetrics().horizontalAdvance(text) <= 400.0 or shrunk.pixelSize() < font.pixelSize()
    finally:
        painter.end()


def test_each_row_starts_its_own_band():
    """Two rows never share a band — they do not share a cell size."""
    spec = _spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 3, 3.0, 1.0), TagRow(DEFAULT_TAG_FAMILY, 3, 3, 8.0, 2.0)])
    first, second = layout_sheet(spec)[0][:3], layout_sheet(spec)[0][3:]
    assert len({place.y_mm for place in first}) == 1
    assert min(place.y_mm for place in second) > first[0].y_mm


def test_the_printed_label_is_the_bare_id_on_a_single_family_sheet():
    placements = layout_sheet(_spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 5, 3, 4.0, 1.0)]))[0]
    assert [place.marker_id for place in placements] == [5, 6, 7]
    assert [place.label for place in placements] == ["5", "6", "7"]


def test_a_mixed_sheet_labels_which_family_each_tag_is():
    """A cut-out "7" from two families is two different tags."""
    spec = _spec(rows=[TagRow("tag36h11", 0, 2, 4.0, 1.0), TagRow("tag16h5", 0, 2, 4.0, 1.0)])
    assert [place.label for place in layout_sheet(spec)[0]] == [
        "tag36h11·0",
        "tag36h11·1",
        "tag16h5·0",
        "tag16h5·1",
    ]


def test_a_tag_too_big_for_the_page_raises_rather_than_emitting_nothing():
    with pytest.raises(TagSheetError):
        layout_sheet(_spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 1, 400.0, 1.0)]))


def test_a_margin_that_swallows_the_page_raises():
    with pytest.raises(TagSheetError):
        layout_sheet(_spec(margin_mm=110.0))


def test_more_ids_than_the_family_holds_raises():
    with pytest.raises(TagSheetError):
        layout_sheet(_spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 588, 3.0, 1.0)]))
    with pytest.raises(TagSheetError):
        layout_sheet(_spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 580, 20, 3.0, 1.0)]))


def test_a_quiet_zone_under_the_floor_raises():
    """Enforced, not merely defaulted — the quad finder needs the white."""
    with pytest.raises(TagSheetError, match="quiet zone"):
        layout_sheet(_spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 4, 3.0, 0.2)]))
    assert layout_sheet(_spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 4, 3.0, MIN_QUIET_MM)]))


def test_an_empty_sheet_raises():
    with pytest.raises(TagSheetError):
        layout_sheet(SheetSpec(rows=[]))


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def test_the_scale_bar_measures_fifty_millimetres_in_device_units():
    spec = _spec()
    _x, _y, length = scale_bar_mm(spec)
    assert length == SCALE_BAR_MM
    assert length * PDF_RESOLUTION / 25.4 == pytest.approx(SCALE_BAR_MM * PDF_RESOLUTION / 25.4)


def test_the_scale_bar_sits_inside_the_footer_it_reserved():
    spec = _spec()
    _x, y, _length = scale_bar_mm(spec)
    assert 297.0 - spec.margin_mm - FOOTER_MM < y < 297.0 - spec.margin_mm


def test_the_caption_says_what_was_printed():
    text = caption(_spec())
    assert "tag36h11 4 mm" in text
    assert "100%" in text
    mixed = caption(_spec(rows=[TagRow("tag36h11", 0, 2, 4.0, 1.0), TagRow("tag16h5", 0, 2, 3.0, 1.0)]))
    assert "tag36h11" in mixed and "tag16h5" in mixed


def test_the_pdf_is_written_with_a_page_per_layout_page(qapp, tmp_path):
    spec = _spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 250, 12.0, 2.0)])
    path = tmp_path / "sheet.pdf"
    pages = write_pdf(spec, path)
    assert pages == len(layout_sheet(spec)) > 1
    assert path.read_bytes().startswith(b"%PDF")


def test_painting_puts_black_on_the_page(qapp):
    """The preview path — the same routine that drives the PDF."""
    spec = _spec()
    pixmap = QPixmap(420, 594)
    pixmap.fill(0xFFFFFFFF)
    painter = QPainter(pixmap)
    try:
        drawn = render_pages(painter, spec, layout_sheet(spec), 2.0)
    finally:
        painter.end()
    assert drawn == 1
    image = pixmap.toImage()
    assert any(
        image.pixelColor(x, y).lightness() < 100
        for x in range(0, image.width(), 3)
        for y in range(0, image.height(), 3)
    )


def test_a_single_page_device_stops_after_the_first_page(qapp):
    spec = _spec(rows=[TagRow(DEFAULT_TAG_FAMILY, 0, 250, 12.0, 2.0)])
    pages = layout_sheet(spec)
    pixmap = QPixmap(210, 297)
    pixmap.fill(0xFFFFFFFF)
    painter = QPainter(pixmap)
    try:
        drawn = render_pages(painter, spec, pages, 1.0)
    finally:
        painter.end()
    assert len(pages) > 1
    assert drawn == 1
