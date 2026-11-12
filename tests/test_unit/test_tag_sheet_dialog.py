"""The tag sheet dialog: what it computes for the user and what it refuses.

Driven through a real ``ObservableAppState`` and no dataset — the sheet is made
before there is any footage, so needing one would be a bug in the feature.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("cv2")

from qtpy.QtWidgets import QApplication  # noqa: E402

from ethograph.gui.app_state import ObservableAppState  # noqa: E402
from ethograph.gui.dialog_tag_sheet import TagSheetDialog  # noqa: E402
from ethograph.gui.pose_tagsheet import DEFAULT_QUIET_MIN_MM, default_quiet_mm, dictionary_info  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp):
    sheet = TagSheetDialog(ObservableAppState())
    yield sheet
    sheet.close()


def _headers(dialog) -> list[str]:
    return [dialog.row_table.horizontalHeaderItem(column).text() for column in range(dialog.row_table.columnCount())]


def test_only_detectable_families_are_printable(dialog):
    """Printing is NOT a superset of detecting.

    A sheet of tags the Detect stage cannot read is worse than no sheet — the
    failure only shows up once the animals are wearing them. ``tag36h10`` is the
    live example: OpenCV renders it, AprilTag 3 dropped it.
    """
    from ethograph.gui.pose_detect import TAG_FAMILIES

    offered = [dialog._rows[0].family.itemText(i) for i in range(dialog._rows[0].family.count())]
    assert offered == list(TAG_FAMILIES)
    assert "tag36h10" not in offered
    assert "Family" in _headers(dialog)


def test_the_sheet_opens_on_the_family_the_detect_tab_uses(qapp):
    """One family setting, not two that can disagree."""
    state = ObservableAppState()
    state.detect_tag_family = "tag25h9"
    sheet = TagSheetDialog(state)
    try:
        assert sheet._rows[0].family.currentText() == "tag25h9"
        assert sheet.spec().rows[0].family == "tag25h9"
        # And picking one here is what the Detect tab will then look for.
        sheet._rows[0].family.setCurrentText("tag16h5")
        assert state.detect_tag_family == "tag16h5"
    finally:
        sheet.close()


def test_a_second_family_warns_that_detection_needs_two_passes(dialog):
    assert "Mixing" not in dialog.warnings.text()
    dialog._on_add_row()
    dialog._rows[1].family.setCurrentText("tag16h5")
    assert "Mixing families" in dialog.warnings.text()
    assert "2 detection passes" in dialog.warnings.text()


def test_add_row_duplicates_the_selected_row_and_continues_its_ids(dialog):
    first = dialog._rows[0]
    first.count.setValue(6)
    first.tag_mm.setValue(12.0)
    dialog._on_add_row()
    second = dialog._rows[1].to_row()
    assert (second.count, second.tag_mm, second.first_id) == (6, 12.0, 6)

    # The copy follows the selection, not the last row.
    dialog.row_table.selectRow(0)
    dialog._on_add_row()
    third = dialog._rows[2].to_row()
    assert (third.count, third.tag_mm, third.first_id) == (6, 12.0, 6)


def test_it_lays_out_a_page_and_says_how_much_fits(dialog):
    assert dialog._pages
    assert "tag(s)" in dialog.summary.text()
    assert not dialog.preview.pixmap().isNull()
    assert dialog.pdf_btn.isEnabled()


def test_the_quiet_zone_is_computed_rather_than_asked(dialog):
    """Two modules of white, derived — there is no control for it."""
    modules = dictionary_info().modules
    dialog._rows[0].tag_mm.setValue(16.0)
    assert dialog.spec().rows[0].quiet_mm == pytest.approx(2 * 16.0 / modules)
    assert not hasattr(dialog._rows[0], "quiet_mm")


def test_a_small_tag_still_gets_a_usable_quiet_zone(dialog):
    dialog._rows[0].tag_mm.setValue(2.0)
    assert dialog.spec().rows[0].quiet_mm == pytest.approx(DEFAULT_QUIET_MIN_MM)
    assert default_quiet_mm(2.0, dictionary_info().modules) == DEFAULT_QUIET_MIN_MM


def test_the_id_range_is_capped_by_the_family(dialog):
    row = dialog._rows[0]
    assert row.first_id.maximum() == 586
    assert row.count.maximum() == 587
    row.first_id.setValue(580)
    assert row.count.maximum() == 7, "no wraparound past the end of the family"


def test_the_minimum_size_appears_only_once_the_camera_is_known(dialog):
    assert dialog._rows[0].minimum.text() == "—", "an unknown camera gets no minimum, not a guessed one"
    dialog.camera_combo.setCurrentText("1920 × 1080")
    dialog.fov_spin.setValue(200.0)
    assert float(dialog._rows[0].minimum.text()) == pytest.approx(8 * 5 * 200.0 / 1920, abs=0.05)


def test_the_camera_is_picked_as_a_resolution_and_only_the_width_counts(dialog):
    """1920×1080 and 1920×1200 are the same camera as far as a tag is concerned."""
    dialog.fov_spin.setValue(200.0)
    dialog.camera_combo.setCurrentText("1920 × 1080")
    wide = dialog._rows[0].minimum.text()
    dialog.camera_combo.setCurrentText("1920 × 1200")
    assert dialog._rows[0].minimum.text() == wide
    dialog.camera_combo.setCurrentText("3840 × 2160")
    assert float(dialog._rows[0].minimum.text()) < float(wide)


def test_the_width_spin_appears_only_for_a_custom_resolution(dialog, qapp):
    dialog.show()
    try:
        dialog.camera_combo.setCurrentText("1920 × 1080")
        assert not dialog.camera_px_spin.isVisible()
        dialog.camera_combo.setCurrentText("Custom…")
        assert dialog.camera_px_spin.isVisible()
        dialog.camera_px_spin.setValue(4096)
        assert dialog._camera_width_px() == pytest.approx(4096.0)
    finally:
        dialog.hide()


def test_a_loaded_video_seeds_the_picker_without_locking_it(qapp):
    """The rig that will FILM the tags need not be the one already loaded."""
    sheet = TagSheetDialog(ObservableAppState(), image_width_px=1920.0)
    try:
        assert sheet._camera_width_px() == pytest.approx(1920.0)
        assert sheet.camera_combo.isEnabled(), "a sheet may be for a camera with no footage yet"
        sheet.camera_combo.setCurrentText("3840 × 2160")
        assert sheet._camera_width_px() == pytest.approx(3840.0)
    finally:
        sheet.close()


def test_a_video_width_with_no_preset_falls_back_to_custom(qapp):
    """512 px is a real video and not a listed resolution — it must still seed."""
    sheet = TagSheetDialog(ObservableAppState(), image_width_px=512.0)
    try:
        assert sheet._camera_width_px() == pytest.approx(512.0)
        assert sheet.camera_combo.currentData() == -1
        assert sheet.camera_px_spin.value() == 512
    finally:
        sheet.close()


def test_the_video_beats_the_remembered_camera(qapp):
    state = ObservableAppState()
    state.tag_sheet_camera_width_px = 640
    sheet = TagSheetDialog(state, image_width_px=1920.0)
    try:
        assert sheet._camera_width_px() == pytest.approx(1920.0)
    finally:
        sheet.close()


def test_without_a_video_the_camera_is_an_input_and_is_remembered(dialog):
    assert dialog.camera_combo is not None
    dialog.camera_combo.setCurrentText("2048 × 1536")
    assert dialog.app_state.tag_sheet_camera_width_px == 2048
    # And a remembered width reopens on the preset that carries it.
    reopened = TagSheetDialog(dialog.app_state)
    try:
        assert reopened._camera_width_px() == pytest.approx(2048.0)
    finally:
        reopened.close()


def test_a_tag_under_the_minimum_is_flagged(dialog):
    dialog.camera_combo.setCurrentText("1920 × 1080")
    dialog.fov_spin.setValue(200.0)
    dialog._rows[0].tag_mm.setValue(2.0)
    assert "Too small for the camera" in dialog.warnings.text()
    dialog._rows[0].tag_mm.setValue(12.0)
    assert "Too small for the camera" not in dialog.warnings.text()


def test_an_impossible_sheet_says_why_and_blocks_export(dialog):
    dialog._rows[0].tag_mm.setValue(150.0)
    dialog.margin_spin.setValue(45.0)
    assert "does not fit" in dialog.preview.text()
    assert dialog.preview.pixmap().isNull()
    assert not dialog.pdf_btn.isEnabled()
    assert not dialog.print_btn.isEnabled()


def test_page_setup_persists_but_a_broken_size_does_not(dialog):
    state = dialog.app_state
    dialog.page_combo.setCurrentText("Letter")
    dialog._rows[0].tag_mm.setValue(8.0)
    assert state.tag_sheet_page == "Letter"
    assert state.tag_sheet_tag_mm == pytest.approx(8.0)

    dialog._rows[0].tag_mm.setValue(300.0)
    assert state.tag_sheet_tag_mm == pytest.approx(8.0), "a size that cannot be printed is not remembered"


def test_the_last_row_cannot_be_removed(dialog):
    dialog._on_remove_row()
    assert len(dialog._rows) == 1
    dialog._on_add_row()
    dialog.row_table.setCurrentCell(1, 0)
    dialog._on_remove_row()
    assert len(dialog._rows) == 1


def test_a_written_pdf_matches_the_preview_page_count(dialog, tmp_path):
    from ethograph.gui.pose_tagsheet import write_pdf

    path = tmp_path / "sheet.pdf"
    assert write_pdf(dialog.spec(), path) == len(dialog._pages)
    assert path.read_bytes().startswith(b"%PDF")
