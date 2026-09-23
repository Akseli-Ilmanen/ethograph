"""The labelling dialog's individual/keypoint tree and its key handling.

Driven through a stub data widget and a fake camera view, so no dataset, video
or GPU canvas is needed — only a QApplication and a headless pygfx scene.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("pygfx")

import pygfx as gfx  # noqa: E402
from qtpy.QtCore import QEvent, Qt  # noqa: E402
from qtpy.QtGui import QColor, QKeyEvent, QPixmap  # noqa: E402
from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

from ethograph.gui import dialog_pose_labelling as dialog_module  # noqa: E402
from ethograph.gui.app_state import ObservableAppState  # noqa: E402
from ethograph.gui.dialog_pose_labelling import (  # noqa: E402
    _FIXED_COLUMNS,
    COLUMNS_PER_KEYPOINT,
    INDIVIDUAL_COLUMN,
    SOURCE_COLUMN,
    PoseLabellingDialog,
)
from ethograph.gui.pose_annotate import RECOMMENDED_LABEL_SHARE  # noqa: E402
from ethograph.gui.pose_convert import COLOR_BY_INDIVIDUAL, COLOR_BY_KEYPOINT  # noqa: E402
from ethograph.gui.pose_edit_mixin import LOOP_MODE, SEQUENTIAL_MODE  # noqa: E402
from ethograph.gui.pose_fill import SplineBackend  # noqa: E402

NAMES = ["beak", "tail", "eye"]


class _FakeRenderWidget(QWidget):
    """The inner widget rendercanvas actually gives focus to.

    Its ``keyPressEvent`` neither ignores the event nor calls the base class,
    exactly like ``rendercanvas.qt.QRenderWidget`` — so a key pressed over the
    video stops here and never propagates to the wrapper or the main window.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        event.accept()


class _FakeView:
    """The slice of CameraView the dialog and the label mode touch."""

    def __init__(self):
        self._scene = gfx.Scene()
        # A wrapper holding the focusable render widget, as rendercanvas nests
        # them: filtering the wrapper alone sees no keys at all.
        self._canvas = QWidget()
        self._render_widget = _FakeRenderWidget(self._canvas)
        self.fps = 25.0
        self.n_frames = 10
        self.start_frame = 0
        self.label_mode = None
        self.pan_with_left_drag = True

    def scene(self):
        return self._scene

    def canvas_widget(self):
        return self._canvas

    def key_target(self):
        focusable = [w for w in self._canvas.findChildren(QWidget) if w.focusPolicy() != Qt.NoFocus]
        return focusable[0] if focusable else self._canvas

    def image_height(self):
        return 480.0

    def image_units_per_pixel(self):
        return 1.0

    def set_label_mode(self, mode):
        self.label_mode = mode
        self.pan_with_left_drag = mode is None or mode.locked

    def set_label_locked(self, locked):
        self.pan_with_left_drag = locked

    def request_draw(self):
        pass


class _FakeShell(QWidget):
    """Stands in for the main window: it owns the video area, and the dialog
    installs a key filter on it (Shift+arrows are pressed while looking at the
    video, so they land here rather than on the dialog)."""

    def __init__(self):
        super().__init__()
        self.video_area = type("_Area", (), {"primary": _FakeView()})()


class _FakePoseManager:
    """Records what the dialog hands to the ordinary pose overlay."""

    def __init__(self):
        self.override = None

    def set_pose_override(self, render):
        self.override = render


class _FakeDataWidget:
    """Everything ``PoseLabellingDialog`` reads off the data widget."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.pose_mgr = _FakePoseManager()
        self.shell = _FakeShell()

    def update_pose(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")  # never touch the real settings
    state.keypoints = list(NAMES)
    state.labelling_keypoints = list(NAMES)
    dlg = PoseLabellingDialog(_FakeDataWidget(state))
    yield dlg
    dlg.close()


def _tree_names(tree) -> list[tuple[str, list[str]]]:
    """Branch/leaf names, read from the UserRole rather than the item text."""
    return [
        (
            tree.topLevelItem(i).data(0, Qt.UserRole)[0],
            [tree.topLevelItem(i).child(k).text(0) for k in range(tree.topLevelItem(i).childCount())],
        )
        for i in range(tree.topLevelItemCount())
    ]


def _key(dlg, key: int, modifiers=Qt.NoModifier, target=None) -> None:
    """Send a key press, by default to the dialog itself.

    *target* stands in for the widget that really has focus — the tree, the
    table or the video canvas — since those are where the keys are pressed in
    practice and only an event filter can catch them there.
    """
    QApplication.sendEvent(target or dlg, QKeyEvent(QEvent.KeyPress, key, modifiers))


def _select_leaf(dialog, individual: str, keypoint: str) -> None:
    """Select a keypoint in the Keypoints tree, as clicking it would."""
    for i in range(dialog.tree.topLevelItemCount()):
        branch = dialog.tree.topLevelItem(i)
        for k in range(branch.childCount()):
            leaf = branch.child(k)
            if leaf.data(0, Qt.UserRole) == (individual, keypoint):
                dialog.tree.setCurrentItem(leaf)
                return
    raise AssertionError(f"no leaf for {(individual, keypoint)}")


def test_tree_shows_one_branch_per_individual(dialog):
    assert _tree_names(dialog.tree) == [("individual_0", NAMES)]


def test_adding_an_individual_adds_a_branch(dialog):
    dialog._apply_schema(individuals=["individual_0", "individual_1"])

    names = _tree_names(dialog.tree)
    assert [branch for branch, _ in names] == ["individual_0", "individual_1"]
    assert all(keypoints == NAMES for _, keypoints in names)
    assert dialog.app_state.labelling_individuals == ["individual_0", "individual_1"]


def test_keypoints_are_shared_by_every_individual(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog._apply_schema(keypoints=[*NAMES, "wing"])

    assert all(keypoints == [*NAMES, "wing"] for _, keypoints in _tree_names(dialog.tree))
    assert dialog.store.n_points == 2 * 4


def test_selecting_a_leaf_sets_the_active_pair(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    branch = dialog.tree.topLevelItem(1)
    dialog.tree.setCurrentItem(branch.child(2))

    assert dialog._mode.active_individual == "b"
    assert dialog._mode.active_keypoint == "eye"
    # The line is rich text (glyph + names in the marker colour), so match parts.
    assert "<b>b</b>" in dialog.active_label.text()
    assert "eye" in dialog.active_label.text()


def test_number_key_switches_individual_while_labelling(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    _key(dialog, Qt.Key_2)

    assert dialog._mode.active_individual == "b"
    assert dialog.tree.currentItem().data(0, Qt.UserRole)[0] == "b"


def test_backspace_deletes_the_selected_keypoint_without_a_mode(dialog):
    """Regression: the key filter only existed while a mode was armed, so
    selecting a keypoint and pressing Backspace did nothing at all."""
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    _select_leaf(dialog, "individual_0", "tail")

    _key(dialog, Qt.Key_Backspace)

    assert dialog.store.is_anchor(0, "tail") is False


def test_backspace_deletes_the_selected_keypoint_from_the_tree(dialog):
    """The tree has focus while you are picking keypoints in it."""
    dialog.store.set_point(0, "eye", (5.0, 6.0))
    _select_leaf(dialog, "individual_0", "eye")

    _key(dialog, Qt.Key_Delete, target=dialog.tree)

    assert dialog.store.is_anchor(0, "eye") is False


def test_backspace_leaves_the_other_keypoints_alone(dialog):
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    dialog.store.set_point(0, "tail", (3.0, 4.0))
    _select_leaf(dialog, "individual_0", "beak")

    _key(dialog, Qt.Key_Backspace)

    assert dialog.store.is_anchor(0, "beak") is False
    assert dialog.store.is_anchor(0, "tail") is True


def test_backspace_deletes_the_active_point_while_labelling(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    dialog._mode.set_active("tail")

    _key(dialog, Qt.Key_Backspace)

    assert dialog.store.is_anchor(0, "tail") is False


def test_backspace_in_a_spin_box_edits_the_number(dialog):
    """Eating the spin box's Backspace would delete a keypoint per keystroke."""
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    _select_leaf(dialog, "individual_0", "beak")
    dialog.suggest_percent_spin.setFocus()

    _key(dialog, Qt.Key_Backspace, target=dialog.suggest_percent_spin)

    assert dialog.store.is_anchor(0, "beak") is True


def test_deleting_a_label_leaves_the_prediction_in_place(dialog):
    """Only the label goes; the fill keeps predicting that point."""
    _fill(dialog.store)
    dialog.store.promote_fill(0)
    _select_leaf(dialog, "individual_0", "beak")

    _key(dialog, Qt.Key_Backspace)

    assert dialog.store.is_anchor(0, "beak") is False
    assert not np.isnan(dialog.store.positions_for(0)[0, 0])  # still filled


def test_the_suggestion_combo_opens_on_a_method_that_can_run(dialog):
    """ "uncertain" needs a fill to rank by; with none, it could only warn."""
    assert dialog.suggest_method_combo.currentData() == "uniform"


def test_number_key_is_ignored_when_not_labelling(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    _key(dialog, Qt.Key_2)
    assert dialog._mode is None


def test_tab_cycles_keypoints_of_the_active_individual(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    _key(dialog, Qt.Key_Tab)
    assert dialog._mode.active_keypoint == "tail"


def test_tab_cycles_when_the_tree_has_focus(dialog):
    """Regression: Tab silently did nothing whenever a child widget had focus.

    Sending the event to the dialog (as the test above does) cannot catch this —
    Qt turns Tab into focus navigation inside the *focused* widget's ``event()``,
    so it never propagates up to the dialog's filter. Only the dialog's own
    QShortcut runs early enough, and the filter has to decline the
    ShortcutOverride for Tab or it suppresses that shortcut itself.
    """
    from qtpy.QtTest import QTest

    dialog.show()
    dialog.activateWindow()
    QApplication.processEvents()
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.tree.setFocus()
    QApplication.processEvents()
    assert isinstance(QApplication.focusWidget(), type(dialog.tree))

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
    QApplication.processEvents()

    assert dialog._mode.active_keypoint == "tail"


def test_backspace_reaches_across_from_the_main_window(dialog):
    """You press it right after clicking the video, and where the click left
    focus is not something the user should have to know."""
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    _select_leaf(dialog, "individual_0", "tail")

    _key(dialog, Qt.Key_Backspace, target=dialog._shell)

    assert dialog.store.is_anchor(0, "tail") is False


def test_backspace_works_when_the_render_widget_has_focus(dialog):
    """Regression: the filter sat on the canvas *wrapper*, which never sees a key.

    Clicking the video to place a point focuses rendercanvas's inner render
    widget, whose ``keyPressEvent`` swallows the event — so Backspace did
    nothing at exactly the moment it is wanted. The filter has to go on
    ``CameraView.key_target()``, the widget the press actually lands on.
    """
    dialog.store.set_point(0, "eye", (5.0, 6.0))
    _select_leaf(dialog, "individual_0", "eye")

    _key(dialog, Qt.Key_Backspace, target=dialog._view.key_target())

    assert dialog._view.key_target() is not dialog._view.canvas_widget()
    assert dialog.store.is_anchor(0, "eye") is False


def test_ctrl_z_undoes_from_the_points_table(dialog):
    """The other half of the key table: the dialog's filter claims Ctrl+Z, and
    the main window binds neither it nor Backspace. Pressed with the table
    focused, since that is where you are while reading down a fill."""
    dialog.store.set_point(0, "beak", (5.0, 6.0))

    _key(dialog, Qt.Key_Z, Qt.ControlModifier, target=dialog.point_table)

    assert dialog.store.is_anchor(0, "beak") is False


def test_tree_marks_the_points_labelled_on_this_frame(dialog):
    dialog.store.set_point(0, "tail", (5.0, 6.0))
    dialog._refresh_tree_marks()

    branch = dialog.tree.topLevelItem(0)
    assert branch.text(1) == f"1/{len(NAMES)}"
    assert branch.child(0).text(1) != branch.child(1).text(1)


def test_removing_the_last_individual_empties_the_tree(dialog):
    """Deleting every individual is allowed — the tree simply goes empty."""
    dialog._on_remove_individual()

    assert dialog.store.n_individuals == 0
    assert dialog.tree.topLevelItemCount() == 0
    assert dialog.app_state.labelling_individuals == []


def test_labelling_needs_an_individual(dialog):
    dialog._on_remove_individual()
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    assert dialog._mode is None
    assert dialog.sequential_btn.isChecked() is False


def test_removing_the_last_individual_stops_labelling(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._on_remove_individual()

    assert dialog._mode is None
    assert dialog.sequential_btn.isChecked() is False


def test_adding_an_individual_back_restores_labelling(dialog):
    dialog._on_remove_individual()
    dialog._apply_schema(individuals=["crow"])
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    assert dialog._mode is not None
    assert dialog._mode.active_individual == "crow"


# ----------------------------------------------------------------------
# Asymmetric schemas: individuals with their own keypoints
# ----------------------------------------------------------------------


def test_unsharing_keeps_every_individual_on_the_full_schema(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)

    assert dialog.store.shared_keypoints is False
    assert _tree_names(dialog.tree) == [("a", NAMES), ("b", NAMES)]


def test_adding_a_keypoint_affects_only_the_selected_individual(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)
    dialog._apply_schema(individual_keypoints=("b", [*NAMES, "wing"]))

    assert _tree_names(dialog.tree) == [("a", NAMES), ("b", [*NAMES, "wing"])]
    assert dialog.store.keypoint_names == [*NAMES, "wing"]
    assert dialog.store.n_schema_points == len(NAMES) + len(NAMES) + 1


def test_removing_a_keypoint_from_one_individual_keeps_it_on_the_other(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)
    dialog._apply_schema(individual_keypoints=("a", ["beak"]))

    assert _tree_names(dialog.tree) == [("a", ["beak"]), ("b", NAMES)]


def test_tree_marks_count_the_individuals_own_keypoints(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)
    dialog._apply_schema(individual_keypoints=("a", ["beak", "eye"]))
    dialog.store.set_point(0, "eye", (5.0, 6.0), "a")
    dialog._refresh_tree_marks()

    branch = dialog.tree.topLevelItem(0)
    assert branch.text(1) == "1/2"
    assert branch.child(1).text(1) != branch.child(0).text(1)


def test_resharing_gives_everyone_the_union(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.shared_toggle.setChecked(False)
    dialog._apply_schema(individual_keypoints=("b", [*NAMES, "wing"]))
    dialog.shared_toggle.setChecked(True)

    assert _tree_names(dialog.tree) == [("a", [*NAMES, "wing"]), ("b", [*NAMES, "wing"])]


# ----------------------------------------------------------------------
# Keypoint colours
# ----------------------------------------------------------------------


def _leaf_colour(dialog, individual: str, keypoint: str) -> str:
    """The colour the tree paints a leaf's per-frame mark in."""
    for i in range(dialog.tree.topLevelItemCount()):
        branch = dialog.tree.topLevelItem(i)
        for k in range(branch.childCount()):
            leaf = branch.child(k)
            if leaf.data(0, Qt.UserRole) == (individual, keypoint):
                return leaf.foreground(1).color().name()
    raise AssertionError(f"no leaf for {(individual, keypoint)}")


def test_a_pinned_colour_reaches_the_tree_and_the_canvas(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    _select_leaf(dialog, "individual_0", "tail")

    dialog.store.set_keypoint_color("tail", "#ff8000")
    dialog._apply_keypoint_colors()

    assert _leaf_colour(dialog, "individual_0", "tail") == "#ff8000"
    colors = dialog._mode._overlay._solid.geometry.colors.data
    np.testing.assert_allclose(colors[1], [1.0, 128 / 255.0, 0.0, 1.0])


def _set_color_by(dialog, mode: str) -> None:
    """Pick a colour mode from the combo, as the user would."""
    dialog.color_by_combo.setCurrentIndex(dialog.color_by_combo.findData(mode))


def test_colour_by_individual_gives_every_keypoint_of_an_animal_one_colour(dialog):
    dialog._apply_schema(individuals=["a", "b"])

    _set_color_by(dialog, COLOR_BY_INDIVIDUAL)

    assert dialog.app_state.pose_color_by == COLOR_BY_INDIVIDUAL
    assert _leaf_colour(dialog, "a", "beak") == _leaf_colour(dialog, "a", "tail")
    assert _leaf_colour(dialog, "a", "beak") != _leaf_colour(dialog, "b", "beak")


def test_colour_by_keypoint_shares_a_colour_across_individuals(dialog):
    dialog._apply_schema(individuals=["a", "b"])

    _set_color_by(dialog, COLOR_BY_KEYPOINT)

    assert _leaf_colour(dialog, "a", "beak") == _leaf_colour(dialog, "b", "beak")
    assert _leaf_colour(dialog, "a", "beak") != _leaf_colour(dialog, "a", "tail")


def test_the_colour_mode_reaches_the_canvas(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    _set_color_by(dialog, COLOR_BY_INDIVIDUAL)

    colors = dialog._mode._overlay.vertex_colors(active_individual=0)
    np.testing.assert_allclose(colors[0, :3], colors[1, :3])  # one animal, one colour


def test_the_picker_colours_the_individual_while_colour_means_individual(dialog, monkeypatch):
    """Otherwise Colour… would edit a palette nothing on screen is drawing."""
    _set_color_by(dialog, COLOR_BY_INDIVIDUAL)
    _select_leaf(dialog, "individual_0", "tail")
    monkeypatch.setattr(dialog_module.QColorDialog, "getColor", lambda *a, **k: QColor("#ff8000"))

    dialog._on_keypoint_color()

    assert dialog.store.individual_color == {"individual_0": "#ff8000"}
    assert dialog.store.keypoint_color == {}
    assert _leaf_colour(dialog, "individual_0", "beak") == "#ff8000"


def test_the_dialog_opens_in_the_apps_colour_mode(qapp, tmp_path):
    """One setting styles the overlay and the canvas, so the dialog inherits it."""
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state.keypoints = list(NAMES)
    state.pose_color_by = COLOR_BY_INDIVIDUAL

    dlg = PoseLabellingDialog(_FakeDataWidget(state))
    try:
        assert dlg.color_by_combo.currentData() == COLOR_BY_INDIVIDUAL
    finally:
        dlg.close()


def test_colouring_a_keypoint_keeps_the_mode_and_the_active_pair(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    _select_leaf(dialog, "individual_0", "tail")

    dialog.store.set_keypoint_color("tail", "#ff8000")
    dialog._apply_keypoint_colors()

    assert dialog.interaction_mode == SEQUENTIAL_MODE
    assert dialog._mode.active_keypoint == "tail"


def test_a_colour_leaves_the_labels_and_the_fill_alone(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    for frame in (2, 5):
        dialog.store.set_point(frame, "tail", (float(frame), 1.0))
    anchors, n_frames = dialog.store.flat_anchors(), dialog.store.n_frames
    dialog.store.set_fill_from_flat(*SplineBackend().fill(anchors, n_frames, None))

    dialog.store.set_keypoint_color("tail", "#ff8000")
    dialog._apply_keypoint_colors()

    assert dialog.store.is_anchor(2, "tail") is True
    assert dialog.store.fill_range == (2, 5)


def test_resetting_hands_every_keypoint_back_to_the_palette(dialog):
    dialog.store.set_keypoint_color("tail", "#ff8000")
    dialog._apply_keypoint_colors()
    before = _leaf_colour(dialog, "individual_0", "beak")

    dialog._on_reset_keypoint_colors()

    assert dialog.store.keypoint_color == {}
    assert _leaf_colour(dialog, "individual_0", "beak") == before
    assert _leaf_colour(dialog, "individual_0", "tail") != "#ff8000"


def test_the_reset_button_is_off_until_a_colour_is_pinned(dialog):
    assert dialog._reset_colours_btn.isEnabled() is False

    dialog.store.set_keypoint_color("tail", "#ff8000")
    dialog._apply_keypoint_colors()

    assert dialog._reset_colours_btn.isEnabled() is True


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------


def test_the_dialog_is_split_into_stages(dialog):
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "Define keypoints",
        "Label && Edit",  # && escapes the mnemonic
        # Between labelling and filling, as in the pipeline: Detect produces
        # observations, and the fill bridges what is left.
        "Detect",
        "Fill and export",
    ]


def test_arming_a_mode_shows_the_label_tab(dialog):
    dialog.tabs.setCurrentIndex(0)
    dialog.set_interaction_mode(SEQUENTIAL_MODE)

    assert dialog.tabs.currentWidget() is dialog._label_page


def test_disarming_leaves_the_current_tab_alone(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.tabs.setCurrentIndex(2)
    dialog.set_interaction_mode(None)

    assert dialog.tabs.currentIndex() == 2


# ----------------------------------------------------------------------
# Sequential vs loop mode
# ----------------------------------------------------------------------


def test_the_buttons_arm_and_disarm_their_mode(dialog):
    dialog.sequential_btn.click()
    assert dialog.interaction_mode == SEQUENTIAL_MODE
    assert dialog.sequential_btn.isChecked() is True

    dialog.sequential_btn.click()
    assert dialog.interaction_mode is None
    assert dialog.sequential_btn.isChecked() is False


def test_switching_mode_keeps_the_same_canvas_mode_object(dialog):
    dialog.sequential_btn.click()
    mode = dialog._mode
    dialog.loop_btn.click()

    assert dialog._mode is mode  # the canvas overlay is not rebuilt
    assert dialog.interaction_mode == LOOP_MODE
    assert dialog.loop_btn.isChecked() is True
    assert dialog.sequential_btn.isChecked() is False


def test_loop_mode_can_be_armed_from_cold(dialog):
    dialog.loop_btn.click()
    assert dialog.interaction_mode == LOOP_MODE


def test_the_lock_suspends_labelling_without_dropping_the_mode(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    mode = dialog._mode
    dialog.lock_check.setChecked(True)

    assert dialog._mode is mode  # still armed, still drawing its anchors
    assert dialog.interaction_mode == SEQUENTIAL_MODE
    assert mode.locked is True
    mode.handle_click(100.0, 100.0)
    assert dialog.store.anchor_frames() == []

    dialog.lock_check.setChecked(False)
    assert mode.locked is False
    mode.handle_click(100.0, 100.0)
    assert dialog.store.anchor_frames() == [0]


def test_the_lock_is_kept_when_the_mode_is_rearmed(dialog):
    """A schema change restarts the canvas mode; the lock must not fall off."""
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.lock_check.setChecked(True)
    dialog._apply_schema(keypoints=[*NAMES, "wing"])

    assert dialog._mode.locked is True


def test_the_lock_says_so_in_place_of_the_target(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.lock_check.setChecked(True)

    assert "Locked" in dialog.active_label.text()
    assert "beak" not in dialog.active_label.text()


def test_the_lock_is_only_offered_while_the_canvas_is_armed(dialog):
    dialog.set_interaction_mode(None)
    assert dialog.lock_check.isEnabled() is False
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog.lock_check.isEnabled() is True


def test_leaving_the_label_tab_locks_the_pointer(dialog):
    """A click while detecting or filling must not drop a stray point.

    The mode itself survives — this is the lock, not a disarm — so the anchor
    overlay stays up and coming back carries on where it left off.
    """
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    mode = dialog._mode
    assert mode.locked is False

    others = [
        dialog.tabs.widget(index)
        for index in range(dialog.tabs.count())
        if dialog.tabs.widget(index) is not dialog._label_page
    ]
    assert len(others) == 3, "Define keypoints, Detect, Fill and export"
    for page in others:
        dialog.tabs.setCurrentWidget(page)
        assert dialog._mode is mode, "the mode is locked, not dropped"
        assert mode.locked is True
        mode.handle_click(100.0, 100.0)
        assert dialog.store.anchor_frames() == []

    dialog.tabs.setCurrentWidget(dialog._label_page)
    assert mode.locked is False
    mode.handle_click(100.0, 100.0)
    assert dialog.store.anchor_frames() == [0]


def test_the_tabs_never_rewrite_the_users_own_lock(dialog):
    """Two reasons to lock, kept apart — or a round trip would strand it on.

    Syncing the tick box on a tab change would leave labelling locked with no
    memory of who locked it, so the box stays the user's standing intent and
    the tab is applied on top of it.
    """
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog.tabs.setCurrentWidget(dialog._detect_page)
    assert dialog.lock_check.isChecked() is False, "the tab locked it, not the user"
    dialog.tabs.setCurrentWidget(dialog._label_page)
    assert dialog._mode.locked is False

    # And the other way: a lock the user set outlives the round trip.
    dialog.lock_check.setChecked(True)
    dialog.tabs.setCurrentWidget(dialog._detect_page)
    dialog.tabs.setCurrentWidget(dialog._label_page)
    assert dialog.lock_check.isChecked() is True
    assert dialog._mode.locked is True


def test_the_active_line_names_what_the_next_click_places(dialog):
    dialog.set_interaction_mode(LOOP_MODE)
    text = dialog.active_label.text()

    assert "individual_0" in text
    assert "beak" in text
    assert "Loop" in text


def test_mode_survives_a_schema_change(dialog):
    dialog.set_interaction_mode(LOOP_MODE)
    dialog._apply_schema(keypoints=[*NAMES, "wing"])

    assert dialog.interaction_mode == LOOP_MODE


# ----------------------------------------------------------------------
# Labelled-frames table
# ----------------------------------------------------------------------


def _table_headers(dialog) -> list[str]:
    """Fixed labels plus the keypoint groups the header paints over each pair."""
    model = dialog.point_model
    fixed = [model.headerData(col, Qt.Horizontal) for col in range(len(_FIXED_COLUMNS))]
    return fixed + dialog.point_table.horizontalHeader().groups()


def _header_tooltips(dialog) -> list[str]:
    """First line of each keypoint column's tooltip — the ``conf`` ones then
    carry the whole scoring scheme beneath it."""
    model = dialog.point_model
    return [
        model.headerData(col, Qt.Horizontal, Qt.ToolTipRole).splitlines()[0]
        for col in range(len(_FIXED_COLUMNS), model.columnCount())
    ]


def _column_of(dialog, keypoint: str, axis: str) -> int:
    """Where a keypoint's ``x``, ``y`` or ``conf`` cell sits, by name."""
    index = dialog.point_model.keypoint_columns.index(keypoint)
    return len(_FIXED_COLUMNS) + COLUMNS_PER_KEYPOINT * index + ("x", "y", "conf").index(axis)


def _table_rows(dialog) -> list[tuple[str, ...]]:
    """Every visible row, as text — through the proxy, so filters apply."""
    proxy = dialog.point_proxy
    return [
        tuple(proxy.index(row, col).data() or "" for col in range(proxy.columnCount()))
        for row in range(proxy.rowCount())
    ]


def _fill(store, value: float = 5.0) -> None:
    """A backend result covering every frame, so the store has predictions."""
    shape = (store.n_frames, store.n_individuals, store.n_keypoints)
    store.set_fill(np.full((*shape, 2), value), np.full(shape, 0.5))


def test_a_row_holds_every_keypoint_labelled_on_that_frame(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog.store.set_point(3, "tail", (30.0, 40.0))
    dialog.store.set_point(7, "beak", (50.0, 60.0))
    dialog._refresh_point_table()

    assert _table_headers(dialog) == ["Frame", "Individual", "Source", "beak", "tail"]
    assert _table_rows(dialog) == [
        ("3", "individual_0", "Human", "10.0", "20.0", "", "30.0", "40.0", ""),
        ("7", "individual_0", "Human", "50.0", "60.0", "", "", "", ""),
    ]


def test_each_keypoint_spans_an_x_a_y_and_a_confidence_column(dialog):
    """One header name over three columns — repeating it in each was the waste."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog.store.set_point(3, "tail", (30.0, 40.0))
    dialog._refresh_point_table()

    assert dialog.point_model.columnCount() == len(_FIXED_COLUMNS) + COLUMNS_PER_KEYPOINT * 2
    assert _header_tooltips(dialog) == [
        "beak x",
        "beak y",
        "beak — fill confidence",
        "tail x",
        "tail y",
        "tail — fill confidence",
    ]


def test_the_two_row_header_renders(dialog):
    """Custom paintSection code only breaks when something actually paints."""
    dialog.store.set_point(3, "beak", (1234.5, 20.0))
    dialog.store.set_point(3, "tail", (30.0, 40.0))
    dialog._refresh_point_table()

    table = dialog.point_table
    table.resize(600, 200)
    pixmap = QPixmap(table.size())
    table.render(pixmap)

    assert not pixmap.isNull()
    assert table.horizontalHeader().height() > table.rowHeight(0)  # two rows of labels


def test_unlabelled_keypoints_get_no_columns(dialog):
    """A 20-keypoint schema must not bury the two columns being worked on."""
    dialog.store.set_point(3, "eye", (1.0, 2.0))
    dialog._refresh_point_table()

    assert _table_headers(dialog) == ["Frame", "Individual", "Source", "eye"]


def test_each_individual_gets_its_own_row(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.store.set_point(4, "beak", (1.0, 2.0), "a")
    dialog.store.set_point(4, "beak", (3.0, 4.0), "b")
    dialog._refresh_point_table()

    assert _table_rows(dialog) == [
        ("4", "a", "Human", "1.0", "2.0", ""),
        ("4", "b", "Human", "3.0", "4.0", ""),
    ]


def test_table_drops_deleted_points(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog._refresh_point_table()
    dialog.store.clear_point(3, "beak")
    dialog._refresh_point_table()

    assert _table_rows(dialog) == []


def test_deleting_one_keypoint_blanks_only_its_cells(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog.store.set_point(3, "tail", (30.0, 40.0))
    dialog.store.set_point(7, "tail", (50.0, 60.0))
    dialog._refresh_point_table()
    dialog.store.clear_point(3, "tail")
    dialog._refresh_point_table()

    assert _table_rows(dialog)[0] == ("3", "individual_0", "Human", "10.0", "20.0", "", "", "", "")


def test_moving_a_point_updates_its_cells_in_place(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    dialog._refresh_point_table()
    dialog.store.set_point(3, "beak", (99.0, 98.0))
    dialog._refresh_point_table()

    assert _table_rows(dialog) == [("3", "individual_0", "Human", "99.0", "98.0", "")]


def _click_cell(dialog, row: int, column: int) -> None:
    dialog._on_table_clicked(dialog.point_proxy.index(row, column))


def test_clicking_a_keypoint_cell_makes_that_point_active(dialog):
    dialog.store.set_point(3, "beak", (1.0, 2.0))
    dialog.store.set_point(3, "eye", (10.0, 20.0))
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._refresh_point_table()

    _click_cell(dialog, 0, _column_of(dialog, "eye", "x"))

    assert dialog._mode.active_keypoint == "eye"


def test_clicking_the_frame_cell_keeps_the_active_keypoint(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.store.set_point(3, "tail", (1.0, 2.0), "b")
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._refresh_point_table()

    _click_cell(dialog, 0, 0)

    assert dialog._mode.active_individual == "b"


def test_an_undo_repaints_the_frame_it_landed_on(dialog):
    """Undo can revert a frame you are not looking at — that row must repaint."""
    _fill(dialog.store)  # dense rows, so the layout cannot change and mask this
    dialog.store.set_point(2, "beak", (10.0, 20.0))
    dialog.app_state.current_frame = 8
    dialog._refresh_point_table(full=True)

    repainted = []
    dialog.point_model.dataChanged.connect(lambda first, last, *_: repainted.append((first.row(), last.row())))
    dialog._on_undo()

    assert (2, 2) in repainted  # the undone frame, not the one on screen
    assert _table_rows(dialog)[2][SOURCE_COLUMN] == "Fill"


# ----------------------------------------------------------------------
# Which frames to label, and navigating them
# ----------------------------------------------------------------------


def _seeks(dialog, monkeypatch) -> list[int]:
    """Record where the dialog seeks — the fake view has no video to move."""
    landed: list[int] = []
    monkeypatch.setattr(dialog, "_seek", lambda frame: landed.append(int(frame)))
    return landed


def test_the_methods_are_ordered_by_when_they_apply(dialog):
    """The three that need nothing but the video come first.

    Then the two that rank what a stage already produced, in the order those
    stages run: Detect, then Fill.
    """
    combo = dialog.suggest_method_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "uniform",
        "motion",
        "diverse",
        "detection_gaps",
        "uncertain",
    ]


def test_the_share_resolves_to_a_frame_count(dialog):
    dialog.suggest_percent_spin.setValue(20.0)

    assert dialog._suggest_count() == 2  # of the fake view's 10 frames
    assert dialog.suggest_count_label.text() == "2 of 10 frames"


def test_the_share_never_resolves_to_zero_frames(dialog):
    dialog.suggest_percent_spin.setValue(dialog.suggest_percent_spin.minimum())
    assert dialog._suggest_count() == 1


def test_the_default_share_is_a_spacing_not_a_count(dialog):
    """Roughly every 10th frame, whatever the clip length — the backends bridge
    gaps, and a gap is measured in frames."""
    assert dialog._default_suggest_percent() == RECOMMENDED_LABEL_SHARE
    assert dialog.suggest_percent_spin.value() == RECOMMENDED_LABEL_SHARE
    assert dialog._suggest_count() == 1  # of the fake view's 10 frames


def test_n_steps_to_the_next_suggested_frame(dialog, monkeypatch):
    """One direction only — the suggestions are a queue, and the points table
    is how you go back to any frame at all."""
    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_N)
    _key(dialog, Qt.Key_N)

    assert landed == [5, 9]


def test_n_wraps_at_the_end_of_the_suggestions(dialog, monkeypatch):
    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 2
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_N)

    assert landed == [1]


def test_n_reaches_the_dialog_from_the_main_window(dialog, monkeypatch):
    """It is pressed while looking at the video, not at the dialog."""
    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_N, target=dialog._shell)

    assert landed == [5]


def test_n_steps_once_per_press_with_the_table_focused(dialog, qapp, monkeypatch):
    """``N`` is printable, so ``QAbstractItemView`` eats it as a type-ahead
    search — only the dialog's own QShortcut runs early enough. And exactly one
    of the shortcut and the event filter may act, or a press skips a frame."""
    from qtpy.QtTest import QTest

    dialog._suggestions = [1, 5, 9]
    dialog._suggestion_index = 0
    dialog.show()
    dialog.activateWindow()
    dialog.point_table.setFocus()
    QApplication.processEvents()
    landed = _seeks(dialog, monkeypatch)

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_N)
    QApplication.processEvents()

    assert landed == [5]


def test_the_main_window_keeps_its_number_keys(dialog):
    """`1`-`9` are behaviour labels over there, so they must not reach across —
    unlike Backspace and Ctrl+Z, which nothing in the main window binds."""
    dialog._suggestions = [1, 5, 9]
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._apply_schema(individuals=["a", "b"])

    _key(dialog, Qt.Key_2, target=dialog._shell)

    assert dialog._mode.active_individual == "a"


def test_the_arrows_are_left_to_the_main_window(dialog, monkeypatch):
    """Both plain and Shift: frame stepping and window stepping stay global
    shortcuts, and the suggestions moved off the arrows entirely."""
    dialog._suggestions = [1, 5, 9]
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_Right)
    _key(dialog, Qt.Key_Right, Qt.ShiftModifier)
    _key(dialog, Qt.Key_Left, Qt.ShiftModifier)

    assert landed == []


def test_n_warns_instead_of_seeking_without_suggestions(dialog, monkeypatch):
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_N)

    assert landed == []


def test_loop_stays_on_the_frame_when_asked(dialog, monkeypatch):
    dialog.set_interaction_mode(LOOP_MODE)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("stay"))
    landed = _seeks(dialog, monkeypatch)

    dialog._advance_frame()

    assert landed == []


def test_loop_steps_one_frame_when_asked(dialog, monkeypatch):
    dialog.set_interaction_mode(LOOP_MODE)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("frame"))
    dialog.app_state.current_frame = 3
    landed = _seeks(dialog, monkeypatch)

    dialog._advance_frame()

    assert landed == [4]


def test_loop_follows_the_suggestions_when_asked(dialog, monkeypatch):
    dialog.set_interaction_mode(LOOP_MODE)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("suggestion"))
    dialog._suggestions = [2, 6]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    dialog._advance_frame()

    assert landed == [6]


def test_the_then_go_to_row_belongs_to_loop_mode(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog.after_click_row.isVisibleTo(dialog) is False

    dialog.set_interaction_mode(LOOP_MODE)
    assert dialog.after_click_row.isVisibleTo(dialog) is True


def test_the_then_go_to_row_also_appears_for_approving(dialog):
    """It says where Shift+H lands too, and approving needs no mode.

    ``isHidden`` rather than ``isVisibleTo``: with no mode armed the dialog is
    still on the Keypoints tab, so the whole Label page is hidden — arming one
    to look would defeat the point of the test.
    """
    dialog.set_interaction_mode(None)
    assert dialog.after_click_row.isHidden() is True

    _fill(dialog.store)
    dialog._refresh_active_label()

    assert dialog.after_click_row.isHidden() is False


# ----------------------------------------------------------------------
# Approving a fill (Shift+H)
# ----------------------------------------------------------------------


def test_shift_h_keeps_the_frames_predictions_as_labels(dialog):
    _fill(dialog.store, value=7.0)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert all(dialog.store.is_anchor(4, name) for name in NAMES)
    assert dialog.store.anchor_positions_for(4).tolist() == [[7.0, 7.0]] * len(NAMES)


def test_shift_h_leaves_the_other_frames_predicted(dialog):
    """Approving is per frame — that is what makes it a review step."""
    _fill(dialog.store)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert dialog.store.is_human(4) is True
    assert dialog.store.is_human(5) is False


def test_shift_h_never_overwrites_a_label_you_placed(dialog):
    dialog.store.set_point(4, "beak", (1.0, 2.0))
    _fill(dialog.store, value=7.0)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert dialog.store.anchor_positions_for(4)[0].tolist() == [1.0, 2.0]
    assert dialog.store.anchor_positions_for(4)[1].tolist() == [7.0, 7.0]


def test_shift_h_then_goes_where_the_dropdown_says(dialog, monkeypatch):
    _fill(dialog.store)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("frame"))
    dialog.app_state.current_frame = 4
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert landed == [5]


def test_shift_h_can_follow_the_suggestions_instead(dialog, monkeypatch):
    _fill(dialog.store)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("suggestion"))
    dialog._suggestions = [2, 6]
    dialog._suggestion_index = 0
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert landed == [6]


def test_shift_h_moves_on_from_an_already_approved_frame(dialog, monkeypatch):
    """Agreeing with a frame that is already yours still means "next"."""
    _fill(dialog.store)
    dialog.app_state.current_frame = 4
    dialog.store.promote_fill(4)
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("frame"))
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert landed == [5]


def test_shift_h_stays_put_when_the_frame_carries_nothing(dialog, monkeypatch):
    """Silently advancing past empty frames would look like a broken key."""
    dialog.app_state.current_frame = 4
    landed = _seeks(dialog, monkeypatch)

    _key(dialog, Qt.Key_H, Qt.ShiftModifier)

    assert landed == []


def test_plain_h_is_left_to_the_main_windows_behaviour_label(dialog):
    _fill(dialog.store)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H)

    assert dialog.store.is_human(4) is False


def test_shift_h_works_when_the_points_table_has_focus(dialog, qapp):
    """Regression: the table ate it as type-ahead search.

    You pick the frame to review by clicking its row, which leaves focus on the
    table — and ``QAbstractItemView`` turns any printable key into a keyboard
    search and accepts it, so the press never propagated to the dialog's filter.
    Only the dialog's own QShortcut runs early enough.
    """
    from qtpy.QtTest import QTest

    _fill(dialog.store)
    dialog.show()
    dialog.activateWindow()
    dialog.app_state.current_frame = 4
    dialog.point_table.setFocus()
    QApplication.processEvents()

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_H, Qt.ShiftModifier)
    QApplication.processEvents()

    assert dialog.store.is_human(4) is True


def test_shift_h_works_when_the_tree_has_focus(dialog, qapp):
    """Same swallowing, from the other item view."""
    from qtpy.QtTest import QTest

    _fill(dialog.store)
    dialog.show()
    dialog.activateWindow()
    dialog.app_state.current_frame = 4
    dialog.tree.setFocus()
    QApplication.processEvents()

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_H, Qt.ShiftModifier)
    QApplication.processEvents()

    assert dialog.store.is_human(4) is True


def test_shift_h_approves_once_per_press(dialog, qapp, monkeypatch):
    """The QShortcut and the event filter both know the key — only one may act.

    Firing twice would silently skip the frame after the one approved.
    """
    from qtpy.QtTest import QTest

    _fill(dialog.store)
    dialog.show()
    dialog.activateWindow()
    dialog.after_click_combo.setCurrentIndex(dialog.after_click_combo.findData("frame"))
    dialog.app_state.current_frame = 4
    dialog.tree.setFocus()
    QApplication.processEvents()
    landed = _seeks(dialog, monkeypatch)

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_H, Qt.ShiftModifier)
    QApplication.processEvents()

    assert landed == [5]


def test_shift_h_reaches_across_from_the_main_window(dialog):
    """You press it while looking at the video, not at the dialog."""
    _fill(dialog.store)
    dialog.app_state.current_frame = 4

    _key(dialog, Qt.Key_H, Qt.ShiftModifier, target=dialog._shell)

    assert dialog.store.is_human(4) is True


def test_the_approve_button_waits_for_a_fill(dialog):
    """With no predictions on screen there is nothing to approve."""
    assert dialog.approve_btn.isHidden() is True

    _fill(dialog.store)
    dialog._refresh_active_label()

    assert dialog.approve_btn.isHidden() is False


# ----------------------------------------------------------------------
# Human labels vs filled predictions
# ----------------------------------------------------------------------


def test_filling_gives_every_frame_a_row(dialog):
    """A prediction you cannot see is a prediction you cannot correct."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog._refresh_point_table()

    assert dialog.point_model.rowCount() == dialog.store.n_frames
    assert _table_headers(dialog) == ["Frame", "Individual", "Source", *NAMES]


def test_the_table_stops_where_the_fill_does(dialog):
    """A fill bridges the labels and no further, and rows follow it.

    Rows past the last label would be permanently empty ones — nothing can ever
    populate them short of labelling further out, which changes the span anyway.
    """
    for frame in (2, 5):
        dialog.store.set_point(frame, "beak", (float(frame), 1.0))
    anchors, n_frames = dialog.store.flat_anchors(), dialog.store.n_frames
    dialog.store.set_fill_from_flat(*SplineBackend().fill(anchors, n_frames, None))
    dialog._refresh_point_table(full=True)

    assert {frame for frame, _individual in dialog.point_model.rows} == {2, 3, 4, 5}


def test_filled_rows_are_marked_fill_and_labelled_ones_human(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)  # confidence 0.5 everywhere, 1.0 on the labelled point
    dialog._refresh_point_table()

    rows = _table_rows(dialog)
    labelled = ("10.0", "20.0", "1.00")
    predicted = ("5.0", "5.0", "0.50")
    assert rows[3] == ("3", "individual_0", "Human", *labelled, *predicted, *predicted)
    assert rows[4] == ("4", "individual_0", "Fill", *predicted, *predicted, *predicted)


def test_one_corrected_keypoint_makes_the_whole_row_human(dialog):
    """The rule: touch one point of a (frame, individual) and the row is yours."""
    _fill(dialog.store)
    dialog.store.set_point(6, "tail", (1.0, 2.0))
    dialog._refresh_point_table(full=True)

    assert _table_rows(dialog)[6][SOURCE_COLUMN] == "Human"


def test_predicted_coordinates_are_dimmed(dialog):
    """Per-point provenance, so a mixed row still reads correctly. A keypoint's
    score is dimmed with its coordinates — the triple is one thing."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog._refresh_point_table()

    def dimmed(keypoint, axis):
        return dialog.point_proxy.index(3, _column_of(dialog, keypoint, axis)).data(Qt.ForegroundRole) is not None

    assert dimmed("beak", "x") is False  # labelled
    assert dimmed("beak", "conf") is False
    assert dimmed("tail", "x") is True  # predicted
    assert dimmed("tail", "conf") is True


def test_a_filled_row_can_be_deleted_from_the_table(dialog):
    """Regression: "Delete labels" is a no-op on a row that is all prediction."""
    _fill(dialog.store)
    dialog._refresh_point_table()

    dialog._clear_fill_rows([(4, "individual_0")])

    assert dialog.store.has_fill(4) is False
    row = _table_rows(dialog)[4]
    assert row[SOURCE_COLUMN] == ""  # no source left to name
    assert row[len(_FIXED_COLUMNS) :] == ("",) * (COLUMNS_PER_KEYPOINT * len(NAMES))


def test_deleting_a_filled_row_keeps_the_labels_on_it(dialog):
    _fill(dialog.store)
    dialog.store.set_point(4, "beak", (10.0, 20.0))
    dialog._refresh_point_table(full=True)

    dialog._clear_fill_rows([(4, "individual_0")])

    row = _table_rows(dialog)[4]
    assert row[: len(_FIXED_COLUMNS)] == ("4", "individual_0", "Human")
    assert row[_column_of(dialog, "beak", "x")] == "10.0"
    assert row[_column_of(dialog, "beak", "conf")] == "1.00"  # the label survived the delete


def test_the_canvas_shows_predictions_alongside_labels(dialog):
    """Judging a prediction means seeing it — hollow, next to the solid labels."""
    dialog.store.set_point(0, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    overlay = dialog._mode._overlay

    drawn = overlay._solid.geometry.positions.data[:, 0] > -1e5
    predicted = overlay._hollow.geometry.positions.data[:, 0] > -1e5
    assert list(drawn) == [True, False, False]
    assert list(predicted) == [False, True, True]


def test_the_pose_overlay_yields_the_canvas_while_a_mode_is_armed(dialog):
    """Otherwise every point carries two markers, one of them provenance-blind."""
    _fill(dialog.store)
    dialog._push_pose_override()
    assert dialog._data_widget.pose_mgr.override is not None

    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog._data_widget.pose_mgr.override is None

    dialog.set_interaction_mode(None)
    assert dialog._data_widget.pose_mgr.override is not None


def test_the_legend_appears_only_once_predictions_are_on_screen(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    assert dialog.legend_label.isVisibleTo(dialog) is False

    _fill(dialog.store)
    dialog._refresh_active_label()
    assert dialog.legend_label.isVisibleTo(dialog) is True


def test_the_disagreement_tolerance_belongs_to_the_tracking_backends(dialog):
    """The spline scores by distance from an anchor — the tolerance is meaningless."""
    combo = dialog.backend_combo
    combo.setCurrentIndex(combo.findData("spline"))
    assert dialog.disagreement_row.isHidden() is True

    combo.setCurrentIndex(combo.findData("flow"))
    assert dialog.disagreement_row.isHidden() is False


def test_the_disagreement_tolerance_is_remembered(dialog):
    dialog.disagreement_spin.setValue(25.0)
    assert dialog.app_state.labelling_disagreement_px == 25.0


def test_custom_weights_belong_to_posepal(dialog):
    """Only PosePAL loads a state dict — the others have nothing to point at."""
    combo = dialog.backend_combo
    combo.setCurrentIndex(combo.findData("flow"))
    assert dialog.checkpoint_row.isHidden() is True

    combo.setCurrentIndex(combo.findData(dialog_module.POSEPAL_BACKEND))
    assert dialog.checkpoint_row.isHidden() is False


def test_custom_weights_are_remembered(dialog, tmp_path):
    """The stock checkpoint is a default: fine-tuned weights must not mean editing pose_fill."""
    weights = tmp_path / "animals.pth"
    dialog.checkpoint_edit.setText(str(weights))
    dialog.checkpoint_edit.editingFinished.emit()
    assert dialog.app_state.labelling_cotracker_checkpoint == str(weights)


def test_no_custom_weights_means_the_stock_checkpoint(dialog):
    """Empty is the default, not a path — build_backend must see None, not ''."""
    assert dialog.app_state.labelling_cotracker_checkpoint == ""
    assert (dialog.app_state.labelling_cotracker_checkpoint or None) is None


def test_the_confidence_columns_are_empty_before_a_fill(dialog):
    """There is nothing to be confident about until a backend has run."""
    dialog.store.set_point(3, "beak", (1.0, 2.0))
    dialog._refresh_point_table()

    assert _table_rows(dialog)[0][_column_of(dialog, "beak", "conf")] == ""


def test_each_keypoint_carries_its_own_score(dialog):
    """The whole point of the split: one lost point must not be averaged away."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)  # 0.5 everywhere, 1.0 on the labelled point
    dialog.store.confidence[3, 0, 2] = 0.1  # 'eye'
    dialog._refresh_point_table(full=True)

    row = _table_rows(dialog)[3]
    assert row[_column_of(dialog, "beak", "conf")] == "1.00"  # placed by hand
    assert row[_column_of(dialog, "tail", "conf")] == "0.50"
    assert row[_column_of(dialog, "eye", "conf")] == "0.10"


def test_a_deleted_point_carries_no_confidence(dialog):
    """The fill array is a snapshot; a blank cell must not keep a score beside it."""
    _fill(dialog.store)
    dialog._refresh_point_table(full=True)
    dialog._clear_fill_rows([(4, "individual_0")])

    row = _table_rows(dialog)[4]
    assert row[_column_of(dialog, "beak", "x")] == ""
    assert row[_column_of(dialog, "beak", "conf")] == ""


def test_the_confidence_header_explains_the_number(dialog):
    """The column is meaningless without saying what produced it."""
    dialog.store.set_point(3, "beak", (1.0, 2.0))
    dialog._refresh_point_table()
    tooltip = dialog.point_model.headerData(_column_of(dialog, "beak", "conf"), Qt.Horizontal, Qt.ToolTipRole)

    assert tooltip.startswith("beak — fill confidence")
    assert "forwards" in tooltip and "backwards" in tooltip
    assert "Spline" in tooltip


def test_confidence_carries_no_funnel(dialog):
    """One filter per keypoint would AND — "beak *and* tail below 0.5" — when
    the question is always "any point below 0.5". The suggestion answers that."""
    dialog.store.set_point(3, "beak", (1.0, 2.0))
    dialog._refresh_point_table()
    header = dialog.point_table.horizontalHeader()

    assert header.filterable == {INDIVIDUAL_COLUMN, SOURCE_COLUMN}
    assert header.is_numeric(_column_of(dialog, "beak", "conf")) is False


def test_pinning_a_row_turns_its_predictions_into_labels(dialog):
    """ "Accepting" a fill: the next fill must treat these as ground truth."""
    _fill(dialog.store)
    dialog._refresh_point_table()

    dialog._pin_table_rows([(4, "individual_0")])

    assert dialog.store.is_human(4) is True
    assert 4 in dialog.store.flat_anchors()
    assert _table_rows(dialog)[4][SOURCE_COLUMN] == "Human"


# ----------------------------------------------------------------------
# Column filters
# ----------------------------------------------------------------------


def _frames_shown(dialog) -> list[str]:
    return [row[0] for row in _table_rows(dialog)]


def test_source_filter_keeps_only_the_labelled_rows(dialog):
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog._refresh_point_table()

    dialog.point_proxy.set_cat_filter(SOURCE_COLUMN, {"Human"})

    assert _frames_shown(dialog) == ["3"]


def test_individual_filter_keeps_one_animal(dialog):
    dialog._apply_schema(individuals=["a", "b"])
    dialog.store.set_point(4, "beak", (1.0, 2.0), "a")
    dialog.store.set_point(5, "beak", (3.0, 4.0), "b")
    dialog._refresh_point_table()

    dialog.point_proxy.set_cat_filter(INDIVIDUAL_COLUMN, {"b"})

    assert _table_rows(dialog) == [("5", "b", "Human", "3.0", "4.0", "")]


def test_clicking_a_funnel_applies_the_chosen_categories(dialog, monkeypatch):
    """The whole header path: funnel → popup → proxy → the funnel marked active."""
    dialog.store.set_point(3, "beak", (10.0, 20.0))
    _fill(dialog.store)
    dialog._refresh_point_table()

    class _PickedHuman:
        def __init__(self, *args):
            pass

        def exec_(self):
            return True

        def get_allowed(self):
            return {"Human"}

    monkeypatch.setattr(dialog_module, "CategoryFilterDialog", _PickedHuman)
    dialog._on_filter_requested(SOURCE_COLUMN)

    assert _frames_shown(dialog) == ["3"]
    assert dialog.point_proxy.active_filters() == {SOURCE_COLUMN}


def test_the_filterable_columns_are_the_categorical_ones(dialog):
    """Filtering a coordinate is meaningless, and neither Frame nor confidence
    earns a funnel — see ``test_confidence_carries_no_funnel``."""
    assert dialog.point_table.horizontalHeader().filterable == {INDIVIDUAL_COLUMN, SOURCE_COLUMN}


def test_a_schema_change_drops_the_filters(dialog):
    """A filter naming an individual that no longer exists empties the table."""
    dialog._apply_schema(individuals=["a", "b"])
    dialog.store.set_point(4, "beak", (1.0, 2.0), "a")
    dialog._refresh_point_table()
    dialog.point_proxy.set_cat_filter(INDIVIDUAL_COLUMN, {"b"})

    dialog._apply_schema(individuals=["a"])

    assert dialog.point_proxy.active_filters() == set()
    assert _frames_shown(dialog) == ["4"]


# ----------------------------------------------------------------------
# Test-time refinement
# ----------------------------------------------------------------------


def _select_backend(dialog, key: str) -> bool:
    index = dialog.backend_combo.findData(key)
    if index < 0:
        return False
    dialog.backend_combo.setCurrentIndex(index)
    return True


def test_the_refinement_row_belongs_to_posepal_alone(dialog):
    """Fit state is the one thing no other backend has — and it shows nowhere else."""
    # isHidden(), not isVisibleTo(): the tab holding the Fill group is itself
    # hidden while another tab is current, which says nothing about this row.
    _select_backend(dialog, "spline")
    dialog._refresh_backend_rows()
    assert dialog.refinement_row.isHidden()

    if not _select_backend(dialog, dialog_module.POSEPAL_BACKEND):
        pytest.skip("PosePAL not offered on this machine")
    dialog._refresh_backend_rows()
    assert not dialog.refinement_row.isHidden()


def test_an_unfitted_refinement_says_the_fill_will_fit(dialog):
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    assert "Not fitted" in dialog._refinement_status_text()


def test_labelling_another_frame_changes_the_refinement_signature(dialog):
    """A new label must mark a fit stale — it was made from fewer frames."""
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    before = dialog._refinement_signature()

    dialog.store.set_point(5, "beak", (3.0, 4.0))
    assert dialog._refinement_signature() != before

    # ...and moving a point counts just as much as adding one.
    moved = dialog._refinement_signature()
    dialog.store.set_point(5, "beak", (9.0, 9.0))
    assert dialog._refinement_signature() != moved


# ----------------------------------------------------------------------
# The refinement subclass: the labelling dialog minus schema/Detect
# ----------------------------------------------------------------------


def test_refinement_dialog_keeps_label_and_fill_only(qapp, tmp_path):
    from ethograph.gui.dialog_pose_refinement import SCOPE_MY_LABELS, PoseRefinementDialog

    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    dlg = PoseRefinementDialog(_FakeDataWidget(state))
    try:
        assert [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())] == [
            "Label && Edit",
            "Fill and save",
        ]
        # No resolvable pose file in the fake session: an empty store, and the
        # context label says why — the dialog must still construct and close.
        assert dlg.store.keypoint_names == []
        assert "No pose file resolves" in dlg.context_label.text()
        # The export group retires whole; the fill gains the scope choice.
        assert dlg.invert_y_check.parentWidget().isHidden()
        assert dlg.fill_scope_combo.currentData() == SCOPE_MY_LABELS
        # The Keypoints tree survives tab removal — key handling reads it.
        assert dlg.tree is not None
    finally:
        dlg.close()


def test_a_schema_change_changes_the_refinement_signature(dialog):
    """The delta is indexed by point row, so a renamed keypoint invalidates it."""
    dialog.store.set_point(0, "beak", (1.0, 2.0))
    before = dialog._refinement_signature()
    dialog._apply_schema(individuals=["a", "b"])
    assert dialog._refinement_signature() != before


def test_filling_is_the_only_fit_button(dialog):
    """Fill fits by itself when the labels have changed, so nothing else may.

    A second button for a step the first one already takes reads as a choice
    about the result, and there has never been one: a refit is a fresh fit.
    """
    assert not hasattr(dialog, "refit_btn")


class _CancellingBusy:
    """A progress dialog the user cancels as soon as work reports progress."""

    def __init__(self, label, parent=None):
        self.label = label

    def setLabelText(self, text) -> None:
        pass

    def pump_events(self) -> None:
        pass

    def wasCanceled(self) -> bool:
        return True

    def execute(self, fn, *args, **kwargs):
        return fn(*args, **kwargs), None


def test_cancelling_a_fill_keeps_the_fill_it_would_have_replaced(dialog, monkeypatch):
    """Cancelling is a way out of the wait, not a way to a worse fill.

    Backends answer a cancel with the spline seed they started from — they have
    arrays to return — so the dialog is what has to refuse it.
    """
    for frame in (0, 5):
        dialog.store.set_point(frame, "beak", (float(frame), 1.0))
    _fill(dialog.store, value=5.0)
    before = dialog.store.filled.copy()

    monkeypatch.setattr(dialog_module, "BusyProgressDialog", _CancellingBusy)
    dialog._on_fill()

    np.testing.assert_array_equal(dialog.store.filled, before)


def test_cancelling_the_first_fill_leaves_no_fill(dialog, monkeypatch):
    for frame in (0, 5):
        dialog.store.set_point(frame, "beak", (float(frame), 1.0))
    monkeypatch.setattr(dialog_module, "BusyProgressDialog", _CancellingBusy)

    dialog._on_fill()

    assert dialog.store.filled is None


# ----------------------------------------------------------------------
# Detect tab
# ----------------------------------------------------------------------


def _detections(store, frame: int, keypoint: str, xy=(11.0, 12.0), quality: float = 0.9) -> None:
    """One detected point on one frame, as a detector run would leave it."""
    points = np.full((store.n_individuals, store.n_keypoints, 2), np.nan)
    scores = np.full((store.n_individuals, store.n_keypoints), np.nan)
    points[0, store.keypoint_index(keypoint)] = xy
    scores[0, store.keypoint_index(keypoint)] = quality
    merged = dict(store.detections)
    merged_scores = dict(store.detection_confidence)
    merged[frame] = points
    merged_scores[frame] = scores
    store.set_detections(merged, merged_scores)


class _RecordingBackend:
    """A spline that records the sparse observations it was handed."""

    name = "recording"
    requires_video = False

    def __init__(self, seen):
        self._seen = seen

    def fill(self, anchors, n_frames, frames, progress):
        self._seen["anchors"] = dict(anchors)
        return SplineBackend().fill(anchors, n_frames, None)


def test_the_detect_tab_offers_every_detector(dialog):
    """One today — the combo is the seam a second arrives through."""
    keys = [dialog.detector_combo.itemData(i) for i in range(dialog.detector_combo.count())]
    assert keys == [dialog_module.APRILTAG_DETECTOR]


def test_detector_parameter_rows_follow_the_detector(dialog):
    """isHidden(), not isVisibleTo(): the Detect tab is not the shown one."""
    dialog.detector_combo.setCurrentIndex(dialog.detector_combo.findData(dialog_module.APRILTAG_DETECTOR))
    assert dialog.tag_row.isHidden() is False


def test_the_family_combo_offers_only_detectable_families(dialog):
    """A combo, but a closed one.

    Some pupil-apriltags families fail to allocate their decode table and take
    the whole process with them, and ``tag36h10`` renders in OpenCV but cannot
    be detected at all — so the list, not the user, is the guard.
    """
    from ethograph.gui.pose_detect import TAG_FAMILIES

    offered = [dialog.tag_family_combo.itemData(i) for i in range(dialog.tag_family_combo.count())]
    assert offered == list(TAG_FAMILIES)
    assert "tag36h10" not in offered
    assert dialog._detector_params().keys() == {"family", "quad_decimate", "decode_sharpening", "parts"}


def test_choosing_a_family_rebuilds_the_detector_and_is_remembered(dialog):
    first = dialog._current_detector()
    dialog.tag_family_combo.setCurrentText("tag16h5")

    assert dialog.app_state.detect_tag_family == "tag16h5"
    rebuilt = dialog._current_detector()
    assert rebuilt is not first
    assert rebuilt.family == "tag16h5"
    # A smaller grid is the whole point: less paper for the same pixels/module.
    assert rebuilt.min_side_px < first.min_side_px


def test_detection_reads_frames_at_full_resolution(dialog, monkeypatch):
    """The one thing a tag decoder cannot do without is pixels.

    Every other stage downscales to MAX_SIDE; at 512 px an 8 mm tag filmed at
    1920 px is ~12 px across, which the quad finder does not even propose.
    """
    asked: list = []

    def _record(max_side=dialog_module.MAX_SIDE):
        asked.append(max_side)
        raise ValueError("no video in this test")

    monkeypatch.setattr(dialog, "_open_frames", _record)
    # The preview only opens a source once there is a video to open.
    monkeypatch.setattr(dialog, "_video_path", lambda: "clip.mp4")
    monkeypatch.setattr(dialog, "_fps", lambda: 30.0)
    assert dialog._preview_frame_source() is None
    for runner in (dialog._run_learn_assignment, dialog._run_detection):
        with pytest.raises(ValueError):
            runner(lambda _f: True)

    assert dialog_module.DETECT_MAX_SIDE is None
    assert asked == [None, None, None], "every detect path decodes full-size"


def test_the_detect_tab_does_not_offer_to_print_tags(dialog):
    """By the time there is a video the tags are already on the animals.

    Printing lives on the cover page's pre-recording tools, which is the only
    screen that exists before a recording does.
    """
    assert not hasattr(dialog, "_on_print_tag_sheet")


def test_a_detected_row_reads_as_detected_not_fill(dialog):
    _detections(dialog.store, 4, "beak")
    dialog._refresh_point_table(full=True)

    rows = {row[0]: row for row in _table_rows(dialog)}
    assert rows["4"][SOURCE_COLUMN] == dialog_module.DETECTED_SOURCE
    # The detector's own quality, not the fill's.
    assert rows["4"][_column_of(dialog, "beak", "conf")] == "0.90"


def test_correcting_a_detection_makes_the_row_human(dialog):
    _detections(dialog.store, 4, "beak")
    dialog.store.set_point(4, "beak", (99.0, 99.0))
    dialog._refresh_point_table(full=True)

    rows = {row[0]: row for row in _table_rows(dialog)}
    assert rows["4"][SOURCE_COLUMN] == dialog_module.HUMAN_SOURCE
    assert rows["4"][_column_of(dialog, "beak", "x")] == "99.0"


def _confirm(monkeypatch, answer=True):
    """Answer the bulk-approval confirmation, which is not suppressed by default."""
    from qtpy.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.Yes if answer else QMessageBox.No),
    )


def test_the_bulk_approvals_are_offered_only_when_there_is_something_to_take(dialog):
    """Both start dead: nothing has been detected and nothing has been filled."""
    assert dialog.approve_detections_btn.isEnabled() is False
    assert dialog.approve_fill_btn.isEnabled() is False

    _detections(dialog.store, 4, "beak")
    dialog._after_detections_changed()
    assert dialog.approve_detections_btn.isEnabled() is True
    assert dialog.approve_fill_btn.isEnabled() is False, "a detection is not a fill"

    _fill(dialog.store)
    dialog._on_store_changed(full=True)
    assert dialog.approve_fill_btn.isEnabled() is True


def test_approving_all_detections_makes_them_human(dialog, monkeypatch):
    _confirm(monkeypatch)
    for frame in (2, 4):
        _detections(dialog.store, frame, "beak")

    dialog._on_approve_all_detections()

    assert dialog.store.anchor_frames() == [2, 4]
    assert dialog.store.is_human(2) and dialog.store.is_human(4)
    assert not dialog.store.is_detected(2), "it reads as a label now, not a detection"
    rows = {row[0]: row for row in _table_rows(dialog)}
    assert rows["2"][SOURCE_COLUMN] == dialog_module.HUMAN_SOURCE


def test_approving_all_filled_points_makes_them_human(dialog, monkeypatch):
    _confirm(monkeypatch)
    _fill(dialog.store)
    span = dialog.store.fill_range

    dialog._on_approve_all_fill()

    assert dialog.store.anchor_frames() == list(range(span[0], span[1] + 1))
    assert not dialog.store.has_predictions(span[0] + 1)


def test_declining_the_confirmation_changes_nothing(dialog, monkeypatch):
    """It cannot be undone, so the question has to be a real gate."""
    _confirm(monkeypatch, answer=False)
    _detections(dialog.store, 4, "beak")

    dialog._on_approve_all_detections()

    assert dialog.store.anchor_frames() == []
    assert dialog.store.is_detected(4)


def test_a_bulk_approval_never_overwrites_a_label(dialog, monkeypatch):
    _confirm(monkeypatch)
    _detections(dialog.store, 4, "beak", xy=(11.0, 12.0))
    dialog.store.set_point(4, "beak", (99.0, 99.0))

    dialog._on_approve_all_detections()

    np.testing.assert_allclose(dialog.store.anchor_positions(4)[0, 0], [99.0, 99.0])


def test_the_source_filter_offers_all_three_provenances(dialog):
    assert dialog._filter_values(SOURCE_COLUMN) == [
        dialog_module.HUMAN_SOURCE,
        dialog_module.DETECTED_SOURCE,
        dialog_module.FILL_SOURCE,
    ]


def test_a_detected_frame_gets_a_table_row_of_its_own(dialog):
    dialog.store.set_point(1, "beak", (1.0, 2.0))
    _detections(dialog.store, 6, "tail")
    dialog._refresh_point_table(full=True)

    assert [row[0] for row in _table_rows(dialog)] == ["1", "6"]


def test_the_table_layout_notices_a_detector_run(dialog):
    dialog.store.set_point(1, "beak", (1.0, 2.0))
    dialog._refresh_point_table(full=True)
    before = dialog._layout_signature()

    _detections(dialog.store, 6, "tail")
    assert dialog._layout_signature() != before


def test_filling_bridges_the_detections(dialog, monkeypatch):
    """The one line of coupling: a fill is built from the observations."""
    _detections(dialog.store, 0, "beak", (0.0, 0.0))
    _detections(dialog.store, 8, "beak", (80.0, 0.0))
    seen = {}
    monkeypatch.setattr(dialog_module, "build_backend", lambda *a, **k: _RecordingBackend(seen))
    dialog.backend_combo.setCurrentIndex(dialog.backend_combo.findData("spline"))

    dialog._on_fill()

    assert sorted(seen["anchors"]) == [0, 8]
    assert dialog.store.fill_range == (0, 8)


def test_approving_a_frame_accepts_a_detection_with_no_fill(dialog):
    _detections(dialog.store, 4, "beak")
    dialog.app_state.current_frame = 4

    dialog._approve_frame()

    assert dialog.store.is_anchor(4, "beak")


def test_rejecting_a_detection_keeps_the_labels(dialog):
    _detections(dialog.store, 4, "beak")
    dialog.store.set_point(4, "tail", (1.0, 1.0))

    dialog._clear_detection_rows([(4, "individual_0")])

    assert not dialog.store.detections
    assert dialog.store.is_anchor(4, "tail")


def test_pinning_a_detection_makes_it_a_label(dialog):
    _detections(dialog.store, 4, "beak", (7.0, 8.0))

    dialog._pin_detection_rows([(4, "individual_0")])

    assert dialog.store.is_anchor(4, "beak")
    np.testing.assert_allclose(dialog.store.anchor_positions(4)[0, 0], [7.0, 8.0])


def test_the_assignment_table_lists_every_label(dialog):
    dialog.store.assignment.set(3, None, "beak")
    dialog.store.assignment.set(1, None, "tail")
    dialog._refresh_assignment_table()

    labels = [
        dialog.assignment_table.item(row, 0).data(Qt.UserRole) for row in range(dialog.assignment_table.rowCount())
    ]
    assert labels == [1, 3]


def test_editing_an_assignment_makes_it_manual(dialog):
    """And names the individual: the table never offers "the first one"."""
    dialog.store.assignment.set(1, None, "beak")
    dialog._refresh_assignment_table()

    combo = dialog.assignment_table.cellWidget(0, 2)
    combo.setCurrentIndex(combo.findData("eye"))

    assert dialog.store.assignment.target(1) == ("individual_0", "eye")
    assert dialog.store.assignment.get(1).source == dialog_module.MANUAL


def test_set_by_says_who_chose_the_row_not_where_a_coordinate_came_from(dialog):
    """Deliberately NOT called Source — the points table owns that word."""
    dialog.store.assignment.set(1, None, "beak", dialog_module.LEARNED)
    dialog.store.assignment.set(2, None, "tail", dialog_module.MANUAL)
    dialog._refresh_assignment_table()

    headers = [
        dialog.assignment_table.horizontalHeaderItem(c).text() for c in range(dialog.assignment_table.columnCount())
    ]
    assert headers == ["Label", "Individual", "Keypoint", "Matched on", "Set by"]
    assert [dialog.assignment_table.item(row, 4).text() for row in (0, 1)] == ["learning", "you"]


def test_the_individual_picker_never_offers_two_spellings_of_one_point(dialog):
    """`None` and the first individual's name are one row, so only one is offered."""
    dialog.store.assignment.set(1, None, "beak")
    dialog._refresh_assignment_table()

    combo = dialog.assignment_table.cellWidget(0, 1)
    assert [combo.itemData(i) for i in range(combo.count())] == ["individual_0"]


def test_two_labels_resolving_to_one_point_are_flagged(dialog):
    """`None` and `individual_0` are the same row — `owner_of` cannot see that."""
    dialog.store.assignment.set(1, None, "beak")
    dialog.store.assignment.set(2, "individual_0", "beak")

    assert dialog.store.assignment.invalid_labels(dialog.store) == {2}
    # And only the lower label is written, so no point comes from two labels.
    assert dialog.store.assignment_rows() == {1: dialog.store.keypoint_index("beak")}


def test_an_invalid_assignment_is_flagged_not_dropped(dialog):
    dialog.store.assignment.set(1, None, "beak")
    dialog._apply_schema(keypoints=["tail", "eye"])

    assert dialog.store.assignment.target(1) == (None, "beak")
    assert "no longer exists" in dialog.assignment_warning.text()


def test_running_needs_an_assignment(dialog, monkeypatch):
    warned = []
    monkeypatch.setattr(dialog_module, "notify", lambda message, level="info": warned.append((level, message)))

    dialog._on_run_detector()

    assert warned and warned[0][0] == "warning"
    assert not dialog.store.detections


def test_the_quality_threshold_retunes_without_re_running(dialog):
    n_points = dialog.store.n_points
    positions = {4: np.full((n_points, 2), np.nan), 5: np.full((n_points, 2), np.nan)}
    quality = {4: np.full(n_points, np.nan), 5: np.full(n_points, np.nan)}
    positions[4][0], quality[4][0] = (1.0, 2.0), 0.9
    positions[5][0], quality[5][0] = (3.0, 4.0), 0.2
    dialog._raw_detections = (positions, quality, {})

    dialog.detect_quality_spin.setValue(0.5)
    assert dialog.store.detection_frames() == [4]

    dialog.detect_quality_spin.setValue(0.1)
    assert dialog.store.detection_frames() == [4, 5]


def test_the_detection_cache_round_trips(dialog, tmp_path, monkeypatch):
    monkeypatch.setattr(dialog, "_video_path", lambda: str(tmp_path / "clip.mp4"))
    _detections(dialog.store, 4, "beak")
    dialog.store.assignment.set(0, None, "beak")

    dialog._save_detections()
    dialog.store.clear_detections()
    dialog._load_detections()

    assert dialog.store.detection_frames() == [4]


def test_a_different_threshold_is_a_cache_miss(dialog, tmp_path, monkeypatch):
    """The cache holds what was kept, so a looser threshold has to re-run."""
    monkeypatch.setattr(dialog, "_video_path", lambda: str(tmp_path / "clip.mp4"))
    _detections(dialog.store, 4, "beak")
    dialog._save_detections()

    dialog.store.clear_detections()
    dialog.detect_quality_spin.setValue(dialog.detect_quality_spin.value() + 0.2)
    dialog._load_detections()

    assert dialog.store.detection_frames() == []


def test_detection_gaps_needs_a_run(dialog, monkeypatch):
    warned = []
    monkeypatch.setattr(dialog_module, "notify", lambda message, level="info": warned.append((level, message)))
    dialog.suggest_method_combo.setCurrentIndex(dialog.suggest_method_combo.findData("detection_gaps"))

    dialog._on_suggest()

    assert warned and "detector" in warned[0][1]


def test_detection_gaps_suggests_the_blind_frames(dialog, monkeypatch):
    landed = []
    monkeypatch.setattr(dialog, "_seek", lambda frame: landed.append(int(frame)))
    for frame in (0, 1, 8, 9):
        _detections(dialog.store, frame, "beak")
    dialog.suggest_method_combo.setCurrentIndex(dialog.suggest_method_combo.findData("detection_gaps"))
    dialog.suggest_percent_spin.setValue(10.0)

    dialog._on_suggest()

    assert landed and 2 <= landed[0] <= 7


def test_the_legend_names_the_detected_style_only_after_a_run(dialog):
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    _fill(dialog.store)
    dialog._refresh_legend()
    assert "detected" not in dialog.legend_label.text()

    _detections(dialog.store, 4, "beak")
    dialog._refresh_legend()
    assert "detected" in dialog.legend_label.text()


# ----------------------------------------------------------------------
# Detector preview
# ----------------------------------------------------------------------


def _show_detect_tab(dialog) -> None:
    dialog.tabs.setCurrentWidget(dialog._detect_page)


def test_the_preview_costs_nothing_while_its_tab_is_hidden(dialog, monkeypatch):
    """Every redraw decodes a frame; the Label tab must not pay for it."""
    drawn = []
    monkeypatch.setattr(dialog, "_refresh_preview", lambda: drawn.append(1))

    dialog.tabs.setCurrentWidget(dialog._label_page)
    dialog._schedule_preview()
    assert dialog._preview_timer.isActive() is False

    _show_detect_tab(dialog)
    dialog._schedule_preview()
    assert dialog._preview_timer.isActive() is True


def test_unticking_show_preview_stops_it(dialog):
    _show_detect_tab(dialog)
    dialog.preview_check.setChecked(False)
    dialog._schedule_preview()

    assert dialog._preview_timer.isActive() is False


def test_scrubbing_coalesces_into_one_redraw(dialog):
    """A drag emits a frame change per tick — the timer restarts, not stacks."""
    _show_detect_tab(dialog)
    for frame in range(5):
        dialog._on_frame_changed(frame)

    assert dialog._preview_timer.isActive() is True
    assert dialog._preview_timer.interval() == dialog_module.PREVIEW_DEBOUNCE_MS


def test_the_preview_says_when_there_is_no_video(dialog, monkeypatch):
    monkeypatch.setattr(dialog, "_video_path", lambda: None)
    _show_detect_tab(dialog)

    dialog._refresh_preview()

    assert "Load a video" in dialog.mask_preview.text()


def test_the_preview_names_both_ways_of_finding_nothing(dialog, monkeypatch):
    """The whole point of the panel: a misread tag is not the same as no tag."""
    from ethograph.gui.pose_detect import DetectionPreview, PreviewShape

    monkeypatch.setattr(dialog, "_preview_frame_source", lambda: _StubFrames())
    monkeypatch.setattr(dialog, "_current_detector", lambda progress=None: _StubDetector())
    preview = DetectionPreview(
        shapes=[
            PreviewShape(xy=np.array([2.0, 2.0]), label=0, accepted=True),
            PreviewShape(xy=np.array([6.0, 6.0]), label=None, accepted=False, reason="2 bit error(s) — not trusted"),
        ],
        size=(16, 16),
    )
    monkeypatch.setattr(dialog_module, "diagnose_frame", lambda _d, _f: preview)
    dialog.store.assignment.set(0, None, "beak")
    _show_detect_tab(dialog)

    dialog._refresh_preview()

    text = dialog.preview_summary.text()
    assert "1 tag(s) decoded" in text
    assert "1 rejected" in text


def test_the_preview_says_when_nothing_decoded_at_all(dialog, monkeypatch):
    from ethograph.gui.pose_detect import DetectionPreview

    monkeypatch.setattr(dialog, "_preview_frame_source", lambda: _StubFrames())
    monkeypatch.setattr(dialog, "_current_detector", lambda progress=None: _StubDetector())
    monkeypatch.setattr(dialog_module, "diagnose_frame", lambda _d, _f: DetectionPreview(size=(16, 16)))
    _show_detect_tab(dialog)

    dialog._refresh_preview()

    assert "no tag decoded" in dialog.preview_summary.text()


def test_the_detector_is_rebuilt_only_when_its_settings_change(dialog):
    """The preview asks for it on every redraw."""
    first = dialog._current_detector()
    assert dialog._current_detector() is first

    dialog.tag_corners_check.setChecked(not dialog.tag_corners_check.isChecked())
    assert dialog._current_detector() is not first


class _StubFrames:
    """A frame source with the indexing contract, holding one tiny frame."""

    scale = 1.0

    def __len__(self):
        return 10

    def __getitem__(self, key):
        if isinstance(key, slice):
            return np.zeros((1, 16, 16, 3), dtype=np.uint8)
        return np.zeros((16, 16, 3), dtype=np.uint8)

    def close(self):
        pass


class _StubDetector:
    name = "stub"

    def detect(self, frame):
        return []


# ----------------------------------------------------------------------
# Head direction
# ----------------------------------------------------------------------


def _detect_an_oriented_marker(dialog, frames=(0, 1, 2)) -> None:
    """A detector run that found ONE tag on `beak`, facing up the frame.

    One tagged keypoint is the whole input: nothing here creates a second
    keypoint, and nothing nominates a left/right pair.
    """
    n_ind, n_kp = dialog.store.n_individuals, dialog.store.n_keypoints
    beak = dialog.store.keypoint_index("beak")
    positions, orientation = {}, {}
    for frame in frames:
        points = np.full((n_ind, n_kp, 2), np.nan)
        points[0, beak] = (100.0, 200.0)
        vectors = np.full((n_ind, n_kp, 2), np.nan)
        vectors[0, beak] = (0.0, -1.0)
        positions[frame], orientation[frame] = points, vectors
    dialog.store.set_detections(positions, orientation=orientation)
    dialog._refresh_head_direction_row()


def test_head_direction_is_unavailable_without_an_oriented_marker(dialog):
    """A bare keypoint cannot face anywhere, so there is nothing to offer."""
    assert not dialog.head_direction_check.isEnabled()
    assert not dialog.head_direction_check.isChecked()
    assert dialog._head_direction_wanted() is False


def test_detecting_a_tag_turns_head_direction_on(dialog):
    _detect_an_oriented_marker(dialog)
    assert dialog.head_direction_check.isEnabled()
    assert dialog.head_direction_check.isChecked()


def test_there_is_nothing_to_pick(dialog):
    """The regression this design exists for: one tag is one keypoint, so the
    dialog must not ask which two keypoints face left and right."""
    _detect_an_oriented_marker(dialog)
    assert not hasattr(dialog, "head_left_combo")
    assert not hasattr(dialog, "head_right_combo")


def test_clearing_the_detections_withdraws_head_direction(dialog):
    _detect_an_oriented_marker(dialog)
    dialog.store.clear_detections()
    dialog._refresh_head_direction_row()

    assert not dialog.head_direction_check.isEnabled()
    assert not dialog.head_direction_check.isChecked()


def test_unticking_head_direction_is_respected(dialog):
    """Offered once when a run first produces it, never re-ticked behind you."""
    _detect_an_oriented_marker(dialog)
    dialog.head_direction_check.setChecked(False)

    dialog._refresh_head_direction_row()

    assert not dialog.head_direction_check.isChecked()
    assert dialog._head_direction_wanted() is False


def test_head_direction_reaches_the_dataset(dialog):
    pytest.importorskip("movement")
    _detect_an_oriented_marker(dialog)

    ds = dialog._build_dataset()

    assert {"head_direction", "heading"} <= set(ds.data_vars)
    vector = ds["head_direction"].isel(time=0, individual=0).sel(keypoint="beak")
    # The store keeps image coordinates; the export flips y, and the heading
    # has to flip with it or it points opposite to the trajectory.
    np.testing.assert_allclose(vector.values, [0.0, 1.0])


def test_load_into_the_gui_hands_over_one_dataset(dialog, monkeypatch, tmp_path):
    """The GUI is handed the dataset itself; the file beside the video is a copy."""
    pytest.importorskip("movement")
    video = tmp_path / "clip.mp4"
    monkeypatch.setattr(dialog, "_video_path", lambda: str(video))
    _detect_an_oriented_marker(dialog)

    handed: list = []
    monkeypatch.setattr(
        dialog._data_widget,
        "load_keypoint_dataset",
        lambda ds: handed.append(ds) or True,
        raising=False,
    )
    dialog._on_load_into_gui()

    assert len(handed) == 1
    assert {"position", "confidence", "head_direction", "heading"} <= set(handed[0].data_vars)
    assert list(handed[0].coords["keypoint"].values) == NAMES

    # ...and the same thing landed on disk beside the video.
    with xr.open_dataset(tmp_path / "clip.mp4.keypoints.nc") as saved:
        assert saved.attrs["ds_type"] == "poses"
        assert set(saved.data_vars) == set(handed[0].data_vars)


def test_a_read_only_folder_still_loads(dialog, monkeypatch, tmp_path):
    """Saving the copy is not how the GUI is fed, so it cannot block the load."""
    pytest.importorskip("movement")
    monkeypatch.setattr(dialog, "_video_path", lambda: str(tmp_path / "clip.mp4"))
    _detect_an_oriented_marker(dialog)

    def _refuse(*_args, **_kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _refuse)
    handed: list = []
    monkeypatch.setattr(
        dialog._data_widget,
        "load_keypoint_dataset",
        lambda ds: handed.append(ds) or True,
        raising=False,
    )

    dialog._on_load_into_gui()

    assert len(handed) == 1


def test_labels_alone_reach_the_ordinary_pose_overlay(dialog):
    """Loading is also how hand-placed points become a normal pose source, so
    the Pose sidebar's keypoint filter and confidence threshold act on them."""
    dialog.set_interaction_mode(None)
    dialog.store.set_point(0, "beak", (10.0, 20.0))
    assert dialog.store.filled is None

    dialog._push_pose_override()

    override = dialog._data_widget.pose_mgr.override
    assert override is not None
    assert "beak" in override.keypoints
