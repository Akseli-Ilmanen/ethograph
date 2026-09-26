"""Each individual's swatch beside its name: the overlay's colour, the user's pick winning."""

import pytest
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication, QComboBox, QMenu

from ethograph.gui.individual_colors import decorate_individual_combo, individual_swatches
from ethograph.gui.plots_container import add_pin_choices
from ethograph.gui.pose_convert import INDIVIDUAL_PALETTE


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _State:
    def __init__(self, names, overrides=None):
        self._names = names
        self.pose_individual_colors = overrides or {}

    def label_individuals(self):
        return list(self._names)


def _icon_hex(icon) -> str:
    return QColor(icon.pixmap(4, 4).toImage().pixel(1, 1)).name().upper()


def test_swatches_follow_the_palette_by_dataset_order_and_the_users_picks(qapp):
    swatches = individual_swatches(_State(["a", "b"], {"b": "#123456"}), ["a", "b"])
    assert _icon_hex(swatches["a"]) == INDIVIDUAL_PALETTE[0]
    assert _icon_hex(swatches["b"]) == "#123456"


def test_a_combo_gets_one_swatch_per_individual_and_none_for_other_entries(qapp):
    combo = QComboBox()
    combo.addItem("None", "")
    combo.addItem("A (display)", "a")  # the sidebar keeps the raw name in the data
    combo.addItem("b")  # a plain addItems combo matches by text
    decorate_individual_combo(combo, _State(["a", "b"]))
    assert combo.itemIcon(0).isNull()
    assert _icon_hex(combo.itemIcon(1)) == INDIVIDUAL_PALETTE[0]
    assert _icon_hex(combo.itemIcon(2)) == INDIVIDUAL_PALETTE[1]


def test_pin_choices_carry_the_swatches(qapp):
    menu = QMenu()
    add_pin_choices(menu, ["a", "b"], "b", "a", lambda _n: None, app_state=_State(["a", "b"]))
    named = [action for action in menu.actions() if action.text() in ("a", "b")]
    assert [_icon_hex(action.icon()) for action in named] == list(INDIVIDUAL_PALETTE[:2])
    assert [action.isChecked() for action in named] == [False, True]
    assert menu.actions()[0].icon().isNull(), "'Follow sidebar' names nobody"
