"""One colour per individual, shown wherever an individual is named.

The pose overlay and the skeleton panel colour each animal from
:func:`~ethograph.gui.pose_convert.individual_color_map` (the palette by the
dataset's order, the user's own picks winning). The same swatch goes next to the
name in the sidebar's Individual and Receiver combos and in every pin menu, so
"which one is mouse 2" is answered by looking, not by reading a legend.
"""

from __future__ import annotations

from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import QComboBox, QMenu

from ethograph.gui.pose_convert import individual_color_map

_SWATCH_PX = 14


def color_swatch_icon(color: str | tuple, size: int = _SWATCH_PX) -> QIcon:
    """A filled square: *color* is a hex string or an RGBA tuple of 0–1 floats."""
    if isinstance(color, str):
        qcolor = QColor(color)
    else:
        qcolor = QColor.fromRgbF(*(float(c) for c in color[:3]))
    pixmap = QPixmap(size, size)
    pixmap.fill(qcolor)
    return QIcon(pixmap)


def individual_swatches(app_state, names: list[str]) -> dict[str, QIcon]:
    """The swatch of every individual in *names* — the colour the overlay draws it in.

    *names* should be the dataset's full list (``app_state.label_individuals()``)
    so an animal keeps its colour whatever subset a widget offers.
    """
    overrides = getattr(app_state, "pose_individual_colors", None) or {}
    return {name: color_swatch_icon(rgba) for name, rgba in individual_color_map(names, overrides).items()}


def decorate_individual_combo(combo: QComboBox, app_state, names: list[str] | None = None) -> None:
    """Put each individual's swatch beside its entry; entries naming nobody keep none.

    Items are matched by their data (the raw name) first, then their text, so
    both the sidebar's display-label combos and plain ``addItems`` combos work.
    """
    if names is None:
        names = list(app_state.label_individuals())
    swatches = individual_swatches(app_state, [str(n) for n in names])
    for index in range(combo.count()):
        key = combo.itemData(index)
        icon = swatches.get(str(key)) if key not in (None, "") else None
        if icon is None:
            icon = swatches.get(combo.itemText(index))
        combo.setItemIcon(index, icon if icon is not None else QIcon())


def decorate_menu(menu: QMenu, app_state, names: list[str] | None = None) -> None:
    """Swatch every action of *menu* whose text is an individual's name."""
    if names is None:
        names = list(app_state.label_individuals())
    swatches = individual_swatches(app_state, [str(n) for n in names])
    for action in menu.actions():
        icon = swatches.get(action.text())
        if icon is not None:
            action.setIcon(icon)
