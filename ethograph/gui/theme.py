"""Fixed application chrome theme for the standalone (napari-free) GUI.

napari used to supply the dark chrome; after the pygfx migration nothing did,
so this module re-applies the same grey-blue, near-white-text look napari's
dark theme had. It only styles the **Qt chrome** (menus, docks, sidebar,
buttons, inputs) via a ``QPalette`` on the cross-platform Fusion style.

Plots are deliberately left alone: the pyqtgraph panels stay white (set per
widget in ``BasePlot``), and the ephys neural-trace panel keeps its own
phy-style dark background. The pygfx video canvas paints its own dark
background.
"""

from __future__ import annotations

from qtpy.QtGui import QColor, QPalette

# napari "dark" theme colours (RGB) — the source of the grey-blue chrome.
_BACKGROUND = QColor(38, 41, 48)  # #262930 — sidebar / window
_FOREGROUND = QColor(65, 72, 81)  # #414851 — buttons, tooltips, alt rows
_BASE = QColor(33, 36, 43)  # slightly darker than window — text-entry fields
_TEXT = QColor(240, 241, 242)  # #F0F1F2 — near-white high-contrast text
_DISABLED = QColor(134, 142, 147)  # #868E93 — secondary / disabled text
_CURRENT = QColor(0, 122, 204)  # #007ACC — selection / links
_WARNING = QColor(226, 121, 121)  # #E27979 — bright/attention text


def _napari_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, _BACKGROUND)
    p.setColor(QPalette.ColorRole.WindowText, _TEXT)
    p.setColor(QPalette.ColorRole.Base, _BASE)
    p.setColor(QPalette.ColorRole.AlternateBase, _FOREGROUND)
    p.setColor(QPalette.ColorRole.ToolTipBase, _FOREGROUND)
    p.setColor(QPalette.ColorRole.ToolTipText, _TEXT)
    p.setColor(QPalette.ColorRole.Text, _TEXT)
    p.setColor(QPalette.ColorRole.Button, _FOREGROUND)
    p.setColor(QPalette.ColorRole.ButtonText, _TEXT)
    p.setColor(QPalette.ColorRole.BrightText, _WARNING)
    p.setColor(QPalette.ColorRole.Link, _CURRENT)
    p.setColor(QPalette.ColorRole.Highlight, _CURRENT)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.PlaceholderText, _DISABLED)

    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, _DISABLED)
    return p


def apply_theme(app) -> None:
    """Apply the fixed napari-style chrome to the whole application."""
    app.setStyle("Fusion")
    app.setPalette(_napari_palette())
