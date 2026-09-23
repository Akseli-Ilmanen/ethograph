"""3x3 grid tab bar replacing the collapsible accordion headers.

GridSectionContainer is a drop-in replacement for CollapsibleWidgetContainer.
Instead of 9 stacked clickable header rows it shows a compact 3-column grid of
buttons at the top; clicking one expands that section's content below (one at a
time, same accordion semantics as before).
"""

from __future__ import annotations

from qtpy.QtCore import QObject, QSize, Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

_GRID_COLS = 3

# Short labels shown on the buttons; order must match add_widget call order in
# widgets_meta.py.
_SHORT_LABELS = ["Data", "Labels", "Trials"]

_STATUS_CHARS = set("✅❌🎯⭕")

_BTN_STYLE = """
QPushButton {{
    padding: 3px 2px;
    font-size: 7pt;
    border: 1px solid rgba(255,255,255,35);
    border-radius: 3px;
    background: rgba(255,255,255,12);
    color: #cccccc;
    min-height: 22px;
}}
QPushButton:hover {{
    background: rgba(255,255,255,25);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,60);
}}
QPushButton:checked {{
    background: rgba(65,125,195,160);
    border: 1px solid rgba(90,150,220,200);
    color: #ffffff;
    font-weight: bold;
}}
"""


class _AdaptiveStack(QStackedWidget):
    """QStackedWidget that reports only the *current* page's size hint.

    The default QStackedWidget always sizes itself to the largest of all its
    pages.  That causes the content area to be over-tall when a short section
    is active, which makes QVBoxLayout distribute the spare pixels as blank
    gaps between the section's own widgets.  Overriding sizeHint / minimumSizeHint
    so they reflect the current page fixes this entirely.
    """

    def sizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class SectionProxy(QObject):
    """Duck-type stand-in for qt_niu.CollapsibleWidget.

    Exposes the subset of the QCollapsible API used by MetaWidget so that
    widgets_meta.py needs no structural changes.
    """

    toggled = Signal(bool)

    def __init__(self, container: GridSectionContainer, index: int) -> None:
        super().__init__()
        self._c = container
        self._i = index

    # --- QCollapsible-compatible API ---

    def expand(self, animate: bool = True) -> None:
        self._c._expand(self._i)

    def collapse(self, animate: bool = False) -> None:
        self._c._collapse(self._i)

    def isExpanded(self) -> bool:
        return self._c._active == self._i

    def content(self) -> QWidget | None:
        widgets = self._c._content_widgets
        return widgets[self._i] if self._i < len(widgets) else None

    def setText(self, full_title: str) -> None:
        self._c._update_button_label(self._i, full_title)

    def setTitle(self, full_title: str) -> None:
        self.setText(full_title)

    def _expand_collapse(self, *_args, **_kwargs) -> None:
        """No-op: _AdaptiveStack + updateGeometry handles sizing."""

    def _emit_toggled(self, state: bool) -> None:
        self.toggled.emit(state)


class GridSectionContainer(QWidget):
    """Container with a 3-column grid of tab buttons + a single content area.

    API mirrors CollapsibleWidgetContainer so MetaWidget.add_widget() and
    MetaWidget.collapsible_widgets work unchanged.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        outer.setAlignment(Qt.AlignTop)

        # ── 3×3 grid of buttons ──────────────────────────────────────────────
        # Wrapped in its own panel with a distinct background so the navigation
        # grid reads as a separate "tab bar" from the section content below it.
        self._grid_widget = QWidget()
        self._grid_widget.setObjectName("sectionGridPanel")
        self._grid_widget.setStyleSheet(
            "#sectionGridPanel {"
            " background: rgba(255,255,255,8);"
            " border: 1px solid rgba(255,255,255,30);"
            " border-radius: 4px;"
            "}"
        )
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        self._grid_layout.setSpacing(2)
        for col in range(_GRID_COLS):
            self._grid_layout.setColumnStretch(col, 1)
        outer.addWidget(self._grid_widget)
        outer.addSpacing(10)

        # ── Separator between the grid tab bar and the content below ─────────
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Plain)
        separator.setFixedHeight(1)
        separator.setStyleSheet("background: rgba(255,255,255,45); border: none;")
        outer.addWidget(separator)
        outer.addSpacing(10)

        # ── Content area: shows the active section ───────────────────────────
        # _AdaptiveStack reports only the current page's sizeHint so the outer
        # layout never allocates more vertical space than the content needs.
        self._stack = _AdaptiveStack()
        self._stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._stack.hide()
        outer.addWidget(self._stack, 0, Qt.AlignTop)

        self._buttons: list[QPushButton] = []
        #: Section index -> button text a mode put in place of the default.
        self._label_overrides: dict[int, str | None] = {}
        self._content_widgets: list[QWidget] = []
        self._active: int | None = None

        # Public list of proxy objects (replaces CollapsibleWidgetContainer.collapsible_widgets)
        self.collapsible_widgets: list[SectionProxy] = []

    # ── Public API (mirrors CollapsibleWidgetContainer) ──────────────────────

    def add_widget(
        self,
        widget: QWidget,
        collapsible: bool = True,
        widget_title: str = "",
    ) -> None:
        index = len(self._content_widgets)
        label = _SHORT_LABELS[index] if index < len(_SHORT_LABELS) else widget_title

        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setToolTip(widget_title)
        btn.setStyleSheet(_BTN_STYLE)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.clicked.connect(lambda _, i=index: self._on_btn_clicked(i))

        row, col = divmod(index, _GRID_COLS)
        self._grid_layout.addWidget(btn, row, col)
        self._buttons.append(btn)

        self._stack.addWidget(widget)
        self._content_widgets.append(widget)

        proxy = SectionProxy(self, index)
        self.collapsible_widgets.append(proxy)

    def set_short_label(self, index: int, label: str | None) -> None:
        """Rename a section button (``None`` restores the default)."""
        if not (0 <= index < len(self._buttons)):
            return
        self._label_overrides[index] = label
        self._update_button_label(index, self._buttons[index].toolTip())

    def notify_content_resized(self) -> None:
        """Call this when the active section's content changes height.

        EphysWidget and PlotSettingsWidget call meta_widget.refresh_widget_layout()
        after toggling their sub-panels; that call routes here so the adaptive
        stack recalculates its size hint.
        """
        self._stack.updateGeometry()
        self.updateGeometry()

    # ── Internal expand / collapse ───────────────────────────────────────────

    def _on_btn_clicked(self, index: int) -> None:
        if self._active == index:
            self._collapse(index)
        else:
            self._expand(index)

    def _expand(self, index: int) -> None:
        if self._active is not None and self._active != index:
            prev = self._active
            self._buttons[prev].setChecked(False)
            self.collapsible_widgets[prev]._emit_toggled(False)
        self._active = index
        self._buttons[index].setChecked(True)
        self._stack.setCurrentIndex(index)
        self._stack.show()
        self._stack.updateGeometry()
        self.updateGeometry()
        self.collapsible_widgets[index]._emit_toggled(True)

    def _collapse(self, index: int) -> None:
        self._active = None
        self._buttons[index].setChecked(False)
        self._stack.hide()
        self._stack.updateGeometry()
        self.updateGeometry()
        self.collapsible_widgets[index]._emit_toggled(False)

    def _update_button_label(self, index: int, full_title: str) -> None:
        """Keep the short base label but append a trailing status character."""
        if not (0 <= index < len(self._buttons)):
            return
        base = self._label_overrides.get(index) or (_SHORT_LABELS[index] if index < len(_SHORT_LABELS) else "")
        label = f"{base} {full_title[-1]}" if full_title and full_title[-1] in _STATUS_CHARS else base
        self._buttons[index].setText(label)
        self._buttons[index].setToolTip(full_title)
