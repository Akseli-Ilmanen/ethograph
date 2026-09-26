"""General active-panel management.

One system for EVERY panel type (video, audio trace, spectrogram, line plot,
heatmap, space plot, ephys, …): it tracks the last-clicked panel, draws a green
edge around it, and emits :pyattr:`ActivePanelManager.active_changed` so the
right sidebar can show the controls specific to that panel's ``kind``.

A panel is just a ``QWidget`` plus a ``kind`` string and a click signal. Register
each panel once; the manager does the rest. This keeps "what is the active panel"
and "highlight it" in a single place instead of being re-implemented per widget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QWidget


class PanelKind:
    VIDEO = "video"
    AUDIOTRACE = "audiotrace"
    SPECTROGRAM = "spectrogram"
    LINEPLOT = "lineplot"
    HEATMAP = "heatmap"
    SPACE = "space"
    RADIAL = "radial"
    SKELETON = "skeleton"
    EPHYS = "ephys"
    NEO = "neo"
    RASTER = "raster"

    #: Kinds that are feature (line/heatmap) plots and carry a per-panel state.
    FEATURE = frozenset({LINEPLOT, HEATMAP})


@dataclass
class PanelRegistration:
    """One registered panel."""

    widget: QWidget
    kind: str
    plot: Any | None = None  # the plot object (for feature plots)
    meta: dict = field(default_factory=dict)


class ActivePanelManager(QObject):
    """Tracks the active panel, highlights it, and announces changes."""

    #: Emitted with the newly-active :class:`PanelRegistration`.
    active_changed = Signal(object)

    _EDGE_ON = "border: 2px solid #2ecc71;"
    _EDGE_OFF = "border: 2px solid transparent;"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regs: list[PanelRegistration] = []
        self._active: PanelRegistration | None = None
        self._counter = 0

    # ------------------------------------------------------------------

    def register(self, widget: QWidget, kind: str, *, clicked_signal=None, plot=None) -> PanelRegistration:
        """Register *widget* as a panel of *kind*. ``clicked_signal`` (any Qt
        signal the panel emits on user click) makes it selectable."""
        existing = self.registration_for(widget)
        if existing is not None:
            return existing
        reg = PanelRegistration(widget=widget, kind=kind, plot=plot)
        self._regs.append(reg)
        self._ensure_object_name(widget)
        self._set_edge(widget, on=False)
        if clicked_signal is not None:
            clicked_signal.connect(lambda *_a, r=reg: self.set_active(r))
        return reg

    def unregister(self, widget: QWidget) -> None:
        reg = self.registration_for(widget)
        if reg is None:
            return
        self._regs.remove(reg)
        if self._active is reg:
            self._active = None

    def registration_for(self, widget: QWidget) -> PanelRegistration | None:
        for reg in self._regs:
            if reg.widget is widget:
                return reg
        return None

    @property
    def active(self) -> PanelRegistration | None:
        return self._active

    def set_active(self, reg: PanelRegistration) -> None:
        """Make *reg* active and announce it.

        ``active_changed`` is emitted even when *reg* is already active: the
        sidebar context can go stale while the panel stays active (zen mode,
        Labels/Navigation section open suppress the context swap), so a click
        must always re-announce — never be a silent no-op.
        """
        if reg is not self._active:
            if self._active is not None:
                self._set_edge(self._active.widget, on=False)
            self._active = reg
            self._set_edge(reg.widget, on=True)
        self.active_changed.emit(reg)

    # ------------------------------------------------------------------

    def _ensure_object_name(self, widget: QWidget) -> None:
        if not widget.objectName():
            self._counter += 1
            widget.setObjectName(f"ethopanel_{self._counter}")

    def _set_edge(self, widget: QWidget, on: bool) -> None:
        try:
            name = widget.objectName()
            style = self._EDGE_ON if on else self._EDGE_OFF
            # Scope to the widget itself so child widgets don't inherit the border.
            widget.setStyleSheet(f"#{name} {{ {style} }}")
        except (RuntimeError, AttributeError):
            pass
