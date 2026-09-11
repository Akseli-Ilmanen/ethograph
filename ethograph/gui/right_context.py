"""Context-sensitive right sidebar body.

The "Data" section of the sidebar is *minimal*: instead of showing every
control at once, it shows only the controls relevant to the plot the user last
clicked.  The relevant section widgets (group boxes / panels) are borrowed from
``DataPanel``, ``PlotSettingsWidget`` and ``NavigationWidget`` and re-parented
into a single :class:`RightContextPanel`; :meth:`set_context` then shows the
subset mapped to each plot type.

Mapping (per the design brief):

============  ==================================================
plot type     sections shown
============  ==================================================
``video``     Crop (crop/uncrop the clicked camera), Label overlay (hide the
              label name drawn on the video), Overlay source (which feature
              is drawn), Pose (if pose data) — or Bounding boxes instead,
              when the drawn feature has a ``shape`` companion
``audio``     Energy envelope, Spectrogram settings, shared axes
``lineplot``  Xarray coords, Overlays, Line-plot axes, shared axes
``heatmap``   Xarray coords, Overlays, Heatmap, shared axes
``space``     Xarray coords, Space-plot, shared axes
``radial``    Radial-plot (feature + which value is up)
============  ==================================================

The **Individual** group sits above all of them, outside the mapping: which
animal (and, for dyadic behaviours, which receiver) is being shown and
labelled is a question every panel answers — the sole exception is the video,
whose overlays follow the pose settings instead.

The sidebar refreshes only when the clicked plot *type* changes (not on every
click), and updates are skipped entirely in zen mode or when the Labels /
Navigation section is active (handled by the caller).
"""

from __future__ import annotations

from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

# plot type -> ordered list of section keys to show
_CONTEXT_MAP: dict[str, list[str]] = {
    # The old napari-era "Space/Cameras" group (slot) is gone — cameras are
    # opened by drag-drop and there is no layers/space-plot toggle.
    "video": ["videocrop", "videolabel", "overlay", "pose", "bbox"],
    # Audio trace: channel + envelope controls + shared axes. Spectrogram:
    # channel + its panel.
    "audiotrace": ["audiochannel", "energy", "shared"],
    "audio": ["audiochannel", "energy", "shared"],  # alias used for the default context
    "spectrogram": ["audiochannel", "spectrogram"],
    "feature": ["coords", "lineplot", "shared"],
    "lineplot": ["coords", "lineplot", "shared"],
    "heatmap": ["coords", "heatmap", "shared"],
    # Space: its own X/Y/Z + 3D + space controls (now inside spaceplot_panel);
    # the lineplot "coords" group is intentionally excluded.
    "space": ["spaceplot", "shared"],
    # Radial (compass): its feature + which value points up. No shared axes —
    # it has no time axis to autoscale or lock.
    "radial": ["radialplot"],
    # Phy-like ephys trace: the full Kilosort trace controls (channel/gain/
    # pyramid/probe select + cluster table), borrowed from EphysWidget. No
    # shared axes group — autoscale/lock-axes don't apply to the trace view.
    "ephys": ["phy"],
    # Neo trace (generic per-modality stream): channels are chosen at drop time
    # via the source popup; the sidebar exposes per-panel gain + channel spacing.
    "neo": ["neocontrols"],
}

# plot type -> friendly caption shown (in the active-panel green) at the bottom
# of the sidebar, so it's clear which clicked plot the controls belong to.
_CONTEXT_TITLE: dict[str, str] = {
    "video": "Video playback settings",
    "audiotrace": "Audiotrace settings",
    "audio": "Audiotrace settings",
    "spectrogram": "Spectrogram settings",
    "feature": "Lineplot settings",
    "lineplot": "Lineplot settings",
    "heatmap": "Heatmap settings",
    "space": "Space plot settings",
    "radial": "Radial plot settings",
    "ephys": "Phy viewer settings",
    "neo": "Neo viewer settings",
}

#: The active-panel green edge colour (see ``ActivePanelManager._EDGE_ON``).
_ACTIVE_GREEN = "#2ecc71"

#: Contexts that do NOT get the Individual selector. The video's own
#: per-individual display is the pose overlay's business.
_NO_INDIVIDUAL_CONTEXTS = frozenset({"video"})


class RightContextPanel(QWidget):
    """Hosts all setting sections and shows only the clicked plot's subset."""

    def __init__(self, sections: dict[str, QWidget | None], parent=None):
        super().__init__(parent)
        self._sections = {k: v for k, v in sections.items() if v is not None}
        #: Shown above the caption for every context but the video's — it says
        #: *whose* data and labels the panel below is about, so it is not one
        #: of the per-plot-type sections.
        self._individual = self._sections.pop("individual", None)
        self._current: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        if self._individual is not None:
            layout.addWidget(self._individual)
            self._individual.setVisible(False)

        # Top caption naming the active plot type, coloured to match the
        # panel's green selection edge so the link is obvious to the user.
        self._title = QLabel("")
        self._title.setWordWrap(True)
        self._title.setStyleSheet(f"color: {_ACTIVE_GREEN}; font-weight: bold; font-size: 14px; padding: 4px 2px;")
        self._title.setVisible(False)
        layout.addWidget(self._title)

        self._placeholder = QLabel("Click a plot or the video to show its settings.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet("color: rgba(255,255,255,140);")
        layout.addWidget(self._placeholder)

        for widget in self._sections.values():
            # Re-parent each borrowed section in and start hidden.
            layout.addWidget(widget)
            widget.setVisible(False)
        layout.addStretch()

    def set_context(self, plot_type: str, has_pose: bool = True, pose_kind: str | None = None) -> bool:
        """Show only the sections mapped to *plot_type*.

        *pose_kind* says what the pose overlay draws: ``"bboxes"`` swaps the
        Pose section for the Bounding boxes one (no skeleton, a colour per
        individual); anything else keeps the Pose section.

        Returns ``True`` if the context changed (so the caller can refresh the
        surrounding layout), ``False`` if it was already showing *plot_type*.
        """
        key = (plot_type, has_pose, pose_kind == "bboxes")
        if key == self._current:
            return False
        self._current = key
        want = set(_CONTEXT_MAP.get(plot_type, []))
        want.discard("bbox" if pose_kind != "bboxes" else "pose")
        if not has_pose:
            want.discard("overlay")
            want.discard("pose")
            want.discard("bbox")
        self._placeholder.setVisible(not want)
        if self._individual is not None:
            self._individual.setVisible(bool(want) and plot_type not in _NO_INDIVIDUAL_CONTEXTS)
        for name, widget in self._sections.items():
            widget.setVisible(name in want)
        title = _CONTEXT_TITLE.get(plot_type, "")
        self._title.setText(title)
        self._title.setVisible(bool(title and want))
        return True

    def current_context(self) -> str | None:
        return self._current[0] if self._current is not None else None
