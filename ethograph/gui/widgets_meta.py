"""Widget container for other collapsible widgets."""

import logging
from pathlib import Path

from qtpy.QtCore import QLocale, Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.validation import IMAGE_FILE_FILTER
from ethograph.utils.paths import default_config_dir
from ethograph.utils.qt import (
    apply_compact_widget_style,
    normalize_child_layouts,
)

from .app_constants import (
    DEFAULT_LAYOUT_MARGIN,
    DEFAULT_LAYOUT_SPACING,
    LABELLING_MODE_FRAME,
    PLOT_CONTAINER_MIN_HEIGHT,
    SIDEBAR_DEFAULT_WIDTH_RATIO,
    SIDEBAR_MIN_WIDTH_PX,
)
from .app_state import ObservableAppState
from .file_dialogs import browse_open_file
from .grid_section_container import GridSectionContainer
from .make_pretty import LayoutManager
from .notify import notify
from .plots_container import UnifiedPanelContainer
from .shortcuts import bind_global_shortcuts
from .source_popup import IMAGE_BROWSE, ChannelSelectDialog, PlotTypePicker, SourcePopup, allowed_plot_types
from .widget_trials import TrialsWidget
from .widgets_changepoints import ChangepointsWidget
from .widgets_data import DataPanel, DataWidget
from .widgets_ephys import EphysWidget
from .widgets_help import HelpWidget
from .widgets_io import IOWidget
from .widgets_labels import LabelsWidget
from .widgets_navigation import NavigationWidget
from .widgets_plot_settings import PlotSettingsWidget

logger = logging.getLogger(__name__)


class MetaWidget(GridSectionContainer):
    def __init__(self, shell):
        """Initialize the meta-widget.

        Parameters
        ----------
        shell : EthographMainWindow
            The main application window hosting video, plots and this sidebar.
        """
        super().__init__()

        # Dot-decimal everywhere: widgets inherit their parent's locale, so
        # besides the default (for parentless widgets/dialogs) the shell's
        # whole tree must be forced to the C locale — the OS locale may use
        # "," as decimal separator.
        QLocale.setDefault(QLocale.c())
        if isinstance(shell, QWidget):
            shell.setLocale(QLocale.c())

        self.shell = shell
        #: Set when a load found no saved layout, cleared once the camera
        #: docks it tiles have been created (arrange_camera_grid_if_default).
        self._camera_grid_pending = False

        # Set smaller font for this widget and all children
        self._set_compact_font()

        # Create centralized app_state with YAML persistence
        global_settings = default_config_dir() / "gui_settings.yaml"
        logger.info("Settings file: %s", global_settings)

        self.app_state = ObservableAppState(yaml_path=str(global_settings))
        self.app_state._layout_snapshot_provider = self._snapshot_layouts

        # Try to load previous settings
        self.app_state.load_from_yaml()

        # Initialize all widgets with app_state
        self._create_widgets()

        self.collapsible_widgets[0].expand()  # Expand Data by default

        self._connect_collapsible_layout_refresh()

        self._bind_global_shortcuts(self.labels_widget, self.data_widget)

        # Set sidebar to 30% of the window by default (user can resize freely)
        self._set_sidebar_default_width()

    def _create_widgets(self):
        """Create all widgets with app_state passed to each one."""

        # Unified container replaces both PlotContainer and MultiPanelContainer
        self.plot_container = UnifiedPanelContainer(self.app_state)

        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_container.setMinimumHeight(PLOT_CONTAINER_MIN_HEIGHT)

        # Add-panel popup: drag Media/Feature sources onto the plot area (or
        # press Enter) to add panels. Opened via the bottom bar's "➕ Add
        # panel" button or Shift+N.
        # Parented to the shell so the top-level popup window is destroyed
        # with it (a parentless popup outlives QApplication → crash at exit).
        self.source_popup = SourcePopup(self.app_state, parent=self.shell)
        self.source_popup.on_activate = self._on_source_dropped
        self.plot_container._on_source_drop = self._on_source_dropped

        self.layout_mgr = LayoutManager(self.shell, self.plot_container)

        # Create all widgets with app_state
        self.help_widget = HelpWidget(self.app_state)
        self.plot_settings_widget = PlotSettingsWidget(self.shell, self.app_state)
        self.changepoints_widget = ChangepointsWidget(self.shell, self.app_state)
        self.labels_widget = LabelsWidget(self.shell, self.app_state)
        self.navigation_widget = NavigationWidget(self.shell, self.app_state)
        self.trials_widget = TrialsWidget(self.app_state)
        self.ephys_widget = EphysWidget(self.shell, self.app_state)

        # Create I/O widget first, then pass it to data widget
        self.io_widget = IOWidget(self.app_state, None, self.labels_widget)
        self.data_panel = DataPanel(self.app_state)
        self.data_widget = DataWidget(self.shell, self.app_state, self, self.io_widget)
        self.data_widget.set_data_panel(self.data_panel)
        # The container needs the data widget's catalog for the canonical
        # feature list (add_lineplot / heatmap drops).
        self.plot_container._data_widget = self.data_widget

        # Now set the data_widget reference in io_widget
        self.io_widget.data_widget = self.data_widget
        self.io_widget.changepoints_widget = self.changepoints_widget
        self.io_widget.meta_widget = self

        # Set up cross-references between widgets
        self.labels_widget.set_plot_container(self.plot_container)
        self.labels_widget.set_meta_widget(self)
        self.labels_widget.set_data_widget(self.data_widget)

        self.labels_widget.changepoints_widget = self.changepoints_widget
        self.labels_widget.io_widget = self.io_widget
        self.plot_settings_widget.set_plot_container(self.plot_container)
        self.plot_settings_widget.set_meta_widget(self)
        self.changepoints_widget.set_plot_container(self.plot_container)
        self.changepoints_widget.set_meta_widget(self)
        self.changepoints_widget.data_widget = self.data_widget
        self.changepoints_widget.set_motif_mappings(self.labels_widget._mappings)
        self.navigation_widget.set_mappings(self.labels_widget._mappings)
        self.navigation_widget._data_widget = self.data_widget
        self.navigation_widget.set_plot_container(self.plot_container)
        self.ephys_widget.set_plot_container(self.plot_container)
        self.ephys_widget.set_meta_widget(self)
        self.ephys_widget.set_data_widget(self.data_widget)
        self.ephys_widget.io_widget = self.io_widget

        # Wire IOWidget signals to LabelsWidget and EphysWidget methods
        self.io_widget.wire_label_signals()
        self.io_widget.wire_ephys_signals(self.ephys_widget)

        # Signal connections for decoupled communication
        self.plot_container.labels_redraw_needed.connect(self._on_labels_redraw_needed)
        # A label's labeling_method changed: the green/red trial colouring
        # re-reads the per-trial verdict (the bottom bar listens on its own).
        self.app_state.curation_changed.connect(self.data_widget.update_trials_combo)
        self.plot_container.panel_content_changed.connect(self._rebind_console)
        self.app_state.trial_changed.connect(self.data_widget.on_trial_changed)
        # After DataWidget.on_trial_changed: the catalog it rebuilds is what the
        # console's derived features are removed from.
        self.app_state.trial_changed.connect(self._on_trial_changed_console)
        self.app_state.trial_changed.connect(self.changepoints_widget._update_cp_status)
        self.changepoints_widget.changepoint_correction_checkbox.stateChanged.connect(
            self.update_changepoints_widget_title
        )

        # The one widget to rule them all (loading data, updating plots, managing sync)
        self.data_widget.set_references(
            self.plot_container,
            self.labels_widget,
            self.plot_settings_widget,
            self.navigation_widget,
            self.changepoints_widget,
            ephys_widget=self.ephys_widget,
            layout_mgr=self.layout_mgr,
            trials_widget=self.trials_widget,
        )
        self.data_widget.on_pose_updated = self._on_pose_updated

        for widget in [
            self.help_widget,
            self.io_widget,
            self.data_panel,
            self.labels_widget,
            self.changepoints_widget,
            self.ephys_widget,
            self.plot_settings_widget,
            self.navigation_widget,
            self.trials_widget,
        ]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # ── Right sidebar: exactly three sections — Data | Labels | Navigation.
        # The "Data" section is context-sensitive: it borrows the setting groups
        # from DataPanel / PlotSettingsWidget / NavigationWidget and shows only
        # the ones relevant to the plot the user last clicked (see _on_plot_focus).
        self.context_panel = self._build_context_panel()
        self._add_feature_view_switch()

        # The label-name overlay drawn on the video is a video display setting,
        # so its checkbox lives in the sidebar's video context.
        video_label_gb = getattr(self.data_panel, "videolabel_groupbox", None)
        if video_label_gb is not None:
            self.labels_widget.attach_video_groupbox(video_label_gb)

        # The label-branch overlay selectors (Main / top1 / top2) belong with the
        # Labels section, retitled "Label overlay".
        overlay_gb = getattr(self.data_panel, "overlays_groupbox", None)
        if overlay_gb is not None:
            overlay_gb.setTitle("Label overlay")
            row1 = getattr(self.data_panel, "overlays_row1_layout", None)
            self.labels_widget.attach_overlay_groupbox(overlay_gb, row_layout=row1)
            if self.labels_widget.layout() is not None:
                self.labels_widget.layout().addWidget(overlay_gb)

        self.add_widget(self.context_panel, collapsible=True, widget_title="Data")
        self.add_widget(self.labels_widget, collapsible=True, widget_title="Labels")
        self.add_widget(self._build_nav_tab(), collapsible=True, widget_title="Navigation")

        # Everything else moved to the top-bar pop-ups. Park those widgets in a
        # hidden holder so they stay alive and can be borrowed by SectionPopup.
        # data_panel / plot_settings are now shells (their groups live in the
        # context panel) but kept alive here so their attributes stay valid.
        self._detached_holder = QWidget(self)
        self._detached_holder.setVisible(False)
        holder_layout = QVBoxLayout(self._detached_holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        for w in (
            self.help_widget,
            self.io_widget,
            self.ephys_widget,
            self.changepoints_widget,
            self.data_panel,
            self.plot_settings_widget,
        ):
            holder_layout.addWidget(w)
            w.setVisible(False)

        normalize_child_layouts(
            self,
            spacing=DEFAULT_LAYOUT_SPACING,
            margin=DEFAULT_LAYOUT_MARGIN,
        )

        self._overlays_relocated = False
        self._wire_plot_focus()

    def _relocate_overlay_checkboxes(self):
        """Move the Confidence checkbox to the predictions importer and the
        Envelope checkbox to the spectrogram (audio) settings — reparenting
        preserves their existing overlay signal connections.

        The checkboxes are built lazily during data load, so this runs once
        after the first dataset is configured.
        """
        if getattr(self, "_overlays_relocated", False):
            return
        conf_cb = getattr(self.data_widget, "show_confidence_checkbox", None)
        pred_row = getattr(self.io_widget, "_pred_controls_row", None)
        if conf_cb is not None and pred_row is not None:
            pred_row.insertWidget(0, conf_cb)

        # Envelope belongs with the audio-trace energy controls, not spectrogram.
        env_cb = getattr(self.data_widget, "show_envelope_checkbox", None)
        energy_group = getattr(self.data_panel, "energy_group", None)
        if env_cb is not None and energy_group is not None and energy_group.layout() is not None:
            energy_group.layout().addWidget(env_cb, 2, 0, 1, 3)
            env_cb.show()

        if conf_cb is not None or env_cb is not None:
            self._overlays_relocated = True

    def _build_context_panel(self):
        """Borrow the per-plot setting groups into one context-sensitive panel."""
        from .right_context import RightContextPanel

        dp = self.data_panel
        ps = self.plot_settings_widget
        sections = {
            "individual": getattr(dp, "individual_groupbox", None),
            "coords": getattr(dp, "coords_groupbox", None),
            "slot": getattr(dp, "slot_groupbox", None),
            "videocrop": getattr(dp, "videocrop_groupbox", None),
            "videolabel": getattr(dp, "videolabel_groupbox", None),
            "overlay": getattr(dp, "overlay_groupbox", None),
            "pose": getattr(dp, "pose_groupbox", None),
            "bbox": getattr(dp, "bbox_groupbox", None),
            "energy": getattr(dp, "energy_group", None),
            "audiochannel": getattr(ps, "audio_channel_group", None),
            "neocontrols": getattr(ps, "neo_controls_group", None),
            "phy": getattr(self.ephys_widget, "traceview_panel", None),
            "lineplot": getattr(ps, "lineplot_panel", None),
            "spaceplot": getattr(ps, "spaceplot_panel", None),
            "radialplot": getattr(ps, "radialplot_panel", None),
            "spectrogram": getattr(ps, "spectrogram_panel", None),
            "heatmap": getattr(ps, "heatmap_panel", None),
            "shared": getattr(ps, "shared_widget", None),
        }
        panel = RightContextPanel(sections)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return panel

    def _add_feature_view_switch(self):
        """Add a per-panel "Feature plot type" combo (Lineplot/Heatmap) to the
        Xarray-coords group (shown for both the lineplot and heatmap contexts).

        The combo reflects the *active* feature panel's type and converts it
        in place: the panel's own settings (feature, dim selections, color)
        are carried over to the target type."""
        coords_layout = getattr(self.data_panel, "coords_groupbox_layout", None)
        if coords_layout is None:
            return
        self.feature_view_combo = QComboBox()
        self.feature_view_combo.addItems(["Lineplot", "Heatmap"])
        self.feature_view_combo.setToolTip("How the active feature panel is rendered: line plot or heatmap.")
        self.feature_view_combo.currentTextChanged.connect(self._on_feature_view_changed)
        coords_layout.insertRow(0, "Feature plot type:", self.feature_view_combo)

    def _on_feature_view_changed(self, text: str):
        if not self.app_state.ready:
            return
        pc = self.plot_container
        active = pc.active_feature_plot
        target = "heatmap" if text == "Heatmap" else "lineplot"
        if getattr(active, "panel_type", None) == target:
            return

        # Convert the active feature panel in place: a new instance of the
        # target type takes over its settings, the old instance is removed.
        convertible = getattr(active, "panel_group", None) == "feature"
        settings = active.panel_settings() if convertible else None
        plot = pc.add_panel(target, feature=(settings or {}).get("feature"))
        if plot is None:
            return
        if settings is not None:
            plot.apply_panel_settings(settings)
        if convertible:
            pc.remove_panel(active)
        pc.active_feature_plot = plot
        plot.update_plot()
        self._activate_panel(plot, target)

    def _build_nav_tab(self) -> QWidget:
        """Navigation section = trials table (on top) + navigation controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.trials_widget, stretch=1)
        layout.addWidget(self.navigation_widget)
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return tab

    def _wire_plot_focus(self):
        """Register every panel (plots + video + space) with the ActivePanelManager
        so clicking any of them highlights it (green edge) and shows its controls."""
        from .active_panel import ActivePanelManager, PanelKind

        pc = self.plot_container
        self.active_panels = ActivePanelManager(self)
        self.active_panels.active_changed.connect(self._on_active_panel)
        pc.active_panels = self.active_panels  # dynamic panels register in add_panel
        # Leaving zen mode: the sidebar skipped every context swap while zen
        # was on, so re-sync it to whatever panel is active now.
        self.app_state.zen_mode_changed.connect(self._on_zen_mode_changed)

        # Only the fixed singletons register here; every dynamic panel
        # (lineplot/heatmap/audiotrace/spectrogram) registers per instance.
        fixed = [
            (pc.ephys_trace_plot, PanelKind.EPHYS, None),
            (pc.raster_plot, PanelKind.RASTER, None),
        ]
        for widget, kind, plot in fixed:
            if widget is not None and hasattr(widget, "plot_clicked"):
                self.active_panels.register(widget, kind, clicked_signal=widget.plot_clicked, plot=plot)

        # Video: primary camera view + any extra cameras added later.
        video_area = getattr(self.shell, "video_area", None)
        if video_area is not None:
            primary = getattr(video_area, "primary", None)
            if primary is not None and hasattr(primary, "clicked"):
                self.active_panels.register(primary, PanelKind.VIDEO, clicked_signal=primary.clicked)
            if hasattr(video_area, "camera_added"):
                video_area.camera_added.connect(
                    lambda view: self.active_panels.register(view, PanelKind.VIDEO, clicked_signal=view.clicked)
                )
            if hasattr(video_area, "camera_view_removed"):
                video_area.camera_view_removed.connect(self.active_panels.unregister)

    _CONTEXT_KINDS = frozenset({"audiotrace", "spectrogram", "lineplot", "heatmap", "space", "radial", "ephys", "neo"})

    def _track_subject_panel(self, widget) -> None:
        """The clicked panel's individual becomes the labelling subject (see ``selected_individual``)."""
        state = self.app_state
        before = state.selected_individual()
        state.set_subject_panel(widget)
        if state.ready and state.selected_individual() != before:
            self.data_widget.on_labelling_subject_changed()

    def _on_active_panel(self, reg):
        """A panel was clicked → track it, and show its controls in the sidebar.

        The green edge is drawn by the manager for every panel; here we only swap
        the sidebar context for panels that have one (ephys/neo/raster keep the
        current sidebar — their controls live in the top-bar Neural menu)."""
        from .active_panel import PanelKind

        kind = reg.kind
        if kind in PanelKind.FEATURE and reg.plot is not None:
            plot_changed = self.plot_container.active_feature_plot is not reg.plot
            self.plot_container.active_feature_plot = reg.plot
            self._track_subject_panel(reg.plot)
            # The dotted prediction-confidence curve is hosted on the current
            # plot — re-render so it follows (or hides on) the new active plot.
            # A re-click of the same panel re-announces (sidebar sync) but the
            # overlay host didn't move, so skip the re-render.
            if plot_changed and self.app_state.ready:
                self.data_widget._update_confidence_overlay()
        if kind == PanelKind.SPACE:
            self.data_widget.set_active_space_plot(reg.widget)
        if kind == PanelKind.RADIAL:
            self.data_widget.set_active_radial_plot(reg.widget)
        if kind in (PanelKind.AUDIOTRACE, PanelKind.SPECTROGRAM):
            self.plot_settings_widget.set_active_audio_plot(reg.widget)
            # Playback follows the last-clicked audio panel (its pin, else global).
            self.app_state.playback_mic_key = getattr(reg.widget, "mic_name", None) or self.app_state.mics_sel
        if kind == PanelKind.NEO:
            self.plot_settings_widget.set_active_neo_plot(reg.widget)
        self._update_video_selection(reg if kind == PanelKind.VIDEO else None)
        if kind == PanelKind.VIDEO:
            self._track_subject_panel(reg.widget)
            self.focus_video_context()
        elif kind in self._CONTEXT_KINDS:
            self._on_plot_focus(kind)

    def _update_video_selection(self, reg):
        """Mark exactly the active camera view as selected so wheel-zoom only
        acts on it; a non-video active panel deselects all views (``reg`` None)."""
        video_area = getattr(self.shell, "video_area", None)
        if video_area is None:
            return
        active = reg.widget if reg is not None else None
        for view in [getattr(video_area, "primary", None), *video_area.extras.values()]:
            if view is not None:
                view.selected = view is active

    def show_source_popup(self, anchor: QWidget | None = None):
        """Open the add-panel popup (bottom-bar ➕ button or Shift+N).

        With *anchor* (the ➕ button) the popup opens upward from it; without,
        it opens at the plot area's top-left corner.
        """
        if not self.app_state.ready:
            notify("Load a dataset before adding panels.", "warning")
            return
        self.refresh_source_popup()
        if anchor is not None:
            pos = anchor.mapToGlobal(anchor.rect().topLeft())
            self.source_popup.popup_at(pos, open_upward=True)
        else:
            pos = self.plot_container.mapToGlobal(self.plot_container.rect().topLeft())
            self.source_popup.popup_at(pos)

    def _phy_available(self) -> bool:
        """Whether the raw-data Phy trace can be offered/added."""
        ew = self.ephys_widget
        return bool(getattr(self.app_state, "has_neurons", False)) and ew is not None and ew.has_phy_trace()

    def refresh_source_popup(self):
        """Repopulate the add-panel popup from the current session (Media,
        Features, Neo streams, and the Phy trace when raw ephys is loaded)."""
        try:
            neo_streams = self.data_widget.neo_stream_names()
        except Exception:  # ephys probing must never block the add-panel popup
            logger.exception("neo_stream_names failed; opening popup without Neo sources")
            neo_streams = []
        self.source_popup.refresh(
            catalog=self.data_widget.catalog,
            neo_streams=neo_streams,
            phy_available=self._phy_available(),
        )

    def _on_source_dropped(self, kind: str, name: str):
        """Popup source dropped on the plot area (or Enter) → pick type & create panel."""
        if not self.app_state.ready:
            notify("Load a dataset before adding panels.", "warning")
            return
        options = allowed_plot_types(kind, name, self.app_state)
        if not options:
            return
        # Only one possible plot type (e.g. video) → no need to ask.
        if len(options) == 1:
            self._create_panel_for_source(kind, name, options[0])
            return
        picker = PlotTypePicker(options, parent=self.shell)
        if picker.exec_() and picker.choice:
            self._create_panel_for_source(kind, name, picker.choice)

    def _activate_panel(self, widget, kind: str):
        """Make a just-created panel the active one (green edge + sidebar
        controls showing its feature/selections)."""
        from .active_panel import PanelKind

        mgr = self.active_panels
        reg = mgr.registration_for(widget) if mgr is not None else None
        if reg is not None:
            # set_active always re-announces (even if already active), so
            # _on_active_panel syncs the sidebar for us.
            mgr.set_active(reg)
            # A brand-new panel has no click behind it, so nothing else
            # announces its contents to the console.
            if reg.kind in PanelKind.FEATURE and reg.plot is not None:
                self.plot_container.panel_content_changed.emit(reg.plot)
        else:
            # Not (yet) registered with the manager — still sync the sidebar.
            self._on_plot_focus(kind)

    def _create_panel_for_source(self, kind: str, name: str, plot_type: str):
        pc = self.plot_container
        if kind == "feature":
            if plot_type == "Lineplot":
                plot = pc.add_lineplot(feature=name)
                if plot is not None:
                    self._activate_panel(plot, "lineplot")
            elif plot_type == "Heatmap":
                plot = pc.add_panel("heatmap", feature=name)
                if plot is not None:
                    self._activate_panel(plot, "heatmap")
                    notify(f"Heatmap: {name}")
            elif plot_type == "Radial":
                self.data_widget.add_radial_plot(feature=name)
                notify(f"Radial: {name}")
            elif plot_type.startswith("Space"):
                # Every drop creates a new instance — space plots never
                # replace each other.
                self.app_state.space_plot_type = "Space Plot"
                self.data_widget.add_space_plot(feature=name, view_3d=plot_type == "Space (3D)")
                notify(f"{plot_type}: {name}")
        elif kind == "audio":
            # The popup only lists mics that exist in the alignment.
            self.app_state.has_audio = True
            # audio_path may be unset if no audio panel existed at load —
            # re-resolve it before wiring the new panel's source.
            self.data_widget.update_audio()
            panel_type = "spectrogram" if plot_type == "Spectrogram Trace" else "audiotrace"
            keys = self._audio_channel_keys(name)
            if len(keys) > 1:
                source_map = getattr(self.app_state, "audio_source_map", None) or {}
                labels = [f"Channel {source_map.get(k, (k, 0))[1] + 1}" for k in keys]
                picker = PlotTypePicker(labels, parent=self.shell, title="Channel")
                if not (picker.exec_() and picker.choice):
                    return
                mic_key = keys[labels.index(picker.choice)]
            elif keys:
                mic_key = keys[0]
            else:
                mic_key = self._mic_key_for_source(name)
            # Every drop creates a NEW panel instance — duplicates are fine
            # (e.g. the same mic on two channels side by side); the user
            # removes extras via ✕.
            plot = pc.add_audio_panel(panel_type, mic_name=mic_key)
            if plot is not None:
                self._activate_panel(plot, panel_type)
        elif kind == "neo":
            self._add_neo_panel(name)
        elif kind == "phy":
            self._add_phy_panel()
        elif kind == "video":
            self._add_camera_view(name)
        elif kind == "image":
            self._add_image_view(name)
        elif kind == "console":
            self._add_console_panel()
        elif kind == "labels":
            plot = pc.add_panel("labels")
            if plot is not None:
                self._activate_panel(plot, "labels")

    def ensure_label_ribbon(self):
        """Open a label timeline when nothing else is on screen.

        The timeline belongs to labelling at the current frame (the checkbox
        is shown only there), so plots mode never gets one unasked.
        A session with only a video opens with no panel, so a label placed
        from the video would be invisible. The Labels tab's checkbox
        (``label_ribbon_auto``) asks for the empty axis in that case; a
        session that already shows a panel gets nothing extra.
        """
        pc = self.plot_container
        if self.app_state.get_with_default("labelling_mode") != LABELLING_MODE_FRAME:
            return
        if not self.app_state.get_with_default("label_ribbon_auto"):
            return
        if pc.has_open_plots():
            return
        pc.add_panel("labels")

    def _add_console_panel(self):
        """Open (or re-show) the Python console panel and bind whatever feature
        panel is already active, so it is usable without a second click."""
        panel = self.plot_container.add_console_panel()
        if not getattr(panel, "_features_wired", False):
            panel.features_changed.connect(self._on_derived_features_changed)
            panel._features_wired = True
        active = self.plot_container.active_feature_plot
        if active is not None:
            panel.bind_panel(active)

    def _on_derived_features_changed(self):
        """A console assignment added or removed a derived feature — refresh the
        one canonical feature list everywhere it is shown."""
        self.data_widget.refresh_feature_choices()
        self.data_widget.refresh_radial_plots()
        self._close_panels_for_missing_features()
        self.data_widget.refresh_overlay_choices()

    def _close_panels_for_missing_features(self):
        """Close any panel whose feature no longer exists.

        Derived features are dropped by ``forget()``, ``clear(all=True)`` and
        every trial change; a panel left showing one can never render again, so
        it would sit there permanently blank under a name nothing answers to.
        """
        available = set(self.plot_container._available_features())
        if not available:
            return  # no catalog yet — "unknown", not "nothing exists"
        for plot in [*self.plot_container.line_plots, *self.plot_container.heatmap_plots]:
            feature = plot._effective_feature()
            if feature and feature not in available:
                self.plot_container.remove_panel(plot)
        for space_plot in list(self.data_widget.space_plots):
            feature = space_plot.feature_combo.currentText()
            if feature and feature not in available:
                self.data_widget.remove_space_plot(space_plot)
        for radial_plot in list(self.data_widget.radial_plots):
            feature = radial_plot.feature_combo.currentText()
            if feature and feature not in available:
                self.data_widget.remove_radial_plot(radial_plot)

    def _rebind_console(self, plot):
        """Keep the console describing the panel as it is NOW.

        ``panel_content_changed`` is the console's ONLY rebind channel: it
        fires on every click AND on sidebar feature/selection edits, and a
        click may also emit ``active_changed`` — binding there too would
        double-bind.
        """
        console = self.plot_container.console_panel
        if console is not None and plot is not None:
            console.bind_panel(plot)

    def _on_trial_changed_console(self, *_):
        """Derived features live for one trial only (see ``reset_for_trial``)."""
        console = self.plot_container.console_panel
        if console is not None:
            console.reset_for_trial()

    def _add_phy_panel(self):
        """Add (or re-show) the Phy-like raw-data trace panel. It is a singleton
        toggled visible — closing it hides it; re-adding here brings it back."""
        ew = self.ephys_widget
        pc = self.plot_container
        if ew is None or not ew.has_phy_trace():
            notify("No raw ephys/Kilosort data loaded for the Phy viewer.", "warning")
            return
        pc.set_neural_panel_mode("trace")
        ew.configure_ephys_trace_plot()
        self._activate_panel(pc.ephys_trace_plot, "ephys")
        pc.schedule_labels_redraw()

    def _add_neo_panel(self, stream_name: str):
        """Dropping a Neo stream/modality → pick channels (default all) → add a
        new Neo trace panel showing those channels of that stream."""
        dw = self.data_widget
        n_ch = dw.neo_stream_channel_count(stream_name)
        if n_ch <= 0:
            notify(f"Could not read channels for Neo stream '{stream_name}'.", "warning")
            return
        channels = None
        if n_ch > 1:
            dlg = ChannelSelectDialog(n_ch, parent=self.shell, title=f"Channels — {stream_name}")
            if not dlg.exec_():
                return
            channels = dlg.selected_channels()
            if channels is not None and not channels:
                return
        plot = dw.add_neo_panel(stream_name, channels=channels)
        if plot is not None:
            self._activate_panel(plot, "neo")

    def _audio_channel_keys(self, name: str) -> list[str]:
        """Ordered ``audio_source_map`` keys (one per channel) for a popup
        audio source *name* (a mic device label, or already a map key)."""
        groups = getattr(self.app_state, "audio_mic_channels", None) or {}
        if name in groups:
            return list(groups[name])
        source_map = getattr(self.app_state, "audio_source_map", None) or {}
        if name in source_map:
            for keys in groups.values():
                if name in keys:
                    return list(keys)
            return [name]
        return []

    def _mic_key_for_source(self, name: str) -> str | None:
        """Map a popup audio-source name to an ``audio_source_map`` key so the
        new panel is pinned to that mic/channel. None → follow the Mic combo."""
        source_map = getattr(self.app_state, "audio_source_map", None) or {}
        if name in source_map:
            return name
        for key in source_map:
            if name in key:
                return key
        return None

    def _add_camera_view(self, name: str):
        """Dropping a video source adds a view of that camera — always.

        If no video is currently open, the dropped camera becomes the primary
        view (which sets up playback/sync). Otherwise a NEW extra follower
        view is created — duplicates of an already-shown camera (including
        the primary) are fine; the user removes extras themselves.
        """
        dw = self.data_widget
        vm = getattr(dw, "video_mgr", None)
        if vm is None:
            notify("No video is available for this session.", "warning")
            return

        name = str(name)
        # Base this on whether a video is actually loaded in the primary view —
        # app_state.video (the sync object) can linger from a previous session.
        # A loaded-but-hidden primary (its dock closed without the teardown
        # path, e.g. a restored layout) counts as NOT open: re-adding must
        # re-show the primary, never fork an extra over an invisible one.
        dock = getattr(self.shell, "_video_dock", None)
        primary_hidden = dock is not None and not dock.isVisible()
        primary_open = vm.primary_view.has_video and not primary_hidden

        if not primary_open:
            # Nothing playing yet → open this camera as the primary video.
            self.app_state.primary_camera = name
            combo = getattr(dw, "primary_camera_combo", None)
            if combo is not None:
                idx = combo.findText(name)
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
            self.shell.set_video_viewer_visible(True)
            vm.update_video(plot_container=self.plot_container)
            if getattr(self.app_state, "video", None) is None and not vm.primary_view.has_video:
                notify(f"No video file found for camera '{name}'.", "warning")
            else:
                # Loading a video and overlaying its pose always go together
                # (same pairing as on_trial_changed / _on_primary_camera_changed)
                # — a closed-then-re-added primary starts from a cleared view.
                dw.update_pose()
                notify(f"Opened video: {name}")
            return

        video_path = vm._resolve_video_path(name, self.app_state.video_folder)
        if not video_path:
            notify(f"No video file found for camera '{name}'.", "warning")
            return
        self.shell.set_video_viewer_visible(True)
        vm.add_camera(
            camera_name=name,
            video_path=video_path,
            layout_mgr=getattr(dw, "layout_mgr", None),
            meta_widget=self,
            duplicate=True,
        )
        pose_mgr = getattr(dw, "pose_mgr", None)
        if pose_mgr is not None and hasattr(dw, "get_hidden_keypoints"):
            try:
                pose_mgr.update_extra_camera_pose(name, dw.get_hidden_keypoints())
            except Exception:  # noqa: BLE001 - pose is best-effort
                pass
        notify(f"Added camera view: {name}")

    def _add_image_view(self, name: str):
        """Dropping an image source adds a static view of it — always.

        ``IMAGE_BROWSE`` (the popup's "Image — browse…" entry) first asks for a
        file and registers it in ``app_state.image_paths`` so it stays listed
        as a Media source. The primary camera's pose/skeleton is overlaid and
        animates with the time marker.
        """
        dw = self.data_widget
        vm = getattr(dw, "video_mgr", None)
        if vm is None:
            notify("Load a dataset before adding image views.", "warning")
            return

        path = name
        if path == IMAGE_BROWSE:
            path = browse_open_file(
                self.shell,
                self.app_state,
                "Choose an image",
                IMAGE_FILE_FILTER,
                preferred_dir=self.app_state.nc_file_path,
            )
            if not path:
                return
            images = list(getattr(self.app_state, "image_paths", None) or [])
            if path not in images:
                self.app_state.image_paths = [*images, path]

        self.shell.set_video_viewer_visible(True)
        view = vm.add_image_view(path)
        if view is None:
            return
        if getattr(dw, "pose_mgr", None) is not None and hasattr(dw, "get_hidden_keypoints"):
            try:
                dw.pose_mgr._display_pose_on_image(view, dw.get_hidden_keypoints())
            except Exception:  # noqa: BLE001 - pose is best-effort
                pass
        notify(f"Added image view: {Path(path).name}")

    def _on_plot_focus(self, panel_type: str):
        """Show only the clicked plot's settings, unless zen / Labels / Nav active.

        Refreshes only when the plot *type* changes (RightContextPanel.set_context
        is a no-op for the same type).
        """
        if getattr(self.app_state, "zen_mode", False):
            return
        # Skip when the user is on the Labels (1) or Navigation (2) section.
        if getattr(self, "_active", None) in (1, 2):
            return
        # "feature" resolves to whichever plot the feature slot is showing.
        if panel_type == "feature":
            panel_type = getattr(self.plot_container, "_feature_type", "lineplot")
        if panel_type in ("lineplot", "heatmap"):
            if hasattr(self, "feature_view_combo"):
                self.feature_view_combo.blockSignals(True)
                self.feature_view_combo.setCurrentText("Heatmap" if panel_type == "heatmap" else "Lineplot")
                self.feature_view_combo.blockSignals(False)
            # Show the clicked plot's own feature/dim/colour/All selections + axes.
            if hasattr(self.data_widget, "sync_sidebar_from_active_plot"):
                self.data_widget.sync_sidebar_from_active_plot()
            if hasattr(self.plot_settings_widget, "sync_axes_to_active_plot"):
                self.plot_settings_widget.sync_axes_to_active_plot()
        has_pose = bool(getattr(self.app_state, "has_pose", False)) or self._pose_available()
        if self.collapsible_widgets:
            self.collapsible_widgets[0].expand()
        if self.context_panel.set_context(panel_type, has_pose=has_pose, pose_kind=self._pose_kind()):
            self.refresh_widget_layout(self.context_panel)

    def _pose_kind(self) -> str | None:
        pose_mgr = getattr(self.data_widget, "pose_mgr", None)
        return pose_mgr.pose_kind() if pose_mgr is not None else None

    def _pose_available(self) -> bool:
        sio = getattr(self.app_state, "nwb_alignment", None)
        if sio is not None and getattr(sio, "pose_keys", None):
            return True
        ds = getattr(self.app_state, "ds", None)
        if ds is not None:
            return any("position" in str(v) for v in getattr(ds, "data_vars", {}))
        return False

    def _on_pose_updated(self):
        """The overlay was redrawn: if the video's settings are showing, re-read
        what is drawn, so boxes get the Bounding boxes section without a click."""
        if self.context_panel.current_context() == "video":
            self.focus_video_context()

    def focus_video_context(self):
        """Called when the video viewer is clicked → show pose + playback."""
        if getattr(self.app_state, "zen_mode", False):
            return
        if getattr(self, "_active", None) in (1, 2):
            return
        if self.collapsible_widgets:
            self.collapsible_widgets[0].expand()
        if self.context_panel.set_context("video", has_pose=self._pose_available(), pose_kind=self._pose_kind()):
            self.refresh_widget_layout(self.context_panel)

    def _sync_context_to_active(self):
        """Re-apply the sidebar context for the manager's active panel.

        The context swap is suppressed while zen mode is on or the Labels /
        Navigation section is open, so the sidebar can go stale relative to
        the active panel; call this when those suppressors lift.
        """
        from .active_panel import PanelKind

        mgr = getattr(self, "active_panels", None)
        reg = mgr.active if mgr is not None else None
        if reg is None:
            return
        if reg.kind == PanelKind.VIDEO:
            self.focus_video_context()
        elif reg.kind in self._CONTEXT_KINDS:
            self._on_plot_focus(reg.kind)

    def _on_zen_mode_changed(self, on: bool):
        if not on:
            self._sync_context_to_active()

    def _expand(self, index: int) -> None:
        """Returning to the Data section re-syncs the context — panel clicks
        made while Labels/Navigation was open were deliberately not applied."""
        was = self._active
        super()._expand(index)
        if index == 0 and was != 0:
            self._sync_context_to_active()

    def _set_default_context(self):
        """Pick an initial context after load so the Data section isn't empty."""
        if getattr(self.app_state, "video", None) is not None:
            self.context_panel.set_context("video", has_pose=self._pose_available(), pose_kind=self._pose_kind())
        else:
            sio = getattr(self.app_state, "nwb_alignment", None)
            if sio is not None and getattr(sio, "mics", None):
                self.context_panel.set_context("audio")
            else:
                self.context_panel.set_context("lineplot")

    def _connect_collapsible_layout_refresh(self):
        from qtpy.QtCore import QEvent

        self._layout_refresh_timer = QTimer(self)
        self._layout_refresh_timer.setSingleShot(True)
        self._layout_refresh_timer.setInterval(50)
        self._layout_refresh_timer.timeout.connect(self._recalc_collapsible_heights)
        self._watched_events = {QEvent.Type.LayoutRequest, QEvent.Type.Resize}
        for collapsible in self.collapsible_widgets:
            collapsible.toggled.connect(self._schedule_layout_refresh)
            content = collapsible.content()
            if content:
                content.installEventFilter(self)

    def eventFilter(self, obj, event):
        if hasattr(self, "_watched_events") and event.type() in self._watched_events:
            self._schedule_layout_refresh()
        return False

    def _schedule_layout_refresh(self, *_args):
        self._layout_refresh_timer.start()

    def _recalc_collapsible_heights(self):
        from qtpy.QtCore import QPropertyAnimation

        for collapsible in self.collapsible_widgets:
            if not collapsible.isExpanded():
                continue
            content = collapsible.content()
            if content is None:
                continue
            content.updateGeometry()
            layout = content.layout()
            if layout:
                layout.invalidate()
                layout.activate()
            collapsible._expand_collapse(
                QPropertyAnimation.Direction.Forward,
                animate=False,
                emit=False,
            )

    def refresh_widget_layout(self, widget: QWidget):
        self._schedule_layout_refresh()
        self.notify_content_resized()

    def _cycle_channel(self, direction: int):
        if not self.app_state.ready:
            return
        if self.plot_container and self.plot_container.is_ephystrace():
            spin = self.ephys_widget.ephys_channel_spin
            if spin.isVisible():
                new_val = spin.value() + direction
                new_val = max(spin.minimum(), min(new_val, spin.maximum()))
                spin.setValue(new_val)

    def _on_labels_redraw_needed(self):
        if not self.app_state.ready:
            return
        self.data_widget.update_label_plot()
        self.data_widget.update_trials_combo()

    def update_changepoints_widget_title(self):
        if hasattr(self, "collapsible_widgets") and len(self.collapsible_widgets) > 5:
            cp_collapsible = self.collapsible_widgets[5]
            correction_enabled = self.changepoints_widget.changepoint_correction_checkbox.isChecked()
            indicator = "\U0001f3af" if correction_enabled else "⭕"
            self._set_collapsible_title(cp_collapsible, f"Changepoints (CPs) {indicator}")

    @staticmethod
    def _set_collapsible_title(collapsible, new_title: str):
        if hasattr(collapsible, "setText"):
            collapsible.setText(new_title)
        elif hasattr(collapsible, "setTitle"):
            collapsible.setTitle(new_title)
        elif hasattr(collapsible, "_title_widget") and hasattr(collapsible._title_widget, "setText"):
            collapsible._title_widget.setText(new_title)

    def flush_pending_writes(self):
        """Write out anything still sitting in a debounce timer (app close)."""
        self.trials_widget.flush_metadata()

    def _check_unsaved_changes(self, event):
        """Check for unsaved changes and prompt. Returns True if OK to close.

        Asks the labels file, not just the ``changes_saved`` flag: the flag is
        cleared by anything label-adjacent, which had metadata-only sessions
        being asked to save labels the user never touched.
        """
        if self.app_state.labels_dirty():
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Unsaved Changes")
            msg_box.setText("You have unsaved changes to your labels.")
            msg_box.setInformativeText("Would you like to save your changes before closing?")
            msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Save)
            response = msg_box.exec_()
            if response == QMessageBox.Save:
                try:
                    self.app_state.save_labels()
                    return True
                except (OSError, PermissionError) as e:
                    error_msg = QMessageBox()
                    error_msg.setWindowTitle("Save Error")
                    error_msg.setText(f"Failed to save changes: {str(e)}")
                    error_msg.exec_()
                    event.ignore()
                    return False
            elif response == QMessageBox.Cancel:
                event.ignore()
                return False
        return True

    def _bind_global_shortcuts(self, labels_widget, data_widget):
        bind_global_shortcuts(self)

    def _set_compact_font(self, font_size: int = 8):
        apply_compact_widget_style(self, font_size=font_size)

    def _set_sidebar_default_width(self):

        self.setMinimumWidth(SIDEBAR_MIN_WIDTH_PX)

        def _apply():
            dock = getattr(self.shell, "_sidebar_dock", None)
            if dock is None:
                return
            total_w = self.shell.width()
            if total_w <= 0:
                return
            ratio = max(0.15, min(0.6, SIDEBAR_DEFAULT_WIDTH_RATIO))
            target_w = max(SIDEBAR_MIN_WIDTH_PX, int(total_w * ratio))
            self.shell.resizeDocks([dock], [target_w], Qt.Horizontal)

        QTimer.singleShot(0, _apply)

    def configure_layout_for_data(self):
        """Configure panel visibility and layout after a dataset load."""
        self.plot_container.configure_panels()
        self.source_popup.refresh(
            catalog=self.data_widget.catalog,
            neo_streams=self.data_widget.neo_stream_names(),
            phy_available=self._phy_available(),
        )
        self._relocate_overlay_checkboxes()
        self._set_default_context()

        # Neo + Phy trace panels are heavy and are added on demand from the
        # popup, not shown automatically. Just resolve the Phy loader stream
        # and pre-wire its source so it renders instantly when added.
        if self.app_state.ephys_path:
            self.data_widget._ensure_default_ephys_stream()

        if self.app_state.has_neurons:
            self.data_widget._configure_ephys_trace_plot()

        self.layout_mgr.register_docks()

        # The space plot no longer appears by default (that was a napari-era
        # artefact). It is created only when the user drags a Feature → Space.

        self.apply_saved_panel_layout()
        self.ensure_label_ribbon()

    def _snapshot_layouts(self):
        """Refresh the layout snapshots before every auto-save (registered as
        app_state._layout_snapshot_provider): panel layout → the dataset's
        local_settings.yaml, window state → gui_settings.yaml."""
        if self.app_state.ready:
            layout = self.plot_container.layout_state()
            layout["space_plots"] = self.data_widget.space_layout_state()
            layout["radial_plots"] = self.data_widget.radial_layout_state()
            # Shell dock arrangement (space plots, cameras) is per-dataset
            # state and travels with the dataset's local_settings.yaml.
            layout["shell_dock_state_b64"] = self.shell.capture_dock_state_b64()
            self.app_state.panel_layout = layout
        self.app_state.window_state = self.shell.capture_window_state()

    def apply_saved_panel_layout(self):
        """Apply the dataset's saved panel layout after a load.

        ``app_state.panel_layout`` is auto-loaded with the dataset's
        local_settings.yaml — including one shipped as a template release
        asset. Absent → data-availability defaults stand.

        A saved layout is untrusted input (stale for the data now loaded,
        hand-edited, written by an older version): a failure applying it must
        never abort the load. The broken layout is discarded — so the next
        auto-save snapshots a working one — and the data-availability default
        panels are rebuilt.
        """
        layout = getattr(self.app_state, "panel_layout", None)
        if not layout:
            # Nothing to honour, so the cameras get tiled instead — but they
            # do not exist yet (the views are created with the first trial,
            # after this runs), so only the decision is made here.
            self._camera_grid_pending = True
            return
        try:
            self.plot_container.apply_layout_state(layout)
            self.data_widget.apply_space_layout_state(layout.get("space_plots"))
            self.data_widget.apply_radial_layout_state(layout.get("radial_plots"))
            blob = layout.get("shell_dock_state_b64")
            if blob:
                self.shell.apply_dock_state_b64(blob)
        except Exception:
            logger.exception("Saved panel layout could not be applied; resetting to defaults")
            self.app_state.panel_layout = None
            self._camera_grid_pending = True
            self._rebuild_default_panels()
            notify("Saved panel layout could not be applied and was reset to defaults.", "warning")

    def arrange_camera_grid_if_default(self) -> None:
        """Tile the camera docks, once, when the dataset brought no layout.

        Called after the first trial created the camera views: every camera is
        a dock in the same area, so four dropped videos arrive as one row of
        slivers (4 → 2×2, 6 → 2×3, 9 → 3×3)."""
        if not self._camera_grid_pending:
            return
        self._camera_grid_pending = False
        self.shell.video_area.arrange_grid()

    def _rebuild_default_panels(self):
        """Recover from a saved layout that failed mid-apply: drop whatever
        panels it left behind and recreate the data-availability defaults
        (the same set ``_setup_panel_controls`` builds on a fresh load)."""
        pc = self.plot_container
        for plot in list(pc._dyn_panels):
            pc.remove_panel(plot)
        dw = self.data_widget
        if self.app_state.has_audio or self.app_state.audio_path:
            mic_names = dw.catalog.mics if (self.app_state.has_audio and dw.catalog) else []
            dw._create_default_audio_panels(mic_names)
        if dw.catalog and dw.catalog.features and not pc.line_plots:
            pc.add_lineplot()
        self.ensure_label_ribbon()
        pc.schedule_labels_redraw()
