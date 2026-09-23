"""The GUI over a DeepLabCut / LightningPose project: two stages, one switch.

Entered from the cover page's **Refine DLC / LightningPose training data**
button with the project's root folder. From then on the top bar carries only
**Docs**, **Help** and the two stage buttons, and the Labels section shows
the stage's own panel:

- **Extract Frames** — the project's ``videos/`` as one trial per video with
  the model's curves (:class:`~ethograph.gui.widgets_frame_extract.FrameExtractPanel`);
- **Refine Pose** — ``labeled-data/`` as one trial per frame folder, every
  point draggable, the labels table written back
  (:class:`~ethograph.gui.dialog_collected_data.CollectedDataDialog`).

Each stage is an ordinary session folder under ``<root>/.ethograph/``
(:mod:`ethograph.labels.pose_project`), loaded through the same path as any
other session, so navigation, the trials table and the curated colouring
all work unchanged. Switching stages rebuilds the target session first —
frames extracted since are new trials — and writes any pending labels.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, Qt, QTimer
from qtpy.QtWidgets import QInputDialog, QWidget

from ethograph.gui.notify import notify, notify_dialog
from ethograph.io.session_layout import alignment_path, metadata_path
from ethograph.labels import pose_project as pp
from ethograph.labels.curation import CURATED_COLUMN, CURATED_NO, CURATED_YES

if TYPE_CHECKING:
    from ethograph.gui.dialog_collected_data import CollectedDataDialog
    from ethograph.gui.widgets_frame_extract import FrameExtractPanel

logger = logging.getLogger(__name__)

STAGE_TITLES = {pp.STAGE_EXTRACT: "Extract Frames", pp.STAGE_REFINE: "Refine Pose"}
#: What the Labels section button reads in each stage.
SECTION_LABELS = {pp.STAGE_EXTRACT: "Extract", pp.STAGE_REFINE: "Refine Pose"}


class PoseProjectMode(QObject):
    """Owns the project and the stage panels; everything else reads ``app_state``."""

    def __init__(self, meta_widget):
        super().__init__(meta_widget)
        self.meta = meta_widget
        self.app_state = meta_widget.app_state
        self.project: pp.PoseProject | None = None
        self._extract_panel: FrameExtractPanel | None = None
        self._refine_dialog: CollectedDataDialog | None = None
        self._last_trial = None
        self.app_state.trial_changed.connect(self._on_trial_changed)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.project is not None

    @property
    def stage(self) -> str:
        return str(self.app_state.pose_project_stage)

    @property
    def refine_dialog(self):
        return self._refine_dialog

    # ------------------------------------------------------------------
    # Entering and switching
    # ------------------------------------------------------------------

    def ask_scorer(self, root: Path, parent: QWidget | None) -> str | None:
        """Who is labelling: the config's scorer when it names one, else asked and remembered."""
        named = pp.PoseProject.scorer_of(root)
        if named and (root / pp.CONFIG_FILE).is_file():
            return named
        default = named or str(self.app_state.pose_project_scorer or "")
        name, ok = QInputDialog.getText(
            parent,
            "Who is labelling?",
            "Scorer name written into the labels table (a lab has several — use your own):",
            text=default,
        )
        name = name.strip()
        if not ok or not name:
            return None
        self.app_state.pose_project_scorer = name
        return name

    def enter(self, root: str | Path, parent: QWidget | None = None) -> bool:
        """Open *root* as a pose project and load its extract stage. ``False`` when refused."""
        root = Path(root)
        try:
            scorer = self.ask_scorer(root, parent) if (root / pp.VIDEOS_DIR).is_dir() else None
            if (root / pp.VIDEOS_DIR).is_dir() and (root / pp.LABELED_DATA_DIR).is_dir() and scorer is None:
                return False
            project = pp.PoseProject.open(root, scorer=scorer)
        except pp.PoseProjectError as e:
            notify_dialog(str(e), "error", "Not a pose project", parent)
            return False
        self.project = project
        self.app_state.pose_project_root = str(project.root)
        return self.switch(pp.STAGE_EXTRACT, force=True, parent=parent)

    def switch(self, stage: str, force: bool = False, parent: QWidget | None = None) -> bool:
        """Show *stage*: build its session, load it, swap the panels and the top bar."""
        if self.project is None:
            return False
        if stage == self.stage and not force:
            return True
        self.flush()
        self._release_dataset()
        try:
            session_dir = self._build_session(stage, parent)
        except (pp.PoseProjectError, RuntimeError) as e:
            # A refused project, or a session file another ethograph window still
            # holds open (Windows will not let it be rewritten): say so, stay put.
            notify_dialog(str(e), "warning", STAGE_TITLES[stage], parent)
            self._sync_top_bar()
            return False
        self.app_state.pose_project_stage = stage
        self._last_trial = None
        self._load_session(session_dir, stage)
        self._apply_stage_ui(stage)
        return bool(self.app_state.ready)

    def _release_dataset(self) -> None:
        """Close the open ``session.nc`` and alignment so a rebuilt stage can rewrite them.

        Windows refuses to replace a file another handle holds. pynwb objects
        sit in reference cycles, so an alignment replaced earlier may still be
        alive until the collector runs — hence the explicit pass.
        """
        for holder in (getattr(self.app_state, "dt", None), getattr(self.app_state, "nwb_alignment", None)):
            close = getattr(holder, "close", None)
            if close is not None:
                close()
        gc.collect()

    def _build_session(self, stage: str, parent: QWidget | None) -> Path:
        from ethograph.gui.dialog_busy_progress import BusyProgressDialog

        assert self.project is not None
        if stage == pp.STAGE_REFINE:
            return pp.build_refine_session(self.project)
        dialog = BusyProgressDialog("Reading the project's videos and predictions…", parent=parent)

        def _progress(fraction: float, name: str) -> None:
            dialog.setLabelText(f"{name} ({int(fraction * 100)}%)")
            dialog.pump_events()

        outcome, error = dialog.execute_blocking(pp.build_extract_session, self.project, _progress)
        if error is not None:
            if isinstance(error, pp.PoseProjectError):
                raise error
            raise RuntimeError(f"Could not read the project: {error}") from error
        if outcome.untracked:
            notify(f"{len(outcome.untracked)} video(s) without predictions: video only, no curves.", "warning")
        return outcome.session_dir

    def _load_session(self, session_dir: Path, stage: str) -> None:
        """Point the loader at the stage's session folder and load it."""
        state = self.app_state
        io_widget = self.meta.io_widget
        state.metadata_path = None
        state.labels_import_path = None
        state.video_folder = None
        state.audio_folder = None
        state.pose_folder = None
        state.ephys_path = None
        state.neurons_path = None
        state.image_paths = []
        state.nc_file_path = str(session_dir)
        state.nwb_file_path = str(alignment_path(session_dir))
        if stage == pp.STAGE_REFINE:
            state.metadata_path = str(metadata_path(session_dir))
        state.primary_camera = pp.CAMERA
        edit = getattr(io_widget, "nc_file_path_edit", None)
        if edit is not None:
            edit.setText(str(session_dir))
        io_widget._on_load_clicked()

    # ------------------------------------------------------------------
    # Stage UI
    # ------------------------------------------------------------------

    def _apply_stage_ui(self, stage: str) -> None:
        labels_widget = self.meta.labels_widget
        if stage == pp.STAGE_EXTRACT:
            self._drop_refine_dialog()
            self.app_state.curation_status_override = None
            if self._extract_panel is None:
                from ethograph.gui.widgets_frame_extract import FrameExtractPanel

                self._extract_panel = FrameExtractPanel(self.app_state, labels_widget, self)
            labels_widget.set_mode_widget(self._extract_panel)
            self._extract_panel.refresh_summary()
            self._split_video_and_plots()
        else:
            from ethograph.gui.dialog_collected_data import CollectedDataDialog

            assert self.project is not None
            self.app_state.curation_status_override = pp.read_curated(self.project.session_dir(pp.STAGE_REFINE))
            self._drop_refine_dialog()
            self._refine_dialog = CollectedDataDialog(self.meta.data_widget, self.project)
            labels_widget.set_mode_widget(self._refine_dialog)
            self._last_trial = self.app_state.trials_sel
        self.meta.set_short_label(1, SECTION_LABELS[stage])
        self.app_state.curation_changed.emit()
        self._sync_top_bar()
        # The stage's panel is the point of the section: open it.
        self.meta.collapsible_widgets[1].expand()

    def _split_video_and_plots(self) -> None:
        """Video on top, curves below, half the window each — unless the user arranged it.

        The seeded layout names panels but no dock geometry; a layout the GUI
        saved carries ``shell_dock_state_b64`` and that arrangement wins.
        """
        layout = getattr(self.app_state, "panel_layout", None) or {}
        if layout.get("shell_dock_state_b64"):
            return
        shell = self.meta.shell
        dock = getattr(shell, "_video_dock", None)
        if dock is None:
            return

        def _apply() -> None:
            height = shell.height()
            if height > 0:
                shell.resizeDocks([dock], [height // 2], Qt.Vertical)

        QTimer.singleShot(0, _apply)

    def _drop_refine_dialog(self) -> None:
        if self._refine_dialog is None:
            return
        self._refine_dialog.save_now()
        self._refine_dialog.close()
        self.meta.labels_widget.set_mode_widget(None)
        self._refine_dialog.deleteLater()
        self._refine_dialog = None

    def _sync_top_bar(self) -> None:
        top_bar = getattr(self.meta.shell, "_top_bar", None)
        if top_bar is not None:
            top_bar.build()

    # ------------------------------------------------------------------
    # Extract stage
    # ------------------------------------------------------------------

    def marked_frames(self) -> tuple[list[tuple[int, int]], list[int]]:
        """``(segments, points)`` of the current video in its own frames."""
        df = self.app_state.label_intervals
        fps = self._video_fps()
        if df is None or df.empty or not fps:
            return [], []
        labels = df["labels"].astype(int)
        segments = [
            (int(round(float(row.onset_s) * fps)), int(round(float(row.offset_s) * fps)))
            for row in df[labels == pp.EXTRACT_SEGMENT_LABEL].itertuples()
        ]
        points = [int(round(float(row.onset_s) * fps)) for row in df[labels == pp.EXTRACT_FRAME_LABEL].itertuples()]
        return segments, points

    def _video_fps(self) -> float | None:
        fps = self.app_state.video_fps
        return float(fps) if fps else None

    def extract_current_video(self) -> None:
        """Write the marked frames of the current video into ``labeled-data/<video>/``."""
        from ethograph.gui.dialog_busy_progress import BusyProgressDialog
        from ethograph.gui.pose_fill import VideoFrameSource

        if self.project is None or self.stage != pp.STAGE_EXTRACT:
            return
        trial = self.app_state.trials_sel
        video = next((v for v in self.project.videos() if v.stem == str(trial)), None)
        if video is None:
            notify(f"No video named {trial} in {self.project.videos_dir}", "warning")
            return
        segments, points = self.marked_frames()
        if not segments and not points:
            notify("Nothing marked in this video: mark a segment (1) or a frame (2) first.", "warning")
            return
        fps = self._video_fps()
        n_frames = int(self.app_state.num_frames or 0)
        if not fps or not n_frames:
            notify("The video is not loaded yet.", "warning")
            return
        method = str(self.app_state.pose_extract_method)
        coverage = float(self.app_state.pose_extract_coverage) / 100.0
        project, table = self.project, self.project.labels_table(video.stem)
        existing = {f for f in (pp.frame_index_of(name) for name in table.read()) if f is not None}
        dialog = BusyProgressDialog("Extracting frames…", parent=self.meta.shell)

        def _run() -> pp.ExtractOutcome:
            thumbnails = (
                VideoFrameSource(video.video, fps, n_frames, max_side=64) if method == pp.METHOD_DIVERSE else None
            )
            frames = pp.plan_frames(n_frames, segments, points, method, coverage, frames=thumbnails, exclude=existing)
            if not frames:
                raise pp.PoseProjectError("Every marked frame is already in the folder.")
            predictions = None
            if video.tracking is not None:
                keypoints, individuals, _scorer = table.schema()
                predictions = pp.predictions_by_frame(pp.tracking_dataset(video, fps), keypoints, individuals)
            source = VideoFrameSource(video.video, fps, n_frames)

            def _progress(fraction: float) -> bool:
                dialog.setLabelText(f"Writing frames… {int(fraction * 100)}%")
                dialog.pump_events()
                return True

            return pp.extract_frames(project, video, frames, source.__getitem__, n_frames, predictions, _progress)

        outcome, error = dialog.execute_blocking(_run)
        if isinstance(error, pp.PoseProjectError):
            # Nothing to do is not a failure: say so in passing, not in a dialog.
            notify(str(error), "warning")
            return
        if error is not None:
            notify_dialog(str(error), "error", "Extract frames", self.meta.shell)
            return
        # The folders are the refine stage's trials: its alignment follows every extraction
        # (and is rebuilt again on every switch, so it is never stale).
        pp.build_refine_session(project)
        notify(
            f"{len(outcome.images)} frame(s) in {outcome.folder.parent.name}/{outcome.folder.name}/ — "
            f"labels in {outcome.table.name}. Open Refine Pose to review them.",
            "info",
        )

    # ------------------------------------------------------------------
    # Refine stage: curation of frame folders
    # ------------------------------------------------------------------

    def curate_current_trial(self) -> None:
        """Ctrl+C in the refine stage: this folder's frames are all reviewed."""
        trial = self.app_state.trials_sel
        if trial is not None:
            self.mark_curated(trial)

    def mark_curated(self, trial, curated: bool = True) -> None:
        if self.project is None or self.stage != pp.STAGE_REFINE:
            return
        status = dict(self.app_state.curation_status_override or {})
        if status.get(str(trial)) == curated:
            return
        status[str(trial)] = curated
        self.app_state.curation_status_override = status
        trials_widget = getattr(self.meta, "trials_widget", None)
        if trials_widget is not None and self.app_state.ready:
            trials_widget.set_column_values(CURATED_COLUMN, {str(trial): CURATED_YES if curated else CURATED_NO})
        else:
            session_dir = self.project.session_dir(pp.STAGE_REFINE)
            pp.write_curated(session_dir, [str(t) for t in (self.app_state.trials or [])], status)
        self.app_state.curation_changed.emit()
        if curated:
            notify(f"{trial}: every frame reviewed.", "info")

    def _on_trial_changed(self) -> None:
        if self.project is None or self.stage != pp.STAGE_REFINE:
            return
        previous, self._last_trial = self._last_trial, self.app_state.trials_sel
        if previous is not None and previous != self._last_trial and self.app_state.pose_refine_next_curates:
            self.mark_curated(previous)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write whatever the refine stage still holds (stage switch, app close)."""
        if self._refine_dialog is not None:
            self._refine_dialog.save_now()
