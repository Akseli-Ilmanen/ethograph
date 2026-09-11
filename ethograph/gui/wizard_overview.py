"""The Data wizard: pair media files, or write the notebook that aligns them.

Pages, in order: mode → sources → per-modality folders/patterns → (timing,
modes 2 and 3) → trial table → write. The per-modality page's filename
pattern is optional: leaving it blank pairs by natural sort, one device —
the same answer a single camera would give by hand.

Mode 1 pairs the files in the wizard, writes the session file and the sidecar,
and saves the notebook as the record. Modes 2 and 3 only write the notebook.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.file_dialogs import browse_open_file
from ethograph.gui.notify import notify_dialog
from ethograph.gui.wizard_pages import ModePage, SourcesPage, TimingPage, WritePage
from ethograph.gui.wizard_state import ModalityConfig, WizardState

__all__ = ["ModalityConfig", "NCWizardDialog", "WizardState"]


class NCWizardDialog(QDialog):
    def __init__(self, app_state, io_widget, parent: QWidget | None = None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self._state = WizardState()

        self.setWindowTitle("Data wizard")
        self.setMinimumWidth(950)
        self.setMinimumHeight(750)
        self.resize(1050, 820)

        self._stack = QStackedWidget()
        self._page_mode = ModePage()
        self._page_mode.import_requested.connect(self._open_import)
        self._page_sources = SourcesPage(app_state)
        self._page_timing = TimingPage(app_state)
        self._page_write = WritePage(app_state)
        self._page_patterns = None
        self._page_trials = None
        for page in (self._page_mode, self._page_sources, self._page_timing, self._page_write):
            self._stack.addWidget(page)

        #: The pages this run visits, in order; rebuilt whenever an answer changes it.
        self._route: list[QWidget] = [self._page_mode]
        self._pos = 0

        layout = QVBoxLayout(self)
        layout.addWidget(self._stack)
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Previous")
        self._prev_btn.clicked.connect(self._on_previous)
        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._on_next)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        for btn in (self._prev_btn, self._next_btn, cancel_btn):
            btn.setAutoDefault(False)
            btn.setDefault(False)
        nav.addWidget(self._prev_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        nav.addWidget(cancel_btn)
        layout.addLayout(nav)
        self._show(0)

    # ── navigation ──

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def _show(self, pos: int) -> None:
        self._pos = pos
        page = self._route[pos]
        self._stack.setCurrentWidget(page)
        self._prev_btn.setEnabled(pos > 0)
        last = page is self._page_write
        if last:
            self._next_btn.setText("Pair and load" if self._state.mode == "pair" else "Write notebook")
        else:
            self._next_btn.setText("Next →")

    def _on_previous(self) -> None:
        if self._pos > 0:
            self._show(self._pos - 1)

    def _on_next(self) -> None:
        page = self._route[self._pos]
        err = self._leave(page)
        if err:
            notify_dialog(err, "warning", "Input error", self)
            return
        if page is self._page_write:
            self._finish()
            return
        self._show(self._pos + 1)

    def _leave(self, page: QWidget) -> str | None:
        """Validate + collect the page being left, and extend the route from it."""
        state = self._state
        if page is self._page_mode:
            self._page_mode.collect_state(state)
            self._page_sources.set_mode(state.mode)
            self._route = [self._page_mode, self._page_sources]
            return None
        if page is self._page_sources:
            err = self._page_sources.validate()
            if err:
                return err
            self._page_sources.collect_state(state)
            self._route = [self._page_mode, self._page_sources, self._ensure_patterns_page()]
            if state.mode != "pair":
                self._page_timing.set_mode(state.mode)
                self._route.append(self._page_timing)
            self._route += [self._ensure_trials_page(), self._page_write]
            return None
        if page is self._page_patterns:
            err = self._page_patterns.validate(state)
            if err:
                return err
            self._page_patterns.collect_state(state)
            return self._populate_trials_from_sources()
        if page is self._page_timing:
            err = self._page_timing.validate()
            if err:
                return err
            self._page_timing.collect_state(state)
            return None
        if page is self._page_trials:
            err = self._page_trials.validate(state)
            if err:
                return err
            self._page_trials.collect_state(state)
            self._page_write.populate_from_state(state)
            return None
        if page is self._page_write:
            err = self._page_write.validate(state)
            if err:
                return err
            self._page_write.collect_state(state)
            return None
        return None

    def _ensure_patterns_page(self):
        from ethograph.gui.wizard_multi_tabs import ModalityConfigPage

        if self._page_patterns is not None:
            self._stack.removeWidget(self._page_patterns)
            self._page_patterns.deleteLater()
        self._page_patterns = ModalityConfigPage(self._state)
        self._stack.addWidget(self._page_patterns)
        return self._page_patterns

    def _ensure_trials_page(self):
        from ethograph.gui.wizard_multi_trials import TrialsPage

        if self._page_trials is None:
            self._page_trials = TrialsPage()
            self._stack.addWidget(self._page_trials)
        return self._page_trials

    def _populate_trials_from_sources(self) -> str | None:
        """The pairing table for every enabled stream: sorted folders (one device),
        or a drawn filename pattern (2+ devices) — :func:`discover_media` handles both."""
        from ethograph.io.pairing import discover_media

        state = self._state
        sources = _source_specs(state)
        first = next(c.folder_path for c in (state.video, state.pose, state.audio) if c.enabled and c.folder_path)
        try:
            table = discover_media(state.session_dir or Path(first).parent, sources)
        except ValueError as exc:
            return str(exc)
        self._page_trials.populate_from_table(state, table)
        return None

    # ── imports off page 0 ──

    def _open_import(self, kind: str) -> None:
        if kind == "dandi":
            from ethograph.gui.wizard_nwb import NWBImportDialog

            if NWBImportDialog(self.app_state, self.io_widget, self).exec_():
                self.accept()
        elif kind == "boris":
            from ethograph.gui.wizard_boris import BorisImportDialog

            if BorisImportDialog(self.app_state, self.io_widget, self).exec_():
                self.accept()
        else:
            path = browse_open_file(self, self.app_state, "Open an NWB file", "NWB (*.nwb)")
            if path:
                self.app_state.nc_file_path = path
                self.io_widget.nc_file_path_edit.setText(path)
                self.accept()

    # ── finish ──

    def _finish(self) -> None:
        from ethograph.gui.wizard_notebook import write_notebook

        state = self._state
        notebook = Path(state.notebook_path)
        try:
            write_notebook(_rig_spec(state), notebook)
        except (OSError, ValueError) as exc:
            notify_dialog(f"Could not write the notebook:\n{exc}", "error", "Error", self)
            return

        if state.mode != "pair":
            notify_dialog(
                f"Notebook written:\n{notebook}\n\nRun it; it writes session.nwb next to your media. "
                "Then pick that file on the start page.",
                "info",
                "Notebook written",
                self,
            )
            self.accept()
            return

        from ethograph.gui.dialog_busy_progress import BusyProgressDialog
        from ethograph.gui.wizard_multi_builder import build_multi_trial_dt

        progress = BusyProgressDialog("Pairing files…", parent=self)
        dt, error = progress.execute(build_multi_trial_dt, state)
        if progress.was_cancelled or error:
            if error:
                notify_dialog(f"Pairing failed:\n{error}", "error", "Error", self)
            return
        save_progress = BusyProgressDialog("Saving session file…", parent=self)
        _, save_error = save_progress.execute(dt.to_netcdf, state.output_path)
        if save_error:
            notify_dialog(f"Failed to save:\n{save_error}", "error", "Error", self)
            return

        self.app_state.nwb_alignment = state.nwb_alignment
        self._populate_io_fields()
        notify_dialog(f"Paired and written:\n{state.output_path}\nNotebook: {notebook}", "info", "Done", self)
        self.accept()

    def _populate_io_fields(self) -> None:
        state = self._state
        self.app_state.nc_file_path = state.output_path
        self.io_widget.nc_file_path_edit.setText(state.output_path)
        if state.video.enabled and state.video.folder_path:
            self.app_state.video_folder = state.video.folder_path
            self.io_widget.video_folder_edit.setText(state.video.folder_path)
        if state.audio.enabled and state.audio.folder_path:
            self.app_state.audio_folder = state.audio.folder_path
            if hasattr(self.io_widget, "audio_folder_edit"):
                self.io_widget.audio_folder_edit.setText(state.audio.folder_path)
        if state.pose.enabled and state.pose.folder_path:
            self.app_state.pose_folder = state.pose.folder_path
            self.io_widget.pose_folder_edit.setText(state.pose.folder_path)


# ─── state → library objects ─────────────────────────────────────────────────


def _relative(path: str, session_dir: str) -> str:
    """*path* relative to *session_dir* when it lies inside it, else as given."""
    try:
        return str(Path(path).relative_to(session_dir))
    except ValueError:
        return path


def _source_specs(state: WizardState) -> list:
    from ethograph.io.pairing import SourceSpec

    specs = []
    for stream, cfg in (("video", state.video), ("pose", state.pose), ("audio", state.audio)):
        if not cfg.enabled or cfg.is_continuous_mode:
            continue
        pattern = cfg.pattern.regex_pattern if cfg.pattern is not None and cfg.n_devices > 1 else None
        specs.append(
            SourceSpec(
                stream=stream,
                folder=cfg.folder_path or None,
                files=tuple(cfg.files),
                pattern=pattern,
                software=cfg.source_software if stream == "pose" else None,
            )
        )
    return specs


def _rig_spec(state: WizardState):
    from ethograph.gui.wizard_notebook import RigSource, RigSpec

    session_dir = state.session_dir
    sources = []
    for stream, cfg in (("video", state.video), ("pose", state.pose), ("audio", state.audio)):
        if not cfg.enabled:
            continue
        session_wide = stream == "audio" and cfg.is_continuous_mode
        if session_wide:
            file = cfg.files[0] if cfg.files else cfg.single_file_path
            folder = _relative(file, session_dir)
        else:
            folder = _relative(cfg.folder_path, session_dir)
        sources.append(
            RigSource(
                stream=stream,
                folder=folder,
                pattern=cfg.pattern.regex_pattern if cfg.pattern is not None and cfg.n_devices > 1 else None,
                n_devices=cfg.n_devices,
                rate=cfg.audio_sr if stream == "audio" else None,
                software=cfg.source_software if stream == "pose" else None,
                session_wide=session_wide,
                offset_s=cfg.constant_offset if session_wide else None,
            )
        )
    return RigSpec(
        rig_name=state.rig_name,
        session_dir=session_dir,
        mode=state.mode,
        sources=sources,
        timing=state.timing,
        split_files=state.split_files,
        offset_s=state.offset_s,
        recording_file=_relative(state.recording_file, session_dir) if state.recording_file else None,
        recording_interface=state.recording_interface,
        frame_line=state.frame_line,
        trigger_line=state.trigger_line,
        trial_table_file=state.trial_table_path,
        burst_gap=state.burst_gap,
    )
