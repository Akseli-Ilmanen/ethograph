"""Wizard for importing a BORIS project as an ethograph project.

Writes:
    <output>/.ethograph/alignment.nwb   — trials + media, one trial per BORIS file
    <output>/.ethograph/mapping.txt     — behavior name -> int id
    <output>/boris_labels.tsv           — intervals per trial

Optionally wires pose files into the alignment NWB by basename-matching
a user-supplied folder against the BORIS video files.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ethograph.gui.notify import notify_dialog
from ethograph.io.pairing import pair_media
from ethograph.labels.boris import (
    behavior_event_types,
    build_trial_table,
    extract_intervals,
    list_media_observations,
    load_boris_project,
    match_pose_files,
    observation_media_files,
    resolve_media_paths,
    unique_behavior_codes,
)
from ethograph.labels.converters import build_mapping_from_labels
from ethograph.labels.intervals import EVENT_TYPE_STATE, save_label_mapping
from ethograph.labels.tsv_store import TSV_COLUMNS

logger = logging.getLogger(__name__)


class BorisImportDialog(QDialog):
    """Convert a BORIS ``.boris`` project into an ethograph project folder."""

    def __init__(self, app_state, io_widget, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Import BORIS project")
        self.setMinimumWidth(700)
        self._project: dict | None = None
        self._boris_path: Path | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "<b>Convert a BORIS project to ethograph</b><br>"
                "One ethograph trial = one media file in Player 1. "
                "Events are split across file boundaries into trial-local intervals."
            )
        )
        layout.addSpacing(6)

        form = QFormLayout()

        self._boris_edit, row_boris = self._file_row(
            "Select .boris project file…",
            self._on_browse_boris,
        )
        form.addRow("BORIS project:", row_boris)

        self._obs_combo = QComboBox()
        self._obs_combo.setEnabled(False)
        form.addRow("Observation:", self._obs_combo)

        self._out_edit, row_out = self._folder_row(
            "Defaults to parent folder of .boris",
            self._on_browse_output,
        )
        form.addRow("Output folder:", row_out)

        self._video_edit, row_vid = self._folder_row(
            "Optional — folder with the videos if not next to .boris",
            self._on_browse_video,
        )
        form.addRow("Video folder:", row_vid)

        self._pose_edit, row_pose = self._folder_row(
            "Optional — pose files auto-matched by filename stem",
            self._on_browse_pose,
        )
        form.addRow("Pose folder:", row_pose)

        layout.addLayout(form)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #888; padding-top: 6px;")
        layout.addWidget(self._summary)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self._import_btn = QPushButton("Import")
        self._import_btn.clicked.connect(self._on_import)
        self._import_btn.setEnabled(False)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._import_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def _file_row(placeholder: str, browse_cb):
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("Browse…")
        btn.clicked.connect(browse_cb)
        row.addWidget(edit)
        row.addWidget(btn)
        return edit, row

    @staticmethod
    def _folder_row(placeholder: str, browse_cb):
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("Browse…")
        btn.clicked.connect(browse_cb)
        row.addWidget(edit)
        row.addWidget(btn)
        return edit, row

    def _on_browse_boris(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select BORIS project",
            "",
            "BORIS project (*.boris *.boris.gz);;All files (*)",
        )
        if not path:
            return
        self._boris_edit.setText(path)
        self._boris_path = Path(path)
        if not self._out_edit.text():
            self._out_edit.setText(str(self._boris_path.parent))
        self._load_boris()

    def _on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Output folder")
        if folder:
            self._out_edit.setText(folder)

    def _on_browse_video(self):
        folder = QFileDialog.getExistingDirectory(self, "Video folder")
        if folder:
            self._video_edit.setText(folder)

    def _on_browse_pose(self):
        folder = QFileDialog.getExistingDirectory(self, "Pose folder (optional)")
        if folder:
            self._pose_edit.setText(folder)

    def _load_boris(self):
        try:
            self._project = load_boris_project(self._boris_path)
        except Exception as exc:
            notify_dialog(f"Failed to parse BORIS project:\n{exc}", "error", "Error", self)
            return

        obs_keys = list_media_observations(self._project)
        self._obs_combo.clear()
        self._obs_combo.addItems(obs_keys)
        self._obs_combo.setEnabled(bool(obs_keys))
        self._import_btn.setEnabled(bool(obs_keys))

        n_beh = len(unique_behavior_codes(self._project))
        if obs_keys:
            first = self._project["observations"][obs_keys[0]]
            n_files = len(observation_media_files(first))
            n_events = len(first.get("events", []))
            self._summary.setText(
                f"<b>{len(obs_keys)}</b> MEDIA observation(s), "
                f"<b>{n_beh}</b> behaviors. "
                f"First observation: {n_files} media file(s), {n_events} event rows."
            )
        else:
            self._summary.setText(f"{n_beh} behaviors but no MEDIA observations — nothing to import.")

    def _on_import(self):
        try:
            self._do_import()
        except Exception as exc:
            logger.exception("BORIS import failed")
            notify_dialog(
                f"Import failed:\n{exc}\n\n{traceback.format_exc()}",
                "error",
                "Import error",
                self,
            )
            return

        notify_dialog(
            f"Imported BORIS project to:\n{self._out_edit.text()}",
            "info",
            "Success",
            self,
        )
        self.accept()

    def _do_import(self) -> None:
        if self._project is None or self._boris_path is None:
            raise ValueError("No BORIS project loaded")

        out_dir = Path(self._out_edit.text() or self._boris_path.parent)
        obs_key = self._obs_combo.currentText()
        video_folder = self._video_edit.text() or None
        pose_folder = self._pose_edit.text() or None

        out_dir.mkdir(parents=True, exist_ok=True)
        ethograph_dir = out_dir / ".ethograph"
        ethograph_dir.mkdir(exist_ok=True)

        observation = self._project["observations"][obs_key]

        label_names = unique_behavior_codes(self._project)
        if not label_names:
            raise ValueError("BORIS project declares no behaviors")
        name_to_id = build_mapping_from_labels(label_names)
        event_types = behavior_event_types(self._project)
        mapping_path = ethograph_dir / "mapping.txt"
        save_label_mapping(
            mapping_path,
            {
                label_id: {
                    "name": name,
                    "branch": 0,
                    "event_type": event_types.get(name, EVENT_TYPE_STATE),
                }
                for name, label_id in name_to_id.items()
            },
        )

        trials_df = build_trial_table(observation)

        fps_info = observation.get("media_info", {}).get("fps", {})
        rates = {float(v) for v in fps_info.values() if v}
        if not rates:
            raise ValueError("BORIS media_info.fps is empty; cannot write alignment.nwb without a video frame rate")
        if len(rates) > 1:
            raise ValueError(
                f"Mixed FPS across media files ({sorted(rates)}); normalize the "
                "videos to a single rate or open them as separate observations"
            )
        rate = next(iter(rates))

        search_dirs = [self._boris_path.parent, out_dir]
        if video_folder:
            search_dirs.insert(0, Path(video_folder))
        video_paths = resolve_media_paths(observation, search_dirs)

        pairing = trials_df[["trial", "start_time", "stop_time"]].assign(**{"video_cam-1": video_paths})
        stream_rates = {"video": rate}

        if pose_folder:
            pose_paths = match_pose_files(Path(pose_folder), video_paths)
            if pose_paths:
                pairing["pose_cam-1"] = pose_paths
                stream_rates["pose"] = rate
            else:
                logger.warning("No pose files matched any video in %s", pose_folder)

        nwb_out = ethograph_dir / "alignment.nwb"
        pair_media(pairing, stream_rates=stream_rates, output_path=nwb_out)

        labels_df = extract_intervals(observation, name_to_id)
        keep_cols = [c for c in TSV_COLUMNS if c in labels_df.columns]
        labels_df = labels_df[keep_cols]
        labels_path = out_dir / "boris_labels.tsv"
        labels_df.to_csv(labels_path, sep="\t", index=False, encoding="utf-8-sig")

        logger.info(
            "BORIS import: %s -> %s  (alignment=%s, mapping=%s, labels=%s)",
            self._boris_path,
            out_dir,
            nwb_out,
            mapping_path,
            labels_path,
        )
