"""Refine Pose: the labelling dialog over a project's own labels table.

The refine stage of a DeepLabCut / LightningPose project
(:mod:`ethograph.labels.pose_project`) shows one ``labeled-data/<video>/``
folder per trial, and this dialog — the keypoint labelling dialog reduced to
its **Label & Edit** tab — edits that folder's labels table in place:

- every row of the table is a label (solid, draggable), the frame index
  being the image's position in the folder;
- edits go back to ``CollectedData_<scorer>.csv`` + ``.h5`` (or the root
  ``CollectedData.csv``) a few seconds after the last change, on every folder
  switch, on **Save now** and when the mode closes — never per drag, since
  the ``.h5`` rewrite is not free and a half-written table is worse than a
  stale one;
- static keypoints (:mod:`ethograph.gui.pose_static_group`) are written on
  every frame, and remembered per project so the next folder starts with
  them placed.

Nothing here fills, detects or exports: the frames were chosen in the
extract stage, every one of them is to be reviewed, and the table *is* the
export. Embedded in the left sidebar as a plain widget, not opened as a
window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ethograph.gui.dialog_pose_labelling import PoseLabellingDialog
from ethograph.gui.notify import notify
from ethograph.gui.pose_annotate import KeypointStore
from ethograph.gui.pose_edit_mixin import SEQUENTIAL_MODE
from ethograph.gui.pose_static_group import StaticKeypointsGroup
from ethograph.io.image_sequence import IMAGE_SEQUENCE_RATE, ImageSequence
from ethograph.labels.pose_project import SINGLE_INDIVIDUAL, PoseProject

logger = logging.getLogger(__name__)

#: Seconds after the last edit before the table is rewritten.
AUTOSAVE_MS = 3000
STATIC_TEMPLATE = "static_keypoints.json"


def static_template_path(project: PoseProject) -> Path:
    """Where the project's static keypoints are remembered between folders."""
    return project.root / ".ethograph" / STATIC_TEMPLATE


class CollectedDataDialog(PoseLabellingDialog):
    """Label & Edit over one ``labeled-data/<video>/`` folder's labels table."""

    def __init__(self, data_widget, project: PoseProject, parent=None):
        self.project = project
        self._sequence: ImageSequence | None = None
        self._dirty = False
        self._autosave = QTimer()
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(AUTOSAVE_MS)
        self._autosave.timeout.connect(self.save_now)
        super().__init__(data_widget, parent)
        self.setWindowTitle("Refine pose")
        # A widget in the sidebar, not a window of its own.
        self.setWindowFlags(Qt.Widget)
        self._graft_refine_ui()
        if self._can_label(quiet=True):
            self.set_interaction_mode(SEQUENTIAL_MODE)

    # ------------------------------------------------------------------
    # The folder as a clip
    # ------------------------------------------------------------------

    def _folder(self) -> Path | None:
        video = self._video_path()
        return Path(video) if video and Path(video).is_dir() else None

    def _fps(self) -> float | None:
        return IMAGE_SEQUENCE_RATE

    def _n_frames(self) -> int:
        sequence = self._current_sequence()
        return len(sequence) if sequence is not None else 0

    def _current_sequence(self) -> ImageSequence | None:
        folder = self._folder()
        if folder is None:
            return None
        if self._sequence is None or self._sequence.folder != folder:
            self._sequence = ImageSequence(folder)
        return self._sequence

    def _load_store(self) -> KeypointStore:
        """The folder's labels table as anchors, one frame per image."""
        folder = self._folder()
        sequence = self._current_sequence()
        if folder is None or sequence is None:
            return KeypointStore(keypoint_names=list(self.project.keypoints), n_frames=0)
        table = self.project.labels_table(folder.name)
        keypoints, individuals, _scorer = table.schema()
        store = KeypointStore(
            keypoint_names=keypoints,
            n_frames=len(sequence),
            individual_names=individuals or [SINGLE_INDIVIDUAL],
        )
        for image, positions in table.read().items():
            index = sequence.index_of(image)
            if index is None:
                logger.warning("%s names %s, which is not in the folder", table.path.name, image)
                continue
            for i, individual in enumerate(store.individual_names):
                for k, keypoint in enumerate(keypoints):
                    xy = positions[i, k]
                    if np.all(np.isfinite(xy)):
                        store.set_point(index, keypoint, (float(xy[0]), float(xy[1])), individual)
        store._history.clear()
        self._seed_static(store)
        self._dirty = False
        return store

    def _seed_static(self, store: KeypointStore) -> None:
        """Static keypoints carry over from the last folder of this project."""
        path = static_template_path(self.project)
        if not path.is_file():
            return
        try:
            template = KeypointStore.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, OSError) as e:
            notify(f"Could not read {path.name}: {e}", "warning")
            return
        store.seed_static_from(template)

    def _write_template(self, _video: str | None = None) -> None:
        if not self.store.static_keypoints:
            return
        template = KeypointStore(
            keypoint_names=list(self.store.keypoint_names),
            n_frames=1,
            individual_names=list(self.store.individual_names),
        )
        template.seed_static_from(self.store)
        path = static_template_path(self.project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template.to_dict()), encoding="utf-8")

    def _save_store(self) -> None:
        self.save_now()

    def save_now(self) -> None:
        """Write the folder's labels table now (a no-op with nothing changed)."""
        self._autosave.stop()
        folder = self._folder()
        sequence = self._current_sequence()
        if folder is None or sequence is None or not self._dirty:
            return
        frames = {sequence.name_of(f): self.store.observation_positions(f) for f in range(len(sequence))}
        table = self.project.labels_table(folder.name)
        written = table.write(frames)
        self._write_template()
        self._dirty = False
        logger.info("Wrote %s", ", ".join(p.name for p in written))
        self._refresh_save_state()

    def _switch_clip(self) -> None:
        """Another folder: write this table, load that folder's."""
        video = self._video_path()
        if video == self._store_video:
            return
        if self._store_video:
            self.save_now()
        self._store_video = video
        self.store = self._load_store()
        if self._mode is not None:
            self._mode.store = self.store
        self._rebuild_tree()
        self._on_store_changed(full=True)
        self._refresh_save_state()
        self.static_group.refresh()

    def _lock_wanted(self) -> bool:
        """Only the Lock tick: there is no other tab a click could belong to."""
        return self.lock_check.isChecked()

    def _load_detections(self) -> None:
        return

    def _save_detections(self) -> None:
        return

    def _on_store_changed(self, full: bool = False, frame: int | None = None) -> None:
        super()._on_store_changed(full=full, frame=frame)
        if self._mode is not None and self._mode.dragging:
            return
        self._dirty = True
        self._autosave.start()
        self._refresh_save_state()

    # ------------------------------------------------------------------
    # Static keypoints
    # ------------------------------------------------------------------

    def on_static_changed(self) -> None:
        """The static tag of a keypoint changed in the sidebar group."""
        self._rebuild_tree()
        if self._mode is not None:
            self._mode.refresh()
        self._on_store_changed(full=True)

    def propagate_static_from_current_frame(self) -> int:
        """Place every static keypoint where it sits on the frame on screen; returns how many."""
        frame = int(self.app_state.current_frame or 0)
        observed = self.store.observation_positions(frame)
        placed = 0
        for name in list(self.store.static_keypoints):
            k = self.store.keypoint_index(name)
            for i, individual in enumerate(self.store.individual_names):
                xy = observed[i, k]
                if np.all(np.isfinite(xy)):
                    self.store.set_point(frame, name, (float(xy[0]), float(xy[1])), individual)
                    placed += 1
        if placed:
            if self._mode is not None:
                self._mode.refresh()
            self._on_store_changed(full=True)
        return placed

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _graft_refine_ui(self) -> None:
        """Only Label & Edit stays, unwrapped; fill, detect, export and clips are not this stage's.

        Taken *out* of the layout rather than hidden: an emptied row still
        costs its spacing, and the tab widget its frame — in a sidebar column
        that was a band of nothing between the mode buttons and the table.
        """
        outer = self.layout()
        assert isinstance(outer, QVBoxLayout)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        while outer.count():
            item = outer.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
            layout = item.layout()
            if layout is not None:
                _clear_layout(layout)
        self.tabs.hide()
        self.tabs.setParent(None)

        page = self._label_page
        page_layout = page.layout()
        assert isinstance(page_layout, QVBoxLayout)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(4)
        for name in ("approve_detections_btn", "approve_fill_btn"):
            getattr(self, name).hide()
        _remove_from_layout(page_layout, self.clips_label)
        _remove_from_layout(page_layout, self.next_clip_btn)
        suggest_group = self.suggest_method_combo.parentWidget()
        if suggest_group is not None:
            _remove_from_layout(page_layout, suggest_group)
        outer.addWidget(page, stretch=1)

        # Static landmarks are this stage's own business, so the group sits
        # between the mode buttons and the table, not in the Data tab.
        self.static_group = StaticKeypointsGroup()
        self.static_group.bind(self)
        page_layout.insertWidget(1, self.static_group)

        row = QHBoxLayout()
        self.save_btn = QPushButton("Save labels now")
        self.save_btn.setToolTip(
            "Write this folder's labels table now. It is also written a few\n"
            "seconds after every edit, when you move to another folder, and on close."
        )
        self.save_btn.clicked.connect(self.save_now)
        row.addWidget(self.save_btn)
        self.save_state = QLabel("")
        self.save_state.setStyleSheet("color: rgba(255,255,255,150);")
        row.addWidget(self.save_state, stretch=1)
        self.check_btn = QPushButton("Check labels")
        self.check_btn.setToolTip(
            "Write every frame of this folder with its labels drawn on, into\n"
            "labeled-data/<video>_labeled/ — DeepLabCut's check_labels — so the\n"
            "reviewed poses can be looked over in any image viewer."
        )
        self.check_btn.clicked.connect(self.check_labels)
        row.addWidget(self.check_btn)
        outer.addLayout(row)

        self.next_curates_check = QCheckBox("Going to the next folder marks this one curated")
        self.next_curates_check.setToolTip(
            "Every frame of a folder is meant to be reviewed before moving on,\n"
            "so leaving it counts as approving it. Untick to curate by hand (Ctrl+C)."
        )
        self.next_curates_check.setChecked(bool(self.app_state.pose_refine_next_curates))
        self.next_curates_check.toggled.connect(
            lambda checked: setattr(self.app_state, "pose_refine_next_curates", bool(checked))
        )
        outer.addWidget(self.next_curates_check)
        self._refresh_save_state()

    def check_labels(self) -> Path | None:
        """Render this folder's labels into ``<folder>_labeled/`` (after writing the table)."""
        from ethograph.gui.dialog_busy_progress import BusyProgressDialog
        from ethograph.labels.check_labels import check_labels

        folder = self._folder()
        if folder is None:
            return None
        self.save_now()
        dialog = BusyProgressDialog(f"Drawing labels onto {folder.name}…", parent=self._shell)
        output, error = dialog.execute_blocking(check_labels, self.project, folder.name)
        if error is not None:
            notify(f"Check labels failed: {error}", "warning")
            return None
        notify(f"Labelled frames written to {output.parent.name}/{output.name}/", "info")
        return output

    def _refresh_save_state(self) -> None:
        if not hasattr(self, "save_state"):
            return
        folder = self._folder()
        if folder is None:
            self.save_state.setText("")
            return
        table = self.project.labels_table(folder.name).path.name
        self.save_state.setText(f"{table}: {'unsaved edits' if self._dirty else 'saved'}")

    def closeEvent(self, event):
        self.save_now()
        super().closeEvent(event)


def _clear_layout(layout) -> None:
    """Take every item out of *layout*, hiding the widgets, so its spacing goes too."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
        child = item.layout()
        if child is not None:
            _clear_layout(child)


def _remove_from_layout(layout, widget) -> None:
    """Remove *widget* from *layout*, wherever it sits — directly or in a nested row."""
    widget.hide()
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        if item.widget() is widget:
            layout.takeAt(i)
            widget.setParent(None)
            return
        child = item.layout()
        if child is not None:
            _remove_from_layout(child, widget)
            if child.count() == 0:
                layout.takeAt(i)
            return
