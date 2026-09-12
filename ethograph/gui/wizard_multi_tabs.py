"""Per-modality configuration tabs for the NC creation wizard (Page 2)."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.wizard_media_files import StreamPanel, extract_file_row
from ethograph.gui.wizard_pages import _muted
from ethograph.gui.wizard_state import AVAILABLE_SOFTWARES, ModalityConfig, WizardState
from ethograph.io.validation import (
    AUDIO_FILE_FILTER,
    EPHYS_FILE_FILTER,
    VIDEO_EXTENSIONS,
    VIDEO_FILE_FILTER,
)


def _pattern_device_count(pattern, device_key: str) -> int:
    """How many distinct devices *pattern* named, or 1 when it named none
    (no pattern drawn at all, or a pattern with no ``camera``/``mic`` group —
    every file is then one device, matching ``pairing.discover_media``'s
    pattern-less natural-sort behaviour)."""
    if pattern is None:
        return 1
    devices = pattern.summary().get(device_key, [])
    return len(devices) if devices else 1


# ─── base tab ─────────────────────────────────────────────────────────────────


class _BaseConfigTab(QWidget):
    """Base for modality config tabs with single-file vs folder support."""

    def __init__(self, config: ModalityConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        self._is_multi = config.file_mode in ("aligned_to_trial", "aligned_to_session")
        self._is_irregular = config.file_mode == "aligned_to_session"

    def _add_offset_row(self, form: QFormLayout) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-100000.0, 100000.0)
        spin.setDecimals(4)
        spin.setValue(0.0)
        spin.setSuffix(" s")
        spin.setToolTip("Constant time offset for this stream (seconds)")
        form.addRow("Constant offset:", spin)
        return spin

    def _mode_title(self, aligned_label: str, continuous_label: str) -> str:
        if self._is_irregular:
            return continuous_label
        if self._is_multi:
            return aligned_label
        return "Single file"

    def _add_file_browse(self, form: QFormLayout, label: str, placeholder: str, file_filter: str) -> QLineEdit:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setReadOnly(True)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._do_file_browse(edit, file_filter))
        row.addWidget(edit)
        row.addWidget(browse)
        form.addRow(label, container)
        return edit

    def _add_folder_browse(self, form: QFormLayout, label: str) -> QLineEdit:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText("Select folder...")
        edit.setReadOnly(True)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._do_folder_browse(edit))
        row.addWidget(edit)
        row.addWidget(browse)
        form.addRow(label, container)
        return edit

    def _do_file_browse(self, edit: QLineEdit, file_filter: str):
        result = QFileDialog.getOpenFileName(self, "Select file", "", file_filter)
        if result and result[0]:
            edit.setText(result[0])
            self._on_file_selected(result[0])

    def _do_folder_browse(self, edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            edit.setText(folder)

    def _on_file_selected(self, path: str):
        pass

    # ── required-field helpers ────────────────────────────────────────────
    _REQUIRED_STYLE = "border: 1.5px solid #e8a020;"

    def _mark_required(self, spin: QSpinBox | QDoubleSpinBox) -> None:
        """Visually flag a spinbox that still needs explicit user input."""
        spin.setProperty("_required", True)
        spin.setStyleSheet(self._REQUIRED_STYLE)
        spin.setToolTip("⚠ Please set this value")

    def _clear_required(self, spin: QSpinBox | QDoubleSpinBox) -> None:
        spin.setProperty("_required", False)
        spin.setStyleSheet("")

    @staticmethod
    def _spin_is_required(spin: QSpinBox | QDoubleSpinBox) -> bool:
        return bool(spin.property("_required"))

    def _collect_offsets(
        self,
        config: ModalityConfig,
        *,
        constant_checkbox: QCheckBox | None,
        constant_spin: QDoubleSpinBox | None,
        device_spins: dict[str, QDoubleSpinBox],
    ) -> None:
        if constant_checkbox is None or constant_spin is None:
            config.offset_constant_across_devices = True
            config.constant_offset = 0.0
            config.device_offsets = {}
            return
        config.offset_constant_across_devices = constant_checkbox.isChecked()
        config.constant_offset = float(constant_spin.value()) if constant_checkbox.isChecked() else 0.0
        config.device_offsets = {device: float(spin.value()) for device, spin in device_spins.items()}

    def _validate_one_file_per_device(self, role: str, label: str) -> str | None:
        stream_panel = getattr(self, "_stream_panel", None)
        if stream_panel is None or stream_panel.pattern is None:
            return None

        pattern = stream_panel.pattern
        rows = [
            extract_file_row(
                f,
                pattern.segments,
                pattern.tokenize_mode,
                regex_pattern=pattern.regex_pattern,
            )
            for f in pattern.files
        ]
        seen: set[str] = set()
        for row in rows:
            device = str(row.get(role, "")).strip()
            if not device:
                continue
            if device in seen:
                return f"{label}: continuous mode allows exactly one file per {role}. Duplicate '{device}' found."
            seen.add(device)
        return None

    def collect_state(self, config: ModalityConfig):
        raise NotImplementedError

    def validate(self) -> str | None:
        return None


# ─── Video tab ────────────────────────────────────────────────────────────────


class VideoConfigTab(_BaseConfigTab):
    fps_changed = Signal()

    def __init__(self, config: ModalityConfig, parent: QWidget | None = None):
        super().__init__(config, parent)
        self._auto_detecting_fps = False
        self._detected_fps_by_camera: dict[str, int] = {}
        self._fps_by_camera_spins: dict[str, QSpinBox] = {}
        self._camera_offset_spins: dict[str, QDoubleSpinBox] = {}
        layout = QVBoxLayout(self)
        form = QFormLayout()

        mode_label = QLabel(
            f"<b>Mode:</b> {self._mode_title('Files aligned to trial period (Trials x Cameras)', 'Continuous recording across session (Cameras)')}"  # noqa: E501
        )
        layout.addWidget(mode_label)

        if self._is_multi:
            _video_roles = [r for r in ["ignore", "camera"] if True] if self._is_irregular else None
            self._stream_panel = StreamPanel("video", allowed_roles=_video_roles)
            self._stream_panel.set_folder(config.folder_path)
            self._stream_panel.changed.connect(self._detect_and_set_fps_from_folder)
            self._stream_panel.changed.connect(self._refresh_camera_offset_controls)
            if self._is_irregular:
                layout.addWidget(QLabel("<b>Session video files (one per camera)</b>"))
            else:
                layout.addWidget(QLabel("<b>Trial-aligned video files</b>"))
            layout.addWidget(self._stream_panel, stretch=1)
            self._file_edit = None
        else:
            self._stream_panel = None
            self._file_edit = self._add_file_browse(
                form,
                "Video file:",
                "Select video file...",
                VIDEO_FILE_FILTER,
            )

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 1000)
        self._fps_spin.setValue(config.fps if config.fps is not None else 30)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setToolTip("Auto-detected from selected video files")
        if config.fps is None:
            self._mark_required(self._fps_spin)
        self._fps_spin.valueChanged.connect(self._on_fps_manually_changed)
        self._fps_spin.valueChanged.connect(lambda _v: self.fps_changed.emit())

        # FPS row with spinner + inline note
        fps_container = QWidget()
        fps_row = QHBoxLayout(fps_container)
        fps_row.setContentsMargins(0, 0, 0, 0)
        fps_row.addWidget(self._fps_spin)
        fps_row.addStretch()
        form.addRow("Frame rate:", fps_container)

        self._per_camera_fps_box = QGroupBox("Per-camera FPS")
        self._per_camera_fps_form = QFormLayout(self._per_camera_fps_box)
        self._per_camera_fps_box.setVisible(False)
        form.addRow(self._per_camera_fps_box)

        self._motion_cb = QCheckBox("Extract video motion features")
        form.addRow("", self._motion_cb)

        # Offset controls (only in continuous mode)
        self._const_offset_cb: QCheckBox | None = None
        self._const_offset_spin: QDoubleSpinBox | None = None
        self._per_camera_offset_box: QGroupBox | None = None
        self._per_camera_offset_form: QFormLayout | None = None

        if self._is_irregular:
            self._const_offset_cb = QCheckBox("Offset is constant across cameras")
            self._const_offset_cb.setChecked(config.offset_constant_across_devices)
            self._const_offset_cb.toggled.connect(self._on_const_offset_toggled)
            form.addRow("", self._const_offset_cb)

            self._const_offset_spin = QDoubleSpinBox()
            self._const_offset_spin.setRange(-100000.0, 100000.0)
            self._const_offset_spin.setDecimals(4)
            self._const_offset_spin.setValue(config.constant_offset or 0.0)
            self._const_offset_spin.setSuffix(" s")
            self._const_offset_spin.setToolTip("Constant time offset for all cameras (seconds)")
            form.addRow("Constant offset:", self._const_offset_spin)

            self._per_camera_offset_box = QGroupBox("Per-camera offsets")
            self._per_camera_offset_form = QFormLayout(self._per_camera_offset_box)
            self._per_camera_offset_box.setVisible(not config.offset_constant_across_devices)
            form.addRow(self._per_camera_offset_box)

        layout.addLayout(form)

    def _on_file_selected(self, path: str):
        from ethograph.gui.wizard_single import get_video_fps

        fps = get_video_fps(path)
        if fps is not None:
            self._auto_detecting_fps = True
            self._fps_spin.setValue(fps)
            self._auto_detecting_fps = False

    def _on_fps_manually_changed(self):
        """Reset styling when user manually changes FPS."""
        if not self._auto_detecting_fps:
            self._clear_required(self._fps_spin)
            self._fps_spin.setToolTip("Manually set frame rate")

    def _detect_and_set_fps_from_folder(self):
        """Auto-detect FPS from video files in the selected folder."""
        if not self._stream_panel:
            return

        pat = self._stream_panel.pattern
        if not pat or not pat.files:
            return

        fps_values = list(self._refresh_detected_video_fps().values())
        if not fps_values:
            return

        unique_fps = set(fps_values)
        if len(unique_fps) == 1:
            detected_fps = fps_values[0]
            self._auto_detecting_fps = True
            self._fps_spin.setValue(detected_fps)
            self._auto_detecting_fps = False
            self._clear_required(self._fps_spin)
            self._fps_spin.setStyleSheet("color: #50c8b4; font-weight: bold;")
        else:
            self._clear_required(self._fps_spin)
            self._fps_spin.setStyleSheet("color: #e8737a;")
            self._fps_spin.setToolTip("Mixed camera FPS detected; configure per-camera FPS below")

    def _refresh_detected_video_fps(self) -> dict[str, int]:
        from ethograph.gui.wizard_single import get_video_fps

        if self._stream_panel is None or self._stream_panel.pattern is None:
            self._detected_fps_by_camera = {}
            return self._detected_fps_by_camera

        pat = self._stream_panel.pattern
        fps_by_camera: dict[str, list[int]] = {}
        for file_path in pat.files:
            if Path(file_path).suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            fps = get_video_fps(str(file_path))
            if fps is None:
                continue
            row = extract_file_row(
                file_path,
                pat.segments,
                pat.tokenize_mode,
                regex_pattern=pat.regex_pattern,
            )
            camera = row.get("camera", "camera_1")
            fps_by_camera.setdefault(camera, []).append(int(round(fps)))

        detected: dict[str, int] = {}
        for camera, values in fps_by_camera.items():
            if values:
                detected[camera] = values[0]
        self._detected_fps_by_camera = detected
        return detected

    def set_camera_fps_controls(self, camera_names: list[str]):
        while self._per_camera_fps_form.rowCount() > 0:
            self._per_camera_fps_form.removeRow(0)
        self._fps_by_camera_spins.clear()

        if len(camera_names) <= 1:
            self._per_camera_fps_box.setVisible(False)
            return

        self._per_camera_fps_box.setVisible(True)
        for camera in camera_names:
            spin = QSpinBox()
            spin.setRange(1, 1000)
            saved_fps = self._config.fps_by_camera.get(camera)
            detected_fps = self._detected_fps_by_camera.get(camera)
            spin.setValue(
                saved_fps
                if saved_fps is not None
                else (detected_fps if detected_fps is not None else self._fps_spin.value())
            )
            spin.setSuffix(" fps")
            spin.valueChanged.connect(lambda _v: self.fps_changed.emit())
            self._per_camera_fps_form.addRow(f"{camera}:", spin)
            self._fps_by_camera_spins[camera] = spin

    def set_camera_offset_controls(self, camera_names: list[str]):
        if not self._per_camera_offset_form:
            return
        while self._per_camera_offset_form.rowCount() > 0:
            self._per_camera_offset_form.removeRow(0)
        self._camera_offset_spins.clear()

        if len(camera_names) <= 1:
            self._per_camera_offset_box.setVisible(False)
            return

        self._per_camera_offset_box.setVisible(not (self._const_offset_cb and self._const_offset_cb.isChecked()))
        for camera in camera_names:
            spin = QDoubleSpinBox()
            spin.setRange(-100000.0, 100000.0)
            spin.setDecimals(4)
            saved_offset = self._config.device_offsets.get(camera, 0.0)
            spin.setValue(saved_offset)
            spin.setSuffix(" s")
            self._per_camera_offset_form.addRow(f"{camera}:", spin)
            self._camera_offset_spins[camera] = spin

    def _on_const_offset_toggled(self, checked: bool):
        if not self._const_offset_spin or not self._per_camera_offset_box:
            return
        is_constant = self._const_offset_cb and self._const_offset_cb.isChecked()
        self._const_offset_spin.setVisible(is_constant)
        self._per_camera_offset_box.setVisible(not is_constant and len(self._camera_offset_spins) > 1)

    def _refresh_camera_offset_controls(self):
        if not self._stream_panel or not self._stream_panel.pattern:
            return

        pat = self._stream_panel.pattern
        cameras: set[str] = set()
        for file_path in pat.files:
            row = extract_file_row(
                file_path,
                pat.segments,
                pat.tokenize_mode,
                regex_pattern=pat.regex_pattern,
            )
            camera = row.get("camera", "camera_1")
            cameras.add(camera)

        self.set_camera_offset_controls(sorted(cameras))

    def get_detected_fps_by_camera(self) -> dict[str, int]:
        return dict(self._detected_fps_by_camera)

    def collect_state(self, config: ModalityConfig):
        config.fps = None if self._spin_is_required(self._fps_spin) else self._fps_spin.value()
        config.fps_by_camera = {camera: spin.value() for camera, spin in self._fps_by_camera_spins.items()}
        config.video_motion = self._motion_cb.isChecked()
        if self._is_irregular:
            self._collect_offsets(
                config,
                constant_checkbox=self._const_offset_cb,
                constant_spin=self._const_offset_spin,
                device_spins=self._camera_offset_spins,
            )
        if self._is_multi and self._stream_panel:
            sc = self._stream_panel.get_config()
            # No upfront checkbox any more: a tab with nothing dropped in it
            # simply isn't enabled — the presence of a folder is the answer.
            config.enabled = bool(sc)
            if sc:
                config.folder_path = sc.folder
                config.nested_subfolders = sc.nested
            config.pattern = self._stream_panel.pattern
            config.n_devices = _pattern_device_count(config.pattern, "camera")
        elif self._file_edit:
            config.enabled = bool(self._file_edit.text())
            config.single_file_path = self._file_edit.text()

    def validate(self) -> str | None:
        # An empty tab is fine — it's just not enabled (see collect_state).
        # A folder that resolved to no files at all is a mistake, not that.
        if self._is_multi:
            if self._stream_panel and self._stream_panel.has_unresolved_folder():
                return "Video: no video files found in that folder."
        elif self._file_edit and not self._file_edit.text():
            return "Video: select a video file."
        return None


# ─── Pose tab ─────────────────────────────────────────────────────────────────


class PoseConfigTab(_BaseConfigTab):
    def __init__(self, config: ModalityConfig, parent: QWidget | None = None):
        super().__init__(config, parent)
        #: Whether the Video tab currently has a folder — every tab now
        #: exists unconditionally, so this can't be known at construction; it
        #: starts pessimistic (no video) and is corrected live by
        #: ``ModalityConfigPage`` via :meth:`set_has_video`.
        self._has_video = False
        self._pose_fps_by_camera_spins: dict[str, QSpinBox] = {}
        self._pose_offset_spins: dict[str, QDoubleSpinBox] = {}
        self._pose_to_video_map: dict[str, str] = {}
        self._video_fps_by_camera: dict[str, int] = {}
        self._match_with_camera_cb: QCheckBox | None = None
        self._const_offset_cb: QCheckBox | None = None
        self._const_offset_spin: QDoubleSpinBox | None = None
        self._per_pose_offset_box: QGroupBox | None = None
        self._per_pose_offset_form: QFormLayout | None = None
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        form = QFormLayout()

        mode_label = QLabel(
            f"<b>Mode:</b> {self._mode_title('Files aligned to trial period (Trials x Cameras)', 'Continuous recording across session (Cameras)')}"  # noqa: E501
        )
        layout.addWidget(mode_label)

        self._software_combo = QComboBox()
        self._software_combo.addItems(AVAILABLE_SOFTWARES)
        if config.source_software in AVAILABLE_SOFTWARES:
            self._software_combo.setCurrentText(config.source_software)
        form.addRow("Source software:", self._software_combo)

        # A pose file carries no frame rate of its own; without a video to
        # read it from, it must be given explicitly (never defaulted). Built
        # unconditionally and shown/required by ``set_has_video``, since
        # whether the Video tab has anything in it can change live.
        self._no_video_fps_spin = QDoubleSpinBox()
        self._no_video_fps_spin.setRange(0.0, 100000.0)
        self._no_video_fps_spin.setDecimals(3)
        self._no_video_fps_spin.setSuffix(" fps")
        self._no_video_fps_spin.setValue(config.fps if config.fps is not None else 0.0)
        self._no_video_fps_spin.setToolTip("The camera fps the pose was tracked at (no video to read it from)")
        if config.fps is None:
            self._mark_required(self._no_video_fps_spin)
        self._no_video_fps_spin.valueChanged.connect(lambda _: self._clear_required(self._no_video_fps_spin))
        form.addRow("Frame rate:", self._no_video_fps_spin)
        self._pose_form = form

        if self._is_multi:
            layout.addLayout(form)
            _pose_roles = ["ignore", "camera"] if self._is_irregular else None
            self._stream_panel = StreamPanel("pose", allowed_roles=_pose_roles)
            self._stream_panel.setMinimumHeight(350)
            self._stream_panel.set_folder(config.folder_path)
            self._stream_panel.changed.connect(self._on_pose_pattern_changed)
            self._stream_panel.changed.connect(self._refresh_pose_offset_controls)
            if self._is_irregular:
                layout.addWidget(QLabel("<b>Session pose files (one per camera)</b>"))
            else:
                layout.addWidget(QLabel("<b>Trial-aligned pose files</b>"))
            layout.addWidget(self._stream_panel)
            self._file_edit = None
        else:
            self._stream_panel = None
            self._file_edit = self._add_file_browse(
                form,
                "Pose file:",
                "Select pose file...",
                "Pose files (*.h5 *.hdf5 *.csv *.slp *.nwb);;All files (*)",
            )
            layout.addLayout(form)

        form2 = QFormLayout()

        # Camera-pose matching table
        layout.addWidget(QLabel("<b>Camera ↔ Pose matching</b>"))
        layout.addWidget(
            QLabel("Map pose file stems to camera names. Use 'Add None' for cameras without pose or vice versa.")
        )
        from ethograph.gui.dialog_pose_video_matcher import (
            PoseVideoMatcherWidget,
        )

        self._matcher = PoseVideoMatcherWidget()
        layout.addWidget(self._matcher)

        match_row = QHBoxLayout()
        add_none_pose_btn = QPushButton("Add None to pose")
        add_none_pose_btn.clicked.connect(self._add_none_pose_row)
        clear_none_pose_btn = QPushButton("Clear None from pose")
        clear_none_pose_btn.clicked.connect(self._clear_none_pose_rows)
        add_none_video_btn = QPushButton("Add None to video")
        add_none_video_btn.clicked.connect(self._add_none_video_row)
        clear_none_video_btn = QPushButton("Clear None from video")
        clear_none_video_btn.clicked.connect(self._clear_none_video_rows)
        match_row.addWidget(add_none_pose_btn)
        match_row.addWidget(clear_none_pose_btn)
        match_row.addWidget(add_none_video_btn)
        match_row.addWidget(clear_none_video_btn)
        match_row.addStretch()
        layout.addLayout(match_row)

        # Pose FPS section (below matching)
        layout.addWidget(QLabel("<b>Pose FPS</b>"))
        self._match_with_camera_cb = QCheckBox("Match with camera")
        self._match_with_camera_cb.setChecked(self._has_video)
        # In aligned mode, always match with camera (disable user control)
        # In continuous mode, allow manual choice
        self._match_with_camera_cb.setEnabled(self._has_video and self._is_irregular)
        self._match_with_camera_cb.toggled.connect(self._on_match_with_camera_changed)
        layout.addWidget(self._match_with_camera_cb)

        self._per_camera_pose_fps_box = QGroupBox("Per-pose-file FPS")
        self._per_camera_pose_fps_form = QFormLayout(self._per_camera_pose_fps_box)
        self._per_camera_pose_fps_box.setVisible(False)
        layout.addWidget(self._per_camera_pose_fps_box)

        # Offset controls (only in continuous mode)
        if self._is_irregular:
            layout.addWidget(QLabel("<b>Pose offsets</b>"))
            self._const_offset_cb = QCheckBox("Offset is constant across poses")
            self._const_offset_cb.setChecked(config.offset_constant_across_devices)
            self._const_offset_cb.toggled.connect(self._on_const_offset_toggled)
            form2.addRow("", self._const_offset_cb)

            self._const_offset_spin = QDoubleSpinBox()
            self._const_offset_spin.setRange(-100000.0, 100000.0)
            self._const_offset_spin.setDecimals(4)
            self._const_offset_spin.setValue(config.constant_offset or 0.0)
            self._const_offset_spin.setSuffix(" s")
            self._const_offset_spin.setToolTip("Constant time offset for all poses (seconds)")
            form2.addRow("Constant offset:", self._const_offset_spin)

            self._per_pose_offset_box = QGroupBox("Per-pose offsets")
            self._per_pose_offset_form = QFormLayout(self._per_pose_offset_box)
            self._per_pose_offset_box.setVisible(not config.offset_constant_across_devices)
            form2.addRow(self._per_pose_offset_box)

        layout.addLayout(form2)
        layout.addStretch()

        scroll.setWidget(inner)
        outer_layout.addWidget(scroll)

    def _on_pose_pattern_changed(self):
        pat = self._stream_panel.pattern if self._stream_panel else None
        if not pat:
            return
        summary = pat.summary()
        cameras = summary.get("camera", [])
        if cameras:
            self._matcher._pose_list.set_items(cameras)
        else:
            stems = sorted({f.stem for f in pat.files})
            self._matcher._pose_list.set_items(stems)
        self._matcher.mapping_changed.emit(self._matcher.get_mapping())

    def _add_none_pose_row(self):
        current = self._matcher._pose_list.get_items()
        current.append("None")
        self._matcher._pose_list.set_items(current)
        self._matcher.mapping_changed.emit(self._matcher.get_mapping())

    def _clear_none_pose_rows(self):
        current = self._matcher._pose_list.get_items()
        filtered = [item for item in current if item != "None"]
        self._matcher._pose_list.set_items(filtered)
        self._matcher.mapping_changed.emit(self._matcher.get_mapping())

    def _add_none_video_row(self):
        current = self._matcher._video_list.get_items()
        current.append("None")
        self._matcher._video_list.set_items(current)
        self._matcher.mapping_changed.emit(self._matcher.get_mapping())

    def _clear_none_video_rows(self):
        current = self._matcher._video_list.get_items()
        filtered = [item for item in current if item != "None"]
        self._matcher._video_list.set_items(filtered)
        self._matcher.mapping_changed.emit(self._matcher.get_mapping())

    def set_camera_names(self, cameras: list[str]):
        if cameras:
            self._matcher._video_list.set_items(cameras)
            self._matcher.mapping_changed.emit(self._matcher.get_mapping())

    def set_has_video(self, has_video: bool) -> None:
        """Called live by ``ModalityConfigPage`` whenever the Video tab's
        folder changes — with no upfront checkbox, this is the only way to
        know whether pose needs its own explicit frame rate."""
        if has_video == self._has_video:
            return
        self._has_video = has_video
        self._no_video_fps_spin.setVisible(not has_video)
        label = self._pose_form.labelForField(self._no_video_fps_spin)
        if label is not None:
            label.setVisible(not has_video)
        if self._match_with_camera_cb is not None:
            self._match_with_camera_cb.setChecked(has_video)
            self._match_with_camera_cb.setEnabled(has_video and self._is_irregular)

    def set_pose_fps_controls(
        self,
        mapping: list[tuple[str, str]],
        video_fps_by_camera: dict[str, int],
    ):
        self._video_fps_by_camera = dict(video_fps_by_camera)
        self._pose_to_video_map.clear()

        pose_names: list[str] = []
        for video_name, pose_name in mapping:
            if pose_name and pose_name != "None" and pose_name not in pose_names:
                pose_names.append(pose_name)
            if pose_name and pose_name != "None" and video_name and video_name != "None":
                self._pose_to_video_map.setdefault(pose_name, video_name)

        while self._per_camera_pose_fps_form.rowCount() > 0:
            self._per_camera_pose_fps_form.removeRow(0)
        self._pose_fps_by_camera_spins.clear()

        if len(pose_names) == 0:
            self._per_camera_pose_fps_box.setVisible(False)
            return

        self._per_camera_pose_fps_box.setVisible(True)
        for pose_name in pose_names:
            spin = QSpinBox()
            spin.setRange(1, 1000)
            saved_fps = self._config.fps_by_camera.get(pose_name)
            mapped_video = self._pose_to_video_map.get(pose_name)
            detected_fps = self._video_fps_by_camera.get(mapped_video) if mapped_video else None
            if detected_fps is not None and self._match_with_camera_cb and self._match_with_camera_cb.isChecked():
                spin.setValue(detected_fps)
            else:
                spin.setValue(
                    saved_fps
                    if saved_fps is not None
                    else (
                        detected_fps
                        if detected_fps is not None
                        else (self._config.fps if self._config.fps is not None else 30)
                    )
                )
            spin.setSuffix(" fps")
            label = f"{pose_name}:"
            if mapped_video:
                label = f"{pose_name} (from {mapped_video}):"
            self._per_camera_pose_fps_form.addRow(label, spin)
            self._pose_fps_by_camera_spins[pose_name] = spin

        self._apply_match_with_camera_mode()

    def _on_match_with_camera_changed(self):
        if self._match_with_camera_cb and self._match_with_camera_cb.isChecked():
            for pose_name, spin in self._pose_fps_by_camera_spins.items():
                mapped_video = self._pose_to_video_map.get(pose_name)
                detected_fps = self._video_fps_by_camera.get(mapped_video) if mapped_video else None
                if detected_fps is not None:
                    spin.setValue(detected_fps)
        self._apply_match_with_camera_mode()

    def _apply_match_with_camera_mode(self):
        is_matched = bool(self._match_with_camera_cb and self._match_with_camera_cb.isChecked())
        for spin in self._pose_fps_by_camera_spins.values():
            spin.setEnabled(not is_matched)

    def set_pose_offset_controls(self, pose_names: list[str]):
        if not self._per_pose_offset_form:
            return
        while self._per_pose_offset_form.rowCount() > 0:
            self._per_pose_offset_form.removeRow(0)
        self._pose_offset_spins.clear()

        if len(pose_names) <= 1:
            self._per_pose_offset_box.setVisible(False)
            return

        self._per_pose_offset_box.setVisible(not (self._const_offset_cb and self._const_offset_cb.isChecked()))
        for pose in pose_names:
            spin = QDoubleSpinBox()
            spin.setRange(-100000.0, 100000.0)
            spin.setDecimals(4)
            saved_offset = self._config.device_offsets.get(pose, 0.0)
            spin.setValue(saved_offset)
            spin.setSuffix(" s")
            self._per_pose_offset_form.addRow(f"{pose}:", spin)
            self._pose_offset_spins[pose] = spin

    def _on_const_offset_toggled(self, checked: bool):
        if not self._const_offset_spin or not self._per_pose_offset_box:
            return
        is_constant = self._const_offset_cb and self._const_offset_cb.isChecked()
        self._const_offset_spin.setVisible(is_constant)
        self._per_pose_offset_box.setVisible(not is_constant and len(self._pose_offset_spins) > 1)

    def _refresh_pose_offset_controls(self):
        if not self._stream_panel or not self._stream_panel.pattern:
            return

        pat = self._stream_panel.pattern
        poses: set[str] = set()
        for file_path in pat.files:
            row = extract_file_row(
                file_path,
                pat.segments,
                pat.tokenize_mode,
                regex_pattern=pat.regex_pattern,
            )
            pose = row.get("camera", "pose_1")
            poses.add(pose)

        self.set_pose_offset_controls(sorted(poses))

    def collect_state(self, config: ModalityConfig):
        config.source_software = self._software_combo.currentText()
        config.fps_by_camera = {pose_name: spin.value() for pose_name, spin in self._pose_fps_by_camera_spins.items()}
        if config.fps_by_camera:
            first_key = next(iter(config.fps_by_camera))
            config.fps = config.fps_by_camera[first_key]
        elif not self._has_video:
            config.fps = None if self._spin_is_required(self._no_video_fps_spin) else self._no_video_fps_spin.value()
        if self._is_irregular:
            self._collect_offsets(
                config,
                constant_checkbox=self._const_offset_cb,
                constant_spin=self._const_offset_spin,
                device_spins=self._pose_offset_spins,
            )
        if self._is_multi and self._stream_panel:
            sc = self._stream_panel.get_config()
            config.enabled = bool(sc)
            if sc:
                config.folder_path = sc.folder
                config.nested_subfolders = sc.nested
            config.pattern = self._stream_panel.pattern
            config.n_devices = _pattern_device_count(config.pattern, "camera")
        elif self._file_edit:
            config.enabled = bool(self._file_edit.text())
            config.single_file_path = self._file_edit.text()

    def validate(self) -> str | None:
        # An empty tab is fine — it's just not enabled (see collect_state).
        enabled = False
        if self._is_multi:
            if self._stream_panel and self._stream_panel.has_unresolved_folder():
                return "Pose: no pose files found in that folder."
            enabled = bool(self._stream_panel and self._stream_panel.get_config())
        else:
            enabled = bool(self._file_edit and self._file_edit.text())
        if enabled and not self._has_video and self._spin_is_required(self._no_video_fps_spin):
            return "Pose: give the frame rate (no video to read it from)."
        return None


# ─── Audio tab ────────────────────────────────────────────────────────────────


class AudioConfigTab(_BaseConfigTab):
    def __init__(self, config: ModalityConfig, parent: QWidget | None = None, mode: str = "pair"):
        super().__init__(config, parent)
        #: "Starts at"-style offsets are the one sanctioned fixed-offset
        #: mechanism, and that is mode 2 (free-running, known offset). Mode 1
        #: (pair) assumes every file is already perfectly aligned.
        self._offset_allowed = mode == "free_running"
        self._mic_offset_spins: dict[str, QDoubleSpinBox] = {}
        self._const_offset_cb: QCheckBox | None = None
        self._const_offset_spin: QDoubleSpinBox | None = None
        self._per_mic_offset_box: QGroupBox | None = None
        self._per_mic_offset_form: QFormLayout | None = None
        layout = QVBoxLayout(self)
        form = QFormLayout()

        mode_label = QLabel(
            f"<b>Mode:</b> {self._mode_title('Files aligned to trial period (Trials x Mics)', 'Continuous recording across session (Mics)')}"  # noqa: E501
        )
        layout.addWidget(mode_label)

        if self._is_multi:
            _audio_roles = ["ignore", "mic"] if self._is_irregular else None
            self._stream_panel = StreamPanel("audio", allowed_roles=_audio_roles)
            self._stream_panel.set_folder(config.folder_path)
            self._stream_panel.changed.connect(self._refresh_mic_offset_controls)
            self._stream_panel.changed.connect(self._detect_sr_from_folder)
            if self._is_irregular:
                layout.addWidget(QLabel("<b>Session audio files (one per mic)</b>"))
            else:
                layout.addWidget(QLabel("<b>Trial-aligned audio files</b>"))
            layout.addWidget(self._stream_panel, stretch=1)
            self._file_edit = None
        else:
            self._stream_panel = None
            self._file_edit = self._add_file_browse(
                form,
                "Audio file:",
                "Select audio file...",
                AUDIO_FILE_FILTER,
            )

        self._sr_spin = QDoubleSpinBox()
        self._sr_spin.setRange(1000, 192000)
        self._sr_spin.setDecimals(2)
        self._sr_spin.setValue(config.audio_sr if config.audio_sr is not None else 44100.0)
        self._sr_spin.setSuffix(" Hz")
        self._sr_spin.setReadOnly(True)
        self._sr_spin.setToolTip("Auto-detected from audio file")
        if config.audio_sr is None:
            self._mark_required(self._sr_spin)
        form.addRow("Sample rate:", self._sr_spin)

        # Offset controls (only in continuous mode, and only mode 2 — see
        # ``_offset_allowed``. Still built for mode 1 so collect_state has
        # somewhere to read a definite 0.0 from; just not shown or editable.
        if self._is_irregular:
            self._const_offset_cb = QCheckBox("Offset is constant across mics")
            self._const_offset_cb.setChecked(config.offset_constant_across_devices)
            self._const_offset_cb.toggled.connect(self._on_const_offset_toggled)
            self._const_offset_cb.setVisible(self._offset_allowed)
            form.addRow("", self._const_offset_cb)

            self._const_offset_spin = QDoubleSpinBox()
            self._const_offset_spin.setRange(-100000.0, 100000.0)
            self._const_offset_spin.setDecimals(4)
            self._const_offset_spin.setValue(config.constant_offset or 0.0)
            self._const_offset_spin.setSuffix(" s")
            self._const_offset_spin.setToolTip("Constant time offset for all mics (seconds)")
            self._const_offset_spin.setVisible(self._offset_allowed)
            form.addRow("Constant offset:", self._const_offset_spin)

            self._per_mic_offset_box = QGroupBox("Per-mic offsets")
            self._per_mic_offset_form = QFormLayout(self._per_mic_offset_box)
            self._per_mic_offset_box.setVisible(self._offset_allowed and not config.offset_constant_across_devices)
            form.addRow(self._per_mic_offset_box)
            if not self._offset_allowed:
                note = _muted(
                    "A fixed offset only applies in mode 2 (free-running, known offset); "
                    "mode 1 assumes every file is already aligned."
                )
                form.addRow("", note)

        layout.addLayout(form)

    def _on_file_selected(self, path: str):
        from ethograph.utils.audio import get_audio_sr

        sr = get_audio_sr(path)
        self._sr_spin.setValue(sr)
        self._clear_required(self._sr_spin)

    def _detect_sr_from_folder(self):
        """Auto-detect the sample rate from the folder's first file (the
        multi-device path had no auto-detection at all before)."""
        if not self._stream_panel:
            return
        first = self._stream_panel.first_file()
        if not first:
            return
        from ethograph.utils.audio import get_audio_sr

        try:
            sr = get_audio_sr(first)
        except (OSError, ValueError):
            return
        self._sr_spin.setValue(sr)
        self._clear_required(self._sr_spin)

    def set_mic_offset_controls(self, mic_names: list[str]):
        if not self._per_mic_offset_form:
            return
        while self._per_mic_offset_form.rowCount() > 0:
            self._per_mic_offset_form.removeRow(0)
        self._mic_offset_spins.clear()

        if len(mic_names) <= 1:
            self._per_mic_offset_box.setVisible(False)
            return

        self._per_mic_offset_box.setVisible(not (self._const_offset_cb and self._const_offset_cb.isChecked()))
        for mic in mic_names:
            spin = QDoubleSpinBox()
            spin.setRange(-100000.0, 100000.0)
            spin.setDecimals(4)
            saved_offset = self._config.device_offsets.get(mic, 0.0)
            spin.setValue(saved_offset)
            spin.setSuffix(" s")
            self._per_mic_offset_form.addRow(f"{mic}:", spin)
            self._mic_offset_spins[mic] = spin

    def _on_const_offset_toggled(self, checked: bool):
        if not self._const_offset_spin or not self._per_mic_offset_box:
            return
        is_constant = self._const_offset_cb and self._const_offset_cb.isChecked()
        self._const_offset_spin.setVisible(is_constant)
        self._per_mic_offset_box.setVisible(not is_constant and len(self._mic_offset_spins) > 1)

    def _refresh_mic_offset_controls(self):
        if not self._stream_panel or not self._stream_panel.pattern:
            return

        pat = self._stream_panel.pattern
        mics: set[str] = set()
        for file_path in pat.files:
            row = extract_file_row(
                file_path,
                pat.segments,
                pat.tokenize_mode,
                regex_pattern=pat.regex_pattern,
            )
            mic = row.get("mic", "mic_1")
            mics.add(mic)

        self.set_mic_offset_controls(sorted(mics))

    def collect_state(self, config: ModalityConfig):
        config.audio_sr = None if self._spin_is_required(self._sr_spin) else self._sr_spin.value()
        if self._is_irregular:
            self._collect_offsets(
                config,
                constant_checkbox=self._const_offset_cb,
                constant_spin=self._const_offset_spin,
                device_spins=self._mic_offset_spins,
            )
        if self._is_multi and self._stream_panel:
            sc = self._stream_panel.get_config()
            config.enabled = bool(sc)
            if sc:
                config.folder_path = sc.folder
                config.nested_subfolders = sc.nested
            config.pattern = self._stream_panel.pattern
            config.n_devices = _pattern_device_count(config.pattern, "mic")
        elif self._file_edit:
            config.enabled = bool(self._file_edit.text())
            config.single_file_path = self._file_edit.text()

    def validate(self) -> str | None:
        # An empty tab is fine — it's just not enabled (see collect_state).
        if self._is_multi:
            if self._stream_panel and self._stream_panel.has_unresolved_folder():
                return "Audio: no audio files found in that folder."
        elif self._file_edit and not self._file_edit.text():
            return "Audio: select an audio file."
        return None


# ─── Ephys tab ────────────────────────────────────────────────────────────────


class EphysConfigTab(_BaseConfigTab):
    def __init__(self, config: ModalityConfig, parent: QWidget | None = None):
        super().__init__(config, parent)
        layout = QVBoxLayout(self)

        info = QLabel(
            "Generate from an electrophysiology recording and/or kilosort folder.\n"
            "For supported extensions, see: https://neo.readthedocs.io/en/latest/iolist.html.\n\n"
            "At least one of ephys file or kilosort folder is required."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        # Ephys file
        ephys_container = QWidget()
        ephys_row = QHBoxLayout(ephys_container)
        ephys_row.setContentsMargins(0, 0, 0, 0)
        self._ephys_edit = QLineEdit()
        self._ephys_edit.setPlaceholderText("Select ephys file...")
        self._ephys_edit.setReadOnly(True)
        ephys_browse = QPushButton("Browse")
        ephys_browse.clicked.connect(self._browse_ephys)
        ephys_clear = QPushButton("Clear")
        ephys_clear.clicked.connect(lambda: self._ephys_edit.clear())
        ephys_row.addWidget(self._ephys_edit)
        ephys_row.addWidget(ephys_browse)
        ephys_row.addWidget(ephys_clear)
        form.addRow("Ephys file:", ephys_container)

        # Kilosort folder
        ks_container = QWidget()
        ks_row = QHBoxLayout(ks_container)
        ks_row.setContentsMargins(0, 0, 0, 0)
        self._ks_edit = QLineEdit()
        self._ks_edit.setPlaceholderText("Select kilosort output folder...")
        self._ks_edit.setReadOnly(True)
        ks_browse = QPushButton("Browse")
        ks_browse.clicked.connect(self._browse_kilosort)
        ks_clear = QPushButton("Clear")
        ks_clear.clicked.connect(lambda: self._ks_edit.clear())
        ks_row.addWidget(self._ks_edit)
        ks_row.addWidget(ks_browse)
        ks_row.addWidget(ks_clear)
        form.addRow("Kilosort folder:", ks_container)

        self._sr_spin = QSpinBox()
        self._sr_spin.setRange(1, 200000)
        self._sr_spin.setValue(config.ephys_sr if config.ephys_sr is not None else 30000)
        self._sr_spin.setSuffix(" Hz")
        if config.ephys_sr is None:
            self._mark_required(self._sr_spin)
        self._sr_spin.valueChanged.connect(lambda _: self._clear_required(self._sr_spin))
        form.addRow("Ephys sampling rate:", self._sr_spin)

        self._nchan_spin = QSpinBox()
        self._nchan_spin.setRange(1, 10000)
        self._nchan_spin.setValue(config.n_channels)
        form.addRow("N channels:", self._nchan_spin)

        self._offset_spin = self._add_offset_row(form)
        layout.addLayout(form)
        layout.addStretch()

    def _browse_ephys(self):
        result = QFileDialog.getOpenFileName(
            self,
            "Select ephys file",
            "",
            EPHYS_FILE_FILTER,
        )
        if result and result[0]:
            self._ephys_edit.setText(result[0])
            self._probe_ephys(result[0])
            self._auto_detect_kilosort(result[0])

    def _probe_ephys(self, path: str):
        try:
            from ethograph.io.ephys_loader import load_ephys

            loader = load_ephys(path)
            self._sr_spin.setValue(int(loader.rate))
            self._nchan_spin.setValue(loader.n_channels)
            self._clear_required(self._sr_spin)
        except (ValueError, ImportError):
            pass

    def _auto_detect_kilosort(self, ephys_path: str):
        ks_folder = Path(ephys_path).parent / "kilosort4"
        if ks_folder.is_dir() and not self._ks_edit.text():
            self._ks_edit.setText(str(ks_folder))

    def _browse_kilosort(self):
        start = self._ephys_edit.text()
        if start:
            start = str(Path(start).parent)
        folder = QFileDialog.getExistingDirectory(self, "Select kilosort folder", start)
        if folder:
            self._ks_edit.setText(folder)

    def collect_state(self, config: ModalityConfig):
        config.single_file_path = self._ephys_edit.text()
        config.neurons_path = self._ks_edit.text()
        config.ephys_sr = None if self._spin_is_required(self._sr_spin) else self._sr_spin.value()
        config.n_channels = self._nchan_spin.value()
        config.constant_offset = self._offset_spin.value()

    def validate(self) -> str | None:
        if not self._ephys_edit.text() and not self._ks_edit.text():
            return "Ephys: select at least one of ephys file or kilosort folder."
        return None


# ─── Page 2 container ─────────────────────────────────────────────────────────


class ModalityConfigPage(QWidget):
    def __init__(self, state: WizardState, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 2 — Configure each modality</b>"))
        layout.addSpacing(6)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("QTabBar::tab { min-width: 100px; padding: 6px 16px; }")
        layout.addWidget(self._tabs)

        # No upfront "which streams do you have" question any more: video,
        # pose and audio always get a tab, and whether a stream ends up
        # enabled is read back from whether its tab has anything in it
        # (each tab's collect_state). Ephys stays opt-in — it is not part
        # of this flow (mode "pair" never sets it, modes 2/3 configure it
        # entirely through the Timing page's recording-system fields).
        self._tab_map: dict[str, _BaseConfigTab] = {}
        self._tab_map["video"] = VideoConfigTab(state.video)
        self._tabs.addTab(self._tab_map["video"], "Video")
        self._tab_map["pose"] = PoseConfigTab(state.pose)
        self._tabs.addTab(self._tab_map["pose"], "Pose")
        self._tab_map["audio"] = AudioConfigTab(state.audio, mode=state.mode)
        self._tabs.addTab(self._tab_map["audio"], "Audio")
        if state.ephys.enabled:
            self._tab_map["ephys"] = EphysConfigTab(state.ephys)
            self._tabs.addTab(self._tab_map["ephys"], "Ephys")

        # Wire video-pose matching list updates
        video_tab = self._tab_map["video"]
        pose_tab = self._tab_map["pose"]

        if video_tab._stream_panel:
            video_tab._stream_panel.changed.connect(lambda: self._sync_cameras_to_pose())
            video_tab._stream_panel.changed.connect(lambda: self._refresh_per_camera_fps_controls())
            video_tab.fps_changed.connect(lambda: self._refresh_per_camera_fps_controls())
        pose_tab._matcher.mapping_changed.connect(lambda _m: self._refresh_per_camera_fps_controls())
        # Pose pattern changes are already handled within PoseConfigTab via _on_pose_pattern_changed()

        self._refresh_per_camera_fps_controls()

    def _sync_cameras_to_pose(self):
        """Update camera (video) list in pose matcher when video pattern changes."""
        video_tab: VideoConfigTab = self._tab_map["video"]
        pose_tab: PoseConfigTab = self._tab_map["pose"]
        pat = video_tab._stream_panel.pattern if video_tab._stream_panel else None
        if pat:
            summary = pat.summary()
            cameras = summary.get("camera", [])
            if cameras:
                pose_tab.set_camera_names(cameras)

    def _refresh_per_camera_fps_controls(self):
        video_tab: VideoConfigTab = self._tab_map["video"]
        pose_tab: PoseConfigTab = self._tab_map["pose"]

        pose_tab.set_has_video(bool(video_tab._stream_panel and video_tab._stream_panel.get_config()))

        video_tab._refresh_detected_video_fps()
        mapping = pose_tab._matcher.get_mapping()

        camera_names: list[str] = []
        for video_name, _pose_name in mapping:
            if video_name and video_name != "None" and video_name not in camera_names:
                camera_names.append(video_name)

        if not camera_names:
            pat = video_tab._stream_panel.pattern if video_tab._stream_panel else None
            if pat:
                camera_names = pat.summary().get("camera", [])

        video_tab.set_camera_fps_controls(camera_names)
        pose_tab.set_pose_fps_controls(mapping, video_tab.get_detected_fps_by_camera())

    def collect_state(self, state: WizardState):
        for name, tab in self._tab_map.items():
            cfg: ModalityConfig = getattr(state, name)
            tab.collect_state(cfg)

        # Collect camera/mic names from patterns
        video_tab: VideoConfigTab = self._tab_map["video"]
        if video_tab._stream_panel and video_tab._stream_panel.pattern:
            state.camera_names = video_tab._stream_panel.pattern.summary().get("camera", [])

        audio_tab: AudioConfigTab = self._tab_map["audio"]
        if audio_tab._stream_panel and audio_tab._stream_panel.pattern:
            state.mic_names = audio_tab._stream_panel.pattern.summary().get("mic", [])

        pose_tab: PoseConfigTab = self._tab_map["pose"]
        state.pose_camera_mapping = pose_tab._matcher.get_mapping()

    def validate(self, state: WizardState) -> str | None:
        for tab in self._tab_map.values():
            err = tab.validate()
            if err:
                return err
        video_tab: VideoConfigTab = self._tab_map["video"]
        has_video = bool(video_tab._stream_panel and video_tab._stream_panel.get_config())
        if state.mode != "pair" and not has_video:
            return "Video: modes 2 and 3 align the camera to the recording system; give it a folder."
        tabs = (video_tab, self._tab_map["pose"], self._tab_map["audio"])
        if not any(t._stream_panel and t._stream_panel.get_config() for t in tabs):
            return "Give at least one of video, pose or audio a folder."
        return None
