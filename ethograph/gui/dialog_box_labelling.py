"""Box labelling — OCTRON's workflow inside EthoGraph, one time index across cameras.

**Tools ▸ Box labelling (OCTRON)…** A non-modal dialog laid out as OCTRON's own
dock (a toolbox with *Manage project*, *Generate annotation data*, *Train
model*, *Analyze videos*) so OCTRON users are at home. It drives every open
camera view at once and calls the OCTRON fork headless for everything below
the canvas (``labels/octron_project.py`` for the files, ``gui/box_annotate.py``
for SAM). What is EthoGraph's: the label manager works *by individual and
camera* — pick an individual, click it in each camera where it is visible,
``X`` where it is not, ``Tab`` for the next — frame suggestion, a per-camera
balance table, and training / prediction as separate ``octron`` processes
configured by ``octron.yaml``. Segmentation is out of scope: OCTRON's detect
mode trains on the box of each mask. Design: ``notes/Design_box-labelling-octron.md``.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

import numpy as np
from qtpy.QtCore import QProcess, Qt
from qtpy.QtGui import QBrush, QColor, QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.box_annotate import CHUNK_SIZE, SamSession, TrackState, sam_models
from ethograph.gui.box_overlay import BoxLabelMode, MaskLayer, MaskOverlay
from ethograph.gui.dialog_busy_progress import BusyProgressDialog
from ethograph.gui.notify import notify
from ethograph.gui.pose_fill import VideoFrameSource, video_size
from ethograph.gui.pose_suggest import suggest_frames
from ethograph.gui.project import project_dir_of
from ethograph.io.derived import DerivedFeature, derived_loader_for
from ethograph.labels.octron_project import (
    OctronConfig,
    OctronProject,
    VideoEntry,
    read_tracks,
    trackers,
    yolo_models,
)
from ethograph.utils.paths import defaults_dir

logger = logging.getLogger(__name__)

SUGGEST_METHODS = (
    ("mixed", "Mixed: motion + diverse (k-means)"),
    ("motion", "Motion"),
    ("diverse", "Diverse (k-means)"),
    ("uniform", "Uniform"),
)
SUGGEST_MAX_SIDE = 160
#: Window the motion area is summed over when ranking frames: half a second of
#: sustained movement outranks a one-frame flicker of any size.
MOTION_WINDOW_S = 0.5
#: With this many cameras or more the motion panel opens as a heatmap, one row per camera.
HEATMAP_FROM_CAMERAS = 3
DOCK_WIDTH = 600
#: One colour per individual, by position in the list — the same in every camera
#: and written into OCTRON's organizer, so OCTRON shows it too. RGBA in [0, 1].
INDIVIDUAL_PALETTE: tuple[tuple[float, float, float, float], ...] = (
    (0.24, 0.86, 0.52, 1.0),  # green
    (0.95, 0.30, 0.30, 1.0),  # red
    (0.35, 0.60, 0.98, 1.0),  # blue
    (0.98, 0.70, 0.25, 1.0),  # orange
    (0.75, 0.45, 0.95, 1.0),  # purple
    (0.30, 0.85, 0.90, 1.0),  # cyan
    (0.95, 0.90, 0.30, 1.0),  # yellow
    (0.95, 0.55, 0.80, 1.0),  # pink
)


class _Camera:
    """One open camera view and what this dialog hangs on it."""

    def __init__(self, name: str, view, video_path: Path):
        self.name = name
        self.view = view
        self.video_path = video_path
        self.entry: VideoEntry | None = None
        self.session: SamSession | None = None
        self.overlay: MaskOverlay | None = None
        self.mode: BoxLabelMode | None = None
        self.track = TrackState()


class BoxLabellingDialog(QDialog):
    def __init__(self, data_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Box labelling (OCTRON)")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.setMinimumWidth(DOCK_WIDTH)
        self.resize(DOCK_WIDTH + 90, 760)
        self._position_at_right(parent)

        self._data_widget = data_widget
        self.app_state = data_widget.app_state
        self._shell = data_widget.shell
        self._cameras: dict[str, _Camera] = {}
        self._active_camera: str | None = None
        #: The camera under the mouse — what X acts on, so no click is needed to pick it.
        self._hover_camera: str | None = None
        self._individuals: list[str] = list(self.app_state.label_individuals() or [])
        self._active_individual: str | None = self._individuals[0] if self._individuals else None
        self._suggestions: list[int] = []
        self._suggestion_index = 0
        #: Cameras whose pixel-motion panel is already open, so Suggest never opens a second.
        self._motion_panels: set[str] = set()
        #: video → motion trace (bytes per frame), for this session only.
        self._motion_traces: dict[str, np.ndarray] = {}
        self._process: QProcess | None = None
        self._shortcuts: list[QShortcut] = []

        self.project = self._resolve_project()
        self.cfg = self.project.load_config()

        self._build_ui()
        self._refresh_cameras()
        self._bind_shortcuts()
        self.app_state.current_frame_changed.connect(self._on_frame_changed)
        self.app_state.video_path_changed.connect(self._on_video_changed)

    def _position_at_right(self, parent) -> None:
        """Open along the right edge of the main window, clamped to the screen."""
        ref = parent.window() if parent is not None else None
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        if ref is not None and ref.isVisible():
            geo = ref.frameGeometry()
            x = geo.right() - self.width()
            y = geo.top()
        elif avail is not None:
            x = avail.right() - self.width()
            y = avail.top()
        else:
            return
        if avail is not None:
            x = min(max(x, avail.left()), avail.right() - self.width())
            y = min(max(y, avail.top()), avail.bottom() - self.height())
        self.move(x, y)

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    def _resolve_project(self) -> OctronProject:
        project_dir = project_dir_of(self.app_state)
        root = (
            OctronProject.for_project(project_dir) if project_dir is not None else OctronProject(defaults_dir("octron"))
        )
        root.root.mkdir(parents=True, exist_ok=True)
        return root

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.toolbox = QToolBox()
        self.toolbox.addItem(self._build_project_page(), "Manage project")
        self.toolbox.addItem(self._build_annotate_page(), "Generate annotation data")
        self.toolbox.addItem(self._build_train_page(), "Train model")
        self.toolbox.addItem(self._build_analyze_page(), "Analyze videos")
        self.toolbox.setCurrentIndex(1)
        layout.addWidget(self.toolbox)

    # -- page 1 ---------------------------------------------------------------

    def _build_project_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        folder = QGroupBox("Project folder")
        fl = QHBoxLayout(folder)
        self.project_label = QLabel(str(self.project.root))
        self.project_label.setWordWrap(True)
        fl.addWidget(self.project_label, 1)
        v.addWidget(folder)

        cams = QGroupBox("Cameras of this trial (one video each)")
        cl = QVBoxLayout(cams)
        self.camera_table = QTableWidget(0, 3)
        self.camera_table.setHorizontalHeaderLabels(["camera", "hash", "labelled frames"])
        self.camera_table.horizontalHeader().setStretchLastSection(True)
        self.camera_table.verticalHeader().setVisible(False)
        cl.addWidget(self.camera_table)
        hint = QLabel("Every open camera view of the current trial is registered here on first use.")
        hint.setWordWrap(True)
        cl.addWidget(hint)
        v.addWidget(cams)
        v.addStretch()
        return page

    # -- page 2 ---------------------------------------------------------------

    def _build_annotate_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)

        model_box = QGroupBox("Model selection")
        ml = QHBoxLayout(model_box)
        self.sam_combo = QComboBox()
        for key, entry in sam_models().items():
            self.sam_combo.addItem(entry["name"], key)
            self.sam_combo.setItemData(self.sam_combo.count() - 1, entry.get("tooltip", ""), Qt.ToolTipRole)
        idx = self.sam_combo.findData(self.cfg.sam_model)
        self.sam_combo.setCurrentIndex(max(0, idx))
        self.sam_combo.currentIndexChanged.connect(self._on_sam_model_changed)
        ml.addWidget(self.sam_combo, 1)
        load_btn = QPushButton("Load model")
        load_btn.setToolTip("Build SAM for every open camera now; otherwise it is built on the first click.")
        load_btn.clicked.connect(self._on_load_model)
        ml.addWidget(load_btn)
        v.addWidget(model_box)

        labels_box = QGroupBox("Label manager — by individual and camera")
        ll = QVBoxLayout(labels_box)
        scheme_row = QHBoxLayout()
        self.same_type_cb = QCheckBox("Tracked individuals are of the same type")
        self.same_type_cb.setChecked(self.cfg.label_scheme == "suffix")
        self.same_type_cb.setToolTip(
            "Checked: one OCTRON label (one YOLO class) for every individual, with the individual as\n"
            "OCTRON's suffix — the 'LED 1 / LED 2' case in OCTRON's docs. Right for animals that look\n"
            "alike: the detector learns the kind, identity comes from tracking.\n"
            "Unchecked: each individual is its own label and class — only for animals the detector\n"
            "can tell apart (male / female plumage, colour rings).\n"
            "https://octron-tracking.github.io/OCTRON-docs/annotating/#label-manager"
        )
        self.same_type_cb.toggled.connect(self._on_same_type_toggled)
        scheme_row.addWidget(self.same_type_cb, 1)
        self.class_name_edit = QLineEdit(self.cfg.class_name)
        self.class_name_edit.setPlaceholderText("class name")
        self.class_name_edit.setMaximumWidth(110)
        self.class_name_edit.setToolTip("The one OCTRON label (YOLO class) all individuals share, e.g. bird")
        self.class_name_edit.editingFinished.connect(self._on_class_name_edited)
        scheme_row.addWidget(self.class_name_edit)
        self._on_same_type_toggled(self.same_type_cb.isChecked())
        ll.addLayout(scheme_row)

        add_row = QHBoxLayout()
        self.individual_edit = QLineEdit()
        self.individual_edit.setPlaceholderText("Individual …")
        add_row.addWidget(self.individual_edit, 1)
        add_btn = QPushButton("⊕ Create")
        add_btn.clicked.connect(self._on_add_individual)
        add_row.addWidget(add_btn)
        remove_btn = QPushButton("⊖ Remove")
        remove_btn.setToolTip("Remove the selected individual and, in every camera, its masks (OCTRON's Remove label)")
        remove_btn.clicked.connect(self._on_remove_individual)
        add_row.addWidget(remove_btn)
        ll.addLayout(add_row)

        self.individual_list = QListWidget()
        self.individual_list.setMaximumHeight(90)
        self.individual_list.currentRowChanged.connect(self._on_individual_row)
        ll.addWidget(self.individual_list)

        self.frame_table = QTableWidget(0, 1)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.horizontalHeader().setStretchLastSection(True)
        self.frame_table.setMaximumHeight(140)
        self.frame_table.setToolTip("Click a '✓ mask' cell to clear that mask; no mask means missing")
        self.frame_table.cellClicked.connect(self._on_frame_cell_clicked)
        ll.addWidget(self.frame_table)
        keys = QLabel(
            "left click = include · right click = exclude · no mask = <b>missing</b> in that camera · "
            "click a '✓ mask' cell to clear it · "
            "<b>Tab</b> = next individual · <b>N</b> / <b>B</b> = next / previous suggested frame · "
            "<b>Shift</b>+drag = pan the video · scroll = zoom"
        )
        keys.setWordWrap(True)
        keys.setTextFormat(Qt.RichText)
        ll.addWidget(keys)
        reset_row = QHBoxLayout()
        clear_btn = QPushButton("Clear this frame")
        clear_btn.setToolTip("Forget the active individual's mask and clicks on this frame in every camera")
        clear_btn.clicked.connect(self._on_clear_frame)
        reset_row.addWidget(clear_btn)
        ll.addLayout(reset_row)
        v.addWidget(labels_box)

        timeline = QGroupBox("Timeline control")
        tl = QVBoxLayout(timeline)
        sug = QHBoxLayout()
        self.suggest_method = QComboBox()
        for key, name in SUGGEST_METHODS:
            self.suggest_method.addItem(name, key)
        sug.addWidget(self.suggest_method, 1)
        self.suggest_count = QSpinBox()
        self.suggest_count.setRange(1, 500)
        self.suggest_count.setValue(24)
        sug.addWidget(self.suggest_count)
        self.motion_share_spin = QSpinBox()
        self.motion_share_spin.setRange(0, 100)
        self.motion_share_spin.setSuffix(" % motion")
        self.motion_share_spin.setValue(int(round(self.cfg.suggest_motion_share * 100)))
        self.motion_share_spin.setToolTip(
            "Mixed only: this share of the picks are the strongest movements; the rest are\n"
            "k-means picks among the moving frames. 0 = all diverse, 100 = all motion."
        )
        sug.addWidget(self.motion_share_spin)
        self.suggest_method.currentIndexChanged.connect(self._on_suggest_method_changed)
        self._on_suggest_method_changed(self.suggest_method.currentIndex())
        suggest_btn = QPushButton("Suggest")
        suggest_btn.clicked.connect(self._on_suggest)
        sug.addWidget(suggest_btn)
        tl.addLayout(sug)
        nav = QHBoxLayout()
        self.prev_suggested_btn = QPushButton("Previous (B)")
        self.prev_suggested_btn.setEnabled(False)
        self.prev_suggested_btn.clicked.connect(self._previous_suggestion)
        nav.addWidget(self.prev_suggested_btn)
        self.next_suggested_btn = QPushButton("Next suggested (N)")
        self.next_suggested_btn.setEnabled(False)
        self.next_suggested_btn.clicked.connect(self._next_suggestion)
        nav.addWidget(self.next_suggested_btn)
        self.suggest_status = QLabel("")
        nav.addWidget(self.suggest_status, 1)
        self.show_motion_cb = QCheckBox("Show video motion")
        self.show_motion_cb.setChecked(True)
        self.show_motion_cb.setToolTip(
            "After a motion-based suggestion, plot every open camera's motion trace\n"
            "(bytes per compressed frame, each normalised by its 95th percentile) as one\n"
            "panel: traces for up to two cameras, a heatmap for more.\n"
            "Frames are ranked by the sum of these traces."
        )
        nav.addWidget(self.show_motion_cb)
        tl.addLayout(nav)
        v.addWidget(timeline)

        batch = QGroupBox("Batch prediction")
        bl = QHBoxLayout(batch)
        self.one_btn = QPushButton("▷ 1")
        self.one_btn.clicked.connect(lambda: self._on_propagate(1))
        bl.addWidget(self.one_btn)
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 500)
        self.chunk_spin.setValue(CHUNK_SIZE)
        self.chunk_spin.valueChanged.connect(lambda _v: self._refresh_predict_buttons())
        self.many_btn = QPushButton()
        self.many_btn.clicked.connect(lambda: self._on_propagate(self.chunk_spin.value()))
        bl.addWidget(self.many_btn)
        bl.addWidget(self.chunk_spin)
        bl.addWidget(QLabel("Skip"))
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 100)
        bl.addWidget(self.skip_spin)
        v.addWidget(batch)
        self._refresh_predict_buttons()
        v.addStretch()
        return page

    # -- page 3 ---------------------------------------------------------------

    def _build_train_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        gen = QGroupBox("Generate training data")
        gl = QVBoxLayout(gen)
        row = QHBoxLayout()
        self.prune_cb = QCheckBox("Prune")
        self.prune_cb.setToolTip(
            "OCTRON's Prune: keep only frames where every label is present. Leave off with animals that leave a view."
        )
        self.prune_cb.setChecked(self.cfg.prune)
        row.addWidget(self.prune_cb)
        gen_btn = QPushButton("Generate")
        gen_btn.setToolTip("Runs `octron split --mode detect` on the project (a separate process)")
        gen_btn.clicked.connect(self._on_generate)
        row.addWidget(gen_btn)
        gl.addLayout(row)
        self.balance_table = QTableWidget(0, 2)
        self.balance_table.setHorizontalHeaderLabels(["camera · object", "labelled frames"])
        self.balance_table.horizontalHeader().setStretchLastSection(True)
        self.balance_table.verticalHeader().setVisible(False)
        gl.addWidget(self.balance_table)
        gl.addWidget(QLabel("Detection only (bboxes from masks) — no segmentation choice."))
        v.addWidget(gen)

        train = QGroupBox("Train")
        form = QFormLayout(train)
        self.model_combo = QComboBox()
        self.model_combo.addItems(yolo_models())
        self.model_combo.setCurrentText(self.cfg.model.lower())
        form.addRow("Choose model", self.model_combo)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["640", "1024", "1280"])
        self.imgsz_combo.setCurrentText(str(self.cfg.imgsz))
        form.addRow("Img. size", self.imgsz_combo)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(self.cfg.epochs)
        form.addRow("Epochs", self.epochs_spin)
        self.save_period_spin = QSpinBox()
        self.save_period_spin.setRange(1, 5000)
        self.save_period_spin.setValue(self.cfg.save_period)
        form.addRow("Save period", self.save_period_spin)
        self.batch_edit = QLineEdit(str(self.cfg.batch))
        self.batch_edit.setToolTip(
            "auto = OCTRON's AutoBatch; a number overrides it. Batch 1 is what broke the first run."
        )
        form.addRow("Batch", self.batch_edit)
        self.device_edit = QLineEdit(self.cfg.device)
        form.addRow("Device", self.device_edit)
        split_row = QHBoxLayout()
        self.train_frac = QDoubleSpinBox()
        self.train_frac.setRange(0.05, 0.95)
        self.train_frac.setSingleStep(0.05)
        self.train_frac.setValue(self.cfg.train_fraction)
        self.val_frac = QDoubleSpinBox()
        self.val_frac.setRange(0.0, 0.9)
        self.val_frac.setSingleStep(0.05)
        self.val_frac.setValue(self.cfg.val_fraction)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 10_000)
        self.seed_spin.setValue(self.cfg.seed)
        split_row.addWidget(QLabel("train"))
        split_row.addWidget(self.train_frac)
        split_row.addWidget(QLabel("val"))
        split_row.addWidget(self.val_frac)
        split_row.addWidget(QLabel("seed"))
        split_row.addWidget(self.seed_spin)
        form.addRow("Split", split_row)
        flags = QHBoxLayout()
        self.resume_cb = QCheckBox("Resume")
        self.overwrite_train_cb = QCheckBox("Overwrite")
        flags.addWidget(self.resume_cb)
        flags.addWidget(self.overwrite_train_cb)
        form.addRow("", flags)
        buttons = QHBoxLayout()
        self.train_btn = QPushButton("Start")
        self.train_btn.clicked.connect(self._on_train)
        buttons.addWidget(self.train_btn)
        copy_btn = QPushButton("Copy command")
        copy_btn.clicked.connect(lambda: self._copy_command(self._train_command()))
        buttons.addWidget(copy_btn)
        form.addRow("", buttons)
        self.train_note = QLabel("Writes octron.yaml · runs as a separate process, outside the GUI.")
        self.train_note.setWordWrap(True)
        form.addRow(self.train_note)
        v.addWidget(train)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Process output appears here.")
        v.addWidget(self.log, 1)
        return page

    # -- page 4 ---------------------------------------------------------------

    def _build_analyze_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        videos = QGroupBox("Videos to be analyzed")
        vl = QVBoxLayout(videos)
        self.predict_videos = QListWidget()
        self.predict_videos.setMaximumHeight(90)
        vl.addWidget(self.predict_videos)
        vl.addWidget(QLabel("This trial's open cameras, one video each."))
        v.addWidget(videos)

        pred = QGroupBox("Create predictions from videos")
        form = QFormLayout(pred)
        self.weights_label = QLabel(str(self.project.weights_path))
        self.weights_label.setWordWrap(True)
        form.addRow("Model", self.weights_label)
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(trackers())
        self.tracker_combo.setCurrentText(self.cfg.tracker)
        form.addRow("Tracker", self.tracker_combo)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.cfg.conf_thresh)
        form.addRow("Confidence", self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(self.cfg.iou_thresh)
        form.addRow("IOU", self.iou_spin)
        self.pred_skip = QSpinBox()
        self.pred_skip.setRange(0, 100)
        self.pred_skip.setValue(self.cfg.skip_frames)
        form.addRow("Skip frames", self.pred_skip)
        flags = QHBoxLayout()
        self.one_subject_cb = QCheckBox("1 subject")
        self.one_subject_cb.setChecked(self.cfg.one_object_per_label)
        self.overwrite_pred_cb = QCheckBox("Overwrite")
        flags.addWidget(self.one_subject_cb)
        flags.addWidget(self.overwrite_pred_cb)
        form.addRow("", flags)
        buttons = QHBoxLayout()
        self.predict_btn = QPushButton("Let's go!")
        self.predict_btn.clicked.connect(self._on_predict)
        buttons.addWidget(self.predict_btn)
        import_btn = QPushButton("Import tracks")
        import_btn.setToolTip("Read the tracks OCTRON wrote for these videos into the session as per-camera features")
        import_btn.clicked.connect(self._on_import_tracks)
        buttons.addWidget(import_btn)
        copy_btn = QPushButton("Copy command")
        copy_btn.clicked.connect(lambda: self._copy_command(self._predict_command()))
        buttons.addWidget(copy_btn)
        form.addRow("", buttons)
        v.addWidget(pred)
        v.addStretch()
        return page

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    def _open_views(self) -> list[tuple[str, object, str | None]]:
        area = self._shell.video_area
        out: list[tuple[str, object, str | None]] = []
        primary = area.primary
        name = getattr(primary, "camera_name", None) or self.app_state.primary_camera or "cam-1"
        out.append((str(name), primary, self.app_state.video_path))
        for key, view in getattr(area, "extras", {}).items():
            cam = getattr(view, "camera_name", key)
            out.append((str(cam), view, getattr(view, "source_video_path", None)))
        return out

    def _refresh_cameras(self) -> None:
        """Bind an overlay + click mode to every open view; drop views that closed."""
        seen: set[str] = set()
        for name, view, video in self._open_views():
            if not video or not getattr(view, "has_video", False):
                continue
            seen.add(name)
            cam = self._cameras.get(name)
            if cam is not None and cam.view is view and cam.video_path == Path(video):
                continue
            if cam is not None:
                self._release_camera(cam)
            cam = _Camera(name, view, Path(video))
            src = video_size(str(video)) or view.image_size()
            cam.overlay = MaskOverlay(view.scene(), tuple(view.image_size()), tuple(src))
            cam.mode = BoxLabelMode(view, name, cam.overlay, self._on_point, on_hover=self._set_hover_camera)
            try:
                view.clicked.connect(lambda n=name: self._set_active_camera(n))
            except (RuntimeError, AttributeError):
                pass
            if cam.entry is None:
                try:
                    cam.entry = self.project.register_video(cam.video_path)
                except Exception:  # noqa: BLE001 - a video that cannot be probed is registered on first click
                    logger.debug("could not register %s", cam.video_path, exc_info=True)
            self._cameras[name] = cam
        for name in list(self._cameras):
            if name not in seen:
                self._release_camera(self._cameras.pop(name))
        if self._active_camera not in self._cameras:
            self._active_camera = next(iter(self._cameras), None)
        self._refresh_individual_list()
        self._refresh_frame_table()
        self._refresh_camera_table()
        self._refresh_predict_videos()
        self._redraw_all()

    def _release_camera(self, cam: _Camera) -> None:
        if cam.session is not None:
            cam.session.release()
            cam.session = None
        if cam.mode is not None:
            cam.mode.detach()
            cam.mode = None

    def _set_hover_camera(self, name: str) -> None:
        if name != self._hover_camera:
            self._hover_camera = name
            self._refresh_frame_table()

    def _target_camera(self):
        """The camera a key acts on: under the mouse if any, else the last clicked."""
        return self._cameras.get(self._hover_camera or "") or self._cameras.get(self._active_camera or "")

    def _set_active_camera(self, name: str) -> None:
        self._active_camera = name
        self._refresh_frame_table()

    def _ensure_session(self, cam: _Camera) -> SamSession:
        if cam.session is not None and cam.session.ready:
            return cam.session
        if cam.entry is None:
            cam.entry = self.project.register_video(cam.video_path)
        session = SamSession(self.project, cam.entry, self.cfg.sam_model)
        busy = BusyProgressDialog(f"Loading SAM for {cam.name}…", parent=self)
        _, error = busy.execute(session.build)
        if error:
            raise RuntimeError(f"Could not build SAM for {cam.name}: {error}")
        cam.session = session
        for obj in session.objects.values():  # a camera clicked in another order keeps the individual's colour
            ind = self._individual_of(obj)
            if ind in self._individuals:
                session.object_for(obj.label, obj.suffix, color=list(self._color_for(ind)))
        self._refresh_camera_table()
        return session

    # ------------------------------------------------------------------
    # Individuals
    # ------------------------------------------------------------------

    def _color_for(self, individual: str) -> tuple[float, float, float, float]:
        i = self._individuals.index(individual) if individual in self._individuals else 0
        return INDIVIDUAL_PALETTE[i % len(INDIVIDUAL_PALETTE)]

    def _qcolor_for(self, individual: str) -> QColor:
        r, g, b, _ = self._color_for(individual)
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    def _refresh_individual_list(self) -> None:
        self.individual_list.blockSignals(True)
        self.individual_list.clear()
        frame = self._current_frame()
        for name in self._individuals:
            done = sum(1 for cam in self._cameras.values() if self._has_mask(cam, name, frame))
            item = QListWidgetItem(f"●  {name}   {done}/{len(self._cameras)}")
            item.setData(Qt.UserRole, name)
            item.setForeground(QBrush(self._qcolor_for(name)))
            self.individual_list.addItem(item)
            if name == self._active_individual:
                self.individual_list.setCurrentItem(item)
        self.individual_list.blockSignals(False)

    def _on_individual_row(self, row: int) -> None:
        item = self.individual_list.item(row)
        if item is not None:
            self._active_individual = item.data(Qt.UserRole)
            self._refresh_frame_table()

    def _on_add_individual(self) -> None:
        name = self.individual_edit.text().strip()
        if not name or name in self._individuals:
            return
        self._individuals.append(name)
        self._active_individual = name
        self.individual_edit.clear()
        self._refresh_individual_list()
        self._refresh_frame_table()

    def _on_remove_individual(self) -> None:
        name = self._active_individual
        if name is None:
            return
        label, suffix = self.cfg.label_for(name)
        n_masks = 0
        for cam in self._cameras.values():
            if cam.entry is not None:
                n_masks += self.project.label_counts(cam.entry).get(f"{label} {suffix}".strip(), 0)
        if n_masks:
            answer = QMessageBox.question(
                self,
                "Remove individual",
                f"Remove {name!r} and its {n_masks} labelled frame(s) across cameras? This deletes the mask stores.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        for cam in self._cameras.values():
            if cam.session is not None:
                for obj in list(cam.session.objects.values()):
                    if obj.label == label and obj.suffix == suffix:
                        cam.session.remove_object(obj.obj_id)
            elif cam.entry is not None:
                self.project.remove_object(cam.entry, label, suffix)
        self._individuals.remove(name)
        self._active_individual = self._individuals[0] if self._individuals else None
        self._refresh_individual_list()
        self._refresh_frame_table()
        self._refresh_camera_table()
        self._refresh_balance_table()
        self._redraw_all()

    def _cycle_individual(self, step: int = 1) -> None:
        if not self._individuals:
            return
        i = self._individuals.index(self._active_individual) if self._active_individual in self._individuals else -1
        self._active_individual = self._individuals[(i + step) % len(self._individuals)]
        self._refresh_individual_list()
        self._refresh_frame_table()

    # ------------------------------------------------------------------
    # Frame table + overlays
    # ------------------------------------------------------------------

    def _current_frame(self) -> int:
        return int(self.app_state.current_frame or 0)

    def _individual_of(self, obj) -> str:
        """The individual an OCTRON object stands for under the current scheme."""
        for ind in self._individuals:
            if self.cfg.label_for(ind) == (obj.label, obj.suffix):
                return ind
        return obj.layer_name

    def _has_mask(self, cam: _Camera, individual: str, frame: int) -> bool:
        if cam.session is None:
            return False
        label, suffix = self.cfg.label_for(individual)
        for obj in cam.session.objects.values():
            if obj.label == label and obj.suffix == suffix:
                return cam.session.mask(obj.obj_id, frame) is not None
        return False

    def _refresh_frame_table(self) -> None:
        frame = self._current_frame()
        cams = list(self._cameras.values())
        self.frame_table.setRowCount(len(cams))
        self.frame_table.setColumnCount(1 + len(self._individuals))
        self.frame_table.setHorizontalHeaderLabels(["camera", *self._individuals])
        for c, ind in enumerate(self._individuals, start=1):
            header = QTableWidgetItem(ind)
            header.setForeground(QBrush(self._qcolor_for(ind)))
            self.frame_table.setHorizontalHeaderItem(c, header)
        for r, cam in enumerate(cams):
            marker = "▶ " if cam.name == (self._hover_camera or self._active_camera) else ""
            name = QTableWidgetItem(marker + cam.name)
            self.frame_table.setItem(r, 0, name)
            for c, ind in enumerate(self._individuals, start=1):
                has = self._has_mask(cam, ind, frame)
                cell = QTableWidgetItem("✓ mask" if has else "missing")
                if has:
                    cell.setForeground(QBrush(self._qcolor_for(ind)))
                self.frame_table.setItem(r, c, cell)
        self.frame_table.resizeColumnsToContents()

    def _on_frame_cell_clicked(self, row: int, column: int) -> None:
        """Clear the clicked (camera, individual) mask on this frame — a wrong click undone."""
        cams = list(self._cameras.values())
        if column < 1 or row >= len(cams) or column - 1 >= len(self._individuals):
            return
        cam, ind = cams[row], self._individuals[column - 1]
        frame = self._current_frame()
        if cam.session is None or not self._has_mask(cam, ind, frame):
            return
        label, suffix = self.cfg.label_for(ind)
        for obj in cam.session.objects.values():
            if obj.label == label and obj.suffix == suffix:
                cam.session.clear_frame(obj.obj_id, frame)
        self._redraw(cam, frame)
        self._refresh_individual_list()
        self._refresh_frame_table()

    def _redraw_all(self) -> None:
        frame = self._current_frame()
        for cam in self._cameras.values():
            self._redraw(cam, frame)

    def _redraw(self, cam: _Camera, frame: int) -> None:
        if cam.overlay is None:
            return
        layers: list[MaskLayer] = []
        points: list[tuple[float, float, bool]] = []
        if cam.session is not None and cam.session.ready:
            for obj in cam.session.objects.values():
                mask = cam.session.mask(obj.obj_id, frame)
                if mask is not None:
                    layers.append(MaskLayer(mask, self._color_for(self._individual_of(obj))))
                state = cam.session.prompts.get((obj.obj_id, frame))
                if state is not None:
                    points.extend((x, y, bool(lbl)) for (x, y), lbl in zip(state.points, state.labels, strict=True))
        cam.overlay.set_masks(layers)
        cam.overlay.set_points(points)
        try:
            cam.view.request_draw()
        except (RuntimeError, AttributeError):
            pass

    def _on_frame_changed(self, frame: int) -> None:
        # A new frame starts with the first individual, so the loop is always
        # click · Tab · click · N — never "which one was I on?".
        if self._individuals:
            self._active_individual = self._individuals[0]
        chunk = self.chunk_spin.value()
        for cam in self._cameras.values():
            if cam.track.seek_resets(int(frame), chunk):
                self._reset_memory(cam)
        self._redraw_all()
        self._refresh_individual_list()
        self._refresh_frame_table()
        self._refresh_predict_buttons()

    def _on_video_changed(self, *_args) -> None:
        self._refresh_cameras()

    # ------------------------------------------------------------------
    # Clicks
    # ------------------------------------------------------------------

    def _on_point(self, camera: str, x: float, y: float, positive: bool) -> None:
        cam = self._cameras.get(camera)
        if cam is None or self._active_individual is None:
            notify("Pick an individual first.", severity="warning")
            return
        self._active_camera = camera
        frame = self._current_frame()
        try:
            session = self._ensure_session(cam)
            label, suffix = self.cfg.label_for(self._active_individual)
            obj = session.object_for(label, suffix, color=list(self._color_for(self._active_individual)))
            if cam.track.click_resets(frame):
                self._reset_memory(cam)
            mask = session.add_point(obj.obj_id, frame, x, y, positive)
            if mask is None:
                # SAM2-HQ refuses a new object once tracking started: start
                # the track afresh from this click.
                self._reset_memory(cam)
                mask = session.add_point(obj.obj_id, frame, x, y, positive)
        except Exception as e:  # noqa: BLE001 - outermost GUI boundary
            logger.exception("SAM click failed")
            notify(f"SAM failed: {e}", severity="error")
            return
        if mask is None:
            notify("SAM returned no mask for this click.", "warning")
        else:
            cam.track.note_click(frame)
        self._redraw(cam, frame)
        self._refresh_individual_list()
        self._refresh_frame_table()
        self._refresh_predict_buttons()

    def _on_clear_frame(self) -> None:
        if self._active_individual is None:
            return
        frame = self._current_frame()
        label, suffix = self.cfg.label_for(self._active_individual)
        for cam in self._cameras.values():
            if cam.session is None:
                continue
            for obj in cam.session.objects.values():
                if obj.label == label and obj.suffix == suffix:
                    cam.session.clear_frame(obj.obj_id, frame)
            self._redraw(cam, frame)
        self._refresh_individual_list()
        self._refresh_frame_table()

    def _on_load_model(self) -> None:
        for cam in self._cameras.values():
            try:
                self._ensure_session(cam)
            except Exception as e:  # noqa: BLE001 - outermost GUI boundary
                logger.exception("SAM build failed")
                notify(str(e), severity="error")
                return
        self._redraw_all()

    def _on_sam_model_changed(self, _index: int) -> None:
        self.cfg.sam_model = self.sam_combo.currentData()
        self.cfg.save(self.project.config_path)
        for cam in self._cameras.values():
            if cam.session is not None:
                cam.session.release()
                cam.session = None

    def _on_same_type_toggled(self, checked: bool) -> None:
        self.class_name_edit.setVisible(bool(checked))
        scheme = "suffix" if checked else "per_individual"
        if scheme != self.cfg.label_scheme:
            self.cfg.label_scheme = scheme
            self.cfg.save(self.project.config_path)
            self._refresh_frame_table()

    def _on_class_name_edited(self) -> None:
        name = self.class_name_edit.text().strip() or "animal"
        self.class_name_edit.setText(name)
        if name != self.cfg.class_name:
            self.cfg.class_name = name
            self.cfg.save(self.project.config_path)
            self._refresh_frame_table()

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def _predict_starts(self) -> dict[str, int]:
        """Camera → the frame Predict would run from, for the cameras allowed to predict now."""
        current = self._current_frame()
        starts: dict[str, int] = {}
        for cam in self._cameras.values():
            if cam.session is None:
                continue
            start = cam.track.predict_start(current)
            if start is not None:
                starts[cam.name] = start
        return starts

    def _refresh_predict_buttons(self) -> None:
        starts = self._predict_starts()
        n = self.chunk_spin.value()
        self.one_btn.setEnabled(bool(starts))
        self.many_btn.setEnabled(bool(starts))
        if starts:
            first = min(starts.values())
            self.many_btn.setText(f"▷ {n} frames from {first}")
            where = ", ".join(f"{name}: {frame}" for name, frame in starts.items())
            tip = f"Predict from the seeded frame in every camera — {where}"
        else:
            self.many_btn.setText(f"▷ {n} frames")
            tip = (
                "Click an individual first. After a prediction, click inside the predicted "
                "frames to correct it, or stand on the last predicted frame to continue."
            )
        self.one_btn.setToolTip(tip)
        self.many_btn.setToolTip(tip)

    def _reset_memory(self, cam: _Camera) -> None:
        if cam.session is not None:
            cam.session.reset()
        cam.track.reset()

    def _on_propagate(self, n_frames: int) -> None:
        starts = self._predict_starts()
        if not starts:
            notify("Click an individual in at least one camera first.", severity="warning")
            return
        cams = [self._cameras[name] for name in starts]
        start = min(starts.values())
        # Guard 1: prediction always begins where the memory was seeded, so
        # the playhead goes back there before anything runs.
        if start != self._current_frame():
            self._seek(start)
        skip = self.skip_spin.value()
        busy = BusyProgressDialog("Propagating masks…", parent=self)
        last_frame = start

        def run():
            nonlocal last_frame
            for cam in cams:
                cam_start = starts[cam.name]
                cam_last = cam_start
                for i, (frame, _masks) in enumerate(cam.session.propagate(cam_start, n_frames, skip)):
                    cam_last = max(cam_last, frame)
                    busy.setLabelText(f"{cam.name}: frame {frame} ({i + 1}/{n_frames})")
                    busy.pump_events()
                    if busy.wasCanceled():
                        break
                cam.session.flush()
                cam.track.note_predicted(cam_start, cam_last)
                last_frame = max(last_frame, cam_last)

        _, error = busy.execute(run)
        if error:
            logger.error("propagation failed: %s", error)
            notify(f"Propagation failed: {error}", severity="error")
        self._redraw_all()
        self._refresh_camera_table()
        self._refresh_balance_table()
        self._refresh_predict_buttons()
        if last_frame > start:
            # Review what propagation did: play from the frame the user was on
            # to the last predicted frame, masks redrawing on every frame, and
            # stop there. The GUI's own segment playback, so speed and audio
            # behave as everywhere else.
            video = self.app_state.video
            if video is not None:
                video.play_segment(start, last_frame)
            else:
                self._seek(last_frame)

    # ------------------------------------------------------------------
    # Timeline: annotated frames + suggestions
    # ------------------------------------------------------------------

    def _annotated_frames(self) -> np.ndarray:
        frames: set[int] = set()
        for cam in self._cameras.values():
            if cam.session is None:
                continue
            for obj_id in cam.session.objects:
                frames.update(int(f) for f in cam.session.annotated_frames(obj_id))
        return np.array(sorted(frames), dtype=int)

    def _on_suggest(self) -> None:
        cam = self._cameras.get(self._active_camera or "") or next(iter(self._cameras.values()), None)
        if cam is None:
            notify("Open a camera view first.", severity="warning")
            return
        method = self.suggest_method.currentData()
        count = self.suggest_count.value()
        n_frames = int(getattr(cam.view, "n_frames", 0) or self.app_state.num_frames or 0)
        busy = BusyProgressDialog("Scanning frames for suggestions…", parent=self)

        def progress(fraction: float) -> bool:
            busy.setLabelText(f"Scanning frames… {int(fraction * 100)}%")
            busy.pump_events()
            return not busy.wasCanceled()

        share = self.motion_share_spin.value() / 100.0
        if share != self.cfg.suggest_motion_share:
            self.cfg.suggest_motion_share = share
            self.cfg.save(self.project.config_path)

        def run():
            fps = float(getattr(cam.view, "fps", 0) or self.app_state.video_fps or 0)
            start = int(getattr(cam.view, "start_frame", 0) or 0)
            trace = None
            if method in ("motion", "mixed") and fps:
                busy.setLabelText("Computing video motion…")
                busy.pump_events()
                trace = self._combined_motion(start, n_frames, fps, busy)
            window = max(1, int(round(MOTION_WINDOW_S * fps))) if fps else 1
            if method == "motion" and trace is not None:
                exclude = set(int(f) for f in self._annotated_frames())
                return suggest_frames(method, count, n_frames, exclude=exclude, motion=trace, motion_window=window)
            with VideoFrameSource(
                str(cam.video_path),
                fps=fps,
                n_frames=n_frames,
                max_side=SUGGEST_MAX_SIDE,
                start_frame=int(getattr(cam.view, "start_frame", 0) or 0),
            ) as frames:
                exclude = set(int(f) for f in self._annotated_frames())
                return suggest_frames(
                    method,
                    count,
                    n_frames,
                    frames=frames,
                    exclude=exclude,
                    progress=progress,
                    motion=trace,
                    motion_window=window,
                    motion_share=share,
                    motion_gate=self.cfg.suggest_motion_gate,
                )

        result, error = busy.execute(run)
        if error or result is None:
            if error:
                notify(f"Suggestion failed: {error}", severity="error")
            return
        self._suggestions = sorted(int(f) for f in result)
        self._suggestion_index = 0
        self.next_suggested_btn.setEnabled(bool(self._suggestions))
        self.prev_suggested_btn.setEnabled(bool(self._suggestions))
        self.suggest_status.setText(f"{len(self._suggestions)} suggested")
        if method in ("motion", "mixed") and self.show_motion_cb.isChecked():
            self._show_motion_panel(cam)
        if self._suggestions:
            self._go_to_suggestion(0)

    def _motion_trace(self, cam: _Camera, fps: float) -> np.ndarray:
        """The camera video's motion trace: bytes per compressed frame, read without decoding.

        Kept in memory for this dialog only — it takes well under a second per
        video, so nothing is written to disk.
        """
        from ethograph.features.movement import extract_packet_motion

        key = str(cam.video_path)
        trace = self._motion_traces.get(key)
        if trace is None:
            trace = extract_packet_motion(cam.video_path, fps=fps).values.astype(np.float32)
            self._motion_traces[key] = trace
        return trace

    def _on_suggest_method_changed(self, _index: int) -> None:
        self.motion_share_spin.setVisible(self.suggest_method.currentData() == "mixed")

    def _camera_motion(self, start: int, n_frames: int, fps: float, busy=None) -> tuple[list[str], np.ndarray]:
        """``(camera names, (T, n_cameras))`` normalised motion traces over every open camera.

        Each camera's trace is divided by its own 95th percentile — a mirror
        view and a top view differ in size, distance and contrast, and the raw
        pixel differences would let the biggest view decide alone. A camera on a
        different frame grid is left out; it cannot be combined per frame.
        """
        names: list[str] = []
        traces: list[np.ndarray] = []
        for other in self._cameras.values():
            if busy is not None:
                busy.setLabelText(f"Computing video motion: {other.name}…")
                busy.pump_events()
            fps_other = float(getattr(other.view, "fps", 0) or fps)
            other_start = int(getattr(other.view, "start_frame", 0) or 0)
            trace = self._motion_trace(other, fps_other)[other_start : other_start + n_frames]
            if len(trace) != n_frames:
                continue
            scale = float(np.percentile(trace, 95)) or 1.0
            names.append(other.name)
            traces.append(trace / scale)
        if not traces:
            raise RuntimeError("No camera has a motion trace on this trial's frame grid.")
        return names, np.stack(traces, axis=1).astype(np.float32)

    def _combined_motion(self, start: int, n_frames: int, fps: float, busy=None) -> np.ndarray:
        """What ranks a frame: the normalised traces summed, so movement seen in
        several cameras at once outranks the same movement seen in one."""
        _names, traces = self._camera_motion(start, n_frames, fps, busy)
        return traces.sum(axis=1)

    def _show_motion_panel(self, cam: _Camera) -> None:
        """Register the camera's motion trace as a derived feature and open a line plot on it."""
        loader = derived_loader_for(self.app_state)
        container = getattr(self._data_widget, "plot_container", None)
        if loader is None or container is None:
            return
        fps = float(getattr(cam.view, "fps", 0) or self.app_state.video_fps or 0)
        if not fps:
            return
        start = int(getattr(cam.view, "start_frame", 0) or 0)
        n_frames = int(getattr(cam.view, "n_frames", 0) or self.app_state.num_frames or 0)
        names, traces = self._camera_motion(start, n_frames, fps)
        frames = np.arange(len(traces))
        video = self.app_state.video
        if video is not None and cam.view is self._shell.video_area.primary:
            time = np.array([video.frame_to_time(int(f)) for f in frames], dtype=float)
        else:
            offset = 0.0
            alignment = self.app_state.nwb_alignment
            if alignment is not None:
                offset = float(alignment.stream_offset_for_trial(self.app_state.trials_sel, "video", device=cam.name))
            time = frames / fps + offset
        name = "video_motion"
        loader.register(
            DerivedFeature(name, time=time, values=traces.astype(float), dim_labels=names, n_columns=len(names))
        )
        self._data_widget.refresh_feature_choices()
        if name not in self._motion_panels:
            # Two traces read fine side by side; three or more cameras read
            # better as rows of a heatmap, one per camera.
            panel_type = "heatmap" if len(names) >= HEATMAP_FROM_CAMERAS else "lineplot"
            container.add_panel(panel_type, feature=name)
            self._motion_panels.add(name)
        container.schedule_labels_redraw()

    def _go_to_suggestion(self, index: int) -> None:
        if not self._suggestions:
            return
        self._suggestion_index = index % len(self._suggestions)
        self.suggest_status.setText(f"{self._suggestion_index + 1} / {len(self._suggestions)} suggested")
        self._seek(self._suggestions[self._suggestion_index])

    def _next_suggestion(self) -> None:
        if not self._suggestions:
            return
        self._go_to_suggestion(self._suggestion_index + 1)

    def _previous_suggestion(self) -> None:
        if not self._suggestions:
            return
        self._go_to_suggestion(self._suggestion_index - 1)

    def _seek(self, frame: int) -> None:
        video = self.app_state.video
        if video is not None:
            video.seek_to_frame(int(frame))

    # ------------------------------------------------------------------
    # Tables on the other pages
    # ------------------------------------------------------------------

    def _refresh_camera_table(self) -> None:
        cams = list(self._cameras.values())
        self.camera_table.setRowCount(len(cams))
        for r, cam in enumerate(cams):
            entry = cam.entry
            n = 0
            if cam.session is not None:
                n = int(len(self._annotated_for(cam)))
            self.camera_table.setItem(r, 0, QTableWidgetItem(cam.name))
            self.camera_table.setItem(r, 1, QTableWidgetItem(entry.hash8 if entry else "—"))
            self.camera_table.setItem(r, 2, QTableWidgetItem(str(n)))
        self.camera_table.resizeColumnsToContents()

    def _annotated_for(self, cam: _Camera) -> set[int]:
        frames: set[int] = set()
        if cam.session is not None:
            for obj_id in cam.session.objects:
                frames.update(int(f) for f in cam.session.annotated_frames(obj_id))
        return frames

    def _refresh_balance_table(self) -> None:
        rows: list[tuple[str, int]] = []
        for cam in self._cameras.values():
            if cam.entry is None:
                continue
            for layer, n in self.project.label_counts(cam.entry).items():
                rows.append((f"{cam.name} · {layer}", n))
        self.balance_table.setRowCount(len(rows))
        for r, (name, n) in enumerate(rows):
            self.balance_table.setItem(r, 0, QTableWidgetItem(name))
            self.balance_table.setItem(r, 1, QTableWidgetItem(str(n)))
        self.balance_table.resizeColumnsToContents()

    def _refresh_predict_videos(self) -> None:
        self.predict_videos.clear()
        for cam in self._cameras.values():
            self.predict_videos.addItem(f"{cam.name}  —  {cam.video_path.name}")

    # ------------------------------------------------------------------
    # Train / predict as separate processes
    # ------------------------------------------------------------------

    def _read_config_from_form(self) -> OctronConfig:
        self.cfg.model = self.model_combo.currentText()
        self.cfg.imgsz = int(self.imgsz_combo.currentText())
        self.cfg.epochs = self.epochs_spin.value()
        self.cfg.save_period = self.save_period_spin.value()
        self.cfg.batch = self.batch_edit.text().strip() or "auto"
        self.cfg.device = self.device_edit.text().strip() or "auto"
        self.cfg.train_fraction = float(self.train_frac.value())
        self.cfg.val_fraction = float(self.val_frac.value())
        self.cfg.seed = self.seed_spin.value()
        self.cfg.prune = self.prune_cb.isChecked()
        self.cfg.tracker = self.tracker_combo.currentText()
        self.cfg.conf_thresh = float(self.conf_spin.value())
        self.cfg.iou_thresh = float(self.iou_spin.value())
        self.cfg.skip_frames = self.pred_skip.value()
        self.cfg.one_object_per_label = self.one_subject_cb.isChecked()
        self.cfg.save(self.project.config_path)
        return self.cfg

    def _train_command(self) -> list[str]:
        cfg = self._read_config_from_form()
        return self.project.train_command(
            cfg, overwrite=self.overwrite_train_cb.isChecked(), resume=self.resume_cb.isChecked()
        )

    def _predict_command(self) -> list[str]:
        cfg = self._read_config_from_form()
        videos = [cam.video_path for cam in self._cameras.values()]
        return self.project.predict_command(cfg, videos, overwrite=self.overwrite_pred_cb.isChecked())

    def _copy_command(self, cmd: list[str]) -> None:
        text = " ".join(shlex.quote(c) for c in cmd)
        QApplication.clipboard().setText(text)
        notify("Command copied.", "info")

    def _on_generate(self) -> None:
        for cam in self._cameras.values():
            if cam.session is not None:
                cam.session.flush()
        self._run_process(self.project.split_command(self._read_config_from_form()), "split")

    def _on_train(self) -> None:
        for cam in self._cameras.values():
            if cam.session is not None:
                cam.session.release()  # free the GPU for the trainer
                cam.session = None
            cam.track.reset()
        self._redraw_all()
        self._refresh_predict_buttons()
        self._run_process(self._train_command(), "train")

    def _on_predict(self) -> None:
        if not self.project.weights_path.is_file():
            notify(f"No trained weights at {self.project.weights_path}. Train first.", severity="warning")
            return
        self._run_process(self._predict_command(), "predict")

    def _run_process(self, cmd: list[str], what: str) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            notify("A process is still running.", severity="warning")
            return
        self.log.clear()
        self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd))
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._append_log(proc))
        proc.finished.connect(lambda code, _status: self._on_process_finished(what, code))
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.project.root))
        self._process = proc
        self.toolbox.setCurrentIndex(2)
        proc.start()
        if not proc.waitForStarted(5000):
            notify(f"Could not start {cmd[0]!r}. Is the octron package installed in this environment?", "error")
            self._process = None

    def _append_log(self, proc: QProcess) -> None:
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.replace("\r", "\n").split("\n"):
            if line.strip():
                self.log.appendPlainText(line)

    def _on_process_finished(self, what: str, code: int) -> None:
        self.log.appendPlainText(f"[{what} finished with exit code {code}]")
        notify(f"octron {what} finished (exit {code})", "info" if code == 0 else "error")
        self._refresh_balance_table()
        self.weights_label.setText(str(self.project.weights_path))

    # ------------------------------------------------------------------
    # Import tracks into the session
    # ------------------------------------------------------------------

    def _on_import_tracks(self) -> None:
        loader = derived_loader_for(self.app_state)
        if loader is None:
            notify("No dataset loaded.", severity="warning")
            return
        cfg = self._read_config_from_form()
        alignment = self.app_state.nwb_alignment
        trial = self.app_state.trials_sel
        n_added = 0
        for cam in self._cameras.values():
            folder = self.project.prediction_folder(cam.video_path, cfg.tracker)
            if not folder.is_dir():
                continue
            fps = float(getattr(cam.view, "fps", 0) or self.app_state.video_fps or 0)
            if not fps:
                notify(f"Unknown frame rate for {cam.name}; cannot place its tracks in time.", "error")
                continue
            offset = 0.0
            if alignment is not None:
                offset = float(alignment.stream_offset_for_trial(trial, "video", device=cam.name))
            for track in read_tracks(folder):
                time = track.frame_idx / fps + offset
                values = np.column_stack([track.pos_x, track.pos_y])
                name = f"box_{cam.name}_{track.label}_{track.track_id}"
                feature = DerivedFeature(name, time=time, values=values, dim_labels=["x", "y"], n_columns=2)
                loader.register(feature)
                n_added += 1
        if n_added:
            self._data_widget.refresh_feature_choices()
        notify(f"Imported {n_added} track(s) as features for this trial.", "info")

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _bind_shortcuts(self) -> None:
        for keys, slot in (
            ("Tab", lambda: self._cycle_individual(1)),
            ("Shift+Tab", lambda: self._cycle_individual(-1)),
            ("N", self._next_suggestion),
            ("B", self._previous_suggestion),
        ):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        for cam in list(self._cameras.values()):
            self._release_camera(cam)
        self._cameras.clear()
        for signal, slot in (
            (self.app_state.current_frame_changed, self._on_frame_changed),
            (self.app_state.video_path_changed, self._on_video_changed),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        for sc in self._shortcuts:
            sc.setEnabled(False)
        super().closeEvent(event)
