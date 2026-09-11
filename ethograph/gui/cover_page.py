"""Full-screen start / cover page shown before a dataset is loaded.

Presents three entry points (matching the design in the project brief):

1. **Template datasets** — reuse :meth:`IOWidget._on_select_template_clicked`.
2. **Drag & drop files** — drop already-aligned media/feature/label files;
   ethograph classifies them by extension and loads. What several files of
   one stream *mean* is the one question asked up front, via the drop
   layout combo (``app_state.drop_layout``, remembered across restarts):

   - ``"same_trial"`` (default) — several files are several cameras/mics on
     one trial; the user is optionally asked which video is which camera,
     and a single-trial alignment NWB is synthesised so the normal loader
     can consume the loose media.
   - ``"multi_trial"`` — several files of one stream are several trials of
     one device, natural-sort paired (:meth:`CoverPage._populate_io_from_multi_trial`,
     the Data wizard's single-camera "Pair" route driven headlessly): a
     folder of ``trial001.mp4, trial002.mp4, ...`` (optionally with matching
     pose/audio files) becomes a real multi-trial ``TrialTree``.

   With a **project folder** chosen (remembered in ``gui_settings.yaml``) the
   drop lands in the project's ``sessions/{timestamp}/`` and is listed for
   reopening (``gui/project.py``); without one it is throwaway in the system
   temp dir (unique name per drop; stale ones are cleaned up best-effort).
3. **Data wizard** — reuse :meth:`IOWidget._on_create_nc_clicked`.

The page runs *before* the main window is shown: it accepts once a dataset
is loaded (``app_state.ready``); closing the dialog (X / Esc) means the GUI
never opens.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import natsort
import pandas as pd
import xarray as xr
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.datasets import DATASETS
from ethograph.gui.dialog_select_template import TEMPLATE_ASSETS_DIR
from ethograph.gui.file_dialogs import browse_open_dir
from ethograph.gui.project import DropRecord, list_drops, new_drop_dir, project_dir_of, record_drop, restore_drop
from ethograph.io.audio_extract import ensure_extracted_audio, has_embedded_audio
from ethograph.io.nc_drop import concat_on_camera, positions_fit_frame
from ethograph.io.validation import (
    AUDIO_EXTENSIONS,
    EPHYS_EXTENSIONS,
    EPHYS_EXTENSIONS_RAW,
    IMAGE_EXTENSIONS,
    POSE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    movement_dataset_info,
)
from ethograph.utils.paths import tmp_alignment_base

# POSE_SOFTWARES is shared with the pose-overlay prompt in pose_render.
from .app_constants import POSE_SOFTWARES
from .notify import notify, notify_dialog

logger = logging.getLogger(__name__)

FEATURE_EXTENSIONS = {".nc", ".nwb", ".npz"}
NPY_EXTENSIONS = {".npy"}
LABEL_EXTENSIONS = {".tsv"}
# Pose extensions whose source software cannot be inferred from the suffix alone
# (a ``.slp`` is always SLEAP; a ``.h5``/``.csv`` could be several tools).
AMBIGUOUS_POSE_EXTENSIONS = {".h5", ".hdf5", ".csv"}

# One accent colour per entry point — repeated on the card border, the number
# badge and (for drag & drop) the drop zone, so the three options read as
# three distinct paths.
_ACCENTS = {
    "template": "#4fc3f7",
    "drop": "#81c784",
    "custom": "#ffb74d",
}

# The card paddings, button heights and preview sizes below were tuned on a
# 1080 px-tall screen. On shorter screens (13" laptops, scaled displays) they
# are multiplied by ``CoverPage._scale`` so the page still fits vertically.
_REFERENCE_SCREEN_HEIGHT = 1080
_MIN_SCALE = 0.6


def _available_geometry(widget=None):
    """Available geometry of the widget's screen (primary screen as fallback)."""
    handle = getattr(widget, "screen", None) if widget is not None else None
    screen = handle() if callable(handle) else None
    if screen is None:
        screen = QApplication.primaryScreen()
    return screen.availableGeometry()


def classify_files(paths: list[str]) -> dict[str, list[str]]:
    """Bucket dropped file paths by ethograph data type (by extension).

    A ``.nwb`` can be pose, ephys or a feature/session file; it is treated as a
    *session* (feature) file here since that is the loadable unit.  A folder is a
    Kilosort output when it holds ``spike_times.npy`` (``neurons`` bucket),
    otherwise a pynapple ``session`` folder.
    """
    buckets: dict[str, list[str]] = {
        "session": [],
        "video": [],
        "pose": [],
        "image": [],
        "audio": [],
        "ephys": [],
        "neurons": [],
        "npy": [],
        "labels": [],
        "unknown": [],
    }
    for p in paths:
        path = Path(p)
        if path.is_dir():
            if (path / "spike_times.npy").exists():
                buckets["neurons"].append(p)
            else:
                buckets["session"].append(p)
            continue
        ext = path.suffix.lower()
        if ext == ".nc" and movement_dataset_info(p) is not None:
            # A movement poses/bboxes dataset is a pose file that needs no
            # conversion: it pairs with a camera like a DLC .h5 does.
            buckets["pose"].append(p)
        elif ext in FEATURE_EXTENSIONS:
            buckets["session"].append(p)
        elif ext in NPY_EXTENSIONS:
            buckets["npy"].append(p)
        elif ext in VIDEO_EXTENSIONS:
            buckets["video"].append(p)
        elif ext in IMAGE_EXTENSIONS:
            buckets["image"].append(p)
        elif ext in POSE_EXTENSIONS:
            buckets["pose"].append(p)
        elif ext in AUDIO_EXTENSIONS:
            buckets["audio"].append(p)
        elif ext in EPHYS_EXTENSIONS or ext in EPHYS_EXTENSIONS_RAW:
            buckets["ephys"].append(p)
        elif ext in LABEL_EXTENSIONS:
            buckets["labels"].append(p)
        else:
            buckets["unknown"].append(p)
    return buckets


def _audio_info(path: str) -> tuple[float, float]:
    """Read a real audio (sample_rate, duration_s) — never hardcode a fallback.

    Tries soundfile (wav/flac/ogg), then PyAV (mp4/mov/mkv containers), then the
    stdlib ``wave`` module.  Raises if no audio stream / rate can be read.
    """
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.samplerate), float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001 - fall through to PyAV / wave
        pass
    try:
        import av

        with av.open(path) as container:
            for stream in container.streams:
                if stream.type == "audio" and stream.rate:
                    duration = float(container.duration) / av.time_base if container.duration else 0.0
                    return float(stream.rate), duration
    except Exception:  # noqa: BLE001 - fall through to wave
        pass
    import wave

    with wave.open(path, "rb") as w:
        rate = float(w.getframerate())
        return rate, w.getnframes() / rate


@dataclass(frozen=True)
class _FeatureEntry:
    """One feature source of a drop: the camera (or file) it is named after,
    its path, and the rate of the video it is paired with, if any."""

    camera: str
    path: str
    fps: float | None


def _is_trial_tree(nc_path: str) -> bool:
    """A ``.nc`` saved by ethograph holds one child per trial: a session, not a feature file."""
    try:
        with xr.open_datatree(nc_path, engine="netcdf4") as tree:
            return bool(tree.children)
    except (OSError, ValueError):
        return False


def _pose_file_fps(pose_path: str) -> float | None:
    """The frame rate a pose file carries itself: only a movement ``.nc`` has one."""
    info = movement_dataset_info(pose_path)
    return info[1] if info is not None else None


def _open_pose_dataset(pose_path: str, source_software: str | None, fps: float | None):
    """A pose file as a movement dataset: a ``.nc`` is opened as is, anything
    else is converted by ``movement.io.load_dataset`` (which needs *fps*)."""
    from movement.io import load_dataset

    if Path(pose_path).suffix.lower() == ".nc":
        with xr.open_dataset(pose_path) as opened:
            return opened.load()
    if not fps:
        raise RuntimeError("A pose file dropped without a video needs its frame rate.")
    return load_dataset(pose_path, source_software, fps)


def _pose_duration(pose_path: str, source_software: str | None, fps: float) -> float:
    """Duration in seconds of a pose file at *fps* (its only time source)."""
    from ethograph.gui.pose_render import load_pose_from_file

    pr = load_pose_from_file(pose_path, source_software, fps)
    if len(pr.data) == 0:
        raise RuntimeError(f"Pose file {Path(pose_path).name} holds no frames.")
    frame_col = 1 if pr.data.shape[1] > 3 else 0
    n_frames = int(pr.data[:, frame_col].max()) + 1
    return n_frames / fps


class _CamMatchDialog(QDialog):
    """Single-trial video↔camera / pose↔camera assignment.

    Shown only when pose files need pairing with videos (poses dropped and
    more than one video or pose): the user orders the videos as cam1, cam2, …
    and the pose files are paired by row.
    """

    def __init__(self, videos: list[str], poses: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign cameras")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Order the videos top-to-bottom as cam1, cam2, …  Pose files on "
                "the same row are paired with that camera."
            )
        )

        cols = QHBoxLayout()
        self._video_list = QListWidget()
        self._video_list.setDragDropMode(QListWidget.InternalMove)
        for v in videos:
            self._video_list.addItem(Path(v).name)
        self._pose_list = QListWidget()
        self._pose_list.setDragDropMode(QListWidget.InternalMove)
        for p in poses:
            self._pose_list.addItem(Path(p).name)

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("<b>Videos (cam order)</b>"))
        vbox.addWidget(self._video_list)
        pbox = QVBoxLayout()
        pbox.addWidget(QLabel("<b>Pose files</b>"))
        pbox.addWidget(self._pose_list)
        cols.addLayout(vbox)
        cols.addLayout(pbox)
        layout.addLayout(cols)

        self._videos = {Path(v).name: v for v in videos}
        self._poses = {Path(p).name: p for p in poses}

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def ordered_videos(self) -> list[str]:
        return [self._videos[self._video_list.item(i).text()] for i in range(self._video_list.count())]

    def ordered_poses(self) -> list[str]:
        return [self._poses[self._pose_list.item(i).text()] for i in range(self._pose_list.count())]


class _DropDetailsDialog(QDialog):
    """Follow-up prompt for the few dropped inputs that cannot be inferred.

    Everything detectable (video fps, audio rate, ephys params) is read on drop
    and never surfaced. Only genuinely unknowable values appear here: a numpy
    file's sample rate, a pose file's source software when its extension is
    ambiguous, and the pose frame rate when no video was dropped alongside
    (image + pose drop). The dialog shows only the rows that are actually
    needed. One preference also lives here: whether to extract a dropped
    video's embedded audio track for the audio trace / spectrogram.
    """

    def __init__(
        self,
        need_npy_sr: bool,
        npy_name: str | None,
        need_pose_software: bool,
        need_pose_fps: bool = False,
        audio_track_videos: list[str] | None = None,
        extract_audio_default: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("A few more details")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)

        self._sr_spin = None
        self._software_combo = None
        self._pose_fps_spin = None
        self._extract_audio_cb = None

        if audio_track_videos:
            names = ", ".join(f"<b>{Path(v).name}</b>" for v in audio_track_videos)
            note = QLabel(f"{names} contains an audio track.")
            note.setTextFormat(Qt.RichText)
            note.setWordWrap(True)
            layout.addWidget(note)
            self._extract_audio_cb = QCheckBox("Extract the audio for audio trace / spectrogram plots")
            self._extract_audio_cb.setChecked(extract_audio_default)
            layout.addWidget(self._extract_audio_cb)

        if need_npy_sr:
            layout.addWidget(
                QLabel(
                    f"<b>{npy_name}</b><br>Sampling rate of the numpy data "
                    "(samples per second) — this cannot be read from the file."
                )
            )
            row = QHBoxLayout()
            row.addWidget(QLabel("Data sampling rate:"))
            self._sr_spin = QSpinBox()
            self._sr_spin.setRange(1, 1000000)
            self._sr_spin.setValue(30)
            self._sr_spin.setSuffix(" Hz")
            row.addWidget(self._sr_spin, 1)
            layout.addLayout(row)

        if need_pose_software:
            layout.addWidget(QLabel("Which software produced the pose / tracking file?"))
            row = QHBoxLayout()
            row.addWidget(QLabel("Source software:"))
            self._software_combo = QComboBox()
            self._software_combo.addItems(POSE_SOFTWARES)
            row.addWidget(self._software_combo, 1)
            layout.addLayout(row)

        if need_pose_fps:
            layout.addWidget(
                QLabel(
                    "No video was dropped, so the pose frame rate cannot be "
                    "read from anywhere — the camera fps the pose was tracked at:"
                )
            )
            row = QHBoxLayout()
            row.addWidget(QLabel("Pose frame rate:"))
            self._pose_fps_spin = QDoubleSpinBox()
            self._pose_fps_spin.setRange(0.001, 100000.0)
            self._pose_fps_spin.setDecimals(3)
            self._pose_fps_spin.setValue(30.0)
            self._pose_fps_spin.setSuffix(" fps")
            row.addWidget(self._pose_fps_spin, 1)
            layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def data_sr(self) -> int | None:
        return self._sr_spin.value() if self._sr_spin is not None else None

    def source_software(self) -> str | None:
        return self._software_combo.currentText() if self._software_combo is not None else None

    def pose_fps(self) -> float | None:
        return self._pose_fps_spin.value() if self._pose_fps_spin is not None else None

    def extract_audio(self) -> bool:
        return self._extract_audio_cb is not None and self._extract_audio_cb.isChecked()


class _DropList(QListWidget):
    """A QListWidget that accepts file drops and records the paths."""

    def __init__(self, parent=None, accent: str = "rgba(255,255,255,60)", min_height: int = 160):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.paths: list[str] = []
        self.setStyleSheet(
            f"QListWidget {{ border: 2px dashed {accent}; border-radius: 8px; min-height: {min_height}px; }}"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p and p not in self.paths:
                self.paths.append(p)
                self.addItem(Path(p).name)
        event.acceptProposedAction()

    def clear_paths(self):
        self.paths = []
        self.clear()


class CoverPage(QDialog):
    """Modal start page shown until a dataset is loaded."""

    def __init__(self, shell, io_widget, parent=None):
        # No Qt parent by default: the main window is still hidden at startup,
        # and a dialog parented to a hidden window gets no Windows taskbar
        # entry (it can vanish behind other windows with no way back).
        super().__init__(parent)
        self.shell = shell
        self.io_widget = io_widget
        self.app_state = io_widget.app_state
        self._drop_tmp_dir: Path | None = None
        #: Set when a recorded drop was picked from the reopen list: the IO
        #: fields already hold its state, so Load must not rebuild it.
        self._reopened_drop: Path | None = None
        self.setWindowTitle("ethograph — get started")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setSizeGripEnabled(True)

        avail = _available_geometry(self)
        self._scale = max(_MIN_SCALE, min(1.0, avail.height() / _REFERENCE_SCREEN_HEIGHT))

        # All content lives in a scroll area: without it the dialog's minimum
        # size hint (three cards + the supported-types strip) exceeds a short
        # screen and the window cannot be made smaller than its content.
        content = QWidget()
        outer = QVBoxLayout(content)
        m = self._px(24)
        outer.setContentsMargins(m, m, m, m)
        outer.setSpacing(self._px(16))

        outer.addLayout(self._build_prerecording_bar())
        outer.addWidget(self._build_project_bar())

        body = QHBoxLayout()
        body.setSpacing(self._px(16))
        body.addWidget(self._build_template_card(), 2)

        # Cards 2 + 3 share a column with the load bar directly beneath them —
        # the bar belongs to those two paths only, not to templates.
        right = QVBoxLayout()
        right.setSpacing(self._px(16))
        cards = QHBoxLayout()
        cards.setSpacing(self._px(16))
        cards.addWidget(self._build_drop_card(), 2)
        cards.addWidget(self._build_custom_card(), 5)
        right.addLayout(cards, 1)
        right.addWidget(self._build_load_bar())
        body.addLayout(right, 7)
        outer.addLayout(body)

        outer.addWidget(self._build_supported_types_strip())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.viewport().setAutoFillBackground(False)
        content.setAutoFillBackground(False)
        scroll.setWidget(content)
        shell_layout = QVBoxLayout(self)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.addWidget(scroll)
        # Small enough that the user can always shrink the window; the scroll
        # area takes over once the content no longer fits.
        self.setMinimumSize(min(700, avail.width()), min(420, avail.height()))
        self._refresh_project_ui()

    def _px(self, value: float) -> int:
        """Scale a pixel size tuned for a 1080 px-tall screen to this screen."""
        return max(1, int(round(value * self._scale)))

    def _pt(self, value: float) -> float:
        """Scale a point font size, rounded to a half-point."""
        return round(value * self._scale * 2) / 2

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_prerecording_bar(self) -> QHBoxLayout:
        """Tools for the work that happens **before** there is anything to load.

        The cover page is the only screen a user sees before a recording exists,
        which is exactly when tags have to be printed — putting that behind a
        loaded dataset (or the keypoint dialog, which needs a video) means it is
        only reachable once it is too late to use.

        A menu rather than a bare button: printing tags is the first of this
        kind of tool, not the last.
        """
        row = QHBoxLayout()
        tools = QToolButton()
        tools.setText("🛠  Pre-recording tools")
        tools.setPopupMode(QToolButton.InstantPopup)
        tools.setToolTip("Things to do before a single frame is recorded.")
        menu = QMenu(tools)
        menu.addAction("Print tag sheet…", self._open_tag_sheet)
        tools.setMenu(menu)
        self._tools_button = tools
        row.addWidget(tools)
        row.addStretch()
        return row

    def _open_tag_sheet(self) -> None:
        """Print-ready fiducial tags, with no video and no dataset in sight."""
        from ethograph.gui.dialog_tag_sheet import TagSheetDialog

        self._tag_sheet = TagSheetDialog(self.app_state, parent=self)
        self._tag_sheet.show()
        self._tag_sheet.raise_()

    # ------------------------------------------------------------------
    # Project folder
    # ------------------------------------------------------------------

    def _build_project_bar(self) -> QFrame:
        """Where this study's drops are kept — chosen once, remembered globally.

        Data never moves into the folder; only what a drop synthesises
        (alignment NWB, derived ``.nc``, layout) lands under ``sessions/``,
        which is what makes the drop reopenable from the list on card 2.
        """
        bar = QFrame()
        bar.setObjectName("projectBar")
        bar.setStyleSheet(
            "QFrame#projectBar { border: 1px solid rgba(255,255,255,35);"
            " border-radius: 10px; background-color: rgba(255,255,255,8); }"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(self._px(16), self._px(8), self._px(16), self._px(8))
        row.setSpacing(self._px(10))

        label = QLabel("<b>Project folder</b>")
        label.setToolTip(
            "A folder for this study. Drag &amp; drops made while it is set are kept in\n"
            "its sessions/ folder (one timestamped folder per drop) and can be reopened\n"
            "from the drop card. Remembered across restarts. Your data stays where it is."
        )
        row.addWidget(label)

        self._project_edit = QLineEdit()
        self._project_edit.setReadOnly(True)
        self._project_edit.setPlaceholderText("None — drops are throwaway until a folder is chosen")
        row.addWidget(self._project_edit, 1)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_project)
        row.addWidget(browse_btn)
        self._project_clear_btn = QPushButton("Clear")
        self._project_clear_btn.clicked.connect(self._on_clear_project)
        row.addWidget(self._project_clear_btn)
        return bar

    def _on_drop_layout_changed(self, index: int) -> None:
        self.app_state.drop_layout = self._drop_layout_combo.itemData(index)

    def _drop_layout(self) -> str:
        return self._drop_layout_combo.currentData() or "same_trial"

    def _on_browse_project(self) -> None:
        current = getattr(self.app_state, "project_path", None)
        path = browse_open_dir(self, self.app_state, "Choose a project folder", preferred_dir=current)
        if not path:
            return
        self.app_state.project_path = str(Path(path))
        self._refresh_project_ui()

    def _on_clear_project(self) -> None:
        self.app_state.project_path = None
        self._refresh_project_ui()

    def _project_dir(self) -> Path | None:
        return project_dir_of(self.app_state)

    def _refresh_project_ui(self) -> None:
        """Mirror ``app_state.project_path`` into the bar and the reopen list."""
        project = self._project_dir()
        self._project_edit.setText(str(project) if project else "")
        self._project_clear_btn.setEnabled(project is not None)

        combo = self._reopen_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Reopen a previous drop…", None)
        drops = list_drops(project) if project else []
        for folder, record in drops:
            combo.addItem(record.title, str(folder))
        combo.blockSignals(False)
        self._reopen_row.setVisible(project is not None)
        combo.setEnabled(bool(drops))

    def _on_reopen_drop(self, index: int) -> None:
        """Put a recorded drop's state into the IO fields; Load then loads it as-is."""
        folder = self._reopen_combo.itemData(index)
        if not folder:
            self._reopened_drop = None
            return
        record = DropRecord.load(folder)
        try:
            restore_drop(record, self.app_state)
        except Exception as e:  # noqa: BLE001 - outermost GUI boundary
            logger.exception("Failed to restore dropped session %s", folder)
            notify_dialog(f"Could not reopen this drop:\n{e}", "error")
            return
        self._reopened_drop = Path(folder)
        self._drop.clear_paths()
        self._drop.addItem(f"↩ {record.title}")

    def _build_supported_types_strip(self) -> QFrame:
        """A one-line reference of what can be dragged & dropped (with examples)."""

        def _fmt(exts) -> str:
            return " ".join(f"<code>{e}</code>" for e in sorted(exts))

        pose_estimation = (
            "<b>DeepLabCut</b> (<code>.csv</code>, <code>.h5</code>)"
            "  ·  <b>SLEAP</b> (<code>.slp</code>, <code>.h5</code>)"
            "  ·  <b>LightningPose</b> (<code>.csv</code>)"
            "  ·  <b>Anipose</b> (<code>.csv</code>)"
            "  ·  <b>VIA-tracks</b> (<code>.csv</code>)"
        )

        ephys_docs = "https://akseli-ilmanen.github.io/ethograph/getting_started/loading_ephys.html"
        ephys = (
            "all Neo-supported formats, raw data (<code>.dat</code>, <code>.bin</code>, …) "
            "and Kilosort folders — "
            f"<a href='{ephys_docs}'>see docs</a>"
        )

        rows = [
            (
                "🎞 Video",
                _fmt(VIDEO_EXTENSIONS) + "  — if the video contains audio, just drop the file: the GUI "
                "can extract it for the audio trace / spectrogram plots",
            ),
            ("🔊 Audio", _fmt(AUDIO_EXTENSIONS)),
            ("📈 Pose estimation", pose_estimation),
            ("⚡ Ephys", ephys),
            ("📊 Features", _fmt(NPY_EXTENSIONS) + " / <code>.nc</code> / <code>.nwb</code> / <code>.npz</code>"),
        ]
        items = "".join(f"<li><b>{name}</b>&nbsp;&nbsp;{exts}</li>" for name, exts in rows)
        cells = f"<ul style='margin:0; -qt-list-indent:1;'>{items}</ul>"

        font_pt = self._pt(10)
        frame = QFrame()
        frame.setObjectName("typesStrip")
        frame.setStyleSheet(
            "QFrame#typesStrip { border-top: 1px solid rgba(255,255,255,25);"
            f" padding-top: {self._px(8)}px; }}"
            " QFrame#typesStrip code { color: #81c784; }"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(self._px(4), self._px(6), self._px(4), 0)
        lay.setSpacing(self._px(2))
        heading = QLabel("Supported files — drag any of these onto the drop zone:")
        heading.setStyleSheet(f"color: rgba(255,255,255,150); font-size: {font_pt}pt;")
        lay.addWidget(heading)
        body = QLabel(cells)
        body.setTextFormat(Qt.RichText)
        body.setOpenExternalLinks(True)
        body.setWordWrap(True)
        body.setStyleSheet(f"font-size: {font_pt}pt;")
        lay.addWidget(body)
        return frame

    def _make_card(self, num: int, title: str, subtitle: str, accent: str) -> tuple[QFrame, QVBoxLayout]:
        """A numbered, accent-coloured card holding one entry point."""
        card = QFrame()
        card.setObjectName("coverCard")
        card.setStyleSheet(
            f"QFrame#coverCard {{ border: 1px solid rgba(255,255,255,35);"
            f" border-top: 3px solid {accent}; border-radius: 10px;"
            f" background-color: rgba(255,255,255,10); }}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(self._px(16), self._px(14), self._px(16), self._px(14))
        layout.setSpacing(self._px(10))

        header = QLabel(
            f'<span style="color:{accent}; font-size:{self._pt(16)}pt; font-weight:700;">{num}</span>'
            f'&nbsp;&nbsp;<span style="font-size:{self._pt(12)}pt; font-weight:600;">{title}</span>'
        )
        layout.addWidget(header)

        sub = QLabel(subtitle)
        sub.setWordWrap(True)
        sub.setStyleSheet("color: rgba(255,255,255,150);")
        layout.addWidget(sub)
        return card, layout

    def _build_template_card(self) -> QFrame:
        card, layout = self._make_card(
            1,
            "Template datasets",
            "Fastest way to try the GUI: download a ready-made example dataset.",
            _ACCENTS["template"],
        )
        # BMP text glyph (not a colour emoji) — renders as a stable black
        # symbol on Windows / macOS / Linux system fonts.
        template_btn = QPushButton("🐦‍⬛  Browse templates…")
        template_btn.setMinimumHeight(self._px(48))
        template_btn.clicked.connect(self._on_template)
        layout.addWidget(template_btn)
        for preview in self._build_template_previews():
            layout.addWidget(preview, alignment=Qt.AlignCenter)
        layout.addStretch()
        return card

    def _build_template_previews(self, limit: int | None = None) -> list[QLabel]:
        """Stacked preview images of the first few template datasets.

        Fills the otherwise empty lower half of card 1 with a taste of what
        "Browse templates…" opens. Animated previews are skipped — a still
        strip should not draw the eye away from the drop zone. Short screens
        show fewer (and smaller) previews so the cards stay readable.
        """
        if limit is None:
            limit = 3 if self._scale > 0.85 else (2 if self._scale > 0.7 else 1)
        previews: list[QLabel] = []
        for ds in DATASETS.values():
            if len(previews) >= limit:
                break
            name = ds.get("image", "")
            if not name or name.lower().endswith(".gif"):
                continue
            path = TEMPLATE_ASSETS_DIR / name
            if not path.exists():
                continue
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                continue
            label = QLabel()
            # Height cap is generous so near-square previews still get a
            # reasonable width; wide ones stay bound by the 210 px width.
            label.setPixmap(pixmap.scaled(self._px(210), self._px(150), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            label.setAlignment(Qt.AlignCenter)
            label.setToolTip(ds.get("name", ""))
            previews.append(label)
        return previews

    def _build_drop_card(self) -> QFrame:
        card, layout = self._make_card(
            2,
            "Drag &amp; drop",
            "Quick exploration: drop already-aligned media / feature / label files.",
            _ACCENTS["drop"],
        )

        layout_row = QHBoxLayout()
        layout_row.addWidget(QLabel("Several files of one stream are:"))
        self._drop_layout_combo = QComboBox()
        self._drop_layout_combo.addItem("Several cameras/mics filming one trial", "same_trial")
        self._drop_layout_combo.addItem("Several trials of one device (natural sort)", "multi_trial")
        self._drop_layout_combo.setToolTip(
            "Same trial: drop 2+ video files and they become cam-1, cam-2, ... of one trial "
            "(order them in the dialog that follows).\n"
            "Multiple trials: drop 2+ video/pose/audio/.nc files and they natural-sort into "
            "one trial each — a folder of trial001.mp4, trial002.mp4, ... works.\n"
            "Remembered for next time."
        )
        saved = getattr(self.app_state, "drop_layout", "same_trial")
        index = self._drop_layout_combo.findData(saved)
        self._drop_layout_combo.setCurrentIndex(index if index >= 0 else 0)
        self._drop_layout_combo.currentIndexChanged.connect(self._on_drop_layout_changed)
        layout_row.addWidget(self._drop_layout_combo, 1)
        layout.addLayout(layout_row)

        self._drop = _DropList(accent=_ACCENTS["drop"], min_height=self._px(160))
        layout.addWidget(self._drop, 1)

        # Recorded drops of the current project, newest first. Hidden without
        # a project (there is nothing to list); populated by _refresh_project_ui.
        self._reopen_row = QWidget()
        reopen_layout = QHBoxLayout(self._reopen_row)
        reopen_layout.setContentsMargins(0, 0, 0, 0)
        self._reopen_combo = QComboBox()
        self._reopen_combo.setToolTip("Drops made with this project folder set, kept under its sessions/ folder.")
        self._reopen_combo.activated.connect(self._on_reopen_drop)
        reopen_layout.addWidget(self._reopen_combo, 1)
        layout.addWidget(self._reopen_row)

        self._video_motion_cb = QCheckBox("Compute video motion  (video only)")
        self._video_motion_cb.setToolTip(
            "Adds a motion feature with a (time, camera) shape, so you can pick a camera "
            "in the Feature controls or view all cameras as a heatmap.\n"
            "Read from the compressed stream (bytes per frame, no decoding — instant).\n"
            "Does nothing if no video is dropped."
        )
        layout.addWidget(self._video_motion_cb)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._drop.clear_paths)
        layout.addWidget(clear_btn)
        return card

    def _build_custom_card(self) -> QFrame:
        card, layout = self._make_card(
            3,
            "Custom set-up",
            "Your own multi-trial data: point the loader at a session file "
            "(.nc / .nwb / .npz / pynapple folder) plus media folders. "
            "Run the wizard first if your data is not yet aligned.",
            _ACCENTS["custom"],
        )
        wizard_btn = QPushButton("🧙  Data wizard — prepare my data")
        wizard_btn.setMinimumHeight(self._px(48))
        wizard_btn.clicked.connect(self._on_wizard)
        layout.addWidget(wizard_btn)

        # Host for the IOWidget load panel (session/media path fields + Load
        # button). Borrowed on show, returned to the IO tab on hide.
        self._load_panel_borrowed = False
        self._load_host = QWidget()
        self._load_host_layout = QVBoxLayout(self._load_host)
        self._load_host_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._load_host)
        layout.addStretch()
        return card

    def _build_load_bar(self) -> QFrame:
        """Shared Load strip below the cards — one button for options 2 and 3.

        A drop auto-populates the custom set-up fields, so both paths end in
        the same load; a single button makes that explicit.
        """
        bar = QFrame()
        bar.setObjectName("loadBar")
        bar.setStyleSheet(
            "QFrame#loadBar { border: 1px solid rgba(255,255,255,35);"
            " border-radius: 10px; background-color: rgba(255,255,255,10); }"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(self._px(16), self._px(10), self._px(16), self._px(10))
        row.setSpacing(self._px(16))

        label = QLabel(
            f'<span style="color:{_ACCENTS["drop"]}; font-weight:700;">Drag and drop files (2)</span>'
            f' or <span style="color:{_ACCENTS["custom"]}; font-weight:700;">'
            "define custom multi-trial set-up (3)</span>"
            ", then click load."
        )
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        row.addWidget(label, 1)

        self._shared_load_btn = QPushButton("Load")
        self._shared_load_btn.setMinimumHeight(self._px(44))
        self._shared_load_btn.setMinimumWidth(self._px(220))
        self._shared_load_btn.clicked.connect(self._on_shared_load)
        row.addWidget(self._shared_load_btn)
        return bar

    # ------------------------------------------------------------------
    # Load panel borrow / return (pattern shared with top-bar popups)
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self._borrow_load_panel()

    def hideEvent(self, event):
        self._return_load_panel()
        super().hideEvent(event)

    def _borrow_load_panel(self):
        """Reparent the IOWidget load panel into the right column."""
        if self._load_panel_borrowed:
            return
        self._load_panel_borrowed = True
        io = self.io_widget
        io.load_buttons_row.hide()
        # The panel's own Load button is replaced by the shared load bar
        # below the cards (which serves both drag & drop and custom set-up).
        io.load_button.hide()
        self._load_host_layout.addWidget(io.load_panel)
        io.load_panel.show()

    def _return_load_panel(self):
        """Give the load panel back to the IO tab (it must outlive this dialog)."""
        if not self._load_panel_borrowed:
            return
        self._load_panel_borrowed = False
        io = self.io_widget
        self._load_host_layout.removeWidget(io.load_panel)
        io.load_button.show()
        io.load_buttons_row.show()
        # Re-insert at the top of the IO widget (original position).
        io.layout().insertWidget(0, io.load_panel)

    # ------------------------------------------------------------------
    # Option handlers
    # ------------------------------------------------------------------

    def _on_template(self):
        self.io_widget._on_select_template_clicked()
        self._close_if_loaded()

    def _on_wizard(self):
        self.io_widget._on_create_nc_clicked()
        # The wizard populates fields but may not auto-load; trigger a load if a
        # session path is now set.
        if getattr(self.app_state, "nc_file_path", None) and not self._is_loaded():
            self.io_widget._on_load_clicked()
        self._close_if_loaded()

    def _on_shared_load(self):
        """Shared Load button: dropped files (if any) win, else the custom fields."""
        if self._drop.paths:
            if not self._prepare_dropped():
                return
        self._maybe_offer_alignment_from_trials()
        self.io_widget._on_load_clicked()
        self._close_if_loaded()

    def _maybe_offer_alignment_from_trials(self):
        """Pynapple folder with a trials IntervalSet but no alignment NWB:
        offer to convert it into ``.ethograph/alignment.nwb`` once.

        The alignment NWB is the only trial-timing source the loader reads —
        a ``trials.npz`` is never consulted directly. This conversion (start/
        end plus any metadata columns → trials table) is the one sanctioned
        bridge, and it only happens with the user's explicit yes.
        """
        from qtpy.QtWidgets import QMessageBox

        from ethograph.io.nwb_alignment import alignment_from_trials_ep
        from ethograph.io.pynapple import find_trials_intervalset

        folder_str = getattr(self.app_state, "nc_file_path", None)
        if not folder_str:
            return
        folder = Path(folder_str)
        if not folder.is_dir():
            return
        sidecar = folder / ".ethograph" / "alignment.nwb"
        explicit = getattr(self.app_state, "nwb_file_path", None)
        if sidecar.exists() or (explicit and Path(explicit).exists()):
            return
        try:
            ep = find_trials_intervalset(folder)
        except Exception:  # noqa: BLE001 - a broken npz must not block loading
            logger.exception("Scanning for a trials IntervalSet failed")
            return
        if ep is None or len(ep) == 0:
            return

        answer = QMessageBox.question(
            self,
            "Create alignment from trials?",
            f"This folder has no alignment NWB, but contains a trials IntervalSet "
            f"({len(ep)} trials).\n\n"
            "Create .ethograph/alignment.nwb from it? Trial timing (and any "
            "per-trial metadata it carries) will come from that file from now on.\n\n"
            "Without it, the session loads as a single continuous recording.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            alignment_from_trials_ep(ep, sidecar)
            logger.info("Created %s from trials IntervalSet (%d trials)", sidecar, len(ep))
        except Exception as e:  # noqa: BLE001 - outermost GUI boundary
            logger.exception("Failed to create alignment from trials IntervalSet")
            notify_dialog(f"Could not create alignment.nwb:\n{e}", "error")

    def _prepare_dropped(self) -> bool:
        """Classify the dropped files and populate the IO fields from them.

        Returns True when the IO fields are ready to load, False when the user
        cancelled a follow-up prompt or preparation failed.
        """
        self._reopened_drop = None
        self._drop_tmp_dir = None
        dropped = list(self._drop.paths)
        buckets = classify_files(dropped)
        try:
            details = self._collect_drop_details(buckets)
            if details is None:
                return False  # user cancelled the follow-up prompt
            self._populate_io_from_buckets(buckets, details)
            if self._drop_tmp_dir is not None and self._project_dir() is not None:
                record_drop(self._drop_tmp_dir, dropped, self.app_state)
                self._refresh_project_ui()
        except Exception as e:  # noqa: BLE001 - outermost GUI boundary
            logger.exception("Failed to prepare dropped files")
            notify_dialog(f"Could not prepare dropped files:\n{e}", "error")
            return False
        return True

    # ------------------------------------------------------------------
    # Drag & drop → IO fields
    # ------------------------------------------------------------------

    def _collect_drop_details(self, buckets: dict[str, list[str]]) -> dict | None:
        """Ask once for the dropped values that cannot be inferred.

        Returns a dict of resolved details, or ``None`` if the user cancelled.
        Skips the prompt entirely when nothing needs asking.
        """
        need_npy_sr = bool(buckets["npy"])
        ambiguous_pose = any(Path(p).suffix.lower() in AMBIGUOUS_POSE_EXTENSIONS for p in buckets["pose"])
        # Pose without a video (image + pose drop): the fps cannot be read
        # from anywhere, so it must be asked — unless every pose file is a
        # movement dataset carrying its own.
        need_pose_fps = (
            bool(buckets["pose"]) and not buckets["video"] and not all(_pose_file_fps(p) for p in buckets["pose"])
        )
        # Videos with an embedded audio track: ask whether to extract it (npy
        # drops ignore audio, so don't offer it there).
        audio_track_videos = [] if need_npy_sr else [v for v in buckets["video"] if has_embedded_audio(v)]
        if not need_npy_sr and not ambiguous_pose and not need_pose_fps and not audio_track_videos:
            return {
                "data_sr": None,
                "source_software": None,
                "pose_fps": None,
                "extract_audio": False,
                "audio_track_videos": [],
            }

        npy_name = Path(buckets["npy"][0]).name if need_npy_sr else None
        dlg = _DropDetailsDialog(
            need_npy_sr,
            npy_name,
            ambiguous_pose,
            need_pose_fps,
            audio_track_videos=audio_track_videos,
            extract_audio_default=not buckets["audio"],
            parent=self,
        )
        if not dlg.exec_():
            return None
        return {
            "data_sr": dlg.data_sr(),
            "source_software": dlg.source_software(),
            "pose_fps": dlg.pose_fps(),
            "extract_audio": dlg.extract_audio(),
            "audio_track_videos": audio_track_videos,
        }

    @staticmethod
    def _is_multi_trial_drop(buckets: dict[str, list[str]]) -> bool:
        """Whether several files sit in one stream — the signal that this is
        several trials of one device, not several cameras on one trial."""
        return any(len(buckets[k]) > 1 for k in ("video", "pose", "audio"))

    def _populate_io_from_buckets(self, buckets: dict[str, list[str]], details: dict):
        io = self.io_widget
        app_state = self.app_state

        videos = buckets["video"]
        poses = buckets["pose"]
        images = buckets["image"]

        if self._drop_layout() == "multi_trial" and self._is_multi_trial_drop(buckets):
            self._populate_io_from_multi_trial(buckets, details)
            return

        if buckets["npy"]:
            self._populate_io_from_npy(buckets, details)
            return

        no_media = not (buckets["session"] or videos or poses or buckets["audio"])
        if no_media and (buckets["ephys"] or buckets["neurons"]):
            self._populate_io_from_ephys(buckets)
            return
        if images and no_media:
            raise RuntimeError(
                "An image alone has no time axis — drop it together with a pose, video, audio or session file."
            )

        cam_map: list[tuple[str, str | None]] = []  # (video|image, pose|None)
        if videos:
            # The camera-order dialog only matters when pose files must be paired
            # with videos; videos alone get arbitrary cam-1, cam-2, … names.
            if poses and (len(videos) > 1 or len(poses) > 1):
                dlg = _CamMatchDialog(videos, poses, parent=self)
                if not dlg.exec_():
                    raise RuntimeError("Camera assignment cancelled.")
                videos = dlg.ordered_videos()
                poses = dlg.ordered_poses()
            for i, v in enumerate(videos):
                cam_map.append((v, poses[i] if i < len(poses) else None))
        elif poses:
            # Pose without video: a still image (when dropped) stands in as the
            # "camera" background (static view, pose animates on top of it).
            # With no image either, the pose stands alone — no camera view;
            # position/confidence load as plottable features instead.
            for i, p in enumerate(poses):
                cam_map.append((images[min(i, len(images) - 1)] if images else None, p))

        # A .nc holding trials, an .nwb, an .npz or a folder is a session as
        # it is; a plain .nc is one more feature source (see _feature_entries).
        nc_sessions = [s for s in buckets["session"] if Path(s).suffix.lower() == ".nc" and not _is_trial_tree(s)]
        other_sessions = [s for s in buckets["session"] if s not in nc_sessions]
        audio_files = list(buckets["audio"])
        has_media = bool(cam_map or audio_files)
        feature_entries = self._feature_entries(cam_map, nc_sessions)
        needs_session_file = not other_sessions and self._session_nc_is_written(feature_entries)
        if has_media or needs_session_file:
            # Fresh per-drop temp dir so throwaway files never share a
            # .ethograph/local_settings.yaml with a previous drop.
            self._drop_tmp_dir = self._prepare_drop_dir()
        if details.get("extract_audio") and details.get("audio_track_videos"):
            # The user opted to pull the videos' embedded audio: each track
            # becomes a throwaway .wav that joins the dropped audio files.
            audio_files += self._extract_video_audio(details["audio_track_videos"])
        nwb_path: Path | None = None
        if has_media:
            # Media always needs a synthesised alignment (a session file's
            # folder usually has no .ethograph sidecar); the loader picks it
            # up via nwb_file_path.
            nwb_path = self._build_tmp_alignment(cam_map, audio_files, details)
            app_state.nwb_file_path = str(nwb_path)
        if other_sessions:
            # A real session file was provided — use it directly.
            app_state.nc_file_path = other_sessions[0]
        elif feature_entries:
            # Every dropped .nc — and any pose file with no camera to draw on —
            # is the session's data: position/confidence/… become catalog
            # features, one .nc per drop, stacked on a `camera` dim when there
            # are several. The overlay reads the same files through the
            # alignment's pose streams, so a file can be both.
            app_state.nc_file_path = str(self._compute_session_nc(feature_entries, details))
        elif cam_map and self._video_motion_cb.isChecked():
            # Video motion requested → the session is an xarray .nc holding a
            # (time, camera) motion feature; media still comes from the tmp
            # alignment above. Feature/camera dropdown + heatmap come for free.
            app_state.nc_file_path = str(self._compute_video_motion_nc(cam_map))
        else:
            app_state.nc_file_path = str(nwb_path)

        # Drag & drop never takes a metadata table — setting nc_file_path above
        # reloads local settings, which can restore a stale metadata_path (e.g.
        # a previous drop's tmp alignment NWB). Clear it so the drop's own
        # trial timing wins; sidecar TSV discovery via source_path still works.
        app_state.metadata_path = None
        # Same story for an explicit labels override from a previous drop.
        app_state.labels_import_path = None

        # A drop defines a fresh single-trial session, so it is authoritative:
        # clear every media path first, then set only the ones actually dropped.
        # Otherwise a stale path from a previous drop survives (these are
        # persisted SCOPE_LOCAL fields, and setting nc_file_path above reloaded
        # the dropped folder's local_settings.yaml, which can restore a previous
        # session's ephys/kilosort folders) and _validate_media_files warns that
        # the new alignment has no matching media.
        app_state.video_folder = None
        app_state.audio_folder = None
        app_state.pose_folder = None
        app_state.ephys_path = None
        app_state.neurons_path = None
        app_state.image_paths = list(images)

        if videos:
            app_state.video_folder = str(Path(videos[0]).parent)
        n_camera_backed = sum(1 for v, _ in cam_map if v is not None)
        if n_camera_backed > 1:
            # Several videos (or image-backed pose cams) dropped = one trial
            # filmed by multiple cameras in parallel, so show every camera as
            # its own view (not just cam-1). Devices match the
            # `video_cam-{i+1}` streams synthesised above. Standalone pose
            # entries have no camera stream, so no view is created for them.
            app_state.primary_camera = "cam-1"
            app_state.extra_cameras = [f"cam-{i + 1}" for i in range(1, n_camera_backed)]
        if poses:
            app_state.pose_folder = str(Path(poses[0]).parent)
            app_state.source_software = details.get("source_software")
        if audio_files:
            app_state.audio_folder = str(Path(audio_files[0]).parent)
        if buckets["ephys"]:
            app_state.ephys_path = buckets["ephys"][0]
        if buckets["neurons"]:
            app_state.neurons_path = buckets["neurons"][0]
        if buckets["labels"] and hasattr(io, "import_labels_checkbox"):
            # Persisted per-dataset override (SCOPE_LOCAL) — read by
            # IOWidget.get_import_labels_path() at load time. Explicit, so it
            # is never re-guessed from the .nc filename on later loads.
            app_state.labels_import_path = buckets["labels"][0]
            io.import_labels_checkbox.setChecked(True)

    def _populate_io_from_multi_trial(self, buckets: dict[str, list[str]], details: dict) -> None:
        """Several files of one device = one file per trial, natural-sort paired.

        Drives the Data wizard's single-camera "Pair" route headlessly:
        :func:`~ethograph.gui.wizard_multi_builder.build_multi_trial_dt` builds
        the ``TrialTree`` (embedding pose, same as the wizard) and the
        ``.ethograph/alignment.nwb`` sidecar next to it (video/audio stay pure
        media, referenced from there). Feature ``.nc`` files and ephys are not
        supported in this drop path yet — use the Data wizard for those.
        """
        from ethograph.gui.video_manager import probe_video
        from ethograph.gui.wizard_multi_builder import build_multi_trial_dt
        from ethograph.gui.wizard_state import WizardState

        videos = natsort.natsorted(buckets["video"])
        poses = natsort.natsorted(buckets["pose"])
        audios = natsort.natsorted(buckets["audio"])
        counts = {k: len(v) for k, v in (("video", videos), ("pose", poses), ("audio", audios)) if v}
        if not counts:
            raise RuntimeError("Multiple trials needs at least one video, pose or audio file.")
        if len(set(counts.values())) > 1:
            detail = ", ".join(f"{k}={n}" for k, n in counts.items())
            raise RuntimeError(f"Multiple trials: streams disagree on file count ({detail}).")
        n_trials = next(iter(counts.values()))

        state = WizardState(mode="pair")
        columns: dict[str, list] = {"trial": list(range(1, n_trials + 1))}
        if videos:
            fps = probe_video(videos[0]).fps
            if not fps:
                raise RuntimeError(f"Could not read frame rate from {Path(videos[0]).name}.")
            state.video.enabled = True
            state.video.files = videos
            state.video.file_mode = "aligned_to_trial"
            state.video.fps = fps
            columns["video_cam-1"] = videos
        if poses:
            pose_fps = state.video.fps if videos else details.get("pose_fps")
            if not pose_fps:
                raise RuntimeError("A pose file dropped without a video needs its frame rate.")
            state.pose.enabled = True
            state.pose.files = poses
            state.pose.file_mode = "aligned_to_trial"
            state.pose.fps = pose_fps
            state.pose.source_software = details.get("source_software") or "DeepLabCut"
            columns["pose_cam-1"] = poses
        if audios:
            rate, _duration = _audio_info(audios[0])
            state.audio.enabled = True
            state.audio.files = audios
            state.audio.file_mode = "aligned_to_trial"
            state.audio.audio_sr = rate
            columns["audio_mic-1"] = audios

        state.trial_table = pd.DataFrame(columns)
        self._drop_tmp_dir = self._prepare_drop_dir()
        state.session_dir = str(self._drop_tmp_dir)
        state.output_path = str(self._drop_tmp_dir / "session.nc")

        dt = build_multi_trial_dt(state)
        dt.to_netcdf(state.output_path)

        app_state = self.app_state
        app_state.nc_file_path = state.output_path
        app_state.metadata_path = None
        app_state.labels_import_path = None
        app_state.video_folder = None
        app_state.audio_folder = None
        app_state.pose_folder = None
        app_state.ephys_path = None
        app_state.neurons_path = None
        app_state.image_paths = []
        app_state.nwb_alignment = state.nwb_alignment
        if videos:
            app_state.video_folder = str(Path(videos[0]).parent)
        if poses:
            app_state.pose_folder = str(Path(poses[0]).parent)
            app_state.source_software = state.pose.source_software
        if audios:
            app_state.audio_folder = str(Path(audios[0]).parent)

    def _populate_io_from_npy(self, buckets: dict[str, list[str]], details: dict):
        """Convert a dropped .npy into a .nc (the one case that must persist data).

        A numpy array carries no time axis, so its sample rate is asked for via a
        follow-up prompt; an optional dropped video supplies fps and alignment.
        Other buckets are ignored here — npy is a standalone feature source.
        """
        from ethograph.gui.video_manager import probe_video
        from ethograph.io.data_loader import wizard_single_from_npy_file

        npy_path = buckets["npy"][0]
        video_path = buckets["video"][0] if buckets["video"] else None
        fps = probe_video(video_path).fps if video_path else None

        output_path = str(Path(npy_path).with_suffix(".nc"))
        dt = wizard_single_from_npy_file(
            video_path=video_path,
            fps=fps,
            npy_path=npy_path,
            data_sr=details["data_sr"],
            output_nc_path=output_path,
        )
        dt.to_netcdf(output_path)

        app_state = self.app_state
        app_state.nc_file_path = output_path
        app_state.metadata_path = None
        app_state.labels_import_path = None
        app_state.image_paths = list(buckets["image"])
        if video_path:
            app_state.video_folder = str(Path(video_path).parent)

    def _populate_io_from_ephys(self, buckets: dict[str, list[str]]):
        """Build a bare session .nc for an ephys- and/or kilosort-only drop.

        Ephys traces and neurons load from their own paths; this only writes a
        minimal session (with a single-trial alignment) so the loader has an
        anchor, mirroring the old ephys wizard dialog.
        """
        from ethograph.io.data_loader import wizard_single_from_ephys

        ephys_path = buckets["ephys"][0] if buckets["ephys"] else None
        neurons_path = buckets["neurons"][0] if buckets["neurons"] else None
        anchor = ephys_path or neurons_path
        output_path = str(Path(anchor).with_suffix(".nc")) if ephys_path else str(Path(anchor) / "session.nc")

        dt = wizard_single_from_ephys(
            output_nc_path=output_path,
            ephys_path=ephys_path,
            neurons_path=neurons_path,
        )
        dt.to_netcdf(output_path)

        app_state = self.app_state
        app_state.nc_file_path = output_path
        app_state.metadata_path = None
        app_state.labels_import_path = None
        app_state.image_paths = list(buckets["image"])
        if ephys_path:
            app_state.ephys_path = ephys_path
        if neurons_path:
            app_state.neurons_path = neurons_path

    def _extract_video_audio(self, videos: list[str]) -> list[str]:
        """Decode each video's audio track to a cached .wav behind a busy dialog."""
        from ethograph.gui.dialog_busy_progress import BusyProgressDialog

        dlg = BusyProgressDialog("Extracting audio from video…", parent=self)
        # Cached centrally (not in the per-drop temp dir): the extract is media,
        # carries no local_settings, and re-dropping the same video should not
        # decode it again.
        paths, error = dlg.execute(lambda: [str(ensure_extracted_audio(v)) for v in videos])
        if error or paths is None:
            raise RuntimeError(f"Could not extract audio from video: {error}")
        return paths

    def _compute_video_motion_nc(self, cam_map) -> Path:
        """Compute per-camera motion energy behind a busy dialog and return the .nc."""
        from ethograph.gui.dialog_busy_progress import BusyProgressDialog

        dlg = BusyProgressDialog("Computing video motion…", parent=self)
        nc_path, error = dlg.execute(self._build_video_motion_nc, cam_map, self._drop_tmp_dir)
        if error or nc_path is None:
            raise RuntimeError(f"Could not compute video motion: {error}")
        return nc_path

    @staticmethod
    def _build_video_motion_nc(cam_map, out_dir: Path) -> Path:
        """Write a ``(time, camera)`` video-motion feature to a throwaway .nc.

        One motion trace per dropped video, stacked on a ``camera`` dim
        (values ``cam-1``, ``cam-2``, … matching the alignment's video streams),
        so the catalog offers a camera dropdown and a heatmap view for free.
        The trace is the compressed stream's bytes per frame
        (:func:`~ethograph.features.movement.extract_packet_motion`): no
        decoding, so the drop loads in a second instead of a minute per video.
        """
        import numpy as np

        from ethograph.features.movement import extract_packet_motion
        from ethograph.gui.video_manager import probe_video

        motions: list[np.ndarray] = []
        cam_names: list[str] = []
        fps_used: float | None = None
        for i, (video, _pose) in enumerate(cam_map):
            fps = probe_video(video).fps
            if not fps:
                raise RuntimeError(f"Could not read frame rate from {Path(video).name}.")
            da = extract_packet_motion(video, fps=fps)
            motions.append(np.asarray(da.values, dtype=float))
            cam_names.append(f"cam-{i + 1}")
            fps_used = fps

        max_len = max(len(m) for m in motions)
        arr = np.full((max_len, len(motions)), np.nan)
        for j, m in enumerate(motions):
            arr[: len(m), j] = m
        time = np.arange(max_len) / fps_used

        ds = xr.Dataset(
            {"video_motion": (["time", "camera"], arr)},
            # ``individuals`` is required by TrialTree validation even though
            # motion energy is not per-individual.
            coords={"time": time, "camera": cam_names, "individuals": ["individual 1"]},
        )
        ds.attrs["fps"] = fps_used

        out_path = out_dir / f"video_motion-{uuid4().hex[:8]}.nc"
        ds.to_netcdf(out_path)
        return out_path

    def _feature_entries(
        self, cam_map: list[tuple[str | None, str | None]], nc_sessions: list[str]
    ) -> list[_FeatureEntry]:
        """What feeds the session dataset, and what stays on the video.

        Every dropped ``.nc`` is a feature source, named after its camera when
        it is paired with one (``cam-N``, the alignment's stream) and after
        its file otherwise. A pose file with no video is a feature source too
        (that is all it can be). A pose ``.nc`` paired with a video stays its
        overlay only while its positions plausibly are that video's pixels
        (:func:`positions_fit_frame`); otherwise the pairing is dropped in
        *cam_map* — features, never drawn — and the user is told.
        """
        from ethograph.gui.video_manager import probe_video

        entries: list[_FeatureEntry] = []
        for i, (video, pose) in enumerate(cam_map):
            if pose is None:
                continue
            camera = f"cam-{i + 1}"
            if video is None:
                entries.append(_FeatureEntry(camera, pose, None))
                continue
            if Path(pose).suffix.lower() != ".nc":
                continue  # a tracking tool's own file with a video: overlay only, as before
            if Path(video).suffix.lower() in IMAGE_EXTENSIONS:
                entries.append(_FeatureEntry(camera, pose, None))
                continue
            probe = probe_video(video)
            entries.append(_FeatureEntry(camera, pose, probe.fps or None))
            if probe.width and probe.height:
                with xr.open_dataset(pose) as ds:
                    fits = positions_fit_frame(ds, probe.width, probe.height)
                if not fits:
                    cam_map[i] = (video, None)
                    notify(
                        f"{Path(pose).name}: positions are not in {Path(video).name}'s pixels "
                        f"({probe.width}×{probe.height}) — loaded as features, not drawn on the video.",
                        "warning",
                    )
        for nc in nc_sessions:
            entries.append(_FeatureEntry(Path(nc).stem, nc, None))
        return entries

    @staticmethod
    def _session_nc_is_written(entries: list[_FeatureEntry]) -> bool:
        """A single ``.nc`` is used in place; anything else is written to the drop dir."""
        return bool(entries) and not (len(entries) == 1 and Path(entries[0].path).suffix.lower() == ".nc")

    def _compute_session_nc(self, entries: list[_FeatureEntry], details: dict) -> Path:
        """The session ``.nc`` for *entries*: the file itself when it is one
        ``.nc``, else the combined file, built behind a busy dialog."""
        if not self._session_nc_is_written(entries):
            return Path(entries[0].path)
        from ethograph.gui.dialog_busy_progress import BusyProgressDialog

        dlg = BusyProgressDialog("Reading pose data…", parent=self)
        nc_path, error = dlg.execute(
            self._build_session_nc,
            entries,
            details.get("source_software"),
            details.get("pose_fps"),
            self._drop_tmp_dir,
        )
        if error or nc_path is None:
            raise RuntimeError(f"Could not read pose data: {error}")
        return nc_path

    @staticmethod
    def _build_session_nc(
        entries: list[_FeatureEntry], source_software: str | None, fps: float | None, out_dir: Path
    ) -> Path:
        """Write the drop's feature sources as one plottable ``.nc``.

        A ``.nc`` is opened as it is, a tracking tool's file is converted by
        movement (which needs *fps*, the drop dialog's answer). A file
        without a rate of its own takes its camera's, then *fps*. Several
        stack on a ``camera`` dim matching the alignment's ``pose_cam-N``
        streams (:func:`concat_on_camera`).
        """
        datasets = []
        for entry in entries:
            ds = _open_pose_dataset(entry.path, source_software, fps)
            if not ds.attrs.get("fps") and (entry.fps or fps):
                ds.attrs["fps"] = float(entry.fps or fps)
            datasets.append(ds)
        ds = concat_on_camera(datasets, [e.camera for e in entries])
        if not ds.attrs.get("fps"):
            raise RuntimeError("A pose file dropped without a video needs its frame rate.")
        out_path = out_dir / f"session-{uuid4().hex[:8]}.nc"
        ds.to_netcdf(out_path)
        return out_path

    def _build_tmp_alignment(self, cam_map, audio_files, details: dict | None = None) -> Path:
        """Create a single-trial alignment.tmp.nwb from loose media files.

        A ``cam_map`` entry may pair a pose file with a still image instead of
        a video (image + pose drop): the image becomes a static ``video_cam-N``
        stream at the user-provided pose fps, and the session duration comes
        from the pose file itself. A pose dropped with no video *and* no image
        gets a ``pose_cam-N`` stream alone (no camera view exists for it).
        """
        import pandas as pd

        from ethograph.gui.video_manager import probe_video
        from ethograph.io.nwb_alignment import align_media_from_streams

        if not cam_map and not audio_files:
            raise RuntimeError("No media files to build an alignment from.")

        details = details or {}
        streams: list[dict] = []
        stop_time = 0.0
        for i, (video, pose) in enumerate(cam_map):
            if video is None:
                # Standalone pose (no video, no image): the pose stream is the
                # camera slot's only content, at the user-provided frame rate.
                pose_fps = details.get("pose_fps") or (_pose_file_fps(pose) if pose else None)
                if not pose_fps or not pose:
                    raise RuntimeError("A pose file dropped without a video needs its frame rate.")
                duration = _pose_duration(pose, details.get("source_software"), pose_fps)
                stop_time = max(stop_time, duration)
                streams.append({"name": f"pose_cam-{i + 1}", "files": [pose], "rate": pose_fps})
                logger.info(
                    "Drop alignment cam-%d: standalone pose %s at %s fps",
                    i + 1,
                    Path(pose).name,
                    pose_fps,
                )
                continue
            if Path(video).suffix.lower() in IMAGE_EXTENSIONS:
                pose_fps = details.get("pose_fps") or (_pose_file_fps(pose) if pose else None)
                if not pose_fps or not pose:
                    raise RuntimeError("An image-backed camera needs a pose file and its frame rate.")
                duration = _pose_duration(pose, details.get("source_software"), pose_fps)
                stop_time = max(stop_time, duration)
                streams.append({"name": f"video_cam-{i + 1}", "files": [video], "rate": pose_fps})
                streams.append({"name": f"pose_cam-{i + 1}", "files": [pose], "rate": pose_fps})
                logger.info(
                    "Drop alignment cam-%d: static image %s, pose %s at %s fps",
                    i + 1,
                    Path(video).name,
                    Path(pose).name,
                    pose_fps,
                )
                continue
            probe = probe_video(video)
            fps = probe.fps
            if not fps:
                raise RuntimeError(f"Could not read frame rate from {Path(video).name}.")
            duration = probe.nframes / fps if probe.nframes else 0.0
            stop_time = max(stop_time, duration)
            streams.append({"name": f"video_cam-{i + 1}", "files": [video], "rate": fps})
            logger.info(
                "Drop alignment cam-%d: assumed fps=%s (from %s)%s",
                i + 1,
                fps,
                Path(video).name,
                f"; pose {Path(pose).name} uses this fps" if pose else "",
            )
            if pose:
                # Pose shares the matching video's frame rate (per spec).
                streams.append({"name": f"pose_cam-{i + 1}", "files": [pose], "rate": fps})

        for j, audio in enumerate(audio_files):
            rate, duration = _audio_info(audio)
            stop_time = max(stop_time, duration)
            streams.append(
                {
                    "name": f"audio_mic-{j + 1}",
                    "files": [audio],
                    "rate": rate,
                    "starting_time": 0.0,
                }
            )

        if stop_time <= 0.0:
            raise RuntimeError("Could not determine session duration from the dropped media.")

        trials = pd.DataFrame({"trial": [1], "start_time": [0.0], "stop_time": [stop_time]})

        out_path = self._drop_tmp_dir / f"alignment-{uuid4().hex[:8]}.tmp.nwb"
        align_media_from_streams(trials, streams, out_path)
        return out_path

    def _prepare_drop_dir(self) -> Path:
        """Return a fresh, empty per-drop dir for the synthesised alignment/.nc.

        Each drop gets its OWN subdirectory so its ``.ethograph/local_settings.yaml``
        starts empty — a shared directory would leak a previous drop's panel layout
        (e.g. one saved with no video panel) into the next drop, so dropped media
        would silently fail to appear.

        With a project folder set the dir is ``{project}/sessions/{timestamp}``
        and is kept (that is what a reopen reads). Otherwise it is a throwaway
        under the system temp dir: older drop dirs are removed best-effort; a
        dir whose files are still open (Windows locks HDF5) is simply left.
        """
        project = self._project_dir()
        if project is not None:
            return new_drop_dir(project)
        base = tmp_alignment_base()
        base.mkdir(parents=True, exist_ok=True)
        for stale in base.iterdir():
            try:
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                else:
                    stale.unlink()
            except OSError:
                pass
        drop_dir = base / uuid4().hex[:8]
        drop_dir.mkdir(parents=True, exist_ok=True)
        return drop_dir

    # ------------------------------------------------------------------
    # Loaded-state helpers
    # ------------------------------------------------------------------

    def _is_loaded(self) -> bool:
        return bool(getattr(self.app_state, "ready", False)) or (getattr(self.app_state, "dt", None) is not None)

    def _close_if_loaded(self):
        if self._is_loaded():
            self.accept()


def show_cover_page(shell) -> bool:
    """Run the start dialog before the (still hidden) main window is shown.

    Returns True if the GUI should open: a dataset was loaded (or is already
    loaded). Returns False when the user closed the dialog, meaning the app
    should exit without showing the main window.
    """
    meta = getattr(shell, "meta_widget", None)
    io_widget = getattr(meta, "io_widget", None)
    if io_widget is None:
        return True
    app_state = io_widget.app_state
    if getattr(app_state, "ready", False) or getattr(app_state, "dt", None) is not None:
        return True
    page = CoverPage(shell, io_widget)
    shell._cover_page = page
    # Size from the screen (the shell is still hidden, its geometry pending);
    # wide enough that the custom-loader path fields are readable. Short screens
    # get a larger fraction of the available height — 75% of a 768 px laptop
    # screen leaves the cards clipped, while 75% of a 1440 px one is plenty.
    screen = _available_geometry(page)
    height_ratio = 0.75 if screen.height() >= _REFERENCE_SCREEN_HEIGHT else 0.92
    width = min(int(screen.width() * 0.85), screen.width())
    height = min(int(screen.height() * height_ratio), screen.height())
    page.resize(width, height)
    # Centre on the available area so a full-height window is not pushed under
    # the taskbar (the dialog stays freely resizable from any edge).
    page.move(
        screen.x() + (screen.width() - width) // 2,
        screen.y() + (screen.height() - height) // 2,
    )
    return page.exec_() == QDialog.Accepted
