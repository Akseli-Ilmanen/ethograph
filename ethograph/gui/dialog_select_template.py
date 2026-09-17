"""Dialog for selecting a template dataset to pre-fill IO paths."""

import logging
import traceback
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

from qtpy.QtCore import QSize, Qt, QThread, Signal  # noqa: E402
from qtpy.QtGui import QMovie, QPixmap  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
)

from ethograph.datasets import (  # noqa: E402
    DATASETS,
    DOWNLOAD_BASE,
    are_prediction_runs_downloaded,
    are_video_features_downloaded,
    dataset_dir,
    get_gui_assets,
    get_prediction_run_assets,
    get_video_feature_assets,
    is_dataset_downloaded,
    resolve_dataset_paths,
    video_features_size_mb,
)
from ethograph.gui.notify import notify_dialog  # noqa: E402
from ethograph.utils.download import (  # noqa: E402
    download_assets,
    download_prediction_runs,
    download_template_local_settings,
    download_video_features,
    ensure_alignment_nwb,
    ensure_default_configs,
    write_example_configs,
)

TEMPLATE_ASSETS_DIR = Path(__file__).parent / "assets" / "templates"

try:
    import audioio as _audioio  # noqa: F401

    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

# Backward-compat re-exports used by test code
_DOWNLOAD_BASE = DOWNLOAD_BASE
TEMPLATES = list(DATASETS.values())


def _template_dir(key_or_dict) -> Path:
    """Backward-compat helper — accepts a key or a legacy template dict."""
    if isinstance(key_or_dict, str):
        return dataset_dir(key_or_dict)
    return DOWNLOAD_BASE / key_or_dict["folder"]


def _template_downloaded(key_or_dict) -> bool:
    """Backward-compat helper — accepts a key or a legacy template dict."""
    if isinstance(key_or_dict, str):
        return is_dataset_downloaded(key_or_dict)
    return is_dataset_downloaded(key_or_dict["dataset_key"])


def _resolve_template_paths(key_or_dict) -> dict:
    """Backward-compat helper — accepts a key or a legacy template dict."""
    if isinstance(key_or_dict, str):
        return resolve_dataset_paths(key_or_dict)
    return resolve_dataset_paths(key_or_dict["dataset_key"])


def _build_alignment_nwb(key_or_dict) -> None:
    """Backward-compat helper — accepts a key or a legacy template dict."""
    if isinstance(key_or_dict, str):
        ensure_alignment_nwb(key_or_dict)
    else:
        ensure_alignment_nwb(key_or_dict["dataset_key"])


def _n_prediction_run_files(key: str) -> int:
    return sum(len(files) for files in get_prediction_run_assets(key).values())


class _DownloadWorker(QThread):
    """Downloads template assets in a background thread."""

    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, key: str, video_features: bool = False):
        super().__init__()
        self._key = key
        self._video_features = video_features
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        info = DATASETS[self._key]
        n_base = len(get_gui_assets(self._key))
        try:
            download_assets(
                release_tag=info["release_tag"],
                assets=get_gui_assets(self._key),
                dest=dataset_dir(self._key),
                on_progress=self.progress.emit,
                cancelled=lambda: self._cancelled,
            )
            # Shipped prediction runs are part of the dataset, not an option:
            # tiny, and the point of the template is comparing them.
            if not self._cancelled:
                download_prediction_runs(
                    self._key,
                    on_progress=lambda count, name: self.progress.emit(n_base + count, name),
                    cancelled=lambda: self._cancelled,
                )
            n_runs = _n_prediction_run_files(self._key)
            if self._video_features and not self._cancelled:
                download_video_features(
                    self._key,
                    on_progress=lambda count, name: self.progress.emit(n_base + n_runs + count, name),
                    cancelled=lambda: self._cancelled,
                )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        if not self._cancelled:
            self.finished.emit()


class TemplateDialog(QDialog):
    """Popup showing template datasets as clickable cards with images."""

    _CARDS_PER_ROW = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_template = None
        self._video_feature_boxes: dict[str, QCheckBox] = {}
        self.setWindowTitle("Select Templates")

        outer = QVBoxLayout()
        outer.setSpacing(12)
        self.setLayout(outer)

        for i, key in enumerate(DATASETS):
            if i % self._CARDS_PER_ROW == 0:
                row = QHBoxLayout()
                row.setSpacing(12)
                outer.addLayout(row)
            card = self._create_card(key)
            row.addWidget(card)

    def _create_card(self, key: str) -> QFrame:
        ds = DATASETS[key]
        audio_locked = ds.get("has_audio", False) and not _AUDIO_AVAILABLE

        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        if audio_locked:
            card.setCursor(Qt.ArrowCursor)
            card.setStyleSheet("QFrame { background-color: palette(mid); color: gray; }")
        else:
            card.setCursor(Qt.PointingHandCursor)
            card.setStyleSheet("QFrame:hover { background-color: palette(midlight); }")

        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(8, 8, 8, 8)
        card.setLayout(card_layout)

        image_label = QLabel()
        image_label.setFixedSize(220, 160)
        image_label.setAlignment(Qt.AlignCenter)
        self._load_image_into_label(image_label, ds.get("image", ""))
        if audio_locked:
            image_label.setEnabled(False)
        card_layout.addWidget(image_label, alignment=Qt.AlignCenter)

        text_label = QLabel(ds["name"])
        text_label.setWordWrap(True)
        text_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(text_label)

        link_url = ds.get("paper_url") or ds.get("dataset_url")
        if link_url:
            link_text = "Open dataset" if ds.get("dataset_url") and not ds.get("paper_url") else "Open paper"
            link = QPushButton(link_text)
            link.setFlat(True)
            link.setCursor(Qt.PointingHandCursor)
            link.setStyleSheet("color: palette(link); text-decoration: underline;")
            link.clicked.connect(lambda _checked, u=link_url: webbrowser.open(u))
            card_layout.addWidget(link, alignment=Qt.AlignCenter)

        status = QLabel()
        status.setAlignment(Qt.AlignCenter)
        if audio_locked:
            status.setText('Requires audio — uv pip install "ethograph[gui,audio]"')
            status.setStyleSheet("color: #c07000; font-style: italic;")
            status.setWordWrap(True)
        elif is_dataset_downloaded(key):
            status.setText("Downloaded")
            status.setStyleSheet("color: green; font-weight: bold;")
        else:
            status.setText(f"Click to download (~{ds['size_mb']} MB)")
            status.setStyleSheet("color: gray;")
        card_layout.addWidget(status)

        if ds.get("video_feature_folders") and not audio_locked:
            folders = ", ".join(ds["video_feature_folders"])
            if are_video_features_downloaded(key):
                box = QCheckBox("Video features downloaded")
                box.setChecked(True)
                box.setEnabled(False)
            else:
                box = QCheckBox(f"Download video features (~{video_features_size_mb(key)} MB)")
            box.setToolTip(
                f"Per-video embeddings ({folders}) saved next to the dataset.\n"
                "Import a folder via add panel → import to view it as a heatmap."
            )
            card_layout.addWidget(box, alignment=Qt.AlignCenter)
            self._video_feature_boxes[key] = box

        if not audio_locked:
            card.mousePressEvent = lambda event, k=key: self._on_card_clicked(k)
        return card

    def _load_image_into_label(self, label: QLabel, image_name: str) -> None:
        if not image_name:
            label.setText("(no preview)")
            return
        image_path = TEMPLATE_ASSETS_DIR / image_name
        if not image_path.exists():
            label.setText("Downloading preview...")
            return
        if image_path.suffix.lower() == ".gif":
            movie = QMovie(str(image_path))
            movie.setScaledSize(QSize(220, 160))
            label.setMovie(movie)
            movie.start()
        else:
            pixmap = QPixmap(str(image_path))
            label.setPixmap(pixmap.scaled(220, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _wants_video_features(self, key: str) -> bool:
        box = self._video_feature_boxes.get(key)
        return box is not None and box.isChecked() and not are_video_features_downloaded(key)

    def _on_card_clicked(self, key: str):
        if is_dataset_downloaded(key) and are_prediction_runs_downloaded(key) and not self._wants_video_features(key):
            self._finalize(key)
            return
        self._download_and_select(key)

    def _finalize(self, key: str):
        ds = DATASETS[key]
        if ds.get("nc_filename") is None and ds.get("audio_file"):
            self._generate_nc_from_audio(key)
            return
        ensure_default_configs()
        write_example_configs(key, dataset_dir(key))
        try:
            ensure_alignment_nwb(key)
        except Exception:
            logger.warning("Failed to obtain alignment NWB", exc_info=True)
        # Optional template settings (incl. panel layout) from the release:
        # lands in the dataset's .ethograph/local_settings.yaml, which the
        # normal settings autoload applies when the dataset opens.
        download_template_local_settings(key)
        self.selected_template = resolve_dataset_paths(key)
        self.accept()

    def _generate_nc_from_audio(self, key: str):
        ds = DATASETS[key]
        dest = dataset_dir(key)
        audio_path = str(dest / ds["audio_file"])
        nc_path = str(dest / (Path(ds["audio_file"]).stem + ".nc"))

        if not Path(nc_path).exists():
            try:
                from ethograph.io.data_loader import wizard_single_from_audio
                from ethograph.utils.audio import get_audio_sr

                audio_sr = get_audio_sr(audio_path)
                dt = wizard_single_from_audio(
                    video_path=None,
                    fps=30,
                    audio_path=audio_path,
                    audio_sr=audio_sr,
                )
                dt.to_netcdf(nc_path)
            except Exception as e:
                traceback.print_exc()
                notify_dialog(f"Failed to generate .nc from audio:\n{e}", "error", "Error", self)
                return

        download_template_local_settings(key)
        resolved = resolve_dataset_paths(key)
        resolved["nc_file_path"] = nc_path
        resolved["audio_folder"] = str(dest)
        logger.debug("Canary template resolved paths: %s", resolved)
        self.selected_template = resolved
        self.accept()

    def _download_and_select(self, key: str):
        video_features = self._wants_video_features(key)
        total = len(get_gui_assets(key)) + _n_prediction_run_files(key)
        if video_features:
            total += sum(len(names) for names in get_video_feature_assets(key).values())
        progress = QProgressDialog("Downloading example data...", "Cancel", 0, total, self)
        progress.setWindowTitle("Downloading")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = _DownloadWorker(key, video_features=video_features)

        def on_progress(count, name):
            if not progress.wasCanceled():
                progress.setValue(count)
                progress.setLabelText(f"Downloading {name}...")

        def on_finished():
            progress.close()
            worker.deleteLater()
            self._finalize(key)

        def on_error(msg):
            progress.close()
            worker.deleteLater()
            notify_dialog(msg, "warning", "Download Error", self)

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        progress.canceled.connect(worker.cancel)

        worker.start()
