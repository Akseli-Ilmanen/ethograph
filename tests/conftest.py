from pathlib import Path

import pytest

import ethograph as eto
import ethograph.utils.paths as paths_module
from ethograph.datasets import (
    DATASETS,
    DOWNLOAD_BASE,
    dataset_dir,
    get_gui_assets,
    is_dataset_downloaded,
    resolve_dataset_paths,
)
from ethograph.io.catalog import catalog_from_xarray
from ethograph.utils.download import (
    download_assets,
    ensure_alignment_nwb,
    ensure_default_configs,
    write_example_configs,
)

try:
    from qtpy.QtWidgets import QApplication, QMessageBox

    from ethograph.gui.app_state import ObservableAppState
    from ethograph.gui.main_window import EthographMainWindow
    from ethograph.gui.widgets_meta import MetaWidget

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

requires_gui = pytest.mark.skipif(not GUI_AVAILABLE, reason="Qt/pygfx not installed")


def pytest_addoption(parser):
    parser.addoption("--show", action="store_true", default=False, help="Show the GUI window for 15s after each test")


BIRDPARK_DIR = dataset_dir("birdpark")
BIRDPARK_NC = BIRDPARK_DIR / "copExpBP08_trim.nc"
MOLL_DIR = dataset_dir("moll2025")
MOLL_NC = MOLL_DIR / "Trial_data.nc"
MOLL_PYNAPPLE_DIR = DOWNLOAD_BASE / "Moll2025_pynapple"


def _skip_if_not_downloaded(key: str) -> None:
    if not is_dataset_downloaded(key):
        pytest.skip(f"{key} not downloaded")


def _ensure_alignment_nwb(key: str) -> None:
    """Build/fetch alignment.nwb only when it does not already exist."""
    nwb_path = dataset_dir(key) / ".ethograph" / "alignment.nwb"
    if not nwb_path.exists():
        ensure_alignment_nwb(key)


def _apply_template(meta, key: str, downsample: bool = False) -> None:
    _ensure_alignment_nwb(key)
    resolved = resolve_dataset_paths(key)

    io = meta.io_widget
    io._clear_all_line_edits()

    if resolved["nc_file_path"]:
        io.nc_file_path_edit.setText(resolved["nc_file_path"])
        meta.app_state.nc_file_path = resolved["nc_file_path"]
    if resolved["video_folder"]:
        io.video_folder_edit.setText(resolved["video_folder"])
        meta.app_state.video_folder = resolved["video_folder"]
    if resolved["audio_folder"]:
        io.audio_folder_edit.setText(resolved["audio_folder"])
        meta.app_state.audio_folder = resolved["audio_folder"]
    if resolved.get("pose_folder"):
        io.pose_folder_edit.setText(resolved["pose_folder"])
        meta.app_state.pose_folder = resolved["pose_folder"]
    if resolved.get("import_labels"):
        io.import_labels_checkbox.setChecked(True)
    if resolved.get("library_geometry"):
        meta.app_state.space_library_geometry = resolved["library_geometry"]

    io.downsample_checkbox.setChecked(downsample)
    if downsample:
        io.downsample_spin.setValue(100)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()


def _load_template_gui(gui, key: str, downsample: bool = False):
    _skip_if_not_downloaded(key)
    viewer, meta = gui
    _apply_template(meta, key, downsample=downsample)
    assert meta.app_state.ready, f"Failed to load {key}"
    return viewer, meta


# ---------------------------------------------------------------------------
# Dataset download — runs once per session
# ---------------------------------------------------------------------------


def _ensure_dataset(key: str):
    """Download example dataset if not already present."""
    if not is_dataset_downloaded(key):
        dest = dataset_dir(key)
        dest.mkdir(parents=True, exist_ok=True)
        download_assets(
            release_tag=DATASETS[key]["release_tag"],
            assets=get_gui_assets(key),
            dest=dest,
        )
        ensure_default_configs()
        write_example_configs(key, dest)


def pytest_configure(config):
    """Download required datasets if not already present (skipped in CI without GUI)."""
    if not GUI_AVAILABLE:
        return

    from ethograph.gui.plots_space import ensure_geometry_library

    ensure_geometry_library()

    _ensure_dataset("birdpark")
    assert BIRDPARK_NC.exists(), f"BirdPark NC not found after download: {BIRDPARK_NC}"

    _ensure_dataset("moll2025")
    assert MOLL_NC.exists(), f"Moll2025 NC not found after download: {MOLL_NC}"


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def moll2025_nc_path() -> str:
    return str(MOLL_NC)


@pytest.fixture
def birdpark_data_dir() -> str:
    return str(BIRDPARK_DIR)


@pytest.fixture
def moll2025_trial_tree(moll2025_nc_path):
    return eto.open(moll2025_nc_path)


@pytest.fixture
def moll2025_first_trial_ds(moll2025_trial_tree):
    return moll2025_trial_tree.itrial(0)


@pytest.fixture
def moll2025_catalog(moll2025_first_trial_ds, moll2025_trial_tree):
    return catalog_from_xarray(moll2025_first_trial_ds, moll2025_trial_tree)


@pytest.fixture
def moll2025_type_vars_dict(moll2025_catalog):
    """Backwards-compat fixture — prefer ``catalog`` in new tests."""
    return moll2025_catalog.to_type_vars_dict()


@pytest.fixture
def moll2025_label_dt(moll2025_trial_tree):
    return moll2025_trial_tree.get_label_dt()


@pytest.fixture
def app_state(qtbot, tmp_path):
    yaml_path = str(tmp_path / "test_gui_settings.yaml")
    state = ObservableAppState(yaml_path=yaml_path, auto_save_interval=999999)
    yield state
    state.stop_auto_save()


# ---------------------------------------------------------------------------
# Dialog suppression
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _suppress_dialogs(monkeypatch):
    """Suppress all GUI popups during tests via the SUPPRESS flag.

    Also patches QMessageBox as a safety net for any direct callers.
    """
    if not GUI_AVAILABLE:
        return
    import ethograph.gui.notify as _notify_mod

    monkeypatch.setattr(_notify_mod, "SUPPRESS", True)

    def _noop(*a, **kw):
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "critical", _noop)
    monkeypatch.setattr(QMessageBox, "warning", _noop)
    monkeypatch.setattr(QMessageBox, "information", _noop)


@pytest.fixture(autouse=True)
def _hermetic_global_settings(monkeypatch, tmp_path):
    """Never let a test touch the real ``~/.ethograph/gui_settings.yaml``.

    ``ObservableAppState._global_settings_path`` ignores the ``yaml_path``
    passed to the constructor on every niladic ``save_to_yaml()``/
    ``load_from_yaml()`` call — autosave included — and always resolves to
    the real global settings file. Any test that builds an ``ObservableAppState``
    (directly, or via ``MetaWidget``/the ``gui`` fixture) and lets its
    10-second autosave timer fire, or calls one of those methods with no
    argument, would otherwise overwrite the user's real settings — e.g.
    ``project_path`` — with whatever this test happened to set.
    """
    if not GUI_AVAILABLE:
        return
    global_path = tmp_path / "gui_settings.yaml"
    monkeypatch.setattr(ObservableAppState, "_global_settings_path", lambda self: global_path)


# ---------------------------------------------------------------------------
# GUI fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gui(request, qtbot, tmp_path, monkeypatch):
    if not GUI_AVAILABLE:
        pytest.skip("Qt/pygfx not installed")
    show = request.config.getoption("--show")

    test_config_dir = tmp_path / ".ethograph"
    test_config_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(
        paths_module,
        "default_config_dir",
        lambda data_dir=None: test_config_dir,
    )

    from ethograph.gui.app_state import ObservableAppState

    shell = EthographMainWindow()
    qtbot.addWidget(shell)
    meta = MetaWidget(shell)
    shell.attach_meta_widget(meta)
    meta._check_unsaved_changes = lambda event: True
    # Hermetic per-dataset state: local_settings.yaml is read from and written
    # to this test's tmp dir, never the shared example dataset dirs — so a
    # flag left behind by a GUI session (or another test) cannot reach a test.
    local_dir = tmp_path / "local_settings"
    local_dir.mkdir(exist_ok=True)

    def _hermetic_local_settings_path(self):
        nc_file_path = getattr(self, "nc_file_path", None)
        if not nc_file_path:
            return None
        return local_dir / Path(str(nc_file_path)).stem / ObservableAppState.LOCAL_SETTINGS_FILENAME

    monkeypatch.setattr(ObservableAppState, "_local_settings_path", _hermetic_local_settings_path)
    # Hermetic layout state: never apply a panel layout a previous run left behind.
    meta.app_state._layout_snapshot_provider = None
    _real_apply = meta.apply_saved_panel_layout
    meta.apply_saved_panel_layout = lambda: setattr(meta.app_state, "panel_layout", None) or _real_apply()
    if show:
        shell.show()

    yield shell, meta
    if show:
        qtbot.wait(15_000)
    shell.close()


@pytest.fixture
def birdpark_audio_only_gui(gui, qtbot):
    """Load birdpark dataset with audio folder only (no video) — triggers 3-panel mode."""
    viewer, meta = gui
    nc_path = str(BIRDPARK_NC)

    meta.io_widget.nc_file_path_edit.setText(nc_path)
    meta.app_state.nc_file_path = nc_path
    meta.io_widget.audio_folder_edit.setText(str(BIRDPARK_DIR))
    meta.app_state.audio_folder = str(BIRDPARK_DIR)
    # Deliberately do NOT set video_folder
    meta.app_state.video_folder = None
    meta.io_widget.downsample_checkbox.setChecked(False)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()

    return viewer, meta


@pytest.fixture
def no_video_gui(birdpark_audio_only_gui):
    """Backward-compatible alias for tests that still request no_video_gui."""
    return birdpark_audio_only_gui


@pytest.fixture
def birdpark_gui(gui, qtbot):
    return _load_template_gui(gui, "birdpark")


@pytest.fixture
def moll2025_gui(gui, qtbot):
    return _load_template_gui(gui, "moll2025")


@pytest.fixture
def lockbox_gui(gui, qtbot):
    return _load_template_gui(gui, "lockbox")


@pytest.fixture
def philodoptera_gui(gui, qtbot):
    return _load_template_gui(gui, "philodoptera")


@pytest.fixture
def canary_gui(gui, qtbot):
    _skip_if_not_downloaded("canary")
    viewer, meta = gui
    ds_info = DATASETS["canary"]
    resolved = resolve_dataset_paths("canary")
    nc_path = resolved.get("nc_file_path")
    if not nc_path:
        audio_name = ds_info.get("audio_file")
        if audio_name:
            candidate = dataset_dir("canary") / (Path(audio_name).stem + ".nc")
            if candidate.exists():
                nc_path = str(candidate)
    if not nc_path or not Path(nc_path).exists():
        pytest.skip("Canary .nc is missing; generate it via template dialog first")

    io = meta.io_widget
    io._clear_all_line_edits()
    io.nc_file_path_edit.setText(nc_path)
    meta.app_state.nc_file_path = nc_path
    if resolved.get("audio_folder"):
        io.audio_folder_edit.setText(resolved["audio_folder"])
        meta.app_state.audio_folder = resolved["audio_folder"]

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    assert meta.app_state.ready, "Failed to load canary"
    return viewer, meta


@pytest.fixture
def birdpark_gui_downsampled(gui, qtbot):
    return _load_template_gui(gui, "birdpark", downsample=True)


@pytest.fixture
def moll2025_pynapple_gui(gui, qtbot):
    """Load Moll2025 pynapple .npz data with alignment NWB linking to original media."""
    npz_dir = MOLL_PYNAPPLE_DIR
    speed_npz = npz_dir / "beakTip_speed.npz"
    alignment = npz_dir / ".ethograph" / "alignment.nwb"
    if not speed_npz.exists():
        pytest.skip("Moll2025_pynapple not set up")
    if not alignment.exists():
        from ethograph.utils.download import setup_moll2025_pynapple

        setup_moll2025_pynapple(npz_dir)
    assert alignment.exists(), f"alignment.nwb not found at {alignment}"

    viewer, meta = gui
    io = meta.io_widget
    io._clear_all_line_edits()

    meta.app_state._suspend_local_autoload = True
    io.nc_file_path_edit.setText(str(speed_npz))
    meta.app_state.nc_file_path = str(speed_npz)
    meta.app_state._suspend_local_autoload = False

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()

    assert meta.app_state.ready, "Failed to load Moll2025 pynapple"
    return viewer, meta
