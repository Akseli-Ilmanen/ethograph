"""Two camera views of one clip: pairing, the 2D + 3D datasets, and the dialog's second view."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import xarray as xr
from aniposelib.cameras import Camera, CameraGroup

pytest.importorskip("pygfx")

import pygfx as gfx  # noqa: E402
from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

import ethograph as eto  # noqa: E402
from ethograph.gui.app_state import ObservableAppState  # noqa: E402
from ethograph.gui.dialog_pose_labelling import PoseLabellingDialog  # noqa: E402
from ethograph.gui.pose_annotate import KeypointStore, sidecar_path  # noqa: E402
from ethograph.gui.pose_edit_mixin import LOOP_MODE, SEQUENTIAL_MODE  # noqa: E402
from ethograph.gui.pose_two_view import ViewSource, pair_datasets, paired_frames  # noqa: E402
from ethograph.gui.triangulation import triangulate_loaded_session  # noqa: E402
from ethograph.io.overlay_source import overlay_candidates  # noqa: E402
from ethograph.triangulate.session import POSITION_3D  # noqa: E402

NAMES = ["beak", "tail"]
_MATRIX = np.array([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
_BEAK = np.array([0.5, -0.4, 20.0])
_TAIL = np.array([-0.8, 0.3, 21.0])


def _rig() -> CameraGroup:
    turn = cv2.Rodrigues(np.array([0.05, -0.4, 0.02]))[0]
    return CameraGroup(
        [
            Camera(name="left", size=[640, 480], matrix=_MATRIX),
            Camera(name="right", size=[640, 480], matrix=_MATRIX, rvec=cv2.Rodrigues(turn)[0], tvec=[-6.0, 0.3, 1.2]),
        ]
    )


def _pixels(point: np.ndarray) -> np.ndarray:
    """*point* in the left and the right camera, ``(2, 2)``."""
    return _rig().project(point[None])[:, 0]


def _store(n_frames: int) -> KeypointStore:
    return KeypointStore(keypoint_names=list(NAMES), n_frames=n_frames, individual_names=["bird"])


# -- pairing and datasets (no Qt) ---------------------------------------------------


def test_views_are_paired_by_moment_not_by_frame_index():
    left = ViewSource("left", _store(10), fps=10.0, time_offset=0.0)
    right = ViewSource("right", _store(12), fps=20.0, time_offset=-0.2)
    # Left frame 3 is at 0.3 s; the right camera started 0.2 s earlier at twice the rate.
    assert paired_frames(np.array([0, 3, 9]), left, right).tolist() == [4, 10, -1]


def test_pair_exports_2d_with_a_camera_dim_and_3d_on_the_first_views_clock():
    left = ViewSource("left", _store(10), fps=10.0, time_offset=0.0)
    right = ViewSource("right", _store(12), fps=20.0, time_offset=-0.2)
    left.store.set_point(3, "beak", tuple(_pixels(_BEAK)[0]), "bird")
    right.store.set_point(10, "beak", tuple(_pixels(_BEAK)[1]), "bird")
    left.store.set_point(3, "tail", tuple(_pixels(_TAIL)[0]), "bird")  # one view only: no 3D point

    ds_2d, ds_3d, _ = pair_datasets([left, right], _rig())

    assert [c.name for c in overlay_candidates(ds_2d, None, "right", (640, 480))] == ["position"]
    beak = ds_3d["position"].sel(keypoint="beak", individual="bird")
    np.testing.assert_allclose(beak.isel(time=3).values, _BEAK, atol=1e-3)
    assert beak.isel(time=2).isnull().all()
    assert ds_3d["position"].sel(keypoint="tail").isnull().all()
    assert float(ds_3d["confidence"].sel(keypoint="beak", individual="bird").isel(time=3)) == 1.0

    ds_2d_only, none_3d, _ = pair_datasets([left, right])
    assert none_3d is None and ds_2d_only["position"].sizes["camera"] == 2


# -- the dialog's second view -------------------------------------------------------


class _View:
    """The slice of CameraView the dialog, the label mode and the second view touch."""

    def __init__(self, camera: str, video: str):
        self._scene = gfx.Scene()
        self._canvas = QWidget()
        self.camera_name = camera
        self.source_video_path = video
        self.fps = 25.0
        self.n_frames = 10
        self.start_frame = 0
        self.time_offset = 0.0

    def scene(self):
        return self._scene

    def key_target(self):
        return self._canvas

    def image_height(self):
        return 480.0

    def overlay_scale(self):
        return 1.0

    def image_units_per_pixel(self):
        return 1.0

    def set_label_mode(self, mode):
        pass

    def set_label_locked(self, locked):
        pass

    def request_draw(self):
        pass


class _SpacePlot:
    def __init__(self):
        self.points = None

    def set_live_points(self, points, colors=None):
        self.points = points


class _Shell(QWidget):
    def __init__(self, primary: _View, extra: _View):
        super().__init__()
        self.video_area = type("_Area", (), {"primary": primary, "extras": {extra.camera_name: extra}})()


class _DataWidget:
    def __init__(self, app_state, shell):
        self.app_state = app_state
        self.shell = shell
        self.pose_mgr = None
        self.space_plots = [_SpacePlot()]

    def update_pose(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def two_views(qapp, tmp_path):
    """The dialog armed in Sequential on ``left``, with ``right`` picked as its second view."""
    project = tmp_path / "project"
    (project / "calibration").mkdir(parents=True)
    _rig().dump(str(project / "calibration" / "rig.toml"))

    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state.project_path = str(project)
    state.keypoints = list(NAMES)
    state.labelling_individuals = ["bird"]
    state.video_path = str(tmp_path / "left.mp4")
    primary, extra = _View("left", state.video_path), _View("right", str(tmp_path / "right.mp4"))
    dialog = PoseLabellingDialog(_DataWidget(state, _Shell(primary, extra)))
    dialog.set_interaction_mode(SEQUENTIAL_MODE)
    dialog._second.activate(extra)
    yield dialog
    dialog.close()


def test_sequential_moves_on_only_once_both_views_have_the_point(two_views):
    dialog, second = two_views, two_views._second
    left, right = _pixels(_BEAK)

    dialog._mode.handle_click(*left)
    assert dialog._mode.active_keypoint == second.mode.active_keypoint == "beak"

    second.mode.handle_click(*right)
    assert dialog._mode.active_keypoint == second.mode.active_keypoint == "tail"
    assert dialog.store.is_anchor(0, "beak", "bird") and second.store.is_anchor(0, "beak", "bird")


def test_a_point_in_both_views_is_a_3d_point_in_the_space_plot(two_views):
    dialog, second = two_views, two_views._second
    space_plot = dialog._data_widget.space_plots[0]
    left, right = _pixels(_BEAK)

    dialog._mode.handle_click(*left)
    dialog._mode.handle_release(*left)
    assert space_plot.points is None or len(space_plot.points) == 0

    second.mode.handle_click(*right)
    second.mode.handle_release(*right)
    np.testing.assert_allclose(space_plot.points, [_BEAK], atol=1e-3)


def test_each_camera_keeps_its_own_sidecar_and_backspace_follows_the_pointer(two_views, tmp_path):
    dialog, second = two_views, two_views._second
    left, right = _pixels(_BEAK)
    dialog._mode.handle_click(*left)
    second.mode.handle_click(*right)
    second.mode.set_active("beak", "bird")
    second._on_changed()

    assert dialog._delete_selected_point()
    assert dialog.store.is_anchor(0, "beak", "bird") and not second.store.is_anchor(0, "beak", "bird")

    second.mode.handle_click(*right)
    dialog._save_store()
    assert KeypointStore.load(sidecar_path(tmp_path / "left.mp4")).is_anchor(0, "beak", "bird")
    assert KeypointStore.load(sidecar_path(tmp_path / "right.mp4")).is_anchor(0, "beak", "bird")


def test_loop_stays_on_the_frame_until_the_second_view_has_the_point(two_views, monkeypatch):
    dialog, second = two_views, two_views._second
    advanced = []
    monkeypatch.setattr(dialog, "_advance_frame", lambda: advanced.append(True))
    dialog.set_interaction_mode(LOOP_MODE)
    left, right = _pixels(_BEAK)

    dialog._mode.handle_click(*left)
    assert not advanced
    second.mode.handle_click(*right)
    assert advanced == [True]


# -- Tools ▸ 3D: Triangulate poses… -------------------------------------------------


class _LoadedSession:
    def __init__(self, app_state):
        self.app_state = app_state
        self.served = 0

    def serve_current_tree(self):
        self.served += 1


def test_tools_entry_triangulates_the_shown_trials_and_refreshes_the_data_layer(qapp, tmp_path):
    project = tmp_path / "project"
    (project / "calibration").mkdir(parents=True)
    _rig().dump(str(project / "calibration" / "rig.toml"))
    pixels = np.stack([_pixels(_BEAK)] * 5, axis=1)  # (camera, time, 2)
    trials = [
        xr.Dataset(
            {"position": (("camera", "time", "space", "individual"), pixels[..., None])},
            coords={
                "camera": ["left", "right"],
                "time": np.arange(5) / 10.0,
                "space": ["x", "y"],
                "individual": ["bird"],
            },
            attrs={"trial": trial},
        )
        for trial in (1, 2)
    ]
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state.project_path = str(project)
    state.dt = eto.from_datasets(trials)
    state.trials = [2]
    session = _LoadedSession(state)

    assert triangulate_loaded_session(session)

    assert session.served == 1
    assert POSITION_3D not in state.dt.trial(1)  # filtered out of the trials table: left alone
    beak = state.dt.trial(2)[POSITION_3D].isel(time=0).values.ravel()
    np.testing.assert_allclose(beak, _BEAK, atol=1e-3)
