"""The skeleton plot as a panel over a real session: offered for ``position``
only, created from the popup, redrawn as the time marker moves."""

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from ethograph.gui.plots_skeleton import SKELETON_2D
from ethograph.gui.source_popup import allowed_plot_types


@pytest.fixture
def skeleton_panel(moll2025_gui):
    _, meta = moll2025_gui
    if "position" not in meta.app_state.data_loader.catalog.feature_choices():
        pytest.skip("this dataset has no position variable")
    meta._create_panel_for_source("feature", "position", SKELETON_2D)
    QApplication.processEvents()
    return meta, meta.data_widget.skeleton_plots[-1]


def test_skeleton_is_offered_for_position_only(moll2025_gui):
    _, meta = moll2025_gui
    features = meta.app_state.data_loader.catalog.feature_choices()
    assert SKELETON_2D in allowed_plot_types("feature", "position", meta.app_state)
    other = next(f for f in features if f != "position")
    assert SKELETON_2D not in allowed_plot_types("feature", other, meta.app_state)


def test_dropping_position_creates_a_skeleton_panel(skeleton_panel):
    meta, kp = skeleton_panel
    assert kp.dock_widget is not None
    assert kp.keypoint_list.count() >= 2, "one row per keypoint"
    assert meta.context_panel.current_context() == "skeleton"
    assert kp.controls_widget.isVisibleTo(meta.context_panel)


def test_the_pose_follows_the_time_marker(skeleton_panel):
    meta, kp = skeleton_panel
    bounds = meta.app_state.window_bounds
    frames = []
    for t in np.linspace(bounds.start_s, bounds.end_s, 20):
        kp.set_time(float(t))
        frame = kp.current_frame()
        if frame is not None and len(frame.points):
            frames.append(frame.points.copy())
    assert len(frames) >= 2, "no pose anywhere in the window"
    assert not np.allclose(frames[0], frames[-1]), "the pose never moved"


def test_hiding_a_keypoint_removes_its_point(skeleton_panel):
    meta, kp = skeleton_panel
    bounds = meta.app_state.window_bounds
    t = next(
        t
        for t in np.linspace(bounds.start_s, bounds.end_s, 20)
        if (kp.set_time(float(t)) or True) and kp.current_frame() is not None and len(kp.current_frame().points)
    )
    kp.set_time(float(t))
    before = len(kp.current_frame().points)
    kp.set_hidden_keypoints({kp.keypoint_list.item(0).text()})
    after = len(kp.current_frame().points)
    assert after < before


def test_settings_round_trip_through_the_saved_layout(skeleton_panel):
    meta, kp = skeleton_panel
    hidden = kp.keypoint_list.item(0).text()
    kp.set_hidden_keypoints({hidden})
    state = meta.data_widget.skeleton_layout_state()
    assert state[-1]["hidden_keypoints"] == [hidden]

    meta.data_widget.apply_skeleton_layout_state(state)
    QApplication.processEvents()
    restored = meta.data_widget.skeleton_plots[-1]
    assert restored.hidden_keypoints() == {hidden}


def test_closing_the_dock_drops_the_instance(skeleton_panel):
    meta, kp = skeleton_panel
    n_before = len(meta.data_widget.skeleton_plots)
    kp.closed.emit(kp)
    QApplication.processEvents()
    assert len(meta.data_widget.skeleton_plots) == n_before - 1
