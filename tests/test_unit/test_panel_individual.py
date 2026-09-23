"""Whose data a panel shows, and whose label a click places.

Every panel resolves its individual through one rule: its pin, else the
sidebar's individual — and its title says which of the two it is.
The labelling subject is the last clicked panel's individual. Covered here
Qt-free (the resolution, a feature panel's pin state, the camera title, the
pose filter); the loaded-GUI case is
``tests/test_integration/test_panel_pin.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ethograph.gui.pose_render import PoseRenderData, apply_individual_filter
from ethograph.gui.video_manager import camera_dock_title


class _Panel(SimpleNamespace):
    """A panel is anything with a ``pinned_individual``."""


def _state(app_state, sidebar: str | None):
    app_state.set_key_sel("individual", sidebar)
    return app_state


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_an_unpinned_panel_follows_the_sidebar_and_a_pinned_one_does_not(app_state):
    state = _state(app_state, "bird_1")
    assert state.panel_individual(_Panel(pinned_individual=None)) == "bird_1"
    assert state.panel_individual(_Panel(pinned_individual="bird_2")) == "bird_2"
    assert state.panel_individual(None) == "bird_1"
    state.set_key_sel("individual", "bird_3")
    assert state.panel_individual(_Panel(pinned_individual=None)) == "bird_3"
    assert state.panel_individual(_Panel(pinned_individual="bird_2")) == "bird_2"


def test_a_panel_title_says_which_mode_it_is_in(app_state):
    state = _state(app_state, "bird_1")
    state._all_labels_df = pd.DataFrame({"individual": ["bird_1", "bird_2"]})
    assert state.panel_mode_suffix(_Panel(pinned_individual=None)) == " \u2014 bird_1 (sidebar)"
    assert state.panel_mode_suffix(_Panel(pinned_individual="bird_2")) == " \u2014 bird_2 (pinned)"
    state._all_labels_df = pd.DataFrame({"individual": ["bird_1"]})
    assert state.panel_mode_suffix(_Panel(pinned_individual="bird_2")) == "", "one animal: nothing to say"


def test_the_labelling_subject_is_the_clicked_panels_individual(app_state):
    state = _state(app_state, "bird_1")
    announced: list = []
    state.labelling_subject_changed.connect(lambda *_: announced.append(state.labelling_subject))

    assert state.selected_individual() == "bird_1"
    state.set_subject_panel(_Panel(pinned_individual="bird_2"))
    assert state.selected_individual() == "bird_2"
    assert announced[-1] == "bird_2"
    state.set_subject_panel(_Panel(pinned_individual=None))
    assert state.selected_individual() == "bird_1", "an unpinned panel labels the sidebar's individual"
    state.set_key_sel("individual", "bird_3")
    state.refresh_labelling_subject()
    assert state.selected_individual() == "bird_3" and announced[-1] == "bird_3"


# ---------------------------------------------------------------------------
# A feature panel's pin
# ---------------------------------------------------------------------------


def _feature_panel(app_state, feature: str, dims: dict[str, list[str]]):
    from ethograph.gui.plots_base import PanelStateMixin

    class Panel(PanelStateMixin):
        def __init__(self):
            self.app_state = app_state
            self.panel_state["feature"] = feature

    app_state.data_loader = SimpleNamespace(feature_dims=lambda f: dims, dims=list(dims), catalog=None)
    return Panel()


def test_a_pin_is_injected_into_the_panels_selections_and_survives_a_layout_round_trip(app_state):
    state = _state(app_state, "bird_1")
    panel = _feature_panel(state, "speed", {"individual": ["bird_1", "bird_2"], "keypoint": ["beak"]})
    panel.panel_state["selections"] = {"keypoint": "beak"}

    assert panel._effective_selections()["individual"] == "bird_1"
    panel.set_pinned_individual("bird_2")
    assert panel._effective_selections()["individual"] == "bird_2"
    assert panel.panel_settings()["individual"] == "bird_2"

    other = _feature_panel(state, "speed", {"individual": ["bird_1", "bird_2"], "keypoint": ["beak"]})
    other.apply_panel_settings(panel.panel_settings())
    assert other.pinned_individual == "bird_2"
    other.apply_panel_settings({"feature": "speed"})
    assert other.pinned_individual is None, "a layout without the key unpins"


def test_a_feature_without_an_individual_dim_is_left_alone(app_state):
    state = _state(app_state, "bird_1")
    panel = _feature_panel(state, "audio_envelope", {"channel": ["left", "right"]})
    panel.panel_state["selections"] = {"channel": "left"}
    panel.set_pinned_individual("bird_2")
    assert panel._effective_selections() == {"channel": "left"}


# ---------------------------------------------------------------------------
# Cameras and poses
# ---------------------------------------------------------------------------


def test_camera_title_names_the_pin():
    assert camera_dock_title("cam-1", "/v/front.mp4") == "cam-1 (front.mp4)"
    suffix = " \u2014 bird_2 (pinned)"
    assert camera_dock_title("cam-1", "/v/front.mp4", suffix) == "cam-1 (front.mp4)" + suffix
    assert camera_dock_title("cam-1", None) == "cam-1"


def test_pose_filter_keeps_one_individuals_points():
    props = pd.DataFrame({"keypoint": ["beak", "beak", "tail"], "individual": ["a", "b", "a"]})
    pr = PoseRenderData(np.zeros((3, 3)), props, np.array([True, True, False]), "f")
    kept = apply_individual_filter(pr, "a")
    assert kept.data_not_nan.tolist() == [True, False, False]
    assert pr.data_not_nan.tolist() == [True, True, False], "masks are copies"


# ---------------------------------------------------------------------------
# Space and radial plots: the individual combo is the pin
# ---------------------------------------------------------------------------


@pytest.fixture
def two_bird_state(app_state, monkeypatch):
    """The real app_state over a real loader of a two-bird pose dataset."""
    import xarray as xr

    import ethograph as eto
    from ethograph.io.catalog import XarrayLoader, catalog_from_xarray
    from ethograph.io.derived import DerivedLoader
    from ethograph.io.time_model import TimeRange

    time = np.linspace(0.0, 10.0, 1001)
    sweep = np.linspace(0.0, 359.0, 1001)
    position = np.stack([np.cos(np.radians(sweep)), np.sin(np.radians(sweep))], axis=1)  # (T, space)
    ds = xr.Dataset(
        {
            "position": (
                ("time", "space", "keypoints", "individuals"),
                np.broadcast_to(position[:, :, None, None], (1001, 2, 2, 2)).copy(),
            ),
            "heading": (("time", "individuals"), np.column_stack([sweep, sweep])),
        },
        coords={
            "time": time,
            "space": ["x", "y"],
            "keypoints": ["beak", "tail"],
            "individuals": ["bird_1", "bird_2"],
        },
        attrs={"trial": 1},
    )
    catalog = catalog_from_xarray(ds, eto.from_datasets([ds]))
    app_state.ds = ds
    app_state.data_loader = DerivedLoader(XarrayLoader(ds, catalog))
    monkeypatch.setattr(type(app_state), "window_bounds", property(lambda _self: TimeRange(0.0, 10.0)))
    app_state.set_key_sel("individuals", "bird_1")
    return app_state


def test_a_space_plot_pins_through_its_own_individual_combo(two_bird_state):
    from ethograph.gui.plots_space import SpacePlot

    state = two_bird_state
    sp = SpacePlot(shell=None, app_state=state)
    sp.set_store(state.data_loader)
    sp.feature_combo.setCurrentIndex(sp.feature_combo.findText("position"))
    combo = sp._individual_combos()["individuals"]
    assert combo.currentText() == "bird_1", "a new panel shows the sidebar's individual"
    assert state.panel_mode_suffix(sp) == " \u2014 bird_1 (sidebar)"

    pinned: list = []
    sp.pin_changed.connect(pinned.append)
    sp.set_pinned_individual("bird_2")
    assert combo.currentText() == "bird_2"
    assert sp.space_settings()["individual"] == "bird_2"
    assert pinned == [], "a pin set from outside is not announced back"

    # The sidebar moves: the pinned panel stays, the follower moves.
    state.set_key_sel("individuals", "bird_2")
    state.set_key_sel("individuals", "bird_1")
    sp.sync_individual()
    assert combo.currentText() == "bird_2"

    # Picking a name in the panel's own combo is a pin, announced to the owner.
    combo.setCurrentIndex(combo.findText("bird_1"))
    assert sp.pinned_individual == "bird_1" and pinned == [sp]
    assert state.panel_mode_suffix(sp) == " \u2014 bird_1 (pinned)"

    # Follow the sidebar again: the combo takes the sidebar's individual.
    state.set_key_sel("individuals", "bird_2")
    sp.set_pinned_individual(None)
    assert sp.pinned_individual is None and combo.currentText() == "bird_2"
    assert "individual" not in sp.space_settings()

    other = SpacePlot(shell=None, app_state=state)
    other.set_store(state.data_loader)
    other.apply_space_settings({**sp.space_settings(), "individual": "bird_1"})
    assert other.pinned_individual == "bird_1"
    assert other._individual_combos()["individuals"].currentText() == "bird_1"


def test_a_radial_plot_pins_through_its_own_individual_combo(two_bird_state):
    from ethograph.gui.plots_radial import RadialPlot

    state = two_bird_state
    rp = RadialPlot(shell=None, app_state=state)
    rp.set_store(state.data_loader)
    rp.configure(feature="heading")
    combo = rp._dim_combos["individuals"]
    all_check = rp._dim_all_checks["individuals"]
    assert combo.currentText() == "bird_1"

    pinned: list = []
    rp.pin_changed.connect(pinned.append)
    combo.setCurrentIndex(combo.findText("bird_2"))
    assert rp.pinned_individual == "bird_2" and pinned == [rp]
    assert rp.radial_settings()["individual"] == "bird_2"

    # Freeing the dim draws every bird and leaves the pin alone; pinning again unfrees it.
    all_check.setChecked(True)
    assert rp.pinned_individual == "bird_2" and "individuals" not in rp.selections()
    rp.set_pinned_individual("bird_1")
    assert not all_check.isChecked() and combo.currentText() == "bird_1"

    # A follower with a freed dim stays free when the sidebar moves.
    rp.set_pinned_individual(None)
    all_check.setChecked(True)
    state.set_key_sel("individuals", "bird_2")
    rp.sync_individual()
    assert all_check.isChecked()

    other = RadialPlot(shell=None, app_state=state)
    other.set_store(state.data_loader)
    other.apply_radial_settings({"feature": "heading", "selections": {"individuals": "bird_1"}, "individual": "bird_1"})
    assert other.pinned_individual == "bird_1"
    assert other._dim_combos["individuals"].currentText() == "bird_1"
