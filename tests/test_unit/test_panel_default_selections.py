"""A fresh panel plots one trace: its dims start pinned, never "All".

"All" is something the user ticks. A line plot opens with every multi-value dim
pinned to its first value; the heatmap, whose rows *are* a dim, keeps one free.
Switching feature pins only the dims the new feature brings, so an "All" the
user ticked on a dim both features share survives.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("qtpy")

from ethograph.gui.plots_base import PanelStateMixin  # noqa: E402

DIMS = {
    "speed": {"keypoints": ["nose", "tail"], "space": ["x", "y"]},
    "position": {"keypoints": ["nose", "tail"], "space": ["x", "y"], "camera": ["top", "side"]},
    # A console `stack`: its columns have no sidebar combo.
    "rings": {"column": ["sin", "cos"]},
}


class _Loader:
    catalog = SimpleNamespace(combos={"keypoints": None, "space": None, "camera": None})

    def feature_dims(self, feature):
        return DIMS[feature]


class _State:
    data_loader = _Loader()
    features_sel = "speed"

    def get_selections(self):
        return {}

    def panel_individual(self, _panel):
        return None


class _Panel(PanelStateMixin):
    def __init__(self, free: int = 0):
        self.app_state = _State()
        self.default_free_dims = free


def test_a_new_line_plot_pins_every_dim():
    panel = _Panel()
    panel._ensure_panel_state()
    assert panel.panel_state["selections"] == {"keypoints": "nose", "space": "x"}


def test_a_new_heatmap_keeps_one_dim_free():
    panel = _Panel(free=1)
    panel._ensure_panel_state()
    assert panel.panel_state["selections"] == {"space": "x"}


def test_a_dim_without_a_sidebar_combo_stays_free():
    panel = _Panel()
    panel.app_state.features_sel = "rings"
    panel._ensure_panel_state()
    assert panel.panel_state["selections"] == {}


def test_switching_feature_keeps_a_ticked_all_and_pins_the_new_dim():
    panel = _Panel()
    panel._ensure_panel_state()
    panel.set_panel_control("keypoints", None)  # the user ticks All

    panel.set_panel_control("features", "position")

    assert panel.panel_state["selections"] == {"space": "x", "camera": "top"}
