"""A poses dataset has to be *plottable*, not merely loadable.

``position`` and its derivatives carry three non-time dims — ``space``,
``keypoint``, ``individual``. A feature plot renders what ``sel_valid`` returns,
and that has to come back ``(time,)`` or ``(time, dim)``: ``sel_valid`` asserts
it, so a panel whose selections pin nothing produces no data at all — silently,
from the user's side, as an empty panel.

The wiring that makes the dims *selectable* is an ordinary dataset load, covered
in ``tests/test_integration/test_keypoint_dataset_load.py``. What is unit-tested
here is the other half: a panel reducing its own selections to something a plot
can draw, whatever they were before the feature was picked.
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui.plots_base import PanelStateMixin
from ethograph.gui.pose_annotate import KINEMATICS, KeypointStore, store_to_dataset
from ethograph.io.catalog import XarrayLoader

FPS = 25.0
N_FRAMES = 12


@pytest.fixture
def keypoint_ds():
    """The dataset 'Load into the GUI' writes: poses plus every kinematic."""
    pytest.importorskip("movement")
    store = KeypointStore(keypoint_names=["beak", "tail"], n_frames=N_FRAMES, individual_names=["a", "b"])
    filled = np.zeros((N_FRAMES, 2, 2, 2))
    filled[:, :, :, 0] = np.arange(N_FRAMES)[:, None, None]
    store.set_fill(filled, np.ones((N_FRAMES, 2, 2)))
    return store_to_dataset(store, FPS, kinematics=KINEMATICS)


class _FakeAppState:
    """The slice of app_state a feature panel reads."""

    def __init__(self, loader, selections: dict):
        self.data_loader = loader
        self.features_sel = None
        self.colors_sel = None
        self._selections = selections

    def get_selections(self) -> dict:
        return dict(self._selections)

    def panel_individual(self, _panel):
        return None


class _Panel(PanelStateMixin):
    def __init__(self, app_state):
        self.app_state = app_state


def _rendered(loader, panel, feature: str):
    plot_data = loader.select(feature, panel._effective_selections())
    assert plot_data is not None
    return plot_data.data


# ----------------------------------------------------------------------


def test_a_keypoint_feature_has_three_dims_to_pin(keypoint_ds):
    loader = XarrayLoader(keypoint_ds)
    assert set(loader.feature_dims("position")) == {"space", "keypoint", "individual"}


def test_unpinned_dims_never_reach_a_plot(keypoint_ds):
    """The failure this guards. `sel_valid` refuses to return four axes, so a
    panel whose selections pin nothing produces no data at all — silently, from
    the user's side: the panel is simply empty."""
    loader = XarrayLoader(keypoint_ds)
    with pytest.raises(AssertionError):
        _rendered(loader, _Panel(_FakeAppState(loader, {})), "position")


@pytest.mark.parametrize(
    "feature",
    # `speed` is in the list because it is the odd one out: no `space` dim, so
    # it reduces from a different starting shape than the other three.
    ["position", "velocity", "acceleration", "speed"],
)
def test_picking_the_feature_reduces_it_to_something_plottable(keypoint_ds, feature):
    """Choosing the feature is what re-reduces the panel's stale selections."""
    loader = XarrayLoader(keypoint_ds)
    panel = _Panel(_FakeAppState(loader, {}))

    panel.set_panel_control("features", feature)

    assert _rendered(loader, panel, feature).ndim <= 2


def test_exactly_one_dim_is_left_free(keypoint_ds):
    """One free dim is the multi-trace case; two is the broken one."""
    loader = XarrayLoader(keypoint_ds)
    panel = _Panel(_FakeAppState(loader, {}))
    panel.set_panel_control("features", "position")

    dims = loader.feature_dims("position")
    free = [dim for dim, values in dims.items() if dim not in panel._effective_selections() and len(values) > 1]
    assert len(free) <= 1


def test_a_forked_panel_is_reduced_on_its_first_render(keypoint_ds):
    """`_ensure_panel_state` forks the globals, which mirror whichever panel was
    edited last — so they can leave this one's feature wide open."""
    loader = XarrayLoader(keypoint_ds)
    app_state = _FakeAppState(loader, {})
    app_state.features_sel = "position"
    panel = _Panel(app_state)

    panel._ensure_panel_state()

    assert _rendered(loader, panel, "position").ndim <= 2


def test_a_pinned_dim_is_never_overridden(keypoint_ds):
    """Reducing must not undo a choice the user actually made."""
    loader = XarrayLoader(keypoint_ds)
    panel = _Panel(_FakeAppState(loader, {"keypoint": "tail", "individual": "b"}))
    panel.set_panel_control("features", "position")

    selections = panel._effective_selections()
    assert selections["keypoint"] == "tail"
    assert selections["individual"] == "b"
