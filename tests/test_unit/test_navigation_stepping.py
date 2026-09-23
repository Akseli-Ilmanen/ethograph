"""Window stepping (Shift+arrows) in the navigation widget.

Regression cover for a crash: ``_step_window`` still called a
``_step_time_no_video`` helper and a ``plot_container.time_slider`` that both
went away with the bottom-bar refactor, so every Shift+arrow raised
AttributeError from the global shortcut.

The widget is built headless with a bare ``QWidget`` for the shell — nothing in
this path touches the real main window.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication, QWidget

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.widgets_navigation import NavigationWidget

VIEW_SPAN = 4.0
FPS = 25.0


class _FakeMarker:
    def __init__(self, time_s: float):
        self._t = time_s

    def value(self) -> float:
        return self._t


class _FakePlot:
    def __init__(self, marker_time: float):
        self.time_marker = _FakeMarker(marker_time)


class _FakePlotContainer:
    """What window stepping needs off the plot container."""

    def __init__(self, xlim=(0.0, 10.0), marker_time: float | None = None):
        self._xlim = xlim
        self._marker_time = marker_time
        self.marked: list[float] = []

    def _visible_plots(self):
        if self._marker_time is None:
            return []
        return [_FakePlot(self._marker_time)]

    def get_current_xlim(self):
        return self._xlim

    def update_time_marker_by_time(self, time_s: float):
        self.marked.append(time_s)


class _FakeVideo:
    """A playhead: frames at a fixed rate, recording every seek."""

    def __init__(self, frame: int = 0):
        self.frame = frame
        self.seeks: list[int] = []

    def frame_to_time(self, frame: int) -> float:
        return frame / FPS

    def time_to_frame(self, time_s: float, round_nearest: bool = False) -> int:
        return int(round(time_s * FPS))

    def seek_to_frame(self, frame: int) -> None:
        self.frame = frame
        self.seeks.append(frame)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def nav(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")  # never touch the real settings
    widget = NavigationWidget(QWidget(), state)
    widget.plot_container = _FakePlotContainer()
    state.ready = True
    state.xlim_mode = "fixed"
    state.fixed_window_s = VIEW_SPAN
    state.time_jump_s = VIEW_SPAN  # the "Jump step" spinbox drives stepping
    yield widget
    widget.close()


def test_stepping_forward_moves_one_view_span(nav):
    """Without a visible marker the playhead is the middle of the visible window."""
    nav.step_window_forward()

    assert nav.plot_container.marked == [5.0 + VIEW_SPAN]


def test_stepping_backward_moves_the_other_way(nav):
    nav.step_window_backward()

    assert nav.plot_container.marked == [5.0 - VIEW_SPAN]


def test_stepping_follows_the_time_marker_when_there_is_one(nav):
    """The marker is the playhead — it is the one thing always on the plot
    axis's clock (the video follows it during playback, not the reverse)."""
    nav.plot_container._marker_time = 2.0
    nav.app_state.video = _FakeVideo()

    nav.step_window_forward()

    assert nav.plot_container.marked == [2.0 + VIEW_SPAN]
    assert nav.app_state.video.seeks == [int(round((2.0 + VIEW_SPAN) * FPS))]


def test_stepping_does_nothing_before_a_dataset_is_ready(nav):
    nav.app_state.ready = False

    nav.step_window_forward()

    assert nav.plot_container.marked == []


def test_stepping_does_nothing_without_a_span(nav):
    """A zero jump step would step nowhere."""
    nav.app_state.time_jump_s = 0.0

    nav.step_window_forward()

    assert nav.plot_container.marked == []


class _FakeLoader:
    def __init__(self, backend):
        self.backend = backend


def test_session_scope_disabled_for_multitrial_xarray(nav):
    """Multi-trial .nc data is stored per trial — the session-scope combo
    entry is disabled and a persisted session scope is coerced to trial."""
    nav.app_state.data_loader = _FakeLoader("xarray")
    nav.app_state.trials = [1, 2, 3]
    nav.app_state.slider_scope = "session"

    nav.update_scope_availability()

    item = nav.scope_combo.model().item(2)  # "Session start → Session end"
    assert not item.isEnabled()
    assert nav.app_state.slider_scope == "trial"


def test_session_scope_stays_enabled_for_pynapple(nav):
    nav.app_state.data_loader = _FakeLoader("pynapple")
    nav.app_state.trials = [1, 2, 3]

    nav.update_scope_availability()

    assert nav.scope_combo.model().item(2).isEnabled()
