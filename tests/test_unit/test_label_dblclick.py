"""A label's closing click is never swallowed by double-click-to-autoscale.

pyqtgraph reports the second of two quick clicks in the same spot as a
double-click (``GraphicsScene.mouseDoubleClickEvent`` appends a
``MouseClickEvent(..., double=True)``). ``BasePlot._handle_click`` answers a
left double-click by autoscaling and returning, which is right for a plot being
read and wrong for a label being drawn: the offset click never reaches
``LabelsWidget._on_plot_clicked``, so nothing snaps, nothing is committed, and
the interval stays pending while the y-axis rescales instead.

Labelling fast, or labelling a short event while zoomed out, both put the two
boundary clicks inside the system's double-click time *and* distance.
"""

import pytest

pytest.importorskip("qtpy")

from qtpy.QtCore import Qt  # noqa: E402

from ethograph.gui.plots_base import BasePlot  # noqa: E402


class _AppState:
    def __init__(self, armed: bool):
        self.label_drawing_armed = armed


class _Event:
    """The parts of a pyqtgraph MouseClickEvent that _handle_click reads."""

    def __init__(self, double: bool, button=Qt.LeftButton):
        self._double = double
        self._button = button

    def double(self):
        return self._double

    def button(self):
        return self._button

    def modifiers(self):
        return Qt.NoModifier

    def scenePos(self):
        return object()


class _Plot:
    """Bare host for the handler, so no GL canvas or dataset is built."""

    _handle_click = BasePlot._handle_click

    def __init__(self, armed: bool):
        self.app_state = _AppState(armed)
        self._interaction_enabled = True
        self.autoscaled = False
        self.emitted: list = []
        self.plot_item = self
        self.vb = self
        self.plot_clicked = self

    # -- stand-ins for the pyqtgraph pieces --------------------------------
    def mapSceneToView(self, _pos):
        return _Point(4.2)

    def emit(self, click_info):
        self.emitted.append(click_info)

    def autoscale(self):
        self.autoscaled = True


class _Point:
    def __init__(self, x):
        self._x = x

    def x(self):
        return self._x

    def y(self):
        return 0.0


class TestDoubleClickWhileDrawing:
    def test_closing_click_reaches_the_label_handler(self):
        """Armed: the double-click is the offset click and must be delivered."""
        plot = _Plot(armed=True)
        plot._handle_click(_Event(double=True))
        assert plot.emitted, "the closing click was swallowed — the label can never be committed"
        assert plot.emitted[0]["x"] == 4.2
        assert not plot.autoscaled, "a label being drawn must not rescale the axis instead"

    def test_autoscale_still_works_when_not_drawing(self):
        """Not armed: double-click keeps its ordinary meaning."""
        plot = _Plot(armed=False)
        plot._handle_click(_Event(double=True))
        assert plot.autoscaled
        assert not plot.emitted

    def test_single_click_is_unaffected_either_way(self):
        for armed in (True, False):
            plot = _Plot(armed=armed)
            plot._handle_click(_Event(double=False))
            assert plot.emitted, f"a plain click must always be delivered (armed={armed})"
            assert not plot.autoscaled

    def test_right_double_click_never_autoscales(self):
        plot = _Plot(armed=False)
        plot._handle_click(_Event(double=True, button=Qt.RightButton))
        assert not plot.autoscaled
        assert plot.emitted


def test_arming_mirrors_onto_app_state():
    """The plots read the flag off app_state, so the setter must keep it true.

    Guards the mirror rather than the widget: a future edit that assigns
    ``_ready_for_label_click`` directly would silently re-break the fix above.
    """
    from ethograph.gui.widgets_labels import LabelsWidget

    class _Host:
        ready_for_label_click = LabelsWidget.ready_for_label_click

        def __init__(self):
            self._ready_for_label_click = False
            self.app_state = _AppState(False)

    host = _Host()
    host.ready_for_label_click = True
    assert host.app_state.label_drawing_armed is True
    host.ready_for_label_click = False
    assert host.app_state.label_drawing_armed is False
