"""The measure tool's report: duration is order-free, y-range only within one panel."""

from types import SimpleNamespace

from ethograph.gui.dialog_measure import MeasuredPoint, measure_report


def _plot(label: str, units: str = ""):
    axis = SimpleNamespace(labelText=label, labelUnits=units)
    return SimpleNamespace(plot_item=SimpleNamespace(getAxis=lambda _side: axis))


class _TrialBasis:
    @staticmethod
    def from_display(t):
        return ("trial_1", t)


def test_report_same_panel_sorts_ranges_and_carries_units():
    spec = _plot("Frequency", "Hz")
    text = measure_report(MeasuredPoint(2.5, 4000.0, spec), MeasuredPoint(1.0, 1000.0, spec), _TrialBasis())
    assert "Duration: 1.500000 s" in text
    assert "t-range: (1.000000, 2.500000)" in text
    assert "y-range: (1000, 4000)" in text
    assert "Frequency [Hz]" in text


def test_report_across_panels_has_no_y_range():
    a, b = _plot("Amplitude"), _plot("Value")
    text = measure_report(MeasuredPoint(0.0, 1.0, a), MeasuredPoint(3.0, 2.0, b), _TrialBasis())
    assert "Duration: 3.000000 s" in text
    assert "y-range: n/a" in text
