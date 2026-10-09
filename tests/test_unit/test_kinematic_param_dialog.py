"""The kinematic Configure dialog's copied snippet runs as pasted."""

from ethograph.gui.dialog_function_params import (
    _build_editor_text,
    _changepoint_ds_call,
    _get_param_infos,
    get_registry,
)

KINEMATIC_KEYS = ("find_troughs", "find_turning_points")


def test_copied_call_adds_the_mask_to_the_dataset():
    """The snippet is ``eto.add_changepoints_to_ds`` under the method's own name, not a bare call."""
    for key in KINEMATIC_KEYS:
        spec = get_registry()[key]
        call = _changepoint_ds_call(spec, {"prominence": 0.5})
        assert call.startswith("ds = eto.add_changepoints_to_ds(")
        assert f"changepoint_name={spec.changepoint_name!r}" in call
        assert f"changepoint_func={spec.func.__name__}" in call
        assert "    prominence=0.5," in call
        assert "sr" not in spec.auto_params


def test_troughs_exposes_find_peaks_knobs():
    """``prominence`` and ``distance`` are editable, not hidden behind ``**kwargs``."""
    spec = get_registry()["find_troughs"]
    names = {pi.name for pi in _get_param_infos(spec)}
    assert {"prominence", "distance"} <= names
    text = _build_editor_text(spec, _get_param_infos(spec), {})
    assert "prominence=" in text
