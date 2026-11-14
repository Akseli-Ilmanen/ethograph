"""The burst form remembers each detector's parameters and can put them back."""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication, QFormLayout, QGroupBox, QPushButton

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.burst_form import build_burst_inputs, read_burst_rule, saved_burst_parameters
from ethograph.labels.feature_events import LOG_ISI, MAX_INTERVAL, POISSON_SURPRISE, burst_parameters


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def state(tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    return state


def _form():
    box = QGroupBox()
    return box, QFormLayout(box)


def _rebuild(form, inputs, method, state):
    for stale in inputs.values():
        form.removeRow(stale)
    return build_burst_inputs(form, method, state)


def test_a_change_is_remembered_per_detector_and_a_new_form_opens_at_it(qapp, state):
    box, form = _form()
    inputs = build_burst_inputs(form, MAX_INTERVAL, state)
    inputs["min_spikes_in_burst"].setValue(7)
    assert state.burst_detector_params == {MAX_INTERVAL: {"min_spikes_in_burst": 7}}

    # Another detector is untouched by it, and coming back finds the choice again.
    inputs = _rebuild(form, inputs, LOG_ISI, state)
    assert read_burst_rule(LOG_ISI, inputs).params == burst_parameters(LOG_ISI)
    inputs = _rebuild(form, inputs, MAX_INTERVAL, state)
    assert inputs["min_spikes_in_burst"].value() == 7
    assert len(box.findChildren(QPushButton)) == 1  # one revert button, not one per rebuild


def test_revert_puts_one_detector_back_and_forgets_its_choices(qapp, state):
    state.burst_detector_params = {MAX_INTERVAL: {"min_spikes_in_burst": 7}, LOG_ISI: {"fallback": False}}
    box, form = _form()
    inputs = build_burst_inputs(form, MAX_INTERVAL, state)
    assert inputs["min_spikes_in_burst"].value() == 7
    box.findChild(QPushButton).click()
    assert read_burst_rule(MAX_INTERVAL, inputs).params == burst_parameters(MAX_INTERVAL)
    assert state.burst_detector_params == {LOG_ISI: {"fallback": False}}  # the other detector keeps its own


def test_a_saved_value_of_the_wrong_kind_is_ignored(qapp, state):
    state.burst_detector_params = {MAX_INTERVAL: {"min_spikes_in_burst": "many", "gone": 3, "max_isi_start": True}}
    assert saved_burst_parameters(state, MAX_INTERVAL) == {}


def test_rows_are_named_by_argument_and_explained_on_hover(qapp, state):
    box, form = _form()
    inputs = build_burst_inputs(form, MAX_INTERVAL, state)
    label = form.labelForField(inputs["max_isi_start"])
    assert label.text() == "max_isi_start:"
    assert "first of two spikes closer than this" in label.toolTip()
    assert "Default: 0.17 s" in inputs["max_isi_start"].toolTip()
    assert "Default: 3" in inputs["min_spikes_in_burst"].toolTip()


def test_poisson_surprise_offers_its_two_parameters_and_says_what_0_means(qapp, state):
    box, form = _form()
    inputs = build_burst_inputs(form, POISSON_SURPRISE, state)
    assert set(inputs) == {"min_surprise", "max_isi_burst"}
    assert inputs["max_isi_burst"].text() == "2 × mean ISI"  # its default, 0, is worked out per train
    assert "Default: 2 × mean ISI" in inputs["max_isi_burst"].toolTip()
    assert "Default: 5" in inputs["min_surprise"].toolTip()
    assert read_burst_rule(POISSON_SURPRISE, inputs).params == {"min_surprise": 5.0, "max_isi_burst": 0.0}
