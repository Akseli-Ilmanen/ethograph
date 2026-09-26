"""Tools ▸ Label bulk editing… — a form in front of CurationPanel's bulk methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_bulk_labels import LabelBulkEditDialog
from ethograph.gui.widgets_curation import CurationPanel
from ethograph.labels.intervals import LABELING_AUTOMATED, LABELING_MANUAL


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _LabelsStub(QWidget):
    def __init__(self, mappings, curation_panel):
        super().__init__()
        self._mappings = mappings
        self.curation_panel = curation_panel


class _TrialsStub:
    def __init__(self, trials):
        self._all_trials = list(trials)

    def all_trials(self):
        return list(self._all_trials)


class _Meta:
    def __init__(self, app_state, labels_widget, trials_widget):
        self.app_state = app_state
        self.labels_widget = labels_widget
        self.trials_widget = trials_widget
        self.data_widget = None
        self.io_widget = None


MAPPINGS = {
    0: {"name": "background"},
    4: {"name": "peck"},
    6: {"name": "hop"},
    8: {"name": "mount"},
}


def _labels_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset_s": [1.0, 2.0, 2.5, 0.5],
            "offset_s": [np.nan, 3.0, np.nan, np.nan],
            "labels": [4, 8, 6, 4],
            "individual": ["a", "a", "a", "a"],
            "individual_rec": ["", "", "", ""],
            "event_type": ["point", "state", "point", "point"],
            "confidence": [0.3, 1.0, 0.6, 0.9],
            "labeling_method": [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_AUTOMATED, LABELING_AUTOMATED],
            "trial": ["0", "0", "0", "1"],
        }
    )


@pytest.fixture()
def dialog(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state._all_labels_df = _labels_df()
    state.trials = ["0", "1"]
    state.trials_sel = "0"
    state.ready = True
    trials = _TrialsStub(["0", "1"])
    panel = CurationPanel(state, None)
    labels = _LabelsStub(MAPPINGS, panel)
    meta = _Meta(state, labels, trials)
    panel.set_meta(meta)
    dlg = LabelBulkEditDialog(meta)
    yield dlg
    dlg.close()
    panel.close()


def _label_item(dlg, label_id):
    for i in range(dlg.label_list.count()):
        item = dlg.label_list.item(i)
        if int(item.data(Qt.UserRole)) == label_id:
            return item
    raise AssertionError(f"no item for label {label_id}")


class TestForm:
    def test_background_is_excluded_from_the_checklist(self, dialog):
        assert dialog.label_list.count() == 3
        ids = {int(dialog.label_list.item(i).data(Qt.UserRole)) for i in range(dialog.label_list.count())}
        assert ids == {4, 6, 8}

    def test_all_starts_unchecked_and_nothing_is_ticked(self, dialog):
        """A destructive tool must never default to "affects everything"."""
        assert not dialog.all_labels_cb.isChecked()
        assert dialog.label_list.isEnabled()
        assert dialog._label_ids() == set()

    def test_ticking_all_disables_the_list_and_means_every_class(self, dialog):
        dialog.all_labels_cb.setChecked(True)
        assert not dialog.label_list.isEnabled()
        assert dialog._label_ids() is None

    def test_ticking_individual_items_reads_the_checked_ids(self, dialog):
        _label_item(dialog, 4).setCheckState(Qt.Checked)
        _label_item(dialog, 8).setCheckState(Qt.Checked)
        assert dialog._label_ids() == {4, 8}

    def test_nothing_ticked_with_all_off_is_an_empty_set(self, dialog):
        assert not dialog.all_labels_cb.isChecked()
        assert dialog._label_ids() == set()

    def test_trial_scope_defaults_to_filtered(self, dialog):
        assert dialog._trial_scope() == "filtered"


class TestGuardedActions:
    def test_curate_reaches_the_panel_with_the_form_s_choices(self, dialog, monkeypatch):
        calls = []
        monkeypatch.setattr(dialog.panel, "curate_trial_labels", lambda *a, **kw: calls.append((a, kw)) or 3)
        dialog.trial_scope_combo.setCurrentIndex(dialog.trial_scope_combo.findData("all"))
        _label_item(dialog, 4).setCheckState(Qt.Checked)
        _label_item(dialog, 6).setCheckState(Qt.Checked)
        dialog._curate()
        assert calls == [(("all", {4, 6}), {"confirm": True})]

    def test_delete_reaches_the_panel(self, dialog, monkeypatch):
        calls = []
        monkeypatch.setattr(dialog.panel, "delete_trial_labels", lambda *a, **kw: calls.append((a, kw)) or 1)
        dialog.all_labels_cb.setChecked(True)
        dialog._delete()
        assert calls == [(("filtered", None), {"confirm": True})]

    def test_purge_reaches_the_panel_with_the_threshold(self, dialog, monkeypatch):
        calls = []
        monkeypatch.setattr(dialog.panel, "purge_trial_labels", lambda *a, **kw: calls.append((a, kw)) or 0)
        dialog.purge_spin.setValue(0.25)
        dialog.all_labels_cb.setChecked(True)
        dialog._purge()
        assert calls == [(("filtered", 0.25, None), {"confirm": True})]

    def test_an_empty_checklist_refuses_rather_than_meaning_every_class(self, dialog, monkeypatch):
        """scope_mask reads an empty set as "every class" — the dialog must
        never let an unticked checklist silently touch everything."""
        called = []
        monkeypatch.setattr(dialog.panel, "delete_trial_labels", lambda *a, **kw: called.append(True))
        dialog._delete()  # default state: All off, nothing ticked
        assert called == []
