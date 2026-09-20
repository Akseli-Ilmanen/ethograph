"""The export registry: a named exporter adds a lab's columns, and cannot
take away the ones a labels file is made of."""

from __future__ import annotations

import pandas as pd
import pytest

from ethograph.labels import exporters
from ethograph.labels.export import enrich_labels_df
from ethograph.labels.exporters import STANDARD, ExportContext, run


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": [1, 1, 2],
            "individual": ["c1", "c1", "c1"],
            "labels": [1, 2, 1],
            "onset_s": [0.5, 1.5, 0.2],
            "offset_s": [1.0, 2.0, 0.9],
        }
    )


class _Tree:
    """The smallest thing the crow exporter reads."""

    attrs = {"session": "ses-01"}
    trials = [1, 2]

    class _Trial(dict):
        def __init__(self):
            super().__init__(pulse_onsets=1)
            self.pulse_onsets = type("V", (), {"values": [3, 4]})()

    def itrial(self, _index):
        return self._Trial()

    def trial(self, _trial):
        return self._Trial()


class TestRegistry:
    def test_standard_is_registered_and_first(self):
        names = [name for name, _ in exporters.choices()]
        assert names[0] == STANDARD

    def test_every_choice_resolves(self):
        for name, _ in exporters.choices():
            assert callable(exporters.get(name))

    def test_crowlab_is_registered(self):
        assert "crowlab" in [name for name, _ in exporters.choices()]

    def test_a_labs_label_says_it_adds_to_the_standard(self):
        """Picking a lab is never a choice to go without the standard columns."""
        labels = dict(exporters.choices())
        assert labels["crowlab"].startswith("Standard + ")
        assert not labels[STANDARD].startswith("Standard + ")

    def test_unknown_name_falls_back_to_standard(self, caplog):
        """A stale setting must not fail the save."""
        assert exporters.get("a-lab-that-left") is exporters.get(STANDARD)
        assert "Unknown label exporter" in caplog.text


class TestContract:
    def test_dropping_a_required_column_is_refused(self):
        def bad(df, ctx):
            return df.drop(columns=["onset_s"])

        with pytest.raises(ValueError, match="onset_s"):
            run(bad, _labels(), ExportContext())

    def test_adding_a_column_is_allowed(self):
        def good(df, ctx):
            df["mine"] = 1
            return df

        assert "mine" in run(good, _labels(), ExportContext()).columns


class TestThroughEnrich:
    def test_standard_adds_no_lab_columns(self):
        out = enrich_labels_df(_labels(), dt=_Tree(), exporter=exporters.get(STANDARD))
        assert "session" not in out.columns
        assert "pulse_onsets" not in out.columns

    def test_crowlab_adds_its_own(self):
        out = enrich_labels_df(_labels(), dt=_Tree(), exporter=exporters.get("crowlab"))
        assert out["session"].unique().tolist() == ["ses-01"]
        assert out["session_trial"].tolist() == ["ses-01_1", "ses-01_1", "ses-01_2"]

    def test_pulse_onsets_only_on_a_trials_first_row(self):
        out = enrich_labels_df(_labels(), dt=_Tree(), exporter=exporters.get("crowlab"))
        assert out["pulse_onsets"].tolist() == ["3–4", "", "3–4"]

    def test_crowlab_on_a_session_without_a_tree(self):
        out = enrich_labels_df(_labels(), dt=None, exporter=exporters.get("crowlab"))
        assert "session" not in out.columns
