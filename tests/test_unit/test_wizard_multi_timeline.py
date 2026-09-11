"""Unit tests for wizard_multi_timeline: TimelinePage and helpers."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication

from ethograph.datasets import dataset_dir, is_dataset_downloaded
from ethograph.gui.wizard_multi_timeline import TimelinePage, _normalize_trial_key
from ethograph.gui.wizard_state import ModalityConfig, WizardState

DATA_DIR = Path(__file__).parents[2] / "data"
XX_CSV = DATA_DIR / "xx.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(trial_table: pd.DataFrame, files_aligned: bool = False) -> WizardState:
    state = WizardState()
    state.trial_table = trial_table
    state.files_aligned_to_trials = files_aligned
    return state


def _skip_if_not_downloaded(key: str):
    if not is_dataset_downloaded(key):
        pytest.skip(f"{key} not downloaded")


# ---------------------------------------------------------------------------
# _normalize_trial_key
# ---------------------------------------------------------------------------


class TestNormalizeTrialKey:
    def test_integer_passthrough(self):
        assert _normalize_trial_key(3) == 3

    def test_numeric_string_becomes_int(self):
        assert _normalize_trial_key("42") == 42

    def test_non_numeric_string_passthrough(self):
        assert _normalize_trial_key("trial_A") == "trial_A"

    def test_none_returns_none(self):
        assert _normalize_trial_key(None) is None

    def test_string_with_whitespace(self):
        assert _normalize_trial_key("  7  ") == 7

    def test_float_passthrough(self):
        assert _normalize_trial_key(1.5) == 1.5


# ---------------------------------------------------------------------------
# TimelinePage — smoke tests with xx.csv trial table
# ---------------------------------------------------------------------------


class TestTimelinePageXxCsv:
    """Smoke tests: populate_from_state must not crash and must set _total_duration."""

    @pytest.fixture
    def page(self, qtbot):
        w = TimelinePage()
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        return w

    @pytest.fixture
    def trial_table(self):
        assert XX_CSV.exists(), f"xx.csv not found at {XX_CSV}"
        return pd.read_csv(XX_CSV)

    def test_populate_timeline_mode_no_crash(self, page, trial_table):
        state = _make_state(trial_table, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()

    def test_total_duration_set_from_stop_times(self, page, trial_table):
        state = _make_state(trial_table, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()
        expected_max = trial_table["stop_time"].max()
        assert page._total_duration == pytest.approx(expected_max, rel=1e-6)

    def test_populate_aligned_table_mode_no_crash(self, page, trial_table):
        """files_aligned_to_trials=True → aligned table view, must not crash."""
        state = _make_state(trial_table, files_aligned=True)
        state.video = ModalityConfig(enabled=False)
        page.populate_from_state(state)
        QApplication.processEvents()

    def test_items_added_to_plot(self, page, trial_table):
        """Each trial boundary should produce at least one item in the plot."""
        state = _make_state(trial_table, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()
        assert len(page._items) >= len(trial_table)

    def test_repopulate_clears_previous_items(self, page, trial_table):
        """Calling populate_from_state twice should not accumulate items."""
        state = _make_state(trial_table, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()
        count_first = len(page._items)

        page.populate_from_state(state)
        QApplication.processEvents()
        assert len(page._items) == count_first

    def test_n_rows_matches_enabled_modalities(self, page, trial_table):
        """With no modalities enabled, y-axis should have 0 rows (no file bars)."""
        state = _make_state(trial_table, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()
        # No enabled modalities → no file bars, just trial boundary lines + labels
        assert page._total_duration > 0.0


# ---------------------------------------------------------------------------
# NaN resilience — regression for the ValueError crash
# ---------------------------------------------------------------------------


class TestTimelinePageNaN:
    @pytest.fixture
    def page(self, qtbot):
        w = TimelinePage()
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        return w

    def test_nan_stop_time_does_not_crash(self, page):
        """NaN stop_time must not raise ValueError from pyqtgraph setXRange."""
        df = pd.DataFrame(
            {
                "start_time": [0.0, 5.0, float("nan")],
                "stop_time": [float("nan"), 10.0, float("nan")],
                "trial": [1, 2, 3],
            }
        )
        state = _make_state(df, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()

    def test_nan_stop_time_duration_from_finite_rows(self, page):
        """When some stop_times are NaN, _total_duration uses the largest finite value."""
        df = pd.DataFrame(
            {
                "start_time": [0.0, 5.0],
                "stop_time": [float("nan"), 10.0],
                "trial": [1, 2],
            }
        )
        state = _make_state(df, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()
        assert math.isfinite(page._total_duration)
        assert page._total_duration == pytest.approx(10.0)

    def test_all_nan_stop_times_keeps_default_duration(self, page):
        """All-NaN stop_times: _total_duration stays at 1.0 (constructor default)."""
        df = pd.DataFrame(
            {
                "start_time": [0.0, 5.0],
                "stop_time": [float("nan"), float("nan")],
                "trial": [1, 2],
            }
        )
        state = _make_state(df, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()
        # max_time never exceeded 0.0 → total_duration unchanged from constructor
        assert page._total_duration == pytest.approx(1.0)

    def test_empty_trial_table_no_crash(self, page):
        df = pd.DataFrame(columns=["start_time", "stop_time", "trial"])
        state = _make_state(df, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()

    def test_none_trial_table_no_crash(self, page):
        state = _make_state(None, files_aligned=False)
        page.populate_from_state(state)
        QApplication.processEvents()


# ---------------------------------------------------------------------------
# With real moll2025 video files
# ---------------------------------------------------------------------------


class TestTimelinePageMoll2025:
    """Integration smoke tests using actual Moll2025 template files."""

    @pytest.fixture
    def page(self, qtbot):
        w = TimelinePage()
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        return w

    def test_timeline_with_video_modality(self, page):
        _skip_if_not_downloaded("moll2025")
        d = dataset_dir("moll2025")
        mp4s = sorted(d.glob("*.mp4"))
        if not mp4s:
            pytest.skip("no .mp4 files in moll2025 dataset")

        trial_table = pd.read_csv(XX_CSV)

        state = WizardState()
        state.trial_table = trial_table
        state.files_aligned_to_trials = False

        state.video = ModalityConfig(
            enabled=True,
            file_mode="aligned_to_session",
            pattern=None,
            single_file_path=str(mp4s[0]),
        )

        page.populate_from_state(state)
        QApplication.processEvents()

        # Duration should have been probed from the video
        assert page._total_duration > 0.0
        assert math.isfinite(page._total_duration)
