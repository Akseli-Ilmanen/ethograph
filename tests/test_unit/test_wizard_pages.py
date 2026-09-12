"""The Data wizard's pages hand the state what each mode needs and nothing it forbids."""

from __future__ import annotations

from pathlib import Path

import pytest

from ethograph.gui.wizard_pages import FIGURES, ModePage, TimingPage, WritePage
from ethograph.gui.wizard_state import WizardState


class _AppState:
    project_path = None
    last_browse_dir = None


@pytest.fixture
def app_state():
    return _AppState()


def test_figures_ship_with_the_package():
    for path in FIGURES.values():
        assert path.is_file(), path


def test_mode_page_defaults_to_pairing(qapp):
    page = ModePage()
    state = WizardState()
    page.collect_state(state)
    assert state.mode == "pair"
    assert state.timing == "none"


class TestTimingPage:
    def test_free_running_offers_offset_and_split(self, qapp, app_state):
        page = TimingPage(app_state)
        page.set_mode("free_running")
        assert not page._rb_onsets.isVisibleTo(page)
        page._rb_offset.setChecked(True)
        page._offset.setValue(-0.4)
        page._rec_file.setText("ephys/session.rhd")
        assert page.validate() is None
        state = WizardState()
        page.collect_state(state)
        assert state.timing == "offset"
        assert state.offset_s == pytest.approx(-0.4)
        assert state.recording_interface == "IntanRecordingInterface"

    def test_triggered_offers_onsets_and_needs_the_line(self, qapp, app_state):
        page = TimingPage(app_state)
        page.set_mode("triggered")
        assert not page._rb_offset.isVisibleTo(page)
        page._rec_file.setText("session.rhd")
        assert "trigger" in page.validate().lower()
        page._trigger_line.setText("DIGITAL-IN-01")
        assert page.validate() is None
        state = WizardState()
        page.collect_state(state)
        assert state.timing == "onsets"
        assert state.trigger_line == "DIGITAL-IN-01"
        assert state.split_files is False


class TestWritePage:
    def test_pair_asks_for_a_session_file_and_names_the_notebook(self, qapp, app_state, tmp_path: Path):
        page = WritePage(app_state)
        state = WizardState(mode="pair")
        state.video.enabled = True
        state.video.folder_path = str(tmp_path / "sess" / "video")
        page.populate_from_state(state)
        assert state.session_dir == str(tmp_path / "sess")
        assert page._output.isVisibleTo(page)
        assert page.validate(state) is None
        page.collect_state(state)
        assert state.rig_name == "sess"
        assert state.notebook_path.endswith("sess.ipynb")
        assert state.output_path.endswith("session.nc")

    def test_triggered_writes_only_the_notebook(self, qapp, app_state, tmp_path: Path):
        page = WritePage(app_state)
        state = WizardState(mode="triggered", timing="onsets")
        state.video.enabled = True
        state.video.folder_path = str(tmp_path / "sess" / "video")
        page.populate_from_state(state)
        assert not page._output.isVisibleTo(page)
        assert "timing (onsets)" in page._cells.text()
