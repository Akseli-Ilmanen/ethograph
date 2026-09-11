"""The Data wizard's pages hand the state what each mode needs and nothing it forbids."""

from __future__ import annotations

from pathlib import Path

import pytest

from ethograph.gui.wizard_pages import FIGURES, ModePage, SourcesPage, TimingPage, WritePage
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


class TestSourcesPage:
    """No camera/mic-count question here any more: every source's folder,
    optional filename pattern, and per-device settings live on the next
    page (``ModalityConfigPage``), which answers a single device just as
    well by leaving the pattern blank."""

    def test_pair_mode_never_enables_ephys(self, qapp, app_state):
        page = SourcesPage(app_state)
        page.set_mode("pair")
        state = WizardState()
        page.collect_state(state)
        assert state.ephys.enabled is False
        assert state.video.enabled and state.video.is_aligned_mode
        assert state.audio.enabled is False

    def test_free_running_makes_video_session_wide(self, qapp, app_state):
        page = SourcesPage(app_state)
        page.set_mode("free_running")
        state = WizardState()
        page.collect_state(state)
        assert state.video.is_continuous_mode

    def test_validation_needs_at_least_one_source(self, qapp, app_state):
        page = SourcesPage(app_state)
        page.set_mode("pair")
        assert page.validate() is None  # video ticked by default
        page._video_cb.setChecked(False)
        assert page.validate() is not None
        page._pose_cb.setChecked(True)
        assert page.validate() is None

    def test_video_is_optional(self, qapp, app_state):
        page = SourcesPage(app_state)
        page.set_mode("pair")
        page._video_cb.setChecked(False)
        page._pose_cb.setChecked(True)
        assert page.validate() is None
        state = WizardState()
        page.collect_state(state)
        assert state.video.enabled is False
        assert state.pose.enabled and state.pose.is_aligned_mode

    def test_modes_two_and_three_require_video(self, qapp, app_state):
        page = SourcesPage(app_state)
        page.set_mode("triggered")
        page._video_cb.setChecked(False)
        assert page._video_cb.isChecked()

    def test_audio_layout_choice_sets_file_mode(self, qapp, app_state):
        page = SourcesPage(app_state)
        page.set_mode("pair")
        page._audio_cb.setChecked(True)
        state = WizardState()
        page.collect_state(state)
        assert state.audio.is_aligned_mode  # "one file per trial" is the default

        page._audio_session.setChecked(True)
        page.collect_state(state)
        assert state.audio.is_continuous_mode

    def test_free_running_forces_audio_session_wide(self, qapp, app_state):
        page = SourcesPage(app_state)
        page._audio_cb.setChecked(True)
        page.set_mode("free_running")
        assert page._audio_session.isChecked()
        assert not page._audio_per_trial.isEnabled()


class TestFolderOrFilesDragAndDrop:
    def test_dropping_a_folder_seeds_it(self, qapp, app_state, tmp_path: Path):
        from ethograph.gui.wizard_pages import _FolderOrFiles

        video_dir = tmp_path / "video"
        video_dir.mkdir()
        picker = _FolderOrFiles(app_state, "video")

        class _FakeMime:
            def hasUrls(self_inner):
                return True

            def urls(self_inner):
                from qtpy.QtCore import QUrl

                return [QUrl.fromLocalFile(str(video_dir))]

        class _FakeEvent:
            def mimeData(self_inner):
                return _FakeMime()

            def acceptProposedAction(self_inner):
                pass

        picker.dropEvent(_FakeEvent())
        assert Path(picker.folder) == video_dir

    def test_dropping_files_matching_the_stream_sets_the_file_list(self, qapp, app_state, tmp_path: Path):
        from ethograph.gui.wizard_pages import _FolderOrFiles

        a = tmp_path / "cam1_trial001.mp4"
        b = tmp_path / "cam1_trial002.mp4"
        a.touch()
        b.touch()
        picker = _FolderOrFiles(app_state, "video")

        class _FakeMime:
            def hasUrls(self_inner):
                return True

            def urls(self_inner):
                from qtpy.QtCore import QUrl

                return [QUrl.fromLocalFile(str(a)), QUrl.fromLocalFile(str(b))]

        class _FakeEvent:
            def mimeData(self_inner):
                return _FakeMime()

            def acceptProposedAction(self_inner):
                pass

        picker.dropEvent(_FakeEvent())
        assert sorted(Path(f) for f in picker.files) == sorted([a, b])


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
