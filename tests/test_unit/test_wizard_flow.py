"""The wizard walks its route from the answers and hands the library what each mode needs."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from ethograph.gui import wizard_overview
from ethograph.gui.wizard_overview import NCWizardDialog, _rig_spec, _source_specs
from ethograph.gui.wizard_state import WizardState


class _AppState:
    project_path = None
    last_browse_dir = None
    nc_file_path = ""
    video_folder = ""
    audio_folder = ""
    pose_folder = ""
    nwb_alignment = None


class _Edit:
    def __init__(self):
        self.text = ""

    def setText(self, value):
        self.text = value


class _IOWidget:
    def __init__(self):
        self.nc_file_path_edit = _Edit()
        self.video_folder_edit = _Edit()
        self.pose_folder_edit = _Edit()
        self.audio_folder_edit = _Edit()


@pytest.fixture
def session(tmp_path: Path) -> Path:
    for i in (1, 2, 3):
        (tmp_path / "video").mkdir(exist_ok=True)
        (tmp_path / "video" / f"cam1_trial00{i}.mp4").touch()
        (tmp_path / "pose").mkdir(exist_ok=True)
        (tmp_path / "pose" / f"dlc_cam1_trial00{i}.h5").touch()
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "session_ch1.wav").touch()
    return tmp_path


def _pair_state(session: Path) -> WizardState:
    state = WizardState(mode="pair", session_dir=str(session), rig_name="rig")
    state.video.enabled = True
    state.video.folder_path = str(session / "video")
    state.video.file_mode = "aligned_to_trial"
    state.video.fps = 30
    state.audio.enabled = True
    state.audio.file_mode = "aligned_to_session"
    state.audio.files = [str(session / "audio" / "session_ch1.wav")]
    state.audio.folder_path = str(session / "audio")
    state.audio.audio_sr = 48000.0
    state.audio.constant_offset = -0.4
    return state


class TestAdapters:
    def test_session_wide_audio_is_paired_not_discovered(self, session: Path):
        state = _pair_state(session)
        assert [s.stream for s in _source_specs(state)] == ["video"]
        spec = _rig_spec(state)
        audio = next(s for s in spec.sources if s.stream == "audio")
        assert audio.session_wide is True
        assert audio.offset_s == pytest.approx(-0.4)
        assert audio.folder == str(Path("audio") / "session_ch1.wav")
        assert spec.timing == "none"

    def test_notebook_for_pair_mode_pairs_the_wav_with_its_offset(self, session: Path, tmp_path: Path):
        from ethograph.gui.wizard_notebook import build_notebook

        nb = build_notebook(_rig_spec(_pair_state(session)))
        code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "session_wide" in code and "-0.4" in code
        assert "neuroconv" not in code


class TestDialogRoute:
    def test_pair_mode_walks_sources_table_write_and_writes_the_notebook(self, qtbot, session: Path, monkeypatch):
        monkeypatch.setattr("ethograph.gui.wizard_single.get_video_fps", lambda _p: 30)
        monkeypatch.setattr(wizard_overview, "notify_dialog", lambda *a, **k: None)
        written: dict = {}

        def fake_build(state):
            written["state"] = state
            raise RuntimeError("stop before building the session file")

        monkeypatch.setattr("ethograph.gui.wizard_multi_builder.build_multi_trial_dt", fake_build)

        class _NoModal:
            """The real one spins a modal loop; run the job inline instead."""

            was_cancelled = False

            def __init__(self, *_a, **_k):
                pass

            def execute(self, fn, *args, **kwargs):
                try:
                    return fn(*args, **kwargs), None
                except RuntimeError as exc:
                    return None, exc

        monkeypatch.setattr("ethograph.gui.dialog_busy_progress.BusyProgressDialog", _NoModal)

        app_state = _AppState()
        app_state.project_path = str(session / "project")
        (session / "project").mkdir()
        dlg = NCWizardDialog(app_state, _IOWidget())
        qtbot.addWidget(dlg)

        dlg._on_next()  # mode → sources
        assert dlg._route[dlg._pos] is dlg._page_sources
        dlg._page_sources._pose_cb.setChecked(True)
        dlg._on_next()  # sources → per-modality folders/patterns
        assert dlg._route[dlg._pos] is dlg._page_patterns
        dlg._page_patterns._tab_map["video"]._stream_panel.set_folder(str(session / "video"))
        dlg._page_patterns._tab_map["pose"]._stream_panel.set_folder(str(session / "pose"))
        dlg._on_next()  # → trial table (single device each: natural sort, no pattern drawn)
        assert dlg._route[dlg._pos] is dlg._page_trials
        table = dlg._page_trials._auto_df
        assert list(table.columns) == ["trial", "video_cam-1", "pose_cam-1"]
        assert len(table) == 3
        dlg._on_next()  # table → write
        assert dlg._route[dlg._pos] is dlg._page_write
        assert dlg._page_write._notebook.text().endswith(str(Path("project") / "wizard" / f"{session.name}.ipynb"))
        dlg._on_next()  # write: notebook first, then the build (stubbed)
        nb_path = Path(dlg._state.notebook_path)
        assert nb_path.is_file()
        nb = nbformat.read(str(nb_path), as_version=4)
        assert nb.cells[1].metadata["tags"] == ["parameters"]
        assert written["state"].mode == "pair"

    def test_triggered_mode_visits_timing_and_only_writes_the_notebook(self, qtbot, session: Path, monkeypatch):
        monkeypatch.setattr("ethograph.gui.wizard_single.get_video_fps", lambda _p: 30)
        monkeypatch.setattr(wizard_overview, "notify_dialog", lambda *a, **k: None)
        app_state = _AppState()
        app_state.project_path = str(session / "project")
        (session / "project").mkdir()
        dlg = NCWizardDialog(app_state, _IOWidget())
        qtbot.addWidget(dlg)
        dlg._page_mode._blocks["triggered"].radio.setChecked(True)
        dlg._on_next()
        dlg._on_next()  # sources → per-modality folders/patterns
        assert dlg._route[dlg._pos] is dlg._page_patterns
        dlg._page_patterns._tab_map["video"]._stream_panel.set_folder(str(session / "video"))
        dlg._on_next()
        assert dlg._route[dlg._pos] is dlg._page_timing
        dlg._page_timing._rec_file.setText(str(session / "ephys" / "session.rhd"))
        dlg._page_timing._trigger_line.setText("DIGITAL-IN-01")
        dlg._on_next()
        assert dlg._route[dlg._pos] is dlg._page_trials
        dlg._on_next()
        assert dlg._route[dlg._pos] is dlg._page_write
        assert not dlg._page_write._output.isVisibleTo(dlg._page_write)
        dlg._on_next()
        nb = nbformat.read(dlg._state.notebook_path, as_version=4)
        code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "start_at(" in code and "DIGITAL-IN-01" in code
        assert dlg.result() == 1  # accepted without building anything
