"""The alignment dialog turns accepted pairs into a session whose audio plays at the found offset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy")
sf = pytest.importorskip("soundfile")

from ethograph.gui.dialog_audio_align import AudioAlignDialog  # noqa: E402
from ethograph.gui.project import drop_record_path  # noqa: E402
from ethograph.io.audio_align import ANALYSIS_RATE, AlignedPair  # noqa: E402
from ethograph.io.nwb_alignment import NWBAlignment  # noqa: E402

RATE = ANALYSIS_RATE


def _bursts(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * RATE), dtype=np.float32)
    for _ in range(int(seconds * 2)):
        start = rng.integers(0, len(out) - RATE // 4)
        out[start : start + RATE // 4] += rng.standard_normal(RATE // 4).astype(np.float32) * np.hanning(RATE // 4)
    return out


def _write(path: Path, signal: np.ndarray) -> str:
    sf.write(path, signal, RATE)
    return str(path)


@pytest.fixture
def media(tmp_path: Path, monkeypatch):
    """Two 'videos' whose soundtracks are WAVs standing in for the extract, and two mic files.

    Mic 1 started 2 s before camera 1; mic 2 started 1.5 s after camera 2. The
    audio files are dropped in the wrong order, so content has to pair them.
    """
    scenes = [_bursts(20.0, seed=1), _bursts(20.0, seed=2)]
    tracks = [_write(tmp_path / f"cam{i}.track.wav", s * 0.3) for i, s in enumerate(scenes, start=1)]
    videos = [str(tmp_path / f"cam{i}.mp4") for i in (1, 2)]
    for v in videos:
        Path(v).touch()

    def shifted(scene: np.ndarray, start_s: float) -> np.ndarray:
        out = np.zeros(len(scene), dtype=np.float32)
        i0 = int(start_s * RATE)
        if i0 >= 0:
            out[: len(scene) - i0] = scene[i0:]
        else:
            out[-i0:] = scene[: len(scene) + i0]
        return out

    mics = [
        _write(tmp_path / "mic_b.wav", shifted(scenes[1], 1.5)),
        _write(tmp_path / "mic_a.wav", shifted(scenes[0], -2.0)),
    ]
    import ethograph.gui.dialog_audio_align as mod

    monkeypatch.setattr(mod, "has_embedded_audio", lambda path: True)
    monkeypatch.setattr(mod, "ensure_extracted_audio", lambda path: Path(tracks[videos.index(str(path))]))
    monkeypatch.setattr(mod, "probe_duration", lambda path, stream: 20.0)
    import ethograph.io.audio_align as core
    import ethograph.utils.stream_durations as durations

    monkeypatch.setattr(core, "_video_fps", lambda path: 30.0)
    monkeypatch.setattr(durations, "probe_duration", lambda path, stream, fps=None: 20.0)
    return videos, mics


def test_analysis_pairs_by_content_and_places_each_file(qtbot, app_state, media):
    videos, mics = media
    dlg = AudioAlignDialog(app_state)
    qtbot.addWidget(dlg)
    dlg.add_videos(videos)
    dlg.add_audio(mics)

    pairs = dlg.analyse()

    assert [Path(p.audio).name for p in pairs] == ["mic_a.wav", "mic_b.wav"]
    assert pairs[0].offset_s == pytest.approx(-2.0, abs=0.02)
    assert pairs[1].offset_s == pytest.approx(1.5, abs=0.02)
    assert all(not p.accepted for p in pairs), "nothing is accepted for the user"


def test_session_is_written_only_once_every_row_is_accepted(qtbot, app_state, media, tmp_path: Path):
    videos, mics = media
    dlg = AudioAlignDialog(app_state)
    qtbot.addWidget(dlg)
    dlg.add_videos(videos)
    dlg.add_audio(mics)
    dlg.analyse()
    dlg._fill_table()
    assert not dlg._buttons.button(dlg._buttons.Ok).isEnabled()

    with pytest.raises(ValueError, match="accepted"):
        dlg.write_session(tmp_path)
    for pair in dlg.pairs:
        pair.accepted = True

    session = dlg.write_session(tmp_path)

    alignment = NWBAlignment(session / ".ethograph" / "alignment.nwb")
    assert alignment.stream_offset_for_trial(1, "audio", "mic-1") == pytest.approx(-2.0, abs=0.02)
    assert alignment.stream_offset_for_trial(2, "audio", "mic-1") == pytest.approx(1.5, abs=0.02)
    assert alignment.stream_offset_for_trial(1, "audio", "cam-1-track") == 0.0
    assert Path(alignment.resolve_media_path(1, "audio", "cam-1-track")).name == "video-track_cam1.wav"
    assert drop_record_path(session).is_file()
    assert app_state.nc_file_path == str(session)


def test_changing_a_row_s_file_swaps_with_the_row_that_held_it(qtbot, app_state, media):
    videos, mics = media
    dlg = AudioAlignDialog(app_state)
    qtbot.addWidget(dlg)
    dlg.add_videos(videos)
    dlg.add_audio(mics)
    dlg.analyse()
    dlg._fill_table()

    dlg._on_audio_changed(0, 1)  # row 0 takes mic_b, which row 1 held

    assert [Path(p.audio).name for p in dlg.pairs] == ["mic_b.wav", "mic_a.wav"]
    assert all(not p.accepted for p in dlg.pairs)


def test_offset_sentence_in_preview_title(qtbot, app_state, media):
    videos, mics = media
    dlg = AudioAlignDialog(app_state)
    qtbot.addWidget(dlg)
    dlg.add_videos(videos)
    dlg.add_audio(mics)
    dlg.analyse()
    dlg._fill_table()

    assert "audio starts 2.00 s before video" in dlg._preview_title.text()
    assert AlignedPair(videos[0], mics[1], -2.0, 20.0).describe_offset() in dlg._preview_title.text()
