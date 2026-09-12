"""The pattern step (2+ cameras/mics) must not make the user re-pick a folder
they already chose on the Sources page."""

from __future__ import annotations

from pathlib import Path

import pytest

from ethograph.gui.wizard_multi_tabs import AudioConfigTab, ModalityConfigPage, PoseConfigTab, VideoConfigTab
from ethograph.gui.wizard_state import ModalityConfig, WizardState


def test_video_pose_audio_tabs_inherit_the_folder_already_chosen(qapp, tmp_path: Path):
    video_dir = tmp_path / "video"
    pose_dir = tmp_path / "pose"
    audio_dir = tmp_path / "audio"
    for d in (video_dir, pose_dir, audio_dir):
        d.mkdir()

    state = WizardState()
    state.n_cameras = 2
    state.video.enabled = True
    state.video.folder_path = str(video_dir)
    state.video.n_devices = 2
    state.video.file_mode = "aligned_to_trial"
    state.pose.enabled = True
    state.pose.folder_path = str(pose_dir)
    state.pose.n_devices = 2
    state.pose.file_mode = "aligned_to_trial"
    state.audio.enabled = True
    state.audio.folder_path = str(audio_dir)
    state.audio.n_devices = 2
    state.audio.file_mode = "aligned_to_trial"

    page = ModalityConfigPage(state)

    assert page._tab_map["video"]._stream_panel._folder.text() == str(video_dir)
    assert page._tab_map["pose"]._stream_panel._folder.text() == str(pose_dir)
    assert page._tab_map["audio"]._stream_panel._folder.text() == str(audio_dir)


def test_a_single_device_needs_no_drawn_pattern(qapp, tmp_path: Path):
    """A camera count of one no longer exists as a question — a folder with
    no filename pattern painted is a valid, single-device answer (natural
    sort), the same as leaving 'Sources' folder-picking to this page."""
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "trial001.mp4").touch()
    (video_dir / "trial002.mp4").touch()

    config = ModalityConfig(enabled=True, file_mode="aligned_to_trial", folder_path=str(video_dir))
    tab = VideoConfigTab(config)
    assert tab.validate() is None  # no pattern painted; folder + files is enough

    tab.collect_state(config)
    assert config.pattern is None
    assert config.n_devices == 1


def test_pose_without_video_requires_an_explicit_frame_rate(qapp, tmp_path: Path):
    """A pose file carries no frame rate of its own; with no video to read it
    from, a hardcoded default would violate the 'never hardcode rates' rule."""
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    (pose_dir / "trial001.h5").touch()

    config = ModalityConfig(enabled=True, file_mode="aligned_to_trial", folder_path=str(pose_dir))
    tab = PoseConfigTab(config)
    assert "frame rate" in tab.validate()

    tab._no_video_fps_spin.setValue(60.0)
    assert tab.validate() is None
    tab.collect_state(config)
    assert config.fps == 60.0


def test_audio_offset_only_editable_in_free_running_mode(qapp, tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "session.wav").touch()

    config = ModalityConfig(enabled=True, file_mode="aligned_to_session", folder_path=str(audio_dir))
    pair_tab = AudioConfigTab(config, mode="pair")
    assert not pair_tab._const_offset_cb.isVisibleTo(pair_tab)
    pair_tab.collect_state(config)
    assert config.constant_offset == 0.0

    free_config = ModalityConfig(enabled=True, file_mode="aligned_to_session", folder_path=str(audio_dir))
    free_tab = AudioConfigTab(free_config, mode="free_running")
    assert free_tab._const_offset_cb.isVisibleTo(free_tab)
    free_tab._const_offset_spin.setValue(-3.0)
    free_tab.collect_state(free_config)
    assert free_config.constant_offset == pytest.approx(-3.0)
