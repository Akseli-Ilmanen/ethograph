"""What the Data wizard collects, page by page (Qt-free)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, get_args

import pandas as pd
from movement.io import load_dataset

if TYPE_CHECKING:
    from ethograph.gui.wizard_media_files import FilePattern

AVAILABLE_SOFTWARES: list[str] = list(get_args(load_dataset.__annotations__["source_software"]))

#: The three modes of page 0.
MODES = ("pair", "free_running", "triggered")
#: How the recording system knows about the camera (modes 2 and 3).
TIMINGS = ("none", "offset", "pulses", "onsets")


@dataclass
class ModalityConfig:
    enabled: bool = False
    file_mode: str = "single"  # "single" | "aligned_to_trial" | "aligned_to_session"
    single_file_path: str = ""
    folder_path: str = ""
    files: list[str] = field(default_factory=list)  # explicit list; wins over folder_path
    n_devices: int = 1
    pattern: FilePattern | None = None
    nested_subfolders: bool = False
    fps: int | None = None
    fps_by_camera: dict[str, int] = field(default_factory=dict)
    audio_sr: float | None = None
    n_channels: int = 1
    source_software: str = "DeepLabCut"
    constant_offset: float = 0.0
    video_motion: bool = False
    neurons_path: str = ""
    ephys_sr: int | None = None
    gap_mode: str = "gap_between"  # "gap_between" | "onset_interval"
    gap_value: float = 0.0
    offset_constant_across_devices: bool = True
    device_offsets: dict[str, float] = field(default_factory=dict)

    @property
    def is_aligned_mode(self) -> bool:
        return self.file_mode == "aligned_to_trial"

    @property
    def is_continuous_mode(self) -> bool:
        return self.file_mode == "aligned_to_session"


@dataclass
class WizardState:
    video: ModalityConfig = field(default_factory=ModalityConfig)
    pose: ModalityConfig = field(default_factory=ModalityConfig)
    audio: ModalityConfig = field(default_factory=ModalityConfig)
    npy: ModalityConfig = field(default_factory=ModalityConfig)
    ephys: ModalityConfig = field(default_factory=ModalityConfig)

    mode: str = "pair"
    n_cameras: int = 1
    n_mics: int = 0

    # Modes 2 and 3: the notebook's timing cell.
    timing: str = "none"
    split_files: bool = False
    offset_s: float | None = None
    recording_file: str | None = None
    recording_interface: str = "IntanRecordingInterface"
    frame_line: str | None = None
    trigger_line: str | None = None
    burst_gap: float = 10.0

    files_aligned_to_trials: bool = True
    trial_table: pd.DataFrame | None = None
    trial_table_path: str | None = None
    nwb_alignment: object | None = None

    camera_names: list[str] = field(default_factory=list)
    mic_names: list[str] = field(default_factory=list)
    pose_camera_mapping: list[tuple[str, str]] = field(default_factory=list)
    individuals: list[str] = field(default_factory=list)

    session_dir: str = ""
    rig_name: str = ""
    notebook_path: str = ""
    output_path: str = ""
    file_durations: dict[str, dict[str, float]] = field(default_factory=dict)

    def modality_configs(self) -> list[tuple[str, ModalityConfig]]:
        return [
            ("video", self.video),
            ("pose", self.pose),
            ("audio", self.audio),
            ("npy", self.npy),
            ("ephys", self.ephys),
        ]

    def enabled_modalities(self) -> list[tuple[str, ModalityConfig]]:
        return [(name, cfg) for name, cfg in self.modality_configs() if cfg.enabled]

    def has_aligned_modalities(self) -> bool:
        return any(cfg.is_aligned_mode for _, cfg in self.enabled_modalities())

    def has_continuous_modalities(self) -> bool:
        return any(cfg.is_continuous_mode for _, cfg in self.enabled_modalities())

    def is_fully_aligned(self) -> bool:
        non_ephys = [(n, c) for n, c in self.enabled_modalities() if n != "ephys"]
        return (
            bool(non_ephys)
            and all(cfg.is_aligned_mode for _, cfg in non_ephys)
            and not self.has_continuous_modalities()
        )
