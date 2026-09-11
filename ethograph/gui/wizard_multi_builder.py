"""Backend builder: assembles a TrialTree from WizardState (no Qt)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
from movement.io import load_dataset

from ethograph.gui.wizard_state import ModalityConfig, WizardState
from ethograph.io.trialtree import TrialTree

INTERVAL_COLUMNS = {"trial", "onset_s", "offset_s", "labels", "individual"}


def build_multi_trial_dt(state: WizardState) -> TrialTree:
    trial_table = state.trial_table
    if trial_table is None or trial_table.empty:
        raise ValueError("No trial table available. Go back and configure trials.")

    trial_ids = trial_table["trial"].tolist()
    datasets: list[xr.Dataset] = []

    individuals = state.individuals or ["individual_1"]

    if state.video.enabled:
        fps = state.video.fps
    elif state.pose.enabled:
        fps = state.pose.fps
    else:
        fps = None  # audio-only: no frame clock to record

    for i, trial_id in enumerate(trial_ids):
        ds = _build_single_trial_ds(state, trial_table, i, trial_id, fps, individuals)
        datasets.append(ds)

    dt = TrialTree.from_datasets(datasets, validate=True)

    # Build NWB file with trials table + acquisition items
    nwb_path = _build_nwb_file(dt, state, trial_table, trial_ids, fps)
    if nwb_path:
        from ethograph.io.nwb_alignment import make_nwb_alignment

        state.nwb_alignment = make_nwb_alignment(nwb_path)

    return dt


def _build_single_trial_ds(
    state: WizardState,
    trial_table: pd.DataFrame,
    trial_idx: int,
    trial_id,
    fps: float | None,
    individuals: list[str],
) -> xr.Dataset:
    row = trial_table.iloc[trial_idx]

    ds = xr.Dataset(coords={"individuals": individuals})
    ds.attrs["trial"] = trial_id
    if fps is not None:
        ds.attrs["fps"] = fps

    if state.pose.enabled:
        pose_path = _get_file_for_trial(row, "pose")
        if pose_path:
            ds = _load_pose_into_ds(ds, pose_path, state.pose)

    return ds


def _get_file_for_trial(row: pd.Series, modality: str) -> str | None:
    for col in row.index:
        if col.startswith(modality) and col != "trial":
            val = row[col]
            if pd.notna(val) and str(val):
                return str(val)
    return None


def _load_pose_into_ds(
    ds: xr.Dataset,
    pose_path: str,
    cfg: ModalityConfig,
) -> xr.Dataset:
    pose_ds = load_dataset(
        pose_path,
        source_software=cfg.source_software,
    )

    ds.attrs["source_software"] = cfg.source_software

    if "position" in pose_ds:
        time_coord = pose_ds["position"].coords[next(c for c in pose_ds["position"].coords if "time" in str(c))]

        ds.coords["time"] = time_coord
        for var_name in pose_ds.data_vars:
            ds[var_name] = pose_ds[var_name]
        for coord_name in pose_ds.coords:
            if coord_name not in ds.coords:
                ds.coords[coord_name] = pose_ds.coords[coord_name]
    return ds


def _build_nwb_file(
    dt: TrialTree,
    state: WizardState,
    trial_table: pd.DataFrame,
    trial_ids: list,
    fps: float,
) -> Path | None:
    """Create an alignment NWB file from wizard state.

    Writes to ``.ethograph/alignment.nwb`` relative to the output path,
    or falls back to a temp location.
    """
    from ethograph.io.pairing import pair_media

    if state.output_path:
        output_dir = Path(state.output_path).parent
    else:
        output_dir = Path.cwd()

    nwb_path = output_dir / ".ethograph" / "alignment.nwb"

    stream_rates: dict[str, float] = {}
    if state.video.enabled and fps:
        stream_rates["video"] = float(fps)
    if state.pose.enabled and fps:
        stream_rates["pose"] = float(fps)
    session_wide: dict[str, tuple[str, float, float]] = {}
    if state.audio.enabled and state.audio.audio_sr:
        if state.audio.is_continuous_mode:
            file = state.audio.files[0] if state.audio.files else state.audio.single_file_path
            session_wide["audio_mic-1"] = (str(file), float(state.audio.audio_sr), float(state.audio.constant_offset))
        else:
            stream_rates["audio"] = float(state.audio.audio_sr)

    table = trial_table.copy()
    if not {"start_time", "stop_time"}.issubset(table.columns):
        table = _infer_trial_times(table, state, fps)

    pair_media(
        trial_table=table,
        stream_rates=stream_rates,
        session_wide=session_wide or None,
        output_path=nwb_path,
    )

    return nwb_path


def _infer_trial_times(table: pd.DataFrame, state: WizardState, fps: float | None) -> pd.DataFrame:
    """Compute start_time / stop_time for each trial by probing media file durations."""
    table = table.copy()
    starts: list[float] = []
    stops: list[float] = []
    cursor = 0.0
    for _, row in table.iterrows():
        dur = _probe_row_duration(row, state, fps)
        starts.append(cursor)
        stops.append(cursor + dur)
        cursor += dur
    table["start_time"] = starts
    table["stop_time"] = stops
    return table


def _probe_row_duration(row: pd.Series, state: WizardState, fps: float | None) -> float:
    """Return the duration (seconds) of a trial by probing its media files."""
    from ethograph.utils.stream_durations import probe_duration

    for stream in ["video", "audio", "pose"]:
        cfg = getattr(state, stream)
        if not cfg.enabled:
            continue
        stream_fps = fps if stream == "pose" else None
        for col in row.index:
            if not col.startswith(f"{stream}_"):
                continue
            path_str = row.get(col)
            if not path_str or pd.isna(path_str):
                continue
            path = Path(str(path_str))
            if not path.exists():
                continue
            dur = probe_duration(str(path), stream, stream_fps)
            if dur is not None:
                return dur

    raise ValueError(f"Could not probe duration for trial row: {row.to_dict()}")
