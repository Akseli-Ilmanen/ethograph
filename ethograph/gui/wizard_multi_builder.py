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
        pose_name = _get_file_for_trial(row, "pose")
        if pose_name:
            pose_path = resolve_media_path(state.pose, pose_name)
            if pose_path is None:
                raise ValueError(f"Trial {trial_id}: pose file {pose_name!r} is not in the pose folder")
            ds = _load_pose_into_ds(ds, str(pose_path), state.pose)

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

    pair_media(
        trial_table=with_media_paths(trial_table, state),
        stream_rates=stream_rates,
        session_wide=session_wide or None,
        output_path=nwb_path,
        pose_fps=fps,
        individuals=state.individuals or None,
    )

    return nwb_path


def with_media_paths(table: pd.DataFrame, state: WizardState) -> pd.DataFrame:
    """The trial table with every media cell resolved to its file's full path.

    The wizard's table names files by basename; ``pair_media`` wants to know where
    they are, both to probe trial durations and to record the paths in the NWB. A
    name that resolves to no file raises, naming the trial and column.
    """
    table = table.copy()
    for stream in ("video", "audio", "pose"):
        cfg: ModalityConfig = getattr(state, stream)
        if not cfg.enabled:
            continue
        for col in [c for c in table.columns if c.startswith(f"{stream}_") and not c.endswith("_start")]:
            resolved: list[str] = []
            for trial, value in zip(table["trial"], table[col]):
                if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
                    resolved.append("")
                    continue
                path = resolve_media_path(cfg, str(value))
                if path is None:
                    raise ValueError(f"Trial {trial}: {col} names {value!r}, which is not in the {stream} folder")
                resolved.append(str(path))
            table[col] = resolved
    return table


def resolve_media_path(cfg: ModalityConfig, name: str) -> Path | None:
    """Map a trial-table cell (a bare filename or a full path) to an existing file of *cfg*.

    The trial table stores filenames only; the modality config knows where they live —
    first through the files its pattern was built from, then through its folder.
    """
    direct = Path(name)
    if direct.is_absolute() and direct.exists():
        return direct
    basename = direct.name
    known = list(cfg.pattern.files) if cfg.pattern else []
    known += [Path(f) for f in cfg.files]
    for candidate in known:
        if candidate.name == basename and candidate.exists():
            return candidate
    if cfg.folder_path:
        folder = Path(cfg.folder_path)
        flat = folder / basename
        if flat.exists():
            return flat
        if cfg.nested_subfolders:
            for candidate in folder.rglob(basename):
                return candidate
    return None
