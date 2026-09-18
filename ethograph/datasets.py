"""Unified dataset registry — single source of truth for all example datasets.

Merges what was previously split across:
- ``EXAMPLE_DATASETS`` in ``utils/download.py`` (download manifests)
- ``TEMPLATES`` in ``gui/dialog_select_template.py`` (GUI card definitions)
- ``EXAMPLE_CONFIGS`` in ``utils/download.py`` (per-dataset config files)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ethograph.labels.onset_curves import RUN_PREFIX
from ethograph.utils.paths import cache_dir

if TYPE_CHECKING:
    import xarray as xr

DOWNLOAD_BASE = cache_dir("example_data")

DATASETS: dict[str, dict] = {
    "moll2025": {
        # Display
        "name": "Moll et al., 2025 — Tool-using crows",
        "image": "moll1.png",
        "paper_url": "https://doi.org/10.1016/j.cub.2025.08.033",
        # Structure
        "folder": "Moll2025",
        "nc_filename": "Trial_data.nc",
        "has_audio": False,
        "import_labels": True,
        # Reference geometry from the bundled defaults (ethograph/defaults/config/space/)
        "library_geometry": "moll2025",
        # The trial number is the recording number in the filename, so these
        # rows are in date order while dt.trials is numerically sorted --
        # "trial" is what pairs them, never the row position.
        "media": [
            {
                "trial": 115,
                "video_cam-1": "2024-12-17_115_Crow1-cam-1.mp4",
                "pose_cam-1": "2024-12-17_115_Crow1-cam-1DLC.csv",
            },
            {
                "trial": 41,
                "video_cam-1": "2024-12-18_041_Crow1-cam-1.mp4",
                "pose_cam-1": "2024-12-18_041_Crow1-cam-1DLC.csv",
            },
        ],
        # Download
        "release_tag": "moll2025",
        # Ship the authored alignment.nwb as a release asset instead of
        # rebuilding it locally — it carries the real per-trial timing, which
        # probing the media cannot reproduce.
        "download_alignment": True,
        "size_mb": 14,
        # Optional per-video embeddings, one folder each, offered as an
        # unchecked box on the template card. Files are ``{video stem}.npy``
        # so the folder imports as a video feature (add panel → import).
        "video_feature_folders": {
            "embeddings_feral": {"release_tag": "v0.2.35", "size_mb": 7},
            "embeddings_s3d": {"release_tag": "v0.2.34", "size_mb": 18},
        },
        "extra_gui_assets": ["Trial_data_labels.tsv"],
        # Prediction runs of different models over this session, shipped so
        # they can be stacked against each other (File → Import predictions →
        # From folder). Each suffix names a release asset pair
        # ``{nc stem}_predictions_{suffix}.tsv`` + ``{nc stem}_probs_{suffix}.npz``
        # and lands in ``labels/predictions_{suffix}/``.
        "prediction_runs": ["mlp", "c2f-tcn", "feral-cp", "feral-nocp"],
        "assets_notebook": [
            "Trial_data.nc",
            "2024-12-17_115_Crow1-cam-1.mp4",
            "2024-12-17_115_Crow1-cam-1DLC.csv",
            "2024-12-17_115_Crow1-cam-2DLC.csv",
            "2024-12-17_115_Crow1_DLC_3D.csv",
            "2024-12-17_115_Crow1-cam-1_s3d.npy",
            "2024-12-18_041_Crow1-cam-1.mp4",
            "2024-12-18_041_Crow1-cam-1DLC.csv",
            "2024-12-18_041_Crow1-cam-2DLC.csv",
            "2024-12-18_041_Crow1_DLC_3D.csv",
            "2024-12-18_041_Crow1-cam-1_s3d.npy",
        ],
        "assets_pynapple": [
            "beakTip_speed.npz",
            "beakTip_velocity.npz",
            "beakTip_position.npz",
            "trials.npz",
        ],
        # Configs written to dest/.ethograph/ (existing files are kept)
        "configs": {
            "mapping.txt": (
                "0 background\n"
                "1 pullOutStick\n"
                "2 diagonalToBox\n"
                "3 toss\n"
                "4 swoop\n"
                "5 reachLeftCorner\n"
                "6 right\n"
                "7 pullOutAlongWall\n"
                "8 swoopOut\n"
                "9 stickToDisp\n"
                "10 stickInDisp\n"
                "11 lookToPellet\n"
                "12 snapPellet\n"
                "13 eat\n"
                "14 reachRightCorner\n"
                "15 nodding\n"
                "31 pelletStickFirstContact 1 point\n"
                "32 pelletStickLastContact 1 point\n"
            ),
            "local_settings.yaml": (
                "colors_sel: None\n"
                "features_sel: speed\n"
                "features_sel_previous: confidence\n"
                # Keys are dim names: this dataset is movement-style singular.
                "individual_sel: Crow1\n"
                "keypoint_sel: beakTip\n"
                "keypoint_sel_previous: beakTip\n"
                "s3d_dims_sel: '0'\n"
                "s3d_dims_sel_previous: '0'\n"
                "space_sel_previous: x\n"
                "trials_sel: 41\n"
                "files_aligned_to_trials: true\n"
                "labels_visible: true\n"
                "feature_view_mode: LinePlot\n"
                "space_library_geometry: moll2025\n"
                "space_plot_type: Space Plot\n"
                "ephys_offset: 0.0\n"
                "panel_layout:\n"
                "  panels:\n"
                "  - type: lineplot\n"
                "    feature: speed\n"
                "    selections:\n"
                "      individuals: Crow1\n"
                "      keypoint: beakTip\n"
                "      keypoints: beakTip\n"
                "  - type: lineplot\n"
                "    feature: velocity\n"
                "    selections:\n"
                "      individuals: Crow1\n"
                "      s3d_dims: '0'\n"
                "      keypoints: beakTip\n"
                "  dock_state_b64: AAAA/wAAAAD9AAAAAQAAAAAAAAVIAAACNvwCAAAABvsAAAASAHAAYQBuAGUAbABfAG4AZQBvAAAAAAD/////AAAAWQD////7AAAAFgBwAGEAbgBlAGwAXwBlAHAAaAB5AHMAAAAAAP////8AAABZAP////sAAAAYAHAAYQBuAGUAbABfAHIAYQBzAHQAZQByAAAAAAD/////AAAAWQD////7AAAAGgBwAGEAbgBlAGwAXwBoAGUAYQB0AG0AYQBwAAAAAAD/////AAAAWQD////7AAAAIABwAGEAbgBlAGwAXwBsAGkAbgBlAHAAbABvAHQAXwAwAQAAAAAAAAEbAAAAWQD////7AAAAIABwAGEAbgBlAGwAXwBsAGkAbgBlAHAAbABvAHQAXwAxAQAAASEAAAEVAAAAWQD///8AAAAAAAACNgAAAAQAAAAEAAAACAAAAAj8AAAAAA==\n"  # noqa: E501
                "  space_plots:\n"
                "  - feature: position\n"
                "    view_3d: true\n"
                "    space_dim: space\n"
                "    x: x\n"
                "    y: y\n"
                "    z: z\n"
                "    dims:\n"
                "      keypoints: stickTip\n"
                "      individuals: Crow1\n"
                "    color: Labels\n"
                "  - feature: position\n"
                "    view_3d: true\n"
                "    space_dim: space\n"
                "    x: x\n"
                "    y: y\n"
                "    z: z\n"
                "    dims:\n"
                "      keypoints: beakTip\n"
                "      individuals: Crow1\n"
                "    color: Labels\n"
                "  shell_dock_state_b64: AAAA/wAAAAL9AAAAAwAAAAEAAAIwAAAD2fwCAAAAAfsAAAAWAFMAaQBkAGUAYgBhAHIARABvAGMAawEAAAAVAAAD2QAAAFgA////AAAAAgAABUgAAAFw/AEAAAAD+wAAABIAVgBpAGQAZQBvAEQAbwBjAGsBAAAAAAAAAf4AAAB4AP////sAAAAeAFMAcABhAGMAZQBQAGwAbwB0AEQAbwBjAGsAXwAwAQAAAgQAAAGDAAAAeAD////7AAAAHgBTAHAAYQBjAGUAUABsAG8AdABEAG8AYwBrAF8AMQEAAAONAAABuwAAAHgA////AAAAAwAABUgAAAAn/AEAAAAB+wAAABoAQgBvAHQAdABvAG0AQgBhAHIARABvAGMAawEAAAAAAAAFSAAAAgsA////AAAFSAAAAjYAAAABAAAAAgAAAAEAAAAC/AAAAAA=\n"  # noqa: E501
            ),
        },
    },
    "birdpark": {
        "name": "Rüttimann et al., 2025 — Zebra finches in BirdPark",
        "image": "birdpark0.png",
        "paper_url": "https://doi.org/10.7717/peerj.20203",
        "folder": "BirdPark",
        "nc_filename": "copExpBP08_trim.nc",
        "has_audio": True,
        "media": [
            {
                "video_cam-1": "BP_2021-05-25_08-12-51_655154_0380000.mp4",
                "audio_mic-1": "BP_2021-05-25_08-12-51_655154_0380000.wav",
            },
        ],
        "release_tag": "birdpark",
        # Ship the authored alignment.nwb as a release asset instead of
        # rebuilding it locally — it carries the real per-trial timing, which
        # probing the media cannot reproduce.
        "download_alignment": True,
        "size_mb": 76,
    },
    "philodoptera": {
        "name": "Philodoptera — Motor control of sound production in crickets",
        "image": "cricket0.png",
        "paper_url": "",
        "folder": "Philodoptera",
        "nc_filename": "philodoptera.nc",
        "has_audio": True,
        "media": [
            {
                "video_cam-1": "philodoptera.mp4",
                "audio_mic-1": "philodoptera.wav",
                "pose_cam-1": "philodoptera.csv",
            },
        ],
        "release_tag": "philodoptera",
        "size_mb": 4,
    },
    "lockbox": {
        "name": "Reiske et al., 2025 — Mouse Lockbox",
        "image": "lockbox2.gif",
        "paper_url": "https://doi.org/10.1007/s11263-026-02908-x ",
        "folder": "Lockbox",
        "nc_filename": "lockbox.nc",
        "has_audio": False,
        "media": [
            {
                "video_front-view": "2021-02-15_07-32-44_segment1_mouse324_ball_front-view.avi",
                "video_side-view": "2021-02-15_07-32-44_segment1_mouse324_ball_side-view.avi",
                "video_top-down-view": "2021-02-15_07-32-44_segment1_mouse324_ball_top-down-view.avi",
                "pose_front-view": "2021-02-15_07-32-44_segment1_mouse324_ball_front-view-tracks_individual_0.csv",
                "pose_side-view": "2021-02-15_07-32-44_segment1_mouse324_ball_side-view-tracks_individual_0.csv",
                "pose_top-down-view": "2021-02-15_07-32-44_segment1_mouse324_ball_top-down-view-tracks_individual_0.csv",  # noqa: E501
            },
            {
                "video_front-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_front-view.avi",
                "video_side-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_side-view.avi",
                "video_top-down-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_top-down-view.avi",
                "pose_front-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_front-view-tracks_individual_0.csv",  # noqa: E501
                "pose_side-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_side-view-tracks_individual_0.csv",  # noqa: E501
                "pose_top-down-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_top-down-view-tracks_individual_0.csv",  # noqa: E501
            },
            {
                "video_front-view": "2021-05-31_07-34-21_segment3_mouse291_stick_front-view.avi",
                "video_side-view": "2021-05-31_07-34-21_segment3_mouse291_stick_side-view.avi",
                "video_top-down-view": "2021-05-31_07-34-21_segment3_mouse291_stick_top-down-view.avi",
                "pose_front-view": "2021-05-31_07-34-21_segment3_mouse291_stick_front-view-tracks_individual_0.csv",
                "pose_side-view": "2021-05-31_07-34-21_segment3_mouse291_stick_side-view-tracks_individual_0.csv",
                "pose_top-down-view": "2021-05-31_07-34-21_segment3_mouse291_stick_top-down-view-tracks_individual_0.csv",  # noqa: E501
            },
        ],
        "release_tag": "lockbox",
        "size_mb": 70,
        "assets_notebook": [
            "lockbox.nc",
            "2021-02-15_07-32-44_segment1_mouse324_ball_front-view.avi",
            "2021-02-15_07-32-44_segment1_mouse324_ball_front-view-tracks_individual_0.csv",
            "2021-02-15_07-32-44_segment1_mouse324_ball_side-view.avi",
            "2021-02-15_07-32-44_segment1_mouse324_ball_side-view-tracks_individual_0.csv",
            "2021-02-15_07-32-44_segment1_mouse324_ball_top-down-view.avi",
            "2021-02-15_07-32-44_segment1_mouse324_ball_top-down-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_front-view.avi",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_front-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_side-view.avi",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_side-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_top-down-view.avi",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_top-down-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment3_mouse291_stick_front-view.avi",
            "2021-05-31_07-34-21_segment3_mouse291_stick_front-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment3_mouse291_stick_side-view.avi",
            "2021-05-31_07-34-21_segment3_mouse291_stick_side-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment3_mouse291_stick_top-down-view.avi",
            "2021-05-31_07-34-21_segment3_mouse291_stick_top-down-view-tracks_individual_0.csv",
        ],
        # Configs written to dest/.ethograph/ (existing files are kept). Loading
        # all three camera views by default: front-view as the primary video,
        # side-view and top-down-view as extra follower panels.
        "configs": {
            "local_settings.yaml": (
                "primary_camera: front-view\n"
                "extra_cameras:\n"
                "- side-view\n"
                "- top-down-view\n"
                "files_aligned_to_trials: true\n"
                "labels_visible: true\n"
            ),
        },
    },
    "canary": {
        "name": "Giraudon et al. 2021 - Canary song",
        "image": "canary.png",
        "dataset_url": "https://zenodo.org/records/6521932",
        "folder": "Canary",
        "nc_filename": None,
        "has_audio": True,
        "audio_file": "100_marron1_May_24_2016_62101389.wav",
        "labels_file": "100_marron1_May_24_2016_62101389.audacity.txt",
        "release_tag": "canary",
        "size_mb": 2,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_gui_assets(key: str) -> list[str]:
    """Derive the GUI asset list from dataset metadata.

    Collects: nc_filename + unique media filenames + extra_gui_assets +
    standalone files (audio_file, labels_file).
    """
    ds = DATASETS[key]
    assets: list[str] = []
    seen: set[str] = set()

    def _add(name: str | None) -> None:
        if name and name not in seen:
            assets.append(name)
            seen.add(name)

    _add(ds.get("nc_filename"))
    for row in ds.get("media", []):
        # "trial" is a pairing key, not a media file.
        for col, fname in row.items():
            if col != "trial":
                _add(fname)
    for extra in ds.get("extra_gui_assets", []):
        _add(extra)
    _add(ds.get("audio_file"))
    _add(ds.get("labels_file"))
    return assets


def get_video_feature_assets(key: str) -> dict[str, list[str]]:
    """Map each optional video-feature folder of *key* to its ``{video stem}.npy`` files."""
    ds = DATASETS[key]
    stems = sorted(
        {Path(fname).stem for row in ds.get("media", []) for col, fname in row.items() if col.startswith("video_")}
    )
    return {folder: [f"{stem}.npy" for stem in stems] for folder in ds.get("video_feature_folders", {})}


def video_features_size_mb(key: str) -> int:
    """Total download size of *key*'s optional video-feature folders."""
    return sum(spec["size_mb"] for spec in DATASETS[key].get("video_feature_folders", {}).values())


def are_video_features_downloaded(key: str) -> bool:
    """Whether every optional video-feature file of *key* is present locally."""
    dest = dataset_dir(key)
    return all(
        (dest / folder / name).exists() for folder, names in get_video_feature_assets(key).items() for name in names
    )


def get_prediction_run_assets(key: str) -> dict[str, dict[str, str]]:
    """Map each shipped prediction run of *key* to its release assets.

    Keys are the run folders under the session's ``labels/``
    (``predictions_{suffix}``); each value maps a release asset name to the
    filename it is saved under. The release carries the model in the file
    name, the local copy in the folder name: ``{stem}_predictions.tsv`` +
    ``{stem}_probs.npz`` is exactly what a run writes and what
    :class:`~ethograph.labels.predictions.PredictionsStore` opens.
    """
    ds = DATASETS[key]
    suffixes = ds.get("prediction_runs", [])
    if not suffixes:
        return {}
    stem = Path(ds["nc_filename"]).stem
    return {
        f"{RUN_PREFIX}{suffix}": {
            f"{stem}_predictions_{suffix}.tsv": f"{stem}_predictions.tsv",
            f"{stem}_probs_{suffix}.npz": f"{stem}_probs.npz",
        }
        for suffix in suffixes
    }


def prediction_runs_dir(key: str) -> Path:
    """The ``labels/`` folder the shipped prediction runs of *key* land in."""
    return dataset_dir(key) / "labels"


def are_prediction_runs_downloaded(key: str) -> bool:
    """Whether every shipped prediction-run file of *key* is present locally."""
    root = prediction_runs_dir(key)
    return all(
        (root / folder / local).exists()
        for folder, files in get_prediction_run_assets(key).items()
        for local in files.values()
    )


def get_notebook_assets(key: str) -> list[str]:
    """Return notebook assets — explicit list if provided, else same as GUI."""
    return DATASETS[key].get("assets_notebook") or get_gui_assets(key)


def dataset_dir(key: str) -> Path:
    """Return the local download directory for a dataset."""
    return DOWNLOAD_BASE / DATASETS[key]["folder"]


def is_template_path(path: str | Path | None) -> bool:
    """Check whether *path* lives inside the downloaded template datasets tree."""
    if not path:
        return False
    try:
        Path(path).expanduser().resolve().relative_to(DOWNLOAD_BASE.resolve())
    except (ValueError, OSError):
        return False
    return True


def is_dataset_downloaded(key: str) -> bool:
    """Check whether all GUI assets for a dataset are present locally."""
    if key not in DATASETS:
        return False
    dest = dataset_dir(key)
    return all((dest / name).exists() for name in get_gui_assets(key))


def sample_data() -> "xr.Dataset":
    """Download and return one trial of real pose data (Moll et al., 2025).

    Downloads the Moll2025 crow tool-use dataset (~14 MB) on first call,
    then returns ``itrial(0)`` — an :class:`xarray.Dataset` with position,
    velocity, speed, and acceleration for 14 keypoint.

    Returns
    -------
    xarray.Dataset
        Single-trial dataset with dimensions ``(time, space, keypoint, individual)``.

    Examples
    --------
    >>> import ethograph as eto
    >>> ds = eto.sample_data()
    >>> ds["speed"]
    <xarray.DataArray 'speed' (time: ..., keypoint: 14, individual: 1)>
    """
    from ethograph.utils.download import download_assets

    key = "moll2025"
    info = DATASETS[key]
    dest = dataset_dir(key)
    nc_path = dest / info["nc_filename"]

    if not nc_path.exists():
        dest.mkdir(parents=True, exist_ok=True)
        download_assets(
            release_tag=info["release_tag"],
            assets=[info["nc_filename"]],
            dest=dest,
        )

    import ethograph as eto

    dt = eto.open(str(nc_path))
    return dt.itrial(0)


def resolve_dataset_paths(key: str) -> dict:
    """Resolve a dataset's metadata into absolute paths for the IO widget."""
    ds = DATASETS[key]
    dest = dataset_dir(key)
    nc_filename = ds.get("nc_filename")
    result = {
        "name": ds["name"],
        "dataset_key": key,
        "nc_file_path": str(dest / nc_filename) if nc_filename else "",
        "video_folder": str(dest),
        "audio_folder": str(dest) if ds.get("has_audio") else "",
        "pose_folder": str(dest),
        "import_labels": ds.get("import_labels", False),
        "downsample": ds.get("downsample"),
        "library_geometry": ds.get("library_geometry"),
    }
    if ds.get("labels_file"):
        result["labels_file"] = str(dest / ds["labels_file"])
    if ds.get("audio_file"):
        result["audio_file"] = str(dest / ds["audio_file"])
    return result
