"""Download example datasets from GitHub releases."""

import logging
import time
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ethograph.datasets import (
    DATASETS,
    DOWNLOAD_BASE,
    dataset_dir,
    get_gui_assets,
    get_notebook_assets,
    get_prediction_run_assets,
    get_video_feature_assets,
    is_dataset_downloaded,
    prediction_runs_dir,
)
from ethograph.utils.paths import BUNDLED_DEFAULTS_DIR

logger = logging.getLogger(__name__)

_RELEASE_BASE = "https://github.com/Akseli-Ilmanen/Ethograph/releases/download"

#: Network settings for release-asset fetches. GitHub redirects every asset to
#: ``release-assets.githubusercontent.com``, a four-IP anycast range that some
#: institutional firewalls block or throttle even though ``github.com`` itself
#: stays reachable — so a connect timeout on one asset says nothing about the
#: next. Without an explicit timeout the OS default (~21 s per address, four
#: addresses) turns that into a minutes-long freeze ending in WinError 10060.
_DOWNLOAD_TIMEOUT_S = 30.0
_DOWNLOAD_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0


def _fetch(url: str, attempts: int = _DOWNLOAD_ATTEMPTS) -> bytes:
    """Fetch *url*, retrying transient network failures with backoff.

    HTTP errors (a genuinely missing asset) are raised immediately — only
    connection-level failures are worth a second try.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as resp:  # noqa: S310
                return resp.read()
        except HTTPError as exc:
            raise HTTPError(url, exc.code, f"{exc.reason} ({url})", exc.headers, None) from exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            logger.warning("Download attempt %d/%d failed for %s: %s", attempt, attempts, url, exc)
            if attempt < attempts:
                time.sleep(_RETRY_BACKOFF_S * 2 ** (attempt - 1))
    raise ConnectionError(
        f"Could not reach {url} after {attempts} attempts ({last_error}). "
        "GitHub serves release assets from release-assets.githubusercontent.com "
        "(185.199.108-111.133); check whether a firewall, VPN or proxy is blocking that host."
    ) from last_error


#: Optional per-template local-settings asset. If a ``local_settings.yaml``
#: exists among a template's GitHub release assets, it is downloaded into the
#: dataset's ``.ethograph/`` folder — shipping the author's panel layout (and
#: any other per-dataset settings) through the normal settings system.
TEMPLATE_LOCAL_SETTINGS_FILENAME = "local_settings.yaml"

#: Optional per-template alignment asset. A dataset whose release ships an
#: authored ``alignment.nwb`` sets ``"download_alignment": True`` in
#: :data:`~ethograph.datasets.DATASETS` and gets the file fetched into
#: ``.ethograph/`` instead of rebuilt by :func:`build_alignment_nwb` — use it
#: whenever the real timing or stream offsets cannot be recovered by probing
#: the media.
TEMPLATE_ALIGNMENT_FILENAME = "alignment.nwb"

# The default label vocabulary ships as ethograph/defaults/mapping.txt and is
# seeded into ~/.ethograph/defaults/ with the other bundled defaults.
DEFAULT_MAPPING_PATH = BUNDLED_DEFAULTS_DIR / "mapping.txt"


_OPEN_TSV_VBS = (
    'Set objExcel = CreateObject("Excel.Application")\n'
    "objExcel.Visible = True\n"
    "objExcel.Workbooks.OpenText WScript.Arguments(0), , , 1, 1, False, True\n"
)


def _tsv_reg_content(vbs_path: Path) -> str:
    """Generate .reg file content pointing to *vbs_path*."""
    escaped = str(vbs_path).replace("\\", "\\\\")
    return (
        "Windows Registry Editor Version 5.00\n"
        "\n"
        "[HKEY_CLASSES_ROOT\\.tsv]\n"
        '@="TsvFile"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile]\n"
        '@="Tab-Separated Values"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile\\DefaultIcon]\n"
        '@="excel.exe,1"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile\\shell]\n"
        '@="openexcel"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile\\shell\\openexcel]\n"
        '@="Open TSV with Excel"\n'
        '"FriendlyAppName"="Excel (Tab-Separated)"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile\\shell\\openexcel\\command]\n"
        f'@="wscript.exe \\"{escaped}\\" \\"%1\\""\n'
    )


_SETUP_TSV_MAC = """\
#!/bin/bash
# Double-click this file to make .tsv files always open in Excel.

if ! command -v duti &> /dev/null; then
    echo "Installing duti (requires Homebrew)..."
    brew install duti
fi

if command -v duti &> /dev/null; then
    duti -s com.microsoft.Excel .tsv all
    echo "Done — .tsv files will now open with Excel."
else
    echo "Could not install duti."
    echo "Manual alternative: right-click any .tsv → Get Info → Open With → Microsoft Excel → Change All"
fi
"""


def _register_tsv_windows(vbs_path: Path) -> None:
    """Register .tsv → Excel file association via the current-user registry."""
    import winreg

    cls = r"Software\Classes"
    vbs_str = str(vbs_path)

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\.tsv") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "TsvFile")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Tab-Separated Values")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile\DefaultIcon") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "excel.exe,1")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile\shell") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "openexcel")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile\shell\openexcel") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Open TSV with Excel")
        winreg.SetValueEx(key, "FriendlyAppName", 0, winreg.REG_SZ, "Excel (Tab-Separated)")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile\shell\openexcel\command") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, f'wscript.exe "{vbs_str}" "%1"')


def _register_tsv_mac() -> None:
    """Set Excel as default app for .tsv on macOS via duti (if available)."""
    import subprocess

    try:
        subprocess.run(["duti", "-s", "com.microsoft.Excel", ".tsv", "all"], check=True)
    except FileNotFoundError:
        logger.debug("duti not installed — skipping .tsv association on macOS")


def ensure_default_configs() -> None:
    """Seed ``~/.ethograph/defaults/`` and the platform helpers if they don't exist yet.

    Moves a pre-layout home folder into the ``cache/`` + ``defaults/`` shape
    first (:func:`~ethograph.utils.paths.migrate_home_layout`), so the
    bundled defaults never shadow files a user already had.
    """
    import sys

    from ethograph.utils.paths import ethograph_home, migrate_home_layout, seed_defaults

    migrate_home_layout()
    global_dir = ethograph_home()
    global_dir.mkdir(parents=True, exist_ok=True)
    seed_defaults()

    if sys.platform == "win32":
        internal = global_dir / "_internal"
        internal.mkdir(parents=True, exist_ok=True)
        vbs = internal / "open-tsv.vbs"
        if not vbs.exists():
            vbs.write_text(_OPEN_TSV_VBS, encoding="utf-8")
        try:
            _register_tsv_windows(vbs)
        except OSError:
            logger.debug("Could not write TSV registry keys", exc_info=True)
    elif sys.platform == "darwin":
        _register_tsv_mac()


def write_example_configs(dataset_key: str, dest: Path) -> None:
    """Write bundled config files into ``dest/.ethograph/``.

    Existing files are kept — they hold user state (edited mappings, the
    auto-saved layout in local_settings.yaml) that re-selecting a template
    must not reset."""
    configs = DATASETS.get(dataset_key, {}).get("configs")
    if not configs:
        return
    from ethograph.utils.paths import SETTINGS_DIR

    config_dir = Path(dest) / SETTINGS_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, content in configs.items():
        path = config_dir / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def download_assets(
    release_tag: str,
    assets: list[str],
    dest: Path,
    on_progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    local_names: dict[str, str] | None = None,
) -> None:
    """Download asset files from a GitHub release to *dest*.

    Parameters
    ----------
    release_tag : str
        GitHub release tag (e.g. ``"moll2025"``).
    assets : list[str]
        Filenames to download.
    dest : Path
        Local directory to save files into (created if missing).
    on_progress : callable, optional
        ``(completed_count, current_filename)`` callback.
    cancelled : callable, optional
        Returns ``True`` to abort the download loop.
    local_names : dict, optional
        Release asset name → filename to save it under; an asset not listed
        keeps its name.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    local_names = local_names or {}

    for i, name in enumerate(assets):
        if cancelled and cancelled():
            return
        local_path = dest / local_names.get(name, name)
        if local_path.exists():
            if on_progress:
                on_progress(i + 1, name)
            continue
        url = f"{_RELEASE_BASE}/{release_tag}/{name}"
        if on_progress:
            on_progress(i, name)
        data = _fetch(url)
        # Write via .part so an interrupted write can never leave a truncated
        # file that the exists() check above would later treat as complete.
        part_path = local_path.with_name(local_path.name + ".part")
        part_path.write_bytes(data)
        part_path.replace(local_path)
        if on_progress:
            on_progress(i + 1, name)


def download_video_features(
    key: str,
    on_progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Download *key*'s optional video-feature folders into its dataset folder.

    Each folder comes from its own release and lands as
    ``{dataset}/{folder}/{video stem}.npy``, ready to import as a video feature.
    *on_progress* counts files across all folders.
    """
    folders = DATASETS[key].get("video_feature_folders", {})
    done = 0
    for folder, assets in get_video_feature_assets(key).items():
        offset = done

        def _progress(count: int, name: str, offset: int = offset, folder: str = folder) -> None:
            if on_progress:
                on_progress(offset + count, f"{folder}/{name}")

        download_assets(
            release_tag=folders[folder]["release_tag"],
            assets=assets,
            dest=dataset_dir(key) / folder,
            on_progress=_progress,
            cancelled=cancelled,
        )
        done += len(assets)


def download_prediction_runs(
    key: str,
    on_progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Download *key*'s shipped prediction runs into its ``labels/`` folder.

    Each run lands as ``{dataset}/labels/predictions_{suffix}/`` holding the
    ``*_predictions.tsv`` + ``*_probs.npz`` pair, so File → Import
    predictions → From folder opens it like any other run and several can be
    stacked against each other. *on_progress* counts files across all runs.
    """
    release_tag = DATASETS[key]["release_tag"]
    done = 0
    for folder, files in get_prediction_run_assets(key).items():
        offset = done

        def _progress(count: int, name: str, offset: int = offset, folder: str = folder) -> None:
            if on_progress:
                on_progress(offset + count, f"{folder}/{name}")

        download_assets(
            release_tag=release_tag,
            assets=list(files),
            dest=prediction_runs_dir(key) / folder,
            on_progress=_progress,
            cancelled=cancelled,
            local_names=files,
        )
        done += len(files)


def download_template_local_settings(key: str) -> Path | None:
    """Fetch the optional ``local_settings.yaml`` release asset for *key* into
    the dataset's ``.ethograph/`` folder.

    Never overwrites an existing local file (the user's own settings win).
    Returns the local path, or ``None`` when the release has no such asset.
    Template authors ship a layout by uploading their dataset's
    ``.ethograph/local_settings.yaml`` to the GitHub release.
    """
    from ethograph.utils.paths import SETTINGS_DIR

    local_path = dataset_dir(key) / SETTINGS_DIR / TEMPLATE_LOCAL_SETTINGS_FILENAME
    if local_path.exists():
        return local_path
    url = f"{_RELEASE_BASE}/{DATASETS[key]['release_tag']}/{TEMPLATE_LOCAL_SETTINGS_FILENAME}"
    try:
        # Optional asset — most releases 404 here, so don't spend retries on it.
        data = _fetch(url, attempts=1)
    except (URLError, OSError):
        return None
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    return local_path


def download_template_alignment(key: str) -> Path:
    """Fetch the ``alignment.nwb`` release asset for *key* into ``.ethograph/``.

    Only for datasets declaring ``"download_alignment": True``. Unlike
    :func:`download_template_local_settings` the asset is *required*, so a
    missing one raises rather than returning ``None`` — falling back to
    :func:`build_alignment_nwb` would silently hand the user a differently
    timed dataset while appearing to succeed.

    An existing local file is kept: the GUI edits stream offsets straight into
    this file, so re-selecting the template must not discard that.
    """
    from ethograph.utils.paths import SETTINGS_DIR

    local_path = dataset_dir(key) / SETTINGS_DIR / TEMPLATE_ALIGNMENT_FILENAME
    if local_path.exists():
        return local_path
    url = f"{_RELEASE_BASE}/{DATASETS[key]['release_tag']}/{TEMPLATE_ALIGNMENT_FILENAME}"
    data = _fetch(url)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = local_path.with_name(local_path.name + ".part")
    part_path.write_bytes(data)
    part_path.replace(local_path)
    logger.info("Downloaded alignment NWB: %s", local_path)
    return local_path


def ensure_alignment_nwb(key: str) -> None:
    """Make sure the dataset's ``.ethograph/alignment.nwb`` exists.

    Downloaded when the dataset ships one (``"download_alignment"``),
    constructed from its media mapping otherwise.
    """
    if DATASETS[key].get("download_alignment"):
        download_template_alignment(key)
        return
    build_alignment_nwb(key)


def _pair_media_with_trials(trials, media_rows: list[dict], key: str) -> list[tuple]:
    """Pair each trial id in *trials* with its row from *media_rows*.

    A row carrying an explicit ``"trial"`` is matched **by id**; that is the
    only safe form when the metadata is not written in the same order as
    ``dt.trials`` (which comes back numerically sorted, so date-ordered media
    lists cross over). Rows without one are taken positionally, which is why
    the row count then has to match exactly.
    """
    if all("trial" in row for row in media_rows):
        by_trial = {str(row["trial"]): row for row in media_rows}
        missing = [t for t in trials if str(t) not in by_trial]
        if missing:
            raise KeyError(f"{key}: no media row for trial(s) {missing}")
        return [(t, {k: v for k, v in by_trial[str(t)].items() if k != "trial"}) for t in trials]

    if len(media_rows) != len(trials):
        raise ValueError(
            f"{key}: {len(media_rows)} media rows for {len(trials)} trials. "
            "Add a 'trial' key to each media row so they can be matched by id."
        )
    return list(zip(trials, media_rows))


def build_alignment_nwb(key: str) -> None:
    """Create an alignment.nwb from the dataset's media mapping and NC file.

    Prefer :func:`ensure_alignment_nwb`, which honours a dataset's shipped
    alignment asset instead of rebuilding it.
    """
    dataset = DATASETS[key]
    media_rows = dataset.get("media")
    if not media_rows:
        return

    dest = dataset_dir(key)
    nc_filename = dataset.get("nc_filename")
    if not nc_filename:
        return
    nc_path = dest / nc_filename
    if not nc_path.exists():
        return

    import pandas as pd

    import ethograph as eto
    from ethograph.io.pairing import pair_media
    from ethograph.utils.stream_durations import (
        get_audio_duration,
        get_video_duration,
    )

    dt = eto.open(str(nc_path))
    trials = dt.trials
    fps = dt.itrial(0).fps

    rows = []
    for trial_id, media in _pair_media_with_trials(trials, media_rows, key):
        if "video_cam-1" in media:
            video_path = dest / media["video_cam-1"]
            stop_time = get_video_duration(str(video_path))
        elif "audio_mic-1" in media:
            audio_path = dest / media["audio_mic-1"]
            stop_time = get_audio_duration(str(audio_path))
        elif key == "canary":
            audio_path = dest / media["audio_file"]
            stop_time = get_audio_duration(str(audio_path))
        elif key == "lockbox":
            stop_time = dt.trial(trial_id).time.values[-1]

        row = {"trial": trial_id, "start_time": 0, "stop_time": stop_time}
        row.update(media)
        rows.append(row)

    trial_table = pd.DataFrame(rows)
    nwb_path = dest / ".ethograph" / "alignment.nwb"
    stream_rates: dict[str, float] = {}
    if fps is not None:
        stream_rates["video"] = float(fps)
        stream_rates["pose"] = float(fps)

    audio_cols = [c for c in trial_table.columns if c.startswith("audio_")]
    if audio_cols:
        first_audio_file = dest / str(trial_table[audio_cols[0]].iloc[0])
        if first_audio_file.exists():
            from ethograph.utils.audio import get_audio_sr

            sr = get_audio_sr(str(first_audio_file))
            if sr is not None:
                stream_rates["audio"] = float(sr)

    pair_media(trial_table, stream_rates=stream_rates, output_path=nwb_path)
    logger.info("Created alignment NWB: %s", nwb_path)


def download_example_dataset(
    key: str,
    dest: Path,
    verbose: bool = True,
) -> Path | None:
    """High-level helper: download an example dataset by key.

    Downloads assets and writes bundled configs (e.g. ``mapping.txt``)
    into ``dest/.ethograph/``.

    Parameters
    ----------
    key : str
        One of the keys in ``DATASETS``.
    dest : Path
        Directory to download into.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    Path to ``mapping.txt`` if one was created, otherwise ``None``.
    """
    info = DATASETS[key]
    assets = get_notebook_assets(key)

    def _print_progress(count: int, name: str) -> None:
        total = len(assets)
        if count < total:
            print(f"Downloading {name}... ({count}/{total})")
        else:
            print(f"  {name} ({count}/{total})")

    download_assets(
        release_tag=info["release_tag"],
        assets=assets,
        dest=dest,
        on_progress=_print_progress if verbose else None,
    )

    ensure_default_configs()
    write_example_configs(key, dest)
    from ethograph.utils.paths import SETTINGS_DIR

    mapping_path = Path(dest) / SETTINGS_DIR / "mapping.txt"
    if mapping_path.exists():
        if verbose:
            print(f"  mapping: {mapping_path}")
        return mapping_path
    return None


def ensure_template_dataset(key: str, verbose: bool = True) -> Path:
    """Download template *key* as the GUI's template dialog would, if missing.

    Fetches the GUI assets, the shipped prediction runs, the bundled configs
    and the alignment NWB into :func:`~ethograph.datasets.dataset_dir`.
    Files already present are kept, so calling it again is cheap.

    Contributors run it over every key in ``DATASETS`` once so the integration
    tests exercise every template instead of skipping it::

        python -c "from ethograph.datasets import DATASETS; \\
                   from ethograph.utils.download import ensure_template_dataset; \\
                   [ensure_template_dataset(k) for k in DATASETS]"
    """
    dest = dataset_dir(key)
    assets = get_gui_assets(key)

    def _print_progress(count: int, name: str) -> None:
        if count == len(assets):
            print(f"  {key}: {name} ({count}/{len(assets)})")

    download_assets(
        release_tag=DATASETS[key]["release_tag"],
        assets=assets,
        dest=dest,
        on_progress=_print_progress if verbose else None,
    )
    download_prediction_runs(key)
    ensure_default_configs()
    write_example_configs(key, dest)
    if not (dest / ".ethograph" / "alignment.nwb").exists():
        ensure_alignment_nwb(key)
    return dest


def setup_birdpark_continuous(
    dest: Path | None = None,
    n_trials: int = 3,
    chunk: float = 20.0,
    verbose: bool = True,
) -> Path:
    """Create a BirdPark Continuous variant with a multi-trial alignment NWB.

    Symlinks (or copies on Windows) the BirdPark assets into *dest* and
    creates a ``.ethograph/alignment.nwb`` that treats the single 60 s
    recording as session-wide continuous media split into *n_trials* trials.

    Parameters
    ----------
    dest
        Output directory. Defaults to ``~/.ethograph/cache/example_data/BirdParkContinuous``.
    n_trials
        Number of equal-length trials to split the recording into.
    chunk
        Duration of each trial in seconds.
    verbose
        Print progress to stdout.

    Returns
    -------
    Path to the created directory.
    """
    import shutil

    import numpy as np
    import pandas as pd
    import pynwb
    from pynwb import NWBHDF5IO
    from pynwb.image import ImageSeries

    if dest is None:
        dest = DOWNLOAD_BASE / "BirdParkContinuous"

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    bp_src = dataset_dir("birdpark")

    for asset in get_gui_assets("birdpark"):
        src = bp_src / asset
        dst = dest / asset
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(
                f"BirdPark asset not found: {src}. "
                "Download birdpark first via download_example_dataset('birdpark', ...)"
            )
        shutil.copy2(src, dst)
        if verbose:
            print(f"  copied {asset}")

    # Also copy the .nc
    nc_name = "copExpBP08_trim.nc"
    nc_dst = dest / nc_name
    if not nc_dst.exists():
        shutil.copy2(bp_src / nc_name, nc_dst)

    video_name = "BP_2021-05-25_08-12-51_655154_0380000.mp4"
    audio_name = "BP_2021-05-25_08-12-51_655154_0380000.wav"
    fps = 47.68

    epochs = pd.DataFrame(
        {
            "trial": list(range(1, n_trials + 1)),
            "start_time": [i * chunk for i in range(n_trials)],
            "stop_time": [(i + 1) * chunk for i in range(n_trials)],
        }
    )

    from datetime import datetime
    from uuid import uuid4

    from dateutil.tz import tzlocal

    nwbfile = pynwb.NWBFile(
        session_description="BirdPark continuous — session-wide media alignment.",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    nwbfile.add_trial_column(name="trial", description="Trial number")
    nwbfile.add_trial_column(name="video_cam-1", description="video filename")
    nwbfile.add_trial_column(name="audio_mic-1", description="audio filename")
    for _, row in epochs.iterrows():
        nwbfile.add_trial(
            start_time=float(row["start_time"]),
            stop_time=float(row["stop_time"]),
            trial=row["trial"],
            **{"video_cam-1": video_name, "audio_mic-1": audio_name},
        )

    session_end = float(epochs["stop_time"].max())
    n_video_frames = int(session_end * fps)
    video_ts = np.arange(n_video_frames) / fps

    nwbfile.create_device(name="cam-1", description="video device cam-1")
    nwbfile.add_acquisition(
        ImageSeries(
            name="video_cam-1",
            description="video from cam-1",
            external_file=[video_name],
            format="external",
            starting_frame=np.array([0], dtype=np.int32),
            timestamps=video_ts,
        )
    )

    nwb_path = dest / ".ethograph" / "alignment.nwb"
    nwb_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    ensure_default_configs()

    if verbose:
        print(f"BirdPark Continuous ready at {dest}")
        print(f"  {n_trials} trials x {chunk}s, alignment: {nwb_path}")

    return dest


def setup_moll2025_pynapple(
    dest: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Create a Moll2025 Pynapple variant with alignment NWB linking to original media.

    Copies pynapple ``.npz`` files and labels TSV, writes configs
    (``mapping.txt``), and creates
    ``.ethograph/alignment.nwb`` whose trials table references the video
    and pose files in the original ``Moll2025`` folder.

    Parameters
    ----------
    dest
        Output directory.  Defaults to
        ``~/.ethograph/cache/example_data/Moll2025_pynapple``.
    verbose
        Print progress to stdout.

    Returns
    -------
    Path to the created directory.
    """
    import shutil

    import numpy as np
    import pynapple as nap
    import pynwb
    from pynwb import NWBHDF5IO
    from pynwb.image import ImageSeries

    if dest is None:
        dest = DOWNLOAD_BASE / "Moll2025_pynapple"

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    moll_src = dataset_dir("moll2025")
    if not moll_src.exists():
        raise FileNotFoundError(
            f"Moll2025 source not found: {moll_src}. "
            "Download moll2025 first via download_example_dataset('moll2025', ...)"
        )

    # Copy pynapple assets + extras
    pynapple_assets = DATASETS["moll2025"].get("assets_pynapple", [])
    extra_assets = ["beakTip_angle_rgb.npz", "beakTip_speed_troughs.npz"]
    labels_tsv = "Trial_data_labels.tsv"

    for asset in pynapple_assets + extra_assets:
        src = moll_src / asset
        dst_file = dest / asset
        if dst_file.exists() or not src.exists():
            continue
        shutil.copy2(src, dst_file)
        if verbose:
            print(f"  copied {asset}")

    labels_src = moll_src / labels_tsv
    labels_dst = dest / labels_tsv
    if labels_src.exists() and not labels_dst.exists():
        shutil.copy2(labels_src, labels_dst)
        if verbose:
            print(f"  copied {labels_tsv}")

    # Write configs
    write_example_configs("moll2025", dest)

    # Read trial epochs from pynapple trials.npz
    trials_npz = dest / "trials.npz"
    trials_obj = nap.load_file(str(trials_npz))
    if isinstance(trials_obj, nap.IntervalSet):
        trials_ep = trials_obj
    elif isinstance(trials_obj, dict):
        trials_ep = next(
            (v for v in trials_obj.values() if isinstance(v, nap.IntervalSet)),
            None,
        )
    else:
        trials_ep = None
    if trials_ep is None:
        raise ValueError(f"No IntervalSet found in {trials_npz}")

    # Moll2025: trial 1 = recording 115, trial 2 = recording 41
    media_per_trial = DATASETS["moll2025"]["media"]
    fps = 200.0

    from datetime import datetime
    from uuid import uuid4

    from dateutil.tz import tzlocal

    nwbfile = pynwb.NWBFile(
        session_description="Moll2025 pynapple — alignment to original media.",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    nwbfile.add_trial_column(name="trial", description="Trial number")
    nwbfile.add_trial_column(name="video_cam-1", description="video filename")
    nwbfile.add_trial_column(name="pose_cam-1", description="pose filename")
    for i in range(len(trials_ep)):
        trial_id = i + 1
        media = media_per_trial[i] if i < len(media_per_trial) else {}
        nwbfile.add_trial(
            start_time=float(trials_ep.start[i]),
            stop_time=float(trials_ep.end[i]),
            trial=trial_id,
            **{
                "video_cam-1": media.get("video_cam-1", ""),
                "pose_cam-1": media.get("pose_cam-1", ""),
            },
        )

    # Per-trial ImageSeries — each trial has its own video file
    nwbfile.create_device(name="cam-1", description="video device cam-1")
    video_files = [media_per_trial[i]["video_cam-1"] for i in range(len(trials_ep))]
    n_frames_per_trial = [
        int((float(trials_ep.end[i]) - float(trials_ep.start[i])) * fps) for i in range(len(trials_ep))
    ]
    timestamps_parts = []
    starting_frames = []
    frame_count = 0
    for i in range(len(trials_ep)):
        t0 = float(trials_ep.start[i])
        n = n_frames_per_trial[i]
        ts = t0 + np.arange(n) / fps
        timestamps_parts.append(ts)
        starting_frames.append(frame_count)
        frame_count += n

    nwbfile.add_acquisition(
        ImageSeries(
            name="video_cam-1",
            description="video from cam-1",
            external_file=video_files,
            format="external",
            starting_frame=np.array(starting_frames, dtype=np.int32),
            timestamps=np.concatenate(timestamps_parts),
        )
    )

    nwb_path = dest / ".ethograph" / "alignment.nwb"
    nwb_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    if verbose:
        print(f"Moll2025 Pynapple ready at {dest}")
        print(f"  {len(trials_ep)} trials, alignment: {nwb_path}")
        print(f"  video/pose from: {moll_src}")

    return dest


# ---------------------------------------------------------------------------
# Backward-compat re-exports (used by external code / notebooks)
# ---------------------------------------------------------------------------

EXAMPLE_DATASETS = DATASETS
EXAMPLE_CONFIGS = {k: v["configs"] for k, v in DATASETS.items() if "configs" in v}


def is_downloaded(key: str, dest: Path) -> bool:  # noqa: ARG001
    """Deprecated — use ``is_dataset_downloaded(key)`` instead."""
    return is_dataset_downloaded(key)
