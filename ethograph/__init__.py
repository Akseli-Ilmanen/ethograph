from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ethograph")
except PackageNotFoundError:
    # package is not installed
    pass


from ethograph.datasets import sample_data
from ethograph.io.catalog import (
    DataCatalog,
    DataLoader,
    PlotData,
    PynappleLoader,
    XarrayLoader,
    catalog_from_pynapple,
    catalog_from_xarray,
)
from ethograph.io.dataset import (
    add_angle_rgb_to_ds,
    add_changepoints_to_ds,
    downsample_trialtree,
)
from ethograph.io.nwb_alignment import (
    NWBAlignment,
    align_media_from_streams,
    align_media_per_trial,
)
from ethograph.io.pairing import (
    SourceSpec,
    discover_media,
    pair_media,
)
from ethograph.io.pynapple import load_nap_data
from ethograph.io.time_model import (
    SourceCollection,
    TimeRange,
    TimeSource,
)
from ethograph.io.trialtree import TrialTree
from ethograph.utils.paths import ethograph_home, get_project_root
from ethograph.utils.xr_utils import get_ds_duration, get_time_coord, sel_valid


def __getattr__(name: str):
    """Expose the model pipelines lazily — they import torch, which the base install lacks."""
    if name == "segment":
        import ethograph.segment as segment

        return segment
    if name == "spot":
        import ethograph.spot as spot

        return spot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def open(path: str) -> TrialTree:
    """Load a TrialTree from a saved NetCDF file.

    Shorthand for :meth:`TrialTree.open <ethograph.io.trialtree.TrialTree.open>`.

    Parameters
    ----------
    path : str or Path
        Path to a ``.nc`` file previously saved with ``dt.save()``.

    Returns
    -------
    TrialTree

    Examples
    --------
    >>> import ethograph as eto
    >>> dt = eto.open("experiment.nc")
    >>> dt.trials
    [1, 2, 3]
    >>> ds = dt.itrial(0)
    """
    return TrialTree.open(path)


def from_datasets(datasets: list) -> TrialTree:
    """Build a TrialTree from a list of per-trial xarray Datasets.

    Shorthand for :meth:`TrialTree.from_datasets <ethograph.io.trialtree.TrialTree.from_datasets>`.

    Each dataset must have ``attrs["trial"]`` set to a unique trial
    identifier.

    Parameters
    ----------
    datasets : list[xarray.Dataset]
        One Dataset per trial.

    Returns
    -------
    TrialTree

    Examples
    --------
    >>> import xarray as xr, numpy as np, ethograph as eto
    >>> trials = []
    >>> for i in range(1, 4):
    ...     ds = xr.Dataset({"speed": xr.DataArray(np.random.rand(300), dims=["time"])})
    ...     ds.attrs["trial"] = i
    ...     trials.append(ds)
    >>> dt = eto.from_datasets(trials)
    >>> dt.trials
    [1, 2, 3]
    """
    return TrialTree.from_datasets(datasets)


def from_continuous(ds, epochs) -> TrialTree:
    """Build a TrialTree from a single continuous recording + trial epochs.

    Shorthand for :meth:`TrialTree.from_continuous`.

    Parameters
    ----------
    ds : xarray.Dataset
        Full recording dataset.
    epochs : pandas.DataFrame or pynapple.IntervalSet
        Trial boundaries.  DataFrame must have columns ``trial``,
        ``start_time``, ``stop_time``.

    Returns
    -------
    TrialTree
    """
    return TrialTree.from_continuous(ds, epochs)
