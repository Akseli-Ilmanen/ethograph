"""Opening sessions headless — the same loaders the GUI uses, without Qt.

A session is opened through :func:`ethograph.io.data_loader.load_features_dataset`,
so every backend the GUI reads (xarray ``.nc``, pynapple folders/``.npz``,
NWB) is read here too, with the same catalog, alignment and labels. Trials
come from the alignment; the one trial filter is the metadata-table filter
in ``config.trials.where``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from ethograph.features.columns import FS_RTOL, sampling_rate
from ethograph.features.neural import transform_units
from ethograph.io import schema
from ethograph.io.catalog import INDIVIDUAL_DIMS, PynappleLoader, XarrayLoader, catalog_from_pynapple
from ethograph.io.data_loader import LoadResult, load_features_dataset
from ethograph.labels.intervals import LABELING_AUTOMATED
from ethograph.labels.tsv_store import get_trial_from_tsv
from ethograph.segment.config import (
    MERGED_CHANGEPOINTS,
    NeuralFeaturesConfig,
    SegmentConfig,
    SessionSpec,
    TrialsConfig,
)
from ethograph.utils.paths import session_id

logger = logging.getLogger(__name__)


@dataclass
class TrialWindow:
    """One trial as the loader sees it: ``select(..., t0, t1)`` on *loader*'s
    clock; subtracting *shift* makes times trial-relative."""

    trial: int | str
    loader: Any
    t0: float | None
    t1: float | None
    shift: float


@dataclass
class Session:
    spec: SessionSpec
    id: str
    result: LoadResult
    _sidecar: dict[str, dict[str, Any]] | None = field(default=None, repr=False)
    _video_devices: dict[str, str] = field(default_factory=dict, repr=False)
    #: The feature :func:`expand_neural_features` added, once it has.
    _neural_feature: str | None = field(default=None, repr=False)

    @property
    def source(self) -> Path:
        return self.spec.source

    @property
    def stem(self) -> str:
        p = self.source
        return p.name if p.is_dir() else p.stem

    @property
    def backend(self) -> str:
        return self.result.data_loader.backend

    @property
    def trial_ids(self) -> list[int | str]:
        return list(self.result.trial_ids)

    @property
    def individual_dim(self) -> str | None:
        """The dim spelling this dataset uses for individuals (``None`` = no individual dim)."""
        return self.result.catalog.individual_combo

    def individuals(self, config: SegmentConfig) -> list[str]:
        """The individuals that become samples of this session."""
        if config.features.individuals is not None:
            return [str(i) for i in config.features.individuals]
        dim = self.individual_dim
        if dim is not None:
            return [str(v) for v in self.result.catalog.combos[dim].values]
        actors = sorted(set(self.result.all_labels_df["individual"].dropna().astype(str)))
        if not actors:
            raise ValueError(
                f"{self.source}: the dataset has no individual dim and its labels name no "
                "individual — set config.individual (single-animal) or features.individuals in the config."
            )
        return actors

    def trial_windows(self, trials: list[int | str]) -> Iterator[TrialWindow]:
        base = self.result.data_loader
        if base.backend == "xarray":
            dt = self.result.dt
            for tid in trials:
                yield TrialWindow(tid, XarrayLoader(dt.trial(tid)), None, None, 0.0)
            return
        sc = self.result.source_collection
        fresh = PynappleLoader(base.data, base.catalog)
        for tid in trials:
            idx = sc.trial_index(tid)
            if idx is None:
                raise ValueError(f"{self.source}: trial {tid!r} has no time range in the alignment")
            rng = sc.trial_range(idx)
            yield TrialWindow(tid, fresh, rng.start_s, rng.end_s, rng.start_s)

    def curated_labels(self, trial: int | str) -> pd.DataFrame:
        """This trial's labels a model may learn from: ``manual`` and ``curated`` rows only."""
        df = get_trial_from_tsv(self.result.all_labels_df, trial)
        if df.empty:
            return df
        return df[df["labeling_method"] != LABELING_AUTOMATED].reset_index(drop=True)

    def trial_dataset(self, trial: int | str):
        """The trial's ``xr.Dataset`` (xarray sessions only, else ``None``)."""
        dt = self.result.dt
        return None if dt is None else dt.trial(trial)

    @property
    def sidecar(self) -> dict[str, dict[str, Any]]:
        """The session's ``.ethograph/schema.yaml``, read once (empty when absent)."""
        if self._sidecar is None:
            self._sidecar = schema.read_sidecar(self.source)
        return self._sidecar

    def variable_attrs(self, name: str, trial: int | str | None = None) -> dict[str, Any]:
        """The schema attrs of one variable, whatever the backend provides.

        An xarray variable carries its own; a pynapple one has nowhere to put
        them (a ``Tsd`` has no attrs at all), so they come from the session's
        sidecar. Where both exist the variable's own win — the file is the
        more specific statement.
        """
        ds = self.trial_dataset(trial if trial is not None else self.trial_ids[0])
        own = dict(ds[name].attrs) if ds is not None and name in ds.data_vars else {}
        return {**self.sidecar.get(name, {}), **own}

    def declares_schema(self) -> bool:
        """Whether anything in this session declares a ``kind``.

        ``False`` means the ablation axis and per-column normalisation have
        nothing to act on here — worth saying out loud rather than silently
        doing nothing.
        """
        if any(schema.KIND in attrs for attrs in self.sidecar.values()):
            return True
        ds = self.trial_dataset(self.trial_ids[0])
        return ds is not None and any(schema.kind_of(var) for var in ds.data_vars.values())

    def video_device(self, camera: str | None) -> str | None:
        """The alignment's own name for *camera* (``None`` = the default camera, passed through).

        The alignment's name when it has one so called; else the one camera
        whose video files carry the tag — an alignment written with its
        cameras numbered ``0``, ``1`` still points at ``…-cam-1.mp4`` — said
        once in the log. A camera nothing matches is an error naming what the
        alignment has, instead of every trial skipped one warning at a time.
        """
        if camera is None or camera in self._video_devices:
            return None if camera is None else self._video_devices[camera]
        alignment = self.result.nwb_alignment
        cameras = [str(c) for c in alignment.cameras]
        if camera in cameras:
            self._video_devices[camera] = camera
            return camera
        tagged = [
            c for c in cameras if any(camera in str(alignment.get_media(t, "video", c) or "") for t in self.trial_ids)
        ]
        if len(tagged) != 1:
            raise ValueError(
                f"{self.spec.label}: no camera {camera!r} in the alignment (it has {cameras}), "
                f"and {'no' if not tagged else 'several'} camera's files carry that name — "
                "set labels.camera to one of them"
            )
        logger.warning(
            "%s: the alignment names its cameras %s; %r taken as camera %r from its file names",
            self.spec.label,
            cameras,
            camera,
            tagged[0],
        )
        self._video_devices[camera] = tagged[0]
        return tagged[0]

    def media_path(self, trial: int | str, stream: str = "video", device: str | None = None) -> Path | None:
        """Resolve a media file for *trial*: the alignment's own path, else ``spec.video_dir``."""
        alignment = self.result.nwb_alignment
        found = alignment.resolve_media_path(trial, stream, device=device, fallback_folder=self.spec.video_dir)
        return Path(found) if found else None


def open_session(
    spec: SessionSpec, config: SegmentConfig | None = None, *, expand_changepoints: bool = True
) -> Session:
    """Open one session with the GUI's loaders (no Qt involved).

    *config* is only consulted for ``features.changepoint_features``; a
    pipeline with no feature engineering at all (``ethograph.spot``) passes
    ``None``. ``materialise`` opens with ``expand_changepoints=False``, reads
    the labels, and expands afterwards through
    :func:`expand_changepoint_features` — that is where a config whose
    scales are still to be derived gets them.
    """
    source = spec.source
    if not source.exists():
        raise FileNotFoundError(f"Session source does not exist: {source}")
    if spec.alignment is not None and not spec.alignment.is_file():
        raise FileNotFoundError(f"{spec.label}: alignment {spec.alignment} does not exist")
    labels_path = str(spec.labels_path) if spec.labels_path is not None else None
    alignment = str(spec.alignment) if spec.alignment is not None else None
    result = load_features_dataset(str(source), labels_path=labels_path, alignment_path=alignment)
    sid = session_id(source)
    logger.info("Opened session %s (%s backend, %d trials)", sid, result.data_loader.backend, len(result.trial_ids))
    session = Session(spec=spec, id=sid, result=result)
    expand_neural_features(session, config)
    if expand_changepoints:
        expand_changepoint_features(session, config)
    return session


def expand_neural_features(session: Session, config: SegmentConfig | None) -> None:
    """Apply ``config.features.neural`` to the session's spike trains, once.

    Runs the transform over the whole session (every trial reads its window
    off the one result), puts the ``TsdFrame`` into the session's pynapple
    objects under ``neural.name``, rebuilds the catalog and loader so it is
    selected like any other feature, and declares it ``neural_feature`` in
    the in-memory sidecar so the layout records its kind and normalises it.
    Nothing is written to disk. A session with an ``xr.Dataset`` backend has
    no spike trains to bin, and is refused.
    """
    cfg = None if config is None else config.features.neural
    if cfg is None or session._neural_feature == cfg.name:
        return
    data = session.result.pynapple_data
    if data is None:
        raise ValueError(
            f"{session.source}: features.neural is set but this session has no pynapple backend — "
            "spike trains are a TsGroup, which only a pynapple session (.npz / folder / NWB) carries."
        )
    import pynapple as nap

    if cfg.units not in data:
        groups = sorted(k for k, v in data.items() if isinstance(v, nap.TsGroup))
        raise ValueError(
            f"{session.source}: features.neural.units names {cfg.units!r}, which the session does not "
            f"have (its TsGroups: {groups or 'none'})"
        )
    units = data[cfg.units]
    if not isinstance(units, nap.TsGroup):
        raise ValueError(f"{session.source}: {cfg.units!r} is a {type(units).__name__}, not a TsGroup of spike trains")
    if cfg.name in data:
        raise ValueError(
            f"{session.source}: the session already has a variable called {cfg.name!r} — "
            "give features.neural.name a name the session does not use"
        )
    frame = transform_units(units, cfg.transform)
    logger.info(
        "%s: features.neural → %r, %d units at %.6g Hz (%s)",
        session.id,
        cfg.name,
        frame.shape[1],
        sampling_rate(np.asarray(frame.t, dtype=np.float64)),
        " | ".join(cfg.transform),
    )
    data[cfg.name] = frame
    catalog = catalog_from_pynapple(data, source_path=session.source)
    session.result.catalog = catalog
    session.result.data_loader = PynappleLoader(data, catalog)
    session._sidecar = {**session.sidecar, cfg.name: {schema.KIND: schema.NEURAL_FEATURE, schema.NORMALISE: 1}}
    session._neural_feature = cfg.name


def neural_columns(session: Session, cfg: NeuralFeaturesConfig) -> dict[str, list[str]]:
    """The ``features.columns`` entry the session's units resolve to: ``{dim: [unit ids]}``.

    Read off the loader, so the dim is spelled exactly as ``select()`` reads
    it (``{name}_columns`` for a lone frame). A session that has not been
    expanded, or whose transform left a single column with no dim to pin,
    is an error.
    """
    if session._neural_feature != cfg.name:
        raise ValueError(f"{session.source}: features.neural has not been applied to this session")
    dims = session.result.data_loader.feature_dims(cfg.name)
    if not dims:
        raise ValueError(f"{session.source}: the neural feature {cfg.name!r} has no column dim to pin")
    return {dim: [str(v) for v in values] for dim, values in dims.items()}


def expand_changepoint_features(session: Session, config: SegmentConfig | None) -> None:
    """Apply ``config.features.changepoint_features`` (if set) to every trial, once.

    Runs before anything reads the session, so ``trial_windows``,
    ``trial_dataset`` and ``variable_attrs`` all see the expanded columns
    consistently. A config whose scales are unresolved is resolved from the
    materialised dataset's ``columns.yaml`` (:func:`~ethograph.segment.materialise.resolved_config`),
    so every stage after ``materialise`` expands at the recorded scale; a
    session already expanded is left alone.
    """
    cfg = None if config is None else config.features.changepoint_features
    if cfg is None:
        return
    assert config is not None
    if session.result.dt is None:
        raise ValueError(
            f"{session.source}: features.changepoint_features is set but this session has no "
            "xr.Dataset backend (pynapple changepoints are event times, not a dense mask — "
            "changepoint expansion is only implemented for xarray sessions)."
        )
    if cfg.unresolved:
        from ethograph.segment.materialise import resolved_config  # circular: materialise imports sessions

        cfg = resolved_config(config).features.changepoint_features
        assert cfg is not None and not cfg.unresolved
    from ethograph.features.changepoints import add_changepoint_features, merge_changepoints

    generated = list(cfg.expanded_columns())

    def _expand(ds):
        if generated and all(name in ds.data_vars for name in generated):
            return ds
        # add_changepoint_features already recognises the legacy
        # `attrs["type"] = "changepoints"` spelling without migration (see
        # schema.kind_of/is_changepoint); migrate here anyway to normalise
        # the attrs and drop any other stale `type` value the file carries.
        ds = schema.migrate_legacy_attrs(ds)
        vars = list(cfg.inputs)
        if cfg.merge:
            # Keep the individual dim standing: one animal's changepoints must
            # not OR into another's, and the target feature still carries it.
            keep = [d for d in INDIVIDUAL_DIMS if d in ds.dims]
            ds, _target = merge_changepoints(ds, vars=vars, keep_dims=keep)
            vars = [MERGED_CHANGEPOINTS]
        return add_changepoint_features(
            ds,
            sigmas=cfg.sigmas,
            distribution=cfg.distribution,
            transforms=cfg.transforms,
            vars=vars,
            horizon=cfg.horizon,
            scale_by=cfg.scale_by,
            max_length=cfg.max_length,
        )

    session.result.dt = session.result.dt.map_trials(_expand)


def _probe_feature(loader: Any, window: TrialWindow, feature: str) -> tuple[dict[str, list[str]], float] | None:
    """*feature*'s dims plus its actual sampling rate, or ``None`` if ``select()``
    cannot resolve it to at least two time samples (pinning every dim to its
    first value; the rate does not depend on which values are pinned)."""
    dims = loader.feature_dims(feature) or {}
    probe = {d: values[0] for d, values in dims.items() if values}
    plot_data = loader.select(feature, probe, window.t0, window.t1)
    if plot_data is None:
        return None
    time = np.asarray(plot_data.time, dtype=np.float64)
    if time.size < 2:
        return None
    return dims, sampling_rate(time)


def feature_sampling_rates(session: Session, *, trial: int | str | None = None) -> dict[str, float]:
    """Every feature's own sampling rate — use this to find the *fs* to pass
    to :func:`discover_columns` when a session's rate is not a round number,
    or to see why a call to it came back empty."""
    tid = trial if trial is not None else session.trial_ids[0]
    window = next(session.trial_windows([tid]))
    loader = window.loader
    rates: dict[str, float] = {}
    for feature in loader.catalog.feature_choices():
        probed = _probe_feature(loader, window, feature)
        if probed is not None:
            rates[feature] = probed[1]
    return rates


def feature_sampling_rates_from_source(source: str | Path, *, trial: int | str | None = None) -> dict[str, float]:
    """:func:`feature_sampling_rates` from a raw session path — no config required."""
    source = Path(source)
    result = load_features_dataset(str(source))
    session = Session(spec=SessionSpec(source=source), id=session_id(source), result=result)
    return feature_sampling_rates(session, trial=trial)


def discover_columns(
    session: Session,
    fs: float,
    *,
    rtol: float = FS_RTOL,
    trial: int | str | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, dict[str, list[str]]]:
    """Every feature in *session* running at *fs* Hz, as a ``features.columns`` block.

    Probes one trial (the first, or *trial*) for each feature's own sampling
    rate (see :func:`feature_sampling_rates` to inspect these directly) and
    keeps every value of every non-individual dim (the individual dim is
    pinned per sample, so it is never part of a config) for the features whose
    rate matches *fs* within *rtol*. Paste the result into ``features.columns``
    and delete what you don't want — most sessions want "everything at this
    rate", not a hand-picked list. Returns ``{}`` with no error when nothing
    matches; check :func:`feature_sampling_rates` if that is unexpected.
    """
    tid = trial if trial is not None else session.trial_ids[0]
    window = next(session.trial_windows([tid]))
    loader = window.loader
    ind_dim = session.individual_dim
    skip = set(exclude)

    columns: dict[str, dict[str, list[str]]] = {}
    for feature in loader.catalog.feature_choices():
        if feature in skip:
            continue
        probed = _probe_feature(loader, window, feature)
        if probed is None:
            continue
        dims, actual_fs = probed
        if not np.isclose(actual_fs, fs, rtol=rtol):
            continue
        columns[feature] = {d: list(values) for d, values in dims.items() if d != ind_dim}
    return columns


def discover_columns_from_source(
    source: str | Path,
    fs: float,
    *,
    rtol: float = FS_RTOL,
    trial: int | str | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, dict[str, list[str]]]:
    """:func:`discover_columns` from a raw session path — no config required.

    *source* is anything :func:`~ethograph.io.data_loader.load_features_dataset`
    opens: a ``.nc`` file, an NWB file, or a pynapple folder/``.npz``.
    """
    source = Path(source)
    result = load_features_dataset(str(source))
    session = Session(spec=SessionSpec(source=source), id=session_id(source), result=result)
    return discover_columns(session, fs, rtol=rtol, trial=trial, exclude=exclude)


def filter_trials(session: Session, trials: TrialsConfig) -> list[int | str]:
    """Trials passing the metadata filter (all trials when the filter is empty),
    cut to ``trials.limit`` when one is set."""
    ids = _filter_by_columns(session, trials)
    if trials.limit is not None:
        if trials.limit < 1:
            raise ValueError(f"{session.source}: trials.limit must be >= 1, got {trials.limit}")
        ids = ids[: trials.limit]
    return ids


def _filter_by_columns(session: Session, trials: TrialsConfig) -> list[int | str]:
    ids = session.trial_ids
    if not trials.where:
        return ids
    df = session.result.metadata_df
    if df is None or df.empty or "trial" not in df.columns:
        raise ValueError(f"{session.source}: trials.where is set but the session has no metadata table")
    keep = pd.Series(True, index=df.index)
    for column, values in trials.where.items():
        if column not in df.columns:
            raise ValueError(
                f"{session.source}: trials.where names column {column!r} which the metadata table "
                f"does not have (columns: {list(df.columns)})"
            )
        allowed = {str(v) for v in values}
        keep &= df[column].astype(str).isin(allowed)
    allowed_ids = {str(t) for t in df.loc[keep, "trial"]}
    return [t for t in ids if str(t) in allowed_ids]


def individual_dim_name(loader: Any, feature: str) -> str | None:
    """Which individual-dim spelling *feature* carries, if any."""
    dims = loader.feature_dims(feature)
    return next((d for d in INDIVIDUAL_DIMS if d in dims), None)


def changepoint_times(session: Session, trial: int | str, selections: dict[str, str]) -> np.ndarray:
    """Changepoint times on the trial clock; empty when the session has none.

    Both backends mark changepoints the same way (see
    :mod:`ethograph.io.schema`) but store them differently: xarray as a
    per-frame binary mask, pynapple as a ``TsGroup`` whose units already
    *are* the event times.
    """
    ds = session.trial_dataset(trial)
    if ds is not None:
        return _changepoint_times_xarray(ds, selections)
    return _changepoint_times_pynapple(session, trial, selections)


def _changepoint_times_xarray(ds, selections: dict[str, str]) -> np.ndarray:
    from ethograph.features.changepoints import dataset_changepoint_times

    return dataset_changepoint_times(ds, selections=selections)


def _changepoint_times_pynapple(session: Session, trial: int | str, selections: dict[str, str]) -> np.ndarray:
    """Event times of the session's changepoint ``TsGroup``s, over this trial.

    *selections* narrows by the units' ``source_label`` (the column each
    changepoint was detected on) when any value matches one; a selection
    naming nothing here — the usual case, since it is written for keypoint
    dims — leaves every unit in.
    """
    import pynapple as nap

    data = session.result.pynapple_data or {}
    window = next(session.trial_windows([trial]))
    wanted = {str(v) for v in selections.values()}
    times: list[np.ndarray] = []
    for obj in data.values():
        if not isinstance(obj, nap.TsGroup):
            continue
        meta = getattr(obj, "metadata", None)
        units = schema.changepoint_units(meta)
        if not units:
            continue
        labels = meta["source_label"] if "source_label" in meta.columns else None
        matching = [u for u in units if str(labels[u]) in wanted] if labels is not None and wanted else []
        for uid in matching or units:
            ts = obj[uid]
            t = np.asarray(ts.t, dtype=np.float64)
            if window.t0 is not None and window.t1 is not None:
                t = t[(t >= window.t0) & (t <= window.t1)]
            times.append(t - window.shift)
    if not times:
        return np.array([], dtype=np.float64)
    return np.unique(np.concatenate(times))
