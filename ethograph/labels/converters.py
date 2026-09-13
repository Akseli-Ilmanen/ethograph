"""External format converters: crowsetta, NWB, pynapple, and other label I/O."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.labels.intervals import (
    EVENT_TYPE_STATE,
    _rows_to_df,
    load_label_mapping,
    save_label_mapping,
)
from ethograph.labels.tsv_store import (
    TRIAL_META_DEFAULTS,
    init_empty_labels,
    labels_tsv_path,
    load_labels_tsv,
)

logger = logging.getLogger(__name__)

CROWSETTA_SEQ_FORMATS = [
    "aud-seq",
    "simple-seq",
    "generic-seq",
    "notmat",
    "textgrid",
    "timit",
    "yarden",
]


# ---------------------------------------------------------------------------
# Base converter
# ---------------------------------------------------------------------------


class LabelConverter:
    """Base class for converting external label sources to ethograph intervals.

    Subclasses override :meth:`extract` to pull intervals from their source
    (NWB, pynapple, crowsetta, …).  The shared :meth:`resolve_labels` method
    centralises the "TSV on disk → extract from source → empty" fallback chain
    used by every ``LoadResult``-producing function in ``data_loader``.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._label_map: dict[str, int] = {}

    @property
    def label_map(self) -> dict[str, int]:
        return self._label_map

    def extract(self, trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return an all-labels DataFrame (with ``trial`` column).

        Parameters
        ----------
        trials_df
            Must contain ``trial``, ``start_time``, ``stop_time`` columns.
            Required for sources with global timestamps (NWB, pynapple).
        """
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------

    def _global_to_trial_rows(
        self,
        epochs: list[dict],
        trials_df: pd.DataFrame,
    ) -> list[dict]:
        """Convert global-time epochs to trial-relative interval rows."""
        has_trial_col = "trial" in trials_df.columns
        rows: list[dict] = []
        for idx, (_, trial_row) in enumerate(trials_df.iterrows()):
            t_start, t_stop = trial_row["start_time"], trial_row["stop_time"]
            trial_id = trial_row["trial"] if has_trial_col else idx
            for ep in epochs:
                if ep["offset_s"] <= t_start or ep["onset_s"] >= t_stop:
                    continue
                label_id = self._label_map.get(ep["label_name"], 0)
                if label_id == 0:
                    continue
                rows.append(
                    {
                        "onset_s": max(0.0, ep["onset_s"] - t_start),
                        "offset_s": min(t_stop - t_start, ep["offset_s"] - t_start),
                        "labels": label_id,
                        "individual": ep.get("individual", "individual_0"),
                        "trial": trial_id,
                    }
                )
        return rows

    def _rows_to_labels_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a TSV-compatible all-labels DataFrame from row dicts."""
        if not rows:
            return init_empty_labels([])
        df = pd.DataFrame(rows)
        for col, default in TRIAL_META_DEFAULTS.items():
            df[col] = default
        return df

    # -- resolve (TSV → extract → empty) ----------------------------------

    def resolve_labels(
        self,
        source_path: str | Path,
        trial_ids: list,
        trials_df: pd.DataFrame | None = None,
        labels_path: Path | None = None,
    ) -> pd.DataFrame:
        """Load labels with fallback: existing TSV → extract from source → empty.

        Parameters
        ----------
        source_path
            Primary data file; used to derive the default TSV path.
        trial_ids
            Trial identifiers (for the empty-labels fallback).
        trials_df
            Passed to :meth:`extract` for global→trial conversion.
        labels_path
            Override the TSV path (e.g. for NWB project directories).
        """
        tsv = labels_path if labels_path is not None else labels_tsv_path(Path(source_path))
        if tsv.exists():
            logger.info("Loaded labels from %s", tsv.name)
            return load_labels_tsv(tsv)
        df = self.extract(trials_df)
        if not df.empty:
            logger.info("Extracted %d label intervals via %s", len(df), self.name)
            return df
        return init_empty_labels(trial_ids)


def resolve_labels_tsv(
    source_path: str | Path,
    trial_ids: list,
    labels_path: Path | None = None,
) -> pd.DataFrame:
    """Load labels from TSV if it exists, otherwise return empty.

    Use this when there is no converter (e.g. xarray/nc files).
    """
    tsv = labels_path if labels_path is not None else labels_tsv_path(Path(source_path))
    if tsv.exists():
        logger.info("Loaded labels from %s", tsv.name)
        return load_labels_tsv(tsv)
    return init_empty_labels(trial_ids)


# ---------------------------------------------------------------------------
# Standalone helpers (kept for backwards compat / direct use)
# ---------------------------------------------------------------------------


def build_mapping_from_labels(string_labels: list[str]) -> dict[str, int]:
    """Build a name->id mapping from a list of unique string labels.

    Sorts labels alphabetically; 0 is reserved for 'background'.
    """
    unique = sorted(set(string_labels))
    mapping = {"background": 0}
    for i, name in enumerate(unique, start=1):
        mapping[name] = i
    return mapping


def crowsetta_to_intervals(
    file_path: str | Path,
    format_name: str,
    name_to_id: dict[str, int],
    individual: str = "ind0",
) -> pd.DataFrame:
    """Convert a crowsetta annotation file to an intervals DataFrame."""
    import crowsetta

    scribe = crowsetta.Transcriber(format=format_name)
    annot = scribe.from_file(file_path).to_annot()
    seq = annot.seq

    rows: list[dict] = []
    for segment in seq.segments:
        label_str = str(segment.label)
        label_id = name_to_id.get(label_str)
        if label_id is None or label_id == 0:
            continue
        rows.append(
            {
                "onset_s": float(segment.onset_s),
                "offset_s": float(segment.offset_s),
                "labels": label_id,
                "individual": individual,
            }
        )

    return _rows_to_df(rows)


def extract_crowsetta_labels(
    file_path: str | Path,
    format_name: str,
) -> list[str]:
    """Extract unique string labels from a crowsetta annotation file."""
    import crowsetta

    scribe = crowsetta.Transcriber(format=format_name)
    annot = scribe.from_file(file_path).to_annot()
    return list({str(seg.label) for seg in annot.seq.segments})


def write_mapping_file(
    output_path: str | Path,
    name_to_id: dict[str, int],
) -> None:
    """Write a mapping file in '<id> <name>' format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{idx} {name}" for name, idx in sorted(name_to_id.items(), key=lambda x: x[1])]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


#: Imported class names that mean "no behaviour" and are never added to a mapping.
BACKGROUND_NAMES = frozenset({"background", "sil"})


def extend_mapping(names: list[str], mapping_path: str | Path) -> tuple[dict[str, int], list[str]]:
    """Name -> id for *names* in the mapping at *mapping_path*, appending any it lacks.

    Existing classes keep their ids, branches and event types; a new name gets the
    next free id as a state class on branch 0. The file is rewritten only when a
    name was added, so one vocabulary serves every import format.

    Returns the name -> id lookup and the names that were added.
    """
    mapping_path = Path(mapping_path)
    mappings = (
        load_label_mapping(mapping_path)
        if mapping_path.exists()
        else {0: {"name": "background", "branch": 0, "event_type": EVENT_TYPE_STATE}}
    )
    name_to_id = {data["name"]: label_id for label_id, data in mappings.items() if isinstance(label_id, int)}
    next_id = max(name_to_id.values(), default=0) + 1
    added: list[str] = []
    for name in sorted(set(names)):
        if name in name_to_id or name.lower() in BACKGROUND_NAMES:
            continue
        mappings[next_id] = {"name": name, "branch": 0, "event_type": EVENT_TYPE_STATE}
        name_to_id[name] = next_id
        added.append(name)
        next_id += 1
    if added:
        save_label_mapping(mapping_path, mappings)
    return name_to_id, added


# ---------------------------------------------------------------------------
# Crowsetta converter
# ---------------------------------------------------------------------------


class CrowsettaLabelConverter(LabelConverter):
    """Convert crowsetta annotation files to ethograph intervals.

    Crowsetta labels are already in file-local time, so no trial table
    is needed for time conversion.  If a ``trials_df`` is provided the
    first trial id is attached; otherwise ``trial=1``.
    """

    name = "crowsetta"

    def __init__(
        self,
        file_path: str | Path,
        format_name: str,
        name_to_id: dict[str, int],
        individual: str = "ind0",
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self._format_name = format_name
        self._label_map = dict(name_to_id)
        self._individual = individual

    def extract(self, trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = crowsetta_to_intervals(
            self._file_path,
            self._format_name,
            self._label_map,
            self._individual,
        )
        if df.empty:
            return init_empty_labels([])
        trial_id = trials_df.iloc[0]["trial"] if trials_df is not None and len(trials_df) > 0 else 1
        df["trial"] = trial_id
        for col, default in TRIAL_META_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        return df

    @staticmethod
    def _collect_time_intervals(table, source: str, individual: str, out: list[dict]) -> None:
        df = table.to_dataframe()
        label_col = "label" if "label" in df.columns else None
        for _, row in df.iterrows():
            out.append(
                {
                    "onset_s": float(row["start_time"]),
                    "offset_s": float(row["stop_time"]),
                    "label_name": str(row[label_col]) if label_col else source.split("/")[-1],
                    "individual": individual,
                    "source": source,
                }
            )

    @staticmethod
    def _collect_interval_series(series, source: str, individual: str, out: list[dict]) -> None:
        data = np.asarray(series.data[:])
        timestamps = np.asarray(series.timestamps[:])
        label_name = getattr(series, "name", source.split("/")[-1])
        starts = np.where(data > 0)[0]
        for i in starts:
            j = next((k for k in range(i + 1, len(data)) if data[k] < 0), None)
            if j is not None:
                out.append(
                    {
                        "onset_s": float(timestamps[i]),
                        "offset_s": float(timestamps[j]),
                        "label_name": label_name,
                        "individual": individual,
                        "source": source,
                    }
                )


# ---------------------------------------------------------------------------
# Pynapple interval label converter
# ---------------------------------------------------------------------------


class PynappleLabelConverter(LabelConverter):
    """Extract labels from pynapple IntervalSet objects.

    Collects every ``nap.IntervalSet`` in the data dict except the one
    :func:`~ethograph.io.pynapple.detect_trials` reads as trial boundaries.
    Each IntervalSet name becomes a label class.
    """

    name = "pynapple_intervals"

    def __init__(self, data: dict, trials_ep=None) -> None:
        super().__init__()
        self._trials_df = trials_df_from_intervalset(trials_ep)
        self._epochs = self._extract_interval_epochs(data)
        if self._epochs:
            self._label_map = build_mapping_from_labels(sorted({e["label_name"] for e in self._epochs}))

    def _extract_interval_epochs(self, data: dict) -> list[dict]:
        from ethograph.io.pynapple import label_intervalsets

        epochs: list[dict] = []
        for key, obj in label_intervalsets(data).items():
            meta_cols = list(getattr(obj, "metadata_columns", []))
            starts = np.asarray(obj.start)
            ends = np.asarray(obj.end)

            if not meta_cols:
                epochs.extend(
                    {
                        "onset_s": float(s),
                        "offset_s": float(e),
                        "label_name": key,
                        "individual": "individual_0",
                    }
                    for s, e in zip(starts, ends)
                )
            else:
                meta = pd.DataFrame({c: np.asarray(obj[c]) for c in meta_cols})
                for group_key, group in meta.groupby(meta_cols):
                    label_name = "_".join(str(v) for v in group_key) if isinstance(group_key, tuple) else str(group_key)
                    idx = group.index.values
                    epochs.extend(
                        {
                            "onset_s": float(s),
                            "offset_s": float(e),
                            "label_name": label_name,
                            "individual": "individual_0",
                        }
                        for s, e in zip(starts[idx], ends[idx])
                    )
        return epochs

    def extract(self, trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
        t_df = trials_df if trials_df is not None else self._trials_df
        if not self._epochs or t_df.empty:
            return init_empty_labels([])
        rows = self._global_to_trial_rows(self._epochs, t_df)
        return self._rows_to_labels_df(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def trials_df_from_intervalset(trials_ep) -> pd.DataFrame:
    """Build a trials DataFrame from a pynapple IntervalSet."""
    if trials_ep is None or len(trials_ep) == 0:
        return pd.DataFrame(columns=["trial", "start_time", "stop_time"])
    return pd.DataFrame(
        {
            "trial": list(range(1, len(trials_ep) + 1)),
            "start_time": [float(s) for s in trials_ep.start],
            "stop_time": [float(e) for e in trials_ep.end],
        }
    )
