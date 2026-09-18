"""Figure 1: raw vs. post-processed predictions, against ground truth.

One cross-validation fold — three Freddy sessions train, ``TEST_SESSION`` is
held out — predicted once per entry in :data:`VARIANTS` from that single
trained run. Every variant is the same weights and the same probabilities;
only ``infer.postprocess`` differs, so what separates the rows is the
post-processing and nothing else.

Rows: curated ground truth, one per variant, and the model's per-frame
confidence. One PNG per labelled trial in ``{root}/plots``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ethograph as eto
import ethograph.segment as seg
from ethograph.features.changepoints import changepoint_mask_times
from ethograph.labels.intervals import LABELING_AUTOMATED, load_label_mapping
from ethograph.labels.plots import plot_label_segments
from ethograph.labels.predictions import PredictionsStore, prediction_to_labels_and_confidence
from ethograph.labels.tsv_store import load_labels_tsv
from ethograph.segment.sessions import open_session

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(r"C:\Users\aksel\Documents\Code\ethograph\configs\paper\data\fig1_c2f_tcn_crow1.yaml")
TEST_SESSION = (
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy"
    r"\ses-000_date-20250526_01\behav\Trial_data3.nc"
)
SEL_TRIAL = None # trial 33
VAL_FRACTION = 0.0  # default hyperparameters: best.pt is the last epoch
#: An already-trained fold to re-plot from, skipping training entirely.
#: ``None`` trains the fold now — its path is printed and goes here.
RUN_DIR: Path | None = None
# Path(
#     r"C:\Users\aksel\Documents\Code\ethograph\configs\paper\data\runs"
#     r"\cv_c2f_tcn_classic\fold-Trial_data3-12f7f748_20260831-1810"
# )
#: Row label → the ``infer.postprocess`` override that produces it, applied on
#: top of the config. Each step is spelled out rather than left to the file, so
#: editing ``changepoint_correction:`` in the YAML cannot collapse two rows into
#: the same prediction set.
VARIANTS = (
    ("model (raw)", "infer.postprocess={}"),
    ("model\n(purge & stitch)", "infer.postprocess.changepoint_correction=false"),
)
ROW_LABELS = ("ground truth", *(label for label, _ in VARIANTS), "confidence")
#: The pin every mask and trace below is read at. This script's own, not the
#: config's: keep it equal to ``infer.postprocess.changepoints`` or the lines
#: drawn are not the ones the snapping step used.
SELECTIONS = {"keypoint": "beakTip"}
#: Changepoint masks drawn as vertical lines, unioned. Empty draws none.
CHANGEPOINT_VARS = ("speed_troughs", "speed_turning_points")
#: Variable drawn as a curve above the label rows, at the same pin — so it is
#: the signal the changepoints were detected on. ``None`` leaves the row out.
TRACE_VAR: str | None = "speed"


def confidence_curve(npz_path: Path, trial: int | str) -> tuple[np.ndarray, np.ndarray] | None:
    """One trial's time axis and per-frame confidence from a run's ``_probs.npz``.

    ``None`` when the run predicted no such trial — ``trials.where`` filtered
    it out, or it carries no video — which is a trial to skip, not an error.
    """
    marker = f"_trial{trial}_"
    with np.load(npz_path) as npz:
        key = next((k for k in npz.files if marker in k and not k.endswith(("_time", "_boundary"))), None)
        if key is None:
            return None
        probs = np.asarray(npz[key], dtype=np.float64)
        time = np.asarray(npz[f"{key}_time"], dtype=np.float64)
    _, confidence = prediction_to_labels_and_confidence(probs)
    if confidence is None:
        raise ValueError(f"{npz_path}: {key!r} is not (T, C) probabilities")
    return time, confidence


def session_pin(session, config) -> dict[str, str]:
    """:data:`SELECTIONS` with this session's individual dim pinned to its first individual."""
    dim = session.individual_dim
    return SELECTIONS if dim is None else {**SELECTIONS, dim: session.individuals(config)[0]}


def trial_changepoints(session, trial: int | str, pin: dict[str, str]) -> np.ndarray:
    """The union of :data:`CHANGEPOINT_VARS`' firing times on *trial*'s clock."""
    ds = session.trial_dataset(trial)
    return np.unique(np.concatenate([changepoint_mask_times(ds[v], pin) for v in CHANGEPOINT_VARS]))


def trial_trace(session, trial: int | str, pin: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """*trial*'s :data:`TRACE_VAR` on its own clock, at the changepoints' pin."""
    da = session.trial_dataset(trial)[TRACE_VAR]
    values, _ = eto.sel_valid(da, pin)
    return np.asarray(eto.get_time_coord(da).values, dtype=np.float64), np.asarray(values, dtype=np.float64)


def trial_labels(df: pd.DataFrame, trial: int | str) -> pd.DataFrame:
    """*trial*'s rows, restricted to the classes this model predicts."""
    return df[(df["trial"] == trial) & (df["event_type"] == "state")]


def trained_fold(project: seg.Project) -> Path:
    """``RUN_DIR`` when it names a trained run, else the fold trained now."""
    if RUN_DIR is not None:
        return RUN_DIR
    project.materialise()
    folds = project.cross_validate(folds=[TEST_SESSION], val_fraction=VAL_FRACTION, predict=False)
    return Path(folds.iloc[0]["run_dir"])


def plot_trial(
    trial: int | str,
    rows: list[pd.DataFrame],
    time: np.ndarray,
    confidence: np.ndarray,
    mapping: dict,
    path: Path,
    changepoints: np.ndarray | None = None,
    trace: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Write one trial's comparison — the trace, a row per frame in *rows*, then confidence."""
    n = len(rows) + 1 + (trace is not None)
    fig, axs = plt.subplots(n, 1, figsize=(15, 1.3 * n), sharex=True)
    fig.suptitle(f"trial {trial}", fontsize=9)
    row_labels = list(ROW_LABELS)
    label_axes = list(axs)
    if trace is not None:
        axs[0].plot(*trace, color="tab:blue", lw=0.8)
        row_labels.insert(0, TRACE_VAR or "")
        label_axes = list(axs[1:])
    for ax, df in zip(label_axes[:-1], rows):
        plot_label_segments(ax, df, mapping)
    axs[-1].plot(time, confidence, color="black", lw=0.8)
    axs[-1].set_ylim(0, 1.05)

    if changepoints is not None and len(changepoints):
        inside = changepoints[(changepoints >= time[0]) & (changepoints <= time[-1])]
        for ax in axs:
            # Spans the axes vertically whatever each row's data limits are.
            ax.vlines(inside, 0, 1, transform=ax.get_xaxis_transform(), color="0.35", lw=0.4, alpha=0.7, zorder=3)

    for ax, row_label in zip(axs, row_labels):
        ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(time[0], time[-1])


    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)  # a figure per trial, never left open


def main() -> Path:
    project = seg.Project(CONFIG_PATH)
    run_dir = trained_fold(project)
    logger.info("Predicting %s from %s", TEST_SESSION, run_dir)

    # One trained run, one set of weights; infer.postprocess is the only difference.
    predicted: list[pd.DataFrame] = []
    npz_path: Path | None = None
    for label, override in VARIANTS:
        store = PredictionsStore(
            seg.Project(CONFIG_PATH, override).inference(run=run_dir, sessions=[TEST_SESSION])[0].parent
        )
        logger.info("%s -> %s", label.replace("\n", " "), store.folder.name)
        predicted.append(load_labels_tsv(store.tsv_path))
        # Post-processing never touches the probabilities, so any variant's
        # curve is every variant's curve.
        npz_path = npz_path or store.npz_path
    if npz_path is None:
        raise FileNotFoundError("No variant wrote a *_probs.npz to read the confidence from")

    config = project.config
    mapping = load_label_mapping(config.features.labels.mapping)
    # Drawn whether or not the variant snaps, so a boundary can be read against
    # the changepoint it would have been pulled onto.
    spec = config.select_sessions([TEST_SESSION])[0]
    session = open_session(spec, config) if CHANGEPOINT_VARS or TRACE_VAR else None
    pin = session_pin(session, config) if session is not None else {}
    ground_truth = load_labels_tsv(config.select_sessions([TEST_SESSION])[0].labels_path)
    curated = ground_truth[ground_truth["labeling_method"] != LABELING_AUTOMATED]

    out_dir = project.root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    made, skipped = 0, []
    for trial in sorted(ground_truth["trial"].unique()):
        if SEL_TRIAL is not None and trial != SEL_TRIAL:
            continue

        curve = confidence_curve(npz_path, trial)
        if curve is None:
            skipped.append(trial)
            continue
        time, confidence = curve
        rows = [trial_labels(curated, trial), *(trial_labels(df, trial) for df in predicted)]
        cps = trial_changepoints(session, trial, pin) if session is not None and CHANGEPOINT_VARS else None
        trace = trial_trace(session, trial, pin) if session is not None and TRACE_VAR else None
        plot_trial(
            trial,
            rows,
            time,
            confidence,
            mapping,
            out_dir / f"{CONFIG_PATH.stem}_trial{trial}.png",
            changepoints=cps,
            trace=trace,
        )
        made += 1

    logger.info("Wrote %d trials to %s", made, out_dir)
    if skipped:
        logger.warning("%d trials the run never predicted, skipped: %s", len(skipped), skipped)
    return out_dir


if __name__ == "__main__":
    print(main())
