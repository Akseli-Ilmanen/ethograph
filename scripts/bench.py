"""Which loss terms and which feature groups earn their place — per individual, per architecture.

Two axes, crossed where it is worth the GPU. The **objective** is a sum of up
to three terms (``docs/source/models/segment/config.md``, *Losses*): the
frame cross-entropy, the consistency (smoothing) term it carries at
``train.loss.alpha``, and the circle metric-learning term at
``train.circle.weight``. The **inputs** fall into three declared kinds
(``ethograph/io/schema.py``): the pose-derived columns
(``kinematic_feature``), the S3D video columns (``video_feature``), and the
columns ``features.changepoint_features`` expands out of the raw changepoint
masks (``changepoint_feature``).

An arm is one point of :data:`LOSS_TERMS` × :data:`FEATURE_SETS`, named
``{loss}_{features}``:

=====================  ==================================================
``all``                every term, every column — the reference
``no_smooth``          ``train.loss.alpha = 0`` — no consistency term
``no_circle``          ``train.circle.weight = 0`` — no circle term
``all_no_cp``          every term, no changepoint columns
``all_no_kin``         every term, no pose columns (S3D + changepoints only)
``all_no_s3d``         every term, no video columns
``no_smooth_no_cp``    no consistency term, no changepoint columns
``no_smooth_no_kin``   no consistency term, no pose columns
``no_smooth_no_s3d``   no consistency term, no video columns
=====================  ==================================================

The circle term is crossed with nothing: :data:`LOSS_TERMS` asks whether it
earns its place at all, and :data:`FEATURE_SETS` is crossed with the two
objectives worth ablating features under.

Every knob is pinned in every arm rather than read from the project config:
the "with" values are :data:`SMOOTHING_ALPHA` and :data:`CIRCLE_WEIGHT`, so
what an arm trained with is in this file and in the run's ``config.yaml``,
nowhere else. Dropping a feature group is ``train.drop_kinds`` — the
run-level ablation axis — never a second column list, so one materialised
dataset per individual serves every arm. That only works if the session
declares its kinds: run ``python scripts/describe_sessions.py`` once (it
writes each session's ``.ethograph/schema.yaml``), or
:func:`check_kinds_declared` refuses to train an arm that would silently drop
nothing.

**One model per individual.** Each entry of :data:`INDIVIDUALS` is a config
beside the project's — ``data/crow1.yaml`` inherits ``project.yaml`` through
``base:`` and lists only that individual's sessions under its own
``features.name`` — and every cell is ``Project.cross_validate()`` on it:
leave-one-session-out, so a cell's score is the mean over sessions the
model never saw, with one dot per session in every figure. The stem of the
config is how the individual is named in every output (``crow 1``); nothing
here knows anything else about it.

**Resumable, fold by fold.** A cross-validation's folds are ordinary runs
under ``runs/cv_{run_name}/fold-{session}_{timestamp}/``, and each writes
its ``test_metrics.yaml`` + ``test_eval.npz`` as it finishes. A cell whose
sessions all have such a fold is read back, never retrained; a cell with some
missing holds out only those, through ``cross_validate(folds=...)``; a fold
that evaluated but crashed before writing its prediction set gets that set by
inference alone. Every cell is sequential — they all want the GPU.

The output is ``data/bench_loss.pdf`` (:func:`ethograph.segment.plotting.write_factorial_pdf`):
the summary grids first — segmental F1 at every threshold and the frame-level
scores, one row per individual plus all of them pooled, architectures along
x and a bar per arm — then one page per individual × architecture with the
arms' IoU distributions, boundary deltas and class-wise F1 side by side.
``data/bench_loss.tsv`` holds every fold's numbers.

    python scripts/bench.py                 # train what is missing, then draw
    python scripts/bench.py --report-only   # draw from what has finished
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

import ethograph as eto
from ethograph.io.schema import CHANGEPOINT_FEATURE, KINEMATIC_FEATURE, VIDEO_FEATURE
from ethograph.labels.onset_model import session_id
from ethograph.segment.crossval import cross_validation_name_for
from ethograph.segment.inference import PREDICTIONS_PREFIX, prediction_run_dir
from ethograph.segment.materialise import COLUMNS_FILE, read_layout
from ethograph.segment.metrics import EVAL_ARRAYS_FILE, TEST_METRICS_FILE
from ethograph.segment.plotting import FactorCell, load_run_eval, write_factorial_pdf
from ethograph.segment.samples import ClassTable

#: Where the configs live. Set BENCH_CONFIG_DIR to point at another machine's copy.
CONFIG_DIR = Path(os.environ.get("BENCH_CONFIG_DIR") or Path(__file__).resolve().parents[1] / "data")

#: One config per individual, ``{stem}.yaml`` in :data:`CONFIG_DIR`; the stem
#: names the individual everywhere (``crow1`` → ``crow 1``).
INDIVIDUALS = ["crow1", "crow2", "crow3"]

#: Compared at their upstream defaults (``model.params: {}``): the bench asks
#: about the objective and the inputs, not the hyperparameters — those are
#: ``bench_search.py``'s.
ARCHITECTURES = ["mlp", "c2f_tcn", "c2f_transformer", "mstcn"]

#: ``train.loss.alpha`` of the arms that keep the smoothing term: DLC2Action's
#: own YAML default, so ``all`` is the objective the project config trains
#: with and every other arm is read against it. MS-TCN's published λ is
#: ``0.15`` (what the archived CETNet script used, ``0.15 * mse_loss``), two
#: orders of magnitude up — at ``0.001`` the term is a light touch, so expect
#: ``all`` and ``no_smooth`` to sit close and read the gap accordingly.
SMOOTHING_ALPHA = 0.001

#: ``train.circle.weight`` of the arms that keep the circle term — the same
#: script's ``0.001 * CircleLoss(m=0.25, gamma=128)``; ``m`` and ``gamma`` stay
#: at those defaults.
CIRCLE_WEIGHT = 0.001

#: The objective axis: what each arm's loss is made of.
LOSS_TERMS: dict[str, dict[str, Any]] = {
    "all": {"train.loss.alpha": SMOOTHING_ALPHA, "train.circle.weight": CIRCLE_WEIGHT},
    "no_smooth": {"train.loss.alpha": 0.0, "train.circle.weight": CIRCLE_WEIGHT},
    "no_circle": {"train.loss.alpha": SMOOTHING_ALPHA, "train.circle.weight": 0.0},
}

#: The input axis: which declared kind each arm withholds. ``""`` keeps every
#: column and is the half of a name that is left unwritten (``all``, not
#: ``all_full``). ``scripts/describe_sessions.py`` is what puts these kinds on
#: the sessions in the first place.
FEATURE_SETS: dict[str, list[str]] = {
    "": [],
    "no_cp": [CHANGEPOINT_FEATURE],
    "no_kin": [KINEMATIC_FEATURE],
    "no_s3d": [VIDEO_FEATURE],
}

#: Which points of ``LOSS_TERMS`` × ``FEATURE_SETS`` are trained. The full
#: cross is 12 cells per individual per architecture; dropping a feature group
#: under ``no_circle`` as well would answer nothing the other two do not.
CROSS: list[tuple[str, str]] = [
    ("all", ""),
    ("no_smooth", ""),
    ("no_circle", ""),
    ("all", "no_cp"),
    ("all", "no_kin"),
    ("all", "no_s3d"),
    ("no_smooth", "no_cp"),
    ("no_smooth", "no_kin"),
    ("no_smooth", "no_s3d"),
]


def arm_name(loss: str, features: str) -> str:
    """``("all", "no_cp")`` → ``"all_no_cp"``; the full feature set adds nothing."""
    return f"{loss}_{features}" if features else loss


#: The arms, in the order the figures draw them. Every knob is spelled in
#: every arm, so no arm inherits one.
ARMS: dict[str, dict[str, Any]] = {
    arm_name(loss, features): {**LOSS_TERMS[loss], "train.drop_kinds": FEATURE_SETS[features]}
    for loss, features in CROSS
}

#: Appended to every run name, so one bench's folds stay together under ``runs/``.
SUFFIX = "loss"

OUTPUT = CONFIG_DIR / "bench_loss.pdf"
TABLE = CONFIG_DIR / "bench_loss.tsv"

#: Every run appends a timestamped log here, so a night that ended in a
#: killed process can still be read the next morning.
LOG_DIR = CONFIG_DIR / "bench_logs"

#: Stop a pass after this many cells fail back to back. One cell failing is
#: weather (a locked file, a CUDA OOM); this many in a row is something
#: systemic, and grinding through the remaining hundred would only bury the
#: first traceback under a hundred copies of itself.
MAX_CONSECUTIVE_FAILURES = 8

#: :func:`main` exit codes — ``scripts/bench_watch.ps1`` reads them.
EXIT_DONE, EXIT_MORE_TO_DO, EXIT_NO_PROGRESS = 0, 1, 2

logger = logging.getLogger("bench")


def display_name(individual: str) -> str:
    """``crow1`` → ``crow 1`` — the config stem, as the figures spell it."""
    return re.sub(r"(?<=\D)(\d+)$", r" \1", individual)


def run_name(individual: str, architecture: str, arm: str) -> str:
    """The cell's base run name; its cross-validation is ``cv_`` + this."""
    return f"{individual}_{architecture}_{arm}_{SUFFIX}"


def project_for(individual: str, architecture: str, arm: str) -> eto.segment.Project:
    """The individual's config with the cell's architecture, arm and run name pinned."""
    overrides = eto.segment.as_overrides(
        {
            "model.architecture": architecture,
            "train.run_name": run_name(individual, architecture, arm),
            **ARMS[arm],
        }
    )
    return eto.segment.Project(CONFIG_DIR / f"{individual}.yaml", *overrides)


def finished_folds(project: eto.segment.Project) -> dict[str, Path]:
    """Session source → its newest fold run that finished its test evaluation.

    A fold interrupted before ``test_eval.npz`` does not count, so rerunning
    the bench trains it again rather than reading half a result.
    """
    config = project.config
    folds_dir = config.runs_dir / cross_validation_name_for(config)
    done: dict[str, Path] = {}
    for spec in config.sessions:
        candidates = sorted(folds_dir.glob(f"fold-{session_id(spec.source)}_*"))
        finished = [d for d in candidates if (d / TEST_METRICS_FILE).is_file() and (d / EVAL_ARRAYS_FILE).is_file()]
        if finished:
            done[str(spec.source)] = finished[-1]
    return done


def has_predictions(source: str, run_dir: Path) -> bool:
    """Whether *run_dir*'s prediction set for the session at *source* was written beside it."""
    labels = prediction_run_dir(Path(source), run_dir.name, "").parent
    return any(labels.glob(f"{PREDICTIONS_PREFIX}_{run_dir.name}_*/*_predictions.tsv"))


def check_kinds_declared(project: eto.segment.Project, arm: str) -> None:
    """Materialise if needed, and refuse an ablation the layout cannot perform.

    ``train.drop_kinds`` names a kind, and a column that declares none is
    always kept: an arm asking for a kind the materialised layout does not
    hold would train the *reference* model under an ablation's name and
    quietly report it as a result. The layout is a derived artefact, so a
    session described since it was written is fixed by materialising again
    (the feature arrays come out identical — ``kind`` is a label, and only
    ``normalise`` changes arithmetic); a kind still missing after that is the
    session's to declare, not this bench's to guess.
    """
    wanted = set(ARMS[arm]["train.drop_kinds"])
    if not wanted:
        return
    data_dir = project.config.data_dir
    if (data_dir / COLUMNS_FILE).is_file():
        if wanted <= set(read_layout(data_dir).kinds) - {None}:
            return
        logger.info("%s declares no column of kind %s — materialising again", data_dir, sorted(wanted))
    project.materialise()
    missing = sorted(wanted - (set(read_layout(data_dir).kinds) - {None}))
    if missing:
        raise RuntimeError(
            f"Arm {arm!r} drops kind(s) {missing}, which no column of {data_dir} declares, so it would "
            f"train the full model under an ablation's name. Run `python scripts/describe_sessions.py` "
            f"to write each session's .ethograph/schema.yaml, then try again."
        )


def cross_validate_cell(individual: str, architecture: str, arm: str) -> dict[str, Path]:
    """Train the cell's missing folds, if any, and return every session's finished fold.

    A fold evaluates before it predicts, so a crash between the two (the
    session file locked by another process, say) leaves a fold with metrics
    and no prediction set. Those are completed here by inference alone —
    the trained run is on disk — never by training again.
    """
    project = project_for(individual, architecture, arm)
    check_kinds_declared(project, arm)
    sessions = [str(s.source) for s in project.config.sessions]
    done = finished_folds(project)
    missing = [s for s in sessions if s not in done]
    label = f"{display_name(individual)} / {architecture} / {arm}"
    if missing:
        logger.info("[%s] %d of %d folds to train: %s", label, len(missing), len(sessions), ARMS[arm])
        project.cross_validate(folds=missing)
        done = finished_folds(project)
        still_missing = [s for s in sessions if s not in done]
        if still_missing:
            raise RuntimeError(
                f"[{label}] cross_validate returned, but these folds wrote no test evaluation: {still_missing}"
            )
    else:
        logger.info("[%s] every fold finished — read back", label)
    for source, run_dir in done.items():
        if not has_predictions(source, run_dir):
            logger.info("[%s] %s evaluated but never predicted its session — predicting now", label, run_dir.name)
            project.inference(run=run_dir, sessions=[source])
    return done


#: How long to wait before the one retry of a cell that hit a transient HDF5 failure.
HDF_RETRY_S = 60.0


def run_cell(individual: str, architecture: str, arm: str) -> dict[str, Path]:
    """:func:`cross_validate_cell`, retried once if netCDF/HDF5 fails to open a session.

    ``NetCDF: HDF error`` is what the HDF5 library reports for a file it could
    not open at that moment — a lock held by another process, an antivirus
    pass over a large file — and it has been seen once in a night of folds on
    a file that opens fine before and after. The work already done is on
    disk, so the retry only picks up what the failure interrupted (the
    prediction set, usually). Anything else is raised as is.
    """
    try:
        return cross_validate_cell(individual, architecture, arm)
    except RuntimeError as exc:
        if "HDF error" not in str(exc):
            raise
        logger.warning(
            "[%s / %s / %s] %s — waiting %.0f s, then retrying once",
            display_name(individual),
            architecture,
            arm,
            exc,
            HDF_RETRY_S,
        )
        time.sleep(HDF_RETRY_S)
        return cross_validate_cell(individual, architecture, arm)


def cells() -> list[tuple[str, str, str]]:
    """Every (individual, architecture, arm) the bench trains, in order."""
    return [(i, a, m) for i in INDIVIDUALS for a in ARCHITECTURES for m in ARMS]


def cell_is_finished(individual: str, architecture: str, arm: str) -> bool:
    """Whether every fold of this cell has both its test evaluation and its prediction set.

    Read-only and cheap (a YAML read and a glob), so the sweep can skip what
    is done without building a model or opening a session.
    """
    project = project_for(individual, architecture, arm)
    done = finished_folds(project)
    if len(done) != len(project.config.sessions):
        return False
    return all(has_predictions(source, run_dir) for source, run_dir in done.items())


def train_all(passes: int) -> list[str]:
    """Train every unfinished cell, surviving the ones that fail; return what is still unfinished.

    A cell that raises is logged with its traceback and left for the next
    pass rather than taking the other hundred down with it — the point of an
    unattended run is that the morning finds 107 finished cells and one
    traceback, not one traceback. Everything already on disk is skipped, so a
    pass costs nothing for the cells that finished earlier (or in an earlier
    process, after a crash).
    """
    for pass_no in range(1, passes + 1):
        todo = [c for c in cells() if not cell_is_finished(*c)]
        if not todo:
            logger.info("Every one of the %d cells has finished.", len(cells()))
            return []
        logger.info("Pass %d of %d: %d of %d cells to do", pass_no, passes, len(todo), len(cells()))
        failed: list[str] = []
        consecutive = 0
        for done_count, (individual, architecture, arm) in enumerate(todo, start=1):
            label = f"{display_name(individual)} / {architecture} / {arm}"
            try:
                run_cell(individual, architecture, arm)
                consecutive = 0
            except Exception:
                logger.exception("[%s] failed — leaving it for the next pass", label)
                failed.append(label)
                consecutive += 1
                # A CUDA OOM leaves its allocation behind; the next cell would
                # inherit a card that is already full and fail for a reason
                # that is not its own.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "%d cells failed in a row — stopping this pass rather than burying the first "
                        "traceback under a hundred copies of it",
                        consecutive,
                    )
                    untried = [f"{display_name(i)} / {a} / {m}" for i, a, m in todo[done_count:]]
                    return failed + untried
        if failed:
            logger.warning("Pass %d left %d cell(s) unfinished: %s", pass_no, len(failed), failed)
    return [f"{display_name(i)} / {a} / {m}" for i, a, m in cells() if not cell_is_finished(i, a, m)]


def collect() -> tuple[list[FactorCell], pd.DataFrame, ClassTable | None]:
    """Every cell with at least one finished fold, its folds loaded, plus one row per fold."""
    cells: list[FactorCell] = []
    rows: list[dict[str, Any]] = []
    classes: ClassTable | None = None
    for individual in INDIVIDUALS:
        for architecture in ARCHITECTURES:
            for arm in ARMS:
                project = project_for(individual, architecture, arm)
                done = finished_folds(project)
                if not done:
                    logger.warning("%s / %s / %s: no finished fold", display_name(individual), architecture, arm)
                    continue
                folds = []
                for spec in project.config.sessions:
                    run_dir = done.get(str(spec.source))
                    if run_dir is None:
                        continue
                    e = load_run_eval(run_dir, name=session_id(spec.source))
                    folds.append(e)
                    row: dict[str, Any] = {
                        "individual": display_name(individual),
                        "architecture": architecture,
                        "arm": arm,
                        "session": e.name,
                        "run_dir": str(run_dir),
                        "train_seconds": e.train_seconds,
                    }
                    for stage, metrics in (("raw", e.raw), ("postprocessed", e.processed)):
                        row.update({f"{stage}.{k}": v for k, v in metrics.items() if k != "classwise"})
                    rows.append(row)
                    if classes is None:
                        classes = ClassTable.from_dict(
                            yaml.safe_load((run_dir / "classes.yaml").read_text(encoding="utf-8"))
                        )
                cells.append(FactorCell(display_name(individual), architecture, arm, folds))
    return cells, pd.DataFrame(rows), classes


def setup_logging() -> Path:
    """Log to the console and to a timestamped file under :data:`LOG_DIR`."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"bench_{dt.datetime.now():%Y%m%d-%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
    )
    return path


def draw_report() -> None:
    """Write the TSV and the PDF from whatever has finished."""
    cells_, table, classes = collect()
    if not cells_ or classes is None:
        raise SystemExit("No finished folds under any individual's runs/ — nothing to draw.")
    table.to_csv(TABLE, sep="\t", index=False)
    select_on = eto.segment.Project(CONFIG_DIR / f"{INDIVIDUALS[0]}.yaml").config.train.select_on
    column = f"postprocessed.{select_on}"
    summary = (
        table.groupby(["individual", "architecture", "arm"])[column]
        .mean()
        .unstack("arm")
        .reindex(columns=list(ARMS))
    )
    print(f"\nMean post-processed {select_on} over held-out sessions:\n{summary.round(1).to_string()}\n")

    title = (
        f"Loss + changepoint-feature ablation — {len(INDIVIDUALS)} individuals × "
        f"{len(ARCHITECTURES)} architectures, cross-validated"
    )
    path = write_factorial_pdf(
        OUTPUT,
        cells_,
        classes,
        title=title,
        classwise_key=select_on if select_on.startswith("f1@") else None,
    )
    logger.info("Wrote %s and %s", path, TABLE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--report-only", action="store_true", help="draw from the folds that finished; train nothing")
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="sweep the unfinished cells this many times before giving up on them (default 2)",
    )
    args = parser.parse_args()
    log_path = setup_logging()
    logger.info("Logging to %s", log_path)

    if args.report_only:
        draw_report()
        return EXIT_DONE

    before = sum(cell_is_finished(*c) for c in cells())
    unfinished = train_all(args.passes)
    after = sum(cell_is_finished(*c) for c in cells())
    logger.info("Cells finished: %d → %d of %d", before, after, len(cells()))

    # The report is worth having even when cells are missing, and a failure to
    # draw it must not be read as "the training is unfinished" — that is what
    # decides whether the supervisor starts another process.
    try:
        draw_report()
    except Exception:
        logger.exception("Could not draw the report from what has finished")

    if not unfinished:
        return EXIT_DONE
    logger.warning("%d cell(s) still unfinished:\n  %s", len(unfinished), "\n  ".join(unfinished))
    if after > before:
        logger.info("Progress was made this run — rerunning continues from here.")
        return EXIT_MORE_TO_DO
    logger.error("No cell finished this run. Rerunning would repeat it; read the traceback above first.")
    return EXIT_NO_PROGRESS


if __name__ == "__main__":
    raise SystemExit(main())
