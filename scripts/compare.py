"""Draw the comparison figures over runs that already finished.

``scripts/bench.py`` compares architectures by training them; this compares them by
*reading what they wrote*. Every run writes ``test_metrics.yaml`` (the scalars
and class-wise F1) and ``test_eval.npz`` (the matched-segment IoUs and
onset/offset deltas) the moment it finishes, so a bench that crashed — or one
still running in another terminal — has already left everything this needs on
disk. Nothing is retrained.

Runs are grouped by **architecture** (each run's ``config.yaml`` says which),
and each architecture contributes its single best run — the one scoring
highest on :data:`SELECT_ON` — so a sweep of ten mstcn cells shows up as one
mstcn. It all lands in one timestamped ``compare.pdf``:

*page 1*
    The models against each other: segmental F1 at every IoU threshold they
    evaluated (``f1@50 … f1@90``), post-processed solid and raw dashed, beside
    the frame-level scores that curve leaves out.

*one page per model, titled with it*
    That model's own evaluation — the IoU illustration, its overall metrics,
    its IoU distribution with a TP/FP/FN inset, its boundary-delta histogram,
    and its class-wise F1 raw vs post-processed. It is the successor to the
    old ``plot_metrics_best_model``, reading a run directory instead of
    ``test_results_epoch{N}.npy``.

``search()`` and ``cross_validate()`` draw the comparison figure over their own
trials and folds (``searches/{name}/eval_comparison.pdf``,
``cross_validation/{name}/eval_comparison.pdf``). This one is for everything
else: a search's winner against another search's winner, a hand-trained run
against a swept cell, whatever finished before the crash.

    python scripts/compare.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

import ethograph as eto
from ethograph.segment.metrics import EVAL_ARRAYS_FILE
from ethograph.segment.plotting import RunEval, load_run_eval, write_model_report_pdf
from ethograph.segment.samples import ClassTable

CONFIG = Path(r"D:\Akseli\Code\ethograph\data\project.yaml")

#: Which runs to consider, as paths relative to ``runs/`` — a run trained by
#: hand is one level deep (``asformer_kin_v1_20260824_1200``), a search trial
#: or a swept cell two (``asformer_enc_alpha_kin_v1/trial003_…``). Empty
#: considers **every** run that wrote ``test_eval.npz``; grouping by
#: architecture keeps that readable, since only the best run of each survives.
RUNS: list[str] = []

#: Which metric picks an architecture's best run, read off the post-processed
#: test scores. ``f1@90`` is what ``bench.py``'s own search selects on, the metric
#: that still moves -- f1@50 saturates too early to tell close runs apart. ``frame_f1``
#: or ``acc`` work the same way as either.
SELECT_ON = "f1@90"

OUTPUT = CONFIG.with_name("compare.pdf")

logger = logging.getLogger("compare")


def run_dirs(runs_dir: Path) -> list[Path]:
    """The run directories to consider — :data:`RUNS` if it names any, else all of them."""
    if RUNS:
        chosen = [runs_dir / name for name in RUNS]
        missing = [d for d in chosen if not (d / EVAL_ARRAYS_FILE).is_file()]
        if missing:
            raise FileNotFoundError(
                f"No {EVAL_ARRAYS_FILE} in {[str(d) for d in missing]} — that run never finished its "
                "test evaluation (it may have been interrupted, or trained with train.split.test_fraction=0)."
            )
        return chosen
    return sorted(p.parent for p in runs_dir.rglob(EVAL_ARRAYS_FILE))


def architecture(run_dir: Path) -> str:
    """The architecture a run trained, from its own saved config."""
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["model"]["architecture"]


def best_per_architecture(dirs: list[Path]) -> list[tuple[str, Path, RunEval]]:
    """One ``(architecture, run_dir, eval)`` per architecture — its highest :data:`SELECT_ON` scorer."""
    best: dict[str, tuple[Path, RunEval]] = {}
    for d in dirs:
        arch = architecture(d)
        e = load_run_eval(d, name=arch)
        current = best.get(arch)
        if current is None or e.processed[SELECT_ON] > current[1].processed[SELECT_ON]:
            best[arch] = (d, e)
    return [(arch, *best[arch]) for arch in sorted(best)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = eto.segment.Project(CONFIG).config
    dirs = run_dirs(config.runs_dir)
    if not dirs:
        raise SystemExit(f"No finished runs under {config.runs_dir}.")

    chosen = best_per_architecture(dirs)
    logger.info("%d runs → %d architectures (best on postprocessed %s)", len(dirs), len(chosen), SELECT_ON)
    for arch, run_dir, e in chosen:
        scalars = {k: round(v, 2) for k, v in e.processed.items() if k != "classwise"}
        duration = f"{e.train_seconds:.0f} s" if e.train_seconds is not None else "unknown"
        logger.info("%-12s %-50s %-10s %s", arch, str(run_dir.relative_to(config.runs_dir)), duration, scalars)

    evals = [e for _, _, e in chosen]
    classes = ClassTable.from_dict(yaml.safe_load((chosen[0][1] / "classes.yaml").read_text(encoding="utf-8")))
    title = f"{len(evals)} architectures compared — best run of each on postprocessed {SELECT_ON}"
    logger.info("Wrote %s", write_model_report_pdf(OUTPUT, evals, classes, title=title))


if __name__ == "__main__":
    main()
