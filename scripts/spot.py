"""Run the spot pipeline on data/spot/project.yaml, stage by stage — you name the stages.

Which stages you need is decided by what is available when the model runs
(docs/source/models/spot/index.md, "Which stages to run"):

    # pose in every session, now and later: features ride into the GRU
    python scripts/spot.py materialise baseline evaluate
    python scripts/spot.py inference --run ctx2s_res10ms_features --sessions 20260308_01

    # video only, no pose anywhere
    python scripts/spot.py materialise baseline evaluate           # with no features: listed

    python scripts/spot.py crossval                                # one fold per session

Stages: materialise | baseline | evaluate | inference | crossval.
`evaluate` scores every run and writes runs/compare.tsv;
`--run` names the run `inference` predicts with (default: the newest under runs/).
"""

from __future__ import annotations

import argparse
import logging
import sys

import ethograph as eto
from ethograph.utils.logging import enable_console_logging

CONFIG = "data/spot/project.yaml"
STAGES = ("materialise", "baseline", "evaluate", "crossval", "inference")


def finished_run(run_dir, epochs: int) -> bool:
    return (run_dir / f"checkpoint_{epochs - 1:03d}.pt").is_file()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=STAGES, help="which stages, in this order — see the recipes above")
    parser.add_argument("--limit", type=int, default=None, help="first N trials per session (a smoke run)")
    parser.add_argument("--sessions", nargs="*", default=None, help="sessions to predict into (default: all)")
    parser.add_argument(
        "--run", default=None, help="run to predict with, a name under runs/ (default: the newest)"
    )
    parser.add_argument("--force", action="store_true", help="rerun stages that look finished")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="trials decoded at once (default: cores minus two, capped at 16)",
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = parser.parse_args(argv)
    stages = args.stages
    enable_console_logging("run")  # the pipelines already print their own records on import
    for noisy in ("ethograph.io", "ethograph.gui", "ethograph.labels"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    overrides = ([f"trials.limit={args.limit}"] if args.limit else []) + list(args.set)
    project = eto.spot.Project(CONFIG, *overrides)
    cfg = project.config
    log = logging.getLogger("run")
    log.info("%s | frames -> %s", project, cfg.frames_dir)

    if "materialise" in stages:
        if (cfg.dataset_dir / "train.json").is_file() and not args.force:
            log.info("materialise: dataset/ exists, features/ refreshed only — pass --force to redo")
        project.materialise(workers=args.workers)

    if "baseline" in stages:
        name = project.run_name()
        if finished_run(cfg.run_dir(name), cfg.train.epochs) and not args.force:
            log.info("baseline: %s finished — skipped (--force to redo)", name)
        else:
            project.train()

    if "evaluate" in stages:
        # Every run, so compare.tsv is complete.
        from ethograph.spot.inference import resolve_run_dir, run_reads_features, trained_runs

        for run_dir in trained_runs(cfg):
            project.evaluate(run=run_dir)
            if run_reads_features(resolve_run_dir(cfg, run_dir)):
                # what the features contribute: the same model, features zeroed
                project.evaluate(run=run_dir, zero_features=True)
        table = project.compare()
        if len(table) > 1:
            log.info("compare (%s):\n%s", cfg.runs_dir / "compare.tsv", table.to_string(index=False))

    if "crossval" in stages:
        for fold in project.cross_validate(sessions=args.sessions, workers=args.workers):
            log.info("fold: %s -> %s", fold.name, fold.run_dir / "test_metrics.yaml")

    if "inference" in stages:
        for tsv in project.inference(run=args.run, sessions=args.sessions, workers=args.workers):
            log.info("predictions: %s", tsv)


if __name__ == "__main__":
    main(sys.argv[1:])
