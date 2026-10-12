"""Run the SAM fill, with before/after plots, over every session folder under a root.

A session folder is any direct child of the root that holds a ``segments.tsv``.
Its results go to a sibling folder ``{name}_filled`` (plots in ``plots_before``
and ``plots_after`` inside it). Videos already filled are skipped unless
``--redo`` is given, so an interrupted batch resumes where it stopped; a folder
that fails is reported and the rest continue. One model serves the whole batch.

    python scripts/samfill_batch.py "C:/Users/aksel/Documents/Pose_corrections" --keypoint pellet
    python scripts/samfill_batch.py ROOT --model sam2_small --low 0.9 --seed 0.99 --seeding anchor --redo
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ethograph import samfill
from ethograph.samfill.fill import REPORT_NAME

FILLED_SUFFIX = "_filled"
SEGMENTS_NAME = "segments.tsv"

#: Where a pellet detection cannot be true on this rig: the feeder box per camera, and
#: anything below the table edge. Such frames (plus 5 either side) are dropped and filled.
EXCLUSIONS = {
    "cam-1": samfill.PixelExclusion(boxes=((91, 3, 166, 76),), y_max=245, pad_frames=5),
    "cam-2": samfill.PixelExclusion(boxes=((101, 6, 179, 70),), y_max=245, pad_frames=5),
}


def session_folders(root: Path) -> list[Path]:
    """Direct children of *root* that hold a segments table and are not fill outputs themselves."""
    return sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.endswith(FILLED_SUFFIX) and (p / SEGMENTS_NAME).is_file()
    )


def run_folder(folder: Path, cfg: samfill.FillConfig, redo: bool) -> str:
    """Fill one session; videos already written are skipped unless *redo*, so a batch resumes mid-session."""
    filled = folder.parent / f"{folder.name}{FILLED_SUFFIX}"
    segments = folder / SEGMENTS_NAME
    t0 = time.time()
    if not (filled / "plots_before").is_dir() or redo:
        samfill.inspect_folders(folder, folder, segments, cfg, out_dir=filled / "plots_before")
    runs = samfill.fill_folders(folder, folder, segments, filled, cfg, skip_existing=not redo)
    if not any(runs.values()) and (filled / REPORT_NAME).is_file() and not redo:
        return "skipped (already filled; --redo to repeat)"
    samfill.inspect_folders(folder, folder, segments, cfg, filled_dir=filled, out_dir=filled / "plots_after")
    n_runs = sum(len(r) for r in runs.values())
    n_filled = sum(rep.n_filled for r in runs.values() for rep in r)
    return f"{len(runs)} videos, {n_runs} runs, {n_filled} frames filled in {time.time() - t0:.0f}s"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="folder whose session subfolders hold videos, pose files and segments.tsv")
    ap.add_argument("--keypoint", action="append", default=None, help="keypoint to fill (repeatable); default pellet")
    ap.add_argument("--model", default="sam2_tiny", choices=sorted(samfill.SAM_MODELS))
    ap.add_argument("--low", type=float, default=0.9, help="frames below this confidence are filled")
    ap.add_argument("--seed", type=float, default=0.99, help="a seed must reach this confidence")
    ap.add_argument("--seeding", default="gap", choices=["gap", "anchor"])
    ap.add_argument(
        "--output",
        default="netcdf",
        choices=["netcdf", "source"],
        help="netcdf: movement .nc with its edited flag (default); source: the tracker's own format",
    )
    ap.add_argument("--no-samurai", action="store_true", help="plain SAM 2 instead of SAMURAI")
    ap.add_argument("--redo", action="store_true", help="fill folders that already have a report")
    ap.add_argument(
        "--no-exclusions", action="store_true", help="keep detections inside the feeder box / below the table"
    )
    ap.add_argument("--only", default=None, help="only folders whose name contains this text")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    cfg = samfill.FillConfig(
        keypoints=args.keypoint or ["pellet"],
        low_confidence=args.low,
        seed_confidence=args.seed,
        model=args.model,
        samurai=not args.no_samurai,
        seeding=args.seeding,
        output=args.output,
        exclusions={} if args.no_exclusions else EXCLUSIONS,
    )
    folders = session_folders(args.root)
    if args.only:
        folders = [f for f in folders if args.only in f.name]
    if not folders:
        print(f"no session folder with a {SEGMENTS_NAME} under {args.root}", file=sys.stderr)
        return 1

    outcomes: dict[str, str] = {}
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] {folder.name}", flush=True)
        try:
            outcomes[folder.name] = run_folder(folder, cfg, args.redo)
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the batch; it is reported below
            logging.exception("%s failed", folder.name)
            outcomes[folder.name] = f"FAILED: {type(exc).__name__}: {exc}"
        print(f"    {outcomes[folder.name]}", flush=True)

    print("\nSummary")
    for name, outcome in outcomes.items():
        print(f"  {name:32s} {outcome}")
    return 1 if any(o.startswith("FAILED") for o in outcomes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
