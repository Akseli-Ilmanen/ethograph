"""Re-run a trained FERAL checkpoint on the test split with different inference chunking.

Inference chunking is independent of training (FERAL's ``max`` preset trains at 66 %
overlap and infers at 80 %). This writes a sibling export folder
``{OUT_DIR}_{name}`` holding only the ``inference`` split, the trained checkpoint as
``starting_checkpoint``, and the new ``eval_chunk_shift`` / ``eval_smoothing_window``;
``run_feral.py`` on it then skips training, and ``score_feral.py`` scores it as its own run.

``eval_chunk_shift`` must be a multiple of ``chunk_step``: only then do overlapping
chunks predict the same frames and get averaged, rather than interleaving.

Usage (either env):
    python reinfer_feral.py OUT_DIR NAME --shift 78 [--smoothing 9]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("name")
    parser.add_argument("--shift", type=int, required=True, help="eval_chunk_shift in raw video frames")
    parser.add_argument("--smoothing", type=int, default=None, help="eval_smoothing_window in raw video frames")
    args = parser.parse_args()

    source: Path = args.out_dir.resolve()
    overrides = yaml.safe_load((source / "feral_overrides.yaml").read_text(encoding="utf-8"))
    step = overrides["data"]["chunk_step"]
    if args.shift % step:
        raise ValueError(f"--shift {args.shift} is not a multiple of chunk_step {step}")
    checkpoint = source / "checkpoints" / f"{overrides['run_name']}_best_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    target = source.with_name(f"{source.name}_{args.name}")
    target.mkdir(exist_ok=True)
    labels = json.loads((source / "labels.json").read_text(encoding="utf-8"))
    labels["splits"] = {"inference": labels["splits"]["test"]}
    (target / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    shutil.copy(source / "videos.tsv", target / "videos.tsv")

    overrides["run_name"] = f"{overrides['run_name']}_{args.name}"
    overrides["starting_checkpoint"] = str(checkpoint)
    overrides["data"]["label_json"] = str(target / "labels.json")
    overrides["data"]["eval_chunk_shift"] = args.shift
    overrides["data"]["eval_smoothing_window"] = args.smoothing
    (target / "feral_overrides.yaml").write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    span = 64 * step
    print(f"{target}: shift {args.shift} ({1 - args.shift / span:.0%} overlap), smoothing {args.smoothing}")


if __name__ == "__main__":
    main()
