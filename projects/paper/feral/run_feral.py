"""Train FERAL's lite preset on an export of export_feral.py (feral env, not ethograph).

FERAL writes ``answers/`` and ``checkpoints/`` into the working directory, so the
export folder is made the working directory. W&B stays off.

Usage:
    python run_feral.py OUT_DIR
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from feral.cli import _load_default_config
from feral.presets import apply_mode
from feral.train import main as train_main


def deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    overrides = yaml.safe_load((out / "feral_overrides.yaml").read_text(encoding="utf-8"))
    cfg = deep_merge(apply_mode(_load_default_config(), "lite"), overrides)
    cfg.pop("wandb", None)
    (out / "feral_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    os.chdir(out)
    train_main(cfg)


if __name__ == "__main__":
    main()
