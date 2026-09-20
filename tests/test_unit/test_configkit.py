"""The config toolkit both scripted pipelines stand on (``utils/configkit.py``).

One contract, checked once for every pipeline: a config file, saved and read
back, is the same config. Anything a pipeline's schema forgets to name — a
path field left as a string, a nested section left as a dict — breaks it.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
import yaml

from ethograph.segment import config as segment_config
from ethograph.spot import config as spot_config

SEGMENT = {
    "sessions": ["a/Trial_data.nc", {"source": "b/Trial_data.nc", "name": "second"}],
    "features": {"columns": {"speed": {}}, "labels": {"branch": 1, "mapping": "mapping.txt"}},
    "train": {"split": {"holdout_sessions": ["a/Trial_data.nc"]}},
}
SPOT = {
    "sessions": ["a/Trial_data.nc", {"source": "b/Trial_data.nc", "name": "second"}],
    "labels": {"classes": [3]},
    "features": {"speed": {}},
    "frames": "frames",
}
PIPELINES = [pytest.param(segment_config, SEGMENT, id="segment"), pytest.param(spot_config, SPOT, id="spot")]


def _write(tmp_path: Path, data: dict) -> Path:
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data), encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"base": "base.yaml", "individual": "bird1"}), encoding="utf-8")
    return child


@pytest.mark.parametrize(("module", "data"), PIPELINES)
def test_a_saved_config_reads_back_equal(module: ModuleType, data: dict, tmp_path: Path):
    cfg = module.load_config(_write(tmp_path, data), ["trials.where.state=[sleep]"])
    assert cfg.individual == "bird1"
    assert cfg.sessions[0].source == (tmp_path / "a" / "Trial_data.nc").resolve()

    saved = module.save_config(cfg, tmp_path / "elsewhere" / "config.yaml")
    again = module.load_config(saved)

    assert module.config_to_dict(again) == module.config_to_dict(cfg)


@pytest.mark.parametrize(("module", "data"), PIPELINES)
def test_with_overrides_leaves_the_original_alone(module: ModuleType, data: dict, tmp_path: Path):
    cfg = module.load_config(_write(tmp_path, data))
    copy = module.with_overrides(cfg, individual="bird2")
    copy.sessions[0].name = "renamed"
    assert (cfg.individual, cfg.sessions[0].name) == ("bird1", None)


@pytest.mark.parametrize(("module", "data"), PIPELINES)
def test_a_session_is_selected_by_name_or_stem(module: ModuleType, data: dict, tmp_path: Path):
    cfg = module.load_config(_write(tmp_path, data))
    assert [s.label for s in cfg.select_sessions(["second"])] == ["second"]
    assert [s.label for s in cfg.select_sessions(["Trial_data"])] == [s.label for s in cfg.sessions]
    with pytest.raises(ValueError, match="No session matches"):
        cfg.select_sessions(["third"])
