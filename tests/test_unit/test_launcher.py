"""The console script compiles an uncompiled environment once, before importing anything heavy."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

import ethograph_launcher


@pytest.fixture
def fake_site_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    site = tmp_path / "site-packages"
    pkg = site / "probe_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("X = 1\n", newline="\n")
    (pkg / "sub.py").write_text("Y = 2\n", newline="\n")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    return site


def test_uncompiled_probe_is_reported_until_compiled(fake_site_packages: Path) -> None:
    assert ethograph_launcher.uncompiled_probes(["probe_pkg", "not_installed_anywhere"]) == ["probe_pkg"]

    took = ethograph_launcher.precompile(str(fake_site_packages), workers=1)

    assert took is not None
    pyc = importlib.util.cache_from_source(str(fake_site_packages / "probe_pkg" / "sub.py"))
    assert Path(pyc).exists()
    assert ethograph_launcher.uncompiled_probes(["probe_pkg"]) == []


def test_precompile_respects_dont_write_bytecode(fake_site_packages: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    assert ethograph_launcher.precompile(str(fake_site_packages), workers=1) is None
    assert ethograph_launcher.uncompiled_probes(["probe_pkg"]) == ["probe_pkg"]


def test_launcher_imports_nothing_heavy() -> None:
    """The whole point: the probe runs before numpy and friends are imported."""
    source = Path(ethograph_launcher.__file__).read_text()
    for heavy in ("import ethograph\n", "import numpy", "from ethograph.cli import", "qtpy"):
        top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not any(heavy in line for line in top_level), heavy
