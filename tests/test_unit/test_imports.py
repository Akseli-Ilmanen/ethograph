from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import ethograph as eto


def _iter_ethograph_modules() -> list[str]:
    package_root = Path(eto.__file__).resolve().parent
    names = []
    for module in pkgutil.walk_packages([str(package_root)], prefix="ethograph."):
        names.append(module.name)
    return sorted(names)


def test_targeted_imports() -> None:
    from ethograph.gui.widgets_data import DataWidget
    from ethograph.gui.widgets_ephys import EphysWidget
    from ethograph.io.plot_sources import FileSource, PlotSource, WindowedBuffer, XarraySource
    from ethograph.io.time_model import (
        RestrictionWindow,
        TimeRange,
        TrialVideoBounds,
        compute_trial_video_bounds,
    )

    assert TimeRange is not None
    assert RestrictionWindow is not None
    assert TrialVideoBounds is not None
    assert compute_trial_video_bounds is not None
    assert FileSource is not None
    assert PlotSource is not None
    assert XarraySource is not None
    assert WindowedBuffer is not None
    assert DataWidget is not None
    assert EphysWidget is not None
    assert eto.from_continuous.__doc__


@pytest.mark.parametrize("module_name", [m for m in _iter_ethograph_modules() if m != "ethograph.__main__"])
def test_import_all_ethograph_modules(module_name: str) -> None:
    # Entry-point modules are excluded from smoke imports.
    importlib.import_module(module_name)
