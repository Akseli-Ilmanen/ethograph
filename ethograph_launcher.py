"""The ``ethograph`` and ``ethograph-gui`` console scripts.

This module lives *outside* the ``ethograph`` package on purpose. Importing
the package pulls in numpy, xarray, pandas, Qt and friends, and when the
environment holds no bytecode yet (``uv tool install`` and ``uv pip install``
do not compile it) that first import compiles every module it touches, one
file at a time, with the GUI nowhere to be seen for minutes. Here only the
standard library is imported, so the whole environment can be compiled once,
in parallel, before :mod:`ethograph.cli` takes over.
"""

from __future__ import annotations

import compileall
import importlib.util
import os
import sys
import sysconfig
import time
import warnings
from collections.abc import Iterable

# Heavy packages the GUI always imports. Each one's ``__init__`` has bytecode
# iff it has been imported (or compiled) once in this environment.
PROBES = ("numpy", "scipy", "pandas", "xarray", "matplotlib", "pygfx", "qtpy", "h5py", "pynwb", "pynapple")


def uncompiled_probes(names: Iterable[str] = PROBES) -> list[str]:
    """The installed packages among *names* whose ``__init__`` has no bytecode yet."""
    missing = []
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            continue
        if not os.path.exists(importlib.util.cache_from_source(spec.origin)):
            missing.append(name)
    return missing


def precompile(site_packages: str | None = None, workers: int = 0) -> float | None:
    """Compile every ``.py`` under *site_packages* (default: this environment's).

    Returns the seconds it took, or ``None`` when bytecode is disabled or the
    directory is not writable. ``workers=0`` uses every CPU.
    """
    if sys.dont_write_bytecode:
        return None
    if site_packages is None:
        site_packages = sysconfig.get_paths()["purelib"]
    if not os.access(site_packages, os.W_OK):
        return None
    start = time.perf_counter()
    # The worker processes read their warning filters from the environment;
    # third-party files with a stray "\s" would otherwise print SyntaxWarnings.
    previous = os.environ.get("PYTHONWARNINGS")
    os.environ["PYTHONWARNINGS"] = "ignore::SyntaxWarning"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compileall.compile_dir(site_packages, quiet=2, workers=workers)
    finally:
        if previous is None:
            del os.environ["PYTHONWARNINGS"]
        else:
            os.environ["PYTHONWARNINGS"] = previous
    return time.perf_counter() - start


def precompile_on_first_launch() -> None:
    """Compile the environment once, saying so, when the GUI's packages never ran here."""
    if not uncompiled_probes():
        return
    print(
        "ethograph: first launch - compiling the installed packages to bytecode "
        "so later launches start fast (one-time, about a minute) ...",
        file=sys.stderr,
        flush=True,
    )
    took = precompile()
    if took is not None:
        print(f"ethograph: compiled in {took:.0f} s", file=sys.stderr, flush=True)


def main() -> None:
    """``ethograph <command>``."""
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        precompile_on_first_launch()
    from ethograph.cli import main as cli_main

    cli_main()


def launch() -> None:
    """``ethograph-gui``."""
    precompile_on_first_launch()
    from ethograph.cli import launch as cli_launch

    cli_launch()


if __name__ == "__main__":
    main()
