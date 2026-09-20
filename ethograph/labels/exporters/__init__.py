"""Lab-specific export columns, chosen in the export settings.

``labels/export.py`` produces the columns that hold for any dataset, and every
exporter here runs *after* it, on the table it produced. So an exporter is
always **standard plus its own**: there is no way to choose one and lose the
standard columns, and :func:`run` refuses a result that dropped one.

Anything that only means something for one lab lives here, as a named
exporter:

.. code-block:: python

    @register("mylab", "My lab: the columns my analysis expects")
    def my_columns(df: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
        df["my_column"] = ...
        return df

Add a module beside this one, register it, import it in :func:`_load_builtin`
below, and open a PR. The name is what the user picks in **Export labels…**;
it is stored in the global settings, so it follows the user rather than the
session.

An exporter may add columns and change its own; it may not drop the columns a
labels file is made of --- :func:`run` refuses that by name, since a table
missing them would not load back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ethograph.labels.tsv_store import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

#: The exporter every session gets unless another is chosen: the columns from
#: ``labels/export.py`` and nothing else.
STANDARD = "standard"


@dataclass(frozen=True)
class ExportContext:
    """Everything an exporter may read about the session being exported.

    Fields are added here rather than to each exporter's signature, so a lab
    needing something new does not change how every other exporter is called.
    """

    #: The data tree, for a variable or a trial ``attrs`` entry. ``None`` for a
    #: session that has no xarray tree (a pynapple or NWB one).
    dt: object = None
    #: The session alignment, for trial timing and media.
    alignment: object = None
    #: The trial metadata table.
    metadata_df: pd.DataFrame | None = None
    #: The session folder, for a sidecar file the lab keeps beside its data.
    session_dir: Path | None = None


#: Takes the enriched labels table and the session's context; returns the table.
Exporter = Callable[[pd.DataFrame, ExportContext], pd.DataFrame]

_REGISTRY: dict[str, tuple[Exporter, str]] = {}


def register(name: str, description: str) -> Callable[[Exporter], Exporter]:
    """Register an exporter under *name*, shown in the settings as *description*."""

    def decorator(func: Exporter) -> Exporter:
        if name in _REGISTRY:
            raise ValueError(f"Export '{name}' is already registered by {_REGISTRY[name][0].__module__}.")
        _REGISTRY[name] = (func, description)
        return func

    return decorator


@register(STANDARD, "Standard — duration, sequence, session timing, trial metadata")
def _standard(df: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    """The columns every dataset gets; ``labels/export.py`` has already added them."""
    return df


def _load_builtin() -> None:
    """Import every exporter module, so registering is a side effect of import."""
    from ethograph.labels.exporters import crowlab  # noqa: F401


def choices() -> list[tuple[str, str]]:
    """Every registered exporter as ``(name, label)`` for a combo, standard first.

    A lab's exporter runs *after* the standard columns and only adds to them,
    so its label says so: picking one is never a choice to go without the
    standard export.
    """
    _load_builtin()
    rest = sorted((name, f"Standard + {desc}") for name, (_, desc) in _REGISTRY.items() if name != STANDARD)
    return [(STANDARD, _REGISTRY[STANDARD][1]), *rest]


def get(name: str) -> Exporter:
    """The exporter registered as *name*.

    A name no longer registered (a lab's module removed, a typo in the
    settings) is a bad setting, not a bug: it falls back to the standard
    exporter and says so, rather than failing the save.
    """
    _load_builtin()
    if name in _REGISTRY:
        return _REGISTRY[name][0]
    logger.warning(
        "Unknown label exporter %r; exporting the standard columns. Known exporters: %s.",
        name,
        ", ".join(sorted(_REGISTRY)),
    )
    return _REGISTRY[STANDARD][0]


def run(exporter: Exporter, df: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    """Apply *exporter*, refusing a result that dropped a required column."""
    out = exporter(df, ctx)
    missing = REQUIRED_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(
            f"Export '{exporter.__module__}.{exporter.__name__}' dropped required "
            f"column(s) {sorted(missing)}; an exporter may add columns, never remove these."
        )
    return out
