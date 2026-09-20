"""Crow lab export columns.

``session``, ``session_trial`` and ``pulse_onsets`` mean something only for
datasets converted from the lab's MATLAB recordings. A session carrying none of
them comes back unchanged, so choosing this exporter elsewhere is harmless ---
but it is not the default.
"""

from __future__ import annotations

import pandas as pd

from ethograph.labels.exporters import ExportContext, register

#: Separates the pulse times of one trial in the exported cell.
PULSE_SEPARATOR = "–"


def _session_columns(df: pd.DataFrame, dt) -> pd.DataFrame:
    """``session`` and ``session_trial``, from the tree's ``session`` attr.

    A tree recorded without one simply has no session name, so both columns
    are left out.
    """
    session_name = getattr(dt, "attrs", {}).get("session")
    if session_name is None:
        return df
    df["session"] = session_name
    df["session_trial"] = df["trial"].map(lambda trial: f"{session_name}_{trial}")
    return df


def _pulse_onset_column(df: pd.DataFrame, dt) -> pd.DataFrame:
    """``pulse_onsets``, written once per trial --- on the trial's first row.

    The pulse times are a property of the trial, not of any one label, so
    repeating them on every row would say the same thing many times.
    """
    if not dt.trials or "pulse_onsets" not in dt.itrial(0):
        return df

    pulses = {trial: dt.trial(trial).pulse_onsets.values for trial in dt.trials}
    first_in_trial = ~df["trial"].duplicated()
    df["pulse_onsets"] = ""
    df.loc[first_in_trial, "pulse_onsets"] = df.loc[first_in_trial, "trial"].map(
        lambda trial: PULSE_SEPARATOR.join(map(str, pulses.get(trial, [])))
    )
    return df


@register("crowlab", "Crow lab — session, session_trial, pulse_onsets")
def crow_columns(df: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    """The lab's own columns, added to an enriched labels table."""
    if ctx.dt is None or df.empty:
        return df
    df = _session_columns(df, ctx.dt)
    df = _pulse_onset_column(df, ctx.dt)
    return df
