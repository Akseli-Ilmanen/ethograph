"""The wizard's session file always ends in ``.nc``, whatever the user typed."""

from __future__ import annotations

import pytest

from ethograph.gui.wizard_state import session_output_path


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("C:/data/session", "C:/data/session.nc"),
        ("  C:/data/session.nc  ", "C:/data/session.nc"),
        ("C:/data/Session.NC", "C:/data/Session.NC"),
        ("", ""),
    ],
)
def test_session_output_path_enforces_suffix(typed, expected):
    assert session_output_path(typed) == expected
