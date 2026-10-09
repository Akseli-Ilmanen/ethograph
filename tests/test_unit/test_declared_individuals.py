"""Who a session's individuals are, when the dataset and the session record disagree.

The dataset's individual dim is the first word, the record's list adds to it,
and the pynapple placeholder stands in only when neither names anyone.
"""

from __future__ import annotations

from types import SimpleNamespace

from ethograph.gui.app_state import ObservableAppState
from ethograph.io.catalog import ComboSpec


def _state(data: tuple[str, ...] | None, record: list[str], placeholder: bool = False):
    combos = {"individual": ComboSpec("individual", data, placeholder=placeholder)} if data is not None else {}
    catalog = SimpleNamespace(
        combos=combos,
        individual_combo="individual" if data is not None else None,
    )
    state = SimpleNamespace(
        data_loader=SimpleNamespace(catalog=catalog),
        nwb_alignment=SimpleNamespace(individuals=record),
    )
    return ObservableAppState.declared_individuals(state)


def test_dataset_dim_comes_first_and_record_adds_to_it():
    assert _state(("Crow1",), ["Crow2", "Crow1"]) == ["Crow1", "Crow2"]


def test_record_alone_names_them_when_dataset_has_no_dim():
    assert _state(None, ["Crow1"]) == ["Crow1"]


def test_record_overrides_a_placeholder():
    assert _state(("individual_0",), ["Crow1"], placeholder=True) == ["Crow1"]


def test_placeholder_stands_in_when_nothing_declares_a_name():
    assert _state(("individual_0",), [], placeholder=True) == ["individual_0"]
