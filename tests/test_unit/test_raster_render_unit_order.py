"""The pasted unit order: a Python list literal in, the same out."""

import pytest

from ethograph.gui.raster_render import format_unit_order, order_units, parse_unit_order


def test_copy_round_trips_through_paste():
    ids = [20, 35, 201]
    assert parse_unit_order(format_unit_order(ids)) == ids
    assert format_unit_order(ids) == "[20, 35, 201]"


@pytest.mark.parametrize("text", ["20, 35, 201", "[20 35 201]", "  [20,35,201]\n", "[20, 35, 201, 35]"])
def test_paste_is_lenient_about_brackets_separators_and_repeats(text):
    assert parse_unit_order(text) == [20, 35, 201]


@pytest.mark.parametrize("text", ["", "[]", "[20, foo]", "[1.5, 2]"])
def test_paste_refuses_anything_that_is_not_unit_ids(text):
    with pytest.raises(ValueError):
        parse_unit_order(text)


def test_units_outside_the_pasted_list_follow_in_their_own_order():
    assert order_units([1, 2, 3, 4], [3, 1, 99]) == [3, 1, 2, 4]
