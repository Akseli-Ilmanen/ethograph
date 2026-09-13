"""One mapping.txt serves every import format: known names keep their ids, new ones append."""

from __future__ import annotations

from pathlib import Path

from ethograph.labels.converters import extend_mapping
from ethograph.labels.intervals import load_label_mapping


def test_known_names_keep_ids_and_attributes(tmp_path: Path):
    mapping = tmp_path / "mapping.txt"
    mapping.write_text("0 background\n1 walk 1\n5 peck 0 point\n", encoding="utf-8")

    name_to_id, added = extend_mapping(["peck", "run", "walk", "sil"], mapping)

    assert added == ["run"]
    assert name_to_id["walk"] == 1
    assert name_to_id["peck"] == 5
    assert name_to_id["run"] == 6
    assert "sil" not in name_to_id
    reread = load_label_mapping(mapping)
    assert reread[1]["branch"] == 1
    assert reread[5]["event_type"] == "point"


def test_no_new_names_leaves_file_untouched(tmp_path: Path):
    mapping = tmp_path / "mapping.txt"
    text = "0 background\n1 walk\n"
    mapping.write_text(text, encoding="utf-8")

    _, added = extend_mapping(["walk"], mapping)

    assert added == []
    assert mapping.read_text(encoding="utf-8") == text


def test_missing_file_is_created_with_background(tmp_path: Path):
    mapping = tmp_path / ".ethograph" / "mapping.txt"

    name_to_id, _ = extend_mapping(["b", "a"], mapping)

    assert name_to_id == {"background": 0, "a": 1, "b": 2}
    assert load_label_mapping(mapping)[0]["name"] == "background"
