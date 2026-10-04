"""Parametric space shapes: their wireframes, their YAML round trip, and the editor's preview."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ethograph.gui.plots_space import (
    geometry_config,
    load_geometry_yaml,
    load_library_geometries,
    load_reference_geometries,
    write_geometry_yaml,
)
from ethograph.gui.space_shapes import SHAPE_TYPES, Shape, axis_ticks, parse_shapes, shape_wireframe


def _extent(shape: Shape) -> np.ndarray:
    vertices, _ = shape_wireframe(shape)
    return vertices.max(axis=0) - vertices.min(axis=0)


@pytest.mark.parametrize("shape_type", SHAPE_TYPES)
def test_every_edge_indexes_a_vertex(shape_type: str):
    vertices, edges = shape_wireframe(Shape("s", shape_type))
    assert edges
    assert max(max(e) for e in edges) < len(vertices)


def test_sizes_and_centre_place_the_solid():
    box = Shape("b", "box", center=(1.0, 2.0, 3.0), params={"width": 2.0, "length": 4.0, "height": 6.0})
    vertices, _ = shape_wireframe(box)
    np.testing.assert_allclose(vertices.mean(axis=0), [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(_extent(box), [2.0, 4.0, 6.0], atol=1e-12)

    rotated = Shape("r", "rectangle", params={"width": 2.0, "length": 4.0, "rotation": 90.0})
    np.testing.assert_allclose(_extent(rotated), [4.0, 2.0, 0.0], atol=1e-12)


def test_a_cylinder_runs_along_its_axis():
    cylinder = Shape("c", "cylinder", params={"radius": 0.5, "height": 10.0}, axis="x")
    np.testing.assert_allclose(_extent(cylinder), [10.0, 1.0, 1.0], atol=1e-2)


def test_retyping_keeps_the_sizes_both_types_share():
    box = Shape("b", "box", params={"width": 2.0, "length": 3.0, "height": 4.0, "rotation": 30.0})
    assert box.retyped("rectangle").params == {"width": 2.0, "length": 3.0, "rotation": 30.0}
    assert box.retyped("cylinder").params == {"radius": 0.5, "height": 4.0}


def test_a_saved_file_reads_back_with_its_references_untouched(tmp_path: Path):
    references = [{"name": "floor", "vertices": [[0, 0, 0], [1, 0, 0]], "edges": [[0, 1]], "color": "black"}]
    shapes = [Shape("perch", "cylinder", (1.0, 2.0, 1.5), {"radius": 0.05, "height": 1.0}, axis="y", color="red")]
    path = tmp_path / "arena.yaml"
    write_geometry_yaml(path, geometry_config({"references": references}, shapes))

    cfg = load_geometry_yaml(path)
    assert cfg["references"] == references
    assert parse_shapes(cfg) == shapes
    drawn = load_library_geometries([tmp_path])["arena"]
    assert [r.name for r in drawn] == ["floor", "perch"]
    assert drawn[1].color == "red"


def test_the_unsaved_edit_stands_in_only_for_its_own_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / "home"))
    project = tmp_path / "study"
    write_geometry_yaml(project / "space" / "arena.yaml", geometry_config({}, [Shape("file", "sphere")]))
    preview = {"name": "arena", "config": geometry_config({}, [Shape("edit", "circle")])}
    state = SimpleNamespace(
        space_library_geometry="arena",
        space_geometry_preview=preview,
        nc_file_path=None,
        nwb_file_path=None,
        project_path=str(project),
    )
    monkeypatch.setattr("ethograph.gui.plots_space.project_dir_of", lambda _state: project)

    assert [r.name for r in load_reference_geometries(state)] == ["edit"]
    state.space_geometry_preview = {**preview, "name": "other"}
    assert [r.name for r in load_reference_geometries(state)] == ["file"]


def test_axis_numbers_are_round_coordinates_around_the_centre():
    assert axis_ticks(1.0, 5.0) == [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
    assert axis_ticks(0.25, 0.05) == [0.2, 0.22, 0.24, 0.26, 0.28, 0.3]
    assert "-0.0" not in map(str, axis_ticks(0.0, 3.0)), "a tick at zero never reads -0"
