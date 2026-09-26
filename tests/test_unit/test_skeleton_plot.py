"""The skeleton plot's gate and geometry — Qt-free where the logic is.

The gate decides which variables the add-panel popup offers "Skeleton" for;
the geometry turns one instant of ``position`` into points and bones.
"""

import numpy as np
import pytest
import xarray as xr

import ethograph as eto
from ethograph.gui.plots_skeleton import (
    SKELETON_2D,
    SKELETON_3D,
    PoseDims,
    pose_dims,
    pose_frame,
    skeleton_connections,
    skeleton_plot_types,
)
from ethograph.gui.pose_convert import COLOR_BY_INDIVIDUAL, COLOR_BY_KEYPOINT
from ethograph.io.catalog import XarrayLoader, catalog_from_xarray
from ethograph.io.derived import DerivedLoader
from ethograph.io.time_model import TimeRange

KEYPOINTS = ["nose", "tail", "wing"]
INDIVIDUALS = ["a", "b"]


def _dataset(axes: list[str], extra: dict[str, list[str]] | None = None) -> xr.Dataset:
    """A pose; *extra* adds dims beyond space/keypoint/individual (``{"position_type": [...]}``)."""
    time = np.linspace(0.0, 1.0, 11)
    extra = extra or {}
    dims = ("time", *extra, "space", "keypoint", "individual")
    shape = (len(time), *(len(v) for v in extra.values()), len(axes), len(KEYPOINTS), len(INDIVIDUALS))
    position = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return xr.Dataset(
        {
            "position": (dims, position),
            "speed": (("time", "individual"), np.zeros((len(time), len(INDIVIDUALS)))),
        },
        coords={"time": time, "space": axes, "keypoint": KEYPOINTS, "individual": INDIVIDUALS, **extra},
        attrs={"trial": 1},
    )


def _state(ds: xr.Dataset):
    catalog = catalog_from_xarray(ds, eto.from_datasets([ds]))

    class _State:
        data_loader = DerivedLoader(XarrayLoader(ds, catalog))
        window_bounds = TimeRange(0.0, 1.0)

    return _State()


def test_a_3d_position_is_offered_in_both_views():
    state = _state(_dataset(["x", "y", "z"]))
    assert skeleton_plot_types(state, "position") == [SKELETON_2D, SKELETON_3D]


def test_a_2d_position_is_offered_in_2d_only():
    state = _state(_dataset(["x", "y"]))
    assert skeleton_plot_types(state, "position") == [SKELETON_2D]


def test_only_position_is_offered():
    """Another variable — however it is shaped — is a trajectory, never a body."""
    state = _state(_dataset(["x", "y", "z"]))
    assert skeleton_plot_types(state, "speed") == []
    renamed = _state(_dataset(["x", "y"]).rename({"position": "beak_position"}))
    assert skeleton_plot_types(renamed, "beak_position") == []


def test_the_popup_offers_skeleton_beside_space():
    from ethograph.gui.source_popup import allowed_plot_types

    options = allowed_plot_types("feature", "position", _state(_dataset(["x", "y", "z"])))
    assert "Space (3D)" in options
    assert options[-2:] == [SKELETON_2D, SKELETON_3D]


def test_pose_dims_read_the_catalog_spelling():
    dims = pose_dims(_state(_dataset(["x", "y"])).data_loader)
    assert dims == PoseDims("space", ("x", "y"), "keypoint", tuple(KEYPOINTS), "individual", tuple(INDIVIDUALS))
    assert not dims.has_z
    assert pose_dims(_state(_dataset(["x", "y"])).data_loader, "speed") is None


def test_a_pose_without_x_and_y_is_not_a_pose():
    assert pose_dims(_state(_dataset(["u", "v"])).data_loader) is None


# ---------------------------------------------------------------------------
# One instant → vertex arrays
# ---------------------------------------------------------------------------

RED, BLUE, GREEN = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0)


def _coords() -> np.ndarray:
    """(I=2, K=3, D=2): individual ``b``'s tail is missing."""
    coords = np.arange(12, dtype=float).reshape(2, 3, 2)
    coords[1, 1] = np.nan
    return coords


def test_points_are_coloured_by_keypoint_and_padded_to_3d():
    frame = pose_frame(
        _coords(),
        KEYPOINTS,
        INDIVIDUALS,
        shown_keypoints=set(KEYPOINTS),
        color_by=COLOR_BY_KEYPOINT,
        color_map={"nose": RED, "tail": BLUE, "wing": GREEN},
        connections=[],
    )
    assert frame.points.shape == (5, 3)  # six keypoints, one NaN
    assert np.all(frame.points[:, 2] == 0.0)
    # Both animals' noses are red — the colour says the body part.
    assert [tuple(c) for c in frame.point_colors[[0, 3]]] == [RED, RED]
    assert len(frame.edges) == 0


def test_points_are_coloured_by_individual_when_asked():
    frame = pose_frame(
        _coords(),
        KEYPOINTS,
        INDIVIDUALS,
        shown_keypoints=set(KEYPOINTS),
        color_by=COLOR_BY_INDIVIDUAL,
        color_map={"a": RED, "b": BLUE},
        connections=[],
    )
    assert {tuple(c) for c in frame.point_colors[:3]} == {RED}
    assert {tuple(c) for c in frame.point_colors[3:]} == {BLUE}


def test_a_bone_needs_both_of_its_keypoints():
    """``b`` has no tail, so its nose–tail bone is dropped; ``a`` keeps it."""
    frame = pose_frame(
        _coords(),
        KEYPOINTS,
        INDIVIDUALS,
        shown_keypoints=set(KEYPOINTS),
        color_by=COLOR_BY_KEYPOINT,
        color_map={},
        connections=[("nose", "tail"), ("tail", "wing")],
    )
    assert frame.edges.shape == (4, 3)  # two bones for ``a``, none for ``b``
    assert np.allclose(frame.edges[0, :2], _coords()[0, 0])
    assert np.allclose(frame.edges[1, :2], _coords()[0, 1])
    assert frame.edge_colors.shape == (4, 4)


def test_bones_are_black_when_colour_says_the_keypoint():
    frame = pose_frame(
        _coords(),
        KEYPOINTS,
        INDIVIDUALS,
        shown_keypoints=set(KEYPOINTS),
        color_by=COLOR_BY_KEYPOINT,
        color_map={"nose": RED, "tail": BLUE, "wing": GREEN},
        connections=[("nose", "wing")],
    )
    assert {tuple(c) for c in frame.edge_colors} == {(0.0, 0.0, 0.0, 1.0)}


def test_bones_take_the_animals_colour_when_colour_says_the_individual():
    frame = pose_frame(
        _coords(),
        KEYPOINTS,
        INDIVIDUALS,
        shown_keypoints=set(KEYPOINTS),
        color_by=COLOR_BY_INDIVIDUAL,
        color_map={"a": RED, "b": BLUE},
        connections=[("nose", "wing")],
    )
    assert [tuple(c) for c in frame.edge_colors] == [RED, RED, BLUE, BLUE]


def test_hiding_a_keypoint_drops_its_point_and_its_bones():
    frame = pose_frame(
        _coords(),
        KEYPOINTS,
        INDIVIDUALS,
        shown_keypoints={"nose", "wing"},
        color_by=COLOR_BY_KEYPOINT,
        color_map={},
        connections=[("nose", "tail"), ("nose", "wing")],
    )
    assert frame.points.shape == (4, 3)
    assert frame.edges.shape == (4, 3)  # nose–wing for both animals, no nose–tail


def test_a_bone_naming_an_unknown_keypoint_is_skipped_not_fatal():
    frame = pose_frame(
        _coords()[:1],
        KEYPOINTS,
        ["a"],
        shown_keypoints=set(KEYPOINTS),
        color_by=COLOR_BY_KEYPOINT,
        color_map={},
        connections=[("nose", "beak")],
    )
    assert frame.points.shape == (3, 3)
    assert len(frame.edges) == 0


def test_an_unmapped_colour_falls_back_to_something_visible():
    frame = pose_frame(
        _coords()[:1],
        KEYPOINTS,
        ["a"],
        shown_keypoints=set(KEYPOINTS),
        color_by=COLOR_BY_KEYPOINT,
        color_map={},
        connections=[],
    )
    assert np.all(frame.point_colors[:, 3] == 1.0)


def test_skeleton_connections_read_the_resolved_config():
    config = {"connections": [{"start": "nose", "end": "tail", "color": "#FF0000"}, {"start": "tail", "end": "wing"}]}
    assert skeleton_connections(config) == [("nose", "tail"), ("tail", "wing")]
    assert skeleton_connections(None) == []


# ---------------------------------------------------------------------------
# The panel's data path (one widget, a real loader)
# ---------------------------------------------------------------------------


def test_an_extra_dim_is_recorded_so_it_can_be_pinned():
    """A ``position`` with an aligned/absolute axis is still a pose; the panel
    draws it at one value of that axis."""
    dims = pose_dims(_state(_dataset(["x", "y", "z"], {"position_type": ["aligned", "absolute"]})).data_loader)
    assert dims.extra == (("position_type", ("aligned", "absolute")),)


@pytest.fixture(params=[None, {"position_type": ["aligned", "absolute"]}], ids=["plain", "extra_dim"])
def panel(request, qtbot, tmp_path, monkeypatch):
    from ethograph.gui.app_state import ObservableAppState
    from ethograph.gui.plots_skeleton import SkeletonPlot

    ds = _dataset(["x", "y", "z"], request.param)
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")  # never touch the real settings
    state.pose_hide_threshold = 0.0
    state.data_loader = _state(ds).data_loader
    monkeypatch.setattr(ObservableAppState, "window_bounds", property(lambda _self: TimeRange(0.0, 1.0)))
    plot = SkeletonPlot(shell=None, app_state=state)
    qtbot.addWidget(plot)
    plot.set_store(state.data_loader)
    return plot, ds


def _pinned(ds: xr.Dataset, plot) -> xr.DataArray:
    """``position`` at the panel's extra-dim choices (its first values by default)."""
    return ds["position"].sel(plot.extra_selections())


def test_the_panel_indexes_the_cached_window_by_time(panel):
    plot, ds = panel
    plot.cb_all_individuals.setChecked(True)
    plot.set_time(0.5)
    frame = plot.current_frame()
    assert frame is not None
    expected = _pinned(ds, plot).sel(time=0.5, method="nearest").transpose("individual", "keypoint", "space").values
    assert np.allclose(frame.points, expected.reshape(-1, 3))


def test_the_individual_combo_selects_one_animal(panel):
    plot, ds = panel
    plot.individual_combo.setCurrentText("b")
    plot.set_time(0.0)
    frame = plot.current_frame()
    expected = _pinned(ds, plot).sel(time=0.0, individual="b").transpose("keypoint", "space").values
    assert np.allclose(frame.points, expected)
    assert plot.pinned_individual == "b", "picking an animal in the combo pins the panel"


def test_an_extra_dim_combo_picks_which_pose_is_drawn(panel):
    plot, ds = panel
    if not plot._extra_combos:
        pytest.skip("this pose has no extra dim")
    combo = plot._extra_combos["position_type"]
    plot.set_time(0.0)
    aligned = plot.current_frame().points.copy()
    combo.setCurrentText("absolute")
    absolute = plot.current_frame().points
    expected = ds["position"].sel(time=0.0, position_type="absolute", individual=plot.individual_combo.currentText())
    assert np.allclose(absolute, expected.transpose("keypoint", "space").values)
    assert not np.allclose(aligned, absolute)
    assert plot.skeleton_settings()["dims"] == {"position_type": "absolute"}


def test_settings_round_trip(panel):
    plot, _ = panel
    plot.set_hidden_keypoints({"tail"})
    plot.cb_flip_y.setChecked(True)
    plot.color_by_combo.setCurrentIndex(plot.color_by_combo.findData(COLOR_BY_INDIVIDUAL))
    settings = plot.skeleton_settings()
    assert settings["hidden_keypoints"] == ["tail"]
    assert settings["flip_y"] is True
    assert settings["color_by"] == COLOR_BY_INDIVIDUAL

    plot.set_hidden_keypoints(set())
    plot.cb_flip_y.setChecked(False)
    plot.apply_skeleton_settings(settings)
    assert plot.hidden_keypoints() == {"tail"}
    assert plot.cb_flip_y.isChecked()
    assert plot.color_by() == COLOR_BY_INDIVIDUAL
