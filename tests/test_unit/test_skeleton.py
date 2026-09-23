"""Taken from https://github.com/neuroinformatics-unit/movement/pull/763"""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from ethograph.skeleton.config import (
    config_to_arrays,
    connections_to_edge_indices,
    hex_to_rgba,
    load_yaml_config,
    rgba_to_hex,
    save_yaml_config,
    validate_config,
)
from ethograph.skeleton.renderers import PrecomputedRenderer


def _make_pose_dataset(n_frames: int, n_keypoints: int, spatial_dims: int) -> xr.Dataset:
    """A movement-style pose dataset, positions centred around 100."""
    rng = np.random.default_rng(0)
    space = ["x", "y", "z"][:spatial_dims]
    position = 100.0 + rng.normal(scale=10.0, size=(n_frames, spatial_dims, n_keypoints, 1))
    confidence = rng.uniform(size=(n_frames, n_keypoints, 1))
    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(position, dims=["time", "space", "keypoints", "individuals"]),
            "confidence": xr.DataArray(confidence, dims=["time", "keypoints", "individuals"]),
        },
        coords={
            "time": np.arange(n_frames),
            "space": space,
            "keypoints": [f"kp_{i}" for i in range(n_keypoints)],
            "individuals": ["ind_0"],
        },
        attrs={"ds_type": "poses"},
    )


@pytest.fixture
def synthetic_skeleton_dataset() -> Callable[..., xr.Dataset]:
    def make(n_frames: int = 10, n_keypoints: int = 3, spatial_dims: int = 2) -> xr.Dataset:
        return _make_pose_dataset(n_frames, n_keypoints, spatial_dims)

    return make


@pytest.fixture
def skeleton_dataset_with_nans() -> xr.Dataset:
    ds = _make_pose_dataset(n_frames=10, n_keypoints=3, spatial_dims=2)
    ds["position"][0, :, 0, 0] = np.nan
    ds["position"][5, :, :, 0] = np.nan
    return ds


@pytest.fixture
def simple_skeleton_config() -> dict[str, Any]:
    return {
        "keypoints": ["kp_0", "kp_1", "kp_2"],
        "connections": [
            {"start": "kp_0", "end": "kp_1", "color": "#FF0000", "width": 2.0, "segment": "side1"},
            {"start": "kp_1", "end": "kp_2", "color": "#00FF00", "width": 2.0, "segment": "side2"},
            {"start": "kp_2", "end": "kp_0", "color": "#0000FF", "width": 2.0, "segment": "side3"},
        ],
    }


def test_precomputed_renderer_init(synthetic_skeleton_dataset, simple_skeleton_config):
    """Test PrecomputedRenderer initialization."""
    ds = synthetic_skeleton_dataset(n_frames=50, n_keypoints=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    assert renderer.name == "Precomputed"
    assert renderer.supports_3d is True
    assert renderer.requires_gpu is False
    assert renderer.vectors is None  # Not prepared yet


def test_precomputed_renderer_2d_vector_format(synthetic_skeleton_dataset, simple_skeleton_config):
    """Test that 2D vectors have correct napari format (N, 2, 3)."""
    # Create 2D dataset
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3, spatial_dims=2)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    # Prepare (compute vectors)
    renderer.prepare()

    # Check vector format
    assert renderer.vectors is not None
    assert renderer.vectors.ndim == 3
    assert renderer.vectors.shape[1] == 2  # [start, direction]
    assert renderer.vectors.shape[2] == 3  # [t, y, x] for 2D+time

    # Check that we have reasonable number of vectors
    # 10 frames * 1 individual * 3 connections = 30 max vectors
    assert len(renderer.vectors) > 0
    assert len(renderer.vectors) <= 30


def test_precomputed_renderer_3d_vector_format(synthetic_skeleton_dataset, simple_skeleton_config):
    """Test that 3D vectors have correct napari format (N, 2, 4)."""
    # Create 3D dataset
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3, spatial_dims=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    # Check vector format
    assert renderer.vectors.shape[1] == 2  # [start, direction]
    assert renderer.vectors.shape[2] == 4  # [t, z, y, x] for 3D+time


def test_precomputed_renderer_coordinate_order(synthetic_skeleton_dataset, simple_skeleton_config):
    """Test that coordinates are in correct napari order [t, y, x]."""
    ds = synthetic_skeleton_dataset(n_frames=5, n_keypoints=3, spatial_dims=2)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    # Take first vector
    if len(renderer.vectors) > 0:
        first_vector = renderer.vectors[0]
        start_pos = first_vector[0]

        # Check that time component is integer (frame index)
        assert start_pos[0] == int(start_pos[0])
        assert 0 <= start_pos[0] < 5  # Within frame range

        # Check that spatial coordinates are reasonable
        # (Our synthetic data is centered around 100)
        assert 0 < start_pos[1] < 200  # y coordinate
        assert 0 < start_pos[2] < 200  # x coordinate


def test_precomputed_renderer_direction_vector(synthetic_skeleton_dataset, simple_skeleton_config):
    """Test that direction vectors are computed correctly."""
    ds = synthetic_skeleton_dataset(n_frames=5, n_keypoints=3, spatial_dims=2)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    if len(renderer.vectors) > 0:
        first_vector = renderer.vectors[0]
        start_pos = first_vector[0]
        direction = first_vector[1]

        # Direction should have same dimensions as start
        assert direction.shape == start_pos.shape

        # Time component of direction should be 0
        # (vectors don't span across time)
        assert direction[0] == 0


def test_precomputed_renderer_nan_handling(skeleton_dataset_with_nans, simple_skeleton_config):
    """Test that NaN keypoints are skipped correctly."""
    ds = skeleton_dataset_with_nans
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    # Verify no NaN values in output vectors
    assert not np.any(np.isnan(renderer.vectors))

    # Verify we have fewer vectors than total possible
    # (because some were skipped due to NaN)
    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individuals"]
    n_connections = len(connections)
    max_vectors = n_frames * n_individuals * n_connections

    assert len(renderer.vectors) < max_vectors


def test_precomputed_renderer_cleanup(synthetic_skeleton_dataset, simple_skeleton_config):
    """Test that cleanup properly releases memory."""
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()
    assert renderer.vectors is not None

    renderer.cleanup()
    assert renderer.vectors is None


def test_precomputed_renderer_estimate_memory(synthetic_skeleton_dataset, simple_skeleton_config):
    """Test memory estimation is reasonable."""
    ds = synthetic_skeleton_dataset(n_frames=100, n_keypoints=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    estimated_mb = renderer.estimate_memory()

    # Should be a positive number
    assert estimated_mb > 0

    # Should be reasonable (not gigabytes for small dataset)
    assert estimated_mb < 100


def test_precomputed_renderer_get_info(synthetic_skeleton_dataset, simple_skeleton_config):
    """Test get_info method returns correct information."""
    ds = synthetic_skeleton_dataset(n_frames=20, n_keypoints=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()
    info = renderer.get_info()

    assert info["name"] == "Precomputed"
    assert info["n_frames"] == 20
    assert info["n_connections"] == 3
    assert "n_precomputed_vectors" in info
    assert "actual_memory_mb" in info


def test_precomputed_renderer_no_valid_vectors(simple_skeleton_config):
    """Test handling when all keypoints are NaN."""

    # Create dataset with all NaN positions
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                np.full((5, 2, 3, 1), np.nan),
                dims=["time", "space", "keypoints", "individuals"],
            ),
            "confidence": xr.DataArray(
                np.zeros((5, 3, 1)),
                dims=["time", "keypoints", "individuals"],
            ),
        },
        coords={
            "time": np.arange(5),
            "space": ["x", "y"],
            "keypoints": ["kp_0", "kp_1", "kp_2"],
            "individuals": ["ind_0"],
        },
        attrs={"ds_type": "poses"},
    )

    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, _ = config_to_arrays(simple_skeleton_config, keypoint_names)

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    # Should return empty array with correct shape
    assert len(renderer.vectors) == 0
    assert renderer.vectors.shape == (0, 2, 3)


def test_hex_to_rgba():
    """Test hex color to RGBA conversion."""
    # Test with hash
    rgba = hex_to_rgba("#FF0000")
    assert rgba == (1.0, 0.0, 0.0, 1.0)

    # Test without hash
    rgba = hex_to_rgba("00FF00")
    assert rgba == (0.0, 1.0, 0.0, 1.0)

    # Test with alpha
    rgba = hex_to_rgba("#0000FF", alpha=0.5)
    assert rgba == (0.0, 0.0, 1.0, 0.5)


def test_hex_rgba_roundtrip():
    """Test that hex->rgba->hex conversion is lossless."""
    original = "#AABBCC"
    rgba = hex_to_rgba(original)
    result = rgba_to_hex(rgba)
    assert result == original


def test_connections_to_edge_indices(simple_skeleton_config):
    """Test conversion of connection names to indices."""
    keypoint_names = simple_skeleton_config["keypoints"]
    connections = simple_skeleton_config["connections"]

    edges = connections_to_edge_indices(connections, keypoint_names)

    assert len(edges) == 3
    assert edges[0] == (0, 1)  # kp_0 -> kp_1
    assert edges[1] == (1, 2)  # kp_1 -> kp_2
    assert edges[2] == (2, 0)  # kp_2 -> kp_0


def test_connections_to_edge_indices_invalid_keypoint():
    """Test that invalid keypoint names raise ValueError."""
    connections = [{"start": "invalid", "end": "kp_1"}]
    keypoint_names = ["kp_0", "kp_1", "kp_2"]

    with pytest.raises(ValueError, match="not found"):
        connections_to_edge_indices(connections, keypoint_names)


def test_config_to_arrays(simple_skeleton_config):
    """Test conversion of config dict to renderer arrays."""
    keypoint_names = simple_skeleton_config["keypoints"]

    connections, colors, widths, labels = config_to_arrays(simple_skeleton_config, keypoint_names)

    # Check connections
    assert len(connections) == 3
    assert connections[0] == (0, 1)

    # Check colors
    assert colors.shape == (3, 4)
    assert np.allclose(colors[0], (1.0, 0.0, 0.0, 1.0))  # Red

    # Check widths
    assert widths.shape == (3,)
    assert widths[0] == pytest.approx(2.0)

    # Check labels
    assert len(labels) == 3
    assert labels[0] == "side1"


def test_save_and_load_yaml_config(simple_skeleton_config):
    """Test saving and loading YAML configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"

        # Save config
        save_yaml_config(simple_skeleton_config, config_path)
        assert config_path.exists()

        # Load config
        loaded_config = load_yaml_config(config_path)

        # Verify content
        assert loaded_config["keypoints"] == simple_skeleton_config["keypoints"]
        assert len(loaded_config["connections"]) == len(simple_skeleton_config["connections"])


def test_load_yaml_config_nonexistent_file():
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("/nonexistent/path/config.yaml")


def test_validate_config_valid(simple_skeleton_config, synthetic_skeleton_dataset):
    """Test validation of a valid configuration."""
    ds = synthetic_skeleton_dataset(n_keypoints=3)

    is_valid, errors = validate_config(simple_skeleton_config, ds)

    assert is_valid
    assert len(errors) == 0


def test_validate_config_missing_connections(synthetic_skeleton_dataset):
    """Test validation fails when connections key is missing."""
    ds = synthetic_skeleton_dataset(n_keypoints=3)
    invalid_config = {"keypoints": ["kp_0", "kp_1"]}

    is_valid, errors = validate_config(invalid_config, ds)

    assert not is_valid
    assert any("connections" in err for err in errors)


def test_validate_config_invalid_keypoint(simple_skeleton_config, synthetic_skeleton_dataset):
    """Test validation fails when connection references invalid keypoint."""
    ds = synthetic_skeleton_dataset(n_keypoints=2)  # Only 2 keypoints

    # Config has connections to kp_2 which doesn't exist
    is_valid, errors = validate_config(simple_skeleton_config, ds)

    assert not is_valid
    assert any("not found" in err for err in errors)


def test_validate_config_no_keypoints_in_dataset(simple_skeleton_config):
    """Test validation fails when dataset has no keypoints coordinate."""

    # Create dataset without keypoints
    ds = xr.Dataset(attrs={"ds_type": "poses"})

    is_valid, errors = validate_config(simple_skeleton_config, ds)

    assert not is_valid
    assert any("keypoints" in err.lower() for err in errors)
