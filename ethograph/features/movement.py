"""Features related to movements/kinematics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.distance import cdist


class Position3DCalibration:
    """Convert 3D positions from DLC to real-world coordinates."""

    def __init__(
        self,
        conv_factor: float = -2.60976 + 0.0937,
        offset_x: float = 2.7,
        offset_y: float = -6.0415 + 13.3,
        offset_z: float = -45.7488,
        rot_x: float = 15,
        rot_y: float = -3.6802 + 9,
        rot_z: float = -7.47301 + 1.206338,
    ):
        self.conv_factor = conv_factor
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z

    def _rotate_x(self, data: np.ndarray, theta: float) -> np.ndarray:
        theta = np.radians(theta)
        rot_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        return data @ rot_mat

    def _rotate_y(self, data: np.ndarray, theta: float) -> np.ndarray:
        theta = np.radians(theta)
        rot_mat = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        return data @ rot_mat

    def _rotate_z(self, data: np.ndarray, theta: float) -> np.ndarray:
        theta = np.radians(theta)
        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return data @ rot_mat

    def transform(self, ds):

        pos_data = ds.position.values.copy()

        space_dim = ds.position.dims.index("space")
        # Move space dimension to last position for easier indexing
        pos_data = np.moveaxis(pos_data, space_dim, -1)

        # Verify we have 3D coordinates
        if pos_data.shape[-1] != 3:
            raise ValueError(f"Expected 3 coordinates (x,y,z), got {pos_data.shape[-1]}")

        # Invert x-axis
        pos_data[..., 0] = -pos_data[..., 0]

        # pos_data[..., 2] = -pos_data[..., 2] # z

        # Convert units to cm
        pos_data *= self.conv_factor

        pos_data[..., 0] -= self.offset_x
        pos_data[..., 1] -= self.offset_y
        pos_data[..., 2] -= self.offset_z

        # Apply rotations - reshape to (N, 3) for matrix multiplication
        original_shape = pos_data.shape
        pos_data_flat = pos_data.reshape(-1, 3)

        pos_data_flat = self._rotate_x(pos_data_flat, self.rot_x)
        pos_data_flat = self._rotate_y(pos_data_flat, self.rot_y)
        pos_data_flat = self._rotate_z(pos_data_flat, self.rot_z)

        pos_data = pos_data_flat.reshape(original_shape)

        # Move space dimension back to original position
        pos_data = np.moveaxis(pos_data, -1, space_dim)

        ds.position.values = pos_data

        return ds


def compute_distance_to_constant(
    data: xr.Dataset,
    reference_point: np.ndarray | list | tuple,
    keypoint: str = None,
    individual: str = None,
    metric: str = "euclidean",
    **kwargs,
) -> xr.DataArray:
    """
    Compute distance from keypoint(s) to a constant reference point.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing position data with dims: time, individual, keypoint, space.
        Space dimension must contain either ``['x', 'y']`` or ``['x', 'y', 'z']``.
    reference_point : array-like
        Constant reference point [x, y] for 2D or [x, y, z] for 3D
    keypoint : str, optional
        Specific keypoint to compute distance for
    individual : str, optional
        Specific individual to compute distance for
    metric : str, default "euclidean"
        Distance metric (see scipy.spatial.distance.cdist)
    **kwargs
        Additional arguments for cdist

    Returns
    -------
    xarray.DataArray
        Distances with preserved dimensions.

    Raises
    ------
    ValueError
        If space coordinates and reference_point dimensions don't match
    """
    if keypoint:
        data = data.sel(keypoint=keypoint)
    if individual:
        data = data.sel(individual=individual)

    if isinstance(reference_point, (list, tuple)):
        reference_point = np.array(reference_point)

    space_coords = list(data.coords["space"].values)
    ref_len = len(reference_point)

    if space_coords == ["x", "y"] and ref_len == 2:
        space_selection = ["x", "y"]
    elif space_coords == ["x", "y", "z"] and ref_len == 3:
        space_selection = ["x", "y", "z"]
    else:
        raise ValueError(
            f"Dimension mismatch: space coords {space_coords} "
            f"incompatible with reference_point length {ref_len}. "
            f"Expected either ['x', 'y'] with len=2 or ['x', 'y', 'z'] with len=3"
        )

    distances = xr.apply_ufunc(
        lambda pos: cdist(
            pos.reshape(-1, ref_len),
            reference_point.reshape(1, -1),
            metric=metric,
            **kwargs,
        ).squeeze(),
        data.sel(space=space_selection),
        input_core_dims=[["space"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )

    return distances


def move_vector_to_rgb(data: np.ndarray) -> np.ndarray:
    """
    Source: https://www.s3dlib.org/examples/vectors/rgb_normals.html

    Map (N,3) direction vectors (e.g. velocity) to RGB using the surface-normal convention.

    Same as s3dlib's rgbColorXYZ:
        R = (sqrt(3)*x + 1) / 2
        G = (sqrt(3)*y + 1) / 2
        B = (sqrt(3)*z + 1) / 2

    Vectors are normalized to unit length first.
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit = data / norms

    c = np.sqrt(3)
    rgb = (c * unit + 1) / 2.0
    return np.clip(rgb, 0, 1)


def calculate_movement_angles(data, input_type="position"):
    """
    Calculate movement angles in the x-y plane (azimuth), and optionally
    elevation if z-data is available.

    Converts Cartesian displacement/velocity vectors to polar angle coordinates
    using arctan2. See: https://movement.neuroinformatics.dev/latest/examples/compute_head_direction.html#head-to-snout-vector-in-polar-coordinates

    Parameters
    ----------
    data : array-like or xarray.DataArray
        If input_type="position": array of shape (N, D) where D >= 2.
            Uses first two columns as x, y (and optionally third as z).
            Only plain arrays supported.
        If input_type="velocity": array of shape (N, 2+) or xarray.DataArray
            with a "space" dimension containing "x", "y" (and optionally "z").
            Velocity can be computed via
            ``ds.move.compute_velocity()``, which uses np.gradient internally.

    input_type : str, {"position", "velocity"}
        Whether ``data`` contains positions or velocities.

    Returns
    -------
    azimuth : ndarray of shape (N,)
        Angles in degrees from positive y-axis, measured counterclockwise.
    elevation : ndarray of shape (N,) or None
        Angle in degrees from the x-y plane. Only returned (non-None) when
        z-data is available. Positive = above the plane, negative = below.

    Notes
    -----
    Both input types use ``np.gradient`` (central differences in the interior,
    forward/backward at edges) and return N values aligned with the original
    timestamps. Since velocity is just displacement scaled by 1/dt, the
    direction (angle) is the same for both approaches.
    """
    if input_type == "position":
        data = np.asarray(data)
        dx = np.gradient(data[:, 0])
        dy = np.gradient(data[:, 1])
        dz = np.gradient(data[:, 2]) if data.shape[1] > 2 else None
    elif input_type == "velocity":
        if hasattr(data, "sel"):  # xarray DataArray
            dx = data.sel(space="x").values
            dy = data.sel(space="y").values
            dz = data.sel(space="z").values if "z" in data.space.values else None
        else:
            data = np.asarray(data)
            dx, dy = data[:, 0], data[:, 1]
            dz = data[:, 2] if data.shape[1] > 2 else None
    else:
        raise ValueError(f"input_type must be 'position' or 'velocity', got '{input_type}'")

    azimuth = np.degrees(np.arctan2(dx, dy))

    if dz is not None:
        elevation = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))
    else:
        elevation = None

    return azimuth, elevation


def get_angle_rgb(xy_pos, smooth_func=None, smoothing_params=None, input_type="position"):
    """
    Convert movement angles to RGB colors using a colormap, with optional smoothing.

    Parameters
    ----------
    xy_pos : array-like of shape (N, D)
        N points with D dimensions. Uses first two columns as x, y coordinates.
    smooth_func : callable, optional
        Function to smooth xy_pos before angle calculation. Should accept and return array-like of shape (N, D).
    smooth_params: dict, optional
        Parameters to pass to the smoothing function.

    Returns
    -------
    rgb_matrix : ndarray of shape (N, 3)
        RGB values corresponding to each input point.
    angles : ndarray of shape (N,)
        Angles in degrees (0-360°).
    """
    xy_pos = np.asarray(xy_pos)

    if np.all(np.isnan(xy_pos)):
        nan_rgb = np.full((xy_pos.shape[0], 3), np.nan)
        nan_angles = np.full(xy_pos.shape[0], np.nan)
        return nan_rgb, nan_angles

    if smooth_func is not None:
        xy_pos = smooth_func(xy_pos, **smoothing_params)

    cmap = matplotlib.colormaps["hsv"]
    cm = cmap(np.linspace(0, 1, 256))[:, :3]  # Get RGB, exclude alpha

    curr_angles, _ = calculate_movement_angles(xy_pos, input_type)

    assert np.all((curr_angles[~np.isnan(curr_angles)] >= -180) & (curr_angles[~np.isnan(curr_angles)] <= 180)), (
        f"Angles out of expected range [-180, 180]: min={np.nanmin(curr_angles):.2f}, max={np.nanmax(curr_angles):.2f}"
    )

    col_lines = np.linspace(0, len(cm) - 1, 360)

    angles = np.floor(curr_angles + 180).clip(0, 359)
    valid = ~np.isnan(angles)

    cmap_indices = np.full(len(angles), np.nan)
    cmap_indices[valid] = np.round(col_lines[angles[valid].astype(int)]).astype(int)

    rgb_matrix = np.full((len(angles), 3), np.nan)
    rgb_matrix[valid] = cm[cmap_indices[valid].astype(int)]

    return rgb_matrix, angles


#: Frames masked on either side of a keyframe: the frames right after a fresh
#: I-frame still settle onto it (measured on x264 output).
KEYFRAME_HALO = 3


def gop_relative_sizes(sizes: np.ndarray, keyframes: np.ndarray) -> np.ndarray:
    """Packet size divided by the typical size at the same offset within its GOP.

    An encoder spends bytes by frame *role* before it spends them on motion:
    the I-frame, then the reference frames of its B-pyramid, then the rest —
    a pattern that repeats every GOP and dwarfs the motion signal. The
    median size per offset, pooled over every GOP, is that pattern; the
    ratio to it is what the frame cost beyond its role, i.e. the change.
    """
    sizes = np.asarray(sizes, dtype=np.float32)
    if len(keyframes) < 2 or len(sizes) == 0:
        return sizes / (np.median(sizes) or 1.0)
    starts = np.asarray(keyframes, dtype=int)
    gop_index = np.searchsorted(starts, np.arange(len(sizes)), side="right") - 1
    gop_index = np.clip(gop_index, 0, len(starts) - 1)
    offsets = np.arange(len(sizes)) - starts[gop_index]
    profile = np.full(int(offsets.max()) + 1, np.nan, dtype=np.float32)
    for j in np.unique(offsets):
        members = sizes[offsets == j]
        if len(members) >= 2:
            profile[j] = np.median(members)
    fallback = float(np.nanmedian(profile)) if np.isfinite(profile).any() else float(np.median(sizes)) or 1.0
    profile = np.where(np.isfinite(profile) & (profile > 0), profile, fallback)
    return sizes / profile[offsets]


def _packets_in_display_order(video_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """``(sizes, is_keyframe)`` per frame in display order, from the container alone.

    Packets come out of the demuxer in decode order; with B-frames that is not
    the order frames are shown in, so a keyframe's packet index is not its frame
    index. Sorting by presentation timestamp puts every packet at the frame it
    belongs to.
    """
    import av

    rows: list[tuple[int, int, bool]] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, packet in enumerate(container.demux(stream)):
            if packet.size == 0:
                continue
            pts = packet.pts if packet.pts is not None else (packet.dts if packet.dts is not None else i)
            rows.append((int(pts), int(packet.size), bool(packet.is_keyframe)))
    rows.sort(key=lambda r: r[0])
    sizes = np.asarray([r[1] for r in rows], dtype=np.float32)
    keys = np.asarray([r[2] for r in rows], dtype=bool)
    return sizes, keys


def mask_keyframes(motion: np.ndarray, keyframes: np.ndarray, halo: int = KEYFRAME_HALO) -> np.ndarray:
    """Replace the values at keyframes (± *halo* frames) by interpolation from their neighbours.

    A keyframe is a whole picture, encoded afresh: its packet is large whatever
    moved, and the frames right after it settle onto the new picture. The
    spikes are periodic (the encoder's GOP) and carry no information about the
    animal.
    """
    motion = np.asarray(motion, dtype=np.float32).copy()
    if len(keyframes) == 0 or len(motion) == 0:
        return motion
    bad = np.zeros(len(motion), dtype=bool)
    for k in keyframes:
        bad[max(0, k - halo) : min(len(motion), k + halo + 1)] = True
    good = ~bad
    if good.sum() < 2:
        return motion
    idx = np.arange(len(motion))
    motion[bad] = np.interp(idx[bad], idx[good], motion[good])
    return motion


def extract_packet_motion(video_path: Path | str, fps: float, time_coord_name: str = "time") -> xr.DataArray:
    """Motion from the compressed stream alone: bytes per frame, no decoding.

    A predicted frame costs the encoder bytes in proportion to what changed
    since the previous one, so packet size tracks motion at a fraction of the
    cost of decoding — the whole file is scanned in well under a second.
    Two corrections, both from the container: sizes are taken relative to
    their GOP offset (:func:`gop_relative_sizes`) and keyframes are masked
    (:func:`mask_keyframes`). The scale is the encoder's, so it is only
    comparable within one file; callers normalise per video.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if fps <= 0:
        raise ValueError("fps must be positive — read it from the video, do not default it.")
    sizes, keys = _packets_in_display_order(video_path)
    keyframes = np.flatnonzero(keys)
    motion = mask_keyframes(gop_relative_sizes(sizes, keyframes), keyframes)
    return xr.DataArray(motion, dims=[time_coord_name], coords={time_coord_name: np.arange(len(motion)) / fps})


def compute_aux_velocity_and_speed(
    a_aux_trial: np.ndarray,
    time_intan: np.ndarray,
    fps: float = 30000.0,
    mov_mean_window1: int = 6001,
    mov_mean_window2: int = 15001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute drift-corrected velocity and scalar speed from accelerometer data.

    Removes slow drift with rolling-mean baselines, integrates acceleration
    to velocity via cumulative trapezoidal integration (matches MATLAB
    ``cumtrapz``), and returns the L2 norm of velocity as scalar speed.

    Parameters
    ----------
    a_aux_trial : numpy.ndarray
        Accelerometer array of shape ``(N, D)`` — N time samples, D axes.
    time_intan : numpy.ndarray
        Timestamps in seconds, shape ``(N,)``.
    fps : float
        Sampling rate of the recording in Hz.
    mov_mean_window1 : int
        Rolling-mean window (samples) for drift removal from acceleration.
    mov_mean_window2 : int
        Rolling-mean window (samples) for drift removal from velocity.

    Returns
    -------
    a_corr : numpy.ndarray
        Drift-corrected acceleration, shape ``(N, D)``.
    v_corr : numpy.ndarray
        Drift-corrected velocity, shape ``(N, D)``.
    speed : numpy.ndarray
        L2 norm of ``v_corr`` — scalar speed at each time point, shape ``(N,)``.
    """
    if a_aux_trial.shape[0] != len(time_intan):
        raise ValueError(
            f"Shape mismatch: a_aux_trial has {a_aux_trial.shape[0]} samples "
            f"but time_intan has {len(time_intan)} samples"
        )

    dt = 1 / fps

    # MATLAB-style outlier detection (median + scaled MAD)
    med = np.median(a_aux_trial, axis=0)
    mad = np.median(np.abs(a_aux_trial - med), axis=0)
    threshold = 3 * mad * 1.4826
    outlier_mask = np.abs(a_aux_trial - med) > threshold

    a_good = a_aux_trial.copy()
    a_good[outlier_mask] = np.nan

    drift = pd.DataFrame(a_good).rolling(mov_mean_window1, center=True, min_periods=1).mean().values

    drift_interp = np.empty_like(a_aux_trial)
    for col in range(a_aux_trial.shape[1]):
        valid_idx = ~np.isnan(a_good[:, col])
        f = interpolate.interp1d(
            time_intan[valid_idx],
            drift[valid_idx, col],
            kind="linear",
            fill_value="extrapolate",
        )
        drift_interp[:, col] = f(time_intan)

    a_aux_trial_driftcorr = a_aux_trial - drift_interp

    # Trapezoidal integration (matches MATLAB cumtrapz)
    v_aux_trial = cumulative_trapezoid(a_aux_trial_driftcorr, dx=dt, axis=0, initial=0)

    drift2 = pd.DataFrame(v_aux_trial).rolling(mov_mean_window2, center=True, min_periods=1).mean().values

    v_aux_trial_driftcorr = v_aux_trial - drift2
    v_aux_speed = np.linalg.norm(v_aux_trial_driftcorr, axis=1)

    return a_aux_trial_driftcorr, v_aux_trial_driftcorr, v_aux_speed
