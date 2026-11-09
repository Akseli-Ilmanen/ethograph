"""Features related to movements/kinematics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation, uniform_filter1d
from scipy.spatial.distance import cdist


class Position3DCalibration:
    """Convert 3D positions from DLC to real-world coordinates.

    Hand-tuned for the Moll et al., 2025 rig and kept for that dataset's
    example. A new rig gets a world frame beside its calibration instead
    (``ethograph.triangulate.frame``).
    """

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
        N time points with D dimensions, time first. Uses first two columns as x, y coordinates.
    smooth_func : callable, optional
        Function to smooth xy_pos before angle calculation. Called with ``axis=0`` (time),
        so it should accept an ``axis`` keyword and return array-like of shape (N, D).
    smoothing_params: dict, optional
        Parameters to pass to the smoothing function, without ``axis``.

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
        xy_pos = smooth_func(xy_pos, axis=0, **smoothing_params)

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


# --- Periodic movement -------------------------------------------------------
#
# NOTE: incomplete and provisional. These are first-pass scores for fast
# periodic movements sampled near the Nyquist rate of the video (tail
# quivering in birds at ~24 Hz on a 47 fps camera; rodent sniffing/whisking
# is the next candidate). More features — a proper spectral estimate, a period
# estimate, per-keypoint variants — are expected to join them in the near
# future; treat the API as unstable.
#
# Every window is given in *frames*; callers convert from seconds with the
# video's own rate (never a hardcoded fps): ``window = round(0.3 * fps)``.


def _moving_mean(a: np.ndarray, window: int) -> np.ndarray:
    return uniform_filter1d(a, window, mode="nearest")


def _fill_nans(x: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), np.nanmedian(x), x)


def periodic_valid_mask(x: np.ndarray, window: int) -> np.ndarray:
    """Samples whose windowed statistic is not contaminated by a gap or the array edge.

    Any NaN (and the two ends of the array) is dilated by half a window so
    that no returned value was computed over filled-in data.
    """
    x = np.asarray(x, dtype=float)
    invalid = np.isnan(x)
    invalid[[0, -1]] = True
    return ~binary_dilation(invalid, iterations=window // 2 + 1)


def alternation(x: np.ndarray, window: int) -> np.ndarray:
    """Lag-1 anticorrelation of the detrended signal, clipped to ``[0, 1]``.

    A signal that reverses direction every frame (a movement at about half the
    sampling rate) has a lag-1 autocorrelation near -1; smooth movement and
    white noise sit at or above 0. The score is ``clip(-r1, 0, 1)`` with
    ``r1`` estimated over a moving *window*. NaN where the window is not
    valid (:func:`periodic_valid_mask`).
    """
    x = np.asarray(x, dtype=float)
    filled = _fill_nans(x)
    centred = filled - _moving_mean(filled, window)
    covariance = _moving_mean(centred[1:] * centred[:-1], window)
    variance = _moving_mean(centred**2, window)[1:]
    r1 = np.pad(covariance / np.where(variance > 1e-6, variance, np.nan), (1, 0), constant_values=np.nan)
    return np.where(periodic_valid_mask(x, window), np.clip(-r1, 0, 1), np.nan)


def flip_rate(x: np.ndarray, window: int) -> np.ndarray:
    """Fraction of frames at which the direction of change reverses, over a moving *window*.

    1.0 means a reversal on every frame; smooth movement reads near 0. The
    first two samples are NaN (two differences are needed for one flip).
    """
    x = np.asarray(x, dtype=float)
    step = np.diff(x)
    flips = np.sign(step[1:]) * np.sign(step[:-1]) < 0
    rate = _moving_mean(flips.astype(float), window)
    return np.pad(rate, (2, 0), constant_values=np.nan)


def high_band_ratio(x: np.ndarray, window: int, high_pass_frames: int = 3) -> np.ndarray:
    """Share of the windowed variance that lives above a *high_pass_frames* boxcar.

    Near 1 when all the movement is frame-to-frame jitter, near 0 when it is
    a slow sweep. NaN where the window's total variance is negligible.
    """
    x = np.asarray(x, dtype=float)
    high_pass = x - _moving_mean(x, high_pass_frames)
    high_var = _moving_mean(high_pass**2, window)
    total_var = _moving_mean((x - _moving_mean(x, window)) ** 2, window)
    return high_var / np.where(total_var > 1e-6, total_var, np.nan)


def jitter_features(x: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """``flip_rate``, ``high_band_ratio`` and their product, each NaN outside :func:`periodic_valid_mask`.

    The product is the "jitter score": high only when the signal both
    reverses often *and* most of its energy is at the frame scale.
    """
    x = np.asarray(x, dtype=float)
    valid = periodic_valid_mask(x, window)
    filled = _fill_nans(x)
    rate = np.where(valid, flip_rate(filled, window), np.nan)
    ratio = np.where(valid, high_band_ratio(filled, window), np.nan)
    return {"flip_rate": rate, "high_band_ratio": ratio, "jitter_score": rate * ratio}


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
