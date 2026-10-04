"""What the spike raster draws for one view: a tick per spike, or spike counts per pixel.

Qt-free. Ticks show every spike exactly and stop being readable once they
overlap; the density image shows how many spikes fell in each pixel-sized
cell, so it stays truthful however crowded the view is. ``choose_render``
picks between them from how crowded the view is, never from the unit count.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

RENDER_MODES: dict[str, str] = {
    "auto": "Auto",
    "ticks": "Ticks",
    "density": "Density",
}

#: What a raster row is and how rows are ordered: a probe channel by depth, or
#: one row per unit in the cluster table's order or in a fitted Rastermap order.
ROW_ORDERS: dict[str, str] = {
    "depth": "Probe depth",
    "table": "Cluster table",
    "rastermap": "Rastermap",
}

#: Spikes per pixel cell above which Auto switches to the density image, and
#: the lower value it must fall under to switch back — two thresholds so a
#: view near the boundary does not flicker while zooming.
DENSITY_ON = 0.5
DENSITY_OFF = 0.25

#: Spikes in view above which ticks are refused even when asked for: painting
#: them would stall the GUI, the image costs the same however many there are.
MAX_TICKS = 500_000

#: Share of a row's height a tick spans.
TICK_ROW_FRACTION = 0.8

#: Tick width in pixels that means "pick it from the view" (``auto_tick_width``).
TICK_WIDTH_AUTO = 0
MAX_TICK_WIDTH = 10
#: The widest an automatic tick gets, and the share of the mean gap between
#: neighbouring spikes of a row it may fill before ticks start to touch.
_AUTO_TICK_WIDTH_MAX = 3
_AUTO_TICK_GAP_FRACTION = 0.25

#: Counts are shown on a log scale; a cell holding a single spike still gets
#: this much of the full brightness, so sparse spikes stay visible.
_MIN_BRIGHTNESS = 0.35
_BRIGHTEST_PERCENTILE = 99.0

#: One colour group of spikes: times (sorted), the y of each spike, and an RGB(A) colour.
SpikeGroup = tuple[NDArray, NDArray, tuple]


def choose_render(mode: str, current: str, n_spikes: int, n_cells: int) -> str:
    """``"ticks"`` or ``"density"`` for a view holding ``n_spikes`` in ``n_cells`` pixel cells.

    ``mode`` is the user's setting (a ``RENDER_MODES`` key) and ``current`` is
    what is drawn now, which decides the side of the hysteresis band.
    """
    if mode not in RENDER_MODES:
        raise ValueError(f"mode must be one of {list(RENDER_MODES)}, got {mode!r}")
    if mode == "density" or n_spikes > MAX_TICKS:
        return "density"
    if mode == "ticks":
        return "ticks"
    threshold = DENSITY_OFF if current == "density" else DENSITY_ON
    return "density" if n_spikes > threshold * max(n_cells, 1) else "ticks"


def auto_tick_width(n_spikes: int, n_cells: int, tick_height_px: float) -> int:
    """A tick width in pixels that reads well for ``n_spikes`` in ``n_cells`` pixel cells.

    As wide as the spacing allows: a quarter of the mean gap between
    neighbouring spikes in a row, so sparse views get bold ticks and crowded
    ones thin ticks that stay apart. Never wider than a third of the tick's
    height, so a tick in a thin row still reads as a tick and not a blob.
    """
    gap_px = max(n_cells, 1) / max(n_spikes, 1)
    widest = min(_AUTO_TICK_WIDTH_MAX, int(tick_height_px // 3))
    return max(1, min(int(gap_px * _AUTO_TICK_GAP_FRACTION), widest))


def order_units(unit_ids: list[int], fitted: list[int] | None) -> list[int]:
    """``unit_ids`` in the ``fitted`` order; units the fit never saw follow, in their own order."""
    if not fitted:
        return list(unit_ids)
    shown = set(unit_ids)
    ordered = [unit for unit in fitted if unit in shown]
    seen = set(ordered)
    return ordered + [unit for unit in unit_ids if unit not in seen]


def group_by_color(
    times: NDArray,
    units: NDArray,
    rows: dict[int, int],
    colors: dict[int, tuple],
    default_color: tuple,
) -> list[SpikeGroup]:
    """Spikes split into one ``(times, row keys, colour)`` entry per colour.

    ``rows`` names the units to draw and the row key of each; a spike of any
    other unit is dropped. Units without an entry in ``colors`` share
    ``default_color`` and come first, so the coloured ones are drawn on top.
    """
    if not rows or len(times) == 0:
        return []
    known = np.array(sorted(rows), dtype=np.int64)
    idx = np.searchsorted(known, units)
    idx[idx == len(known)] = 0
    keep = known[idx] == units
    times, idx = times[keep], idx[keep]

    palette = [default_color]
    for unit in known:
        color = colors.get(int(unit))
        if color is not None and color not in palette:
            palette.append(color)
    group_of_unit = np.array([palette.index(colors.get(int(unit), default_color)) for unit in known])
    row_of_unit = np.array([rows[int(unit)] for unit in known], dtype=np.int64)

    group = group_of_unit[idx]
    row = row_of_unit[idx]
    entries = []
    for g, color in enumerate(palette):
        mask = group == g
        if mask.any():
            entries.append((times[mask], row[mask], color))
    return entries


def tick_segments(times: NDArray, y: NDArray, half_height: float) -> tuple[NDArray, NDArray]:
    """Vertical tick endpoints as consecutive pairs, for a ``connect="pairs"`` curve."""
    x = np.repeat(np.asarray(times, dtype=np.float64), 2)
    ys = np.repeat(np.asarray(y, dtype=np.float64), 2)
    ys[0::2] -= half_height
    ys[1::2] += half_height
    return x, ys


def density_image(
    groups: list[SpikeGroup],
    t0: float,
    t1: float,
    n_x: int,
    y0: float,
    cell_height: float,
    n_y: int,
) -> NDArray[np.uint8]:
    """Spike counts on an ``(n_x, n_y)`` grid as an RGBA image, shape ``(n_x, n_y, 4)``.

    The grid starts at ``(t0, y0)`` with cells ``(t1 - t0) / n_x`` wide and
    ``cell_height`` tall. A cell's colour is the count-weighted mean of the
    colours of the spikes in it; its alpha is the count on a log scale, so an
    empty cell is transparent.
    """
    if n_x <= 0 or n_y <= 0:
        raise ValueError(f"grid must be at least 1x1, got {n_x}x{n_y}")
    if t1 <= t0 or cell_height <= 0:
        raise ValueError(f"grid cells must have a positive size, got t0={t0}, t1={t1}, cell_height={cell_height}")

    n_cells = n_x * n_y
    total = np.zeros(n_cells, dtype=np.float64)
    rgb = np.zeros((n_cells, 3), dtype=np.float64)
    x_scale = n_x / (t1 - t0)
    for times, y, color in groups:
        ix = np.floor((np.asarray(times, dtype=np.float64) - t0) * x_scale).astype(np.int64)
        iy = np.floor((np.asarray(y, dtype=np.float64) - y0) / cell_height).astype(np.int64)
        inside = (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)
        counts = np.bincount(ix[inside] * n_y + iy[inside], minlength=n_cells)
        hit = np.flatnonzero(counts)
        total[hit] += counts[hit]
        rgb[hit] += counts[hit, None] * np.asarray(color[:3], dtype=np.float64)

    image = np.zeros((n_cells, 4), dtype=np.uint8)
    filled = total > 0
    if filled.any():
        brightest = max(float(np.percentile(total[filled], _BRIGHTEST_PERCENTILE)), 1.0)
        scaled = np.minimum(np.log1p(total[filled]) / np.log1p(brightest), 1.0)
        brightness = _MIN_BRIGHTNESS + (1.0 - _MIN_BRIGHTNESS) * scaled
        image[filled, :3] = np.round(rgb[filled] / total[filled, None])
        image[filled, 3] = np.round(255.0 * brightness)
    return image.reshape(n_x, n_y, 4)
