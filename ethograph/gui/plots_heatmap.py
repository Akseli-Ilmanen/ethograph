"""Heatmap plot for visualizing feature sub-dimensions as color-coded rows."""

from typing import Optional

import numpy as np
import pyqtgraph as pg

import ethograph as eto
from ethograph.features.preprocessing import z_normalize
from ethograph.io.plot_sources import WindowedBuffer, XarraySource, audio_display_offset

from .app_constants import (
    COLORBAR_WIDTH_PX,
    DEFAULT_BUFFER_MULTIPLIER,
    HEATMAP_DEBOUNCE_MS,
    Z_INDEX_BACKGROUND,
)
from .heatmap_sort import argmax_window_order, row_window
from .make_pretty import clean_display_labels
from .plots_base import BasePlot, PanelStateMixin, ThrottleDebounce


class HeatmapPlot(PanelStateMixin, BasePlot):
    """MNE-style stacked heatmap rendering feature data as color-coded rows.

    Uses a global y-coordinate space where each channel has a fixed position
    (like EphysTracePlot). The image always covers all channels; zooming
    changes the viewport, not the coordinate system. This prevents jiggling
    when the user zooms in/out.
    """

    default_free_dims = 1

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        self.setLabel("left", "Channel", Fontsize="14pt")

        self.image_item = pg.ImageItem(autoDownsample=False)
        self.image_item.setZValue(Z_INDEX_BACKGROUND)
        self.addItem(self.image_item)
        self.vb.invertY(True)

        self._init_colormap()
        self._init_colorbar()

        self.label_items = []
        self._n_channels = 1
        self._n_rows_shown = 1
        self._channel_labels = []
        self._sort_order: np.ndarray | None = None
        # (feature, trial, selections) the trial-window sort was last computed for.
        self._auto_sort_context: tuple | None = None

        # Unified buffer for xarray feature data
        self._buffer_multiplier = DEFAULT_BUFFER_MULTIPLIER
        self._buffer = WindowedBuffer(buffer_multiplier=DEFAULT_BUFFER_MULTIPLIER)
        self._buffered_data = None
        self._buffered_time = None
        self._buffer_t0 = 0.0
        self._buffer_t1 = 0.0
        self._current_feature = None
        self._current_trial = None
        self._current_ds_kwargs_hash = None

        # Cached normalization (avoids recomputing on every pan)
        self._normalized_buffer = None
        self._cached_norm_mode = None
        self._norm_data_id = None
        self._cached_levels: tuple[float, float] | None = None

        # Track last-rendered labels to skip redundant axis updates
        self._last_visible_labels: list[str] | None = None

        # Debounce-only (no throttle) — render is expensive; buffer check gates triggers
        self._td = ThrottleDebounce(
            debounce_ms=HEATMAP_DEBOUNCE_MS,
            throttle_cb=self._do_range_update,
            debounce_cb=self._do_range_update,
        )
        self._rendering = False

        self.vb.sigXRangeChanged.connect(self._on_view_range_changed)

        # Enable y-axis panning/zooming (like ephys multichannel)
        self.vb.setMouseEnabled(x=True, y=True)

    def _setup_global_y_space(self):
        n = self._n_rows_shown
        if n <= 0:
            return
        margin = 0.5
        self.vb.setLimits(yMin=-margin, yMax=n - 1 + margin)
        self.plot_item.setYRange(-margin, n - 1 + margin, padding=0)

    def refresh_row_window(self) -> None:
        """Re-render after the row-window percent or position changed."""
        self._last_visible_labels = None
        if self._buffered_data is not None:
            t0, t1 = self.get_current_xlim()
            self._render_heatmap(t0, t1)

    def set_sort_order(self, order: np.ndarray | None):
        self._sort_order = order
        self._last_visible_labels = None
        if self._buffered_data is not None:
            t0, t1 = self.get_current_xlim()
            self._render_heatmap(t0, t1)

    # --- Peak-window sorting (settings: heatmap_sort_*) ---

    def _sort_params(self) -> tuple[float, float]:
        window_s = float(self.app_state.get_with_default("heatmap_sort_window_s"))
        overlap = float(self.app_state.get_with_default("heatmap_sort_overlap"))
        return window_s, overlap

    def _order_for_range(self, t0: float, t1: float) -> np.ndarray | None:
        if self._normalized_buffer is None or self._buffered_time is None:
            return None
        mask = (self._buffered_time >= t0) & (self._buffered_time <= t1)
        if not np.any(mask):
            return None
        window_s, overlap = self._sort_params()
        return argmax_window_order(self._normalized_buffer[mask], self._buffered_time[mask], window_s, overlap)

    def sort_by_visible_window(self) -> bool:
        """Order rows by their peak window inside the visible x-range; keep it."""
        t0, t1 = self.get_current_xlim()
        # Render first: a pan still waiting on its debounce, or a feature/trial
        # change, leaves a buffer that does not hold what is on screen.
        self._render_heatmap(t0, t1)
        order = self._order_for_range(t0, t1)
        if order is None:
            return False
        self.set_sort_order(order)
        return True

    def resort_for_trial(self) -> None:
        """Re-run the trial-window sort now (mode switched, parameters edited)."""
        self._auto_sort_context = None
        if self._buffered_data is not None or self._effective_feature() is not None:
            t0, t1 = self.get_current_xlim()
            self._render_heatmap(t0, t1)

    def _trial_sort_range(self) -> tuple[float, float] | None:
        bounds = self.app_state.window_bounds
        if bounds is None:
            return None
        return float(bounds.start_s), float(bounds.end_s)

    def _trial_sort_pending(self) -> tuple | None:
        """The (feature, trial, selections) a trial-window sort is still owed for, else None."""
        if self.app_state.get_with_default("heatmap_sort_mode") != "trial":
            return None
        context = (self._effective_feature(), getattr(self.app_state, "trials_sel", None), self._get_selections_hash())
        return None if context == self._auto_sort_context else context

    def _apply_trial_sort(self, context: tuple, tr0: float, tr1: float):
        """Sort once per (feature, trial, selections) over the whole trial, from the buffer."""
        self._auto_sort_context = context
        order = self._order_for_range(tr0, tr1)
        if order is not None:
            self._sort_order = order
            self._last_visible_labels = None

    def get_normalized_data_for_range(self, t0: float, t1: float) -> np.ndarray | None:
        if self._normalized_buffer is None or self._buffered_time is None:
            return None
        mask = (self._buffered_time >= t0) & (self._buffered_time <= t1)
        if not np.any(mask):
            return None
        return np.asarray(self._normalized_buffer[mask])

    def _init_colormap(self):
        colormap_name = self.app_state.get_with_default("heatmap_colormap")
        try:
            self._cmap = pg.colormap.get(colormap_name, source="matplotlib")
        except (KeyError, ValueError, TypeError):
            self._cmap = pg.colormap.get("RdBu_r", source="matplotlib")
        self.image_item.setColorMap(self._cmap)

    def _init_colorbar(self):
        # Lands in the right gutter every panel reserves (BasePlot._GUTTER_CELL),
        # so a heatmap's plotting rectangle ends where a line plot's does.
        self.colorbar = pg.ColorBarItem(
            values=(-1, 1),
            colorMap=self._cmap,
            interactive=False,
            width=COLORBAR_WIDTH_PX,
        )
        self.colorbar.setImageItem(self.image_item, insert_in=self.plot_item)

    def update_colormap(self, name: str):
        try:
            cmap = pg.colormap.get(name, source="matplotlib")
            self._cmap = cmap
            self.image_item.setColorMap(cmap)
            self.colorbar.setColorMap(cmap)
        except (KeyError, ValueError, TypeError):
            pass

    # --- Context tracking (same pattern as LinePlot) ---

    def _get_selections_hash(self) -> str:
        selections = self._effective_selections()
        return str(sorted(selections.items()))

    def _context_changed(self) -> bool:
        feature = self._effective_feature()
        trial = getattr(self.app_state, "trials_sel", None)
        sel_hash = self._get_selections_hash()
        return (
            feature != self._current_feature or trial != self._current_trial or sel_hash != self._current_ds_kwargs_hash
        )

    def _update_context(self):
        self._current_feature = self._effective_feature()
        self._current_trial = getattr(self.app_state, "trials_sel", None)
        self._current_ds_kwargs_hash = self._get_selections_hash()

    def _clear_buffer(self):
        self._buffer.invalidate()
        self._buffered_data = None
        self._buffered_time = None
        self._buffer_t0 = 0.0
        self._buffer_t1 = 0.0
        self._normalized_buffer = None
        self._cached_norm_mode = None
        self._norm_data_id = None
        self._cached_levels = None
        self._last_visible_labels = None

    def _normalize_buffer(self):
        if self._buffered_data is None:
            self._normalized_buffer = None
            self._cached_levels = None
            return
        norm_mode = self.app_state.get_with_default("heatmap_normalization")
        data = self._buffered_data
        if norm_mode == "none":
            normalized = np.asarray(data, dtype=np.float32)
        elif norm_mode == "global":
            mu = np.nanmean(data)
            std = np.nanstd(data)
            normalized = ((data - mu) / std if std > 0 else data - mu).astype(np.float32)
        else:
            normalized = z_normalize(data).astype(np.float32)
        np.nan_to_num(normalized, copy=False, nan=0.0)
        self._normalized_buffer = normalized
        self._cached_norm_mode = norm_mode
        self._norm_data_id = id(self._buffered_data)
        self._cached_levels = self._compute_symmetric_levels(normalized)

    def _downsample_for_display(self, data: np.ndarray, max_samples: int) -> np.ndarray:
        n_samples = data.shape[0]
        if n_samples <= max_samples:
            return data
        block_size = -(-n_samples // max_samples)  # ceil division — covers all data
        n_full = n_samples // block_size
        usable = n_full * block_size
        result = data[:usable].reshape(n_full, block_size, data.shape[1]).mean(axis=1)
        if usable < n_samples:
            last = data[usable:].mean(axis=0, keepdims=True)
            result = np.vstack([result, last])
        return result

    # --- Audio envelope loading ---

    def _get_buffered_audio_envelope(self, t0: float, t1: float):
        """Load audio, compute per-channel envelope using selected metric, and cache."""
        from .plots_spectrogram import SharedAudioCache

        audio_path = getattr(self.app_state, "audio_path", None)
        if not audio_path:
            return None, None
        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            return None, None
        # t0/t1 are display-clock; the file starts at file_start on that axis.
        file_start = audio_display_offset(self.app_state)
        fs = loader.rate
        file_end = file_start + len(loader) / fs

        window_size = t1 - t0
        buffer_size = window_size * self._buffer_multiplier
        load_t0 = max(file_start, t0 - buffer_size / 2)
        load_t1 = min(file_end, t1 + buffer_size / 2)

        margin = (t1 - t0) * 0.2
        if self._buffered_data is not None and self._buffer_t0 <= t0 - margin and self._buffer_t1 >= t1 + margin:
            return self._buffered_data, self._buffered_time

        start_idx = max(0, int((load_t0 - file_start) * fs))
        stop_idx = min(len(loader), int((load_t1 - file_start) * fs))
        if stop_idx <= start_idx:
            return None, None

        audio_data = np.array(loader[start_idx:stop_idx], dtype=np.float64)
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]

        n_channels = audio_data.shape[1]
        metric = self.app_state.get_with_default("energy_metric")

        from .widgets_transform import compute_energy_envelope

        env_channels = []
        for ch in range(n_channels):
            _, ch_env = compute_energy_envelope(audio_data[:, ch], fs, metric, self.app_state)
            env_channels.append(ch_env)

        # Align channels to same length (may differ slightly between metrics)
        min_len = min(len(e) for e in env_channels)
        env_data = np.stack([e[:min_len] for e in env_channels], axis=1)
        env_time = np.linspace(load_t0, load_t1, env_data.shape[0])

        self._channel_labels = [f"Ch {i}" for i in range(n_channels)]
        self._n_channels = n_channels

        self._buffered_data = env_data
        self._buffered_time = env_time
        self._buffer_t0 = load_t0
        self._buffer_t1 = load_t1

        return env_data, env_time

    # --- Ephys envelope loading ---

    def _get_buffered_ephys_envelope(self, t0: float, t1: float):
        """Load ephys data, compute per-channel envelope, and cache."""
        from ..io.ephys_loader import load_ephys
        from .plots_ephystrace import ephys_display_offset

        ephys_path, stream_id, _ = self.app_state.get_ephys_source()
        if not ephys_path:
            return None, None
        try:
            loader = load_ephys(ephys_path, stream_id)
        except Exception:
            return None, None
        # t0/t1 are display-clock. file_time = display_time + file_off — the
        # same conversion the trace plot applies (this envelope previously
        # ignored the offset entirely and disagreed with it in every scope).
        file_off = ephys_display_offset(
            self.app_state, scalar=float(getattr(self.app_state, "ephys_offset", 0.0) or 0.0)
        )
        fs = loader.rate
        disp_min = -file_off
        disp_max = len(loader) / fs - file_off

        window_size = t1 - t0
        buffer_size = window_size * self._buffer_multiplier
        load_t0 = max(disp_min, t0 - buffer_size / 2)
        load_t1 = min(disp_max, t1 + buffer_size / 2)

        margin = (t1 - t0) * 0.2
        if self._buffered_data is not None and self._buffer_t0 <= t0 - margin and self._buffer_t1 >= t1 + margin:
            return self._buffered_data, self._buffered_time

        start_idx = max(0, int((load_t0 + file_off) * fs))
        stop_idx = min(len(loader), int((load_t1 + file_off) * fs))
        if stop_idx <= start_idx:
            return None, None

        raw = np.array(loader[start_idx:stop_idx], dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]

        n_channels = raw.shape[1]

        # Compute amplitude envelope per channel via RMS in short windows
        win_samples = max(1, int(0.01 * fs))  # 10 ms windows
        n_windows = raw.shape[0] // win_samples
        if n_windows == 0:
            return None, None

        usable = n_windows * win_samples
        reshaped = raw[:usable].reshape(n_windows, win_samples, n_channels)
        env_data = np.sqrt(np.mean(reshaped**2, axis=1))  # (n_windows, n_channels)
        env_time = np.linspace(load_t0, load_t1, env_data.shape[0])

        if hasattr(loader, "channel_names"):
            self._channel_labels = loader.channel_names[:n_channels]
        else:
            self._channel_labels = [f"Ch {i}" for i in range(n_channels)]
        self._n_channels = n_channels

        self._buffered_data = env_data
        self._buffered_time = env_time
        self._buffer_t0 = load_t0
        self._buffer_t1 = load_t1

        return env_data, env_time

    # --- Buffered data loading ---

    def _ensure_xarray_source(self):
        ds = self.app_state.ds
        time_coord = self.app_state.time_coord
        if ds is None or time_coord is None:
            self._buffer.set_source(None)
            return
        bounds = self.app_state.window_bounds
        source = XarraySource(ds, time_coord.name)
        self._buffer.set_source(source, bounds=bounds)

    def _get_buffered_data(self, t0: float, t1: float):
        """Load and cache feature data for the visible time range with buffer."""
        if self._context_changed():
            self._clear_buffer()
            self._update_context()

        margin = (t1 - t0) * 0.2
        if self._buffered_data is not None and self._buffer_t0 <= t0 - margin and self._buffer_t1 >= t1 + margin:
            return self._buffered_data, self._buffered_time

        feature_sel = self._effective_feature()
        if feature_sel is None:
            return None, None
        view_mode = getattr(self.app_state, "feature_view_mode", "Heatmap")

        if view_mode == "Heatmap (Audio)":
            return self._get_buffered_audio_envelope(t0, t1)
        if view_mode == "Heatmap (Ephys)":
            return self._get_buffered_ephys_envelope(t0, t1)

        selections = self._effective_selections()
        store = getattr(self.app_state, "data_loader", None)

        if store is not None:
            # DataLoader path (pynapple or xarray via loader)
            buf_t0 = t0 - (t1 - t0) * 2
            buf_t1 = t1 + (t1 - t0) * 2
            plot_data = store.select(feature_sel, selections, t0=buf_t0, t1=buf_t1)
            if plot_data is None:
                return None, None

            data = plot_data.data
            time = plot_data.time
            if data.ndim == 1:
                data = data[:, np.newaxis]

            if plot_data.dim_labels:
                self._channel_labels = clean_display_labels(plot_data.dim_labels)
            elif data.shape[1] > 1:
                self._channel_labels = [str(i) for i in range(data.shape[1])]
            else:
                self._channel_labels = [feature_sel]
        else:
            # Legacy xarray buffer path
            if self._buffer.source is None:
                self._ensure_xarray_source()

            buffered_ds = self._buffer.get(t0, t1)
            if buffered_ds is None:
                return None, None

            ds = self.app_state.ds
            time_coord = self.app_state.time_coord
            da = buffered_ds[feature_sel]
            data, _ = eto.sel_valid(da, selections)

            if data.ndim == 1:
                data = data[:, np.newaxis]

            da_full = ds[feature_sel]
            dims_after_sel = [d for d in da_full.dims if "time" not in d and d not in selections]
            if dims_after_sel and dims_after_sel[0] in da_full.coords:
                self._channel_labels = clean_display_labels([str(v) for v in da_full.coords[dims_after_sel[0]].values])
            elif data.shape[1] > 1:
                self._channel_labels = [str(i) for i in range(data.shape[1])]
            else:
                self._channel_labels = [feature_sel]

            time = buffered_ds.coords[time_coord.name].values

        self._n_channels = data.shape[1]
        self._buffered_data = data
        self._buffered_time = time
        self._buffer_t0 = t0 - (t1 - t0) * 2
        self._buffer_t1 = t1 + (t1 - t0) * 2

        return data, time

    # --- Rendering ---

    def _compute_symmetric_levels(self, data: np.ndarray) -> tuple[float, float]:
        """Compute symmetric color range using exclusion percentile from app_state."""
        percentile = self.app_state.get_with_default("heatmap_exclusion_percentile")
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return -1.0, 1.0
        vmax = np.percentile(np.abs(valid), percentile)
        if vmax < 1e-10:
            vmax = 1.0
        return -vmax, vmax

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        self._ensure_panel_state()
        if self._effective_feature() is None:
            return

        if t0 is None or t1 is None:
            t0, t1 = self.get_current_xlim()

        self._render_heatmap(t0, t1)

    def _render_heatmap(self, t0: float, t1: float):
        self._rendering = True
        try:
            view_mode = getattr(self.app_state, "feature_view_mode", "Heatmap")
            if view_mode == "Heatmap":
                self.setTitle(self._effective_feature())
            else:
                self.setTitle(None)

            # A pending trial-window sort widens the load to the whole trial,
            # so the sort sees every sample and later pans stay in the buffer.
            pending = self._trial_sort_pending()
            trial_range = self._trial_sort_range() if pending is not None else None
            if trial_range is not None:
                load_t0, load_t1 = min(t0, trial_range[0]), max(t1, trial_range[1])
            else:
                load_t0, load_t1 = t0, t1

            result = self._get_buffered_data(load_t0, load_t1)
            if result[0] is None:
                return

            data, time_vals = result
            if len(time_vals) == 0:
                return

            norm_mode = self.app_state.get_with_default("heatmap_normalization")
            if self._normalized_buffer is None or self._norm_data_id != id(data) or self._cached_norm_mode != norm_mode:
                self._normalize_buffer()

            if pending is not None and trial_range is not None:
                self._apply_trial_sort(pending, *trial_range)
            normalized = self._normalized_buffer

            if self._sort_order is not None and len(self._sort_order) == normalized.shape[1]:
                normalized = normalized[:, self._sort_order]
                sorted_labels = [self._channel_labels[i] for i in self._sort_order]
            else:
                sorted_labels = list(self._channel_labels)

            rows = row_window(
                normalized.shape[1],
                float(self.app_state.get_with_default("heatmap_row_percent")),
                float(self.app_state.get_with_default("heatmap_row_position")),
            )
            normalized = normalized[:, rows]
            sorted_labels = sorted_labels[rows]
            n_total = normalized.shape[1]
            self._n_rows_shown = n_total

            vmin, vmax = self._cached_levels or self._compute_symmetric_levels(normalized)

            pixel_width = self.width() or 800
            display_data = self._downsample_for_display(normalized, pixel_width * 2)

            self.image_item.setImage(display_data, autoLevels=False)
            self.image_item.setLevels([vmin, vmax])
            self.colorbar.setLevels(values=(vmin, vmax))

            buf_t0 = float(time_vals[0])
            buf_t1 = float(time_vals[-1])
            n_display = display_data.shape[0]
            duration = buf_t1 - buf_t0

            # Half-pixel correction: each image pixel is a bin, not a point.
            # Shift rect so pixel centers align with sample times.
            if n_display > 1:
                dt = duration / (n_display - 1)
            else:
                dt = max(duration, 1e-6)
            rect_x = buf_t0 - dt / 2
            rect_w = duration + dt

            # Image covers all channels in global y-space [0, n_total]
            self.image_item.setRect(pg.QtCore.QRectF(rect_x, 0, rect_w, n_total))

            # Set up global y-space on first render or channel count change
            self._setup_global_y_space()

            if sorted_labels != self._last_visible_labels:
                self._update_y_axis_ticks(sorted_labels)
                self._last_visible_labels = sorted_labels
        finally:
            self._rendering = False

    def _update_y_axis_ticks(self, labels=None):
        """Set y-axis tick labels to channel names."""
        if labels is None:
            labels = self._channel_labels
        left_axis = self.plot_item.getAxis("left")
        ticks = [(i + 0.5, label) for i, label in enumerate(labels)]
        left_axis.setTicks([ticks])

    # --- View range handling ---

    def _buffer_covers(self, t0: float, t1: float) -> bool:
        if self._buffered_data is None or self._context_changed():
            return False
        margin = (t1 - t0) * 0.2
        return self._buffer_t0 <= t0 - margin and self._buffer_t1 >= t1 + margin

    def _on_view_range_changed(self):
        if self._rendering:
            return
        if not self.isVisible():
            return
        if not hasattr(self.app_state, "ds") or self.app_state.ds is None:
            return
        t0, t1 = self.get_current_xlim()
        if self._buffer_covers(t0, t1):
            return
        self._td.trigger()

    def _do_range_update(self):
        t0, t1 = self.get_current_xlim()
        self._render_heatmap(t0, t1)

    # --- Y-axis management ---

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        if ymin is None and ymax is None:
            return
        if ymin is None or ymax is None:
            cur_lo, cur_hi = self.vb.viewRange()[1]
            ymin = cur_lo if ymin is None else ymin
            ymax = cur_hi if ymax is None else ymax
        self.plot_item.setYRange(ymin, ymax)

    def _apply_y_constraints(self):
        n = self._n_rows_shown
        margin = 0.5
        self.vb.setLimits(yMin=-margin, yMax=n - 1 + margin)
