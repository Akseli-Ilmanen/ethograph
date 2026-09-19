"""Constants used across the GUI module, that should be rarely modified by the user."""

# =============================================================================
# UI DIMENSIONS
# =============================================================================

# Labels table (widgets_labels.py)
LABELS_TABLE_MAX_HEIGHT = 500
LABELS_TABLE_ROW_HEIGHT = 20
LABELS_TABLE_ID_COLUMN_WIDTH = 20
LABELS_TABLE_COLOR_COLUMN_WIDTH = 20
LABELS_WIDGET_SIZE_HINT_HEIGHT = 800

# Cluster info table (widgets_plot_settings.py)
CLUSTER_TABLE_ROW_HEIGHT = 20
CLUSTER_TABLE_MAX_HEIGHT = 300

# Ordered list of pose/bbox source software offered wherever the user must
# pick one (drop-details dialog, pose-overlay prompt).
POSE_SOFTWARES = ["DeepLabCut", "SLEAP", "LightningPose", "Anipose", "VIA-tracks"]

# Session-basis label overlay (widgets_data.update_label_plot): draw only the
# labels intersecting the viewport, and none at all above this cap — at that
# density the rectangles are sub-pixel mush and thousands of items per panel
# crawl. Zooming in shrinks the visible set below the cap and they reappear.
SESSION_LABELS_MAX_DRAWN = 100

# Label overlay box on video (widgets_labels.py)
LABELS_OVERLAY_BOX_WIDTH = 250
LABELS_OVERLAY_BOX_HEIGHT = 50
LABELS_OVERLAY_BOX_MARGIN = 5
LABELS_OVERLAY_TEXT_SIZE = 18
LABELS_OVERLAY_FALLBACK_SIZE = (100, 100)

#: Fixed branch -> draw-position mapping: branch 0 renders "full" (main), 1
#: "top1", 2 "top2". A hard rule of the renderer, not a preference.
MAX_LABEL_BRANCHES = 3

# Per-plot-type label rendering modes (label_drawing_mixin.py, widgets_labels.py)
LABEL_OVERLAY_MODE_FULL = "full"
LABEL_OVERLAY_MODE_BOTTOM = "bottom"
LABEL_OVERLAY_MODE_NONE = "none"

# Where a new label's boundaries come from (widgets_labels.py,
# app_state.labelling_mode): a click on a panel, or the label key pressed at
# the frame on screen.
LABELLING_MODE_PLOTS = "plots"
LABELLING_MODE_FRAME = "frame"
LABELLING_MODES = {
    LABELLING_MODE_FRAME: "Frame by frame labelling",
    LABELLING_MODE_PLOTS: "Graph based labelling",
}

# type key -> display name shown in the "Show labels per plot type" dialog
LABEL_OVERLAY_PLOT_TYPES = {
    "lineplot": "Line plot",
    "audio": "Audio trace",
    "spectrogram": "Spectrogram",
    "heatmap": "Heatmap",
    "ephys": "Ephys trace (phy)",
    "neo": "Ephys trace (neo)",
}

DEFAULT_LABEL_OVERLAY_MODES = {
    "lineplot": LABEL_OVERLAY_MODE_FULL,
    "audio": LABEL_OVERLAY_MODE_FULL,
    "spectrogram": LABEL_OVERLAY_MODE_BOTTOM,
    "heatmap": LABEL_OVERLAY_MODE_BOTTOM,
    "ephys": LABEL_OVERLAY_MODE_NONE,
    "neo": LABEL_OVERLAY_MODE_NONE,
}

# Qt maximum size sentinel (QWIDGETSIZE_MAX) used to un-cap widget dimensions.
MAX_WIDGET_SIZE = 16777215


SIDEBAR_DEFAULT_WIDTH_RATIO = 0.40
SIDEBAR_AFTER_LOAD_WIDTH_RATIO = 0.25
SIDEBAR_MIN_WIDTH_PX = 280

# The playback bar's controls are ~900 px wide; without a small explicit
# minimum its dock pins the window layout and the sidebar cannot be widened.
BOTTOM_BAR_MIN_WIDTH_PX = 240

# Dock layout (layout_manager.py)
LAYER_DOCK_WIDTH_RATIO = 0.20
VERTICAL_SPLIT_RATIO = 0.45
LAYOUT_RELEASE_DELAY_MS = 300

# Plot container (plot_container.py, widgets_meta.py)
#
# The media/plots separator must be draggable across the whole window (~10/90
# either way), and Qt clamps a separator drag at the minimum size of whatever
# sits on each side.  Every minimum below is therefore deliberately tiny — a
# panel squeezed to a sliver is the user's call, not the layout's.  Default
# proportions come from sizeHints and resizeDocks, never from minimums.
PLOT_CONTAINER_MIN_HEIGHT = 48
PLOT_CONTAINER_SIZE_HINT_HEIGHT = 300
#: Minimum height of a single plot panel (its dock adds the 17 px title bar).
PANEL_MIN_HEIGHT = 24
#: Minimum size of a camera / space-plot view (the media half of the split).
MEDIA_VIEW_MIN_WIDTH = 60
MEDIA_VIEW_MIN_HEIGHT = 40
DOCK_WIDGET_BOTTOM_MARGIN = 50

# Right gutter of every stacked panel (plots_base.BasePlot.reserve_right_gutter).
# Each panel keeps this much empty space right of its plotting rectangle —
# exactly the footprint of a heatmap's colorbar — so a colorbar panel and a
# plain panel end their time axes on the same pixel whatever the colorbar's
# tick labels say. A line plot parks its legend there. The numbers are
# pyqtgraph's own: a ColorBarItem inserted into a PlotItem fixes a 5 px spacer
# column and lays itself out as COLORBAR_WIDTH_PX of bar + a 45 px axis + its
# frame. Covered by tests/test_unit/test_panel_gutter.py.
COLORBAR_WIDTH_PX = 15
PANEL_RIGHT_SPACER_PX = 5
PANEL_RIGHT_GUTTER_PX = 62

# Layout spacing (widgets_data.py, widgets_labels.py)
DEFAULT_LAYOUT_SPACING = 2
DEFAULT_LAYOUT_MARGIN = 2

# =============================================================================
# PLOT SETTINGS
# =============================================================================

# Axis locking (plots_base.py)
LOCKED_RANGE_MIN_FACTOR = 0.8  # window_size * 0.8 when locked
LOCKED_RANGE_MAX_FACTOR = 1.5  # window_size * 1.5 when locked
AXIS_LIMIT_PADDING_RATIO = 0.05  # 5% of data range as padding for xMin/xMax

# Label drawing (plot_container.py)
PREDICTION_LABELS_HEIGHT_RATIO = 0.10  # Height as fraction of y-range
SPECTROGRAM_LABELS_HEIGHT_RATIO = 0.10  # Height as fraction of y-range
SPECTROGRAM_OVERLAY_OPACITY = 0.6
PREDICTION_FALLBACK_Y_TOP = 20000
PREDICTION_FALLBACK_Y_HEIGHT = 2000
SPECTROGRAM_FALLBACK_Y_HEIGHT = 1600

# Zoom thresholds for spectrogram overlay refresh (plot_container.py)
SPECTROGRAM_OVERLAY_ZOOM_OUT_THRESHOLD = 0.5  # Refresh when width < old * 0.5
SPECTROGRAM_OVERLAY_ZOOM_IN_THRESHOLD = 2.0  # Refresh when width > old * 2.0

# Changepoint line styles based on zoom level (plot_container.py)
CP_ZOOM_VERY_OUT_THRESHOLD = 10.0  # seconds visible
CP_ZOOM_MEDIUM_THRESHOLD = 2.0  # seconds visible
CP_LINE_WIDTH_THIN = 0.1
CP_LINE_WIDTH_MEDIUM = 1.0
CP_LINE_WIDTH_THICK = 2.0

# =============================================================================
# TIMING / DEBOUNCE
# =============================================================================

LINEPLOT_DEBOUNCE_MS = 10
AUDIOTRACE_DEBOUNCE_MS = 30
ENVELOPE_OVERLAY_DEBOUNCE_MS = 30
SPECTROGRAM_DEBOUNCE_MS = 30
HEATMAP_DEBOUNCE_MS = 50
EPHYSTRACE_DEBOUNCE_MS = 30
RASTER_DEBOUNCE_MS = 30

# =============================================================================
# AUDIO / SPECTROGRAM
# =============================================================================

# Buffer settings (plots_spectrogram.py)
DEFAULT_BUFFER_MULTIPLIER = 4.0
DEFAULT_BUFFER_MULTIPLIER_AUDIO = 2.0
DEFAULT_BUFFER_MULTIPLIER_EPHYS = 2.0
BUFFER_COVERAGE_MARGIN = 0.2  # 20% margin for buffer coverage check

# Frequency limits (plots_spectrogram.py)
DEFAULT_FALLBACK_MAX_FREQUENCY = 25000  # Hz, fallback when audio not loaded

# =============================================================================
# DATA PROCESSING
# =============================================================================

# Multi-resolution pyramid (plots using timeseries pyramid)

# Multi-resolution pyramid levels for ephys/audio downsampling
# Controls the zoom strategy for envelope downsampling in trace plots.
# Each value is a downsampling factor: e.g., 64 means min/max pairs for every 64 samples.
# Used in plots_ephystrace.py to select pyramid level based on zoom.

# at 30kHz
# 262144 = ~8.7s window -> collapsed to min/max range
# 16 = ~0.0005s, 4 = ~0.00013s (1ms window)
PYRAMID_LEVELS = (262144, 65536, 16384, 4096, 1024, 256, 64, 16, 4)
PYRAMID_RAW_DATA_THRESHOLD_S = 2  # Below this window size, always use raw data (level 0)


# Z-index values for layering
Z_INDEX_BACKGROUND = -20
Z_INDEX_LABELS = -10
Z_INDEX_PREDICTIONS = 10
Z_INDEX_CHANGEPOINTS = 50
Z_INDEX_TIME_MARKER = 1000
Z_INDEX_LABELS_OVERLAY = 1000

# =============================================================================
# COLORS (RGBA tuples)
# =============================================================================

# Changepoint colors (plot_container.py)
CP_COLOR_WAVEFORM = (0, 0, 0, 200)  # Black for waveform plot
CP_COLOR_SPECTROGRAM = (255, 255, 255, 200)  # White for spectrogram
CP_COLOR_OSC_EVENT = (0, 200, 200, 200)  # Cyan/teal for oscillatory events

# One colour per column of a multi-column feature (keypoints, individuals, …).
# Shared by every panel that draws several columns at once, so a keypoint keeps
# the same colour whether it is a line-plot trace or a compass arrow.
MULTIDIM_COLORS = [
    "#1f77b4",  # Blue
    "#d62728",  # Red
    "#2ca02c",  # Green
    "#ff7f0e",  # Orange
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# Envelope overlay (plot_container.py)
ENVELOPE_OVERLAY_COLOR = "#ff8800"
ENVELOPE_OVERLAY_WIDTH = 2

# User-drawn horizontal reference lines on a line plot (widgets_plot_settings.py)
HLINE_COLOR = "#555555"
HLINE_WIDTH = 1

# =============================================================================
# AUDIO PLAYBACK (widgets_navigation.py, unified_container.py)
# =============================================================================
AUDIO_SPEED_MIN = 0.1
AUDIO_SPEED_MAX = 10.0
AUDIO_SPEED_STEP = 0.25
AUDIO_SPEED_DEFAULT = 1.0

# =============================================================================
# PLAYBACK MODE (video_sync.py, widgets_bottom_bar.py)
# =============================================================================
# Auto follows audio presence (audio → synced, none → smooth); the others are
# explicit user overrides. Audio is only played in "synced" mode.
PLAYBACK_MODE_AUTO = "auto"
PLAYBACK_MODE_SYNCED = "synced"  # audio-master clock; drops video frames to stay locked
PLAYBACK_MODE_SMOOTH = "smooth"  # decode-paced; every frame, may run slower than fps; no audio
PLAYBACK_MODE_SKIP = "skip"  # approximate real-time fps by skipping frames; no audio

# (display label, mode value) for the bottom-bar combo, in order.
PLAYBACK_MODE_CHOICES = [
    ("Audio-synced", PLAYBACK_MODE_SYNCED),
    ("Smooth (every frame)", PLAYBACK_MODE_SMOOTH),
    ("Real-time (skip frames)", PLAYBACK_MODE_SKIP),
]
