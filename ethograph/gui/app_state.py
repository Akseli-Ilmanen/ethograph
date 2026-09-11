"""Settings that the user can modify and are saved in gui_settings.yaml"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from qtpy.QtCore import QObject, QTimer, Signal

import ethograph as eto
from ethograph.gui.app_constants import DEFAULT_LABEL_OVERLAY_MODES, LABELLING_MODE_PLOTS
from ethograph.gui.notify import notify
from ethograph.io.catalog import INDIVIDUAL_DIMS
from ethograph.io.metadata_table import load_metadata_df
from ethograph.io.time_model import (
    RestrictionWindow,
    TimeRange,
    TrialVideoBounds,
)
from ethograph.labels import workflow as wf
from ethograph.labels.curation import trial_curation_status
from ethograph.labels.tsv_store import (
    LabelEdit,
    LabelHistory,
    get_trial_from_tsv,
    get_trial_meta,
    labels_equal,
    labels_tsv_path,
    load_labels_tsv,
    save_labels_tsv,
    set_trial_in_tsv,
    set_trial_meta_attr,
)
from ethograph.utils.paths import auto_git_commit, ethograph_home, is_throwaway_path, sanitize_path_state
from ethograph.utils.qt import find_combo_index

logger = logging.getLogger(__name__)

SIMPLE_SIGNAL_TYPES = (int, float, str, bool)


def get_signal_type(type_hint):
    """Derive Qt Signal-compatible type from a type hint."""
    if type_hint in SIMPLE_SIGNAL_TYPES:
        return type_hint
    return object


def check_type(value, type_hint) -> bool:
    """Check if value matches type_hint. Returns True if valid."""
    if value is None:
        origin = get_origin(type_hint)
        if origin is type(int | str):  # UnionType
            return type(None) in get_args(type_hint)
        return type_hint is type(None)

    origin = get_origin(type_hint)

    if origin is type(int | str):  # UnionType (e.g., str | None)
        return any(check_type(value, arg) for arg in get_args(type_hint))

    if origin is list:
        if not isinstance(value, list):
            return False
        args = get_args(type_hint)
        if args:
            return all(isinstance(item, args[0]) for item in value)
        return True

    if origin is dict:
        if not isinstance(value, dict):
            return False
        args = get_args(type_hint)
        if len(args) == 2:
            key_type, val_type = args
            return all(isinstance(k, key_type) for k in value.keys())
        return True

    if isinstance(type_hint, type):
        return isinstance(value, type_hint)

    return True


class AppStateSpec:
    SCOPE_GLOBAL = "global"
    SCOPE_LOCAL = "local"

    # Variable name: (type, default, save_to_yaml)
    VARS = {
        # Video
        "current_frame": (int, 0, False),
        "changes_saved": (bool, True, False),
        "video": (object | None, None, False),
        "num_frames": (int, 0, False),
        "_info_data": (dict[str, Any], {}, False),
        "sync_state": (str | None, None, False),
        "before_s_trial": (float, 0.0, True),
        "after_s_trial": (float, 0.0, True),
        "before_s_label": (float, 1.0, True),
        "after_s_label": (float, 1.0, True),
        "before_s_sequence": (float, 1.0, True),
        "after_s_sequence": (float, 1.0, True),
        # Frame-by-frame review (widgets_curation): total seconds of time
        # series shown around the boundary being nudged, seed centred.
        # Decoupled from the navigation Before/After padding — a reviewing
        # preference, so global like autoplay_on_navigate.
        "refine_window_s": (float, 0.5, True),
        # Curation (widgets_curation, docs advanced/labels/curation.md).
        # curation_mode: "manual" | "inspect" | "frame" — SCOPE_LOCAL, because
        # "inspect is enough" curates a trial by merely opening it and must not
        # silently follow the user into the next dataset. curation_label_ids:
        # the label classes dropped into the scope area (None = every class),
        # per dataset too. curation_next_curates: N in frame-by-frame review
        # also curates the boundary it leaves — a reviewing preference.
        "curation_mode": (str, "manual", True, SCOPE_LOCAL),
        "curation_label_ids": (list | None, None, True, SCOPE_LOCAL),
        "curation_next_curates": (bool, True, True),
        # curation_auto_advance: Enter/Backspace in frame-by-frame review jump
        # to the next target once they commit/delete the current one. On by
        # default; untick to confirm or delete without leaving the boundary.
        "curation_auto_advance": (bool, True, True),
        # Frame-by-frame review queue: skip manual/curated boundaries, only
        # queuing automated ones — a human already vouched for the rest, so
        # there is nothing to re-review. On by default; a reviewing
        # preference like curation_next_curates.
        "frame_review_automated_only": (bool, True, True),
        # Frame-by-frame review order: "trial" (every boundary of a trial,
        # then the next trial) or "label" (every instance of a class across
        # trials, then the next class) — see labels/curation.REVIEW_ORDERS.
        # A reviewing preference like curation_next_curates.
        "curation_review_order": (str, "trial", True),
        # curation_active: is anyone curating this session? Off until the user
        # drops label rows into the scope area or curates something — only then
        # does the per-trial verdict get a metadata file to live in, and only
        # then does the sync timer run. Never saved: opening the GUI curates
        # nothing, so it must not come back armed.
        "curation_active": (bool, False, False),
        # Which run's onset_curves.npz frame-by-frame review draws as the
        # confidence overlay. Session-only (never saved, reset when a dataset
        # comes or goes): set once by the onset-model Predict dialog when more
        # than one run exists (labels/onset_curves.py), so review itself never
        # has to ask — asking there re-fired on every restart_review().
        "curve_run_path": (str | None, None, False),
        # How the plot x-limits are derived: "interval" (follows slider scope:
        # trial/label/sequence extent + before/after padding) or "fixed"
        # (fixed-size window from t=0). User preference, not tied to how the
        # dataset was loaded — SCOPE_GLOBAL (default) so it persists across
        # datasets instead of being guessed per load path.
        "xlim_mode": (str, "interval", True),
        "fixed_window_s": (float, 10.0, True, SCOPE_LOCAL),
        "labels_visible": (bool, True, True, SCOPE_LOCAL),
        # Receiver of the labelled behaviour (dyadic interactions): the actor
        # is whichever individual is selected, and (actor, receiver) together
        # are the label subject — each pair its own track. "" = solo behaviour,
        # the default. SCOPE_LOCAL: individual names belong to one dataset.
        "individual_receiver": (str, "", True, SCOPE_LOCAL),
        # Per-plot-type label rendering: "full" | "bottom" | "none"
        "label_overlay_modes": (dict[str, str], dict(DEFAULT_LABEL_OVERLAY_MODES), True),
        # labelling_mode: where a new label's boundaries come from — "plots"
        # (a click on a panel) or "frame" (the label key itself places the
        # boundary at the frame on screen; a state class takes two presses).
        # A labelling habit, so it follows the user across datasets.
        "labelling_mode": (str, LABELLING_MODE_PLOTS, True),
        # label_ribbon_auto: open a label timeline panel on load when the
        # session has no panel at all (a video and nothing else), so placed
        # labels have somewhere to be seen and clicked.
        "label_ribbon_auto": (bool, True, True),
        "feature_view_mode": (str, "LinePlot", True, SCOPE_LOCAL),
        # Panel layout (UnifiedPanelContainer.layout_state()): per-dataset,
        # auto-saved to .ethograph/local_settings.yaml like other local vars.
        "panel_layout": (dict | None, None, True, SCOPE_LOCAL),
        # Outer window state (geometry base64 blob): app-wide, auto-saved to
        # gui_settings.yaml. No JSON layout files exist.
        "window_state": (dict | None, None, True),
        # Data
        "data_loader": (object | None, None, False),
        "source_collection": (object | None, None, False),
        "ds": (xr.Dataset | None, None, False),
        "ds_temp": (xr.Dataset | None, None, False),
        "dt": (xr.DataTree | None, None, False),
        "labels_confidence_ds": (xr.Dataset | None, None, False),
        "pred_labels_df": (pd.DataFrame | None, None, False),
        "pred_store": (object, None, False),
        "pred_confidence_threshold": (float, 0.75, True),
        "pred_segment_confidence_threshold": (float, 0.6, True),
        # Import Predictions panel's "Load as" combo — "overlay" or "labels".
        # A global preference like import_labels_nc_data: it's how the user
        # tends to use predictions, not something tied to one dataset.
        "pred_load_mode": (str, "overlay", True),
        # Import Predictions panel's "Merge with existing labels" checkbox,
        # shown only when importing as labels onto a session that already has
        # some. Off by default (import replaces), a global preference like
        # import_labels_nc_data — it says how the user wants imports to
        # behave, not something tied to one dataset.
        "merge_imported_predictions": (bool, False, True),
        "trial_conditions": (list | None, None, False),
        "keypoints": (list[str], [], False),
        # Global preference: the "Import labels" checkbox is remembered across
        # datasets (gui_settings.yaml). Safe as a sticky global because the
        # canonical {stem}_labels.tsv guess is only seeded when that file
        # exists — a dataset with no labels file loads without labels instead
        # of erroring. Only an explicitly-set labels_import_path (SCOPE_LOCAL)
        # that has gone missing still raises at load.
        "import_labels_nc_data": (bool, False, True),
        # Playback speed as a % of the original recording speed (100 = native
        # speed). Drives both the video frame rate and the audio pitch/rate
        # together — there is no separate FPS or audio-speed control.
        "playback_speed_pct": (float, 100.0, True),
        # Review-grid layout (dialog_video_grid, dialog_label_gridview): how
        # the grids are laid out is a viewing habit, not a dataset property —
        # global, like refine_window_s. Curation is done over many trials, so
        # every knob a reviewer tunes here is remembered across sessions
        # instead of resetting each time a grid dialog opens.
        "video_grid_point_window_s": (float, 0.5, True),
        "video_grid_per_page": (int, 6, True),
        "video_grid_columns": (int, 3, True),
        "label_grid_columns": (int, 3, True),
        # How the grids order what they show (see dialog_label_gridview
        # GRID_SORT_ORDERS). Remembered like every other grid knob: an order
        # that suits a model's output suits the next review of it too.
        "label_grid_sort": (str, "trial", True),
        "video_grid_sort": (str, "duration", True),
        "label_grid_window_s": (float, 1.0, True),
        # The video grid's own playback speed — deliberately decoupled from
        # playback_speed_pct (tweaking it for a slow-motion review must not
        # disturb the main GUI's speed) but, like the rest of the grid
        # layout, sticky across sessions rather than resetting on every open.
        "video_grid_speed_pct": (float, 100.0, True),
        # "Flag confidence below" — shared by the frame grid and the video
        # grid, so a threshold picked while reviewing one still applies when
        # switching to the other. Typed in full rather than stepped, so a
        # threshold of 0.0002 is as easy to set as 0.5
        # (dialog_label_gridview.ConfidenceEdit).
        "grid_confidence_threshold": (float, wf.DEFAULT_CONFIDENCE, True),
        # The confidence knob beside the grids' Histogram… button
        # (labels/rescore.py): which reading of a prediction's curve is the
        # confidence, the slider of the custom rule, and the "same event"
        # window in ms. A review preference, remembered like the threshold.
        "grid_confidence_rule": (str, "product", True),
        "grid_confidence_alpha": (float, 0.5, True),
        "grid_confidence_window_ms": (float, 100.0, True),
        # The grids' "Labeling method" filter: a key of
        # dialog_label_gridview.GRID_METHOD_FILTERS ("all" | "manual" |
        # "curated" | "human" | "automated"). Global like the rest of the grid
        # setup — which slice of the labels a reviewer is working through
        # outlives one dataset.
        "grid_method_filter": (str, "all", True),
        # Cameras ticked in the grid dialogs' Cameras list, by name. None =
        # no preference recorded yet (defaults to all checked). A name absent
        # from the current dataset's cameras is simply never offered.
        "grid_selected_cameras": (list[str] | None, None, True),
        # Output volume as a % (0–100) applied inside EthoGraph's own audio
        # stream, independent of the system volume. Global — a listening
        # preference, not a dataset property.
        "playback_volume_pct": (float, 100.0, True),
        # "auto" | "synced" | "smooth" | "skip" — global playback preference.
        "playback_mode": (str, "auto", True),
        "hide_label_text": (bool, False, True),
        "filter_warnings": (bool, True, True),
        "center_playback": (bool, False, True),
        # "Auto-play on navigate": start segment playback (after the fixed
        # delay in widgets_navigation) whenever navigation lands. Global — a
        # reviewing habit, not a property of any one dataset.
        "autoplay_on_navigate": (bool, False, True),
        "time_jump_s": (float, 0.1, True),
        "time": (
            xr.DataArray | None,
            None,
            False,
        ),  # for feature variables (e.g. 'time' or 'time_aux')
        "label_intervals": (pd.DataFrame | None, None, False),
        "metadata_df": (pd.DataFrame | None, None, False),
        "metadata_path": (str | None, None, True, SCOPE_LOCAL),
        # Whether the trials table lets the current trial's metadata cells be
        # edited by double-click. Per dataset, and off by default: a stray
        # double-click must never rewrite somebody's metadata file.
        "metadata_edit_enabled": (bool, False, True, SCOPE_LOCAL),
        "trial_alignment": (TrialVideoBounds | None, None, False),
        "ephys_offset": (float, 0.0, True, SCOPE_LOCAL),
        "navigate_mode": (str, "trial", True, SCOPE_LOCAL),
        "slider_scope": (str, "trial", True, SCOPE_LOCAL),
        "restrict_window": (RestrictionWindow | None, None, False),
        "label_instance_idx": (int, 0, False),
        "sequence_pattern": (str, "", True),
        "sequence_match_idx": (int, 0, False),
        "trials": (list[int | str], [], False),
        "downsample_enabled": (bool, False, True),
        "downsample_factor": (int, 100, True),
        # Boolean
        "has_audio": (bool, False, False),
        "has_neo": (bool, False, False),
        "has_neurons": (bool, False, False),
        "files_aligned_to_trials": (bool, True, True, SCOPE_LOCAL),
        # Paths
        "nc_file_path": (str | None, None, False),
        # Folder the last browse dialog resolved to. SCOPE_GLOBAL: it follows
        # the user, not the dataset — the point is that the *next* session's
        # file dialogs open where the previous one left off, before any
        # dataset (and so any local_settings.yaml) exists.
        "last_browse_dir": (str | None, None, True),
        # The project directory chosen on the cover page (gui/project.py):
        # drag & drops are kept under its sessions/ and can be reopened from
        # there. SCOPE_GLOBAL so the next start opens on the same project; a
        # folder that no longer exists is dropped on load like any PATH_VAR.
        "project_path": (str | None, None, True),
        # Cover page drop card: "same_trial" (several files = several cameras
        # filming one trial, the existing behaviour) or "multi_trial" (several
        # files of one stream = one file per trial, natural-sort paired like a
        # single-camera wizard "Pair" run). SCOPE_GLOBAL: a viewing habit, not
        # a per-dataset fact — the next drop is usually shaped like the last one.
        "drop_layout": (str, "same_trial", True),
        "_labels_file_path": (
            str | None,
            None,
            False,
        ),  # Tracks active labels file (canonical or predictions)
        # Explicit "Import labels" override: persisted per-dataset
        # (.ethograph/local_settings.yaml) so it is remembered instead of
        # re-guessed from the .nc filename on every load. Seeded once (guess
        # from labels_tsv_path) the first time the checkbox is ticked for a
        # dataset that has never set it; from then on it is the sole source
        # of truth for where "Import labels" reads from, and a missing file
        # raises rather than silently loading nothing.
        "labels_import_path": (str | None, None, True, SCOPE_LOCAL),
        "nwb_file_path": (str | None, None, True, SCOPE_LOCAL),
        "video_folder": (str | None, None, True, SCOPE_LOCAL),
        "audio_folder": (str | None, None, True, SCOPE_LOCAL),
        "pose_folder": (str | None, None, True, SCOPE_LOCAL),
        "ephys_path": (str | None, None, True, SCOPE_LOCAL),
        "neurons_path": (str | None, None, True, SCOPE_LOCAL),
        "video_path": (str | None, None, False),
        # Playback quality: "full" decodes the source video; "proxy" decodes a
        # cached low-res/short-GOP copy for smooth navigation. Global viewing
        # pref (not per-dataset). Only affects which file the DECODER reads;
        # all alignment/frame math stays on the source.
        "video_quality_mode": (str, "full", True),
        "audio_path": (str | None, None, False),
        # audio_source_map key driving audio PLAYBACK (last-clicked audio panel);
        # None follows the global mic combo. Distinct from what each panel draws.
        "playback_mic_key": (str | None, None, False),
        "pose_path": (str | None, None, False),
        "source_software": (str | None, None, True, SCOPE_LOCAL),
        "image_paths": (list[str], [], True, SCOPE_LOCAL),
        "nwb_pose_keys": (list[str], [], True, SCOPE_LOCAL),
        "pose_hide_threshold": (float, 0.9, True),
        "pose_show_skeleton": (bool, False, True),
        # Pose overlay design: how the markers, their text labels and the
        # skeleton edges LOOK. A viewing habit, so SCOPE_GLOBAL — it carries
        # from one dataset to the next.
        "pose_show_keypoints": (bool, True, True),
        "pose_show_text": (bool, False, True),
        "pose_point_size": (float, 10.0, True),
        "pose_text_size": (float, 12.0, True),
        "pose_skeleton_width": (float, 2.0, True),
        # WHICH keypoints are drawn, by name — SCOPE_LOCAL, since the keypoint
        # schema belongs to one dataset and a name means nothing in the next.
        "pose_hidden_keypoints": (list[str], [], True, SCOPE_LOCAL),
        # Which catalog feature the video panel overlays — any variable in the
        # camera's pixels qualifies (io/overlay_source.py); None means
        # ``position``. SCOPE_LOCAL: a feature name belongs to one dataset.
        "pose_overlay_feature": (str | None, None, True, SCOPE_LOCAL),
        # Which axis of the pose hierarchy colour encodes: "keypoint" (one hue
        # per body part, shared across animals) or "individual" (one hue per
        # animal, shared across its keypoints) — SLEAP's toggle. Read by the
        # pose overlay AND the keypoint labelling canvas, so one choice styles
        # every surface that draws a keypoint.
        "pose_color_by": (str, "keypoint", True),
        "pose_points_use_base": (bool, False, True),
        "pose_points_base_color": (str | None, "#FF3333", True),
        "pose_individual_colors": (dict, {}, True, SCOPE_LOCAL),
        "skeleton_use_base": (bool, True, True),
        "skeleton_base_color": (str | None, "#00CC66", True),
        "skeleton_config_override": (dict | None, None, True),
        # Keypoint labelling: the schema being labelled and the chosen fill
        # backend. The labelled coordinates themselves are project data and go
        # to a sidecar next to the video, never here.
        "labelling_keypoints": (list[str], [], True, SCOPE_LOCAL),
        "labelling_individuals": (list[str], [], True, SCOPE_LOCAL),
        "labelling_backend": (str, "spline", True),
        # Labelling marker diameter, in SCREEN pixels (zoom-independent).
        "labelling_point_size": (float, 16.0, True),
        # Source-video pixels of forward/backward tracking disagreement that
        # costs a fill point a factor 1/e of confidence. Tighter on a small
        # frame than on a 4K one, so it is a setting rather than a constant.
        "labelling_disagreement_px": (float, 10.0, True),
        # Custom CoTracker3 weights, empty for the stock checkpoint. A model
        # fine-tuned on animal footage is a drop-in state dict, so which weights
        # to load is a user choice rather than a constant in pose_fill. Global:
        # it is a property of the machine's models, not of one dataset.
        "labelling_cotracker_checkpoint": (str, "", True),
        # Where landmark world (cm) coordinates were last imported from (the
        # Calibrate tab's "Load coordinates…"). Global: one layout file serves
        # many sessions, and re-importing per session is the workflow.
        "calibration_coords_path": (str, "", True),
        # Point detection (Detect tab): which detector and how it is tuned. The
        # detections themselves are derived data cached next to the video, and
        # what each detector label *means* is project data in the anchor
        # sidecar — neither belongs here.
        "detect_detector": (str, "apriltag", True),
        # A share of GOOD_DECISION_MARGIN: 0.3 is a decode margin of 30, which
        # sits in the empty band between spurious reads (2–13) and real ones
        # (~119). There is no tag family setting — one family is detected.
        "detect_quality_min": (float, 0.3, True),
        # Which AprilTag family, from pose_detect.TAG_FAMILIES. Shared with the
        # tag sheet on purpose: printing one family and then looking for another
        # is the mistake worth designing out, so there is one setting, not two.
        "detect_tag_family": (str, "tag36h11", True),
        # quad_decimate is local because it trades speed against how big the
        # tags are in THIS footage; pupil-apriltags' own default of 2.0 halves
        # the effective tag size, which is why the default here is 1.0.
        "detect_quad_decimate": (float, 1.0, True, SCOPE_LOCAL),
        "detect_decode_sharpening": (float, 0.25, True),
        "detect_tag_corners": (bool, False, True),
        # Printing the tags (Tools ▸ Print tag sheet…). Page setup is a property
        # of the *printer*, so it is global; the camera figures the minimum tag
        # size is computed from describe THIS rig, so they are local. The rows of
        # a sheet are not settings at all.
        "tag_sheet_page": (str, "A4", True),
        "tag_sheet_margin_mm": (float, 10.0, True),
        "tag_sheet_labels": (bool, True, True),
        # 4 mm rather than 3: tag36h11 is 8 modules to DICT_4X4's 6, so the same
        # camera needs a third more paper for the same pixels per module.
        "tag_sheet_tag_mm": (float, 4.0, True),
        "tag_sheet_camera_width_px": (int, 0, True, SCOPE_LOCAL),
        "tag_sheet_fov_mm": (float, 0.0, True, SCOPE_LOCAL),
        # Plotting
        "ymin": (float | None, None, True),
        "ymax": (float | None, None, True),
        "spec_ymin": (float | None, None, True),
        "spec_ymax": (float | None, None, True),
        "ready": (bool, False, False),
        "downsample_factor_used": (int | None, None, False),
        "nfft": (int, 256, True),
        "hop_frac": (float, 0.5, True),
        "vmin_db": (float, -120.0, True),
        "vmax_db": (float, -20.0, True),
        "buffer_multiplier": (float, 5.0, True),
        "percentile_ylim": (float, 99.5, True),
        "space_plot_type": (str, "Layers", True, SCOPE_LOCAL),
        "space_feature": (str | None, None, True),
        "space_dim": (str | None, None, True),
        "space_color": (str | None, None, True),
        "space_x_axis": (str | None, None, True),
        "space_y_axis": (str | None, None, True),
        "space_z_axis": (str | None, None, True),
        "space_3d": (bool, False, True),
        "space_percentile_xyzlim": (float, 100.0, True),
        # Space plots: show only ± this many seconds of trajectory around the
        # time marker (0 = the whole window). Sliding window during playback.
        "space_window_s": (float, 0.0, True),
        "space_marker_visible": (bool, True, True),
        "space_confidence_filter": (bool, False, True),
        "space_confidence_threshold": (float, 0.6, True),
        "space_limit_to_window": (
            bool,
            False,
            False,
        ),  # May confuse user, better not keep saved.
        "space_lock_axes": (bool, False, False),
        "space_sync_views": (bool, True, True),
        "space_hide_zeros": (bool, False, True),
        "space_show_references": (bool, True, True),
        "space_library_geometry": (str | None, None, True, SCOPE_LOCAL),
        "primary_camera": (str | None, None, True),
        "primary_camera_previous": (str | None, None, False),
        "extra_cameras": (list[str], [], True),
        "lock_axes": (bool, False, False),
        "zen_mode": (bool, False, False),
        "spec_colormap": (str, "CET-R4", True),
        "spec_levels_mode": (str, "auto", True),
        # All checkbox states for dimension combos (e.g., {"keypoint": True, "space": False})
        "all_checkbox_states": (dict[str, bool], {}, True),
        # Camera views pinned to an individual, keyed by the view's layout
        # key ("primary", or an extra view's dock key). Feature panels keep
        # their pin in their own panel_state instead.
        "camera_individuals": (dict[str, str], {}, True, SCOPE_LOCAL),
        # The subject a new label lands on, announced for the bottom bar.
        # Computed by selected_individual(); never saved.
        "labelling_subject": (str | None, None, False),
        # Audio processing
        "audio_cp_hop_length_ms": (float, 5.0, True),
        "audio_cp_min_level_db": (float, -70.0, True),
        "audio_cp_min_syllable_length_s": (float, 0.02, True),
        "audio_cp_silence_threshold": (float, 0.1, True),
        "show_changepoints": (bool, True, True),
        "plot_has_changepoints": (bool, False, False),
        # label_drawing_armed: a label class is selected and the next plot
        # click places a boundary. Mirrored from LabelsWidget so the plots can
        # tell a closing click from a double-click-to-autoscale. Never saved.
        "label_drawing_armed": (bool, False, False),
        "apply_changepoint_correction": (bool, True, True),
        "cp_step_purge": (bool, True, True),
        "cp_step_stitch": (bool, True, True),
        "cp_step_snap": (bool, True, True),
        "cp_step_purge_after": (bool, True, True),
        "automatic_min_label_length_s": (float, 1e-3, True),
        "automatic_stitch_gap_s": (float, 0.0, True),
        "remote_backup_enabled": (bool, False, True),
        "remote_backup_path": (str | None, None, True),
        "remote_backup_mode": (str, "timestamp", True),
        "remote_path_depth": (int, 0, True),
        # Envelope / energy (general, used by both heatmap and overlay)
        "energy_metric": (str, "energy_lowpass", True),
        "env_rate": (float, 2000.0, True),
        "env_cutoff": (float, 500.0, True),
        "freq_cutoffs_min": (float, 500.0, True),
        "freq_cutoffs_max": (float, 10000.0, True),
        "smooth_win": (float, 2.0, True),
        "band_env_min": (float, 300.0, True),
        "band_env_max": (float, 6000.0, True),
        "band_env_rate": (float, 1000.0, True),
        "ava_min_freq": (float, 30000.0, True),
        "ava_max_freq": (float, 110000.0, True),
        "ava_smoothing_timescale": (float, 0.007, True),
        "ava_use_softmax_amp": (bool, True, True),
        # Heatmap-specific display
        "heatmap_exclusion_percentile": (float, 98.0, True),
        "heatmap_colormap": (str, "RdBu_r", True),
        "heatmap_normalization": (str, "per_channel", True),
        # Firing rate
        "fr_bin_size": (float, 0.01, True),
        "fr_sigma": (float, 2.0, True),
        # Changepoint correction
        "cp_min_label_length_s": (float, 0.05, True),
        "cp_stitch_gap_len_s": (float, 0.015, True),
        "cp_max_expansion_s": (float, 0.05, True),
        "cp_max_shrink_s": (float, 0.05, True),
        "cp_label_thresholds": (dict, {}, True),
        # Function params cache (dialog_function_params.py)
        "function_params_cache": (dict, {}, True),
    }

    #: Saveable state keys that hold a filesystem path, mapped to what must
    #: exist for the value to be usable ("file" | "dir" | "any").  Settings
    #: files travel between machines and drives, so :func:`sanitize_path_state`
    #: drops the ones that name nothing here as they are read — an absent
    #: folder must leave the field empty, not feed the media resolvers a path
    #: that makes every trial report a missing video/pose/audio file.
    PATH_VARS: dict[str, str] = {
        "metadata_path": "file",
        "labels_import_path": "file",
        "nwb_file_path": "file",
        "video_folder": "dir",
        "audio_folder": "dir",
        "pose_folder": "dir",
        "ephys_path": "any",
        "neurons_path": "any",
        "image_paths": "file",
        "last_browse_dir": "dir",
        "project_path": "dir",
        "remote_backup_path": "dir",
        "labelling_cotracker_checkpoint": "file",
        "calibration_coords_path": "file",
    }

    @classmethod
    def get_meta(cls, key):
        if key not in cls.VARS:
            raise KeyError(f"No metadata for key: {key}")
        value = cls.VARS[key]
        if len(value) == 3:
            type_hint, default, save = value
            scope = cls.SCOPE_GLOBAL
            return type_hint, default, save, scope
        type_hint, default, save, scope = value
        return type_hint, default, save, scope

    @classmethod
    def get_default(cls, key):
        return cls.get_meta(key)[1]

    @classmethod
    def get_type(cls, key):
        return cls.get_meta(key)[0]

    @classmethod
    def saveable_attributes(cls, scope: str | None = None) -> set[str]:
        attrs = set()
        for key in cls.VARS:
            _, _, save, key_scope = cls.get_meta(key)
            if not save:
                continue
            if scope is None or scope == key_scope:
                attrs.add(key)
        return attrs


class ObservableAppState(QObject):
    """State container with change notifications and computed properties."""

    # Signals for state changes (auto-derive signal type from type hint)
    for var in AppStateSpec.VARS:
        type_hint, _, _, _ = AppStateSpec.get_meta(var)
        locals()[f"{var}_changed"] = Signal(get_signal_type(type_hint))

    trial_changed = Signal()
    #: Some label's ``labeling_method`` changed (or a whole trial's): the
    #: trial colouring in the navigation combo and bottom bar re-reads the
    #: per-trial verdict. Emitted by the curation panel, never by the store.
    curation_changed = Signal()
    GLOBAL_SETTINGS_FILENAME = "gui_settings.yaml"
    LOCAL_SETTINGS_FILENAME = "local_settings.yaml"
    SETTINGS_DIRNAME = ".ethograph"
    _TIME_REFRESH_KEYS = {"ds", "dt", "video", "video_path", "audio_path"}

    def __init__(self, yaml_path: str | None = None, auto_save_interval: int = 10000):
        super().__init__()
        object.__setattr__(self, "_values", {})
        for var in AppStateSpec.VARS:
            _, default, _, _ = AppStateSpec.get_meta(var)
            self._values[var] = default

        # Path settings read from YAML that name nothing on this machine
        # (see AppStateSpec.PATH_VARS). They are kept out of the live state but
        # written back on save, so an unplugged drive does not permanently
        # erase the folder from the settings file.
        object.__setattr__(self, "_unavailable_paths", {})

        self.audio_source_map: dict[str, tuple[str, int]] = {}
        # mic device label -> ordered audio_source_map keys (one per channel)
        self.audio_mic_channels: dict[str, list[str]] = {}
        self.ephys_source_map: dict[
            str, tuple[str, str, int]
        ] = {}  # filepath, neo_stream_id, channel_idx, e.g.("/data/session.rhd", "1", 0)).
        self.ephys_stream_sel: str | None = None
        self._suspend_local_autoload = False
        self._all_labels_df: pd.DataFrame | None = None
        self._label_history = LabelHistory()
        self._metadata_df: pd.DataFrame | None = None
        self._label_mappings: dict | None = None
        # Label branches have a fixed position mapping: branch 0 always draws
        # "full" (the entire plot), branch 1 always draws "top1", branch 2
        # always draws "top2" — there are never more than 3 branches. Only
        # one branch is "active" (editable by clicking/labeling) at a time;
        # each branch's overlay visibility is independent of whether it's active.
        self._active_branch: int = 0
        self._branch_shown: dict[int, bool] = {0: True}
        self._show_predictions_overlay: bool = False
        # The panel the user last clicked, whose pinned individual (if any)
        # is the one a new label is about. See selected_individual().
        self._subject_panel = None

        from ethograph.io.nwb_alignment import EmpytAlignment

        self.nwb_alignment = EmpytAlignment()

        self._yaml_path = yaml_path or "gui_settings.yaml"
        self._auto_save_timer = QTimer()
        self._auto_save_timer.timeout.connect(self.save_to_yaml)
        self._auto_save_timer.start(auto_save_interval)

    @property
    def video_fps(self) -> float | None:
        """Frame rate of the primary camera's video.

        The loaded view wins over the stored stream rate: it carries the rate
        probed from the file actually being decoded, while the stored rate can
        still describe the previously selected camera — and every time↔frame
        conversion in the GUI runs through here, so a stale rate desyncs the
        primary view from the marker and from the other camera views.
        """
        sync = getattr(self, "video", None)
        fps = getattr(getattr(sync, "view", None), "fps", 0.0)
        if fps:
            return float(fps)
        return self.nwb_alignment.get_stream_rate("video", self.primary_camera)

    @property
    def sel_attrs(self) -> dict:
        """
        Return all attributes ending with _sel as a dict.
        """
        result = {}
        for attr in dir(self):
            if attr.endswith("_sel"):
                value = getattr(self, attr, None)
                if not callable(value):
                    result[attr] = value
        return result

    @property
    def active_label_ids(self) -> set[int] | None:
        """Return label IDs belonging to any branch currently shown as an overlay.

        Returns None when no mappings are loaded (meaning all IDs allowed).
        This gates which existing labels can be clicked/selected/displayed —
        it is independent of which branch is *editable* (see
        :attr:`editable_label_ids`).
        """
        mappings = self._label_mappings
        if not mappings:
            return None
        shown = self._shown_branches
        return {lid for lid, data in mappings.items() if isinstance(lid, int) and data.get("branch", 0) in shown}

    @property
    def _shown_branches(self) -> set[int]:
        """Set of branch indices whose visibility checkbox is currently on."""
        return {b for b, shown in self._branch_shown.items() if shown}

    @property
    def editable_label_ids(self) -> set[int] | None:
        """Label IDs belonging to the currently active (editable) branch.

        New labels are only ever drawn into the active branch; labels of
        every other branch must never be trimmed/overwritten by it,
        regardless of whether those other branches are currently shown.
        """
        mappings = self._label_mappings
        if not mappings:
            return None
        return {
            lid
            for lid, data in mappings.items()
            if isinstance(lid, int) and data.get("branch", 0) == self._active_branch
        }

    @property
    def trial_bounds(self) -> TimeRange | None:
        """Time range for the current trial, sourced from TrialVideoBounds.trial_range."""
        alignment = getattr(self, "trial_alignment", None)
        if alignment is not None:
            return alignment.trial_range
        return None

    # --- Display-basis authority -------------------------------------------
    # The plot x-axis speaks exactly one clock at a time, decided here and
    # nowhere else. Every conversion between the axis and per-trial storage
    # (labels, video frames, per-trial data) goes through to_display /
    # from_display; hand-rolling `trial_offset + t` anywhere else recreates
    # the three-clock drift this authority exists to end.

    @property
    def display_basis(self) -> str:
        """``"session"`` or ``"trial"`` — the clock of the plot x-axis.

        Session-absolute only when the slider scope is the whole session AND
        navigation isn't showing a label/sequence window (those windows are
        built from trial-relative label onsets, so they are trial-basis even
        under session scope).
        """
        scope = getattr(self, "slider_scope", "trial")
        mode = getattr(self, "navigate_mode", "trial")
        if scope == "session" and mode not in ("label", "sequence"):
            return "session"
        return "trial"

    def to_display(self, trial_id, t_rel: float) -> float:
        """Trial-relative time in *trial_id* → the plot axis's clock."""
        sc = getattr(self, "source_collection", None)
        if self.display_basis == "session" and sc is not None:
            return sc.to_session(trial_id, t_rel)
        return t_rel

    def from_display(self, t_display: float, *, strict: bool = False) -> tuple[Any, float] | None:
        """Plot-axis time → ``(trial_id, trial-relative time)``.

        In session basis the trial is found under *t_display*
        (``strict=True`` → ``None`` in inter-trial gaps, for label placement;
        ``strict=False`` snaps to the closest trial). In trial basis the time
        belongs to the current trial verbatim.
        """
        sc = getattr(self, "source_collection", None)
        if self.display_basis == "session" and sc is not None:
            return sc.to_trial(t_display, strict=strict)
        return getattr(self, "trials_sel", None), t_display

    @property
    def before_s(self) -> float:
        mode = getattr(self, "navigate_mode", "trial")
        return self._values.get(f"before_s_{mode}", 0.0)

    @property
    def after_s(self) -> float:
        mode = getattr(self, "navigate_mode", "trial")
        return self._values.get(f"after_s_{mode}", 0.0)

    @property
    def view_span(self) -> float:
        if self.get_with_default("xlim_mode") == "fixed":
            return self.get_with_default("fixed_window_s")
        return self.before_s + self.after_s

    @property
    def window_bounds(self) -> TimeRange | None:
        """Core data range — the actual trial/label/sequence extent without padding.

        Plots use this for x-axis limits and zoom constraints.
        The padded ``restrict_window.time_range`` is for slider/scroll limits.
        In fixed x-limits mode the core window is just a viewport that slides
        over the full scope extent, so the extent is the data range.
        """
        rw = getattr(self, "restrict_window", None)
        if rw is not None:
            return rw.time_range if rw.mode == "fixed" else rw.core_range
        return self.trial_bounds

    @property
    def padded_bounds(self) -> TimeRange | None:
        """Padded display range including before/after context.

        Use for scroll/slider limits where the user should be able to pan
        beyond the core trial range.
        """
        rw = getattr(self, "restrict_window", None)
        if rw is not None:
            return rw.time_range
        return self.trial_bounds

    @property
    def time_coord(self) -> xr.DataArray | None:
        """Get the time coordinate for the currently selected features."""
        ds = getattr(self, "ds", None)
        features_sel = getattr(self, "features_sel", None)
        if ds is not None and features_sel in ds.data_vars:
            return eto.get_time_coord(ds[features_sel])
        return None

    def get_with_default(self, key):
        """Return value from app state, or default from AppStateSpec if None."""
        value = getattr(self, key, None)
        if value is None:
            value = AppStateSpec.get_default(key)
        return value

    def get_ephys_source(self) -> tuple[str | None, str, int]:
        """Get ephys file path, stream_id, and channel index from current ephys_stream_sel.

        Returns (ephys_path, stream_id, channel_idx) tuple. Uses ephys_source_map
        to resolve the display name.
        """
        import os

        stream_sel = getattr(self, "ephys_stream_sel", None)
        if not stream_sel or not self.ephys_source_map:
            return None, "0", 0

        entry = self.ephys_source_map.get(stream_sel)
        if entry is None:
            return None, "0", 0

        filename, stream_id, channel_idx = entry

        if not filename:
            return None, stream_id, channel_idx

        if os.path.isabs(filename):
            ephys_path = os.path.normpath(filename)
        else:
            base_ephys_path = getattr(self, "ephys_path", None)
            if not base_ephys_path:
                return None, stream_id, channel_idx
            ephys_path = os.path.normpath(os.path.join(os.path.dirname(base_ephys_path), filename))

        return ephys_path, stream_id, channel_idx

    def playback_mic_selection(self) -> str | None:
        """audio_source_map key that drives playback: the last-clicked panel's
        pin, else the global mic. Only returns a key valid in the current
        dataset (a stale key from a prior dataset is ignored)."""
        for key in (self.playback_mic_key, getattr(self, "mics_sel", None)):
            if key and key in self.audio_source_map:
                return key
        return None

    def playback_audio_label(self) -> str | None:
        """Compact indicator label ``ChN: first-10-chars…`` (full name in tooltip)."""
        key = self.playback_mic_selection()
        if not key:
            return None
        mic_file, ch = self.audio_source_map.get(key, (key, 0))
        name = str(mic_file)
        short = name[:10] + ("…" if len(name) > 10 else "")
        return f"Ch{ch + 1}: {short}"

    def playback_audio_tooltip(self) -> str | None:
        """Full channel description for the indicator's hover tooltip."""
        key = self.playback_mic_selection()
        if not key:
            return None
        mic_file, ch = self.audio_source_map.get(key, (key, 0))
        return f"Playback channel {ch + 1} — {mic_file}"

    def has_playback_audio(self) -> bool:
        """Whether an audio channel is available to play back."""
        return bool(getattr(self, "has_audio", False) or self.audio_path or self.playback_mic_selection())

    def effective_playback_mode(self) -> str:
        """Resolve ``playback_mode`` to a concrete mode for the current data.

        ``auto`` follows audio presence; an explicit ``synced`` with no audio
        degrades to ``smooth`` (there is nothing to synchronise to).
        """
        from .app_constants import PLAYBACK_MODE_AUTO, PLAYBACK_MODE_SMOOTH, PLAYBACK_MODE_SYNCED

        mode = self.playback_mode
        has_audio = self.has_playback_audio()
        if mode == PLAYBACK_MODE_AUTO:
            return PLAYBACK_MODE_SYNCED if has_audio else PLAYBACK_MODE_SMOOTH
        if mode == PLAYBACK_MODE_SYNCED and not has_audio:
            return PLAYBACK_MODE_SMOOTH
        return mode

    def get_audio_source(self, mic_name: str | None = None) -> tuple[str | None, int]:
        """Get audio file path and channel index for a mic selection.

        *mic_name* overrides the global ``mics_sel`` (used by audio panels
        pinned to one mic/channel). Returns (audio_path, channel_idx) tuple.
        Uses audio_source_map to resolve the display name to
        (mic_file, channel_idx).
        """
        mics_sel = mic_name or getattr(self, "mics_sel", None)
        if not mics_sel or not self.audio_source_map:
            return None, 0

        entry = self.audio_source_map.get(mics_sel)
        if entry is None:
            # A key that is not in the map is a stale pin (panel recreated, map
            # rebuilt for another trial), never a path. Display keys carry the
            # channel suffix — "mic1.wav (Ch 3)" — so reading one as a file name
            # sent the audio loader after a file that cannot exist.
            logger.warning("Unknown audio source '%s' — no audio for this panel.", mics_sel)
            return None, 0

        mic_file, channel_idx = entry
        if not mic_file:
            return None, channel_idx

        audio_folder = getattr(self, "audio_folder", None)

        # Try resolve via nwb_alignment (ImageSeries path → fallback folder)
        for mic_dev in self.nwb_alignment.mics:
            trial = getattr(self, "trials_sel", None)
            if trial is None:
                break
            media = self.nwb_alignment.get_media(trial, "audio", mic_dev)
            if media and (media == mic_file or Path(media).name == mic_file):
                resolved = self.nwb_alignment.resolve_media_path(
                    trial,
                    "audio",
                    device=mic_dev,
                    fallback_folder=audio_folder,
                )
                if resolved:
                    return resolved, channel_idx
            if not media:
                # Stream-based alignments (drag & drop) have no trials-table
                # filename columns — match the ImageSeries file directly.
                resolved = self.nwb_alignment.resolve_media_path(
                    trial,
                    "audio",
                    device=mic_dev,
                    fallback_folder=audio_folder,
                )
                if resolved and (mic_file == str(mic_dev) or Path(resolved).name == mic_file):
                    return resolved, channel_idx

        # Direct fallback
        if audio_folder:
            import os

            path = os.path.normpath(os.path.join(audio_folder, mic_file))
            return path, channel_idx

        return None, channel_idx

    def __getattr__(self, name):
        # Check for class attributes/properties first
        cls = type(self)
        if hasattr(cls, name):
            attr = getattr(cls, name)
            # If it's a property, use its getter
            if hasattr(attr, "__get__"):
                return attr.__get__(self)
            return attr
        if name in AppStateSpec.VARS:
            return self._values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in (
            "time",
            "_values",
            "settings",
            "_yaml_path",
            "_auto_save_timer",
            "navigation_widget",
            "lineplot",
            "audio_source_map",
            "audio_mic_channels",
            "ephys_source_map",
            "ephys_stream_sel",
            "_suspend_local_autoload",
            "_layout_snapshot_provider",
            "_all_labels_df",
            "_label_history",
            "_metadata_df",
            "_label_mappings",
            "_active_branch",
            "_branch_shown",
            "_show_predictions_overlay",
        ):
            super().__setattr__(name, value)
            return

        if name in AppStateSpec.VARS:
            type_hint = AppStateSpec.get_type(name)
            if not check_type(value, type_hint):
                raise TypeError(f"{name}: expected {type_hint}, got {type(value).__name__} = {value!r}")

            old_value = self._values.get(name)
            self._values[name] = value

            signal = getattr(self, f"{name}_changed", None)
            if signal:
                try:
                    changed = bool(old_value != value)
                except (ValueError, TypeError):
                    changed = old_value is not value
                if changed:
                    signal.emit(value)

            if name == "nc_file_path" and not self._suspend_local_autoload:
                self.load_local_settings()

            # Auto-sync nwb_file_path → nwb_alignment
            # Skip if alignment was already set by the data loader (e.g. remote NWB)
            if name == "nwb_file_path":
                existing = getattr(self, "nwb_alignment", None)
                if existing is None or getattr(existing, "_path", None) is not None:
                    from ethograph.io.nwb_alignment import make_nwb_alignment

                    self.nwb_alignment = make_nwb_alignment(value)

            if name == "metadata_path":
                # Metadata is purely additive (joined on its trial column) —
                # trial timing always comes from the alignment NWB, never from
                # a metadata file.
                if value:
                    metadata_df, resolved_path = load_metadata_df(
                        source_path=self.nc_file_path,
                        metadata_path=value,
                        nwb_alignment=self.nwb_alignment,
                        trial_ids=getattr(self, "trials", None) or None,
                    )
                    self._values[name] = resolved_path or value
                    self.metadata_df = metadata_df
                else:
                    self.metadata_df = None

            return

        super().__setattr__(name, value)

    # --- Dynamic _sel variables ---
    def get_ds_kwargs(self):
        ds_kwargs = {}

        for dim in self.ds.dims:
            if "time" in dim:
                continue
            attr_name = f"{dim}_sel"
            if not hasattr(self, attr_name):
                continue

            output = getattr(self, attr_name)
            if output is None or output in ["", "None"]:
                continue

            # Check if dim has coords and determine appropriate type
            if dim in self.ds.coords:
                coord_dtype = self.ds.coords[dim].dtype
                if coord_dtype.kind in ("i", "u"):
                    ds_kwargs[dim] = int(output)
                else:
                    ds_kwargs[dim] = str(output)
            else:
                # Dim without coord - assume integer index
                ds_kwargs[dim] = int(output)

        return ds_kwargs

    def get_selections(self) -> dict[str, str]:
        """Backend-agnostic selection dict from combo *_sel attributes.

        Uses ``data_loader.dims`` when available (pynapple path),
        falls back to ``get_ds_kwargs()`` for pure xarray.
        """
        store = getattr(self, "data_loader", None)
        if store is None:
            if self.ds is None:
                return {}
            return self.get_ds_kwargs()

        selections: dict[str, str] = {}
        for dim_name in store.dims:
            attr_name = f"{dim_name}_sel"
            if not hasattr(self, attr_name):
                continue
            val = getattr(self, attr_name)
            if val is not None and val not in ("", "None"):
                selections[dim_name] = str(val)
        return selections

    def sidebar_individual(self) -> str | None:
        """The sidebar's individual, in whichever spelling this dataset's dim
        uses (movement is singular, older wizard data plural).

        Combos are named after their dim, so the ``*_sel`` attribute differs
        per dataset — read it through here instead of hardcoding a spelling.
        Deliberately NOT named ``individual_sel``: that is the dynamically
        generated combo attribute for a singular dim, and a method of the same
        name is silently replaced by it.
        """
        for name in INDIVIDUAL_DIMS:
            val = getattr(self, f"{name}_sel", None)
            if val is not None and val not in ("", "None"):
                return str(val)
        return None

    def pinned_individual_of(self, panel) -> str | None:
        """The individual *panel* is pinned to, or ``None`` when it follows the sidebar.

        A pin is the panel's own ``pinned_individual`` (a feature plot keeps
        it in its ``panel_state``, a camera view as a plain attribute). A
        panel is in exactly one of the two modes, and its title says which
        (:func:`panel_mode_suffix`).
        """
        if panel is None:
            return None
        pin = getattr(panel, "pinned_individual", None)
        return str(pin) if pin not in (None, "", "None") else None

    def panel_individual(self, panel) -> str | None:
        """Whose data and labels *panel* shows: its pin, else the sidebar's individual."""
        return self.pinned_individual_of(panel) or self.sidebar_individual()

    def selected_individual(self) -> str | None:
        """The subject a new label is about: the last clicked panel's individual.

        A pinned panel that was clicked makes its individual the labelling
        subject; an unpinned one (or no click yet) leaves it at the sidebar's.
        Every label path reads the actor through here.
        """
        return self.panel_individual(getattr(self, "_subject_panel", None))

    def panel_mode_suffix(self, panel) -> str:
        """``" — bird_2 (pinned)"`` / ``" — bird_1 (sidebar)"`` for a panel title; empty with one individual.

        Every panel is either pinned or following the sidebar, and the
        title is where that is visible: change the combo and the
        ``(sidebar)`` panels move while the ``(pinned)`` ones stay.
        """
        if len(self.label_individuals()) < 2:
            return ""
        pinned = self.pinned_individual_of(panel)
        individual = pinned or self.sidebar_individual()
        if individual is None:
            return ""
        return f" \u2014 {individual} ({'pinned' if pinned else 'sidebar'})"

    def set_subject_panel(self, panel) -> None:
        """Record the clicked panel; the labelling subject follows its pin."""
        self._subject_panel = panel
        self.refresh_labelling_subject()

    def refresh_labelling_subject(self) -> None:
        """Re-announce the labelling subject (``labelling_subject_changed``) after a pin or sidebar change."""
        self.labelling_subject = self.selected_individual()

    def selected_receiver(self) -> str:
        """The receiver of the behaviour being labelled, ``""`` for none.

        The counterpart of :meth:`selected_individual`: the two together are
        the label subject, and only labels of that exact pair are drawn,
        hit-tested and created.
        """
        return str(getattr(self, "individual_receiver", "") or "")

    def label_individuals(self) -> list[str]:
        """Every individual that can act or receive, backend-agnostic.

        The dataset's individual dim when it has one (whatever its spelling),
        otherwise the names the labels themselves use — a session with no
        individual dimension still labels *somebody*. Falls back to a single
        ``"default"`` so the selector is never empty.
        """
        loader = getattr(self, "data_loader", None)
        catalog = getattr(loader, "catalog", None)
        if catalog is not None and catalog.individual_combo:
            values = [str(v) for v in catalog.combo_values(catalog.individual_combo)]
            if values:
                return values
        names: list[str] = []
        df = self._all_labels_df
        if df is not None and not df.empty and "individual" in df.columns:
            names = [str(v) for v in pd.unique(df["individual"].dropna())]
        return names or ["default"]

    def labels_name_our_individuals(self, df: pd.DataFrame | None) -> bool:
        """Whether *df* names individuals the way this dataset names its own.

        A pynapple session synthesises ``individual_0`` while its labels file
        says ``Crow1``: the two namings are disjoint, so filtering the overlay
        by the dataset's name would blank every trial's labels. Where they
        overlap at all the filter is honest — an individual with no labels yet
        is an empty canvas, which is exactly what labelling a second animal
        starts from.
        """
        if df is None or df.empty or "individual" not in df.columns:
            return True
        named = {str(v) for v in pd.unique(df["individual"].dropna())}
        return not named.isdisjoint(self.label_individuals())

    def key_sel_exists(self, type_key: str) -> bool:
        """Check if a key selection exists for a given type."""
        return hasattr(self, f"{type_key}_sel")

    def get_key_sel(self, type_key: str):
        """Get current value for a given info key."""
        attr_name = f"{type_key}_sel"
        return getattr(self, attr_name, None)

    def _coerce_to_list_type(self, value, reference_list: list):
        """Coerce value to match the type of items in reference_list."""
        if not reference_list:
            return value
        sample = reference_list[0]
        if isinstance(sample, int) and not isinstance(value, int):
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        return value

    def set_key_sel(self, type_key, currentValue):
        """Set current value for a given info key.

        When currentValue is None, the dimension will not be filtered in
        get_ds_kwargs(), effectively showing all values for that dimension.
        """
        if type_key == "trials" and hasattr(self, "trials") and self.trials:
            currentValue = self._coerce_to_list_type(currentValue, self.trials)

        attr_name = f"{type_key}_sel"
        prev_attr_name = f"{type_key}_sel_previous"

        current_stored_value = getattr(self, attr_name, None)
        if current_stored_value != currentValue and current_stored_value is not None:
            setattr(self, prev_attr_name, current_stored_value)

        setattr(self, attr_name, currentValue)

    def toggle_key_sel(self, type_key, data_widget):
        """Toggle between current and previous value for a given key.

        If a previous value exists, swap current and previous.
        Otherwise, cycle to the next item in the combo box.

        Special case: type_key="Audio Waveform" toggles the features
        selection to/from Audio Waveform.
        """

        attr_name = f"{type_key}_sel"
        prev_attr_name = f"{type_key}_sel_previous"

        current_value = getattr(self, attr_name, None)
        previous_value = getattr(self, prev_attr_name, None)

        if previous_value is not None:
            setattr(self, attr_name, previous_value)
            setattr(self, prev_attr_name, current_value)
            if data_widget is not None:
                self._update_combo_box(type_key, previous_value, data_widget)
        elif data_widget is not None:
            self._cycle_combo_box(type_key, data_widget)

    def cycle_key_sel(self, type_key, data_widget):
        """Cycle to the next item in the combo box for a given key."""
        if data_widget is not None:
            self._cycle_combo_box(type_key, data_widget)

    def _update_combo_box(self, type_key, new_value, data_widget):
        """Update the corresponding combo box in the UI and trigger its change signal."""
        try:
            combo = data_widget.io_widget.combos.get(type_key) or data_widget.combos.get(type_key)

            if combo is not None:
                index = find_combo_index(combo, str(new_value))
                if index < 0 and type_key == "mics":
                    for i in range(combo.count()):
                        if combo.itemText(i).startswith(str(new_value)):
                            index = i
                            break
                if index >= 0:
                    combo.setCurrentIndex(index)
        except (AttributeError, TypeError) as e:
            logger.error("Error updating combo box for %s: %s", type_key, e)

    def _cycle_combo_box(self, type_key, data_widget):
        """Cycle the combo box to the next item when no previous selection exists."""
        try:
            combo = data_widget.io_widget.combos.get(type_key) or data_widget.combos.get(type_key)
            if combo is not None and combo.count() > 1:
                next_index = (combo.currentIndex() + 1) % combo.count()
                combo.setCurrentIndex(next_index)
        except (AttributeError, TypeError) as e:
            logger.error("Error cycling combo box for %s: %s", type_key, e)

    # --- Save/Load methods ---
    PATH_SUFFIXES = ("_path", "_folder")

    def _global_settings_path(self) -> Path:
        return ethograph_home() / self.GLOBAL_SETTINGS_FILENAME

    def _local_settings_path(self) -> Path | None:
        nc_file_path = getattr(self, "nc_file_path", None)
        if not nc_file_path:
            return None
        try:
            nc_path = Path(nc_file_path)
        except (TypeError, ValueError):
            return None
        return nc_path.parent / self.SETTINGS_DIRNAME / self.LOCAL_SETTINGS_FILENAME

    def _yaml_read(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # os.replace() on Windows needs exclusive access to the destination; a
    # second EthoGraph instance briefly opening the same global settings
    # file for its own autosave is enough to raise WinError 32. Retry a
    # few times before giving up.
    _REPLACE_RETRIES = 5
    _REPLACE_RETRY_DELAY_S = 0.05

    def _yaml_write(self, path: Path, state_dict: dict) -> None:
        # Atomic replace: a crash mid-write must never truncate the settings
        # file the next launch will load.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(self._to_native(state_dict), f, default_flow_style=False, sort_keys=False)
        for attempt in range(self._REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == self._REPLACE_RETRIES - 1:
                    raise
                time.sleep(self._REPLACE_RETRY_DELAY_S)

    def _to_native(self, value):
        """Recursively convert numpy types to native Python types for YAML serialization."""
        if isinstance(value, dict):
            return {key: self._to_native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_native(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        return value

    def get_saveable_state_dict(self, scope: str | None = None) -> dict:
        state_dict = {}
        for attr in AppStateSpec.saveable_attributes(scope=scope):
            value = self._values.get(attr)
            if attr in AppStateSpec.PATH_VARS and is_throwaway_path(value):
                continue
            if value is not None and isinstance(value, (str, float, int, bool)):
                state_dict[attr] = self._to_native(value)
            elif isinstance(value, dict) and value:
                state_dict[attr] = value

        for attr, value in self._unavailable_paths.items():
            if attr in state_dict or self._values.get(attr) is not None:
                continue
            if attr in AppStateSpec.saveable_attributes(scope=scope):
                state_dict[attr] = value

        if scope in (None, AppStateSpec.SCOPE_LOCAL):
            for attr in dir(self):
                if attr.endswith("_sel") or attr.endswith("_sel_previous"):
                    try:
                        value = getattr(self, attr)
                        if not callable(value) and value is not None:
                            if isinstance(value, (str, float, int, bool)):
                                state_dict[attr] = self._to_native(value)
                    except (AttributeError, TypeError) as exc:
                        logger.error("Error accessing %s: %s", attr, exc)
        return state_dict

    def _sort_state_dict(self, state_dict: dict) -> dict:
        """Sort state dict by category: paths, bools, _sel, strings, numbers, nested dicts."""

        def _category_key(item):
            key, value = item
            is_nested = isinstance(value, dict)
            is_path = any(key.endswith(s) for s in self.PATH_SUFFIXES)
            is_sel = key.endswith("_sel") or key.endswith("_sel_previous")
            is_bool = isinstance(value, bool)
            is_str = isinstance(value, str)

            if is_nested:
                order = 5
            elif is_path:
                order = 0
            elif is_bool:
                order = 2
            elif is_sel:
                order = 1
            elif is_str:
                order = 3
            else:
                order = 4
            return (order, key)

        return dict(sorted(state_dict.items(), key=_category_key))

    def print_state(self) -> None:
        """Print all simple-typed app state vars, grouped by category."""
        _PRINTABLE = (str, int, float, bool, list, dict, type(None))
        _CATEGORY_LABELS = {
            0: "Paths",
            1: "Selections",
            2: "Booleans",
            3: "Strings",
            4: "Numbers",
            5: "Lists/Dicts",
            6: "None",
        }

        def _category_key(item):
            key, value = item
            if isinstance(value, (dict, list)):
                return 5
            if value is None:
                return 6
            if any(key.endswith(s) for s in self.PATH_SUFFIXES):
                return 0
            if key.endswith("_sel") or key.endswith("_sel_previous"):
                return 1
            if isinstance(value, bool):
                return 2
            if isinstance(value, str):
                return 3
            return 4

        state = {}
        for attr in self._values:
            value = self._values[attr]
            if isinstance(value, _PRINTABLE):
                state[attr] = self._to_native(value) if isinstance(value, (str, int, float, bool)) else value
        # Also include dynamic _sel attributes
        for attr in dir(self):
            if attr.endswith("_sel") or attr.endswith("_sel_previous"):
                try:
                    value = getattr(self, attr)
                    if not callable(value) and isinstance(value, _PRINTABLE):
                        state[attr] = self._to_native(value) if isinstance(value, (str, int, float, bool)) else value
                except (AttributeError, TypeError):
                    pass

        current_cat = None
        for key, value in sorted(state.items(), key=lambda item: (_category_key(item), item[0])):
            cat = _category_key((key, value))
            if cat != current_cat:
                print(f"\n{'=' * 50}")
                print(f"  {_CATEGORY_LABELS[cat]}")
                print(f"{'=' * 50}")
                current_cat = cat

            if isinstance(value, list) and len(value) > 10:
                value = f"{value[:10]}... (total {len(value)} items)"

            print(f"  {key}: {value}")

    def load_from_dict(self, state_dict: dict):
        # Migrate legacy keys
        if "restrict_mode" in state_dict:
            val = state_dict.pop("restrict_mode")
            if val == "video":
                val = "trial"
            state_dict.setdefault("navigate_mode", val)

        # Migrate old before_s/after_s or restrict_extra_t0/t1 → per-category
        old_before = state_dict.pop("before_s", state_dict.pop("restrict_extra_t0", None))
        old_after = state_dict.pop("after_s", state_dict.pop("restrict_extra_t1", None))
        if old_before is not None:
            for suffix in ("trial", "label", "sequence"):
                state_dict.setdefault(f"before_s_{suffix}", old_before)
        if old_after is not None:
            for suffix in ("trial", "label", "sequence"):
                state_dict.setdefault(f"after_s_{suffix}", old_after)

        state_dict.pop("window_size", None)

        cleaned = sanitize_path_state(state_dict, AppStateSpec.PATH_VARS)
        self._unavailable_paths.update({k: v for k, v in state_dict.items() if cleaned.get(k) != v})
        state_dict = cleaned

        self._suspend_local_autoload = True
        try:
            for key, value in state_dict.items():
                if value is None:
                    continue
                if key in AppStateSpec.VARS or key.endswith("_sel") or key.endswith("_sel_previous"):
                    setattr(self, key, value)
        finally:
            self._suspend_local_autoload = False

    def load_local_settings(self) -> bool:
        try:
            local_path = self._local_settings_path()
            if local_path is None:
                return False
            # Missing per-dataset paths belong to the dataset being left, never
            # to the one being loaded.
            for key in AppStateSpec.saveable_attributes(scope=AppStateSpec.SCOPE_LOCAL):
                self._unavailable_paths.pop(key, None)
            state_dict = self._yaml_read(local_path)
            if not state_dict:
                return False
            # Drop global-preference keys left in a local file by older
            # versions (e.g. import_labels_nc_data before it went global) —
            # a per-dataset leftover must never override a global preference.
            global_keys = AppStateSpec.saveable_attributes(scope=AppStateSpec.SCOPE_GLOBAL)
            state_dict = {k: v for k, v in state_dict.items() if k not in global_keys}
            if not state_dict:
                return False
            self.load_from_dict(state_dict)
            logger.info("Local state loaded from %s", local_path)
            return True
        except (OSError, yaml.YAMLError) as e:
            logger.error("Error loading local state from YAML: %s", e)
            return False

    def save_to_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                # Backward-compatible single-file save.
                path = Path(yaml_path)
                state_dict = self._sort_state_dict(self.get_saveable_state_dict())
                self._yaml_write(path, state_dict)
                return True

        except (OSError, yaml.YAMLError):
            logger.exception("Error saving state to %s", yaml_path)
            return False

        # Refresh panel/window layout snapshots (set by MetaWidget) so the
        # periodic auto-save always persists the live arrangement. A snapshot
        # failure must not block saving the rest of the state (this runs in
        # the auto-save QTimer slot, where an uncaught exception would
        # silently kill every save).
        provider = getattr(self, "_layout_snapshot_provider", None)
        if provider is not None:
            try:
                provider()
            except Exception:
                logger.exception("Layout snapshot failed; saving state without a layout refresh")

        ok = True
        try:
            global_path = self._global_settings_path()
            global_state = self._sort_state_dict(self.get_saveable_state_dict(scope=AppStateSpec.SCOPE_GLOBAL))
            self._yaml_write(global_path, global_state)
        except (OSError, yaml.YAMLError):
            logger.exception("Error saving global state to %s", self._global_settings_path())
            ok = False

        try:
            local_path = self._local_settings_path()
            if local_path is not None:
                local_state = self._sort_state_dict(self.get_saveable_state_dict(scope=AppStateSpec.SCOPE_LOCAL))
                self._yaml_write(local_path, local_state)
        except (OSError, yaml.YAMLError):
            logger.exception("Error saving local state to %s", self._local_settings_path())
            ok = False

        return ok

    def load_from_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                path = Path(yaml_path)
                if not path.exists():
                    logger.warning("YAML file %s not found, using defaults", path)
                    return False
                state_dict = self._yaml_read(path)
                self.load_from_dict(state_dict)
                logger.info("State loaded from %s", path)
                return True

            loaded_any = False

            global_path = self._global_settings_path()
            global_state = self._yaml_read(global_path)
            # Drop per-dataset keys left in the global file by older versions
            # (e.g. navigate_mode/slider_scope) — they must never become a
            # sticky default that follows the user into the next dataset.
            local_keys = AppStateSpec.saveable_attributes(scope=AppStateSpec.SCOPE_LOCAL)
            global_state = {k: v for k, v in global_state.items() if k not in local_keys}
            if global_state:
                self.load_from_dict(global_state)
                logger.info("Global state loaded from %s", global_path)
                loaded_any = True

            if self.load_local_settings():
                loaded_any = True

            if not loaded_any:
                logger.warning("No settings YAML found, using defaults")
            return loaded_any
        except (OSError, yaml.YAMLError) as e:
            logger.error("Error loading state from YAML: %s", e)
            return False

    def delete_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                p = Path(yaml_path)
                if p.exists():
                    p.unlink()
                    logger.info("Deleted YAML file %s", yaml_path)
                    return True
                logger.warning("YAML file %s does not exist", yaml_path)
                return False

            deleted_any = False
            global_path = self._global_settings_path()
            if global_path.exists():
                global_path.unlink()
                logger.info("Deleted YAML file %s", global_path)
                deleted_any = True

            local_path = self._local_settings_path()
            if local_path is not None and local_path.exists():
                local_path.unlink()
                logger.info("Deleted YAML file %s", local_path)
                deleted_any = True

            if not deleted_any:
                logger.warning("No YAML settings files found to delete")
            return deleted_any
        except OSError as e:
            logger.error("Error deleting YAML file: %s", e)
            return False

    def reset_local_settings(self) -> bool:
        """Reset this dataset's SCOPE_LOCAL settings to defaults.

        Deletes ``.ethograph/local_settings.yaml`` AND resets the in-memory
        local vars — deleting the file alone would be undone by the next
        auto-save, which writes local settings from live state.
        """
        path = self._local_settings_path()
        if path is None:
            return False
        deleted = self.delete_yaml(str(path))
        for var in AppStateSpec.saveable_attributes(scope=AppStateSpec.SCOPE_LOCAL):
            setattr(self, var, AppStateSpec.get_default(var))
        for attr in list(dir(self)):
            if attr.endswith("_sel") or attr.endswith("_sel_previous"):
                try:
                    delattr(self, attr)
                except AttributeError:
                    pass
        return deleted

    def stop_auto_save(self):
        if self._auto_save_timer.isActive():
            self._auto_save_timer.stop()
            self.save_to_yaml()

    # --- Interval label helpers ---
    def get_trial_intervals(self, trial) -> pd.DataFrame:
        return get_trial_from_tsv(self._all_labels_df, trial)

    def get_display_intervals(self) -> pd.DataFrame | None:
        """Labels as they belong on the plot axis.

        Trial basis: the current trial's rows verbatim (``label_intervals``).
        Session basis: EVERY trial's rows, onsets shifted to session time —
        the axis spans all trials, so all labels belong on it. Storage stays
        trial-relative; this is a read-only view for drawing/hit-testing.
        """
        if self.display_basis != "session":
            return self.label_intervals
        df = self._all_labels_df
        sc = getattr(self, "source_collection", None)
        if df is None or df.empty or sc is None:
            return self.label_intervals
        out = df.copy()
        offsets = {tid: sc.to_session(tid, 0.0) for tid in out["trial"].unique()}
        shift = out["trial"].map(offsets).astype(float)
        out["onset_s"] = out["onset_s"] + shift
        out["offset_s"] = out["offset_s"] + shift
        return out.reset_index(drop=True)

    def set_trial_intervals(self, trial, df: pd.DataFrame) -> None:
        self._all_labels_df = set_trial_in_tsv(self._all_labels_df, trial, df)
        nav = getattr(self, "navigation_widget", None)
        if nav is not None and hasattr(nav, "on_labels_changed"):
            nav.on_labels_changed()

    # --- Label undo history ---
    def record_label_edit(self, description: str, trial=None) -> None:
        """Snapshot the labels of *trial* before an edit, for ``Ctrl+Z``.

        Call this once at the top of a handler, before anything mutates the
        labels: everything the handler goes on to do (trimming, sliver purge,
        changepoint correction) then belongs to the same undo step.
        """
        if trial is None:
            trial = getattr(self, "trials_sel", None)
        if trial is None:
            return
        self._label_history.record(self._all_labels_df, trial, description)

    def undo_label_edit(self) -> LabelEdit | None:
        """Take back the last recorded label edit; returns what it took back.

        ``label_intervals`` is re-read from the restored table when the edit
        belongs to the trial on screen — an undo can land on another trial, and
        the caller is the one that knows whether to navigate there.
        """
        result = self._label_history.undo(self._all_labels_df)
        if result is None:
            return None
        self._all_labels_df, edit = result
        current = getattr(self, "trials_sel", None)
        if current is not None and str(edit.trial) == str(current):
            self.label_intervals = get_trial_from_tsv(self._all_labels_df, edit.trial)
        nav = getattr(self, "navigation_widget", None)
        if nav is not None and hasattr(nav, "on_labels_changed"):
            nav.on_labels_changed()
        return edit

    def can_undo_labels(self) -> bool:
        return len(self._label_history) > 0

    def clear_label_history(self) -> None:
        """Drop the undo stack — the table it describes is being replaced."""
        self._label_history.clear()

    def get_trial_meta(self, trial) -> dict:
        return get_trial_meta(self._all_labels_df, trial)

    def set_trial_meta_attr(self, trial, key: str, value) -> None:
        self._all_labels_df = set_trial_meta_attr(self._all_labels_df, trial, key, value)

    # --- Curation (labels/curation.py) ---
    def curation_scope(self) -> set[int] | None:
        """The label classes curation acts on; ``None`` means every class."""
        ids = self.curation_label_ids
        return {int(i) for i in ids} if ids else None

    def trial_curation_status(self) -> dict[str, bool]:
        """``{str(trial): curated?}`` over the trials the table shows."""
        return trial_curation_status(self._all_labels_df, self.trials or [])

    def trial_is_curated(self, trial) -> bool:
        """Whether no label of *trial* is still automated."""
        return trial_curation_status(self._all_labels_df, [trial])[str(trial)]

    def replace_all_labels(self, df: pd.DataFrame | None) -> None:
        """Swap in a table whose rows were restamped (curation), re-reading
        the current trial's view. Row identities are unchanged, so the undo
        stack is left alone — a method change is not an edit it tracks."""
        self._all_labels_df = df
        current = getattr(self, "trials_sel", None)
        if current is not None:
            self.label_intervals = get_trial_from_tsv(df, current)

    def get_global_meta_attr(self, key: str, default=0):
        """Check if ALL trials with labels have a meta attr set to truthy."""
        if self._all_labels_df is None or self._all_labels_df.empty:
            return default
        if not self.trials:
            return default
        trials_with_labels = set(self._all_labels_df["trial"].unique())
        if not trials_with_labels:
            return default
        for trial in self.trials:
            if trial not in trials_with_labels:
                continue  # no labels → nothing to correct, skip
            meta = get_trial_meta(self._all_labels_df, trial)
            if not meta.get(key, 0):
                return default
        return 1

    def set_global_meta_attr(self, key: str, value) -> None:
        """Set a meta attr on ALL trials."""
        for trial in self.trials:
            self._all_labels_df = set_trial_meta_attr(self._all_labels_df, trial, key, value)

    def _get_downsampled_suffix(self) -> str:
        if self.downsample_factor_used:
            return f"_downsampled_{self.downsample_factor_used}x"
        return ""

    def labels_file_path(self) -> Path | None:
        """The file :meth:`save_labels` would write the labels to."""
        if self._labels_file_path and Path(self._labels_file_path).exists():
            return Path(self._labels_file_path)
        if not self.nc_file_path:
            return None
        return labels_tsv_path(Path(self.nc_file_path), self._get_downsampled_suffix())

    def labels_dirty(self) -> bool:
        """Whether the labels in memory really differ from the ones on disk.

        ``changes_saved`` over-reports: it is cleared by anything that *might*
        have touched labels, so a session that only edited trial metadata was
        still asked to save labels it never changed. The flag stays the cheap
        path; the file decides.
        """
        if self.changes_saved:
            return False
        if self._all_labels_df is None:
            return False
        path = self.labels_file_path()
        if path is None:
            return not self._all_labels_df.empty  # nowhere to compare against
        try:
            on_disk = load_labels_tsv(path)
            if labels_equal(self._all_labels_df, on_disk):
                return False
            logger.info(
                "Labels differ from %s (%d rows in memory, %d on disk)",
                path.name,
                len(self._all_labels_df),
                len(on_disk),
            )
            return True
        except (OSError, ValueError) as e:
            # An unreadable/invalid labels file is exactly when the user wants
            # to be offered the save.
            logger.warning("Could not read %s to compare labels: %s", path.name, e)
            return True

    def save_labels(self, remote_path: str | None = None, remote_mode: str | None = None) -> None:
        """Save labels to active file (canonical or predictions TSV) + local backup + optional remote backup.

        Parameters
        ----------
        remote_path : str, optional
            Folder path for remote backup. Falls back to ``self.remote_backup_path``.
        remote_mode : str, optional
            "timestamp", "overwrite", or "git". Falls back to ``self.remote_backup_mode``.
        """
        if self._all_labels_df is None:
            return

        effective_remote_path = remote_path or self.remote_backup_path or None
        effective_remote_mode = remote_mode if remote_mode is not None else self.remote_backup_mode

        nc_path = Path(self.nc_file_path)
        suffix = self._get_downsampled_suffix()
        stem = f"{nc_path.stem}{suffix}"

        # Enrich with computed columns (duration, sequence, global timing, trial attrs)
        from ethograph.labels.export import enrich_labels_df

        keep_attrs = self.trial_conditions if self.trial_conditions else []
        enriched = enrich_labels_df(
            self._all_labels_df,
            nwb_alignment=self.nwb_alignment,
            keep_attrs=keep_attrs,
            dt=self.dt,
            metadata_df=self.metadata_df,
        )
        save_df = enriched if not enriched.empty else self._all_labels_df

        # 1. Primary file: use _labels_file_path if set (predictions/custom), otherwise canonical
        primary_tsv = self.labels_file_path()
        assert primary_tsv is not None  # nc_file_path resolved above
        save_labels_tsv(primary_tsv, save_df)

        # 2. Local backup with timestamp
        backup_dir = nc_path.parent / "labels" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_labels_tsv(backup_dir / f"{stem}_labels_{timestamp}.tsv", save_df)

        # 3. Remote backup (optional)
        # remote_path_depth controls how many parent folders to mirror inside remote_root:
        #   0 = flat (Trial_data_labels.tsv)
        #   1 = behav/Trial_data_labels.tsv
        #   2 = ses-000/behav/Trial_data_labels.tsv  (etc.)
        if self.remote_backup_enabled and effective_remote_path:
            remote_root = Path(effective_remote_path)
            depth = self.remote_path_depth
            if depth > 0:
                parent_parts = nc_path.parent.parts[1:]  # strip drive / leading '/'
                mirror_parts = parent_parts[max(0, len(parent_parts) - depth) :]
                remote_dir = remote_root.joinpath(*mirror_parts)
            else:
                remote_dir = remote_root
            remote_dir.mkdir(parents=True, exist_ok=True)
            remote_file = remote_dir / f"{stem}_labels.tsv"
            if effective_remote_mode in ("overwrite", "git"):
                save_labels_tsv(remote_file, save_df)
                if effective_remote_mode == "git":
                    auto_git_commit(remote_file)
            else:
                save_labels_tsv(remote_dir / f"{stem}_labels_{timestamp}.tsv", save_df)

        notify(f"Saved labels: {primary_tsv.name}")
        self.changes_saved = True
