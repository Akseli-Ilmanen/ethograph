# CLAUDE.md

## Working style

- Implement, don't propose. Don't wait for "yes go".
- Write idiomatic, typed Python (mypy/ruff clean). Prefer clean design patterns over cleverness.
- Imports: sorted stdlib → third-party → local, explicit (never `from x import *`), never inside a function (only exception: circular imports).
- Comments only where the logic isn't obvious — which should be rare. **Never remove human-authored comments** (TODO/FIXME/NOTE/explanatory); only remove comments you added yourself.
- **Fail fast**: bugs (wrong type, missing key, unexpected `None`) crash; runtime conditions (missing file, bad user input) are handled. Never `try/except` into a silent `None`. Catch broad exceptions only at the outermost GUI boundary.
- Test/debug scripts live in `tests/`, never the project root. Prefix ad-hoc debug scripts `_test_` so pytest skips them.
- Docs/docstrings: don't name individuals (Poppy, Freddy, Ivy).
- Claude Code may change any file in this repo.

### How to maintain this file

This file describes **the architecture**, not the history of how it got here, and not the feature you just built. Every line here is loaded into every session, so a line has to pay for itself.

- **Finishing a task never adds to this file.** A new setting, a new dialog, a fixed bug, a layout rule, a scope decision — none of these belong here. Encode the invariant in a test; the test is enforced, prose is not. If you feel the urge to write "X is now Y", write `tests/test_unit/test_x.py` instead and stop.
- **Only a genuine architectural change touches this file**: a new subsystem, a new layer between existing ones, or a rule that changes how every module must be written. And when it does, the edit is preferably a deletion.
- **No war-stories.** No "the old code did X, which caused Y, so never do X". No "was removed because". No lists of exceptions. State the invariant positively in one line, or not at all.
- **Prefer deletion over accumulation.** A rule the code already makes obvious (by name, type, or signature) is a restatement — remove it.

### What earns a test

A test earns its place if **its failure would surprise you**. If the only way it can go red is "someone deliberately moved this", it is a change-detector, not a test.

**Keep**: invariant guards (a rule stated here), contract guards (two things must agree and nothing forces them to), branching logic (clock conversions, filters, resolution priority), regression guards (it broke in a way nobody saw coming).

**Cut**: tombstones (`..._relocated`, `..._has_no_y`), structural pins (menu-title lists, widget parentage, `hasattr`), unasserted actions, restatements (the expected value is recomputed the way the code computes it).

**A test never pays for data it does not read.** Take the cheapest fixture that can still fail: `gui` for widget wiring, `qapp` for one widget, none for Qt-free logic. Behaviour that holds for every dataset is tested once, not once per dataset.

## Project Overview

EthoGraph is a GUI for labelling start/stop times of animal movements, paired with a workflow using action-segmentation transformers to predict segments. It loads NetCDF/NWB/pynapple datasets and displays synchronized video/audio/ephys.

```python
import ethograph as eto
dt = eto.open("data.nc"); dt = eto.from_datasets([ds1, ds2])
time = eto.get_time_coord(da); data, filt = eto.sel_valid(da, kwargs)
```

## Hard Rules

**Never hardcode rates.** No frame rates, sample rates, or 1-second fallbacks. Use source metadata (`video.fps`, `ImageSeries.rate`, audio rate) or user settings; if unknown, raise or return `None`.

**Never hardcode a device.** `resolve_device()` picks CUDA → MPS → CPU.

**Never special-case a `dataset_key` in GUI code** — put per-dataset settings in `DATASETS` metadata.

**Never call `QFileDialog` directly** — go through `gui/file_dialogs.py` (wizard tabs excepted: they hold no `app_state`).

**Never hand-roll `trial_offset + t`.** Conversions go through `app_state.to_display` / `from_display`.

## File Structure

```
ethograph/__init__.py         # Public API

ethograph/gui/
    app_state.py              # AppStateSpec + ObservableAppState
    data_loader.py            # Dataset dispatch (.nc / .nwb / pynapple / DANDI)
    data_sources.py, plot_sources
    pose_render.py            # Pose loading (NWB + movement), PoseDisplayManager
    pose_annotate.py          # KeypointStore + movement export
    pose_fill.py              # Fill backends: Spline, OpticalFlow (+ private CoTracker3)
    pose_refine.py            # PosePAL: CoTracker3 + test-time refinement, GPU only
    pose_detect.py            # Detect stage: AprilTag tag36h11, assignment learning
    pose_detect_preview.py    # PreviewPanel — what the detector sees on this frame
    pose_tagsheet.py          # Tag sheet: layout maths + vector PDF/SVG/printer output
    dialog_tag_sheet.py       # Print tag sheet… (cover page pre-recording tools only)
    pose_edit_mixin.py        # KeypointLabelMode (canvas anchor editing)
    dialog_pose_labelling.py, dialog_skeleton_editor.py
    widgets_curation.py       # Curation section of the Labels tab
    dialog_label_gridview.py  # Label grid view: frames at label times
    dialog_video_grid.py      # Video grid: clips of one label class side by side
    dialog_label_table.py     # Label table: every trial's rows as a spreadsheet
    dialog_pose_refinement.py # Refine imported poses: correct DLC/SLEAP files, _refined copies
    dialog_onset_model.py     # Model menu: GradBoost onset-detector train/predict dialogs
    dialog_curation_workflow.py # Saved curation routines: step editor + WorkflowRunner
    dialog_box_labelling.py   # Box labelling (OCTRON): OCTRON's dock over every open camera; one time index, individual-major clicks
    box_annotate.py           # SamSession — OCTRON's SAM predictor headless, one per camera video
    box_overlay.py            # Mask + prompt-point overlay on a pygfx view, BoxLabelMode (right click = exclude)
    plots_base.py             # BasePlot, PanelStateMixin
    plots_container.py        # UnifiedPanelContainer
    plots_{audiotrace,spectrogram,ephystrace,lineplot,heatmap,raster,space}.py
    plots_timeseriessource.py # Re-exports io/time_model.py (compat)
    plots_console.py          # ConsolePanel — REPL over what the clicked panel plots
    plots_radial.py           # RadialPlot — compass arrow for a heading at the current time
    label_drawing_mixin.py, video_sync.py, video_manager.py
    widgets_meta.py           # MetaWidget — creates + wires everything
    widgets_data.py           # DataWidget — central orchestrator
    widgets_{io,labels,navigation,changepoints,ephys,plot_settings,transform}.py
    widget_intervals.py, right_context.py, main_window.py, top_bar.py, cover_page.py
    table_filter.py           # Funnel-header column filters (ephys + keypoint tables)
    file_dialogs.py           # Browse dialogs that remember the last folder
    nwb_alignment.py, shortcuts.py

ethograph/labels/
    intervals.py              # Interval ops, mapping loaders
    ml.py                     # Dense↔interval conversion only
    onset_model.py            # GradBoost point-event onset detection (train/predict core)
    curation.py               # labeling_method transitions, per-trial verdicts, review queues (Qt-free)
    workflow.py               # Curation workflows: the STEP_KINDS contract + the YAML store (Qt-free)
    tsv_store.py, predictions.py, crowsetta_format.py, converters.py, export.py
    octron_project.py         # The OCTRON project folder in OCTRON's own layout (hash folders, organizer JSON, octron.yaml, CLI commands)

ethograph/io/
    catalog.py                # DataCatalog + XarrayLoader/PynappleLoader, pose discovery
    derived.py                # TracedArray + DerivedFeature + DerivedLoader (console features)
    trialtree.py              # TrialTree (xr.DataTree subclass)
    time_model.py             # TimeRange, RestrictionWindow, TimeSource, SourceCollection
    time_sources.py           # XarrayTrialSource, PynappleSource
    audio_extract.py          # Container audio (AAC/MP4) → cached WAV, resolve_audio_path
    nc_drop.py                # A dropped .nc is always features (several concat on a `camera` dim); a pose overlay only in its video's pixels
    schema.py                 # Variable schema: kind / is_egocentric / normalise attrs (movement#978)
    dataset.py, validation.py, pynapple.py, metadata_table.py, ephys_loader.py

ethograph/video_features/     # S3D video features: plan.py (seconds → frames per video rate), frames.py (streaming PyAV), extract.py

ethograph/features/
    columns.py                # FeatureColumn + enumerate_columns/extract_features — the one input-layout definition
    geometry.py               # Egocentric/pairwise/heading/angle/area features (xarray in → xarray out)
    changepoints.py           # Changepoint detection + add_changepoint_features
    movement.py, preprocessing.py, energy.py, oscillatory.py, neural.py, audio_changepoints.py

ethograph/segment/            # Segmentation pipeline (docs: docs/add_to_docs_later/segment/, ADRs: notes/adr/)
    config.py                 # SegmentConfig dataclasses: YAML + base: merge + dotlist overrides
    sessions.py               # Headless session opening (GUI loaders, no Qt), trial filter, media paths
    samples.py                # ClassTable, ColumnLayout, one (trial, individual) sample
    materialise.py            # Stage 1: the literature layout + index.tsv/columns.yaml/classes.yaml
    preprocess.py, augment.py, dataset.py, losses.py, metrics.py, postprocess.py, plotting.py
    train.py                  # Stage 2: assign_roles (the split), stats, run dir, compare_runs
    inference.py              # Stage 3: prediction sets ({stem}_predictions.tsv + _probs.npz)
    search.py                 # Workflow stage 1: Optuna on the val split → searches/{name}/best.yaml
    crossval.py               # Workflow stage 2: leave-one-session-out folds, or n_folds trial folds for a one-session project
    windows.py                # Windows tiled over a trial-less epoch (sleep) → a second alignment listed as sessions[].alignment
    project.py                # Project — the one entry point: a config + a method per stage
    video_features.py         # S3D: a folder of videos or a config's sessions, + merge into a session
    models/__init__.py        # Architecture registry + contract
    models/vendored.py        # DLC2Action adapters: the registry contract + upstream's YAML defaults
    models/rnn.py             # Ours: bidirectional GRU/LSTM baseline, defaults in models/config/rnn.yaml
    dlc2action/               # Vendored AGPL: model/, loss/, config/ — see its NOTICE.md; excluded from ruff/mypy; index of every vendored tree: THIRD_PARTY_NOTICES.md

ethograph/spot/               # Pixel point-event spotting (E2E-Spot); docs: docs/add_to_docs_later/spot/
    config.py                 # SpotConfig: ClipConfig (durations → stride/clip_len/dilate_len), no features section
    dataset.py                # Stage 1 export: sessions → frames/ + E2E-Spot's {split}.json
    confidence.py             # focus/ratio — the shape statistic, measured
    predict.py                # A run's scores → labels TSV + onset_curves
    project.py                # Project — materialise / train / inference / cross_validate, the segment pipeline's words
    inference.py              # Stage 4: best epoch by the sweep → test_e2e.py → labels TSV + onset_curves per session
    stream.py                 # Inference decodes the video straight into the model (rolling one-window buffer, JPEG round trip in memory; only the stride grid is converted + prepared, once each; the forward is a replayed CUDA graph — tests/test_unit/test_spot_stream.py TestRollingBuffer); never the frame folder
    metrics.py                # evaluate(): a run's chosen epoch on a labelled split → test_metrics.yaml (misses, error in ms, hit rate per tolerance)
    vendored.py               # Locating and driving the E2E-Spot clone
    msagsm.py                 # MultiScaleGatedShift from the paper, on the BSD GSM; `rny008_msagsm` (dilations are durations)
    features.py               # The pose side: the listed features per trial (features/*.npz + features.json), the z-scored block (features/block/) for the student
    pose_model.py             # PoseSpotter: features → multi-scale shift blocks → bi-GRU → (B,T,K+1)
    teacher.py                # Stage 1 of the distillation recipe: train the pose teacher on features/
    pose_batch.py             # Headless: fill every labelled clip's sidecar → <video>.keypoints.nc → merge onto the trial clock
    # distillation lives in the vendored trainer (--stage 2/3, scripts/spot_windows_compat.patch); Project.distil() drives it

ethograph/utils/              # io.py, xr_utils.py (sel_valid, get_time_coord), sequences.py, device.py (resolve_device)
ethograph/utils/system_check.py # Linux preflight for the GUI wheels' system libs (`ethograph check`)
ethograph/skeleton/           # PrecomputedRenderer, SkeletonState, config.py, shapes.py
```

## Architecture

### Two data-source layers

**Rendering** (`io/plot_sources`) — `PlotSource` protocol (`name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`): `FileSource`, `XarraySource`, `PynappleSource`. `WindowedBuffer` caches wider than the viewport; per-plot buffers specialise.

**Navigation** (`io/time_model.py` + `time_sources.py`) — session-level time metadata via `TimeSource`; `SourceCollection` is the registry. Uses only `time_range`, never `get_data()`.

### TrialTree

`TrialTree` inherits `xr.DataTree`; each trial is a child node with `attrs["trial"]`. API: `dt.trial(id)`, `dt.itrial(idx)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`, `dt.get_label_dt()`. Session metadata (trial timing, media paths, FPS, offsets) comes from `app_state.nwb_alignment`, not the tree.

### State: `app_state.py`

`AppStateSpec` is a type-checked spec; `ObservableAppState` auto-generates a Qt signal per variable, exposes dynamic `*_sel` attributes, and auto-saves to YAML.

- **`SCOPE_LOCAL`** (per-dataset `local_settings.yaml`): anything that only means something for one dataset — the plot x-extent (`fixed_window_s`, `navigate_mode`, `slider_scope`), names of its individuals/keypoints, its paths.
- **`SCOPE_GLOBAL`** (`gui_settings.yaml`): viewing habits and plain preferences that should follow the user to the next dataset.
- **Path settings are validated on load** (`AppStateSpec.PATH_VARS`, `sanitize_path_state()`); a path from another machine is dropped, not restored, and remembered in `_unavailable_paths`. Covered by `tests/test_unit/test_path_sanitize.py`.

### Time model + navigation

`TimeRange` (immutable), `TimeSource` (Protocol), `SourceCollection` (`union_range`, `session_range`, `sources_at(t)`, `trial_range`, `find_trial`, `trial_offset`; built in `data_loader.py`), `RestrictionWindow` (mode `"session"|"trial"|"label"|"sequence"`).

`app_state.window_bounds` (falls back to `trial_bounds`) drives every plot's x-limits; `app_state.session_time_range` is the full extent. Navigation modes: Session / Trial / Label / Sequence (pattern e.g. `1-2-3-5`, `utils/sequences.py`).

**The one clock rule** (docs: `advanced/time_slider_trial_session.md`): `app_state.display_basis` is `"session"` iff `slider_scope == "session"` and `navigate_mode` is not label/sequence; else `"trial"`. It is the authority on which clock the plot axis speaks. `VideoSync.frame_to_time`/`time_to_frame` speak the display clock; audio indexes through `audio_display_offset()`, ephys through `ephys_display_offset()`; labels draw from `app_state.get_display_intervals()`. Session scope is disabled for multi-trial xarray. Covered by `tests/test_unit/test_time_basis.py` + `tests/test_integration/test_session_scope.py`.

### Catalog + loader: `catalog.py`

`DataCatalog` declares features, dimensions and streams; built by `catalog_from_xarray()` / `catalog_from_pynapple()`. Features are auto-detected (any `data_var` with a time dim; any pynapple `Tsd`/`TsdFrame`/`TsdTensor`).

- **A combo is named after the dim it selects from.** `INDIVIDUAL_DIMS` lists the spellings most-preferred first; read whichever a dataset uses via `catalog.individual_combo` / `app_state.selected_individual()` — never hardcode one.
- `DataLoader` Protocol: `select(feature, selections, t0, t1) → PlotData`, always in **display coordinates**. Loaders ignore dims a feature lacks.
- `PanelStateMixin._sanitize_selections` guarantees **at most one free multi-value dim**. A saved `panel_layout` is untrusted: `MetaWidget.apply_saved_panel_layout` catches failures and rebuilds defaults.
- `XarrayLoader` wraps `xr.Dataset` (`update_ds()` per trial). `PynappleLoader` holds no trial state; pynapple objects live in absolute session time and the loader bridges to display time via a display-offset provider. NWB loads via pynapple.
- **A pynapple feature's columns are a dim named in exactly one place**: `_column_axes()` returns a `ColumnAxis(dim, labels)` per feature, read by `catalog_from_pynapple`, `feature_dims()` and `select()`.

### Alignment: `nwb_alignment.py`

`.nwb` sources are read/edited directly (`edit_nwb`); `.ethograph/alignment.nwb` sidecars exist only for non-NWB sources. `NWBAlignment(path)`: `get_stream_rate`, `resolve_media_path`, `stream_offset_for_trial`.

- **Trial timing has one source: the alignment NWB trials table.** The one sanctioned bridge: the cover page offers, with consent, to convert a pynapple trials IntervalSet into `.ethograph/alignment.nwb`.
- **Metadata is purely additive** — a tabular file joined on `trial`. Resolution priority (`_resolve_alignment`): explicit `alignment_path` → NWB source trials → sidecar TSV. Edited in the trials table (`widget_trials.py`), written back to the source it was read from (`io/metadata_edit.py`). Covered by `tests/test_unit/test_metadata_edit.py`.
- **Media filenames are the alignment's, joined onto the metadata table's right-hand side** (`metadata_table.attach_media_columns`): one `{stream}_{device}` column per device, read-only (never a `condition_columns` entry) and never written to a metadata file (`stored_columns`), so the conditions stay on the left and a filename filter is still one of the trials table's filters. Covered by `tests/test_unit/test_metadata_table.py`.
- **The trials table's filters are the one trial filter, and every operation runs over `app_state.trials`.** Navigation, curation queues/grids, onset-model training/inference and cross-trial label operations read that list and nothing else; no dialog gets a metadata filter of its own. The label filter (Tools ▸ Find label inconsistencies…, `utils/sequences.trials_matching_labels`) is a second, session-only slot intersected after the column filters. Covered by `tests/test_unit/test_label_matching.py` + `test_label_inconsistencies.py`.
- **Drag & drop = single-trial loading** (`cover_page.py`, `classify_files()`).
- **Video-container audio is decoded once to a cached WAV** (`io/audio_extract.py`, `resolve_audio_path()`). Never add a video extension to `AUDIO_EXTENSIONS`. Covered by `tests/test_unit/test_audio_extract.py`.

### Pose rendering

Two paths unified into `PoseRenderData`: `load_pose_from_file()` (movement) and `load_pose_from_nwb_direct()` (lazy HDF5). Filtering acts on masks (`data_not_nan`, `shown`) — it never recreates layers.

**The video overlays any catalog feature in the camera's pixels** (`io/overlay_source.py`): a time dim, a `space` dim with x/y, at most keypoint/individual dims, values inside the frame. `position` is the default; the Overlay source combo lists the rest, console results included. Companions by name: `confidence` filters, `shape` means boxes. The loaded dataset is drawn first; a pose file is read only when the dataset has nothing in that camera's pixels. Covered by `tests/test_unit/test_overlay_source.py`.

**Colour encodes one axis, chosen by the user**: `app_state.pose_color_by` ∈ `{"keypoint", "individual"}`; text labels carry the other axis. The same setting drives the labelling canvas. Colour is the only identity channel.

### Keypoint labelling + fill

Label a few frames by clicking the video, let a point tracker fill the rest. **Tools ▸ Keypoint labelling…** (`DataWidget.open_keypoint_labelling()`). Scope: one camera, one trial; `TrialTree` datasets are not supported.

<<<<<<< HEAD
**A static keypoint is labelled once and present on every frame** (`KeypointStore.static_keypoints`, the tree's *Static* column): the canvas, the fill's input, `pin_static` on its output and the export all read it through `anchor_positions`/`_overlay_static`; placing it again moves it everywhere, and a fresh clip of the same camera is seeded from the last saved sidecar. Covered by `tests/test_unit/test_static_keypoints.py`. **The binding design rules live in `docs/source/advanced/keypoint_labelling/`** — store/provenance, fill backends, PosePAL, detection, calibration, tag printing, canvas editing, keys, points table, export. **Read those pages before editing `gui/pose_*.py`, `dialog_pose_labelling.py`, `dialog_tag_sheet.py` or `table_filter.py`.**
=======
**The binding design rules live in `docs/source/advanced/keypoint_labelling/`. Read those pages before editing `gui/pose_*.py`, `dialog_pose_labelling.py`, `dialog_tag_sheet.py` or `table_filter.py`.**
>>>>>>> b5e49dad75c10cbaab9adb0b13f42c4012b844c9

### Skeleton visualization

`ethograph/skeleton/`: `PrecomputedRenderer` turns a movement poses Dataset into a Vectors layer; `SkeletonState`/`config.py` manage connections/colors/widths. Colour precedence: `skeleton_config_override` (user-drawn) > NWB config recoloured with `skeleton_base_color`. **Anchored shapes** (`skeleton/shapes.py`): templates bound to ≥2 control points, stored under `"shapes"` in the skeleton config.

### Panels are layout instances — no per-plot-type toggles

Panels are instances created via the layout (`SourcePopup`, ➕ / Shift+N, or drop), never on/off toggles; dropping an already-shown source creates another instance.

- **Panels are dock widgets** (`UnifiedPanelContainer` hosts a nested `QMainWindow`). Minimums in the media/plots split are deliberately slivers; never raise a minimum to get a default proportion. Covered by `tests/test_integration/test_split_ratio.py`.
- **Layout persistence is automatic**: `app_state.panel_layout` is SCOPE_LOCAL, `app_state.window_state` SCOPE_GLOBAL.
- **Audio panels, extra camera views, line plots, space plots, radial plots are all instances** with `add_*`/`remove_*` on their owner; anything that must follow the trial iterates the live views. The heatmap is a fixed singleton toggled by `set_feature_view(...)`. Covered by `tests/test_integration/test_camera_trial_follow.py` + `test_camera_panel_identity.py`.
- **One canonical feature list**: `catalog.feature_choices()` feeds combo, popup and panel creation — never `ds.data_vars`.
- **Feature plots render only from their own `panel_state`** (`PanelStateMixin`). Never make a feature plot read `app_state.features_sel` for rendering.
- **Space reference geometry** comes from `~/.ethograph/defaults/config/space/*.yaml`; **templates ship layouts via `local_settings.yaml`** (never overwriting a local file).

### Console panel + derived features

The one singleton panel (`plot_container.console_panel`). **The console binds what a panel renders**, not what backs it. Numpy, not xarray: `TracedArray` (`io/derived.py`) records the expression graph; each assigned name becomes a feature via `DerivedLoader.register()`. `stack(a, b)` makes a `(T, D)` feature usable as a space plot. Derived features live for one trial. The console rebinds only on `plot_container.panel_content_changed`.

### Plot system

All plots inherit `BasePlot` (pyqtgraph `PlotWidget`). `UnifiedPanelContainer` holds all panels and links x-axes.

### Video sync

`NapariVideoSync.frame_to_time(frame)` / `time_to_frame(t)` speak the **display clock**; all widgets use these — never raw `frame / fps`. Video decode stays clipped to the current trial (`trial_frame_window()` in `io/time_model.py`).

- **The playhead is always exact; only the video quantizes.** Audio is sliced at true sub-frame bounds; the video shows the nearest frame.
- **The audio-master clock (`gui/audio_clock.py`) is DAC-anchored, chunked and rate-bounded.** It must never lead the sound. Covered by `tests/test_unit/test_audio_clock.py` + `test_audio_playback_sync.py`.
- **The video render chain is guarded and watched** (`install_animate_guard`, `VideoSync`'s watchdog). Covered by `tests/test_unit/test_render_watchdog.py`.
- **Reloading the same file reuses the `PlotVideo`.** Covered by `tests/test_integration/test_video_reload.py`.
- **Cropping a camera is display-only**, keyed by camera name, session state. Covered by `tests/test_unit/test_video_crop.py`.

### Labels

**Storage:** TSV (`{name}_labels.tsv`) alongside the `.nc`. Columns: `onset_s, offset_s, labels (int), individual, individual_rec, event_type, confidence, labeling_method, trial, changepoint_corrected, prediction_source, n_samples`. **`onset_s`/`offset_s` are canonically trial-relative.** Label names in `mapping.txt`. In memory: `app_state._all_labels_df` (all trials) and `app_state.label_intervals` (current trial).

<<<<<<< HEAD
- **A label's subject is a pair: `individual` (actor) + `individual_rec` (the GUI calls it the receiver).** `NO_RECIPIENT` (`""`) means a solo behaviour — the column name and the backend's `recipient` vocabulary (`NO_RECIPIENT`, `individual_rec`, `subject_mask`) stay as they are; only the GUI's own terminology says "receiver". The receiver is an attribute of the label, never a second track: exclusivity (`add_interval`, overlap resolution, export) is per actor (`TRACK_COLUMNS`), stitching and row identity keep `SUBJECT_COLUMNS`, the overlay draws every receiver (tagged `→ name`) and the combo only sets the next label's; `select_subject`/`subject_mask` (`None` = "any") is the one place a filter is expressed. The GUI reads the pair from `app_state.selected_individual()` + `selected_receiver()`, never from `ds_kwargs`. A naming disjoint from the dataset's own skips the actor filter (`app_state.labels_name_our_individuals`). Covered by `tests/test_unit/test_label_intervals.py` (`TestRecipient`) + `tests/test_integration/test_individual_receiver.py`.
- **A visible label is a selectable label.** Click-selection (`_check_labels_click`) gates on `app_state.active_label_ids` (shown branches). Branch scope lives on mutation: `_delete_label`/`_edit_label` refuse a selection outside the active branch (`_refuse_foreign_branch`). Covered by `tests/test_unit/test_label_click_branches.py`.
- **Every label carries a `labeling_method`** ∈ {`manual`, `automated`, `curated`} (ndx-ethogram vocabulary; `labels/curation.py`, docs `advanced/labels/curation.md`). `add_point`/`add_interval` default to `manual`; a model passes `automated` (`predictions.py`, `dialog_onset_model.py`); a trimmed remnant keeps its row's method. The only transition the GUI performs is automated → curated (`curate_trial`/`curate_label`) — manual is never rewritten, and nothing runs backwards except an edit (→ manual) or a new prediction (→ automated). `ensure_labeling_method` reads a file without the column off its `confidence` (`< 1.0` → automated). A trial is **curated** iff none of its labels is automated (`trial_curation_status`); that verdict colours the trial combo + bottom bar (green/red, refreshed on `app_state.curation_changed`) and is written to the metadata table's `curated` column — string-valued `"yes"`/`"no"` (`CURATED_YES`/`CURATED_NO`), not `0`/`1`, so the trials table's funnel filter reads it as a categorical yes/no checklist instead of a numeric range — on a timer (`CurationPanel.sync_metadata`, `METADATA_SYNC_MS`) — never on the labelling path. Covered by `tests/test_unit/test_curation.py`.
- **Curation state is ours, and it lives in a tabular metadata file.** The sync timer runs only while `app_state.curation_active` (never saved — a dataset load disarms it; dropping label classes into the scope area or curating anything arms it via `CurationPanel.activate`), so a session that curates nothing writes nothing. Arming is the one place a metadata TSV is created: `ensure_tabular_target` (`io/metadata_edit.py`) turns an NWB target into the sidecar `{stem}_metadata.tsv`, seeded from the loaded table, and `app_state.metadata_path` then points at it. `DERIVED_COLUMNS` never reach `write_trials_metadata` — that write is in place, and for a non-NWB dataset the alignment NWB is the sole holder of the trial timing. Covered by `tests/test_unit/test_metadata_edit.py` + `tests/test_unit/test_curation_review.py` (`TestActive`).
- **Curation has one home: the Curation section of the Labels tab** (`widgets_curation.py`). *Scope* = label classes dragged out of the branch tables (`app_state.curation_label_ids`, SCOPE_LOCAL; `None` = all). *Mode* (`app_state.curation_mode`, SCOPE_LOCAL): `manual` (edits → manual, **Ctrl+C** → `curate_current_trial`), `inspect` (`trial_changed` → curate the trial, deferred a tick), `frame` (inline frame-by-frame review: queue from `build_review_queue` sorted (trial, onset), restricted to automated labels only when `frame_review_automated_only` (SCOPE_GLOBAL, on by default — a human already vouched for manual/curated boundaries), each reached via `NavigationWidget.jump_to_label_instance`; Enter commits `video.frame_to_time(current_frame)` → `from_display` — moved → manual + `HUMAN_CONFIDENCE`, unmoved → curated; Backspace deletes the whole row; B/N back/next, N curating the boundary it leaves when `curation_next_curates`). Enter/Backspace/Delete/B/N are ApplicationShortcuts installed on `_begin`, removed on `_stop`, disabled while typing; no button is default/autoDefault. The review view is `refine_window_s` (SCOPE_GLOBAL). Every label mutation in `LabelsWidget` ends with `curation_panel.note_labels_edited()`. Covered by `tests/test_unit/test_curation_review.py`.
- **Automated labels draw dotted, manual/curated solid** (`label_drawing_mixin.py`, `_method_style`; covered by `tests/test_unit/test_label_method_style.py`). Drawn items are indexed by `draw_key` (class, display onset, subject); a one-label transition goes through `plot_container.restyle_label(key, automated)` (pens swapped in place, 0 → full redraw), a whole-trial transition through `schedule_labels_redraw()`.
- **Every label carries a `confidence`**: `HUMAN_CONFIDENCE` (1.0) for a hand-placed one, the producing model's own score otherwise. It is part of `INTERVAL_COLUMNS`, so it survives every per-trial round trip; `ensure_confidence` fills it in for files written without it. Covered by `tests/test_unit/test_label_intervals.py` (`TestConfidence`).
- **The grids are review surfaces over the scope, never separate curation systems.** `dialog_label_gridview.py` (Label grid view…) prints confidence + method per tile, outlines tiles below the threshold in red, plots `ConfidenceHistogramsDialog` sharing that threshold, and exports a PDF; `dialog_video_grid.py` (Video grid…) plays clips of **one label class at a time, sorted by duration, a non-scrolling screenful at a time** (Previous/Next label, Previous/Next clips — greyed out at the ends) driven by one Play/slider spanning the longest clip, played once then stopped at its own speed, deliberately decoupled from the GUI's `playback_speed_pct` (the tick advances wall-clock × speed, `_apply_timer_interval`) (shorter clips hold their last frame; point events get a red corner marker on their frame, `marker_visible`; ←/→ pause and step one frame, `step_frame`), decoded one screenful at a time at `CLIP_MAX_SIDE` with **the next page decoding ahead** (`VideoGridPlayer.next_page` → `VideoGridDialog._prefetch_page`: `plan_clip_jobs` resolves videos on the GUI thread — the alignment NWB is not thread-safe — then `decode_clip_jobs` runs on one worker; a jump onto that page waits for it, `_await_prefetch`, never decodes twice); a window cut at the video's start/end is said on the tile (`clip_note`), as is where in the trial the label sits. Both share `LabelSetupPage` + `GridModeBar`/`TileVerdicts` (keyed by `entry_key`, so all tiles of a label share a verdict). **A double click always navigates** (jump, or `CurationPanel.start_review_at` in frame mode) — in every mode, and it undoes the toggle from the press Qt delivers first, so a jump leaves no verdict; a **single** click is the verdict `GRID_MODES` names: *curate* (Done curates the clicked) or *uncurate* (Done curates everything else; **Mark low-confidence as uncurated** pre-clicks the threshold's tiles and is enabled only in this mode — a low score argues for doubt, never approval). Done applies through `CurationPanel.curate_labels`. **Both grids filter by `labeling_method` on the shared setup page** (`GRID_METHOD_FILTERS` / `methods_for_filter`, `app_state.grid_method_filter` SCOPE_GLOBAL, mirrored by `labels/workflow.py`'s `GRID_METHOD_CHOICES`): all / manual only / curated only / manual+curated / automated — manual and curated also stay combined as one choice, since both mean a human vouched for the label, alongside letting a reviewer isolate either one. **`GridModeBar.entries_fn` returns what is on screen, and that is what every operation runs over** — the label grid's **Label** filter (`label_filter_choices`/`filter_entries`, one class at a time) therefore narrows Done, Mark-flagged, the click count and the PDF, not just the view; `labels/workflow.py`'s `GRID_MODE_CHOICES` mirrors `GRID_MODES`, and a mode it no longer offers leaves the grid on its default rather than failing. **Grid layout is a viewing habit, SCOPE_GLOBAL** — reviewing runs over many trials, so every knob here is remembered instead of reset on each open: `video_grid_point_window_s` (0.5 s), `video_grid_per_page`, `video_grid_columns`, `label_grid_sort` / `video_grid_sort` (`GRID_SORT_ORDERS` = trial | confidence_asc | confidence_desc, plus `duration` for the video grid, whose clips play together; `sort_entries` is the one implementation, applied in `LabelGridView.visible_entries` and `group_clips` so Done, the flag count and the PDF all follow the order on screen — verdicts key on the label, so reordering never moves one), `video_grid_speed_pct` (the video grid's own playback speed), `label_grid_columns`, `label_grid_window_s`, and `grid_confidence_threshold` (the "Flag confidence below" spin, shared by both grids) — each spin opens at the saved value and writes back on change. Covered by `tests/test_unit/test_label_gridview.py` + `tests/test_unit/test_video_grid.py`. **The confidence rule lives in the grids' `Histogram…` popup** (a *Confidence rule* panel above the bars — `ConfidenceHistogramsDialog(rule_controller=)` + `ConfidenceRuleController` in `dialog_label_gridview.py`, arithmetic in `labels/rescore.py`, Qt-free): rule ∈ `product | ratio | focus | peak | custom(α)` plus the window in ms (`grid_confidence_rule/_alpha/_window_ms`, SCOPE_GLOBAL); every change redraws the bars and restyles the tiles from the session's merged curves (`read_all_curves`); **Apply** re-scores only automated labels that have a curve — one `record_label_edit` per trial, then `replace_all_labels` — and closing without Apply reverts. Covered by `tests/test_unit/test_rescore.py` + `test_confidence_rule.py`.
- **A curation workflow is a recording of the GUI, never a second way of doing anything** (`labels/workflow.py` Qt-free + `gui/dialog_curation_workflow.py`, **Workflows…** in the Curation section and Model ▸ Curation workflows…, docs `advanced/labels/workflows.md`). A workflow is a name and an ordered list of `WorkflowStep(kind, params)`, stored as YAML in `~/.ethograph/defaults/workflows/{name}.yaml`. `STEP_KINDS` is the one contract: each `StepKind` declares its `ParamSpec` list (type, default, bounds, choices), so the editor's form, the YAML and `_HANDLERS` all read one declaration — a kind with no handler raises at import. Every handler drives the widget a user would: `TrialsWidget.apply_column_filters` (filters keyed by **column name**, so a workflow moves between datasets; an unmatched column is reported and skipped, never fatal), `dialog_onset_model.predict_onsets`, `CurationPanel.set_scope`/`open_grid_view`/`open_video_grid`/`start_review`/`curate_visible_trials` (whose manual twin is the Curation section's **Curate visible trials…** button — `confirm=True` there, because one click would otherwise vouch for labels nobody looked at; a recorded step does not ask). **Curating is not an undoable label edit**: `_commit` records no `record_label_edit` snapshot and the history is per-trial anyway, so `Ctrl+Z` never takes a curation back — which is what the bulk confirmation says out loud. Covered by `tests/test_unit/test_curation_review.py` (`TestTrialLevel`), `IOWidget._save_labels`. `WorkflowStep.value` falls back to the declared default, so a stored workflow survives a kind gaining a parameter. The runner walks on the event loop (never blocking): an `interactive` step hands over and resumes on the dialog's `finished` or `CurationPanel.review_finished`. **Exactly one thing is carried between steps** — the label ids the last prediction wrote, so a `scope` step with an empty class list reviews what this run produced. Covered by `tests/test_unit/test_curation_workflow.py`.
- **LightGBM onset models predict point events only, at most one per target class per trial** (`labels/onset_model.py` + `dialog_onset_model.py`, Model menu, docs `docs/add_to_docs_later/labels/onset_model.md`, confidence across models in `docs/add_to_docs_later/confidence.md`). One model holds `config.targets` (`{label_id: name}`) and fits one binary classifier per target over shared features. A model lives in `~/.ethograph/defaults/runs/lightgbm/{name}`: `config.yaml` (frozen at creation — it defines the classifier's input columns), `train_data/{session_id}/trial_*.npz`, `model.joblib`. Feature columns come from the catalog's `select()` path with every dim pinned (explicit values, never "all"); all columns must share one sampling rate. A feature listed in `config.derivatives` (the tree's **d/dt** tick) contributes its `np.gradient` time derivative as a second column beside each value column — `features/columns.py` is the one place that expansion happens, so training and inference cannot disagree about it. **The session's own labels are inputs too** (`config.label_inputs`, `labels/label_inputs.py` — Qt-free): a state class renders as its on/off indicator, a point class as a Laplacian bump at each hard-coded `POINT_SIGMAS_S`, appended after every feature column onto the feature time base (`extract_model_features(..., labels=, shift=)` is the one assembly path, and `config.column_names()` the one layout). A class the model predicts can never be one of its own inputs — at training the label is there and at inference it is not — refused by `OnsetModelConfig.validate()` on construction *and* on save, and greyed out in the tree. `retarget_individual` re-points label inputs by the same single-individual rule as features. A trial that does not carry a target contributes nothing to it. **A label sits on its curve's tallest peak and its `confidence` is a statistic of the curve there** (`labels/curve_confidence.py`, shared with the pixel spotter: `peak` height, `focus`, `ratio`, their product `focus × ratio` (the confidence); `tallest_peak` uses `scipy.signal.find_peaks` — a peak, never the maximum, so a curve still climbing at the trial's edge does not read as certain). **Which statistic is an empirical question per model**: `fit_confidence_calibration` ranks every candidate by AUC on the cross-fitted record and writes `TargetCalibration.statistic` — `peak` unless a shape statistic wins by `MIN_AUC_GAIN`; the pixel spotter writes `infer.confidence` (`product` = focus × ratio by default; `ratio` / `focus` / `peak` / `custom` with `infer.confidence_alpha`) — the same `RULES` the grids' histogram popup previews, whose *Copy for project.yaml* button hands the choice back as those `infer:` lines, so review and pipeline never disagree about what the number means. No mean, median or centre of mass anywhere; every candidate is readable off the drawn curve. Covered by `tests/test_unit/test_curve_confidence.py`. It has to be readable straight off the curve the review draws, because that is what lets the threshold be set by looking; everything it omits — a rival peak, a broad ramp, a trial where nothing rose — the curve shows. **Every target is read independently off its own curve** — no class's evidence is allowed to move another's (covered by `TestPlacement`); a linear-chain CRF that used to jointly decode the event order was removed because moving an event off its own evidence is invisible on the curve. **How often a model is right is a verdict on the model, not on a label**: `fit_confidence_calibration` (`_CV_FOLDS`, cross-fitted) counts held-out predictions landing within `tolerance_s` and `train_model` reports `hit_rate = (hits + 1) / (trials + 2)` per class. It is never applied to a label's confidence. Inference never overrides an existing target event and respects the trials-table filter + a metadata-column filter. **A run keeps the curves it read its events off** (`labels/onset_curves.py`, numpy-only so the GUI reads them without the model stack): one folder per run, `labels/predictions_lightgbm_{timestamp}/onset_curves.npz` beside the session (the `labels/` folder the label backups use), written once and never edited. `read_all_curves` merges every run newest-first per (trial, class), so a run filtered to one class never erases what an earlier run said about another. Frame-by-frame review draws them for the classes **in scope** (`CurationPanel._draw_curves` → `plot_container.show_onset_curves`, one scaled overlay per class against a fixed 0–1 range so classes are comparable) — an aid to review, never something a session depends on. Covered by `tests/test_unit/test_onset_model.py` + `test_onset_curves.py` + `test_curation_review.py` (`TestOnsetCurves`).
- **The label table is one spreadsheet over the whole TSV** (`dialog_label_table.py`, "Label table…" in the Labels tab). It reads `_all_labels_df` and addresses rows by **position**, so a frame replaced elsewhere is re-read rather than written through at stale positions (`_is_current`, also checked on window activation). Only `INTERVAL_COLUMNS` are editable — `trial` and `TRIAL_META_COLUMNS` hold one value per trial repeated per row, and editing one row's copy would only desync it. Every change calls `app_state.record_label_edit` first (one undo step per trial it touches) and lands through `replace_all_labels`; the parent redraws through `LabelsWidget._on_label_table_changed`, and every label mutation elsewhere reaches an open table through `refresh_if_stale()` on the `refresh_labels_shapes_layer` funnel. **The keys a table owns are taken back from the shell's application shortcuts** (`Ctrl+A` autoscales, `Ctrl+C` curates the trial — both fire before a focused table sees the key, and both are only guarded against *text* fields): the dialog accepts the `ShortcutOverride` for `Ctrl+A`/`Ctrl+C`/`Delete` and handles them itself, rather than binding shortcuts of its own that would be ambiguous with the global ones. Covered by `tests/test_unit/test_label_table.py`.
- **The close prompt asks the file, not the flag.** `MetaWidget._check_unsaved_changes` calls `app_state.labels_dirty()`, comparing `_all_labels_df` against `labels_file_path()` via `labels_equal` (canonical `TSV_COLUMNS`, order-insensitive). Covered by `tests/test_unit/test_labels_dirty.py`.
- **A label edit is one undo step, snapshotted before it runs.** `Ctrl+Z` (`LabelsWidget.undo_last_label_edit`) walks `LabelHistory` in `labels/tsv_store.py`: a bounded stack of **per-trial** snapshots (rows + trial flags), never the whole table. Every handler that mutates labels calls `app_state.record_label_edit(...)` once at the top, before anything changes — so the trimming, sliver purge and changepoint correction that follow are inside the same step. `clear_label_history()` runs wherever `_all_labels_df` is replaced wholesale (dataset load, label/prediction import). Covered by `tests/test_unit/test_label_undo.py`.
- **Per-plot-type rendering:** `app_state.label_overlay_modes` maps plot type key → `"full"|"bottom"|"none"`, applied to every instance of that type. Defaults in `DEFAULT_LABEL_OVERLAY_MODES`; edited via `LabelsPerPlotDialog`.
- **Labels on new panels:** any path creating/showing a panel ends with `plot_container.schedule_labels_redraw()` (deferred — it must run after content render). Never emit `labels_redraw_needed` synchronously from a panel-creation path.
- **A state label being drawn is visible from the first click**: `LabelDrawingMixin.show_pending_label()` / `clear_pending_label()`; every path abandoning or committing a half-placed label goes through `LabelsWidget._reset_label_clicks()`. Covered by `tests/test_unit/test_pending_label_preview.py`.
- **Predictions:** per-trial `.npy`/`.pickle`, shape `(T, n_classes)` → confidence via `1 - normalized_entropy`, labels via `argmax`. Confidence stays in memory. Dotted confidence curve gated per feature plot by `show_predictions` in `panel_state`.
- **Crowsetta:** `EthographSeq` registered at import; export converts int→string via the mapping. Internal storage stays integer-based.
- **Top-bar File menu:** "Import labels…" / "Import predictions…" / "Export labels…" borrow their own I/O sub-panel (`IOWidget.restore_subpanel()`). Data loading happens only on the cover page.
=======
- **A label's subject is a pair: `individual` + `individual_rec`** (`NO_RECIPIENT` = solo). The receiver is an attribute, never a second track: exclusivity is per actor (`TRACK_COLUMNS`), the overlay shows every receiver and the combo only sets the next label's. Covered by `tests/test_unit/test_label_intervals.py`.
- **A visible label is a selectable label**; mutation refuses a selection outside the active branch. Covered by `tests/test_unit/test_label_click_branches.py`.
- **Every label carries a `labeling_method`** ∈ {`manual`, `automated`, `curated`} and a `confidence` (`HUMAN_CONFIDENCE` for a hand-placed one). The only transition the GUI performs is automated → curated; an edit → manual, a new prediction → automated. A trial is curated iff none of its labels is automated; the verdict is written to the metadata table's `curated` column as `"yes"`/`"no"`, on a timer, only while `app_state.curation_active`. Covered by `tests/test_unit/test_curation.py` + `test_curation_review.py`.
- **Curation has one home: the Curation section of the Labels tab** (`widgets_curation.py`, docs `advanced/labels/curation.md`): scope (label classes), mode (`manual` / `inspect` / `frame`), frame-by-frame review. **The grids are review surfaces over the scope, never separate curation systems** (`dialog_label_gridview.py`, `dialog_video_grid.py`; `GridModeBar.entries_fn` returns what is on screen, and that is what every operation runs over). Covered by `tests/test_unit/test_label_gridview.py` + `test_video_grid.py`.
- **A curation workflow is a recording of the GUI, never a second way of doing anything** (`labels/workflow.py` + `dialog_curation_workflow.py`, docs `advanced/labels/workflows.md`). `STEP_KINDS` is the one contract; every handler drives the widget a user would. Covered by `tests/test_unit/test_curation_workflow.py`.
- **Onset models predict point events only, at most one per target class per trial** (`labels/onset_model.py`, docs `advanced/labels/onset_model.md`). Feature columns come from `features/columns.py` — the one input-layout definition shared by training and inference. A label sits on its curve's tallest peak and its `confidence` is the height there; every target is read independently off its own curve. Runs keep their curves (`labels/onset_curves.py`). Covered by `tests/test_unit/test_onset_model.py` + `test_onset_curves.py`.
- **The label table is one spreadsheet over the whole TSV** (`dialog_label_table.py`); only `INTERVAL_COLUMNS` are editable. Covered by `tests/test_unit/test_label_table.py`.
- **A label edit is one undo step, snapshotted before it runs** (`app_state.record_label_edit`, `LabelHistory` per trial). Curating is not an undoable edit. Covered by `tests/test_unit/test_label_undo.py`.
- **The close prompt asks the file, not the flag** (`app_state.labels_dirty()`). Covered by `tests/test_unit/test_labels_dirty.py`.
- **Automated labels draw dotted, manual/curated solid.** Covered by `tests/test_unit/test_label_method_style.py`.
- **Labels on new panels:** any path creating/showing a panel ends with `plot_container.schedule_labels_redraw()` (deferred). Never emit `labels_redraw_needed` synchronously from a panel-creation path.
- **Predictions:** per-trial `.npy`/`.pickle`, shape `(T, n_classes)` → confidence via `1 - normalized_entropy`, labels via `argmax`.
- **Crowsetta:** `EthographSeq` registered at import; internal storage stays integer-based.
>>>>>>> b5e49dad75c10cbaab9adb0b13f42c4012b844c9

### Variable schema (`ethograph/io/schema.py`)

What a data variable *is*, following movement's proposal (issue #978). Docs: `docs/add_to_docs_later/variable_schema.md`.

- **`kind` is advisory and nothing may require it; it is a label, never a switch.** Anything that changes maths reads a behavioural attr (`normalise`). Covered by `tests/test_unit/test_schema.py`.
- **Flags are written `0`/`1`, never `True`/`False`** (NetCDF has no boolean attr).
- **Both backends use one vocabulary**; a pynapple `Tsd` declares its schema in `{session}/.ethograph/schema.yaml`. Every reader goes through `schema.attrs_of`.
- **For changepoints, the label and the mask marker are different attrs**: `kind="changepoint_feature"` labels the family, `changepoint_mask` marks a raw mask (`schema.is_changepoint`).
- **`train.drop_kinds` is the ablation axis, `train.subsample` the rate axis, `train.loss.tau` the smoothing truncation** — all run-level, so one materialised dataset serves every run. Covered by `tests/test_unit/test_segment_pipeline.py` + `test_segment_losses.py`.

### Segmentation pipeline (`ethograph/segment/`)

Code-first, never in the GUI, and **scripted — there is no CLI**. One YAML config becomes a `Project`; every stage is a method on it; overrides are dotted `key=value` strings. Vocabulary in `CONTEXT.md`; design in `docs/add_to_docs_later/segment/`; decisions in `notes/adr/`.

<<<<<<< HEAD
- **The workflow is two stages, and they divide the trials differently.** Stage 1 `search()`: trials pooled and cut by the three ratios of `train.split` (60/20/20), Optuna maximising `train.select_on` on **val**, winner written to `searches/{name}/best.yaml` as a config that inherits the one searched. Stage 2 `cross_validate()`: leave-one-session-out, each fold predicting the session it held out so the prediction set can be opened in the GUI against the curated labels. A search refuses to run without val; cross-validation defaults to no val (the parameters are settled). Covered by `tests/test_unit/test_segment_pipeline.py` (`TestSearchAndCrossValidation`).
- **A session has no role, and `train.split` is three ratios summing to 1.** `train_fraction`/`val_fraction`/`test_fraction`, drawn by whole trial (`assign_roles`), so no trial is ever in two roles. The one way a whole session gets a role is `split.holdout_sessions`, which `cross_validate` writes per fold — never spelled per session in the YAML (a `role:` key is a config error naming the replacement). `split.holdout_trials` is the same one level down: `cross_validate(n_folds=k)` deals every trial into one fold (`crossval.trial_folds`), predicts each fold's own trials and merges them into one prediction set per session (`inference.merge_prediction_sets`) — the cross-validation of a one-session project. Covered by `tests/test_unit/test_segment_pipeline.py` (`TestSplit`) + `test_segment_neural.py` (`TestTrialFolds`).
- **A search space is keyed by the dotted override path** (`train.learning_rate`, `model.params.num_f_maps`), so a setting has exactly one spelling in the file, in an override and in `search.params`. Build them programmatically with `as_overrides({...})`, never an f-string. Search tunes the model, never the features: the dataset is materialised once, before the study.
- **`model.params` is per architecture, and a wrong key is refused before training.** The vendored models share almost no hyperparameter names, so a sweep needs one search space each (and one `train.run_name` each, or their studies pool). `_kwargs` in `models/vendored.py` validates against upstream's own YAML — `tunable_params(name)` is the same list, public. Covered by `tests/test_unit/test_segment_architectures.py`.
- **`config_to_dict` round-trips.** Every stage that re-reads a config it wrote (`save_config` → `load_run`, and both `search` and `cross_validate` rebuilding per trial/fold) depends on it, so the changepoint columns `config_from_dict` generates are left out of the dump rather than re-read as explicit entries colliding with their own expansion. Covered by `tests/test_unit/test_segment_changepoint_features.py`.

- **Features are built with the session, never by the pipeline.** Stage 1 only *selects* existing variables through `features/columns.py` (the onset model's schema: feature → dim → pinned values) and preprocesses them. Geometry (`features/geometry.py`) is a library function the user calls when building the file, so every model input is plottable in the GUI. Outputs that must not be z-scored carry `attrs["normalise"] = 0`.
- **Changepoint mask expansion is the one exception**: `features.changepoint_features` (`sigmas`, `distribution`, `inputs`: feature → dim → values naming the raw masks to expand, `transforms`: subset of `binary`/`proximity`/`offset`/`length`, `scale_by`; `sigmas`/`horizon`/`max_length` are read off the curated label durations at materialise unless spelled, recorded in `columns.yaml` with a `note`, and read back by every later stage — `materialise.resolved_config`) makes `config_from_dict` merge the generated columns straight into `features.columns` — never spelled out by hand — and makes `open_session` run `add_changepoint_features` over every trial once, so those columns exist before anything reads the session. A deterministic expansion of an existing mask, not a modelling choice. Xarray sessions only (pynapple changepoints are event times, not a dense mask); `examples/segment_changepoint_features.ipynb` shows what it looks like on real data before turning it on in a config. Covered by `tests/test_unit/test_segment_changepoint_features.py`.
- **Spike trains are the second exception**: `features.neural` (`units`, `name`, `transform`: pynapple expressions on `x`, `features/neural.py`) bins a pynapple session's `TsGroup` into a `TsdFrame` feature at `open_session`, in memory, never a file; its unit columns are read off the session at materialise, recorded in `columns.yaml` (`neural_columns`), and read back by train (`resolved_config`) and inference (`inherit_neural_columns`), so `features.columns` never spells them. One session per project — units are not consistent across recordings. Covered by `tests/test_unit/test_segment_neural.py`.
- **A sample is one (trial, individual).** The individual dim is never listed in `features.columns`; it is pinned per sample (`individual=self` in the layout), a second individual dim is `other: "*"` (`other1`, `other2`, … in dataset order). Every session must carry the same number of individuals. A layout that differs from the materialised `columns.yaml` is a `ValueError`, never a reorder.
- **Only `manual`/`curated` labels are training targets** (`Session.curated_labels`); one branch per model, or several as a multi-label target decoded per (subject, branch) track; point events are the onset model's. Background is class index 0; `classes.yaml` maps indices back to label ids.
- **The materialised dataset is role-agnostic and in the literature layout** (`features/*.npy (F, T)`, `groundTruth/*.txt`, `mapping.txt`) + `index.tsv`, `columns.yaml`, `classes.yaml`. Roles (`splits/*.bundle`) and normalisation statistics (`stats.npz`, training samples only) belong to the run — which is why one materialised dataset serves a whole study and every fold.
- **The models and the loss are DLC2Action's, vendored in upstream's own layout** under `segment/dlc2action/` (`model/`, `loss/`, `config/`) — never edited beyond what its `NOTICE.md` lists, and excluded from ruff/mypy. `models/vendored.py` adapts the architectures, `losses.py` the loss. Upstream's toolbox layer (project/data/ssl/metric/transformer) is not vendored; we have our own.
- **Every architecture speaks one contract**: `model(x (B,F,T), mask (B,1,T)) → logits (S,B,C,T)`, and **`logits[-1]` is the prediction** — the finest stage. Registered via `@register_architecture` or the `ethograph.segment.architectures` entry-point group. A model emitting stages finest-first flips them in the adapter (`reverse_stages`). A model with heads beyond the class logits returns a `ModelOutput` carrying the same `logits` plus `boundary (S,B,1,T)` and/or `query_logits`/`query_masks`; every consumer reads `as_output(model(x, mask)).logits` and never learns which heads exist. Covered by `tests/test_unit/test_segment_architectures.py`.
- **A head predicts *where*, never a reweighting of *what*.** `asrf` wraps any vendored backbone that keeps the time axis and adds ASRF's boundary branch off the shared trunk — the class branch stays bit-identical to that backbone's, so the baseline comparison is a comparison of heads. `baformer` reuses the same vendored `AttModule` encoder under BaFormer's query-voting head (no detectron2/einops/timm); its dense logits are the soft query composition in train mode and the hard boundary-aware vote in eval, so `model.eval()` is not optional. Boundary-weighted CE and Circle loss were ablated and are never re-added. Docs: `docs/add_to_docs_later/segment/boundaries.md`. Covered by `tests/test_unit/test_segment_architectures.py` + `test_segment_heads.py`.
- **Every boundary setting is a duration, resolved against the dataset's own rate.** `train.boundary.tolerance_s` and `infer.postprocess.boundary_snap_s` are seconds; `boundary.tolerance_frames(s, fs)` is the one place the conversion happens, and a run records both spellings in `test_metrics.yaml`. The literature's frame-unit boundary tolerances were tuned at 15–30 fps and mean something else at 200 Hz. Covered by `tests/test_unit/test_segment_boundary.py` (`TestTolerance`).
- **Boundary refinement re-cuts the dense prediction; changepoint correction snaps an interval edge.** They are different steps and both live in `infer.postprocess`: `boundary_refinement` ∈ `none`/`predicted`/`hybrid` runs before `dense_to_intervals`, so a span can change class outright, and `hybrid` restricts the model's peaks to the detected changepoints (the learned version of the hand rule). A run with no boundary head ignores the mode rather than failing. The four modes are compared on *one* trained model, never four. Covered by `tests/test_unit/test_segment_boundary.py` + `test_segment_heads.py` (`TestRefinementModes`).
- **The objective is one composite, itemised.** `build_objective` composes frame (`train.frame_weight`) + boundary (`train.boundary.weight`) + query set loss, returns `(total, parts)` so every term is logged and written to the run, and raises when a weight names a head the architecture does not have. A trial with more segments than `model.params.num_queries` is a `ValueError` naming the number to set, never a silent truncation. Covered by `tests/test_unit/test_segment_queries.py`.
- **A default is never written in our code.** Architectures read `dlc2action/config/model/{stem}.yaml` (`upstream_defaults()`), the loss reads `config/losses.yaml` (`build_loss()`), and `TrainConfig` reads the settings that carry over from `config/training.yaml` (`upstream_training_default()`, which coerces — YAML 1.1 reads `lr: 1e-3` as a string). `batch_size`, `grad_clip`, `eval_every` and the whole of `SplitConfig` (60/20/20) are deliberately ours; each says so at its definition. `train.loss` and `model.params` are override dicts over those YAMLs. Reach the config tree by path (`DLC2ACTION_CONFIG`), never by importing the package — that re-enters `ethograph.__init__`'s lazy loader. Covered by `tests/test_unit/test_segment_losses.py`.
- **Sessions open headless through `io/data_loader.load_features_dataset`** — Qt is imported lazily there, only on the notify paths.
- **S3D sidecars are on the video's clock; merging converts.** `{stem}_s3d.nc` holds `(time_s3d, s3d_dims)` with frame 0 at t=0; `merge_video_features` samples it at `trial_time - stream_offset_for_trial(...)` (the direction `VideoSync.frame_to_time` fixes: trial = video + offset) and writes a **sibling** session file, never the source unless `--in-place`. Covered by `tests/test_unit/test_segment_video_features.py`.
- **Prediction sets are the GUI's labels TSV** (`labeling_method=automated`, `prediction_source=run name`, per-segment `confidence`), written into the session's own `labels/` folder — the same one label backups and the LightGBM onset model's `predictions_lightgbm_{timestamp}/` runs use (`labels/onset_curves.py`) — under `labels/predictions_{run}_{timestamp}/`, one folder per call to `infer()` so a re-run never overwrites an earlier one; the `.npz` exists only for the confidence overlay. Post-processing goes through `features/changepoints.correct_changepoints`, the GUI's own function. This is the point of cross-validation: a fold's predictions and the curated labels are the same kind of object on the same axis.
- **The GUI's `gui_settings.yaml` is the default source of the post-processing numbers.** `infer.postprocess.gui_settings: true` (or a path) reads the GUI's correction settings through `config.GUI_POSTPROCESS_KEYS` — the one translation, held to `AppStateSpec.VARS` by test — on every config load; a key spelled beside it (file, `base:`, dotlist) wins; a saved run config carries the resolved values plus the path, so a run never follows the GUI afterwards. The GUI never reads a project config (ADR 0006). Covered by `tests/test_unit/test_segment_gui_postprocess.py`.
- **A study's trials and a fold's runs are ordinary runs, nested one level deeper** (`runs/{search or cv name}/…`), so `compare_runs` — which reads only the top level of `runs/` — keeps showing the runs trained by hand. Only the winning trial keeps its weights (`search.keep_weights`).
- Covered by `tests/test_unit/test_segment_pipeline.py`, `test_segment_architectures.py`, `test_geometry_features.py`.
=======
- **Two workflow stages**: `search()` (trials pooled and cut by `train.split`, Optuna on val, winner → `searches/{name}/best.yaml`) and `cross_validate()` (leave-one-session-out, each fold predicting its held-out session; `n_folds=k` folds by trial for a one-session project, merging the folds into one prediction set). A session has no role; `train.split` is three ratios summing to 1, drawn by whole trial. Covered by `tests/test_unit/test_segment_pipeline.py` + `test_segment_neural.py`.
- **A setting has exactly one spelling**: the dotted override path, in the file, in an override and in `search.params`. `model.params` is per architecture, validated against upstream's own YAML. Covered by `tests/test_unit/test_segment_architectures.py`.
- **Features are built with the session, never by the pipeline.** Stage 1 only selects existing variables through `features/columns.py`. The two exceptions are changepoint mask expansion (`features.changepoint_features`) and spike binning (`features.neural`: a pynapple `TsGroup` → `TsdFrame` feature by pynapple expressions at session open, unit columns read off the session at materialise and carried by the run). Covered by `tests/test_unit/test_segment_changepoint_features.py` + `test_segment_neural.py`.
- **A sample is one (trial, individual)**; the individual dim is pinned per sample. Only `manual`/`curated` labels are training targets. Background is class 0.
- **The materialised dataset is role-agnostic and in the literature layout**; roles and normalisation statistics belong to the run.
- **The models and the loss are DLC2Action's, vendored in upstream's layout**, never edited beyond `NOTICE.md`. **Every architecture speaks one contract**: `model(x (B,F,T), mask (B,1,T)) → logits (S,B,C,T)`, `logits[-1]` is the prediction, read through `as_output(...)`. **A default is never written in our code** — read from `dlc2action/config/` via `DLC2ACTION_CONFIG`. Covered by `tests/test_unit/test_segment_architectures.py` + `test_segment_losses.py`.
- **Prediction sets are the GUI's labels TSV** (`labeling_method=automated`, `prediction_source=run name`) under `predictions/{run}/`. Post-processing goes through the GUI's own `correct_changepoints`; the GUI's `gui_settings.yaml` is the default source of its numbers, and the GUI never reads a project config (ADR 0006). Covered by `tests/test_unit/test_segment_gui_postprocess.py`.
- **S3D sidecars are on the video's clock; merging converts.** Covered by `tests/test_unit/test_segment_video_features.py`.
>>>>>>> b5e49dad75c10cbaab9adb0b13f42c4012b844c9

### Widget orchestration

`MetaWidget` creates all widgets and wires signals; `DataWidget` is the central orchestrator. Flow: `NavigationWidget` → `trial_changed` → `DataWidget.on_trial_changed()` → everything else.

- **Context-sensitive right sidebar:** `_CONTEXT_MAP` in `gui/right_context.py`. There is exactly one individual combo in the sidebar (`DataWidget.refresh_individual_choices`).
- **Guarded shortcuts are disabled while typing, never a no-op** (`shell.bind_shortcut(guarded=True)`, `typing_in_text_field()`). Covered by `tests/test_unit/test_source_popup_nav.py`.
- **Neo and Phy trace panels are dynamic instances, added on demand from the popup** (heavy, never auto-loaded).

### Neurons + ephys

Two paths → `nap.TsGroup` + cluster table: Kilosort folder (full features) and pynapple file (raster only). **Kilosort has two index spaces**: site index (indexes `channel_positions.npy`) vs hardware channel (`channel_map.npy`). **Always index `channel_positions` by site index.**

### Pixel event spotting (`ethograph/spot/`)

Point events learned from video with E2E-Spot, scripted like `segment` and **speaking its words** — `eto.spot.Project` with `materialise()` / `train()` / `train_teacher()` / `distil()` / `evaluate()` / `compare()` / `inference()` / `cross_validate()` (docs: `docs/add_to_docs_later/spot/`). **The sessions layer is shared, the stage graph is not**: `SessionSpec`, `TrialsConfig`, `SplitConfig`, `open_session`, `filter_trials` and `assign_roles` are imported from `segment` unchanged — combining sessions and drawing a split are properties of the data, not the model — while `materialise`, the `features:` section and the `(B,F,T)` architecture contract are not, while a top-level `features:` lists pose variables directly (segment's `features.columns` spelling) as the pixel model's optional second input — a `columns:` key under it (segment's section shape) is a `ValueError` naming the difference.

- **Every temporal setting is a duration, resolved against the video's own rate.** `ClipConfig(context_s, resolution_ms, positive_window_ms).resolve(fps, max_frames)` is the one place upstream's `stride` / `clip_len` / `dilate_len` are computed — `resolution_ms` unset = the finest grid that fits `context_s` in the card's frame budget (`vendored.frame_budget()`: the measured 200 frames per 10 GB, scaled; every stage resolves through `SpotConfig.resolve_clip`, a trained run reads its stride back from `config.json`), and `ResolvedClip.to_frame` maps a strided prediction back to the **centre** of its bin (`bin*k + (k-1)/2`). Upstream's frame counts were tuned at 25 fps and mean something else at 200. A combination exceeding `MAX_FRAMES_PER_BATCH` is refused naming the *duration* to change. Covered by `tests/test_unit/test_spot_config.py`.
- **A prediction's `confidence` is its curve's shape, not its peak height** (`confidence.py`: `focus × ratio`; a rival is another local maximum outside the window, never the peak's own shoulder; a curve whose peak is below `MIN_PEAK` or that has no interior peak reads 0 — found nothing, flagged). Measured on a held-out session: peak height separates a >50 ms error at AUC 0.58, the shape statistics at ~0.8 — a tie among them — and `ratio` is the one that is bimodal, so the histogram has a gap to threshold in; for the onset model peak height wins because its curve is shape-constrained by construction. The statistic is per model; that it stays readable off the drawn curve is not.
- Curves are written through `labels/onset_curves.py` unchanged, so frame-by-frame review draws them with no new GUI code. **Every model's prediction run lives under `labels/predictions_{model}_{timestamp}/`** (`onset_curves.run_dir(session, timestamp, model=)`), and `run_dirs` orders them by timestamp, not name, so "newest wins" holds across models. Covered by `tests/test_unit/test_onset_curves.py` (`TestManyModels`).
- **The pose side is a flat `features:` list, nothing else** (ADR 0008): variables in the session file in segment's `features.columns` spelling — positions, velocities, the distances the user computed and can plot — read through `features/columns.extract_features` and written once per trial to `features/{video_id}.npz` (`features.py`; names in `features/features.json`). There is no graph, no adjacency, no `fuse:` section; a config spelling `graph:` or `fuse:` is refused by name. **Four models, by what exists at inference** (docs `spot/index.md`): the LightGBM onset model (pose, GUI); E2E-Spot (video; `rny008_msagsm` optional); E2E-Spot + `features:` as a second GRU input (`train.features_as_input`, default on when listed — the block `features/block/` is z-scored on the training split and reused at that scale for a session predicted later; `--fuse_dir/--fuse_dim/--fuse_dropout` in the vendored trainer concatenate it before the GRU; run named `{clip}_features`; **nothing makes a network use an input, so `evaluate(zero_features=True)` measures it** (`test_metrics_nofeatures.yaml`) and `train.features_dropout` keeps that ablation honest); and the pose teacher (`pose_model.PoseSpotter`: features → linear → UMEG-Net's parameter-free multi-scale shift blocks → bi-GRU → `K+1` softmax, E2E-Spot's output contract; `teacher.py` writes val predictions in E2E-Spot's recall schema and embeddings at the sweep-best epoch) distilled into E2E-Spot for video-only inference (`train.features_as_input: false`). **Distillation is a stage flag in the vendored trainer** (`--stage 2`: trunk + GRU match the teacher's per-clip embeddings from `features/embeddings/`, no labels; `--stage 3`: CNN frozen, head on labels), driven by `Project.distil()` into `runs/{baseline}_distil_{fingerprint}/stage{2,3}/`, warm-started from the label-only baseline; the `features` + `teacher` sections are fingerprinted (`config.features_fingerprint`) into the teacher's folder (`teacher/{clip}_{fingerprint}`) and the student's, so an edited list lands beside the earlier result and `distil()` refuses embeddings another list's teacher wrote; the student's loader refuses embeddings on another stride. **The gate**: distil only from a teacher that beats the baseline on the same test split (`evaluate()` scores teachers too). **Inference always decodes the video straight into the model** (`stream.py`): training needs the JPEG folder (random access), inference reads each trial once, so it decodes → crop → resize → JPEG-in-memory → tensor with a one-window rolling buffer, mirroring `test_e2e.py`'s windows, padding, transform and score accumulation exactly (covered by `tests/test_unit/test_spot_stream.py` against the vendored reader on an exported folder); the frame folder is never read or written at inference. Worker counts are a machine property (`dataset.default_workers()`: cores − 2, capped), never in the YAML. `check_vram` raises before any run the card cannot hold. `inference()` flags an out-of-order trial (confidence 0), never reorders. Docs: `docs/add_to_docs_later/spot/multimodal.md`, ADR 0007 + 0008. Covered by `tests/test_unit/test_spot_features.py` + `test_spot_teacher.py` + `test_spot_distil.py`.

### Video features (S3D)

`ethograph/video_features/`: **configured in seconds, resolved per video** — `S3DConfig` → `plan_s3d(video_fps, cfg)` → `S3DPlan`, refused loudly when the rate cannot carry the window. The rate comes from `io/video_probe.py`, never a setting. Covered by `tests/test_unit/test_s3d_plan.py` + `test_s3d_extract.py`.

### Changepoint correction

Bridge pattern: intervals → dense → correct → intervals. **A click snaps to what the clicked panel draws** (`features/changepoints.changepoint_fired` is the one reading of a mask). Covered by `tests/test_unit/test_changepoint_snap.py`.

## Dataset Structure

- NetCDF with trials. Time coords: anything containing `time`.
- Every `data_var` with a time dim is a feature. Changepoints are found via `schema.is_changepoint`; colour vars by "rgb" in the name.
- Media/session metadata: `.nwb` sources read directly; non-NWB read `.ethograph/alignment.nwb`.
- Labels live in `_labels.tsv`, not the `.nc`.
