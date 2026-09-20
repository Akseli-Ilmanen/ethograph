# CLAUDE.md

## Working style

- Implement, don't propose. Don't wait for "yes go".
- Write idiomatic, typed Python (mypy/ruff clean). Prefer clean design patterns over cleverness.
- Imports: sorted stdlib → third-party → local, explicit (never `from x import *`), never inside a function (only exception: circular imports).
- Comments only where the logic isn't obvious. **Never remove human-authored comments** (TODO/FIXME/NOTE/explanatory); only remove comments you added yourself.
- **Fail fast**: bugs (wrong type, missing key, unexpected `None`) crash; runtime conditions (missing file, bad user input) are handled. Never `try/except` into a silent `None`. Catch broad exceptions only at the outermost GUI boundary.
- Test/debug scripts live in `tests/`, never the project root. Prefix ad-hoc debug scripts `_test_` so pytest skips them.
- Docs/docstrings: don't name individuals (Poppy, Freddy, Ivy).
- `docs/add_to_docs_later/` is unpublished drafts: nothing links to or cites a file in it.
- `docs/` is public-facing; `notes/adr/` is private (for the maintainer and Claude). Docs never cite an ADR.
- Claude Code may change any file in this repo.

### How to maintain this file

This file describes **the architecture**, not the history of how it got here, and not the feature you just built. Every line is loaded into every session, so a line has to pay for itself.

- **Finishing a task never adds to this file.** A new setting, a dialog, a fixed bug, a layout rule, a scope decision — none belong here. Encode the invariant in a test; the test is enforced, prose is not.
- **Only a genuine architectural change touches this file**: a new subsystem, a new layer, or a rule that changes how every module must be written. Preferably as a deletion.
- **No war-stories, no measurements, no "covered by" lists.** State the invariant positively in one line, or not at all. A rule the code already makes obvious is a restatement — remove it.

### What earns a test

A test earns its place if **its failure would surprise you**. Keep invariant guards, contract guards (two things must agree and nothing forces them to), branching logic, regression guards. Cut tombstones, structural pins (`hasattr`, widget parentage, menu-title lists), unasserted actions, restatements. **A test never pays for data it does not read**: the cheapest fixture that can still fail (`gui` for widget wiring, `qapp` for one widget, none for Qt-free logic); behaviour that holds for every dataset is tested once.

## Project Overview

Ethograph is a GUI for labelling start/stop times of animal movements, paired with model pipelines that predict labels. It loads NetCDF/NWB/pynapple datasets and displays synchronized video/audio/ephys.

```python
import ethograph as eto

dt = eto.open("data.nc")
dt = eto.from_datasets([ds1, ds2])
time = eto.get_time_coord(da)
data, filt = eto.sel_valid(da, kwargs)
```

## Hard Rules

- **Never hardcode rates.** Use source metadata (`video.fps`, `ImageSeries.rate`, audio rate) or user settings; if unknown, raise or return `None`.
- **Never hardcode a device.** `resolve_device()` picks CUDA → MPS → CPU.
- **Never special-case a `dataset_key` in GUI code** — per-dataset settings live in `DATASETS` metadata.
- **Never call `QFileDialog` directly** — go through `gui/file_dialogs.py` (wizard tabs excepted: they hold no `app_state`).
- **Never hand-roll `trial_offset + t`.** Conversions go through `app_state.to_display` / `from_display`.

## File Structure

```
ethograph/gui/
    app_state.py              # AppStateSpec + ObservableAppState
    data_loader.py            # Dataset dispatch (.nc / .nwb / pynapple / DANDI)
    widgets_meta.py           # MetaWidget — creates + wires everything
    widgets_data.py           # DataWidget — central orchestrator
    widgets_{io,labels,navigation,changepoints,ephys,plot_settings,transform,curation}.py
    plots_base.py             # BasePlot, PanelStateMixin
    plots_container.py        # UnifiedPanelContainer
    plots_{audiotrace,spectrogram,ephystrace,lineplot,heatmap,raster,space,radial,console}.py
    video_sync.py, video_manager.py, audio_clock.py, label_drawing_mixin.py
    pose_render.py            # Pose loading (NWB + movement), PoseDisplayManager
    pose_{annotate,fill,refine,detect,detect_preview,tagsheet,edit_mixin}.py   # keypoint labelling + fill
    box_annotate.py, box_overlay.py, dialog_box_labelling.py                    # box labelling (OCTRON)
    dialog_{pose_labelling,skeleton_editor,pose_refinement,tag_sheet}.py
    dialog_{label_gridview,video_grid,label_table,onset_model,curation_workflow}.py
    right_context.py, main_window.py, top_bar.py, cover_page.py, table_filter.py, file_dialogs.py
    nwb_alignment.py, shortcuts.py

ethograph/labels/
    intervals.py              # Interval ops, mapping loaders
    onset_model.py            # GradBoost point-event onset detection
    onset_curves.py           # Per-run prediction curves (every model writes through it)
    curation.py, workflow.py  # labeling_method transitions + curation workflows (Qt-free)
    octron_project.py         # The OCTRON project folder in OCTRON's own layout
    ml.py, tsv_store.py, predictions.py, crowsetta_format.py, converters.py, export.py

ethograph/io/
    catalog.py                # DataCatalog + XarrayLoader/PynappleLoader
    derived.py                # TracedArray + DerivedLoader (console features)
    trialtree.py              # TrialTree (xr.DataTree subclass)
    time_model.py             # TimeRange, RestrictionWindow, TimeSource, SourceCollection
    time_sources.py, overlay_source.py, video_feature_files.py, audio_extract.py, nc_drop.py
    schema.py                 # Variable schema attrs (movement#978)
    dataset.py, validation.py, pynapple.py, metadata_table.py, metadata_edit.py, ephys_loader.py

ethograph/features/
    columns.py                # FeatureColumn + extract_features — the one input-layout definition
    label_inputs.py           # Labels of another branch rendered as input columns
    geometry.py, changepoints.py, movement.py, preprocessing.py, energy.py, oscillatory.py, neural.py, audio_changepoints.py

ethograph/video_features/     # S3D video features, configured in seconds, resolved per video rate

ethograph/segment/            # Segmentation pipeline (docs: docs/source/models/segment/)
    config.py, sessions.py, samples.py, materialise.py, train.py, inference.py, search.py, crossval.py, windows.py, project.py
    feral.py                  # FERAL's inputs from the sessions; its embeddings come back as the `feral` variable
    preprocess.py, augment.py, dataset.py, losses.py, metrics.py, postprocess.py, plotting.py, video_features.py
    models/                   # Architecture registry + contract; vendored.py adapts DLC2Action; rnn.py is ours
    dlc2action/, specscalpel/, lady/   # Vendored — see each NOTICE.md; excluded from ruff/mypy

ethograph/spot/               # Pixel point-event spotting (docs: docs/source/models/spot/)
    config.py, dataset.py, features.py, project.py, inference.py, stream.py, predict.py, confidence.py, metrics.py, pose_batch.py
    vendored.py               # Driving the vendored E2E-Spot (python -m subprocesses, retries, logs)
    e2espot/                  # Vendored E2E-Spot in upstream's layout — see its NOTICE.md; excluded from ruff/mypy
    msagsm.py                 # MultiScaleGatedShift, written from the paper

ethograph/utils/              # io.py, xr_utils.py, sequences.py, device.py, system_check.py
    configkit.py              # Chained YAML + dotted overrides ↔ dataclass tree; segment and spot each pass a Schema
ethograph/skeleton/           # PrecomputedRenderer, SkeletonState, config.py, shapes.py
THIRD_PARTY_NOTICES.md        # Index of every vendored tree and adapted file
```

## Architecture

### Two data-source layers

**Rendering** (`io/plot_sources`): `PlotSource` protocol (`name`, `time_range`, `sampling_rate`, `identity`, `get_data(t0, t1)`); `WindowedBuffer` caches wider than the viewport. **Navigation** (`io/time_model.py` + `time_sources.py`): `TimeSource` / `SourceCollection`, using only `time_range`, never `get_data()`.

### TrialTree

`TrialTree` inherits `xr.DataTree`; each trial is a child node with `attrs["trial"]` (`dt.trial(id)`, `dt.trials`, `dt.trial_items()`, `dt.map_trials(fn)`, `dt.update_trial(id, fn)`). Session metadata (trial timing, media paths, rates, offsets) comes from `app_state.nwb_alignment`, not the tree.

### State: `app_state.py`

`AppStateSpec` is a type-checked spec; `ObservableAppState` auto-generates a Qt signal per variable and auto-saves to YAML. `SCOPE_LOCAL` (per-dataset `local_settings.yaml`) holds anything that only means something for one dataset — the plot x-extent, its individuals/keypoints, its paths; `SCOPE_GLOBAL` (`gui_settings.yaml`) holds preferences that follow the user. Path settings are validated on load; a path from another machine is dropped, not restored.

### Time model + navigation

`app_state.window_bounds` drives every plot's x-limits; `app_state.session_time_range` is the full extent. Navigation modes: Session / Trial / Label / Sequence.

**The one clock rule** (docs: `advanced/time_slider_trial_session.md`): `app_state.display_basis` is `"session"` iff `slider_scope == "session"` and `navigate_mode` is not label/sequence; else `"trial"`. It is the authority on which clock the plot axis speaks: `VideoSync.frame_to_time`/`time_to_frame` speak it, audio indexes through `audio_display_offset()`, ephys through `ephys_display_offset()`, labels draw from `app_state.get_display_intervals()`.

### Catalog + loader: `catalog.py`

`DataCatalog` declares features, dimensions and streams; every `data_var` with a time dim and every pynapple `Tsd`/`TsdFrame`/`TsdTensor` is a feature. `DataLoader.select(feature, selections, t0, t1) → PlotData`, always in display coordinates; loaders ignore dims a feature lacks.

- **A combo is named after the dim it selects from.** `INDIVIDUAL_DIMS` lists the spellings; read whichever a dataset uses via `catalog.individual_combo` / `app_state.selected_individual()` — never hardcode one.
- **At most one free multi-value dim per panel** (`PanelStateMixin._sanitize_selections`). A saved `panel_layout` is untrusted and rebuilt on failure.
- `PynappleLoader` holds no trial state; pynapple objects live in absolute session time and the loader bridges to display time. NWB loads via pynapple. **A pynapple feature's columns are a dim named in exactly one place** (`_column_axes()`).

### Alignment: `nwb_alignment.py`

`.nwb` sources are read/edited directly; `.ethograph/alignment.nwb` sidecars exist only for non-NWB sources.

- **Trial timing has one source: the alignment NWB trials table.**
- **A session is a folder: it holds `.ethograph/alignment.nwb` or a root `.nwb`, and at most one `.nc` (its dataset).** A second root `.nc` is an old version and refused (`AmbiguousSessionError`) until `project.yaml`'s `ignore` names it; a file named explicitly is always loaded. Every session file is named by the folder through `io/session_layout.py` (`session_dir_of`, never `Path(...).parent`). Media folders are absolute and separate: the pairing table carries full paths (`discover_media` takes no root); the NWB stores basenames per trial and full paths per stream. The session record declares the individuals; a dataset's individual dim may only name a subset.
- **A drop makes the dropped folder the session** (`.ethograph/` goes there; several source folders → one popup, `gui/session_folder.py` ranks kinds); a folder that already has an alignment without a drop record is never overwritten.
- **Metadata is purely additive** — a tabular file joined on `trial`, resolved explicit `alignment_path` → NWB source trials → sidecar TSV, edited in the trials table and written back to the source it was read from. Media filenames are the alignment's, joined read-only onto the table's right-hand side, never written to a metadata file.
- **The trials table's filters are the one trial filter, and every operation runs over `app_state.trials`.** No dialog gets a metadata filter of its own; the label filter is a second, session-only slot intersected after the column filters.
- **Drag & drop = single-trial loading** (`cover_page.classify_files()`). A dropped `.nc` is always features; a pose overlay only in its video's pixels.
- **Video features from files attach in memory, matched by video name** (`io/video_feature_files.py`), sampled onto the trial clock through the alignment; never written back by `TrialTree.save`.
- **Video-container audio is decoded once to a cached WAV** (`io/audio_extract.py`). Never add a video extension to `AUDIO_EXTENSIONS`.

### Pose rendering + keypoint labelling

`PoseRenderData` unifies file (movement) and NWB (lazy HDF5) poses; filtering acts on masks, never recreates layers. **The video overlays any catalog feature in the camera's pixels** (`io/overlay_source.py`): a time dim, a `space` dim with x/y, at most keypoint/individual dims; `position` is the default; companions by name (`confidence` filters, `shape` means boxes); a pose file is read only when the dataset has nothing in that camera's pixels. **Colour encodes one axis, chosen by the user** (`app_state.pose_color_by`); text labels carry the other.

**Keypoint labelling's binding design rules live in `docs/source/advanced/keypoint_labelling/`. Read them before editing `gui/pose_*.py`, `dialog_pose_labelling.py`, `dialog_tag_sheet.py` or `table_filter.py`.** Scope: one camera, one trial.

Skeleton precedence: `skeleton_config_override` (user-drawn) > the project's `project.yaml` skeleton > NWB config; each recoloured with `skeleton_base_color`. **`project.yaml` holds study defaults** (individuals, cameras, mics, rig, pose software, skeleton); a session's record or dataset overrides them, machine paths never go in it (`gui/project.py`). Anchored shapes (`skeleton/shapes.py`) are templates bound to ≥2 control points.

### Panels are layout instances — no per-plot-type toggles

Panels are dock widgets in `UnifiedPanelContainer`, created via the layout (`SourcePopup`, ➕ / Shift+N, or drop), never on/off toggles; dropping an already-shown source creates another instance. Audio panels, camera views, line/space/radial plots are instances with `add_*`/`remove_*` on their owner; anything that must follow the trial iterates the live views. The heatmap is a fixed singleton. Layout persistence is automatic (`panel_layout` local, `window_state` global). Split minimums are deliberately slivers; never raise one to get a default proportion.

- **One canonical feature list**: `catalog.feature_choices()` — never `ds.data_vars`.
- **Feature plots render only from their own `panel_state`**, never from `app_state.features_sel`.
- **Labels on new panels:** any path creating/showing a panel ends with `plot_container.schedule_labels_redraw()` (deferred), never a synchronous `labels_redraw_needed`.
- Space reference geometry comes from `~/.ethograph/defaults/config/space/*.yaml`; templates ship layouts via `local_settings.yaml`, never overwriting a local file.

### Console panel + derived features

The one singleton panel. **The console binds what a panel renders**, not what backs it. Numpy, not xarray: `TracedArray` (`io/derived.py`) records the expression graph; each assigned name becomes a feature via `DerivedLoader.register()`; `stack(a, b)` makes a `(T, D)` feature usable as a space plot. Derived features live for one trial.

### Video sync

`VideoSync.frame_to_time` / `time_to_frame` speak the display clock; never raw `frame / fps`. Video decode stays clipped to the current trial. **The playhead is always exact; only the video quantizes.** **The audio-master clock (`gui/audio_clock.py`) is DAC-anchored, chunked and rate-bounded; it never leads the sound.** The render chain is guarded and watched. Reloading the same file reuses the `PlotVideo`. Cropping a camera is display-only, keyed by camera name.

### Labels

**Storage:** TSV (`labels.tsv`, one per session folder; `io/session_layout.py` names every session file), columns `onset_s, offset_s, labels (int), individual, individual_rec, event_type, confidence, labeling_method, trial, changepoint_corrected, prediction_source, n_samples`; `onset_s`/`offset_s` are trial-relative. Names in `mapping.txt`. In memory: `app_state._all_labels_df` (all trials), `app_state.label_intervals` (current trial).

- **A label's subject is a pair: `individual` + `individual_rec`** (`NO_RECIPIENT` = solo). The receiver is an attribute, never a second track; exclusivity is per actor.
- **A visible label is a selectable label**; mutation refuses a selection outside the active branch.
- **Every label carries a `labeling_method`** ∈ {`manual`, `automated`, `curated`} and a `confidence`. The GUI's only transition is automated → curated; an edit → manual, a new prediction → automated. A trial is curated iff none of its labels is automated; the verdict goes to the metadata table's `curated` column, only while `app_state.curation_active`. Automated labels draw dotted, manual/curated solid.
- **Curation has one home: the Curation section of the Labels tab** (`widgets_curation.py`). **The grids are review surfaces over the scope, never separate curation systems**: what is on screen is what every operation runs over.
- **A curation workflow is a recording of the GUI, never a second way of doing anything** (`labels/workflow.py`); `STEP_KINDS` is the one contract and every handler drives the widget a user would.
- **The LightGBM onset model predicts point events only, at most one per target class per trial** (`labels/onset_model.py`). Its input layout comes from `features/columns.py`, shared by training and inference. A label sits on its curve's tallest peak; runs keep their curves (`labels/onset_curves.py`).
- **The label table is one spreadsheet over the whole TSV**; only `INTERVAL_COLUMNS` are editable.
- **A label edit is one undo step, snapshotted before it runs** (`app_state.record_label_edit`). Curating is not an undoable edit.
- **The close prompt asks the file, not the flag** (`app_state.labels_dirty()`).
- **Every model's prediction run lives under `labels/predictions_{model}_{timestamp}/`** (`onset_curves.run_dir`), ordered by timestamp, so "newest wins" holds across models.

### Variable schema (`io/schema.py`)

- **`kind` is advisory and nothing may require it; it is a label, never a switch.** Anything that changes maths reads a behavioural attr (`normalise`).
- **Flags are written `0`/`1`, never `True`/`False`.**
- **Both backends use one vocabulary**; a pynapple `Tsd` declares its schema in `{session}/.ethograph/schema.yaml`. Every reader goes through `schema.attrs_of`.
- **For changepoints, the label and the mask marker are different attrs**: `kind="changepoint_feature"` labels the family, `changepoint_mask` marks a raw mask (`schema.is_changepoint`).

### Segmentation pipeline (`ethograph/segment/`)

Code-first, never in the GUI, **scripted — there is no CLI**. One YAML config becomes a `Project`; every stage is a method on it; overrides are dotted `key=value` strings. Vocabulary in `CONTEXT.md`; design in `docs/source/models/segment/`; decisions in `notes/adr/`.

- **Two workflow stages**: `search()` (Optuna on the val split → `searches/{name}/best.yaml`) and `cross_validate()` (leave-one-session-out; `n_folds=k` by trial for a one-session project). A session has no role; `train.split` is three ratios drawn by whole trial.
- **A setting has exactly one spelling**: the dotted path, in the file, in an override and in `search.params`. `model.params` is per architecture, validated against upstream's own YAML.
- **Features are built with the session, never by the pipeline.** Stage 1 only selects existing variables through `features/columns.py`; the three exceptions are changepoint mask expansion, spike binning (`features.neural`) and label inputs (`features.label_inputs`). **An input branch is never a target branch.**
- **A sample is one (trial, individual)**. Only `manual`/`curated` labels are training targets. Background is class 0.
- **The materialised dataset is role-agnostic and in the literature layout**; roles and normalisation statistics belong to the run. `train.drop_kinds`, `train.subsample` and `train.loss.tau` are run-level, so one dataset serves every run.
- **Vendored models are never edited beyond their `NOTICE.md`.** **Every architecture speaks one contract**: `model(x (B,F,T), mask (B,1,T)) → logits (S,B,C,T)`, `logits[-1]` is the prediction. **A default is never written in our code** — read from the vendored config via `DLC2ACTION_CONFIG`.
- **Prediction sets are the GUI's labels TSV** (`labeling_method=automated`, `prediction_source=run name`). Post-processing goes through the GUI's own `correct_changepoints`; the GUI never reads a project config.
- **S3D sidecars are on the video's clock; merging converts.** `video_features/` is configured in seconds and resolved per video rate, refused loudly when the rate cannot carry the window.

### Pixel event spotting (`ethograph/spot/`)

Point events learned from video with the vendored E2E-Spot, scripted like `segment` and speaking its words: `eto.spot.Project` with `materialise()` / `train()` / `evaluate()` / `compare()` / `inference()` / `cross_validate()`. **The sessions layer is shared, the stage graph is not**: `SessionSpec`, `TrialsConfig`, `SplitConfig`, `open_session`, `filter_trials`, `assign_roles` are imported from `segment` unchanged.

- **Every temporal setting is a duration, resolved against the video's own rate.** `ClipConfig.resolve(fps, max_frames)` is the one place upstream's `stride` / `clip_len` / `dilate_len` are computed; a trained run reads its stride back from its `config.json`; `ResolvedClip.to_frame` maps a strided prediction to the **centre** of its bin. A combination exceeding the card's frame budget is refused naming the *duration* to change.
- **A prediction's `confidence` is its curve's shape, not its peak height** (`confidence.py`: `focus × ratio`); a curve with no interior peak reads 0 — found nothing, flagged. Curves are written through `labels/onset_curves.py` unchanged.
- **The pose side is a flat `features:` list, nothing else** (ADR 0008): session variables in segment's `features.columns` spelling, written per trial to `features/{video_id}.npz` and, z-scored on the training split, fed to the model as a second GRU input (run named `{clip}_features`). `graph:`, `fuse:` and a `columns:` key are refused by name. `evaluate(zero_features=True)` measures what the features contribute; `train.features_dropout` is off by default.
- **Inference always decodes the video straight into the model** (`stream.py`), mirroring `test_e2e.py`'s windows, padding, transform and score accumulation; the frame folder is training's alone. Worker counts are a machine property, never in the YAML. `check_vram` raises before any run the card cannot hold. `inference()` flags an out-of-order trial, never reorders.

### Widget orchestration

`MetaWidget` creates all widgets and wires signals; `DataWidget` is the central orchestrator. Flow: `NavigationWidget` → `trial_changed` → `DataWidget.on_trial_changed()` → everything else. The right sidebar is context-sensitive (`_CONTEXT_MAP` in `right_context.py`) with exactly one individual combo. **Guarded shortcuts are disabled while typing, never a no-op.** Neo and Phy trace panels are added on demand, never auto-loaded.

### Neurons + ephys

Two paths → `nap.TsGroup` + cluster table: Kilosort folder (full features) and pynapple file (raster only). **Kilosort has two index spaces**: site index (indexes `channel_positions.npy`) vs hardware channel (`channel_map.npy`). **Always index `channel_positions` by site index.**

### Changepoint correction

Bridge pattern: intervals → dense → correct → intervals. **A click snaps to what the clicked panel draws** (`features/changepoints.changepoint_fired` is the one reading of a mask).

## Dataset Structure

- NetCDF with trials. Time coords: anything containing `time`. Every `data_var` with a time dim is a feature; changepoints via `schema.is_changepoint`; colour vars by "rgb" in the name.
- Media/session metadata: `.nwb` sources read directly; non-NWB read `.ethograph/alignment.nwb`.
- Labels live in the session folder's `labels.tsv`, not the `.nc`.
