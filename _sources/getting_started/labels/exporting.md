(target-exporting-labels)=
# Exporting labels

**File → Export labels…** opens the export panel described below (save button,
post-processing actions, remote backup settings).

## Label file format

Labels are stored in a **TSV file** alongside the `.nc` data file. The `.nc`
file is read-only after creation — it holds features, trial structure, media
references and non-label metadata. Labels live exclusively in the TSV.

```
session_20260903/
├── data.nc                          # features, ephys, trial structure (read-only)
├── data_labels.tsv                  # recommended file location for most-up-to-date labels
├── labels/
│   ├── backups/
│       ├── data_labels_20240315_101230.tsv  # timestamped backups
│       └── data_labels_20240314_111420.tsv
```

The TSV uses integer label IDs in the `labels` column. Label names are
managed centrally in `mapping.txt` — rename a label once there, and it
applies everywhere. See {doc}`mapping` for the format and where it lives.

---

## Saving labels (Ctrl+S)

Each save writes the canonical `data_labels.tsv` alongside the `.nc`, plus a
timestamped backup in `labels/backups/`. An optional remote backup can be
configured (see {ref}`Advanced <target-labels-advanced>`).

---

## Column reference

Every saved TSV contains the following columns. Core columns are written by
the GUI; computed columns are derived on each save from the data file.

### Core columns (editable)

| Column | Type | Description |
|--------|------|-------------|
| `onset_s` | float | Segment start in **trial-relative** seconds (time starts at 0 for each trial) |
| `offset_s` | float | Segment end in **trial-relative** seconds |
| `labels` | int | Label class ID from `mapping.txt` (0 = background, excluded from display) |
| `event_type` | str | `state` for an interval with an onset and offset, `point` for an instantaneous event |
| `individual` | str | The individual performing the behaviour, i.e. the actor (e.g. `"mouse1"`) |
| `individual_rec` | str | The recipient of a dyadic behaviour (e.g. the bird being mounted), shown as **Receiver** in the GUI; empty for a solo behaviour |
| `trial` | int/str | Trial identifier, matches the TrialTree |
| `confidence` | float | How sure the label is: `1.0` for a hand-placed label, the model's own score for a predicted one |
| `labeling_method` | str | Who vouches for the label: `manual` (placed or edited by hand), `automated` (a model's output nobody has looked at) or `curated` (automated, then approved) — see {doc}`../../models/curation`. A file without the column reads `automated` for any row with `confidence < 1.0`, `manual` otherwise |

`individual` and `individual_rec` together are the **subject** of a label. The
receiver is an attribute of each label, not a separate track: overlapping labels
are resolved per actor, every label of the actor is drawn (tagged `→ receiver`
when it has one), and the sidebar's Receiver combo only sets the receiver of the
next label you create. Files written before recipients existed have no
`individual_rec` column and read back as solo behaviours.

### Per-trial metadata columns

These have the same value for every row in a trial:

| Column | Type | Description |
|--------|------|-------------|
| `changepoint_corrected` | int | 1 if changepoint correction has been applied to this trial |
| `prediction_source` | str | Path to the prediction file that produced these labels (empty for human-labeled) |
| `n_samples` | int | The trial's sample count, used for dense conversion; `0` if unknown |


### Computed columns (generated on save)

These are recomputed from the core columns + the `.nc` data on every save. If
you manually edit `onset_s` in a backup file and reload it, these will be
recalculated.

| Column | Type | Description |
|--------|------|-------------|
| `session` | str | Session identifier from `ds.attrs["session"]`. Only present when `session` is set in the dataset attributes. |
| `session_trial` | str | `"{session}_{trial}"` for grouping across sessions. Only present when `session` is set. |
| `duration` | float | `offset_s - onset_s` in seconds |
| `sequence_idx` | int | Zero-based position of this segment in the trial's label sequence |
| `sequence` | str | Dash-joined label IDs for the trial (e.g. `"1-3-2-1"`) |
| *(trial metadata columns)* | str/int/float | Trial-level metadata columns exported from the labels metadata table and carried over automatically for matching trials (for example `stimulus`, `num_pellets`, `condition`). |

### Timing columns (computed, requires alignment)

These columns convert trial-relative times to session-absolute times. They
only appear when the {ref}`NWB alignment file <target-nwb-alignment>` has real
trial start/stop times (i.e. `alignment.has_real_timing` is `True`).

| Column | Type | Description |
|--------|------|-------------|
| `trial_onset` | float | Absolute start time of the trial within the session (seconds) |
| `trial_offset` | float | Absolute end time of the trial (if `stop_time` available) |
| `onset_global` | float | `trial_onset + onset_s` — segment start in session-absolute time |
| `offset_global` | float | `trial_onset + offset_s` — segment end in session-absolute time |

```{tip}
**When to use which time?**
- Use `onset_s` / `offset_s` when working within a single trial (plotting, ML training).
- Use `onset_global` / `offset_global` when aligning across trials or comparing to session-level events (e.g. neural recordings with session-absolute timestamps).
```

### Example

| onset_s | offset_s | labels | individual | trial | duration | sequence_idx | sequence | onset_global |
|---------|----------|--------|------------|-------|----------|--------------|----------|--------------|
| 0.41    | 0.505    | 1      | mouse1      | 1     | 0.095    | 0            | 1-2-3    | 120.41       |
| 0.51    | 0.620    | 2      | mouse1      | 1     | 0.110    | 1            | 1-2-3    | 120.51       |
| 0.77    | 0.885    | 3      | mouse1      | 1     | 0.115    | 2            | 1-2-3    | 120.77       |

Here trial 1 starts at 120s into the session, so `onset_global = 120 + onset_s`.


---

(target-labels-advanced)=
## Advanced

### Remote backup

A remote backup folder can be configured in the label controls. Three save
modes are available:

| Mode | Behavior | Use case |
|------|----------|----------|
| **Save with timestamp** (default) | Creates a new timestamped file on each save | Safe, auditable history |
| **Overwrite file** | Saves to a single file, overwriting the previous version | Simple cloud sync |
| **Overwrite + git commit** | Overwrites a single file and auto-commits to git | Full version control |

For git mode, one-time setup:

```bash
cd /path/to/remote_backup_folder
git init
```

After that, every Ctrl+S auto-commits the label file. Git must be installed
and on your PATH; the remote folder must be a git repository.
