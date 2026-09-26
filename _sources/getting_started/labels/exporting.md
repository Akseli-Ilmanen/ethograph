(target-exporting-labels)=
# Exporting labels

**File → Export labels…** opens the export panel described below (save button,
the **Columns** selector, remote backup settings).

## Label file format

Labels are stored in a **TSV file** alongside the `.nc` data file. The `.nc`
file is read-only after creation — it holds features, trial structure, media
references and non-label metadata. Labels live exclusively in the TSV.

```
session_20260903/
├── data.nc                          # features, ephys, trial structure (read-only)
├── labels.tsv                       # the session's labels: one file per session folder
├── labels/
│   ├── backups/
│       ├── data_labels_20240315_101230.tsv  # timestamped backups
│       └── labels_20240314_111420.tsv
```

The TSV uses integer label IDs in the `labels` column. Label names are
managed centrally in `mapping.txt` — rename a label once there, and it
applies everywhere. See {doc}`mapping` for the format and where it lives.

---

## Saving labels (Ctrl+S)

Each save writes the canonical `labels.tsv` in the session folder, plus a
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
| `labeling_method` | str | Who vouches for the label: `manual` (placed or edited by hand), `automated` (a model's output nobody has looked at) or `curated` (automated, then approved) — see {doc}`../../models/curation/index`. A file without the column reads `automated` for any row with `confidence < 1.0`, `manual` otherwise |

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

(target-lab-specific-columns)=
### Lab-specific columns

Some columns only mean something for one lab's data. The export panel's
**Columns** selector chooses which — if any — are added:

| Choice | What the TSV carries |
|--------|----------------------|
| **Standard** (default) | Only the columns documented above |
| **Standard + \<lab\>** | The standard columns **plus** that lab's own |

A lab's exporter always runs *after* the standard columns, on the table they
produced. Choosing one never means going without the standard export, and an
exporter that dropped a required column is refused rather than written.

The choice is stored in the global settings (`gui_settings.yaml`), so it
follows you across sessions rather than being a property of one dataset.

Ethograph ships one:

| Exporter | Columns | Requires |
|----------|---------|----------|
| **Crow lab** | `session` — session identifier<br>`session_trial` — `"{session}_{trial}"`, for grouping across sessions<br>`pulse_onsets` — the trial's pulse times, on the trial's first row only | `ds.attrs["session"]`, a `pulse_onsets` variable |

A session missing those simply gets no such column, so picking an exporter
whose data you do not have is harmless.

#### Adding your lab's columns

Exporters live in `ethograph/labels/exporters/`. Add a module beside the
others, register it, and open a pull request:

```python
# ethograph/labels/exporters/mylab.py
from ethograph.labels.exporters import ExportContext, register


@register("mylab", "My lab — the columns my analysis expects")
def my_columns(df: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    df["my_column"] = ...
    return df
```

Then add `from ethograph.labels.exporters import mylab` to `_load_builtin()`
in that package's `__init__.py`, so registering happens on import.

`df` is the table with every standard column already on it, one row per
non-background label. `ctx` carries what the session knows: `ctx.dt` (the data
tree), `ctx.alignment`, `ctx.metadata_df` and `ctx.session_dir` — read a
sidecar file from the last of these if your columns come from outside the
dataset. Return the table; add or change your own columns, but leave the ones
a labels file is made of alone.

The registered description is shown in the selector prefixed with
`Standard + `, so write only what your exporter adds.

---

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
