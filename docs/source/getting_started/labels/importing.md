(target-importing-labels)=
# Importing labels

Open **File → Import labels…** to bring up the import panel. The **Labels
format** combo offers:

| Option | Source | Converter |
|--------|--------|-----------|
| **`.tsv`** | EthoGraph TSV (backup, colleague's labels, manual edit) | (native) |
| **`pynapple (.npz)`** | Pynapple file with {class}`~pynapple.IntervalSet` objects | {class}`~ethograph.labels.converters.PynappleLabelConverter` |
| **`pynapple (.nwb)`** | NWB file loaded via {class}`~pynapple.IntervalSet` objects | {class}`~ethograph.labels.converters.PynappleLabelConverter` |
| **BORIS** (`.boris`) | [BORIS](https://www.boris.unito.it/) project files | {class}`~ethograph.labels.boris.BorisLabelConverter` (via the BORIS import wizard) |
| **Crowsetta formats** (aud-seq, simple-seq, textgrid, notmat, timit, yarden, ...) | [crowsetta](https://crowsetta.readthedocs.io/)-supported annotation tools (Audacity, Praat, Raven, ...) | {class}`~ethograph.labels.converters.CrowsettaLabelConverter` |


---

(target-prediction-panels)=
## Comparing predictions

**File → Import predictions…** loads a model's prediction file (a run folder
or a plain `.tsv`). With **Load as: overlay**, each file opens in its own
**Predictions — `<file>.tsv`** panel: a thin strip above the time-series
panels, below the video. Import several files to stack them and compare
models against each other and against your labels.

Each file gets at most one panel. Closed one? Add it back from the ➕ **Add
panel** popup, which lists every imported file. Click a prediction on its
panel to select it and press `V` to play it back. Predictions are read-only.
If you want to load predictions and accept/curate them, use **File → Import predictions → Import as Labels…**

---


## Pynapple / NWB IntervalSets

Selecting **`pynapple (.npz)`** or **`pynapple (.nwb)`** loads the file with
{func}`pynapple.load_file` and extracts every
{class}`~pynapple.IntervalSet` in the data dict **except** the one used as the
trial boundaries: `trials` if present, otherwise `epochs`, then `intervals`,
then the first set with `trial` in its name. Every other set, `epochs` included
when `trials` exists, is imported as labels.
Each `IntervalSet` name becomes a label class.

Label names already in the active `mapping.txt` keep their IDs; new names are
appended to it (see {ref}`target-auto-mapping`). The labels are written to the
canonical `_labels.tsv` alongside the `.nc`.

Global-time intervals are split across trials using the `trials` /
`epochs` `IntervalSet` (or the session's trial table). See
{class}`~ethograph.labels.converters.PynappleLabelConverter` for the
conversion logic.

---

## BORIS

BORIS observations bind one or more media files (concatenated in Player 1)
to a list of events coded in observation-global time.  The wizard splits
events across media boundaries, treating **one media file as one trial**.

The importer preserves BORIS's two event kinds (see
{ref}`target-point-events`):

- **State events** become intervals (`onset_s` → `offset_s`).  Events that
  span a file boundary are clipped at the boundary with a warning.
- **Point events** become rows with `event_type = "point"` and `offset_s = NaN`.

The per-behavior `type` field from BORIS is written into
the generated `mapping.txt` so the kind is preserved on round-trip and the
labelling shortcut behaves correctly. The BORIS `Image index` column is ignored, as `ethograph`
stores label times in time (seconds), and does not round to nearest frame. This becomes
import when labelling multimodal & multi sampling rate data (video, audio, accelerometer, ...).

The `.boris` JSON is parsed via {func}`~ethograph.labels.boris.load_boris_project`;
the import wizard lives at `ethograph.gui.wizard_boris`.

---

## Crowsetta interop

EthoGraph registers an `ethograph-seq`
[crowsetta](https://crowsetta.readthedocs.io/) format for sharing labels with
string names (resolved via `mapping.txt`):

```python
from ethograph.labels.crowsetta_format import EthographSeq

# Export: int labels -> string labels via mapping
ethoseq = EthographSeq.from_intervals_df(df, id_to_name={1: "Head bob", 2: "Song"})
ethoseq.to_file("labels_for_sharing.tsv")

# Import via crowsetta
import crowsetta
scribe = crowsetta.Transcriber(format="ethograph-seq")
annot = scribe.from_file("labels_for_sharing.tsv").to_annot()
```

On import, label names already in the active `mapping.txt` keep their IDs and
new names are appended to it (see {ref}`target-auto-mapping`); `background` and
`sil` count as background.

---

## Programmatic usage

All converters expose the same `resolve_labels(...)` contract, which falls
back through `existing TSV → extract from source → empty`:

```python
from pathlib import Path
from ethograph.labels.converters import PynappleLabelConverter
import pynapple as nap

data = nap.load_file("session.nwb")
trials_ep = data["trials"] if "trials" in data.keys() else None

converter = PynappleLabelConverter(data, trials_ep=trials_ep)
df = converter.resolve_labels(
    source_path=Path("session.nwb"),
    trial_ids=[1, 2, 3],
)
```

See {doc}`exporting` for the full TSV column reference.
