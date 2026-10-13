(target-importing-labels)=
# Importing labels

Open **File → Import labels…** to bring up the import panel. The **Labels
format** dropdown offers:

| Option | Source | Converter |
|--------|--------|-----------|
| **`.tsv`** | Ethograph TSV (backup, colleague's labels, manual edit) | (native) |
| **`pynapple (.npz)`** | Pynapple file with {class}`~pynapple.IntervalSet` objects | {class}`~ethograph.labels.converters.PynappleLabelConverter` |
| **`pynapple (.nwb)`** | NWB file loaded via {class}`~pynapple.IntervalSet` objects | {class}`~ethograph.labels.converters.PynappleLabelConverter` |
| **Crowsetta formats** (aud-seq, simple-seq, textgrid, notmat, timit, yarden, ...) | [crowsetta](https://crowsetta.readthedocs.io/)-supported annotation tools (Audacity, Praat, Raven, ...) | {class}`~ethograph.labels.converters.CrowsettaLabelConverter` |

A [BORIS](https://www.boris.unito.it/) project is not a label file for an open
session: it holds the media and the trials too, so it is imported as whole
sessions from the start page — see {doc}`../../advanced/migrating_from_boris`.


---

(target-prediction-panels)=
## Comparing predictions

**File → Import predictions…** loads a model's prediction file (a run folder
or a plain `.tsv`). With **Load as: overlay**, each file opens in its own
**Predictions — `<file>.tsv`** panel: a thin strip above the time-series
panels, below the video. Import several files to stack them and compare
models against each other and against your labels. Every run folder writes
the same `<stem>_predictions.tsv`, so a run's panel is named with its folder
in front (`predictions_mstcn_20260101/<stem>_predictions.tsv`); a plain
`.tsv` keeps its own name.

The Predictions group lists every set currently loaded. Select one to make
it current — the Curation section's **Confidence curves…** falls back to
it when the labels carry no run of their own; **Remove** unloads it and
closes its panel. A run folder's frame-by-frame confidence curve is drawn
dashed inside that run's own panel, on its 0–1 axis, and nowhere else —
the **Confidence** checkbox shows or hides it on every panel at once. A
plain `.tsv` has no curve.

The Moll et al. 2025 template ships several such runs to try this on: they
download with the dataset into its `labels/predictions_<model>/` folders
(one per model, e.g. `predictions_mlp`, `predictions_c2f-tcn`,
`predictions_feral-cp`), each ready for **Import predictions → From folder**.

Each file gets at most one panel. Closed one? Add it back from the ➕ **Add
panel** popup, which lists every loaded file. Click a prediction on its
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
canonical `labels.tsv` in the session folder.

Global-time intervals are split across trials using the `trials` /
`epochs` `IntervalSet` (or the session's trial table). See
{class}`~ethograph.labels.converters.PynappleLabelConverter` for the
conversion logic.

---

## Crowsetta interop

Ethograph registers an `ethograph-seq`
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
