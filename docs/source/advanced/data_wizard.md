(target-data-wizard)=
# Data wizard

**Data wizard, prepare my data** on the start page turns a folder of media
files into an alignment file, or into a notebook that makes one with neuroconv.
It is one long page; the mode you pick at the top decides which of the pages
below appear. The background is in
{ref}`Pairing and alignment <target-nwb-alignment>`.

## Mode

Three modes, each with a figure of the set-up it covers:

| Mode | Pick it when | Ends with |
|---|---|---|
| **1 Pair my media files** | Your files already share a clock (one file per trial, or session-wide files with a known start) | The wizard writes `.ethograph/alignment.nwb` and a notebook, and you load the session |
| **2 Align media to a recording system, free-running camera** | One camera stream for the whole session, logged by a recorder (Intan, Open Ephys, ...) | The wizard writes a notebook and stops |
| **3 Align media to a recording system, triggered camera** | One video file per trial, each started by the recorder | The wizard writes a notebook and stops |

Modes 2 and 3 need `neuroconv` installed; the wizard says so if it is missing.

## Per-modality folders and patterns

Straight after picking a mode: one tab each for **Video**, **Pose**, **Audio**
— always all three, with no upfront question about which streams exist or
how many cameras/microphones there are. A tab left empty (no folder given)
simply isn't enabled; a folder that resolves to no files at all is flagged as
a mistake, not treated as "no source of this kind". Drop a folder or the
files themselves in the tab(s) you need. A filename pattern is **optional**:

- No pattern drawn: files pair to trials by natural sort (so `t2` comes
  before `t10`) as one device — the answer a single camera gives without
  touching the pattern editor at all.
- A pattern drawn (pick a role — Trial/Camera/Mic — then highlight the
  matching part of an example filename): names 2+ devices, or disambiguates
  trial numbering; the wizard shows which file lands in which cell of the
  table before you go on.

Whether each stream is one file per trial or one file for the whole session
follows the mode picked on page 0 (modes 1 and 3: per trial; mode 2: whole
session) — not a separate question per stream. A pose file with no video
needs its frame rate given explicitly here (never defaulted). Audio's offset
controls (constant, or per-mic) only take effect in mode 2 (free-running/known
offset) — mode 1 (pair) assumes every file is already aligned.

The result is the pairing table: a `trial` column and one `{stream}_{device}`
column per source. Rows are trials.

## Timing (modes 2 and 3 only)

How the camera is wired to the recorder. The choice selects the neuroconv
recipe the notebook will contain.

| Mode | Choices |
|---|---|
| **2 Free-running** | **Known offset** (the video starts a known number of seconds into the recording), or **pulse per frame** (the camera's frame-out pin is on a digital input). Tick **The recorder split the video into files** if one stream came out as several files laid end to end |
| **3 Triggered** | **Trial onsets only** (one trigger per file on a digital input), or **pulse per frame** (frame pulses arrive in one burst per trial) |

Every choice that reads a digital line asks for the line's name as the
recorder's file spells it (for Intan, `DIGITAL-IN-02`).

## Write

- **Rig name**: the notebook is named after it, `wizard/{rig_name}.ipynb` in
  the project folder.
- **Notebook path**: shown, so you can find it.

## What it writes

| Mode | Files |
|---|---|
| 1 | `.ethograph/alignment.nwb` beside the session file, plus `wizard/{rig_name}.ipynb` with the {func}`~ethograph.discover_media` + {func}`~ethograph.pair_media` calls it ran |
| 2, 3 | `wizard/{rig_name}.ipynb` only. Run it; it writes `session.nwb`, which you then select in the **Custom set-up** card. {doc}`../getting_started/video_and_ephys` shows what such a notebook contains |

The notebook is **per rig, not per session**. Its first cell is tagged
`parameters` and holds `session_dir`, the one value to change next time. The
mode, sources, wiring and line names are baked into the cells below it.
