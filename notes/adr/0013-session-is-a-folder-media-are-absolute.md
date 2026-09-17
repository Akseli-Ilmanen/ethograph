# ADR 0013 — A session is a folder; media folders are absolute

**Status:** accepted (2026-09-16). Steps 1 and 2 implemented, except the
rename of `alignment.nwb` to `session.nwb`, which is deferred (52 files, no
behaviour change) and stays specified below.

## Context

The wizard-generated notebook wrote `session_dir = "C:\Users\..."` and
used one `session_dir` for two things: the folder that receives
`session.nc` and `.ethograph/alignment.nwb`, and the root that every media
folder was spelled relative to. The wizard derived it as *the parent of
the first media folder*, so a `VidData/20241217_01_Crow1/` video folder
made `VidData/` the session and put the session name inside `rig`. "The
one thing to change next session" was then the wrong thing.

Underneath: `discover_media` returned basenames and threw the folder away,
so `pair_media` needed a single `media_root`, which cannot serve two
cameras in two folders, or video and pose in different trees. And the
`.nc` file was carrying two jobs: feature container, and session identity
(what is opened, what labels and settings are named after, owner of the
`individuals` coordinate). A video-only session therefore needed a `.nc`
whose only content was `individuals = ["individual_1"]`, and adding an
individual meant editing a data file.

## Decisions

### Step 1 — pairing and the notebook

- **`session_dir` is the folder that holds `.ethograph/`** (and any
  `.nc`). It keeps its name. It may be a media folder, it need not be.
- **Media folders are absolute, one per source.** `SourceSpec.folder` and
  `RigSource.folder` are absolute paths; nothing is relative to
  `session_dir`. The notebook's parameters cell says honestly: change
  `session_dir` and every `folder`.
- **The pairing table carries full paths.** `discover_media(sources)`
  takes no root; each cell is the file's full path. `pair_media` writes
  the basename into the trials table and probes the full path. `media_root`
  is gone, no alias.
- **`pair_media` records each stream's folder in the NWB** so the GUI can
  find the media after the notebook route without asking again; the GUI
  folder settings default to it and may override it.
- **`alignment.nwb` has one home: `{session_dir}/.ethograph/`** in every
  mode, the notebook and the GUI builder alike.
- **The trial metadata file is an absolute path** in `rig`, read directly.
- **The wizard prefills `session_dir`** with the top-ranked media folder by
  the kind hierarchy below; the field is editable.
- **Docs show the general case**: absolute media folders next to a
  separate session folder; the tidy `session/video/` layout is the special
  case where they coincide.

### Step 2 — folder sessions

- **A session is a folder.** It is a session iff it holds
  `.ethograph/alignment.nwb` or a root `.nwb`. A folder with a dataset but
  neither is promoted on first open: ask once, derive the trials table
  from the dataset, write the alignment. Neither dataset nor alignment:
  not a session, offer the wizard. A root `.nwb` *and* a sidecar is
  refused.
- **The record is `.ethograph/session.nwb` in every mode**, hidden and
  derived; a user's own NWB source stays where it is and is edited in
  place. (`alignment.nwb` is renamed when step 2 lands.)
- **Individuals live in the session record** (a small `individuals` table
  in the NWB), written by `pair_media(..., individuals=[...])`, edited in
  the GUI. A `.nc` individual dim is checked against it: a name not in the
  session list is a load error; the dim only restricts feature plots.
- **Labels are `labels.tsv`, one per folder**, with `labels/backups/` and
  `labels/predictions_*/` beside it. On first open, a lone
  `{stem}_labels.tsv` is renamed once and logged.
- **One `.nc` per session folder.** (Amended the same day: "several `.nc`
  are layers" was a guess; video features live as `.npy` folders, so the only
  real case is old versions.) A second root `.nc` raises
  `AmbiguousSessionError`; the remedy is the project's `ignore` list in
  `project.yaml`, offered by the GUI as a button, and a file named by
  `source:` is always loaded. Files are never moved. One backend per folder:
  `.nwb`/pynapple beside `.nc` is refused.
- **Opening takes folders only.** Browse and recent entries name folders;
  a file names its folder. The IO widget shows the session folder with the
  feature files listed beneath it.
- **Drop rule.** Files from one folder: that folder is the session,
  `.ethograph/` is created there, no prompt. Files from several folders:
  one popup, "Where to put session files (alignment, labels, metadata)?",
  radio per source folder labelled by kind and count plus "Other folder…".
  Preselection follows the kind hierarchy `.nc` > `.npz` > `.npy` > video
  > pose > audio. "Remember my choice", on by default, promotes the chosen
  *kind* to the front of the hierarchy as a global preference; folder
  paths are never remembered. `projects/sessions/<timestamp>/` is no
  longer an automatic destination.

## Consequences

- One notebook, one rule for where things are; media anywhere.
- `media_root` and `_relative` disappear; the duration probe reads paths
  straight from the table.
- Step 2 touches `classify_files`, the loader dispatch, label and settings
  paths, the individual combo and the IO widget, and gets its own tests.
