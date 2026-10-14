(target-migrating-from-boris)=
# Migrating from BORIS

[BORIS](https://www.boris.unito.it/) keeps a whole study in one `.boris` file:
an ethogram (the behaviours), the subjects, and a list of **observations**, each
binding one or more media files to events coded in observation time. Ethograph
keeps a study as a **project folder** with one vocabulary (`mapping.txt`) and
one **session folder** per recording, where labels are in trial time (see
{doc}`folder_layout`). The importer maps one onto the other:

| BORIS | Ethograph |
|---|---|
| Project file (`.boris`, `.boris.gz`) | A project folder, or part of one |
| Ethogram (`behaviors_conf`) | `mapping.txt` — behaviour codes become label names, ids assigned alphabetically |
| Subjects (`subjects_conf`) | The session's individuals |
| One **MEDIA observation** | One **session folder**, named after the observation |
| One media file in **Player 1** | One **trial**; the file's duration sets the trial's length |
| State event (START/STOP pair) | An interval, `onset_s` → `offset_s`, in trial time |
| Point event | A row with `event_type = "point"` and `offset_s = NaN` |
| Observation time | Trial time: the cumulative length of the files before it is subtracted |
| LIVE observation | Skipped — there is no media to align the events to |

The import never moves or copies media. A session only records where each file
is, so your videos stay where they are.

## In the GUI

On the start page open **🛠 More tools ▸ Import BORIS projects…**.

1. **Add .boris…** — pick one or more project files. The summary below the list
   says how many sessions, trials and events each will become.
2. **Project folder** — your Ethograph project folder, the one chosen on the
   start page (see {doc}`folder_layout`). It
   receives `mapping.txt` and one session folder per observation. No project
   yet? Make an empty folder and choose it; the folder holding your `.boris`
   files works too, since nothing is copied into it and the sessions only
   reference your media. The `mapping.txt` there receives every behaviour code
   not already in it, so several `.boris` files imported into one project share
   one vocabulary and a behaviour keeps its id across them.
3. **Media folder** — leave empty unless the recordings have moved. The
   `.boris` file stores the full path of every media file, so files that are
   still where BORIS last saw them are found as they are, in as many folders as
   you keep them. A file whose stored path no longer exists is looked for by
   name next to the `.boris` file (BORIS's own fallback) and then in this folder.
4. **Pose folder** — optional. A tracking file whose name starts with the video's
   name (`cam1.mp4` → `cam1DLC.csv`) is paired with it.
5. **Import.** The sessions appear in the project bar's reopen list; pick one and
   press **Load**, or drag the session folder onto the start page.

Media files the importer could not find are listed in the final message. The
session is still written with the stored file names; on load the GUI warns
which videos it cannot find, and the **Video folder** field of the IO panel
points it to them.

## From Python, for an archive

The GUI drives one function. A loop over a folder of `.boris` files migrates a
whole archive into one project folder. `project` below is the Ethograph
{doc}`project folder <folder_layout>`: an existing
one, or a new empty folder that you then choose on the start page. It receives
`mapping.txt` and one session folder per observation; the media stays where it
is.

```python
from pathlib import Path

from ethograph.labels.boris import import_boris_project

project = Path("D:/study")  # the Ethograph project folder: mapping.txt + one session folder per observation

for boris in Path("D:/boris_archive").glob("*.boris"):
    for result in import_boris_project(boris, project):
        print(result.session_dir, result.n_trials, "trials", result.n_labels, "labels")
        if result.missing_media:
            print("  not found:", *result.missing_media)
```

`import_boris_project(boris_path, output_dir, *, observations=None,
video_folder=None, pose_folder=None)` converts every MEDIA observation, or just
the ones named in `observations`. Pass `video_folder` only when the stored
media paths are stale, as in the GUI. It returns one
{class}`~ethograph.labels.boris.BorisSessionResult` per session with the paths
it wrote and the media it could not find. Afterwards, choose `D:/study` as the
project folder on the start page. Sessions made from Python are not in its
reopen list (that list is the GUI's), so open each by dragging its folder onto
the start page.

## Ephys and other timeseries

BORIS codes events on video and audio, and the importer pairs a pose file with
its video, so for those streams the import is the whole job. Ephys, sensor or
pre-computed feature streams are something BORIS never saw, so the importer
cannot place them. It still does half the work: every session folder it writes
has its trials (one per media file) and its labels in trial time. The other
half is a short Python script per session: load the stream, cut it into those
trials on the video's clock, and save it as the session's dataset (`.nc`)
inside the session folder ({doc}`dataset_schema`, {doc}`trials`; for Kilosort
and raw ephys, {doc}`ephys`). A recording that started earlier or later than
the video gets that offset in the alignment ({doc}`media_alignment`), not a
shifted copy of the data.

## What to look at afterwards

The import logs a warning for each label it had to change, and these are the
only ones worth a look:

- **A state spanning two files.** Its START and STOP fall in different trials,
  so it is clipped at the end of the first trial.
- **A state still open at the end of the observation.** It is closed at the end
  of its trial.

One thing is refused rather than changed: an observation whose files run at
different frame rates, because it would become one session with one video
stream. Split it into one observation per rate in BORIS.

Not imported: modifiers, comments and the *Image index* column. Label times are
BORIS's times in seconds, never rounded to a video frame, which is what lets a
label sit beside audio or sensor streams on other clocks. Every imported label
is `labeling_method = manual` with confidence 1, and a behaviour's kind (state
or point) goes into `mapping.txt` the first time the behaviour is seen
({ref}`target-point-events`).
