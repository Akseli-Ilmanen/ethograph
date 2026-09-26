# Folder layout

Three folders play a part. Where the session and project folders live is up to
you:

| Folder | Who writes it | What lives there |
|---|---|---|
| **Session folder** — one per recording | You: the session file. The GUI: labels, alignment, layout. | Any location — inside the project folder or anywhere else. Media folders are separate, absolute paths recorded in the alignment; they may be inside the session folder, and a folder of one session's videos is itself a fine session folder, but nothing requires it. |
| **Project folder** — one per research project | You, on the start page. | Everything that spans sessions: the label vocabulary, pipeline configs, trained models, curation workflows, a list of the sessions made by drag & drop. Your session folders can live here too, but don't have to — Ethograph never copies data into it. |
| `~/.ethograph/` | The GUI. | Your settings, caches, and the **starter project** — the project folder a fresh install begins in. |

## The project folder

**A project folder is always set.** A fresh install starts in the *starter
project*, `~/.ethograph/defaults/` — it ships a `mapping.txt`, the example configs
and the reference geometries, so the GUI works the moment you open it and a quick
look at a dataset needs no set-up at all.

Choose a folder of your own before you start a study: labelling writes the label
names, and models write their configs and runs, into the project folder, and
having that be the starter one means every study shares a single vocabulary. The
start page's **Project folder** bar says which one you are in and links here;
**Clear** returns to the starter project rather than to none.

```{note}
**Changed in a recent version.** The project folder used to be optional, and
`project.yaml` inside it held the individuals, the excluded files and the
skeleton. Neither is true any more — see [what moved where](#what-moved-out-of-projectyaml)
below. Nothing is lost: opening a folder that still has a `project.yaml` folds its
lists into your settings once and tells you it did.
```

Sessions are *listed* from wherever they are, so you can keep your session folders
inside the project (as below) or on another drive. Either way, the folder holds
what you build on top of them:

```
my_project/                            # chosen on the start page
    ├── mapping.txt                    # the project's label_id → name vocabulary
    ├── segment.yaml                   # action-segmentation config (copy from ~/.ethograph/defaults/config/)
    ├── spot.yaml                      # pixel event-spotting config
    │
    ├── config/
    │   ├── skeleton/                  # one YAML per skeleton (crow.yaml, mouse.yaml, …)
    │   └── space/                     # reference geometries for the Space plot
    │
    ├── workflows/                     # curation workflows
    └── wizard/                        # Data wizard notebooks, one per rig
```

```{important}
**One session folder, one `.nc`.** A second `.nc` at the root is an old version,
and Ethograph refuses to guess which one is current — in the GUI and in a
`segment` or `spot` run alike. Name the old one once, on the start page
(**Excluded files…**) or from the dialog that asks; it is kept in your
`gui_settings.yaml` as `ignore_files` and applies in every session folder:

```yaml
ignore_files: [Trial_data.nc, "*_old.nc"]   # file names or globs
```

A file named explicitly (`source: .../Trial_data3.nc`) is always loaded, whatever
the list says. The list is yours rather than the project's on purpose: a project
folder may be copied to another machine, and which file is current is answered by
whoever is sitting in front of it.
```

### What moved out of `project.yaml`

**There is no `project.yaml` any more.** The project folder holds *files* — the
label vocabulary, the pipeline configs, the skeletons, the runs — and nothing you
have to learn a settings schema for. What used to live in that file now lives where
it belongs:

| Used to be | Now |
|---|---|
| `individuals` | `extra_individuals` in `gui_settings.yaml` — added to the names your data declares, edited in **Settings ▸ Create / edit individuals…** or the sidebar's **Edit individuals…** |
| `ignore` | `ignore_files` in `gui_settings.yaml`, edited on the start page |
| `pose.skeleton` | `config/skeleton/{name}.yaml`, one file per skeleton |
| `rig` | Nothing: the wizard names the rig after the session folder |
| `pose.source_software` | Remembered from the last time you answered the drop card |

If you already have a `project.yaml`, choosing that project folder folds its
`individuals` and `ignore` lists into your settings once and says so. The file is
left alone; nothing reads it afterwards.

### Who can be labelled

**The data names its individuals; your own list only adds to them.** First what
the data declares — the session record, else the dataset's `individual` dimension
([movement](https://movement.neuroinformatics.dev)'s convention, and the spelling
a pose file carries: `position (time, individual, keypoint, space)`; the older
plural `individuals` is read too) — then `extra_individuals`. There is no mode and
no override, so the answer is always a superset of what your data declares, and
the dialog shows exactly that: the data's names greyed out on top, yours editable
below.

Until somebody is named *and* a `mapping.txt` holds a class, the Labels tab is
greyed out and the label keys refuse to place anything — there is no individual to
attribute a label to. The tab says which half is missing and opens the dialog that
fixes it.

### The skeleton

One YAML per skeleton in `config/skeleton/`, named by its file (`crow.yaml` →
"crow"), because a study with two rigs has two skeletons. **Settings ▸ Edit
skeleton…** draws one on your pose data and saves it under a name; with no project
folder it goes to your own library under `~/.ethograph/defaults/config/skeleton/`,
so drawing a skeleton never requires a project either.

Precedence: a skeleton you drew for this session outranks everything, then the
data's own (an NWB that carries one) and the library's in the order the dialog
asks for — **the data's by default**, since a file that describes its own skeleton
is usually right. Pick the library's to edit a skeleton without touching an NWB.
A library holding exactly one skeleton needs no choosing.

The starter project, `~/.ethograph/defaults/`, has this same shape and ships with
a `mapping.txt`, the example configs and the geometries — it *is* a project folder,
not a fallback, which is why a fresh install can label and plot straight away. See
{ref}`target-label-mapping` for how the mapping is resolved between a session and
its project.

## The session folder

One per recording. A folder is a session when it holds `.ethograph/alignment.nwb`
(written by the wizard, a notebook or a drag & drop) or an `.nwb` of its own; the
`.nc` files in it are feature layers, and there may be none. Session files are named
by the folder, never by a data file: `labels.tsv`, `metadata.tsv`, `labels/`. On the
start page, open the folder (or any data file in it). Per backend:

::::{tab-set}

:::{tab-item} xarray (.nc)

```
session_01/
    ├── session.nc                     # The dataset (TrialTree or plain Dataset); optional, and only one
    ├── labels.tsv                     # Session labels
    ├── metadata.tsv                   # Trial-level metadata
    ├── .ethograph/
    │   ├── alignment.nwb              # Media paths, trial timing, stream offsets
    │   ├── local_settings.yaml        # Session-specific GUI state
    │   └── mapping.txt                # Optional: overrides the project's for this session
    │
    ├── labels/
    │   ├── backups/
    │   │  └── labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215_101230/
    │       ├── session_predictions.tsv     # labels TSV, labeling_method=automated
    │       └── session_probs.npz           # per-sample class probabilities
    │
    ├── video/
    │   ├── camera1_trial001.mp4
    │   └── camera2_trial001.mp4
    ├── pose/                          # External pose files (DLC, SLEAP, ...)
    │   ├── trial001_pose.h5
    │   └── ...
    ├── audio/
    │   ├── mic1_trial001.wav
    │   └── ...
    └── ephys/
        ├── recording.rhd
        └── kilosort4/
            ├── params.py
            ├── spike_times.npy
            ├── spike_clusters.npy
            ├── channel_positions.npy
            ├── channel_map.npy
            ├── templates.npy
            └── cluster_info.tsv
```
:::

:::{tab-item} NWB (.nwb)

```
session_01/
    ├── session.nwb                    # Self-contained: trials, time series,
    │                                  # pose (PoseEstimationSeries), video
    │                                  # refs (ImageSeries.external_file)
    │
    ├── labels.tsv                     # Session labels
    ├── metadata.tsv                   # Trial-level metadata
    │
    ├── .ethograph/
    │   ├── alignment.nwb              # inherit/overwrite alignment in session.nwb
    │   ├── local_settings.yaml        # Session-specific GUI state
    │   └── mapping.txt                # Optional: overrides the project's for this session
    │
    ├── labels/
    │   ├── backups/
    │   │   └── labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215_101230/
    │       ├── session_predictions.tsv     # labels TSV, labeling_method=automated
    │       └── session_probs.npz           # per-sample class probabilities
    │
    └── video/                         # Only needed if ImageSeries paths
        ├── camera1_trial001.mp4       # no longer point to the right location
        └── camera2_trial001.mp4
```

Pose estimation and other behavioural time series are stored inside the `.nwb`
file — no external tracking folder needed.
:::

:::{tab-item} Pynapple (.npz / folder)

```
session_01/
    ├── position.npz                   # Pynapple Tsd/TsdFrame objects
    ├── speed.npz
    ├── units.npz                      # TsGroup of spike times
    │
    ├── labels.tsv                     # Session labels
    ├── metadata.tsv                   # Trial-level metadata
    │
    ├── .ethograph/
    │   ├── alignment.nwb              # Media paths, trial timing, stream offsets
    │   ├── local_settings.yaml        # Session-specific GUI state
    │   └── mapping.txt                # Optional: overrides the project's for this session
    │
    ├── labels/
    │   ├── backups/
    │   │   └── labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215_101230/
    │       ├── labels_predictions.tsv     # labels TSV, labeling_method=automated
    │       └── labels_probs.npz           # per-sample class probabilities
    │
    ├── video/
    │   ├── camera1_trial001.mp4
    │   └── ...
    └── audio/
        ├── mic1_trial001.wav
        └── ...
```
:::

::::

You write the session file and the media folders. Everything under
`.ethograph/` and `labels/backups/` is created by the GUI on first load and
first save.

## The home folder

```
~/.ethograph/
    ├── gui_settings.yaml              # your layout, playback and dialog folders
    ├── logs/
    ├── cache/                         # video proxies, extracted audio, example data — safe to delete
    └── defaults/                      # the starter project: mapping.txt, config/, …
```
