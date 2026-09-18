# Folder layout

Three folders play a part. Where the session and project folders live is up to
you:

| Folder | Who writes it | What lives there |
|---|---|---|
| **Session folder** — one per recording | You: the session file. The GUI: labels, alignment, layout. | Any location — inside the project folder or anywhere else. Media folders are separate, absolute paths recorded in the alignment; they may be inside the session folder, and a folder of one session's videos is itself a fine session folder, but nothing requires it. |
| **Project folder** — one per research project | You, on the start page. | Everything that spans sessions: the label vocabulary, pipeline configs, trained models, curation workflows, a list of the sessions made by drag & drop. Your session folders can live here too, but don't have to — Ethograph never copies data into it. |
| `~/.ethograph/` | The GUI. | Your settings, caches, and a starter project used while no project folder is chosen. |

## The project folder

Chosen once on the start page (**Project folder**) and remembered across
restarts. Sessions are *listed* from wherever they are, so you can keep your
session folders inside it (as below) or on another drive. Either way, the
folder holds what you build on top of them:

```
my_project/                            # chosen on the start page
    ├── data/                          # optional: your session folders, if you keep them here
    ├── mapping.txt                    # the project's label_id → name vocabulary
    ├── project.yaml                   # study defaults: individuals, cameras, mics, rig, pose software, skeleton
    ├── config/
    │   ├── segment.yaml               # action-segmentation config (copy from ~/.ethograph/defaults/config/)
    │   ├── spot.yaml                  # pixel event-spotting config
    │   └── space/                     # reference geometries for the Space plot
    ├── runs/
    │   └── lightgbm/                  # lightgbm models trained from the Model menu
    ├── feral/                         # FERAL's inputs, checkpoints and embeddings (models/segment/feral)
    ├── workflows/                     # curation workflows
    ├── wizard/                        # Data wizard notebooks, one per rig
    └── sessions.txt                   # session folders made by drag & drop with this project set, reopenable
```

```{important}
**One session folder, one `.nc`.** A second `.nc` at the root is an old version,
and Ethograph refuses to guess which one is current — in the GUI and in a
`segment` or `spot` run alike. Tell the project which files to skip, once, in
`project.yaml`:

```yaml
ignore: [Trial_data.nc, "*_old.nc"]   # file names or globs, matched in every session folder
```

The GUI offers this as a button when it hits the case. A file named explicitly
(`source: .../Trial_data3.nc`) is always loaded, whatever the list says.
```

`project.yaml` holds what recurs across the study's sessions. Every key is
optional and every key is a default: a session's own record (the individuals the
wizard wrote) or its dataset overrides it. Machine paths never go in it, so it can
be committed and shared.

```yaml
individuals: [crow1, crow2]
ignore: [Trial_data.nc]        # old dataset versions to skip in every session folder
cameras: [cam-1, cam-2]
mics: [mic-1]
rig: CrowBench                 # the wizard notebook under wizard/ to start from
pose:
  source_software: DeepLabCut
  skeleton:                    # the skeleton itself, same shape as the skeleton editor saves
    keypoints: [beak, head, tail]
    connections:
      - {start: beak, end: head}
      - {start: head, end: tail}
```

Without a project folder, `~/.ethograph/defaults/` stands in — it has the same
shape and ships with a default `mapping.txt`, example configs and geometries.
See {ref}`target-label-mapping` for how the mapping is resolved between the
session, the project and that backup.

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
