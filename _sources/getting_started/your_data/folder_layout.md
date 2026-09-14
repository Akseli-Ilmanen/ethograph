# Folder layout

Three folders play a part. Where the session and project folders live is up to
you:

| Folder | Who writes it | What lives there |
|---|---|---|
| **Session folder** — one per recording | You: the session file and the media. The GUI: labels, alignment, layout. | The data. Any location — inside the project folder or anywhere else; media folders are selected in the GUI and can live elsewhere too. |
| **Project folder** — one per research project | You, on the start page. | Everything that spans sessions: the label vocabulary, pipeline configs, trained models, curation workflows, kept drag & drops. Your session folders can live here too, but don't have to — EthoGraph never copies data into it. |
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
    ├── config/
    │   ├── segment.yaml               # action-segmentation config (copy from ~/.ethograph/defaults/config/)
    │   ├── spot.yaml                  # pixel event-spotting config
    │   └── space/                     # reference geometries for the Space plot
    ├── runs/
    │   └── lightgbm/                  # lightgbm models trained from the Model menu
    ├── workflows/                     # curation workflows
    ├── wizard/                        # Data wizard notebooks, one per rig
    └── sessions/                      # drag & drops made with this project set
        └── 2026-09-06_21-47-12/       # one timestamped folder per drop, reopenable
```

Without a project folder, `~/.ethograph/defaults/` stands in — it has the same
shape and ships with a default `mapping.txt`, example configs and geometries.
See {ref}`target-label-mapping` for how the mapping is resolved between the
session, the project and that backup.

## The session folder

One per recording, per backend:

::::{tab-set}

:::{tab-item} xarray (.nc)

```
session_01/
    ├── session.nc                     # Behavioural dataset (TrialTree or plain Dataset)
    ├── session_labels.tsv             # Session labels
    ├── session_metadata.tsv           # Trial-level metadata
    ├── .ethograph/
    │   ├── alignment.nwb              # Media paths, trial timing, stream offsets
    │   ├── local_settings.yaml        # Session-specific GUI state
    │   └── mapping.txt                # Optional: overrides the project's for this session
    │
    ├── labels/
    │   ├── backups/
    │   │  └── session_labels_20240315_101230.tsv
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
    ├── session_labels.tsv             # Session labels
    ├── session_metadata.tsv           # Trial-level metadata
    │
    ├── .ethograph/
    │   ├── alignment.nwb              # inherit/overwrite alignment in session.nwb
    │   ├── local_settings.yaml        # Session-specific GUI state
    │   └── mapping.txt                # Optional: overrides the project's for this session
    │
    ├── labels/
    │   ├── backups/
    │   │   └── session_labels_20240315_101230.tsv
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
