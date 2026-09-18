(target-segment-feral)=
# FERAL

FERAL {cite:p}`skovorodnikov2025feral` fine-tunes a video foundation model
(V-JEPA 2) to predict a class per frame from the pixels alone: no pose, no
features. There are two ways to use it from ethograph, and both start the
same way: the labels you curate in the GUI become FERAL's `labels.json`,
without the chunking, class-name and split bookkeeping being done by hand.

1. **{ref}`FERAL's labels <target-feral-labels>`** — the simple one. Label
   in ethograph → export → train and predict in FERAL → import its
   predicted segments → curate them in the GUI. No segmentation model, no
   features; the released FERAL is enough.
2. **{ref}`FERAL as a video feature <target-feral-feature>`** — label in
   ethograph → export → train FERAL and save its per-frame **embeddings** →
   they attach to every trial as a variable `feral`, optionally next to
   pose, changepoint or sensor columns → train a segmentation model on them
   → curate its predictions in the GUI.

Why the second, for two reasons:

- **Other features join the video.** Exported as an embedding, FERAL is one
  input among several: pose, IMU or any other session variable sits next to
  it in the segmentation model.
- **Temporal context without losing temporal precision.** A FERAL chunk is
  always 64 frames: either 64 consecutive ones, or with `chunk_step` > 1
  every n-th frame over a longer span, which gives up temporal precision.
  On high-frame-rate video of behaviours that need a long context, one
  solution is to export at a stride of 1: FERAL learns short-context
  embeddings on every frame, and the segmentation model bridges the longer
  context.

Start with the first; move to the second when either applies.

## The two environments

FERAL runs in its own environment: its dependency pins clash with the GUI's,
so it is never installed alongside ethograph. What crosses is files. The
project writes FERAL's inputs into `{root}/feral/`; you train and infer
there with FERAL's own commands; what FERAL writes is read back from the
same folder.

**FERAL's**, once, following [FERAL's own README](https://github.com/Skovorp/feral)
(a CUDA GPU of compute capability 8.0 or newer, PyTorch 2.5 or later):

```bash
conda create -n feral python=3.11 && conda activate feral
pip install torch torchvision          # a CUDA build; see FERAL's README for the version table
pip install feral
```

**ethograph's** is where everything else happens: the config, the export,
the import, the GUI — and, for the second workflow, the segmentation model.

(target-feral-labels)=
## Workflow 1: FERAL's labels

**1. A config**, naming the sessions, the labels and the camera:

```yaml
sessions:
  - source: D:\data\ses-01\behav
    video_dir: D:\data\ses-01\videos          # if the alignment names the files without a folder
  - source: D:\data\ses-02\behav
    video_dir: D:\data\ses-02\videos
individual: bird1                                # a FERAL video is one trial of one animal
features:
  columns: {feral: {}}                           # never materialised here; a config names at least one column
  labels: {mapping: ../mapping.txt, branch: 0}
video_features:
  extractor: feral
  camera: cam-1                                  # which camera's file, when the alignment holds several
  context_s: null                                # seconds one FERAL chunk spans; null = every frame
  preset: lite                                   # FERAL's recipe: lite, max, rare; null = its default
train:
  run_name: crows
```

`context_s` and `preset` set FERAL's own parameters (`chunk_step` and
`--mode`); [FERAL's config documentation](https://www.getferal.ai/#config-docs)
explains them and every other one, and {ref}`target-feral-tuning` says where
to change them.

**2. Export, in ethograph.** With labels curated in the GUI:

```python
import ethograph as eto

project = eto.segment.Project("config/feral.yaml")
project.export_feral()
```

writes `{root}/feral/`:

```
feral/
    labels.json        # FERAL's label file: a class per frame of every video, the splits
    config.yaml        # a complete FERAL config: feral train-config reads it
    videos.tsv         # which session, trial and individual each video is
    export.yaml        # FERAL class → label id, the rate, the stride, the held-out sessions
    export.log
```

and logs the commands for the next step. Each trial's *curated* labels are
rasterised on its own video's frames through the alignment (trial = video
+ offset), so the label list is exactly as long as the video. The
`train` / `val` / `test` roles are drawn by whole trial from
`train.split`; `inference` is every video. A class no training video holds
is background for FERAL (its inverse-frequency class weight would otherwise
swamp the loss) and `export.yaml` lists it.

**3. Train and predict, in FERAL's environment.** The log prints these with
your paths filled in:

```bash
conda activate feral
feral train-config D:\project\feral\config.yaml
feral infer D:\project\feral\checkpoints\crows_best_checkpoint.pt D:\data\ses-01\videos --output D:\project\feral\predictions\ses-01__videos.json
feral infer D:\project\feral\checkpoints\crows_best_checkpoint.pt D:\data\ses-02\videos --output D:\project\feral\predictions\ses-02__videos.json
```

`feral infer` takes one flat folder of videos, so it runs once per video
folder, each writing its own JSON into `feral/predictions/`. Copy the
`--output` paths from the log: the import looks for exactly those names.
FERAL's own scores on its `val` and `test` splits are in `feral/answers/`.

**4. Import, back in ethograph:**

```python
project.import_feral_predictions()
```

Every video's per-frame probabilities are placed on its trial's clock
through the alignment (frames outside the trial are dropped), the most
probable class per frame becomes intervals with the project's own label
ids, and `infer.postprocess` (minimum duration, stitching, snapping to
changepoints — all off unless you set them) runs over them as it does for
any model. Each session gets a prediction set
`labels/predictions_feral_{run_name}_{timestamp}/`, in the GUI's labels
format with `labeling_method=automated`, FERAL's probabilities beside it
for the confidence overlay.

**5. Curate in the GUI.** Open the session and use **File → Import
predictions → Import as Labels…** on that folder; the predictions draw
dotted until you accept or edit them
({doc}`/getting_started/labels/importing`). Curated labels are training
targets for the next export.

(target-feral-feature)=
## Workflow 2: FERAL as a video feature

Here FERAL is a video feature like S3D or DINOv2 in {doc}`video_features`:
its per-frame embedding becomes the variable `feral` on every trial, and a
segmentation model reads it.

```{note}
Exporting embeddings needs FERAL's `--save_embeddings` option, which is a
[pull request](https://github.com/Akseli-Ilmanen/feral/pull/1) against
FERAL at the time of writing. Until it is merged, install FERAL from that
branch instead of the release:
`pip install git+https://github.com/Akseli-Ilmanen/feral@export-embeddings`.
```

**1. The config** is the same one, with the columns the segmentation model
reads and the sessions to hold out:

```yaml
sessions:
  - source: D:\data\ses-01\behav
    video_dir: D:\data\ses-01\videos
  - source: D:\data\ses-02\behav
    video_dir: D:\data\ses-02\videos
  - source: D:\data\ses-03\behav
    video_dir: D:\data\ses-03\videos

individual: bird1

video_features:
  extractor: feral
  camera: cam-1
  context_s: 0.64
  preset: lite

features:
  columns:
    feral: {feral_dims: 0..767}                  # the embedding, once FERAL has written it
    speed_changepoints_prox: {}                  # ... next to whatever else the session holds
  labels: {mapping: ../mapping.txt, branch: 0}

train:
  run_name: feral_c2f
  split:
    holdout_sessions: [D:\data\ses-03\behav]   # the true test set — see Leakage
```

**2. Export** as above; `project.video_features()` is the same export for
`extractor: feral`.

**3. Train and infer in FERAL's environment**, adding `--save_embeddings`
to every `feral infer` (the log names the folder):

```bash
conda activate feral
feral train-config D:\project\feral\config.yaml
feral infer D:\project\feral\checkpoints\feral_c2f_best_checkpoint.pt D:\data\ses-01\videos --output D:\project\feral\predictions\ses-01__videos.json --save_embeddings D:\project\feral\embeddings
feral infer D:\project\feral\checkpoints\feral_c2f_best_checkpoint.pt D:\data\ses-02\videos --output D:\project\feral\predictions\ses-02__videos.json --save_embeddings D:\project\feral\embeddings
feral infer D:\project\feral\checkpoints\feral_c2f_best_checkpoint.pt D:\data\ses-03\videos --output D:\project\feral\predictions\ses-03__videos.json --save_embeddings D:\project\feral\embeddings
```

Every call adds `{video stem}.npy` files to the same `feral/embeddings/`.
The `--output` JSONs still import as in workflow 1: FERAL alone, the
video-only baseline to compare the segmentation model against.

**4. Back in ethograph**, nothing to import: a session opened under this
config finds its videos' files in `feral/embeddings/` and carries the
variable `feral` (a `(time, feral_dims)` array, memory-mapped and sampled
onto the trial clock the way every video feature is). Then the pipeline is
the usual one:

```python
project.materialise()  # feral + changepoint columns → the materialised dataset
result = project.train()  # holdout_sessions is the test set
project.compare()
project.inference()  # a prediction set beside every session
```

**5. Curate in the GUI**, as in workflow 1. Drop `feral/embeddings/` onto
the GUI window to see the embedding as a heatmap under the video, exactly
as {doc}`video_features` describes for any folder of `.npy` files.

### Leakage: which sessions FERAL may see

A FERAL fine-tuned on a session's labels embeds that session better than
one it never saw, so **any downstream score on a session FERAL trained on
is inflated**. `train.split.holdout_sessions` is the guard: the sessions it
names are FERAL's `test` and `inference` only, never `train` or `val`, and
downstream they are the test set. Name them **before** exporting, and
`train()`'s test score is one you can report, with FERAL's own score in
`feral/answers/` as the video-only baseline on the same sessions.
`cross_validate()` over FERAL features warns about this: it still compares
feature sets on the same inflated footing, but it is not a held-out score
(that would take one FERAL training and export per fold, which the project
does not do for you).

(target-feral-tuning)=
## Tuning FERAL

FERAL's parameters are FERAL's, and
[its config documentation](https://www.getferal.ai/#config-docs) is where
they are explained. `export_feral()` writes `feral/config.yaml` as a complete FERAL
config: its packaged default, with only the paths, the run name and the two
settings below filled in. Edit that file as you see fit before
`feral train-config`; FERAL reads it verbatim, and the import does not
depend on anything in it. Exporting again rewrites it, so keep a copy of
your edits.

The ethograph config sets two of FERAL's parameters, both refused with any
other extractor:

- `video_features.preset` is FERAL's `--mode` (`lite`, `max`, `rare`;
  `null` = its default), laid over the default as `feral train --mode` does.
- `video_features.context_s` is the seconds one chunk spans. The export
  reads the videos' rate, resolves FERAL's `chunk_step` from it and scales
  the chunk shifts by the same stride; `null` is every frame
  (`chunk_step: 1`), whatever the rate.
