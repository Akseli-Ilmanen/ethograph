(target-segment-feral)=
# FERAL as a video feature

FERAL {cite:p}`skovorodnikov2025feral` fine-tunes a video foundation model
(V-JEPA 2) to predict a class per frame from the pixels alone: no pose, no
features. It is a strong *what* detector; where it is weaker is the exact
frame a behaviour starts and stops, because its 64-frame chunk covers a
fraction of a second at high frame rates and the attention pooling averages
over it. Here it is a **video feature**, like S3D or DINOv2 in
{doc}`video_features`: its per-frame embedding becomes a variable `feral`
on every trial, which a segmentation model reads next to the changepoint
and kinematic columns that know *where*.

FERAL runs in its own environment: its dependency pins clash with the GUI's,
so it is never installed alongside ethograph. What crosses is files. The
project writes FERAL's inputs into `{root}/feral/`; you train and infer
there with FERAL's own commands; the embeddings it writes attach to every
session the next time it opens under this config.

```{note}
Exporting embeddings needs FERAL's `--save_embeddings` option, which is a
[pull request](https://github.com/Akseli-Ilmanen/feral/pull/1) against
FERAL at the time of writing. Until it is merged, install FERAL from that
branch (below). Without it FERAL still trains and predicts — and its
predictions still score in `feral/answers/` — but nothing comes back into
the project.
```

## The two environments

**FERAL's**, once, following [FERAL's own README](https://github.com/Skovorp/feral)
(a CUDA GPU of compute capability 8.0 or newer, PyTorch 2.5 or later):

```bash
conda create -n feral python=3.11 && conda activate feral
pip install torch torchvision          # a CUDA build; see FERAL's README for the version table
pip install feral                      # release
pip install git+https://github.com/Akseli-Ilmanen/feral@export-embeddings   # or: with --save_embeddings
```

**ethograph's** is where everything else happens: the config, the export,
the segmentation model, cross-validation, the GUI.

## One config, one extra key

`video_features.extractor: feral` selects it. Two settings are FERAL's own
and are refused with any other extractor:

```yaml
sessions:
  - source: D:\data\ses-01\behav\Trial_data.nc
    video_dir: D:\data\ses-01\videos          # if the alignment names the files without a folder
  - source: D:\data\ses-02\behav\Trial_data.nc
    video_dir: D:\data\ses-02\videos
  - source: D:\data\ses-03\behav\Trial_data.nc
    video_dir: D:\data\ses-03\videos

individual: Freddy                               # one animal per video — see below

video_features:
  extractor: feral
  camera: cam-1                                  # which camera's file, when the alignment holds several
  context_s: 0.64                                # seconds one FERAL chunk spans; null = every frame
  preset: lite                                   # FERAL's recipe: lite, max, rare; null = its default

features:
  columns:
    feral: {feral_dims: 0..767}                  # the embedding, once FERAL has written it
    speed_changepoints_prox: {}                  # ... next to whatever else the session holds
  labels: {mapping: ../mapping.txt, branch: 0}

train:
  run_name: feral_c2f
  split:
    holdout_sessions: [D:\data\ses-03\behav\Trial_data.nc]   # the true test set — see Leakage
```

`context_s` is the one temporal setting, in seconds. FERAL's chunk is 64
frames spread `chunk_step` frames apart; the export reads each video's rate
and resolves the stride so that the chunk spans `context_s`, then scales
FERAL's chunk shifts by the same stride so overlapping chunks keep
predicting the same frames. `null` is FERAL's own chunk (every frame),
whatever the rate. On a 200 fps recording of 0.2 s behaviours, 0.64 s
(every second frame) placed boundaries far better than 2 s did; on 25–30
fps video, `null` is FERAL's published setting.

`preset` is FERAL's `--mode`: `lite` is its smallest backbone fully
fine-tuned, `max` its published CalMS21 recipe, `rare` its rare-class
recipe. The config ethograph writes is FERAL's packaged default with the
preset laid over it, exactly as `feral train --mode` does, so anything you
want to change further you change in `feral/config.yaml` — it is FERAL's
file, and FERAL reads it verbatim.

## The round trip

**1. Export, in ethograph.** With labels curated in the GUI:

```python
import ethograph as eto

project = eto.segment.Project("config/segment.yaml")
project.video_features()        # for extractor: feral, this is the export
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
`train.split`, the same draw `train()` makes; `inference` is every video,
because the embeddings are what the project trains on. A class no
training video holds is background for FERAL (its inverse-frequency class
weight would otherwise swamp the loss) and `export.yaml` lists it.

**2. Train and infer, in FERAL's environment.** The log prints these with
your paths filled in:

```bash
conda activate feral
feral train-config D:\project\feral\config.yaml
feral infer D:\project\feral\checkpoints\feral_c2f_best_checkpoint.pt D:\data\ses-01\videos --save_embeddings D:\project\feral\embeddings
feral infer D:\project\feral\checkpoints\feral_c2f_best_checkpoint.pt D:\data\ses-02\videos --save_embeddings D:\project\feral\embeddings
feral infer D:\project\feral\checkpoints\feral_c2f_best_checkpoint.pt D:\data\ses-03\videos --save_embeddings D:\project\feral\embeddings
```

`feral infer` takes one flat folder of videos, so it runs once per video
folder; every call adds `{video stem}.npy` files to the same
`feral/embeddings/`. FERAL's own scores on its `val` and `test` splits are
in `feral/answers/` — that is FERAL alone, with no features, and worth a
look before going on.

**3. Back in ethograph**, nothing to import: a session opened under this
config finds its videos' files in `feral/embeddings/` and carries the
variable `feral` (a `(time, feral_dims)` array, memory-mapped and sampled
onto the trial clock the way every video feature is). Then the pipeline is
the usual one:

```python
project.materialise()           # feral + changepoint columns → the materialised dataset
result = project.train()        # holdout_sessions is the test set
project.compare()
```

Drop `feral/embeddings/` onto the GUI window to see the embedding as a
heatmap under the video, exactly as {doc}`video_features` describes for any
folder of `.npy` files.

## Leakage: which sessions FERAL may see

A FERAL fine-tuned on a session's labels embeds that session better than
one it never saw — it has, in effect, been shown the answers. A
segmentation model trained on those embeddings then learns to trust them
more than it should, and scores that session too well. **Any downstream
score on a session FERAL trained on is inflated**, and `cross_validate()`
over FERAL features warns about exactly this: every fold but one holds out
a session FERAL has seen.

`train.split.holdout_sessions` is the guard, and it does double duty. The
sessions it names are FERAL's `test` and `inference` only — never `train`
or `val` — so their embeddings come from a model that never saw their
labels; and downstream they are the test set, as they always were. Name
your true test sessions there **before** exporting, and `train()`'s test
score is a number you can report. FERAL's own test score in
`feral/answers/` is then the video-only baseline on the same sessions.

Cross-validation over FERAL features is still useful for *comparing*
feature sets on the same inflated footing (with and without changepoints,
say); it is not a held-out score. A clean cross-validation would need one
FERAL per fold, each exported from a model that never saw its fold — that
is K exports and K FERAL trainings, and the project does not do it for
you.

## Without ethograph's models

If you only want FERAL — no changepoints, no segmentation model — the
export is the whole story: a minimal config with the sessions, the labels
and the camera, and `eto.segment.export_feral` writes `feral/` for it. The
labels you curate in the GUI become FERAL's `labels.json` without the
chunking, class-name and split bookkeeping being done by hand.

```yaml
sessions:
  - source: D:\data\ses-01\behav\Trial_data.nc
    video_dir: D:\data\ses-01\videos
  - source: D:\data\ses-02\behav\Trial_data.nc
    video_dir: D:\data\ses-02\videos
individual: Freddy
features:
  columns: {feral: {}}                           # never materialised; a config names at least one column
  labels: {mapping: ../mapping.txt, branch: 0}
video_features: {extractor: feral, camera: cam-1, preset: lite}
```

```python
import ethograph as eto

eto.segment.export_feral(eto.segment.load_config("config/feral_only.yaml"))
```

## What a FERAL video is

**One trial of one individual.** FERAL labels the whole frame with one
class, so a session with several animals in one camera needs one cropped
video per animal first (a `video_dir` per individual, and one config per
individual with `individual:` set); an export with more than one
individual is refused naming this.

The video may start before its trial or run past it — the frames outside
the trial are labelled background and `videos.tsv` counts them — but a
file holding several trials is refused once more than half its frames lie
outside the trial, since FERAL would learn the neighbours' behaviour as
background. Cut such recordings into one clip per trial first.

All videos share one frame rate: a FERAL config has one chunk stride.
Videos on different drives are refused too, since FERAL reads everything
under one folder prefix.
