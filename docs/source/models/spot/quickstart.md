(target-spot-quickstart)=
# Quickstart

Here is a minimal setup for spotting point events from video: three sessions
with curated point labels, one camera, and plain E2E-Spot {cite:p}`hong2022e2espot`
on pixels only. It trains on two sessions, is scored on the third, and writes
predictions you then open in the GUI next to the labels you drew.

Everything here is a default. {doc}`index` is the same pipeline with the
choices put back in: pose features, the teacher, distillation, MSAGSM.

ethograph does not install PyTorch for you. Install it first, then the extra
(see *Train models* in {doc}`../../getting_started/installation`):

```bash
uv pip install --torch-backend=auto torch torchvision
uv pip install "ethograph[model]"
```

The trainer is upstream's own E2E-Spot code, run from a clone. Clone it as
`spot/` beside the ethograph repository, or put it anywhere and set
`ETHOGRAPH_SPOT_ROOT` to that folder:

```bash
git clone https://github.com/jhong93/spot.git
```

## 1. What a session needs

- **Point labels.** Label the events in the GUI as usual. You get
  `{name}_labels.tsv` beside the session file, with the label ids in
  `mapping.txt`. Only `manual` and `curated` labels are training targets.
- **A video per trial.** The alignment already knows the video path, frame
  rate and offset for each trial. If the paths it holds are not valid on this
  machine, add `video_dir` to the session line.

No features, and no preprocessing. The model reads the frames.

## 2. `spot.yaml`

Put this beside your data. It is the whole config:

```yaml
sessions:
  - source: ses-01.nc
  - source: ses-02.nc
  - source: ses-03.nc                # the held-out session, named below

labels:
  classes: [31, 32]                  # the point-event label ids to spot, in the order they happen
  camera: cam-1                      # one camera per project

train:
  split:
    train_fraction: 0.8
    val_fraction: 0.2
    test_fraction: 0.0
    holdout_sessions: [ses-03.nc]    # every trial of this session is `test`
```

Three things worth knowing about it:

- **`labels_path` is left out** because it defaults to `{stem}_labels.tsv`
  beside the source, which is where the GUI writes it.
- **No `clip:` section.** The model sees 2 s of video at a time
  (`context_s`). The label grid (`resolution_ms`) is left unset, so it is as
  fine as your GPU's memory allows. Both are durations and are converted using
  each video's own frame rate, so the same file works at 60 fps and at 200 fps.
  See {ref}`target-spot-seconds`.
- **No `features:` section.** That is what makes this option 2 in
  {doc}`index`: pixels in, events out.

## 3. Train and score

```python
import ethograph as eto

project = eto.spot.Project("spot.yaml")
project.materialise()               # every trial's video -> frames/ + E2E-Spot's index; resumable
result = project.train()            # runs/ctx2s_res…ms/
metrics = project.evaluate()        # ses-03, never trained on -> test_metrics.yaml

print(result.run_dir)
print(metrics)                      # per class: misses, spurious, error in ms, hit rate at 10/20/50/100 ms
```

`materialise()` decodes every trial to JPEGs once; later runs and folds reuse
them. `train()` runs for the whole `train.epochs` budget. The epoch it keeps
is the one with the fewest misses on the validation trials, not the last one.

```{note}
`train()` checks up front whether the clip fits on your GPU and stops with an
error if it does not, naming the duration to shorten. On a small card, try
`eto.spot.Project("spot.yaml", "clip.context_s=1.5")`.
```

## 4. Look at the mistakes in the GUI

```python
paths = project.inference(sessions=["ses-03.nc"])
print(paths[0])
# labels/predictions_spot_ctx2s_res…ms_20260914_151203/ses-03_predictions.tsv
```

That folder sits beside `ses-03.nc`. Each call writes a new one, so an earlier
run is never overwritten. Inference reads the video directly; it does not
export frames for this session. Open ses-03 in the GUI and load the TSV with
**File ▸ Import labels…**. Every predicted event arrives as `automated`, drawn
dotted, and carries a `confidence` read from the shape of its curve. The
curves are saved next to the TSV, so frame-by-frame review shows where the
model hesitated. See {doc}`../curation`.

## Where to go from here

- **A wider temporal aperture**: `model.architecture=rny008_msagsm`, then
  `project.compare()` shows the two runs side by side (see {doc}`index`).
- **You have pose**: list pose variables under `features:` and the model reads
  them next to the pixels. If pose exists only for the training sessions,
  distil a pose teacher instead. See {doc}`multimodal`.
- **Every session held out in turn**: `project.cross_validate()`, so each
  session gets predictions from a model that never saw it.
- **Every key**, with its default: {doc}`config`.
