(target-segment-quickstart)=
# Quickstart

Here is a minimalistic setup for training an action segmentation model: Three curated sessions, a few kinematic
features, one `c2f_tcn` {cite:p}`singhania2021c2ftcn` trained on two of them and judged on the third — whose
predictions you then open in the GUI beside the labels you drew.

Everything here is a default. {doc}`guide` is the same pipeline with the
choices put back in.

ethograph does not install PyTorch for you. Install it first, then the extra
(see *Train models* in {doc}`../../getting_started/installation`):

```bash
uv pip install --torch-backend=auto torch torchvision
uv pip install "ethograph[model]"
```

## 1. Put the features in the session file

The pipeline never computes features; it selects variables that are already in
your `.nc`. Which basic kinematic features to add depends on the animal:

- **Orientation relatively fixed** (head-fixed, perched, a fixed camera
  view of a limb): `position`, `velocity` and `speed`. Arena coordinates
  already mean the same thing in every trial.
- **Freely moving**: `position_ego` and `speed`. Egocentric position is
  centred on one keypoint and rotated so the animal faces +x, so a behaviour
  reads the same whichever way the animal is turned.

Either way, add `intra`: the distance between each pair of keypoints, which
describes body posture (stretched, curled) regardless of position or heading.

```python
from movement.io import load_poses
from movement.kinematics import compute_speed, compute_velocity

from ethograph.features.geometry import egocentric_position, intra_distances

ds = load_poses.from_dlc_file("ses-01_pose.csv", fps=60)
ds["speed"] = compute_speed(ds.position)  # (time, keypoint, individual)
ds["intra"] = intra_distances(ds.position)  # (time, pair, individual), pair = "snout-tailBase", …

# orientation fixed
ds["velocity"] = compute_velocity(ds.position)  # (time, space, keypoint, individual)

# freely moving: centre on tailBase, heading tailBase → snout
ds["position_ego"] = egocentric_position(ds.position, "tailBase", heading_keypoint="snout")

ds.to_netcdf("/data/sessions/ses-01/session.nc")  # one folder per session; the wizard or a drop adds .ethograph/
```

Do that for each session. The same variables are then plottable in the GUI, so
you can look at exactly what the model will read (`examples/create_dataset_cricket.ipynb`
is this, end to end, with video and audio alignment).

Labels come out of the GUI as they always do: `labels.tsv` in the session folder beside the
`.nc`, with the label ids described in a `mapping.txt`.

## 2. `segment.yaml`

Put this beside your data. It is the whole config — every key not written
here has a default that is fine for a first run.

```yaml
sessions:
  - source: /data/sessions/ses-01      # a session folder: its .nc, .ethograph/, labels.tsv
  - source: /data/sessions/ses-02      # absolute, or relative to this file's folder
  - source: /data/sessions/ses-03      # the held-out session, named below by its folder name

features:
  name: kinematics                     # → data/kinematics/
  columns:                             # feature → dim → values; the individual dim is never listed
    # orientation fixed
    position:     {space: [x, y], keypoint: [snout, tailBase]}
    velocity:     {space: [x, y], keypoint: [snout, tailBase]}
    speed:        {keypoint: [snout, tailBase]}
    intra:        {pair: [snout-tailBase]}   # list the pairs, named "a-b"
    # freely moving: replace the four above with
    # position_ego: {space: [x, y], keypoint: [snout]}   # tailBase is the origin, always 0
    # speed:        {keypoint: [snout, tailBase]}
    # intra:        {pair: [snout-tailBase]}
  labels:
    branch: 0                          # one model per branch of your mapping.txt

model:
  architecture: c2f_tcn                # DLC2Action's default, and ours

train:
  run_name: quickstart
  epochs: 100
  split:
    train_fraction: 0.8
    val_fraction: 0.2
    test_fraction: 0.0
    holdout_sessions: [ses-03]         # every trial of this session is `test`
```

Three things worth knowing about it:

- **A session has no role.** `holdout_sessions` is what makes ses-03 the test
  session; the fractions then split ses-01 + ses-02's trials 80/20 into train
  and val. Without it, all three sessions' trials would be pooled and cut by
  the ratios.
- **`mapping.txt` defaults to `~/.ethograph/defaults/mapping.txt`** — the one the GUI
  wrote. If yours lives beside the data instead, say so:
  `labels: {mapping: .ethograph/mapping.txt, branch: 0}`.
- **Only `manual` and `curated` labels are training targets**, and point
  events are skipped — they belong to the {doc}`lightgbm model <../onset_model>`.

## 3. Train

```python
import ethograph as eto

project = eto.segment.Project("project.yaml")
result = project.train()  # materialises the dataset first, if needed

print(result.run_dir)  # runs/quickstart_20260827-1412/
print(result.test_metrics["postprocessed"]["f1@50"])  # ses-03, never trained on
```

Progress prints as it goes, and the run directory keeps the config it ran, the
split it drew, the weights, `metrics.tsv` and `test_metrics.yaml`.

```{note}
`c2f_tcn` needs trials of at least 384 frames — it pools the time axis in
half six times. If yours are shorter, `model.architecture: mstcn` {cite:p}`kozlova2025dlc2action,li2020mstcnpp` works at any
length and is the usual baseline; nothing else in the config changes.
```

## 4. Look at the mistakes in the GUI

The number tells you how well it did; the predictions tell you *where* it went
wrong. Write them out for the held-out session:

```python
paths = project.inference(sessions=["ses-03"])
print(paths[0])
# labels/predictions_quickstart_20260827-1412_20260827_151203/ses-03_predictions.tsv
```

That is a `labels/` folder in the `ses-03` session folder — one per call, so a
re-run never overwrites an earlier one — and the TSV in it is the GUI's own
labels format. Open ses-03 in the GUI and load it
with **File ▸ Import labels…**: every row arrives as `automated`, drawn dotted
next to your curated labels, and confirming one makes it `curated`. See
{doc}`../curation/index`.

## Where to go from here

- **Try another architecture** — a one-line change, and `project.compare()`
  puts the runs side by side:

  ```python
  for architecture in ("c2f_tcn", "mstcn", "mlp"):
      eto.segment.Project(
          "project.yaml",
          f"model.architecture={architecture}",
          f"train.run_name={architecture}",
      ).train()

  print(eto.segment.Project("project.yaml").compare())
  ```

  `mlp` sees one frame at a time — it is the floor that tells you how much
  temporal context is actually buying you.
- **Better features**: distances between individuals, headings and changepoint
  proximity, all built with the session — see
  {doc}`guide` and {mod}`ethograph.features.geometry`.
- **The real workflow**: `project.search()` to find hyperparameters on a
  validation split, then `project.cross_validate()` to hold out each session
  in turn — so every session ends up with predictions from a model that never
  saw it, not just ses-03. That is {doc}`guide`.
- **Every key**, with its default: {doc}`config`.
