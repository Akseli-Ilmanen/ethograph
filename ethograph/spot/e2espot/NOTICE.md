# Vendored E2E-Spot

A partial copy of **[E2E-Spot](https://github.com/jhong93/spot)** (ECCV 2022,
*Spotting Temporally Precise, Fine-Grained Events in Video*) at commit
`edec420` (`main`, 2023-02-15). BSD-3-Clause, © 2022 James Hong, Haotian
Zhang, Matthew Fisher, Michael Gharbi, Kayvon Fatahalian — see `LICENSE`.

Kept in upstream's own layout. `../vendored.py` runs the two scripts as
subprocesses (`python -m ethograph.spot.e2espot.train_e2e` / `.test_e2e`);
`../stream.py` imports `E2EModel` for streaming inference.

## What is vendored

| Path | Upstream origin |
|---|---|
| `train_e2e.py`, `test_e2e.py` | same |
| `dataset/__init__.py`, `dataset/frame.py`, `dataset/transform.py` | same |
| `model/__init__.py`, `model/common.py`, `model/shift.py`, `model/modules.py` | same |
| `model/impl/__init__.py`, `model/impl/asformer.py`, `model/impl/gsm.py`, `model/impl/tsm.py` | same |
| `util/__init__.py`, `util/dataset.py`, `util/eval.py`, `util/io.py`, `util/score.py` | same |
| `LICENSE` | same |

## Third-party origins noted upstream

Each file keeps its original licence header:

- `model/impl/gsm.py` — Gate Shift Module by Swathikiran Sudhakaran, © 2019 FBK, BSD-2-Clause, https://github.com/swathikirans/GSM
- `model/impl/tsm.py` — Temporal Shift Module, © 2021 MIT HAN Lab, MIT, https://github.com/mit-han-lab/temporal-shift-module
- `model/impl/asformer.py` — ASFormer, © 2021 ChinaYi, MIT, https://github.com/ChinaYi/ASFormer

## Edits made to the copies

Every `.py` carries the two vendor header lines (after the shebang where there
is one); the linter and mypy skip this directory (`pyproject.toml`).

- **Imports**: upstream's bare top-level imports (`from util.io …`,
  `from model.shift …`, `from dataset.frame …`, `from train_e2e …`) are
  repointed to `ethograph.spot.e2espot.…`, so the package imports from any
  directory and cannot collide with another top-level `util` or `model`.
- **`dataset/frame.py`**: transforms are plain `nn.Sequential` rather than
  `torch.jit.script` (fails on Windows); `ActionSpotDataset` and
  `ActionSpotVideoDataset` take `fuse_dir` / `zero_fuse` (a per-video pose
  block, `{video}.npz`, fed beside the CNN features) and `ActionSpotDataset`
  takes `teacher_dir` (per-video teacher embeddings for distillation), read by
  the new `load_side_array` / `load_side_clip`; mixup blends the pose block
  like the frames; `get_labels` floors `num_frames / stride` (rounding up left
  the truth one strided frame longer than the prediction and crashed every
  stride > 1 run); `np.int` → `int`; a `clip_len` property.
- **`model/common.py`**: `_get_params` skips frozen parameters; `load` takes
  `strict`.
- **`model/shift.py`**: `GatedShift` / `make_temporal_shift` take a
  `shift_module` in place of `_GSM` (for `../msagsm.py`); timm's
  `ConvBnAct` → `timm.layers.ConvNormAct` (timm ≥ 1.0).
- **`train_e2e.py`**: the dataset argument is a name under `data/` or any
  directory; `--stride`, `--epoch_num_frames`; the `*_msagsm` architectures
  with `--shift_dilations` / `--attention_groups`; `--stage 1|2|3`
  (labels / distil a teacher embedding / labels with the CNN frozen) with
  `--teacher_dir`, `--distil_dim`, `--init_from`; `--fuse_dir`, `--fuse_dim`,
  `--fuse_dropout` (the pose block concatenated before the temporal head, with
  modality dropout); the inference batch sized by frames
  (`INFERENCE_BATCH_FRAMES`); the config is printed rather than written to
  `/dev/stdout`; a picklable `worker_init_fn` (Windows spawns workers).
- **`test_e2e.py`**: builds the model with the stored `shift_dilations`,
  `attention_groups`, `distil_dim`, `fuse_dim`, reads `stride` and `fuse_dir`
  from the run config; `--zero_fuse`.
- **`util/dataset.py`**: `crow_pellet` added to `DATASETS`.

## Not vendored

The benchmark splits (`data/`), the Flask viewer (`web/`, `view.py`), the
feature baseline (`baseline.py`, `dataset/feature.py`, `model/feature.py`,
`model/impl/gtad.py`, `model/impl/calf.py`), frame extraction and dataset
parsing scripts, the ensemble / SoccerNet evaluators, `external/`,
`util/video.py`, and `requirements.txt` (its `torch==1.11` pins are not ours).
