# Vendored DLC2Action

This directory is a partial copy of
[DLC2Action](https://github.com/amathislab/DLC2Action) at commit `1d7d690`
(`main`, 2026-06-01), kept in **upstream's own layout** (`model/`, `loss/`,
`config/`, `version.py`) so that refreshing it is "copy these directories at
commit X" and adding a model or a loss is "copy two more files".

Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
Licensed under the GNU Affero General Public License v3.0 or later — see
`LICENSE.APGPL` in this directory. AGPL-3.0-or-later is compatible with
ethograph's GPL-3.0-or-later; the combined work is distributed under the AGPL
for these files.

## What is vendored

| Path | Upstream origin |
|---|---|
| `model/*.py` | `dlc2action/model/` |
| `loss/*.py` | `dlc2action/loss/` |
| `config/model/*.yaml`, `config/losses.yaml`, `config/training.yaml` | `dlc2action/config/` |
| `version.py`, `utils.py`, `colors.txt` | `dlc2action/` |
| `LICENSE.APGPL` | `dlc2action/LICENSE.APGPL` |

Two values in `config/model/` are not numbers: upstream's `dataset_*`
sentinels ("fill this in from the dataset") and OmegaConf's `???` ("required,
no default"). Both reach `../models/vendored.py` untouched — resolving one
against *this* project's dataset is the builder's job, and refusing to build
without a required one is too.

`config/` holds the three groups this project reads, unmodified:

| File | Read by | For |
|---|---|---|
| `config/model/*.yaml` | `../models/vendored.py` | every architecture's hyperparameters |
| `config/losses.yaml` | `../losses.py` | the `ms_tcn` block — the whole loss |
| `config/training.yaml` | `../config.py` | only `num_epochs`, `lr`, `weight_decay`, `val_frac`, `test_frac` |

Upstream's other config groups (`general.yaml`, `metrics.yaml`, `ssl.yaml`,
`annotation/`, `augmentations/`, `data/`, `features/`) are not copied: they
configure the toolbox layer below, which is not vendored either.

`training.yaml` is different in kind from the other two. The model and loss
YAMLs feed a constructor we call verbatim; `training.yaml` configures
DLC2Action's *own* training loop, and we run ours — so only the settings that
carry over unchanged are read from it. `batch_size`, `grad_clip` and
`eval_every` are deliberately ours (upstream's `batch_size: 64` counts
fixed-length 128-frame windows; a sample here is a whole trial), and each says
so where it is defined.


## Third-party origins noted upstream

The upstream `NOTICE.yml` records that several of these files incorporate code
adapted from MIT-licensed originals (combined work licensed under AGPLv3):

- `model/asformer.py` — ASFormer by ChinaYi, © 2021 ChinaYi, https://github.com/ChinaYi/ASFormer
- `model/c2f_tcn.py` — C2F-TCN by dipika-singhania, © 2021 dipika-singhania, https://github.com/dipika-singhania/C2F-TCN
- `model/edtcn.py` — ASRF by yiskw713, © 2020 yiskw713, https://github.com/yiskw713/asrf/blob/main/libs/models/tcn.py
- `model/ms_tcn.py`, `loss/ms_tcn.py` — MS-TCN++ by yabufarha, © 2019 June01, https://github.com/sj-li/MS-TCN2

The per-file header comment blocks carrying these attributions are preserved.

## Edits made to the copies

Every `.py` keeps its upstream header comment block, followed by two added
lines: `# Vendored from DLC2Action — see NOTICE.md` and `# ruff: noqa` (the
linter and mypy skip this directory — see `pyproject.toml`).

- **All files**: `from dlc2action.X import ...` rewritten to
  `from ethograph.segment.dlc2action.X import ...`. This is the only change to
  every file but the two below.
- **`model/asformer.py`**: upstream defines
  `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` at
  module level and uses it in 18 places, which pins the model to CUDA and
  breaks CPU and MPS. The module-level line is removed;
  `AttLayer.construct_window_mask` returns a CPU tensor that
  `_sliding_window_self_att` moves with `self.window_mask.to(q.device)`; every
  other `.to(device)` became `.to(q.device)` (all operands share the query's
  device). No device is hardcoded anywhere. This is required by ethograph's
  "never hardcode a device" rule and is covered by
  `tests/test_unit/test_segment_architectures.py`.
- **`config/**`**: unmodified, byte-for-byte upstream, comment headers included.
- **Deleted**: upstream's `options.py` (its name→class registry, which imports
  the data stores, SSL modules, metrics and transformers this project does not
  vendor — our equivalent is the architecture registry in
  `../models/__init__.py`) and its `__init__.py` (it imports
  `dlc2action.project` and `dlc2action.preprocessing`; replaced with a minimal
  one). Nothing under `model/` or `loss/` depends on either.

`model/motionbert.py` and `model/motionbert_modules.py` are the only files
here needing `einops`, which the `model` extra declares.

## How these are used

`ethograph/segment/models/vendored.py` adapts each architecture to this
project's registry contract and reads its hyperparameters from
`config/model/{stem}.yaml`; `ethograph/segment/losses.py` does the same for
`MS_TCN_Loss` from `config/losses.yaml`. **No default is written on our side** —
upstream's `"dataset_features"` / `"dataset_inverse_weights"` sentinels are
resolved from the run's own data.

The adapters add only what a config file cannot carry: the padding mask
(upstream cuts fixed-length windows and has no mask), the stage-order and
temporal-resolution declarations, `exclusive`, which upstream takes from the
task's single- vs multi-label problem type, and the window fold MotionBERT's and the Transformer's
fixed-length position encodings force once a sample is a whole trial.

## Adding another upstream model or loss

1. Copy `dlc2action/model/{name}.py` (or `loss/{name}.py`) here and rewrite its
   `from dlc2action.` imports; add the two header lines.
2. Its `config/` YAML is already here — the whole tree is vendored.
3. Add a builder in `../models/vendored.py` calling
   `upstream_defaults("{name}", n_features)`. It must not restate any value
   from the YAML.
4. Add the name to `VENDORED` and `CONFIG_STEMS` in
   `tests/test_unit/test_segment_architectures.py`; the contract and no-drift
   tests then cover it.
5. Record it in the table above, and any edit under "Edits made to the copies".
