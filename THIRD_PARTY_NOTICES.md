# Third-party notices

EthoGraph is licensed under the GNU General Public License v3.0 or later
(`LICENSE`). It ships copies of, and code adapted from, the projects below.
Each vendored tree carries its own `LICENSE` and a `NOTICE.md` naming the
upstream commit and every edit made to the copy; this file is the index.

Vendored trees are excluded from the linter and type checker and are never
edited beyond what their `NOTICE.md` lists (`pyproject.toml`,
`[tool.ruff]` / `[tool.mypy]` `exclude`).

## Vendored copies

| Path | Upstream | Commit | Licence | What |
|---|---|---|---|---|
| `ethograph/segment/dlc2action/` | [DLC2Action](https://github.com/amathislab/DLC2Action), A. Mathis Group | `1d7d690` (2026-06-01) | AGPL-3.0-or-later | Action-segmentation models (MS-TCN++, ASFormer, C2F-TCN, C2F-Transformer, EDTCN, MLP, MotionBERT), the MS-TCN loss, their default configs |
| `ethograph/segment/specscalpel/` | [SpecScalpel](https://github.com/HaoyuJi/SpecScalpel), Haoyu Ji | `01adb1e` (2025-07-10) | MIT | Skeleton-based action-segmentation model |
| `ethograph/segment/lady/` | [LaDy](https://github.com/HaoyuJi/LaDy), Haoyu Ji | `bdba27d` (2025-12-02) | MIT | Skeleton-based action-segmentation model with a Lagrangian-dynamics stream |
| `ethograph/spot/e2espot/` | [E2E-Spot](https://github.com/jhong93/spot), James Hong et al. | `edec420` (2023-02-15) | BSD-3-Clause | The pixel event-spotting trainer, evaluator and model (RegNet/ResNet + GSM/TSM + GRU head) |
| `ethograph/_vendor/vocalseg/` | [vocalization-segmentation](https://github.com/timsainb/vocalization-segmentation), Tim Sainburg | `8bc85ee` (2021-04-12) | MIT | Dynamic-threshold and continuity segmentation of a spectrogram (audio changepoint candidates) |
| `ethograph/utils/arraytools.py` | [thunderhopper](https://github.com/bendalab/thunderhopper), Jona Hartling and Jan Benda, Benda Lab | `5cc0c35` (2025-07-21) | AGPL-3.0 | Five array slicing / edge-extension helpers from its `arraytools.py` and two sequence helpers from its `misctools.py`, in one file; the header names each |
| `ethograph/segment/feral_defaults/` | [FERAL](https://github.com/Skovorp/feral), Peter Skovorodnikov and Jacopo Razzauti | `f42aa4d` (2026-09-08, release 1.0.1) | MIT | Its `default_config.yaml` verbatim and its three presets as YAML — the defaults of the FERAL config ethograph writes; no code |

DLC2Action's own `NOTICE.yml` records that several of its model files
incorporate MIT-licensed code (ASFormer, C2F-TCN, ASRF, MS-TCN++); those
attributions are listed in `ethograph/segment/dlc2action/NOTICE.md` and kept in
the per-file headers. E2E-Spot's `model/impl/` likewise carries GSM
(BSD-2-Clause, FBK), TSM (MIT, MIT HAN Lab) and ASFormer (MIT); see
`ethograph/spot/e2espot/NOTICE.md`.

**On the AGPL copies.** GPL-3.0 section 13 permits combining a GPL-3.0 work
with an AGPL-3.0 work. The files under `ethograph/segment/dlc2action/` and
`ethograph/utils/arraytools.py` remain under the AGPL-3.0, and the AGPL's
network-interaction clause (its section 13) applies to them; the rest of
EthoGraph stays GPL-3.0-or-later. Thunderhopper's two functions in
`arraytools.py` note in their own docstrings that they were in turn adapted
from `scipy.signal._arraytools` (BSD-3-Clause).

## Adapted code

Code written in EthoGraph's own modules from a published implementation.
The adapted function says so in its docstring.

| Where | Upstream | Licence | What |
|---|---|---|---|
| `ethograph/video_features/s3d.py` | [S3D](https://github.com/kylemin/S3D), Kyle Min | MIT | The S3D network definition. The Kinetics-400 checkpoint it loads (`video_features/checkpoint/`, not packaged) is that repository's released weight file |
| `ethograph/spot/msagsm.py` | [GSM](https://github.com/swathikirans/GSM), Swathikiran Sudhakaran, FBK (vendored as `ethograph/spot/e2espot/model/impl/gsm.py`) | BSD-2-Clause | Multi-scale attention gated shift, written from the MSAGSM paper (arXiv 2507.07381); follows GSM's structure without copying its code, and takes no code from the MSAGSM reference repository, which carries no licence |
| `ethograph/spot/stream.py` | [E2E-Spot](https://github.com/jhong93/spot), James Hong et al. | BSD-3-Clause | Streaming inference that mirrors its `test_e2e.py`: window starts, padding, evaluation transform, score accumulation |
| `ethograph/gui/plots_ephystrace.py` | [phy](https://github.com/cortex-lab/phy), Cortex Lab | BSD-3-Clause | Right-drag box scaling and the trace-view plotting algorithm |

## Documentation figures

| Where | Upstream | Licence | What |
|---|---|---|---|
| `docs/source/_static/neuroconv/` | [neuroconv](https://github.com/catalystneuro/neuroconv), CatalystNeuro | BSD-3-Clause | `video_setup_free_running.png` and `video_setup_triggered.png`, from the "align external video" how-to added in PR #2037 |
