---
html_theme.sidebar_secondary.remove: true
---

(target-models)=
# Models

Once you have labelled some trials by hand, a model can label the rest — and
you then curate what it predicted. Which model depends on **whether you have a
GPU**, on **the shape of the label**, and on **what the model sees**.

```{mermaid}
flowchart TD
    gpu{Do you have<br/>a GPU?}
    gpu -->|No| lgbm[<b>LightGBM</b><br/>point events, CPU<br/><i>Model ▸ LightGBM</i>]
    gpu -->|Yes| shape{What kind<br/>of label?}

    shape -->|"Point event<br/>(one moment)"| input{What does the<br/>model see?}
    shape -->|"State event<br/>(onset + offset)"| big{A large GPU<br/>or a cluster?}

    input -->|Video only| spot[<b>E2E-Spot</b><br/><code>eto.spot</code>]
    input -->|"Video + pose, sensors,<br/>custom features…"| spotfeat[<b>E2E-Spot</b><br/>+ features]

    big -->|Yes| sees{What does<br/>FERAL see?}
    sees -->|Video only| feral[<b>FERAL alone</b><br/><code>export_feral</code>]
    sees -->|"Video + pose, sensors,<br/>custom features…"| feralseg[<b>FERAL embeddings</b><br/>as a video feature]
    big -->|No| segment[<b>Action segmentation</b><br/><code>eto.segment</code>]
    feralseg --> segment

    click segment "segment/index.html"
    click lgbm "onset_model.html"
    click spot "spot/index.html"
    click spotfeat "spot/multimodal.html"
    click feral "segment/feral.html"
    click feralseg "segment/feral.html"
```

Every box is a link. Whichever model you pick, its predictions come back to the
GUI: {doc}`curate them <curation>`, then {doc}`read and threshold their
confidence <confidence>`.

| Model | Label | Description |
|---|---|---|
| {doc}`LightGBM <onset_model>` | point event | Simple to use, in the GUI, on a CPU. Fed a few hand-crafted time-series features. |
| {doc}`E2E-Spot <spot/index>` (`eto.spot`) | point event | Simple to use: it learns end to end from the video, on a GPU. For when you want a few alignment points per trial — the frame a beak touches the tool, a lever press, the first step of a reach. |
| {doc}`Action segmentation <segment/index>` (`eto.segment`) | state event | For action intervals — a bout of grooming, sniffing or tool use movement, each with an onset and an offset — and have pose data. Good in a lab environment, where a calibrated camera setup gives robust features. |
| {doc}`FERAL alone <segment/feral>` | state event | Fine-tunes the V-JEPA 2 video foundation model. Can learn individual and collective behaviours directly from video, even in naturalistic setups. Requires a heavy GPU. |
| {doc}`FERAL embeddings <segment/feral>` | state event | The best of two above: FERAL's embedding of the video, read by `eto.segment` beside your pose features. |

## Where the models come from

- **LightGBM** {cite:p}`ke2017lightgbm` — runs on a CPU and lives entirely in the GUI: no scripting, and no
  separate Python environment. The GUI install (`uv tool install "ethograph[gui,audio]"`, see
  {doc}`../getting_started/installation`) is all it needs.
- **Action segmentation** — `eto.segment` vendors the models and loss of DLC2Action {cite:p}`kozlova2025dlc2action`, adapted there to pose/kinematic input:
  - DLC2Action's own variants: `mstcn` (MS-TCN3, which feeds the last two layers of the first stage into the second; from MS-TCN++ {cite:p}`li2020mstcnpp`) and `c2f_transformer` (C2F-TCN {cite:p}`singhania2021c2ftcn` with attention in place of convolution).
  - Original architectures, as adapted in DLC2Action: `asformer` {cite:p}`yi2021asformer`, `c2f_tcn` {cite:p}`singhania2021c2ftcn`, `edtcn` {cite:p}`lea2017edtcn`, `motionbert` {cite:p}`zhu2023motionbert`, and `mlp`, a per-frame baseline.
  - Added in Ethograph: `rnn` (a bidirectional GRU/LSTM baseline), `specscalpel` {cite:p}`ji2026specscalpel` and `lady` {cite:p}`ji2026lady`.
- **Precise event spotting** — E2E-Spot {cite:p}`hong2022e2espot`.
- **FERAL** {cite:p}`skovorodnikov2025feral` — a video foundation model (V-JEPA 2) fine-tuned on the pixels alone, in its own
  environment on a large GPU. Either on its own, with the GUI's labels exported to its `labels.json`, or as a video feature
  whose per-frame embedding `eto.segment` reads next to changepoint and pose columns ({doc}`segment/feral`).

```{toctree}
:maxdepth: 1
:hidden:
Trial windows <trial_windows>
LightGBM (point events, CPU-only) <onset_model>
Action segmentation (state events) <segment/index>
PES (point events from pixels) <spot/index>
confidence
curation
workflows
```
