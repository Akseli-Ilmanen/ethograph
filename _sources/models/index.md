(target-models)=
# Models

Once you have labelled some trials by hand, a model can label the rest — and
you then curate what it predicted. Which model depends on **whether you have a
GPU**, on **the shape of the label**, and on **what the model sees**.

```{mermaid}
flowchart TD
    start([Hand-labelled trials]) --> gpu{Do you have a GPU?}

    gpu -->|No| lgbm[LightGBM model<br/><i>point events only</i><br/><i>CPU, GUI native</i><br/><i>Model ▸ LightGBM: Train… / Predict…</i>]
    gpu -->|Yes| shape{What kind of label?}

    shape -->|"State event<br/>(an interval: onset + offset)"| segment[Action segmentation<br/><code>eto.segment</code><br/><i>DLC2Action models + added architectures</i>]
    shape -->|"Point event<br/>(one moment per trial)"| input{What does the model see?}

    input -->|Video only| spot[Event spotting: E2E-Spot<br/><code>eto.spot</code>]
    input -->|Video + pose| spotfeat[E2E-Spot + features]

    segment --> curate
    lgbm --> curate
    spot --> curate
    spotfeat --> curate
    curate([Curate the predictions in the GUI]) --> conf[Read and threshold confidence]

    click segment "segment/index.html"
    click lgbm "onset_model.html"
    click spot "spot/index.html"
    click spotfeat "spot/multimodal.html"
    click curate "curation.html"
    click conf "confidence.html"
```

## Where the models come from

- **LightGBM** {cite:p}`ke2017lightgbm` — runs on a CPU and lives entirely in the GUI: no scripting, and no
  separate Python environment. The GUI install (`uv tool install "ethograph[gui,audio]"`, see
  {doc}`../getting_started/installation`) is all it needs.
- **Action segmentation** — `eto.segment` vendors the models and loss of DLC2Action {cite:p}`kozlova2025dlc2action`, adapted there to pose/kinematic input:
  - DLC2Action's own variants: `mstcn` (MS-TCN3, which feeds the last two layers of the first stage into the second; from MS-TCN++ {cite:p}`li2020mstcnpp`) and `c2f_transformer` (C2F-TCN {cite:p}`singhania2021c2ftcn` with attention in place of convolution).
  - Original architectures, as adapted in DLC2Action: `asformer` {cite:p}`yi2021asformer`, `c2f_tcn` {cite:p}`singhania2021c2ftcn`, `edtcn` {cite:p}`lea2017edtcn`, `motionbert` {cite:p}`zhu2023motionbert`, and `mlp`, a per-frame baseline.
  - Added in EthoGraph: `rnn` (a bidirectional GRU/LSTM baseline), `specscalpel` {cite:p}`ji2026specscalpel` and `lady` {cite:p}`ji2026lady`.
- **Precise event spotting** — E2E-Spot {cite:p}`hong2022e2espot`.

```{toctree}
:maxdepth: 1
:hidden:

LightGBM (point events, CPU-only) <onset_model>
Action segmentation (state events) <segment/index>
PES (point events from pixels) <spot/index>
confidence
curation
```
