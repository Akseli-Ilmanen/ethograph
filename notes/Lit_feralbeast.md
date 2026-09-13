# EthoGraph — video-model review and decisions

*Date: 2026-08-25. Covers E2E-Spot and successors, FERAL, BEAST, and how they fit the EthoGraph library design.*

---

## 1. Decisions

*Revised 2026-09-06 after the grill in `Lit_videofeat.md`: FERAL is pip-installable and MIT, so the
"importer only" verdict is void; ViT-MAE is dropped as a named extractor; BEAST is out entirely.*

1. **FERAL: both an extractor and a model — but nothing lands in the repo before talking to Peter
   and Jacopo** (agenda: `Discuss_feral-interop.md`). Frozen FERAL is the expensive video-feature
   tier; fine-tuned FERAL is the video-side segmenter ("one default per slot": kinematics →
   ASFormer, video → FERAL). Its exact dependency pins (`transformers==5.5.3`, `timm==1.0.26`,
   `pandas==2.3.3`) are the blocker for an `ethograph[feral]` extra; until they loosen, FERAL runs
   in its own env by subprocess, exchanging files — the E2E-Spot pattern.
2. **One default per slot.** Every slot (extractor, head, codec) has exactly one default; alternatives are opt-in by name. Defaults:
   - State events: `kinematics → ASFormer → StateCodec`
   - Point events: `regnety-200mf-gsm → GRU (+ displacement head) → PointCodec (Soft-NMS)`
   - Video feature: `s3d` (measured on our data, weights in the package, faster than DINOv2-B at
     518 px on ten clips); `timm` second, by name, with `vit_base_patch14_reg4_dinov2.lvd142m` and
     timm's own pooling (CLS, the classification token, for DINOv2); FERAL by name later.
3. **GUI as a feature-engineering tool.** Video features are first-class `(time, feature)` streams in the TrialTree, next to kinematics. Cohen's-d subset selection, before/after-fine-tuning comparison and label-coloured embedding views all operate on them. ~~First frozen extractor: ImageNet ViT-MAE CLS~~ — with `timm` as the loader, ViT-MAE is a model string, not a slot; DINOv2 is timm's default backbone.
4. **No SSL pretraining, no semi-supervised heads.** Frozen SSL backbones and supervised fine-tuning are in; running MAE/contrastive objectives on user data is out.
5. **Vendor nothing new; extractors are pip-installed** (ADR 0009). E2E-Spot stays vendored; everything else is a pip extra plus an adapter, or an importer. The S3D checkpoint in the package is the grandfathered exception.
6. **BEAST: out entirely** — no pretraining, no checkpoint import, no lightning-action head. Its
   findings are cited below as literature only. The one it left behind — Δ-features help every
   stream — is already the lightgbm model's `derivatives` spelling and needs nothing new.

---

## 2. Architecture: three protocols, one seam

The TAS literature's shared contract — per-frame features `(T, D)` plus per-frame labels `(T,)` — is the seam. Everything plugs into one of three protocols.

```python
class FeatureExtractor(Protocol):
    def fit(self, videos: Sequence[Path]) -> Self: ...        # no-op for all shipped extractors
    def transform(self, video: Path) -> xr.DataArray: ...     # dims (time, feature)

class TemporalHead(Protocol):
    def fit(self, features: xr.DataArray, targets: xr.DataArray) -> Self: ...
    def predict_proba(self, features: xr.DataArray) -> xr.DataArray: ...   # (time, class)

class TargetCodec(Protocol):
    def encode(self, events: EventTable, n_frames: int) -> xr.DataArray: ...
    def decode(self, proba: xr.DataArray) -> EventTable: ...
```

- **State vs point events live only in `TargetCodec`.** `StateCodec`: argmax + min-duration smoothing. `PointCodec`: background class + displacement target, Soft-NMS on decode. Heads are agnostic.
- **End-to-end models** (E2E-Spot, fine-tuned FERAL) are one object implementing both `FeatureExtractor` and `TemporalHead`; `transform` returns their penultimate features.
- **Config surface:** three names plus a checkpoint path. Model-native config passes through as an opaque `extra: dict`, never mirrored.

```python
@dataclass
class ExtractorConfig:
    name: str                    # "kinematics" | "s3d-kinetics" | "vit-mae-b" | "vjepa2-vitb" | "regnety-200mf-gsm"
    trainable_layers: int = 0    # 0 = frozen + cached features; >0 = fine-tune last N (fused training, [gpu])
```

`trainable_layers == 0` is today's workflow unchanged (extract once, cache, train head in seconds). `> 0` switches to the fused loop the vendored E2E-Spot already implements. Everything downstream — codec, evaluation, GUI review — is identical in both modes.

---

## 3. Inclusion criteria (apply from the paper/repo, no benchmarking)

| # | Rule | Removes |
|---|------|---------|
| 1 | Fits extractor / head / codec without a fourth concept | BEAST pretraining, semi-supervised heads |
| 2 | pip-installable with a Python entry point | anything needing vendoring |
| 3 | Runs on CPU (`base`) or an 8 GB consumer GPU (`[gpu]`) | 24 GB / Ampere-only backends |
| 4 | Serves a user need you can name today | "SOTA by 2 mAP" |
| 5 | Gain over current default exceeds its noise, per the paper's own ablations | T-DEED, AdaSpot, F3ED |
| 6 | One default per slot; alternatives selectable by name only | duplicate backbone names |

### Verdicts

| Option | Verdict | Rule |
|---|---|---|
| E2E-Spot 200MF + displacement head + Soft-NMS | **in**, default point-event backend | 4, 5 |
| MSAGSM as GSM replacement | in **as a flag on the same extractor** (`shift_module: gsm\|msagsm`), not a new name | 5, 6 |
| T-DEED, AdaSpot, F3ED | out; borrow eval code (tolerance-F1, edit score) | 2, 5 |
| Frozen DINOv2 (any timm model) as extractor | **in**, default video feature; ViT-MAE dropped as a name | 1, 2 |
| `trainable_layers` fine-tuning knob | in, opt-in, `[gpu]` | 1, 3 |
| FERAL | in, as extractor and as model — after the interop discussion (`Discuss_feral-interop.md`) | 2, 4 |
| BEAST (pretraining, checkpoints, lightning-action head) | **out entirely** | 1, 3 |
| Semi-supervised anything | out | 1 |

---

## 4. Model notes (what each is and is not)

### E2E-Spot (Hong et al., ECCV 2022)
- RegNet-Y 200MF/800MF with GSM (channel shift between neighbouring frames) → 1-layer bi-GRU → per-frame K+1 softmax. 2.8 M + 1.7 M params, 0.3 ms/frame on A5000, ~23 GFLOPs per 100-frame clip at 224².
- Built for ±1-frame precision. Long-range context via GRU over 100–500-frame clips; FS/FineGym lose 7–15 mAP going from 100 to 8 frames — clip length must cover event dependencies.
- Weaknesses: data-hungry (Tennis 33k events; FineDiving 12k events → 68 mAP), ImageNet init only, resolution matters (112 px: −7.6 mAP) → **crop around the animal**, one event per frame, states are second-class (onset/offset as two classes, end frames noisier).
- Low-data behaviour: most robust of the RGB models in UMEG-Net's 100-clip benchmark; T-DEED collapses there.
- Cheap upgrades from successors: **displacement head** (regress offset to nearest event within r_E, MSE; replaces label dilation, +3–4 mAP), **Soft-NMS** 3-frame window (+3–4 mAP), **GSF in second half of backbone only**, **MSAGSM** (14/15 wins, minimal overhead). Diagnostic: cosine similarity of adjacent-frame tokens to sequence mean (T-DEED Fig. 5) tells you whether the head smears boundaries.
- Successors and speed: T-DEED 200MF ≈ same FLOPs but 3.6× params; AdaSpot +30%; 800MF variants ≈ 4×. Real cost for users is video decoding, not the model — decode straight to tensors, no `spot_frames/` on disk; don't drop below 224 px; single pass at inference.

### FERAL (Skovorodnikov & Razzauti et al., bioRxiv 2025/2026, `pip install feral`, MIT)
- **A segmenter, not a feature extractor**: V-JEPA2 (ViT-L, Diving48 ckpt, last 12/24 layers fine-tuned) → 64 learned query tokens cross-attend over patch tokens → one embedding per frame → BN → linear → per-frame CE with label smoothing, √inv-freq class weights, mixup. 64-frame chunks at 256², 50% overlap, averaged.
- Frozen backbone drops 94.5 → 88.4 mAP on CalMS21: the fine-tuning *is* the value. Features (`FeralModel.clip_projector` output, `(64, 768)` per chunk) are meaningful only after fine-tuning.
- No long-range temporal model (context = 64 frames ≈ 2 s). No segment-level metrics reported (no edit score, no F1@k). V-JEPA tubelets are 2 frames → systematic ±1-frame ambiguity for point events (mitigations: odd `chunk_shift`, frame doubling).
- Public API: `feral.run_training(cfg)`, `feral.run_inference_folder(ckpt, folder)`, `feral.apply_mode(cfg, "lite"|"max"|"rare")`, `FeralModel`, `BACKBONES`. `rare` preset = no mixup, no label smoothing, grad-clip, class-weight cap — matches sparse-event data.
- Hardware: Ampere+ (bf16, flash-attn), 24 GB recommended; `lite` (ViT-B) + `gradient_checkpointing` + `train_bs 2` should fit a 10 GB RTX 3080. CUDA only. Windows via `triton-windows`; prefer WSL; set `training.compile: false` on first run.
- Potential two-stage use: fine-tune FERAL → dump per-frame embeddings → ASFormer for long-range coherence. Unpublished comparison; directly tests whether your head adds anything over a chunk-level transformer.

### BEAST (Wang, Yu et al., ICLR 2026, `paninski-lab/beast`, MIT)
- **A backbone-pretraining recipe**, not a segmenter. ViT-B/16 on single frames, ImageNet-MAE init, then MAE (75% mask) + temporal contrastive loss (positive = t±1, InfoNCE on CLS through a BN projector). Frame selection: motion-energy filter + k-means, ~600 anchor triplets/video, ~100k frames total. 800 epochs, 8 A40s, 12–25 h.
- Per-frame features at full frame rate (no tubelets); CLS = 768-d storable; patch tokens = 196×768, streamed. Inference 17.6 GFLOPs / 5.4 ms per frame on A100 (~75× E2E-Spot 200MF per frame).
- Downstream lives elsewhere: `lightning-pose` (pose), `lightning-action` (segmentation: linear or 2-block dilated TCN, features + Δ-features, sliding window), IBL RRR / Facemap TCN (neural encoding).
- Key findings: **frozen ImageNet ViT-MAE already beats keypoints, PCA, CEBRA, DINOv2, CLIP** for neural encoding with no pretraining. Domain pretraining adds a moderate increment (IBL seg 0.84→0.87, CalMS21 0.74→0.81); contrastive-only hurts. CLS wins for neural encoding, attention-pooled patches win for segmentation (0.81 vs 0.63 on CalMS21). Δ-features help across every feature type. Pose: ViT-MAE backbone beats ResNet-50/DLC/Lightning-Pose with 100 labels; domain pretraining helps further.
- Not a foundation model — checkpoints are dataset-specific and won't transfer to a crow.

### Ethology context
- No ethology/neuroscience paper uses E2E-Spot or successors; that lineage is entirely sports (SoccerNet). The animal side jumped straight to video foundation models (FERAL, TRACE, BEAST, Autobehaver, PlayClass), all state-oriented, none evaluated at frame tolerance. The frame-precise point-event gap for animals is open.
- Transferable non-sport insights from the E2E-Spot citation graph: few-shot (UMEG-Net), class-imbalance losses (Santra et al. SoftIC), label dilation vs displacement (T-DEED), dense events under handheld/occluded cameras (TTA dataset), sequence metrics (F3Set edit score), single-frame-label ambiguity and loss/metric mismatch (BME runner-up report).
- SSL in practice: (1) frozen SSL backbone — indistinguishable from S3D-Kinetics from the user's side, no SSL code runs; (2) fine-tune with your labels — still supervised, only the init differs; (3) run the SSL objective on your unlabeled data — the only level where contrastive/masked losses enter your code. EthoGraph does 1 and 2, not 3.

---


## 6. References

- Hong et al. 2022, *Spotting Temporally Precise, Fine-Grained Events in Video*, ECCV. https://arxiv.org/abs/2207.10213
- Xarles et al. 2024, *T-DEED*, CVPRW. https://github.com/arturxe2/T-DEED
- Xu et al. 2025, *MSAGSM / Multi-Focus Temporal Shifting*. https://arxiv.org/abs/2507.07381
- Liu et al. 2025, *F3Set / F3ED*, ICLR. https://github.com/F3Set/F3Set
- Xarles et al. 2026, *AdaSpot*. https://arxiv.org/abs/2602.22073
- *Few-Shot PES via Unified Multi-Entity Graph (UMEG-Net)*, AAAI 2026. https://arxiv.org/abs/2511.14186
- Wang, Guo, Liu 2023, *BME runner-up*, SoccerNet BAS. https://arxiv.org/abs/2306.05772
- Skovorodnikov, Razzauti et al. 2025/26, *FERAL*. https://github.com/Skovorp/feral · https://www.biorxiv.org/content/10.1101/2025.11.16.688666
- Wang, Yu et al. 2026, *BEAST*, ICLR. https://github.com/paninski-lab/beast · https://arxiv.org/abs/2507.09513
- Blau et al. 2024, *A study of animal action segmentation algorithms across supervised, unsupervised, and semi-supervised paradigms*. https://github.com/paninski-lab/lightning-action
- RegNet: https://arxiv.org/abs/2003.13678 · GSM: https://github.com/swathikirans/GSM · GSF: https://github.com/swathikirans/GSF
