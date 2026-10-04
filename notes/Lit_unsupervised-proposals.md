# Unsupervised segmentation as label proposals — takeaways (2026-10-01)

Summary of a discussion on whether Ethograph should load or compute
unsupervised labels as proposals for the supervised loop. Tools looked at:
mosaic, neuroconv's VAME and keypoint-MoSeq interfaces, SMQ, Dynaclade (Bach
lab talk, Bernstein 2026), A-SOiD, OpenLabCluster, CASTLE, B-SOiD, and the
Blau et al. 2024 comparison. Nothing here is implemented and nothing was
measured on our data. Claims marked *(unverified)* were read through a
summarising fetch or seen only as a title.

## The idea

1. An unsupervised method proposes action-level segments for a session.
2. The annotator reviews them **by cluster**, names the clusters that are real
   behaviours, merges the rest, and corrects boundaries.
3. The corrected labels train a supervised segmentation model, whose
   predictions go back through the existing curation loop.

## Position (recommended, not yet decided)

1. **Import, never compute.** Ethograph reads the output of an unsupervised
   tool; it does not run, vendor or maintain one. Consistent with
   `Lit_feralbeast.md` decision 4 (no SSL pretraining, no semi-supervised
   heads) and ADR 0009 (extractors are not vendored).
2. **Two importers cover the field.**
   - *The ndx-ethogram bout table in an NWB file.* neuroconv's
     `MoseqKeyPointsInterface` (reads `results.h5`) and its VAME interface
     (neuroconv 0.10.0, 2026-08-18) both write `EthogramBouts` plus an
     `Ethogram` catalogue; the BORIS interface is in the same family. One
     reader, three tools, and neuroconv maintains the parsing.
     `PynappleLabelConverter` already turns every non-trial `IntervalSet` into
     labels, so the gap may be small. **Open: does pynapple expose
     `EthogramBouts` as an `IntervalSet`?** Not checked.
   - *A per-frame integer array plus a rate*, run-length-encoded into
     intervals. Covers SMQ (per-frame `.txt`), mosaic, B-SOiD and anything a
     user clustered themselves.
3. **Imported segments are a prediction set**: `labeling_method=automated`,
   `prediction_source=<tool>`, under `labels/predictions_{model}_{timestamp}/`.
   Curation works unchanged. `confidence` is empty (TODO.md item 2 already
   notes this).
4. **Clusters are not classes.** "Syllable 12" must not land in `mapping.txt`.
   Clusters need their own branch and a many-to-one rename step into real
   classes. This rename/merge step is the only genuinely new GUI work.
5. **Review by cluster is where the time is saved**: the grid view over every
   bout of one cluster, then name and bulk-assign.
6. **A second use comes free**: an imported cluster branch can feed the
   supervised model as input columns through `features/label_inputs.py`
   (ADR 0011), since an input branch is never a target branch.

## Syllables vs action-level segments

| | Syllables | Action-level segments |
|---|---|---|
| Duration | ~0.1–0.5 s | seconds to minutes |
| Defined by | movement dynamics | behavioural meaning |
| Number of types | tens to ~100 | a few to a couple of dozen |
| Methods | keypoint-MoSeq, VAME, B-SOiD, MotionMapper | SMQ; supervised MS-TCN-style models |

An action is roughly a sequence of syllables. Ethograph's labels are
action-level, so syllable output is not a label proposal as it stands: one
real label is covered by many fragments of several types and needs merging.
The keypoint-MoSeq paper reports median durations of ~400 ms for its
syllables against 33–100 ms for VAME, B-SOiD and MotionMapper. For **point
events** the opposite may hold: an onset is a change in dynamics, so syllable
boundaries could be the better proposal. The line is soft: MoSeq's stickiness
lengthens syllables, SMQ's cluster count shortens actions.

## Two answers to over-fragmentation

| | SMQ patches | Sticky HMM + Viterbi (mosaic's AR-HMM, MoSeq family) |
|---|---|---|
| Constraint | hard minimum length | soft cost per switch |
| Boundary position | fixed grid | any frame |
| Segment length | multiples of the patch | variable |
| Knob | patch length (frames) | sticky weight |

- **SMQ** (read from the code): the latent sequence is cut into fixed
  windows, each window takes its nearest codebook entry by argmax, and the id
  is repeated per frame. No Viterbi in the repo. Patch sizes are hardcoded per
  dataset in `main.py` (60 / 50 / 30 frames), so for us it would have to be a
  duration resolved per rate. Adjacent patches are assigned independently.
  The paper presents patching as the replacement for the Viterbi step that
  CTE and TOT need.
- **mosaic** (`behavior/feature_library/arhmm_model.py`): `sticky_weight`
  (default 100) is added to the diagonal of the transition counts; Viterbi
  then finds the best path, switching only where evidence outweighs the cost.
- **For us:** Viterbi-style output gives better boundaries, which matters
  because curators correct boundaries; patch-style output gives cleaner,
  longer segments with onsets off by up to half a patch.
- **Untested idea:** SMQ returns per-frame distances to every code; a sticky
  Viterbi over those would give frame-accurate boundaries and address the
  authors' own stated limitation.

## Tool notes

### mosaic (EcodylicScience)

A whole pipeline, not a comparison layer: drives trackers (TRex, SLEAP,
Lightning Pose, Ultralytics), 45 registered features, t-SNE, k-means, Ward,
AR-HMM, keypoint-MoSeq, and supervised classifiers including FERAL. AGPL-3.0,
Python ≥ 3.12, version 0.x. No VAME or B-SOiD, so it is not the "compare
unsupervised methods" tool. Overlaps Ethograph's scope. Verdict: not a
backend; its per-frame output is importable like anyone's. Cloned at
`repos/mosaic`.

### SMQ (Gökay, Spurio, Bach, Gall — ICCV 2025)

- Dilated-TCN autoencoder, each joint encoded independently with shared
  weights, latent patches quantised to "skeleton motion words".
- Benchmarks are human only: HuGaDB (six-axis IMU on thighs, shins, feet),
  LARa (mocap positions + orientations), BABEL (SMPL joints).
- LARa ablation: frame-wise MoF 33.9 / F1@50 8.6 → 1 s patches MoF 37.4 /
  F1@50 16.4 *(unverified)*. Absolute scores are low: proposals would need
  heavy curation even on its own benchmarks.
- K is the ground-truth class count by protocol; the paper suggests the
  silhouette score for choosing it in practice *(unverified)*.
- Stated limitation: transitions only at patch boundaries.
- **Non-pose inputs: partly.** It runs on IMU data unchanged, but assumes
  nodes × channels × time with identical channels per node; the reconstruction
  loss is built on inter-node distances. A set of identical sensors fits; a
  heterogeneous feature table (speed, a distance, an audio envelope) does not
  without changing the method. How much the code enforces this was not read.
- Research code: ~2000 lines, 5 commits, conda env, no pip install, no
  custom-dataset instructions, AGPL-3.0.
- Follow-ups exist, titles only *(unverified)*: hierarchical spatiotemporal
  VQ (arXiv 2604.15196), MASQ (arXiv 2608.29891). Wiring in one method risks
  it being outdated within a year.
- `repos/SMQ` is an empty failed clone (`.git` only, no commits); re-clone
  before reading it there.

### Dynaclade (Bach lab)

Known only from the talk "The grammar of human behaviour in a biological
environment", satellite workshop "Towards a quantitative approach to
behavior", Bernstein Conference, Frankfurt, 2026-09-28. No code, preprint or
listing found under that name. From the talk: takes derived input features
(derivatives, joint angles, …), not only keypoints, and has a human-in-the-loop
step ("cut clustering solution, semantic labelling & merging, sort into
categories") — the same step position 4 above needs. An email asking about
code release, hand-crafted features and how well that step generalises is
drafted; `email_bachlab_action_proposals.md` holds an older version of it.

### Existing tools that do parts of the loop

| Tool | Does | Differs from the loop above |
|---|---|---|
| A-SOiD (Nat. Methods 2024) | Active learning on pose features, then unsupervised discovery of sub-classes inside each label | Starts from labels; frame-wise classifier, not a segmentation model |
| OpenLabCluster | Clusters keypoint segments, active learning picks what to annotate, GUI | Segments must be pre-cut; classifies clips, finds no boundaries |
| CASTLE (bioRxiv 2025) | Foundation-model video features; human cuts and names a hierarchical clustering | Closest to Dynaclade's step; no supervised model after; no pose or hand-crafted features |
| B-SOiD | Clusters pose features, trains a random forest to reproduce the clusters | Human only names clusters; no boundary correction |
| keypoint-MoSeq, VAME | Syllables/motifs with dendrogram or community merging, grid movies | Syllable-level; ends at the unsupervised result |
| ABEL (bioRxiv 2026) | "Active-learning behavior estimation and labeling platform" | Title only *(unverified)* — read before claiming a gap |

No tool found that combines action-level proposals, boundary correction on a
timeline with synced media, and a temporal segmentation model at the end.
Ethograph already has the last two. A-SOiD is the one to cite and
distinguish. CASTLE reports its cut-and-name step working across mice, flies
and worms, weak evidence that the step transfers between labs.

CASTLE's code is public although the preprint does not link it:
https://github.com/CASTLE-ai/castle-ai — Apache 2.0, on PyPI (also Colab and
Docker), browser GUI, video input only, benchmarked on a 12 GB RTX 3060,
last push 2026-09-24.

Read from the code (2026-10-01):

- **Export is directly importable.** `cluster/time_series_{video}.csv` has one
  `behavior` id per frame, `cluster/id.csv` maps `Id` → `Name`, `Color`; an
  `.srt` per video carries the same bouts. The per-frame importer plus the
  id table covers it.
- **The clustering core is feature-agnostic and small** (`utils/latent_explorer.py`,
  ~300 lines): `Latent(raw (T, D), time_window)` concatenates frames into bins,
  every bin starts in one cluster `init`; the user selects a cluster, runs
  UMAP then DBSCAN on **that subset only**, names the sub-clusters, imports
  them back, merges. State is one int per bin plus an id → name/colour table;
  merge relabels to the smallest id. Nothing in it needs video features.
- **Feature recipe**: SAM mask + DeAOT tracking → ROI centred, optionally
  rotated to a tail point or averaged over 7 rotations → DINOv2. Most of the
  24k lines are the vendored SAM and DeAOT.
- **What it lacks**: no temporal prior (bins are clustered independently,
  DBSCAN runs in UMAP space), bin size is in frames, no boundary correction,
  no supervised model. GUI is gradio; `cuml` for GPU UMAP/DBSCAN.
- **Worth adopting as ideas, not code**: (1) the select → embed → cluster →
  name → merge loop over *any* `(T, D)` feature, as the cluster-branch step
  of position 4; (2) the id-table + per-frame layout as the import contract;
  (3) cropping to the animal before a frame-wise extractor, from pose
  keypoints here, with no SAM or tracker.
- **Not worth adopting**: SAM/DeAOT, the gradio app, its project layout.
- **CASTLE does not solve fragmentation.** Its only temporal device is the
  optional time window, default 1 frame; there is no smoothing, minimum
  duration or decoding step anywhere in the clustering code. Continuity comes
  only from adjacent frames having similar features. Human merging removes
  flicker between clusters that end up under one name, never flicker between
  different behaviours. So fragmentation is a property of the assignment
  step, not of the human loop. Three fixes, cheapest for us first:
  (a) **segment first, cluster second** — cluster the segments between our
  changepoints, so every unit is continuous by construction; (b) a sticky
  Viterbi over per-frame cluster distances; (c) features that carry temporal
  context (windows, derivatives, clip-wise S3D).

### OpenLabCluster — not installable beside Ethograph

On PyPI (`openlabcluster` 0.0.37, one release, 2023-06-30), BSD-style licence,
~5400 lines, wxPython GUI. Every dependency is pinned to 2021: `numpy==1.19.5`,
`torch==1.8.0`, `pandas==1.1.5`, `scikit-learn==0.24.1`, `deeplabcut==2.2.0.6`.
None of those has a wheel for Python ≥ 3.11, which Ethograph requires, so it
cannot share an environment; it would be a separate env exchanging files.
It also does not find *where* a behaviour is: it embeds and classifies
segments the user has already cut. Its reusable idea is the active-learning
ranking — which segment to annotate next (cluster centres first, then
uncertain ones) — which pairs with segment-first clustering.

### Blau et al. 2024 — supervised vs unsupervised vs semi-supervised

| Paradigm | Models |
|---|---|
| Supervised | TCN, random forest, XGBoost |
| Unsupervised | keypoint-MoSeq |
| Semi-supervised | S3LDS (switching linear dynamical system with a TCN inference network), plus two variants without temporal dynamics |

- The fully supervised TCN won on every dataset once velocity features were
  added.
- Semi-supervision helped only with position-only inputs; velocities removed
  the gain. Our feature columns already carry derivatives, so this weakens
  the case for a semi-supervised model here.
- keypoint-MoSeq, scored by mapping each state to its highest-overlap
  behaviour, was competitive but worse. That many-to-one mapping is the merge
  step a curator would do by hand.
- Supports the loop: unsupervised states are a starting proposal, a
  supervised model on corrected labels ends up better.
- Datasets: head-fixed fly, freely moving mouse, IBL head-fixed mouse, and
  HuGaDB — the same IMU set SMQ uses, a rough common reference.

## Segment first, cluster second — prior art

The approach is established; it is not a gap.

- **DISSeCT** (PLOS Biology 2025), rodent IMU: kernel change-point detection
  into statistically homogeneous segments, then a Gaussian mixture over ~240
  per-segment features. The closest published match to experiment 3 below,
  and on non-pose sensor data.
- **Behavior Atlas** (Huang et al. 2021, Nat. Commun.), 3D mouse pose:
  decomposes the pose sequence into movement segments, then embeds and
  clusters the segments *(details from memory, unverified)*.
- **ABD** (Du et al., CVPR 2022), human video: change points in frame
  similarity with non-maximum suppression, then clustering of the segments.
  Training-free.
- **Birdsong** (e.g. Sainburg et al. 2020, AVGN): segment syllables by
  amplitude, then embed and cluster them. The default in bioacoustics, and
  what our audio changepoints already resemble *(from memory)*.
- **OpenLabCluster** assumes this structure but leaves the cutting to the user.

Known weaknesses: a missed changepoint merges two behaviours into one segment
that no clustering can split, so the detector should over-segment and the
clustering merge; segments of unequal length need a fixed-size descriptor
(summary statistics, as DISSeCT does, or time-warped distances); gradual
transitions have no sharp changepoint. Joint models (keypoint-MoSeq, VAME's
HMM) exist because of the first point.

## Cheapest experiments, in order

1. Write an NWB with neuroconv's keypoint-MoSeq interface and see whether its
   bouts already load through the pynapple label path.
2. Add the per-frame array importer.
3. Without any new dependency: cluster the segments between our existing
   changepoints. Takes any feature time series, which SMQ does not.
4. SMQ on one pose session, outside Ethograph: export pose in its `.npy`
   layout, train, load the `.txt` back as a prediction set, look at it in the
   grid view. Keep the export script in `tests/` or `examples/`; document as
   a recipe only if the proposals are useful.
5. The cluster → class rename/merge step in the Curation section.

## Sources

- mosaic: https://github.com/EcodylicScience/mosaic
- NeuroConv 0.10.0 release: https://catalystneuro.com/blog/neuroconv-0100-release/
- neuroconv keypoint-MoSeq interface: https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/behavior/moseq_keypoints.html
- neuroconv VAME interface PR: https://github.com/catalystneuro/neuroconv/pull/1737
- Weinreb et al. 2024, *Keypoint-MoSeq*, Nat. Methods. https://www.nature.com/articles/s41592-024-02318-2
- Gökay et al. 2025, *Skeleton Motion Words (SMQ)*, ICCV. https://arxiv.org/abs/2508.04513 · https://github.com/bachlab/SMQ
- *Hierarchical Spatiotemporal Vector Quantization*. https://arxiv.org/abs/2604.15196
- *MASQ*. https://arxiv.org/pdf/2608.29891
- Bach lab at Bernstein 2026: https://www.caian.uni-bonn.de/en/news-events/bc_2026
- Tillmann et al. 2024, *A-SOiD*, Nat. Methods. https://www.nature.com/articles/s41592-024-02200-1
- Li, Keselman, Shlizerman 2025, *OpenLabCluster*. https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1630654/full
- *CASTLE*. https://www.biorxiv.org/content/10.1101/2025.08.22.671685v2 · https://github.com/CASTLE-ai/castle-ai
- *ABEL*. https://www.biorxiv.org/content/10.64898/2026.08.30.748115v2.full
- Fayat et al. 2025, *DISSeCT*, PLOS Biology. https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003431
- Huang et al. 2021, *Behavior Atlas*, Nat. Commun. https://www.nature.com/articles/s41467-021-22970-y
- Du et al. 2022, *ABD*, CVPR. https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Fast_and_Unsupervised_Action_Boundary_Detection_for_Action_Segmentation_CVPR_2022_paper.pdf
- Blau et al. 2024, *A study of animal action segmentation algorithms across supervised, unsupervised, and semi-supervised learning paradigms*. https://arxiv.org/abs/2407.16727
