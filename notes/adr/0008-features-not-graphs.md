# ADR 0008 — The pose side is a list of features, not a graph

**Status:** accepted (2026-08-27). Supersedes the graph half of ADR 0007's
premise; the distillation mechanism it records stands.

## Context

The multimodal recipe was built on UMEG-Net's unified entity graph: moving
keypoints and static landmarks as nodes, user-declared edges, an adaptive
spatial graph convolution, distilled into E2E-Spot. Making it fit a rig with a
beak, a stick and a pellet grew the config a `graph:` section with `nodes`,
`static` (names or fixed coordinates), `edges` (or `all`), `along`
(interpolated points on a segment), and the teacher three input switches
(`seen_channel`, `edge_distances`, `box_frame`) plus a trunk switch.

Measured on 590 trials, share of held-out events within 20 ms
(first contact / last contact):

| model | input | first | last |
|---|---|---|---|
| LightGBM lightgbm model on hand-picked distances | pose | 94 % | 46 % |
| E2E-Spot baseline | video | 86 % | 65 % |
| graph teacher (17 nodes, edge distances, extras) | pose | 81 % | 48 % |
| distilled student | video | 86 % | 57 % |
| curves fused, no training | both | 86 % | 56–58 % |

Two things followed. The graph teacher lost to a tree ensemble on hand-picked
distances with *more* inputs: `ReLU(A·H·W)` is linear in coordinates before
the nonlinearity and cannot compute a norm, so the events — distance
crossings — were carried entirely by the explicit `edge_distances` channels,
and the message passing added averaging. And the vocabulary needed to
express "which part of the stick, relative to which part of the wall" grew
past what a user can reason about, while restating quantities (a wall
distance is a coordinate; four corner distances are a position) a user could
have written down directly and plotted.

## Decision

- **The pose side of `eto.spot` is a flat `features:` list** in the
  segmentation pipeline's column spelling — variables in the session file the
  user computed (`features/geometry.py` or their own code) and can plot in the
  GUI. No graph, no adjacency, no interpolated nodes, no input switches.
- **Four models, decided by what exists at inference**: the LightGBM onset
  model (pose only, GUI); E2E-Spot (video only, MSAGSM optional); E2E-Spot
  with `features:` as a second input to its GRU (`train.features_as_input`,
  modality dropout `train.features_dropout`, ablated by
  `evaluate(zero_features=True)`); and the pose teacher — the listed features
  through multi-scale shifts and a bi-GRU — distilled into E2E-Spot for the
  case where pose exists only for the training sessions.
- **The graph code is deleted**, not switched off: `GraphConfig`, `along`,
  `edges`, `static`, the GCN block, `edge_distances`, `box_frame`,
  `seen_channel`, `trunk`, and the `fuse:` section. A config that still
  spells them is refused by name with the replacement.

## Consequences

- A user who wants "distance from the middle of the stick to the pellet"
  writes that variable into the `.nc` and lists it. It is then visible, and
  the model gets exactly it.
- The teacher is the only neural pose model, and it is deliberately the
  simplest one: features → shifts → GRU. Its use is distillation; the gate is
  that it must beat the video baseline on the same test split before a
  student is distilled from it.
- UMEG-Net's contribution that survives is the parameter-free multi-scale
  temporal shift and the distillation recipe; the graph does not, for this
  data. The literature regime it was built for (many homogeneous joints,
  several animals) is not served by this pipeline, and is not meant to be.
