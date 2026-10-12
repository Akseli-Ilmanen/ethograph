# Vendored SAMURAI

A partial copy of [SAMURAI](https://github.com/yangchris11/samurai) by
Cheng-Yen Yang, Hsiang-Wei Huang, Wenhao Chai, Zhongyu Jiang and Jenq-Neng
Hwang (Information Processing Lab, University of Washington) at commit
`76ba195` (`master`, 2025-03-18). Apache License 2.0 — see `LICENSE`.

Vendored 2026-10-07. Used by `ethograph/samfill/samurai.py`, which re-implements
SAMURAI's two changes to SAM 2 (`sam2/modeling/sam2_base.py` in the upstream
fork) as a subclass of OCTRON's predictor instead of a replacement `sam2`
package: the Kalman-filter re-ranking of SAM's candidate masks and the
motion-aware memory bank. Only the filter itself is copied.

## What is vendored

| Path | Upstream origin |
|---|---|
| `kalman_filter.py` | `sam2/sam2/utils/kalman_filter.py` (verbatim) |
| `LICENSE` | `LICENSE` |

## Edits

None.
