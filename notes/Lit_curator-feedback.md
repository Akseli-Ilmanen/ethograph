# Curator feedback: which idea came from where

Private notes for a later paper. What Ethograph does, the source each part
leans on, and what each source does *not* support. Bib keys are the ones in
`docs/source/references.bib`.

## The design, in one paragraph

Curation is human review of a model's predictions. Two signals come out of
it and Ethograph keeps them apart. **Confidence** (the model's own number)
decides what the human looks at first: grids sorted lowest-first, flagged
tiles, the frame-by-frame curve. **Disagreement** (the human overruling the
model) decides what the next training run sees more often: the `difficulty`
column, `hard` / `normal`, read by `train.oversample`. A hard flag is only
ever written by a human — by hand (`Ctrl+T`), or once from the per-trial F1
histogram after looking at the distribution. Confidence never writes it.
Trial-level confidence exists too (`model_confidence`, the mean of the frame
curve), but as a column to sort and filter the trials table on — which trials
to open, which to curate in bulk — with no threshold in the GUI and no path
into `difficulty`.

## Idea → source

| Idea in Ethograph | Source | What to take from it | What it does not support |
|---|---|---|---|
| Sort and flag labels by the model's confidence so the doubtful ones are reviewed first | `settles2009active` (survey) | The vocabulary: this is *uncertainty sampling* — least-confident, margin (our bimodal `ratio`), entropy (our frame curve). Fewer labels needed when the learner picks them. | Nothing about training weights. And the survey's own caveat: uncertainty sampling is blind to confident errors and can spend its budget on outliers. |
| Train on everything the curator corrected, aggregated across rounds, rather than on a fixed hand-labelled pool | `ross2011dagger` (DAgger) | The corrections must come from the states the *current* learner visits; aggregate them, retrain, repeat. Formal result: errors stop compounding. Our curate → score → retrain loop is this with "policy" read as "classifier". | Says nothing about *weighting* the corrections. Its benefit is already obtained by training on all curated trials; oversampling is extra. |
| Show the trials the model got wrong more often (`train.oversample`) | `freund1997adaboost` (boosting), `shrivastava2016ohem` (OHEM) | Upweighting misclassified examples for the next learner is a standard, effective move. OHEM does it per example inside one training run. | Both assume "wrong" means informative. With label noise or genuine ambiguity the same move memorises noise (AdaBoost is fragile to noisy labels). Our weight is per *trial*, coarser than either. |
| The human decides what the model learns from, not a rule | `simard2017machineteaching` | The framing: the human as teacher, tooling built around the teacher's decisions. | A position paper; no mechanism. Cite for the framing only. |
| No default cut-off; the flag is a deliberate press from the histogram, and few trials should carry it | — (design decision) | Follows from the two caveats above: a threshold at 0.5 is arbitrary, and over-flagging turns the training set into mostly "hard" trials. | — |

## Distinctions worth stating in a paper

- Uncertainty sampling finds what the model knows it does not know. Curator
  disagreement finds what it does not know it does not know. A-SOiD-style
  loops never see a confident mistake because they never ask about it.
- The F1 against the run is anchored: a curated label is a prediction a human
  accepted, so a session confirmed with `N` throughout scores near 1 whatever
  the model did. It measures how much the curator had to change, not model
  accuracy. The pipeline's held-out evaluation is the benchmark.
- Trial-level oversampling is an approximation of hard-example weighting. The
  honest claim is "the weight is chosen by cross-validation", and the
  experiment is weight 1 vs weight 3 on the same folds. Not yet run.
- Curriculum learning (easy first) argues the opposite of hard-example
  emphasis, and the evidence is dataset-dependent. Small, imperfectly
  labelled lab datasets are the regime where emphasis is riskiest. This is
  the argument for keeping the flag rare and human.

## Not built, deliberately

- Per-frame weights on the disagreeing spans (the OHEM-shaped version).
  One to two days through materialise → dataset → augment → loss; not worth
  it before the weight-1-vs-3 experiment shows trial-level weighting moves
  anything.
- Any automatic flag from confidence or from F1.
- Boosting ensembles, committees, expected-error-reduction querying.

## Reading order

1. DAgger (`ross2011dagger`), section 3 — the loop, eight pages.
2. Settles survey, first half — the vocabulary and the failure modes.
3. AdaBoost intro + section 3 — skim for the reweighting rule only.
4. OHEM and machine teaching — cite, do not read.
