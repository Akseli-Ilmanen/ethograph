(target-curation-difficulty)=
# Hard trials and review F1

Some trials are simply difficult, and some the model barely managed. Both are
worth showing it more often next time.

## Flagging a trial by hand

`Ctrl+T` (or the **Hard trial** box in the Curation section) flags the
current trial as `hard` in the metadata table's `difficulty` column, and
again puts it back to `normal`. The column is a checklist in the trials
table's filter like `curated`, so "show me the hard trials" is one click, and
a training run reads it back: in the segmentation pipeline,
`train.oversample: {column: difficulty, weights: {hard: 3.0}}` draws a hard
trial three times as often ({doc}`../segment/config`).

## Scoring how the model did

Which trials the model barely managed is measured, not guessed. Every
prediction run keeps the labels it wrote in its folder under `labels/`
(`{stem}_predictions.tsv`, beside `onset_curves.npz`), and the moment the last
trial the table shows becomes curated, each trial's final labels are scored
against what its run said — an F1 per trial and event type:

- **State labels** match at IoU ≥ 0.5, the `f1@50` the pipeline selects its
  checkpoints on.
- **Point labels** match within the run's own **tolerance** — the precision the
  model was trained to, read from the run folder (the onset model's
  `tolerance_s`, half of spot's `focus_window_ms`). Nothing to pick; set the
  section's **Tolerance** only to judge every run at one number, or for an old
  run folder that carries none.

The scores land in the metadata table as `review_f1_state` and
`review_f1_point` (blank where a trial had nothing of that type), a message
says how the session did, and every trial scoring **below the threshold**
(the section's *Flag below F1*, default 0.5) is flagged `hard`. Auto-flagging
only ever adds: a trial you flagged by hand stays hard whatever it scored.
A trial no run predicted into is not scored. **Score now** runs the same
comparison without waiting for the last trial.
