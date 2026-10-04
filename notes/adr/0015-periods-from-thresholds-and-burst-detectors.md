# ADR 0015 — Periods from the data: a threshold for long events, burst detectors for short ones

**Status:** accepted (2026-10-02); amended 2026-10-04 — the burst
detectors are in the GUI now, not waiting on pynapple. Records which ways of turning a feature
or a spike train into browsable periods are kept, which were considered and
left out, and what would reopen them. Options and reading:
`notes/Lit_active-periods.md`.

## Context

A user wants to look at the moments something happens in the data — a unit
is active, a keypoint moves fast — with the tools that already exist for
labels: label navigation, playback, the two review grids. Such periods
overlap across units and nobody judged them, so they cannot be rows of
`labels.tsv`. They are a read-only label source held in memory
(`labels/feature_events.py`, `PredictionSet.mappings`,
`app_state.label_source`), made from Tools ▸ Labels: Create from a feature (a session's features) or Tools ▸
Neural: Neuronal firing / burst detection (the units) — one dialog, two entries.

Many methods were on the table for *finding* the periods: a threshold on
three scales, Poisson and rank surprise, a two-state HMM, changepoints
(`ruptures`, Bayesian binning), slope thresholds for ramps, and the burst
detectors of the Cotterill & Eglen comparison. Each is a setting a user has
to understand and a method a paper has to justify.

## Decision

Two tools, split by timescale. Nothing else.

- **Long events (hundreds of milliseconds to seconds): a threshold on the
  feature.** The z-scored threshold is the default answer: each series
  against its own mean and spread across the trials shown, so one number
  serves every unit or keypoint of a dim. Stitching short gaps and dropping
  short periods is the only clean-up. It is one sentence in a methods
  section and needs no citation.
- **Short events (bursts of tens of milliseconds): the two established
  detectors, MaxInterval and logISI**, ranked first and second of eight by
  Cotterill et al. (2016). They are citable, deterministic and have a
  reference implementation to test against. They belong in pynapple:
  `notes/drafts/pynapple_burst_detection.py` is the draft,
  `notes/drafts/pynapple_burst_issue.md` the text of the pynapple issue
  proposing it (related: #627). Until pynapple ships them they live in
  `ethograph/features/bursts.py`, the same code typed, and
  `tests/_test_burst_r_reference.py` checks both copies against the R
  references. The units' dialog offers them as a method beside the threshold
  (`labels/feature_events.burst_events`, `BurstRule`): the method is
  picked first, and a detector's parameters replace the threshold's. Their
  `IntervalSet` becomes a label source like any other, one class per unit.
- **Thresholding `instantaneous_rate` (1 / ISI)** stays as the burst
  tool on the threshold side. It is MaxInterval's core rule — an ISI limit, a minimum
  duration, a merge gap — without its separate start threshold and minimum
  spike count. It is also a method with a citation of its own: the birdsong
  literature defines a burst as the interval over which the instantaneous
  firing rate exceeds a threshold, 125 Hz in Leonardo & Fee (2005, J Neurosci
  25:652) and 100 Hz elsewhere. The dialog opens the instantaneous rate at
  125 Hz (`features.neural.BURST_INSTANTANEOUS_RATE_HZ`), and the pynapple
  draft carries it as a third detector, `detect_bursts_instantaneous_rate`.
  The number is that paper's, for zebra finch RA neurons: a starting point.

The threshold keeps three scales:

- **Raw**, for a number that means something (a distance, a known rate).
- **Z-score**, the default for a dim read value by value.
- **Percentile**, kept. It is the only scale that assumes nothing about the
  distribution, and a firing rate is skewed and mostly zeros — exactly where
  a mean and a standard deviation mislead. Its known cost is stated in the
  dialog: every series gets its share by construction, so it ranks periods
  within a series and cannot say whether the series was ever active. It is
  one dropdown entry and is already tested; removing it would save nothing
  a user sees.

## Considered and left out

- **Bayesian binning (Endres et al.).** Finds boundaries, not activity; was
  built for spikes pooled over repeated trials; cost grows with the square
  of the number of time points; no maintained Python implementation.
- **Poisson surprise, rank surprise, cumulative moving average.** Ranked
  fourth or lower in the same comparison; Poisson surprise and CMA
  overestimate bursting in trains with little of it and under a drifting
  rate. They fit the burst detectors' interface and can be added upstream
  if someone needs them.
- **Two-state HMM.** A fit per unit, and it returns two states for a unit
  that has one.
- **Ramps (a threshold on a sliding-window slope).** Generic and cheap to
  add, but not an established method: only its ingredients are citable, and
  on single trials a slope cannot tell a ramp from a step smoothed by the
  rate kernel (Latimer et al. 2015).
- **Changepoints as a period finder.** A segmentation still needs a rule
  for which segments count, which brings the threshold back.

## Consequences

- One dialog, one rule to explain. A new kind of event is first a new
  *feature* (a population rate, an active-unit count) thresholded the same
  way, not a new method.
- The detectors exist twice until pynapple takes them: the draft and
  `ethograph/features/bursts.py`. When it does, the module becomes an
  import of pynapple's; if it declines, the module simply stays.
- The automatic per-unit threshold that logISI offers stays out of the
  threshold dialog. Its valley score ignores how many values are in a bin
  and produced a meaningless threshold on a Poisson train during testing.

## Reopen when

- Sessions show slow baseline drift that a session-wide mean cannot absorb —
  then a running baseline for the z-score, not a new method.
- Boundaries of threshold periods need to be precise — then neural
  changepoints as a *changepoint feature* that edges snap to
  (`correct_changepoints`), with `ruptures` tried before anything exact.
- Ramps or population state changes become the question being asked, with
  data where the answer is known by eye to judge a detector against.
