# Finding periods where a cell is active — options (2026-10-02)

The question: a very simple, generic way to turn one unit's spikes into
"interesting periods" that can be walked, played back and looked at in the
grids as a read-only label source. Nothing here is implemented beyond options
1 and 2; this is a list of what could replace or sit beside them.

**Status of the citations: written from memory, not checked against the
papers. Verify before citing or building on a detail.**

## What "generic" has to mean here

- One knob at most, and it should mean the same for every unit. Units differ
  in baseline rate by two orders of magnitude.
- No assumption that the session has repeated, aligned trials. Free behaviour
  is one long recording; a method built for a PSTH does not transfer as is.
- Robust to a rate distribution that is skewed and, for sparse units, mostly
  zeros.
- Cheap enough for hundreds of units over a whole session.
- Says something about a unit that is *never* really active, rather than
  inventing periods for it.

## What exists now

`labels/feature_events.py`, driven from Tools ▸ Labels: Create from a feature.
Since this note was first written: option 3 (per-unit percentile) is
implemented as a third threshold scale, and the instantaneous rate
(1 / ISI, option 7a below) is offered as a feature beside the binned rate.
Bayesian binning (option 10) was dropped as a period finder.

1. **Fixed threshold on the smoothed rate.** Honest only when the number has
   a meaning (20 Hz for this unit). One threshold fits no population.
2. **Z-scored threshold**, each unit against its own mean and spread across
   the trials shown. The current default answer to "generic". Weakness: mean
   and std are the wrong summary of a skewed, zero-inflated rate — a few
   bursts inflate the std and hide the next ones, and the result depends on
   the bin and smoothing chosen in the firing-rate panel.

Both find periods as threshold crossings, then stitch gaps and drop blips.
Crossing times are biased by the smoothing kernel: wider σ moves onsets
earlier and offsets later.

## Options on the rate (small changes to what exists)

3. **Per-unit quantile.** "The top 5 % of this unit's own time." No
   distributional assumption, one knob, identical meaning for every unit.
   Cost of that: every unit gets 5 % by construction, so it ranks periods
   within a unit and cannot say whether the unit was ever active. Good for
   browsing, wrong for counting.
4. **Robust z (median and MAD).** Fixes the inflated-std problem for units
   that fire most of the time. Breaks for sparse units: when more than half
   the bins are empty the MAD is 0. Would need a fallback, which is a second
   rule — not simpler than option 2 in practice.
5. **Hysteresis (two thresholds).** Enter above a high threshold, leave below
   a lower one. Replaces most of what stitch + drop do with something that has
   a clearer meaning, and stops a rate hovering at the threshold from
   chattering. An add-on to 2 or 3, not an alternative.

## Options on the spike times (no bin, no smoothing)

6. **Poisson surprise** (Legéndy & Salcman 1985) and its non-parametric
   cousin **rank surprise** (Gourévitch & Eggermont 2007). For a run of
   spikes, how improbable is that many spikes in that little time, given the
   unit's own mean rate. One knob (the surprise threshold), meaning the same
   for every unit, and no bin size or kernel at all. A silent unit yields
   nothing, which is the right answer. Decades of use as a burst detector.
   Weaknesses: assumes a stationary baseline, so slow drift across a long
   session reads as one long "burst"; designed for bursts of tens of
   milliseconds, so behaviourally long periods need the surprise maximised
   over longer runs. Probably the best simplicity-to-principle ratio on this
   list.
7a. **Instantaneous rate, 1 / ISI** (the birdsong field's definition: from
   each spike until the next, the rate that interval implies). A step
   function with no bin and no kernel; thresholding it is the max-interval
   rule below written as a rate. Implemented (`features.neural.instantaneous_rate`).
   Noisy for irregular firing: one short interval is a one-interval "burst",
   so it leans on the minimum-duration rule. Thresholding it is the
   birdsong field's burst definition: "the interval over which the
   instantaneous firing rate exceeded a threshold of 125 Hz" (Leonardo & Fee
   2005, J Neurosci 25:652, https://www.jneurosci.org/content/25/3/652);
   other songbird papers use 100 Hz. Checked 2026-10-02: the figure is 125,
   not 200.
7. **Max-interval / ISI threshold.** Spikes closer than x ms belong to one
   period. The simplest thing that works on spike times, but x is per unit
   again — the same problem as option 1.

## Options that model states

8. **Two-state HMM per unit** (low rate / high rate, Poisson emissions,
   Viterbi path). No threshold: the two rates and the switching
   probabilities are fitted. Persistence is part of the model, so stitching
   and blip removal are no longer separate steps. Fails quietly when a unit
   is not bimodal — it will still return two states. Needs a fit per unit;
   fine for tens of units, slow for hundreds unless counts are coarse.

## Changepoints, including Bayesian binning

9. **Generic changepoint detection on counts** (`ruptures` is already a
   dependency: PELT with a Poisson or L2 cost). One knob, the penalty.
   Linear-ish in time. Returns boundaries only.
10. **Bayesian binning** (Endres, Oram, Schindelin & Földiák, NIPS 2007,
    "Bayesian binning beats approximate alternatives"; Endres & Oram 2010,
    J Comput Neurosci, on latency; Endres, Christensen, Omlor & Giese 2011,
    "Emulating human observers with Bayesian binning: segmentation of action
    streams"; and the Endres & Giese temporal-segmentation work that
    followed). The series is modelled as piecewise — constant firing
    probability per bin for spikes, piecewise polynomial for joint
    trajectories in the Giese line — and the posterior over the number of
    bins and where the boundaries sit is computed **exactly** by dynamic
    programming, rather than by picking one segmentation.
    - What it gives that 9 does not: a posterior probability for every
      possible boundary, the number of segments inferred rather than set by
      a penalty, and error bars on each segment's rate. No smoothing kernel,
      so boundaries are not biased the way threshold crossings are.
    - What it costs: roughly quadratic in the number of time points (times
      the maximum number of bins). Comfortable for a PSTH of a few hundred
      bins; a whole free-behaviour session at 10 ms is not. It would have to
      run per trial or per window at a coarse grid.
    - What it was built for: spikes pooled over repeated presentations of a
      stimulus. On a single pass through free behaviour each fine bin holds
      0 or 1 spike, so the posterior on boundaries will be broad for all but
      strongly modulated units. That is an honest result, but it is not many
      periods.
    - Availability: I do not know of a maintained Python implementation.
      Assume it would be written from the paper, like `msagsm.py`.

**The catch with every changepoint method: it finds boundaries, not
activity.** A segmentation still needs a rule for which segments count as
"active" — segment rate against the unit's baseline — which brings back a
threshold (though on a few segment means instead of thousands of noisy bins,
which is a real improvement).

**Where changepoints fit this codebase.** Ethograph already has the
machinery: a mask marked `changepoint_mask` with a `target_feature` is drawn
on that feature's panel, clicks snap to it, and `snap_boundaries` /
`correct_changepoints` move label edges onto it. So neural changepoints are
most naturally a **changepoint feature of the firing rate**, not a period
detector:

- periods come from a cheap rule (2, 3 or 6);
- their edges are then snapped to the nearest neural changepoint, exactly as
  hand-drawn labels are snapped to kinematic ones.

That reuses one concept instead of adding a second way of making periods,
and the snapping step is where the smoothing bias of threshold crossings
gets corrected.

## Recommendation

- **Simplest generic default:** keep the z-scored threshold, add the
  per-unit quantile as the alternative scale and hysteresis as the blip
  rule. All three are a few lines in `feature_events.py` and need no new
  concept.
- **First principled step up:** Poisson surprise on spike times. One knob,
  no bin, silent units stay silent. Worth a prototype against the z-score on
  a session where the answer is known by eye.
- **Bayesian binning:** not as the period finder. Worth having as a neural
  changepoint feature — boundaries with a posterior — once there is a need to
  place edges precisely, and only per trial. Try `ruptures` with a Poisson
  cost first: if its boundaries are good enough, the exact posterior is not
  buying anything a user would see.
- **Not on the list but adjacent:** a population criterion ("at least 3 of
  these 10 units active") is a derived feature thresholded the same way, not
  a new method.

## Open questions

- Is the interesting thing a period per unit, or the moments many units
  change together? The second is a population changepoint problem and
  favours 9/10 on a low-dimensional projection.
- How long is a period worth looking at — 50 ms bursts or seconds-long
  elevations? Poisson surprise suits the first, HMM and changepoints the
  second.
- Is baseline drift across the session large enough to break a stationary
  baseline (options 2, 3, 6)? A running baseline fixes it at the cost of a
  window-length knob.

## Reading (checked 2026-10-02)

Bursts of tens of milliseconds:

- Cotterill & Eglen, *Burst detection methods* (review chapter):
  https://arxiv.org/abs/1802.01287 — the place to start; describes Poisson
  surprise, rank surprise, MaxInterval, logISI and others side by side.
- Cotterill, Charlesworth, Thomas, Paulsen & Eglen 2016, *A comparison of
  computational methods for detecting bursts in neuronal spike trains*,
  J Neurophysiol: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4969396/ —
  eight methods scored on synthetic trains. **MaxInterval and logISI came out
  ahead; several common methods, Poisson surprise among those tested, did
  poorly on some kinds of train.** MaxInterval is an ISI threshold plus
  minimum-duration rules, i.e. close to thresholding the instantaneous rate
  with the blip rules we already have. This weakens the case made above for
  Poisson surprise as the next step.
- Legéndy & Salcman 1985, J Neurophysiol 53(4):926–939 — the original
  Poisson surprise.
- Gourévitch & Eggermont 2007 — rank surprise, the non-parametric version
  (described in the review above).

Elevations lasting seconds:

- Chen, Vijayan, Barbieri, Wilson & Brown 2009, *Discrete- and
  continuous-time probabilistic models and algorithms for inferring neuronal
  UP and DOWN states*, Neural Computation 21:1797–1862:
  https://dspace.mit.edu/handle/1721.1/70846 — the two-state HMM, and a
  semi-Markov version that models how long each state lasts.
- Truong, Oudre & Vayatis 2020, *Selective review of offline change point
  detection methods*, Signal Processing 167:107299:
  https://arxiv.org/abs/1801.00718 — the paper behind `ruptures`; cost
  function, search method and penalty laid out separately.
