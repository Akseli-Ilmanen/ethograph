# 3. The variable schema is advisory, and `kind` never selects behaviour

Date: 2026-08-23

## Status

Accepted. `type="changepoints"` and `kind="changepoint_feature"` are both
read as equivalent spellings for a raw changepoint mask (see "Both spellings
are read" below); a raw mask is additionally marked by `changepoint_mask` so
the mask predicate does not depend on which spelling a file uses.
`ethograph.io.schema.migrate_legacy_attrs` still exists to normalise an old
file onto the full stamp and to drop other stale `type` values, but is no
longer required for a changepoint mask to be recognised.

*History:* this clause was briefly amended (2026-08-23) to drop the legacy
spelling entirely — one spelling, migrate-on-read. That traded a live
fallback for a migration step, which turned out to be the wrong trade for a
file nobody had migrated yet; the amendment was reverted the same day.

## Context

movement [issue #978](https://github.com/neuroinformatics-unit/movement/issues/978)
proposes describing derived features by their `attrs` — `kind`, `source`,
`units`, `is_egocentric` — so that trackers and behaviour classifiers can
exchange engineered features, selected with `ds.filter_by_attrs(...)`. The
issue is explicitly a sketch and asks segmentation-tool authors what they
need. Ethograph is such a tool, and already had two ad-hoc conventions doing
part of the job: `attrs["type"] = "changepoints"` (read in a dozen places)
and `attrs["normalise"]` (invented for the segmentation pipeline's
normalisation).

Adopting a shared vocabulary is only worth it if it removes mechanisms
rather than adding a layer over them. Two risks had to be settled:

1. A dataset that predates the convention must not degrade. Most existing
   files carry no `kind` at all.
2. A category is a tempting place to hang behaviour ("kinematic features get
   z-scored"), and that is wrong: `speed` and `heading` are both kinematic,
   but z-scoring a unit vector destroys it.

## Decision

Adopt the schema in `ethograph/io/schema.py` with three rules.

**Advisory.** Nothing validates `kind` and no code path requires it. Feature
detection stays "any variable with a time dim"; plot-type gating stays
shape-based. `kind` only refines: it groups a feature list, and it names a
group to drop in an ablation (`train.drop_kinds`).

**A label, never a switch.** No arithmetic is chosen by `kind`. Behaviour
that depends on a variable's nature reads a separate *behavioural* attr —
today only `normalise`, read where normalisation happens.

**Both spellings are read; both are written for changepoints.** `kind_of()`
maps the legacy `type="changepoints"` onto `changepoint_feature`, and
`is_changepoint()` maps it onto the mask predicate too — every historical use
of that attr marked a raw mask, never an expansion, so the one legacy value
carries both meanings at once. Changepoint producers write both spellings
(`changepoint_attrs()` sets `kind`, `changepoint_mask` *and* `type`), so no
file needs migrating for either direction: an old file is read correctly
with no migration, and a new file is read correctly by legacy-only code.
`changepoint_mask` still exists as its own attr — it is what `kind` doing a
mask's job would have meant to avoid — so the mask predicate has one home
regardless of which spelling produced it.

`units` is deliberately **not** adopted: we wrote it in three places and read
it nowhere, and the one plausible consumer (`plots_radial.angular_unit`)
deliberately probes the data instead of trusting metadata.

## Consequences

* Ablation by category costs one run instead of a re-materialisation, and
  replaces the old pipeline's hardcoded `no_changepoint | no_kinematic |
  no_s3d` string literals.
* A mislabelled `kind` groups a feature oddly; it cannot corrupt training.
  That is the property that makes the attr cheap to adopt incrementally.
* Two attrs mean two things to set. `describe()` is the single place that
  stamps them, so the pairing stays consistent.
* Flags are stored as `0`/`1`: NetCDF has no boolean attribute type and
  refuses to save one.
* We owe #978 feedback: that `kind` must stay advisory for consumers, that a
  category cannot carry normalisation semantics, and that what a consumer
  actually needs in order to build a model input is *what pins a column*
  (dims plus coord values), which the sketch does not cover.
