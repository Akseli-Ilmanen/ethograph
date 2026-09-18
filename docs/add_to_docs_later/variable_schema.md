(target-variable-schema)=
# Variable schema

What a data variable *is*, written on the variable itself. Ethograph follows
the schema sketched in movement's
[issue #978](https://github.com/neuroinformatics-unit/movement/issues/978):
a feature is an ordinary `DataArray` beside `position`/`confidence`, described
by its `attrs` and selected idiomatically.

```python
import ethograph as eto
from ethograph.io.schema import KINEMATIC_FEATURE, describe

describe(ds["speed"], KINEMATIC_FEATURE, is_egocentric=False)
ds.filter_by_attrs(kind="kinematic_feature")
```

## The attrs

| Attr | Values | Meaning |
|---|---|---|
| `kind` | `kinematic_feature`, `video_feature`, `changepoint_feature`, `neural_feature` — or any string | The category the variable belongs to. A **label**, for grouping and ablation. |
| `is_egocentric` | `0` / `1` | Is it expressed in an animal-centred frame? |
| `normalise` | `0` (or absent, meaning `1`) | Should this variable be z-scored? Binaries and other non-scalable columns are written `0`. |
| `changepoint_mask` | `1` (or absent) | Is this a raw binary 0/1 changepoint marker? Read by `is_changepoint()` — the changepoint machinery acts on exactly these. |

```{note}
Flags are stored as `0`/`1`, not `True`/`False`. NetCDF has no boolean
attribute type and refuses to save one — `describe()` converts for you.
```

## Three rules

**Advisory, never required.** Nothing validates `kind` and nothing needs it to
work. A dataset with no `kind` anywhere is still perfectly usable: features
are still "any variable with a time dim", every plot still works, training
still works. `kind` only *refines* — it groups the feature list and names a
group to drop in an ablation. If your files predate this convention, you can
add it whenever you like, or never.

The one exception is **changepoints**, which had a convention before this one;
see the note below.

**A label, not a switch.** No arithmetic is ever chosen by `kind`. A category
says what a thing is; it cannot say how to treat it. `speed` and `heading` are
both `kinematic_feature`, but z-scoring a unit vector is wrong — so
normalisation reads `normalise`, not `kind`. This is why the two attrs exist
separately, and it is the main thing worth saying back to movement's proposal.

**Two spellings for changepoints.** `attrs["type"] = "changepoints"`, the
ad-hoc convention this replaces, is still read as a synonym for the pair of
attrs below — a file that predates this schema needs no migration for its
changepoints to be recognised. Changepoint producers write both spellings, so
code that only knows the old convention keeps working too.

```{admonition} Changepoints: the label covers the family, the marker means "mask"
:class: note

A raw changepoint variable is a binary 0/1 **mask**, and things are done to
masks: they get OR-ed together, range-checked, and hidden from the feature
list. Its smooth expansions (`*_cp_sigma3`, `*_cp_segment_id`, …) are
ordinary model inputs that must *not* be treated that way.

So the two jobs use two attrs. `kind="changepoint_feature"` labels the whole
family — mask and expansions alike — which is what makes
`drop_kinds=[changepoint_feature]` remove every changepoint column at once.
`changepoint_mask=1` marks only the masks, and is what `is_changepoint()`
reads — alongside the legacy `type="changepoints"`, since every historical
use of that attr marked a mask and never an expansion.
{func}`~ethograph.io.schema.changepoint_attrs` writes `kind`, `changepoint_mask`
*and* the legacy `type`, so a mask you make yourself is stamped the same way
ethograph's own are, and is recognisable to old and new readers alike.
```

## Pynapple: a sidecar, because there is nowhere else

A `Tsd` has **no attrs at all**, so a pynapple session cannot describe its
variables in place. It declares them in a sidecar instead —
`{session}/.ethograph/schema.yaml`, the same hidden folder the alignment NWB
lives in:

```yaml
speed:
  kind: kinematic_feature
heading_angle:
  kind: kinematic_feature
  normalise: false
s3d:
  kind: video_feature
```

```python
from ethograph.io import schema

schema.write_sidecar(
    session_folder,
    {
        "speed": {"kind": schema.KINEMATIC_FEATURE},
        "heading_angle": {"kind": schema.KINEMATIC_FEATURE, "normalise": False},
        "s3d": {"kind": schema.VIDEO_FEATURE},
    },
)
```

Without it a pynapple session works exactly as before, but nothing is
declared: `train.drop_kinds` has nothing to drop, every column gets
z-scored, and `rank_video_features()` cannot tell which columns are video.
Materialising says so in a warning rather than doing it silently.

An xarray session may use a sidecar too — its variables' own attrs win where
both speak. Covered by `tests/test_unit/test_segment_pynapple.py`.

## Pynapple: the same names, as columns

A `TsGroup` describes its units with metadata *columns* rather than attrs, so
the vocabulary is carried over unchanged — there is one way to say "this is a
changepoint mask", whatever the backend.
{func}`~ethograph.io.schema.changepoint_metadata` is the counterpart of
`changepoint_attrs()`, and {func}`~ethograph.io.schema.changepoint_units`
the counterpart of `is_changepoint()`:

```python
from ethograph.io import schema

group.set_info(
    source_label=["nose", "tail"],
    **schema.changepoint_metadata(2, target_feature="speed"),
)
schema.changepoint_units(group.metadata)  # [0, 1]
```

## Normalising a file written before this convention

A file that predates this schema does not need converting: its
`attrs["type"] = "changepoints"` masks are already recognised — merging,
range-checking, hiding them from the feature list, the GUI's changepoint
correction all see them, with no read required first.

What that file is still missing is the *label*, `changepoint_mask` marker and
smooth-expansion grouping a freshly-produced file carries — so
{func}`~ethograph.io.schema.migrate_legacy_attrs` is worth running once to
normalise onto the full stamp, and to drop any other stale `type` value
(`"pca"`, `"audio_changepoints"`, `"features"`) that nothing ever read.
Everything else is untouched — a variable with no `type` gets no `kind`
invented for it.

```python
import ethograph as eto
from ethograph.io.schema import migrate_legacy_attrs

dt = eto.open("Trial_data.nc")
for trial in dt.trials:
    dt.update_trial(trial, migrate_legacy_attrs)
dt.save("Trial_data.nc")
```

```{note}
Nothing migrates on load, and nothing warns — nothing needs to, since the
legacy spelling is read live. A dataset built by a current version of
ethograph already writes both spellings.
```

## What it buys

**Ablations.** The whole point of naming a category is to be able to remove
it. `train.drop_kinds` leaves a category out of a run — the materialised
dataset is untouched, so an ablation costs one run, not a re-materialisation:

```python
import ethograph as eto

eto.segment.Project("project.yaml", "train.run_name=full").train()
eto.segment.Project(
    "project.yaml",
    "train.run_name=no_video",
    "train.drop_kinds=[video_feature]",
).train()

print(eto.segment.Project("project.yaml").compare())
```

Columns whose kind is undeclared are never dropped — dropping happens only on
a positive declaration.

**Grouping.** `kinds_in(ds)` gives `{kind: [names]}`, which is what makes a
feature list readable when 1024 S3D columns sit next to 30 kinematic ones.

**Selecting a subset of video features.** See {doc}`segment/video_features`:
`kind="video_feature"` is how the S3D ranking dialog knows what it may rank.

## Setting it on your own data

`describe()` is the one place that stamps, so a feature you compute yourself
looks like the ones ethograph computes:

```python
from ethograph.io.schema import KINEMATIC_FEATURE, changepoint_attrs, describe

ds["my_metric"] = (("time", "individual"), values)
describe(ds["my_metric"], KINEMATIC_FEATURE, is_egocentric=False)

# an angle or a unit vector: never z-score it
describe(ds["my_heading"], KINEMATIC_FEATURE, is_egocentric=True, normalise=False)

# a binary 0/1 changepoint mask: the family's label plus the marker
ds["my_troughs"].attrs.update(changepoint_attrs())
```

Everything ethograph produces is already described: the geometry helpers in
{mod}`ethograph.features.geometry`, the raw changepoint variables written by
{mod}`ethograph.features.changepoints` (their smooth expansions carry the
same `kind` but no `changepoint_mask` — they are ordinary model inputs), the
`movement` kinematics
added at load time, the PCA scores in {mod}`ethograph.features.neural`, and the
S3D features in {mod}`ethograph.video_features`.

## Reading it

```python
from ethograph.io import schema

schema.kind_of(ds["speed"])  # "kinematic_feature" (or None)
schema.is_changepoint(ds["troughs"])  # True — a raw binary mask
schema.is_normalise(ds["heading"])  # False
schema.is_egocentric(ds["pos_ego"])  # True / False / None when unstated
schema.kinds_in(ds)  # {"kinematic_feature": [...], ...}
schema.select_kinds(ds, ["video_feature"])
```
