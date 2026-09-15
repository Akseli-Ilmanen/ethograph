"""What a data variable *is* — the attrs convention for features.

Follows the schema sketched in movement's
`issue #978 <https://github.com/neuroinformatics-unit/movement/issues/978>`_:
a feature is an ordinary ``DataArray`` beside ``position``/``confidence``,
described by its ``attrs`` and selected idiomatically::

    ds["speed"].attrs["kind"] = "kinematic_feature"
    ds.filter_by_attrs(kind="kinematic_feature")

Three rules keep this cheap:

* **Advisory, never required.** Nothing validates it and nothing depends on
  it to work. A dataset with no ``kind`` anywhere behaves exactly as it did
  before — features are still whatever has a time dim. ``kind`` only
  *refines*: it groups the feature list, and it names a group to drop in an
  ablation.
* **A label, not a switch.** No arithmetic is ever chosen by ``kind``. A
  category says what a thing is; it cannot say how to normalise it —
  ``speed`` and ``heading`` are both kinematic features, but z-scoring a
  unit vector is wrong. Anything that changes maths reads a *behavioural*
  attr instead (:data:`NORMALISE`).
* **Two spellings for changepoints.** ``attrs["type"] = "changepoints"``, the
  convention this replaces, is still read as a synonym for
  ``kind="changepoint_feature"`` plus :data:`CHANGEPOINT_MASK` — a dataset
  built before this module needs no migration. Changepoint producers write
  both spellings (:func:`changepoint_attrs`), so a reader that only knows the
  legacy convention keeps working. :func:`migrate_legacy_attrs` still exists
  to normalise an old file's attrs and to drop other stale ``type`` values
  (``"pca"``, ``"audio_changepoints"``, ``"features"``) that nothing reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import xarray as xr
import yaml

#: Attr naming the category a variable belongs to (movement #978).
KIND = "kind"

#: Attr: is the variable expressed in an animal-centred frame? (movement #978)
IS_EGOCENTRIC = "is_egocentric"

#: Attr: should this variable be z-scored? Unit vectors, angles, binary
#: flags and normalised ids carry a meaning their spread does not, so they
#: are written ``normalise=0``. This is *behavioural* — it is read where
#: normalisation happens, and is deliberately not derivable from :data:`KIND`.
NORMALISE = "normalise"

#: Attr: is this variable a raw changepoint **mask** — a binary 0/1 marker?
#: Behavioural, like :data:`NORMALISE`: it is what the changepoint machinery
#: acts on. A mask's smooth expansions share its :data:`KIND` but are ordinary
#: model inputs, so the two questions need two attrs (see :func:`is_changepoint`).
CHANGEPOINT_MASK = "changepoint_mask"

#: The attr this convention replaced. Written alongside the new spelling by
#: :func:`changepoint_attrs`, and read live as a synonym by :func:`kind_of` /
#: :func:`is_changepoint` — a file needs no migration for changepoints to be
#: recognised. :func:`migrate_legacy_attrs` still normalises it away and
#: drops stale non-changepoint ``type`` values.
_LEGACY_TYPE = "type"
_LEGACY_CHANGEPOINTS = "changepoints"

#: Every attr this module owns; :func:`clear` strips exactly these.
MANAGED_ATTRS: tuple[str, ...] = (KIND, IS_EGOCENTRIC, NORMALISE, CHANGEPOINT_MASK, _LEGACY_TYPE)

KINEMATIC_FEATURE = "kinematic_feature"
VIDEO_FEATURE = "video_feature"
CHANGEPOINT_FEATURE = "changepoint_feature"
NEURAL_FEATURE = "neural_feature"
#: Existing labels rendered as input columns (:mod:`ethograph.features.label_inputs`).
LABEL_INPUT = "label_input"

#: The kinds this project writes. Any other string is allowed — a third
#: party's kind must not be an error — but these are the ones our own code
#: groups and offers.
KNOWN_KINDS: tuple[str, ...] = (KINEMATIC_FEATURE, VIDEO_FEATURE, CHANGEPOINT_FEATURE, NEURAL_FEATURE, LABEL_INPUT)


def describe(
    da: xr.DataArray,
    kind: str,
    *,
    is_egocentric: bool | None = None,
    normalise: bool | None = None,
    **extra: Any,
) -> xr.DataArray:
    """Stamp the schema attrs onto *da* (mutating it) and return it.

    Only what is given is written: a feature that is neither egocentric nor
    exempt from normalisation carries no such attr, so absence keeps meaning
    "unknown" and a reader's default stays the honest one (normalise).

    Flags are written as ``0``/``1``, not ``True``/``False``: NetCDF has no
    boolean attribute type and refuses to save one. Covered by
    ``tests/test_unit/test_segment_pipeline.py`` (which round-trips a
    described dataset through ``.nc``).
    """
    da.attrs[KIND] = str(kind)
    if is_egocentric is not None:
        da.attrs[IS_EGOCENTRIC] = int(bool(is_egocentric))
    if normalise is False:
        da.attrs[NORMALISE] = 0
    da.attrs.update(extra)
    return da


def attrs_of(var: Any) -> Mapping[str, Any]:
    """*var*'s schema attrs — it may be a ``DataArray`` or a plain mapping.

    pynapple has no per-object attrs (a ``Tsd`` has nowhere to put them), so
    its schema comes from a sidecar and arrives here as a dict. Every reader
    below goes through this, and so works for both.
    """
    if isinstance(var, Mapping):
        return var
    return getattr(var, "attrs", None) or {}


def kind_of(var: xr.DataArray | xr.Dataset | Mapping[str, Any] | Any) -> str | None:
    """The kind of *var*, or ``None`` when it does not say.

    ``attrs["type"] = "changepoints"`` is read as a synonym for
    ``kind="changepoint_feature"`` — see the module docstring.
    """
    attrs = attrs_of(var)
    kind = attrs.get(KIND)
    if kind:
        return str(kind)
    if attrs.get(_LEGACY_TYPE) == _LEGACY_CHANGEPOINTS:
        return CHANGEPOINT_FEATURE
    return None


def is_kind(var: Any, *kinds: str) -> bool:
    """Whether *var* declares one of *kinds*."""
    return kind_of(var) in kinds


def is_changepoint(var: Any) -> bool:
    """Whether *var* is a raw changepoint **mask** — a binary 0/1 marker.

    This is the *predicate* the changepoint machinery reads: what
    ``merge_changepoints`` ORs together, what ``validate_changepoints``
    range-checks, and what the catalog hides from the feature list. It is
    deliberately **not** :data:`KIND`: the smooth expansions of a mask
    (``*_cp_sigma3``, ``*_cp_since``, …) are ordinary model inputs that
    share the ``changepoint_feature`` *label* but are not masks, and treating
    them as masks would OR float curves into an all-True mask and hide them
    from the GUI. Giving the marker its own attr is what lets :data:`KIND`
    stay a pure label (see this module's docstring).

    The legacy ``attrs["type"] = "changepoints"`` is read as a synonym: every
    historical use of that attr marked a raw mask, never an expansion, so it
    carries the same meaning as :data:`CHANGEPOINT_MASK` without migration.
    """
    attrs = attrs_of(var)
    return bool(attrs.get(CHANGEPOINT_MASK)) or attrs.get(_LEGACY_TYPE) == _LEGACY_CHANGEPOINTS


def changepoint_attrs(**extra: Any) -> dict[str, Any]:
    """Attrs marking a raw changepoint mask: the family's label, the marker, and the legacy spelling.

    The legacy ``type="changepoints"`` is written alongside the new attrs so
    a reader that only knows that convention still recognises the mask.
    """
    return {KIND: CHANGEPOINT_FEATURE, CHANGEPOINT_MASK: 1, _LEGACY_TYPE: _LEGACY_CHANGEPOINTS, **extra}


def clear(attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop every attr this module owns from *attrs* (mutating it) and return it.

    For a derived variable that inherits its source's attrs: the schema of the
    output is the output's own, never the input's.
    """
    for name in MANAGED_ATTRS:
        attrs.pop(name, None)
    return attrs


#: A session's schema sidecar, beside its alignment.
SIDECAR_NAME = "schema.yaml"


def sidecar_path(source: str | Path) -> Path:
    """Where *source*'s schema sidecar lives: ``{session}/.ethograph/schema.yaml``.

    The same folder the alignment NWB sits in, so one hidden directory holds
    everything about a session that is not the data itself.
    """
    p = Path(source)
    return (p if p.is_dir() else p.parent) / ".ethograph" / SIDECAR_NAME


def read_sidecar(source: str | Path) -> dict[str, dict[str, Any]]:
    """``{variable: attrs}`` declared beside *source*; empty when there is none.

    This is how a **pynapple** session says what its variables are: a `Tsd`
    has no attrs to write on, so the schema lives in a file instead. An
    xarray session may use one too, but its variables' own attrs win.
    """
    path = sidecar_path(source)
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of variable name -> attrs, got {type(raw).__name__}")
    out: dict[str, dict[str, Any]] = {}
    for name, attrs in raw.items():
        if not isinstance(attrs, dict):
            raise ValueError(f"{path}: {name!r} must map to a mapping of attrs, got {type(attrs).__name__}")
        out[str(name)] = {k: int(v) if isinstance(v, bool) else v for k, v in attrs.items()}
    return out


def write_sidecar(source: str | Path, variables: Mapping[str, Mapping[str, Any]]) -> Path:
    """Write ``{variable: attrs}`` beside *source*, creating ``.ethograph/``."""
    path = sidecar_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = {
        str(name): {k: int(v) if isinstance(v, bool) else v for k, v in attrs.items()}
        for name, attrs in variables.items()
    }
    path.write_text(yaml.safe_dump(plain, sort_keys=False), encoding="utf-8")
    return path


def migrate_legacy_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Convert a dataset written before this convention, in place.

    ``attrs["type"] = "changepoints"`` is already read live as a synonym for
    ``kind="changepoint_feature"`` (see :func:`kind_of` / :func:`is_changepoint`),
    so this is no longer required for a changepoint mask to be recognised —
    it exists to normalise old files onto the full :func:`changepoint_attrs`
    stamp, and to drop other stale ``type`` values (``"pca"``,
    ``"audio_changepoints"``, ``"features"``) that nothing ever read.
    Everything else is untouched — a variable that carries no ``type`` gets
    no ``kind`` invented for it.

    Returns *ds* so it can be chained onto a load::

        dt.update_trial(trial, migrate_legacy_attrs)
    """
    for var in ds.data_vars.values():
        stale = var.attrs.pop(_LEGACY_TYPE, None)
        if stale == _LEGACY_CHANGEPOINTS:
            var.attrs.update(changepoint_attrs())
    return ds


def changepoint_metadata(n_units: int, **extra: Any) -> dict[str, list[Any]]:
    """The pynapple counterpart of :func:`changepoint_attrs`.

    A ``TsGroup`` describes its units with metadata *columns* rather than
    attrs, so the same two names become two columns carrying one value per
    unit. The vocabulary is deliberately identical to the xarray side —
    there is one way to say "this is a changepoint mask", whatever the
    backend.
    """
    return {name: [value] * n_units for name, value in changepoint_attrs(**extra).items()}


def changepoint_units(meta: Any) -> list[Any]:
    """Row labels of the changepoint units in a ``TsGroup``'s metadata table.

    Empty when the table has no such column — the counterpart of
    :func:`is_changepoint` for the pynapple backend.
    """
    columns = getattr(meta, "columns", None)
    if meta is None or columns is None or CHANGEPOINT_MASK not in columns:
        return []
    return list(meta.index[meta[CHANGEPOINT_MASK].astype(bool)])


def changepoint_vars(ds: xr.Dataset) -> list[str]:
    """Names of *ds*'s raw changepoint masks, in dataset order."""
    return [str(name) for name, var in ds.data_vars.items() if is_changepoint(var)]


def filter_changepoints(ds: xr.Dataset) -> xr.Dataset:
    """*ds* reduced to its raw changepoint masks (its smooth expansions are not masks)."""
    return ds[changepoint_vars(ds)]


def is_normalise(var: Any) -> bool:
    """Whether *var* should be z-scored; ``True`` unless it says otherwise."""
    return bool(attrs_of(var).get(NORMALISE, True))


def is_egocentric(var: Any) -> bool | None:
    """Whether *var* is in an animal-centred frame; ``None`` when it does not say."""
    attrs = attrs_of(var)
    if IS_EGOCENTRIC not in attrs:
        return None
    return bool(attrs[IS_EGOCENTRIC])


def kinds_in(ds: xr.Dataset) -> dict[str, list[str]]:
    """``{kind: [variable names]}`` for every variable that declares one.

    Variables with no ``kind`` are absent — the caller decides what to do
    with them, which is always "treat as before".
    """
    out: dict[str, list[str]] = {}
    for name, var in ds.data_vars.items():
        kind = kind_of(var)
        if kind is not None:
            out.setdefault(kind, []).append(str(name))
    return out


def select_kinds(ds: xr.Dataset, kinds: Iterable[str]) -> list[str]:
    """Names of the variables whose kind is in *kinds*, in dataset order."""
    wanted = set(kinds)
    return [str(name) for name, var in ds.data_vars.items() if kind_of(var) in wanted]


def drop_kinds(names: Iterable[str], ds: xr.Dataset, kinds: Iterable[str]) -> list[str]:
    """*names* minus every variable whose kind is in *kinds*.

    The ablation primitive: "train without the video features" is
    ``drop_kinds(features, ds, ["video_feature"])``. A name the dataset does
    not carry, or one with no kind, is kept — dropping is only ever done on
    a positive declaration.
    """
    unwanted = set(kinds)
    return [n for n in names if not (n in ds.data_vars and kind_of(ds[n]) in unwanted)]
