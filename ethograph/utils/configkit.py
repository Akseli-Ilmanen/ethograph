"""A chained YAML file becomes a typed dataclass tree, and back.

The machinery under every scripted pipeline's config (``ethograph.segment``,
``ethograph.spot``), and nothing about either of them:

* ``base: other.yaml`` — a file is deep-merged *over* ``other.yaml``
  (relative to itself).
* dotted ``key=value`` overrides, values parsed as YAML.
* a :class:`Schema` naming which fields are nested dataclasses, paths or
  number pairs, so a mapping builds into the pipeline's own types. Unknown
  keys are an error: a typo must not silently become a default.

A pipeline's config module holds its dataclasses, its :class:`Schema` and its
validation; reading, building and dumping live here.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

#: ``(value, where, base_dir) -> built value`` — a field whose YAML spelling
#: no table describes (``sessions``: a path or a mapping per entry).
Converter = Callable[[Any, str, Path], Any]


@dataclass(frozen=True)
class Schema:
    """How one pipeline's YAML field names build into its dataclass tree.

    Keyed by field *name*, so a name means one thing throughout a tree —
    which is why each pipeline has its own: ``train``, ``model`` and
    ``labels`` are different dataclasses in each.
    """

    #: Field name -> the dataclass its mapping builds.
    nested: Mapping[str, type] = field(default_factory=dict)
    #: Fields resolved against the config file's folder.
    paths: frozenset[str] = frozenset()
    #: Lists of such paths.
    path_lists: frozenset[str] = frozenset()
    #: ``{name: path}`` mappings of such paths.
    path_maps: frozenset[str] = frozenset()
    #: ``(float, float)`` pairs.
    pairs: frozenset[str] = frozenset()
    converters: Mapping[str, Converter] = field(default_factory=dict)
    #: Removed settings, as the dotted path :func:`build` sees -> why. A key
    #: here is dropped with a log line instead of being refused as unknown,
    #: so a saved run config written before the removal still loads.
    retired: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# dict -> dataclass tree
# ---------------------------------------------------------------------------


def build(cls: type[ConfigT], data: Any, where: str, base_dir: Path, schema: Schema) -> ConfigT:
    """Build dataclass *cls* from *data*, failing on unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    retired = [name for name in data if f"{where}.{name}" in schema.retired]
    if retired:
        data = {k: v for k, v in data.items() if k not in retired}
        for name in retired:
            logger.info("%s.%s is a retired setting, ignored: %s", where, name, schema.retired[f"{where}.{name}"])
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; valid keys: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        if value is None and known[name].default_factory is not MISSING:
            # `params:` with nothing after it — or only a comment — is YAML
            # null, and for a field whose default is built by a factory (every
            # dict, list and nested config here) that plainly means "leave it
            # at the default". Passing the None on would hand a `None` to code
            # expecting a mapping, far from the line that wrote it.
            continue
        kwargs[name] = _convert(name, known[name].type, value, f"{where}.{name}", base_dir, schema)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _convert(name: str, annotation: Any, value: Any, where: str, base_dir: Path, schema: Schema) -> Any:
    if value is None:
        return None
    if name in schema.nested:
        return build(schema.nested[name], value, where, base_dir, schema)
    if name in schema.converters:
        return schema.converters[name](value, where, base_dir)
    if name in schema.paths:
        return resolve_path(value, base_dir)
    if name in schema.path_maps:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping of name -> path, got {type(value).__name__}")
        return {str(k): resolve_path(v, base_dir) for k, v in value.items()}
    if name in schema.path_lists:
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a list of paths, got {type(value).__name__}")
        return [resolve_path(v, base_dir) for v in value]
    if name in schema.pairs:
        if len(value) != 2:
            raise ValueError(f"{where}: expected two numbers, got {value!r}")
        return (float(value[0]), float(value[1]))
    cast = _numeric_cast(annotation)
    if cast is not None and isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return _as_number(cast, value, where)
    return value


_NUMERIC: dict[str, type] = {"float": float, "int": int}


def _numeric_cast(annotation: Any) -> type | None:
    """The coercion a ``float``/``int`` (optionally ``| None``) field needs, else ``None``.

    Not cosmetic: YAML 1.1 reads ``learning_rate: 1e-4`` as the *string*
    ``"1e-4"`` — no dot, no sign — so an unconverted value would reach the
    optimizer as text. Covered by ``tests/test_unit/test_segment_pipeline.py``.
    """
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    return _NUMERIC.get(text.replace(" ", "").removesuffix("|None"))


def _as_number(cast: type, value: Any, where: str) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: expected a number, got {value!r}") from exc
    if cast is int and not float(number).is_integer():
        raise ValueError(f"{where}: expected a whole number, got {value!r}")
    return cast(number)


def resolve_path(value: Any, base_dir: Path) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


# ---------------------------------------------------------------------------
# YAML in
# ---------------------------------------------------------------------------


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge *over* onto *base*, returning a new dict."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml_chain(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    base_ref = raw.pop("base", None)
    if base_ref is None:
        return raw
    base_path = resolve_path(base_ref, path.parent)
    if not base_path.is_file():
        raise FileNotFoundError(f"{path}: base config {base_path} does not exist")
    return deep_merge(read_yaml_chain(base_path), raw)


def as_overrides(params: dict[str, Any]) -> list[str]:
    """``{"train.epochs": 40}`` → ``["train.epochs=40"]``, the spelling :func:`apply_overrides` takes.

    Values go through YAML, so a dict, a list, a path or a bool round-trips
    exactly as the file would have spelled it — which is what a script
    building overrides programmatically wants, rather than ``str()`` and its
    Python-repr quoting.
    """
    return [f"{key}={_dump_value(value)}" for key, value in params.items()]


def _dump_value(value: Any) -> str:
    """*value* as a one-line YAML scalar/flow collection.

    PyYAML ends a scalar *document* with a ``...`` marker on its own line
    (``1e-05`` dumps as ``"1.0e-05\\n...\\n"``), which would travel into the
    override string and out again into anything that prints or reuses it.
    """
    text = yaml.safe_dump(value, default_flow_style=True).strip()
    lines = [line for line in text.splitlines() if line.strip() != "..."]
    return " ".join(lines)


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` dotlist overrides (values parsed as YAML)."""
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} is not of the form key.path=value")
        key, _, raw_value = item.partition("=")
        value = yaml.safe_load(raw_value) if raw_value != "" else None
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            if node.get(part) is None:
                # `params:` with nothing after it is YAML null — "the default",
                # exactly as build() reads it — so an override may descend into it.
                node[part] = {}
            node = node[part]
            if not isinstance(node, dict):
                raise ValueError(f"Override {item!r}: {part!r} is not a mapping")
        node[parts[-1]] = value
    return out


def read_config(path: str | Path, overrides: list[str] | None = None) -> tuple[dict, Path]:
    """A config file's data (following ``base:``, overrides applied) and its resolved path."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    data = read_yaml_chain(path)
    if overrides:
        data = apply_overrides(data, list(overrides))
    return data, path


# ---------------------------------------------------------------------------
# dataclass tree -> YAML
# ---------------------------------------------------------------------------


def to_plain(obj: Any, skip: frozenset[str] = frozenset()) -> Any:
    """*obj* as plain YAML-able data; dataclass fields named in *skip* are left out."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name), skip) for f in fields(obj) if f.name not in skip}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v, skip) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v, skip) for v in obj]
    return obj


def write_yaml(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def replaced(cfg: ConfigT, **changes: Any) -> ConfigT:
    """An independent copy of *cfg* with top-level fields replaced."""
    return replace(copy.deepcopy(cfg), **changes)  # type: ignore[type-var]
