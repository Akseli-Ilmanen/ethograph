"""The skeleton as a graph: named keypoints and the undirected edges between them.

One value type for every source a skeleton can come from, so a consumer reads
one shape and never asks where it came from:

* an ndx-pose ``Skeleton`` inside an NWB file — ``nodes`` (names) and
  ``edges`` (0-based index pairs into ``nodes``), the de-facto interchange
  standard (:meth:`Skeleton.from_nwb`);
* this project's own YAML skeleton config — ``connections: [{start, end,
  color, width}, ...]``, the file the GUI's skeleton editor writes
  (:meth:`Skeleton.from_config`);
* the plain ``{nodes, edges}`` mapping :meth:`Skeleton.to_dict` writes, which
  is how a materialised dataset records the skeleton it was built with.

Edges are undirected pairs of *names*, exactly ndx-pose's model: no direction,
no colours, no symmetry. Everything richer stays in the GUI's config layer.
Rooting (which keypoint is the trunk, which pair spans the body) is not a
property of the skeleton and is asked for separately (:meth:`Skeleton.parents`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

NWB_SUFFIX = ".nwb"

#: ndx-pose's type names, as written in each group's ``neurodata_type`` attribute.
NDX_SKELETON = "Skeleton"
NDX_POSE_ESTIMATION = "PoseEstimation"


@dataclass(frozen=True)
class Skeleton:
    """Named keypoints and the undirected edges between them."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        nodes = tuple(str(n) for n in self.nodes)
        if len(set(nodes)) != len(nodes):
            dupes = sorted({n for n in nodes if nodes.count(n) > 1})
            raise ValueError(f"Skeleton nodes are not unique: {dupes}")
        edges = tuple((str(a), str(b)) for a, b in self.edges)
        known = set(nodes)
        for a, b in edges:
            if a == b:
                raise ValueError(f"Skeleton edge {a!r}-{b!r} joins a node to itself")
            if a not in known or b not in known:
                raise ValueError(
                    f"Skeleton edge {a!r}-{b!r} names a node the skeleton does not declare ({sorted(known)})"
                )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skeleton:
        """From the ``{nodes: [...], edges: [[a, b], ...]}`` mapping :meth:`to_dict` writes."""
        return cls(tuple(data["nodes"]), tuple((a, b) for a, b in data.get("edges") or []))

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": list(self.nodes), "edges": [list(e) for e in self.edges]}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Skeleton:
        """From this project's YAML skeleton config (``connections`` of ``start``/``end``).

        ``keypoints``, when the config lists them, is the node order; otherwise
        the nodes are the connection endpoints in first-appearance order.
        """
        connections = config.get("connections")
        if not isinstance(connections, list):
            raise ValueError("A skeleton config needs a 'connections' list of {start, end} entries")
        edges = tuple((str(c["start"]), str(c["end"])) for c in connections)
        listed = config.get("keypoints")
        if listed:
            nodes: tuple[str, ...] = tuple(str(n) for n in listed)
        else:
            nodes = _ordered_unique(n for edge in edges for n in edge)
        return cls(nodes, edges)

    def to_config(self) -> dict[str, Any]:
        """The GUI's skeleton config dict (colours from the default palette)."""
        from ethograph.skeleton.config import nwb_skeleton_to_config

        index = {n: i for i, n in enumerate(self.nodes)}
        return nwb_skeleton_to_config(list(self.nodes), np.array([[index[a], index[b]] for a, b in self.edges]))

    @classmethod
    def from_nodes_edges(cls, nodes: Sequence[str], edges: np.ndarray | Sequence[Sequence[int]]) -> Skeleton:
        """From ndx-pose's shape: names plus ``(n_edges, 2)`` index pairs into them."""
        names = [str(n) for n in nodes]
        pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if pairs.size and (pairs.min() < 0 or pairs.max() >= len(names)):
            raise ValueError(f"Skeleton edges index outside its {len(names)} nodes: {pairs.tolist()}")
        return cls(tuple(names), tuple((names[a], names[b]) for a, b in pairs))

    @classmethod
    def from_nwb(cls, path: str | Path) -> Skeleton:
        """The one ndx-pose ``Skeleton`` in an NWB file (0.2+ ``Skeleton`` groups, or
        0.1.x ``nodes``/``edges`` attributes on ``PoseEstimation``).

        Several groups holding the *same* skeleton are one skeleton; several
        different ones are an error, since nothing says which is meant.
        """
        found = read_nwb_skeletons(path)
        if not found:
            raise ValueError(f"{path} holds no ndx-pose Skeleton (and no PoseEstimation with nodes/edges attributes)")
        distinct: dict[Skeleton, list[str]] = {}
        for name, skeleton in found.items():
            distinct.setdefault(skeleton, []).append(name)
        if len(distinct) > 1:
            listing = "; ".join(
                f"{sorted(groups)}: {len(s.nodes)} nodes, {len(s.edges)} edges" for s, groups in distinct.items()
            )
            raise ValueError(
                f"{path} holds {len(distinct)} different skeletons — {listing}. Export the one you mean to a YAML."
            )
        return next(iter(distinct))

    @classmethod
    def load(cls, path: str | Path) -> Skeleton:
        """From a path: an ``.nwb`` (ndx-pose) or a YAML (skeleton config or ``{nodes, edges}``)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Skeleton file not found: {path}")
        if path.suffix.lower() == NWB_SUFFIX:
            return cls.from_nwb(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a skeleton YAML is a mapping with 'connections' or 'nodes'/'edges'")
        if "connections" in data:
            return cls.from_config(data)
        if "nodes" in data:
            return cls.from_dict(data)
        raise ValueError(f"{path}: neither a skeleton config ('connections') nor a {{nodes, edges}} mapping")

    # -- graph queries -----------------------------------------------------

    def restricted(self, keypoints: Sequence[str]) -> Skeleton:
        """This skeleton on *keypoints* only, in that order; edges touching any other node are dropped."""
        keep = [str(k) for k in keypoints]
        known = set(keep)
        return Skeleton(tuple(keep), tuple(e for e in self.edges if e[0] in known and e[1] in known))

    def _order(self, order: Sequence[str] | None) -> list[str]:
        names = list(self.nodes) if order is None else [str(n) for n in order]
        unknown = [n for n in names if n not in self.nodes]
        if unknown:
            raise ValueError(f"Keypoints {unknown} are not nodes of the skeleton ({list(self.nodes)})")
        return names

    def adjacency(self, order: Sequence[str] | None = None) -> np.ndarray:
        """Symmetric binary ``(V, V)`` adjacency over *order* (default: node order)."""
        names = self._order(order)
        index = {n: i for i, n in enumerate(names)}
        a = np.zeros((len(names), len(names)), dtype=np.float32)
        for u, v in self.edges:
            if u in index and v in index:
                a[index[u], index[v]] = a[index[v], index[u]] = 1.0
        return a

    def hop_distances(self, order: Sequence[str] | None = None) -> np.ndarray:
        """Shortest-path length in edges between every pair, ``-1`` where unreachable."""
        names = self._order(order)
        index = {n: i for i, n in enumerate(names)}
        neighbours: dict[int, list[int]] = {i: [] for i in range(len(names))}
        for u, v in self.edges:
            if u in index and v in index:
                neighbours[index[u]].append(index[v])
                neighbours[index[v]].append(index[u])
        dist = np.full((len(names), len(names)), -1, dtype=np.int64)
        for start in range(len(names)):
            dist[start, start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nxt in neighbours[node]:
                    if dist[start, nxt] < 0:
                        dist[start, nxt] = dist[start, node] + 1
                        queue.append(nxt)
        return dist

    def parents(self, root: str, order: Sequence[str] | None = None) -> dict[int, int]:
        """The tree hanging from *root*: ``child index → parent index`` over *order*.

        Breadth-first from the root, so every node's parent is its neighbour
        nearest the root. Every node must be reachable — a keypoint the edges do
        not connect to the root has no parent and no joint angle.
        """
        names = self._order(order)
        if root not in names:
            raise ValueError(f"root {root!r} is not one of the keypoints {names}")
        dist = self.hop_distances(names)
        r = names.index(root)
        unreachable = [names[i] for i in range(len(names)) if dist[r, i] < 0]
        if unreachable:
            raise ValueError(f"Keypoints {unreachable} are not connected to root {root!r} by the skeleton's edges")
        index = {n: i for i, n in enumerate(names)}
        adjacent = {i: set() for i in range(len(names))}
        for u, v in self.edges:
            if u in index and v in index:
                adjacent[index[u]].add(index[v])
                adjacent[index[v]].add(index[u])
        parents: dict[int, int] = {}
        for child in range(len(names)):
            if child == r:
                continue
            # the neighbour one hop closer to the root (ties: first in order)
            parents[child] = min(n for n in adjacent[child] if dist[r, n] == dist[r, child] - 1)
        return parents


def _ordered_unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return tuple(seen)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_nwb_skeletons(path: str | Path) -> dict[str, Skeleton]:
    """Every ndx-pose skeleton in an NWB file, keyed by its HDF5 path.

    Read with h5py so no ndx-pose install is needed to *read* a file someone
    else wrote with it. Finds ``neurodata_type == "Skeleton"`` groups (ndx-pose
    0.2+) and, for 0.1.x files, ``PoseEstimation`` groups carrying
    ``nodes``/``edges`` attributes.
    """
    import h5py

    found: dict[str, Skeleton] = {}

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Group):
            return
        kind = _text(obj.attrs.get("neurodata_type", ""))
        if kind == NDX_SKELETON and "nodes" in obj:
            nodes = [_text(n) for n in np.asarray(obj["nodes"][()]).ravel()]
            edges = np.asarray(obj["edges"][()]) if "edges" in obj else np.empty((0, 2), dtype=np.int64)
            found[name] = Skeleton.from_nodes_edges(nodes, edges)
        elif kind == NDX_POSE_ESTIMATION and "nodes" in obj.attrs:
            nodes = [_text(n) for n in np.asarray(obj.attrs["nodes"]).ravel()]
            edges = np.asarray(obj.attrs["edges"]) if "edges" in obj.attrs else np.empty((0, 2), dtype=np.int64)
            found[name] = Skeleton.from_nodes_edges(nodes, edges)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return found
