"""Keypoint annotation state + export.

:class:`KeypointStore` owns every coordinate the user labels and everything a
fill backend produces. The GUI never mutates the arrays directly — it calls
``set_point`` / ``clear_point`` / ``undo`` and reads back through ``positions``.

Hierarchy
---------
The store is two-level, like SLEAP's skeleton/instance split: one keypoint
schema (the skeleton) is shared by every individual, and each individual is an
instance of that schema on a given frame. Anchors are therefore
``(n_individuals, n_keypoints, 2)`` per frame, and the active target the canvas
writes to is an ``(individual, keypoint)`` pair. Single-individual labelling is
just the ``n_individuals == 1`` case — every method takes ``individual=None``
to mean "the first (usually only) one". ``n_individuals == 0`` is legal too: it
is the state right after deleting the last individual, and nothing can be
labelled until one is added back.

Asymmetric schemas
------------------
With ``shared_keypoints = False`` each individual carries its own subset of the
keypoint schema (label the wings of the flying bird only, say). ``keypoint_names``
stays the union across individuals — the arrays remain rectangular over it, so
backends, exports and the overlay are unchanged — and :attr:`keypoint_sets` says
which pairs actually exist. Points outside an individual's set are permanently
``NaN`` and :meth:`set_point` refuses them.

Coordinate convention
---------------------
Everything here is ``(x, y)`` in **pixel coordinates of the source video**,
``NaN`` where unlabelled. Note that :func:`~ethograph.gui.pose_convert.poses_ds_to_points`
emits ``(track_id, frame, y, x)`` — the axis swap lives in this module (in
:func:`store_to_movement_ds`) and nowhere else. The store itself, the sidecar
and the canvas overlay never hold anything but source-video pixels.

Naming follows the rest of the codebase (and movement ≥0.17): the *dimension*
of a poses dataset is singular — ``keypoint``, ``individual`` — even though it
holds many of them. Plural names here are Python containers and counts only.

A frame counts as an anchor if *any* keypoint of *any* individual is labelled.
Partially labelled anchors are normal: the user labels the beak on some frames
and the tail on others, so backends must handle per-point anchor sets rather
than a single shared frame list (see :meth:`KeypointStore.anchor_frames_for`).

Provenance: observations vs inference
------------------------------------
Two *kinds* of coordinate live here, and the split is not "human vs machine":

**Observations** are positions asserted from the pixels of one specific frame —
:attr:`~KeypointStore.anchors`, which the user clicked, and
:attr:`~KeypointStore.detections`, which a marker detector read off that same
frame (see :mod:`~ethograph.gui.pose_detect`). Both are sparse ``frame -> array``
dicts, and :meth:`~KeypointStore.observations` merges them with **manual winning
per point**, which is what the fill backends consume.

**Inference** is :attr:`~KeypointStore.filled`: what a backend produced for the
frames carrying no evidence at all. A fill never feeds the next fill —
:meth:`~KeypointStore.flat_observations` reads the observations alone, so
re-filling is a pure function of what was actually observed.

Precedence is therefore **manual > detected > filled** everywhere, which is what
makes correcting a detection need no new code: clicking one calls the ordinary
:meth:`~KeypointStore.set_point`, that writes an anchor, and every reader prefers
it from then on. :meth:`~KeypointStore.human_mask` and
:meth:`~KeypointStore.detected_mask` report the provenance per point and
:meth:`~KeypointStore.is_human` / :meth:`~KeypointStore.is_detected` per
``(frame, individual)`` row — one human point anywhere in the row makes the row
human, which is the rule the dialog's ``Source`` column shows.

A predicted point becomes human by being touched: clicking one on the canvas
pins it where it was (see :meth:`~KeypointStore.promote_fill` for the same thing
over a whole frame, and :meth:`~KeypointStore.promote_detections` for detections
alone). That is how a prediction is "accepted" — there is no state between
labelled and predicted.

Detections are derived data, like the fill, and are never written to the JSON
sidecar; they are cached separately (see :func:`detections_path`) only because
re-running a detector over a long video costs minutes. What *is* user intent, and
so does live in the sidecar beside the anchors, is the :class:`AssignmentTable`:
what each detector label — tag ``7``, the orange dot — means in terms of an
``(individual, keypoint)`` pair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

#: Sidecar suffix appended to the video path to persist anchors.
SIDECAR_SUFFIX = ".keypoints.json"

#: Sidecar suffix for a cached test-time refinement (see :mod:`pose_refine`).
REFINEMENT_SUFFIX = ".posepal.pt"

#: Sidecar suffix for cached detections (see :mod:`pose_detect`).
DETECTIONS_SUFFIX = ".detections.npz"

#: Suffix for the poses dataset *Load into the GUI* writes and then loads. Output
#: rather than state: regenerated from the store on every load.
KEYPOINTS_DATASET_SUFFIX = ".keypoints.nc"

#: Share of a video worth labelling, as a percentage — roughly every 10th frame.
#:
#: A *spacing* rather than a count, because what the fill backends care about is
#: the gap they have to bridge, and a gap is measured in frames. Twenty labels
#: is dense on a 200-frame clip and nothing on an hour of footage. The figure
#: follows the ratio CoTracker3 is evaluated at for this task — Pan et al. 2025
#: report 6 annotated frames for a video of 60 (arXiv:2506.03868).
RECOMMENDED_LABEL_SHARE = 10.0

#: Name given to the first individual when the user has not named any.
DEFAULT_INDIVIDUAL = "individual_0"

#: How a pinned keypoint colour is written: ``"#rrggbb"``, lower case.
COLOR_LENGTH = 7


class KeypointStoreError(Exception):
    """Base for keypoint store failures."""


def normalise_color(spec: str) -> str:
    """Validate a keypoint colour and return it as lower-case ``"#rrggbb"``.

    One spelling is stored so a colour can be compared, round-tripped through
    the sidecar and handed to both Qt and pygfx without either of them having to
    guess at a format.
    """
    text = str(spec).strip().lower()
    if len(text) != COLOR_LENGTH or not text.startswith("#") or any(c not in "0123456789abcdef" for c in text[1:]):
        raise KeypointStoreError(f"{spec!r} is not a colour — expected '#rrggbb'.")
    return text


class UnknownKeypointError(KeypointStoreError):
    """Raised when a keypoint name is not in the store's schema."""


class UnknownIndividualError(KeypointStoreError):
    """Raised when an individual name is not in the store's schema."""


class AssignmentError(KeypointStoreError):
    """Raised when an assignment would make one point mean two things."""


#: An assignment the detector proposed by matching its output to the labels.
LEARNED = "learned"
#: An assignment the user typed or picked, which a re-learn must never touch.
MANUAL = "manual"


@dataclass(frozen=True)
class Assignment:
    """What one detector label means: a single ``(individual, keypoint)`` pair.

    A tag decodes to the integer ``7`` and a colour blob to "class 2"; neither
    knows it is the *beak*, or *bee 12*. This closes that gap, and the target
    being a **pair** is what lets one mechanism cover every marker layout:
    a colour per keypoint on one animal (``(None, "beak")``), a tag per
    individual (``("bee_07", "thorax")``), or both at once.

    ``individual=None`` means the first (usually only) individual, exactly as
    everywhere else in this module.
    """

    label: int
    individual: str | None
    keypoint: str
    source: str = LEARNED
    #: How many labelled frames agreed on this target, for the dialog's table.
    matched_frames: int = 0

    def __post_init__(self):
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "keypoint", str(self.keypoint))
        if self.individual is not None:
            object.__setattr__(self, "individual", str(self.individual))
        if self.source not in (LEARNED, MANUAL):
            raise AssignmentError(f"Unknown assignment source {self.source!r}; expected {LEARNED!r} or {MANUAL!r}")

    @property
    def target(self) -> tuple[str | None, str]:
        return self.individual, self.keypoint

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "individual": self.individual,
            "keypoint": self.keypoint,
            "source": self.source,
            "matched_frames": self.matched_frames,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Assignment:
        return cls(
            label=payload["label"],
            individual=payload.get("individual"),
            keypoint=payload["keypoint"],
            source=payload.get("source", LEARNED),
            matched_frames=int(payload.get("matched_frames", 0)),
        )


class AssignmentTable:
    """``label -> (individual, keypoint)``, keyed by the detector's own label.

    Keyed by the label rather than by the target so that renaming an individual
    in the schema does not orphan the mapping — the tag is still tag 7. Two
    labels may never share a target: one row of the point grid is one point, and
    a second label writing to it would silently overwrite the first.

    Learning is a *proposal*. :meth:`learn` never replaces an entry the user
    edited (``source == MANUAL``), which is what makes correcting one row and
    re-learning the rest a usable workflow.
    """

    def __init__(self, entries: Sequence[Assignment] = ()):
        self._by_label: dict[int, Assignment] = {}
        for entry in entries:
            self._by_label[entry.label] = entry

    # -- reading -------------------------------------------------------

    @property
    def entries(self) -> list[Assignment]:
        """Every assignment, ordered by label — the dialog's row order."""
        return [self._by_label[label] for label in sorted(self._by_label)]

    def __len__(self) -> int:
        return len(self._by_label)

    def __iter__(self):
        return iter(self.entries)

    def __contains__(self, label: object) -> bool:
        return label in self._by_label

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AssignmentTable) and self.entries == other.entries

    def __repr__(self) -> str:
        return f"AssignmentTable({self.entries!r})"

    def get(self, label: int) -> Assignment | None:
        return self._by_label.get(int(label))

    def target(self, label: int) -> tuple[str | None, str] | None:
        entry = self.get(label)
        return None if entry is None else entry.target

    def labels(self) -> list[int]:
        return sorted(self._by_label)

    def owner_of(self, individual: str | None, keypoint: str) -> int | None:
        """Which label writes to this pair, if any."""
        for entry in self.entries:
            if entry.individual == individual and entry.keypoint == keypoint:
                return entry.label
        return None

    # -- editing -------------------------------------------------------

    def set(
        self,
        label: int,
        individual: str | None,
        keypoint: str,
        source: str = MANUAL,
        matched_frames: int = 0,
    ) -> Assignment:
        """Point *label* at one pair; raises if another label already owns it."""
        entry = Assignment(label, individual, keypoint, source, matched_frames)
        owner = self.owner_of(entry.individual, entry.keypoint)
        if owner is not None and owner != entry.label:
            raise AssignmentError(
                f"{entry.individual or 'the first individual'} / {entry.keypoint} is already detected by "
                f"label {owner} — one point cannot come from two labels."
            )
        self._by_label[entry.label] = entry
        return entry

    def remove(self, label: int) -> bool:
        return self._by_label.pop(int(label), None) is not None

    def clear(self) -> None:
        self._by_label.clear()

    def learn(self, proposals: Sequence[Assignment]) -> int:
        """Merge learned proposals; returns how many were taken.

        A row the user touched is never overwritten, and a proposal whose target
        is already owned by another label is dropped rather than guessed at.
        """
        taken = 0
        for proposal in proposals:
            existing = self.get(proposal.label)
            if existing is not None and existing.source == MANUAL:
                continue
            owner = self.owner_of(proposal.individual, proposal.keypoint)
            if owner is not None and owner != proposal.label:
                continue
            self._by_label[proposal.label] = proposal
            taken += 1
        return taken

    # -- validation ----------------------------------------------------

    def invalid_labels(self, store: KeypointStore) -> set[int]:
        """Labels that cannot be written: a missing target, or a taken one.

        Deliberately not pruned: a stale row shown in red says *why* a detector
        run came back empty, where a silently dropped one does not. Its
        detections are simply never written (see :meth:`KeypointStore.assignment_rows`).

        Two labels resolving to the *same* point are caught here rather than only
        in :meth:`set`, because ``individual=None`` (the first individual) and
        that individual's own name are two spellings of one row — the equality
        check in :meth:`owner_of` cannot see that, but the schema can. The lower
        label keeps the point; the other is invalid, not silently overwriting it.
        """
        invalid: set[int] = set()
        claimed: dict[int, int] = {}
        for entry in self.entries:
            if entry.individual is not None and entry.individual not in store.individual_names:
                invalid.add(entry.label)
                continue
            if not store.individual_names or not store.has_keypoint(entry.keypoint, entry.individual):
                invalid.add(entry.label)
                continue
            row = store.individual_index(entry.individual) * store.n_keypoints + store.keypoint_index(entry.keypoint)
            if row in claimed:
                invalid.add(entry.label)
            else:
                claimed[row] = entry.label
        return invalid

    # -- persistence ---------------------------------------------------

    def to_list(self) -> list[dict]:
        return [entry.to_dict() for entry in self.entries]

    @classmethod
    def from_list(cls, payload: Sequence[dict] | None) -> AssignmentTable:
        """Rebuild from a sidecar; a missing key is simply an empty table."""
        return cls([Assignment.from_dict(item) for item in (payload or ())])


@dataclass
class KeypointStore:
    """Per-frame keypoint coordinates: user anchors + backend fill.

    Attributes
    ----------
    keypoint_names
        Keypoint schema, in display order — the union across individuals.
    n_frames
        Number of frames in the video being labelled.
    individual_names
        Individuals being labelled, in display order. May be empty.
    shared_keypoints
        ``True`` (the default) when every individual is an instance of the whole
        schema; ``False`` when each carries its own subset.
    keypoint_sets
        ``individual -> keypoints``, only meaningful (and only populated) when
        ``shared_keypoints`` is ``False``.
    keypoint_color
        ``keypoint -> "#rrggbb"`` for the keypoints whose colour the user has
        pinned. Sparse on purpose: an unlisted keypoint takes its colour from
        the generated palette, which is what makes a fresh schema legible
        without anyone choosing anything.
    individual_color
        The same, per individual: the colour markers take when the display is
        colouring by individual rather than by keypoint. Two palettes rather
        than one because the two modes are two different questions ("which
        body part is this?" / "which animal is this?"), and a pinned colour
        must survive switching between them.
    anchors
        ``frame -> (n_individuals, n_keypoints, 2)`` array of user-placed
        ``(x, y)``, ``NaN`` where that point is unlabelled on that frame.
    detections
        The same layout, read off the pixels of each frame by a detector rather
        than clicked. Sparse for the same reason: a marker is only found where
        it is visible. Manual anchors win over these point by point.
    detection_confidence
        ``frame -> (n_individuals, n_keypoints)`` detector quality in ``[0, 1]``,
        ``NaN`` where nothing was detected.
    detection_orientation
        ``frame -> (n_individuals, n_keypoints, 2)`` unit forward vector in
        **image coordinates**, ``NaN`` for a point whose marker has no
        orientation. An AprilTag is a square, so one decode fixes which way it
        faces; keeping that beside the position is what lets a head direction be
        read off a *single* tagged keypoint instead of being reconstructed from
        two keypoints that happen to sit on the same marker.
    assignment
        What each detector label means — see :class:`AssignmentTable`. Persisted
        in the sidecar next to the anchors, because it is user intent rather
        than something recomputable from the video.
    filled
        ``(n_frames, n_individuals, n_keypoints, 2)`` backend output, or
        ``None`` before the first fill. Anchor frames are copied through
        verbatim.
    confidence
        ``(n_frames, n_individuals, n_keypoints)`` in ``[0, 1]``; anchors
        are ``1.0``, and a frame the fill did not cover is ``NaN``.
    fill_range
        ``(first, last)`` frame the current fill covers — derived from
        :attr:`filled` when it is set, never assigned from outside. A fill only
        bridges the gaps between labels, so this is normally the labelled span
        and not the whole video.
    static_keypoints
        Keypoints that do not move — the corners of an arena, a fixed
        landmark. Labelled **once**, on any frame, and read as that position
        on every frame: the canvas shows them everywhere, the fill leaves them
        alone (:meth:`pin_static`), and the export writes them on every frame.
        Placing one again moves it, everywhere. Persisted in the sidecar, and
        carried to the next video of the same camera (:meth:`seed_static_from`).
    """

    keypoint_names: list[str]
    n_frames: int
    individual_names: list[str] = field(default_factory=lambda: [DEFAULT_INDIVIDUAL])
    shared_keypoints: bool = True
    keypoint_sets: dict[str, list[str]] = field(default_factory=dict)
    keypoint_color: dict[str, str] = field(default_factory=dict)
    individual_color: dict[str, str] = field(default_factory=dict)
    anchors: dict[int, np.ndarray] = field(default_factory=dict)
    detections: dict[int, np.ndarray] = field(default_factory=dict)
    detection_confidence: dict[int, np.ndarray] = field(default_factory=dict)
    detection_orientation: dict[int, np.ndarray] = field(default_factory=dict)
    assignment: AssignmentTable = field(default_factory=AssignmentTable)
    filled: np.ndarray | None = None
    confidence: np.ndarray | None = None
    fill_range: tuple[int, int] | None = None
    static_keypoints: list[str] = field(default_factory=list)
    _history: list[tuple[int, int, int, np.ndarray | None]] = field(default_factory=list, repr=False)
    #: Bumped on every change to the detections. A detector run can touch every
    #: frame of the video, so readers that would otherwise re-derive their state
    #: from the dict (the points table, which does it per mouse move of a drag)
    #: compare this integer instead.
    _detections_revision: int = field(default=0, repr=False)

    def __post_init__(self):
        self.keypoint_names = [str(n) for n in self.keypoint_names]
        self.individual_names = [str(n) for n in self.individual_names]
        self.n_frames = int(self.n_frames)
        self._normalise_keypoint_sets()
        self._normalise_colors()
        self._prune_unowned()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoint_names)

    @property
    def n_individuals(self) -> int:
        return len(self.individual_names)

    @property
    def n_points(self) -> int:
        """Rows in the flat point grid — one per ``(individual, keypoint)`` pair.

        This is the array size the fill backends see, so it counts pairs that
        an asymmetric schema leaves permanently unlabelled too; for the number
        of points the user can actually place, see :attr:`n_schema_points`.
        """
        return self.n_individuals * self.n_keypoints

    @property
    def n_schema_points(self) -> int:
        """Points the schema actually contains; ``== n_points`` when shared."""
        if self.shared_keypoints:
            return self.n_points
        return sum(len(self.keypoint_sets.get(name, ())) for name in self.individual_names)

    def keypoint_index(self, name: str) -> int:
        try:
            return self.keypoint_names.index(name)
        except ValueError:
            raise UnknownKeypointError(f"{name!r} is not in {self.keypoint_names}") from None

    def individual_index(self, name: str | None) -> int:
        """Index of *name*; ``None`` means the first (usually only) individual."""
        if not self.individual_names:
            raise UnknownIndividualError("There are no individuals — add one before labelling.")
        if name is None:
            return 0
        try:
            return self.individual_names.index(name)
        except ValueError:
            raise UnknownIndividualError(f"{name!r} is not in {self.individual_names}") from None

    def keypoints_for(self, individual: str | None = None) -> list[str]:
        """Keypoints belonging to *individual*, in schema order."""
        if self.shared_keypoints:
            return list(self.keypoint_names)
        return list(self.keypoint_sets.get(self.individual_names[self.individual_index(individual)], []))

    def has_keypoint(self, keypoint: str, individual: str | None = None) -> bool:
        """Whether *individual* carries *keypoint* under the current schema."""
        return keypoint in self.keypoints_for(individual)

    def keypoint_mask(self) -> np.ndarray:
        """``(n_individuals, n_keypoints)`` bool: which pairs exist."""
        mask = np.ones((self.n_individuals, self.n_keypoints), dtype=bool)
        if self.shared_keypoints:
            return mask
        for i, individual in enumerate(self.individual_names):
            owned = set(self.keypoint_sets.get(individual, ()))
            mask[i] = [name in owned for name in self.keypoint_names]
        return mask

    def set_keypoint_names(self, names: list[str]) -> None:
        """Replace the keypoint schema, carrying labelled points over by name."""
        self._reschema([str(n) for n in names], self.individual_names)

    def set_individual_names(self, names: list[str]) -> None:
        """Replace the individuals, carrying labelled points over by name."""
        self._reschema(self.keypoint_names, [str(n) for n in names])

    def set_shared_keypoints(self, shared: bool) -> None:
        """Switch between one shared schema and per-individual subsets.

        Neither direction touches the anchors: the arrays already span the union
        of keypoints, so turning sharing off simply gives every individual the
        whole schema to start from, and turning it back on re-admits the pairs
        that were excluded (they are ``NaN``, having never been labelled).
        """
        shared = bool(shared)
        if shared == self.shared_keypoints:
            return
        self.shared_keypoints = shared
        self.keypoint_sets = {} if shared else {name: list(self.keypoint_names) for name in self.individual_names}

    def set_keypoints_for(self, individual: str | None, names: list[str]) -> None:
        """Replace one individual's keypoints; names new to the union are added."""
        if self.shared_keypoints:
            raise KeypointStoreError(
                "Keypoints are shared by every individual — turn sharing off before editing one individual's set."
            )
        individual = self.individual_names[self.individual_index(individual)]
        names = list(dict.fromkeys(str(n) for n in names))
        union = self.keypoint_names + [name for name in names if name not in self.keypoint_names]
        if union != self.keypoint_names:
            self._reschema(union, self.individual_names)
        self.keypoint_sets[individual] = [name for name in self.keypoint_names if name in set(names)]
        self._prune_unowned()

    def set_keypoint_color(self, keypoint: str, color: str | None) -> None:
        """Pin *keypoint* to ``"#rrggbb"``, or hand it back to the palette.

        A colour is schema data, not a setting: it is keyed by keypoint name and
        travels with the anchors in the sidecar, so the beak stays the colour it
        was chosen to be when the project is reopened on another machine.
        """
        self.keypoint_index(keypoint)  # a colour for a keypoint that is not in the schema is a bug
        if color is None:
            self.keypoint_color.pop(keypoint, None)
            return
        self.keypoint_color[keypoint] = normalise_color(color)

    def color_for(self, keypoint: str) -> str | None:
        """The colour pinned to *keypoint*, or ``None`` when it uses the palette."""
        self.keypoint_index(keypoint)
        return self.keypoint_color.get(keypoint)

    def keypoint_color_list(self) -> list[str | None]:
        """Pinned colours aligned with :attr:`keypoint_names`, ``None`` where unset.

        The form the palette wants: one slot per keypoint, so the caller never
        has to know which keypoints were pinned.
        """
        return [self.keypoint_color.get(name) for name in self.keypoint_names]

    def clear_keypoint_colors(self) -> None:
        """Drop every pinned colour — both palettes go back to generated ones.

        One button for both axes: "reset colours" means the schema looks the way
        a fresh one does, whichever axis the display is colouring by.
        """
        self.keypoint_color = {}
        self.individual_color = {}

    def set_individual_color(self, individual: str, color: str | None) -> None:
        """Pin *individual* to ``"#rrggbb"``, or hand it back to the palette."""
        self.individual_index(individual)  # a colour for an unknown individual is a bug
        if color is None:
            self.individual_color.pop(individual, None)
            return
        self.individual_color[individual] = normalise_color(color)

    def individual_color_list(self) -> list[str | None]:
        """Pinned colours aligned with :attr:`individual_names`, ``None`` where unset."""
        return [self.individual_color.get(name) for name in self.individual_names]

    def _normalise_colors(self) -> None:
        """Keep the pinned colours to names the schema still has.

        A colour for a deleted keypoint or individual is state nothing can show,
        so it goes with the name rather than lingering in the sidecar.
        """
        self.keypoint_color = {
            name: normalise_color(self.keypoint_color[name])
            for name in self.keypoint_names
            if name in self.keypoint_color
        }
        self.individual_color = {
            name: normalise_color(self.individual_color[name])
            for name in self.individual_names
            if name in self.individual_color
        }

    def _normalise_keypoint_sets(self) -> None:
        """Keep :attr:`keypoint_sets` in step with the individuals and the union."""
        if self.shared_keypoints:
            self.keypoint_sets = {}
            return
        normalised: dict[str, list[str]] = {}
        for individual in self.individual_names:
            owned = set(self.keypoint_sets.get(individual, self.keypoint_names))
            normalised[individual] = [name for name in self.keypoint_names if name in owned]
        self.keypoint_sets = normalised

    def _prune_unowned(self) -> None:
        """Drop anchors and detections for pairs the schema no longer contains.

        Detections go the same way as anchors: a pair an asymmetric schema
        excludes cannot hold a position, whoever placed it. The *assignments*
        naming those pairs are deliberately left alone — see
        :meth:`AssignmentTable.invalid_labels`.
        """
        unowned = ~self.keypoint_mask()
        if not unowned.any():
            return
        pruned = False
        for frame in list(self.anchors):
            points = self.anchors[frame]
            if np.any(~np.isnan(points[unowned])):
                points[unowned] = np.nan
                pruned = True
                self._drop_if_empty(frame)
        for frame in list(self.detections):
            points = self.detections[frame]
            if np.any(~np.isnan(points[unowned])):
                points[unowned] = np.nan
                self.detection_confidence[frame][unowned] = np.nan
                if frame in self.detection_orientation:
                    self.detection_orientation[frame][unowned] = np.nan
                pruned = True
                self._detections_revision += 1
                self._drop_detections_if_empty(frame)
        if pruned:
            self._history.clear()
            self.clear_fill()

    def _reschema(self, keypoints: list[str], individuals: list[str]) -> None:
        """Re-index anchors and detections onto a new schema by name.

        Dropped names lose their points; any existing fill is invalidated, since
        its axes no longer match.
        """
        if keypoints == self.keypoint_names and individuals == self.individual_names:
            return
        old_kp = {n: i for i, n in enumerate(self.keypoint_names)}
        old_ind = {n: i for i, n in enumerate(self.individual_names)}

        def remap(sparse: dict[int, np.ndarray], trailing: tuple[int, ...]) -> dict[int, np.ndarray]:
            out: dict[int, np.ndarray] = {}
            for frame, points in sparse.items():
                new_points = np.full((len(individuals), len(keypoints), *trailing), np.nan, dtype=np.float64)
                for i, individual in enumerate(individuals):
                    if individual not in old_ind:
                        continue
                    for k, keypoint in enumerate(keypoints):
                        if keypoint in old_kp:
                            new_points[i, k] = points[old_ind[individual], old_kp[keypoint]]
                if np.any(~np.isnan(new_points)):
                    out[frame] = new_points
            return out

        remapped = remap(self.anchors, (2,))
        self.detections = remap(self.detections, (2,))
        self._detections_revision += 1
        self.detection_confidence = {
            frame: array for frame, array in remap(self.detection_confidence, ()).items() if frame in self.detections
        }
        self.detection_orientation = {
            frame: array for frame, array in remap(self.detection_orientation, (2,)).items() if frame in self.detections
        }
        self.keypoint_names = keypoints
        self.individual_names = individuals
        self.anchors = remapped
        self._history.clear()
        self.clear_fill()
        self._normalise_keypoint_sets()
        self._normalise_colors()
        self._prune_unowned()

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def set_point(self, frame: int, keypoint: str, xy: tuple[float, float], individual: str | None = None) -> None:
        """Place *keypoint* of *individual* at ``xy`` on *frame*."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        if not self.has_keypoint(keypoint, individual):
            raise UnknownKeypointError(f"{keypoint!r} is not a keypoint of {self.individual_names[i]!r}")
        frame = int(frame)
        if keypoint in self.static_keypoints:
            # One position everywhere: placing it again moves it, so the
            # previous frame's copy goes before the new one is written.
            for other in [f for f, pts in self.anchors.items() if f != frame and not np.isnan(pts[i, k, 0])]:
                self._record(other, i, k, self.anchors[other][i, k])
                self.anchors[other][i, k] = np.nan
                self._drop_if_empty(other)
        points = self.anchors.get(frame)
        if points is None:
            points = np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
            self.anchors[frame] = points
        self._record(frame, i, k, points[i, k])
        points[i, k] = (float(xy[0]), float(xy[1]))

    def clear_point(self, frame: int, keypoint: str, individual: str | None = None) -> None:
        """Remove *keypoint* of *individual* from *frame*; drops empty anchors."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        frame = int(frame)
        if keypoint in self.static_keypoints:
            # It is on every frame, so clearing it clears it wherever it lives.
            for other in [f for f, pts in self.anchors.items() if not np.isnan(pts[i, k, 0])]:
                self._record(other, i, k, self.anchors[other][i, k])
                self.anchors[other][i, k] = np.nan
                self._drop_if_empty(other)
            return
        points = self.anchors.get(frame)
        if points is None:
            return
        self._record(frame, i, k, points[i, k])
        points[i, k] = np.nan
        self._drop_if_empty(frame)

    def clear_individual(self, frame: int, individual: str | None = None) -> None:
        """Remove every keypoint of *individual* on *frame* (one undo step each)."""
        i = self.individual_index(individual)
        points = self.anchors.get(int(frame))
        if points is None:
            return
        for k, keypoint in enumerate(self.keypoint_names):
            if not np.isnan(points[i, k, 0]):
                self.clear_point(frame, keypoint, self.individual_names[i])

    def clear_all_labels(self) -> None:
        """Drop every labelled point, on every frame and individual.

        Mirrors :meth:`clear_individual` for the whole table rather than one
        row; any existing fill is left alone, same as a single row's deletion.
        History is discarded too — there is nothing left to undo back to.
        """
        self.anchors.clear()
        self._history.clear()

    # ------------------------------------------------------------------
    # Detections
    # ------------------------------------------------------------------

    @property
    def detections_revision(self) -> int:
        """Changes whenever the detections do — see :attr:`_detections_revision`."""
        return self._detections_revision

    def _drop_detections_if_empty(self, frame: int) -> None:
        points = self.detections.get(frame)
        if points is not None and not np.any(~np.isnan(points)):
            del self.detections[frame]
            self.detection_confidence.pop(frame, None)
            self.detection_orientation.pop(frame, None)

    def set_detections(
        self,
        positions: dict[int, np.ndarray],
        confidence: dict[int, np.ndarray] | None = None,
        quality_min: float = 0.0,
        orientation: dict[int, np.ndarray] | None = None,
    ) -> int:
        """Replace every detection with the result of a detector run.

        *positions* maps frame to ``(n_individuals, n_keypoints, 2)`` and
        *confidence* to ``(n_individuals, n_keypoints)``; points scoring below
        *quality_min* are dropped here rather than in the detector, so the
        threshold can be returned without decoding the video again.

        *orientation* is the same shape as *positions* and is filtered by the
        same mask — a point dropped for quality takes its heading with it, since
        the heading is only as good as the decode it came from.

        The fill is discarded: it was inferred from the previous observations,
        and a fill that ignores the evidence now sitting beside it is worse than
        no fill at all. Deleting a *single* detection deliberately does not do
        this, exactly as deleting a single label does not.
        """
        confidence = confidence or {}
        orientation = orientation or {}
        unowned = ~self.keypoint_mask()
        kept_positions: dict[int, np.ndarray] = {}
        kept_confidence: dict[int, np.ndarray] = {}
        kept_orientation: dict[int, np.ndarray] = {}
        total = 0
        for frame, points in positions.items():
            points = np.asarray(points, dtype=np.float64).copy()
            if points.shape != (self.n_individuals, self.n_keypoints, 2):
                raise KeypointStoreError(
                    f"detections for frame {frame} have shape {points.shape}, "
                    f"expected {(self.n_individuals, self.n_keypoints, 2)}"
                )
            scores = confidence.get(frame)
            scores = (
                np.full((self.n_individuals, self.n_keypoints), np.nan)
                if scores is None
                else np.asarray(scores, dtype=np.float64).copy()
            )
            if scores.shape != (self.n_individuals, self.n_keypoints):
                raise KeypointStoreError(
                    f"detection confidence for frame {frame} has shape {scores.shape}, "
                    f"expected {(self.n_individuals, self.n_keypoints)}"
                )
            drop = unowned | (np.nan_to_num(scores, nan=1.0) < quality_min)
            points[drop] = np.nan
            scores[drop] = np.nan
            found = ~np.isnan(points[:, :, 0])
            if not found.any():
                continue
            scores[~found] = np.nan
            headings = orientation.get(frame)
            headings = (
                np.full((self.n_individuals, self.n_keypoints, 2), np.nan)
                if headings is None
                else np.asarray(headings, dtype=np.float64).copy()
            )
            if headings.shape != (self.n_individuals, self.n_keypoints, 2):
                raise KeypointStoreError(
                    f"detection orientation for frame {frame} has shape {headings.shape}, "
                    f"expected {(self.n_individuals, self.n_keypoints, 2)}"
                )
            headings[~found] = np.nan
            kept_positions[int(frame)] = points
            kept_confidence[int(frame)] = scores
            if not np.all(np.isnan(headings)):
                kept_orientation[int(frame)] = headings
            total += int(found.sum())
        self.detections = kept_positions
        self.detection_confidence = kept_confidence
        self.detection_orientation = kept_orientation
        self._detections_revision += 1
        self.clear_fill()
        return total

    def set_detections_from_flat(
        self,
        positions: dict[int, np.ndarray],
        confidence: dict[int, np.ndarray] | None = None,
        quality_min: float = 0.0,
        orientation: dict[int, np.ndarray] | None = None,
    ) -> int:
        """As :meth:`set_detections`, given flat ``(n_points, …)`` rows.

        Detectors know nothing of the hierarchy, exactly like the fill backends
        (see :meth:`flat_anchors`), so this is the shape they hand back.
        """
        shape = (self.n_individuals, self.n_keypoints)
        return self.set_detections(
            {frame: np.asarray(points).reshape(*shape, 2) for frame, points in positions.items()},
            {frame: np.asarray(scores).reshape(shape) for frame, scores in (confidence or {}).items()},
            quality_min,
            {frame: np.asarray(vectors).reshape(*shape, 2) for frame, vectors in (orientation or {}).items()},
        )

    def clear_detections(self) -> None:
        """Discard the whole detector run (and the fill derived from it)."""
        if not self.detections:
            return
        self.detections = {}
        self.detection_confidence = {}
        self.detection_orientation = {}
        self._detections_revision += 1
        self.clear_fill()

    def clear_detections_for(self, frame: int, individual: str | None = None) -> int:
        """Reject the detections on one frame; returns how many went.

        For the frames where the detector locked onto a reflection or misread a
        tag. Labels are untouched, and so is the fill — this is a local
        correction, like deleting one label.
        """
        frame = int(frame)
        points = self.detections.get(frame)
        if points is None:
            return 0
        found = ~np.isnan(points[:, :, 0])
        if individual is not None:
            keep = np.zeros_like(found)
            keep[self.individual_index(individual)] = found[self.individual_index(individual)]
            found = keep
        if not found.any():
            return 0
        points[found] = np.nan
        self.detection_confidence[frame][found] = np.nan
        if frame in self.detection_orientation:
            self.detection_orientation[frame][found] = np.nan
        self._detections_revision += 1
        self._drop_detections_if_empty(frame)
        return int(found.sum())

    def detection_frames(self) -> list[int]:
        """Sorted frames carrying at least one detection."""
        return sorted(self.detections)

    @property
    def has_orientation(self) -> bool:
        """Whether any detection carries a heading — i.e. whether a head
        direction can be computed at all. Only oriented markers produce one, so
        a session with no detector run, or one whose detector reports no
        orientation, simply has no head direction to offer."""
        return any(np.any(~np.isnan(vectors)) for vectors in self.detection_orientation.values())

    def detected_mask(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints)`` bool: points a *detector* placed.

        Points the user also labelled are excluded — manual wins, so their
        position no longer comes from the detector even though it found them.
        """
        points = self.detections.get(int(frame))
        if points is None:
            return np.zeros((self.n_individuals, self.n_keypoints), dtype=bool)
        return ~np.isnan(points[:, :, 0]) & ~self.human_mask(frame)

    def is_detected(self, frame: int, individual: str | None = None) -> bool:
        """Whether any point of this row comes from a detector."""
        mask = self.detected_mask(frame)
        return bool(mask[self.individual_index(individual)].any() if individual is not None else mask.any())

    def promote_detections(self, frame: int, individual: str | None = None) -> int:
        """Pin this frame's detections as user labels; returns how many.

        Blessing a detector run into ground truth — for seeding a training set,
        or for freezing a run before retuning the detector's parameters. Points
        already labelled are left alone, and each promotion is its own undo step.
        """
        return self._promote(frame, individual, self.detections.get(int(frame)))

    def observations(self) -> dict[int, np.ndarray]:
        """Every frame-local position: anchors merged over detections.

        The evidence a fill backend interpolates between, as opposed to the
        inference it produces. Manual wins per *point*, not per frame, so a
        corrected keypoint sits beside the detector's other ones.
        """
        merged: dict[int, np.ndarray] = {frame: points.copy() for frame, points in self.detections.items()}
        for frame, points in self.anchors.items():
            labelled = ~np.isnan(points[:, :, 0])
            if frame in merged:
                merged[frame][labelled] = points[labelled]
            else:
                merged[frame] = points.copy()
        if self.static_keypoints:
            # A static keypoint rides along on every observed frame, but the
            # frame it was clicked on is not an observation of anything that
            # moves: left in, it would widen the span the fill covers and
            # extrapolate the moving points towards it.
            static = np.zeros(self.n_keypoints, dtype=bool)
            for name in self.static_keypoints:
                if name in self.keypoint_names:
                    static[self.keypoint_index(name)] = True
            merged = {frame: points for frame, points in merged.items() if not np.isnan(points[:, ~static, 0]).all()}
            for points in merged.values():
                self._overlay_static(points)
        return merged

    def flat_observations(self) -> dict[int, np.ndarray]:
        """:meth:`observations` as ``frame -> (n_points, 2)``, for the backends."""
        return {frame: points.reshape(-1, 2) for frame, points in self.observations().items()}

    def observation_positions(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints, 2)`` of observed points on one frame."""
        points = self.anchor_positions(frame)
        detected = self.detections.get(int(frame))
        if detected is not None:
            missing = np.isnan(points[:, :, 0])
            points[missing] = detected[missing]
        return points

    def assignment_rows(self) -> dict[int, int]:
        """``detector label -> flat point row``, skipping invalid assignments.

        The flat row is the index a detector's output is written to (see
        :meth:`flat_anchors`), so this is the whole of what a detector run needs
        to know about the hierarchy.
        """
        invalid = self.assignment.invalid_labels(self)
        rows: dict[int, int] = {}
        for entry in self.assignment:
            if entry.label in invalid:
                continue
            i = self.individual_index(entry.individual)
            rows[entry.label] = i * self.n_keypoints + self.keypoint_index(entry.keypoint)
        return rows

    # ------------------------------------------------------------------

    def predicted_mask(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints)`` bool: points the *backend* inferred.

        Observations are copied into :attr:`filled` verbatim, so "has a filled
        position" is not the same question as "was predicted"; this excludes
        both the labelled and the detected ones, leaving the points that exist
        only because a backend interpolated them.
        """
        if self.filled is None or not 0 <= int(frame) < self.n_frames:
            return np.zeros((self.n_individuals, self.n_keypoints), dtype=bool)
        return ~np.isnan(self.filled[int(frame), :, :, 0]) & ~self.human_mask(frame) & ~self.detected_mask(frame)

    def clear_fill_for(self, frame: int, individual: str | None = None) -> int:
        """Discard the predicted points of *frame*; returns how many went.

        The counterpart of :meth:`promote_fill`: where that keeps a prediction,
        this throws one away — for the frames where the animal is occluded or
        out of shot and the backend placed a position anyway. **Labels are left
        alone**, so a row strips back to what the user actually placed.

        The fill is derived data (it is never persisted), so a later fill will
        produce these points again unless the reason they were wrong — a missing
        anchor nearby — has been fixed.
        """
        frame = int(frame)
        predicted = self.predicted_mask(frame)
        if not predicted.any():
            return 0
        if individual is not None:
            keep = np.zeros_like(predicted)
            keep[self.individual_index(individual)] = predicted[self.individual_index(individual)]
            predicted = keep
        self.filled[frame][predicted] = np.nan
        self.confidence[frame][predicted] = np.nan
        return int(predicted.sum())

    def has_fill(self, frame: int, individual: str | None = None) -> bool:
        """Whether a backend produced any point for *individual* on *frame*."""
        predicted = self.predicted_mask(frame)
        if individual is not None:
            predicted = predicted[self.individual_index(individual)]
        return bool(predicted.any())

    def promote_fill(self, frame: int, individual: str | None = None) -> int:
        """Pin everything shown on *frame* as user labels; returns how many.

        "Accepting" a prediction, in bulk: every point that is not already an
        anchor becomes one exactly where it sits, so the next fill treats it as
        ground truth rather than re-deriving it. That covers a detection as well
        as an interpolated point — what the user is agreeing with is what is on
        the screen, and the screen shows both. Points already labelled are left
        alone: a human position always wins. Each promotion is its own undo
        step, as if it had been clicked.
        """
        return self._promote(frame, individual, self.positions(int(frame)))

    def _promote(self, frame: int, individual: str | None, source: np.ndarray | None) -> int:
        """Copy *source* positions into the anchors wherever nothing is labelled."""
        frame = int(frame)
        if source is None or not 0 <= frame < self.n_frames:
            return 0
        rows = range(self.n_individuals) if individual is None else [self.individual_index(individual)]
        owned = self.keypoint_mask()
        placed = self.anchor_positions(frame)
        promoted = 0
        for i in rows:
            for k, keypoint in enumerate(self.keypoint_names):
                if not owned[i, k] or not np.isnan(placed[i, k, 0]) or np.isnan(source[i, k, 0]):
                    continue
                self.set_point(frame, keypoint, tuple(source[i, k]), self.individual_names[i])
                promoted += 1
        return promoted

    def promote_all_detections(self) -> int:
        """Pin **every** detected point on every frame as a user label.

        The bulk form of :meth:`promote_detections`, with the same rule: a point
        already labelled is left alone, because a human position always wins.
        """
        return self._promote_bulk(self.detection_frames(), self.detections.get)

    def promote_all_fill(self) -> int:
        """Pin **everything shown** across the fill's span as user labels.

        The bulk form of :meth:`promote_fill`, and like it this covers detected
        points as well as interpolated ones — what is being agreed with is what
        is on screen, and the screen shows both.
        """
        span = self.fill_range
        frames = range(span[0], span[1] + 1) if span else self.detection_frames()
        return self._promote_bulk(frames, self.positions)

    def _promote_bulk(self, frames, source_for) -> int:
        """Promote across many frames at once, vectorised and unundoable.

        Two departures from :meth:`_promote`, both forced by scale — this runs
        over a whole fill span, which can be every frame of the video:

        **It does not go through `set_point`.** A per-point Python loop over
        100k frames is tens of millions of iterations; the mask below is a
        handful of numpy ops per frame.

        **It discards the undo history**, exactly as :meth:`clear_all_labels`
        does. Recording an entry per promoted point would build a stack nobody
        can walk back — hundreds of thousands of presses of ``Ctrl+Z`` — and
        keeping the *old* history while the arrays move underneath it would let
        a later undo restore a point into a frame it no longer describes. The
        callers confirm first, for that reason.
        """
        owned = self.keypoint_mask()
        promoted = 0
        for frame in frames:
            frame = int(frame)
            source = source_for(frame)
            if source is None or not 0 <= frame < self.n_frames:
                continue
            points = self.anchors.get(frame)
            if points is None:
                points = np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
                self.anchors[frame] = points
            missing = owned & np.isnan(points[:, :, 0]) & ~np.isnan(source[:, :, 0])
            if not missing.any():
                self._drop_if_empty(frame)
                continue
            points[missing] = source[missing]
            promoted += int(missing.sum())
        self._history.clear()
        return promoted

    def _drop_if_empty(self, frame: int) -> None:
        points = self.anchors.get(frame)
        if points is not None and not np.any(~np.isnan(points)):
            del self.anchors[frame]

    def _record(self, frame: int, i: int, k: int, previous: np.ndarray) -> None:
        self._history.append((frame, i, k, None if np.all(np.isnan(previous)) else previous.copy()))

    def undo(self) -> int | None:
        """Revert the last ``set_point`` / ``clear_point``; returns its frame.

        The frame is returned rather than remembered, because an undo can land
        anywhere — the caller redraws it, and there is nothing left to go stale.
        """
        if not self._history:
            return None
        frame, i, k, previous = self._history.pop()
        points = self.anchors.get(frame)
        if previous is None:
            if points is not None:
                points[i, k] = np.nan
                self._drop_if_empty(frame)
            return frame
        if points is None:
            points = np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
            self.anchors[frame] = points
        points[i, k] = previous
        return frame

    def clear_fill(self) -> None:
        self.filled = None
        self.confidence = None
        self.fill_range = None

    def has_predictions(self, frame: int, individual: str | None = None) -> bool:
        """Whether *frame* carries anything the user has not placed themselves.

        What "Approve frame" acts on: a detection is as much a proposal to agree
        with as an interpolated point.
        """
        return self.has_fill(frame, individual) or self.is_detected(frame, individual)

    def confidence_at(self, frame: int) -> np.ndarray | None:
        """``(n_individuals, n_keypoints)`` score behind :meth:`positions`.

        Composed in the same precedence: ``1.0`` where the user labelled,
        the detector's own quality where it found a marker, and the fill's score
        elsewhere. ``None`` when the frame carries neither a detection nor a
        fill. A point placed *after* a fill still carries the fill's old score in
        :attr:`confidence` — that array is a snapshot — so the human ``1.0`` is
        re-applied here rather than read back out of it.
        """
        frame = int(frame)
        has_fill = self.confidence is not None and 0 <= frame < len(self.confidence)
        detected = self.detections.get(frame)
        if not has_fill and detected is None:
            return None
        out = np.full((self.n_individuals, self.n_keypoints), np.nan, dtype=np.float64)
        if has_fill:
            out[:] = self.confidence[frame]
        if detected is not None:
            found = self.detected_mask(frame)
            out[found] = self.detection_confidence[frame][found]
        out[self.human_mask(frame)] = 1.0
        return out

    def set_fill(self, filled: np.ndarray, confidence: np.ndarray) -> None:
        """Store a backend result, re-asserting the observations over it.

        Backends are expected to return what they were given verbatim;
        re-applying it here makes that invariant hold regardless of the backend.
        A detection keeps the detector's own quality rather than an anchor's
        ``1.0`` — it is evidence, but read by a machine.
        """
        filled = np.asarray(filled, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        expected = (self.n_frames, self.n_individuals, self.n_keypoints, 2)
        if filled.shape != expected:
            raise KeypointStoreError(f"filled has shape {filled.shape}, expected {expected}")
        if confidence.shape != expected[:3]:
            raise KeypointStoreError(f"confidence has shape {confidence.shape}, expected {expected[:3]}")
        for frame, points in self.detections.items():
            if not 0 <= frame < self.n_frames:
                continue
            found = ~np.isnan(points[:, :, 0])
            filled[frame][found] = points[found]
            confidence[frame][found] = self.detection_confidence[frame][found]
        for frame, points in self.anchors.items():
            if not 0 <= frame < self.n_frames:
                continue
            labelled = ~np.isnan(points[:, :, 0])
            filled[frame][labelled] = points[labelled]
            confidence[frame][labelled] = 1.0
        # Backends track the whole grid; pairs an asymmetric schema excludes are
        # tracked from nothing, so their output is meaningless — blank it out.
        unowned = ~self.keypoint_mask()
        filled[:, unowned] = np.nan
        confidence[:, unowned] = np.nan
        self.filled = filled
        self.confidence = confidence
        # Measured from the result rather than from the anchors: a backend fills
        # the gaps between labels, and what it actually covered is what readers
        # (the points table above all) must not go looking beyond.
        covered = np.flatnonzero(np.isfinite(filled[..., 0]).any(axis=(1, 2)))
        self.fill_range = (int(covered[0]), int(covered[-1])) if len(covered) else None

    def set_fill_from_flat(self, filled: np.ndarray, confidence: np.ndarray) -> None:
        """Store a backend result given as flat points (see :meth:`flat_anchors`)."""
        shape = (self.n_frames, self.n_individuals, self.n_keypoints)
        self.set_fill(np.asarray(filled).reshape(*shape, 2), np.asarray(confidence).reshape(shape))

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def flat_anchors(self) -> dict[int, np.ndarray]:
        """Anchors as ``frame -> (n_points, 2)`` for the fill backends.

        Backends track independent points and neither know nor care which
        individual a point belongs to, so the hierarchy is flattened here and
        restored by :meth:`set_fill_from_flat`.
        """
        return {frame: points.reshape(-1, 2).copy() for frame, points in self.anchors.items()}

    def anchor_frames(self) -> list[int]:
        """Sorted frames carrying at least one labelled point."""
        return sorted(self.anchors)

    def anchor_frames_for(self, keypoint: str, individual: str | None = None) -> list[int]:
        """Sorted frames where this ``(individual, keypoint)`` is labelled."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        return sorted(f for f, points in self.anchors.items() if not np.isnan(points[i, k, 0]))

    def labelled_points(
        self,
        individual: str | None = None,
        keypoint: str | None = None,
    ) -> list[tuple[int, str, str, float, float]]:
        """Every user-placed point as ``(frame, individual, keypoint, x, y)``.

        Sorted by frame, then by the schema order of individuals and keypoints —
        the order the dialog's table shows. Both filters are optional; passing
        neither returns everything.
        """
        out: list[tuple[int, str, str, float, float]] = []
        for frame in sorted(self.anchors):
            points = self.anchors[frame]
            for i, name in enumerate(self.individual_names):
                if individual is not None and name != individual:
                    continue
                for k, kp in enumerate(self.keypoint_names):
                    if keypoint is not None and kp != keypoint:
                        continue
                    x, y = points[i, k]
                    if not np.isnan(x):
                        out.append((frame, name, kp, float(x), float(y)))
        return out

    def anchor_frames_for_individual(self, individual: str | None = None) -> list[int]:
        """Sorted frames where any keypoint of *individual* is labelled."""
        i = self.individual_index(individual)
        return sorted(f for f, points in self.anchors.items() if np.any(~np.isnan(points[i, :, 0])))

    def positions(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints, 2)``: manual over detected over fill."""
        out = np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
        if self.filled is not None and 0 <= frame < self.n_frames:
            out[:] = self.filled[frame]
        detected = self.detections.get(int(frame))
        if detected is not None:
            found = ~np.isnan(detected[:, :, 0])
            out[found] = detected[found]
        points = self.anchor_positions(frame)
        labelled = ~np.isnan(points[:, :, 0])
        out[labelled] = points[labelled]
        return out

    def positions_for(self, frame: int, individual: str | None = None) -> np.ndarray:
        """``(n_keypoints, 2)`` for one individual: anchors over fill."""
        return self.positions(frame)[self.individual_index(individual)]

    def anchor_positions(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints, 2)`` of user-placed points only.

        A static keypoint counts as placed on every frame (see
        :attr:`static_keypoints`).
        """
        points = self.anchors.get(int(frame))
        out = (
            np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
            if points is None
            else points.copy()
        )
        self._overlay_static(out)
        return out

    # -- static keypoints ------------------------------------------------

    def is_static(self, keypoint: str) -> bool:
        return keypoint in self.static_keypoints

    def set_static(self, keypoint: str, static: bool) -> None:
        """Mark *keypoint* as fixed in place (or moving again).

        Making a keypoint static keeps its **first** labelled frame and drops
        the others — one position is what "static" means.
        """
        if keypoint not in self.keypoint_names:
            raise UnknownKeypointError(f"{keypoint!r} is not a keypoint")
        if static and keypoint not in self.static_keypoints:
            self.static_keypoints.append(keypoint)
            k = self.keypoint_index(keypoint)
            for i in range(self.n_individuals):
                labelled = sorted(f for f, pts in self.anchors.items() if not np.isnan(pts[i, k, 0]))
                for other in labelled[1:]:
                    self.anchors[other][i, k] = np.nan
                    self._drop_if_empty(other)
        elif not static and keypoint in self.static_keypoints:
            self.static_keypoints.remove(keypoint)

    def static_anchor(self, keypoint: str, individual: str | None = None) -> np.ndarray:
        """The one ``(x, y)`` a static keypoint was placed at, or ``NaN``s."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        for pts in self.anchors.values():
            if not np.isnan(pts[i, k, 0]):
                return pts[i, k].copy()
        return np.full(2, np.nan)

    def _overlay_static(self, points: np.ndarray) -> None:
        """Write every static keypoint's position into *points* (in place)."""
        for name in self.static_keypoints:
            if name not in self.keypoint_names:
                continue
            k = self.keypoint_index(name)
            for i in range(self.n_individuals):
                xy = self.static_anchor(name, self.individual_names[i])
                if not np.isnan(xy[0]):
                    points[i, k] = xy

    def pin_static(self, filled: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A fill's output with every static keypoint held at its one position.

        Backends track every point they are handed; a corner tracked across a
        gap drifts with the noise. The one position is authoritative, so it is
        written back over the whole span, at anchor confidence.
        """
        filled = np.array(filled, copy=True)
        confidence = np.array(confidence, copy=True)
        # Backends speak the flat ``(n_frames, n_points, 2)`` layout; the store
        # keeps ``(n_frames, n_individuals, n_keypoints, 2)``. Either is fine
        # here, and comes back in the layout it arrived in.
        view = filled.reshape(len(filled), self.n_individuals, self.n_keypoints, 2)
        conf_view = confidence.reshape(len(confidence), self.n_individuals, self.n_keypoints)
        for name in self.static_keypoints:
            if name not in self.keypoint_names:
                continue
            k = self.keypoint_index(name)
            for i in range(self.n_individuals):
                xy = self.static_anchor(name, self.individual_names[i])
                if not np.isnan(xy[0]):
                    view[:, i, k] = xy
                    conf_view[:, i, k] = 1.0
        return filled, confidence

    def seed_static_from(self, other: KeypointStore, frame: int = 0) -> int:
        """Copy *other*'s static keypoints and their positions in, for a new video.

        Same camera, same pixels: the corners of the box are where they were in
        the last clip. Only keypoints this schema also has are copied, onto
        *frame*; nothing already placed here is overwritten. Returns how many
        were seeded.
        """
        seeded = 0
        for name in other.static_keypoints:
            if name not in self.keypoint_names:
                continue
            if name not in self.static_keypoints:
                self.static_keypoints.append(name)
            for individual in self.individual_names:
                if individual not in other.individual_names or not self.has_keypoint(name, individual):
                    continue
                xy = other.static_anchor(name, individual)
                if np.isnan(xy[0]) or not np.isnan(self.static_anchor(name, individual)[0]):
                    continue
                self.set_point(frame, name, (float(xy[0]), float(xy[1])), individual)
                seeded += 1
        return seeded

    def anchor_positions_for(self, frame: int, individual: str | None = None) -> np.ndarray:
        """``(n_keypoints, 2)`` of user-placed points for one individual."""
        return self.anchor_positions(frame)[self.individual_index(individual)]

    def human_mask(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints)`` bool: which points the user placed.

        The complement of this over :meth:`positions` is what a fill backend
        produced, so the two together say where every coordinate came from.
        """
        return ~np.isnan(self.anchor_positions(frame)[:, :, 0])

    def is_anchor(self, frame: int, keypoint: str, individual: str | None = None) -> bool:
        """Whether this ``(individual, keypoint)`` was placed by the user on *frame*."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        points = self.anchors.get(int(frame))
        return points is not None and not bool(np.isnan(points[i, k, 0]))

    def is_human(self, frame: int, individual: str | None = None) -> bool:
        """Whether *individual* carries any user-placed point on *frame*.

        One labelled or corrected point is enough: a row the user has touched is
        theirs, even if a backend supplied the rest of its keypoints.
        """
        mask = self.human_mask(frame)
        return bool(mask[self.individual_index(individual)].any() if individual is not None else mask.any())

    def labelled_count(self, frame: int, individual: str | None = None) -> int:
        """Keypoints labelled on *frame* — for one individual, or all of them."""
        points = self.anchor_positions(frame)
        if individual is not None:
            points = points[self.individual_index(individual)][None]
        return int(np.count_nonzero(~np.isnan(points[..., 0])))

    def nearest(
        self,
        frame: int,
        xy: tuple[float, float],
        radius: float,
        include_fill: bool = False,
    ) -> tuple[str, str] | None:
        """``(individual, keypoint)`` of the point within *radius* px of ``xy``.

        Anchors only by default. With *include_fill* the filled points are
        candidates too, which is what makes a prediction grabbable on the canvas;
        deleting deliberately does not, since only an anchor can be removed.
        """
        points = self.positions(int(frame)) if include_fill else self.anchors.get(int(frame))
        if points is None or points.size == 0:
            return None
        distances = np.hypot(points[:, :, 0] - xy[0], points[:, :, 1] - xy[1])
        distances[np.isnan(distances)] = np.inf
        i, k = np.unravel_index(int(np.argmin(distances)), distances.shape)
        if distances[i, k] > radius:
            return None
        return self.individual_names[i], self.keypoint_names[k]

    # ------------------------------------------------------------------
    # Sidecar persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        payload = {
            "keypoint": list(self.keypoint_names),
            "individual": list(self.individual_names),
            "shared_keypoints": self.shared_keypoints,
            "n_frames": self.n_frames,
            "anchors": {str(f): points.tolist() for f, points in sorted(self.anchors.items())},
        }
        if not self.shared_keypoints:
            payload["keypoint_set"] = {name: list(kps) for name, kps in self.keypoint_sets.items()}
        if self.keypoint_color:
            payload["keypoint_color"] = dict(self.keypoint_color)
        if self.individual_color:
            payload["individual_color"] = dict(self.individual_color)
        if self.static_keypoints:
            payload["static"] = list(self.static_keypoints)
        # Detections are NOT written here: they are recomputable from the video
        # and the detector's parameters, and a sidecar carrying a frame per
        # detected frame is no longer a file anyone can read. What each label
        # *means* is the part nothing can recompute, so that stays.
        if len(self.assignment):
            payload["assignment"] = self.assignment.to_list()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> KeypointStore:
        """Rebuild a store from a sidecar payload, migrating older formats.

        Sidecars written before the individual/keypoint hierarchy used a
        ``"names"`` key and 2-D ``(n_keypoints, 2)`` anchor rows. Those load as
        one individual: the names stay keypoints, because reinterpreting them
        would silently change what the user labelled.
        """
        keypoints = payload.get("keypoint", payload.get("names"))
        if keypoints is None:
            raise KeypointStoreError("Sidecar has no keypoint names — expected a 'keypoint' or 'names' key.")
        keypoints = list(keypoints)
        # A missing key is a legacy sidecar (one individual); an empty list is a
        # user who deleted every individual, and must stay empty.
        saved_individuals = payload.get("individual")
        individuals = [DEFAULT_INDIVIDUAL] if saved_individuals is None else list(saved_individuals)
        expected = (len(individuals), len(keypoints), 2)

        anchors: dict[int, np.ndarray] = {}
        for frame, points in payload.get("anchors", {}).items():
            array = np.asarray(points, dtype=np.float64)
            if array.ndim == 2:  # legacy flat layout — one individual
                array = array[None, ...]
            if array.shape != expected:
                raise KeypointStoreError(
                    f"Sidecar anchor for frame {frame} has shape {array.shape}, expected {expected}."
                )
            anchors[int(frame)] = array

        # Unknown keys are ignored, so sidecars carrying the old
        # "last_labelled_frame" still load; it simply has no meaning any more.
        return cls(
            keypoint_names=keypoints,
            n_frames=int(payload["n_frames"]),
            individual_names=individuals,
            shared_keypoints=bool(payload.get("shared_keypoints", True)),
            keypoint_sets={str(k): list(v) for k, v in (payload.get("keypoint_set") or {}).items()},
            keypoint_color={str(k): str(v) for k, v in (payload.get("keypoint_color") or {}).items()},
            individual_color={str(k): str(v) for k, v in (payload.get("individual_color") or {}).items()},
            anchors=anchors,
            assignment=AssignmentTable.from_list(payload.get("assignment")),
            static_keypoints=[str(k) for k in (payload.get("static") or []) if str(k) in keypoints],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> KeypointStore:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Detection cache
    # ------------------------------------------------------------------

    def save_detections(self, path: str | Path, signature: str) -> None:
        """Cache the current detections, tagged with what produced them.

        Derived data, so this is a cache and not a document: it exists only
        because scanning an hour of footage for tags costs minutes. *signature*
        identifies the detector and its parameters — :meth:`load_detections`
        refuses anything else rather than showing one detector's output as
        another's.
        """
        path = Path(path)
        if not self.detections:
            path.unlink(missing_ok=True)
            return
        frames = np.asarray(sorted(self.detections), dtype=np.int64)
        blank = np.full((self.n_individuals, self.n_keypoints, 2), np.nan)
        np.savez_compressed(
            path,
            signature=np.asarray(str(signature)),
            keypoint=np.asarray(self.keypoint_names, dtype=object),
            individual=np.asarray(self.individual_names, dtype=object),
            frames=frames,
            positions=np.stack([self.detections[int(f)] for f in frames]),
            confidence=np.stack([self.detection_confidence[int(f)] for f in frames]),
            orientation=np.stack([self.detection_orientation.get(int(f), blank) for f in frames]),
        )

    def load_detections(self, path: str | Path, signature: str) -> bool:
        """Restore cached detections; ``False`` when they do not apply here.

        A cache that cannot be read, was written by another detector, or belongs
        to another schema is a **miss**, not an error — the run is simply redone.
        """
        path = Path(path)
        if not path.is_file():
            return False
        with np.load(path, allow_pickle=True) as data:
            if str(data["signature"]) != str(signature):
                return False
            if list(data["keypoint"]) != self.keypoint_names or list(data["individual"]) != self.individual_names:
                return False
            frames = [int(f) for f in data["frames"]]
            positions = {frame: data["positions"][i] for i, frame in enumerate(frames)}
            confidence = {frame: data["confidence"][i] for i, frame in enumerate(frames)}
            # Absent in caches written before markers carried a heading; a miss
            # on this one alone would throw away a perfectly good run.
            vectors = data["orientation"] if "orientation" in data else None
            orientation = {} if vectors is None else {frame: vectors[i] for i, frame in enumerate(frames)}
        self.set_detections(positions, confidence, orientation=orientation)
        return True


def sidecar_path(video_path: str | Path) -> Path:
    """Anchor sidecar location for *video_path* (``<video>.keypoints.json``)."""
    video = Path(video_path)
    return video.with_name(video.name + SIDECAR_SUFFIX)


def refinement_path(video_path: str | Path) -> Path:
    """Where a test-time refinement for *video_path* is cached.

    Project data like the anchors, and next to them for the same reason: the fit
    is minutes of GPU time that belongs to this video, so reopening tomorrow must
    not re-pay it. Lives here rather than in :mod:`pose_refine` so the dialog can
    find it without importing torch.
    """
    video = Path(video_path)
    return video.with_name(video.name + REFINEMENT_SUFFIX)


def detections_path(video_path: str | Path) -> Path:
    """Where a detector run for *video_path* is cached (``<video>.detections.npz``).

    Beside the refinement cache and for the same reason: derived from the video
    plus a set of parameters, but minutes of work to reproduce. Unlike the
    anchors it is never the document of record — deleting it costs a re-run.
    """
    video = Path(video_path)
    return video.with_name(video.name + DETECTIONS_SUFFIX)


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


def store_to_movement_ds(
    store: KeypointStore,
    fps: float,
    image_height: float | None = None,
) -> xr.Dataset:
    """Build a movement-format poses dataset from *store*.

    Dims are ``(time, space, keypoint, individual)`` — singular, as movement
    ≥0.17 and the rest of Ethograph name them — so the result feeds the existing
    ``PoseRenderData`` path and everything downstream (overlay, filtering,
    kinematics, NWB) unchanged. Time is in seconds; ``fps`` must come from the
    video, never a default.

    ``image_height`` flips y to a **y-up** convention (``y_out = height - y``).
    Anchors are stored in image coordinates, where y grows *downward* from the
    top-left corner — the convention every pose file uses. Plots are y-up, so
    an unflipped trajectory comes out vertically mirrored: a keypoint at the top
    of the frame is drawn at the bottom of the plot. The canvas overlays handle
    this themselves (``y_world = img_height - y``) and DeepLabCut expects raw
    image coordinates, so the flip belongs only on the paths that hand data to a
    plot. Leave it ``None`` to keep image coordinates.

    """
    if fps <= 0:
        raise KeypointStoreError("fps must be positive — read it from the video, do not default it.")
    if image_height is not None and image_height <= 0:
        raise KeypointStoreError("image_height must be positive — read it from the video, do not default it.")

    n_ind, n_kp = store.n_individuals, store.n_keypoints
    position = np.full((store.n_frames, 2, n_kp, n_ind), np.nan, dtype=np.float64)
    confidence = np.full((store.n_frames, n_kp, n_ind), np.nan, dtype=np.float64)

    if store.filled is not None:
        # store: (time, individual, keypoint, space) -> ds: (time, space, keypoint, individual)
        position[:] = store.filled.transpose(0, 3, 2, 1)
    if store.confidence is not None:
        confidence[:] = store.confidence.transpose(0, 2, 1)

    # Same precedence as ``positions``: detections over the fill, labels over
    # both. Written out even with no fill loaded — a detector run is a pose
    # dataset in its own right.
    for frame, points in store.detections.items():
        if not 0 <= frame < store.n_frames:
            continue
        for i, k in zip(*np.nonzero(~np.isnan(points[:, :, 0]))):
            position[frame, :, k, i] = points[i, k]
            confidence[frame, k, i] = store.detection_confidence[frame][i, k]

    for frame, points in store.anchors.items():
        if not 0 <= frame < store.n_frames:
            continue
        labelled = ~np.isnan(points[:, :, 0])
        for i, k in zip(*np.nonzero(labelled)):
            position[frame, :, k, i] = points[i, k]
            confidence[frame, k, i] = 1.0

    # A static keypoint was placed once and is there on every frame.
    for name in store.static_keypoints:
        if name not in store.keypoint_names:
            continue
        k = store.keypoint_index(name)
        for i, individual in enumerate(store.individual_names):
            xy = store.static_anchor(name, individual)
            if not np.isnan(xy[0]):
                position[:, :, k, i] = xy
                confidence[:, k, i] = 1.0

    if image_height is not None:
        position[:, 1] = image_height - position[:, 1]

    # ``space_unit`` is what the overlay reads to know the data is in the
    # camera's own pixels (see io/overlay_source).
    attrs = {"ds_type": "poses", "fps": float(fps), "source_software": "ethograph", "space_unit": "pixels"}

    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(position, dims=["time", "space", "keypoint", "individual"]),
            "confidence": xr.DataArray(confidence, dims=["time", "keypoint", "individual"]),
        },
        coords={
            "time": np.arange(store.n_frames) / fps,
            "space": ["x", "y"],
            "keypoint": list(store.keypoint_names),
            "individual": list(store.individual_names),
        },
        attrs=attrs,
    )


# ----------------------------------------------------------------------
# Derived kinematics, for inspecting a fill without leaving the GUI
# ----------------------------------------------------------------------

#: Quantities derivable from the labelled + filled keypoint trajectories.
KINEMATICS = ("velocity", "speed", "acceleration")

#: What :func:`store_to_head_direction` produces. Both fall out of one forward
#: vector: the vector itself, and its angle — the same quantity in the two forms
#: people actually plot (a trajectory arrow, and a heading trace over time).
HEAD_DIRECTION = ("head_direction", "heading")


def _observed_derivative(position: xr.DataArray, order: int) -> xr.DataArray:
    """The *order*-th time derivative, taken over each point's OWN observed frames.

    ``movement.kinematics`` differentiates with ``DataArray.differentiate``, a
    central difference over the *stored* grid, so a single ``NaN`` blanks both
    its neighbours. Keypoint labelling produces exactly that shape of data: with
    a handful of anchors and no fill every observed frame is surrounded by NaN,
    and differentiating the grid returns an array that is empty everywhere —
    a velocity nobody can plot, from positions that are plainly there.

    Differentiating over the frames a point was actually seen on instead gives
    the average velocity across each gap, placed on the frames the evidence sits
    on. Where the data *is* dense — after a fill, which is the case these
    features exist to inspect — the observed frames are every frame, so this
    reduces to ``np.gradient`` over the full grid and agrees with movement value
    for value.

    Each ``(keypoint, individual)`` is handled separately because their anchor
    sets differ: the beak is labelled on some frames and the tail on others.
    """
    ordered = position.transpose("time", "space", "keypoint", "individual")
    values = ordered.values
    time = ordered.coords["time"].values.astype(float)
    out = np.full(values.shape, np.nan)

    for k in range(values.shape[2]):
        for i in range(values.shape[3]):
            track = values[:, :, k, i]
            seen = np.flatnonzero(np.all(np.isfinite(track), axis=1))
            if len(seen) <= order:
                continue
            derived = track[seen]
            for _ in range(order):
                derived = np.gradient(derived, time[seen], axis=0)
            out[seen, :, k, i] = derived

    return ordered.copy(data=out)


def store_to_kinematics(ds: xr.Dataset, features: Sequence[str] = KINEMATICS) -> dict[str, xr.DataArray]:
    """Derive kinematics from a poses dataset's ``position``.

    ``velocity`` and ``acceleration`` are the first and second time derivatives
    over each point's observed frames (see :func:`_observed_derivative`), and
    ``speed`` is the magnitude of the velocity — movement's own definition.

    Filled frames are included: the point of computing these in the GUI is to
    inspect what a fill produced, not only the frames that were labelled by hand.

    Names are plain and the time dim is the poses dataset's own, because these
    go straight back into that dataset — see :func:`store_to_dataset`.
    """
    from movement.utils.vector import compute_norm

    unknown = set(features) - set(KINEMATICS)
    if unknown:
        raise KeypointStoreError(f"Unknown kinematic(s) {sorted(unknown)}; expected {list(KINEMATICS)}")

    wanted = set(features)
    position = ds["position"]
    velocity = _observed_derivative(position, 1) if wanted & {"velocity", "speed"} else None

    computed: dict[str, xr.DataArray] = {}
    if "velocity" in wanted:
        computed["velocity"] = velocity
    if "speed" in wanted:
        computed["speed"] = compute_norm(velocity)
    if "acceleration" in wanted:
        computed["acceleration"] = _observed_derivative(position, 2)
    return computed


def store_to_head_direction(
    store: KeypointStore,
    fps: float,
    y_flipped: bool = False,
) -> dict[str, xr.DataArray]:
    """Forward vector and heading angle, read off each marker's own geometry.

    A head direction needs an *oriented* marker, and an AprilTag is exactly
    that: a square whose four corners fix which way it faces on every frame it
    decodes. The detector measures that when it reads the tag (see
    :func:`~ethograph.gui.pose_detect.quad_forward_vector`) and it is carried
    here in :attr:`KeypointStore.detection_orientation`, so a heading belongs to
    **one keypoint** — the tagged point itself.

    It is deliberately not derived from two keypoints. A pair of separately
    labelled points can give a direction, but only by making the user turn one
    physical marker into several keypoints and then say which two of them face
    left and right; get that pairing wrong and the heading is off by a quarter
    turn with nothing on screen to show it. The tag already knows, so it is
    asked instead — and a session with no oriented markers simply has no head
    direction, rather than one invented from an arbitrary pair.

    Both arrays therefore keep the ``keypoint`` dimension, unlike movement's
    ``compute_forward_vector``: one tagged keypoint is one heading, and a
    keypoint carrying no oriented marker is ``NaN`` throughout. Frames where the
    tag did not decode are ``NaN`` too — a heading is a measurement, and is
    never interpolated by the fill.

    ``y_flipped`` says the positions were flipped to a y-up convention by
    :func:`store_to_movement_ds`. Orientation is measured in image coordinates,
    so the flip has to be applied here as well or the arrow points the opposite
    way to the trajectory drawn under it; mirroring y negates the vector's y
    component and the angle follows from that.

    Angles are in **degrees**: this is read off a plot rather than fed into
    another calculation, and nobody eyeballs radians.

    Returns the arrays under their plain movement names on the poses dataset's
    own ``time`` dim, so they can be assigned straight into an exported file;
    pass them through :func:`to_keypoint_features` for the GUI instead.
    """
    if fps <= 0:
        raise KeypointStoreError("fps must be positive — read it from the video, do not default it.")

    n_ind, n_kp = store.n_individuals, store.n_keypoints
    vectors = np.full((store.n_frames, 2, n_kp, n_ind), np.nan, dtype=np.float64)
    for frame, measured in store.detection_orientation.items():
        if 0 <= frame < store.n_frames:
            # store: (individual, keypoint, space) -> ds: (space, keypoint, individual)
            vectors[frame] = np.asarray(measured, dtype=np.float64).transpose(2, 1, 0)

    if y_flipped:
        vectors[:, 1] *= -1.0

    # atan2 on a NaN pair yields NaN, so unmeasured frames stay unmeasured.
    with np.errstate(invalid="ignore"):
        angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))

    coords = {
        "time": np.arange(store.n_frames) / fps,
        "keypoint": list(store.keypoint_names),
        "individual": list(store.individual_names),
    }
    computed = {
        "head_direction": xr.DataArray(
            vectors,
            dims=["time", "space", "keypoint", "individual"],
            coords={**coords, "space": ["x", "y"]},
        ),
        "heading": xr.DataArray(angles, dims=["time", "keypoint", "individual"], coords=coords),
    }
    for array in computed.values():
        array.attrs["source"] = "marker orientation (AprilTag corners)"
    computed["heading"].attrs["ylabel"] = "heading (deg)"
    return computed


def store_to_dataset(
    store: KeypointStore,
    fps: float,
    image_height: float | None = None,
    kinematics: Sequence[str] = (),
    head_direction: bool = False,
) -> xr.Dataset:
    """The whole result of a labelling session as ONE movement poses dataset.

    This is the single artifact the feature produces. *Load into the GUI* writes
    it and loads it; *Export poses* writes the same thing to a path of the
    user's choosing. There is deliberately no second, GUI-only shape: a dataset
    that plots correctly when opened from disk is one that plots correctly when
    the dialog hands it over, and the way to guarantee that is for them to be
    the same bytes.

    Everything shares the poses dataset's own dims — ``(time, space, keypoint,
    individual)`` — so `keypoint` and `individual` are ordinary selectable
    dimensions rather than names that have to be kept from colliding with
    whatever the session already had.
    """
    ds = store_to_movement_ds(store, fps, image_height)
    derived = dict(store_to_kinematics(ds, kinematics))
    if head_direction:
        derived.update(store_to_head_direction(store, fps, y_flipped=image_height is not None))
    return ds.assign(derived) if derived else ds


def keypoints_dataset_path(video_path: str | Path) -> Path:
    """Where *Load into the GUI* writes the poses dataset (``<video>.keypoints.nc``).

    Beside the anchors and for the same reason — it belongs to this video — but
    unlike them it is **output**, regenerated from the store every time. Being a
    real file rather than a temporary is what makes loading it identical to
    loading any other dataset: the labels TSV lands next to it, reopening finds
    it, and it can be handed to anyone.
    """
    video = Path(video_path)
    return video.with_name(video.name + KEYPOINTS_DATASET_SUFFIX)
