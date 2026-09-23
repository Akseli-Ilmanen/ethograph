"""Canvas keypoint editing: place, drag and delete anchors on a camera view.

:class:`KeypointLabelMode` is attached to a :class:`~ethograph.gui.pygfx_video.CameraView`
while the labelling dialog is open. It owns the pointer interaction and a small
pygfx overlay that draws the *anchors* of the current frame — deliberately
distinct from the pose overlay that shows filled predictions: anchors are drawn
as large markers with a hairline white outline around the active one, so a
labelled point is never confused with a model output.

Labelling is hierarchical (see :mod:`~ethograph.gui.pose_annotate`): the active
target is an ``(individual, keypoint)`` pair. **Colour is the identity channel
for both axes, one at a time** — SLEAP's model, and the same
``app_state.pose_color_by`` toggle the pose display reads:

``"keypoint"`` (the default)
    One colour per keypoint, *shared across individuals* — the beak is the same
    colour on every animal. What you want while labelling: the question a click
    answers is "which body part is this?".
``"individual"``
    One colour per individual, shared across that animal's keypoints — the
    question becomes "which animal is this?", for pulling two overlapping
    animals apart.

Markers are all circles: a per-individual shape alphabet was tried and dropped,
because it made every marker a second thing to decode and still could not be
read at labelling sizes. What tells the individuals apart within a mode instead
is that **every individual other than the active one is dimmed**, each carries
its name at the centroid of its points, and the active point wears a thin white
outline. Colours come from a generated palette unless the user pins one
(``KeypointStore.keypoint_color`` / ``individual_color``); either way
:func:`keypoint_colors_for` / :func:`individual_colors_for` are what every
surface reads them through, so the canvas, the tree and the points table can
never disagree about which colour the beak is.

Editing is always available
--------------------------
There is no separate "edit" mode. A click on empty space places the active
keypoint; a click on an existing point selects it (switching to its individual
and keypoint) and drags it. ``Backspace`` / ``Delete`` remove the *selected*
point — the one the white outline is drawn around, falling back to whatever is
under the cursor — and ``Ctrl+Z`` undoes, always: correcting a mistake should
never require changing mode first.

*Filled* points are grabbable too, and grabbing one pins it as a label (see
:meth:`KeypointStore.promote_fill`) — that is how a prediction is accepted or
corrected. Deleting stays anchors-only: there is nothing to remove from a
prediction, and clearing the label under it simply hands the point back to the
next fill.

Provenance is the third visual channel: fill
--------------------------------------------
The overlay draws **every** kind of point, because accepting a prediction means
looking at it first. A *label* is a solid marker; a *prediction* is the same
colour drawn **hollow** — the interior is left empty so the pixels being judged
stay visible underneath, and the colour moves to the edge. Fill/hollow is a
channel of its own, so identity never has to be spent on it, and a click that
turns a hollow marker solid is exactly the act of pinning it.

A *detection* (:mod:`~ethograph.gui.pose_detect`) is the third state, and it
belongs on the same channel: drawn hollow **with a pip**, where the ring says
"not yours" and the centre dot says "read off these pixels, not interpolated
between two frames". That ranks the three styles the way the store ranks them,
solid → pip → empty, which is also their order of trustworthiness.

Because this overlay now shows everything, the labelling dialog does *not* push
its pose override while a mode is attached: the ordinary pose overlay would draw
a second marker on every point, in a colour scheme that says nothing about where
the point came from.

Locked: canvas control without labelling
----------------------------------------
``locked`` suspends the *pointer* interaction without detaching the mode: the
anchor overlay stays on screen, the active pair is kept, and left-drag goes back
to panning. Looking around a frame at high zoom is otherwise only possible by
disarming, which drops the overlay and the target with it. Keyboard editing
(``Backspace``, ``Ctrl+Z``) is deliberately untouched — the lock is about what
the pointer does, and those keys act on a point the user has selected on
purpose.

Two labelling modes, after napari-deeplabcut
--------------------------------------------
They differ only in what happens *after* a point is placed:

``"sequential"``
    Advance to the *first* keypoint this individual still lacks on this frame,
    in schema order — which is the left-to-right column order of the dialog's
    points table, so that table fills from the left. The playhead never moves on
    its own: you label a frame, then navigate yourself.
``"loop"``
    Keep the same keypoint and jump to the next frame. Sweeping one keypoint
    across many frames is what the fill backends actually want, since each
    keypoint is interpolated over its *own* anchor set.

``Tab`` cycles the keypoint and ``1``–``9`` select the individual in both modes;
the dialog owns the key handling.

While a mode is active and unlocked, left-drag pans no longer — panning moves to
``Shift`` + left-drag so the wheel/zoom controls keep working.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np
import pygfx as gfx

from ethograph.gui.pose_annotate import KeypointStore, normalise_color
from ethograph.gui.pose_convert import (
    COLOR_BY_INDIVIDUAL,
    COLOR_BY_KEYPOINT,
    COLOR_BY_MODES,
    sample_colormap,
)

#: Sentinel position far outside any frame, used to hide unlabelled markers.
_OFFSCREEN = -1.0e6

_Z_ANCHORS = 4.0
_Z_ACTIVE = 4.2
_Z_TEXT = 4.5

#: Screen-pixel radius for hit-testing an existing anchor.
HIT_RADIUS_PX = 12.0

#: Default marker diameter in screen pixels; the dialog can override it.
_ANCHOR_SIZE = 16.0
#: The active point's ring is drawn this much larger than the markers.
_ACTIVE_SIZE_RATIO = 22.0 / 16.0
_ACTIVE_EDGE = (1.0, 1.0, 1.0, 1.0)
_IDLE_EDGE = (0.0, 0.0, 0.0, 0.8)

#: The active marker is an outline only — a transparent fill with a hairline
#: edge. pygfx's "ring" marker draws a filled donut whose thickness is fixed, so
#: it swamped the keypoint underneath; a circle with no fill leaves just the edge.
_ACTIVE_FILL = (0.0, 0.0, 0.0, 0.0)
_ACTIVE_EDGE_WIDTH = 1.0

#: Alpha applied to the individuals that are not currently being edited.
_INACTIVE_ALPHA = 0.35

#: Predictions are drawn hollow — no interior, the point's colour on the edge —
#: so the pixels being judged stay visible and provenance costs no identity
#: channel. The edge is drawn a little heavier than a label's, since an outline
#: of the same width reads as fainter than a filled disc.
_FILL_INTERIOR = (0.0, 0.0, 0.0, 0.0)
_FILL_EDGE_WIDTH = 2.5

#: A detection is drawn as the hollow marker plus a solid pip at its centre —
#: a circle at this fraction of the marker size, in the point's colour. Small
#: enough to leave the pixels around the point visible, big enough to read at a
#: glance as "there is something inside this one".
_PIP_SIZE_RATIO = 0.3

#: Every marker is a circle: identity is carried by colour alone (see the module
#: docstring). Must exist in ``pygfx.utils.enums.MarkerShape``.
MARKER_SHAPE = "circle"

#: The interaction modes; see the module docstring.
#: Label every keypoint on one frame; the playhead never moves on its own.
SEQUENTIAL_MODE = "sequential"
#: Label one keypoint, then jump straight to the next frame.
LOOP_MODE = "loop"


def keypoint_colors(n: int, overrides: Sequence[str | None] | None = None) -> np.ndarray:
    """``(n, 4)`` RGBA, one distinct colour per name — the generated palette.

    Used for both axes: *n* keypoints or *n* individuals, whichever the display
    is colouring by. *overrides* aligns with that axis' names: a ``"#rrggbb"``
    entry pins that slot's colour and ``None`` leaves it on the palette. The
    palette is still sampled for the whole axis, so pinning one name never
    shifts the colours of the others — the two colours a user distinguishes are
    the one they chose and the one it used to be, not their neighbours'.
    """
    colors = sample_colormap(max(n, 1), "turbo")
    rgba = np.array([c if len(c) == 4 else (*c, 1.0) for c in colors], dtype=np.float32)[:n]
    for i, spec in enumerate(overrides or ()):
        if spec is not None and i < n:
            rgba[i] = _color_to_rgba(spec)
    return rgba


def keypoint_colors_for(store: KeypointStore) -> np.ndarray:
    """The colours *store*'s keypoints are drawn in, palette plus pinned ones.

    The single source for every surface that colours a keypoint — the canvas
    overlay, the dialog's tree and the points table header — so none of them can
    disagree about which colour the beak is.
    """
    return keypoint_colors(store.n_keypoints, store.keypoint_color_list())


def individual_colors_for(store: KeypointStore) -> np.ndarray:
    """The same, per individual — what colour-by-individual draws."""
    return keypoint_colors(store.n_individuals, store.individual_color_list())


def _color_to_rgba(spec: str) -> tuple[float, float, float, float]:
    """``"#rrggbb"`` -> RGBA floats, opaque; the store validates the spelling."""
    text = normalise_color(spec)
    return (int(text[1:3], 16) / 255.0, int(text[3:5], 16) / 255.0, int(text[5:7], 16) / 255.0, 1.0)


class AnchorOverlay:
    """pygfx markers + labels for the points of one frame.

    **Three** :class:`pygfx.Points` layers in total, one per provenance style:
    solid (a label), hollow (a prediction) and the pip that marks a hollow one a
    detector placed. An edge style is a material property rather than a
    per-vertex one, which is why provenance needs an object of its own; identity
    is per-vertex colour, so every individual shares these three layers and the
    vertex buffers are flat ``(n_individuals * n_keypoints)``.

    Colour follows ``color_by`` — per keypoint (shared across individuals) or
    per individual (shared across keypoints). Whichever mode is on, individuals
    other than the active one are drawn dimmed, keypoint names are drawn for the
    active individual only (with a full schema on several individuals the canvas
    is otherwise unreadable), and each individual's own name sits at the centroid
    of its points.
    """

    def __init__(
        self,
        scene: gfx.Scene,
        keypoint_names: list[str],
        individual_names: list[str],
        img_height: float,
        colors: np.ndarray | None = None,
        individual_colors: np.ndarray | None = None,
        color_by: str = COLOR_BY_KEYPOINT,
    ):
        self._scene = scene
        self._keypoint_names = list(keypoint_names)
        self._individual_names = list(individual_names)
        self._img_height = float(img_height)

        n_kp = len(self._keypoint_names)
        n_ind = len(self._individual_names)
        self._colors = keypoint_colors(n_kp) if colors is None else np.asarray(colors, dtype=np.float32)
        self._individual_colors = (
            keypoint_colors(n_ind) if individual_colors is None else np.asarray(individual_colors, dtype=np.float32)
        )
        self._color_by = color_by if color_by in COLOR_BY_MODES else COLOR_BY_KEYPOINT
        self._solid: gfx.Points | None = None
        self._hollow: gfx.Points | None = None
        self._pip: gfx.Points | None = None
        # No layers when there is nothing to draw: a zero-vertex geometry is not
        # worth defending against downstream.
        if n_kp and n_ind:
            self._solid = self._add_layer(n_ind * n_kp, human=True)
            self._hollow = self._add_layer(n_ind * n_kp, human=False)
            self._pip = self._add_pip_layer(n_ind * n_kp)

        # The active point is a separate single-vertex object so it can carry a
        # white outline; marker materials only expose one edge colour for all
        # vertices.
        self._active = gfx.Points(
            gfx.Geometry(positions=np.full((1, 3), _OFFSCREEN, dtype=np.float32)),
            gfx.PointsMarkerMaterial(
                size=_ANCHOR_SIZE * _ACTIVE_SIZE_RATIO,
                marker=MARKER_SHAPE,
                color=_ACTIVE_FILL,
                edge_width=_ACTIVE_EDGE_WIDTH,
                edge_color=_ACTIVE_EDGE,
            ),
        )
        self._active.local.z = _Z_ACTIVE
        scene.add(self._active)

        self._keypoint_texts = [self._add_text(name, 11) for name in self._keypoint_names]
        # With one individual its name says nothing the canvas does not already
        # show; with several it is what tells them apart at a glance.
        self._individual_texts = [self._add_text(name, 13) for name in self._individual_names] if n_ind > 1 else []

    @property
    def _layers(self) -> list[gfx.Points]:
        """The three marker layers, skipping the ones an empty schema left unbuilt."""
        return [layer for layer in (self._solid, self._hollow, self._pip) if layer is not None]

    def _add_layer(self, n_points: int, human: bool) -> gfx.Points:
        """One markers object for every point of one provenance style.

        *human* picks the style: a solid marker for a label, a hollow one for a
        prediction. The hollow layer carries its colours in ``edge_colors``
        rather than ``colors``, since that is the buffer the edge is drawn from
        once ``edge_color_mode`` is per-vertex.
        """
        colors = np.tile(np.float32([1.0, 1.0, 1.0, 1.0]), (n_points, 1))
        geometry = gfx.Geometry(
            positions=np.full((n_points, 3), _OFFSCREEN, dtype=np.float32),
            **({"colors": colors} if human else {"edge_colors": colors}),
        )
        shared = dict(
            size=_ANCHOR_SIZE,
            size_space="screen",  # constant on screen, unaffected by zoom
            marker=MARKER_SHAPE,
        )
        material = (
            gfx.PointsMarkerMaterial(color_mode="vertex", edge_width=2.0, edge_color=_IDLE_EDGE, **shared)
            if human
            else gfx.PointsMarkerMaterial(
                color=_FILL_INTERIOR,
                color_mode="uniform",
                edge_color_mode="vertex",
                edge_width=_FILL_EDGE_WIDTH,
                **shared,
            )
        )
        layer = gfx.Points(geometry, material)
        layer.local.z = _Z_ANCHORS
        self._scene.add(layer)
        return layer

    def _add_pip_layer(self, n_points: int) -> gfx.Points:
        """The centre dot marking a hollow marker as a *detection*."""
        geometry = gfx.Geometry(
            positions=np.full((n_points, 3), _OFFSCREEN, dtype=np.float32),
            colors=np.tile(np.float32([1.0, 1.0, 1.0, 1.0]), (n_points, 1)),
        )
        layer = gfx.Points(
            geometry,
            gfx.PointsMarkerMaterial(
                size=_ANCHOR_SIZE * _PIP_SIZE_RATIO,
                size_space="screen",
                marker=MARKER_SHAPE,
                color_mode="vertex",
                edge_width=0.0,
            ),
        )
        layer.local.z = _Z_ANCHORS
        self._scene.add(layer)
        return layer

    def _add_text(self, name: str, size: int) -> gfx.Text:
        text = gfx.Text(
            text=str(name),
            font_size=size,
            screen_space=True,
            anchor="bottom-left",
            material=gfx.TextMaterial(color=gfx.Color(1.0, 1.0, 1.0)),
        )
        text.local.z = _Z_TEXT
        text.visible = False
        self._scene.add(text)
        return text

    def vertex_colors(self, active_individual: int = 0) -> np.ndarray:
        """``(n_individuals * n_keypoints, 4)`` RGBA in buffer order.

        The whole colour model in one place: one hue per keypoint repeated down
        the individuals, or one hue per individual repeated across its
        keypoints, with every individual but the active one dimmed.
        """
        n_ind, n_kp = len(self._individual_names), len(self._keypoint_names)
        if self._color_by == COLOR_BY_INDIVIDUAL:
            cube = np.repeat(self._individual_colors[:, None, :], n_kp, axis=1).copy()
        else:
            cube = np.repeat(self._colors[None, :, :], n_ind, axis=0).copy()
        if n_ind > 1:
            dim = np.arange(n_ind) != active_individual
            cube[dim, :, 3] *= _INACTIVE_ALPHA
        return cube.reshape(-1, 4)

    def set_positions(
        self,
        positions: np.ndarray,
        human: np.ndarray,
        active_individual: int,
        active_keypoint: int,
        detected: np.ndarray | None = None,
    ) -> None:
        """Draw ``(n_individuals, n_keypoints, 2)`` image-space points.

        *positions* is every point on the frame — labels over detections over
        fill — and *human* / *detected* are the matching
        ``(n_individuals, n_keypoints)`` provenance masks, which decide whether a
        marker is drawn solid, hollow with a pip, or hollow. ``NaN`` hides it.
        *active_individual* / *active_keypoint* index the pair the next click
        will write to.
        """
        n_ind, n_kp = positions.shape[0], positions.shape[1]
        if detected is None:
            detected = np.zeros_like(human)
        if self._solid is None or n_kp == 0 or n_ind == 0:
            self._active.visible = False
            for layer in self._layers:
                layer.visible = False
            for text in [*self._keypoint_texts, *self._individual_texts]:
                text.visible = False
            return
        if (n_ind, n_kp) != (len(self._individual_names), len(self._keypoint_names)):
            raise ValueError(
                f"Pose overlay built for {len(self._individual_names)}x{len(self._keypoint_names)} "
                f"but asked to draw {n_ind}x{n_kp} — the schema changed without rebuilding the overlay."
            )

        world = positions.copy()
        world[:, :, 1] = self._img_height - world[:, :, 1]
        shown = ~np.isnan(positions[:, :, 0])

        colors = self.vertex_colors(active_individual)
        flat = world.reshape(-1, 2)
        self._draw_layer(self._solid, self._solid.geometry.colors, flat, (shown & human).ravel(), colors)
        self._draw_layer(self._hollow, self._hollow.geometry.edge_colors, flat, (shown & ~human).ravel(), colors)
        self._draw_layer(self._pip, self._pip.geometry.colors, flat, (shown & detected).ravel(), colors)

        active_buffer = self._active.geometry.positions
        active_shown = shown[active_individual, active_keypoint] if n_kp else False
        active_buffer.data[0, :2] = world[active_individual, active_keypoint] if active_shown else _OFFSCREEN
        active_buffer.data[0, 2] = 0.0
        active_buffer.update_full()
        self._active.visible = bool(active_shown)

        for k, text in enumerate(self._keypoint_texts):
            text.visible = bool(shown[active_individual, k])
            if text.visible:
                x, y = world[active_individual, k]
                text.local.position = (float(x) + 4, float(y) + 4, _Z_TEXT)

        for i, text in enumerate(self._individual_texts):
            visible = bool(np.any(shown[i]))
            text.visible = visible
            if visible:
                centre = np.nanmean(world[i][shown[i]], axis=0)
                text.local.position = (float(centre[0]) + 6, float(centre[1]) - 14, _Z_TEXT)

    def _draw_layer(
        self,
        layer: gfx.Points,
        color_buffer,
        world: np.ndarray,
        shown: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Position and tint the markers of one provenance style.

        *color_buffer* is whichever buffer that layer draws its colours from —
        ``colors`` for a solid marker, ``edge_colors`` for a hollow one — so a
        recolour is the same operation either way.
        """
        layer.visible = True
        positions = layer.geometry.positions
        positions.data[:, 2] = 0.0
        positions.data[:, :2] = np.where(shown[:, None], world, _OFFSCREEN)
        positions.update_full()

        color_buffer.data[:] = colors
        color_buffer.update_full()

    def set_colors(self, colors: np.ndarray, individual_colors: np.ndarray | None = None) -> None:
        """Recolour in place — the caller redraws the positions.

        Every layer re-uploads its colours on the next draw, so a colour change
        costs no rebuild: the geometries and materials are keyed by provenance,
        which colour does not touch.
        """
        self._colors = np.asarray(colors, dtype=np.float32)
        if individual_colors is not None:
            self._individual_colors = np.asarray(individual_colors, dtype=np.float32)

    def set_color_by(self, color_by: str) -> None:
        """Switch which axis colour encodes; same cost as a recolour."""
        if color_by not in COLOR_BY_MODES:
            raise ValueError(f"Unknown colour mode {color_by!r} — expected one of {COLOR_BY_MODES}.")
        self._color_by = color_by

    def set_point_size(self, size: float) -> None:
        """Resize every marker in place — no rebuild, so a spinbox can drive it."""
        for layer in (self._solid, self._hollow):
            if layer is not None:
                layer.material.size = float(size)
        if self._pip is not None:
            self._pip.material.size = float(size) * _PIP_SIZE_RATIO
        self._active.material.size = float(size) * _ACTIVE_SIZE_RATIO

    def clear(self) -> None:
        for obj in [
            *self._layers,
            self._active,
            *self._keypoint_texts,
            *self._individual_texts,
        ]:
            self._scene.remove(obj)
        self._solid = None
        self._hollow = None
        self._pip = None
        self._keypoint_texts = []
        self._individual_texts = []


class KeypointLabelMode:
    """Pointer-driven anchor editing on one camera view."""

    def __init__(
        self,
        view,
        store: KeypointStore,
        on_changed: Callable[[], None] | None = None,
        mode: str = SEQUENTIAL_MODE,
        on_advance_frame: Callable[[], None] | None = None,
        on_released: Callable[[], None] | None = None,
        point_size: float = _ANCHOR_SIZE,
        locked: bool = False,
        color_by: str = COLOR_BY_KEYPOINT,
    ):
        self.view = view
        self.store = store
        self.on_changed = on_changed or (lambda: None)
        #: Called by LOOP mode after a placement — the dialog owns navigation.
        self.on_advance_frame = on_advance_frame or (lambda: None)
        #: Called when the pointer is released, so work too heavy for every
        #: mouse move of a drag can happen once the edit has settled.
        self.on_released = on_released or (lambda: None)
        self.mode = mode
        #: While True the pointer does nothing here and pans the view instead;
        #: read by the view when it binds left-drag (see ``set_label_locked``).
        self._locked = bool(locked)
        self.frame = 0
        #: The active pair is held by name, not by index: with per-individual
        #: keypoint sets an index means different things on different branches,
        #: and with no individuals at all it means nothing.
        self._individual: str | None = None
        self._keypoint: str | None = None
        self._dragging: tuple[str, str] | None = None
        #: True once the current drag has written a point, so further motion
        #: collapses into that single undo step instead of popping an unrelated one.
        self._drag_recorded = False
        self._cursor: tuple[float, float] | None = None

        scene = view.scene()
        if scene is None:
            raise ValueError("Camera view has no scene to draw annotations on.")
        self._overlay = AnchorOverlay(
            scene,
            store.keypoint_names,
            store.individual_names,
            view.image_height(),
            colors=keypoint_colors_for(store),
            individual_colors=individual_colors_for(store),
            color_by=color_by,
        )
        self.point_size = float(point_size)
        self._overlay.set_point_size(self.point_size)
        view.set_label_mode(self)
        self._sync_active()
        self.refresh()

    # ------------------------------------------------------------------
    # Active individual + keypoint
    # ------------------------------------------------------------------

    @property
    def active_individual(self) -> str | None:
        """The individual clicks write to, or ``None`` when there are none."""
        return self._individual

    @property
    def active_keypoints(self) -> list[str]:
        """The schema of the active individual — what ``Tab`` cycles through."""
        return [] if self._individual is None else self.store.keypoints_for(self._individual)

    @property
    def active_keypoint(self) -> str | None:
        return self._keypoint

    def _sync_active(self) -> None:
        """Keep the active pair pointing at something the schema still holds."""
        if self._individual not in self.store.individual_names:
            self._individual = self.store.individual_names[0] if self.store.individual_names else None
        keypoints = self.active_keypoints
        if self._keypoint not in keypoints:
            self._keypoint = keypoints[0] if keypoints else None

    def set_active(self, keypoint: str, individual: str | None = None) -> None:
        if individual is not None:
            self._individual = self.store.individual_names[self.store.individual_index(individual)]
        self._keypoint = self.store.keypoint_names[self.store.keypoint_index(keypoint)]
        self._sync_active()
        self.refresh()

    def set_active_individual(self, individual: str) -> None:
        self._individual = self.store.individual_names[self.store.individual_index(individual)]
        self._sync_active()
        self.refresh()

    def cycle(self, step: int = 1) -> None:
        """Move to the next/previous keypoint of the active individual."""
        keypoints = self.active_keypoints
        if not keypoints:
            return
        index = keypoints.index(self._keypoint) if self._keypoint in keypoints else 0
        self._keypoint = keypoints[(index + step) % len(keypoints)]
        self.refresh()

    def cycle_individual(self, step: int = 1) -> None:
        names = self.store.individual_names
        if not names:
            return
        index = names.index(self._individual) if self._individual in names else 0
        self._individual = names[(index + step) % len(names)]
        self._sync_active()
        self.refresh()

    def select_individual_by_number(self, number: int) -> bool:
        """Select the *number*-th individual (1-based, SLEAP's number keys)."""
        if not 1 <= number <= self.store.n_individuals:
            return False
        self._individual = self.store.individual_names[number - 1]
        self._sync_active()
        self.refresh()
        return True

    def _advance_to_unlabelled(self) -> None:
        """Move to the FIRST keypoint this individual still lacks on this frame.

        Ordered by the schema, which is exactly the left-to-right column order of
        the dialog's points table — so sequential labelling fills that table from
        the left, and jumping around out of order still comes back to the
        leftmost gap rather than continuing from wherever the last click landed.

        When the frame is complete the active keypoint stays put: there is
        nothing left to advance to, and clicking an existing point grabs it
        rather than overwriting, so nothing is at risk.
        """
        keypoints = self.active_keypoints
        if not keypoints:
            return
        placed = self.store.anchor_positions_for(self.frame, self._individual)[:, 0]
        for name in sorted(keypoints, key=self.store.keypoint_index):
            if np.isnan(placed[self.store.keypoint_index(name)]):
                self._keypoint = name
                return

    # ------------------------------------------------------------------
    # Pointer handling (called by CameraView)
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch what happens after a point is placed."""
        if mode not in (SEQUENTIAL_MODE, LOOP_MODE):
            raise ValueError(f"Unknown labelling mode {mode!r}")
        self.mode = mode
        self._dragging = None
        self._drag_recorded = False
        self.refresh()

    @property
    def locked(self) -> bool:
        """Whether the pointer pans the view instead of labelling."""
        return self._locked

    def set_locked(self, locked: bool) -> None:
        """Suspend (or resume) pointer labelling, keeping everything else.

        The overlay and the active pair survive, so unlocking carries on exactly
        where labelling left off; the view hands left-drag back to its pan
        controller for as long as the lock is on.
        """
        self._locked = bool(locked)
        self._dragging = None
        self._drag_recorded = False
        self.view.set_label_locked(self._locked)
        self.refresh()

    def handle_click(self, x: float, y: float) -> None:
        """Place the active keypoint, or grab an existing point to move it.

        Correcting is always available — clicking a point that is already there
        selects and drags it rather than dropping a second one on top. Filled
        points count as "there": grabbing one **pins it** as a label where the
        backend put it, so a prediction that is already right is accepted by
        clicking it and a wrong one is fixed by dragging it. Either way the next
        fill sees a human point, since fills are re-derived from labels alone.

        Locked, the pointer belongs to the camera: nothing is placed, grabbed or
        pinned, and the drag pans instead.
        """
        if self._locked:
            return
        self._cursor = (x, y)
        existing = self.store.nearest(self.frame, (x, y), self._hit_radius(), include_fill=True)
        if existing is not None:
            individual, keypoint = existing
            self._individual, self._keypoint = existing
            self._dragging = existing
            self._drag_recorded = False
            if not self.store.is_anchor(self.frame, keypoint, individual):
                position = self.store.positions_for(self.frame, individual)[self.store.keypoint_index(keypoint)]
                self.store.set_point(self.frame, keypoint, tuple(position), individual)
                # Recorded, so dragging on from here collapses into this one
                # undo step rather than popping something unrelated.
                self._drag_recorded = True
            self._changed()
            return

        keypoint, individual = self.active_keypoint, self._individual
        if keypoint is None or individual is None:
            return
        self.store.set_point(self.frame, keypoint, (x, y), individual)
        self._dragging = (individual, keypoint)
        self._drag_recorded = True
        if self.mode == LOOP_MODE:
            # Same keypoint, next frame. The dialog owns navigation, so it
            # decides whether "next" means the next suggestion or the next frame.
            self._changed()
            self.on_advance_frame()
            return
        self._advance_to_unlabelled()
        self._changed()

    def handle_move(self, x: float, y: float) -> None:
        if self._locked:
            return
        self._cursor = (x, y)
        if self._dragging is None:
            return
        if self._drag_recorded:
            self.store.undo()  # collapse the whole drag into one undo step
        individual, keypoint = self._dragging
        self.store.set_point(self.frame, keypoint, (x, y), individual)
        self._drag_recorded = True
        self._changed()

    @property
    def dragging(self) -> bool:
        """Whether a point is currently being moved by the pointer."""
        return self._dragging is not None

    def handle_release(self, x: float, y: float) -> None:
        if self._locked:
            return
        self._cursor = (x, y)
        self._dragging = None
        self._drag_recorded = False
        self.on_released()

    def delete_selected(self) -> bool:
        """What ``Backspace`` removes: the active point, else the one hovered.

        The active point is what the white outline is drawn around, so deleting
        it is what the canvas already promises. Falling back to the cursor keeps
        hover-and-delete working for points that are not the active one, and
        matters most when the active pair is unlabelled on this frame.
        """
        return self.delete_active() or self.delete_under_cursor()

    def delete_active(self) -> bool:
        """Delete the active ``(individual, keypoint)`` on this frame."""
        if self._individual is None or self._keypoint is None:
            return False
        placed = self.store.anchor_positions_for(self.frame, self._individual)
        if np.isnan(placed[self.store.keypoint_index(self._keypoint), 0]):
            return False
        self.store.clear_point(self.frame, self._keypoint, self._individual)
        self._changed()
        return True

    def delete_under_cursor(self) -> bool:
        """Delete the anchor nearest the cursor; ``True`` if one was removed."""
        if self._cursor is None:
            return False
        target = self.store.nearest(self.frame, self._cursor, self._hit_radius())
        if target is None:
            return False
        individual, keypoint = target
        self.store.clear_point(self.frame, keypoint, individual)
        self._changed()
        return True

    def set_point_size(self, size: float) -> None:
        """Change the marker size; also widens hit-testing (see _hit_radius)."""
        self.point_size = float(size)
        self._overlay.set_point_size(self.point_size)
        self.view.request_draw()

    def _hit_radius(self) -> float:
        """Grab radius in image units.

        Both terms are screen pixels — the markers are sized in screen space —
        converted through the current zoom, so the radius always matches what
        the user sees. It never shrinks below :data:`HIT_RADIUS_PX`, and grows
        with the marker so a click inside a large one selects it.
        """
        screen_px = max(HIT_RADIUS_PX, self.point_size / 2.0)
        return screen_px * self.view.image_units_per_pixel()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def set_frame(self, frame: int) -> None:
        self.frame = int(frame)
        self._dragging = None
        self._drag_recorded = False
        self.refresh()

    def refresh(self) -> None:
        """Redraw the frame's points — solid, pipped or hollow by provenance."""
        self._sync_active()
        self._overlay.set_positions(
            self.store.positions(self.frame),
            self.store.human_mask(self.frame),
            self.store.individual_index(self._individual) if self._individual is not None else 0,
            self.store.keypoint_index(self._keypoint) if self._keypoint is not None else 0,
            detected=self.store.detected_mask(self.frame),
        )
        self.view.request_draw()

    def refresh_colors(self) -> None:
        """Re-read the store's colours and redraw.

        A colour change is not a schema change: the layers still match the
        hierarchy, so the mode is recoloured rather than restarted (which would
        drop the active pair and rebuild every pygfx object).
        """
        self._overlay.set_colors(keypoint_colors_for(self.store), individual_colors_for(self.store))
        self.refresh()

    def set_color_by(self, color_by: str) -> None:
        """Switch which axis colour encodes — keypoint or individual."""
        self._overlay.set_color_by(color_by)
        self.refresh()

    def _changed(self) -> None:
        self.refresh()
        self.on_changed()

    def detach(self) -> None:
        self._overlay.clear()
        self.view.set_label_mode(None)
        self.view.request_draw()
