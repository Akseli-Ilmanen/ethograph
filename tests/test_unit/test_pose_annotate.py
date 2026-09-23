"""KeypointStore editing, NaN handling and export round-trips.

The store is hierarchical — individuals × keypoints — so every test here reads
positions as ``(n_individuals, n_keypoints, 2)``. Single-individual labelling is
the ``n_individuals == 1`` case and gets the ``individual=None`` default.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ethograph.gui.pose_annotate import (
    KINEMATICS,
    KeypointStore,
    KeypointStoreError,
    UnknownIndividualError,
    UnknownKeypointError,
    sidecar_path,
    store_to_dataset,
    store_to_head_direction,
    store_to_kinematics,
    store_to_movement_ds,
)

NAMES = ["beak", "tail", "eye"]
FPS = 25.0


@pytest.fixture
def store() -> KeypointStore:
    return KeypointStore(keypoint_names=list(NAMES), n_frames=10)


@pytest.fixture
def pair() -> KeypointStore:
    """Two individuals sharing one keypoint schema."""
    return KeypointStore(keypoint_names=list(NAMES), n_frames=10, individual_names=["crow_a", "crow_b"])


def test_defaults_to_one_individual(store):
    assert store.individual_names == ["individual_0"]
    assert store.n_points == len(NAMES)


def test_set_and_read_point(store):
    store.set_point(3, "beak", (12.5, 7.0))
    assert store.anchor_frames() == [3]
    np.testing.assert_allclose(store.positions_for(3)[0], [12.5, 7.0])
    assert np.all(np.isnan(store.positions_for(3)[1:]))


def test_unknown_keypoint_raises(store):
    with pytest.raises(UnknownKeypointError):
        store.set_point(0, "wing", (1.0, 2.0))


def test_unknown_individual_raises(store):
    with pytest.raises(UnknownIndividualError):
        store.set_point(0, "beak", (1.0, 2.0), individual="nobody")


def test_partial_anchors_are_per_keypoint(store):
    store.set_point(1, "beak", (1.0, 1.0))
    store.set_point(5, "tail", (2.0, 2.0))
    assert store.anchor_frames() == [1, 5]
    assert store.anchor_frames_for("beak") == [1]
    assert store.anchor_frames_for("tail") == [5]
    assert store.anchor_frames_for("eye") == []


def test_clear_point_drops_empty_anchor(store):
    store.set_point(2, "beak", (1.0, 1.0))
    store.clear_point(2, "beak")
    assert store.anchor_frames() == []


def test_clear_point_keeps_partly_labelled_anchor(store):
    store.set_point(2, "beak", (1.0, 1.0))
    store.set_point(2, "tail", (3.0, 4.0))
    store.clear_point(2, "beak")
    assert store.anchor_frames() == [2]
    assert np.isnan(store.positions_for(2)[0, 0])
    np.testing.assert_allclose(store.positions_for(2)[1], [3.0, 4.0])


def test_undo_restores_previous_value(store):
    store.set_point(4, "beak", (1.0, 1.0))
    store.set_point(4, "beak", (9.0, 9.0))
    store.undo()
    np.testing.assert_allclose(store.positions_for(4)[0], [1.0, 1.0])
    store.undo()
    assert store.anchor_frames() == []


def test_undo_of_clear_restores_point(store):
    store.set_point(4, "tail", (5.0, 6.0))
    store.clear_point(4, "tail")
    store.undo()
    np.testing.assert_allclose(store.positions_for(4)[1], [5.0, 6.0])


def test_undo_on_empty_history_is_a_noop(store):
    store.undo()
    assert store.anchor_frames() == []


def test_nearest_respects_radius(store):
    store.set_point(0, "beak", (10.0, 10.0))
    assert store.nearest(0, (11.0, 10.0), radius=3.0) == ("individual_0", "beak")
    assert store.nearest(0, (30.0, 10.0), radius=3.0) is None
    assert store.nearest(7, (10.0, 10.0), radius=3.0) is None


def test_set_keypoint_names_carries_points_and_clears_fill(store):
    store.set_point(0, "beak", (1.0, 2.0))
    store.set_point(0, "tail", (3.0, 4.0))
    store.set_fill(np.zeros((10, 1, 3, 2)), np.zeros((10, 1, 3)))

    store.set_keypoint_names(["tail", "beak"])

    assert store.keypoint_names == ["tail", "beak"]
    np.testing.assert_allclose(store.positions_for(0)[0], [3.0, 4.0])
    np.testing.assert_allclose(store.positions_for(0)[1], [1.0, 2.0])
    assert store.filled is None


def test_set_keypoint_names_drops_removed_keypoints(store):
    store.set_point(0, "eye", (1.0, 1.0))
    store.set_keypoint_names(["beak", "tail"])
    assert store.anchor_frames() == []


def test_set_fill_reasserts_anchors(store):
    store.set_point(2, "beak", (100.0, 200.0))
    store.set_fill(np.zeros((10, 1, 3, 2)), np.full((10, 1, 3), 0.5))

    np.testing.assert_allclose(store.filled[2, 0, 0], [100.0, 200.0])
    assert store.confidence[2, 0, 0] == 1.0
    # Untouched keypoints keep the backend's values.
    np.testing.assert_allclose(store.filled[2, 0, 1], [0.0, 0.0])
    assert store.confidence[2, 0, 1] == 0.5


def test_fill_range_reports_what_the_fill_covered(store):
    """Backends fill the span between labels, and readers must not run past it."""
    assert store.fill_range is None

    filled = np.full((10, 1, 3, 2), np.nan)
    filled[3:8] = 1.0
    store.set_fill(filled, np.full((10, 1, 3), 0.5))
    assert store.fill_range == (3, 7)

    store.clear_fill()
    assert store.fill_range is None


def test_set_fill_rejects_wrong_shape(store):
    with pytest.raises(KeypointStoreError):
        store.set_fill(np.zeros((5, 1, 3, 2)), np.zeros((5, 1, 3)))


def test_positions_prefers_anchor_over_fill(store):
    store.set_fill(np.ones((10, 1, 3, 2)), np.ones((10, 1, 3)))
    store.set_point(6, "eye", (42.0, 43.0))
    np.testing.assert_allclose(store.positions_for(6)[2], [42.0, 43.0])
    np.testing.assert_allclose(store.positions_for(6)[0], [1.0, 1.0])


def test_sidecar_round_trip(store, tmp_path):
    store.set_point(1, "beak", (1.5, 2.5))
    store.set_point(8, "tail", (3.5, 4.5))
    path = tmp_path / "clip.mp4.keypoints.json"
    store.save(path)

    restored = KeypointStore.load(path)
    assert restored.keypoint_names == store.keypoint_names
    assert restored.individual_names == store.individual_names
    assert restored.n_frames == store.n_frames
    assert restored.anchor_frames() == [1, 8]
    np.testing.assert_allclose(restored.positions_for(1)[0], [1.5, 2.5])


def test_sidecar_path_sits_next_to_the_video(tmp_path):
    assert sidecar_path(tmp_path / "clip.mp4").name == "clip.mp4.keypoints.json"


def test_multi_individual_sidecar_round_trip(pair, tmp_path):
    pair.set_point(2, "beak", (1.0, 2.0), "crow_a")
    pair.set_point(2, "tail", (3.0, 4.0), "crow_b")
    path = tmp_path / "clip.mp4.keypoints.json"
    pair.save(path)

    restored = KeypointStore.load(path)
    assert restored.individual_names == ["crow_a", "crow_b"]
    np.testing.assert_allclose(restored.positions_for(2, "crow_a")[0], [1.0, 2.0])
    np.testing.assert_allclose(restored.positions_for(2, "crow_b")[1], [3.0, 4.0])


# ----------------------------------------------------------------------
# Legacy sidecars written before the individual/keypoint hierarchy
# ----------------------------------------------------------------------


def _legacy_payload() -> dict:
    """A pre-hierarchy sidecar: "names" key, 2-D (n_keypoints, 2) anchors."""
    return {
        "names": ["male", "female"],
        "n_frames": 2861,
        "anchors": {
            "89": [[510.7, 442.8], [464.8, 500.0]],
            "173": [[476.0, 522.4], [441.2, 640.0]],
        },
    }


def test_legacy_flat_sidecar_loads(tmp_path):
    """Regression: a flat sidecar raised KeyError('keypoint') and broke the dialog."""
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    restored = KeypointStore.load(path)

    assert restored.keypoint_names == ["male", "female"]
    assert restored.individual_names == ["individual_0"]
    assert restored.n_frames == 2861
    assert restored.anchor_frames() == [89, 173]


def test_legacy_anchors_gain_an_individual_axis(tmp_path):
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    restored = KeypointStore.load(path)

    assert restored.anchors[89].shape == (1, 2, 2)
    np.testing.assert_allclose(restored.positions_for(89)[0], [510.7, 442.8])
    np.testing.assert_allclose(restored.positions_for(89)[1], [464.8, 500.0])


def test_legacy_names_stay_keypoints(tmp_path):
    """Never silently reinterpret old names as individuals — that would change
    what the user labelled."""
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    restored = KeypointStore.load(path)

    assert restored.n_individuals == 1
    assert restored.n_keypoints == 2


def test_legacy_sidecar_resaves_in_the_new_format(tmp_path):
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    KeypointStore.load(path).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "names" not in payload
    assert payload["keypoint"] == ["male", "female"]
    assert payload["individual"] == ["individual_0"]


def test_sidecar_without_names_raises_clearly(tmp_path):
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps({"n_frames": 10, "anchors": {}}), encoding="utf-8")
    with pytest.raises(KeypointStoreError, match="keypoint names"):
        KeypointStore.load(path)


def test_mismatched_anchor_shape_raises(tmp_path):
    """A truncated/corrupt anchor row must fail loudly, not load half a frame."""
    payload = _legacy_payload()
    payload["anchors"]["89"] = [[1.0, 2.0]]  # one keypoint, schema says two
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KeypointStoreError, match="expected"):
        KeypointStore.load(path)


# ----------------------------------------------------------------------
# Hierarchy: several individuals sharing one keypoint schema
# ----------------------------------------------------------------------


def test_individuals_are_independent(pair):
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(0, "beak", (50.0, 50.0), "crow_b")

    np.testing.assert_allclose(pair.positions_for(0, "crow_a")[0], [1.0, 1.0])
    np.testing.assert_allclose(pair.positions_for(0, "crow_b")[0], [50.0, 50.0])
    assert pair.labelled_count(0) == 2
    assert pair.labelled_count(0, "crow_a") == 1


def test_anchor_frames_for_is_per_individual(pair):
    pair.set_point(1, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(4, "beak", (2.0, 2.0), "crow_b")
    assert pair.anchor_frames() == [1, 4]
    assert pair.anchor_frames_for("beak", "crow_a") == [1]
    assert pair.anchor_frames_for("beak", "crow_b") == [4]


def test_nearest_reports_the_individual(pair):
    pair.set_point(0, "tail", (10.0, 10.0), "crow_b")
    assert pair.nearest(0, (10.5, 10.0), radius=3.0) == ("crow_b", "tail")


def test_clear_individual_leaves_the_others_alone(pair):
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(0, "tail", (2.0, 2.0), "crow_a")
    pair.set_point(0, "beak", (3.0, 3.0), "crow_b")

    pair.clear_individual(0, "crow_a")

    assert pair.labelled_count(0, "crow_a") == 0
    np.testing.assert_allclose(pair.positions_for(0, "crow_b")[0], [3.0, 3.0])


def test_adding_an_individual_keeps_existing_points(store):
    store.set_point(0, "beak", (7.0, 8.0))
    store.set_individual_names(["individual_0", "individual_1"])

    assert store.n_points == 2 * len(NAMES)
    np.testing.assert_allclose(store.positions_for(0, "individual_0")[0], [7.0, 8.0])
    assert np.all(np.isnan(store.positions_for(0, "individual_1")))


def test_removing_an_individual_drops_its_points(pair):
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(0, "beak", (2.0, 2.0), "crow_b")

    pair.set_individual_names(["crow_a"])

    assert pair.n_individuals == 1
    np.testing.assert_allclose(pair.positions_for(0, "crow_a")[0], [1.0, 1.0])


def test_removing_every_individual_is_allowed(pair):
    """The dialog lets the last individual go; the store must not resurrect one."""
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")

    pair.set_individual_names([])

    assert pair.individual_names == []
    assert pair.n_individuals == 0
    assert pair.n_points == 0
    assert pair.anchor_frames() == []
    assert pair.positions(0).shape == (0, len(NAMES), 2)


def test_labelling_without_an_individual_raises(store):
    store.set_individual_names([])
    with pytest.raises(UnknownIndividualError):
        store.set_point(0, "beak", (1.0, 1.0))


def test_empty_individuals_survive_the_sidecar(store, tmp_path):
    store.set_individual_names([])
    path = tmp_path / "clip.mp4.keypoints.json"
    store.save(path)

    assert KeypointStore.load(path).individual_names == []


def test_flat_anchors_round_trip_through_a_fill(pair):
    pair.set_point(0, "beak", (1.0, 2.0), "crow_a")
    pair.set_point(0, "eye", (3.0, 4.0), "crow_b")

    flat = pair.flat_anchors()
    assert flat[0].shape == (pair.n_points, 2)
    np.testing.assert_allclose(flat[0][0], [1.0, 2.0])  # crow_a / beak
    np.testing.assert_allclose(flat[0][5], [3.0, 4.0])  # crow_b / eye

    filled = np.zeros((pair.n_frames, pair.n_points, 2))
    pair.set_fill_from_flat(filled, np.zeros((pair.n_frames, pair.n_points)))

    assert pair.filled.shape == (pair.n_frames, 2, len(NAMES), 2)
    np.testing.assert_allclose(pair.filled[0, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(pair.filled[0, 1, 2], [3.0, 4.0])


# ----------------------------------------------------------------------
# Reading points back: the dialog's table
# ----------------------------------------------------------------------


def test_labelled_points_are_sorted_by_frame(pair):
    pair.set_point(7, "tail", (3.0, 4.0), "crow_b")
    pair.set_point(2, "beak", (1.0, 2.0), "crow_a")

    assert pair.labelled_points() == [
        (2, "crow_a", "beak", 1.0, 2.0),
        (7, "crow_b", "tail", 3.0, 4.0),
    ]


def test_labelled_points_can_be_filtered(pair):
    pair.set_point(0, "beak", (1.0, 1.0), "crow_a")
    pair.set_point(0, "tail", (2.0, 2.0), "crow_a")
    pair.set_point(0, "beak", (3.0, 3.0), "crow_b")

    assert [row[1:3] for row in pair.labelled_points(individual="crow_a")] == [
        ("crow_a", "beak"),
        ("crow_a", "tail"),
    ]
    assert [row[1] for row in pair.labelled_points(keypoint="beak")] == ["crow_a", "crow_b"]


def test_undo_reports_the_frame_it_changed(store):
    """An undo can land anywhere, so the caller is told what to redraw."""
    store.set_point(4, "beak", (1.0, 1.0))
    store.set_point(9, "tail", (2.0, 2.0))

    assert store.undo() == 9
    assert store.undo() == 4
    assert store.undo() is None  # nothing left in the history


# ----------------------------------------------------------------------
# Provenance: human labels vs filled predictions
# ----------------------------------------------------------------------


def _fill(store: KeypointStore, value: float = 5.0) -> None:
    shape = (store.n_frames, store.n_individuals, store.n_keypoints)
    store.set_fill(np.full((*shape, 2), value), np.full(shape, 0.5))


def test_human_mask_marks_only_placed_points(store):
    store.set_point(2, "tail", (1.0, 2.0))
    _fill(store)

    assert store.human_mask(2).tolist() == [[False, True, False]]
    assert not store.human_mask(3).any()  # frame 3 is entirely the backend's


def test_is_human_needs_one_point_anywhere_in_the_row(pair):
    _fill(pair)
    pair.set_point(2, "tail", (1.0, 2.0), "crow_b")

    assert pair.is_human(2, "crow_b") is True
    assert pair.is_human(2, "crow_a") is False
    assert pair.is_human(2) is True  # any individual counts for the frame


def test_is_anchor_distinguishes_a_label_from_a_fill(store):
    store.set_point(2, "tail", (1.0, 2.0))
    _fill(store)

    assert store.is_anchor(2, "tail") is True
    assert store.is_anchor(2, "beak") is False  # filled, not labelled
    assert not np.isnan(store.positions_for(2)[0, 0])  # but it does have a position


def test_nearest_ignores_filled_points_unless_asked(store):
    _fill(store, value=10.0)

    assert store.nearest(0, (10.0, 10.0), radius=3.0) is None
    assert store.nearest(0, (10.0, 10.0), radius=3.0, include_fill=True) == ("individual_0", "beak")


def test_promote_fill_turns_predictions_into_labels(store):
    _fill(store, value=7.0)

    assert store.promote_fill(3) == len(NAMES)
    assert store.is_human(3) is True
    np.testing.assert_allclose(store.anchor_positions(3)[0, 1], [7.0, 7.0])


def test_promote_fill_never_overwrites_a_label(store):
    store.set_point(3, "beak", (1.0, 2.0))
    _fill(store, value=7.0)

    store.promote_fill(3)

    np.testing.assert_allclose(store.anchor_positions(3)[0, 0], [1.0, 2.0])


def test_promote_fill_can_take_one_individual(pair):
    _fill(pair)

    pair.promote_fill(3, "crow_b")

    assert pair.is_human(3, "crow_b") is True
    assert pair.is_human(3, "crow_a") is False


def test_clear_fill_for_removes_one_rows_predictions(store):
    """The counterpart of pinning: the animal is not in this frame at all."""
    store.set_point(3, "beak", (1.0, 2.0))
    _fill(store, value=7.0)

    assert store.clear_fill_for(4) == len(NAMES)

    assert store.has_fill(4) is False
    assert np.all(np.isnan(store.positions(4)))
    assert store.has_fill(5) is True  # its neighbours are untouched


def test_clear_fill_for_keeps_the_labels(store):
    store.set_point(3, "beak", (1.0, 2.0))
    _fill(store, value=7.0)

    store.clear_fill_for(3)

    np.testing.assert_allclose(store.positions_for(3)[0], [1.0, 2.0])
    assert store.is_human(3) is True
    assert np.all(np.isnan(store.positions_for(3)[1]))  # the predicted ones are gone


def test_clear_fill_for_can_take_one_individual(pair):
    _fill(pair)

    pair.clear_fill_for(3, "crow_b")

    assert pair.has_fill(3, "crow_b") is False
    assert pair.has_fill(3, "crow_a") is True


def test_clear_fill_for_without_a_fill_does_nothing(store):
    assert store.clear_fill_for(3) == 0
    assert store.has_fill(3) is False


def test_promote_fill_without_a_fill_does_nothing(store):
    assert store.promote_fill(3) == 0
    assert store.anchor_frames() == []


def test_refilling_never_feeds_on_the_previous_fill(store):
    """The invariant behind the Human/Fill split: fills come from labels only."""
    store.set_point(2, "beak", (1.0, 2.0))
    _fill(store, value=7.0)

    anchors = store.flat_anchors()
    assert list(anchors) == [2]  # not one entry per filled frame
    np.testing.assert_allclose(anchors[2][0], [1.0, 2.0])


# ----------------------------------------------------------------------
# Asymmetric schemas: individuals with their own keypoints
# ----------------------------------------------------------------------


def test_keypoints_are_shared_by_default(pair):
    assert pair.shared_keypoints is True
    assert pair.keypoints_for("crow_a") == NAMES
    assert pair.n_schema_points == pair.n_points


def test_unsharing_starts_everyone_on_the_full_schema(pair):
    pair.set_shared_keypoints(False)
    assert pair.keypoints_for("crow_a") == NAMES
    assert pair.keypoints_for("crow_b") == NAMES
    assert pair.n_schema_points == pair.n_points


def test_unsharing_keeps_labelled_points(pair):
    pair.set_point(0, "beak", (1.0, 2.0), "crow_a")
    pair.set_shared_keypoints(False)
    np.testing.assert_allclose(pair.positions_for(0, "crow_a")[0], [1.0, 2.0])


def test_set_keypoints_for_narrows_one_individual(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])

    assert pair.keypoints_for("crow_a") == ["beak"]
    assert pair.keypoints_for("crow_b") == NAMES
    assert pair.keypoint_names == NAMES  # the union is untouched
    assert pair.n_schema_points == 1 + len(NAMES)


def test_set_keypoints_for_adds_new_names_to_the_union_only(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_b", [*NAMES, "wing"])

    assert pair.keypoint_names == [*NAMES, "wing"]
    assert pair.keypoints_for("crow_a") == NAMES
    assert pair.keypoints_for("crow_b") == [*NAMES, "wing"]


def test_narrowing_drops_that_individuals_points(pair):
    pair.set_point(0, "eye", (1.0, 1.0), "crow_a")
    pair.set_point(0, "eye", (2.0, 2.0), "crow_b")
    pair.set_shared_keypoints(False)

    pair.set_keypoints_for("crow_a", ["beak", "tail"])

    assert np.all(np.isnan(pair.positions_for(0, "crow_a")))
    np.testing.assert_allclose(pair.positions_for(0, "crow_b")[2], [2.0, 2.0])


def test_labelling_a_keypoint_the_individual_lacks_raises(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])

    with pytest.raises(UnknownKeypointError):
        pair.set_point(0, "tail", (1.0, 1.0), "crow_a")
    pair.set_point(0, "tail", (1.0, 1.0), "crow_b")  # still fine for the other


def test_set_keypoints_for_refuses_while_shared(pair):
    with pytest.raises(KeypointStoreError):
        pair.set_keypoints_for("crow_a", ["beak"])


def test_fill_blanks_points_outside_an_individuals_schema(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])
    pair.set_fill_from_flat(
        np.ones((pair.n_frames, pair.n_points, 2)),
        np.full((pair.n_frames, pair.n_points), 0.5),
    )

    assert np.all(np.isnan(pair.filled[:, 0, 1:]))  # crow_a: tail, eye
    assert np.all(np.isnan(pair.confidence[:, 0, 1:]))
    np.testing.assert_allclose(pair.filled[:, 0, 0], 1.0)
    np.testing.assert_allclose(pair.filled[:, 1], 1.0)


def test_asymmetric_schema_survives_the_sidecar(pair, tmp_path):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])
    pair.set_point(0, "beak", (1.0, 2.0), "crow_a")
    path = tmp_path / "clip.mp4.keypoints.json"
    pair.save(path)

    restored = KeypointStore.load(path)

    assert restored.shared_keypoints is False
    assert restored.keypoints_for("crow_a") == ["beak"]
    assert restored.keypoints_for("crow_b") == NAMES
    np.testing.assert_allclose(restored.positions_for(0, "crow_a")[0], [1.0, 2.0])


def test_a_new_individual_gets_the_whole_schema(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])
    pair.set_individual_names([*pair.individual_names, "crow_c"])

    assert pair.keypoints_for("crow_c") == NAMES
    assert pair.keypoints_for("crow_a") == ["beak"]


def test_resharing_readmits_every_keypoint(pair):
    pair.set_shared_keypoints(False)
    pair.set_keypoints_for("crow_a", ["beak"])
    pair.set_shared_keypoints(True)

    assert pair.keypoints_for("crow_a") == NAMES
    assert pair.keypoint_sets == {}
    pair.set_point(0, "eye", (1.0, 1.0), "crow_a")


# ----------------------------------------------------------------------
# Keypoint colours
# ----------------------------------------------------------------------


def test_keypoints_start_on_the_palette(store):
    assert store.keypoint_color == {}
    assert store.color_for("beak") is None
    assert store.keypoint_color_list() == [None, None, None]


def test_a_pinned_colour_is_normalised(store):
    store.set_keypoint_color("beak", "#FF8000")

    assert store.color_for("beak") == "#ff8000"
    assert store.keypoint_color_list() == ["#ff8000", None, None]


def test_a_colour_that_is_not_a_colour_raises(store):
    with pytest.raises(KeypointStoreError):
        store.set_keypoint_color("beak", "orange")


def test_colouring_an_unknown_keypoint_raises(store):
    with pytest.raises(UnknownKeypointError):
        store.set_keypoint_color("wing", "#ff8000")


def test_a_colour_can_be_handed_back_to_the_palette(store):
    store.set_keypoint_color("beak", "#ff8000")
    store.set_keypoint_color("beak", None)

    assert store.color_for("beak") is None


def test_clearing_drops_every_pinned_colour(store):
    store.set_keypoint_color("beak", "#ff8000")
    store.set_keypoint_color("eye", "#00ff00")
    store.clear_keypoint_colors()

    assert store.keypoint_color == {}


def test_colours_survive_the_sidecar(store, tmp_path):
    store.set_keypoint_color("eye", "#00ff00")
    path = tmp_path / "clip.mp4.keypoints.json"
    store.save(path)

    assert KeypointStore.load(path).color_for("eye") == "#00ff00"


def test_a_colour_follows_its_keypoint_through_a_schema_change(store):
    store.set_keypoint_color("eye", "#00ff00")
    store.set_keypoint_names(["wing", "eye"])

    assert store.color_for("eye") == "#00ff00"


def test_removing_a_keypoint_drops_its_colour(store):
    store.set_keypoint_color("eye", "#00ff00")
    store.set_keypoint_names(["beak", "tail"])

    assert store.keypoint_color == {}


def test_legacy_sidecars_are_shared(tmp_path):
    path = tmp_path / "clip.mp4.keypoints.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    assert KeypointStore.load(path).shared_keypoints is True


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


def test_store_to_movement_ds_axis_order(store):
    """x goes to space="x" and y to space="y" — the one place the swap lives."""
    store.set_point(0, "beak", (11.0, 22.0))
    ds = store_to_movement_ds(store, fps=FPS)

    assert ds.position.dims == ("time", "space", "keypoint", "individual")
    assert list(ds.coords["space"].values) == ["x", "y"]
    assert ds.position.sel(space="x", keypoint="beak").isel(time=0, individual=0) == 11.0
    assert ds.position.sel(space="y", keypoint="beak").isel(time=0, individual=0) == 22.0


def test_store_to_movement_ds_dims_are_singular(pair):
    ds = store_to_movement_ds(pair, fps=FPS)
    assert set(ds.dims) == {"time", "space", "keypoint", "individual"}
    assert list(ds.coords["individual"].values) == ["crow_a", "crow_b"]


def test_store_to_movement_ds_keeps_individuals_apart(pair):
    pair.set_point(2, "beak", (1.0, 2.0), "crow_a")
    pair.set_point(2, "beak", (30.0, 40.0), "crow_b")
    ds = store_to_movement_ds(pair, fps=FPS)

    assert ds.position.sel(space="x", keypoint="beak", individual="crow_a").isel(time=2) == 1.0
    assert ds.position.sel(space="x", keypoint="beak", individual="crow_b").isel(time=2) == 30.0


def test_store_to_movement_ds_carries_the_fill(pair):
    pair.set_fill_from_flat(
        np.arange(pair.n_frames * pair.n_points * 2, dtype=float).reshape(pair.n_frames, pair.n_points, 2),
        np.full((pair.n_frames, pair.n_points), 0.25),
    )
    ds = store_to_movement_ds(pair, fps=FPS)

    expected = pair.filled[3, 1, 2]
    got = ds.position.isel(time=3, individual=1).sel(keypoint=NAMES[2]).values
    np.testing.assert_allclose(got, expected)
    assert ds.confidence.isel(time=3, individual=1, keypoint=2) == 0.25


def test_store_to_movement_ds_time_uses_fps(store):
    ds = store_to_movement_ds(store, fps=FPS)
    assert ds.sizes["time"] == store.n_frames
    np.testing.assert_allclose(ds.coords["time"].values[1], 1.0 / FPS)


def test_store_to_movement_ds_anchor_confidence_is_one(store):
    store.set_point(3, "tail", (1.0, 2.0))
    ds = store_to_movement_ds(store, fps=FPS)
    assert ds.confidence.sel(keypoint="tail").isel(time=3, individual=0) == 1.0
    assert np.isnan(ds.confidence.sel(keypoint="tail").isel(time=4, individual=0))


def test_store_to_movement_ds_rejects_missing_fps(store):
    with pytest.raises(KeypointStoreError):
        store_to_movement_ds(store, fps=0.0)


def test_store_to_movement_ds_round_trips_through_pose_render(pair):
    """The exported dataset must survive the normal display pipeline."""
    from ethograph.gui.pose_render import movement_ds_to_pose_render

    pair.set_point(0, "beak", (11.0, 22.0), "crow_a")
    pair.set_point(4, "tail", (33.0, 44.0), "crow_b")
    pr = movement_ds_to_pose_render(store_to_movement_ds(pair, fps=FPS), "test")

    # points rows are (track_id, frame, y, x) — the swap back out again.
    rows = pr.data[pr.data_not_nan]
    assert len(rows) == 2
    assert {(row[2], row[3]) for row in rows} == {(22.0, 11.0), (44.0, 33.0)}
    assert set(pr.properties["individual"].unique()) == {"crow_a", "crow_b"}


# ----------------------------------------------------------------------
# Derived kinematics ("Load into the GUI")
# ----------------------------------------------------------------------


def _moving_store(step: float = 10.0, n: int = 11) -> KeypointStore:
    """A store whose fill moves every keypoint *step* px per frame in x."""
    store = KeypointStore(keypoint_names=["beak", "tail"], n_frames=n, individual_names=["bird"])
    filled = np.zeros((n, 1, 2, 2))
    filled[:, 0, :, 0] = (np.arange(n) * step)[:, None]
    store.set_fill(filled, np.ones((n, 1, 2)))
    return store


def test_kinematics_keep_the_poses_datasets_own_names_and_dims():
    """They go back into that dataset, so they share its names and its axes —
    no prefix, no renamed time. One dataset is the whole point."""
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), KINEMATICS)

    assert set(arrays) == set(KINEMATICS)
    assert arrays["velocity"].dims == ("time", "space", "keypoint", "individual")
    # Speed is a magnitude, so the space axis is gone.
    assert arrays["speed"].dims == ("time", "keypoint", "individual")


def test_speed_is_in_units_per_second_not_per_frame():
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_moving_store(step=10.0), FPS), ["speed"])
    speed = arrays["speed"].isel({"time": 5, "keypoint": 0, "individual": 0})
    np.testing.assert_allclose(float(speed), 10.0 * FPS)


def test_constant_velocity_has_no_acceleration():
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), ["acceleration"])
    middle = arrays["acceleration"].isel({"time": 5})
    np.testing.assert_allclose(middle.values, 0.0, atol=1e-9)


def test_kinematics_use_the_filled_frames():
    """Inspecting a fill is the point — anchors alone would be almost all NaN."""
    pytest.importorskip("movement")
    store = _moving_store()
    assert len(store.anchor_frames()) == 0  # nothing labelled by hand
    speed = store_to_kinematics(store_to_movement_ds(store, FPS), ["speed"])["speed"]
    assert not np.any(np.isnan(speed.values[1:-1]))


def test_nothing_ticked_derives_nothing():
    """`position` is already in the dataset; this only adds to it."""
    pytest.importorskip("movement")
    assert store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), []) == {}


def test_unknown_kinematic_raises():
    pytest.importorskip("movement")
    with pytest.raises(KeypointStoreError, match="Unknown kinematic"):
        store_to_kinematics(store_to_movement_ds(_moving_store(), FPS), ["jerk"])


def test_time_axis_is_the_videos_frame_grid():
    """The keypoints are a dataset in their own right: one sample per frame,
    spaced by 1/fps, so nothing else has to exist for them to be plottable."""
    store = _moving_store(n=11)
    ds = store_to_movement_ds(store, FPS)
    time = ds.coords["time"].values

    assert len(time) == store.n_frames
    np.testing.assert_allclose(time, np.arange(store.n_frames) / FPS)
    np.testing.assert_allclose(np.diff(time), 1.0 / FPS)


def test_kinematics_keep_the_frame_grid():
    pytest.importorskip("movement")
    store = _moving_store(n=11)
    arrays = store_to_kinematics(store_to_movement_ds(store, FPS), ["speed"])
    time = arrays["speed"].coords["time"].values
    np.testing.assert_allclose(time, np.arange(store.n_frames) / FPS)


# ----------------------------------------------------------------------
# One dataset: what both output paths produce
# ----------------------------------------------------------------------


def test_the_dataset_is_a_movement_poses_file():
    pytest.importorskip("movement")
    ds = store_to_dataset(_moving_store(), FPS, kinematics=["speed"])

    assert ds.attrs["ds_type"] == "poses"
    assert set(ds.dims) == {"time", "space", "keypoint", "individual"}
    assert {"position", "confidence", "speed"} <= set(ds.data_vars)


def test_the_dataset_carries_only_what_was_asked_for():
    pytest.importorskip("movement")
    ds = store_to_dataset(_moving_store(), FPS)
    assert set(ds.data_vars) == {"position", "confidence"}


def test_the_dataset_follows_the_y_flip():
    store = KeypointStore(keypoint_names=["beak"], n_frames=3, individual_names=["bird"])
    store.set_point(0, "beak", (10.0, 30.0))
    ds = store_to_dataset(store, FPS, image_height=IMAGE_HEIGHT)
    assert float(ds.position.sel(space="y", keypoint="beak").isel(time=0, individual=0)) == IMAGE_HEIGHT - 30.0


# ----------------------------------------------------------------------
# y-flip: image coordinates (y down) -> plot coordinates (y up)
# ----------------------------------------------------------------------

IMAGE_HEIGHT = 480.0


def test_image_coordinates_are_kept_by_default(store):
    """The overlay and the DLC export both need raw image coordinates."""
    store.set_point(0, "beak", (10.0, 30.0))
    ds = store_to_movement_ds(store, FPS)
    assert float(ds.position.sel(space="y", keypoint="beak").isel(time=0, individual=0)) == 30.0


def test_flipping_puts_the_top_of_the_frame_at_the_top_of_the_plot(store):
    """A keypoint near the top (small y in image space) must end up with a LARGE
    y, since plots count upward."""
    store.set_point(0, "beak", (10.0, 30.0))
    ds = store_to_movement_ds(store, FPS, image_height=IMAGE_HEIGHT)
    assert float(ds.position.sel(space="y", keypoint="beak").isel(time=0, individual=0)) == IMAGE_HEIGHT - 30.0


def test_flipping_leaves_x_alone(store):
    store.set_point(0, "beak", (10.0, 30.0))
    ds = store_to_movement_ds(store, FPS, image_height=IMAGE_HEIGHT)
    assert float(ds.position.sel(space="x", keypoint="beak").isel(time=0, individual=0)) == 10.0


def test_flipping_preserves_vertical_ordering(store):
    """Whatever is higher in the video stays higher in the plot."""
    store.set_point(0, "beak", (0.0, 10.0))  # near the top of the frame
    store.set_point(0, "tail", (0.0, 400.0))  # near the bottom
    ds = store_to_movement_ds(store, FPS, image_height=IMAGE_HEIGHT)
    y = ds.position.sel(space="y").isel(time=0, individual=0)
    assert float(y.sel(keypoint="beak")) > float(y.sel(keypoint="tail"))


def test_flipping_applies_to_filled_frames_too(store):
    filled = np.full((store.n_frames, 1, store.n_keypoints, 2), 100.0)
    store.set_fill(filled, np.ones((store.n_frames, 1, store.n_keypoints)))
    ds = store_to_movement_ds(store, FPS, image_height=IMAGE_HEIGHT)
    y = ds.position.sel(space="y").isel(time=3, individual=0, keypoint=0)
    assert float(y) == IMAGE_HEIGHT - 100.0


def test_flip_rejects_a_missing_image_height(store):
    """A zero height would silently mirror everything about y=0."""
    with pytest.raises(KeypointStoreError, match="image_height must be positive"):
        store_to_movement_ds(store, FPS, image_height=0.0)


def test_flipped_velocity_changes_sign(store):
    """Kinematics follow the flip, so velocity's y sign matches what is drawn."""
    pytest.importorskip("movement")
    filled = np.zeros((11, 1, 2, 2))
    filled[:, 0, :, 1] = (np.arange(11) * 5.0)[:, None]  # moving DOWN the image
    down = KeypointStore(keypoint_names=["beak", "tail"], n_frames=11, individual_names=["bird"])
    down.set_fill(filled, np.ones((11, 1, 2)))

    raw = store_to_kinematics(store_to_movement_ds(down, FPS), ["velocity"])
    flipped = store_to_kinematics(store_to_movement_ds(down, FPS, image_height=IMAGE_HEIGHT), ["velocity"])

    y_raw = raw["velocity"].sel(space="y").isel({"time": 5, "keypoint": 0, "individual": 0})
    y_flip = flipped["velocity"].sel(space="y").isel({"time": 5, "keypoint": 0, "individual": 0})
    assert float(y_raw) > 0  # y grows downward in image space
    np.testing.assert_allclose(float(y_flip), -float(y_raw))


# ----------------------------------------------------------------------
# Kinematics over sparse anchors — the no-fill case
# ----------------------------------------------------------------------


def _sparse_store(step: float = 10.0, gap: int = 10, n_anchors: int = 5) -> KeypointStore:
    """Anchors every *gap* frames and NO fill — what labelling alone produces."""
    n_frames = gap * n_anchors
    store = KeypointStore(keypoint_names=["beak", "tail"], n_frames=n_frames, individual_names=["bird"])
    for index in range(n_anchors):
        frame = index * gap
        store.set_point(frame, "beak", (frame * step, 0.0))
        store.set_point(frame, "tail", (frame * step, 50.0))
    return store


def test_sparse_anchors_still_give_a_velocity():
    """Regression: differentiating the raw grid returned NaN EVERYWHERE.

    A central difference over the stored frames blanks both neighbours of any
    NaN, and with anchors 10 frames apart every anchor is such a neighbour — so
    velocity, speed and acceleration all came back empty from positions that
    were plainly there, and the plot drew nothing.
    """
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_sparse_store(), FPS), KINEMATICS)
    for name in ("velocity", "speed", "acceleration"):
        values = arrays[name].values
        assert np.isfinite(values).any(), f"{name} is entirely NaN over sparse anchors"


def test_sparse_velocity_is_the_average_across_the_gap():
    """10 px per frame, sampled every 10th frame, is still 10 px/frame."""
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_sparse_store(step=10.0, gap=10), FPS), ["velocity"])
    x = arrays["velocity"].sel(space="x").isel({"time": 20, "keypoint": 0, "individual": 0})
    np.testing.assert_allclose(float(x), 10.0 * FPS)


def test_kinematics_are_only_scored_where_a_point_was_observed():
    """Between two anchors there is no evidence, so there is no velocity."""
    pytest.importorskip("movement")
    arrays = store_to_kinematics(store_to_movement_ds(_sparse_store(gap=10), FPS), ["velocity"])
    velocity = arrays["velocity"].isel({"keypoint": 0, "individual": 0})
    assert np.isfinite(velocity.isel({"time": 20}).values).all()
    assert np.isnan(velocity.isel({"time": 25}).values).all()


def test_dense_kinematics_still_match_movement():
    """The observed-frame path must reduce to movement's own answer when the
    data is dense — otherwise a fill's kinematics would quietly change."""
    movement = pytest.importorskip("movement.kinematics")
    ds = store_to_movement_ds(_moving_store(step=7.0, n=21), FPS)
    ours = store_to_kinematics(ds, ["velocity"])["velocity"]
    theirs = movement.compute_velocity(ds["position"])
    np.testing.assert_allclose(ours.values, theirs.values)


# ----------------------------------------------------------------------
# Head direction
# ----------------------------------------------------------------------


#: A tag facing the top of the frame, in image coordinates (y DOWN).
FACING_UP = (0.0, -1.0)


def _tagged_store(frames=(0, 1, 2), heading=FACING_UP) -> KeypointStore:
    """One individual wearing ONE tag on its `marker` keypoint.

    The tag is a single keypoint, not four: its orientation rides along with the
    detection, which is the whole point of the design.
    """
    store = KeypointStore(keypoint_names=["marker", "beak"], n_frames=5, individual_names=["bird"])
    positions, orientation = {}, {}
    for frame in frames:
        points = np.full((1, 2, 2), np.nan)
        points[0, 0] = (100.0, 200.0)  # `marker`, the tagged keypoint
        vectors = np.full((1, 2, 2), np.nan)
        vectors[0, 0] = heading
        positions[frame], orientation[frame] = points, vectors
    store.set_detections(positions, orientation=orientation)
    return store


def test_head_direction_comes_from_one_tagged_keypoint():
    """No pair to nominate: the marker is the keypoint, and it knows its heading."""
    arrays = store_to_head_direction(_tagged_store(), FPS)
    vector = arrays["head_direction"].isel(time=0, individual=0).sel(keypoint="marker")
    np.testing.assert_allclose(vector.values, FACING_UP)


def test_an_untagged_keypoint_has_no_heading():
    """`beak` was never detected by a marker, so it faces nowhere — not zero."""
    arrays = store_to_head_direction(_tagged_store(), FPS)
    beak = arrays["head_direction"].isel(individual=0).sel(keypoint="beak")
    assert np.isnan(beak.values).all()


def test_head_direction_keeps_the_keypoint_dimension():
    """One tag is one heading, so it is per keypoint — unlike movement's own
    forward vector, which drops the dim because it consumes a pair."""
    arrays = store_to_head_direction(_tagged_store(), FPS)
    assert arrays["head_direction"].dims == ("time", "space", "keypoint", "individual")
    assert arrays["heading"].dims == ("time", "keypoint", "individual")


def test_frames_without_a_decode_stay_empty():
    """A heading is a measurement; nothing interpolates it."""
    arrays = store_to_head_direction(_tagged_store(frames=(0, 2)), FPS)
    marker = arrays["head_direction"].isel(individual=0).sel(keypoint="marker")
    assert not np.isnan(marker.isel(time=0).values).any()
    assert np.isnan(marker.isel(time=1).values).all()
    assert np.isnan(marker.isel(time=4).values).all()


def test_head_direction_follows_the_y_flip():
    """Orientation is measured in image coordinates, so flipping the positions
    without flipping it would point the arrow opposite to the trajectory."""
    store = _tagged_store()
    raw = store_to_head_direction(store, FPS, y_flipped=False)
    flipped = store_to_head_direction(store, FPS, y_flipped=True)

    def y_of(arrays):
        return float(arrays["head_direction"].isel(time=0, individual=0).sel(keypoint="marker", space="y"))

    np.testing.assert_allclose(y_of(flipped), -y_of(raw))


def test_the_flip_leaves_x_alone():
    store = _tagged_store(heading=(0.6, -0.8))
    raw = store_to_head_direction(store, FPS, y_flipped=False)
    flipped = store_to_head_direction(store, FPS, y_flipped=True)

    def x_of(arrays):
        return float(arrays["head_direction"].isel(time=0, individual=0).sel(keypoint="marker", space="x"))

    np.testing.assert_allclose(x_of(flipped), x_of(raw))


def test_heading_is_the_angle_of_the_vector_in_degrees():
    heading = store_to_head_direction(_tagged_store(), FPS)["heading"]
    # Forward is (0, -1) and the reference is +x, so the signed angle is -90°.
    np.testing.assert_allclose(float(heading.isel(time=0, individual=0).sel(keypoint="marker")), -90.0)


def test_heading_is_nan_where_nothing_was_measured():
    """atan2 of a NaN pair must not become an angle."""
    heading = store_to_head_direction(_tagged_store(frames=(0,)), FPS)["heading"]
    assert np.isnan(float(heading.isel(time=3, individual=0).sel(keypoint="marker")))


def test_head_direction_joins_the_poses_dataset_unrenamed():
    """It rides in the same dataset as `position`, so it shares its axes."""
    arrays = store_to_head_direction(_tagged_store(), FPS)
    assert set(arrays) == {"head_direction", "heading"}
    assert all("time" in array.dims for array in arrays.values())

    ds = store_to_dataset(_tagged_store(), FPS, head_direction=True)
    assert {"head_direction", "heading"} <= set(ds.data_vars)


def test_head_direction_is_on_the_videos_frame_grid():
    time = store_to_head_direction(_tagged_store(), FPS)["heading"].coords["time"].values
    np.testing.assert_allclose(time, np.arange(5) / FPS)


def test_head_direction_rejects_a_missing_fps():
    with pytest.raises(KeypointStoreError, match="fps must be positive"):
        store_to_head_direction(_tagged_store(), 0.0)


def test_a_store_with_no_oriented_marker_offers_nothing(store):
    """The precondition, stated: no tags means no head direction."""
    assert store.has_orientation is False
    arrays = store_to_head_direction(store, FPS)
    assert np.isnan(arrays["head_direction"].values).all()
    assert np.isnan(arrays["heading"].values).all()


def test_has_orientation_reports_a_tag_run():
    assert _tagged_store().has_orientation is True


def test_clearing_the_detections_takes_the_orientation_with_them():
    tagged = _tagged_store()
    tagged.clear_detections()
    assert tagged.has_orientation is False


def test_a_rejected_detection_drops_its_heading():
    """Rejecting a misread must not leave its heading behind."""
    tagged = _tagged_store(frames=(0, 1))
    tagged.clear_detections_for(0)
    marker = store_to_head_direction(tagged, FPS)["head_direction"].isel(individual=0).sel(keypoint="marker")
    assert np.isnan(marker.isel(time=0).values).all()
    assert not np.isnan(marker.isel(time=1).values).any()


def test_a_point_dropped_for_quality_drops_its_heading():
    """A heading is only as good as the decode it came from."""
    store = KeypointStore(keypoint_names=["marker"], n_frames=3, individual_names=["bird"])
    store.set_detections(
        {0: np.array([[[10.0, 20.0]]])},
        {0: np.array([[0.1]])},
        quality_min=0.5,
        orientation={0: np.array([[FACING_UP]])},
    )
    assert store.has_orientation is False


def test_pixel_export_says_pixels(store):
    assert store_to_movement_ds(store, FPS).attrs["space_unit"] == "pixels"
