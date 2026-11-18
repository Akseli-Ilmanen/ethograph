"""Who the panels are about: the Individual selector and its receiver.

Picking an individual used to be a feature-plot privilege — the combo lived in
the "Xarray coords" group, so an audio, space or ephys panel had no way to say
whose labels it was showing, and a pynapple session filtered labels by nobody
at all.  The selector now sits above every context but the video's, and carries
a second combo, the receiver of the next label: recorded on the label, shown
as a tag on it, never a filter — one actor's labels exclude each other
whoever they are directed at.

The selector is shown in *every* context, the video and the label timeline
included: a click on either places a label on whoever is selected, so hiding
the answer would leave the user labelling blind.
"""

from __future__ import annotations

import pytest

from ethograph.labels.intervals import add_interval, empty_intervals

pytest.importorskip("qtpy")


def test_the_actor_combo_lives_above_the_contexts_not_among_the_coords(moll2025_gui):
    _, meta = moll2025_gui
    dw = meta.data_widget

    key = dw._individual_actor_key()
    assert key in dw.combos, "every session has an individual to select"
    assert dw._combo_row_layouts[key] is dw.individual_layout
    assert dw.combos[key] is not None


def test_every_context_gets_the_selector(moll2025_gui):
    """Including the video and a panel type the map does not name at all."""
    _, meta = moll2025_gui
    panel = meta.context_panel

    for context in ("audiotrace", "lineplot", "heatmap", "spectrogram", "space", "radial", "ephys", "neo", "video", ""):
        panel.set_context(context)
        assert panel._individual.isVisibleTo(panel), f"{context} must say whose data it shows"


def test_the_receiver_is_a_tag_on_the_label_never_a_filter(moll2025_gui):
    _, meta = moll2025_gui
    state = meta.app_state
    dw = meta.data_widget

    actor = state.selected_individual()
    trial = state.trials_sel
    df = add_interval(empty_intervals(), 0.5, 1.0, 1, actor)
    df = add_interval(df, 2.0, 2.5, 1, actor, individual_rec="partner")
    state.set_trial_intervals(trial, df)
    state.label_intervals = state.get_trial_intervals(trial)

    for receiver in ("", "partner"):
        state.individual_receiver = receiver
        shown = dw._subject_intervals(state.get_display_intervals())
        assert list(shown["onset_s"]) == [0.5, 2.0], "switching the receiver changes nothing on screen"
    assert list(shown["individual_rec"]) == ["", "partner"], "but the label remembers whom it was directed at"

    # Drawing must not raise, and the directed label carries its tag.
    dw.update_label_plot()


def test_the_actor_is_never_offered_as_its_own_receiver(moll2025_gui):
    _, meta = moll2025_gui
    state = meta.app_state
    dw = meta.data_widget

    dw._populate_receiver_combo()
    combo = dw.individual_rec_combo
    offered = [combo.itemData(i) for i in range(combo.count())]
    assert offered[0] == "", "None (solo) is always the default"
    assert state.selected_individual() not in offered


def test_labels_written_under_other_names_are_not_filtered_away(moll2025_gui):
    """A pynapple session synthesises ``individual_0`` while its labels file
    says ``Crow1``: filtering by the dataset's own name would blank every
    trial's overlay, so a disjoint naming skips the actor filter entirely."""
    _, meta = moll2025_gui
    state = meta.app_state
    dw = meta.data_widget

    trial = state.trials_sel
    df = add_interval(empty_intervals(), 0.5, 1.0, 1, "somebody_else_entirely")
    state.set_trial_intervals(trial, df)
    state.label_intervals = state.get_trial_intervals(trial)
    state.individual_receiver = ""

    assert not state.labels_name_our_individuals(state.label_intervals)
    assert len(dw._subject_intervals(state.get_display_intervals())) == 1


def test_imported_labels_pull_the_selector_to_whoever_they_name(moll2025_gui):
    """A selection with no labels moves to the most-labelled name, created if unknown; one with labels stays."""
    _, meta = moll2025_gui
    state = meta.app_state
    dw = meta.data_widget
    combo = dw.combos[dw._individual_actor_key()]
    before = state.selected_individual()

    state._all_labels_df = None  # the session's own labels name the selected individual
    df = add_interval(empty_intervals(), 0.5, 1.0, 1, "rare")
    df = add_interval(df, 2.0, 2.5, 1, "newcomer")
    df = add_interval(df, 3.0, 3.5, 1, "newcomer")
    state.set_trial_intervals(state.trials_sel, df)
    dw.select_labelled_individual()

    assert state.selected_individual() == "newcomer"
    assert combo.currentData() == "newcomer"
    assert "newcomer" in state.extra_individuals
    assert "rare" not in state.extra_individuals

    df = add_interval(df, 4.0, 4.5, 1, before)
    for onset in (5.0, 6.0, 7.0):
        df = add_interval(df, onset, onset + 0.5, 1, "rare")
    state.set_trial_intervals(state.trials_sel, df)
    dw.select_labelled_individual()
    assert state.selected_individual() == "newcomer", "a selection that has labels is left alone"


def test_the_combo_offers_whoever_you_add(moll2025_gui):
    """Your own list names someone the dataset's dim never had.

    The combo used to refuse to refill while the key was a catalog dim, so a name
    added in Settings was unreachable — the labels of an animal with no tracked
    data could not be placed at all.
    """
    _, meta = moll2025_gui
    dw = meta.data_widget
    key = dw._individual_actor_key()
    combo = dw.combos[key]
    declared = [str(combo.itemData(i)) for i in range(combo.count())]

    meta.app_state.extra_individuals = ["ghost"]
    dw.refresh_individual_choices()

    offered = [str(combo.itemData(i)) for i in range(combo.count())]
    assert offered == declared + ["ghost"]
    assert meta.app_state.label_individuals() == offered
