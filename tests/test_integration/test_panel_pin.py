"""A panel pinned to one individual, beside one that follows the sidebar.

Birdpark has two individuals. One rule decides whose data and labels a panel
shows — its pin, else the sidebar's individual — and the labelling subject
is the last clicked panel's. The Qt-free half of this is
``tests/test_unit/test_panel_individual.py``.
"""

from __future__ import annotations

import pytest

from ethograph.gui.label_drawing_mixin import draw_key
from ethograph.labels.intervals import add_interval, empty_intervals

pytest.importorskip("qtpy")


def _feature_with_individual_dim(state) -> tuple[str, str] | None:
    loader = state.data_loader
    for feature in loader.catalog.feature_choices():
        dims = loader.feature_dims(feature)
        for dim in dims:
            if dim in ("individual", "individuals") and len(dims[dim]) > 1:
                return feature, dim
    return None


def test_a_pinned_panel_shows_and_labels_its_own_individual(birdpark_gui):
    """Birdpark has two individuals: the sidebar drives one panel, a pin the other."""
    _, meta = birdpark_gui
    state = meta.app_state
    dw = meta.data_widget
    pc = meta.plot_container
    names = state.label_individuals()
    assert len(names) >= 2
    a, b = names[0], names[1]
    found = _feature_with_individual_dim(state)
    assert found is not None, "birdpark carries a per-individual feature"
    feature, dim = found

    state.set_key_sel(dw._individual_actor_key(), a)
    following = pc.add_panel("lineplot", feature=feature)
    pinned = pc.add_panel("lineplot", feature=feature)
    pc.pin_panel(pinned, b)

    assert following._effective_selections()[dim] == a
    assert pinned._effective_selections()[dim] == b
    assert pc.panel_title(pinned) == f"{feature} \u2014 {b} (pinned)"
    assert pc.panel_title(following) == f"{feature} \u2014 {a} (sidebar)"

    # Switching the sidebar moves the follower only — and its title with it.
    dw.apply_panel_control(dw._individual_actor_key(), b)
    assert following._effective_selections()[dim] == b
    assert pc._dyn_docks[following].titleBarWidget().title() == f"{feature} \u2014 {b} (sidebar)"
    assert pc._dyn_docks[pinned].titleBarWidget().title() == f"{feature} \u2014 {b} (pinned)"
    dw.apply_panel_control(dw._individual_actor_key(), a)

    # Labels: a has two, b has one; each panel draws its own individual's.
    trial = state.trials_sel
    df = add_interval(empty_intervals(), 0.5, 1.0, 1, a)
    df = add_interval(df, 1.5, 2.0, 1, a)
    df = add_interval(df, 2.5, 3.0, 1, b)
    state.set_trial_intervals(trial, df)
    state.label_intervals = state.get_trial_intervals(trial)
    state.individual_receiver = ""
    dw.update_label_plot()
    shown = state.get_display_intervals()
    assert list(dw._subject_intervals(shown, following)["onset_s"]) == [0.5, 1.5]
    assert list(dw._subject_intervals(shown, pinned)["onset_s"]) == [2.5]
    pinned_items = {id(item) for item in pinned.label_items}
    b_items = {id(item) for item, *_ in pc._label_item_index[draw_key(1, 2.5, b, "")]}
    assert b_items and b_items <= pinned_items, "b's label is drawn on the pinned panel"
    a_items = {id(item) for item, *_ in pc._label_item_index[draw_key(1, 0.5, a, "")]}
    assert not (a_items & pinned_items), "and a's are not"

    # Clicking the pinned panel makes b the labelling subject.
    pc.active_panels.set_active(pc.active_panels.registration_for(pinned))
    assert state.selected_individual() == b


def _pick_when_shown(qtbot, menu, choose):
    """Menus block in ``exec_``; pick an action from a timer once the menu is up."""
    from qtpy.QtCore import QTimer

    def _pick():
        action = choose(menu)
        if action is None:
            menu.close()
            raise AssertionError(f"no such entry among {[a.text() for a in menu.actions()]}")
        action.trigger()
        menu.close()

    QTimer.singleShot(50, _pick)


def test_the_sidebar_pin_button_pins_the_clicked_panel(birdpark_gui, qtbot):
    """Click a plot, click 📌, pick a name: the plot is pinned; pick Unpin: it follows again."""
    from qtpy.QtCore import Qt

    _, meta = birdpark_gui
    state = meta.app_state
    dw = meta.data_widget
    pc = meta.plot_container
    names = state.label_individuals()
    feature, dim = _feature_with_individual_dim(state)
    plot = pc.add_panel("lineplot", feature=feature)
    pc.active_panels.set_active(pc.active_panels.registration_for(plot))  # what a click on the plot does

    button = dw.individual_pin_button
    menu = button.menu()
    other = names[1]
    seen: list[list[str]] = []

    def _choose(text_match):
        def _pick(m):
            seen.append([a.text() for a in m.actions() if a.text()])
            return next((a for a in m.actions() if text_match(a.text())), None)

        return _pick

    _pick_when_shown(qtbot, menu, _choose(lambda t: t == other))
    qtbot.mouseClick(button, Qt.LeftButton)
    assert plot.pinned_individual == other
    assert plot._effective_selections()[dim] == other
    assert pc.panel_title(plot) == f"{feature} \u2014 {other} (pinned)"
    # One radio choice: follow the sidebar (naming what that means), or a name.
    assert seen[-1][1] == f"Follow sidebar ({names[0]})"
    assert seen[-1][2:4] == names[:2]

    _pick_when_shown(qtbot, menu, _choose(lambda t: t.startswith("Follow sidebar")))
    qtbot.mouseClick(button, Qt.LeftButton)
    assert plot.pinned_individual is None
    assert plot._effective_selections()[dim] == state.sidebar_individual()
    assert pc.panel_title(plot) == f"{feature} \u2014 {names[0]} (sidebar)"


def test_unpin_all_puts_every_panel_back_on_the_sidebar(birdpark_gui, qtbot):
    from qtpy.QtCore import Qt

    _, meta = birdpark_gui
    state = meta.app_state
    dw = meta.data_widget
    pc = meta.plot_container
    names = state.label_individuals()
    feature, dim = _feature_with_individual_dim(state)
    one = pc.add_panel("lineplot", feature=feature)
    two = pc.add_panel("lineplot", feature=feature)
    pc.pin_panel(one, names[1])
    pc.pin_panel(two, names[1])
    assert one._effective_selections()[dim] == names[1]

    button = dw.individual_pin_button
    _pick_when_shown(
        qtbot, button.menu(), lambda m: next((a for a in m.actions() if a.text().startswith("Unpin all")), None)
    )
    qtbot.mouseClick(button, Qt.LeftButton)
    assert one.pinned_individual is None and two.pinned_individual is None
    assert one._effective_selections()[dim] == state.sidebar_individual()
    assert not dw._any_pinned()


def test_the_pin_button_is_hidden_for_one_individual(moll2025_gui):
    _, meta = moll2025_gui
    dw = meta.data_widget
    if len(meta.app_state.label_individuals()) > 1:
        pytest.skip("needs a one-individual dataset")
    assert not dw.individual_pin_button.isVisibleTo(dw)


def test_the_panel_move_menu_offers_pin_entries(birdpark_gui, qtbot):
    """The ⠿ button of a feature panel opens the move menu, with pin entries at its end."""
    from qtpy.QtCore import Qt

    _, meta = birdpark_gui
    state = meta.app_state
    pc = meta.plot_container
    names = state.label_individuals()
    feature, dim = _feature_with_individual_dim(state)
    plot = pc.add_panel("lineplot", feature=feature)
    dock = pc._dyn_docks[plot]
    from qtpy.QtWidgets import QPushButton

    move_btn = dock.titleBarWidget().findChild(QPushButton, "panel_move_btn")
    assert move_btn is not None

    def _pin_entry(menu):
        sub = next((a.menu() for a in menu.actions() if a.text() == "Individual"), None)
        return None if sub is None else next((a for a in sub.actions() if a.text() == names[1]), None)

    # The move menu is built on click; grab it through its exec_ by patching QMenu.exec_ once.
    from qtpy.QtWidgets import QMenu

    opened = []
    real_exec = QMenu.exec_

    def _capture(menu, *args, **kwargs):
        opened.append(menu)
        _pick_when_shown(qtbot, menu, _pin_entry)
        return real_exec(menu, *args, **kwargs)

    QMenu.exec_ = _capture
    try:
        qtbot.mouseClick(move_btn, Qt.LeftButton)
    finally:
        QMenu.exec_ = real_exec
    assert opened, "the ⠿ button opened a menu"
    assert plot.pinned_individual == names[1]
    assert plot._effective_selections()[dim] == names[1]


def test_the_pending_label_previews_only_on_the_subjects_panels(birdpark_gui):
    """After the first click of a state label, the dashed anchor appears on the panels of that bird only."""
    _, meta = birdpark_gui
    state = meta.app_state
    pc = meta.plot_container
    names = state.label_individuals()
    feature, _dim = _feature_with_individual_dim(state)
    following = pc.add_panel("lineplot", feature=feature)
    pinned = pc.add_panel("lineplot", feature=feature)
    pc.pin_panel(pinned, names[1])

    pc.active_panels.set_active(pc.active_panels.registration_for(following))
    pc.show_pending_label(0.5, (255, 0, 0))
    plots = {id(p) for p, _line in pc._pending_label_items}
    assert id(following) in plots and id(pinned) not in plots

    pc.active_panels.set_active(pc.active_panels.registration_for(pinned))
    pc.show_pending_label(0.5, (255, 0, 0))
    plots = {id(p) for p, _line in pc._pending_label_items}
    assert id(pinned) in plots and id(following) not in plots
    pc.clear_pending_label()


def test_any_panel_can_be_pinned_even_without_an_individual_dim(birdpark_gui):
    """A label timeline has no data to select from, but it still shows one bird's labels.

    Pinning it decides whose labels it draws and whose label a click on it
    creates, its title says so, and the pin survives a layout round trip.
    """
    _, meta = birdpark_gui
    state = meta.app_state
    dw = meta.data_widget
    pc = meta.plot_container
    names = state.label_individuals()
    a, b = names[0], names[1]
    state.set_key_sel(dw._individual_actor_key(), a)

    ribbon = pc.add_panel("labels")
    assert pc._dyn_docks[ribbon].titleBarWidget().title() == f"Labels \u2014 {a} (sidebar)"
    pc.active_panels.set_active(pc.active_panels.registration_for(ribbon))
    assert dw._pinnable_widget() is ribbon
    dw.pin_panel(ribbon, b)
    assert ribbon.pinned_individual == b
    assert pc._dyn_docks[ribbon].titleBarWidget().title() == f"Labels \u2014 {b} (pinned)"
    assert state.selected_individual() == b, "the clicked panel's pin is the labelling subject"

    trial = state.trials_sel
    df = add_interval(empty_intervals(), 0.5, 1.0, 1, a)
    df = add_interval(df, 2.5, 3.0, 1, b)
    state.set_trial_intervals(trial, df)
    state.label_intervals = state.get_trial_intervals(trial)
    state.individual_receiver = ""
    dw.update_label_plot()
    shown = state.get_display_intervals()
    assert list(dw._subject_intervals(shown, ribbon)["onset_s"]) == [2.5]

    layout = pc.layout_state()
    assert {"type": "labels", "individual": b} in layout["panels"]
    pc.apply_layout_state(layout)
    restored = pc._label_ribbons()
    assert [r.pinned_individual for r in restored] == [b]
    assert pc._dyn_docks[restored[0]].titleBarWidget().title() == f"Labels \u2014 {b} (pinned)"
