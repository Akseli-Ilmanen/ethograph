"""Integration tests for automatic panel-layout persistence.

The ``gui`` fixture disables ``app_state._layout_snapshot_provider`` for
hermeticity, so the auto-save → snapshot → local_settings.yaml path is
otherwise untested. These tests re-enable it against a loaded dataset and
drive the exact code path the periodic auto-save timer and the close-save use.
"""

from __future__ import annotations

import yaml


def _redirect_settings_paths(app_state, tmp_path, monkeypatch):
    local_path = tmp_path / "local_settings.yaml"
    global_path = tmp_path / "gui_settings.yaml"
    monkeypatch.setattr(type(app_state), "_local_settings_path", lambda self: local_path)
    monkeypatch.setattr(type(app_state), "_global_settings_path", lambda self: global_path)
    return local_path, global_path


def test_panel_layout_persisted_by_autosave(moll2025_gui, tmp_path, monkeypatch):
    viewer, meta = moll2025_gui
    app_state = meta.app_state
    assert app_state.ready

    app_state._layout_snapshot_provider = meta._snapshot_layouts
    local_path, global_path = _redirect_settings_paths(app_state, tmp_path, monkeypatch)

    assert app_state.save_to_yaml()

    local_state = yaml.safe_load(local_path.read_text(encoding="utf-8"))
    layout = local_state.get("panel_layout")
    assert layout, "panel_layout missing from local_settings.yaml"
    assert layout.get("dock_state_b64")
    assert layout.get("shell_dock_state_b64"), "shell dock arrangement is per-dataset"
    panel_types = {p["type"] for p in layout["panels"]}
    assert "lineplot" in panel_types or "heatmap" in panel_types

    global_state = yaml.safe_load(global_path.read_text(encoding="utf-8"))
    assert "panel_layout" not in global_state, "panel_layout is per-dataset (local) state"
    assert global_state.get("window_state"), "window_state missing from gui_settings.yaml"


def test_dataset_dock_state_applies_on_show(qtbot):
    """The per-dataset shell blob (panel_layout['shell_dock_state_b64']) must
    place docks on a machine with NO window_state of its own — the template /
    shared-local-settings scenario. Applying while hidden defers to show."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel

    from ethograph.gui.main_window import EthographMainWindow

    win = EthographMainWindow()
    qtbot.addWidget(win)
    dock = win.add_dock_widget(QLabel("space"), area="top", name="Space Plot", object_name="SpacePlotDock_0")
    win.addDockWidget(Qt.BottomDockWidgetArea, dock)
    blob = win.capture_dock_state_b64()
    win.close()

    win2 = EthographMainWindow()  # fresh machine: no window_state anywhere
    qtbot.addWidget(win2)
    dock2 = win2.add_dock_widget(QLabel("space"), area="top", name="Space Plot", object_name="SpacePlotDock_0")
    win2.apply_dock_state_b64(blob)  # dataset load happens while hidden
    assert win2.dockWidgetArea(dock2) == Qt.TopDockWidgetArea  # deferred
    win2.show()
    qtbot.wait(50)

    assert win2.dockWidgetArea(dock2) == Qt.BottomDockWidgetArea
    assert not dock2.isHidden()
    win2.hide()


def test_space_dock_gets_canonical_objectname(moll2025_gui, qtbot):
    """Space docks must carry their canonical objectName from creation —
    that's what ties them to the window-state blob across sessions."""
    viewer, meta = moll2025_gui
    dw = meta.data_widget

    for existing in list(dw.space_plots):  # dataset load may auto-create one
        dw.remove_space_plot(existing)

    sp = dw.add_space_plot(focus=False)
    qtbot.wait(50)
    assert sp.dock_widget.objectName() == "SpacePlotDock_0"

    meta._snapshot_layouts()
    entry = meta.app_state.panel_layout["space_plots"][0]
    assert "dock_area" not in entry  # placement is the shell blob's job now

    dw.remove_space_plot(sp)


def test_loaded_layout_allows_single_all_dim(moll2025_gui, qtbot):
    """sel_valid output must stay (time,) or (time, dim), so a loaded panel
    layout may leave at most ONE multi-value dim unselected ("All"). A layout
    violating this (template, hand-edited settings) keeps the first missing
    dim as "All" and pins the rest to their first value."""
    viewer, meta = moll2025_gui
    pc = meta.plot_container

    plot = pc.add_lineplot(feature="position")
    try:
        plot.apply_panel_settings({"feature": "position", "selections": {}})
        sels = plot.panel_state["selections"]
        dims = meta.app_state.data_loader.feature_dims("position")
        multi = [d for d, v in dims.items() if len(v) > 1]
        assert len(multi) >= 2, "test needs a feature with ≥2 multi-value dims"
        missing = [d for d in multi if d not in sels]
        assert missing == [multi[0]], f"exactly the first dim may stay 'All': {sels}"
        for d in multi[1:]:
            assert sels[d] == dims[d][0]
    finally:
        pc.remove_lineplot(plot)


def test_save_survives_snapshot_failure(moll2025_gui, tmp_path, monkeypatch):
    """A raising layout snapshot must not kill the whole save — the auto-save
    QTimer slot would swallow the exception and every save would silently
    stop happening (regression: no settings written for the rest of the
    session)."""
    viewer, meta = moll2025_gui
    app_state = meta.app_state

    def broken_snapshot():
        raise RuntimeError("wrapped C/C++ object deleted")

    app_state._layout_snapshot_provider = broken_snapshot
    local_path, global_path = _redirect_settings_paths(app_state, tmp_path, monkeypatch)

    assert app_state.save_to_yaml()
    assert local_path.exists()
    assert global_path.exists()


def test_broken_saved_layout_falls_back_to_defaults(moll2025_gui, monkeypatch):
    """A saved panel layout that fails to apply (stale for the data now
    loaded, hand-edited, older version) must never abort the load: the
    layout is discarded — so the next auto-save snapshots a working one —
    and the data-availability default panels are rebuilt."""
    viewer, meta = moll2025_gui
    app_state = meta.app_state
    pc = meta.plot_container

    def poisoned_apply(state):
        raise AssertionError("stale selections blew sel_valid")

    monkeypatch.setattr(pc, "apply_layout_state", poisoned_apply)
    app_state.panel_layout = {"panels": [{"type": "lineplot"}]}

    meta.apply_saved_panel_layout()  # must not raise

    assert app_state.panel_layout is None, "broken layout must be discarded"
    assert pc.line_plots, "data-availability default lineplot must be rebuilt"


def test_reset_panels_rebuilds_the_same_layout(moll2025_gui):
    """Ctrl+R replaces every panel with a fresh instance, yet the layout the
    user had (panel set, features, selections) comes back unchanged."""
    viewer, meta = moll2025_gui
    del meta.apply_saved_panel_layout  # the fixture's stub ignores every saved layout
    pc = meta.plot_container
    pc.add_lineplot(feature="position")
    before_panels = list(pc._dyn_panels)
    before_layout = pc.layout_state()["panels"]

    meta.reset_panels()

    assert pc.layout_state()["panels"] == before_layout
    assert not set(map(id, before_panels)) & set(map(id, pc._dyn_panels)), "panels must be rebuilt, not reused"
