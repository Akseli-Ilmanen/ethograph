def test_dbg(gui, monkeypatch):
    _shell, meta = gui
    calls = []
    monkeypatch.setattr(meta.shell.video_area, "arrange_grid", lambda: (calls.append(1), __import__("traceback").print_stack()))
    monkeypatch.setattr(meta.plot_container, "apply_layout_state", lambda layout: None)
    monkeypatch.setattr(meta.data_widget, "apply_space_layout_state", lambda s: None)
    monkeypatch.setattr(meta.data_widget, "apply_radial_layout_state", lambda s: None)
    meta.app_state.panel_layout = {"panels": []}
    print("BEFORE", calls)
    print("LAYOUT", meta.app_state.panel_layout, "ready", getattr(meta.app_state, "ready", None))
    meta.apply_saved_panel_layout()
    print("CALLS", calls)
    assert calls == []
