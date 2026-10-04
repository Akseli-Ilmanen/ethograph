"""Debug: what a saved firing-rate heatmap shows when the layout is restored."""

import pytest

pytest.importorskip("qtpy")


def _shape(plot):
    return None if plot._buffered_data is None else plot._buffered_data.shape


def test_restore(moll2025_gui):
    _, meta = moll2025_gui
    pc = meta.plot_container
    loader = meta.app_state.data_loader
    print("\nglobal selections:", meta.app_state.get_selections(), "features_sel:", meta.app_state.features_sel)

    pc.apply_layout_state({"panels": [{"type": "heatmap", "feature": "firing_rate", "selections": {}}]})
    hm = pc.heatmap_plots[-1]
    print("RESTORE  derived:", loader.is_derived("firing_rate"))
    print("RESTORE  state:", hm.panel_state, "labels:", hm._channel_labels[:4], "buffer:", _shape(hm))

    try:
        got = loader.select("firing_rate", {}, t0=0.0, t1=5.0)
        print("RESTORE  select:", None if got is None else (got.data.shape, got.dim_labels))
    except Exception as e:  # debug script: show whatever the render would hit
        print("RESTORE  select raised:", type(e).__name__, e)
    hm._render_heatmap(0.0, 5.0)
    print("RESTORE  rendered:", hm._channel_labels[:4], _shape(hm))

    meta._add_firing_rate_panel()
    hm._clear_buffer()
    hm._render_heatmap(0.0, 5.0)
    print("AFTER ADD rendered (restored panel):", hm._channel_labels[:4], _shape(hm))
    hm2 = pc.heatmap_plots[-1]
    print("ADD      state:", hm2.panel_state, "labels:", hm2._channel_labels[:4], "buffer:", _shape(hm2))
    hm.update_plot()
    print("RESTORED panel after compute:", hm._channel_labels[:4], _shape(hm))
