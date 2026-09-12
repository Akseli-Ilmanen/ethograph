"""Throwaway verification: does the quiet/clipping BirdPark mic actually render?

Not a permanent test — deleted after use. The birdpark example dataset already
ships this exact WAV as one of its mics, so this drives the real GUI path
end-to-end (no injection) and inspects what actually got painted.
"""

import numpy as np
from qtpy.QtWidgets import QApplication


def test_quiet_mic_renders(birdpark_audio_only_gui, qtbot):
    viewer, meta = birdpark_audio_only_gui
    app_state = meta.app_state
    dw = meta.data_widget
    pc = dw.plot_container

    keys = list(app_state.audio_source_map.keys())
    print("mics_sel choices:", keys)
    target = next(k for k in keys if "BP_2021-05-25_08-12-51_655154_0380000" in k and "Ch 1" in k)
    print("selecting", target, "->", app_state.audio_source_map[target])

    app_state.mics_sel = target
    pc.update_audio_panels()
    QApplication.processEvents()

    trace = pc.audio_trace_plots[0] if pc.audio_trace_plots else None
    spec = pc.spectrogram_plots[0] if pc.spectrogram_plots else None

    print("trace panel present:", trace is not None)
    print("spec panel present:", spec is not None)

    if trace is not None:
        ydata = trace.trace_item.yData
        print("trace yData is None?", ydata is None, "len", 0 if ydata is None else len(ydata))
        if ydata is not None and len(ydata):
            print("trace yData min/max", np.min(ydata), np.max(ydata))

    if spec is not None:
        img = spec.spec_item.image
        print("spec image is None?", img is None)
        if img is not None:
            print("spec image shape/dtype", img.shape, img.dtype)
            print("spec image min/max", img.min(), img.max())
        print("spec buffer Sxx_db is None?", spec.buffer.Sxx_db is None)
        if spec.buffer.Sxx_db is not None:
            print("Sxx_db min/max", spec.buffer.Sxx_db.min(), spec.buffer.Sxx_db.max())
        print("spec_item levels", spec.spec_item.levels)
        print(
            "app_state vmin_db/vmax_db",
            app_state.vmin_db,
            app_state.vmax_db,
            "mode",
            getattr(app_state, "spec_levels_mode", None),
        )
        print("colormap", getattr(app_state, "spec_colormap", None))
