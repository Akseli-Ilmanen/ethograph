"""A Space plot shows 3D poses loaded from the labelling dialog, and the dialog's live points on top.

One `birdpark_gui` load, several assertions: see test_keypoint_dataset_load.py
for why loads are rationed.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
import xarray as xr
from qtpy.QtWidgets import QApplication

N_FRAMES = 40


def _poses_3d() -> xr.Dataset:
    t = np.arange(N_FRAMES) / 20.0
    position = np.stack([np.sin(t), np.cos(t), t], axis=1)[:, :, None, None]
    return xr.Dataset(
        {"position": (("time", "space", "keypoint", "individual"), position)},
        coords={"time": t, "space": ["x", "y", "z"], "keypoint": ["beak"], "individual": ["bird"]},
        attrs={"ds_type": "poses", "fps": 20.0, "space_unit": "m"},
    )


def _live_items(plot) -> list:
    widget = plot.space_widget
    holder = widget if isinstance(widget, gl.GLViewWidget) else widget.getPlotItem()
    return [item for item in holder.items if getattr(item, "_is_live", False)]


def test_an_open_space_plot_follows_the_loaded_poses_and_keeps_the_live_points(birdpark_gui):
    _, meta = birdpark_gui
    data_widget = meta.data_widget
    plot = data_widget.add_space_plot(view_3d=False)
    QApplication.processEvents()

    assert data_widget.load_keypoint_dataset(_poses_3d())
    QApplication.processEvents()
    # The plot was open before the load: it must now read the new dataset, not the one it replaced.
    assert plot.feature_combo.currentText() == "position"
    assert [plot.z_combo.itemText(i) for i in range(plot.z_combo.count())] == ["x", "y", "z"]

    plot.set_live_points(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), ["#ff0000", "#00ff00"])
    assert len(_live_items(plot)) == 1
    plot.refresh()
    QApplication.processEvents()
    assert len(_live_items(plot)) == 1  # a re-render redraws the trajectory, not the dialog's points

    plot.set_live_points(None)
    assert not _live_items(plot)
