import numpy as np

from ethograph.gui.app_constants import PANEL_RIGHT_GUTTER_PX
from ethograph.gui.plots_lineplot import LinePlot, PlotData, render_plot_data


def test_probe(qapp, app_state):
    a, b = LinePlot(app_state), LinePlot(app_state)
    for w in (a, b):
        w.resize(800, 300)
        w.show()
    for _ in range(3):
        qapp.processEvents()
    t = np.linspace(0, 1, 50)
    cps = {"distance_decrease_onset_cp": np.array([0.2, 0.5]), "distance_decrease_end_cp": np.array([0.3])}
    render_plot_data(
        b.plot_item, PlotData(time=t, data=t, title="f", ylabel="y", changepoints=cps), legend_host=b.legend_host()
    )
    for _ in range(3):
        qapp.processEvents()
    print(
        "\nA right",
        a.plot_item.vb.geometry().right(),
        "B right",
        b.plot_item.vb.geometry().right(),
        "host min",
        b.legend_host().minimumWidth(),
        "gutter",
        PANEL_RIGHT_GUTTER_PX,
        "col5",
        b.plot_item.layout.columnMinimumWidth(5),
    )
