"""Each imported prediction file is drawn on its own panel, and nowhere else.

The labels and a prediction file are compared by stacking panels, so a
prediction slot must stay on its panel, the working labels must stay off it,
and re-importing a file must not open a second panel for it.
"""

from pathlib import Path

import pandas as pd
import pyqtgraph as pg
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.label_drawing_mixin import LabelDrawingMixin  # noqa: E402
from ethograph.labels.predictions import PredictionSet, add_prediction_set  # noqa: E402

LABEL_ID = 3


class _Container(LabelDrawingMixin):
    def __init__(self, plots):
        self._plots = plots
        self.app_state = type("_State", (), {"label_overlay_modes": {}})()
        self.label_mappings = {LABEL_ID: {"name": "peck", "color": (1.0, 0.0, 0.0)}}

    def _get_all_plots(self) -> list:
        return list(self._plots)


def _plot(qtbot, panel_type: str):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.plot_item = widget.getPlotItem()
    widget.panel_type = panel_type
    return widget


def _df(onset: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"labels": LABEL_ID, "onset_s": onset, "offset_s": onset + 0.5, "event_type": "state", "individual": "a"}]
    )


def test_labels_and_predictions_each_stay_on_their_own_panels(qtbot):
    feature = _plot(qtbot, "lineplot")
    run_a = _plot(qtbot, "predictions")
    run_b = _plot(qtbot, "predictions")
    container = _Container([feature, run_a, run_b])

    labels_slot = {"df": _df(1.0), "label_ids": {LABEL_ID}, "position": "main"}
    container.draw_all_labels([labels_slot])
    n_label_items = len(feature.label_items)

    container.draw_all_labels([labels_slot, {"df": _df(2.0), "label_ids": None, "position": "main", "plots": [run_a]}])

    assert n_label_items and len(feature.label_items) == n_label_items, "a prediction leaked onto a feature panel"
    assert len(run_a.label_items) == n_label_items, "the prediction was not drawn (alone) on its panel"
    assert not run_b.label_items, "a prediction panel drew rows that are not its file's"


def test_reimporting_a_file_replaces_it_rather_than_adding_a_second():
    path = Path("run_predictions.tsv")
    sets = add_prediction_set([], PredictionSet(path, _df(1.0)))
    sets = add_prediction_set(sets, PredictionSet(Path("other_predictions.tsv"), _df(1.0)))
    sets = add_prediction_set(sets, PredictionSet(path, _df(5.0)))

    assert [s.path for s in sets] == [path, Path("other_predictions.tsv")]
    assert sets[0].labels_df["onset_s"].iloc[0] == 5.0
