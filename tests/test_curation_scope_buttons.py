"""The Curation section's grid buttons open on the dropped scope, and an empty
review queue says why when the method filter alone emptied it."""

from qtpy.QtCore import QMimeData, QPointF, Qt
from qtpy.QtGui import QDropEvent


def _drop(panel, ids):
    mime = QMimeData()
    mime.setText(",".join(str(i) for i in ids))
    panel.scope_area.dropEvent(QDropEvent(QPointF(5, 5), Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier))


def test_grid_buttons_open_on_dropped_scope(moll2025_gui):
    _win, meta = moll2025_gui
    panel = meta.labels_widget.curation_panel
    ids = sorted(meta.app_state._all_labels_df["labels"].unique().tolist())[:1]
    _drop(panel, ids)
    assert panel.scope_or_all_ids() == ids
    # QPushButton.clicked carries ``checked=False``; it must not become label_ids.
    panel.grid_btn.click()
    assert panel._grid_dialog.setup.selected_label_ids() == ids
    panel._grid_dialog.close()
    panel.video_grid_btn.click()
    assert panel._video_dialog.setup.selected_label_ids() == ids
    panel._video_dialog.close()


def test_empty_queue_names_the_method_filter(moll2025_gui, monkeypatch):
    _win, meta = moll2025_gui
    panel = meta.labels_widget.curation_panel
    df = meta.app_state._all_labels_df
    assert (df["labeling_method"] == "manual").all()
    messages = []
    monkeypatch.setattr("ethograph.gui.widgets_curation.notify", lambda msg, **kw: messages.append(msg))
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("segment"))
    panel.automated_only_cb.setChecked(True)
    assert not panel.start_review()
    assert "Show automated only" in messages[-1]
    assert panel.automated_only_cb.isChecked()
    panel.automated_only_cb.setChecked(False)
    assert panel.start_review()
    panel._stop()
