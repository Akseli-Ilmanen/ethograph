"""Ad-hoc: count sidebar combos after switching stages back and forth."""

import numpy as np
from qtpy.QtWidgets import QApplication, QComboBox, QFormLayout

from ethograph.labels import pose_project as pp


def _count(layout: QFormLayout, label: str) -> int:
    n = 0
    for i in range(layout.rowCount()):
        item = layout.itemAt(i, QFormLayout.LabelRole)
        if item is not None and item.widget() is not None and item.widget().text() == label:
            n += 1
    return n


def test_dupes(gui, pose_project_root):
    shell, meta = gui
    mode = meta.pose_project
    assert mode.enter(pose_project_root, parent=None)
    QApplication.processEvents()
    dw = meta.data_widget
    before = (
        _count(dw.individual_layout, "Individual:"),
        _count(dw.coords_groupbox_layout, "Keypoint:"),
        _count(dw.coords_groupbox_layout, "Space:"),
        len(dw.individual_groupbox.findChildren(QComboBox)),
    )
    video = mode.project.videos()[0]
    pp.extract_frames(mode.project, video, [1], lambda f: np.zeros((32, 32, 3), np.uint8), 20)
    assert mode.switch(pp.STAGE_REFINE, parent=None)
    QApplication.processEvents()
    assert mode.switch(pp.STAGE_EXTRACT, parent=None)
    QApplication.processEvents()
    after = (
        _count(dw.individual_layout, "Individual:"),
        _count(dw.coords_groupbox_layout, "Keypoint:"),
        _count(dw.coords_groupbox_layout, "Space:"),
        len(dw.individual_groupbox.findChildren(QComboBox)),
    )
    print("BEFORE", before, "AFTER", after)
    visible = [c.objectName() for c in dw.coords_groupbox.findChildren(QComboBox) if not c.isHidden()]
    print("VISIBLE coords combos", visible)
    print(
        "VISIBLE indiv combos",
        [c.objectName() for c in dw.individual_groupbox.findChildren(QComboBox) if not c.isHidden()],
    )
    assert before == after
