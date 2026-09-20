"""The branch-table area of the Labels tab is as tall as its branches, no taller."""

from qtpy.QtWidgets import QApplication, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ethograph.gui.widgets_labels import _FitContentScrollArea


def _branch(height: int) -> QWidget:
    widget = QWidget()
    widget.setFixedHeight(height)
    return widget


def test_area_takes_only_its_branches_height(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    area = _FitContentScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(_FitContentScrollArea.NoFrame)
    area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    content = QWidget()
    branches = QVBoxLayout(content)
    branches.setContentsMargins(0, 0, 0, 0)
    branches.setSpacing(0)
    branches.addWidget(_branch(100))
    area.setWidget(content)
    outer.addWidget(area)
    below = QPushButton("below")
    outer.addWidget(below)
    outer.addStretch(1)

    host.resize(300, 800)
    host.show()
    QApplication.processEvents()
    assert area.height() == 100
    assert below.y() == 100

    # A second branch claims its own height; the spare room was never reserved.
    branches.addWidget(_branch(60))
    QApplication.processEvents()
    QApplication.processEvents()
    assert area.height() == 160
    assert below.y() == 160
