"""Ad-hoc: render the Labels tab and dump its geometry + a screenshot."""

import os

from qtpy.QtWidgets import QApplication, QLabel

OUT = os.environ.get("ETO_SHOT", "labels_tab.png")


def test_dump_labels_tab(moll2025_gui, qtbot):
    shell, meta = moll2025_gui
    shell.resize(1280, 720)
    shell.show()
    meta.collapsible_widgets[1].expand()
    for _ in range(5):
        QApplication.processEvents()
    qtbot.wait(300)

    lw = meta.labels_widget
    cp = lw.curation_panel
    tables = [s["table"] for s in lw._branch_sections.values()]
    scroll = tables[0].parent()
    while scroll is not None and type(scroll).__name__ != "_FitContentScrollArea":
        scroll = scroll.parent()
    print("branches:", len(tables), "table h:", [t.height() for t in tables])
    print("rows:", tables[0].rowCount(), "rowH:", tables[0].rowHeight(0), "hdr:", tables[0].horizontalHeader().height(), "vp:", tables[0].viewport().height())
    print("scroll h:", scroll.height(), "hint:", scroll.sizeHint().height())
    print("curation y:", cp.y(), "h:", cp.height(), "labels widget h:", lw.height())
    print("groupbox font pt/bold:", cp.font().pointSizeF(), cp.font().bold())
    child = cp.findChildren(QLabel)[0]
    print("child label font pt/bold:", child.font().pointSizeF(), child.font().bold(), repr(child.text()))
    print("mode combo tooltips:", [cp.mode_combo.itemData(i, 3) for i in range(cp.mode_combo.count())])
    lw.grab().save(OUT)
