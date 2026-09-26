import sys
from qtpy.QtWidgets import QApplication, QScrollArea
app = QApplication.instance() or QApplication(sys.argv)
from ethograph.gui.dialog_fit import install_dialog_fitter
install_dialog_fitter(app)
sys.path.insert(0, "tests/test_unit")
import test_dialog_bulk_labels as t
import tempfile, pathlib
from ethograph.gui.app_state import ObservableAppState
state = ObservableAppState()
state._yaml_path = str(pathlib.Path(tempfile.mkdtemp()) / "g.yaml")
state._all_labels_df = t._labels_df()
from ethograph.gui.dialog_bulk_labels import LabelBulkEditDialog
meta = t._Meta(state, t._LabelsStub(t.MAPPINGS, None), t._TrialsStub(["0", "1"]))
d = LabelBulkEditDialog(meta)
print("before", d.sizeHint(), d.minimumSizeHint())
d.show(); app.processEvents()
print("after", d.size(), d.minimumSizeHint(), d.findChild(QScrollArea) is not None, d.screen().availableGeometry())
d.resize(d.width(), 300); app.processEvents(); print("shrunk", d.size())
d.grab().save(sys.argv[1])
