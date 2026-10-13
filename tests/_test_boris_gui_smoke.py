import os
from pathlib import Path

from qtpy.QtWidgets import QApplication


def test_boris_session_loads(gui):
    viewer, meta = gui
    session = Path(os.environ["TEMP"]) / "claude/boris_out/moll2025_concat"
    meta.app_state.project_path = str(session.parent)
    meta.app_state.nc_file_path = str(session)
    meta.io_widget.nc_file_path_edit.setText(str(session))
    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    assert meta.app_state.ready
    print("TRIALS", meta.app_state.trials)
    df = meta.app_state._all_labels_df
    print("LABELS", len(df), df["labels"].unique()[:5])
    print("MAPPING", meta.labels_widget._mapping_file_path)
    print("VIDEO", meta.app_state.video_folder, meta.app_state.nwb_alignment.individuals)
