"""Ad-hoc probe: load the user's pynapple folder through the real GUI and list every file written."""

from pathlib import Path

B = Path(r"C:\Users\aksel\Documents\AI_data\derivatives\sub-01_id-Ivy\ses-000_date-20260417_01\behav")


def _snap():
    return {str(p): p.stat().st_mtime for p in B.rglob("*") if p.is_file()}


def test_probe(gui, qtbot):
    shell, meta = gui
    st = meta.app_state
    io = meta.io_widget
    before = _snap()
    io.nc_file_path_edit.setText(str(B / "pynapple"))
    st.nc_file_path = str(B / "pynapple")
    io.nwb_file_path_edit.setText(str(B / ".ethograph" / "alignment.nwb"))
    st.nwb_file_path = str(B / ".ethograph" / "alignment.nwb")
    st.metadata_path = str(B / "metadata.tsv")
    st.labels_import_path = str(B / "labels.tsv")
    io.neurons_path_edit.setText(str(B / "pynapple" / "units.npz"))
    st.neurons_path = str(B / "pynapple" / "units.npz")
    st.video_folder = r"C:\Users\aksel\Documents\VidData\20260417_01_Ivy"
    mid = _snap()
    print("\nNEW after setting fields:", sorted(set(mid) - set(before)))
    meta.data_widget.on_load_clicked()
    qtbot.wait(2000)
    after = _snap()
    print("TRIALS:", len(st.trials or []), "has_neurons:", st.has_neurons)
    print("NEW after load:", sorted(set(after) - set(mid)))
    print("MODIFIED after load:", sorted(k for k in mid if k in after and after[k] != mid[k]))
    print("nwb_file_path now:", st.nwb_file_path)
