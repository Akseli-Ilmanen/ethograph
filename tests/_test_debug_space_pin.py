"""Ad-hoc probe: what the space plot offers on birdpark (not collected by pytest)."""


def test_debug(birdpark_gui):
    _, meta = birdpark_gui
    dw = meta.data_widget
    sp = dw.add_space_plot()
    fc = sp.feature_combo
    print("FEATURES", [fc.itemText(i) for i in range(fc.count())], "current", fc.currentText())
    print("STORE", sp._store is not None, "DIMS", sp._feature_dims(), "space_dim", sp.space_dim_combo.currentText())
    print("DIM_COMBOS", list(sp._dim_combos))
    for i in range(fc.count()):
        f = fc.itemText(i)
        print(" ", f, meta.app_state.data_loader.feature_dims(f))
