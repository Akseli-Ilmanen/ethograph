def test_dbg(moll2025_gui):
    viewer, meta = moll2025_gui
    pc = meta.plot_container
    pc.add_lineplot(feature="position")
    before = list(pc._dyn_panels)
    orig = pc.apply_layout_state
    def spy(state):
        print("APPLY", [e for e in state["panels"]])
        orig(state)
        print("AFTER", [p in before for p in pc._dyn_panels])
    pc.apply_layout_state = spy
    meta.reset_panels()
    print("END", [p in before for p in pc._dyn_panels], len(pc._dyn_panels))
