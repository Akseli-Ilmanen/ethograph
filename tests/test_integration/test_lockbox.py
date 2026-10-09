"""Lockbox integration tests: multi-camera, camera combos, trial navigation."""

from qtpy.QtWidgets import QApplication


class TestLockbox:
    def test_state_after_load(self, lockbox_gui):
        _, meta = lockbox_gui
        s = meta.app_state
        assert s.ready is True
        assert s.dt is not None

    def test_three_cameras_exist(self, lockbox_gui):
        _, meta = lockbox_gui
        cameras = meta.app_state.nwb_alignment.cameras
        assert len(cameras) == 3
        assert "front-view" in cameras
        assert "side-view" in cameras
        assert "top-down-view" in cameras

    def test_primary_camera_combo_exists(self, lockbox_gui):
        _, meta = lockbox_gui
        dw = meta.data_widget
        assert hasattr(dw, "primary_camera_combo")
        combo = dw.primary_camera_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        assert len(items) == 3
        assert "front-view" in items
        assert "side-view" in items
        assert "top-down-view" in items

    def test_extra_camera_combos_exist(self, lockbox_gui):
        _, meta = lockbox_gui
        dw = meta.data_widget
        assert hasattr(dw, "_extra_camera_combos")
        assert len(dw._extra_camera_combos) >= 2

    def test_select_camera_side(self, lockbox_gui):
        _, meta = lockbox_gui
        dw = meta.data_widget
        combo = dw._extra_camera_combos[0]
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "side-view" in items
        idx = items.index("side-view")
        combo.setCurrentIndex(idx)
        QApplication.processEvents()

    def test_select_camera_top(self, lockbox_gui):
        _, meta = lockbox_gui
        dw = meta.data_widget
        combo = dw._extra_camera_combos[1] if len(dw._extra_camera_combos) > 1 else dw._extra_camera_combos[0]
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "top-down-view" in items
        idx = items.index("top-down-view")
        combo.setCurrentIndex(idx)
        QApplication.processEvents()

    def test_multiple_trials(self, lockbox_gui):
        _, meta = lockbox_gui
        assert len(meta.app_state.trials) == 3

    def test_trial_navigation(self, lockbox_gui):
        _, meta = lockbox_gui
        meta.navigation_widget.scope_combo.setCurrentText("Trial start → Trial end")
        QApplication.processEvents()
        trials = meta.app_state.trials
        meta.navigation_widget.navigate_to_trial(str(trials[0]))
        QApplication.processEvents()
        first = meta.app_state.trials_sel
        meta.navigation_widget.next_trial()
        QApplication.processEvents()
        assert meta.app_state.trials_sel != first

    def test_features_available(self, lockbox_gui):
        _, meta = lockbox_gui
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        assert combo.count() > 0
