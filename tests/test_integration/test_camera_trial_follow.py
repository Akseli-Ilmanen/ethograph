"""Extra camera views must follow the trial, like the primary and the pose do.

Extra views are layout instances dropped from the add-panel popup, so they
appear in no camera combo. The trial-change refresh used to iterate
``_extra_camera_combos`` instead of the live views, which left every
drag-dropped panel frozen on whichever trial was open when it was created:
cam-2 kept showing trial 12's file while its pose overlay — which does iterate
the live views — correctly moved on.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QApplication


def _extra_camera_views(vm):
    return [v for v in vm.extra_widgets.values() if not getattr(v, "static_image_path", None)]


def _goto_trial(meta, trial):
    meta.navigation_widget.navigate_to_trial(str(trial))
    QApplication.processEvents()


class TestExtraCameraFollowsTrial:
    def _setup(self, lockbox_gui):
        """Drop a second camera exactly as the add-panel popup does."""
        shell, meta = lockbox_gui
        dw = meta.data_widget
        vm = dw.video_mgr
        cameras = [str(c) for c in meta.app_state.nwb_alignment.cameras]
        extra = next(c for c in cameras if c != meta.app_state.primary_camera)
        meta._add_camera_view(extra)
        QApplication.processEvents()
        return meta, vm, extra

    def test_drop_creates_a_view_outside_the_combos(self, lockbox_gui):
        meta, vm, extra = self._setup(lockbox_gui)
        assert vm.views_for_camera(extra), "dropping a camera must create a view"
        assert extra not in meta.data_widget._get_desired_extra_cameras(), (
            "a popup-dropped view is in no combo — the refresh must not be driven off them"
        )

    def test_view_reloads_on_trial_change(self, lockbox_gui):
        meta, vm, extra = self._setup(lockbox_gui)
        trials = list(meta.app_state.trials)
        assert len(trials) > 1

        view = vm.views_for_camera(extra)[0]
        _goto_trial(meta, trials[0])
        first = view.source_video_path
        _goto_trial(meta, trials[1])
        second = view.source_video_path

        assert first and second
        assert first != second, f"extra camera stayed on {Path(first).name} after switching trial"

    def test_view_shows_the_file_the_alignment_names(self, lockbox_gui):
        meta, vm, extra = self._setup(lockbox_gui)
        sio = meta.app_state.nwb_alignment
        view = vm.views_for_camera(extra)[0]

        for trial in meta.app_state.trials:
            _goto_trial(meta, trial)
            expected = sio.resolve_media_path(trial, "video", device=extra, fallback_folder=meta.app_state.video_folder)
            assert expected is not None
            assert Path(view.source_video_path).name == Path(expected).name

    def test_extra_stays_in_step_with_the_primary(self, lockbox_gui):
        """Primary and extra must land on the same trial, not drift apart."""
        meta, vm, extra = self._setup(lockbox_gui)
        view = vm.views_for_camera(extra)[0]

        for trial in meta.app_state.trials:
            _goto_trial(meta, trial)
            primary_stem = Path(vm.primary_view.source_video_path).stem
            extra_stem = Path(view.source_video_path).stem
            # Lockbox names each file "{trial}_{...}_{camera}" — the trial part
            # must agree even though the camera part differs.
            assert primary_stem.split("_mouse")[0] == extra_stem.split("_mouse")[0]

    def test_switching_back_restores_the_first_file(self, lockbox_gui):
        """The reload is not a one-way latch, and the view stays playable."""
        meta, vm, extra = self._setup(lockbox_gui)
        trials = list(meta.app_state.trials)
        view = vm.views_for_camera(extra)[0]

        _goto_trial(meta, trials[0])
        first = view.source_video_path
        _goto_trial(meta, trials[1])
        _goto_trial(meta, trials[0])

        assert view.source_video_path == first
        assert view.has_video
