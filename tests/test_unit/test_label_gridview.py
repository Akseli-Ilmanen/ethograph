"""Pure logic of the label grid view (Labels ▸ Curation ▸ Label grid view…)."""

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QApplication, QCheckBox, QLabel, QListWidget, QListWidgetItem, QWidget

from ethograph.gui import dialog_label_gridview as gv
from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_label_gridview import (
    DEFAULT_CONFIDENCE,
    LOW_CONFIDENCE_COLOR,
    ConfidenceEdit,
    FrameEntry,
    LabelGridView,
    LabelGridViewDialog,
    TileVerdicts,
    _draw_disc,
    _entry_info,
    _entry_title,
    build_frame_entries,
    capture_panel_images,
    confidence_groups,
    confidence_style,
    confidence_text,
    crop_thumbnail,
    decode_entry_images,
    draw_pose_points,
    filter_entries,
    flagged_trials,
    histogram_bar_color,
    is_low_confidence,
    label_filter_choices,
    methods_for_filter,
    open_gui_panels,
    pack_tiles,
    seeds_from_entries,
    split_histogram,
)
from ethograph.gui.pose_render import PoseRenderData
from ethograph.labels.intervals import LABELING_AUTOMATED, LABELING_CURATED, LABELING_MANUAL

MAPPINGS = {
    1: {"name": "peck", "event_type": "point", "color": (1.0, 0.0, 0.0)},
    2: {"name": "hop", "event_type": "state", "color": (0.0, 1.0, 0.0)},
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def labels_df():
    return pd.DataFrame(
        {
            "trial": [1, 1, 2, 3],
            "labels": [1, 2, 2, 1],
            "onset_s": [0.5, 1.0, 2.0, 3.0],
            "offset_s": [np.nan, 1.8, 2.6, np.nan],
            "individual": ["a", "a", "b", "a"],
            "individual_rec": ["", "", "", ""],
        }
    )


class TestMethodFilter:
    """The frame grid's half of the shared labeling-method filter.

    A labels frame with no ``labeling_method`` column is read off its
    confidence, exactly as the rest of the GUI reads it — so an imported file
    still filters instead of vanishing.
    """

    def test_each_choice_keeps_its_own_side(self, labels_df):
        df = labels_df.copy()
        df["labeling_method"] = [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_CURATED, LABELING_AUTOMATED]
        automated = build_frame_entries(df, MAPPINGS, [1, 2], [None], None, methods_for_filter("automated"))
        human = build_frame_entries(df, MAPPINGS, [1, 2], [None], None, methods_for_filter("human"))
        assert {e.labeling_method for e in automated} == {LABELING_AUTOMATED}
        assert {e.labeling_method for e in human} == {LABELING_MANUAL, LABELING_CURATED}
        # Between them they are the whole grid, and neither drops a boundary.
        assert len(automated) + len(human) == len(build_frame_entries(df, MAPPINGS, [1, 2], [None]))

    def test_manual_and_curated_split_out_of_human(self, labels_df):
        df = labels_df.copy()
        df["labeling_method"] = [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_CURATED, LABELING_AUTOMATED]
        manual = build_frame_entries(df, MAPPINGS, [1, 2], [None], None, methods_for_filter("manual"))
        curated = build_frame_entries(df, MAPPINGS, [1, 2], [None], None, methods_for_filter("curated"))
        human = build_frame_entries(df, MAPPINGS, [1, 2], [None], None, methods_for_filter("human"))
        assert {e.labeling_method for e in manual} == {LABELING_MANUAL}
        assert {e.labeling_method for e in curated} == {LABELING_CURATED}
        assert len(manual) + len(curated) == len(human)

    def test_a_file_without_the_column_reads_off_its_confidence(self, labels_df):
        df = labels_df.copy()
        df["confidence"] = [0.4, 1.0, 1.0, 0.7]
        entries = build_frame_entries(df, MAPPINGS, [1, 2], [None], None, methods_for_filter("automated"))
        assert {e.trial for e in entries} == {1, 3}


class TestBuildFrameEntries:
    def test_one_tile_per_label_whatever_its_kind(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1, 2], [None])
        by_label = {}
        for e in entries:
            by_label.setdefault(e.label_id, []).append(e)
        # 2 point rows and 2 state rows -> one tile each; a state tile is two frames.
        assert len(by_label[1]) == 2
        assert len(by_label[2]) == 2
        assert all(e.boundary == "point" and e.span == 1 for e in by_label[1])
        assert all(e.boundary == "state" and e.span == 2 for e in by_label[2])

    def test_a_state_tile_holds_onset_and_offset(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [2], [None])
        tile = entries[0]
        assert tile.t_rel == 1.0 and tile.offset_s == 1.8
        assert tile.frames() == [("start", 1.0), ("end", 1.8)]

    def test_label_filter(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1], [None])
        assert {e.label_id for e in entries} == {1}

    def test_cameras_multiply_entries(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1], ["cam-1", "cam-2"])
        assert len(entries) == 4
        assert {e.camera for e in entries} == {"cam-1", "cam-2"}

    def test_allowed_trials_filter(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1, 2], [None], allowed_trials={"1"})
        assert {str(e.trial) for e in entries} == {"1"}

    def test_unmapped_label_falls_back_to_state(self, labels_df):
        df = labels_df.assign(labels=[7, 7, 7, 7])
        entries = build_frame_entries(df, MAPPINGS, [7], [None])
        # Finite-offset rows become state tiles; NaN offsets stay points.
        assert [e.boundary for e in entries if str(e.trial) == "1"] == ["point", "state"]

    def test_empty_df(self):
        assert build_frame_entries(pd.DataFrame(), MAPPINGS, [1], [None]) == []
        assert build_frame_entries(None, MAPPINGS, [1], [None]) == []

    def test_confidence_read_from_rows(self):
        df = pd.DataFrame(
            {
                "trial": [1, 2],
                "labels": [1, 1],
                "onset_s": [0.5, 1.5],
                "offset_s": [np.nan, np.nan],
                "individual": ["a", "a"],
                "individual_rec": ["", ""],
                "confidence": [0.4, 1.0],
            }
        )
        entries = build_frame_entries(df, MAPPINGS, [1], [None])
        assert [e.confidence for e in entries] == [0.4, 1.0]

    def test_missing_confidence_column_reads_as_certain(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1], [None])
        assert all(e.confidence == 1.0 for e in entries)

    def test_mapping_color_carried(self, labels_df):
        entries = build_frame_entries(labels_df, MAPPINGS, [1], [None])
        assert entries[0].color_hex == "#ff0000"


class TestTitles:
    def test_point_has_no_boundary_suffix(self):
        entry = FrameEntry(
            trial=2,
            camera="cam-1",
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=0.5,
            onset_s=0.5,
            offset_s=float("nan"),
            individual="a",
        )
        assert _entry_title(entry) == "peck (1)"
        assert _entry_info(entry) == "trial 2  ·  cam-1  ·  a  ·  at 0.500 s  ·  manual"

    def test_state_reads_span_and_duration(self):
        entry = FrameEntry(
            trial=1,
            camera=None,
            label_id=2,
            name="hop",
            event_type="state",
            boundary="state",
            t_rel=1.0,
            onset_s=1.0,
            offset_s=1.8,
        )
        assert _entry_title(entry) == "hop (2)"
        assert _entry_info(entry) == "trial 1  ·  1.000–1.800 s  ·  0.800 s  ·  manual"

    def test_cropped_entry_says_so(self):
        entry = FrameEntry(
            trial=1,
            camera="cam-1",
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=0.5,
            onset_s=0.5,
            offset_s=float("nan"),
            cropped=True,
        )
        assert _entry_info(entry) == "trial 1  ·  cam-1  ·  at 0.500 s  ·  manual  ·  cropped"

    def test_predicted_confidence_shown(self):
        entry = FrameEntry(
            trial=1,
            camera=None,
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=0.5,
            onset_s=0.5,
            offset_s=float("nan"),
            confidence=0.42,
        )
        assert confidence_text(entry) == "0.420"
        # Red exactly when the tile is flagged — the two must not disagree.
        assert LOW_CONFIDENCE_COLOR in confidence_style(entry, 0.5)
        assert LOW_CONFIDENCE_COLOR not in confidence_style(entry, 0.4)


class TestConfidenceEdit:
    """The threshold is typed, so a score of 0.0002 is reachable."""

    def test_small_thresholds_survive_the_round_trip(self, qapp):
        edit = ConfidenceEdit(0.0002)
        assert edit.text() == "0.0002"
        assert edit.value() == pytest.approx(0.0002)

    def test_typing_announces_the_number_typed(self, qapp):
        edit = ConfidenceEdit(DEFAULT_CONFIDENCE)
        seen = []
        edit.valueChanged.connect(seen.append)
        edit.setText("0.00025")
        assert edit.value() == pytest.approx(0.00025)
        assert seen == [pytest.approx(0.00025)]

    def test_typing_one_key_at_a_time_lands_on_the_number(self, qapp):
        """Every prefix of a small threshold is itself a valid number, so the
        box follows the keystrokes and ends where the typing ended."""
        edit = ConfidenceEdit(0.25)
        for text in ("0", "0.", "0.0", "0.00", "0.000", "0.0002"):
            edit.setText(text)
        assert edit.value() == pytest.approx(0.0002)

    def test_text_no_number_can_be_read_out_of_keeps_the_value(self, qapp):
        edit = ConfidenceEdit(0.25)
        edit.setText("half")
        assert edit.value() == pytest.approx(0.25)
        edit.editingFinished.emit()
        assert edit.text() == "0.25"

    def test_empty_is_off_and_out_of_range_is_clamped(self, qapp):
        edit = ConfidenceEdit(0.25)
        edit.setText("")
        assert edit.value() == 0.0
        edit.setValue(5.0)
        assert edit.value() == 1.0

    def test_two_boxes_bound_to_each_other_settle(self, qapp):
        """The grid's box and the histogram's are bound both ways."""
        left, right = ConfidenceEdit(0.1), ConfidenceEdit(0.1)
        left.valueChanged.connect(right.setValue)
        right.valueChanged.connect(left.setValue)
        left.setValue(0.0004)
        assert right.value() == pytest.approx(0.0004)
        assert left.value() == pytest.approx(0.0004)


class TestConfidenceFlagging:
    def test_threshold_zero_flags_nothing(self):
        entry = FrameEntry(
            trial=1,
            camera=None,
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=0.5,
            onset_s=0.5,
            offset_s=float("nan"),
            confidence=0.0,
        )
        assert not is_low_confidence(entry, 0.0)
        assert is_low_confidence(entry, 0.5)

    def test_human_label_never_flagged(self):
        entry = FrameEntry(
            trial=1,
            camera=None,
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=0.5,
            onset_s=0.5,
            offset_s=float("nan"),
        )
        assert entry.confidence == 1.0
        assert not is_low_confidence(entry, 1.0)

    def test_grid_outlines_only_low_tiles(self, qapp):
        low = FrameEntry(
            trial=1,
            camera=None,
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=0.5,
            onset_s=0.5,
            offset_s=float("nan"),
            confidence=0.3,
        )
        high = FrameEntry(
            trial=2,
            camera=None,
            label_id=1,
            name="peck",
            event_type="point",
            boundary="point",
            t_rel=1.5,
            onset_s=1.5,
            offset_s=float("nan"),
            confidence=0.9,
        )
        dlg = LabelGridView(_Meta(ObservableAppState(), None), [low, high])
        try:
            assert dlg._cells[0].styleSheet() == ""  # threshold starts off
            dlg.threshold_edit.setValue(0.5)
            assert LOW_CONFIDENCE_COLOR in dlg._cells[0].styleSheet()
            assert dlg._cells[1].styleSheet() == ""
        finally:
            dlg.close()


class TestCropThumbnail:
    def test_crop_slices_source_pixels(self):
        image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        out = crop_thumbnail(image, (4, 2, 10, 8), scale=1.0)
        assert out.shape == (6, 6, 3)
        assert np.array_equal(out, image[2:8, 4:10])

    def test_crop_rect_divided_by_decode_scale(self):
        # Source 40x60 decoded at half size: rect in source pixels lands halved.
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        out = crop_thumbnail(image, (10, 4, 30, 16), scale=2.0)
        assert out.shape == (6, 10, 3)

    def test_crop_clamps_to_image(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        out = crop_thumbnail(image, (-5, -5, 100, 100), scale=1.0)
        assert out.shape == (20, 30, 3)

    def test_degenerate_crop_returns_unchanged(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        out = crop_thumbnail(image, (10, 10, 10, 10), scale=1.0)
        assert out.shape == (20, 30, 3)


class _LabelsStub(QWidget):
    def __init__(self, mappings):
        super().__init__()
        self._mappings = mappings

    def refresh_labels_shapes_layer(self):
        pass


class _Meta:
    def __init__(self, app_state, labels_widget, nav=None, data_widget=None):
        self.app_state = app_state
        self.labels_widget = labels_widget
        self.navigation_widget = nav
        self.data_widget = data_widget
        self.io_widget = None


class TestConfigDialog:
    @pytest.fixture()
    def dialog(self, qapp, tmp_path, labels_df):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        state.metadata_df = pd.DataFrame({"trial": [1, 2, 3], "genotype": ["wt", "ko", "wt"]})
        dlg = LabelGridViewDialog(_Meta(state, _LabelsStub({0: {"name": "none"}, **MAPPINGS})), label_ids=[0, 1, 2])
        yield dlg
        dlg.close()

    def test_label_list_echoes_the_scope_read_only(self, dialog):
        ids = [dialog.label_list.item(i).data(Qt.UserRole) for i in range(dialog.label_list.count())]
        assert ids == [1, 2]  # background never; the scope, in order
        assert not (dialog.label_list.item(0).flags() & Qt.ItemIsUserCheckable)
        assert dialog.setup.selected_label_ids() == [1, 2]

    def test_columns_are_global_and_remembered(self, dialog):
        """The grid's column count lives in gui_settings.yaml — a viewing
        habit that follows the user across datasets (so does the panel
        capture's time window, label_grid_window_s)."""
        from ethograph.gui.app_state import AppStateSpec

        for key in ("label_grid_columns", "label_grid_window_s"):
            assert AppStateSpec.get_meta(key)[3] == AppStateSpec.SCOPE_GLOBAL
        dialog.app_state.trials = [1, 3]
        dialog._generate()
        assert dialog.grid_view.columns_spin.value() == 3
        dialog.grid_view.columns_spin.setValue(5)
        assert dialog.app_state.label_grid_columns == 5
        dialog._generate()
        assert dialog.grid_view.columns_spin.value() == 5

    def test_the_trials_table_is_the_only_trial_filter(self, dialog):
        """No filters of its own: the dialog covers what the trials table shows."""
        assert not hasattr(dialog.setup, "_filters")
        assert dialog.setup.allowed_trials() is None  # no trials known yet
        dialog.app_state.trials = [1, 3]
        assert dialog.setup.allowed_trials() == {"1", "3"}
        assert dialog.setup.trials_note.text().startswith("Runs over the 2 trial(s) the trials table")

    def test_generate_fills_the_frames_tab(self, dialog):
        """No resolvable video (EmpytAlignment) → every tile carries an error,
        and the grid tab still opens so the user sees what went wrong."""
        assert not dialog.tabs.isTabEnabled(1)
        dialog.app_state.trials = [1, 3]  # what the trials table shows
        dialog._generate()
        grid = dialog.grid_view
        assert grid is not None
        assert dialog.tabs.currentIndex() == 1 and dialog.tabs.widget(1) is grid
        # Scope [1, 2]: trial 1 holds a point (label 1) and a state (label 2,
        # one tile); trial 3 a point — trial 2 is filtered out.
        assert [str(e.trial) for e in grid._entries] == ["1", "1", "3"]
        assert all(e.image is None and e.error == "video not found" for e in grid._entries)
        assert len(grid._cells) == 3

    def test_building_the_grid_shows_no_top_level_window(self, dialog):
        """Every tile is parented before it is shown: a parentless widget
        made visible — even for the instant before a layout adopts it — is
        a top-level window, and one flashed per tile."""
        from qtpy.QtCore import QEvent, QObject

        class _Spy(QObject):
            shown: list[str] = []

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Show and isinstance(obj, QWidget) and obj.parent() is None:
                    self.shown.append(type(obj).__name__)
                return False

        spy = _Spy()
        dialog.app_state.trials = [1, 3]
        QApplication.instance().installEventFilter(spy)
        try:
            dialog._generate()
        finally:
            QApplication.instance().removeEventFilter(spy)
        assert len(dialog.grid_view._cells) == 3
        assert spy.shown == []

    def test_regenerating_replaces_the_grid_tab(self, dialog):
        """A second run swaps the tab's grid — never a second Frames tab."""
        dialog._generate()
        first = dialog.grid_view
        dialog._generate()
        assert dialog.tabs.count() == 2
        assert dialog.grid_view is not first and dialog.tabs.widget(1) is dialog.grid_view

    def test_window_offers_maximise(self, dialog):
        """The grid goes full screen from the title bar, in one click."""
        assert dialog.windowFlags() & Qt.WindowMaximizeButtonHint


def _write_video(path, n_frames=20, fps=10, size=32):
    """Synthetic video whose frame *i* is a uniform gray of value ``i * 10``."""
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(np.full((size, size, 3), i * 10, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


class _AlignmentStub:
    """Per-camera video paths; no rates, no offsets, no pose."""

    def __init__(self, paths):
        self.paths = paths

    def resolve_media_path(self, trial, stream, device=None, fallback_folder=None):
        return str(self.paths[device]) if stream == "video" and device in self.paths else None

    def get_stream_rate(self, stream, device=None):
        return None

    def stream_offset_for_trial(self, trial, stream, device=None):
        return 0.0


class TestDecodeEntryImages:
    def test_two_cameras_decode_in_parallel_groups(self, qapp, tmp_path):
        paths = {f"cam-{k}": _write_video(tmp_path / f"cam{k}.mp4") for k in (1, 2)}
        entries = [_point_entry(1, "cam-1", 0.5), _point_entry(1, "cam-2", 1.2)]
        decode_entry_images(
            entries,
            alignment=_AlignmentStub(paths),
            video_folder=None,
            pose_folder=None,
            source_software=None,
            pose_color_by="keypoint",
            camera_crops={"cam-2": (0, 0, 16, 16)},
        )
        assert [e.error for e in entries] == [None, None]
        # fps 10, offset 0: t=0.5 → frame 5 (gray 50), t=1.2 → frame 12 (gray 120).
        assert entries[0].frame_idx == 5 and entries[1].frame_idx == 12
        assert abs(float(entries[0].image.mean()) - 50) < 15
        assert abs(float(entries[1].image.mean()) - 120) < 15
        assert entries[0].image.shape == (32, 32, 3)
        assert entries[1].cropped and entries[1].image.shape == (16, 16, 3)

    def test_unresolvable_video_marks_errors_without_raising(self, qapp):
        entries = [_point_entry(1, "cam-1", 0.5)]
        decode_entry_images(
            entries,
            alignment=_AlignmentStub({}),
            video_folder=None,
            pose_folder=None,
            source_software=None,
            pose_color_by="keypoint",
        )
        assert entries[0].image is None and entries[0].error == "video not found"


class _NavStub:
    def __init__(self):
        self.jumps = []

    def jump_to_label_instance(self, inst, **kwargs):
        self.jumps.append((inst, kwargs))


def _point_entry(trial, camera, t_rel):
    return FrameEntry(
        trial=trial,
        camera=camera,
        label_id=1,
        name="peck",
        event_type="point",
        boundary="point",
        t_rel=t_rel,
        onset_s=t_rel,
        offset_s=float("nan"),
    )


class TestPanelCapture:
    def test_no_open_panels_for_stub_meta(self):
        assert open_gui_panels(_Meta(None, None)) == []

    def test_cameras_share_one_capture(self, qapp):
        """Two cameras of the same boundary → one GUI jump, shared shots."""
        panel = QLabel("plot")
        panel.setFixedSize(60, 40)
        entries = [_point_entry(1, "cam-1", 2.0), _point_entry(1, "cam-2", 2.0)]
        nav = _NavStub()
        visited = capture_panel_images(entries, nav=nav, panels=[("Lineplot", panel)], window_s=4.0)
        assert visited == 1
        assert len(nav.jumps) == 1
        inst, kwargs = nav.jumps[0]
        assert kwargs["seek_rel"] == 2.0
        assert kwargs["view_rel"].start_s == 0.0 and kwargs["view_rel"].end_s == 4.0
        assert entries[0].panels and entries[0].panels is entries[1].panels
        title, image = entries[0].panels[0]
        assert title == "Lineplot"
        assert isinstance(image, QImage) and not image.isNull()

    def test_autoscale_toggle_and_restore_autorange(self, qapp):
        """``autoscale=False`` disables y autorange for the capture and puts
        the panel's own view state back afterwards; True does the reverse."""
        plot = pg.PlotWidget()
        plot.plot([0.0, 1.0, 2.0], [0.0, 5.0, 1.0])
        vb = plot.getViewBox()
        vb.enableAutoRange(y=True)  # the user's own setting

        seen = {}

        class _AxisNav(_NavStub):
            def jump_to_label_instance(self, inst, **kwargs):
                seen["auto_y"] = bool(vb.autoRangeEnabled()[1])
                super().jump_to_label_instance(inst, **kwargs)

        capture_panel_images(
            [_point_entry(1, "cam-1", 2.0)], nav=_AxisNav(), panels=[("P", plot)], window_s=1.0, autoscale=False
        )
        assert seen["auto_y"] is False
        assert bool(vb.autoRangeEnabled()[1]) is True  # restored

        vb.enableAutoRange(y=False)  # now the user has a manual range
        capture_panel_images(
            [_point_entry(1, "cam-1", 2.0)], nav=_AxisNav(), panels=[("P", plot)], window_s=1.0, autoscale=True
        )
        assert seen["auto_y"] is True
        assert bool(vb.autoRangeEnabled()[1]) is False  # restored
        plot.close()

    def test_skip_video_suppresses_loading_during_capture(self, qapp, tmp_path):
        """With 'skip video' ticked, trial jumps run with suppress_video_load
        set, and the media reloads exactly once afterwards."""

        class _VideoMgrStub:
            def __init__(self):
                self.synced = 0

            def sync_proxies(self):
                self.synced += 1

        class _DataWidgetStub:
            def __init__(self):
                self.suppress_video_load = False
                self.video_mgr = _VideoMgrStub()
                self.calls = []

            def update_video(self):
                self.calls.append("video")

            def _init_or_update_extra_cameras(self):
                self.calls.append("extras")

            def update_pose(self):
                self.calls.append("pose")

        class _FlagNav(_NavStub):
            def __init__(self, data_widget):
                super().__init__()
                self.data_widget = data_widget
                self.flags = []

            def jump_to_label_instance(self, inst, **kwargs):
                self.flags.append(self.data_widget.suppress_video_load)
                super().jump_to_label_instance(inst, **kwargs)

        data_widget = _DataWidgetStub()
        nav = _FlagNav(data_widget)
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        dlg = LabelGridViewDialog(_Meta(state, _LabelsStub(MAPPINGS), nav=nav, data_widget=data_widget))
        try:
            panel = QLabel("plot")
            panel.setFixedSize(40, 30)
            dlg.panel_list = QListWidget()
            item = QListWidgetItem("Lineplot")
            item.setData(Qt.UserRole, panel)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            dlg.panel_list.addItem(item)
            dlg.skip_video_cb = QCheckBox()
            dlg.skip_video_cb.setChecked(True)
            dlg.window_spin = None

            entries = [_point_entry(1, "cam-1", 2.0)]
            dlg._capture_panels(entries)

            assert nav.flags == [True]  # suppressed while jumping to the seed
            assert data_widget.suppress_video_load is False
            assert data_widget.calls == ["video", "extras", "pose"]
            assert data_widget.video_mgr.synced == 1
            assert entries[0].panels
        finally:
            dlg.close()

    def test_grid_cell_stacks_panel_snapshots(self, qapp):
        entry = _point_entry(1, "cam-1", 2.0)
        entry.image = np.zeros((20, 30, 3), dtype=np.uint8)
        entry.panels = [("Lineplot — speed", QImage(40, 20, QImage.Format_RGB888))]
        dlg = LabelGridView(_Meta(ObservableAppState(), None), [entry])
        try:
            # The frame pixmap rescales to one column, the panel pixmap to the
            # tile's whole width — both on relayout.
            assert len(dlg._cells[0]._pix_labels) == 1
            assert len(dlg._cells[0]._wide_labels) == 1
        finally:
            dlg.close()

    def test_a_state_tile_shows_both_frames(self, qapp):
        entry = _entry(boundary="state", t_rel=1.0, offset_s=2.0, camera="cam-1")
        entry.image = np.zeros((20, 30, 3), dtype=np.uint8)
        entry.end_image = np.zeros((20, 30, 3), dtype=np.uint8)
        dlg = LabelGridView(_Meta(ObservableAppState(), None), [entry])
        try:
            assert len(dlg._cells[0]._pix_labels) == 2
        finally:
            dlg.close()


class TestPoseDrawing:
    def test_draw_disc_clips_at_edges(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        _draw_disc(image, 0, 0, 2, (255, 0, 0))
        assert image[0, 0, 0] == 255
        _draw_disc(image, 50, 50, 2, (255, 0, 0))  # fully outside: no raise

    def test_draw_pose_points_only_target_frame(self):
        # rows: [frame, y, x] — frame 3 lands inside, frame 4 does not.
        data = np.array([[3.0, 5.0, 5.0], [4.0, 20.0, 20.0]])
        pose = PoseRenderData(
            data=data,
            properties=pd.DataFrame({"keypoint": ["beak", "beak"], "individual": ["a", "a"]}),
            data_not_nan=np.array([True, True]),
            file_name="pose.nc",
        )
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_pose_points(image, pose, frame_idx=3, scale=1.0, color_by="keypoint")
        assert image[5, 5].any()
        assert not image[20, 20].any()

    def test_scale_divides_coordinates(self):
        data = np.array([[0.0, 10.0, 10.0]])
        pose = PoseRenderData(
            data=data,
            properties=pd.DataFrame({"keypoint": ["beak"]}),
            data_not_nan=np.array([True]),
            file_name="pose.nc",
        )
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        draw_pose_points(image, pose, frame_idx=0, scale=2.0, color_by="keypoint")
        assert image[5, 5].any()
        assert not image[10, 10].any()

    def test_hidden_keypoints_are_not_drawn(self):
        """The sidebar's keypoint filter reaches the thumbnails."""
        data = np.array([[0.0, 5.0, 5.0], [0.0, 20.0, 20.0]])
        pose = PoseRenderData(
            data=data,
            properties=pd.DataFrame({"keypoint": ["beak", "tail"], "individual": ["a", "a"]}),
            data_not_nan=np.array([True, True]),
            file_name="pose.nc",
        )
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_pose_points(image, pose, frame_idx=0, scale=1.0, color_by="keypoint", hidden_keypoints={"tail"})
        assert image[5, 5].any()
        assert not image[20, 20].any()

    def test_text_names_the_other_axis(self):
        """Colour is one axis, text the other: coloured by individual, the
        keypoint's name lands to the right of its point."""
        data = np.array([[0.0, 40.0, 10.0]])
        pose = PoseRenderData(
            data=data,
            properties=pd.DataFrame({"keypoint": ["beak"], "individual": ["a"]}),
            data_not_nan=np.array([True]),
            file_name="pose.nc",
        )
        silent = np.zeros((80, 120, 3), dtype=np.uint8)
        draw_pose_points(silent, pose, frame_idx=0, scale=1.0, color_by="individual")
        named = np.zeros((80, 120, 3), dtype=np.uint8)
        draw_pose_points(named, pose, frame_idx=0, scale=1.0, color_by="individual", show_text=True)
        assert not silent[:, 20:].any()
        assert named[:, 20:].any()

    def test_nan_rows_skipped(self):
        data = np.array([[0.0, np.nan, np.nan]])
        pose = PoseRenderData(
            data=data,
            properties=pd.DataFrame({"keypoint": ["beak"]}),
            data_not_nan=np.array([False]),
            file_name="pose.nc",
        )
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        draw_pose_points(image, pose, frame_idx=0, scale=1.0, color_by="keypoint")
        assert not image.any()


# ----------------------------------------------------------------------
# Ticking tiles -> frame-by-frame refinement
# ----------------------------------------------------------------------


def _entry(
    trial="1",
    label_id=1,
    boundary="point",
    t_rel=0.5,
    camera=None,
    confidence=1.0,
    offset_s=float("nan"),
    individual="a",
    labeling_method=LABELING_MANUAL,
    name="peck",
):
    return FrameEntry(
        trial=trial,
        camera=camera,
        label_id=label_id,
        name=name,
        event_type="point" if boundary == "point" else "state",
        boundary=boundary,
        t_rel=t_rel,
        onset_s=t_rel,
        offset_s=offset_s,
        individual=individual,
        individual_rec="",
        confidence=confidence,
        labeling_method=labeling_method,
    )


class TestSeedsFromEntries:
    def test_cameras_of_one_boundary_collapse_to_one_seed(self):
        entries = [_entry(camera="cam-1"), _entry(camera="cam-2")]
        seeds = seeds_from_entries(entries)
        assert len(seeds) == 1
        assert seeds[0]["field"] == "point"
        assert seeds[0]["labels"] == 1 and str(seeds[0]["trial"]) == "1"

    def test_a_state_tile_is_two_seeds(self):
        entries = [_entry(boundary="state", t_rel=1.0, offset_s=2.0)]
        assert [s["field"] for s in seeds_from_entries(entries)] == ["start", "end"]

    def test_two_cameras_of_a_state_tile_still_two_seeds(self):
        entries = [
            _entry(boundary="state", t_rel=1.0, offset_s=2.0, camera="cam-1"),
            _entry(boundary="state", t_rel=1.0, offset_s=2.0, camera="cam-2"),
        ]
        assert [s["field"] for s in seeds_from_entries(entries)] == ["start", "end"]


class TestPackTiles:
    """A state tile is two columns wide and never straddles a row break."""

    def test_points_fill_rows_left_to_right(self):
        placed = pack_tiles([_entry(t_rel=t) for t in (0.1, 0.2, 0.3)], columns=2)
        assert [(r, c, span) for _, r, c, span in placed] == [(0, 0, 1), (0, 1, 1), (1, 0, 1)]

    def test_a_state_tile_spans_two_columns(self):
        state = _entry(boundary="state", t_rel=1.0, offset_s=2.0)
        placed = pack_tiles([state, _entry(t_rel=3.0)], columns=3)
        assert [(r, c, span) for _, r, c, span in placed] == [(0, 0, 2), (0, 2, 1)]

    def test_a_state_tile_wraps_rather_than_splitting(self):
        state = _entry(boundary="state", t_rel=1.0, offset_s=2.0)
        placed = pack_tiles([_entry(t_rel=0.5), _entry(t_rel=0.6), state], columns=3)
        assert [(r, c, span) for _, r, c, span in placed] == [(0, 0, 1), (0, 1, 1), (1, 0, 2)]

    def test_one_column_shrinks_a_state_tile_to_fit(self):
        state = _entry(boundary="state", t_rel=1.0, offset_s=2.0)
        placed = pack_tiles([state, state], columns=1)
        assert [(r, c, span) for _, r, c, span in placed] == [(0, 0, 1), (1, 0, 1)]

    def test_different_trials_stay_separate(self):
        entries = [_entry(trial="1"), _entry(trial="2")]
        assert len(seeds_from_entries(entries)) == 2

    def test_seed_carries_the_subject_pair(self):
        seed = seeds_from_entries([_entry()])[0]
        assert seed["individual"] == "a" and seed["individual_rec"] == ""


class _PanelStub:
    """Stands in for the Labels tab's curation panel."""

    def __init__(self, mode="manual"):
        self._mode = mode
        self.curated: list[dict] = []
        self.reviews: list[tuple[dict, str]] = []
        self.session_active = False
        self.restarted = 0

    def mode(self):
        return self._mode

    def reviews_on_jump(self):
        return self._mode in ("segment", "frame")

    def curate_labels(self, insts):
        self.curated.extend(insts)
        return len(insts)

    def start_review_at(self, inst, field):
        self.reviews.append((inst, field))
        return True

    def restart_review(self):
        self.restarted += 1


class _NavStub2:
    def __init__(self):
        self.jumps = []

    def jump_to_label_instance(self, inst, **kwargs):
        self.jumps.append(inst)


class TestGridVerdicts:
    @pytest.fixture()
    def grid(self, qapp, tmp_path, labels_df):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        labels = _LabelsStub(MAPPINGS)
        labels.curation_panel = _PanelStub()
        entries = [
            _entry(trial="1", confidence=0.2, labeling_method=LABELING_AUTOMATED),
            _entry(trial="2", confidence=0.9, labeling_method=LABELING_AUTOMATED),
            _entry(trial="3", confidence=1.0, labeling_method=LABELING_MANUAL),
        ]
        meta = _Meta(state, labels, nav=_NavStub2())
        dlg = LabelGridView(meta, entries)
        dlg._meta = meta
        yield dlg
        dlg.close()

    def test_double_click_jumps(self, grid):
        grid._on_tile_double_clicked(grid._entries[0])
        assert [i["trial"] for i in grid._meta.navigation_widget.jumps] == ["1"]

    def test_double_click_leaves_the_verdicts_alone(self, grid):
        """Qt opens a double click with a plain press, which toggles the tile;
        the double click toggles it back, so a jump tags nothing."""
        grid._on_tile_clicked(grid._entries[0])  # the press Qt delivers first
        grid._on_tile_double_clicked(grid._entries[0])
        assert not grid.verdict_bar.verdicts.clicked
        assert grid.verdict_bar.count_label.text() == ""

    def test_double_click_on_a_tagged_tile_keeps_it_tagged(self, grid):
        grid._on_tile_clicked(grid._entries[0])  # tagged
        grid._on_tile_clicked(grid._entries[0])  # the press: unmarks
        grid._on_tile_double_clicked(grid._entries[0])  # the double click: back on
        assert grid.verdict_bar.verdicts.is_clicked(grid._entries[0])

    def test_double_click_drops_into_the_frame_review_when_that_curation_mode_is_on(self, grid):
        grid._meta.labels_widget.curation_panel._mode = "frame"
        grid._on_tile_double_clicked(grid._entries[1])
        panel = grid._meta.labels_widget.curation_panel
        assert panel.reviews and panel.reviews[0][0]["trial"] == "2" and panel.reviews[0][1] == "point"
        assert grid._meta.navigation_widget.jumps == []

    def test_done_curates_every_untagged_automated_label(self, grid):
        grid._on_tile_clicked(grid._entries[0])
        assert grid.verdict_bar.count_label.text() == "1 tagged"
        grid._on_tile_clicked(grid._entries[0])  # a second click untags
        grid._on_tile_clicked(grid._entries[1])
        grid.verdict_bar.apply_done()
        panel = grid._meta.labels_widget.curation_panel
        assert [i["trial"] for i in panel.curated] == ["1"]  # not the tagged one, not the manual one
        assert grid._entries[0].labeling_method == LABELING_CURATED
        assert grid._entries[1].labeling_method == LABELING_AUTOMATED  # tagged: left for review
        assert not grid.verdict_bar.verdicts.clicked

    def test_done_restarts_an_active_frame_review(self, grid):
        """Done may curate a label the frame-by-frame session is reviewing —
        its queue must be rebuilt, not left stale."""
        panel = grid._meta.labels_widget.curation_panel
        panel.session_active = True
        grid._on_tile_clicked(grid._entries[1])
        grid.verdict_bar.apply_done()
        assert panel.restarted == 1

    def test_done_leaves_an_inactive_review_alone(self, grid):
        panel = grid._meta.labels_widget.curation_panel
        assert not panel.session_active
        grid._on_tile_clicked(grid._entries[1])
        grid.verdict_bar.apply_done()
        assert panel.restarted == 0

    def test_tag_flagged_tags_what_the_threshold_outlines(self, grid):
        grid.threshold_edit.setValue(0.5)
        grid.verdict_bar._tag_flagged()
        assert [grid.verdict_bar.verdicts.is_clicked(e) for e in grid._entries] == [True, False, False]


class TestLabelFilterChoices:
    """The filter's choices: all of them first, then one per class with its count."""

    def _entries(self):
        return [
            _entry(label_id=2, name="hop", trial="1"),
            _entry(label_id=1, name="peck", trial="2"),
            _entry(label_id=1, name="peck", trial="3"),
        ]

    def test_all_comes_first_and_counts_every_tile(self):
        assert label_filter_choices(self._entries())[0] == (None, "All labels (3)")

    def test_one_choice_per_class_by_name_with_its_count(self):
        assert label_filter_choices(self._entries())[1:] == [(2, "hop (1)"), (1, "peck (2)")]

    def test_no_entries_offers_only_all(self):
        assert label_filter_choices([]) == [(None, "All labels (0)")]

    def test_filter_none_is_every_entry(self):
        entries = self._entries()
        assert filter_entries(entries, None) == entries

    def test_filter_keeps_one_class(self):
        assert [e.trial for e in filter_entries(self._entries(), 1)] == ["2", "3"]


class TestLabelFilterGrid:
    """The filter narrows the operations, not just the view."""

    @pytest.fixture()
    def grid(self, qapp, tmp_path, labels_df):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        labels = _LabelsStub(MAPPINGS)
        labels.curation_panel = _PanelStub()
        entries = [
            _entry(label_id=1, name="peck", trial="1", confidence=0.2, labeling_method=LABELING_AUTOMATED),
            _entry(label_id=2, name="hop", trial="2", confidence=0.2, labeling_method=LABELING_AUTOMATED),
            _entry(label_id=2, name="hop", trial="3", confidence=0.9, labeling_method=LABELING_AUTOMATED),
        ]
        dlg = LabelGridView(_Meta(state, labels, nav=_NavStub2()), entries)
        yield dlg
        dlg.close()

    def _select(self, grid, label_id):
        grid.label_filter.setCurrentIndex(grid.label_filter.findData(label_id))

    def test_the_filter_shows_up_only_with_more_than_one_class(self, grid):
        assert grid._filter_row.isVisibleTo(grid)
        one_class = LabelGridView(grid.meta, [_entry(label_id=1, name="peck", trial="1")])
        assert not one_class._filter_row.isVisibleTo(one_class)
        one_class.close()

    def test_opens_unfiltered(self, grid):
        assert grid.label_filter.currentData() is None
        assert len(grid.visible_entries()) == 3
        assert grid.count_label.text() == "3 labels"

    def test_filtering_hides_the_other_classes_tiles(self, grid):
        self._select(grid, 2)
        assert [c.isVisibleTo(grid) for c in grid._cells] == [False, True, True]
        assert grid.count_label.text() == "2 of 3 labels"

    def test_done_leaves_the_hidden_classes_untouched(self, grid):
        """The dangerous one: 'the rest is curated' must mean the rest *on screen*."""
        self._select(grid, 2)
        grid._on_tile_clicked(grid._entries[1])
        grid.verdict_bar.apply_done()
        assert [i["trial"] for i in grid.meta.labels_widget.curation_panel.curated] == ["3"]
        assert grid._entries[0].labeling_method == LABELING_AUTOMATED

    def test_tag_flagged_only_reaches_the_shown_tiles(self, grid):
        self._select(grid, 2)
        grid.threshold_edit.setValue(0.5)
        grid.verdict_bar._tag_flagged()
        assert [grid.verdict_bar.verdicts.is_clicked(e) for e in grid._entries] == [False, True, False]

    def test_the_tag_count_follows_the_filter(self, grid):
        grid._on_tile_clicked(grid._entries[0])
        grid._on_tile_clicked(grid._entries[1])
        assert grid.verdict_bar.count_label.text() == "2 tagged"
        self._select(grid, 2)
        assert grid.verdict_bar.count_label.text() == "1 tagged"
        self._select(grid, None)
        assert grid.verdict_bar.count_label.text() == "2 tagged"

    def test_the_hint_names_the_filtered_class(self, grid):
        assert "Filtered to" not in grid.hint.text()
        self._select(grid, 2)
        assert "Filtered to 'hop'" in grid.hint.text()


class TestTileVerdicts:
    def test_cameras_of_one_label_share_a_verdict(self):
        a = _entry(trial="1", camera="c1", labeling_method=LABELING_AUTOMATED)
        b = _entry(trial="1", camera="c2", labeling_method=LABELING_AUTOMATED)
        verdicts = TileVerdicts()
        assert [i["trial"] for i in verdicts.insts_for_done([a, b])] == ["1"]  # untagged: once
        assert verdicts.toggle(a) is True
        assert verdicts.is_clicked(b)
        assert verdicts.insts_for_done([a, b]) == []  # tagged through either camera

    def test_manual_labels_are_never_part_of_a_verdict(self):
        manual = _entry(trial="1", labeling_method=LABELING_MANUAL)
        verdicts = TileVerdicts()
        assert verdicts.insts_for_done([manual]) == []
        verdicts.toggle(manual)
        assert verdicts.insts_for_done([manual]) == []


class TestFlaggedTrials:
    """One flagged event pulls its whole trial into the review."""

    def _entries(self):
        return [
            _entry(trial="20", t_rel=0.5, confidence=0.95),
            _entry(trial="20", t_rel=1.5, confidence=0.20),
            _entry(trial="21", t_rel=0.5, confidence=0.90),
        ]

    def test_trials_named_by_their_worst_event(self):
        assert flagged_trials(self._entries(), 0.6) == {"20"}

    def test_threshold_off_flags_nothing(self):
        assert flagged_trials(self._entries(), 0.0) == set()

    def test_flagged_tiles_are_outlined_and_tag_flagged_follows_the_threshold(self, qapp, tmp_path, labels_df):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        grid = LabelGridView(_Meta(state, _LabelsStub(MAPPINGS)), self._entries())
        grid.threshold_edit.setValue(0.6)
        assert [bool(c.styleSheet()) for c in grid._cells] == [False, True, False]

        grid.verdict_bar._tag_flagged()
        assert [grid.verdict_bar.verdicts.is_clicked(e) for e in grid._entries] == [False, True, False]
        grid.close()


class TestConfidenceGroups:
    """Confidences are grouped per label class, and per animal when there is
    more than one — one histogram each."""

    def test_one_group_per_label_class(self):
        entries = [
            _entry(label_id=1, trial="1", confidence=0.2),
            _entry(label_id=1, trial="2", confidence=0.8),
            _entry(label_id=2, trial="1", confidence=0.5),
        ]
        groups = confidence_groups(entries)
        assert [g.label_id for g in groups] == [1, 2]
        assert sorted(groups[0].values) == [0.2, 0.8]
        assert all(g.individual is None for g in groups)

    def test_cameras_count_the_event_once(self):
        entries = [
            _entry(camera="cam-1", boundary="state", t_rel=1.0, offset_s=2.0, confidence=0.4),
            _entry(camera="cam-2", boundary="state", t_rel=1.0, offset_s=2.0, confidence=0.4),
        ]
        assert confidence_groups(entries)[0].values == [0.4]

    def test_multi_animal_splits_per_individual(self):
        entries = [
            _entry(trial="1", individual="alice", confidence=0.3),
            _entry(trial="2", individual="bob", confidence=0.9),
            _entry(trial="3", individual="bob", confidence=0.7),
        ]
        groups = confidence_groups(entries)
        assert [(g.label_id, g.individual) for g in groups] == [(1, "alice"), (1, "bob")]
        assert groups[0].values == [0.3]
        assert sorted(groups[1].values) == [0.7, 0.9]

    def test_no_entries_no_groups(self):
        assert confidence_groups([]) == []


class TestSplitHistogram:
    def test_the_two_halves_add_up_to_every_value(self):
        values = [0.1, 0.35, 0.55, 0.9, 1.0]
        _, below, above = split_histogram(values, 0.5, bins=10)
        assert below.sum() + above.sum() == len(values)
        assert below.sum() == 2

    def test_a_straddled_bin_splits_instead_of_picking_a_side(self):
        # One bin (0.4-0.5 at 10 bins) holds a flagged and an unflagged value.
        edges, below, above = split_histogram([0.42, 0.48], 0.45, bins=10)
        idx = int(np.searchsorted(edges, 0.45)) - 1
        assert below[idx] == 1 and above[idx] == 1

    def test_threshold_off_flags_nothing(self):
        _, below, above = split_histogram([0.0, 0.5], 0.0, bins=10)
        assert below.sum() == 0 and above.sum() == 2


class TestStickyThreshold:
    """Curation runs over many trials — the flag threshold is remembered (SCOPE_GLOBAL),
    shared with the video grid's own threshold spin."""

    def test_opens_at_the_saved_value_and_writes_back(self, qapp, tmp_path, labels_df):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        state.grid_confidence_threshold = 0.3
        entries = [_entry(trial="1", confidence=0.2)]
        grid = LabelGridView(_Meta(state, _LabelsStub(MAPPINGS)), entries)
        try:
            assert grid.threshold_edit.value() == pytest.approx(0.3)
            grid.threshold_edit.setValue(0.6)
            assert state.grid_confidence_threshold == pytest.approx(0.6)
        finally:
            grid.close()


class TestHistogramDialog:
    """The popup and the grid share one threshold, in both directions."""

    @pytest.fixture()
    def grid(self, qapp, tmp_path, labels_df):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        entries = [
            _entry(trial="1", confidence=0.2),
            _entry(trial="2", confidence=0.9),
        ]
        dlg = LabelGridView(_Meta(state, _LabelsStub(MAPPINGS)), entries)
        yield dlg
        dlg.close()

    def test_one_plot_per_group(self, grid):
        grid._show_histograms()
        assert len(grid._hist_dialog._plots) == 1
        grid._hist_dialog.close()

    def test_the_popup_opens_on_the_grids_threshold(self, grid):
        grid.threshold_edit.setValue(0.5)
        grid._show_histograms()
        assert grid._hist_dialog.threshold_edit.value() == pytest.approx(0.5)
        grid._hist_dialog.close()

    def test_moving_it_in_the_popup_moves_the_grid(self, grid):
        grid._show_histograms()
        grid._hist_dialog.threshold_edit.setValue(0.4)
        assert grid.threshold_edit.value() == pytest.approx(0.4)
        assert LOW_CONFIDENCE_COLOR in grid._cells[0].styleSheet()
        grid._hist_dialog.close()

    def test_moving_it_in_the_grid_moves_the_popup(self, grid):
        grid._show_histograms()
        grid.threshold_edit.setValue(0.7)
        assert grid._hist_dialog.threshold_edit.value() == pytest.approx(0.7)
        grid._hist_dialog.close()

    def test_closing_it_lets_the_next_click_reopen(self, grid):
        grid._show_histograms()
        dialog = grid._hist_dialog
        dialog.close()
        QApplication.processEvents()
        assert grid._hist_dialog is None
        grid._show_histograms()
        assert grid._hist_dialog is not None and grid._hist_dialog is not dialog
        grid._hist_dialog.close()


class TestHistogramBarColor:
    """A reddish label class must not swallow its own flagged bars."""

    def test_a_distinct_colour_is_kept(self):
        assert histogram_bar_color("#55aaff") == "#55aaff"
        assert histogram_bar_color("#ffcc44") == "#ffcc44"

    def test_a_reddish_colour_falls_back_to_neutral(self):
        assert histogram_bar_color("#ff5555") != "#ff5555"
        assert histogram_bar_color(LOW_CONFIDENCE_COLOR) != LOW_CONFIDENCE_COLOR


class _SortStub:
    """The fields sort_entries reads, off both grids' entry types."""

    def __init__(self, trial, onset_s, confidence, duration=0.0):
        self.trial = trial
        self.onset_s = onset_s
        self.confidence = confidence
        self.duration = duration


class TestSortEntries:
    """Ordering is a review strategy: by confidence, the doubtful labels land
    on the first screens instead of scattered through the trials."""

    ENTRIES = [
        _SortStub("2", 1.0, 0.9, duration=5.0),
        _SortStub("10", 0.0, 0.1, duration=1.0),
        _SortStub("1", 3.0, 0.5, duration=9.0),
    ]

    def _trials(self, order):
        return [e.trial for e in gv.sort_entries(self.ENTRIES, order)]

    def test_trial_order_is_numeric_not_textual(self):
        """Trial 10 comes after trial 2, not between 1 and 2."""
        assert self._trials("trial") == ["1", "2", "10"]

    def test_confidence_ascending_puts_the_doubtful_first(self):
        assert self._trials("confidence_asc") == ["10", "1", "2"]

    def test_confidence_descending_reverses_it(self):
        assert self._trials("confidence_desc") == ["2", "1", "10"]

    def test_duration_is_the_video_grids_own_order(self):
        assert self._trials("duration") == ["10", "2", "1"]

    def test_ties_fall_back_to_trial_and_time(self):
        tied = [_SortStub("2", 0.0, 0.5), _SortStub("1", 9.0, 0.5), _SortStub("1", 0.0, 0.5)]
        assert [(e.trial, e.onset_s) for e in gv.sort_entries(tied, "confidence_asc")] == [
            ("1", 0.0),
            ("1", 9.0),
            ("2", 0.0),
        ]

    def test_sorting_never_drops_or_adds_an_entry(self):
        for order in gv.VIDEO_GRID_SORT_ORDERS:
            assert len(gv.sort_entries(self.ENTRIES, order)) == len(self.ENTRIES)

    def test_an_unknown_order_is_refused(self):
        with pytest.raises(ValueError, match="Unknown grid sort order"):
            gv.sort_entries(self.ENTRIES, "sideways")

    def test_duration_is_offered_only_by_the_video_grid(self):
        """The frame grid shows one frame per label — length means nothing there."""
        assert "duration" not in gv.GRID_SORT_ORDERS
        assert "duration" in gv.VIDEO_GRID_SORT_ORDERS
