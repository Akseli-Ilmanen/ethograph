"""The video grid: clips grouped per label class, sorted by duration, played together."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QWidget

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_label_gridview import entry_inst, entry_key, methods_for_filter
from ethograph.gui.dialog_video_grid import (
    ClipEntry,
    VideoGridDialog,
    VideoGridPlayer,
    build_clip_entries,
    clip_note,
    frame_index,
    group_clips,
    marker_visible,
    page_duration,
    page_fps,
    paginate,
)
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
            "onset_s": [0.2, 1.0, 2.0, 3.0],
            "offset_s": [np.nan, 1.8, 2.3, np.nan],
            "individual": ["a", "a", "b", "a"],
            "individual_rec": ["", "", "", ""],
            "confidence": [0.3, 1.0, 0.9, 0.8],
            "labeling_method": [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_AUTOMATED, LABELING_AUTOMATED],
        }
    )


class TestMethodFilter:
    """Reviewing a model's output means its labels and nothing else.

    Both grids share this filter (it lives on ``LabelSetupPage``), so it is
    tested where the entries are built: a mixed set of labels must come out
    one side of the human/model divide at a time.
    """

    def test_automated_only_drops_the_human_labels(self, labels_df):
        methods = methods_for_filter("automated")
        entries = build_clip_entries(labels_df, MAPPINGS, [1, 2], [None], 0.5, None, methods)
        assert {e.labeling_method for e in entries} == {LABELING_AUTOMATED}
        assert len(entries) == 3

    def test_human_keeps_manual_and_curated_together(self, labels_df):
        """Which of the two a label is says only how it got there."""
        df = labels_df.copy()
        df.loc[0, "labeling_method"] = LABELING_CURATED
        entries = build_clip_entries(df, MAPPINGS, [1, 2], [None], 0.5, None, methods_for_filter("human"))
        assert {e.labeling_method for e in entries} == {LABELING_MANUAL, LABELING_CURATED}

    def test_all_is_the_unfiltered_grid(self, labels_df):
        methods = methods_for_filter("all")
        assert methods is None
        assert len(build_clip_entries(labels_df, MAPPINGS, [1, 2], [None], 0.5, None, methods)) == 4


class TestBuildClipEntries:
    def test_state_clips_span_the_label_and_point_clips_a_window(self, labels_df):
        entries = build_clip_entries(labels_df, MAPPINGS, [1, 2], [None], point_window_s=0.5)
        by = {(e.trial, e.label_id): e for e in entries}
        assert (by[(1, 2)].t0, by[(1, 2)].t1) == (1.0, 1.8)
        assert (by[(3, 1)].t0, by[(3, 1)].t1) == (2.5, 3.5)
        assert by[(3, 1)].point_t == pytest.approx(0.5)
        assert by[(1, 2)].point_t is None

    def test_a_point_window_never_starts_before_the_trial(self, labels_df):
        entries = build_clip_entries(labels_df, MAPPINGS, [1], [None], point_window_s=0.5)
        first = next(e for e in entries if e.trial == 1)
        assert first.t0 == 0.0 and first.t1 == pytest.approx(0.7)
        assert first.point_t == pytest.approx(0.2)

    def test_cameras_multiply_and_filters_narrow(self, labels_df):
        assert len(build_clip_entries(labels_df, MAPPINGS, [1, 2], ["c1", "c2"], 0.5)) == 8
        assert {e.trial for e in build_clip_entries(labels_df, MAPPINGS, [1, 2], [None], 0.5, {"2"})} == {2}

    def test_method_and_identity_are_carried(self, labels_df):
        entries = build_clip_entries(labels_df, MAPPINGS, [2], [None], 0.5)
        assert [e.labeling_method for e in entries] == [LABELING_MANUAL, LABELING_AUTOMATED]
        assert entry_key(entries[1]) == ("2", 2, 2.0, "b", "")
        assert entry_inst(entries[1])["onset_s"] == 2.0


def _clip(label_id, trial, t0, t1, n_frames=None, fps=10.0, event_type="state", onset=None):
    entry = ClipEntry(
        trial=trial,
        camera=None,
        label_id=label_id,
        name=f"L{label_id}",
        event_type=event_type,
        onset_s=t0 if onset is None else onset,
        offset_s=t1,
        t0=t0,
        t1=t1,
        labeling_method=LABELING_AUTOMATED,
    )
    if n_frames is not None:
        entry.frames = np.zeros((n_frames, 8, 12, 3), dtype=np.uint8)
        entry.fps = fps
    return entry


class TestGrouping:
    def test_one_group_per_label_sorted_by_duration(self):
        entries = [_clip(2, "1", 0, 3.0), _clip(1, "1", 0, 1.0), _clip(2, "2", 0, 0.5), _clip(1, "3", 0, 0.2)]
        groups = group_clips(entries)
        assert [[e.label_id for e in g] for g in groups] == [[1, 1], [2, 2]]
        assert [e.duration for e in groups[1]] == [0.5, 3.0]
        assert [e.trial for e in groups[0]] == ["3", "1"]

    def test_paginate(self):
        group = [_clip(1, str(i), 0, 1.0) for i in range(5)]
        pages = paginate(group, 2)
        assert [len(p) for p in pages] == [2, 2, 1]
        assert paginate([], 3) == [[]]

    def test_page_duration_and_fps(self):
        page = [_clip(1, "1", 0, 1.0, n_frames=10, fps=10.0), _clip(1, "2", 0, 2.5, n_frames=75, fps=30.0)]
        assert page_duration(page) == 2.5
        assert page_fps(page) == 30.0
        assert page_fps([_clip(1, "1", 0, 1.0)]) is None


class TestFrames:
    def test_frame_index_holds_the_last_frame_after_the_clip_ends(self):
        entry = _clip(1, "1", 0, 1.0, n_frames=11, fps=10.0)
        assert frame_index(entry, 0.0) == 0
        assert frame_index(entry, 0.5) == 5
        assert frame_index(entry, 4.0) == 10
        assert frame_index(_clip(1, "1", 0, 1.0), 0.5) is None

    def test_marker_is_on_for_the_frame_the_point_falls_on(self):
        entry = _clip(1, "1", 0.5, 1.5, n_frames=11, fps=10.0, event_type="point", onset=1.0)
        assert entry.point_t == pytest.approx(0.5)
        assert marker_visible(entry, 0.5)
        assert marker_visible(entry, 0.54)
        assert not marker_visible(entry, 0.7)
        assert not marker_visible(_clip(1, "1", 0, 1.0, n_frames=5), 0.5)  # state event: no marker


class TestClipNote:
    def test_a_window_inside_the_video_has_no_note(self):
        assert clip_note(10, 40, 100) is None

    def test_cut_at_either_end_is_named(self):
        assert clip_note(-5, 20, 100) == "cut 5 frames at video start"
        assert clip_note(90, 110, 100) == "cut 10 frames at video end"
        assert clip_note(-5, 110, 100) == "cut 5 frames at video start; cut 10 frames at video end"

    def test_a_window_past_the_end_or_an_empty_video(self):
        assert clip_note(120, 140, 100) == "window lies past the video end"
        assert clip_note(0, 10, 0) == "video reports no frames"


class _PanelStub:
    def __init__(self):
        self.curated: list[dict] = []
        self._mode = "manual"
        self.session_active = False
        self.restarted = 0

    def mode(self):
        return self._mode

    def reviews_on_jump(self):
        return self._mode in ("segment", "frame")

    def curate_labels(self, insts):
        self.curated.extend(insts)
        return len(insts)

    def restart_review(self):
        self.restarted += 1


class _LabelsStub(QWidget):
    def __init__(self):
        super().__init__()
        self._mappings = MAPPINGS
        self.curation_panel = _PanelStub()


class _NavStub:
    def __init__(self):
        self.jumps = []

    def jump_to_label_instance(self, inst, **kwargs):
        self.jumps.append((inst, kwargs))


class _Meta:
    def __init__(self, state):
        self.app_state = state
        self.labels_widget = _LabelsStub()
        self.navigation_widget = _NavStub()
        self.data_widget = None


@pytest.fixture
def player(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    entries = [
        _clip(1, "1", 0, 1.0, n_frames=11, fps=10.0, event_type="point", onset=0.5),
        _clip(1, "2", 0, 0.5, n_frames=6, fps=10.0, event_type="point", onset=0.2),
        _clip(2, "1", 0, 2.0, n_frames=21, fps=10.0),
    ]
    meta = _Meta(state)
    widget = VideoGridPlayer(meta, entries, columns=2, per_page=2, decode_fn=lambda page: None)
    widget.resize(800, 600)
    widget._meta = meta
    yield widget
    widget.stop()
    widget.close()


class TestPlayer:
    def test_first_screen_is_the_first_label_class_shortest_first(self, player):
        assert [t.entry.trial for t in player._tiles] == ["2", "1"]
        assert "label 1 of 2" in player.header.text() and "2 clips" in player.header.text()
        assert player.slider.maximum() == 1000  # the longest clip on screen

    def test_labels_walk_with_the_buttons_and_grey_out_at_the_ends(self, player):
        assert not player.prev_label_btn.isEnabled() and player.next_label_btn.isEnabled()
        player._step_label(+1)
        assert [t.entry.label_id for t in player._tiles] == [2]
        assert player.slider.maximum() == 2000
        assert player.prev_label_btn.isEnabled() and not player.next_label_btn.isEnabled()
        player._step_label(+1)  # nowhere to go — stays
        assert [t.entry.label_id for t in player._tiles] == [2]

    def test_a_single_label_class_greys_out_the_label_buttons(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        entries = [_clip(1, str(i), 0, 1.0, n_frames=11, fps=10.0) for i in range(3)]
        player = VideoGridPlayer(_Meta(state), entries, columns=2, per_page=2, decode_fn=lambda page: None)
        try:
            assert not player.prev_label_btn.isEnabled() and not player.next_label_btn.isEnabled()
            assert "label 1 of" not in player.header.text()
            assert "clips 1–2 of 3" in player.header.text()
            # Three clips, two on screen: only "Next clips" leads anywhere.
            assert not player.prev_clips_btn.isEnabled() and player.next_clips_btn.isEnabled()
            player._step_clips(+1)
            assert len(player._tiles) == 1
            assert player.prev_clips_btn.isEnabled() and not player.next_clips_btn.isEnabled()
        finally:
            player.close()

    def test_seek_renders_and_play_toggles(self, player):
        player.seek(0.5)
        assert player.time == pytest.approx(0.5)
        assert player.time_label.text().startswith("0.50")
        player.toggle_play()
        assert player.playing
        player.toggle_play()
        assert not player.playing

    def test_playback_stops_at_the_page_end_and_play_rewinds(self, player):
        player.play()
        player.seek(0.95)
        player._tick()
        assert player.time == pytest.approx(1.0)  # clamped to the longest clip
        assert not player.playing  # stopped, not looped
        player.play()
        assert player.time == 0.0 and player.playing
        player.stop()

    def test_arrow_keys_step_one_frame_and_pause(self, player):
        """←/→ move every tile one frame of the page's fastest clip, clamped
        to the page, and stop playback first."""
        player.play()
        player.step_frame(+1)
        assert not player.playing
        assert player.time == pytest.approx(0.1)  # 10 fps → one frame
        assert player.slider.value() == 100
        player.step_frame(-1)
        player.step_frame(-1)
        assert player.time == 0.0  # never before the start
        player.seek(0.95)
        player.step_frame(+1)
        assert player.time == pytest.approx(1.0)  # clamped to the longest clip
        # The keys reach the player itself (keyPressEvent) and, through a
        # shortcut scoped to it, any child that has focus.
        player.seek(0.0)
        QTest.keyClick(player, Qt.Key_Right)
        QTest.keyClick(player, Qt.Key_Right)
        assert player.time == pytest.approx(0.2)
        QTest.keyClick(player, Qt.Key_Left)
        assert player.time == pytest.approx(0.1)

    def test_speed_stretches_the_tick_and_stays_in_the_grid(self, player):
        assert player.speed_spin.value() == 100
        assert player._timer.interval() == 100  # 10 fps, real time
        player.speed_spin.setValue(25)
        assert player.app_state.playback_speed_pct == 100.0  # the GUI's own speed is untouched
        assert player.app_state.video_grid_speed_pct == 25.0  # but the grid's own sticky setting follows
        assert player._timer.interval() == 400  # one frame per tick, four times as long
        player.play()
        player._tick()
        assert player.time == pytest.approx(0.1)  # still exactly one frame
        player.stop()
        player.speed_spin.setValue(400)
        assert player._timer.interval() == 25
        player._tick()
        assert player.time == pytest.approx(0.2)  # wall-clock × 4

    def test_speed_opens_at_the_saved_grid_speed_not_the_gui_speed(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state.playback_speed_pct = 50.0
        state.video_grid_speed_pct = 75.0
        entries = [_clip(1, "1", 0, 1.0, n_frames=11, fps=10.0)]
        player = VideoGridPlayer(_Meta(state), entries, columns=1, per_page=1, decode_fn=lambda page: None)
        try:
            assert player.speed_spin.value() == 75
            assert player._timer.interval() == 133  # round(1000 / (10 fps * 0.75))
        finally:
            player.close()

    def test_caption_says_where_the_label_sits(self, player):
        captions = [t.caption.text() for t in player._tiles]
        assert "at 0.20 s" in captions[0]  # point event, trial 2
        player._step_label(+1)
        assert "0.00–2.00 s" in player._tiles[0].caption.text()  # state event

    def test_double_click_navigates(self, player):
        tile = player._tiles[0]
        tile.double_clicked.emit(tile.entry)
        jumps = player._meta.navigation_widget.jumps
        assert len(jumps) == 1 and jumps[0][0]["trial"] == "2"

    def test_double_click_leaves_the_verdicts_alone(self, player):
        """The press Qt delivers before a double click toggles the tile; the
        double click toggles it back, so a jump tags nothing."""
        tile = player._tiles[0]
        tile.clicked.emit(tile.entry)  # the press Qt delivers first
        tile.double_clicked.emit(tile.entry)
        assert not player.verdict_bar.verdicts.clicked
        assert player._meta.navigation_widget.jumps

    def test_a_click_tags_and_done_curates_everything_else(self, player):
        tile = player._tiles[0]
        tile.clicked.emit(tile.entry)
        assert player.verdict_bar.verdicts.is_clicked(tile.entry)
        player.verdict_bar.apply_done()
        panel = player._meta.labels_widget.curation_panel
        # Every automated clip of every group except the tagged one.
        assert sorted((i["trial"], i["labels"]) for i in panel.curated) == [("1", 1), ("1", 2)]
        assert tile.entry.labeling_method == LABELING_AUTOMATED
        assert not player.verdict_bar.verdicts.clicked


class TestStickyGridSettings:
    """Curation runs over many trials — the grid's own knobs are remembered (SCOPE_GLOBAL)."""

    def test_threshold_opens_at_the_saved_value_and_writes_back(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state.grid_confidence_threshold = 0.3
        entries = [_clip(1, "1", 0, 1.0, n_frames=11, fps=10.0)]
        player = VideoGridPlayer(_Meta(state), entries, columns=1, per_page=1, decode_fn=lambda page: None)
        try:
            assert player.threshold_edit.value() == pytest.approx(0.3)
            player.threshold_edit.setValue(0.6)
            assert state.grid_confidence_threshold == pytest.approx(0.6)
        finally:
            player.close()


class TestHistogramDialog:
    """The popup and the player share one threshold, in both directions — same as the frame grid's."""

    def test_one_plot_per_group(self, player):
        player._show_histograms()
        assert len(player._hist_dialog._plots) == 2  # label 1 and label 2
        player._hist_dialog.close()

    def test_the_popup_opens_on_the_players_threshold(self, player):
        player.threshold_edit.setValue(0.5)
        player._show_histograms()
        assert player._hist_dialog.threshold_edit.value() == pytest.approx(0.5)
        player._hist_dialog.close()

    def test_moving_it_in_the_popup_moves_the_player(self, player):
        player._show_histograms()
        player._hist_dialog.threshold_edit.setValue(0.4)
        assert player.threshold_edit.value() == pytest.approx(0.4)
        player._hist_dialog.close()

    def test_moving_it_in_the_player_moves_the_popup(self, player):
        player._show_histograms()
        player.threshold_edit.setValue(0.7)
        assert player._hist_dialog.threshold_edit.value() == pytest.approx(0.7)
        player._hist_dialog.close()

    def test_closing_it_lets_the_next_click_reopen(self, player):
        player._show_histograms()
        dialog = player._hist_dialog
        dialog.close()
        QApplication.processEvents()
        assert player._hist_dialog is None
        player._show_histograms()
        assert player._hist_dialog is not None and player._hist_dialog is not dialog
        player._hist_dialog.close()


class TestPrefetch:
    def test_next_page_is_next_clips_then_next_label(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        entries = [
            _clip(1, "1", 0, 1.0, n_frames=11, fps=10.0),
            _clip(1, "2", 0, 0.5, n_frames=6, fps=10.0),
            _clip(1, "3", 0, 2.0, n_frames=21, fps=10.0),
            _clip(2, "1", 0, 2.0, n_frames=21, fps=10.0),
        ]
        ahead: list[list[ClipEntry]] = []
        player = VideoGridPlayer(
            _Meta(state), entries, columns=2, per_page=2, decode_fn=lambda page: None, prefetch_fn=ahead.append
        )
        try:
            # Label 1 has two pages; showing its first asks for its second.
            assert [e.trial for e in player.next_page()] == ["3"]
            assert [[e.trial for e in p] for p in ahead] == [["3"]]
            player._step_clips(+1)  # last page of label 1 → label 2's first page
            assert [e.label_id for e in player.next_page()] == [2]
            player._step_label(+1)  # nothing beyond the last label
            assert player.next_page() == []
            assert len(ahead) == 2
        finally:
            player.close()

    def test_a_jump_onto_the_prefetched_page_waits_instead_of_decoding_twice(
        self, qapp, tmp_path, labels_df, monkeypatch
    ):
        import threading
        import time

        from ethograph.gui import dialog_video_grid as mod

        opened: list[str] = []
        release = threading.Event()

        class _SlowSource:
            def __init__(self, path, fps, nframes, max_side=None):
                opened.append(path)
                self.n = nframes

            def __enter__(self):
                release.wait(5.0)
                return self

            def __exit__(self, *args):
                return False

            def __getitem__(self, s):
                return np.zeros((len(range(*s.indices(self.n))), 4, 4, 3), np.uint8)

        monkeypatch.setattr(mod, "VideoFrameSource", _SlowSource)
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        dialog = VideoGridDialog(_Meta(state), label_ids=[1])
        page = [_clip(1, "1", 0, 1.0), _clip(1, "3", 0, 1.0)]  # nothing decoded yet
        # Plan on the GUI thread (stubbed — no alignment here), decode off-thread.
        monkeypatch.setattr(mod, "plan_clip_jobs", lambda entries, **kw: [(list(entries), "vid.mp4", 10.0, 0.0, 11)])
        try:
            dialog._prefetch_page(page)
            assert dialog._prefetch is not None and not dialog._prefetch.future.done()
            # Let the worker through just after the jump starts waiting.
            threading.Timer(0.1, release.set).start()
            t0 = time.monotonic()
            dialog._decode_page(page)
            assert time.monotonic() - t0 >= 0.05  # it waited for the worker
            assert opened == ["vid.mp4"]  # one decode, not two
            assert all(e.frames is not None and len(e.frames) == 11 for e in page)
            assert dialog._prefetch is None
        finally:
            release.set()
            dialog.close()


class TestDialog:
    def test_setup_lists_the_scope_read_only(self, qapp, tmp_path, labels_df):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        dialog = VideoGridDialog(_Meta(state), label_ids=[2])
        try:
            listed = [
                dialog.label_list.item(i).data(0x0100)  # Qt.UserRole
                for i in range(dialog.label_list.count())
            ]
            assert listed == [2]  # the scope, read-only
            assert not dialog.tabs.isTabEnabled(1)
        finally:
            dialog.close()

    def test_layout_choices_are_global_and_remembered(self, qapp, tmp_path, labels_df):
        """Point window (0.5 s default), clips on screen and columns live in
        gui_settings.yaml — a viewing habit that follows the user across datasets."""
        from ethograph.gui.app_state import AppStateSpec

        for key in ("video_grid_point_window_s", "video_grid_per_page", "video_grid_columns"):
            assert AppStateSpec.get_meta(key)[3] == AppStateSpec.SCOPE_GLOBAL
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        dialog = VideoGridDialog(_Meta(state), label_ids=[2])
        try:
            assert dialog.point_window_spin.value() == 0.5
            assert (dialog.per_page_spin.value(), dialog.columns_spin.value()) == (6, 3)
            dialog.point_window_spin.setValue(1.25)
            dialog.per_page_spin.setValue(9)
            dialog.columns_spin.setValue(4)
        finally:
            dialog.close()
        assert state.video_grid_point_window_s == 1.25
        assert (state.video_grid_per_page, state.video_grid_columns) == (9, 4)
        again = VideoGridDialog(_Meta(state), label_ids=[2])
        try:
            assert again.point_window_spin.value() == 1.25
            assert (again.per_page_spin.value(), again.columns_spin.value()) == (9, 4)
        finally:
            again.close()

    def test_the_method_filter_opens_where_it_was_left(self, qapp, tmp_path, labels_df):
        """Which half of the labels one is reviewing outlives a dialog, and a
        dataset — global like the rest of the grid setup."""
        from ethograph.gui.app_state import AppStateSpec

        assert AppStateSpec.get_meta("grid_method_filter")[3] == AppStateSpec.SCOPE_GLOBAL
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = labels_df
        dialog = VideoGridDialog(_Meta(state), label_ids=[2])
        try:
            assert dialog.setup.selected_methods() is None  # opens on "All labels"
            dialog.setup.method_combo.setCurrentIndex(dialog.setup.method_combo.findData("automated"))
            assert dialog.setup.selected_methods() == frozenset({LABELING_AUTOMATED})
        finally:
            dialog.close()
        assert state.grid_method_filter == "automated"
        again = VideoGridDialog(_Meta(state), label_ids=[2])
        try:
            assert again.setup.method_combo.currentData() == "automated"
        finally:
            again.close()
