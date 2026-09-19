"""The label-name overlay drawn on the video is a video display setting, so
its "Hide label" checkbox lives in the sidebar's video context."""

from qtpy.QtWidgets import QApplication

from ethograph.gui.right_context import _CONTEXT_MAP


def test_hide_label_sits_in_the_video_context(moll2025_gui):
    _, meta = moll2025_gui
    cb = meta.labels_widget.hide_label_cb
    gb = meta.data_widget.videolabel_groupbox

    assert cb.parent() is gb
    assert "videolabel" in _CONTEXT_MAP["video"]
    assert cb not in meta.data_widget.overlays_groupbox.findChildren(type(cb))

    meta.context_panel.set_context("video")
    QApplication.processEvents()
    assert gb.isVisibleTo(meta.context_panel)
    assert cb.isVisibleTo(meta.context_panel)


def test_the_overlay_combo_sets_every_plot_type_and_reads_mixed_back(moll2025_gui):
    """The Labels tab's top row carries both one-line choices; the combo is the
    per-plot-type dialog's shorthand, so the two must agree in both directions."""
    from ethograph.gui.app_constants import (
        LABEL_OVERLAY_MODE_BOTTOM,
        LABEL_OVERLAY_MODE_FULL,
        LABEL_OVERLAY_MODE_NONE,
        LABEL_OVERLAY_PLOT_TYPES,
    )
    from ethograph.gui.widgets_labels import _OVERLAY_PER_PLOT

    _, meta = moll2025_gui
    lw = meta.labels_widget
    combo = lw.label_overlay_combo

    index = combo.findData(LABEL_OVERLAY_MODE_BOTTOM)
    combo.setCurrentIndex(index)
    combo.activated.emit(index)
    assert set(meta.app_state.label_overlay_modes) == set(LABEL_OVERLAY_PLOT_TYPES)
    assert set(meta.app_state.label_overlay_modes.values()) == {LABEL_OVERLAY_MODE_BOTTOM}

    # Modes the combo cannot say in one word read back as "Per plot type…".
    mixed = dict.fromkeys(LABEL_OVERLAY_PLOT_TYPES, LABEL_OVERLAY_MODE_FULL)
    mixed["lineplot"] = LABEL_OVERLAY_MODE_NONE
    meta.app_state.label_overlay_modes = mixed
    lw._sync_overlay_combo()
    assert combo.currentData() == _OVERLAY_PER_PLOT


def test_the_labels_tab_gates_on_somebody_and_something_to_label(moll2025_gui):
    """Nothing to label ⇒ the body greys out and the gate says which half is missing.

    Each button is offered only when its own half is missing, so what is on
    screen is the next step and never a dead end.
    """
    _, meta = moll2025_gui
    lw = meta.labels_widget

    assert lw.can_label()
    assert lw._body.isEnabled() and not lw._gate.isVisibleTo(lw)

    only_background = {0: {"name": "background", "branch": 0, "event_type": "state"}}
    lw._mappings = only_background
    lw.refresh_gate()
    assert not lw.has_labels() and not lw.can_label()
    # Greyed out, not hidden: the tables stay on screen, the gate says what is missing.
    assert lw._gate.isVisibleTo(lw) and not lw._body.isEnabled()
    assert lw._gate_labels_btn.isVisibleTo(lw._gate)
    # The dataset names its individuals, so that half is already answered.
    assert not lw._gate_individuals_btn.isVisibleTo(lw._gate)


def test_the_shortcut_keys_refuse_to_place_a_label_while_gated(moll2025_gui):
    """Greying out the panel leaves the keys live, so the refusal is in the handler.

    Otherwise a label would land on the synthesised ``default`` individual and
    only be noticed once it was already in the TSV.
    """
    _, meta = moll2025_gui
    lw = meta.labels_widget
    key = next(iter(lw.KEY_TO_labels))

    lw._mappings = {}  # nothing to label
    lw.refresh_gate()
    before = lw.current_labels
    lw.activate_label(key)
    assert lw.current_labels == before, "a gated key press must not arm or place a label"
