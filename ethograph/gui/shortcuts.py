"""Global keyboard shortcut bindings for the ethograph GUI.

Shortcuts are QShortcuts on the main window (napari keymaps are gone).
Plain-letter and arrow-key shortcuts are guarded: while the user types in a
text field or spin box the shell **disables** them (``_sync_guarded_shortcuts``
on ``focusChanged``) rather than letting them fire a no-op. They are
application-context shortcuts, so an enabled one swallows the key press before
the focus widget receives it — a no-op guard left arrow keys dead in every
input, including the ↑/↓ selection walk in the add-panel popup's filter box.

A binding on a key the focused text editor owns (``Ctrl+V``, ``Ctrl+A``,
word-wise cursor moves, …) is guarded *automatically*, whatever the call site
asks for.
"""

import logging

from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit

logger = logging.getLogger(__name__)

_TEXT_WIDGETS = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)

#: Keys a focused text editor owns. A global binding on one of these is always
#: guarded: unguarded, ``Ctrl+C`` would swallow the copy out of a metadata cell
#: *and* run the action behind it (curate the trial), which flags the labels
#: unsaved after a metadata-only edit. Same for ``Ctrl+A``/``Ctrl+V``
#: (select-all, paste) and the cursor keys.
_TEXT_EDITING_KEYS = frozenset(
    QKeySequence(key).toString()
    for key in (
        "Ctrl+A",
        "Ctrl+C",
        "Ctrl+V",
        "Ctrl+X",
        "Ctrl+Z",
        "Ctrl+Y",
        "Ctrl+Left",
        "Ctrl+Right",
        "Shift+Left",
        "Shift+Right",
        "Ctrl+Home",
        "Ctrl+End",
        "Home",
        "End",
        "Delete",
        "Backspace",
    )
)


def typing_in_text_field() -> bool:
    """True when keystrokes belong to a text entry rather than a shortcut."""
    widget = QApplication.focusWidget()
    if widget is None:
        return False
    if isinstance(widget, _TEXT_WIDGETS):
        return True
    if isinstance(widget, QComboBox) and widget.isEditable():
        return True
    return False


def bind_global_shortcuts(meta_widget):
    shell = meta_widget.shell
    app_state = meta_widget.app_state
    labels_widget = meta_widget.labels_widget
    data_widget = meta_widget.data_widget
    navigation_widget = meta_widget.navigation_widget
    plot_settings_widget = meta_widget.plot_settings_widget
    changepoints_widget = meta_widget.changepoints_widget
    io_widget = meta_widget.io_widget
    plot_container = meta_widget.plot_container

    shell.clear_shortcuts()

    def bind(key, callback, guarded=False):
        """Bind *key*; guarded shortcuts are disabled while typing in a field."""
        text_key = QKeySequence(key).toString() in _TEXT_EDITING_KEYS
        shell.bind_shortcut(key, callback, guarded=guarded or text_key)

    # --- Playback / navigation ---
    # Same handler as the Save labels button and menu entry: it carries the
    # remote-backup folder from the I/O panel and reports failures in a dialog.
    bind("Ctrl+S", io_widget._save_labels)

    def toggle_zen_mode():
        """Zen mode: hide the right sidebar for a distraction-free view."""
        shell.set_zen_mode(not shell._zen_mode)

    bind("Shift+Z", toggle_zen_mode, guarded=True)

    def toggle_pause_resume():
        data_widget.toggle_pause_resume()
        bottom_bar = getattr(shell, "bottom_bar", None)
        if bottom_bar is not None:
            bottom_bar._sync_play_icon()

    bind("Space", toggle_pause_resume, guarded=True)
    bind("V", labels_widget._play_segment, guarded=True)
    bind("Shift+Left", navigation_widget.step_window_backward)
    bind("Shift+Right", navigation_widget.step_window_forward)
    bind("Left", navigation_widget.step_frame_backward, guarded=True)
    bind("Right", navigation_widget.step_frame_forward, guarded=True)
    bind("Down", navigation_widget.next_trial, guarded=True)
    bind("Up", navigation_widget.prev_trial, guarded=True)
    bind("Ctrl+Down", lambda: meta_widget._cycle_channel(+1))
    bind("Ctrl+Up", lambda: meta_widget._cycle_channel(-1))
    bind("Shift+N", meta_widget.show_source_popup, guarded=True)

    def toggle_autoscale():
        autoscale_status = plot_settings_widget.autoscale_checkbox.isChecked()
        plot_settings_widget.autoscale_checkbox.setChecked(not autoscale_status)

    bind("Ctrl+A", toggle_autoscale)

    def toggle_lock():
        lock_status = plot_settings_widget.lock_axes_checkbox.isChecked()
        plot_settings_widget.lock_axes_checkbox.setChecked(not lock_status)

    bind("Ctrl+L", toggle_lock)
    # Curate the current trial: every automated label in scope becomes curated
    # (manual ones stay manual). Auto-guarded — Ctrl+C is copy in a text field.
    bind("Ctrl+C", meta_widget.curate_current_trial)
    # Flag the current trial as hard (or back to normal) in the metadata
    # table's difficulty column — a trial the model should see more often.
    bind("Ctrl+T", labels_widget.curation_panel.toggle_difficulty)

    def toggle_changepoint_correction():
        checkbox = changepoints_widget.changepoint_correction_checkbox
        checkbox.setChecked(not checkbox.isChecked())

    bind("Ctrl+B", toggle_changepoint_correction)
    bind("Ctrl+R", meta_widget.reset_panels)

    def _change_spacing(delta: float):
        pc = plot_container
        if pc and pc.is_ephystrace():
            buf = pc.ephys_trace_plot.buffer
            buf.channel_spacing = min(max(buf.channel_spacing + delta, 0.5), 20.0)
            xmin, xmax = pc.get_current_xlim()
            pc.ephys_trace_plot.update_plot_content(xmin, xmax)

    bind("Ctrl+=", lambda: _change_spacing(+0.5))
    bind("Ctrl+-", lambda: _change_spacing(-0.5))

    def _jump_spike(delta: int):
        if not plot_container or not plot_container.is_ephystrace():
            return
        plot_container.ephys_trace_plot.jump_to_spike(delta)

    bind("Alt+Right", lambda: _jump_spike(+1))
    bind("Alt+Left", lambda: _jump_spike(-1))

    def stop_recording():
        top_bar = getattr(shell, "_top_bar", None)
        recorder = getattr(top_bar, "_record_controller", None)
        if recorder is not None:
            recorder._stop_recording()

    bind("Ctrl+Space", stop_recording)

    # --- Label activation grid layout ---
    number_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    qwerty_row = ["Q", "W", "E", "R", "T", "Z", "U", "I", "O", "P"]
    home_row = ["A", "S", "D", "F", "G", "H", "J", "K", "L", "Y"]
    fn_row = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]

    for i, key in enumerate(number_keys):
        labels = i + 1 if key != "0" else 10
        bind(key, lambda mk=labels: labels_widget.activate_label(mk), guarded=True)

    for i, key in enumerate(qwerty_row):
        bind(key, lambda mk=i + 11: labels_widget.activate_label(mk), guarded=True)

    for i, key in enumerate(home_row):
        bind(key, lambda mk=i + 21: labels_widget.activate_label(mk), guarded=True)

    for i, key in enumerate(fn_row):
        bind(key, lambda mk=i + 31: labels_widget.activate_label(mk), guarded=True)

    bind("Ctrl+E", labels_widget._edit_label)
    bind("Ctrl+D", labels_widget._delete_label)
    # Auto-guarded (Ctrl+Z is a text-editing key), so it undoes labels only
    # when the user is not typing.
    bind("Ctrl+Z", labels_widget.undo_last_label_edit)
    bind("Shift+B", labels_widget.toggle_branch, guarded=True)
    bind("Ctrl+I", lambda: app_state.toggle_key_sel("individual", data_widget))
    bind("Ctrl+K", lambda: app_state.toggle_key_sel("keypoint", data_widget))

    def cycle_cameras():
        combo = getattr(data_widget, "primary_camera_combo", None)
        if combo is not None and combo.count() > 1:
            next_index = (combo.currentIndex() + 1) % combo.count()
            combo.setCurrentIndex(next_index)

    bind("Shift+C", cycle_cameras, guarded=True)
    bind("Shift+M", lambda: app_state.toggle_key_sel("mics", data_widget), guarded=True)
    bind("Ctrl+H", data_widget.cycle_neural_view)

    bind("Ctrl+Right", lambda: changepoints_widget.jump_changepoint(+1))
    bind("Ctrl+Left", lambda: changepoints_widget.jump_changepoint(-1))
