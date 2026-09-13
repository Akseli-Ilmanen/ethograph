# Windows EthoGraph 0.2.12 Test Overview

Hi,

Here is a concise Windows-focused follow-up from testing `ethograph 0.2.12`
before our meeting. Overall, the new version is a major improvement for my use
case, especially compared with the earlier version I tested.

## Short Summary

- Regular/main playback now produces audible audio in `Audio-synced` mode on
  Windows.
- The new multi-panel layout is useful and mostly works well. I could use video
  plus multiple audio traces/spectrograms with label overlays visible.
- Fresh annotation from an empty-label session works: label categories were
  still available, I created pseudo labels, saved, closed, reopened, and the
  annotations persisted.
- The main remaining blocker is intermittent video freezing while audio and the
  timeline/playhead continue.
- The most reproducible smaller playback issue is very short Play -> Stop
  bursts: if playback runs for less than roughly `0.4 s`, audio can play while
  the video/current-frame position does not commit. The next Play then starts
  again from where the previous short burst began.

## Environment

- OS: Windows 11
- Python: `3.12.13`
- EthoGraph: `0.2.12`
- Relevant playback stack: `pygfx 0.17.0`, `pynaviz 0.2.0`,
  `wgpu 0.32.0`, `rendercanvas 2.7.2`, `sounddevice 0.5.5`
- Tested through a local debug launcher that logs video/audio/playback state.

## Diagnostic Media

The main Windows tests used short local diagnostic sessions, not the full
original dataset:

- Video proxy: H.264 MP4, about `1280 x 936`, `47.683716 fps`, no embedded
  audio.
- Trials: two short trial windows, about `120 s` each.
- Audio variants:
  - mono WAV at `8 kHz` and `16 kHz` PCM16,
  - original mic WAV around `24.414 kHz`,
  - 3-channel WAV around `24.414 kHz`,
  - 6-channel WAV in the empty-label annotation smoke test.
- Labels: imported diagnostic label set has 51 rows for the main experimental
  trial, with point and state events split across three branches.

## What Works Now

- `Audio-synced` regular playback is audible on Windows.
- The 8 kHz/3-channel proxy workflows are much more usable than the older
  EthoGraph version.
- Label overlays and playheads render correctly across multiple stacked panels.
- Channel selection works after clicking an audio trace/spectrogram panel and
  using the right-side channel dropdown.
- Fresh annotation creation and save/reopen persistence passed.
- Active/editable label branches work once I understood that clicking the branch
  title, not the checkbox, changes the editable branch.

## Remaining Issues

### 1. Visual Video Freeze While Audio/Timeline Continue

I reproduced a freeze where the video image stops updating, but audio and the
timeline/playhead continue. Debugging strongly suggests this is not a full GUI
hang and not a simple decoder stall.

Evidence from the strongest freeze:

- Session: 3-channel proxy session with multiple panels.
- `VideoSync._advance_from_clock`, `CameraView.seek_video_frame`, and
  `PlotVideo.buffer_frame` continued.
- `audio_clock=True` continued.
- Worker/buffer threads were alive.
- `needs_redraw=True` and `pending_ui_q=1` were present.
- `PlotVideo.animate()` had not run for about `1000 s`.

This points to a stale pygfx/render-canvas presentation loop or stale video
view, after playback/decode state has continued to advance.

Important recovery clue: closing the frozen video panel and adding a new video
panel restored the video without restarting the whole session.

### 2. Panel Lifecycle Cleanup

After closing all panels and reopening only video plus spectrogram, the GUI got
into a partially broken state with:

```text
RuntimeError: wrapped C/C++ object of type QRenderCanvas has been deleted
```

The traceback path was:

```text
widgets_data.on_trial_changed()
video_manager.update_video()
video_manager._cleanup_primary_video()
primary_view.clear()
self.layout().removeWidget(self._plot.canvas)
```

This looks like the video manager can retain a reference to a deleted
Qt/rendercanvas object after manual panel closure/recreation.

### 3. Very Short Playback Bursts

For Play -> Stop intervals shorter than about `0.4 s`:

- the audio snippet can play,
- the video frame/current-frame position does not seem to commit,
- the next Play starts from where the short segment began.

This feels like audio output can start before the video/current-frame state has
advanced enough to survive Stop.

### 4. Multichannel Channel Switching

In a 3-channel session:

- switching channels while playback is running updates the displayed audio
  trace/spectrogram,
- but the audible audio channel only reloads after Stop and the next Play.

Also, during panel recreation, the spectrogram loader tried to open filenames
like:

```text
...mic1.wav (Ch 3)
```

as literal files. It should probably split this into base audio file plus
channel index before opening the audio file.

### 5. Embedded MP4/AAC Audio

Embedded MP4/AAC still should be treated as unsupported or at least unreliable
for my workflow. Separate WAV audio is the safe path.

### 6. Lower-Priority Annotation/UI Notes

- Branch visibility checkbox vs active editable branch is easy to confuse. The
  checkbox controls display, while clicking the branch title makes it editable.
- On this Windows/PyQt6 setup, branch checkbox state handling can fail if
  `stateChanged` emits integer `2` and the code compares directly to
  `Qt.Checked`. Normalizing with `Qt.CheckState(qt_state) == Qt.Checked`
  restored hide/re-show behavior locally.
- While drawing a state label, there is no obvious temporary start marker or
  preview interval after the first click. A shaded preview interval or persistent
  start anchor would make state annotation easier.

## Optional Files

The full report with more detail is:

```text
docs/ethograph_annotation_feasibility_report.md
```

Most relevant freeze log:

```text
.home/ethograph_video_debug_freeze_20260818_152425.log
```

This log is useful because it captured the video freeze while playback/audio
bookkeeping continued and the render heartbeat had stopped.
