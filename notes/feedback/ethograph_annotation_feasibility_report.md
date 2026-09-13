# Windows EthoGraph Annotation Feasibility Report

Last updated: 2026-08-18

## Most Important Findings

- The Windows test environment was upgraded from `ethograph 0.2.4.4` to
  `ethograph 0.2.12`, the current PyPI release reported by `pip index`.
- `0.2.12` contains relevant upstream changes for the issues found on
  `0.2.4.4`: regular playback now has an audio-clock path, playback modes are
  explicit, playback speed is one shared percent control, and singleton
  time-slice selection no longer drops the time dimension.
- Automated smoke tests passed for the local `.nc` diagnostic sessions:
  TrialTree files open, normal feature selection works, and singleton time
  slices return shape `(1,)` instead of raising the old "No dimension containing
  'time'" error.
- WAV audio loads correctly through both `audioio` and EthoGraph's
  `SharedAudioCache` for the tested 24.414 kHz float WAV, 16 kHz PCM16 WAV, and
  8 kHz PCM16 WAV variants.
- The new `AudioClock` can open a Windows `sounddevice` output stream and
  resamples playback to a fixed 48 kHz output rate. This directly targets the
  previous regular-playback audio/synchronization limitation.
- Embedded MP4/AAC audio still does not load as an EthoGraph audio source.
  `0.2.12` now surfaces a clearer message saying embedded video-container audio
  is not decoded in place and that separate WAV/FLAC/OGG should be supplied.
  Direct `audioio.load_audio()` probing of the MP4 still produced a Python
  shutdown crash after the decode failure, so this path should be avoided.
- Initial GUI retest confirms that the updated launcher works, the diagnostic
  session loads, and regular/main playback now produces audible audio in
  `Audio-synced` mode.
- A new remaining blocker was observed in `Audio-synced` mode: after moving
  around while listening, the video display froze on a frame/time around
  `62.7258 s`, while audio continued and the timeline playhead continued moving.
  This suggests audio/playhead can now advance independently of a stalled video
  renderer or async frame-seek queue.
- Switching to `Smooth (every frame)` did not unfreeze the video, and switching
  back to `Audio-synced` also did not recover it. The freeze therefore appears
  to be a persistent video-view/renderer/seek-queue state, not only a transient
  audio-synced mode issue.
- A screen recording of a later freeze captured the video overlay timestamp
  stuck at `61.2788 s` for `12.567 s`, while the audio trace/spectrogram and
  timeline regions continued updating. The matching debug log shows the
  playback controller and frame-buffer path still advancing, which points more
  toward a stale pygfx/render-canvas presentation problem than a whole-GUI hang.
- A separate branch-overlay bug was found on Windows/PyQt6: after hiding and
  re-showing label branches, timeline annotations can remain hidden even though
  the branch checkboxes appear checked. Code inspection points to
  `stateChanged` delivering integer `2`, while the handler compares directly to
  `Qt.Checked`; in this environment `2 == Qt.Checked` is false.
- The new multi-panel layout worked well in a Windows GUI retest with four
  total timeline/audio panels visible. Labels and playheads appeared correctly
  across the stacked tracks, and playback remained workable. Starting/stopping
  had a small delay but was much improved compared with the old version.
- Mouse-wheel panning instead of zooming appears explainable by normal GUI
  state: `Lock Axes` or `Nav -> Fixed window` keeps the x-span fixed, so wheel
  interactions slide the timeline rather than changing zoom.
- A later 3-channel multi-panel freeze produced stronger diagnostics: decoded
  frame buffering and `VideoSync._advance_from_clock` continued with
  `audio_clock=True`, but `PlotVideo.animate()` had not run for about 1000 s.
  This points to the pygfx/render-canvas animate loop stopping while playback,
  audio, and frame-buffer bookkeeping continue.
- Recreating only the video panel can recover a frozen video view, which further
  suggests the broken state is local to the video canvas/render view. However,
  closing/reopening panels can also leave the video manager holding a deleted
  `QRenderCanvas` reference, so panel lifecycle cleanup still needs attention.
- In the 3-channel WAV session, changing channels while playback is running
  updates the visible audio trace/spectrogram, but the audible audio channel
  only changes after Stop and the next Play.
- Very short Play -> Stop bursts remain problematic: if playback runs for less
  than roughly `0.4 s`, the audio snippet can play while the video/current-frame
  position does not commit, and the next Play restarts from the previous burst
  onset.
- Spectrogram panel creation can treat channel-qualified display names such as
  `mic1.wav (Ch 3)` as literal filenames. The loader should split these into
  base audio file plus channel index before opening the audio file.
- Fresh annotation from an empty labels file preserves the label
  categories/template, but drawing state events has a usability issue: after
  the first click, there is no obvious temporary start marker or preview
  interval while choosing the end time. A transient shaded interval/anchor would
  make state annotation easier to control.
- Fresh annotation save/reopen passed on Windows: pseudo labels created in the
  empty-label session were still present after saving, closing, and reopening
  the `.nc` session.
- Remaining GUI characterization should focus on video-frame freshness during
  audio-clock playback, zooming, repeated stop/start, latency, and comparison
  with `Smooth (every frame)` and `Real-time (skip frames)`.

## Current Windows Environment

Local bundle:

```text
C:\Users\yapur\Documents\ETH\PhD\windows_usb_handoff_20260717\ethograph_windows_handoff
```

Installed versions after update:

```text
Python: 3.12.13
ethograph: 0.2.12
napari: 0.7.1
napari-pyav: 0.1.1
audioio: 2.8.1
sounddevice: 0.5.5
soundfile: 0.14.0
pygfx: 0.17.0
pynaviz: 0.2.0
wgpu: 0.32.0
rendercanvas: 2.7.2
```

Dependency check:

```text
pip check: No broken requirements found.
```

`requirements_windows.txt` is now pinned to:

```text
ethograph[gui,audio]==0.2.12
napari==0.7.1
napari-pyav==0.1.1
audioio==2.8.1
sounddevice==0.5.5
```

## Media And Diagnostic Data

The Windows bundle uses two approximately 2-minute trial windows from local
source videos. The private video content is not required to reproduce the main
technical issues; synthetic media with the same dimensions, duration, frame
rate, audio format, and trial structure should be enough.

Source MP4 media:

| source file | size | video codec | pixel format | dimensions | fps | duration | embedded audio |
|:--|--:|:--|:--|:--|--:|--:|:--|
| `CopExpBP01_CTRL_video01.mp4` | 163.7 MB | H.264 | yuv420p | `2756 x 2016` | 47.683716 | 360.025 s | AAC stereo, 24 kHz |
| `CopExpBP01_Exp_video01.mp4` | 173.2 MB | H.264 | yuv420p | `2756 x 2016` | 47.683716 | 360.025 s | AAC stereo, 24 kHz |

Trial-window metadata:

| window type | start | stop | duration | fps | reason |
|:--|--:|--:|--:|--:|:--|
| copulation-relevant window | 239.243 s | 359.997 s | 120.754 s | 47.683716 | 1 minute before cloacal contact through video end |
| control window | 84.496 s | 204.496 s | 120.000 s | 47.683716 | deterministic random 2-minute control window |

Diagnostic session sizes:

| session | trials | video files | audio files | video total | audio total | `.nc` size | `.nwb` alignment size | audio rate | audio channels |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `original_noaudio` | 2 | 2 | 0 | 126.2 MB | 0 B | 17.0 KB | 177.0 KB | | |
| `proxy1280_noaudio` | 2 | 2 | 0 | 15.8 MB | 0 B | 17.0 KB | 177.0 KB | | |
| `proxy1280_audio` | 2 | 2 | 2 | 15.8 MB | 134.5 MB | 19.4 KB | 0 B observed in bundle | 24414 Hz | 6 |
| `proxy1280_audio3ch` | 2 | 2 | 2 | 15.8 MB | 67.3 MB | 19.4 KB | 45.0 MB | 24414 Hz | 3 |
| `proxy1280_mic1` | 2 | 2 | 2 | 15.8 MB | 22.4 MB | 19.4 KB | 45.0 MB | 24414 Hz | 1 |
| `proxy1280_mic1_pcm16_16k` | 2 | 2 | 2 | 15.8 MB | 7.3 MB | 19.3 KB | 29.6 MB | 16000 Hz | 1 |
| `proxy1280_mic1_pcm16_8k` | 2 | 2 | 2 | 15.8 MB | 3.7 MB | 19.3 KB | 14.9 MB | 8000 Hz | 1 |
| `proxy1280_mpaudio` | 2 | 2 | 0 separate | 19.5 MB | embedded | 19.4 KB | 44.3 MB | 24000 Hz | 2 |

The zero-byte `proxy1280_audio` alignment file appears to be a copied diagnostic
bundle artifact and was not used for the final conclusion. The most useful
retest sessions are `proxy1280_noaudio`, `proxy1280_mic1`,
`proxy1280_mic1_pcm16_16k`, `proxy1280_mic1_pcm16_8k`, and
`proxy1280_mpaudio`.

Representative external audio:

| session | representative audio | size | codec | sample rate | channels | duration |
|:--|:--|--:|:--|--:|--:|--:|
| `proxy1280_mic1` | H5-derived mic1 WAV | 11.2 MB | PCM float32 | 24414 Hz | 1 | 120.000 s |
| `proxy1280_mic1_pcm16_16k` | mic1 downsampled WAV | 3.7 MB | PCM int16 | 16000 Hz | 1 | 120.000 s |
| `proxy1280_mic1_pcm16_8k` | mic1 downsampled WAV | 1.8 MB | PCM int16 | 8000 Hz | 1 | 120.000 s |

## Results From `0.2.4.4`

These results are retained as the baseline for the `0.2.12` retest.

- `original_noaudio`: loaded after applying a local singleton-time-dimension
  patch. No sound expected. Labels visible. Video mostly ran at 30 fps but had
  minor lags; source-rate playback around 47.684 fps was choppy.
- `proxy1280_noaudio`: smooth at source rate when Skip Frames was disabled. With
  Skip Frames enabled, playback became choppy even on proxy video.
- `proxy1280_mic1`: no audio heard during main stream playback. Playback was
  choppy at 30 fps and source rate, with or without Skip Frames.
- `proxy1280_mic1_pcm16_16k`: initially better at 30 fps with Skip Frames
  disabled, but became overloaded/timed out after zooming or repeated
  start/stop. State-label playback produced audible audio, but with noticeable
  delay. Main stream playback remained video-only/no-audio.
- `proxy1280_mic1_pcm16_8k`: worked somewhat at 30 fps initially, but zooming or
  repeated stop/start again caused choppiness, delay, or timeouts.
- Integrated EthoGraph Downsample did not materially change playback behaviour,
  consistent with it downsampling TrialTree/data arrays rather than external
  media streams.
- `proxy1280_mpaudio`: no audio trace/spectrogram and no audible audio.
  Non-GUI `audioio.load_audio()` also failed on MP4/AAC.

Baseline conclusion before update: Windows native audio was a partial success
because interval/state-label audio could play, but the full audio-synchronized
annotation workflow was not reliable.

## Automated `0.2.12` Retest

Completed checks:

- `pip index versions ethograph` reported latest `ethograph 0.2.12`.
- Upgrade succeeded with `ethograph[gui,audio]==0.2.12`.
- `pip check` reported no broken requirements.
- Diagnostic TrialTree `.nc` files open under `0.2.12`.
- Normal feature selection and singleton time-slice selection work for all key
  diagnostic sessions tested:
  - `original_noaudio`
  - `proxy1280_noaudio`
  - `proxy1280_mic1`
  - `proxy1280_mic1_pcm16_16k`
  - `proxy1280_mic1_pcm16_8k`
  - `proxy1280_mpaudio`
- Singleton time-slice smoke test returns shape `(1,)`; the original
  zero-width/singleton time-axis GUI loading issue is likely fixed upstream.
- Literal scalar `time=value` selection still raises `ValueError: No dimension
  containing 'time' found in the DataArray`, but the GUI path appears to use
  slices rather than scalar selection.
- WAV audio smoke tests:
  - 24.414 kHz float WAV: `audioio.load_audio` OK, `SharedAudioCache` OK.
  - 16 kHz PCM16 WAV: `audioio.load_audio` OK, `SharedAudioCache` OK.
  - 8 kHz PCM16 WAV: `audioio.load_audio` OK, `SharedAudioCache` OK.
- `AudioClock` smoke test:
  - Can prepare a fixed 48 kHz output buffer.
  - Can open and stop a Windows `sounddevice.OutputStream` using the default
    output device.
- MP4/AAC smoke test:
  - EthoGraph's `SharedAudioCache` now returns a clearer message: embedded
    video-container audio is not decoded in place; re-drop video with audio
    extraction enabled or supply separate WAV/FLAC/OGG.
  - Direct `audioio.load_audio()` on the MP4 still fails and produced a Python
    application error at interpreter shutdown during the smoke test.

## Relevant Upstream Changes In `0.2.12`

The upgrade directly targets several previous findings:

- Playback controls changed from separate FPS/audio-speed and Skip Frames
  controls to:
  - `Audio-synced`: audio and video locked; may drop video frames.
  - `Smooth (every frame)`: every video frame; may run slower; no audio.
  - `Real-time (skip frames)`: approximate speed by dropping frames; no audio.
- Playback speed is now one percentage control:
  - `100%` means native recording speed.
  - Video frame rate and audio pitch/rate are derived from the same speed.
- Regular playback now attempts to build an audio-master clock when audio is
  available. Video and marker position are driven from the audio clock's elapsed
  time, not from an independent video timer.
- The audio clock resamples spans to a fixed 48 kHz output rate, avoiding the old
  device-rate limitation.
- The singleton-time-dimension helper now squeezes only non-time singleton
  dimensions, preserving length-1 time axes.
- There is a new GUI `Proxy` control that can generate and reuse low-resolution
  video proxies next to the source videos.

## GUI Retest Protocol For `0.2.12`

Focus on the old failure modes rather than retesting everything.

### 1. Preferred Audio Session

Load:

```text
outputs\playback_diagnostics\proxy1280_mic1_pcm16_8k\annotation_pilot_imported_proxy1280_mic1_pcm16_8k.nc
```

Expected setup:

- Playback mode: `Audio-synced`
- Speed: `100%`
- Audio trace visible
- Spectrogram visible
- Center playback off initially

Test:

1. Press regular/main Play.
2. Confirm whether audio starts during regular playback.
3. Watch whether video, red playhead, and audio stay synchronized.
4. Stop/start playback repeatedly.
5. Zoom in/out while audio trace and spectrogram are visible.
6. Repeat with spectrogram hidden, then with audio trace hidden.
7. Try `Smooth (every frame)` and confirm it is video-only but visually smoother
   or slower.
8. Try `Real-time (skip frames)` and confirm it is video-only and whether it
   remains choppy.

Record:

```text
regular Play audio audible:
audio delay noticeable:
video smoothness:
playhead smoothness:
zoom responsiveness:
repeated stop/start stable:
timeouts/application errors:
best mode:
```

### 2. Higher-Rate WAV Session

Repeat the same checks with:

```text
outputs\playback_diagnostics\proxy1280_mic1_pcm16_16k\annotation_pilot_imported_proxy1280_mic1_pcm16_16k.nc
```

This checks whether `0.2.12` makes 16 kHz usable after the new audio clock.

### 3. Original Mic1 WAV Session

Repeat a short version with:

```text
outputs\playback_diagnostics\proxy1280_mic1\annotation_pilot_imported_proxy1280_mic1.nc
```

This checks whether the original 24.414 kHz float WAV is now usable.

### 4. No-Audio Video Control

Load:

```text
outputs\playback_diagnostics\proxy1280_noaudio\annotation_pilot_imported_proxy1280_noaudio.nc
```

Compare `Smooth (every frame)` and `Real-time (skip frames)` for video-only
smoothness. This is the control for whether any remaining choppiness is audio
specific.

### 5. Embedded MP4/AAC Negative Control

Load:

```text
outputs\playback_diagnostics\proxy1280_mpaudio\annotation_pilot_imported_proxy1280_mpaudio.nc
```

Expected result: EthoGraph should not decode embedded AAC as an audio source,
but it should fail gracefully without crashing the GUI.

## Creator-Facing Reproduction Notes

A synthetic two-trial dataset should be enough to reproduce the Windows
behaviour without private data:

- Two trials, each about `120 s`.
- H.264/yuv420p MP4 video at `1280 x 936`, `47.683716 fps`, no embedded audio.
- External mono WAV at `8000 Hz`, `16000 Hz`, and optionally `24414 Hz`.
- Minimal TrialTree with one `timeline` variable over `time x individuals`.
- A small labels TSV with at least one state interval per trial.
- NWB/media alignment linking `video_cam-1` and `audio_mic-1` per trial.
- Optional MP4/AAC negative control: H.264 video plus AAC stereo at 24 kHz.

Useful synthetic content:

- Video: moving timestamp/grid/noise pattern.
- Audio: click train or chirp, because visible transients make delay obvious.
- Labels: arbitrary state intervals; behaviour semantics are not needed for
  playback debugging.

Pass criteria for this project:

- Selected audio plays during regular/main playback, label playback, and
  Auto-play on navigate.
- Audio, video frame, and playhead remain synchronized after zooming and after
  repeated stop/start.
- In `Audio-synced` mode, the displayed video frame must keep updating while
  the audio clock and timeline playhead advance. A stale video frame while audio
  continues should be treated as a playback failure.
- Timeline zoom remains responsive with audio trace/spectrogram visible.
- `1280 x 936` proxy video with mono WAV audio remains usable at native speed or
  at a clearly documented lower review speed.
- Embedded MP4/AAC either works reliably or is documented as unsupported for
  EthoGraph audio trace/spectrogram/playback.

Likely implementation area for the observed `0.2.12` freeze: in
`Audio-synced` regular playback, the audio clock can continue while video frame
updates are requested asynchronously. Once the freeze occurs, switching playback
modes does not recover the video view. The playback/video layer should detect
stale video frames or a backed-up seek queue and either force a synchronous
refresh, reset/recreate the video view safely, drop to a known-good proxy path,
or visibly warn that video has fallen behind audio.

Additional evidence from `freeze_1610_28072026.mp4`:

- The recording is `12.567 s`, `2560 x 1460`, `30 fps`, H.264, with no audio
  track in the recording itself.
- The visible video overlay timestamp stays fixed at `61.2788 s` throughout the
  recording, and the camera frame region is effectively unchanged.
- The audio trace, spectrogram, and timeline/control regions still repaint,
  including visible zoom/pan changes around 9-10 s into the recording.
- The debug log written at the same time shows `VideoSync._advance_from_clock`,
  `CameraView.seek_video_frame`, and frame-buffer callbacks continuing from
  roughly frame 660 through frame 1020 with live worker/buffer threads and no
  obvious request/response queue buildup.
- This specific run used an older debug wrapper that did not yet log
  `PlotVideo.animate()`, so the recording narrows the bug to after high-level
  playback/decode progress, but a new reproduction is still needed to separate
  an `animate()` heartbeat stop from stale texture/render presentation.

Additional freeze evidence from the 3-channel Windows retest:

- Session: `proxy1280_audio3ch`, with two spectrograms on different channels and
  a second audio trace added.
- The multi-panel display was otherwise workable, including branch overlay
  toggling during playback.
- Freeze snapshot: `.home/ethograph_video_debug_freeze_20260818_152425.log`;
  matching Python fault log was empty.
- The active debug log's last `PlotVideo.animate` heartbeat was at `15:05:17`.
  During playback around `15:21-15:22`, `VideoSync._advance_from_clock`,
  `CameraView.seek_video_frame`, and `PlotVideo.buffer_frame` continued with
  `audio_clock=True`, live worker/buffer threads, `pending_ui_q=1`,
  `needs_redraw=True`, and `animate_age_s` increasing to about `1008 s`.
- Interpretation: decoded frames and playback state keep advancing, but the
  render/canvas presentation loop no longer runs.
- Help -> Print current state after the freeze showed a coherent app state:
  `playback_mode=synced`, `has_audio=True`, `video_quality_mode=full`,
  `current_frame=2997`, `slider_scope=trial`, and `playback_speed_pct=100`.
  The active media were the 3-channel diagnostic trial
  `CopExpBP01_Exp_cloacal_window_w1280.mp4` plus
  `CopExpBP01_Exp_cloacal_window_h5_radio12_mic1.wav (Ch 1)`.
- The stored panel layout at freeze time contained five bottom panels:
  audiotrace, spectrogram Ch 3, spectrogram Ch 2, audiotrace Ch 3, and
  lineplot/timeline.
- The same dump listed 51 label rows for the current trial and NWB acquisition
  `audio_mic-1` with `5,877,770` dense timestamps.
- One warning before the state dump may be relevant:
  `FileNotFoundError` opening shared-memory mapping `wnsm_ef55d1e1`.
  The later `Process-5 ... KeyboardInterrupt` traceback was caused by pressing
  Ctrl+C in PowerShell while copying the terminal output and should be ignored
  as reproduction evidence.
- Recovery test: closing the frozen video panel and adding a new video panel
  via `+ Add panel` restored video display without restarting the session.
  This points to a stale video canvas/render view rather than a corrupted
  dataset, stopped audio clock, or unrecoverable GUI state. A practical fix
  could be stale-render detection plus an automatic video-view reset, or a
  visible manual "reset video view" command.
- Follow-up after closing all panels and reopening only video plus spectrogram:
  the GUI entered a partially broken state with repeated
  `RuntimeError: wrapped C/C++ object of type QRenderCanvas has been deleted`.
  The traceback path was `widgets_data.on_trial_changed()` ->
  `video_manager.update_video()` -> `video_manager._cleanup_primary_video()` ->
  `primary_view.clear()` -> `self.layout().removeWidget(self._plot.canvas)`.
  This suggests manual panel closure/recreation can leave
  `video_mgr.primary_view` pointing at a deleted Qt/rendercanvas object.
- In the same panel-recreation pass, spectrogram loading failed when
  channel-qualified display names were treated as literal filenames, for
  example `...mic1.wav (Ch 3)`. The intended model appears to be base WAV file
  plus selected channel index, not a separate per-channel file path.

Additional branch-overlay finding:

- On this Windows/PyQt6 environment, `2 == Qt.Checked` and
  `Qt.Checked == 2` both evaluate false, while `Qt.CheckState(2) == Qt.Checked`
  evaluates true.
- `LabelsWidget._on_branch_shown_changed()` currently stores
  `self.app_state._branch_shown[branch_idx] = qt_state == Qt.Checked`.
- If `QCheckBox.stateChanged` emits integer `2` for checked, re-checking a
  branch records it as hidden. The visual checkbox is checked, but branch
  overlays are still filtered out by `_compute_label_slots()`.
- The likely upstream fix is to normalize checkbox state before comparison, for
  example `Qt.CheckState(qt_state) == Qt.Checked`, or use a boolean/toggled
  signal.
- A local launcher workaround using that normalization restored annotations
  after branch hide/re-show in the Windows GUI test.

Multi-panel layout retest:

- The 8 kHz diagnostic session remained usable with four total tracks visible:
  audio trace, spectrogram, added audio trace, and added spectrogram.
- Label overlays and playheads were visible across the stacked timeline panels.
- No duplicate/stale playheads were reported during this pass.
- The video freeze was not reproduced during this multi-panel pass.
- Starting and stopping playback had a small delay, but the user described it as
  workable and clearly better than the old EthoGraph version.
- In the 3-channel session, very short playback bursts have a timing-sensitive
  failure mode. The delay after pressing Stop does not seem to be the main
  factor; the critical factor is how long playback was actually allowed to run.
- The refined estimate is that bursts shorter than roughly `0.4 s` do not work
  reliably: the audio snippet plays, but the video stays frozen / the current
  frame position does not commit. On the next Play, playback starts again from
  where the last short segment began. Logs from the same session showed normal
  `audio_clock=True` playback followed by short/rapid playback attempts with
  `audio_clock=False`, consistent with audio-clock/output-stream or
  marker/video-current-frame state not being robustly initialized/advanced for
  very short played intervals.
- Channel switching during playback works visually for audio trace/spectrogram
  display, but the audible channel only reloads after Stop and the next Play.
  In other words, plot channel selection can update live, while the active audio
  stream appears to keep using the channel/source it was opened with.

Mouse-wheel navigation note:

- In the timeline plots, `Lock Axes` is implemented as "prevent zoom but allow
  panning."
- `Nav -> Fixed window` also locks the visible x-span to `fixed_window_s`,
  making wheel/drag interactions behave like timeline panning.
- If mouse-wheel zoom appears lost, first check that `Plots -> Lock Axes` is off
  and `Nav` x-limits mode is `Slider scope`, then wheel over the plot body rather
  than the bottom time slider.

## Current Verdict

`ethograph 0.2.12` is a meaningful improvement on Windows. The automated checks
show that several previous blockers have been addressed or improved, especially
singleton time slices and regular playback having an audio clock. GUI retesting
confirms that regular/main playback can now produce audible audio in
`Audio-synced` mode, the new multi-panel layout is usable with four stacked
tracks, and the branch overlay issue has a small Qt-state normalization fix.
However, video has frozen at least twice while audio and the timeline playhead
continued moving, so the main unresolved risk is video-frame freshness/recovery
under realistic annotation interactions.

Current Windows verdict:

- Video-only short-window annotation remains likely usable.
- Audio-synchronized annotation is no longer ruled out by the old automated
  failures; regular audio playback and four-track layout use are confirmed in
  `Audio-synced` mode, but video-frame refresh/recovery and mild start/stop
  latency still need to be characterized.
- Fresh annotation creation and save/reopen persistence work in the empty-label
  Windows test session.
- Separate WAV audio remains the recommended path; embedded MP4/AAC should be
  treated as unsupported or at least not yet reliable.
