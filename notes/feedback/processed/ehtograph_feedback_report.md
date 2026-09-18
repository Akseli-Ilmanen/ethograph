# Ethograph Feedback Report: Audio/Video Annotation Feasibility

Prepared for: Akseli Ilmanen
Prepared by: Luca Yapura with Codex assistance
Date: 2026-07-23
Ethograph version tested: `0.2.4.4`

## Short Summary

We tested Ethograph for a two-animal copulation annotation workflow that needs synchronized video, imported behaviour labels, and external audio. Ethograph worked well for loading converted sessions, inspecting labels, using branch mappings, and video-only short-window annotation. The main unresolved problem is audio-synchronized annotation: once audio is involved, playback becomes choppy/delayed, the playhead can become unreliable during repeated play/pause, and regular/main playback does not appear to start the selected audio stream in the upstream/native Windows path.

The strongest positive result is that Windows native audio output can work in principle: state-label/interval playback produced audible audio from a mono PCM16 WAV diagnostic. The strongest negative result is that this did not translate into robust continuous audio/video playback suitable for precise annotation.

## What The Project Needs

Our annotation workflow needs:

- short video windows around behaviour events, usually about 2 minutes,
- reliable imported-label inspection,
- manual creation/editing of point and state events,
- accurate audio/video synchronization,
- responsive timeline zooming and repeated play/pause during fine annotation,
- ideally side-by-side behavioural lanes for Male and Female,
- access to at least one microphone/radio audio channel during annotation.

The practical bar is not cinematic smoothness; it is that the displayed video frame, audible audio, and red playhead remain synchronized enough that a human can place labels confidently.

## Environment Tested

### Linux Workstation

```text
OS: Linux
Python: 3.12
ethograph: 0.2.4.4
napari: 0.7.1
napari-pyav: 0.1.1
audioio: 2.8.1
sounddevice: 0.5.5
```

Linux-specific finding:

```text
sounddevice default = [-1, -1]
audioio: cannot open any device for audio output
```

The Linux install needed a local workaround that routed playback through `pw-play`/`paplay`/`aplay` to make audio audible. That workaround helped audibility but was not tightly synchronized enough for precision annotation.

### Windows Laptop

```text
OS: Windows 11, user-confirmed
Python: 3.12.13
ethograph: 0.2.4.4
napari: 0.7.1
napari-pyav: 0.1.1
audioio: 2.8.1
sounddevice: 0.5.5
soundfile: 0.14.0
numpy: 2.4.6
xarray: 2026.7.0
pynwb: 4.0.0
```

Windows setup notes:

- PowerShell script execution was blocked initially.
- `py` was not on `PATH`; install used the available Python 3.12 executable.
- When launched from Codex, napari needed `HOME`, `USERPROFILE`, `LOCALAPPDATA`, and `APPDATA` pointed inside a local `.home` folder.
- Direct `sounddevice` output from the same venv worked: a 440 Hz test tone was audible through Realtek speakers.
- WAV files were non-silent and could be decoded by `audioio.load_audio`.

## Data Used For Testing

The final diagnostics used two short trial windows, each about 2 minutes:

```text
Copulation-relevant trial:
  source video: CopExpBP01_Exp_video01
  source window: 239.243-359.997112 s
  duration: 120.754112 s
  anchor: cloacal contact at trial-relative 60.000-60.797 s

Control trial:
  source video: CopExpBP01_CTRL_video01
  source window: 84.49571132319915-204.49571132319915 s
  duration: 120.000 s
  anchor: deterministic random control section, seed 20260717
```

Original source video properties:

```text
codec: H.264 / yuv420p
resolution: 2756 x 2016
fps: 47.6837158203125
duration: about 360 s per source clip
embedded audio: AAC stereo, 24 kHz
```

Diagnostic media variants:

```text
original_noaudio:
  original-resolution short-window MP4s, no audio

proxy1280_noaudio:
  1280 x 936 proxy MP4s, no audio

proxy1280_mic1:
  1280 x 936 proxy MP4s
  external mono WAV, 24414 Hz, float32, one DAQ mic channel

proxy1280_mic1_pcm16_16k:
  same proxy video
  external mono PCM16 WAV, 16000 Hz

proxy1280_mic1_pcm16_8k:
  same proxy video
  external mono PCM16 WAV, 8000 Hz

proxy1280_audio3ch:
  same proxy video
  external 3-channel WAV, 24414 Hz, float32

proxy1280_audio:
  same proxy video
  external 6-channel WAV, 24414 Hz, float32

proxy1280_mpaudio:
  1280 x 936 MP4s with embedded AAC stereo audio, 24 kHz
  no separate WAV
```

The `.nc` files are intentionally small and contain only a minimal timeline variable plus trial metadata and label sidecars. Video/audio are external files referenced through the Ethograph alignment sidecars. This isolates GUI media playback, audio plotting, zooming, and synchronization rather than testing a large dense feature dataset.

## Results: What Worked

### Data Loading And Labels

- Converted Ethograph sessions loaded.
- Imported behaviour labels loaded.
- Branch mapping worked after selecting the relevant branch in the GUI.
- Manual label creation/editing looked usable in short-window sessions.
- Local video files worked better than reading media from a network/NAS mount.

### Video-Only Playback

- `proxy1280_noaudio` was smooth at the source frame rate when `Skip Frames` was disabled.
- `original_noaudio` was also mostly smooth after an initial wait, with only slight zoom delays.
- This suggests that video alone is not the main blocker for our workflow.

### Native Windows Audio, In Principle

- A direct `sounddevice` tone test was audible on Windows.
- `audioio.load_audio` could decode the external WAV files.
- State-label/interval playback produced audible audio from the 16 kHz PCM WAV diagnostic.

This means the Windows audio device and Python audio stack were functional.

## Results: Main Problems

### 1. Regular/Main Play Did Not Start Selected Audio

In the upstream/native Windows test path, pressing the regular/main Play button advanced video but did not start the selected WAV audio stream. Audio was heard from state-label/interval playback, so the audio file and audio device were not the limiting issue.

Expected behaviour:

```text
If an audio source/mic is selected, regular/main Play should play video and audio together.
```

Observed behaviour:

```text
Regular/main Play: video-only / no audible selected audio.
State-label/interval playback: audible audio, but with noticeable delay.
```

Why this matters:

For manual annotation, users need to scrub, play, pause, and annotate from the main timeline, not only play pre-existing label intervals.

### 2. Audio-Linked Sessions Became Choppy Or Delayed

The no-audio proxy session was smooth, but adding audio made playback less reliable. This remained true even after reducing audio from 6 channels to 1 channel and after pre-downsampling to 16 kHz and 8 kHz mono PCM16 WAV.

Observed with audio diagnostics:

- playback initially improved at 30 fps with lower-rate PCM WAVs,
- zooming the timeline caused delays,
- repeated start/stop caused choppy playback or timeouts,
- red playhead movement could become delayed/choppy,
- rapid play/pause could repeat the same section rather than continuing cleanly.

Interpretation:

The bottleneck does not look like raw video decoding or WAV file size alone. It appears to involve the audio-linked playback/plotting/synchronization path and GUI event-loop responsiveness.

### 3. Playhead State Can Become Unreliable During Rapid Play/Pause

Rapid repeated spacebar presses are important for fine annotation. In testing, this sometimes caused:

- duplicate/stale red vertical playhead bars,
- repeated playback of the same section,
- delayed playhead movement relative to audio.

Expected behaviour:

```text
Repeated play/pause should be idempotent: stop previous playback, clear stale marker state, and resume from the current playhead position.
```

### 4. Timeline Zoom And Audio Plot Refresh Can Stall The GUI

When audio trace/spectrogram were visible, timeline zooming and channel switching could become slow. On Linux, changing channel after zooming caused an `org.napari.python3 is not responding` warning that recovered after choosing Wait. On Windows, lower-rate audio improved initial responsiveness but still became delayed/choppy or timed out after zooming and repeated playback interactions.

Hiding spectrogram/audio trace did not fully fix playhead accuracy in Linux tests, so the issue may not be only spectrogram rendering. Still, waveform/spectrogram refresh appears to be one important stressor.

Expected behaviour:

```text
Zooming and channel switching should not block playback or make the UI temporarily unresponsive.
```

### 5. Embedded MP4/AAC Audio Did Not Load

The source/proxy MP4 files contain valid embedded AAC audio according to `ffprobe`, but Ethograph/audioio did not load the MP4 as an audio source on either Linux or Windows.

Windows GUI console showed repeated errors like:

```text
ethograph.gui.plots_spectrogram - ERROR - Failed to load audio file ... .mp4
soundfile: format not recognized
wave: file does not start with RIFF
```

Non-GUI `audioio.load_audio` also failed on the embedded-AAC MP4 files.

Expected behaviour, if MP4 audio is intended to be supported:

```text
MP4/AAC audio should load for trace, spectrogram, and playback.
```

If it is not intended to be supported, it would help if the documentation and file picker clearly say that separate WAV/FLAC/etc. files are required for audio analysis/playback.

### 6. Singleton Time-Dimension Loading Bug

The `No dimension containing 'time' found in the DataArray` error occurred on both Linux and Windows when loading our session. The local fix was to preserve the time dimension when squeezing singleton selections in:

```text
ethograph/utils/xr_utils.py
```

The working local behaviour is:

```text
Find the time dimension after applying selection, then squeeze only non-time singleton dimensions.
```

A smoke test with a zero-width timeline selection returned shape `(1,)` after the patch.

### 7. Alignment Sidecar Size Scales With Audio Sample Rate

For two short 2-minute trials, approximate `.nwb` alignment sizes were:

```text
no-audio alignment:   about 177 KB
8 kHz mono audio:     about 14.9 MB
16 kHz mono audio:    about 29.6 MB
24.414 kHz audio:     about 45.0 MB
```

This looks like dense per-sample timestamp storage. For constant-rate audio, rate/start-time metadata would likely scale much better.

Why this matters:

Our full dataset would contain many trials. Dense alignment sidecars could become unnecessarily large even if the actual audio is kept small.

## Reproduction Steps

The private biological videos are probably not necessary to reproduce the core issues. A synthetic dataset should be enough.

### Suggested Synthetic Test Dataset

Create a two-trial Ethograph session with:

```text
Trials:
  2 trials, each about 120 s

Video:
  H.264/yuv420p MP4
  1280 x 936
  47.683716 fps
  no embedded audio for external-WAV tests

External audio:
  mono WAV, PCM16, 16000 Hz
  optionally repeat with 8000 Hz and float32 24414 Hz

Labels:
  a few state intervals and point events
  two individuals, e.g. Male and Female

Alignment:
  video_cam-1 and audio_mic-1 linked per trial
```

Useful synthetic media content:

- video with a moving timestamp/grid/counter,
- audio click train or chirp aligned to visible flashes,
- a few state events around known audio/video events.

### GUI Reproduction Actions

1. Load the two-trial session.
2. Select the audio source/mic.
3. Press regular/main Play.
4. Check whether selected audio plays with video.
5. Play a state-label/interval and check whether audio is audible.
6. Set `Playback FPS = 30`, `Skip Frames = false`; then repeat at source rate `47.683716 fps`.
7. Keep audio trace and spectrogram visible and zoom the timeline repeatedly.
8. Press spacebar rapidly several times.
9. Hide audio trace and spectrogram and repeat playback/zoom interactions.
10. Repeat with an embedded-AAC MP4 instead of separate WAV.

### Observed Behaviour To Compare Against

```text
No-audio proxy video: smooth.
External WAV selected, regular/main Play: video advances but no selected audio heard.
External WAV, state-label/interval playback: audio audible on Windows, but delayed.
External WAV, after zoom/repeated start-stop: choppy/delayed or timeout.
Embedded MP4/AAC: audio trace/spectrogram fails to load.
Rapid play/pause: stale/duplicate playhead markers or repeated section can occur.
```

## Suggested Improvements

These are listed from most blocking to more ergonomic.

### Audio/Video Playback

- Start selected audio during regular/main Play, not only during interval/label playback.
- Use one shared media clock for audio, video frame selection, and playhead movement.
- Make repeated play/pause idempotent: stop previous playback, clear stale marker state, and resume from the current playhead.
- Add a general linked playback-speed control such as `0.25x`, `0.5x`, `1.0x`, `2.0x`, with video FPS and audio speed derived from that setting.
- Keep `Skip Frames` behaviour clearly tied to real-time playback versus visual continuity.

### Audio Backend And Format Support

- Add an explicit audio output-device selector.
- Surface clear errors when no usable output backend/device is available.
- Document which audio formats are supported for waveform/spectrogram/playback.
- If MP4/AAC audio is intended to work, route it through a decoder that handles MP4 containers reliably.
- If MP4/AAC is not intended to work, make that clear in the GUI and docs.

### Plotting Responsiveness

- Throttle, cache, or background waveform/spectrogram recomputation during zoom and channel switching.
- Avoid blocking the GUI thread during audio channel changes.
- Keep the red playhead responsive even when plots are refreshing.

### Alignment Storage

- For constant-rate audio/video, store start time and rate metadata rather than dense per-sample timestamps where possible.
- Warn or summarize when alignment sidecars become unexpectedly large.

### Annotation Usability

- Show multiple audio channels as stacked traces or make channel switching cheaper.
- Show meaningful channel names rather than only `Ch 1`, `Ch 2`, etc.
- Support Male/Female label lanes simultaneously.
- Support branch names, not only branch numbers.
- Provide a clear `show all imported labels` review mode or auto-populate branch overlay slots when a mapping contains multiple branches.
