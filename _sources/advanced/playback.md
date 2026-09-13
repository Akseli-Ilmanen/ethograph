# Video & Audio playback

## Playback modes

The **playback mode** dropdown in the bottom bar picks how video (and audio) play:

- **Audio-synced** — audio and video are locked together, driven by the audio
  output itself, so they never drift. Under load the video may drop frames to
  stay in sync. This is the only mode that plays sound.
- **Smooth (every frame)** — shows every video frame in order; under load it
  runs slower than the set FPS rather than skipping frames. No audio.
- **Real-time (skip frames)** — approximates the set FPS by skipping frames when
  needed. No audio.


### Speed and high-rate (ultrasonic) audio

Playback speed is set as a **% of the original recording** (the "Speed" field
in the bottom bar), not a raw FPS — 100% is native speed, 50% half (one octave
lower), 200% double. It drives video FPS and audio pitch/rate together; a
readout shows the effective rates, e.g. `(120.0 fps, 44.1 kHz)`.

Audio is resampled to a fixed 48 kHz output, so the sound card's max rate never
caps playback. Any speed works (very fast audio just chirps), and **high-rate
recordings** (e.g. ultrasonic audio at 200–384 kHz) play even though no card
outputs those rates directly — set the speed **well below** 100% to
**time-expand** them into the audible range (e.g. 10% shifts a 50 kHz call to
~5 kHz), still locked to the video.


## The red marker timeline marker

Video is recorded as discrete frames (e.g. ~47 per second), but audio and other
signals have far finer time resolution — a birdsong syllable can be shorter than
a single video frame. So the red timeline marker (the playhead) tracks the true
**time**, not the nearest video frame.

When you click on a plot (waveform, spectrogram, line plot):

- the marker jumps to **exactly where you clicked**, down to sub-frame precision;
- the video seeks to the **nearest frame** to that time and displays it;
- a label you place is stored at the **exact clicked time**, not snapped to a frame.

The same holds during playback and when playing a label segment: **audio and
the marker are always exact, the video always shows the nearest frame.** There
is no setting for this — the playhead never snaps to the frame grid.

```{note}
Because the video can only show whole frames, the displayed frame and the red
marker can sit up to half a frame apart (~10 ms). This is expected and lets you
place a syllable boundary precisely between two frames — you judge the boundary
from the waveform/spectrogram, and the video is a reference. To see where the
frames actually fall in time, step through them with the **left/right arrow
keys**: that moves the marker onto exact frame times, so you can read off the
frame boundaries around your label.
```



## Video proxies for smooth navigation

Large or high-resolution video can be slow to move through. The **Proxy**
checkbox in the bottom bar makes it responsive by playing a smaller,
low-resolution copy instead of the original.

Enabling it generates that copy in the background for every visible video (the
app stays responsive meanwhile). A badge on each panel shows progress —
**⏳ proxy…**, **✓ proxy**, or **⚠ proxy** on failure (which keeps the original)
— and the panel switches to the proxy once ready. Uncheck to return to full
resolution.

```{note}
The proxy has the **same frame rate and frame count** as the original — only
the resolution drops — so labels and timing stay exactly aligned. Use full
resolution to inspect fine visual detail, the proxy for general navigation.
```

Proxies are cached in one central folder, **`~/.ethograph/cache/proxies`**, reused
across sessions and generated only once per source file. To free disk space,
just delete that folder — proxies regenerate on demand.


## Which channel Play will sound

When you press **Play** (or Space), the video and the selected audio play
together. With several audio panels open, playback follows the **last audio
selection you made**, so you always hear what you are looking at:

- clicking an audio trace / spectrogram panel, or
- changing that panel's **Channel** dropdown in the right sidebar (shown when an
  audio panel is active).

The bottom bar shows a small speaker icon with the active channel as
`ChN: filename…`; hover over it to see the full channel name.

## Playing back a selected label segment

Left-click a label to select it, then press **V** (or the segment play
button) to play just that segment. The audio is sliced at the label's true
(sub-frame) onset and offset, and the red marker stops on that exact offset —
so playback ends where the label ends, not where the frame grid happens to
fall. The video shows the **nearest frame** to each bound, since it can only
display whole frames; use the **left/right arrow keys** afterwards to step
frame by frame and see where those frames sit relative to the label.

**Auto-play on navigate** starts playback immediately, even while the video
decoder is still warming up after a trial or camera switch: until it is ready,
frames are decoded directly (slightly more work per frame, never a dropped
one), and decoding hands back to the normal path the moment the decoder
responds. Playback therefore begins smoothly on the first frame instead of
freezing or jumping to the segment's end.

```{note}
During playback the marker is driven by the **audio output itself** (an
audio-master clock), so the playhead, the video frame and the sound stay locked
and don't drift apart over long clips. If no audio device is available, playback
falls back to a frame timer; above ~2.5× speed the audio is dropped (it would be
unintelligible) and the video free-runs.
```
