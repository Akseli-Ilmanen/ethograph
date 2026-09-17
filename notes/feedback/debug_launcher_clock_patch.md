# `launch_ethograph_debug.py` — `_clock_start_frame` AttributeError (ethograph ≥ 2.16)

## What happened

The debug launcher monkeypatches `VideoSync._advance_from_clock` with a copy of the
implementation from an older ethograph build:

```python
target = min(self._clock_start_frame + int(round(elapsed * self.fps)), self.total_frames - 1)
```

`_clock_start_frame` no longer exists. The audio-clock start anchor was split into two
attributes with *different units*:

| attribute             | meaning                                                        |
|-----------------------|----------------------------------------------------------------|
| `_clock_start_t`      | trial seconds where the played audio span begins (`frame / fps`) |
| `_clock_start_marker` | display-clock time of the playhead at Play (`frame_to_time(frame)`) |

`_clock_start_marker` is a *time on the display clock*, not a frame index — substituting it
into the old expression multiplies a time by fps and seeks to a nonsense frame.

Also note the current implementation sets `marker_time_override`, which is what keeps the
playhead on the exact sub-frame audio position. A replacement that omits it re-quantizes
the marker to the frame grid — i.e. the instrumentation itself would introduce the A/V
offset being measured.

## Fix: wrap, don't reimplement

Replace the whole `logged_advance_from_clock` block with a wrapper that calls the real
method and only logs around it. This is version-proof — it keeps working across further
changes to the clock internals.

```python
_orig_advance_from_clock = VideoSync._advance_from_clock


def logged_advance_from_clock(self):
    clock = self._audio_clock
    elapsed = clock.elapsed_s() if clock is not None else 0.0
    before = self._current_frame

    _orig_advance_from_clock(self)

    log.info(
        "clock elapsed=%.4f  frame %d -> %d  marker=%s  audio_t=%.4f",
        elapsed,
        before,
        self._current_frame,
        f"{self.marker_time_override:.4f}" if self.marker_time_override is not None else "None",
        self._clock_start_t + elapsed,
    )


VideoSync._advance_from_clock = logged_advance_from_clock
```

`self._clock_start_t + elapsed` is the position the audio device is actually at;
`self.marker_time_override` is where the playhead is drawn. Those two agreeing is the
invariant worth logging — the frame number is expected to lag by up to half a frame,
which is correct behaviour, not drift.

## While you're in that file

Any other patch in the launcher that reaches into `VideoSync` internals is exposed to the
same breakage. The stable surface is: `frame_to_time()` / `time_to_frame()`,
`current_frame`, `marker_time_override`, `is_playing`. Everything prefixed `_clock_*`,
`_smooth_mode`, `_segment_end_*` is private and moves between releases.
