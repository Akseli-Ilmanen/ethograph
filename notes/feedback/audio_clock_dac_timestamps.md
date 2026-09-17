# Design note: driving the playhead from DAC timestamps, not a wall-clock guess

Written 2026-08-18, in response to "audio sync seems worse than before". Companion to
`processed/handoff_playback_and_video_freeze.md` (work item A). Proposes replacing the
wall-clock bridge in `gui/audio_clock.py` with the mechanism Audacity/mpv use.

---

## 1. The problem, one more time

`AudioClock.elapsed_s()` must answer: *how many media-seconds are audible right now?*
Everything hangs off that number — the red marker, the video frame, and the position
committed on Stop.

What the clock actually knows is `_idx`: output frames **handed to** the device. That
runs *ahead* of what the listener hears by the device's output buffering — 0.1–0.3 s
on Windows. The history:

| Version | `elapsed_s()` | Failure |
|---|---|---|
| 0.2.12 (tester's) | `max(0, idx/rate − latency)` | Pinned at 0 for the first ~0.4 s → short Play→Stop bursts never commit |
| current (`fbd64f9`) | wall-clock bridge blended into the device counter | Marker starts moving at *Play press*, but sound starts ~latency later → **marker/video lead the audio by up to the full device latency** during the early window — the regression being reported |

Both versions guess. The device knows.

## 2. How the established players do it

**Audacity** (same stack as us: PortAudio). Its audio callback stores
`timeInfo->outputBufferDacTime` — PortAudio's statement of *when the first sample of
this buffer physically hits the DAC* — and the playback cursor is derived from that
against the stream clock (`Pa_GetStreamTime()`). The playhead simply does not move
until the DAC time of the first buffer arrives.

**mpv / VLC.** Audio is the master clock; the audible position is
`samples_written − device_reported_delay`, where the delay is **queried live from the
OS every tick** (WASAPI `IAudioClock::GetPosition`, ALSA `snd_pcm_delay`) — never a
static latency constant read once at stream open.

**Browsers.** `AudioContext.getOutputTimestamp()` / `outputLatency` — same idea.

Common denominator: **nobody bridges with wall time.** The audible position comes from
a device timestamp, and a playhead that sits still for ~150 ms after Play is *correct*
— it matches the ear.

## 3. What we change, concretely

All contained in `gui/audio_clock.py`. `video_sync.py`, `audio_player.py`, the chunked
resampler, `stop()`/commit semantics: untouched.

### 3.1 The callback publishes a DAC anchor

`sounddevice` passes the same `time_info` PortAudio gives Audacity. Each callback we
record one **anchor**: "output frame `idx` becomes audible at stream-clock time `dac`".
A tuple assignment is atomic in CPython, so the GUI thread can read it lock-free.

```python
def _callback(self, outdata, frames, time_info, status):
    ...existing chunk-filling code unchanged...
    dac = float(getattr(time_info, "outputBufferDacTime", 0.0) or 0.0)
    now = float(getattr(time_info, "currentTime", 0.0) or 0.0)
    # Sanity-gate: MME hands back zeros/garbage; WASAPI/WDM-KS are good.
    if dac > 0.0 and now > 0.0 and now <= dac <= now + 2.0:
        self._dac_anchor = (idx_before_this_buffer, dac)   # (frames, stream time)
    else:
        self._dac_bad += 1                                 # fall back after a few
    self._idx += filled
    ...
```

### 3.2 `elapsed_s()` extrapolates from the anchor

```python
def elapsed_s(self) -> float:
    if self._final_media_s is not None:
        return self._final_media_s
    anchor, stream = self._dac_anchor, self._stream
    if anchor is not None and stream is not None:
        idx0, dac0 = anchor
        audible = idx0 + (stream.time - dac0) * self._out_rate  # Pa_GetStreamTime
        audible = max(0.0, min(audible, self._idx))  # never beyond handed
        media_s = audible / self._out_rate * self._speed
    else:
        media_s = self._fallback_estimate()  # §3.3
    media_s = min(media_s, self.duration_s)
    media_s = max(media_s, self._last_media_s)  # monotonic floor kept
    self._last_media_s = media_s
    return media_s
```

Properties, for free:

- **Before the first sample is audible** (`stream.time < dac0`): extrapolation is
  negative → clamped to 0. The marker waits exactly as long as the sound does.
- **From the first audible sample**: sample-accurate tracking, zero lead, zero lag.
- **Short bursts commit**: by Stop time the anchor exists (the first callback fires
  within milliseconds of `start()`), so `stop()`'s freeze captures the true audible
  position — never a stuck 0. The 0.4 s burst bug stays dead.
- **Producer underrun**: silence is padded without advancing `_idx`, and the clamp
  `min(audible, _idx)` holds the marker with the sound. Unchanged behaviour.

### 3.3 The fallback (garbage timestamps) stops leading too

Keep a wall-clock estimate for host APIs with unusable `time_info` — but assume sound
starts **latency after** the Play press, not at it:

```python
def _fallback_estimate(self) -> float:
    handed_s = self._idx / self._out_rate
    wall = self._now() - self._wall_start
    est = max(0.0, min(wall - self._latency_s, handed_s - self._latency_s))
    return est * self._speed
```

This is the tester-era formula made honest: it can *lag* the sound slightly if
`stream.latency` over-reports (conservative — a lagging marker never places a label
early), but it can no longer freeze at 0 for a full latency window, because `wall`
keeps advancing while `_idx` alone would not. The blended bridge, `frac`, and the
lead it caused are deleted.

### 3.4 Removed / kept

| Piece | Fate |
|---|---|
| `_wall_start`, `_now` | kept (fallback + tests) |
| bridge blend (`frac`, `min(wall, handed)`) | **deleted** |
| `_last_media_s` monotonic floor | kept (anchor handovers + fallback→anchor switch) |
| `stop()` freeze + `abort()` | unchanged |
| `VideoSync._commit_clock_position` | unchanged — just receives a truer number |

Optional follow-up, not part of this change: open the stream with
`sd.OutputStream(..., latency="low")` and/or prefer the WASAPI host API on Windows
(`sd.default.device` / `hostapi` selection) — smaller buffers shrink the wait-at-Play
window and WASAPI guarantees good timestamps. Worth testing but orthogonal.

## 4. What the user will notice

- On Play, the marker **holds for ~0.1–0.3 s, then moves in lock-step with the
  sound**. That hold is the device buffer filling; Audacity does the same. Today the
  marker jumps immediately and runs ahead of the audio — the "doesn't seem aligned"
  report.
- Stop after any burst length leaves the playhead where the last audible sample was.
- No change to Play latency, channel switching, or speed playback.

## 5. Test plan

Extend `tests/test_unit/test_audio_clock.py` (no device needed — drive the callback by
hand with a fake `time_info` and inject `stream.time`):

1. **Waits, then tracks**: feed callbacks with `dac = now + 0.2`; assert `elapsed_s()`
   is 0 while `stream.time < dac0`, then equals `(stream.time − dac0) · speed` after.
2. **Never exceeds handed frames**: with one small buffer handed, advance
   `stream.time` far past it; assert the position clamps at `_idx / out_rate · speed`.
3. **Burst commit**: two callbacks, `stop()` at `stream.time = dac0 + 0.15`; assert the
   frozen position is 0.15 · speed — non-zero, unlike the 0.2.12 clamp.
4. **Garbage timestamps** (`dac = 0`): anchor never publishes; assert the fallback is
   monotonic, ≤ handed frames, and starts from 0 (no lead).
5. **Monotonic across anchor updates**: interleave anchors with jittered `dac`; assert
   `elapsed_s()` never decreases.

On-device (user's hands, Windows): the tester's original pass criteria still apply —
0.2 s burst advances the playhead; audio/video/marker stay aligned across zoom and
repeated stop/start; channel switch audible without Stop. Plus the new one: **at Play,
the marker starts moving at the same instant the sound becomes audible**, not before.
