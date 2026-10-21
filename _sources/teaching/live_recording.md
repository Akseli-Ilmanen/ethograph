# Live audio recording

Record from a microphone inside the GUI and watch the sound arrive as an
oscillogram and a spectrogram, the way Raven's live mode does. Every take
becomes a WAV file, and the folder of takes opens as an ordinary session, so
annotation starts the moment recording stops.

Open it from the start page: **🛠 More tools → Live recording → Record audio…**.
A project folder must be chosen first, because the recordings live under it by
default and the label vocabulary comes from it. The entry needs the audio
extra (`pip install ethograph[audio]`).

## What the window does

- **Capture strip.** The input device, how many of its channels to record, the
  sample rate (the device's default is preselected; a rate the device refuses
  is reported, never silently replaced) and the WAV format (16-bit, 24-bit or
  float). Several microphones must be channels of one device: separate
  devices run on separate clocks and drift apart.
- **Output strip.** Where files go and what they are called. The default is
  `{project}/recordings/{YYYY-MM-DD}/`, created on the first take of the day;
  *Browse…* picks any other folder and remembers it, *Today's folder* goes
  back to the default. Files are named `{prefix}_{YYYYMMDD}_{HHMMSS}.wav` and
  never overwrite. A take longer than *Roll every* continues in a new file.
- **Plots.** Monitoring starts as soon as the window opens, before anything is
  written. The right edge is "now"; the window length steps through a fixed
  ladder (0.5 s to 60 s) with the **−** / **+** buttons or Ctrl+wheel. There is
  no free panning while live. After *Stop* the plots show the finished file
  with ordinary pan and zoom, so a take can be checked before the next one.
- **Spectrogram settings.** `nfft`, hop, dB range and colormap are the GUI's own
  spectrogram settings: what is tuned here is what the session opens with.
  Changing one greys the panel for a moment while the visible window is
  recomputed. The dB range fits itself once a second of audio is on screen,
  by the same rule the main GUI uses (noise floor at the bottom of the
  colormap, the loudest bin at the top); *Auto* refits it at any time. The
  frequency axis zooms with the mouse. *Show channel* picks which recorded
  channel the plots draw; every channel is written regardless.
- **Amplitude.** The oscillogram shows full scale (±1) by default, so clipping
  is visible and nothing re-fits while you watch. Double-click it to fit the
  loudest sample on screen at that moment and hold that range; *Full scale*
  goes back.
- **Channel count.** The native Windows and macOS audio APIs accept only the
  device's own channel layout. If the chosen count or rate is refused, the
  recorder retries with the device's native layout and says so in the status
  line.
- **Transport.** *Record*, *Pause* / *Resume*, *Stop*. A peak meter per channel
  and a **CLIP** light that latches until clicked; there is no software gain,
  because gain after the converter changes nothing about clipping.

## Pause

Pausing stops writing but keeps the file open. The file stays contiguous, so a
label's time inside it is a sample time, never a wall-clock time. Each pause is
logged beside the file as `{name}.pauses.tsv` with the file time at which the
gap sits and how long the wall clock jumped.

## Opening the recordings

*Open in Ethograph* loads the folder as a session with one trial per WAV,
exactly as dropping the files onto the start page would. A multichannel WAV
appears as one microphone per channel. Recording again on the same day adds
takes to the same folder; opening it again rebuilds the trial list and keeps
the labels already made.

## What it does not do

Video alongside the audio, and sync pulses for electrophysiology, are not
part of this window. Align such recordings afterwards through the trials
table.
