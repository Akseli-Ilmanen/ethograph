(target-troubleshooting)=
# Troubleshooting

Report bugs on [GitHub Issues](https://github.com/Akseli-Ilmanen/ethograph/issues). If possible:

1) In the top bar, **Help ▸ Print current state**. Share this message along with your error.
2) If you have data loading problems, send some sample data to akseli.ilmanen@gmail.com, so I can test it myself.


---

## Quick fixes

| Problem | Solution |
|---------|----------|
| A plot or video is frozen, black, blank or out of date | Press `Ctrl+R` (**Help ▸ Reset panels**). Every plot is rebuilt and every video reloaded in place, as if the window had been closed and reopened; the layout, the dataset and your labels are untouched. |
| Unexpected error in the GUI | Save labels (`Ctrl + S`), then restart the GUI. Save semi-regularly! |
| Error with user settings | In the top bar, first try **Help ▸ Reset local settings (this dataset)**. If that does not help, use **Help ▸ Reset global settings (gui_settings.yaml)**. |
| Video, audio or ephys drift apart — the alignment looks off | **Help ▸ Visualize data alignment** draws every stream on its own clock, so you can see which one disagrees. It is usually a mis-specified `alignment.nwb`: a trial's timing, a stream offset or a media path pointing at the wrong file. |
| A dialog you opened is nowhere to be seen | It is behind the main window or on a screen you have since unplugged — dialogs belong to the main window, so they get no taskbar entry of their own. **Help ▸ Gather all windows onto this screen** pulls every one back and raises it; **Window** lists them by name. |
| You upgraded, but nothing changed | Re-run the install with `--force --refresh` instead of upgrading (see {ref}`upgrade-did-nothing`). |
| Plots are squeezed to thin strips or seem missing | The saved layout holds more panels than fit. Close the extras with their ✕, or use **Help ▸ Reset local settings (this dataset)**: it rebuilds the default panels immediately. |

---

## FAQ

### My dataset format is not supported

I/O support for new data formats is actively being expanded. If your format is not yet represented, please send a sample dataset to [akseli.ilmanen@gmail.com](mailto:akseli.ilmanen@gmail.com) and I will work on adding loading support for it.


### What does this button do?

A lot of buttons, spinners, dropdowns, etc. have a little description tooltip when you hover over them.

### Installation fails with "resolution-too-deep"

This happens when using plain `pip` to install ethograph with extras like
`[gui]` or `[all]`. The dependency tree (pygfx + movement + pynwb) is too
complex for pip's resolver.

**Fix:** Use `uv` instead of `pip`:

```bash
uv pip install "ethograph[all]"
```

See {doc}`../getting_started/installation` for full instructions.

(command-not-found)=
### `ethograph` is not recognized as a command

The terminal reports something like:

```text
The term 'ethograph' is not recognized as the name of a cmdlet, function,
script file, or operable program.
```

The install worked — the `ethograph` command just isn't on your `PATH` yet.
`uv tool install` puts it in uv's own bin directory, which uv has to register
with your shell:

```bash
uv tool update-shell
```

Then **close the terminal and open a new one**. `PATH` is only read when a shell
starts, so the current window will keep reporting the same error. After that,
`ethograph launch` works from any directory.


(upgrade-did-nothing)=
### Upgrading did not change anything

The GUI still behaves like the old version. Re-run the install instead of
upgrading — this ignores uv's cached package index and rebuilds the environment
from scratch:

```bash
uv tool install --python 3.12 --force --refresh "ethograph[gui,audio]"
```

`uv tool list` shows which version is installed.


(linux-system-libraries)=
### Linux: system libraries the wheels need

The Python wheels bring their own Qt, OpenGL bindings and wgpu, but on Linux
they load a few shared libraries from the distribution. A desktop install
usually has them; a minimal container, a lab server or a fresh WSL distro
usually does not. **Install them once, before the first launch:**

::::{tab-set}

:::{tab-item} Debian / Ubuntu / WSL
```bash
sudo apt install libgl1 libopengl0 libegl1 libxcb-cursor0 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-keysyms1 libxcb-image0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libfontconfig1 libdbus-1-3 \
    libvulkan1 mesa-vulkan-drivers \
    libportaudio2 libasound2-plugins
```
:::

:::{tab-item} Fedora / RHEL
```bash
sudo dnf install mesa-libGL libglvnd-opengl mesa-libEGL xcb-util-cursor \
    libxkbcommon-x11 xcb-util-wm xcb-util-keysyms xcb-util-image \
    xcb-util-renderutil libxcb fontconfig dbus-libs vulkan-loader \
    mesa-vulkan-drivers portaudio alsa-plugins-pulseaudio
```
:::

::::

The last line of each is only needed with the `audio` extra: PortAudio itself,
plus the ALSA→PipeWire/PulseAudio bridge without which PortAudio cannot reach a
modern sound server.

Run the preflight to see which are still missing on *your* machine — it prints
the exact install line for your distribution:

```bash
ethograph check
```

`ethograph launch` prints the same warning before it opens a window, so if a
launch ends in one of the errors below, scroll up: the fix is already on
screen. (The `libxcb-*` entries are only needed when Qt runs on X11 — on a
Wayland desktop or WSLg the check leaves them out. The ALSA bridge is a plugin
rather than a library, so it is the one item the check cannot see.)

Each missing library fails in its own words:

| You see | What is missing |
|---------|-----------------|
| `OpenGL.platform.ctypesloader \| Failed to load library ( 'libOpenGL.so.0' )` | `libopengl0` (and usually `libgl1`) — PyOpenGL's GLVND dispatch |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | the xcb libraries (`libxcb-cursor0`, `libxkbcommon-x11-0`, …) |
| The GUI opens but the video panel stays black, or wgpu reports no adapter | `libvulkan1` + `mesa-vulkan-drivers` |
| `OSError: PortAudio library not found` | `libportaudio2` (audio extra) |
| The GUI runs but playback is silent | `libasound2-plugins` — see {ref}`no-audio-device` |

See {ref}`wsl` for Windows Subsystem for Linux.

(wsl)=
### Windows Subsystem for Linux (WSL)

The GUI runs under WSLg (Windows 11, or `wsl --update` on Windows 10). A WSL
distro is a minimal install, so start with the system libraries above — every
one of them is typically missing — then `ethograph check`.

Qt runs on Wayland there, which needs none of the `libxcb-*` libraries; only if
you set `QT_QPA_PLATFORM=xcb` yourself do they become mandatory (the check
reads that variable). Video is best-effort: wgpu does not officially support
WSL and falls back to mesa's software renderer or Microsoft's `dzn` driver,
both in `mesa-vulkan-drivers`. If the video panel stays black after installing
them, run ethograph natively on Windows — the data on `/mnt/c/...` is the same
data.

(no-audio-device)=
### Silent audio on Linux

If playback is silent, check what PortAudio sees:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

A default device of `[-1, -1]` means PortAudio found no device. On a
PipeWire/PulseAudio desktop this is almost always the missing ALSA bridge —
install it and restart ethograph:

```bash
sudo apt install libasound2-plugins pipewire-alsa
```

(Debian/Ubuntu's bundled PortAudio only speaks ALSA, so it needs this bridge
to reach a modern sound server.) Routing playback through `pw-play`/`paplay`
instead makes audio audible but not sample-accurately synced, so it isn't
suitable for precise annotation.

(audio-formats)=
### Supported audio formats

The waveform, spectrogram and playback read audio through
[`audioio`](https://github.com/bendalab/audioio) (libsndfile): `.wav`
(recommended — fastest and best tested), `.flac`, `.ogg`, and `.mp3` with a
recent libsndfile (>= 1.1).

**Audio embedded in a video (`.mp4`/`.mov`/`.avi`) is decoded to a WAV first,
never read in place** — libsndfile reads no video container, and an AAC track
has no sample-exact random access. The first time a video is used as an audio
source, its track is decoded once (through PyAV, bundled with the `gui` extra)
into `~/.ethograph/cache/audio_tracks/` and every reader opens that file instead.
This happens both when dropping a video with the **"Extract the audio for audio trace / spectrogram plots"** box ticked
and when an alignment or `.nwb` audio stream points at the video itself. The
cache is keyed by path, size and mtime, so a re-recorded video is never served
a stale extract; delete the folder any time to reclaim the space. A file that
cannot be decoded produces one log line naming it and the reason.

Separate audio files are still better when you can supply them: AAC encoder
priming can shift an extracted track by a few milliseconds, and the extract is
uncompressed. Converting in bulk up front is faster for a whole project:

```bash
for f in *.mp4; do ffmpeg -i "$f" -vn -acodec pcm_s16le "${f%.mp4}.wav"; done
```

### Opening `.tsv` label files in Excel

Excel on Windows may not correctly parse `.tsv` files when double-clicked due to regional delimiter settings.

**Automatic fix:** Ethograph automatically registers `.tsv` files to open correctly in Excel with tab delimiters the first time you run it. On Windows this writes to the current-user registry (no admin prompt); on macOS it uses `duti` if installed.

If the association is not working, you can re-run it manually:

```python
from ethograph.utils.download import ensure_default_configs

ensure_default_configs()
```

**Manual alternative:**

1. Open Excel -> **File -> Open -> Browse**
2. Change file filter to **"All Files (\*.\*)"**
3. Select the `.tsv` file
4. In the Text Import Wizard, select **Tab** as delimiter
