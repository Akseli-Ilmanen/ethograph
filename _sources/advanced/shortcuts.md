(target-shortcuts)=
# Keyboard Shortcuts

Plain-letter shortcuts (`Space`, `V`, label keys, arrow keys) are suppressed
while you are typing in a text field, spin box or editable combo box.

There's a rough convention to the modifier: labelling actions are usually a
single key (e.g. `E` to activate label 5), opening/switching/hiding panels is
usually `Shift+`, and everything else (miscellaneous plot/navigation/data
actions) is usually `Ctrl+`. Not every shortcut fits perfectly, but it's a
useful guide when guessing a binding.

## Video/Audio Control

| Shortcut | Action |
|----------|--------|
| `Space` | Toggle play/pause video and audio (or audio-only in no-video mode) |
| `V` | Play selected label segment |
| `Left` / `Right` | Step one frame backward / forward (video mode) or one time-step (no-video mode) |
| `Shift+Left` / `Shift+Right` | Jump backward / forward by customizable time step (see "Jump step" in Navigation) |
| `Ctrl+Space` | Stop screen recording |

## Navigation

| Shortcut | Action |
|----------|--------|
| `Up` | Previous trial |
| `Down` | Next trial |
| `Ctrl+Up` / `Ctrl+Down` | Previous / next channel (ephys or audio mic) |

## Mouse Controls

| Action | Function |
|--------|----------|
| **Left Click** | Select label (when not in label mode) |
| **Double Left Click** | Autoscale axes |
| **Right Click** | Seek video to clicked time |
| **Mouse wheel** | Zoom in/out both axes |
| **Right Click + drag horizontally** | Zoom in/out along the time axis |
| **Right Click + drag vertically** | Zoom in/out along the y-axis |

## Labelling

| Shortcut / Action | Description |
|-------------------|-------------|
| `1-9`, `0` | Activate label 1-10 |
| `Q W E R T Z U I O P` | Activate label 11-20 |
| `A S D F G H J K L Y` | Activate label 21-30 |
| `F1-F10` | Activate label 31-40 |
| Click twice on line plot | Define label boundaries (set start/end) — *Graph based labelling* mode |
| Label key at a frame | Place the boundary at the frame on screen: a point event on one press, a state event on two — *Frame by frame labelling* mode (Labels tab, **Mode**) |
| Left-click on label | Select existing label |
| `Ctrl+E` | Edit selected label boundaries (after selecting label, click twice for new boundaries) |
| `Ctrl+D` | Delete selected label (after selecting label) |
| `Ctrl+Z` | Undo the last label placed, moved or deleted (cancels a half-placed label first) |
| `Ctrl+S` | Save `labels.tsv` file |
| `Shift+B` | Swap the active branch with the previously-active one |

## Curation

See {doc}`../models/curation`. `Ctrl+C` is always bound; the others are live
only while a frame-by-frame review runs (started from the Labels tab's
Curation section), wherever the key is pressed.

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Curate the current trial: every automated label in scope becomes curated (manual ones stay manual) |
| `Enter` | Confirm the frame on screen as the boundary — the label becomes manual if it moved, curated if not — and move on |
| `Backspace` / `Delete` | Delete the event being reviewed and move on |
| `N` | Next boundary (curates the one you leave when **Click N curates current** is ticked) |
| `B` | Previous boundary |
| `Left` / `Right` | Step one frame — the main window's own binding |

## Keypoint Labelling

Live while the {doc}`keypoint labelling dialog <keypoint_labelling/index>` is
open, wherever the key is pressed — on the dialog, on the video canvas or
anywhere in the main window. `Backspace`, `Delete`, `Ctrl+Z`, `Shift+H` and `N`
work even with no labelling mode armed; the rest need one.

| Shortcut | Action |
|----------|--------|
| `Backspace` / `Delete` | Delete the active point on this frame (the outlined one, else the one under the cursor; with no mode armed, the pair selected in the Keypoints tree) |
| `Ctrl+Z` | Undo the last point placed, moved or deleted |
| `Shift+H` | Approve this frame — keep every predicted point on it as your own label, for all individuals at once, then go where **Then go to:** says |
| `Tab` / `Shift+Tab` | Cycle to the next / previous keypoint of the active individual |
| `1-9` | Select the individual to label |
| `N` | Jump to the next suggested frame, wrapping at the end (there is no "previous" — click any row of the points table instead) |
| `Left` / `Right` | Step one frame — the main window's own binding, untouched |
| Left-click on the video | Place the active keypoint, or select and drag an existing point |
| Left-click on a filled point | Pin the prediction as a label (drag to correct it first) |
| `Shift`+left-drag | Pan the video while a mode is armed (tick **Lock** to pan with a plain drag) |

## Selection Cycling

| Shortcut | Action |
|----------|--------|
| `Ctrl+I` | Toggle to last-visited individual, or cycle to next |
| `Ctrl+K` | Toggle to last-visited keypoint, or cycle to next |
| `Shift+C` | Cycle to next camera |
| `Shift+M` | Toggle to last-visited microphone, or cycle to next |

## Plot Controls

| Shortcut | Action |
|----------|--------|
| `Shift+N` | Open the "Add panel" popup |
| `Ctrl+R` | Reset panels: rebuild every plot and reload every video in place, keeping the layout (also **Help ▸ Reset panels**) |
| `Ctrl+A` | Toggle autoscale |
| Left double-click | Autoscale (once) |
| `Ctrl+L` | Toggle lock axes |

## Layout

| Shortcut | Action |
|----------|--------|
| `Shift+Z` | Zen mode — slide the sidebar out / back in and skip its refreshes |

## Changepoint Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl+Right` | Jump to next changepoint (audio CPs if audio/spectrogram panel was last clicked, otherwise kinematic CPs) |
| `Ctrl+Left` | Jump to previous changepoint (same panel context) |
| `Ctrl+B` | Toggle changepoint correction |

## Ephys Trace

| Shortcut | Action |
|----------|--------|
| `Ctrl+H` | Cycle neural view: Multi Trace -> Raster |
| `Ctrl+=` / `Ctrl+-` | Increase / decrease channel spacing |
| `Alt+Right` / `Alt+Left` | Jump to next / previous spike |
| **Ctrl+Wheel** | Adjust display gain |

## 3D Space

| Control | Action |
|---------|--------|
| **Left drag** | Orbit (rotate around center) |
| **Middle drag** | Pan (move look-at point) |
| **Scroll** | Zoom in / out |
| **Arrow keys** | Fine rotation (azimuth / elevation) |
