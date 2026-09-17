(target-installation)=
# Installation

## 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager, and the one
ethograph installs with.

::::{tab-set}

:::{tab-item} macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
:::

:::{tab-item} Windows
```
winget install astral-sh.uv
```
Works from both PowerShell and Command Prompt. `winget` is built into Windows 11.
:::

::::

## 2. Install ethograph

Pick what you want to do, then how you manage environments.

`````{tab-set}

````{tab-item} Use the GUI
For annotating data, teaching, or trying it out. No virtual environment
needed — `uv tool` keeps ethograph in its own, managed for you:

```bash
uv tool install --python 3.12 "ethograph[gui,audio]"
ethograph check     # Linux/WSL only: lists missing system libraries
ethograph launch
```

`ethograph launch` says *"not recognized"* or *"command not found"*? Run
`uv tool update-shell` once and **open a new terminal**
(see {ref}`command-not-found`).

Upgrade with `uv tool upgrade ethograph`, remove with
`uv tool uninstall ethograph`.
````

````{tab-item} Write scripts
For importing ethograph in your own code.

:::::{tab-set}

::::{tab-item} uv
```bash
uv venv --python=3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install "ethograph[gui,audio]"
ethograph check                  # Linux/WSL only: lists missing system libraries
```
::::

::::{tab-item} conda
```bash
conda create -y -n ethograph -c conda-forge python=3.12 -y
conda activate ethograph
uv pip install "ethograph[gui,audio]"
ethograph check                  # Linux/WSL only: lists missing system libraries
```

conda only creates the environment; ethograph itself is installed with uv.
::::

:::::

Upgrade with `uv pip install -U "ethograph[gui,audio]"`; if that doesn't seem
to take effect, start from a fresh environment.
````

````{tab-item} Train models
For {doc}`action segmentation <../models/segment/quickstart>` and
{doc}`event spotting <../models/spot/index>`. **PyTorch is installed first,
separately.**

:::::{tab-set}

::::{tab-item} uv
```bash
uv venv --python=3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install --torch-backend=auto torch torchvision
uv pip install "ethograph[gui,audio,model]"
ethograph check                  # Linux/WSL only: lists missing system libraries
```
::::

::::{tab-item} conda
```bash
conda create -y -n ethograph -c conda-forge python=3.12 -y
conda activate ethograph
uv pip install --torch-backend=auto torch torchvision
uv pip install "ethograph[gui,audio,model]"
ethograph check                  # Linux/WSL only: lists missing system libraries
```

Creating the environment from `conda-forge` keeps shared libraries on one
channel, avoiding ABI conflicts with the libraries PyTorch/CUDA depend on.
::::

:::::

Check PyTorch sees your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
````

`````

```{danger}
**Linux/WSL:** missing system libraries can cause a black or failed launch.
`ethograph check` lists what to install; see
{ref}`Linux: system libraries <linux-system-libraries>`.
```

For an editable development install, see {doc}`../community/contributing`.

## Optional extras

Extras are combined with commas, e.g. `"ethograph[gui,audio,dandi]"`. Plain
`ethograph` is the library alone: `TrialTree`, xarray utilities, feature
extraction and label I/O.

| Extra   | What it adds                                                        |
|---------|---------------------------------------------------------------------|
| `gui`   | Graphical interface (PyQtGraph, pygfx/pynaviz, neural tools)        |
| `audio` | Waveform, spectrogram, playback (`sounddevice` etc.)                |
| `model` | Segmentation and spotting pipelines — **install PyTorch first** (see *Train models*) |
| `dandi` | Download client for the [DANDI archive](https://dandiarchive.org/)  |
| `proxy` | Bundled ffmpeg for smoother scrubbing in long videos                |
| `dev`   | Testing and linting tools                                           |
| `docs`  | Documentation build dependencies                                    |

```{note}
On Linux, `audio` also needs PortAudio and the ALSA plugins from your
distribution — part of the one line in
{ref}`Linux: system libraries <linux-system-libraries>`. Silent playback:
see {ref}`no-audio-device`.
```

**`proxy`** — ethograph works fully without ffmpeg; ffmpeg only generates
low-resolution proxies that make seeking long, high-resolution videos smoother.
A system ffmpeg is picked up automatically, or set `ETHOGRAPH_FFMPEG`. The
bundled one has no NVENC, so GPU proxy encoding falls back to `libx264`; for
NVENC use `conda install -c conda-forge ffmpeg`.

(target-keypoint-fill)=
**PosePAL keypoint fill** (GPU only) — the spline and optical-flow fills come
with `gui`; PosePAL needs torch and CoTracker3
(see {doc}`../advanced/keypoint_labelling/fill`):

```bash
uv pip install --torch-backend=auto torch "cotracker @ git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
```

## Where settings live

Global settings live in `~/.ethograph` (override with `ETHOGRAPH_HOME`):

    ~/.ethograph/
    ├── gui_settings.yaml   # your layout, playback and dialog folders
    ├── logs/               # one log per session
    ├── cache/              # derived media: video proxies, extracted audio,
    │                       # example datasets, downloaded weights — safe to delete
    └── defaults/           # a starter project, used while no project folder is chosen:
                            # mapping.txt, config/segment.yaml + spot.yaml to copy from,
                            # config/space/ geometries, runs/lightgbm/ lightgbm models,
                            # workflows/, wizard/ notebooks

An older home folder is rearranged into this shape the first time the GUI starts.
