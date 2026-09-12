(target-installation)=
# Installation

## Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager.
ethograph uses uv for installation.

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

## Install ethograph

Use this if you just want to **run the ethograph GUI** — for teaching, for
annotating data, or to try it out. It is one command, and there is no
environment to create or activate.

```bash
uv tool install --python 3.12 "ethograph[gui,audio]"
```

Then launch it from any terminal:

```bash
ethograph check # Linux only : Check for missing libraries
ethograph launch
```

`uv tool install` puts ethograph in its own isolated environment and adds the
`ethograph` command to your PATH, so it never clashes with your other Python
projects and is always available without activating anything.

```{danger}
**Linux/WSL:** missing system libraries can cause a black or failed launch. Run `ethograph check` to see what you need to install. See further {ref}`Linux: system libraries <linux-system-libraries>`.
```

First `ethograph launch` may fail with *"not recognized"* (Windows) or *"command not found"* (macOS/Linux) — uv hasn't added its bin dir to `PATH` yet. Fix once with:

    uv tool update-shell

then **open a new terminal** (PATH is only read at shell start). See {ref}`command-not-found` for alternatives.

```{tip}
To update later, run `uv tool upgrade ethograph`; to remove it,
`uv tool uninstall ethograph`. Global settings live in `~/.ethograph`
(override with the `ETHOGRAPH_HOME` environment variable):

    ~/.ethograph/
    ├── gui_settings.yaml   # your layout, playback and dialog folders
    ├── logs/               # one log per session
    ├── cache/              # derived media: video proxies, extracted audio,
    │                       # example datasets, downloaded weights — safe to delete
    └── defaults/           # a starter project, used while no project folder is chosen:
                            # mapping.txt, config/segment.yaml + spot.yaml to copy from,
                            # config/space/ geometries, runs/lightgbm/ onset models,
                            # workflows/, wizard/ notebooks

An older home folder is rearranged into this shape the first time the GUI starts.
```

## Install into a virtual environment

Use this approach instead if you want to **write scripts or code against
ethograph** — import `TrialTree`, build pipelines, or develop the package. Here
ethograph is installed *into an environment you activate*, alongside whatever
else you import, rather than as a standalone tool.

You can use either conda or uv to create the environment — conda is only
used for environment creation, not for installing ethograph itself.

::::{tab-set}

:::{tab-item} conda
```bash
conda create -y -n ethograph python=3.12
conda activate ethograph
```
:::

:::{tab-item} uv
```bash
uv venv --python=3.12
```

Activate it:
```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```
:::

::::

With the environment activated — you should see its name (e.g. `(ethograph)`)
in your prompt — install the package:

```bash
uv pip install "ethograph[gui,audio]"
```

Plain `ethograph`, with no extras, gives you the library alone: the `TrialTree`
data structure, xarray utilities, feature extraction and label I/O, with no GUI
and no audio. Upgrade later with `uv pip install -U "ethograph[gui,audio]"`.

```{hint}
If an upgrade doesn't seem to take effect, create a fresh environment and
install from scratch.
```

In a **conda environment** you can also create a desktop shortcut, which
launches the GUI on double-click:

```bash
ethograph shortcut
```

## Optional extras

ethograph uses optional extras to keep the base install lightweight. Combine
them with commas — `uv pip install "ethograph[gui,audio,dandi]"`, or
`uv tool install --python 3.12 "ethograph[gui,audio,dandi]"` for a tool install.

| Extra          | What it adds                                                        |
|----------------|---------------------------------------------------------------------|
| `gui`          | Full graphical interface (PyQtGraph, pygfx/pynaviz, neural tools)   |
| `audio`        | Waveform, spectrogram, vocalisation analysis (`sounddevice` etc.)   |
| `dandi`        | Download client for the [DANDI archive](https://dandiarchive.org/)  |
| `proxy`        | Bundled ffmpeg for faster scrubbing in long videos                  |
| `model`        | Segmentation pipeline — model training (see below)                  |
| `dev`          | Testing and linting tools                                            |
| `docs`         | Documentation build dependencies                                    |

```{note}
On Linux, `audio` also needs PortAudio and the ALSA plugins from the
distribution — both are part of the one system-library line in
{ref}`Linux: system libraries <linux-system-libraries>`. If playback stays
silent afterwards, see
{ref}`no-audio-device`.
```

### Faster scrubbing in long videos (`proxy`)

EthoGraph works **fully without ffmpeg** — every feature is available, video
included. The only thing ffmpeg adds is *proxy generation*: a low-resolution
copy that makes seeking through long, high-resolution recordings smoother.
The `proxy` extra bundles a private ffmpeg binary (via `imageio-ffmpeg`) with
no PATH setup; a system ffmpeg is picked up automatically, or point at one with
the `ETHOGRAPH_FFMPEG` environment variable.

```{note}
The bundled ffmpeg has no NVENC, so GPU (`cuda`) proxy encoding falls back to
software `libx264`. For NVENC, install ffmpeg from conda-forge
(`conda install -c conda-forge ffmpeg`) or your system package manager.
```

## For developers

To install latest development version in editable mode see {doc}`../community/contributing`.


## Segmentation pipeline (model training)

The segmentation pipeline learns your
curated state labels and predicts them back into the GUI. It is scripted,
not a command line: one config becomes a `Project` with a method per stage.
It needs PyTorch; install a build matching your GPU first, then the extra:

```bash
uv pip install --torch-backend=auto torch torchvision # --torch-backend=auto for windows
uv pip install "ethograph[model]"
```

```python
import ethograph as eto

eto.segment.architectures()         # lists the available models
```


```{tip}
Using `conda-forge` (`conda create -y -n ethograph -c conda-forge python=3.12`)
can help here: it keeps all conda-installed packages on one channel, avoiding
ABI conflicts between `defaults` and `conda-forge` builds of shared libraries
that PyTorch/CUDA packages depend on.
```


(target-keypoint-fill)=
## Keypoint labelling backends (optional)

**Tools ▸ Keypoint labelling…** lets you label a few frames by clicking the
video and fill the rest automatically.

| Backend | Method | Uses video | Speed | Hardware |
|---------|--------|------------|-------|----------|
| **Spline** (default) | Monotone piecewise cubic (PCHIP) interpolation per keypoint, over that keypoint's own labelled frames[^pchip] | No — geometry only | Instant; nothing is decoded | CPU |
| **Optical flow** | Pyramidal Lucas-Kanade sparse tracking, run forward and backward across each gap[^lk] | Yes | Roughly real-time | CPU |
| **PosePAL (CoTracker3 + refinement)** | A transformer point tracker[^cotracker] whose per-keypoint appearance features are first fitted to the frames you labelled[^posepal], then run forward and backward across each gap | Yes | A few minutes for the fit, once; seconds per fill after that | **GPU only** — CUDA or Apple Silicon |

Spline and optical come with `ethograph[gui]`. PosePAL (GPU only) reequires a separate install:

```bash
uv pip install --torch-backend=auto torch "cotracker @ git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
```




[^pchip]: Fritsch, F. N. & Carlson, R. E. (1980). [Monotone Piecewise Cubic Interpolation](https://doi.org/10.1137/0717021). *SIAM Journal on Numerical Analysis*, 17(2), 238–246. Implemented by [`scipy.interpolate.PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

[^lk]: Lucas, B. D. & Kanade, T. (1981). [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). *IJCAI*, 674–679. The pyramidal form used here is Bouguet, J.-Y. (2001), [Pyramidal Implementation of the Lucas Kanade Feature Tracker](https://robots.stanford.edu/cs223b04/algo_tracking.pdf), via [`cv2.calcOpticalFlowPyrLK`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323).

[^cotracker]: Karaev, N., Makarov, I., Wang, J., Neverova, N., Vedaldi, A. & Rupprecht, C. (2024). [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://arxiv.org/abs/2410.11831). [Project page](https://cotracker3.github.io/) · [GitHub](https://github.com/facebookresearch/co-tracker)

[^posepal]: Pan, Z., Pan, B., Yang, G., Harley, A. W. & Guibas, L. (2025). [Animal Pose Labeling Using General-Purpose Point Trackers](https://arxiv.org/abs/2506.03868). Reference implementation: [PosePAL](https://github.com/Zhuoyang-Pan/PosePAL). EthoGraph implements the method against upstream CoTracker3 rather than the authors' fork; the optimiser settings follow the paper (Adam, 1e-3 → 1e-5, Huber tracking loss, L1 pull-back weighted 0.01).
