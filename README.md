# ethograph

<p align="center">
  <img src="docs/source/_static/media/demo.gif" alt="EthoGraph demo" width="40.8%">
  <img src="docs/source/_static/media/label_basic_downsampled.gif" alt="Labelling in EthoGraph" width="58.1%">
</p>

EthoGraph is a graphical user interface for visualizing and segmenting
multimodal timeseries behavioural data. It builds upon a number of
[open-source libraries](#support) to load and quickly render video and pose
files, audio and spectrograms, ephys recordings in various formats, and
arbitrary multi-dimensional timeseries — all on one shared timeline.

Note this GUI is still in development. I welcome people testing it and
[providing feedback](https://akseli-ilmanen.github.io/ethograph/community/contributing.html)!

📖 **[Documentation](https://akseli-ilmanen.github.io/ethograph/)**

🎥 **[Watch the demo](https://vimeo.com/1206424641)**

## Quickstart

First install [uv](https://docs.astral.sh/uv/), a fast Python package manager:

```bash
# macOS / Linux (except WSL)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
winget install astral-sh.uv
```

Then install the GUI as a standalone tool — one command, no environment to
create or activate:

```bash
uv tool install --python 3.12 "ethograph[gui,audio]"
```

To open the GUI, run:

```bash
ethograph check # Linux only : Check for missing libraries
ethograph launch
```

> **`ethograph` not recognized?** uv has not added its bin directory to your
> `PATH` yet. Run `uv tool update-shell`, then open a **new** terminal.

For installing into a dedicated virtual environment and optional extras, and troubleshooting, see
[installation guide](https://akseli-ilmanen.github.io/ethograph/getting_started/installation.html).

After launching, there are some
[example datasets](https://akseli-ilmanen.github.io/ethograph/examples/index.html)
you can explore the GUI with. To learn more about all the functionalities, I
recommend the
[user manual](https://akseli-ilmanen.github.io/ethograph/getting_started/user_manual.html).

## Support

<img src="docs/source/_static/media/opensource.png" alt="Open-source projects EthoGraph depends on" width="60%">

EthoGraph is built on top of a number of open-source projects:
[PyAV](https://pyav.org/docs/stable/),
[audioio](https://github.com/bendalab/audioio),
[Neo](https://neo.readthedocs.io),
[crowsetta](https://github.com/vocalpy/crowsetta),
[Neurodata Without Borders](https://www.nwb.org/),
[xarray](https://docs.xarray.dev/),
[pynapple](https://pynapple.org/index.html),
[movement](https://movement.neuroinformatics.dev/),
[phy](https://github.com/cortex-lab/phy),
[PyQtGraph](https://www.pyqtgraph.org/), and
[pygfx](https://pygfx.org/) (via
[pynaviz](https://github.com/pynapple-org/pynaviz)).

EthoGraph is GPL-3.0-or-later. The code it vendors or adapts from other
projects, and each one's licence, is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
