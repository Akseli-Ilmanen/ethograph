(target-loading-ephys)=
# From an ephys recording

Use this path for extracellular electrophysiology data, with optional Kilosort spike-sorting output.

At least one of: an ephys file **or** a Kilosort folder is required.

Ephys is a **session-wide stream** — the raw recording file is selected in the GUI rather than embedded in the `session.nc`. For datasets with multiple behavioural trials alongside ephys, see {ref}`Ephys with multiple trials <target-ephys-multi-trial>`.

---

## Load it — drag & drop

```{tip}
{doc}`Install EthoGraph <../getting_started/installation>` if you haven't already, then launch via shortcut or:
`conda activate ethograph && ethograph launch`
```

1. On the start page, drag your **ephys file** and/or **Kilosort folder** (and optionally a **video** and/or **audio** file) onto the **Drag & drop** zone.
2. Click **Load**.

EthoGraph recognises a Kilosort folder by the `spike_times.npy` inside it, and reads the sample rate and channel count from the file header or `params.py` automatically — no output path and no questions for known formats. Raw binary (`.dat` / `.bin` / `.raw`) has no header; drop it together with its Kilosort folder so the metadata comes from `params.py` (see below).

---

(target-ephys-viewers)=
## Two ephys trace viewers

Raw traces can be shown in either of two panels, added from **➕ Add panel** (`Shift+N`) under the **Ephys** header.

| | **Neo (`stream`)** | **Ephys (Phy-like viewer)** |
|---|---|---|
| Reads | Any [Neo](https://neo.readthedocs.io)-supported format ({ref}`table below <target-ephys-formats>`) | Raw binary (`.dat` / `.bin` / `.raw`) + Kilosort folder |
| Strength | Wide format compatibility | Fast zooming across many channels, with Kilosort spike waveforms overlaid on the trace |

The Phy-like viewer is inspired by [phy](https://github.com/cortex-lab/phy).

---

(target-ephys-formats)=
## Supported formats

EthoGraph uses [Neo](https://neo.readthedocs.io) to read files with recognised headers — sample rate, channel count, and dtype are extracted automatically. Raw binary files have no header, so they are handled via phylib and require a Kilosort folder.

### Known formats (headers auto-detected)

| Extension(s) | System |
|---|---|
| `.rhd`, `.rhs` | Intan |
| `.oebin` | Open Ephys Binary |
| `.ns1`-`.ns6`, `.nev`, `.nsx` | Blackrock |
| `.abf` | Axon (pCLAMP) |
| `.edf`, `.bdf` | EDF/BDF |
| `.vhdr` | BrainVision |
| `.smr`, `.smrx` | Spike2 (CED) |
| `.ncs`, `.nse`, `.ntt` | Neuralynx |
| `.plx`, `.pl2` | Plexon |
| `.rec` | SpikeGadgets |
| `.meta` | SpikeGLX |
| `.xdat` | NeuroNexus |
| `.tbk` / `.tev` / `.tsq` / ... | TDT |
| `.trc` | Micromed |
| `.edr`, `.wcp` | WinEDR / WinWCP |
| `.nwb` | NWB file|

When a format carries multiple signal streams (e.g. amplifier vs auxiliary channels in Intan), each stream appears as its own `Neo (stream)` entry in the **➕ Add panel** popup, so you can open them side by side in separate panels.

### Raw binary (`.dat` / `.bin` / `.raw`)

Raw binary files produced by Kilosort carry no metadata. They are loaded via [phylib](https://github.com/cortex-lab/phylib) using `n_channels` and `sample_rate` read from `params.py`. Use the **Kilosort folder** picker rather than the ephys file browser — EthoGraph resolves the `.dat` path from `params.py` internally. This is what backs the {ref}`Phy-like viewer <target-ephys-viewers>`.

---

## Kilosort spike sorting output

Point the GUI at a Kilosort output folder via the **Kilosort folder** picker in the Ephys tab.

**Auto-detection:** If a `kilosort4/` or `kilosort/` directory exists next to your ephys file, EthoGraph fills the field automatically on selection.

### Expected files

| File | Required | Description |
|---|---|---|
| `spike_times.npy` | Yes | Sample indices of each spike |
| `spike_clusters.npy` | Yes | Cluster ID per spike |
| `cluster_info.tsv` | Yes | Per-cluster metadata (group, ch, depth, ...) |
| `params.py` | Auto-created if missing | Sample rate, channel count, raw data path |
| `channel_positions.npy` | Yes | Probe site coordinates (x, y) in um |
| `channel_map.npy` | Yes | Site index -> hardware channel mapping |

### `params.py`

`params.py` is a plain Python file written by Kilosort:

```python
dat_path = r'C:\data\recording.dat'
n_channels_dat = 385
dtype = 'int16'
sample_rate = 30000.0
hp_filtered = False
```

EthoGraph reads `dat_path`, `n_channels_dat`, and `sample_rate` from it. If the file is missing or `dat_path` no longer points to a valid file, a dialog prompts for the values and writes a new `params.py` so the step is not repeated.


### What gets loaded

- `cluster_info.tsv` — cluster groups, best hardware channel (`ch`), depth, firing-rate statistics. Both `KSLabel` (automatic Kilosort classification) and `group` (phy manual curation) are imported.
- `channel_positions.npy` + `channel_map.npy` — probe geometry for the raster and probe-channel dialog.
- The raw `.dat` file (from `dat_path`) — displayed in the {ref}`Phy-like viewer <target-ephys-viewers>`, with spike waveforms overlaid on the trace.

---

(target-ephys-multi-trial)=
## Ephys with multiple trials

The native route is {doc}`video_and_ephys`: neuroconv puts the video on the recorder's clock and writes a `session.nwb` with trials. Without a sync line, {ref}`pair the files <target-nwb-alignment>` into a `session.nc` instead and select the ephys file in the GUI as above.
