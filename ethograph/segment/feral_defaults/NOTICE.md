# FERAL default config — copied, not written

`default_config.yaml` is a verbatim copy of `feral/default_config.yaml` from
[FERAL](https://github.com/Skovorp/feral) (Peter Skovorodnikov, Jacopo
Razzauti; MIT licence), release 1.0.1, commit `f42aa4d` (2026-09-08).
`presets.yaml` transcribes the three sparse overlays of its `feral/presets.py`
(`lite`, `max`, `rare`) at the same commit, value for value.

FERAL is **not** installed in the ethograph environment, and `feral train-config` reads a YAML file
verbatim, without merging it over the packaged defaults. So the config
ethograph writes for it (`ethograph/segment/feral.py`) has to be complete,
and these two files are where its defaults come from: no FERAL default is
written in our code.

When FERAL changes its defaults, replace `default_config.yaml` with the new
copy, update `presets.yaml` from `presets.py`, and bump the version and
commit above. The config ethograph writes names the version it was built
from in its header, so a mismatch with the installed FERAL is visible.
