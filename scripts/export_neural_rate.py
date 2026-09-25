"""Write a smoothed firing-rate TsdFrame beside a session's ``units.npz`` so the GUI can show it.

Usage::

    python scripts/export_neural_rate.py <session>/pynapple/units.npz [name]

The transform is the one ``projects/neural/decoding.yaml`` uses for the
``rate_5ms_gauss25ms`` runs: spikes counted in 5 ms bins, expressed in
spikes/s, then Gaussian-smoothed with a 25 ms std. The result is saved as
``<name>.npz`` next to ``units.npz`` (pynapple's own format, so the GUI picks
it up as a feature with one column per unit) and declared in the session's
``.ethograph/schema.yaml`` as a ``neural_feature``.
"""

import sys
from pathlib import Path

import numpy as np
import pynapple as nap

from ethograph.io import schema

BIN_S = 0.005
SMOOTH_STD_S = 0.025


def main(units_path: Path, name: str) -> Path:
    units = nap.load_file(str(units_path))
    if not isinstance(units, nap.TsGroup):
        raise ValueError(f"{units_path} is a {type(units).__name__}, expected a TsGroup")
    rate = (units.count(BIN_S) / BIN_S).smooth(std=SMOOTH_STD_S)
    rate = nap.TsdFrame(
        t=rate.t, d=np.asarray(rate.d, dtype=np.float32), columns=rate.columns, time_support=rate.time_support
    )
    out = units_path.with_name(f"{name}.npz")
    rate.save(str(out))
    declared = schema.read_sidecar(units_path)
    declared[name] = {"kind": "neural_feature", "units": "spikes/s"}
    schema.write_sidecar(units_path, declared)
    print(f"{out}: {rate.shape[0]} bins x {rate.shape[1]} units at {1 / BIN_S:g} Hz")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "rate_5ms_gauss25ms")
