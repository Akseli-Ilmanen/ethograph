"""Loading a session with many trials must stay linear in the number of trials.

The trials table asks the alignment for one filename per trial per camera;
a per-trial read of the acquisition's ``external_file`` dataset, or a scan of
the trials table per lookup, turns that into a quadratic load (146 s for
2000 trials, once). The ratio check catches the shape of the growth, so it
holds on a slow machine too; the budget catches a constant-factor blow-up.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ethograph.io.metadata_table import media_columns_from_alignment
from ethograph.io.nwb_alignment import NWBAlignment
from ethograph.io.pairing import pair_media

N_SMALL = 400
N_LARGE = 1600
#: Linear growth gives 4; quadratic gives 16. Leave room for a noisy timer.
MAX_RATIO = 9.0
#: Seconds for the large table's media columns; the linear code takes ~1 s.
BUDGET_S = 12.0
#: Seconds for 100 trial switches on the large session.
SWITCH_BUDGET_S = 2.0


def _alignment(tmp_path: Path, n: int) -> NWBAlignment:
    rows = [
        {
            "trial": i,
            "start_time": 10.0 * (i - 1),
            "stop_time": 10.0 * i,
            "video_cam-1": f"D:/vids/a_{i}.mp4",
            "video_cam-2": f"D:/vids/b_{i}.mp4",
        }
        for i in range(1, n + 1)
    ]
    out = tmp_path / f"{n}" / "alignment.nwb"
    out.parent.mkdir()
    pair_media(pd.DataFrame(rows), stream_rates={"video": 30.0}, output_path=out)
    return NWBAlignment(out)


def _time_media_columns(alignment: NWBAlignment, n: int) -> float:
    t0 = time.perf_counter()
    df = media_columns_from_alignment(alignment, list(range(1, n + 1)))
    assert len(df) == n and df["video_cam-1"].iloc[-1] == f"a_{n}.mp4"
    return time.perf_counter() - t0


def test_trials_table_media_columns_scale_linearly(tmp_path: Path) -> None:
    small = _time_media_columns(_alignment(tmp_path, N_SMALL), N_SMALL)
    large_alignment = _alignment(tmp_path, N_LARGE)
    large = _time_media_columns(large_alignment, N_LARGE)

    assert large < BUDGET_S, f"{N_LARGE} trials took {large:.1f} s to name their media"
    # Tiny tables are timer noise; only judge the growth once the small run is measurable.
    if small > 0.05:
        assert large / small < MAX_RATIO, f"{N_SMALL}->{N_LARGE} trials: {small:.2f} s -> {large:.2f} s"

    t0 = time.perf_counter()
    for trial in range(1, 101):
        large_alignment.resolve_media_path(trial, "video", "cam-1")
        large_alignment.stream_offset_for_trial(trial, "video", "cam-1")
    assert time.perf_counter() - t0 < SWITCH_BUDGET_S
