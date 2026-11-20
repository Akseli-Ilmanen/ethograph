"""Compare the MaxInterval burst detector against its R reference implementation.

Generates synthetic spike trains, runs sjemea's ``mi.find.bursts`` on them with
Rscript from the ``rcheck`` conda env, and counts exact agreements with
``ethograph.features.bursts.detect_bursts_max_interval`` (pynapple's detector).

Run from the repo root:  python tests/_test_burst_r_reference.py
"""

import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pynapple as nap

from ethograph.features.bursts import detect_bursts_max_interval

ROOT = Path(__file__).resolve().parents[1]
R_ENV = Path(r"C:\Users\aksel\anaconda3\envs\rcheck")
RSCRIPT = R_ENV / "Lib" / "R" / "bin" / "Rscript.exe"

N_PER_FAMILY = 125
DURATION_S = 300.0
MI_PARAMS = dict(
    max_isi_start=0.17,
    max_isi_burst=0.3,
    min_inter_burst_interval=0.2,
    min_burst_duration=0.01,
    min_spikes_in_burst=3,
)

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
repo <- args[1]; input <- args[2]; output <- args[3]
source(file.path(repo, "repos/sjemea/R/surprise2.R"))
source(file.path(repo, "repos/sjemea/R/maxinterval.R"))
mi.par <<- list(beg.isi = 0.17, end.isi = 0.3, min.ibi = 0.2, min.durn = 0.01, min.spikes = 3)

fmt <- function(b) {
  if (is.null(b) || length(b) == 0 || (length(b) == 1 && is.na(b))) return("")
  paste(b[, "beg"], b[, "end"], collapse = " ")
}
lines <- readLines(input)
out <- file(output, "w")
for (line in lines) {
  st <- as.numeric(strsplit(line, " ")[[1]])
  writeLines(fmt(mi.find.bursts(st)), out)
}
close(out)
"""


def bursty_train(rng, isi_mean_range, spikes_range, background_range, burst_rate_range):
    background = rng.uniform(*background_range)
    times = [np.cumsum(rng.exponential(1 / background, int(DURATION_S * background * 2)))]
    onsets = np.cumsum(rng.exponential(1 / rng.uniform(*burst_rate_range), 500))
    isi_mean = rng.uniform(*isi_mean_range)
    for onset in onsets[onsets < DURATION_S]:
        n = rng.integers(*spikes_range)
        isis = rng.gamma(4.0, isi_mean / 4.0, n - 1)
        times.append(onset + np.concatenate([[0.0], np.cumsum(isis)]))
    times = np.concatenate(times)
    return np.unique(times[times < DURATION_S])


def make_trains(rng):
    trains = []
    for _ in range(N_PER_FAMILY):
        trains.append(("bursty", bursty_train(rng, (0.005, 0.03), (3, 13), (0.2, 2.0), (0.1, 0.5))))
    for _ in range(N_PER_FAMILY):
        trains.append(("slow_bursts", bursty_train(rng, (0.05, 0.2), (4, 11), (0.1, 0.5), (0.05, 0.3))))
    for _ in range(N_PER_FAMILY):
        rate = rng.uniform(0.5, 30.0)
        t = np.cumsum(rng.exponential(1 / rate, int(DURATION_S * rate * 1.5)))
        trains.append(("poisson", t[t < DURATION_S]))
    for _ in range(N_PER_FAMILY):
        shape, rate = rng.uniform(0.3, 5.0), rng.uniform(1.0, 15.0)
        t = np.cumsum(rng.gamma(shape, 1 / (shape * rate), int(DURATION_S * rate * 1.5)))
        trains.append(("gamma", np.unique(t[t < DURATION_S])))
    return [(family, t) for family, t in trains if len(t) > 3]


def run_r(trains, workdir):
    input_path = workdir / "trains.txt"
    output_path = workdir / "r_bursts.txt"
    script_path = workdir / "reference.R"
    with open(input_path, "w") as f:
        for _, t in trains:
            f.write(" ".join("%.17g" % x for x in t) + "\n")
    script_path.write_text(R_SCRIPT)
    # Git Bash's mingw DLLs on PATH break R; give it only its own env and Windows.
    env = {
        "PATH": os.pathsep.join(
            str(p)
            for p in [
                R_ENV / "Lib" / "R" / "bin",
                R_ENV / "Library" / "bin",
                R_ENV / "Library" / "mingw-w64" / "bin",
                R_ENV,
                Path(r"C:\Windows\System32"),
                Path(r"C:\Windows"),
            ]
        ),
        "SYSTEMROOT": r"C:\Windows",
        "TEMP": str(workdir),
        "TMP": str(workdir),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
    }
    subprocess.run(
        [str(RSCRIPT), str(script_path), ROOT.as_posix(), input_path.as_posix(), output_path.as_posix()],
        check=True,
        env=env,
    )
    lines = output_path.read_text().splitlines()
    assert len(lines) == len(trains), (len(lines), len(trains))
    return [parse_bursts(line) for line in lines]


def parse_bursts(line):
    # R indices are 1-based; convert to 0-based (beg, end) pairs.
    values = [int(float(x)) - 1 for x in line.split()]
    return list(zip(values[::2], values[1::2]))


def to_index_pairs(times, bursts):
    starts = np.searchsorted(times, bursts.start)
    ends = np.searchsorted(times, bursts.end)
    assert np.array_equal(times[starts], bursts.start) and np.array_equal(times[ends], bursts.end)
    assert np.array_equal(ends - starts + 1, bursts.n_spikes.values)
    return list(zip(starts.tolist(), ends.tolist()))


def main():
    rng = np.random.default_rng(20261004)
    trains = make_trains(rng)
    print(f"{len(trains)} trains, {sum(len(t) for _, t in trains)} spikes")

    with tempfile.TemporaryDirectory() as tmp:
        r_results = run_r(trains, Path(tmp))

    counts = Counter()
    unexplained = []
    for (family, times), r_mi in zip(trains, r_results):
        counts[family] += 1
        py_mi = to_index_pairs(times, detect_bursts_max_interval(nap.Ts(t=times), **MI_PARAMS))
        if py_mi == r_mi:
            counts["mi_exact"] += 1
        else:
            unexplained.append((family, times, py_mi, r_mi))

    print("Trains per family:", {k: counts[k] for k in ["bursty", "slow_bursts", "poisson", "gamma"]})
    print(f"MaxInterval exact: {counts['mi_exact']} / {len(trains)}")
    print(f"Unexplained mismatches: {len(unexplained)}")
    for family, times, py, r in unexplained[:5]:
        only_py = sorted(set(py) - set(r))
        only_r = sorted(set(r) - set(py))
        print(f"  {family} n={len(times)}: python-only {only_py[:5]}, r-only {only_r[:5]}")
    return 0 if not unexplained else 1


if __name__ == "__main__":
    sys.exit(main())
