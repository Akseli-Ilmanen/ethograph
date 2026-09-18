from datetime import datetime
import os
from sys import meta_path

from narwhals import exclude

from ethograph import get_project_root
from ethograph.labels.intervals import load_label_mapping
from neural.utils.paths import find_session_paths, paths
import numpy as np
import pynapple as nap
import pandas as pd
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import glob
from matplotlib.backends.backend_pdf import PdfPages
from collections.abc import Iterable
import matplotlib as mpl
import matplotlib.colors as mcolors
import re
import pickle



user = "Akseli"
STYLE_PATH = get_project_root() / "configs" / "style" / "style.mplstyle"
META_TSV = paths[user]["neuron_metadata_tsv"]
JANUS_WIDE_PATH = get_project_root() / "data" / "janus_wide"
JANUS_PATH = get_project_root() / "data" / "janus"

PLOT_DIR = get_project_root() / "plots"

# 51 sessions
# Sessions are split by whose data folder they live in (user_paths.json)
AKSELI_SESSIONS: dict[str, list[str]] = {
    "Poppy": [# "20260304_01","20260305_02", # exclude for glm as less clear 
              "20260306_01", "20260307_01", # 
              "20260308_01", "20260309_01", "20260310_01", "20260311_01",
               "20260312_01", "20260313_01", "20260317_01"],
    # "Poppy": ["20260308_01", "20260313_01"],      
    "Ivy":   ["20260413_01", "20260414_01", "20260415_01", "20260416_01",
              "20260417_01", 
              "20260420_01", "20260421_01", # control
              # "20260424_01" # stimulation
            ]
}

ALICE_SESSIONS: dict[str, list[str]] = {
    "Ivy": [
        "20250306_01", "20250307_01",
        "20250308_01", "20250309_01",
        "20250503_02", "20250504_01", "20250505_01", "20250506_02",
        "20250507_02", "20250507_03", "20250508_01", "20250508_02",
        "20250509_01", "20250512_01", "20250513_01", "20250514_01",
        "20250515_01", "20250516_01", "20250519_01", "20250521_01",
        "20250522_01"
    ],
    "Freddy": [
        "20250526_01", "20250526_02", "20250527_01", "20250527_02",
        "20250528_01", "20250528_02", "20250529_01", 
        "20250530_01",
    ],
    # "Ivy": ["20250308_01", "20250307_01"], 
    # "Freddy": ["20250529_01"] # , "20250528_02"]
}





# Brain region per recording set: Akseli → NCL, Alice → AId
SESSION_REGION: dict[str, str] = {
    **{date_id: "AId" for sessions in AKSELI_SESSIONS.values() for date_id in sessions},
    **{date_id: "NCL" for sessions in ALICE_SESSIONS.values() for date_id in sessions},
}

BIRD_RELABEL: dict[str, dict[int, int]] = {
    "Freddy": {15: 4},
    "Ivy":    {},
    "Poppy":  {15:26, 4:15},
}

BIRD_NAMES = {"Freddy": "Crow #1", "Poppy": "Crow #2", "Ivy": "Crow #3"}



# NOT USED anywhere currently
# BIRD_MERGE_CONSECUTIVE_SYLLABLES: dict[str, list[set[int]]] = {
#     "Freddy": [3, 7, 8],
#     "Ivy":    [7, 8, 10],
#     "Poppy":  [7, 8, 10],
# }

SKIP_GAPS: list[tuple[int, int]] = [(1, 2), (9, 10), (10, 11)]



# Master sequence per bird/side: the maximal sequence covering all possible
# repeat counts and tail lengths.  Pass to FrankensteinPlot(master_sequence=...)
# to get one unified block where every trial is a row, missing columns are grey,
# and rows are sorted by (-n_repeats, -total_duration) per repeated label.

## REPEATS MASTER SEQUENCES
# MASTER_SEQUENCES: dict[str, dict[str, str]] = {
#     "Poppy": {
#         "right": "1-2-3-3-15-26-14-7-8-9-9-10-10-11-12-13",
#         "left":  "1-2-3-3-15-26-5-23-24-9-9-10-10-11-12-13",
#     },
#     "Ivy":    {
#         "left":  "1-2-15-5-6-7-7-8-9-9-10-10-11-12-13-16-17-17-17",
#         "right": "1-2-15-14-7-7-8-9-9-10-10-11-12-13-16-17-17-17",
#         },
#     "Freddy": {
#         "left":  "1-2-3-3-4-5-6-7-7-8-9-9-10-11-12-13",
#         "right": "1-2-3-3-4-14-7-7-8-9-9-10-11-12-13",
#         },
# }

# for LABEL_MAPPING sorting
# 21, 22, could place in teh middle for poppy, ends for now


# for compatibility keep, but try to use BIRD_LABEL_MAPPING
mapping_path = get_project_root() / "configs" / "mapping_order.txt"
LABEL_MAPPING = load_label_mapping(mapping_path) # order=ORDER


BIRD_SEQUENCES: dict[str, dict[str, str]] = {
    "Poppy": {
        "left pellet":  "1-2-3-15-26-5-23-24-9-10-11-12-13",
        "right pellet": "1-2-3-15-26-14-7-8-9-10-11-12-13",
        "zero pellets": "1-2-3-9-10",
        "order": [1, 2, 3, 15, 26, 5, 14, 23, 7, 24, 8, 9, 10, 11, 12, 13],
    },
    "Ivy":    {
        "left pellet":  "1-2-15-5-6-7-8-9-10-11-12-13-16-17",
        "right pellet": "1-2-15-14-7-8-9-10-11-12-13-16-17",
        "zero pellets": "1-2-9-10",
        "order": [1, 2, 15, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 16, 17],
    },
    "Freddy": {
        "left pellet":  "1-2-3-4-5-6-7-8-9-10-11-12-13",
        "right pellet": "1-2-3-4-14-7-8-9-10-11-12-13",
        "zero pellets": "1-2-3-9-10",
        "order": [1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13],
    },
}

# Per Bird, these transitions occured more than 50x across sessions
BIRD_TRANSITIONS: dict[str, list[tuple[int, int]]] = {
    "Poppy": [
        (15, 26), (10, 10), (1, 2),   (9, 10),  (3, 15),  (2, 3),
        (10, 11), (11, 12), (12, 13), (9, 9),   (26, 14), (26, 5),
        (3, 3),   (14, 7),  (7, 8),   (8, 9),   (5, 23),  (23, 24),
        (24, 9),  (7, 7),   (23, 23), (13, 13), (15, 15), (11, 11),
        (26, 15), (5, 9),   (26, 26), (8, 8),   (14, 9),
    ],
    "Ivy": [
        (7, 8),   (1, 2),   (9, 10),  (12, 13), (2, 15),  (11, 12),
        (8, 9),   (10, 11), (9, 9),   (16, 17), (14, 7),  (13, 16),
        (5, 6),   (15, 14), (6, 7),   (15, 5),  (10, 10), (6, 6),
        (17, 17), (13, 9),  (13, 13), (10, 16), (8, 11),  (7, 7),
        (17, 16), (8, 12),  (12, 12), (2, 14),  (15, 15), (8, 8),
        (2, 5),   (8, 7),   (8, 14),  (16, 16), (1, 1),
    ],
    "Freddy": [
        (1, 2),   (2, 3),   (9, 10),  (3, 4),   (8, 9),   (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 7),  (5, 6),   (4, 5),
        (4, 14),  (6, 7),   (3, 3),   (4, 4),   (7, 7),   (9, 9),
        (6, 6),
    ],
}


MATCHING_GROUPS:  dict[str, list[tuple[int, ...]]]  = {
    "Poppy":  [(5, 14), (23, 7), (24, 8)],
    "Ivy":    [],
    "Freddy": [],
}


NEURON_COLORS = ["#3CB371", "#E8388A", "#4472C4", "#FF8C00"]
SIDE_COLORS = {"left pellet": "blue", "right pellet": "red", "zero pellets": "green"}
SIDES  = ["left pellet", "right pellet"] # "zero pellets"


BIRD_LABEL_MAPPING: dict[str, dict[int, dict]] = {
    "Poppy": {
        1:  {"name": "pullOutStick",        "color": np.array([1.        , 0.4       , 0.69803922]), "letter": "A"},
        2:  {"name": "diagonalToBox",       "color": np.array([0.4       , 0.61960784, 1.        ]), "letter": "B"},
        3:  {"name": "toss",                "color": np.array([0.6       , 0.2       , 1.        ]), "letter": "C"},
        15: {"name": "nodding",             "color": np.array([0.02745098, 0.02745098, 0.84313725]), "letter": "D"},
        26: {"name": "postNod",             "color": np.array([0.95,     0.35,      0.05]),          "letter": "E"},
        9:  {"name": "stickToDisp",         "color": np.array([1.        , 1.        , 0.        ]), "letter": "H"},
        10: {"name": "stickInDisp",         "color": np.array([0.        , 0.8       , 0.8       ]), "letter": "I"},
        11: {"name": "rightToPellet",       "color": np.array([0.50196078, 0.50196078, 0.        ]), "letter": "J"},
        12: {"name": "snapPellet",          "color": np.array([1.        , 0.        , 1.        ]), "letter": "K"},
        13: {"name": "eat",                 "color": np.array([1.        , 0.64705882, 0.        ]), "letter": "L"},
        5:  {"name": "reachToWall",         "color": np.array([0.        , 0.50196078, 1.        ]), "letter": "L1"},
        23: {"name": "pullOutLeft",         "color": np.array([0.        , 0.6       , 0.        ]), "letter": "L2"},
        24: {"name": "swoopOutLeft",        "color": np.array([0.69803922, 0.43529412, 0.17254902]), "letter": "L3"},
        14: {"name": "curvedRight",         "color": np.array([0.        , 0.50196078, 1.        ]), "letter": "R1"},
        7:  {"name": "pullOutAlongWall",    "color": np.array([0.        , 0.6       , 0.        ]), "letter": "R2"},
        8:  {"name": "swoopOut",            "color": np.array([0.69803922, 0.43529412, 0.17254902]), "letter": "R3"},
    },
    "Ivy": {
        1:  {"name": "pullOutStick",        "color": np.array([1.        , 0.4       , 0.69803922]), "letter": "A"},
        2:  {"name": "diagonalToBox",       "color": np.array([0.4       , 0.61960784, 1.        ]), "letter": "B"},
        15: {"name": "nodding",             "color": np.array([0.02745098, 0.02745098, 0.84313725]), "letter": "D"},
        7:  {"name": "pullOutAlongWall",    "color": np.array([0.        , 0.6       , 0.        ]), "letter": "F"},
        8:  {"name": "swoopOut",            "color": np.array([0.69803922, 0.43529412, 0.17254902]), "letter": "G"},
        9:  {"name": "stickToDisp",         "color": np.array([1.        , 1.        , 0.        ]), "letter": "H"},
        10: {"name": "stickInDisp",         "color": np.array([0.        , 0.8       , 0.8       ]), "letter": "I"},
        11: {"name": "rightToPellet",       "color": np.array([0.50196078, 0.50196078, 0.        ]), "letter": "J"},
        12: {"name": "snapPellet",          "color": np.array([1.        , 0.        , 1.        ]), "letter": "K"},
        13: {"name": "eat",                 "color": np.array([1.        , 0.64705882, 0.        ]), "letter": "L"},
        16: {"name": "beakToDisp",          "color": np.array([0.50196078, 0.        , 1.        ]), "letter": "M"},
        17: {"name": "stickInDispTwo",      "color": np.array([1.0       , 0.843     , 0.0       ]), "letter": "N"},
        5:  {"name": "reachToWall",         "color": np.array([0.4       , 1.        , 0.4       ]), "letter": "L1"},
        6:  {"name": "right",               "color": np.array([1.        , 0.6       , 0.4       ]), "letter": "L2"},
        14: {"name": "curvedRight",         "color": np.array([0.        , 0.50196078, 1.        ]), "letter": "R1"},
    },
    "Freddy": {
        1:  {"name": "pullOutStick",        "color": np.array([1.        , 0.4       , 0.69803922]), "letter": "A"},
        2:  {"name": "diagonalToBox",       "color": np.array([0.4       , 0.61960784, 1.        ]), "letter": "B"},
        3:  {"name": "toss",                "color": np.array([0.6       , 0.2       , 1.        ]), "letter": "C"},
        4:  {"name": "swoopPreBox",         "color": np.array([1.        , 0.2       , 0.2       ]), "letter": "D"},
        7:  {"name": "pullOutAlongWall",    "color": np.array([0.        , 0.6       , 0.        ]), "letter": "F"},
        8:  {"name": "swoopOut",            "color": np.array([0.69803922, 0.43529412, 0.17254902]), "letter": "G"},
        9:  {"name": "stickToDisp",         "color": np.array([1.        , 1.        , 0.        ]), "letter": "H"},
        10: {"name": "stickInDisp",         "color": np.array([0.        , 0.8       , 0.8       ]), "letter": "I"},
        11: {"name": "rightToPellet",       "color": np.array([0.50196078, 0.50196078, 0.        ]), "letter": "J"},
        12: {"name": "snapPellet",          "color": np.array([1.        , 0.        , 1.        ]), "letter": "K"},
        13: {"name": "eat",                 "color": np.array([1.        , 0.64705882, 0.        ]), "letter": "L"},
        5:  {"name": "reachToWall",         "color": np.array([0.4       , 1.        , 0.4       ]), "letter": "L1"},
        6:  {"name": "right",               "color": np.array([1.        , 0.6       , 0.4       ]), "letter": "L2"},
        14: {"name": "curvedRight",         "color": np.array([0.        , 0.50196078, 1.        ]), "letter": "R1"},
    },
}
BIRD_LETTER_DICT = {bird: {lab: entry["letter"] for lab, entry in labels.items()} for bird, labels in BIRD_LABEL_MAPPING.items()}

OVERRIDE_LABEL_COLORS_BY_CONDITION = True  # when True: color_l=red, color_r=blue, color=grey
LEFT_COLOR  = np.array([1.0, 0.0, 0.0])
RIGHT_COLOR = np.array([0.0, 0.0, 1.0])
NEUTRAL_GREY  = np.array([0.5, 0.5, 0.5])


# for t-SNE change to 0.6, 1.4
DARK_FACTOR   = 0.9
BRIGHT_FACTOR = 1.1

def _scale(color, factor):
    r, g, b = mcolors.to_rgb(color)
    if factor <= 1.0:
        return (r * factor, g * factor, b * factor)
    t = factor - 1.0
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)



new_mapping = {}
for bird, labels in BIRD_LABEL_MAPPING.items():
    new_mapping[bird] = {}

    for lab, entry in labels.items():
        base_color = np.array(entry["color"])

        if OVERRIDE_LABEL_COLORS_BY_CONDITION:
            color_l = LEFT_COLOR
            color_r = RIGHT_COLOR
            if BIRD_LABEL_MAPPING[bird][lab]["letter"] in ("L1", "L2", "L3"):
                base_color_out = LEFT_COLOR
            elif BIRD_LABEL_MAPPING[bird][lab]["letter"] in ("R1", "R2", "R3"):
                base_color_out = RIGHT_COLOR
            else:
                base_color_out = NEUTRAL_GREY
        else:
            color_l = _scale(base_color, BRIGHT_FACTOR)
            color_r = _scale(base_color, DARK_FACTOR)
            base_color_out = tuple(base_color)

        new_mapping[bird][lab] = {
            **entry,
            "color":   base_color_out,
            "color_orig": tuple(base_color),
            "color_orig_l": _scale(base_color, BRIGHT_FACTOR),
            "color_orig_r": _scale(base_color, DARK_FACTOR),
            "color_l": color_l,
            "color_r": color_r,
        }

BIRD_LABEL_MAPPING = new_mapping



def parse_sequence(seq: str) -> set[int]:
    return {int(s) for s in seq.split("-")}


def derive_canonical_labels() -> dict[tuple[str, str], set[int]]:
    return {
        (bird, condition): set(parse_sequence(sequence))
        for bird, variants in BIRD_SEQUENCES.items()
        for condition, sequence in variants.items()
        if condition != "order"
    }

def drop_non_canonical(df):
    canonical = derive_canonical_labels()
    keep = np.ones(len(df), dtype=bool)
    for (bird, condition), labels in canonical.items():
        in_group = (df.individual == bird) & (df.condition == condition)
        keep &= ~(in_group & ~df.labels.isin(labels))
    return df[keep].reset_index(drop=True)




# ── Signal / processing parameters ───────────────────────────────────────────
SR = 30_000             # ephys sample rate (Hz)
FPS = 200                # video frame rate (Hz)


# for prep pynapple


# prep pynapple derives position_stickTip, position_pellet, angles_stickTip, etc...
# pairwise distances features (e.g. pellet_stickTip_dist) -> create those as needed, e.g. since we maybe do pre-procesing (e.g. nan filling) before deriving these
SIGNAL_NAMES = ("position", "velocity", "speed", "acceleration", "angles", "angle_rgb",
                "aux_acceleration", "pellet_stickClosest_angular_similarity") # create via prep_pynapple.py, SKIP_EPHYS = False, uncomment after 1x) 


SIGNAL_NAMES = ("velocity", "speed", "pellet_stickClosest_dist", "pellet_stickClosestMedian_dist") 


SIGNAL_NAMES = ("position", "velocity", "speed", "angles", "angle_rgb",
                "position_stickTip", "velocity_stickTip", "speed_stickTip", "angles_stickTip", 
                "position_pellet", "velocity_pellet", "speed_pellet", "angles_pellet",
                "acceleration", "aux_acceleration",
                "angle_rgb", "angle_rgb_stickTip",  # visualization
                "pellet_stickClosest_dist", "pellet_stickClosestMedian_dist" # pellet
                )

# for prep_pynapple don't add the _stickTip suffix



# "pellet_beakTip_dist", "pellet_stickTip_dist" # don't include in Fig1, GLM


DT = 0.005 # Stride of spike count so aligned with behaviour (1/fps)
BIN_SIZE = 0.05 # I compared 0.05 and 0.1, and some harp modulation (e.g. valley in two-peak PETH) was lost at 0.1, keep 0.05!
assert math.isclose(round(BIN_SIZE / DT), BIN_SIZE / DT, rel_tol=1e-9)

SESSION_DURATION = 36000.0 
SESSION_ID_STRIDE = 10_000


# DATA PRE-PROCESING
# see also decisions in pipeline.iypnb, e.g. clipping

# Empiricially determined, for pellet, stickTip, beaktip, take np.percentile(diff(abs(position)), 99)
DERIVATIVE_THRESHOLDS = {
    "x": 0.335020,
    "y": 0.256299,
    "z": 0.590033
}


# ── IFR smoothing (gaussian_filter1d) ─────────────────────────────────────────
# sigma controls smoothing width in bins (1 bin = DT s = 1 ms).
# radius is derived automatically so the Gaussian tail is never clipped.
IFR_SMOOTH_SIGMA  = 20
IFR_SMOOTH_RADIUS = max(30, int(4 * IFR_SMOOTH_SIGMA + 0.5))

# ── Pipeline flags ────────────────────────────────────────────────────────────
SKIP_EPHYS    = True
SKIP_FEATURES = False


# VISUALIZATION PARAMETERS

# Heatmap plot
SPEED_VMAX = 95 # (equivalent to trimming to 95% percentile) to avoid outliers dominating colormap; set to None to disable



# Janus panel heights (inches; figure height = sum of all rows)
JANUS_SPEED_LINE_HEIGHT    = 3   # joint speed line at top
JANUS_COLOR_STRIP_HEIGHT   = 0.5   # syllable letter bar (per condition)
JANUS_SPEED_HEATMAP_HEIGHT = 3   # speed heatmap (per condition)
JANUS_SIGNAL_HEATMAP_HEIGHT =3  # angle-RGB heatmap (per condition)
RASTER_ROW_HEIGHT          = 0.01  # per trial row in the raster
JANUS_PSTH_HEIGHT          = 2   # PSTH band above each unit's raster block, inches

# Janus count-trace z-score display range
JANUS_COUNT_ZCLIP = 5.0

# Janus raster paging
NEURONS_PER_PDF = 10

# Minimum trials (summed across conditions) for a unit to get a PETH/raster block
MIN_PETH_TRIALS = 5



# ── frank.py analysis controls ───────────────────────────────────────────────
# None = all, 0 = first, int = specific cluster id

# Defaults (leave uncommented)
selected_bird = None;  selected_session_id = None; selected_clu = None


# selected_bird = "Freddy"; # selected_session_id = "20250528_01"; selected_clu = None #
# selected_bird = "Ivy"; selected_session_id = "20260414_01"; # selected_clu = 154
# selected_bird = "Ivy";  selected_session_id = "20260413_01"; selected_clu = 154


EXCLUDED_SESSIONS    =  []
GAP_PRE_FIRST_SYLLABLE = 1.5  # will only be half since we half pre gap
GAP_POST_LAST_SYLLABLE = 0.75  # same
AVERAGE              = "median"
# for wide plot (for figure 2 showing baselien firing, set to 2, 2, not 1.5, 0.75)



# for multi neuron frankenstein
SELECT_NEAR_MEDIAN = 15  # rows per column: trials closest to median syllable duration

# ── Derived ───────────────────────────────────────────────────────────────────
def _filter_bird_dict(d):
    if selected_bird is None:
        return d
    return {k: v for k, v in d.items() if k == selected_bird}

sessions = find_session_paths(_filter_bird_dict(AKSELI_SESSIONS), skip_ephys=SKIP_EPHYS, user="Akseli")

sessions2 = find_session_paths(_filter_bird_dict(ALICE_SESSIONS), skip_ephys=SKIP_EPHYS, user="Alice")
sessions += sessions2




# Make sure sessions not double specified
ids = [s.date_id for s in sessions]
dupes = {k: v for k, v in Counter(ids).items() if v > 1}
assert not dupes, f"Duplicate session.date_id from find_session_paths: {dupes}"

 

if selected_bird is None:
    sessions_filtered = [s for s in sessions if s.date_id not in EXCLUDED_SESSIONS]
else:
    sessions_filtered = [s for s in sessions if s.bird == selected_bird and s.date_id not in EXCLUDED_SESSIONS]
    
if selected_session_id is not None:
    sessions_filtered = [s for s in sessions_filtered if s.date_id == selected_session_id]
    


def _decode(s):
    if not isinstance(s, str) or not s:
        return np.array([], dtype=np.int64)
    return np.array(s.split("–"), dtype=np.int64)



 
def load_session(session, offset: float, session_idx: int, exclude_conditions: bool = False, drive: bool = False, exclude_all_two_pellets: bool = False, signal_names=None):
    py_path = session.nc_path.parent / "pynapple"
    if signal_names is None:
        signal_names = SIGNAL_NAMES
    

    if drive:
        tsv_drive_folder = Path(paths[user]["tsv_drive_folder"])

        backup_folders = [
            p for p in tsv_drive_folder.iterdir()
            if p.is_dir() and "backup" in p.name.lower()
        ]

        # get most recently created folder
        latest_backup_folder = max(
            backup_folders,
            key=lambda p: p.stat().st_ctime
        )
        label_dir = Path(r"G:\My Drive\Crow lab\data\Akseli\backup_20260818") # latest_backup_folder
        suffix = f"_{session.date_id}_{session.bird}"
    else:
        label_dir = session.nc_path.parent
        suffix = ""

    df = pd.read_csv(label_dir / f"Trial_data_labels{suffix}.tsv", sep="\t")
    df.sort_values(["onset_global"], inplace=True) 
    
    
    assert df.onset_global.max() <= SESSION_DURATION, f"For {session.date_id}: Increase SESSION_DURATION to at least {df.onset_global.max()}s"
    df["trial_onset"] += offset
    
    if "trial_offset" in df.columns:
        df["trial_offset"]  += offset
    
    df["onset_global"] += offset
    df["offset_global"] += offset


    df["session_idx"] = session_idx

    for old, new in BIRD_RELABEL[session.bird].items():
        df["labels"] = df["labels"].replace(old, new)
        df["sequence"] = df["sequence"].str.replace(rf"\b{old}\b", str(new), regex=True)

    if session.bird == "Ivy":
        df = remove_labels(df, [4]) # Label 4 is an inconsistent syllable for Ivy, exclude from all analyses

    # ADD BACK
    df = correct_labels(df, offset=offset)
    df = add_conditions(df, exclude_all_two_pellets)
    df["region"] = SESSION_REGION.get(session.date_id, "unknown")

    if exclude_conditions:
        # Removes roughly 100 trials, remaining ~6000 (poscat 5, 6 sessions excluded)
        df = df[~df.condition.isin(["two+ sep. pellets", "other"])]
    

    df["date_id"] = session.date_id

    signals = {
        name: shift_tsd(nap.load_file(py_path / f"{name}.npz"), offset)
        for name in signal_names
    }
    tsgroup = nap.load_file(py_path / "units.npz")

    assert df.session.nunique() == 1
    tsgroup = shift_and_relabel_units(
        tsgroup, offset, df.session.iloc[0],
        bird=session.bird,
        brain_region=SESSION_REGION.get(session.date_id, "unknown"),
    )

    tsgroup = _set_session_trial_metadata(tsgroup, df)

    return df, tsgroup, signals


def load_sessions(date_ids=None, exclude_conditions: bool = False, drive: bool = False,
                  exclude_all_two_pellets: bool = False, signal_names=None):
    """Load and merge several sessions (the canonical multi-session loading loop).

    date_ids: iterable of session date_ids to load, in sessions_filtered order;
    None loads every session in sessions_filtered. Each session is shifted by
    SESSION_DURATION, so onset_global / unit spike times stay disjoint across
    sessions. Pass signal_names=() to skip loading behavior signals entirely.

    Returns (df, tsgroup, signals): concatenated label df (with a `date_id`
    column), merged nap.TsGroup (date-prefixed unit ids, see full_unit_id),
    and merged signals dict.
    """
    if date_ids is None:
        selected = sessions_filtered
    else:
        wanted = set(date_ids)
        selected = [s for s in sessions_filtered if s.date_id in wanted]
        missing = wanted - {s.date_id for s in selected}
        assert not missing, f"date_ids not in sessions_filtered: {sorted(missing)}"

    all_dfs, all_units, all_signals = [], [], defaultdict(list)
    offset = 0.0
    for session_idx, session in enumerate(selected):
        session_df, session_units, session_signals = load_session(
            session, offset, session_idx,
            exclude_conditions=exclude_conditions, drive=drive,
            exclude_all_two_pellets=exclude_all_two_pellets, signal_names=signal_names,
        )
        all_dfs.append(session_df)
        all_units.append(session_units)
        for name, tsd in session_signals.items():
            all_signals[name].append(tsd)
        offset += SESSION_DURATION

    df = pd.concat(all_dfs, ignore_index=True)
    tsgroup = all_units[0] if len(all_units) == 1 else merge_units(all_units)
    signals = merge_signals(all_signals)
    return df, tsgroup, signals


def full_unit_id(date_id: str, clu: int) -> int:
    """Merged unit id for a session-local cluster id (inverse of `original_id`).

    Ids >= SESSION_ID_STRIDE are already date-prefixed and pass through, so
    both `136` and `20260308010136` are accepted.
    """
    clu = int(clu)
    if clu >= SESSION_ID_STRIDE:
        return clu
    return _date_prefix(date_id) * SESSION_ID_STRIDE + clu




def _filter_units(count_by_side: dict, exclude: set) -> dict:
    exclude = set(exclude)
    return {
        s: {u: ca[u] for u in ca.keys() if u not in exclude}
        for s, ca in count_by_side.items()
    }
def load_janus(fname: str, janus_path: Path = None, exclude: list = None) -> dict:
    if exclude is None:
        meta_df = pd.read_csv(META_TSV, sep="\t")
        exclude = list(meta_df.id[~meta_df.kept_all])

    if janus_path is None:
        janus_path = JANUS_WIDE_PATH

    with open(Path(janus_path) / fname, "rb") as f:
        data = pickle.load(f)
    



    bird = fname.split("_")[0]
    count_by_side = {s: data["result"][1][s] for s in SIDES}
    peth_by_side = _filter_units({s: data["result"][0][s] for s in SIDES}, exclude=exclude)
    count_by_side = _filter_units(count_by_side, exclude=exclude)
    data["result"] = (peth_by_side, count_by_side, *data["result"][2:])
    return data, bird, data["bin_size"]



def color_strip_panel(ax: plt.Axes, syl_intervals, bird: str, color: str) -> None:
    letter_dict = BIRD_LETTER_DICT[bird]
    for start, end, label in zip(syl_intervals.start, syl_intervals.end, syl_intervals.label):
        width = end - start
        ax.add_patch(mpatches.Rectangle((start, 0), width, 1, facecolor=color, edgecolor="black", linewidth=1.0))
        ax.text(start + width / 2, 0.5, letter_dict.get(int(label), str(int(label))), ha="center", va="center", fontweight="bold")
    ax.axis("off")


def remove_labels(df: pd.DataFrame, labels_to_remove: list[int]) -> pd.DataFrame:
    to_remove = set(labels_to_remove)
    return df[~df["labels"].isin(to_remove)].reset_index(drop=True)



def correct_labels(
    df: pd.DataFrame,
    fps: int = 200,
    sr: int = 30000,
    eps: float = 1e-4,
    offset: float = 0.0,
) -> pd.DataFrame:
    """Snap label boundaries to digital-pulse times, correcting irregular sampling.

    Digital pulses are occasionally off by one sample. Each label's onset/offset
    (originally in video-frame seconds) is rounded to a pulse index and remapped
    to that pulse's sample time. A small gap `eps` is shaved off every offset so
    adjacent intervals don't touch (pynapple requirement), clamped to half the
    label duration so durations stay positive. Labels whose onset and offset
    round to the same pulse index are dropped (occurs rarely).

    Point events (`event_type == "point"`, e.g. pellet-pickup markers) have no
    offset: only their onset is snapped, and offset_global/offset_s/duration
    stay NaN. They are never dropped by the same-pulse rule.
    """
    _assert_constant_trial_onset(df)
    trial_col = "session_trial"
    trial_pulses = _build_trial_pulse_dict(df, trial_col)

    is_point = _point_mask(df)
    assert df.loc[~is_point, "offset_s"].notna().all(), "state labels must have an offset_s"

    onset_s = df["onset_s"].to_numpy()
    onset_idxs = np.round(onset_s * fps).astype(int)
    offset_idxs = np.round(np.where(is_point, onset_s, df["offset_s"].to_numpy()) * fps).astype(int)

    onset_global = np.full(len(df), np.nan)
    offset_global = np.full(len(df), np.nan)


    for session_trial, row_idx in df.groupby(trial_col, sort=False).indices.items():

        pulses = trial_pulses[session_trial]
        trial_onset_idxs = onset_idxs[row_idx]
        trial_offset_idxs = offset_idxs[row_idx]

        _check_in_bounds(df, row_idx, trial_onset_idxs, trial_offset_idxs, pulses, session_trial, fps)

        keep = (trial_onset_idxs != trial_offset_idxs) | is_point[row_idx]
        kept_rows = row_idx[keep]

        onset_global[kept_rows] = pulses[trial_onset_idxs[keep]] / sr + offset
        offset_global[kept_rows] = pulses[trial_offset_idxs[keep]] / sr + offset

    valid = ~np.isnan(onset_global)
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"correct_labels: dropped {n_dropped} labels (onset/offset rounded to same pulse)")
        print()

    raw_onset = onset_global[valid]
    raw_offset = offset_global[valid]
    point = is_point[valid]

    assert (raw_offset >= raw_onset).all(), "offsets must be after onsets after correction"
    gap = np.minimum(eps, (raw_offset - raw_onset) / 2)
    corrected_offset = np.where(point, np.nan, raw_offset - gap)

    df = df.loc[valid].assign(
        onset_global=raw_onset,
        offset_global=corrected_offset,
        duration=corrected_offset - raw_onset,
    )
    df["onset_s"] = df["onset_global"] - df["trial_onset"]
    df["offset_s"] = df["offset_global"] - df["trial_onset"]

    assert (df.loc[~point, "duration"] > 0).all(), "non-positive durations after correction"
    return df.reset_index(drop=True)







def add_conditions(
    df: pd.DataFrame,
    exclude_two_pellets: bool = False,
) -> pd.DataFrame:
    valid_pellets = {1} if exclude_two_pellets else {1, 2}
    
    def classify_condition(row: pd.Series) -> str:
        poscat = int(row["poscat"])
        num_pellets = int(row["num_pellets"])
        
        if num_pellets == 0 and poscat == 0:
            return "zero pellets"
        if poscat in {13, 4} and num_pellets >= 2:
            return "two+ sep. pellets"
        if poscat == 1 and num_pellets in valid_pellets:
            return "left pellet"
        if poscat == 3 and num_pellets in valid_pellets:
            return "right pellet"
        if poscat in {4, 5, 6}: # 5, 6, control conditions 
            return "other"
        
        raise ValueError(
            f"Unexpected poscat={poscat}, num_pellets={num_pellets} "
            f"in session_trial={row['session_trial']}"
        )
    
    return df.assign(condition=df.apply(classify_condition, axis=1))

 



# HELPERS
def _assert_constant_trial_onset(df: pd.DataFrame) -> None:
    max_unique = df.groupby("session_trial")["trial_onset"].nunique().max()
    if max_unique != 1:
        raise ValueError(
            f"trial_onset must be constant within each session_trial, "
            f"found up to {max_unique} unique values per trial"
        )


def _point_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows that are point events (an onset but no offset)."""
    if "event_type" in df.columns:
        return (df["event_type"] == "point").to_numpy()
    return df["offset_s"].isna().to_numpy()


def _build_trial_pulse_dict(df: pd.DataFrame, trial_col: str) -> dict[object, np.ndarray]:
    filled = df["pulse_onsets"].replace("", np.nan).ffill()
    return {
        trial: _decode(s)
        for trial, s in filled.groupby(df[trial_col], sort=False).first().items()
    }

def _check_in_bounds(
    df: pd.DataFrame,
    row_idx: np.ndarray,
    onset_idxs: np.ndarray,
    offset_idxs: np.ndarray,
    pulses: np.ndarray,
    session_trial: object,
    fps: int,
) -> None:
    n_pulses = len(pulses)
    bad = (onset_idxs >= n_pulses) | (offset_idxs >= n_pulses) | (onset_idxs < 0) | (offset_idxs < 0)
    if not bad.any():
        return

    bad_rows = df.iloc[row_idx[bad]]
    print(f"\n=== Trial {session_trial}: {bad.sum()} label(s) out of bounds ===")
    print(f"n_pulses={n_pulses} → trial duration ≈ {(n_pulses - 1) / fps:.4f} s. Exclude trial.")
    cols = ["labels", "onset_s", "offset_s"]
    if {"trial_onset", "trial_offset"}.issubset(df.columns):
        bad_rows = bad_rows.assign(trial_span_s=bad_rows["trial_offset"] - bad_rows["trial_onset"])
        cols += ["trial_onset", "trial_offset", "trial_span_s"]
    print(bad_rows[cols].to_string())
    raise ValueError(f"Trial {session_trial} has labels beyond recorded pulses")

def _decode(s):
    if not isinstance(s, str) or not s:
        return np.array([], dtype=np.int64)
    return np.array(s.split("–"), dtype=np.int64)




def get_trial_periods(
    X: pd.DataFrame | nap.IntervalSet,
    gap: float = 1.0,
    point_events: bool = False,
) -> nap.IntervalSet:
    if isinstance(X, nap.IntervalSet):
        df = X.as_dataframe()
        start_col, end_col = "start", "end"
    elif isinstance(X, pd.DataFrame):
        df = X
        start_col, end_col = "onset_global", "offset_global"
    else:
        raise TypeError(f"Expected DataFrame or IntervalSet, got {type(X).__name__}")

    grouped = df.groupby("session_trial",sort=False)

    if point_events:
        trials = (
            pd.DataFrame({
                "start": grouped[start_col].min() - gap,
                "end": grouped[start_col].max() + gap,
            })
            .sort_values("start")
            .reset_index()
        )
    else:
        trials = (
            pd.DataFrame({
                "start": grouped[start_col].min() - gap,
                "end": grouped[end_col].max() + gap,
            })
            .sort_values("start")
            .reset_index()
        )


    optional_cols = ["session", "individual", "region", "condition", "final_pellet"]
    for col in optional_cols:
        if col in df.columns:
            trials[col] = grouped[col].first().values

    meta_cols = ["session_trial"] + [c for c in optional_cols if c in trials.columns]
    return nap.IntervalSet(
        start=trials["start"].values,
        end=trials["end"].values,
        metadata=trials[meta_cols],
    )

def _no_overlap_report(no_overlap: dict, trial_periods: nap.IntervalSet, session: str) -> str:
    lines = [
        f"{session}: {len(no_overlap)} unit(s) span no complete trial | "
        f"trials {trial_periods.start[0]:.1f}-{trial_periods.end[-1]:.1f}s "
        f"(last trial starts {trial_periods.start[-1]:.1f}s, n={len(trial_periods)})"
    ]
    for clu, (t0, t1) in no_overlap.items():
        lines.append(f"  unit {clu}: " + ("no spikes" if t0 is None else f"spikes {t0:.1f}-{t1:.1f}s"))
    return "\n".join(lines)


def _set_session_trial_metadata(tsgroup: nap.TsGroup, df: pd.DataFrame) -> nap.TsGroup:
    """Tag each unit with the span of trials it was recorded across.

    A unit counts as present for the trials lying entirely within its first..last
    spike, so both indices are searched against the matching edge of the trial
    windows. A unit that appears only after the final trial started, that drops out
    before the first trial ended, or that has no spikes at all covers no such trial
    and gets NaN bounds with zero counts -- legitimate for a late-drifting unit, but
    if it holds for EVERY unit the labels .tsv and units.npz disagree on time.
    """
    trial_periods = get_trial_periods(df)
    trial_cond = df.groupby("session_trial")["condition"].first()

    first, last, first_t, last_t = {}, {}, {}, {}
    n_left, n_right, n_zero, n_trials = {}, {}, {}, {}
    no_overlap = {}
    for clu in tsgroup.keys():
        t0, t1 = tsgroup[clu].start, tsgroup[clu].end
        if t0 is None:
            i_first, i_last = 0, -1
        else:
            i_first = np.searchsorted(trial_periods.start, float(t0), "left")
            i_last  = np.searchsorted(trial_periods.end,   float(t1), "right") - 1

        if i_first > i_last:
            no_overlap[clu] = (t0, t1)
            first[clu] = last[clu] = None
            first_t[clu] = last_t[clu] = np.nan
            n_left[clu] = n_right[clu] = n_zero[clu] = n_trials[clu] = 0
            continue

        first[clu]   = trial_periods.session_trial.iloc[i_first]
        last[clu]    = trial_periods.session_trial.iloc[i_last]
        first_t[clu] = float(trial_periods.start[i_first])
        last_t[clu]  = float(trial_periods.end[i_last])

        unit_trials = trial_periods.session_trial.iloc[i_first : i_last + 1]
        counts = trial_cond.reindex(unit_trials).value_counts()
        n_left[clu]   = int(counts.get("left pellet",   0))
        n_right[clu]  = int(counts.get("right pellet",  0))
        n_zero[clu]   = int(counts.get("zero pellets",  0))
        n_trials[clu] = int(counts.sum())

    if no_overlap:
        report = _no_overlap_report(no_overlap, trial_periods, str(df.session.iloc[0]))
        if len(no_overlap) == len(tsgroup):
            raise ValueError(
                f"{report}\nNo unit overlaps any trial: the labels .tsv and units.npz "
                f"are on different timebases (with drive=True the .tsv comes from the "
                f"latest backup folder, which may predate/postdate this recording)."
            )
        print(report)

    tsgroup.set_info(
        first_trial=pd.Series(first),
        last_trial=pd.Series(last),
        first_trial_t=pd.Series(first_t),
        last_trial_t=pd.Series(last_t),
        n_trials=pd.Series(n_trials),
        n_left=pd.Series(n_left),
        n_right=pd.Series(n_right),
        n_zero=pd.Series(n_zero),
    )
    return tsgroup



def merge_units(all_units: list[nap.TsGroup]) -> nap.TsGroup:
    """Merge per-session TsGroups into one, aligning metadata columns.

    Sessions' units.npz files may carry different metadata columns; pynapple's
    merge requires an identical, identically-ordered column set across groups.
    Reconstruct each TsGroup with the union of columns in a fixed order (missing
    → NaN; "rate" is dropped and recomputed by the constructor).
    """
    meta_cols = sorted(set().union(*(u.metadata.columns for u in all_units)) - {"rate"})
    aligned = [
        nap.TsGroup(
            {k: u[k] for k in u.keys()},
            time_support=u.time_support,
            metadata=u.metadata.drop(columns=["rate"], errors="ignore").reindex(columns=meta_cols),
        )
        for u in all_units
    ]
    return aligned[0].merge(*aligned[1:], reset_time_support=True)


def merge_signals(signal_lists: dict[str, list]) -> dict:
    merged = {}
    for name, parts in signal_lists.items():
        times = np.concatenate([p.times() for p in parts])
        order = np.argsort(times)
        if isinstance(parts[0], nap.TsdFrame):
            values = np.vstack([p.values for p in parts])[order]
            merged[name] = nap.TsdFrame(t=times[order], d=values, columns=parts[0].columns)
        else:
            values = np.concatenate([p.values for p in parts])[order]
            merged[name] = nap.Tsd(t=times[order], d=values)
    return merged


def _edge_nan_mask(sub: np.ndarray) -> np.ndarray:
    """Per column of one trial: True on the leading and trailing runs of NaN.

    A column with no finite sample at all is edge NaN end to end.

        col:   nan nan  1.2  nan  3.4  nan nan
        mask:    T   T    F    F    F    T   T
    """
    valid = ~np.isnan(sub)
    first = np.argmax(valid, axis=0)
    last = len(sub) - 1 - np.argmax(valid[::-1], axis=0)
    rows = np.arange(len(sub))[:, None]
    mask = (rows < first) | (rows > last)
    mask[:, ~valid.any(axis=0)] = True
    return mask


def _like(sig, d: np.ndarray, t: np.ndarray | None = None):
    t = sig.t if t is None else t
    if isinstance(sig, nap.TsdFrame):
        return nap.TsdFrame(t=t, d=d, columns=sig.columns, time_support=sig.time_support)
    return nap.Tsd(t=t, d=d, time_support=sig.time_support)


def fill_nans(
    signals: dict,
    trial_periods: nap.IntervalSet,
    substring: str = "sticktip",
    fill_values: dict | None = None,
    default: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Fill the leading/trailing NaN run of each trial with a constant, per column.

    Tracking for a keypoint can drop out -- the stick isn't in frame at the start and
    end of a trial -- and a single NaN sample poisons the bin it lands in
    (`bin_average` is a plain mean). At a trial EDGE the keypoint hasn't entered yet
    or has already left, so there is nothing to interpolate from on one side and only
    a constant will do -- which is only meaningful if you know what the signal is.
    Hence the `substring` scope: only signals whose name contains it (case-insensitive)
    or that have an explicit `fill_values` entry are touched.

    A NaN gap in the MIDDLE of a trial is a real tracking failure with data on both
    sides; this function leaves it alone. Use `drop_nan_bins` to take those bins out
    of the design.

    `fill_values` overrides `default` per signal, either as a scalar or as one value
    per column, e.g. the dispenser coordinates for the stick tip position::

        signals = S.fill_nans(signals, trial_periods, fill_values={
            "position_stickTip": [-10.23, -5.907, -1.395],   # [x, y, z]
        })

    Samples outside `trial_periods` are never touched, and non-float signals are
    skipped. Returns a new dict; the input signals are not modified.
    """
    fill_values = fill_values or {}
    unknown = set(fill_values) - set(signals)
    assert not unknown, f"fill_values names not in signals: {sorted(unknown)}"

    edge_names = {n for n in signals if substring.lower() in n.lower()} | set(fill_values)
    out, report = dict(signals), []
    for name, sig in signals.items():
        d = np.asarray(sig.values)
        if not np.issubdtype(d.dtype, np.floating):
            continue
        flat = d.astype(float).reshape(len(sig), -1).copy()
        if not np.isnan(flat).any():
            continue

        fill_edges = name in edge_names
        value = np.asarray(fill_values.get(name, default), dtype=float).ravel()
        if value.size == 1:
            value = np.repeat(value, flat.shape[1])
        assert not fill_edges or value.size == flat.shape[1], (
            f"{name}: fill value has {value.size} entries but the signal has "
            f"{flat.shape[1]} column(s)")

        if not fill_edges:
            continue

        trial_of_sample = trial_periods.in_interval(sig)
        n_filled, trials_hit = 0, set()
        for trial in np.unique(trial_of_sample[~np.isnan(trial_of_sample)]).astype(int):
            idx = np.flatnonzero(trial_of_sample == trial)
            sub = flat[idx]
            edge = _edge_nan_mask(sub)
            if not edge.any():
                continue
            sub[edge] = np.broadcast_to(value, sub.shape)[edge]
            flat[idx] = sub
            n_filled += int(edge.sum())
            trials_hit.add(trial)

        if not n_filled:
            continue
        out[name] = _like(sig, flat.reshape(d.shape))
        report.append((name, n_filled, len(trials_hit), value,
                       int(np.isnan(flat[~np.isnan(trial_of_sample)]).sum())))

    if verbose:
        print(f"fill_nans: edge-filled {len(report)} signal(s) over {len(trial_periods)} "
              f"trials (names containing {substring!r}"
              f"{' + ' + ', '.join(sorted(set(fill_values))) if fill_values else ''})")
        for name, n_filled, n_trials, value, left in report:
            print(f"  {name:28s} {n_trials:4d} trial(s) | filled {n_filled} edge NaN(s) "
                  f"with {value.round(3).tolist()}"
                  + (f" | {left} interior NaN(s) left" if left else ""))
    return out


def nan_bin_mask(
    signals: dict,
    signal_names,
    trial_periods: nap.IntervalSet,
    bin_size: float,
    verbose: bool = True,
) -> np.ndarray:
    """Which bins of the `bin_size` grid to drop: those a NaN sample falls in.

    The alternative to repairing a NaN (`fill_nans`) is to pretend that stretch of
    time never happened. A bin is unusable as soon as ONE sample inside it is NaN in
    ONE feature of ONE signal in `signal_names` -- `bin_average` is a plain mean, so
    a covariate whose window is half NaN has no defined value -- and the matching
    SPIKE COUNT bin has to go with it, or y would keep counts from a window the
    design cannot describe. Hence a mask over bins, applied to X and y together
    (`drop_bins`), rather than a mask over signals.

    The flag is built at native sample resolution and binned onto the same grid, so
    partial coverage is caught. Bins holding no samples at all are dropped too.

    Returns a bool array, True = DROP, over the bins of
    `X.restrict(trial_periods).bin_average(bin_size, ep=trial_periods)`.
    """
    signal_names = list(signal_names)
    missing = [n for n in signal_names if n not in signals]
    assert not missing, f"signal_names not in signals: {missing}"
    first = signals[signal_names[0]]
    assert all(np.array_equal(signals[n].t, first.t) for n in signal_names), \
        "signals are on different time grids -- align them before masking"

    def to_bins(bad_samples):
        flag = nap.Tsd(t=first.t, d=bad_samples.astype(float),
                       time_support=first.time_support)
        return flag.restrict(trial_periods).bin_average(bin_size, ep=trial_periods)

    per_signal = {n: np.isnan(np.asarray(signals[n].values, dtype=float)
                              .reshape(len(signals[n]), -1)).any(axis=1)
                  for n in signal_names}
    any_bad = np.logical_or.reduce(list(per_signal.values()))

    binned = to_bins(any_bad)
    # != 0 rather than > 0: a bin with no samples averages to NaN, and is unusable too
    drop = np.asarray(binned) != 0

    if verbose:
        bin_trial = trial_periods.in_interval(binned).astype(int)
        hit = np.unique(bin_trial[drop])
        lost = [t for t in hit if drop[bin_trial == t].all()]
        print(f"nan_bin_mask: dropping {drop.sum()} / {len(drop)} bins "
              f"({drop.mean():.2%}) touching {len(hit)} / {len(trial_periods)} trials"
              + (f", {len(lost)} of which lose EVERY bin" if lost else ""))
        in_trial = ~np.isnan(trial_periods.in_interval(first))   # ignore between-trial NaNs
        for n in signal_names:
            n_samples = int((per_signal[n] & in_trial).sum())
            if not n_samples:
                continue
            n_bins = int((np.asarray(to_bins(per_signal[n])) != 0).sum())
            print(f"  {n:28s} {n_samples:7d} NaN sample(s) in trials -> {n_bins:6d} bin(s)")
    return drop


def drop_bins(keep: np.ndarray, *objs):
    """Subset every binned object to `keep`, so X, y and friends stay row-aligned."""
    keep = np.asarray(keep, dtype=bool)
    out = []
    for o in objs:
        assert len(o) == len(keep), f"expected {len(keep)} rows, got {len(o)}"
        if isinstance(o, (nap.Tsd, nap.TsdFrame)):
            out.append(_like(o, np.asarray(o)[keep], t=np.asarray(o.t)[keep]))
        else:
            out.append(np.asarray(o)[keep])
    return out[0] if len(out) == 1 else tuple(out)


def shift_tsd(obj, offset: float):
    if isinstance(obj, nap.TsdFrame):
        return nap.TsdFrame(t=obj.times() + offset, d=obj.values, columns=obj.columns)
    return nap.Tsd(t=obj.times() + offset, d=obj.values)
 
 
def _date_prefix(session_date: str) -> int:
    # "20250309_01" → 2025030901
    return int(session_date.replace("_", "").replace("-", ""))


def shift_and_relabel_units(tsgroup: nap.TsGroup, offset: float,
                             session_date: str, bird: str = "",
                             brain_region: str = "") -> nap.TsGroup:
    new_id = lambda k: _date_prefix(session_date) * SESSION_ID_STRIDE + int(k)
    session_support = nap.IntervalSet(offset, offset + SESSION_DURATION)
    relabeled = nap.TsGroup(
        {new_id(k): nap.Ts(t=tsgroup[k].times() + offset, time_support=session_support)
         for k in tsgroup.keys()},
        time_support=session_support,
    )
    exclude = ["rate", "Amplitude", "ContamPct", "KSLabel", "amp", 
               "n_spikes", "shm", "fr", "group", "group_order"]
    info = tsgroup.metadata.drop(columns=exclude, errors="ignore").copy()

    info.index = [new_id(k) for k in info.index]
    info["session"] = session_date
    info["original_id"] = info.index.astype(str).str[10:].astype(int)
    info["individual"] = bird
    info["region"] = brain_region

    relabeled.set_info(info)
    return relabeled



import numpy as np
import matplotlib.pyplot as plt


def plot_grid(tensor, cols=5, plot_type="line"):
    """
    tensor shape: (space, trials, time)

    plot_type:
        "line" -> x, y, z over time
        "2d"   -> xy trajectory
        "3d"   -> xyz trajectory
    """

    if tensor.ndim == 2: # (trials, time)
        tensor = tensor[None, :, :]  # Add a space dimension if missing
        if plot_type in ["2d", "3d"]:
            raise ValueError("2D and 3D plots require a 3D tensor (space, trials, time)")

    n_trials = tensor.shape[1]
    rows = int(np.ceil(n_trials / cols))

    if plot_type == "3d":
        # One 3D subplot per trial
        fig = plt.figure(figsize=(5 * cols, 5 * rows))

        for i in range(n_trials):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

            ax.plot(
                tensor[0, i, :],  # x(t)
                tensor[1, i, :],  # y(t)
                tensor[2, i, :],  # z(t)
            )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"Trial {i}")

        plt.tight_layout()
        plt.show()

    else:
        # 2D grid
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(4 * cols, 4 * rows),
            squeeze=False,
        )

        axes = axes.flat

        for i in range(n_trials):
            ax = axes[i]

            if plot_type == "2d":
                # XY trajectory
                ax.plot(
                    tensor[0, i, :],  # x(t)
                    tensor[1, i, :],  # y(t)
                )

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)

            elif plot_type == "line":
                # x/y/z over time

                ax.plot(
                    tensor[:, i, :].T
                )

                ax.set_xlabel("time")
                ax.set_ylabel("position")
                ax.legend(
                    ["x", "y", "z"],
                    fontsize=6,
                    loc="upper right",
                )

            else:
                raise ValueError(
                    f"Unknown plot_type '{plot_type}'. "
                    "Use 'line', '2d', or '3d'."
                )

            ax.set_title(f"Trial {i}")

        # Hide unused subplots
        for i in range(n_trials, rows * cols):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


def save_png(
    fig: plt.Figure,
    folder: str,
    filename: str,
    close_fig: bool = True,
    dpi: int = 300,
) -> Path:
    """Write `fig` to plots/<folder>/<filename>.png, overwriting any previous run.

    The name is the identity: re-running with the same inputs replaces the file
    instead of leaving a trail of _p0/_p1 versions. That also makes the write safe
    under parallel workers -- the old counter was derived by globbing the directory,
    so two processes could pick the same index and clobber each other. Callers must
    therefore put whatever distinguishes a figure (session, unit id, mode) INTO
    `filename`.
    """
    out_dir = PLOT_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{filename}.png"
    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return path



def save_text(
    text: str,
    folder: str,
    filename: str = "01_columns",
    ext: str = ".txt",
) -> Path:
    """Write `text` to plots/<folder>/<filename><ext>, next to that folder's figures.

    Same folder resolution and same overwrite-by-name rule as `save_png`, because it
    is the same output: a folder of figures whose axis labels are variant names is
    unreadable a month later without a note saying what each variant contained. The
    default name starts with a digit so it sorts to the TOP of the folder, ahead of
    every PNG -- it is meant to be the first thing opened.
    """
    out_dir = PLOT_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{filename}{ext}"
    path.write_text(text, encoding="utf-8")
    return path


def save_pdf(
    figs: plt.Figure | Iterable[plt.Figure],
    folder: str,
    filename: str,
    close_figs: bool = True,
    dpi: int = 150,
    save_svg: bool = False,
    save_eps: bool = False,
) -> Path:

    mpl.rcParams.update({
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "path.simplify": False,
    })

    with plt.style.context(STYLE_PATH):
        if isinstance(figs, plt.Figure):
            figs = [figs]
        elif isinstance(figs, Iterable):
            figs = list(figs)
        else:
            raise TypeError("figs must be a Figure or iterable of Figures")

        valid_figs = [f for f in figs if isinstance(f, plt.Figure)]
        if not valid_figs:
            raise ValueError("no valid Figures to save")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = PLOT_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{filename}_{timestamp}.pdf"

        _save_native(valid_figs, path, dpi, close_figs)

        if save_svg:
            for i, fig in enumerate(valid_figs):
                svg_file = out_dir / f"{filename}_{timestamp}_p{i}.svg"
                fig.savefig(svg_file, format="svg")

        if save_eps:
            for i, fig in enumerate(valid_figs):
                eps_file = out_dir / f"{filename}_{timestamp}_p{i}.eps"
                fig.savefig(eps_file, format="eps")

        return path

def _save_native(
    figs: list[plt.Figure], path: Path, dpi: int, close_figs: bool
) -> None:
    with PdfPages(path, keep_empty=False) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=dpi, bbox_inches=None,
                        metadata={"Creator": "matplotlib"})
            if close_figs:
                plt.close(fig)