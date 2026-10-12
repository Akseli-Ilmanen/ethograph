"""Fill low-confidence keypoints with SAM2, where a lookup table says to.

Scripted and Qt-free, like :mod:`ethograph.triangulate`. A folder of 2D pose
files (DeepLabCut by default) is paired with a folder of videos by name; a
TSV names the stretches of each video worth the compute. Inside those, every
run of low-confidence frames for a chosen keypoint is re-tracked by SAM2 from
a confident frame shortly before the drop, and the mask centroid replaces the
keypoint until confidence recovers. Tracking is SAMURAI's motion-aware SAM 2
by default (:mod:`ethograph.samfill.samurai`); ``FillConfig(samurai=False)``
is plain SAM 2.

    from ethograph import samfill

    cfg = samfill.FillConfig(keypoints=["pellet"], low_confidence=0.9, seed_confidence=0.99)
    samfill.inspect_folders("poses/", "videos/", "segments.tsv", cfg, out_dir="plots/before")
    samfill.fill_folders("poses/", "videos/", "segments.tsv", "filled/", cfg)
    samfill.inspect_folders("poses/", "videos/", "segments.tsv", cfg, filled_dir="filled/", out_dir="plots/after")
"""

from ethograph.samfill.config import FillConfig, PixelExclusion, Segment, read_segments_tsv, segments_to_frames
from ethograph.samfill.fill import FillReport, fill_folders, fill_pair
from ethograph.samfill.pairing import Pair, pair_folders
from ethograph.samfill.plan import FillRun, plan_runs
from ethograph.samfill.review import gather_traces, inspect_folders, plot_traces
from ethograph.samfill.sam import SAM_MODELS, SamModel, SamPointTracker, ensure_checkpoint

__all__ = [
    "SAM_MODELS",
    "FillConfig",
    "FillReport",
    "FillRun",
    "Pair",
    "PixelExclusion",
    "SamModel",
    "SamPointTracker",
    "Segment",
    "ensure_checkpoint",
    "fill_folders",
    "fill_pair",
    "gather_traces",
    "inspect_folders",
    "pair_folders",
    "plan_runs",
    "plot_traces",
    "read_segments_tsv",
    "segments_to_frames",
]
