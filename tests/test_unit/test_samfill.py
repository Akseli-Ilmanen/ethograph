"""SAM fill: planning, pairing, the lookup table and the write-back, all without a model."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ethograph.samfill.config import FillConfig, PixelExclusion, Segment, read_segments_tsv, segments_to_frames
from ethograph.samfill.fill import EDITED, fill_dataset
from ethograph.samfill.pairing import pair_folders, pose_matches_video
from ethograph.samfill.plan import excluded_frames, plan_anchor_run, plan_runs
from ethograph.samfill.sam import BadSeedError, TrackedPoint, mask_centroid, seed_shape

# -- planning ----------------------------------------------------------------------


def _plan(conf, window=None, **kw):
    conf = np.asarray(conf, dtype=float)
    args = dict(low_confidence=0.5, seed_confidence=0.99, margin=2)
    args.update(kw)
    return plan_runs(conf, window or range(len(conf)), **args)


def test_plan_seeds_a_margin_before_the_drop_and_stops_after_a_margin_of_recovery():
    #            0  1  2  3  4  5    6    7  8  9
    conf = [1, 1, 1, 1, 1, 0.1, 0.2, 1, 1, 1]
    (run,) = _plan(conf)
    assert run.seed == 3  # 5 - margin 2
    assert run.frames == range(4, 7)  # tracked up to the first confident pair at 7, 8
    assert run.fill == (5, 6)


def test_plan_walks_back_to_a_confident_seed_or_gives_up():
    conf = [1, 0.8, 0.8, 0.8, 0.1, 0.1, 1, 1]
    (run,) = _plan(conf, margin=1)
    assert run.seed == 0
    assert _plan([0.8, 0.8, 0.1, 1, 1], margin=1) == []


def test_plan_fills_only_low_frames_but_tracks_through_the_band():
    conf = [1, 1, 1, 0.1, 0.7, 0.1, 1, 1]
    (run,) = _plan(conf, margin=1)
    assert run.frames == range(3, 6)
    assert run.fill == (3, 5)


def test_plan_runs_to_window_end_without_recovery_and_never_fills_before_window():
    conf = [1, 1, 0.1, 0.1, 0.1, 0.1]
    (run,) = _plan(conf, window=range(3, 6), margin=1)
    assert run.seed == 1  # 3 - 1 = 2 is low, walk back to 1
    assert run.frames == range(2, 6)
    assert run.fill == (3, 4, 5)


def test_plan_treats_nan_as_low_and_splits_gaps():
    conf = [1, 1, np.nan, 1, 1, 1, 0.1, 1, 1]
    runs = _plan(conf, margin=1)
    assert [(r.seed, r.fill) for r in runs] == [(1, (2,)), (5, (6,))]


def test_anchor_run_seeds_once_at_the_first_confident_frame_and_fills_every_later_low_frame():
    conf = [0.5, 0.95, 1.0, 1.0, 0.1, 1.0, 0.2, 0.2, 1.0]
    (run,) = plan_anchor_run(np.array(conf), range(0, 9), low_confidence=0.5, seed_confidence=0.99)
    assert run.seed == 2
    assert run.frames == range(3, 9)
    assert run.fill == (4, 6, 7)
    assert plan_anchor_run(np.array(conf), range(5, 9), 0.5, 0.99)[0].seed == 5  # seed inside the window
    assert plan_anchor_run(np.array([0.9, 0.9, 0.1]), range(0, 3), 0.5, 0.99) == []  # nothing confident
    assert plan_anchor_run(np.array([1.0, 1.0, 1.0]), range(0, 3), 0.5, 0.99) == []  # nothing to fill


def test_plans_carry_alternate_seeds_in_order_of_preference():
    conf = np.array([1, 1, 1, 0.1, 1, 1, 1, 1, 0.1, 1, 1])
    (run,) = plan_runs(conf, range(0, 4), 0.5, 0.99, margin=1, max_tries=3)
    assert (run.seed, run.alternates) == (2, (1, 0))
    (run,) = plan_anchor_run(conf, range(0, 11), 0.5, 0.99, max_tries=3)
    assert (run.seed, run.alternates) == (0, (1, 2))
    # spaced alternates skip a rejected seed's near-identical neighbours
    conf = np.ones(30)
    conf[20] = 0.1
    (run,) = plan_runs(conf, range(0, 30), 0.5, 0.99, margin=5, max_tries=4)
    assert (run.seed, run.alternates) == (15, (10, 5, 0))
    (run,) = plan_anchor_run(conf, range(0, 30), 0.5, 0.99, max_tries=3, spacing=5)
    assert (run.seed, run.alternates) == (0, (5, 10))


# -- lookup table ----------------------------------------------------------------------


def test_segments_round_to_nearest_frame_inclusive_and_merge(tmp_path: Path):
    tsv = tmp_path / "segments.tsv"
    tsv.write_text("file\tonset_s\toffset_s\ntrial1.mp4\t1.0\t2.0\ntrial1\t1.98\t3.0\n# comment\ntrial2.mp4\t0\t0.5\n")
    segs = read_segments_tsv(tsv)
    assert segs[0] == Segment("trial1.mp4", 1.0, 2.0)
    assert segments_to_frames(segs, "trial1.mp4", fps=10.0, n_frames=100) == [range(10, 31)]
    assert segments_to_frames(segs, "trial2.mp4", fps=30.0, n_frames=10) == [range(0, 10)]
    assert segments_to_frames(segs, "trial3.mp4", fps=30.0, n_frames=10) == []


def test_segments_empty_end_runs_to_the_end_of_the_video(tmp_path: Path):
    tsv = tmp_path / "segments.tsv"
    tsv.write_text("trial1.mp4\t2.0\t\ntrial2.mp4\ntrial3.mp4\t\t1.0\n")
    segs = read_segments_tsv(tsv)
    assert segs == [Segment("trial1.mp4", 2.0, None), Segment("trial2.mp4"), Segment("trial3.mp4", 0.0, 1.0)]
    assert segments_to_frames(segs, "trial1.mp4", fps=10.0, n_frames=100) == [range(20, 100)]
    assert segments_to_frames(segs, "trial2.mp4", fps=10.0, n_frames=100) == [range(0, 100)]
    assert segments_to_frames(segs, "trial3.mp4", fps=10.0, n_frames=100) == [range(0, 11)]


def test_segments_tsv_without_header_and_bad_header(tmp_path: Path):
    assert Segment("a", 0, 1).file == "a"
    with pytest.raises(ValueError):
        Segment("a", 2, 1)
    bad = tmp_path / "s.tsv"
    bad.write_text("file\tfoo\tbar\n")
    with pytest.raises(ValueError):
        read_segments_tsv(bad)


def test_fill_config_rejects_impossible_thresholds():
    with pytest.raises(ValueError):
        FillConfig(keypoints=[], low_confidence=0.6, seed_confidence=0.99)
    with pytest.raises(ValueError):
        FillConfig(keypoints=["a"], low_confidence=0.9, seed_confidence=0.5)


# -- pairing ----------------------------------------------------------------------


def test_pose_matches_video_accepts_dlc_suffix_but_not_a_longer_stem():
    assert pose_matches_video("trial1DLC_resnet50_projMay1shuffle1_100000", "trial1")
    assert pose_matches_video("trial1_filtered", "trial1")
    assert not pose_matches_video("trial10DLC_resnet50", "trial1")
    assert not pose_matches_video("trial1", "trial10")


def test_pose_matches_video_ignores_separator_differences_before_the_dlc_suffix():
    pose = "2025-05-12_009_Ivy_-cam-1DLC_resnet50_Felix_cross_SessionsAug1shuffle1_200000_filtered"
    assert pose_matches_video(pose, "2025-05-12_009_Ivy-cam-1")
    assert not pose_matches_video(pose, "2025-05-12_009_Ivy-cam-2")
    assert not pose_matches_video(pose, "2025-05-12_027_Ivy-cam-1")


def test_pair_folders_works_on_one_mixed_folder(tmp_path: Path):
    for name in (
        "2025-05-12_009_Ivy-cam-1.mp4",
        "2025-05-12_009_Ivy-cam-2.mp4",
        "2025-05-12_042_Ivy-cam-1.mp4",
        "2025-05-12_009_Ivy_-cam-1DLC_resnet50_Felix_cross_SessionsAug1shuffle1_200000_filtered.csv",
        "2025-05-12_009_Ivy_-cam-2DLC_resnet50_Felix_cross_SessionsAug1shuffle1_200000_filtered.csv",
    ):
        (tmp_path / name).touch()
    pairs, lone_poses, lone_videos = pair_folders(tmp_path, tmp_path)
    assert [(p.pose.name[:27], p.video.name) for p in pairs] == [
        ("2025-05-12_009_Ivy_-cam-1DL", "2025-05-12_009_Ivy-cam-1.mp4"),
        ("2025-05-12_009_Ivy_-cam-2DL", "2025-05-12_009_Ivy-cam-2.mp4"),
    ]
    assert lone_poses == []
    assert [v.name for v in lone_videos] == ["2025-05-12_042_Ivy-cam-1.mp4"]
    assert pairs[0].output_stem == pairs[0].pose.stem and pairs[0].name == "2025-05-12_009_Ivy-cam-1"


def test_pair_folders_keeps_only_files_in_both(tmp_path: Path):
    poses, videos = tmp_path / "poses", tmp_path / "videos"
    poses.mkdir()
    videos.mkdir()
    for name in ("trial1DLC_resnet50.h5", "trial1DLC_resnet50_filtered.h5", "trial10DLC.h5", "orphan.h5", "notes.txt"):
        (poses / name).touch()
    for name in ("trial1.mp4", "trial10.mp4", "trial2.avi"):
        (videos / name).touch()
    pairs, lone_poses, lone_videos = pair_folders(poses, videos)
    assert [(p.pose.name, p.video.name) for p in pairs] == [
        ("trial1DLC_resnet50.h5", "trial1.mp4"),
        ("trial10DLC.h5", "trial10.mp4"),
    ]
    assert [p.name for p in lone_poses] == ["orphan.h5", "trial1DLC_resnet50_filtered.h5"]
    assert [v.name for v in lone_videos] == ["trial2.avi"]


# -- write-back ----------------------------------------------------------------------


def test_mask_centroid():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    assert mask_centroid(mask) == (3.0, 1.5, 6)
    assert mask_centroid(np.zeros((2, 2), dtype=bool)) is None


class _FakeTracker:
    """Yields a point at (frame, frame) for every tracked frame; records the seeds."""

    bad_seeds: set[int] = set()

    def __init__(self, n_frames: int):
        self.n_frames = n_frames
        self.video = Path("fake.mp4")
        self.seeds: list[tuple[int, float, float]] = []

    def track(
        self,
        seed_frame: int,
        x: float,
        y: float,
        frames: range,
        box_px: int | None = None,
        min_circularity: float = 0.0,
    ) -> Iterator[TrackedPoint]:
        self.seeds.append((seed_frame, x, y))
        self.box_px = box_px
        if seed_frame in self.bad_seeds:
            raise BadSeedError(seed_frame, None, "test says no")
        for f in frames:
            yield TrackedPoint(f, float(f), float(f) + 100, 0.5)


def _poses(conf_pellet: list[float]) -> xr.Dataset:
    n = len(conf_pellet)
    position = np.zeros((n, 2, 2, 1))
    position[:, 0, 0, 0] = 10  # pellet x
    position[:, 1, 0, 0] = 20  # pellet y
    position[:, :, 1, 0] = 7  # paw
    confidence = np.ones((n, 2, 1))
    confidence[:, 0, 0] = conf_pellet
    return xr.Dataset(
        {
            "position": (("time", "space", "keypoints", "individuals"), position),
            "confidence": (("time", "keypoints", "individuals"), confidence),
        },
        coords={
            "time": np.arange(n) / 10.0,
            "space": ["x", "y"],
            "keypoints": ["pellet", "paw"],
            "individuals": ["ind0"],
        },
    )


def test_fill_dataset_overwrites_only_low_frames_of_chosen_keypoint_and_flags_them():
    ds = _poses([1, 1, 1, 0.1, 0.1, 1, 1, 1, 0.2, 1, 1])
    cfg = FillConfig(keypoints=["pellet"], margin=1, low_confidence=0.5, seed_confidence=0.99)
    tracker = _FakeTracker(len(ds.time))
    out, reports = fill_dataset(ds, tracker, [range(0, 9)], cfg)  # window excludes frames 9, 10

    assert tracker.seeds == [(2, 10.0, 20.0), (7, 10.0, 20.0)]
    pellet = out.position.sel(keypoints="pellet", individuals="ind0")
    assert pellet.sel(space="x").values.tolist() == [10, 10, 10, 3, 4, 10, 10, 10, 8, 10, 10]
    assert pellet.sel(space="y").values.tolist() == [20, 20, 20, 103, 104, 20, 20, 20, 108, 20, 20]
    assert out.confidence.sel(keypoints="pellet", individuals="ind0").values.tolist() == [
        1,
        1,
        1,
        0.5,
        0.5,
        1,
        1,
        1,
        0.5,
        1,
        1,
    ]
    assert out[EDITED].astype(int).sel(keypoints="pellet", individuals="ind0").values.tolist() == [
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
    ]
    assert (out.position.sel(keypoints="paw") == 7).all()
    assert out[EDITED].dtype == bool and not out[EDITED].sel(keypoints="paw").any()
    assert (ds.position.sel(keypoints="pellet", space="x") == 10).all()  # input untouched
    assert [(r.seed, r.n_low, r.n_filled) for r in reports] == [(2, 2, 2), (7, 1, 1)]
    assert tracker.box_px == 24

    anchored = FillConfig(
        keypoints=["pellet"], low_confidence=0.5, seed_confidence=0.99, seeding="anchor", seed_box_px=None
    )
    tracker = _FakeTracker(len(ds.time))
    _, reports = fill_dataset(ds, tracker, [range(0, 9)], anchored)
    assert tracker.seeds == [(0, 10.0, 20.0)] and tracker.box_px is None
    assert [(r.seed, r.first, r.last, r.n_low, r.n_filled) for r in reports] == [(0, 1, 8, 3, 3)]


def test_seed_shape_separates_a_disc_from_a_rod():
    yy, xx = np.mgrid[:40, :40]
    disc = (xx - 20) ** 2 + (yy - 20) ** 2 <= 8**2
    rod = np.zeros((40, 40), bool)
    rod[5:35, 18:21] = True
    assert seed_shape(disc).circularity > 0.8
    assert seed_shape(rod).circularity < 0.4
    assert seed_shape(np.zeros((4, 4), bool)) is None


def test_a_rejected_seed_falls_back_to_the_next_confident_frame():
    ds = _poses([1, 1, 1, 0.1, 0.1, 1, 1, 1, 0.2, 1, 1])
    cfg = FillConfig(keypoints=["pellet"], low_confidence=0.5, seed_confidence=0.99, margin=1)
    tracker = _FakeTracker(len(ds.time))
    tracker.bad_seeds = {2, 7, 6}
    out, reports = fill_dataset(ds, tracker, [range(0, 9)], cfg)
    # first gap: 2 rejected, 1 accepted; second gap: 7 and 6 rejected, 5 accepted
    assert tracker.seeds[:2] == [(2, 10.0, 20.0), (1, 10.0, 20.0)]
    assert [(r.seed, r.seeds_rejected, r.first, r.n_filled) for r in reports] == [(1, 1, 2, 2), (5, 2, 6, 1)]
    assert out[EDITED].astype(int).sel(keypoints="pellet", individuals="ind0").values.tolist() == [
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
    ]

    tracker = _FakeTracker(len(ds.time))
    tracker.bad_seeds = {0, 1, 2}
    out, reports = fill_dataset(ds, tracker, [range(0, 5)], cfg)
    assert reports[0].n_filled == 0 and reports[0].seeds_rejected == 3
    assert (out.position == ds.position).all()


def test_excluded_frames_cover_boxes_the_y_floor_and_padding():
    excl = PixelExclusion(boxes=((10, 10, 20, 20),), y_max=100, pad_frames=1)
    x = np.array([0, 15, 0, 0, 0, 0, 0, 0])
    y = np.array([0, 15, 0, 0, 0, 150, 0, 0])
    assert excluded_frames(x, y, excl).tolist() == [True, True, True, False, True, True, True, False]
    assert not excluded_frames(x, y, PixelExclusion()).any()
    with pytest.raises(ValueError):
        PixelExclusion(boxes=((5, 5, 1, 1),))


def test_exclusion_is_chosen_by_the_longest_matching_video_name_key():
    cam1, cam2, anything = PixelExclusion(y_max=1), PixelExclusion(y_max=2), PixelExclusion(y_max=3)
    cfg = FillConfig(
        keypoints=["pellet"],
        low_confidence=0.5,
        seed_confidence=0.99,
        exclusions={"cam-1": cam1, "cam-2": cam2, "": anything},
    )
    assert cfg.exclusion_for("2026_045_Poppy-cam-2.mp4") is cam2
    assert cfg.exclusion_for("2026_045_Poppy-cam-7.mp4") is anything
    assert FillConfig(keypoints=["pellet"], low_confidence=0.5, seed_confidence=0.99).exclusion_for("x") is None


def test_excluded_detections_are_dropped_filled_and_never_seeds():
    ds = _poses([1, 1, 1, 1, 1, 1, 1, 1])
    # frames 3 and 4 sit inside the forbidden box although DLC is sure of them
    ds.position.loc[{"keypoints": "pellet", "space": "x", "time": ds.time.values[[3, 4]]}] = 500
    cfg = FillConfig(
        keypoints=["pellet"],
        low_confidence=0.5,
        seed_confidence=0.99,
        margin=1,
        exclusions={"fake": PixelExclusion(boxes=((490, 0, 510, 50),))},
    )
    tracker = _FakeTracker(len(ds.time))
    out, reports = fill_dataset(ds, tracker, [range(0, 8)], cfg)
    assert tracker.seeds == [(2, 10.0, 20.0)]  # seeded before the dropped frames, never on them
    assert out.position.sel(keypoints="pellet", space="x").values.ravel().tolist() == [10, 10, 10, 3, 4, 10, 10, 10]
    assert out[EDITED].astype(int).sel(keypoints="pellet").values.ravel().tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert [(r.seed, r.n_low, r.n_filled) for r in reports] == [(2, 2, 2)]

    # with no SAM result the dropped frames stay NaN instead of keeping the wrong pixel
    tracker = _FakeTracker(len(ds.time))
    tracker.bad_seeds = {0, 1, 2}
    out, _ = fill_dataset(ds, tracker, [range(0, 8)], cfg)
    assert np.isnan(out.position.sel(keypoints="pellet", space="x").values.ravel()[[3, 4]]).all()
    assert out[EDITED].sel(keypoints="pellet").values.ravel().tolist() == [False] * 3 + [True] * 2 + [False] * 3


def test_edited_flag_is_absent_when_nothing_changed():
    ds = _poses([1, 1, 1])
    cfg = FillConfig(keypoints=["pellet"], low_confidence=0.5, seed_confidence=0.99)
    out, reports = fill_dataset(ds, _FakeTracker(3), [range(0, 3)], cfg)
    assert reports == [] and EDITED not in out


def test_fill_dataset_refuses_unknown_keypoint():
    ds = _poses([1, 1])
    with pytest.raises(KeyError):
        fill_dataset(
            ds, _FakeTracker(2), [range(0, 2)], FillConfig(keypoints=["tail"], low_confidence=0.6, seed_confidence=0.99)
        )


def test_write_output_follows_the_source_format_or_writes_netcdf(tmp_path: Path):
    from ethograph.samfill.fill import write_output

    ds = _poses([1, 1, 1]).rename({"keypoints": "keypoint", "individuals": "individual"})
    ds.attrs.update(source_software="DeepLabCut", fps=10.0, ds_type="poses")
    ds[EDITED] = xr.ones_like(ds.confidence, dtype=bool)

    source = write_output(ds, tmp_path, "trial", "source")
    netcdf = write_output(ds, tmp_path, "trial", "netcdf")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trial_edited.h5", "trial_edited.nc"]
    assert source.suffix == ".h5" and netcdf.suffix == ".nc"
    with xr.open_dataset(netcdf) as back:
        assert EDITED in back

    ds.attrs["source_software"] = "SLEAP"
    assert write_output(ds, tmp_path / "sleap", "trial", "source").name == "trial_edited.h5"
    ds.attrs["source_software"] = "Anipose"
    with pytest.raises(ValueError, match="netcdf"):
        write_output(ds, tmp_path, "trial", "source")
    with pytest.raises(ValueError):
        write_output(ds, tmp_path, "trial", "both")


def test_traces_cover_only_the_windows_and_plot_before_over_after(tmp_path: Path):
    from ethograph.samfill.review import gather_traces, plot_traces, traces_from_dataset

    cfg = FillConfig(keypoints=["pellet"], low_confidence=0.6, seed_confidence=0.99)
    before = _poses([1, 1, 0.1, 0.1, 1, 1, 1, 1])
    after = before.copy(deep=True)
    after.position.loc[{"keypoints": "pellet", "space": "x"}] = 42
    windows = [range(1, 4), range(6, 8)]
    traces = pd.concat(
        [
            traces_from_dataset(before, windows, cfg, "trial", "before"),
            traces_from_dataset(after, windows, cfg, "trial", "after"),
        ]
    )
    assert traces["frame"].tolist() == [1, 2, 3, 6, 7] * 2
    assert traces["keypoint"].unique().tolist() == ["pellet"]
    assert traces[traces["stage"] == "after"]["x"].tolist() == [42] * 5
    assert traces[traces["stage"] == "before"]["confidence"].tolist() == [1, 0.1, 0.1, 1, 1]

    (path,) = plot_traces(traces, cfg, tmp_path)
    assert path.name == "trial_traces.png" and path.stat().st_size > 0
    with pytest.raises(ValueError):
        traces_from_dataset(before, windows, cfg, "trial", "during")
    assert gather_traces.__doc__


def test_samurai_memory_bank_admits_only_well_scored_frames():
    pytest.importorskip("octron")
    from ethograph.samfill.samurai import SamuraiParams, select_memory_frames

    params = SamuraiParams()
    good, bad = {"best_iou_score": 0.9, "object_score_logits": 2.0}, {"best_iou_score": 0.2, "object_score_logits": 2.0}
    outputs = {10: good, 11: bad, 12: good, 13: good, 14: bad, 20: good}
    # the frame just before is always in, even when it scored badly; 20 is not in the past
    assert select_memory_frames(outputs, 15, params, max_frames=16) == [10, 12, 13, 14]
    assert select_memory_frames(outputs, 15, params, max_frames=3) == [12, 13, 14]
    assert select_memory_frames(outputs, 14, params, max_frames=16) == [10, 12, 13]
    assert select_memory_frames({5: {}}, 6, params, max_frames=16) == [5]  # a prompted frame has no scores
    assert select_memory_frames({}, 6, params, max_frames=16) == []
    assert "++model.kf_score_weight=0.25" in params.hydra_overrides()


def test_output_defaults_to_netcdf():
    assert FillConfig(keypoints=["pellet"], low_confidence=0.6, seed_confidence=0.99).output == "netcdf"
