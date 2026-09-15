"""The pixel pipeline's config: durations in, upstream's frame counts out.

What earns a test here is the arithmetic that has an edge case — resolving two
durations against a rate, refusing what will not fit, and mapping a strided
prediction back to the full-rate clock. The dataclass fields themselves are
not tested: moving one would only report the edit back to whoever made it.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from ethograph.spot.confidence import CurveStats, curve_stats, densify, window_samples
from ethograph.spot.config import (
    MAX_FRAMES_PER_BATCH,
    ClipConfig,
    config_from_dict,
    load_config,
    save_config,
)


class TestResolve:
    """`context_s` / `resolution_ms` become stride, clip_len and dilate_len."""

    def test_reproduces_the_ladder_winner(self):
        # A2: 2.0 s of context at 10 ms resolution on 200 fps video.
        clip = ClipConfig(context_s=2.0, resolution_ms=10.0, positive_window_ms=10.0).resolve(200.0)
        assert (clip.stride, clip.clip_len, clip.dilate_len) == (2, 200, 1)

    def test_same_durations_follow_the_rate(self):
        """The point of durations: one config, several rigs."""
        cfg = ClipConfig(context_s=2.0, resolution_ms=10.0)
        assert cfg.resolve(25.0).clip_len == 50  # 25 fps: stride 1, 2 s = 50 frames
        assert cfg.resolve(200.0).clip_len == 200  # 200 fps: stride 2, 2 s = 200
        for fps in (25.0, 60.0, 200.0):
            assert cfg.resolve(fps).context_s == pytest.approx(2.0, abs=0.05)

    @pytest.mark.parametrize(
        "resolution_ms, expected",
        [(5.0, (1, 200, 2)), (10.0, (2, 100, 1)), (20.0, (4, 50, 0))],
    )
    def test_positive_window_holds_in_real_time(self, resolution_ms, expected):
        """dilate_len is in *strided* frames, so stride and dilation multiply.

        Deriving it from a duration is what stops dilation confounding a
        comparison between two resolutions — and it reproduces the dilations
        the stride ladder was run with, which were chosen by hand for exactly
        this reason (``z_notes/Z_conclusions.md``).
        """
        clip = ClipConfig(context_s=1.0, resolution_ms=resolution_ms, positive_window_ms=10.0).resolve(200.0)
        assert (clip.stride, clip.clip_len, clip.dilate_len) == expected
        # The positive half-width lands within one bin of the 10 ms asked for;
        # at 20 ms resolution the bin is already wider than the window.
        assert clip.dilate_len * clip.resolution_ms == pytest.approx(10.0, abs=clip.resolution_ms)

    def test_unset_resolution_is_the_finest_grid_the_budget_allows(self):
        """The default: every frame when it fits, else the smallest stride that does."""
        cfg = ClipConfig(context_s=2.0)  # resolution_ms unset
        assert cfg.resolve(25.0).stride == 1  # 50 frames fit outright
        assert cfg.resolve(200.0, max_frames=200).stride == 2  # 400 frames do not: every second one
        assert cfg.resolve(200.0, max_frames=480).stride == 1  # a bigger card: every frame
        assert cfg.resolve(200.0, max_frames=100).stride == 4
        for fps, budget in ((25.0, 200), (200.0, 200), (200.0, 480), (200.0, 100)):
            assert cfg.resolve(fps, max_frames=budget).clip_len <= budget

    def test_a_spelled_resolution_is_pinned_and_refused_when_it_does_not_fit(self):
        pinned = ClipConfig(context_s=2.0, resolution_ms=10.0)
        assert pinned.resolve(200.0, max_frames=480).stride == 2  # a bigger card changes nothing
        with pytest.raises(ValueError, match="raise resolution_ms"):
            pinned.resolve(200.0, max_frames=100)

    def test_the_budget_scales_with_the_card_and_is_the_measured_one_without(self, monkeypatch):
        import torch

        from ethograph.spot.vendored import frame_budget

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert frame_budget() == MAX_FRAMES_PER_BATCH
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1.0e9, 24.0e9))
        assert frame_budget() == 480

    def test_refuses_what_will_not_fit_and_names_the_duration(self):
        with pytest.raises(ValueError, match="context_s"):
            ClipConfig(context_s=8.0, resolution_ms=5.0).resolve(200.0)

    def test_refuses_a_clip_too_short_to_integrate(self):
        with pytest.raises(ValueError, match="context_s"):
            ClipConfig(context_s=0.05, resolution_ms=40.0).resolve(200.0)

    def test_the_ceiling_is_the_batch_ceiling(self):
        clip = ClipConfig(context_s=1.0, resolution_ms=5.0).resolve(200.0)
        assert clip.frames_per_batch <= MAX_FRAMES_PER_BATCH

    @pytest.mark.parametrize("fps", [25.0, 30.0, 60.0, 200.0])
    def test_rate_is_never_assumed(self, fps):
        clip = ClipConfig(context_s=1.0, resolution_ms=10.0).resolve(fps)
        assert clip.fps == fps

    def test_a_bad_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ClipConfig().resolve(0.0)


class TestStrideRecovery:
    """A strided prediction lands on the centre of its bin, never its edge."""

    def test_centre_not_edge(self):
        clip = ClipConfig(context_s=2.0, resolution_ms=20.0).resolve(200.0)
        assert clip.stride == 4
        # The dataset bins truth as floor(frame / 4), so bin 10 covers frames
        # 40..43 and its expected full-rate frame is 41.5.
        assert clip.to_frame(10) == pytest.approx(41.5)

    def test_stride_one_is_the_identity(self):
        clip = ClipConfig(context_s=1.0, resolution_ms=5.0).resolve(200.0)
        assert clip.stride == 1
        assert clip.to_frame(37) == 37

    def test_recovery_is_unbiased(self):
        """Averaged over a bin, the centre rule has no systematic error.

        Reading a bin back as `bin * k` is early by (k-1)/2 frames every time
        — 1.5 frames at stride 4, against a 20 ms budget.
        """
        k = 4
        clip = ClipConfig(context_s=2.0, resolution_ms=20.0).resolve(200.0)
        truth = np.arange(0, 400)
        recovered = np.array([clip.to_frame(t // k) for t in truth])
        assert np.mean(recovered - truth) == pytest.approx(0.0, abs=1e-9)


class TestConfigFile:
    """Building, refusing and round-tripping a config."""

    def _minimal(self, tmp_path):
        source = tmp_path / "ses-01.nc"
        source.touch()
        return {"sessions": [str(source)], "labels": {"classes": [31, 32]}}

    def test_builds_and_defaults_the_labels_path(self, tmp_path):
        cfg = config_from_dict(self._minimal(tmp_path), tmp_path)
        assert cfg.sessions[0].labels_path.name == "ses-01_labels.tsv"

    def test_segments_features_section_shape_is_refused_by_name(self, tmp_path):
        data = self._minimal(tmp_path) | {"features": {"columns": {}}}
        with pytest.raises(ValueError, match="lists the pose variables directly"):
            config_from_dict(data, tmp_path)
        cfg = config_from_dict(self._minimal(tmp_path) | {"features": {"speed": {}}}, tmp_path)
        assert cfg.fusing

    def test_the_confidence_rule_is_a_config_key_and_is_validated(self, tmp_path):
        custom = {"infer": {"confidence": "custom", "confidence_alpha": 0.8}}
        cfg = config_from_dict(self._minimal(tmp_path) | custom, tmp_path)
        assert (cfg.infer.confidence, cfg.infer.confidence_alpha) == ("custom", 0.8)
        with pytest.raises(ValueError, match="infer.confidence must be one of"):
            config_from_dict(self._minimal(tmp_path) | {"infer": {"confidence": "entropy"}}, tmp_path)
        with pytest.raises(ValueError, match="confidence_alpha"):
            config_from_dict(self._minimal(tmp_path) | {"infer": {"confidence_alpha": 2.0}}, tmp_path)

    def test_no_classes_is_refused(self, tmp_path):
        data = self._minimal(tmp_path) | {"labels": {"classes": []}}
        with pytest.raises(ValueError, match="labels.classes"):
            config_from_dict(data, tmp_path)

    def test_duplicate_classes_are_refused(self, tmp_path):
        data = self._minimal(tmp_path) | {"labels": {"classes": [31, 31]}}
        with pytest.raises(ValueError, match="more than once"):
            config_from_dict(data, tmp_path)

    def test_unknown_key_is_an_error_not_a_default(self, tmp_path):
        data = self._minimal(tmp_path) | {"clip": {"contextt_s": 2.0}}
        with pytest.raises(ValueError, match="unknown key"):
            config_from_dict(data, tmp_path)

    def test_nested_names_build_this_pipelines_types(self, tmp_path):
        """`train` and `model` exist in both pipelines and mean different things."""
        data = self._minimal(tmp_path) | {"train": {"epoch_frames": 1000}, "model": {"head": "gru"}}
        cfg = config_from_dict(data, tmp_path)
        assert cfg.train.epoch_frames == 1000
        assert cfg.model.head == "gru"

    def test_round_trips_through_yaml(self, tmp_path):
        cfg = config_from_dict(self._minimal(tmp_path) | {"clip": {"context_s": 4.0, "resolution_ms": 20.0}}, tmp_path)
        path = save_config(cfg, tmp_path / "spot.yaml")
        again = load_config(path)
        assert again.clip.context_s == 4.0
        assert again.labels.classes == [31, 32]
        assert [s.source for s in again.sessions] == [s.source for s in cfg.sessions]

    def test_overrides_are_dotted(self, tmp_path):
        path = tmp_path / "spot.yaml"
        path.write_text(yaml.safe_dump(self._minimal(tmp_path)), encoding="utf-8")
        cfg = load_config(path, ["clip.context_s=4.0", "train.epochs=3"])
        assert (cfg.clip.context_s, cfg.train.epochs) == (4.0, 3)

    def test_class_names_round_trip(self, tmp_path):
        cfg = config_from_dict(self._minimal(tmp_path), tmp_path)
        assert cfg.class_label(cfg.class_name(31)) == 31
        with pytest.raises(ValueError, match="not one of"):
            cfg.class_label("label_99")


class TestIndividual:
    """`individual` is the single value stamped into every exported label row."""

    def _minimal(self, tmp_path):
        source = tmp_path / "ses-01.nc"
        source.touch()
        return {"sessions": [str(source)], "labels": {"classes": [31]}}

    def test_defaults_to_none(self, tmp_path):
        cfg = config_from_dict(self._minimal(tmp_path), tmp_path)
        assert cfg.individual is None

    def test_round_trips_through_yaml(self, tmp_path):
        cfg = config_from_dict(self._minimal(tmp_path) | {"individual": "A"}, tmp_path)
        path = save_config(cfg, tmp_path / "spot.yaml")
        assert load_config(path).individual == "A"


class TestCrop:
    """`labels.crop` builds a `CropConfig` and is validated at config-build time."""

    def _minimal(self, tmp_path):
        source = tmp_path / "ses-01.nc"
        source.touch()
        return {"sessions": [str(source)], "labels": {"classes": [31]}}

    def test_builds_a_crop_config(self, tmp_path):
        crop = {"x0": 10, "y0": 20, "x1": 110, "y1": 220}
        data = self._minimal(tmp_path) | {"labels": {"classes": [31], "crop": crop}}
        cfg = config_from_dict(data, tmp_path)
        assert cfg.labels.crop.as_tuple() == (10, 20, 110, 220)

    def test_no_crop_is_none(self, tmp_path):
        cfg = config_from_dict(self._minimal(tmp_path), tmp_path)
        assert cfg.labels.crop is None

    def test_an_inverted_crop_is_refused_at_build_time(self, tmp_path):
        data = self._minimal(tmp_path) | {"labels": {"classes": [31], "crop": {"x0": 100, "y0": 0, "x1": 10, "y1": 50}}}
        with pytest.raises(ValueError, match="empty"):
            config_from_dict(data, tmp_path)

    def test_a_negative_crop_is_refused_at_build_time(self, tmp_path):
        data = self._minimal(tmp_path) | {"labels": {"classes": [31], "crop": {"x0": -5, "y0": 0, "x1": 10, "y1": 50}}}
        with pytest.raises(ValueError, match="outside the frame"):
            config_from_dict(data, tmp_path)

    def test_round_trips_through_yaml(self, tmp_path):
        crop = {"x0": 10, "y0": 20, "x1": 110, "y1": 220}
        data = self._minimal(tmp_path) | {"labels": {"classes": [31], "crop": crop}}
        cfg = config_from_dict(data, tmp_path)
        path = save_config(cfg, tmp_path / "spot.yaml")
        again = load_config(path)
        assert again.labels.crop.as_tuple() == (10, 20, 110, 220)


class TestConfidence:
    """Confidence is the curve's shape, and the shape has to be read right."""

    def _bump(self, length, centre, height, width=3):
        x = np.arange(length)
        return height * np.exp(-0.5 * ((x - centre) / width) ** 2)

    def test_one_clean_bump_scores_high(self):
        stats = curve_stats(self._bump(500, 250, 0.9), window=10)
        assert stats.index == pytest.approx(250, abs=1)
        assert stats.shape > 0.8

    def test_a_rival_peak_lowers_it(self):
        curve = self._bump(500, 250, 0.9) + self._bump(500, 100, 0.85)
        stats = curve_stats(curve, window=10)
        alone = curve_stats(self._bump(500, 250, 0.9), window=10)
        assert stats.shape < alone.shape
        assert stats.peak == pytest.approx(alone.peak, abs=0.02)

    def test_a_smeared_curve_lowers_it_at_the_same_height(self):
        """Peak height cannot tell these apart; focus can."""
        sharp = curve_stats(self._bump(500, 250, 0.9, width=3), window=10)
        broad = curve_stats(self._bump(500, 250, 0.9, width=60), window=10)
        assert sharp.peak == pytest.approx(broad.peak, abs=1e-6)
        assert broad.focus < sharp.focus

    def test_an_empty_curve_is_zero_not_a_crash(self):
        assert curve_stats(np.zeros(100), window=10) == CurveStats(index=0, peak=0.0, focus=0.0, ratio=0.0, found=False)
        assert curve_stats(np.array([]), window=10).shape == 0.0

    def test_a_rising_edge_is_not_a_peak(self):
        """A curve still climbing at the trial's end must not read as certain."""
        rising = np.linspace(0.0, 0.9, 200)
        assert curve_stats(rising, window=10).index == 199  # argmax stands in
        with_peak = rising.copy()
        with_peak[50] = 1.0
        assert curve_stats(with_peak, window=10).index == 50

    def test_densify_restores_the_zeros(self):
        curve = densify(np.array([3, 7]), np.array([0.5, 0.9]), length=10)
        assert curve.shape == (10,)
        assert curve[3] == pytest.approx(0.5)
        assert curve[7] == pytest.approx(0.9)
        assert curve.sum() == pytest.approx(1.4)

    def test_densify_drops_candidates_past_the_end(self):
        curve = densify(np.array([3, 99]), np.array([0.5, 0.9]), length=10)
        assert curve.sum() == pytest.approx(0.5)

    def test_the_focus_window_is_a_duration(self):
        assert window_samples(0.1, 100.0) == 10
        assert window_samples(0.1, 200.0) == 20
        with pytest.raises(ValueError, match="positive"):
            window_samples(0.1, 0.0)


class TestSessionNaming:
    """Four sessions called Trial_data3.nc must not collide in the outputs."""

    def _two(self, tmp_path):
        a = tmp_path / "ses-01" / "Trial_data3.nc"
        b = tmp_path / "ses-02" / "Trial_data3.nc"
        for p in (a, b):
            p.parent.mkdir()
            p.touch()
        return a, b

    def test_same_stem_without_names_is_told_apart_by_folder(self, tmp_path):
        a, b = self._two(tmp_path)
        cfg = config_from_dict({"sessions": [str(a), str(b)], "labels": {"classes": [31]}}, tmp_path)
        assert [s.label for s in cfg.sessions] == ["ses-01_Trial_data3", "ses-02_Trial_data3"]
        assert [s.label for s in cfg.select_sessions(["ses-02_Trial_data3"])] == ["ses-02_Trial_data3"]

    def test_the_distinguishing_folder_may_be_higher_up(self, tmp_path):
        a = tmp_path / "AK" / "ses-01" / "behav" / "Trial_data3.nc"
        b = tmp_path / "AI" / "ses-01" / "behav" / "Trial_data3.nc"
        for p in (a, b):
            p.parent.mkdir(parents=True)
            p.touch()
        cfg = config_from_dict({"sessions": [str(a), str(b)], "labels": {"classes": [31]}}, tmp_path)
        assert [s.label for s in cfg.sessions] == ["AK_Trial_data3", "AI_Trial_data3"]

    def test_explicit_names_and_a_twice_listed_source_are_still_refused(self, tmp_path):
        a, b = self._two(tmp_path)
        named = [{"source": str(a), "name": "x"}, {"source": str(b), "name": "x"}]
        with pytest.raises(ValueError, match="distinct `name:`"):
            config_from_dict({"sessions": named, "labels": {"classes": [31]}}, tmp_path)
        with pytest.raises(ValueError, match="listed more than once"):
            config_from_dict({"sessions": [str(a), str(a)], "labels": {"classes": [31]}}, tmp_path)

    def test_names_disambiguate_and_select(self, tmp_path):
        a, b = self._two(tmp_path)
        cfg = config_from_dict(
            {
                "sessions": [{"source": str(a), "name": "20260304_01"}, {"source": str(b), "name": "20260305_02"}],
                "labels": {"classes": [31]},
            },
            tmp_path,
        )
        assert [s.label for s in cfg.sessions] == ["20260304_01", "20260305_02"]
        assert [s.label for s in cfg.select_sessions(["20260305_02"])] == ["20260305_02"]

    def test_frames_dir_defaults_under_root_and_can_be_pointed_elsewhere(self, tmp_path):
        source = tmp_path / "ses-01.nc"
        source.touch()
        base = {"sessions": [str(source)], "labels": {"classes": [31]}}
        assert config_from_dict(base, tmp_path).frames_dir == (tmp_path / "frames").resolve()
        shared = config_from_dict(base | {"frames": "../shared_frames"}, tmp_path)
        assert shared.frames_dir == (tmp_path / ".." / "shared_frames").resolve()

    def test_a_crop_does_not_rename_the_frames_folder(self, tmp_path):
        """One folder per project; export.json per trial is what tells a stale crop apart."""
        source = tmp_path / "ses-01.nc"
        source.touch()
        base = {"sessions": [str(source)], "labels": {"classes": [31]}}
        crop = {"crop": {"x0": 0, "y0": 0, "x1": 8, "y1": 8}}
        cropped = config_from_dict(base | {"labels": base["labels"] | crop}, tmp_path)
        assert cropped.frames_dir == config_from_dict(base, tmp_path).frames_dir

    def test_a_numeric_looking_name_must_be_quoted(self, tmp_path):
        """YAML 1.1 reads `name: 20260304_01` as 2026030401 and the spelling is lost."""
        source = tmp_path / "ses-01.nc"
        source.touch()
        path = tmp_path / "spot.yaml"
        path.write_text(
            f"sessions:\n  - source: {source}\n    name: 20260304_01\nlabels: {{classes: [31]}}\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Quote it"):
            load_config(path)
        path.write_text(
            f"sessions:\n  - source: {source}\n    name: '20260304_01'\nlabels: {{classes: [31]}}\n", encoding="utf-8"
        )
        assert load_config(path).sessions[0].label == "20260304_01"


class TestTrialLimit:
    """`trials.limit` cuts the filtered list — a smoke run before a night of GPU."""

    def _session(self, ids):
        from types import SimpleNamespace

        return SimpleNamespace(trial_ids=list(ids), source="fake.nc", result=SimpleNamespace(metadata_df=None))

    def test_limit_takes_the_first_n(self):
        from ethograph.segment.config import TrialsConfig
        from ethograph.segment.sessions import filter_trials

        assert filter_trials(self._session([1, 2, 3, 4]), TrialsConfig(limit=2)) == [1, 2]
        assert filter_trials(self._session([1, 2, 3, 4]), TrialsConfig()) == [1, 2, 3, 4]

    def test_limit_below_one_is_refused(self):
        from ethograph.segment.config import TrialsConfig
        from ethograph.segment.sessions import filter_trials

        with pytest.raises(ValueError, match="limit"):
            filter_trials(self._session([1, 2]), TrialsConfig(limit=0))


class TestArchitectures:
    """The list comes from the vendored trainer's own CLI, never a copy of it."""

    def test_read_off_the_choices_block(self, tmp_path):
        from ethograph.spot.vendored import feature_architectures

        (tmp_path / "train_e2e.py").write_text(
            "parser.add_argument(\n    '-m', '--feature_arch', type=str, required=True, choices=[\n"
            "        # comment\n        'rn18',\n        'rny008_gsm',\n        'rny008_msagsm'\n    ], help='x')\n",
            encoding="utf-8",
        )
        assert feature_architectures(tmp_path) == ["rn18", "rny008_gsm", "rny008_msagsm"]

    def test_a_file_without_the_block_is_refused(self, tmp_path):
        from ethograph.spot.vendored import feature_architectures

        (tmp_path / "train_e2e.py").write_text("nothing here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="choices"):
            feature_architectures(tmp_path)

    def test_describe_spells_backbone_and_module(self):
        from ethograph.spot.vendored import describe_architecture

        assert describe_architecture("rny008_msagsm").startswith("RegNetY-800MF")
        assert "Multi-scale" in describe_architecture("rny008_msagsm")
        assert "no temporal mixing" in describe_architecture("rn18")

    def test_the_vendored_trainer_lists_the_two_that_matter(self):
        from ethograph.spot import architectures

        names = architectures()
        assert {"rny008_gsm", "rny008_msagsm"} <= set(names)
