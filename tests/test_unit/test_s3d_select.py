"""Supervised S3D feature selection: Cohen's d, ranking, persistence."""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.io.schema import IS_EGOCENTRIC, KIND, VIDEO_FEATURE, is_egocentric, kind_of
from ethograph.video_features.extract import FEATURE_DIM, TIME_DIM, _to_dataarray
from ethograph.video_features.plan import S3DConfig, S3DPlan
from ethograph.video_features.select import FeatureRanking, cohens_d, rank_features

INFORMATIVE = (1, 4)


def reference_cohens_d(values: np.ndarray, labels: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Literal transcription of the notebook loop: ``(F, C)`` effect sizes."""
    n_features = values.shape[1]
    out = np.zeros((n_features, len(class_ids)))
    for f in range(n_features):
        column = values[:, f]
        if not np.any(column != 0):
            continue
        for j, target in enumerate(class_ids):
            during = column[labels == target]
            not_during = column[labels != target]
            var_during = np.var(during, ddof=1) if len(during) > 1 else 0.0
            var_not = np.var(not_during, ddof=1) if len(not_during) > 1 else 0.0
            pooled_std = np.sqrt((var_during + var_not) / 2)
            if pooled_std == 0:
                continue
            out[f, j] = abs(np.mean(during) - np.mean(not_during)) / pooled_std
    return out


def make_trial(
    n_frames: int = 400,
    n_features: int = 6,
    seed: int = 0,
    classes: tuple[int, ...] = (0, 1, 2),
    effect: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Noise everywhere, plus a strong class-dependent shift on ``INFORMATIVE``."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(np.asarray(classes), size=n_frames)
    values = rng.normal(size=(n_frames, n_features))
    values[labels == classes[-1], INFORMATIVE[0]] += effect
    values[labels == classes[1], INFORMATIVE[1]] += effect
    return values, labels


class TestCohensD:
    def test_matches_literal_notebook_loop(self):
        values, labels = make_trial()
        max_d, per_class, class_ids = cohens_d(values, labels)
        expected = reference_cohens_d(values, labels, class_ids)
        np.testing.assert_allclose(per_class, expected)
        np.testing.assert_allclose(max_d, expected.max(axis=1))

    def test_matches_literal_loop_with_background_included(self):
        values, labels = make_trial(seed=3)
        _, per_class, class_ids = cohens_d(values, labels, background=None)
        np.testing.assert_allclose(per_class, reference_cohens_d(values, labels, class_ids))

    def test_all_zero_column_scores_zero(self):
        values, labels = make_trial()
        values[:, 2] = 0.0
        max_d, per_class, _ = cohens_d(values, labels)
        assert max_d[2] == 0.0
        assert np.all(per_class[2] == 0.0)

    def test_constant_column_scores_zero(self):
        values, labels = make_trial()
        values[:, 3] = 7.5
        max_d, _, _ = cohens_d(values, labels)
        assert max_d[3] == 0.0

    def test_background_excluded_by_default(self):
        values, labels = make_trial()
        _, _, class_ids = cohens_d(values, labels)
        assert list(class_ids) == [1, 2]

    def test_background_none_includes_it(self):
        values, labels = make_trial()
        _, per_class, class_ids = cohens_d(values, labels, background=None)
        assert list(class_ids) == [0, 1, 2]
        assert per_class.shape == (6, 3)

    def test_other_background_id(self):
        values, labels = make_trial()
        _, _, class_ids = cohens_d(values, labels, background=2)
        assert list(class_ids) == [0, 1]

    def test_min_frames_skips_a_rare_class(self):
        values, labels = make_trial()
        labels[labels == 2] = 1
        labels[:5] = 2  # class 2 now has exactly 5 frames
        _, _, plenty = cohens_d(values, labels, min_frames=5)
        _, _, filtered = cohens_d(values, labels, min_frames=6)
        assert list(plenty) == [1, 2]
        assert list(filtered) == [1]

    def test_min_frames_zero_keeps_everything_present(self):
        values, labels = make_trial()
        _, _, class_ids = cohens_d(values, labels, min_frames=0)
        assert list(class_ids) == [1, 2]

    def test_negative_min_frames_raises(self):
        values, labels = make_trial()
        with pytest.raises(ValueError, match="min_frames"):
            cohens_d(values, labels, min_frames=-1)

    def test_single_class_trial_scores_nothing(self):
        values, labels = make_trial()
        labels[:] = 1
        max_d, per_class, class_ids = cohens_d(values, labels)
        assert len(class_ids) == 0
        assert per_class.shape == (6, 0)
        np.testing.assert_array_equal(max_d, np.zeros(6))

    def test_all_background_trial_scores_nothing(self):
        values, labels = make_trial()
        labels[:] = 0
        max_d, _, class_ids = cohens_d(values, labels)
        assert len(class_ids) == 0
        np.testing.assert_array_equal(max_d, np.zeros(6))

    def test_length_mismatch_raises(self):
        values, labels = make_trial()
        with pytest.raises(ValueError, match="frames"):
            cohens_d(values, labels[:-1])

    def test_non_2d_values_raises(self):
        _, labels = make_trial()
        with pytest.raises(ValueError, match="2-D"):
            cohens_d(np.zeros(len(labels)), labels)

    def test_blocked_pass_matches_unblocked(self, monkeypatch):
        values, labels = make_trial(n_frames=1000, seed=7)
        wide = cohens_d(values, labels)[1]
        monkeypatch.setattr("ethograph.video_features.select._BLOCK", 37)
        np.testing.assert_allclose(cohens_d(values, labels)[1], wide)


class TestRankFeatures:
    def test_finds_the_informative_columns(self):
        trials = [make_trial(seed=s) for s in range(4)]
        ranking = rank_features(trials)
        assert set(ranking.top(2).tolist()) == set(INFORMATIVE)
        assert ranking.n_trials == 4
        assert ranking.n_features == 6
        assert list(ranking.class_ids) == [1, 2]
        assert ranking.per_class.shape == (6, 2)

    def test_top_is_best_first(self):
        trials = [make_trial(seed=s) for s in range(3)]
        ranking = rank_features(trials)
        order = ranking.top(ranking.n_features)
        assert len(order) == ranking.n_features
        assert np.all(np.diff(ranking.scores[order]) <= 0)

    def test_top_clamps_and_accepts_zero(self):
        ranking = rank_features([make_trial()])
        assert len(ranking.top(99)) == 6
        assert len(ranking.top(0)) == 0
        with pytest.raises(ValueError, match="non-negative"):
            ranking.top(-1)

    def test_scores_are_the_mean_over_trials(self):
        trials = [make_trial(seed=s) for s in range(3)]
        ranking = rank_features(trials)
        by_hand = np.mean([cohens_d(v, lab)[0] for v, lab in trials], axis=0)
        np.testing.assert_allclose(ranking.scores, by_hand)

    def test_a_missing_class_contributes_zero_to_its_column(self):
        rich = make_trial(seed=1)
        poor_values, poor_labels = make_trial(seed=2)
        poor_labels[poor_labels == 2] = 1  # trial 1 has no class 2 at all
        ranking = rank_features([rich, (poor_values, poor_labels)])
        column = list(ranking.class_ids).index(2)
        only_rich = cohens_d(*rich)[1][:, list(cohens_d(*rich)[2]).index(2)]
        np.testing.assert_allclose(ranking.per_class[:, column], only_rich / 2)

    def test_unscoreable_trial_still_counts_in_the_average(self):
        good = make_trial(seed=1)
        blank_values, blank_labels = make_trial(seed=2)
        blank_labels[:] = 0
        ranking = rank_features([good, (blank_values, blank_labels)])
        np.testing.assert_allclose(ranking.scores, cohens_d(*good)[0] / 2)

    def test_no_trials_raises(self):
        with pytest.raises(ValueError, match="at least one trial"):
            rank_features([])

    def test_all_single_class_raises(self):
        trials = []
        for seed in range(2):
            values, labels = make_trial(seed=seed)
            labels[:] = 1
            trials.append((values, labels))
        with pytest.raises(ValueError, match="two classes to contrast"):
            rank_features(trials)

    def test_min_frames_starving_every_trial_raises(self):
        trials = [make_trial(seed=s) for s in range(2)]
        with pytest.raises(ValueError, match="min_frames=10000"):
            rank_features(trials, min_frames=10_000)

    def test_feature_count_mismatch_raises(self):
        first = make_trial(seed=0, n_features=6)
        second = make_trial(seed=1, n_features=5)
        with pytest.raises(ValueError, match="Trial 1 has 5 features"):
            rank_features([first, second])

    def test_length_mismatch_inside_a_trial_raises(self):
        values, labels = make_trial()
        with pytest.raises(ValueError, match="frames"):
            rank_features([(values, labels[:-1])])

    def test_accepts_a_generator(self):
        ranking = rank_features((make_trial(seed=s) for s in range(3)))
        assert ranking.n_trials == 3


class TestPersistence:
    def test_dict_round_trip(self):
        ranking = rank_features([make_trial(seed=s) for s in range(2)])
        data = ranking.to_dict()
        assert isinstance(data["scores"], list)
        assert isinstance(data["per_class"], list)
        assert isinstance(data["class_ids"], list)
        assert all(isinstance(c, int) for c in data["class_ids"])
        back = FeatureRanking.from_dict(data)
        np.testing.assert_allclose(back.scores, ranking.scores)
        np.testing.assert_allclose(back.per_class, ranking.per_class)
        np.testing.assert_array_equal(back.class_ids, ranking.class_ids)
        assert back.n_trials == ranking.n_trials

    def test_npz_round_trip(self, tmp_path):
        ranking = rank_features([make_trial(seed=s) for s in range(2)])
        path = ranking.save(tmp_path / "ranking.npz")
        assert path.exists()
        back = FeatureRanking.load(path)
        np.testing.assert_allclose(back.scores, ranking.scores)
        np.testing.assert_allclose(back.per_class, ranking.per_class)
        np.testing.assert_array_equal(back.class_ids, ranking.class_ids)
        assert back.n_trials == ranking.n_trials
        np.testing.assert_array_equal(back.top(3), ranking.top(3))

    def test_save_appends_the_suffix(self, tmp_path):
        ranking = rank_features([make_trial()])
        path = ranking.save(tmp_path / "nested" / "ranking")
        assert path.name == "ranking.npz"
        assert FeatureRanking.load(path).n_trials == 1


class TestSchemaAttrs:
    """The extracted DataArray declares what it is — without running S3D."""

    def test_to_dataarray_stamps_video_feature(self):
        feats = np.zeros((10, 4), dtype=np.float32)
        plan = S3DPlan(video_fps=30.0, step=1, stack_frames=13)
        da = _to_dataarray(feats, plan, "clip.mp4", S3DConfig(), "full", crop=None)
        assert da.attrs[KIND] == VIDEO_FEATURE
        assert kind_of(da) == VIDEO_FEATURE
        # Stored as 0/1: NetCDF has no boolean attribute type.
        assert da.attrs[IS_EGOCENTRIC] == 0
        assert is_egocentric(da) is False
        assert da.dims == (TIME_DIM, FEATURE_DIM)
        assert da.attrs["video_path"] == "clip.mp4"
