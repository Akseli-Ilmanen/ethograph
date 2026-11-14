"""Trial-concatenated behaviour PCA: windows of every trial on one set of components."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ethograph.features.behaviour_pca import fit_behaviour_pca, window_means, window_starts
from ethograph.gui import dialog_burst_scatter as scatter


class TestWindows:
    def test_starts_slide_by_the_step_and_a_short_trial_still_has_one(self):
        assert window_starts(1.0, 0.4, 0.2) == pytest.approx([0.0, 0.2, 0.4, 0.6])
        assert window_starts(0.1, 0.4, 0.2) == pytest.approx([0.0])
        with pytest.raises(ValueError, match="longer than zero"):
            window_starts(1.0, 0.0, 0.2)

    def test_means_skip_nan_and_an_empty_window_is_nan(self):
        time = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
        values = np.array([[1.0, 5.0], [3.0, np.nan], [5.0, np.nan], [7.0, 9.0], [0.0, 0.0]])
        means = window_means(time, values, np.array([0.0, 0.2, 0.5]), 0.2)
        assert means[0] == pytest.approx([2.0, 5.0])  # samples at 0.0 and 0.1; the NaN is skipped
        assert means[1] == pytest.approx([6.0, 9.0])
        assert np.isnan(means[2]).all()  # nothing between 0.5 and 0.7


class TestFit:
    def test_trials_share_components_and_the_main_axis_is_the_correlated_one(self):
        rng = np.random.default_rng(0)
        drive = {"1": rng.normal(size=200), "2": rng.normal(size=150)}
        # Columns 0 and 1 move together; column 2 is small independent noise.
        blocks = {t: np.column_stack([d, d, 0.1 * rng.normal(size=len(d))]) for t, d in drive.items()}
        centers = {t: np.arange(len(d), dtype=float) for t, d in drive.items()}
        pca = fit_behaviour_pca(blocks, centers, ["a", "b", "noise"], 2)
        assert pca.scores["1"].shape == (200, 2) and pca.scores["2"].shape == (150, 2)
        assert pca.explained[0] > 0.6
        assert {name for name, _ in pca.top_loadings(0, 2)} == {"a", "b"}
        assert abs(np.corrcoef(pca.scores["2"][:, 0], drive["2"])[0, 1]) > 0.99

    def test_a_missing_feature_is_the_mean_not_a_dropped_window(self):
        blocks = {"1": np.array([[1.0, 1.0], [3.0, 3.0]]), "2": np.array([[2.0, np.nan]])}
        pca = fit_behaviour_pca(blocks, {"1": np.array([0.0, 1.0]), "2": np.array([0.0])}, ["a", "b"], 1)
        assert pca.scores["2"].ravel() == pytest.approx([0.0])  # both columns sit at their mean

    def test_a_span_reads_the_windows_inside_it_else_the_nearest(self):
        blocks = {"1": np.array([[0.0], [1.0], [2.0], [3.0]])}
        pca = fit_behaviour_pca(blocks, {"1": np.array([0.5, 1.5, 2.5, 3.5])}, ["a"], 1)
        score = pca.scores["1"][:, 0]
        assert pca.at("1", 1.0, 3.0, 0) == pytest.approx(score[1:3].mean())
        assert pca.at("1", 1.6, 1.7, 0) == pytest.approx(score[1])  # no centre inside: the nearest window
        assert np.isnan(pca.at("9", 0.0, 1.0, 0))


class _Loader:
    """One trial's features: ``speed`` (one series) and ``position`` (keypoint x space), every 0.1 s.

    Like the real loaders, it reads one series at a time: every dim must be pinned.
    """

    def __init__(self, n: int, with_position: bool = True):
        self.time = np.arange(n) * 0.1
        self.with_position = with_position

    def feature_dims(self, feature):
        return {"keypoint": ["beak", "tail"], "space": ["x", "y"]} if feature == "position" else {}

    def select(self, feature, selections, t0, t1):
        if feature == "speed":
            return SimpleNamespace(time=self.time, data=np.sin(self.time))
        if feature == "position" and self.with_position:
            assert set(selections) == {"keypoint", "space"}
            sign = 1.0 if selections["space"] == "x" else -1.0
            return SimpleNamespace(time=self.time, data=sign * self.time * (1 + (selections["keypoint"] == "tail")))
        return None


def test_a_precomputed_colour_is_read_at_the_middle_of_the_burst_and_scaled_to_unit(monkeypatch):
    time = np.arange(10) * 0.1
    colours = np.zeros((10, 3))
    colours[:, 0] = np.arange(10) * 25.5  # red ramps 0..229.5 on a 0-255 scale
    loader = SimpleNamespace(select=lambda feature, sel, t0, t1: SimpleNamespace(time=time, data=colours))
    monkeypatch.setattr(
        scatter, "iter_trial_windows", lambda state, trials: ((t, loader, None, None, 0.0) for t in trials)
    )
    bursts = [
        SimpleNamespace(trial="1", onset_s=0.35, offset_s=0.45),  # middle 0.4: sample 4
        SimpleNamespace(trial="2", onset_s=0.0, offset_s=0.1),  # a trial not on offer: no colour
    ]
    out = scatter.rgb_column(SimpleNamespace(trials=["1"]), bursts, "angle_rgb", {})
    assert out[0] == pytest.approx([0.4, 0.0, 0.0])
    assert np.isnan(out[1]).all()


def test_every_series_of_every_trial_is_windowed_and_a_missing_one_keeps_the_trials_windows(monkeypatch):
    loaders = {"1": _Loader(50), "2": _Loader(30, with_position=False)}
    monkeypatch.setattr(
        scatter, "iter_trial_windows", lambda state, trials: ((t, loaders[t], None, None, 0.0) for t in trials)
    )
    pca = scatter.behaviour_pca(SimpleNamespace(trials=["1", "2"]), ["speed", "position", "absent"], 0.5, 0.5, 2)
    assert pca.columns == [
        "speed",
        "position [beak, x]",
        "position [beak, y]",
        "position [tail, x]",
        "position [tail, y]",
    ]
    assert len(pca.centers["1"]) == 9 and len(pca.centers["2"]) == 5
    assert pca.centers["1"][0] == pytest.approx(0.25)
    assert np.isfinite(pca.scores["2"]).all()


def test_a_colours_channel_dim_is_found_by_its_size_whatever_it_is_called():
    """A colour feature is read whole: its channels are never something the user picks between."""
    assert scatter.channel_dim({"channel": ["0", "1", "2"]}) == "channel"
    assert scatter.channel_dim({"keypoint": ["a", "b", "c"], "RGB": ["0", "1", "2"]}) == "RGB"
    assert scatter.channel_dim({"keypoint": ["a", "b", "c"], "dim_1": ["0", "1", "2"]}) == "dim_1"
    assert scatter.channel_dim({"keypoint": ["a", "b"]}) is None
