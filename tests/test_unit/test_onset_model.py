"""Unit tests for the LightGBM onset model (labels/onset_model.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from ethograph.io.catalog import XarrayLoader
from ethograph.labels import label_inputs as li
from ethograph.labels import onset_model as om


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point ~/.ethograph at a temp dir so tests never touch the real store."""
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / ".ethograph"))


def _make_ds(
    t_event: float,
    fs: float = 50.0,
    dur: float = 10.0,
    seed: int = 0,
    t_second: float | None = None,
) -> xr.Dataset:
    """One synthetic trial: Gaussian bumps on (x, y) plus noise.

    *t_event* drives both channels; *t_second*, when given, adds a bump only
    on y — a second point-event class with its own signature.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / fs)
    bump = np.exp(-0.5 * ((t - t_event) / 0.05) ** 2)
    x = 5.0 * bump + rng.normal(0, 0.1, t.size)
    y = -3.0 * bump + rng.normal(0, 0.1, t.size)
    if t_second is not None:
        y = y + 8.0 * np.exp(-0.5 * ((t - t_second) / 0.05) ** 2)
    return xr.Dataset(
        {"signal": (("time", "space"), np.column_stack([x, y]))},
        coords={"time": t, "space": ["x", "y"]},
    )


FEATURES = {"signal": {"space": ["x", "y"]}}


def _make_config(name: str = "test-model", targets: dict[int, str] | None = None) -> om.OnsetModelConfig:
    return om.OnsetModelConfig(
        name=name,
        targets=targets if targets is not None else {3: "peck"},
        features=FEATURES,
        window_s=0.4,
        tolerance_s=0.1,
        max_iter=60,
    )


# ---------------------------------------------------------------------------
# Columns + config
# ---------------------------------------------------------------------------


def test_enumerate_columns_is_deterministic():
    cols = om.enumerate_columns({"signal": {"space": ["x", "y"]}, "flat": {}})
    assert [c.name for c in cols] == ["signal|space=x", "signal|space=y", "flat"]
    assert cols[0].selections == {"space": "x"}
    assert cols[2].selections == {}


def test_config_roundtrip():
    config = _make_config()
    om.save_config(config)
    assert om.list_models() == ["test-model"]
    loaded = om.load_config("test-model")
    assert loaded == config


def test_legacy_single_target_config_upgrades(tmp_path):
    """A config written before multi-target support still loads."""
    raw = {
        "name": "old-model",
        "target_label": 7,
        "target_name": "land",
        "features": FEATURES,
    }
    d = om.model_dir("old-model")
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = om.load_config("old-model")
    assert config.targets == {7: "land"}
    assert config.target_name(7) == "land"


# ---------------------------------------------------------------------------
# Windowing + targets
# ---------------------------------------------------------------------------


def test_lag_offsets_symmetric_and_capped():
    offsets = om.lag_offsets(fs=1000.0, window_s=1.0)
    assert len(offsets) <= om.MAX_LAGS
    assert offsets[0] == -offsets[-1]
    assert 0 in offsets


def test_build_windows_nan_edges():
    data = np.arange(10.0)
    x = om.build_windows(data, np.array([-1, 0, 1]))
    assert x.shape == (10, 3)
    assert np.isnan(x[0, 0]) and np.isnan(x[-1, 2])
    assert x[5, 0] == 4.0 and x[5, 1] == 5.0 and x[5, 2] == 6.0


def test_make_targets_gaussian_peak():
    time = np.arange(0.0, 1.0, 0.01)
    y, w = om.make_targets(time, 0.5, tolerance_s=0.05)
    assert y.sum() > 0
    assert np.argmax(w * y) == np.argmin(np.abs(time - 0.5))
    assert np.all(w[y == 0] == 1.0)


def test_make_targets_outside_range_raises():
    time = np.arange(0.0, 1.0, 0.01)
    with pytest.raises(ValueError, match="outside"):
        om.make_targets(time, 5.0, tolerance_s=0.05)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extract_features_shape():
    loader = XarrayLoader(_make_ds(3.0))
    time, data = om.extract_features(loader, FEATURES)
    assert data.shape == (len(time), 2)
    assert np.isclose(om.sampling_rate(time), 50.0)


def test_extract_features_rejects_rate_mismatch():
    ds = _make_ds(3.0)
    t_aux = np.arange(0.0, 10.0, 1.0 / 100.0)
    ds["audio"] = ("time_aux", np.zeros(t_aux.size))
    ds = ds.assign_coords(time_aux=t_aux)
    loader = XarrayLoader(ds)
    with pytest.raises(ValueError, match="[Ss]ampling-rate mismatch"):
        om.extract_features(loader, {**FEATURES, "audio": {}})


def test_extract_features_missing_feature_raises():
    loader = XarrayLoader(_make_ds(3.0))
    with pytest.raises(ValueError, match="not available"):
        om.extract_features(loader, {"nope": {}})


# ---------------------------------------------------------------------------
# Derivative columns
# ---------------------------------------------------------------------------


def test_derivative_column_follows_the_value_it_is_taken_from():
    cols = om.enumerate_columns({"signal": {"space": ["x", "y"]}, "flat": {}}, ["signal"])
    assert [c.name for c in cols] == [
        "signal|space=x",
        "signal|space=x|d/dt",
        "signal|space=y",
        "signal|space=y|d/dt",
        "flat",
    ]
    assert [c.derivative for c in cols] == [False, True, False, True, False]


def test_extract_features_derivative_is_np_gradient():
    loader = XarrayLoader(_make_ds(3.0))
    time, data = om.extract_features(loader, FEATURES, derivatives=["signal"])
    assert data.shape == (len(time), 4)
    for value, derivative in ((0, 1), (2, 3)):
        assert np.allclose(data[:, derivative], np.gradient(data[:, value], time))


def test_extract_model_features_reads_the_config_derivatives():
    config = _make_config()
    config.derivatives = ["signal"]
    loader = XarrayLoader(_make_ds(3.0))
    _time, data = om.extract_model_features(loader, config)
    assert data.shape[1] == len(config.columns()) == 4


def test_a_config_written_before_derivatives_reads_back_with_none():
    config = _make_config()
    om.save_config(config)
    path = om.model_dir(config.name) / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["derivatives"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert om.load_config(config.name).derivatives == []


# ---------------------------------------------------------------------------
# Re-pinning the individual: one animal's model on another animal's session
# ---------------------------------------------------------------------------


def _make_two_animal_ds(t_freddy: float, t_ivy: float, fs: float = 50.0, dur: float = 10.0, seed: int = 0):
    """One trial holding both animals: each one's bump at its own time."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / fs)

    def _pair(t_event: float) -> np.ndarray:
        bump = np.exp(-0.5 * ((t - t_event) / 0.05) ** 2)
        return np.column_stack([5.0 * bump + rng.normal(0, 0.1, t.size), -3.0 * bump + rng.normal(0, 0.1, t.size)])

    data = np.stack([_pair(t_freddy), _pair(t_ivy)], axis=1)
    return xr.Dataset(
        {"signal": (("time", "individual", "space"), data)},
        coords={"time": t, "individual": ["Freddy", "Ivy"], "space": ["x", "y"]},
    )


def _animal_config(individual: str = "Freddy", name: str = "two-animal") -> om.OnsetModelConfig:
    return om.OnsetModelConfig(
        name=name,
        targets={3: "peck"},
        features={"signal": {"individual": [individual], "space": ["x", "y"]}},
        window_s=0.4,
        tolerance_s=0.1,
        max_iter=30,
    )


class TestRetarget:
    """A classifier is fitted on numbers; the individual is only the key that
    selects them. Re-pinning it is what lets a model trained on one animal run
    on another's session in the same rig."""

    def test_a_config_without_an_individual_dim_is_unchanged(self):
        config = _make_config()
        assert om.individual_dim(config) is None
        assert om.retarget_individual(config, "Ivy") is config

    def test_repinning_only_changes_the_individual(self):
        config = _animal_config()
        retargeted = om.retarget_individual(config, "Ivy")
        assert om.config_individuals(retargeted) == ["Ivy"]
        assert [c.name for c in retargeted.columns()] == [c.name.replace("Freddy", "Ivy") for c in config.columns()]

    def test_the_same_individual_is_a_no_op(self):
        config = _animal_config()
        assert om.retarget_individual(config, "Freddy") is config
        assert om.retarget_individual(config, "") is config

    def test_a_two_animal_model_still_runs_on_its_own_session(self):
        """A model built on an actor *and* a partner is not a single-animal
        model with a name to swap: asked for one of the animals it reads, it
        keeps reading both, and the combo only says whose labels these are."""
        config = _animal_config()
        config.features["signal"]["individual"] = ["Freddy", "Ivy"]
        assert om.retarget_individual(config, "Freddy") is config
        assert om.retarget_individual(config, "Ivy") is config

    def test_a_two_animal_model_on_a_third_animal_is_refused(self):
        """Collapsing two columns onto one animal would hand the classifier
        the same data in the slots it learned as two different animals."""
        config = _animal_config()
        config.features["signal"]["individual"] = ["Freddy", "Ivy"]
        with pytest.raises(ValueError, match="2 individuals"):
            om.retarget_individual(config, "Poppy")

    def test_the_repinned_columns_are_the_other_animal_s(self):
        """The contract: re-pinning reads exactly what a config written for
        that animal would, in the same order."""
        loader = XarrayLoader(_make_two_animal_ds(2.0, 6.0))
        retargeted = om.extract_model_features(loader, om.retarget_individual(_animal_config(), "Ivy"))
        native = om.extract_model_features(loader, _animal_config("Ivy"))
        assert np.array_equal(retargeted[1], native[1])
        assert not np.array_equal(retargeted[1], om.extract_model_features(loader, _animal_config())[1])

    def _train_on_freddy(self, name: str = "two-animal") -> om.OnsetModelConfig:
        config = _animal_config(name=name)
        om.save_config(config)
        for i, (t_freddy, t_ivy) in enumerate([(1.5, 7.0), (3.2, 8.4), (5.0, 1.1), (6.7, 2.9), (2.4, 5.6), (7.3, 4.2)]):
            loader = XarrayLoader(_make_two_animal_ds(t_freddy, t_ivy, seed=i))
            time, data = om.extract_model_features(loader, config)
            om.write_trial_training_data(config.name, "sess-freddy", i + 1, time, data, {3: t_freddy})
        om.train_model(config.name)
        return config

    def test_a_model_trained_on_one_animal_finds_the_other_s_event(self):
        config = self._train_on_freddy()
        bundle = om.load_bundle(config.name)

        loader = XarrayLoader(_make_two_animal_ds(1.0, 6.4, seed=99))
        read_config = om.retarget_individual(om.bundle_config(bundle), "Ivy")
        time, data = om.extract_model_features(loader, read_config)
        prediction = om.predict_events(bundle, time, data)[3]

        assert abs(prediction.time - 6.4) < 0.2

    def test_config_yaml_edited_after_training_is_reported(self):
        """The trap this guards: editing config.yaml to name another animal
        looks like it retargets the model, and the bundle carries the layout
        the classifiers were actually fitted on."""
        config = self._train_on_freddy()
        assert om.config_drifted(config.name) is None

        om.save_config(_animal_config("Ivy", name=config.name))
        trained = om.config_drifted(config.name)
        assert trained is not None and om.config_individuals(trained) == ["Freddy"]

    def test_a_copied_model_folder_is_not_drift(self):
        """Only what a trained model reads counts — a rename does not."""
        config = self._train_on_freddy()
        renamed = om.load_config(config.name)
        renamed.name = "copied"
        om.save_config(renamed)
        (om.model_dir(config.name) / "model.joblib").replace(om.model_dir("copied") / "model.joblib")
        assert om.config_drifted("copied") is None


# ---------------------------------------------------------------------------
# Train + predict end-to-end
# ---------------------------------------------------------------------------


def test_train_and_predict_recovers_event():
    config = _make_config()
    om.save_config(config)

    event_times = [1.5, 3.2, 5.0, 6.7, 8.1, 2.4, 4.9, 7.3]
    for i, t_event in enumerate(event_times):
        loader = XarrayLoader(_make_ds(t_event, seed=i))
        time, data = om.extract_features(loader, config.features)
        om.write_trial_training_data(config.name, "sess-a", i + 1, time, data, {3: t_event})
    om.write_session_meta(config.name, "sess-a", {"n_trials": len(event_times)})

    summary = om.train_model(config.name)
    assert summary["n_trials"] == len(event_times)
    assert summary["n_sessions"] == 1
    assert summary["targets"][3]["n_positive"] > 0
    assert om.is_trained(config.name)

    bundle = om.load_bundle(config.name)
    t_true = 4.2
    loader = XarrayLoader(_make_ds(t_true, seed=99))
    time, data = om.extract_features(loader, config.features)
    predictions = om.predict_events(bundle, time, data)
    assert set(predictions) == {3}
    assert abs(predictions[3].time - t_true) <= 0.2
    assert 0.0 <= predictions[3].confidence <= 1.0


def test_two_targets_predicted_together():
    """One model, two point-event classes, one pass over the features."""
    config = _make_config(targets={3: "peck", 4: "land"})
    om.save_config(config)

    events = [(1.5, 6.0), (3.2, 8.0), (5.0, 1.2), (6.7, 2.5), (8.1, 4.0), (2.4, 7.1)]
    for i, (t_peck, t_land) in enumerate(events):
        loader = XarrayLoader(_make_ds(t_peck, seed=i, t_second=t_land))
        time, data = om.extract_features(loader, config.features)
        om.write_trial_training_data(config.name, "sess-a", i + 1, time, data, {3: t_peck, 4: t_land})

    summary = om.train_model(config.name)
    assert set(summary["targets"]) == {3, 4}

    bundle = om.load_bundle(config.name)
    assert set(bundle["models"]) == {3, 4}
    loader = XarrayLoader(_make_ds(4.2, seed=99, t_second=8.4))
    time, data = om.extract_features(loader, config.features)
    result = om.predict_trial(bundle, time, data)
    predictions = result.events
    assert abs(predictions[3].time - 4.2) <= 0.3
    assert abs(predictions[4].time - 8.4) <= 0.3
    assert predictions[3].name == "peck"
    # The curves the events were read off come back with them: one per
    # target, on the caller's clock, so review can draw what the single
    # confidence number compressed.
    assert set(result.curves) == {3, 4}
    assert all(len(curve) == len(time) for curve in result.curves.values())
    assert om.predict_events(bundle, time, data)[3].time == predictions[3].time


def test_target_without_training_data_raises():
    """A target no stored trial carries cannot be trained — say so."""
    config = _make_config(targets={3: "peck", 4: "land"})
    om.save_config(config)
    loader = XarrayLoader(_make_ds(2.0))
    time, data = om.extract_features(loader, config.features)
    om.write_trial_training_data(config.name, "sess-a", 1, time, data, {3: 2.0})
    with pytest.raises(ValueError, match="No training trial carries 'land'"):
        om.train_model(config.name)


def _bump(n_frames: int, at: int, amplitude: float = 0.8, width: float = 2.5) -> np.ndarray:
    """A Gaussian bump of *width* frames on a low baseline."""
    t = np.arange(n_frames)
    return 0.02 + amplitude * np.exp(-0.5 * ((t - at) / width) ** 2)


def _hump(n_frames: int, at: int, amplitude: float, width: float) -> np.ndarray:
    """A bare Gaussian, to add as a rival to a :func:`_bump`."""
    t = np.arange(n_frames)
    return amplitude * np.exp(-0.5 * ((t - at) / width) ** 2)


class TestConfidence:
    """Confidence is the height of the curve's tallest peak — nothing else.

    The number has to be readable straight off the curve the review draws,
    because that is what lets a user set a threshold by looking at it.
    """

    def test_the_tallest_peak_is_the_event(self):
        curve = _bump(3000, 1500, amplitude=0.8) + _hump(3000, 750, 0.4, 2.5)
        index, height = om.tallest_peak(curve)
        assert index == 1500
        assert height == pytest.approx(0.82, abs=0.01)

    def test_a_rising_edge_is_not_a_peak(self):
        """A curve still climbing at the trial's end must not read as certain."""
        ramp = np.linspace(0.0, 0.9, 500)
        index, height = om.tallest_peak(ramp + _hump(500, 200, 0.3, 5.0))
        assert index == 200
        assert height == pytest.approx(0.66, abs=0.02)

    def test_a_flat_curve_scores_its_own_level(self):
        assert om.tallest_peak(np.zeros(100))[1] == 0.0
        assert om.tallest_peak(np.full(100, 0.3))[1] == pytest.approx(0.3)

    def test_an_empty_curve_is_not_confident(self):
        assert om.tallest_peak(np.array([]))[1] == 0.0
        assert om.tallest_peak(np.array([0.7]))[1] == pytest.approx(0.7)

    def test_height_is_the_confidence_the_curve_shows(self):
        """What lands in the TSV is a point on the drawn curve."""
        curve = _bump(3000, 1500, amplitude=0.55)
        index, height = om.tallest_peak(curve)
        assert om.OnsetPrediction(3, "peck", 15.0, confidence=height).confidence == pytest.approx(curve[index])


class TestPlacement:
    """A point label goes at the frame the curve names — no derived statistic."""

    def test_the_event_is_placed_at_the_tallest_peak(self):
        time = np.arange(0, 10, 0.01)
        curve = _bump(len(time), 640, amplitude=0.7)
        index, height = om.tallest_peak(curve)
        assert index == 640
        assert float(time[index]) == pytest.approx(6.40)
        assert height == pytest.approx(curve[640])

    def test_no_class_moves_another(self):
        """Every target is read off its own curve, in isolation — no class's
        evidence is allowed to pull another's event toward it."""
        time = np.arange(0, 10, 0.01)
        curves = {3: _bump(len(time), 700), 4: _bump(len(time), 300)}
        placed = {label: float(time[om.tallest_peak(curve)[0]]) for label, curve in curves.items()}
        assert placed[3] == pytest.approx(7.0)
        assert placed[4] == pytest.approx(3.0)


class TestCalibration:
    """How often the model is right is a verdict on the **model**.

    It is fitted on trials the classifiers did not see and reported when the
    model trains. It is deliberately *not* applied to a label's confidence:
    that number stays the height of the drawn curve, so the user can set a
    threshold by looking at the picture rather than by trusting a transform.
    """

    def test_a_perfect_record_still_admits_doubt(self):
        """8 of 8 held out is 9/10, never 1.0 — eight trials cannot say more."""
        cal = om.TargetCalibration(n_trials=8, n_hits=8, slope=1.0, intercept=0.0)
        assert cal.hit_rate == pytest.approx(9 / 10)

    def test_nothing_held_out_knows_nothing(self):
        """One training trial: nothing is known about how often it is right."""
        assert om.TargetCalibration(n_trials=0, n_hits=0, slope=1.0, intercept=0.0).hit_rate == 0.5

    def test_the_record_does_not_touch_a_label(self):
        """A label's confidence is its curve's peak — the model's record is
        reported at training, never folded into the number on a label."""
        curve = _bump(3000, 1500, amplitude=0.9)
        assert om.OnsetPrediction(1, "a", 1.0, confidence=om.tallest_peak(curve)[1]).confidence == pytest.approx(
            curve.max()
        )

    def test_fit_reads_the_held_out_curves(self):
        """A hit is a prediction within the tolerance of the labelled event."""
        config = _make_config()
        time = np.arange(0, 10, 0.01)
        trials = [om.TrainingTrial(time, np.zeros((len(time), 2)), {3: 5.0}, 100.0) for _ in range(4)]
        curves = [
            {3: _bump(len(time), 500)},  # right on the event
            {3: _bump(len(time), 505)},  # within the tolerance
            {3: _bump(len(time), 900)},  # nowhere near
            {3: _bump(len(time), 100)},
        ]
        cal = om.fit_confidence_calibration(trials, curves, config)[3]
        assert (cal.n_trials, cal.n_hits) == (4, 2)
        assert cal.hit_rate == pytest.approx(3 / 6)

    def test_training_records_the_held_out_score(self):
        config = _make_config()
        om.save_config(config)
        for i, t_event in enumerate([1.5, 3.2, 5.0, 6.7, 8.1, 2.4]):
            loader = XarrayLoader(_make_ds(t_event, seed=i))
            time, data = om.extract_features(loader, config.features)
            om.write_trial_training_data(config.name, "sess-a", i + 1, time, data, {3: t_event})
        summary = om.train_model(config.name)
        assert summary["calibration"][3]["n_trials"] == 6

        bundle = om.load_bundle(config.name)
        assert isinstance(bundle["calibration"][3], om.TargetCalibration)
        loader = XarrayLoader(_make_ds(4.2, seed=99))
        time, data = om.extract_features(loader, config.features)
        result = om.predict_trial(bundle, time, data)
        # The label's number is the statistic the held-out record chose,
        # read off the curve that was drawn for it.
        cal = bundle["calibration"][3]
        stats = om.read_curve(result.curves[3], 100.0, 0.05)
        assert result.events[3].statistic == cal.statistic
        assert result.events[3].confidence == pytest.approx(stats.statistic(cal.statistic))

    def test_a_record_where_only_shape_tells_hits_apart_switches_the_statistic(self):
        """Peak heights identical, misses ride on a rival bump: shape must win."""
        config = _make_config()
        time = np.arange(0, 10, 0.01)
        trials = [om.TrainingTrial(time, np.zeros((len(time), 2)), {3: 5.0}, 100.0) for _ in range(8)]
        curves = []
        for i in range(8):
            if i % 2 == 0:
                curves.append({3: _bump(len(time), 500)})  # clean hit
            else:
                # the tallest bump is far off (a miss), a second one at the event
                curves.append({3: _bump(len(time), 800) + 0.9 * _bump(len(time), 500)})
        cal = om.fit_confidence_calibration(trials, curves, config)[3]
        assert (cal.n_trials, cal.n_hits) == (8, 4)
        assert cal.aucs["ratio"] > cal.aucs["peak"] + 0.05
        assert cal.statistic != "peak"


def test_train_without_data_raises():
    om.save_config(_make_config())
    with pytest.raises(ValueError, match="[Nn]o training data"):
        om.train_model("test-model")


def test_predict_rejects_rate_mismatch():
    config = _make_config()
    om.save_config(config)
    loader = XarrayLoader(_make_ds(2.0))
    time, data = om.extract_features(loader, config.features)
    om.write_trial_training_data(config.name, "sess-a", 1, time, data, {3: 2.0})
    om.train_model(config.name)
    bundle = om.load_bundle(config.name)

    other = XarrayLoader(_make_ds(2.0, fs=100.0))
    time_other, data_other = om.extract_features(other, config.features)
    with pytest.raises(ValueError, match="[Ss]ampling-rate mismatch"):
        om.predict_events(bundle, time_other, data_other)


# ---------------------------------------------------------------------------
# Existing labels as inputs
# ---------------------------------------------------------------------------


def _labels_df(rows: list[dict]) -> pd.DataFrame:
    """A trial's label rows, in the columns the renderer reads."""
    return pd.DataFrame(rows, columns=["labels", "onset_s", "offset_s", "event_type", "individual"])


def _state_row(label: int, onset: float, offset: float, individual: str = "Freddy") -> dict:
    return {
        "labels": label,
        "onset_s": onset,
        "offset_s": offset,
        "event_type": "state",
        "individual": individual,
    }


def _point_row(label: int, onset: float, individual: str = "Freddy") -> dict:
    return {
        "labels": label,
        "onset_s": onset,
        "offset_s": np.nan,
        "event_type": "point",
        "individual": individual,
    }


class TestLabelInputs:
    """What a class already tells you about *when* is evidence, and the
    classifier gets it as columns like any other."""

    TIME = np.arange(0.0, 10.0, 0.02)

    def test_a_state_renders_its_on_off_vector(self):
        inp = li.LabelInput(label=5, name="approach", event_type="state")
        column = inp.render(_labels_df([_state_row(5, 2.0, 4.0)]), self.TIME)
        assert column.shape == (self.TIME.size, 1)
        inside = (self.TIME >= 2.0) & (self.TIME <= 4.0)
        assert np.array_equal(column[:, 0] > 0, inside)
        assert set(np.unique(column)) == {0.0, 1.0}

    def test_a_point_renders_a_laplacian_at_each_sigma(self):
        inp = li.LabelInput(label=6, name="cue", event_type="point")
        columns = inp.render(_labels_df([_point_row(6, 5.0)]), self.TIME)
        assert columns.shape == (self.TIME.size, len(li.POINT_SIGMAS_S))
        for j, sigma in enumerate(li.POINT_SIGMAS_S):
            assert np.isclose(columns[:, j].max(), 1.0)
            assert abs(self.TIME[int(np.argmax(columns[:, j]))] - 5.0) < 0.02
            assert np.allclose(columns[:, j], np.exp(-np.abs(self.TIME - 5.0) / sigma), atol=1e-9)

    def test_two_events_never_push_the_channel_past_one(self):
        """The maximum, not the sum: a value cannot depend on how many other
        events the trial happens to contain."""
        inp = li.LabelInput(label=6, name="cue", event_type="point")
        columns = inp.render(_labels_df([_point_row(6, 5.0), _point_row(6, 5.05)]), self.TIME)
        assert columns.max() <= 1.0

    def test_a_class_the_trial_does_not_carry_is_zeros(self):
        """Which is the state every trial the model runs on is in."""
        inp = li.LabelInput(label=5, name="approach", event_type="state")
        assert not inp.render(_labels_df([_state_row(9, 2.0, 4.0)]), self.TIME).any()
        assert not inp.render(None, self.TIME).any()

    def test_the_frozen_event_type_decides_what_is_read(self):
        """A mapping edited after training must not change a column shape."""
        rows = _labels_df([_state_row(5, 2.0, 4.0), _point_row(5, 8.0)])
        assert li.LabelInput(label=5, name="a", event_type="state").render(rows, self.TIME).shape[1] == 1
        point = li.LabelInput(label=5, name="a", event_type="point").render(rows, self.TIME)
        assert abs(self.TIME[int(np.argmax(point[:, 0]))] - 8.0) < 0.02

    def test_each_individual_is_its_own_column(self):
        inp = li.LabelInput(label=5, name="approach", event_type="state", individuals=["Freddy", "Ivy"])
        rows = _labels_df([_state_row(5, 2.0, 4.0, "Freddy"), _state_row(5, 6.0, 7.0, "Ivy")])
        columns = inp.render(rows, self.TIME)
        assert columns.shape == (self.TIME.size, 2)
        assert np.array_equal(columns[:, 0] > 0, (self.TIME >= 2.0) & (self.TIME <= 4.0))
        assert np.array_equal(columns[:, 1] > 0, (self.TIME >= 6.0) & (self.TIME <= 7.0))

    def test_no_individual_pinned_reads_whoever_labelled_it(self):
        """A single-individual session has nothing to choose, exactly as a
        single-valued dim is never drawn as a row."""
        inp = li.LabelInput(label=5, name="approach", event_type="state")
        rows = _labels_df([_state_row(5, 2.0, 4.0, "Freddy"), _state_row(5, 6.0, 7.0, "Ivy")])
        columns = inp.render(rows, self.TIME)
        assert columns.shape == (self.TIME.size, 1)
        assert columns[:, 0].sum() > 0

    def test_column_names_line_up_with_what_is_rendered(self):
        inputs = [
            li.LabelInput(label=5, name="approach", event_type="state", individuals=["Freddy", "Ivy"]),
            li.LabelInput(label=6, name="cue", event_type="point"),
        ]
        names = li.label_columns(inputs)
        rendered = li.render_label_inputs(inputs, _labels_df([_state_row(5, 2.0, 4.0)]), self.TIME)
        assert rendered.shape == (self.TIME.size, len(names))
        assert names[0] == "label:approach(5)|individual=Freddy"
        assert names[-1] == f"label:cue(6)|sigma={li.POINT_SIGMAS_S[-1]:g}"

    def test_an_unknown_event_type_is_refused(self):
        with pytest.raises(ValueError, match="unknown event type"):
            li.LabelInput(label=5, name="approach", event_type="interval")


class TestLabelInputConfig:
    """The config is the one input layout, label columns included."""

    def _config(self, **kwargs) -> om.OnsetModelConfig:
        config = _make_config(**kwargs)
        config.label_inputs = [li.LabelInput(label=5, name="approach", event_type="state")]
        return config

    def test_the_layout_is_features_then_labels(self):
        config = self._config()
        assert config.column_names() == ["signal|space=x", "signal|space=y", "label:approach(5)"]

    def test_extraction_appends_the_label_columns(self):
        config = self._config()
        loader = XarrayLoader(_make_ds(3.0))
        time, data = om.extract_model_features(loader, config, labels=_labels_df([_state_row(5, 2.0, 4.0)]))
        assert data.shape == (time.size, len(config.column_names()))
        assert np.array_equal(data[:, 2] > 0, (time >= 2.0) & (time <= 4.0))

    def test_a_shift_puts_the_labels_where_the_events_are(self):
        """Labels are trial-relative; a pynapple loader clock is not."""
        config = self._config()
        ds = _make_ds(3.0)
        shifted = ds.assign_coords(time=ds["time"].values + 100.0)
        time, data = om.extract_model_features(
            XarrayLoader(shifted), config, labels=_labels_df([_state_row(5, 2.0, 4.0)]), shift=100.0
        )
        trial_time = time - 100.0
        assert np.array_equal(data[:, 2] > 0, (trial_time >= 2.0) & (trial_time <= 4.0))

    def test_labels_are_required_when_the_config_reads_them(self):
        config = self._config()
        with pytest.raises(ValueError, match="reads existing labels"):
            om.extract_model_features(XarrayLoader(_make_ds(3.0)), config)

    def test_a_target_cannot_be_its_own_input(self):
        """At training the label is there; at inference it is not, because
        prediction only ever runs on trials that lack the target."""
        approach = li.LabelInput(label=5, name="approach", event_type="state")
        with pytest.raises(ValueError, match="cannot read the class it is asked to place"):
            om.OnsetModelConfig(name="clash", targets={5: "approach"}, label_inputs=[approach])

    def test_a_clash_assembled_field_by_field_never_reaches_disk(self):
        """A config built a field at a time skips __post_init__ — saving is the
        second gate, so nothing trains on a column that means two things."""
        config = self._config()
        config.targets = {5: "approach"}
        with pytest.raises(ValueError, match="cannot read the class it is asked to place"):
            om.save_config(config)

    def test_label_inputs_survive_a_yaml_round_trip(self):
        config = self._config()
        config.label_inputs.append(li.LabelInput(label=6, name="cue", event_type="point", individuals=["Ivy"]))
        om.save_config(config)
        assert om.load_config(config.name) == config

    def test_a_config_written_before_label_inputs_reads_back_with_none(self):
        om.save_config(_make_config())
        path = om.model_dir("test-model") / "config.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        del raw["label_inputs"]
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        assert om.load_config("test-model").label_inputs == []

    def test_retargeting_repoints_the_label_inputs_too(self):
        """A model that times a peck off a partner approach reads the other
        animal approach when it runs on them."""
        config = _animal_config()
        config.label_inputs = [li.LabelInput(label=5, name="approach", event_type="state", individuals=["Freddy"])]
        retargeted = om.retarget_individual(config, "Ivy")
        assert retargeted.label_inputs[0].individuals == ["Ivy"]
        assert om.config_individuals(retargeted) == ["Ivy"]

    def test_a_label_input_reading_two_animals_is_left_alone(self):
        config = _make_config()
        config.label_inputs = [
            li.LabelInput(label=5, name="approach", event_type="state", individuals=["Freddy", "Ivy"])
        ]
        assert om.retarget_individual(config, "Poppy") is config


def _noise_ds(seed: int, fs: float = 50.0, dur: float = 10.0) -> xr.Dataset:
    """A trial whose features say nothing at all about when anything happened."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / fs)
    return xr.Dataset(
        {"signal": (("time", "space"), rng.normal(0, 1.0, (t.size, 2)))},
        coords={"time": t, "space": ["x", "y"]},
    )


def test_a_model_can_time_an_event_off_a_label_alone():
    """The end-to-end claim: with the features pure noise, an existing state
    label that ends at the event is enough to place it."""
    config = _make_config(targets={3: "peck"})
    config.label_inputs = [li.LabelInput(label=5, name="approach", event_type="state")]
    om.save_config(config)

    for i, t_event in enumerate([1.5, 3.2, 5.0, 6.7, 8.1, 2.4, 4.9, 7.3]):
        labels = _labels_df([_state_row(5, t_event - 1.0, t_event)])
        time, data = om.extract_model_features(XarrayLoader(_noise_ds(i)), config, labels=labels)
        om.write_trial_training_data(config.name, "sess-a", i + 1, time, data, {3: t_event})
    om.train_model(config.name)

    bundle = om.load_bundle(config.name)
    assert bundle["columns"] == config.column_names()
    t_true = 4.2
    labels = _labels_df([_state_row(5, t_true - 1.0, t_true)])
    time, data = om.extract_model_features(XarrayLoader(_noise_ds(99)), config, labels=labels)
    assert abs(om.predict_events(bundle, time, data)[3].time - t_true) <= 0.2
