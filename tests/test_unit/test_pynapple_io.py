"""Tests for ethograph.io.pynapple loading and DataLoader access."""

import numpy as np
import pynapple as nap
import pytest

from ethograph.io import schema
from ethograph.io.catalog import (
    PynappleLoader as PynappleStore,
)
from ethograph.io.catalog import (
    _column_axes,
    catalog_from_pynapple,
)
from ethograph.io.pynapple import detect_trials

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_nap_data():
    """Pynapple data dict with speed (Tsd), velocity (TsdFrame), and trials."""
    trials = nap.IntervalSet(start=[0, 15], end=[10, 25])
    t = np.linspace(0, 25, 2500)
    speed = nap.Tsd(t=t, d=np.random.randn(len(t)), time_support=trials)
    velocity = nap.TsdFrame(
        t=t,
        d=np.random.randn(len(t), 3),
        columns=["x", "y", "z"],
        time_support=trials,
    )
    return {"speed": speed, "velocity": velocity, "trials": trials}


@pytest.fixture
def multi_tsdframe_data():
    """Multiple TsdFrames sharing the same columns (x, y, z)."""
    t = np.linspace(0, 10, 1000)
    return {
        "position": nap.TsdFrame(t=t, d=np.random.randn(1000, 3), columns=["x", "y", "z"]),
        "velocity": nap.TsdFrame(t=t, d=np.random.randn(1000, 3), columns=["x", "y", "z"]),
        "speed": nap.Tsd(t=t, d=np.random.randn(1000)),
    }


# ---------------------------------------------------------------------------
# detect_trials
# ---------------------------------------------------------------------------


def test_detect_trials_found(simple_nap_data):
    ep = detect_trials(simple_nap_data)
    assert ep is not None
    assert len(ep) == 2


def test_detect_trials_not_found():
    data = {"speed": nap.Tsd(t=np.arange(10), d=np.zeros(10))}
    assert detect_trials(data) is None


# ---------------------------------------------------------------------------
# Shared column dimensions
# ---------------------------------------------------------------------------


def test_shared_columns_merged():
    """TsdFrames with identical columns produce one shared dimension name."""
    t = np.linspace(0, 10, 100)
    objs = {
        "position": nap.TsdFrame(t=t, d=np.random.randn(100, 2), columns=["ch1", "ch2"]),
        "velocity": nap.TsdFrame(t=t, d=np.random.randn(100, 2), columns=["ch1", "ch2"]),
    }
    axes = _column_axes(objs)
    assert axes["position"].dim == axes["velocity"].dim == "columns"


def test_xyz_columns_are_the_space_dim():
    """x/y/z is movement's `space` dim on either backend, so a selection means
    the same thing whichever one the session came from."""
    t = np.linspace(0, 10, 100)
    objs = {
        "position": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["x", "y", "z"]),
        "velocity": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["x", "y", "z"]),
    }
    axes = _column_axes(objs)
    assert axes["position"].dim == axes["velocity"].dim == "space"
    assert axes["position"].labels == ("x", "y", "z")


def test_different_columns_stay_separate():
    """TsdFrames with different columns get separate dimension names."""
    t = np.linspace(0, 10, 100)
    objs = {
        "position": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["x", "y", "z"]),
        "emg": nap.TsdFrame(t=t, d=np.random.randn(100, 2), columns=["ch1", "ch2"]),
    }
    axes = _column_axes(objs)
    assert axes["position"].dim != axes["emg"].dim


def test_single_tsdframe_keeps_prefixed_dim():
    """A lone TsdFrame gets '{name}_columns' (no merging needed)."""
    t = np.linspace(0, 10, 100)
    objs = {
        "velocity": nap.TsdFrame(t=t, d=np.random.randn(100, 3), columns=["a", "b", "c"]),
    }
    axes = _column_axes(objs)
    assert axes["velocity"].dim == "velocity_columns"


def test_tsdtensor_gets_a_column_axis():
    """A tensor's flattened axis is selectable like any other, so a panel can
    pin one column instead of being stuck on all of them."""
    t = np.linspace(0, 10, 100)
    objs = {"frames": nap.TsdTensor(t=t, d=np.random.randn(100, 2, 3))}
    axis = _column_axes(objs)["frames"]
    assert axis.dim == "frames_columns"
    assert axis.labels == ("0", "1", "2", "3", "4", "5")


# ---------------------------------------------------------------------------
# catalog_from_pynapple (replaces extract_type_vars_pynapple)
# ---------------------------------------------------------------------------


def test_catalog_basic(simple_nap_data):
    cat = catalog_from_pynapple(simple_nap_data)
    assert "speed" in cat.features
    assert "velocity" in cat.features
    assert "individual" in cat.combos  # dim-named, movement-style singular


def test_catalog_detects_tsdframe_columns(simple_nap_data):
    cat = catalog_from_pynapple(simple_nap_data)
    assert "space" in cat.combos
    assert list(cat.combo_values("space")) == ["x", "y", "z"]


def test_catalog_shared_columns(multi_tsdframe_data):
    """Shared columns produce one combo instead of two."""
    cat = catalog_from_pynapple(multi_tsdframe_data)
    assert "space" in cat.combos
    assert "position_columns" not in cat.combos
    assert "velocity_columns" not in cat.combos
    assert list(cat.combo_values("space")) == ["x", "y", "z"]


def test_catalog_combos_match_loader_dims(multi_tsdframe_data):
    """Every combo the catalog offers is one `select()` actually reads.

    The combo name, `feature_dims()` and the key `select()` looks up used to
    be decided in three places; a combo named differently from the dim the
    loader reads is one the "All" checkbox cannot free.
    """
    cat = catalog_from_pynapple(multi_tsdframe_data)
    store = PynappleStore(multi_tsdframe_data, cat)
    for feature in cat.features:
        for dim in store.feature_dims(feature):
            assert dim in cat.combos, f"{feature}: dim {dim!r} has no combo"


def test_catalog_detects_changepoints():
    cp_times = np.array([10.0, 25.0, 50.0, 75.0])
    group = nap.TsGroup({0: nap.Ts(t=cp_times)})
    group.set_info(**schema.changepoint_metadata(1))
    data = {"cp_group": group}
    cat = catalog_from_pynapple(data)
    assert "cp_group" in cat.changepoints


def test_catalog_skips_intervalset(simple_nap_data):
    cat = catalog_from_pynapple(simple_nap_data)
    assert "trials" not in cat.features


# ---------------------------------------------------------------------------
# PynappleStore
# ---------------------------------------------------------------------------


def test_store_features(multi_tsdframe_data):
    store = PynappleStore(multi_tsdframe_data)
    assert set(store.features) == {"position", "velocity", "speed"}


def test_store_dims_shared(multi_tsdframe_data):
    store = PynappleStore(multi_tsdframe_data)
    assert "space" in store.dims
    assert list(store.dims["space"]) == ["x", "y", "z"]
    assert "individual" in store.dims


def test_store_get_type_vars(multi_tsdframe_data):
    store = PynappleStore(multi_tsdframe_data)
    tvd = store.get_type_vars()
    assert "features" in tvd
    assert "space" in tvd
    assert "individual" in tvd


def test_store_select_tsd(multi_tsdframe_data):
    """Selecting a Tsd feature returns 1-D data."""
    store = PynappleStore(multi_tsdframe_data)
    pd = store.select("speed", {}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.data.ndim == 1
    assert len(pd.time) == len(pd.data)


def test_store_select_tsdframe_all_columns(multi_tsdframe_data):
    """Without column selection, TsdFrame returns 2-D data."""
    store = PynappleStore(multi_tsdframe_data)
    pd = store.select("position", {}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.data.ndim == 2
    assert pd.data.shape[1] == 3
    assert pd.dim_labels == ["x", "y", "z"]


def test_store_select_tsdframe_single_column(multi_tsdframe_data):
    """With column selection, TsdFrame returns 1-D data."""
    store = PynappleStore(multi_tsdframe_data)
    pd = store.select("position", {"space": "x"}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.data.ndim == 1


def test_store_select_ignores_a_dim_the_feature_lacks(multi_tsdframe_data):
    """A selection key IS a dim of the feature or it is inert — exactly the
    xarray rule. Matching any selection *value* against the columns kept a dim
    pinned after the user had set it to "All"."""
    data = dict(multi_tsdframe_data)
    data["emg"] = nap.TsdFrame(t=data["position"].t, d=np.random.randn(1000, 2), columns=["x", "q"])
    store = PynappleStore(data)
    # 'emg_columns' belongs to emg, not to position: position's own dim is free.
    pd = store.select("position", {"emg_columns": "x", "individual": "individual_0"}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.data.shape[1] == 3
    assert pd.dim_labels == ["x", "y", "z"]


def test_store_select_numeric_column_from_combo_string(multi_tsdframe_data):
    """feature_dims() stringifies labels for the combo, so the value comes back
    as a string even for numeric columns — it must still pin."""
    data = dict(multi_tsdframe_data)
    data["emg"] = nap.TsdFrame(t=data["position"].t, d=np.random.randn(1000, 3), columns=[0, 1, 2])
    store = PynappleStore(data)
    assert store.feature_dims("emg") == {"emg_columns": ["0", "1", "2"]}
    pd = store.select("emg", {"emg_columns": "1"}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.data.ndim == 1
    assert len(pd.data) == len(data["emg"])
    np.testing.assert_allclose(pd.data, data["emg"].values[:, 1])


@pytest.fixture
def pose_nap_data(monkeypatch):
    """A pose session: two keypoint TsdFrames plus a plain feature sharing x/y."""
    import ethograph.io.catalog as catalog_mod

    t = np.linspace(0, 10, 1000)

    def _frame():
        return nap.TsdFrame(t=t, d=np.random.randn(1000, 2), columns=["x", "y"])

    data = {"nose": _frame(), "tail": _frame(), "centroid": _frame()}
    monkeypatch.setattr(catalog_mod, "_discover_pose_keypoints", lambda _p: {"nose", "tail"})
    return data, catalog_from_pynapple(data, source_path="pose.nwb")


def test_pose_column_dim_is_the_shared_space_dim(pose_nap_data):
    """The keypoints' columns are the same axis a plain x/y feature has, so one
    combo serves both — two combos over one axis is what let the pinned one win
    while the user was clicking "All" on the other."""
    data, cat = pose_nap_data
    store = PynappleStore(data, cat)
    assert store.feature_dims("pose_estimation") == {"space": ["x", "y"], "keypoint": ["nose", "tail"]}
    assert store.feature_dims("centroid") == {"space": ["x", "y"]}
    assert "columns" not in cat.combos


def test_pose_all_columns_with_keypoint_pinned(pose_nap_data):
    """Keypoint pinned, column dim on "All" → one curve per column."""
    data, cat = pose_nap_data
    store = PynappleStore(data, cat)
    pinned = store.select("pose_estimation", {"keypoint": "nose", "space": "x"}, t0=0.0, t1=5.0)
    assert pinned.data.ndim == 1
    freed = store.select("pose_estimation", {"keypoint": "nose"}, t0=0.0, t1=5.0)
    assert freed.data.shape[1] == 2
    assert freed.dim_labels == ["x", "y"]


def test_pose_all_keypoints_with_column_pinned(pose_nap_data):
    """Column pinned, keypoint dim on "All" → one curve per keypoint."""
    data, cat = pose_nap_data
    store = PynappleStore(data, cat)
    pd = store.select("pose_estimation", {"space": "x"}, t0=0.0, t1=5.0)
    assert pd.data.shape[1] == 2
    assert pd.dim_labels == ["nose", "tail"]


def test_store_select_with_time_window(multi_tsdframe_data):
    """Narrower time window restricts the returned data (time stays absolute)."""
    store = PynappleStore(multi_tsdframe_data)
    full = store.select("speed", {}, t0=0.0, t1=10.0)
    windowed = store.select("speed", {}, t0=2.0, t1=5.0)
    assert windowed is not None
    assert len(windowed.time) < len(full.time)
    assert windowed.time[0] >= 2.0 - 0.1
    assert windowed.time[-1] <= 5.0 + 0.1


def test_store_absolute_time():
    """Returned time keeps absolute session coordinates (never re-based to t0),
    so a viewport starting anywhere (e.g. a sliding fixed window) aligns with
    the plot x-axis."""
    trials = nap.IntervalSet(start=[100, 200], end=[110, 210])
    t = np.linspace(100, 210, 11000)
    speed = nap.Tsd(t=t, d=np.random.randn(len(t)), time_support=trials)
    data = {"speed": speed}

    store = PynappleStore(data)
    pd = store.select("speed", {}, t0=100.0, t1=110.0)
    assert pd is not None
    assert pd.time[0] >= 100.0 - 0.1
    assert pd.time[-1] <= 110.0 + 0.1

    pd2 = store.select("speed", {}, t0=200.0, t1=210.0)
    assert pd2 is not None
    assert pd2.time[0] >= 200.0 - 0.1


def test_store_select_nonexistent_feature(multi_tsdframe_data):
    store = PynappleStore(multi_tsdframe_data)
    assert store.select("nonexistent", {}, t0=0.0, t1=10.0) is None


def test_store_select_sparse_changepoints():
    """Sparse Ts changepoints should only mark actual CP frames, not every frame."""
    t = np.linspace(0, 10, 2000)
    speed = nap.Tsd(t=t, d=np.random.randn(2000))

    cp_times = np.array([1.0, 3.5, 7.2])
    group = nap.TsGroup({0: nap.Ts(t=cp_times)})
    group.set_info(
        **schema.changepoint_metadata(1),
        target_feature=["speed"],
        source_label=["unit_0"],
    )

    store = PynappleStore({"speed": speed, "cps": group})
    pd = store.select("speed", {}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.changepoints is not None
    cp_binary = list(pd.changepoints.values())[0]
    assert cp_binary.sum() == 3
    assert len(cp_binary) == len(pd.time)


def test_store_select_dense_tsd_changepoints():
    """Dense Tsd changepoints (legacy) should only mark non-zero frames."""
    t = np.linspace(0, 10, 2000)
    speed = nap.Tsd(t=t, d=np.random.randn(2000))

    mask = np.zeros(2000, dtype=np.float32)
    mask[100] = 1.0
    mask[500] = 1.0
    mask[1500] = 1.0
    dense_cp = nap.Tsd(t=t, d=mask)
    group = nap.TsGroup({0: dense_cp})
    group.set_info(
        **schema.changepoint_metadata(1),
        target_feature=["speed"],
        source_label=["unit_0"],
    )

    store = PynappleStore({"speed": speed, "cps": group})
    pd = store.select("speed", {}, t0=0.0, t1=10.0)
    assert pd is not None
    assert pd.changepoints is not None
    cp_binary = list(pd.changepoints.values())[0]
    assert cp_binary.sum() == 3
    assert len(cp_binary) == len(pd.time)


# ---------------------------------------------------------------------------
# get_cp_times (for changepoint correction)
# ---------------------------------------------------------------------------


def test_store_get_cp_times_sparse():
    """get_cp_times returns sparse trial-relative timestamps for correction."""
    t = np.linspace(0, 10, 2000)
    speed = nap.Tsd(t=t, d=np.random.randn(2000))

    cp_times = np.array([1.0, 3.5, 7.2])
    group = nap.TsGroup({0: nap.Ts(t=cp_times)})
    group.set_info(
        **schema.changepoint_metadata(1),
        target_feature=["speed"],
        source_label=["unit_0"],
    )

    store = PynappleStore({"speed": speed, "cps": group})
    result = store.get_cp_times("speed", t0=0.0, t1=10.0)
    assert len(result) == 3
    np.testing.assert_allclose(result, cp_times, atol=1e-6)


def test_store_get_cp_times_no_changepoints():
    t = np.linspace(0, 10, 1000)
    store = PynappleStore({"speed": nap.Tsd(t=t, d=np.random.randn(1000))})
    result = store.get_cp_times("speed", t0=0.0, t1=10.0)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Display offset (trial-local windows over absolute pynapple time)
# ---------------------------------------------------------------------------


def test_select_without_provider_is_absolute(simple_nap_data):
    """No provider installed -> pure absolute-time access, as before."""
    loader = PynappleStore(simple_nap_data)
    plot_data = loader.select("speed", {}, t0=15.0, t1=25.0)
    assert plot_data is not None
    assert plot_data.time[0] >= 15.0 - 1e-9


def test_select_with_offset_rebases_to_display_time(simple_nap_data):
    """A trial-local query is shifted into absolute time and back."""
    loader = PynappleStore(simple_nap_data)
    loader.set_display_offset_provider(lambda: 15.0)  # trial 2 starts at 15 s

    plot_data = loader.select("speed", {}, t0=0.0, t1=10.0)
    assert plot_data is not None
    assert 0.0 <= plot_data.time[0] < 0.1
    assert plot_data.time[-1] <= 10.0 + 1e-9

    # Same span queried absolute must give identical values.
    loader.set_display_offset_provider(None)
    absolute = loader.select("speed", {}, t0=15.0, t1=25.0)
    np.testing.assert_array_equal(np.asarray(plot_data.data), np.asarray(absolute.data))
    np.testing.assert_allclose(np.asarray(plot_data.time) + 15.0, np.asarray(absolute.time))


def test_offset_is_pulled_per_call(simple_nap_data):
    """The provider is consulted on every select, so trial changes need no re-sync."""
    offset = {"value": 0.0}
    loader = PynappleStore(simple_nap_data)
    loader.set_display_offset_provider(lambda: offset["value"])

    first_trial = loader.select("speed", {}, t0=0.0, t1=10.0)
    offset["value"] = 15.0
    second_trial = loader.select("speed", {}, t0=0.0, t1=10.0)
    assert first_trial is not None and second_trial is not None
    assert not np.array_equal(np.asarray(first_trial.data), np.asarray(second_trial.data))


def test_get_cp_times_with_offset():
    """Changepoint times come back in display coordinates."""
    t = np.linspace(0, 25, 2500)
    speed = nap.Tsd(t=t, d=np.random.randn(2500))
    cp_times = np.array([16.0, 18.5, 22.2])
    group = nap.TsGroup({0: nap.Ts(t=cp_times)})
    group.set_info(
        **schema.changepoint_metadata(1),
        target_feature=["speed"],
        source_label=["unit_0"],
    )

    loader = PynappleStore({"speed": speed, "cps": group})
    loader.set_display_offset_provider(lambda: 15.0)
    result = loader.get_cp_times("speed", t0=0.0, t1=10.0)
    np.testing.assert_allclose(result, cp_times - 15.0, atol=1e-6)
