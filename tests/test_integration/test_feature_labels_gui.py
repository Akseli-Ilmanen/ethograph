"""Tools ▸ Labels: Create from a feature, and its neural twin — a read-only label source from the data."""

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from ethograph.gui.dialog_feature_labels import _BURST_PARAMETERS, THRESHOLD, FeatureLabelsDialog
from ethograph.labels.feature_events import BURST_DETECTORS, LOG_ISI, MAX_INTERVAL, PERCENTILE, ZSCORE, burst_parameters
from ethograph.labels.tsv_store import labels_equal

pytestmark = pytest.mark.usefixtures("gui")


def test_a_thresholded_feature_is_walked_and_never_written_to_the_labels(moll2025_gui, qtbot):
    _viewer, meta = moll2025_gui
    state = meta.app_state
    working = state._all_labels_df.copy()

    dialog = FeatureLabelsDialog(meta)
    qtbot.addWidget(dialog)
    dialog.feature_combo.setCurrentText("speed")
    # A dim opens read value by value: every keypoint is a class of its own.
    assert dialog.query().dim == "keypoint"
    dialog.threshold_spin.setValue(float(dialog.range_label.text().split()[-1].rstrip(".")) / 4)
    dialog.stitch_spin.setValue(0.02)
    dialog.min_duration_spin.setValue(0.02)
    created = dialog.create_labels()
    QApplication.processEvents()

    assert created is not None
    names = [m["name"] for m in created.mappings.values()]
    assert names == ["keypoint_beakTip", "keypoint_stickTip", "keypoint_pellet"]
    events = created.labels_df
    assert set(events["trial"]) <= set(state.trials)
    assert ((events["offset_s"] - events["onset_s"]) >= 0.02).all()
    assert set(events["individual"]) == {"Crow1"}

    # Its own panel, one row per keypoint — and the working labels untouched.
    (panel,) = meta.plot_container.prediction_panels()
    assert set(panel.lanes) == {1, 2, 3}
    assert labels_equal(state._all_labels_df, working)

    # Navigation and the grids now read it.
    nav = meta.navigation_widget
    assert not state.label_source().writable
    assert [nav.label_combo.itemText(i) for i in range(nav.label_combo.count())] == [
        f"{i + 1} ({name})" for i, name in enumerate(names)
    ]
    curation = meta.labels_widget.curation_panel
    assert curation.scope_or_all_ids() == [1, 2, 3]
    assert curation.source_combo.currentData() == str(created.path)

    # Removing the set hands the tools back to the working labels.
    meta.labels_widget._remove_selected_prediction_set()
    QApplication.processEvents()
    assert state.label_source().writable
    assert nav.label_combo.count() == len([i for i in meta.labels_widget._mappings if i != 0])


def test_firing_rates_are_thresholded_in_every_trial_not_just_the_one_on_screen(moll2025_gui, qtbot):
    """The ``firing_rate`` feature is a snapshot of one trial; the tool bins the spikes of each."""
    nap = pytest.importorskip("pynapple")
    _viewer, meta = moll2025_gui
    state = meta.app_state
    other = next(t for t in state.trials if t != state.trials_sel)
    start = float(state.nwb_alignment.start_time(other))

    # Two units bursting at 200 Hz in the other trial, half overlapping; one silent unit.
    ew = meta.ephys_widget
    bursts = {0: (0.5, 1.0), 5: (0.75, 1.25)}
    trains = {uid: np.arange(start + a, start + b, 0.005) for uid, (a, b) in bursts.items()}
    group = nap.TsGroup({**{uid: nap.Ts(t=t) for uid, t in trains.items()}, 9: nap.Ts(t=np.array([start + 0.1]))})
    ew._tsgroup = group
    ew._neurons_source = "pynapple"
    ew._cluster_df = ew._build_cluster_df_from_tsgroup(group)
    ew._populate_cluster_table(ew._cluster_df)
    ew.fr_bin_spin.setValue(0.01)
    ew.fr_sigma_spin.setValue(0.0)

    dialog = FeatureLabelsDialog(meta, neural=True)
    qtbot.addWidget(dialog)
    assert dialog.feature_combo.currentText() == "firing_rate"
    assert dialog.query().dim == "unit"
    dialog.threshold_spin.setValue(100.0)
    created = dialog.create_labels()

    assert created is not None
    names = {lid: m["name"] for lid, m in created.mappings.items()}
    assert names == {1: "unit_0", 2: "unit_5", 3: "unit_9"}
    events = created.labels_df
    assert set(events["trial"]) == {other}, "the bursts were found in the wrong trial"
    found = {names[int(r.labels)]: (r.onset_s, r.offset_s) for r in events.itertuples()}
    assert set(found) == {"unit_0", "unit_5"}
    for unit, (a, b) in bursts.items():
        assert np.allclose(found[f"unit_{unit}"], (a, b), atol=0.02), "not on the trial's own clock"


def test_the_distribution_is_drawn_on_the_scale_the_threshold_reads(moll2025_gui, qtbot):
    _viewer, meta = moll2025_gui
    dialog = FeatureLabelsDialog(meta)
    qtbot.addWidget(dialog)
    dialog.feature_combo.setCurrentText("speed")

    dialog.show_distribution()
    assert not dialog.histogram.isHidden()
    raw_edges = dialog.histogram.viewRange()[0]

    # Z-scored, the same values sit around 0 — and the threshold is read in z.
    dialog.scale_combo.setCurrentIndex(dialog.scale_combo.findData(ZSCORE))
    z_edges = dialog.histogram.viewRange()[0]
    assert z_edges[0] < 0 < z_edges[1] and z_edges != raw_edges
    assert dialog.rule().describe() == ">2z", "a z-score opens at 2 standard deviations"

    # The line is the threshold: dragging it sets the spin and the share that passes.
    dialog.threshold_line.setValue(2.0)
    assert dialog.threshold_spin.value() == 2.0
    share = float(dialog.passing_label.text().split("%")[0])
    assert 0.0 < share < 50.0

    created = dialog.create_labels()
    assert created is not None and set(created.labels_df["labels"]) <= {1, 2, 3}

    # A percentile is each keypoint's own cut: no one line on a pooled axis, and the top 5 % of each.
    dialog.scale_combo.setCurrentIndex(dialog.scale_combo.findData(PERCENTILE))
    assert dialog.rule().describe() == ">95%"
    assert not dialog.threshold_line.isVisible()
    assert dialog.passing_label.text().startswith("5.0% of each series")
    assert "across series" in dialog.passing_label.text()
    events = dialog.create_labels().labels_df
    assert {1, 2} <= set(events["labels"])

    # Another selection: the drawn values are no longer what the threshold is compared against.
    dialog._dim_combos["keypoint"].setCurrentIndex(1)
    assert dialog.histogram.isHidden()


def test_the_instantaneous_rate_finds_a_burst_from_its_first_spike_to_its_last(moll2025_gui, qtbot):
    nap = pytest.importorskip("pynapple")
    _viewer, meta = moll2025_gui
    state = meta.app_state
    other = next(t for t in state.trials if t != state.trials_sel)
    start = float(state.nwb_alignment.start_time(other))

    # 200 Hz from 0.5 to 1.0 s of the other trial, between two lone spikes a second away.
    burst = start + np.arange(0.5, 1.0001, 0.005)
    ew = meta.ephys_widget
    group = nap.TsGroup({3: nap.Ts(t=np.concatenate([[start - 0.5], burst, [start + 2.0]]))})
    ew._tsgroup = group
    ew._neurons_source = "pynapple"
    ew._cluster_df = ew._build_cluster_df_from_tsgroup(group)
    ew._populate_cluster_table(ew._cluster_df)
    ew.fr_bin_spin.setValue(0.005)

    dialog = FeatureLabelsDialog(meta, neural=True)
    qtbot.addWidget(dialog)
    dialog.scale_combo.setCurrentIndex(dialog.scale_combo.findData(ZSCORE))
    dialog.feature_combo.setCurrentText("instantaneous_rate")
    # It opens at the birdsong burst criterion: above 125 Hz, in the rate's own units.
    assert dialog.rule().describe() == ">125"
    events = dialog.create_labels().labels_df

    # No smoothing to blur the edges: the period is the burst, to within a grid step.
    assert events["trial"].tolist() == [other]
    assert np.allclose(events[["onset_s", "offset_s"]].to_numpy(), [[0.5, 1.0]], atol=0.006)


def test_a_burst_detector_replaces_the_threshold_and_reads_the_spikes_of_every_trial(moll2025_gui, qtbot):
    nap = pytest.importorskip("pynapple")
    _viewer, meta = moll2025_gui
    state = meta.app_state

    other = next(t for t in state.trials if t != state.trials_sel)
    start = float(state.nwb_alignment.start_time(other))
    ew = meta.ephys_widget
    trains = {0: start + np.arange(0.5, 1.0001, 0.005), 5: start + np.arange(0.75, 1.2501, 0.005)}
    group = nap.TsGroup({**{uid: nap.Ts(t=t) for uid, t in trains.items()}, 9: nap.Ts(t=np.array([start + 0.1]))})
    ew._tsgroup = group
    ew._neurons_source = "pynapple"
    ew._cluster_df = ew._build_cluster_df_from_tsgroup(group)
    ew._populate_cluster_table(ew._cluster_df)

    # Nothing to review per unit yet: the area under the cluster table is off, and rows cannot be dragged.
    area = ew.unit_events
    assert not area.isEnabled() and not ew.cluster_table.dragEnabled()

    # The feature dialog stays about the session's features: no spike rates, no method to pick.
    plain = FeatureLabelsDialog(meta)
    qtbot.addWidget(plain)
    features = [plain.feature_combo.itemText(i) for i in range(plain.feature_combo.count())]
    assert "speed" in features and not set(features) & {"firing_rate", "instantaneous_rate"}
    assert plain.method_row.isHidden() and plain.method() == THRESHOLD

    # The neural one reads only the units: a rate to threshold, or a burst detector.
    dialog = FeatureLabelsDialog(meta, neural=True)
    qtbot.addWidget(dialog)
    assert [dialog.feature_combo.itemText(i) for i in range(dialog.feature_combo.count())] == [
        "firing_rate",
        "instantaneous_rate",
    ]
    assert dialog.method() == THRESHOLD and dialog.burst_group.isHidden()
    dialog.method_combo.setCurrentIndex(dialog.method_combo.findData(MAX_INTERVAL))
    # The form is now the detector's: its parameters at their defaults, no threshold to set.
    assert dialog.rule_group.isHidden() and dialog.feature_group.isHidden() and not dialog.burst_group.isHidden()
    assert dialog.burst_rule().params == burst_parameters(MAX_INTERVAL)
    assert dialog.name_edit.text() == "bursts max_interval (per unit)"
    created = dialog.create_labels()

    assert created is not None
    names = {lid: m["name"] for lid, m in created.mappings.items()}
    assert names == {1: "unit_0", 2: "unit_5", 3: "unit_9"}
    events = created.labels_df
    assert set(events["trial"]) == {other}, "the bursts were found in the wrong trial"
    found = {names[int(r.labels)]: (r.onset_s, r.offset_s) for r in events.itertuples()}
    # First spike to last spike, exactly: no bin grid in between.
    assert set(found) == {"unit_0", "unit_5"}
    assert np.allclose(found["unit_0"], (0.5, 1.0)) and np.allclose(found["unit_5"], (0.75, 1.25))
    assert not state.label_source().writable

    # The set has a class per unit, so units can now be dragged in to pick what the grids show.
    assert area.isEnabled() and ew.cluster_table.dragEnabled()
    assert area.label_ids() == [1, 2, 3], "empty means every unit of the set"
    # Selecting a unit in the table is enough: the area holds the selection.
    ew.cluster_table.selectRow(1)
    assert area.units() == ["5"] and area.label_ids() == [2]
    # The grids read the label source: opening one from here points it back at this set.
    state.label_source_path = None
    assert area.unit_set() is created
    state.label_source_path = str(created.path)

    # The set's panel shows the raster's rows: a unit filtered out of the neuron table loses its row.
    (panel,) = meta.plot_container.prediction_panels()
    meta.data_widget.update_label_plot()
    assert list(panel.lanes) == [1, 2, 3]
    spikes_col = ew._find_col_by_header("", exact="n_spikes")
    ew._cluster_proxy.set_numeric_filter(spikes_col, ">=", 50)
    ew._on_unit_filter_changed()
    meta.data_widget.update_label_plot()
    assert ew.firing_rate_units() == ["0", "5"] and list(panel.lanes) == [1, 2]
    area.reset()
    assert area.label_ids() == [1, 2], "empty means the filtered units"

    # A unit selected in the table is marked in the panel, on its own row.
    ew.cluster_table.selectRow(1)
    assert [band.getRegion() for band in panel._row_highlights] == [panel.lanes[2]]
    assert area.units() == ["5"]
    # Reset: nothing selected in the table, marked in a panel or held in the area.
    area.reset_btn.click()
    assert panel._row_highlights == [] and ew.selected_unit_ids() == [] and area.units() == []
    ew._cluster_proxy.set_numeric_filter(spikes_col, None, None)
    ew._on_unit_filter_changed()

    # logISI on one unit: a single interval length has no valley, so the fallback decides.
    dialog.method_combo.setCurrentIndex(dialog.method_combo.findData(LOG_ISI))
    dialog.unit_combo.setCurrentIndex(dialog.unit_combo.findData("5"))
    assert dialog.name_edit.text() == "bursts log_isi [unit=5]"
    assert dialog.create_labels().labels_df["labels"].tolist() == [1]
    dialog._burst_inputs["fallback"].setChecked(False)
    assert dialog.create_labels() is None


def test_every_burst_parameter_has_a_row_to_show_it_in():
    for method in BURST_DETECTORS:
        assert set(burst_parameters(method)) <= set(_BURST_PARAMETERS), method


def test_a_set_without_unit_classes_leaves_the_unit_area_off(moll2025_gui, qtbot):
    _viewer, meta = moll2025_gui
    dialog = FeatureLabelsDialog(meta)
    qtbot.addWidget(dialog)
    dialog.feature_combo.setCurrentText("speed")
    dialog.scale_combo.setCurrentIndex(dialog.scale_combo.findData(PERCENTILE))
    assert dialog.create_labels() is not None
    assert not meta.ephys_widget.unit_events.isEnabled()
