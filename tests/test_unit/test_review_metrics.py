"""Post-curation review: predictions against curated labels, and the hard flag.

Pure pandas — ``labels/review_metrics.py`` with no Qt involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ethograph.labels import onset_curves as oc
from ethograph.labels import review_metrics as rm
from ethograph.labels.intervals import (
    EVENT_TYPE_POINT,
    EVENT_TYPE_STATE,
    LABELING_AUTOMATED,
    LABELING_CURATED,
    LABELING_MANUAL,
    NO_RECIPIENT,
)
from ethograph.labels.tsv_store import save_labels_tsv


def _rows(spec, *, method=LABELING_MANUAL, individual="a", source=""):
    """``spec``: (trial, label, onset, offset | None) tuples → a labels frame."""
    rows = []
    for trial, label, onset, offset in spec:
        rows.append(
            {
                "trial": trial,
                "labels": label,
                "onset_s": onset,
                "offset_s": np.nan if offset is None else offset,
                "event_type": EVENT_TYPE_POINT if offset is None else EVENT_TYPE_STATE,
                "individual": individual,
                "individual_rec": NO_RECIPIENT,
                "confidence": 0.9,
                "labeling_method": method,
                "prediction_source": source,
            }
        )
    columns = [
        "trial",
        "labels",
        "onset_s",
        "offset_s",
        "event_type",
        "individual",
        "individual_rec",
        "confidence",
        "labeling_method",
        "prediction_source",
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_state_match_is_iou_at_half():
    pred = _rows([(1, 1, 0.0, 1.0), (1, 1, 5.0, 6.0)])
    final = _rows([(1, 1, 0.2, 1.2), (1, 1, 5.8, 6.8)])  # IoU 0.67 and 0.11
    c = rm.match_intervals(pred, final)
    assert (c.tp, c.fp, c.fn) == (1, 1, 1)
    assert c.f1 == pytest.approx(0.5)


def test_class_and_subject_must_agree():
    pred = _rows([(1, 1, 0.0, 1.0)])
    other_class = _rows([(1, 2, 0.0, 1.0)])
    other_actor = _rows([(1, 1, 0.0, 1.0)], individual="b")
    assert rm.match_intervals(pred, other_class).tp == 0
    assert rm.match_intervals(pred, other_actor).tp == 0


def test_a_final_label_is_matched_once():
    pred = _rows([(1, 1, 0.0, 1.0), (1, 1, 0.1, 1.1)])
    final = _rows([(1, 1, 0.0, 1.0)])
    c = rm.match_intervals(pred, final)
    assert (c.tp, c.fp, c.fn) == (1, 1, 0)


def test_point_match_within_tolerance():
    pred = _rows([(1, 1, 1.00, None), (1, 1, 5.00, None)])
    final = _rows([(1, 1, 1.04, None), (1, 1, 5.20, None)])
    c = rm.match_points(pred, final, tolerance_s=0.05)
    assert (c.tp, c.fp, c.fn) == (1, 1, 1)
    with pytest.raises(ValueError):
        rm.match_points(pred, final, tolerance_s=0.0)


def test_nothing_on_either_side_has_no_score():
    assert rm.Counts().f1 is None
    empty = _rows([])
    assert rm.match_intervals(empty, empty).total == 0


def test_review_trial_splits_by_event_type_and_needs_a_tolerance_for_points():
    pred = _rows([(1, 1, 0.0, 1.0), (1, 2, 3.0, None)])
    final = _rows([(1, 1, 0.0, 1.0), (1, 2, 3.01, None)])
    with_tol = rm.review_trial(1, Path("run"), pred, final, tolerance_s=0.05)
    assert with_tol.f1_state == 1.0 and with_tol.f1_point == 1.0
    without = rm.review_trial(1, Path("run"), pred, final, tolerance_s=None)
    assert without.f1_state == 1.0 and without.point is None and without.f1_point is None
    # A trial with no point events needs no tolerance to be fully scored.
    states_only = rm.review_trial(1, Path("run"), pred.iloc[:1], final.iloc[:1], tolerance_s=None)
    assert states_only.f1_point is None and states_only.point is not None


# ---------------------------------------------------------------------------
# Which run, at what tolerance
# ---------------------------------------------------------------------------


def _write_run(session: Path, timestamp: str, rows: pd.DataFrame, inference: dict, model="lightgbm", config=None):
    folder = oc.run_dir(session, timestamp, model=model)
    folder.mkdir(parents=True)
    save_labels_tsv(folder / f"{session.stem}{oc.PREDICTIONS_SUFFIX}", rows)
    oc.write_provenance(folder, model_config=config or {}, inference=inference)
    return folder


@pytest.fixture
def session(tmp_path):
    src = tmp_path / "s1" / "s1.nc"
    src.parent.mkdir()
    src.write_bytes(b"")
    return src


def test_tolerance_comes_from_the_run(session):
    declared = _write_run(session, "20260101_000001", _rows([]), {"model": "lightgbm", "tolerance_s": 0.08})
    legacy_lightgbm = _write_run(
        session, "20260101_000002", _rows([]), {"model": "lightgbm"}, config={"tolerance_s": 0.05}
    )
    legacy_spot = _write_run(
        session, "20260101_000003", _rows([]), {"model": "spot", "infer": {"focus_window_ms": 100.0}}, model="spot"
    )
    segment = _write_run(session, "20260101_000004", _rows([]), {"model": "segment", "tolerance_s": None})
    assert rm.run_tolerance_s(declared) == pytest.approx(0.08)
    assert rm.run_tolerance_s(legacy_lightgbm) == pytest.approx(0.05)
    assert rm.run_tolerance_s(legacy_spot) == pytest.approx(0.05)
    assert rm.run_tolerance_s(segment) is None


def test_resolve_run_by_source_then_newest_holding_the_trial(session):
    older = _write_run(session, "20260101_000001", _rows([(1, 1, 0.0, None)]), {"prediction_source": "lightgbm:a"})
    newer = _write_run(session, "20260101_000002", _rows([(2, 1, 0.0, None)]), {"prediction_source": "lightgbm:b"})
    assert rm.resolve_run(session, 1, "lightgbm:a") == older
    assert rm.resolve_run(session, 2, "lightgbm:b") == newer
    # No declared source (a trial predicted before runs said who they were):
    # the newest run holding the trial.
    assert rm.resolve_run(session, 1, None) == older
    assert rm.resolve_run(session, 2, "") == newer
    assert rm.resolve_run(session, 3, None) is None


def test_review_session_scores_each_trial_against_its_own_run(session):
    _write_run(
        session,
        "20260101_000001",
        _rows([(1, 1, 1.0, None), (1, 1, 4.0, None), (2, 2, 0.0, 1.0)], method=LABELING_AUTOMATED),
        {"prediction_source": "lightgbm:m", "tolerance_s": 0.05},
    )
    # Trial 1: one point accepted, one deleted, one added by hand → tp 1, fp 1, fn 1.
    # Trial 2: the state label moved slightly (IoU 0.8) → accepted. Trial 3
    # was labelled by hand alone and is not reviewed.
    final = pd.concat(
        [
            _rows([(1, 1, 1.02, None)], method=LABELING_CURATED, source="lightgbm:m"),
            _rows([(1, 1, 7.0, None)], method=LABELING_MANUAL, source="lightgbm:m"),
            _rows([(2, 2, 0.1, 1.1)], method=LABELING_MANUAL, source="lightgbm:m"),
            _rows([(3, 1, 2.0, None)], method=LABELING_MANUAL),
        ],
        ignore_index=True,
    )
    reviews = rm.review_session(session, final, [1, 2, 3])
    assert set(reviews) == {"1", "2"}
    assert reviews["1"].f1_point == pytest.approx(0.5)
    assert reviews["1"].f1_state is None
    assert reviews["2"].f1_state == 1.0
    assert reviews["2"].tolerance_s == pytest.approx(0.05)
    # The override replaces the run's tolerance: at 1 ms the moved point misses.
    strict = rm.review_session(session, final, [1], tolerance_override_s=0.001)
    assert strict["1"].point is not None and strict["1"].point.tp == 0


# ---------------------------------------------------------------------------
# Into the metadata table
# ---------------------------------------------------------------------------


def _review(trial, state=None, point=None):
    """*state*/*point*: (tp, fp, fn) or ``None`` for nothing of that type."""
    counts = rm.Counts(*state) if state else rm.Counts()
    return rm.TrialReview(str(trial), Path("run"), counts, rm.Counts(*point) if point else None, 0.05)


def _scored_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": [1, 2, 3, 4],
            rm.REVIEW_F1_STATE: [0.9, 0.9, np.nan, ""],
            rm.REVIEW_F1_POINT: [np.nan, 0.2, 0.7, ""],
        }
    )


def test_trials_below_f1_on_either_event_type():
    assert rm.trials_below_f1(_scored_table(), 0.5) == {"2"}
    assert rm.trials_below_f1(_scored_table(), 0.95) == {"1", "2", "3"}
    assert rm.trials_below_f1(_scored_table(), 0.0) == set()
    assert rm.trials_below_f1(pd.DataFrame({"trial": [1]}), 0.5) == set()


def test_f1_scores_and_scored_count_skip_blanks():
    scores = rm.f1_scores(_scored_table())
    assert scores[rm.REVIEW_F1_STATE] == [0.9, 0.9]
    assert scores[rm.REVIEW_F1_POINT] == [pytest.approx(0.2), pytest.approx(0.7)]
    assert rm.scored_trial_count(_scored_table()) == 3
    assert rm.scored_trial_count(None) == 0


def test_trial_confidence_means_skip_trials_without_a_curve():
    means = rm.trial_confidence_means({"0": np.array([0.9, 0.7]), "1": None, "2": np.array([])})
    assert means == {"0": pytest.approx(0.8)}


def test_flag_share_note_warns_past_a_quarter():
    assert "3 of 12" in rm.flag_share_note(3, 12, 0.5)
    assert "more than" not in rm.flag_share_note(3, 12, 0.5)
    assert "more than" in rm.flag_share_note(4, 12, 0.5)
    assert rm.flag_share_note(0, 0, 0.5) == "No scored trials."


def test_auto_flagging_never_clears_a_hand_set_flag():
    mdf = pd.DataFrame({"trial": [1, 2, 3, 4], rm.DIFFICULTY_COLUMN: ["hard", "normal", "", np.nan]})
    values = rm.difficulty_values(mdf, hard={"3"}, trials=[1, 2, 3, 4])
    # 1 stays hard without a write; 2 already normal; 3 flagged; 4 blank → normal.
    assert values == {"3": rm.DIFFICULTY_HARD, "4": rm.DIFFICULTY_NORMAL}
    assert rm.difficulty_values(None, hard={"1"}, trials=[1, 2]) == {"1": "hard", "2": "normal"}


def test_review_columns_and_summary():
    reviews = {"1": _review(1, state=(9, 2, 0)), "2": _review(2, point=(1, 8, 0))}
    cols = rm.review_columns(reviews)
    assert cols[rm.REVIEW_F1_STATE]["1"] == pytest.approx(0.9)
    assert np.isnan(cols[rm.REVIEW_F1_STATE]["2"])
    assert cols[rm.REVIEW_F1_POINT]["2"] == pytest.approx(0.2)
    text = rm.summary(reviews)
    assert "Scored 2" in text and "state F1@50" in text and "point F1" in text
    assert "no trial" in rm.summary({})


def test_derived_columns_never_reach_an_nwb():
    from ethograph.io.metadata_edit import DERIVED_COLUMNS

    assert rm.DIFFICULTY_COLUMN in DERIVED_COLUMNS
    assert set(rm.REVIEW_COLUMNS) <= DERIVED_COLUMNS


def test_prediction_run_dirs_need_a_predictions_file(session):
    with_labels = _write_run(session, "20260101_000001", _rows([]), {})
    curves_only = oc.run_dir(session, "20260101_000002")
    curves_only.mkdir(parents=True)
    (curves_only / oc.CURVES_FILE).write_bytes(b"")
    assert oc.prediction_run_dirs(session) == [with_labels]
    assert oc.predictions_file(curves_only) is None
    assert yaml.safe_load((with_labels / oc.INFERENCE_FILE).read_text()) == {}
