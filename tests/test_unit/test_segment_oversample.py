"""``train.oversample``: a metadata column becomes per-sample draw weights."""

from __future__ import annotations

import pandas as pd
import pytest

from ethograph.io.session_layout import metadata_path
from ethograph.segment.config import OversampleConfig, SegmentConfig, TrainConfig
from ethograph.segment.roles import sample_weights


def _config(**kwargs) -> SegmentConfig:
    return SegmentConfig(sessions=[], train=TrainConfig(oversample=OversampleConfig(**kwargs)))


def _index(source, trials) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": [f"s1_t{t}_a" for t in trials],
            "source": [str(source)] * len(trials),
            "trial": trials,
            "individual": ["a"] * len(trials),
        }
    )


@pytest.fixture
def session(tmp_path):
    src = tmp_path / "s1" / "s1.nc"
    src.parent.mkdir()
    src.write_bytes(b"")
    pd.DataFrame({"trial": [1, 2, 3], "difficulty": ["hard", "normal", None]}).to_csv(
        metadata_path(src), sep="\t", index=False
    )
    return src


def test_weights_follow_the_sidecar_column(session):
    weights = sample_weights(_config(column="difficulty", weights={"hard": 3.0}), _index(session, [1, 2, 3]))
    # A blank value and an unnamed value both keep weight 1.
    assert weights == {"s1_t1_a": 3.0, "s1_t2_a": 1.0, "s1_t3_a": 1.0}


def test_no_column_means_no_weighting(session):
    assert set(sample_weights(_config(), _index(session, [1, 2])).values()) == {1.0}


def test_a_missing_column_is_an_error_naming_the_session(session):
    with pytest.raises(ValueError, match="s1.*train.oversample.column='mood'"):
        sample_weights(_config(column="mood", weights={"bad": 2.0}), _index(session, [1]))


def test_config_refuses_weights_without_a_column_and_non_positive_weights():
    with pytest.raises(ValueError, match="column"):
        OversampleConfig(weights={"hard": 2.0})
    with pytest.raises(ValueError, match="positive"):
        OversampleConfig(column="difficulty", weights={"hard": 0})
    assert OversampleConfig(column="difficulty", weights={"hard": 2}).weights == {"hard": 2.0}
