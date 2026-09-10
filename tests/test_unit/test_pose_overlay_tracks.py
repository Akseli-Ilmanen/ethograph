"""The pose overlay's track table: one row per (individual, keypoint).

Regression: the table was built by joining the two names with a NUL byte and
splitting again. pandas 2.3 with pyarrow-backed strings drops the NUL, so the
split gave one part, the DataFrame constructor raised, and no pose was drawn
anywhere in the GUI — bounding boxes included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ethograph.gui.pose_overlay import _tracks_from_properties


def test_tracks_are_keyed_by_both_columns():
    props = pd.DataFrame(
        {
            "individual": pd.array(["a", "a", "b", "b"], dtype="string[pyarrow]"),
            "keypoint": pd.array(["nose", "tail", "nose", "tail"], dtype="string[pyarrow]"),
        }
    )
    codes, table = _tracks_from_properties(props)

    assert list(codes) == [0, 1, 2, 3]
    assert table["individual"].tolist() == ["a", "a", "b", "b"]
    assert table["keypoint"].tolist() == ["nose", "tail", "nose", "tail"]


def test_boxes_have_no_keypoint_column():
    props = pd.DataFrame({"individual": ["m", "f", "m"], "confidence": [0.9, 0.8, 0.7]})
    codes, table = _tracks_from_properties(props)

    assert np.array_equal(codes, [0, 1, 0])
    assert table["individual"].tolist() == ["m", "f"]
    assert table["keypoint"].tolist() == ["", ""]
