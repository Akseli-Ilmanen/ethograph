"""The add-panel popup's "Video features" section: every folder-attached
feature with its folder (and nowhere else, so two exports stay tellable
apart), then the browse entry that adds another."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("qtpy")

from ethograph.gui.source_popup import _ROLE_HEADER, _ROLE_NAME, VIDEO_FEATURES_BROWSE, SourcePopup  # noqa: E402
from ethograph.io.catalog import ComboSpec, DataCatalog  # noqa: E402
from ethograph.io.video_feature_files import ATTACHED_FROM  # noqa: E402


def _rows(popup: SourcePopup) -> list[tuple[str, bool]]:
    return [
        (popup._list.item(i).text().strip(), bool(popup._list.item(i).data(_ROLE_HEADER)))
        for i in range(popup._list.count())
    ]


def test_attached_embedding_is_listed_with_its_folder(qapp):
    ds = xr.Dataset(
        {
            "speed": ("time", np.zeros(3)),
            "feral_cfgA": (("time", "feral_cfgA_dims"), np.zeros((3, 2)), {ATTACHED_FROM: "D:/emb/cfgA"}),
        }
    )
    state = SimpleNamespace(ds=ds, nwb_alignment=None)
    catalog = DataCatalog(combos={"features": ComboSpec("features", ("speed", "feral_cfgA"))})
    popup = SourcePopup(state)

    popup.refresh(catalog=catalog)

    rows = _rows(popup)
    features_at = rows.index(("Features", True))
    video_at = rows.index(("Video features", True))
    assert rows[features_at + 1 : video_at] == [("speed", False)]
    assert rows[video_at + 1] == ("feral_cfgA  (D:/emb/cfgA)", False)
    names = [popup._list.item(i).data(_ROLE_NAME) for i in range(popup._list.count())]
    assert names.count("feral_cfgA") == 1
    assert names[video_at + 2] == VIDEO_FEATURES_BROWSE


def test_the_section_always_offers_to_browse(qapp):
    state = SimpleNamespace(ds=xr.Dataset({"speed": ("time", np.zeros(3))}), nwb_alignment=None)
    popup = SourcePopup(state)
    popup.refresh(catalog=DataCatalog(combos={"features": ComboSpec("features", ("speed",))}))
    rows = _rows(popup)
    video_at = rows.index(("Video features", True))
    names = [popup._list.item(i).data(_ROLE_NAME) for i in range(popup._list.count())]
    assert names[video_at + 1] == VIDEO_FEATURES_BROWSE
    assert popup._list.item(video_at + 1).data(_ROLE_HEADER) is False
