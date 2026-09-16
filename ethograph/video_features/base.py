"""What every extractor has in common: the sidecar, the registry, the crop.

An **extractor** turns one video into one **video feature** — a
``(time_video, {name}_dims)`` DataArray on the video's own clock (frame 0 at
t = 0, every ``step``-th frame), stamped ``kind="video_feature"``. Which
network made it is the extractor's *name* (``s3d``, ``timm``); the file it
lands in is ``{video stem}_{name}.nc`` and, once merged onto a trial, the
variable is called ``{name}``.

Extractors are looked up by name in :data:`EXTRACTORS` and imported only
when built, so a missing optional package (``timm``) is an error at the call
that needs it, naming the extra to install — never at import of this package.
Torch is deliberately not imported here: the segmentation config reads this
module to validate a project file.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np
import xarray as xr

from ethograph.io.schema import VIDEO_FEATURE, describe
from ethograph.utils.xr_utils import get_time_coord

logger = logging.getLogger(__name__)

#: Time dim of every sidecar — the video's own clock.
TIME_DIM = "time_video"

#: ``name -> (module, class)``; the module is imported when the extractor is built.
EXTRACTORS: dict[str, tuple[str, str]] = {
    "s3d": ("ethograph.video_features.extract", "S3DExtractor"),
    "timm": ("ethograph.video_features.timm_extract", "TimmExtractor"),
}

#: The pip extra that provides each extractor's package, for the error message.
_EXTRA: dict[str, str] = {"timm": "timm"}

#: Extractors that run **outside** this environment: a registered name, so a
#: config can select them and their variable is spelled like any other
#: (``feral``, ``feral_dims``), but nothing here can build them. The value
#: says where the features come from instead.
EXTERNAL: dict[str, str] = {
    "feral": (
        "FERAL runs in its own environment. Export for it with project.video_features() "
        "(labels.json + config.yaml under {root}/feral/), train and infer there, and its "
        "embeddings attach from {root}/feral/embeddings/ — see docs/models/segment/feral."
    ),
}


def feature_dim(name: str) -> str:
    """The feature dim of extractor *name*'s sidecar: ``{name}_dims``."""
    return f"{name}_dims"


def sidecar_path(video: str | Path, out_dir: str | Path, name: str) -> Path:
    """Where *video*'s features from extractor *name* live: ``{out_dir}/{stem}_{name}.nc``."""
    return Path(out_dir) / f"{Path(video).stem}_{name}.nc"


def time_dim_of(da: xr.DataArray) -> str:
    """The sidecar's time dim, found by name rather than assumed.

    A sidecar written before the registry used ``time_s3d``; one written now
    uses :data:`TIME_DIM`. Both are time coords to :func:`get_time_coord`.
    """
    coord = get_time_coord(da)
    if coord is None:
        raise ValueError(f"{da.name!r} has no time coordinate — not a video-feature sidecar (dims {da.dims})")
    return str(coord.name)


def feature_dim_of(da: xr.DataArray) -> str:
    """The one dim of a sidecar that is not time."""
    others = [str(d) for d in da.dims if d != time_dim_of(da)]
    if len(others) != 1:
        raise ValueError(f"A video-feature sidecar has exactly two dims (time, features); {da.name!r} has {da.dims}")
    return others[0]


@dataclass(frozen=True)
class CropBox:
    """A pixel rectangle cut from every decoded frame before the network sees it.

    Corners as the GUI's crop tool reports them — ``(x0, y0)`` top-left,
    ``(x1, y1)`` bottom-right exclusive, y down — so the numbers copy straight
    across from *Tools ▸ Video: Pick a crop…*. One box per video; a crop that
    follows an individual is a later extension, not this class.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(f"crop: the top-left corner ({self.x0}, {self.y0}) is outside the frame")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"crop: ({self.x0}, {self.y0})-({self.x1}, {self.y1}) is empty")

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the box in pixels."""
        return self.x1 - self.x0, self.y1 - self.y0

    def validate(self, width: int, height: int, what: str) -> None:
        """Refuse a box that reaches outside a ``width`` x ``height`` frame, and
        warn when it is not square.

        Every extractor here resizes the shorter side to the network's input
        and takes the centre square (the Kinetics / ImageNet evaluation
        transform), so the long side of a non-square box is silently cut
        off — the warning says how much, so the box can be squared instead.
        """
        if self.x1 > width or self.y1 > height:
            raise ValueError(
                f"crop ({self.x0}, {self.y0})-({self.x1}, {self.y1}) reaches outside {what}, which is {width}x{height}"
            )
        w, h = self.size
        if w != h:
            lost = 1.0 - min(w, h) / max(w, h)
            axis = "width" if w > h else "height"
            logger.warning(
                "crop for %s is %dx%d, not square: the network takes the centre square, so %.0f%% of the "
                "box's %s is dropped. Pick a square box to keep all of it.",
                what,
                w,
                h,
                100 * lost,
                axis,
            )

    def apply(self, frames: np.ndarray) -> np.ndarray:
        """``(N, H, W, 3)`` → the box, as a view."""
        return frames[:, self.y0 : self.y1, self.x0 : self.x1]


class Plan(Protocol):
    """An extractor's settings resolved against one video's rate."""

    video_fps: float
    step: int

    def describe(self) -> str: ...


class Extractor(Protocol):
    """One network, configured; runs over one video at a time."""

    @property
    def name(self) -> str: ...

    def plan(self, video_fps: float) -> Plan:
        """Resolve for a video at *video_fps*; raises when the rate cannot carry the settings."""
        ...

    def extract(
        self,
        video: str | Path,
        *,
        device: str | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> xr.DataArray:
        """The video feature of *video*: ``(time_video, {name}_dims)``."""
        ...


def check_extractor_name(name: str) -> None:
    if name not in EXTRACTORS and name not in EXTERNAL:
        raise ValueError(f"Unknown video-feature extractor {name!r}; choose from {sorted({*EXTRACTORS, *EXTERNAL})}")


def extractor_module(name: str) -> ModuleType:
    """Import the module holding extractor *name*.

    A name outside :data:`EXTRACTORS` is a ``ValueError`` listing the
    choices; a registered extractor whose package is not installed is an
    ``ImportError`` naming the extra.
    """
    check_extractor_name(name)
    if name in EXTERNAL:
        raise ImportError(f"The {name!r} extractor cannot be built here: {EXTERNAL[name]}")
    module_name, _ = EXTRACTORS[name]
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        extra = _EXTRA.get(name)
        if extra is None or exc.name is None or exc.name.split(".")[0] not in (extra, name):
            raise
        raise ImportError(
            f"The {name!r} extractor needs the {exc.name!r} package: pip install 'ethograph[{extra}]'"
        ) from exc


def build_extractor(name: str, **params: Any) -> Extractor:
    """Instantiate the extractor registered as *name* with *params* (see :func:`extractor_module`)."""
    module = extractor_module(name)
    return getattr(module, EXTRACTORS[name][1])(**params)


def to_dataarray(
    feats: np.ndarray,
    *,
    name: str,
    video_fps: float,
    step: int,
    attrs: dict[str, Any],
) -> xr.DataArray:
    """Wrap ``(T, D)`` features as the sidecar DataArray on the video clock."""
    if feats.ndim != 2:
        raise ValueError(f"features must be (T, D), got shape {feats.shape}")
    time = np.arange(feats.shape[0]) * step / video_fps
    da = xr.DataArray(
        feats.astype(np.float32),
        dims=(TIME_DIM, feature_dim(name)),
        coords={TIME_DIM: time, feature_dim(name): np.arange(feats.shape[1])},
        name=name,
        attrs={
            "extractor": name,
            "video_fps": float(video_fps),
            "step": int(step),
            "effective_fps": float(video_fps) / step,
            "time_basis": "video",
            **attrs,
        },
    )
    return describe(da, VIDEO_FEATURE, is_egocentric=False)
