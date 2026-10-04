"""Low-resolution proxy media for fast video navigation.

A *proxy* is a downscaled, short-GOP re-encode of a source video used for
smooth navigation (Adobe/DaVinci-style). It preserves the source frame count
and per-frame timing **exactly** — only resolution and GOP structure change —
so frame indices, offsets, and labels stay aligned with the original. Callers
swap the proxy path in for the source path at the last moment (the decoder is
handed a different file; nothing else changes).

Two lifecycles:
- Persistent projects (.nc/.nwb + alignment): proxies cached under
  ``.ethograph/proxies/`` keyed by source-file identity, generated once and
  reused across sessions.
- Drag & drop throwaway sessions: pass the drop's temp dir as ``cache_dir``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ethograph.utils.ffmpeg import ffmpeg_executable
from ethograph.utils.paths import media_cache_key

#: Bump when the encode recipe changes so stale proxies are regenerated.
_PROXY_RECIPE_VERSION = 1

#: Proxy height when the user has not chosen one, and the heights offered.
DEFAULT_PROXY_HEIGHT = 480
PROXY_HEIGHTS = (360, 480, 720, 1080)


def _source_key(video_path: Path) -> str:
    """Deterministic cache key from source identity (path, size, mtime)."""
    return media_cache_key(video_path, _PROXY_RECIPE_VERSION)


def proxy_cache_path(video_path: Path | str, cache_dir: Path | str, scale_height: int = DEFAULT_PROXY_HEIGHT) -> Path:
    """Return the deterministic proxy path for *video_path* at *scale_height* under *cache_dir*."""
    video_path = Path(video_path)
    # The default height keeps the bare name, so proxies cached before the
    # height was a choice are still found.
    suffix = "" if scale_height == DEFAULT_PROXY_HEIGHT else f"_{int(scale_height)}p"
    return Path(cache_dir) / f"{_source_key(video_path)}{suffix}.mp4"


def build_proxy_command(
    video_path: Path | str,
    proxy_path: Path | str,
    scale_height: int = DEFAULT_PROXY_HEIGHT,
    gop: int = 10,
    crf: int = 23,
    preset: str = "veryfast",
    hwaccel: str | None = None,
) -> list[str]:
    """Build the ffmpeg argv for a proxy encode (see :func:`generate_proxy`).

    Exposed so a background worker can run it via ``Popen`` (cancellable),
    keeping the encode recipe in one place.
    """
    video_path = Path(video_path)
    proxy_path = Path(proxy_path)
    use_nvenc = hwaccel == "cuda"

    cmd = [ffmpeg_executable(), "-y"]
    if hwaccel:
        cmd.extend(["-hwaccel", hwaccel])
    elif sys.platform == "darwin":
        cmd.extend(["-hwaccel", "videotoolbox"])

    # -2 keeps width even while preserving aspect ratio; min() never upscales
    # a source that is already smaller than the proxy height.
    cmd.extend(["-i", video_path.as_posix(), "-vf", f"scale=-2:'min(ih,{int(scale_height)})'"])

    if use_nvenc:
        cmd.extend(["-c:v", "h264_nvenc", "-cq", str(crf)])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", preset, "-crf", str(crf)])

    # Short GOP: cap keyframe interval AND minimum so scene-cut detection can
    # only ADD keyframes (which only helps seeking), never lengthen the GOP.
    cmd.extend(["-g", str(gop), "-keyint_min", str(gop)])
    # No audio — handled by the separate audio pipeline.
    cmd.append("-an")
    cmd.append(proxy_path.as_posix())
    return cmd


def generate_proxy(
    video_path: Path | str,
    proxy_path: Path | str,
    scale_height: int = DEFAULT_PROXY_HEIGHT,
    gop: int = 10,
    crf: int = 23,
    preset: str = "veryfast",
    hwaccel: str | None = None,
    verbose: bool = True,
) -> Path:
    """Transcode *video_path* to a low-res, short-GOP proxy at *proxy_path*.

    The proxy has the **same frame count and per-frame timing** as the source
    (no ``-r``/frame-rate change), so frame index N maps to frame N in both.
    Only resolution (``scale_height``) and GOP length (``gop``) change.

    Parameters
    ----------
    video_path : Path or str
        Source video.
    proxy_path : Path or str
        Output path (``.mp4``). Overwritten if it exists.
    scale_height : int
        Target height in pixels; width is derived to preserve aspect ratio and
        kept even (required by H.264). A source shorter than this keeps its
        own height. ``scale=-2:min(ih,H)``.
    gop : int
        Maximum keyframe interval in frames. Small values (e.g. 10, or 1 for
        all-intra) make seeking cheap at the cost of file size. Since the whole
        access pattern is *seek-to-keyframe → decode forward*, this dominates
        how quickly an arbitrary point in the video can be shown.
    crf : int
        x264 constant-rate-factor quality (lower = better/larger). Ignored for
        NVENC, which uses ``-cq``.
    preset : str
        x264 speed/efficiency preset.
    hwaccel : str or None
        ffmpeg hardware backend. ``"cuda"`` selects the NVENC encoder
        (``h264_nvenc``); on macOS ``"videotoolbox"`` is used automatically
        when None. Otherwise software ``libx264``.
    verbose : bool
        Stream ffmpeg output to the terminal when True.

    Returns
    -------
    Path
        ``proxy_path``.
    """
    video_path = Path(video_path)
    proxy_path = Path(proxy_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    proxy_path.parent.mkdir(parents=True, exist_ok=True)

    # Encode to a temp file, then atomically move into place, so an
    # interrupted/failed encode never leaves a half-written proxy that a
    # decode path would pick up as complete.
    # Keep the real extension (…​.part.mp4) so ffmpeg can infer the muxer.
    tmp_path = proxy_path.with_suffix(".part" + proxy_path.suffix)
    cmd = build_proxy_command(
        video_path,
        tmp_path,
        scale_height=scale_height,
        gop=gop,
        crf=crf,
        preset=preset,
        hwaccel=hwaccel,
    )

    if verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        err = result.stderr if getattr(result, "stderr", None) else "Unknown error"
        raise RuntimeError(f"FFmpeg proxy error: {err}")

    tmp_path.replace(proxy_path)
    return proxy_path


def ensure_proxy(
    video_path: Path | str,
    cache_dir: Path | str,
    scale_height: int = DEFAULT_PROXY_HEIGHT,
    **kwargs,
) -> Path:
    """Return a cached proxy for *video_path*, generating it if missing.

    The cache is keyed by source identity (:func:`_source_key`) and height,
    so a proxy is reused across sessions and never applied to changed media. Extra keyword
    arguments are forwarded to :func:`generate_proxy`.
    """
    video_path = Path(video_path)
    proxy_path = proxy_cache_path(video_path, cache_dir, scale_height)
    if proxy_path.exists():
        return proxy_path
    return generate_proxy(video_path, proxy_path, scale_height=scale_height, **kwargs)


def proxy_cache_size(cache_dir: Path | str) -> int:
    """Total bytes of cached proxies (and any temp files) under *cache_dir*."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    return sum(f.stat().st_size for f in cache_dir.glob("*") if f.is_file())


def clear_proxy_cache(cache_dir: Path | str, keep: set[str] | None = None) -> int:
    """Delete cached proxies under *cache_dir*; return bytes freed.

    Files whose absolute path is in *keep* are preserved (e.g. proxies for
    videos currently being decoded — deleting those would break playback).
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    keep = {str(Path(p)) for p in (keep or set())}
    freed = 0
    for f in cache_dir.glob("*"):
        if not f.is_file() or str(f) in keep:
            continue
        try:
            size = f.stat().st_size
            f.unlink()
            freed += size
        except OSError:
            pass
    return freed
