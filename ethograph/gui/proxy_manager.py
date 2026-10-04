"""Background low-res proxy generation, reconciled against visible videos.

:class:`ProxyManager` owns at most one :class:`_ProxyJob` per source video. A
job runs ffmpeg in a worker thread (via ``Popen``, so it can be killed) and
writes atomically (``.part`` → rename), so a cancelled encode never leaves a
half-file that a decode path would treat as a finished proxy.

The manager is driven purely by reconciliation: callers pass the set of
currently-visible source videos to :meth:`sync`; jobs for sources that
disappeared are cancelled, jobs for new sources without a cached proxy are
started. This keeps the running-thread count bounded to what is on screen —
panel closes, opens, and trial changes all funnel through the same call.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from qtpy.QtCore import QObject, QThread, Signal

from ethograph.io.video_proxy import DEFAULT_PROXY_HEIGHT, build_proxy_command, proxy_cache_path


class _ProxyJob(QThread):
    """Encode one source video's proxy in the background; cancellable."""

    #: Emitted on the main thread: (source_path, success).
    done = Signal(str, bool)

    def __init__(self, source_path: str, proxy_path: Path, scale_height: int, parent=None):
        super().__init__(parent)
        self._source = source_path
        self.proxy_path = proxy_path
        self._scale_height = scale_height
        # Keep the real extension (…​.part.mp4) so ffmpeg can infer the muxer.
        self._tmp = proxy_path.with_suffix(".part" + proxy_path.suffix)
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def run(self) -> None:  # runs in the worker thread
        try:
            self.proxy_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = build_proxy_command(self._source, self._tmp, scale_height=self._scale_height)
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ret = self._proc.wait()
            if self._cancelled:
                self._tmp.unlink(missing_ok=True)
                return
            if ret == 0 and self._tmp.exists():
                self._tmp.replace(self.proxy_path)
                self.done.emit(self._source, True)
            else:
                self._tmp.unlink(missing_ok=True)
                self.done.emit(self._source, False)
        except Exception:  # noqa: BLE001 - background best-effort; never crash
            self._tmp.unlink(missing_ok=True)
            self.done.emit(self._source, False)

    def cancel(self) -> None:
        """Request cancellation and kill the ffmpeg subprocess if running."""
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass


class ProxyManager(QObject):
    """Start/stop background proxy jobs to match the visible video set."""

    #: Emitted when a source's proxy becomes available (on the main thread).
    proxy_ready = Signal(str)
    #: Emitted when a job for a source starts / fails (source path).
    proxy_started = Signal(str)
    proxy_failed = Signal(str)

    def __init__(self, cache_dir_fn, parent=None):
        super().__init__(parent)
        #: source_path -> proxy cache dir. Callable so callers control layout.
        self._cache_dir_fn = cache_dir_fn
        self._jobs: dict[str, _ProxyJob] = {}

    def sync(self, sources: list[str], scale_height: int = DEFAULT_PROXY_HEIGHT) -> None:
        """Reconcile running jobs against the currently-visible *sources*.

        Cancels jobs whose source is no longer visible or whose proxy is of
        another height; starts jobs for visible sources that lack a cached
        proxy at *scale_height* and aren't already running. An empty list
        cancels everything (e.g. proxy mode turned off).
        """
        wanted = {src: proxy_cache_path(src, self._cache_dir_fn(src), scale_height) for src in sources}
        for src, job in list(self._jobs.items()):
            if wanted.get(src) != job.proxy_path:
                self._stop(src)
        for src, proxy in wanted.items():
            if src in self._jobs:
                continue
            if proxy.exists():
                continue  # already cached — nothing to generate
            self._start(src, proxy, scale_height)

    def cancel_all(self) -> None:
        """Cancel and join every running job (call on close/teardown)."""
        for src in list(self._jobs):
            self._stop(src)

    def _start(self, source: str, proxy: Path, scale_height: int) -> None:
        job = _ProxyJob(source, proxy, scale_height, parent=self)
        job.done.connect(self._on_done)
        self._jobs[source] = job
        job.start()
        self.proxy_started.emit(source)

    def _stop(self, source: str) -> None:
        job = self._jobs.pop(source, None)
        if job is None:
            return
        job.cancel()
        job.wait(3000)

    def _on_done(self, source: str, success: bool) -> None:
        job = self._jobs.pop(source, None)
        if job is not None:
            job.wait(100)
        if success:
            self.proxy_ready.emit(source)
        else:
            self.proxy_failed.emit(source)
