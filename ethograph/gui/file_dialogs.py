"""File dialogs that remember where the user last browsed.

A ``QFileDialog`` opened without a start directory lands in the process's
current working directory — the install folder for a launched app, which is
never where anyone's data lives.  ``app_state.last_browse_dir`` (SCOPE_GLOBAL,
so it survives into the next session and is available on the cover page,
before any dataset has been loaded) records the folder each browse resolved
to, and every dialog here starts there.

A caller may pass a ``preferred_dir`` — the loaded dataset's folder, say —
which wins whenever it still exists; the remembered folder is the fallback.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QAbstractItemView, QFileDialog, QListView, QTreeView, QWidget

ALL_FILES = "All files (*)"


def browse_start_dir(app_state, preferred_dir: str | Path | None = None) -> str:
    """Directory a browse dialog should open in.

    Returns ``""`` when neither *preferred_dir* nor the remembered folder
    exists, which leaves Qt's own default in place.
    """
    for candidate in (preferred_dir, getattr(app_state, "last_browse_dir", None)):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            path = path.parent
        if path.is_dir():
            return str(path)
    return ""


def remember_browse_dir(app_state, path: str | Path | None) -> None:
    """Record the folder holding *path* as the next dialog's start point."""
    if not path:
        return
    folder = Path(path)
    if not folder.is_dir():
        folder = folder.parent
    if folder.is_dir():
        app_state.last_browse_dir = str(folder)


def browse_open_file(
    parent: QWidget | None,
    app_state,
    caption: str,
    file_filter: str = ALL_FILES,
    preferred_dir: str | Path | None = None,
) -> str:
    """Pick an existing file. Returns ``""`` when the user cancels."""
    path, _ = QFileDialog.getOpenFileName(
        parent,
        caption=caption,
        dir=browse_start_dir(app_state, preferred_dir),
        filter=file_filter,
    )
    remember_browse_dir(app_state, path)
    return path


def browse_open_dir(
    parent: QWidget | None,
    app_state,
    caption: str,
    preferred_dir: str | Path | None = None,
) -> str:
    """Pick an existing folder. Returns ``""`` when the user cancels."""
    path = QFileDialog.getExistingDirectory(
        parent,
        caption=caption,
        dir=browse_start_dir(app_state, preferred_dir),
    )
    remember_browse_dir(app_state, path)
    return path


def browse_open_dirs(
    parent: QWidget | None,
    app_state,
    caption: str,
    preferred_dir: str | Path | None = None,
) -> list[str]:
    """Pick one or more existing folders. Returns ``[]`` when the user cancels.

    Native dialogs pick a single folder only, so this is Qt's own dialog
    with its file views switched to extended selection (Ctrl/Shift-click).
    """
    dialog = QFileDialog(parent, caption, browse_start_dir(app_state, preferred_dir))
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    if not dialog.exec():
        return []
    folders = [f for f in dialog.selectedFiles() if Path(f).is_dir()]
    if folders:
        remember_browse_dir(app_state, folders[0])
    return folders


def browse_save_file(
    parent: QWidget | None,
    app_state,
    caption: str,
    default_name: str,
    file_filter: str = ALL_FILES,
    preferred_dir: str | Path | None = None,
) -> str:
    """Pick a save location, pre-filled with *default_name*."""
    start = browse_start_dir(app_state, preferred_dir)
    path, _ = QFileDialog.getSaveFileName(
        parent,
        caption=caption,
        dir=str(Path(start) / default_name) if start else default_name,
        filter=file_filter,
    )
    remember_browse_dir(app_state, path)
    return path
