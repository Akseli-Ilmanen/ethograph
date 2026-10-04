#!/usr/bin/env python
"""Command-line interface for ethograph."""

import logging
import os
import sys
import warnings

# Suppress noisy dependency warnings before any imports trigger them
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"logging")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"vispy\.")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"numpy\.")

# PyOpenGL info message goes through logging, not warnings
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)


def _ensure_qt_plugins():
    """Set QT_PLUGIN_PATH for conda-forge Qt installs (needed by menuinst shortcuts)."""
    if os.environ.get("QT_PLUGIN_PATH"):
        return
    candidates = [
        os.path.join(sys.prefix, "Library", "plugins"),  # Windows conda-forge
        os.path.join(sys.prefix, "lib", "qt6", "plugins"),  # Linux conda-forge Qt6
        os.path.join(sys.prefix, "lib", "qt5", "plugins"),  # Linux conda-forge Qt5 (fallback)
        os.path.join(sys.prefix, "lib", "qt", "plugins"),  # macOS conda-forge
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "platforms")):
            os.environ["QT_PLUGIN_PATH"] = path
            return


def _fix_wayland_opengl():
    """Force XCB/GLX on GNOME+Wayland to avoid OpenGL crashes with napari.

    See https://github.com/Akseli-Ilmanen/ethograph/issues/1
    """
    if sys.platform != "linux":
        return
    wayland_active = bool(os.environ.get("WAYLAND_DISPLAY")) or (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )
    if not wayland_active:
        return
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
    if "GNOME" not in desktop:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")


def _linux_preflight() -> None:
    """Name every missing system library up front, with the line that installs it.

    The wheels fail one library at a time and each in its own words; the GUI
    still tries to start afterwards, since some of them are only needed by
    panels the user may never open.
    """
    if sys.platform != "linux":
        return
    from ethograph.utils.system_check import (
        display_available,
        install_hint,
        is_wsl,
        missing_system_libs,
        package_manager,
        wsl_notes,
    )

    log = logging.getLogger("ethograph")
    missing = missing_system_libs()
    if missing:
        names = ", ".join(lib.soname for lib in missing)
        log.warning(
            "Missing Linux system libraries: %s\n    fix: %s\n    (details: `ethograph check`)",
            names,
            install_hint(missing, package_manager()),
        )
    if is_wsl():
        for note in wsl_notes():
            log.warning("WSL: %s", note)
    if not display_available():
        log.warning("No DISPLAY or WAYLAND_DISPLAY is set; Qt cannot open a window.")


def _require_gui_extra(exc: ImportError) -> None:
    """Turn a missing GUI dependency into an actionable install hint."""
    # A DLL load failure is an ImportError without a module name; its text
    # is the only clue (typically a Qt built against another Python).
    cause = f"{exc.name} is missing" if exc.name else str(exc)
    sys.exit(
        f"ethograph: the GUI dependencies are not installed ({cause}).\n"
        '\n    pip install "ethograph[gui]"          # GUI\n'
        '    pip install "ethograph[gui,audio]"    # GUI + audio\n'
        "\nThe plain `ethograph` install is the library only (TrialTree, I/O, labels)."
    )


class _ConsoleFormatter(logging.Formatter):
    """Compact console lines: drop the ``ethograph.`` prefix and the INFO level.

    INFO is what almost every line is, so naming it says nothing; anything
    louder still announces itself.
    """

    _QUIET = logging.Formatter("%(name)s | %(message)s")
    _LOUD = logging.Formatter("%(levelname)s %(name)s | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        original = record.name
        if original.startswith("ethograph."):
            record.name = original[len("ethograph.") :]
        try:
            style = self._QUIET if record.levelno <= logging.INFO else self._LOUD
            return style.format(record)
        finally:
            record.name = original


def launch():
    """Launch the ethograph GUI."""
    from ethograph.utils.logging import start_session_log

    log_path = start_session_log("gui")

    logging.basicConfig(level=logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(_ConsoleFormatter())
    _fix_wayland_opengl()
    _ensure_qt_plugins()
    _linux_preflight()

    from ethograph.utils.paths import ethograph_home, migrate_home_layout

    for old, new in migrate_home_layout():
        logging.getLogger("ethograph").info("Home layout: moved %s -> %s", old, new)
    logging.getLogger("ethograph").info("Global settings directory: %s", ethograph_home())
    logging.getLogger("ethograph").info("Session log: %s", log_path)

    try:
        from qtpy.QtCore import QLocale
        from qtpy.QtWidgets import QApplication
    except ImportError as exc:
        _require_gui_extra(exc)

    # Before any widget exists: children inherit the locale of the widget tree
    # they are inserted into, so the OS locale (possibly comma-decimal) must
    # never leak into the shell. Dot decimals everywhere.
    QLocale.setDefault(QLocale.c())

    try:
        from ethograph.gui import theme
        from ethograph.gui.dialog_fit import install_dialog_fitter
        from ethograph.gui.main_window import EthographMainWindow
        from ethograph.gui.widgets_meta import MetaWidget
    except ImportError as exc:
        _require_gui_extra(exc)

    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply_theme(app)
    install_dialog_fitter(app)

    shell = EthographMainWindow()
    meta_widget = MetaWidget(shell)
    shell.attach_meta_widget(meta_widget)

    # Start dialog first; the main window only appears after the user picks
    # something to load (or skips). Closing the dialog exits without a GUI.
    from ethograph.gui.cover_page import show_cover_page

    if not show_cover_page(shell):
        return

    shell.show()
    shell.raise_()
    shell.activateWindow()

    # A dataset loaded through the cover page was loaded while the main window
    # was still hidden, so every isVisible()-guarded viewport update (audio
    # trace / spectrogram range handlers) was skipped. Redo them once shown.
    if getattr(meta_widget.app_state, "ready", False):
        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, meta_widget.plot_container.update_audio_panels)

    app.exec()


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: ethograph <command>")
        print("Commands:")
        print("  launch    Launch the ethograph GUI")
        print("  shortcut  Install desktop/Start Menu shortcut")
        print("  check     Check for missing Linux system libraries")
        sys.exit(1)

    command = sys.argv[1]

    if command == "launch":
        launch()
    elif command == "shortcut":
        from ethograph.shortcuts import install_shortcut

        sys.exit(install_shortcut())
    elif command == "check":
        from ethograph.utils.system_check import run_check

        sys.exit(run_check())
    else:
        print(f"Unknown command: {command}")
        print("Available commands: launch, shortcut, check")
        sys.exit(1)


if __name__ == "__main__":
    main()
