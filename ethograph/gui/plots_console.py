"""Python console panel — work with what a panel plots, not what backs it.

Added from the add-panel popup like any other panel, so it lives in a normal
dock widget the user can drag anywhere or float off the window.

The contract is deliberately narrow, and it is the whole point of the feature:

* Click a feature panel → its **rendered** contents are bound to a Python name
  (the feature name).  That is a plain ``(T,)``/``(T, D)`` numpy array for the
  panel's current feature, selections and visible window — never the
  multi-dimensional DataArray behind it, never other trials.
* Anything the user assigns becomes a new feature in the add-panel popup, so
  ``theta = np.deg2rad(angle)`` then ``np.cos(theta)`` builds up plottable
  variables one line at a time.

Numpy, not xarray: the values are what the axes show, and the time axis rides
along on the array rather than being something the user has to manage.  See
``io/derived.py`` for how a recipe survives panning, zooming and trial changes.
"""

from __future__ import annotations

import code
import contextlib
import io
import logging
import re

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ethograph.io.derived import DerivedLoader, Root, TracedArray, make_derived, stack

logger = logging.getLogger(__name__)

#: Names the console owns; assignments to them are never made into features.
_RESERVED = frozenset({"np", "stack", "t", "features", "forget", "clear", "_", "__builtins__"})

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def to_identifier(name: str) -> str:
    """A feature name as a usable Python identifier (``head/x`` → ``head_x``)."""
    cleaned = re.sub(r"\W+", "_", str(name)).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned


class _ClearCommand:
    """``clear`` typed bare, like a console command — no parentheses needed.

    A REPL evaluates a bare name and prints its repr, so the clearing happens
    in ``__repr__``. ``__call__`` keeps the explicit forms working, which is
    where the full reset lives (``clear(all=True)``).
    """

    def __init__(self, clear):
        self._clear = clear

    def __repr__(self) -> str:
        self._clear(all=False)
        return ""

    def __call__(self, all: bool = False) -> None:
        self._clear(all=all)


class _ConsoleInput(QPlainTextEdit):
    """The prompt line: Enter submits, Shift+Enter continues, ↑/↓ walk history."""

    submitted = Signal(str)
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Consolas", 9))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setFixedHeight(24)
        self.setPlaceholderText("theta = np.deg2rad(angle)    (? for commands)")
        self._history: list[str] = []
        self._pos = 0
        self.textChanged.connect(self._fit_height)

    def _fit_height(self):
        lines = max(1, self.document().blockCount())
        self.setFixedHeight(min(160, 6 + 18 * lines))

    def remember(self, text: str):
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._pos = len(self._history)

    def _recall(self, delta: int):
        if not self._history:
            return
        self._pos = max(0, min(len(self._history), self._pos + delta))
        self.setPlainText(self._history[self._pos] if self._pos < len(self._history) else "")
        self.moveCursor(QTextCursor.End)

    def keyPressEvent(self, event):
        key = event.key()
        multiline = self.document().blockCount() > 1
        if key == Qt.Key_L and event.modifiers() & Qt.ControlModifier:
            self.clear_requested.emit()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submitted.emit(self.toPlainText())
            return
        if key == Qt.Key_Up and not multiline:
            self._recall(-1)
            return
        if key == Qt.Key_Down and not multiline:
            self._recall(+1)
            return
        super().keyPressEvent(event)


class ConsolePanel(QWidget):
    """Interactive namespace over the plotted arrays."""

    #: Emitted after any statement that changed the derived-feature list, so
    #: the add-panel popup and the features combo can repopulate.
    features_changed = Signal()
    #: Emitted on click, so the panel joins the normal active-panel machinery.
    plot_clicked = Signal(object)

    panel_type = "console"
    panel_group = "console"

    #: Shown by the ? button, not printed. The transcript stays a record of
    #: what the user did, so the first thing in it is the panel they clicked.
    _HELP = (
        "Click a feature panel to bind its plotted values here.\n"
        "  theta = np.deg2rad(angle)          assign anything → a new ➕ Add panel feature\n"
        "  stack(sin, cos)                    one panel, several named + coloured curves\n"
        "                                     (or stack(sin=np.sin(rad), cos=np.cos(rad)))\n"
        "  t                                  the panel's time vector\n"
        "                                     np.gradient(pos, t, axis=0) is per second\n"
        "  features()                         what is bound, and what has been derived\n"
        "  forget('name')                     drop one derived feature\n"
        "  clear   (or Ctrl+L)                wipe this transcript\n"
        "  clear(all=True)                    also drop every variable\n"
        "Derived features last for the current trial only."
    )

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._bound: dict[str, Root] = {}
        self._times: list[np.ndarray] = []
        # Names of the derived features assigned here — the ones a reset may drop.
        self._features: set[str] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addStretch()
        self.help_button = QPushButton("?")
        self.help_button.setCheckable(True)
        self.help_button.setFixedSize(16, 16)
        self.help_button.setToolTip("Show the console commands")
        self.help_button.setStyleSheet(
            "QPushButton { color:#ddd; background:rgba(40,40,40,160); border:none;"
            " border-radius:8px; font-size:9px; }"
            "QPushButton:hover, QPushButton:checked { color:#fff; background:rgba(80,120,200,200); }"
        )
        self.help_button.toggled.connect(self._on_help_toggled)
        header.addWidget(self.help_button)
        layout.addLayout(header)

        self.help_label = QLabel(self._HELP)
        self.help_label.setFont(QFont("Consolas", 8))
        self.help_label.setWordWrap(True)
        self.help_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.help_label.setStyleSheet("color: rgba(255,255,255,150); padding: 2px 4px;")
        self.help_label.setVisible(False)
        layout.addWidget(self.help_label)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 9))
        layout.addWidget(self.output, 1)

        self.input = _ConsoleInput()
        self.input.submitted.connect(self._on_submitted)
        self.input.clear_requested.connect(self._ns_clear)
        layout.addWidget(self.input)

        self.ns: dict = {
            "np": np,
            "stack": stack,
            "features": self._ns_features,
            "forget": self._ns_forget,
            "clear": _ClearCommand(self._ns_clear),
        }
        self._interp = code.InteractiveInterpreter(self.ns)
        self._pending: list[str] = []

    def _on_help_toggled(self, shown: bool):
        self.help_label.setVisible(shown)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write(self, text: str):
        # Whitespace-only output is dropped: a bare `clear` evaluates to an
        # empty repr, and the displayhook's trailing newline would otherwise
        # land in the transcript that was just wiped.
        if not text or not text.strip():
            return
        self.output.appendPlainText(text.rstrip("\n"))
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())

    def mousePressEvent(self, event):
        # Same payload shape every plot emits ({"x": time, "button": …}); the
        # console has no time axis, and a None x is what consumers already
        # treat as "nothing to place here".
        self.plot_clicked.emit({"x": None, "button": Qt.NoButton})
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    # Binding the active panel's plotted values
    # ------------------------------------------------------------------

    def bind_panel(self, plot) -> None:
        """Bind the *rendered* contents of a feature *plot* to a name.

        The name is the panel's feature; the value is what that panel is
        showing right now — its own feature, its own selections, its own
        visible window.  Rebinding on every click is what keeps the console
        honest: the variable always describes the panel you last clicked.
        """
        loader = getattr(self.app_state, "data_loader", None)
        feature = plot._effective_feature() if hasattr(plot, "_effective_feature") else None
        if loader is None or not feature:
            return
        selections = plot._effective_selections()
        t0, t1 = plot.get_current_xlim()
        plot_data = loader.select(feature, selections, t0=t0, t1=t1)
        if plot_data is None or plot_data.data is None or len(plot_data.time) == 0:
            self.write(f"# {feature}: nothing plotted in this window")
            return

        name = to_identifier(feature)
        root = Root(feature=str(feature), pinned=tuple(sorted((str(k), str(v)) for k, v in selections.items())))
        data = np.asarray(plot_data.data)
        time = np.asarray(plot_data.time)
        self.ns[name] = TracedArray(data, time=time, node=root, name=name)
        # The panel's time vector, so rates come from the data rather than an
        # assumed sample spacing: np.gradient(pos, t, axis=0) is per second.
        self.ns["t"] = time
        self._bound[name] = root
        self._remember_time(time)

        pinned = ", ".join(f"{k}={v}" for k, v in root.pinned) or "all dims"
        # Every click says what it bound, even when it re-binds the same thing:
        # the line is the answer to "what am I holding now", and staying silent
        # on a re-click reads as the click having done nothing.
        self.write(f"{name}: shape {tuple(data.shape)}  [{pinned}]  t = {time[0]:.3f}…{time[-1]:.3f} s")

    def _remember_time(self, time: np.ndarray) -> None:
        """Keep the bound time axes so an untraceable result can still be
        matched to one (``savgol_filter(speed, 11, 3)`` returns a bare array)."""
        if not any(len(t) == len(time) and np.array_equal(t, time) for t in self._times):
            self._times.append(time)
        del self._times[:-8]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _on_submitted(self, text: str):
        line = text.rstrip()
        self.input.remember(line)
        self.input.clear()
        prompt = "... " if self._pending else ">>> "
        self.write(f"{prompt}{line}")
        self._pending.append(line)

        source = "\n".join(self._pending)
        before = dict(self.ns)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            incomplete = self._interp.runsource(source, "<console>")
        if incomplete:
            return
        self._pending.clear()
        self.write(buffer.getvalue())
        self._register_new_names(before)

    def _register_new_names(self, before: dict) -> None:
        """Turn every name the statement created or rebound into a feature."""
        loader = getattr(self.app_state, "data_loader", None)
        if not isinstance(loader, DerivedLoader):
            return
        added = False
        for name, value in list(self.ns.items()):
            if name in _RESERVED or name.startswith("_") or not _IDENTIFIER.match(name):
                continue
            if name in before and before[name] is value:
                continue
            if name in self._bound and isinstance(value, TracedArray) and value.eto_node is self._bound[name]:
                continue  # the panel binding itself, not a user expression
            if isinstance(value, TracedArray):
                # Remember what the user called it, so `stack(sin, cos)` can
                # label its columns without the caller repeating the names.
                value._eto_name = name
            derived = make_derived(name, value, fallback_times=self._times) if isinstance(value, np.ndarray) else None
            if derived is None:
                reason = self._rejection_reason(name, value)
                if reason:
                    self.write(f"# {reason}")
                continue
            loader.register(derived)
            self._features.add(name)
            kind = "snapshot" if derived.is_snapshot else derived.describe()
            self.write(f"# added feature '{name}'  ({kind})")
            added = True
        if added:
            self.features_changed.emit()

    def _rejection_reason(self, name: str, value) -> str | None:
        """Why *value* did not become a feature — ``None`` when it was never a
        candidate (a number, a string) and silence is the right answer.

        Anything array-shaped gets an explanation: a value the user clearly
        meant to plot vanishing without a word is the worst outcome here.
        """
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], np.ndarray):
            # np.gradient and friends return one array PER AXIS when no axis is
            # given — the single most likely way to land here.
            return (
                f"{name}: {len(value)} arrays, not one — not added. "
                f"Pass axis=0 for the time axis, or index it ({name}[0])"
            )
        if not isinstance(value, np.ndarray):
            return None
        array = np.asarray(value)
        if array.ndim > 2:
            return f"{name}: {array.ndim}-D {array.shape} — not added; a feature is (T,) or (T, D)"
        if not np.issubdtype(array.dtype, np.number):
            return f"{name}: dtype {array.dtype} — not added; a feature must be numeric"
        if array.ndim == 0 or array.shape[0] < 2:
            return f"{name}: not a time series — not added"
        return f"{name}: no time axis matches its length {array.shape[0]} — not added"

    # ------------------------------------------------------------------
    # Namespace helpers
    # ------------------------------------------------------------------

    def _ns_features(self):
        """List what is bound (panels) and what has been derived."""
        loader = getattr(self.app_state, "data_loader", None)
        for name, root in self._bound.items():
            self.write(f"  {name}  ← panel: {root.feature}")
        for name, derived in (loader.derived if isinstance(loader, DerivedLoader) else {}).items():
            self.write(f"  {name}  = {derived.describe()}")

    def _ns_clear(self, all: bool = False):
        """Clear the transcript — typed as a bare ``clear``, or Ctrl+L.

        ``clear(all=True)`` additionally resets the session: bound panel
        variables and every derived feature go away, so the add-panel popup is
        back to the dataset's own features. Plain ``clear()`` only wipes the
        text — variables you built stay usable, which is what you want after
        the scrollback gets noisy mid-analysis.
        """
        self.output.clear()
        # An unfinished multi-line block would otherwise be silently prepended
        # to whatever is typed next, with nothing on screen to explain it.
        self._pending.clear()
        if not all:
            return
        self._drop_variables()
        self.features_changed.emit()

    def reset_for_trial(self) -> None:
        """Drop every variable because the trial changed.

        Derived features are **per trial**. A recipe would happily re-evaluate
        on the new trial, but its meaning would not survive the move: it was
        built from what one trial's panel showed, and silently carrying it
        forward makes a variable that reads as this trial's while describing
        the last one's setup. So they die with the trial. The transcript stays
        — it is the record of how you got here — with a line saying they went.
        """
        dropped = self._drop_variables()
        if dropped:
            self.write(f"# trial changed — dropped {', '.join(sorted(dropped))}")
            self.features_changed.emit()

    def _drop_variables(self) -> list[str]:
        """Forget every binding and every derived feature made here; return the names dropped.

        A derived feature the console did not assign (the firing rates, a
        camera's motion trace) belongs to whoever registered it and stays.
        """
        loader = getattr(self.app_state, "data_loader", None)
        dropped: list[str] = []
        if isinstance(loader, DerivedLoader):
            for name in list(loader.derived):
                if name not in self._features:
                    continue
                loader.unregister(name)
                dropped.append(name)
        self._features.clear()
        for name in list(self.ns):
            if name not in _RESERVED:
                del self.ns[name]
        self._bound.clear()
        self._times.clear()
        return dropped

    def _ns_forget(self, name: str):
        """Remove a derived feature from the add-panel popup."""
        loader = getattr(self.app_state, "data_loader", None)
        if isinstance(loader, DerivedLoader):
            loader.unregister(str(name))
        self.ns.pop(str(name), None)
        self.write(f"# forgot '{name}'")
        self.features_changed.emit()
