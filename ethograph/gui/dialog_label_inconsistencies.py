"""Find label inconsistencies — filter the trials table by what the labels do.

**Tools ▸ Labels: Find label inconsistencies…**

The trials table filters on *metadata*: genotype, session, drug. This filters
on the *labels*: which trials have one event without its partner, which ran
the classes in an order they should not have, which are missing a sequence
altogether. Those are the trials worth looking at first, and no metadata
column knows about them.

The five questions it can ask about a set of label classes, in the order the
dialog lists them (:data:`~ethograph.utils.sequences.LABEL_MATCH_MODES`):

* **All of them occur** — the classes are all somewhere in the trial.
* **Some but not all occur** — one event without its partner. This is the
  "spot uncoupled labels" case, and it is not the negation of the first: a
  trial with *none* of them is not a broken pair, it is a trial where the
  behaviour did not happen.
* **Any of them occurs more than once** — a class that should happen once per
  trial happens twice: a doubled click, a prediction that fired twice.
* **In this order** — ``1-2-6-8`` in that order, other labels allowed in
  between (so ``1-2-6-6-8`` matches).
* **In this order, one straight after another** — the same, contiguously (so
  ``1-2-6-6-8`` does *not* match ``1-2-6-8``).

**Invert** turns any of them into "find the trials where this is *not* true",
which is how "which trials are missing the sequence" is asked.

The result becomes the trials table's **label filter**, a slot of its own that
sits on top of the column filters — so "wild-type trials where the sequence
broke" is one question, and answering it does not throw the genotype filter
away. **Clear** takes it off again; nothing about the labels is modified,
ever. Every operation downstream then narrows with the table, because the
table's visible trials are the one trial filter in Ethograph.
"""

from __future__ import annotations

import logging

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ethograph.gui.notify import notify
from ethograph.utils.sequences import (
    LABEL_MATCH_MODES,
    parse_label_pattern,
    trials_matching_labels,
)

logger = logging.getLogger(__name__)

_ALL_INDIVIDUALS = "All"


class LabelInconsistencyDialog(QDialog):
    """Pick a question about the labels; filter the trials table by the answer."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find label inconsistencies")
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Filter the trials table by what the labels do, rather than by metadata.\n"
            "The result sits on top of the column filters — both apply."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(intro)

        form = QFormLayout()
        self.pattern_edit = QLineEdit()
        self.pattern_edit.setPlaceholderText("e.g. 1-2-6-8")
        self.pattern_edit.setToolTip(
            "Label ids, in the order you expect them — the same spelling the\n"
            "Sequence navigate mode takes. Only the 'in this order' modes read the order."
        )
        self.pattern_edit.textChanged.connect(self._refresh_preview)
        form.addRow("Labels:", self.pattern_edit)

        self.classes_label = QLabel("")
        self.classes_label.setWordWrap(True)
        self.classes_label.setStyleSheet("color: grey; font-size: 10px;")
        form.addRow("", self.classes_label)

        self.mode_combo = QComboBox()
        for key, text in LABEL_MATCH_MODES.items():
            self.mode_combo.addItem(text, key)
        self.mode_combo.currentIndexChanged.connect(self._refresh_preview)
        form.addRow("Match:", self.mode_combo)

        self.individual_combo = QComboBox()
        self.individual_combo.setToolTip(
            "Whose labels to read. With two animals labelled in one trial their\n"
            "events interleave, and an order across both means nothing."
        )
        self.individual_combo.currentIndexChanged.connect(self._refresh_preview)
        form.addRow("Individual:", self.individual_combo)

        self.invert_check = QCheckBox("Invert — find the trials where this is NOT true")
        self.invert_check.setToolTip(
            "Turns 'in this order' into 'the order is broken or missing', and\n"
            "'all of them occur' into 'at least one is missing'."
        )
        self.invert_check.toggled.connect(self._refresh_preview)
        form.addRow("", self.invert_check)
        layout.addLayout(form)

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        buttons = QHBoxLayout()
        self.apply_btn = QPushButton("Filter trials to these")
        self.apply_btn.setAutoDefault(False)
        self.apply_btn.clicked.connect(self._apply)
        buttons.addWidget(self.apply_btn)
        self.clear_btn = QPushButton("Clear label filter")
        self.clear_btn.setAutoDefault(False)
        self.clear_btn.setToolTip("Take the label filter off again; the column filters stay as they are.")
        self.clear_btn.clicked.connect(self._clear)
        buttons.addWidget(self.clear_btn)
        layout.addLayout(buttons)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(self.status_label)

        self.resize(460, 300)
        self.refresh()

    # ------------------------------------------------------------------

    def _trials_widget(self):
        return getattr(self.meta, "trials_widget", None)

    def refresh(self) -> None:
        """Re-read the session: which classes exist, which individuals."""
        mappings = getattr(getattr(self.meta, "labels_widget", None), "_mappings", {}) or {}
        named = ", ".join(f"{lid} ({info.get('name', lid)})" for lid, info in sorted(mappings.items()) if lid)
        self.classes_label.setText(f"Classes in this session: {named}" if named else "No label classes loaded.")

        current = self.individual_combo.currentText()
        self.individual_combo.blockSignals(True)
        self.individual_combo.clear()
        self.individual_combo.addItem(_ALL_INDIVIDUALS)
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is not None and not df.empty and "individual" in df.columns:
            for name in sorted({str(v) for v in df["individual"].dropna().unique()}):
                self.individual_combo.addItem(name)
        idx = self.individual_combo.findText(current)
        self.individual_combo.setCurrentIndex(max(0, idx))
        self.individual_combo.blockSignals(False)

        self._sync_clear_button()
        self._refresh_preview()

    def _sync_clear_button(self) -> None:
        widget = self._trials_widget()
        self.clear_btn.setEnabled(bool(widget is not None and widget.label_filter_active()))

    def matching_trials(self) -> set[str] | None:
        """The trials the current question picks out, or None with no pattern."""
        target = parse_label_pattern(self.pattern_edit.text())
        if not target:
            return None
        individual = self.individual_combo.currentText()
        return trials_matching_labels(
            getattr(self.app_state, "_all_labels_df", None),
            target,
            mode=self.mode_combo.currentData(),
            invert=self.invert_check.isChecked(),
            # Every trial of the session, not just the visible ones: the
            # answer must not change with what is already filtered out, and a
            # trial carrying no labels only exists in this list.
            trials=self._all_trials(),
            individual=None if individual == _ALL_INDIVIDUALS else individual,
        )

    def _all_trials(self) -> list[str]:
        widget = self._trials_widget()
        base = getattr(widget, "_base_trials", None) if widget is not None else None
        return [str(t) for t in (base or self.app_state.trials or [])]

    def _refresh_preview(self, *_args) -> None:
        hits = self.matching_trials()
        if hits is None:
            self.preview_label.setText("Type the label ids to match, e.g. 1-2-6-8.")
            self.apply_btn.setEnabled(False)
            return
        total = len(self._all_trials())
        self.apply_btn.setEnabled(bool(hits))
        if not hits:
            self.preview_label.setText(f"No trials match — nothing to filter to (of {total}).")
            return
        listed = ", ".join(sorted(hits, key=self._sort_key)[:8])
        more = "…" if len(hits) > 8 else ""
        self.preview_label.setText(f"<b>{len(hits)}</b> of {total} trials match: {listed}{more}")

    @staticmethod
    def _sort_key(trial: str):
        try:
            return (0, int(trial))
        except ValueError:
            return (1, trial)

    def _description(self) -> str:
        parts = [self.pattern_edit.text().strip(), self.mode_combo.currentText().lower()]
        if self.invert_check.isChecked():
            parts.append("inverted")
        return " · ".join(p for p in parts if p)

    def _apply(self) -> None:
        widget = self._trials_widget()
        hits = self.matching_trials()
        if widget is None or not hits:
            return
        shown = widget.set_label_filter(hits, self._description())
        self._sync_clear_button()
        message = f"Trials table filtered to {shown} trial(s) by labels ({self._description()})."
        self.status_label.setText(message + " Clear the label filter to undo.")
        notify(message)

    def _clear(self) -> None:
        widget = self._trials_widget()
        if widget is None:
            return
        shown = widget.set_label_filter(None)
        self._sync_clear_button()
        self.status_label.setText(f"Label filter cleared — {shown} trial(s) shown by the column filters.")

    def showEvent(self, event):
        # Reopened after labels or a dataset changed: re-read before showing.
        self.refresh()
        super().showEvent(event)


def open_label_inconsistencies(meta, parent=None) -> LabelInconsistencyDialog | None:
    """Open (or raise) the dialog; ``None`` when no labels are loaded."""
    df = getattr(meta.app_state, "_all_labels_df", None)
    if df is None or df.empty:
        notify("No labels loaded — nothing to check.", severity="warning")
        return None
    dialog = LabelInconsistencyDialog(meta, parent=parent)
    dialog.setAttribute(Qt.WA_DeleteOnClose, False)
    return dialog
