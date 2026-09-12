"""Label bulk editing: curate / delete / purge / correct-offsets across a
chosen trial and label scope.

Opened from **Tools ▸ Label bulk editing…** (`top_bar.py`). Everything here
drives :class:`~ethograph.gui.widgets_curation.CurationPanel`'s own bulk
methods (`curate_trial_labels`, `delete_trial_labels`, `purge_trial_labels`,
`correct_offsets`) — this dialog is a form in front of them, not a second
implementation. Every one of these is also a :mod:`~ethograph.labels.workflow`
step (``curate_trials``, ``delete_labels``, ``purge_labels``,
``correct_offsets``), so anything doable here can be recorded and replayed.

Two choices apply to curate/delete/purge (offset correction is never scoped
by label class — see below):

* **Trials** — one of :data:`~ethograph.labels.workflow.TRIAL_SCOPE_CHOICES`
  (current trial / all trials / the trials table's filtered set / what its
  filters hide).
* **Label classes** — an explicit checklist with its own **All** checkbox,
  independent of the Curation section's drag-and-drop scope area: a bulk
  action's blast radius should always be exactly what this dialog shows on
  screen, not whatever happens to be sitting in the scope area. Starts with
  nothing ticked and **All** off — a destructive tool should never open with
  "affects everything" as the silent default.
"""

from __future__ import annotations

import logging

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from ethograph.gui.notify import notify
from ethograph.labels import workflow as wf

logger = logging.getLogger(__name__)

#: The purge spin box opens here — short enough to catch stray clicks and
#: jitter, not so short it silently keeps something meant as background.
_DEFAULT_PURGE_S = 0.010


def _curation_panel(meta):
    return getattr(getattr(meta, "labels_widget", None), "curation_panel", None)


class LabelBulkEditDialog(QDialog):
    """Curate / delete / purge over an explicit trial and label-class scope."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self.panel = _curation_panel(meta)
        self.setWindowTitle("Label bulk editing")
        self.setMinimumWidth(420)
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Curate, delete or purge labels across many trials at once — pick the "
            "trials and label classes below, then run one of the actions."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(intro)

        lay.addWidget(self._build_trials_group())
        lay.addWidget(self._build_labels_group())
        lay.addWidget(self._build_curate_group())
        lay.addWidget(self._build_delete_group())
        lay.addWidget(self._build_purge_group())
        lay.addWidget(self._build_correct_offsets_group())

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        lay.addWidget(buttons)

    def _build_trials_group(self) -> QGroupBox:
        group = QGroupBox("Trials")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Act on:"))
        self.trial_scope_combo = QComboBox()
        for key, text in wf.TRIAL_SCOPE_CHOICES.items():
            self.trial_scope_combo.addItem(text, key)
        # "Filtered" (what the table shows now) is the common case: narrow
        # the trials table to a metadata condition first, then act on it.
        default_index = self.trial_scope_combo.findData(wf.TRIAL_SCOPE_FILTERED)
        self.trial_scope_combo.setCurrentIndex(max(0, default_index))
        row.addWidget(self.trial_scope_combo, stretch=1)
        return group

    def _build_labels_group(self) -> QGroupBox:
        group = QGroupBox("Label classes")
        lay = QVBoxLayout(group)
        self.all_labels_cb = QCheckBox("All")
        # Starts unticked: opening this dialog must never default to "every
        # class" — the user picks classes (or explicitly checks All) each time.
        self.all_labels_cb.setChecked(False)
        self.all_labels_cb.toggled.connect(self._on_all_labels_toggled)
        lay.addWidget(self.all_labels_cb)

        self.label_list = QListWidget()
        mappings = getattr(self.meta.labels_widget, "_mappings", {}) or {}
        for label_id in sorted(lid for lid in mappings if isinstance(lid, int) and lid != 0):
            info = mappings.get(label_id, {})
            item = QListWidgetItem(f"{label_id} — {info.get('name', str(label_id))}")
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.label_list.addItem(item)
        self.label_list.setMaximumHeight(max(40, min(160, 22 * self.label_list.count() + 6)))
        lay.addWidget(self.label_list)
        return group

    def _build_curate_group(self) -> QGroupBox:
        group = QGroupBox("Curate")
        lay = QVBoxLayout(group)
        hint = QLabel("Every automated label in the classes above becomes curated. Manual labels stay manual.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(hint)
        self.curate_btn = QPushButton("Curate…")
        self.curate_btn.setAutoDefault(False)
        self.curate_btn.clicked.connect(self._curate)
        lay.addWidget(self.curate_btn)
        return group

    def _build_delete_group(self) -> QGroupBox:
        group = QGroupBox("Delete")
        lay = QVBoxLayout(group)
        hint = QLabel(
            "Deletes the events outright — manual and curated labels included, not just\n"
            "automated ones. Ctrl+Z can take a trial's deletion back while this session is open."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(hint)
        self.delete_btn = QPushButton("Delete…")
        self.delete_btn.setAutoDefault(False)
        self.delete_btn.setStyleSheet("color: #ff7b72;")
        self.delete_btn.clicked.connect(self._delete)
        lay.addWidget(self.delete_btn)
        return group

    def _build_purge_group(self) -> QGroupBox:
        group = QGroupBox("Purge short labels")
        lay = QVBoxLayout(group)
        hint = QLabel("Drops state-interval labels shorter than the threshold. Point events are never touched.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(hint)
        row = QHBoxLayout()
        row.addWidget(QLabel("Purge labels shorter than:"))
        self.purge_spin = QDoubleSpinBox()
        self.purge_spin.setRange(0.0, 100000.0)
        self.purge_spin.setDecimals(3)
        self.purge_spin.setSingleStep(0.005)
        self.purge_spin.setSuffix(" s")
        self.purge_spin.setValue(_DEFAULT_PURGE_S)
        row.addWidget(self.purge_spin)
        self.purge_btn = QPushButton("Purge…")
        self.purge_btn.setAutoDefault(False)
        self.purge_btn.setStyleSheet("color: #ff7b72;")
        self.purge_btn.clicked.connect(self._purge)
        row.addWidget(self.purge_btn)
        lay.addLayout(row)
        return group

    def _build_correct_offsets_group(self) -> QGroupBox:
        group = QGroupBox("Correct offsets")
        lay = QVBoxLayout(group)
        hint = QLabel(
            "Pulls back each label's offset across a near-zero gap to the next onset of the\n"
            "same subject, so every interval is strictly separated (pynapple can then resolve\n"
            "them). Not scoped by label class above — a subject's whole sequence has to be\n"
            "seen together to find a gap."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(hint)
        self.correct_offsets_btn = QPushButton("Correct offsets…")
        self.correct_offsets_btn.setAutoDefault(False)
        self.correct_offsets_btn.clicked.connect(self._correct_offsets)
        lay.addWidget(self.correct_offsets_btn)
        return group

    # ------------------------------------------------------------------
    # Reading the form
    # ------------------------------------------------------------------

    def _on_all_labels_toggled(self, checked: bool) -> None:
        self.label_list.setEnabled(not checked)

    def _trial_scope(self) -> str:
        return str(self.trial_scope_combo.currentData())

    def _label_ids(self) -> set[int] | None:
        """``None`` for every class ("All" ticked); else the checked ids."""
        if self.all_labels_cb.isChecked():
            return None
        ids = set()
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.add(int(item.data(Qt.UserRole)))
        return ids

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _guarded(self, run) -> None:
        """Run *run* unless "All" is off and nothing is ticked — an empty
        checklist means nothing, never every class (:func:`scope_mask` reads
        an empty set that way, which would silently touch everything the
        checklist appears to exclude)."""
        if self.panel is None:
            notify("No Curation section in this window.", severity="warning")
            return
        label_ids = self._label_ids()
        if label_ids is not None and not label_ids:
            notify("Tick at least one label class, or check All.", severity="warning")
            return
        run(label_ids)

    def _curate(self) -> None:
        self._guarded(lambda label_ids: self.panel.curate_trial_labels(self._trial_scope(), label_ids, confirm=True))

    def _delete(self) -> None:
        self._guarded(lambda label_ids: self.panel.delete_trial_labels(self._trial_scope(), label_ids, confirm=True))

    def _purge(self) -> None:
        self._guarded(
            lambda label_ids: self.panel.purge_trial_labels(
                self._trial_scope(), self.purge_spin.value(), label_ids, confirm=True
            )
        )

    def _correct_offsets(self) -> None:
        # No label-class guard: offset correction reads a whole subject's
        # sequence, so the checklist above does not apply to it.
        if self.panel is None:
            notify("No Curation section in this window.", severity="warning")
            return
        self.panel.correct_offsets(self._trial_scope(), confirm=True)
