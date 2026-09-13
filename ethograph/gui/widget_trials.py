"""Trials metadata widget: tabular display, filtering, and in-place editing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.notify import notify
from ethograph.io.metadata_edit import (
    TARGET_NWB,
    blank_column,
    coerce_value,
    ensure_tabular_target,
    fits_dtype,
    resolve_metadata_target,
    write_metadata,
)
from ethograph.io.metadata_table import condition_columns

logger = logging.getLogger(__name__)

_MAX_INT_CATEGORICAL_VALUES = 15

#: The current trial's row, while editing is on — the row you can change.
_CURRENT_ROW_COLOR = QColor(80, 140, 210, 60)

#: Edits are written this long after the last one (an NWB trials table is
#: expensive to rewrite, and editing comes in bursts).
_SAVE_DELAY_MS = 1200


class _CatFilterDialog(QDialog):
    """Checkbox popup for categorical column filtering."""

    def __init__(self, all_values: list[str], active: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)

        self._all_cb = QCheckBox("(All)")
        self._all_cb.setChecked(not active)
        layout.addWidget(self._all_cb)

        self._checks: list[tuple[str, QCheckBox]] = []
        for val in sorted(all_values):
            cb = QCheckBox(val)
            cb.setChecked((not active) or (val in active))
            layout.addWidget(cb)
            self._checks.append((val, cb))

        self._all_cb.toggled.connect(self._on_all)
        for _, cb in self._checks:
            cb.toggled.connect(self._on_item)

        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def _on_all(self, checked: bool):
        for _, cb in self._checks:
            cb.setChecked(checked)

    def _on_item(self, _):
        self._all_cb.blockSignals(True)
        self._all_cb.setChecked(all(cb.isChecked() for _, cb in self._checks))
        self._all_cb.blockSignals(False)

    def get_allowed(self) -> set[str]:
        checked = {v for v, cb in self._checks if cb.isChecked()}
        return set() if checked == {v for v, _ in self._checks} else checked


class _NumFilterDialog(QDialog):
    """Threshold filter dialog for numeric columns."""

    def __init__(self, current: tuple[str, float] | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter")
        self._cleared = False

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        op_row = QHBoxLayout()
        self._op_combo = QComboBox()
        self._op_combo.addItems(["≥", "≤"])
        op_row.addWidget(self._op_combo)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(-1e9, 1e9)
        self._spin.setDecimals(6)
        op_row.addWidget(self._spin)
        layout.addLayout(op_row)

        if current:
            op, value = current
            self._op_combo.setCurrentText("≥" if op == ">=" else "≤")
            self._spin.setValue(value)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        clear_btn = QPushButton("Remove filter")
        clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

    def _clear(self):
        self._cleared = True
        self.accept()

    def get_filter(self) -> tuple[str, float] | None:
        if self._cleared:
            return None
        return (
            ">=" if self._op_combo.currentText() == "≥" else "<=",
            self._spin.value(),
        )


class _FilterHeaderView(QHeaderView):
    """Column header that draws filter icons and opens filter dialogs on click."""

    filter_requested = Signal(int)
    _FILTER_ZONE_W = 20

    def __init__(self, cat_cols: set[int], num_cols: set[int], parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._cat_cols = cat_cols
        self._num_cols = num_cols
        self._active: set[int] = set()
        self.setSectionsClickable(True)

    @property
    def _all_filterable(self) -> set[int]:
        return self._cat_cols | self._num_cols

    def set_filterable(self, cat_cols: set[int], num_cols: set[int]):
        self._cat_cols = cat_cols
        self._num_cols = num_cols
        self.viewport().update()

    def set_active_filters(self, active: set[int]):
        self._active = active
        self.viewport().update()

    def _filter_zone_x(self, logical: int) -> int:
        return self.sectionViewportPosition(logical) + self.sectionSize(logical) - self._FILTER_ZONE_W

    def _icon_rect(self, logical: int) -> QRect:
        size = 11
        zone_x = self._filter_zone_x(logical)
        h = self.height()
        x = zone_x + (self._FILTER_ZONE_W - size) // 2
        return QRect(x, (h - size) // 2, size, size)

    def paintSection(self, painter, rect, logical):
        painter.save()
        super().paintSection(painter, rect, logical)
        painter.restore()

        if logical not in self._all_filterable:
            return

        zone_x = rect.right() - self._FILTER_ZONE_W + 1
        painter.save()
        painter.setPen(QPen(QColor(120, 120, 120, 80), 1))
        painter.drawLine(zone_x, rect.top() + 3, zone_x, rect.bottom() - 3)
        painter.restore()

        ir = QRect(
            zone_x + (self._FILTER_ZONE_W - 11) // 2,
            rect.top() + (rect.height() - 11) // 2,
            11,
            11,
        )
        x, y, s = ir.x(), ir.y(), ir.width()
        color = QColor(255, 215, 0) if logical in self._active else QColor(180, 180, 180)
        painter.save()
        painter.setPen(QPen(color, 1.5))
        painter.drawLine(x, y, x + s, y)
        painter.drawLine(x + 2, y + 4, x + s - 2, y + 4)
        painter.drawLine(x + 4, y + 8, x + s - 4, y + 8)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            logical = self.logicalIndexAt(event.pos())
            if logical in self._all_filterable:
                section_right = self.sectionViewportPosition(logical) + self.sectionSize(logical)
                if event.pos().x() >= section_right - self._FILTER_ZONE_W:
                    self.filter_requested.emit(logical)
                    event.accept()
                    return
        super().mousePressEvent(event)


class _ValueDelegate(QStyledItemDelegate):
    """Cell editor: pick a value the column already uses, or type a new one.

    Metadata columns are almost always a small set of repeated values, so the
    editor offers them — but it is an editable combo, not a fixed list: adding
    a value the column has never held has to cost nothing.
    """

    def __init__(self, options_for_column, parent=None):
        super().__init__(parent)
        self._options_for_column = options_for_column

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.addItems(self._options_for_column(index.column()))
        completer = combo.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
        return combo

    def setEditorData(self, editor, index):
        editor.setCurrentText(str(index.data(Qt.DisplayRole) or ""))
        line_edit = editor.lineEdit()
        if line_edit is not None:
            line_edit.selectAll()

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText().strip(), Qt.EditRole)


class TrialsWidget(QWidget):
    """Tabular metadata display with combinatorial filtering.

    Shows one row per trial, all metadata columns, and a dynamic filter
    combo per condition column.  Filters are AND-combined.

    Signals
    -------
    trials_filtered : list
        Emitted when active filters change; carries the filtered trial list.
    """

    trials_filtered = Signal(list)

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._metadata_df: pd.DataFrame = pd.DataFrame()
        self._base_trials: list = []
        self._cat_cols: set[int] = set()
        self._num_cols: set[int] = set()
        self._cat_values: dict[int, list[str]] = {}
        self._cat_active: dict[int, set[str]] = {}
        self._num_active: dict[int, tuple[str, float]] = {}
        #: Trials the label filter allows (see :meth:`set_label_filter`), or
        #: None when it is off. Session state — a dataset load starts clean.
        self._label_trials: set[str] | None = None
        self._label_note: str = ""
        self._editable_cols: set[int] = set()
        self._dirty_columns: set[str] = set()
        self._building = False
        self._applying = False

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)

        # Editing: off by default, so a stray double-click never rewrites the
        # metadata file. Stays available with no metadata yet — that is exactly
        # when the first column gets added.
        edit_row = QHBoxLayout()
        self._edit_checkbox = QCheckBox("Edit values on double-click")
        self._edit_checkbox.setToolTip(
            "Double-click a cell in the current trial's row to change its value.\n"
            "Values the column already uses are offered as you type; anything else is accepted.\n"
            "Saved straight to the metadata file."
        )
        edit_row.addWidget(self._edit_checkbox)
        edit_row.addStretch()
        self._add_column_button = QPushButton("Add column…")
        self._add_column_button.setToolTip("Add an empty metadata column to fill in per trial.")
        self._add_column_button.clicked.connect(self._on_add_column)
        edit_row.addWidget(self._add_column_button)
        layout.addLayout(edit_row)

        self._empty_label = QLabel("No trial metadata yet — start one with “Add column…”.")
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._empty_label)

        # Table
        self._table = QTableWidget()
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.setItemDelegate(_ValueDelegate(self._column_options, self._table))
        self._table.cellClicked.connect(self._on_row_clicked)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(280)
        layout.addWidget(self._table)

        # Status
        self._status_label = QLabel()
        layout.addWidget(self._status_label)

        self._note_label = QLabel("Visible rows define which trials are used for trial navigation.")
        self._note_label.setWordWrap(True)
        self._note_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._note_label)

        # Writes are batched: one file write per burst of edits, plus one
        # whenever the user moves on to another trial.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(_SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self.flush_metadata)
        app_state.trial_changed.connect(self.flush_metadata)
        app_state.trial_changed.connect(self._apply_edit_state)
        app_state.trial_changed.connect(self._sync_selection_to_trial)

        # Restored last — _on_edit_toggled configures the table, which only
        # exists by now.
        self._edit_checkbox.toggled.connect(self._on_edit_toggled)
        self._edit_checkbox.setChecked(bool(app_state.get_with_default("metadata_edit_enabled")))

        self._update_visibility()

    def setup(self, metadata_df: pd.DataFrame) -> None:
        """Populate the widget from a metadata DataFrame."""
        self._building = True
        if "trial" not in metadata_df.columns:
            raise ValueError("TrialsWidget requires a metadata table with a 'trial' column")
        self._metadata_df = metadata_df.copy()
        self._base_trials = self._metadata_df["trial"].dropna().drop_duplicates().tolist()
        self._cat_active.clear()
        self._num_active.clear()

        # Populate table
        self._refresh_table()
        # Initial order: ascending by trial. Items store numeric trial IDs
        # natively (DisplayRole), so this sorts 1, 2, … 10, not "1", "10", "2".
        trial_col = list(self._metadata_df.columns).index("trial")
        self._table.sortByColumn(trial_col, Qt.AscendingOrder)
        self._setup_filter_header()

        self._building = False
        self._apply_filters()
        self._apply_edit_state()
        self._sync_selection_to_trial()
        self._update_visibility()

    def _update_visibility(self) -> None:
        """Show the table only when there is real metadata to filter on (i.e.
        columns beyond the bare 'trial' number). The editing controls stay
        visible either way — with no metadata yet is exactly when the first
        column gets added."""
        has_metadata = self._has_metadata()
        self._table.setVisible(has_metadata)
        self._status_label.setVisible(has_metadata)
        self._note_label.setVisible(has_metadata)
        self._empty_label.setVisible(not has_metadata)

    def _has_metadata(self) -> bool:
        df = getattr(self, "_metadata_df", None)
        if df is None or df.empty:
            return False
        return any(str(c) != "trial" for c in df.columns)

    def _refresh_table(self) -> None:
        """Rebuild the table from _metadata_df."""
        df = self._metadata_df
        # Every setItem below emits itemChanged; none of them is a user edit.
        self._applying = True
        try:
            self._rebuild_table_items(df)
        finally:
            self._applying = False
        self._editable_cols = {
            list(df.columns).index(name) for name in condition_columns(df) if name in list(df.columns)
        }

    def _rebuild_table_items(self, df: pd.DataFrame) -> None:
        if df.empty:
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return

        cols = list(df.columns)
        self._table.setSortingEnabled(False)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(df))

        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(cols):
                val = row[col]
                item = QTableWidgetItem()
                if pd.notna(val):
                    # Store numeric types natively so Qt sorts numerically
                    if isinstance(val, (int, np.integer)):
                        item.setData(Qt.DisplayRole, int(val))
                    elif isinstance(val, (float, np.floating)):
                        item.setData(Qt.DisplayRole, float(val))
                    else:
                        item.setData(Qt.DisplayRole, str(val))
                else:
                    item.setData(Qt.DisplayRole, "")
                item.setData(Qt.UserRole, row.get("trial"))
                self._table.setItem(r, c, item)

        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _setup_filter_header(self) -> None:
        self._cat_cols.clear()
        self._num_cols.clear()
        self._cat_values.clear()
        col_names = list(self._metadata_df.columns)

        df = self._metadata_df
        for col_idx, col_name in enumerate(df.columns):
            series = df[col_name].dropna()
            if series.empty:
                self._cat_cols.add(col_idx)
                self._cat_values[col_idx] = []
                continue

            if pd.api.types.is_integer_dtype(series) and series.nunique() <= _MAX_INT_CATEGORICAL_VALUES:
                self._cat_cols.add(col_idx)
                int_vals = sorted({int(v) for v in series.tolist()})
                self._cat_values[col_idx] = [str(v) for v in int_vals]
                continue

            if pd.api.types.is_bool_dtype(series):
                self._cat_cols.add(col_idx)
                self._cat_values[col_idx] = sorted({str(v) for v in series.tolist()})
                continue

            if pd.api.types.is_numeric_dtype(series):
                self._num_cols.add(col_idx)
            else:
                self._cat_cols.add(col_idx)
                self._cat_values[col_idx] = sorted({str(v) for v in series.tolist()})

        header = _FilterHeaderView(self._cat_cols, self._num_cols, self._table)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        header.setVisible(True)
        header.filter_requested.connect(self._on_filter_header_clicked)
        # Carry the sort over to the replacement header: setHorizontalHeader()
        # re-runs setSortingEnabled() internally, which re-sorts by the NEW
        # header's indicator — and a fresh QHeaderView defaults to section 0,
        # *descending*. Without this the table lands on trial 10, 9, 8, … and a
        # rebuild (reload_metadata) silently discards the user's sort column.
        old_header = self._table.horizontalHeader()
        section = old_header.sortIndicatorSection() if old_header is not None else 0
        order = old_header.sortIndicatorOrder() if old_header is not None else Qt.AscendingOrder
        header.setSortIndicator(max(0, section), order)
        self._table.setHorizontalHeader(header)
        # Re-apply header labels after replacing the header view to ensure text is visible.
        self._table.setHorizontalHeaderLabels(col_names)
        self._update_header_active_filters()

    def _on_filter_header_clicked(self, logical_col: int) -> None:
        if logical_col in self._cat_cols:
            active = self._cat_active.get(logical_col, set())
            dialog = _CatFilterDialog(self._cat_values.get(logical_col, []), active, self)
            if dialog.exec_() != QDialog.Accepted:
                return
            allowed = dialog.get_allowed()
            if allowed:
                self._cat_active[logical_col] = allowed
            else:
                self._cat_active.pop(logical_col, None)
            self._update_header_active_filters()
            self._apply_filters()
            return

        if logical_col in self._num_cols:
            current = self._num_active.get(logical_col)
            dialog = _NumFilterDialog(current, self)
            if dialog.exec_() != QDialog.Accepted:
                return
            value = dialog.get_filter()
            if value is None:
                self._num_active.pop(logical_col, None)
            else:
                self._num_active[logical_col] = value
            self._update_header_active_filters()
            self._apply_filters()

    def _update_header_active_filters(self) -> None:
        header = self._table.horizontalHeader()
        if isinstance(header, _FilterHeaderView):
            active = set(self._cat_active.keys()) | set(self._num_active.keys())
            header.set_active_filters(active)

    def _apply_filters(self) -> None:
        """Compute filtered trial list and emit signal."""
        original_trials = set(self._base_trials)
        filtered = set(original_trials)

        mdf = self._metadata_df
        if not mdf.empty:
            for _, row in mdf.iterrows():
                trial = row.get("trial")
                if pd.isna(trial):
                    continue

                keep = True
                for col_idx, allowed in self._cat_active.items():
                    col_name = mdf.columns[col_idx]
                    if str(row.get(col_name, "")) not in allowed:
                        keep = False
                        break

                if keep:
                    for col_idx, (op, threshold) in self._num_active.items():
                        col_name = mdf.columns[col_idx]
                        val = row.get(col_name)
                        if pd.isna(val):
                            keep = False
                            break
                        if op == ">=" and float(val) < threshold:
                            keep = False
                            break
                        if op == "<=" and float(val) > threshold:
                            keep = False
                            break

                if not keep and trial in filtered:
                    filtered.remove(trial)

        # The label filter sits ON TOP of the column filters rather than
        # replacing them: "wild-type trials where the sequence broke" is one
        # question, and answering it must not throw the genotype away.
        if self._label_trials is not None:
            filtered = {t for t in filtered if str(t) in self._label_trials}

        try:
            sorted_trials = sorted(filtered, key=int)
        except (ValueError, TypeError):
            sorted_trials = sorted(filtered, key=str)

        # Never produce an empty trial list — fall back to unfiltered.
        if not sorted_trials:
            sorted_trials = list(self._base_trials)

        # Keep table and navigation in sync with filtered subset.
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, 0)
            trial_id = item.data(Qt.UserRole) if item is not None else None
            self._table.setRowHidden(row_idx, trial_id not in filtered)

        self.app_state.trials = sorted_trials

        n_total = len(original_trials)
        n_shown = len(sorted_trials)
        note = f"Showing {n_shown} of {n_total} trials"
        if self._label_trials is not None:
            # Say it out loud: a filter the header funnels cannot show would
            # otherwise look like trials going missing for no reason.
            note += f" · label filter: {self._label_note or 'on'}"
        self._status_label.setText(note)

        self.trials_filtered.emit(sorted_trials)

    def all_trials(self) -> list:
        """Every trial this dataset has, filters or no — what ``app_state.trials``
        (the visible/filtered subset) is drawn from."""
        return list(self._base_trials)

    # ------------------------------------------------------------------
    # Filters by column name — how a saved curation workflow replays them
    # ------------------------------------------------------------------

    def column_filters(self) -> list[dict]:
        """The active filters, keyed by column name rather than index.

        The serialisable form :mod:`ethograph.labels.workflow` stores: one
        entry per filtered column, carrying either ``values`` (categorical)
        or ``op``/``value`` (numeric).
        """
        columns = list(self._metadata_df.columns)
        out: list[dict] = []
        for col_idx, allowed in sorted(self._cat_active.items()):
            if col_idx < len(columns):
                out.append({"column": str(columns[col_idx]), "values": sorted(str(v) for v in allowed)})
        for col_idx, (op, value) in sorted(self._num_active.items()):
            if col_idx < len(columns):
                out.append({"column": str(columns[col_idx]), "op": str(op), "value": float(value)})
        return out

    def filterable_columns(self) -> dict[str, list[str] | None]:
        """Column name → its categorical values, or ``None`` when numeric."""
        out: dict[str, list[str] | None] = {}
        for col_idx, name in enumerate(self._metadata_df.columns):
            if col_idx in self._cat_cols:
                out[str(name)] = list(self._cat_values.get(col_idx, []))
            elif col_idx in self._num_cols:
                out[str(name)] = None
        return out

    def clear_column_filters(self) -> None:
        """Drop every filter, so every trial is shown again."""
        self._cat_active.clear()
        self._num_active.clear()
        self._label_trials = None
        self._label_note = ""
        self._update_header_active_filters()
        self._apply_filters()

    # ------------------------------------------------------------------
    # Label filter — a restriction the metadata columns cannot express
    # ------------------------------------------------------------------

    def set_label_filter(self, trials, note: str = "") -> int:
        """Restrict to *trials* on top of the column filters; ``None`` clears it.

        Set by **Tools ▸ Labels: Find label inconsistencies…**, which works out which
        trials look wrong from the labels themselves — something no metadata
        column knows. It is deliberately a separate slot from the column
        filters: clearing one leaves the other alone, and the status line says
        when it is on. Returns how many trials the table now shows.
        """
        self._label_trials = None if trials is None else {str(t) for t in trials}
        self._label_note = note if trials is not None else ""
        self._apply_filters()
        return len(self.app_state.trials or [])

    def label_filter_active(self) -> bool:
        return self._label_trials is not None

    def apply_column_filters(self, filters: list[dict], *, clear_first: bool = True) -> list[str]:
        """Apply filters named by column, as :meth:`column_filters` returns them.

        Returns the entries that could not be applied, as sentences — a column
        this dataset's metadata does not have, or a comparison on a column
        that is not numeric here. Applying nothing is not an error: the
        remaining filters still take effect.
        """
        columns = {str(name): idx for idx, name in enumerate(self._metadata_df.columns)}
        if clear_first:
            self._cat_active.clear()
            self._num_active.clear()
        skipped: list[str] = []
        for entry in filters or []:
            column = str(entry.get("column", ""))
            col_idx = columns.get(column)
            if col_idx is None:
                skipped.append(f"no metadata column {column!r}")
                continue
            values = entry.get("values")
            if isinstance(values, list) and values:
                if col_idx not in self._cat_cols:
                    skipped.append(f"{column!r} is numeric here, so a value list does not apply")
                    continue
                self._cat_active[col_idx] = {str(v) for v in values}
                known = set(self._cat_values.get(col_idx, []))
                missing = sorted({str(v) for v in values} - known)
                if missing:
                    skipped.append(f"{column!r} has no value(s) {', '.join(missing)} in this dataset")
                continue
            op, value = entry.get("op"), entry.get("value")
            if op in (">=", "<=") and value is not None:
                if col_idx not in self._num_cols:
                    skipped.append(f"{column!r} is categorical here, so a comparison does not apply")
                    continue
                self._num_active[col_idx] = (str(op), float(value))
                continue
            skipped.append(f"the filter on {column!r} states no condition")
        self._update_header_active_filters()
        self._apply_filters()
        return skipped

    # ==================================================================
    # Editing metadata in place (double-click, current trial only)
    # ==================================================================

    def _on_edit_toggled(self, enabled: bool) -> None:
        self.app_state.metadata_edit_enabled = enabled
        self._table.setEditTriggers(QTableWidget.DoubleClicked if enabled else QTableWidget.NoEditTriggers)
        self._apply_edit_state()

    def _apply_edit_state(self) -> None:
        """Make exactly the current trial's condition cells editable, and mark
        that row.

        Only the open trial can be edited: the value being typed describes what
        the user is looking at, and a double-click two rows down would
        otherwise silently score a trial they never saw.
        """
        enabled = self._edit_checkbox.isChecked()
        current_row = self._current_trial_row() if enabled else None
        self._applying = True
        try:
            for row in range(self._table.rowCount()):
                background = _CURRENT_ROW_COLOR if row == current_row else None
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    if item is None:
                        continue
                    editable = row == current_row and col in self._editable_cols
                    flags = item.flags()
                    item.setFlags(flags | Qt.ItemIsEditable if editable else flags & ~Qt.ItemIsEditable)
                    item.setData(Qt.BackgroundRole, background)
        finally:
            self._applying = False

    def _current_trial_row(self) -> int | None:
        return self._row_of_trial(getattr(self.app_state, "trials_sel", None))

    def _sync_selection_to_trial(self) -> None:
        """Select the current trial's row, exactly as a click would.

        Navigation runs both ways: clicking a row navigates to that trial, and
        navigating to a trial (Next/Previous, combo, label/sequence, jump)
        selects its row here. ``selectRow`` never fires ``cellClicked``, so
        this cannot loop back into ``_on_row_clicked``.
        """
        row = self._current_trial_row()
        if row is None:
            self._table.clearSelection()
            return
        if self._table.currentRow() != row:
            self._table.selectRow(row)
        item = self._table.item(row, 0)
        if item is not None:
            self._table.scrollToItem(item)

    def _row_of_trial(self, trial) -> int | None:
        if trial is None:
            return None
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and str(item.data(Qt.UserRole)) == str(trial):
                return row
        return None

    def _column_options(self, col_idx: int) -> list[str]:
        """Values this column already uses, for the editor's autocomplete."""
        cols = list(self._metadata_df.columns)
        if not 0 <= col_idx < len(cols):
            return []
        series = self._metadata_df[cols[col_idx]].dropna()
        return sorted({str(v) for v in series.tolist() if str(v) != ""})

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """A cell was committed by the delegate — write it through."""
        if self._building or self._applying:
            return
        col = item.column()
        if col not in self._editable_cols:
            return
        row_head = self._table.item(item.row(), 0)
        trial = row_head.data(Qt.UserRole) if row_head is not None else None
        if trial is None:
            return
        column = list(self._metadata_df.columns)[col]
        self.set_metadata_value(trial, column, str(item.data(Qt.DisplayRole) or ""))

    def set_metadata_value(self, trial, column: str, text: str) -> None:
        """Set *column* to *text* for *trial*, then show and save it.

        Filters are deliberately NOT re-applied: editing a filtered column
        would otherwise hide the trial being edited, out from under the user.
        The filter picks the change up the next time it is opened.
        """
        df = getattr(self.app_state, "metadata_df", None)
        df = self._metadata_df.copy() if df is None else df.copy()
        mask = df["trial"].astype(str) == str(trial)
        if not mask.any():
            notify(f"Trial {trial} has no row in the metadata table.", "warning")
            return

        value = coerce_value(text, df[column] if column in df.columns else None)
        if column not in df.columns:
            df[column] = blank_column(df)
        elif not fits_dtype(df[column], value):
            df[column] = df[column].astype(object)
        df.loc[mask, column] = value

        self.app_state.metadata_df = df
        self._metadata_df = df.copy()
        self._show_value(trial, column, value)
        self._register_filter_value(column, value)
        self._queue_save(column)

    def set_column_values(self, column: str, values: dict[str, object]) -> None:
        """Set *column* for many trials at once (keyed by trial id as text), then save.

        The curation sync's path: it runs on a timer, so a new column costs one
        table rebuild and an existing one only the changed cells. Filters are
        left alone for the same reason :meth:`set_metadata_value` leaves them.
        """
        df = getattr(self.app_state, "metadata_df", None)
        df = self._metadata_df.copy() if df is None else df.copy()
        if df.empty or "trial" not in df.columns:
            return
        new_column = column not in df.columns
        if new_column:
            df[column] = blank_column(df)
        trials = df["trial"].astype(str)
        hit = trials.isin(values.keys())
        if not hit.any():
            return
        incoming = [values[t] for t in trials[hit]]
        if not all(fits_dtype(df[column], v) for v in incoming):
            df[column] = df[column].astype(object)
        df.loc[hit, column] = incoming

        self.app_state.metadata_df = df
        self._metadata_df = df.copy()
        if new_column:
            self.reload_metadata()
        else:
            for trial, value in values.items():
                self._show_value(trial, column, value)
        for value in set(values.values()):
            self._register_filter_value(column, value)
        self._queue_save(column)

    def _show_value(self, trial, column: str, value) -> None:
        """Put the stored (typed) value in the cell."""
        item = self._cell_item(trial, column)
        if item is None:
            return
        self._applying = True
        self._table.setSortingEnabled(False)
        try:
            if isinstance(value, (int, np.integer)):
                item.setData(Qt.DisplayRole, int(value))
            elif isinstance(value, (float, np.floating)):
                item.setData(Qt.DisplayRole, float(value))
            else:
                item.setData(Qt.DisplayRole, str(value))
        finally:
            self._table.setSortingEnabled(True)
            self._applying = False

    def _on_add_column(self) -> None:
        """Add an empty metadata column, ready to fill in per trial."""
        if not getattr(self.app_state, "ready", False):
            notify("Load a dataset before adding metadata columns.", "warning")
            return
        name, ok = QInputDialog.getText(self, "New metadata column", "Column name:")
        name = name.strip()
        if not ok or not name:
            return
        df = getattr(self.app_state, "metadata_df", None)
        df = self._metadata_df.copy() if df is None else df.copy()
        if name in df.columns:
            notify(f"Column {name!r} already exists.", "warning")
            return

        df[name] = blank_column(df)
        self.app_state.metadata_df = df
        self.reload_metadata()
        self._queue_save(name)
        # Adding a column is asking to fill it in.
        self._edit_checkbox.setChecked(True)

    # -- saving -------------------------------------------------------

    def _queue_save(self, column: str) -> None:
        self._dirty_columns.add(column)
        self._save_timer.start()

    def _alignment_nwb_path(self) -> str | None:
        alignment = getattr(self.app_state, "nwb_alignment", None)
        path = getattr(alignment, "_path", None)
        if path is None or Path(path).suffix.lower() != ".nwb":
            return None
        return path

    def _metadata_target(self):
        return resolve_metadata_target(
            getattr(self.app_state, "nc_file_path", None),
            metadata_path=getattr(self.app_state, "metadata_path", None),
            alignment_path=self._alignment_nwb_path(),
        )

    def ensure_tabular_metadata_file(self) -> Path | None:
        """Make sure the metadata table lives in a file we may write freely.

        Curation calls this when it starts: its ``curated`` column is derived
        state and must not go into a recording or the alignment NWB
        (``io/metadata_edit.DERIVED_COLUMNS``). With an NWB target the loaded
        table is copied to the sidecar TSV, which becomes the metadata table
        for this dataset from then on.
        """
        target = ensure_tabular_target(
            getattr(self.app_state, "nc_file_path", None),
            getattr(self.app_state, "metadata_df", None),
            metadata_path=getattr(self.app_state, "metadata_path", None),
            alignment_path=self._alignment_nwb_path(),
            trials=getattr(self.app_state, "trials", None),
        )
        if target is None:
            return None
        if str(target.path) != str(getattr(self.app_state, "metadata_path", None) or ""):
            # Reloads the table from the TSV and makes it what the next load
            # reads first.
            self.app_state.metadata_path = str(target.path)
            self.reload_metadata()
        return target.path

    def flush_metadata(self) -> None:
        """Write pending edits to the metadata source. A no-op when there are none."""
        self._save_timer.stop()
        if not self._dirty_columns:
            return
        target = self._metadata_target()
        df = getattr(self.app_state, "metadata_df", None)
        if target is None or df is None:
            self._dirty_columns.clear()
            return

        alignment = getattr(self.app_state, "nwb_alignment", None) if target.kind == TARGET_NWB else None
        try:
            if alignment is not None:
                # pynwb holds the file open for reading; HDF5 refuses a second,
                # writable handle while it does.
                alignment.reload()
            write_metadata(target, df, columns=sorted(self._dirty_columns))
        except (OSError, ValueError, KeyError) as e:
            logger.exception("Writing metadata to %s failed", target.path)
            notify(f"Could not save metadata: {e}", "error")
            return  # stay dirty — the next flush retries
        finally:
            if alignment is not None:
                alignment.reload()

        self._dirty_columns.clear()
        if target.kind != TARGET_NWB and not getattr(self.app_state, "metadata_path", None):
            # Make the sidecar the authoritative source from now on.
            self.app_state.metadata_path = str(target.path)

    def reload_metadata(self) -> None:
        """Rebuild the table from ``app_state.metadata_df``, keeping filters.

        Used when the shape of the table changed (a new column). Active
        filters survive because new columns are appended, so the existing
        column indices they key on still point at the same columns.
        """
        df = getattr(self.app_state, "metadata_df", None)
        if df is None or df.empty:
            return
        cat_active, num_active = dict(self._cat_active), dict(self._num_active)
        self._metadata_df = df.copy()
        self._base_trials = self._metadata_df["trial"].dropna().drop_duplicates().tolist()
        self._refresh_table()
        self._setup_filter_header()
        self._cat_active, self._num_active = cat_active, num_active
        self._update_header_active_filters()
        self._apply_edit_state()
        self._sync_selection_to_trial()
        self._update_visibility()

    def _cell_item(self, trial, column: str) -> QTableWidgetItem | None:
        cols = list(self._metadata_df.columns)
        if column not in cols:
            return None
        col_idx = cols.index(column)
        for row in range(self._table.rowCount()):
            first = self._table.item(row, 0)
            if first is not None and str(first.data(Qt.UserRole)) == str(trial):
                return self._table.item(row, col_idx)
        return None

    def _register_filter_value(self, column: str, value) -> None:
        """Make a newly introduced value selectable in that column's filter."""
        cols = list(self._metadata_df.columns)
        if column not in cols:
            return
        col_idx = cols.index(column)
        if col_idx not in self._cat_cols:
            return
        values = self._cat_values.setdefault(col_idx, [])
        if str(value) not in values:
            values.append(str(value))
            values.sort()

    def _on_row_clicked(self, row: int, _col: int) -> None:
        """Navigate to the trial in the clicked row."""
        item = self._table.item(row, 0)
        if item is None:
            return
        trial_id = item.data(Qt.UserRole)
        if trial_id is not None and trial_id in set(self.app_state.trials) and hasattr(self.app_state, "set_key_sel"):
            self.app_state.set_key_sel("trials", trial_id)
            self.app_state.trial_changed.emit()
