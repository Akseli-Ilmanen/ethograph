"""Spreadsheet view of the whole label table.

The Labels tab shows one trial's *classes*; this shows what
:mod:`ethograph.labels.tsv_store` is holding right now — every trial's rows, in
the columns that get written to ``_labels.tsv`` — and lets small corrections be
made in place: retype a cell, delete a selection of rows.

Only the per-label columns (:data:`~ethograph.labels.intervals.INTERVAL_COLUMNS`)
are editable. ``trial`` and the per-trial metadata columns
(:data:`~ethograph.labels.tsv_store.TRIAL_META_COLUMNS`) hold one value for a
whole trial repeated on every row, so editing one row's cell would only desync
it from its siblings.

Every change is recorded with ``app_state.record_label_edit`` before it runs,
so ``Ctrl+Z`` in the main window takes it back like any other label edit — one
step per trial the change touched.

Bulk edits (right-click a column header, or the "Find & replace…" button) run
over whatever is currently visible in the table, so an active filter narrows
their scope like everything else here.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from qtpy.QtCore import QEvent, QItemSelectionModel, Qt
from qtpy.QtGui import QColor, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
)

from ethograph.gui.notify import notify
from ethograph.gui.table_filter import (
    SORT_ROLE,
    CategoryFilterDialog,
    FilterHeaderView,
    MultiColumnFilterProxy,
    NumericFilterDialog,
)
from ethograph.labels.intervals import (
    EVENT_TYPE_POINT,
    EVENT_TYPE_STATE,
    INTERVAL_COLUMNS,
    INTERVAL_DTYPES,
    LABEL_SCHEMA,
    LABELING_METHODS,
)
from ethograph.labels.tsv_store import TRIAL_META_COLUMNS, TSV_COLUMNS

logger = logging.getLogger(__name__)

#: Title of the derived leading column: the label class's name and colour.
NAME_COLUMN_TITLE = "label"

#: Position of a row in ``_all_labels_df``, carried on the first cell of a row.
POSITION_ROLE = Qt.UserRole + 2

#: Columns whose values come from a fixed vocabulary — edited with a combo.
CHOICE_COLUMNS: dict[str, tuple[str, ...]] = {
    "event_type": (EVENT_TYPE_STATE, EVENT_TYPE_POINT),
    "labeling_method": tuple(LABELING_METHODS),
}

#: Columns shown with a fixed number of decimals (the stored value is untouched).
_DECIMALS = {col.name: col.decimals for col in LABEL_SCHEMA if col.decimals is not None}

#: Columns filtered by a threshold rather than a checklist.
_NUMERIC_FILTER_COLUMNS = {col.name for col in LABEL_SCHEMA if col.numeric_filter}

#: Columns eligible for find & replace: string-valued and editable. Numeric
#: columns (onset_s, offset_s, confidence, labels) are excluded — a substring
#: match against a number's text is a different, riskier kind of edit than the
#: same match against a name, so it stays out of this tool.
_TEXT_COLUMNS = tuple(c for c in INTERVAL_COLUMNS if INTERVAL_DTYPES.get(c, object) is object)

_TABLE_STYLE = """
    QTableView { gridline-color: #555; background: #3b3b3b; color: #fff; }
    QTableView::item { padding: 0px 3px; }
    QTableView::item:selected { background: #3a5070; color: #fff; }
    QHeaderView::section {
        padding: 2px 4px; background: #888; color: #fff;
        border: none; border-right: 1px solid #666; font-size: 11px;
    }
"""


def _owns_key(event) -> bool:
    """Whether *event* is a key the table means something by."""
    modifiers = event.modifiers()
    if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
        return not modifiers
    if modifiers & Qt.ControlModifier:
        return event.key() in (Qt.Key_A, Qt.Key_C)
    return False


def parse_cell(column: str, text: str):
    """The value *text* stands for in *column*, or raise ``ValueError``.

    An empty ``offset_s`` is a point event's missing end, so it parses to NaN;
    everywhere else an empty number is a mistake, not a value.
    """
    text = text.strip()
    if column in CHOICE_COLUMNS:
        if text not in CHOICE_COLUMNS[column]:
            raise ValueError(f"{column} must be one of: {', '.join(CHOICE_COLUMNS[column])}")
        return text

    dtype = INTERVAL_DTYPES.get(column, object)
    if dtype is object:
        return text
    if not text:
        if column == "offset_s":
            return float("nan")
        raise ValueError(f"{column} needs a number")
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{column} needs a number, not {text!r}") from None
    if np.issubdtype(dtype, np.integer):
        return int(round(number))
    return number


def _replace_value(
    current: str, find_text: str, replace_text: str, whole_cell: bool, case_sensitive: bool
) -> str | None:
    """The replaced text, or ``None`` when *current* does not match at all.

    *replace_text* is inserted literally — a lambda substitution, so a ``\\1``
    or backslash the user types is never read as a regex backreference.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(find_text), flags)
    if whole_cell:
        return replace_text if pattern.fullmatch(current) else None
    if not pattern.search(current):
        return None
    return pattern.sub(lambda _match: replace_text, current)


class FindReplaceDialog(QDialog):
    """Replace every occurrence of one string with another, within one column.

    Restricted to :data:`_TEXT_COLUMNS` — the numeric columns are left out on
    purpose, since a substring match against a number's text can silently
    reinterpret the value rather than just rename it.
    """

    def __init__(self, columns: list[str], initial: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find & replace")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._column_combo = QComboBox()
        self._column_combo.addItems(columns)
        if initial in columns:
            self._column_combo.setCurrentText(initial)
        form.addRow("Column:", self._column_combo)
        self._find_edit = QLineEdit()
        form.addRow("Find:", self._find_edit)
        self._replace_edit = QLineEdit()
        form.addRow("Replace with:", self._replace_edit)
        layout.addLayout(form)

        self._whole_cell = QCheckBox("Match whole cell only (safer)")
        self._whole_cell.setChecked(True)
        layout.addWidget(self._whole_cell)
        self._case_sensitive = QCheckBox("Case sensitive")
        self._case_sensitive.setChecked(True)
        layout.addWidget(self._case_sensitive)

        note = QLabel("Applies to every matching cell currently visible — an active filter narrows this.")
        note.setStyleSheet("QLabel { color: #aaa; font-size: 11px; }")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        ok_btn = QPushButton("Replace all")
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
        self._find_edit.setFocus()

    def column(self) -> str:
        return self._column_combo.currentText()

    def find_text(self) -> str:
        return self._find_edit.text()

    def replace_text(self) -> str:
        return self._replace_edit.text()

    def whole_cell(self) -> bool:
        return self._whole_cell.isChecked()

    def case_sensitive(self) -> bool:
        return self._case_sensitive.isChecked()


class _ChoiceDelegate(QStyledItemDelegate):
    """Combo editor for a column with a fixed vocabulary."""

    def __init__(self, choices: tuple[str, ...], parent=None):
        super().__init__(parent)
        self._choices = choices

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self._choices)
        return combo

    def setEditorData(self, editor: QComboBox, index):
        idx = editor.findText(index.data(Qt.DisplayRole) or "")
        editor.setCurrentIndex(max(idx, 0))

    def setModelData(self, editor: QComboBox, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)


class LabelTableDialog(QDialog):
    """The label table as a spreadsheet: filter, sort, edit cells, delete rows.

    *on_changed* is called after every change that lands, so the caller can
    redraw the plots and mark the session unsaved exactly as its own label
    handlers do.
    """

    def __init__(self, app_state, mappings: dict | None = None, on_changed=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label table")
        # Minimise/maximise sit next to the close button: a wide table is worth
        # the whole screen, and one click is what that should cost.
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.setModal(False)
        self.resize(1100, 640)
        self.app_state = app_state
        self._mappings = mappings or {}
        self._on_changed = on_changed
        # The exact frame the model was built from: an edit is applied by row
        # position, so a table replaced elsewhere must be spotted rather than
        # written through blindly.
        self._source_df: pd.DataFrame | None = None
        self._columns: list[str] = []
        self._loading = False
        self._setup_ui()
        self.reload()

    # ── construction ────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            "Every trial's labels, as they will be written to the TSV. "
            "Double-click a cell to edit it; select rows and press Delete to remove them. "
            "Right-click a column header to set or find & replace the whole column. "
            "Ctrl+Z in the main window undoes."
        )
        info.setStyleSheet("QLabel { color: #aaa; }")
        info.setWordWrap(True)
        layout.addWidget(info)

        self._model = QStandardItemModel(self)
        self._proxy = MultiColumnFilterProxy(self)
        self._proxy.setSortRole(SORT_ROLE)
        self._proxy.setSourceModel(self._model)

        self.table = QTableView()
        self.table.setModel(self._proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.setStyleSheet(_TABLE_STYLE)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.verticalHeader().setDefaultSectionSize(20)

        self._header = FilterHeaderView(set(), set())
        self._header.setSectionResizeMode(QHeaderView.Interactive)
        self._header.setStretchLastSection(True)
        self._header.filter_requested.connect(self._on_filter_requested)
        self._header.setContextMenuPolicy(Qt.CustomContextMenu)
        self._header.customContextMenuRequested.connect(self._show_header_context_menu)
        self.table.setHorizontalHeader(self._header)
        self._header.setSortIndicator(-1, Qt.AscendingOrder)
        layout.addWidget(self.table, stretch=1)

        self._status = QLabel("")
        self._status.setStyleSheet("QLabel { color: #aaa; font-size: 11px; }")
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        delete_btn = QPushButton("Delete selected rows")
        delete_btn.setToolTip("Remove every row the selection touches (Delete)")
        delete_btn.clicked.connect(self.delete_selected_rows)
        buttons.addWidget(delete_btn)

        set_btn = QPushButton("Set selected cells…")
        set_btn.setToolTip("Give every selected cell of one editable column the same value")
        set_btn.clicked.connect(self.set_selected_cells)
        buttons.addWidget(set_btn)

        replace_btn = QPushButton("Find & replace…")
        replace_btn.setToolTip("Replace every occurrence of one string with another, in one text column")
        replace_btn.clicked.connect(lambda: self.find_replace())
        buttons.addWidget(replace_btn)

        clear_btn = QPushButton("Clear filters")
        clear_btn.clicked.connect(self._clear_filters)
        buttons.addWidget(clear_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Re-read the label table (labels changed in the main window)")
        refresh_btn.clicked.connect(self.reload)
        buttons.addWidget(refresh_btn)

        buttons.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        # Keys the table owns are taken back from the main window (see
        # eventFilter) rather than bound as shortcuts of our own, which would
        # only be ambiguous with the global ones.
        self.installEventFilter(self)
        self.table.installEventFilter(self)
        self.table.setFocus()

        self._model.itemChanged.connect(self._on_item_changed)

    # ── keys ────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        """Take back the keys a table owns from the main window's shortcuts.

        The shell binds ``Ctrl+A`` (autoscale), ``Ctrl+C`` (curate the trial)
        and friends as *application* shortcuts, which fire before the focused
        widget ever sees the key — and they are only disabled while a text
        field has focus, which a table is not. Accepting the ShortcutOverride
        for the handful of keys that mean something here lets the key press
        through: select-all reaches the view, copy and delete reach us.
        """
        if event.type() not in (QEvent.ShortcutOverride, QEvent.KeyPress):
            return False
        if not _owns_key(event):
            return False
        if event.type() == QEvent.ShortcutOverride:
            event.accept()
            return True
        key = event.key()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_rows()
            return True
        if key == Qt.Key_C:
            self.copy_selection()
            return True
        return False  # Ctrl+A: the view's own select-all is what we want

    # ── model ───────────────────────────────────────────────────────────

    def changeEvent(self, event):
        """Catch up on coming back to the front: the labels may have moved on.

        An undo or a label placed in the main window rewrites the table, and
        rows are addressed here by position — so a stale view is re-read before
        the user can click in it.
        """
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self.refresh_if_stale()
        super().changeEvent(event)

    def refresh_if_stale(self) -> None:
        """Re-read the table when the frame it was built from has been replaced.

        The caller's own edits leave it current, so this costs nothing on the
        paths that land through :meth:`reload` already; a label placed, deleted
        or undone in the main window is what it exists for.
        """
        if not self._is_current():
            self.reload()

    def set_mappings(self, mappings: dict | None) -> None:
        """Point at the label classes as they stand now (mapping.txt reloaded)."""
        self._mappings = mappings or {}
        self.reload()

    def _labels_df(self) -> pd.DataFrame | None:
        return getattr(self.app_state, "_all_labels_df", None)

    def reload(self) -> None:
        """Rebuild the table from the label table as it stands now.

        A rebuild replaces every row, so the scroll offset is put back: the
        rows a user is working through are the ones they were just looking at,
        not the first screenful.
        """
        df = self._labels_df()
        self._source_df = df
        scroll = self.table.verticalScrollBar().value()
        self._loading = True
        try:
            self._model.clear()
            if df is None or df.empty:
                self._columns = []
                self._header.set_filterable(set(), set())
                self._update_status(0)
                return

            self._columns = [c for c in TSV_COLUMNS if c in df.columns]
            self._columns += [c for c in df.columns if c not in self._columns]
            self._model.setHorizontalHeaderLabels([NAME_COLUMN_TITLE, *self._columns])
            for pos in range(len(df)):
                self._model.appendRow(self._build_row(df, pos))
            self._apply_filterable_columns()
            self._install_delegates()
            self.table.resizeColumnsToContents()
            self._update_status(len(df))
        finally:
            self._loading = False
            self._scroll_to(scroll)

    def _scroll_to(self, value: int) -> None:
        bar = self.table.verticalScrollBar()
        bar.setValue(min(value, bar.maximum()))

    def _build_row(self, df: pd.DataFrame, pos: int) -> list[QStandardItem]:
        row = [self._name_item(df, pos)]
        row[0].setData(pos, POSITION_ROLE)
        for column in self._columns:
            row.append(self._cell_item(df, pos, column))
        return row

    def _name_item(self, df: pd.DataFrame, pos: int) -> QStandardItem:
        label_id = self._label_id(df, pos)
        entry = self._mappings.get(label_id, {})
        item = QStandardItem(str(entry.get("name", f"id {label_id}")))
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        color = entry.get("color")
        if color is not None:
            item.setBackground(QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 110))
        return item

    def _cell_item(self, df: pd.DataFrame, pos: int, column: str) -> QStandardItem:
        value = df.iat[pos, df.columns.get_loc(column)]
        item = QStandardItem(_format(column, value))
        _set_sort_value(item, column, value)
        if column in INTERVAL_COLUMNS:
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        else:
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setForeground(QColor(170, 170, 170))
            item.setToolTip(
                "Read-only: one value for the whole trial"
                if column in TRIAL_META_COLUMNS
                else "Read-only: a row belongs to its trial"
            )
        return item

    def _label_id(self, df: pd.DataFrame, pos: int) -> int:
        value = df.iat[pos, df.columns.get_loc("labels")]
        return int(value) if pd.notna(value) else -1

    def _install_delegates(self) -> None:
        for column, choices in CHOICE_COLUMNS.items():
            if column in self._columns:
                col = self._columns.index(column) + 1
                self.table.setItemDelegateForColumn(col, _ChoiceDelegate(choices, self.table))

    def _update_status(self, n_rows: int) -> None:
        path = self.app_state.labels_file_path() if hasattr(self.app_state, "labels_file_path") else None
        parts = [f"{n_rows} rows"]
        if path is not None:
            parts.append(str(path))
        if not getattr(self.app_state, "changes_saved", True):
            parts.append("unsaved changes")
        self._status.setText("  ·  ".join(parts))

    # ── filtering ───────────────────────────────────────────────────────

    def _apply_filterable_columns(self) -> None:
        cat_cols, num_cols = {0}, set()
        for idx, column in enumerate(self._columns, start=1):
            if column in _NUMERIC_FILTER_COLUMNS:
                num_cols.add(idx)
            else:
                cat_cols.add(idx)
        self._header.set_filterable(cat_cols, num_cols)
        self._header.set_active_filters(self._proxy.active_filters())

    def _column_values(self, col: int) -> list[str]:
        return sorted({self._model.item(row, col).text() for row in range(self._model.rowCount())})

    def _on_filter_requested(self, col: int) -> None:
        if self._header.is_numeric(col):
            dialog = NumericFilterDialog(col, self._proxy.num_filter(col), self)
            if dialog.exec_() == QDialog.Accepted:
                criterion = dialog.get_filter()
                self._proxy.set_numeric_filter(col, *(criterion or (None, None)))
        else:
            dialog = CategoryFilterDialog(col, self._column_values(col), self._proxy.cat_filter(col), self)
            if dialog.exec_() == QDialog.Accepted:
                self._proxy.set_cat_filter(col, dialog.get_allowed())
        self._header.set_active_filters(self._proxy.active_filters())

    def _clear_filters(self) -> None:
        self._proxy.clear_filters()
        self._header.set_active_filters(set())

    # ── selection ───────────────────────────────────────────────────────

    def _selected_source_indexes(self) -> list:
        return [self._proxy.mapToSource(idx) for idx in self.table.selectionModel().selectedIndexes()]

    def _position_of(self, source_row: int) -> int:
        return self._model.item(source_row, 0).data(POSITION_ROLE)

    def _selected_positions(self) -> list[int]:
        rows = {idx.row() for idx in self._selected_source_indexes()}
        return sorted(self._position_of(row) for row in rows)

    # ── editing ─────────────────────────────────────────────────────────

    def _is_current(self) -> bool:
        """Whether the table still holds the frame this view was built from."""
        return self._source_df is not None and self._labels_df() is self._source_df

    def _refuse_stale(self) -> bool:
        if self._is_current():
            return False
        notify("The labels changed elsewhere — reloading the table", severity="warning")
        self.reload()
        return True

    def _on_item_changed(self, item: QStandardItem) -> None:
        if self._loading or item.column() == 0:
            return
        column = self._columns[item.column() - 1]
        pos = self._position_of(item.row())
        self._commit(pos, column, item.text())
        self._refresh_row(item.row(), pos)

    def _commit(self, pos: int, column: str, text: str) -> None:
        """Write one cell back to the label table, recording it for undo."""
        if self._refuse_stale():
            return
        df = self._source_df
        try:
            value = parse_cell(column, text)
            self._validate(df, pos, column, value)
        except ValueError as exc:
            notify(str(exc), severity="warning")
            return

        trial = df.iat[pos, df.columns.get_loc("trial")]
        self.app_state.record_label_edit(f"edit {column}", trial)
        df.iat[pos, df.columns.get_loc(column)] = value
        self.app_state.replace_all_labels(df)
        self._changed()

    def _validate(self, df: pd.DataFrame, pos: int, column: str, value) -> None:
        """Reject a value the rest of the row contradicts."""
        if column == "labels" and self._mappings and value not in self._mappings:
            raise ValueError(f"No label class with id {value}")
        if column in ("onset_s", "offset_s"):
            onset = value if column == "onset_s" else df.iat[pos, df.columns.get_loc("onset_s")]
            offset = value if column == "offset_s" else df.iat[pos, df.columns.get_loc("offset_s")]
            # A negative duration is dropped on save, so it must not be typed in.
            if pd.notna(offset) and float(offset) < float(onset):
                raise ValueError(f"offset_s ({float(offset):g}) is before onset_s ({float(onset):g})")

    def _refresh_row(self, source_row: int, pos: int) -> None:
        """Redraw one row from the table (formatting, and the name it now has)."""
        df = self._labels_df()
        if df is None or pos >= len(df):
            return
        self._loading = True
        try:
            name = self._name_item(df, pos)
            first = self._model.item(source_row, 0)
            first.setText(name.text())
            first.setBackground(name.background())
            for idx, column in enumerate(self._columns, start=1):
                value = df.iat[pos, df.columns.get_loc(column)]
                item = self._model.item(source_row, idx)
                item.setText(_format(column, value))
                _set_sort_value(item, column, value)
        finally:
            self._loading = False

    def _prompt_value(self, column: str, title: str) -> str | None:
        """Ask for one value to write into *column* — a combo for a fixed vocabulary."""
        if column in CHOICE_COLUMNS:
            text, ok = QInputDialog.getItem(self, title, f"{column}:", list(CHOICE_COLUMNS[column]), 0, False)
        else:
            text, ok = QInputDialog.getText(self, title, f"{column}:")
        return text if ok else None

    def _apply_to_rows(self, source_rows: list[int], column: str, text: str) -> int:
        """Commit *text* to *column* for each of *source_rows*; returns how many landed."""
        n = 0
        for row in sorted(source_rows):
            pos = self._position_of(row)
            self._commit(pos, column, text)
            self._refresh_row(row, pos)
            n += 1
        return n

    def set_selected_cells(self) -> None:
        """Give every selected cell of one editable column the same value."""
        indexes = [idx for idx in self._selected_source_indexes() if idx.column() > 0]
        columns = {idx.column() for idx in indexes}
        if len(columns) != 1:
            notify("Select cells in exactly one column first", severity="warning")
            return
        column = self._columns[columns.pop() - 1]
        if column not in INTERVAL_COLUMNS:
            notify(f"{column} is read-only", severity="warning")
            return

        text = self._prompt_value(column, "Set cells")
        if text is None:
            return
        self._apply_to_rows({idx.row() for idx in indexes}, column, text)

    def _set_entire_column(self, col_logical: int) -> None:
        """Give every row currently visible in the table the same value in one column."""
        column = self._columns[col_logical - 1]
        if column not in INTERVAL_COLUMNS:
            notify(f"{column} is read-only", severity="warning")
            return
        text = self._prompt_value(column, f"Set '{column}' column")
        if text is None:
            return
        rows = [self._proxy.mapToSource(self._proxy.index(r, col_logical)).row() for r in range(self._proxy.rowCount())]
        if not rows:
            notify("No visible rows to update", severity="warning")
            return
        n = self._apply_to_rows(rows, column, text)
        notify(f"Set {n} cell{'s' if n != 1 else ''} in {column}")

    def find_replace(self, column: str | None = None) -> None:
        """Replace every occurrence of one string with another, in one text column."""
        if not _TEXT_COLUMNS:
            return
        dialog = FindReplaceDialog(list(_TEXT_COLUMNS), column, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        find_text = dialog.find_text()
        if not find_text:
            notify("Enter text to find first", severity="warning")
            return
        col_name = dialog.column()
        replace_text = dialog.replace_text()
        if col_name in CHOICE_COLUMNS and replace_text not in CHOICE_COLUMNS[col_name]:
            notify(f"{col_name} must be one of: {', '.join(CHOICE_COLUMNS[col_name])}", severity="warning")
            return
        whole_cell, case_sensitive = dialog.whole_cell(), dialog.case_sensitive()

        col_logical = self._columns.index(col_name) + 1
        n_matches = 0
        for view_row in range(self._proxy.rowCount()):
            source_row = self._proxy.mapToSource(self._proxy.index(view_row, col_logical)).row()
            current = self._model.item(source_row, col_logical).text()
            new_text = _replace_value(current, find_text, replace_text, whole_cell, case_sensitive)
            if new_text is None:
                continue
            pos = self._position_of(source_row)
            self._commit(pos, col_name, new_text)
            self._refresh_row(source_row, pos)
            n_matches += 1

        if n_matches:
            notify(f"Replaced {n_matches} cell{'s' if n_matches != 1 else ''} in {col_name}")
        else:
            notify("No matching cells found", severity="warning")

    def _show_header_context_menu(self, point) -> None:
        col_logical = self._header.logicalIndexAt(point)
        if col_logical <= 0 or col_logical - 1 >= len(self._columns):
            return
        column = self._columns[col_logical - 1]
        menu = QMenu(self)
        if column in INTERVAL_COLUMNS:
            menu.addAction(f"Set entire '{column}' column…", lambda: self._set_entire_column(col_logical))
            if column in _TEXT_COLUMNS:
                menu.addAction(f"Find & replace in '{column}'…", lambda: self.find_replace(column))
        else:
            action = menu.addAction(f"'{column}' is read-only")
            action.setEnabled(False)
        menu.exec_(self._header.mapToGlobal(point))

    def delete_selected_rows(self) -> None:
        """Remove every row the selection touches."""
        if self._refuse_stale():
            return
        positions = self._selected_positions()
        if not positions:
            notify("Select the rows to delete first", severity="warning")
            return
        if len(positions) > 1:
            answer = QMessageBox.question(
                self,
                "Delete labels",
                f"Delete {len(positions)} label rows?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        # Where the deletion happened on screen, so the next row can take the
        # cursor — deleting a run of labels should not send anyone back to the
        # top of the table between rows.
        view_row = min(idx.row() for idx in self.table.selectionModel().selectedIndexes())

        df = self._source_df
        # One undo step per trial: an undo snapshot holds a single trial's rows.
        for trial in pd.unique(df.iloc[positions]["trial"]):
            self.app_state.record_label_edit("delete labels (table)", trial)
        remaining = df.drop(df.index[positions]).reset_index(drop=True)
        self.app_state.replace_all_labels(remaining)
        self._changed()
        self.reload()
        self._select_view_row(view_row)
        notify(f"Deleted {len(positions)} label rows")

    def _select_view_row(self, view_row: int) -> None:
        """Put the cursor on *view_row*, or the last row when it was the end."""
        rows = self._proxy.rowCount()
        if not rows:
            return
        index = self._proxy.index(min(view_row, rows - 1), 0)
        self.table.setCurrentIndex(index)
        self.table.selectionModel().select(
            index,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )

    def copy_selection(self) -> None:
        """Put the selected cells on the clipboard as TSV."""
        indexes = self._selected_source_indexes()
        if not indexes:
            return
        cells: dict[int, dict[int, str]] = {}
        for idx in indexes:
            cells.setdefault(idx.row(), {})[idx.column()] = self._model.item(idx.row(), idx.column()).text()
        lines = ["\t".join(row[col] for col in sorted(row)) for _, row in sorted(cells.items())]
        QApplication.clipboard().setText("\n".join(lines))

    def _changed(self) -> None:
        self._source_df = self._labels_df()
        if self._on_changed is not None:
            self._on_changed()
        self._update_status(len(self._source_df) if self._source_df is not None else 0)

    # ── context menu ────────────────────────────────────────────────────

    def _show_context_menu(self, point) -> None:
        menu = QMenu(self)
        n_rows = len(self._selected_positions())
        delete = menu.addAction(f"Delete {n_rows} selected rows" if n_rows != 1 else "Delete selected row")
        delete.setEnabled(n_rows > 0)
        delete.triggered.connect(self.delete_selected_rows)
        menu.addAction("Set selected cells…", self.set_selected_cells)
        menu.addAction("Find & replace…", lambda: self.find_replace())
        menu.addAction("Copy", self.copy_selection)
        menu.addSeparator()
        menu.addAction("Refresh", self.reload)
        menu.exec_(self.table.viewport().mapToGlobal(point))


def _format(column: str, value) -> str:
    if value is None or pd.isna(value):
        return ""
    decimals = _DECIMALS.get(column)
    if decimals is not None:
        return f"{float(value):.{decimals}f}"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    return str(value)


def _set_sort_value(item: QStandardItem, column: str, value) -> None:
    """Give *item* a comparable value where its column holds numbers.

    The shared proxy compares :data:`SORT_ROLE` numerically and falls back to
    the displayed text when it is absent, so only numeric columns carry it. A
    missing number (a point event's ``offset_s``) sorts first.
    """
    if not _is_numeric_column(column):
        return
    item.setData(float("-inf") if value is None or pd.isna(value) else float(value), SORT_ROLE)


def _is_numeric_column(column: str) -> bool:
    dtype = INTERVAL_DTYPES.get(column, object)
    return column in _NUMERIC_FILTER_COLUMNS or (dtype is not object and np.issubdtype(dtype, np.number))
