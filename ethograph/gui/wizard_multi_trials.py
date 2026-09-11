"""Trial table page for the NC creation wizard (Page 3)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.notify import notify_dialog
from ethograph.gui.wizard_state import WizardState

logger = logging.getLogger(__name__)


class TrialsPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        layout.addWidget(QLabel("<b>Step 3 — Trial table</b>"))
        layout.addSpacing(6)

        # Explanatory text (changes with mode)
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #a0a0a0; padding: 2px 0; font-size: 11px;")
        layout.addWidget(self._info_label)
        layout.addSpacing(4)

        # ═══ Split view: Auto-generated | Import metadata ═══
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # LEFT: Auto-generated table
        auto_group = QGroupBox("① Auto-generated from files")
        auto_layout = QVBoxLayout(auto_group)
        auto_layout.setSpacing(4)

        self._auto_status = QLabel("No multi-file modalities detected")
        self._auto_status.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        auto_layout.addWidget(self._auto_status)

        self._auto_table = QTableWidget()
        self._auto_table.setAlternatingRowColors(True)
        self._auto_table.setStyleSheet(
            "QTableWidget { background: #1a1d21; alternate-background-color: #212428;"
            " color: #a0a0a0; gridline-color: #2a2f37; border: 1px solid #2a2f37; }"
            "QHeaderView::section { background: #252a32; color: #50c8b4;"
            " border: 1px solid #2a2f37; padding: 3px; font-weight: bold; }"
            "QTableWidget::item { padding: 2px 4px; }"
        )
        self._auto_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._auto_table.setMaximumHeight(200)
        auto_layout.addWidget(self._auto_table)
        splitter.addWidget(auto_group)

        # RIGHT: Import/paste metadata
        import_group = QGroupBox("② Add trial metadata (optional)")
        import_layout = QVBoxLayout(import_group)
        import_layout.setSpacing(6)

        hint = QLabel(
            "Import or paste a table with a 'trial' column.\n"
            "Additional columns will be <b>merged</b> with the auto-generated table."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #a0a0a0; font-size: 10px; padding: 2px;")
        import_layout.addWidget(hint)

        btn_row = QHBoxLayout()
        self._import_btn = QPushButton("📁 Import CSV/TSV")
        self._import_btn.clicked.connect(self._import_table)
        self._paste_btn = QPushButton("📋 Paste from clipboard")
        self._paste_btn.clicked.connect(self._paste_from_clipboard)
        btn_row.addWidget(self._import_btn)
        btn_row.addWidget(self._paste_btn)
        btn_row.addStretch()
        import_layout.addLayout(btn_row)

        self._import_status = QLabel("No metadata imported")
        self._import_status.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        import_layout.addWidget(self._import_status)

        # Column requirements display
        self._requirements_label = QLabel()
        self._requirements_label.setWordWrap(True)
        self._requirements_label.setStyleSheet(
            "font-size: 10px; padding: 4px; background: #1a1d21; border: 1px solid #2a2f37; border-radius: 3px;"
        )
        self._requirements_label.hide()  # Hidden until requirements exist
        import_layout.addWidget(self._requirements_label)

        self._import_table_widget = QTableWidget()
        self._import_table_widget.setAlternatingRowColors(True)
        self._import_table_widget.setStyleSheet(
            "QTableWidget { background: #1a1d21; alternate-background-color: #212428;"
            " color: #a0a0a0; gridline-color: #2a2f37; border: 1px solid #2a2f37; }"
            "QHeaderView::section { background: #252a32; color: #e8737a;"
            " border: 1px solid #2a2f37; padding: 3px; font-weight: bold; }"
            "QTableWidget::item { padding: 2px 4px; }"
        )
        self._import_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._import_table_widget.setMaximumHeight(200)
        import_layout.addWidget(self._import_table_widget)

        clear_row = QHBoxLayout()
        self._clear_btn = QPushButton("✕ Clear imported data")
        self._clear_btn.clicked.connect(self._clear_imported)
        self._clear_btn.setEnabled(False)
        clear_row.addWidget(self._clear_btn)
        clear_row.addStretch()
        import_layout.addLayout(clear_row)

        splitter.addWidget(import_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        layout.addSpacing(8)

        # ═══ Final merged result ═══
        result_label = QLabel("③ <b>Final table</b> (merged result)")
        result_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(result_label)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background: #1e2126; alternate-background-color: #262a30;"
            " color: #d0d0d0; gridline-color: #3a3f47; }"
            "QHeaderView::section { background: #2a2e35; color: #d0d0d0;"
            " border: 1px solid #3a3f47; padding: 4px; }"
            "QTableWidget::item { padding: 3px 6px; }"
            "QTableWidget::item:selected { background: #3a5f8a; color: #ffffff; }"
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setMinimumHeight(200)
        layout.addWidget(self._table, stretch=1)

        # Individuals
        ind_row = QHBoxLayout()
        ind_row.addWidget(QLabel("Individuals (optional):"))
        self._individuals_edit = QLineEdit()
        self._individuals_edit.setPlaceholderText("e.g., bird1, bird2 (comma-separated, leave empty for default)")
        ind_row.addWidget(self._individuals_edit)
        layout.addLayout(ind_row)

        scroll.setWidget(inner)
        outer_layout.addWidget(scroll)

        self._auto_df: pd.DataFrame | None = None
        self._imported_df: pd.DataFrame | None = None
        self._imported_path: str | None = None  # Store import path
        self._wizard_state: WizardState | None = None

    def populate_from_table(self, state: WizardState, table: pd.DataFrame) -> None:
        """Show the pairing table :func:`~ethograph.io.pairing.discover_media` built."""
        self._wizard_state = state
        self._auto_df = table
        self._update_auto_table()
        self._update_requirements_display()
        self._refresh_table()

    def _update_auto_table(self):
        """Update the auto-generated table preview."""
        if self._auto_df is None or self._auto_df.empty:
            self._auto_table.setRowCount(0)
            self._auto_table.setColumnCount(0)
            self._auto_status.setText("No multi-file modalities detected")
            self._auto_status.setStyleSheet("color: #888; font-style: italic;")
            return

        cols = list(self._auto_df.columns)
        n_rows = min(len(self._auto_df), 50)
        self._auto_table.setColumnCount(len(cols))
        self._auto_table.setHorizontalHeaderLabels(cols)
        self._auto_table.setRowCount(n_rows)

        for r in range(n_rows):
            for c, col in enumerate(cols):
                val = self._auto_df.iloc[r][col]
                if pd.notna(val):
                    txt = Path(str(val)).name if "/" in str(val) or "\\" in str(val) else str(val)
                else:
                    txt = ""
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._auto_table.setItem(r, c, item)

        self._auto_status.setText(f"✓ {len(self._auto_df)} trials detected")
        self._auto_status.setStyleSheet("color: #50c8b4; font-weight: bold;")
        self._update_requirements_display()

    def _update_import_table(self):
        """Update the imported metadata preview."""
        if self._imported_df is None or self._imported_df.empty:
            self._import_table_widget.setRowCount(0)
            self._import_table_widget.setColumnCount(0)
            self._import_status.setText("No metadata imported")
            self._import_status.setStyleSheet("color: #888; font-style: italic;")
            self._clear_btn.setEnabled(False)
            self._update_requirements_display()
            return

        cols = list(self._imported_df.columns)
        n_rows = min(len(self._imported_df), 50)
        self._import_table_widget.setColumnCount(len(cols))
        self._import_table_widget.setHorizontalHeaderLabels(cols)
        self._import_table_widget.setRowCount(n_rows)

        for r in range(n_rows):
            for c, col in enumerate(cols):
                val = self._imported_df.iloc[r][col]
                txt = str(val) if pd.notna(val) else ""
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._import_table_widget.setItem(r, c, item)

        extra_cols = [c for c in cols if c != "trial"]
        self._import_status.setText(f"✓ {len(extra_cols)} metadata column(s): {', '.join(extra_cols)}")
        self._import_status.setStyleSheet("color: #e8737a; font-weight: bold;")
        self._clear_btn.setEnabled(True)
        self._update_requirements_display()

    def _align_trial_types(self, df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Ensure both DataFrames have the same type for 'trial' column to avoid merge errors.

        Smart type conversion:
        - If both columns contain pure numbers (e.g., "20", "30"), convert to int
        - If either contains letters (e.g., "trial_1", "A1"), convert both to strings
        """
        if "trial" not in df1.columns or "trial" not in df2.columns:
            return df1, df2

        def can_convert_to_int(series: pd.Series) -> bool:
            """Check if all values in series can be safely converted to int."""
            try:
                pd.to_numeric(series, errors="raise").astype(int)
                return True
            except (ValueError, TypeError):
                return False

        # Check if both columns can be converted to integers
        can_int_1 = can_convert_to_int(df1["trial"])
        can_int_2 = can_convert_to_int(df2["trial"])

        if can_int_1 and can_int_2:
            # Both are pure numbers - convert to int
            df1 = df1.copy()
            df2 = df2.copy()
            df1["trial"] = pd.to_numeric(df1["trial"], errors="coerce").astype(int)
            df2["trial"] = pd.to_numeric(df2["trial"], errors="coerce").astype(int)
        else:
            # At least one contains non-numeric values - convert both to string
            df1 = df1.copy()
            df2 = df2.copy()
            df1["trial"] = df1["trial"].astype(str)
            df2["trial"] = df2["trial"].astype(str)

        return df1, df2

    def _is_fully_aligned(self) -> bool:
        """Check if all enabled modalities are in aligned mode (file_mode == 'aligned_to_trial')."""
        if not self._wizard_state:
            return True
        return self._wizard_state.is_fully_aligned()

    def _refresh_table(self):
        if self._auto_df is not None:
            df = self._auto_df.copy()
            if self._imported_df is not None:
                # Merge imported metadata
                if "trial" in self._imported_df.columns:
                    extra_cols = [c for c in self._imported_df.columns if c != "trial" and c not in df.columns]
                    if extra_cols:
                        # Ensure consistent trial column types before merging
                        imported_subset = self._imported_df[["trial"] + extra_cols].copy()
                        df, imported_subset = self._align_trial_types(df, imported_subset)
                        df = df.merge(imported_subset, on="trial", how="left")
            self._display_df(df)
        elif self._imported_df is not None:
            self._display_df(self._imported_df)
        else:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)

    def _display_df(self, df: pd.DataFrame):
        cols = list(df.columns)
        n_rows = min(len(df), 200)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(n_rows)
        for r in range(n_rows):
            for c, col in enumerate(cols):
                val = df.iloc[r][col]
                if pd.notna(val):
                    txt = Path(str(val)).name if "/" in str(val) or "\\" in str(val) else str(val)
                else:
                    txt = ""
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(r, c, item)

    def _on_wizard_state_changed(self):
        """Called when wizard state changes (e.g., modality mode selection changed)."""
        self._update_info_text()
        self._update_auto_table()
        self._update_import_table()
        self._update_requirements_display()
        self._refresh_table()

    def _update_info_text(self):
        is_aligned = self._is_fully_aligned()
        if is_aligned:
            self._info_label.setText(
                "📌 <b>Aligned mode:</b> Each file corresponds to one trial. "
                "The trial table is built automatically from file patterns. "
                "You can add trial metadata (e.g., conditions, genotype) by importing or pasting a CSV/TSV — "
                "any extra columns will be merged with the auto-generated table."
                "Note: 'start_time' table in trials will be ignored."
            )
        else:
            self._info_label.setText(
                "📌 <b>Non-aligned mode:</b> Some files span multiple trials. "
                "Import or paste a table with 'trial' and 'start_time' columns (seconds). "
                "'stop_time' or trial metadata (e.g., conditions, genotype) is optional."
            )

    def _import_table(self):
        result = QFileDialog.getOpenFileName(
            self,
            "Import trial table",
            "",
            "Table files (*.csv *.tsv *.txt);;All files (*)",
        )
        if not (result and result[0]):
            return
        path = result[0]
        try:
            df = pd.read_csv(path, sep=None, engine="python")  # Let pandas sniff the separator
            self._imported_df = df
            self._imported_path = path
            self._update_import_table()
            self._refresh_table()
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            logger.exception("Failed to read file")
            notify_dialog(f"Failed to read file:\n{e}", "error", "Import error", self)

    def _paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text.strip():
            return
        first_line = text.strip().split("\n")[0]
        sep = "\t" if "\t" in first_line else ","
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            self._imported_df = df
            self._imported_path = None
            self._update_import_table()
            self._refresh_table()
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            logger.exception("Failed to parse clipboard")
            notify_dialog(f"Failed to parse clipboard:\n{e}", "error", "Paste error", self)

    def _clear_imported(self):
        self._imported_df = None
        self._imported_path = None
        self._update_import_table()
        self._refresh_table()

    def _get_required_columns(self) -> dict[str, bool]:
        """Compute required columns based on wizard state.

        Returns dict: {column_name: is_required}
        - is_required=True: must be present
        - is_required=False: optional but useful
        """
        if self._wizard_state is None:
            return {}

        required = {}

        # Non-aligned mode requires timing columns
        if not self._is_fully_aligned():
            required["start_time"] = True
            required["stop_time"] = False  # Optional

        return required

    def _update_requirements_display(self):
        """Update the column requirements display with color coding."""
        required_cols = self._get_required_columns()

        if not required_cols:
            self._requirements_label.hide()
            return

        self._requirements_label.show()

        # Check which columns are present in imported data
        present_cols = set()
        if self._imported_df is not None and not self._imported_df.empty:
            present_cols = set(self._imported_df.columns)

        # Build status text
        lines = ["<b>Import table columns:</b>"]
        missing_required = []
        has_required = any(is_req for is_req in required_cols.values())

        for col, is_required in sorted(required_cols.items()):
            is_present = col in present_cols
            if is_present:
                color = "#50c8b4"  # green
                icon = "✓"
            else:
                if is_required:
                    color = "#e8737a"  # red
                    icon = "✗"
                    missing_required.append(col)
                else:
                    color = "#888"  # gray
                    icon = "○"

            req_label = "<b>[required]</b>" if is_required else "[optional]"
            lines.append(f"  <span style='color: {color};'>{icon} <code>{col}</code></span> {req_label}")

        if missing_required:
            lines.append(
                f"<br><span style='color: #e8737a;'><b>⚠ Missing required:</b> {', '.join(missing_required)}</span>"
            )
        elif has_required:
            lines.append("<br><span style='color: #50c8b4;'><b>✓ All required columns present</b></span>")

        self._requirements_label.setText("<br>".join(lines))

    def collect_state(self, state: WizardState):
        state.files_aligned_to_trials = self._is_fully_aligned()

        # Build the final trial table from what's displayed
        df = self._get_current_df()
        state.trial_table = df
        state.trial_table_path = self._imported_path  # Store import path

        # Parse individuals
        ind_text = self._individuals_edit.text().strip()
        if ind_text:
            state.individuals = [s.strip() for s in ind_text.split(",")]
        else:
            state.individuals = []

    def _get_current_df(self) -> pd.DataFrame | None:
        if self._auto_df is not None:
            df = self._auto_df.copy()
            if self._imported_df is not None and "trial" in self._imported_df.columns:
                extra_cols = [c for c in self._imported_df.columns if c != "trial" and c not in df.columns]
                if extra_cols:
                    imported_subset = self._imported_df[["trial"] + extra_cols].copy()
                    df, imported_subset = self._align_trial_types(df, imported_subset)
                    df = df.merge(imported_subset, on="trial", how="left")
            return df
        if self._imported_df is not None:
            return self._imported_df
        return self._auto_df

    def validate(self, state: WizardState) -> str | None:
        df = self._get_current_df()
        is_aligned = self._is_fully_aligned()

        # Non-aligned mode requires timing columns
        if not is_aligned:
            if df is None or df.empty:
                return "Non-aligned mode requires a trial table (import or paste)."
            if "start_time" not in df.columns:
                return "Non-aligned mode requires a 'start_time' column (seconds)."
        # Note: File duration equality check for aligned mode is handled in the temporal alignment page
        return None
