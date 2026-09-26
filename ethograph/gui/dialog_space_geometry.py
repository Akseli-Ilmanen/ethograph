"""Settings ▸ Edit space geometry: place parametric shapes in a geometry file and watch them move.

One geometry file of the space library (:func:`~ethograph.gui.plots_space.geometry_dirs`)
is edited at a time, in a narrow window meant to sit beside the plots. The dialog
draws nothing itself: every change goes, through ``app_state.space_geometry_preview``,
to the open space and skeleton plots — the pose is what a shape is placed against —
with the selected shape highlighted. Nothing is written until Save; closing the
dialog drops the preview and the plots go back to the file.

The file's raw ``references`` (vertex/edge wireframes) are drawn too and saved back
unchanged; only its ``shapes`` are edited here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.notify import notify
from ethograph.gui.plots_space import (
    ReferenceGeometry,
    geometry_config,
    geometry_dirs_of,
    library_geometry_files,
    load_geometry_yaml,
    parse_geometry,
    write_geometry_yaml,
)
from ethograph.gui.project import project_dir_of
from ethograph.gui.space_shapes import (
    ANGLE_PARAMS,
    AXES,
    SHAPE_PARAMS,
    SHAPE_TYPES,
    Shape,
    axes_half_length,
    parse_shapes,
)

#: The shape selected in the list, as the plots draw it while it is edited.
_SELECTED_COLOR = "#FF6600"
#: Spin-box steps offered for nudging a shape; the data's units decide which is sensible.
_STEPS = ("0.001", "0.01", "0.1", "1", "10", "100")
_LIMIT = 1e6


def _save_targets(app_state) -> list[tuple[str, Path]]:
    """The library folders a geometry can be saved to, nearest first, each with a name a user knows."""
    dirs = geometry_dirs_of(app_state)
    project = project_dir_of(app_state)
    project_space = dirs[-2] if project is not None else None
    targets = []
    for folder in dirs:
        if folder == dirs[-1]:
            label = "Your defaults"
        elif folder == project_space:
            label = "Project"
        else:
            label = "This session"
        targets.append((label, folder))
    return targets


class SpaceGeometryDialog(QDialog):
    """Add, place and size the shapes of one space geometry file, then save it."""

    def __init__(
        self,
        app_state,
        on_saved: Callable[[], None] | None = None,
        panels_open: Callable[[], bool] | None = None,
        open_panel: Callable[[], None] | None = None,
        data_center: Callable[[], tuple[float, float, float] | None] | None = None,
        parent=None,
    ):
        """*panels_open* says whether a space or skeleton plot shows the edit, *open_panel*
        opens one, and *data_center* is where the animal is now — where a new shape goes."""
        super().__init__(parent)
        self.setWindowTitle("Edit space geometry")
        self.app_state = app_state
        self._on_saved = on_saved
        self._panels_open = panels_open
        self._open_panel = open_panel
        self._data_center = data_center
        self._name: str | None = None
        self._base: dict = {}
        self._shapes: list[Shape] = []
        self._dirty = False
        self._loading = False
        self._path: Path | None = None
        self._references: list[ReferenceGeometry] = []
        self._param_spins: dict[str, QDoubleSpinBox] = {}

        root = QVBoxLayout(self)

        # Shown while no plot draws the edit: without one, nothing is seen.
        self._no_panel = QWidget()
        no_panel_layout = QVBoxLayout(self._no_panel)
        no_panel_layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Shapes are drawn in the space and skeleton plots. Open one to place them against the pose.")
        hint.setWordWrap(True)
        no_panel_layout.addWidget(hint)
        open_btn = QPushButton("Open a 3D plot")
        open_btn.clicked.connect(self._on_open_panel)
        open_btn.setVisible(open_panel is not None)
        no_panel_layout.addWidget(open_btn)
        root.addWidget(self._no_panel)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        geometry_row = QHBoxLayout()
        self._geometry_combo = QComboBox()
        self._geometry_combo.setToolTip("One file per geometry in the space library")
        self._geometry_combo.currentTextChanged.connect(self._on_geometry_picked)
        geometry_row.addWidget(self._geometry_combo, 1)
        new_btn = QPushButton("New…")
        new_btn.clicked.connect(self._new_geometry)
        geometry_row.addWidget(new_btn)
        form.addRow("Geometry", geometry_row)
        self._target_combo = QComboBox()
        for label, folder in _save_targets(app_state):
            self._target_combo.addItem(label, str(folder))
            self._target_combo.setItemData(self._target_combo.count() - 1, str(folder), Qt.ItemDataRole.ToolTipRole)
        form.addRow("Save to", self._target_combo)
        root.addLayout(form)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        root.addWidget(self._build_editor(), 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close, parent=self
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.finished.connect(self._drop_preview)
        # The geometry is what the user came to see.
        self.app_state.space_show_references = True
        self._reload_library(select=getattr(app_state, "space_library_geometry", None))
        self._sync_panel_hint()
        self.resize(340, 640)

    # -- layout ------------------------------------------------------------

    def _build_editor(self) -> QWidget:
        editor = QWidget()
        layout = QVBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)

        self._shape_list = QListWidget()
        self._shape_list.setMinimumHeight(80)
        self._shape_list.currentRowChanged.connect(self._on_shape_selected)
        layout.addWidget(self._shape_list, 1)

        add_row = QHBoxLayout()
        self._add_type = QComboBox()
        self._add_type.addItems(list(SHAPE_TYPES))
        add_row.addWidget(self._add_type, 1)
        add_btn = QPushButton("Add")
        add_btn.setToolTip("Add a shape of this type where the animal is now")
        add_btn.clicked.connect(self._add_shape)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)
        edit_row = QHBoxLayout()
        self._duplicate_btn = QPushButton("Duplicate")
        self._duplicate_btn.clicked.connect(self._duplicate_shape)
        edit_row.addWidget(self._duplicate_btn)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_shape)
        edit_row.addWidget(self._remove_btn)
        layout.addLayout(edit_row)

        self._form_widget = QWidget()
        form = QFormLayout(self._form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        self._name_edit = QLineEdit()
        self._name_edit.textEdited.connect(self._on_field_changed)
        form.addRow("Name", self._name_edit)
        self._type_combo = QComboBox()
        self._type_combo.addItems(list(SHAPE_TYPES))
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        form.addRow("Type", self._type_combo)
        self._color_btn = QPushButton()
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Colour", self._color_btn)
        self._center_spins = [self._spin() for _ in range(3)]
        for axis, spin in zip("XYZ", self._center_spins):
            form.addRow(f"Centre {axis}", spin)
        self._axis_combo = QComboBox()
        self._axis_combo.addItems(list(AXES))
        self._axis_combo.setToolTip("The axis the cylinder's height runs along")
        self._axis_combo.currentTextChanged.connect(self._on_field_changed)
        form.addRow("Axis", self._axis_combo)
        self._params_form = QFormLayout()
        self._params_form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self._params_form)
        self._step_combo = QComboBox()
        self._step_combo.addItems(list(_STEPS))
        self._step_combo.setCurrentText("0.1")
        self._step_combo.setToolTip("How far one arrow press or wheel notch moves or grows a shape")
        self._step_combo.currentTextChanged.connect(self._apply_step)
        form.addRow("Step", self._step_combo)
        layout.addWidget(self._form_widget)
        return editor

    def _spin(self, angle: bool = False, minimum: float = -_LIMIT) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-360.0 if angle else minimum, 360.0 if angle else _LIMIT)
        spin.setDecimals(1 if angle else 4)
        spin.setKeyboardTracking(False)
        if angle:
            spin.setSuffix(" °")
            spin.setSingleStep(5.0)
        spin.valueChanged.connect(self._on_field_changed)
        return spin

    def _apply_step(self, *_args) -> None:
        step = float(self._step_combo.currentText())
        for spin in [*self._center_spins, *self._param_spins.values()]:
            if spin.suffix() == "":
                spin.setSingleStep(step)

    # -- the library -------------------------------------------------------

    def _reload_library(self, select: str | None = None, row: int = 0) -> None:
        self._files = library_geometry_files(geometry_dirs_of(self.app_state))
        names = sorted(set(self._files) | ({self._name} if self._name and self._dirty else set()))
        self._geometry_combo.blockSignals(True)
        self._geometry_combo.clear()
        self._geometry_combo.addItems(names)
        index = self._geometry_combo.findText(select or "")
        self._geometry_combo.setCurrentIndex(index if index >= 0 else 0)
        self._geometry_combo.blockSignals(False)
        self._load(self._geometry_combo.currentText() or None, row=row)

    def _load(self, name: str | None, row: int = 0) -> None:
        """Show geometry *name* as its file holds it (a new, unsaved name starts empty)."""
        path = self._files.get(name or "")
        base = (load_geometry_yaml(path) or {}) if path is not None else {}
        try:
            shapes = parse_shapes(base)
            references = parse_geometry({"references": base.get("references")})
        except (KeyError, TypeError, ValueError) as exc:
            notify(f"Could not read {path}: {exc}", "warning")
            base, shapes, references = {}, [], []
        self._name, self._base, self._shapes, self._dirty = name, base, shapes, False
        self._path, self._references = path, references
        if path is not None:
            index = self._target_combo.findData(str(path.parent))
            if index >= 0:
                self._target_combo.setCurrentIndex(index)
        self._refresh_list(select=row)
        self._update_summary()

    def _on_geometry_picked(self, name: str) -> None:
        if name == self._name or not self._confirm_discard():
            self._geometry_combo.blockSignals(True)
            self._geometry_combo.setCurrentText(self._name or "")
            self._geometry_combo.blockSignals(False)
            return
        self._load(name or None)

    def _new_geometry(self) -> None:
        if not self._confirm_discard():
            return
        self._dirty = False
        name, ok = QInputDialog.getText(self, "New geometry", "Name (one file):")
        name = name.strip()
        if not ok or not name:
            return
        if name in self._files:
            self._geometry_combo.setCurrentText(name)
            return
        self._load(name)
        self._dirty = True
        project_index = self._target_combo.findText("Project")
        self._target_combo.setCurrentIndex(project_index if project_index >= 0 else 0)
        self._geometry_combo.blockSignals(True)
        self._geometry_combo.addItem(name)
        self._geometry_combo.setCurrentText(name)
        self._geometry_combo.blockSignals(False)
        self._update_summary()

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(self, "Unsaved shapes", f"Discard the unsaved changes to {self._name}?")
        return answer == QMessageBox.StandardButton.Yes

    # -- the shape list ----------------------------------------------------

    def _refresh_list(self, select: int | None = None) -> None:
        row = self._shape_list.currentRow() if select is None else select
        self._shape_list.blockSignals(True)
        self._shape_list.clear()
        for shape in self._shapes:
            self._shape_list.addItem(f"{shape.name}  ({shape.type})")
        self._shape_list.blockSignals(False)
        row = min(row, len(self._shapes) - 1)
        self._shape_list.setCurrentRow(row)
        self._on_shape_selected(row)

    def _selected(self) -> Shape | None:
        row = self._shape_list.currentRow()
        return self._shapes[row] if 0 <= row < len(self._shapes) else None

    def _new_center(self) -> tuple[float, float, float]:
        """Where a new shape goes: at the animal, else amid the geometry, else the origin."""
        center = self._data_center() if self._data_center is not None else None
        if center is not None:
            return center
        refs = parse_geometry(self._config())
        if not refs:
            return (0.0, 0.0, 0.0)
        vertices = np.vstack([np.pad(r.vertices, ((0, 0), (0, 3 - r.vertices.shape[1]))) for r in refs])
        mid = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        return (float(mid[0]), float(mid[1]), float(mid[2]))

    def _add_shape(self) -> None:
        if self._name is None:
            self._new_geometry()
            if self._name is None:
                return
        shape_type = self._add_type.currentText()
        taken = {s.name for s in self._shapes}
        name = next(f"{shape_type}{i}" for i in range(1, len(taken) + 2) if f"{shape_type}{i}" not in taken)
        self._shapes.append(Shape(name, shape_type, center=self._new_center()))
        self._changed(select=len(self._shapes) - 1)

    def _duplicate_shape(self) -> None:
        shape = self._selected()
        if shape is None:
            return
        self._shapes.append(shape.retyped(shape.type))
        self._shapes[-1].name = f"{shape.name} copy"
        self._changed(select=len(self._shapes) - 1)

    def _remove_shape(self) -> None:
        row = self._shape_list.currentRow()
        if 0 <= row < len(self._shapes):
            del self._shapes[row]
            self._changed(select=max(row - 1, 0))

    # -- the form ----------------------------------------------------------

    def _on_shape_selected(self, _row: int) -> None:
        shape = self._selected()
        self._form_widget.setEnabled(shape is not None)
        self._duplicate_btn.setEnabled(shape is not None)
        self._remove_btn.setEnabled(shape is not None)
        if shape is not None:
            self._show_shape(shape)
        self._publish()

    def _show_shape(self, shape: Shape) -> None:
        """Fill the form from *shape* without writing back into it."""
        self._loading = True
        self._name_edit.setText(shape.name)
        self._type_combo.setCurrentText(shape.type)
        self._set_color_swatch(shape.color)
        for spin, value in zip(self._center_spins, shape.center):
            spin.setValue(value)
        self._axis_combo.setCurrentText(shape.axis)
        self._axis_combo.setEnabled(shape.type == "cylinder")
        self._rebuild_param_spins(shape)
        self._loading = False

    def _rebuild_param_spins(self, shape: Shape) -> None:
        while self._params_form.rowCount():
            self._params_form.removeRow(0)
        self._param_spins = {}
        for param in SHAPE_PARAMS[shape.type]:
            angle = param in ANGLE_PARAMS
            spin = self._spin(angle=angle, minimum=0.0)
            spin.blockSignals(True)
            spin.setValue(shape.params[param])
            spin.blockSignals(False)
            self._params_form.addRow(param.capitalize(), spin)
            self._param_spins[param] = spin
        self._apply_step()

    def _set_color_swatch(self, color: str) -> None:
        self._color_btn.setText(color)
        self._color_btn.setStyleSheet(f"QPushButton {{ border-left: 16px solid {QColor(color).name()}; }}")

    def _pick_color(self) -> None:
        shape = self._selected()
        if shape is None:
            return
        color = QColorDialog.getColor(QColor(shape.color), self, "Shape colour")
        if color.isValid():
            shape.color = color.name().upper()
            self._set_color_swatch(shape.color)
            self._changed()

    def _on_type_changed(self, shape_type: str) -> None:
        row = self._shape_list.currentRow()
        if self._loading or not 0 <= row < len(self._shapes):
            return
        self._shapes[row] = self._shapes[row].retyped(shape_type)
        self._changed(select=row)

    def _on_field_changed(self, *_args) -> None:
        shape = self._selected()
        if self._loading or shape is None:
            return
        shape.name = self._name_edit.text().strip() or shape.type
        shape.center = (
            self._center_spins[0].value(),
            self._center_spins[1].value(),
            self._center_spins[2].value(),
        )
        shape.axis = self._axis_combo.currentText()
        shape.params = {param: spin.value() for param, spin in self._param_spins.items()}
        item = self._shape_list.currentItem()
        if item is not None:
            item.setText(f"{shape.name}  ({shape.type})")
        self._changed(select=None, rebuild=False)

    def _changed(self, select: int | None = None, rebuild: bool = True) -> None:
        self._dirty = True
        if rebuild:
            self._refresh_list(select=select)
        self._publish()
        self._update_summary()

    def _update_summary(self) -> None:
        if self._name is None:
            self._summary.setText("No geometry yet: New… starts one.")
            return
        where = str(self._path) if self._path is not None else "Not saved yet"
        text = f"{where}{' (unsaved changes)' if self._dirty else ''} — {len(self._shapes)} shape(s)"
        if self._references:
            text += f", plus {len(self._references)} raw wireframe(s) kept as they are"
        self._summary.setText(text + ".")

    # -- drawing -----------------------------------------------------------

    def _config(self) -> dict:
        return geometry_config(self._base, self._shapes)

    def _publish(self) -> None:
        """Show the edit in every open space and skeleton plot, the selected shape highlighted."""
        if self._name is None:
            self.app_state.space_geometry_preview = None
            return
        selected = self._selected()
        shown = [
            Shape(s.name, s.type, s.center, dict(s.params), s.axis, _SELECTED_COLOR) if s is selected else s
            for s in self._shapes
        ]
        preview: dict = {"name": self._name, "config": geometry_config(self._base, shown)}
        if selected is not None:
            preview["axes"] = {"center": list(selected.center), "half_length": axes_half_length(selected)}
        self.app_state.space_geometry_preview = preview
        self.app_state.space_library_geometry = self._name

    # -- save / close ------------------------------------------------------

    def _save(self) -> None:
        if self._name is None:
            return
        path = Path(self._target_combo.currentData()) / f"{self._name}.yaml"
        try:
            write_geometry_yaml(path, self._config())
        except OSError as exc:
            notify(f"Could not save {path}: {exc}", "warning")
            return
        self._dirty = False
        notify(f"Saved {path}")
        self._reload_library(select=self._name, row=max(self._shape_list.currentRow(), 0))
        if self._on_saved is not None:
            self._on_saved()

    def _sync_panel_hint(self) -> None:
        self._no_panel.setVisible(self._panels_open is not None and not self._panels_open())

    def _on_open_panel(self) -> None:
        if self._open_panel is not None:
            self._open_panel()
        self._publish()
        self._sync_panel_hint()

    def changeEvent(self, event) -> None:
        # A plot opened or closed while this window was in the background.
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self._sync_panel_hint()
        super().changeEvent(event)

    def _drop_preview(self, *_args) -> None:
        """The plots go back to drawing the file."""
        self.app_state.space_geometry_preview = None

    def reject(self) -> None:
        if self._confirm_discard():
            super().reject()
