from __future__ import annotations

from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .controller import ModelSelectionController


class ModelSelection(QWidget):
    """Final model selection and pack registration surface."""

    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.setLayout(self._build_layout())
        self.controller = ModelSelectionController(widget=self)

    def _build_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self._build_workspace_group())
        layout.addWidget(self._build_selection_group())
        layout.addWidget(self._build_import_group())
        layout.addWidget(self._build_pack_group())
        return layout

    def _build_workspace_group(self):
        group = QGroupBox("Workspace + Candidates")
        vbox = QVBoxLayout()

        row = QHBoxLayout()
        self.workdir_line = QLineEdit()
        self.browse_workdir_btn = QPushButton("Browse")
        self.refresh_btn = QPushButton("Refresh")
        row.addWidget(self.workdir_line)
        row.addWidget(self.browse_workdir_btn)
        row.addWidget(self.refresh_btn)

        self.workspace_models_list = QListWidget()
        self.workspace_models_list.setMinimumHeight(120)

        self.benchmark_summary_text = QTextEdit()
        self.benchmark_summary_text.setReadOnly(True)
        self.benchmark_summary_text.setMinimumHeight(90)

        vbox.addLayout(row)
        vbox.addWidget(QLabel("Workspace model artifacts"))
        vbox.addWidget(self.workspace_models_list)
        vbox.addWidget(QLabel("Benchmark summary"))
        vbox.addWidget(self.benchmark_summary_text)
        group.setLayout(vbox)
        return group

    def _build_selection_group(self):
        group = QGroupBox("Final Model Selection")
        form = QFormLayout()
        self.lateral_model_combo = QComboBox()
        self.axial_model_combo = QComboBox()
        self.lateral_model_combo.addItem("<none>")
        self.axial_model_combo.addItem("<none>")
        self.notes_line = QLineEdit()
        form.addRow("Lateral model:", self.lateral_model_combo)
        form.addRow("Axial model:", self.axial_model_combo)
        form.addRow("Notes:", self.notes_line)
        group.setLayout(form)
        return group

    def _build_import_group(self):
        group = QGroupBox("Import External Models")
        vbox = QVBoxLayout()
        row = QHBoxLayout()
        self.import_model_btn = QPushButton("Import Model JSON")
        self.clear_imports_btn = QPushButton("Clear Imports")
        row.addWidget(self.import_model_btn)
        row.addWidget(self.clear_imports_btn)
        self.imported_models_list = QListWidget()
        self.imported_models_list.setMinimumHeight(80)
        vbox.addLayout(row)
        vbox.addWidget(self.imported_models_list)
        group.setLayout(vbox)
        return group

    def _build_pack_group(self):
        group = QGroupBox("Finalize + Model Packs")
        vbox = QVBoxLayout()
        row = QHBoxLayout()
        self.finalize_btn = QPushButton("Finalize Model Pack")
        self.view_pack_btn = QPushButton("View Selected Pack")
        row.addWidget(self.finalize_btn)
        row.addWidget(self.view_pack_btn)

        self.pack_list = QListWidget()
        self.pack_list.setMinimumHeight(100)
        self.status_label = QLabel("Idle")

        vbox.addLayout(row)
        vbox.addWidget(self.pack_list)
        vbox.addWidget(self.status_label)
        group.setLayout(vbox)
        return group
