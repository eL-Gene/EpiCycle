from __future__ import annotations

from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .controller import LargeScaleSegmentationController


class LargeScaleSegmentation(QWidget):
    """Whole-tissue segmentation job bridge (local placeholder + SLURM bundle)."""

    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.setLayout(self._build_layout())
        self.controller = LargeScaleSegmentationController(widget=self, viewer=viewer)

    def _build_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self._build_inputs_group())
        layout.addWidget(self._build_runtime_group())
        layout.addWidget(self._build_actions_group())
        return layout

    def _build_inputs_group(self):
        group = QGroupBox("Scaled Segmentation Inputs")
        form = QFormLayout()

        self.workdir_line = QLineEdit()
        self.browse_workdir_btn = QPushButton("Browse")
        workdir_row = QHBoxLayout()
        workdir_row.addWidget(self.workdir_line)
        workdir_row.addWidget(self.browse_workdir_btn)
        workdir_widget = QWidget()
        workdir_widget.setLayout(workdir_row)
        form.addRow("Working directory:", workdir_widget)

        self.round_id_line = QLineEdit("round_00")
        form.addRow("Round ID:", self.round_id_line)

        self.input_volume_line = QLineEdit()
        self.browse_input_btn = QPushButton("Browse")
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_volume_line)
        input_row.addWidget(self.browse_input_btn)
        input_widget = QWidget()
        input_widget.setLayout(input_row)
        form.addRow("Input volume (.tif/.npy):", input_widget)

        self.model_artifact_line = QLineEdit()
        self.browse_model_btn = QPushButton("Browse")
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_artifact_line)
        model_row.addWidget(self.browse_model_btn)
        model_widget = QWidget()
        model_widget.setLayout(model_row)
        form.addRow("Model artifact (.json):", model_widget)

        group.setLayout(form)
        return group

    def _build_runtime_group(self):
        group = QGroupBox("Tiling + Runtime")
        grid = QGridLayout()

        self.run_mode_combo = QComboBox()
        self.run_mode_combo.addItems(["local", "slurm"])
        grid.addWidget(QLabel("Run mode"), 0, 0)
        grid.addWidget(self.run_mode_combo, 0, 1)

        self.tile_z_sb = QSpinBox()
        self.tile_y_sb = QSpinBox()
        self.tile_x_sb = QSpinBox()
        for sb, val in ((self.tile_z_sb, 32), (self.tile_y_sb, 256), (self.tile_x_sb, 256)):
            sb.setRange(1, 4096)
            sb.setValue(val)
        grid.addWidget(QLabel("Tile Z"), 1, 0)
        grid.addWidget(self.tile_z_sb, 1, 1)
        grid.addWidget(QLabel("Tile Y"), 1, 2)
        grid.addWidget(self.tile_y_sb, 1, 3)
        grid.addWidget(QLabel("Tile X"), 1, 4)
        grid.addWidget(self.tile_x_sb, 1, 5)

        self.overlap_z_sb = QSpinBox()
        self.overlap_y_sb = QSpinBox()
        self.overlap_x_sb = QSpinBox()
        for sb, val in ((self.overlap_z_sb, 4), (self.overlap_y_sb, 32), (self.overlap_x_sb, 32)):
            sb.setRange(0, 2048)
            sb.setValue(val)
        grid.addWidget(QLabel("Overlap Z"), 2, 0)
        grid.addWidget(self.overlap_z_sb, 2, 1)
        grid.addWidget(QLabel("Overlap Y"), 2, 2)
        grid.addWidget(self.overlap_y_sb, 2, 3)
        grid.addWidget(QLabel("Overlap X"), 2, 4)
        grid.addWidget(self.overlap_x_sb, 2, 5)

        self.partition_line = QLineEdit("gpu")
        self.gpus_sb = QSpinBox()
        self.gpus_sb.setRange(0, 16)
        self.gpus_sb.setValue(1)
        self.cpus_sb = QSpinBox()
        self.cpus_sb.setRange(1, 128)
        self.cpus_sb.setValue(8)
        self.mem_line = QLineEdit("32G")
        self.time_line = QLineEdit("08:00:00")

        grid.addWidget(QLabel("SLURM partition"), 3, 0)
        grid.addWidget(self.partition_line, 3, 1)
        grid.addWidget(QLabel("GPUs"), 3, 2)
        grid.addWidget(self.gpus_sb, 3, 3)
        grid.addWidget(QLabel("CPUs/task"), 3, 4)
        grid.addWidget(self.cpus_sb, 3, 5)

        grid.addWidget(QLabel("Memory"), 4, 0)
        grid.addWidget(self.mem_line, 4, 1)
        grid.addWidget(QLabel("Time"), 4, 2)
        grid.addWidget(self.time_line, 4, 3)

        group.setLayout(grid)
        return group

    def _build_actions_group(self):
        group = QGroupBox("Job Actions")
        layout = QVBoxLayout()

        actions = QHBoxLayout()
        self.generate_job_btn = QPushButton("Generate / Run")
        self.clear_jobs_btn = QPushButton("Clear")
        self.view_output_btn = QPushButton("View Selected Output")
        actions.addWidget(self.generate_job_btn)
        actions.addWidget(self.clear_jobs_btn)
        actions.addWidget(self.view_output_btn)

        self.jobs_list = QListWidget()
        self.status_label = QLabel("Idle")

        layout.addLayout(actions)
        layout.addWidget(self.jobs_list)
        layout.addWidget(self.status_label)
        group.setLayout(layout)
        return group
