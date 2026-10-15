from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy.QtWidgets import QFileDialog

from napari_epicycle.domain import ScaledSegmentationJobSpec
from napari_epicycle.io_registry import FileArtifactRegistry

from .logic import ScaledSegmentationBridgeService


class LargeScaleSegmentationController:
    def __init__(
        self,
        widget,
        viewer,
        *,
        artifact_registry: Optional[FileArtifactRegistry] = None,
        bridge_service: Optional[ScaledSegmentationBridgeService] = None,
    ):
        self.ui = widget
        self.viewer = viewer
        self.working_dir = ""
        self.artifact_registry = artifact_registry or FileArtifactRegistry()
        self.bridge_service = bridge_service or ScaledSegmentationBridgeService()
        self._connect_signals()

    def _connect_signals(self) -> None:
        self.ui.browse_workdir_btn.clicked.connect(self._browse_workdir)
        self.ui.browse_input_btn.clicked.connect(self._browse_input_volume)
        self.ui.browse_model_btn.clicked.connect(self._browse_model_artifact)
        self.ui.generate_job_btn.clicked.connect(self._generate_job)
        self.ui.clear_jobs_btn.clicked.connect(self._clear_jobs)
        self.ui.view_output_btn.clicked.connect(self._view_selected_output)

    def _browse_workdir(self) -> None:
        path = QFileDialog.getExistingDirectory(None, "Select Working Directory")
        if not path:
            return
        self.working_dir = path
        self.ui.workdir_line.setText(path)

    def _browse_input_volume(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Input Volume",
            "",
            "Volume Files (*.tif *.tiff *.npy)",
        )
        if path:
            self.ui.input_volume_line.setText(path)

    def _browse_model_artifact(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Model Artifact JSON",
            "",
            "JSON Files (*.json)",
        )
        if path:
            self.ui.model_artifact_line.setText(path)

    def _build_spec(self) -> ScaledSegmentationJobSpec:
        round_id = self.ui.round_id_line.text().strip() or "round_00"
        return ScaledSegmentationJobSpec(
            schema_version="1.0",
            round_id=round_id,
            input_volume_uri=self.ui.input_volume_line.text().strip(),
            model_artifact_uri=self.ui.model_artifact_line.text().strip(),
            output_target=str(Path(self.working_dir) / "artifacts" / "scaled_segmentation"),
            tile_shape_zyx=[
                int(self.ui.tile_z_sb.value()),
                int(self.ui.tile_y_sb.value()),
                int(self.ui.tile_x_sb.value()),
            ],
            overlap_zyx=[
                int(self.ui.overlap_z_sb.value()),
                int(self.ui.overlap_y_sb.value()),
                int(self.ui.overlap_x_sb.value()),
            ],
            run_mode=self.ui.run_mode_combo.currentText(),
            scheduler_params={
                "partition": self.ui.partition_line.text().strip() or "gpu",
                "gpus": int(self.ui.gpus_sb.value()),
                "cpus_per_task": int(self.ui.cpus_sb.value()),
                "mem": self.ui.mem_line.text().strip() or "32G",
                "time": self.ui.time_line.text().strip() or "08:00:00",
            },
        )

    def _generate_job(self) -> None:
        if not self.working_dir:
            print("Please set a working directory first.")
            return
        spec = self._build_spec()
        self.artifact_registry.save_scaled_segmentation_job_spec(self.working_dir, spec)

        if spec.run_mode == "slurm":
            result = self.bridge_service.create_submission_bundle(working_dir=self.working_dir, spec=spec)
            self.ui.jobs_list.addItem(result["submit_script_path"])
            self.ui.status_label.setText(f"SLURM bundle ready: {result['job_id']}")
            print(f"[INFO] Generated SLURM bundle: {result['submit_script_path']}")
            return

        result = self.bridge_service.run_local_placeholder(
            working_dir=self.working_dir,
            spec=spec,
            source_volume_path=spec.input_volume_uri,
        )
        self.ui.jobs_list.addItem(result["manifest_path"])
        self.ui.status_label.setText(f"Local placeholder run complete: {result['job_id']}")
        print(f"[INFO] Generated local scaled-segmentation output: {result['manifest_path']}")

    def _clear_jobs(self) -> None:
        self.ui.jobs_list.clear()
        self.ui.status_label.setText("Idle")

    def _view_selected_output(self) -> None:
        if self.viewer is None:
            print("Viewer is not initialized.")
            return
        selected = self.ui.jobs_list.selectedItems()
        if not selected:
            print("Select a local manifest path to view output labels.")
            return
        path = Path(selected[0].text())
        if path.suffix.lower() != ".json" or not path.exists():
            print("Selected item is not a local output manifest.")
            return
        import json
        import tifffile

        payload = json.loads(path.read_text(encoding="utf-8"))
        labels_path = payload.get("labels_path", "")
        if not labels_path or not Path(labels_path).exists():
            print("Manifest has no valid labels path.")
            return
        labels = tifffile.imread(labels_path)
        layer_name = f"scaledseg_{payload.get('job_id', 'latest')}"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
        self.viewer.add_labels(labels, name=layer_name)
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.register_existing(layer_name, "step6/tiled")

    # ------------------------------------------------------------------
    # Step lifecycle hooks (called by workflow shell on step switch)
    # ------------------------------------------------------------------

    def on_step_exit(self):
        """Called when user navigates away from Step 6."""
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.clear_step("step6")

    def on_step_enter(self):
        """Called when user navigates to Step 6.  Currently a no-op."""
        pass
