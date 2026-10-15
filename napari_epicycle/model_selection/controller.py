from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from qtpy.QtWidgets import QFileDialog

from .io import ModelPackStore
from .logic import FinalModelRegistryService


class ModelSelectionController:
    def __init__(
        self,
        widget,
        *,
        service: Optional[FinalModelRegistryService] = None,
        pack_store: Optional[ModelPackStore] = None,
    ):
        self.ui = widget
        self.service = service or FinalModelRegistryService()
        self.pack_store = pack_store or ModelPackStore()
        self.working_dir = ""
        self.imported_models: List[str] = []
        self.model_by_id: Dict[str, str] = {}
        self._connect_signals()

    def _connect_signals(self) -> None:
        self.ui.browse_workdir_btn.clicked.connect(self._browse_workdir)
        self.ui.refresh_btn.clicked.connect(self._refresh_workspace)
        self.ui.import_model_btn.clicked.connect(self._import_model)
        self.ui.clear_imports_btn.clicked.connect(self._clear_imports)
        self.ui.finalize_btn.clicked.connect(self._finalize_model_pack)
        self.ui.view_pack_btn.clicked.connect(self._view_selected_pack)

    def _browse_workdir(self) -> None:
        path = QFileDialog.getExistingDirectory(None, "Select Working Directory")
        if not path:
            return
        self.working_dir = path
        self.ui.workdir_line.setText(path)
        self._refresh_workspace()

    def _refresh_workspace(self) -> None:
        self.ui.workspace_models_list.clear()
        self.ui.lateral_model_combo.clear()
        self.ui.axial_model_combo.clear()
        self.ui.pack_list.clear()
        self.ui.lateral_model_combo.addItem("<none>")
        self.ui.axial_model_combo.addItem("<none>")
        self.model_by_id = {}
        if not self.working_dir:
            self.ui.status_label.setText("Set working directory.")
            return

        models = self.service.list_workspace_models(self.working_dir)
        for m in models:
            model_id = str(m.model_id)
            view_type = str(m.view_type).lower()
            display = f"{model_id} [{view_type}]"
            self.ui.workspace_models_list.addItem(display)
            self.model_by_id[model_id] = view_type
            if view_type in {"spatial", "xy"}:
                self.ui.lateral_model_combo.addItem(model_id)
            if view_type in {"axial", "xz", "yz"}:
                self.ui.axial_model_combo.addItem(model_id)

        benchmark = self.service.load_benchmark_summary(self.working_dir)
        if benchmark is None:
            self.ui.benchmark_summary_text.setPlainText("No benchmark summary found.")
        else:
            best = benchmark.get("best_round", "N/A")
            loss = benchmark.get("best_final_loss", "N/A")
            total = benchmark.get("num_rounds", 0)
            self.ui.benchmark_summary_text.setPlainText(
                f"Rounds: {total}\nBest: {best}\nBest final loss: {loss}"
            )

        for pack_path in self.pack_store.list_packs(self.working_dir):
            self.ui.pack_list.addItem(pack_path)
        self.ui.status_label.setText("Workspace refreshed.")

    def _import_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(None, "Import Model Artifact JSON", "", "JSON Files (*.json)")
        if not path:
            return
        if path not in self.imported_models:
            self.imported_models.append(path)
            self.ui.imported_models_list.addItem(path)
        self.ui.status_label.setText(f"Imported model: {Path(path).name}")

    def _clear_imports(self) -> None:
        self.imported_models = []
        self.ui.imported_models_list.clear()
        self.ui.status_label.setText("Cleared imported models.")

    def _finalize_model_pack(self) -> None:
        if not self.working_dir:
            self.ui.status_label.setText("Set working directory first.")
            return
        lateral = self.ui.lateral_model_combo.currentText().strip()
        axial = self.ui.axial_model_combo.currentText().strip()
        lateral = "" if lateral == "<none>" else lateral
        axial = "" if axial == "<none>" else axial
        notes = self.ui.notes_line.text().strip()
        result = self.service.register_final_model_pack(
            working_dir=self.working_dir,
            lateral_model_id=lateral,
            axial_model_id=axial,
            imported_model_uris=list(self.imported_models),
            notes=notes,
        )
        self.ui.pack_list.insertItem(0, result["pack_path"])
        self.ui.status_label.setText(f"Created model pack: {result['pack_id']}")

    def _view_selected_pack(self) -> None:
        selected = self.ui.pack_list.selectedItems()
        if not selected:
            return
        payload = self.pack_store.load_pack(selected[0].text())
        lines = [
            f"pack_id: {payload.get('pack_id', '')}",
            f"lateral_model_id: {payload.get('lateral_model_id', '')}",
            f"axial_model_id: {payload.get('axial_model_id', '')}",
            f"imported_models: {len(payload.get('imported_model_uris', []))}",
        ]
        self.ui.benchmark_summary_text.setPlainText("\n".join(lines))
        self.ui.status_label.setText("Loaded selected model pack details.")

    # ------------------------------------------------------------------
    # Step lifecycle hooks (called by workflow shell on step switch)
    # ------------------------------------------------------------------

    def on_step_exit(self):
        """Called when user navigates away from Finalize Model.  No-op."""
        pass

    def on_step_enter(self):
        """Called when user navigates to Finalize Model.  No-op."""
        pass
