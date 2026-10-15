from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from packages.epicycle_domain import ModelArtifact


class FinalModelRegistryService:
    def _models_dir(self, working_dir: str) -> Path:
        return Path(working_dir) / "artifacts" / "models"

    def list_workspace_models(self, working_dir: str) -> List[ModelArtifact]:
        models_dir = self._models_dir(working_dir)
        if not models_dir.exists():
            return []
        out: List[ModelArtifact] = []
        for path in sorted(models_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                out.append(ModelArtifact.from_dict(payload))
            except Exception:
                continue
        return out

    def load_benchmark_summary(self, working_dir: str) -> Optional[Dict[str, Any]]:
        path = Path(working_dir) / "artifacts" / "benchmarks" / "round_benchmark_summary.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def register_final_model_pack(
        self,
        *,
        working_dir: str,
        lateral_model_id: str,
        axial_model_id: str,
        imported_model_uris: List[str],
        notes: str = "",
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc)
        pack_id = f"final_model_pack_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        out_dir = Path(working_dir) / "artifacts" / "model_registry"
        out_dir.mkdir(parents=True, exist_ok=True)

        pack = {
            "pack_id": pack_id,
            "created_at": timestamp.isoformat(),
            "lateral_model_id": lateral_model_id,
            "axial_model_id": axial_model_id,
            "imported_model_uris": list(imported_model_uris),
            "notes": notes,
        }
        pack_path = out_dir / f"{pack_id}.json"
        pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")

        latest_path = out_dir / "final_model_pack.json"
        latest_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
        return {
            "pack_id": pack_id,
            "pack_path": str(pack_path),
            "latest_path": str(latest_path),
        }
