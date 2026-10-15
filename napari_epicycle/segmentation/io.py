from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class StackSegmentationRunStore:
    def list_manifests(self, working_dir: str) -> List[str]:
        # Step 4 outputs: artifacts/step4/runs/
        root = Path(working_dir) / "artifacts" / "step4" / "runs"
        if not root.exists():
            # Fallback: legacy path
            root = Path(working_dir) / "artifacts" / "segmentation" / "stack_runs"
        if not root.exists():
            return []
        manifests = [p for p in root.glob("*/manifest.json") if p.is_file()]
        manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p) for p in manifests]

    def load_manifest(self, manifest_path: str) -> Dict[str, Any]:
        path = Path(manifest_path)
        return json.loads(path.read_text(encoding="utf-8"))
