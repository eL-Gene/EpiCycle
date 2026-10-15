from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ModelPackStore:
    def list_packs(self, working_dir: str) -> List[str]:
        root = Path(working_dir) / "artifacts" / "model_registry"
        if not root.exists():
            return []
        packs = [p for p in root.glob("final_model_pack_*.json") if p.is_file()]
        packs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p) for p in packs]

    def load_pack(self, pack_path: str) -> Dict[str, Any]:
        path = Path(pack_path)
        return json.loads(path.read_text(encoding="utf-8"))
