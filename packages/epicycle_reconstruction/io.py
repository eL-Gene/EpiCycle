from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReconstructionRunStore:
    def list_manifests(
        self, working_dir: str, reconstructions_root: Optional[str] = None,
    ) -> List[str]:
        # 2026-05-20 Step 3 directory restructure: reconstruction runs
        # now live under ``artifacts/step3/runs/{patch_name}/`` (was
        # ``artifacts/reconstruction/{run_id}/``).  Path hardcoded here
        # because this module lives in ``packages/`` and the layer
        # boundary rule (packages never import the plugin) prevents importing
        # ``napari_epicycle.fusion_reconstruction._paths``.  When this
        # module is folded into ``napari_epicycle/core/reconstruction/``,
        # switch to
        # ``_paths.runs_dir(working_dir)``.
        #
        # 2026-06-27 Step 3 multi-run (Phase 1b-ii): the napari controller
        # can pass ``reconstructions_root`` — a plain string it computed via
        # ``_paths.reconstructions_dir(ws, run_id)`` (keeps this package
        # layout-dumb + napari-free).  When set, glob directly under it.
        # For the DEFAULT run that root IS ``artifacts/step3/runs`` so the
        # glob is byte-identical to the hardcoded default below.
        if reconstructions_root is not None:
            root = Path(reconstructions_root)
        else:
            root = Path(working_dir) / "artifacts" / "step3" / "runs"
        if not root.exists():
            return []
        manifests = [p for p in root.glob("*/manifest.json") if p.is_file()]
        manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p) for p in manifests]

    def load_manifest(self, manifest_path: str) -> Dict[str, Any]:
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
