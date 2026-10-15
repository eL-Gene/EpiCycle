from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from packages.epicycle_domain import RoundMetadata, SelectedIndicesRecord, round_metadata_from_legacy

from .artifact_registry import FileArtifactRegistry


def build_selected_indices_from_round_metadata(metadata: RoundMetadata) -> SelectedIndicesRecord:
    selections: Dict[str, Dict[str, List[int]]] = {"XY": {}, "XZ": {}, "YZ": {}}

    for sample in metadata.samples:
        view = sample.view.upper()
        if view not in selections:
            continue
        selections[view].setdefault(sample.stack_id, [])
        selections[view][sample.stack_id].append(int(sample.slice_index))

    for view, stacks in selections.items():
        for stack_id, indices in stacks.items():
            stacks[stack_id] = sorted(set(indices))

    return SelectedIndicesRecord(
        schema_version=metadata.schema_version,
        round_id=metadata.round_id,
        selections=selections,
    )


def load_round_metadata_payload(path: Path) -> RoundMetadata:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        return RoundMetadata.from_dict(payload)
    except Exception:
        return round_metadata_from_legacy(payload)


def migrate_training_data_tree(
    working_dir: str,
    training_data_root: str = "training_data",
    write_selected_indices: bool = True,
    overwrite_selected_indices: bool = False,
) -> List[Dict[str, Any]]:
    base = Path(working_dir) / training_data_root
    if not base.exists():
        return []

    registry = FileArtifactRegistry(training_data_root=training_data_root)
    summaries: List[Dict[str, Any]] = []

    for round_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("round_")):
        metadata_path = round_dir / "round_metadata.json"
        if not metadata_path.exists():
            continue

        metadata = load_round_metadata_payload(metadata_path)
        registry.save_round_metadata(working_dir, metadata)

        selected_written = False
        if write_selected_indices:
            existing = registry.load_selected_indices(working_dir, metadata.round_id)
            if existing is None or overwrite_selected_indices:
                selected = build_selected_indices_from_round_metadata(metadata)
                registry.save_selected_indices(working_dir, selected)
                selected_written = True

        summaries.append(
            {
                "round_id": metadata.round_id,
                "samples": len(metadata.samples),
                "selected_indices_written": selected_written,
            }
        )

    return summaries
