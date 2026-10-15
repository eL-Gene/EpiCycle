from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from packages.epicycle_domain import (
    RoundMetadata,
    SelectedIndicesRecord,
    SliceAnnotationRecord,
    round_metadata_from_legacy,
)

from .artifact_registry import FileArtifactRegistry
from .state_store import JSONStateStore

logger = logging.getLogger(__name__)


class HITLPersistenceService:
    """
    Phase A service facade for HITL round + state persistence.

    Keeps controller code thin and centralizes compatibility behavior.
    """

    def __init__(self, state_store: JSONStateStore | None = None, artifact_registry: FileArtifactRegistry | None = None) -> None:
        self.state_store = state_store or JSONStateStore()
        self.artifact_registry = artifact_registry or FileArtifactRegistry()

    @staticmethod
    def _round_id(current_round: int) -> str:
        return f"round_{int(current_round):02d}"

    def _build_round_metadata(
        self,
        current_round: int,
        current_config: Dict[str, Any],
        training_data_by_round: Dict[Any, Any],
    ) -> RoundMetadata:
        round_id = self._round_id(current_round)
        # Phase B.1.1 fix (audit 2 HIGH-2): controller post-B.1 writes
        # ONLY composite keys `f"{bucket}_{N}"`.  Bare-int / str-int
        # legacy keys are never written.  Falling back through them
        # alone produces empty `raw_samples` → empty metadata + empty
        # selected_indices files on the legacy persistence path
        # (e.g. when the v3 save throws or for pre-migration users).
        # Now: also union the per-bucket composite keys for both
        # lateral and axial at the current round.
        raw_samples = training_data_by_round.get(current_round, [])
        if not raw_samples:
            raw_samples = training_data_by_round.get(str(current_round), [])
        if not raw_samples:
            # Composite-key fallback — collect from both buckets
            composite = []
            for _b in ("lateral", "axial"):
                _vals = training_data_by_round.get(f"{_b}_{int(current_round)}")
                if isinstance(_vals, list):
                    composite.extend(_vals)
            if composite:
                raw_samples = composite

        normalized_samples: List[SliceAnnotationRecord] = []
        for sample in raw_samples:
            if not isinstance(sample, dict):
                # Legacy in-memory state used simple slice label strings.
                continue
            required_keys = {"stack_id", "view", "slice_index"}
            has_paths = {"img_uri", "mask_uri"} <= set(sample.keys()) or {"img_path", "mask_path"} <= set(sample.keys())
            if not (required_keys <= set(sample.keys()) and has_paths):
                continue
            img_uri = sample.get("img_uri", sample.get("img_path"))
            mask_uri = sample.get("mask_uri", sample.get("mask_path"))
            try:
                normalized_samples.append(
                    SliceAnnotationRecord(
                        schema_version="1.0",
                        dataset_id=str(sample.get("dataset_id", "unknown_dataset")),
                        stack_id=str(sample["stack_id"]),
                        view=str(sample["view"]).upper(),
                        slice_index=int(sample["slice_index"]),
                        channel_map=dict(sample.get("channel_map", {})),
                        img_uri=str(img_uri),
                        mask_uri=str(mask_uri),
                        round_id=round_id,
                        sample_split=sample.get("sample_split"),
                    )
                )
            except Exception:
                # Keep state saving robust even for malformed legacy sample objects.
                continue

        return RoundMetadata(
            schema_version="1.0",
            round_id=round_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            config_snapshot=dict(current_config),
            samples=normalized_samples,
        )

    def _legacy_round_metadata_path(self, working_dir: str, round_id: str) -> Path:
        root = Path(working_dir) / self.artifact_registry.training_data_root / round_id
        return root / "round_metadata.json"

    def _build_selected_indices_record(self, round_id: str, training_data_by_round: Dict[Any, Any], current_round: int) -> SelectedIndicesRecord:
        # Phase B.1.1 fix (audit 2 HIGH-2): same composite-key fallback
        # as `_build_round_metadata` above.  Without this, the legacy
        # persistence path writes empty selected_indices.json files for
        # all post-B.1 state JSONs.
        raw_samples = training_data_by_round.get(current_round, [])
        if not raw_samples:
            raw_samples = training_data_by_round.get(str(current_round), [])
        if not raw_samples:
            composite = []
            for _b in ("lateral", "axial"):
                _vals = training_data_by_round.get(f"{_b}_{int(current_round)}")
                if isinstance(_vals, list):
                    composite.extend(_vals)
            if composite:
                raw_samples = composite

        selections: Dict[str, Dict[str, List[int]]] = {"XY": {}, "XZ": {}, "YZ": {}}
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            if not {"stack_id", "view", "slice_index"} <= set(sample.keys()):
                continue
            view = str(sample["view"]).upper()
            if view not in selections:
                continue
            stack_id = str(sample["stack_id"])
            idx = int(sample["slice_index"])
            selections[view].setdefault(stack_id, [])
            selections[view][stack_id].append(idx)

        for view, stacks in selections.items():
            for stack_id, indices in stacks.items():
                stacks[stack_id] = sorted(set(int(i) for i in indices))

        return SelectedIndicesRecord(schema_version="1.0", round_id=round_id, selections=selections)

    def _load_round_metadata_with_compat(self, working_dir: str, round_id: str) -> RoundMetadata | None:
        # First try canonical contract loader.
        try:
            metadata = self.artifact_registry.load_round_metadata(working_dir, round_id)
            if metadata is not None:
                return metadata
        except Exception:
            pass

        # Then attempt legacy adapter path.
        legacy_path = self._legacy_round_metadata_path(working_dir, round_id)
        if not legacy_path.exists():
            return None
        # Release-hardening #12 (2026-09-07): guard the legacy adapter the
        # same way the canonical loader above is guarded.  `round_metadata_
        # from_legacy` raises when a legacy sample lacks img/mask uris (a
        # cherry-picked pair, schema 1.1) — an unguarded raise here aborted
        # the whole workspace restore instead of degrading to "no legacy
        # metadata for this round".
        try:
            with legacy_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            metadata = round_metadata_from_legacy(payload)
            # Normalize by writing back canonical schema.
            self.artifact_registry.save_round_metadata(working_dir, metadata)
            return metadata
        except Exception as exc:
            logger.warning(
                "Legacy round metadata at %s could not be adapted (%s: %s); "
                "treating this round as having no legacy metadata",
                legacy_path, type(exc).__name__, exc,
            )
            return None

    def save_hitl_state(
        self,
        working_dir: str,
        state_dict: Dict[str, Any],
        current_round: int,
        current_config: Dict[str, Any],
        training_data_by_round: Dict[Any, Any],
    ) -> Tuple[Path, Path]:
        state_path = self.state_store.save_state(working_dir, state_dict)
        metadata = self._build_round_metadata(current_round, current_config, training_data_by_round)
        metadata_path = self.artifact_registry.save_round_metadata(working_dir, metadata)
        selected_indices = self._build_selected_indices_record(
            round_id=self._round_id(current_round),
            training_data_by_round=training_data_by_round,
            current_round=current_round,
        )
        self.artifact_registry.save_selected_indices(working_dir, selected_indices)
        return state_path, metadata_path

    def load_hitl_state(self, working_dir: str) -> Tuple[Dict[str, Any] | None, RoundMetadata | None]:
        state = self.state_store.load_state(working_dir)
        round_idx = 0 if not state else int(state.get("current_round", 0))
        round_id = self._round_id(round_idx)
        metadata = self._load_round_metadata_with_compat(working_dir, round_id)
        return state, metadata
