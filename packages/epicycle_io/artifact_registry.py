from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from packages.epicycle_domain import (
    CurationSessionRecord,
    InferenceArtifact,
    ModelArtifact,
    ReconstructionJobSpec,
    ScaledSegmentationJobSpec,
    StackSegmentationJobSpec,
    RoundMetadata,
    SelectedIndicesRecord,
)

from .state_store import atomic_write_json


class ArtifactRegistry(Protocol):
    def save_round_metadata(self, working_dir: str, metadata: RoundMetadata) -> Path:
        ...

    def load_round_metadata(self, working_dir: str, round_id: str) -> RoundMetadata | None:
        ...

    def save_model_artifact(self, working_dir: str, artifact: ModelArtifact) -> Path:
        ...

    def save_selected_indices(self, working_dir: str, record: SelectedIndicesRecord) -> Path:
        ...

    def load_selected_indices(self, working_dir: str, round_id: str) -> SelectedIndicesRecord | None:
        ...

    def load_model_artifact(self, working_dir: str, model_id: str) -> ModelArtifact | None:
        ...

    def save_inference_artifact(self, working_dir: str, artifact: InferenceArtifact) -> Path:
        ...

    def load_inference_artifact(self, working_dir: str, model_id: str, view_type: str) -> InferenceArtifact | None:
        ...

    def save_reconstruction_job_spec(self, working_dir: str, spec: ReconstructionJobSpec) -> Path:
        ...

    def save_curation_session(self, working_dir: str, session: CurationSessionRecord) -> Path:
        ...

    def save_scaled_segmentation_job_spec(self, working_dir: str, spec: ScaledSegmentationJobSpec) -> Path:
        ...

    def save_stack_segmentation_job_spec(self, working_dir: str, spec: StackSegmentationJobSpec) -> Path:
        ...


class FileArtifactRegistry:
    """
    Filesystem-backed artifact registry for Phase A.

    This is a landing-zone implementation for stable contracts.
    It intentionally uses simple JSON persistence and can be swapped later.
    """

    def __init__(self, training_data_root: str = "training_data", artifacts_root: str = "artifacts") -> None:
        self.training_data_root = training_data_root
        self.artifacts_root = artifacts_root

    def _ensure_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_json(self, path: Path, payload: dict) -> Path:
        """Atomic JSON write (release-hardening 1b#4, 2026-09-07).

        This is the canonical writer behind `save_round_metadata` and six
        siblings (model / inference / dataset / session / spec artifacts).
        It was truncate-then-rewrite: a crash mid-write left a truncated
        artifact that every downstream reader treats as corrupt.  Now shares
        `atomic_write_json` with the state store.
        """
        self._ensure_dir(path.parent)
        return atomic_write_json(path, payload, tmp_prefix=".epicycle_artifact_")

    def _read_json(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _round_dir(self, working_dir: str, round_id: str) -> Path:
        return Path(working_dir) / self.training_data_root / round_id

    def save_round_metadata(self, working_dir: str, metadata: RoundMetadata) -> Path:
        path = self._round_dir(working_dir, metadata.round_id) / "round_metadata.json"
        return self._write_json(path, metadata.to_dict())

    def load_round_metadata(self, working_dir: str, round_id: str) -> RoundMetadata | None:
        path = self._round_dir(working_dir, round_id) / "round_metadata.json"
        payload = self._read_json(path)
        if payload is None:
            return None
        return RoundMetadata.from_dict(payload)

    def save_model_artifact(self, working_dir: str, artifact: ModelArtifact) -> Path:
        path = Path(working_dir) / self.artifacts_root / "models" / f"{artifact.model_id}.json"
        return self._write_json(path, artifact.to_dict())

    def save_selected_indices(self, working_dir: str, record: SelectedIndicesRecord) -> Path:
        path = self._round_dir(working_dir, record.round_id) / "selected_indices.json"
        return self._write_json(path, record.to_dict())

    def load_selected_indices(self, working_dir: str, round_id: str) -> SelectedIndicesRecord | None:
        path = self._round_dir(working_dir, round_id) / "selected_indices.json"
        payload = self._read_json(path)
        if payload is None:
            return None
        return SelectedIndicesRecord.from_dict(payload)

    def load_model_artifact(self, working_dir: str, model_id: str) -> ModelArtifact | None:
        path = Path(working_dir) / self.artifacts_root / "models" / f"{model_id}.json"
        payload = self._read_json(path)
        if payload is None:
            return None
        return ModelArtifact.from_dict(payload)

    def save_inference_artifact(self, working_dir: str, artifact: InferenceArtifact) -> Path:
        file_name = f"{artifact.model_id}_{artifact.view_type}.json"
        path = Path(working_dir) / self.artifacts_root / "inference" / file_name
        return self._write_json(path, artifact.to_dict())

    def load_inference_artifact(self, working_dir: str, model_id: str, view_type: str) -> InferenceArtifact | None:
        file_name = f"{model_id}_{view_type}.json"
        path = Path(working_dir) / self.artifacts_root / "inference" / file_name
        payload = self._read_json(path)
        if payload is None:
            return None
        return InferenceArtifact.from_dict(payload)

    def save_reconstruction_job_spec(self, working_dir: str, spec: ReconstructionJobSpec) -> Path:
        path = Path(working_dir) / self.artifacts_root / "reconstruction" / "latest_job_spec.json"
        return self._write_json(path, spec.to_dict())

    def save_curation_session(self, working_dir: str, session: CurationSessionRecord) -> Path:
        path = Path(working_dir) / self.artifacts_root / "curation" / "latest_session.json"
        return self._write_json(path, session.to_dict())

    def save_scaled_segmentation_job_spec(self, working_dir: str, spec: ScaledSegmentationJobSpec) -> Path:
        path = Path(working_dir) / self.artifacts_root / "scaled_segmentation" / "latest_job_spec.json"
        return self._write_json(path, spec.to_dict())

    def save_stack_segmentation_job_spec(self, working_dir: str, spec: StackSegmentationJobSpec) -> Path:
        path = Path(working_dir) / self.artifacts_root / "segmentation" / "latest_stack_job_spec.json"
        return self._write_json(path, spec.to_dict())
