from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SUPPORTED_VIEWS = {"XY", "XZ", "YZ"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required key: {key}")
    return data[key]


@dataclass(frozen=True)
class SliceAnnotationRecord:
    schema_version: str
    dataset_id: str
    stack_id: str
    view: str
    slice_index: int
    channel_map: Dict[str, int]
    img_uri: str
    mask_uri: str
    round_id: str
    sample_split: Optional[str] = None

    def validate(self) -> None:
        if self.view not in SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported view: {self.view}")
        if self.slice_index < 0:
            raise ValueError("slice_index must be >= 0")
        if not self.dataset_id or not self.stack_id:
            raise ValueError("dataset_id and stack_id must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SliceAnnotationRecord":
        record = cls(
            schema_version=str(_required(data, "schema_version")),
            dataset_id=str(_required(data, "dataset_id")),
            stack_id=str(_required(data, "stack_id")),
            view=str(_required(data, "view")),
            slice_index=int(_required(data, "slice_index")),
            channel_map=dict(_required(data, "channel_map")),
            img_uri=str(_required(data, "img_uri")),
            mask_uri=str(_required(data, "mask_uri")),
            round_id=str(_required(data, "round_id")),
            sample_split=data.get("sample_split"),
        )
        record.validate()
        return record


@dataclass(frozen=True)
class RoundMetadata:
    schema_version: str
    round_id: str
    created_at: str
    config_snapshot: Dict[str, Any]
    samples: List[SliceAnnotationRecord] = field(default_factory=list)
    parent_round: Optional[str] = None

    def validate(self) -> None:
        if not self.round_id:
            raise ValueError("round_id must be non-empty")
        for sample in self.samples:
            sample.validate()

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "created_at": self.created_at,
            "config_snapshot": self.config_snapshot,
            "parent_round": self.parent_round,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoundMetadata":
        samples = [SliceAnnotationRecord.from_dict(item) for item in data.get("samples", [])]
        obj = cls(
            schema_version=str(_required(data, "schema_version")),
            round_id=str(_required(data, "round_id")),
            created_at=str(data.get("created_at", _utcnow_iso())),
            config_snapshot=dict(data.get("config_snapshot", {})),
            parent_round=data.get("parent_round"),
            samples=samples,
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class SelectedIndicesRecord:
    schema_version: str
    round_id: str
    selections: Dict[str, Dict[str, List[int]]]

    def validate(self) -> None:
        if not self.round_id:
            raise ValueError("round_id must be non-empty")
        for view, stacks in self.selections.items():
            if view not in SUPPORTED_VIEWS:
                raise ValueError(f"Unsupported view in selections: {view}")
            for stack_id, indices in stacks.items():
                if not stack_id:
                    raise ValueError("stack_id must be non-empty")
                for idx in indices:
                    if int(idx) < 0:
                        raise ValueError("slice indices must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectedIndicesRecord":
        selections = dict(_required(data, "selections"))
        normalized: Dict[str, Dict[str, List[int]]] = {}
        for view, stacks in selections.items():
            normalized[view] = {}
            for stack_id, indices in dict(stacks).items():
                normalized[view][stack_id] = sorted(set(int(x) for x in list(indices)))
        obj = cls(
            schema_version=str(_required(data, "schema_version")),
            round_id=str(_required(data, "round_id")),
            selections=normalized,
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class ModelArtifact:
    schema_version: str
    model_id: str
    view_type: str
    checkpoint_uri: str
    training_data_ref: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.view_type not in {"spatial", "axial", "xy", "xz", "yz"}:
            raise ValueError(f"Unsupported view_type: {self.view_type}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelArtifact":
        obj = cls(
            schema_version=str(_required(data, "schema_version")),
            model_id=str(_required(data, "model_id")),
            view_type=str(_required(data, "view_type")),
            checkpoint_uri=str(_required(data, "checkpoint_uri")),
            training_data_ref=str(_required(data, "training_data_ref")),
            metrics={k: float(v) for k, v in dict(data.get("metrics", {})).items()},
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class InferenceArtifact:
    schema_version: str
    model_id: str
    dataset_id: str
    view_type: str
    prediction_uri: str
    timestamp: str = field(default_factory=_utcnow_iso)

    def validate(self) -> None:
        if self.view_type not in {"XY", "XZ", "YZ", "spatial", "axial"}:
            raise ValueError(f"Unsupported view_type: {self.view_type}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceArtifact":
        obj = cls(
            schema_version=str(_required(data, "schema_version")),
            model_id=str(_required(data, "model_id")),
            dataset_id=str(_required(data, "dataset_id")),
            view_type=str(_required(data, "view_type")),
            prediction_uri=str(_required(data, "prediction_uri")),
            timestamp=str(data.get("timestamp", _utcnow_iso())),
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class ReconstructionJobSpec:
    schema_version: str
    input_predictions: List[str]
    fusion_params: Dict[str, Any]
    output_target: str
    run_mode: str

    def validate(self) -> None:
        if self.run_mode not in {"local", "slurm"}:
            raise ValueError(f"Unsupported run_mode: {self.run_mode}")
        if not self.input_predictions:
            raise ValueError("input_predictions must contain at least one prediction path")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconstructionJobSpec":
        obj = cls(
            schema_version=str(_required(data, "schema_version")),
            input_predictions=list(_required(data, "input_predictions")),
            fusion_params=dict(data.get("fusion_params", {})),
            output_target=str(_required(data, "output_target")),
            run_mode=str(_required(data, "run_mode")),
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class ScaledSegmentationJobSpec:
    schema_version: str
    round_id: str
    input_volume_uri: str
    model_artifact_uri: str
    output_target: str
    tile_shape_zyx: List[int]
    overlap_zyx: List[int]
    run_mode: str
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.run_mode not in {"local", "slurm"}:
            raise ValueError(f"Unsupported run_mode: {self.run_mode}")
        if len(self.tile_shape_zyx) != 3 or any(int(x) <= 0 for x in self.tile_shape_zyx):
            raise ValueError("tile_shape_zyx must contain 3 positive integers")
        if len(self.overlap_zyx) != 3 or any(int(x) < 0 for x in self.overlap_zyx):
            raise ValueError("overlap_zyx must contain 3 non-negative integers")
        if not self.round_id:
            raise ValueError("round_id must be non-empty")
        if not self.input_volume_uri:
            raise ValueError("input_volume_uri must be non-empty")
        if not self.model_artifact_uri:
            raise ValueError("model_artifact_uri must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScaledSegmentationJobSpec":
        obj = cls(
            schema_version=str(_required(data, "schema_version")),
            round_id=str(_required(data, "round_id")),
            input_volume_uri=str(_required(data, "input_volume_uri")),
            model_artifact_uri=str(_required(data, "model_artifact_uri")),
            output_target=str(_required(data, "output_target")),
            tile_shape_zyx=[int(x) for x in list(_required(data, "tile_shape_zyx"))],
            overlap_zyx=[int(x) for x in list(_required(data, "overlap_zyx"))],
            run_mode=str(_required(data, "run_mode")),
            scheduler_params=dict(data.get("scheduler_params", {})),
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class StackSegmentationJobSpec:
    schema_version: str
    input_volume_uri: str
    model_pack_uri: str
    output_target: str
    run_mode: str
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.run_mode not in {"local"}:
            raise ValueError(f"Unsupported run_mode: {self.run_mode}")
        if not self.input_volume_uri:
            raise ValueError("input_volume_uri must be non-empty")
        if not self.model_pack_uri:
            raise ValueError("model_pack_uri must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StackSegmentationJobSpec":
        obj = cls(
            schema_version=str(_required(data, "schema_version")),
            input_volume_uri=str(_required(data, "input_volume_uri")),
            model_pack_uri=str(_required(data, "model_pack_uri")),
            output_target=str(_required(data, "output_target")),
            run_mode=str(_required(data, "run_mode")),
            params=dict(data.get("params", {})),
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class CurationSessionRecord:
    schema_version: str
    source_artifact: str
    workflows_used: List[str]
    handoff_chain: List[str]
    accept_events: List[Dict[str, Any]] = field(default_factory=list)
    reject_events: List[Dict[str, Any]] = field(default_factory=list)
    final_label_id_map: Dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.source_artifact:
            raise ValueError("source_artifact must be non-empty")
        if not self.workflows_used:
            raise ValueError("workflows_used must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurationSessionRecord":
        obj = cls(
            schema_version=str(_required(data, "schema_version")),
            source_artifact=str(_required(data, "source_artifact")),
            workflows_used=list(_required(data, "workflows_used")),
            handoff_chain=list(data.get("handoff_chain", [])),
            accept_events=list(data.get("accept_events", [])),
            reject_events=list(data.get("reject_events", [])),
            final_label_id_map={str(k): int(v) for k, v in dict(data.get("final_label_id_map", {})).items()},
        )
        obj.validate()
        return obj
