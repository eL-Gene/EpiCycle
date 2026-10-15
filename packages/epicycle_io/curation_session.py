from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from packages.epicycle_domain import CurationSessionRecord

from .artifact_registry import FileArtifactRegistry


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CurationSessionService:
    def __init__(self, artifact_registry: FileArtifactRegistry) -> None:
        self.artifact_registry = artifact_registry
        self._active: Optional[CurationSessionRecord] = None

    @property
    def active(self) -> Optional[CurationSessionRecord]:
        return self._active

    def start_session(self, *, source_artifact: str, workflow_name: str, handoff_from: Optional[str] = None) -> CurationSessionRecord:
        handoff_chain = [f"{handoff_from}->{workflow_name}"] if handoff_from else [workflow_name]
        self._active = CurationSessionRecord(
            schema_version="1.0",
            source_artifact=source_artifact,
            workflows_used=[workflow_name],
            handoff_chain=handoff_chain,
            accept_events=[],
            reject_events=[],
            final_label_id_map={},
        )
        return self._active

    def ensure_session(self, *, source_artifact: str, workflow_name: str) -> CurationSessionRecord:
        if self._active is None:
            return self.start_session(source_artifact=source_artifact, workflow_name=workflow_name)
        return self._active

    def record_handoff(self, *, to_workflow: str, from_workflow: Optional[str] = None) -> None:
        if self._active is None:
            return
        record = self._active
        workflows = list(record.workflows_used)
        if to_workflow not in workflows:
            workflows.append(to_workflow)
        chain = list(record.handoff_chain)
        prev = from_workflow or (workflows[-2] if len(workflows) > 1 else workflows[0])
        chain.append(f"{prev}->{to_workflow}")
        self._active = CurationSessionRecord(
            schema_version=record.schema_version,
            source_artifact=record.source_artifact,
            workflows_used=workflows,
            handoff_chain=chain,
            accept_events=list(record.accept_events),
            reject_events=list(record.reject_events),
            final_label_id_map=dict(record.final_label_id_map),
        )

    def record_accept(self, *, slice_label: str, label_id: int, voxel_count: int) -> None:
        if self._active is None:
            return
        record = self._active
        events = list(record.accept_events)
        events.append(
            {
                "timestamp": _utcnow_iso(),
                "slice_label": slice_label,
                "label_id": int(label_id),
                "voxel_count": int(voxel_count),
            }
        )
        self._active = CurationSessionRecord(
            schema_version=record.schema_version,
            source_artifact=record.source_artifact,
            workflows_used=list(record.workflows_used),
            handoff_chain=list(record.handoff_chain),
            accept_events=events,
            reject_events=list(record.reject_events),
            final_label_id_map=dict(record.final_label_id_map),
        )

    def record_reject(self, *, slice_label: str, reason: str) -> None:
        if self._active is None:
            return
        record = self._active
        events = list(record.reject_events)
        events.append(
            {
                "timestamp": _utcnow_iso(),
                "slice_label": slice_label,
                "reason": reason,
            }
        )
        self._active = CurationSessionRecord(
            schema_version=record.schema_version,
            source_artifact=record.source_artifact,
            workflows_used=list(record.workflows_used),
            handoff_chain=list(record.handoff_chain),
            accept_events=list(record.accept_events),
            reject_events=events,
            final_label_id_map=dict(record.final_label_id_map),
        )

    def finalize(self, *, final_label_id_map: Dict[str, int]) -> Optional[CurationSessionRecord]:
        if self._active is None:
            return None
        record = self._active
        self._active = CurationSessionRecord(
            schema_version=record.schema_version,
            source_artifact=record.source_artifact,
            workflows_used=list(record.workflows_used),
            handoff_chain=list(record.handoff_chain),
            accept_events=list(record.accept_events),
            reject_events=list(record.reject_events),
            final_label_id_map={str(k): int(v) for k, v in dict(final_label_id_map).items()},
        )
        return self._active

    def persist_active(self, working_dir: str) -> Optional[str]:
        if self._active is None:
            return None
        path = self.artifact_registry.save_curation_session(working_dir, self._active)
        return str(path)

    def snapshot(self) -> Optional[Dict[str, Any]]:
        if self._active is None:
            return None
        return deepcopy(self._active.to_dict())
