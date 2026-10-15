from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .contracts import RoundMetadata, SliceAnnotationRecord

SCHEMA_VERSION = "1.0"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_round_id(payload: Dict[str, Any]) -> str:
    if "round_id" in payload:
        return str(payload["round_id"])
    if "round" in payload:
        return f"round_{int(payload['round']):02d}"
    raise ValueError("Legacy payload must contain `round_id` or `round`")


def _normalize_sample(sample: Dict[str, Any], round_id: str) -> SliceAnnotationRecord:
    img_uri = sample.get("img_uri", sample.get("img_path"))
    mask_uri = sample.get("mask_uri", sample.get("mask_path"))
    if not img_uri or not mask_uri:
        raise ValueError("Sample must contain image and mask uri/path")

    return SliceAnnotationRecord(
        schema_version=SCHEMA_VERSION,
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


def round_metadata_from_legacy(payload: Dict[str, Any]) -> RoundMetadata:
    """
    Convert legacy round metadata shape to the v1 contract.

    Legacy examples observed:
      - {"round": 1, "date": "...", "samples": [{"img_path": "...", "mask_path": "..."}]}
      - {"round_id": "round_01", "samples": [{"img_uri": "...", "mask_uri": "..."}]}
    """

    round_id = _normalize_round_id(payload)
    created_at = str(payload.get("created_at", payload.get("date", _utcnow_iso())))
    raw_samples: List[Dict[str, Any]] = list(payload.get("samples", []))
    samples = [_normalize_sample(sample, round_id) for sample in raw_samples]

    config_snapshot = dict(payload.get("config_snapshot", {}))
    if not config_snapshot:
        config_snapshot = {
            "mode": payload.get("mode"),
            "channels": payload.get("channels"),
            "normalize": payload.get("normalize"),
            "cellprob_threshold": payload.get("cellprob_threshold"),
            "flow_threshold": payload.get("flow_threshold"),
            "training_config": payload.get("training_config"),
        }

    return RoundMetadata(
        schema_version=SCHEMA_VERSION,
        round_id=round_id,
        created_at=created_at,
        config_snapshot=config_snapshot,
        parent_round=payload.get("parent_round"),
        samples=samples,
    )
