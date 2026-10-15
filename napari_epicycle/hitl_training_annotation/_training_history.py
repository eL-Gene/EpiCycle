"""Phase A — Training history reader for the Step 2 history dialog.

Walks ``runs/{run_id}/{bucket}/round_*/model/manifest.json`` (schema 2.1)
and returns structured per-round data for visualization.

Pure Python, no Qt deps.  Defensively parses each manifest — missing
optional fields degrade to safe defaults; corrupted JSON is logged and
skipped without crashing the scan.

"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


VALID_BUCKETS = ("lateral", "axial")

# Match `round_NN` directory names where NN is a non-negative integer.
_ROUND_DIR_RE = re.compile(r"^round_(\d+)$")


@dataclass
class RoundHistoryEntry:
    """One round's training entry, parsed from its manifest.json.

    Fields default to safe values when the underlying manifest doesn't
    have them — common for partial schema 2.0 manifests or runs that
    crashed before the schema 2.1 fields were populated.

    Note: the writer (``packages/epicycle_training/step2_model_manifest.py``)
    stores ``channels_used`` as a list of channel NAME strings (e.g.
    ``["membrane", "nuclei"]``), not integer indices.  And it stores
    ``n_epochs`` (not ``epochs``).  Learning rate is not currently
    written to the manifest — it's a known gap (follow-up: extend
    writer to capture ``learning_rate`` too).
    """
    bucket: str
    round_idx: int
    init_mode: str = "scratch"
    init_path: Optional[str] = None
    init_combo_text: str = ""
    channels_used: List[str] = field(default_factory=list)
    n_train_pairs: int = 0
    train_pair_labels: List[str] = field(default_factory=list)
    n_epochs: int = 0
    cellpose_version: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    weights_filename: str = ""
    manifest_path: str = ""


def _step2_runs_root(workspace: Union[Path, str]) -> Path:
    """Resolve `{workspace}/artifacts/step2/runs/` without depending on
    `_paths` (avoid circular-import potential — this module is read-only)."""
    return Path(workspace) / "artifacts" / "step2" / "runs"


def _parse_manifest_to_entry(
    manifest_path: Path,
    *,
    bucket: str,
    round_idx: int,
) -> Optional[RoundHistoryEntry]:
    """Read + parse one manifest.json into a RoundHistoryEntry.

    Returns None on failure (corrupted JSON, missing file, etc.) so
    callers can silently skip the round.  Failure is printed to stdout
    for diagnostics.
    """
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[HISTORY-READ] skipping corrupted manifest "
              f"{manifest_path}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(data, dict):
        print(f"[HISTORY-READ] skipping malformed manifest "
              f"(not a dict): {manifest_path}")
        return None

    init = data.get("init", {})
    if not isinstance(init, dict):
        init = {}
    training = data.get("training", {})
    if not isinstance(training, dict):
        training = {}
    env = data.get("environment", {})
    if not isinstance(env, dict):
        env = {}
    timing = data.get("timing", {})
    if not isinstance(timing, dict):
        timing = {}

    # Defensive type coercion — manifests may carry odd types from
    # legacy versions.  Each .get is wrapped in a forgiving cast.
    def _as_int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _as_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _as_str(v, default=""):
        if v is None:
            return default
        return str(v)

    def _as_list(v, default=None):
        if isinstance(v, list):
            return list(v)
        return default if default is not None else []

    return RoundHistoryEntry(
        bucket=bucket,
        round_idx=round_idx,
        init_mode=_as_str(init.get("mode"), "scratch"),
        init_path=init.get("path") if init.get("path") else None,
        init_combo_text=_as_str(init.get("init_combo_text")),
        # Writer stores List[str] of channel names (e.g. ["membrane", "nuclei"]),
        # NOT integer indices — see step2_model_manifest.py:110, 320.
        channels_used=_as_list(training.get("channels_used")),
        n_train_pairs=_as_int(training.get("n_train_pairs")),
        train_pair_labels=_as_list(training.get("train_pair_labels")),
        # Audit HIGH fix 2026-05-27: writer uses key "n_epochs" not "epochs"
        # (step2_model_manifest.py:108, 318).  Reading "epochs" silently
        # returned 0.  Now reads the correct key with a defensive fallback
        # to the old name in case legacy manifests ever existed.
        n_epochs=_as_int(training.get("n_epochs", training.get("epochs"))),
        cellpose_version=_as_str(env.get("cellpose_version")),
        started_at=_as_str(timing.get("started_at")),
        finished_at=_as_str(timing.get("finished_at")),
        duration_seconds=_as_float(timing.get("duration_seconds")),
        weights_filename=_as_str(data.get("weights_filename")),
        manifest_path=str(manifest_path.resolve()),
    )


def read_history_for_bucket(
    workspace: Union[Path, str],
    run_id: str,
    bucket: str,
) -> List[RoundHistoryEntry]:
    """Read all per-round manifests under ``runs/{run_id}/{bucket}/round_*/``
    and return a list of ``RoundHistoryEntry`` sorted by round_idx ascending.

    Defensive against:
    * Missing workspace / run / bucket dirs → empty list
    * Invalid bucket name → empty list
    * Round dirs with no manifest.json → skipped silently
    * Corrupted manifest JSON → skipped + logged
    * Partial schema 2.0 manifests → parsed with defaults

    Returns empty list on any error; never raises.
    """
    if bucket not in VALID_BUCKETS:
        return []
    bucket_root = _step2_runs_root(workspace) / run_id / bucket
    if not bucket_root.is_dir():
        return []

    entries: List[RoundHistoryEntry] = []
    for child in bucket_root.iterdir():
        if not child.is_dir():
            continue
        m = _ROUND_DIR_RE.match(child.name)
        if m is None:
            continue
        round_idx = int(m.group(1))
        manifest_path = child / "model" / "manifest.json"
        entry = _parse_manifest_to_entry(
            manifest_path, bucket=bucket, round_idx=round_idx,
        )
        if entry is not None:
            entries.append(entry)

    # Sort by round_idx ascending (filesystem iter order is not guaranteed).
    entries.sort(key=lambda e: e.round_idx)
    return entries


def read_full_history(
    workspace: Union[Path, str],
    run_id: str,
) -> Dict[str, List[RoundHistoryEntry]]:
    """Read both buckets' histories for a run.

    Returns:
        ``{"lateral": [...], "axial": [...]}`` — always both keys present,
        empty list when no rounds.
    """
    return {
        bucket: read_history_for_bucket(workspace, run_id, bucket)
        for bucket in VALID_BUCKETS
    }


# ---------------------------------------------------------------------------
# Visualization helpers (used by Phase C timeline widget but pure-Python
# enough to keep here so they can be unit-tested without Qt).
# ---------------------------------------------------------------------------

def has_continuous_lineage_from_prior(
    entries: List[RoundHistoryEntry],
) -> List[bool]:
    """For each round, return whether the link from the prior round is
    a CONTINUOUS model lineage.

    A continuous link means the user trained Cumulative mode + the
    prior round's checkpoint was inherited.  Broken links happen when
    ``init.mode != "inherited"`` (scratch, pretrained, browse, etc.).

    Index 0 is always False (no incoming link — start of lineage).
    Index i (i>0) is True iff entries[i].init_mode == "inherited".
    """
    return [
        i > 0 and entries[i].init_mode == "inherited"
        for i in range(len(entries))
    ]
