"""Forget a patch whose file is gone.

Pure and Qt-free.  :func:`forget_items` lists, as found on disk, the ACTIVE
run's Step 3 artifacts of one patch; :func:`move_to_trash` MOVES them -- a
move, never a delete -- to ``artifacts/step3/.trash/{patch}-{stamp}/``, each
at its path relative to ``artifacts/step3/`` (Step 2's run-delete pattern,
``hitl_training_annotation/_run_management.delete_run_from_disk``), writing
``forgotten.json`` FIRST and updating it after every item, so a failure part
of the way leaves a complete list of what moved and where.

The durable trees stay: ``snapshots/`` and ``training_archive/``; so do
other runs' files and records.  ``.trash/`` sits beside ``runs/``, which no
run scan reads.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import _paths
from . import _qc_extraction_context as _qx
from . import _step3_manifests as _s3m
from ._reshape_flags import sidecar_path as _reshape_sidecar

TRASH_DIRNAME = ".trash"
LOG_NAME = "forgotten.json"
#: the order the confirm names the kinds (``_step3_pr_text.S3_FORGET_WHAT``)
KINDS = ("recon", "features", "curation", "predictions", "pairs")
VIEWS = ("xy", "xz", "yz")
ARRAYS = ("flows", "probs", "masks")


@dataclass
class ForgetResult:
    """What a move did: *moved* (original, trashed) pairs, the trash
    folder, and the error that stopped it (``None`` when every item
    moved)."""
    trash: Path
    moved: List[Tuple[str, str]] = field(default_factory=list)
    total: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.moved) == self.total


def trash_root(workspace) -> Path:
    """``{ws}/artifacts/step3/.trash/``."""
    return _paths.step3_root(workspace) / TRASH_DIRNAME


def forget_items(workspace, patch_name: str,
                 run_id: Optional[str] = None) -> List[Tuple[str, Path]]:
    """``[(kind, path)]`` of the patch's artifacts in run *run_id* that are
    on disk now, in ``KINDS`` order.  The prediction cache is listed file by
    file by its exact names (a ``{stem}_*`` glob would also take another
    patch's files whose name starts with this stem)."""
    ws = workspace
    stem = Path(patch_name).stem
    cands: List[Tuple[str, Path]] = [
        ("recon", _paths.patch_run_dir(ws, patch_name, run_id)),
        ("features", _paths.qc_features_csv(ws, stem, run_id)),
        # the per-patch QC sidecar ``<stem>_extraction_context.json`` (R7a)
        ("features", _qx._sidecar_path(
            _paths.qc_features_csv(ws, stem, run_id))),
        ("features", _paths.qc_features_raw_csv(ws, stem, run_id)),
        ("features", _paths.qc_adjacency_raw_csv(ws, stem, run_id)),
        ("features", _paths.qc_adjacency_curated_csv(ws, stem, run_id)),
        ("features", _reshape_sidecar(ws, patch_name, run_id)),
        ("curation", _paths.patch_curation_dir(ws, stem, run_id)),
    ]
    cache = _paths.prediction_cache_dir(ws, run_id)
    for v in VIEWS:
        for a in ARRAYS:
            cands.append(("predictions", cache / f"{stem}_{v}_{a}.npy"))
    cands.append(("predictions", _s3m.prediction_manifest_path(cache,
                                                               patch_name)))
    cands.append(("pairs", _paths.patch_pairs_dir(ws, stem, run_id)))
    seen = set()
    out = []
    for kind, path in cands:
        p = Path(path)
        if p in seen or not p.exists():
            continue
        seen.add(p)
        out.append((kind, p))
    return out


def kinds_present(items) -> List[str]:
    """The kinds of *items*, once each, in ``KINDS`` order."""
    have = {k for k, _p in items}
    return [k for k in KINDS if k in have]


def new_trash_dir(workspace, patch_name: str, now: Optional[datetime] = None
                  ) -> Path:
    """``.trash/{patch}-{YYYYmmdd-HHMMSS}`` (``-2``, ``-3`` on a clash);
    not created."""
    root = trash_root(workspace)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    dest = root / f"{patch_name}-{stamp}"
    n = 1
    while dest.exists():
        n += 1
        dest = root / f"{patch_name}-{stamp}-{n}"
    return dest


def _write_log(path: Path, payload: Dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".forgotten-",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def move_to_trash(workspace, patch_name: str, run_id: Optional[str],
                  items, record: Optional[dict] = None, *,
                  run_label: str = "", now: Optional[datetime] = None,
                  mover=shutil.move) -> ForgetResult:
    """Move *items* (``forget_items``' list) into a new trash folder.

    ``forgotten.json`` (patch, run, time, the record, every original ->
    trashed path, and ``moved`` per item) is written BEFORE the first move
    and after each one.  The first failure stops the move: the items
    already moved stay in the trash (the log says where), the rest stay
    where they were, and the result carries the error.  Never raises for a
    failed item; raises only if the log itself cannot be written."""
    root = _paths.step3_root(workspace)
    dest = new_trash_dir(workspace, patch_name, now)
    dest.mkdir(parents=True, exist_ok=False)
    entries = []
    for kind, path in items:
        p = Path(path)
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = Path(p.name)
        entries.append({"kind": kind, "original": str(p),
                        "trashed": str(dest / rel), "moved": False})
    log = {
        "patch": patch_name,
        "run": run_label or (run_id or "default"),
        "forgotten_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "record": record or {},
        "items": entries,
    }
    log_path = dest / LOG_NAME
    _write_log(log_path, log)
    result = ForgetResult(trash=dest, total=len(entries))
    for entry in entries:
        target = Path(entry["trashed"])
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            mover(entry["original"], str(target))
        except Exception as exc:      # noqa: BLE001 -- reported, not raised
            result.error = f"{entry['original']}: {exc!r}"
            log["stopped"] = result.error
            _write_log(log_path, log)
            return result
        entry["moved"] = True
        result.moved.append((entry["original"], entry["trashed"]))
        _write_log(log_path, log)
    return result
