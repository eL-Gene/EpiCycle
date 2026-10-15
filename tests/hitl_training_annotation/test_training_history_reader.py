"""Tests for the Phase A manifest-history reader.

Phase A of the training-history plan is a pure-Python module that walks
``runs/{run_id}/{bucket}/round_*/model/manifest.json`` and returns a
structured per-bucket list of ``RoundHistoryEntry`` dataclasses.
No Qt dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# Module under test (doesn't exist yet — tests will fail with ImportError
# until Phase A implementation lands).
def _import_module():
    from napari_epicycle.hitl_training_annotation._training_history import (
        RoundHistoryEntry, read_history_for_bucket, read_full_history,
    )
    return RoundHistoryEntry, read_history_for_bucket, read_full_history


# ============================================================================
# Test fixtures: build a fake workspace tree.
# ============================================================================

def _make_manifest(
    round_dir: Path,
    *,
    run_id: str,
    bucket: str,
    round_idx: int,
    init_mode: str = "scratch",
    init_path: str | None = None,
    init_combo_text: str = "Cumulative — reuse prior model (default)",
    channels: list = None,
    n_pairs: int = 4,
    pair_labels: list = None,
    epochs: int = 200,
    lr: float = 0.01,
    cellpose_version: str = "3.1.1.2",
    started_at: str = "2026-05-27T10:00:00+00:00",
    finished_at: str = "2026-05-27T10:01:30+00:00",
    duration_seconds: float = 90.0,
) -> Path:
    """Write a minimal schema 2.1 manifest.json into round_dir/model/."""
    model_dir = round_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_filename = f"{run_id}_{bucket}_round_{round_idx:02d}"
    manifest = {
        "schema_version": "2.1",
        "run_id": run_id,
        "bucket": bucket,
        "round_number": round_idx,
        "weights_filename": weights_filename,
        "init": {
            "mode": init_mode,
            "path": init_path,
            "init_combo_text": init_combo_text,
        },
        "training": {
            # Match the ACTUAL writer field names (n_epochs not epochs)
            # per packages/epicycle_training/step2_model_manifest.py:108,
            # 318.  Writer stores channels_used as List[str] (channel
            # names like "membrane"), not integer indices.
            "n_epochs": epochs,
            "channels_used": channels if channels is not None else ["membrane", "none"],
            "n_train_pairs": n_pairs,
            "train_pair_labels": pair_labels if pair_labels is not None
                                 else [f"stk_XY_z{i:04d}" for i in range(n_pairs)],
        },
        "environment": {
            "cellpose_version": cellpose_version,
            "torch_version": "2.0.0",
            "python_version": "3.10.18",
            "device": "NVIDIA GPU",
            "platform": "Windows-10",
        },
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
        },
    }
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _make_run_tree(tmp_path: Path, run_id: str, rounds: dict) -> Path:
    """Build `runs/{run_id}/{bucket}/round_NN/model/manifest.json` tree.

    Args:
        rounds: dict[bucket, list[dict_of_round_overrides]]
                e.g. {"lateral": [{"init_mode": "scratch"},
                                   {"init_mode": "inherited", "init_path": "..."}],
                      "axial": []}
    """
    runs_root = tmp_path / "artifacts" / "step2" / "runs" / run_id
    for bucket, round_specs in rounds.items():
        for idx, overrides in enumerate(round_specs):
            round_dir = runs_root / bucket / f"round_{idx:02d}"
            _make_manifest(
                round_dir,
                run_id=run_id, bucket=bucket, round_idx=idx,
                **overrides,
            )
    return tmp_path  # workspace root


# ============================================================================
# Tests
# ============================================================================

def test_read_empty_workspace_returns_empty_dict(tmp_path):
    """No runs/ tree at all → returns empty per-bucket lists, no crash."""
    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    assert result == {"lateral": [], "axial": []}


def test_read_three_lateral_one_axial_round(tmp_path):
    """A typical workspace: 3 lateral rounds + 1 axial round.
    Verify counts + ordering."""
    _import_module()
    _make_run_tree(tmp_path, "nuclei", {
        "lateral": [
            {"init_mode": "scratch"},
            {"init_mode": "inherited", "init_path": "round_00 ckpt"},
            {"init_mode": "inherited", "init_path": "round_01 ckpt"},
        ],
        "axial": [
            {"init_mode": "scratch"},
        ],
    })
    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    assert len(result["lateral"]) == 3
    assert len(result["axial"]) == 1
    # Verify ordering: round_idx 0, 1, 2
    assert [e.round_idx for e in result["lateral"]] == [0, 1, 2]
    assert [e.round_idx for e in result["axial"]] == [0]


def test_manifest_missing_fields_degrades_gracefully(tmp_path):
    """A manifest missing optional fields (e.g., timing.duration_seconds)
    should not crash — defaults fill in."""
    _import_module()
    runs_root = tmp_path / "artifacts" / "step2" / "runs" / "nuclei"
    round_dir = runs_root / "lateral" / "round_00" / "model"
    round_dir.mkdir(parents=True, exist_ok=True)
    # Write a manifest missing `timing`, `environment`, partial `training`.
    # Use the ACTUAL writer field names — n_epochs not epochs.
    minimal = {
        "schema_version": "2.1",
        "run_id": "nuclei",
        "bucket": "lateral",
        "round_number": 0,
        "weights_filename": "nuclei_lateral_round_00",
        "init": {"mode": "scratch"},
        "training": {"n_epochs": 100},
        # no environment, no timing
    }
    (round_dir / "manifest.json").write_text(json.dumps(minimal), encoding="utf-8")

    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    assert len(result["lateral"]) == 1
    entry = result["lateral"][0]
    assert entry.n_epochs == 100
    assert entry.init_mode == "scratch"
    # Missing fields → safe defaults
    assert entry.n_train_pairs == 0
    assert entry.channels_used == []
    assert entry.duration_seconds == 0.0
    assert entry.started_at == ""
    assert entry.finished_at == ""


def test_corrupted_manifest_skipped_not_crashed(tmp_path):
    """A corrupted JSON file should be silently skipped (logged)."""
    _import_module()
    runs_root = tmp_path / "artifacts" / "step2" / "runs" / "nuclei"
    round_dir = runs_root / "lateral" / "round_00" / "model"
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")
    # Also add a valid round_01 to confirm scanning continues past the error
    _make_manifest(runs_root / "lateral" / "round_01",
                   run_id="nuclei", bucket="lateral", round_idx=1)

    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    # round_00 dropped (corrupted), round_01 kept
    assert [e.round_idx for e in result["lateral"]] == [1]


def test_round_dir_without_manifest_skipped(tmp_path):
    """A round_NN dir without model/manifest.json is silently skipped.
    Happens when training crashed before manifest write."""
    _import_module()
    runs_root = tmp_path / "artifacts" / "step2" / "runs" / "nuclei"
    # round_00 has manifest, round_01 doesn't.
    _make_manifest(runs_root / "lateral" / "round_00",
                   run_id="nuclei", bucket="lateral", round_idx=0)
    (runs_root / "lateral" / "round_01" / "model").mkdir(parents=True)
    # No manifest.json written for round_01.

    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    assert [e.round_idx for e in result["lateral"]] == [0]


def test_init_mode_extracted_for_all_six_types(tmp_path):
    """All 6 init modes (scratch, inherited, pretrained, browse,
    step3-refined, run-model) should be parsed verbatim."""
    _import_module()
    runs_root = tmp_path / "artifacts" / "step2" / "runs" / "nuclei"
    for idx, mode in enumerate([
        "scratch", "inherited", "pretrained",
        "browse", "step3-refined", "run-model",
    ]):
        _make_manifest(
            runs_root / "lateral" / f"round_{idx:02d}",
            run_id="nuclei", bucket="lateral", round_idx=idx,
            init_mode=mode, init_path=f"/some/path/for/{mode}",
        )

    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    modes = [e.init_mode for e in result["lateral"]]
    assert modes == ["scratch", "inherited", "pretrained",
                     "browse", "step3-refined", "run-model"]


def test_lineage_break_detection(tmp_path):
    """A round with init.mode != "inherited" represents a broken model
    lineage (user retrained from scratch / new base).  Reader should
    expose enough info for the visualization to detect this."""
    _import_module()
    runs_root = tmp_path / "artifacts" / "step2" / "runs" / "nuclei"
    # Lineage: 00 (scratch) → 01 (inherited) → 02 (pretrained — BROKEN)
    #          → 03 (inherited from cyto3) → 04 (inherited)
    _make_manifest(runs_root / "lateral" / "round_00",
                   run_id="nuclei", bucket="lateral", round_idx=0,
                   init_mode="scratch")
    _make_manifest(runs_root / "lateral" / "round_01",
                   run_id="nuclei", bucket="lateral", round_idx=1,
                   init_mode="inherited", init_path="...round_00 ckpt")
    _make_manifest(runs_root / "lateral" / "round_02",
                   run_id="nuclei", bucket="lateral", round_idx=2,
                   init_mode="pretrained", init_path="cyto3")
    _make_manifest(runs_root / "lateral" / "round_03",
                   run_id="nuclei", bucket="lateral", round_idx=3,
                   init_mode="inherited", init_path="...round_02 ckpt")
    _make_manifest(runs_root / "lateral" / "round_04",
                   run_id="nuclei", bucket="lateral", round_idx=4,
                   init_mode="inherited", init_path="...round_03 ckpt")

    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    entries = result["lateral"]
    # Visualization rule: link from round N-1 → round N is BROKEN
    # when entries[N].init_mode != "inherited".
    # round_00: first round, no incoming link (start of lineage)
    # round_01: inherited → link to round_00 is CONTINUOUS
    # round_02: pretrained → BROKEN link to round_01 (lineage restart)
    # round_03: inherited → CONTINUOUS link to round_02
    # round_04: inherited → CONTINUOUS link to round_03
    has_continuous_link_from_prior = [
        i > 0 and entries[i].init_mode == "inherited"
        for i in range(len(entries))
    ]
    assert has_continuous_link_from_prior == [False, True, False, True, True]


def test_read_history_for_single_bucket(tmp_path):
    """read_history_for_bucket should return just one bucket's list."""
    _import_module()
    _make_run_tree(tmp_path, "nuclei", {
        "lateral": [{"init_mode": "scratch"}, {"init_mode": "inherited"}],
        "axial":   [{"init_mode": "scratch"}],
    })
    _, read_history_for_bucket, _ = _import_module()
    lat = read_history_for_bucket(tmp_path, "nuclei", "lateral")
    ax = read_history_for_bucket(tmp_path, "nuclei", "axial")
    assert len(lat) == 2
    assert len(ax) == 1
    assert lat[0].bucket == "lateral"
    assert ax[0].bucket == "axial"


def test_invalid_bucket_returns_empty(tmp_path):
    """Asking for an invalid bucket name returns empty list, no crash."""
    _import_module()
    _, read_history_for_bucket, _ = _import_module()
    result = read_history_for_bucket(tmp_path, "nuclei", "not-a-real-bucket")
    assert result == []


def test_manifest_path_is_absolute_in_entry(tmp_path):
    """RoundHistoryEntry.manifest_path is the absolute path users can
    `Open in folder` from the UI."""
    _import_module()
    _make_run_tree(tmp_path, "nuclei", {
        "lateral": [{"init_mode": "scratch"}],
    })
    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    entry = result["lateral"][0]
    assert Path(entry.manifest_path).is_absolute()
    assert Path(entry.manifest_path).exists()
    assert entry.manifest_path.endswith("manifest.json")


def test_entries_sorted_by_round_idx_ascending(tmp_path):
    """Reader must return entries sorted by round_idx ascending, not
    filesystem walk order (which is platform-dependent)."""
    _import_module()
    # Create rounds in non-sequential order (02, 00, 01) to verify sort.
    runs_root = tmp_path / "artifacts" / "step2" / "runs" / "nuclei"
    for idx in [2, 0, 1]:
        _make_manifest(
            runs_root / "lateral" / f"round_{idx:02d}",
            run_id="nuclei", bucket="lateral", round_idx=idx,
        )

    _, _, read_full_history = _import_module()
    result = read_full_history(tmp_path, "nuclei")
    assert [e.round_idx for e in result["lateral"]] == [0, 1, 2]
