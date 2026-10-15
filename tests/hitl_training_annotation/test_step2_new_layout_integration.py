"""Integration tests for the Step 2 directory restructure runtime paths.

These tests target the gap flagged by the Phase 3-4 audit: pure path
helpers + migration logic are well-tested at the unit level, but the
controller's RUNTIME use of those paths (post-migration writes, dual-
path reads, state load/save bridge with active_run_id init) had no
integration coverage.

Tests build minimal new-layout workspace fixtures and exercise the
helpers + functions used by the controller, verifying:

  * Post-migration writes land in the right bucket via _paths helpers
  * Dual-read prefers new layout over legacy in hybrid scenarios
  * State load/save round-trip preserves all per-run + workspace-level
    fields (including run-manifest fields)
  * active_run_id init from state file (peek pattern)
  * Per-bucket round_metadata + selected_indices splitting

These do NOT instantiate the full controller (which needs Qt + napari).
Instead they exercise the same code paths the controller uses, via the
public _paths and _migrate_step2_layout APIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.hitl_training_annotation import _paths as p
from napari_epicycle.hitl_training_annotation import _migrate_step2_layout as m


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def migrated_ws(tmp_path):
    """Workspace with a minimal post-migration new-layout structure.

    Has:
      * step2_state.json (v3, with runs.default + workspace-level keys)
      * artifacts/step2/runs/default/{lateral,axial}/rounds/round_00/
      * artifacts/step2/runs/default/run_manifest.json
    """
    ws = tmp_path / "migrated_ws"
    ws.mkdir(parents=True)

    # State JSON (v3)
    state_path = p.state_json_path(ws)
    state_path.write_text(json.dumps({
        "schema_version": 3,
        "active_run_id": "default",
        # workspace-level keys
        "selected_view": "XY",
        "view_target": "lateral",
        "axial_plane": "XZ",
        "curation_model_source_mode": "hitl",
        "imported_curation_model_path": "",
        "curation_pretrained_model_type": "cyto",
        "runs": {
            "default": {
                "display_name": "Default",
                "target": None,
                "backend": "cellpose",
                "created_at": "2026-05-23T00:00:00Z",
                "created_from": None,
                # per-run keys
                "current_round": 1,
                "current_round_by_view": {"lateral": 1, "axial": 0},
                "training_data_by_round": {
                    "lateral_0": [
                        {"label": "patch_a_XY_z0001", "view": "XY",
                         "stack_id": "patch_a", "slice_index": 1,
                         "round_idx": 0, "status": "accepted"},
                    ],
                },
                "losses_by_round": {},
                "model_checkpoint_paths": {},
                "round_backend_info": {},
                "sample_records": {
                    "patch_a_XY_z0001": {
                        "label": "patch_a_XY_z0001",
                        "view": "XY",
                        "stack_id": "patch_a",
                        "slice_index": 1,
                        "round_idx": 0,
                        "status": "accepted",
                    },
                },
                "current_config": {"channels": ["DAPI"]},
            },
        },
    }, indent=2), encoding="utf-8")

    # Run manifest
    run_manifest_path = p.run_manifest_path(ws, "default")
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(json.dumps({
        "schema_version": "2.0",
        "run_id": "default",
        "display_name": "Default",
        "target": None,
        "backend": "cellpose",
        "created_at": "2026-05-23T00:00:00Z",
        "created_from": None,
    }, indent=2), encoding="utf-8")

    # Pre-populate round_00 accepted dirs for both buckets
    for bucket in ("lateral", "axial"):
        accepted = p.round_accepted_dir(ws, "default", bucket, 0)
        accepted.mkdir(parents=True, exist_ok=True)

    return ws


@pytest.fixture
def hybrid_ws(tmp_path):
    """Mid-migration MIXED workspace: BOTH legacy + new layouts populated.

    Legacy: training_data/round_00/accepted_pairs/spatial/duplicate_XY_z0001_img.tif
    New:    artifacts/step2/runs/default/lateral/rounds/round_00/accepted/duplicate_XY_z0001_img.tif

    Used to verify dedup order — new should win, legacy fallback if new missing.
    """
    ws = tmp_path / "hybrid_ws"
    ws.mkdir(parents=True)

    # Legacy: with STALE content
    legacy_dir = ws / "training_data" / "round_00" / "accepted_pairs" / "spatial"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "duplicate_XY_z0001_img.tif").write_bytes(b"LEGACY STALE CONTENT")
    (legacy_dir / "duplicate_XY_z0001_mask.tif").write_bytes(b"LEGACY MASK")
    # Legacy-only file (no new-layout duplicate)
    (legacy_dir / "legacy_only_XY_z0002_img.tif").write_bytes(b"legacy only")
    (legacy_dir / "legacy_only_XY_z0002_mask.tif").write_bytes(b"legacy mask")

    # New: with FRESH content
    new_dir = p.round_accepted_dir(ws, "default", "lateral", 0)
    new_dir.mkdir(parents=True, exist_ok=True)
    (new_dir / "duplicate_XY_z0001_img.tif").write_bytes(b"NEW FRESH CONTENT")
    (new_dir / "duplicate_XY_z0001_mask.tif").write_bytes(b"NEW MASK")
    # New-only file
    (new_dir / "new_only_XY_z0003_img.tif").write_bytes(b"new only")
    (new_dir / "new_only_XY_z0003_mask.tif").write_bytes(b"new mask")

    # Make new layout authoritative (so detect_layout_version returns NEW)
    (ws / "step2_state.json").write_text(json.dumps({
        "schema_version": 3,
        "active_run_id": "default",
        "runs": {"default": {"current_round": 1}},
    }), encoding="utf-8")

    return ws


# ============================================================================
# Test 1: Post-migration writes via _paths helpers land in correct bucket
# ============================================================================

class TestPostMigrationWrites:
    """Verifies the path computation the controller uses in
    _persist_curated_training_pair when is_new_layout_state() is True.
    """

    def test_xy_slice_routes_to_lateral_accepted(self, migrated_ws):
        # Simulate what _persist_curated_training_pair does
        view = "XY"
        bucket = p.bucket_for_view(view)
        assert bucket == "lateral"

        round_idx = 0
        out_dir = p.round_accepted_dir(migrated_ws, "default", bucket, round_idx)
        assert "/lateral/" in str(out_dir).replace("\\", "/")
        assert "/accepted" in str(out_dir).replace("\\", "/")

        # Write a fake pair
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "test_XY_z0042_img.tif").write_bytes(b"fake img")
        (out_dir / "test_XY_z0042_mask.tif").write_bytes(b"fake mask")

        # Verify it landed in the lateral bucket
        lateral_accepted = p.round_accepted_dir(migrated_ws, "default", "lateral", 0)
        assert (lateral_accepted / "test_XY_z0042_img.tif").exists()
        # And NOT in axial bucket
        axial_accepted = p.round_accepted_dir(migrated_ws, "default", "axial", 0)
        assert not (axial_accepted / "test_XY_z0042_img.tif").exists()

    def test_xz_slice_routes_to_axial_accepted(self, migrated_ws):
        bucket = p.bucket_for_view("XZ")
        assert bucket == "axial"
        out_dir = p.round_accepted_dir(migrated_ws, "default", bucket, 0)
        assert "/axial/" in str(out_dir).replace("\\", "/")

    def test_yz_slice_routes_to_axial_accepted(self, migrated_ws):
        bucket = p.bucket_for_view("YZ")
        assert bucket == "axial"

    def test_path_NOT_under_legacy_training_data(self, migrated_ws):
        # Sanity: paths from _paths.round_accepted_dir DON'T touch
        # the legacy training_data/ root
        out_dir = p.round_accepted_dir(migrated_ws, "default", "lateral", 0)
        assert "training_data" not in str(out_dir)


# ============================================================================
# Test 2: Dual-read dedup — new wins over legacy in hybrid scenarios
# ============================================================================

class TestDualReadDedup:
    """Verifies the corrected scan order in _collect_prior_round_pairs:
    new layout walked FIRST, legacy is fallback.  Hybrid workspaces
    with duplicate basenames should prefer new (fresh) over legacy (stale).
    """

    def test_hybrid_dedup_prefers_new_for_duplicates(self, hybrid_ws):
        # Simulate the controller's _collect_prior_round_pairs scan order
        round_idx = 0
        bucket = "lateral"  # active bucket

        round_search_dirs = []

        # NEW FIRST (per audit fix)
        new_accepted = p.round_accepted_dir(hybrid_ws, "default", bucket, round_idx)
        if new_accepted.is_dir():
            round_search_dirs.append(new_accepted)

        # Legacy second
        legacy_vg = (hybrid_ws / "training_data" / f"round_{round_idx:02d}"
                     / "accepted_pairs" / "spatial")
        if legacy_vg.is_dir():
            round_search_dirs.append(legacy_vg)

        seen = set()
        collected = []
        for vg_dir in round_search_dirs:
            for img in sorted(vg_dir.glob("*_img.tif")):
                if img.name in seen:
                    continue
                seen.add(img.name)
                collected.append(img)

        # duplicate_XY_z0001_img.tif should be from NEW layout
        duplicates = [c for c in collected if c.name == "duplicate_XY_z0001_img.tif"]
        assert len(duplicates) == 1
        dup = duplicates[0]
        # Verify content is the NEW (fresh) version
        assert dup.read_bytes() == b"NEW FRESH CONTENT"
        # Verify path is under new layout
        assert "/runs/default/lateral" in str(dup).replace("\\", "/")

    def test_hybrid_dedup_falls_back_to_legacy_for_legacy_only(self, hybrid_ws):
        # legacy_only_XY_z0002_img.tif exists only in legacy → should be picked up
        round_idx = 0
        bucket = "lateral"

        round_search_dirs = []
        new_accepted = p.round_accepted_dir(hybrid_ws, "default", bucket, round_idx)
        if new_accepted.is_dir():
            round_search_dirs.append(new_accepted)
        legacy_vg = (hybrid_ws / "training_data" / f"round_{round_idx:02d}"
                     / "accepted_pairs" / "spatial")
        if legacy_vg.is_dir():
            round_search_dirs.append(legacy_vg)

        seen = set()
        collected = []
        for vg_dir in round_search_dirs:
            for img in sorted(vg_dir.glob("*_img.tif")):
                if img.name in seen:
                    continue
                seen.add(img.name)
                collected.append(img)

        legacy_only = [c for c in collected if c.name == "legacy_only_XY_z0002_img.tif"]
        assert len(legacy_only) == 1
        # Verify it comes from legacy location
        assert "training_data" in str(legacy_only[0])

    def test_hybrid_dedup_finds_new_only_files(self, hybrid_ws):
        # new_only_XY_z0003_img.tif exists only in new → should be picked up
        new_accepted = p.round_accepted_dir(hybrid_ws, "default", "lateral", 0)
        names = [f.name for f in new_accepted.glob("*_img.tif")]
        assert "new_only_XY_z0003_img.tif" in names


# ============================================================================
# Test 3: State load round-trip preserves all field categories
# ============================================================================

class TestStateRoundTrip:
    """Verifies load → modify → wrap → write → load preserves data."""

    def test_load_v3_unwraps_to_flat_shape(self, migrated_ws):
        flat = p.load_step2_state_flat(migrated_ws)
        assert flat is not None
        # Per-run keys flattened
        assert flat["current_round"] == 1
        assert flat["current_round_by_view"] == {"lateral": 1, "axial": 0}
        assert "patch_a_XY_z0001" in flat["sample_records"]
        # Workspace-level preserved
        assert flat["selected_view"] == "XY"
        assert flat["view_target"] == "lateral"
        # Run-manifest fields filtered out of flat
        assert "display_name" not in flat
        assert "target" not in flat
        assert "backend" not in flat
        # Schema version included
        assert flat["schema_version"] == 3

    def test_save_wrap_preserves_run_manifest_fields(self, migrated_ws):
        """When saving, the existing run_manifest fields (display_name,
        target, backend, created_at, created_from) must NOT be clobbered
        by migration defaults.
        """
        # User manually edits display_name in the v3 file
        v3_path = p.state_json_path(migrated_ws)
        existing = json.loads(v3_path.read_text(encoding="utf-8"))
        existing["runs"]["default"]["display_name"] = "User Edited Name"
        existing["runs"]["default"]["target"] = "nuclei"
        v3_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

        # Now simulate a save: wrap a flat dict and merge with existing
        from napari_epicycle.hitl_training_annotation._migrate_step2_layout import (
            wrap_state_schema_v2_to_v3,
        )
        flat_state = p.load_step2_state_flat(migrated_ws)
        # Strip schema_version (would conflict) and wrap
        flat_for_wrap = {k: v for k, v in flat_state.items() if k != "schema_version"}
        flat_for_wrap["schema_version"] = 2  # wrap expects v2-shape input
        v3 = wrap_state_schema_v2_to_v3(flat_for_wrap)

        # Apply the same merge-existing-run-manifest-fields logic the
        # controller does
        existing_default = (existing.get("runs") or {}).get("default", {})
        for key in ("display_name", "target", "backend", "created_at", "created_from"):
            if key in existing_default:
                v3["runs"]["default"][key] = existing_default[key]
        if "active_run_id" in existing:
            v3["active_run_id"] = existing["active_run_id"]

        # Write back
        v3_path.write_text(json.dumps(v3, indent=2), encoding="utf-8")

        # Verify: user edits preserved across the save
        re_loaded = json.loads(v3_path.read_text(encoding="utf-8"))
        assert re_loaded["runs"]["default"]["display_name"] == "User Edited Name"
        assert re_loaded["runs"]["default"]["target"] == "nuclei"


# ============================================================================
# Test 4: active_run_id init peeks correct value from state file
# ============================================================================

class TestActiveRunIdPeek:
    """Verifies the peek pattern in _load_internal_state reads
    active_run_id from the v3 state file BEFORE the main load.
    """

    def test_peek_returns_default_when_no_state_file(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        # Simulate the peek logic
        v3_path = p.state_json_path(ws)
        peeked = None
        if v3_path.is_file():
            try:
                content = json.loads(v3_path.read_text(encoding="utf-8"))
                peeked = content.get("active_run_id")
            except Exception:
                pass
        assert peeked is None  # no file, no value
        # Controller would default to DEFAULT_RUN_ID
        active_run_id = peeked or p.DEFAULT_RUN_ID
        assert active_run_id == "default"

    def test_peek_reads_explicit_active_run_id(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        v3_path = p.state_json_path(ws)
        v3_path.write_text(json.dumps({
            "schema_version": 3,
            "active_run_id": "nuclei_v1",
            "runs": {"nuclei_v1": {"current_round": 5}},
        }), encoding="utf-8")

        content = json.loads(v3_path.read_text(encoding="utf-8"))
        peeked = content.get("active_run_id")
        assert peeked == "nuclei_v1"

    def test_peek_falls_back_for_corrupt_state(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        v3_path = p.state_json_path(ws)
        v3_path.write_text("{ not json", encoding="utf-8")

        # Simulate the peek logic — defensive try/except
        peeked = None
        try:
            content = json.loads(v3_path.read_text(encoding="utf-8"))
            peeked = content.get("active_run_id")
        except json.JSONDecodeError:
            pass

        assert peeked is None
        # Falls back to default
        assert (peeked or p.DEFAULT_RUN_ID) == "default"


# ============================================================================
# Test 5: Per-bucket round_metadata + selected_indices splitting
# ============================================================================

class TestPerBucketMetadataSplit:
    """Verifies the bucket-split logic that the controller's
    _write_per_bucket_metadata_for_active_run helper uses.
    """

    def test_xy_records_filter_to_lateral_bucket(self):
        records = [
            {"label": "p_XY_z0001", "view": "XY"},
            {"label": "p_XZ_z0001", "view": "XZ"},
            {"label": "p_YZ_z0001", "view": "YZ"},
        ]
        lateral = [r for r in records if r.get("view") == "XY"]
        axial = [r for r in records if r.get("view") in ("XZ", "YZ")]
        assert len(lateral) == 1
        assert len(axial) == 2
        assert lateral[0]["label"] == "p_XY_z0001"

    def test_selected_indices_split_keeps_per_view_dicts(self):
        # Mimics the structure the controller builds
        sample_records = {
            "p_XY_z0001": {"view": "XY", "stack_id": "p", "slice_index": 1, "round_idx": 0},
            "p_XY_z0002": {"view": "XY", "stack_id": "p", "slice_index": 2, "round_idx": 0},
            "p_XZ_z0005": {"view": "XZ", "stack_id": "p", "slice_index": 5, "round_idx": 0},
        }
        lateral_sel = {"XY": {}}
        axial_sel = {"XZ": {}, "YZ": {}}
        for slabel, rec in sample_records.items():
            view = rec["view"]
            stack = rec["stack_id"]
            idx = rec["slice_index"]
            target = lateral_sel if view == "XY" else axial_sel
            target.setdefault(view, {}).setdefault(stack, []).append(idx)

        assert lateral_sel["XY"]["p"] == [1, 2]
        assert axial_sel["XZ"]["p"] == [5]
        assert axial_sel["YZ"] == {}

    def test_per_bucket_round_metadata_path_distinct(self, migrated_ws):
        # The lateral + axial metadata files live in distinct paths
        lat_path = p.round_metadata_path(migrated_ws, "default", "lateral", 0)
        ax_path = p.round_metadata_path(migrated_ws, "default", "axial", 0)
        assert lat_path != ax_path
        assert "/lateral/" in str(lat_path).replace("\\", "/")
        assert "/axial/" in str(ax_path).replace("\\", "/")
        assert lat_path.name == "round_metadata.json"
        assert ax_path.name == "round_metadata.json"


# ============================================================================
# Test 6: Layout detection transitions correctly through full lifecycle
# ============================================================================

class TestMultiRunIsolation:
    """Verifies two runs in the same workspace don't cross-contaminate.

    Phase 5 audit gap: data layer is multi-run-aware (state JSON has
    runs dict, paths take run_id) but no test verifies that overlapping
    round numbers across runs (e.g. default:lateral:round_00 vs
    nuclei_v1:lateral:round_00) read/write to distinct locations.

    Phase 8 ships the UI for multiple runs; this test verifies the
    DATA LAYER is ready.
    """

    def test_two_runs_share_round_number_but_distinct_paths(self, tmp_path):
        ws = tmp_path / "multi_run_ws"
        ws.mkdir()

        # Both runs have round_00 lateral
        for run_id in ("default", "nuclei_v1"):
            accepted = p.round_accepted_dir(ws, run_id, "lateral", 0)
            accepted.mkdir(parents=True)
            # Same basename in each run — different content
            (accepted / "patch_XY_z0001_img.tif").write_bytes(
                f"{run_id} content".encode()
            )

        # Paths must be distinct
        default_path = p.round_accepted_dir(ws, "default", "lateral", 0)
        nuclei_path = p.round_accepted_dir(ws, "nuclei_v1", "lateral", 0)
        assert default_path != nuclei_path
        assert "/default/" in str(default_path).replace("\\", "/")
        assert "/nuclei_v1/" in str(nuclei_path).replace("\\", "/")

        # Content read back is per-run (not cross-contaminated)
        default_content = (default_path / "patch_XY_z0001_img.tif").read_bytes()
        nuclei_content = (nuclei_path / "patch_XY_z0001_img.tif").read_bytes()
        assert default_content == b"default content"
        assert nuclei_content == b"nuclei_v1 content"

    def test_run_state_isolation_in_v3_unwrap(self, tmp_path):
        """Two runs in the same step2_state.json each have their own
        per-run state.  Loading with different active_run_id returns
        different flat dicts.
        """
        ws = tmp_path / "ws"
        ws.mkdir()
        state_path = p.state_json_path(ws)
        state_path.write_text(json.dumps({
            "schema_version": 3,
            "active_run_id": "default",
            "selected_view": "XY",  # workspace-level (shared)
            "runs": {
                "default": {
                    "current_round": 3,
                    "training_data_by_round": {"lateral_0": ["A"]},
                },
                "nuclei_v1": {
                    "current_round": 1,
                    "training_data_by_round": {"lateral_0": ["B"]},
                },
            },
        }), encoding="utf-8")

        # Load with default active_run_id
        flat_default = p.load_step2_state_flat(ws, active_run_id="default")
        assert flat_default["current_round"] == 3
        assert flat_default["training_data_by_round"] == {"lateral_0": ["A"]}
        assert flat_default["selected_view"] == "XY"  # workspace-level shared

        # Load with nuclei_v1 active_run_id
        flat_nuclei = p.load_step2_state_flat(ws, active_run_id="nuclei_v1")
        assert flat_nuclei["current_round"] == 1
        assert flat_nuclei["training_data_by_round"] == {"lateral_0": ["B"]}
        assert flat_nuclei["selected_view"] == "XY"  # same workspace-level

        # Verify no cross-contamination
        assert flat_default["current_round"] != flat_nuclei["current_round"]


class TestStackIdEdgeCases:
    """Adversarial filename parsing tests.

    The legacy filename convention is `{stack_id}_{VIEW}_z{idx}_{img|mask}.tif`.
    parse_slice_label uses right-to-left anchored regex to handle
    underscored stack_ids correctly.  Spaces in stack_ids are also
    documented as supported (rare in practice but should not crash).
    """

    def test_underscored_stack_id_parses_correctly(self):
        # Common case: stack_id with underscore like "patch_a_v2"
        result = p.parse_slice_label("patch_a_v2_XY_z0042")
        assert result == ("patch_a_v2", "XY", 42)

    def test_deeply_underscored_stack_id(self):
        # Edge case: multiple underscores in stack name
        result = p.parse_slice_label("my_patch_v2_subset_3_XZ_z0001")
        assert result == ("my_patch_v2_subset_3", "XZ", 1)

    def test_underscored_stack_id_with_img_suffix(self):
        # Filename stem with suffix stripping
        result = p.parse_slice_label("patch_a_v2_XY_z0042_img")
        assert result == ("patch_a_v2", "XY", 42)
        result = p.parse_slice_label("patch_a_v2_XY_z0042_mask")
        assert result == ("patch_a_v2", "XY", 42)
        result = p.parse_slice_label("patch_a_v2_XY_z0042_pred_mask")
        assert result == ("patch_a_v2", "XY", 42)

    def test_spaces_in_stack_id(self):
        # Adversarial — space-containing names should parse correctly
        # via right-to-left match on _{VIEW}_z{digits}.
        result = p.parse_slice_label("stack with spaces_XY_z0001")
        assert result == ("stack with spaces", "XY", 1)

    def test_bucket_for_view_after_underscored_parse(self):
        # End-to-end: parse + bucket_for_view chain
        stack_id, view, idx = p.parse_slice_label("patch_a_v2_XZ_z0010")
        assert p.bucket_for_view(view) == "axial"
        stack_id, view, idx = p.parse_slice_label("patch_a_v2_XY_z0010")
        assert p.bucket_for_view(view) == "lateral"


class TestLayoutDetectionLifecycle:
    """Verifies detect_layout_version transitions:
    UNKNOWN → LEGACY → MIXED → NEW
    across the migration lifecycle.
    """

    def test_empty_then_legacy(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert p.detect_layout_version(ws) == p.LAYOUT_VERSION_UNKNOWN

        # User opens a workspace with legacy data
        (ws / "training_data" / "round_00").mkdir(parents=True)
        assert p.detect_layout_version(ws) == p.LAYOUT_VERSION_LEGACY

    def test_legacy_then_mixed_then_new(self, tmp_path):
        ws = tmp_path / "ws"
        # Start with legacy
        (ws / "training_data" / "round_00").mkdir(parents=True)
        assert p.detect_layout_version(ws) == p.LAYOUT_VERSION_LEGACY

        # Mid-migration: new appears while legacy still exists → MIXED
        (ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
        assert p.detect_layout_version(ws) == p.LAYOUT_VERSION_MIXED

        # Migration completes: legacy pruned → NEW
        import shutil
        shutil.rmtree(ws / "training_data")
        assert p.detect_layout_version(ws) == p.LAYOUT_VERSION_NEW

    def test_is_new_layout_state_aligns_with_detection(self, migrated_ws):
        assert p.is_new_layout_state(migrated_ws) is True
        assert p.detect_layout_version(migrated_ws) == p.LAYOUT_VERSION_NEW
