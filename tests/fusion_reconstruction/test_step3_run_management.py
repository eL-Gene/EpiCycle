"""Tests for ``napari_epicycle/fusion_reconstruction/_run_management.py``.

Pure-Python (no Qt / napari) — the Step 3 mirror of Step 2's run-disk +
summaries machinery.

Covers: create_run_on_disk (dir + manifest, duplicate guard, the
run-context-vs-default-patch-recon collision guard, default-run refusal);
delete (confirm + default refusal); archive/unarchive/edit; the
default-run-synthesizing summaries + named-run summaries with patch_counts
+ is_active + the patch-recon-dir classification; find_runs_inheriting_from;
resolve_init_spec for pretrained / step3_refined / browse / step2_run.
"""

from __future__ import annotations

import json

import pytest

from napari_epicycle.fusion_reconstruction import _paths as p
from napari_epicycle.fusion_reconstruction import _run_management as rm
from packages.epicycle_io import step3_run_manifest as man


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _runs_root(ws):
    return p.step3_runs_root(ws)


def _make_default_patch_recon(ws, patch_name):
    """A default-run per-patch reconstruction dir (NO run_manifest.json)."""
    d = _runs_root(ws) / patch_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "labels_3d.tif").write_bytes(b"\x00")
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    return d


def _write_state(ws, envelope):
    """Write a step3_state.json envelope at the workspace root."""
    p.state_json_path(ws).write_text(
        json.dumps(envelope, indent=2), encoding="utf-8"
    )


def _make_step2_run(ws, run_id, *, round_idx=0, buckets=("lateral", "axial")):
    """Build a minimal Step-2 run layout the bridge can scan.

    Creates ``artifacts/step2/runs/{run_id}/run_manifest.json`` plus, per
    bucket, a ``{bucket}/round_NN/model/{run_id}_{bucket}`` weights file
    (the new flat layout ``list_run_model_sources`` walks).  Returns the
    ``(lateral_path, axial_path)`` it wrote (a bucket omitted from
    ``buckets`` returns ``None`` in that slot).
    """
    from napari_epicycle.hitl_training_annotation import _paths as s2p

    rm_path = s2p.run_manifest_path(ws, run_id)
    rm_path.parent.mkdir(parents=True, exist_ok=True)
    if not rm_path.exists():
        rm_path.write_text(
            json.dumps({"run_id": run_id, "display_name": run_id}),
            encoding="utf-8",
        )
    paths = {"lateral": None, "axial": None}
    for bucket in buckets:
        model_dir = (
            s2p.bucket_root(ws, run_id, bucket)
            / f"round_{round_idx:02d}" / "model"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        wp = model_dir / f"{run_id}_{bucket}"
        wp.write_bytes(b"\x00")
        paths[bucket] = wp
    return paths["lateral"], paths["axial"]


# ---------------------------------------------------------------------------
# create_run_on_disk
# ---------------------------------------------------------------------------

class TestCreateRun:

    def test_creates_dir_and_manifest(self, tmp_path):
        manifest = rm.create_run_on_disk(
            tmp_path, "cells", display_name="Cells", target="cells",
        )
        run_dir = p.run_root(tmp_path, "cells")
        assert run_dir.is_dir()
        mp = p.run_manifest_path(tmp_path, "cells")
        assert mp.is_file()
        on_disk = man.read_run_manifest(mp)
        assert on_disk["run_id"] == "cells"
        assert on_disk["display_name"] == "Cells"
        assert on_disk["target"] == "cells"
        assert manifest == on_disk

    def test_create_with_init_spec_and_model(self, tmp_path):
        spec = man.init_spec_pretrained(pretrained_name="cyto3")
        manifest = rm.create_run_on_disk(
            tmp_path, "nuclei", display_name="Nuclei",
            init_spec=spec,
            model={"lateral_model_path": "/a", "axial_model_path": "/b",
                   "model_source_mode": "pretrained"},
        )
        assert manifest["created_from"]["mode"] == "pretrained"
        assert manifest["model"]["lateral_model_path"] == "/a"

    def test_duplicate_slug_raises(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        with pytest.raises(rm.RunAlreadyExistsError):
            rm.create_run_on_disk(tmp_path, "cells", display_name="Cells 2")

    def test_case_folded_duplicate_raises(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "Cells", display_name="Cells")
        with pytest.raises(rm.RunAlreadyExistsError):
            rm.create_run_on_disk(tmp_path, "cells", display_name="cells")

    def test_collision_with_default_patch_recon_dir_raises(self, tmp_path):
        # A default-run per-patch recon dir named "patchX" already exists
        # under runs/; creating a run "patchX" must be refused.
        _make_default_patch_recon(tmp_path, "patchX")
        with pytest.raises(rm.RunAlreadyExistsError):
            rm.create_run_on_disk(tmp_path, "patchX", display_name="Clash")
        # ...case-folded too.
        _make_default_patch_recon(tmp_path, "PatchY")
        with pytest.raises(rm.RunAlreadyExistsError):
            rm.create_run_on_disk(tmp_path, "patchy", display_name="Clash")

    def test_default_slug_refused(self, tmp_path):
        with pytest.raises(rm.RunAlreadyExistsError):
            rm.create_run_on_disk(tmp_path, "default", display_name="Nope")
        # No runs/default/ context dir is created.
        assert not p.run_root(tmp_path, "default").exists()

    def test_default_slug_refused_case_folded(self, tmp_path):
        with pytest.raises(rm.RunAlreadyExistsError):
            rm.create_run_on_disk(tmp_path, "Default", display_name="Nope")

    def test_invalid_slug_raises(self, tmp_path):
        with pytest.raises(rm.InvalidRunIdError):
            rm.create_run_on_disk(tmp_path, "../evil", display_name="X")

    def test_named_run_does_not_collide_with_patch_recon(self, tmp_path):
        # A patch recon dir + a differently-named run coexist fine.
        _make_default_patch_recon(tmp_path, "patch_001")
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        assert p.run_root(tmp_path, "cells").is_dir()


# ---------------------------------------------------------------------------
# delete_run_from_disk
# ---------------------------------------------------------------------------

class TestDeleteRun:

    def test_delete_requires_confirm(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        with pytest.raises(ValueError):
            rm.delete_run_from_disk(tmp_path, "cells")  # confirm defaults False
        assert p.run_root(tmp_path, "cells").is_dir()  # still there

    def test_delete_with_confirm(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        rm.delete_run_from_disk(tmp_path, "cells", confirm=True)
        assert not p.run_root(tmp_path, "cells").exists()

    def test_delete_refuses_default(self, tmp_path):
        with pytest.raises(ValueError):
            rm.delete_run_from_disk(tmp_path, "default", confirm=True)

    def test_delete_missing_raises(self, tmp_path):
        with pytest.raises(rm.RunNotFoundError):
            rm.delete_run_from_disk(tmp_path, "ghost", confirm=True)


# ---------------------------------------------------------------------------
# archive / unarchive / edit
# ---------------------------------------------------------------------------

class TestManifestMutations:

    def test_archive_unarchive(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        assert man.read_run_manifest(
            p.run_manifest_path(tmp_path, "cells")
        )["archived_at"] is None

        rm.archive_run(tmp_path, "cells")
        assert man.is_run_archived(
            man.read_run_manifest(p.run_manifest_path(tmp_path, "cells"))
        )

        rm.unarchive_run(tmp_path, "cells")
        assert not man.is_run_archived(
            man.read_run_manifest(p.run_manifest_path(tmp_path, "cells"))
        )

    def test_edit_metadata(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        updated = rm.edit_run_metadata(
            tmp_path, "cells",
            display_name="Renamed", target="nuclei",
            description="d", notes="n",
        )
        assert updated["display_name"] == "Renamed"
        assert updated["target"] == "nuclei"
        assert updated["description"] == "d"
        assert updated["notes"] == "n"
        # persisted
        on_disk = man.read_run_manifest(p.run_manifest_path(tmp_path, "cells"))
        assert on_disk["display_name"] == "Renamed"

    def test_edit_none_leaves_unchanged(self, tmp_path):
        rm.create_run_on_disk(
            tmp_path, "cells", display_name="Cells", target="cells"
        )
        rm.edit_run_metadata(tmp_path, "cells", description="only desc")
        on_disk = man.read_run_manifest(p.run_manifest_path(tmp_path, "cells"))
        assert on_disk["display_name"] == "Cells"   # unchanged
        assert on_disk["target"] == "cells"          # unchanged
        assert on_disk["description"] == "only desc"

    def test_edit_empty_display_name_raises(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        with pytest.raises(ValueError):
            rm.edit_run_metadata(tmp_path, "cells", display_name="  ")

    def test_archive_default_raises(self, tmp_path):
        with pytest.raises(rm.RunNotFoundError):
            rm.archive_run(tmp_path, "default")

    def test_archive_missing_raises(self, tmp_path):
        with pytest.raises(rm.RunNotFoundError):
            rm.archive_run(tmp_path, "ghost")

    def test_edit_corrupt_manifest_raises(self, tmp_path):
        run_dir = _runs_root(tmp_path) / "cells"
        run_dir.mkdir(parents=True)
        (run_dir / man.RUN_MANIFEST_FILENAME).write_text("{ bad", encoding="utf-8")
        with pytest.raises(rm.RunNotFoundError):
            rm.edit_run_metadata(tmp_path, "cells", description="x")


# ---------------------------------------------------------------------------
# list_run_summaries
# ---------------------------------------------------------------------------

class TestListRunSummaries:

    def test_synthesizes_default_even_with_no_state(self, tmp_path):
        summaries = rm.list_run_summaries(tmp_path)
        assert len(summaries) == 1
        d = summaries[0]
        assert d["run_id"] == "default"
        assert d["display_name"] == "Default"
        assert d["is_active"] is True
        assert d["manifest_ok"] is True
        assert d["manifest_path"] == ""
        assert d["patch_counts"] == {
            "reconstructed": 0, "curated": 0, "accepted": 0,
            "finalized": 0, "total": 0,
        }

    def test_default_patch_counts_from_state(self, tmp_path):
        _write_state(tmp_path, {
            "schema_version": 3,
            "active_run_id": "default",
            "runs": {
                "default": {
                    "display_name": "Default",
                    "patches": {
                        "p1": {"status": "reconstructed"},
                        "p2": {"status": "curating"},
                        "p3": {"status": "accepted"},
                        "p4": {"status": "finalized"},
                        "p5": {"status": "unprocessed"},
                        "p6": {"status": "predicted"},
                    },
                }
            },
        })
        d = rm.list_run_summaries(tmp_path)[0]
        assert d["patch_counts"] == {
            "reconstructed": 1, "curated": 1, "accepted": 1,
            "finalized": 1, "total": 6,
        }

    def test_lists_named_run_and_marks_active(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells",
                              target="cells")
        _write_state(tmp_path, {
            "schema_version": 3,
            "active_run_id": "cells",
            "runs": {
                "default": {"patches": {"p1": {"status": "reconstructed"}}},
                "cells": {
                    "patches": {
                        "pc1": {"status": "accepted"},
                        "pc2": {"status": "accepted"},
                    }
                },
            },
        })
        summaries = {s["run_id"]: s for s in rm.list_run_summaries(tmp_path)}
        assert set(summaries) == {"default", "cells"}
        # default NOT active; cells active.
        assert summaries["default"]["is_active"] is False
        assert summaries["cells"]["is_active"] is True
        # patch counts pulled per-run from state.
        assert summaries["default"]["patch_counts"]["reconstructed"] == 1
        assert summaries["cells"]["patch_counts"]["accepted"] == 2
        assert summaries["cells"]["patch_counts"]["total"] == 2
        # manifest fields surfaced.
        assert summaries["cells"]["display_name"] == "Cells"
        assert summaries["cells"]["target"] == "cells"
        assert summaries["cells"]["manifest_ok"] is True
        assert summaries["cells"]["manifest_path"].endswith("run_manifest.json")

    def test_patch_recon_dir_not_classified_as_run(self, tmp_path):
        # A default-run per-patch recon dir (no run_manifest.json) must NOT
        # appear as a run in the summaries.
        _make_default_patch_recon(tmp_path, "patch_001")
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        ids = {s["run_id"] for s in rm.list_run_summaries(tmp_path)}
        assert ids == {"default", "cells"}
        assert "patch_001" not in ids

    def test_corrupt_named_manifest_flagged(self, tmp_path):
        run_dir = _runs_root(tmp_path) / "broken"
        run_dir.mkdir(parents=True)
        (run_dir / man.RUN_MANIFEST_FILENAME).write_text("{ bad", encoding="utf-8")
        summaries = {s["run_id"]: s for s in rm.list_run_summaries(tmp_path)}
        assert "broken" in summaries
        assert summaries["broken"]["manifest_ok"] is False
        # falls back to slug for display_name
        assert summaries["broken"]["display_name"] == "broken"

    def test_archived_flag_surfaced(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        rm.archive_run(tmp_path, "cells")
        summaries = {s["run_id"]: s for s in rm.list_run_summaries(tmp_path)}
        assert summaries["cells"]["archived"] is True
        assert summaries["default"]["archived"] is False

    def test_never_raises_on_corrupt_state(self, tmp_path):
        p.state_json_path(tmp_path).write_text("{ not json", encoding="utf-8")
        summaries = rm.list_run_summaries(tmp_path)
        # default still synthesized.
        assert any(s["run_id"] == "default" for s in summaries)


# ---------------------------------------------------------------------------
# find_runs_inheriting_from
# ---------------------------------------------------------------------------

class TestFindRunsInheritingFrom:

    def test_finds_step3_refined_dependents(self, tmp_path):
        rm.create_run_on_disk(
            tmp_path, "child",
            display_name="Child",
            init_spec=man.init_spec_step3_refined(source_step3_bundle="parent"),
        )
        rm.create_run_on_disk(tmp_path, "unrelated", display_name="U")
        deps = rm.find_runs_inheriting_from(tmp_path, "parent")
        assert deps == ["child"]

    def test_finds_step2_run_dependents(self, tmp_path):
        rm.create_run_on_disk(
            tmp_path, "child2",
            display_name="Child2",
            init_spec=man.init_spec_step2_run(source_step2_run_id="step2A"),
        )
        deps = rm.find_runs_inheriting_from(tmp_path, "step2A")
        assert deps == ["child2"]

    def test_none_when_no_dependents(self, tmp_path):
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        assert rm.find_runs_inheriting_from(tmp_path, "cells") == []

    def test_empty_when_no_runs(self, tmp_path):
        assert rm.find_runs_inheriting_from(tmp_path, "x") == []


# ---------------------------------------------------------------------------
# resolve_init_spec_to_checkpoint
# ---------------------------------------------------------------------------

class TestResolveInitSpec:

    def test_pretrained(self, tmp_path):
        spec = man.init_spec_pretrained(pretrained_name="cyto3")
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, spec) == (
            "cyto3", "cyto3"
        )

    def test_browse(self, tmp_path):
        spec = man.init_spec_browse(external_path="/m/weights")
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, spec) == (
            "/m/weights", "/m/weights"
        )

    def test_step3_refined_resolves_when_both_exist(self, tmp_path):
        bundle = "bundleA"
        lat = p.refined_model_weights_path(tmp_path, bundle, "lateral")
        ax = p.refined_model_weights_path(tmp_path, bundle, "axial")
        lat.parent.mkdir(parents=True, exist_ok=True)
        lat.write_bytes(b"\x00")
        ax.write_bytes(b"\x00")
        spec = man.init_spec_step3_refined(source_step3_bundle=bundle)
        resolved = rm.resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert resolved == (str(lat), str(ax))

    def test_step3_refined_none_when_missing(self, tmp_path):
        spec = man.init_spec_step3_refined(source_step3_bundle="ghost")
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, spec) is None

    def test_step2_run_none_when_run_absent(self, tmp_path):
        # No Step-2 layout on disk → unresolvable.
        spec = man.init_spec_step2_run(source_step2_run_id="step2A")
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, spec) is None

    def test_step2_run_resolves_latest_pair(self, tmp_path):
        # Build a Step-2 run with lateral + axial weights across two rounds;
        # resolution should pick the LATEST round per bucket.
        lat0, ax0 = _make_step2_run(tmp_path, "step2A", round_idx=0)
        lat1, ax1 = _make_step2_run(tmp_path, "step2A", round_idx=1)
        spec = man.init_spec_step2_run(source_step2_run_id="step2A")
        resolved = rm.resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert resolved == (str(lat1), str(ax1))

    def test_step2_run_none_when_only_one_bucket(self, tmp_path):
        # Only a lateral model → a Step 3 run needs the PAIR → unresolvable.
        _make_step2_run(tmp_path, "step2B", round_idx=0, buckets=("lateral",))
        spec = man.init_spec_step2_run(source_step2_run_id="step2B")
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, spec) is None

    def test_step3_refined_resolves_in_named_run(self, tmp_path):
        # A bundle living under a NAMED run (not flat) resolves only when the
        # spec carries that run's id (Phase 5 audit HIGH-1B: exact resolution,
        # no name-scan).
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        lat = p.refined_model_weights_path(tmp_path, "bX", "lateral", "cells")
        ax = p.refined_model_weights_path(tmp_path, "bX", "axial", "cells")
        lat.parent.mkdir(parents=True, exist_ok=True)
        lat.write_bytes(b"\x00")
        ax.write_bytes(b"\x00")
        spec = man.init_spec_step3_refined(
            source_step3_bundle="bX", source_step3_run_id="cells"
        )
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, spec) == (
            str(lat), str(ax)
        )

    def test_step3_refined_name_only_does_not_scan_named_run(self, tmp_path):
        # HIGH-1B: a name-only spec (no run_id) resolves the FLAT default
        # ONLY — it must NOT silently fall back to a named-run bundle.
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        lat = p.refined_model_weights_path(tmp_path, "bX", "lateral", "cells")
        ax = p.refined_model_weights_path(tmp_path, "bX", "axial", "cells")
        lat.parent.mkdir(parents=True, exist_ok=True)
        lat.write_bytes(b"\x00")
        ax.write_bytes(b"\x00")
        spec = man.init_spec_step3_refined(source_step3_bundle="bX")
        # No flat-default bundle named bX exists → unresolvable.
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, spec) is None

    def test_step3_refined_same_name_resolves_by_run_id(self, tmp_path):
        # HIGH-1B regression: two runs each hold a SAME-named trained bundle.
        # Resolution must return EACH run's own pair (the audit's missing case).
        rm.create_run_on_disk(tmp_path, "runA", display_name="Run A")
        rm.create_run_on_disk(tmp_path, "runB", display_name="Run B")
        pairs = {}
        for run_id, tag in (("runA", b"A"), ("runB", b"B")):
            lat = p.refined_model_weights_path(
                tmp_path, "shared", "lateral", run_id
            )
            ax = p.refined_model_weights_path(
                tmp_path, "shared", "axial", run_id
            )
            lat.parent.mkdir(parents=True, exist_ok=True)
            lat.write_bytes(b"LAT" + tag)
            ax.write_bytes(b"AX" + tag)
            pairs[run_id] = (str(lat), str(ax))

        specA = man.init_spec_step3_refined(
            source_step3_bundle="shared", source_step3_run_id="runA"
        )
        specB = man.init_spec_step3_refined(
            source_step3_bundle="shared", source_step3_run_id="runB"
        )
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, specA) == pairs["runA"]
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, specB) == pairs["runB"]
        # Sanity: the two runs really resolved to DIFFERENT files.
        assert pairs["runA"] != pairs["runB"]

    def test_none_spec(self, tmp_path):
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, None) is None

    def test_malformed_spec(self, tmp_path):
        assert rm.resolve_init_spec_to_checkpoint(tmp_path, {"mode": "??"}) is None


# ---------------------------------------------------------------------------
# list_cross_run_refined_models
# ---------------------------------------------------------------------------

class TestCrossRunRefinedModels:

    def test_walks_flat_and_named_runs(self, tmp_path):
        # Flat default bundle.
        for bucket in ("lateral", "axial"):
            wp = p.refined_model_weights_path(tmp_path, "flatA", bucket)
            wp.parent.mkdir(parents=True, exist_ok=True)
            wp.write_bytes(b"\x00")
        # Named-run bundle.
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        for bucket in ("lateral", "axial"):
            wp = p.refined_model_weights_path(tmp_path, "cellsB", bucket, "cells")
            wp.parent.mkdir(parents=True, exist_ok=True)
            wp.write_bytes(b"\x00")

        out = rm.list_cross_run_refined_models(tmp_path)
        by_key = {(e["bundle_name"], e["run_id"], e["bucket"]) for e in out}
        assert ("flatA", None, "lateral") in by_key
        assert ("flatA", None, "axial") in by_key
        assert ("cellsB", "cells", "lateral") in by_key
        assert ("cellsB", "cells", "axial") in by_key
        # Backend defaulted for a manifest-less bundle.
        assert all(e["backend"] == "cellpose" for e in out)

    def test_empty_when_no_models(self, tmp_path):
        assert rm.list_cross_run_refined_models(tmp_path) == []

    def test_default_nested_reads_runs_default_refinement(self, tmp_path):
        """Phase 7: with default_nested=True the default run's bundles are
        discovered under runs/default/refinement/models and tagged with the
        'default' run_id (not None)."""
        # Bundle physically under the nested default-run tree.
        for bucket in ("lateral", "axial"):
            wp = p.refined_model_weights_path(
                tmp_path, "nestedA", bucket, rm.DEFAULT_RUN_ID
            )
            wp.parent.mkdir(parents=True, exist_ok=True)
            wp.write_bytes(b"\x00")

        # Flat scan (default_nested=False) does NOT see it.
        flat = rm.list_cross_run_refined_models(tmp_path)
        assert all(e["bundle_name"] != "nestedA" for e in flat)

        # Nested scan finds it under the 'default' run_id.
        out = rm.list_cross_run_refined_models(tmp_path, default_nested=True)
        keys = {(e["bundle_name"], e["run_id"], e["bucket"]) for e in out}
        assert ("nestedA", rm.DEFAULT_RUN_ID, "lateral") in keys
        assert ("nestedA", rm.DEFAULT_RUN_ID, "axial") in keys


# ---------------------------------------------------------------------------
# trained-vs-seed discriminator + trained_only filter (audit HIGH-1A)
# ---------------------------------------------------------------------------

def _make_trained_bundle(ws, bundle, run_id=None):
    """A refinement-TRAINED bundle: weights + a schema-2.0 manifest.json."""
    for bucket in ("lateral", "axial"):
        wp = p.refined_model_weights_path(ws, bundle, bucket, run_id)
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_bytes(b"\x00")
    mp = p.refined_model_manifest_path(ws, bundle, run_id)
    mp.write_text(
        json.dumps({
            "schema_version": "2.0",
            "name": bundle,
            "lateral": {"backend": "cellpose", "weights_path": "lateral"},
            "axial": {"backend": "cellpose", "weights_path": "axial"},
        }),
        encoding="utf-8",
    )


def _make_seed_bundle(ws, run_id=None, *, bundle=None):
    """A copy-on-reuse SEED bundle: weights + a model_manifest.json marker."""
    name = bundle or rm.INIT_MODEL_NAME
    for bucket in ("lateral", "axial"):
        wp = p.refined_model_weights_path(ws, name, bucket, run_id)
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_bytes(b"\x00")
    mp = p.refined_model_dir(ws, name, run_id) / rm.MODEL_MANIFEST_FILENAME
    mp.write_text(
        json.dumps({
            "schema_version": "1.0",
            "backend": "cellpose",
            "model_source_mode": "step3_refined",
            "copied_from": {"lateral": "/src/lat", "axial": "/src/ax"},
        }),
        encoding="utf-8",
    )
    return name


class TestSeedDiscriminator:

    def test_is_seed_bundle_by_name(self, tmp_path):
        # The fixed copy-on-reuse name is always a seed.
        assert rm.is_seed_bundle(tmp_path, rm.INIT_MODEL_NAME) is True

    def test_is_seed_bundle_by_marker_file(self, tmp_path):
        # A bundle NOT named init_model but carrying model_manifest.json is
        # still a seed (marker-file discriminator, not just the name).
        _make_seed_bundle(tmp_path, bundle="weird_seed_name")
        assert rm.is_seed_bundle(tmp_path, "weird_seed_name") is True

    def test_trained_bundle_is_not_seed(self, tmp_path):
        _make_trained_bundle(tmp_path, "trainedA")
        assert rm.is_seed_bundle(tmp_path, "trainedA") is False

    def test_trained_only_excludes_seed_includes_trained(self, tmp_path):
        # In a single run: a trained bundle + a seed copy coexist.
        rm.create_run_on_disk(tmp_path, "cells", display_name="Cells")
        _make_trained_bundle(tmp_path, "trainedA", run_id="cells")
        _make_seed_bundle(tmp_path, run_id="cells")  # init_model seed

        # Default (trained_only=False) sees BOTH bundles.
        all_bundles = {
            e["bundle_name"]
            for e in rm.list_cross_run_refined_models(tmp_path)
        }
        assert "trainedA" in all_bundles
        assert rm.INIT_MODEL_NAME in all_bundles

        # trained_only=True drops the seed, keeps the trained bundle.
        trained = rm.list_cross_run_refined_models(tmp_path, trained_only=True)
        names = {e["bundle_name"] for e in trained}
        assert "trainedA" in names
        assert rm.INIT_MODEL_NAME not in names
        # And the trained entry still yields both buckets.
        buckets = {e["bucket"] for e in trained if e["bundle_name"] == "trainedA"}
        assert buckets == {"lateral", "axial"}

    def test_trained_only_excludes_marker_seed_not_named_init_model(self, tmp_path):
        _make_trained_bundle(tmp_path, "trainedA")
        _make_seed_bundle(tmp_path, bundle="masquerade")  # marker, odd name
        names = {
            e["bundle_name"]
            for e in rm.list_cross_run_refined_models(tmp_path, trained_only=True)
        }
        assert names == {"trainedA"}


# ---------------------------------------------------------------------------
# materialize_run_model — copy-on-reuse
# ---------------------------------------------------------------------------

class TestMaterializeRunModel:

    def test_pretrained_no_copy(self, tmp_path):
        spec = man.init_spec_pretrained(pretrained_name="cyto3")
        sel = rm.materialize_run_model(tmp_path, "cells", spec)
        assert sel == {
            "lateral_model_path": "cyto3",
            "axial_model_path": "cyto3",
            "model_source_mode": "pretrained",
        }
        # No bundle dir created for a pretrained run.
        assert not p.refined_model_dir(
            tmp_path, rm.INIT_MODEL_NAME, "cells"
        ).exists()

    def test_step3_refined_copies_into_run(self, tmp_path):
        # Source bundle (flat).
        src_lat = p.refined_model_weights_path(tmp_path, "src", "lateral")
        src_ax = p.refined_model_weights_path(tmp_path, "src", "axial")
        src_lat.parent.mkdir(parents=True, exist_ok=True)
        src_lat.write_bytes(b"LAT")
        src_ax.write_bytes(b"AX")
        spec = man.init_spec_step3_refined(source_step3_bundle="src")

        sel = rm.materialize_run_model(tmp_path, "cells", spec, target="cells")

        dst_lat = p.refined_model_weights_path(
            tmp_path, rm.INIT_MODEL_NAME, "lateral", "cells"
        )
        dst_ax = p.refined_model_weights_path(
            tmp_path, rm.INIT_MODEL_NAME, "axial", "cells"
        )
        # Copied into the run's own tree.
        assert dst_lat.is_file() and dst_lat.read_bytes() == b"LAT"
        assert dst_ax.is_file() and dst_ax.read_bytes() == b"AX"
        # Originals untouched.
        assert src_lat.read_bytes() == b"LAT"
        assert src_ax.read_bytes() == b"AX"
        # Returned paths point at the run's copies.
        assert sel["lateral_model_path"] == str(dst_lat)
        assert sel["axial_model_path"] == str(dst_ax)
        assert sel["model_source_mode"] == "step3_refined"
        # model_manifest.json written with copied_from + created_from.
        mpath = (
            p.refined_model_dir(tmp_path, rm.INIT_MODEL_NAME, "cells")
            / rm.MODEL_MANIFEST_FILENAME
        )
        assert mpath.is_file()
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        assert manifest["backend"] == "cellpose"
        assert manifest["target"] == "cells"
        assert manifest["model_source_mode"] == "step3_refined"
        assert manifest["created_from"]["mode"] == "step3_refined"
        assert manifest["copied_from"] == {
            "lateral": str(src_lat), "axial": str(src_ax),
        }

    def test_browse_copies_file_into_both_buckets(self, tmp_path):
        ext = tmp_path / "external_weights"
        ext.write_bytes(b"EXT")
        spec = man.init_spec_browse(external_path=str(ext))
        sel = rm.materialize_run_model(tmp_path, "nuclei", spec)
        dst_lat = p.refined_model_weights_path(
            tmp_path, rm.INIT_MODEL_NAME, "lateral", "nuclei"
        )
        dst_ax = p.refined_model_weights_path(
            tmp_path, rm.INIT_MODEL_NAME, "axial", "nuclei"
        )
        assert dst_lat.read_bytes() == b"EXT"
        assert dst_ax.read_bytes() == b"EXT"
        assert sel["model_source_mode"] == "browse"
        # External source untouched.
        assert ext.read_bytes() == b"EXT"

    def test_unresolvable_raises_and_cleans_up(self, tmp_path):
        spec = man.init_spec_step2_run(source_step2_run_id="ghost")
        with pytest.raises(rm.ModelMaterializationError):
            rm.materialize_run_model(tmp_path, "cells", spec)
        # No partial bundle left behind.
        assert not p.refined_model_dir(
            tmp_path, rm.INIT_MODEL_NAME, "cells"
        ).exists()

    def test_no_init_spec_returns_blank(self, tmp_path):
        sel = rm.materialize_run_model(tmp_path, "cells", None)
        assert sel == {
            "lateral_model_path": None,
            "axial_model_path": None,
            "model_source_mode": None,
        }


# ---------------------------------------------------------------------------
# map_model_source_mode_to_state — vocabulary mapping (audit MED-1)
# ---------------------------------------------------------------------------

class TestMapModelSourceModeToState:

    _STATE_VOCAB = {"auto", "import", "pretrained"}

    def test_pretrained_maps_to_pretrained(self):
        assert rm.map_model_source_mode_to_state("pretrained") == "pretrained"

    def test_import_family_maps_to_import(self):
        for mode in ("browse", "step2_run", "step3_refined"):
            assert rm.map_model_source_mode_to_state(mode) == "import", mode

    def test_none_and_unknown_map_to_auto(self):
        assert rm.map_model_source_mode_to_state(None) == "auto"
        assert rm.map_model_source_mode_to_state("") == "auto"
        assert rm.map_model_source_mode_to_state("garbage") == "auto"

    def test_never_emits_value_outside_state_vocab(self):
        # Every materialize-mode (and the blank/None case) must map into the
        # consumer vocabulary the combo + set_model_selection understand.
        for mode in (
            "pretrained", "browse", "step2_run", "step3_refined",
            None, "", "auto", "import", "nonsense",
        ):
            assert rm.map_model_source_mode_to_state(mode) in self._STATE_VOCAB


# ---------------------------------------------------------------------------
# run_path_id_for / summarize_run_intermediates / clean_run_intermediates
# (the "Clean intermediates" feature)
# ---------------------------------------------------------------------------

def _make_cache_file(ws, run_path_id, patch_stem, view, kind, size=100):
    """Write a fake prediction-cache .npy for (patch_stem, view, kind)."""
    cache = p.prediction_cache_dir(ws, run_path_id)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{patch_stem}_{view}_{kind}.npy"
    f.write_bytes(b"\x00" * size)
    return f


def _seed_intermediates_workspace(ws, run_path_id=None, slug="default"):
    """Build a state envelope + prediction cache + qc/curation sidecars.

    Patch set (each patch gets a distinct cache footprint):
      * patch_a (reconstructed) — flows + probs            (2 files)
      * patch_b (curating)      — flows + probs + masks     (3 files)
      * patch_c (accepted)      — flows + probs             (2 files)
      * patch_d (finalized)     — flows                     (1 file)
      * patch_e (predicted)     — flows + probs  [IGNORED]  (2 files)
      * patch_f (unprocessed)   — no cache

    Returns the set of cache Paths that BELONG to reconstructed-or-beyond
    patches (a..d) and the set that must be PRESERVED (patch_e's).
    """
    _write_state(ws, {
        "schema_version": 3,
        "active_run_id": slug,
        "runs": {
            slug: {
                "patches": {
                    "patch_a.tif": {"status": "reconstructed"},
                    "patch_b.tif": {"status": "curating"},
                    "patch_c.tif": {"status": "accepted"},
                    "patch_d.tif": {"status": "finalized"},
                    "patch_e.tif": {"status": "predicted"},
                    "patch_f.tif": {"status": "unprocessed"},
                },
            },
        },
    })
    recon_files = set()
    recon_files.add(_make_cache_file(ws, run_path_id, "patch_a", "xy", "flows", 100))
    recon_files.add(_make_cache_file(ws, run_path_id, "patch_a", "xy", "probs", 100))
    recon_files.add(_make_cache_file(ws, run_path_id, "patch_b", "xz", "flows", 50))
    recon_files.add(_make_cache_file(ws, run_path_id, "patch_b", "xz", "probs", 50))
    recon_files.add(_make_cache_file(ws, run_path_id, "patch_b", "xz", "masks", 50))
    recon_files.add(_make_cache_file(ws, run_path_id, "patch_c", "yz", "flows", 10))
    recon_files.add(_make_cache_file(ws, run_path_id, "patch_c", "yz", "probs", 10))
    recon_files.add(_make_cache_file(ws, run_path_id, "patch_d", "xy", "flows", 7))
    # predicted-only patch — its cache must NEVER be summarized or cleaned.
    preserved = set()
    preserved.add(_make_cache_file(ws, run_path_id, "patch_e", "xy", "flows", 999))
    preserved.add(_make_cache_file(ws, run_path_id, "patch_e", "xy", "probs", 999))
    # qc + curation sidecars that must be left intact by clean.
    qc = p.qc_dir(ws, run_id=run_path_id)
    qc.mkdir(parents=True, exist_ok=True)
    qc_csv = qc / "patch_a_features.csv"
    qc_csv.write_text("id,vol\n1,2\n", encoding="utf-8")
    preserved.add(qc_csv)
    return recon_files, preserved


class TestRunPathIdFor:

    def test_default_flat(self):
        assert rm.run_path_id_for("default", default_nested=False) is None

    def test_default_nested(self):
        assert rm.run_path_id_for("default", default_nested=True) == "default"

    def test_named_run_ignores_flag(self):
        assert rm.run_path_id_for("cells", default_nested=False) == "cells"
        assert rm.run_path_id_for("cells", default_nested=True) == "cells"


class TestSummarizeRunIntermediates:

    def test_counts_only_reconstructed_patches(self, tmp_path):
        recon_files, _ = _seed_intermediates_workspace(tmp_path)
        info = rm.summarize_run_intermediates(
            tmp_path, "default", run_path_id=None
        )
        assert info["n_files"] == len(recon_files)  # 8
        assert info["n_patches"] == 4  # a, b, c, d
        assert info["bytes"] == sum(f.stat().st_size for f in recon_files)

    def test_ignores_predicted_patch_cache(self, tmp_path):
        _, preserved = _seed_intermediates_workspace(tmp_path)
        info = rm.summarize_run_intermediates(
            tmp_path, "default", run_path_id=None
        )
        # patch_e's 999+999 bytes (predicted) are NOT in the total.
        pred_bytes = sum(
            f.stat().st_size for f in preserved if "patch_e" in f.name
        )
        assert info["bytes"] != pred_bytes
        # bytes only reflect a..d
        assert info["bytes"] < pred_bytes  # a..d total (234) < 1998

    def test_named_run_scope(self, tmp_path):
        recon_files, _ = _seed_intermediates_workspace(
            tmp_path, run_path_id="cells", slug="cells"
        )
        info = rm.summarize_run_intermediates(
            tmp_path, "cells", run_path_id="cells"
        )
        assert info["n_files"] == len(recon_files)
        assert info["n_patches"] == 4

    def test_missing_cache_dir_returns_zeros(self, tmp_path):
        _write_state(tmp_path, {
            "schema_version": 3,
            "active_run_id": "default",
            "runs": {"default": {"patches": {
                "patch_a.tif": {"status": "reconstructed"},
            }}},
        })
        info = rm.summarize_run_intermediates(
            tmp_path, "default", run_path_id=None
        )
        assert info == {"n_patches": 0, "bytes": 0, "n_files": 0}

    def test_no_state_returns_zeros(self, tmp_path):
        info = rm.summarize_run_intermediates(
            tmp_path, "default", run_path_id=None
        )
        assert info == {"n_patches": 0, "bytes": 0, "n_files": 0}


class TestCleanRunIntermediates:

    def test_deletes_only_reconstructed_patch_cache(self, tmp_path):
        recon_files, preserved = _seed_intermediates_workspace(tmp_path)
        expected_bytes = sum(f.stat().st_size for f in recon_files)
        res = rm.clean_run_intermediates(
            tmp_path, "default", run_path_id=None
        )
        assert res["n_cleared"] == len(recon_files)  # 8
        assert res["bytes_freed"] == expected_bytes
        # reconstructed-patch cache gone …
        for f in recon_files:
            assert not f.exists(), f
        # … predicted-patch cache + qc/curation preserved.
        for f in preserved:
            assert f.exists(), f

    def test_predicted_cache_survives(self, tmp_path):
        _, preserved = _seed_intermediates_workspace(tmp_path)
        rm.clean_run_intermediates(tmp_path, "default", run_path_id=None)
        for f in preserved:
            if "patch_e" in f.name:
                assert f.exists(), f

    def test_named_run_scope(self, tmp_path):
        recon_files, _ = _seed_intermediates_workspace(
            tmp_path, run_path_id="cells", slug="cells"
        )
        res = rm.clean_run_intermediates(
            tmp_path, "cells", run_path_id="cells"
        )
        assert res["n_cleared"] == len(recon_files)

    def test_missing_dir_never_raises(self, tmp_path):
        res = rm.clean_run_intermediates(
            tmp_path, "default", run_path_id=None
        )
        assert res == {"n_cleared": 0, "bytes_freed": 0}

    def test_idempotent_second_call_zero(self, tmp_path):
        _seed_intermediates_workspace(tmp_path)
        rm.clean_run_intermediates(tmp_path, "default", run_path_id=None)
        res2 = rm.clean_run_intermediates(tmp_path, "default", run_path_id=None)
        assert res2 == {"n_cleared": 0, "bytes_freed": 0}
