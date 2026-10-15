"""Tests for ``packages/epicycle_io/step3_run_manifest.py``.

Pure-Python (no Qt / napari) — the Step 3 mirror of Step 2's
run-manifest schema/read/write/scanner.

Covers: build/write/read round-trip; corrupt → None; the run-context
discriminator + scanner (incl. corrupt-manifest dirs yielded,
default-run patch-recon dirs excluded); init-spec builders; validation
of backend + created_from mode.
"""

from __future__ import annotations

import json

import pytest

from packages.epicycle_io import step3_run_manifest as m


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_filename(self):
        assert m.RUN_MANIFEST_FILENAME == "run_manifest.json"

    def test_schema_version(self):
        assert m.RUN_MANIFEST_SCHEMA_VERSION == "1.0"

    def test_valid_init_modes(self):
        assert m.VALID_INIT_MODES == frozenset(
            {"pretrained", "step2_run", "step3_refined", "browse"}
        )
        # Step 3 has NO scratch / inherited / cross_bucket / fork.
        for absent in ("scratch", "inherited", "cross_bucket", "fork"):
            assert absent not in m.VALID_INIT_MODES

    def test_backends(self):
        assert "cellpose" in m.VALID_BACKENDS
        assert {"stardist", "omnipose"}.issubset(m.VALID_BACKENDS)
        assert m.ENABLED_BACKENDS == frozenset({"cellpose"})

    def test_valid_targets(self):
        assert {"", "cells", "nuclei", "membrane", "general"}.issubset(
            m.VALID_TARGETS
        )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

class TestBuild:

    def test_minimal_build(self):
        man = m.build_step3_run_manifest(run_id="cells", display_name="Cells")
        assert man["schema_version"] == "1.0"
        assert man["run_id"] == "cells"
        assert man["display_name"] == "Cells"
        assert man["target"] == ""
        assert man["backend"] == "cellpose"
        assert man["archived_at"] is None
        assert man["created_from"] is None
        assert man["created_at"].endswith("Z")
        # model block normalized to its 3 canonical keys
        assert set(man["model"]) == {
            "lateral_model_path", "axial_model_path", "model_source_mode"
        }

    def test_build_with_all_fields(self):
        spec = m.init_spec_pretrained(pretrained_name="cyto3")
        man = m.build_step3_run_manifest(
            run_id="nuclei",
            display_name="Nuclei",
            target="nuclei",
            description="nuclear seg",
            backend="cellpose",
            created_from=spec,
            model={
                "lateral_model_path": "/a/lat",
                "axial_model_path": "/a/ax",
                "model_source_mode": "pretrained",
            },
            notes="hello",
        )
        assert man["target"] == "nuclei"
        assert man["description"] == "nuclear seg"
        assert man["notes"] == "hello"
        assert man["created_from"]["mode"] == "pretrained"
        assert man["created_from"]["pretrained_name"] == "cyto3"
        # optional created_from keys filled with None
        assert man["created_from"]["external_path"] is None
        assert man["model"]["lateral_model_path"] == "/a/lat"
        assert man["model"]["model_source_mode"] == "pretrained"

    def test_build_rejects_empty_run_id(self):
        with pytest.raises(ValueError):
            m.build_step3_run_manifest(run_id="", display_name="X")

    def test_build_rejects_empty_display_name(self):
        with pytest.raises(ValueError):
            m.build_step3_run_manifest(run_id="x", display_name="")

    def test_build_rejects_bad_backend(self):
        with pytest.raises(ValueError):
            m.build_step3_run_manifest(
                run_id="x", display_name="X", backend="torchseg"
            )

    def test_build_rejects_bad_created_from_mode(self):
        with pytest.raises(ValueError):
            m.build_step3_run_manifest(
                run_id="x", display_name="X",
                created_from={"mode": "scratch"},
            )

    def test_build_rejects_created_from_without_mode(self):
        with pytest.raises(ValueError):
            m.build_step3_run_manifest(
                run_id="x", display_name="X", created_from={"foo": "bar"},
            )


# ---------------------------------------------------------------------------
# Write / read round-trip
# ---------------------------------------------------------------------------

class TestWriteRead:

    def test_round_trip(self, tmp_path):
        man = m.build_step3_run_manifest(
            run_id="cells", display_name="Cells", target="cells",
            created_from=m.init_spec_browse(external_path="/m/weights"),
        )
        path = tmp_path / m.RUN_MANIFEST_FILENAME
        m.write_run_manifest(path, man)
        assert path.is_file()
        loaded = m.read_run_manifest(path)
        assert loaded == man

    def test_write_requires_existing_parent(self, tmp_path):
        man = m.build_step3_run_manifest(run_id="x", display_name="X")
        missing = tmp_path / "nope" / m.RUN_MANIFEST_FILENAME
        with pytest.raises(OSError):
            m.write_run_manifest(missing, man)

    def test_write_does_not_mkdir(self, tmp_path):
        # The caller owns the directory; write must not create it.
        man = m.build_step3_run_manifest(run_id="x", display_name="X")
        target_dir = tmp_path / "run_dir"
        with pytest.raises(OSError):
            m.write_run_manifest(target_dir / m.RUN_MANIFEST_FILENAME, man)
        assert not target_dir.exists()

    def test_write_rejects_missing_required_key(self, tmp_path):
        path = tmp_path / m.RUN_MANIFEST_FILENAME
        with pytest.raises(ValueError):
            m.write_run_manifest(path, {"run_id": "x"})  # no schema_version etc.

    def test_read_missing_returns_none(self, tmp_path):
        assert m.read_run_manifest(tmp_path / "absent.json") is None

    def test_read_corrupt_returns_none(self, tmp_path):
        path = tmp_path / m.RUN_MANIFEST_FILENAME
        path.write_text("{ this is not json", encoding="utf-8")
        assert m.read_run_manifest(path) is None

    def test_read_non_object_returns_none(self, tmp_path):
        path = tmp_path / m.RUN_MANIFEST_FILENAME
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert m.read_run_manifest(path) is None


# ---------------------------------------------------------------------------
# Init-spec builders
# ---------------------------------------------------------------------------

class TestInitSpecBuilders:

    def test_pretrained(self):
        s = m.init_spec_pretrained(pretrained_name="cyto3")
        assert s == {"mode": "pretrained", "pretrained_name": "cyto3"}

    def test_pretrained_requires_name(self):
        with pytest.raises(ValueError):
            m.init_spec_pretrained(pretrained_name="")

    def test_step2_run(self):
        s = m.init_spec_step2_run(source_step2_run_id="default")
        assert s == {"mode": "step2_run", "source_step2_run_id": "default"}

    def test_step2_run_requires_id(self):
        with pytest.raises(ValueError):
            m.init_spec_step2_run(source_step2_run_id="")

    def test_step3_refined(self):
        s = m.init_spec_step3_refined(source_step3_bundle="bundleA")
        # Name-only spec stays backward-compatible: no run_id key.
        assert s == {"mode": "step3_refined", "source_step3_bundle": "bundleA"}

    def test_step3_refined_with_run_id(self):
        s = m.init_spec_step3_refined(
            source_step3_bundle="bundleA", source_step3_run_id="runA"
        )
        assert s == {
            "mode": "step3_refined",
            "source_step3_bundle": "bundleA",
            "source_step3_run_id": "runA",
        }

    def test_step3_refined_empty_run_id_omits_key(self):
        # An empty/None run_id means the flat default → key omitted.
        for rid in (None, ""):
            s = m.init_spec_step3_refined(
                source_step3_bundle="bundleA", source_step3_run_id=rid
            )
            assert "source_step3_run_id" not in s

    def test_step3_refined_run_id_survives_build_normalize(self):
        spec = m.init_spec_step3_refined(
            source_step3_bundle="bundleA", source_step3_run_id="runA"
        )
        man = m.build_step3_run_manifest(
            run_id="r", display_name="R", created_from=spec
        )
        cf = man["created_from"]
        assert cf["source_step3_run_id"] == "runA"
        # Name-only spec → normalizer fills the key with None (no KeyError).
        spec2 = m.init_spec_step3_refined(source_step3_bundle="bundleA")
        man2 = m.build_step3_run_manifest(
            run_id="r", display_name="R", created_from=spec2
        )
        assert man2["created_from"]["source_step3_run_id"] is None

    def test_step3_refined_requires_bundle(self):
        with pytest.raises(ValueError):
            m.init_spec_step3_refined(source_step3_bundle="")

    def test_browse(self):
        s = m.init_spec_browse(external_path="/path/to/weights")
        assert s["mode"] == "browse"
        assert s["external_path"] == "/path/to/weights"
        assert s["external_label"] == "weights"  # derived from filename

    def test_browse_explicit_label(self):
        s = m.init_spec_browse(external_path="/p/w", external_label="My Model")
        assert s["external_label"] == "My Model"

    def test_browse_requires_path(self):
        with pytest.raises(ValueError):
            m.init_spec_browse(external_path="")

    def test_builders_pass_through_build(self):
        # Every builder produces a created_from build() accepts.
        for spec in (
            m.init_spec_pretrained(pretrained_name="cyto3"),
            m.init_spec_step2_run(source_step2_run_id="default"),
            m.init_spec_step3_refined(source_step3_bundle="b"),
            m.init_spec_browse(external_path="/p/w"),
        ):
            man = m.build_step3_run_manifest(
                run_id="r", display_name="R", created_from=spec
            )
            assert man["created_from"]["mode"] == spec["mode"]


# ---------------------------------------------------------------------------
# Run-context discrimination + scanner
# ---------------------------------------------------------------------------

def _make_run_context(runs_root, slug, **manifest_kwargs):
    """Helper: create runs/{slug}/run_manifest.json."""
    run_dir = runs_root / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    man = m.build_step3_run_manifest(
        run_id=slug, display_name=manifest_kwargs.get("display_name", slug),
        **{k: v for k, v in manifest_kwargs.items() if k != "display_name"}
    )
    m.write_run_manifest(run_dir / m.RUN_MANIFEST_FILENAME, man)
    return run_dir


def _make_default_patch_recon(runs_root, patch_name):
    """Helper: create a default-run per-patch recon dir (NO run_manifest)."""
    p = runs_root / patch_name
    p.mkdir(parents=True, exist_ok=True)
    (p / "labels_3d.tif").write_bytes(b"\x00")
    (p / "manifest.json").write_text("{}", encoding="utf-8")  # patch manifest, not run
    return p


class TestRunContextDiscriminator:

    def test_run_context_dir_true(self, tmp_path):
        rd = _make_run_context(tmp_path / "runs", "cells")
        assert m.is_run_context_dir(rd) is True

    def test_patch_recon_dir_false(self, tmp_path):
        pd = _make_default_patch_recon(tmp_path / "runs", "patch_001")
        # Has a manifest.json but NOT run_manifest.json → NOT a run context.
        assert m.is_run_context_dir(pd) is False

    def test_nonexistent_false(self, tmp_path):
        assert m.is_run_context_dir(tmp_path / "absent") is False

    def test_corrupt_manifest_still_run_context(self, tmp_path):
        # Present-but-corrupt run_manifest.json STILL marks a run context.
        rd = tmp_path / "runs" / "broken"
        rd.mkdir(parents=True)
        (rd / m.RUN_MANIFEST_FILENAME).write_text("{ nope", encoding="utf-8")
        assert m.is_run_context_dir(rd) is True


class TestScanner:

    def test_scanner_lists_run_contexts_only(self, tmp_path):
        runs_root = tmp_path / "runs"
        _make_run_context(runs_root, "cells")
        _make_run_context(runs_root, "nuclei")
        _make_default_patch_recon(runs_root, "patch_001")
        _make_default_patch_recon(runs_root, "patch_002")

        entries = m.list_runs_with_manifests(runs_root)
        ids = sorted(e["run_id"] for e in entries)
        # patch_001 / patch_002 are NOT yielded.
        assert ids == ["cells", "nuclei"]
        assert all(e["manifest_ok"] for e in entries)

    def test_scanner_yields_corrupt_manifest_dirs(self, tmp_path):
        runs_root = tmp_path / "runs"
        _make_run_context(runs_root, "good")
        broken = runs_root / "broken"
        broken.mkdir(parents=True)
        (broken / m.RUN_MANIFEST_FILENAME).write_text("{ bad", encoding="utf-8")

        entries = {e["run_id"]: e for e in m.list_runs_with_manifests(runs_root)}
        assert set(entries) == {"good", "broken"}
        assert entries["good"]["manifest_ok"] is True
        assert entries["broken"]["manifest_ok"] is False
        assert entries["broken"]["manifest"] is None

    def test_scanner_empty_root(self, tmp_path):
        assert m.list_runs_with_manifests(tmp_path / "no_runs_here") == []

    def test_scanner_skips_dotfiles(self, tmp_path):
        runs_root = tmp_path / "runs"
        _make_run_context(runs_root, "cells")
        hidden = runs_root / ".hidden"
        hidden.mkdir(parents=True)
        (hidden / m.RUN_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
        ids = [e["run_id"] for e in m.list_runs_with_manifests(runs_root)]
        assert ids == ["cells"]


class TestIsRunArchived:

    def test_none(self):
        assert m.is_run_archived(None) is False

    def test_not_dict(self):
        assert m.is_run_archived("nope") is False

    def test_no_field(self):
        assert m.is_run_archived({"run_id": "x"}) is False

    def test_archived(self):
        assert m.is_run_archived({"archived_at": "2026-06-27T00:00:00Z"}) is True

    def test_null_field(self):
        assert m.is_run_archived({"archived_at": None}) is False
