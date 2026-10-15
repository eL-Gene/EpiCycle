"""Tests for ``packages/epicycle_io/step2_run_manifest.py``.

Pure-Python (no Qt, no controller) tests for the run-manifest read/write
helpers + multi-run discovery scanner.  Covers:

* build_run_manifest validation + field defaults
* All 7 init-spec builders (scratch / inherited / cross_bucket /
  step3_refined / pretrained / browse / fork)
* write_run_manifest atomic + parent-must-exist
* read_run_manifest defensive (missing / corrupt / non-dict)
* list_run_ids / list_runs_with_manifests scanner
* is_run_archived flag handling
* Backend whitelist enforcement
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.epicycle_io.step2_run_manifest import (
    ENABLED_BACKENDS,
    RUN_MANIFEST_FILENAME,
    RUN_MANIFEST_SCHEMA_VERSION,
    VALID_BACKENDS,
    VALID_INIT_MODES,
    build_run_manifest,
    init_spec_browse,
    init_spec_cross_bucket,
    init_spec_fork,
    init_spec_inherited,
    init_spec_pretrained,
    init_spec_scratch,
    init_spec_step3_refined,
    is_run_archived,
    list_run_ids,
    list_runs_with_manifests,
    read_run_manifest,
    write_run_manifest,
)


# ----------------------------------------------------------------------------
# build_run_manifest
# ----------------------------------------------------------------------------

class TestBuildRunManifest:

    def test_minimal_fields_succeed(self):
        m = build_run_manifest(run_id="default", display_name="Default")
        assert m["run_id"] == "default"
        assert m["display_name"] == "Default"
        assert m["backend"] == "cellpose"
        assert m["target"] == ""
        assert m["description"] == ""
        assert m["notes"] == ""
        assert m["archived_at"] is None
        assert m["channels_snapshot"] == []
        assert m["created_from"]["mode"] == "scratch"
        assert m["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION

    def test_full_fields_round_trip(self):
        m = build_run_manifest(
            run_id="nuclei_v1",
            display_name="Nuclei v1",
            target="nuclei",
            description="Pilot run",
            backend="cellpose",
            created_from=init_spec_pretrained(pretrained_name="nuclei"),
            channels_snapshot=["DAPI", "actin"],
            notes="seeded from cyto3",
        )
        assert m["target"] == "nuclei"
        assert m["description"] == "Pilot run"
        assert m["channels_snapshot"] == ["DAPI", "actin"]
        assert m["notes"] == "seeded from cyto3"
        assert m["created_from"]["mode"] == "pretrained"
        assert m["created_from"]["pretrained_name"] == "nuclei"

    def test_empty_run_id_rejected(self):
        with pytest.raises(ValueError, match="run_id"):
            build_run_manifest(run_id="", display_name="Anything")

    def test_none_run_id_rejected(self):
        with pytest.raises(ValueError, match="run_id"):
            build_run_manifest(run_id=None, display_name="Anything")

    def test_empty_display_name_rejected(self):
        with pytest.raises(ValueError, match="display_name"):
            build_run_manifest(run_id="default", display_name="")

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            build_run_manifest(
                run_id="x", display_name="X", backend="madeup"
            )

    def test_all_valid_backends_accepted(self):
        for b in VALID_BACKENDS:
            m = build_run_manifest(run_id="x", display_name="X", backend=b)
            assert m["backend"] == b

    def test_invalid_init_mode_rejected(self):
        with pytest.raises(ValueError, match="VALID_INIT_MODES"):
            build_run_manifest(
                run_id="x", display_name="X",
                created_from={"mode": "telepathy"},
            )

    def test_missing_init_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            build_run_manifest(
                run_id="x", display_name="X",
                created_from={"source_run_id": "default"},
            )

    def test_explicit_created_at_preserved(self):
        m = build_run_manifest(
            run_id="x", display_name="X",
            created_at="2026-01-01T00:00:00Z",
        )
        assert m["created_at"] == "2026-01-01T00:00:00Z"

    def test_archived_at_can_be_set(self):
        m = build_run_manifest(
            run_id="x", display_name="X",
            archived_at="2026-04-15T12:00:00Z",
        )
        assert m["archived_at"] == "2026-04-15T12:00:00Z"

    def test_created_from_optional_fields_normalized_to_none(self):
        """Sparse created_from dicts get filled in with None for all optional fields."""
        m = build_run_manifest(
            run_id="x", display_name="X",
            created_from={"mode": "scratch"},
        )
        cf = m["created_from"]
        # All optional fields should exist, all None for scratch
        for k in ("source_run_id", "source_bucket", "source_round_id",
                  "source_step3_bundle", "source_step3_bucket",
                  "pretrained_name", "external_path", "external_label"):
            assert k in cf, f"missing {k!r} in normalized created_from"
            assert cf[k] is None, f"{k!r} should be None for scratch"

    def test_enabled_backends_subset_of_valid(self):
        assert ENABLED_BACKENDS.issubset(VALID_BACKENDS)
        assert "cellpose" in ENABLED_BACKENDS


# ----------------------------------------------------------------------------
# Init-spec builders
# ----------------------------------------------------------------------------

class TestInitSpecBuilders:

    def test_scratch(self):
        assert init_spec_scratch() == {"mode": "scratch"}

    def test_inherited_happy(self):
        spec = init_spec_inherited(
            source_run_id="default",
            source_bucket="lateral",
            source_round_id="round_03",
        )
        assert spec["mode"] == "inherited"
        assert spec["source_run_id"] == "default"
        assert spec["source_bucket"] == "lateral"
        assert spec["source_round_id"] == "round_03"

    def test_inherited_empty_run_id_rejected(self):
        with pytest.raises(ValueError, match="source_run_id"):
            init_spec_inherited(
                source_run_id="", source_bucket="lateral", source_round_id="r0"
            )

    def test_inherited_bad_bucket_rejected(self):
        with pytest.raises(ValueError, match="source_bucket"):
            init_spec_inherited(
                source_run_id="default", source_bucket="diagonal",
                source_round_id="r0",
            )

    def test_inherited_empty_round_rejected(self):
        with pytest.raises(ValueError, match="source_round_id"):
            init_spec_inherited(
                source_run_id="default", source_bucket="lateral",
                source_round_id="",
            )

    def test_cross_bucket_happy(self):
        spec = init_spec_cross_bucket(source_bucket="axial", source_round_id="round_01")
        assert spec["mode"] == "cross_bucket"
        assert spec["source_bucket"] == "axial"

    def test_cross_bucket_bad_bucket_rejected(self):
        with pytest.raises(ValueError, match="source_bucket"):
            init_spec_cross_bucket(source_bucket="weird", source_round_id="r0")

    def test_step3_refined_happy(self):
        spec = init_spec_step3_refined(
            source_step3_bundle="my_bundle_2026_05_10",
            source_step3_bucket="lateral",
        )
        assert spec["mode"] == "step3_refined"
        assert spec["source_step3_bundle"] == "my_bundle_2026_05_10"

    def test_step3_refined_bad_bucket_rejected(self):
        with pytest.raises(ValueError, match="source_step3_bucket"):
            init_spec_step3_refined(
                source_step3_bundle="b", source_step3_bucket="oblique"
            )

    def test_pretrained_happy(self):
        spec = init_spec_pretrained(pretrained_name="cyto3")
        assert spec == {"mode": "pretrained", "pretrained_name": "cyto3"}

    def test_pretrained_empty_rejected(self):
        with pytest.raises(ValueError, match="pretrained_name"):
            init_spec_pretrained(pretrained_name="")

    def test_browse_happy(self):
        spec = init_spec_browse(
            external_path="/some/abs/path/model.weights",
            external_label="my_external",
        )
        assert spec["mode"] == "browse"
        assert spec["external_path"] == "/some/abs/path/model.weights"
        assert spec["external_label"] == "my_external"

    def test_browse_label_defaults_to_filename(self):
        spec = init_spec_browse(external_path="/some/abs/path/model.weights")
        assert spec["external_label"] == "model.weights"

    def test_browse_empty_path_rejected(self):
        with pytest.raises(ValueError, match="external_path"):
            init_spec_browse(external_path="")

    def test_fork_happy(self):
        spec = init_spec_fork(source_run_id="default", copied_pair_count=42)
        assert spec["mode"] == "fork"
        assert spec["source_run_id"] == "default"
        assert spec["copied_pair_count"] == 42

    def test_fork_empty_source_rejected(self):
        with pytest.raises(ValueError, match="source_run_id"):
            init_spec_fork(source_run_id="")


# ----------------------------------------------------------------------------
# write_run_manifest
# ----------------------------------------------------------------------------

class TestWriteRunManifest:

    def test_round_trip(self, tmp_path):
        run_dir = tmp_path / "runs" / "default"
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / RUN_MANIFEST_FILENAME
        original = build_run_manifest(run_id="default", display_name="Default")
        write_run_manifest(manifest_path, original)
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == original

    def test_atomic_no_tempfile_left_behind_on_success(self, tmp_path):
        run_dir = tmp_path / "runs" / "default"
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / RUN_MANIFEST_FILENAME
        m = build_run_manifest(run_id="default", display_name="Default")
        write_run_manifest(manifest_path, m)
        leftovers = [p.name for p in run_dir.iterdir() if p.name.startswith(".run_manifest_")]
        assert leftovers == [], f"tempfiles left behind: {leftovers}"

    def test_missing_parent_dir_raises(self, tmp_path):
        manifest_path = tmp_path / "nonexistent" / RUN_MANIFEST_FILENAME
        m = build_run_manifest(run_id="x", display_name="X")
        with pytest.raises(OSError, match="parent directory does not exist"):
            write_run_manifest(manifest_path, m)

    def test_missing_required_key_raises(self, tmp_path):
        run_dir = tmp_path / "runs" / "x"
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / RUN_MANIFEST_FILENAME
        bad = {"run_id": "x", "display_name": "X"}  # missing backend, etc.
        with pytest.raises(ValueError, match="missing required key"):
            write_run_manifest(manifest_path, bad)

    def test_overwrite_existing_succeeds(self, tmp_path):
        run_dir = tmp_path / "runs" / "default"
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / RUN_MANIFEST_FILENAME

        m1 = build_run_manifest(run_id="default", display_name="Default v1")
        write_run_manifest(manifest_path, m1)

        m2 = build_run_manifest(run_id="default", display_name="Default v2")
        write_run_manifest(manifest_path, m2)

        loaded = read_run_manifest(manifest_path)
        assert loaded["display_name"] == "Default v2"


# ----------------------------------------------------------------------------
# read_run_manifest
# ----------------------------------------------------------------------------

class TestReadRunManifest:

    def test_missing_file_returns_none(self, tmp_path):
        assert read_run_manifest(tmp_path / "does_not_exist.json") is None

    def test_corrupt_json_returns_none(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not json", encoding="utf-8")
        assert read_run_manifest(bad) is None

    def test_non_dict_returns_none(self, tmp_path):
        list_file = tmp_path / "list.json"
        list_file.write_text("[1, 2, 3]", encoding="utf-8")
        assert read_run_manifest(list_file) is None

    def test_empty_file_returns_none(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("", encoding="utf-8")
        assert read_run_manifest(empty) is None

    def test_valid_manifest_round_trips(self, tmp_path):
        run_dir = tmp_path / "runs" / "default"
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / RUN_MANIFEST_FILENAME
        m = build_run_manifest(run_id="default", display_name="Default")
        write_run_manifest(manifest_path, m)
        loaded = read_run_manifest(manifest_path)
        assert loaded == m


# ----------------------------------------------------------------------------
# Discovery: list_run_ids / list_runs_with_manifests
# ----------------------------------------------------------------------------

class TestListRunIds:

    def test_missing_root_returns_empty(self, tmp_path):
        assert list_run_ids(tmp_path / "nonexistent") == []

    def test_empty_root_returns_empty(self, tmp_path):
        (tmp_path / "runs").mkdir()
        assert list_run_ids(tmp_path / "runs") == []

    def test_lists_subdirs_sorted(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "zebra").mkdir()
        (runs / "alpha").mkdir()
        (runs / "mango").mkdir()
        assert list_run_ids(runs) == ["alpha", "mango", "zebra"]

    def test_ignores_files(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "default").mkdir()
        (runs / "stray_file.json").write_text("{}", encoding="utf-8")
        assert list_run_ids(runs) == ["default"]

    def test_ignores_hidden_dirs(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "default").mkdir()
        (runs / ".cache").mkdir()
        assert list_run_ids(runs) == ["default"]


class TestListRunsWithManifests:

    def test_missing_root_returns_empty(self, tmp_path):
        assert list_runs_with_manifests(tmp_path / "nonexistent") == []

    def test_run_dir_with_valid_manifest(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        default_dir = runs / "default"
        default_dir.mkdir()
        m = build_run_manifest(run_id="default", display_name="Default Run")
        write_run_manifest(default_dir / RUN_MANIFEST_FILENAME, m)

        result = list_runs_with_manifests(runs)
        assert len(result) == 1
        assert result[0]["run_id"] == "default"
        assert result[0]["manifest_ok"] is True
        assert result[0]["manifest"]["display_name"] == "Default Run"

    def test_run_dir_with_missing_manifest_returned_with_ok_false(self, tmp_path):
        """Run folder exists but no manifest — must still appear in list,
        flagged as manifest_ok=False so UI can surface the problem."""
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "default").mkdir()  # no manifest written

        result = list_runs_with_manifests(runs)
        assert len(result) == 1
        assert result[0]["run_id"] == "default"
        assert result[0]["manifest"] is None
        assert result[0]["manifest_ok"] is False

    def test_run_dir_with_corrupt_manifest_returned_with_ok_false(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        default_dir = runs / "default"
        default_dir.mkdir()
        (default_dir / RUN_MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

        result = list_runs_with_manifests(runs)
        assert len(result) == 1
        assert result[0]["manifest_ok"] is False

    def test_multiple_runs_sorted_by_id(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        for slug in ("zebra", "alpha", "mango"):
            d = runs / slug
            d.mkdir()
            m = build_run_manifest(run_id=slug, display_name=slug.title())
            write_run_manifest(d / RUN_MANIFEST_FILENAME, m)

        result = list_runs_with_manifests(runs)
        assert [r["run_id"] for r in result] == ["alpha", "mango", "zebra"]

    def test_manifest_path_always_populated(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "default").mkdir()  # no manifest

        result = list_runs_with_manifests(runs)
        assert result[0]["manifest_path"].endswith(RUN_MANIFEST_FILENAME)
        assert "default" in result[0]["manifest_path"]


# ----------------------------------------------------------------------------
# is_run_archived
# ----------------------------------------------------------------------------

class TestIsRunArchived:

    def test_none_manifest_not_archived(self):
        assert is_run_archived(None) is False

    def test_non_dict_not_archived(self):
        assert is_run_archived("not a dict") is False
        assert is_run_archived([1, 2, 3]) is False

    def test_no_archived_at_field_not_archived(self):
        m = build_run_manifest(run_id="x", display_name="X")
        assert is_run_archived(m) is False

    def test_archived_at_null_not_archived(self):
        m = build_run_manifest(run_id="x", display_name="X")
        m["archived_at"] = None
        assert is_run_archived(m) is False

    def test_archived_at_set_is_archived(self):
        m = build_run_manifest(run_id="x", display_name="X",
                               archived_at="2026-04-15T12:00:00Z")
        assert is_run_archived(m) is True

    def test_empty_string_archived_at_not_archived(self):
        m = build_run_manifest(run_id="x", display_name="X")
        m["archived_at"] = ""
        assert is_run_archived(m) is False
