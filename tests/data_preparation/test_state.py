"""Tests for the Step 1 state manager (step1_state.json).

Pure-Python — no Qt, no napari. Covers schema, atomic write, corrupt-tolerant
load, source dedup-by-path provenance, and patch registration.
"""
import json
import os
import tempfile

import pytest

from napari_epicycle.data_preparation._paths import (
    step1_state_backup_path,
    step1_state_path,
)
from napari_epicycle.data_preparation._state import (
    STEP1_SCHEMA_VERSION,
    _classify_voxel_source,
    all_patches,
    empty_state,
    load_state,
    normalize_box_style,
    patches_for_source,
    register_patch,
    register_source,
    remove_patch,
    resolve_box_style,
    save_state,
    set_corrupt_state_hook,
    set_patch_box_style,
    set_source_box_style,
    upsert_source,
)


# ---------------------------------------------------------------------------
# empty_state
# ---------------------------------------------------------------------------
class TestEmptyState:
    def test_shape(self):
        st = empty_state()
        assert st["schema_version"] == STEP1_SCHEMA_VERSION
        assert st["sources"] == []
        assert st["patches"] == []
        assert isinstance(st["settings"], dict)

    def test_independent_instances(self):
        a, b = empty_state(), empty_state()
        a["patches"].append({"x": 1})
        assert b["patches"] == []  # no shared mutable default


# ---------------------------------------------------------------------------
# load / save round-trip + atomicity + corruption tolerance
# ---------------------------------------------------------------------------
class TestLoadSave:
    def test_load_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as ws:
            st = load_state(ws)
            assert st == empty_state()

    def test_save_then_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            st["settings"]["last_prefix"] = "kidney_patch"
            save_state(ws, st)
            assert step1_state_path(ws).exists()
            loaded = load_state(ws)
            assert loaded["settings"]["last_prefix"] == "kidney_patch"
            assert loaded["schema_version"] == STEP1_SCHEMA_VERSION

    def test_save_is_atomic_no_tmp_left(self):
        with tempfile.TemporaryDirectory() as ws:
            save_state(ws, empty_state())
            leftovers = [f for f in os.listdir(ws) if f.endswith(".tmp")]
            assert leftovers == []

    def test_save_creates_workspace_dir(self):
        with tempfile.TemporaryDirectory() as parent:
            ws = os.path.join(parent, "new_ws")
            save_state(ws, empty_state())
            assert step1_state_path(ws).exists()

    def test_load_corrupt_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("{not valid json", encoding="utf-8")
            assert load_state(ws) == empty_state()

    def test_load_non_dict_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("[1, 2, 3]", encoding="utf-8")
            assert load_state(ws) == empty_state()

    def test_load_normalizes_missing_keys(self):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            st = load_state(ws)
            assert st["sources"] == []
            assert st["patches"] == []
            assert isinstance(st["settings"], dict)


# ---------------------------------------------------------------------------
# register_source — provenance, dedup by path
# ---------------------------------------------------------------------------
class TestRegisterSource:
    def test_creates_source_with_fields(self):
        st = empty_state()
        sid = register_source(
            st, path="D:/data/kidney", display_name="kidney",
            axes="CZYX", voxel_size=None, timestamp="2026-06-29T00:00:00Z",
        )
        assert len(st["sources"]) == 1
        src = st["sources"][0]
        assert src["source_id"] == sid
        assert src["path"] == "D:/data/kidney"
        assert src["display_name"] == "kidney"
        # schema v2: the on-disk order lives under `source_axes` (anti-lie home).
        assert src["source_axes"] == "CZYX"
        assert src["voxel_size"] is None
        assert src["voxel_size_source"] == "unknown"   # no voxel -> honest typing
        assert src["origin"] == ["exported"]           # export path provenance
        assert src["added_at"] == "2026-06-29T00:00:00Z"

    def test_dedup_same_path_returns_same_id(self):
        st = empty_state()
        sid1 = register_source(st, path="D:/data/kidney", display_name="kidney",
                               axes="CZYX", timestamp="t1")
        sid2 = register_source(st, path="D:/data/kidney", display_name="kidney",
                               axes="CZYX", timestamp="t2")
        assert sid1 == sid2
        assert len(st["sources"]) == 1  # not duplicated

    def test_dedup_trailing_slash(self):
        # normpath strips trailing separators on every platform.
        st = empty_state()
        sid1 = register_source(st, path="/data/kidney/", display_name="k",
                               axes="CZYX", timestamp="t1")
        sid2 = register_source(st, path="/data/kidney", display_name="k",
                               axes="CZYX", timestamp="t2")
        assert sid1 == sid2
        assert len(st["sources"]) == 1

    @pytest.mark.skipif(
        os.name != "nt", reason="backslash is a path separator only on Windows"
    )
    def test_dedup_separator_and_case_windows(self):
        # On Windows, normcase lowercases AND converts / -> \, so these dedup.
        st = empty_state()
        sid1 = register_source(st, path="D:/data/Kidney/", display_name="k",
                               axes="CZYX", timestamp="t1")
        sid2 = register_source(st, path="D:\\data\\kidney", display_name="k",
                               axes="CZYX", timestamp="t2")
        assert sid1 == sid2
        assert len(st["sources"]) == 1

    def test_different_path_new_id(self):
        st = empty_state()
        sid1 = register_source(st, path="D:/data/kidney", display_name="k",
                               axes="CZYX", timestamp="t1")
        sid2 = register_source(st, path="D:/data/liver", display_name="l",
                               axes="CZYX", timestamp="t2")
        assert sid1 != sid2
        assert len(st["sources"]) == 2

    def test_stores_numbering_prefix(self):
        """Commit 3 (D-3): the source persists the numbering prefix its patches
        were exported under, so the badge can match custom-named patches."""
        st = empty_state()
        register_source(st, path="D:/d/kidney.tif", display_name="kidney",
                        axes="CZYX", timestamp="t", prefix="kidney_patch")
        assert st["sources"][0]["prefix"] == "kidney_patch"

    def test_prefix_updated_on_reregister_last_wins(self):
        """Revisiting a file + extracting under a changed name updates the badge
        prefix (last wins), but keeps the original identity/metadata."""
        st = empty_state()
        register_source(st, path="D:/d/kidney.tif", display_name="kidney",
                        axes="CZYX", timestamp="t1", prefix="kidney_patch")
        register_source(st, path="D:/d/kidney.tif", display_name="kidney",
                        axes="CZYX", timestamp="t2", prefix="kidney_ant_patch")
        assert len(st["sources"]) == 1
        assert st["sources"][0]["prefix"] == "kidney_ant_patch"
        assert st["sources"][0]["added_at"] == "t1"  # original metadata preserved

    def test_no_prefix_kwarg_omits_field(self):
        st = empty_state()
        register_source(st, path="D:/d/kidney.tif", display_name="kidney",
                        axes="CZYX", timestamp="t")
        assert "prefix" not in st["sources"][0]

    def test_isotropic_default_voxel_typed_unknown_not_user(self):
        """Honest typing (invariant #4): a 1/1/1 export must NOT be recorded as
        an authoritative "user" spacing — else it masks real spacing downstream."""
        st = empty_state()
        register_source(st, path="D:/d", display_name="d", axes="CZYX",
                        voxel_size=[1.0, 1.0, 1.0], timestamp="t")
        assert st["sources"][0]["voxel_size_source"] == "unknown"

    def test_real_voxel_typed_user(self):
        st = empty_state()
        register_source(st, path="D:/d", display_name="d", axes="CZYX",
                        voxel_size=[2.0, 0.5, 0.5], timestamp="t")
        assert st["sources"][0]["voxel_size_source"] == "user"


# ---------------------------------------------------------------------------
# _classify_voxel_source — honest voxel typing (schema-v2 invariant #4)
# ---------------------------------------------------------------------------
class TestClassifyVoxelSource:
    def test_none_is_unknown(self):
        assert _classify_voxel_source(None) == "unknown"

    def test_isotropic_default_is_unknown(self):
        assert _classify_voxel_source([1.0, 1.0, 1.0]) == "unknown"
        assert _classify_voxel_source((1, 1, 1)) == "unknown"

    def test_real_voxel_is_user(self):
        assert _classify_voxel_source([2.0, 0.5, 0.5]) == "user"
        assert _classify_voxel_source([1.0, 1.0, 0.5]) == "user"   # one axis real

    def test_malformed_or_empty_is_unknown(self):
        assert _classify_voxel_source([]) == "unknown"
        assert _classify_voxel_source("bad") == "unknown"
        assert _classify_voxel_source([1.0, None, 1.0]) == "unknown"


# ---------------------------------------------------------------------------
# register_patch — provenance link + index
# ---------------------------------------------------------------------------
class TestRegisterPatch:
    def test_appends_patch_with_fields(self):
        st = empty_state()
        register_patch(
            st, patch_name="kidney_patch_1",
            relative_path="artifacts/step1/patches/kidney_patch_1.tiff",
            sidecar_path="artifacts/step1/patches/kidney_patch_1_metadata.json",
            source_id="src_0", timestamp="2026-06-29T00:01:00Z",
        )
        assert len(st["patches"]) == 1
        p = st["patches"][0]
        assert p["patch_name"] == "kidney_patch_1"
        assert p["relative_path"].endswith("kidney_patch_1.tiff")
        assert p["sidecar_path"].endswith("kidney_patch_1_metadata.json")
        assert p["source_id"] == "src_0"
        assert p["extracted_at"] == "2026-06-29T00:01:00Z"

    def test_reregister_same_name_replaces(self):
        st = empty_state()
        register_patch(st, patch_name="p_1", relative_path="a.tiff",
                       sidecar_path="a.json", source_id="src_0", timestamp="t1")
        register_patch(st, patch_name="p_1", relative_path="b.tiff",
                       sidecar_path="b.json", source_id="src_1", timestamp="t2")
        assert len(st["patches"]) == 1
        assert st["patches"][0]["relative_path"] == "b.tiff"
        assert st["patches"][0]["source_id"] == "src_1"

    def test_roundtrip_preserves_registrations(self):
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            sid = register_source(st, path="D:/d", display_name="d",
                                  axes="CZYX", timestamp="t0")
            register_patch(st, patch_name="d_1", relative_path="d_1.tiff",
                           sidecar_path="d_1.json", source_id=sid, timestamp="t1")
            save_state(ws, st)
            loaded = load_state(ws)
            assert len(loaded["sources"]) == 1
            assert len(loaded["patches"]) == 1
            assert loaded["patches"][0]["source_id"] == sid


# ---------------------------------------------------------------------------
# remove_patch — delete a patch registry row (the ④ bin's schema half)
# ---------------------------------------------------------------------------
class TestRemovePatch:
    def _two(self):
        st = empty_state()
        register_patch(st, patch_name="p_1", relative_path="p_1.tiff",
                       sidecar_path="p_1.json", source_id="src_0", timestamp="t1")
        register_patch(st, patch_name="p_2", relative_path="p_2.tiff",
                       sidecar_path="p_2.json", source_id="src_0", timestamp="t2")
        return st

    def test_removes_the_named_row_only(self):
        st = self._two()
        assert remove_patch(st, "p_1") is True
        assert [p["patch_name"] for p in st["patches"]] == ["p_2"]

    def test_case_insensitive_match(self):
        st = self._two()
        assert remove_patch(st, "P_1") is True   # NTFS case-fold parity
        assert [p["patch_name"] for p in st["patches"]] == ["p_2"]

    def test_returns_false_when_absent(self):
        st = self._two()
        assert remove_patch(st, "nope") is False
        assert len(st["patches"]) == 2

    def test_tolerates_missing_patches_key(self):
        assert remove_patch({}, "p_1") is False


# ---------------------------------------------------------------------------
# Audit-driven hardening (atomicity / contract / schema audits, 2026-06-29)
# ---------------------------------------------------------------------------
class TestAuditHardening:
    def test_load_filters_non_dict_list_entries(self):
        """Hand-edited garbage in the lists must not crash later .get() calls."""
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text(
                json.dumps({
                    "schema_version": 1,
                    "sources": [{"source_id": "src_0", "path": "D:/d"}, 1, None, "x"],
                    "patches": [{"patch_name": "p"}, 42],
                    "settings": {},
                }),
                encoding="utf-8",
            )
            st = load_state(ws)
            # The surviving source row is field-upgraded to v2 in memory
            # (additive: voxel_size_source + origin), keeping its own keys.
            assert st["sources"] == [{
                "source_id": "src_0", "path": "D:/d",
                "voxel_size_source": "unknown", "origin": ["exported"],
            }]
            assert st["patches"] == [{"patch_name": "p"}]
            # And a subsequent register must not crash on the cleaned lists.
            register_source(st, path="D:/other", display_name="o", axes="CZYX",
                            timestamp="t")

    def test_load_preserves_future_schema_unknown_keys(self):
        """v(future) file loaded by v1 code must not drop unknown keys on save."""
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text(
                json.dumps({
                    "schema_version": 99,
                    "sources": [],
                    "patches": [],
                    "settings": {},
                    "future_key": {"keep": "me"},
                }),
                encoding="utf-8",
            )
            st = load_state(ws)
            assert st["future_key"] == {"keep": "me"}
            save_state(ws, st)
            reloaded = load_state(ws)
            assert reloaded["future_key"] == {"keep": "me"}

    def test_current_schema_does_not_leak_unknown_keys(self):
        """At the current schema version, stray keys are NOT preserved."""
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text(
                json.dumps({"schema_version": 1, "bogus": 1}), encoding="utf-8"
            )
            assert "bogus" not in load_state(ws)

    def test_saved_file_is_valid_loadable_json(self):
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            register_source(st, path="D:/d", display_name="d", axes="CZYX",
                            timestamp="t")
            save_state(ws, st)
            on_disk = json.loads(step1_state_path(ws).read_text(encoding="utf-8"))
            assert on_disk["sources"][0]["path"] == "D:/d"

    def test_register_source_rejects_empty_path(self):
        st = empty_state()
        with pytest.raises(ValueError):
            register_source(st, path="", display_name="x", axes="CZYX")
        with pytest.raises(ValueError):
            register_source(st, path="   ", display_name="x", axes="CZYX")

    def test_register_source_rejects_none_path(self):
        st = empty_state()
        with pytest.raises(ValueError):
            register_source(st, path=None, display_name="x", axes="CZYX")

    def test_register_source_dedup_keeps_first_metadata(self):
        """Export path is NON-CLOBBERING: re-registering the
        same path keeps the original provenance (display_name / source_axes /
        added_at) untouched — only `prefix` would update."""
        st = empty_state()
        register_source(st, path="D:/d", display_name="first", axes="CZYX",
                        timestamp="t1")
        register_source(st, path="D:/d", display_name="second", axes="ZCYX",
                        timestamp="t2")
        assert len(st["sources"]) == 1
        assert st["sources"][0]["display_name"] == "first"
        assert st["sources"][0]["source_axes"] == "CZYX"
        assert st["sources"][0]["added_at"] == "t1"

    def test_next_source_id_skips_gaps(self):
        st = empty_state()
        st["sources"] = [{"source_id": "src_0"}, {"source_id": "src_2"}]
        sid = register_source(st, path="D:/new", display_name="n", axes="CZYX",
                              timestamp="t")
        assert sid == "src_3"

    def test_register_patch_rejects_absolute_paths(self):
        st = empty_state()
        with pytest.raises(ValueError, match="workspace-relative"):
            register_patch(st, patch_name="p", relative_path="C:\\abs\\p.tiff",
                           sidecar_path="rel.json", source_id="src_0")
        with pytest.raises(ValueError, match="workspace-relative"):
            register_patch(st, patch_name="p", relative_path="rel.tiff",
                           sidecar_path="/abs/p.json", source_id="src_0")

    def test_register_patch_removes_preexisting_duplicates(self):
        """If the registry already has duplicate names, re-register collapses them."""
        st = empty_state()
        st["patches"] = [
            {"patch_name": "p1", "relative_path": "a.tiff"},
            {"patch_name": "p1", "relative_path": "b.tiff"},
        ]
        register_patch(st, patch_name="p1", relative_path="c.tiff",
                       sidecar_path="c.json", source_id="src_0", timestamp="t")
        names = [p["patch_name"] for p in st["patches"]]
        assert names.count("p1") == 1
        assert st["patches"][-1]["relative_path"] == "c.tiff"

    def test_register_patch_dedup_is_case_insensitive(self):
        """Commit 1 (CASE): NTFS holds one file for case-variants, so the
        registry must not accrue two rows for `Kidney_Patch_1` / `kidney_patch_1`."""
        st = empty_state()
        register_patch(st, patch_name="Kidney_Patch_1", relative_path="a.tiff",
                       sidecar_path="a.json", source_id="", timestamp="t1")
        register_patch(st, patch_name="kidney_patch_1", relative_path="b.tiff",
                       sidecar_path="b.json", source_id="", timestamp="t2")
        assert len(st["patches"]) == 1
        assert st["patches"][0]["patch_name"] == "kidney_patch_1"  # later wins
        assert st["patches"][0]["relative_path"] == "b.tiff"


# ---------------------------------------------------------------------------
# Schema v1 -> v2 migration (additive, lossless, field-only, NO I/O)
# ---------------------------------------------------------------------------
class TestSchemaV2Migration:
    def _write_v1(self, ws, sources, extra=None):
        payload = {
            "schema_version": 1,
            "sources": sources,
            "patches": [],
            "settings": {},
        }
        if extra:
            payload.update(extra)
        step1_state_path(ws).write_text(json.dumps(payload), encoding="utf-8")

    def test_v1_source_row_upgrades_fields_in_memory(self):
        with tempfile.TemporaryDirectory() as ws:
            self._write_v1(ws, [{
                "source_id": "src_0", "path": "D:/d/k.tif",
                "display_name": "k", "axes": "ZCYX",
                "voxel_size": [2.0, 0.5, 0.5], "added_at": "t0",
            }])
            src = load_state(ws)["sources"][0]
            # renamed (legacy `axes` KEPT for a v1 reader; `source_axes` added)
            assert src["source_axes"] == "ZCYX"
            assert src["axes"] == "ZCYX"
            # a present voxel is honestly typed as user-provided
            assert src["voxel_size_source"] == "user"
            assert src["origin"] == ["exported"]

    def test_v1_row_without_voxel_is_unknown(self):
        with tempfile.TemporaryDirectory() as ws:
            self._write_v1(ws, [{
                "source_id": "src_0", "path": "D:/d/k.tif",
                "display_name": "k", "axes": "CZYX",
            }])
            src = load_state(ws)["sources"][0]
            assert src["voxel_size_source"] == "unknown"

    def test_migration_preserves_unknown_per_entry_keys(self):
        """Forward-compat: a hand-added / future per-row key survives the
        in-memory upgrade (the row dict is mutated, never rebuilt)."""
        with tempfile.TemporaryDirectory() as ws:
            self._write_v1(ws, [{
                "source_id": "src_0", "path": "D:/d/k.tif",
                "axes": "ZYX", "future_field": {"keep": 1},
            }])
            src = load_state(ws)["sources"][0]
            assert src["future_field"] == {"keep": 1}

    def test_v1_to_v2_roundtrip_is_lossless(self):
        """Load a v1 file (auto-upgrade in memory), save, reload: the v2 fields
        persist and nothing is lost."""
        with tempfile.TemporaryDirectory() as ws:
            self._write_v1(ws, [{
                "source_id": "src_0", "path": "D:/d/k.tif",
                "display_name": "k", "axes": "ZCYX",
                "voxel_size": [3.0, 1.0, 1.0], "prefix": "k_patch",
            }])
            st = load_state(ws)
            assert st["schema_version"] == 2          # auto-upgraded in memory
            save_state(ws, st)                        # a save trigger persists it

            reloaded = load_state(ws)
            r = reloaded["sources"][0]
            assert reloaded["schema_version"] == 2
            assert r["source_axes"] == "ZCYX"
            assert r["voxel_size"] == [3.0, 1.0, 1.0]
            assert r["voxel_size_source"] == "user"
            assert r["origin"] == ["exported"]
            assert r["prefix"] == "k_patch"           # untouched

    def test_migration_is_idempotent_on_v2_row(self):
        """A row already carrying v2 fields is not re-defaulted (origin kept)."""
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text(json.dumps({
                "schema_version": 2,
                "sources": [{
                    "source_id": "src_0", "path": "D:/d/k.tif",
                    "source_axes": "ZCYX", "voxel_size_source": "detected",
                    "origin": ["browsed", "prepared"],
                }],
                "patches": [], "settings": {},
            }), encoding="utf-8")
            src = load_state(ws)["sources"][0]
            assert src["voxel_size_source"] == "detected"   # NOT re-defaulted
            assert src["origin"] == ["browsed", "prepared"]


# ---------------------------------------------------------------------------
# upsert_source — LAST-WINS mutator (the explicit edit / Prepare signals)
# ---------------------------------------------------------------------------
class TestUpsertSource:
    def test_creates_row_and_allocates_id(self):
        st = empty_state()
        sid = upsert_source(st, path="D:/d/k.tif", display_name="k",
                            source_axes="ZCYX")
        assert sid == "src_0"
        assert st["sources"][0]["display_name"] == "k"
        assert st["sources"][0]["source_axes"] == "ZCYX"
        assert st["sources"][0]["path"] == "D:/d/k.tif"

    def test_last_wins_overwrites_user_fields(self):
        st = empty_state()
        upsert_source(st, path="D:/d/k.tif", display_name="old",
                      source_axes="CZYX")
        sid2 = upsert_source(st, path="D:/d/k.tif", display_name="new",
                             source_axes="ZCYX")
        assert len(st["sources"]) == 1               # same row (dedup by path)
        assert st["sources"][0]["source_id"] == sid2
        assert st["sources"][0]["display_name"] == "new"   # last wins
        assert st["sources"][0]["source_axes"] == "ZCYX"

    def test_none_value_does_not_clobber(self):
        st = empty_state()
        upsert_source(st, path="D:/d/k.tif", voxel_size=[2.0, 0.5, 0.5])
        upsert_source(st, path="D:/d/k.tif", voxel_size=None,
                      display_name="k")   # absent voxel must not wipe the stored one
        assert st["sources"][0]["voxel_size"] == [2.0, 0.5, 0.5]
        assert st["sources"][0]["display_name"] == "k"

    def test_origin_is_unioned_not_replaced(self):
        st = empty_state()
        upsert_source(st, path="D:/d/k.tif", origin=["browsed"])
        upsert_source(st, path="D:/d/k.tif", origin=["prepared"])
        upsert_source(st, path="D:/d/k.tif", origin="prepared")   # dup ignored
        assert st["sources"][0]["origin"] == ["browsed", "prepared"]

    def test_locate_by_source_id(self):
        st = empty_state()
        upsert_source(st, path="D:/d/k.tif", source_id="src_5",
                      display_name="k")
        # a later call keyed by id (no path) finds the same row
        upsert_source(st, source_id="src_5", display_name="renamed")
        assert len(st["sources"]) == 1
        assert st["sources"][0]["display_name"] == "renamed"

    def test_requires_path_or_source_id(self):
        st = empty_state()
        with pytest.raises(ValueError):
            upsert_source(st, display_name="x")


# ---------------------------------------------------------------------------
# all_patches / patches_for_source — read-only registry queries (F2)
# ---------------------------------------------------------------------------
def _state_with_patches():
    st = empty_state()
    register_patch(st, patch_name="a_1", relative_path="artifacts/step1/patches/a_1.tiff",
                   sidecar_path="artifacts/step1/patches/a_1_metadata.json", source_id="src_0")
    register_patch(st, patch_name="a_2", relative_path="artifacts/step1/patches/a_2.tiff",
                   sidecar_path="artifacts/step1/patches/a_2_metadata.json", source_id="src_0")
    register_patch(st, patch_name="b_1", relative_path="artifacts/step1/patches/b_1.tiff",
                   sidecar_path="artifacts/step1/patches/b_1_metadata.json", source_id="src_1")
    register_patch(st, patch_name="legacy_1", relative_path="artifacts/step1/patches/legacy_1.tiff",
                   sidecar_path="artifacts/step1/patches/legacy_1_metadata.json", source_id="")
    return st


class TestAllPatches:
    def test_returns_every_row(self):
        st = _state_with_patches()
        names = {p["patch_name"] for p in all_patches(st)}
        assert names == {"a_1", "a_2", "b_1", "legacy_1"}

    def test_returns_fresh_list_not_the_stored_one(self):
        st = _state_with_patches()
        got = all_patches(st)
        got.clear()                       # mutating the result must not…
        assert len(st["patches"]) == 4    # …touch the stored registry

    def test_tolerates_missing_key_and_garbage(self):
        assert all_patches({}) == []
        assert all_patches({"patches": ["not-a-dict", 3]}) == []
        assert all_patches(None) == []


class TestPatchesForSource:
    def test_filters_by_source_id(self):
        st = _state_with_patches()
        got = {p["patch_name"] for p in patches_for_source(st, "src_0")}
        assert got == {"a_1", "a_2"}
        got = {p["patch_name"] for p in patches_for_source(st, "src_1")}
        assert got == {"b_1"}

    def test_unknown_source_is_empty(self):
        assert patches_for_source(_state_with_patches(), "src_99") == []

    def test_falsy_source_id_is_empty_not_legacy(self):
        # Legacy patches carry source_id == "" and must NOT all collapse into
        # one "source" — an empty id returns nothing, never every legacy row.
        st = _state_with_patches()
        assert patches_for_source(st, "") == []
        assert patches_for_source(st, None) == []


class TestBoxStylePersistence:
    """Per-patch/source ``box_style`` (P2 Brick 2, schema-v2 additive): colour +
    opacity + shape only (width is GLOBAL); patch → source-default → None."""

    def _state_with_patch(self):
        st = empty_state()
        sid = upsert_source(st, path="C:/data/img.tif")
        register_patch(
            st, patch_name="img_patch_1",
            relative_path="patches/img_patch_1.tiff",
            sidecar_path="patches/img_patch_1.json", source_id=sid)
        return st, sid

    def test_normalize_rgb_promotes_to_rgba_and_clamps(self):
        assert normalize_box_style(
            {"color": [2.0, -1.0, 0.5]})["color"] == [1.0, 0.0, 0.5, 1.0]

    def test_normalize_status_shape_opacity(self):
        s = normalize_box_style(
            {"color": "status", "opacity": 1.5, "shape": "wireframe"})
        assert s == {"color": "status", "opacity": 1.0, "shape": "wireframe"}

    def test_normalize_drops_width_junk_and_invalid(self):
        # width is GLOBAL — never a per-box field; junk keys + bad shape dropped.
        assert normalize_box_style({"width": 5.0, "junk": 1}) is None
        assert normalize_box_style({"shape": "dashed"}) is None
        assert normalize_box_style(None) is None
        assert normalize_box_style("nope") is None

    def test_set_and_resolve_patch_style(self):
        st, _ = self._state_with_patch()
        assert set_patch_box_style(
            st, "img_patch_1", {"color": [0.1, 0.2, 0.3, 1.0], "shape": "wireframe"})
        got = resolve_box_style(st, "img_patch_1")
        assert got["color"] == [0.1, 0.2, 0.3, 1.0] and got["shape"] == "wireframe"

    def test_set_patch_style_missing_row_is_false(self):
        st, _ = self._state_with_patch()
        assert set_patch_box_style(st, "nope", {"color": "status"}) is False

    def test_resolve_falls_through_to_source_default(self):
        st, sid = self._state_with_patch()
        set_source_box_style(st, sid, {"color": [0.9, 0.9, 0.1, 1.0]})
        assert resolve_box_style(st, "img_patch_1")["color"] == [0.9, 0.9, 0.1, 1.0]

    def test_patch_style_overrides_source_default(self):
        st, sid = self._state_with_patch()
        set_source_box_style(st, sid, {"color": [0.9, 0.9, 0.1, 1.0]})
        set_patch_box_style(st, "img_patch_1", {"color": [0.1, 0.1, 0.1, 1.0]})
        assert resolve_box_style(st, "img_patch_1")["color"] == [0.1, 0.1, 0.1, 1.0]

    def test_clear_patch_style_falls_back_to_source(self):
        st, sid = self._state_with_patch()
        set_source_box_style(st, sid, {"color": "status"})
        set_patch_box_style(st, "img_patch_1", {"color": [0.1, 0.1, 0.1, 1.0]})
        set_patch_box_style(st, "img_patch_1", None)   # clear
        assert resolve_box_style(st, "img_patch_1") == {"color": "status"}

    def test_resolve_unknown_patch_is_none(self):
        st, _ = self._state_with_patch()
        assert resolve_box_style(st, "ghost") is None

    def test_persists_across_save_load(self, tmp_path):
        st, _ = self._state_with_patch()
        set_patch_box_style(
            st, "img_patch_1", {"color": [0.2, 0.4, 0.6, 0.8], "shape": "wireframe"})
        ws = str(tmp_path)
        save_state(ws, st)
        got = resolve_box_style(load_state(ws), "img_patch_1")
        assert got["color"] == [0.2, 0.4, 0.6, 0.8] and got["shape"] == "wireframe"


# ---------------------------------------------------------------------------
# corrupt-file quarantine + .bak rotation
# ---------------------------------------------------------------------------
class TestCorruptQuarantine:
    """A truncated / unparseable ``step1_state.json`` used to load as EMPTY and
    be silently overwritten by the next edit — losing every source, voxel,
    recipe, store pointer and patch registration.  Load stays tolerant, but the
    bad file is RENAMED aside first and the caller is told."""

    def _corrupt_files(self, ws):
        return [f for f in os.listdir(ws) if ".corrupt-" in f]

    def test_corrupt_file_is_renamed_aside(self):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text('{"sources": [', encoding="utf-8")
            st = load_state(ws)
            assert st == empty_state()
            # the bad file no longer occupies the canonical name ...
            assert not step1_state_path(ws).exists()
            # ... it was preserved under a timestamped .corrupt- name
            quarantined = self._corrupt_files(ws)
            assert len(quarantined) == 1
            assert quarantined[0].startswith("step1_state.json.corrupt-")
            saved = os.path.join(ws, quarantined[0])
            assert open(saved, encoding="utf-8").read() == '{"sources": ['

    def test_corrupt_load_reports_to_the_caller(self):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("{ truncated", encoding="utf-8")
            seen = []
            st = load_state(ws, on_corrupt=seen.append)
            assert st == empty_state()
            assert len(seen) == 1
            assert "corrupt-" in seen[0]

    def test_the_report_points_at_the_backup(self):
        """Audit note 3: a corrupt state is exactly when the user needs to be
        told a recoverable predecessor exists, and where it now lives."""
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("{ truncated", encoding="utf-8")
            seen = []
            load_state(ws, on_corrupt=seen.append)
            assert "artifacts/step1/state_backup/" in seen[0], seen[0]
            assert "previous save" in seen[0], seen[0]

    def test_non_dict_json_is_also_quarantined(self):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("[1, 2, 3]", encoding="utf-8")
            seen = []
            assert load_state(ws, on_corrupt=seen.append) == empty_state()
            assert len(self._corrupt_files(ws)) == 1
            assert len(seen) == 1

    def test_healthy_load_never_quarantines_or_reports(self):
        with tempfile.TemporaryDirectory() as ws:
            save_state(ws, empty_state())
            seen = []
            load_state(ws, on_corrupt=seen.append)
            assert seen == []
            assert self._corrupt_files(ws) == []
            assert step1_state_path(ws).exists()

    def test_missing_file_never_reports(self):
        with tempfile.TemporaryDirectory() as ws:
            seen = []
            assert load_state(ws, on_corrupt=seen.append) == empty_state()
            assert seen == []

    def test_load_never_raises_even_if_quarantine_fails(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("{ bad", encoding="utf-8")

            def boom(*a, **kw):
                raise OSError("rename refused")

            monkeypatch.setattr(os, "replace", boom)
            monkeypatch.setattr(os, "rename", boom)

            def raising_cb(_msg):
                raise RuntimeError("callback blew up")

            assert load_state(ws, on_corrupt=raising_cb) == empty_state()


class TestBackupRotation:
    """A save rotates the previous GOOD file to
    ``artifacts/step1/state_backup/step1_state.json.bak`` so a bad write (or a
    later corruption) has a recoverable predecessor.

    B4 moved it out of the workspace ROOT, where
    it was clutter nothing ever read.
    """

    def test_first_save_writes_no_bak(self):
        with tempfile.TemporaryDirectory() as ws:
            save_state(ws, empty_state())
            assert not os.path.exists(str(step1_state_backup_path(ws)))
            assert not os.path.exists(str(step1_state_path(ws)) + ".bak")

    def test_second_save_rotates_previous_to_bak(self):
        with tempfile.TemporaryDirectory() as ws:
            first = empty_state()
            first["settings"]["last_prefix"] = "gen1"
            save_state(ws, first)
            second = empty_state()
            second["settings"]["last_prefix"] = "gen2"
            save_state(ws, second)

            bak = str(step1_state_backup_path(ws))
            assert os.path.exists(bak)
            assert json.load(open(bak, encoding="utf-8"))["settings"][
                "last_prefix"] == "gen1"
            assert load_state(ws)["settings"]["last_prefix"] == "gen2"

    def test_no_bak_is_left_in_the_workspace_root(self):
        with tempfile.TemporaryDirectory() as ws:
            save_state(ws, empty_state())
            save_state(ws, empty_state())
            assert [f for f in os.listdir(ws) if f.endswith(".bak")] == []

    def test_the_backup_folder_is_created_on_demand(self):
        with tempfile.TemporaryDirectory() as ws:
            save_state(ws, empty_state())
            assert not os.path.isdir(
                str(step1_state_backup_path(ws).parent)), (
                "nothing is created until there is something to rotate")
            save_state(ws, empty_state())
            assert os.path.isdir(str(step1_state_backup_path(ws).parent))

    def test_save_failure_still_raises_and_leaves_no_tmp(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws:
            save_state(ws, empty_state())

            real_replace = os.replace

            def flaky(src, dst):
                if str(dst).endswith("step1_state.json"):
                    raise OSError("disk full")
                return real_replace(src, dst)

            monkeypatch.setattr(os, "replace", flaky)
            with pytest.raises(OSError):
                save_state(ws, empty_state())
            assert [f for f in os.listdir(ws) if f.endswith(".tmp")] == []
            assert [f for f in os.listdir(ws) if f.endswith(".bak")] == []


# ---------------------------------------------------------------------------
# module-level corrupt hook (Brick 1b #1)
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_corrupt_hook():
    """Reset the module-level hook around each test (it is process-global)."""
    set_corrupt_state_hook(None)
    yield
    set_corrupt_state_hook(None)


class TestCorruptStateHook:
    """``on_corrupt=`` reaches only ONE of ~25 ``load_state`` call sites, so a
    corrupt state discovered by any of the others was still silent.  A
    module-level hook, registered once by the controller, covers them all."""

    def test_bare_load_invokes_the_hook(self, clean_corrupt_hook):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("{ nope", encoding="utf-8")
            seen = []
            set_corrupt_state_hook(seen.append)
            assert load_state(ws) == empty_state()
            assert len(seen) == 1
            assert "corrupt-" in seen[0]

    def test_explicit_on_corrupt_still_wins(self, clean_corrupt_hook):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("{ nope", encoding="utf-8")
            hook_seen, explicit_seen = [], []
            set_corrupt_state_hook(hook_seen.append)
            load_state(ws, on_corrupt=explicit_seen.append)
            assert len(explicit_seen) == 1
            assert hook_seen == []          # not double-reported

    def test_hook_exceptions_never_propagate(self, clean_corrupt_hook):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("{ nope", encoding="utf-8")

            def boom(_msg):
                raise RuntimeError("hook blew up")

            set_corrupt_state_hook(boom)
            assert load_state(ws) == empty_state()

    def test_healthy_load_never_invokes_the_hook(self, clean_corrupt_hook):
        with tempfile.TemporaryDirectory() as ws:
            save_state(ws, empty_state())
            seen = []
            set_corrupt_state_hook(seen.append)
            load_state(ws)
            assert seen == []

    def test_hook_can_be_cleared(self, clean_corrupt_hook):
        with tempfile.TemporaryDirectory() as ws:
            step1_state_path(ws).write_text("{ nope", encoding="utf-8")
            seen = []
            set_corrupt_state_hook(seen.append)
            set_corrupt_state_hook(None)
            load_state(ws)
            assert seen == []
