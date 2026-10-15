"""Tests for the reshape-flag sidecar.

The sidecar exists because QC re-extraction preserves ONLY ``qc_tier`` (a
left-merge by label in ``controller._on_qc_extract_finished``); every other
curation-authored column, ``reshape_flag`` included, is dropped.  The
re-extraction survival test at the bottom of this file is the one the spec
explicitly demands: "write a flag, re-extract, assert the flag is still there".
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _reshape_flags as rf
from napari_epicycle.fusion_reconstruction._paths import qc_features_csv


# ----------------------------------------------------------------------------
# Path
# ----------------------------------------------------------------------------

def test_sidecar_sits_beside_features_csv_default_run(tmp_path):
    csv = qc_features_csv(tmp_path, "patch_001")
    side = rf.sidecar_path(tmp_path, "patch_001")
    assert side.parent == csv.parent
    assert side.name == "patch_001" + rf.SIDECAR_NAME_SUFFIX
    assert side.name == "patch_001_reshape_flags.json"


def test_sidecar_sits_beside_features_csv_named_run(tmp_path):
    csv = qc_features_csv(tmp_path, "patch_001", "cells")
    side = rf.sidecar_path(tmp_path, "patch_001", "cells")
    assert side.parent == csv.parent
    assert "cells" in side.parts
    assert side.name == "patch_001_reshape_flags.json"
    # And a named run's sidecar must NOT collide with the default run's.
    assert side != rf.sidecar_path(tmp_path, "patch_001")


def test_sidecar_path_is_pure(tmp_path):
    side = rf.sidecar_path(tmp_path, "patch_001")
    assert not side.exists()
    assert not side.parent.exists()


# ----------------------------------------------------------------------------
# Round trip + int-key normalisation
# ----------------------------------------------------------------------------

def test_write_then_read_round_trip(tmp_path):
    rf.write_flags(tmp_path, "p", {1: "dividing", 7: "low-evidence"})
    assert rf.read_flags(tmp_path, "p") == {1: "dividing", 7: "low-evidence"}


def test_read_returns_int_keys(tmp_path):
    rf.write_flags(tmp_path, "p", {12: "dividing"})
    flags = rf.read_flags(tmp_path, "p")
    assert list(flags) == [12]
    assert all(isinstance(k, int) for k in flags)
    assert not any(isinstance(k, str) for k in flags)


def test_on_disk_keys_are_strings_and_read_back_as_ints(tmp_path):
    """JSON has string-only object keys - the normalisation trap."""
    rf.write_flags(tmp_path, "p", {12: "dividing"})
    payload = json.loads(rf.sidecar_path(tmp_path, "p").read_text(encoding="utf-8"))
    assert payload["schema_version"] == rf.SCHEMA_VERSION
    assert "written_at" in payload
    assert payload["flags"] == {"12": "dividing"}  # strings on disk
    assert rf.read_flags(tmp_path, "p") == {12: "dividing"}  # ints in memory


def test_hand_written_string_keys_read_back_as_ints(tmp_path):
    side = rf.sidecar_path(tmp_path, "p")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "written_at": "2026-09-08T00:00:00+00:00",
                "flags": {"3": "split-override", "9": "dividing"},
            }
        ),
        encoding="utf-8",
    )
    assert rf.read_flags(tmp_path, "p") == {3: "split-override", 9: "dividing"}


def test_string_keys_accepted_on_write(tmp_path):
    rf.write_flags(tmp_path, "p", {"5": "dividing"})
    assert rf.read_flags(tmp_path, "p") == {5: "dividing"}


# ----------------------------------------------------------------------------
# Corrupt tolerance
# ----------------------------------------------------------------------------

def test_missing_file_reads_empty(tmp_path):
    assert rf.read_flags(tmp_path, "never-written") == {}


def test_corrupt_json_reads_empty_and_leaves_file_alone(tmp_path):
    side = rf.sidecar_path(tmp_path, "p")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text("{not valid json", encoding="utf-8")
    assert rf.read_flags(tmp_path, "p") == {}
    assert side.is_file()
    assert side.read_text(encoding="utf-8") == "{not valid json"


def test_unknown_schema_version_reads_empty(tmp_path):
    side = rf.sidecar_path(tmp_path, "p")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(
        json.dumps({"schema_version": 99, "flags": {"1": "dividing"}}),
        encoding="utf-8",
    )
    assert rf.read_flags(tmp_path, "p") == {}


def test_non_dict_payload_reads_empty(tmp_path):
    side = rf.sidecar_path(tmp_path, "p")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert rf.read_flags(tmp_path, "p") == {}


def test_non_dict_flags_reads_empty(tmp_path):
    side = rf.sidecar_path(tmp_path, "p")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(
        json.dumps({"schema_version": 1, "flags": ["dividing"]}), encoding="utf-8"
    )
    assert rf.read_flags(tmp_path, "p") == {}


def test_bad_entries_skipped_good_ones_survive(tmp_path):
    side = rf.sidecar_path(tmp_path, "p")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "flags": {
                    "1": "dividing",          # good
                    "not-an-int": "dividing",  # bad key
                    "2": "banana",             # unknown flag value
                    "3": 17,                   # non-string value
                    "4": "split-override",     # good
                },
            }
        ),
        encoding="utf-8",
    )
    assert rf.read_flags(tmp_path, "p") == {1: "dividing", 4: "split-override"}


# ----------------------------------------------------------------------------
# Write validation + atomicity + empty removal
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["divide", "Dividing", "", None, 3])
def test_write_rejects_invalid_flag_value(tmp_path, bad):
    with pytest.raises(ValueError):
        rf.write_flags(tmp_path, "p", {1: bad})


def test_write_rejects_invalid_flag_before_touching_disk(tmp_path):
    rf.write_flags(tmp_path, "p", {1: "dividing"})
    before = rf.sidecar_path(tmp_path, "p").read_text(encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        rf.write_flags(tmp_path, "p", {1: "dividing", 2: "typo-flag"})
    assert "typo-flag" in str(exc.value)
    # Existing file untouched.
    assert rf.sidecar_path(tmp_path, "p").read_text(encoding="utf-8") == before
    assert rf.read_flags(tmp_path, "p") == {1: "dividing"}


def test_write_rejects_non_integer_label_id(tmp_path):
    with pytest.raises(ValueError):
        rf.write_flags(tmp_path, "p", {"cell-a": "dividing"})


def test_write_rejects_non_mapping(tmp_path):
    with pytest.raises(ValueError):
        rf.write_flags(tmp_path, "p", [(1, "dividing")])


def test_all_valid_flags_are_writable(tmp_path):
    payload = {i: flag for i, flag in enumerate(rf.VALID_FLAGS, start=1)}
    rf.write_flags(tmp_path, "p", payload)
    assert rf.read_flags(tmp_path, "p") == payload
    assert rf.VALID_FLAGS == ("dividing", "low-evidence", "split-override")


def test_write_creates_parent_dirs(tmp_path):
    assert not rf.sidecar_path(tmp_path, "p").parent.exists()
    path = rf.write_flags(tmp_path, "p", {1: "dividing"})
    assert path.is_file()


def test_write_leaves_no_temp_files(tmp_path):
    rf.write_flags(tmp_path, "p", {1: "dividing"})
    rf.write_flags(tmp_path, "p", {2: "dividing"})
    parent = rf.sidecar_path(tmp_path, "p").parent
    assert [p.name for p in parent.iterdir()] == ["p_reshape_flags.json"]


def test_write_empty_removes_sidecar(tmp_path):
    side = rf.write_flags(tmp_path, "p", {1: "dividing"})
    assert side.is_file()
    rf.write_flags(tmp_path, "p", {})
    assert not side.exists()
    assert rf.read_flags(tmp_path, "p") == {}


def test_write_empty_on_missing_sidecar_is_a_noop(tmp_path):
    side = rf.write_flags(tmp_path, "p", {})
    assert not side.exists()


# ----------------------------------------------------------------------------
# merge_flags / drop_flags
# ----------------------------------------------------------------------------

def test_merge_is_additive_across_two_calls(tmp_path):
    first = rf.merge_flags(tmp_path, "p", {1: "dividing"})
    assert first == {1: "dividing"}
    second = rf.merge_flags(tmp_path, "p", {2: "low-evidence"})
    assert second == {1: "dividing", 2: "low-evidence"}
    assert rf.read_flags(tmp_path, "p") == {1: "dividing", 2: "low-evidence"}


def test_merge_is_last_wins(tmp_path):
    rf.merge_flags(tmp_path, "p", {1: "dividing"})
    merged = rf.merge_flags(tmp_path, "p", {1: "split-override"})
    assert merged == {1: "split-override"}
    assert rf.read_flags(tmp_path, "p") == {1: "split-override"}


def test_merge_rejects_invalid_flag_and_keeps_stored(tmp_path):
    rf.merge_flags(tmp_path, "p", {1: "dividing"})
    with pytest.raises(ValueError):
        rf.merge_flags(tmp_path, "p", {2: "nope"})
    assert rf.read_flags(tmp_path, "p") == {1: "dividing"}


def test_merge_onto_corrupt_sidecar_starts_clean(tmp_path):
    side = rf.sidecar_path(tmp_path, "p")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text("garbage", encoding="utf-8")
    merged = rf.merge_flags(tmp_path, "p", {5: "dividing"})
    assert merged == {5: "dividing"}
    assert rf.read_flags(tmp_path, "p") == {5: "dividing"}


def test_drop_removes_only_named_ids(tmp_path):
    rf.write_flags(
        tmp_path, "p", {1: "dividing", 2: "low-evidence", 3: "split-override"}
    )
    remaining = rf.drop_flags(tmp_path, "p", [2])
    assert remaining == {1: "dividing", 3: "split-override"}
    assert rf.read_flags(tmp_path, "p") == {1: "dividing", 3: "split-override"}


def test_drop_ignores_unknown_and_non_integer_ids(tmp_path):
    rf.write_flags(tmp_path, "p", {1: "dividing"})
    remaining = rf.drop_flags(tmp_path, "p", [99, "not-an-int", None])
    assert remaining == {1: "dividing"}


def test_drop_everything_removes_sidecar(tmp_path):
    rf.write_flags(tmp_path, "p", {1: "dividing", 2: "dividing"})
    remaining = rf.drop_flags(tmp_path, "p", [1, 2])
    assert remaining == {}
    assert not rf.sidecar_path(tmp_path, "p").exists()


def test_drop_on_missing_sidecar_is_safe(tmp_path):
    assert rf.drop_flags(tmp_path, "p", [1]) == {}


# ----------------------------------------------------------------------------
# apply_flags_to_df
# ----------------------------------------------------------------------------

def _fresh_df():
    """A stand-in for a freshly extracted QC features df (no reshape_flag)."""
    return pd.DataFrame(
        {
            "label": [1, 2, 3],
            "area": [100.0, 200.0, 300.0],
            "qc_tier": ["good", "review", "bad"],
        }
    )


def test_apply_sets_the_column(tmp_path):
    out = rf.apply_flags_to_df(_fresh_df(), {2: "dividing"})
    assert list(out["reshape_flag"]) == ["", "dividing", ""]


def test_apply_uses_empty_string_not_nan(tmp_path):
    out = rf.apply_flags_to_df(_fresh_df(), {2: "dividing"})
    assert out["reshape_flag"].isna().sum() == 0
    assert out["reshape_flag"].dtype == object
    assert out.loc[out["label"] == 1, "reshape_flag"].iloc[0] == ""


def test_apply_leaves_other_columns_identical():
    df = _fresh_df()
    out = rf.apply_flags_to_df(df, {2: "dividing"})
    for col in ("label", "area", "qc_tier"):
        assert list(out[col]) == list(df[col])
    assert list(out.columns) == ["label", "area", "qc_tier", "reshape_flag"]


def test_apply_does_not_mutate_the_input():
    df = _fresh_df()
    before = df.copy()
    rf.apply_flags_to_df(df, {2: "dividing"})
    assert "reshape_flag" not in df.columns
    pd.testing.assert_frame_equal(df, before)


def test_apply_returns_df_unchanged_when_label_column_missing():
    df = pd.DataFrame({"cell_id": [1, 2], "area": [1.0, 2.0]})
    out = rf.apply_flags_to_df(df, {1: "dividing"})
    assert out is df
    assert "reshape_flag" not in out.columns


def test_apply_with_empty_flags_gives_an_all_blank_column():
    out = rf.apply_flags_to_df(_fresh_df(), {})
    assert list(out["reshape_flag"]) == ["", "", ""]


def test_apply_ignores_flags_for_labels_not_present():
    out = rf.apply_flags_to_df(_fresh_df(), {99: "dividing"})
    assert list(out["reshape_flag"]) == ["", "", ""]


def test_apply_skips_invalid_values_defensively():
    out = rf.apply_flags_to_df(_fresh_df(), {1: "banana", 2: "dividing"})
    assert list(out["reshape_flag"]) == ["", "dividing", ""]


def test_apply_accepts_string_keys():
    out = rf.apply_flags_to_df(_fresh_df(), {"3": "low-evidence"})
    assert list(out["reshape_flag"]) == ["", "", "low-evidence"]


def test_apply_honours_a_custom_column_name():
    out = rf.apply_flags_to_df(_fresh_df(), {1: "dividing"}, column="rf2")
    assert list(out["rf2"]) == ["dividing", "", ""]
    assert "reshape_flag" not in out.columns


def test_apply_overwrites_an_existing_flag_column():
    df = _fresh_df()
    df["reshape_flag"] = ["stale", "stale", "stale"]
    out = rf.apply_flags_to_df(df, {2: "dividing"})
    assert list(out["reshape_flag"]) == ["", "dividing", ""]


def test_apply_none_df_is_none():
    assert rf.apply_flags_to_df(None, {1: "dividing"}) is None


def test_apply_survives_csv_round_trip(tmp_path):
    out = rf.apply_flags_to_df(_fresh_df(), {2: "dividing"})
    csv = tmp_path / "features.csv"
    out.to_csv(csv, index=False)
    back = pd.read_csv(csv, keep_default_na=False)
    assert list(back["reshape_flag"]) == ["", "dividing", ""]


# ----------------------------------------------------------------------------
# The re-extraction survival test the spec demands (grill A15 / B4)
# ----------------------------------------------------------------------------

def test_flag_survives_re_extraction(tmp_path):
    """Write a flag, simulate a re-extraction, assert the flag is still there.

    A re-extraction hands the controller a FRESH DataFrame straight from the
    worker: it has no ``reshape_flag`` column at all, and only ``qc_tier`` is
    carried forward by the controller's left-merge.  Re-applying the sidecar
    is what keeps the arbiter verdict alive.
    """
    # Commit time: the arbiter recorded a verdict.
    rf.merge_flags(tmp_path, "patch_001", {2: "dividing", 3: "split-override"})

    # ... user re-extracts QC features.  The worker's df knows nothing.
    fresh = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "area": [10.0, 20.0, 30.0],
            "label_status": ["active", "active", "active"],
        }
    )
    assert "reshape_flag" not in fresh.columns

    restored = rf.apply_flags_to_df(
        fresh, rf.read_flags(tmp_path, "patch_001")
    )
    assert list(restored["reshape_flag"]) == ["", "dividing", "split-override"]

    # And a SECOND re-extraction still finds them - the sidecar is the source
    # of truth, not the previous df.
    fresh2 = pd.DataFrame({"label": [1, 2, 3], "area": [10.0, 20.0, 30.0]})
    restored2 = rf.apply_flags_to_df(
        fresh2, rf.read_flags(tmp_path, "patch_001")
    )
    assert list(restored2["reshape_flag"]) == ["", "dividing", "split-override"]


def test_flags_are_per_run_isolated(tmp_path):
    rf.write_flags(tmp_path, "patch_001", {1: "dividing"})
    rf.write_flags(tmp_path, "patch_001", {1: "low-evidence"}, "cells")
    assert rf.read_flags(tmp_path, "patch_001") == {1: "dividing"}
    assert rf.read_flags(tmp_path, "patch_001", "cells") == {1: "low-evidence"}


def test_flags_are_per_patch_isolated(tmp_path):
    rf.write_flags(tmp_path, "patch_001", {1: "dividing"})
    assert rf.read_flags(tmp_path, "patch_002") == {}


# ----------------------------------------------------------------------------
# Encoding hygiene (Windows cp1252 console)
# ----------------------------------------------------------------------------

def test_module_source_is_cp1252_encodable():
    src = Path(rf.__file__).read_text(encoding="utf-8")
    src.encode("cp1252")  # raises UnicodeEncodeError if a stray glyph crept in
    src.encode("ascii")
