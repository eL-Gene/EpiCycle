"""S1: the root ``patch_spacing.json``."""
from __future__ import annotations

import json
import os

import pytest

from napari_epicycle.data_preparation import _voxel_spacing as VS


def _file(ws):
    return os.path.join(str(ws), "patch_spacing.json")


def test_round_trip_and_shape(tmp_path):
    assert VS.read_patch_spacing_file(str(tmp_path)) == {}
    assert VS.write_patch_spacing_entry(str(tmp_path), "patches/a.tif",
                                        (2, 0.5, 0.5), "user") is True
    got = VS.read_patch_spacing_file(str(tmp_path))
    assert got["patches/a.tif"]["zyx"] == [2.0, 0.5, 0.5]
    assert got["patches/a.tif"]["source"] == "user"
    raw = json.load(open(_file(tmp_path), encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert "set_at" in raw["patches"]["patches/a.tif"]


def test_same_value_writes_nothing(tmp_path):
    VS.write_patch_spacing_entry(str(tmp_path), "a.tif", (2, 1, 1), "step1")
    before = os.stat(_file(tmp_path)).st_mtime_ns
    assert VS.write_patch_spacing_entry(str(tmp_path), "a.tif", [2.0, 1, 1],
                                        "step1") is False
    assert os.stat(_file(tmp_path)).st_mtime_ns == before


@pytest.mark.parametrize("source", ["step1", "file"])
def test_user_is_never_replaced(tmp_path, source):
    VS.write_patch_spacing_entry(str(tmp_path), "a.tif", (3, .4, .4), "user")
    assert VS.write_patch_spacing_entry(str(tmp_path), "a.tif", (2, .5, .5),
                                        source) is False
    assert VS.read_patch_spacing_file(str(tmp_path))["a.tif"]["zyx"] == \
        [3.0, .4, .4]
    assert VS.write_patch_spacing_entry(str(tmp_path), "a.tif", (4, .4, .4),
                                        "user") is True


def test_reload_before_write_keeps_another_writers_entry(tmp_path):
    VS.write_patch_spacing_entry(str(tmp_path), "a.tif", (2, 1, 1), "user")
    # a second "controller" writes b between our read and write
    raw = json.load(open(_file(tmp_path), encoding="utf-8"))
    raw["patches"]["b.tif"] = {"zyx": [1.5, .3, .3], "source": "user",
                               "set_at": "x"}
    json.dump(raw, open(_file(tmp_path), "w", encoding="utf-8"))
    VS.write_patch_spacing_entry(str(tmp_path), "c.tif", (2, 1, 1), "file")
    assert set(VS.read_patch_spacing_file(str(tmp_path))) == \
        {"a.tif", "b.tif", "c.tif"}


def test_clear(tmp_path):
    VS.write_patch_spacing_entry(str(tmp_path), "a.tif", (2, 1, 1), "user")
    assert VS.clear_patch_spacing_entry(str(tmp_path), "a.tif",
                                        user_too=False) is False
    assert VS.clear_patch_spacing_entry(str(tmp_path), "a.tif",
                                        user_too=True) is True
    assert VS.read_patch_spacing_file(str(tmp_path)) == {}
    assert VS.clear_patch_spacing_entry(str(tmp_path), "zz", user_too=True) \
        is False


def test_corrupt_file_is_renamed_aside_and_empty(tmp_path):
    open(_file(tmp_path), "w").write("{not json")
    assert VS.read_patch_spacing_file(str(tmp_path)) == {}
    assert not os.path.exists(_file(tmp_path))
    assert any(".corrupt-" in n for n in os.listdir(tmp_path))
    assert VS.write_patch_spacing_entry(str(tmp_path), "a.tif", (2, 1, 1),
                                        "user") is True


def test_bad_entries_are_skipped_not_fatal(tmp_path):
    json.dump({"schema_version": 1, "patches": {
        "a.tif": {"zyx": "x", "source": "user"},
        "b.tif": {"zyx": [1, 2, 3], "source": "bogus"},
        "c.tif": {"zyx": [1, 2, 3], "source": "user"}}},
        open(_file(tmp_path), "w", encoding="utf-8"))
    assert list(VS.read_patch_spacing_file(str(tmp_path))) == ["c.tif"]


def test_bad_arguments_raise(tmp_path):
    with pytest.raises(ValueError):
        VS.write_patch_spacing_entry(str(tmp_path), "a", (0, 1, 1), "user")
    with pytest.raises(ValueError):
        VS.write_patch_spacing_entry(str(tmp_path), "a", (1, 1, 1), "qc")


def test_key_is_relative_with_forward_slashes(tmp_path):
    p = tmp_path / "patches" / "a.tif"
    assert VS.patch_spacing_key(str(p), str(tmp_path)) == "patches/a.tif"
    out = os.path.abspath(os.path.join(str(tmp_path), "..", "elsewhere.tif"))
    assert VS.patch_spacing_key(out, str(tmp_path)) == out.replace(os.sep, "/")


# ------------------------------------------------------- S1 audit folds


def test_h1_a_permission_error_is_not_corruption(tmp_path, monkeypatch):
    from pathlib import Path
    for k in ("a", "b", "c"):
        VS.write_patch_spacing_entry(str(tmp_path), f"{k}.tif", (2, 1, 1),
                                     "user")
    real = Path.read_text
    state = {"n": 0}

    def flaky(self, *a, **k):
        if self.name == "patch_spacing.json" and state["n"] == 0:
            state["n"] += 1
            raise PermissionError("locked by another process")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    with pytest.raises(OSError):
        VS.write_patch_spacing_entry(str(tmp_path), "d.tif", (3, 1, 1),
                                     "user")
    monkeypatch.setattr(Path, "read_text", real)
    assert not any(".corrupt-" in n for n in os.listdir(tmp_path))
    assert set(VS.read_patch_spacing_file(str(tmp_path))) == \
        {"a.tif", "b.tif", "c.tif"}


def test_h1_a_plain_read_that_fails_returns_empty_without_renaming(
        tmp_path, monkeypatch):
    from pathlib import Path
    VS.write_patch_spacing_entry(str(tmp_path), "a.tif", (2, 1, 1), "user")
    real = Path.read_text

    def boom(self, *a, **k):
        if self.name == "patch_spacing.json":
            raise PermissionError("no")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", boom)
    assert VS.read_patch_spacing_file(str(tmp_path)) == {}
    monkeypatch.setattr(Path, "read_text", real)
    assert not any(".corrupt-" in n for n in os.listdir(tmp_path))
    assert "a.tif" in VS.read_patch_spacing_file(str(tmp_path))


def test_m3_a_non_renaming_read_leaves_a_corrupt_file(tmp_path):
    open(_file(tmp_path), "w").write("{nope")
    assert VS.read_patch_spacing_file(str(tmp_path), quarantine=False) == {}
    assert os.path.exists(_file(tmp_path))


def test_l2_two_asides_in_one_second_do_not_overwrite(tmp_path, monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "strftime", lambda *a, **k: "20260101-000000")
    for text in ("{one", "{two"):
        open(_file(tmp_path), "w").write(text)
        VS.read_patch_spacing_file(str(tmp_path))
    assert len([n for n in os.listdir(tmp_path) if ".corrupt-" in n]) == 2


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive file systems")
def test_l3_keys_match_case_insensitively_on_windows(tmp_path):
    VS.write_patch_spacing_entry(str(tmp_path), "Patches/A.tif", (2, 1, 1),
                                 "user")
    entries = VS.read_patch_spacing_file(str(tmp_path))
    assert VS.find_patch_spacing(entries, "patches/a.tif") is not None
    assert VS.write_patch_spacing_entry(str(tmp_path), "patches/a.tif",
                                        (2, 1, 1), "user") is False
    assert list(VS.read_patch_spacing_file(str(tmp_path))) == ["Patches/A.tif"]
