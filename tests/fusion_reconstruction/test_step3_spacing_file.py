"""S1: Step 3's per-patch voxel size is backed by the root
``patch_spacing.json``."""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("qtpy")

from napari_epicycle.data_preparation import _voxel_spacing as VS  # noqa: E402
from napari_epicycle.fusion_reconstruction.state_manager import (  # noqa: E402
    ReconstructionStateManager,
)


def _mgr(ws, stack_dir=None):
    m = ReconstructionStateManager()
    m.suppress_save = False
    m.load(str(ws))
    if stack_dir is not None:
        m.set_spacing_stack_dir(str(stack_dir))
    return m


def test_a_write_lands_in_the_file_under_a_relative_key(tmp_path):
    sd = tmp_path / "patches"
    sd.mkdir()
    m = _mgr(tmp_path, sd)
    assert m.set_patch_spacing("a.tif", (2, .5, .5), "user") is True
    got = VS.read_patch_spacing_file(str(tmp_path))
    assert got["patches/a.tif"]["zyx"] == [2.0, .5, .5]
    assert m.get_patch_spacing("a.tif") == {"zyx": [2.0, .5, .5],
                                            "source": "user"}
    assert m.get_all_patch_spacing() == {
        "a.tif": {"zyx": [2.0, .5, .5], "source": "user"}}


def test_another_writer_is_seen_at_once(tmp_path):
    sd = tmp_path / "patches"
    sd.mkdir()
    m = _mgr(tmp_path, sd)
    assert m.get_patch_spacing("a.tif") is None
    VS.write_patch_spacing_entry(str(tmp_path), "patches/a.tif", (3, 1, 1),
                                 "user")                  # "Step 4" writes
    assert m.get_patch_spacing("a.tif")["zyx"] == [3.0, 1.0, 1.0]
    assert m.set_patch_spacing("a.tif", (2, 1, 1), "step1") is False
    assert m.clear_patch_spacing("a.tif", user_too=True) is True
    assert VS.read_patch_spacing_file(str(tmp_path)) == {}


def _envelope(ws, spacing):
    json.dump({"schema_version": 3, "active_run_id": "default",
               "patch_spacing": spacing,
               "runs": {"default": {"patches": {}}}},
              open(os.path.join(str(ws), "step3_state.json"), "w",
                   encoding="utf-8"))


def test_the_envelope_map_migrates_once(tmp_path):
    sd = tmp_path / "patches"
    sd.mkdir()
    _envelope(tmp_path, {"a.tif": {"zyx": [2, .5, .5], "source": "user"},
                         "b.tif": {"zyx": [1.5, .3, .3], "source": "step1"}})
    _mgr(tmp_path, sd)
    got = VS.read_patch_spacing_file(str(tmp_path))
    assert set(got) == {"patches/a.tif", "patches/b.tif"}
    assert got["patches/a.tif"]["source"] == "user"
    # a later load adds nothing that is already there and replaces nothing
    path = VS.patch_spacing_path(str(tmp_path))
    VS.write_patch_spacing_entry(str(tmp_path), "patches/b.tif", (9, 9, 9),
                                 "step1")
    before = path.read_bytes()
    _mgr(tmp_path, sd)
    assert path.read_bytes() == before


def test_a_folder_outside_the_workspace_keys_by_absolute_path(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    out = tmp_path / "elsewhere"
    out.mkdir()
    _envelope(ws, {"a.tif": {"zyx": [2, .5, .5], "source": "user"}})
    m = _mgr(ws, out)
    keys = list(VS.read_patch_spacing_file(str(ws)))
    assert keys == [str(out / "a.tif").replace(os.sep, "/")]
    assert m.get_patch_spacing("a.tif")["zyx"] == [2.0, .5, .5]


def test_no_folder_known_reads_the_envelope_and_writes_no_file(tmp_path):
    _envelope(tmp_path, {"a.tif": {"zyx": [2, .5, .5], "source": "user"}})
    m = _mgr(tmp_path)
    assert m.get_patch_spacing("a.tif")["zyx"] == [2.0, .5, .5]
    assert not VS.patch_spacing_path(str(tmp_path)).exists()


def test_m1_step4_wrote_the_file_first_and_the_envelope_still_migrates(
        tmp_path):
    sd = tmp_path / "patches"
    sd.mkdir()
    VS.write_patch_spacing_entry(str(tmp_path), "patches/z.tif", (4, 1, 1),
                                 "user")                    # Step 4 first
    _envelope(tmp_path, {"a.tif": {"zyx": [2, .5, .5], "source": "user"}})
    m = _mgr(tmp_path, sd)
    assert m.get_patch_spacing("a.tif")["zyx"] == [2.0, .5, .5]
    assert m.get_patch_spacing("z.tif")["zyx"] == [4.0, 1.0, 1.0]


def test_m1_a_failed_entry_is_retried(tmp_path, monkeypatch):
    sd = tmp_path / "patches"
    sd.mkdir()
    _envelope(tmp_path, {"a.tif": {"zyx": [2, .5, .5], "source": "user"}})
    real = VS.write_patch_spacing_entry
    monkeypatch.setattr(VS, "write_patch_spacing_entry",
                        lambda *a, **k: (_ for _ in ()).throw(
                            PermissionError("locked")))
    m = _mgr(tmp_path, sd)
    assert not VS.patch_spacing_path(str(tmp_path)).exists()
    monkeypatch.setattr(VS, "write_patch_spacing_entry", real)
    m.set_spacing_stack_dir(str(sd))                     # tried again
    assert "patches/a.tif" in VS.read_patch_spacing_file(str(tmp_path))


def test_m3_a_read_only_adopt_writes_no_file(tmp_path):
    sd = tmp_path / "patches"
    sd.mkdir()
    _envelope(tmp_path, {"a.tif": {"zyx": [2, .5, .5], "source": "user"}})
    m = ReconstructionStateManager()
    m.suppress_save = True
    m.load(str(tmp_path))
    m.set_spacing_stack_dir(str(sd))
    assert m.set_patch_spacing("b.tif", (3, 1, 1), "step1") is True
    assert m.get_patch_spacing("b.tif")["zyx"] == [3.0, 1.0, 1.0]
    assert m.get_patch_spacing("a.tif")["zyx"] == [2.0, .5, .5]
    assert not VS.patch_spacing_path(str(tmp_path)).exists()
    open(VS.patch_spacing_path(str(tmp_path)), "w").write("{bad")
    m.get_patch_spacing("a.tif")                        # reads, never renames
    assert VS.patch_spacing_path(str(tmp_path)).exists()


def test_m4_a_copied_workspace_with_the_source_still_there(tmp_path):
    import shutil
    src = tmp_path / "src_ws"
    (src / "patches").mkdir(parents=True)
    (src / "patches" / "a.tif").write_bytes(b"x")
    m = _mgr(src, src / "patches")
    m.loaded_stack_dir = str(src / "patches")       # a persisted Browse
    m.set_patch_spacing("a.tif", (2, .5, .5), "user")
    copy = tmp_path / "copy_ws"
    shutil.copytree(src, copy)
    # the copy's saved state still names the ORIGINAL folder
    c = ReconstructionStateManager()
    c.load(str(copy))
    c._state["loaded_stack_dir"] = str(src / "patches")
    assert c.get_patch_spacing("a.tif")["zyx"] == [2.0, .5, .5]
    c.set_patch_spacing("a.tif", (5, 1, 1), "user")
    assert VS.read_patch_spacing_file(str(copy))["patches/a.tif"]["zyx"] ==         [5.0, 1.0, 1.0]
    assert VS.read_patch_spacing_file(str(src))["patches/a.tif"]["zyx"] ==         [2.0, .5, .5]


def test_l4_no_folder_known_refuses_a_write_when_asked(tmp_path):
    m = _mgr(tmp_path)
    m.require_stack_dir = True
    assert m.set_patch_spacing("a.tif", (2, 1, 1), "user") is False
    assert m.last_spacing_error
    assert not VS.patch_spacing_path(str(tmp_path)).exists()


def test_l1_the_cache_follows_a_same_size_replace(tmp_path):
    sd = tmp_path / "patches"
    sd.mkdir()
    m = _mgr(tmp_path, sd)
    m.set_patch_spacing("a.tif", (2, 1, 1), "user")
    assert m.get_patch_spacing("a.tif")["zyx"][0] == 2.0
    VS.write_patch_spacing_entry(str(tmp_path), "patches/a.tif", (3, 1, 1),
                                 "user")                  # same byte length
    assert m.get_patch_spacing("a.tif")["zyx"][0] == 3.0


def test_m2_a_failed_write_is_reported(tmp_path, monkeypatch):
    sd = tmp_path / "patches"
    sd.mkdir()
    m = _mgr(tmp_path, sd)
    monkeypatch.setattr(VS, "write_patch_spacing_entry",
                        lambda *a, **k: (_ for _ in ()).throw(
                            PermissionError("locked")))
    assert m.set_patch_spacing("a.tif", (2, 1, 1), "user") is False
    assert "locked" in m.last_spacing_error


def test_an_old_envelope_only_workspace_still_reads(tmp_path):
    sd = tmp_path / "patches"
    sd.mkdir()
    _envelope(tmp_path, {"a.tif": {"zyx": [2, .5, .5], "source": "user"}})
    m = ReconstructionStateManager()
    m.load(str(tmp_path))                # no stack folder yet
    assert m.get_patch_spacing("a.tif")["zyx"] == [2.0, .5, .5]
