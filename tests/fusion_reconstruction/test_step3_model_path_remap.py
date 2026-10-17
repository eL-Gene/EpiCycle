"""A copied Step 3 workspace must find its own models again.

The stored model paths are absolute.  After a workspace folder is copied
to another machine they point at a folder that is gone, discovery then
silently picks the newest model file, and Predict runs with the wrong
model.  Restore now re-roots a missing path onto the current workspace
when the tail after ``artifacts/step2`` or ``artifacts/step3`` exists
there, all or nothing, and saves the healed paths.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)

OLD = "D:\\labdata\\Src"
S2_TAIL = ("artifacts", "step2", "runs", "Cell", "models", "m_lat")
S3_TAIL = ("artifacts", "step3", "runs", "cells", "refinement", "models",
           "init_model", "m_ax")


def _win(root, tail):
    return root + "\\" + "\\".join(tail)


def _make(ws: Path, tail) -> Path:
    p = ws.joinpath(*tail)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"w")
    return p


# ---------------------------------------------------------------- pure

def test_step3_anchor_remaps(tmp_path):
    f = _make(tmp_path, S3_TAIL)
    assert _paths.remap_model_path_to_workspace(
        _win(OLD, S3_TAIL), tmp_path) == f


def test_step2_anchor_remaps_posix_separators(tmp_path):
    f = _make(tmp_path, S2_TAIL)
    old = "/data/lab/Src/" + "/".join(S2_TAIL)
    assert _paths.remap_model_path_to_workspace(old, tmp_path) == f


def test_anchor_case_insensitive(tmp_path):
    f = _make(tmp_path, S2_TAIL)
    old = "D:\\labdata\\Src\\Artifacts\\STEP2\\runs\\Cell\\models\\m_lat"
    got = _paths.remap_model_path_to_workspace(old, tmp_path)
    assert got == f
    # The rebuilt path is spelled the way the plugin writes the folders, so
    # it exists on a case-sensitive filesystem too; the stored spelling only
    # locates the anchor.
    assert got.parts[-6:-4] == ("artifacts", "step2")


def test_runs_alone_is_not_an_anchor(tmp_path):
    _make(tmp_path, ("runs", "Cell", "m"))
    assert _paths.remap_model_path_to_workspace(
        "D:\\x\\runs\\Cell\\m", tmp_path) is None


def test_no_anchor_returns_none(tmp_path):
    _make(tmp_path, ("models", "m"))
    assert _paths.remap_model_path_to_workspace(
        "D:\\x\\models\\m", tmp_path) is None


def test_anchor_but_file_absent_returns_none(tmp_path):
    assert _paths.remap_model_path_to_workspace(
        _win(OLD, S3_TAIL), tmp_path) is None


def test_relative_and_bare_names_return_none(tmp_path, monkeypatch):
    _make(tmp_path, S3_TAIL)
    monkeypatch.chdir(tmp_path)
    assert _paths.remap_model_path_to_workspace(
        "/".join(S3_TAIL), tmp_path) is None
    assert _paths.remap_model_path_to_workspace("cyto3", tmp_path) is None
    assert _paths.remap_model_path_to_workspace("", tmp_path) is None
    assert _paths.remap_model_path_to_workspace(None, tmp_path) is None


def test_last_anchor_wins(tmp_path):
    """A path whose old root itself holds an ``artifacts/step3`` pair
    still re-roots from the last pair."""
    f = _make(tmp_path, S3_TAIL)
    old = "D:\\artifacts\\step3\\Src\\" + "\\".join(S3_TAIL)
    assert _paths.remap_model_path_to_workspace(old, tmp_path) == f


def test_same_workspace_returns_none(tmp_path):
    f = _make(tmp_path, S3_TAIL)
    assert _paths.remap_model_path_to_workspace(str(f), tmp_path) is None


def test_live_old_workspace_returns_none(tmp_path):
    old_ws = tmp_path / "Old"
    old_ws.mkdir()
    (old_ws / _paths.STEP3_STATE_FILENAME).write_text("{}")
    new_ws = tmp_path / "New"
    _make(new_ws, S3_TAIL)
    old_path = str(old_ws.joinpath(*S3_TAIL))
    assert _paths.remap_model_path_to_workspace(old_path, new_ws) is None


def test_old_root_spellings_compare_equal():
    a = _paths.model_path_old_root("D:\\Lab\\Src\\artifacts\\step3\\x")
    b = _paths.model_path_old_root("d:/lab/src/artifacts/step3/x")
    assert a == b and a is not None
    assert _paths.model_path_old_root("cyto3") is None


def test_old_root_present_without_step3_state_is_refused(tmp_path):
    """Any old root still on this machine is a live workspace, even one
    holding only a Step 2 state file."""
    old_ws = tmp_path / "Old"
    old_ws.mkdir()
    (old_ws / "step2_state.json").write_text("{}")
    new_ws = tmp_path / "New"
    _make(new_ws, S3_TAIL)
    old_path = str(old_ws.joinpath(*S3_TAIL))
    assert _paths.remap_model_path_to_workspace(old_path, new_ws) is None


def test_old_root_absent_heals(tmp_path):
    new_ws = tmp_path / "New"
    f = _make(new_ws, S3_TAIL)
    old_path = str((tmp_path / "Gone").joinpath(*S3_TAIL))
    assert _paths.remap_model_path_to_workspace(old_path, new_ws) == f


def test_dot_segments_in_tail_return_none(tmp_path):
    _make(tmp_path, S3_TAIL)
    up = OLD + '\\artifacts\\step3\\runs\\..\\..\\..\\x'
    assert _paths.remap_model_path_to_workspace(up, tmp_path) is None


def test_unc_spellings_share_an_old_root():
    a = _paths.model_path_old_root(
        '\\\\srv\\sh\\ws\\artifacts\\step3\\runs\\m')
    b = _paths.model_path_old_root("//srv/sh/ws/artifacts/step3/runs/m")
    assert a is not None and a == b


def test_trailing_separator_heals(tmp_path):
    f = _make(tmp_path, S3_TAIL)
    assert _paths.remap_model_path_to_workspace(
        _win(OLD, S3_TAIL) + '\\', tmp_path) == f


def test_drive_relative_and_root_only_return_none(tmp_path):
    _make(tmp_path, S3_TAIL)
    assert _paths.remap_model_path_to_workspace(
        "C:old\\" + "\\".join(S3_TAIL), tmp_path) is None
    assert _paths.remap_model_path_to_workspace(
        "\\old\\" + "\\".join(S3_TAIL), tmp_path) is None


# ---------------------------------------------------------- controller

def _ctrl(ws: Path, sm):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    ctrl = C.__new__(C)
    texts = {"lat": "", "ax": ""}

    def field(key):
        w = MagicMock()
        w.setText = lambda t, k=key: texts.__setitem__(k, t)
        w.text = lambda k=key: texts[k]
        return w

    ctrl.ui = types.SimpleNamespace(
        lateral_model_path=field("lat"), axial_model_path=field("ax"),
        status_lateral_model_value=MagicMock(),
        status_axial_model_value=MagicMock())
    ctrl._log = []
    ctrl._append_log = lambda m: ctrl._log.append(m)
    ctrl.working_dir = str(ws)
    ctrl.state_manager = sm
    ctrl._refresh_model_status = lambda: None
    ctrl._persist_current_model_selection = types.MethodType(
        C._persist_current_model_selection, ctrl)
    ctrl._restore = types.MethodType(
        C._restore_persisted_model_selection, ctrl)
    ctrl._texts = texts
    return ctrl


def _sm(ws: Path, lat, ax, mode="import"):
    sm = ReconstructionStateManager()
    sm.load(str(ws))
    sm.set_model_selection(mode=mode, lateral_path=lat, axial_path=ax)
    return sm


def test_stale_pair_is_healed_and_persisted(tmp_path):
    ws = tmp_path / "New"
    lat = _make(ws, S2_TAIL)
    ax = _make(ws, S3_TAIL)
    sm = _sm(ws, _win(OLD, S2_TAIL), _win(OLD, S3_TAIL))
    ctrl = _ctrl(ws, sm)
    assert ctrl._restore() is True
    assert ctrl._texts == {"lat": str(lat), "ax": str(ax)}
    sel = sm.get_model_selection()
    assert sel["lateral_path"] == str(lat)
    assert sel["axial_path"] == str(ax)
    assert sel["mode"] == "import"
    assert any("[MODEL-RESTORE] re-rooted" in m for m in ctrl._log)


def test_only_one_healable_means_no_heal(tmp_path):
    ws = tmp_path / "New"
    _make(ws, S2_TAIL)  # axial tail absent
    stale_lat, stale_ax = _win(OLD, S2_TAIL), _win(OLD, S3_TAIL)
    sm = _sm(ws, stale_lat, stale_ax)
    ctrl = _ctrl(ws, sm)
    assert ctrl._restore() is False
    assert ctrl._texts == {"lat": "", "ax": ""}
    sel = sm.get_model_selection()
    assert sel["lateral_path"] == stale_lat
    assert sel["axial_path"] == stale_ax
    assert not any("re-rooted" in m for m in ctrl._log)


def test_different_old_roots_do_not_heal(tmp_path):
    ws = tmp_path / "New"
    _make(ws, S2_TAIL)
    _make(ws, S3_TAIL)
    sm = _sm(ws, _win(OLD, S2_TAIL), _win("E:\\other\\Src", S3_TAIL))
    ctrl = _ctrl(ws, sm)
    assert ctrl._restore() is False
    assert ctrl._texts == {"lat": "", "ax": ""}


def test_no_anchor_missing_path_still_returns_false(tmp_path):
    sm = _sm(tmp_path, "/nonexistent/missing.npy", None)
    ctrl = _ctrl(tmp_path, sm)
    assert ctrl._restore() is False
    assert ctrl._texts == {"lat": "", "ax": ""}
    assert sm.get_model_selection()["lateral_path"] == "/nonexistent/missing.npy"


def test_existing_path_beside_healable_one(tmp_path):
    ws = tmp_path / "New"
    lat = _make(ws, S2_TAIL)
    keep = tmp_path / "elsewhere_model"
    keep.write_bytes(b"w")
    sm = _sm(ws, _win(OLD, S2_TAIL), str(keep))
    ctrl = _ctrl(ws, sm)
    assert ctrl._restore() is True
    assert ctrl._texts == {"lat": str(lat), "ax": str(keep)}


def test_bare_pretrained_name_beside_healable_one(tmp_path):
    ws = tmp_path / "New"
    lat = _make(ws, S2_TAIL)
    sm = _sm(ws, _win(OLD, S2_TAIL), "cyto3")
    ctrl = _ctrl(ws, sm)
    assert ctrl._restore() is True
    assert ctrl._texts == {"lat": str(lat), "ax": "cyto3"}


def test_empty_working_dir_no_heal(tmp_path):
    ws = tmp_path / "New"
    _make(ws, S2_TAIL)
    sm = _sm(ws, _win(OLD, S2_TAIL), None)
    ctrl = _ctrl(ws, sm)
    ctrl.working_dir = ""
    assert ctrl._restore() is False
    assert ctrl._texts == {"lat": "", "ax": ""}
    assert sm.get_model_selection()["lateral_path"] == _win(OLD, S2_TAIL)


def test_persist_failure_does_not_break_restore(tmp_path):
    """A setter that raises (a legacy mode such as ``browse``) must not
    break the heal: it goes through the guarded persist helper."""
    ws = tmp_path / "New"
    lat = _make(ws, S2_TAIL)
    sm = _sm(ws, _win(OLD, S2_TAIL), None)

    def picky(mode=None, lateral_path=None, axial_path=None):
        raise ValueError("legacy mode")
    sm.set_model_selection = picky
    ctrl = _ctrl(ws, sm)
    assert ctrl._restore() is True
    assert ctrl._texts["lat"] == str(lat)


def test_adopt_under_suppress_save_stays_in_memory(tmp_path):
    ws = tmp_path / "New"
    lat = _make(ws, S2_TAIL)
    sm = _sm(ws, _win(OLD, S2_TAIL), None)
    sm.suppress_save = True
    ctrl = _ctrl(ws, sm)
    assert ctrl._restore() is True
    assert sm.get_model_selection()["lateral_path"] == str(lat)
    sm.save()  # a read-only adopt writes nothing
    state_file = ws / _paths.STEP3_STATE_FILENAME
    on_disk = (state_file.read_text(encoding="utf-8")
               if state_file.exists() else "")
    assert json.dumps(str(lat))[1:-1] not in on_disk


def test_normal_save_writes_the_heal(tmp_path):
    ws = tmp_path / "New"
    lat = _make(ws, S2_TAIL)
    sm = _sm(ws, _win(OLD, S2_TAIL), None)
    ctrl = _ctrl(ws, sm)
    assert ctrl._restore() is True
    sm.save()
    text = (ws / _paths.STEP3_STATE_FILENAME).read_text(encoding="utf-8")
    assert json.dumps(str(lat))[1:-1] in text


# a Qt application must outlive the debounced-save timers
_APP = []


def _flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, pre + "/" + k))
    else:
        out[pre] = d
    return out


def _tree_digest(root: Path) -> dict:
    import hashlib
    return {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _two_run_workspace(root: Path):
    a = root / "A"
    for t in (S2_TAIL, S3_TAIL):
        _make(a, t)
    sm = ReconstructionStateManager()
    sm.load(str(a))
    sm.register_patches(["p1"])
    sm.set_model_selection(mode="import",
                           lateral_path=str(a.joinpath(*S2_TAIL)),
                           axial_path=str(a.joinpath(*S3_TAIL)))
    sm.save()
    sm.create_run("other")
    sm.switch_active_run("other")
    sm.set_model_selection(mode="pretrained", lateral_path="cyto3",
                           axial_path=str(a.joinpath(*S2_TAIL)))
    sm.save()
    sm.switch_active_run(sorted(set(sm.list_run_ids()) - {"other"})[0])
    sm.flush()
    sm.save()
    return a


def test_copied_two_run_workspace_heals_only_active_run(tmp_path):
    import shutil
    from qtpy.QtWidgets import QApplication
    _APP.append(QApplication.instance() or QApplication([]))
    a = _two_run_workspace(tmp_path)
    b = tmp_path / "B"
    shutil.copytree(a, b)
    gone = tmp_path / "A_gone"
    a.rename(gone)
    gone_digest = _tree_digest(gone)
    before = json.loads(
        (b / _paths.STEP3_STATE_FILENAME).read_text(encoding="utf-8"))

    sm2 = ReconstructionStateManager()
    sm2.load(str(b))
    ctrl = _ctrl(b, sm2)
    discovered = []
    ctrl._discover_workspace_models = lambda *a, **k: discovered.append(1)
    assert ctrl._restore() is True
    assert discovered == []
    assert ctrl._texts == {"lat": str(b.joinpath(*S2_TAIL)),
                           "ax": str(b.joinpath(*S3_TAIL))}
    sm2.flush()
    after = json.loads(
        (b / _paths.STEP3_STATE_FILENAME).read_text(encoding="utf-8"))
    fb, fa = _flat(before), _flat(after)
    changed = {k for k in set(fb) | set(fa) if fb.get(k) != fa.get(k)}
    run = "/runs/" + before["active_run_id"]
    assert changed == {"/last_updated", run + "/lateral_model_path",
                       run + "/axial_model_path"}
    assert sm2.get_model_selection()["mode"] == "import"
    assert _tree_digest(gone) == gone_digest


def test_stored_step2_run_mode_does_not_break_the_heal(tmp_path):
    import shutil
    from qtpy.QtWidgets import QApplication
    _APP.append(QApplication.instance() or QApplication([]))
    a = _two_run_workspace(tmp_path)
    b = tmp_path / "B"
    shutil.copytree(a, b)
    a.rename(tmp_path / "A_gone")
    state_file = b / _paths.STEP3_STATE_FILENAME
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["runs"][data["active_run_id"]]["model_source_mode"] = "step2_run"
    state_file.write_text(json.dumps(data), encoding="utf-8")

    sm2 = ReconstructionStateManager()
    sm2.load(str(b))
    ctrl = _ctrl(b, sm2)
    assert ctrl._restore() is True
    assert ctrl._texts["lat"] == str(b.joinpath(*S2_TAIL))
    sm2.flush()
    after = json.loads(state_file.read_text(encoding="utf-8"))
    run = after["runs"][after["active_run_id"]]
    assert run["lateral_model_path"] == str(b.joinpath(*S2_TAIL))
