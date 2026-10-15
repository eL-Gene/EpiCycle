"""A COPIED workspace must not keep working on the source (live driver,
2026-09-29, HIGH).

Every absolute path Step 4 stores (stack folders, the model, each record's
labels / probability / original / features / border / source paths, the
model snapshots) was kept as written, so a copy's Predict read the
source's TIFFs, Save wrote the source's labels, and Reset / Reconstruct
again DELETED the source's derived files.  Loading now re-roots every path
under the state's former workspace root onto the current one (the root is
recorded at save; for an older file it is inferred from its
``.../artifacts/...`` paths); a path outside that root is left as it is;
and Step 4 never deletes a file outside the current workspace.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from napari_epicycle.segmentation import state_manager as SM


def _tree_digest(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha1(
                p.read_bytes()).hexdigest()
    return out


def _source_workspace(tmp_path, *, with_root_key=True, outside=None):
    a = tmp_path / "A"
    run = a / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    for n in ("labels_3d.tif", "labels_3d_original.tif", "prob.npy"):
        (run / n).write_bytes(b"x")
    qc = a / "artifacts" / "step4" / "qc"
    qc.mkdir(parents=True)
    for n in ("P1.tif_features.csv", "P1.tif_adjacency.csv",
              "P1.tif_features_raw.csv"):
        (qc / n).write_text("label\n1\n", encoding="utf-8")
    (qc / "border_cells.tif").write_bytes(b"b")
    src = a / "artifacts" / "step1" / "patches" / "P1.tif"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"s")
    model = a / "artifacts" / "step2" / "runs" / "Cell" / "model_l"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"w")
    rec = {"status": "qc_extracted", "labels_path": str(run / "labels_3d.tif"),
           "prob_path": str(run / "prob.npy"),
           "original_labels_path": str(run / "labels_3d_original.tif"),
           "qc_features_csv": str(qc / "P1.tif_features.csv"),
           "border_cells_path": str(qc / "border_cells.tif"),
           "source_stack_path": str(src),
           "made_with": {"lateral": str(model), "axial": str(model),
                         "kind": "step2_run"}}
    stacks = {"P1.tif": rec}
    if outside:
        stacks["Q.tif"] = {"status": "predicted",
                           "source_stack_path": str(outside)}
    state = {"schema_version": 2,
             "session": {"stack_folders": [str(a / "artifacts" / "step1"
                                               / "patches")],
                         "active_model": {"lateral": str(model),
                                          "axial": str(model),
                                          "kind": "step2_run"}},
             "stacks": stacks}
    if with_root_key:
        state["workspace_root"] = str(a)
    (a / "step4_state.json").write_text(json.dumps(state), encoding="utf-8")
    return a


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _under(p, root):
    """*p* is *root* itself (the recorded workspace_root) or inside it."""
    p = os.path.normcase(os.path.abspath(p))
    r = os.path.normcase(os.path.abspath(str(root)))
    return p == r or p.startswith(r + os.sep)


def _copy(a, tmp_path):
    b = tmp_path / "B"
    shutil.copytree(a, b)
    return b


def test_a_copy_is_re_rooted_and_the_source_untouched(tmp_path):
    a = _source_workspace(tmp_path)
    b = _copy(a, tmp_path)
    before = _tree_digest(a)
    m = SM.StackStateManager()
    m.load(str(b))
    m.save()
    state = m._state
    paths = [s for s in _strings(state) if os.path.isabs(s)]
    assert paths
    assert all(_under(p, b) for p in paths), [p for p in paths
                                              if not _under(p, b)]
    assert state["workspace_root"] == str(b)
    assert _tree_digest(a) == before                 # byte-identical
    assert any(l.startswith("[state]") for l in m.take_logs())
    saved = json.loads((b / "step4_state.json").read_text(encoding="utf-8"))
    assert all(_under(p, b) for p in _strings(saved) if os.path.isabs(p))


def test_an_older_file_without_the_root_is_inferred(tmp_path):
    a = _source_workspace(tmp_path, with_root_key=False)
    b = _copy(a, tmp_path)
    m = SM.StackStateManager()
    m.load(str(b))
    rec = m.get_record("P1.tif")
    assert _under(rec["labels_path"], b)
    assert _under(m._state["session"]["stack_folders"][0], b)


def test_a_path_outside_the_old_root_is_left_as_is(tmp_path):
    other = tmp_path / "elsewhere" / "Q.tif"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"q")
    a = _source_workspace(tmp_path, outside=other)
    b = _copy(a, tmp_path)
    m = SM.StackStateManager()
    m.load(str(b))
    assert m.get_record("Q.tif")["source_stack_path"] == str(other)


def test_the_same_workspace_is_not_rewritten(tmp_path):
    a = _source_workspace(tmp_path)
    raw = (a / "step4_state.json").read_bytes()
    m = SM.StackStateManager()
    m.load(str(a))
    assert (a / "step4_state.json").read_bytes() == raw
    assert not [l for l in m.take_logs() if "re-rooted" in l]


def test_an_adopt_re_roots_in_memory_and_writes_nothing(tmp_path):
    a = _source_workspace(tmp_path)
    b = _copy(a, tmp_path)
    raw = (b / "step4_state.json").read_bytes()
    m = SM.StackStateManager()
    m.load(str(b), write=False)
    assert _under(m.get_record("P1.tif")["labels_path"], b)
    assert (b / "step4_state.json").read_bytes() == raw


def test_the_step1_guard_still_matches_after_the_re_root(tmp_path):
    from napari_epicycle.data_preparation._downstream import downstream_uses
    a = _source_workspace(tmp_path)
    b = _copy(a, tmp_path)
    m = SM.StackStateManager()
    m.load(str(b))
    m.save()
    assert downstream_uses(str(b), "P1").step4_kinds == ("reconstruction",)


def test_a_delete_outside_the_workspace_is_refused(tmp_path):
    """Second layer: even an un-re-rooted record never deletes a file that
    is not in the current workspace."""
    from ._step4_fakes import controller
    a = _source_workspace(tmp_path)
    b = _copy(a, tmp_path)
    before = _tree_digest(a)
    c = controller(b)
    rec = json.loads((a / "step4_state.json").read_text(
        encoding="utf-8"))["stacks"]["P1.tif"]        # paths into A
    c._delete_derived_files("P1.tif", rec)
    assert _tree_digest(a) == before
    assert any("outside" in l for l in c.ui.log_output.lines)


def test_a_copy_whose_state_is_still_under_artifacts_is_re_rooted(tmp_path):
    """The live driver's case: an older workspace (state at
    artifacts/step4/, no workspace_root key) copied, then Browsed.  The move
    to the root must not stamp the new root before the re-root runs."""
    a = _source_workspace(tmp_path, with_root_key=False)
    old = a / "artifacts" / "step4" / "step4_state.json"
    (a / "step4_state.json").replace(old)
    b = _copy(a, tmp_path)
    m = SM.StackStateManager()
    m.load(str(b))
    saved = json.loads((b / "step4_state.json").read_text(encoding="utf-8"))
    bad = [p for p in _strings(saved) if os.path.isabs(p) and not _under(p, b)]
    assert not bad, bad
    assert _under(saved["session"]["stack_folders"][0], b)


def test_a_model_from_another_workspace_does_not_set_the_root(tmp_path):
    """Live driver B0-1b: a legacy state whose only 'artifacts' path is the
    imported model of ANOTHER workspace must not be taken as copied from
    there -- the root is inferred from Step 4's own artifacts/step4/ paths
    only."""
    other_model = tmp_path / "Other" / "artifacts" / "step2" / "m_l"
    other_model.parent.mkdir(parents=True)
    other_model.write_bytes(b"w")
    ws = tmp_path / "W"
    (ws / "files").mkdir(parents=True)
    lab = ws / "files" / "a_labels.tif"
    lab.write_bytes(b"x")
    (ws / "step4_state.json").write_text(json.dumps({
        "schema_version": 2,
        "session": {"active_model": {"lateral": str(other_model),
                                     "axial": str(other_model),
                                     "kind": "external"}},
        "stacks": {"a.tif": {"status": "reconstructed",
                             "labels_path": str(lab)}}}), encoding="utf-8")
    m = SM.StackStateManager()
    m.load(str(ws))
    assert m._state["session"]["active_model"]["lateral"] == str(other_model)
    assert m.get_record("a.tif")["labels_path"] == str(lab)
