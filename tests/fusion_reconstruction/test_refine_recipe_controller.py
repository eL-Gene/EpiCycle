"""The controller side.

Qt-free: a bare controller (the ``test_batch_recipe_controller.py`` and
``test_step3_commit_before_qc_guard.py`` pattern).  The three recipe
callbacks over ``refine_chain_recipes.json`` (``value_type=dict``), and the
debug trail -- ``steps`` / ``recipes`` in the undo entry's ``extra`` and in
the log -- on Accept.
"""

from __future__ import annotations

import json

import numpy as np

from napari_epicycle.fusion_reconstruction import _batch_recipe_store, _paths
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)


def _ctrl(working_dir):
    c = C.__new__(C)
    c.working_dir = str(working_dir) if working_dir else ""
    c.logs = []
    c._append_log = c.logs.append
    return c


def _rec(n=2):
    return {"kind": "refine_chain", "version": 1, "created": "",
            "spacing_known": True, "spacing": [3.0, 1.0, 1.0],
            "steps": [{"method": "fill_holes", "params": {"axis": "3D"}}] * n}


def test_save_get_delete_round_trip(tmp_path):
    c = _ctrl(tmp_path)
    assert C._get_saved_refine_recipes(c) == ({}, None)
    assert C._on_save_refine_recipe(c, "tidy", _rec(3)) is None
    assert C._on_save_refine_recipe(c, "keep", _rec(1)) is None
    store, problem = C._get_saved_refine_recipes(c)
    assert problem is None and set(store) == {"tidy", "keep"}
    assert C._on_delete_refine_recipe(c, "tidy") is None
    store, _ = C._get_saved_refine_recipes(c)
    assert set(store) == {"keep"}
    assert _paths.refine_chain_recipes_path(tmp_path).exists()
    assert any("Saved Refine recipe" in m for m in c.logs)
    assert any("Deleted Refine recipe" in m for m in c.logs)
    # the Batch file is untouched
    assert not _paths.batch_refine_recipes_path(tmp_path).exists()


def test_no_workspace(tmp_path):
    c = _ctrl(None)
    # audit L1: no workspace is a PROBLEM, so the menu can say so
    assert C._get_saved_refine_recipes(c) == (
        None, "Recipes need a workspace — open this from Step 3.")
    assert C._on_save_refine_recipe(c, "tidy", _rec()) == (
        "no workspace is open")
    assert C._on_delete_refine_recipe(c, "tidy") == "no workspace is open"
    assert len(c.logs) == 2


def test_a_failure_is_logged_and_returned(tmp_path, monkeypatch):
    c = _ctrl(tmp_path)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(_batch_recipe_store, "save_named_recipe", boom)
    assert C._on_save_refine_recipe(c, "tidy", _rec()) == "disk full"
    assert any("Failed to save Refine recipe" in m for m in c.logs)


def test_a_corrupt_file_is_a_problem_and_is_never_overwritten(tmp_path):
    c = _ctrl(tmp_path)
    p = _paths.refine_chain_recipes_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{{{", encoding="utf-8")
    store, problem = C._get_saved_refine_recipes(c)
    assert store == {} and problem == "The recipes file could not be read."
    assert C._on_save_refine_recipe(c, "tidy", _rec()) == problem
    assert C._on_delete_refine_recipe(c, "tidy") == problem
    assert p.read_text(encoding="utf-8") == "{{{"


def test_a_wrong_type_entry_can_be_deleted_and_survives_other_saves(tmp_path):
    c = _ctrl(tmp_path)
    p = _paths.refine_chain_recipes_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"old": [1, 2]}), encoding="utf-8")
    assert C._on_save_refine_recipe(c, "tidy", _rec()) is None
    assert set(json.loads(p.read_text("utf-8"))) == {"old", "tidy"}
    assert C._on_delete_refine_recipe(c, "old") is None
    assert set(json.loads(p.read_text("utf-8"))) == {"tidy"}


def test_the_launch_call_wires_all_three():
    import inspect

    from napari_epicycle.fusion_reconstruction import controller as mod

    src = inspect.getsource(mod)
    assert "on_save_refine_recipe=self._on_save_refine_recipe" in src
    assert "get_saved_refine_recipes=self._get_saved_refine_recipes" in src
    assert "on_delete_refine_recipe=self._on_delete_refine_recipe" in src


# ---------------------------------------------------------------------------
# -- the debug trail on Accept
# ---------------------------------------------------------------------------

ROI = (slice(0, 2), slice(0, 2), slice(0, 2))
OLD = np.zeros((2, 2, 2), np.int32)
STEPS = [{"method": "sdf", "params": {"sigma_z": 3.0}},
         {"method": "dilate_erode", "params": {"radius": 1, "axis": "3D"}},
         {"method": "settle", "params": {}}]


def _commit_ctrl():
    from tests.fusion_reconstruction.test_step3_commit_before_qc_guard import (
        _base_df,
        _make_controller,
    )

    return _make_controller(df=_base_df())


def test_steps_and_recipes_reach_the_undo_entry_and_the_log(capsys):
    c = _commit_ctrl()
    c._on_annotator_label_smoothed(7, roi_slices=ROI, old_roi_data=OLD,
                                   steps=STEPS, recipes=["tidy"])
    entry = c._curation_undo_stack.peek()
    assert entry.action_type == "smooth"
    assert entry.extra == {"steps": STEPS, "recipes": ["tidy"]}
    out = capsys.readouterr().out
    assert ("[CURATION-COMMIT] smooth label 7: 3 steps (Smooth -> "
            "Dilate/Erode -> Settle) recipe=tidy") in out
    assert c.logs == [
        "Annotator: accepted smooth for label 7 → curated "
        "(3 steps; recipe 'tidy')"]


def test_a_hand_built_chain_has_an_empty_recipe_list():
    c = _commit_ctrl()
    c._on_annotator_label_smoothed(7, roi_slices=ROI, old_roi_data=OLD,
                                   steps=STEPS[:1], recipes=[])
    assert c._curation_undo_stack.peek().extra == {
        "steps": STEPS[:1], "recipes": []}
    assert c.logs == [
        "Annotator: accepted smooth for label 7 → curated (1 step)"]


def test_the_old_call_shape_still_works():
    c = _commit_ctrl()
    c._on_annotator_label_smoothed(7, roi_slices=ROI, old_roi_data=OLD)
    assert c._curation_undo_stack.peek().extra == {"steps": [],
                                                   "recipes": []}
    assert c.logs == ["Annotator: accepted smooth for label 7 → curated"]


def test_the_annotator_passes_the_trail_on_accept(qapp_or_skip=None):
    """Both `_on_label_smoothed_cb` sites pass `steps=` and `recipes=`."""
    import inspect

    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    src = inspect.getsource(Curation3DAnnotator._accept_smooth)
    assert src.count("steps=prov_steps, recipes=prov_recipes)") == 2


def test_the_launcher_threads_the_three_keywords():
    import inspect

    from napari_epicycle.annotator import launcher

    sig = inspect.signature(launcher.launch_3d_curation_annotator)
    src = inspect.getsource(launcher.launch_3d_curation_annotator)
    for kw in ("on_save_refine_recipe", "get_saved_refine_recipes",
               "on_delete_refine_recipe"):
        assert sig.parameters[kw].default is None
        assert "{0}={0},".format(kw) in src
