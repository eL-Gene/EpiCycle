"""Tests for the annotator-side Batch-Refine recipe history wiring (2026-06-07).

Two layers:
  * session-recent list — _append_batch_recipe_history serializes + de-dupes +
    caps the in-memory _batch_recipe_history; _on_batch_accept appends the
    applied recipe (source check).
  * save / refresh forwarders — _save_batch_recipe routes through the injected
    _save_batch_recipe_cb (guarded when absent); _refresh_batch_recipe_history
    pulls recent + saved and hands them to the dialog.

Qt-free: Curation3DAnnotator.__new__ stubs (no display) + source introspection.
"""

from __future__ import annotations

import inspect

from packages.epicycle_annotator.batch_refine import RefineStep
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator


# --------------------------------------------------------------------------- #
# set_curation_callbacks stores the new recipe callbacks
# --------------------------------------------------------------------------- #
def test_set_curation_callbacks_accepts_recipe_cbs():
    sig = inspect.signature(Curation3DAnnotator.set_curation_callbacks)
    assert "on_save_batch_recipe" in sig.parameters
    assert "get_saved_batch_recipes" in sig.parameters


def test_set_curation_callbacks_stores_recipe_cbs_live():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    save = lambda name, rj: None  # noqa: E731
    get = lambda: {}  # noqa: E731
    ann.set_curation_callbacks(
        on_save_batch_recipe=save, get_saved_batch_recipes=get
    )
    assert ann._save_batch_recipe_cb is save
    assert ann._get_saved_batch_recipes_cb is get


# --------------------------------------------------------------------------- #
# session-recent list — append, de-dupe, cap
# --------------------------------------------------------------------------- #
def _recipe(method="sdf"):
    if method == "sdf":
        return [RefineStep(method="sdf", params={"sigma_z": 1.5, "sigma_xy": 1.5})]
    if method == "fill":
        return [RefineStep(method="fill_holes", params={"axis": "3D"})]
    return [RefineStep(method="dilate_erode", params={"radius": 2, "axis": "XY"})]


def test_append_history_serializes_and_orders():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    ann._batch_recipe_history = []
    ann._append_batch_recipe_history(_recipe("sdf"))
    ann._append_batch_recipe_history(_recipe("fill"))
    hist = ann._batch_recipe_history
    assert len(hist) == 2
    # Most-recent first.
    assert hist[0][0]["method"] == "fill_holes"
    assert hist[1][0]["method"] == "sdf"
    # Serialized to JSON dicts, not RefineStep.
    assert all(isinstance(r, list) and isinstance(r[0], dict) for r in hist)


def test_append_history_dedupes_identical():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    ann._batch_recipe_history = []
    ann._append_batch_recipe_history(_recipe("sdf"))
    ann._append_batch_recipe_history(_recipe("fill"))
    ann._append_batch_recipe_history(_recipe("sdf"))  # duplicate of the first
    hist = ann._batch_recipe_history
    # Still 2 unique entries; the re-added sdf moved to the front.
    assert len(hist) == 2
    assert hist[0][0]["method"] == "sdf"
    assert hist[1][0]["method"] == "fill_holes"


def test_append_history_caps_at_ten():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    ann._batch_recipe_history = []
    # 15 distinct recipes (vary sigma_z so each is unique).
    for i in range(15):
        r = [RefineStep(method="sdf", params={"sigma_z": float(i) / 10 + 0.5, "sigma_xy": 1.5})]
        ann._append_batch_recipe_history(r)
    assert len(ann._batch_recipe_history) == Curation3DAnnotator._BATCH_RECIPE_HISTORY_CAP
    assert Curation3DAnnotator._BATCH_RECIPE_HISTORY_CAP == 10


def test_append_history_lazily_inits_list():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    # Attribute present but None (the lazy-init guard's real defense; a bare
    # QWidget __new__ raises on a truly-missing attr via Qt's __getattr__).
    ann._batch_recipe_history = None
    ann._append_batch_recipe_history(_recipe("sdf"))
    assert len(ann._batch_recipe_history) == 1


def test_append_history_ignores_empty():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    ann._batch_recipe_history = []
    ann._append_batch_recipe_history([])
    assert ann._batch_recipe_history == []


# --------------------------------------------------------------------------- #
# _on_batch_accept appends to the session-recent list (source check)
# --------------------------------------------------------------------------- #
def test_batch_accept_appends_recipe_history_source():
    src = inspect.getsource(Curation3DAnnotator._on_batch_accept)
    assert "_append_batch_recipe_history" in src, (
        "_on_batch_accept must record the applied recipe in the session-recent "
        "history"
    )
    # The capture must read the dialog recipe BEFORE teardown.
    capture_idx = src.index("_append_batch_recipe_history")
    teardown_idx = src.index("_teardown_batch_dialog")
    assert capture_idx < teardown_idx, (
        "recipe-history capture must run before the dialog is torn down"
    )


# --------------------------------------------------------------------------- #
# _save_batch_recipe — routes through the cb, guards absence
# --------------------------------------------------------------------------- #
class _FakeDialog:
    def __init__(self, recipe):
        self._recipe = recipe
        self.status = []
        self.history = None

    def recipe(self):
        return self._recipe

    def set_status(self, text):
        self.status.append(text)

    def set_recipe_history(self, recent, saved):
        self.history = (recent, saved)


def test_save_batch_recipe_fires_callback():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    dlg = _FakeDialog(_recipe("sdf"))
    ann._batch_refine_dialog = dlg
    ann._batch_recipe_history = []
    seen = {}
    ann._save_batch_recipe_cb = lambda name, rj: seen.update(name=name, rj=rj)
    ann._get_saved_batch_recipes_cb = lambda: {"prev": []}
    ann._save_batch_recipe("MyName")
    assert seen["name"] == "MyName"
    assert isinstance(seen["rj"], list) and seen["rj"][0]["method"] == "sdf"
    # Saved dropdown refreshed after the save.
    assert dlg.history is not None


def test_save_batch_recipe_no_cb_sets_status():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    dlg = _FakeDialog(_recipe("sdf"))
    ann._batch_refine_dialog = dlg
    ann._save_batch_recipe_cb = None
    ann._save_batch_recipe("X")  # must not raise
    assert any("workspace" in s.lower() for s in dlg.status)


def test_save_batch_recipe_empty_recipe_no_fire():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    dlg = _FakeDialog([])  # empty recipe
    ann._batch_refine_dialog = dlg
    seen = {"n": 0}
    ann._save_batch_recipe_cb = lambda name, rj: seen.__setitem__("n", seen["n"] + 1)
    ann._save_batch_recipe("X")
    assert seen["n"] == 0


def test_save_batch_recipe_swallows_cb_error():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    dlg = _FakeDialog(_recipe("sdf"))
    ann._batch_refine_dialog = dlg

    def _boom(name, rj):
        raise RuntimeError("kaboom")

    ann._save_batch_recipe_cb = _boom
    ann._save_batch_recipe("X")  # must not raise
    # A fixed sentence on screen (UX audit M5), never the raw exception.
    from packages.epicycle_annotator.batch_refine import BATCH_SAVE_FAILED

    assert dlg.status[-1] == BATCH_SAVE_FAILED
    assert not any("kaboom" in s for s in dlg.status)


# --------------------------------------------------------------------------- #
# _refresh_batch_recipe_history — pulls recent + saved, hands to dialog
# --------------------------------------------------------------------------- #
def test_refresh_history_passes_recent_and_saved():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    dlg = _FakeDialog(_recipe("sdf"))
    ann._batch_refine_dialog = dlg
    ann._batch_recipe_history = [[{"method": "sdf", "params": {}}]]
    ann._get_saved_batch_recipes_cb = lambda: {"S": [{"method": "fill_holes", "params": {}}]}
    ann._refresh_batch_recipe_history(dlg)
    recent, saved = dlg.history
    assert recent == ann._batch_recipe_history
    assert "S" in saved


def test_refresh_history_no_cb_passes_empty_saved():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    dlg = _FakeDialog(_recipe("sdf"))
    ann._batch_refine_dialog = dlg
    ann._batch_recipe_history = []
    ann._get_saved_batch_recipes_cb = None
    ann._refresh_batch_recipe_history(dlg)
    recent, saved = dlg.history
    assert saved == {}


def test_refresh_history_swallows_cb_error():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    dlg = _FakeDialog(_recipe("sdf"))
    ann._batch_refine_dialog = dlg
    ann._batch_recipe_history = []

    def _boom():
        raise RuntimeError("nope")

    ann._get_saved_batch_recipes_cb = _boom
    ann._refresh_batch_recipe_history(dlg)  # must not raise
    recent, saved = dlg.history
    assert saved == {}
