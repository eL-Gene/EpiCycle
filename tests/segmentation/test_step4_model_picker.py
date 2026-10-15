"""Change…: Step 3's picker with Step 4's rows,
the model card, and the chips that follow the active model."""
from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump, refined_bundle

ROOT = Path(__file__).resolve().parents[2]
S3C = ROOT / "napari_epicycle" / "fusion_reconstruction" / "controller.py"


@functools.lru_cache(maxsize=None)
def _step3_label_for():
    """Step 3's nested ``_label_for`` (``_refresh_model_status``), compiled
    from its source so the parity is against the real rule."""
    tree = ast.parse(S3C.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_label_for":
            ns = {"_P": Path}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(S3C),
                         "exec"), ns)  # noqa: S102 -- our own source
            return ns["_label_for"]
    raise AssertionError("Step 3's _label_for not found")


@pytest.mark.parametrize("text", [
    "", "cyto3", "nuclei",
    "C:/ws/artifacts/step3/refinement/models/wound-v2/lateral",
    "C:/ws/artifacts/step3/refinement/models/wound-v2/axial_model",
    "C:/ws/models/round_003/model",
    "D:\\external\\my_model.pt",
])
def test_label_for_is_step3s(text):
    assert M.label_for(text) == _step3_label_for()(text)


def test_model_indicator_rules():
    text, colour = M.model_indicator(None)
    from napari_epicycle.ui_shell.theme import MODEL_ACTIVE_PAIR, MODEL_NONE
    assert (text, colour) == (MODEL_NONE, M.MODEL_COLOR_NONE)
    pre = M.model_snapshot("cyto3", "cyto3", kind="pretrained")
    assert M.model_indicator(pre) == ("Pretrained: cyto3",
                                      M.MODEL_COLOR_PRETRAINED)
    ref = M.model_snapshot("C:/w/refinement/models/b/lateral",
                           "C:/w/refinement/models/b/axial",
                           kind="refined", run_id="runA", bundle="b")
    assert M.model_indicator(ref) == ("b · runA", M.MODEL_COLOR_FILE)
    ext = M.model_snapshot("C:/x/l.pt", "C:/x/a.pt", kind="external")
    assert M.model_indicator(ext) == (
        MODEL_ACTIVE_PAIR.format(lateral="l.pt", axial="a.pt"),
        M.MODEL_COLOR_PAIR)


def test_rows_merge_per_run_and_bundle(tmp_path):
    ws = make_workspace(tmp_path)
    refined_bundle(ws, None, "wound-v2")
    from napari_epicycle.fusion_reconstruction import _run_management as RM
    entries = [
        {"bundle_name": "b", "run_id": "r", "bucket": "lateral",
         "weights_path": "C:/w/b/lateral", "manifest": {
             "created_at": "t", "lateral": {"n_train_pairs": 4}}},
        {"bundle_name": "b", "run_id": "r", "bucket": "axial",
         "weights_path": "C:/w/b/axial", "manifest": {
             "created_at": "t", "axial": {"n_train_pairs": 5}}},
        {"bundle_name": "b", "run_id": None, "bucket": "lateral",
         "weights_path": "C:/w0/b/lateral", "manifest": None},
    ]
    rows = M.step4_model_rows(entries)
    assert len(rows) == 2
    both = next(r for r in rows if r["run_id"] == "r")
    assert both["mode"] == "Both"
    assert both["lateral_pairs"] == 4 and both["axial_pairs"] == 5
    for key in ("name", "mode", "lateral_pairs", "axial_pairs", "created",
                "init_summary", "lateral_weights", "axial_weights",
                "bundle_dir", "run_id"):
        assert key in both
    assert RM.list_cross_run_refined_models(ws, trained_only=True)


def test_the_dialog_builds_with_step4_rows_and_hides_init(qapp):
    from napari_epicycle.fusion_reconstruction.refinement_registry_dialog \
        import RefinementRegistryDialog
    rows = M.step4_model_rows([
        {"bundle_name": "b", "run_id": "r", "bucket": "lateral",
         "weights_path": "C:/w/b/lateral", "manifest": {}}])
    dlg = RefinementRegistryDialog(rows=rows, show_init=False)
    try:
        assert not dlg._init_btn.isVisibleTo(dlg)
        assert dlg._table.rowCount() >= 1
    finally:
        dlg.deleteLater()
    step3 = RefinementRegistryDialog(rows=rows)
    try:
        assert step3._init_btn.isVisibleTo(step3)        # Step 3 unchanged
    finally:
        step3.deleteLater()


@pytest.fixture
def ws_with_models(tmp_path):
    ws = make_workspace(tmp_path)
    refined_bundle(ws, None, "wound-v2", channels=(2, 1))
    return ws


def test_change_opens_the_picker_with_every_runs_bundles(
        qapp, ws_with_models, monkeypatch):
    from napari_epicycle.fusion_reconstruction import (
        refinement_registry_dialog as RRD,
    )
    seen = {}
    monkeypatch.setattr(RRD.RefinementRegistryDialog, "show_dialog",
                        classmethod(lambda cls, **kw: seen.update(kw)))
    w = panel(qapp, ws_with_models)
    try:
        w.model_change_btn.click()
        assert seen["show_init"] is False and seen["on_init"] is None
        assert [r["name"] for r in seen["rows"]] == ["wound-v2"]
        assert seen["on_use"] == w.controller._on_model_use
    finally:
        close(w)


def test_use_makes_it_the_active_model(qapp, ws_with_models):
    w = panel(qapp, ws_with_models)
    try:
        c = w.controller
        rows = M.step4_model_rows(
            __import__("napari_epicycle.fusion_reconstruction._run_management",
                       fromlist=["x"]).list_cross_run_refined_models(
                ws_with_models, trained_only=True))
        c._on_model_use(dict(rows[0], _kind="refined"))
        pump(qapp)
        snap = c.state_manager.get_active_model()
        assert snap["kind"] == "refined" and snap["bundle"] == "wound-v2"
        assert w.model_active_indicator_label.text() == "wound-v2 · default"
        assert w.model_trained_label.text() == T.SEG_MODEL_TRAINED_ON.format(
            n=2, names="membrane (1), nuclei (0)")
        # the trained channels set the combos
        assert w.chosen_channels() == (1, 0)
        # a built-in model: the chips of a stack made by wound-v2 go amber
        c.state_manager.set_stack_status(
            "P1.tif", "reconstructed", made_with=snap, labels_path="x")
        c._on_model_use({"_kind": "pretrained", "name": "cyto3"})
        pump(qapp)
        row = next(r for r in c._rows if r["name"] == "P1.tif")
        assert row["chips"][0][1] is True
        assert w.model_trained_label.text() == T.SEG_MODEL_TRAINED_UNKNOWN
    finally:
        close(w)


def test_missing_model_files_turn_the_card_amber(qapp, tmp_path):
    ws = make_workspace(tmp_path)
    w = panel(qapp, ws)
    try:
        c = w.controller
        c._on_model_use({"_kind": "external",
                         "lateral_weights": str(tmp_path / "gone" / "l.pt"),
                         "axial_weights": str(tmp_path / "gone" / "a.pt")})
        pump(qapp)
        assert w.model_trained_label.text() == T.MODEL_MISSING
        assert "#d4a843" in w.model_trained_label.styleSheet()
        assert not w.predict_btn.isEnabled()
    finally:
        close(w)
