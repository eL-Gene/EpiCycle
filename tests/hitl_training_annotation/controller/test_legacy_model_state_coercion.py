"""Item 5: persisted 'none' / 'scratch' (items no longer in the combos) are
mapped to 'auto' / 'cyto3' so the combos never silently disagree."""
from __future__ import annotations

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController as C)


class _Combo:
    def __init__(self, items, cur):
        self.items, self.cur = list(items), cur

    def setCurrentText(self, t):
        if t in self.items:
            self.cur = t

    def blockSignals(self, b):
        return False


def _ctrl():
    c = C.__new__(C)
    c.curation_model_source_mode = "none"
    c.curation_pretrained_model_type = "scratch"
    return c


def test_coerce_maps_legacy_values():
    c = _ctrl()
    c._coerce_legacy_model_state()
    assert c.curation_model_source_mode == "auto"
    assert c.curation_pretrained_model_type == "cyto3"


def test_coerce_leaves_current_values_alone():
    c = _ctrl()
    c.curation_model_source_mode = "round"
    c.curation_pretrained_model_type = "nuclei"
    c._coerce_legacy_model_state()
    assert (c.curation_model_source_mode, c.curation_pretrained_model_type) == (
        "round", "nuclei")


def test_refresh_from_legacy_state_selects_auto_and_cyto3():
    c = _ctrl()
    ui = type("UI", (), {})()
    ui.model_source_combo = _Combo(
        ["Auto (Recommended)", "Round model", "External model file",
         "Cellpose pretrained"], "Round model")
    ui.curation_pretrained_combo = _Combo(["cyto3", "nuclei"], "nuclei")
    ui.imported_model_path = None
    c.ui = ui
    c.imported_curation_model_path = ""
    c._pop_signals_blocked = lambda w, prev: None
    c._push_signals_blocked = lambda w: False
    c._MODEL_SOURCE_LABELS = C._MODEL_SOURCE_LABELS
    c._refresh_model_widgets_from_state()
    assert ui.model_source_combo.cur == "Auto (Recommended)"
    assert ui.curation_pretrained_combo.cur == "cyto3"


def test_load_internal_state_coerces_and_the_next_save_persists_it(tmp_path):
    """The restore site: a state file holding the removed values loads as
    'auto' / 'cyto3' (in memory), and the next save writes the coerced
    values, so the legacy ones leave the disk on the first save."""
    import json

    from tests.hitl_training_annotation.controller.test_state_persistence_integration import (
        _controller_stub)

    ctrl = _controller_stub(tmp_path)
    ctrl._save_internal_state()
    state_path = tmp_path / "hitl_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["curation_model_source_mode"] = "none"
    state["curation_pretrained_model_type"] = "scratch"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    ctrl._load_internal_state()
    assert ctrl.curation_model_source_mode == "auto"
    assert ctrl.curation_pretrained_model_type == "cyto3"

    ctrl._save_internal_state()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["curation_model_source_mode"] == "auto"
    assert saved["curation_pretrained_model_type"] == "cyto3"
