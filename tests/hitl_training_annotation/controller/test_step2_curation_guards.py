"""Step 2 Brick A — curation-loss guards.

Resolves two defects: #3 (predict over a `curating` slice destroys hand work) and #4
(a patch switch inside the 5 s debounce writes the NEW patch's pixels
into the OLD slice's pair).

All fakes are Qt-free; the real controller methods are bound to them.
"""

from __future__ import annotations

import pytest

from napari_epicycle.hitl_training_annotation.controller import (
    training_annotation_controller as tac,
)
from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _UI:
    def __init__(self):
        self.training_log_output = _LogOutput()


class _FakeMessageBox:
    """QMessageBox stand-in recording the question() calls."""

    Yes = 1
    No = 0

    def __init__(self, reply):
        self._reply = reply
        self.calls = []

    def question(self, parent, title, body, buttons=None, default=None):
        self.calls.append((title, body))
        return self._reply


# ---------------------------------------------------------------------------
# #3 — re-segment must guard `curating` in BOTH predict paths
# ---------------------------------------------------------------------------

def _make_guard_ctrl(status):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.sample_records = {"patchA_XY_z0007": {"status": status}}
    ctrl.touched = []
    ctrl._touch_sample = lambda label, **kw: ctrl.touched.append((label, kw))
    ctrl._append_ui_log = TrainingAnnotationController._append_ui_log.__get__(ctrl)
    ctrl._confirm_resegment = TrainingAnnotationController._confirm_resegment.__get__(ctrl)
    return ctrl


@pytest.mark.parametrize("status", ["accepted", "staged", "committed", "curating"])
def test_resegment_guard_asks_before_overwriting_curated_or_edited(monkeypatch, status):
    ctrl = _make_guard_ctrl(status)
    box = _FakeMessageBox(reply=_FakeMessageBox.No)
    monkeypatch.setattr(tac, "QMessageBox", box)

    assert ctrl._confirm_resegment("patchA_XY_z0007") is False
    assert len(box.calls) == 1, f"status={status!r} must be guarded"
    assert ctrl.touched == []


def test_resegment_guard_names_the_slice_and_the_discard_route(monkeypatch):
    ctrl = _make_guard_ctrl("curating")
    box = _FakeMessageBox(reply=_FakeMessageBox.No)
    monkeypatch.setattr(tac, "QMessageBox", box)

    ctrl._confirm_resegment("patchA_XY_z0007")

    log = " | ".join(ctrl.ui.training_log_output.lines)
    assert "patchA_XY_z0007" in log
    assert "still has your drawing" in log
    assert "Discard the slice first" in log
    # The dialog says plainly what is lost and what it costs.
    title, body = box.calls[0]
    assert title == "Replace the drawing on this slice?"
    assert "You have drawn on patchA_XY_z0007" in body
    assert "cannot be undone" in body
    assert "earlier rounds" in body


def test_finished_slice_prompt_uses_its_own_plain_copy(monkeypatch):
    ctrl = _make_guard_ctrl("accepted")
    box = _FakeMessageBox(reply=_FakeMessageBox.No)
    monkeypatch.setattr(tac, "QMessageBox", box)

    assert ctrl._confirm_resegment("patchA_XY_z0007") is False

    title, body = box.calls[0]
    assert title == "Replace a slice you already finished?"
    assert "already marked patchA_XY_z0007 as accepted" in body
    log = " | ".join(ctrl.ui.training_log_output.lines)
    assert "Left patchA_XY_z0007 alone" in log
    assert "already marked it accepted" in log


def test_confirming_drops_the_pending_stroke_for_that_slice(monkeypatch):
    """The debounce would otherwise write the OLD drawing over the new
    prediction seconds later."""
    ctrl = _make_guard_ctrl("curating")
    ctrl._curating_persist_pending = {
        "patchA_XY_z0007": ("patchA_XY_z0007", "XY", 7),
        "patchA_XY_z0009": ("patchA_XY_z0009", "XY", 9),
    }
    monkeypatch.setattr(
        tac, "QMessageBox", _FakeMessageBox(reply=_FakeMessageBox.Yes)
    )

    assert ctrl._confirm_resegment("patchA_XY_z0007") is True

    assert "patchA_XY_z0007" not in ctrl._curating_persist_pending
    assert "patchA_XY_z0009" in ctrl._curating_persist_pending


def test_resegment_guard_passes_through_for_predicted_and_unknown(monkeypatch):
    box = _FakeMessageBox(reply=_FakeMessageBox.No)
    monkeypatch.setattr(tac, "QMessageBox", box)
    for status in ("predicted", "", None):
        ctrl = _make_guard_ctrl(status)
        assert ctrl._confirm_resegment("patchA_XY_z0007") is True
    ctrl = _make_guard_ctrl("accepted")
    assert ctrl._confirm_resegment("no_such_slice_XY_z0001") is True
    assert box.calls == []


def test_resegment_guard_yes_demotes_to_predicted(monkeypatch):
    ctrl = _make_guard_ctrl("curating")
    box = _FakeMessageBox(reply=_FakeMessageBox.Yes)
    monkeypatch.setattr(tac, "QMessageBox", box)

    assert ctrl._confirm_resegment("patchA_XY_z0007") is True
    assert ctrl.touched == [("patchA_XY_z0007", {"status": "predicted", "origin": "cellpose"})]


class _HardSentinel(BaseException):
    """BaseException so the predict paths' `except Exception` cannot eat it."""


class _ExplodingSegManager:
    """Anything the predict paths touch AFTER the guard blows up."""

    segmenter = None

    def get_masks(self, *a, **k):
        raise _HardSentinel()

    def set_model_preference(self, *a, **k):
        raise _HardSentinel()

    def segment(self, *a, **k):
        raise _HardSentinel()


def _make_predict_ctrl():
    import types

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.sample_records = {}
    ctrl.loaded_patch_name = "patchA"
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl.segmentation_manager = _ExplodingSegManager()
    ctrl.viewer = types.SimpleNamespace(dims=types.SimpleNamespace(ndisplay=2))
    ctrl.current_config = {}

    ctrl.guard_calls = []
    ctrl._confirm_resegment = lambda label: (
        ctrl.guard_calls.append(label) or False
    )
    for name in (
        "_ensure_runtime_defaults", "_set_view_radio", "_set_axial_combo_enabled",
        "_apply_curation_checkpoint_for_view", "apply_active_view_to_viewer",
        "_exit_refine_mode", "_ensure_segmentation_config_defaults",
        "_apply_inference_params_from_ui", "_adjust_channels_for_input",
    ):
        setattr(ctrl, name, lambda *a, **k: None)
    ctrl._append_ui_log = TrainingAnnotationController._append_ui_log.__get__(ctrl)
    ctrl._apply_model_source_selection = lambda **k: {"can_segment": True}
    ctrl._collect_current_slice_input = lambda: ("patchA_XY_z0007", object())
    ctrl._segment_current_slice_from_native = (
        TrainingAnnotationController._segment_current_slice_from_native.__get__(ctrl)
    )
    ctrl._run_segmentation = (
        TrainingAnnotationController._run_segmentation.__get__(ctrl)
    )
    return ctrl


def test_annotator_predict_path_stops_at_the_guard():
    ctrl = _make_predict_ctrl()

    result = ctrl._segment_current_slice_from_native("XY")

    assert result["ok"] is False
    assert "cancelled" in result["message"]
    assert ctrl.guard_calls == ["patchA_XY_z0007"], (
        "the annotator predict path must consult the shared guard exactly once"
    )
    # _HardSentinel never fired => nothing past the guard ran.


def test_main_panel_predict_path_stops_at_the_guard():
    ctrl = _make_predict_ctrl()

    assert ctrl._run_segmentation() is None
    assert ctrl.guard_calls == ["patchA_XY_z0007"]


# ---------------------------------------------------------------------------
# #4 — a patch switch inside the debounce must not write the NEW patch's
#      pixels into the OLD slice's pair
# ---------------------------------------------------------------------------

class _Sentinel(Exception):
    """Aborts `_load_patch` right after the flush block."""


class _FakeLayer:
    def __init__(self, data):
        self.data = data


class _FakeSegManager:
    def __init__(self):
        self.data_masks = {}


class _FakeTimer:
    def __init__(self):
        self.active = True
        self.stopped = 0

    def isActive(self):
        return self.active

    def stop(self):
        self.stopped += 1
        self.active = False


def _make_switch_ctrl():
    import numpy as np

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.loaded_patch_name = "OLD"
    old_volume = np.zeros((4, 3, 3), dtype="int32")
    old_volume[2] = 7  # the pending stroke, on OLD's XY slice 2
    ctrl._old_layer = _FakeLayer(old_volume)
    ctrl.segmentation_manager = _FakeSegManager()
    ctrl._curating_persist_pending = {"OLD_XY_z0002": ("OLD_XY_z0002", "XY", 2)}
    ctrl._curating_persist_timer = _FakeTimer()

    ctrl.persisted = []
    ctrl._work_labels_layer_name_for_view = lambda view_name=None: (
        f"{ctrl.loaded_patch_name}_work_labels_lateral_3d"
    )
    ctrl._layer_by_name = lambda name: (
        ctrl._old_layer if name.startswith("OLD") else _FakeLayer(np.zeros((4, 3, 3), "int32"))
    )
    ctrl._persist_curated_training_pair = lambda label, async_write=False: (
        ctrl.persisted.append(
            (label, ctrl.segmentation_manager.data_masks[label].copy(), async_write)
        )
        or True
    )
    ctrl._schedule_state_save = lambda: None
    ctrl._refresh_run_picker = lambda: None
    ctrl._flush_curating_persist_pending = (
        TrainingAnnotationController._flush_curating_persist_pending.__get__(ctrl)
    )
    # Abort `_load_patch` immediately AFTER the flush block: anything that
    # runs before `_exit_refine_mode` is what the test is about.
    def _boom():
        raise _Sentinel()
    ctrl._exit_refine_mode = _boom
    ctrl._load_patch = TrainingAnnotationController._load_patch.__get__(ctrl)
    return ctrl


def test_patch_switch_flushes_the_debounce_before_touching_layers():
    ctrl = _make_switch_ctrl()

    # The user switches patches 1 s into the 5 s debounce.
    with pytest.raises(_Sentinel):
        ctrl._load_patch()

    assert len(ctrl.persisted) == 1, (
        "the pending curating edit must be flushed at the top of _load_patch, "
        "while the OLD patch's layers are still loaded"
    )
    label, mask, async_write = ctrl.persisted[0]
    assert label == "OLD_XY_z0002"
    assert int(mask.max()) == 7, "the OLD patch's own pixels must be written"
    assert ctrl._curating_persist_pending == {}, "pending must be empty after the flush"
    assert ctrl._curating_persist_timer.stopped == 1, (
        "the debounce timer must be stopped so it cannot re-fire against the NEW patch"
    )


def test_patch_switch_flush_is_harmless_with_nothing_pending():
    ctrl = _make_switch_ctrl()
    ctrl._curating_persist_pending = {}
    with pytest.raises(_Sentinel):
        ctrl._load_patch()
    assert ctrl.persisted == []
    assert ctrl._curating_persist_pending == {}


def test_pending_entry_with_no_layer_is_dropped_not_carried_over():
    """The annotator was closed: the work-labels layer is gone.

    `_flush_curating_persist_pending` `continue`s without popping in that
    case, so the entry would drain later against the NEW patch's layers —
    the very bug the pre-switch flush exists to prevent.
    """
    ctrl = _make_switch_ctrl()
    ctrl._layer_by_name = lambda name: None

    with pytest.raises(_Sentinel):
        ctrl._load_patch()

    assert ctrl.persisted == [], "nothing can be written without a layer"
    assert ctrl._curating_persist_pending == {}, (
        "an unsaveable pending edit must be dropped, never carried into the "
        "next patch"
    )
    log = " | ".join(ctrl.ui.training_log_output.lines)
    assert "could not be saved" in log
