"""Doors that start or change an edit refuse, with one sentence, while
another edit is open - before anything is written."""
from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from napari_epicycle.annotator._staged_op_text import STAGED_WORK_OPEN
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K
from napari_epicycle.annotator.refinement_state import RefinementOpState
from napari_epicycle.fusion_reconstruction import _step3_pr_text

_FLAGS = dict(
    _split_running=False, _split_fragment_refine=False, _reshape_running=False,
    _smooth_running=False, _interp_running=False, _interp_snapshot=None,
    _interp_refine_mode=False, _add_running=False, _add_review=False,
    _batch_running=False, _batch_staged=False, _batch_refine_dialog=None,
    _merge_dialog=None, _draw_snapshot=None, _refine_running=False,
    _refine_sync_busy=False, _active_labels_name="",
)

_BOUND = (
    "_sync_refine_state", "staged_kinds", "staged_op_kind",
    "_refuse_for_open_edit", "_add_refused_while_staged",
    "_add_refused_while_batch", "_add_refresh_label_buttons",
)


class _Adv:
    def __init__(self):
        self.focused = []

    def focus_tab(self, name):
        self.focused.append(name)
        return True


def make(*bind, **flags):
    f = SimpleNamespace(_refine_state=RefinementOpState(), **_FLAGS)
    f.__dict__.update(flags)
    f.adv = _Adv()
    f.said = []
    f._get_advanced_features_widget = lambda: f.adv
    f._push_add_ui = lambda **kw: f.said.append(("add", kw))
    f._update_split_ui = lambda **kw: f.said.append(("reshape", kw))
    f._update_smooth_ui = lambda **kw: f.said.append(("refine", kw))
    f._refresh_label_action_buttons_state = lambda: f.said.append("labels")
    f._refresh_merge_button_state = lambda: f.said.append("merge")
    for name in _BOUND + bind:
        setattr(f, name, types.MethodType(getattr(K, name), f))
    return f


def _texts(f, slot):
    return [kw.get("status_text") for s, kw in
            (x for x in f.said if isinstance(x, tuple)) if s == slot]


def test_sentence_constants_are_equal():
    assert STAGED_WORK_OPEN == _step3_pr_text.S3_STAGED_WORK_OPEN


@pytest.mark.parametrize("flags", [
    {"_split_running": True}, {"_smooth_running": True},
    {"_interp_snapshot": object()}, {"_merge_dialog": object()},
])
def test_add_start_refuses_over_any_open_edit(flags):
    f = make("_start_add", **flags)
    f._ensure_shared_data_aligned = lambda *a: pytest.fail("touched volume")
    K._start_add(f)
    assert _texts(f, "add") == [STAGED_WORK_OPEN]
    assert f.adv.focused == ["Add"]
    assert f._add_running is False


def test_a_refusal_shows_and_raises_a_hidden_advanced_dock():
    """napari's X only hides the Advanced Features dock; a door that brings
    a tab forward shows the dock again instead of focusing a hidden page."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

    app = QApplication.instance() or QApplication([])

    class _Panel(QWidget):
        def __init__(self):
            super().__init__()
            self.focused = []

        def focus_tab(self, name):
            self.focused.append(name)
            return True

    main = QMainWindow()
    try:
        panel = _Panel()
        dock = QDockWidget("adv", main)
        dock.setWidget(panel)
        main.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.hide()
        f = make(_split_running=True)
        f.adv = panel
        f._refuse_for_open_edit("add start", "Add", f.said.append)
        assert panel.focused == ["Add"]
        assert not dock.isHidden()
        assert f.said == [STAGED_WORK_OPEN]
    finally:
        main.close()
        main.deleteLater()
        app.processEvents()


def test_add_another_refuses_before_clearing_the_review():
    f = make(_add_review=True, _smooth_running=True)
    f._start_add = lambda **k: pytest.fail("started")
    K._add_another(f)
    assert f._add_review is True
    assert _texts(f, "add") == [STAGED_WORK_OPEN]


def test_undo_all_refuses_over_an_open_refine_and_never_calls_the_controller():
    f = make(_smooth_running=True, _add_session=[{"ids": [1]}])
    f._on_add_undo_session_cb = lambda n: pytest.fail("undid")
    K._undo_add_all(f)
    assert _texts(f, "add") == [STAGED_WORK_OPEN]


def test_undo_all_is_not_blocked_by_its_own_review():
    f = make(_add_review=True, _add_session=[])
    # REVIEW is not an open edit: the refusal helper lets it through
    assert f._add_refused_while_staged("undo all") is False
    assert f.said == []


def test_batch_keeps_its_own_add_sentence():
    from packages.epicycle_annotator.batch_refine import ADD_BATCH_BUSY

    f = make(_batch_staged=True)
    assert f._add_refused_while_staged("start") is True
    assert _texts(f, "add") == [ADD_BATCH_BUSY]


def test_add_start_greys_the_label_buttons_via_the_refreshers():
    src = __import__("inspect").getsource(K._start_add)
    assert src.index("_add_running = True") < src.index(
        "_add_refresh_label_buttons()")


# --- start rules for Reshape / Refine / Interpolate -----------------------

def test_reshape_start_refuses_over_add_and_over_a_held_interpolation():
    for flags in ({"_add_running": True}, {"_interp_snapshot": object()}):
        f = make(**flags)
        K._start_split(f)
        assert _texts(f, "reshape") == [STAGED_WORK_OPEN]
        assert f.adv.focused == ["Reshape"]
        assert f._split_running is False


def test_refine_preview_refuses_over_add_and_over_a_held_interpolation():
    for flags in ({"_add_running": True}, {"_interp_snapshot": object()}):
        f = make(**flags)
        K._preview_smooth(f)
        assert _texts(f, "refine") == [STAGED_WORK_OPEN]
        assert f.adv.focused == ["Refine"]


def test_refine_preview_is_exempt_inside_refine_further():
    f = make(_interp_snapshot=object(), _interp_refine_mode=True,
             _ensure_shared_data_aligned=lambda *a: None)
    K._preview_smooth(f)
    # it got past the open-edit gate and stopped at the later volume check
    assert STAGED_WORK_OPEN not in _texts(f, "refine")


def test_interpolate_refuses_over_add_and_over_a_held_snapshot():
    for flags in ({"_add_running": True}, {"_interp_snapshot": object()}):
        f = make(**flags)
        f._interp_status_label = SimpleNamespace(
            setText=lambda t: f.said.append(("interp", {"status_text": t})))
        K._run_interpolation(f)
        assert _texts(f, "interp") == [STAGED_WORK_OPEN]
        assert f._interp_running is False


# --- the annotator's patch Switch --------------------------------------------

class _Nav:
    def __init__(self, f):
        self.f = f

    def _show_op_active_status(self, text="", msec=2000):
        self.f.said.append(("header", {"status_text": text}))


def _switch(f):
    f._navigator_widget = _Nav(f)
    f._say_in_navigator_header = lambda t: K._say_in_navigator_header(f, t)
    f._reject_smooth = lambda: pytest.fail("silently threw the Refine away")
    f._interp_status_label = SimpleNamespace(
        setText=lambda t: f.said.append(("interp", {"status_text": t})))
    # a refusal returns before the base switch (which would fail on a fake)
    K._switch_patch_context_from_combo(f)
    return []


@pytest.mark.parametrize("flags,slot,tab", [
    ({"_smooth_running": True}, "refine", "Refine"),
    ({"_split_running": True}, "reshape", "Reshape"),
    ({"_add_running": True}, "add", "Add"),
    ({"_interp_snapshot": object()}, "interp", None),
    ({"_batch_staged": True}, "header", None),
    ({"_merge_dialog": object()}, "header", None),
])
def test_switch_refuses_over_any_open_edit(flags, slot, tab):
    f = make(**flags)
    assert _switch(f) == []
    assert any(s == slot for s, _ in (x for x in f.said if isinstance(x, tuple)))
    assert (f.adv.focused == [tab]) if tab else (f.adv.focused == [])


def test_switch_with_split_and_add_keeps_the_older_sentences():
    f = make(_split_running=True)
    _switch(f)
    assert _texts(f, "reshape") == [
        "Complete or cancel split before switching patches."]
    f = make(_add_running=True)
    _switch(f)
    assert _texts(f, "add") == [
        "Finish or cancel Add before switching patches."]


def test_switch_still_reaches_the_base_switch_when_nothing_is_open():
    import inspect

    src = inspect.getsource(K._switch_patch_context_from_combo)
    assert "if kind is not None:" in src
    assert src.rindex("super()._switch_patch_context_from_combo()") > src.index(
        "return")
