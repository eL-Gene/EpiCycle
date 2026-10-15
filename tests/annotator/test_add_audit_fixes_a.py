"""The combined fix pass after the A1-A4 audit (group A).

* A-M1 -- napari re-raises "Contours are not displayed during 3D rendering"
  on EVERY refresh of a contoured 3D layer (a refill, a position move), not
  only on the ``contour`` write; while the Add outline is on a live
  candidate layer that one message is filtered, and the filter goes with
  the layer.
* A-M2 -- the amber default-spacing note names QC Features -> Settings...,
  which is GREY while a box holds the Step 3 lock: with a box it says
  Cancel first.
* A-L1 / A-L2 -- "Accept and refine" follows every ``accept_enabled`` push,
  and both footer buttons go grey at the top of the two Accept paths.
* A-L3 -- the spacing notice's box branch: nothing while busy, the
  candidates invalidated, nothing when the spacing did not change; a QC
  dialog Cancel notifies nothing.
* A-L4 / A-L5 / A-L7 -- no patch is amber in both places; the amber is the
  theme's gold; the help line IS the tip constant.
"""

from __future__ import annotations

import types
import warnings

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread, Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


# --------------------------------------------------------------------------- #
# A-M1: the warning on refresh and on a position move
# --------------------------------------------------------------------------- #
class _Fake:
    pass


def _contour_fake():
    from napari.components import ViewerModel
    from napari.layers import Labels

    v = ViewerModel()
    v.dims.ndisplay = 3
    cand = Labels(np.zeros((6, 8, 8), np.int32),
                  name="_epicycle_add_candidate")
    v.layers.append(cand)
    f = _Fake()
    f.viewer = v
    f._add_candidate_layer = cand
    f._add_contour_on = False
    f.sync_manager = None
    f.subviewer_manager = types.SimpleNamespace(
        remove_layer_from_subviewers=lambda *_a: None)
    f._push_add_ui = lambda **_k: None
    K = _K()
    for n in ("_on_add_contour_toggled", "_add_set_contour",
              "_remove_add_candidate", "_add_put_outline_down"):
        setattr(f, n, getattr(K, n).__get__(f, _Fake))
    return f, v, cand


def _contour_msgs(rec):
    return [r for r in rec
            if "Contours are not displayed" in str(r.message)]


@pytest.fixture
def outline():
    from napari_epicycle.annotator import curation_3d_widget as CW

    f, v, cand = _contour_fake()
    yield f, v, cand
    CW._add_outline_filter(False)


def test_a_refresh_with_the_outline_on_emits_no_warning(outline):
    f, _v, cand = outline
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f._on_add_contour_toggled(True)
        cand.refresh()
    assert cand.contour == 1
    assert _contour_msgs(rec) == []


def test_a_position_move_with_the_outline_on_emits_no_warning(outline):
    f, v, _cand = outline
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f._on_add_contour_toggled(True)
        v.dims.current_step = (3, 0, 0)
        v.dims.current_step = (4, 0, 0)
    assert _contour_msgs(rec) == []


def test_the_filter_goes_with_the_layer(outline):
    """Removed at the candidate's removal: the same napari message from a
    layer that is not the Add outline gets out again."""
    from napari.layers import Labels

    f, v, _cand = outline
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f._on_add_contour_toggled(True)
        f._remove_add_candidate()
        other = Labels(np.zeros((6, 8, 8), np.int32), name="other")
        v.layers.append(other)
        other.contour = 1
        other.refresh()
    assert _contour_msgs(rec), "no Add outline, no filter"


def test_the_filter_goes_when_the_outline_goes_off(outline):
    f, _v, cand = outline
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f._on_add_contour_toggled(True)
        f._on_add_contour_toggled(False)
        cand.contour = 2          # someone else's contour, after ours
    assert _contour_msgs(rec)


def test_a_different_warning_still_gets_out(outline):
    f, _v, _cand = outline
    f._on_add_contour_toggled(True)
    with pytest.warns(UserWarning, match="something else"):
        warnings.warn("something else")


def test_the_refill_path_filters_its_refresh():
    import inspect

    src = inspect.getsource(_K()._render_add_candidate)
    head = src.split("cand.refresh()")[0]
    assert "catch_warnings" in head.split("cand.data[roi] = result")[-1]


# --------------------------------------------------------------------------- #
# A-M2: the note while a box is placed
# --------------------------------------------------------------------------- #
def test_the_boxed_spacing_note_says_cancel_first(widget):
    reset_add(widget)
    widget.set_add_preset("cells", silent=True)
    try:
        widget.update_add_ui_state(default_spacing=True, box_active=False)
        assert not widget.add_spacing_note.isHidden()
        assert widget.add_spacing_note.text() == AP.ADD_NOTE_DEFAULT_SPACING
        widget.update_add_ui_state(box_active=True)
        assert not widget.add_spacing_note.isHidden()
        assert widget.add_spacing_note.text() == (
            AP.ADD_NOTE_DEFAULT_SPACING_BOXED)
        assert "Cancel the box" in AP.ADD_NOTE_DEFAULT_SPACING_BOXED
        assert "QC Features → Settings…" in AP.ADD_NOTE_DEFAULT_SPACING_BOXED
        widget.update_add_ui_state(box_active=False)
        assert widget.add_spacing_note.text() == AP.ADD_NOTE_DEFAULT_SPACING
    finally:
        widget.update_add_ui_state(default_spacing=False, box_active=False)
        reset_add(widget)


# --------------------------------------------------------------------------- #
# A-L1: the mirror
# --------------------------------------------------------------------------- #
def test_accept_and_refine_follows_every_accept_push(widget):
    reset_add(widget)
    ids = list(range(400, 404))
    widget.update_add_ui_state(box_active=True, review_active=False)
    widget.update_add_ui_state(candidate_ids=ids,
                               candidate_sizes={i: 9 for i in ids},
                               candidate_edge_ids=set())
    try:
        assert widget.add_accept_refine_btn.isEnabled()
        widget.update_add_ui_state(accept_enabled=False, accept_count=0)
        assert not widget.add_accept_btn.isEnabled()
        assert not widget.add_accept_refine_btn.isEnabled(), (
            "the voxels-taken refusal left it enabled")
        widget.update_add_ui_state(accept_enabled=True)
        assert widget.add_accept_refine_btn.isEnabled()
    finally:
        widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                                   candidate_edge_ids=set(),
                                   box_active=False)


# --------------------------------------------------------------------------- #
# A-L2: the re-entry belt
# --------------------------------------------------------------------------- #
@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((20, 30, 30), np.int32), images={})
    hh.place_box((slice(2, 18), slice(2, 28), slice(2, 28)))
    res = np.zeros((16, 26, 26), np.int32)
    res[2:6, 2:6, 2:6] = 40
    hh._add_preview_result = res
    hh._add_preview_roi = tuple(hh._add_roi)
    hh.adv.checked = [40]
    hh._on_add_committed_cb = None
    hh._batch_running = False
    hh._batch_staged = False
    hh._open_batch_refine_dialog = lambda **_k: None
    return hh


@pytest.mark.parametrize("door", ["_accept_add", "_accept_add_and_refine"])
def test_both_footer_buttons_go_grey_first(h, door):
    getattr(h, door)()
    first = next(kw for kw in h.adv.pushes if "accept_enabled" in kw)
    assert first == {"accept_enabled": False}, first


@pytest.mark.parametrize("door", ["_accept_add", "_accept_add_and_refine"])
def test_a_batch_refusal_leaves_the_buttons_alone(h, door):
    h._batch_running = True
    getattr(h, door)()
    assert all("accept_enabled" not in kw for kw in h.adv.pushes)


# --------------------------------------------------------------------------- #
# A-L3: the spacing notice's box branch
# --------------------------------------------------------------------------- #
@pytest.fixture
def hs(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((12, 20, 20), np.int32), images={},
                 spacing=(1.0, 1.0, 1.0))
    hh.adv.preset = "cells"
    hh.place_box((slice(2, 10), slice(2, 18), slice(2, 18)))
    return hh


def test_busy_changes_nothing(hs):
    hs._add_busy = True
    hs._add_dark_cache = {"k": 1}
    n = len(FakeThread.instances)
    hs.ctx["spacing"] = (0.5, 0.517, 0.517)
    hs._add_spacing_changed()
    assert hs._add_dark_cache == {"k": 1}
    assert len(FakeThread.instances) == n


def test_candidates_up_are_invalidated(hs):
    hs._add_preview_result = np.ones((8, 16, 16), np.int32)
    hs._add_preview_roi = tuple(hs._add_roi)
    hs.ctx["spacing"] = (0.5, 0.517, 0.517)
    hs._add_spacing_changed()
    assert hs._add_preview_result is None, "made at the old spacing"
    assert hs.adv.last("accept_enabled") is False


def test_an_unchanged_spacing_re_measures_nothing(hs):
    hs._add_info_spacing = (1.0, 1.0, 1.0)
    hs._add_dark_cache = {"k": 1}
    hs._add_preview_result = np.ones((8, 16, 16), np.int32)
    n = len(FakeThread.instances)
    hs._add_spacing_changed()
    assert hs._add_dark_cache == {"k": 1}
    assert len(FakeThread.instances) == n
    assert hs._add_preview_result is not None
    assert hs.adv.last("default_spacing") is True, "the note still follows"


def test_the_info_launch_records_its_spacing(hs):
    hs.ctx["spacing"] = (0.5, 0.517, 0.517)
    hs._request_add_box_info()
    assert hs._add_info_spacing == (0.5, 0.517, 0.517)


def test_a_qc_settings_cancel_notifies_nothing(monkeypatch):
    from napari_epicycle.fusion_reconstruction import controller as CTRL
    from napari_epicycle.fusion_reconstruction import qc_extraction_dialog

    class _Dlg:
        Accepted = 1

        def __init__(self, *a, **k):
            self.spacing_auto_btn = types.SimpleNamespace(
                clicked=types.SimpleNamespace(connect=lambda *_a: None))

        def exec_(self):
            return 0                                   # Cancel

        def get_settings(self):
            raise AssertionError("a Cancel reads no settings")

    monkeypatch.setattr(qc_extraction_dialog, "QCExtractionDialog", _Dlg)
    told = []
    widget = types.SimpleNamespace(
        notify_spacing_changed=lambda: told.append(True))
    fake = types.SimpleNamespace(
        ui=types.SimpleNamespace(), _qc_settings={"spacing": None},
        viewer=None, _qc_features_df=None,
        _append_log=lambda *_a: None,
        _detect_available_channels=lambda: [],
        _check_qc_extraction_staleness=lambda: None,
        _maybe_show_qc_staleness_notification=lambda: None,
        _get_curation_annotator_widget=lambda: widget)
    K = CTRL.FusionReconstructionController
    fake._notify_spacing_changed = K._notify_spacing_changed.__get__(fake)
    K._open_qc_settings(fake)
    assert told == []
    assert fake._qc_settings == {"spacing": None}


# --------------------------------------------------------------------------- #
# A-L4 / A-L5 / A-L7
# --------------------------------------------------------------------------- #
def test_no_patch_is_warn_in_the_table_too():
    s = {"has_patch": False}
    assert AP.s1_sentence(s) == AP.ADD_S1_NO_PATCH
    assert AP.s1_kind(s) == "warn"


def test_the_amber_is_the_themes_gold():
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_S1_WARN_STYLE,
    )
    from napari_epicycle.ui_shell.theme import PANEL_GOLD_HOVER

    assert ADD_S1_WARN_STYLE == "color: {}; font-size: 10px;".format(
        PANEL_GOLD_HOVER)
    assert ADD_S1_WARN_STYLE == "color: #e6c26a; font-size: 10px;"


def test_the_help_line_is_the_tip_constant():
    from napari_epicycle.annotator import advanced_features_widget as AFW

    assert (AFW.ADD_HELP_LINES["add_accept_refine_btn"]
            is AP.ADD_ACCEPT_REFINE_TIP)
