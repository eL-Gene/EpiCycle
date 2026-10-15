"""Shift+arrow nudges the Add box.

The two real side-view key filters, ``_SliceStepFilter`` and
``_UpDownBlocker``, fed real ``QKeyEvent``s, with the real
``_add_nudge_box`` on the Qt-free harness as their nudge callback:

* plain Up/Down still navigate labels, plain Left/Right still step the
  slice (step > 1) or fall through to napari (step 1) -- unchanged;
* Shift+arrow outside Add (no session, another tab, REVIEW) behaves as
  today;
* Shift+arrow in Add moves the box one voxel along that view's displayed
  axis, is consumed, and does not navigate (the selection is kept);
* busy: consumed, refused with ``ADD_S1_BUSY_REFUSED``, box unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication

from napari_epicycle.annotator import curation_3d_widget as CW
from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes, marker_zyx
from .add_widget_fixture import qapp, widget  # noqa: F401

_APP = QApplication.instance() or QApplication([])

KEYS = {"Up": Qt.Key_Up, "Down": Qt.Key_Down,
        "Left": Qt.Key_Left, "Right": Qt.Key_Right}


def _key(name, shift=False):
    mods = Qt.ShiftModifier if shift else Qt.NoModifier
    return QKeyEvent(QEvent.KeyPress, KEYS[name], mods)


class _Dims:
    def __init__(self):
        self.current_step = (5,)
        self.range = ((0, 20, 1),)
        self.set_calls = []

    def set_current_step(self, axis, value):
        self.set_calls.append((axis, value))
        self.current_step = (value,)


class _VM:
    def __init__(self):
        self.dims = _Dims()


class _Step:
    def __init__(self, v):
        self.v = v

    def value(self):
        return self.v


class _Tabs:
    def __init__(self):
        self.text = "Add"

    def currentIndex(self):  # noqa: N802 - Qt name
        return 0

    def tabText(self, _i):  # noqa: N802 - Qt name
        return self.text


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((40, 60, 80), np.int32))
    hh.adv._tool_tabs = _Tabs()
    hh.place_box((slice(10, 30), slice(20, 40), slice(30, 60)))
    hh.nav = []
    return hh


def _filters(h, orient, step=1):
    vm = _VM()
    nav = h.nav.append
    f = CW._SliceStepFilter(vm, _Step(step), navigate_callback=nav,
                            nudge_callback=h._add_nudge_box, orient=orient)
    b = CW._UpDownBlocker(navigate_callback=nav,
                          nudge_callback=h._add_nudge_box, orient=orient)
    return f, b, vm


def _centre(h):
    b = h._add_box
    return (b.cz, b.cy, b.cx)


# --------------------------------------------------------------------------
# unchanged
# --------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["filter", "blocker"])
def test_plain_up_down_still_navigate_labels(h, which):
    f, b, _vm = _filters(h, "XY")
    flt = f if which == "filter" else b
    assert flt.eventFilter(None, _key("Up")) is True
    assert flt.eventFilter(None, _key("Down")) is True
    assert h.nav == ["prev", "next"]
    assert _centre(h) == (20, 30, 45)


def test_plain_left_right_still_step_the_slice(h):
    f, b, vm = _filters(h, "XY", step=3)
    assert f.eventFilter(None, _key("Right")) is True
    assert vm.dims.set_calls == [(0, 8)]
    assert f.eventFilter(None, _key("Left")) is True
    assert vm.dims.set_calls[-1] == (0, 5)
    assert b.eventFilter(None, _key("Left")) is False   # never the blocker's
    assert _centre(h) == (20, 30, 45)


def test_plain_left_right_at_step_one_fall_through_to_napari(h):
    f, _b, vm = _filters(h, "XY", step=1)
    assert f.eventFilter(None, _key("Right")) is False
    assert vm.dims.set_calls == []


def test_filters_built_without_a_nudge_callback_are_as_today(h):
    vm = _VM()
    f = CW._SliceStepFilter(vm, _Step(1), navigate_callback=h.nav.append)
    b = CW._UpDownBlocker(navigate_callback=h.nav.append)
    assert f.eventFilter(None, _key("Up", shift=True)) is True
    assert b.eventFilter(None, _key("Down", shift=True)) is True
    assert h.nav == ["prev", "next"]
    assert _centre(h) == (20, 30, 45)


# --------------------------------------------------------------------------
# Shift outside Add: as today
# --------------------------------------------------------------------------

def test_shift_up_with_no_add_session_navigates_as_today(h):
    h._add_running = False
    f, b, _vm = _filters(h, "XY")
    assert f.eventFilter(None, _key("Up", shift=True)) is True
    assert b.eventFilter(None, _key("Up", shift=True)) is True
    assert h.nav == ["prev", "prev"]


def test_shift_arrow_on_another_tab_is_as_today(h):
    h.adv._tool_tabs.text = "Refine"
    f, _b, vm = _filters(h, "XY", step=2)
    assert f.eventFilter(None, _key("Down", shift=True)) is True
    assert f.eventFilter(None, _key("Right", shift=True)) is True
    assert h.nav == ["next"]
    assert vm.dims.set_calls == [(0, 7)]
    assert _centre(h) == (20, 30, 45)


def test_shift_arrow_in_review_is_as_today(h):
    h._add_review = True
    f, _b, _vm = _filters(h, "XY")
    assert f.eventFilter(None, _key("Up", shift=True)) is True
    assert h.nav == ["prev"]
    assert _centre(h) == (20, 30, 45)


# --------------------------------------------------------------------------
# Shift in Add: the nudge
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orient,key,centre", [
    ("XY", "Up", (20, 29, 45)), ("XY", "Down", (20, 31, 45)),
    ("XY", "Left", (20, 30, 44)), ("XY", "Right", (20, 30, 46)),
    ("XZ", "Up", (19, 30, 45)), ("XZ", "Down", (21, 30, 45)),
    ("XZ", "Left", (20, 30, 44)), ("XZ", "Right", (20, 30, 46)),
    ("YZ", "Up", (20, 29, 45)), ("YZ", "Down", (20, 31, 45)),
    ("YZ", "Left", (19, 30, 45)), ("YZ", "Right", (21, 30, 45)),
])
@pytest.mark.parametrize("which", ["filter", "blocker"])
def test_shift_arrow_in_add_nudges_one_voxel(h, orient, key, centre, which):
    f, b, vm = _filters(h, orient, step=3)
    flt = f if which == "filter" else b
    assert flt.eventFilter(None, _key(key, shift=True)) is True
    assert _centre(h) == centre
    assert h.nav == [], "the label selection is kept"
    assert vm.dims.set_calls == [], "the slice did not step"
    assert h.adv.last("box_center") == centre
    assert (h._add_box.sz, h._add_box.sy, h._add_box.sx) == (20, 20, 30)


def test_shift_nudge_at_the_volume_edge_is_consumed_and_stays(h):
    h.place_box((slice(0, 20), slice(20, 40), slice(30, 60)))
    f, _b, _vm = _filters(h, "XZ")
    before = _centre(h)
    for _ in range(5):
        assert f.eventFilter(None, _key("Up", shift=True)) is True
    b = h._add_box.bounds_dict((40, 60, 80))
    assert b["z0"] == 0 and b["z1"] - b["z0"] == 20
    assert _centre(h) == before
    assert h.nav == []


def test_shift_nudge_while_busy_is_refused_and_consumed(h):
    h._add_busy = True
    f, b, _vm = _filters(h, "XY")
    assert f.eventFilter(None, _key("Up", shift=True)) is True
    assert b.eventFilter(None, _key("Left", shift=True)) is True
    assert _centre(h) == (20, 30, 45)
    assert h._add_transient == AP.ADD_S1_BUSY_REFUSED
    assert h.nav == []


def test_a_nudge_drops_clicks_that_leave_the_box(h):
    h._add_click_markers = [(20, 20, 45), (20, 30, 45)]
    f, _b, _vm = _filters(h, "XY")
    f.eventFilter(None, _key("Down", shift=True))      # y 21..41
    assert marker_zyx(h) == [(20, 30, 45)]


def test_key_release_and_other_keys_are_untouched(h):
    f, b, _vm = _filters(h, "XY")
    rel = QKeyEvent(QEvent.KeyRelease, Qt.Key_Up, Qt.ShiftModifier)
    assert f.eventFilter(None, rel) is False
    other = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.ShiftModifier)
    assert f.eventFilter(None, other) is False
    assert b.eventFilter(None, other) is False
    assert _centre(h) == (20, 30, 45)


def test_xz_shift_down_on_a_box_touching_the_bottom_stays(h):
    """B3 fix pass: a keep-size box on the bottom edge is not pushed UP."""
    h.place_box((slice(20, 40), slice(20, 40), slice(30, 60)))
    f, _b, _vm = _filters(h, "XZ")
    before = h._add_box
    assert f.eventFilter(None, _key("Down", shift=True)) is True
    assert h._add_box == before
    assert h._add_box.bounds_dict((40, 60, 80))["z1"] == 40


def test_a_nudge_moves_only_its_own_axis(h):
    """The slicing axis and the view's other axis are never normalised by
    a nudge, even when the box overflows there."""
    from packages.epicycle_annotator.roi_box import BoxRec

    h._add_box = BoxRec(38, 30, 45, 6, 20, 30)          # z overflows
    f, _b, _vm = _filters(h, "XY")
    assert f.eventFilter(None, _key("Left", shift=True)) is True
    b = h._add_box
    assert (b.cz, b.sz, b.cy) == (38, 6, 30)
    assert b.cx == 44



# --------------------------------------------------------------------------
# B3 fix pass: a failing nudge, the ball render, the filters on re-install
# --------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["filter", "blocker"])
def test_a_failing_nudge_is_still_consumed(h, which):
    """LOW-7: once Add was ready the key is Add's -- a nudge that raises
    must not ALSO navigate labels."""
    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    h._on_add_box_changed = _boom
    f, b, _vm = _filters(h, "XY")
    flt = f if which == "filter" else b
    assert flt.eventFilter(None, _key("Up", shift=True)) is True
    assert h.nav == []


def test_a_nudge_keeping_every_click_renders_no_balls(h):
    """Item 13: a pure translation never moves a ball -- the full-volume
    ball layer is re-rendered only when a click is dropped."""
    renders = []
    # Only the dropped balls are rewritten (``changed=``)
    h._render_add_balls = lambda changed=None: renders.append(
        [tuple(m["zyx"]) for m in changed] if changed is not None else 1)
    h._add_click_markers = [(20, 30, 45)]
    f, _b, _vm = _filters(h, "XY")
    for _ in range(3):
        assert f.eventFilter(None, _key("Down", shift=True)) is True
    assert renders == []
    assert marker_zyx(h) == [(20, 30, 45)]
    h._add_click_markers = [(20, 23, 45), (20, 30, 45)]    # y 23..43 now
    assert f.eventFilter(None, _key("Down", shift=True)) is True
    assert marker_zyx(h) == [(20, 30, 45)]
    assert renders == [[(20, 23, 45)]]


class _Canvas:
    def __init__(self, native):
        self.native = native


def _fake_qt_viewer():
    from qtpy.QtWidgets import QWidget

    w = QWidget()
    native = QWidget(w)
    QWidget(w)                                   # another child
    w.canvas = _Canvas(native)
    return w


def _filter_host():
    from qtpy.QtWidgets import QSpinBox, QWidget

    class _Host(QWidget):
        _install_arrow_key_filters = CW.Curation3DAnnotator.\
            _install_arrow_key_filters
        _remove_arrow_key_filters = CW.Curation3DAnnotator.\
            _remove_arrow_key_filters

        def _block_main_viewer_arrow_keys(self):
            pass

        def _on_subviewer_navigate(self, _d):
            pass

        def _add_nudge_box(self, _o, _k):
            return False

    host = _Host()
    host._slice_step_sb = QSpinBox(host)
    for o in ("xy", "xz", "yz"):
        setattr(host, f"qt_viewer_{o}", _fake_qt_viewer())
        setattr(host, f"viewer_model_{o}", _VM())
    return host


def _alive(obj):
    from qtpy import sip

    return not sip.isdeleted(obj)


def test_reinstalling_the_arrow_filters_leaves_exactly_one_set():
    """LOW-3: ``_install_arrow_key_filters`` runs on every patch switch; the
    previous set comes off (``removeEventFilter`` + ``deleteLater``)."""
    from qtpy.QtCore import QCoreApplication, QEvent as _QE

    host = _filter_host()
    host._install_arrow_key_filters()
    first = [f for f, _t in host._arrow_key_filters]
    assert len(first) == 6            # 3 slice filters + 3 blockers
    host._install_arrow_key_filters()
    second = [f for f, _t in host._arrow_key_filters]
    assert len(second) == 6
    assert not set(map(id, first)) & set(map(id, second))
    QCoreApplication.sendPostedEvents(None, _QE.DeferredDelete)
    assert not any(_alive(f) for f in first), "the old set is deleted"
    assert all(_alive(f) for f in second)
    # one blocker per view, on the viewer AND its children
    for f, targets in host._arrow_key_filters:
        if isinstance(f, CW._UpDownBlocker):
            assert len(targets) == 3
    host._remove_arrow_key_filters()
    host._remove_arrow_key_filters()
    assert host._arrow_key_filters == []



def test_each_arrow_filter_carries_its_own_views_orient():
    """Item 15: the filter on ``qt_viewer_xz`` nudges as XZ, the one on
    ``qt_viewer_yz`` as YZ -- a swap must fail here."""
    host = _filter_host()
    host._install_arrow_key_filters()
    seen = {}
    for f, targets in host._arrow_key_filters:
        for o in ("xy", "xz", "yz"):
            qv = getattr(host, f"qt_viewer_{o}")
            if qv in targets or qv.canvas.native in targets:
                seen.setdefault(o.upper(), set()).add(f._orient)
    assert seen == {"XY": {"XY"}, "XZ": {"XZ"}, "YZ": {"YZ"}}
    host._remove_arrow_key_filters()


# --------------------------------------------------------------------------- #
# A5 -- ticking a marker row does not steal Shift+arrow
# --------------------------------------------------------------------------- #
def test_a_marker_list_click_leaves_the_focus_where_it_was(widget):
    """The marker list (and the candidate list) are NoFocus: a click that
    ticks a row never takes the keyboard off the side view, so the next
    Shift+arrow still reaches the side view's key filter."""
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QLineEdit

    from .add_widget_fixture import reset_add

    reset_add(widget)
    widget.set_add_preset("cells", silent=True)
    widget.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_NO_RUN, False, ("labels", None)),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))], box_active=True)
    widget.add_marker_source_combo.setCurrentIndex(2)
    widget.add_markers_btn.setChecked(True)
    seen = []
    widget.set_add_callbacks(on_click_tick=lambda i, t: seen.append((i, t)))
    widget.update_add_ui_state(click_count=2, click_markers=[
        (8, (1, 2, 3), True), (9, (4, 5, 6), True)])
    widget.resize(340, 900)
    widget.focus_tab("Add")
    widget.show()
    side = QLineEdit(widget)          # stands in for the side view's canvas
    side.setFocus()
    _APP.processEvents()
    before = _APP.focusWidget()
    lst = widget.add_click_list
    assert not lst.isHidden()
    rect = lst.visualItemRect(lst.item(1))
    for x in range(rect.left() + 2, rect.left() + 30, 3):
        QTest.mouseClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier,
                         rect.center().__class__(x, rect.center().y()))
        _APP.processEvents()
        if seen:
            break
    assert seen == [(9, False)], "the click really unticked the row"
    assert _APP.focusWidget() is before
    assert not lst.hasFocus()
    widget.update_add_ui_state(click_count=0, click_markers=[])
    side.deleteLater()
    widget.hide()
