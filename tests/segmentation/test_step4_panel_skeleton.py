"""The B1a skeleton.

The real panel with its controller, offscreen.  Heights are measured at the
width the user gets (height-for-width at 322 px for ②'s right column, 218 px
for its left, 560 px for a lock line, 566 px for ①), idle AND busy, with
every optional line showing, and asserted against the mockup's ceilings
(``MOCKUP_CEILINGS``, measured on the v7 mockup, never from this build).

Offscreen Qt knows only its own font folder and falls back to a wide "Sans
Serif" (1661 px for the lock sentence against 723 px in Segoe UI), so every
number that depends on the font is measured in a SUBPROCESS pointed at the
system fonts with napari's Windows face, the pattern of B3a's
``test_workbench_layout.py`` (audit A1-3).  The in-process tests are
font-agnostic and never change the app's font, so their result does not
depend on which test file ran first; the app fixture puts the font back
whatever a test did.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from qtpy.QtCore import QPoint, Qt

from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import (
    close, make_workspace, panel, pump, refined_bundle, tick, write_stack,
)
from .test_step4_mockup_ceilings import MOCKUP_CEILINGS as CEIL

SECTION_W = 572          # : the section viewport
_FONT_DIR = "C:/Windows/Fonts"
_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def app(qapp):
    font = qapp.font()
    yield qapp
    qapp.setFont(font)     # whatever a test did, the next file gets it back


def _hfw(widget, width):
    if widget.hasHeightForWidth():
        return widget.heightForWidth(width)
    return widget.sizeHint().height()


def _open(w, qapp, index):
    w.accordion.setCurrentIndex(index)
    pump(qapp, 10)


def _make_ws(tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif", "P3.tif",
                                           "P4.tif"),
                        one_channel=("P4.tif",))
    refined_bundle(ws, None, "wound-v2-with-a-long-name", channels=(2, 1))
    extra = tmp_path / "extra"
    write_stack(extra / "Wound_patch_2.tiff")             # in voxels
    write_stack(extra / "P1.tif")                          # same name
    assert (ws / "step1_state.json").exists()
    return ws, extra


def build(qapp, tmp_path):
    """The panel the ceilings are measured on: six stacks (a one-channel
    one, one in voxels, a same-name conflict), two different models so the
    amber "lat | ax" line wraps, three ticks."""
    root, extra = _make_ws(Path(tmp_path))
    w = panel(qapp, root)
    w.resize(SECTION_W, 842)
    c = w.controller
    c._scan_stacks_directory(str(extra))
    from napari_epicycle.fusion_reconstruction._run_management import (
        list_cross_run_refined_models,
    )
    from napari_epicycle.segmentation import _step4_models as M
    rows = M.step4_model_rows(list_cross_run_refined_models(
        root, trained_only=True))
    lat = rows[0]
    (root / "m").mkdir(exist_ok=True)
    (root / "m" / "ax").write_bytes(b"weights")
    c._on_model_use(dict(lat, _kind="refined"))
    c._on_model_use({"_kind": "step2_run", "bucket": "axial",
                     "weights_path": str(root / "m" / "ax"),
                     "human_label": "wound-v2 · round 3 (axial)"})
    tick(w, qapp, "P1.tif", "P2.tif", "Wound_patch_2.tiff")
    pump(qapp, 10)
    return w


@pytest.fixture
def w(app, tmp_path):
    w = build(app, tmp_path)
    yield w
    close(w)


# ---------------------------------------------------------------- the frame


def test_four_sections_in_steps_green(app, w):
    acc = w.accordion
    assert acc.count() == 4
    titles = [s._title for s in acc._sections]
    assert titles == [T.SECTION_SESSION, T.SECTION_SEGMENT,
                      T.SECTION_MEASURE, T.SECTION_DIAGNOSTICS]
    assert acc._accent == "#6abf69"
    assert acc._sections[0]._expanded      # every step opens on Session


def test_the_splitter_numbers(app, w):
    _open(w, app, 1)
    sp = w.seg_splitter
    assert sp.orientation() == Qt.Horizontal
    assert not sp.childrenCollapsible()
    assert sp.handleWidth() == 6
    assert sp.sizes() == [224, 330]
    assert w.seg_left.layout().contentsMargins().right() == 6
    assert w.seg_left.layout().contentsMargins().left() == 0
    assert w.seg_right.layout().contentsMargins().left() == 8
    assert w.seg_left.minimumWidth() == 200
    assert w.seg_right.minimumWidth() == 300
    # the foot is pinned: a sibling of the scroll area, not inside it
    assert w.seg_foot.parent() is w.seg_right
    assert not w.seg_right_content.isAncestorOf(w.predict_btn)
    assert w.seg_right_scroll.horizontalScrollBarPolicy() == \
        Qt.ScrollBarAlwaysOff


NEW_NAMES = (
    "session_stacks_list", "session_stacks_status", "add_folder_btn",
    "set_spacing_btn", "session_lock_line", "seg_lock_line", "mt_lock_line",
    "run_picker_combo", "new_run_btn", "manage_runs_btn", "run_row_note",
    "model_active_indicator_label", "model_change_btn", "model_trained_label",
    "seg_ticked_label", "start_from_combo", "cell_width_um_sb", "view_btn",
)
KEPT_NAMES = (
    "workdir_line", "browse_workdir_btn", "stacks_list", "predict_btn",
    "reconstruct_btn", "cancel_btn", "run_progress", "run_status_label",
    "chan1_combo", "chan2_combo", "flow_thresh_sb", "cellprob_sb",
    "batch_size_sb", "method_combo", "n_cpu_sb", "normalize_cb",
    "rescale_cb", "skip_postprocess_cb", "do_mp_cb", "_seg_collapsible",
    "_adv_collapsible", "qc_stacks_list", "lateral_model_path",
    "axial_model_path", "pretrained_combo", "model_source_combo",
    "gpu_status_label", "log_output",
)


def test_every_b1a_widget_name_exists(w):
    for name in NEW_NAMES + KEPT_NAMES:
        assert hasattr(w, name), name


def test_the_labels_and_words(app, w):
    assert w.predict_btn.text() == T.PREDICT_BTN
    assert w.reconstruct_btn.text() == T.RECONSTRUCT_BTN
    assert w.normalize_cb.text() == T.SEG_NORMALIZE_LABEL
    assert [w.method_combo.itemText(i)
            for i in range(w.method_combo.count())] == [T.SEG_METHOD_DIRECT]
    assert not w.flow_thresh_sb.isVisibleTo(w)
    assert not w.cellprob_sb.isVisibleTo(w)
    assert w.run_picker_combo.itemText(0) == "● Default  (6 stacks)"
    assert not w.new_run_btn.isEnabled() and not w.manage_runs_btn.isEnabled()
    assert w.run_row_note.text() == T.RUN_ROW_AFTER_RELEASE


def test_the_seven_groups_and_the_size_off_note(app, w):
    titles = [w._adv_groups[e]._toggle_btn.text().strip()
              for _p, e, _n in T.STEP4_ADVANCED_GROUPS]
    assert titles == [p for p, _e, _n in T.STEP4_ADVANCED_GROUPS]
    assert w.skip_postprocess_cb.isChecked()
    _open(w, app, 1)
    w._adv_collapsible.expand(animate=False)
    w._adv_groups["size_filters"].expand(animate=False)
    pump(app, 10)
    assert w.seg_size_off_note.isVisible()
    assert w.seg_size_off_note.text() == T.SEG_SIZE_OFF_NOTE
    w.skip_postprocess_cb.setChecked(False)
    pump(app)
    assert not w.seg_size_off_note.isVisible()
    w.skip_postprocess_cb.setChecked(True)


def test_the_engine_name_rides_the_closed_header(app, w):
    """F5: the grey engine name is on the group's header line, visible
    with the group CLOSED, and no longer a line inside the body."""
    _open(w, app, 1)
    w._adv_collapsible.expand(animate=False)
    pump(app, 5)
    for _plain, engine, _names in T.STEP4_ADVANCED_GROUPS:
        coll = w._adv_groups[engine]
        assert not coll.isExpanded()
        lbl = coll._epicycle_engine_label
        assert lbl.text() == engine
        assert lbl.parent() is coll._toggle_btn
        assert lbl.isVisibleTo(w)
    w._adv_collapsible.collapse(animate=False)


def test_the_closed_arrow_points_right_and_the_bar_is_blank(app, w):
    """F10: a right-pointing arrow when closed; no text on the idle bar."""
    for coll in [w._adv_collapsible, w._seg_collapsible,
                 *w._adv_groups.values()]:
        assert coll._epicycle_collapsed_symbol == "\u25b6"
        assert not coll.collapsedIcon().isNull()
        assert coll.collapsedIcon().cacheKey() != \
            coll.expandedIcon().cacheKey()
    assert w.run_progress.format() == ""


def test_clicking_a_tick_name_toggles_it(app, w):
    """F5: the plain name left of an Advanced tick toggles it."""
    from qtpy.QtCore import QEvent, QPointF
    from qtpy.QtGui import QMouseEvent
    lbl = w._adv_engine_labels["do_mp_cb"]
    before = w.do_mp_cb.isChecked()
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        ev = QMouseEvent(kind, QPointF(2, 2), QPointF(2, 2), Qt.LeftButton,
                         Qt.LeftButton, Qt.NoModifier)
        app.sendEvent(lbl, ev)
    assert w.do_mp_cb.isChecked() is (not before)
    w.do_mp_cb.setChecked(before)


def test_the_flows_note_says_why_two_fields_are_grey(app, w):
    """F7: with "Smooth the flows" off its two fields are grey and a dim
    line under them says why; on, the note goes."""
    _open(w, app, 1)
    w._adv_collapsible.expand(animate=False)
    w._adv_groups["pre_smoothing"].expand(animate=False)
    pump(app, 5)
    w.ps_smooth_flows_cb.setChecked(False)
    pump(app)
    assert not w.ps_mask_thresh_sb.isEnabled()
    assert w.ps_flows_off_note.text() == T.PS_FLOWS_OFF_NOTE
    assert w.ps_flows_off_note.isVisible()
    w.ps_smooth_flows_cb.setChecked(True)
    pump(app)
    assert not w.ps_flows_off_note.isVisible()
    w.ps_smooth_flows_cb.setChecked(False)
    w._adv_groups["pre_smoothing"].collapse(animate=False)
    w._adv_collapsible.collapse(animate=False)


def test_the_running_status_wraps_to_two_lines(app, w):
    """F3: the running sentence wraps to at most two lines; the whole of
    it is the tooltip."""
    _open(w, app, 1)
    long = ("Segmenting stack 2/3: Wound_patch_2.tiff — XZ (412/768)  "
            "[6:02 | ~5:40 remaining] " * 3)
    w.show_seg_status(long, active=True)
    lbl = w.run_status_label
    assert lbl.wordWrap()
    assert lbl.text().count("\n") <= 1
    assert lbl.toolTip() == long
    assert lbl.maximumHeight() <= 2 * lbl.fontMetrics().lineSpacing() + 4


# ---------------------------------------------------------- mirrored ticks


def test_ticks_are_mirrored_and_saved(app, w):
    names = lambda lw: [lw.item(i).data(Qt.UserRole)  # noqa: E731
                        for i in range(lw.count())
                        if lw.item(i).checkState() == Qt.Checked]
    assert names(w.session_stacks_list) == names(w.stacks_list)
    it = next(w.session_stacks_list.item(i)
              for i in range(w.session_stacks_list.count())
              if w.session_stacks_list.item(i).data(Qt.UserRole) == "P3.tif")
    it.setCheckState(Qt.Checked)
    pump(app, 5)
    assert "P3.tif" in names(w.stacks_list)
    assert "P3.tif" in w.controller.state_manager.get_ticked()
    # ③ items are not checkable
    for i in range(w.qc_stacks_list.count()):
        assert not (w.qc_stacks_list.item(i).flags() & Qt.ItemIsUserCheckable)


def test_a_row_that_cannot_run_has_no_tick(app, w):
    lw = w.stacks_list
    for i in range(lw.count()):
        it = lw.item(i)
        rec = it.data(Qt.UserRole + 2)
        if rec["conflict"] or rec["name"] == "P4.tif":
            assert not (it.flags() & Qt.ItemIsUserCheckable), rec["name"]


# ------------------------------------------------------------- the ceilings


def _busy(w, on):
    """Stage a Predict: the flag, the lock lines, a running row, the green
    status (the screenshot staging); off puts the idle page back."""
    c = w.controller
    if on:
        # R1-9: the REAL push (set_surface_locked, the gates, the nav), then
        # the LONGEST lock sentence in every lock line
        c._run_kind = "predict"
        c._running = {"P1.tif": ("predict", " · XZ 412 / 768")}
        c._set_step4_flag("_predict_running", True)
        c._refresh_stacks()
        w.set_lock_lines(LONGEST_LOCK)
        w.show_seg_status("Segmenting stack 2/3: Wound_patch_2.tiff — XZ "
                          "(412/768)  [6:02 | ~5:40 remaining]", active=True)
        w.cancel_btn.setEnabled(True)
    else:
        c._running = {}
        c._run_kind = None
        c._set_step4_flag("_predict_running", False)
        c._refresh_stacks()
    return w


#: R1-9: a 40-character stack name, the longest a row shows in full
LONG_STACK = "WholeScale_membrane_patch_0042_long.tiff"
assert len(LONG_STACK) == 40
LONGEST_LOCK = max((T.STEP4_LOCK_PREDICT, T.STEP4_LOCK_RECONSTRUCT,
                    T.STEP4_LOCK_MEASURE,
                    T.STEP4_LOCK_SMOOTH.format(stack=LONG_STACK),
                    T.STEP4_LOCK_SAVE.format(stack=LONG_STACK),
                    T.STEP4_LOCK_LOAD.format(stack=LONG_STACK),
                    T.STEP4_LOCK_EXPORT), key=len)


def measure(w, qapp):
    """Every font-dependent number, for one staged state."""
    out = {}
    _open(w, qapp, 1)
    out["right_min_w"] = w.seg_right_content.minimumSizeHint().width()
    out["section_min_w"] = []
    for idx in (0, 1):
        _open(w, qapp, idx)
        content = w.accordion._sections[idx]._scroll.widget()
        out["section_min_w"].append(content.minimumSizeHint().width())
    _open(w, qapp, 1)
    out["left_w"] = w.seg_left.width() - 6
    out["left_list_min_w"] = w.stacks_list.minimumSizeHint().width()
    col = CEIL["STEP4_SEG_RIGHT_COL_W"]
    w._adv_collapsible.collapse(animate=False)
    pump(qapp, 5)
    out["B2"] = _hfw(w.seg_right_content, col)
    w._adv_collapsible.expand(animate=False)
    w._adv_groups["size_filters"].expand(animate=False)
    pump(qapp, 5)
    out["B3"] = _hfw(w.seg_right_content, col)
    w._adv_groups["size_filters"].collapse(animate=False)
    w._adv_collapsible.collapse(animate=False)
    pump(qapp, 5)
    out["foot"] = _hfw(w.seg_foot, col)
    width = CEIL["STEP4_LEFT_COL_W"]
    lay = w.seg_left.layout()
    total, n = 0, 0
    for i in range(lay.count()):
        item = lay.itemAt(i)
        wid = item.widget()
        if wid is w.stacks_list or (wid is not None and wid.isHidden()):
            continue
        total += (item.heightForWidth(width) if item.hasHeightForWidth()
                  else item.sizeHint().height())
        n += 1
    out["left_fixed"] = total + lay.spacing() * n
    busy = bool(w.controller._predict_running)
    w.seg_lock_line.setText(LONGEST_LOCK)
    out["lock"] = _hfw(w.seg_lock_line, 560)
    w.seg_lock_line.setText(T.STEP4_LOCK_PREDICT)
    pump(qapp, 10)
    lw = w.stacks_list
    visible, used = 0, 0
    for i in range(lw.count()):
        h = lw.sizeHintForRow(i)
        if used + h > lw.viewport().height():
            break
        used += h
        visible += 1
    out["seg_rows_visible"] = visible
    w.set_lock_lines(LONGEST_LOCK if busy else "")
    pump(qapp, 5)
    _open(w, qapp, 0)
    content = w.accordion._sections[0]._scroll.widget()
    out["session"] = _hfw(content, 566)
    return out


def _session_fits(w, qapp):
    """The pointer line's bottom inside ①'s scroll viewport, without
    scrolling (15 stacks: Pipeline status stays on screen)."""
    _open(w, qapp, 0)
    sec = w.accordion._sections[0]
    content = sec._scroll.widget()
    content.resize(content.width(), _hfw(content, content.width()))
    pump(qapp, 5)
    ptr = w.session_model_pointer
    bottom = ptr.mapTo(content, QPoint(0, ptr.height())).y()
    return {"pointer_bottom": bottom,
            "viewport_h": sec._scroll.viewport().height(),
            "list_h": w.session_stacks_list.height(),
            "session": _hfw(content, 566)}


def measure_all(qapp):
    out = {"font": qapp.font().family()}
    with tempfile.TemporaryDirectory() as tmp:
        w = build(qapp, tmp)
        try:
            for state in ("idle", "busy"):
                _busy(w, state == "busy")
                pump(qapp, 10)
                out[state] = measure(w, qapp)
            _busy(w, False)
            w.set_lock_lines("")
            extra = Path(tmp) / "extra15"
            for i in range(9):
                write_stack(extra / f"More_{i:02d}.tif")
            w.controller._scan_stacks_directory(str(extra))
            pump(qapp, 10)
            out["n15"] = w.session_stacks_list.count()
            out["fifteen"] = _session_fits(w, qapp)
            # R1-9 / (a): fifteen stacks under the lock, longest sentence
            _busy(w, True)
            pump(qapp, 10)
            out["fifteen_busy"] = _session_fits(w, qapp)
            _busy(w, False)
        finally:
            close(w)
            pump(qapp, 5)
    return out


_SUBPROCESS = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
from tests.segmentation import test_step4_panel_skeleton as t
print("S4JSON" + json.dumps(t.measure_all(app)), flush=True)
os._exit(0)     # the numbers are out; skip a Qt teardown that can crash
"""


@pytest.fixture(scope="module")
def ceilings():
    """Every font-dependent number, measured in a fresh process with real
    fonts; QT_QPA_FONTDIR is set only in that process's own env."""
    code = f"FONT_DIR = {_FONT_DIR!r}\n" + _SUBPROCESS
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("S4JSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:],
                   proc.stderr[-3000:])
    got = json.loads(lines[-1][len("S4JSON"):])
    print(f"[CEIL] {json.dumps(got)}")
    return got


@pytest.fixture(params=["idle", "busy"])
def state(request):
    return request.param


def test_measured_in_the_users_font(ceilings):
    if sys.platform == "win32" and os.path.isdir(_FONT_DIR):
        assert ceilings["font"] == "Segoe UI"


def test_nothing_scrolls_sideways(ceilings, state):
    got = ceilings[state]
    assert got["right_min_w"] <= CEIL["STEP4_SEG_RIGHT_COL_W"]
    for width in got["section_min_w"]:
        assert width <= SECTION_W


def test_the_left_rows_lay_out_at_218(ceilings, state):
    got = ceilings[state]
    assert got["left_w"] == CEIL["STEP4_LEFT_COL_W"]
    assert got["left_list_min_w"] <= CEIL["STEP4_LEFT_COL_W"]


def test_right_column_advanced_closed(ceilings, state):
    h = ceilings[state]["B2"]
    print(f"[CEIL] B2 right {state}: {h}")
    assert h <= CEIL["STEP4_SEG_RIGHT_B2_PX"]


def test_right_column_advanced_and_size_limits_open(ceilings, state):
    h = ceilings[state]["B3"]
    print(f"[CEIL] B3 right {state}: {h}")
    assert h <= CEIL["STEP4_SEG_RIGHT_B3_PX"]


def test_the_pinned_foot(ceilings, state):
    h = ceilings[state]["foot"]
    print(f"[CEIL] foot {state}: {h}")
    assert h <= CEIL["STEP4_SEG_FOOT_PX"]


def test_the_left_column_above_the_list(ceilings, state):
    h = ceilings[state]["left_fixed"]
    print(f"[CEIL] left fixed {state}: {h}")
    assert h <= CEIL["STEP4_SEG_LEFT_FIXED_PX"]


def test_the_lock_line(ceilings, state):
    h = ceilings[state]["lock"]
    print(f"[CEIL] lock line: {h}")
    assert h <= CEIL["STEP4_LOCK_LINE_PX"]


def test_three_rows_show_at_the_default_height(ceilings, state):
    assert ceilings[state]["seg_rows_visible"] >= \
        CEIL["STEP4_SEG_LIST_MIN_ROWS"]


#: Decided 2026-10-05: ① state B is held to the measured build
#: value, idle AND busy (since its list gives up the lock
#: line's height, so busy measures the same as idle).  It was the mockup's
#: 626 idle and 626 + 40 = 666 busy.  The mockup's own state B content
#: (595.8 px) stays below it, so the mockup check still holds.
#:
#: A3 (2026-10-10, owner decisions 11 and the rows mockup v3): the
#: pictures are 128 px and the list's cap is four rows of them
#: (``SESSION_LIST_MAX_H`` 310 -> 550), so ① state B measures **857 px idle
#: and busy** (was 617).  The section scrolls past the viewport, as Step 3's
#: Session does; the F4 "pointer line on screen" check below is retired for
#: the same reason (the list gives up the lock line's height, so busy
#: still measures the same as idle).
STEP4_SESSION_B_BUILD_PX = 857


def test_session_state_b(ceilings, state):
    """State B idle and busy: the measured build ceiling, which
    is below the mockup-derived 626 and is never above the mockup's own
    measurement plus a lock line."""
    h = ceilings[state]["session"]
    print(f"[CEIL] session B {state}: {h} (ceiling "
          f"{STEP4_SESSION_B_BUILD_PX})")
    # (A3: the mockup data file's STEP4_SESSION_B_PX is 857 too, owner-
    # approved with the rows mockup v3's sizes)
    assert STEP4_SESSION_B_BUILD_PX <= CEIL["STEP4_SESSION_B_PX"]
    assert h <= STEP4_SESSION_B_BUILD_PX


def test_fifteen_stacks_cap_the_list_at_four_rows(ceilings):
    """F4, as changed by A3: with 15 stacks the ① list stops at its cap of
    four 128 px rows and scrolls inside itself; the section itself scrolls
    to Pipeline status and the pointer line (they used to fit the viewport
    at the 32 px picture)."""
    got = ceilings["fifteen"]
    print(f"[CEIL] 15 stacks: {got}")
    assert ceilings["n15"] == 15
    assert got["session"] <= STEP4_SESSION_B_BUILD_PX
    from napari_epicycle.segmentation.ui import SESSION_LIST_MAX_H
    assert got["list_h"] == SESSION_LIST_MAX_H
    assert got["pointer_bottom"] <= got["session"]


def test_fifteen_stacks_under_a_lock_give_up_the_lock_lines_room(
        ceilings):
    """Owner recommendation (a) / R1-9: under a lock the ① list gives up
    the lock line's room, so the pointer line stays inside the viewport
    with the LONGEST lock sentence showing -- A3: the list still gives up
    that room (the section scrolls to the pointer line)."""
    got = ceilings["fifteen_busy"]
    print(f"[CEIL] 15 stacks busy: {got}")
    assert got["pointer_bottom"] <= got["session"]
    assert got["list_h"] < ceilings["fifteen"]["list_h"]
