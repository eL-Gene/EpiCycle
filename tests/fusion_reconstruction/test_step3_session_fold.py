"""A2 fold (audit 2): the counts worker and the Session panel.

* the Windows file-lock race -- a CSV held open by a reader no longer makes
  the extraction's auto-save (or Save / Commit Triage) fail: the replace
  retries briefly; the extraction counts only AFTER its auto-save;
* two one-patch triggers in a row both land;
* Session's own status line carries the eye's / the inline button's
  refusal;
* the eye only on rows that have a reconstruction to view;
* the Resume link is elided, never widens the panel;
* the tier line's "…" is always drawn when a tier is dropped;
* a run switch / Browse clear the old run's counts first;
* the dim folder sentence hidden with no workspace, elided when long;
* a result already emitted before a stop never lands.
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
import threading
import time

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _step3_tier_counts as TC  # noqa: E402
from napari_epicycle.fusion_reconstruction.controller import (  # noqa: E402
    FusionReconstructionController as C,
)

ITEMS = (("good", 3, (80, 200, 120)), ("review", 1, (255, 200, 60)),
         ("bad", 2, (255, 120, 40)), ("garbage", 2, (240, 60, 60)),
         ("curated", 5, (0, 188, 212)))


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _pump(app, until, deadline=20.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(until())


def _P():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    return P


# -- 9. the file-lock race --------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows file locks")
def test_a_csv_held_open_by_a_reader_is_still_replaced(tmp_path):
    """A reader (the count worker, a notebook) holding the CSV open makes
    ``os.replace`` raise PermissionError on Windows; the save retries."""
    from napari_epicycle.fusion_reconstruction.qc_features import (
        save_patch_csv,
    )
    p = tmp_path / "a_features.csv"
    pd.DataFrame({"label": [1], "qc_tier": ["bad"]}).to_csv(p, index=False)
    fh = open(p, "rb")
    threading.Timer(0.15, fh.close).start()
    save_patch_csv(pd.DataFrame({"label": [1], "qc_tier": ["good"]}), p)
    assert pd.read_csv(p)["qc_tier"].tolist() == ["good"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file locks")
def test_a_csv_held_open_for_ever_still_raises(tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction import qc_features as Q
    monkeypatch.setattr(Q, "REPLACE_RETRY_DELAY_S", 0.001)
    p = tmp_path / "a.csv"
    p.write_text("label\n1\n")
    with open(p, "rb"):
        with pytest.raises(PermissionError):
            Q.save_patch_csv(pd.DataFrame({"label": [2]}), p)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".csv")
                and f != "a.csv"]                    # no temp file left


def test_the_commit_triage_promote_retries():
    from napari_epicycle.fusion_reconstruction import controller as M
    src = inspect.getsource(M._commit_triage_persist)
    assert "replace_with_retry(" in src


def _trigger_lines(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_start_tier_counts"]


def test_the_extraction_counts_after_its_auto_save_only():
    """The top-of-handler count runs only when the auto-save will NOT
    (no table in memory, another patch, another run); the post-save count
    sits OUTSIDE the auto-save's try, so a count failure is never logged
    as "Auto-save FAILED"."""
    src = textwrap.dedent(inspect.getsource(C._on_qc_extract_finished))
    tree = ast.parse(src)
    calls = _trigger_lines(C._on_qc_extract_finished)
    assert len(calls) == 2
    # no count call inside a try whose handler logs the auto-save failure
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.handlers:
            handlers = ast.get_source_segment(src, node.handlers[0]) or ""
            if "Auto-save to disk FAILED" in handlers:
                body = ast.Module(body=node.body, type_ignores=[])
                assert not any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_start_tier_counts"
                    for n in ast.walk(body))
    assert "_extract_autosave_will_run" in src


def test_the_auto_save_predicate(tmp_path):
    c = C.__new__(C)
    c._qc_view_patch_name = "A.tif"
    c._run_epoch = 2
    c._qc_extract_launch_epoch = 2
    c._qc_features_df = None
    assert c._extract_autosave_will_run("A.tif") is False
    c._qc_features_df = pd.DataFrame({"label": [1]})
    assert c._extract_autosave_will_run("A.tif") is True
    assert c._extract_autosave_will_run("B.tif") is False
    c._qc_extract_launch_epoch = 1
    assert c._extract_autosave_will_run("A.tif") is False


# -- 10. two one-patch triggers ------------------------------------------------------------

class _Sink:
    def __init__(self):
        self.results = []

    def counted(self, stamp, result):
        self.results.append((stamp, result))


def _csv(path, tiers):
    pd.DataFrame({"label": range(1, len(tiers) + 1),
                  "qc_tier": tiers}).to_csv(path, index=False)
    return path


class _UI:
    def __init__(self):
        self.pushed = []

    def set_tier_counts(self, mapping, *, replace=False):
        self.pushed.append((dict(mapping), replace))


def test_two_one_patch_triggers_in_a_row_both_land(app, tmp_path,
                                                   monkeypatch):
    c = C.__new__(C)
    c.working_dir = str(tmp_path)
    c._run_epoch = 0
    c.ui = _UI()
    c.state_manager = type("SM", (), {"get_active_run_id":
                                      staticmethod(lambda: "default")})()
    qc = c._tier_counts_csv("A.tif").parent
    qc.mkdir(parents=True, exist_ok=True)
    _csv(qc / "A_features.csv", ["good"])
    _csv(qc / "B_features.csv", ["bad", "bad"])
    gate = threading.Event()
    real = TC.count_batch

    def slow(requests, cancel=None):
        gate.wait(5)
        return real(requests, cancel)
    monkeypatch.setattr(TC, "count_batch", slow)
    c._start_tier_counts("commit triage", ["A.tif"])
    c._start_tier_counts("save curation", ["B.tif"])
    gate.set()
    assert _pump(app, lambda: {"A.tif", "B.tif"} <= set(
        getattr(c, "_tier_counts", {})))
    assert c._tier_counts["A.tif"]["good"] == 1
    assert c._tier_counts["B.tif"]["bad"] == 2
    c._stop_tier_counts()


def test_a_stale_result_resets_the_whole_list_flag(tmp_path):
    c = C.__new__(C)
    c.working_dir = str(tmp_path)
    c._run_epoch = 1
    c.ui = _UI()
    c._tier_counts_whole_running = True
    c._on_tier_counts_counted(("elsewhere", None, 0, True), {"a": None})
    assert c._tier_counts_whole_running is False
    assert c.ui.pushed == []


# -- 18. a result emitted before the stop never lands ------------------------------------

def test_a_result_emitted_before_the_stop_never_lands(app, tmp_path):
    a = _csv(tmp_path / "a.csv", ["good"])
    sink = _Sink()
    state = TC.start_tier_counts([("a.tif", str(a))], "s", sink.counted)
    thread = state.thread
    # the worker has emitted (queued to the main thread) but nothing pumped
    assert thread.wait(5000)
    TC.stop_tier_counts(state)
    for _ in range(30):
        app.processEvents()
    assert sink.results == []


# -- the panel ---------------------------------------------------------------------------

def _bare():
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    return w


@pytest.fixture
def ui(app):
    w = _bare()
    yield w
    w.deleteLater()
    app.processEvents()


@pytest.fixture
def full(app):
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration(viewer=None)
    c = w.controller
    statuses = {}
    c.state_manager.get_all_patch_statuses = lambda: dict(statuses)
    yield w, c, statuses
    w.deleteLater()
    app.processEvents()


def test_a_refused_eye_says_why_inside_session(full, monkeypatch, app):
    w, c, _s = full
    P = _P()
    w.show()
    w.switch_to_section(0)
    app.processEvents()
    c._save_in_flight = True
    try:
        c._refresh_step3_lock()
        c._on_session_eye("a.tif")
    finally:
        c._save_in_flight = False
    lab = w.session_status_label
    assert w._session_panel.isAncestorOf(lab)
    assert lab.toolTip() == dict(P.S3_LOCK_SENTENCES)["_save_in_flight"]
    assert lab.isVisible()
    # the next action that works clears it
    monkeypatch.setattr(c, "_view_patch", lambda name: None)
    c._refresh_step3_lock()
    c._on_session_eye("a.tif")
    assert lab.isHidden() and lab.toolTip() == ""
    w.hide()


def test_a_refused_inline_button_says_why_inside_session(full, monkeypatch):
    w, c, _s = full
    P = _P()
    monkeypatch.setattr(c, "_spacing_locked", lambda: True)
    c._on_session_row_action("a.tif", "set_spacing")
    assert w.session_status_label.toolTip() == P.S3_SET_SPACING_LOCKED
    assert not w.session_status_label.isHidden()


@pytest.mark.parametrize("status,eye", [
    ("unprocessed", False), ("predicted", False), ("reconstructed", True),
    ("curating", True), ("accepted", True), ("finalized", True)])
def test_the_eye_only_on_rows_with_a_reconstruction(app, status, eye):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont

    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    rec = R.build_session_row("p.tif", status, "CZYX")
    d = D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=True)
    g = d.layout(QRect(0, 0, 420, 0), rec, QFont())
    assert ("eye" in g) is eye
    assert "menu" in g                    # the ⋯ stays on every row
    # Step 4's rows never carry the key: their eye is unchanged
    s4 = d.layout(QRect(0, 0, 420, 0), {"name": "s", "word": "x"}, QFont())
    assert "eye" in s4


def test_the_resume_link_is_elided_and_never_widens_the_panel(ui):
    long = "WholeScale_patch_with_a_very_long_name_0123456789abcdef.tiff"
    before = ui._session_panel.minimumSizeHint().width()
    ui.refresh_session_activity("3 days ago", f"{long}  (curating)", True)
    assert ui._session_panel.minimumSizeHint().width() <= max(before, 566)
    assert ui.session_resume_btn.toolTip() == f"{long}  (curating)"


def test_the_tier_line_always_shows_the_ellipsis_when_a_tier_drops(app):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont

    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    d = D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=True)
    rec = R.build_session_row("p.tif", "curating", "CZYX",
                              tier_counts=ITEMS)
    for width in range(160, 700, 7):
        g = d.layout(QRect(0, 0, width, 0), rec, QFont())
        if g["tiers_elided"]:
            assert "tier_ellipsis" in g, width
            assert g["tier_ellipsis"].right() <= g["lines"]["tiers"].right()


def test_a_run_switch_and_a_browse_clear_the_old_counts():
    """Both reach ``_reset_workspace_scoped_caches`` (the run switch
    directly, Browse through ``_adopt_workspace_quiet``) before the rows
    are rebuilt; it clears the counts."""
    assert "_clear_tier_counts(" in inspect.getsource(
        C._reset_workspace_scoped_caches)
    for method in ("_switch_active_run", "_adopt_workspace_quiet"):
        src = inspect.getsource(getattr(C, method))
        assert "_reset_workspace_scoped_caches()" in src, method
    assert "_adopt_workspace_quiet(" in inspect.getsource(C._browse_workdir)


def test_clear_tier_counts_empties_both_maps(ui):
    c = C.__new__(C)
    c.ui = ui
    c._tier_counts = {"a.tif": {"good": 1}}
    ui.set_tier_counts({"a.tif": ITEMS})
    c._clear_tier_counts()
    assert c._tier_counts == {}
    assert ui._tier_count_items == {}


def test_the_pr_lists_records_never_carry_tier_counts(ui):
    """The Predict + Reconstruct list's narrow rows draw no tier line, so
    ``set_tier_counts`` names the two wide lists and never touches it."""
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    ui.refresh_patch_state_list({"a.tif": "curating"})
    ui.set_tier_counts({"a.tif": ITEMS}, replace=True)
    lst = ui._patch_state_list
    assert lst.count() == 1
    assert "tier_counts" not in (lst.item(0).data(ROLE_RECORD) or {})


def test_the_folder_sentence_hides_with_no_workspace_and_elides(ui):
    P = _P()
    ui.set_patch_dir_source(True, "", workspace=False)
    assert ui.patch_dir_qualifier.isHidden()
    long = "D:/data/" + "a_really_long_folder_name_" * 4
    ui.set_patch_dir_source(False, long, workspace=True)
    lab = ui.patch_dir_qualifier
    assert not lab.isHidden()
    full = P.S3_SESSION_USING_FOLDER.format(folder=os.path.basename(long))
    assert lab.text() == full and lab.toolTip() == long
    # elided when narrow: it never asks the row for its text's width
    assert lab.minimumSizeHint().width() == 0
