"""The annotator's close-time orphan sweep must only take its own windows.

Every pyqtgraph ``PlotWidget`` owns hidden top-level windows (menu popups,
combo popups, and a parentless nameless ``QWidget``, the default widget of a
menu action).  If a sweep deletes the latter, destroying that plot later
aborts the whole process ("shared QObject was deleted directly").  The sweep
used to treat every top-level whose Python ``id()`` was not in a snapshot as
its own.  ``topLevelWidgets()`` returns short-lived wrappers for widgets Qt
made itself, so their ``id()`` changes between calls, and windows other
panels made after the snapshot were taken too.

The rule now: the sweep may take the windows that appeared while the
annotator built itself, plus those that appeared while its close ran.  That
set is fixed when the close runs its sweep; the deferred sweep a tick later
takes nothing that appeared after the close returned.  Windows are matched by
C++ address, and an own window's address must still hold the same wrapper.

The scenario runs in a SUBPROCESS (offscreen) so an abort fails the test
instead of killing pytest.  No annotator is built: the production sweep runs
on a stub that carries the state the annotator's constructor and close leave
behind.
"""
from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pyqtgraph")

_REPO = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import faulthandler, gc, os, sys, types
    faulthandler.enable()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from qtpy import sip
    from qtpy.QtCore import QCoreApplication, QEvent, QObject
    from qtpy.QtWidgets import QApplication, QFrame, QVBoxLayout, QWidget
    app = QApplication.instance() or QApplication([])
    import pyqtgraph as pg
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator as NMA,
    )

    class Stub:
        # Class-level names (the orphan classes, the snapshot helper) come
        # from the annotator itself; instance state is set by the test.
        def __getattr__(self, name):
            return getattr(NMA, name)

    def addr(w):
        return sip.unwrapinstance(w)

    def flush():
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()

    # The app's main window: the foreign panels live inside it, as plots
    # live in docks of napari's window.
    root = QWidget()
    QVBoxLayout(root)
    root.show()

    def plot_host():
        # Every orphan-class top-level the plot's panel brings, by ADDRESS
        # (no wrapper is kept, so the sweep sees fresh wrappers).
        before = {addr(w) for w in app.topLevelWidgets()}
        host = QWidget(root)
        root.layout().addWidget(host)
        QVBoxLayout(host).addWidget(pg.PlotWidget())
        host.show()
        app.processEvents()
        made = [
            addr(w) for w in app.topLevelWidgets()
            if addr(w) not in before and type(w).__name__ in NMA._ORPHAN_CLASSES
        ]
        hidden_qwidget = [
            addr(w) for w in app.topLevelWidgets()
            if addr(w) not in before and w is not host
            and type(w).__name__ == "QWidget" and w.parentWidget() is None
        ]
        assert hidden_qwidget, "the plot made no hidden parentless QWidget"
        return host, made, hidden_qwidget[0]

    def missing(addrs):
        alive = {addr(w) for w in app.topLevelWidgets()}
        return sum(1 for a in addrs if a not in alive)

    def churn():
        junk = [QObject() for _ in range(5000)]
        del junk
        gc.collect()
        for _ in range(3):
            app.topLevelWidgets()

    # -- teeth: a stub with no own set sweeps nothing ---------------------
    bare = Stub()
    bare.viewer = types.SimpleNamespace(window=None)
    bare._creation_toplevel_ids = set(NMA._toplevel_widgets_by_address())
    bare._close_start_toplevel_ids = set(NMA._toplevel_widgets_by_address())
    stray = QFrame()
    NMA._destroy_orphan_toplevel_widgets(bare)
    flush()
    print("TEETH_STRAY_DELETED", sip.isdeleted(stray), flush=True)
    stray.deleteLater()
    flush()

    # -- before the annotator: a foreign panel -----------------------------
    host1, made1, hidden1 = plot_host()
    print("FOREIGN_COUNT", len(made1), flush=True)
    print("FOREIGN_NOT_QWIDGET", sum(
        1 for w in app.topLevelWidgets()
        if addr(w) in made1 and type(w).__name__ != "QWidget"), flush=True)

    # -- the constructor's bookkeeping -------------------------------------
    stub = Stub()
    stub.viewer = types.SimpleNamespace(window=None)
    pre = NMA._toplevel_widgets_by_address()
    stub._creation_toplevel_ids = set(pre)
    own_frame = QFrame()                  # a window the annotator itself made
    after = NMA._toplevel_widgets_by_address()
    stub._own_toplevel_widgets = {
        a: w for a, w in after.items() if a not in stub._creation_toplevel_ids
    }
    assert addr(own_frame) in stub._own_toplevel_widgets
    # Wrapper identity: an own entry whose wrapper is not the window now at
    # that address must not make the sweep take that window.
    stub._own_toplevel_widgets[hidden1] = object()
    del pre, after

    churn()
    host2, made2, _ = plot_host()         # a foreign panel made later

    # -- the close's bookkeeping, as _close_subviewers does it -------------
    stub._close_start_toplevel_ids = set(NMA._toplevel_widgets_by_address())
    late_frame = QFrame()                 # appears during the close
    NMA._freeze_close_sweep_set(stub)
    NMA._destroy_orphan_toplevel_widgets(stub)      # the synchronous sweep
    flush()

    # -- after the close returned, before its deferred sweep ---------------
    host3, made3, _ = plot_host()
    churn()
    NMA._destroy_orphan_toplevel_widgets(stub)      # the deferred sweep
    flush()

    print("FOREIGN_DELETED", missing(made1 + made2), flush=True)
    print("IDENTITY_PIN_DELETED", missing([hidden1]), flush=True)
    print("GAP_DELETED", missing(made3), flush=True)
    print("OWN_DELETED", sip.isdeleted(own_frame), flush=True)
    print("LATE_DELETED", sip.isdeleted(late_frame), flush=True)

    for h in (host1, host2, host3):
        h.deleteLater()
    flush()
    root.deleteLater()
    flush()
    gc.collect()
    flush()
    print("HOSTS_DELETED True", flush=True)
    """
)


@pytest.fixture(scope="module")
def run() -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONFAULTHANDLER"] = "1"
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _value(proc: subprocess.CompletedProcess, key: str):
    for line in proc.stdout.splitlines():
        if line.startswith(key + " "):
            return line.split(" ", 1)[1].strip()
    return None


def _explain(proc: subprocess.CompletedProcess) -> str:
    return (
        f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout[-3000:]}"
        f"\n--- stderr ---\n{proc.stderr[-3000:]}"
    )


def test_a_stub_without_an_own_set_sweeps_nothing(run):
    assert _value(run, "TEETH_STRAY_DELETED") == "False", _explain(run)


def test_a_plot_brings_many_orphan_class_windows(run):
    # Not just the hidden QWidget: combo popups (QFrame) and others, so a
    # sweep that only stopped taking "QWidget" would still fail below.
    count = _value(run, "FOREIGN_COUNT")
    assert count is not None and int(count) >= 2, _explain(run)
    not_qwidget = _value(run, "FOREIGN_NOT_QWIDGET")
    assert not_qwidget is not None and int(not_qwidget) >= 1, _explain(run)


def test_foreign_plot_windows_survive_the_sweep(run):
    assert _value(run, "FOREIGN_DELETED") == "0", _explain(run)


def test_an_own_entry_with_another_wrapper_does_not_take_the_window(run):
    assert _value(run, "IDENTITY_PIN_DELETED") == "0", _explain(run)


def test_a_plot_made_after_the_close_survives_the_deferred_sweep(run):
    assert _value(run, "GAP_DELETED") == "0", _explain(run)


def test_the_annotators_own_window_is_still_swept(run):
    assert _value(run, "OWN_DELETED") == "True", _explain(run)


def test_a_window_that_appears_during_the_close_is_still_swept(run):
    assert _value(run, "LATE_DELETED") == "True", _explain(run)


def test_the_hosts_delete_and_the_process_exits_cleanly(run):
    assert _value(run, "HOSTS_DELETED") == "True", _explain(run)
    assert "deleted directly" not in run.stderr, _explain(run)
    assert run.returncode == 0, _explain(run)


# -- source order: when the snapshots are taken ------------------------------


def _function_ast(fn) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return tree.body[0]


def _assign_lines(fn_ast, attr: str, value_types=None) -> list:
    lines = []
    for node in ast.walk(fn_ast):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            if (isinstance(t, ast.Attribute) and t.attr == attr
                    and isinstance(t.value, ast.Name) and t.value.id == "self"):
                if value_types is None or isinstance(node.value, value_types):
                    lines.append(node.lineno)
    return sorted(lines)


def _call_lines(fn_ast, predicate) -> list:
    return sorted(
        node.lineno for node in ast.walk(fn_ast)
        if isinstance(node, ast.Call) and predicate(node)
    )


def _is_self_method(name):
    def pred(call):
        f = call.func
        return (isinstance(f, ast.Attribute) and f.attr == name
                and isinstance(f.value, ast.Name) and f.value.id == "self")
    return pred


def test_init_builds_the_own_set_after_the_qtviewers():
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    fn = _function_ast(NativeMultiviewAnnotator.__init__)
    qtviewers = _call_lines(
        fn, lambda c: isinstance(c.func, ast.Name) and c.func.id == "QtViewer")
    own_built = _assign_lines(
        fn, "_own_toplevel_widgets", (ast.Dict, ast.DictComp))
    assert qtviewers and own_built, (qtviewers, own_built)
    assert own_built[0] > qtviewers[-1], (qtviewers, own_built)


def test_close_takes_its_snapshot_first_and_freezes_before_the_deferred_sweep():
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    fn = _function_ast(NativeMultiviewAnnotator._close_subviewers)
    start = _assign_lines(fn, "_close_start_toplevel_ids")
    qtv_close = _call_lines(
        fn, lambda c: isinstance(c.func, ast.Attribute) and c.func.attr == "close"
        and isinstance(c.func.value, ast.Name) and c.func.value.id == "qtv")
    sweep = _call_lines(fn, _is_self_method("_destroy_orphan_toplevel_widgets"))
    freeze = _call_lines(fn, _is_self_method("_freeze_close_sweep_set"))
    deferred = _call_lines(
        fn, lambda c: isinstance(c.func, ast.Attribute)
        and c.func.attr == "singleShot" and len(c.args) >= 2
        and isinstance(c.args[1], ast.Attribute)
        and c.args[1].attr == "_destroy_orphan_toplevel_widgets")
    assert start and qtv_close and sweep and freeze and deferred, (
        start, qtv_close, sweep, freeze, deferred)
    assert start[0] < qtv_close[0], (start, qtv_close)
    assert start[0] < sweep[0], (start, sweep)
    assert freeze[0] < sweep[0], (freeze, sweep)
    assert freeze[0] < deferred[0], (freeze, deferred)
