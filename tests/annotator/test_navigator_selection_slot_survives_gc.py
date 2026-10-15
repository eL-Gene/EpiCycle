"""The navigator's selection-model slot must survive a garbage collection.

Root cause (traced 2026-09-27).  ``_connect_signals`` connected a LAMBDA to
``self._table.selectionModel().selectionChanged``.  Only PyQt's slot proxy
held that lambda.  When the selection model's Python wrapper is caught in a
garbage cycle (a dead frame of ``_rebuild_table`` holding its
``sel_model`` local, kept alive by numba's first-call compile inside
``DirectLabelColormap.map``), PyQt's GC traversal of the wrapper "explains
away" the lambda's only reference, so ``gc.collect()`` -- which napari runs
on every layer removal -- CLEARS the lambda's closure while PyQt still
calls it.  The next ``selectionChanged`` faults in ``<lambda>`` on its first
instruction (every faulthandler dump since 2026-06-09: ``line ??? in
<lambda>``).

The fix connects a bound method (no closure; its function is reachable from
the class).  Tests (1) and (3) run in a SUBPROCESS so the old code's access
violation fails the test instead of killing pytest.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

_PRELUDE = textwrap.dedent(
    """
    import faulthandler, gc, os, sys
    faulthandler.enable()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator,
    )
    nav = CurationLabelNavigator()
    calls = [0]
    def _counter():
        calls[0] += 1
    nav._on_table_selection_changed = _counter
    nav.set_items([
        {"label_id": i, "tier": "good", "centroid": (i, i, i), "area": 10 * i}
        for i in range(1, 6)
    ])
    for _ in range(5):
        app.processEvents()
    assert nav._table.rowCount() == 5, nav._table.rowCount()
    """
)

_CYCLE = textwrap.dedent(
    """
    class H:
        pass
    h = H()
    h.h = h
    h.sm = nav._table.selectionModel()
    del h
    gc.collect()
    nav._table.selectRow(1)
    app.processEvents()
    print("CALLS", calls[0], flush=True)
    """
)

_REAL_TRIGGER = textwrap.dedent(
    """
    import numpy as np
    from napari.utils.colormaps import DirectLabelColormap

    def _rebuild_like(cmap):
        # A first-time DirectLabelColormap.map (numba compile) under a frame
        # that holds the selection model -- what _rebuild_table does via
        # _batch_swatch_colors.
        arr = cmap.map(np.array([1, 2, 5, 300], dtype=np.intp))
        sel_model = nav._table.selectionModel()
        return arr

    cmap = DirectLabelColormap(
        color_dict={1: "red", 2: "blue", 5: "green", None: "black"}
    )
    _rebuild_like(cmap)
    gc.collect()
    nav._table.selectRow(2)
    app.processEvents()
    print("CALLS", calls[0], flush=True)
    """
)


def _run(body: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONFAULTHANDLER"] = "1"
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + body],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _calls(proc: subprocess.CompletedProcess) -> int:
    for line in proc.stdout.splitlines():
        if line.startswith("CALLS "):
            return int(line.split()[1])
    return -1


def _explain(proc: subprocess.CompletedProcess) -> str:
    return (
        f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout[-3000:]}"
        f"\n--- stderr ---\n{proc.stderr[-3000:]}"
    )


def test_slot_survives_a_garbage_cycle_holding_the_selection_model():
    proc = _run(_CYCLE)
    assert proc.returncode == 0, _explain(proc)
    assert _calls(proc) >= 1, _explain(proc)


def test_slot_survives_the_real_trigger_colormap_compile_then_collect():
    pytest.importorskip("napari")
    proc = _run(_REAL_TRIGGER)
    assert proc.returncode == 0, _explain(proc)
    assert _calls(proc) >= 1, _explain(proc)


def test_the_selection_slot_is_a_bound_method_not_a_closure(qapp):
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator,
    )

    assert "_on_sel_model_changed" in CurationLabelNavigator.__dict__
    nav = CurationLabelNavigator()
    try:
        slot = nav._on_sel_model_changed
        assert slot.__self__ is nav
        assert getattr(slot.__func__, "__closure__", None) is None
        # And the wiring really goes to it: a selection change reaches the
        # handler through the bound method.
        seen = []
        nav._on_table_selection_changed = lambda: seen.append(1)
        nav.set_items([
            {"label_id": i, "tier": "good", "centroid": (i, i, i), "area": i}
            for i in range(1, 4)
        ])
        qapp.processEvents()
        nav._table.selectRow(1)
        assert seen, "selectionChanged did not reach the handler"
    finally:
        nav.setParent(None)
        nav.deleteLater()


def test_wire_signals_connects_the_bound_method_and_no_lambda():
    """U3 (fixes-audit LOW): by AST, ``_wire_signals`` (the method the
    brief called ``_connect_signals``) connects ``selectionChanged`` to
    ``self._on_sel_model_changed`` and no ``selectionChanged.connect`` in it takes a lambda -- a stale-closure
    slot cannot come back unnoticed."""
    import ast
    import inspect
    import textwrap

    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator,
    )

    src = textwrap.dedent(inspect.getsource(
        CurationLabelNavigator._wire_signals))
    tree = ast.parse(src)
    targets = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"):
            continue
        sig = node.func.value
        if not (isinstance(sig, ast.Attribute)
                and sig.attr == "selectionChanged"):
            continue
        assert node.args, "selectionChanged.connect() with no slot"
        slot = node.args[0]
        assert not isinstance(slot, ast.Lambda), (
            "a lambda on selectionChanged -- the GC crash class")
        targets.append(ast.unparse(slot))
    assert "self._on_sel_model_changed" in targets, targets
