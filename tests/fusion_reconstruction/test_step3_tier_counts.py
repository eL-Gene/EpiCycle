"""A2: the tier counts on Step 3's rows -- the pure
counter, the record helper, the worker on a REAL ``QThread`` (a
fake thread that emits ``finished`` by itself hides a hang), and the
controller's apply with its stale guard and its triggers.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _step3_tier_counts as TC
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)

COLOURS = {"good": (80, 200, 120), "review": (255, 200, 60),
           "bad": (255, 120, 40), "garbage": (240, 60, 60),
           "curated": (0, 188, 212)}


def _csv(path, tiers, statuses=None, *, column="qc_tier"):
    data = {"label": list(range(1, len(tiers) + 1)), column: tiers,
            "volume": [1.0] * len(tiers)}
    if statuses is not None:
        data["label_status"] = statuses
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# -- the counter --------------------------------------------------------------

def test_counts_active_rows_by_tier(tmp_path):
    p = _csv(tmp_path / "a_features.csv",
             ["good", "good", "review", "bad", "garbage", "good", "bad"],
             ["active", "active", "active", "deleted", "active",
              "superseded", "active"])
    assert TC.tier_counts_from_csv(p) == {"good": 2, "review": 1, "bad": 1,
                                          "garbage": 1, "curated": 0}


def test_curated_is_the_first_tier_drawn(tmp_path):
    """A2 fold L2 (the owner, 2026-10-10): five tiers, curated FIRST (A3 fold)."""
    assert TC.TIERS == ("curated", "good", "review", "bad", "garbage")
    p = _csv(tmp_path / "c.csv", ["curated", "curated", "good"])
    assert TC.tier_counts_from_csv(p)["curated"] == 2
    items = TC.tier_items(TC.tier_counts_from_csv(p), COLOURS)
    assert items[0] == ("curated", 2, (0, 188, 212))


def test_a_csv_with_no_status_column_counts_every_row(tmp_path):
    p = _csv(tmp_path / "a.csv", ["good", "review", "review"])
    assert TC.tier_counts_from_csv(p) == {"good": 1, "review": 2, "bad": 0,
                                          "garbage": 0, "curated": 0}


def test_the_old_tier_column_name_is_read(tmp_path):
    p = _csv(tmp_path / "a.csv", ["bad", "bad"], column="tier")
    assert TC.tier_counts_from_csv(p)["bad"] == 2


def test_no_file_or_no_tier_column_is_none(tmp_path):
    assert TC.tier_counts_from_csv(tmp_path / "missing.csv") is None
    p = tmp_path / "b.csv"
    pd.DataFrame({"label": [1, 2]}).to_csv(p, index=False)
    assert TC.tier_counts_from_csv(p) is None
    bad = tmp_path / "c.csv"
    bad.write_bytes(b"\x00\x01not a csv\xff")
    assert TC.tier_counts_from_csv(bad) is None


def test_the_counter_reads_two_columns_only(tmp_path, monkeypatch):
    p = _csv(tmp_path / "a.csv", ["good"], ["active"])
    seen = {}
    real = pd.read_csv

    def spy(path, *a, **k):
        df = real(path, *a, **k)
        seen["cols"] = list(df.columns)
        return df
    monkeypatch.setattr(pd, "read_csv", spy)
    TC.tier_counts_from_csv(p)
    assert sorted(seen["cols"]) == ["label_status", "qc_tier"]


# -- the record items -----------------------------------------------------------

def test_items_in_tier_order_with_the_plots_colours():
    items = TC.tier_items({"review": 3, "good": 5, "bad": 0, "garbage": 1},
                          COLOURS)
    assert items == (("curated", 0, (0, 188, 212)),
                     ("good", 5, (80, 200, 120)), ("review", 3, (255, 200, 60)),
                     ("bad", 0, (255, 120, 40)), ("garbage", 1, (240, 60, 60)))


def test_no_counts_or_all_zero_is_none():
    assert TC.tier_items(None, COLOURS) is None
    assert TC.tier_items({t: 0 for t in TC.TIERS}, COLOURS) is None


def test_the_colours_are_the_plots_own():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _TIER_COLORS,
    )
    items = TC.tier_items({"good": 1}, _TIER_COLORS)
    assert dict((w, c) for w, _n, c in items)["good"] == tuple(
        _TIER_COLORS["good"])


# -- the worker on a real QThread ------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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


class _Sink:
    """Bound-method slots (never lambdas: a slot held only by PyQt can lose its
    closure when napari garbage-collects during a layer removal)."""

    def __init__(self):
        self.results = []
        self.finished = 0

    def counted(self, stamp, result):
        self.results.append((stamp, result))

    def done(self):
        self.finished += 1


def test_the_worker_counts_on_a_real_thread_and_the_thread_quits(
        qapp, tmp_path):
    a = _csv(tmp_path / "a.csv", ["good", "bad"], ["active", "active"])
    sink = _Sink()
    state = TC.start_tier_counts([("a.tif", str(a)),
                                  ("b.tif", str(tmp_path / "none.csv"))],
                                 ("ws", "default", 0), sink.counted,
                                 on_finished=sink.done)
    thread = state.thread
    assert _pump(qapp, lambda: sink.finished == 1 and not thread.isRunning())
    assert thread.quit_called
    assert sink.results == [(("ws", "default", 0),
                             {"a.tif": {"good": 1, "review": 0, "bad": 1,
                                        "garbage": 0, "curated": 0},
                              "b.tif": None})]
    assert _pump(qapp, lambda: thread not in TC._LIVE_COUNT_THREADS)


def test_a_new_batch_stops_the_older_one_whose_result_never_lands(
        qapp, tmp_path, monkeypatch):
    a = _csv(tmp_path / "a.csv", ["good"], ["active"])
    gate = __import__("threading").Event()
    real = TC.count_batch

    def slow(requests, cancel=None):
        gate.wait(5)
        return real(requests, cancel)
    monkeypatch.setattr(TC, "count_batch", slow)
    sink = _Sink()
    state = TC.start_tier_counts([("a.tif", str(a))], "old", sink.counted)
    old = state.thread
    monkeypatch.setattr(TC, "count_batch", real)
    TC.start_tier_counts([("a.tif", str(a))], "new", sink.counted,
                         state=state)
    gate.set()
    assert _pump(qapp, lambda: not old.isRunning()
                 and not state.thread.isRunning())
    qapp.processEvents()
    assert [s for s, _r in sink.results] == ["new"]


def test_stop_disconnects_and_is_idempotent(qapp, tmp_path):
    a = _csv(tmp_path / "a.csv", ["good"], ["active"])
    sink = _Sink()
    state = TC.start_tier_counts([("a.tif", str(a))], "s", sink.counted)
    assert TC.stop_tier_counts(state) is True
    assert state.thread is None and state.worker is None
    assert TC.stop_tier_counts(state) is True
    assert TC.stop_tier_counts(None) is True
    for _ in range(20):
        qapp.processEvents()
    assert sink.results == []


def test_no_requests_starts_nothing(qapp):
    state = TC.start_tier_counts([], "s", _Sink().counted)
    assert state.thread is None


# -- the controller: stamp, apply, triggers --------------------------------------

class _UI:
    def __init__(self):
        self.pushed = []

    def set_tier_counts(self, mapping, *, replace=False):
        self.pushed.append((dict(mapping), replace))


def _ctrl(tmp_path):
    c = C.__new__(C)
    c.working_dir = str(tmp_path)
    c._run_epoch = 3
    c.ui = _UI()
    c.state_manager = type("SM", (), {"get_active_run_id":
                                      staticmethod(lambda: "default")})()
    return c


def test_the_apply_pushes_the_items_and_keeps_the_map(tmp_path):
    c = _ctrl(tmp_path)
    stamp = c._tier_counts_stamp()
    c._on_tier_counts_counted(stamp + (True,), {
        "a.tif": {"good": 2, "review": 0, "bad": 0, "garbage": 0},
        "b.tif": None})
    (mapping, replace), = c.ui.pushed
    assert replace is True
    assert dict((w, n) for w, n, _c in mapping["a.tif"])["good"] == 2
    assert mapping["b.tif"] is None
    c._on_tier_counts_counted(stamp + (False,), {
        "b.tif": {"good": 0, "review": 0, "bad": 1, "garbage": 0}})
    mapping, replace = c.ui.pushed[-1]
    assert replace is False and list(mapping) == ["b.tif"]
    assert set(c._tier_counts) == {"a.tif", "b.tif"}


@pytest.mark.parametrize("change", ["workspace", "run", "epoch"])
def test_a_result_from_another_workspace_or_run_is_discarded(tmp_path,
                                                             change):
    c = _ctrl(tmp_path)
    stamp = c._tier_counts_stamp()
    if change == "workspace":
        c.working_dir = str(tmp_path / "other")
    elif change == "run":
        c.state_manager.get_active_run_id = lambda: "cells"
    else:
        c._run_epoch = 4
    c._on_tier_counts_counted(stamp + (True,),
                              {"a.tif": {"good": 1}})
    assert c.ui.pushed == []
    assert not getattr(c, "_tier_counts", {})


def test_the_requests_point_at_the_active_runs_features_csv(tmp_path,
                                                             monkeypatch):
    c = _ctrl(tmp_path)
    seen = {}

    def fake_start(requests, stamp, on_counted, **kw):
        seen.update(requests=requests, stamp=stamp, slot=on_counted)
        return TC.TierCountState()
    monkeypatch.setattr(TC, "start_tier_counts", fake_start)
    c._listed_patch_names = lambda: ["p1.tif", "p2.tif"]
    c._start_tier_counts("folder load")
    from napari_epicycle.fusion_reconstruction import _paths
    from napari_epicycle.fusion_reconstruction.controller import (
        _active_run_path_id,
    )
    want = _paths.qc_dir(str(tmp_path), run_id=_active_run_path_id(c))
    assert seen["requests"] == [
        ("p1.tif", str(want / "p1_features.csv")),
        ("p2.tif", str(want / "p2_features.csv"))]
    assert seen["stamp"] == c._tier_counts_stamp() + (True,)
    assert seen["slot"].__self__ is c          # a bound method, no lambda
    c._start_tier_counts("commit triage", ["p2.tif"])
    assert seen["requests"] == [("p2.tif", str(want / "p2_features.csv"))]
    assert seen["stamp"][-1] is False


def test_no_workspace_starts_nothing(tmp_path, monkeypatch):
    c = _ctrl(tmp_path)
    c.working_dir = ""
    monkeypatch.setattr(TC, "start_tier_counts",
                        lambda *a, **k: pytest.fail("started"))
    c._start_tier_counts("folder load")


_TRIGGERS = {
    "_load_stack_dir": "folder load",
    "_switch_active_run": "run switch",
    "_refresh_lists_on_step_enter": "step enter",
    "_on_qc_extract_finished": "extraction",
    "_on_commit_triage_finished": "commit triage",
    "_on_save_curation_finished": "save curation",
    "_forget_patch": "forget",
    "_cleanup_stale_qc": "stale QC",
}


@pytest.mark.parametrize("method,why", sorted(_TRIGGERS.items()))
def test_every_disk_door_starts_a_count(method, why):
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(C, method))))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_start_tier_counts"]
    assert calls, f"{method} starts no tier count"
    assert any(isinstance(a, ast.Constant) and a.value == why
               for n in calls for a in n.args)


@pytest.mark.parametrize("method", ["on_step_exit", "shutdown"])
def test_the_count_is_stopped_on_exit_and_shutdown(method):
    import inspect
    assert "_stop_tier_counts()" in inspect.getsource(getattr(C, method))
