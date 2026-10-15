"""Brick 5 fix 4 - the two main-thread Step-1 operations with NO indicator and
no way to judge them headlessly get a `[STEP1-PERF]`-gated measurement.

`_maybe_restore_step1_sources` (a `load_state` + a `largedata/` listdir + a
per-store `.zattrs` read, on step-enter / workspace-set) and `_rebuild_loc_boxes`
(O(visible boxes), each a sidecar lookup, re-run on every eye toggle) are both
synchronous main-thread work whose real cost depends on a real workspace.  An
indicator would be a guess; a measurement is not.  Gated on the existing
`EPICYCLE_SYNC_DEBUG=1` probe so the steady-state cost is one `os.environ.get`.

Qt-free.
"""
from __future__ import annotations

import types

from napari_epicycle.data_preparation import controller as ctlmod


def _ctl(monkeypatch, on=True):
    monkeypatch.setenv("EPICYCLE_SYNC_DEBUG", "1" if on else "0")
    c = ctlmod.DataPreparationController.__new__(ctlmod.DataPreparationController)
    c.logged = []
    c._log = lambda tag, msg: c.logged.append((tag, msg))
    return c


def _perf_lines(c):
    return [m for t, m in c.logged if t == "STEP1-PERF"]


# --------------------------------------------------------------------------- #
# source restore / store reconcile
# --------------------------------------------------------------------------- #

def _restore_ctl(monkeypatch, tmp_path, on=True):
    c = _ctl(monkeypatch, on)
    c.widget = types.SimpleNamespace(
        output_file_path=types.SimpleNamespace(text=lambda: str(tmp_path)))
    c._step1_sources_restored = set()
    c._restore_step1_sources = lambda ws: c.logged.append(("RAN", ws))
    return c


def test_restore_is_measured(monkeypatch, tmp_path):
    c = _restore_ctl(monkeypatch, tmp_path)
    c._maybe_restore_step1_sources()
    assert ("RAN", str(tmp_path)) in c.logged
    assert any("source restore + store reconcile" in m for m in _perf_lines(c))
    assert any("ms" in m for m in _perf_lines(c))


def test_restore_is_not_measured_when_the_probe_is_off(monkeypatch, tmp_path):
    c = _restore_ctl(monkeypatch, tmp_path, on=False)
    c._maybe_restore_step1_sources()
    assert _perf_lines(c) == []


def test_a_failing_restore_is_still_measured_and_still_swallowed(
        monkeypatch, tmp_path):
    """The measurement lives in a `finally`, so a best-effort failure - the case
    most worth timing - is not the one case that reports nothing."""
    c = _restore_ctl(monkeypatch, tmp_path)

    def _boom(_ws):
        raise RuntimeError("state unreadable")

    c._restore_step1_sources = _boom
    c._maybe_restore_step1_sources()          # must not raise
    assert any("restore failed" in m for t, m in c.logged if t == "STEP1-RESTORE")
    assert _perf_lines(c)


def test_restore_is_still_once_per_workspace(monkeypatch, tmp_path):
    c = _restore_ctl(monkeypatch, tmp_path)
    c._maybe_restore_step1_sources()
    c._maybe_restore_step1_sources()
    assert len([1 for t, _ in c.logged if t == "RAN"]) == 1


# --------------------------------------------------------------------------- #
# patch-location overlay rebuild
# --------------------------------------------------------------------------- #

def _loc_ctl(monkeypatch, on=True):
    c = _ctl(monkeypatch, on)
    c._visible_patch_locations = {"p1", "p2"}
    c._rebuild_loc_boxes_inner = lambda: c.logged.append(("RAN", "loc"))
    return c


def test_overlay_rebuild_is_measured(monkeypatch):
    c = _loc_ctl(monkeypatch)
    c._rebuild_loc_boxes()
    assert ("RAN", "loc") in c.logged
    line = _perf_lines(c)[0]
    assert "patch-location overlay rebuild" in line
    assert "2 visible" in line, "the count is what makes the number readable"


def test_overlay_rebuild_is_not_measured_when_the_probe_is_off(monkeypatch):
    c = _loc_ctl(monkeypatch, on=False)
    c._rebuild_loc_boxes()
    assert _perf_lines(c) == []


def test_a_raising_rebuild_still_reports_its_time(monkeypatch):
    c = _loc_ctl(monkeypatch)

    def _boom():
        raise RuntimeError("overlay gone")

    c._rebuild_loc_boxes_inner = _boom
    try:
        c._rebuild_loc_boxes()
    except RuntimeError:
        pass
    else:                                     # pragma: no cover
        raise AssertionError("the wrapper must not swallow the failure")
    assert _perf_lines(c), "the timing lives in a finally"
