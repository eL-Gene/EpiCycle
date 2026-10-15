"""Brick 5 fix 3 - the ④ export paths get honest indicators.

Two defects, both main-thread:

* **Export all ready** set its determinate bar per patch but never pumped Qt, so
  it rendered AT REST for the whole run and then jumped to idle.  A decorative
  bar is worse than no bar - it says "nothing is happening" while a permanent
  action writes files.  Its sibling `_export_named_patches` DID pump; the two
  copies are now one `_run_patch_export_loop`.
* **Single-patch export** wrote a whole TIFF with no indicator and no lock at
  all - on slow storage the UI simply froze, and a second click could start a
  second write of the same patch.

The loop stays on the main thread on purpose: the cost is `tifffile.imwrite`,
and the existing `_ExtractWorker` cannot be reused (it slices the SOURCE volume
from boxes and is the sole `step1_state.json` writer while it runs).  What the
pump buys is a repaint; what the `_left_step1` break buys is not writing on into
a step the user has left - the pump is what makes that reachable.

Qt-free: `__new__` controllers + a recording bar stub.
"""
from __future__ import annotations

import sys
import types

import pytest

from napari_epicycle.data_preparation import controller as ctlmod


class _Bar:
    def __init__(self):
        self.ranges = []
        self.values = []
        self.formats = []

    def setRange(self, lo, hi):
        self.ranges.append((lo, hi))

    def setValue(self, v):
        self.values.append(v)

    def setFormat(self, f):
        self.formats.append(f)


@pytest.fixture()
def pumped(monkeypatch):
    """A fake `qtpy.QtWidgets.QApplication` whose `processEvents` is counted."""
    calls = []
    mod = types.ModuleType("qtpy.QtWidgets")
    mod.QApplication = types.SimpleNamespace(
        processEvents=lambda: calls.append(1))
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", mod)
    return calls


def _ctl(bar=None):
    c = ctlmod.DataPreparationController.__new__(ctlmod.DataPreparationController)
    c.widget = types.SimpleNamespace(export_progress_bar=bar)
    c._left_step1 = False
    c._migration_offer_timer = None
    c.status = []
    c.busy = []
    c.exported = []
    c._set_export_status = lambda m, success=True: c.status.append((m, success))
    c._set_step1_busy = lambda on: c.busy.append(bool(on))
    c._refresh_conditioning_status = lambda: None
    c._export_patch_by_name = (
        lambda name, require_conditioned=False:
        c.exported.append((name, require_conditioned)) or True)
    return c


# --------------------------------------------------------------------------- #
# the shared loop
# --------------------------------------------------------------------------- #

def test_the_loop_repaints_once_per_patch(pumped):
    bar = _Bar()
    c = _ctl(bar)
    c._run_patch_export_loop(["a", "b", "c"], require_conditioned=False,
                             noun="patch(es)")
    assert len(pumped) == 3, "one repaint per patch, or the bar is decorative"
    assert bar.values[:3] == [1, 2, 3]
    assert bar.ranges[0] == (0, 3)
    assert bar.ranges[-1] == (0, 1), "the bar returns to rest"
    assert c.busy == [True, False]
    assert c.status == [("Exported 3 patch(es).", True)]


def test_leaving_step_1_mid_export_stops_the_loop(pumped):
    """`processEvents` can deliver the step switch; without the break the loop
    goes on writing behind a bar the user can no longer see."""
    bar = _Bar()
    c = _ctl(bar)

    def _leave(name, require_conditioned=False):
        c.exported.append(name)
        c._left_step1 = True
        return True

    c._export_patch_by_name = _leave
    c._run_patch_export_loop(["a", "b", "c"], require_conditioned=False,
                             noun="patch(es)")
    assert c.exported == ["a"]
    assert c.status[0][1] is False
    assert "left Step 1" in c.status[0][0]
    assert bar.ranges[-1] == (0, 1)
    assert c.busy == [True, False], "the lock still releases"


def test_a_failure_mid_loop_still_unlocks_and_says_so(pumped):
    bar = _Bar()
    c = _ctl(bar)

    def _boom(name, require_conditioned=False):
        raise RuntimeError("disk full")

    c._export_patch_by_name = _boom
    with pytest.raises(RuntimeError):
        c._run_patch_export_loop(["a"], require_conditioned=False,
                                 noun="patch(es)")
    assert c.busy == [True, False]
    assert bar.ranges[-1] == (0, 1)
    assert c.status[0] == ("Export failed after 0 of 1.", False)


def test_partial_failures_are_reported(pumped):
    c = _ctl(_Bar())
    c._export_patch_by_name = lambda name, require_conditioned=False: name != "b"
    c._run_patch_export_loop(["a", "b"], require_conditioned=False,
                             noun="ROI(s)", success_tail=" successfully")
    assert c.status == [("Exported 1 ROI(s); 1 failed.", False)]


def test_the_loop_survives_a_widget_with_no_bar(pumped):
    c = _ctl(None)
    c._run_patch_export_loop(["a"], require_conditioned=True, noun="ROI(s)")
    assert c.exported == [("a", True)]
    assert c.busy == [True, False]


# --------------------------------------------------------------------------- #
# the two callers
# --------------------------------------------------------------------------- #

def test_export_all_ready_goes_through_the_pumped_loop(pumped):
    c = _ctl(_Bar())
    c.extracted_patches = {
        "p1": {"selected_z_range": (0, 4)},
        "p2": {"selected_z_range": None},
        "p3": {"selected_z_range": (2, 6)},
    }
    c._on_export_all_ready_clicked()
    assert c.exported == [("p1", True), ("p3", True)]
    assert len(pumped) == 2, "export-all never repainted before this fix"
    assert c.status == [("Exported 2 ROI(s) successfully.", True)]


def test_export_named_patches_goes_through_the_same_loop(pumped):
    c = _ctl(_Bar())
    c._export_named_patches(["p1", "p2"])
    assert c.exported == [("p1", False), ("p2", False)]
    assert len(pumped) == 2
    assert c.status == [("Exported 2 patch(es).", True)]


# --------------------------------------------------------------------------- #
# single-patch export
# --------------------------------------------------------------------------- #

def _single(bar):
    c = _ctl(bar)
    c.widget.extracted_roi_list = types.SimpleNamespace(
        currentItem=lambda: types.SimpleNamespace(text=lambda: "wound_patch_1"))
    return c


def test_single_patch_export_shows_a_marquee_and_takes_the_lock():
    bar = _Bar()
    c = _single(bar)
    c._export_current_patch()
    assert c.exported == [("wound_patch_1", False)]
    assert bar.ranges[0] == (0, 0), "one file has no denominator -> marquee"
    assert "wound_patch_1" in bar.formats[0]
    assert bar.ranges[-1] == (0, 1), "back to rest"
    assert c.busy == [True, False]


def test_single_patch_export_unlocks_even_when_the_write_raises():
    bar = _Bar()
    c = _single(bar)

    def _boom(name, require_conditioned=False):
        raise OSError("no space left")

    c._export_patch_by_name = _boom
    with pytest.raises(OSError):
        c._export_current_patch()
    assert c.busy == [True, False]
    assert bar.ranges[-1] == (0, 1)


def test_single_patch_export_with_no_selection_touches_nothing():
    bar = _Bar()
    c = _single(bar)
    c.widget.extracted_roi_list = types.SimpleNamespace(currentItem=lambda: None)
    c._export_current_patch()
    assert c.busy == [], "no lock taken for a no-op"
    assert bar.ranges == []
    assert c.status == [("No patch selected.", False)]
