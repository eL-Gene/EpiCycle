"""Closing the annotator with Add work on screen asks first (B2.6).

``OperationSlot("add", priority=25)``: active for busy / candidates / clicks,
not for REVIEW alone nor idle; the description by state; Discard runs
``_reject_add`` (writes nothing, releases the lock), Go Back keeps every
artefact; ``_reject_add`` twice is a no-op the second time.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.annotator.curation_3d_widget import (
    Curation3DAnnotator,
    OperationSlot,
)
from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness
from .test_add_click_markers import h  # noqa: F401 - the fixture


def _slot(hh):
    hh._reject_smooth = hh._reject_split = hh._reject_interpolation = (
        lambda: None)
    hh._discard_batch_refine = lambda: None
    hh._smooth_running = hh._split_running = False
    hh._interp_snapshot = None
    hh._interp_refine_mode = False
    Harness._register_operation_slots = (
        Curation3DAnnotator._register_operation_slots)
    Harness._has_uncommitted_work = Curation3DAnnotator._has_uncommitted_work
    Harness._discard_uncommitted_work = (
        Curation3DAnnotator._discard_uncommitted_work)
    hh._register_operation_slots()
    slot = [s for s in hh._operation_slots if s.name == "add"]
    assert len(slot) == 1
    return slot[0]


def test_the_slot_is_registered_at_priority_25(h):  # noqa: F811
    s = _slot(h)
    assert isinstance(s, OperationSlot)
    assert s.priority == 25


def test_idle_and_review_are_not_work(h):  # noqa: F811
    s = _slot(h)
    assert not s.is_active()                    # a box, nothing else
    h._add_running = False
    h._add_review = True
    h._add_click_markers = [(1, 1, 1)]
    assert not s.is_active(), "REVIEW alone is not work"


@pytest.mark.parametrize("state,want", [
    ("busy", AP.ADD_CLOSE_BUSY),
    ("candidates", AP.ADD_CLOSE_CANDIDATES.format(n=2)),
    ("clicks", AP.ADD_CLOSE_CLICKS_ONE),
])
def test_busy_candidates_clicks_are_work_with_their_sentence(h, state, want):  # noqa: F811,E501
    s = _slot(h)
    if state == "busy":
        h._add_busy = True
    elif state == "candidates":
        h._add_preview_result = np.zeros((2, 2, 2), np.int32)
        h._add_preview_res = AP.AddPreviewResult(ids=[40, 41])
    else:
        h._add_click_markers = [(12, 20, 20)]
    assert s.is_active()
    assert s.description() == want
    assert h._has_uncommitted_work(include_edited_slices=False) == (True, want)


def test_discard_rejects_and_releases_the_lock(h):  # noqa: F811
    s = _slot(h)
    h._add_marker_click((12, 20, 20))
    # _discard_uncommitted_work calls reject() on every ACTIVE slot
    assert s.is_active()
    s.reject()
    assert not h._add_running
    assert h._add_marker_layer is None
    assert h.locks[-1] is False


def test_go_back_keeps_everything(h, monkeypatch):  # noqa: F811
    _slot(h)
    h._add_marker_click((12, 20, 20))
    h._confirm_discard_dialog = lambda reason: False

    class _Ev:
        ignored = False

        def ignore(self):
            self.ignored = True

    ev = _Ev()
    Curation3DAnnotator.closeEvent(h, ev)
    assert ev.ignored
    assert h._add_running and h._add_click_markers
    assert h._add_marker_layer is not None


def test_reject_twice_touches_nothing_the_second_time(h):  # noqa: F811
    _slot(h)
    h._add_marker_click((12, 20, 20))
    h._reject_add()
    pushes = len(h.adv.pushes)
    locks = list(h.locks)
    h._reject_add()                   # _close_cleanup's second call
    assert h.locks == locks
    assert not h._add_running
    assert len(h.adv.pushes) >= pushes          # a UI refresh, no raise
