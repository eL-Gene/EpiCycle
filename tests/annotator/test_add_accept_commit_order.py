"""Accept commits BEFORE it tears anything down (B2 fix pass L-M1 / L-L3).

The rule: by the time Accept tears the page down the
volume is already written, and the commit callback is the ONLY thing that
sets the dirty flag and pushes the undo entry.  It used to run after seven
teardown calls, so an exception in any of them lost the edit (written, no
dirty flag, no undo).  And the session list -- what "Undo all" takes back --
was appended before the callback ran and kept even when it raised.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .test_add_click_markers import h  # noqa: F401 - the fixture


@pytest.fixture
def hx(h):  # noqa: F811
    h.adv.close_add_numbers = lambda: None
    res = np.zeros((16, 26, 26), np.int32)
    res[8:12, 15:19, 15:19] = 40
    h._add_preview_result = res
    h._add_preview_roi = tuple(h._add_roi)
    h.adv.checked = [40]
    return h


def test_the_commit_callback_runs_before_the_teardown(hx):
    calls = []
    hx._on_add_committed_cb = (
        lambda ids, **k: calls.append((list(ids), list(hx._add_session))))

    def boom():
        raise RuntimeError("a teardown step failed")

    hx._remove_add_candidate = boom
    with pytest.raises(RuntimeError):
        hx._accept_add()
    assert calls and calls[0][0] == [40], (
        "the edit is written but was never committed: no dirty flag, no undo")
    vol = hx.sync_manager.shared_labels["L"]
    assert (vol == 40).any()
    assert not hx._add_running, "the lock is still released exactly once"


def test_the_session_is_appended_only_after_the_callback_returned(hx):
    seen = {}
    hx._on_add_committed_cb = (
        lambda ids, **k: seen.setdefault("session", list(hx._add_session)))
    hx._accept_add()
    assert seen["session"] == []
    assert [e["ids"] for e in hx._add_session] == [[40]]
    assert hx._add_piece_ids == [40]


def test_a_raising_callback_does_not_count_the_ids_for_undo_all(hx):
    def raises(ids, **k):
        raise ValueError("C:/some/long/path/that/must/not/reach/the/page")

    hx._on_add_committed_cb = raises
    hx._accept_add()
    assert hx._add_session == []
    assert hx._add_piece_ids == []
    assert hx._add_transient == AP.ADD_S1_COMMIT_FAILED
    s1 = hx.adv.last("status_text")
    assert s1 == AP.ADD_S1_COMMIT_FAILED
    assert "path" not in s1
