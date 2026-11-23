"""Clear objects on a uint32 patch.

On a uint32 labels file the annotator's shared volume starts as an int32
COPY of the main layer's data.  A Clear re-points the shared volume at the
main layer before it counts and before it writes, so the objects it zeroes
are zeroed where the user sees them, and an Undo -- which restores the main
layer -- leaves nothing in a stale copy that a later sync could paste back.
"""

from __future__ import annotations

import numpy as np

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread, install_fakes
from .test_add_clear import BOX, _controller, _df, _harness, _volume


def _uint32_harness():
    main = _volume(np.uint32)
    h = _harness(main)
    # what the launch leaves on a uint32 patch: shared is an int32 copy
    h.sync_manager.shared_labels["L"] = main.astype(np.int32)
    assert h.sync_manager.shared_labels["L"] is not main
    return h, main


def test_the_clear_writes_the_array_the_user_sees(monkeypatch):
    install_fakes(monkeypatch)
    h, main = _uint32_harness()
    before = main.copy()
    h._on_add_clear_requested("box")
    assert h.sync_manager.shared_labels["L"] is main, "aligned first"
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert not np.isin(main, [3, 4, 5]).any()
    assert np.array_equal(main[main != 0], before[np.isin(before, [6, 9])])


def test_no_resurrection_after_undo_or_the_next_sync(monkeypatch):
    install_fakes(monkeypatch)
    h, main = _uint32_harness()
    before = main.copy()
    ctrl = _controller(h, _df())
    h._on_add_clear_requested("patch")
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert not main.any()
    h._reject_add()
    ctrl._undo_last_curation()
    layer = h.viewer.layers[-1]
    assert np.array_equal(layer.data, before)
    shared = h.sync_manager.shared_labels["L"]
    assert np.array_equal(shared, before)
    # the next op realigns and reads what the user sees: every object the
    # Undo put back is there to count, and no stale copy is in between
    h.place_box(BOX)
    h.confirm_answer = False
    h._on_add_clear_requested("patch")
    assert h.sync_manager.shared_labels["L"] is layer.data
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert AP.ADD_CLEAR_CONFIRM_PATCH.format(n=5) in h.confirms[-1]
    assert np.array_equal(layer.data, before)


def test_an_array_replaced_before_the_confirm_clears_nothing(monkeypatch):
    """Another write replaced the main layer's array while the confirm was
    open: the realign after the confirm points shared at the new array,
    its identity no longer matches the count's, and nothing is cleared."""
    install_fakes(monkeypatch)
    h, main = _uint32_harness()
    layer = h.viewer.layers[-1]
    replaced = []

    def _confirm(text):
        layer.data = main.copy()
        replaced.append(layer.data)
        return True

    h._add_clear_confirm = _confirm
    h._on_add_clear_requested("box")
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert replaced and not h.commits
    assert h._add_transient == AP.ADD_S1_CLEAR_STALE
    assert h.sync_manager.shared_labels["L"] is replaced[0]
    assert np.isin(replaced[0], [3, 4, 5]).any()
    assert np.isin(main, [3, 4, 5]).any()
    assert not h._add_busy
