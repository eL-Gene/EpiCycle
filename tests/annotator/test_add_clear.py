"""Clear objects on the Add page: the orchestrator, end to end.

The real annotator methods on the Qt-free Add harness; the counting runs on
a ``FakeThread`` whose ``started`` the test emits; the confirm is a seam.
The controller half (the undo entry, the table rows) is the real
controller built with ``__new__``.

Pinned:

* one Clear = one undo entry; Undo puts the volume back byte for byte and
  the table rows back as they were;
* only the listed objects lose voxels -- all of their voxels, inside and
  outside the box -- and nothing else changes;
* new ids stay above every id a Clear removed, with no QC table at all;
  that floor outlives the box and is dropped at a patch switch and a close;
* Stop drops a count; a commit that raised after its undo entry landed
  keeps the Clear; Commit Triage empties the curation undo;
* the refusals (busy, counting, candidates, the review, Batch, no box);
* the count holds the page busy: Preview is refused while it runs;
* a box that changed between the count and the confirm clears nothing;
* after a Clear the draw snapshot is dropped AFTER the commit, so Clear
  Edits cannot paste the objects back;
* cleared objects leave a live isolation (after the commit) and "Added
  this session";
* the count runs on the worker, never on the main thread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread, Harness, install_fakes

BOX = (slice(0, 6), slice(0, 10), slice(0, 10))


class _Layer:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1


def _volume(dtype=np.int32):
    v = np.zeros((8, 16, 12), dtype)
    v[1:3, 1:4, 1:4] = 3          # wholly inside the box
    v[2, 5:13, 6] = 5             # 5 inside (y 5..9), 3 outside -> taken
    v[4, 7:15, 8] = 6             # 3 inside, 5 outside -> stays
    v[7, 2:5, 2:5] = 9            # z 7 is outside the box -> stays
    v[0, 0, 0] = 4                # wholly inside
    return v


def _harness(vol, *, with_layer=True):
    h = Harness(vol)
    if with_layer:
        h.viewer.layers.append(_Layer("L", vol))
    h.place_box(BOX)
    h.commits = []
    h.events = []
    h.confirms = []
    h.confirm_answer = True

    def _cb(ids, **kw):
        h.events.append("commit")
        h.commits.append((list(ids), kw))

    h._on_discard_garbage_cb = _cb
    h._add_clear_post = lambda fn: fn()

    def _confirm(text):
        h.confirms.append(text)
        return h.confirm_answer

    h._add_clear_confirm = _confirm
    real_drop = h._clear_draw_snapshot

    def _drop():
        h.events.append("drop_snapshot")
        real_drop()

    h._clear_draw_snapshot = _drop
    return h


@pytest.fixture
def fakes(monkeypatch):
    install_fakes(monkeypatch)


def _clear(h, mode="box", finish=True):
    h._on_add_clear_requested(mode)
    th = FakeThread.instances[-1]
    th.run()
    if finish:
        th.end()
    return th


def _df():
    return pd.DataFrame({
        "label": [3, 4, 5, 6, 9, 5],
        "qc_tier": ["good", "curated", "review", "good", "good", "review"],
        "feature_state": ["fresh"] * 6,
        "label_status": ["active"] * 5 + ["superseded"],
        "area": [18, 1, 8, 8, 9, 7],
    })


# --------------------------------------------------------------------------- #
# what is zeroed
# --------------------------------------------------------------------------- #
def test_a_box_clear_zeroes_exactly_the_listed_objects_whole(fakes):
    vol = _volume()
    before = vol.copy()
    h = _harness(vol)
    _clear(h)
    assert len(h.confirms) == 1
    assert h.commits and sorted(h.commits[0][0]) == [3, 4, 5]
    gone = np.isin(before, [3, 4, 5])
    assert not np.isin(vol, [3, 4, 5]).any()
    # object 5 went WHOLE, its 3 voxels outside the box too
    assert (before[2, 10:13, 6] == 5).all() and (vol[2, 10:13, 6] == 0).all()
    np.testing.assert_array_equal(vol[~gone], before[~gone])
    assert not h._add_busy


def test_a_patch_clear_takes_every_object(fakes):
    vol = _volume()
    h = _harness(vol)
    _clear(h, mode="patch")
    assert sorted(h.commits[0][0]) == [3, 4, 5, 6, 9]
    assert not vol.any()


def test_the_confirm_says_the_count_the_outside_and_the_good(fakes):
    vol = _volume()
    h = _harness(vol)
    h._get_kept_tier_count_cb = lambda ids: 2
    _clear(h)
    text = h.confirms[0]
    assert AP.ADD_CLEAR_CONFIRM_BOX.format(n=3) in text
    assert AP.ADD_CLEAR_CONFIRM_OUTSIDE_ONE in text
    assert AP.ADD_CLEAR_CONFIRM_GOOD.format(g=2) in text
    assert AP.ADD_CLEAR_CONFIRM_UNDO in text


def test_go_back_changes_nothing(fakes):
    vol = _volume()
    before = vol.copy()
    h = _harness(vol)
    h.confirm_answer = False
    _clear(h)
    np.testing.assert_array_equal(vol, before)
    assert not h.commits
    assert not h._add_busy
    assert h._add_transient is None


def test_nothing_mostly_inside_says_so_and_asks_nothing(fakes):
    vol = np.zeros((8, 16, 12), np.int32)
    vol[7, 2:5, 2:5] = 9
    h = _harness(vol)
    _clear(h)
    assert not h.confirms and not h.commits
    assert h._add_transient == AP.ADD_S1_CLEAR_NOTHING
    h2 = _harness(np.zeros((8, 16, 12), np.int32))
    _clear(h2, mode="patch")
    assert h2._add_transient == AP.ADD_S1_CLEAR_EMPTY_PATCH


def test_s1_and_the_footer_after_a_clear(fakes):
    vol = _volume()
    h = _harness(vol)
    _clear(h)
    s1 = h._add_transient
    assert s1.startswith(AP.ADD_S1_CLEARED.format(n=3))
    assert s1.endswith(AP.ADD_S1_CLEARED_OUTSIDE)
    assert h.adv.last("footer_text") == AP.ADD_FOOT_BOX_CLEARED
    # the box is measured again (its empty share changed)
    assert h._add_info is None


# --------------------------------------------------------------------------- #
# the worker, busy, stale
# --------------------------------------------------------------------------- #
def test_the_count_runs_on_the_worker(fakes, monkeypatch):
    calls = []
    real = AP.clear_plan

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(AP, "clear_plan", spy)
    h = _harness(_volume())
    h._on_add_clear_requested("box")
    assert calls == [], "nothing counted on the main thread"
    th = FakeThread.instances[-1]
    assert h._add_clear_worker.kind == "add_clear"
    th.run()
    assert len(calls) == 1
    # the worker reads the live shared array (the Add lock is held)
    assert calls[0]["labels"] is h.sync_manager.shared_labels["L"]


def test_the_count_holds_the_page_busy(fakes):
    h = _harness(_volume())
    h._on_add_clear_requested("box")
    assert h._add_busy
    assert h.adv.last("status_text") == AP.ADD_S1_CLEAR_COUNTING
    assert h.adv.last("stop_enabled") is True       # Stop drops the count
    n = len(FakeThread.instances)
    h._preview_add()
    assert len(FakeThread.instances) == n, "Preview refused while counting"
    h._on_add_clear_requested("patch")
    assert len(FakeThread.instances) == n, "one count at a time"


def test_a_box_moved_before_the_confirm_is_stale(fakes):
    vol = _volume()
    before = vol.copy()
    h = _harness(vol)

    def _confirm(text):
        h._add_roi = (slice(0, 6), slice(0, 9), slice(0, 10))
        return True

    h._add_clear_confirm = _confirm
    _clear(h)
    np.testing.assert_array_equal(vol, before)
    assert not h.commits
    assert h._add_transient == AP.ADD_S1_CLEAR_STALE
    assert not h._add_busy


def test_a_volume_written_before_the_confirm_is_stale(fakes):
    vol = _volume()
    h = _harness(vol)

    def _confirm(text):
        h._add_volume_epoch = int(getattr(h, "_add_volume_epoch", 0)) + 1
        return True

    h._add_clear_confirm = _confirm
    _clear(h)
    assert not h.commits
    assert h._add_transient == AP.ADD_S1_CLEAR_STALE


def test_an_exit_while_counting_drops_the_answer(fakes):
    vol = _volume()
    before = vol.copy()
    h = _harness(vol)
    h._on_add_clear_requested("box")
    th = FakeThread.instances[-1]
    h._reject_add()
    th.run()
    np.testing.assert_array_equal(vol, before)
    assert not h.confirms and not h.commits


def test_a_failed_count_says_so_and_releases_busy(fakes, monkeypatch):
    def boom(**_kw):
        raise RuntimeError("no")

    monkeypatch.setattr(AP, "clear_plan", boom)
    h = _harness(_volume())
    _clear(h)
    assert h._add_transient == AP.ADD_S1_CLEAR_FAILED
    assert not h._add_busy


def test_a_failed_commit_puts_the_volume_back(fakes):
    vol = _volume()
    before = vol.copy()
    h = _harness(vol)

    def _raise(ids, **kw):
        raise RuntimeError("controller down")

    h._on_discard_garbage_cb = _raise
    _clear(h)
    np.testing.assert_array_equal(vol, before)
    assert h._add_transient == AP.ADD_S1_CLEAR_FAILED


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def _refused(h):
    n = len(FakeThread.instances)
    h._on_add_clear_requested("box")
    return len(FakeThread.instances) == n


def test_refused_with_candidates_up(fakes):
    h = _harness(_volume())
    h._add_preview_result = np.zeros((6, 10, 10), np.int32)
    assert _refused(h)


def test_refused_in_the_review_and_with_no_box(fakes):
    h = _harness(_volume())
    h._add_review = True
    assert _refused(h)
    h2 = _harness(_volume())
    h2._add_running = False
    assert _refused(h2)


def test_refused_while_a_preview_runs(fakes):
    h = _harness(_volume())
    h._add_busy = True
    assert _refused(h)
    assert h._add_transient == AP.ADD_S1_BUSY_REFUSED


def test_refused_while_batch_runs(fakes):
    h = _harness(_volume())
    h._batch_running = True
    assert _refused(h)


# --------------------------------------------------------------------------- #
# what a Clear leaves behind
# --------------------------------------------------------------------------- #
def test_the_draw_snapshot_is_dropped_after_the_commit(fakes):
    h = _harness(_volume())
    h._draw_snapshot = np.ones((2, 2, 2), np.int32)
    _clear(h)
    assert "commit" in h.events and "drop_snapshot" in h.events
    assert h.events.index("drop_snapshot") > h.events.index("commit")


def test_cleared_ids_leave_a_live_isolation(fakes):
    h = _harness(_volume())
    forgotten = []
    h._iso_live_ids = lambda: frozenset({3, 6, 5})

    def _forget(ids):
        h.events.append("forget")
        forgotten.append(set(ids))

    h._iso_forget = _forget
    _clear(h)
    assert forgotten == [{3, 5}]
    # once the Clear stands: after the commit
    assert h.events.index("forget") > h.events.index("commit")


def test_a_failed_commit_leaves_the_isolation_alone(fakes):
    h = _harness(_volume())
    forgotten = []
    h._iso_live_ids = lambda: frozenset({3, 6, 5})
    h._iso_forget = lambda ids: forgotten.append(set(ids))

    def _raise(ids, **kw):
        raise RuntimeError("controller down")

    h._on_discard_garbage_cb = _raise
    _clear(h)
    assert forgotten == []
    assert h._add_transient == AP.ADD_S1_CLEAR_FAILED


def test_nothing_isolated_forgets_nothing(fakes):
    h = _harness(_volume())
    forgotten = []
    h._iso_live_ids = lambda: frozenset()
    h._iso_forget = lambda ids: forgotten.append(set(ids))
    _clear(h)
    assert forgotten == []


def test_cleared_ids_leave_the_session_list(fakes):
    h = _harness(_volume())
    h._add_session = [{"ids": [3, 6], "sizes": {3: 18, 6: 8}},
                      {"ids": [4], "sizes": {4: 1}}]
    h._add_piece_ids = [3, 6, 4]
    _clear(h)
    # the rows leave; each entry keeps what its Accept wrote, and an Accept
    # cleared to nothing stays (its undo entry is still on the stack)
    assert h._add_session == [
        {"ids": [6], "sizes": {6: 8}, "accept_ids": [3, 6],
         "accept_sizes": {3: 18, 6: 8}},
        {"ids": [], "sizes": {}, "accept_ids": [4], "accept_sizes": {4: 1}}]
    assert h._add_piece_ids == [6]


def test_the_undo_entry_carries_its_words_and_its_number(fakes):
    h = _harness(_volume())
    _clear(h)
    ids, kw = h.commits[0]
    assert kw["description"] == AP.ADD_CLEAR_UNDO_TEXT.format(n=3)
    seq = kw["extra"]["add_clear_seq"]
    assert isinstance(seq, int) and seq > 0
    assert kw["roi_slices"] is not None and kw["old_roi_data"] is not None
    # the cleared ids are recorded under the same number -- ids only, no
    # mask (where they were is in the undo entry)
    assert h._add_clear_records[seq] == (3, 4, 5)


# --------------------------------------------------------------------------- #
# the id floor
# --------------------------------------------------------------------------- #
def test_ids_after_clear_skip_cleared_ids_without_qc(fakes):
    vol = np.zeros((8, 16, 12), np.int32)
    vol[1, 1, 1] = 3
    vol[2, 2, 2] = 50              # the highest id, inside the box
    vol[7, 1, 1] = 4               # outside the box
    h = _harness(vol)
    h._get_label_id_floor_cb = None      # no QC table
    _clear(h)
    assert vol.max() == 4
    assert h._add_next_label() == 51
    # the floor outlives the box ...
    h._reject_add()
    assert h._add_next_label() == 51
    # ... and is dropped at a patch switch
    h._teardown_add_for_switch()
    assert h._add_next_label() == 5


def test_the_floor_reads_no_context_and_goes_at_a_close(fakes):
    """A plain number: a marker click resolves no context for it.  A run
    switch closes the annotator, and a close drops it."""
    vol = np.zeros((8, 16, 12), np.int32)
    vol[2, 2, 2] = 50
    h = _harness(vol)
    _clear(h)
    calls = []
    real = h._reshape_context

    def _ctx():
        calls.append(1)
        return real()

    h._reshape_context = _ctx
    assert h._add_clear_floor_now() == 50
    assert calls == []
    assert h._add_next_label(cached=True) == 51
    h._add_forget_clears()              # what the close cleanup calls
    assert h._add_next_label() == 1


# --------------------------------------------------------------------------- #
# Stop, the words of one object, the reason Preview is grey
# --------------------------------------------------------------------------- #
def test_stop_drops_the_count(fakes):
    vol = _volume()
    before = vol.copy()
    h = _harness(vol)
    h._on_add_clear_requested("box")
    th = FakeThread.instances[-1]
    h._stop_add_preview()
    assert not h._add_busy
    assert h._add_transient is None
    assert h._add_clear_phase is None
    assert h.adv.last("status_text") != AP.ADD_S1_CLEAR_COUNTING
    # the count that could not stop mid-way lands later: dropped
    th.run()
    th.end()
    np.testing.assert_array_equal(vol, before)
    assert not h.confirms and not h.commits
    assert h._add_clear_thread is None
    # and the page counts again
    _clear(h)
    assert h.commits


def test_stop_is_grey_while_the_write_runs(fakes):
    h = _harness(_volume())
    h._add_busy = True
    h._add_clear_phase = "clearing"
    h._add_clear_n = 3
    h._add_refresh_ui()
    assert h.adv.last("stop_enabled") is False
    h._add_clear_phase = "counting"
    h._add_refresh_ui()
    assert h.adv.last("stop_enabled") is True


def test_the_confirm_of_one_object_says_it(fakes):
    vol = np.zeros((8, 16, 12), np.int32)
    vol[2, 5:13, 6] = 5                 # 5 inside, 3 outside -> taken
    h = _harness(vol)
    h._get_kept_tier_count_cb = lambda ids: 1
    _clear(h)
    text = h.confirms[0]
    assert text.startswith(AP.ADD_CLEAR_CONFIRM_BOX_ONE)
    assert AP.ADD_CLEAR_CONFIRM_OUTSIDE_IT in text
    assert AP.ADD_CLEAR_CONFIRM_GOOD_IT in text
    assert text.endswith(AP.ADD_CLEAR_CONFIRM_UNDO_IT)
    assert "of them" not in text and " 1 " not in text
    assert h._add_transient == (AP.ADD_S1_CLEARED_ONE + " "
                                + AP.ADD_S1_CLEARED_OUTSIDE_IT)


def test_the_cleared_line_keeps_why_preview_is_grey(fakes, monkeypatch):
    monkeypatch.setattr(AP, "preview_block",
                        lambda state: AP.ADD_S1_PICK_CHANNEL)
    h = _harness(_volume())
    _clear(h)
    assert h._add_transient == " ".join((
        AP.ADD_S1_CLEARED.format(n=3), AP.ADD_S1_CLEARED_OUTSIDE,
        AP.ADD_S1_PICK_CHANNEL))
    # a measure still running is not a reason
    monkeypatch.setattr(AP, "preview_block", lambda state: AP.ADD_S1_MEASURING)
    h2 = _harness(_volume())
    _clear(h2)
    assert h2._add_transient.endswith(AP.ADD_S1_CLEARED_OUTSIDE)


# --------------------------------------------------------------------------- #
# with the real controller: one undo entry, Undo byte for byte
# --------------------------------------------------------------------------- #
def _controller(h, df):
    from tests.fusion_reconstruction.test_undo_realigns_annotator_shared import (
        _make_controller,
    )

    ctrl = _make_controller(h.viewer.layers[-1], h)
    ctrl._qc_features_df = df
    ctrl._tiers_dirty = False
    h._on_discard_garbage_cb = ctrl._on_annotator_discard_garbage
    h._get_kept_tier_count_cb = ctrl._add_kept_tier_count
    h._get_live_add_clears_cb = ctrl._live_add_clear_seqs
    return ctrl


@pytest.mark.parametrize("mode", ["box", "patch"])
def test_clear_undo_byte_identical(fakes, mode):
    vol = _volume()
    before = vol.copy()
    h = _harness(vol)
    df = _df()
    ctrl = _controller(h, df.copy())
    _clear(h, mode=mode)
    assert len(ctrl._curation_undo_stack) == 1
    entry = ctrl._curation_undo_stack.peek()
    assert entry.action_type == "discard_garbage"
    seq = entry.extra["add_clear_seq"]
    assert ctrl._live_add_clear_seqs() == {seq}
    cleared = ctrl._qc_features_df
    gone = [3, 4, 5] if mode == "box" else [3, 4, 5, 6, 9]
    act = cleared[cleared["label_status"] != "superseded"]
    assert set(act[act["label"].isin(gone)]["label_status"]) == {"deleted"}
    # the superseded ghost of 5 keeps its status
    assert (cleared["label_status"] == "superseded").sum() == 1
    # Undo is the main Undo, after the box is closed
    h._reject_add()
    ctrl._undo_last_curation()
    np.testing.assert_array_equal(h.viewer.layers[-1].data, before)
    pd.testing.assert_frame_equal(ctrl._qc_features_df, df)
    assert ctrl._live_add_clear_seqs() == set()
    assert h._add_cleared_in_force() == []


def test_the_clears_in_force_follow_the_stack(fakes):
    h = _harness(_volume())
    ctrl = _controller(h, _df())
    _clear(h)
    seq = ctrl._curation_undo_stack.peek().extra["add_clear_seq"]
    assert h._add_cleared_in_force() == [(seq, (3, 4, 5))]
    h._reject_add()
    ctrl._undo_last_curation()
    assert h._add_cleared_in_force() == []
    assert h._add_clear_records == {}


def test_a_commit_that_raises_after_its_entry_landed_keeps_the_clear(fakes):
    """The controller pushed the entry and marked the rows, then its
    refresh raised: the Clear stands (putting the volume back would leave
    an entry that pastes over it), and Undo still takes it back."""
    vol = _volume()
    before = vol.copy()
    h = _harness(vol)
    ctrl = _controller(h, _df())

    def _boom():
        raise RuntimeError("refresh down")

    ctrl._refresh_curation_ui = _boom
    forgotten = []
    h._iso_live_ids = lambda: frozenset({3})
    h._iso_forget = lambda ids: forgotten.append(set(ids))
    _clear(h)
    assert len(ctrl._curation_undo_stack) == 1
    assert not np.isin(h.viewer.layers[-1].data, [3, 4, 5]).any()
    assert h._add_transient.startswith(AP.ADD_S1_CLEARED.format(n=3))
    assert forgotten == [{3}]
    assert h._add_clear_floor_now() == 5
    ctrl._refresh_curation_ui = lambda: None
    h._reject_add()
    ctrl._undo_last_curation()
    np.testing.assert_array_equal(h.viewer.layers[-1].data, before)


def test_commit_triage_empties_the_curation_undo(fakes, tmp_path):
    """Clear, close the box, Commit Triage: the Clear's undo entry is gone
    (its snapshot would paste objects back over garbage the commit zeroed),
    and the main Undo says there is nothing to undo."""
    import types

    vol = _volume()
    h = _harness(vol)
    ctrl = _controller(h, _df())
    _clear(h)
    h._reject_add()
    assert len(ctrl._curation_undo_stack) == 1
    after = h.viewer.layers[-1].data.copy()
    ctrl.working_dir = str(tmp_path)
    ctrl._qc_view_patch_name = "A.tif"
    ctrl._qc_table_patch = "A.tif"
    ctrl.ui = types.SimpleNamespace()
    ctrl._zero_labels_in_viewer = lambda ids: None
    ctrl._get_active_labels_path = lambda: ""
    ctrl.state_manager = types.SimpleNamespace(
        get_active_run_id=lambda: "default")
    assert ctrl._commit_triage_prepare() is not None
    assert len(ctrl._curation_undo_stack) == 0
    assert ctrl._live_add_clear_seqs() == set()
    logs = []
    ctrl._append_log = logs.append
    ctrl._undo_last_curation()
    assert "Nothing to undo." in logs
    np.testing.assert_array_equal(h.viewer.layers[-1].data, after)


def test_the_good_count_reads_active_good_and_curated_rows(fakes):
    vol = _volume()
    h = _harness(vol)
    ctrl = _controller(h, _df())
    assert ctrl._add_kept_tier_count([3, 4, 5]) == 2
    _clear(h)
    assert AP.ADD_CLEAR_CONFIRM_GOOD.format(g=2) in h.confirms[0]


def test_undo_all_says_a_clear_is_still_in_force(fakes):
    vol = _volume()
    h = _harness(vol)
    ctrl = _controller(h, _df())
    _clear(h)
    seq = next(iter(h._add_clear_records))
    assert h._add_session_clear_seqs == [seq]
    # an Accept in the same session, then Undo all takes only the Accept
    h._add_session = [{"ids": [77], "sizes": {77: 1}}]
    h._add_running = False
    h._add_review = True
    h._add_confirm = lambda text: True
    h._on_add_undo_session_cb = lambda n: n
    h._undo_add_all()
    assert AP.ADD_UNDO_DONE_CLEARED in (h.adv.last("status_text") or "")
    assert ctrl._live_add_clear_seqs() == {seq}
