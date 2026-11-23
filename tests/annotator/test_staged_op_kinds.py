"""One answer to "is an edit open?": the set of open kinds."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K
from napari_epicycle.annotator.refinement_state import (
    ISOLATION_BLOCKING,
    RefinementOpState,
    StagedOp,
)

_FLAGS = dict(
    _split_running=False, _reshape_running=False, _smooth_running=False,
    _interp_running=False, _interp_snapshot=None, _interp_refine_mode=False,
    _add_running=False, _batch_running=False, _batch_staged=False,
    _batch_refine_dialog=None, _merge_dialog=None, _draw_snapshot=None,
)


def _fake(**flags):
    f = SimpleNamespace(_refine_state=RefinementOpState(), **_FLAGS)
    f.__dict__.update(flags)
    f._sync_refine_state = lambda: K._sync_refine_state(f)
    f.staged_kinds = lambda: K.staged_kinds(f)
    return f


@pytest.mark.parametrize("flags,kind", [
    ({"_batch_running": True}, StagedOp.BATCH),
    ({"_batch_staged": True}, StagedOp.BATCH),
    ({"_split_running": True}, StagedOp.RESHAPE),
    ({"_interp_running": True}, StagedOp.INTERPOLATE),
    ({"_interp_snapshot": object()}, StagedOp.INTERPOLATE),
    ({"_interp_refine_mode": True}, StagedOp.INTERPOLATE),
    ({"_add_running": True}, StagedOp.ADD),
    ({"_smooth_running": True}, StagedOp.REFINE),
    ({"_merge_dialog": object()}, StagedOp.MERGE),
])
def test_each_flag_maps_to_its_kind(flags, kind):
    assert K.staged_kinds(_fake(**flags)) == frozenset({kind})


def test_idle_is_empty_and_an_idle_batch_dialog_is_not_a_kind():
    f = _fake()
    assert K.staged_kinds(f) == frozenset()
    f._batch_refine_dialog = object()
    assert K.staged_kinds(f) == frozenset()
    # the dialog still closes the start gates
    assert f._refine_state.any_op_active


def test_simultaneous_ops_keep_every_kind_and_priority_picks_one():
    f = _fake(_split_running=True, _smooth_running=True, _batch_staged=True)
    kinds = K.staged_kinds(f)
    assert kinds == {StagedOp.BATCH, StagedOp.RESHAPE, StagedOp.REFINE}
    assert f._refine_state.staged_op_kind() is StagedOp.BATCH


def test_merge_alone_does_not_count_as_any_op_active():
    f = _fake(_merge_dialog=object())
    K.staged_kinds(f)
    assert StagedOp.MERGE in K.staged_kinds(f)
    assert not f._refine_state.any_op_active


def test_add_counts_as_any_op_active():
    f = _fake(_add_running=True)
    K.staged_kinds(f)
    assert f._refine_state.any_op_active


def test_isolation_blocking_follows_per_flag():
    assert ISOLATION_BLOCKING == {
        StagedOp.RESHAPE, StagedOp.REFINE, StagedOp.INTERPOLATE}
    # a running interpolation blocks; a reshape-worker flag alone does not
    assert K.staged_kinds(_fake(_interp_running=True)) & ISOLATION_BLOCKING
    assert not (K.staged_kinds(_fake(_reshape_running=True)) & ISOLATION_BLOCKING)
    assert not (K.staged_kinds(_fake(_add_running=True)) & ISOLATION_BLOCKING)


def test_navigator_predicate_reads_the_blocking_set():
    f = _fake(_interp_running=True)
    f.staged_kinds = lambda: K.staged_kinds(f)
    assert K._is_op_active_for_navigator(f) is True
    assert K._is_op_active_for_navigator(_fake(_add_running=True)) is False


def test_old_predicate_is_gone():
    assert not hasattr(K, "_refinement_op_active")


@pytest.mark.parametrize("flags", [
    {"_split_running": True}, {"_smooth_running": True},
    {"_interp_snapshot": object()}, {"_interp_refine_mode": True},
    {"_batch_staged": True}, {"_batch_running": True},
    {"_add_running": True, "_add_busy": True},
])
def test_uncommitted_work_implies_a_non_empty_set(flags):
    f = _fake(_operation_slots=[], _active_labels_name="",
              _split_label_id=1, _split_fragment_refine=False,
              _smooth_label_id=1, _interp_label_id=1, _split_previewed=False,
              _add_busy=False)
    f.__dict__.update(flags)
    for name in ("_reject_split", "_reject_smooth", "_reject_interpolation",
                 "_discard_batch_refine", "_reject_add"):
        setattr(f, name, lambda *a, **k: None)
    f._add_has_work = lambda: K._add_has_work(f)
    f._add_close_description = lambda: "add"
    K._register_operation_slots(f)
    has_work, _why = K._has_uncommitted_work(f, include_edited_slices=False)
    assert has_work
    assert K.staged_kinds(f)
