"""Tests for `napari_epicycle.annotator.refinement_state.RefinementOpState`.

Pure-Python, Qt-free. The dataclass mirrors per-op flags used by
`Curation3DAnnotator`; these tests only exercise the dataclass + its
predicates. End-to-end integration with the annotator is covered by the
existing curation_3d test files in Phase 2 of the migration.

Coverage targets:
- Default state is clean and idle.
- `any_op_active` flips correctly.
- `can_start_interp` / `can_start_split` / `can_start_smooth` encode
  the cross-op gating rules (which ops may start while another is active).
- `is_smooth_chaining` returns True only on same-label re-entry.
- `can_open_merge_dialog` blocks during any active op.
- Sub-mode flags (interp_refine_mode, split_fragment_refine) relax the
  smooth gate.
- Pattern-compliance test from doc : every <op>_active field has a
  matching can_start_<op> predicate.
"""
from dataclasses import fields

import pytest

from napari_epicycle.annotator.refinement_state import RefinementOpState


# ----------------------- default / idle ------------------------------


def test_default_state_is_idle():
    s = RefinementOpState()
    assert s.interp_active is False
    assert s.interp_refine_mode is False
    assert s.split_active is False
    assert s.split_previewed is False
    assert s.split_fragment_refine is False
    assert s.smooth_active is False
    assert s.smooth_chain_count == 0
    assert s.batch_refine_active is False
    assert s.any_op_active is False


def test_default_predicates_allow_everything():
    s = RefinementOpState()
    assert s.can_start_interp() is True
    assert s.can_start_split() is True
    assert s.can_start_smooth(label_id=42) is True
    assert s.can_open_merge_dialog() is True
    assert s.can_start_batch_refine() is True
    assert s.is_smooth_chaining(label_id=42) is False


# ----------------------- any_op_active -------------------------------


@pytest.mark.parametrize(
    "field_name",
    ["interp_active", "split_active", "smooth_active", "batch_refine_active"],
)
def test_any_op_active_flips_with_each_active_field(field_name):
    s = RefinementOpState()
    setattr(s, field_name, True)
    assert s.any_op_active is True


def test_any_op_active_false_when_only_submode_flag_set():
    """Sub-mode flags alone (without their parent op active) shouldn't
    register as 'any op active'. This is defensive — sub-mode flags
    should only ever be True while their parent op is active."""
    s = RefinementOpState()
    s.interp_refine_mode = True
    s.split_fragment_refine = True
    assert s.any_op_active is False


# ----------------------- can_start_interp ----------------------------


def test_can_start_interp_blocked_by_smooth():
    s = RefinementOpState(smooth_active=True, smooth_label_id=1)
    assert s.can_start_interp() is False


def test_can_start_interp_blocked_by_split():
    s = RefinementOpState(split_active=True)
    assert s.can_start_interp() is False


def test_can_start_interp_allowed_during_split_fragment_refine():
    """Mirrors curation_3d_widget.py:3101-3102 — split blocks interp
    UNLESS split_fragment_refine sub-mode is active."""
    s = RefinementOpState(split_active=True, split_fragment_refine=True)
    assert s.can_start_interp() is True


def test_can_start_interp_blocked_by_self():
    """If interp is already active, starting interp again is a no-op
    in the existing code; we mirror that by treating the predicate as
    permissive (re-entry is detected by the caller via interp_active)."""
    s = RefinementOpState(interp_active=True)
    # Note: the existing code's interp re-entry guards happen at the
    # call site, not at can_start_interp. The predicate only encodes
    # cross-op rules. This is intentional.
    assert s.can_start_interp() is True


# ----------------------- can_start_split -----------------------------


@pytest.mark.parametrize(
    "field_name",
    ["interp_active", "smooth_active", "split_active"],
)
def test_can_start_split_blocked_by_any_active_op(field_name):
    s = RefinementOpState()
    setattr(s, field_name, True)
    assert s.can_start_split() is False


def test_can_start_split_allowed_when_clean():
    s = RefinementOpState()
    assert s.can_start_split() is True


# ----------------------- can_start_smooth ----------------------------


def test_can_start_smooth_chains_on_same_label():
    """The same-label smooth re-entry should be allowed (chain pattern).
    Mirrors curation_3d_widget.py:4473-4490."""
    s = RefinementOpState(smooth_active=True, smooth_label_id=7)
    assert s.can_start_smooth(label_id=7) is True


def test_can_start_smooth_blocked_on_different_label():
    """Smooth on a different label while smooth is already active
    should be blocked. Mirrors :4475-4477."""
    s = RefinementOpState(smooth_active=True, smooth_label_id=7)
    assert s.can_start_smooth(label_id=8) is False


def test_can_start_smooth_blocked_by_split():
    s = RefinementOpState(split_active=True)
    assert s.can_start_smooth(label_id=1) is False


def test_can_start_smooth_allowed_during_fragment_refine():
    """Smooth can chain into the split's fragment-refine sub-mode.
    Mirrors :4438-4442 + the split fragment-refine branch."""
    s = RefinementOpState(split_active=True, split_fragment_refine=True)
    assert s.can_start_smooth(label_id=1) is True


def test_can_start_smooth_allowed_during_interp():
    """The existing code allows smooth during interp (smooth_active
    only blocks if it's a DIFFERENT smooth on a different label).
    Interp by itself does not block smooth — see :4438-4442 which
    only checks _split_running, not _interp_running."""
    s = RefinementOpState(interp_active=True)
    assert s.can_start_smooth(label_id=1) is True


# ----------------------- is_smooth_chaining --------------------------


def test_is_smooth_chaining_true_on_same_label():
    s = RefinementOpState(smooth_active=True, smooth_label_id=42)
    assert s.is_smooth_chaining(label_id=42) is True


def test_is_smooth_chaining_false_on_different_label():
    s = RefinementOpState(smooth_active=True, smooth_label_id=42)
    assert s.is_smooth_chaining(label_id=43) is False


def test_is_smooth_chaining_false_when_smooth_idle():
    s = RefinementOpState(smooth_active=False, smooth_label_id=None)
    assert s.is_smooth_chaining(label_id=42) is False


# ----------------------- can_open_merge_dialog -----------------------


@pytest.mark.parametrize(
    "field_name",
    ["interp_active", "split_active", "smooth_active", "batch_refine_active"],
)
def test_can_open_merge_dialog_blocked_by_any_op(field_name):
    s = RefinementOpState()
    setattr(s, field_name, True)
    assert s.can_open_merge_dialog() is False


def test_can_open_merge_dialog_allowed_when_clean():
    s = RefinementOpState()
    assert s.can_open_merge_dialog() is True


def test_can_open_merge_dialog_allowed_with_only_submode_flags():
    """Sub-mode flags alone shouldn't block merge — they only encode
    sub-state of their parent op, and a sub-mode flag with a False
    parent flag is itself a transient illegal state we're defensive
    against."""
    s = RefinementOpState(interp_refine_mode=True, split_fragment_refine=True)
    assert s.can_open_merge_dialog() is True


# ----------------------- pattern compliance --------------------------


def test_every_active_field_has_can_start_predicate():
    """Pattern-compliance guardrail.

    Every `<op>_active: bool` field on RefinementOpState must have a
    corresponding `can_start_<op>` method. Catches forgotten predicate
    additions when a new op is registered.

    Note: triage actions and merge are intentionally NOT modeled as
    `*_active` fields — they're not refinement ops in the state machine
    sense. Merge has `can_open_merge_dialog` but no `merge_active` (yet
    — Phase 4 of the migration adds it).
    """
    state = RefinementOpState()
    active_fields = [
        f for f in fields(state)
        if f.name.endswith("_active") and f.type in (bool, "bool")
    ]
    assert len(active_fields) >= 3, (
        "Expected at least 3 active fields (interp, split, smooth); "
        f"found {len(active_fields)}: {[f.name for f in active_fields]}"
    )
    for f in active_fields:
        op_name = f.name[: -len("_active")]
        predicate_name = f"can_start_{op_name}"
        assert hasattr(state, predicate_name), (
            f"{f.name} exists but {predicate_name} does not. "
            f"Every op family "
            f"needs a can_start_<op> predicate."
        )


# ----------------------- chain count ----------------------------------


def test_smooth_chain_count_is_a_separate_field():
    """smooth_chain_count is an int,
    not a bool. It's incremented by the chain re-entry path at
    :4488 and used for diagnostics / future undo-history."""
    s = RefinementOpState()
    assert s.smooth_chain_count == 0
    s.smooth_chain_count = 3
    assert s.smooth_chain_count == 3


# ----------------------- batch refine (2026-06-05) -------------------


def test_batch_refine_active_registers_as_any_op_active():
    s = RefinementOpState(batch_refine_active=True)
    assert s.any_op_active is True


@pytest.mark.parametrize("predicate", ["can_start_interp", "can_start_split"])
def test_staged_batch_blocks_other_ops_starting(predicate):
    s = RefinementOpState(batch_refine_active=True)
    assert getattr(s, predicate)() is False


def test_staged_batch_blocks_smooth_starting():
    s = RefinementOpState(batch_refine_active=True)
    assert s.can_start_smooth(label_id=1) is False


def test_staged_batch_blocks_merge_dialog():
    s = RefinementOpState(batch_refine_active=True)
    assert s.can_open_merge_dialog() is False


def test_staged_batch_blocks_destructive_label_op():
    s = RefinementOpState(batch_refine_active=True)
    assert s.can_start_destructive_label_op() is False


@pytest.mark.parametrize(
    "field_name",
    ["interp_active", "split_active", "smooth_active", "batch_refine_active"],
)
def test_can_start_batch_refine_blocked_by_any_op(field_name):
    s = RefinementOpState()
    setattr(s, field_name, True)
    assert s.can_start_batch_refine() is False


def test_can_start_batch_refine_blocked_by_draw_snapshot():
    s = RefinementOpState(draw_snapshot_present=True)
    assert s.can_start_batch_refine() is False


def test_can_start_batch_refine_blocked_by_open_merge_dialog():
    s = RefinementOpState(merge_dialog_open=True)
    assert s.can_start_batch_refine() is False


def test_can_start_batch_refine_allowed_when_clean():
    s = RefinementOpState()
    assert s.can_start_batch_refine() is True


def test_can_start_batch_refine_allowed_with_only_submode_flags():
    """Sub-mode flags alone (parent op idle) shouldn't block batch —
    consistent with can_open_merge_dialog's defensive behaviour."""
    s = RefinementOpState(interp_refine_mode=True, split_fragment_refine=True)
    assert s.can_start_batch_refine() is True
