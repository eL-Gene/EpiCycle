"""Tests for ``MergeDialogState`` in ``napari_epicycle.annotator.merge_dialog``.

Pure-Python, Qt-free. Covers the state machine the merge dialog widget
delegates to. The Qt widget itself (``MergeDialog``) is tested via
end-to-end integration when the dialog is wired to the annotator;
this file targets the logic that determines: which labels are
isolated, when buttons enable/disable, how stage transitions work,
and whether snapshot rollback is wired correctly.

These are the "Pure Logic Tests" of the merge dialog.
"""
import pytest

from napari_epicycle.annotator.merge_dialog import MergeDialogState


# -------- builders --------------------------------------------------


def _three_neighbors():
    """Sample neighbor_info shape, as returned by get_neighbor_ids()."""
    return [
        {"label_id": 17, "contact_voxels": 240, "first_radius": 1},
        {"label_id": 23, "contact_voxels": 89, "first_radius": 1},
        {"label_id": 55, "contact_voxels": 18, "first_radius": 2},
    ]


def _three_metadata():
    return {
        17: {"area": 2340, "tier": "review", "centroid_z": 12},
        23: {"area": 891, "tier": "bad", "centroid_z": 14},
        55: {"area": 3102, "tier": "good", "centroid_z": 11},
    }


# -------- defaults / construction ----------------------------------


def test_default_state_in_selecting():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata=_three_metadata())
    assert s.stage == "selecting"
    assert s.checked_ids == set()
    assert s.snapshot is None
    assert s.absorbed_ids == []


def test_isolation_set_starts_with_keep_only():
    """On open, the dialog must isolate ONLY the keep label until the
    user checks neighbors. This is the spec's force-isolate-on-open."""
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    assert s.isolation_set == {42}


def test_buttons_disabled_initially():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    assert s.can_merge_selected() is False  # no neighbors checked
    assert s.can_accept() is False  # not merged


def test_empty_dialog_is_empty():
    """Per spec: empty neighbor_info renders placeholder; state still
    valid for navigation but can_merge_selected stays False."""
    s = MergeDialogState(keep_id=42, neighbor_info=[],
                         label_metadata={})
    assert s.is_empty() is True
    assert s.can_merge_selected() is False
    assert s.isolation_set == {42}


def test_non_empty_dialog_not_empty():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    assert s.is_empty() is False


# -------- toggle_check ---------------------------------------------


def test_toggle_check_adds_neighbor():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    assert 17 in s.checked_ids
    assert s.isolation_set == {42, 17}


def test_toggle_check_removes_neighbor():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.toggle_check(17)
    assert 17 not in s.checked_ids
    assert s.isolation_set == {42}


def test_toggle_check_keep_id_is_no_op():
    """The keep label cannot be toggled (it is implicitly always
    in the isolation set as the merge target)."""
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(42)
    assert 42 not in s.checked_ids
    assert s.isolation_set == {42}


def test_multiple_toggles_accumulate():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.toggle_check(23)
    assert s.checked_ids == {17, 23}
    assert s.isolation_set == {42, 17, 23}


# -------- can_merge_selected --------------------------------------


def test_can_merge_with_one_checked():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    assert s.can_merge_selected() is True


def test_can_merge_with_many_checked():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.toggle_check(23)
    s.toggle_check(55)
    assert s.can_merge_selected() is True


def test_cannot_merge_in_merged_stage():
    """Once merged, [Merge Selected] is disabled (one-shot per dialog)."""
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.transition_to_merged(snapshot=("roi", "data"), absorbed_ids=[17])
    assert s.can_merge_selected() is False


# -------- transition_to_merged + can_accept ------------------------


def test_transition_moves_to_merged_stage():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.toggle_check(23)
    s.transition_to_merged(snapshot=("roi", "old_data"), absorbed_ids=[17, 23])
    assert s.stage == "merged"
    assert s.snapshot == ("roi", "old_data")
    assert s.absorbed_ids == [17, 23]


def test_can_accept_only_after_merge():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    assert s.can_accept() is False
    s.toggle_check(17)
    assert s.can_accept() is False  # still selecting, just checked
    s.transition_to_merged(snapshot=None, absorbed_ids=[17])
    assert s.can_accept() is True


def test_double_transition_is_an_error():
    """Defensive: cannot move to merged twice. Caller must rollback
    via reset_to_selecting() before another transition."""
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.transition_to_merged(snapshot=None, absorbed_ids=[17])
    with pytest.raises(RuntimeError, match="only 'selecting' is valid"):
        s.transition_to_merged(snapshot=None, absorbed_ids=[17])


# -------- reset_to_selecting -------------------------------------


def test_reset_clears_merged_artifacts():
    """Used when Cancel-post-merge fires after the volume is rolled back.
    Stage returns to selecting, snapshot + absorbed_ids cleared."""
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.transition_to_merged(snapshot=("roi", "data"), absorbed_ids=[17])
    s.reset_to_selecting()
    assert s.stage == "selecting"
    assert s.snapshot is None
    assert s.absorbed_ids == []


def test_reset_preserves_checked_ids():
    """User's checkbox selections are kept after a rollback so they
    can adjust and try again without re-selecting from scratch."""
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.toggle_check(23)
    s.transition_to_merged(snapshot=None, absorbed_ids=[17, 23])
    s.reset_to_selecting()
    assert s.checked_ids == {17, 23}  # preserved


# -------- isolation_set semantics ---------------------------------


def test_isolation_set_updates_with_each_toggle():
    """The dialog's live-isolation feature relies on isolation_set
    reflecting current checked state. Each toggle must be visible."""
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    assert s.isolation_set == {42}
    s.toggle_check(17)
    assert s.isolation_set == {42, 17}
    s.toggle_check(55)
    assert s.isolation_set == {42, 17, 55}
    s.toggle_check(17)
    assert s.isolation_set == {42, 55}


def test_isolation_set_in_merged_stage():
    """After merge, the isolation set still reflects keep + checked.
    (Even though the volume now reflects the merge result.)"""
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.transition_to_merged(snapshot=None, absorbed_ids=[17])
    assert s.isolation_set == {42, 17}


# -------- absorbed_ids ordering ---------------------------------


def test_absorbed_ids_preserved_in_call_order():
    s = MergeDialogState(keep_id=42, neighbor_info=_three_neighbors(),
                         label_metadata={})
    s.toggle_check(17)
    s.toggle_check(55)
    # Pass the IDs in a specific order (e.g. sorted by caller).
    s.transition_to_merged(snapshot=None, absorbed_ids=[55, 17])
    assert s.absorbed_ids == [55, 17]
