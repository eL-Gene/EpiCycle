"""The rule for which cells stay isolated after an edit, on plain values."""
from __future__ import annotations

from napari_epicycle.annotator._isolation_bracket import (
    DEFAULT_PIECES_CAP,
    IsolationSnapshot,
    biggest_first,
    restore_target,
)


def _snap(mode, ids, visible=True, box=True):
    return IsolationSnapshot(mode, frozenset(ids), visible, box)


def test_isolation_off_gives_the_empty_set():
    s = _snap("off", [])
    assert restore_target(s, 5, (), (7, 8), pieces_cap=10) == frozenset()


def test_target_not_isolated_or_nothing_created_leaves_ids_unchanged():
    s = _snap("multi", [1, 2, 3])
    assert restore_target(s, 9, (), (7,), pieces_cap=10) == {1, 2, 3}
    assert restore_target(s, 2, set(), (), pieces_cap=10) == {1, 2, 3}


def test_a_consumed_cell_goes_even_when_nothing_replaces_it():
    s = _snap("multi", [1, 2, 3])
    assert restore_target(s, 2, {2}, (), pieces_cap=10) == {1, 3}
    only = _snap("single", [2])
    assert restore_target(only, 2, {2}, (), pieces_cap=10) == frozenset()
    # A consumed cell that is not the target goes too.
    assert restore_target(s, 9, {3}, (7,), pieces_cap=10) == {1, 2}


def test_a_surviving_original_stays_and_the_pieces_are_added():
    s = _snap("multi", [1, 2, 3])
    assert restore_target(s, 2, (), (7, 8), pieces_cap=10) == {1, 2, 3, 7, 8}


def test_a_consumed_original_is_replaced_by_its_pieces():
    s = _snap("single", [2])
    assert restore_target(s, 2, {2}, (7, 8, 9), pieces_cap=10) == {7, 8, 9}


def test_pieces_above_the_cap_collapse_to_the_first_created_id():
    s = _snap("single", [2])
    created = tuple(range(100, 100 + DEFAULT_PIECES_CAP))
    assert restore_target(s, 2, {2}, created, pieces_cap=DEFAULT_PIECES_CAP) == set(created)
    over = created + (999,)
    assert restore_target(s, 2, {2}, over, pieces_cap=DEFAULT_PIECES_CAP) == {100}
    # The cap is the caller's: the same pieces under a cap of 2.
    assert restore_target(s, 2, {2}, (7, 8, 9), pieces_cap=2) == {7}


def test_the_cap_keeps_the_piece_with_voxels_not_an_empty_id():
    s = _snap("single", [5])
    counts = {858: 0, 859: 0, 860: 12763}
    created = biggest_first([858, 859, 860], counts)
    assert created == (860,)
    got = restore_target(s, 5, (5,), created, pieces_cap=1)
    assert got == {860}


def test_biggest_first_orders_by_voxels_and_breaks_ties_by_id():
    counts = {1: 10, 2: 30, 3: 30, 4: 5}
    assert biggest_first([1, 2, 3, 4], counts) == (2, 3, 1, 4)
    assert biggest_first([], counts) == ()
