"""B2: unit tests for the async-save dirty-flag backstop helper.

Qt-free — exercises the pure decision logic that stops
``_on_save_curation_finished`` from clearing ``_curation_dirty`` when a mask
edit was committed DURING the async save (so the next Save persists it).
"""
from napari_epicycle.fusion_reconstruction._save_dirty_guard import (
    should_clear_dirty_on_save_finish,
)


def test_clears_when_no_edit_during_save():
    # Epoch unchanged between dispatch and finish -> safe to clear dirty.
    assert should_clear_dirty_on_save_finish(3, 3) is True


def test_keeps_dirty_when_one_edit_committed_during_save():
    # A mask edit after dispatch bumps the epoch -> keep dirty for the next Save.
    assert should_clear_dirty_on_save_finish(3, 4) is False


def test_keeps_dirty_when_multiple_edits_during_save():
    assert should_clear_dirty_on_save_finish(0, 5) is False


def test_zero_epochs_clear():
    assert should_clear_dirty_on_save_finish(0, 0) is True
