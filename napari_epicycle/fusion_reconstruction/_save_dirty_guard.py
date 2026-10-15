"""Pure helper for the async Save Curation dirty-flag backstop (B2).

See ``controller._on_save_curation_finished``.  Kept Qt-free so it is
unit-testable without importing napari/Qt.
"""


def should_clear_dirty_on_save_finish(dispatch_epoch: int, current_epoch: int) -> bool:
    """Return True iff no curation edit was committed since the save dispatched.

    The async Save snapshots the volume/df at dispatch time and writes that
    snapshot off-thread.  Every mask-edit commit bumps the controller's
    ``_curation_dirty_epoch``.  If the epoch advanced after dispatch, a new edit
    landed mid-save that is NOT in the snapshot being written — so the
    ``_curation_dirty`` flag must stay set so the next Save persists it.  Only
    clear the flag when the epoch is unchanged.
    """
    return current_epoch == dispatch_epoch
