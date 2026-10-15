"""Regression: `_accept_interpolation` must call `_sync_all_views_after_operation`.

User-reported 2026-05-03 bug:
  After Accept on a slice interpolation, the main 3D viewer correctly
  shows the interpolated mask but the subviewer panels go blank
  (image-only, label disappeared).  Other refinement ops (split,
  smooth, merge, reject-interp) all call `_sync_all_views_after_operation`
  to push the shared_data change to subviewers + isolation proxy.
  `_accept_interpolation` was the lone exception (a known, pre-existing bug).

Fix: invoke `_sync_all_views_after_operation(roi=undo_roi,
label_id=label_id, skip_3d_refresh=...)` inside `_accept_interpolation`,
matching the pattern from split's accept handler.

This test is source-inspection — verifies the call exists.  A
behavioural test would require a full napari+Qt stack (the operation
mutates shared_data, refreshes subviewers, rebuilds the isolation
proxy).  Source inspection is sufficient because the symptom is
"call missing", not "call malformed".
"""

from __future__ import annotations

import inspect


def test_accept_interpolation_calls_sync():
    """`_accept_interpolation` must invoke `_sync_all_views_after_operation`."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    source = inspect.getsource(Curation3DAnnotator._accept_interpolation)
    assert "_sync_all_views_after_operation" in source, (
        "_accept_interpolation must call _sync_all_views_after_operation "
        "to keep subviewers + isolation proxy in sync after Accept. "
        "Without it the subviewers go blank — pre-2026-05-03 bug."
    )


def test_accept_interpolation_passes_label_id_to_sync():
    """Sync call needs label_id so the isolation proxy can rebuild."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    source = inspect.getsource(Curation3DAnnotator._accept_interpolation)
    # Find the sync invocation block.  Must mention label_id so the
    # isolation proxy rebuild path triggers when isolation is active.
    sync_idx = source.find("_sync_all_views_after_operation")
    assert sync_idx >= 0
    # Inspect a window around the call site.
    window = source[sync_idx:sync_idx + 400]
    assert "label_id" in window, (
        "Sync call should pass label_id so the isolation proxy can "
        "rebuild when isolation is active during Accept Interpolation."
    )


def test_reject_interpolation_already_calls_sync():
    """Sanity: confirm the existing pattern in reject is preserved."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    source = inspect.getsource(Curation3DAnnotator._reject_interpolation)
    assert "_sync_all_views_after_operation" in source, (
        "_reject_interpolation has historically called the sync helper. "
        "If this assertion fails, the symmetric pattern between Accept "
        "and Reject has been broken — investigate."
    )
