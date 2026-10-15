"""Guard: segmenting forces the main viewer to 2D (2026-05-28,
fix/step2-3d-render-freeze).

In 3D display mode the mask insert + visibility/refresh during a segment each
force a full label-VOLUME GPU re-render on the main thread (4-11s freeze,
confirmed via [SEGMENT-TIMING] main_ndisplay=3).  In 2D only one slice
redraws.  `_segment_current_slice_from_native` therefore calls
`apply_active_view_to_viewer(..., force_2d=True)` (which sets
`dims.ndisplay = 2`) before running segmentation.  This is independent of the
subviewer ViewerModels, so it cannot disturb subviewer sync.

Pure source inspection — no Qt/napari.
"""
from __future__ import annotations

import inspect

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


def test_segment_path_forces_main_viewer_to_2d():
    src = inspect.getsource(
        TrainingAnnotationController._segment_current_slice_from_native
    )
    # Both call sites (no-slice + explicit-slice) must force 2D.
    assert "apply_active_view_to_viewer(force_2d=True)" in src, (
        "the no-slice segment call must pass force_2d=True"
    )
    assert "slice_index=int(slice_index), force_2d=True" in src, (
        "the explicit-slice segment call must pass force_2d=True"
    )
    # No bare call left that would leave the viewer in 3D.
    assert "self.apply_active_view_to_viewer()" not in src, (
        "the no-arg apply_active_view_to_viewer() call in the segment path "
        "must pass force_2d=True"
    )


def test_apply_active_view_to_viewer_force_2d_sets_ndisplay():
    """The reused mechanism: force_2d=True must set dims.ndisplay = 2."""
    src = inspect.getsource(
        TrainingAnnotationController.apply_active_view_to_viewer
    )
    assert "force_2d" in src
    assert "ndisplay = 2" in src, (
        "apply_active_view_to_viewer's force_2d path must set dims.ndisplay = 2"
    )
