"""When the controller rewrites the label volume under the annotator (undo,
Commit Triage, auto-discard) the annotator's pre-draw snapshot is dropped,
so a later Clear Edits cannot paste stale voxels back."""
from __future__ import annotations

import types


def _controller(widget):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.viewer = types.SimpleNamespace(
        window=types.SimpleNamespace(
            _dock_widgets={"EpiCycle 3D Curation Annotator": widget}))
    return ctrl


def test_refresh_drops_the_annotators_draw_snapshot():
    calls = []
    widget = types.SimpleNamespace(
        _clear_draw_snapshot=lambda: calls.append("drop"),
        _refresh_label_data_in_subviewers=lambda: calls.append("refresh"))
    _controller(widget)._refresh_annotator_subviewers()
    assert calls == ["drop", "refresh"]


def test_refresh_works_with_an_annotator_that_has_no_snapshot_hook():
    calls = []
    widget = types.SimpleNamespace(
        _refresh_label_data_in_subviewers=lambda: calls.append("refresh"))
    _controller(widget)._refresh_annotator_subviewers()
    assert calls == ["refresh"]
