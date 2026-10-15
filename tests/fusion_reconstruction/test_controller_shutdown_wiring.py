"""Shutdown wiring tests for FusionReconstructionController."""

from __future__ import annotations

import inspect


def test_controller_source_wires_viewer_close_to_shutdown():
    """Controller should subscribe to viewer close for thread cleanup."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    source = inspect.getsource(FusionReconstructionController)
    assert "viewer.events.close.connect" in source
    assert "_on_viewer_closing" in source
    assert "self.shutdown()" in source

