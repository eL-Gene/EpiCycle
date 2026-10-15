"""Tests for the Tier 1b dual-init resolver.

Verifies ``_resolve_init_mode_for_bucket(bucket)`` correctly routes:
* The LATERAL bucket → primary widget values (the pre-Tier-1b shared widgets)
* The AXIAL bucket when Mode=Both → axial twin widget values
* The AXIAL bucket when Mode=Lateral/Axial OR when twin is empty →
  falls back to primary widget values (preserves the "shared init"
  behavior for single-orientation training).

These tests use a lightweight fake UI so no Qt event loop runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest


def _make_fake_ui(
    *,
    init_text: str,
    mode_text: str = "Fine-tune Lateral",
    source_lat: str = "",
    source_axial: str = "",
    pretrained_lat: str = "",
    pretrained_axial: str = "",
    import_lat: str = "",
    import_axial: str = "",
):
    """Build a minimal fake UI with the widgets the resolver reads."""
    class _Combo:
        def __init__(self, text):
            self._text = text
        def currentText(self):
            return self._text
        def currentData(self):
            # Mirror the production text/data split: the source combos store
            # the weights path as item-data; the resolver reads currentData().
            return self._text

    class _Line:
        def __init__(self, text):
            self._text = text
        def text(self):
            return self._text

    ui = SimpleNamespace()
    ui.refine_init_combo = _Combo(init_text)
    ui.refinement_mode_combo = _Combo(mode_text)
    ui.refinement_source_combo = _Combo(source_lat)
    ui.refinement_source_combo_axial = _Combo(source_axial)
    ui.refine_pretrained_combo = _Combo(pretrained_lat)
    ui.refine_pretrained_combo_axial = _Combo(pretrained_axial)
    ui.refine_import_path = _Line(import_lat)
    ui.refine_import_path_axial = _Line(import_axial)
    return ui


def _make_resolver(ui):
    """Wrap the controller's resolver method against a fake UI."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = SimpleNamespace(ui=ui)
    ctrl._resolve_init_mode_for_bucket = (
        FusionReconstructionController._resolve_init_mode_for_bucket.__get__(ctrl)
    )
    return ctrl._resolve_init_mode_for_bucket


# ---------------------------------------------------------------------------
# Lateral bucket — always reads primary widgets
# ---------------------------------------------------------------------------


class TestLateralBucket:

    def test_from_scratch_returns_scratch(self):
        ui = _make_fake_ui(init_text="From scratch")
        resolve = _make_resolver(ui)
        assert resolve("lateral") == ("scratch", None)

    def test_from_hitl_returns_custom_with_lateral_source(self):
        ui = _make_fake_ui(
            init_text="From HITL model",
            source_lat="/path/to/lateral_hitl",
            source_axial="/path/to/axial_hitl",
            mode_text="Fine-tune Both",
        )
        resolve = _make_resolver(ui)
        mode, path = resolve("lateral")
        assert mode == "custom"
        assert path == "/path/to/lateral_hitl"

    def test_from_pretrained_returns_cellpose_with_lateral_choice(self):
        ui = _make_fake_ui(
            init_text="From Cellpose pretrained",
            pretrained_lat="cyto3",
            pretrained_axial="cyto2",
            mode_text="Fine-tune Both",
        )
        resolve = _make_resolver(ui)
        assert resolve("lateral") == ("cellpose", "cyto3")

    def test_import_returns_custom_with_lateral_path(self):
        ui = _make_fake_ui(
            init_text="Import model",
            import_lat="/abs/lateral.pth",
            import_axial="/abs/axial.pth",
            mode_text="Fine-tune Both",
        )
        resolve = _make_resolver(ui)
        assert resolve("lateral") == ("custom", "/abs/lateral.pth")

    def test_empty_source_falls_back_to_scratch(self):
        ui = _make_fake_ui(init_text="From HITL model", source_lat="")
        resolve = _make_resolver(ui)
        assert resolve("lateral") == ("scratch", None)

    def test_empty_pretrained_falls_back_to_cyto3_default(self):
        ui = _make_fake_ui(
            init_text="From Cellpose pretrained", pretrained_lat="",
        )
        resolve = _make_resolver(ui)
        assert resolve("lateral") == ("cellpose", "cyto3")


# ---------------------------------------------------------------------------
# Axial bucket — reads twin when Mode=Both, otherwise primary
# ---------------------------------------------------------------------------


class TestAxialBucketBothMode:
    """When Mode=Both AND the axial twin has a value, use the twin."""

    def test_hitl_with_distinct_axial_source(self):
        ui = _make_fake_ui(
            init_text="From HITL model",
            mode_text="Fine-tune Both",
            source_lat="/path/to/lateral_hitl",
            source_axial="/path/to/axial_hitl",
        )
        resolve = _make_resolver(ui)
        mode, path = resolve("axial")
        assert mode == "custom"
        assert path == "/path/to/axial_hitl"

    def test_pretrained_with_distinct_axial_choice(self):
        ui = _make_fake_ui(
            init_text="From Cellpose pretrained",
            mode_text="Fine-tune Both",
            pretrained_lat="cyto3",
            pretrained_axial="nuclei",
        )
        resolve = _make_resolver(ui)
        assert resolve("axial") == ("cellpose", "nuclei")

    def test_import_with_distinct_axial_path(self):
        ui = _make_fake_ui(
            init_text="Import model",
            mode_text="Fine-tune Both",
            import_lat="/abs/lateral.pth",
            import_axial="/abs/axial.pth",
        )
        resolve = _make_resolver(ui)
        assert resolve("axial") == ("custom", "/abs/axial.pth")


class TestAxialBucketFallback:
    """Axial bucket falls back to primary widget when:
       - Mode is Lateral or Axial (not Both)
       - Mode=Both but axial twin widget is empty
    """

    def test_mode_lateral_axial_reads_primary(self):
        """When Mode=Fine-tune Lateral, the axial bucket isn't being
        trained — but if the resolver is called, return primary."""
        ui = _make_fake_ui(
            init_text="From HITL model",
            mode_text="Fine-tune Lateral",
            source_lat="/path/lateral",
            source_axial="/path/AXIAL_UNUSED",
        )
        resolve = _make_resolver(ui)
        # Mode is Lateral, axial twin shouldn't be consulted.
        mode, path = resolve("axial")
        assert path == "/path/lateral"

    def test_mode_axial_reads_primary(self):
        ui = _make_fake_ui(
            init_text="Import model",
            mode_text="Fine-tune Axial",
            import_lat="/abs/lateral.pth",
            import_axial="/abs/AXIAL_NOT_USED.pth",
        )
        resolve = _make_resolver(ui)
        mode, path = resolve("axial")
        # Mode=Axial → no twin consultation → primary widget value.
        assert path == "/abs/lateral.pth"

    def test_mode_both_but_axial_twin_empty_falls_back(self):
        """User selected Mode=Both but left axial twin empty —
        the resolver falls back to the primary, preserving the
        pre-Tier-1b "shared init" behavior."""
        ui = _make_fake_ui(
            init_text="From HITL model",
            mode_text="Fine-tune Both",
            source_lat="/path/shared",
            source_axial="",  # empty axial → fall back
        )
        resolve = _make_resolver(ui)
        mode, path = resolve("axial")
        assert path == "/path/shared"

    def test_mode_both_axial_pretrained_empty_falls_back(self):
        ui = _make_fake_ui(
            init_text="From Cellpose pretrained",
            mode_text="Fine-tune Both",
            pretrained_lat="cyto3",
            pretrained_axial="",
        )
        resolve = _make_resolver(ui)
        assert resolve("axial") == ("cellpose", "cyto3")


# ---------------------------------------------------------------------------
# Bucket validation
# ---------------------------------------------------------------------------


class TestBucketValidation:

    def test_invalid_bucket_raises(self):
        ui = _make_fake_ui(init_text="From scratch")
        resolve = _make_resolver(ui)
        with pytest.raises(ValueError, match="Invalid bucket"):
            resolve("spatial")

    def test_both_accepted(self):
        ui = _make_fake_ui(init_text="From scratch")
        resolve = _make_resolver(ui)
        # Should not raise:
        resolve("lateral")
        resolve("axial")


# ---------------------------------------------------------------------------
# Backward-compat shim
# ---------------------------------------------------------------------------


class TestBackwardCompatShim:
    """The old ``_resolve_init_mode()`` (no bucket arg) should still
    work and return the lateral resolution."""

    def test_shim_returns_lateral(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ui = _make_fake_ui(
            init_text="Import model",
            mode_text="Fine-tune Both",
            import_lat="/abs/lateral.pth",
            import_axial="/abs/axial.pth",
        )
        ctrl = SimpleNamespace(ui=ui)
        ctrl._resolve_init_mode_for_bucket = (
            FusionReconstructionController._resolve_init_mode_for_bucket.__get__(ctrl)
        )
        ctrl._resolve_init_mode = (
            FusionReconstructionController._resolve_init_mode.__get__(ctrl)
        )
        assert ctrl._resolve_init_mode() == ("custom", "/abs/lateral.pth")
