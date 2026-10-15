"""Tests for the race-condition follow-up fixes (P0 #1 + P0 #6).

Two latent bugs surfaced in the 2026-05-10 audit; downsample-on-load
makes them more likely to fire:

  P0 #1: ``_isolation_label_set`` cache was not invalidated after
         split-accept.  Subsequent isolation calls fast-skipped at
         ``_apply_multi_isolation_colormap`` line ~2143 ``if frozen ==
         prev_cache: return`` with a cached set containing labels that
         no longer exist in the volume.
  P0 #6: Patch switch path (``_load_selected_layers``) did not
         explicitly clear refinement snapshots.  An Accept/Reject on
         the new patch could write the OLD patch's snapshot data into
         the NEW patch's volume → silent corruption.

Tests use the bare-instance pattern + ``types.SimpleNamespace`` for
helpers — same pattern as other curation_3d_widget tests in this
repo.  No Qt instantiation required for the state-clearing
assertions.
"""
from __future__ import annotations

import inspect
import types

import pytest


# =============================================================================
# P0 #1 — _isolation_label_set cleared in _reset_split_state
# =============================================================================


class TestP01_ResetSplitStateClearsIsolationCache:
    """After split-accept (which goes through ``_reset_split_state``),
    the multi-isolation cache must be invalidated so the next
    ``_apply_multi_isolation_colormap`` doesn't fast-skip with stale
    label IDs."""

    def _make_bare_annotator(self, isolation_set=None):
        """Bare Curation3DAnnotator with the minimum state needed for
        ``_reset_split_state`` to run end-to-end without Qt."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
        ann._split_snapshot = object()  # any non-None sentinel
        ann._split_roi = (slice(0, 1),)
        ann._split_label_id = 47
        ann._split_running = True
        ann._split_previewed = True
        ann._split_seed_ids = [100, 101]
        ann._split_isolate_was_on = True
        ann._split_main_visible = True
        ann._refine_pre_preview = None
        ann._split_fragment_refine = False
        ann._split_fragment_ids = []
        ann._fragment_refine_roi = None
        ann._detected_peaks = None
        ann._auto_detect_active = False
        ann._seed_id_base = 0
        # The cache field that should be invalidated.
        ann._isolation_label_set = (
            frozenset(isolation_set) if isolation_set else None
        )
        # _refresh_merge_button_state is called inside _reset_split_state;
        # stub it to a no-op so we don't need merge_button machinery.
        ann._refresh_merge_button_state = lambda: None
        return ann

    def test_isolation_label_set_cleared_after_reset(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator(isolation_set={12, 15, 47})
        # Pre-condition: cache contains the about-to-be-destroyed label
        assert 47 in ann._isolation_label_set

        # Run _reset_split_state — simulates the cleanup at the end of
        # _accept_split (or _reject_split / error paths).
        Curation3DAnnotator._reset_split_state(ann)

        assert ann._isolation_label_set is None, (
            "Multi-isolation cache must be invalidated after split-accept "
            "so subsequent _apply_multi_isolation_colormap does NOT "
            "fast-skip with a stale label set containing IDs that no "
            "longer exist in the volume."
        )

    def test_isolation_cleared_even_when_already_none(self):
        """Idempotent: clearing None to None is safe."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator(isolation_set=None)
        Curation3DAnnotator._reset_split_state(ann)
        assert ann._isolation_label_set is None

    def test_other_split_state_still_cleared(self):
        """Regression guard: the new cache-clear line must not have
        accidentally removed any of the existing state resets."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator(isolation_set={47})
        Curation3DAnnotator._reset_split_state(ann)
        assert ann._split_snapshot is None
        assert ann._split_roi is None
        assert ann._split_label_id is None
        assert ann._split_running is False
        assert ann._split_previewed is False
        assert ann._split_seed_ids == []
        assert ann._split_isolate_was_on is False
        assert ann._split_main_visible is True
        assert ann._refine_pre_preview is None
        assert ann._split_fragment_refine is False
        assert ann._split_fragment_ids == []
        assert ann._fragment_refine_roi is None
        assert ann._detected_peaks is None
        assert ann._auto_detect_active is False
        assert ann._seed_id_base == 0


# =============================================================================
# P0 #6 — refinement snapshots cleared on patch switch
# =============================================================================


class TestP06_PatchSwitchClearsSnapshots:
    """The ``_load_selected_layers`` override in ``Curation3DAnnotator``
    runs on every patch switch.  Refinement snapshots
    (``_interp_snapshot``, ``_split_snapshot``, ``_smooth_snapshot``,
    ``_draw_snapshot``) must be explicitly cleared so an Accept/Reject
    on the new patch can't write stale snapshot data from the old
    patch into the new patch's shared volume.

    These tests inspect the source to confirm the clears are present
    in the patch-switch code path — full integration testing is
    covered by manual user testing in napari."""

    def test_source_contains_snapshot_clears(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
        # Each refinement snapshot must be explicitly set to None.
        # The format checked is "self._<name>_snapshot = None" which
        # is the canonical assignment used elsewhere in the code.
        for snap_attr in (
            "_interp_snapshot",
            "_split_snapshot",
            "_smooth_snapshot",
            "_draw_snapshot",
        ):
            assert f"self.{snap_attr} = None" in src, (
                f"_load_selected_layers must explicitly clear "
                f"self.{snap_attr} on patch switch — otherwise a "
                f"stale snapshot from the previous patch could write "
                f"into the new patch's volume on Accept/Reject."
            )

    def test_source_contains_roi_clears(self):
        """The ROI fields that pair with snapshots (defining where in
        the volume to write) must also clear, otherwise a stale ROI
        could target wrong coordinates on the new patch."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
        for roi_attr in ("_interp_roi", "_split_roi"):
            assert f"self.{roi_attr} = None" in src, (
                f"_load_selected_layers must explicitly clear "
                f"self.{roi_attr} on patch switch."
            )


# =============================================================================
# Reference: where the fix is documented in code
# =============================================================================


class TestFixDocumentation:
    """Make sure the rationale stays alive in the source — future
    refactors that touch these reset paths should see the audit
    reference and understand why the clears are there."""

    def test_reset_split_state_documents_p01(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._reset_split_state)
        assert "P0 #1" in src

    def test_load_selected_layers_documents_p06(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
        assert "P0 #6" in src
