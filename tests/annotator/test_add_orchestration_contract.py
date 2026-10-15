"""Contract tests for the de-novo Add-mode orchestration + wiring.

Source-inspection only (Qt-free). These pin the safety-critical invariants
that can't otherwise be exercised without a running napari display:
background-only safe-write, lock fired on every exit path, patch-switch
force-reject, and the controller lock/guard membership.
"""

from __future__ import annotations

import inspect


def _ann():
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    return Curation3DAnnotator


def _ctrl():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    return FusionReconstructionController


class TestSafeWrite:
    def test_accept_is_background_only(self):
        """Accept must never overwrite an existing label — background-only."""
        src = inspect.getsource(_ann()._accept_add)
        assert "(result > 0) & (target == 0)" in src, (
            "Add accept must safe-write only into empty voxels"
        )

    def test_accept_passes_undo_payload_to_controller(self):
        src = inspect.getsource(_ann()._accept_add)
        # The pre-commit crop is captured + handed to the controller so the
        # add_denovo undo entry can restore it.
        assert "shared[roi].copy()" in src
        assert "old_roi_data=" in src

    def test_edge_cleanup_never_deletes_only_object(self):
        src = inspect.getsource(_ann()._apply_add_edge_cleanup)
        # The "would delete everything → keep all" intercept.
        assert "len(flagged) < len(ids)" in src


class TestLockLifecycle:
    def test_start_fires_lock_true(self):
        src = inspect.getsource(_ann()._start_add)
        assert "_fire_add_mode_lock(True)" in src

    def test_every_exit_fires_lock_false(self):
        for name in ("_accept_add", "_reject_add"):
            src = inspect.getsource(getattr(_ann(), name))
            assert "_fire_add_mode_lock(False)" in src, f"{name} must release lock"

    def test_fire_lock_guarded(self):
        src = inspect.getsource(_ann()._fire_add_mode_lock)
        assert '"_add_mode_lock_cb"' in src
        assert "if cb is None" in src


class TestTeardownAndSwitch:
    def test_patch_switch_force_rejects_add(self):
        src = inspect.getsource(_ann()._load_selected_layers)
        assert "_reject_add()" in src

    def test_close_cleanup_tears_down_add(self):
        src = inspect.getsource(_ann()._close_cleanup)
        assert "_reject_add" in src or "_remove_add_candidate" in src

    def test_candidate_teardown_hides_before_remove(self):
        # The candidate is a napari Labels layer → hide the vispy visual before
        # removing it from the viewer (frees the GPU texture safely).
        src = inspect.getsource(_ann()._remove_add_candidate)
        assert "visible = False" in src, "candidate must hide before remove"

    def test_box_teardown_detaches_overlay(self):
        # The box is a vispy Line OVERLAY (not a napari layer), so the safe
        # teardown is detaching it from the scene (parent = None via destroy()),
        # NOT layer hide-before-remove.  It must not leave a dangling ref.
        src = inspect.getsource(_ann()._remove_add_box_shapes)
        assert "destroy()" in src, "box overlay must be destroyed on teardown"
        assert "_add_box_overlay = None" in src, "box overlay ref must be nulled"


class TestEdgeCleanupIs3D:
    """Verify the edge-touch predicate is fully 3D — it must flag/drop pieces
    that touch the AXIAL (Z top/bottom) walls, not only the lateral (X/Y) walls.
    _apply_add_edge_cleanup now delegates to _add_edge_flags, so the test
    receiver needs both methods (neither reads other `self` attributes).
    """

    @staticmethod
    def _host():
        A = _ann()

        class _H:
            _add_edge_flags = A._add_edge_flags
            _apply_add_edge_cleanup = A._apply_add_edge_cleanup

        return _H()

    @staticmethod
    def _clean(labels, margin=3):
        return TestEdgeCleanupIs3D._host()._apply_add_edge_cleanup(labels, margin)

    def _volume(self):
        import numpy as np
        vol = np.zeros((12, 20, 20), dtype=np.int32)
        vol[5:8, 9:12, 9:12] = 1   # interior — must be KEPT
        vol[0:2, 9:12, 9:12] = 2   # touches Z-TOP (axial) — must be DROPPED
        vol[5:8, 9:12, 0:2] = 3    # touches an X wall (lateral) — must be DROPPED
        return vol

    def test_axial_z_top_piece_is_dropped(self):
        out, dropped = self._clean(self._volume())
        assert 2 in dropped, "a piece touching the Z-top wall must be dropped (3D)"
        assert not (out == 2).any()

    def test_lateral_piece_is_dropped(self):
        out, dropped = self._clean(self._volume())
        assert 3 in dropped, "a piece touching an X wall must be dropped"

    def test_interior_piece_is_kept(self):
        out, dropped = self._clean(self._volume())
        assert 1 not in dropped and (out == 1).any(), "interior piece must survive"

    def test_z_bottom_wall_is_also_covered(self):
        import numpy as np
        vol = np.zeros((12, 20, 20), dtype=np.int32)
        vol[5:8, 9:12, 9:12] = 1        # interior — KEPT
        vol[10:12, 9:12, 9:12] = 2      # touches Z-BOTTOM (mask[-1]) — DROPPED
        out, dropped = self._clean(vol)
        assert 2 in dropped, "a piece touching the Z-bottom wall must be dropped"

    def test_never_deletes_the_only_object(self):
        import numpy as np
        vol = np.zeros((12, 20, 20), dtype=np.int32)
        vol[0:2, 0:2, 0:2] = 1          # single piece, touches walls
        out, dropped = self._clean(vol)
        assert dropped == [] and (out == 1).any(), "must not delete the only object"


class TestReviewPieceSizes:
    """The REVIEW list surfaces per-piece voxel sizes so fragments are visible."""

    def test_accept_records_piece_sizes(self):
        src = inspect.getsource(_ann()._accept_add)
        assert "size_map" in src, "_accept_add must compute per-piece sizes"
        assert "piece_sizes=all_sizes" in src, "_accept_add must forward piece_sizes"
        # Sizes come from the ROI-sized committed mask (bounded, no full scan).
        assert "result[write_mask]" in src

    def test_widget_render_accepts_and_flags_sizes(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        params = set(
            inspect.signature(AdvancedFeaturesWidget.update_add_ui_state).parameters
        )
        assert "piece_sizes" in params
        # Add rebuild 4.1: rows read "Object {id}" + the size in mono; the
        # small rule moved to add_preview.small_piece_ids (pushed as
        # candidate_small_ids, the widget stops computing its own).
        src = inspect.getsource(AdvancedFeaturesWidget.update_add_ui_state)
        assert '"  ·  {:,}"' in src, "list items must show a voxel count"
        from packages.epicycle_annotator import add_preview

        ssrc = inspect.getsource(add_preview.small_piece_ids)
        assert "cut" in ssrc and "median" in ssrc, "must flag small fragments"


class TestReviewBatchRefine:
    """REVIEW can hand this session's pieces to the existing Batch Refine dialog."""

    def test_refine_all_hands_the_session_to_batch_refine(self):
        # 'Refine added…' was removed from REVIEW in 2026-07; BATCH_RECIPE_
        # BRICK brings the hand-off back as "Refine all ▸" (Add now
        # makes MANY labels at once), in the existing piece row, wired to
        # the explicit-target open.
        from napari_epicycle.annotator.advanced_features_widget import (
            ADD_REFINE_ALL_BTN,
            AdvancedFeaturesWidget,
        )

        wsrc = inspect.getsource(AdvancedFeaturesWidget.set_add_callbacks)
        assert '_wire("add_batch_refine_btn", on_batch_refine)' in wsrc
        bsrc = inspect.getsource(AdvancedFeaturesWidget._build_add_denovo_section)
        assert "self.add_batch_refine_btn = QPushButton(ADD_REFINE_ALL_BTN)" in bsrc
        assert ADD_REFINE_ALL_BTN == "Refine all ▸"
        asrc = inspect.getsource(_ann()._open_advanced_features)
        assert "on_batch_refine=self._batch_refine_add_pieces" in asrc
        hsrc = inspect.getsource(_ann()._batch_refine_add_pieces)
        assert 'explicit_targets=ids, default_target="selected"' in hsrc

    def test_resolver_honors_explicit_targets_over_navigator(self):
        r = inspect.getsource(_ann()._resolve_batch_targets)
        assert "_batch_explicit_targets" in r, "selected mode must honor explicit targets"

    def test_open_dialog_accepts_explicit_target_hints(self):
        params = set(
            inspect.signature(_ann()._open_batch_refine_dialog).parameters
        )
        assert "explicit_targets" in params and "default_target" in params


class TestReviewSmooth:
    """The post-commit one-click 'Smooth' REVIEW button was removed — smoothing
    is now pre-commit (auto-smooth-on-Accept), so it was redundant."""

    def test_redundant_review_smooth_button_removed(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        wsrc = inspect.getsource(AdvancedFeaturesWidget.set_add_callbacks)
        assert '_wire("add_smooth_btn"' not in wsrc
        bsrc = inspect.getsource(AdvancedFeaturesWidget._build_add_denovo_section)
        assert "add_smooth_btn = QPushButton" not in bsrc


class TestClearBorderDefault:
    def test_clear_border_defaults_on(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        src = inspect.getsource(AdvancedFeaturesWidget)
        # The clear-border checkbox now defaults checked (edge-cleanup on).
        assert "self.add_clear_border_cb.setChecked(True)" in src


class TestEdgeFlags:
    """The new flag predicate (used by the candidate checklist) — no mutation."""

    def test_flags_axial_and_lateral_not_interior(self):
        import numpy as np
        host = TestEdgeCleanupIs3D._host()
        vol = np.zeros((12, 20, 20), dtype=np.int32)
        vol[5:8, 9:12, 9:12] = 1   # interior — not flagged
        vol[0:2, 9:12, 9:12] = 2   # Z-top (axial) — flagged
        vol[5:8, 9:12, 0:2] = 3    # X wall (lateral) — flagged
        assert host._add_edge_flags(vol, 3) == {2, 3}

    def test_single_piece_never_flagged(self):
        import numpy as np
        host = TestEdgeCleanupIs3D._host()
        vol = np.zeros((12, 20, 20), dtype=np.int32)
        vol[0:2, 0:2, 0:2] = 1     # lone piece, touches walls → stays checked
        assert host._add_edge_flags(vol, 3) == set()

    def test_deep_truncated_cell_is_flagged_by_proximity(self):
        # The fix: a cell clipped DEEP by the box (centroid past the old margin)
        # must still be flagged — the old centroid test let it slip through.
        import numpy as np
        host = TestEdgeCleanupIs3D._host()
        vol = np.zeros((30, 30, 30), dtype=np.int32)
        vol[10:21, 10:21, 10:21] = 1   # centred target — never reaches an edge
        vol[0:9, 10:21, 10:21] = 2     # truncated at z=0, 9 voxels deep (centroid z~4)
        flags = host._add_edge_flags(vol, 3)
        assert 2 in flags, "a deep-truncated wall cell must be flagged (proximity)"
        assert 1 not in flags, "the centred target must not be flagged"

    def test_non_3d_returns_empty(self):
        import numpy as np
        host = TestEdgeCleanupIs3D._host()
        assert host._add_edge_flags(np.zeros((5, 5), dtype=np.int32), 3) == set()


class TestCherryPick:
    """Pick candidate pieces before commit; Accept commits only the checked."""

    def test_preview_flags_edges_and_populates_checklist(self):
        # Add rebuild B2.3: the preview runs on a worker -- the edge flags
        # live in add_preview._finish (edge_flag_ids, on the RAW result) and
        # the checklist is pushed by _apply_add_result.
        from packages.epicycle_annotator import add_preview

        fsrc = inspect.getsource(add_preview._finish)
        assert "edge_flag_ids" in fsrc, "preview must FLAG edges, not drop"
        assert "_apply_add_edge_cleanup" not in fsrc, "preview must not drop pieces"
        src = inspect.getsource(_ann()._apply_add_result)
        assert "candidate_ids=order" in src
        assert "candidate_edge_ids=edge" in src

    def test_accept_commits_only_checked(self):
        src = inspect.getsource(_ann()._accept_add)
        assert "get_checked_add_pieces" in src
        assert "np.isin(result, checked)" in src

    def test_widget_checklist_api(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        params = set(
            inspect.signature(AdvancedFeaturesWidget.update_add_ui_state).parameters
        )
        assert {"candidate_ids", "candidate_sizes", "candidate_edge_ids"} <= params
        assert hasattr(AdvancedFeaturesWidget, "get_checked_add_pieces")


class TestCandidateOverlaySync:
    """Toggling a candidate checkbox re-renders the overlay to the checked set."""

    def test_callback_param_stored(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        params = set(
            inspect.signature(AdvancedFeaturesWidget.set_add_callbacks).parameters
        )
        assert "on_candidate_changed" in params
        src = inspect.getsource(AdvancedFeaturesWidget.set_add_callbacks)
        assert "_on_add_candidate_changed_cb = on_candidate_changed" in src

    def test_annotator_wires_and_masks_to_checked(self):
        wiring = inspect.getsource(_ann().set_curation_callbacks)  # imports class
        assert wiring
        h = inspect.getsource(_ann()._sync_add_candidate_overlay)
        assert "get_checked_add_pieces" in h
        assert "np.isin(result, checked)" in h
        assert "_render_add_candidate" in h
        # Display-only: it must NOT mutate the _add_preview_result source of truth.
        assert "_add_preview_result =" not in h


class TestRecenterOnCrosshair:
    """Place Box drops at the crosshair; a button re-snaps the box centre."""

    def test_callback_param_and_button_wired(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        params = set(
            inspect.signature(AdvancedFeaturesWidget.set_add_callbacks).parameters
        )
        assert "on_recenter" in params
        src = inspect.getsource(AdvancedFeaturesWidget.set_add_callbacks)
        assert '_wire("add_recenter_btn", on_recenter)' in src

    def test_start_add_drops_at_crosshair(self):
        src = inspect.getsource(_ann()._start_add)
        assert "_add_crosshair_center" in src, "Place Box must use the crosshair"

    def test_recenter_reuses_box_changed_path(self):
        h = inspect.getsource(_ann()._recenter_add_box_on_crosshair)
        assert "_add_crosshair_center" in h
        assert "_on_add_box_changed" in h, "must reuse the slider-move path"

    def test_crosshair_center_reads_subviewer_steps(self):
        h = inspect.getsource(_ann()._add_crosshair_center)
        assert "viewer_model_xy" in h and "viewer_model_xz" in h and "viewer_model_yz" in h

    def test_edge_diagnostic_stripped(self):
        src = inspect.getsource(_ann()._preview_add)
        assert "[ADD-EDGE]" not in src, "temp diagnostic must be removed"


class TestBoxSizePresets:
    """S/M/L relative presets + sticky last-size."""

    def test_callback_param_and_helper(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        params = set(
            inspect.signature(AdvancedFeaturesWidget.set_add_callbacks).parameters
        )
        assert "on_size_preset" in params
        assert hasattr(AdvancedFeaturesWidget, "_fire_add_size_preset")

    def test_annotator_wires_and_resizes_relative(self):
        h = inspect.getsource(_ann()._set_add_box_size_fraction)
        assert "_on_add_box_changed" in h, "must reuse the slider-move path"
        assert "frac" in h and "main.data.shape" in h, "must resize relative to patch"

    def test_sticky_size_used_and_updated(self):
        start = inspect.getsource(_ann()._start_add)
        assert "_add_last_size" in start and "size=" in start, "Place Box reuses last size"
        changed = inspect.getsource(_ann()._on_add_box_changed)
        assert "_add_last_size = (" in changed, "box change must remember the size"


class TestAutoSmoothOnAdd:
    """Isotropic SDF smooth on the candidate before commit."""

    def test_isotropic_sigma_and_reuses_batch_backend(self):
        # The body moved to the pure add_segment.smooth_pieces (Add rebuild
        # ); the method delegates to it.
        assert "smooth_pieces" in inspect.getsource(_ann()._smooth_add_result)
        from packages.epicycle_annotator.add_segment import smooth_pieces

        h = inspect.getsource(smooth_pieces)
        # Route through the batch backend (crops each piece to its bbox → fast),
        # not a full-ROI SDF per piece.
        assert "run_batch_recipe" in h
        assert 'method="sdf"' in h
        # Isotropic: same sigma both axes (configurable, default 1.5).
        assert '"sigma_z": float(sigma)' in h and '"sigma_xy": float(sigma)' in h

    def test_preview_smooths_wysiwyg_and_accept_does_not_resmooth(self):
        # Smoothing runs at PREVIEW (WYSIWYG — the preview shows the smoothed
        # cells), not at Accept.  _add_preview_result is already smoothed, so
        # Accept commits it as-is.
        # Add rebuild B2.3: the smoothing runs in the worker function
        # (add_preview._finish -> add_segment.smooth_pieces).
        from packages.epicycle_annotator import add_preview

        psrc = inspect.getsource(add_preview._finish)
        assert "smooth_pieces" in psrc, "preview must smooth (WYSIWYG)"
        assert 'params.get("auto_smooth"' in psrc, "gated on the toggle"
        asrc = inspect.getsource(_ann()._accept_add)
        assert "_smooth_add_result" not in asrc, "accept must NOT re-smooth"

    def test_smooth_preserves_pieces_isotropically(self):
        import numpy as np
        vol = np.zeros((16, 16, 16), dtype=np.int32)
        vol[4:9, 4:9, 4:9] = 5
        vol[4:9, 10:15, 10:15] = 6
        out = _ann()._smooth_add_result(object(), vol)
        labels = set(int(v) for v in np.unique(out)) - {0}
        assert labels == {5, 6}, "both cherry-picked pieces survive the smooth"
        assert not np.any((out == 5) & (out == 6)), "pieces stay non-overlapping"

    def test_widget_checkbox_default_on(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        src = inspect.getsource(AdvancedFeaturesWidget)
        assert "add_auto_smooth_cb" in src
        assert "self.add_auto_smooth_cb.setChecked(True)" in src


class TestWiring:
    def test_open_advanced_wires_add_callbacks(self):
        src = inspect.getsource(_ann()._open_advanced_features)
        assert "set_add_callbacks" in src
        assert "refresh_add_image_combo" in src or "_refresh_add_image_combo" in src

    def test_set_curation_callbacks_accepts_add(self):
        params = set(
            inspect.signature(_ann().set_curation_callbacks).parameters
        )
        assert "on_add_mode_lock" in params
        assert "on_add_committed" in params

    def test_refine_handoff_targets_the_refine_tab_by_name(self):
        """Was ``setCurrentIndex(5)``; the panel rework reaches the tab by
        NAME so a tab rename cannot silently retarget the hand-off."""
        src = inspect.getsource(_ann()._refine_add_piece)
        assert 'focus_tab("Refine")' in src
        assert "setCurrentIndex(5)" not in src


class TestControllerWiring:
    def test_lock_includes_add_mode(self):
        src = inspect.getsource(_ctrl()._refresh_step3_lock)
        assert "_add_mode_running" in src

    def test_switch_guard_includes_add_mode(self):
        src = inspect.getsource(_ctrl()._switch_to_curation_patch)
        assert "_add_mode_running" in src

    def test_commit_handler_uses_commit_curation_action(self):
        src = inspect.getsource(_ctrl()._on_annotator_add_committed)
        assert '"add_denovo"' in src
        assert "_commit_curation_action" in src

    def test_add_denovo_is_mask_mutating(self):
        src = inspect.getsource(_ctrl())
        # The frozenset literal must contain add_denovo for feature-staleness.
        assert '"add_denovo"' in src


class TestAuditFixes:
    """Pins the fixes from the 2026-06-30 adversarial audit."""

    def test_multi_box_session_log(self):
        """Accept appends to a per-commit session log (audit HIGH-1)."""
        src = inspect.getsource(_ann()._accept_add)
        assert "_add_session.append" in src
        # REVIEW shows every piece this session, not just this box.
        assert 'for e in self._add_session for i in e["ids"]' in src

    def test_undo_all_delegates_to_controller_undo(self):
        """Undo-all delegates to the global curation undo (reuses each
        add_denovo entry's volume+df restore) — no parallel annotator restore
        (re-audit MED)."""
        src = inspect.getsource(_ann()._undo_add_all)
        assert "_on_add_undo_session_cb" in src
        # Must NOT do its own bespoke volume restore anymore.
        assert "shared[entry" not in src

    def test_accept_releases_lock_in_finally(self):
        """Lock must release even if a commit step raises (audit MED-5)."""
        src = inspect.getsource(_ann()._accept_add)
        assert "finally:" in src
        # the lock-False is in the finally block (after the try body).
        assert src.index("finally:") < src.rindex("_fire_add_mode_lock(False)")

    def test_start_add_guards_lock_on_failure(self):
        src = inspect.getsource(_ann()._start_add)
        assert "_fire_add_mode_lock(True)" in src
        assert "except Exception" in src and "_reject_add()" in src

    def test_boundary_marker_not_disabled(self):
        """Marker combo must stay switchable (audit UX-H1)."""
        src = inspect.getsource(_ann()._start_add)
        assert "marker_enabled=False" not in src

    def test_candidate_deregisters_from_sync(self):
        """Candidate must be unregistered from the SyncManager (audit H1)."""
        src = inspect.getsource(_ann()._remove_add_candidate)
        assert "disconnect_label_sync" in src
        assert "unregister_layer" in src

    def test_candidate_readonly_in_subviewers(self):
        """Subviewer candidate copies must be non-editable (audit M2)."""
        src = inspect.getsource(_ann()._render_add_candidate)
        assert "editable = False" in src
        assert "get_subviewer_layer" in src

    def test_review_reset_on_patch_switch(self):
        """A committed REVIEW must clear stale state on patch switch (HIGH-2)."""
        src = inspect.getsource(_ann()._load_selected_layers)
        assert "_add_review" in src
        # B2 fix pass L-L2: the REVIEW branch goes through the one switch
        # teardown, which resets the state (and takes S1 from the table)
        assert "_teardown_add_for_switch" in src
        assert "_reset_add_state" in inspect.getsource(
            _ann()._teardown_add_for_switch)

    def test_synchronous_review_teardown_at_switch(self):
        """REVIEW session must be torn down SYNCHRONOUSLY at switch start, not
        lazily (race audit MED-1) — the controller calls _teardown_add_for_switch
        before the layer swap, mirroring the merge-dialog teardown."""
        W = _ann()
        assert hasattr(W, "_teardown_add_for_switch")
        wsrc = inspect.getsource(W._teardown_add_for_switch)
        # Clears the session but does NOT undo the committed work.
        assert "_reset_add_state" in wsrc
        assert "_undo_add_all" not in wsrc          # must not revert labels
        assert "_on_add_undo_session_cb" not in wsrc
        csrc = inspect.getsource(_ctrl()._switch_to_curation_patch)
        assert "_teardown_add_for_switch" in csrc

    def test_merge_deferred_to_phase_1_1(self):
        """Merge is a deferred stub for v1 (clean undo interaction)."""
        src = inspect.getsource(_ann()._merge_add_pieces)
        assert "later version" in src.lower() or "deferred" in src.lower()

    def test_phantom_commit_guarded(self):
        """An all-overlap box (nothing written) must not seed phantom ids
        (audit LOW-1)."""
        src = inspect.getsource(_ann()._accept_add)
        assert "if not committed:" in src
        # must compute would-commit from result[write_mask], no 'or ids' fallback
        assert "or ids" not in src

    def test_param_changes_do_not_run_watershed(self):
        """Box/threshold/CLAHE/advanced changes must NOT call _preview_add —
        the watershed runs only on the Preview button (user: live preview was
        laggy)."""
        for name in ("_on_add_box_changed", "_on_add_threshold_changed",
                     "_on_add_clahe_toggled", "_on_add_advanced_changed",
                     "_do_add_histogram"):
            src = inspect.getsource(getattr(_ann(), name))
            assert "_preview_add(" not in src, (
                f"{name} must not run the watershed live"
            )
        # Each invalidates the stale preview instead.
        for name in ("_on_add_box_changed", "_on_add_threshold_changed"):
            src = inspect.getsource(getattr(_ann(), name))
            assert "_invalidate_add_preview" in src

    def test_controller_undo_session_handler(self):
        c = _ctrl()
        assert hasattr(c, "_on_annotator_add_undo_session")
        src = inspect.getsource(c._on_annotator_add_undo_session)
        assert "_undo_last_curation" in src       # reuses the proven path
        assert '"add_denovo"' in src               # stops at non-add actions
        assert "peek()" in src

    def test_set_curation_callbacks_accepts_undo_session(self):
        params = set(
            inspect.signature(_ann().set_curation_callbacks).parameters
        )
        assert "on_add_undo_session" in params


class TestMethodCallsResolve:
    """Every `self._method(` call in the Add code must resolve to a real method.

    Source-inspection contracts can't catch a typo'd method NAME (it's a runtime
    AttributeError) — this does.  It would have caught the `_add_otsu_reset` vs
    `_on_add_otsu_reset` bug that made _start_add blow up and tear down the box.
    """

    ADD_METHODS = [
        "_start_add", "_on_add_box_changed", "_schedule_add_histogram",
        "_do_add_histogram", "_invalidate_add_preview",
        "_compute_add_histogram", "_on_add_clahe_toggled",
        "_on_add_threshold_changed", "_on_add_otsu_reset",
        "_on_add_advanced_changed", "_preview_add", "_add_existing_roi",
        "_apply_add_edge_cleanup", "_render_add_box", "_render_add_candidate",
        "_remove_add_candidate", "_remove_add_box_shapes", "_accept_add",
        "_reject_add", "_on_add_piece_selected", "_refine_add_piece",
        "_merge_add_pieces", "_add_another", "_undo_add_all",
        "_reset_add_state", "_fire_add_mode_lock", "_refresh_add_image_combo",
        "_push_add_ui", "_add_main_label_layer", "_add_shared_volume",
        "_add_next_label", "_add_spacing", "_add_gather_info",
    ]

    def test_add_state_attrs_initialized(self):
        """Every _add_* attr that _reset_add_state resets must be initialized in
        __init__ (else a read before the first reset is an AttributeError — the
        _add_preview_result bug)."""
        import re

        W = _ann()
        pat = re.compile(r"self\.(_add_\w+)\s*=")
        init_attrs = set(pat.findall(inspect.getsource(W.__init__)))
        reset_attrs = set(pat.findall(inspect.getsource(W._reset_add_state)))
        missing = reset_attrs - init_attrs
        assert not missing, (
            f"_add_* attrs reset but not initialized in __init__: {missing}"
        )

    def test_all_self_method_calls_exist(self):
        import re

        W = _ann()
        pattern = re.compile(r"self\.(_[A-Za-z0-9_]+)\(")
        missing = []
        for name in self.ADD_METHODS:
            assert hasattr(W, name), f"Add method {name} itself is missing"
            src = inspect.getsource(getattr(W, name))
            for called in pattern.findall(src):
                if called == name:
                    continue
                if not hasattr(W, called):
                    missing.append(f"{name} -> self.{called}()")
        assert not missing, "calls to non-existent methods:\n" + "\n".join(missing)
