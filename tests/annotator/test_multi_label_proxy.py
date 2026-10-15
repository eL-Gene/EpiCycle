"""Tests for multi-label isolation proxy (Curation3DAnnotator).

Background: when the user multi-selects labels in 3D mode (e.g. ticking
candidates in the merge dialog), the legacy path applied a
``DirectLabelColormap`` to the full main layer — forcing napari to
re-run iso_categorical extraction over the entire 1024×1024×Z volume
on every checkbox tick.

The fix uses N small SINGLE-label proxy crops (one per selected label)
rather than one multi-label-colormap proxy. Each sub-proxy uses the
proven ``show_selected_label=True`` 3D render path — the same code
path the single-label refine isolation uses, just instantiated N
times. This guarantees the visual matches refine isolation exactly
(clean isosurface render per cell) and avoids napari's unreliable
multi-label DirectLabelColormap render in 3D iso_categorical mode.

Subviewers still get a DirectLabelColormap (cheap in 2D, no
isosurface extraction) for consistent multi-color slice views.

These tests are source-inspection + lightweight invariant checks
(Qt-free), following the pattern of tests/annotator/test_edt_overlay_lock.py.
"""

from __future__ import annotations

import inspect


def _annotator():
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    return Curation3DAnnotator


class TestMethodsExist:
    def test_apply_method_exists(self):
        cls = _annotator()
        assert callable(getattr(cls, "_apply_multi_label_proxy", None)), (
            "Curation3DAnnotator must define _apply_multi_label_proxy"
        )

    def test_build_subproxy_helper_exists(self):
        cls = _annotator()
        assert callable(getattr(cls, "_build_single_label_subproxy", None)), (
            "Curation3DAnnotator must define _build_single_label_subproxy "
            "(per-label crop helper used by _apply_multi_label_proxy)"
        )

    def test_subviewer_propagate_helper_exists(self):
        cls = _annotator()
        assert callable(
            getattr(cls, "_propagate_multi_label_colormap_to_subviewers", None)
        ), (
            "must define _propagate_multi_label_colormap_to_subviewers — "
            "subviewers stay on DirectLabelColormap (cheap in 2D); only "
            "the main 3D viewer uses sub-proxies"
        )

    def test_init_initializes_proxy_set_and_layer_dict(self):
        cls = _annotator()
        src = inspect.getsource(cls.__init__)
        assert "_multi_label_proxy_set" in src
        assert "_multi_label_proxy_layers" in src, (
            "__init__ must initialize _multi_label_proxy_layers (dict[int, Labels]) "
            "to track per-label sub-proxies"
        )


class TestNProxiesPattern:
    """The fix's core idea: one sub-proxy per selected label, each in
    show_selected_label mode (mirroring the single-label refine path)."""

    def test_subproxy_uses_show_selected_label_true(self):
        src = inspect.getsource(_annotator()._build_single_label_subproxy)
        assert "show_selected_label = True" in src, (
            "each sub-proxy must use show_selected_label=True — the "
            "proven 3D render path that matches refine isolation visual"
        )

    def test_subproxy_sets_selected_label(self):
        src = inspect.getsource(_annotator()._build_single_label_subproxy)
        assert "selected_label = label_id" in src

    def test_subproxy_uses_axis_reduced_bbox_scan(self):
        """Same bbox pattern as _update_isolation_proxy single-label —
        axis-reduced np.any avoids materializing dense coordinate arrays."""
        src = inspect.getsource(_annotator()._build_single_label_subproxy)
        assert "np.any" in src
        assert "data == label_id" in src

    def test_subproxy_stored_in_dict_keyed_by_label_id(self):
        src = inspect.getsource(_annotator()._build_single_label_subproxy)
        assert "_multi_label_proxy_layers[label_id]" in src

    def test_apply_diffs_against_current_set(self):
        """Per-toggle should add/remove only the labels that changed,
        not rebuild every sub-proxy."""
        src = inspect.getsource(_annotator()._apply_multi_label_proxy)
        assert "to_remove" in src and "to_add" in src, (
            "must compute set diff (to_remove / to_add) so per-toggle "
            "only adds/removes the changed sub-proxies, not all of them"
        )

    def test_apply_calls_subproxy_builder_for_added_labels(self):
        src = inspect.getsource(_annotator()._apply_multi_label_proxy)
        assert "_build_single_label_subproxy" in src

    def test_apply_no_op_when_set_unchanged(self):
        src = inspect.getsource(_annotator()._apply_multi_label_proxy)
        assert "frozenset" in src
        assert "_multi_label_proxy_set" in src

    def test_apply_hides_main_layer(self):
        src = inspect.getsource(_annotator()._apply_multi_label_proxy)
        assert "main_layer.visible = False" in src

    def test_apply_marks_proxy_active(self):
        src = inspect.getsource(_annotator()._apply_multi_label_proxy)
        assert "_isolation_proxy_active = True" in src
        assert "_multi_label_proxy_set = frozen" in src


class TestRoutingFromOnIsolationChanged:
    def test_branch_checks_ndisplay(self):
        src = inspect.getsource(_annotator()._on_isolation_changed)
        assert ("ndisplay == 3" in src) or ("_ndisplay == 3" in src)

    def test_3d_path_calls_proxy_method(self):
        src = inspect.getsource(_annotator()._on_isolation_changed)
        assert "_apply_multi_label_proxy" in src

    def test_2d_path_keeps_legacy_colormap(self):
        src = inspect.getsource(_annotator()._on_isolation_changed)
        assert "_apply_multi_isolation_colormap" in src

    def test_3d_path_propagates_colormap_to_subviewers_only(self):
        src = inspect.getsource(_annotator()._on_isolation_changed)
        assert "_propagate_multi_label_colormap_to_subviewers" in src, (
            "3D path must propagate the multi-label colormap to "
            "subviewers via the dedicated helper (NOT to the main layer)"
        )


class TestTransitionCleanup:
    def test_remove_proxy_clears_multi_label_dict(self):
        src = inspect.getsource(_annotator()._remove_isolation_proxy)
        assert "_multi_label_proxy_layers" in src, (
            "_remove_isolation_proxy must walk _multi_label_proxy_layers "
            "and tear down each sub-proxy"
        )
        assert "_multi_label_proxy_set = None" in src

    def test_remove_proxy_uses_hide_before_remove(self):
        """Vispy access-violation discipline: hide layer before removing
        it."""
        src = inspect.getsource(_annotator()._remove_isolation_proxy)
        # Walk over the dict tearing each down — visible=False before remove.
        assert "layer.visible = False" in src

    def test_single_label_takeover_tears_down_multi_proxies(self):
        """When _update_isolation_proxy (single) takes over from a
        multi-label proxy, it must tear down all multi sub-proxies so
        both don't render simultaneously."""
        src = inspect.getsource(_annotator()._update_isolation_proxy)
        assert "_multi_label_proxy_layers" in src
        assert "_multi_label_proxy_set = None" in src

    def test_apply_tears_down_single_label_proxy_on_entry(self):
        """When transitioning single→multi, the existing single-label
        proxy must be removed so it doesn't coexist with the new sub-proxies."""
        src = inspect.getsource(_annotator()._apply_multi_label_proxy)
        assert "_isolation_proxy_layer" in src and "remove" in src


class TestMergeMutationRebuildsProxies:
    """After Merge Selected mutates the volume, the multi-label
    sub-proxies for absorb_ids are stale (each one holds a pre-merge
    crop). The mutation must re-apply the multi-label proxy with
    just {keep_id} so:
      - Absorb sub-proxies are torn down (via set-diff in
        _apply_multi_label_proxy).
      - Keep proxy's bbox is rebuilt to include the newly-absorbed
        voxels.
      - Subviewer colormap is re-propagated with just {keep_id} (the
        absorb-id entries are dead after mutation).
    Without this rebuild the main 3D viewer continues showing the
    unmerged absorbed cells while subviewers show the merged result
    (the user-visible symptom that motivated this fix).
    """

    def test_execute_batch_merge_resizes_keep_proxy_inline(self):
        """The rebuild must compute the new keep bbox INLINE via
        union(old proxy bbox, merge ROI) — not via
        _apply_multi_label_proxy's set-diff (which would not rebuild
        the keep proxy because keep_id is already in the dict)."""
        src = inspect.getsource(_annotator()._execute_batch_merge)
        # The inline path resizes keep_proxy.data + translate directly,
        # without calling _build_single_label_subproxy from scratch
        # (which would do a full-volume scan).
        assert "keep_proxy.data" in src and "keep_proxy.translate" in src, (
            "_execute_batch_merge must directly resize the keep "
            "sub-proxy's data + translate to bbox-union(old, ROI) "
            "instead of going through _apply_multi_label_proxy's "
            "set-diff (which would leave the existing keep proxy "
            "untouched because keep_id is already in the dict)"
        )

    def test_execute_batch_merge_tears_down_absorb_proxies(self):
        """Absorb sub-proxies are stale and must be removed (the
        labels no longer exist as separate voxels in the volume)."""
        src = inspect.getsource(_annotator()._execute_batch_merge)
        # Pops each absorb_id's sub-proxy from the dict and removes it
        # from the viewer.
        assert "_multi_label_proxy_layers" in src
        assert "pop(absorb_id" in src or "pop(int(absorb_id)" in src or \
            "absorb_set" in src

    def test_execute_batch_merge_no_full_volume_scan(self):
        """The keep proxy resize must be via direct slice math — NOT
        np.any-axis-reduced bbox compute (which would scan the whole
        volume and undo the snappy-merge win)."""
        src = inspect.getsource(_annotator()._execute_batch_merge)
        # The inline path uses old proxy translate + shape and
        # roi_slices for bbox math — no `np.any` scan in this block.
        assert "old_t" in src or "old_translate" in src, (
            "should read the existing proxy's translate to compute "
            "the bbox union without a fresh full-volume scan"
        )
        assert "old_shape" in src, "should use old proxy data.shape too"
        assert "roi_slices[0].start" in src

    def test_rebuild_is_gated_on_proxy_active(self):
        """The rebuild must be a no-op when no isolation proxy is
        active (e.g. 2D mode, or no isolation at the moment of
        merge — DirectLabelColormap-on-main handles those cases)."""
        src = inspect.getsource(_annotator()._execute_batch_merge)
        assert "_isolation_proxy_active" in src

    def test_subviewer_colormap_repropagated_with_keep_only(self):
        """After mutation, subviewer DirectLabelColormap entries for
        the absorbed labels are dead — repropagate with just
        {keep_id} so the colormap is consistent."""
        src = inspect.getsource(_annotator()._execute_batch_merge)
        assert "_propagate_multi_label_colormap_to_subviewers" in src
        assert "{keep_int}" in src or "{int(keep_id)}" in src

    def test_fallback_on_exception(self):
        """Any error in the inline path falls back to
        _apply_multi_label_proxy. Behavior is always correct,
        just slower."""
        src = inspect.getsource(_annotator()._execute_batch_merge)
        assert "_apply_multi_label_proxy" in src, (
            "the except clause should fall back to the full rebuild "
            "via _apply_multi_label_proxy so behavior is correct "
            "even if the inline bbox math hits an unexpected state"
        )
