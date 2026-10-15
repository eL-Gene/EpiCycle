"""Tests for the manual-recompute triage-lag cure (lighter CSV reload).

Branch: ``fix/step3-recompute-reload`` (2026-06-09).

The post-extract triage-drag lag (in-memory df ~1000x slower for per-column
access than a read_csv df) is cured on the AUTOMATIC post-extract path by the
full switch-in-place.  But the MANUAL recompute buttons had the wrong behaviour:
the Cluster button inherited that heavy ~1.4s full switch (it shares
``_run_clustering`` with auto-cluster), while PCA/UMAP/tSNE did nothing (leaving
the histogram slow).

Fix: a lighter ``_refresh_triage_from_csv`` (reload the fresh CSV df → fast
per-column access, no layer rebuild / annotator teardown) for the manual
buttons; the automatic post-extract path keeps the verified full switch.  A
``post_extract`` flag on ``_run_clustering`` distinguishes the two callers.

Source-introspection + light tests (no Qt/napari viewer) so they run in CI.
"""

from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


def test_run_clustering_has_post_extract_param_and_absorbs_positional():
    """`_run_clustering` must take `post_extract` (to split auto vs manual) and
    `*_args` (to absorb the `clicked(bool)` Qt passes to the button slot)."""
    sig = inspect.signature(FusionReconstructionController._run_clustering)
    assert "post_extract" in sig.parameters, (
        "_run_clustering must take a post_extract kwarg"
    )
    assert any(
        p.kind is inspect.Parameter.VAR_POSITIONAL
        for p in sig.parameters.values()
    ), "_run_clustering must take *_args to absorb the button's clicked(bool)"


def test_cluster_apply_branches_on_post_extract():
    """The cluster apply must use the FULL switch on the automatic path and the
    LIGHTER reload on the manual path."""
    src = inspect.getsource(FusionReconstructionController._run_clustering)
    assert "if post_extract:" in src, "must branch on post_extract"
    assert "self._switch_in_place_after_extract(" in src, (
        "automatic post-extract path must keep the full switch-in-place"
    )
    assert "self._refresh_triage_from_csv(" in src, (
        "manual Cluster button must use the lighter CSV reload"
    )


def test_auto_cluster_caller_passes_post_extract_true():
    """The automatic post-extract auto-cluster must request the full switch."""
    src = inspect.getsource(
        FusionReconstructionController._on_qc_extract_finished
    )
    assert "self._run_clustering(post_extract=True)" in src, (
        "_on_qc_extract_finished must call _run_clustering(post_extract=True) "
        "so the automatic path keeps the full switch"
    )


def test_recompute_refresh_saves_before_reloading():
    """PCA/UMAP/tSNE refresh must SAVE the CSV first, THEN reload from it — the
    reload reads the new columns, and pushing the slow in-memory df is exactly
    what we avoid."""
    src = inspect.getsource(
        FusionReconstructionController._persist_and_refresh_after_recompute
    )
    assert "save_patch_csv(" in src and "_refresh_triage_from_csv(" in src
    assert src.index("save_patch_csv(") < src.index(
        "_refresh_triage_from_csv("
    ), "must save the CSV BEFORE refreshing from it"
    # The in-memory-df push must be the SAVE-FAILED fallback only — i.e. it
    # appears AFTER the reload call in source (the elif branch), not as the
    # primary refresh.
    assert "set_features(df_full)" in src
    assert src.index("_refresh_triage_from_csv(") < src.index(
        "set_features(df_full)"
    ), "the slow in-memory set_features(df_full) must be the fallback, not primary"


def test_refresh_helper_exists_and_refreshes_both_plots():
    """The lighter reload must read the CSV, swap the df, and refresh BOTH the
    navigator/histogram and the scatter."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_triage_from_csv
    )
    assert "load_patch_csv(" in src, "must reload from the on-disk CSV"
    assert "self._qc_features_df = df_fast" in src, "must swap in the fast df"
    assert "_update_flagged_labels_list()" in src, (
        "must refresh navigator + triage histogram"
    )
    assert "set_features(" in src, "must refresh the scatter"


def test_refresh_helper_is_mask_and_dirty_flag_safe():
    """The lighter reload must NOT write the mask or the dirty flags — that's
    what makes it safe to run with unsaved edits (no guard needed)."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_triage_from_csv
    )
    assert "self._tiers_dirty =" not in src, (
        "must NOT clear _tiers_dirty (tiers round-trip via the caller's save)"
    )
    assert "self._curation_dirty =" not in src, (
        "must NOT touch _curation_dirty (never reloads the mask)"
    )


def test_refresh_helper_guards_current_patch():
    """Must skip if a patch switch landed during the worker compute."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_triage_from_csv
    )
    assert "_qc_view_patch_name" in src, (
        "must guard on the currently-viewed patch (no stale-patch refresh)"
    )
