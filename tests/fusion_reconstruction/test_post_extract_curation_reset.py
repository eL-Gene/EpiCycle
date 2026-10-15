"""Regression tests for the post-extract triage-lag fix (switch-in-place).

Branch: ``fix/step3-post-extract-curation-reset`` (2026-06-09).

User report: dragging the ``qc_score`` triage-histogram threshold was
catastrophically slow (~3s/tick) IMMEDIATELY after a FRESH QC extract, but fast
(~13ms/tick) on any patch loaded via a normal patch switch.

Root cause (measured): the in-memory extract+cluster+UMAP DataFrame is ~1000x
slower for per-column access than the same df read back from CSV — the cost
surfaces in ``triage_histogram_panel._populate_combos_from_df`` (~3300ms vs
~3ms), which runs on every ``set_data`` (i.e. every drag tick).  rows/cols/
nblocks are IDENTICAL between the slow + fast states, so the df "looks" the same.

Fix: a patch switch cures it because it RELOADS the df from CSV
(``_reload_current_patch_features`` → ``load_patch_csv``).  After an extract
(and after the async cluster's CSV save) we re-run the FULL patch switch IN
PLACE on the current patch — ``_switch_in_place_after_extract`` →
``_switch_to_curation_patch(patch, force=True)`` — reproducing the known-good
switch (incl. the CSV reload).

These are source-introspection + light-fake tests (no Qt / napari viewer) so
they run in CI alongside the rest of ``tests/fusion_reconstruction/``.
"""

from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# The curation surfaces are reset by the switch-in-place (which re-runs the
# full patch switch, incl. _hard_clear_curation_surfaces) — NOT by a separate
# post-extract call.  Below: the shared helper the switch relies on actually
# resets isolation + authority.
# ---------------------------------------------------------------------------

class _ResettableAuthority:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _BareUI:
    pass


def _make_bare_controller():
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._isolated_label_ids = {3, 7, 11}
    c._isolated_df_indices = [1, 2, 5]
    c._colormap_authority = _ResettableAuthority()
    c.ui = _BareUI()  # no qc_plot_widget -> the pw branch is guarded
    c._get_navigator_widget = lambda: None  # no navigator -> nav branch guarded
    return c


def test_hard_clear_resets_isolation_and_authority():
    """``_hard_clear_curation_surfaces`` must clear the controller isolation
    caches and fully reset the colormap authority — the exact state whose
    staleness made post-extract isolate/lasso no-op."""
    c = _make_bare_controller()
    c._hard_clear_curation_surfaces()
    assert c._isolated_label_ids == set(), "isolation label IDs must be cleared"
    assert c._isolated_df_indices == [], "isolation df indices must be cleared"
    assert c._colormap_authority.reset_calls == 1, (
        "the colormap authority must be fully reset (the owned-authority claim "
        "failure is what made post-extract isolate silently no-op)"
    )


def test_hard_clear_is_safe_without_plot_or_navigator():
    """The helper must not raise when the plot widget / navigator are absent
    (the bare-instance path used by these tests + early-lifecycle calls)."""
    c = _make_bare_controller()
    c._hard_clear_curation_surfaces()  # must not raise


# ---------------------------------------------------------------------------
# Ported perf one-liners (kept from fix/step3-qc-tier-crash).
# ---------------------------------------------------------------------------

def test_extract_finished_consolidates_dataframe():
    """The merge-fragmented df is consolidated once via .copy() so the triage
    hot path isn't 30-100x slower than a CSV-loaded df."""
    src = inspect.getsource(
        FusionReconstructionController._on_qc_extract_finished
    )
    assert "self._qc_features_df = df = df.copy()" in src, (
        "the post-merge df must be consolidated with a single .copy() "
        "(ported from fix/step3-qc-tier-crash 2a49ad3)"
    )


def test_cluster_id_seeded_float_not_object():
    """cluster_id must be seeded as float NaN, not pd.NA — pd.NA forces an
    object-dtype column that makes every full-width triage copy ~100x slower."""
    src = inspect.getsource(FusionReconstructionController._run_clustering)
    assert 'df_full["cluster_id"] = np.nan' in src, (
        "cluster_id must be seeded np.nan (cheap numeric block)"
    )
    assert 'df_full["cluster_id"] = pd.NA' not in src, (
        "cluster_id must NOT be seeded pd.NA (forces object dtype -> slow)"
    )


# ---------------------------------------------------------------------------
# Post-extract "switch in place" — the actual cure for the threshold-drag lag.
# Measured: the in-memory extract df is ~1000x slower for the per-column access
# that _populate_combos_from_df + _redraw do every drag tick (combos ~3300ms vs
# ~3ms on a read-from-CSV df).  A real patch switch cures it because it RELOADS
# the df from CSV; we reproduce the full switch on the current patch after the
# extract (and after the cluster), exactly once against the final on-disk CSV.
# ---------------------------------------------------------------------------

def test_switch_to_curation_patch_accepts_force():
    """_switch_to_curation_patch must take a force= kwarg so the post-extract
    switch-in-place can bypass the same-patch guard on the CURRENT patch."""
    sig = inspect.signature(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "force" in sig.parameters, (
        "_switch_to_curation_patch must accept force= to allow a same-patch "
        "switch-in-place"
    )


def test_same_patch_guard_is_force_gated():
    """The same-patch early-return must be skipped when force=True."""
    src = inspect.getsource(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "if not force and patch_name == self._qc_view_patch_name" in src, (
        "the same-patch guard must be gated on `not force`"
    )


def test_extract_finish_switches_in_place_no_cluster():
    """The no-auto-cluster path must fire the switch-in-place once the CSV is
    final on disk."""
    src = inspect.getsource(
        FusionReconstructionController._on_qc_extract_finished
    )
    assert "self._switch_in_place_after_extract(patch_name)" in src, (
        "_on_qc_extract_finished (no-cluster path) must call "
        "_switch_in_place_after_extract(patch_name)"
    )


def test_cluster_apply_switches_in_place():
    """The auto-cluster path must fire the switch-in-place after its own CSV
    save (so it runs once, against the final cluster_id CSV)."""
    src = inspect.getsource(FusionReconstructionController._run_clustering)
    assert "self._switch_in_place_after_extract(" in src, (
        "_run_clustering's apply must call _switch_in_place_after_extract"
    )
    # Regression guard for the bug that silently no-op'd this path: the call
    # must NOT reference `patch_name` (which is assigned later, inside the save
    # block — using it earlier raised NameError that the try/except swallowed).
    assert "self._switch_in_place_after_extract(patch_name)" not in src, (
        "must NOT pass `patch_name` (used-before-assignment in the cluster "
        "apply); resolve the patch independently"
    )


def test_switch_in_place_helper_forces_and_guards_current_patch():
    """The helper must (a) only act when the extracted patch is still in view,
    (b) re-run the switch with force=True, (c) defer it (busy-lock timing)."""
    src = inspect.getsource(
        FusionReconstructionController._switch_in_place_after_extract
    )
    assert "_qc_view_patch_name" in src, (
        "must guard on the currently-viewed patch (no stale-patch switch)"
    )
    assert "force=True" in src, "must re-run the switch with force=True"
    assert "self._switch_to_curation_patch(" in src, (
        "must call the canonical switch (reuse the known-good path)"
    )
    assert "singleShot" in src, (
        "must defer so the extract/cluster busy-lock has cleared"
    )


def test_switch_in_place_skips_when_unsaved_edits():
    """The deferred switch reloads the patch from disk, so it must SKIP when
    there are unsaved tier/curation edits (else it would discard them and pop a
    surprise unsaved-changes modal).  The lag is the lesser evil vs. data loss;
    the user commits/saves then switches manually."""
    src = inspect.getsource(
        FusionReconstructionController._switch_in_place_after_extract
    )
    assert "_tiers_dirty" in src and "_curation_dirty" in src, (
        "must skip the auto switch-in-place when _tiers_dirty/_curation_dirty "
        "(switch-in-place reloads from disk -> would lose unsaved edits)"
    )
