"""Workspace-scoped cache resets on workspace load.

`_browse_workdir` loads a new workspace but historically did NOT clear the
caches keyed by patch_name / run_id — so stale entries from the previous
workspace bled into the new one (patch names collide across workspaces):
  - `_manifest_cache`        → wrong reconstruction manifest/payload
  - `_patch_tier_thresholds` → wrong triage threshold lines restored
  - `_qc_staleness_notified` → real staleness notifications suppressed
  - `_edited_patches`        → phantom edit badges on clean patches

The fix adds `_reset_workspace_scoped_caches()` and calls it early in
`_browse_workdir` (before state load / artifact scan / UI refresh).
"""

from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


_RESET_ATTRS = (
    "_manifest_cache",
    "_patch_tier_thresholds",
    "_qc_staleness_notified",
    "_edited_patches",
)


def test_reset_workspace_scoped_caches_clears_all_four():
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    # Seed each cache with stale entries from a "previous workspace".
    c._manifest_cache = {"patchA": {"labels_path": "/old/a.tif"}}
    c._patch_tier_thresholds = {"patchA": (0.1, 0.2, 0.3)}
    c._qc_staleness_notified = {("patchA", "spacing")}
    c._edited_patches = {"patchA"}

    c._reset_workspace_scoped_caches()

    assert c._manifest_cache == {}
    assert c._patch_tier_thresholds == {}
    assert c._qc_staleness_notified == set()
    assert c._edited_patches == set()


def test_reset_workspace_scoped_caches_preserves_object_identity():
    """Uses .clear() (in-place) rather than reassignment, so any held ref to
    a cache stays valid after the reset."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._manifest_cache = {"x": 1}
    c._patch_tier_thresholds = {"x": (1, 2, 3)}
    c._qc_staleness_notified = {("x", "y")}
    c._edited_patches = {"x"}
    refs = {a: getattr(c, a) for a in _RESET_ATTRS}

    c._reset_workspace_scoped_caches()

    for a in _RESET_ATTRS:
        assert getattr(c, a) is refs[a], f"{a} must be cleared in place, not replaced"
        assert len(getattr(c, a)) == 0


def test_reset_workspace_scoped_caches_safe_on_bare_instance():
    """No attrs set (partial/bare instance) → must not raise."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    # None of the attrs exist.
    c._reset_workspace_scoped_caches()  # must not raise


def test_reset_workspace_scoped_caches_clears_undo_stacks():
    """'conditionally': the tier + curation undo stacks must clear too so
    an undo from the previous workspace can't restore its snapshot into a
    same-named patch in the new workspace."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    # Minimal stand-ins (both support .clear()): a list + an object with clear().
    c._tier_undo_stack = ["entryA", "entryB"]

    class _Stack:
        def __init__(self):
            self._items = ["x"]
            self.cleared = False

        def clear(self):
            self._items.clear()
            self.cleared = True

    stack = _Stack()
    c._curation_undo_stack = stack
    # Also seed the four core caches so the helper has them.
    c._manifest_cache = {}
    c._patch_tier_thresholds = {}
    c._qc_staleness_notified = set()
    c._edited_patches = set()

    c._reset_workspace_scoped_caches()

    assert c._tier_undo_stack == []
    assert stack.cleared is True
    assert stack._items == []


def test_browse_workdir_calls_reset_before_state_load_source():
    # B2 (shared workspace) moved the workspace load out of
    # ``_browse_workdir`` into ``_adopt_workspace_quiet``, so entering the
    # step and browsing to a folder read it the same way.  Browse still runs
    # it, and the reset still comes first.
    browse = inspect.getsource(FusionReconstructionController._browse_workdir)
    assert "_adopt_workspace_quiet(" in browse, (
        "_browse_workdir must run the shared quiet load"
    )
    src = inspect.getsource(
        FusionReconstructionController._adopt_workspace_quiet)
    assert "_reset_workspace_scoped_caches()" in src, (
        "the workspace load must reset workspace-scoped caches"
    )
    # The reset must run BEFORE state_manager.load (so the load/scan/refresh
    # operate on a clean slate, not stale previous-workspace entries).
    reset_idx = src.index("_reset_workspace_scoped_caches()")
    load_idx = src.index("self.state_manager.load(")
    assert reset_idx < load_idx, (
        "the cache reset must run BEFORE state_manager.load"
    )


def test_reset_helper_covers_exactly_the_documented_caches_source():
    """The helper must clear all four documented caches (regression guard if a
    new workspace-scoped cache is added but not wired here)."""
    src = inspect.getsource(
        FusionReconstructionController._reset_workspace_scoped_caches
    )
    for attr in _RESET_ATTRS + ("_tier_undo_stack", "_curation_undo_stack"):
        assert attr in src, f"{attr} must be cleared by _reset_workspace_scoped_caches"


def test_reset_clears_the_extraction_spacing_and_the_picture_failures():
    """A LOW-2: the launch-spacing stash (``_qc_launch_spacing``, read by
    the finish and the sidecar) and the remembered picture failures
    (``_s3_thumb_failed``) are keyed by patch name -- a same-named patch in
    the next workspace must not inherit them."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._qc_launch_spacing = {"patchA": {"zyx": [1.0, 1.0, 1.0],
                                       "source": "user"}}
    c._s3_thumb_failed = {"patchA": (("p", 1, 2), "tip")}
    held = (c._qc_launch_spacing, c._s3_thumb_failed)
    c._reset_workspace_scoped_caches()
    assert c._qc_launch_spacing == {} and c._s3_thumb_failed == {}
    assert (c._qc_launch_spacing, c._s3_thumb_failed)[0] is held[0]
