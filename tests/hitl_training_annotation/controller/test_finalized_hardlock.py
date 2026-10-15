"""Tests for Step 2 finalized-slice HARD-LOCK (2026-05-28, fix/step2-finalized-hardlock).

Finalized slices (accepted / staged / committed) must be non-editable —
the brush does nothing — until the user Reverts in the panel.  This
extends the edit-states SOFT lock (which only grayed buttons + demoted
on draw) to actually block the brush via layer.editable=False.

Consolidation: the editability is computed in ONE place
(`_enforce_finalized_slice_lock`), called after the base
`_apply_edit_constraints`, on dims-scroll, and after status changes — so
no other editable-writer can leave a finalized slice paintable (no hydra).

Pure-Python fakes — no Qt/napari.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator


class _FakeLayer:
    def __init__(self, name, editable=True, mode="paint", visible=True):
        self.name = name
        self.editable = editable
        self.mode = mode
        self.visible = visible


def _make_annotator(target="lateral", status_map=None, layers=None,
                    training_locked=False):
    """Bind the real hard-lock methods to a MagicMock with fakes.

    status_map: {orient: status_string} returned by the status callback.
    layers: list of _FakeLayer (subviewer label layers).
    """
    status_map = status_map or {}
    ann = MagicMock()
    ann._active_target_mode = target
    ann._training_lock_active = training_locked
    ann._all_subviewer_label_layers = lambda: (layers or [])

    # Real _edit_state_for_orient needs a status callback + _vm_for_orient.
    def _status_cb(view, idx):
        return status_map.get(str(view).upper(), "")
    ann._slice_status_callback = _status_cb
    fake_vm = MagicMock()
    fake_vm.dims.current_step = (0,)
    ann._vm_for_orient = lambda orient: fake_vm

    # Bind the real methods under test.
    ann._edit_state_for_orient = Curation2DAnnotator._edit_state_for_orient.__get__(ann)
    ann._enforce_finalized_slice_lock = (
        Curation2DAnnotator._enforce_finalized_slice_lock.__get__(ann)
    )
    ann._active_bucket_slice_is_finalized = (
        Curation2DAnnotator._active_bucket_slice_is_finalized.__get__(ann)
    )
    return ann


# ---------------------------------------------------------------------------
# _enforce_finalized_slice_lock
# ---------------------------------------------------------------------------


def test_finalized_active_slice_becomes_non_editable():
    """Active-bucket (lateral→XY) layer on a finalized slice → editable=False."""
    layer = _FakeLayer("patch_work_labels_lateral_3d_XY", editable=True, mode="paint")
    ann = _make_annotator(target="lateral", status_map={"XY": "accepted"}, layers=[layer])
    ann._enforce_finalized_slice_lock()
    assert layer.editable is False
    assert layer.mode == "pan_zoom"


def test_staged_and_committed_also_locked():
    for status in ("staged", "committed"):
        layer = _FakeLayer("patch_work_labels_lateral_3d_XY", editable=True, mode="paint")
        ann = _make_annotator(target="lateral", status_map={"XY": status}, layers=[layer])
        ann._enforce_finalized_slice_lock()
        assert layer.editable is False, f"{status} slice must be locked"


def test_editable_active_slice_stays_editable():
    """predicted/curating slice in the active bucket → editable=True."""
    for status in ("predicted", "curating", ""):
        layer = _FakeLayer("patch_work_labels_lateral_3d_XY", editable=True, mode="paint")
        ann = _make_annotator(target="lateral", status_map={"XY": status}, layers=[layer])
        ann._enforce_finalized_slice_lock()
        assert layer.editable is True, f"{status!r} slice must stay editable"


def test_scroll_off_finalized_re_enables():
    """Bidirectional: a layer locked on a finalized slice gets re-enabled
    when the displayed slice is now editable (simulates scrolling away)."""
    layer = _FakeLayer("patch_work_labels_lateral_3d_XY", editable=False, mode="pan_zoom")
    ann = _make_annotator(target="lateral", status_map={"XY": "curating"}, layers=[layer])
    ann._enforce_finalized_slice_lock()
    assert layer.editable is True, "scrolling onto an editable slice must re-enable"


def test_inactive_bucket_layer_untouched():
    """In lateral mode, the axial XZ layer is inactive — enforce must NOT
    touch it (the base _apply_edit_constraints already locked it)."""
    xz = _FakeLayer("patch_work_labels_axial_xz_3d_XZ", editable=False, mode="pan_zoom")
    ann = _make_annotator(target="lateral", status_map={"XZ": "curating"}, layers=[xz])
    ann._enforce_finalized_slice_lock()
    # Untouched — stays as the base set it (False), NOT flipped to True.
    assert xz.editable is False, "inactive-bucket layer must not be re-enabled"


def test_axial_bucket_locks_xz_and_yz():
    xz = _FakeLayer("patch_work_labels_axial_xz_3d_XZ", editable=True, mode="paint")
    yz = _FakeLayer("patch_work_labels_axial_yz_3d_YZ", editable=True, mode="paint")
    ann = _make_annotator(
        target="axial",
        status_map={"XZ": "accepted", "YZ": "curating"},
        layers=[xz, yz],
    )
    ann._enforce_finalized_slice_lock()
    assert xz.editable is False, "finalized XZ must lock"
    assert yz.editable is True, "editable YZ must stay editable"


def test_training_lock_takes_precedence():
    """While training-locked, enforce is a no-op (set_training_lock owns
    editability)."""
    layer = _FakeLayer("patch_work_labels_lateral_3d_XY", editable=False, mode="pan_zoom")
    ann = _make_annotator(
        target="lateral", status_map={"XY": "curating"},
        layers=[layer], training_locked=True,
    )
    ann._enforce_finalized_slice_lock()
    # Not re-enabled — training lock holds.
    assert layer.editable is False


def test_hidden_layer_skipped():
    layer = _FakeLayer("patch_work_labels_lateral_3d_XY", editable=False, mode="pan_zoom", visible=False)
    ann = _make_annotator(target="lateral", status_map={"XY": "curating"}, layers=[layer])
    ann._enforce_finalized_slice_lock()
    # Hidden — don't force editability on.
    assert layer.editable is False


# ---------------------------------------------------------------------------
# _active_bucket_slice_is_finalized (for the LOCKED overlay message)
# ---------------------------------------------------------------------------


def test_active_bucket_finalized_detector():
    ann = _make_annotator(target="lateral", status_map={"XY": "accepted"})
    assert ann._active_bucket_slice_is_finalized() is True

    ann2 = _make_annotator(target="lateral", status_map={"XY": "curating"})
    assert ann2._active_bucket_slice_is_finalized() is False

    # axial: finalized if EITHER XZ or YZ is finalized
    ann3 = _make_annotator(target="axial", status_map={"XZ": "curating", "YZ": "staged"})
    assert ann3._active_bucket_slice_is_finalized() is True

    # training-locked → False (its own overlay handles it)
    ann4 = _make_annotator(target="lateral", status_map={"XY": "committed"}, training_locked=True)
    assert ann4._active_bucket_slice_is_finalized() is False


# ---------------------------------------------------------------------------
# Wiring (source inspection) — the consolidation must run everywhere
# ---------------------------------------------------------------------------


def test_apply_edit_constraints_enforces_after_super():
    src = inspect.getsource(Curation2DAnnotator._apply_edit_constraints)
    assert "super()._apply_edit_constraints()" in src
    assert "_enforce_finalized_slice_lock()" in src
    # enforce must come AFTER super (downgrade after the base sets bucket).
    assert src.index("super()._apply_edit_constraints()") < src.index("_enforce_finalized_slice_lock()")


def test_scroll_handler_enforces():
    src = inspect.getsource(Curation2DAnnotator._sync_slider_from_dims)
    assert "_enforce_finalized_slice_lock()" in src


def test_refresh_visuals_enforces():
    src = inspect.getsource(Curation2DAnnotator.refresh_edit_state_visuals)
    assert "_enforce_finalized_slice_lock()" in src


def test_mode_overlay_shows_revert_hint_on_finalized():
    src = inspect.getsource(Curation2DAnnotator._update_mode_overlays)
    assert "_active_bucket_slice_is_finalized()" in src
    assert "Revert" in src


def test_hardlock_not_on_base_or_3d():
    """Scope: the hard-lock helpers live on Curation2DAnnotator only —
    base + Step 3 must not inherit them."""
    from napari_epicycle.annotator.native_multiview_widget import NativeMultiviewAnnotator
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    assert "_enforce_finalized_slice_lock" in Curation2DAnnotator.__dict__
    assert not hasattr(NativeMultiviewAnnotator, "_enforce_finalized_slice_lock")
    assert Curation2DAnnotator not in Curation3DAnnotator.__mro__


def test_lock_main_canvas_readonly_locks_main_work_labels():
    """Main-canvas read-only (2026-05-28): _lock_main_canvas_readonly sets
    every main-viewer work_labels layer editable=False + pan_zoom."""
    ann = MagicMock()
    main_layers = [
        _FakeLayer("patch_work_labels_lateral_3d", editable=True, mode="paint"),
        _FakeLayer("patch_work_labels_axial_xz_3d", editable=True, mode="erase"),
        _FakeLayer("some_image_layer", editable=True, mode="paint"),  # not work_labels
    ]
    fake_viewer = MagicMock()
    fake_viewer.layers = main_layers
    ann.viewer = fake_viewer
    # Make isinstance(layer, Labels) work for our fakes.
    import napari.layers as _nl
    _orig = _nl.Labels
    try:
        _nl.Labels = _FakeLayer
        ann._lock_main_canvas_readonly = (
            Curation2DAnnotator._lock_main_canvas_readonly.__get__(ann)
        )
        ann._lock_main_canvas_readonly()
    finally:
        _nl.Labels = _orig
    # work_labels layers locked
    assert main_layers[0].editable is False and main_layers[0].mode == "pan_zoom"
    assert main_layers[1].editable is False and main_layers[1].mode == "pan_zoom"
    # non-work-labels layer untouched
    assert main_layers[2].editable is True


def test_main_canvas_lock_wired_in_apply_and_visibility_override():
    """Source guard: _lock_main_canvas_readonly is called from both the
    _apply_edit_constraints override and the _set_local_work_label_visibility
    override (the two main-work_labels editability-setting paths)."""
    apply_src = inspect.getsource(Curation2DAnnotator._apply_edit_constraints)
    assert "_lock_main_canvas_readonly()" in apply_src
    vis_src = inspect.getsource(Curation2DAnnotator._set_local_work_label_visibility)
    assert "super()._set_local_work_label_visibility" in vis_src
    assert "_lock_main_canvas_readonly()" in vis_src


def test_controller_visibility_setter_forces_main_readonly():
    """Source guard: the controller's _set_work_labels_visibility_for_context
    sets main work_labels editable=False (was editable=bool(visible))."""
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )
    src = inspect.getsource(
        TrainingAnnotationController._set_work_labels_visibility_for_context
    )
    assert "layer.editable = False" in src
    assert "layer.editable = bool(visible)" not in src


def test_main_canvas_lock_not_on_base_or_3d():
    """Scope: the main-canvas lock + the _set_local_work_label_visibility
    override live on Curation2DAnnotator only.  Step 3 inherits the BASE
    _set_local_work_label_visibility (main viewer stays editable for 3D
    curation drawing)."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    assert "_lock_main_canvas_readonly" in Curation2DAnnotator.__dict__
    assert "_set_local_work_label_visibility" in Curation2DAnnotator.__dict__
    # Step 3 must NOT inherit the Step 2 override.
    assert "_lock_main_canvas_readonly" not in Curation3DAnnotator.__dict__
    assert Curation2DAnnotator not in Curation3DAnnotator.__mro__


def test_training_unlock_re_enforces_hardlock():
    """Audit-3 gap fix: set_training_lock(False) must re-enforce the
    finalized hard-lock AFTER restoring the pre-training editable snapshot.
    Otherwise a slice finalized mid-training could be wrongly re-enabled by
    the snapshot restore.  Source-inspection: enforce appears in the unlock
    branch, after the snapshot-restore loop."""
    src = inspect.getsource(Curation2DAnnotator.set_training_lock)
    assert "_enforce_finalized_slice_lock()" in src, (
        "set_training_lock unlock branch must re-enforce the hard-lock"
    )
    # Must come AFTER the snapshot restore (so it overrides a wrongly-
    # restored editable=True on a now-finalized slice).
    assert src.index("_training_lock_snapshot") < src.index("_enforce_finalized_slice_lock()")
    # Audit (main-canvas): unlock must ALSO re-lock the main canvas
    # read-only (the snapshot restore includes main-viewer work_labels).
    assert "_lock_main_canvas_readonly()" in src
    assert src.index("_training_lock_snapshot") < src.index("_lock_main_canvas_readonly()")
