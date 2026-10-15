"""Tests for Step 2 training-lock (2026-05-27).

Covers:
  - workflow_shell.lock_step_nav iterates _nav_items and toggles enabled
  - Curation2DAnnotator.set_training_lock locks all Labels layers + buttons
  - _collect_lockable_label_layers collects main + 3 subviewers
  - Controller's _set_training_ui_locked invokes both surfaces
  - Idempotence (locking twice / unlocking from idle is safe)

Pure-Python tests with minimal fakes — no Qt instantiation.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Surface 1 — workflow_shell.lock_step_nav
# ---------------------------------------------------------------------------


class _FakeNavItem:
    """Stand-in for ui_shell.workflow_shell._NavItem."""
    def __init__(self):
        self.enabled = True
        self._setEnabled_calls = []

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._setEnabled_calls.append(bool(enabled))


def _make_shell_with_nav_items(item_count: int = 5):
    """Build a minimal shell-like object exposing _nav_items + the
    lock_step_nav method copied from the real class."""
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    shell = MagicMock()
    shell._nav_items = {f"step_{i}": _FakeNavItem() for i in range(item_count)}
    # Bind the real method to the fake object — preserves the production
    # implementation under test instead of re-typing it.
    shell.lock_step_nav = EpiCycleWorkflowShell.lock_step_nav.__get__(shell)
    return shell


def test_lock_step_nav_disables_all_items():
    shell = _make_shell_with_nav_items(5)
    shell.lock_step_nav(True)
    for item in shell._nav_items.values():
        assert item.enabled is False
    assert all(item._setEnabled_calls[-1] is False for item in shell._nav_items.values())


def test_lock_step_nav_unlock_restores_enabled():
    shell = _make_shell_with_nav_items(5)
    shell.lock_step_nav(True)
    shell.lock_step_nav(False)
    for item in shell._nav_items.values():
        assert item.enabled is True


def test_lock_step_nav_handles_missing_nav_items():
    """If _nav_items doesn't exist (called before _build_left_panel),
    lock_step_nav must be a no-op, not a crash."""
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    shell = MagicMock()
    # Explicitly remove _nav_items so getattr returns None.
    if hasattr(shell, "_nav_items"):
        del shell._nav_items
    shell.lock_step_nav = EpiCycleWorkflowShell.lock_step_nav.__get__(shell)
    # Should not raise.
    shell.lock_step_nav(True)
    shell.lock_step_nav(False)


def test_lock_step_nav_empty_nav_items():
    """Empty dict — no-op, not a crash."""
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    shell = MagicMock()
    shell._nav_items = {}
    shell.lock_step_nav = EpiCycleWorkflowShell.lock_step_nav.__get__(shell)
    shell.lock_step_nav(True)


def test_lock_step_nav_handles_setEnabled_failure():
    """If one item's setEnabled raises, the rest must still be locked."""
    shell = _make_shell_with_nav_items(3)
    # Replace one item's setEnabled with a raising version.
    items_list = list(shell._nav_items.values())
    def _raises(_):
        raise RuntimeError("simulated Qt failure")
    items_list[1].setEnabled = _raises
    shell.lock_step_nav(True)
    # Items 0 and 2 still got locked.
    assert items_list[0].enabled is False
    assert items_list[2].enabled is False


# ---------------------------------------------------------------------------
# Surface 2 — Curation2DAnnotator.set_training_lock
# ---------------------------------------------------------------------------


class _FakeLayer:
    """Stand-in for napari.layers.Labels — but must pass isinstance(layer, Labels).
    We patch the import in tests that need that check."""
    def __init__(self, name="work_labels", editable=True, mode="paint"):
        self.name = name
        self.editable = editable
        self.mode = mode


class _FakeButton:
    def __init__(self):
        self.enabled = True
    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _FakeLabel:
    def __init__(self):
        self.text = ""
        self.style = ""
    def setText(self, text):
        self.text = text
    def setStyleSheet(self, style):
        self.style = style


class _FakeMainViewer:
    def __init__(self, layers):
        self.layers = layers


def _make_annotator_with_set_training_lock():
    """Build a fake annotator object with set_training_lock bound from
    the real class.  Patches the napari.layers.Labels isinstance check."""
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
    ann = MagicMock()
    # Bind real methods.
    ann.set_training_lock = Curation2DAnnotator.set_training_lock.__get__(ann)
    ann._collect_lockable_label_layers = Curation2DAnnotator._collect_lockable_label_layers.__get__(ann)
    ann._lock_per_orient_buttons = Curation2DAnnotator._lock_per_orient_buttons.__get__(ann)
    ann._set_mode_overlay_training_lock = Curation2DAnnotator._set_mode_overlay_training_lock.__get__(ann)
    # Default state: not previously locked.
    ann._training_lock_active = False
    ann._training_lock_snapshot = []
    ann._training_lock_overlay_active = False
    return ann


def test_set_training_lock_disables_per_orient_buttons():
    ann = _make_annotator_with_set_training_lock()
    ann.viewer = _FakeMainViewer(layers=[])  # no main labels
    ann._all_subviewer_label_layers = lambda: []  # no subviewer labels
    ann._mode_status_label = _FakeLabel()
    # Create the 6 per-orient buttons.
    ann.segment_mode_btn = _FakeButton()
    ann.accept_slice_btn = _FakeButton()
    ann.segment_xz_btn = _FakeButton()
    ann.accept_xz_btn = _FakeButton()
    ann.segment_yz_btn = _FakeButton()
    ann.accept_yz_btn = _FakeButton()

    ann.set_training_lock(True)
    for btn_name in (
        "segment_mode_btn", "accept_slice_btn",
        "segment_xz_btn", "accept_xz_btn",
        "segment_yz_btn", "accept_yz_btn",
    ):
        btn = getattr(ann, btn_name)
        assert btn.enabled is False, f"{btn_name} not disabled"


def test_set_training_lock_unlock_re_enables_buttons():
    ann = _make_annotator_with_set_training_lock()
    ann.viewer = _FakeMainViewer(layers=[])
    ann._all_subviewer_label_layers = lambda: []
    ann._mode_status_label = _FakeLabel()
    ann.segment_mode_btn = _FakeButton()
    ann.accept_slice_btn = _FakeButton()
    ann.segment_xz_btn = _FakeButton()
    ann.accept_xz_btn = _FakeButton()
    ann.segment_yz_btn = _FakeButton()
    ann.accept_yz_btn = _FakeButton()
    ann._refresh_segment_button_text = MagicMock()
    ann._update_mode_overlays = MagicMock()

    ann.set_training_lock(True)
    ann.set_training_lock(False)
    # All buttons re-enabled at the time _lock_per_orient_buttons fires;
    # the post-unlock _refresh_segment_button_text re-derives per-bucket
    # state but we mocked it out.
    for btn_name in (
        "segment_mode_btn", "accept_slice_btn",
        "segment_xz_btn", "accept_xz_btn",
        "segment_yz_btn", "accept_yz_btn",
    ):
        btn = getattr(ann, btn_name)
        assert btn.enabled is True, f"{btn_name} not re-enabled"
    # And the refresh helper was called so per-bucket logic re-derives.
    ann._refresh_segment_button_text.assert_called()


def test_set_training_lock_locks_labels_layers(monkeypatch):
    """Subviewer + main-viewer Labels layers get editable=False + pan_zoom."""
    from napari_epicycle.annotator import curation_2d_widget as mod
    # Patch the Labels class to accept _FakeLayer.
    monkeypatch.setattr(mod, "Labels", _FakeLayer, raising=False)
    # The real method does `from napari.layers import Labels` INSIDE the
    # method body, so we have to patch napari.layers.Labels too.
    import napari.layers as _napari_layers
    monkeypatch.setattr(_napari_layers, "Labels", _FakeLayer, raising=False)

    ann = _make_annotator_with_set_training_lock()
    main_labels = _FakeLayer(name="main_labels", editable=True, mode="paint")
    sub_layers = [
        _FakeLayer(name="work_labels_XY", editable=True, mode="paint"),
        _FakeLayer(name="work_labels_XZ", editable=True, mode="paint"),
        _FakeLayer(name="work_labels_YZ", editable=True, mode="paint"),
    ]
    ann.viewer = _FakeMainViewer(layers=[main_labels])
    ann._all_subviewer_label_layers = lambda: sub_layers
    ann._mode_status_label = _FakeLabel()
    ann.segment_mode_btn = _FakeButton()
    ann.accept_slice_btn = _FakeButton()
    ann.segment_xz_btn = _FakeButton()
    ann.accept_xz_btn = _FakeButton()
    ann.segment_yz_btn = _FakeButton()
    ann.accept_yz_btn = _FakeButton()

    ann.set_training_lock(True)
    # Main labels locked.
    assert main_labels.editable is False
    assert main_labels.mode == "pan_zoom"
    # All 3 subviewer labels locked.
    for layer in sub_layers:
        assert layer.editable is False
        assert layer.mode == "pan_zoom"
    # Snapshot stored for restore.
    assert len(ann._training_lock_snapshot) == 4


def test_set_training_lock_unlock_restores_prior_state(monkeypatch):
    """After unlock, each layer's prior editable+mode are restored."""
    from napari_epicycle.annotator import curation_2d_widget as mod
    monkeypatch.setattr(mod, "Labels", _FakeLayer, raising=False)
    import napari.layers as _napari_layers
    monkeypatch.setattr(_napari_layers, "Labels", _FakeLayer, raising=False)

    ann = _make_annotator_with_set_training_lock()
    sub_layers = [
        _FakeLayer(name="work_labels_XY", editable=True, mode="paint"),
        _FakeLayer(name="work_labels_XZ", editable=False, mode="pan_zoom"),
        _FakeLayer(name="work_labels_YZ", editable=True, mode="erase"),
    ]
    ann.viewer = _FakeMainViewer(layers=[])
    ann._all_subviewer_label_layers = lambda: sub_layers
    ann._mode_status_label = _FakeLabel()
    ann.segment_mode_btn = _FakeButton()
    ann.accept_slice_btn = _FakeButton()
    ann.segment_xz_btn = _FakeButton()
    ann.accept_xz_btn = _FakeButton()
    ann.segment_yz_btn = _FakeButton()
    ann.accept_yz_btn = _FakeButton()
    ann._refresh_segment_button_text = MagicMock()
    ann._update_mode_overlays = MagicMock()

    ann.set_training_lock(True)
    # All locked.
    for layer in sub_layers:
        assert layer.editable is False
        assert layer.mode == "pan_zoom"
    ann.set_training_lock(False)
    # Restored to prior values.
    assert sub_layers[0].editable is True
    assert sub_layers[0].mode == "paint"
    assert sub_layers[1].editable is False
    assert sub_layers[1].mode == "pan_zoom"
    assert sub_layers[2].editable is True
    assert sub_layers[2].mode == "erase"


def test_set_training_lock_idempotent_lock(monkeypatch):
    """Calling set_training_lock(True) twice doesn't re-snapshot or
    re-lock — second call is a no-op."""
    from napari_epicycle.annotator import curation_2d_widget as mod
    monkeypatch.setattr(mod, "Labels", _FakeLayer, raising=False)
    import napari.layers as _napari_layers
    monkeypatch.setattr(_napari_layers, "Labels", _FakeLayer, raising=False)

    ann = _make_annotator_with_set_training_lock()
    sub_layers = [_FakeLayer(name="work_labels_XY", editable=True, mode="paint")]
    ann.viewer = _FakeMainViewer(layers=[])
    ann._all_subviewer_label_layers = lambda: sub_layers
    ann._mode_status_label = _FakeLabel()
    ann.segment_mode_btn = _FakeButton()
    ann.accept_slice_btn = _FakeButton()
    ann.segment_xz_btn = _FakeButton()
    ann.accept_xz_btn = _FakeButton()
    ann.segment_yz_btn = _FakeButton()
    ann.accept_yz_btn = _FakeButton()

    ann.set_training_lock(True)
    snapshot1 = list(ann._training_lock_snapshot)
    # Now simulate something that changed the layer between calls.
    sub_layers[0].editable = True  # somebody set it back externally
    ann.set_training_lock(True)  # second call — should NOT re-snapshot
    assert ann._training_lock_snapshot == snapshot1, (
        "second lock(True) call re-snapshotted; original prior state is lost"
    )


def test_set_training_lock_idempotent_unlock(monkeypatch):
    """Calling set_training_lock(False) from idle state is a no-op."""
    from napari_epicycle.annotator import curation_2d_widget as mod
    monkeypatch.setattr(mod, "Labels", _FakeLayer, raising=False)
    import napari.layers as _napari_layers
    monkeypatch.setattr(_napari_layers, "Labels", _FakeLayer, raising=False)

    ann = _make_annotator_with_set_training_lock()
    layer = _FakeLayer(name="work_labels_XY", editable=True, mode="paint")
    ann.viewer = _FakeMainViewer(layers=[])
    ann._all_subviewer_label_layers = lambda: [layer]
    ann._mode_status_label = _FakeLabel()
    ann.segment_mode_btn = _FakeButton()
    ann.accept_slice_btn = _FakeButton()
    ann.segment_xz_btn = _FakeButton()
    ann.accept_xz_btn = _FakeButton()
    ann.segment_yz_btn = _FakeButton()
    ann.accept_yz_btn = _FakeButton()
    ann._refresh_segment_button_text = MagicMock()
    ann._update_mode_overlays = MagicMock()

    # Never locked — calling unlock should be safe.
    ann.set_training_lock(False)
    # Layer must be unchanged (we never touched it).
    assert layer.editable is True
    assert layer.mode == "paint"


def test_set_training_lock_mode_overlay_set(monkeypatch):
    from napari_epicycle.annotator import curation_2d_widget as mod
    monkeypatch.setattr(mod, "Labels", _FakeLayer, raising=False)
    import napari.layers as _napari_layers
    monkeypatch.setattr(_napari_layers, "Labels", _FakeLayer, raising=False)

    ann = _make_annotator_with_set_training_lock()
    ann.viewer = _FakeMainViewer(layers=[])
    ann._all_subviewer_label_layers = lambda: []
    ann._mode_status_label = _FakeLabel()
    ann.segment_mode_btn = _FakeButton()
    ann.accept_slice_btn = _FakeButton()
    ann.segment_xz_btn = _FakeButton()
    ann.accept_xz_btn = _FakeButton()
    ann.segment_yz_btn = _FakeButton()
    ann.accept_yz_btn = _FakeButton()

    ann.set_training_lock(True)
    assert "LOCKED" in ann._mode_status_label.text
    assert "TRAINING" in ann._mode_status_label.text


# ---------------------------------------------------------------------------
# Surface 3 — controller invokes both
# ---------------------------------------------------------------------------


def test_controller_set_training_ui_locked_calls_shell():
    """Source inspection: _set_training_ui_locked must invoke
    shell.lock_step_nav."""
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )
    src = inspect.getsource(TrainingAnnotationController._set_training_ui_locked)
    assert "lock_step_nav" in src, (
        "_set_training_ui_locked must call workflow_shell.lock_step_nav"
    )
    assert "_workflow_shell" in src


def test_controller_set_training_ui_locked_calls_annotator():
    """Source inspection: _set_training_ui_locked must invoke
    annotator.set_training_lock if open."""
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )
    src = inspect.getsource(TrainingAnnotationController._set_training_ui_locked)
    assert "set_training_lock" in src
    assert "_find_open_2d_annotator" in src


def test_controller_find_annotator_uses_dock_helper():
    """Source inspection: _find_open_2d_annotator uses _find_dock_by_title
    with the canonical Step 2 annotator dock title."""
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )
    src = inspect.getsource(TrainingAnnotationController._find_open_2d_annotator)
    assert "_find_dock_by_title" in src
    assert '"EpiCycle 2D Training Annotator"' in src


def test_workflow_shell_injects_workflow_shell_ref():
    """The shell's step-build loop must inject _workflow_shell ref on
    each controller so the lock can be invoked."""
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    src = inspect.getsource(EpiCycleWorkflowShell)
    assert "controller._workflow_shell = self" in src, (
        "workflow_shell must inject self into each controller's "
        "_workflow_shell attribute for the Step 2 training-lock to "
        "reach the side-nav"
    )


# ---------------------------------------------------------------------------
# Scope discipline — do NOT touch Step 3 (Curation3DAnnotator) or base class
# ---------------------------------------------------------------------------


def test_set_training_lock_not_on_base_class():
    """Curation2DAnnotator owns set_training_lock; the base
    NativeMultiviewAnnotator does NOT inherit it.  This protects
    Step 3 (Curation3DAnnotator inherits from the base) from
    accidentally getting Step-2-only behavior."""
    from napari_epicycle.annotator.native_multiview_widget import NativeMultiviewAnnotator
    assert not hasattr(NativeMultiviewAnnotator, "set_training_lock"), (
        "set_training_lock leaked into the base class — Step 3 would "
        "inherit Step-2-only behavior"
    )


def test_set_training_lock_not_on_3d_subclass():
    """Curation3DAnnotator inherits from the base, NOT from the 2D
    subclass.  It must not have set_training_lock."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    # The method may resolve through MRO to the base (which doesn't
    # have it), but it must NOT resolve to the 2D subclass.
    # Easiest check: the 3D class's __mro__ must not include
    # Curation2DAnnotator.
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
    assert Curation2DAnnotator not in Curation3DAnnotator.__mro__, (
        "Curation3DAnnotator inherits from Curation2DAnnotator — Step 2 "
        "behavior would leak into Step 3"
    )


# ---------------------------------------------------------------------------
# Audit-fix #3 — Race-condition fix: blockSignals + _step_nav_locked guard
# ---------------------------------------------------------------------------


class _FakeNavItemWithSignals(_FakeNavItem):
    """Like _FakeNavItem but also records blockSignals calls."""
    def __init__(self):
        super().__init__()
        self.signals_blocked = False

    def blockSignals(self, blocked: bool) -> bool:
        prev = self.signals_blocked
        self.signals_blocked = bool(blocked)
        return prev


def test_lock_step_nav_calls_block_signals():
    """Audit fix #3: lock_step_nav must call blockSignals(True) on each
    nav item to cancel any queued click signals."""
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    shell = MagicMock()
    shell._nav_items = {f"step_{i}": _FakeNavItemWithSignals() for i in range(3)}
    shell.lock_step_nav = EpiCycleWorkflowShell.lock_step_nav.__get__(shell)
    shell.lock_step_nav(True)
    for item in shell._nav_items.values():
        assert item.signals_blocked is True
    shell.lock_step_nav(False)
    for item in shell._nav_items.values():
        assert item.signals_blocked is False


def test_lock_step_nav_sets_step_nav_locked_flag():
    """Audit fix #3: lock_step_nav must set self._step_nav_locked so
    _select_step has a defensive guard."""
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    shell = MagicMock()
    shell._nav_items = {"step_0": _FakeNavItem()}
    shell.lock_step_nav = EpiCycleWorkflowShell.lock_step_nav.__get__(shell)
    shell.lock_step_nav(True)
    assert shell._step_nav_locked is True
    shell.lock_step_nav(False)
    assert shell._step_nav_locked is False


def test_select_step_blocks_when_step_nav_locked():
    """Audit fix #3: _select_step must early-return when
    `_step_nav_locked` is True, so a queued click can't navigate."""
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    shell = MagicMock()
    shell._step_index_by_id = {"step_a": 0, "step_b": 1}
    shell._step_nav_locked = True
    shell._select_step = EpiCycleWorkflowShell._select_step.__get__(shell)
    # Should early-return and NOT touch _active_step_id.
    shell._active_step_id = "step_a"
    shell._select_step("step_b")
    # _active_step_id should not have changed.
    assert shell._active_step_id == "step_a"


def test_select_step_works_when_step_nav_unlocked():
    """When `_step_nav_locked` is False (or absent), _select_step
    must proceed normally — verify the guard doesn't break the
    normal navigation path."""
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    # We can't run the full _select_step body without a real Qt setup,
    # but we can verify the early-return guard does NOT fire when
    # _step_nav_locked is False.  Do this by source-inspection.
    src = inspect.getsource(EpiCycleWorkflowShell._select_step)
    assert "_step_nav_locked" in src, (
        "_select_step must check _step_nav_locked"
    )
    # And the check must be near the top (before any state mutation).
    lines = src.splitlines()
    guard_idx = next(
        (i for i, l in enumerate(lines) if "_step_nav_locked" in l),
        -1,
    )
    # Within the first 20 lines of the method body.
    assert 0 < guard_idx < 20, (
        f"_step_nav_locked check at line {guard_idx} — should be near "
        "the top of _select_step to short-circuit before any state changes"
    )


# ---------------------------------------------------------------------------
# Audit-fix #8 — _apply_edit_constraints override on Curation2DAnnotator
# ---------------------------------------------------------------------------


def test_apply_edit_constraints_overridden_in_2d_subclass():
    """Audit fix #8: Curation2DAnnotator must override
    _apply_edit_constraints to early-return when the training lock is
    active.  Otherwise napari layer events could un-do the lock."""
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
    from napari_epicycle.annotator.native_multiview_widget import NativeMultiviewAnnotator
    # The 2D subclass must DEFINE its own _apply_edit_constraints
    # (not just inherit from the base).
    assert "_apply_edit_constraints" in Curation2DAnnotator.__dict__, (
        "Curation2DAnnotator must override _apply_edit_constraints "
        "to honor the training lock"
    )
    # And the base must still HAVE _apply_edit_constraints (we depend
    # on super() to call it when not locked).
    assert hasattr(NativeMultiviewAnnotator, "_apply_edit_constraints"), (
        "base class missing _apply_edit_constraints — super() call "
        "in the override would crash"
    )


def test_apply_edit_constraints_3d_subclass_unaffected():
    """Audit fix #8: Curation3DAnnotator must NOT inherit the
    2D-subclass-override.  Step 3's MRO goes through the base only."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
    from napari_epicycle.annotator.native_multiview_widget import NativeMultiviewAnnotator
    # Resolve _apply_edit_constraints on the 3D subclass and confirm
    # it comes from the base, not the 2D subclass.
    method = Curation3DAnnotator._apply_edit_constraints
    base_method = NativeMultiviewAnnotator._apply_edit_constraints
    twod_method = Curation2DAnnotator._apply_edit_constraints
    assert method is base_method, (
        "Curation3DAnnotator inherited an overridden _apply_edit_constraints "
        "— Step 2 behavior leaked into Step 3"
    )
    assert method is not twod_method, (
        "Curation3DAnnotator's _apply_edit_constraints resolves to the "
        "Step 2 subclass override — Step 2 behavior leaked into Step 3"
    )


def test_apply_edit_constraints_skips_when_locked():
    """When `_training_lock_active=True`, _apply_edit_constraints
    must early-return WITHOUT calling super()."""
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
    ann = MagicMock(spec=Curation2DAnnotator)
    ann._training_lock_active = True
    # Bind the real method.
    ann._apply_edit_constraints = Curation2DAnnotator._apply_edit_constraints.__get__(ann)
    # Patch super() — if it gets called, we'd see the side effect.
    # Use a flag instead since MagicMock-spec'd super() is complicated.
    called = {"super": False}
    # We can't easily intercept super().method() but the override has
    # a print() statement we can capture instead.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ann._apply_edit_constraints()
    assert "BLOCKED" in buf.getvalue(), (
        "When _training_lock_active=True, _apply_edit_constraints "
        "should print BLOCKED + early-return"
    )
