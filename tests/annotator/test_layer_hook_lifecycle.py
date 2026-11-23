"""The annotator's layer-list hooks must not outlive, or break, a close."""

from __future__ import annotations

import types

import numpy as np
from napari.components import ViewerModel
from napari.layers import Labels

from napari_epicycle.annotator import native_multiview_widget as nmw
from napari_epicycle.annotator.native_multiview_widget import (
    NativeMultiviewAnnotator as N,
)


class _Ann:
    pass


def _make_ann(vm):
    ann = _Ann()
    ann.viewer = vm
    ann.image_combo = object()
    ann.labels_combo = object()
    ann.patch_context_combo = object()
    ann.status_label = object()
    ann.refreshed = 0
    ann.removed_seen = 0
    ann._refresh_layer_options = lambda: setattr(ann, "refreshed", ann.refreshed + 1)
    ann._on_viewer_layers_changed = types.MethodType(N._on_viewer_layers_changed, ann)
    ann._on_main_layer_removed = lambda e=None: setattr(
        ann, "removed_seen", ann.removed_seen + 1
    )
    ann._viewer_inserted_cb = ann._on_viewer_layers_changed
    ann._viewer_removed_cb = ann._on_viewer_layers_changed
    ann._dims_xy_cb = ann._dims_xz_cb = ann._dims_yz_cb = None
    ann._disconnect_rename_handlers = lambda: None
    ann.layer_manager = types.SimpleNamespace(disconnect_layer_listeners=lambda: None)
    ann.sync_manager = types.SimpleNamespace(sync_active_layer=lambda *a: None)
    ann.connect = types.MethodType(N._connect_layer_list_hooks, ann)
    ann.disconnect_combo = types.MethodType(N._disconnect_combo_refresh_hooks, ann)
    return ann


def _real_disconnect(ann):
    try:
        N._disconnect_viewer_events(ann)
    except AttributeError:
        # The trailing part of the real method needs widgets this fake lacks;
        # the layer-list part has already run by then.
        pass


def _counts(vm):
    ev = vm.layers.events
    return len(ev.removed.callbacks), len(ev.inserted.callbacks)


def _lbl():
    return Labels(np.zeros((2, 4, 4), dtype=np.uint8))


def test_open_close_cycles_do_not_accumulate():
    vm = ViewerModel()
    base = _counts(vm)
    seen = []
    for _ in range(3):
        ann = _make_ann(vm)
        ann.connect()
        seen.append(_counts(vm))
        _real_disconnect(ann)
    assert len(set(seen)) == 1
    assert _counts(vm) == base


def test_aborted_close_is_purged_by_next_connect():
    vm = ViewerModel()
    base = _counts(vm)
    old = _make_ann(vm)
    old.connect()  # never disconnected
    new = _make_ann(vm)
    new.connect()
    lay = _lbl()
    vm.layers.append(lay)
    vm.layers.remove(lay)
    assert old.refreshed == 0 and old.removed_seen == 0
    assert new.removed_seen == 1
    _real_disconnect(new)
    assert _counts(vm) == base


def test_orphaned_widget_does_not_raise_or_block_later_handlers():
    vm = ViewerModel()
    ann = _make_ann(vm)
    ann.connect()
    later = []
    vm.layers.events.removed.connect(lambda e: later.append(1))
    ann.image_combo = None
    lay = _lbl()
    vm.layers.append(lay)
    vm.layers.remove(lay)
    assert later == [1]
    assert ann.refreshed == 0


def test_set_parent_none_drops_combo_hooks_keeps_main_removed():
    vm = ViewerModel()
    ann = _make_ann(vm)
    ann.connect()
    ann.disconnect_combo()
    ann.image_combo = None
    ann.labels_combo = None
    lay = _lbl()
    vm.layers.append(lay)
    vm.layers.remove(lay)
    assert ann.removed_seen == 1
    ledger = getattr(vm.layers, nmw._LAYER_HOOKS_ATTR)[1]
    assert [c for _e, c in ledger] == [ann._on_main_layer_removed]
    _real_disconnect(ann)
    assert _counts(vm) == _counts(ViewerModel())


def test_set_parent_override_calls_the_combo_hook_disconnect():
    import inspect

    src = inspect.getsource(N.setParent)
    assert src.index("_disconnect_combo_refresh_hooks") < src.index("Phase 2: null")


def test_main_layer_removed_tolerates_orphaned_status_label():
    vm = ViewerModel()
    ann = _make_ann(vm)
    ann.status_label = None
    ann._managed_layer_names = {"x"}
    ann._active_image_name = ""
    ann._active_labels_name = ""
    ann.sync_manager = types.SimpleNamespace(unregister_layer=lambda n: None)
    ann.subviewer_manager = types.SimpleNamespace(clear=lambda: None)
    ann._clear_all_overlays = lambda: None
    ann._toggle_crosshairs = lambda v: ann.status_label.setText("boom")
    ann._configure_slice_sliders = lambda: None
    lay = _lbl()
    lay.name = "x"
    N._on_main_layer_removed(ann, types.SimpleNamespace(value=lay))


def test_layer_changed_guard_covers_all_ui_refs():
    vm = ViewerModel()
    ann = _make_ann(vm)
    ann.patch_context_combo = None
    ann.status_label = object()
    ann._on_viewer_layers_changed()
    assert ann.refreshed == 0


def test_2d_and_launcher_inserted_hooks_use_the_ledger():
    import inspect

    from napari_epicycle.annotator import curation_2d_widget as c2
    from napari_epicycle.annotator import launcher

    assert "register_layer_hook" in inspect.getsource(c2.Curation2DAnnotator.__init__)
    assert "register_layer_hook" in inspect.getsource(launcher)


def _cleanup_fake(vm, raises):
    ann = _make_ann(vm)
    ann.hidden = False
    ann.hide = lambda: setattr(ann, "hidden", True)
    ann._close_subviewers = lambda: None

    def _boom():
        raise RuntimeError("boom")

    ann._close_cleanup = _boom if raises else (lambda: _real_disconnect(ann))
    return ann


def _ledger_owner(vm):
    stored = getattr(vm.layers, nmw._LAYER_HOOKS_ATTR, None)
    return stored[0] if stored else None


def test_cleanup_that_raises_is_named_and_own_hooks_dropped(capsys):
    from napari_epicycle.annotator import launcher

    vm = ViewerModel()
    base = _counts(vm)
    ann = _cleanup_fake(vm, raises=True)
    ann.connect()
    assert _counts(vm) != base
    launcher._run_widget_cleanup(ann)
    out = capsys.readouterr().out
    assert "[CLOSE-CLEANUP]" in out and "raised" in out
    assert "[HOOK-DROP] _Ann teardown dropped" in out
    assert _ledger_owner(vm) != id(ann)
    assert _counts(vm) == base


def test_normal_close_prints_no_hook_drop(capsys):
    from napari_epicycle.annotator import launcher

    vm = ViewerModel()
    ann = _cleanup_fake(vm, raises=False)
    ann.connect()
    launcher._run_widget_cleanup(ann)
    out = capsys.readouterr().out
    assert "[HOOK-DROP]" not in out and "[CLOSE-CLEANUP]" not in out
    assert nmw.drop_own_layer_hooks(ann) == 0


def test_drop_leaves_another_owners_hooks(capsys):
    from napari_epicycle.annotator import launcher

    vm = ViewerModel()
    a = _cleanup_fake(vm, raises=True)
    b = _make_ann(vm)
    b.connect()  # ledger now belongs to b
    launcher._run_widget_cleanup(a)
    assert _ledger_owner(vm) == id(b)
    assert getattr(vm.layers, nmw._LAYER_HOOKS_ATTR)[1]
    assert "[HOOK-DROP]" not in capsys.readouterr().out


def test_cleanup_fake_without_viewer_does_not_raise():
    from napari_epicycle.annotator import launcher

    fake = types.SimpleNamespace(hide=lambda: None)
    launcher._run_widget_cleanup(fake)
    assert nmw.drop_own_layer_hooks(fake) == 0
