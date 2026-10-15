"""Tests for Step 1 controller — cumulative naming + export safety.

Uses real filesystem for export tests, lightweight stubs for controller logic.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation.logic import (
    extract_patches_from_shapes,
    find_next_patch_index,
)
from napari_epicycle.data_preparation._largedata import open_lazy, materialize


# ---------------------------------------------------------------------------
# Cumulative naming
# ---------------------------------------------------------------------------

class TestCumulativeNaming:
    """Verify patches get cumulative names that don't collide."""

    def _box(self, y0, x0, y1, x1):
        return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

    def test_naming_with_display_name_prefix(self):
        img = np.zeros((2, 4, 16, 16))
        boxes = [self._box(0, 0, 8, 8)]
        patches = extract_patches_from_shapes(
            img, boxes, name_prefix="kidney_patch", start_index=1
        )
        assert "kidney_patch_1" in patches

    def test_continues_from_workspace(self):
        """If workspace already has kidney_patch_1/2, next extraction starts at 3."""
        with tempfile.TemporaryDirectory() as workspace:
            for name in ["kidney_patch_1.tiff", "kidney_patch_2.tiff"]:
                open(os.path.join(workspace, name), "w").close()

            start = find_next_patch_index(workspace, "kidney_patch")
            assert start == 3

            img = np.zeros((2, 4, 16, 16))
            boxes = [self._box(0, 0, 8, 8), self._box(8, 0, 16, 8)]
            patches = extract_patches_from_shapes(
                img, boxes, name_prefix="kidney_patch", start_index=start
            )
            assert "kidney_patch_3" in patches
            assert "kidney_patch_4" in patches

    def test_different_prefixes_are_independent(self):
        """kidney_patch and liver_patch numbering don't interfere."""
        with tempfile.TemporaryDirectory() as workspace:
            open(os.path.join(workspace, "kidney_patch_1.tiff"), "w").close()
            open(os.path.join(workspace, "liver_patch_1.tiff"), "w").close()
            open(os.path.join(workspace, "liver_patch_2.tiff"), "w").close()

            assert find_next_patch_index(workspace, "kidney_patch") == 2
            assert find_next_patch_index(workspace, "liver_patch") == 3

    def test_no_display_name_falls_back_to_patch(self):
        """When no display name is set, prefix defaults to 'patch'."""
        img = np.zeros((2, 4, 16, 16))
        boxes = [self._box(0, 0, 8, 8)]
        patches = extract_patches_from_shapes(img, boxes)
        assert "patch_1" in patches


# ---------------------------------------------------------------------------
# Source display-layer removal / re-show on step exit / enter (v2 item 6)
# ---------------------------------------------------------------------------

class _FakeLayer:
    def __init__(self, name):
        self.name = name
        self.visible = True


class _FakeLayerList(list):
    """The slice of ``viewer.layers`` the source-layer helpers use: ``name in
    layers``, ``layers[name]``, and ``.remove(layer)``."""

    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(l, "name", None) == item for l in self)
        return list.__contains__(self, item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if l.name == key:
                    return l
            raise KeyError(key)
        return list.__getitem__(self, key)

    def remove(self, layer):
        list.remove(self, layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()

    def add_image(self, data, name=None, colormap=None, blending=None):
        layer = _FakeLayer(name)
        self.layers.append(layer)
        return layer


class _FakeCtl:
    """A bare stand-in with the three source-layer helpers bound off the real
    controller (so `_reshow_source_layers`'s `self._add_source_display_layers`
    call resolves), plus just the state / hooks they touch."""

    _HELPERS = ("_add_source_display_layers", "_remove_source_layers_from_viewer",
                "_reshow_source_layers", "progressive_wanted",
                # `_add_source_display_layers` now seeds the object->name map via
                # this helper (rename-sync fix), so the fake must bind it too.
                "_track_source_layer_objects",
                # `_add_source_display_layers` is now a guard WRAPPER (Bug A) that
                # delegates to `_do_add_source_display_layers` — bind it too.
                "_do_add_source_display_layers")

    def __init__(self):
        import types

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C)
        self.viewer = _FakeViewer()
        self._loaded_source_layers = []
        self._source_display_channels = None
        for name in self._HELPERS:
            setattr(self, name, types.MethodType(getattr(C, name), self))

    def _log(self, *_a, **_k):
        pass

    def _on_layers_changed(self, *_a, **_k):
        pass

    # Reshow now also drives the progressive-swap + 3D-detail refresh trio; these
    # no-op stubs let `_reshow_source_layers` run to completion under the fake so
    # its channel-selection-restore tail is exercised (a normal ndarray source
    # makes the real trio no-op anyway, via `_is_multiscale_store_displayed`).
    def _maybe_swap_progressive_on_ndisplay(self, *_a, **_k):
        pass

    def _refresh_3d_detail_for_ndisplay(self, *_a, **_k):
        pass

    def _apply_3d_detail_level(self, *_a, **_k):
        pass


class TestSourceLayerExitReshow:
    """Item 6: step-exit REMOVES the source Image layers (so they don't strand
    on the shared canvas and abort Step 3's annotator), keeping the channel refs
    so step-enter re-shows them — the extract path (which stacks the live
    display channels) must keep working on return."""

    def _channels(self):
        return [("ch0", np.zeros((3, 4, 5))), ("ch1", np.zeros((3, 4, 5)))]

    def _loaded_ctl(self):
        ctl = _FakeCtl()
        chans = self._channels()
        ctl._loaded_source_layers = ctl._add_source_display_layers(chans)
        ctl._source_display_channels = chans
        return ctl

    def test_add_source_display_layers_adds_one_per_channel(self):
        ctl = self._loaded_ctl()
        assert ctl._loaded_source_layers == ["ch0", "ch1"]
        assert len(ctl.viewer.layers) == 2

    def test_remove_clears_viewer_but_keeps_refs(self):
        ctl = self._loaded_ctl()
        ctl._remove_source_layers_from_viewer()
        assert len(ctl.viewer.layers) == 0                  # gone from canvas
        assert ctl._loaded_source_layers == ["ch0", "ch1"]  # names kept
        assert ctl._source_display_channels is not None     # refs kept

    def test_reshow_restores_the_layers(self):
        ctl = self._loaded_ctl()
        ctl._remove_source_layers_from_viewer()
        ctl._reshow_source_layers()
        assert len(ctl.viewer.layers) == 2
        assert [l.name for l in ctl.viewer.layers] == ["ch0", "ch1"]

    def test_reshow_is_idempotent_when_already_present(self):
        ctl = self._loaded_ctl()
        ctl._reshow_source_layers()                         # never removed
        assert len(ctl.viewer.layers) == 2                  # no duplicates

    def test_reshow_noop_when_source_unloaded(self):
        ctl = _FakeCtl()                                    # no channels
        ctl._reshow_source_layers()
        assert len(ctl.viewer.layers) == 0


# ---------------------------------------------------------------------------
# Channel selection survives the Step-1 navigate-away / return round-trip
# ---------------------------------------------------------------------------
#
# Driven through the REAL `DataPreparationWidget` (real `image_layer_combo` +
# real `update_image_layer_list`) with a viewer whose `add_image` builds a REAL
# napari Image layer and whose layer list FIRES inserted/removed events — so the
# controller's real `_on_layers_changed` -> `update_image_layer_list` clearing
# path runs on every add/remove, exactly as in the app.  The `_FakeCtl` suite
# above cannot catch this: its `_on_layers_changed` is a `pass` stub, so the
# combo-clearing path never runs.


@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    yield QApplication([])


class _FiringEvent:
    """napari-style event: stores connected callbacks, invokes them when fired
    (so append/remove drive the controller's real `_on_layers_changed`)."""

    def __init__(self):
        self._cbs = []

    def connect(self, cb):
        self._cbs.append(cb)
        return cb

    def disconnect(self, cb=None):
        if cb is None:
            self._cbs.clear()
        elif cb in self._cbs:
            self._cbs.remove(cb)

    def __call__(self, event=None):
        for cb in list(self._cbs):
            cb(event)


class _LayerEvent:
    def __init__(self, value):
        self.value = value


class _FiringLayerEvents:
    def __init__(self):
        self.inserted = _FiringEvent()
        self.removed = _FiringEvent()


class _EventFiringLayerList(list):
    """`viewer.layers` that fires inserted/removed on append/remove (like napari)
    and looks layers up by name — the two accessors the source-layer helpers and
    `get_combined_image_stack` use."""

    def __init__(self):
        super().__init__()
        self.events = _FiringLayerEvents()

    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(l, "name", None) == item for l in self)
        return list.__contains__(self, item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if getattr(l, "name", None) == key:
                    return l
            raise KeyError(key)
        return list.__getitem__(self, key)

    def append(self, layer):
        list.append(self, layer)
        self.events.inserted(_LayerEvent(layer))

    def remove(self, layer):
        list.remove(self, layer)
        self.events.removed(_LayerEvent(layer))


class _RealImageViewer:
    """Viewer whose `add_image` builds a REAL napari Image layer (so the
    controller's `isinstance(layer, Image)` filter in `_on_layers_changed`
    passes) and whose layer list fires events."""

    def __init__(self):
        self.layers = _EventFiringLayerList()
        self.mouse_drag_callbacks = []

    def add_image(self, data, name=None, colormap=None, blending=None):
        from napari.layers import Image

        layer = Image(np.asarray(data), name=name,
                      colormap=colormap or "gray",
                      blending=blending or "additive")
        self.layers.append(layer)
        return layer


class TestChannelSelectionSurvivesStepRoundTrip:
    """H1 (2026-08-17 audit): inclusion moved to the checkbox `_channel_included`
    dict (A1).  `on_step_exit` REMOVES the source Image layers; if that removal
    ran UNGUARDED it would fire `_on_layers_changed` -> `update_image_layer_list`
    with a shrinking/empty name set and reconcile the dict to {} → on return a
    deliberate channel UNTICK would silently revert to included (Extract stacks a
    channel the user excluded).  The fix removes UNDER the
    `_swapping_source_display` guard (mirroring the raw⇄processed swaps), so the
    inclusion dict survives the round-trip; the re-show reconciles once at the
    settled full count (`_loading_source_layers` guard), preserving every
    untick."""

    def _channels(self):
        return [("ch0", np.zeros((3, 4, 5), dtype=np.uint16)),
                ("ch1", np.ones((3, 4, 5), dtype=np.uint16))]

    def _load(self, ctl):
        chans = self._channels()
        ctl._loaded_source_layers = ctl._add_source_display_layers(chans)
        ctl._source_display_channels = chans

    def test_ticked_channels_survive_exit_reenter(self, qt_app):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget

        w = DataPreparationWidget(viewer=_RealImageViewer())
        try:
            ctl = w.controller
            self._load(ctl)
            combo = w.image_layer_combo
            assert combo.count() == 2
            # All channels are included by default (A1) — both extract.
            _stack, names = ctl.get_combined_image_stack()
            assert names == ["ch0", "ch1"]

            # Deliberately EXCLUDE ch0 (untick).
            w._channel_included["ch0"] = False
            assert w.checked_channel_names() == ["ch1"]

            # -- navigate away: source layers removed UNDER the guard, so the
            #    inclusion dict is NOT reconciled to {} mid-removal (H1).
            ctl._remove_source_layers_from_viewer()
            assert len(ctl.viewer.layers) == 0

            # -- navigate back: re-show source layers (reconciles once at the
            #    settled count).
            ctl._reshow_source_layers()
            assert combo.count() == 2

            # THE H1 FIX: ch0 stays EXCLUDED across the round-trip — Extract does
            # NOT silently re-include the channel the user excluded.
            assert w.checked_channel_names() == ["ch1"]
            _stack2, names2 = ctl.get_combined_image_stack()
            assert names2 == ["ch1"]
        finally:
            w.setParent(None)


def test_recipe_intensity_ops_survive_2d3d_swap_churn(qt_app):
    """2D<->3D swap-churn regression (2026-08-14, two independent traces): the
    progressive swap removes+re-adds source layers ONE AT A TIME; each remove
    fires the real `_on_layers_changed` at N-1 channels, and `normalize_recipe`
    truncates + re-pads-EMPTY the trailing channel's intensity ops.  Net: the
    columns + the shared geometry chip SURVIVE, but a channel's per-channel
    intensity steps (Gaussian/CLAHE/...) get WIPED.  The `_swapping_source_display`
    guard must suppress the transient reconciles so the recipe reconciles ONCE at
    the settled full count and every channel's ops survive."""
    from napari_epicycle.data_preparation.ui import DataPreparationWidget
    from packages.epicycle_preprocessing.recipe import (
        Recipe, IntensityStep, GeometryStep, default_geometry_config)

    w = DataPreparationWidget(viewer=_RealImageViewer())
    try:
        ctl = w.controller
        chans = [("ch0", np.zeros((3, 4, 5), np.uint16)),
                 ("ch1", np.ones((3, 4, 5), np.uint16))]
        ctl._loaded_source_layers = ctl._add_source_display_layers(chans)
        ctl._source_display_channels = chans
        # Ops on BOTH channels + a shared geometry step (geometry must survive).
        w._recipe = Recipe(
            channels=[[IntensityStep("gaussian", {"sigma": 1.5})],
                      [IntensityStep("clahe", {"clip": 0.01, "kernel": 64})]],
            geometry=GeometryStep("isotropic", 1, default_geometry_config()))
        w.update_image_layer_list(["ch0", "ch1"])
        assert [len(c) for c in w._recipe.channels] == [1, 1]
        assert w._recipe.geometry is not None

        # Reproduce the swap's per-channel remove/add churn UNDER the guard.
        ctl._swapping_source_display = True
        try:
            for name in ("ch0", "ch1"):
                ctl.viewer.layers.remove(ctl.viewer.layers[name])  # removed -> _on_layers_changed
                ctl.viewer.add_image(np.ones((3, 4, 5), np.uint16), name=name)
        finally:
            ctl._swapping_source_display = False
        ctl._on_layers_changed()  # single settled reconcile (flag clear)

        # THE FIX: both channels' intensity ops survive the churn; geometry intact.
        assert [len(c) for c in w._recipe.channels] == [1, 1], (
            "intensity ops were wiped by the transient N-1 normalize during the "
            "swap churn — the _swapping_source_display guard must suppress it")
        assert w._recipe.geometry is not None
    finally:
        w.setParent(None)


def test_teardown_stales_in_flight_process(qt_app):
    """Grill HIGH-1: navigating away / switching source while a whole-volume
    Process is in flight must STALE it — the worker's late finish must NOT render
    ⟨processed⟩ layers onto another step's canvas (the Step-3 all-Image-layers
    abort) and must not leave the busy-lock stuck.  `_teardown_preview` (the
    funnel for step-exit / source-switch / unload) cancels the in-flight run:
    bumps `_process_token`, drops `_processed_cache`, releases `_processing`."""
    from napari_epicycle.data_preparation.ui import DataPreparationWidget

    w = DataPreparationWidget(viewer=_RealImageViewer())
    try:
        ctl = w.controller
        # Simulate an in-flight process (a sentinel thread; no real worker).
        ctl._process_thread = object()
        ctl._process_worker = None
        ctl._processing = True
        ctl._processed_cache = {"processed": [np.zeros((2, 2, 2), np.uint16)]}
        ctl._set_step1_busy(True)
        tok_before = ctl._process_token

        ctl._teardown_preview()

        assert ctl._process_token > tok_before, "teardown must bump the process token"
        assert ctl._processed_cache is None, "teardown must drop the cached volume"
        assert ctl._processing is False, "teardown must release the processing flag"
        assert ctl._step1_busy is False, "teardown must release the busy-lock"

        # THE crux: the worker's late finish (stamped with the pre-teardown token)
        # must early-return — no ⟨processed⟩ layers added to the viewer.
        n_before = len(ctl.viewer.layers)
        ctl._on_process_computed(
            [np.zeros((2, 2, 2), np.uint16)], (1.0, 1.0, 1.0), [None], {},
            ["ch0"], {"channels": [[]], "geometry": None}, tok_before)
        assert len(ctl.viewer.layers) == n_before, (
            "a stale (pre-teardown) process finish must render nothing")
        assert ctl._processed_cache is None, "stale finish must not repopulate cache"
    finally:
        w.setParent(None)


def test_recipe_survives_source_load_channel_churn(qt_app):
    """Bug A: `_add_source_display_layers` adds channels one at a time; each
    add fires the real `_on_layers_changed` → `normalize_recipe(recipe, k)` at the
    transient k, truncating the TRAILING channel's recipe ops then re-padding it
    EMPTY.  The `_loading_source_layers` guard must suppress the churn so the recipe
    reconciles ONCE at the full count and ch1's steps survive a load / re-show."""
    from napari_epicycle.data_preparation.ui import DataPreparationWidget
    from packages.epicycle_preprocessing.recipe import (
        Recipe, IntensityStep, GeometryStep, default_geometry_config)

    w = DataPreparationWidget(viewer=_RealImageViewer())
    try:
        ctl = w.controller
        # ch0 has 1 op, ch1 (the tail — Bug A's victim) has 2 + a shared geometry.
        w._recipe = Recipe(
            channels=[[IntensityStep("gaussian", {"sigma": 1.5})],
                      [IntensityStep("clahe", {"clip": 0.01, "kernel": 64}),
                       IntensityStep("background", {"radius": 15})]],
            geometry=GeometryStep("isotropic", 1, default_geometry_config()))
        chans = [("ch0", np.zeros((3, 4, 5), np.uint16)),
                 ("ch1", np.ones((3, 4, 5), np.uint16))]

        ctl._add_source_display_layers(chans)

        # Both channels' ops survive the per-add churn (ch1 was wiped to [] before).
        assert [len(c) for c in w._recipe.channels] == [1, 2], (
            "ch1's recipe ops were truncated by the load-time normalize churn — "
            "the _loading_source_layers guard must suppress it")
        assert w._recipe.geometry is not None
    finally:
        w.setParent(None)


def test_shutdown_flushes_pending_source_save(qt_app):
    """Bug B (close): a plain napari close routes viewer.events.close /
    widget.destroyed → `shutdown()`.  It must FLUSH the 400ms debounced source-save
    or the last 1-2 recipe edits made just before close never reach disk."""
    from napari_epicycle.data_preparation.ui import DataPreparationWidget

    w = DataPreparationWidget(viewer=_RealImageViewer())
    try:
        ctl = w.controller
        called = []
        ctl._flush_source_state_save = lambda *a, **k: called.append(True)
        ctl.shutdown()
        assert called, "shutdown() must flush the pending debounced source save"
    finally:
        w.setParent(None)


def test_select_source_flushes_before_restore(qt_app):
    """Bug B (in-session switch): switching sources must FLUSH the outgoing
    source's pending debounced save BEFORE restoring the incoming one — else a
    switch within the 400ms window restores over un-flushed edits and a later edit
    persists the stale copy, permanently losing the outgoing source's last steps."""
    from napari_epicycle.data_preparation.ui import DataPreparationWidget

    w = DataPreparationWidget(viewer=_RealImageViewer())
    try:
        ctl = w.controller
        order = []
        ctl._selected_source_path = None
        w.output_file_path.setText("")  # empty ws → load_state None → row None
        ctl._flush_source_state_save = lambda *a, **k: order.append("flush")
        ctl._restore_source_widgets_for = lambda *a, **k: order.append("restore")
        order.clear()  # ignore any stray flush from the ws-field wiring during setup
        ctl._select_source("/fake/new_source.tif")
        assert order[:2] == ["flush", "restore"], (
            f"flush must precede restore, got {order}")
    finally:
        w.setParent(None)

    def test_reenter_without_exit_invents_no_selection(self, qt_app):
        """The in-Step-1 flow (no navigation, nothing captured) is unaffected: a
        stray re-show restores nothing and does not fabricate a selection."""
        from napari_epicycle.data_preparation.ui import DataPreparationWidget

        w = DataPreparationWidget(viewer=_RealImageViewer())
        try:
            ctl = w.controller
            self._load(ctl)
            combo = w.image_layer_combo
            assert combo.count() == 2
            assert combo.selectedItems() == []          # nothing ticked yet
            ctl._reshow_source_layers()                 # layers already present
            assert combo.count() == 2
            assert combo.selectedItems() == []          # no selection invented
        finally:
            w.setParent(None)


# ---------------------------------------------------------------------------
# Export overwrite protection
# ---------------------------------------------------------------------------

class TestExportOverwriteProtection:
    """Verify export logic handles existing files safely."""

    def test_export_writes_tiff_and_json(self):
        """Real export (persist_patch) writes both files into artifacts/step1/patches."""
        from napari_epicycle.data_preparation._export import persist_patch
        from napari_epicycle.data_preparation._paths import step1_patches_dir

        with tempfile.TemporaryDirectory() as workspace:
            data = np.random.randint(0, 255, (2, 5, 16, 16), dtype=np.uint8)
            res = persist_patch(
                workspace, "test_patch_1", data,
                channel_names=["ch0", "ch1"],
                original_shape=(2, 5, 16, 16),
                cropped_shape=(2, 5, 16, 16),
                selected_z_range=[0, 5],
                bounding_box=[[0, 0], [0, 16], [16, 16], [16, 0]],
                source_path="D:/src", source_display_name="s", timestamp="t",
            )

            pdir = step1_patches_dir(workspace)
            assert (pdir / "test_patch_1.tiff").exists()
            assert (pdir / "test_patch_1_metadata.json").exists()
            # No longer written to the workspace root.
            assert not os.path.exists(os.path.join(workspace, "test_patch_1.tiff"))

            loaded = tifffile.imread(res["tiff_path"])
            assert loaded.shape == data.shape
            meta = json.loads(open(res["sidecar_path"]).read())
            assert meta["patch_name"] == "test_patch_1"
            assert meta["channel_names"] == ["ch0", "ch1"]
            # (Frozen-schema field set is asserted for real in test_export.py.)


# ---------------------------------------------------------------------------
# ④'s ONE export control — the gates around a permanent write
# ---------------------------------------------------------------------------

class TestExportGates:
    """`_on_export_clicked` is the only export, so its gates ARE the design.

    v4.0 consolidated ④ to one tick-driven control and justified the extra
    click on "an action that is permanent and already requires a confirm
    listing what will be written".  Three of those guarantees were missing:
    an empty tick set fell back to `currentItem()` and wrote anyway, no
    confirm existed, and nothing stated the scope.  Driven through the real
    handler on a stub widget rather than the pure helpers, because the defect
    lived in the handler's control flow.
    """

    def _controller(self, checked, confirm=True):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )

        calls = {"exported": [], "current": 0, "confirmed": 0, "status": []}

        class _Widget:
            def checked_patch_names(self):
                return list(checked)

        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl.widget = _Widget()
        ctl._step1_busy = False
        ctl._confirm_export = lambda names: (
            calls.__setitem__("confirmed", calls["confirmed"] + 1) or confirm
        )
        ctl._export_named_patches = lambda names: calls["exported"].append(
            list(names))
        ctl._export_current_patch = lambda: calls.__setitem__(
            "current", calls["current"] + 1)
        ctl._set_export_status = lambda msg, success=True: calls[
            "status"].append((msg, success))
        return ctl, calls

    def test_empty_tick_set_writes_nothing(self):
        """Untick everything, press Export → nothing is written.

        The old `names or None` made "nothing ticked" indistinguishable from
        "no list", so the handler fell through to the selected row.
        """
        ctl, calls = self._controller([])
        ctl._on_export_clicked()
        assert calls["exported"] == []
        assert calls["current"] == 0, (
            "must NOT fall back to currentItem() — that is the selection scope "
            "#14 rules out, on a permanent action"
        )
        assert calls["status"] and not calls["status"][-1][1]
        assert "ticked" in calls["status"][-1][0].lower()

    def test_confirm_precedes_the_write(self):
        ctl, calls = self._controller(["p1", "p2"])
        ctl._on_export_clicked()
        assert calls["confirmed"] == 1
        assert calls["exported"] == [["p1", "p2"]]

    def test_declining_the_confirm_writes_nothing(self):
        ctl, calls = self._controller(["p1"], confirm=False)
        ctl._on_export_clicked()
        assert calls["exported"] == []
        assert "cancel" in calls["status"][-1][0].lower()

    def test_busy_surface_blocks_export(self):
        """One Step-1 op at a time.  `extract_patches` guards; export did
        not — and its fallback branch took no lock at all."""
        ctl, calls = self._controller(["p1"])
        ctl._step1_busy = True
        ctl._on_export_clicked()
        assert calls["exported"] == [] and calls["confirmed"] == 0


# ---------------------------------------------------------------------------
# Threaded source read (background QThread + Cancel) — the load-off-main-thread
# refactor.  The disk READ (`open_lazy` + `materialize`) runs on a
# `_SourceLoadWorker` QThread so the GUI stays live; a Cancel button discards
# the result (token bump) and unlocks the surface instantly.  All viewer/layer
# work happens back on the main thread in `_on_source_loaded`.
# ---------------------------------------------------------------------------

class _FakeSignal:
    """Records connected callbacks (never fires) — stand-in for a Qt Signal."""

    def __init__(self):
        self.cbs = []

    def connect(self, cb):
        self.cbs.append(cb)


class _FakeThread:
    def __init__(self):
        self.started = _FakeSignal()
        self.finished = _FakeSignal()
        self.started_flag = False
        self.quit_called = False
        self.waited = False
        self.deleted = False

    def start(self):
        self.started_flag = True

    def quit(self):
        self.quit_called = True

    def wait(self, ms=None):
        self.waited = True
        return True

    def deleteLater(self):
        self.deleted = True


class _FakeWorker:
    """Captures constructor args; `run` is a no-op so the read never happens on
    the calling thread (that is exactly the property under test)."""

    def __init__(self, path, use_memmap):
        self.path = path
        self.use_memmap = use_memmap
        self.cancel_requested = False
        self.finished = _FakeSignal()
        self.error = _FakeSignal()
        self.cancelled = _FakeSignal()

    def moveToThread(self, thread):
        self.moved_to = thread

    def request_cancel(self):
        self.cancel_requested = True

    def run(self):
        pass


def _write_source_tiff(tmp_path, arr):
    p = tmp_path / "src.tif"
    tifffile.imwrite(str(p), arr)
    return str(p)


class TestThreadedSourceLoad:
    """The Step-1 source read runs on a background QThread with a Cancel escape
    hatch; the finished/error/cancel handlers own the main-thread continuation
    and the busy-lock lifecycle via a monotonically-bumped token."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    # -- (1) the blocking read is OFF the synchronous path ------------------

    def test_load_source_defers_the_read_to_a_worker(self, qt_app, tmp_path,
                                                      monkeypatch):
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt

        from napari_epicycle.data_preparation import controller as ctlmod

        # Swap the worker + thread for inert fakes: load_source must construct
        # and STORE them (deferring the read), not read on the calling thread.
        monkeypatch.setattr(ctlmod, "_SourceLoadWorker", _FakeWorker)
        monkeypatch.setattr(ctlmod, "QThread", _FakeThread)

        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            item = QListWidgetItem("src")
            item.setData(Qt.UserRole, path)
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)
            w.display_name_edit.setText("myname")

            ctl.load_source()

            # Worker + thread stored, launch requested; NO source layers yet
            # (add_image runs only in the main-thread finished handler); the
            # surface is locked; the Load button flips to Cancel (kept
            # clickable) until the handler clears it.
            assert isinstance(ctl._source_load_worker, _FakeWorker)
            assert ctl._source_load_worker.path == path
            assert ctl._source_load_thread.started_flag is True
            assert len(ctl.viewer.layers) == 0
            assert ctl._step1_busy is True
            assert ctl._source_loading is True
            assert w.load_source_button.text() == "Cancel"
            assert w.load_source_button.isEnabled()      # stays clickable
        finally:
            w.setParent(None)

    def test_load_button_dispatches_load_vs_cancel(self, qt_app):
        """The Load button doubles as Cancel: a click routes to `load_source`
        when idle and to `cancel_source_load` while a read is in flight."""
        w = self._widget()
        try:
            ctl = w.controller
            calls = {"load": 0, "cancel": 0}
            ctl.load_source = lambda: calls.__setitem__("load", calls["load"] + 1)
            ctl.cancel_source_load = lambda: calls.__setitem__(
                "cancel", calls["cancel"] + 1)

            ctl._source_loading = False
            ctl.on_load_or_cancel_clicked()             # idle -> load
            assert calls == {"load": 1, "cancel": 0}

            ctl._source_loading = True
            ctl.on_load_or_cancel_clicked()             # loading -> cancel
            assert calls == {"load": 1, "cancel": 1}
        finally:
            w.setParent(None)

    # -- (2) fresh-token finish adds the layers + clears busy ----------------

    def test_on_source_loaded_fresh_token_adds_layers_and_clears_busy(
            self, qt_app, tmp_path):
        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            lazy = open_lazy(path)
            data = materialize(lazy)
            ctl._source_load_token = 7
            ctl._set_step1_busy(True)
            ctl._source_loading = True
            ctl._set_load_button_cancel(True)           # button now says Cancel

            ctl._on_source_loaded(lazy, data, 7, path, "myname", False)

            assert len(ctl.viewer.layers) == 2          # one layer per channel
            assert ctl._loaded_source_file == path
            assert ctl._source_lazy is lazy
            assert ctl._step1_busy is False             # lock cleared
            assert ctl._source_loading is False
            assert w.load_source_button.text() == "Load"   # reverted
            assert "Loaded" in w.source_status_label.text()
        finally:
            w.setParent(None)

    # -- (3) stale-token finish is a strict no-op ---------------------------

    def test_on_source_loaded_stale_token_is_a_noop(self, qt_app, tmp_path):
        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            lazy = open_lazy(path)
            data = materialize(lazy)
            ctl._source_load_token = 9              # current
            ctl._source_lazy = "SENTINEL"           # must not be clobbered
            ctl._loaded_source_file = "prev"

            ctl._on_source_loaded(lazy, data, 8, path, "myname", False)  # stale

            assert len(ctl.viewer.layers) == 0      # viewer untouched
            assert ctl._source_lazy == "SENTINEL"   # state untouched
            assert ctl._loaded_source_file == "prev"
        finally:
            w.setParent(None)

    # -- (4) cancel bumps the token, unlocks, and signals the worker --------

    def test_cancel_source_load_bumps_token_unlocks_and_cancels_worker(
            self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            fake = _FakeWorker("p", False)
            ctl._source_load_worker = fake
            ctl._source_load_token = 3
            ctl._set_step1_busy(True)
            ctl._source_loading = True
            ctl._set_load_button_cancel(True)

            ctl.cancel_source_load()

            assert ctl._source_load_token == 4          # bumped -> discards finish
            assert fake.cancel_requested is True        # worker told to stop
            assert ctl._step1_busy is False             # unlocked instantly
            assert ctl._source_loading is False
            assert w.load_source_button.text() == "Load"   # reverted
        finally:
            w.setParent(None)

    def test_stale_finish_after_cancel_is_discarded(self, qt_app, tmp_path):
        """Cancel bumps the token; the worker's later finished handler sees the
        stale token and must NOT touch the viewer (Cancel already owns the UI)."""
        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            fake = _FakeWorker("p", False)
            ctl._source_load_worker = fake
            ctl._source_load_token = 5
            ctl._set_step1_busy(True)

            ctl.cancel_source_load()                    # token -> 6, unlocked
            lazy = open_lazy(path)
            data = materialize(lazy)
            ctl._on_source_loaded(lazy, data, 5, path, "n", False)  # stale (5)

            assert len(ctl.viewer.layers) == 0          # discarded, no layers
            assert ctl._step1_busy is False             # still unlocked (no re-clear)
        finally:
            w.setParent(None)

    # -- (5) error path writes the red status + clears busy -----------------

    def test_on_source_load_error_sets_status_and_clears_busy(
            self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            ctl._source_load_token = 2
            ctl._source_lazy = "SENTINEL"
            ctl._set_step1_busy(True)
            ctl._source_loading = True
            ctl._set_load_button_cancel(True)

            ctl._on_source_load_error("boom", 2)

            assert ctl._source_lazy is None
            assert "Failed to read" in w.source_status_label.text()
            assert "boom" in w.source_status_label.text()
            assert ctl._step1_busy is False
            assert ctl._source_loading is False
            assert w.load_source_button.text() == "Load"
        finally:
            w.setParent(None)

    def test_on_source_load_error_stale_token_is_a_noop(self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            ctl._source_load_token = 4
            ctl._source_lazy = "SENTINEL"

            ctl._on_source_load_error("boom", 3)        # stale

            assert ctl._source_lazy == "SENTINEL"       # untouched
        finally:
            w.setParent(None)

    # -- worker.run() emits the right signals -------------------------------

    def test_worker_run_emits_finished_with_byte_identical_data(
            self, qt_app, tmp_path):
        from napari_epicycle.data_preparation.controller import _SourceLoadWorker

        arr = np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8)
        path = _write_source_tiff(tmp_path, arr)
        worker = _SourceLoadWorker(path, False)
        got = []
        errs = []
        worker.finished.connect(lambda lazy, data: got.append((lazy, data)))
        worker.error.connect(lambda m: errs.append(m))

        worker.run()

        assert errs == []
        assert len(got) == 1
        _lazy, data = got[0]
        assert np.array_equal(np.asarray(data), arr)    # byte-identical read

    def test_worker_run_cancelled_before_read_emits_cancelled(
            self, qt_app, tmp_path):
        from napari_epicycle.data_preparation.controller import _SourceLoadWorker

        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        worker = _SourceLoadWorker(path, False)
        events = []
        worker.finished.connect(lambda *a: events.append("finished"))
        worker.cancelled.connect(lambda: events.append("cancelled"))

        worker.request_cancel()
        worker.run()

        assert events == ["cancelled"]                  # never finished

    def test_worker_run_error_on_unreadable_path(self, qt_app):
        from napari_epicycle.data_preparation.controller import _SourceLoadWorker

        worker = _SourceLoadWorker(r"C:/does/not/exist_xyz.tif", False)
        got = []
        errs = []
        worker.finished.connect(lambda *a: got.append(a))
        worker.error.connect(lambda m: errs.append(m))

        worker.run()

        assert got == []
        assert len(errs) == 1 and errs[0]               # non-empty message

    # -- audit HIGH #1: shutdown joins the in-flight/draining threads --------

    def test_shutdown_cancels_and_joins_the_inflight_read(self, qt_app, tmp_path):
        """On close/destroy, `shutdown()` must bump the token, cancel the
        worker, and quit+wait the QThread — so it is never destroyed while
        running (native abort) and a finish arriving afterward is discarded."""
        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            thread, worker = _FakeThread(), _FakeWorker(path, False)
            ctl._source_load_thread = thread
            ctl._source_load_worker = worker
            ctl._source_load_token = 3
            ctl._set_step1_busy(True)

            ctl.shutdown()

            assert worker.cancel_requested is True       # worker told to stop
            assert thread.quit_called and thread.waited   # quit + joined
            assert ctl._source_load_token == 4            # bumped -> finish stale
            assert ctl._source_load_thread is None
            assert ctl._source_load_worker is None

            # A finish arriving AFTER shutdown (old token) is discarded — it must
            # NOT add layers onto a torn-down / foreign canvas.
            lazy = open_lazy(path)
            data = materialize(lazy)
            ctl._on_source_loaded(lazy, data, 3, path, "myname", False)
            assert len(ctl.viewer.layers) == 0
        finally:
            w.setParent(None)

    def test_shutdown_also_joins_draining_threads(self, qt_app, tmp_path):
        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            drain_t, drain_w = _FakeThread(), _FakeWorker(path, False)
            ctl._source_load_thread = None
            ctl._draining_source_threads = [(drain_t, drain_w)]

            ctl.shutdown()

            assert drain_w.cancel_requested is True
            assert drain_t.quit_called and drain_t.waited
            assert ctl._draining_source_threads == []
        finally:
            w.setParent(None)

    def test_shutdown_is_idempotent(self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            ctl.shutdown()
            tok = ctl._source_load_token
            ctl.shutdown()                                # second call no-ops
            assert ctl._source_load_token == tok
        finally:
            w.setParent(None)

    # -- audit HIGH #2: step-exit cancels an in-flight load -----------------

    def test_step_exit_cancels_an_inflight_load(self, qt_app, tmp_path):
        """A read finishing AFTER the user leaves Step 1 must not re-add Step-1
        Image layers onto another step's shared canvas (the pinned Step-3
        all-Image-layers abort).  `on_step_exit` cancels it so the finish goes
        stale + the surface unlocks."""
        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            thread, worker = _FakeThread(), _FakeWorker(path, False)
            ctl._source_load_thread = thread
            ctl._source_load_worker = worker
            ctl._source_load_token = 1
            ctl._set_step1_busy(True)

            ctl.on_step_exit()

            assert ctl._source_load_token == 2            # bumped -> finish stale
            assert worker.cancel_requested is True
            assert ctl._step1_busy is False               # surface unlocked

            # The now-stale finish adds NOTHING to the (foreign) canvas.
            lazy = open_lazy(path)
            data = materialize(lazy)
            ctl._on_source_loaded(lazy, data, 1, path, "myname", False)
            assert len(ctl.viewer.layers) == 0
        finally:
            w.setParent(None)


# ---------------------------------------------------------------------------
# Large-data viewing — the "Prepare for large-data viewing" button converts the
# selected source to a multiscale OME-Zarr store on a background QThread, then
# loads it as a MULTISCALE image.  Mirrors the threaded source-read machinery:
# a `_LargeDataPrepWorker` runs `convert_to_ome_zarr` off the main thread; the
# main-thread `_on_large_data_ready` opens the store and adds per-channel
# multiscale layers; Cancel / step-exit / shutdown discard + join it via a
# monotonically-bumped token.
# ---------------------------------------------------------------------------

from packages.epicycle_largedata.multiscale import (  # noqa: E402
    convert_to_ome_zarr as _convert_to_ome_zarr,
    open_multiscale as _open_multiscale,
)


class _FakePrepWorker:
    """Captures constructor args; `run` is a no-op so the convert never happens
    on the calling thread — the property under test (deferral)."""

    def __init__(self, src, out_store, voxel_size, image_axes,
                 num_workers=None, chunk=None, compressor=None,
                 downsample=None, min_size=None):
        # NOTE: this signature MIRRORS _LargeDataPrepWorker's.  If the real
        # worker gains a kwarg and this does not, construction raises inside the
        # controller's launch guard, which swallows it and leaves
        # `_largedata_worker` as None -- the failure then reads as "the worker
        # was never created", not as a signature mismatch.  Keep them in step.
        self.src = src
        self.out_store = out_store
        self.voxel_size = voxel_size
        self.image_axes = image_axes
        self.num_workers = num_workers
        self.chunk = chunk
        self.compressor = compressor
        self.downsample = downsample
        self.min_size = min_size
        self.cancel_requested = False
        self.finished = _FakeSignal()
        self.error = _FakeSignal()
        self.cancelled = _FakeSignal()
        # Brick 5 fix 1: the real worker reports pyramid levels so the bar is a
        # real percentage instead of a marquee.  The launch connects it, so the
        # double must carry it -- same failure mode as the signature note above:
        # an AttributeError swallowed by the launch guard, reading as "no worker".
        self.progress = _FakeSignal()

    def moveToThread(self, thread):
        self.moved_to = thread

    def request_cancel(self):
        self.cancel_requested = True

    def run(self):
        pass


class _RecordedLayer:
    """A stand-in image layer that PRESERVES whatever `add_image` was handed —
    so a multiscale add records the exact list of pyramid levels as `.data`
    (napari would wrap it in a `MultiScaleData`; the recording keeps the raw
    list, which is what the controller passed and what these tests assert on)."""

    def __init__(self, data, name, colormap, blending, multiscale):
        self.data = data
        self.name = name
        self.colormap = colormap
        self.blending = blending
        self.multiscale = multiscale
        self.visible = True


class _MultiscaleRecordingViewer:
    """Viewer whose `add_image` accepts `multiscale=` and records the passed
    data verbatim (see `_RecordedLayer`).  `_RealImageViewer` cannot stand in
    here — its `np.asarray(data)` would collapse a pyramid list."""

    def __init__(self):
        self.layers = _EventFiringLayerList()
        self.mouse_drag_callbacks = []

    def add_image(self, data, name=None, colormap=None, blending=None,
                  multiscale=False):
        layer = _RecordedLayer(data, name, colormap, blending, multiscale)
        self.layers.append(layer)
        return layer


def _make_store(tmp_path, arr, name="src.zarr", image_axes=None, **kw):
    """Convert `arr` to a small multiscale store (explicit tiny chunk/min_size
    so a small fixture still yields multiple pyramid levels)."""
    src = _write_source_tiff(tmp_path, arr)
    store = str(tmp_path / name)
    kw.setdefault("chunk", 16)
    kw.setdefault("min_size", 8)
    _convert_to_ome_zarr(src, store, image_axes=image_axes, overwrite=True, **kw)
    return store


class TestLargeDataPrepare:
    """The "Prepare for large-data viewing" flow: threaded convert + Cancel +
    multiscale display, wired with the same token / lock / lifecycle guards as
    the source read."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    def _ms_widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_MultiscaleRecordingViewer())

    # -- (0) the button exists, is themed, disabled until a file is selected --

    def test_button_exists_themed_and_selection_gated(self, qt_app):
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt

        w = self._widget()
        try:
            btn = w.prepare_largedata_button
            assert btn is not None
            assert btn.text() == "Prepare for large-data viewing"
            assert btn.styleSheet()                       # themed (secondary)
            assert not btn.isEnabled()                    # nothing selected yet

            item = QListWidgetItem("s")
            item.setData(Qt.UserRole, "x.tif")
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)
            assert btn.isEnabled()                        # selection enables it
        finally:
            w.setParent(None)

    def test_button_click_is_wired_to_prepare(self, qt_app):
        """A click routes through the dispatcher to `prepare_large_data` (the
        dispatcher calls it dynamically, so a monkeypatch is observed)."""
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt

        w = self._widget()
        try:
            called = []
            w.controller.prepare_large_data = lambda: called.append(1)
            item = QListWidgetItem("s")
            item.setData(Qt.UserRole, "x.tif")
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)
            w.prepare_largedata_button.click()
            assert called == [1]
        finally:
            w.setParent(None)

    def test_prepare_button_dispatches_prepare_vs_cancel(self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            calls = {"prep": 0, "cancel": 0}
            ctl.prepare_large_data = lambda: calls.__setitem__(
                "prep", calls["prep"] + 1)
            ctl.cancel_large_data_prep = lambda: calls.__setitem__(
                "cancel", calls["cancel"] + 1)

            ctl._largedata_preparing = False
            ctl.on_prepare_or_cancel_clicked()            # idle -> prepare
            assert calls == {"prep": 1, "cancel": 0}

            ctl._largedata_preparing = True
            ctl.on_prepare_or_cancel_clicked()            # in flight -> cancel
            assert calls == {"prep": 1, "cancel": 1}
        finally:
            w.setParent(None)

    # -- (1) launch defers the convert to a worker; no layers yet -----------

    def test_prepare_defers_convert_to_a_worker(self, qt_app, tmp_path,
                                                 monkeypatch):
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt

        from napari_epicycle.data_preparation import controller as ctlmod

        monkeypatch.setattr(ctlmod, "_LargeDataPrepWorker", _FakePrepWorker)
        monkeypatch.setattr(ctlmod, "QThread", _FakeThread)

        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            item = QListWidgetItem("src")
            item.setData(Qt.UserRole, path)
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)
            w.display_name_edit.setText("myname")
            w.output_file_path.setText(str(tmp_path))
            # The scanned header carries a CLEAN CZYX axes string; the source is
            # NOT loaded (`_loaded_source_file` is None) and the Order combo is
            # left showing the WRONG order (ZCYX).  A stale-combo regression
            # (echoing the combo for an un-loaded file) would hand the converter
            # "ZCYX" and mis-orient the store; the resolver must instead detect
            # "CZYX" from the metadata.
            ctl._scanned_files = [
                {"path": path, "stem": "src", "shape": (2, 4, 8, 8),
                 "axes": "CZYX"}]
            w.axis_order_combo.setCurrentText("ZCYX")

            ctl.prepare_large_data()

            # Worker + thread stored, launch requested; NO layers yet (add_image
            # runs only in the main-thread finished handler); surface locked; the
            # Prepare button flips to Cancel (kept clickable).
            assert isinstance(ctl._largedata_worker, _FakePrepWorker)
            assert ctl._largedata_worker.src == path
            expected = os.path.join(
                str(tmp_path), "artifacts", "step1", "largedata", "myname.zarr")
            assert os.path.normpath(ctl._largedata_worker.out_store) == \
                os.path.normpath(expected)
            assert os.path.isdir(os.path.dirname(expected))   # mkdir parents
            # DETECTED from clean metadata, NOT echoed from the (ZCYX) combo —
            # the un-loaded 4D source is oriented from its header, not the combo.
            assert ctl._largedata_worker.image_axes == "CZYX"
            assert ctl._largedata_thread.started_flag is True
            assert len(ctl.viewer.layers) == 0
            assert ctl._step1_busy is True
            assert ctl._largedata_preparing is True
            assert w.prepare_largedata_button.text() == "Cancel"
            assert w.prepare_largedata_button.isEnabled()
        finally:
            w.setParent(None)

    def test_prepare_forwards_saved_workspace_config_to_worker(
            self, qt_app, tmp_path, monkeypatch):
        """A per-workspace saved config feeds num_workers/chunk/downsample/
        min_size to the convert worker (replaces the old workers-only hardcode)."""
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt
        from napari_epicycle.data_preparation import controller as ctlmod
        from napari_epicycle.data_preparation._state import save_largedata_config

        monkeypatch.setattr(ctlmod, "_LargeDataPrepWorker", _FakePrepWorker)
        monkeypatch.setattr(ctlmod, "QThread", _FakeThread)

        cfg = {"preset": "custom", "num_workers": 3, "chunk": 128,
               "downsample": 3, "min_size": 128}
        save_largedata_config(str(tmp_path), cfg)

        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            item = QListWidgetItem("src")
            item.setData(Qt.UserRole, path)
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)
            w.display_name_edit.setText("myname")
            w.output_file_path.setText(str(tmp_path))
            ctl._scanned_files = [
                {"path": path, "stem": "src", "shape": (2, 4, 8, 8),
                 "axes": "CZYX"}]

            ctl.prepare_large_data()

            worker = ctl._largedata_worker
            assert worker is not None
            assert worker.num_workers == 3
            # Brick 4b: the controller forwards the store's chunk SHAPE and
            # the chosen codec.  This config is `custom`, so its saved knobs are
            # honoured verbatim (a NAMED preset is re-derived instead).
            assert tuple(worker.chunk) == (128, 128, 128)
            assert worker.compressor == "zstd"
            assert worker.downsample == 3
            assert worker.min_size == 128
        finally:
            w.setParent(None)

    def test_prepare_uses_adaptive_default_when_no_config_saved(
            self, qt_app, tmp_path, monkeypatch):
        """With nothing saved, the worker still gets full pyramid knobs from the
        machine-adaptive 'balanced' default (chunk/downsample/min_size non-None)."""
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt
        from napari_epicycle.data_preparation import controller as ctlmod

        monkeypatch.setattr(ctlmod, "_LargeDataPrepWorker", _FakePrepWorker)
        monkeypatch.setattr(ctlmod, "QThread", _FakeThread)

        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            item = QListWidgetItem("src")
            item.setData(Qt.UserRole, path)
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)
            w.display_name_edit.setText("myname")
            w.output_file_path.setText(str(tmp_path))
            ctl._scanned_files = [
                {"path": path, "stem": "src", "shape": (2, 4, 8, 8),
                 "axes": "CZYX"}]

            ctl.prepare_large_data()

            worker = ctl._largedata_worker
            assert worker is not None
            # balanced default: chunk 64, downsample 2, min_size 256, workers>=1
            # Brick 4b: the adaptive default is the anisotropic shape
            assert tuple(worker.chunk) == (32, 128, 128)
            assert worker.downsample == 2
            assert worker.min_size == 256
            assert isinstance(worker.num_workers, int) and worker.num_workers >= 1
        finally:
            w.setParent(None)

    def test_prepare_without_workspace_does_not_launch(self, qt_app, tmp_path,
                                                       monkeypatch):
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt
        from napari_epicycle.data_preparation import controller as ctlmod

        monkeypatch.setattr(ctlmod, "_LargeDataPrepWorker", _FakePrepWorker)
        monkeypatch.setattr(ctlmod, "QThread", _FakeThread)
        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            item = QListWidgetItem("src")
            item.setData(Qt.UserRole, path)
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)
            w.output_file_path.setText("")                 # no workspace

            ctl.prepare_large_data()

            assert ctl._largedata_worker is None           # never launched
            assert ctl._step1_busy is False                # surface not locked
            assert "workspace" in w.source_status_label.text().lower()
        finally:
            w.setParent(None)

    def test_prepare_proceeds_on_ambiguous_4d_with_heuristic(self, qt_app,
                                                             tmp_path, monkeypatch):
        """Full fix: a 4D source with NO clean metadata is NO LONGER blocked
        behind a mandatory Load.  Prepare proceeds using the header heuristic
        (2 ch ≤ 8 → CZYX) — the convert LAUNCHES with `image_axes='CZYX'`, and
        there is no 'confirm the order first' bail."""
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt
        from napari_epicycle.data_preparation import controller as ctlmod

        monkeypatch.setattr(ctlmod, "_LargeDataPrepWorker", _FakePrepWorker)
        monkeypatch.setattr(ctlmod, "QThread", _FakeThread)
        path = _write_source_tiff(
            tmp_path, np.random.randint(0, 255, (2, 4, 8, 8), dtype=np.uint8))
        w = self._widget()
        try:
            ctl = w.controller
            # Scanned BEFORE selection so detect-on-selection can read the shape.
            ctl._scanned_files = [
                {"path": path, "stem": "src", "shape": (2, 4, 8, 8),
                 "axes": None}]                            # ambiguous metadata
            ctl._loaded_source_file = None
            item = QListWidgetItem("src")
            item.setData(Qt.UserRole, path)
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)        # -> detect-on-selection
            w.display_name_edit.setText("amb")
            w.output_file_path.setText(str(tmp_path))

            # Resolves to the heuristic order (no None, no Load required).
            assert ctl._resolve_prepare_image_axes(path) == "CZYX"

            ctl.prepare_large_data()

            assert ctl._largedata_worker is not None        # convert LAUNCHED
            assert ctl._largedata_worker.image_axes == "CZYX"
            assert ctl._largedata_preparing is True
            msg = w.source_status_label.text().lower()
            assert "confirm the order" not in msg           # no bail
        finally:
            w.setParent(None)

    def test_get_combined_image_stack_skips_multiscale(self, qt_app):
        """FIX 2: a ticked MultiScaleData layer (a Sequence, NOT a list/ndarray)
        must be SKIPPED by the eager stack — the old `isinstance(.., (list,
        tuple))` guard missed it and `np.stack` would raise.  A real ndarray
        channel ticked alongside still stacks; no raise."""
        from types import SimpleNamespace
        from qtpy.QtWidgets import QListWidgetItem
        from napari.layers._multiscale_data import MultiScaleData

        w = self._widget()
        try:
            ctl = w.controller
            real = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
            ms = MultiScaleData([
                np.zeros((8, 8, 8), np.uint16),
                np.zeros((4, 4, 4), np.uint16)])
            assert not isinstance(ms, (list, tuple))        # the trap FIX 2 closes
            layers = _FakeLayerList()
            layers.append(SimpleNamespace(name="real_ch", data=real))
            layers.append(SimpleNamespace(name="ms_ch", data=ms))
            ctl.viewer = SimpleNamespace(layers=layers)
            for name in ["real_ch", "ms_ch"]:
                it = QListWidgetItem(name)
                w.image_layer_combo.addItem(it)
                it.setSelected(True)

            combined, names = ctl.get_combined_image_stack()   # must not raise
            assert combined.shape == (1, 8, 8, 8)              # only the ndarray
            assert np.array_equal(combined[0], real)
            assert set(names) == {"real_ch", "ms_ch"}          # selection verbatim
        finally:
            w.setParent(None)

    # -- (2) fresh-token finish adds multiscale layers + wires L0 state ------

    def test_on_large_data_ready_adds_multiscale_layers(self, qt_app, tmp_path):
        arr = np.random.randint(0, 255, (2, 8, 40, 40), dtype=np.uint8)  # CZYX
        store = _make_store(tmp_path, arr, image_axes="CZYX")
        _levels, meta = _open_multiscale(store)
        n_levels = meta["n_levels"]
        assert n_levels >= 2                               # tiny store, real pyramid

        w = self._ms_widget()
        try:
            ctl = w.controller
            ctl._largedata_token = 5
            ctl._set_step1_busy(True)
            ctl._largedata_preparing = True
            ctl._set_prepare_button_cancel(True)           # button now says Cancel

            ctl._on_large_data_ready(store, "myname", 5)

            assert len(ctl.viewer.layers) == 2             # one layer per channel
            for lyr in ctl.viewer.layers:
                assert isinstance(lyr.data, list)          # multiscale = a list
                assert len(lyr.data) == n_levels
                assert lyr.data[0].ndim == 3               # each level is ZYX
                assert lyr.multiscale is True
            assert tuple(ctl._source_lazy.shape) == arr.shape   # level 0, full-res
            assert ctl._source_axes_resolved == "CZYX"
            assert ctl._channel_index_by_name == {
                "myname_ch0": 0, "myname_ch1": 1}
            assert ctl.box_tool._full_shape == (8, 40, 40)  # (Z, Y, X) from level 0
            assert ctl._step1_busy is False                 # lock cleared
            assert ctl._largedata_preparing is False
            assert w.prepare_largedata_button.text() == \
                "Prepare for large-data viewing"            # reverted
            assert "large-data viewing" in w.source_status_label.text()
        finally:
            w.setParent(None)

    def test_on_large_data_ready_zyx_single_channel(self, qt_app, tmp_path):
        arr = np.random.randint(0, 255, (8, 40, 40), dtype=np.uint8)     # ZYX
        store = _make_store(tmp_path, arr, name="mono.zarr")
        w = self._ms_widget()
        try:
            ctl = w.controller
            ctl._largedata_token = 1
            ctl._on_large_data_ready(store, "mono", 1)

            assert len(ctl.viewer.layers) == 1             # single channel
            assert ctl.viewer.layers[0].multiscale is True
            assert ctl._source_axes_resolved == "ZYX"
            assert ctl._channel_index_by_name == {"mono": 0}
            assert tuple(ctl._source_lazy.shape) == arr.shape
            assert ctl.box_tool._full_shape == (8, 40, 40)
        finally:
            w.setParent(None)

    # -- (3) stale-token finish is a strict no-op ---------------------------

    def test_on_large_data_ready_stale_token_is_a_noop(self, qt_app, tmp_path):
        arr = np.random.randint(0, 255, (2, 8, 40, 40), dtype=np.uint8)
        store = _make_store(tmp_path, arr, image_axes="CZYX")
        w = self._ms_widget()
        try:
            ctl = w.controller
            ctl._largedata_token = 9                       # current
            ctl._source_lazy = "SENTINEL"
            ctl._loaded_source_file = "prev"

            ctl._on_large_data_ready(store, "myname", 8)   # stale

            assert len(ctl.viewer.layers) == 0
            assert ctl._source_lazy == "SENTINEL"
            assert ctl._loaded_source_file == "prev"
        finally:
            w.setParent(None)

    # -- (4) cancel bumps the token, unlocks, signals the worker ------------

    def test_cancel_prepare_bumps_token_unlocks_and_cancels_worker(
            self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            fake = _FakePrepWorker("s", "o", None, "CZYX")
            ctl._largedata_worker = fake
            ctl._largedata_token = 3
            ctl._set_step1_busy(True)
            ctl._largedata_preparing = True
            ctl._set_prepare_button_cancel(True)

            ctl.cancel_large_data_prep()

            assert ctl._largedata_token == 4               # bumped -> finish stale
            assert fake.cancel_requested is True
            assert ctl._step1_busy is False
            assert ctl._largedata_preparing is False
            assert w.prepare_largedata_button.text() == \
                "Prepare for large-data viewing"
        finally:
            w.setParent(None)

    # -- (4b) cancel keeps the surface LOCKED until the worker drains --------

    def test_cancel_prepare_stays_locked_until_the_thread_finishes(self, qt_app):
        """The convert is non-interruptible: it keeps WRITING the store after
        Cancel.  Unlocking immediately let the user launch a second Prepare onto
        the same `out_store` -> two writers on one path."""
        w = self._widget()
        try:
            ctl = w.controller
            fake = _FakePrepWorker("s", "o", None, "CZYX")
            thread = _FakeThread()
            ctl._largedata_worker = fake
            ctl._largedata_thread = thread
            ctl._largedata_token = 3
            ctl._set_step1_busy(True)
            ctl._largedata_preparing = True
            ctl._set_prepare_button_cancel(True)

            ctl.cancel_large_data_prep()

            assert ctl._largedata_token == 4          # the finish goes stale
            assert fake.cancel_requested is True
            # still locked -- the worker is draining
            assert ctl._step1_busy is True
            assert ctl._largedata_preparing is True
            assert w.prepare_largedata_button.isEnabled() is False
            # retained against GC so the QThread is never destroyed mid-write
            assert (thread, fake) in ctl._draining_largedata_threads

            # ... the worker finally stops and the thread signals finished
            ctl._cleanup_largedata_thread(thread)

            assert ctl._step1_busy is False
            assert ctl._largedata_preparing is False
            assert ctl._largedata_cancel_draining_thread is None
            assert w.prepare_largedata_button.text() ==                 "Prepare for large-data viewing"
            # and a fresh Prepare dispatches again
            launched = []
            ctl.prepare_large_data = lambda: launched.append(1)
            ctl.on_prepare_or_cancel_clicked()
            assert launched == [1]
        finally:
            w.setParent(None)

    def test_cancel_prepare_with_no_thread_unlocks_immediately(self, qt_app):
        """Nothing to drain (launch failed before the thread started) -> the
        surface must not be left locked forever."""
        w = self._widget()
        try:
            ctl = w.controller
            ctl._largedata_worker = None
            ctl._largedata_thread = None
            ctl._set_step1_busy(True)
            ctl._largedata_preparing = True

            ctl.cancel_large_data_prep()

            assert ctl._step1_busy is False
            assert ctl._largedata_preparing is False
        finally:
            w.setParent(None)

    def test_a_second_cancel_click_cannot_launch_a_prepare_while_draining(
            self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            ctl._largedata_worker = _FakePrepWorker("s", "o", None, "CZYX")
            ctl._largedata_thread = _FakeThread()
            ctl._largedata_preparing = True
            ctl.cancel_large_data_prep()

            launched = []
            ctl.prepare_large_data = lambda: launched.append(1)
            ctl.on_prepare_or_cancel_clicked()

            assert launched == [], "a Prepare launched while a worker was draining"
        finally:
            w.setParent(None)

    # -- (4c) the convert itself stops: real in-loop cancel (Brick 1b #2) ----

    def test_prep_worker_threads_should_cancel_into_the_converter(
            self, qt_app, monkeypatch):
        """`_LargeDataPrepWorker` must hand the converter a live cancel probe,
        not just discard the result afterwards."""
        from napari_epicycle.data_preparation import controller as ctlmod

        seen = {}

        def fake_convert(src, out, **kw):
            seen.update(kw)
            assert callable(kw.get("should_cancel"))
            assert kw["should_cancel"]() is False
            return {"path": str(out)}

        monkeypatch.setattr(ctlmod, "convert_to_ome_zarr", fake_convert)
        worker = ctlmod._LargeDataPrepWorker("s.tif", "o.zarr", None, "ZYX")
        events = []
        worker.finished.connect(lambda p: events.append(("finished", p)))
        worker.cancelled.connect(lambda: events.append(("cancelled",)))
        worker.error.connect(lambda m: events.append(("error", m)))
        worker.run()

        assert events == [("finished", "o.zarr")]
        # the probe follows the flag
        worker.request_cancel()
        assert seen["should_cancel"]() is True

    def test_prep_worker_emits_cancelled_not_error_when_convert_cancels(
            self, qt_app, monkeypatch):
        """A cancelled convert raises `ConvertCancelled`; the worker must route
        it to the `cancelled` signal, NOT `error` (which paints a red
        'Failed to prepare' the user never caused)."""
        from napari_epicycle.data_preparation import controller as ctlmod
        from packages.epicycle_largedata.exceptions import ConvertCancelled

        def cancelling_convert(src, out, **kw):
            raise ConvertCancelled("cancelled at level 1")

        monkeypatch.setattr(ctlmod, "convert_to_ome_zarr", cancelling_convert)
        worker = ctlmod._LargeDataPrepWorker("s.tif", "o.zarr", None, "ZYX")
        events = []
        worker.finished.connect(lambda p: events.append("finished"))
        worker.cancelled.connect(lambda: events.append("cancelled"))
        worker.error.connect(lambda m: events.append(f"error:{m}"))
        worker.request_cancel()
        worker.run()

        assert events == ["cancelled"]

    def test_cancel_on_step_exit_shows_no_busy_bar(self, qt_app):
        """`on_step_exit` sets `_left_step1` and THEN cancels the convert; the
        drain notice must not paint the Step-1 busy bar over another step."""
        w = self._widget()
        try:
            ctl = w.controller
            ctl._largedata_worker = _FakePrepWorker("s", "o", None, "CZYX")
            ctl._largedata_thread = _FakeThread()
            ctl._largedata_preparing = True
            ctl._left_step1 = True
            shown = []
            w._busy_show = lambda *a, **k: shown.append(a)

            ctl.cancel_large_data_prep()

            assert shown == []
            assert ctl._largedata_worker.cancel_requested is True
        finally:
            w.setParent(None)

    def test_cancel_drain_finishing_after_step_exit_is_silent(self, qt_app):
        """A cancelled convert can drain AFTER the user has left Step 1.  The
        unlock/cleanup must still happen, but nothing may re-show the Step-1
        busy bar or otherwise paint Step-1 UI on another step's screen."""
        w = self._widget()
        try:
            ctl = w.controller
            fake = _FakePrepWorker("s", "o", None, "CZYX")
            thread = _FakeThread()
            ctl._largedata_worker = fake
            ctl._largedata_thread = thread
            ctl._set_step1_busy(True)
            ctl._largedata_preparing = True
            ctl.cancel_large_data_prep()

            ctl._left_step1 = True                    # user navigated away
            shown = []
            w._busy_show = lambda *a, **k: shown.append(a)

            ctl._cleanup_largedata_thread(thread)

            assert shown == []                        # no Step-1 UI resurrected
            assert ctl._largedata_preparing is False  # ... but still unlocked
            assert ctl._largedata_cancel_draining_thread is None
            assert ctl._step1_busy is False
        finally:
            w.setParent(None)

    # -- (5) error path writes red status + clears busy ---------------------

    def test_on_prep_error_sets_status_and_clears_busy(self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            ctl._largedata_token = 2
            ctl._set_step1_busy(True)
            ctl._largedata_preparing = True
            ctl._set_prepare_button_cancel(True)

            ctl._on_large_data_prep_error("boom", 2)

            assert "Failed to prepare" in w.source_status_label.text()
            assert "boom" in w.source_status_label.text()
            assert ctl._step1_busy is False
            assert ctl._largedata_preparing is False
            assert w.prepare_largedata_button.text() == \
                "Prepare for large-data viewing"
        finally:
            w.setParent(None)

    def test_on_prep_error_stale_token_is_a_noop(self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            ctl._largedata_token = 4
            ctl.source_status_dummy = None
            ctl._on_large_data_prep_error("boom", 3)       # stale
            assert "boom" not in w.source_status_label.text()
        finally:
            w.setParent(None)

    # -- worker.run() emits the right signals + writes the store ------------

    def test_prep_worker_run_emits_finished_and_writes_store(
            self, qt_app, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            _LargeDataPrepWorker)

        arr = np.random.randint(0, 255, (2, 4, 16, 16), dtype=np.uint8)
        src = _write_source_tiff(tmp_path, arr)
        store = str(tmp_path / "out.zarr")
        worker = _LargeDataPrepWorker(src, store, None, "CZYX")
        got, errs = [], []
        worker.finished.connect(lambda s: got.append(s))
        worker.error.connect(lambda m: errs.append(m))

        worker.run()

        assert errs == []
        assert got == [store]
        assert os.path.isdir(store)                        # DirectoryStore written

    def test_prep_worker_run_cancelled_before_convert(self, qt_app):
        from napari_epicycle.data_preparation.controller import (
            _LargeDataPrepWorker)

        worker = _LargeDataPrepWorker("s", "o", None, "CZYX")
        events = []
        worker.finished.connect(lambda *a: events.append("finished"))
        worker.cancelled.connect(lambda: events.append("cancelled"))

        worker.request_cancel()
        worker.run()

        assert events == ["cancelled"]                     # never converted

    def test_prep_worker_run_error_on_bad_source(self, qt_app):
        from napari_epicycle.data_preparation.controller import (
            _LargeDataPrepWorker)

        worker = _LargeDataPrepWorker(
            r"C:/does/not/exist_xyz.tif", "o", None, "CZYX")
        got, errs = [], []
        worker.finished.connect(lambda *a: got.append(a))
        worker.error.connect(lambda m: errs.append(m))

        worker.run()

        assert got == []
        assert len(errs) == 1 and errs[0]

    # -- lifecycle: shutdown + step-exit cancel/join the prepare thread -----

    def test_shutdown_cancels_and_joins_the_inflight_prepare(
            self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            thread, worker = _FakeThread(), _FakePrepWorker("s", "o", None, "C")
            ctl._largedata_thread = thread
            ctl._largedata_worker = worker
            ctl._largedata_token = 3
            ctl._set_step1_busy(True)

            ctl.shutdown()

            assert worker.cancel_requested is True
            assert thread.quit_called and thread.waited
            assert ctl._largedata_token == 4               # bumped -> finish stale
            assert ctl._largedata_thread is None
            assert ctl._largedata_worker is None
        finally:
            w.setParent(None)

    def test_shutdown_also_joins_draining_prepare_threads(self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            dt, dw = _FakeThread(), _FakePrepWorker("s", "o", None, "C")
            ctl._largedata_thread = None
            ctl._draining_largedata_threads = [(dt, dw)]

            ctl.shutdown()

            assert dw.cancel_requested is True
            assert dt.quit_called and dt.waited
            assert ctl._draining_largedata_threads == []
        finally:
            w.setParent(None)

    def test_step_exit_cancels_an_inflight_prepare(self, qt_app):
        w = self._widget()
        try:
            ctl = w.controller
            thread, worker = _FakeThread(), _FakePrepWorker("s", "o", None, "C")
            ctl._largedata_thread = thread
            ctl._largedata_worker = worker
            ctl._largedata_token = 1
            ctl._set_step1_busy(True)

            ctl.on_step_exit()

            assert ctl._largedata_token == 2               # bumped -> finish stale
            assert worker.cancel_requested is True
            assert ctl._largedata_preparing is False
        finally:
            w.setParent(None)


class TestResolvePrepareImageAxes:
    """`_resolve_prepare_image_axes` orients a "Prepare" WITHOUT forcing a full
    Load.  The Order combo is pointed at the SELECTED source (detect-on-selection,
    tracked in `_axis_combo_source`), so it can orient the file being prepared;
    for an un-tracked path the header heuristic is the fallback (never None for
    4D).  The data-integrity invariant survives: the combo is trusted ONLY for
    the file it describes, so an order chosen for file A can never orient file B."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    def test_3d_shape_returns_none(self, qt_app, tmp_path):
        """A 3D source is ZYX (unambiguous) -> None; the converter resolves it."""
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "mono.tif")
            ctl._scanned_files = [
                {"path": path, "stem": "mono", "shape": (8, 40, 40),
                 "axes": "ZYX"}]
            ctl._loaded_source_file = None
            ctl._axis_combo_source = None
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("ZCYX")      # irrelevant for 3D
            assert ctl._resolve_prepare_image_axes(path) is None
        finally:
            w.setParent(None)

    def test_untracked_ambiguous_4d_uses_heuristic_not_combo(self, qt_app, tmp_path):
        """A 4D file the combo does NOT describe -> the header heuristic (2 ch ≤ 8
        → CZYX), NEVER the (stale) combo.  Proves an order left in the combo for
        a different file cannot orient this one, AND that Prepare is no longer
        blocked behind a Load for a metadata-less 4D source."""
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "amb.tif")
            ctl._scanned_files = [
                {"path": path, "stem": "amb", "shape": (2, 4, 8, 8),
                 "axes": None}]
            ctl._loaded_source_file = None
            ctl._axis_combo_source = None                  # combo not for this file
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("ZCYX")      # must NOT be echoed
            assert ctl._resolve_prepare_image_axes(path) == "CZYX"  # heuristic
        finally:
            w.setParent(None)

    def test_confident_metadata_wins_over_stale_combo(self, qt_app, tmp_path):
        """Clean CZYX metadata on a file the combo does NOT describe -> detected
        'CZYX' even when the combo shows the wrong 'ZCYX' — proves an untracked
        combo is ignored and the header decides."""
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "clean.tif")
            ctl._scanned_files = [
                {"path": path, "stem": "clean", "shape": (2, 4, 8, 8),
                 "axes": "CZYX"}]
            ctl._loaded_source_file = None
            ctl._axis_combo_source = None
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("ZCYX")      # stale / wrong
            assert ctl._resolve_prepare_image_axes(path) == "CZYX"
        finally:
            w.setParent(None)

    def test_confident_metadata_detects_zcyx(self, qt_app, tmp_path):
        """A genuine ZCYX metadata order is detected as 'ZCYX' (channel axis is
        axis 1) — the header heuristic path for an untracked file."""
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "zc.tif")
            # shape chosen so the channel axis (size 2) is UNambiguously axis 1;
            # metadata 'ZCYX' makes detection confident regardless.
            ctl._scanned_files = [
                {"path": path, "stem": "zc", "shape": (40, 2, 40, 40),
                 "axes": "ZCYX"}]
            ctl._loaded_source_file = None
            ctl._axis_combo_source = None
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("CZYX")      # stale / wrong
            assert ctl._resolve_prepare_image_axes(path) == "ZCYX"
        finally:
            w.setParent(None)

    def test_combo_used_when_it_describes_the_file(self, qt_app, tmp_path):
        """When the combo DESCRIBES the file (`_axis_combo_source == path`, as the
        load path + detect-on-selection set it), the user-confirmed combo wins
        even over metadata detection."""
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "loaded.tif")
            ctl._scanned_files = [
                {"path": path, "stem": "loaded", "shape": (2, 4, 8, 8),
                 "axes": "CZYX"}]
            ctl._loaded_source_file = path                 # THIS file is loaded
            ctl._axis_combo_source = path                  # combo describes it
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("ZCYX")
            assert ctl._resolve_prepare_image_axes(path) == "ZCYX"
        finally:
            w.setParent(None)

    def test_loaded_file_clause_does_not_grant_trust_when_combo_describes_another(
            self, qt_app, tmp_path):
        """LOW fix 2026-08-02: `_loaded_source_file` is NOT a trust key.  With
        file A loaded but the combo describing a DIFFERENT file, resolving A must
        fall back to A's own header, NOT echo the combo (which describes another
        file)."""
        w = self._widget()
        try:
            ctl = w.controller
            a = str(tmp_path / "a.tif")
            b = str(tmp_path / "b.tif")
            ctl._scanned_files = [
                {"path": a, "stem": "a", "shape": (2, 4, 8, 8), "axes": "CZYX"},
                {"path": b, "stem": "b", "shape": (40, 2, 40, 40), "axes": None}]
            ctl._loaded_source_file = a                    # A is loaded
            ctl._axis_combo_source = b                     # but combo describes B
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("ZCYX")      # B's order
            # Resolving A uses A's own header (clean CZYX), never B's combo value.
            assert ctl._resolve_prepare_image_axes(a) == "CZYX"
        finally:
            w.setParent(None)

    def test_combo_orients_the_selected_unloaded_file(self, qt_app, tmp_path):
        """THE full-fix capability: when the combo DESCRIBES the selected (but
        un-loaded) source (`_axis_combo_source == path`, as detect-on-selection
        sets it), its value orients the Prepare — no Load required.  The user has
        SEEN the 'guessed … — confirm' value and can flip it."""
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "sel.tif")
            ctl._scanned_files = [
                {"path": path, "stem": "sel", "shape": (2, 4, 8, 8),
                 "axes": None}]                            # ambiguous metadata
            ctl._loaded_source_file = None                 # NOT loaded
            ctl._axis_combo_source = path                  # combo describes it
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("ZCYX")      # user's confirmed pick
            assert ctl._resolve_prepare_image_axes(path) == "ZCYX"
        finally:
            w.setParent(None)

    def test_order_for_another_file_never_orients_this_one(self, qt_app, tmp_path):
        """DATA-INTEGRITY invariant (the audit's concern, preserved): the combo
        describes file A (`_axis_combo_source == A`), but Prepare runs on file B
        -> B is oriented by B's OWN header heuristic, never A's combo value."""
        w = self._widget()
        try:
            ctl = w.controller
            a = str(tmp_path / "a.tif")
            b = str(tmp_path / "b.tif")
            ctl._scanned_files = [
                {"path": a, "stem": "a", "shape": (40, 2, 40, 40), "axes": None},
                {"path": b, "stem": "b", "shape": (2, 4, 8, 8), "axes": None}]
            ctl._loaded_source_file = None
            ctl._axis_combo_source = a                     # combo is for A
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("ZCYX")      # A's order
            # Preparing B must use B's own heuristic (CZYX), NOT A's "ZCYX".
            assert ctl._resolve_prepare_image_axes(b) == "CZYX"
        finally:
            w.setParent(None)


class TestApplyDetectedAxesForSource:
    """`_apply_detected_axes_for_source` points the Order combo at the SELECTED
    source from its header scan (zero I/O), so the dropdown reflects that file
    and Prepare can orient it without a Load."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    def test_guessed_4d_sets_combo_and_tracks_source(self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "g.tif")
            ctl._scanned_files = [
                {"path": path, "stem": "g", "shape": (2, 143, 64, 64),
                 "axes": None}]                            # WoundT01-like
            ctl._axis_order_overrides = {}
            ctl._apply_detected_axes_for_source(path)
            assert w.axis_order_combo.currentText().strip() == "CZYX"
            assert ctl._axis_combo_source == path
            # And that combo now orients Prepare with no Load.
            ctl._loaded_source_file = None
            assert ctl._resolve_prepare_image_axes(path) == "CZYX"
        finally:
            w.setParent(None)

    def test_prior_override_wins_on_selection(self, qt_app, tmp_path):
        from napari_epicycle.data_preparation._state import _norm_path
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "o.tif")
            ctl._scanned_files = [
                {"path": path, "stem": "o", "shape": (2, 143, 64, 64),
                 "axes": None}]
            ctl._axis_order_overrides = {_norm_path(path): "ZCYX"}
            ctl._apply_detected_axes_for_source(path)
            assert w.axis_order_combo.currentText().strip() == "ZCYX"
            assert ctl._axis_combo_source == path
        finally:
            w.setParent(None)

    def test_noop_for_unknown_shape(self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            path = str(tmp_path / "u.tif")
            ctl._scanned_files = []                        # nothing scanned
            ctl._axis_combo_source = "sentinel"
            ctl._apply_detected_axes_for_source(path)
            assert ctl._axis_combo_source == "sentinel"    # untouched
        finally:
            w.setParent(None)


class TestNoteAxisOrderOverride:
    """HIGH fix 2026-08-02: a user's Order flip is recorded against the file the
    combo DESCRIBES (`_axis_combo_source`), NOT the loaded file — otherwise, with
    a different file selected than loaded, the flip would misattribute the
    SELECTED file's chosen Order onto the LOADED file and silently mis-orient the
    loaded file's store on its next Prepare/Load."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    def test_override_keyed_on_combo_source_not_loaded_file(self, qt_app, tmp_path):
        from napari_epicycle.data_preparation._state import _norm_path
        w = self._widget()
        try:
            ctl = w.controller
            a = str(tmp_path / "a.tif")                    # loaded
            b = str(tmp_path / "b.tif")                    # selected (combo shows B)
            ctl._loaded_source_file = a
            ctl._axis_combo_source = b
            ctl._axis_order_overrides = {}
            w.axis_order_combo.setCurrentText("ZCYX")      # user flips for B
            ctl.note_axis_order_override()
            assert _norm_path(b) in ctl._axis_order_overrides       # under B
            assert _norm_path(a) not in ctl._axis_order_overrides   # NOT under A
            assert ctl._axis_order_overrides[_norm_path(b)] == "ZCYX"
        finally:
            w.setParent(None)

    def test_falls_back_to_selected_row_then_loaded(self, qt_app, tmp_path):
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt
        from napari_epicycle.data_preparation._state import _norm_path
        w = self._widget()
        try:
            ctl = w.controller
            ctl._axis_combo_source = None                  # combo describes nothing
            ctl._loaded_source_file = None
            ctl._axis_order_overrides = {}
            sel = str(tmp_path / "sel.tif")
            item = QListWidgetItem("sel")
            item.setData(Qt.UserRole, sel)
            w.source_file_list.addItem(item)
            w.source_file_list.setCurrentItem(item)
            w.axis_order_combo.setCurrentText("ZCYX")
            ctl.note_axis_order_override()
            assert ctl._axis_order_overrides.get(_norm_path(sel)) == "ZCYX"
        finally:
            w.setParent(None)


class TestAxisComboSourceReset:
    """LOW fix 2026-08-02: `_axis_combo_source` is cleared on unload + folder
    browse so a stale pointer can never let an Order flip key onto a file no
    longer selected."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    def test_unload_clears_combo_source(self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            ctl._axis_combo_source = str(tmp_path / "x.tif")
            ctl._loaded_source_layers = []
            ctl.unload_source()
            assert ctl._axis_combo_source is None
        finally:
            w.setParent(None)


class TestExtractProvenanceUsesResolvedAxes:
    """Change #4 2026-08-02: the `source_axes` recorded at extract time comes
    from `_source_axes_resolved` (the order the LOADED data was resolved
    through), NOT the live Order combo — which now tracks the SELECTED source, so
    reading it live could record a different file's order for this loaded
    source's provenance."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    def test_source_axes_from_resolved_not_live_combo(self, qt_app, tmp_path,
                                                      monkeypatch):
        from napari_epicycle.data_preparation import controller as ctlmod
        captured = {}

        def _fake_persist(path, patch_name, data, **kwargs):
            captured["source_axes"] = kwargs.get("source_axes")
            return {"patch_name": patch_name,
                    "tiff_path": str(tmp_path / f"{patch_name}.tif")}

        monkeypatch.setattr(ctlmod, "persist_patch", _fake_persist)
        w = self._widget()
        try:
            ctl = w.controller
            w.output_file_path.setText(str(tmp_path))
            ctl._loaded_source_file = str(tmp_path / "loaded.tif")
            ctl._source_axes_resolved = "ZCYX"             # order captured at load
            w.axis_order_combo.setCurrentText("CZYX")      # combo now shows ANOTHER file
            ctl.extracted_patches = {"p0": {
                "data": np.zeros((2, 4, 8, 8), dtype=np.uint16),
                "channels": ["c0", "c1"],
                "bounding_box": "not stored yet",
                "selected_z_range": None,
            }}
            ok = ctl._export_patch_by_name("p0", require_conditioned=False)
            assert ok is True
            assert captured["source_axes"] == "ZCYX"       # resolved, not combo
        finally:
            w.setParent(None)


class _FakeMSLayer:
    """A stand-in multiscale Image layer for the 3D-detail apply test: exposes
    the three attributes `_apply_3d_detail_level` touches."""

    def __init__(self, name, n_levels, multiscale=True):
        self.name = name
        self.multiscale = multiscale
        self.level_shapes = [(8, 8, 8)] * n_levels
        self.locked_data_level = "UNSET"


class _FakeDims:
    def __init__(self, ndisplay=2):
        self.ndisplay = ndisplay


class _FakeViewer3D:
    def __init__(self, layers, ndisplay=2):
        self._by_name = {l.name: l for l in layers}
        self.dims = _FakeDims(ndisplay)

    @property
    def layers(self):
        return self._by_name          # supports viewer.layers[name]


class TestThreeDDetailControl:
    """Feature #2: the 3D-detail slider pins a sharper pyramid level via each
    multiscale layer's `locked_data_level`, applied ONLY in 3D (2D keeps
    napari's automatic per-zoom picking)."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    def test_apply_pins_level_in_3d_and_releases_in_2d(self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            a = _FakeMSLayer("s_ch0", 3)
            b = _FakeMSLayer("s_ch1", 3)
            plain = _FakeMSLayer("plain", 1, multiscale=False)
            ctl.viewer = _FakeViewer3D([a, b, plain], ndisplay=3)
            ctl._loaded_source_layers = ["s_ch0", "s_ch1", "plain"]
            ctl._locked_3d_level = 0
            ctl._source_layers_on_canvas = True
            ctl._apply_3d_detail_level()
            assert a.locked_data_level == 0
            assert b.locked_data_level == 0
            assert plain.locked_data_level == "UNSET"      # non-multiscale untouched
            # Switching to 2D releases the pin -> napari auto per-zoom level.
            ctl.viewer.dims.ndisplay = 2
            ctl._apply_3d_detail_level()
            assert a.locked_data_level is None
            assert b.locked_data_level is None
        finally:
            w.setParent(None)

    def test_apply_clamps_level_to_layer_range(self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            a = _FakeMSLayer("s_ch0", 2)                   # only levels 0,1
            ctl.viewer = _FakeViewer3D([a], ndisplay=3)
            ctl._loaded_source_layers = ["s_ch0"]
            ctl._locked_3d_level = 5                        # out of range
            ctl._source_layers_on_canvas = True
            ctl._apply_3d_detail_level()
            assert a.locked_data_level == 1                 # clamped to n-1
        finally:
            w.setParent(None)

    def test_readout_label_reflects_level_and_warns_on_finest(self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            ctl.viewer = _FakeViewer3D([], ndisplay=2)
            ctl._loaded_source_layers = []                 # apply no-ops
            ctl._pyramid_level_shapes = [
                (143, 1024, 1024), (72, 512, 512), (36, 256, 256)]
            ctl.on_3d_detail_changed(0)
            assert ctl._locked_3d_level == 0
            assert "143×1024×1024" in w.threed_detail_label.text()
            assert "⚠" in w.threed_detail_label.text()
            ctl.on_3d_detail_changed(2)
            assert "36×256×256" in w.threed_detail_label.text()
            assert "coarsest" in w.threed_detail_label.text().lower()
        finally:
            w.setParent(None)

    def test_configure_shows_and_ranges_slider_parked_at_coarsest(self, qt_app,
                                                                  tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            ctl.viewer = _FakeViewer3D([], ndisplay=2)
            ctl._loaded_source_layers = []
            ctl._pyramid_level_shapes = [
                (143, 1024, 1024), (72, 512, 512), (36, 256, 256)]
            ctl._configure_3d_detail(3)
            assert not w.threed_detail_row.isHidden()
            assert w.threed_detail_slider.maximum() == 2
            assert w.threed_detail_slider.value() == 2      # coarsest default
            assert ctl._locked_3d_level == 2
        finally:
            w.setParent(None)

    def test_single_level_hides_control(self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            w.threed_detail_row.setVisible(True)           # so the hide is real
            ctl._configure_3d_detail(1)
            assert w.threed_detail_row.isHidden()
            assert ctl._locked_3d_level is None
        finally:
            w.setParent(None)

    def test_apply_noops_when_layers_off_canvas(self, qt_app, tmp_path):
        """MED-1: with Step 1's layers OFF the shared canvas (step-exit state),
        the never-disconnected ndisplay hook must NOT touch any layer — even with
        a pin still chosen — so 2D/3D toggles in other steps are inert.  Restoring
        the canvas flag (step re-enter) re-applies the pin."""
        w = self._widget()
        try:
            ctl = w.controller
            a = _FakeMSLayer("s_ch0", 3)
            ctl.viewer = _FakeViewer3D([a], ndisplay=3)
            ctl._loaded_source_layers = ["s_ch0"]
            ctl._locked_3d_level = 0
            ctl._source_layers_on_canvas = False           # off-canvas
            ctl._apply_3d_detail_level()
            assert a.locked_data_level == "UNSET"          # untouched
            ctl._source_layers_on_canvas = True            # back on canvas
            ctl._apply_3d_detail_level()
            assert a.locked_data_level == 0                 # pin re-applied
        finally:
            w.setParent(None)

    def test_unload_hides_and_resets_control(self, qt_app, tmp_path):
        w = self._widget()
        try:
            ctl = w.controller
            ctl._pyramid_level_shapes = [(1, 1, 1), (1, 1, 1)]
            ctl._locked_3d_level = 0
            w.threed_detail_row.setVisible(True)
            ctl._loaded_source_layers = []
            ctl.unload_source()
            assert w.threed_detail_row.isHidden()
            assert ctl._locked_3d_level is None
            assert ctl._pyramid_level_shapes == []
        finally:
            w.setParent(None)


# ---------------------------------------------------------------------------
# Progressive 3D — PROGRESSIVE-3D-ONLY: native in 2D, swap to progressive in 3D
# ---------------------------------------------------------------------------
#
# The display is NATIVE at initial load (2D or 3D) and progressive is engaged
# ONLY by an ndisplay-keyed swap: entering 3D (with the checkbox on + engine
# available) swaps the native multiscale layers to progressive layers — DEFERRED
# one event-loop tick so it runs after napari's ndisplay transition (the path
# proven crash-free) — carrying the native slice-computed contrast; entering 2D
# swaps back to native.  The manual 3D-detail slider (feature #2) shows ONLY in
# 3D-with-progressive-OFF.
#
# Behaviour under test (Qt-free — plain-object fakes for viewer / widget /
# checkbox / dims / QTimer / GLIR meter, the vendored seam monkeypatched):
#   * initial `_apply_multiscale_store` is NATIVE (no seam, no meter) in 2D AND
#     in 3D; a load WHILE in 3D + wanted schedules the deferred swap.
#   * entering 3D + wanted -> deferred native->progressive swap (seam called with
#     the CARRIED contrast_limits; `_progressive_active` True; slider hidden;
#     GLIR meter installed).
#   * entering 2D -> progressive->native swap (meter uninstalled;
#     `_progressive_active` False).
#   * 3D + progressive OFF -> stays native + slider configured.
#   * swap fails CLOSED (seam None / raises) -> native re-added.
#   * the box-tool L0 state (`_source_lazy`, `_full_shape`) survives a swap.
#   * `on_progressive_3d_toggled` swaps live in 3D, no visible change in 2D.
#   * `unload_source` clears the tracked-store fields.

import importlib
from unittest.mock import MagicMock

from napari_epicycle.data_preparation import controller as _ctl_mod


# ---------------------------------------------------------------------------
# Qt-free fakes: layer / evented layer list / viewer (with dims) / widget /
# checkbox / QTimer / GLIR meter.
# ---------------------------------------------------------------------------
class _ProgFakeLayer:
    def __init__(self, name, multiscale=None):
        self.name = name
        self.visible = True
        self.gamma = 1.0
        self.colormap = "gray"
        self.blending = "additive"
        self.contrast_limits = None
        self.locked_data_level = None
        self._multiscale = multiscale
        # Records the live-apply recompute call so a test can prove the change
        # drove a REAL corner recompute (not the inert camera re-emit).
        self.apply_zoom_depth_calls = 0

    @property
    def multiscale(self):
        return bool(self._multiscale)

    def apply_zoom_depth_now(self):
        self.apply_zoom_depth_calls += 1


class _ProgEvent:
    def __init__(self, value):
        self.value = value


class _ProgSignal:
    def __init__(self):
        self._cbs = []

    def connect(self, cb):
        self._cbs.append(cb)

    def disconnect(self, cb=None):
        if cb is None:
            self._cbs.clear()
        else:
            self._cbs = [c for c in self._cbs if c is not cb]

    def emit(self, event):
        for cb in list(self._cbs):
            cb(event)


class _ProgEvents:
    def __init__(self):
        self.removed = _ProgSignal()
        self.inserted = _ProgSignal()


class _ProgFakeLayerList(list):
    """List that looks layers up by name and FIRES inserted/removed events (so
    the seam's `removed` GLIR-meter teardown runs on a real-ish removal)."""

    def __init__(self):
        super().__init__()
        self.events = _ProgEvents()

    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(l, "name", None) == item for l in self)
        return list.__contains__(self, item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if getattr(l, "name", None) == key:
                    return l
            raise KeyError(key)
        return list.__getitem__(self, key)

    def append(self, layer):
        list.append(self, layer)
        self.events.inserted.emit(_ProgEvent(layer))

    def remove(self, layer):
        list.remove(self, layer)
        self.events.removed.emit(_ProgEvent(layer))


class _ProgFakeDims:
    def __init__(self, ndisplay=2):
        self.ndisplay = ndisplay


class _ProgFakeViewer:
    def __init__(self, ndisplay=2):
        self.layers = _ProgFakeLayerList()
        self.dims = _ProgFakeDims(ndisplay)

    def add_image(self, data, name=None, colormap=None, blending=None,
                  multiscale=False, contrast_limits=None, gamma=None,
                  visible=None):
        layer = _ProgFakeLayer(name, multiscale=multiscale)
        if colormap is not None:
            layer.colormap = colormap
        if blending is not None:
            layer.blending = blending
        layer.contrast_limits = contrast_limits
        if gamma is not None:
            layer.gamma = gamma
        if visible is not None:
            layer.visible = visible
        self.layers.append(layer)
        return layer


class _ProgFakeCheckBox:
    def __init__(self, checked=True):
        self._checked = checked

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = bool(checked)


class _ProgStatusLabel:
    def setText(self, *a):
        pass

    def setStyleSheet(self, *a):
        pass


class _ProgFakeWidget:
    """Only what the display path + slider-coexistence read: the checkbox, the
    status label, and the two 3D-detail hooks (as call spies)."""

    def __init__(self, checked=True):
        self.progressive_3d_checkbox = _ProgFakeCheckBox(checked)
        self.source_status_label = _ProgStatusLabel()
        self.hide_calls = 0
        self.configure_calls = []
        # "Zoom depth" spies: the current combo mode (the last `mode` arg
        # `configure_zoom_depth` was asked to select), a hide counter, and the
        # sync-on-show call log (each entry is the requested mode).
        self.zoom_depth_mode = None
        self.hide_zoom_depth_calls = 0
        self.configure_zoom_depth_calls = []

    def hide_3d_detail(self):
        self.hide_calls += 1

    def configure_3d_detail(self, n_levels, default_level=None):
        self.configure_calls.append((n_levels, default_level))

    def configure_zoom_depth(self, mode):
        # Sync-on-show: record the requested mode AND mirror it as the fake
        # combo's current mode so a test can read the resulting selection.
        self.configure_zoom_depth_calls.append(mode)
        self.zoom_depth_mode = mode

    def hide_zoom_depth(self):
        self.hide_zoom_depth_calls += 1


class _ImmediateQTimer:
    """Fake QTimer whose singleShot runs the callback IMMEDIATELY (so the
    deferred swap executes inline in a headless test)."""

    def __init__(self):
        self.scheduled = 0

    def singleShot(self, ms, fn):
        self.scheduled += 1
        fn()


class _DeferredQTimer:
    """Fake QTimer that RECORDS the callback without running it — so a test can
    assert the swap was deferred (still native right after the ndisplay change),
    then fire it via `flush()`."""

    def __init__(self):
        self.pending = []

    def singleShot(self, ms, fn):
        self.pending.append(fn)

    def flush(self):
        fns, self.pending = self.pending, []
        for fn in fns:
            fn()


class _FakeMeter:
    """Stand-in for the process-wide GLIR meter so the seam's real
    install/register/teardown machinery can be exercised headlessly."""

    def __init__(self):
        self._installed = False

    def is_installed(self):
        return self._installed

    def install(self):
        self._installed = True

    def uninstall(self):
        self._installed = False


def _make_prog_ctl(viewer, widget):
    """A real `DataPreparationController` instance (so its methods call each
    other normally) with only the state the display path touches, and `_log` /
    `_on_layers_changed` shadowed to no-ops.  `_ndisplay_connected=True` makes
    `_connect_ndisplay_hook` a no-op."""
    from napari_epicycle.data_preparation.controller import (
        DataPreparationController as C)
    ctl = C.__new__(C)
    ctl.viewer = viewer
    ctl.widget = widget
    ctl._loaded_source_layers = []
    ctl._source_display_channels = None
    ctl._progressive_active = False
    ctl._zoom_depth_wanted = None
    ctl._current_multiscale_store = None
    ctl._current_multiscale_display_name = None
    ctl._current_multiscale_src = None
    ctl._locked_3d_level = None
    ctl._pyramid_level_shapes = []
    ctl._ndisplay_connected = True
    ctl._source_layers_on_canvas = False
    ctl._largedata_src_path = None
    ctl._channel_index_by_name = {}
    ctl._source_lazy = None
    ctl._source_axes_resolved = None
    ctl._axis_combo_source = None
    ctl._saved_channel_selection = None
    ctl._log = lambda *a, **k: None
    ctl._on_layers_changed = lambda *a, **k: None
    return ctl


def _levels_meta(has_channel=False, n_ch=2):
    """A 2-level pyramid + NGFF axes meta (single-channel ZYX by default)."""
    if has_channel:
        levels = [np.zeros((n_ch, 8, 8, 8), dtype=np.uint16),
                  np.zeros((n_ch, 4, 4, 4), dtype=np.uint16)]
        meta = {"axes": [{"type": "channel"}, {"type": "space"},
                         {"type": "space"}, {"type": "space"}]}
    else:
        levels = [np.zeros((8, 8, 8), dtype=np.uint16),
                  np.zeros((4, 4, 4), dtype=np.uint16)]
        meta = {"axes": [{"type": "space"}, {"type": "space"},
                         {"type": "space"}]}
    return levels, meta


class TestProgressive3DDisplay:
    def _prog_module(self):
        return importlib.import_module(
            "packages.epicycle_largedata.progressive")

    @pytest.fixture(autouse=True)
    def _reset_paint_guard(self):
        """These tests call the REAL `install_glir_if_needed`, which installs
        napari's `LayerDelegate._paint_loading` guard.  Production no longer
        tears it down on the per-swap idle path (its crash window is the native
        re-add that follows), so reset it here to avoid leaking the class-level
        napari monkeypatch across tests.  A no-op when it was never installed."""
        yield
        try:
            importlib.import_module(
                "packages.epicycle_largedata.progressive"
            )._uninstall_paint_loading_guard()
        except Exception:
            pass

    def _patch_open(self, monkeypatch, levels, meta):
        monkeypatch.setattr(_ctl_mod, "open_multiscale",
                            lambda _p: (levels, meta))

    def _lightweight_seam(self, monkeypatch, prog, calls):
        """A spy for `add_progressive_image` that just appends a progressive
        layer + records the call (name/colormap/blending/contrast/levels)."""
        monkeypatch.setattr(prog, "progressive_available", lambda: True)

        def _spy(viewer, levels, name=None, colormap=None, blending=None,
                 contrast_limits=None, scale=None, **_kw):
            layer = _ProgFakeLayer(name)
            layer.contrast_limits = contrast_limits
            viewer.layers.append(layer)
            calls.append({"name": name, "colormap": colormap,
                          "blending": blending,
                          "contrast_limits": contrast_limits,
                          "levels": list(levels)})
            return layer

        monkeypatch.setattr(prog, "add_progressive_image", _spy)
        return calls

    def _meter_seam(self, monkeypatch, prog, calls):
        """A spy that also drives the REAL GLIR-meter registry (install +
        register + teardown wiring) against a fake meter, so meter
        install/uninstall can be asserted at the controller level."""
        meter = _FakeMeter()
        monkeypatch.setattr(prog, "_load_metering", lambda: meter)
        monkeypatch.setattr(prog, "progressive_available", lambda: True)
        # The meter is OFF in production (its deferred-DELETE reorder is the
        # 3D<->2D crash root); force it ON here so these tests still exercise the
        # flag-gated install/uninstall lifecycle at the controller level.
        monkeypatch.setattr(prog, "_METER_ENABLED", True)
        prog._live_refs.clear()
        prog._teardown_handlers.clear()

        def _spy(viewer, levels, name=None, colormap=None, blending=None,
                 contrast_limits=None, scale=None, **_kw):
            prog.install_glir_if_needed()
            layer = _ProgFakeLayer(name)
            layer.contrast_limits = contrast_limits
            viewer.layers.append(layer)
            prog._register(layer)
            prog._wire_teardown(viewer, layer)
            calls.append({"name": name, "colormap": colormap,
                          "contrast_limits": contrast_limits})
            return layer

        monkeypatch.setattr(prog, "add_progressive_image", _spy)
        return meter

    # -- (1) initial display is NATIVE (no seam, no swap) --------------------
    def test_initial_display_is_native_in_2d(self, monkeypatch):
        prog = self._prog_module()
        calls = self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        widget = _ProgFakeWidget(checked=True)         # wanted, but 2D
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None

        n = ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")

        assert n == 2
        assert calls == []                             # seam NEVER engaged in 2D
        assert timer.scheduled == 0                    # no deferred swap
        assert ctl._progressive_active is False
        assert len(viewer.layers) == 1
        assert viewer.layers[0]._multiscale is True    # native multiscale
        assert widget.configure_calls == []            # slider hidden in 2D
        assert widget.hide_calls >= 1

    def test_initial_display_native_in_3d_progressive_off(self, monkeypatch):
        prog = self._prog_module()
        calls = self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=False)        # progressive OFF
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None

        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")

        assert calls == []                             # no progressive
        assert timer.scheduled == 0
        assert ctl._progressive_active is False
        assert viewer.layers[0]._multiscale is True
        assert widget.configure_calls == [(2, 1)]      # slider SHOWN in 3D-OFF
        assert widget.hide_calls == 0

    def test_load_in_3d_progressive_on_schedules_and_swaps(self, monkeypatch):
        prog = self._prog_module()
        calls = self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()                     # runs the deferred swap
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=True)         # wanted + 3D
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None

        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")

        assert timer.scheduled == 2                    # swap tick + deferred reveal
        assert len(calls) == 1                         # seam engaged (after tick)
        assert ctl._progressive_active is True
        assert widget.hide_calls >= 1                  # slider hidden
        assert widget.configure_calls == []            # never configured
        assert ctl._loaded_source_layers == ["vol"]

    # -- (2) entering 3D swaps native -> progressive (deferred + contrast) ----
    def test_entering_3d_defers_the_swap(self, monkeypatch):
        """The entering-3D swap must be DEFERRED (QTimer.singleShot) — synchronous
        it would trip the vispy GLIR crash.  Right after the ndisplay change the
        display is STILL native; only after the tick fires is it progressive."""
        prog = self._prog_module()
        calls = self._lightweight_seam(monkeypatch, prog, [])
        timer = _DeferredQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        widget = _ProgFakeWidget(checked=True)
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")   # 2D native

        viewer.dims.ndisplay = 3
        ctl._on_ndisplay_changed()
        # DEFERRED: not yet swapped.
        assert len(timer.pending) == 1
        assert calls == []
        assert ctl._progressive_active is False
        assert viewer.layers[0]._multiscale is True

        timer.flush()                                  # napari transition settled
        assert len(calls) == 1
        assert ctl._progressive_active is True

    def test_entering_3d_carries_contrast_and_installs_meter(self, monkeypatch):
        prog = self._prog_module()
        calls = []
        meter = self._meter_seam(monkeypatch, prog, calls)
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        widget = _ProgFakeWidget(checked=True)
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")   # native, 2D

        # napari computed a contrast from the shown slice; the swap must carry it.
        viewer.layers["vol"].contrast_limits = [10, 200]
        assert meter.is_installed() is False

        viewer.dims.ndisplay = 3
        ctl._on_ndisplay_changed()

        assert len(calls) == 1
        assert calls[0]["contrast_limits"] == [10, 200]   # CARRIED
        assert ctl._progressive_active is True
        assert widget.hide_calls >= 1                      # slider hidden
        assert meter.is_installed() is True                # meter installed

    # -- (3) entering 2D swaps back to native + uninstalls the meter ----------
    def test_entering_2d_swaps_back_and_uninstalls_meter(self, monkeypatch):
        prog = self._prog_module()
        calls = []
        meter = self._meter_seam(monkeypatch, prog, calls)
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=True)
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")   # -> progressive
        assert ctl._progressive_active is True
        assert meter.is_installed() is True

        viewer.dims.ndisplay = 2
        ctl._on_ndisplay_changed()

        assert ctl._progressive_active is False
        assert meter.is_installed() is False               # meter uninstalled
        assert len(viewer.layers) == 1
        assert viewer.layers[0]._multiscale is True        # native again

    def test_entering_2d_multichannel_swaps_all_back(self, monkeypatch):
        prog = self._prog_module()
        calls = []
        self._meter_seam(monkeypatch, prog, calls)
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=True, n_ch=2)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")
        assert ctl._loaded_source_layers == ["vol_ch0", "vol_ch1"]
        assert ctl._progressive_active is True

        viewer.dims.ndisplay = 2
        ctl._on_ndisplay_changed()
        assert ctl._progressive_active is False
        assert ctl._loaded_source_layers == ["vol_ch0", "vol_ch1"]
        assert all(l._multiscale is True for l in viewer.layers)

    # -- (4) progressive OFF stays native in 3D + slider configured ----------
    def test_progressive_off_stays_native_in_3d_and_configures_slider(
            self, monkeypatch):
        prog = self._prog_module()
        calls = self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        widget = _ProgFakeWidget(checked=False)        # OFF
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")

        viewer.dims.ndisplay = 3
        ctl._on_ndisplay_changed()

        assert calls == []                             # never progressive
        assert timer.scheduled == 0
        assert ctl._progressive_active is False
        assert widget.configure_calls[-1] == (2, 1)    # slider configured in 3D
        assert viewer.layers[0]._multiscale is True

    # -- (5) swap fails CLOSED (seam None / raises) -> native re-added --------
    def test_swap_fail_closed_seam_returns_none(self, monkeypatch):
        prog = self._prog_module()
        monkeypatch.setattr(prog, "progressive_available", lambda: True)
        monkeypatch.setattr(prog, "add_progressive_image",
                            lambda *a, **k: None)      # fail closed
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")

        assert ctl._progressive_active is False        # stayed native
        assert len(viewer.layers) == 1
        assert viewer.layers[0]._multiscale is True    # native re-added
        assert viewer.layers[0].visible is True        # revealed (not left hidden)
        assert ctl._loaded_source_layers == ["vol"]

    def test_swap_fail_closed_seam_raises(self, monkeypatch):
        prog = self._prog_module()
        monkeypatch.setattr(prog, "progressive_available", lambda: True)

        def _boom(*a, **k):
            raise RuntimeError("seam blew up")

        monkeypatch.setattr(prog, "add_progressive_image", _boom)
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")   # no crash

        assert ctl._progressive_active is False
        assert len(viewer.layers) == 1
        assert viewer.layers[0]._multiscale is True    # native re-added
        assert viewer.layers[0].visible is True        # revealed (not left hidden)

    # -- (6) box-tool L0 state survives a swap -------------------------------
    def test_box_tool_state_preserved_across_swap(self, monkeypatch):
        import types
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.box_tool = types.SimpleNamespace()
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")   # native, 2D

        lazy_before = ctl._source_lazy
        shape_before = ctl.box_tool._full_shape
        assert lazy_before is not None
        assert shape_before == (8, 8, 8)

        # rotate 2D -> 3D (swap to progressive), then back (swap to native)
        viewer.dims.ndisplay = 3
        ctl._on_ndisplay_changed()
        assert ctl._progressive_active is True
        assert ctl._source_lazy is lazy_before             # L0 handle untouched
        assert ctl.box_tool._full_shape == shape_before    # box full-shape untouched

        viewer.dims.ndisplay = 2
        ctl._on_ndisplay_changed()
        assert ctl._source_lazy is lazy_before
        assert ctl.box_tool._full_shape == shape_before

    # -- (7) live toggle: swaps in 3D, no visible change in 2D ---------------
    def test_toggle_in_3d_swaps_live(self, monkeypatch):
        prog = self._prog_module()
        calls = self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=False)        # start OFF (native, 3D)
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")
        assert ctl._progressive_active is False
        assert viewer.layers[0]._multiscale is True

        # turn ON -> live swap to progressive
        widget.progressive_3d_checkbox._checked = True
        ctl.on_progressive_3d_toggled(True)
        assert len(calls) == 1
        assert ctl._progressive_active is True

        # turn OFF -> live swap back to native + slider re-appears
        widget.progressive_3d_checkbox._checked = False
        ctl.on_progressive_3d_toggled(False)
        assert ctl._progressive_active is False
        assert viewer.layers[0]._multiscale is True
        assert widget.configure_calls[-1] == (2, 1)

    def test_toggle_in_2d_no_visible_change(self, monkeypatch):
        prog = self._prog_module()
        calls = self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        widget = _ProgFakeWidget(checked=False)
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")

        widget.progressive_3d_checkbox._checked = True
        ctl.on_progressive_3d_toggled(True)            # in 2D -> no swap

        assert calls == []
        assert timer.scheduled == 0
        assert ctl._progressive_active is False
        assert viewer.layers[0]._multiscale is True    # still native

    # -- (8) "Zoom depth" live change ----------------------------------------
    # combo indices for the modes exercised below (index <-> mode, controller-owned)
    _FULL_IDX = _ctl_mod.ZOOM_DEPTH_MODES.index(1.0)      # "Full"
    _HALF_IDX = _ctl_mod.ZOOM_DEPTH_MODES.index(0.5)      # "50%"
    _OFF_IDX = _ctl_mod.ZOOM_DEPTH_MODES.index(None)      # "Off"

    def test_zoom_depth_change_applies_live_when_progressive(self, monkeypatch):
        """Selecting a mode while progressive is active in 3D stores the wanted
        mode, stamps `_zoom_depth` on EVERY live progressive layer, AND drives a
        REAL per-layer corner recompute (`apply_zoom_depth_now`).  The recompute
        assertion is what makes this non-vacuous: it FAILS if the handler reverts
        to an inert `camera.events.zoom()` re-emit (which no fake layer records
        and which does not recompute corners on a stationary camera)."""
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=True, n_ch=2)   # two live layers
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")
        assert ctl._progressive_active is True
        assert ctl._loaded_source_layers == ["vol_ch0", "vol_ch1"]
        # swap seeds each layer field from the (default None) wanted mode
        for name in ("vol_ch0", "vol_ch1"):
            assert viewer.layers[name]._zoom_depth is None
            assert viewer.layers[name].apply_zoom_depth_calls == 0

        ctl.on_zoom_depth_changed(self._FULL_IDX)            # -> Full (1.0)
        assert ctl._zoom_depth_wanted == 1.0
        for name in ("vol_ch0", "vol_ch1"):
            assert viewer.layers[name]._zoom_depth == 1.0            # LIVE stamp
            # REAL recompute driven on EACH live layer (not the inert re-emit)
            assert viewer.layers[name].apply_zoom_depth_calls == 1

        ctl.on_zoom_depth_changed(self._OFF_IDX)             # -> Off (None)
        assert ctl._zoom_depth_wanted is None
        for name in ("vol_ch0", "vol_ch1"):
            assert viewer.layers[name]._zoom_depth is None
            assert viewer.layers[name].apply_zoom_depth_calls == 2   # recomputed again

    def test_zoom_depth_change_applies_fraction_and_cube_live(self, monkeypatch):
        """The live recompute wiring fires for a FRACTION (50%) and CUBE too, not
        just Full/Off — each routes through `on_zoom_depth_changed`, stamps the
        mode on EVERY live layer, and drives a REAL `apply_zoom_depth_now`.  Uses
        the mode->index lookups so it stays robust to combo reordering."""
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=True, n_ch=2)   # two live layers
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")
        assert ctl._progressive_active is True

        cube_idx = _ctl_mod.ZOOM_DEPTH_MODES.index("cube")
        for step, (idx, mode) in enumerate(
                [(self._HALF_IDX, 0.5), (cube_idx, "cube")], start=1):
            ctl.on_zoom_depth_changed(idx)
            assert ctl._zoom_depth_wanted == mode
            for name in ("vol_ch0", "vol_ch1"):
                assert viewer.layers[name]._zoom_depth == mode      # LIVE stamp
                # REAL recompute fires for fraction + cube too (not just Full)
                assert viewer.layers[name].apply_zoom_depth_calls == step

    def test_zoom_depth_change_stores_but_no_apply_when_not_progressive(
            self, monkeypatch):
        """OFF-progressive (native 3D): the change STILL records the wanted mode
        (so it applies on the next progressive entry) but touches no native
        layer."""
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=False))   # prog OFF
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")
        assert ctl._progressive_active is False

        ctl.on_zoom_depth_changed(self._FULL_IDX)
        assert ctl._zoom_depth_wanted == 1.0                 # stored for later
        # native layer was NOT stamped (only progressive layers carry the mode)
        assert not hasattr(viewer.layers[0], "_zoom_depth")

    def test_zoom_depth_persists_across_native_progressive_swap(
            self, monkeypatch):
        """A swap builds a FRESH progressive layer (defaulting None); the
        controller must re-stamp the remembered wanted mode onto it."""
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")  # 2D native
        ctl._zoom_depth_wanted = 0.5                         # a prior selection

        viewer.dims.ndisplay = 3
        ctl._on_ndisplay_changed()                           # swap -> progressive
        assert ctl._progressive_active is True
        assert viewer.layers["vol"]._zoom_depth == 0.5       # re-stamped on swap

    def test_zoom_depth_combo_visibility_tracks_ndisplay_and_progressive(
            self, monkeypatch):
        """The combo is shown only in 3D-with-progressive, hidden in 2D."""
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=True)
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")  # 3D prog
        assert len(widget.configure_zoom_depth_calls) >= 1   # shown (sync-on-show)

        viewer.dims.ndisplay = 2
        ctl._on_ndisplay_changed()                           # -> 2D native
        assert widget.hide_zoom_depth_calls >= 1             # hidden in 2D

    def test_zoom_depth_combo_hidden_in_3d_when_progressive_off(
            self, monkeypatch):
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=False)              # progressive OFF
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")

        assert widget.configure_zoom_depth_calls == []       # never shown
        assert widget.hide_zoom_depth_calls >= 1             # hidden (native 3D)

    def test_unload_hides_zoom_depth_and_resets_wanted(self, monkeypatch):
        """Unload hides the zoom-depth combo (its refresh can't once the store is
        cleared) and resets the remembered mode for the next store."""
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=True)
        ctl = _make_prog_ctl(viewer, widget)
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")  # 3D prog
        ctl.on_zoom_depth_changed(self._FULL_IDX)
        assert ctl._zoom_depth_wanted == 1.0

        hide_before = widget.hide_zoom_depth_calls
        ctl.unload_source()                                  # REAL unload

        assert ctl._zoom_depth_wanted is None                # reset for next store
        assert widget.hide_zoom_depth_calls > hide_before    # combo hidden

    # -- (9) sync-on-show: the selection can't drift from the render mode -------
    def test_zoom_depth_show_syncs_selection_to_mirror(self, monkeypatch):
        """The SHOW branch of `_refresh_zoom_depth_for_ndisplay` syncs the combo
        selection to `_zoom_depth_wanted` every time it appears, so a stale mode
        cannot survive a reset-to-None.  FAILS if the show branch reverts to a
        bare show (which leaves the combo as-is: `configure_zoom_depth_calls`
        stays empty and the selection sticks)."""
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=True)
        ctl = _make_prog_ctl(viewer, widget)
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")  # 3D prog

        # A stale Full selection whose mirror has since been reset to None.
        widget.zoom_depth_mode = 1.0
        ctl._zoom_depth_wanted = None

        widget.configure_zoom_depth_calls.clear()
        ctl._refresh_zoom_depth_for_ndisplay()               # shown state

        assert widget.configure_zoom_depth_calls == [None]   # synced to the mirror
        assert widget.zoom_depth_mode is None                # combo reset to Off

    def test_zoom_depth_combo_resyncs_off_on_store_change(
            self, monkeypatch):
        """End-to-end store change: store A set to Full, then load store B (routes
        through the REAL unload, resetting the wanted mirror to None).  Store B
        renders band mode, so the combo MUST end on Off — not stuck on Full."""
        prog = self._prog_module()
        self._lightweight_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        widget = _ProgFakeWidget(checked=True)
        ctl = _make_prog_ctl(viewer, widget)                 # REAL unload (unstubbed)
        # store A: progressive 3D, user picks Full.
        ctl._apply_multiscale_store("A.zarr", "volA", src_path="A.tif")
        ctl.on_zoom_depth_changed(self._FULL_IDX)
        widget.zoom_depth_mode = 1.0                         # user's selection
        assert ctl._zoom_depth_wanted == 1.0

        # store B: load routes through unload (resets wanted) + re-displays.
        ctl._apply_multiscale_store("B.zarr", "volB", src_path="B.tif")

        assert ctl._zoom_depth_wanted is None                # reset by unload
        assert widget.configure_zoom_depth_calls[-1] is None   # last sync = Off
        assert widget.zoom_depth_mode is None                # combo not stuck on Full

    def test_toggle_noops_when_no_store_displayed(self, monkeypatch):
        prog = self._prog_module()
        monkeypatch.setattr(prog, "progressive_available", lambda: True)
        spy = MagicMock()
        monkeypatch.setattr(prog, "add_progressive_image", spy)
        ctl = _make_prog_ctl(_ProgFakeViewer(ndisplay=3),
                             _ProgFakeWidget(checked=True))
        ctl._current_multiscale_store = None           # nothing displayed
        ctl.on_progressive_3d_toggled(True)
        spy.assert_not_called()

    # -- progressive_wanted gate (unchanged semantics) -----------------------
    def test_progressive_wanted_gate(self, monkeypatch):
        prog = self._prog_module()
        monkeypatch.setattr(prog, "progressive_available", lambda: True)
        ctl = _make_prog_ctl(_ProgFakeViewer(), _ProgFakeWidget(checked=True))
        assert ctl.progressive_wanted() is True
        monkeypatch.setattr(prog, "progressive_available", lambda: False)
        assert ctl.progressive_wanted() is False
        monkeypatch.setattr(prog, "progressive_available", lambda: True)
        ctl2 = _make_prog_ctl(_ProgFakeViewer(), _ProgFakeWidget(checked=False))
        assert ctl2.progressive_wanted() is False
        ctl3 = _make_prog_ctl(_ProgFakeViewer(), object())
        assert ctl3.progressive_wanted() is False

    # -- unload clears the tracked-store fields ------------------------------
    def test_unload_clears_tracked_store_fields(self):
        ctl = _make_prog_ctl(_ProgFakeViewer(), _ProgFakeWidget(checked=True))
        ctl._current_multiscale_store = "s.zarr"
        ctl._current_multiscale_display_name = "vol"
        ctl._current_multiscale_src = "vol.tif"
        ctl._progressive_active = True
        ctl.unload_source()
        assert ctl._current_multiscale_store is None
        assert ctl._current_multiscale_display_name is None
        assert ctl._current_multiscale_src is None
        assert ctl._progressive_active is False

    # ======================================================================
    # crash-fix hardening (3D<->2D swap) + 3D-detail pin preservation
    # ======================================================================
    def test_ndisplay_hook_connects_the_handler(self):
        """The ndisplay hook plain-connects `_on_ndisplay_changed` (ordering is
        deliberately NOT used as a lever — napari runs its core callbacks first
        regardless; see `_connect_ndisplay_hook`) and is idempotent."""
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C)
        calls = []

        class _Emitter:
            def connect(self, cb):
                calls.append(cb)

        class _V:
            class dims:
                class events:
                    ndisplay = _Emitter()

        ctl = C.__new__(C)
        ctl.viewer = _V()
        ctl._ndisplay_connected = False
        ctl._log = lambda *a, **k: None
        ctl._connect_ndisplay_hook()
        ctl._connect_ndisplay_hook()   # idempotent — second call is a no-op
        assert calls == [ctl._on_ndisplay_changed]
        assert ctl._ndisplay_connected is True

    def test_configure_3d_detail_preserves_existing_pin(self):
        """Re-entering 3D / step reshow must NOT reset a user-pinned level to
        coarsest: `_configure_3d_detail` parks at the existing valid pin (the
        widget's `configure_3d_detail` always APPLIES its default_level)."""
        widget = _ProgFakeWidget(checked=False)
        ctl = _make_prog_ctl(_ProgFakeViewer(ndisplay=3), widget)
        ctl._locked_3d_level = 0                       # user pinned finest
        ctl._configure_3d_detail(3)
        assert widget.configure_calls[-1] == (3, 0)   # pin preserved (not 2)

    def test_configure_3d_detail_defaults_coarsest_without_pin(self):
        """Initial display (no pin) parks at the coarsest level (n-1)."""
        widget = _ProgFakeWidget(checked=False)
        ctl = _make_prog_ctl(_ProgFakeViewer(ndisplay=3), widget)
        ctl._locked_3d_level = None
        ctl._configure_3d_detail(3)
        assert widget.configure_calls[-1] == (3, 2)   # coarsest

    def test_configure_3d_detail_ignores_out_of_range_pin(self):
        """A stale pin outside the new store's level range falls back to
        coarsest (store swapped for one with fewer levels)."""
        widget = _ProgFakeWidget(checked=False)
        ctl = _make_prog_ctl(_ProgFakeViewer(ndisplay=3), widget)
        ctl._locked_3d_level = 9                       # out of range for 3 levels
        ctl._configure_3d_detail(3)
        assert widget.configure_calls[-1] == (3, 2)   # coarsest fallback

    # -- multiscale contrast estimation (fixes coarsest-level over-saturation) --
    def test_estimate_contrast_none_for_uint8(self):
        import numpy as np
        ctl = _make_prog_ctl(_ProgFakeViewer(), _ProgFakeWidget())
        lv = np.zeros((8, 8, 8), dtype=np.uint8)
        lv[0, 0, 0] = 200
        assert ctl._estimate_multiscale_contrast([lv]) is None  # napari does 0-255

    def test_estimate_contrast_none_for_flat(self):
        import numpy as np
        ctl = _make_prog_ctl(_ProgFakeViewer(), _ProgFakeWidget())
        lv = np.full((8, 8, 8), 42, dtype=np.uint16)
        assert ctl._estimate_multiscale_contrast([lv]) is None  # hi !> lo

    def test_estimate_contrast_prefers_finest_level_within_budget(self):
        import numpy as np
        ctl = _make_prog_ctl(_ProgFakeViewer(), _ProgFakeWidget())
        finest = np.zeros((16, 16, 16), dtype=np.uint16)
        finest[0] = 500                    # bright detail present only at finest
        coarsest = np.full((8, 8, 8), 50, dtype=np.uint16)
        cl = ctl._estimate_multiscale_contrast([finest, coarsest])
        assert cl is not None
        assert cl[1] > 100                 # read the FINEST (500s), not coarsest (50)

    def test_estimate_contrast_recovers_bright_peak(self):
        """The nuclei case: bright voxels on a dark background — the estimate's
        max reflects the bright peak (percentile 99.9), not the background."""
        import numpy as np
        ctl = _make_prog_ctl(_ProgFakeViewer(), _ProgFakeWidget())
        lv = np.full((32, 32, 32), 20, dtype=np.uint16)     # dark background
        lv[:4] = 4000                                        # ~12% bright voxels
        cl = ctl._estimate_multiscale_contrast([lv])
        assert cl is not None
        assert cl[0] < 100 < cl[1]                           # lo near bg, hi near peak
        assert cl[1] >= 1000                                 # bright peak recovered

    def test_multiscale_add_passes_estimated_contrast_to_add_image(self):
        """End-to-end: the multiscale display path passes the fine-level
        percentile contrast to add_image, not napari's coarsest-level default."""
        import numpy as np
        viewer = _ProgFakeViewer(ndisplay=2)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=False))
        lv0 = np.full((4, 16, 16), 20, dtype=np.uint16)   # dark background
        lv0[:1] = 3000                                     # bright detail (finest)
        lv1 = np.full((2, 8, 8), 20, dtype=np.uint16)      # coarser, diluted
        ctl._add_source_display_layers([("vol", [lv0, lv1])])
        layer = viewer.layers["vol"]
        assert layer.contrast_limits is not None
        assert layer.contrast_limits[1] >= 1000            # bright peak, not bg 20

    def test_swap_to_progressive_hides_each_channel_during_build(
            self, monkeypatch):
        """Crash safety: in the multi-channel 2D->3D build each new progressive
        VOLUME node stays hidden during the synchronous build (so a later
        channel's build flush never draws an earlier half-built node), then all
        are revealed on the next tick (here inline via the immediate timer)."""
        prog = self._prog_module()
        monkeypatch.setattr(prog, "progressive_available", lambda: True)
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=True, n_ch=2)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None

        # At each progressive add, snapshot the visibility of the layers ALREADY
        # on the canvas (the earlier channels).
        visibility_at_add = []

        def _spy(v, lv, name=None, colormap=None, blending=None,
                 contrast_limits=None, scale=None, **_kw):
            visibility_at_add.append([l.visible for l in v.layers])
            layer = _ProgFakeLayer(name)
            layer.contrast_limits = contrast_limits
            v.layers.append(layer)          # defaults visible=True, like the seam
            return layer

        monkeypatch.setattr(prog, "add_progressive_image", _spy)

        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")  # native 2D
        viewer.dims.ndisplay = 3
        ctl._on_ndisplay_changed()          # build + deferred reveal run inline

        assert len(visibility_at_add) == 2                    # two channel adds
        assert all(v is False for v in visibility_at_add[0])  # ch1-native hidden
        assert all(v is False for v in visibility_at_add[1])  # ch0-prog hidden
        assert all(l.visible for l in viewer.layers)          # revealed on reveal tick

    def test_deferred_reveal_shows_layers_and_kicks_loader(self, monkeypatch):
        """The reveal + streaming kick run on a SEPARATE tick from the build, so
        the layers stay hidden through the build flush, then become visible and
        each vendored loader's `_check` is called to engage 3D refinement."""
        prog = self._prog_module()
        monkeypatch.setattr(prog, "progressive_available", lambda: True)
        timer = _DeferredQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        checks = []                          # records loader._check() calls

        def _spy(v, lv, name=None, colormap=None, blending=None,
                 contrast_limits=None, scale=None, **_kw):
            layer = _ProgFakeLayer(name)
            layer.contrast_limits = contrast_limits

            class _Loader:
                def _check(_self):
                    checks.append(name)

            layer.metadata = {"progressive_loader": _Loader()}
            v.layers.append(layer)           # defaults visible=True, like the seam
            return layer

        monkeypatch.setattr(prog, "add_progressive_image", _spy)

        viewer = _ProgFakeViewer(ndisplay=2)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")  # 2D native

        viewer.dims.ndisplay = 3
        ctl._on_ndisplay_changed()
        timer.flush()                        # run the swap tick -> builds HIDDEN

        assert ctl._progressive_active is True
        assert viewer.layers["vol"].visible is False   # built hidden
        assert checks == []                            # not kicked yet
        assert len(timer.pending) == 1                 # the deferred reveal

        timer.flush()                        # run the reveal tick
        assert viewer.layers["vol"].visible is True    # revealed after the flush
        assert checks == ["vol"]                       # loader streaming kicked

    def test_swap_to_native_hides_all_channels_before_removing_any(
            self, monkeypatch):
        """Multi-channel 3D->2D: EVERY progressive layer is hidden (visual
        disconnected) before ANY is removed, so a second channel's volume node
        is never live when the first channel's re-add flushes GLIR."""
        prog = self._prog_module()
        self._meter_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=True, n_ch=2)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")
        assert ctl._progressive_active is True

        ops = []
        real_hide, real_remove = (ctl._hide_layer_by_name,
                                  ctl._remove_layer_by_name)

        def _log_hide(n):
            ops.append(("hide", n))
            real_hide(n)

        def _log_remove(n):
            ops.append(("remove", n))
            real_remove(n)

        ctl._hide_layer_by_name = _log_hide
        ctl._remove_layer_by_name = _log_remove

        viewer.dims.ndisplay = 2
        ctl._on_ndisplay_changed()

        assert [k for k, _ in ops] == ["hide", "hide", "remove", "remove"]
        assert ctl._progressive_active is False

    def test_swap_to_progressive_hides_all_channels_before_removing_any(
            self, monkeypatch):
        """Multi-channel 2D->3D build: same hide-all-first invariant so a second
        channel's native coarsest volume node is never live when the first
        channel's progressive add flushes GLIR."""
        prog = self._prog_module()
        self._meter_seam(monkeypatch, prog, [])
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=True, n_ch=2)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=2)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")  # native 2D
        assert ctl._progressive_active is False

        ops = []
        real_hide, real_remove = (ctl._hide_layer_by_name,
                                  ctl._remove_layer_by_name)

        def _log_hide(n):
            ops.append(("hide", n))
            real_hide(n)

        def _log_remove(n):
            ops.append(("remove", n))
            real_remove(n)

        ctl._hide_layer_by_name = _log_hide
        ctl._remove_layer_by_name = _log_remove

        viewer.dims.ndisplay = 3
        ctl._on_ndisplay_changed()   # deferred build runs inline (immediate timer)

        assert [k for k, _ in ops] == ["hide", "hide", "remove", "remove"]
        assert ctl._progressive_active is True

    def test_reshow_in_3d_reengages_progressive(self, monkeypatch):
        """Returning to Step 1 while ALREADY in 3D (no ndisplay event to fire
        the hook) must re-engage progressive, not come back native-coarse
        (lifecycle-audit MED-1)."""
        prog = self._prog_module()
        calls = []
        self._meter_seam(monkeypatch, prog, calls)
        timer = _ImmediateQTimer()
        monkeypatch.setattr(_ctl_mod, "QTimer", timer)
        levels, meta = _levels_meta(has_channel=False)
        self._patch_open(monkeypatch, levels, meta)

        viewer = _ProgFakeViewer(ndisplay=3)
        ctl = _make_prog_ctl(viewer, _ProgFakeWidget(checked=True))
        ctl.unload_source = lambda: None
        ctl._apply_multiscale_store("s.zarr", "vol", src_path="vol.tif")
        assert ctl._progressive_active is True

        # Step-exit (layers off the shared canvas), then re-enter while STILL 3D.
        ctl._remove_source_layers_from_viewer()
        calls.clear()
        ctl._reshow_source_layers()

        assert ctl._progressive_active is True   # re-engaged, not native-coarse
        assert len(calls) == 1                    # progressive rebuilt on reshow


# ---------------------------------------------------------------------------
# F3 — _restore_bounding_boxes seeds the box tool, NOT the legacy Shapes layer
# ---------------------------------------------------------------------------

class _RestoreFakeLazy:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)


class _RestoreText:
    def __init__(self, val):
        self._val = val

    def text(self):
        return self._val


class _RestoreWidget:
    """The slice of the widget `_restore_bounding_boxes` + the box tool read:
    the workspace path, the display-name field, and `draw_box_list` (None so
    the box tool's list refresh no-ops headlessly)."""

    draw_box_list = None

    def __init__(self, workspace, display_name):
        self.output_file_path = _RestoreText(workspace)
        self.display_name_edit = _RestoreText(display_name)


class _RestoreViewer:
    """Records `add_shapes` so the test can assert restore NEVER creates a
    napari Shapes layer (the freed-VBO crash surface F3 retires)."""

    def __init__(self):
        self.layers = []
        self.add_shapes_calls = 0

    def add_shapes(self, *a, **k):
        self.add_shapes_calls += 1
        raise AssertionError("restore must not create a Shapes layer")


class _RestoreCtl:
    _BIND = ("_restore_bounding_boxes", "_prefix_for_file", "_ensure_shapes_layer")

    def __init__(self, workspace, display_name, source_path):
        import types

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C)
        from napari_epicycle.data_preparation._box_tool import Step1BoxTool

        self.viewer = _RestoreViewer()
        self.widget = _RestoreWidget(workspace, display_name)
        self._loaded_source_file = source_path
        self._source_lazy = _RestoreFakeLazy((2, 30, 400, 600))  # (C, Z, Y, X)
        self._source_axes_resolved = "CZYX"
        self._loaded_source_layers = []
        self._committed_box_sigs = set()
        self.box_tool = Step1BoxTool(self)
        for name in self._BIND:
            setattr(self, name, types.MethodType(getattr(C, name), self))
        # `_boxrec_from_sidecar` is a @staticmethod — bind the plain function.
        self._boxrec_from_sidecar = C._boxrec_from_sidecar

    def _log(self, *_a, **_k):
        pass


def _rect(y0, y1, x0, x1):
    return [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]


class TestRestoreBoundingBoxesSeedsBoxTool:
    """F3: restore routes exported boxes into the crash-safe box tool and
    retires the legacy Shapes layer."""

    def _write_sidecars(self, workspace, prefix, entries):
        for idx, (rect, zr) in entries.items():
            meta = {"bounding_box": rect}
            if zr is not None:
                meta["selected_z_range"] = zr
            with open(os.path.join(workspace, f"{prefix}_{idx}_metadata.json"),
                      "w") as f:
                json.dump(meta, f)

    def test_restore_seeds_box_tool_and_no_shapes_layer(self):
        from napari_epicycle.data_preparation.logic import box_signature

        with tempfile.TemporaryDirectory() as ws:
            # Two exported patches, with distinct XY windows + trimmed Z slabs.
            self._write_sidecars(ws, "kidney_patch", {
                1: (_rect(100, 180, 200, 290), [10, 18]),
                2: (_rect(50, 90, 300, 360), [4, 12]),
            })
            ctl = _RestoreCtl(ws, "kidney", os.path.join(ws, "kidney.tif"))

            n = ctl._restore_bounding_boxes("kidney")
            assert n == 2

            # -- box tool seeded with COMMITTED (locked + checked) rows --------
            bt = ctl.box_tool
            assert len(bt.boxes) == 2
            assert bt.box_locked == [True, True]
            assert bt.box_checked == [True, True]
            assert bt._full_shape == (30, 400, 600)

            # BoxRec round-trip: center/size/Z reconstruct the exported bounds.
            b0 = bt.boxes[0]
            assert (b0.cy, b0.cx, b0.sy, b0.sx) == (140, 245, 80, 90)
            assert (b0.cz, b0.sz) == (14, 8)   # trimmed export Z from z_range
            bd = b0.bounds_dict((30, 400, 600))
            assert (bd["y0"], bd["y1"], bd["x0"], bd["x1"]) == (100, 180, 200, 290)
            assert (bd["z0"], bd["z1"]) == (10, 18)

            # -- committed signatures preserved (both XY-only + Z-inclusive) ---
            sig_xy = box_signature(np.array(_rect(100, 180, 200, 290), float))
            sig_z = box_signature(np.array(_rect(100, 180, 200, 290), float),
                                  [10, 18])
            assert sig_xy in ctl._committed_box_sigs
            assert sig_z in ctl._committed_box_sigs

            # A restored box's own rectangle/z-range signature matches too, so a
            # re-draw of the same patch is deduped.
            assert box_signature(bt.rectangles()[0]) in ctl._committed_box_sigs
            assert (box_signature(bt.rectangles()[0], bt.z_ranges()[0])
                    in ctl._committed_box_sigs)

            # -- the legacy Shapes layer is fully absent from restore ----------
            assert ctl.viewer.add_shapes_calls == 0
            assert ctl.viewer.layers == []

    def test_restore_is_idempotent_across_reloads(self):
        with tempfile.TemporaryDirectory() as ws:
            self._write_sidecars(ws, "kidney_patch", {
                1: (_rect(100, 180, 200, 290), [10, 18]),
            })
            ctl = _RestoreCtl(ws, "kidney", os.path.join(ws, "kidney.tif"))
            ctl._restore_bounding_boxes("kidney")
            ctl._restore_bounding_boxes("kidney")   # reload same source
            assert len(ctl.box_tool.boxes) == 1
            assert ctl.viewer.add_shapes_calls == 0

    def test_legacy_sidecar_without_z_range_falls_back_to_full_z(self):
        with tempfile.TemporaryDirectory() as ws:
            self._write_sidecars(ws, "kidney_patch", {
                1: (_rect(100, 180, 200, 290), None),   # no selected_z_range
            })
            ctl = _RestoreCtl(ws, "kidney", os.path.join(ws, "kidney.tif"))
            assert ctl._restore_bounding_boxes("kidney") == 1
            box = ctl.box_tool.boxes[0]
            assert box.sz == 30 and box.cz == 15       # full-Z default (Z=30)


# =========================================================================== #
# Corrupt-state hook: behaviour + THREAD SAFETY (Brick 1c #1)
# =========================================================================== #

class TestCorruptStateHookWiring:
    """The module-level hook is registered by the controller and reports a
    corrupt `step1_state.json` from any bare `load_state`.  Behavioural, not
    source-inspection: an `inspect.getsource(...)` assertion passes on the
    import line alone."""

    def _widget(self):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        return DataPreparationWidget(viewer=_RealImageViewer())

    def _corrupt(self, ws):
        from napari_epicycle.data_preparation._paths import step1_state_path
        step1_state_path(ws).write_text("{ truncated", encoding="utf-8")

    def test_a_bare_load_paints_the_red_inline_status(self, qt_app, tmp_path):
        from napari_epicycle.data_preparation._state import (
            load_state, set_corrupt_state_hook)

        ws = str(tmp_path / "ws"); os.makedirs(ws)
        w = self._widget()
        try:
            self._corrupt(ws)
            load_state(ws)                    # NO on_corrupt= — the hook path
            lbl = w.workspace_status_label
            assert "step1_state.json" in lbl.text()
            assert "corrupt-" in lbl.text()
            assert "#e07070" in lbl.styleSheet()
        finally:
            set_corrupt_state_hook(None)
            w.setParent(None)

    def test_a_worker_thread_never_touches_the_widget_directly(
            self, qt_app, tmp_path):
        """`_export.persist_patch` calls `load_state` and runs on the EXTRACT
        QThread, so a corrupt file mid-extract fires this hook OFF the GUI
        thread.  `_report_state_problem` sets QLabel text/stylesheet and `_log`
        appends to a QTextEdit — main-thread-only.  Touching them from a worker
        is an unguarded crash surface, not a cosmetic issue."""
        import threading

        from napari_epicycle.data_preparation._state import (
            load_state, set_corrupt_state_hook)

        ws = str(tmp_path / "ws2"); os.makedirs(ws)
        w = self._widget()
        try:
            touched_on = []
            real = w.workspace_status_label.setText

            def spy(text, _real=real):
                touched_on.append(threading.current_thread().name)
                _real(text)

            w.workspace_status_label.setText = spy
            self._corrupt(ws)

            worker = threading.Thread(target=lambda: load_state(ws),
                                      name="extract-worker")
            worker.start()
            worker.join(5)

            assert "extract-worker" not in touched_on, (
                "the corrupt-state hook touched a QWidget from a worker thread")
            # ... and once the event loop spins, the report DOES land.
            for _ in range(5):
                qt_app.processEvents()
            assert "corrupt-" in w.workspace_status_label.text()
            assert touched_on and "extract-worker" not in touched_on
        finally:
            set_corrupt_state_hook(None)
            w.setParent(None)

    def test_the_hook_reports_synchronously_on_the_main_thread(
            self, qt_app, tmp_path):
        """No marshalling detour when we are already on the GUI thread."""
        from napari_epicycle.data_preparation._state import (
            load_state, set_corrupt_state_hook)

        ws = str(tmp_path / "ws3"); os.makedirs(ws)
        w = self._widget()
        try:
            self._corrupt(ws)
            load_state(ws)
            assert "corrupt-" in w.workspace_status_label.text()  # no spin
        finally:
            set_corrupt_state_hook(None)
            w.setParent(None)
