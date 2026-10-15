"""Tests for FM8 — proxy-aware channel keybinding (`a/s/d/f`).

Background
----------
The 2026-05-16 visibility/isolation trace identified FM8 as a class of
hide/unhide bugs caused by the channel-aware visibility keybinding
(``_setup_channel_keybindings`` in ``native_multiview_widget.py``)
looking up layers by name only.  When isolation is active, the main
labels layer is intentionally hidden — pressing the key toggled it
back on, unhiding ALL labels and fighting the active isolation.

The fix adds a virtual hook ``_get_toggle_targets_for_name(name)`` on
the base ``NativeMultiviewAnnotator``.  Default impl preserves the
pre-FM8 behavior (main labels + subviewer label layers).
``Curation3DAnnotator`` overrides it with priority order:

  1. ``_refine_layer`` non-None         → SPLIT/REFINE mode
  2. ``_multi_label_proxy_layers`` set  → MULTI-ISO mode
  3. ``_isolation_proxy_layer`` + 3D    → SINGLE-ISO-3D mode
  4. fallback to super()                → DEFAULT (no iso, or 2D)

Tests are Qt-free.  Bare-instance pattern + lightweight fake layer
stand-ins.  Source-level checks confirm the wiring stays in place.
"""
from __future__ import annotations

import inspect
import types


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeLabelsLayer:
    """Minimal Labels stand-in with `.name` and `.visible`."""

    def __init__(self, name="layer", visible=True, mode="paint", editable=True):
        self.name = name
        self.visible = visible
        self.mode = mode
        self.editable = editable


class _FakeViewerModel:
    """ViewerModel stub with a `.layers` list and `.dims.ndisplay`."""

    def __init__(self, layers=None, ndisplay=2):
        self.layers = list(layers or [])
        self.dims = types.SimpleNamespace(ndisplay=ndisplay)


def _layer_by_name_stub(layer_list, name):
    """Match the real `_layer_by_name` signature: search by name."""
    for lyr in layer_list:
        if getattr(lyr, "name", None) == name:
            return lyr
    return None


def _orient_for_vm_stub(self, vm):
    """Map a fake viewer model to a stub orientation."""
    return getattr(vm, "_orient", "XY")


# ---------------------------------------------------------------------------
# Base hook default behavior
# ---------------------------------------------------------------------------


class TestBaseHookDefault:
    """`NativeMultiviewAnnotator._get_toggle_targets_for_name` default
    impl returns main labels + per-orientation subviewer labels.
    This preserves the pre-FM8 behavior for any caller that doesn't
    override the hook (HITL 2D annotator, future subclasses, etc.)."""

    def _make_bare_base(self, main_layers, sub_xy, sub_xz, sub_yz):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann = NativeMultiviewAnnotator.__new__(NativeMultiviewAnnotator)
        ann.viewer = types.SimpleNamespace(layers=main_layers)
        ann.viewer_model_xy = _FakeViewerModel(sub_xy)
        ann.viewer_model_xy._orient = "XY"
        ann.viewer_model_xz = _FakeViewerModel(sub_xz)
        ann.viewer_model_xz._orient = "XZ"
        ann.viewer_model_yz = _FakeViewerModel(sub_yz)
        ann.viewer_model_yz._orient = "YZ"
        ann._layer_by_name = lambda layers, name: _layer_by_name_stub(layers, name)
        ann._orientation_for_vm = lambda vm: _orient_for_vm_stub(ann, vm)
        return ann

    def test_default_returns_main_plus_three_subviewers(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        main = _FakeLabelsLayer("labels_3d")
        sub_xy = _FakeLabelsLayer("labels_3d_XY")
        sub_xz = _FakeLabelsLayer("labels_3d_XZ")
        sub_yz = _FakeLabelsLayer("labels_3d_YZ")
        ann = self._make_bare_base([main], [sub_xy], [sub_xz], [sub_yz])
        targets = NativeMultiviewAnnotator._get_toggle_targets_for_name(
            ann, "labels_3d"
        )
        assert targets == [main, sub_xy, sub_xz, sub_yz]

    def test_default_falls_back_to_name_when_no_orient_suffix(self):
        """Some subviewer setups don't suffix labels by orient — they
        keep the same name across viewers.  The default impl falls
        back to the unsuffixed lookup."""
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        main = _FakeLabelsLayer("labels")
        sub_xy = _FakeLabelsLayer("labels")  # same name, no suffix
        ann = self._make_bare_base([main], [sub_xy], [], [])
        targets = NativeMultiviewAnnotator._get_toggle_targets_for_name(
            ann, "labels"
        )
        assert main in targets
        assert sub_xy in targets

    def test_default_skips_missing_layers(self):
        """If the named layer doesn't exist in a viewer, the default
        impl simply omits it from the targets list (no crash)."""
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann = self._make_bare_base([], [], [], [])
        targets = NativeMultiviewAnnotator._get_toggle_targets_for_name(
            ann, "nonexistent"
        )
        assert targets == []


# ---------------------------------------------------------------------------
# Curation3D override — priority routing
# ---------------------------------------------------------------------------


class TestCuration3DOverride:

    def _make_bare_c3d(
        self,
        *,
        refine_layer=None,
        multi_proxies=None,
        single_proxy=None,
        ndisplay=3,
        main_layers=None,
        sub_xy_layers=None,
        sub_xz_layers=None,
        sub_yz_layers=None,
        subviewer_manager=None,
        labels_names=None,
    ):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
        ann._refine_layer = refine_layer
        ann._multi_label_proxy_layers = multi_proxies or {}
        ann._isolation_proxy_layer = single_proxy
        ann._isolation_proxy_label_id = (
            single_proxy.name if single_proxy is not None else None
        )
        # The image-channel gate consults `_active_labels_names` to
        # decide whether to apply isolation routing.  Default to
        # ["labels_3d"] (matches the test fakes); multi-channel
        # tests override this.
        ann._active_labels_names = (
            list(labels_names) if labels_names is not None else ["labels_3d"]
        )
        # Viewer with ndisplay (single-iso-3D path checks ndisplay == 3).
        ann.viewer = types.SimpleNamespace(
            layers=main_layers or [],
            dims=types.SimpleNamespace(ndisplay=ndisplay),
        )
        ann.viewer_model_xy = _FakeViewerModel(sub_xy_layers or [])
        ann.viewer_model_xy._orient = "XY"
        ann.viewer_model_xz = _FakeViewerModel(sub_xz_layers or [])
        ann.viewer_model_xz._orient = "XZ"
        ann.viewer_model_yz = _FakeViewerModel(sub_yz_layers or [])
        ann.viewer_model_yz._orient = "YZ"
        ann._layer_by_name = lambda layers, name: _layer_by_name_stub(layers, name)
        ann._orientation_for_vm = lambda vm: _orient_for_vm_stub(ann, vm)
        ann.subviewer_manager = subviewer_manager
        return ann

    def test_priority_1_refine_layer_active_routes_to_refine(self):
        """When `_refine_layer` is non-None (split active), the toggle
        targets the Refine layer in main + Refine_<orient> in
        subviewers, NOT the main labels layer."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        refine = _FakeLabelsLayer("Refine")
        refine_xy = _FakeLabelsLayer("Refine_XY")
        refine_xz = _FakeLabelsLayer("Refine_XZ")
        refine_yz = _FakeLabelsLayer("Refine_YZ")
        main = _FakeLabelsLayer("labels_3d")  # present, but must be ignored
        # Fake subviewer_manager.get_subviewer_layer.
        sub_mgr = types.SimpleNamespace(
            get_subviewer_layer=lambda nm, orient: {
                ("Refine", "XY"): refine_xy,
                ("Refine", "XZ"): refine_xz,
                ("Refine", "YZ"): refine_yz,
            }.get((nm, orient))
        )
        ann = self._make_bare_c3d(
            refine_layer=refine,
            main_layers=[main],
            subviewer_manager=sub_mgr,
        )
        targets = Curation3DAnnotator._get_toggle_targets_for_name(ann, "labels_3d")
        assert refine in targets
        assert refine_xy in targets
        assert refine_xz in targets
        assert refine_yz in targets
        # Main labels layer must NOT be in targets — that's the FM8 bug.
        assert main not in targets

    def test_priority_2_multi_proxies_active(self):
        """When `_multi_label_proxy_layers` has entries (lasso / merge
        dialog), all N sub-proxies are toggled together + the subviewer
        label layers."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        proxy_a = _FakeLabelsLayer("_epicycle_isolation_proxy_5")
        proxy_b = _FakeLabelsLayer("_epicycle_isolation_proxy_12")
        proxy_c = _FakeLabelsLayer("_epicycle_isolation_proxy_47")
        sub_xy = _FakeLabelsLayer("labels_3d_XY")
        sub_xz = _FakeLabelsLayer("labels_3d_XZ")
        sub_yz = _FakeLabelsLayer("labels_3d_YZ")
        main = _FakeLabelsLayer("labels_3d")
        ann = self._make_bare_c3d(
            multi_proxies={5: proxy_a, 12: proxy_b, 47: proxy_c},
            main_layers=[main],
            sub_xy_layers=[sub_xy],
            sub_xz_layers=[sub_xz],
            sub_yz_layers=[sub_yz],
        )
        targets = Curation3DAnnotator._get_toggle_targets_for_name(ann, "labels_3d")
        for p in (proxy_a, proxy_b, proxy_c):
            assert p in targets
        for s in (sub_xy, sub_xz, sub_yz):
            assert s in targets
        assert main not in targets

    def test_priority_3_single_proxy_3d_active(self):
        """When `_isolation_proxy_layer` is non-None AND `ndisplay==3`,
        the toggle targets the single proxy + subviewer label layers.
        Main labels layer is intentionally excluded."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        proxy = _FakeLabelsLayer("_epicycle_isolation_proxy")
        sub_xy = _FakeLabelsLayer("labels_3d_XY")
        sub_xz = _FakeLabelsLayer("labels_3d_XZ")
        sub_yz = _FakeLabelsLayer("labels_3d_YZ")
        main = _FakeLabelsLayer("labels_3d")
        ann = self._make_bare_c3d(
            single_proxy=proxy,
            ndisplay=3,
            main_layers=[main],
            sub_xy_layers=[sub_xy],
            sub_xz_layers=[sub_xz],
            sub_yz_layers=[sub_yz],
        )
        targets = Curation3DAnnotator._get_toggle_targets_for_name(ann, "labels_3d")
        assert proxy in targets
        for s in (sub_xy, sub_xz, sub_yz):
            assert s in targets
        assert main not in targets

    def test_priority_3_single_proxy_2d_falls_through_to_default(self):
        """Single-label isolation in 2D (ndisplay=2) uses
        `show_selected_label` on the main labels layer directly, no
        proxy.  Override must fall through to default behavior."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        proxy = _FakeLabelsLayer("_epicycle_isolation_proxy")
        sub_xy = _FakeLabelsLayer("labels_3d_XY")
        sub_xz = _FakeLabelsLayer("labels_3d_XZ")
        sub_yz = _FakeLabelsLayer("labels_3d_YZ")
        main = _FakeLabelsLayer("labels_3d")
        ann = self._make_bare_c3d(
            single_proxy=proxy,  # proxy exists but
            ndisplay=2,           # we're in 2D — fall through
            main_layers=[main],
            sub_xy_layers=[sub_xy],
            sub_xz_layers=[sub_xz],
            sub_yz_layers=[sub_yz],
        )
        targets = Curation3DAnnotator._get_toggle_targets_for_name(ann, "labels_3d")
        # Default path: main + subviewer labels (proxy NOT included).
        assert main in targets
        for s in (sub_xy, sub_xz, sub_yz):
            assert s in targets
        assert proxy not in targets

    def test_priority_4_no_isolation_falls_through_to_default(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        sub_xy = _FakeLabelsLayer("labels_3d_XY")
        main = _FakeLabelsLayer("labels_3d")
        ann = self._make_bare_c3d(
            main_layers=[main],
            sub_xy_layers=[sub_xy],
        )
        targets = Curation3DAnnotator._get_toggle_targets_for_name(ann, "labels_3d")
        assert main in targets
        assert sub_xy in targets

    def test_priority_order_refine_beats_multi_beats_single(self):
        """If multiple isolation modes are active simultaneously (a
        defensive edge case — shouldn't happen in practice), refine
        wins over multi-iso, which wins over single-iso."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        refine = _FakeLabelsLayer("Refine")
        multi_proxy = _FakeLabelsLayer("_epicycle_isolation_proxy_5")
        single_proxy = _FakeLabelsLayer("_epicycle_isolation_proxy")
        sub_mgr = types.SimpleNamespace(
            get_subviewer_layer=lambda nm, orient: None
        )
        ann = self._make_bare_c3d(
            refine_layer=refine,
            multi_proxies={5: multi_proxy},
            single_proxy=single_proxy,
            ndisplay=3,
            subviewer_manager=sub_mgr,
            # Must use a labels-name in `_active_labels_names` for the
            # gate to pass through to the priority routing.
            labels_names=["labels_3d"],
        )
        targets = Curation3DAnnotator._get_toggle_targets_for_name(
            ann, "labels_3d"
        )
        # Refine wins — single + multi proxies must NOT be in targets.
        assert refine in targets
        assert multi_proxy not in targets
        assert single_proxy not in targets


# ---------------------------------------------------------------------------
# Multi-channel correctness — image-key presses MUST NOT route to isolation
# ---------------------------------------------------------------------------


class TestMultiChannelImageKeyGate:
    """The channel keybinding is adaptive: 1 image channel + 1 labels →
    `a` = image, `s` = labels.  2 image channels + 1 labels → `a` = img1,
    `s` = img2, `d` = labels.

    The Curation3D override MUST only apply isolation-aware routing
    when the toggled layer is one of the labels layers.  Pressing an
    image-channel key during split/merge/iso MUST keep the default
    image-toggle behavior (otherwise pressing `a` during split would
    return the Refine layer instead of image ch1 — a real regression).
    """

    def _make(self, **kw):
        """Use TestCuration3DOverride's fixture builder."""
        return TestCuration3DOverride()._make_bare_c3d(**kw)

    def test_image_channel_key_with_split_active_falls_through(self):
        """1-channel data: 'a' = image, 's' = labels.  During split:
        pressing 'a' (image) must NOT return the Refine layer."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        refine = _FakeLabelsLayer("Refine")
        image_layer = _FakeLabelsLayer("image_ch1")
        sub_mgr = types.SimpleNamespace(
            get_subviewer_layer=lambda nm, orient: None
        )
        ann = self._make(
            refine_layer=refine,
            main_layers=[image_layer],
            subviewer_manager=sub_mgr,
            labels_names=["labels_3d"],  # image_ch1 is NOT a labels name
        )
        # Press 'a' (which would map to "image_ch1") during split.
        targets = Curation3DAnnotator._get_toggle_targets_for_name(
            ann, "image_ch1"
        )
        # Must NOT return the Refine layer — image toggle stays default.
        assert refine not in targets
        # Default behavior: returns image_ch1 (in main viewer).
        assert image_layer in targets

    def test_image_channel_key_with_multi_iso_falls_through(self):
        """2-channel data: 'a' = img1, 's' = img2, 'd' = labels.
        Lasso multi-select active: pressing 'a' or 's' must NOT toggle
        the multi-label proxies, only the image channel itself."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        proxy_a = _FakeLabelsLayer("_epicycle_isolation_proxy_5")
        proxy_b = _FakeLabelsLayer("_epicycle_isolation_proxy_12")
        img_ch1 = _FakeLabelsLayer("image_ch1")
        img_ch2 = _FakeLabelsLayer("image_ch2")
        ann = self._make(
            multi_proxies={5: proxy_a, 12: proxy_b},
            main_layers=[img_ch1, img_ch2],
            labels_names=["labels_3d"],
        )
        # Press 'a' (image ch1).
        targets = Curation3DAnnotator._get_toggle_targets_for_name(
            ann, "image_ch1"
        )
        assert proxy_a not in targets
        assert proxy_b not in targets
        assert img_ch1 in targets
        # Press 's' (image ch2 in 2-channel data).
        targets2 = Curation3DAnnotator._get_toggle_targets_for_name(
            ann, "image_ch2"
        )
        assert proxy_a not in targets2
        assert proxy_b not in targets2
        assert img_ch2 in targets2

    def test_labels_key_in_two_channel_data_routes_correctly(self):
        """2-channel data with split active: pressing 'd' (which maps
        to labels_3d) MUST route to Refine layer."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        refine = _FakeLabelsLayer("Refine")
        main_labels = _FakeLabelsLayer("labels_3d")
        sub_mgr = types.SimpleNamespace(
            get_subviewer_layer=lambda nm, orient: None
        )
        ann = self._make(
            refine_layer=refine,
            main_layers=[main_labels],
            subviewer_manager=sub_mgr,
            labels_names=["labels_3d"],
        )
        # Press 'd' (labels in 2-channel data).
        targets = Curation3DAnnotator._get_toggle_targets_for_name(
            ann, "labels_3d"
        )
        assert refine in targets
        assert main_labels not in targets

    def test_multiple_labels_names_all_route_through_isolation(self):
        """If multiple labels layers are active (rare but supported),
        each labels name routes through the isolation logic."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        refine = _FakeLabelsLayer("Refine")
        sub_mgr = types.SimpleNamespace(
            get_subviewer_layer=lambda nm, orient: None
        )
        ann = self._make(
            refine_layer=refine,
            subviewer_manager=sub_mgr,
            labels_names=["labels_a", "labels_b"],
        )
        # Either labels name routes through isolation.
        targets_a = Curation3DAnnotator._get_toggle_targets_for_name(
            ann, "labels_a"
        )
        targets_b = Curation3DAnnotator._get_toggle_targets_for_name(
            ann, "labels_b"
        )
        assert refine in targets_a
        assert refine in targets_b

    def test_image_key_with_no_isolation_unchanged(self):
        """Sanity: no-isolation image-key press behaves like baseline."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        img = _FakeLabelsLayer("image_ch1")
        ann = self._make(
            main_layers=[img],
            labels_names=["labels_3d"],
        )
        targets = Curation3DAnnotator._get_toggle_targets_for_name(
            ann, "image_ch1"
        )
        assert img in targets


# ---------------------------------------------------------------------------
# Source-level: wiring + diagnostic presence
# ---------------------------------------------------------------------------


class TestSourceWiring:

    def test_make_toggle_uses_hook(self):
        """The toggle closure in `_setup_channel_keybindings` must
        consult `_get_toggle_targets_for_name`, not inline the lookup
        as it did pre-FM8."""
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        src = inspect.getsource(NativeMultiviewAnnotator._setup_channel_keybindings)
        assert "_get_toggle_targets_for_name" in src, (
            "FM8 fix: `_make_toggle` must call the hook, not inline "
            "the main-layer / subviewer lookup pattern."
        )

    def test_base_hook_exists_with_default_impl(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        assert hasattr(NativeMultiviewAnnotator, "_get_toggle_targets_for_name")

    def test_curation3d_overrides_hook(self):
        """Override must be defined on the subclass (not inherited)."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        # Subclass method object must differ from base.
        assert (
            Curation3DAnnotator._get_toggle_targets_for_name
            is not NativeMultiviewAnnotator._get_toggle_targets_for_name
        ), "Curation3DAnnotator must define its own _get_toggle_targets_for_name override."

    def test_curation3d_override_has_chkey_diagnostics(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._get_toggle_targets_for_name)
        assert "[CHKEY] FM8 routing" in src

    def test_curation3d_override_documents_fm8(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._get_toggle_targets_for_name)
        assert "FM8" in src

    def test_make_toggle_has_chkey_diagnostic(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        src = inspect.getsource(NativeMultiviewAnnotator._setup_channel_keybindings)
        assert "[CHKEY] toggle" in src


# ---------------------------------------------------------------------------
# Toggle group-visibility behavior
# ---------------------------------------------------------------------------


class TestGroupToggleBehavior:
    """The toggle computes `new_vis` from the first target and applies
    it to all, so multi-label proxies hide/show as a group.  This is
    important when N proxies have drifted into different states."""

    def test_new_vis_derived_from_first_target_when_all_consistent(self):
        """All targets visible → new_vis becomes False on the next
        toggle.  All hidden → new_vis becomes True."""
        # This is verified indirectly through the source check below
        # — the toggle code uses `not bool(getattr(targets[0], 'visible', True))`.
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        src = inspect.getsource(NativeMultiviewAnnotator._setup_channel_keybindings)
        assert "not bool(getattr(targets[0], \"visible\", True))" in src

    def test_all_targets_get_same_new_vis(self):
        """Group toggle: every target gets `lyr.visible = new_vis`,
        not toggled individually.  Verified by source pattern."""
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        src = inspect.getsource(NativeMultiviewAnnotator._setup_channel_keybindings)
        # Look for the group-apply loop pattern.
        assert "for lyr in targets:" in src
        assert "lyr.visible = new_vis" in src
