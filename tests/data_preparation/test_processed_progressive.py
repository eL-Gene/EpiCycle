"""Item 4 — route the ⟨processed⟩ STORE display through the SAME progressive
loading engine the RAW prepared source uses, so the processed image refines in
3D (was a plain ``add_image(multiscale=True)`` — native/coarse-in-3D).

The processed progressive machinery mirrors the raw trio
(``_maybe_swap_progressive_on_ndisplay`` / ``_swap_*`` / ``_reveal_*``) but
operates on the DISPLAY-ONLY processed store layers and must NEVER touch the raw
``_source_*`` state (B4 discipline).  Safe under the clean raw⇄processed swap:
only ONE set is ever on the canvas, so there is never more than one progressive
layer set live.

Covered gates:
  * processed display routes to the progressive path when available (3D + wanted)
    — the ndisplay swap dispatches to the PROCESSED layers, engages the seam with
    the ⟨processed⟩-marker names, and flips ``_processed_progressive_active``;
  * falls back to native when the seam returns None (unavailable) — per channel,
    ``_processed_progressive_active`` stays False;
  * 2D stays native (no seam, no swap);
  * the ndisplay swap targets the PROCESSED store while processed is the active
    view, and does NOT engage the raw path;
  * teardown / clear DETACHES the processed progressive layers (hide-then-remove
    fires the seam's ``removed`` GLIR-meter teardown) + resets the active flag;
  * the raw ``_source_*`` state is untouched by the processed progressive display;
  * ``_show_processed_view`` invokes the swap once processed is the active view.

Qt-free — plain-object fakes for the viewer / layers / widget, and the seam's
``add_progressive_image`` / ``progressive_available`` monkeypatched (mirroring
``test_controller.py``'s ``TestProgressive3DDisplay`` convention).
"""
from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.data_preparation import controller as _ctl_mod
from packages.epicycle_largedata import progressive as prog

DPC = _ctl_mod.DataPreparationController
_PROC = _ctl_mod.PROCESSED_LAYER_MARKER


# --------------------------------------------------------------------------- #
# Qt-free fakes (layer list fires inserted/removed so the seam teardown runs)
# --------------------------------------------------------------------------- #
class _Signal:
    def __init__(self):
        self._cbs = []

    def connect(self, cb):
        self._cbs.append(cb)

    def disconnect(self, cb=None):
        if cb is None:
            self._cbs.clear()
        else:
            self._cbs = [c for c in self._cbs if c is not cb]

    def emit(self, ev):
        for cb in list(self._cbs):
            cb(ev)


class _Event:
    def __init__(self, value):
        self.value = value


class _Events:
    def __init__(self):
        self.removed = _Signal()
        self.inserted = _Signal()


class _FakeLayer:
    def __init__(self, name, multiscale=None):
        self.name = name
        self.visible = True
        self.gamma = 1.0
        self.colormap = "green"
        self.blending = "additive"
        self.contrast_limits = None
        self.metadata = {}
        self.locked_data_level = None
        self._multiscale = multiscale
        self._zoom_depth = None
        self.apply_zoom_depth_calls = 0

    @property
    def multiscale(self):
        return bool(self._multiscale)

    def apply_zoom_depth_now(self):
        self.apply_zoom_depth_calls += 1


class _FakeLayerList(list):
    def __init__(self):
        super().__init__()
        self.events = _Events()

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
        self.events.inserted.emit(_Event(layer))

    def remove(self, layer):
        list.remove(self, layer)
        self.events.removed.emit(_Event(layer))


class _FakeDims:
    def __init__(self, ndisplay=2):
        self.ndisplay = ndisplay


class _FakeViewer:
    def __init__(self, ndisplay=2):
        self.layers = _FakeLayerList()
        self.dims = _FakeDims(ndisplay)
        # window teardown wiring reaches viewer.window._qt_window; None makes it
        # a guarded no-op (headless).
        self.window = None
        self._layer_slicer = type("S", (), {"_force_sync": True})()

    def add_image(self, data, name=None, colormap=None, blending=None,
                  multiscale=False, contrast_limits=None, gamma=None,
                  visible=None):
        layer = _FakeLayer(name, multiscale=multiscale)
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


class _CheckBox:
    def __init__(self, checked=True):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _Widget:
    def __init__(self, checked=True):
        self.progressive_3d_checkbox = _CheckBox(checked)
        self.configure_zoom_depth_calls = []
        self.hide_zoom_depth_calls = 0

    def configure_zoom_depth(self, mode):
        self.configure_zoom_depth_calls.append(mode)

    def hide_zoom_depth(self):
        self.hide_zoom_depth_calls += 1


class _ImmediateQTimer:
    """singleShot runs the callback IMMEDIATELY so deferred swap + reveal run
    inline in a headless test."""

    def __init__(self):
        self.scheduled = 0

    def singleShot(self, ms, fn):
        self.scheduled += 1
        fn()


class _DeferredQTimer:
    def __init__(self):
        self.pending = []

    def singleShot(self, ms, fn):
        self.pending.append(fn)

    def flush(self):
        fns, self.pending = self.pending, []
        for fn in fns:
            fn()


def _levels():
    return [np.zeros((8, 8, 8), dtype=np.uint16),
            np.zeros((4, 4, 4), dtype=np.uint16)]


# --------------------------------------------------------------------------- #
# Controller factory — the PROCESSED store is the active view (raw off canvas)
# --------------------------------------------------------------------------- #
def _make_ctl(viewer, widget, *, n_ch=2, active=False):
    ctl = DPC.__new__(DPC)
    ctl.viewer = viewer
    ctl.widget = widget
    ctl._log = lambda *a, **k: None
    ctl._on_layers_changed = lambda *a, **k: None
    ctl._swapping_source_display = False

    # --- raw state (must stay pristine — SENTINELS the tests assert on) ------
    ctl._loaded_source_layers = ["raw_ch0", "raw_ch1"]
    ctl._source_display_channels = [("raw_ch0", [None]), ("raw_ch1", [None])]
    ctl._progressive_active = False
    ctl._current_multiscale_store = None          # keeps raw gate + zoom-refresh inert
    ctl._current_multiscale_display_name = "vol"
    ctl._current_multiscale_src = "vol.tif"
    ctl._source_lazy = "RAW-LAZY-SENTINEL"
    ctl._source_axes_resolved = "ZYX"
    ctl._channel_index_by_name = {"raw_ch0": 0, "raw_ch1": 1}
    ctl._loaded_source_file = "vol.tif"
    ctl._source_layers_on_canvas = False          # raw OFF canvas -> processed active
    ctl._locked_3d_level = None
    ctl._zoom_depth_wanted = None

    # --- processed store state (native layers on the canvas) -----------------
    marker_names = [f"{_PROC}ch{i}" for i in range(n_ch)]
    proc_layers = [_FakeLayer(nm, multiscale=True) for nm in marker_names]
    for l in proc_layers:
        viewer.layers.append(l)
    ctl._processed_store_layers = proc_layers
    ctl._processed_layers = []
    ctl._processed_display_channels = [(nm, _levels()) for nm in marker_names]
    ctl._current_processed_multiscale_store = "proc.zarr"
    ctl._processed_progressive_active = active
    ctl._processed_available = True
    ctl._processed_redisplay = {"kind": "store", "names": [f"ch{i}" for i in range(n_ch)]}
    return ctl


def _spy_seam(monkeypatch, calls, *, returns_layer=True):
    monkeypatch.setattr(prog, "progressive_available", lambda: True)

    def _spy(viewer, levels, name=None, colormap=None, blending=None,
             contrast_limits=None, scale=None, **_kw):
        calls.append({"name": name, "colormap": colormap,
                      "blending": blending, "contrast_limits": contrast_limits,
                      "levels": list(levels)})
        if not returns_layer:
            return None
        layer = _FakeLayer(name)
        layer.contrast_limits = contrast_limits
        viewer.layers.append(layer)
        return layer

    monkeypatch.setattr(prog, "add_progressive_image", _spy)
    return calls


# --------------------------------------------------------------------------- #
# active-view predicate
# --------------------------------------------------------------------------- #
def test_processed_store_is_active_view_true_when_raw_off_canvas():
    ctl = _make_ctl(_FakeViewer(ndisplay=3), _Widget())
    assert ctl._processed_store_is_active_view() is True


def test_processed_not_active_when_raw_on_canvas():
    ctl = _make_ctl(_FakeViewer(ndisplay=3), _Widget())
    ctl._source_layers_on_canvas = True            # raw is the active view
    assert ctl._processed_store_is_active_view() is False


def test_processed_not_active_for_inmemory_regime():
    ctl = _make_ctl(_FakeViewer(ndisplay=3), _Widget())
    ctl._current_processed_multiscale_store = None   # single-scale / in-memory
    assert ctl._processed_store_is_active_view() is False


# --------------------------------------------------------------------------- #
# zoom-depth combo visibility for the PROCESSED view (the raw-only gap fix)
# --------------------------------------------------------------------------- #
def test_zoom_depth_combo_shown_for_processed_progressive_3d():
    """The 'Zoom depth' combo is surfaced when the processed store is the active
    3D view with progressive on — was raw-only, so the combo never appeared for
    processed and zoom-depth could not be enabled."""
    w = _Widget(checked=True)
    ctl = _make_ctl(_FakeViewer(ndisplay=3), w, active=True)
    ctl._zoom_depth_wanted = "50%"
    ctl._refresh_zoom_depth_for_ndisplay()
    assert w.configure_zoom_depth_calls == ["50%"]   # shown, synced to the mode
    assert w.hide_zoom_depth_calls == 0


def test_zoom_depth_combo_hidden_for_processed_in_2d():
    """In 2D the combo hides (native multiscale has no per-draw tile)."""
    w = _Widget(checked=True)
    ctl = _make_ctl(_FakeViewer(ndisplay=2), w, active=True)
    ctl._refresh_zoom_depth_for_ndisplay()
    assert w.configure_zoom_depth_calls == []
    assert w.hide_zoom_depth_calls == 1


# --------------------------------------------------------------------------- #
# routes to progressive in 3D + wanted (deferred), targets the PROCESSED layers
# --------------------------------------------------------------------------- #
def test_processed_display_routes_to_progressive_in_3d(monkeypatch):
    calls = []
    _spy_seam(monkeypatch, calls)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))

    ctl._maybe_swap_progressive_on_ndisplay()      # dispatches to processed

    assert timer.scheduled == 2                     # swap tick + deferred reveal
    assert ctl._processed_progressive_active is True
    # the seam was engaged with the ⟨processed⟩-marker names + the levels
    assert [c["name"] for c in calls] == [f"{_PROC}ch0", f"{_PROC}ch1"]
    assert all(len(c["levels"]) == 2 for c in calls)
    # the processed store layers are now the progressive layers (marker names)
    names = [getattr(l, "name", None) for l in ctl._processed_store_layers]
    assert names == [f"{_PROC}ch0", f"{_PROC}ch1"]


def test_processed_swap_carries_contrast(monkeypatch):
    calls = []
    _spy_seam(monkeypatch, calls)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))
    # napari computed contrast on the native processed slice — the swap carries it.
    for l in ctl._processed_store_layers:
        l.contrast_limits = [7, 250]

    ctl._maybe_swap_progressive_on_ndisplay()

    assert all(c["contrast_limits"] == [7, 250] for c in calls)


def test_entering_3d_defers_processed_swap(monkeypatch):
    """The processed native->progressive swap must be DEFERRED (QTimer) — right
    after the ndisplay change the display is STILL native; only after the tick
    fires is it progressive."""
    calls = []
    _spy_seam(monkeypatch, calls)
    timer = _DeferredQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))

    ctl._maybe_swap_progressive_on_ndisplay()
    assert len(timer.pending) == 1                  # deferred, not yet run
    assert calls == []
    assert ctl._processed_progressive_active is False

    timer.flush()                                   # swap runs (+ schedules reveal)
    assert len(calls) == 2
    assert ctl._processed_progressive_active is True


# --------------------------------------------------------------------------- #
# 2D stays native — no seam, no swap
# --------------------------------------------------------------------------- #
def test_processed_native_in_2d(monkeypatch):
    calls = []
    _spy_seam(monkeypatch, calls)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=2)
    ctl = _make_ctl(viewer, _Widget(checked=True))

    ctl._maybe_swap_progressive_on_ndisplay()

    assert calls == []
    assert timer.scheduled == 0
    assert ctl._processed_progressive_active is False


# --------------------------------------------------------------------------- #
# fall back to native when the seam is unavailable / returns None
# --------------------------------------------------------------------------- #
def test_processed_falls_back_to_native_when_seam_returns_none(monkeypatch):
    calls = []
    _spy_seam(monkeypatch, calls, returns_layer=False)     # seam None per channel
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))

    ctl._maybe_swap_progressive_on_ndisplay()

    assert len(calls) == 2                          # seam attempted per channel
    assert ctl._processed_progressive_active is False   # stayed native
    # native fallback re-added multiscale ⟨processed⟩ layers with the marker names
    names = [getattr(l, "name", None) for l in ctl._processed_store_layers]
    assert names == [f"{_PROC}ch0", f"{_PROC}ch1"]
    assert all(getattr(l, "multiscale", False) for l in ctl._processed_store_layers)


def test_processed_stays_native_when_progressive_unavailable(monkeypatch):
    monkeypatch.setattr(prog, "progressive_available", lambda: False)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))

    ctl._maybe_swap_progressive_on_ndisplay()

    # progressive_wanted() is False -> no swap scheduled at all
    assert timer.scheduled == 0
    assert ctl._processed_progressive_active is False


# --------------------------------------------------------------------------- #
# the ndisplay swap targets PROCESSED — the raw path is not engaged
# --------------------------------------------------------------------------- #
def test_dispatch_does_not_engage_raw_path(monkeypatch):
    calls = []
    _spy_seam(monkeypatch, calls)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))
    # a raw swap would touch these; a processed swap must not.
    raw_names_before = list(ctl._loaded_source_layers)

    ctl._maybe_swap_progressive_on_ndisplay()

    assert ctl._progressive_active is False          # raw flag untouched
    assert ctl._loaded_source_layers == raw_names_before


# --------------------------------------------------------------------------- #
# raw `_source_*` state is untouched by the processed progressive display
# --------------------------------------------------------------------------- #
def test_raw_source_state_untouched_by_processed_swap(monkeypatch):
    calls = []
    _spy_seam(monkeypatch, calls)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))
    snap = dict(
        loaded=list(ctl._loaded_source_layers),
        chans=list(ctl._source_display_channels),
        lazy=ctl._source_lazy,
        axes=ctl._source_axes_resolved,
        idx=dict(ctl._channel_index_by_name),
        file=ctl._loaded_source_file,
        store=ctl._current_multiscale_store,
        on_canvas=ctl._source_layers_on_canvas,
    )

    ctl._maybe_swap_progressive_on_ndisplay()

    assert ctl._processed_progressive_active is True   # processed DID swap
    # …yet EVERY raw field is byte-identical
    assert ctl._loaded_source_layers == snap["loaded"]
    assert ctl._source_display_channels == snap["chans"]
    assert ctl._source_lazy == snap["lazy"]
    assert ctl._source_axes_resolved == snap["axes"]
    assert ctl._channel_index_by_name == snap["idx"]
    assert ctl._loaded_source_file == snap["file"]
    assert ctl._current_multiscale_store == snap["store"]
    assert ctl._source_layers_on_canvas == snap["on_canvas"]


# --------------------------------------------------------------------------- #
# teardown / clear DETACHES the processed progressive layers + resets the flag
# --------------------------------------------------------------------------- #
def test_clear_detaches_processed_progressive_and_uninstalls_meter(monkeypatch):
    # Drive the REAL GLIR-meter registry so the seam's removed-teardown is
    # exercised: a progressive add registers + wires teardown; the clear's
    # hide-then-remove fires the removed event -> meter uninstalled.
    class _Meter:
        def __init__(self):
            self._on = False

        def is_installed(self):
            return self._on

        def install(self):
            self._on = True

        def uninstall(self):
            self._on = False

    meter = _Meter()
    monkeypatch.setattr(prog, "_load_metering", lambda: meter)
    monkeypatch.setattr(prog, "progressive_available", lambda: True)
    monkeypatch.setattr(prog, "_METER_ENABLED", True)
    prog._live_refs.clear()
    prog._teardown_handlers.clear()

    def _spy(viewer, levels, name=None, colormap=None, blending=None,
             contrast_limits=None, scale=None, **_kw):
        prog.install_glir_if_needed()
        layer = _FakeLayer(name)
        viewer.layers.append(layer)
        prog._register(layer)
        prog._wire_teardown(viewer, layer)
        return layer

    monkeypatch.setattr(prog, "add_progressive_image", _spy)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))

    ctl._maybe_swap_progressive_on_ndisplay()
    assert ctl._processed_progressive_active is True
    assert meter.is_installed() is True
    assert prog._live_count() == 2

    ctl._clear_processed_store_layers()

    assert ctl._processed_store_layers == []
    assert ctl._processed_progressive_active is False
    assert prog._live_count() == 0
    assert meter.is_installed() is False            # meter detached on last remove
    # no ⟨processed⟩ layer strands on the canvas
    assert not any(getattr(l, "name", "").startswith(_PROC) for l in viewer.layers)
    prog._uninstall_paint_loading_guard()           # never leak the napari patch


def test_teardown_preview_resets_processed_progressive_fields():
    ctl = _make_ctl(_FakeViewer(ndisplay=3), _Widget())
    ctl._processed_progressive_active = True
    # Stub the rest of `_teardown_preview`'s cooperators to isolate our reset.
    ctl._clear_preview_layers = lambda: None
    ctl._clear_processed_layers = lambda: None
    ctl._destroy_preview_box = lambda: None
    ctl._refresh_preview_region_active = lambda *a: None
    ctl._refresh_view_toggle_state = lambda: None
    ctl._processed_cache = None

    ctl._teardown_preview()

    assert ctl._current_processed_multiscale_store is None
    assert ctl._processed_display_channels is None
    assert ctl._processed_progressive_active is False
    assert ctl._processed_store_layers == []


# --------------------------------------------------------------------------- #
# progressive->native round-trip (entering 2D / turning progressive off)
# --------------------------------------------------------------------------- #
def test_processed_progressive_swaps_back_to_native_on_2d(monkeypatch):
    calls = []
    _spy_seam(monkeypatch, calls)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))
    ctl._maybe_swap_progressive_on_ndisplay()       # -> progressive
    assert ctl._processed_progressive_active is True

    viewer.dims.ndisplay = 2
    ctl._maybe_swap_progressive_on_ndisplay()       # -> native (synchronous)

    assert ctl._processed_progressive_active is False
    names = [getattr(l, "name", None) for l in ctl._processed_store_layers]
    assert names == [f"{_PROC}ch0", f"{_PROC}ch1"]
    assert all(getattr(l, "multiscale", False) for l in ctl._processed_store_layers)


# --------------------------------------------------------------------------- #
# `_show_processed_view` invokes the swap once processed is the active view
# --------------------------------------------------------------------------- #
def test_show_processed_view_engages_progressive_swap():
    ctl = DPC.__new__(DPC)
    ctl._log = lambda *a, **k: None
    ctl.viewer = _FakeViewer(ndisplay=3)
    ctl._processed_available = True
    ctl._processed_store_layers = [_FakeLayer(f"{_PROC}ch0")]
    ctl._processed_layers = []
    ctl._source_layers_on_canvas = True
    ctl._swapping_source_display = False
    calls = []
    ctl._maybe_swap_progressive_on_ndisplay = lambda: calls.append(True)
    ctl._view_swap_busy_show = lambda *a: None
    ctl._view_swap_busy_hide = lambda: None
    ctl._refresh_view_toggle_state = lambda: None
    ctl._readd_processed_layers = lambda: None

    def _remove_raw():
        ctl._source_layers_on_canvas = False        # clean swap removed raw

    ctl._remove_source_layers_from_viewer = _remove_raw

    ctl._show_processed_view()

    assert calls == [True]                          # swap engaged after raw removal


# --------------------------------------------------------------------------- #
# zoom-depth applies live to the processed progressive layers
# --------------------------------------------------------------------------- #
def test_zoom_depth_applies_to_processed_progressive_layers(monkeypatch):
    calls = []
    _spy_seam(monkeypatch, calls)
    timer = _ImmediateQTimer()
    monkeypatch.setattr(_ctl_mod, "QTimer", timer)
    viewer = _FakeViewer(ndisplay=3)
    ctl = _make_ctl(viewer, _Widget(checked=True))
    ctl._kick_progressive_loader = lambda layer: None
    ctl._maybe_swap_progressive_on_ndisplay()       # -> progressive
    assert ctl._processed_progressive_active is True

    # ZOOM_DEPTH_MODES index 1 is a real fraction (0.25) — a pinning mode.
    ctl.on_zoom_depth_changed(1)

    assert ctl._zoom_depth_wanted == _ctl_mod.ZOOM_DEPTH_MODES[1]
    for l in ctl._processed_store_layers:
        assert l._zoom_depth == _ctl_mod.ZOOM_DEPTH_MODES[1]
        assert l.apply_zoom_depth_calls == 1        # recompute+reslice, not inert
