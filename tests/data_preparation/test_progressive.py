"""Headless seam-safety + GLIR-lifecycle tests for the progressive-3D seam.

Covers the three-audit findings on
``packages/epicycle_largedata/progressive.py`` WITHOUT a GUI or a GL canvas:

* GLIR meter lifecycle — install on add, no-uninstall while >1 layer is live,
  uninstall on the last remove (the process-wide patch must not leak).
* Atomic fail-closed — the vendored engine appends the layer and flips the
  global async slicing settings INSIDE the swap window, before its own later
  failure points.  On any error the seam must roll ALL of that back (remove the
  orphan layer, restore async/force_sync, uninstall the meter it installed) and
  return None, so the caller's manual-slider fallback starts clean.
* Layer-class swap — restores ``napari.layers.Image``/``Labels`` on both normal
  and exception exit; the swap is main-thread-only and non-reentrant.
* Capability gating + viewer requirement — fail closed (return None, never
  raise) when the engine is unavailable or ``viewer`` is None.

Everything runs against plain-object fakes (viewer / layers / settings /
slicer / metering / engine).  The seam's real GL/vispy/napari touch points are
factored behind ``progressive._load_engine`` / ``_load_metering`` /
``_get_settings`` / ``progressive_available``, which these tests monkeypatch.
The only real napari touch is ``import napari.layers`` (Qt-free), needed to
assert the class-swap restore.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

import napari.layers as _nl

from packages.epicycle_largedata import progressive as prog


# ---------------------------------------------------------------------------
# Qt-free fakes
# ---------------------------------------------------------------------------
class FakeSignal:
    """Minimal psygnal-shaped signal: connect / disconnect / emit."""

    def __init__(self):
        self._slots: list = []

    def connect(self, cb):
        self._slots.append(cb)

    def disconnect(self, cb):
        self._slots = [s for s in self._slots if s is not cb]

    def emit(self, event):
        # iterate a copy so a slot may disconnect itself mid-emit (psygnal is
        # reentrancy-safe; the real teardown handler self-disconnects).
        for slot in list(self._slots):
            slot(event)


class FakeEvents:
    def __init__(self):
        self.removed = FakeSignal()


class FakeEvent:
    """napari's removed event carries the removed item on ``.value``."""

    def __init__(self, value):
        self.value = value


class FakeLayers(list):
    """List with a ``removed`` signal that fires on ``remove`` (like napari's
    EventedList) so a real-ish removal drives the teardown handler."""

    def __init__(self):
        super().__init__()
        self.events = FakeEvents()

    def remove(self, item):
        super().remove(item)
        self.events.removed.emit(FakeEvent(item))


class FakeSlicer:
    def __init__(self):
        self._force_sync = True


class FakeViewer:
    def __init__(self):
        self.layers = FakeLayers()
        self._layer_slicer = FakeSlicer()
        # window teardown wiring reaches viewer.window._qt_window; None makes
        # that best-effort branch a guarded no-op (headless).
        self.window = None


class FakeExperimental:
    def __init__(self):
        self.async_ = False


class FakeSettings:
    def __init__(self):
        self.experimental = FakeExperimental()


class FakeLayer:
    """Weakref-able stand-in for a napari layer."""

    def __init__(self, name='layer'):
        self.name = name
        self.visible = True
        self.metadata: dict = {}

    def __repr__(self):  # pragma: no cover - debug aid
        return f'<FakeLayer {self.name!r}>'


class FakeMeter:
    """Stand-in for ``_glir_metering`` (install/uninstall/is_installed)."""

    def __init__(self):
        self._installed = False
        self.install_count = 0
        self.uninstall_count = 0

    def is_installed(self):
        return self._installed

    def install(self):
        if not self._installed:
            self.install_count += 1
        self._installed = True
        return True

    def uninstall(self):
        if self._installed:
            self.uninstall_count += 1
        self._installed = False


class FakeEngine:
    """Stand-in for the vendored ``_progressive_loading`` module.

    Mirrors the engine's in-swap-window mutation ORDER: append the layer, then
    flip the global async slicing settings, THEN (optionally) fail — exactly
    where the real engine's loader ctor / debug overlay can raise.
    """

    def __init__(self, raise_after=False):
        self.raise_after = raise_after
        self.image_calls = 0
        self.labels_calls = 0
        self.appended: list = []

    def _add(self, viewer, name):
        layer = FakeLayer(name)
        viewer.layers.append(layer)  # engine append (BEFORE any failure)
        self.appended.append(layer)
        settings = prog._get_settings()
        settings.experimental.async_ = True  # engine async flip
        viewer._layer_slicer._force_sync = False  # engine force_sync flip
        if self.raise_after:
            raise RuntimeError('engine boom after in-swap-window mutations')
        return layer

    def add_progressive_loading_image(self, levels, viewer=None, **kwargs):
        self.image_calls += 1
        return self._add(viewer, 'progressive-image')

    def add_progressive_loading_labels(self, levels, viewer=None, **kwargs):
        self.labels_calls += 1
        return self._add(viewer, 'progressive-labels')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_progressive_state():
    """Snapshot + restore all module-global state the seam mutates, so tests
    are order-independent and never leak the class swap / registry / meter."""
    saved_live = list(prog._live_refs)
    saved_handlers = list(prog._teardown_handlers)
    saved_active = prog._swap_layer_classes._active
    saved_image = _nl.Image
    saved_labels = _nl.Labels
    prog._live_refs.clear()
    prog._teardown_handlers.clear()
    prog._swap_layer_classes._active = False
    try:
        yield
    finally:
        prog._live_refs[:] = saved_live
        prog._teardown_handlers[:] = saved_handlers
        prog._swap_layer_classes._active = saved_active
        _nl.Image = saved_image
        _nl.Labels = saved_labels


def _patch_success(monkeypatch, meter=None, meter_enabled=True):
    """Wire the seam to a non-raising fake engine + fake meter + fake settings.

    ``meter_enabled`` forces ``prog._METER_ENABLED`` — default True (the
    production config: meter ON for the present/rendering path, with its
    deferred-DELETE reorder disabled at source via ``_glir_metering.
    _DEFER_DELETES``); pass False to exercise the disabled-meter fallback.
    """
    monkeypatch.setattr(prog, 'progressive_available', lambda: True)
    settings = FakeSettings()
    monkeypatch.setattr(prog, '_get_settings', lambda: settings)
    engine = FakeEngine(raise_after=False)
    monkeypatch.setattr(prog, '_load_engine', lambda: engine)
    meter = meter or FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)
    monkeypatch.setattr(prog, '_METER_ENABLED', meter_enabled)
    return engine, meter, settings


# ---------------------------------------------------------------------------
# Smoke: module imports + real availability probe (Qt-free)
# ---------------------------------------------------------------------------
def test_module_imports_and_public_api():
    for name in prog.__all__:
        assert hasattr(prog, name), name


def test_real_available_is_bool_and_viewer_required():
    avail = prog.progressive_available()
    assert isinstance(avail, bool)
    if avail:
        # real code path (no monkeypatch): viewer=None must fail closed
        # without constructing anything heavy or raising.
        assert prog.add_progressive_image(None, [np.zeros((4, 4))]) is None


# ---------------------------------------------------------------------------
# GLIR meter lifecycle (Finding 3)
# ---------------------------------------------------------------------------
def test_install_on_add(monkeypatch):
    engine, meter, _ = _patch_success(monkeypatch)
    viewer = FakeViewer()

    layer = prog.add_progressive_image(viewer, [np.zeros((8, 8))])

    assert layer is not None
    assert engine.image_calls == 1
    assert meter.is_installed() is True
    assert meter.install_count == 1
    assert prog._live_count() == 1
    assert layer in list(viewer.layers)


def test_no_uninstall_while_two_live(monkeypatch):
    engine, meter, _ = _patch_success(monkeypatch)
    viewer = FakeViewer()

    l1 = prog.add_progressive_image(viewer, [np.zeros((8, 8))])
    l2 = prog.add_progressive_image(viewer, [np.zeros((8, 8))])
    assert prog._live_count() == 2
    assert meter.install_count == 1  # 2nd add sees it already installed

    viewer.layers.remove(l1)  # fires removed -> teardown handler(s)

    assert prog._live_count() == 1
    assert meter.is_installed() is True  # one still live -> stays installed
    assert meter.uninstall_count == 0
    assert l2 in list(viewer.layers)


def test_uninstall_on_last_remove(monkeypatch):
    engine, meter, _ = _patch_success(monkeypatch)
    viewer = FakeViewer()

    l1 = prog.add_progressive_image(viewer, [np.zeros((8, 8))])
    l2 = prog.add_progressive_image(viewer, [np.zeros((8, 8))])

    viewer.layers.remove(l1)
    assert meter.is_installed() is True  # not the last one yet
    viewer.layers.remove(l2)

    assert prog._live_count() == 0
    assert meter.is_installed() is False  # last one gone -> uninstalled
    assert meter.uninstall_count == 1


def test_teardown_handler_ignores_foreign_layer_removal(monkeypatch):
    """A removed event for a NON-progressive layer must not tear anything
    down (guards the 2-live invariant against unrelated layers)."""
    engine, meter, _ = _patch_success(monkeypatch)
    viewer = FakeViewer()

    layer = prog.add_progressive_image(viewer, [np.zeros((8, 8))])
    foreign = FakeLayer('foreign')
    viewer.layers.append(foreign)

    viewer.layers.remove(foreign)  # foreign removal

    assert prog._live_count() == 1
    assert meter.is_installed() is True
    assert layer in list(viewer.layers)


# ---------------------------------------------------------------------------
# Atomic fail-closed rollback (Finding 1)
# ---------------------------------------------------------------------------
def test_fail_closed_rolls_back_orphan_settings_and_meter(monkeypatch):
    monkeypatch.setattr(prog, 'progressive_available', lambda: True)
    settings = FakeSettings()  # async_ = False
    monkeypatch.setattr(prog, '_get_settings', lambda: settings)
    engine = FakeEngine(raise_after=True)
    monkeypatch.setattr(prog, '_load_engine', lambda: engine)
    meter = FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)
    monkeypatch.setattr(prog, '_METER_ENABLED', True)  # exercise meter rollback

    viewer = FakeViewer()  # _force_sync = True
    assert _nl.Image is prog._NapariImage  # baseline before swap

    result = prog.add_progressive_image(viewer, [np.zeros((8, 8))])

    # the engine really ran + mutated before raising (so the test is meaningful)
    assert engine.image_calls == 1
    assert len(engine.appended) == 1

    assert result is None
    # orphan removed via safe hide-then-remove
    assert list(viewer.layers) == []
    assert engine.appended[0].visible is False
    # async slicing settings restored to pre-call values
    assert settings.experimental.async_ is False
    assert viewer._layer_slicer._force_sync is True
    # meter uninstalled (this call installed it; no live layer remains)
    assert meter.is_installed() is False
    assert meter.uninstall_count == 1
    # registry clean, layer never left registered
    assert prog._live_count() == 0
    assert prog._live_refs == []
    # class swap restored (unrebound) on exception exit
    assert _nl.Image is prog._NapariImage
    assert _nl.Labels is prog._NapariLabels


def test_fail_closed_preserves_meter_when_another_layer_live(monkeypatch):
    """A failed 2nd add must NOT uninstall the meter the 1st (live) layer
    still needs, and must not turn off async while it is still needed."""
    monkeypatch.setattr(prog, 'progressive_available', lambda: True)
    settings = FakeSettings()
    monkeypatch.setattr(prog, '_get_settings', lambda: settings)
    meter = FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)
    monkeypatch.setattr(prog, '_METER_ENABLED', True)  # exercise meter lifecycle

    viewer = FakeViewer()

    # 1st add succeeds (flips async True, installs meter, registers layer).
    good_engine = FakeEngine(raise_after=False)
    monkeypatch.setattr(prog, '_load_engine', lambda: good_engine)
    l1 = prog.add_progressive_image(viewer, [np.zeros((8, 8))])
    assert l1 is not None
    assert settings.experimental.async_ is True
    assert prog._live_count() == 1

    # 2nd add fails after mutating.
    bad_engine = FakeEngine(raise_after=True)
    monkeypatch.setattr(prog, '_load_engine', lambda: bad_engine)
    result = prog.add_progressive_image(viewer, [np.zeros((8, 8))])

    assert result is None
    # orphan from the failed add is gone; the good layer survives
    assert list(viewer.layers) == [l1]
    assert prog._live_count() == 1
    # meter still installed (installed_here was False on the 2nd add)
    assert meter.is_installed() is True
    assert meter.uninstall_count == 0
    # async restored to pre-2nd-call value (True) — still needed by l1
    assert settings.experimental.async_ is True


# ---------------------------------------------------------------------------
# Layer-class swap (Finding 4)
# ---------------------------------------------------------------------------
def test_swap_restores_on_normal_exit():
    assert _nl.Image is prog._NapariImage
    with prog._swap_layer_classes():
        assert _nl.Image is prog.ProgressiveImage
        assert _nl.Labels is prog.ProgressiveLabels
        assert prog._swap_layer_classes._active is True
    assert _nl.Image is prog._NapariImage
    assert _nl.Labels is prog._NapariLabels
    assert prog._swap_layer_classes._active is False


def test_swap_restores_on_exception_exit():
    with pytest.raises(ValueError, match='boom'):
        with prog._swap_layer_classes():
            assert _nl.Image is prog.ProgressiveImage
            raise ValueError('boom')
    assert _nl.Image is prog._NapariImage
    assert _nl.Labels is prog._NapariLabels
    assert prog._swap_layer_classes._active is False


def test_swap_reentrancy_raises_without_corrupting_originals():
    with prog._swap_layer_classes():
        assert _nl.Image is prog.ProgressiveImage
        with pytest.raises(RuntimeError, match='non-reentrant'):
            with prog._swap_layer_classes():
                pass
        # the failed inner enter must not have corrupted the outer swap
        assert _nl.Image is prog.ProgressiveImage
    assert _nl.Image is prog._NapariImage  # outer restored the real class
    assert prog._swap_layer_classes._active is False


def test_swap_off_thread_raises_and_leaves_bindings_intact():
    errors: list = []

    def run():
        try:
            with prog._swap_layer_classes():
                pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=run)
    t.start()
    t.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert 'main thread' in str(errors[0])
    # the off-thread guard raised BEFORE mutating anything
    assert _nl.Image is prog._NapariImage
    assert _nl.Labels is prog._NapariLabels
    assert prog._swap_layer_classes._active is False


# ---------------------------------------------------------------------------
# Capability gating + viewer requirement (Findings 2 + 4)
# ---------------------------------------------------------------------------
def test_gating_returns_none_when_engine_unavailable(monkeypatch):
    def boom():
        raise ImportError('vendored engine missing')

    # progressive_available() calls _load_engine() first and fails closed.
    monkeypatch.setattr(prog, '_load_engine', boom)
    assert prog.progressive_available() is False

    meter = FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)
    viewer = FakeViewer()

    result = prog.add_progressive_image(viewer, [np.zeros((4, 4))])

    assert result is None  # returned None without raising
    assert list(viewer.layers) == []  # nothing added
    assert meter.install_count == 0  # meter never touched
    assert prog._live_count() == 0


def test_viewer_none_returns_none_without_calling_engine(monkeypatch):
    monkeypatch.setattr(prog, 'progressive_available', lambda: True)
    engine = FakeEngine(raise_after=False)
    monkeypatch.setattr(prog, '_load_engine', lambda: engine)
    meter = FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)

    result = prog.add_progressive_image(None, [np.zeros((4, 4))])

    assert result is None
    assert engine.image_calls == 0  # never entered the swap window
    assert meter.install_count == 0  # meter never installed


def test_setup_failure_fails_closed(monkeypatch):
    """Finding 2: an engine-load failure INSIDE the try (past the availability
    gate) must return None, not raise."""
    monkeypatch.setattr(prog, 'progressive_available', lambda: True)

    def boom():
        raise ImportError('engine import blew up after the gate')

    monkeypatch.setattr(prog, '_load_engine', boom)
    meter = FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)
    viewer = FakeViewer()

    result = prog.add_progressive_image(viewer, [np.zeros((4, 4))])

    assert result is None
    assert list(viewer.layers) == []
    assert prog._live_count() == 0


# ---------------------------------------------------------------------------
# Labels path parity (Finding 4 single-path)
# ---------------------------------------------------------------------------
def test_labels_path_installs_and_registers(monkeypatch):
    engine, meter, _ = _patch_success(monkeypatch)
    viewer = FakeViewer()

    layer = prog.add_progressive_labels(viewer, [np.zeros((8, 8), dtype=int)])

    assert layer is not None
    assert engine.labels_calls == 1
    assert engine.image_calls == 0
    assert meter.is_installed() is True
    assert prog._live_count() == 1


# ---------------------------------------------------------------------------
# GLIR draw guard — skip (+ log) a DRAW whose index buffer is missing, so the
# metered-flush reorder cannot crash the paint via `ibuf.activate() on None`.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_draw_guard():
    """Never leak the vispy GlirProgram.draw monkeypatch between tests."""
    yield
    prog._uninstall_draw_guard()


def test_draw_guard_skips_missing_index_buffer(capsys):
    from vispy.gloo.glir import GlirProgram

    prog._install_draw_guard()
    assert prog._draw_guard_installed is True

    class _P:
        def get_object(self, _id):
            return None                     # index buffer absent -> would crash

    class _Self:
        _parser = _P()

    # Indexed draw with a missing index buffer -> guard SKIPS (returns None) and
    # does NOT reach the real vispy draw (which would do GL work on a fake self).
    out = GlirProgram.draw(_Self(), "triangle_strip", (1308, "UNSIGNED_INT", 14))
    assert out is None
    assert "[PROG-GLGUARD]" in capsys.readouterr().out


def test_draw_guard_install_uninstall_toggles():
    from vispy.gloo.glir import GlirProgram

    original = GlirProgram.draw
    prog._install_draw_guard()
    assert prog._draw_guard_installed is True
    assert GlirProgram.draw is not original
    prog._uninstall_draw_guard()
    assert prog._draw_guard_installed is False
    assert GlirProgram.draw is original


def test_draw_guard_passes_through_valid_draws(monkeypatch):
    """No false positives: a present index buffer AND a non-indexed (len-2)
    selection both call straight through to the real draw (grill-audit LOW-4)."""
    from vispy.gloo.glir import GlirProgram

    calls = []

    def _spy(self, mode, selection, instances=1):
        calls.append((mode, selection, instances))

    # Pre-patch with a spy, THEN install the guard so it captures the spy as the
    # "real" draw -> we can assert call-through without a GL context.  Uninstall
    # in `finally` (restores the spy) BEFORE monkeypatch restores the real draw,
    # so nothing leaks.
    monkeypatch.setattr(GlirProgram, "draw", _spy)
    prog._install_draw_guard()
    try:
        class _Present:
            def get_object(self, _id):
                return object()          # index buffer present

        class _Self:
            _parser = _Present()

        # len-3 selection, buffer present -> passes through
        GlirProgram.draw(_Self(), "triangle_strip", (7, "UNSIGNED_INT", 14))
        # len-2 selection (non-indexed) -> guard doesn't apply, passes through
        GlirProgram.draw(_Self(), "triangles", (0, 100))
    finally:
        prog._uninstall_draw_guard()

    assert calls == [
        ("triangle_strip", (7, "UNSIGNED_INT", 14), 1),
        ("triangles", (0, 100), 1),
    ]


# ---------------------------------------------------------------------------
# napari layer-list loading-spinner guard (napari 0.8 `_paint_loading` bug):
# the reversible `LayerDelegate._paint_loading` graft that stops the
# None-SizeHintRole -> None.height() crash during the synchronous 3D<->2D swap.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_paint_guard():
    """Never leak the napari LayerDelegate._paint_loading monkeypatch."""
    yield
    prog._uninstall_paint_loading_guard()


def test_paint_guard_install_uninstall_toggles():
    pytest.importorskip("napari._qt.containers._layer_delegate")
    from napari._qt.containers._layer_delegate import LayerDelegate

    original = LayerDelegate._paint_loading
    prog._install_paint_loading_guard()
    assert prog._paint_guard_installed is True
    assert LayerDelegate._paint_loading is not original
    prog._uninstall_paint_loading_guard()
    assert prog._paint_guard_installed is False
    assert LayerDelegate._paint_loading is original


def test_paint_guard_skips_stale_none_sizehint(capsys):
    """The crash case: a stale/out-of-range row returns None for SizeHintRole,
    so the guard SKIPS the spinner (returns None) instead of `None.height()`."""
    pytest.importorskip("napari._qt.containers._layer_delegate")
    from napari._qt.containers._layer_delegate import LayerDelegate

    prog._install_paint_loading_guard()
    assert prog._paint_guard_installed is True

    class _StaleIndex:
        def isValid(self):
            return True                 # valid, but out-of-range row
        def data(self, _role):
            return None                 # SizeHintRole -> None -> would crash

    out = LayerDelegate._paint_loading(object(), object(), object(), _StaleIndex())
    assert out is None
    assert "[PROG-PAINTGUARD]" in capsys.readouterr().out


def test_paint_guard_passes_through_valid_index(monkeypatch):
    """No false positives: a VALID row (real size hint) calls straight through
    to the real `_paint_loading`, so genuine loading spinners still draw."""
    pytest.importorskip("napari._qt.containers._layer_delegate")
    from napari._qt.containers._layer_delegate import LayerDelegate

    calls = []

    def _spy(self, painter, option, index):
        calls.append(index)

    # Pre-patch with a spy, THEN install the guard so it captures the spy as the
    # "real" _paint_loading -> assert call-through without real Qt paint objects.
    monkeypatch.setattr(LayerDelegate, "_paint_loading", _spy)
    prog._install_paint_loading_guard()
    try:
        class _QSizeLike:
            def height(self):
                return 34
        class _ValidIndex:
            def isValid(self):
                return True
            def data(self, _role):
                return _QSizeLike()      # non-None size hint -> passes through
        idx = _ValidIndex()
        LayerDelegate._paint_loading(object(), object(), object(), idx)
    finally:
        prog._uninstall_paint_loading_guard()

    assert calls == [idx]


# ---------------------------------------------------------------------------
# Meter DISABLED fallback (`_METER_ENABLED=False`).  Production runs the meter
# ON (needed for the present/rendering path) with its deferred-DELETE reorder
# turned off at source (`_glir_metering._DEFER_DELETES=False`).  These tests
# cover the still-supported disabled config: a progressive add registers +
# guards, but installs no meter.
# ---------------------------------------------------------------------------
def test_meter_disabled_registers_without_installing_meter(monkeypatch):
    engine, meter, _ = _patch_success(monkeypatch, meter_enabled=False)
    viewer = FakeViewer()

    layer = prog.add_progressive_image(viewer, [np.zeros((8, 8))])

    assert layer is not None
    assert prog._live_count() == 1
    assert layer in list(viewer.layers)
    assert meter.is_installed() is False        # meter NOT installed
    assert meter.install_count == 0
    assert prog._draw_guard_installed is True   # guard still on (belt+braces)


def test_meter_install_suppressed_noops_and_restores(monkeypatch):
    monkeypatch.setattr(prog, '_METER_ENABLED', False)
    meter = FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)

    with prog._meter_install_suppressed():
        meter.install()                         # rebound to a no-op
    assert meter.install_count == 0
    assert meter.is_installed() is False

    meter.install()                             # restored: works normally again
    assert meter.install_count == 1
    assert meter.is_installed() is True


def test_meter_install_suppressed_passthrough_when_enabled(monkeypatch):
    monkeypatch.setattr(prog, '_METER_ENABLED', True)
    meter = FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)

    with prog._meter_install_suppressed():
        meter.install()                         # NOT suppressed when enabled
    assert meter.install_count == 1


def test_meter_install_suppressed_neutralizes_the_real_module(monkeypatch):
    """End-to-end (grill LOW-2): the CM neutralizes the REAL `_glir_metering`
    module — the same object the engine calls at :3205 — so a real
    `install()` inside it patches no flush.  Defensive uninstall so a CM bug
    can never leak a real vispy patch into other tests."""
    monkeypatch.setattr(prog, '_METER_ENABLED', False)   # the disabled config
    metering = prog._load_metering()             # the real vendored module
    assert metering.is_installed() is False       # baseline: no meter installed
    try:
        with prog._meter_install_suppressed():    # _METER_ENABLED forced False
            metering.install()                    # engine's call -> suppressed
            assert metering.is_installed() is False
        assert metering.is_installed() is False    # still not installed after
    finally:
        if metering.is_installed():                # never leak a real patch
            metering.uninstall()


def test_meter_off_fail_closed_uninstalls_guard_no_meter(monkeypatch):
    """Meter-OFF fail-closed (grill LOW-1): a failed add rolls back the orphan,
    tears down the draw guard, and never touches the (never-installed) meter."""
    monkeypatch.setattr(prog, 'progressive_available', lambda: True)
    settings = FakeSettings()
    monkeypatch.setattr(prog, '_get_settings', lambda: settings)
    engine = FakeEngine(raise_after=True)
    monkeypatch.setattr(prog, '_load_engine', lambda: engine)
    meter = FakeMeter()
    monkeypatch.setattr(prog, '_load_metering', lambda: meter)
    monkeypatch.setattr(prog, '_METER_ENABLED', False)   # the disabled fallback

    viewer = FakeViewer()
    result = prog.add_progressive_image(viewer, [np.zeros((8, 8))])

    assert result is None                        # fail closed
    assert list(viewer.layers) == []             # orphan rolled back
    assert prog._live_count() == 0
    assert meter.install_count == 0              # meter never installed
    assert meter.uninstall_count == 0            # nor uninstalled
    assert prog._draw_guard_installed is False   # guard torn down on cleanup


# ---------------------------------------------------------------------------
# Deferred-DELETE toggle (the crash fix): DELETEs run INLINE in queue order by
# default, not deferred to a later quiet flush where the drain can run ahead of
# a still-live command referencing the id as a secondary arg.
# ---------------------------------------------------------------------------
class _FakeGlirParser:
    def __init__(self):
        self._objects = {}
        self.parsed = []

    def _parse(self, command):
        self.parsed.append(command)


class _FakeParserState:
    def __init__(self):
        self.deferred_deletes = []
        self.dead_ids = set()
        self.gl_state = {}
        self.budget_left = 10 ** 9
        self.frame_budget = 10 ** 6
        self.executed_metered = False


def test_defer_deletes_off_runs_delete_inline():
    from packages.epicycle_largedata._vendor_progressive import _glir_metering as m

    assert m._DEFER_DELETES is False                  # production default
    parser, state = _FakeGlirParser(), _FakeParserState()
    m._metered_parse(parser, [('DELETE', 42)], state)
    assert ('DELETE', 42) in parser.parsed            # executed INLINE (queue order)
    assert 42 in state.dead_ids                        # dead_ids bookkeeping mirrored
    assert state.deferred_deletes == []                # NOT deferred to a later flush


def test_defer_deletes_on_defers_like_the_original(monkeypatch):
    from packages.epicycle_largedata._vendor_progressive import _glir_metering as m

    monkeypatch.setattr(m, '_DEFER_DELETES', True)
    parser, state = _FakeGlirParser(), _FakeParserState()
    m._metered_parse(parser, [('DELETE', 42)], state)
    assert parser.parsed == []                         # NOT executed inline
    assert state.deferred_deletes == [('DELETE', 42)]  # deferred to a quiet flush
    assert 42 not in state.dead_ids


def test_metered_parse_drops_zero_size_data():
    """Belt-and-suspenders: a zero-extent DATA payload (an empty 3D crop) is
    DROPPED, never reaching GL (a 0-size glTexSubImage would be GL_INVALID_VALUE
    and abort the frame)."""
    from packages.epicycle_largedata._vendor_progressive import _glir_metering as m

    parser, state = _FakeGlirParser(), _FakeParserState()
    empty = np.empty((1, 0, 0, 1), dtype="uint16")
    m._metered_parse(parser, [("DATA", 421, (0, 0, 0), empty)], state)
    assert parser.parsed == []                          # empty DATA dropped


def test_metered_parse_keeps_nonempty_and_shader_data():
    """The zero-size guard must NOT drop real uploads or shader source (a str
    with no `.size`)."""
    from packages.epicycle_largedata._vendor_progressive import _glir_metering as m

    parser, state = _FakeGlirParser(), _FakeParserState()
    real = np.ones((1, 4, 4, 1), dtype="uint16")
    m._metered_parse(
        parser,
        [("DATA", 1, 0, "shader source"), ("DATA", 2, (0, 0, 0), real)],
        state,
    )
    assert ("DATA", 1, 0, "shader source") in parser.parsed   # shader kept
    assert any(c[0] == "DATA" and c[1] == 2 for c in parser.parsed)  # real kept


# ---------------------------------------------------------------------------
# "Zoom depth" (EPICYCLE feature) — the `_zoom_depth` mode parameterizes the
# depth (Z) axis of the 3D sub-volume tile: None => Off (proportional band,
# byte-identical to the pre-feature default); a float in {0.25,0.5,0.75,1.0} =>
# pin depth to that FRACTION of full Z and shrink only the lateral axes; "cube"
# => isotropic box.  Read PER-DRAW in `_corners_for_locked_level`, so a
# controller change applies live.  These tests drive the graft method directly
# on a light plain-object fake (no GL canvas): the method reads only
# `level_shapes`, `ndim`, `dtype`, `_slice_input.ndisplay`, the tile-budget
# attrs, `downsample_factors`, and `_zoom_depth`.
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402


def _make_corners_fake(level_shape, *, dtype=np.uint16, tile_mb=512,
                       interval_mb=512, extent_cap=100_000, zoom_depth=None,
                       ndim=3):
    """A minimal stand-in carrying exactly what `_corners_for_locked_level`
    reads, so the graft method can run headlessly (bound via the mixin)."""
    f = SimpleNamespace()
    f.level_shapes = [tuple(int(s) for s in level_shape)]
    f.downsample_factors = [np.ones(ndim, dtype=int)]
    f.ndim = ndim
    f.dtype = np.dtype(dtype)
    f._slice_input = SimpleNamespace(ndisplay=3)
    f._max_tile_extent_3d = extent_cap
    f._tile_max_bytes_3d = int(tile_mb * 1024 * 1024) if tile_mb else None
    f._interval_max_bytes_3d = (
        int(interval_mb * 1024 * 1024) if interval_mb else None)
    f._zoom_depth = zoom_depth
    return f


def _corners(fake, displayed=(0, 1, 2), data_bbox_int=None):
    return prog._ProgressiveScalarFieldMixin._corners_for_locked_level(
        fake, 0, displayed, data_bbox_int, None)


@pytest.fixture
def _pinned_gl_max(monkeypatch):
    """Pin the 3D GL max-texture size so the tile math is deterministic
    headlessly (the real `get_max_texture_sizes` needs a GL context)."""
    import napari._vispy.utils.gl as _gl

    monkeypatch.setattr(_gl, "get_max_texture_sizes", lambda: (2048, 2048))


class TestZoomDepth:
    _DISP = [0, 1, 2]
    _REF = (381, 4096, 4096)   # thin-Z reference level (uint16), 512 MB budget

    # -- (1) Off (None) is byte-identical to the pre-feature camera band -------
    def test_off_is_camera_centered_band(self, _pinned_gl_max):
        off = _corners(_make_corners_fake(self._REF, zoom_depth=None))
        # Exact pre-feature corners for this level+budget (regression pin): ALL
        # THREE displayed axes shrink together, so depth is a proportional band.
        assert off[0, self._DISP].tolist() == [78, 1472, 1472]
        assert off[1, self._DISP].tolist() == [301, 2623, 2623]
        depth_extent = int(off[1, 0] - off[0, 0] + 1)
        assert depth_extent == 224            # a band, NOT full
        assert depth_extent < 381             # < full Z

    def test_zoom_depth_absent_equals_explicit_none(self, _pinned_gl_max):
        """The `getattr(self, '_zoom_depth', None)` default path (a layer with no
        attr) must be byte-identical to explicit None (Off)."""
        f_none = _make_corners_fake(self._REF, zoom_depth=None)
        f_absent = _make_corners_fake(self._REF, zoom_depth=None)
        del f_absent._zoom_depth              # simulate a pre-feature layer
        assert np.array_equal(_corners(f_none), _corners(f_absent))

    # -- (2) Full (1.0) is byte-identical to the pre-v2 checkbox ON ------------
    def test_full_matches_pre_v2_full_depth_on(self, _pinned_gl_max):
        on = _corners(_make_corners_fake(self._REF, zoom_depth=1.0))
        # The exact corners the pre-v2 "Full depth ON" produced.
        assert on[0, self._DISP].tolist() == [0, 1616, 1616]
        assert on[1, self._DISP].tolist() == [380, 2479, 2479]
        assert int(on[1, 0] - on[0, 0] + 1) == 381   # FULL depth
        assert self._tile_fits(on, budget_mb=512)

    # -- (3) fractional modes: depth ~= frac*full, lateral < full + monotonic --
    @pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
    def test_fraction_pins_depth_and_fits(self, _pinned_gl_max, frac):
        c = _corners(_make_corners_fake(self._REF, zoom_depth=frac))
        depth = int(c[1, 0] - c[0, 0] + 1)
        expected = int(381 * frac)
        # depth ~= frac*full (within one quantum of the ceil rounding)
        assert abs(depth - expected) <= prog._TILE_EXTENT_QUANTUM_3D
        assert depth < 381                              # a fraction, not full
        lat = int(c[1, 1] - c[0, 1] + 1)
        assert lat < 4096                               # lateral shrank to fit
        assert int(c[1, 1] - c[0, 1]) == int(c[1, 2] - c[0, 2])   # isotropic XY
        assert self._tile_fits(c, budget_mb=512)

    def test_lateral_is_monotonic_in_depth_fraction(self, _pinned_gl_max):
        """More depth (larger fraction) => a NARROWER lateral window."""
        lats = []
        for frac in (0.25, 0.5, 0.75, 1.0):
            c = _corners(_make_corners_fake(self._REF, zoom_depth=frac))
            lats.append(int(c[1, 1] - c[0, 1] + 1))
        assert lats == sorted(lats, reverse=True)       # strictly decreasing
        assert all(lats[i] > lats[i + 1] for i in range(len(lats) - 1))

    # -- (4) cube: isotropic on a cubic level, within budget ------------------
    def test_cube_is_isotropic_on_cubic_level(self, _pinned_gl_max):
        # A level whose every dim exceeds the cube edge -> a true E×E×E box.
        c = _corners(_make_corners_fake((1024, 1024, 1024), zoom_depth="cube"))
        d = int(c[1, 0] - c[0, 0] + 1)
        y = int(c[1, 1] - c[0, 1] + 1)
        x = int(c[1, 2] - c[0, 2] + 1)
        assert d == y == x                              # isotropic cube
        assert self._tile_fits(c, budget_mb=512)

    def test_cube_clamps_to_shape_on_thin_level(self, _pinned_gl_max):
        # On the thin-Z reference level, depth (381) < cube edge, so it clamps to
        # the shape while the lateral axes take the cube edge — still within fit.
        c = _corners(_make_corners_fake(self._REF, zoom_depth="cube"))
        assert int(c[1, 0] - c[0, 0] + 1) == 381        # depth clamped to shape
        assert int(c[1, 1] - c[0, 1] + 1) < 4096        # lateral capped to edge
        assert self._tile_fits(c, budget_mb=512)

    # -- (5) byte-fit convergence with depth pinned across budgets ------------
    @pytest.mark.parametrize("mb", [512, 256, 80])
    @pytest.mark.parametrize("zd", [0.5, 1.0, "cube"])
    def test_byte_fit_convergence(self, _pinned_gl_max, mb, zd):
        c = _corners(_make_corners_fake(self._REF, tile_mb=mb,
                                        interval_mb=mb, zoom_depth=zd))
        # loop terminated + every axis >= MIN + tile fits the budget
        for ax in self._DISP:
            assert int(c[1, ax] - c[0, ax] + 1) >= prog._MIN_TILE_EXTENT_3D
        assert self._tile_fits(c, budget_mb=mb)

    def test_fraction_lateral_floors_at_min_on_tight_budget(
            self, _pinned_gl_max):
        """An extreme budget floors lateral at `_MIN_TILE_EXTENT_3D` while the
        pinned (fractional) depth holds and the bounded loop still terminates."""
        c = _corners(_make_corners_fake(self._REF, tile_mb=0.5,
                                        interval_mb=0.5, zoom_depth=1.0))
        assert int(c[1, 0] - c[0, 0] + 1) == 381                  # depth full
        assert int(c[1, 1] - c[0, 1] + 1) == prog._MIN_TILE_EXTENT_3D

    # -- (6) low[depth] never negative for any camera center (float mode) ------
    @pytest.mark.parametrize("bbox", [
        [[100, 500, 500], [300, 1500, 1500]],
        [[0, 0, 0], [380, 4095, 4095]],
        [[350, 3000, 3000], [381, 4096, 4096]],
    ])
    def test_full_low_depth_is_zero(self, _pinned_gl_max, bbox):
        on = _corners(_make_corners_fake(self._REF, zoom_depth=1.0),
                      data_bbox_int=np.asarray(bbox))
        assert int(on[0, 0]) == 0             # depth low pinned to 0 (no negative)
        assert int(on[1, 0]) == 380           # full Z regardless of center

    # -- (6b) hardened type gate: numeric pins, bool/str/None do not ----------
    def test_type_gate_accepts_numbers_excludes_bool(self, _pinned_gl_max):
        """Defensive gate: any real number (int / np.float32) pins depth like the
        equivalent float; bool (an int subclass) and None fall through to Off."""
        float_full = _corners(_make_corners_fake(self._REF, zoom_depth=1.0))
        # int 1 pins identically to float 1.0
        int_full = _corners(_make_corners_fake(self._REF, zoom_depth=1))
        assert np.array_equal(int_full, float_full)
        # np.float32 fraction still pins (depth < full, not the Off band)
        f32 = _corners(_make_corners_fake(self._REF, zoom_depth=np.float32(0.5)))
        assert int(f32[1, 0] - f32[0, 0] + 1) < 381
        assert np.array_equal(
            f32, _corners(_make_corners_fake(self._REF, zoom_depth=0.5)))
        # bool True/False do NOT pin -> byte-identical to Off (None)
        off = _corners(_make_corners_fake(self._REF, zoom_depth=None))
        for b in (True, False):
            assert np.array_equal(
                _corners(_make_corners_fake(self._REF, zoom_depth=b)), off)

    # -- (7) the mode defaults None on a freshly constructed layer ------------
    def test_mode_defaults_none_on_fresh_layers(self):
        img = prog.ProgressiveImage(np.zeros((8, 8), dtype=np.uint8))
        lbl = prog.ProgressiveLabels(np.zeros((8, 8), dtype=int))
        assert img._zoom_depth is None
        assert lbl._zoom_depth is None
        # Class-level default present too (bulletproofing before __init__ chains).
        assert prog._ProgressiveScalarFieldMixin._zoom_depth is None

    # -- helper ---------------------------------------------------------------
    @staticmethod
    def _tile_fits(corners, *, budget_mb, itemsize=2):
        """Loop-invariant byte-fit for EVERY mode.  The shrink loop + pin fit the
        PRE-quantum tile to <= budget; the quantum ceil then inflates each axis
        by < one quantum (texture-reuse rounding — pre-existing, and for a pinned
        FRACTIONAL depth it inflates the depth axis too, so the undo must be
        UNIFORM).  Subtract one quantum from EVERY axis (floor at MIN) and assert
        the fit."""
        disp = [0, 1, 2]
        q = prog._TILE_EXTENT_QUANTUM_3D
        tile = (corners[1, disp] - corners[0, disp] + 1).astype(np.int64)
        adj = np.array(
            [max(int(t) - q, prog._MIN_TILE_EXTENT_3D) for t in tile])
        return int(np.prod(adj)) * itemsize <= budget_mb * 1024 * 1024


def test_our_loader_forces_delete_deferral_off_after_a_revendor(monkeypatch):
    """TRIPWIRE: `_DEFER_DELETES = False` is the
    load-bearing 3D<->2D crash fix, and it used to live ONLY as a hand edit
    inside the DO-NOT-EDIT vendored file — a re-vendor would silently revert it.
    EpiCycle now asserts the value itself every time it loads the module."""
    from packages.epicycle_largedata._vendor_progressive import _glir_metering as m

    monkeypatch.setattr(m, "_DEFER_DELETES", True)   # simulate a fresh re-vendor
    assert prog._load_metering() is m
    assert m._DEFER_DELETES is False, (
        "EpiCycle must force the vendored deferred-DELETE reorder OFF")


def test_install_glir_forces_delete_deferral_off(monkeypatch):
    """The install site goes through the same door, so an install can never run
    with the crash-prone deferral on."""
    from packages.epicycle_largedata._vendor_progressive import _glir_metering as m

    monkeypatch.setattr(m, "_DEFER_DELETES", True)
    monkeypatch.setattr(prog, "_install_draw_guard", lambda: None)
    monkeypatch.setattr(prog, "_install_paint_loading_guard", lambda: None)
    monkeypatch.setattr(prog, "_draw_guard_installed", True)
    monkeypatch.setattr(prog, "_paint_guard_installed", True)
    monkeypatch.setattr(m, "is_installed", lambda: True)   # no real vispy patch

    prog.install_glir_if_needed()

    assert m._DEFER_DELETES is False
