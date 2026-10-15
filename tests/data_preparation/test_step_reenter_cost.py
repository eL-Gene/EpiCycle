"""Brick 3a #8 - the cost of LEAVING and RE-ENTERING Step 1 (assessment #14).

Two independent charges were paid on every re-enter even when nothing changed:

  1. `_estimate_multiscale_contrast` re-read up to 16 Mvox PER CHANNEL off the
     zarr store, because `_reshow_source_layers` re-adds the layers from the
     retained channel refs and the estimate had no memory.
  2. `WorkflowShell._restore_viewer_canvas_mode` ASSIGNED `dims.ndisplay` even
     when the value was unchanged, firing the dims event chain (and on napari
     0.8 a full per-layer visual rebuild / 2D<->3D texture swap) for nothing.

Both are pure accelerators: the cache must never survive a context change, and
the ndisplay skip must still assign when the value genuinely differs.

Qt-free / napari-free: `__new__` controllers and a minimal dims stub.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from napari_epicycle.data_preparation import controller


# --------------------------------------------------------------------------- #
# 1. contrast memoization
# --------------------------------------------------------------------------- #

class _CountingLevel:
    """A duck-array pyramid level that records how often it is realized."""

    def __init__(self, arr, counter):
        self._arr = arr
        self._counter = counter
        self.shape = arr.shape
        self.dtype = arr.dtype

    def __array__(self, dtype=None, copy=None):
        self._counter["reads"] += 1
        return self._arr if dtype is None else self._arr.astype(dtype)


def _ctl():
    ctl = controller.DataPreparationController.__new__(
        controller.DataPreparationController)
    ctl._log = lambda *a, **k: None
    return ctl


def _levels(counter, seed=0):
    rng = np.random.RandomState(seed)
    fine = (rng.rand(8, 64, 64) * 60000).astype(np.uint16)
    return [_CountingLevel(fine, counter)]


def test_contrast_is_computed_once_per_store_and_channel():
    counter = {"reads": 0}
    ctl = _ctl()
    lv = _levels(counter)

    first = ctl._estimate_multiscale_contrast(lv, store_key="/w/s.zarr",
                                              channel_index=0)
    assert first is not None
    assert counter["reads"] == 1

    second = ctl._estimate_multiscale_contrast(lv, store_key="/w/s.zarr",
                                               channel_index=0)
    assert counter["reads"] == 1, "a re-show must NOT re-read the store"
    assert second == first


def test_contrast_cache_is_keyed_by_channel_and_by_store():
    counter = {"reads": 0}
    ctl = _ctl()
    lv0, lv1 = _levels(counter, 0), _levels(counter, 1)

    ctl._estimate_multiscale_contrast(lv0, store_key="/w/a.zarr", channel_index=0)
    ctl._estimate_multiscale_contrast(lv1, store_key="/w/a.zarr", channel_index=1)
    assert counter["reads"] == 2, "channels must not share an entry"

    ctl._estimate_multiscale_contrast(lv0, store_key="/w/b.zarr", channel_index=0)
    assert counter["reads"] == 3, "a different store must not hit"


def test_contrast_is_not_cached_without_an_explicit_key():
    """No key -> compute fresh.  Guessing the store from
    `_current_multiscale_store` would be WRONG on the initial-load path (it still
    names the PREVIOUS store there), so a missing key must never silently hit."""
    counter = {"reads": 0}
    ctl = _ctl()
    lv = _levels(counter)
    ctl._estimate_multiscale_contrast(lv)
    ctl._estimate_multiscale_contrast(lv)
    ctl._estimate_multiscale_contrast(lv, store_key="/w/a.zarr")       # no channel
    assert counter["reads"] == 3
    assert not getattr(ctl, "_contrast_cache", {})


def test_clear_contrast_cache_forces_a_recompute():
    counter = {"reads": 0}
    ctl = _ctl()
    lv = _levels(counter)
    ctl._estimate_multiscale_contrast(lv, store_key="/w/a.zarr", channel_index=0)
    ctl._clear_contrast_cache("test")
    ctl._estimate_multiscale_contrast(lv, store_key="/w/a.zarr", channel_index=0)
    assert counter["reads"] == 2
    ctl._clear_contrast_cache()          # empty cache -> no crash
    ctl._contrast_cache = None
    ctl._clear_contrast_cache()          # missing cache -> no crash


def test_unload_source_clears_the_contrast_cache():
    counter = {"reads": 0}
    ctl = _ctl()
    ctl._estimate_multiscale_contrast(_levels(counter), store_key="/w/a.zarr",
                                      channel_index=0)
    assert ctl._contrast_cache

    ctl.box_tool = None
    ctl._clear_patch_locations = lambda: None
    ctl._teardown_preview = lambda: None
    ctl._loaded_source_layers = []
    ctl.viewer = SimpleNamespace(layers=[])
    ctl._on_layers_changed = lambda: None
    ctl.widget = SimpleNamespace(
        hide_3d_detail=None, hide_zoom_depth=None,
        source_status_label=SimpleNamespace(
            setText=lambda *_a: None, setStyleSheet=lambda *_a: None))
    ctl.unload_source()
    assert ctl._contrast_cache == {}


def test_estimate_never_raises_on_a_bad_level_even_with_a_key():
    ctl = _ctl()

    class _Boom:
        shape = (4, 4, 4)

        def __array__(self, *_a, **_k):
            raise RuntimeError("store gone")

    assert ctl._estimate_multiscale_contrast(
        [_Boom()], store_key="/w/a.zarr", channel_index=0) is None
    assert ctl._contrast_cache == {}, "a failure must not be cached"


# --------------------------------------------------------------------------- #
# 2. the ndisplay no-op skip
# --------------------------------------------------------------------------- #

class _Dims:
    def __init__(self, ndisplay, ndim=3):
        self._ndisplay = ndisplay
        self.ndim = ndim
        self.order = tuple(range(ndim))
        self.sets = []

    @property
    def ndisplay(self):
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, value):
        self.sets.append(value)
        self._ndisplay = value


def _shell(current_ndisplay, snapshot):
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell

    shell = EpiCycleWorkflowShell.__new__(EpiCycleWorkflowShell)
    dims = _Dims(current_ndisplay)
    shell.viewer = SimpleNamespace(dims=dims)
    shell._viewer_canvas_snapshot = snapshot
    return shell, dims


def test_ndisplay_is_not_reassigned_when_unchanged():
    shell, dims = _shell(3, {"ndisplay": 3})
    shell._restore_viewer_canvas_mode()
    assert dims.sets == [], "a same-value set still fires the dims event chain"
    assert dims.ndisplay == 3


def test_ndisplay_is_assigned_when_it_differs():
    shell, dims = _shell(2, {"ndisplay": 3})
    shell._restore_viewer_canvas_mode()
    assert dims.sets == [3]
    assert dims.ndisplay == 3


def test_order_restore_is_unaffected_by_the_ndisplay_skip():
    shell, dims = _shell(3, {"ndisplay": 3, "order": (2, 0, 1)})
    shell._restore_viewer_canvas_mode()
    assert dims.sets == []
    assert dims.order == (2, 0, 1)


def test_restore_is_still_one_shot():
    shell, dims = _shell(2, {"ndisplay": 3})
    shell._restore_viewer_canvas_mode()
    assert shell._viewer_canvas_snapshot is None
    shell._restore_viewer_canvas_mode()          # second call is a no-op
    assert dims.sets == [3]


# --------------------------------------------------------------------------- #
# 3. the WIRING - the key actually reaches the estimator on the re-enter path
# --------------------------------------------------------------------------- #

def test_reshow_source_layers_passes_the_current_store_as_the_contrast_key():
    """The unit tests above prove the cache works when given a key.  This one
    proves the RE-SHOW path actually supplies it: `_reshow_source_layers` ->
    `_add_source_display_layers` -> `_do_add_source_display_layers` ->
    `_estimate_multiscale_contrast(store_key=...)`, with the key equal to
    `_current_multiscale_store`.  Without this the memoization is dead code on
    the one path it exists for.

    Brick 5 note: step RE-ENTER no longer re-shows (see the no-auto-reload tests
    below), so the surviving caller is the raw<->processed view toggle, which
    re-adds raw fresh through this same helper."""
    ctl = _ctl()
    seen = []
    ctl._estimate_multiscale_contrast = (
        lambda levels, store_key=None, channel_index=None:
        seen.append((store_key, channel_index)) or [0.0, 1.0])

    levels = [np.zeros((4, 8, 8), np.uint16)]
    ctl._current_multiscale_store = "/w/store.zarr"
    ctl._source_display_channels = [("ch0", levels), ("ch1", levels)]
    ctl._loaded_source_layers = ["ch0", "ch1"]

    added = []

    class _Layer:
        def __init__(self, name):
            self.name = name
            self.dtype = np.uint16

    class _Viewer:
        layers = []

        def add_image(self, data, **kw):
            lay = _Layer(kw.get("name"))
            added.append(lay)
            return lay

    ctl.viewer = _Viewer()
    ctl._loading_source_layers = False
    ctl._on_layers_changed = lambda: None
    ctl._diag_physical_layers = lambda *_a: None
    ctl._track_source_layer_objects = lambda *_a: None
    ctl._widen_contrast_range_to_dtype = lambda *_a: None
    ctl._preview_spacing = lambda: (1.0, 1.0, 1.0)
    ctl._maybe_swap_progressive_on_ndisplay = lambda: None
    ctl._progressive_active = True
    ctl._source_layers_on_canvas = False

    ctl._reshow_source_layers()

    assert seen == [("/w/store.zarr", 0), ("/w/store.zarr", 1)], seen


def test_initial_load_does_not_reuse_the_previous_store_key():
    """The mirror hazard: on the INITIAL load `_current_multiscale_store` still
    names the PREVIOUS store, so the key must come from the caller, never from
    that attribute.  Passing no key must disable caching entirely."""
    counter = {"reads": 0}
    ctl = _ctl()
    ctl._current_multiscale_store = "/w/OLD.zarr"
    lv = _levels(counter)
    ctl._estimate_multiscale_contrast(lv)          # no key -> fresh
    ctl._estimate_multiscale_contrast(lv)          # still fresh
    assert counter["reads"] == 2
    assert not getattr(ctl, "_contrast_cache", {})


# --------------------------------------------------------------------------- #
# 4. Brick 5 - step re-enter does NOT reload the source
# --------------------------------------------------------------------------- #

class _EnterCtl:
    """The real `on_step_enter` / `on_step_exit` / `_note_source_unloaded_on_reentry`
    bound onto a stand-in whose collaborators are all recorders."""

    def __init__(self, *, channels=None):
        import types

        C = controller.DataPreparationController
        self.calls = []
        self.status = []
        self.viewer = SimpleNamespace(
            layers=[], scale_bar=SimpleNamespace(visible=True))
        self.widget = SimpleNamespace(apply_progressive_default_open=None)
        self._left_step1 = False
        self._source_needs_reload = False
        self._source_display_channels = channels
        self._loaded_source_layers = []
        self._source_layers_on_canvas = bool(channels)
        self._progressive_active = bool(channels)
        self._saved_channel_selection = ["ch0"]
        self._loaded_source_id = "src-1"
        self._log = lambda *a, **k: None
        for name in ("_apply_live_scale_bar", "_maybe_restore_step1_sources",
                     "_maybe_offer_step1_migration", "_reshow_source_layers"):
            setattr(self, name, self._rec(name))
        self._set_source_status_amber = lambda text: self.status.append(text)
        for name in ("on_step_enter", "_note_source_unloaded_on_reentry"):
            setattr(self, name, types.MethodType(getattr(C, name), self))

    def _rec(self, name):
        def _f(*_a, **_k):
            self.calls.append(name)
        return _f


def test_step_enter_does_not_re_add_the_source_layers():
    """User decision: Step 1 tears down on exit and does not put the data back
    by itself - re-adding a prepared store cost 1.1-1.4 s on every re-entry."""
    ctl = _EnterCtl(channels=[("ch0", [])])
    ctl._source_needs_reload = True
    ctl.on_step_enter()
    assert "_reshow_source_layers" not in ctl.calls
    assert ctl.status and "click Load" in ctl.status[0]
    assert "Step 1" in ctl.status[0]
    # the rest of enter still runs
    assert "_maybe_restore_step1_sources" in ctl.calls
    assert ctl._left_step1 is False


def test_reentry_clears_only_the_layer_dependent_bits():
    ctl = _EnterCtl(channels=[("ch0", [])])
    ctl._source_needs_reload = True
    ctl.on_step_enter()
    assert ctl._source_layers_on_canvas is False
    assert ctl._progressive_active is False
    assert ctl._saved_channel_selection is None
    # kept: the library stays interactive for the source they were working on,
    # and the channel refs still describe it (a Load re-shows them).
    assert ctl._loaded_source_id == "src-1"
    assert ctl._source_display_channels


def test_no_notice_when_nothing_was_loaded():
    ctl = _EnterCtl(channels=None)
    ctl.on_step_enter()
    assert ctl.status == []


def test_the_notice_is_shown_once_per_exit():
    ctl = _EnterCtl(channels=[("ch0", [])])
    ctl._source_needs_reload = True
    ctl.on_step_enter()
    ctl.on_step_enter()
    assert len(ctl.status) == 1, "the flag must be consumed on the first enter"


def test_step_exit_records_that_a_source_was_displayed():
    """The flag is what lets the next enter explain the empty canvas."""
    import inspect

    src = inspect.getsource(controller.DataPreparationController.on_step_exit)
    assert "_source_needs_reload" in src
    i_set = src.index("_source_needs_reload")
    i_remove = src.index("_remove_source_layers_from_viewer()")
    assert i_set < i_remove, "record BEFORE the removal (channels are read)"
