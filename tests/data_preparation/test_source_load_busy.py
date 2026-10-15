"""Brick 3a #7 - the (2) scan bar must NAME the source read and always come back.

``load_source`` runs the disk read on ``_SourceLoadWorker`` (a QThread), so the
bar is the only feedback the user gets that anything is happening.  These tests
pin the two halves of the contract:

  * the bar is shown, in SCAN mode, NAMING the file, from the LAUNCH (before the
    thread starts) - not after the first slice paints;
  * and it is hidden again on EVERY terminal: loaded, error, cancel, step exit.

A bar that can be left spinning is worse than no bar, so the terminals are
tested one by one rather than "the happy path works".

Qt-free: the controller is built with ``__new__`` and a ``SimpleNamespace``
widget, and ``_SourceLoadWorker`` / ``QThread`` are replaced with recording fakes
(the same pattern as ``test_process_whole_volume.py``).  No napari, no
QApplication.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_epicycle.data_preparation import controller


class _Bar:
    """Records every ``_busy_show(label)`` / ``_busy_hide()`` in order."""

    def __init__(self):
        self.events = []

    def show(self, label):
        self.events.append(("show", label))

    def hide(self, *_a, **_k):
        self.events.append(("hide", None))

    @property
    def shown_labels(self):
        return [lbl for kind, lbl in self.events if kind == "show"]

    @property
    def bar_events(self):
        return [(k, v) for k, v in self.events if k in ("show", "hide")]

    @property
    def n_hides(self):
        return sum(1 for kind, _ in self.events if kind == "hide")

    @property
    def is_visible(self):
        for kind, _ in reversed(self.events):
            if kind in ("show", "hide"):
                return kind == "show"
        return False


class _Item:
    def __init__(self, path):
        self._path = path

    def data(self, _role):
        return self._path


class _FakeThread:
    """Records its own `start()` INTO the bar's event list, so the ordering claim
    ("the bar is up BEFORE the read starts") is pinned rather than asserted from
    two independent booleans that could both be true in either order."""

    bar = None          # set by `_ctl`

    def __init__(self):
        self.started = SimpleNamespace(connect=lambda *_a: None)
        self.finished = SimpleNamespace(connect=lambda *_a: None)
        self.did_start = False

    def quit(self):
        pass

    def start(self):
        self.did_start = True
        if _FakeThread.bar is not None:
            _FakeThread.bar.events.append(("thread-start", None))


class _FakeWorker:
    def __init__(self, path, use_memmap):
        self.path = path
        self.use_memmap = use_memmap
        self.cancel_requested = False
        for sig in ("finished", "error", "cancelled"):
            setattr(self, sig, SimpleNamespace(connect=lambda *_a: None))

    def moveToThread(self, _t):
        pass

    def run(self):
        pass

    def request_cancel(self):
        self.cancel_requested = True


def _ctl(tmp_path, monkeypatch, bar):
    src = tmp_path / "wound.tif"
    src.write_bytes(b"\0")

    ctl = controller.DataPreparationController.__new__(
        controller.DataPreparationController)
    ctl._log = lambda *a, **k: None
    ctl.widget = SimpleNamespace(
        _busy_show=bar.show,
        _busy_hide=bar.hide,
        source_file_list=SimpleNamespace(currentItem=lambda: _Item(str(src))),
        display_name_edit=SimpleNamespace(
            text=lambda: "wound", setText=lambda *_a: None),
        source_status_label=SimpleNamespace(
            setText=lambda *_a: None, setStyleSheet=lambda *_a: None),
    )
    ctl._flush_source_state_save = lambda: None
    ctl._clear_shapes_layer = lambda: None
    ctl._loaded_source_layers = []
    # Brick 4a added the `allow_missing_source` / `reasons` keywords.
    ctl._fresh_prepared_store_for = lambda _p, **_kw: None
    ctl._preflight_source_route = lambda _p: (True, False)
    ctl._set_step1_busy = lambda _v: None
    ctl._set_load_button_cancel = lambda _v: None
    ctl._source_load_token = 0
    ctl._source_loading = False
    ctl._source_load_thread = None
    ctl._source_load_worker = None
    ctl._draining_source_threads = []
    ctl._cleanup_source_load_thread = lambda _t: None

    monkeypatch.setattr(controller, "_SourceLoadWorker", _FakeWorker)
    monkeypatch.setattr(controller, "QThread", _FakeThread)
    monkeypatch.setattr(_FakeThread, "bar", bar)
    return ctl, str(src)


# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #

def test_bar_is_shown_naming_the_file_before_the_read_starts(tmp_path, monkeypatch):
    bar = _Bar()
    ctl, _src = _ctl(tmp_path, monkeypatch, bar)
    ctl.load_source()
    assert ctl._source_loading is True
    assert ctl._source_load_thread.did_start
    # ORDERING, not just presence: the show must precede the thread start, or the
    # user stares at a frozen surface for the length of the read.
    assert bar.events == [
        ("show", "Reading wound.tif…"),
        ("thread-start", None),
    ], bar.events
    assert bar.is_visible, "the bar must still be up while the read runs"


def test_bar_is_not_shown_when_the_preflight_declines(tmp_path, monkeypatch):
    """A declined >RAM confirm must bail with the bar UNTOUCHED - no launch."""
    bar = _Bar()
    ctl, _src = _ctl(tmp_path, monkeypatch, bar)
    ctl._preflight_source_route = lambda _p: (False, False)
    ctl.load_source()
    assert bar.events == []          # no show, no hide, and no thread start
    assert ctl._source_loading is False


def test_launch_failure_still_hides_the_bar(tmp_path, monkeypatch):
    """If worker/thread construction raises, neither finished nor error fires -
    the launch guard is the terminal and must put the bar back."""
    bar = _Bar()
    ctl, _src = _ctl(tmp_path, monkeypatch, bar)

    def _boom(*_a, **_k):
        raise RuntimeError("no thread for you")

    monkeypatch.setattr(controller, "_SourceLoadWorker", _boom)
    ctl.load_source()
    assert bar.shown_labels and bar.n_hides == 1
    assert not bar.is_visible
    assert ctl._source_loading is False


# --------------------------------------------------------------------------- #
# terminals - error / cancel / stale / step exit
# --------------------------------------------------------------------------- #

def test_error_terminal_hides_the_bar(tmp_path, monkeypatch):
    bar = _Bar()
    ctl, _src = _ctl(tmp_path, monkeypatch, bar)
    ctl.load_source()
    ctl._source_lazy = object()
    ctl._source_axes_resolved = None
    ctl._source_disk_axes = None
    ctl._channel_index_by_name = {}
    ctl._on_source_load_error("bad tiff", ctl._source_load_token)
    assert not bar.is_visible
    assert ctl._source_loading is False


def test_cancel_terminal_hides_the_bar_and_asks_the_worker_to_stop(
        tmp_path, monkeypatch):
    bar = _Bar()
    ctl, _src = _ctl(tmp_path, monkeypatch, bar)
    ctl.load_source()
    worker = ctl._source_load_worker
    ctl.cancel_source_load()
    assert not bar.is_visible
    assert ctl._source_loading is False
    assert worker.cancel_requested is True


def test_stale_error_does_not_touch_the_bar(tmp_path, monkeypatch):
    """A superseded read's error must NOT hide a bar that now belongs to a NEWER
    load - the stale guard owns that, and losing it would blank a live bar."""
    bar = _Bar()
    ctl, _src = _ctl(tmp_path, monkeypatch, bar)
    ctl.load_source()
    stale = ctl._source_load_token
    ctl.cancel_source_load()          # bumps the token, hides once
    ctl.load_source()                 # a NEW read, bar up again
    hides_before = bar.n_hides
    ctl._on_source_load_error("late failure", stale)
    assert bar.n_hides == hides_before
    assert bar.is_visible


def test_on_step_exit_cancels_an_in_flight_read(tmp_path, monkeypatch):
    """Leaving Step 1 mid-read must reach the cancel terminal, so the bar cannot
    be left spinning on another workflow step."""
    bar = _Bar()
    ctl, _src = _ctl(tmp_path, monkeypatch, bar)
    ctl.load_source()
    assert bar.is_visible

    called = []
    real_cancel = ctl.cancel_source_load

    def _spy():
        called.append(1)
        real_cancel()

    ctl.cancel_source_load = _spy
    ctl._left_step1 = False
    ctl._set_review_busy = lambda _v: None
    ctl.viewer = SimpleNamespace(scale_bar=SimpleNamespace(visible=True))
    ctl._extract_running = False
    ctl._largedata_thread = None
    ctl._extract_thread = None
    ctl._trim_thread = None
    ctl._trim_running = False
    try:
        ctl.on_step_exit()
    except Exception:
        # The rest of `on_step_exit` needs far more state than this fake has;
        # we only pin that the source-read cancel runs, and runs FIRST.
        pass
    assert called == [1]
    assert not bar.is_visible


# --------------------------------------------------------------------------- #
# unload - the [STEP1-PERF] probe (env-gated), see the commit message
# --------------------------------------------------------------------------- #

def test_unload_perf_probe_is_silent_unless_the_env_flag_is_set(
        tmp_path, monkeypatch):
    logs = []
    ctl, _src = _ctl(tmp_path, monkeypatch, _Bar())
    ctl._log = lambda prefix, msg: logs.append((prefix, msg))
    ctl.box_tool = None
    ctl._clear_patch_locations = lambda: None
    ctl._teardown_preview = lambda: None
    ctl.viewer = SimpleNamespace(layers=[])
    ctl._on_layers_changed = lambda: None
    ctl.widget.hide_3d_detail = None
    ctl.widget.hide_zoom_depth = None

    monkeypatch.delenv("EPICYCLE_SYNC_DEBUG", raising=False)
    ctl.unload_source()
    assert not [m for p, m in logs if p == "STEP1-PERF"]

    monkeypatch.setenv("EPICYCLE_SYNC_DEBUG", "1")
    ctl._loaded_source_layers = []
    ctl.unload_source()
    perf = [m for p, m in logs if p == "STEP1-PERF"]
    assert len(perf) == 1 and perf[0].startswith("unload ")
    assert "ms" in perf[0] and "source layer(s)" in perf[0]


# --------------------------------------------------------------------------- #
# Brick 3c #7 / 3e - unload / step-exit: the "Unloading ..." bar + hide-before-
# remove.  Brick 3c also BATCHED the removal loops; Brick 3e reverted that after
# measuring it live (napari's `end_batch` re-runs the full gc + glFinish +
# scenegraph rebuild per batch, even an empty one: unload of 2 layers
# 511 -> 734 ms, unload of ZERO layers 1 -> 576 ms).  So these fakes still expose
# `batched_update`, deliberately: if anyone re-introduces batching,
# `test_removal_loops_are_not_batched` fails.
# --------------------------------------------------------------------------- #

class _BatchViewer:
    """A viewer fake whose LayerList records `visible` writes and removals IN
    ORDER, so hide-before-remove is checkable.

    It also exposes `batched_update` and records entry/exit - not because we want
    it used, but so that a future re-introduction of batching is caught by
    `test_removal_loops_are_not_batched` instead of silently costing 200+ ms."""

    class _Layers(list):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def remove(self, layer):
            self.owner.log.append(("remove", layer.name))
            list.remove(self, layer)

        def __contains__(self, item):
            if isinstance(item, str):
                return any(getattr(x, "name", None) == item for x in self)
            return list.__contains__(self, item)

        def __getitem__(self, key):
            if isinstance(key, str):
                for x in self:
                    if getattr(x, "name", None) == key:
                        return x
                raise KeyError(key)
            return list.__getitem__(self, key)

        def batched_update(self):
            import contextlib

            @contextlib.contextmanager
            def _cm():
                self.owner.log.append(("batch-enter", None))
                try:
                    yield
                finally:
                    self.owner.log.append(("batch-exit", None))

            return _cm()

    def __init__(self):
        self.log = []
        self.layers = _BatchViewer._Layers(self)


class _L:
    def __init__(self, name, owner):
        self.name = name
        self._owner = owner
        self._visible = True

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, v):
        self._owner.log.append(("visible", (self.name, v)))
        self._visible = v


def _unloadable(tmp_path, monkeypatch, bar, n_layers=2):
    ctl, _src = _ctl(tmp_path, monkeypatch, bar)
    viewer = _BatchViewer()
    names = [f"ch{i}" for i in range(n_layers)]
    for nm in names:
        viewer.layers.append(_L(nm, viewer))
    ctl.viewer = viewer
    ctl._loaded_source_layers = list(names)
    ctl._loaded_source_file = str(tmp_path / "wound.tif")
    ctl._current_multiscale_display_name = None
    ctl.box_tool = None
    ctl._clear_patch_locations = lambda: None
    ctl._teardown_preview = lambda: None
    ctl._on_layers_changed = lambda: None
    ctl.widget.hide_3d_detail = None
    ctl.widget.hide_zoom_depth = None
    return ctl, viewer


def test_unload_hides_each_layer_before_removing_it(tmp_path, monkeypatch):
    """The vispy visual must be detached from the scene graph
    before the layer is dropped, or a freed GPU texture can be drawn."""
    bar = _Bar()
    ctl, viewer = _unloadable(tmp_path, monkeypatch, bar)
    ctl.unload_source()

    for name in ("ch0", "ch1"):
        hid = viewer.log.index(("visible", (name, False)))
        rem = viewer.log.index(("remove", name))
        assert hid < rem, viewer.log
    assert list(viewer.layers) == []


def test_removal_loops_are_not_batched(tmp_path, monkeypatch):
    """REGRESSION GUARD (Brick 3e).  `viewer.layers.batched_update()` reads like
    the obvious win - one canvas rebuild instead of N - and was shipped in
    a49e6ba7.  Measured live on the real viewer it is a LOSS: napari runs the full
    gc.collect + glFinish + scenegraph rebuild at the end of EVERY batch, empty
    ones included, ON TOP of the per-layer path.  Unload of 2 layers 511 ->
    734 ms; unload of ZERO layers 1 -> 576 ms; the Step 1->3 switch 650 ->
    ~1100 ms.  Do not put it back."""
    bar = _Bar()
    ctl, viewer = _unloadable(tmp_path, monkeypatch, bar)
    ctl.unload_source()
    assert not [k for k, _ in viewer.log if k.startswith("batch-")], viewer.log


def test_unload_shows_and_hides_the_bar_naming_the_source(tmp_path, monkeypatch):
    bar = _Bar()
    ctl, _viewer = _unloadable(tmp_path, monkeypatch, bar)
    ctl.unload_source()
    assert bar.shown_labels == ["Unloading wound.tif…"], bar.events
    assert not bar.is_visible, "the bar must come back down"


def test_unload_prefers_the_multiscale_display_name(tmp_path, monkeypatch):
    bar = _Bar()
    ctl, _viewer = _unloadable(tmp_path, monkeypatch, bar)
    ctl._current_multiscale_display_name = "WoundT01"
    ctl.unload_source()
    assert bar.shown_labels == ["Unloading WoundT01…"]


def test_unload_with_no_layers_does_not_blink_the_bar(tmp_path, monkeypatch):
    bar = _Bar()
    ctl, _viewer = _unloadable(tmp_path, monkeypatch, bar, n_layers=0)
    ctl.unload_source()
    assert bar.events == [], "an empty unload is sub-ms - no bar"


def test_a_raising_teardown_still_hides_the_bar(tmp_path, monkeypatch):
    """The bar is hidden in a `finally`, so a teardown failure cannot leave the
    surface locked behind a spinning bar."""
    bar = _Bar()
    ctl, _viewer = _unloadable(tmp_path, monkeypatch, bar)

    def _boom():
        raise RuntimeError("teardown exploded")

    ctl._teardown_preview = _boom
    with pytest.raises(RuntimeError):
        ctl.unload_source()
    assert bar.shown_labels == ["Unloading wound.tif…"]
    assert not bar.is_visible


def test_step_exit_source_removal_is_not_batched(tmp_path, monkeypatch):
    """The step-exit path (`_remove_source_layers_from_viewer`) keeps the channel
    refs, still hides before removing, and is likewise NOT batched (Brick 3e:
    the step-exit removal measured 273 ms and the whole Step 1->3 switch went
    650 -> ~1100 ms with batching on)."""
    bar = _Bar()
    ctl, viewer = _unloadable(tmp_path, monkeypatch, bar)
    ctl.widget.image_layer_combo = None
    ctl._saved_channel_selection = None
    ctl._swapping_source_display = False
    ctl._locked_3d_level = None
    ctl._source_layers_on_canvas = True
    ctl._progressive_active = False
    try:
        ctl._remove_source_layers_from_viewer()
    except AttributeError:
        pass          # the tail of the method needs more widget state than this
                      # fake has; the removal loop is what this test is about
    assert not [k for k, _ in viewer.log if k.startswith("batch-")], viewer.log
    assert ("remove", "ch0") in viewer.log and ("remove", "ch1") in viewer.log
    for name in ("ch0", "ch1"):
        assert (viewer.log.index(("visible", (name, False)))
                < viewer.log.index(("remove", name))), viewer.log
