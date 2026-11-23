"""One garbage collection per batch of layer removals.

napari collects garbage after every layer it removes; the batch skips those
and runs a single real collection when the outermost batch closes.  Qt-free:
napari's canvas module is replaced by a fake with a counting ``gc``.
"""
from __future__ import annotations

import types

import pytest

from packages.epicycle_annotator import layer_batch as lb


class _FakeGc:
    def __init__(self):
        self.calls = 0

    def collect(self, *a):
        self.calls += 1
        return 0


@pytest.fixture
def canvas(monkeypatch):
    mod = types.SimpleNamespace(gc=_FakeGc())
    original = mod.gc
    monkeypatch.setattr(lb, "_get_canvas_module", lambda: mod)
    real = _FakeGc()
    monkeypatch.setattr(lb, "_real_gc", real)
    mod.original = original
    mod.real = real
    return mod


def _remove_layers(mod, n):
    for _ in range(n):
        mod.gc.collect()      # what napari does after each removal


def test_eight_removals_make_one_real_collect(canvas, capsys):
    with lb.batched_layer_removals():
        _remove_layers(canvas, 8)
        assert canvas.real.calls == 0
        assert canvas.original.calls == 0
    assert canvas.real.calls == 1
    assert canvas.gc is canvas.original
    assert capsys.readouterr().out.count("[GC-PERF]") == 1


def test_nesting_collects_once_at_the_outermost_exit(canvas):
    with lb.batched_layer_removals():
        with lb.batched_layer_removals():
            with lb.batched_layer_removals():
                _remove_layers(canvas, 3)
            assert canvas.real.calls == 0
            assert canvas.gc is not canvas.original
            _remove_layers(canvas, 2)
        assert canvas.real.calls == 0
        assert lb.depth() == 1
    assert lb.depth() == 0
    assert canvas.real.calls == 1
    assert canvas.gc is canvas.original


def test_exception_restores_gc_and_still_collects_once(canvas):
    with pytest.raises(RuntimeError):
        with lb.batched_layer_removals():
            _remove_layers(canvas, 2)
            raise RuntimeError("boom")
    assert canvas.gc is canvas.original
    assert canvas.real.calls == 1
    assert lb.depth() == 0


def test_empty_batch_runs_no_collect(canvas, capsys):
    with lb.batched_layer_removals():
        pass
    assert canvas.real.calls == 0
    assert "[GC-PERF]" not in capsys.readouterr().out
    assert canvas.gc is canvas.original


def test_a_second_batch_after_the_first_works(canvas):
    for _ in range(2):
        with lb.batched_layer_removals():
            _remove_layers(canvas, 4)
    assert canvas.real.calls == 2


def test_no_napari_is_a_no_op(monkeypatch):
    monkeypatch.setattr(lb, "_get_canvas_module", lambda: None)
    with lb.batched_layer_removals():
        pass


def test_finish_is_called_on_the_passed_canvases(canvas):
    done = []
    ctx = types.SimpleNamespace(finish=lambda: done.append(1))
    qv = types.SimpleNamespace(
        canvas=types.SimpleNamespace(
            native=types.SimpleNamespace(objectName=lambda: ""),
            _scene_canvas=types.SimpleNamespace(context=ctx)))
    with lb.batched_layer_removals(qv, object()):
        _remove_layers(canvas, 1)
    assert done == [1]


def test_napari_still_collects_through_its_own_gc_name():
    """If napari moves the collect, the batch would silently do nothing."""
    canvas_mod = pytest.importorskip("napari._vispy.canvas")
    assert hasattr(canvas_mod, "gc")
    names = canvas_mod.VispyCanvas._clean_and_update_scenegraph.__code__.co_names
    assert "gc" in names


def test_closed_side_canvas_is_not_finished():
    calls = []
    ctx = types.SimpleNamespace(finish=lambda: calls.append(1))
    for closed, expected in ((False, 1), (True, 0)):
        calls.clear()
        qv = types.SimpleNamespace(canvas=types.SimpleNamespace(
            native=types.SimpleNamespace(objectName=lambda: ""),
            _scene_canvas=types.SimpleNamespace(context=ctx, _closed=closed)))
        lb._finish_canvas(qv)
        assert len(calls) == expected


def test_deleted_canvas_widget_is_not_finished():
    """A deleted Qt wrapper raises on any call; the flush must skip it."""
    calls = []
    ctx = types.SimpleNamespace(finish=lambda: calls.append(1))

    def _dead():
        raise RuntimeError("wrapped C/C++ object has been deleted")

    qv = types.SimpleNamespace(canvas=types.SimpleNamespace(
        native=types.SimpleNamespace(objectName=_dead),
        _scene_canvas=types.SimpleNamespace(context=ctx)))
    lb._finish_canvas(qv)
    assert calls == []


def test_live_canvas_widget_is_finished():
    calls = []
    ctx = types.SimpleNamespace(finish=lambda: calls.append(1))
    qv = types.SimpleNamespace(canvas=types.SimpleNamespace(
        native=types.SimpleNamespace(objectName=lambda: ""),
        _scene_canvas=types.SimpleNamespace(context=ctx)))
    lb._finish_canvas(qv)
    assert calls == [1]


def test_qt_viewers_of_returns_main_and_side_viewers():
    main = object()
    xy, xz, yz = object(), object(), object()
    w = types.SimpleNamespace(
        viewer=types.SimpleNamespace(window=types.SimpleNamespace(_qt_viewer=main)),
        qt_viewer_xy=xy, qt_viewer_xz=xz, qt_viewer_yz=yz)
    assert lb.qt_viewers_of(w) == [main, xy, xz, yz]


def test_qt_viewers_of_skips_missing_attributes():
    assert lb.qt_viewers_of(object()) == []


def test_real_napari_removal_inside_the_batch_is_intercepted(capsys):
    """One layer removed in a batch must hit the swapped collector once."""
    pytest.importorskip("napari")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    import napari
    try:
        viewer = napari.Viewer(show=False)
    except Exception as exc:  # no display / GL on this machine
        pytest.skip("napari viewer cannot be created: %r" % (exc,))
    try:
        import napari._vispy.canvas as canvas_mod
        original = canvas_mod.gc
        layer = viewer.add_labels(np.zeros((4, 4, 4), dtype=np.uint16))
        with lb.batched_layer_removals(viewer.window._qt_viewer):
            viewer.layers.remove(layer)
            assert canvas_mod.gc is not original
        assert canvas_mod.gc is original
        assert "[GC-PERF] batch n_skipped=1 " in capsys.readouterr().out
    finally:
        viewer.close()
