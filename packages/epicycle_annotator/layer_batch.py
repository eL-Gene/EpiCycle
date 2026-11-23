"""Batch the garbage collection that napari runs after every layer removal.

napari runs one ``gc.collect()`` after each layer it removes from a viewer
that has a canvas (a safeguard against a Windows OpenGL access violation:
dangling GL objects must be collected before the scene graph is rebuilt).
The annotator has four canvases (main plus three side views), so removing one
layer can cost four full collections inside one call.

``batched_layer_removals`` keeps the safeguard and drops the per-layer cost:
while a batch is open the collections are skipped (counted), and when the
outermost batch closes exactly one real collection runs, followed by a GL
``finish()`` on the canvases that were passed in.  napari's own per-removal
``finish()`` and scene-graph update are not touched.

Rules for code inside a batch:

* Main thread only.
* No ``processEvents`` and no modal ``exec`` inside the batch.  A draw that
  happens between the removals and the closing collection could see
  half-freed GL objects.
* An empty batch (no removal) costs nothing: no collection is run.
"""
from __future__ import annotations

import contextlib
import gc as _real_gc
import time
from types import SimpleNamespace

_depth = 0
_saved_gc = None
_skipped = 0
_t_enter = 0.0


def depth() -> int:
    """Current nesting depth (0 when no batch is open)."""
    return _depth


def _get_canvas_module():
    """napari's canvas module, or None when napari cannot be imported."""
    try:
        import napari._vispy.canvas as canvas_mod
        return canvas_mod
    except Exception:
        return None


def _skipped_collect(*_args, **_kwargs) -> int:
    global _skipped
    _skipped += 1
    return 0


def qt_viewers_of(widget) -> list:
    """The main and side-view Qt viewers of an annotator (missing ones skipped)."""
    out = []
    try:
        out.append(widget.viewer.window._qt_viewer)
    except Exception:
        pass
    for name in ("qt_viewer_xy", "qt_viewer_xz", "qt_viewer_yz"):
        qv = getattr(widget, name, None)
        if qv is not None:
            out.append(qv)
    return out


def _finish_canvas(qt_viewer) -> None:
    try:
        # A deleted Qt wrapper raises here; a closed canvas must not be flushed.
        qt_viewer.canvas.native.objectName()
        scene = qt_viewer.canvas._scene_canvas
        if getattr(scene, "_closed", False):
            return
        scene.context.finish()
    except Exception:
        pass


@contextlib.contextmanager
def batched_layer_removals(*qt_viewers):
    """Skip napari's per-removal collection; run one when the batch closes."""
    global _depth, _saved_gc, _skipped, _t_enter
    canvas_mod = _get_canvas_module()
    if canvas_mod is None or not hasattr(canvas_mod, "gc"):
        yield
        return
    if _depth == 0:
        _saved_gc = canvas_mod.gc
        _skipped = 0
        _t_enter = time.perf_counter()
        canvas_mod.gc = SimpleNamespace(collect=_skipped_collect)
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if _depth == 0:
            try:
                canvas_mod.gc = _saved_gc
            finally:
                _saved_gc = None
                n_skipped = _skipped
                _skipped = 0
                if n_skipped > 0:
                    t0 = time.perf_counter()
                    _real_gc.collect()
                    collect_ms = (time.perf_counter() - t0) * 1000.0
                    for qv in qt_viewers:
                        _finish_canvas(qv)
                    wall_ms = (time.perf_counter() - _t_enter) * 1000.0
                    print(
                        "[GC-PERF] batch n_skipped=%d collect_ms=%.0f wall_ms=%.0f"
                        % (n_skipped, collect_ms, wall_ms)
                    )
