"""Best-effort render-state probe for the Step 3 isolation paths (diagnostic).

Prints ONE line per call, only when ``EPICYCLE_SYNC_DEBUG=1``.  Reads napari
internals defensively (``_slice``, ``layer_to_visual``) so it can never raise
into the caller.  Used to see, at the exact moments a lasso/isolation hides,
recolours and re-shows a Labels layer, whether the model, the slice texture
and the vispy node agree — the napari-0.8 "placeholder slice while hidden"
mechanism
and the shared-colormap desync (B-1) show up here.
"""
from __future__ import annotations

import os
import time

_ENABLED = os.environ.get("EPICYCLE_SYNC_DEBUG") == "1"
_T0 = time.perf_counter()


def enabled() -> bool:
    return _ENABLED


def render_state(tag: str, layer, viewer=None, qt_viewer=None) -> None:
    """Print ``[RENDER-DIAG] <tag> <layer> ...`` (no-op unless enabled)."""
    if not _ENABLED or layer is None:
        return
    try:
        parts = [f"[RENDER-DIAG] t={time.perf_counter() - _T0:8.3f}s {tag:28s} {getattr(layer, 'name', '?')!r:44s}"]
        parts.append(f"vis={getattr(layer, 'visible', '?')}")
        # vispy node visibility (the thing actually drawn)
        node_vis = "?"
        try:
            qtv = qt_viewer
            if qtv is None and viewer is not None:
                qtv = viewer.window._qt_viewer
            if qtv is not None:
                vl = qtv.canvas.layer_to_visual[layer]
                node_vis = vl.node.visible
                try:
                    _vd = getattr(vl, "_data", None)
                    _tex = getattr(vl.node, "_texture", None)
                    node_vis = (f"{node_vis} vl._data={getattr(_vd, 'shape', None)}@{id(_vd) % 100000:05d} "
                                f"tex={getattr(_tex, 'shape', None)} node@{id(vl.node) % 100000:05d} cmapnode={type(getattr(vl.node, 'cmap', None)).__name__}")
                except Exception as _e:
                    node_vis = f"{node_vis} (+{_e!r})"
        except Exception:
            node_vis = "n/a"
        parts.append(f"node={node_vis}")
        # slice / texture state
        try:
            sl = layer._slice
            img = getattr(sl, "image", None)
            view = getattr(img, "view", None)
            raw = getattr(img, "raw", None)
            parts.append(
                f"slice.empty={getattr(sl, 'empty', '?')} view={getattr(view, 'dtype', '?')}"
                f"{getattr(view, 'shape', '')} raw={getattr(raw, 'dtype', '?')}"
            )
        except Exception:
            parts.append("slice=n/a")
        # colormap identity + selection flags
        try:
            cm = layer.colormap
            cd = getattr(cm, "color_dict", None)
            n = len(cd) if cd is not None else -1
            parts.append(
                f"cmap={type(cm).__name__}@{id(cm) % 100000:05d} n={n} "
                f"use_sel={getattr(cm, 'use_selection', '?')} sel={getattr(cm, 'selection', '?')}"
            )
        except Exception:
            parts.append("cmap=n/a")
        parts.append(
            f"ssl={getattr(layer, 'show_selected_label', '?')} selected={getattr(layer, 'selected_label', '?')}"
        )
        if viewer is not None:
            try:
                parts.append(f"ndisplay={viewer.dims.ndisplay} step0={viewer.dims.current_step[0]}")
            except Exception:
                pass
        print("  ".join(parts), flush=True)
    except Exception:
        pass
