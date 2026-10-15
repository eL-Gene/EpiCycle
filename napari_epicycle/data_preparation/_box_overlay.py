"""Crash-safe vispy ``Line`` overlay for the Step 1 ③ box tool (v2).

Draws the ③ ROI boxes as raw vispy ``Line`` visuals parented directly to the
shared main canvas scene — the SAME lifecycle idiom the de-novo Add tool uses
(``packages/epicycle_annotator/box_overlay.py``) — deliberately **not** a napari
Shapes layer.

Why: a Shapes-in-3D layer renders its wireframe as a Compound Mesh whose vertex
buffer can be freed while a ``paintGL`` is still queued on patch-switch /
step-exit / teardown, faulting the 3D canvas with a ``glDrawArrays`` access
violation (``0x1C`` = a null / freed VBO base).  A raw vispy ``Line`` owns its
own lifecycle and never enters ``viewer.layers``, so it can neither be missed by
a layer sweep nor orphan into another step.  The access-violation class is gone
*by construction*.

Three visuals (width is per-visual in vispy, so a thicker "active" box needs its
own ``Line``):

* ``idle``   — every non-active box, teal.
* ``active`` — the selected box, amber + thicker.
* ``ghost``  — a faint preview of where the NEXT box will land (feedback #3).

**Coordinate frame (physical-µm, v3):** Step 1 boxes live in **full-res voxel**
coords (raw ``_full_shape`` grid), but the shared canvas scene is now napari
PHYSICAL world space: every display layer sits at ``scale = factor × voxel_size``
(``factor == 1`` for routine full-res data, ``factor > 1`` for a large-source
overview, ``voxel_size`` the anisotropic ``(vz, vy, vx)`` — ``(1, 1, 1)`` when
unknown).  So world µm ``== full_res_voxel × voxel_size`` in every case.  The box
therefore multiplies its full-res-voxel coords by ``voxel_size`` ONLY (never
additionally by the overview ``factor`` — the layer's own ``scale`` already
carries ``factor``), then applies the ZYX→XYZ axis reversal.  Set the voxel via
:meth:`Step1BoxOverlay.set_voxel_size`; unknown ``(1, 1, 1)`` reproduces the old
voxel-index geometry byte-for-byte.

Everything is guarded + idempotent — nothing here may raise out of a Qt slot.
"""

from __future__ import annotations

import numpy as np

from packages.epicycle_annotator.roi_box import box_corners, box_edges

# Box outline colours (RGBA 0..1), matching the ③ accent palette.
_IDLE_COLOR = (0.357, 0.643, 0.812, 0.9)   # teal  (_STEP1_ACCENT)
_ACTIVE_COLOR = (0.831, 0.659, 0.263, 1.0)  # amber (#d4a843)
_GHOST_COLOR = (0.78, 0.82, 0.86, 0.35)     # faint grey-blue preview
_IDLE_WIDTH = 2.0
_ACTIVE_WIDTH = 3.5
_GHOST_WIDTH = 1.5


# ---------------------------------------------------------------------------
# Pure geometry (Qt/vispy-free — unit-tested)
# ---------------------------------------------------------------------------

def _box_segments_zyx(box, shape, style: str = "corners") -> np.ndarray:
    """A single box's wireframe as a ``(2N, 3)`` ZYX segment-endpoint array.

    ``corners`` gives 24 bracket segments (48 points); ``wireframe`` gives the
    12 cube edges (24 points).  Suitable for a vispy ``Line`` with
    ``connect="segments"`` (consecutive point pairs form a segment).
    """
    b = box.bounds_dict(shape)
    segs = box_corners(b) if style == "corners" else box_edges(b)
    if not segs:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate(segs, axis=0)


def boxes_segments_xyz(boxes, shape, style: str = "corners",
                       dims_order=None, voxel_size=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Concatenate every box's wireframe into one ``(M, 3)`` scene-column array.

    The box geometry is full-res ZYX voxel coords.  Two transforms are applied,
    in this order:

    1. **Physical scale** — multiply the ZYX coords by ``voxel_size``
       ``(vz, vy, vx)`` so the box reaches the same PHYSICAL-µm world the display
       layers now sit in (``world == full_res_voxel × voxel_size``).  The
       multiply is done BEFORE the column permutation on purpose: it mirrors
       napari's own ``world = data × scale`` → ``scene[order[::-1]]`` exactly, so
       an anisotropic-Z box stays glued in a rolled XZ/YZ view (a post-permute
       multiply or a fixed STTransform scale would NOT follow the permutation).
       ``voxel_size == (1, 1, 1)`` (unknown) → byte-identical to the old
       voxel-index geometry.
    2. **Displayed-axis permutation** — napari maps a layer into the shared vispy
       scene using its *displayed* world axes reversed (``dims.order[::-1]``), so
       the overlay applies the SAME per-axis permutation to stay glued in ANY
       view — a rolled XZ/YZ 2D view or a 2D↔3D flip (quirk 2, the transpose
       fix).  ``dims_order`` is the current ``tuple(viewer.dims.order)``; when it
       is ``None`` or not a length-3 permutation we fall back to the plain
       ZYX→XYZ reversal — and the default ``order == (0, 1, 2)`` yields exactly
       that, so the default view is byte-identical to before.

    Empty / ``None`` inputs → an empty ``(0, 3)``.
    """
    if shape is None or not boxes:
        return np.zeros((0, 3), dtype=float)
    parts = [_box_segments_zyx(box, shape, style) for box in boxes]
    parts = [p for p in parts if p.size]
    if not parts:
        return np.zeros((0, 3), dtype=float)
    zyx = np.concatenate(parts, axis=0)
    # (1) Physical-µm scale: multiply the ZYX voxel coords by the anisotropic
    # voxel BEFORE the permute (see the docstring).  Guarded — never raises out
    # of the Qt-slot callers; a non-3 / bad voxel is simply ignored.
    try:
        vs = np.asarray(voxel_size, dtype=float).reshape(-1)
        if vs.shape == (3,) and not np.allclose(vs, 1.0):
            zyx = zyx * vs
    except Exception:
        pass
    n = zyx.shape[1]  # 3 (Z, Y, X)
    if (dims_order is not None and len(dims_order) == n
            and sorted(int(a) for a in dims_order) == list(range(n))):
        cols = list(int(a) for a in dims_order)[::-1]   # napari displayed order
    else:
        cols = list(range(n))[::-1]                      # plain ZYX -> XYZ
    return zyx[:, cols].astype(float)


def boxes_segments_and_colors_xyz(boxes, shape, styles, default_color,
                                  dims_order=None, voxel_size=(1.0, 1.0, 1.0)):
    """Per-box variant of :func:`boxes_segments_xyz` for the ④ location overlay.

    Each box uses ITS OWN style's ``shape`` for the segments and ``color`` +
    ``opacity`` for a per-VERTEX ``(N, 4)`` RGBA array — so mixed colours + shapes
    render on the SINGLE idle Line (P2 grill: no per-box Line → no freed-VBO
    ``glDrawArrays 0x1C`` crash; width stays GLOBAL, applied by the caller).
    ``styles[i]`` is a ``{color, opacity, shape}`` dict or ``None`` (→
    ``default_color`` + ``"corners"``); ``color`` is an ``[r,g,b,a]`` or the
    ``"status"`` sentinel (→ ``default_color``).  The voxel scale + the
    displayed-axis permutation match :func:`boxes_segments_xyz` exactly.  Returns
    ``(segments_xyz (M,3), colors (M,4))``; empty inputs → ``((0,3),(0,4))``."""
    if shape is None or not boxes:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 4), dtype=float)
    seg_parts, col_parts = [], []
    for i, box in enumerate(boxes):
        st = (styles[i] if styles and i < len(styles)
              and isinstance(styles[i], dict) else {})
        shp = st.get("shape") if st.get("shape") in ("corners", "wireframe") \
            else "corners"
        seg = _box_segments_zyx(box, shape, shp)
        if not seg.size:
            continue
        c = st.get("color")
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            try:
                rgba = [float(c[0]), float(c[1]), float(c[2]),
                        float(c[3]) if len(c) > 3 else 1.0]
            except (TypeError, ValueError):
                rgba = list(default_color)
        else:
            rgba = list(default_color)   # None / "status" → the default colour
        op = st.get("opacity")
        if op is not None:
            try:
                rgba[3] = float(rgba[3]) * min(1.0, max(0.0, float(op)))
            except (TypeError, ValueError):
                pass
        seg_parts.append(seg)
        col_parts.append(np.tile(rgba, (seg.shape[0], 1)))
    if not seg_parts:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 4), dtype=float)
    zyx = np.concatenate(seg_parts, axis=0)
    colors = np.concatenate(col_parts, axis=0).astype(float)
    try:
        vs = np.asarray(voxel_size, dtype=float).reshape(-1)
        if vs.shape == (3,) and not np.allclose(vs, 1.0):
            zyx = zyx * vs
    except Exception:
        pass
    n = zyx.shape[1]
    if (dims_order is not None and len(dims_order) == n
            and sorted(int(a) for a in dims_order) == list(range(n))):
        cols = list(int(a) for a in dims_order)[::-1]
    else:
        cols = list(range(n))[::-1]
    return zyx[:, cols].astype(float), colors


# ---------------------------------------------------------------------------
# The canvas-attached overlay
# ---------------------------------------------------------------------------

class Step1BoxOverlay:
    """Idle / active / ghost ``Line`` visuals on the shared main canvas.

    Construct with the main ``QtViewer`` (may be ``None`` — the overlay then
    degenerates to a safe no-op).  Call :meth:`update` on every box change,
    :meth:`update_ghost` / :meth:`clear_ghost` for the placement preview, and
    :meth:`destroy` on teardown.
    """

    def __init__(self, main_qt_viewer, style: str = "corners"):
        self._style = style
        self._lines: dict = {}
        self._native = None
        # Live appearance overrides driven by ④'s Render/style controls (via
        # `set_style`).  Defaults reproduce the built-in look, so an overlay
        # whose style is never touched renders exactly as before.
        self._idle_width = _IDLE_WIDTH
        self._active_width = _ACTIVE_WIDTH
        self._alpha = 1.0
        self._color_override = None   # None => status colours; else (r, g, b)
        # Physical-µm scale (v3): the anisotropic voxel the geometry is
        # multiplied by so the box reaches the display's physical world.  Defined
        # even in the no-op (no-viewer) branch so `destroy()` + `set_voxel_size`
        # are always safe.  Unknown (1, 1, 1) reproduces the old voxel geometry.
        self._voxel_size = (1.0, 1.0, 1.0)
        # dims-follow (quirk 2): cache the last geometry so a view roll / 2D<->3D
        # flip can re-permute it, and (below) subscribe to dims.order so the box
        # tracks napari's orientation.  Defined even in the no-op branch so
        # `destroy()` is always safe.
        self._viewer = None
        self._dims_cb = None
        self._last_update = None   # (boxes, active_index, shape, styles)
        self._last_ghost = None    # (box, shape)
        if main_qt_viewer is None:
            return
        try:
            scene = main_qt_viewer.canvas.view.scene
        except Exception:
            return
        try:
            from vispy.scene.visuals import Line
        except Exception:
            return
        for key, color, width in (
            ("idle", _IDLE_COLOR, _IDLE_WIDTH),
            ("active", _ACTIVE_COLOR, _ACTIVE_WIDTH),
            ("ghost", _GHOST_COLOR, _GHOST_WIDTH),
        ):
            try:
                line = Line(
                    pos=np.zeros((2, 3), dtype=float),
                    color=color,
                    width=width,
                    connect="segments",
                    parent=scene,
                )
                line.order = 1e6  # draw on top of the volume / labels
                line.visible = False
                self._lines[key] = line
            except Exception:
                continue
        try:
            self._native = main_qt_viewer.canvas.native
        except Exception:
            self._native = None
        # Follow napari view-orientation changes (roll to XZ/YZ, 2D<->3D flip):
        # both mutate `dims.order`, firing `dims.events.order`.  Camera events are
        # unreliable (a roll may not move the camera), so subscribe to dims.
        self._viewer = getattr(main_qt_viewer, "viewer", None)
        try:
            dims = self._viewer.dims
            dims.events.order.connect(self._on_dims_changed)
            dims.events.ndisplay.connect(self._on_dims_changed)
            self._dims_cb = self._on_dims_changed
        except Exception:
            self._dims_cb = None

    # -- live appearance (④ Render/style) -----------------------------------

    def set_style(self, *, thickness=None, opacity=None, color="__keep__") -> None:
        """Store the appearance overrides ``update`` applies to the idle/active
        boxes.  ``thickness`` (a line width) scales both the idle and the
        (kept-thicker) active outline; ``opacity`` scales the outline alpha;
        ``color`` is ``None`` (status colours), or an ``(r, g, b)`` applied to
        every box, or the sentinel ``"__keep__"`` to leave the colour unchanged.
        Guarded — never raises; the caller re-emits geometry to make it visible."""
        if thickness is not None:
            try:
                t = float(thickness)
                self._idle_width = max(0.5, t)
                self._active_width = max(0.5, t + 1.5)
            except (TypeError, ValueError):
                pass
        if opacity is not None:
            try:
                self._alpha = min(1.0, max(0.05, float(opacity)))
            except (TypeError, ValueError):
                pass
        if color != "__keep__":
            self._color_override = color

    def set_voxel_size(self, voxel_size) -> None:
        """Store the anisotropic ``(vz, vy, vx)`` the geometry is multiplied by so
        the box reaches the display's PHYSICAL-µm world.  Coerces to a 3-tuple of
        positive floats; a bad / non-3 input is ignored (falls back to the last
        good value, defaulting to ``(1, 1, 1)`` = unknown = voxel-index geometry).
        Guarded — NEVER raises; the caller re-emits geometry to make it visible."""
        try:
            vs = tuple(float(v) for v in voxel_size)
        except (TypeError, ValueError):
            return
        if len(vs) == 3 and all(v > 0 for v in vs):
            self._voxel_size = vs

    def _effective_color(self, key: str):
        """The (r, g, b, a) the *key* line draws, folding in colour + opacity."""
        base = _IDLE_COLOR if key == "idle" else _ACTIVE_COLOR
        rgb = self._color_override if self._color_override is not None else base[:3]
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]),
                float(self._alpha * base[3]))

    # -- drawing -------------------------------------------------------------

    def _set(self, key: str, pos, color=None, width=None) -> None:
        line = self._lines.get(key)
        if line is None:
            return
        try:
            pos = np.asarray(pos, dtype=float)
            if pos.ndim == 2 and pos.shape[0] >= 2:
                kw = {"pos": pos, "connect": "segments"}
                # Pass colour + width ALONGSIDE pos (the standard vispy Line
                # path) so a Render/style change is applied on the same redraw.
                if color is not None:
                    kw["color"] = color
                if width is not None:
                    kw["width"] = float(width)
                line.set_data(**kw)
                line.visible = True
            else:
                # vispy Line needs >= 2 points; hide instead of feeding empty.
                line.visible = False
        except Exception:
            pass

    def update(self, boxes, active_index, shape, styles=None) -> None:
        """Redraw the idle + active visuals from ALL current boxes.

        *styles* (OPTIONAL, default None → the pre-P2 single-style behaviour so ③
        stays byte-identical) is a per-box ``{color, opacity, shape}`` list
        parallel to *boxes* — the ④ location overlay's per-patch styling.  When
        given, every box renders idle on the SINGLE idle Line with per-VERTEX
        colour + per-box shape (width GLOBAL); active/ghost stay empty."""
        boxes = list(boxes) if boxes else []
        self._last_update = (boxes, active_index, shape, styles)
        self._render_boxes(boxes, active_index, shape, styles)

    def _render_boxes(self, boxes, active_index, shape, styles=None) -> None:
        """Emit the idle/active geometry permuted for the CURRENT dims order."""
        if shape is None:
            boxes = []
        do = self._dims_order()
        vs = self._voxel_size
        if styles is not None:
            # Per-box mode (④): mixed colours + shapes on ONE idle Line via a
            # per-vertex RGBA array (no per-box Line → no freed-VBO crash); the
            # width stays global.  active/ghost are irrelevant here.
            seg, colors = boxes_segments_and_colors_xyz(
                boxes, shape, styles, self._effective_color("idle"), do, vs)
            self._set("idle", seg, color=colors, width=self._idle_width)
            self._set("active", np.zeros((0, 3), dtype=float))
            self.repaint()
            return
        if (active_index is not None and 0 <= active_index < len(boxes)):
            active = [boxes[active_index]]
            idle = [b for i, b in enumerate(boxes) if i != active_index]
        else:
            active = []
            idle = boxes
        self._set("idle", boxes_segments_xyz(idle, shape, self._style, do, vs),
                  color=self._effective_color("idle"), width=self._idle_width)
        self._set("active", boxes_segments_xyz(active, shape, self._style, do, vs),
                  color=self._effective_color("active"), width=self._active_width)
        self.repaint()

    def update_ghost(self, box, shape) -> None:
        """Redraw the faint placement-preview box (where the next Add lands)."""
        self._last_ghost = (box, shape)
        seg = boxes_segments_xyz(
            [box] if box is not None else [], shape, self._style,
            self._dims_order(), self._voxel_size)
        self._set("ghost", seg)
        self.repaint()

    def clear_ghost(self) -> None:
        self._last_ghost = None
        self._set("ghost", np.zeros((0, 3), dtype=float))
        self.repaint()

    def repaint(self) -> None:
        native = self._native
        if native is None:
            return
        try:
            native.update()
        except Exception:
            pass

    # -- dims-follow (quirk 2: the box must track napari's view orientation) --

    def _dims_order(self):
        """The current napari world dims order as a length-3 permutation, or
        ``None`` (→ the plain ZYX→XYZ fallback).  Guarded — never raises."""
        try:
            order = tuple(int(a) for a in self._viewer.dims.order)
        except Exception:
            return None
        if len(order) == 3 and sorted(order) == [0, 1, 2]:
            return order
        return None

    def _on_dims_changed(self, event=None) -> None:
        """Re-permute + redraw the cached geometry when the view orientation
        changes (roll to XZ/YZ, or 2D↔3D).  Guarded — never raises out of the
        Qt slot."""
        try:
            if self._last_update is not None:
                boxes, active_index, shape, styles = self._last_update
                self._render_boxes(boxes, active_index, shape, styles)
            if self._last_ghost is not None:
                box, shape = self._last_ghost
                seg = boxes_segments_xyz(
                    [box] if box is not None else [], shape, self._style,
                    self._dims_order(), self._voxel_size)
                self._set("ghost", seg)
                self.repaint()
        except Exception:
            pass

    # -- teardown ------------------------------------------------------------

    def destroy(self) -> None:
        """Detach every visual (``parent = None``) + drop the dims subscription.
        Idempotent + guarded."""
        try:
            if self._dims_cb is not None and self._viewer is not None:
                self._viewer.dims.events.order.disconnect(self._dims_cb)
                self._viewer.dims.events.ndisplay.disconnect(self._dims_cb)
        except Exception:
            pass
        self._dims_cb = None
        for line in list(self._lines.values()):
            try:
                line.parent = None
            except Exception:
                pass
        self._lines.clear()
        self._native = None
        self._last_update = None
        self._last_ghost = None
