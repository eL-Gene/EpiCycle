"""Read-only ROI-box overlay for the de-novo Add tool.

Draws the box as vispy ``Line`` visuals — a 3D wireframe on the main canvas plus
a 2D rectangle on each XY/XZ/YZ subviewer — using the SAME lifecycle idiom as the
crosshair overlays (``crosshairs.py``): a visual parented to a ``qt_viewer``'s
scene, whose ``destroy()`` sets ``parent = None``.

Deliberately **not** a napari Shapes layer.  A Shapes-in-3D layer renders its path
as a Compound Mesh whose vertex buffer can be freed while a ``paintGL`` is still
queued on patch-switch / teardown — which faults the 3D canvas with a
``glDrawArrays`` access violation (``0x1C`` = a null/freed VBO base).  A raw vispy
``Line`` owns its own lifecycle and never enters ``viewer.layers``, so it can
neither be missed by the controller's layer sweep nor orphan into the next patch.
The access-violation class is therefore gone *by construction*.

Coordinate frames (verified against ``crosshairs.py`` + ``orientation.transpose_data``):
  - MAIN 3D scene == napari world coords in **XYZ** order: reverse the ZYX box
    points and apply the main labels layer's per-axis ``scale`` + ``translate``.
  - Each subviewer scene is 2D with a scene point == ``(x=col, y=row)``.  The
    displayed (row, col) axes per orientation are:
        XY -> (Y, X)    XZ -> (Z, X)    YZ -> (Y, Z)
"""

from __future__ import annotations

import numpy as np
from vispy.scene.visuals import Line

from .roi_box import box_edges

_BOX_COLOR = (1.0, 0.55, 0.0, 1.0)  # orange
# a side rectangle whose view's slice is outside the box's range: drawn, but
# dimmed -- it cannot be grabbed there (B3 fix pass, LOW-4)
_BOX_COLOR_DIM = (1.0, 0.55, 0.0, 0.3)
_SIDE_KEYS = ("XY", "XZ", "YZ")
_WIDTH = 1.5  # pixels (vispy Line width is screen-space, not data units)


def _native_deleted(native) -> bool:
    """True when native (a canvas's Qt widget) has already been deleted
    by Qt.  Best effort across PyQt (sip) and PySide (shiboken); False
    when it cannot tell."""
    if native is None:
        return False
    for mod_name in ("qtpy.sip", "PyQt6.sip", "PyQt5.sip", "sip"):
        try:
            mod = __import__(mod_name, fromlist=["isdeleted"])
            return bool(mod.isdeleted(native))
        except Exception:
            continue
    for mod_name in ("shiboken6", "shiboken2"):
        try:
            mod = __import__(mod_name)
            return not bool(mod.isValid(native))
        except Exception:
            continue
    return False


def _rect(r0, r1, c0, c1) -> np.ndarray:
    """Closed 5-point rectangle as vispy ``(x=col, y=row)`` scene points."""
    return np.array(
        [[c0, r0], [c1, r0], [c1, r1], [c0, r1], [c0, r0]], dtype=float
    )


class AddBoxOverlay:
    """Orange ROI box: 3D wireframe on the main canvas + 2D rect per subviewer.

    Construct once per Add session with the four ``QtViewer`` handles (the main
    one may be ``None`` — the subviewer rectangles still work); call
    :meth:`update` on every box change and :meth:`destroy` on teardown.
    """

    def __init__(self, main_qt_viewer, qt_viewer_xy, qt_viewer_xz, qt_viewer_yz):
        self._lines: dict[str, Line] = {}
        self._natives: dict[str, object] = {}
        self._dimmed: dict[str, bool] = {}
        for key, qtv, ndim in (
            ("MAIN", main_qt_viewer, 3),
            ("XY", qt_viewer_xy, 2),
            ("XZ", qt_viewer_xz, 2),
            ("YZ", qt_viewer_yz, 2),
        ):
            if qtv is None:
                continue
            try:
                scene = qtv.canvas.view.scene
            except Exception:
                continue
            try:
                line = Line(
                    pos=np.zeros((2, ndim), dtype=float),
                    color=_BOX_COLOR,
                    width=_WIDTH,
                    connect="segments" if key == "MAIN" else "strip",
                    parent=scene,
                )
                line.order = 1e6  # draw on top of the volume/labels
                self._lines[key] = line
                self._natives[key] = qtv.canvas.native
            except Exception:
                continue

    def _set(self, key: str, pos, connect: str) -> None:
        line = self._lines.get(key)
        if line is None:
            return
        try:
            line.set_data(pos=np.asarray(pos, dtype=float), connect=connect)
        except Exception:
            pass

    def set_in_range(self, key: str, inside: bool, *,
                     repaint: bool = True) -> None:
        """Full colour when *key*'s slice is inside the box's range, dimmed
        when it is not (the rectangle is drawn on every slice but grabbable
        only inside the range).  A no-op when nothing changes."""
        key = str(key).upper()
        dim = not bool(inside)
        if bool(self._dimmed.get(key, False)) == dim:
            return
        self._dimmed[key] = dim
        line = self._lines.get(key)
        if line is None:
            return
        try:
            line.set_data(color=_BOX_COLOR_DIM if dim else _BOX_COLOR)
        except Exception:
            pass
        if repaint:
            native = self._natives.get(key)
            try:
                if native is not None:
                    native.update()
            except Exception:
                pass

    def is_dimmed(self, key: str) -> bool:
        return bool(self._dimmed.get(str(key).upper(), False))

    def update(self, box, shape, *, scale=None, translate=None,
               keys=None, in_range=None) -> None:
        """Redraw the visuals for *box* clamped to *shape*.

        *keys* (``"MAIN"`` / ``"XY"`` / ``"XZ"`` / ``"YZ"``) limits the
        redraw; ``None`` redraws all four.  A side-view drag passes the
        three side keys and skips the 3D wireframe ("Live
        redraw"); the release does a full update.  *in_range* maps a side
        key to whether that view's slice is inside the box's range (see
        :meth:`set_in_range`); a key it leaves out keeps its state."""
        for key, inside in dict(in_range or {}).items():
            if str(key).upper() in _SIDE_KEYS:
                self.set_in_range(key, inside, repaint=False)
        b = box.bounds_dict(shape)
        want = None if keys is None else {str(k).upper() for k in keys}

        def _on(key):
            return want is None or key in want

        # --- MAIN 3D wireframe: 12 ZYX edges -> XYZ world coords ---
        edges = box_edges(b) if _on("MAIN") else []
        if edges:
            seg = np.concatenate(edges, axis=0)  # (24, 3) ZYX
            sc = np.asarray(
                scale if scale is not None else (1.0, 1.0, 1.0), dtype=float
            )[-3:][::-1]
            tr = np.asarray(
                translate if translate is not None else (0.0, 0.0, 0.0),
                dtype=float,
            )[-3:][::-1]
            pos3d = seg[:, ::-1] * sc + tr
            self._set("MAIN", pos3d, "segments")
        # --- 2D subviewer rectangles (raw voxel-index scene coords) ---
        if _on("XY"):
            self._set("XY", _rect(b["y0"], b["y1"], b["x0"], b["x1"]),
                      "strip")
        if _on("XZ"):
            self._set("XZ", _rect(b["z0"], b["z1"], b["x0"], b["x1"]),
                      "strip")
        if _on("YZ"):
            self._set("YZ", _rect(b["y0"], b["y1"], b["z0"], b["z1"]),
                      "strip")
        self.repaint()

    def repaint(self) -> None:
        for native in self._natives.values():
            try:
                native.update()
            except Exception:
                pass

    def destroy(self) -> None:
        """Detach every line from its scene.

        On napari's X the annotator's canvases are deleted BEFORE the Add
        cleanup reaches here; detaching then makes vispy touch the dead
        CanvasBackendDesktop, and vispy's own emitter catches the
        RuntimeError and logs it as an ERROR (live-shot fix F6).  So a line
        whose canvas is gone is only dropped, never detached; and the
        detach itself is guarded narrowly for a canvas dying mid-call."""
        for key, line in list(self._lines.items()):
            if _native_deleted(self._natives.get(key)):
                continue
            try:
                line.parent = None
            except RuntimeError:
                pass
        self._lines.clear()
        self._natives.clear()
