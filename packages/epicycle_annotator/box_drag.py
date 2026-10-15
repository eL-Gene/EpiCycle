"""Box drag and nudge in the side views -- the pure half (Add rebuild B3).

No Qt, no napari.  The Add orchestrator (``curation_3d_widget.py``) owns
the mouse and key wiring; this module owns the geometry it asks about:

* :func:`outline_hit` -- is a side-view point on the box's rectangle?
* :func:`inside_box` -- is it inside the box (a marker-click candidate)?
* :func:`outline_tolerance` -- ``6 / zoom`` voxels (6 screen pixels).
* :func:`drag_delta` / :func:`moved_box` -- a drag moves the WHOLE box by
  the drag's delta on the two displayed axes; it keeps its size and slides
  back inside the volume.
* :func:`is_click` -- a press + release closer than ``ADD_CLICK_DRAG_PX``.
* :func:`nudge_delta` -- Shift+arrow, one voxel along that view's axes.

Coordinates are ZYX voxel indices.  Each side view shows two axes as
(row, col) -- the frame ``box_overlay`` draws its rectangles in::

    XY -> (Y, X), slice Z      XZ -> (Z, X), slice Y      YZ -> (Y, Z), slice X

(from ``orientation.transpose_data``; a point is mapped back to ZYX by
``orientation.subview_point_to_zyx`` first).
"""

from __future__ import annotations

import math

#: A press and release at most this many screen pixels apart is a click.
ADD_CLICK_DRAG_PX = 4

#: A press within this many screen pixels of the outline grabs the box.
ADD_OUTLINE_TOL_PX = 6

# (row axis, col axis, slice axis) as ZYX indices, per side view
_AXES = {"XY": (1, 2, 0), "XZ": (0, 2, 1), "YZ": (1, 0, 2)}
_KEYS = ("z", "y", "x")


def view_axes(orient):
    """``(row, col, slice)`` ZYX axis indices for *orient*, or None."""
    return _AXES.get(str(orient or "").upper())


def outline_tolerance(zoom, px=ADD_OUTLINE_TOL_PX, max_vox=None) -> float:
    """``px / zoom`` voxels: *px* screen pixels at the view's zoom (screen
    pixels per voxel).  That is voxels only because the side views' labels
    layer has an IDENTITY scale (the side views show voxel indices; audit
    L4) -- a non-unit side-layer scale would make it world units.  A zoom that is missing, zero or not a number reads
    as 1.  *max_vox* caps the band (see :func:`outline_cap`), so a
    zoomed-out view cannot swallow the box's middle."""
    try:
        z = float(zoom)
    except (TypeError, ValueError):
        z = 1.0
    if not math.isfinite(z) or z <= 1e-6:
        z = 1.0
    tol = float(px) / z
    if max_vox is not None:
        try:
            tol = min(tol, max(0.0, float(max_vox)))
        except (TypeError, ValueError):
            pass
    return tol


def outline_cap(bounds, orient):
    """A quarter of the box's smaller DISPLAYED extent in *orient*, in
    voxels -- the most the grab band may take (B3 fix pass).  None for an
    unknown view."""
    axes = view_axes(orient)
    if axes is None:
        return None
    ext = []
    for axis in axes[:2]:
        lo, hi = _lo_hi(bounds, axis)
        ext.append(hi - lo)
    return min(ext) / 4.0


def _lo_hi(bounds, axis):
    k = _KEYS[axis]
    return float(bounds[f"{k}0"]), float(bounds[f"{k}1"])


def _finite3(zyx):
    try:
        vals = tuple(float(v) for v in zyx)
    except (TypeError, ValueError):
        return None
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        return None
    return vals


def outline_hit(bounds, orient, zyx, tol_vox) -> bool:
    """True when *zyx* lies within *tol_vox* of one of the four edges of the
    box's rectangle in *orient*'s plane AND the view's slice is inside the
    box's range on the third axis.

    *bounds* is ``BoxRec.bounds_dict`` (half-open); the rectangle is drawn
    on the ``a0`` and ``a1`` lines, as ``box_overlay`` draws it.  The slice
    range is ``[a0, a1)``."""
    axes = view_axes(orient)
    p = _finite3(zyx)
    if axes is None or p is None:
        return False
    r_ax, c_ax, s_ax = axes
    tol = max(0.0, float(tol_vox))
    s0, s1 = _lo_hi(bounds, s_ax)
    if not (s0 <= p[s_ax] < s1):
        return False
    r0, r1 = _lo_hi(bounds, r_ax)
    c0, c1 = _lo_hi(bounds, c_ax)
    r, c = p[r_ax], p[c_ax]
    if not (r0 - tol <= r <= r1 + tol and c0 - tol <= c <= c1 + tol):
        return False
    return min(abs(r - r0), abs(r - r1)) <= tol or \
        min(abs(c - c0), abs(c - c1)) <= tol


def slice_in_box(bounds, orient, slice_index) -> bool:
    """True when *orient*'s current slice (a voxel index on that view's
    slicing axis) is inside the box's ``[a0, a1)`` range there -- the same
    test :func:`outline_hit` makes, so a rectangle drawn "in range" is
    exactly a grabbable one.  Unknown view or slice: True (never dim on a
    guess)."""
    axes = view_axes(orient)
    if axes is None:
        return True
    try:
        s = float(slice_index)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(s):
        return True
    s0, s1 = _lo_hi(bounds, axes[2])
    return s0 <= s < s1


def inside_box(bounds, zyx) -> bool:
    """True when *zyx* is inside the (half-open) box on all three axes."""
    p = _finite3(zyx)
    if p is None:
        return False
    for axis in range(3):
        lo, hi = _lo_hi(bounds, axis)
        if not (lo <= p[axis] < hi):
            return False
    return True


def drag_delta(orient, zyx0, zyx1):
    """``(dz, dy, dx)`` whole voxels from *zyx0* to *zyx1*; only the two
    axes *orient* displays move (the slice axis stays 0)."""
    axes = view_axes(orient)
    if axes is None:
        return (0, 0, 0)
    out = [0, 0, 0]
    for axis in axes[:2]:
        out[axis] = int(round(float(zyx1[axis]) - float(zyx0[axis])))
    return tuple(out)


def moved_box(box, delta, shape, axes=None):
    """*box* moved by *delta* ``(dz, dy, dx)``, KEEPING ITS SIZE and slid
    back inside *shape* (B3 fix pass; ``BoxRec.slid_inward``, which is
    idempotent and symmetric -- never ``clamped``, which truncates at the
    high edge and is not idempotent).

    Only the axes in *axes* change: a drag passes the view's two displayed
    axes, a nudge the one axis it moves; ``None`` means the axes with a
    non-zero delta.  On those axes the box is first normalised
    (``slid_inward``), then moved, then slid inward again; every other
    axis -- the slicing axis above all -- keeps its centre and size
    exactly, even when it overflows the volume.  Always computed from the
    PRESS-time box and the total delta, so nothing accumulates over a
    drag.  A zero delta returns the normalised box (equal to *box* when it
    was already inside)."""
    d = [int(v) for v in delta]
    if axes is None:
        axes = [a for a in range(3) if d[a] != 0]
    axes = {int(a) for a in axes}
    if not axes:
        return box
    norm = box.slid_inward(shape)
    c = [box.cz, box.cy, box.cx]
    n = [norm.cz, norm.cy, norm.cx]
    for a in axes:
        c[a] = n[a] + d[a]
    slid = box.with_center(*c).slid_inward(shape)
    s = [slid.cz, slid.cy, slid.cx]
    out = [box.cz, box.cy, box.cx]
    for a in axes:
        out[a] = s[a]
    return box.with_center(*out)


def is_click(px0, px1, threshold_px=ADD_CLICK_DRAG_PX) -> bool:
    """True when two canvas positions are at most *threshold_px* apart
    (a click, not a drag).  An unknown position is never a click."""
    if px0 is None or px1 is None:
        return False
    try:
        d = math.hypot(float(px1[0]) - float(px0[0]),
                       float(px1[1]) - float(px0[1]))
    except (TypeError, ValueError, IndexError):
        return False
    return d <= float(threshold_px)


def nudge_delta(orient, key):
    """``(dz, dy, dx)`` for one Shift+arrow press in *orient*: Left/Right
    move one column, Up/Down one row, in that view's own picture.  None
    for an unknown view or key."""
    axes = view_axes(orient)
    if axes is None:
        return None
    step = {"Up": (0, -1), "Down": (0, 1),
            "Left": (1, -1), "Right": (1, 1)}.get(str(key))
    if step is None:
        return None
    which, sign = step
    out = [0, 0, 0]
    out[axes[which]] = sign
    return tuple(out)
