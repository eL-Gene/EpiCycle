"""3D ROI box geometry for the de-novo Add tool (and reusable box ROIs).

Pure algorithm module — no Qt, no napari.  Lifts the headlessly-tested core of
the EpiCycle-Viz ``napari_roi_tool`` prototype: a center+size box dataclass,
clamped bounds (as a slices-tuple for ROI cropping), and 3D wireframe edge
geometry (data coords) for a read-only Shapes overlay.

napari can only interactively EDIT a Shapes layer in ``ndisplay=2``, so the box
is *positioned via sliders* (center + size) and rendered as a passive wireframe
in the 3D view — "drag via sliders" — confirmed across 3 independent sources
plus this prototype.  The widget layer (advanced_features_widget / the Add mode)
owns the sliders, the live mirroring across the main + 3 subviewers, and the
mouse placement; this module owns only the pure geometry they all share.

Coordinate convention: ZYX voxel indices throughout.  Multiply by the layer
``scale`` (voxel size) at render time — geometry here is in array/data coords.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product

import numpy as np


@dataclass
class BoxRec:
    """An axis-aligned box by center + size, in ZYX voxel coords."""

    cz: int
    cy: int
    cx: int
    sz: int
    sy: int
    sx: int

    # -- construction helpers ------------------------------------------------
    @classmethod
    def centered(cls, shape, *, size=None) -> "BoxRec":
        """A box centered in *shape*; default size = a quarter of each axis."""
        Z, Y, X = (int(s) for s in shape)
        if size is None:
            sz = max(1, Z // 4)
            sy = max(1, Y // 4)
            sx = max(1, X // 4)
        else:
            sz, sy, sx = (max(1, int(v)) for v in size)
        return cls(Z // 2, Y // 2, X // 2, sz, sy, sx)

    def with_center(self, cz, cy, cx) -> "BoxRec":
        return replace(self, cz=int(cz), cy=int(cy), cx=int(cx))

    def with_size(self, sz, sy, sx) -> "BoxRec":
        return replace(self, sz=max(1, int(sz)), sy=max(1, int(sy)), sx=max(1, int(sx)))

    # -- bounds --------------------------------------------------------------
    def bounds_dict(self, shape) -> dict:
        """Clamped half-open bounds ``{z0,z1,y0,y1,x0,x1}`` within *shape*.

        Guarantees ``0 <= a0 < a1 <= dim`` on every axis (always ≥ 1 voxel,
        never out of range).  A box that would overflow an edge is truncated
        at that edge (not shifted), which matches the intuition "the box can't
        extend past the data".
        """
        Z, Y, X = (int(s) for s in shape)
        out = {}
        for axis, c, s, dim in (
            ("z", self.cz, self.sz, Z),
            ("y", self.cy, self.sy, Y),
            ("x", self.cx, self.sx, X),
        ):
            s = max(1, int(s))
            a0 = int(np.clip(int(c) - s // 2, 0, max(0, dim - 1)))
            a1 = int(np.clip(a0 + s, a0 + 1, dim))
            out[f"{axis}0"] = a0
            out[f"{axis}1"] = a1
        return out

    def bounds_slices(self, shape) -> tuple[slice, slice, slice]:
        """Clamped bounds as a ``(slice_z, slice_y, slice_x)`` ROI tuple."""
        b = self.bounds_dict(shape)
        return (
            slice(b["z0"], b["z1"]),
            slice(b["y0"], b["y1"]),
            slice(b["x0"], b["x1"]),
        )

    def shape_in(self, shape) -> tuple[int, int, int]:
        """The (z, y, x) voxel extent of this box clamped to *shape*."""
        b = self.bounds_dict(shape)
        return (b["z1"] - b["z0"], b["y1"] - b["y0"], b["x1"] - b["x0"])

    def clamped(self, shape) -> "BoxRec":
        """A copy with the center pulled so the (clamped) box's center matches.

        Keeps the slider value in sync with what is actually rendered when the
        box hugs an edge.

        .. warning::
            ``clamped`` re-centers but KEEPS the requested size, so on a box
            that overflows an edge it re-widens the clamped extent on the *next*
            pass (dim=10, size=6, center=8 → bounds ``[5,10)`` → after clamp the
            bounds become ``[4,10)``, gaining a column) — i.e. it is NOT
            idempotent and it CHANGES the extracted region.  For syncing the
            geometry widgets to the rendered box use :meth:`shrunk_to_bounds`
            (byte-identical extraction) or :meth:`slid_inward` (keep-size,
            deliberate re-extraction) instead — both are idempotent.
        """
        b = self.bounds_dict(shape)
        return replace(
            self,
            cz=(b["z0"] + b["z1"]) // 2,
            cy=(b["y0"] + b["y1"]) // 2,
            cx=(b["x0"] + b["x1"]) // 2,
        )

    def shrunk_to_bounds(self, shape) -> "BoxRec":
        """A copy whose center+size EXACTLY equal the clamped bounds it renders.

        Unlike :meth:`clamped` (which keeps the size and only re-centers, and so
        re-grows an edge-overflowing box on the next pass), this collapses BOTH
        the center and the size onto :meth:`bounds_dict`'s own output — per axis
        ``center = (a0 + a1) // 2``, ``size = a1 - a0``.  Consequences:

        * ``bounds_dict`` of the result is byte-identical to ``bounds_dict`` of
          this box (extraction reads ``bounds_dict``, so extraction is
          unchanged);
        * repeated application is a fixed point (no creep), because the size is
          already the clamped extent;
        * a widget seeded from the result shows the box that is actually drawn
          and extracted — the "shrink-to-fit" edge behavior.
        """
        b = self.bounds_dict(shape)
        return replace(
            self,
            cz=(b["z0"] + b["z1"]) // 2,
            cy=(b["y0"] + b["y1"]) // 2,
            cx=(b["x0"] + b["x1"]) // 2,
            sz=b["z1"] - b["z0"],
            sy=b["y1"] - b["y0"],
            sx=b["x1"] - b["x0"],
        )

    def slid_inward(self, shape) -> "BoxRec":
        """A copy re-centered to keep the requested SIZE within *shape*.

        Per axis the center is pulled into ``[s // 2, dim - (s - s // 2)]`` via a
        FIXED clamp (``center = min(max(c, s // 2), dim - (s - s // 2))``) — the
        box slides inward off an edge instead of shrinking.  This is idempotent
        (a second application is a no-op, even when ``s > dim``) — the exact
        property :meth:`clamped` lacks — but because the size is preserved it
        DELIBERATELY changes the edge extraction relative to
        :meth:`shrunk_to_bounds` (a full-size patch at a shifted center).
        """
        def _center(c, s, dim):
            s = max(1, int(s))
            lo = s // 2
            hi = int(dim) - (s - s // 2)
            return int(min(max(int(c), lo), hi))

        Z, Y, X = (int(v) for v in shape)
        return replace(
            self,
            cz=_center(self.cz, self.sz, Z),
            cy=_center(self.cy, self.sy, Y),
            cx=_center(self.cx, self.sx, X),
        )


# ---------------------------------------------------------------------------
# Wireframe geometry (data coords, ZYX) — lifted from the EpiCycle-Viz prototype
# ---------------------------------------------------------------------------

def box_edges(b: dict) -> list[np.ndarray]:
    """The 12 cube edges of bounds *b* as ``(2, 3)`` ZYX point pairs.

    An edge connects two of the 8 corners that differ in exactly one coord.
    Suitable as napari Shapes ``path`` data (one path per edge).
    """
    corners = np.array(
        list(product([b["z0"], b["z1"]], [b["y0"], b["y1"]], [b["x0"], b["x1"]])),
        dtype=float,
    )
    return [
        np.stack([corners[i], corners[j]])
        for i, j in combinations(range(8), 2)
        if np.count_nonzero(corners[i] != corners[j]) == 1
    ]


def box_corners(b: dict, frac: float = 0.18) -> list[np.ndarray]:
    """Corner-bracket segments (3 short edges at each of the 8 corners).

    A lighter-weight box render than the full wireframe; ``frac`` is the
    bracket length as a fraction of each edge.
    """
    z0, z1, y0, y1, x0, x1 = b["z0"], b["z1"], b["y0"], b["y1"], b["x0"], b["x1"]
    segs: list[np.ndarray] = []
    for iz, z in enumerate((z0, z1)):
        for iy, y in enumerate((y0, y1)):
            for ix, x in enumerate((x0, x1)):
                p = np.array([z, y, x], float)
                az = (z1 - z0) * frac * (1 if iz == 0 else -1)
                ay = (y1 - y0) * frac * (1 if iy == 0 else -1)
                ax = (x1 - x0) * frac * (1 if ix == 0 else -1)
                segs += [
                    np.array([p, p + [az, 0, 0]]),
                    np.array([p, p + [0, ay, 0]]),
                    np.array([p, p + [0, 0, ax]]),
                ]
    return segs


def box_wireframe(box: BoxRec, shape, style: str = "edges") -> list[np.ndarray]:
    """Convenience: paths for *box* clamped to *shape* (``"edges"``/``"corners"``)."""
    b = box.bounds_dict(shape)
    return box_corners(b) if style == "corners" else box_edges(b)


def box_face_rectangle(b: dict, axis: str) -> np.ndarray:
    """A closed 4-corner rectangle (5 points) on the box's mid-plane for *axis*.

    Used to mirror the box into the 2D subviewers:
      - ``axis="z"`` → XY rectangle at mid-Z (the XY subviewer)
      - ``axis="y"`` → XZ rectangle at mid-Y (the XZ subviewer)
      - ``axis="x"`` → YZ rectangle at mid-X (the YZ subviewer)
    Returned as ``(5, 3)`` ZYX points (closed loop), in data coords.
    """
    z0, z1, y0, y1, x0, x1 = b["z0"], b["z1"], b["y0"], b["y1"], b["x0"], b["x1"]
    if axis == "z":
        z = (z0 + z1) // 2
        pts = [(z, y0, x0), (z, y0, x1), (z, y1, x1), (z, y1, x0), (z, y0, x0)]
    elif axis == "y":
        y = (y0 + y1) // 2
        pts = [(z0, y, x0), (z0, y, x1), (z1, y, x1), (z1, y, x0), (z0, y, x0)]
    elif axis == "x":
        x = (x0 + x1) // 2
        pts = [(z0, y0, x), (z0, y1, x), (z1, y1, x), (z1, y0, x), (z0, y0, x)]
    else:
        raise ValueError(f"axis must be 'z', 'y', or 'x'; got {axis!r}")
    return np.array(pts, dtype=float)
