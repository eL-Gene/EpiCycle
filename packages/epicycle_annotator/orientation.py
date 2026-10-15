"""Orientation transforms for XY/XZ/YZ subviewer representations."""

from __future__ import annotations

import numpy as np


def transpose_data(data: np.ndarray, orientation: str) -> np.ndarray:
    """Return a transposed view for the requested orientation."""
    orient = str(orientation or "").upper()
    if data.ndim == 4:
        if orient == "XY":
            return data
        if orient == "XZ":
            return data.transpose(1, 0, 2, 3)
        if orient == "YZ":
            return data.transpose(2, 1, 0, 3)
        raise ValueError(f"Unsupported orientation: {orientation}")

    if orient == "XY":
        return data
    if orient == "XZ":
        return data.transpose(1, 0, 2)
    if orient == "YZ":
        return data.transpose(2, 1, 0)
    raise ValueError(f"Unsupported orientation: {orientation}")


def inverse_transpose_data(data: np.ndarray, orientation: str) -> np.ndarray:
    """Return data transposed back to canonical ZYX orientation."""
    # These permutations are self-inverse for our axis order.
    return transpose_data(data, orientation)


# (the ZYX axis each of a side view's three data axes holds)
_VIEW_TO_ZYX = {"XY": (0, 1, 2), "XZ": (1, 0, 2), "YZ": (2, 1, 0)}


def subview_point_to_zyx(orientation, point, *, scale=None, translate=None):
    """A side view's point, back in volume ZYX data coordinates.

    *point* is in the side view's own (transposed) axis order -- a napari
    event's ``position``.  Only its LAST three entries are used: a 4th data axis exists only for an RGB side image, where it is
    the LAST axis and napari keeps it out of the world point, so
    ``point[-3:]`` is right.  When
    *scale* / *translate* are given (the side layer's own, same order, last
    three used) the world point is taken back to data first:
    ``data = (world - translate) / scale``.  Then ``transpose_data``'s
    permutation is undone:

        XY -> (p0, p1, p2)    XZ -> (p1, p0, p2)    YZ -> (p2, p1, p0)

    Returns floats; the caller rounds.
    """
    orient = str(orientation or "").upper()
    perm = _VIEW_TO_ZYX.get(orient)
    if perm is None:
        raise ValueError(f"Unsupported orientation: {orientation}")
    p = [float(v) for v in tuple(point)[-3:]]
    if len(p) != 3:
        raise ValueError(f"need at least three coordinates, got {point!r}")
    if translate is not None:
        t = [float(v) for v in tuple(translate)[-3:]]
        p = [a - b for a, b in zip(p, t)]
    if scale is not None:
        s = [float(v) for v in tuple(scale)[-3:]]
        p = [a / (b if abs(b) > 1e-12 else 1.0) for a, b in zip(p, s)]
    zyx = [0.0, 0.0, 0.0]
    for i, axis in enumerate(perm):
        zyx[axis] = p[i]
    return tuple(zyx)
