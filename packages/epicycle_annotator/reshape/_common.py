"""Small helpers shared by the growth modules and the Add tab's preview.

Private: ``compete``, ``outline`` and ``add_preview`` import these so the
spacing check, the cancel poll and the sample stride have ONE definition
(the outline's own threshold sample must use the very stride the box-info
worker cached its dark threshold with).

Pure: numpy.
"""

from __future__ import annotations

import math

import numpy as np

from .engine import ReshapeCancelled

__all__ = ["check_cancel", "sample_stride", "spacing_um"]


def spacing_um(spacing):
    """``spacing`` as three positive finite floats (z, y, x) in um, or a
    plain-word ``ValueError``."""
    try:
        sp = tuple(float(s) for s in spacing)
    except TypeError:
        raise ValueError("spacing_um must be three numbers (z, y, x) in um, "
                         "got {!r}.".format(spacing)) from None
    if len(sp) != 3 or not all(math.isfinite(s) and s > 0 for s in sp):
        raise ValueError("spacing_um must be three positive numbers (z, y, x) "
                         "in um, got {!r}.".format(spacing))
    return sp


def check_cancel(cancel):
    """Raise :class:`~.engine.ReshapeCancelled` when ``cancel`` (an object
    with ``is_set()``, or ``None``) is set."""
    if cancel is not None and cancel.is_set():
        raise ReshapeCancelled()


def sample_stride(shape, budget):
    """The smallest stride whose ``[::s, ::s, ::s]`` sample is at most
    ``budget`` voxels (an empty axis counts as one voxel)."""
    shape = [max(1, int(v)) for v in shape]
    s = 1
    while int(np.prod([math.ceil(n / s) for n in shape])) > int(budget):
        s += 1
    return s
