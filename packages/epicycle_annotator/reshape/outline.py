"""The tissue outline: where new cells may grow.

Two outlines, both plain boolean volumes the size of the patch:

* :func:`tissue_outline` -- a shrink-wrap around the stain.  Every given
  channel is normalised by the patch's 1st / 99.5th percentiles, the
  channels are max-projected and blurred by ``DARK_BLUR_UM``; what is at or
  above ``dark_factor x`` the patch Otsu is "not dark".  The not-dark mask
  is closed with a physical ball (padded with its own edge values, so the
  patch faces neither erode nor glue it), every piece of at least
  ``min_piece_um3`` is kept -- both sides of a wound survive -- and holes
  are NOT filled (a 3D hole fill would fill every closed cavity, a lumen or
  a cyst, whatever the ball).  A closing of radius r fills any gap, fold or
  lumen up to about 2 r + 2 um wide: the blur before the cut widens it
  (measured on 0.5 um voxels: 12 um at r 5, 16 um at r 8).  A caller may
  give the radius (``close_radius_um``) or the gap to bridge
  (``bridge_um`` = 2 x the radius).  The whole build runs at ``stride`` (2:
  one voxel per 2 x 2 x 2 block) and is upsampled nearest.
* :func:`cleared_outline` -- the union of the objects a user cleared,
  closed with a small ball, at full resolution (only the mask's bounding
  box is processed).

:func:`boundary_shell` is the outline's one-voxel skin, for display, and
:func:`markers_outside` names the markers mostly outside an outline.

Nothing here writes into its inputs or into a labels volume: an outline
only limits where NEW objects grow and which automatic markers are dropped.

Pure: numpy / scipy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import ndimage as ndi

from ._common import check_cancel as _check_cancel
from ._common import sample_stride
from ._common import spacing_um as _spacing
from .region import (
    DARK_BLUR_UM,
    DARK_SAMPLE_VOXELS,
    patch_dark_threshold,
)

__all__ = [
    "CLEARED_CLOSE_UM",
    "OUTLINE_CLOSE_UM",
    "OUTLINE_CLOSE_UM_ONE_CHANNEL",
    "OUTLINE_DARK_FACTOR",
    "OUTLINE_DROP_SHARE",
    "OUTLINE_MIN_PIECE_UM3",
    "OUTLINE_SHELL_UM",
    "OUTLINE_STRIDE",
    "boundary_shell",
    "cleared_outline",
    "default_close_radius_um",
    "markers_outside",
    "tissue_outline",
]

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: How dim counts as outside: the share of the patch Otsu.
OUTLINE_DARK_FACTOR = 0.5

#: The closing ball's RADIUS (um) with two or more channels ...
OUTLINE_CLOSE_UM = 5.0

#: ... and with one channel, whichever stain it is: a single channel leaves
#: wider dim gaps inside the tissue.
OUTLINE_CLOSE_UM_ONE_CHANNEL = 8.0

#: Pieces of the outline smaller than this (um^3) are dropped.
OUTLINE_MIN_PIECE_UM3 = 1000.0

#: The shrink-wrap is built on one voxel per ``stride`` in every axis.
OUTLINE_STRIDE = 2

#: The closing ball's RADIUS (um) of the cleared-objects outline.
CLEARED_CLOSE_UM = 3.0

#: An automatic marker with at least this share of its voxels outside the
#: outline is dropped.
OUTLINE_DROP_SHARE = 0.7

#: The display shell's thickness (um).
OUTLINE_SHELL_UM = 0.5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _channels(channels):
    """``{name: 3D array}`` from a mapping, a sequence or one 3D array."""
    if isinstance(channels, Mapping):
        out = dict(channels)
    elif isinstance(channels, np.ndarray) and channels.ndim == 3:
        out = {"ch0": channels}
    else:
        out = {"ch{}".format(i): a for i, a in enumerate(channels or ())}
    if not out:
        raise ValueError("the tissue outline needs at least one image channel.")
    shape = None
    for name, arr in out.items():
        a = np.asarray(arr) if not hasattr(arr, "shape") else arr
        if len(a.shape) != 3:
            raise ValueError("channel {!r} is {}-dimensional; the outline takes "
                             "3D single-channel arrays.".format(name, len(a.shape)))
        if shape is None:
            shape = tuple(a.shape)
        elif tuple(a.shape) != shape:
            raise ValueError("channel {!r} has shape {}, another {}.".format(
                name, tuple(a.shape), shape))
        out[name] = a
    return out, shape


def default_close_radius_um(n_channels):
    """The closing radius for ``n_channels`` channels (a count rule only:
    it never guesses which stain a single channel is)."""
    return OUTLINE_CLOSE_UM_ONE_CHANNEL if int(n_channels) == 1 else OUTLINE_CLOSE_UM


def _close_radius(close_radius_um, bridge_um, default):
    """The closing radius from either the radius or the gap to bridge
    (``bridge_um`` = 2 x the radius); ``default`` when neither is given."""
    if close_radius_um is not None and bridge_um is not None:
        raise ValueError("give the closing radius (close_radius_um) or the gap "
                         "to bridge (bridge_um), not both.")
    if bridge_um is not None:
        b = float(bridge_um)
        if not (math.isfinite(b) and b >= 0):
            raise ValueError("bridge_um must be >= 0, got {!r}.".format(bridge_um))
        return b / 2.0
    r = float(default if close_radius_um is None else close_radius_um)
    if not (math.isfinite(r) and r >= 0):
        raise ValueError("close_radius_um must be >= 0, got {!r}.".format(
            close_radius_um))
    return r


def _block_mean_slab(slab, s):
    """``slab`` (s or fewer z-slices, full Y x X) -> one plane of s-blocks,
    the mean over the voxels each block holds (a partial block at an edge
    averages only its own voxels)."""
    nz, ny, nx = slab.shape
    py, px = (-ny) % s, (-nx) % s
    total = slab.sum(axis=0, dtype=np.float32)
    count = np.full((ny, nx), float(nz), np.float32)
    if py or px:
        total = np.pad(total, ((0, py), (0, px)))
        count = np.pad(count, ((0, py), (0, px)))
    by, bx = total.shape[0] // s, total.shape[1] // s
    total = total.reshape(by, s, bx, s).sum(axis=(1, 3), dtype=np.float32)
    count = count.reshape(by, s, bx, s).sum(axis=(1, 3), dtype=np.float32)
    return total / count


def _close_ball(mask, radius, spacing):
    """Binary closing with a PHYSICAL ball (two EDTs), padded with the mask's
    own edge values."""
    m = np.asarray(mask, bool)
    if radius <= 0:
        return m.copy()
    pad = [int(math.ceil(radius / s)) + 1 for s in spacing]
    mp = np.pad(m, [(p, p) for p in pad], mode="edge")
    dil = ndi.distance_transform_edt(~mp, sampling=spacing) <= radius
    closed = (ndi.distance_transform_edt(dil, sampling=spacing) > radius) | mp
    del dil
    return closed[tuple(slice(p, p + n) for p, n in zip(pad, m.shape))]


def _keep_big(mask, spacing, min_um3):
    """Every 6-connected piece of at least ``min_um3``.  The background
    is never a piece, even at ``min_um3=0`` (its size is -1)."""
    lab, n = ndi.label(mask)
    if n == 0:
        return np.asarray(mask, bool).copy()
    size = np.bincount(lab.ravel()).astype(np.float64) * float(np.prod(spacing))
    size[0] = -1.0
    return (size >= float(min_um3))[lab]


def _upsample(small, s, shape):
    """Nearest upsample by ``s`` in every axis, cut to ``shape``."""
    out = np.empty(shape, bool)
    for z in range(small.shape[0]):
        z0 = z * s
        if z0 >= shape[0]:
            break
        plane = np.repeat(np.repeat(small[z], s, axis=0), s, axis=1)
        out[z0:min(z0 + s, shape[0])] = plane[:shape[1], :shape[2]]
    return out


# --------------------------------------------------------------------------- #
# the outlines
# --------------------------------------------------------------------------- #
def tissue_outline(channels, spacing_um, *, dark_factor=OUTLINE_DARK_FACTOR,
                   close_radius_um=None, bridge_um=None,
                   min_piece_um3=OUTLINE_MIN_PIECE_UM3, stride=OUTLINE_STRIDE,
                   dark_threshold=None, cancel=None):
    """The shrink-wrap around the stain (see the module notes).

    ``channels`` -- ``{name: whole-patch 3D array}`` (or a sequence, or one
    3D array); ``spacing_um`` -- (z, y, x).  ``close_radius_um`` is the
    closing ball's RADIUS, or give ``bridge_um`` (= 2 x the radius) instead;
    with neither, the radius is 5 um, or 8 um when one channel is given.
    Gaps up to about 2 r + 2 um close (the blur widens the ball's 2 r:
    12 um at r 5, 16 um at r 8, measured).  ``dark_threshold`` -- the patch
    threshold
    ``{"t", "lo", "hi"}`` of ``region.patch_dark_threshold`` over the same
    channel names; ``None`` computes it from a strided sample.  ``cancel``
    (an object with ``is_set()``) is polled between steps and raises
    :class:`~.engine.ReshapeCancelled`.  Returns a bool volume the patch's
    shape; the inputs are only read.
    """
    chans, shape = _channels(channels)
    sp = _spacing(spacing_um)
    s = int(stride)
    if s < 1:
        raise ValueError("stride must be >= 1, got {!r}.".format(stride))
    factor = float(dark_factor)
    if not (math.isfinite(factor) and factor > 0):
        raise ValueError("dark_factor must be > 0, got {!r}.".format(dark_factor))
    radius = _close_radius(close_radius_um, bridge_um,
                           default_close_radius_um(len(chans)))
    _check_cancel(cancel)
    if dark_threshold is None:
        dark_threshold = patch_dark_threshold(
            chans, spacing=sp, stride=sample_stride(shape, DARK_SAMPLE_VOXELS))
    lo, hi = dark_threshold["lo"], dark_threshold["hi"]
    for name in chans:
        if name not in lo or name not in hi:
            raise ValueError("the dark threshold has no normalisation for "
                             "channel {!r}.".format(name))
    _check_cancel(cancel)
    # the normalised max projection, s z-slices at a time, block-averaged
    small_shape = tuple(-(-n // s) for n in shape)
    small = np.empty(small_shape, np.float32)
    for zi, z0 in enumerate(range(0, shape[0], s)):
        slab = None
        for name, arr in chans.items():
            a = np.asarray(arr[z0:z0 + s], np.float32)
            span = max(float(hi[name]) - float(lo[name]), 1e-6)
            a = np.clip((a - float(lo[name])) / span, 0.0, 1.0)
            slab = a if slab is None else np.maximum(slab, a)
        small[zi] = _block_mean_slab(slab, s) if s > 1 else slab[0]
        if zi % 16 == 15:
            _check_cancel(cancel)
    small_sp = tuple(v * s for v in sp)
    blurred = ndi.gaussian_filter(
        small, tuple(DARK_BLUR_UM / v for v in small_sp))
    del small
    not_dark = blurred >= factor * float(dark_threshold["t"])
    del blurred
    _check_cancel(cancel)
    closed = _close_ball(not_dark, radius, small_sp)
    del not_dark
    _check_cancel(cancel)
    kept = _keep_big(closed, small_sp, min_piece_um3)
    del closed
    _check_cancel(cancel)
    if s == 1:
        return kept
    return _upsample(kept, s, shape)


def cleared_outline(mask, spacing_um, close_radius_um=None, *, bridge_um=None,
                    cancel=None):
    """The union of the cleared objects (``mask``, bool), closed with a
    physical ball of radius ``close_radius_um`` (default 3 um), or of
    ``bridge_um / 2``; every piece kept, no hole fill, full resolution (no
    blur, so gaps up to about 2 r close).  Only the mask's bounding box (plus
    the ball) is processed; at a patch face the mask is extended by its own
    edge values.
    """
    m = np.asarray(mask)
    if m.ndim != 3:
        raise ValueError("mask must be a 3D volume, got {} dimensions.".format(m.ndim))
    m = m.astype(bool, copy=False)
    sp = _spacing(spacing_um)
    radius = _close_radius(close_radius_um, bridge_um, CLEARED_CLOSE_UM)
    out = np.zeros(m.shape, bool)
    if not m.any():
        return out
    _check_cancel(cancel)
    sl = ndi.find_objects(m.astype(np.uint8))[0]
    margin = [int(math.ceil(radius / v)) + 1 for v in sp]
    box = tuple(slice(max(0, a.start - g), min(n, a.stop + g))
                for a, g, n in zip(sl, margin, m.shape))
    out[box] = _close_ball(m[box], radius, sp)
    return out


def boundary_shell(outline, spacing_um, thickness_um=OUTLINE_SHELL_UM):
    """The outline's skin: outline voxels within ``thickness_um`` of the
    outside (at least the face-adjacent layer).  The patch faces are not a
    boundary -- the tissue goes on past them."""
    from .arbiter import physical_ball

    o = np.asarray(outline)
    if o.ndim != 3:
        raise ValueError("outline must be a 3D volume, got {} dimensions.".format(o.ndim))
    o = o.astype(bool, copy=False)
    sp = _spacing(spacing_um)
    t = float(thickness_um)
    if not (math.isfinite(t) and t > 0):
        raise ValueError("thickness_um must be > 0, got {!r}.".format(thickness_um))
    inner = ndi.binary_erosion(o, structure=physical_ball(t, sp), border_value=1)
    return o & ~inner


def markers_outside(markers, outline, share=OUTLINE_DROP_SHARE):
    """The ids of ``markers`` with at least ``share`` of their voxels outside
    ``outline``, ascending, in the markers' dtype."""
    mk = np.asarray(markers)
    if mk.ndim != 3:
        raise ValueError("markers must be a 3D volume, got {} dimensions.".format(mk.ndim))
    if mk.dtype == np.bool_ or not np.issubdtype(mk.dtype, np.integer):
        raise ValueError("markers must hold integer ids, got dtype {}.".format(mk.dtype))
    o = np.asarray(outline)
    if o.shape != mk.shape:
        raise ValueError("outline has shape {}, the markers {}.".format(o.shape, mk.shape))
    share = float(share)
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be in [0, 1], got {!r}.".format(share))
    nz = mk != 0
    if not nz.any():
        return np.zeros(0, mk.dtype)
    ids, inv = np.unique(mk[nz], return_inverse=True)
    inv = inv.reshape(-1)
    total = np.bincount(inv)
    out = np.bincount(inv[~o.astype(bool, copy=False)[nz]], minlength=total.size)
    return ids[out >= share * total]
