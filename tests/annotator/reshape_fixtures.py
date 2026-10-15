"""Shared synthetic fixtures for the Reshape engine tests.

Deliberately tiny (12 x 40 x 32) so the whole suite stays fast, and
deliberately *noisy*: the arbiter's rule is a percentile of the target's own
elevation, so a fixture with a perfectly flat interior would make every
threshold degenerate and prove nothing.
"""

from __future__ import annotations

import numpy as np

SHAPE = (12, 40, 32)
NOISE = 0.05
INTERIOR = 0.5

# Raw grey levels behind the relative numbers used throughout the suite.
#
# Every threshold in the arbiter is RELATIVE to the crop's normalised range --
# `build_elevation` ends in `norm()`, which maps the 1st..99.5th percentile of
# the CROP onto [0, 1].  So a "contrast of 0.12" is 0.12 of that range, not an
# absolute intensity, and it does not transfer to another dataset unchanged.
#
# On this fixture, in raw units before normalisation:
#     background            0.00
#     cell interior         0.50   (INTERIOR)
#     interior noise sigma  0.05   (NOISE)
#     wall plane            0.50 + contrast
# so the recorded sensitivity floor of contrast 0.12 is a wall 0.62 against a
# 0.50 interior -- 2.4 noise sigmas of separation.  Quoted as an SNR rather
# than a grey level, that is the number that transfers.


def two_columns(contrast, *, noise=NOISE, seed=0, interior=INTERIOR, radius=8.0,
                sep=11.0, shape=SHAPE, z_attenuate=False):
    """Two touching columns merged into ONE label, with a membrane wall between.

    Returns ``(labels, guide, membrane, region)``.

    * ``labels``    - one label (id 7) covering both columns.
    * ``guide``     - a trusted second label layer: one seed voxel column per cell.
    * ``membrane``  - interior at ``interior``, the separating plane at
      ``interior + contrast``, plus gaussian ``noise`` inside the cell.
    * ``contrast=0`` is the "one dividing cell" case: no wall at all.
    * ``z_attenuate`` multiplies each z-slice by a depth ramp, reproducing the
      attenuation that makes ``zdepth_correct`` mandatory under the arbiter.
    """
    nz, ny, nx = shape
    _z, y, x = np.indices(shape).astype(float)
    cy, cx = ny / 2.0, nx / 2.0
    r2 = float(radius) ** 2
    d1 = (y - (cy - sep / 2)) ** 2 + (x - cx) ** 2
    d2 = (y - (cy + sep / 2)) ** 2 + (x - cx) ** 2
    region = (d1 <= r2) | (d2 <= r2)

    labels = np.zeros(shape, np.int32)
    labels[region] = 7

    guide = np.zeros(shape, np.int32)
    guide[:, int(round(cy - sep / 2)), int(round(cx))] = 1
    guide[:, int(round(cy + sep / 2)), int(round(cx))] = 2
    guide[~region] = 0

    rng = np.random.default_rng(seed)
    mem = np.zeros(shape, np.float32)
    mem[region] = float(interior)
    wall = (np.abs(y - cy) <= 0.6) & region
    mem[wall] = float(interior) + float(contrast)
    mem = mem + rng.normal(0.0, float(noise), shape).astype(np.float32) * region
    mem = np.clip(mem, 0, None)

    if z_attenuate:
        ramp = np.linspace(1.0, 0.25, nz, dtype=np.float32)
        mem = mem * ramp[:, None, None]

    return labels, guide, mem, region


def stacked_in_z(contrast, *, noise=NOISE, seed=1, interior=INTERIOR,
                 shape=(24, 24, 24)):
    """One label made of two blocks STACKED IN Z, wall on the z-interface.

    This is the orientation that an interface band with no z-extent gets wrong:
    the two pieces meet across z only, so if the structuring element cannot
    reach a z-neighbour they never appear to touch and the interface goes
    unmeasured.

    The geometry is deliberately roomy -- a 20x16x16 box with two 3x4x4 seed
    blobs well clear of the wall plane -- so the watershed produces a CLEAN
    two-piece z split with an interface of ~800 band voxels.  An earlier,
    tighter version of this fixture yielded 5-7 band voxels of ragged in-plane
    contact, which made every verdict read off it meaningless.
    """
    nz, ny, nx = shape
    z, _y, _x = np.indices(shape).astype(float)
    region = np.zeros(shape, bool)
    region[2:22, 4:20, 4:20] = True

    labels = np.zeros(shape, np.int32)
    labels[region] = 7

    guide = np.zeros(shape, np.int32)
    guide[4:7, 10:14, 10:14] = 1
    guide[17:20, 10:14, 10:14] = 2
    guide[~region] = 0

    rng = np.random.default_rng(seed)
    mem = np.zeros(shape, np.float32)
    mem[region] = float(interior)
    wall = (np.abs(z - (nz / 2.0 - 0.5)) <= 0.6) & region
    mem[wall] = float(interior) + float(contrast)
    mem = mem + rng.normal(0.0, float(noise), shape).astype(np.float32) * region
    return labels, guide, np.clip(mem, 0, None), region


def solid_cell(label_id=7, shape=SHAPE):
    """One solid rectangular label with a distinct neighbour beside it."""
    labels = np.zeros(shape, np.int32)
    labels[:, 8:20, 6:26] = label_id
    labels[:, 24:34, 6:26] = 99          # the neighbour that must never be touched
    return labels


def hollow_shell(label_id=7, r_in=12, r_out=14, n=40):
    """A closed spherical SHELL -- the shape the geometry finder breaks on.

    Nothing about a shell is wrong as a segmentation: a cell imaged only
    at its membrane, or a nucleus whose interior washed out, looks
    exactly like this.  What it does to ``seeds_distance_maxima`` is
    make every point of the shell an equal local maximum of the interior
    distance transform, so the h-maxima finder returns one marker per
    bump of the discretised surface -- dozens of them, and the same
    number at every ``h``, because there is no thickness variation for
    ``h`` to bite on.  Measured 2026-09-20: r 12-14 in a 40-cube is 6.9 %
    fill and yields 68 markers with ``walls="skeleton"``, 98 with
    ``walls="distance"``.
    """
    import numpy as np

    shape = (n, n, n)
    z, y, x = np.indices(shape)
    c = np.array(shape) / 2.0
    d = np.sqrt((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2)
    lab = np.zeros(shape, np.int32)
    lab[(d >= r_in) & (d <= r_out)] = label_id
    return lab


def dumbbell(label_id=7, shape=(24, 40, 24)):
    """Two lobes on a bar -- the shape the finder is FOR.

    Here to prove the shell cap does not touch an ordinary two-lobed
    object: 2 markers, nowhere near the cap.
    """
    import numpy as np

    lab = np.zeros(shape, np.int32)
    z, y, x = np.indices(shape)
    for cy in (12, 28):
        d = np.sqrt((z - 12) ** 2 + (y - cy) ** 2 + (x - 12) ** 2)
        lab[d <= 7] = label_id
    lab[10:15, 12:28, 10:15] = label_id
    return lab
