"""Nuclei ▸ "At the markers I place", one click, one nucleus: the engine.

``segment_at_markers`` takes ``split``: ``"off"`` (the default) is the
markers as they were -- a click claims its whole bright piece --; ``"always"``
splits every clicked piece at its waists, so one click on a nucleus in a
touching group gives that nucleus; ``"single"`` splits only the pieces that
hold exactly one claimed click.

Spheres are drawn in microns from radii given in xy voxels.  A touching
pair has radii ``r1 >= r2`` and a neck (the radius of the disc where the two
surfaces meet) ``rho``; the centres lie ``sqrt(r1^2 - rho^2) +
sqrt(r2^2 - rho^2)`` apart.  "Just that nucleus" means at least 0.8 of it
and at most 0.2 of the object outside it.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest

from packages.epicycle_annotator import add_segment as AS

SP_ISO = (1.0, 1.0, 1.0)
SP_ANI = (2.0, 0.5, 0.5)
SP_THICK = (5.0, 0.3, 0.3)
SP_FINE_Z = (0.2, 0.5, 0.5)

#: "just that nucleus"
COVER = 0.8
OUTSIDE = 0.2


# --------------------------------------------------------------------------- #
# synthetic nuclei
# --------------------------------------------------------------------------- #
def _spheres(spacing, centres_um, radii_um, pad=3):
    """Spheres from physical centres and radii, in the smallest box that
    holds them plus ``pad`` voxels: ``(shape, masks, centres in voxels)``."""
    sp = np.asarray(spacing, float)
    cs = np.asarray(centres_um, float)
    rs = np.asarray(radii_um, float)
    lo = np.floor((cs - rs[:, None]).min(axis=0) / sp) - pad
    hi = np.ceil((cs + rs[:, None]).max(axis=0) / sp) + pad + 1
    shape = tuple(int(v) for v in hi - lo)
    cv = cs / sp - lo
    zz, yy, xx = np.indices(shape).astype(float)
    masks = []
    for c, r in zip(cv, rs):
        d2 = (((zz - c[0]) * sp[0]) ** 2 + ((yy - c[1]) * sp[1]) ** 2
              + ((xx - c[2]) * sp[2]) ** 2)
        masks.append(d2 <= r * r)
    return shape, masks, cv


def _row(spacing, radii_vox, neck, axis=2, pad=3):
    """Spheres in a row along ``axis`` (0 = z, 2 = x), radii in xy voxels;
    each neck is ``neck`` times the smaller of its two radii.  The row sits
    a little off the voxel grid."""
    sxy = float(spacing[-1])
    r_um = [r * sxy for r in radii_vox]
    pos = [0.0]
    for r1, r2 in zip(r_um, r_um[1:]):
        rho = neck * min(r1, r2)
        pos.append(pos[-1] + np.sqrt(r1 * r1 - rho * rho)
                   + np.sqrt(r2 * r2 - rho * rho))
    cs = np.zeros((len(r_um), 3))
    cs[:, axis] = pos
    cs += np.array([0.13, 0.21, 0.17]) * np.asarray(spacing, float)
    return _spheres(spacing, cs, r_um, pad=pad)


def _image(mask, value=1.0):
    img = np.zeros(mask.shape, np.float64)
    img[mask] = value
    return img


def _vox(c):
    return tuple(int(round(float(v))) for v in c)


def _seg(fg, points, *, spacing, existing=None, radius_um=1.0,
         threshold=0.5, image=None, **kw):
    img = _image(fg) if image is None else image
    if existing is None:
        existing = np.zeros(img.shape, np.int32)
    return AS.segment_at_markers(img, existing, points, radius_um=radius_um,
                                 spacing=spacing, threshold=threshold, **kw)


def _just(out, k, nucleus):
    """(the share of ``nucleus`` in object ``k``, the share of object ``k``
    outside ``nucleus``)."""
    obj = out == k
    n = int(obj.sum())
    cover = float((obj & nucleus).sum()) / float(nucleus.sum())
    outside = float((obj & ~nucleus).sum()) / n if n else 1.0
    return cover, outside


def _n_pieces(out, k):
    from scipy import ndimage as ndi

    return int(ndi.label(out == k)[1])


def _one_piece_per_id(out):
    for k in np.unique(out):
        if k:
            n = _n_pieces(out, int(k))
            assert n == 1, (int(k), n)


def _click(cv, which, other, spacing, r_vox, where, axis):
    """A click on nucleus ``which``: its centre, half a radius sideways (in
    y), or half a radius away from its neighbour along the row."""
    c = np.asarray(cv[which], float).copy()
    if where == "side":
        c[1] += 0.5 * r_vox
    elif where == "far":
        away = np.sign(c[axis] - np.asarray(cv[other], float)[axis])
        c[axis] += away * 0.5 * r_vox * spacing[-1] / spacing[axis]
    return _vox(c)


# --------------------------------------------------------------------------- #
# "off" is today, byte for byte
# --------------------------------------------------------------------------- #
def _off_fixture(name):
    """(fg image, existing, points, spacing) of the four pinned fixtures."""
    if name == "touching pair":
        shape, (a, b), cv = _row(SP_ISO, [10, 10], 0.5)
        pts = [_vox(cv[0] + (0, 3, -4)), _vox(cv[1] + (2, 0, 5))]
        return _image(a | b), np.zeros(shape, np.int32), pts, SP_ISO
    if name == "three in a row":
        shape, ms, cv = _row(SP_ANI, [10, 10, 10], 0.5)
        return (_image(ms[0] | ms[1] | ms[2]), np.zeros(shape, np.int32),
                [_vox(cv[1])], SP_ANI)
    if name == "two clicks in one nucleus":
        shape, (a,), cv = _spheres(SP_ISO, [(0.3, 0.1, 0.2)], [8.0])
        c = cv[0]
        pts = [_vox(c + (0, 2, -3)), _vox(c + (1, -1, 4))]
        return _image(a), np.zeros(shape, np.int32), pts, SP_ISO
    if name == "a dim and a kept click":
        shape = (12, 20, 30)
        img = np.zeros(shape, np.float64)
        img[3:9, 5:15, 5:25] = 1.0
        existing = np.zeros(shape, np.int32)
        existing[2:10, 4:16, 4:12] = 7
        pts = [(6, 10, 8), (6, 18, 27), (6, 10, 20)]
        return img, existing, pts, SP_ISO
    raise KeyError(name)


#: today's output of each fixture (measured before ``split`` existed): the
#: SHA-256 of ``pieces.tobytes()``, the voxels per id, and ``why``
OFF_PINS = {
    "touching pair": (
        "cf29205a0b16dbeeacb70d6447b1da280d0c453677b810432dd8f2351f68e888",
        {1: 4101, 2: 4175}, {}),
    "three in a row": (
        "39a0d48ff1e35000ee975a2adbeb9282657b9b8420e944d5a415ed19516051c8",
        {1: 3140}, {}),
    "two clicks in one nucleus": (
        "8f0439cd1d442513dead2eea72fa0e592f1cdc5f8928ecd6477c97ccad093d62",
        {1: 1204, 2: 942}, {}),
    "a dim and a kept click": (
        "43c1640098a85ba01850f46dab47f39ffcd071316b563d2adf0125477f9ed3cb",
        {3: 780}, {1: "kept", 2: "dim"}),
}


def _pin_of(pieces, why):
    ids, counts = np.unique(pieces[pieces > 0], return_counts=True)
    return (hashlib.sha256(pieces.tobytes()).hexdigest(),
            {int(i): int(n) for i, n in zip(ids, counts)}, dict(why))


@pytest.mark.parametrize("name", list(OFF_PINS))
def test_off_is_today_byte_for_byte(name):
    img, existing, pts, sp = _off_fixture(name)
    kw = dict(radius_um=1.0, spacing=sp, threshold=0.5)
    plain = AS.segment_at_markers(img, existing, pts, **kw)
    off = AS.segment_at_markers(img, existing, pts, split="off", **kw)
    assert plain[0].dtype == np.int32
    assert np.array_equal(plain[0], off[0]) and plain[1] == off[1]
    assert off[0].dtype == plain[0].dtype
    assert _pin_of(*off) == OFF_PINS[name]
    assert _pin_of(*plain) == OFF_PINS[name]


def test_an_unknown_mode_is_refused():
    img, existing, pts, sp = _off_fixture("touching pair")
    with pytest.raises(ValueError):
        AS.segment_at_markers(img, existing, pts, radius_um=1.0, spacing=sp,
                              threshold=0.5, split="sideways")
    for bad in (0.0, -2.5, float("nan"), True, np.True_, "5", "2.5"):
        with pytest.raises(ValueError):
            AS.segment_at_markers(img, existing, pts, radius_um=1.0,
                                  spacing=sp, threshold=0.5, split="always",
                                  peak_spacing_um=bad)
    # "off" never reads the Spacing
    off = AS.segment_at_markers(img, existing, pts, radius_um=1.0,
                                spacing=sp, threshold=0.5, split="off",
                                peak_spacing_um=0.0)
    assert _pin_of(*off) == OFF_PINS["touching pair"]
    assert AS.MARKER_SPLIT_MODES == ("off", "single", "always")
    assert AS.ADD_MARKER_SPACING_DEFAULT == 5
    assert AS.MARKER_SPLIT_SHARE == 0.8
    assert AS.MARKER_GAUSS_TRUNCATE == 4.0


# --------------------------------------------------------------------------- #
# one click on one of two touching nuclei
# --------------------------------------------------------------------------- #
PAIR_FLOOR_CASES = [
    (sp, ratio, which, where)
    for sp in (SP_ISO, SP_ANI)
    for ratio in (1.0, 0.7)
    for which in (0, 1)
    for where in ("centre", "side", "far")
]


@pytest.mark.parametrize("sp, ratio, which, where", PAIR_FLOOR_CASES)
def test_one_click_on_a_touching_pair_along_x_gives_that_nucleus(
        sp, ratio, which, where):
    radii = [10, 10 * ratio]
    _shape, ms, cv = _row(sp, radii, 0.5, axis=2)
    pt = _click(cv, which, 1 - which, sp, radii[which], where, 2)
    assert ms[which][pt]
    for mode in ("always", "single"):
        out, why = _seg(ms[0] | ms[1], [pt], spacing=sp, split=mode)
        assert why == {}
        cover, outside = _just(out, 1, ms[which])
        assert cover >= COVER and outside <= OUTSIDE, (mode, cover, outside)
        _one_piece_per_id(out)
    # unticked: the whole bright piece, as before
    out, _why = _seg(ms[0] | ms[1], [pt], spacing=sp, split="off")
    assert np.array_equal(out == 1, ms[0] | ms[1])


#: MEASURED at the build, not assumed: (cover floor, outside ceiling) per
#: case, a little looser than measured.  Pairs stacked in z on thick slices
#: are a known limit -- an unequal pair at z 2 µm and every pair at z 5 µm
#: come back as one object -- and along x at z 5 µm the rule holds.
MEASURED_PAIRS = {
    # (spacing, axis, ratio, which): (cover floor, outside ceiling)
    (SP_ISO, 0, 1.0, 0): (0.97, 0.01),
    (SP_ISO, 0, 1.0, 1): (0.98, 0.01),
    (SP_ISO, 0, 0.7, 0): (0.98, 0.01),
    (SP_ISO, 0, 0.7, 1): (0.97, 0.01),
    (SP_ANI, 0, 1.0, 0): (0.98, 0.01),
    (SP_ANI, 0, 1.0, 1): (0.98, 0.01),
    (SP_ANI, 0, 0.7, 0): (0.98, 0.27),     # measured 1.0 / 0.252: one object
    (SP_ANI, 0, 0.7, 1): (0.98, 0.76),     # measured 1.0 / 0.748: one object
    (SP_THICK, 2, 1.0, 0): (0.96, 0.02),
    (SP_THICK, 2, 1.0, 1): (0.97, 0.01),
    (SP_THICK, 2, 0.7, 0): (0.98, 0.01),
    (SP_THICK, 2, 0.7, 1): (0.97, 0.01),
    (SP_THICK, 0, 1.0, 0): (0.98, 0.50),   # measured 1.0 / 0.492: one object
    (SP_THICK, 0, 1.0, 1): (0.98, 0.52),   # measured 1.0 / 0.509: one object
    (SP_THICK, 0, 0.7, 0): (0.98, 0.35),   # measured 1.0 / 0.334: one object
    (SP_THICK, 0, 0.7, 1): (0.98, 0.68),   # measured 1.0 / 0.666: one object
}


@pytest.mark.parametrize("key", list(MEASURED_PAIRS))
def test_measured_pairs_stacked_in_z_and_on_thick_slices(key):
    sp, axis, ratio, which = key
    floor, ceiling = MEASURED_PAIRS[key]
    radii = [10, 10 * ratio]
    _shape, ms, cv = _row(sp, radii, 0.5, axis=axis)
    for where in ("centre", "far"):
        pt = _click(cv, which, 1 - which, sp, radii[which], where, axis)
        out, why = _seg(ms[0] | ms[1], [pt], spacing=sp, split="always")
        assert why == {}
        cover, outside = _just(out, 1, ms[which])
        assert cover >= floor and outside <= ceiling, (where, cover, outside)
        _one_piece_per_id(out)


# --------------------------------------------------------------------------- #
# rows of touching nuclei, lone nuclei, waists
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sp", [SP_ISO, SP_ANI])
@pytest.mark.parametrize("n, which", [(3, 1), (3, 0), (4, 1), (4, 3),
                                      (5, 2), (5, 4)])
def test_one_click_on_a_row_of_touching_nuclei_gives_that_nucleus(n, which,
                                                                  sp):
    """Rows of 3-5 (r 10, necks 0.5 r), clicked in the middle or at an end."""
    _shape, ms, cv = _row(sp, [10] * n, 0.5)
    fg = np.logical_or.reduce(ms)
    out, why = _seg(fg, [_vox(cv[which])], spacing=sp, split="always")
    assert why == {}
    cover, outside = _just(out, 1, ms[which])
    assert cover >= COVER and outside <= OUTSIDE, (cover, outside)
    _one_piece_per_id(out)


def test_a_five_lobed_lone_nucleus_clicked_at_an_end_stays_whole():
    """Beads r 8 in a row with necks 0.85 r: waists too shallow to cut."""
    _shape, ms, cv = _row(SP_ISO, [8] * 5, 0.85)
    fg = np.logical_or.reduce(ms)
    for which in (0, 4):
        out, why = _seg(fg, [_vox(cv[which])], spacing=SP_ISO, split="always")
        assert why == {}
        cover, outside = _just(out, 1, fg)
        assert cover >= 0.9 and outside == 0.0, (which, cover)


def _ellipsoid(spacing, radii_vox):
    """A lone ellipsoid, radii (z, y, x) in xy voxels."""
    sp = np.asarray(spacing, float)
    r_um = np.asarray(radii_vox, float) * sp[-1]
    shape = tuple(int(v) for v in np.ceil(2 * r_um / sp) + 7)
    c = (np.asarray(shape) - 1) / 2.0 + np.array([0.13, 0.21, 0.17])
    zz, yy, xx = np.indices(shape).astype(float)
    d = sum((((g - cc) * s) / r) ** 2
            for g, cc, s, r in zip((zz, yy, xx), c, sp, r_um))
    return d <= 1.0, c


@pytest.mark.parametrize("sp, radii", [
    (SP_ISO, (10, 10, 10)), (SP_ISO, (10, 10, 17)), (SP_ISO, (17, 10, 10)),
    (SP_ANI, (10, 10, 10)), (SP_ANI, (10, 10, 17))])
@pytest.mark.parametrize("where", ["centre", "edge", "near miss"])
def test_a_lone_nucleus_is_not_cut(sp, radii, where):
    fg, c = _ellipsoid(sp, radii)
    pt = np.asarray(_vox(c))
    if where != "centre":
        # along x from the middle: the last voxel on the nucleus, or the
        # first one past it (its ball still touches the nucleus)
        x = pt[2]
        while fg[pt[0], pt[1], x + 1]:
            x += 1
        pt[2] = x if where == "edge" else x + 1
    pt = tuple(int(v) for v in pt)
    assert fg[pt] == (where != "near miss")
    out, why = _seg(fg, [pt], spacing=sp, split="always")
    assert why == {}
    cover, outside = _just(out, 1, fg)
    assert cover >= 0.9 and outside == 0.0, cover
    _one_piece_per_id(out)


def test_two_nuclei_with_no_waist_stay_one_object():
    """Centres 0.6 r apart: nothing to cut at."""
    _shape, (a, b), cv = _spheres(SP_ISO, [(0.1, 0.2, 0.3),
                                           (0.1, 0.2, 6.3)], [10.0, 10.0])
    fg = a | b
    for which in (0, 1):
        out, why = _seg(fg, [_vox(cv[which])], spacing=SP_ISO,
                        split="always")
        assert why == {}
        cover, _outside = _just(out, 1, fg)
        assert cover >= 0.9, (which, cover)


def test_two_clicks_in_one_lone_nucleus_give_two_objects():
    img, existing, pts, sp = _off_fixture("two clicks in one nucleus")
    for mode in ("always", "single"):
        out, why = _seg(img > 0, pts, spacing=sp, existing=existing,
                        split=mode)
        assert why == {}
        assert (out == 1).any() and (out == 2).any(), mode
        _one_piece_per_id(out)
        assert not out[img == 0].any()


def test_two_clicks_in_the_middle_nucleus_of_a_row_cover_it():
    _shape, ms, cv = _row(SP_ISO, [10, 10, 10], 0.5)
    fg = ms[0] | ms[1] | ms[2]
    c = cv[1]
    pts = [_vox(c + (0, 0, -3)), _vox(c + (0, 2, 3))]
    out, why = _seg(fg, pts, spacing=SP_ISO, split="always")
    assert why == {}
    assert (out == 1).any() and (out == 2).any()
    _one_piece_per_id(out)
    both = np.where(out > 0, 1, 0)
    cover, outside = _just(both, 1, ms[1])
    assert cover >= COVER and outside <= OUTSIDE, (cover, outside)


@pytest.mark.parametrize("sp", [SP_ISO, SP_ANI])
def test_a_row_touching_kept_objects_never_writes_on_them(sp):
    _shape, ms, cv = _row(sp, [10, 10, 10], 0.5)
    fg = ms[0] | ms[1] | ms[2]
    existing = np.zeros(fg.shape, np.int32)
    x2 = int(round(cv[2][2]))
    existing[:, :, x2 + 4:] = 9            # a kept object over the third
    existing[:, :4, :] = 4                 # and one along a side
    out, why = _seg(fg, [_vox(cv[1])], spacing=sp, existing=existing,
                    split="always")
    assert why == {}
    assert not out[existing > 0].any()
    cover, outside = _just(out, 1, ms[1] & (existing == 0))
    assert cover >= COVER and outside <= OUTSIDE, (cover, outside)
    _one_piece_per_id(out)


def test_a_ball_over_two_pieces_stays_in_the_piece_it_claims():
    """The click sits in the dim gap: the piece holding most of its ball
    (a touching pair) is claimed and split; the thin rod is left out."""
    _shape, (a, b), cv = _row(SP_ISO, [10, 10], 0.5, pad=6)
    fg = a | b
    c = _vox(cv[0])
    x_lo = int(np.argwhere(a[c[0], c[1]]).min())
    rod = np.zeros(fg.shape, bool)
    rod[c[0], c[1], 0:x_lo - 1] = True
    pt = (c[0], c[1], x_lo - 1)                     # the gap voxel
    assert not fg[pt] and not rod[pt]
    out, why = _seg(fg | rod, [pt], spacing=SP_ISO, radius_um=1.5,
                    split="always")
    assert why == {}
    assert not out[rod].any()
    assert (out == 1).any() and not out[~fg].any()
    cover, outside = _just(out, 1, a)
    assert cover >= COVER and outside <= OUTSIDE, (cover, outside)


#: MEASURED at the build (recorded, not required): a click right on the
#: saddle of a touching pair -- (the nucleus it gives, its cover floor, its
#: outside ceiling)
SADDLE_PIN = (1, 0.98, 0.02)       # measured 0.996 / 0.005


def test_a_click_on_the_saddle_and_one_near_the_bright_edge():
    _shape, (a, b), cv = _row(SP_ISO, [10, 10], 0.5)
    fg = a | b
    neck = _vox((np.asarray(cv[0]) + np.asarray(cv[1])) / 2.0)
    assert fg[neck] and a[neck] and b[neck]
    out, why = _seg(fg, [neck], spacing=SP_ISO, split="always")
    assert why == {}
    _one_piece_per_id(out)
    which, floor, ceiling = SADDLE_PIN
    cover, outside = _just(out, 1, (a, b)[which])
    assert cover >= floor and outside <= ceiling, (cover, outside)
    # near the far surface of the first nucleus
    c = _vox(cv[0])
    x = int(np.argwhere(a[c[0], c[1]]).min()) + 1
    out, why = _seg(fg, [(c[0], c[1], x)], spacing=SP_ISO, split="always")
    assert why == {}
    _one_piece_per_id(out)
    cover, outside = _just(out, 1, a)
    assert cover >= COVER and outside <= OUTSIDE, (cover, outside)


#: MEASURED at the build (recorded, not required to split): a small nucleus
#: (r 3.5) on a big one (r 5), neck 0.5 of the small one, isotropic -- per
#: clicked nucleus: (cover floor, outside ceiling)
SMALL_BIG_PIN = {
    0: (0.98, 0.27),     # measured 1.0 / 0.251: the big one takes both
    1: (0.95, 0.02),     # measured 0.967 / 0.0: the small one alone
}


@pytest.mark.parametrize("which", [0, 1])
def test_a_small_nucleus_beside_a_big_one_is_pinned(which):
    _shape, ms, cv = _row(SP_ISO, [5, 3.5], 0.5)
    out, why = _seg(ms[0] | ms[1], [_vox(cv[which])], spacing=SP_ISO,
                    split="always")
    assert why == {}
    _one_piece_per_id(out)
    floor, ceiling = SMALL_BIG_PIN[which]
    cover, outside = _just(out, 1, ms[which])
    assert cover >= floor and outside <= ceiling, (cover, outside)


def _rod(spacing, a_um, b_um, r_um, pad=3):
    """A round rod (a capsule) from ``a_um`` to ``b_um``, radius ``r_um``."""
    sp = np.asarray(spacing, float)
    a, b = np.asarray(a_um, float), np.asarray(b_um, float)
    lo = np.floor((np.minimum(a, b) - r_um) / sp) - pad
    hi = np.ceil((np.maximum(a, b) + r_um) / sp) + pad + 1
    shape = tuple(int(v) for v in hi - lo)
    pos = (np.indices(shape).astype(float).reshape(3, -1).T + lo) * sp
    ab = b - a
    t = np.clip(((pos - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    d = np.linalg.norm(pos - (a + t[:, None] * ab), axis=1)
    return (d <= r_um).reshape(shape), (a / sp - lo), (b / sp - lo)


#: MEASURED at the build (a recorded limit, nothing required): a rod 18 µm
#: long, radius 2 µm, rising 9 µm in z, at z 5 µm / y x 0.3 µm, clicked at
#: its lower end -- the share of the rod it gives
ROD_PIN = 0.33                     # measured 0.344


def test_a_rod_tilted_in_z_on_thick_slices_is_pinned():
    fg, a, _b = _rod(SP_THICK, (0.0, 0.0, 0.0), (9.0, 0.0, 15.6), 2.0)
    pt = _vox(a + np.array([0.0, 0.0, 2.0 / 0.3]))
    assert fg[pt]
    out, why = _seg(fg, [pt], spacing=SP_THICK, split="always")
    assert why == {}
    _one_piece_per_id(out)
    cover, outside = _just(out, 1, fg)
    assert outside == 0.0
    assert cover >= ROD_PIN, cover


def test_a_half_size_call_keeps_the_floor():
    """The pair at doubled spacing, Spacing given in microns of the
    full-size image (5 full-size voxels of 0.5 µm)."""
    half = (4.0, 1.0, 1.0)
    for ratio in (1.0, 0.7):
        radii = [5, 5 * ratio]                      # 5 µm and 3.5 µm
        _shape, ms, cv = _row(half, radii, 0.5, axis=2)
        for which in (0, 1):
            out, why = _seg(ms[0] | ms[1], [_vox(cv[which])], spacing=half,
                            split="always", peak_spacing_um=5 * 0.5)
            assert why == {}
            cover, outside = _just(out, 1, ms[which])
            assert cover >= COVER and outside <= OUTSIDE, (ratio, which,
                                                           cover, outside)


def test_the_spacing_is_in_microns():
    n, h = AS._marker_half_widths(None, SP_ANI)
    assert (n, h) == (5, (1, 5, 5))
    assert AS._marker_half_widths(2.5, (4.0, 1.0, 1.0)) == (2.5, (1, 2, 2))
    assert AS._marker_half_widths(2.5, SP_FINE_Z) == (5, (12, 5, 5))
    assert AS._marker_half_widths(None, SP_THICK) == (5, (1, 5, 5))
    assert AS._marker_half_widths(None, SP_ISO) == (5, (5, 5, 5))
    # 2 * 0.15 / 0.2 is 1.4999999999999998 in floats: meant as 1.5, which
    # rounds (half to even) to 2, not 1
    assert round(2.0 * 0.15 / 0.2) == 1
    assert AS._marker_half_widths(0.3, (0.2, 0.15, 0.15)) == (2, (2, 2, 2))
    # 2.5 µm at 0.5 µm xy is the default 5 voxels: the same call
    _shape, ms, cv = _row(SP_ANI, [10, 10, 10], 0.5)
    fg = ms[0] | ms[1] | ms[2]
    pts = [_vox(cv[1]), _vox(cv[2] + (0, 2, 3))]
    by_none = _seg(fg, pts, spacing=SP_ANI, split="always")
    by_um = _seg(fg, pts, spacing=SP_ANI, split="always",
                 peak_spacing_um=2.5)
    assert np.array_equal(by_none[0], by_um[0]) and by_none[1] == by_um[1]


def test_the_crop_margin():
    assert AS._marker_crop_margin(SP_ISO) == (5, 5, 5)
    assert AS._marker_crop_margin(SP_ANI) == (2, 5, 5)
    assert AS._marker_crop_margin(SP_FINE_Z) == (11, 5, 5)


@pytest.mark.parametrize("sp", [SP_ISO, SP_ANI, SP_THICK, SP_FINE_Z,
                                (4.0, 1.0, 1.0), (0.7, 0.3, 0.3),
                                (1.3, 0.21, 0.21), (0.35, 0.13, 0.13)])
def test_the_crop_margin_is_the_smoothing_s_reach_plus_one(sp):
    """Measured on scipy itself: an impulse smoothed as the split smooths
    the distance map reaches exactly the margin minus one voxel."""
    from scipy import ndimage as ndi

    imp = np.zeros((61, 61, 61))
    imp[30, 30, 30] = 1.0
    out = ndi.gaussian_filter(imp, sigma=tuple(sp[-1] / v for v in sp),
                              truncate=AS.MARKER_GAUSS_TRUNCATE)
    nz = np.argwhere(out != 0)
    reach = tuple(int(np.abs(nz[:, a] - 30).max()) for a in range(3))
    assert AS._marker_crop_margin(sp) == tuple(r + 1 for r in reach)


# --------------------------------------------------------------------------- #
# the thinning
# --------------------------------------------------------------------------- #
def _greedy(q, labs, n):
    keep = []
    for i in range(len(q)):
        if all(labs[j] != labs[i] or np.max(np.abs(q[i] - q[j])) > n + 1e-9
               for j in keep):
            keep.append(i)
    return np.asarray(keep, np.int64)


@pytest.mark.parametrize("seed", range(6))
def test_the_vectorised_thinning_is_the_greedy_loop(seed):
    rng = np.random.default_rng(seed)
    P = int(rng.integers(50, 1500))
    sp = [SP_ISO, SP_ANI, SP_THICK, (4.0, 1.0, 1.0), SP_FINE_Z][seed % 5]
    n = [5.0, 2.5, 3.0][seed % 3]
    vox = rng.integers(0, [30, 120, 120], size=(P, 3))
    if seed % 2:
        # pairs exactly n apart along a scaled axis
        step = int(n)
        vox[1::7] = vox[0:-1:7][: len(vox[1::7])] + (0, step, 0)
    q = vox * (np.asarray(sp) / sp[-1])
    labs = rng.integers(1, 6, size=P)
    got = AS._thin_peaks(q, labs, n)
    assert np.array_equal(got, _greedy(q, labs, n))


@pytest.mark.parametrize("seed", range(3))
def test_the_thinning_at_exactly_n_on_a_scaled_z_axis(seed):
    """z voxels 2.5 times the xy voxels and ``n`` 2.5: peaks one slice
    apart are exactly ``n`` apart and suppress each other."""
    rng = np.random.default_rng(100 + seed)
    sp = (2.5, 1.0, 1.0)
    vox = rng.integers(0, [12, 40, 40], size=(600, 3))
    vox[1::3] = vox[0:-1:3][: len(vox[1::3])] + (1, 0, 0)
    vox[2::3] = vox[0:-2:3][: len(vox[2::3])] + (0, 2, 2)
    q = vox * (np.asarray(sp) / sp[-1])
    labs = rng.integers(1, 3, size=len(q))
    got = AS._thin_peaks(q, labs, 2.5)
    assert np.array_equal(got, _greedy(q, labs, 2.5))
    # a pair exactly 2.5 apart: the second is dropped
    two = AS._thin_peaks(np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
                         np.array([1, 1]), 2.5)
    assert two.tolist() == [0]


def test_the_peak_order_is_highest_first_and_ties_keep_their_order():
    heights = np.array([1.0, 3.0, 2.0, 3.0, 1.0, 3.0])
    assert AS._peak_order(heights).tolist() == [1, 3, 5, 2, 0, 4]


# --------------------------------------------------------------------------- #
# the crop equals the whole box
# --------------------------------------------------------------------------- #
def _crop_case(spacing):
    """A clicked touching pair one voxel from the low x face and the low
    z face, an unclicked nucleus far away: ``(fg, points)``."""
    _shape, (a, b), cv = _row(spacing, [10, 10 * 0.7], 0.5, pad=1)
    big = tuple(n + 40 for n in a.shape)
    fg = np.zeros(big, bool)
    sl = tuple(slice(0, n) for n in a.shape)
    fg[sl] = a | b
    _s, (far,), _cf = _spheres(spacing, [(0, 0, 0)], [3.0 * spacing[-1]])
    end = tuple(slice(big[i] - far.shape[i], big[i]) for i in range(3))
    fg[end] |= far
    return fg, [_vox(cv[0]), _vox(cv[1] + (0, 1, 2))]


@pytest.mark.parametrize("sp", [SP_ISO, SP_ANI, SP_FINE_Z])
def test_the_crop_equals_the_whole_box(sp, monkeypatch):
    fg, pts = _crop_case(sp)
    for mode in ("always", "single"):
        cropped = _seg(fg, pts, spacing=sp, split=mode)
        with monkeypatch.context() as m:
            m.setattr(AS, "_marker_crop_margin",
                      lambda spacing, _s=fg.shape: _s)
            whole = _seg(fg, pts, spacing=sp, split=mode)
        assert np.array_equal(cropped[0], whole[0]), mode
        assert cropped[1] == whole[1]
        assert (cropped[0] > 0).any()


# --------------------------------------------------------------------------- #
# the three modes
# --------------------------------------------------------------------------- #
def _clumps():
    """Three separate clumps of touching nuclei in one box, one above the
    other in y: ``(fg, centres per clump, nuclei per clump)``."""
    rows = [_row(SP_ISO, [10, 10, 8], 0.5), _row(SP_ISO, [9, 9], 0.45),
            _row(SP_ISO, [8, 10, 10, 8], 0.5)]
    h = max(r[0][1] for r in rows)
    w = max(r[0][2] for r in rows)
    shape = (max(r[0][0] for r in rows), 3 * h + 2, w)
    fg = np.zeros(shape, bool)
    centres, nuclei = [], []
    for i, (s, ms, cv) in enumerate(rows):
        y0 = i * (h + 1)
        fg[:s[0], y0:y0 + s[1], :s[2]] |= np.logical_or.reduce(ms)
        centres.append([np.asarray(c) + (0, y0, 0) for c in cv])
        placed = []
        for m in ms:
            big = np.zeros(shape, bool)
            big[:s[0], y0:y0 + s[1], :s[2]] = m
            placed.append(big)
        nuclei.append(placed)
    return fg, centres, nuclei


def test_single_is_off_when_every_piece_holds_two_clicks():
    fg, cs, _nuc = _clumps()
    pts = [_vox(cs[0][0]), _vox(cs[0][2]), _vox(cs[1][0]), _vox(cs[1][1]),
           _vox(cs[2][1]), _vox(cs[2][1] + (0, 3, 0))]
    off = _seg(fg, pts, spacing=SP_ISO, split="off")
    single = _seg(fg, pts, spacing=SP_ISO, split="single")
    assert np.array_equal(off[0], single[0]) and off[1] == single[1]


def test_single_is_always_when_every_piece_holds_one_click():
    # the same code path both times: this pins only the dispatch; the next
    # test checks what "single" gives on its own terms
    fg, cs, _nuc = _clumps()
    pts = [_vox(cs[0][1]), _vox(cs[1][0]), _vox(cs[2][2])]
    always = _seg(fg, pts, spacing=SP_ISO, split="always")
    single = _seg(fg, pts, spacing=SP_ISO, split="single")
    assert np.array_equal(always[0], single[0]) and always[1] == single[1]
    assert not np.array_equal(always[0], _seg(fg, pts, spacing=SP_ISO)[0])


def test_single_gives_just_the_clicked_nucleus_in_one_click_pieces():
    """Checked against the nuclei themselves, not against "always"."""
    fg, cs, nuc = _clumps()
    picks = [(0, 1), (1, 0), (2, 2)]
    pts = [_vox(cs[i][j]) for i, j in picks]
    out, why = _seg(fg, pts, spacing=SP_ISO, split="single")
    assert why == {}
    _one_piece_per_id(out)
    for k, (i, j) in enumerate(picks, start=1):
        cover, outside = _just(out, k, nuc[i][j])
        assert cover >= COVER and outside <= OUTSIDE, (k, cover, outside)
    # and a piece holding two clicks keeps today's whole-piece flood
    pts2 = pts + [_vox(cs[1][1])]
    out2, _why = _seg(fg, pts2, spacing=SP_ISO, split="single")
    pair = nuc[1][0] | nuc[1][1]
    assert not np.isin(out2[pair], [0, 1, 3]).any()
    for k, (i, j) in ((1, picks[0]), (3, picks[2])):
        cover, outside = _just(out2, k, nuc[i][j])
        assert cover >= COVER and outside <= OUTSIDE, (k, cover, outside)


def test_single_is_off_and_always_piece_by_piece():
    from scipy import ndimage as ndi

    fg, cs, _nuc = _clumps()
    pts = [_vox(cs[0][1]), _vox(cs[1][0]), _vox(cs[1][1]), _vox(cs[2][2])]
    off = _seg(fg, pts, spacing=SP_ISO, split="off")[0]
    always = _seg(fg, pts, spacing=SP_ISO, split="always")[0]
    single = _seg(fg, pts, spacing=SP_ISO, split="single")[0]
    comp, _n = ndi.label(fg)
    two = comp == comp[pts[1]]               # the piece holding two clicks
    # the "always" half compares the same code path with itself (only the
    # composition is checked here); the "off" half is a real comparison
    want = np.where(two, off, always)
    assert np.array_equal(single, want)
    assert not np.array_equal(off, always)


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #
def _packed(shape, n, r_lo, r_hi, seed):
    """Random overlapping balls in a box: ``(fg, centres)``."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape).astype(float)
    fg = np.zeros(shape, bool)
    cs = []
    for _ in range(n):
        c = rng.uniform([2, 2, 2], np.asarray(shape) - 3)
        r = rng.uniform(r_lo, r_hi)
        fg |= ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r
        cs.append(c)
    return fg, cs


@pytest.mark.parametrize("sp", [SP_ISO, SP_ANI])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ids_one_piece_each_in_click_order_nothing_on_kept_or_unclicked(seed,
                                                                         sp):
    from scipy import ndimage as ndi

    fg, _cs = _packed((20, 60, 60), 45, 3.0, 6.0, seed)
    existing = np.zeros(fg.shape, np.int32)
    existing[:, :, 28:31] = 3
    rng = np.random.default_rng(seed + 10)
    on = np.argwhere(fg & (existing == 0))
    pts = [tuple(int(v) for v in on[i])
           for i in rng.choice(len(on), 12, replace=False)]
    comp, _n = ndi.label(fg & (existing == 0))
    clicked = np.isin(comp, [comp[p] for p in pts])
    for mode in ("always", "single"):
        out, why = _seg(fg, pts, spacing=sp, existing=existing,
                        split=mode)
        assert out.dtype == np.int32
        _one_piece_per_id(out)
        assert not out[existing > 0].any()
        assert not out[~clicked].any()
        for k, p in enumerate(pts, start=1):
            if k in why:
                continue
            # each click's object is numbered by its place in the list and
            # lies in the piece it claimed
            assert (out == k).any(), (mode, k)
            assert np.all(comp[out == k] == comp[p]), (mode, k)
        again = _seg(fg, pts, spacing=sp, existing=existing, split=mode)
        assert np.array_equal(out, again[0]) and why == again[1]


def test_smallest_object_is_decided_after_the_cut():
    """Smallest object at 1.3 times one nucleus: the whole pair ("off")
    is above it, the clicked nucleus alone ("always") is below it."""
    _shape, ms, cv = _row(SP_ISO, [10, 10], 0.5)
    fg = ms[0] | ms[1]
    floor = int(1.3 * int(ms[0].sum()))
    out, why = _seg(fg, [_vox(cv[0])], spacing=SP_ISO, split="off",
                    min_size=floor)
    assert why == {} and int((out == 1).sum()) >= floor
    out, why = _seg(fg, [_vox(cv[0])], spacing=SP_ISO, split="always",
                    min_size=floor)
    assert why == {1: "small"} and not out.any()


def test_dim_and_kept_clicks_are_decided_before_the_split():
    img, existing, pts, sp = _off_fixture("a dim and a kept click")
    out, why = _seg(img > 0, pts, spacing=sp, existing=existing,
                    split="always")
    assert why == {1: "kept", 2: "dim"}
    assert set(np.unique(out).tolist()) == {0, 3}


# --------------------------------------------------------------------------- #
# Stop
# --------------------------------------------------------------------------- #
def _cancelled():
    from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

    return ReshapeCancelled


def test_stop_inside_the_climbs_cancels_promptly(monkeypatch):
    ev = threading.Event()
    calls = []
    real = AS._climb

    def _climb(*a, **k):
        calls.append(1)
        ev.set()                                  # Stop pressed mid-climb
        return real(*a, **k)

    monkeypatch.setattr(AS, "_climb", _climb)
    fg, cs = _packed((20, 60, 60), 45, 3.0, 6.0, 0)
    pts = [_vox(c) for c in cs[:10] if fg[_vox(c)]]
    with pytest.raises(_cancelled()):
        _seg(fg, pts, spacing=SP_ISO, split="always", cancel_event=ev)
    assert len(calls) == 1


def _poll_log(monkeypatch, fire_at=None):
    """Record every Stop poll by the function that made it; with
    ``fire_at``, Stop is pressed just before that poll (1-based)."""
    import sys

    log = []
    ev = threading.Event()
    real = AS._poll_stop

    def poll(cancel_event):
        log.append(sys._getframe(1).f_code.co_name)
        if fire_at is not None and len(log) == fire_at:
            ev.set()
        return real(cancel_event)

    monkeypatch.setattr(AS, "_poll_stop", poll)
    return log, ev


def test_stop_inside_the_thinning_s_loop_cancels(monkeypatch):
    """Polls: on entry, at row 0, at row 4096 -- Stop pressed at the third
    is raised from inside the loop."""
    log, ev = _poll_log(monkeypatch, fire_at=3)
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 400, size=(5000, 3))
    with pytest.raises(_cancelled()):
        AS._thin_peaks(q, np.ones(5000, np.int64), 5.0, cancel_event=ev)
    assert log == ["_thin_peaks"] * 3


def test_stop_inside_the_merge_cancels(monkeypatch):
    ev = threading.Event()
    real = AS._grow_groups
    seen = []

    def _grow(*a, **k):
        seen.append(1)
        ev.set()
        return real(*a, **k)

    monkeypatch.setattr(AS, "_grow_groups", _grow)
    _shape, ms, cv = _row(SP_ISO, [8] * 5, 0.85)
    fg = np.logical_or.reduce(ms)
    with pytest.raises(_cancelled()):
        _seg(fg, [_vox(cv[0])], spacing=SP_ISO, split="always",
             cancel_event=ev)
    assert seen


def test_stop_after_a_guard_flood_cancels():
    """The one-piece guard polls Stop after each flood."""
    from scipy import ndimage as ndi

    ev = threading.Event()
    ev.set()
    _shape, (a, b), _cv = _row(SP_ISO, [10, 10], 0.5)
    mask = a | b
    edt = ndi.distance_transform_edt(mask)
    seeds = np.zeros(mask.shape, np.int32)
    seeds[a & ~b] = 1
    seeds[b & ~a] = 2
    with pytest.raises(_cancelled()):
        AS._guard_markers(edt, seeds, mask, {}, {}, {1, 2}, cancel_event=ev)


def _guard_case():
    """A random packed box (fixed seeds) where the merge leaves a click's
    object in two pieces, so the one-piece guard runs: ``(fg, points,
    spacing)``."""
    rng = np.random.default_rng(1079)
    sp = (4.0, 1.0, 1.0)
    shape = (10, 40, 40)
    n = int(rng.integers(10, 40))
    draw = np.random.default_rng(79)
    zz, yy, xx = np.indices(shape).astype(float)
    fg = np.zeros(shape, bool)
    for _ in range(n):
        c = draw.uniform([0, 0, 0], np.asarray(shape) - 1)
        r = draw.uniform(2.0, 6.0)
        fg |= (((zz - c[0]) * sp[0]) ** 2 + ((yy - c[1]) * sp[1]) ** 2
               + ((xx - c[2]) * sp[2]) ** 2) <= r * r
    on = np.argwhere(fg)
    k = int(rng.integers(1, 12))
    pts = [tuple(int(v) for v in on[i])
           for i in rng.choice(len(on), k, replace=False)]
    return fg, pts, sp


#: where Stop is pressed: (the function that polls, which of its polls)
STOP_AT = {
    "after the distance map": ("_split_clicked_pieces", 0),
    "after the smoothing": ("_split_clicked_pieces", 1),
    "after the flood": ("_split_at_waists", 0),
    "after a guard flood": ("_guard_markers", 0),
}


@pytest.mark.parametrize("where", list(STOP_AT))
def test_stop_through_the_call_at_each_poll(where, monkeypatch):
    fg, pts, sp = _guard_case()
    name, nth = STOP_AT[where]
    with monkeypatch.context() as m:
        log, ev = _poll_log(m)
        _seg(fg, pts, spacing=sp, split="always", cancel_event=ev)
    at = [i for i, n in enumerate(log) if n == name]
    assert len(at) > nth, (where, log)            # the guard did run
    fire = at[nth] + 1
    with monkeypatch.context() as m:
        log, ev = _poll_log(m, fire_at=fire)
        with pytest.raises(_cancelled()):
            _seg(fg, pts, spacing=sp, split="always", cancel_event=ev)
    assert len(log) == fire and log[-1] == name


# --------------------------------------------------------------------------- #
# RECORDS: today's results of the rule, pinned to be revisited, not
# requirements (a change to the distance map would move them)
# --------------------------------------------------------------------------- #
def _hollow(spacing, r_um=5.0, core=0.55):
    """A lone nucleus whose stain is a shell: the core (``core`` of the
    radius) is dim.  ``(image, shell, a click on its equator)``."""
    sp = np.asarray(spacing, float)
    shape = tuple(int(v) for v in np.ceil(2 * r_um / sp) + 7)
    c = (np.asarray(shape) - 1) / 2.0 + np.array([0.13, 0.21, 0.17])
    zz, yy, xx = np.indices(shape).astype(float)
    d = np.sqrt(((zz - c[0]) * sp[0]) ** 2 + ((yy - c[1]) * sp[1]) ** 2
                + ((xx - c[2]) * sp[2]) ** 2)
    outer = d <= r_um
    inner = d <= core * r_um
    img = np.where(outer, 1.0, 0.0)
    img[inner] = 0.1
    pt = (int(round(c[0])), int(round(c[1])),
          int(round(c[2] + 0.775 * r_um / sp[2])))
    return img, outer & ~inner, pt


#: RECORD: the share of a hollow lone nucleus one click keeps ("always");
#: measured 0.704 at z 2 µm, 1.0 at half size (4, 1, 1)
HOLLOW_RECORD = {(2.0, 0.5, 0.5): 0.704, (4.0, 1.0, 1.0): 1.0}


@pytest.mark.parametrize("sp", list(HOLLOW_RECORD))
def test_record_a_hollow_lone_nucleus(sp):
    img, shell, pt = _hollow(sp)
    assert shell[pt]
    out, why = _seg(img > 0.5, [pt], spacing=sp, image=img,
                    split="off")
    assert why == {} and np.array_equal(out == 1, shell)
    out, why = _seg(img > 0.5, [pt], spacing=sp, image=img, split="always")
    assert why == {}
    share = float(((out == 1) & shell).sum()) / float(shell.sum())
    assert abs(share - HOLLOW_RECORD[sp]) <= 0.02, share


#: RECORD: a lone 1.7 : 1 ellipsoid, blurred (sigma 1.5 voxels) with noise
#: (sd 0.05, seed 0), Otsu: the worst share of its bright piece over seven
#: clicks ("always"); measured 1.0
BLURRED_RECORD = 1.0


def test_record_a_blurred_noisy_ellipsoid():
    from scipy import ndimage as ndi

    shape = (27, 27, 41)
    c = (np.asarray(shape) - 1) / 2.0 + np.array([0.13, 0.21, 0.17])
    zz, yy, xx = np.indices(shape).astype(float)
    ell = (((zz - c[0]) / 10) ** 2 + ((yy - c[1]) / 10) ** 2
           + ((xx - c[2]) / 17) ** 2) <= 1
    rng = np.random.default_rng(0)
    img = (ndi.gaussian_filter(ell.astype(float), 1.5)
           + rng.normal(0, 0.05, shape))
    bright = img >= AS._resolve_threshold(img, "otsu")
    lab, _n = ndi.label(bright)
    z, y, x = (int(round(v)) for v in c)
    edge = int(np.argwhere(bright[z, y]).max())
    clicks = [(z, y, x), (z, y, x + 8), (z, y, x - 8), (z, y + 5, x),
              (z + 6, y, x), (z, y, x + 15), (z, y, edge)]
    worst = 1.0
    for pt in clicks:
        piece = lab == lab[pt]
        out, why = _seg(bright, [pt], spacing=SP_ISO, image=img,
                        threshold="otsu", split="always")
        assert why == {}
        worst = min(worst, float(((out == 1) & piece).sum()) / piece.sum())
    assert abs(worst - BLURRED_RECORD) <= 0.01, worst


def _foam(shape=(24, 80, 80), n=1200, r_lo=1.2, r_hi=2.5, seed=0):
    """Stain everywhere but ``n`` small dark holes (a rough, sponge-like
    piece), and a click at its thickest point near the middle."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    cs = rng.uniform([0, 0, 0], np.asarray(shape) - 1, size=(n, 3))
    rs = rng.uniform(r_lo, r_hi, n)
    holes = np.zeros(shape, bool)
    for c, r in zip(cs, rs):
        lo = [max(0, int(np.floor(c[a] - r))) for a in range(3)]
        hi = [min(shape[a], int(np.ceil(c[a] + r)) + 1) for a in range(3)]
        g = np.ogrid[tuple(slice(lo[a], hi[a]) for a in range(3))]
        box = tuple(slice(lo[a], hi[a]) for a in range(3))
        holes[box] |= sum((g[a] - c[a]) ** 2 for a in range(3)) <= r * r
    fg = ~holes
    fg[[0, -1], :, :] = False
    fg[:, [0, -1], :] = False
    fg[:, :, [0, -1]] = False
    dt = ndi.distance_transform_edt(fg)
    mid = np.unravel_index(int(np.argmax(dt[6:18, 30:50, 30:50])),
                           (12, 20, 20))
    return fg, tuple(int(v) + o for v, o in zip(mid, (6, 30, 30)))


#: RECORD: the voxels one click takes on the sponge-like piece, per Spacing
#: (µm at 1 µm voxels).  Not monotonic: a low Spacing lets the group creep
#: through many small basins (each step is compared with the member it
#: leaves from), a high one joins neighbours
FOAM_RECORD = {1: 41171, 2: 23216, 3: 24006, 5: 5458, 8: 10103, 12: 51813}


def test_record_the_spacing_on_a_sponge_like_piece():
    fg, pt = _foam()
    got = {}
    for um in FOAM_RECORD:
        out, why = _seg(fg, [pt], spacing=SP_ISO, split="always",
                        peak_spacing_um=float(um))
        assert why == {}
        got[um] = int((out == 1).sum())
    for um, want in FOAM_RECORD.items():
        assert abs(got[um] - want) <= 0.02 * want, (um, got)
    assert got[1] > 4 * got[5]

