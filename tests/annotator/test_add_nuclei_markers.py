"""Nuclei ▸ "At the markers I place": the engine and the preview, headless.

One click per nucleus.  The threshold decides what is stain exactly as
Anywhere in the box decides it; each click claims one bright piece; only
clicked pieces come back; touching nuclei clicked once each are split at
their neck wherever on the nucleus the click sits; a click whose whole ball
misses the stain makes nothing and says why.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator import add_segment as AS


# --------------------------------------------------------------------------- #
# synthetic nuclei
# --------------------------------------------------------------------------- #
def _balls(shape, centres, r_um, spacing):
    """One boolean ball per centre (physical radius ``r_um``)."""
    zz, yy, xx = np.indices(shape).astype(float)
    sp = [float(v) for v in spacing]
    out = []
    for c in centres:
        d2 = (((zz - c[0]) * sp[0]) ** 2 + ((yy - c[1]) * sp[1]) ** 2
              + ((xx - c[2]) * sp[2]) ** 2)
        out.append(d2 <= r_um * r_um)
    return out


def _image(mask, value=1.0):
    img = np.zeros(mask.shape, np.float64)
    img[mask] = value
    return img


def _seg(img, points, *, existing=None, spacing=(1.0, 1.0, 1.0),
         radius_um=1.0, threshold=0.5, **kw):
    if existing is None:
        existing = np.zeros(img.shape, np.int32)
    return AS.segment_at_markers(img, existing, points, radius_um=radius_um,
                                 spacing=spacing, threshold=threshold, **kw)


# two touching r = 10 spheres, isotropic: the neck is the plane x = 23
ISO_SHAPE = (30, 30, 50)
ISO_C1, ISO_C2 = (15, 15, 15), (15, 15, 31)
ISO_NECK = 23

# the same anisotropic: z voxels 2 µm, y / x voxels 0.5 µm, r = 6 µm
ANI_SP = (2.0, 0.5, 0.5)
ANI_SHAPE = (12, 30, 60)
ANI_C1, ANI_C2 = (6, 15, 18), (6, 15, 40)
ANI_NECK = 29


def _touching(shape, c1, c2, r, spacing, neck):
    a, b = _balls(shape, (c1, c2), r, spacing)
    fg = a | b
    x = np.indices(shape)[2]
    truth = np.where(x < neck, 1, 2) * fg
    return fg, truth


def _split_quality(out, fg, truth):
    """(share of nucleus 1 labelled 1, share of nucleus 2 labelled 2, the
    agreement over both)."""
    one = float((out[truth == 1] == 1).mean())
    two = float((out[truth == 2] == 2).mean())
    agree = float((out[fg] == truth[fg]).mean())
    return one, two, agree


# --------------------------------------------------------------------------- #
# touching nuclei, one click each, anywhere on them
# --------------------------------------------------------------------------- #
ISO_CASES = {
    "centres": [ISO_C1, ISO_C2],
    "first far off-centre": [(15, 15, 7), ISO_C2],
    "first off-centre sideways": [(15, 22, 15), ISO_C2],
    "both far off-centre": [(15, 15, 7), (15, 15, 40)],
    "first near the neck": [(15, 15, 20), ISO_C2],
}


@pytest.mark.parametrize("name", list(ISO_CASES))
def test_one_click_each_splits_touching_nuclei_at_the_neck(name):
    fg, truth = _touching(ISO_SHAPE, ISO_C1, ISO_C2, 10.0, (1, 1, 1),
                          ISO_NECK)
    out, why = _seg(_image(fg), ISO_CASES[name])
    assert why == {}
    one, two, agree = _split_quality(out, fg, truth)
    assert one >= 0.95 and two >= 0.95, (name, one, two)
    assert agree >= 0.98, (name, agree)


ANI_CASES = {
    "centres": [ANI_C1, ANI_C2],
    "first off-centre in z": [(9, 15, 18), ANI_C2],
    "first far off-centre in x": [(6, 15, 10), ANI_C2],
    "first off-centre sideways": [(6, 24, 18), ANI_C2],
    "first near the neck": [(6, 15, 26), ANI_C2],
}


@pytest.mark.parametrize("name", list(ANI_CASES))
def test_the_split_holds_with_anisotropic_voxels(name):
    fg, truth = _touching(ANI_SHAPE, ANI_C1, ANI_C2, 6.0, ANI_SP, ANI_NECK)
    out, why = _seg(_image(fg), ANI_CASES[name], spacing=ANI_SP)
    assert why == {}
    one, two, agree = _split_quality(out, fg, truth)
    assert one >= 0.95 and two >= 0.95, (name, one, two)
    assert agree >= 0.98, (name, agree)


def test_the_plain_seeded_watershed_fails_off_centre_and_the_climb_fixes_it():
    """The failure the peak climb exists for: seeded by its 1 µm ball alone,
    an off-centre click on one of two touching nuclei keeps a sliver (the
    neighbour's marker takes the whole distance hill)."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    from packages.epicycle_annotator.reshape.region import rasterise_points

    fg, truth = _touching(ISO_SHAPE, ISO_C1, ISO_C2, 10.0, (1, 1, 1),
                          ISO_NECK)
    pts = ISO_CASES["first far off-centre"]
    markers = np.where(fg, rasterise_points(pts, fg.shape, radius_um=1.0,
                                            spacing=(1, 1, 1)), 0)
    edt = ndi.distance_transform_edt(fg)
    plain = watershed(-edt, markers, mask=fg)
    assert int((plain == 1).sum()) < 50          # a sliver
    out, _why = _seg(_image(fg), pts)
    assert int((out == 1).sum()) > 3500           # the whole nucleus


def test_two_clicks_in_one_round_nucleus_split_it_near_the_middle():
    """Both clicks climb to the same peak, so neither takes it: the split
    stays between the two balls."""
    (one,) = _balls(ISO_SHAPE, (ISO_C1,), 10.0, (1, 1, 1))
    out, why = _seg(_image(one), [(15, 15, 10), (15, 15, 20)])
    assert why == {}
    n1, n2 = int((out == 1).sum()), int((out == 2).sum())
    total = int(one.sum())
    assert n1 + n2 == total
    assert min(n1, n2) >= 0.35 * total, (n1, n2)


def test_one_click_on_two_touching_nuclei_is_one_object():
    fg, _truth = _touching(ISO_SHAPE, ISO_C1, ISO_C2, 10.0, (1, 1, 1),
                           ISO_NECK)
    out, why = _seg(_image(fg), [(15, 15, 10)])
    assert why == {}
    assert set(np.unique(out).tolist()) == {0, 1}
    assert np.array_equal(out > 0, fg)


# --------------------------------------------------------------------------- #
# every new object is one piece, and a clicked nucleus has no hole
# --------------------------------------------------------------------------- #
def _ellipsoid(shape, centre, radii_um, spacing):
    zz, yy, xx = np.indices(shape).astype(float)
    d = sum((((g - c) * s) / r) ** 2 for g, c, s, r in
            zip((zz, yy, xx), centre, spacing, radii_um))
    return d <= 1.0


def _pieces_of(out, k):
    from scipy import ndimage as ndi

    return int(ndi.label(out == k)[1])


def _one_piece_each_and_no_hole(fg, out, n):
    """Every id 1..n present is ONE 6-connected piece, and every voxel of
    the clicked object is owned."""
    for k in range(1, n + 1):
        assert _pieces_of(out, k) <= 1, (k, _pieces_of(out, k))
    assert not (fg & (out == 0)).any(), int((fg & (out == 0)).sum())


#: a round nucleus, off the voxel grid, and a flat one at z 2 µm, y / x
#: 0.5 µm: the shapes where two clicks used to reach different tops
ROUND = ((24, 24, 24), (11.5, 12.0, 12.3), (5.0, 5.0, 5.0), (1.0, 1.0, 1.0))
FLAT = ((12, 40, 40), (5.6, 19.7, 20.2), (4.0, 8.0, 8.0), (2.0, 0.5, 0.5))


def test_two_clicks_in_one_round_nucleus_never_leave_an_object_in_two():
    """The reproduction: two asymmetric clicks in one round nucleus each
    reached a different top of its distance map, and click 1 came back as
    a 7-voxel sliver at its click plus 264 voxels around the core."""
    shape, centre, radii, sp = ROUND
    fg = _ellipsoid(shape, centre, radii, sp)
    out, why = _seg(_image(fg), [(14, 12, 16), (14, 13, 9)], spacing=sp)
    assert why == {}
    _one_piece_each_and_no_hole(fg, out, 2)


def test_two_clicks_in_one_flat_nucleus_never_leave_an_object_in_two():
    """The same at z 2 µm, y / x 0.5 µm on a flat nucleus."""
    shape, centre, radii, sp = FLAT
    fg = _ellipsoid(shape, centre, radii, sp)
    out, why = _seg(_image(fg), [(5, 26, 15), (5, 27, 12)], spacing=sp)
    assert why == {}
    _one_piece_each_and_no_hole(fg, out, 2)


@pytest.mark.parametrize("shape_name, n_pairs, seed",
                         [("round", 80, 1), ("flat", 80, 2)])
def test_any_two_clicks_in_one_nucleus_give_two_whole_pieces(shape_name,
                                                             n_pairs, seed):
    """A seeded batch of random click pairs in one nucleus (pairs whose
    second ball lies inside the first are skipped): every object is one
    piece and the nucleus has no unowned voxel."""
    shape, centre, radii, sp = ROUND if shape_name == "round" else FLAT
    fg = _ellipsoid(shape, centre, radii, sp)
    vox = np.argwhere(fg)
    rng = np.random.default_rng(seed)
    bad, tried = [], 0
    while tried < n_pairs:
        a, b = vox[rng.integers(len(vox))], vox[rng.integers(len(vox))]
        pts = [tuple(int(v) for v in a), tuple(int(v) for v in b)]
        out, why = _seg(_image(fg), pts, spacing=sp)
        if why:
            continue
        tried += 1
        if (max(_pieces_of(out, 1), _pieces_of(out, 2)) > 1
                or (fg & (out == 0)).any()):
            bad.append(pts)
    assert not bad, (len(bad), bad[:3])


# --------------------------------------------------------------------------- #
# unequal touching nuclei, one click each away from the neck
# --------------------------------------------------------------------------- #
def _power_truth(shape, centres, radii_um, spacing):
    """Spheres (centres in voxels, radii in µm) and which nucleus each voxel
    belongs to: inside both, the nearer by power distance (the plane through
    the circle where the two surfaces meet)."""
    zz, yy, xx = np.indices(shape).astype(float)
    sp = [float(v) for v in spacing]
    pw = np.stack([((zz - c[0]) * sp[0]) ** 2 + ((yy - c[1]) * sp[1]) ** 2
                   + ((xx - c[2]) * sp[2]) ** 2 - r * r
                   for c, r in zip(centres, radii_um)])
    fg = (pw <= 0).any(axis=0)
    return fg, np.where(fg, np.argmin(pw, axis=0) + 1, 0)


def _shares(out, truth, n=2):
    """The share of each nucleus given to its own click's object."""
    return [float((out[truth == k] == k).mean()) for k in range(1, n + 1)]


#: (spacing, box, centres in voxels, radii in µm, clicks, floor per nucleus):
#: one click on each, away from the neck, where a smoothed distance map and
#: a tolerance deeper than the neck made both clicks share one hill
UNEQUAL_CASES = {
    "two equal nuclei at z 5 um": (
        (5.0, 0.3, 0.3), (5, 61, 91), [(1.8, 30, 30), (1.8, 30, 60)],
        [6.0, 6.0], [(2, 30, 18), (2, 30, 70)], [0.95, 0.95]),
    "a small nucleus on a big one, isotropic": (
        (1.0, 1.0, 1.0), (23, 23, 35), [(13, 13, 13), (13, 13, 25)],
        [10.0, 6.0], [(13, 13, 7), (13, 13, 28)], [0.95, 0.85]),
    "a small nucleus on a big one at z 2 um": (
        (2.0, 0.5, 0.5), (15, 55, 65),
        [(6.74, 26.95, 26.95), (6.74, 26.95, 48.83)], [10.47, 4.24],
        [(4, 38, 14), (7, 28, 56)], [0.95, 0.95]),
}


@pytest.mark.parametrize("name", list(UNEQUAL_CASES))
def test_one_click_each_on_unequal_touching_nuclei_keeps_each_whole(name):
    sp, shape, centres, radii, clicks, floors = UNEQUAL_CASES[name]
    fg, truth = _power_truth(shape, centres, radii, sp)
    out, why = _seg(_image(fg), clicks, spacing=sp)
    assert why == {}
    got = _shares(out, truth)
    assert all(g >= f for g, f in zip(got, floors)), (name, got)


def _unequal_touching(rng, spacing):
    """Two unequal nuclei (radius 4-10 µm) side by side in the y / x plane,
    the centres 0.70-0.92 of the two radii apart, and one random click on
    each at least 2 µm from the other nucleus: ``(fg, truth, clicks)``."""
    sp = np.asarray(spacing, float)
    while True:
        r = rng.uniform(4.0, 10.0, 2)
        a = rng.uniform(0.0, 2.0 * np.pi)
        c1 = rng.uniform(0.0, 1.0, 3) * sp
        c2 = c1 + rng.uniform(0.70, 0.92) * r.sum() * np.array(
            [0.0, np.sin(a), np.cos(a)])
        cs = np.stack([c1, c2])
        lo = np.floor((cs - r[:, None]).min(axis=0) / sp) - 1
        hi = np.ceil((cs + r[:, None]).max(axis=0) / sp) + 2
        shape = tuple(int(v) for v in hi - lo)
        centres = cs / sp - lo
        fg, truth = _power_truth(shape, centres, r, sp)
        pos = np.indices(shape).astype(float) * sp[:, None, None, None]
        clicks = []
        for k, other in ((1, 1), (2, 0)):
            d = np.sqrt(sum((pos[a_] - centres[other][a_] * sp[a_]) ** 2
                            for a_ in range(3)))
            vox = np.argwhere((truth == k) & (d >= r[other] + 2.0))
            if not len(vox):
                break
            clicks.append(tuple(int(v) for v in vox[rng.integers(len(vox))]))
        if len(clicks) == 2:
            return fg, truth, clicks


#: per spacing: (seed, the trials out of 100 where a nucleus kept less than
#: 0.95 of itself before the hill grouping existed, as measured with that
#: seed); the batch may miss at most two more than that
UNEQUAL_BATCH = {
    (1.0, 1.0, 1.0): (11, 12),
    (2.0, 0.5, 0.5): (12, 12),
    (5.0, 0.3, 0.3): (13, 5),
}


@pytest.mark.parametrize("spacing", list(UNEQUAL_BATCH))
def test_unequal_touching_nuclei_clicked_once_each_stay_whole(spacing):
    seed, before = UNEQUAL_BATCH[spacing]
    rng = np.random.default_rng(seed)
    low = []
    for _ in range(100):
        fg, truth, clicks = _unequal_touching(rng, spacing)
        out, _why = _seg(_image(fg), clicks, spacing=spacing)
        got = _shares(out, truth)
        if min(got) < 0.95:
            low.append((clicks, [round(g, 3) for g in got]))
    assert len(low) <= before + 2, (len(low), before, low[:3])


def test_a_click_at_the_neck_s_surface_keeps_its_own_nucleus_whole():
    """One click each on two touching nuclei, the second right at the
    surface just past the neck (its ball reaches across the groove): each
    nucleus is one whole object, split at the neck."""
    fg, truth = _touching(ANI_SHAPE, ANI_C1, ANI_C2, 6.0, ANI_SP, ANI_NECK)
    out, why = _seg(_image(fg), [(5, 21, 11), (7, 20, 30)], spacing=ANI_SP)
    assert why == {}
    _one_piece_each_and_no_hole(fg, out, 2)
    one, two, agree = _split_quality(out, fg, truth)
    assert one >= 0.95 and two >= 0.95, (one, two)
    assert agree >= 0.98, agree


# --------------------------------------------------------------------------- #
# unclicked nuclei are left out
# --------------------------------------------------------------------------- #
def test_an_unclicked_bright_piece_comes_back_zero():
    a, b = _balls((20, 20, 40), ((10, 10, 9), (10, 10, 30)), 6.0, (1, 1, 1))
    out, why = _seg(_image(a | b), [(10, 10, 9)])
    assert why == {}
    assert np.array_equal(out == 1, a)
    assert not out[b].any()


def test_skimage_leaves_an_unseeded_basin_at_zero():
    """The masked watershed itself leaves a component with no marker at 0
    (the engine does not rely on it: its mask holds clicked pieces only)."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    a, b = _balls((20, 20, 40), ((10, 10, 9), (10, 10, 30)), 6.0, (1, 1, 1))
    fg = a | b
    markers = np.zeros(fg.shape, np.int32)
    markers[10, 10, 9] = 1
    out = watershed(-ndi.distance_transform_edt(fg), markers, mask=fg)
    assert np.array_equal(out == 1, a)
    assert not out[b].any()


# --------------------------------------------------------------------------- #
# what a click claims
# --------------------------------------------------------------------------- #
def _block(shape, z, y, x, img=None, value=1.0):
    img = np.zeros(shape, np.float64) if img is None else img
    img[z[0]:z[1], y[0]:y[1], x[0]:x[1]] = value
    return img


def test_a_click_whose_whole_ball_misses_the_stain_is_dim():
    img = _block((12, 20, 30), (3, 9), (5, 15), (5, 12))
    out, why = _seg(img, [(6, 10, 22)])
    assert why == {1: "dim"}
    assert not out.any()


def test_a_near_miss_whose_ball_touches_the_stain_claims_the_nucleus():
    """The click's own voxel is dim, but its ball reaches the nucleus."""
    img = _block((12, 20, 30), (3, 9), (5, 15), (5, 12))
    assert img[6, 10, 12] == 0.0                  # just past the edge
    out, why = _seg(img, [(6, 10, 12)])
    assert why == {}
    assert np.array_equal(out == 1, img > 0)


def test_a_click_on_a_small_nucleus_beside_a_big_one_claims_the_small_one():
    """The click's own voxel is on stain: its piece wins, even though the
    ball holds more of the big neighbour."""
    img = _block((12, 20, 30), (5, 7), (9, 11), (8, 10))     # small
    img = _block((12, 20, 30), (1, 11), (2, 18), (11, 25), img)  # big
    out, why = _seg(img, [(5, 9, 9)], radius_um=2.5)
    ball = _ball_at((12, 20, 30), (5, 9, 9), 2.5)
    assert int((ball & (img > 0) & (np.indices(img.shape)[2] >= 11)).sum()) \
        > int((ball & (img > 0) & (np.indices(img.shape)[2] < 11)).sum())
    assert why == {}
    small = np.zeros(img.shape, bool)
    small[5:7, 9:11, 8:10] = True
    assert np.array_equal(out == 1, small)


def _ball_at(shape, centre, r):
    from packages.epicycle_annotator.reshape.region import rasterise_points

    return rasterise_points([centre], shape, radius_um=r,
                            spacing=(1, 1, 1)) == 1


def test_a_ball_over_two_pieces_claims_the_one_holding_most_of_it():
    """The click sits in the dim gap between two pieces: the piece holding
    most of the ball is claimed, the other is left out."""
    shape = (12, 20, 30)
    img = np.zeros(shape, np.float64)
    img[6, 10, 5:10] = 1.0                 # a thin rod: one voxel in the ball
    img[2:10, 5:15, 11:16] = 1.0           # a block: five voxels in the ball
    out, why = _seg(img, [(6, 10, 10)], radius_um=1.5)
    assert why == {}
    block = np.zeros(shape, bool)
    block[2:10, 5:15, 11:16] = True
    assert np.array_equal(out == 1, block)


def test_a_tie_between_two_pieces_goes_to_the_first_piece():
    shape = (12, 20, 30)
    img = np.zeros(shape, np.float64)
    img[2:10, 5:15, 4:10] = 1.0
    img[2:10, 5:15, 11:17] = 1.0
    out, why = _seg(img, [(6, 10, 10)], radius_um=1.0)
    assert why == {}
    first = np.zeros(shape, bool)
    first[2:10, 5:15, 4:10] = True
    assert np.array_equal(out == 1, first)


def test_a_click_on_a_kept_object_makes_nothing_and_says_kept():
    img = _block((12, 20, 30), (3, 9), (5, 15), (5, 25))
    existing = np.zeros(img.shape, np.int32)
    existing[2:10, 4:16, 4:12] = 7
    out, why = _seg(img, [(6, 10, 8), (6, 10, 20)], existing=existing)
    assert why == {1: "kept"}
    assert not out[existing > 0].any()
    assert (out == 2).sum() == int(((img > 0) & (existing == 0)).sum())


def test_a_ball_shadowed_by_an_earlier_ball_is_kept():
    img = _block((12, 20, 30), (3, 9), (5, 15), (5, 25))
    out, why = _seg(img, [(6, 10, 10), (6, 10, 10)], radius_um=2.0)
    assert why == {2: "kept"}
    assert set(np.unique(out).tolist()) == {0, 1}


def test_pieces_under_smallest_object_say_small():
    img = _block((12, 20, 30), (5, 7), (9, 11), (4, 6))      # 8 voxels
    img = _block((12, 20, 30), (2, 10), (4, 16), (12, 24), img)
    out, why = _seg(img, [(5, 9, 4), (6, 10, 18)], min_size=20)
    assert why == {1: "small"}
    assert not (out == 1).any() and (out == 2).any()


def test_the_new_objects_never_touch_a_kept_voxel():
    fg, _ = _touching(ISO_SHAPE, ISO_C1, ISO_C2, 10.0, (1, 1, 1), ISO_NECK)
    existing = np.zeros(fg.shape, np.int32)
    existing[:, :, 30:34] = 5
    out, _why = _seg(_image(fg), [(15, 15, 10), (15, 15, 40)],
                     existing=existing)
    assert not out[existing > 0].any()
    assert out.dtype == np.int32


# --------------------------------------------------------------------------- #
# the threshold means what it means on Anywhere
# --------------------------------------------------------------------------- #
def test_a_flat_box_makes_every_click_dim():
    img = np.full((8, 10, 12), 3.0)
    out, why = _seg(img, [(4, 5, 6), (2, 2, 2)], threshold="otsu")
    assert why == {1: "dim", 2: "dim"}
    assert not out.any()


def test_a_constant_box_with_local_contrast_makes_every_click_dim():
    img = np.full((8, 10, 12), 3.0)
    out, why = _seg(img, [(4, 5, 6)], threshold=0.1, clahe=True)
    assert why == {1: "dim"}
    assert not out.any()


def test_the_auto_threshold_is_anywhere_s_threshold():
    """Otsu on the same values: the clicked piece is exactly what Anywhere
    finds there."""
    rng = np.random.default_rng(3)
    a, b = _balls((20, 24, 40), ((10, 12, 10), (10, 12, 30)), 6.0, (1, 1, 1))
    img = rng.normal(0.1, 0.02, a.shape) + 0.8 * (a | b)
    anywhere = AS.add_segment_region(img, np.zeros(a.shape, np.int32),
                                     threshold="otsu", separate=False,
                                     next_label=1)
    out, why = _seg(img, [(10, 12, 10)], threshold="otsu")
    assert why == {}
    piece = anywhere[10, 12, 10]
    assert piece > 0
    assert np.array_equal(out == 1, anywhere == piece)


# --------------------------------------------------------------------------- #
# Stop
# --------------------------------------------------------------------------- #
def test_stop_before_the_watershed_cancels():
    from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

    ev = threading.Event()
    ev.set()
    fg, _ = _touching(ISO_SHAPE, ISO_C1, ISO_C2, 10.0, (1, 1, 1), ISO_NECK)
    with pytest.raises(ReshapeCancelled):
        _seg(_image(fg), [ISO_C1, ISO_C2], cancel_event=ev)


def _flat_topped_block(shape, n_clicks, seed):
    """A bright block one voxel inside the box (its distance map has a flat
    top, the slices midway between its two z faces) and ``n_clicks`` random
    clicks inside it."""
    fg = np.zeros(shape, bool)
    fg[1:-1, 1:-1, 1:-1] = True
    rng = np.random.default_rng(seed)
    pts = set()
    while len(pts) < n_clicks:
        pts.add(tuple(int(rng.integers(1, s - 1)) for s in shape))
    return fg, sorted(pts)


class _StopAfter:
    """A Stop pressed once the hill grouping has polled it ``n`` times."""

    def __init__(self, n):
        self.n = n
        self.inside = False
        self.polls_inside = 0
        self.pressed_at = None

    def is_set(self):
        import time

        if self.inside:
            self.polls_inside += 1
            if self.polls_inside >= self.n and self.pressed_at is None:
                self.pressed_at = time.perf_counter()
        return self.pressed_at is not None


def test_stop_during_the_hill_grouping_cancels_promptly(monkeypatch):
    """Many clicks on one flat-topped piece: Stop is polled while the hills
    are grouped, and a Stop pressed there raises at once."""
    import time

    from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

    ev = _StopAfter(5)
    real = AS._gain_tops

    def _grouping(*a, **k):
        ev.inside = True
        try:
            return real(*a, **k)
        finally:
            ev.inside = False

    monkeypatch.setattr(AS, "_gain_tops", _grouping)
    fg, pts = _flat_topped_block((30, 96, 96), 100, 4)
    with pytest.raises(ReshapeCancelled):
        _seg(_image(fg), pts, cancel_event=ev)
    raised_at = time.perf_counter()
    assert ev.polls_inside > 0
    assert raised_at - ev.pressed_at < 1.0, raised_at - ev.pressed_at


def test_many_clicks_on_a_flat_topped_piece_group_their_hills_quickly(
        monkeypatch):
    """150 clicks on one flat-topped block: grouping the hills costs less
    than the rest of the call (the distance map and the flood), not many
    times it -- the flat top is labelled once, not once per click."""
    import time

    spent = []
    real = AS._gain_tops

    def _timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return real(*a, **k)
        finally:
            spent.append(time.perf_counter() - t0)

    monkeypatch.setattr(AS, "_gain_tops", _timed)
    fg, pts = _flat_topped_block((40, 128, 128), 150, 5)
    t0 = time.perf_counter()
    _seg(_image(fg), pts)
    total = time.perf_counter() - t0
    assert len(spent) == 1
    rest = total - spent[0]
    assert spent[0] <= 1.0 * rest, (round(spent[0], 2), round(rest, 2))


def test_stop_during_the_watershed_cancels_after_it(monkeypatch):
    import skimage.segmentation as sks

    from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

    ev = threading.Event()
    real = sks.watershed

    def _ws(*a, **k):
        out = real(*a, **k)
        ev.set()                      # Stop pressed while it ran
        return out

    monkeypatch.setattr(sks, "watershed", _ws)
    fg, _ = _touching(ISO_SHAPE, ISO_C1, ISO_C2, 10.0, (1, 1, 1), ISO_NECK)
    with pytest.raises(ReshapeCancelled):
        _seg(_image(fg), [ISO_C1, ISO_C2], cancel_event=ev)


# --------------------------------------------------------------------------- #
# the preview: ids, reasons, refusals
# --------------------------------------------------------------------------- #
OFF = (4, 10, 20)        # the box's corner in the volume


def _roi(shape):
    return tuple(slice(o, o + n) for o, n in zip(OFF, shape))


def _vol(points):
    return [tuple(int(p[a]) + OFF[a] for a in range(3)) for p in points]


def _params(**kw):
    p = {"look_for": AP.LOOK_MARKERS, "threshold": 0.5, "clahe": False,
         "clahe_clip": 0.01, "min_size": 0, "separate": True,
         "min_distance": 3, "click_radius_um": 1.0, "auto_smooth": False,
         "clear_border": True, "edge_margin": 3}
    p.update(kw)
    return p


def _preview(img, points, reserved, *, existing=None, next_label=50,
             id_max=None, spacing=(1.0, 1.0, 1.0), **kw):
    if existing is None:
        existing = np.zeros(img.shape, np.int32)
    return AP.run_add_preview(
        preset="nuclei", roi=_roi(img.shape), existing_roi=existing,
        next_label=next_label, spacing=spacing, params=_params(**kw),
        image_roi=img, click_points=_vol(points), reserved_ids=reserved,
        id_max=id_max)


def _three_nuclei():
    shape = (20, 30, 60)
    a, b, c = _balls(shape, ((10, 15, 10), (10, 15, 30), (10, 15, 50)), 6.0,
                     (1, 1, 1))
    return _image(a | b | c), (a, b, c)


def test_each_clicked_nucleus_carries_its_marker_s_reserved_id():
    img, (a, b, c) = _three_nuclei()
    res = _preview(img, [(10, 15, 10), (10, 15, 50)], [61, 64])
    assert res.status == "ok", res.refusal
    assert res.look_for == AP.LOOK_MARKERS
    assert sorted(res.ids) == [61, 64]
    assert np.array_equal(res.result == 61, a)
    assert np.array_equal(res.result == 64, c)
    assert not res.result[b].any()                 # unclicked: left out
    assert res.marker_why == {} and res.absorbed_ids == []
    assert res.n_markers_dim == 0


def test_the_handed_ids_are_written_as_they_are_and_never_on_a_kept_voxel():
    """The preview writes the ids it is handed, one per marker, and never
    over a kept object; that they sit above the volume's largest id is the
    reservation's work at placement (the orchestrator's tests), not the
    preview's -- it reads no id from the volume."""
    img, _ = _three_nuclei()
    existing = np.zeros(img.shape, np.int32)
    existing[0:2, 0:2, 0:2] = 49
    existing[8:12, 13:17, 26:34] = 49          # a kept object in nucleus 2
    res = _preview(img, [(10, 15, 10), (10, 15, 25)], [50, 51],
                   existing=existing)
    assert res.status == "ok"
    assert sorted(res.ids) == [50, 51]
    assert not res.result[existing > 0].any()


def test_a_reserved_id_near_the_int32_top_is_kept_exactly():
    img, (a, b, _c) = _three_nuclei()
    top = 2 ** 31 + 5
    res = _preview(img, [(10, 15, 10), (10, 15, 30)], [top, top + 1],
                   id_max=2 ** 32 - 1)
    assert res.status == "ok"
    assert sorted(res.ids) == [top, top + 1]
    assert res.result.dtype == np.int64
    assert np.array_equal(res.result == top, a)


def test_a_reserved_id_past_the_dtype_refuses_before_the_engine():
    img, _ = _three_nuclei()
    res = _preview(img, [(10, 15, 10)], [2 ** 31], id_max=2 ** 31 - 1)
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_IDS_FULL


def test_a_dim_and_a_kept_click_mint_nothing_and_say_why():
    img, (a, _b, _c) = _three_nuclei()
    existing = np.zeros(img.shape, np.int32)
    existing[8:13, 2:8, 26:34] = 9
    pts = [(10, 15, 10), (10, 4, 30), (10, 27, 40)]   # stain, kept, dim
    res = _preview(img, pts, [70, 71, 72], existing=existing)
    assert res.status == "ok"
    assert res.ids == [70]
    assert res.absorbed_ids == [71]
    assert res.marker_why == {72: "dim"}
    assert res.n_markers_dim == 1
    assert 71 not in np.unique(res.result) and 72 not in np.unique(res.result)
    note = AP.counts_note(res.counts())
    assert AP.ADD_NOTE_ABSORBED_ONE in note
    assert AP.ADD_NOTE_MARKERS_DIM_ONE in note


def test_nothing_on_stain_refuses_and_still_marks_the_rows():
    img, _ = _three_nuclei()
    res = _preview(img, [(10, 27, 40), (10, 2, 20)], [70, 71])
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_MARKERS_NONE_ON_STAIN
    assert res.marker_why == {70: "dim", 71: "dim"}
    assert res.n_markers_dim == 2
    assert res.result is None


def test_every_marker_kept_refuses_with_the_kept_sentence():
    img, _ = _three_nuclei()
    existing = np.zeros(img.shape, np.int32)
    existing[8:13, 2:8, 26:34] = 9
    existing[8:13, 2:8, 6:14] = 8
    one = _preview(img, [(10, 4, 30)], [70], existing=existing)
    assert one.status == "refused"
    assert one.refusal == AP.ADD_S1_MARKERS_ALL_KEPT_ONE
    assert one.absorbed_ids == [70]
    two = _preview(img, [(10, 4, 30), (10, 4, 10)], [70, 71],
                   existing=existing)
    assert two.refusal == AP.ADD_S1_MARKERS_ALL_KEPT
    assert two.absorbed_ids == [70, 71]


def test_a_dim_click_among_kept_ones_says_lower_the_threshold():
    img, _ = _three_nuclei()
    existing = np.zeros(img.shape, np.int32)
    existing[8:13, 2:8, 26:34] = 9
    res = _preview(img, [(10, 4, 30), (10, 27, 40)], [70, 71],
                   existing=existing)
    assert res.refusal == AP.ADD_S1_MARKERS_NONE_ON_STAIN
    assert res.absorbed_ids == [70] and res.marker_why == {71: "dim"}


def test_everything_below_smallest_object_refuses_and_marks_small():
    img, _ = _three_nuclei()
    res = _preview(img, [(10, 15, 10)], [70], min_size=10 ** 6)
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_ALL_FILTERED
    assert res.marker_why == {70: "small"}
    assert res.n_below_min_size == 1


def test_a_too_small_marker_outranks_a_dim_one_when_nothing_is_found():
    """As built: with nothing found, a marker dropped by Smallest object
    wins over a dim one -- the refusal is the size sentence, and both rows
    are marked."""
    shape = (12, 20, 30)
    img = _block(shape, (5, 7), (9, 11), (4, 6))          # 8 voxels
    res = _preview(img, [(5, 9, 4), (6, 10, 22)], [70, 71], min_size=20)
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_ALL_FILTERED
    assert res.marker_why == {70: "small", 71: "dim"}
    assert res.n_below_min_size == 1 and res.n_markers_dim == 1
    assert res.result is None


def test_a_nucleus_the_smoothing_empties_is_marked_small(monkeypatch):
    img, (a, _b, _c) = _three_nuclei()

    def _smooth(result, sigma=1.5, *, cancel_event=None):
        return np.where(result == 71, 0, result)

    monkeypatch.setattr(AS, "smooth_pieces", _smooth)
    res = _preview(img, [(10, 15, 10), (10, 15, 30)], [70, 71],
                   auto_smooth=True)
    assert res.status == "ok"
    assert res.ids == [70]
    assert res.marker_why == {71: "small"}
    assert res.n_emptied_by_smooth == 1


def test_a_clicked_nucleus_at_the_box_edge_starts_unticked():
    shape = (20, 30, 60)
    a, b = _balls(shape, ((10, 15, 3), (10, 15, 30)), 6.0, (1, 1, 1))
    res = _preview(_image(a | b), [(10, 15, 3), (10, 15, 30)], [70, 71])
    assert res.status == "ok"
    assert res.edge_ids == {70}


def test_without_reserved_ids_the_clicks_number_from_next_label():
    img, (a, _b, c) = _three_nuclei()
    res = _preview(img, [(10, 15, 10), (10, 27, 40), (10, 15, 50)], None,
                   next_label=200)
    assert res.status == "ok"
    assert res.ids == [200, 202]
    assert res.marker_why == {201: "dim"}


def test_no_marker_refuses_with_the_place_markers_sentence():
    img, _ = _three_nuclei()
    res = _preview(img, [], [])
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_NO_MARKERS


def test_stop_before_the_preview_finds_anything_cancels():
    img, _ = _three_nuclei()
    ev = threading.Event()
    ev.set()
    res = AP.run_add_preview(
        preset="nuclei", roi=_roi(img.shape),
        existing_roi=np.zeros(img.shape, np.int32), next_label=50,
        spacing=(1.0, 1.0, 1.0), params=_params(), image_roi=img,
        click_points=_vol([(10, 15, 10)]), reserved_ids=[50],
        cancel_event=ev)
    assert res.status == "cancelled"
    assert res.result is None


def test_the_preview_stages_are_anywhere_s():
    img, _ = _three_nuclei()
    seen = []
    AP.run_add_preview(
        preset="nuclei", roi=_roi(img.shape),
        existing_roi=np.zeros(img.shape, np.int32), next_label=50,
        spacing=(1.0, 1.0, 1.0), params=_params(auto_smooth=True),
        image_roi=img, click_points=_vol([(10, 15, 10)]), reserved_ids=[50],
        progress=lambda stage, i, n: seen.append(stage))
    assert seen == ["add_box", "add_find", "add_smooth"]


def test_anywhere_ignores_click_points():
    """Anywhere in the box with markers on record: the same result as with
    none, and the same as the region finder's own."""
    img, _ = _three_nuclei()
    existing = np.zeros(img.shape, np.int32)
    kw = dict(preset="nuclei", roi=_roi(img.shape), existing_roi=existing,
              next_label=50, spacing=(1.0, 1.0, 1.0),
              params=_params(look_for=AP.LOOK_ANYWHERE), image_roi=img)
    plain = AP.run_add_preview(**kw)
    clicked = AP.run_add_preview(click_points=_vol([(10, 15, 10)]),
                                 reserved_ids=[90], **kw)
    assert plain.status == clicked.status == "ok"
    assert np.array_equal(plain.result, clicked.result)
    assert plain.ids == clicked.ids and len(plain.ids) >= 3
    assert clicked.look_for == AP.LOOK_ANYWHERE
    assert clicked.marker_why == {} and clicked.n_markers_dim == 0


# --------------------------------------------------------------------------- #
# the counts note
# --------------------------------------------------------------------------- #
def test_the_dim_part_comes_right_after_the_outside_cells_part():
    keys = [k for k, _m, _o in AP._COUNT_PARTS]
    assert keys[:6] == ["n_absorbed", "n_cell_has_one", "n_no_nucleus",
                        "n_split_border", "n_outside_cells", "n_markers_dim"]
    counts = {k: 3 for k in keys}
    parts = AP.counts_note(counts).split(AP.ADD_NOTE_COUNTS_SEP)
    assert parts[5] == AP.ADD_NOTE_MARKERS_DIM.format(n=3)


@pytest.mark.parametrize("n,want", [
    (1, "1 marker sat on stain below the threshold and made nothing"),
    (2, "2 markers sat on stain below the threshold and made nothing"),
])
def test_the_dim_part_at_one_and_two(n, want):
    assert AP.counts_note({"n_markers_dim": n}) == want


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
def test_the_new_words_word_for_word():
    assert AP.LOOK_MARKERS == "markers"
    assert AP.ADD_LOOK_MARKERS == "At the markers I place"
    assert AP.ADD_HELP_LOOK_FOR == (
        "Where to look for nuclei: anywhere in the box by brightness, inside "
        "each cell of another run of this patch, or at the markers you "
        "place, one click per nucleus.")
    assert AP.ADD_HELP_NUCLEI_CLICKS == (
        "With Place markers on, click once on each nucleus. The threshold "
        "sets its edge; unclicked nuclei are left out. Ctrl + click removes "
        "one.")
    # markers are placed on a whole patch too: the old refusal is gone
    assert not hasattr(AP, "ADD_S1_WHOLE_MARKERS")
    assert AP.ADD_S1_PATCH_CLICKS == (
        "Whole patch {z} × {y} × {x} voxels · {n} objects marked")
    assert AP.ADD_S1_PATCH_CLICKS_ONE == (
        "Whole patch {z} × {y} × {x} voxels · 1 object marked")
    # both name the Threshold row of the Brightness card by its word
    assert AP.ADD_S1_MARKERS_NONE_ON_STAIN == (
        "No marker sits on stain above the threshold — lower Threshold, or "
        "move the markers onto the nuclei.")
    assert AP.ADD_MARKER_DIM_TIP == (
        "At the last Preview this marker sat on stain below the threshold, "
        "so it made nothing — move it onto the nucleus or lower Threshold "
        "under Brightness.")
    assert AP.ADD_MARKER_SMALL_TIP == (
        "At the last Preview this marker's nucleus was below Smallest object "
        "or vanished when smoothed.")
    # the markers' Advanced summary reads as Anywhere's now ("separate
    # touching" / "one per blob" with the markers' Spacing)
    assert not hasattr(AP, "ADD_SUM_ADVANCED_MARKERS")


def test_the_help_fits_where_the_cells_help_fits():
    assert len(AP.ADD_HELP_NUCLEI_CLICKS) <= len(AP.help_clicks(5.0))


@pytest.mark.parametrize("n,want", [
    (1, "Tick the object to find its nucleus — it is unticked."),
    (2, "Tick at least one object to find its nucleus — all 2 are "
        "unticked."),
])
def test_tick_one_nucleus(n, want):
    assert AP.nuclei_tick_one_line(n) == want


@pytest.mark.parametrize("n,want", [
    (1, "The marker now sits inside a kept object — remove it and click in "
        "empty space."),
    (2, "Every marker now sits inside a kept object — remove them and click "
        "in empty space."),
])
def test_all_kept(n, want):
    assert AP.markers_all_kept_line(n) == want


def test_each_reason_has_its_row_tip():
    assert AP.marker_why_tip("dim") == AP.ADD_MARKER_DIM_TIP
    assert AP.marker_why_tip("small") == AP.ADD_MARKER_SMALL_TIP
    assert AP.marker_why_tip("kept") == AP.ADD_MARKER_ABSORBED_TIP
    assert AP.marker_why_tip("anything else") is None


# --------------------------------------------------------------------------- #
# S1
# --------------------------------------------------------------------------- #
def _state(**kw):
    s = dict(has_patch=True, transient=None, busy=False, review=None,
             candidates=None, has_box=True, preset="nuclei",
             box_shape=(10, 20, 30), box_budget=AP.ADD_BOX_BUDGET,
             channel="nuc", look_for=AP.LOOK_MARKERS, n_clicks=3,
             n_ticked=3, info_pending=False, empty_share=0.5)
    s.update(kw)
    return s


def test_s1_asks_for_a_marker_first():
    assert AP.preview_block(_state(n_clicks=0, n_ticked=0)) == (
        AP.ADD_S1_NO_MARKERS)
    assert AP.preview_block(_state(n_clicks=0, n_ticked=0,
                                   place_markers_on=True)) == (
        AP.ADD_S1_NO_MARKERS_ON)


def test_s1_names_the_channel_before_the_markers():
    assert AP.preview_block(_state(channel="", n_clicks=0)) == (
        AP.ADD_S1_PICK_CHANNEL)


def test_s1_with_every_marker_unticked():
    assert AP.preview_block(_state(n_clicks=2, n_ticked=0)) == (
        AP.nuclei_tick_one_line(2))


def test_s1_ready_line_counts_the_markers():
    assert AP.preview_block(_state()) is None
    assert AP.s1_sentence(_state()) == AP.box_line(
        (10, 20, 30), preset="nuclei", look_for=AP.LOOK_MARKERS, n_clicks=3)
    assert AP.s1_sentence(_state()) == "Box 10 × 20 × 30 voxels · 3 objects marked"
    assert AP.s1_sentence(_state(n_clicks=1, n_ticked=1)) == (
        "Box 10 × 20 × 30 voxels · 1 object marked")
    assert AP.s1_sentence(_state(n_clicks=5, n_ticked=2)) == (
        "Box 10 × 20 × 30 voxels · 5 objects marked · 2 of 5 ticked")
    # the click count does not wait for the box's empty share
    assert AP.s1_sentence(_state(empty_share=None)) == (
        "Box 10 × 20 × 30 voxels · 3 objects marked")
    assert AP.s1_kind(_state()) == "ready"


def test_anywhere_s1_is_unchanged():
    s = _state(look_for=AP.LOOK_ANYWHERE)
    assert AP.preview_block(s) is None
    assert AP.s1_sentence(s) == AP.box_line((10, 20, 30), preset="nuclei",
                                            empty_share=0.5)
    assert AP.s1_sentence(dict(s, empty_share=None)) == AP.ADD_S1_MEASURING


def test_a_whole_patch_takes_markers_and_is_measured():
    """Markers are placed on a whole patch as in a box: Preview is not grey
    with a ticked marker, the ready line counts the markers and says "Whole
    patch", and the patch is measured (Auto, the histogram)."""
    s = _state(whole_patch=True)
    assert AP.preview_block(s) is None
    assert AP.s1_sentence(s) == (
        "Whole patch 10 × 20 × 30 voxels · 3 objects marked")
    assert AP.s1_sentence(_state(whole_patch=True, n_clicks=1,
                                 n_ticked=1)) == (
        "Whole patch 10 × 20 × 30 voxels · 1 object marked")
    assert AP.s1_sentence(_state(whole_patch=True, n_clicks=5,
                                 n_ticked=2)) == (
        "Whole patch 10 × 20 × 30 voxels · 5 objects marked · 2 of 5 "
        "ticked")
    assert AP.s1_kind(s) == "ready"
    # the same rows as a box before a marker is ticked
    assert AP.preview_block(_state(whole_patch=True, n_clicks=0,
                                   n_ticked=0)) == AP.ADD_S1_NO_MARKERS
    assert AP.preview_block(_state(whole_patch=True, n_clicks=0, n_ticked=0,
                                   place_markers_on=True)) == (
        AP.ADD_S1_NO_MARKERS_ON)
    assert AP.preview_block(_state(whole_patch=True, n_clicks=2,
                                   n_ticked=0)) == AP.nuclei_tick_one_line(2)
    # the patch is measured, and the strip names the read
    assert AP.measuring_stage(s, reader="info") == "add_info_patch"
    assert AP.whole_measures_nothing("nuclei", None,
                                     AP.LOOK_MARKERS) is False
    # Anywhere on a whole patch runs as the markers do: no sentence of its
    # own, and the patch is measured
    assert AP.preview_block(_state(whole_patch=True,
                                   look_for=AP.LOOK_ANYWHERE)) is None
    assert AP.whole_measures_nothing("nuclei", None,
                                     AP.LOOK_ANYWHERE) is False
