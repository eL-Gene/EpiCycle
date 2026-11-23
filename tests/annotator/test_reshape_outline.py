"""The tissue outline (``reshape/outline.py``).

The closing rule on synthetic topologies: a closing of radius r bridges a
gap up to about 2 r + 2 um (the blur widens the ball's 2 r) and leaves a
wider one open; a closed cavity is never filled (no hole fill); both sides
of a wound are kept.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from packages.epicycle_annotator.reshape import ReshapeCancelled
from packages.epicycle_annotator.reshape import outline as O

ISO = (1.0, 1.0, 1.0)

#: The local driver data (a read-only Wound patch copy); tests that need it
#: skip when it is absent.
DRIVERS = Path(os.environ.get(
    "EPICYCLE_DERIVE_DRIVERS", Path.home() / "Desktop" / "EpiCycle-drivers"))


def stain(tissue, seed=0):
    """A stain image: tissue bright, outside dark, a little noise."""
    rng = np.random.default_rng(seed)
    img = np.where(tissue, 1.0, 0.05).astype(np.float32)
    img += rng.normal(0, 0.03, tissue.shape).astype(np.float32)
    return np.clip(img, 0, None)


def slabs(gap, shape=(40, 50, 80), z=(8, 32), left=30):
    """Two slabs side by side in x, ``gap`` voxels apart."""
    t = np.zeros(shape, bool)
    t[z[0]:z[1], :, :left] = True
    t[z[0]:z[1], :, left + gap:] = True
    return t


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


# --------------------------------------------------------------------------- #
# the closing rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spacing", [ISO, (2.0, 1.0, 1.0)])
def test_a_wound_wider_than_two_r_stays_open_and_both_sides_are_kept(spacing):
    t = slabs(16)
    out = O.tissue_outline({"a": stain(t), "b": stain(t, 1)}, spacing)
    mid = out[20, :, 30 + 8]
    assert not mid.any()                         # the gap is not bridged
    pieces, n = ndi.label(out)
    assert n == 2                                # both sides survive
    assert out[20, 25, 10] and out[20, 25, 70]


def test_a_gap_narrower_than_two_r_is_bridged():
    t = slabs(6)
    out = O.tissue_outline({"a": stain(t), "b": stain(t, 1)}, ISO)
    assert out[20, 10:40, 33].all()


@pytest.mark.parametrize("lumen,open_", [(8, True), (3, False)])
def test_a_tube_keeps_a_wide_lumen_open_and_fills_a_narrow_one(lumen, open_):
    shape = (30, 64, 64)
    _z, y, x = np.indices(shape)
    r2 = (y - 32) ** 2 + (x - 32) ** 2
    t = (r2 <= 20 ** 2) & (r2 > lumen ** 2)
    out = O.tissue_outline({"a": stain(t), "b": stain(t, 2)}, ISO)
    assert bool(out[15, 32, 32]) is (not open_)
    assert out[15, 32, 32 + 14]                  # the wall itself


def test_a_closed_cavity_is_never_filled():
    shape = (56, 56, 56)
    z, y, x = np.indices(shape)
    r2 = (z - 28) ** 2 + (y - 28) ** 2 + (x - 28) ** 2
    t = (r2 <= 22 ** 2) & (r2 > 10 ** 2)
    out = O.tissue_outline({"a": stain(t), "b": stain(t, 3)}, ISO)
    assert not out[28, 28, 28]
    assert out[28, 28, 28 + 16]
    # a hole fill WOULD fill it -- the test has teeth
    assert ndi.binary_fill_holes(out)[28, 28, 28]


def test_one_channel_bridges_wider_gaps():
    assert O.default_close_radius_um(1) == 8.0
    assert O.default_close_radius_um(2) == 5.0
    assert O.default_close_radius_um(3) == 5.0
    t = slabs(16)
    img = stain(t)
    two = O.tissue_outline({"a": img, "b": img}, ISO)
    one = O.tissue_outline({"a": img}, ISO)
    assert not two[20, 20, 38]                   # r 5: the 16 um gap stays
    assert one[20, 20, 38]                       # r 8: bridged
    assert np.array_equal(one, O.tissue_outline(img, ISO))
    assert np.array_equal(two, O.tissue_outline({"a": img}, ISO, close_radius_um=5.0))


@pytest.mark.parametrize("gap_um,bridged", [(12, True), (13, False)])
def test_r_5_bridges_about_2r_plus_2_um(gap_um, bridged):
    """The blur before the cut widens the ball's 2 r = 10 um to 12 um."""
    sp = (0.5, 0.5, 0.5)
    g = int(round(gap_um / sp[2]))
    t = slabs(g, shape=(40, 60, 120 + g), left=60)
    out = O.tissue_outline({"a": stain(t), "b": stain(t, 1)}, sp)
    assert bool(out[20, 30, 60 + g // 2]) is bridged


def test_the_gap_to_bridge_is_twice_the_radius():
    t = slabs(14)
    chans = {"a": stain(t), "b": stain(t, 1)}
    assert np.array_equal(O.tissue_outline(chans, ISO, bridge_um=12.0),
                          O.tissue_outline(chans, ISO, close_radius_um=6.0))
    with pytest.raises(ValueError, match="not both"):
        O.tissue_outline(chans, ISO, close_radius_um=5.0, bridge_um=10.0)
    with pytest.raises(ValueError, match="bridge_um"):
        O.tissue_outline(chans, ISO, bridge_um=-1.0)
    with pytest.raises(ValueError, match="close_radius_um"):
        O.tissue_outline(chans, ISO, close_radius_um=float("nan"))
    m = t.copy()
    assert np.array_equal(O.cleared_outline(m, ISO, bridge_um=8.0),
                          O.cleared_outline(m, ISO, close_radius_um=4.0))
    assert np.array_equal(O.cleared_outline(m, ISO),
                          O.cleared_outline(m, ISO, close_radius_um=O.CLEARED_CLOSE_UM))
    with pytest.raises(ValueError, match="not both"):
        O.cleared_outline(m, ISO, 3.0, bridge_um=6.0)


def test_small_pieces_are_dropped():
    t = slabs(30, shape=(40, 50, 90))
    t[17:23, 22:28, 42:48] = True               # 216 um3, 12 um from both slabs
    img = stain(t)
    out = O.tissue_outline({"a": img, "b": img}, ISO)
    assert not out[20, 25, 45]
    assert out[20, 25, 10] and out[20, 25, 80]
    kept = O.tissue_outline({"a": img, "b": img}, ISO, min_piece_um3=0.0)
    assert kept[20, 25, 45]


def test_stride_2_agrees_with_full_resolution():
    rng = np.random.default_rng(4)
    shape = (40, 64, 64)
    field = ndi.gaussian_filter(rng.random(shape), 4.0)
    t = field > np.percentile(field, 45)
    chans = {"a": stain(t), "b": stain(t, 5)}
    s2 = O.tissue_outline(chans, (0.5, 0.5, 0.5))
    s1 = O.tissue_outline(chans, (0.5, 0.5, 0.5), stride=1)
    assert s2.shape == shape and s2.dtype == bool
    assert dice(s1, s2) >= 0.98


def test_an_odd_shape_is_upsampled_to_the_patch_shape():
    t = slabs(16, shape=(39, 51, 81))
    out = O.tissue_outline({"a": stain(t)}, ISO)
    assert out.shape == (39, 51, 81)
    assert out[20, 50, 80] and out[20, 50, 0]


def test_a_given_threshold_is_used_and_must_name_every_channel():
    from packages.epicycle_annotator.reshape import region as R

    t = slabs(16)
    chans = {"a": stain(t), "b": stain(t, 1)}
    thr = R.patch_dark_threshold(chans, spacing=ISO, stride=1)
    assert np.array_equal(O.tissue_outline(chans, ISO, dark_threshold=thr),
                          O.tissue_outline(chans, ISO))
    with pytest.raises(ValueError, match="normalisation"):
        O.tissue_outline({"a": chans["a"], "c": chans["b"]}, ISO, dark_threshold=thr)


def test_bad_inputs_are_refused_in_plain_words():
    t = stain(slabs(16))
    with pytest.raises(ValueError, match="at least one"):
        O.tissue_outline({}, ISO)
    with pytest.raises(ValueError, match="3D"):
        O.tissue_outline({"a": t[0]}, ISO)
    with pytest.raises(ValueError, match="shape"):
        O.tissue_outline({"a": t, "b": t[:, :, :10]}, ISO)
    with pytest.raises(ValueError, match="spacing"):
        O.tissue_outline({"a": t}, (1.0, -1.0, 1.0))
    with pytest.raises(ValueError, match="stride"):
        O.tissue_outline({"a": t}, ISO, stride=0)


def test_a_cancel_raises():
    ev = threading.Event()
    ev.set()
    with pytest.raises(ReshapeCancelled):
        O.tissue_outline({"a": stain(slabs(16))}, ISO, cancel=ev)


def test_the_outline_never_writes():
    t = slabs(10)
    a, b = stain(t), stain(t, 1)
    mk = np.zeros(t.shape, np.int32)
    mk[20, 20, 5:10] = 3
    copies = [x.copy() for x in (a, b, mk, t)]
    for x in (a, b, mk, t):
        x.setflags(write=False)
    out = O.tissue_outline({"a": a, "b": b}, ISO)
    O.cleared_outline(t, ISO)
    O.boundary_shell(out, ISO)
    O.markers_outside(mk, out)
    for before, after in zip(copies, (a, b, mk, t)):
        assert np.array_equal(before, after)
    assert not np.shares_memory(out, a) and out.flags.writeable


# --------------------------------------------------------------------------- #
# the cleared-objects outline
# --------------------------------------------------------------------------- #
def test_the_cleared_outline_bridges_small_gaps_and_keeps_cavities():
    shape = (30, 40, 60)
    m = np.zeros(shape, bool)
    m[8:22, 8:32, 4:20] = True
    m[8:22, 8:32, 24:40] = True                   # 4 um apart: bridged at r 3
    m[8:22, 8:32, 48:58] = True                   # 8 um apart: stays apart
    out = O.cleared_outline(m, ISO)
    assert out[15, 20, 22]
    assert not out[15, 20, 44]
    assert (out >= m).all()
    hollow = np.zeros(shape, bool)
    hollow[2:28, 5:35, 5:55] = True
    hollow[8:22, 12:28, 15:45] = False            # a 14-um cavity
    assert not O.cleared_outline(hollow, ISO)[15, 20, 30]


def test_the_cleared_outline_equals_a_whole_volume_closing():
    rng = np.random.default_rng(7)
    shape = (30, 44, 50)
    m = ndi.gaussian_filter(rng.random(shape), 2.0) > 0.53
    m[:, :, :5] = False
    sp = (2.0, 1.0, 0.8)
    assert np.array_equal(O.cleared_outline(m, sp), O._close_ball(m, 3.0, sp))
    # touching a patch face: the face neither erodes nor glues it
    face = np.zeros(shape, bool)
    face[0:10, 10:20, 10:20] = True
    assert np.array_equal(O.cleared_outline(face, ISO)[0:10, 10:20, 10:20],
                          face[0:10, 10:20, 10:20])


def test_an_empty_cleared_mask_gives_no_outline():
    out = O.cleared_outline(np.zeros((4, 5, 6), bool), ISO)
    assert out.shape == (4, 5, 6) and not out.any()


# --------------------------------------------------------------------------- #
# the display shell and the marker drop
# --------------------------------------------------------------------------- #
def test_the_boundary_shell_is_one_hollow_shell():
    shape = (30, 30, 30)
    z, y, x = np.indices(shape)
    ball = (z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 100
    shell = O.boundary_shell(ball, ISO)
    _lab, n = ndi.label(shell, structure=np.ones((3, 3, 3)))
    assert n == 1
    assert not shell[15, 15, 15]
    assert (ball | ~shell).all()                     # inside the outline
    edge = ball & ~ndi.binary_erosion(ball)          # a 6-neighbour outside
    assert (shell >= edge).all()
    assert shell.sum() < 0.5 * ball.sum()


def test_the_patch_faces_are_not_a_boundary():
    o = np.zeros((10, 12, 12), bool)
    o[:, 3:9, 3:9] = True                            # runs through both z faces
    shell = O.boundary_shell(o, ISO)
    assert not shell[0, 5, 5] and not shell[9, 5, 5]
    assert shell[0, 3, 5]


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_markers_mostly_outside_are_named_at_0_7(dtype):
    shape = (1, 1, 300)
    o = np.zeros(shape, bool)
    mk = np.zeros(shape, dtype)
    big = 2_147_483_647 if dtype == np.int32 else 4_000_000_000
    mk[0, 0, 0:100] = big                            # 70 outside: dropped
    o[0, 0, 70:100] = True
    mk[0, 0, 100:200] = 5                            # 69 outside: kept
    o[0, 0, 169:200] = True
    mk[0, 0, 200:300] = 9                            # all inside
    o[0, 0, 200:300] = True
    out = O.markers_outside(mk, o)
    assert out.tolist() == [big]
    assert out.dtype == mk.dtype
    assert O.OUTLINE_DROP_SHARE == 0.7
    assert O.markers_outside(mk, o, share=0.69).tolist() == [5, big]
    assert O.markers_outside(np.zeros(shape, dtype), o).size == 0


# --------------------------------------------------------------------------- #
# the real patch
# --------------------------------------------------------------------------- #
def test_the_outline_on_a_real_patch_matches_the_measured_recipe():
    """On Wound patch 1 the outline must agree with the recipe as it was
    measured (full-resolution not-dark, block majority at stride 2, closed
    r 5, pieces >= 1000 um3) -- saved beside the driver copy."""
    root = DRIVERS
    img_f = root / "derive_d0" / "ws" / "patches" / "Wound_patch_1.tiff"
    ref_f = root / "derive_in_add" / "c0" / "out" / "prep_Wound_patch_1.npz"
    if not (img_f.is_file() and ref_f.is_file()):
        pytest.skip("the Wound patch driver copy is not on this machine")
    import tifffile

    img = tifffile.imread(str(img_f))
    ref = np.load(str(ref_f))["env"]
    out = O.tissue_outline({"ch0": img[0], "ch1": img[1]}, (0.5, 0.5, 0.5))
    assert dice(out, ref) >= 0.99
