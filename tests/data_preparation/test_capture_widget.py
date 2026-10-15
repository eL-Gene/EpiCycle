"""Pure, Qt-free tests for ``compose_capture_rgb``.

Deliberately does NOT construct the QDialog — Qt widget tests crash on
teardown in this repo's CI.  Only the pure reduce-to-RGB function is exercised
here, with plain numpy arrays of known values.

Run with:
    python -m pytest tests/data_preparation/test_capture_widget.py -q
"""
import numpy as np
import pytest

from napari_epicycle.data_preparation._capture_widget import (
    _nice_tick_step,
    compose_capture_rgb,
    cut_plane_quads,
    panels_to_reslice,
    plane_reslice,
    point_click_to_cursor,
    scrub_delta_to_cursor,
    volume_channel_limits,
    wheel_step,
)


def _encoded_volume(C, Z, Y, X):
    """A (C,Z,Y,X) volume whose every voxel encodes its own index, so a
    reslice's orientation can be verified voxel-by-voxel:

        value = c*1_000_000 + z*10_000 + y*100 + x
    """
    vol = np.zeros((C, Z, Y, X), dtype=np.float64)
    for c in range(C):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    vol[c, z, y, x] = c * 1_000_000 + z * 10_000 + y * 100 + x
    return vol


def _grad_plane(y, x, scale=1.0, offset=0.0):
    """A varying (non-constant) plane so percentile hi > lo."""
    base = np.arange(y * x, dtype=np.float32).reshape(y, x)
    return base * scale + offset


def test_4d_mip_returns_yx3_uint8():
    # (C=2, Z=3, Y=5, X=7)
    C, Z, Y, X = 2, 3, 5, 7
    data = np.zeros((C, Z, Y, X), dtype=np.uint16)
    for c in range(C):
        for z in range(Z):
            data[c, z] = _grad_plane(Y, X, scale=(c + 1)) + z
    out = compose_capture_rgb(data, z_index=0, mip=True,
                              channels_on=[True, True])
    assert out.shape == (Y, X, 3)
    assert out.dtype == np.uint8


def test_4d_single_z_slice_returns_yx3():
    C, Z, Y, X = 2, 4, 6, 8
    data = np.zeros((C, Z, Y, X), dtype=np.uint16)
    for c in range(C):
        for z in range(Z):
            data[c, z] = _grad_plane(Y, X, scale=(c + 1)) + z * 3
    out = compose_capture_rgb(data, z_index=2, mip=False,
                              channels_on=[True, True])
    assert out.shape == (Y, X, 3)
    assert out.dtype == np.uint8


def test_z_index_out_of_range_is_clamped():
    C, Z, Y, X = 1, 3, 4, 4
    data = np.zeros((C, Z, Y, X), dtype=np.uint16)
    for z in range(Z):
        data[0, z] = _grad_plane(Y, X) + z
    # Far out-of-range indices must not raise and must clamp into [0, Z-1].
    hi = compose_capture_rgb(data, z_index=999, mip=False, channels_on=[True])
    lo = compose_capture_rgb(data, z_index=-999, mip=False, channels_on=[True])
    assert hi.shape == (Y, X, 3)
    assert lo.shape == (Y, X, 3)
    # z=999 clamps to last slice, z=-999 clamps to first slice -> differ.
    top = compose_capture_rgb(data, z_index=Z - 1, mip=False,
                              channels_on=[True])
    bot = compose_capture_rgb(data, z_index=0, mip=False, channels_on=[True])
    assert np.array_equal(hi, top)
    assert np.array_equal(lo, bot)


def test_3d_input_promoted_to_grayscale():
    # (Z=3, Y=5, X=5) single-channel -> grayscale RGB (R == G == B).
    Z, Y, X = 3, 5, 5
    data = np.zeros((Z, Y, X), dtype=np.uint16)
    for z in range(Z):
        data[z] = _grad_plane(Y, X) + z
    out = compose_capture_rgb(data, z_index=0, mip=True, channels_on=[True])
    assert out.shape == (Y, X, 3)
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])
    # Grayscale of a varying plane must be visible (not all black).
    assert out.max() > 0


def test_all_channels_off_is_black():
    C, Z, Y, X = 3, 2, 4, 4
    data = (np.arange(C * Z * Y * X, dtype=np.uint16)
            .reshape(C, Z, Y, X))
    out = compose_capture_rgb(data, z_index=0, mip=True,
                              channels_on=[False, False, False])
    assert out.shape == (Y, X, 3)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_single_active_channel_among_many_is_visible():
    # 3 channels, only channel 0 has (varying) signal and is on.
    C, Z, Y, X = 3, 2, 6, 6
    data = np.zeros((C, Z, Y, X), dtype=np.uint16)
    data[0, 0] = _grad_plane(Y, X, scale=4.0) + 10  # varying signal
    data[0, 1] = _grad_plane(Y, X, scale=4.0) + 20
    out = compose_capture_rgb(data, z_index=0, mip=True,
                              channels_on=[True, False, False])
    assert out.shape == (Y, X, 3)
    # Must render visible content, not an empty/black image.
    assert out.max() > 0
    # Signal maps to the red plane; green/blue stay dark.
    assert out[..., 0].max() > 0
    assert out[..., 1].max() == 0
    assert out[..., 2].max() == 0


# --------------------------------------------------------------------------- #
# volume_channel_limits — per-channel bounds computed over the WHOLE volume.
# --------------------------------------------------------------------------- #
def test_volume_channel_limits_one_per_channel_gradient():
    # (C=3, Z=4, Y=5, X=6) — each channel a distinct varying gradient.
    C, Z, Y, X = 3, 4, 5, 6
    data = np.zeros((C, Z, Y, X), dtype=np.float32)
    for c in range(C):
        for z in range(Z):
            data[c, z] = _grad_plane(Y, X, scale=(c + 1)) + z * 7.0

    limits = volume_channel_limits(data)
    assert isinstance(limits, list)
    assert len(limits) == C  # exactly one (lo, hi) per channel
    for (lo, hi) in limits:
        # A varying (gradient) volume must give a real, non-degenerate range.
        assert hi > lo


def test_volume_channel_limits_constant_volume_guard():
    # A perfectly constant channel -> percentile lo == hi -> guard to (lo, lo+1).
    C, Z, Y, X = 2, 3, 4, 4
    data = np.full((C, Z, Y, X), 42.0, dtype=np.float32)
    limits = volume_channel_limits(data)
    assert len(limits) == C
    for (lo, hi) in limits:
        # Guard must keep hi strictly above lo so downstream normalize is safe.
        assert hi > lo
        assert lo == 42.0
        assert hi == 43.0


def test_volume_channel_limits_promotes_3d_single_channel():
    # (Z, Y, X) -> promoted to C=1 -> exactly one (lo, hi).
    Z, Y, X = 3, 5, 5
    data = np.zeros((Z, Y, X), dtype=np.float32)
    for z in range(Z):
        data[z] = _grad_plane(Y, X) + z
    limits = volume_channel_limits(data)
    assert len(limits) == 1
    lo, hi = limits[0]
    assert hi > lo


# --------------------------------------------------------------------------- #
# Global (fixed-limits) normalization — the "saturated Z-ends" regression fix.
# --------------------------------------------------------------------------- #
def test_global_limits_darken_low_signal_slice():
    """A low-signal slice should render MUCH darker under volume-global limits
    than under per-slice percentile normalization.

    Volume: one bright middle slice (values up to 1000) and two low-signal
    slices (values <= 10, but still a gradient so per-slice percentiles are
    non-degenerate).  Per-slice norm stretches the low slice's tiny gradient to
    full brightness; global norm scales it against the whole-volume range so it
    stays near-black.
    """
    C, Z, Y, X = 1, 3, 8, 8
    data = np.zeros((C, Z, Y, X), dtype=np.float32)
    # Low-signal Z-ends: small gradient in [0, 10].
    data[0, 0] = np.linspace(0.0, 10.0, Y * X).reshape(Y, X)
    data[0, 2] = np.linspace(0.0, 10.0, Y * X).reshape(Y, X)
    # Bright middle slice: gradient up to 1000.
    data[0, 1] = np.linspace(0.0, 1000.0, Y * X).reshape(Y, X)

    limits = volume_channel_limits(data)

    # Render the LOW-signal slice (z=0) both ways.
    per_slice = compose_capture_rgb(data, z_index=0, mip=False,
                                    channels_on=[True])
    global_norm = compose_capture_rgb(data, z_index=0, mip=False,
                                      channels_on=[True], limits=limits)

    # Per-slice stretch makes the low slice bright; global keeps it dark.
    assert per_slice.mean() > 0
    assert global_norm.mean() < per_slice.mean() * 0.5
    # And absolutely dark, not merely relatively.
    assert global_norm.mean() < 20.0  # << 255


def test_limits_none_matches_legacy_per_slice_path():
    # Back-compat: limits omitted -> identical to explicit per-slice call.
    C, Z, Y, X = 2, 3, 5, 7
    data = np.zeros((C, Z, Y, X), dtype=np.uint16)
    for c in range(C):
        for z in range(Z):
            data[c, z] = _grad_plane(Y, X, scale=(c + 1)) + z
    a = compose_capture_rgb(data, z_index=1, mip=False,
                            channels_on=[True, True])
    b = compose_capture_rgb(data, z_index=1, mip=False,
                            channels_on=[True, True], limits=None)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Reslice orientation — XZ / YZ planes fed through compose_capture_rgb.
# --------------------------------------------------------------------------- #
def test_reslice_xz_plane_orientation():
    """An XZ reslice (plane at fixed y) fed as (C, 1, Z, X) must compose to a
    (Z, X, 3) image — Z rows, X cols — the orientation the dialog draws."""
    C, Z, Y, X = 1, 4, 5, 6
    data = np.zeros((C, Z, Y, X), dtype=np.float32)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                data[0, z, y, x] = z * 100.0 + y * 10.0 + x

    y0 = 2
    xz_plane = data[:, :, y0, :]        # (C, Z, X)
    xz_vol = xz_plane[:, None, :, :]    # (C, 1, Z, X)
    out = compose_capture_rgb(xz_vol, z_index=0, mip=False, channels_on=[True])
    assert out.shape == (Z, X, 3)       # rows = Z, cols = X


def test_reslice_yz_plane_orientation():
    """A YZ reslice (plane at fixed x) is SAGITTAL -> (C, Y, Z); fed through
    compose it yields a (Y, Z, 3) image — Y rows, Z cols."""
    C, Z, Y, X = 1, 4, 5, 6
    data = np.zeros((C, Z, Y, X), dtype=np.float32)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                data[0, z, y, x] = z * 100.0 + y * 10.0 + x

    x0 = 3
    yz = plane_reslice(data, "YZ", x0)  # (C, Y, Z) — sagittal transpose
    assert yz.shape == (C, Y, Z)
    yz_vol = yz[:, None, :, :]          # (C, 1, Y, Z)
    out = compose_capture_rgb(yz_vol, z_index=0, mip=False, channels_on=[True])
    assert out.shape == (Y, Z, 3)       # rows = Y, cols = Z


# --------------------------------------------------------------------------- #
# plane_reslice — the ground-truth reslice contract (per panel).
# --------------------------------------------------------------------------- #
def test_plane_reslice_shapes_per_panel():
    """Each panel returns (C, H, W) with H = the ROW axis (never transposed)."""
    C, Z, Y, X = 2, 4, 5, 6
    vol = _encoded_volume(C, Z, Y, X)
    assert plane_reslice(vol, "XY", 0).shape == (C, Y, X)   # rows=Y cols=X
    assert plane_reslice(vol, "XZ", 0).shape == (C, Z, X)   # rows=Z cols=X
    assert plane_reslice(vol, "YZ", 0).shape == (C, Y, Z)   # rows=Y cols=Z (sagittal)


def test_plane_reslice_xy_values_match_source():
    """XY at z=idx -> vol[:, idx]; [c, y, x] == vol[c, idx, y, x]."""
    C, Z, Y, X = 2, 4, 5, 6
    vol = _encoded_volume(C, Z, Y, X)
    idx = 2
    out = plane_reslice(vol, "XY", idx)          # (C, Y, X)
    for c in range(C):
        for y in range(Y):
            for x in range(X):
                assert out[c, y, x] == vol[c, idx, y, x]


def test_plane_reslice_xz_values_match_source():
    """XZ at y=idx -> vol[:, :, idx, :]; [c, z, x] == vol[c, z, idx, x].

    Verifies the row axis is Z (not transposed to X)."""
    C, Z, Y, X = 2, 4, 5, 6
    vol = _encoded_volume(C, Z, Y, X)
    idx = 3
    out = plane_reslice(vol, "XZ", idx)          # (C, Z, X)
    for c in range(C):
        for z in range(Z):
            for x in range(X):
                assert out[c, z, x] == vol[c, z, idx, x]


def test_plane_reslice_yz_values_match_source():
    """YZ at x=idx is sagittal -> (C, Y, Z); [c, y, z] == vol[c, z, y, idx].

    Verifies the sagittal transpose: rows=Y, cols=Z (not the old rows=Z)."""
    C, Z, Y, X = 2, 4, 5, 6
    vol = _encoded_volume(C, Z, Y, X)
    idx = 4
    out = plane_reslice(vol, "YZ", idx)          # (C, Y, Z)
    assert out.shape == (C, Y, Z)
    for c in range(C):
        for y in range(Y):
            for z in range(Z):
                assert out[c, y, z] == vol[c, z, y, idx]


def test_plane_reslice_promotes_3d_to_single_channel():
    """A bare (Z, Y, X) input is promoted to (1, Z, Y, X) before reslicing."""
    Z, Y, X = 3, 5, 6
    vol3d = _encoded_volume(1, Z, Y, X)[0]       # (Z, Y, X)
    out = plane_reslice(vol3d, "XZ", 1)          # (1, Z, X)
    assert out.shape == (1, Z, X)
    for z in range(Z):
        for x in range(X):
            assert out[0, z, x] == vol3d[z, 1, x]


def test_plane_reslice_index_clamped_at_edges():
    """Out-of-range indices clamp into the panel's slice axis."""
    C, Z, Y, X = 1, 4, 5, 6
    vol = _encoded_volume(C, Z, Y, X)
    # XY slice axis is Z (0..Z-1).
    assert np.array_equal(plane_reslice(vol, "XY", 999), vol[:, Z - 1])
    assert np.array_equal(plane_reslice(vol, "XY", -999), vol[:, 0])
    # XZ slice axis is Y (0..Y-1).
    assert np.array_equal(plane_reslice(vol, "XZ", 999), vol[:, :, Y - 1, :])
    # YZ slice axis is X (0..X-1); sagittal -> transpose last two axes to (C,Y,Z).
    assert np.array_equal(
        plane_reslice(vol, "YZ", 999), vol[:, :, :, X - 1].transpose(0, 2, 1))


def test_plane_reslice_unknown_panel_raises():
    vol = _encoded_volume(1, 2, 3, 4)
    with pytest.raises(ValueError):
        plane_reslice(vol, "ZZ", 0)


# --------------------------------------------------------------------------- #
# panels_to_reslice — CW2 per-panel redraw decision (XY<-z, XZ<-y, YZ<-x).
# --------------------------------------------------------------------------- #
def test_panels_to_reslice_none_prev_returns_all():
    # initial load / content change / forced redraw -> every panel.
    assert panels_to_reslice(None, (1, 2, 3)) == {"XY", "XZ", "YZ"}


def test_panels_to_reslice_z_change_only_xy():
    # a wheel over XY moves z only -> only XY's image needs a re-fetch.
    assert panels_to_reslice((5, 5, 5), (6, 5, 5)) == {"XY"}


def test_panels_to_reslice_y_change_only_xz():
    assert panels_to_reslice((5, 5, 5), (5, 6, 5)) == {"XZ"}


def test_panels_to_reslice_x_change_only_yz():
    assert panels_to_reslice((5, 5, 5), (5, 5, 6)) == {"YZ"}


def test_panels_to_reslice_no_change_returns_empty():
    # the throttle can fire with an unchanged cursor -> nothing to reslice.
    assert panels_to_reslice((5, 5, 5), (5, 5, 5)) == set()


def test_panels_to_reslice_two_axis_scrub_returns_two():
    # a scrub drag moves both in-plane axes of a panel at once.
    assert panels_to_reslice((5, 5, 5), (5, 6, 7)) == {"XZ", "YZ"}


def test_panels_to_reslice_all_axes_change_returns_all():
    assert panels_to_reslice((1, 1, 1), (2, 2, 2)) == {"XY", "XZ", "YZ"}


def test_panels_to_reslice_matches_panel_slice_axis_mapping():
    # Guard the mapping against _panel_slice_index drift: bumping ONE axis must
    # flag exactly the panel that indexes on it.
    base = (4, 4, 4)
    assert panels_to_reslice(base, (5, 4, 4)) == {"XY"}   # z -> XY
    assert panels_to_reslice(base, (4, 5, 4)) == {"XZ"}   # y -> XZ
    assert panels_to_reslice(base, (4, 4, 5)) == {"YZ"}   # x -> YZ


# --------------------------------------------------------------------------- #
# point_click_to_cursor — the click -> cursor mapping (per panel).
# --------------------------------------------------------------------------- #
# shape is (Z, Y, X); cursor is [z, y, x].
def test_point_click_xy_maps_col_to_x_row_to_y():
    shape = (10, 20, 30)   # Z, Y, X
    cur = [5, 5, 5]
    # click at (row=7, col=13) -> y=7, x=13, z unchanged.
    out = point_click_to_cursor("XY", row=7, col=13, shape=shape, cursor=cur)
    assert out == [5, 7, 13]


def test_point_click_xz_maps_col_to_x_row_to_z():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    # rows=Z, cols=X -> row=8 -> z=8, col=14 -> x=14, y unchanged.
    out = point_click_to_cursor("XZ", row=8, col=14, shape=shape, cursor=cur)
    assert out == [8, 5, 14]


def test_point_click_yz_maps_col_to_z_row_to_y():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    # Sagittal YZ: rows=Y, cols=Z -> row=9 -> y=9, col=8 -> z=8, x unchanged.
    out = point_click_to_cursor("YZ", row=9, col=8, shape=shape, cursor=cur)
    assert out == [8, 9, 5]


def test_point_click_row_is_the_vertical_data_axis():
    """The ROW coord must land on the vertical data axis of each panel:
    Y for XY and YZ (sagittal), Z for XZ (never the horizontal axis)."""
    shape = (10, 20, 30)
    base = [1, 2, 3]
    # XY: row -> y (index 1).
    assert point_click_to_cursor("XY", 7, 0, shape, base)[1] == 7
    # XZ: row -> z (index 0).
    assert point_click_to_cursor("XZ", 7, 0, shape, base)[0] == 7
    # YZ (sagittal): row -> y (index 1).
    assert point_click_to_cursor("YZ", 7, 0, shape, base)[1] == 7


def test_point_click_clamps_beyond_bounds_per_panel():
    shape = (10, 20, 30)   # Z, Y, X
    cur = [5, 5, 5]
    # XY: huge col/row clamp to X-1 / Y-1; negatives clamp to 0.
    out = point_click_to_cursor("XY", row=999, col=999, shape=shape, cursor=cur)
    assert out == [5, 19, 29]
    out = point_click_to_cursor("XY", row=-999, col=-999, shape=shape, cursor=cur)
    assert out == [5, 0, 0]
    # XZ: row clamps to Z-1, col to X-1.
    out = point_click_to_cursor("XZ", row=999, col=999, shape=shape, cursor=cur)
    assert out == [9, 5, 29]
    # YZ (sagittal): row clamps to Y-1, col to Z-1.
    out = point_click_to_cursor("YZ", row=999, col=999, shape=shape, cursor=cur)
    assert out == [9, 19, 5]


def test_point_click_does_not_mutate_input_cursor():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    out = point_click_to_cursor("XY", 7, 13, shape, cur)
    assert cur == [5, 5, 5]   # original untouched
    assert out is not cur


def test_point_click_leaves_slice_axis_unchanged_per_panel():
    """Clicking within a panel never moves that panel's own slice axis."""
    shape = (10, 20, 30)
    cur = [4, 6, 8]
    assert point_click_to_cursor("XY", 1, 1, shape, cur)[0] == 4  # z fixed
    assert point_click_to_cursor("XZ", 1, 1, shape, cur)[1] == 6  # y fixed
    assert point_click_to_cursor("YZ", 1, 1, shape, cur)[2] == 8  # x fixed


def test_point_click_round_trips_with_plane_reslice():
    """A click that decodes to (row, col) via plane_reslice's orientation must
    map back to the very voxel value shown at that pixel — proving the display
    and click seams share ONE orientation decision (no half-applied flip)."""
    C, Z, Y, X = 1, 6, 7, 8
    vol = _encoded_volume(C, Z, Y, X)
    cursor = [3, 3, 3]
    # In the XZ panel (rows=Z, cols=X) at slice y=cursor.y, click pixel
    # (row=4, col=5).  That pixel shows vol[0, z=4, y=3, x=5].
    xz = plane_reslice(vol, "XZ", cursor[1])       # (C, Z, X)
    shown = xz[0, 4, 5]
    new = point_click_to_cursor("XZ", row=4, col=5, shape=(Z, Y, X), cursor=cursor)
    z, y, x = new
    assert vol[0, z, y, x] == shown                # z=4, y=3 (unchanged), x=5
    assert (z, y, x) == (4, 3, 5)


def test_point_click_round_trips_with_plane_reslice_yz():
    """Same round-trip for the SAGITTAL YZ panel (rows=Y, cols=Z) — proving the
    transpose is applied consistently in the display AND the click seams (a
    half-applied transpose would be a silent wrong-voxel)."""
    C, Z, Y, X = 1, 6, 7, 8
    vol = _encoded_volume(C, Z, Y, X)
    cursor = [3, 3, 3]
    # In the YZ panel (rows=Y, cols=Z) at slice x=cursor.x, click pixel
    # (row=4, col=5).  That pixel shows vol[0, z=5, y=4, x=3]  (row->Y, col->Z).
    yz = plane_reslice(vol, "YZ", cursor[2])       # (C, Y, Z)
    shown = yz[0, 4, 5]                            # [row=4=Y, col=5=Z]
    new = point_click_to_cursor("YZ", row=4, col=5, shape=(Z, Y, X), cursor=cursor)
    z, y, x = new
    assert vol[0, z, y, x] == shown               # z=5 (col), y=4 (row), x=3
    assert (z, y, x) == (5, 4, 3)


def test_point_click_unknown_panel_raises():
    with pytest.raises(ValueError):
        point_click_to_cursor("ZZ", 1, 1, (2, 3, 4), [0, 0, 0])


# --------------------------------------------------------------------------- #
# scrub_delta_to_cursor — Scrub-mode drag mapping, per panel.
#
# CONVENTION (must match the runtime — a pixel-y-down test would bake a
# compensating sign and hide a Z double-flip): d_data = (dx, dy) is ALREADY in
# FLIPPED DATA-SPACE — dx = change along the panel's COLUMN data-axis, dy =
# change along its ROW data-axis, with screen-down = +row = +dy and
# screen-right = +col = +dx (the camera flip is applied UPSTREAM by the
# panel transform).  The seam applies routing only, sign = +identity.
# shape is (Z, Y, X); cursor is [z, y, x].
# --------------------------------------------------------------------------- #
def test_scrub_xy_dx_steps_x_dy_steps_y():
    shape = (10, 20, 30)   # Z, Y, X
    cur = [5, 5, 5]
    out = scrub_delta_to_cursor("XY", (3, 4), cur, shape)
    assert out == [5, 9, 8]        # x += 3 -> 8, y += 4 -> 9, z unchanged


def test_scrub_yz_dx_steps_z_dy_steps_y():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    # Sagittal YZ: dx -> Z (col), dy -> Y (row).
    out = scrub_delta_to_cursor("YZ", (3, 4), cur, shape)
    assert out == [8, 9, 5]        # z += 3 -> 8, y += 4 -> 9, x unchanged


def test_scrub_xz_dx_steps_x_dy_steps_z():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    out = scrub_delta_to_cursor("XZ", (3, 4), cur, shape)
    assert out == [9, 5, 8]        # x += 3 -> 8, z += 4 -> 9, y unchanged


def test_scrub_sign_per_panel():
    """Net-on-screen sign check (+identity, no re-flip):
    +dy (screen-down) increases Z (XZ) and Y (XY, YZ-sagittal);
    +dx (screen-right) increases X (XY, XZ) and Z (YZ-sagittal)."""
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    # +dy (screen-down):
    assert scrub_delta_to_cursor("XZ", (0, 2), cur, shape)[0] == 7   # z+2
    assert scrub_delta_to_cursor("XY", (0, 2), cur, shape)[1] == 7   # y+2
    assert scrub_delta_to_cursor("YZ", (0, 2), cur, shape)[1] == 7   # y+2 (row=Y)
    # +dx (screen-right):
    assert scrub_delta_to_cursor("XY", (2, 0), cur, shape)[2] == 7   # x+2
    assert scrub_delta_to_cursor("XZ", (2, 0), cur, shape)[2] == 7   # x+2
    assert scrub_delta_to_cursor("YZ", (2, 0), cur, shape)[0] == 7   # z+2 (col=Z)


def test_scrub_negative_delta_decreases():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    out = scrub_delta_to_cursor("XY", (-2, -3), cur, shape)
    assert out == [5, 2, 3]        # x -2 -> 3, y -3 -> 2


def test_scrub_diagonal_steps_both_inplane_axes():
    """A diagonal drag moves BOTH in-plane axes at once (dx and dy)."""
    shape = (10, 20, 30)   # Z max 9, Y max 19, X max 29
    cur = [4, 4, 4]
    # XZ: dx -> X, dy -> Z, both move, Y stays (no clamping here).
    out = scrub_delta_to_cursor("XZ", (5, 3), cur, shape)
    assert out == [7, 4, 9]        # z 4+3=7, x 4+5=9, y unchanged


def test_scrub_clamps_each_axis_at_edges():
    shape = (10, 20, 30)   # Z max 9, Y max 19, X max 29
    cur = [5, 5, 5]
    # Large positive -> clamp to max on both axes.
    out = scrub_delta_to_cursor("YZ", (999, 999), cur, shape)
    assert out == [9, 19, 5]        # y->19, z->9, x fixed
    # Large negative -> clamp to 0.
    out = scrub_delta_to_cursor("YZ", (-999, -999), cur, shape)
    assert out == [0, 0, 5]


def test_scrub_rounds_fractional_delta():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    out = scrub_delta_to_cursor("XY", (2.6, -1.4), cur, shape)
    assert out == [5, 4, 8]        # dx 2.6->3 (x=8), dy -1.4->-1 (y=4)


def test_scrub_does_not_mutate_input_cursor():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    out = scrub_delta_to_cursor("XY", (3, 4), cur, shape)
    assert cur == [5, 5, 5]
    assert out is not cur


def test_scrub_unknown_panel_raises():
    with pytest.raises(ValueError):
        scrub_delta_to_cursor("ZZ", (1, 1), [0, 0, 0], (2, 3, 4))


# --------------------------------------------------------------------------- #
# wheel_step — wheel steps a panel's OWN slice axis, per panel.
# shape is (Z, Y, X); XY->Z, XZ->Y, YZ->X.
# --------------------------------------------------------------------------- #
def test_wheel_xy_steps_z():
    shape = (10, 20, 30)
    assert wheel_step("XY", +1, [5, 5, 5], shape) == [6, 5, 5]
    assert wheel_step("XY", -1, [5, 5, 5], shape) == [4, 5, 5]


def test_wheel_xz_steps_y():
    shape = (10, 20, 30)
    assert wheel_step("XZ", +1, [5, 5, 5], shape) == [5, 6, 5]
    assert wheel_step("XZ", -1, [5, 5, 5], shape) == [5, 4, 5]


def test_wheel_yz_steps_x():
    shape = (10, 20, 30)
    assert wheel_step("YZ", +1, [5, 5, 5], shape) == [5, 5, 6]
    assert wheel_step("YZ", -1, [5, 5, 5], shape) == [5, 5, 4]


def test_wheel_direction_sign_is_one_voxel_per_notch():
    """Any positive direction steps +1, any negative steps -1 (one voxel per
    wheel notch, regardless of the raw delta magnitude)."""
    shape = (10, 20, 30)
    assert wheel_step("XY", 5.0, [5, 5, 5], shape) == [6, 5, 5]
    assert wheel_step("XY", -0.3, [5, 5, 5], shape) == [4, 5, 5]
    assert wheel_step("XY", 0, [5, 5, 5], shape) == [5, 5, 5]   # no move


def test_wheel_clamps_at_slice_axis_edges():
    shape = (10, 20, 30)   # Z max 9
    # At top of Z, wheeling up clamps.
    assert wheel_step("XY", +1, [9, 5, 5], shape) == [9, 5, 5]
    # At bottom, wheeling down clamps.
    assert wheel_step("XY", -1, [0, 5, 5], shape) == [0, 5, 5]
    # YZ slice axis is X (max 29).
    assert wheel_step("YZ", +1, [5, 5, 29], shape) == [5, 5, 29]


def test_wheel_does_not_mutate_input_cursor():
    shape = (10, 20, 30)
    cur = [5, 5, 5]
    out = wheel_step("XY", +1, cur, shape)
    assert cur == [5, 5, 5]
    assert out is not cur


def test_wheel_unknown_panel_raises():
    with pytest.raises(ValueError):
        wheel_step("ZZ", +1, [0, 0, 0], (2, 3, 4))


# --------------------------------------------------------------------------- #
# _nice_tick_step — finer, fixed-nice-step axis ticks (~6-8 majors).
# --------------------------------------------------------------------------- #
def test_nice_tick_step_gives_finer_ticks():
    # A 0-128 axis -> step 20 -> ticks 0,20,..,120 (7 majors), NOT 0/50/100.
    assert _nice_tick_step(128) == 20.0
    # A 0-143 axis (Z) -> also 20 -> 0,20,..,140 (8 majors).
    assert _nice_tick_step(143) == 20.0
    # Count the majors for a 0-128 range at step 20.
    step = _nice_tick_step(128)
    n = int(128 // step) + 1
    assert 6 <= n <= 8


def test_nice_tick_step_scales_and_guards():
    assert _nice_tick_step(1000) == 200.0        # 0,200,..,1000 (6 majors)
    assert _nice_tick_step(10) == 2.0            # 0,2,..,10
    # Degenerate / bad spans never divide-by-zero or return <= 0.
    assert _nice_tick_step(0) > 0
    assert _nice_tick_step(-5) > 0
    assert _nice_tick_step(float("nan")) > 0


# --------------------------------------------------------------------------- #
# cut_plane_quads — the 3 3-D cut-plane quads in the Volume's (x,y,z) frame.
# cursor is [z, y, x]; shape is (Z, Y, X).  Vertices are (x, y, z).
# --------------------------------------------------------------------------- #
def test_cut_plane_quads_shapes_and_panels():
    quads = cut_plane_quads([5, 6, 7], (10, 20, 30))
    assert set(quads) == {"XY", "XZ", "YZ"}
    for panel in ("XY", "XZ", "YZ"):
        assert quads[panel].shape == (4, 3)


def test_cut_plane_xy_is_perp_z_at_z_plus_half_and_spans_xy():
    # XY-cut ⟂ Z at z+0.5, spanning x in [0,X], y in [0,Y].
    Z, Y, X = 10, 20, 30
    q = cut_plane_quads([5, 6, 7], (Z, Y, X))["XY"]
    # All four corners share the same z = z + 0.5.
    assert np.allclose(q[:, 2], 5 + 0.5)
    # x spans 0..X, y spans 0..Y.
    assert q[:, 0].min() == 0.0 and q[:, 0].max() == X
    assert q[:, 1].min() == 0.0 and q[:, 1].max() == Y


def test_cut_plane_xz_is_perp_y_at_y_plus_half_and_spans_xz():
    # XZ-cut ⟂ Y at y+0.5, spanning x in [0,X], z in [0,Z].
    Z, Y, X = 10, 20, 30
    q = cut_plane_quads([5, 6, 7], (Z, Y, X))["XZ"]
    assert np.allclose(q[:, 1], 6 + 0.5)          # constant y = y+0.5
    assert q[:, 0].min() == 0.0 and q[:, 0].max() == X   # x span
    assert q[:, 2].min() == 0.0 and q[:, 2].max() == Z   # z span


def test_cut_plane_yz_is_perp_x_at_x_plus_half_and_spans_yz():
    # YZ-cut ⟂ X at x+0.5, spanning y in [0,Y], z in [0,Z].
    Z, Y, X = 10, 20, 30
    q = cut_plane_quads([5, 6, 7], (Z, Y, X))["YZ"]
    assert np.allclose(q[:, 0], 7 + 0.5)          # constant x = x+0.5
    assert q[:, 1].min() == 0.0 and q[:, 1].max() == Y   # y span
    assert q[:, 2].min() == 0.0 and q[:, 2].max() == Z   # z span


def test_cut_plane_quads_track_cursor():
    # Moving the cursor moves the perpendicular offset (the +0.5 centring).
    a = cut_plane_quads([2, 2, 2], (10, 10, 10))
    b = cut_plane_quads([8, 2, 2], (10, 10, 10))
    assert np.allclose(a["XY"][:, 2], 2.5)
    assert np.allclose(b["XY"][:, 2], 8.5)        # XY (⟂ Z) followed z


def test_cut_plane_quads_are_float32():
    q = cut_plane_quads([1, 1, 1], (4, 5, 6))
    for panel in q:
        assert q[panel].dtype == np.float32
