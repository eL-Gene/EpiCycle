"""A3: the picture
sizes and the labels overlay.  Qt-free except the PNG encode.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.ui_shell import _stack_thumbs as T  # noqa: E402


def test_size_constants():
    assert T.THUMB_WIDE_PX == 128
    assert T.THUMB_NARROW_PX == 80
    assert T.THUMB_OVERLAY_MODE == "fill+outline"
    assert T.THUMB_OVERLAY_ALPHA == 0.45


def test_png_name_carries_size_and_overlay(tmp_path):
    p128 = T.thumb_png_path(tmp_path, "a.tif", "x/a.tif", 1, 2, size=128)
    p80 = T.thumb_png_path(tmp_path, "a.tif", "x/a.tif", 1, 2, size=80)
    assert ".s128." in p128.name and ".s80." in p80.name
    assert p128 != p80
    lab = T.thumb_png_path(tmp_path, "a.tif", "x/a.tif", 1, 2,
                           labels_mtime_ns=7, size=128)
    assert re.search(r"\.l7\.ofo45\.png$", lab.name)
    a = T.thumb_png_path(tmp_path, "a.tif", "x/a.tif", 1, 2,
                         labels_mtime_ns=7, overlay_alpha=0.3)
    b = T.thumb_png_path(tmp_path, "a.tif", "x/a.tif", 1, 2,
                         labels_mtime_ns=7, overlay_mode="outline")
    assert a != lab and b != lab and a != b


def test_stale_pngs_sweep_a_png_of_another_size_and_legacy_names(tmp_path):
    keep = T.thumb_png_path(tmp_path, "a.tif", "x/a.tif", 1, 2, size=128)
    keep.write_bytes(b"x")
    old64 = T.thumb_png_path(tmp_path, "a.tif", "x/a.tif", 1, 2, size=64)
    old64.write_bytes(b"x")
    legacy = tmp_path / ("a.tif." + T.path_hash("x/a.tif") + ".1.2.c2.png")
    legacy.write_bytes(b"x")
    legacy_l = tmp_path / ("a.tif." + T.path_hash("x/a.tif") + ".1.2.c2.l9.png")
    legacy_l.write_bytes(b"x")
    stale = {p.name for p in T.stale_pngs(tmp_path, "a.tif", "x/a.tif", keep)}
    assert stale == {old64.name, legacy.name, legacy_l.name}


def _plane():
    p = np.zeros((20, 30), np.uint16)
    p[2:8, 2:8] = 1
    p[2:8, 12:18] = 2
    p[12:18, 5:25] = 50          # 50 -> colour index ((50-1) % 49) + 1 == 1
    return p


def test_overlay_per_instance_colour_fill_and_outline():
    from napari.utils.colormaps import label_colormap

    cols = np.asarray(label_colormap(49, 0.5).colors)[:, :3]
    want = lambda i: np.rint(cols[((i - 1) % 49) + 1] * 255).astype(int)  # noqa
    rgb = np.full((20, 30, 3), 100, np.uint8)
    out = T.overlay_labels(rgb, _plane(), size=64)
    assert out.shape == rgb.shape and out is not rgb
    # background untouched
    assert (out[0, 0] == 100).all() and (out[10, 10] == 100).all()
    # an interior pixel of label 1: the fill alpha over the grey picture
    inner = out[4, 4].astype(int)
    exp = np.rint(100 * 0.55 + want(1) * 0.45).astype(int)
    assert np.abs(inner - exp).max() <= 1
    # the 1 px inner boundary is the full colour
    assert (out[2, 2].astype(int) == want(1)).all()
    assert (out[2, 12].astype(int) == want(2)).all()
    # each instance has its OWN colour; id 50 wraps to the colour of id 1
    assert not (want(1) == want(2)).all()
    assert (out[12, 5].astype(int) == want(50)).all()
    assert (want(50) == want(1)).all()


def test_overlay_modes():
    rgb = np.full((20, 30, 3), 100, np.uint8)
    o = T.overlay_labels(rgb, _plane(), mode="outline", size=64)
    assert (o[4, 4] == 100).all() and not (o[2, 2] == 100).all()
    f = T.overlay_labels(rgb, _plane(), mode="fill", size=64)
    assert not (f[4, 4] == 100).all()
    # fill only: the boundary pixel is blended, not full strength
    assert not (f[2, 2] == o[2, 2]).all()
    with pytest.raises(ValueError):
        T.overlay_labels(rgb, _plane(), mode="nope")


def test_overlay_nearest_resample_of_a_downsampled_labels_plane():
    rgb = np.full((10, 15, 3), 100, np.uint8)
    out = T.overlay_labels(rgb, _plane(), size=64)    # labels 20x30, rgb 10x15
    assert out.shape == rgb.shape
    assert not (out == 100).all()


def test_step3_sweep_regex_matches_the_new_names_and_the_shared_tail(tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_thumbs as S3

    assert S3._PNG_TAIL == T.PNG_TAIL
    for kw in ({}, {"labels_mtime_ns": 5}, {"size": 80}):
        p = T.thumb_png_path(tmp_path, "My patch.tif", "x/p.tif", 1, 2, **kw)
        m = S3._PNG_NAME.match(p.name)
        assert m and m.group("name") == "My patch.tif"
