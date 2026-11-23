"""Add ▸ Advanced ▸ Smoothing counts voxels of the FULL-size image.

The Smoothing box is a Gaussian sigma in voxels.  In a half-size session
(the patch opened 2x downsampled) one voxel of the session's array spans two
of the full-size image, so the gather divides the box's number by the
session's half-size factor -- read the same way the markers' Spacing reads
it -- and a new object is smoothed as hard, relative to its size, as at full
size.  At full size (factor 1) the value sent is exactly the box's.

Tested through the real orchestrator gather (borrowed by the Qt-free
harness) on every Add route that ends in the smoothing (Nuclei Anywhere, At
the markers I place, Inside each cell, Cells), through the Step 3
controller's own context, and on the real smoothing with the gathered
values.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.add_segment import smooth_pieces

from .add_harness import Harness, install_fakes

BOX = 1.5      # the Smoothing box's default


# --------------------------------------------------------------------------- #
# harnesses, one per Add route
# --------------------------------------------------------------------------- #
def _ball_image(shape=(20, 30, 30)):
    zz, yy, xx = np.indices(shape)
    img = np.zeros(shape, np.float64)
    img[((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) <= 16] = 1.0
    return img


def _nuclei(monkeypatch, look):
    install_fakes(monkeypatch)
    img = _ball_image()
    hh = Harness(np.zeros(img.shape, np.int32), images={"nuc": img},
                 spacing=(1.0, 0.5, 0.5))
    hh.adv.look = look
    hh.place_box(tuple(slice(0, n) for n in img.shape))
    hh._add_threshold_value = 0.5
    hh._add_click_markers = [{"zyx": (10, 10, 10), "id": 1, "ticked": True}]
    return hh


def _anywhere(monkeypatch):
    return _nuclei(monkeypatch, (AP.LOOK_ANYWHERE, None))


def _markers(monkeypatch):
    return _nuclei(monkeypatch, (AP.LOOK_MARKERS, None))


def _inside(monkeypatch):
    from .test_add_derive_wiring import _harness

    hh, _img, _cells = _harness(monkeypatch)
    return hh


def _cells(monkeypatch):
    install_fakes(monkeypatch)
    img = _ball_image()
    hh = Harness(np.zeros(img.shape, np.int32),
                 images={"nuc": img, "mem": img.copy()},
                 spacing=(1.0, 0.5, 0.5))
    hh.adv.preset = "cells"
    hh.adv.source = (AP.SOURCE_CLICKS, None)
    hh.adv.params["dark_outside"] = False
    hh.place_box(tuple(slice(0, n) for n in img.shape))
    hh._add_click_markers = [{"zyx": (10, 10, 10), "id": 1, "ticked": True}]
    hh._add_info = AP.AddBoxInfo(empty_share=1.0)
    return hh


ROUTES = {"anywhere": _anywhere, "markers": _markers, "inside": _inside,
          "cells": _cells}


def _set_factor(hh, factor):
    """The session's half-size factor; Inside each cell's patch levels are
    cached per factor, so the harness's cached levels move to the new key."""
    cache = getattr(hh, "_add_levels_cache", None)
    levels = hh._add_cached_levels() if cache else None
    hh.ctx["downsample_factor"] = factor
    if levels is not None:
        cache.clear()
        cache[hh._add_levels_key()] = levels


def _gather(hh, *, box=BOX, auto=True):
    hh.adv.params["smooth_sigma"] = box
    hh.adv.params["auto_smooth"] = auto
    kw, why = hh._add_gather_preview()
    assert why is None, why
    return kw


# --------------------------------------------------------------------------- #
# the gather
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("box", [BOX, 2.0, 0.1])
def test_full_size_sends_the_box_s_number_as_is(monkeypatch, route, box):
    """Factor 1 (and a context without one): exactly today's value."""
    for factor in (None, 1):
        hh = ROUTES[route](monkeypatch)
        if factor is not None:
            _set_factor(hh, factor)
        got = _gather(hh, box=box)["params"]["smooth_sigma"]
        assert got == box and type(got) is type(box), (route, factor)


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("box", [BOX, 2.0, 0.1])
def test_half_size_sends_half_the_box_s_number(monkeypatch, route, box):
    hh = ROUTES[route](monkeypatch)
    _set_factor(hh, 2)
    got = _gather(hh, box=box)["params"]["smooth_sigma"]
    assert got == pytest.approx(box / 2.0), route


@pytest.mark.parametrize("factor,want", [(3, BOX / 3.0), (4, BOX / 4.0)])
def test_other_factors_divide_too(monkeypatch, factor, want):
    hh = _markers(monkeypatch)
    hh.ctx["downsample_factor"] = factor
    assert _gather(hh)["params"]["smooth_sigma"] == pytest.approx(want)


@pytest.mark.parametrize("factor", [0, -2, None, "", "two", float("nan"),
                                    float("inf")])
def test_a_missing_or_invalid_factor_sends_today_s_value(monkeypatch,
                                                         factor):
    hh = _anywhere(monkeypatch)
    hh.ctx["downsample_factor"] = factor
    got = _gather(hh)["params"]["smooth_sigma"]
    assert got == BOX and type(got) is float


def test_the_conversion_happens_once_per_gather(monkeypatch):
    """Gathering again (Preview again, Add another) does not halve twice."""
    hh = _markers(monkeypatch)
    hh.ctx["downsample_factor"] = 2
    first = _gather(hh)["params"]["smooth_sigma"]
    second = _gather(hh)["params"]["smooth_sigma"]
    assert first == second == pytest.approx(BOX / 2.0)
    assert hh.adv.params["smooth_sigma"] == BOX     # the box is untouched


@pytest.mark.parametrize("factor,want", [(1, BOX), (2, BOX / 2.0)])
def test_through_the_controller_s_context(monkeypatch, factor, want):
    """The Step 3 controller's own context provider feeds the factor."""
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator as C,
    )

    from .test_add_nuclei_one_click_page import _controller_context

    hh = _markers(monkeypatch)
    hh._reshape_context_cb = _controller_context((2.0, 0.5, 0.5), factor)
    hh._reshape_context = types.MethodType(C._reshape_context, hh)
    hh._refine_spacing = types.MethodType(C._refine_spacing, hh)
    assert hh._reshape_context()["downsample_factor"] == factor
    assert _gather(hh)["params"]["smooth_sigma"] == pytest.approx(want)


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
SMOOTH_TIP = ("How much smoothing, in voxels of the full-size image. 1.5 is "
              "gentle.")


def test_the_smoothing_tooltip_says_full_size_voxels():
    """The same words the markers' Spacing uses: "in voxels of the
    full-size image"."""
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    assert ADD_HELP_LINES["add_smooth_sigma_sb"] == SMOOTH_TIP
    assert "in voxels of the full-size image" in ADD_HELP_LINES[
        "add_marker_spacing_sb"]


def test_the_guide_says_it_where_half_size_is_discussed():
    import pathlib

    guide = (pathlib.Path(__file__).resolve().parents[2] / "docs" / "guides"
             / "annotator_3d_add.md").read_text(encoding="utf-8")
    flat = " ".join(guide.split())
    assert ("Smoothing counts voxels of the full-size image, so a new object "
            "is smoothed about as much at half size as at full size: a "
            "little less, so objects come out slightly larger. At half size "
            "a Smoothing of about 1.2 or below does nothing.") in flat
    assert ("Smallest object, Edge zone and Anywhere's Spacing still count "
            "voxels of the half-size image") in flat
    assert ("Smoothing is in voxels of the full-size image: 1.5 is gentle. "
            "In a half-size session it smooths each object about as much as "
            "at full size, a little less, so objects come out slightly "
            "larger; there a Smoothing of about 1.2 or below does "
            "nothing.") in flat
    # never the overstatement
    assert "as much at half size as at full size." not in flat
    assert "still smooths each object as much" not in flat


# --------------------------------------------------------------------------- #
# the smoothing, with the gathered values
# --------------------------------------------------------------------------- #
def _nucleus(radius_vox):
    """A ball of one id, with room around it."""
    n = int(np.ceil(2 * radius_vox)) + 9
    c = (n - 1) / 2.0
    z, y, x = np.indices((n, n, n))
    lab = np.zeros((n, n, n), np.int32)
    lab[((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) <= radius_vox ** 2] = 7
    return lab


def _halve(lab):
    """The half-size version: a voxel is kept where 4 or more of its 2x2x2
    full-size voxels are."""
    n = lab.shape[0] // 2 * 2
    blocks = (lab[:n, :n, :n] == 7).reshape(n // 2, 2, n // 2, 2, n // 2, 2)
    return np.where(blocks.sum(axis=(1, 3, 5)) >= 4, 7, 0).astype(np.int32)


def _smoothed(lab, sigma):
    return smooth_pieces(lab.copy(), sigma)


def _share(lab, out):
    return float((out == 7).sum() / (lab == 7).sum())


def check_shares(sigma_full, sigma_half, sigma_old):
    """A 4 um-radius nucleus at 0.5 um voxels, and its half-size version:
    smoothed at full size with ``sigma_full``, at half size with
    ``sigma_half`` (what the gather sends) and with ``sigma_old`` (the
    box's number undivided).  The half-size object must really be smoothed
    -- changed, and not nearly all kept -- and kept closer to the full-size
    share than the undivided number keeps it."""
    full = _nucleus(4.0 / 0.5)
    half = _halve(full)
    kept_full = _share(full, _smoothed(full, sigma_full))
    out_half = _smoothed(half, sigma_half)
    kept_half = _share(half, out_half)
    kept_old = _share(half, _smoothed(half, sigma_old))
    print(f"[ADD-SMOOTH-HALF] kept: full size {kept_full:.3f} "
          f"(sigma {sigma_full}), half size {kept_half:.3f} "
          f"(sigma {sigma_half}), half size undivided {kept_old:.3f} "
          f"(sigma {sigma_old})")
    assert not np.array_equal(out_half, half), "nothing was smoothed"
    assert kept_half < 0.99
    assert kept_half >= 0.75
    assert abs(kept_full - kept_half) <= 0.15
    assert abs(kept_full - kept_half) < abs(kept_full - kept_old)


def test_a_nucleus_keeps_a_similar_share_at_half_size(monkeypatch):
    """Smoothed with what the gather sends at factor 1 and at factor 2."""
    sigmas = {}
    for factor in (1, 2):
        hh = _anywhere(monkeypatch)
        hh.ctx["downsample_factor"] = factor
        sigmas[factor] = _gather(hh)["params"]["smooth_sigma"]
    check_shares(sigmas[1], sigmas[2], BOX)
