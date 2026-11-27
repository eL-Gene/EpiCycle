"""Inside each cell in a half-size session.

At factor 2 every micron knob reads Step 3's spacing times the factor (the
session's own voxel), the size floors stay in cubic microns, the patch's
levels are measured on the session array, and the amber half-size caution
stands before a Preview -- never over candidates.  A Cells whole patch hands
the engine the session's own spacing too (where its background starts is
measured in microns), and refuses an unknown spacing before it reads
anything.
"""

from __future__ import annotations

import numpy as np

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.add_derive import MIN_NUCLEUS_UM3

from .test_add_derive_nuclei import three_cells_scene
from .test_add_derive_wiring import _harness, _run

#: QC spacing 0.5 um, halved -> the session's voxel is 1.0 um
HALF = (1.0, 1.0, 1.0)


def _half(monkeypatch, **kw):
    img, cells = three_cells_scene()
    # the half-size session's arrays (what the layers hold)
    img2, cells2 = img[::2, ::2, ::2].copy(), cells[::2, ::2, ::2].copy()
    h, img2, cells2 = _harness(monkeypatch, scene=(img2, cells2),
                               spacing=HALF, **kw)
    h.ctx["downsample_factor"] = 2
    # the cache key holds the factor: measured again under it
    h._add_levels_cache = {}
    from packages.epicycle_annotator.add_derive import patch_levels

    h._add_levels_cache[h._add_levels_key()] = patch_levels(img2, cells2,
                                                            HALF)
    return h, img2, cells2


def test_the_micron_knobs_read_the_session_spacing(monkeypatch):
    h, _img, _cells = _half(monkeypatch)
    kw, why = h._add_gather_preview()
    assert why is None
    assert tuple(kw["spacing"]) == HALF == tuple(h._refine_spacing())
    assert kw["params"]["nuc_radius_um"] == 2.5          # microns, unchanged


def test_the_size_floor_is_about_twelve_session_voxels():
    assert MIN_NUCLEUS_UM3 / float(np.prod(HALF)) == 12.5
    assert MIN_NUCLEUS_UM3 / float(np.prod((0.5, 0.5, 0.5))) == 100.0


def test_the_levels_are_measured_on_the_session_array(monkeypatch):
    h, img, cells = _half(monkeypatch)
    h._add_levels_cache = {}
    kw = h._add_gather_info()
    nuclear, guide = kw["levels_sources"]
    assert nuclear.shape == img.shape == cells.shape
    assert guide is cells
    assert tuple(kw["spacing"]) == HALF


def test_the_caution_stands_before_a_preview_only(monkeypatch):
    h, _img, _cells = _half(monkeypatch)
    h._add_refresh_ui()
    assert h.adv.last("derive_note") == AP.ADD_NOTE_HALFSIZE_BOXED
    res = _run(h)
    assert res.ok, res.refusal
    h._add_refresh_ui()
    assert h.adv.last("derive_note") == "", "never over candidates"


def test_full_size_says_nothing(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    h._add_refresh_ui()
    assert h.adv.last("derive_note") == ""


def test_a_mostly_full_box_says_so_after_the_half_size_note(monkeypatch):
    h, _img, _cells = _half(monkeypatch)
    h._add_info.empty_share = 0.1
    h._add_refresh_ui()
    assert h.adv.last("derive_note") == AP.ADD_NOTE_HALFSIZE_BOXED
    h.ctx["downsample_factor"] = 1
    h._add_refresh_ui()
    assert h.adv.last("derive_note") == AP.ADD_AMBER_BOX_MOSTLY_FULL.format(
        p=10)


def test_a_cells_whole_patch_fills_at_the_session_spacing(monkeypatch):
    from packages.epicycle_annotator.reshape import region as R

    from .test_add_whole_fill_wiring import _harness as _cells_harness

    h = _cells_harness(monkeypatch)
    h.ctx["spacing"] = (1.6, 0.8, 0.8)       # QC 0.8 / 0.4 um, halved
    h.ctx["downsample_factor"] = 2
    h._start_add_whole()
    from .test_add_outline_wiring import _land_info

    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert tuple(kw["spacing"]) == (1.6, 0.8, 0.8) == tuple(
        h._refine_spacing())
    seen = []

    class _Stop(Exception):
        pass

    def _fill(existing, channels, spec, **k):
        seen.append((spec.params["growth"], tuple(k["spacing"])))
        raise _Stop()

    monkeypatch.setattr(R, "reshape_empty_region", _fill)
    monkeypatch.setattr(AP, "_warm_up", lambda progress: (AP.NUMBA_OK, ""))
    try:
        AP.run_add_preview(**kw)
    except _Stop:
        pass
    assert seen == [("fill_with_background", (1.6, 0.8, 0.8))]
    # a spacing never set: the whole patch runs as a box does, on the
    # spacing it has (the page shows the amber default-spacing note)
    h.ctx["spacing_source"] = "unknown"
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert kw is not None and kw["params"]["whole_patch"] is True
