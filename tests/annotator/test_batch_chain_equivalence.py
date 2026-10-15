"""Batch == Refine tab.

The Refine tab is driven on the ``_recipe_fake`` panel -- the REAL
``_preview_smooth`` and its tails, the REAL ``AdvancedFeaturesWidget`` and,
for Settle and the contour, the REAL ``RefineWorker`` on a QThread -- step
by step, as a user (or the recipe runner) would.  The batch then runs the
same steps as a saved ``Recipe`` on the same starting volume, and the two
volumes must be byte-identical: every method, chains, the clip cases, a
refusal at step k (both keep steps 1..k-1) and an object at the volume's
edge.  Plus the parity of the kwargs, the margins and the room.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from packages.epicycle_annotator import batch_chain as bc
from packages.epicycle_annotator import refine_recipe as rr
from packages.epicycle_annotator.batch_refine import run_batch_recipe
from tests.annotator import _recipe_fake as rf
from tests.annotator.test_batch_chain_dispatch import (
    contour,
    dil,
    fill,
    guided,
    sdf,
    settle,
)

SPACING = (3.0, 1.0, 1.0)


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _labels(monkeypatch, qapp):
    import napari_epicycle.annotator.curation_3d_widget as mod

    monkeypatch.setattr(mod, "Labels", rf._Layer)


@pytest.fixture
def panel(qapp):
    w = rf.make_panel()
    w.set_refine_channels(["membrane"], current="membrane")
    yield w
    w.setParent(None)


def pump(qapp, predicate, seconds=30.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        qapp.processEvents()
        if predicate():
            qapp.processEvents()
            return True
        time.sleep(0.01)
    return False


def start_vol():
    v = rf.vol_with_notch()
    v[6:22, 10:30, 32:38] = 3                   # a neighbour, for Settle
    return v


def walls(shape):
    img = np.zeros(shape, np.float64)
    img[:, 7, 7:33] = img[:, 32, 7:33] = 1.0
    img[:, 7:33, 7] = img[:, 7:33, 32] = 1.0
    return img


def tab_run(qapp, panel, vol, steps, *, prob=None, walls_fn=None):
    """The Refine tab, one Preview per step; the staged volume after."""
    ch = (walls_fn or walls)(vol.shape)
    f = rf.Fake(vol.copy(), panel, channel_roi=ch, prob=prob,
                spacing=SPACING)
    for st in steps:
        ok, why = panel.set_refine_step_params(st.method, dict(st.params))
        assert ok, why
        f._preview_smooth()
        assert pump(qapp, lambda: not f._refine_running), "worker hung"
    return f.vol().copy(), ch


def batch_run(vol, steps, ch, prob=None):
    res = run_batch_recipe(vol.copy(), [7], rr.Recipe(steps=tuple(steps)),
                           spacing=SPACING, channel=ch, prob=prob,
                           n_workers=1)
    return res


CASES = {
    "fill-smooth-dilate": (fill(), sdf(1.5, 1.0), dil(2)),
    "dilate-twice": (dil(3), dil(3)),
    "smooth-clip": (sdf(5.0, 5.0),),
    "smooth-settle": (sdf(1.5, 1.0), settle(band=3.0)),
    "settle-smooth": (settle(band=2.0), sdf(1.5, 1.0)),
    "contour-xy": (contour(axis="XY", limit=2.0),),
    "smooth-contour-3d": (sdf(1.5, 1.0), contour(axis="3D", limit=2.0)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_batch_equals_the_refine_tab(qapp, panel, name):
    steps = CASES[name]
    vol = start_vol()
    staged, ch = tab_run(qapp, panel, vol, steps)
    res = batch_run(vol, steps, ch)
    assert not res.stopped and not res.emptied_ids
    assert np.array_equal(res.working_volume, staged), name


def test_a_refusal_at_step_k_keeps_the_same_steps(qapp, panel):
    """Smooth, then a contour whose push is below its gate: both keep the
    Smooth."""
    steps = (sdf(1.5, 1.0), contour(axis="XY", balloon=0.1, gate=0.9))
    vol = start_vol()
    staged, ch = tab_run(qapp, panel, vol, steps)
    res = batch_run(vol, steps, ch)
    assert res.stopped == {7: (2, res.stopped[7][1])}
    assert np.array_equal(res.working_volume, staged)
    assert not np.array_equal(res.working_volume, vol)     # the Smooth


def test_guided_grow_equals_the_refine_tab(qapp, panel):
    vol = start_vol()
    prob = np.zeros(vol.shape, np.float32)
    prob[4:24, 8:32, 8:32] = 0.9
    steps = (guided(3, 0.5),)
    staged, ch = tab_run(qapp, panel, vol, steps, prob=prob)
    res = batch_run(vol, steps, ch, prob=prob)
    assert np.array_equal(res.working_volume, staged)


def test_an_object_at_the_volume_edge(qapp, panel):
    vol = np.zeros((20, 40, 40), np.int32)
    vol[0:8, 0:14, 0:14] = 7
    vol[0:8, 0:3, 0:3] = 0
    steps = (sdf(1.5, 1.0), dil(2))
    staged, ch = tab_run(qapp, panel, vol, steps)
    res = batch_run(vol, steps, ch)
    assert np.array_equal(res.working_volume, staged)


# ------------------------------------------------ an interior object
#
# The cases above live in a 28 x 40 x 40 volume whose object sits so close
# to every wall that the margin of 15 clamps each ROI to the whole volume:
# clipping, growth and the per-crop normalisation are never exercised
# there (data audit of commits 6-8, D-TESTGAP).  Here the object sits deep
# inside a 60 x 90 x 90 volume, so every ROI is a real interior crop that
# grows between steps.


def big_vol():
    v = np.zeros((60, 90, 90), np.int32)
    v[22:38, 35:55, 35:55] = 7
    v[22:38, 35:38, 35:38] = 0                  # a notch
    v[28:32, 43:47, 43:47] = 0                  # a hole
    v[22:38, 35:55, 57:63] = 3                  # a neighbour, for Settle
    return v


def big_walls(shape):
    img = np.random.RandomState(0).rand(*shape) * 0.2
    img[:, 32, 32:58] = img[:, 57, 32:58] = 1.0
    img[:, 32:58, 32] = img[:, 32:58, 57] = 1.0
    return img


def brightness(axis="XY"):
    s = contour(axis=axis, limit=2.0)
    p = dict(s.params)
    p["mode"] = "brightness"
    return rr.RecipeStep("contour", p)


INTERIOR_CASES = {
    "dil3-dil3-dil3": (dil(3), dil(3), dil(3)),
    "smooth-clip": (sdf(5.0, 5.0),),
    "dil-neg": (sdf(1.5, 1.0), dil(-2)),
    "fill-xy": (fill("XY"), sdf(1.5, 1.0)),
    "settle3": (settle(band=3.0),),
    "settle-distance": (dil(4), settle(kind="distance", band=4.0)),
    "contour-xy": (contour(axis="XY", limit=2.0),),
    "contour-big-limit": (dil(5), contour(axis="XY", limit=25.0)),
    "brightness": (brightness(),),
}


@pytest.mark.parametrize("name", sorted(INTERIOR_CASES))
def test_an_interior_object_equals_the_refine_tab(qapp, panel, name):
    steps = INTERIOR_CASES[name]
    vol = big_vol()
    staged, ch = tab_run(qapp, panel, vol, steps, walls_fn=big_walls)
    res = batch_run(vol, steps, ch)
    assert int(np.count_nonzero(res.working_volume != staged)) == 0, name
    assert not np.array_equal(staged, vol), "the recipe moved nothing"


# ---------------------------------------------------------------- parity


def _fake(panel, vol=None, spacing=SPACING):
    vol = start_vol() if vol is None else vol
    return rf.Fake(vol, panel, channel_roi=walls(vol.shape), spacing=spacing)


P_CONTOUR = [
    dict(contour_mode=0, wall_kind="membrane", contour_axis="3D",
         contour_sampling=0, contour_share_pct=25, contour_n_slices=12,
         contour_limit_um=3.0, contour_iterations=100, contour_balloon=0.0,
         contour_gate=0.3, contour_blur_um=2.0, contour_sharpness=100,
         contour_rebuild_um=1.5, contour_lambda1=1.0, contour_lambda2=1.0,
         prep_chain="bgsub+gamma"),
    dict(contour_mode=1, wall_kind="distance", contour_axis="XZ",
         contour_sampling=1, contour_share_pct=40, contour_n_slices=9,
         contour_limit_um=12.5, contour_iterations=250, contour_balloon=-0.4,
         contour_gate=0.1, contour_blur_um=4.0, contour_sharpness=300,
         contour_rebuild_um=0.0, contour_lambda1=2.0, contour_lambda2=0.5,
         prep_chain="none"),
    dict(contour_mode=0, wall_kind="probability", contour_axis="YZ",
         contour_sampling=2, contour_share_pct=5, contour_n_slices=60,
         contour_limit_um=0.5, contour_iterations=10, contour_balloon=1.0,
         contour_gate=0.0, contour_blur_um=0.5, contour_sharpness=1,
         contour_rebuild_um=15.0, contour_lambda1=0.1, contour_lambda2=10.0,
         prep_chain="none"),
]


def _ticks_for(chain):
    toks = rr.chain_tokens(chain)
    return {"clahe": "clahe" in toks, "zcorrect": "zcorrect" in toks,
            "bgsub": "bgsub" in toks, "lift": "gamma" in toks,
            "denoise": "denoise" in toks}


def _same(a, b):
    assert set(a) == set(b), set(a) ^ set(b)
    for k in a:
        if isinstance(a[k], np.ndarray) or isinstance(b[k], np.ndarray):
            assert a[k] is b[k], k
        else:
            assert a[k] == b[k] and type(a[k]) is type(b[k]), (k, a[k], b[k])


@pytest.mark.parametrize("P", P_CONTOUR)
def test_contour_kwargs_parity(panel, P):
    f = _fake(panel)
    labels = np.zeros((5, 5, 5), np.int32)
    ch = np.ones((5, 5, 5))
    pr = np.ones((5, 5, 5)) * 0.5
    ev = threading.Event()
    step = rr.record_from_params("contour", P,
                                 prep_ticks=_ticks_for(P["prep_chain"]))
    step.validate()
    # the tab's launch decides which crops go in
    needs_ch = P["contour_mode"] == 1 or P["wall_kind"] not in (
        "probability", "distance", "skeleton")
    wants_pr = P["contour_mode"] == 0 and P["wall_kind"] == "probability"
    fn_t, kw_t = f._refine_contour_call(
        P, labels, 7, ch if needs_ch else None, pr if wants_pr else None,
        cancel_event=ev)
    fn_b, kw_b = bc.chain_step_call(step, labels, 7, ch, pr, SPACING, ev)
    assert fn_t is fn_b
    _same(kw_t, kw_b)


P_SETTLE = [
    dict(wall_kind="membrane", solver="watershed", band_um=2.0, n_iter=10,
         clamp=0.5, compactness=0.0, prep_chain="bgsub+gamma"),
    dict(wall_kind="probability", solver="random_walker", band_um=7.5,
         n_iter=3, clamp=0.25, compactness=0.4, prep_chain="none"),
    dict(wall_kind="skeleton", solver="watershed", band_um=0.5, n_iter=50,
         clamp=0.95, compactness=1.0, prep_chain="none"),
]


@pytest.mark.parametrize("P", P_SETTLE)
def test_settle_kwargs_parity(panel, P):
    vol = start_vol()
    prob = np.full(vol.shape, 0.5, np.float32)
    f = _fake(panel, vol)
    f.prob = prob
    got = {}
    f._launch_refine_worker = (
        lambda fn, kwargs, method, label_id, roi, **k:
        got.update(fn=fn, kw=kwargs, roi=roi))
    roi = (slice(2, 26), slice(5, 35), slice(5, 38))
    f._launch_refine_settle(P, roi, vol[roi], 7)
    step = rr.record_from_params("settle", P,
                                 prep_ticks=_ticks_for(P["prep_chain"]))
    step.validate()
    kw_t = dict(got["kw"])
    fn_b, kw_b = bc.chain_step_call(
        step, kw_t["labels_roi"], 7, kw_t["channel_roi"], kw_t["prob_roi"],
        SPACING)
    assert got["fn"] is fn_b
    # the channel / prob crops are the tab's own crops at that ROI
    if kw_t["channel_roi"] is not None:
        assert np.array_equal(kw_t["channel_roi"],
                              walls(vol.shape)[roi].astype(np.float64))
    if kw_t["prob_roi"] is not None:
        assert np.array_equal(kw_t["prob_roi"], prob[roi].astype(np.float64))
    _same(kw_t, kw_b)


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (3.0, 0.5, 0.5),
                                     (0.1, 0.1, 0.1), (2.0, 0.08, 0.08)])
@pytest.mark.parametrize("method,P", [
    ("sdf", {}), ("dilate_erode", {"radius": 5}), ("fill_holes", {}),
    ("guided_grow", {"max_radius": 12}),
    ("contour", {"contour_limit_um": 1.0}),
    ("contour", {"contour_limit_um": 30.0}),
    ("settle", {"band_um": 0.5}), ("settle", {"band_um": 20.0}),
])
def test_margin_parity(panel, spacing, method, P):
    f = _fake(panel, spacing=spacing)
    assert f._refine_step_margin(method, P) == bc.refine_step_margin(
        method, P, spacing)


def test_the_shared_margin_copy():
    from napari_epicycle.annotator.curation_3d_widget import (
        REFINE_SHARED_ROI_MARGIN,
    )

    assert bc.REFINE_SHARED_ROI_MARGIN == REFINE_SHARED_ROI_MARGIN == 15


@pytest.mark.parametrize("roi", [
    (slice(0, 28), slice(0, 40), slice(0, 40)),        # the whole volume
    (slice(2, 26), slice(5, 35), slice(5, 35)),
    (slice(6, 22), slice(10, 30), slice(10, 30)),     # touching the walls
    (slice(0, 24), slice(8, 40), slice(0, 33)),
])
def test_room_parity(panel, roi):
    vol = start_vol()
    f = _fake(panel, vol)
    f._smooth_roi = roi
    f._smooth_label_id = 7
    need, have = f._refine_room("sdf", {})
    assert need == bc.refine_step_margin("sdf", {}, SPACING)
    assert have == bc.refine_room(vol[roi] == 7, roi, vol.shape)
