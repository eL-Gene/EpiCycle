"""Add's smoothing runs its objects on a thread pool and gives the same volume.

``smooth_pieces`` smooths every new object through ``run_batch_recipe``.  Each
object's compute reads only the untouched input; the writes run after every
compute, sorted by id, so which thread finishes first cannot reach the output.
These tests hold that: the pool's volume equals the one-thread volume voxel for
voxel, also when the finishing order is scrambled on a fixture where two
smoothed objects really do claim the same voxels.  A Stop still lands between
objects and an object that raises still hands the input back unsmoothed.
"""

from __future__ import annotations

import random
import threading

import numpy as np
import pytest

from packages.epicycle_annotator import add_segment as AS
from packages.epicycle_annotator import batch_refine as BR
from packages.epicycle_annotator import pool_workers as PW
from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

SIGMA = 1.5


def _close_pair():
    """Two ragged boxes one voxel apart."""
    rng = np.random.default_rng(2)
    lab = np.zeros((16, 30, 30), np.int32)
    lab[3:13, 4:14, 4:26] = 11
    lab[3:13, 15:26, 4:26] = 12
    lab[(rng.random(lab.shape) < 0.1) & (lab != 0)] = 0      # ragged
    return lab


def _grow_two(volume_roi, label_id, **_k):
    """A stand-in per-object op that GROWS each object by two voxels, so
    neighbours one voxel apart certainly claim the same background voxels:
    the write order alone decides who keeps them."""
    from scipy.ndimage import binary_dilation

    mask = volume_roi == label_id
    grown = binary_dilation(mask, iterations=2)
    out = np.zeros_like(volume_roi)
    out[grown] = label_id
    return out, int(np.count_nonzero(grown != mask)), ""


def _many_blobs(n=40, seed=7):
    """``n`` jagged blobs, some touching, in one volume."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((24, 64, 64), np.int32)
    for i in range(1, n + 1):
        z, y, x = rng.integers(3, 21), rng.integers(4, 60), rng.integers(4, 60)
        r = rng.integers(2, 5)
        zz, yy, xx = np.ogrid[:24, :64, :64]
        ball = ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= r * r
        ball &= rng.random(ball.shape) > 0.15        # ragged surface
        lab[ball & (lab == 0)] = i
    return lab


def _serial(lab):
    """The one-thread reference: the same backend, ``n_workers=1``."""
    recipe = [BR.RefineStep(method="sdf",
                            params={"sigma_z": SIGMA, "sigma_xy": SIGMA})]
    ids = [int(v) for v in np.unique(lab) if v]
    return BR.run_batch_recipe(lab, ids, recipe, n_workers=1,
                               empty="erase").working_volume


def _claims(lab):
    """Per object, the background voxels its per-object result claims."""
    recipe = [BR.RefineStep(method="sdf",
                            params={"sigma_z": SIGMA, "sigma_xy": SIGMA})]
    margin = BR._recipe_margin(recipe, lab.shape)
    out = {}
    for lid in [int(v) for v in np.unique(lab) if v]:
        _l, roi, res, _n = BR._compute_one(lab, lid, margin, recipe)
        full = np.zeros(lab.shape, bool)
        full[roi] = res == lid
        out[lid] = full & (lab == 0)
    return out


def _set_cores(monkeypatch, cpus, affinity=None):
    """The machine reports ``cpus`` cores; ``affinity`` (a count, or None
    for a platform without ``sched_getaffinity``) is what the process may
    use."""
    monkeypatch.setattr(PW.os, "cpu_count", lambda: cpus)
    if affinity is None:
        monkeypatch.delattr(PW.os, "sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr(PW.os, "sched_getaffinity",
                            lambda _pid: set(range(affinity)), raising=False)


def _spy_workers(monkeypatch):
    seen = {}
    real = BR.run_batch_recipe

    def spy(*a, **k):
        seen["n_workers"] = k.get("n_workers")
        return real(*a, **k)

    monkeypatch.setattr(BR, "run_batch_recipe", spy)
    return seen


#: (cores the machine reports, cores the process may use, expected count)
CORE_CASES = [(1, None, 1), (2, None, 1), (3, None, 1), (4, None, 2),
              (8, None, 6), (64, None, 8), (None, None, 1), (64, 4, 2)]


def test_the_fixture_has_contested_voxels(monkeypatch):
    """Teeth: without contention the order test below would prove nothing."""
    monkeypatch.setattr(BR, "smooth_label_sdf", _grow_two)
    claims = _claims(_close_pair())
    assert np.count_nonzero(claims[11] & claims[12]) > 50


def test_smooth_pieces_asks_for_the_pool(monkeypatch):
    seen = _spy_workers(monkeypatch)
    _set_cores(monkeypatch, 20)
    AS.smooth_pieces(_close_pair(), SIGMA)
    assert seen["n_workers"] == AS.ADD_SMOOTH_MAX_WORKERS == 8


@pytest.mark.parametrize("cpus,affinity,want", CORE_CASES)
def test_worker_count_uses_the_cores_the_process_may_use(monkeypatch, cpus,
                                                         affinity, want):
    """min(8, usable - 2), at least 1; usable = the process's CPU affinity
    where the platform has one (a cluster job, a container), else the
    machine's count; an unknown count is one worker."""
    from packages.epicycle_annotator.reshape import walls as W

    _set_cores(monkeypatch, cpus, affinity)
    assert PW.pool_worker_count(8) == want
    assert AS.smooth_worker_count() == want
    assert W.ridge_worker_count() == want


def test_an_unreadable_affinity_falls_back_to_the_machine_count(monkeypatch):
    def broken(_pid):
        raise OSError("no affinity here")

    monkeypatch.setattr(PW.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(PW.os, "sched_getaffinity", broken, raising=False)
    assert PW.pool_worker_count(8) == 4


@pytest.mark.parametrize("cpus,affinity,want", CORE_CASES)
def test_output_equals_the_serial_reference_at_every_count(monkeypatch, cpus,
                                                           affinity, want):
    lab = _many_blobs()
    ref = _serial(lab)
    seen = _spy_workers(monkeypatch)
    _set_cores(monkeypatch, cpus, affinity)
    assert np.array_equal(AS.smooth_pieces(lab.copy(), SIGMA), ref)
    assert seen["n_workers"] == want


@pytest.mark.parametrize("make", [_close_pair, _many_blobs])
def test_pool_equals_the_serial_reference(monkeypatch, make):
    """On a pool for real (20 cores reported, 8 workers asked for)."""
    lab = make()
    ref = _serial(lab)
    seen = _spy_workers(monkeypatch)
    _set_cores(monkeypatch, 20)
    assert np.array_equal(AS.smooth_pieces(lab.copy(), SIGMA), ref)
    assert seen["n_workers"] == 8


@pytest.mark.parametrize("order", ["reversed", "shuffled"])
@pytest.mark.parametrize("make", [_close_pair, _many_blobs])
@pytest.mark.parametrize("op", ["sdf", "grow"])
def test_scrambled_completion_order_gives_the_same_volume(monkeypatch, make,
                                                          order, op):
    """Hand the futures back in a scrambled order: the volume is unchanged."""
    from concurrent.futures import wait

    def scrambled(fs):
        fs = list(fs)
        wait(fs)
        if order == "reversed":
            fs.reverse()
        else:
            random.Random(3).shuffle(fs)
        return iter(fs)

    if op == "grow":
        monkeypatch.setattr(BR, "smooth_label_sdf", _grow_two)
    lab = make()
    want = _serial(lab)
    monkeypatch.setattr(BR, "as_completed", scrambled)
    _set_cores(monkeypatch, 20)
    got = AS.smooth_pieces(lab.copy(), SIGMA)
    assert np.array_equal(got, want)
    if op == "grow" and make is _close_pair:
        # the lower id keeps every voxel both claimed, whoever finished first
        claims = _claims(lab)
        both = claims[11] & claims[12]
        assert both.any() and np.all(got[both] == 11)


def test_stop_lands_between_objects(monkeypatch):
    """A Stop set while the pool runs: the call is cancelled, the objects
    queued behind the Stop never ran, and nothing was written (the input
    is untouched; nothing is returned)."""
    ev = threading.Event()
    started = []
    lock = threading.Lock()
    real = BR._compute_one

    def first_sets_stop(*a, **k):
        with lock:
            started.append(1)
        out = real(*a, **k)
        ev.set()
        return out

    monkeypatch.setattr(BR, "_compute_one", first_sets_stop)
    _set_cores(monkeypatch, 20)
    lab = _many_blobs()
    before = lab.copy()
    n = len([v for v in np.unique(lab) if v])
    with pytest.raises(ReshapeCancelled):
        AS.smooth_pieces(lab, SIGMA, cancel_event=ev)
    assert len(started) < n, "the objects queued behind the Stop never ran"
    assert np.array_equal(lab, before)


def test_a_failing_object_returns_the_input(monkeypatch):
    """One object raising inside a worker: the input comes back unsmoothed,
    as it did on one thread."""
    real = BR.smooth_label_sdf

    def fail_on_12(roi, lid, **k):
        if lid == 12:
            raise RuntimeError("boom")
        return real(roi, lid, **k)

    monkeypatch.setattr(BR, "smooth_label_sdf", fail_on_12)
    _set_cores(monkeypatch, 20)
    lab = _close_pair()
    out = AS.smooth_pieces(lab.copy(), SIGMA)
    assert np.array_equal(out, lab)
