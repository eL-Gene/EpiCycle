"""The ridge wall pictures (Vessel ridges = frangi, Thin-faint ridges =
meijering) are built by our own loop over the scales, equal to skimage's.

The loop exists so that a Stop lands between scales and between the Gaussian
passes (one skimage call ran two minutes on a whole patch with no way in), and
so the passes can run on threads.  It must stay equal to the installed skimage
call, array for array: these tests compare the two on small volumes, one with
noisy membranes, one with wide exactly-flat areas (where a shorter Gaussian
kernel at sigma 1 would show first), and pin the skimage version the loop was
checked against, so an upgrade turns red here instead of drifting quietly.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from scipy import ndimage as ndi

from packages.epicycle_annotator.reshape import engine as E
from packages.epicycle_annotator.reshape import walls as W
from packages.epicycle_annotator.reshape.engine import ReshapeCancelled
from packages.epicycle_annotator.reshape.preproc import clahe_stack, norm

from .reshape_fixtures import two_columns

KINDS = ("frangi", "meijering")


def _sheet(seed=4, shape=(14, 44, 44)):
    """A noisy sheet of cells: bright walls between random seeds."""
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, shape, size=(14, 3))
    markers = np.zeros(shape, np.int32)
    for i, (z, y, x) in enumerate(pts, start=1):
        markers[z, y, x] = i
    _d, (iz, iy, ix) = ndi.distance_transform_edt(markers == 0,
                                                  return_indices=True)
    owner = markers[iz, iy, ix]
    walls = np.zeros(shape, bool)
    for ax in range(3):
        walls |= owner != np.roll(owner, 1, axis=ax)
    img = 0.3 + 0.6 * walls + rng.normal(0, 0.06, shape)
    return ndi.gaussian_filter(img.astype(np.float32), 0.7)


def _flat_blocks(shape=(16, 48, 48)):
    """Two thin walls in wide exactly-flat space (no noise at all)."""
    v = np.zeros(shape, np.float32)
    v[:, 20:22, :] = 1.0
    v[9:11, :, :] = 1.0
    return v


def _columns():
    return two_columns(0.4, seed=11)[2]


VOLUMES = {"sheet": _sheet, "flat": _flat_blocks, "columns": _columns}


def _skimage(kind, image):
    from skimage import filters as skf

    fn = skf.frangi if kind == "frangi" else skf.meijering
    return fn(image, black_ridges=False)


def test_skimage_version_pinned():
    """The loop mirrors one skimage release line by line; a new release
    must be re-read against it before this pin moves."""
    import skimage

    assert W.RIDGE_SKIMAGE_VERSION == "0.25.2"
    assert skimage.__version__ == W.RIDGE_SKIMAGE_VERSION, (
        "skimage changed: re-read frangi / meijering / the Hessian helpers "
        "against walls.ridge_filter, then move the pin")


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("vol", sorted(VOLUMES))
@pytest.mark.parametrize("kind", KINDS)
def test_ridge_equals_skimage(kind, vol, workers):
    """Bit for bit, on the CLAHE'd picture the wall builder feeds it and on
    the raw volume (float32 both), with one thread and with four."""
    raw = VOLUMES[vol]()
    for base in (clahe_stack(raw, 0.01), raw):
        want = _skimage(kind, base)
        got = W.ridge_filter(base, kind, n_workers=workers)
        assert got.dtype == want.dtype
        assert np.array_equal(got, want), "{} drifted on {}".format(kind, vol)


@pytest.mark.parametrize("kind", KINDS)
def test_ridge_equals_skimage_on_float64_and_2d(kind):
    """A float64 volume keeps float64; a 2-D picture is skimage's own call."""
    raw = _sheet().astype(np.float64)
    assert np.array_equal(W.ridge_filter(raw, kind), _skimage(kind, raw))
    flat = raw[5]
    assert np.array_equal(W.ridge_filter(flat, kind), _skimage(kind, flat))


@pytest.mark.parametrize("kind", KINDS)
def test_build_elevation_matches_the_old_skimage_branch(kind):
    """The wall builder's three outputs are the ones the one-call branch
    made: CLAHE base, the normalised ridge, the elevation."""
    raw = _sheet()
    region = np.ones(raw.shape, bool)
    base = clahe_stack(raw, 0.01)
    ridge = norm(np.asarray(_skimage(kind, base), np.float32))
    elev, clahe, ridge_img = W.build_elevation(kind, raw, region)
    assert np.array_equal(clahe, base)
    assert np.array_equal(ridge_img, ridge)
    assert np.array_equal(elev, np.asarray(ridge, np.float32))


def _count_passes(monkeypatch, stop_at, ev):
    """Count the Gaussian passes; set the Stop during pass ``stop_at``."""
    calls = []
    lock = threading.Lock()
    real = ndi.gaussian_filter

    def spy(*a, **k):
        with lock:
            calls.append(1)
            n = len(calls)
        out = real(*a, **k)
        if n == stop_at:
            ev.set()
        return out

    monkeypatch.setattr(W.ndi, "gaussian_filter", spy)
    return calls


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("kind", KINDS)
def test_stop_lands_between_scales(monkeypatch, kind, workers):
    """A Stop during the first scale's last pass: no pass of the second
    scale starts (each scale is nine passes: three first, six second)."""
    image = clahe_stack(_sheet(), 0.01)       # built before the count starts
    ev = threading.Event()
    calls = _count_passes(monkeypatch, 9, ev)
    with pytest.raises(ReshapeCancelled):
        W.ridge_filter(image, kind, cancel_event=ev, n_workers=workers)
    assert len(calls) == 9


@pytest.mark.parametrize("kind", KINDS)
def test_stop_lands_between_the_hessian_passes(monkeypatch, kind):
    """A Stop during a first-derivative pass: no second-derivative pass."""
    image = clahe_stack(_sheet(), 0.01)       # built before the count starts
    ev = threading.Event()
    calls = _count_passes(monkeypatch, 1, ev)
    with pytest.raises(ReshapeCancelled):
        W.ridge_filter(image, kind, cancel_event=ev, n_workers=1)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", KINDS)
def test_stop_is_not_swallowed(kind):
    """The wall builder's fall-back branch must not eat a Stop: a set event
    raises instead of handing back the plain CLAHE picture."""
    ev = threading.Event()
    ev.set()
    raw = _sheet()
    with pytest.raises(ReshapeCancelled):
        W.build_elevation(kind, raw, np.ones(raw.shape, bool),
                          cancel_event=ev)


def test_a_quiet_stop_changes_nothing():
    raw = _sheet()
    region = np.ones(raw.shape, bool)
    a = W.build_elevation("frangi", raw, region)
    b = W.build_elevation("frangi", raw, region,
                          cancel_event=threading.Event())
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("kind", ["membrane", "gradient", "intensity",
                                  "hybrid", "distance", "skeleton", "none"])
def test_other_kinds_ignore_the_stop_argument(kind):
    """Only the ridge kinds poll the Stop; the rest build as they did."""
    raw = _sheet()
    region = np.zeros(raw.shape, bool)
    region[2:12, 6:38, 6:38] = True
    ev = threading.Event()
    ev.set()
    a = W.build_elevation(kind, raw, region)
    b = W.build_elevation(kind, raw, region, cancel_event=ev)
    for x, y in zip(a, b):
        if x is None:
            assert y is None
        else:
            assert np.array_equal(x, y)


def test_wall_stage_hands_its_stop_to_the_builder(monkeypatch):
    """The engine's wall stage passes its own Stop to the wall builder."""
    from packages.epicycle_annotator.reshape import ReshapeSpec

    seen = {}
    real = E.build_elevation

    def spy(*a, **k):
        seen["cancel_event"] = k.get("cancel_event")
        return real(*a, **k)

    monkeypatch.setattr(E, "build_elevation", spy)
    _labels, _guide, mem, region = two_columns(0.4, seed=11)
    spec = ReshapeSpec(target="label", seeds="labels", walls="frangi",
                       params={"wall_channel": "cadherin"}, arbiter=False)
    ev = threading.Event()
    p = dict(spec.params)
    for key, val in {"wall_preproc": "none", "wall_gamma": 1.0, "wall_bg": 0,
                     "wall_denoise": 0, "clip": 0.01, "smooth": 1.0,
                     "ridge": True}.items():
        p.setdefault(key, val)
    roi = tuple(slice(0, n) for n in region.shape)

    class _Sink:
        def emit(self, *_a, **_k):
            pass

    E._build_wall_stage(spec, images={"cadherin": mem}, roi=roi,
                        region=region, spacing=(1, 1, 1), params=p,
                        sink=_Sink(), cancel_event=ev)
    assert seen["cancel_event"] is ev


# --------------------------------------------------------------------------- #
# only the engine's wall stage (Add / Reshape) builds the picture on a pool
# --------------------------------------------------------------------------- #
def _count_executors(monkeypatch):
    """Record every thread pool the wall builder creates."""
    made = []
    real = W.ThreadPoolExecutor

    class _Spy(real):
        def __init__(self, *a, **k):
            made.append(k.get("max_workers", a[0] if a else None))
            super().__init__(*a, **k)

    monkeypatch.setattr(W, "ThreadPoolExecutor", _Spy)
    return made


@pytest.mark.parametrize("kind", KINDS)
def test_a_caller_without_the_keyword_creates_no_pool(monkeypatch, kind):
    """Refine and Settle call the builder per object inside Batch Refine's
    own pool: they get the loop on one thread, no executor, same values."""
    raw = _sheet()
    region = np.ones(raw.shape, bool)
    want = norm(np.asarray(_skimage(kind, clahe_stack(raw, 0.01)), np.float32))
    made = _count_executors(monkeypatch)
    elev, _c, ridge = W.build_elevation(kind, raw, region)
    assert made == []
    assert np.array_equal(ridge, want)
    assert np.array_equal(W.ridge_filter(clahe_stack(raw, 0.01), kind),
                          _skimage(kind, clahe_stack(raw, 0.01)))
    assert made == []


@pytest.mark.parametrize("kind", KINDS)
def test_refine_and_settle_wall_builders_create_no_pool(monkeypatch, kind):
    from packages.epicycle_annotator.refine import contour_with_walls
    from packages.epicycle_annotator.reshape.settle import settle_with_walls

    labels, _guide, mem, _region = two_columns(0.4, seed=11)
    made = _count_executors(monkeypatch)
    settle_with_walls(labels, 7, wall_kind=kind, channel_roi=mem)
    contour_with_walls(labels, 7, mode="snap", wall_kind=kind,
                       channel_roi=mem)
    assert made == []


def test_wall_stage_builds_on_the_pooled_count(monkeypatch):
    """The engine's wall stage (the path with a Stop) asks for the pool."""
    from packages.epicycle_annotator import pool_workers as PW
    from packages.epicycle_annotator.reshape import ReshapeSpec

    monkeypatch.setattr(PW.os, "cpu_count", lambda: 20)
    monkeypatch.delattr(PW.os, "sched_getaffinity", raising=False)
    seen = {}
    real = E.build_elevation

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(E, "build_elevation", spy)
    made = _count_executors(monkeypatch)
    _labels, _guide, mem, region = two_columns(0.4, seed=11)
    spec = ReshapeSpec(target="label", seeds="labels", walls="frangi",
                       params={"wall_channel": "cadherin"}, arbiter=False)

    class _Sink:
        def emit(self, *_a, **_k):
            pass

    roi = tuple(slice(0, n) for n in region.shape)
    E._build_wall_stage(spec, images={"cadherin": mem}, roi=roi,
                        region=region, spacing=(1, 1, 1),
                        params=dict(spec.params), sink=_Sink(),
                        cancel_event=threading.Event())
    assert seen["n_workers"] == 8
    assert made == [8]


# --------------------------------------------------------------------------- #
# the dtype rule is our own: a missing private skimage helper changes nothing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
def test_a_missing_private_skimage_helper_changes_nothing(monkeypatch, kind):
    import sys

    raw = _sheet()
    region = np.ones(raw.shape, bool)
    want = norm(np.asarray(_skimage(kind, clahe_stack(raw, 0.01)), np.float32))
    monkeypatch.setitem(sys.modules, "skimage._shared.utils", None)
    _elev, _c, ridge = W.build_elevation(kind, raw, region)
    assert np.array_equal(ridge, want)


def test_the_float_rule_matches_skimage():
    from skimage._shared.utils import _supported_float_type

    for dt in (np.bool_, np.uint8, np.int16, np.int32, np.int64, np.uint64,
               np.float16, np.float32, np.float64, np.longdouble):
        assert W._ridge_float_type(np.dtype(dt)) == \
            np.dtype(_supported_float_type(np.dtype(dt))), dt


# --------------------------------------------------------------------------- #
# a stopped ridge build leaves no pool thread behind
# --------------------------------------------------------------------------- #
def _ridge_threads():
    return [t for t in threading.enumerate()
            if t.name.startswith(W.RIDGE_THREAD_PREFIX) and t.is_alive()]


@pytest.mark.parametrize("kind", KINDS)
def test_ridge_threads_are_joined_after_a_stop(monkeypatch, kind):
    image = clahe_stack(_sheet(), 0.01)
    ev = threading.Event()
    _count_passes(monkeypatch, 5, ev)
    with pytest.raises(ReshapeCancelled):
        W.ridge_filter(image, kind, cancel_event=ev, n_workers=4)
    assert _ridge_threads() == []
    W.ridge_filter(image, kind, n_workers=4)
    assert _ridge_threads() == []
