"""Crop, then copy -- every Add reader of an image layer ("B4", B4.1).

A channel may be lazy, multiscale or 4-D; ``np.asarray(layer.data)`` would
materialise the WHOLE patch to use one box.  Every Add path therefore
resolves ONE pyramid level first (``_refine_channel_source``) and reads
only the box from it.  Where that cut happens depends on the path:

- the Nuclei preset, the box info (Brightness histogram, Auto, the dark
  threshold) and the labels being edited (``existing_roi``): the main
  thread slices the box and copies it;
- the Cells preset's image channels (the wall channel, every dark channel,
  the nuclei-masks channel): the main thread hands the worker the layer's
  own array UNREAD, and the worker cuts the box (the engine slices before
  it reads).

The layers here have data whose ``__array__`` RAISES -- any whole-layer
read fails the test -- while ``__getitem__`` with a 3-D box (or the
worker's strided read) returns a real array and records the key it was
read with.

Qt-free: the real orchestrator methods on ``add_harness.Harness``.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.annotator import curation_3d_widget as CW
from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread, Harness, install_fakes
from .test_add_nuclei_golden import box_of, nuclei_fixture


class WholeReadError(AssertionError):
    pass


class RaisingData:
    """A lazy array: whole-array reads raise, box reads return a crop.

    ``data[..., 0]`` / ``data[0]`` (the 4-D squeeze) stay lazy; a key of
    slices only (the box crop, the worker's ``[::stride]`` sample) returns a
    real ndarray."""

    def __init__(self, a):
        self._a = np.asarray(a)
        self.shape = self._a.shape
        self.ndim = self._a.ndim
        self.dtype = self._a.dtype
        self.reads = []

    def __array__(self, *a, **k):
        raise WholeReadError("np.asarray of the whole layer")

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if all(isinstance(k, slice) for k in key) and len(key) == self.ndim:
            self.reads.append(key)
            # a VIEW, as zarr / dask-backed numpy slicing can be: the
            # ``not np.shares_memory`` asserts below then prove the readers
            # COPY (final fix pass B8 -- a copy here made them vacuous)
            return self._a[key]
        return RaisingData(self._a[key])

    def __len__(self):
        return self.shape[0]


class FakeImage:
    """Stands in for a napari Image layer (``CW.Image`` is patched to
    accept it)."""

    def __init__(self, name, data, multiscale=False):
        self.name = name
        self.data = data
        self.multiscale = multiscale
        self._epicycle_temporary = False


class FakeMultiScale:
    """napari's ``MultiScaleData``: a sequence of levels that answers
    ``ndim`` / ``shape`` for level 0 (the channel lists read those)."""

    def __init__(self, levels):
        self._levels = levels
        self.ndim = levels[0].ndim
        self.shape = levels[0].shape

    def __getitem__(self, i):
        return self._levels[i]

    def __len__(self):
        return len(self._levels)

    def __array__(self, *a, **k):
        raise WholeReadError("np.asarray of the whole multiscale layer")


def _layer(name, a, kind):
    if kind == "3d":
        return FakeImage(name, RaisingData(a))
    if kind == "4d":           # (Z, Y, X, 3) -- the 4-D the combos list
        other = np.full_like(a, 7.0)
        return FakeImage(name, RaisingData(np.stack([a, other, other], -1)))
    if kind == "multiscale":                      # level 0 is the one read
        return FakeImage(name, FakeMultiScale(
            [RaisingData(a), RaisingData(a[::2, ::2, ::2])]), multiscale=True)
    raise ValueError(kind)


@pytest.fixture
def patched(monkeypatch):
    install_fakes(monkeypatch)
    from napari.layers import Image

    monkeypatch.setattr(CW, "Image", (Image, FakeImage))


def _nuclei_harness(kind):
    img, labels = nuclei_fixture(0)
    h = Harness(labels)
    h.viewer.layers = [_layer("nuc", img, kind)]
    h.place_box(box_of(labels.shape))
    h.adv.params["channel"] = "nuc"
    return h, img, labels


def _cells_harness(kind, source=("labels", "run1"), box=None):
    from .test_reshape_region import _tissue

    mem, nuc, cells = _tissue()
    h = Harness(np.zeros_like(cells), spacing=(0.5, 0.5, 0.5))
    h.viewer.layers = [_layer("mem", mem, kind), _layer("nuc", nuc, kind)]
    h.adv.preset = "cells"
    h.adv.source = source
    h._add_run_guide = cells.copy()
    h.place_box(box if box is not None
                else tuple(slice(0, n) for n in cells.shape))
    return h, mem, nuc, cells


#: a box strictly inside the (12, 40, 40) tissue on every axis, so a cut of
#: the wrong region (or of the whole layer) cannot match the box's crop
INNER_BOX = (slice(1, 11), slice(3, 37), slice(2, 38))


def _seed_dark(h):
    h._add_dark_cache[h._add_dark_key()] = {
        "t": 0.1, "lo": {"mem": 0.0, "nuc": 0.0},
        "hi": {"mem": 1.0, "nuc": 1.0}}


def _run_preview(kw):
    """One worker run of the gathered kwargs (the arrays the main thread
    copied are copied again, so two runs never share a buffer)."""
    kw = dict(kw)
    for k in ("existing_roi", "run_guide_roi"):
        if kw.get(k) is not None:
            kw[k] = np.array(kw[k], copy=True)
    return AP.run_add_preview(**kw)


def _same_preview(got, want):
    assert got.status == want.status == "ok", (got.refusal, want.refusal)
    assert got.roi == want.roi
    assert got.ids == want.ids
    assert got.ids, "the run added nothing: the comparison would be vacuous"
    np.testing.assert_array_equal(got.result, want.result)


def _reads_of(arr):
    """Every key the fake layer array was read with (box reads only)."""
    assert isinstance(arr, RaisingData), type(arr)
    return list(arr.reads)


KINDS = ["3d", "4d", "multiscale"]


@pytest.mark.parametrize("kind", KINDS)
def test_the_fake_layer_really_refuses_a_whole_read(kind):
    img, _ = nuclei_fixture(0)
    layer = _layer("nuc", img, kind)
    with pytest.raises(WholeReadError):
        np.asarray(layer.data)
    if layer.multiscale:
        with pytest.raises(WholeReadError):
            np.asarray(layer.data[0])


@pytest.mark.parametrize("kind", KINDS)
def test_the_box_info_brightness_crop_crops_then_copies(patched, kind):
    """The Brightness histogram's and Auto's crop (``_add_gather_info``'s
    ``image_roi``; ``_get_add_channel_roi``, which had no production caller,
    is gone -- final fix pass B7)."""
    h, img, _ = _nuclei_harness(kind)
    roi = h._add_roi
    crop = h._add_gather_info()["image_roi"]
    assert crop is not None
    assert crop.dtype == np.float64
    np.testing.assert_array_equal(crop, img[roi].astype(np.float64))
    assert not np.shares_memory(crop, img)


def test_the_raising_fake_returns_a_view():
    """B8: the box read of the fake is a VIEW, so a reader that forgot to
    copy would share memory and fail the asserts above."""
    img, _ = nuclei_fixture(0)
    data = RaisingData(img)
    roi = box_of(img.shape)
    assert np.shares_memory(data[roi], img)


@pytest.mark.parametrize("kind", KINDS)
def test_the_cells_nuclei_masks_preview_the_worker_cuts_the_box(patched,
                                                                kind):
    """The Cells preset's nuclei-masks source: ``_add_gather_preview``
    hands the worker the nuclei channel's own array without reading it, and
    the worker reads exactly the box -- its answer is the answer on a
    hand-made crop of the box."""
    h, mem, nuc, cells = _cells_harness(kind, source=("nuclei_masks", None),
                                        box=INNER_BOX)
    h.adv.nuclei_channel = "nuc"
    _seed_dark(h)
    h._add_info = AP.AddBoxInfo(empty_share=1.0, nuclei_auto=0.5)
    kw, why = h._add_gather_preview()
    assert why is None, why
    roi = h._add_roi
    lazy = kw["nuclei_roi"]
    # the main thread read nothing of the channel
    assert _reads_of(lazy) == []
    got = _run_preview(kw)                 # no WholeReadError on the worker
    reads = _reads_of(lazy)
    assert reads and all(r == roi for r in reads), reads
    want = _run_preview(dict(
        kw, nuclei_roi=nuc[roi].astype(np.float64),
        wall_roi=mem[roi].astype(np.float32),
        dark_rois=(None if kw["dark_rois"] is None else {
            "mem": mem[roi].astype(np.float32),
            "nuc": nuc[roi].astype(np.float32)})))
    _same_preview(got, want)


@pytest.mark.parametrize("kind", KINDS)
def test_the_nuclei_preview_reads_only_the_box(patched, kind):
    h, img, _ = _nuclei_harness(kind)
    h._add_threshold_value = 0.5
    kw, why = h._add_gather_preview()
    assert why is None, why
    got = kw["image_roi"]
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, img[h._add_roi].astype(np.float64))
    assert not np.shares_memory(got, img)


@pytest.mark.parametrize("kind", KINDS)
def test_the_cells_preview_the_worker_cuts_the_box(patched, kind):
    """The Cells preset on a box: the wall channel and every dark channel
    reach the worker as the layers' own arrays, unread by the main thread;
    the worker reads exactly the box of each, and its answer is the answer
    on hand-made float crops of the box."""
    h, mem, nuc, cells = _cells_harness(kind, box=INNER_BOX)
    _seed_dark(h)
    h._add_info = AP.AddBoxInfo(empty_share=1.0, run_count=(8, 8))
    kw, why = h._add_gather_preview()
    assert why is None, why
    roi = h._add_roi
    assert set(kw["dark_rois"]) == {"mem", "nuc"}
    lazies = [kw["wall_roi"], kw["dark_rois"]["mem"], kw["dark_rois"]["nuc"]]
    for lazy in lazies:
        assert _reads_of(lazy) == []       # the main thread read nothing
    # the labels being edited are still a box copy made on the main thread
    assert isinstance(kw["existing_roi"], np.ndarray)
    assert kw["existing_roi"].shape == tuple(s.stop - s.start for s in roi)
    got = _run_preview(kw)                 # no WholeReadError on the worker
    for lazy in lazies:
        reads = _reads_of(lazy)
        assert reads and all(r == roi for r in reads), reads
    want = _run_preview(dict(
        kw, wall_roi=mem[roi].astype(np.float32),
        dark_rois={"mem": mem[roi].astype(np.float32),
                   "nuc": nuc[roi].astype(np.float32)}))
    _same_preview(got, want)


@pytest.mark.parametrize("kind", KINDS)
def test_the_box_info_reads_only_the_box_and_the_stride(patched, kind):
    h, mem, nuc, cells = _cells_harness(kind)
    h._request_add_box_info()
    th = FakeThread.instances[-1]
    th.run()
    assert h._add_info is not None
    assert h._add_info.dark_threshold is not None
    assert h._add_info.run_count is not None


@pytest.mark.parametrize("kind", KINDS)
def test_the_cells_nuclei_auto_reads_only_the_box(patched, kind):
    h, mem, nuc, cells = _cells_harness(kind, source=("nuclei_masks", None))
    h.adv.nuclei_channel = "nuc"
    h._request_add_box_info()
    FakeThread.instances[-1].run()
    assert h._add_info is not None
    assert h._add_info.nuclei_auto is not None


@pytest.mark.parametrize("kind", KINDS)
def test_the_brightness_histogram_and_auto_read_only_the_box(patched, kind):
    """B4.2: the histogram and its Auto land from the box-info worker,
    which is handed the box crop -- never the layer."""
    from skimage.filters import threshold_otsu

    h, img, _ = _nuclei_harness(kind)
    h._request_add_box_info()
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    want = float(threshold_otsu(img[h._add_roi].astype(np.float64)))
    assert h._add_info.brightness_auto == want
    assert h.adv.last("histogram_data")["counts"]
    h._add_threshold_value = None
    h._on_add_otsu_reset()
    assert h._add_threshold_value == want
