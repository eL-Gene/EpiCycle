"""``add_preview.add_box_info`` -- the one box-info job.

The empty share, the patch dark threshold (a strided read, only when the
cache misses), the run-count line through ``find_region_seeds`` (equal to
the Preview's marker count), the Markers Auto value, the over-limit refusal
of a run count and a Stop answered as ``cancelled``.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import threading

import numpy as np

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.add_segment import _resolve_threshold
from packages.epicycle_annotator.reshape import region as R

from .test_reshape_region import _tissue

ISO = (1.0, 1.0, 1.0)


def _info(source="labels", *, labels=None, cache=None, stride=1, params=None,
          cancel_event=None, sources=True):
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells) if labels is None else labels
    chans = {"membrane": mem, "nuclei": nuc}
    p = {"preset": "cells", "dark_outside": True}
    p.update(params or {})
    return AP.add_box_info(
        existing_roi=labels, dark_sources=chans if sources else None,
        dark_rois=chans, spacing=ISO, stride=stride, dark_cache=cache,
        source=source, run_guide_roi=cells.copy(), nuclei_roi=nuc, params=p,
        cancel_event=cancel_event), (mem, nuc, cells, chans)


def test_the_empty_share():
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    labels[:, :10, :] = 3
    info, _ = _info(labels=labels)
    assert np.isclose(info.empty_share, 1.0 - 10 / 40)


def test_the_dark_threshold_is_computed_on_a_miss():
    info, (_m, _n, _c, chans) = _info()
    want = R.patch_dark_threshold(chans, spacing=ISO, stride=1)
    assert info.dark_threshold == want


def test_a_cache_hit_is_not_recomputed(monkeypatch):
    called = []
    monkeypatch.setattr(R, "patch_dark_threshold",
                        lambda *a, **k: called.append(1))
    cache = {"t": 0.3, "lo": {"membrane": 0.0, "nuclei": 0.0},
             "hi": {"membrane": 1.0, "nuclei": 1.0}}
    info, _ = _info(cache=cache)
    assert info.dark_threshold is cache
    assert called == []


def test_the_strided_read_is_the_workers():
    reads = []

    class Spy:
        def __init__(self, a):
            self.a = a

        def __getitem__(self, key):
            reads.append(key)
            return self.a[key]

    mem, nuc, cells = _tissue()
    info = AP.add_box_info(
        existing_roi=np.zeros_like(cells),
        dark_sources={"membrane": Spy(mem), "nuclei": Spy(nuc)},
        dark_rois={"membrane": mem, "nuclei": nuc}, spacing=ISO, stride=2,
        dark_cache=None, source="paint", run_guide_roi=None, nuclei_roi=None,
        params={"preset": "cells"})
    assert info.dark_threshold is not None
    assert reads and all(k == (slice(None, None, 2),) * 3 for k in reads)


def test_the_run_count_equals_the_previews_marker_count():
    info, (mem, nuc, cells, chans) = _info("labels")
    assert info.run_count is not None
    n_found, n_markers = info.run_count
    assert n_found == 8
    assert n_markers == 8
    out = AP.run_add_preview(
        preset="cells", roi=tuple(slice(0, n) for n in cells.shape),
        existing_roi=np.zeros_like(cells), next_label=1, spacing=ISO,
        params={"source": "labels", "wall_chain": "none", "auto_smooth": False},
        wall_roi=mem, dark_rois=chans, run_guide_roi=cells.copy(),
        dark_threshold=info.dark_threshold)
    assert out.ok
    assert len(out.ids) <= n_markers


def test_no_run_count_over_the_limit():
    info, _ = _info("labels", params={"box_budget": 100})
    assert info.over_limit is True
    assert info.run_count is None


def test_the_nuclei_auto_is_the_otsu_of_the_crop():
    info, (_m, nuc, _c, _ch) = _info("nuclei_masks")
    assert info.nuclei_auto == _resolve_threshold(nuc.astype(np.float64), "otsu")
    assert info.run_count is None


def test_the_nuclei_preset_asks_for_no_dark_threshold():
    info, _ = _info("labels", params={"preset": "nuclei"})
    assert info.dark_threshold is None
    assert info.run_count is None
    assert info.over_limit is False


def test_a_stop_is_cancelled():
    ev = threading.Event()
    ev.set()
    info, _ = _info(cancel_event=ev)
    assert info.status == "cancelled"


def test_dark_stride_keeps_the_sample_under_budget():
    assert AP.dark_stride((10, 10, 10)) == 1
    s = AP.dark_stride((317, 480, 480))
    n = np.prod([-(-v // s) for v in (317, 480, 480)])
    assert n <= R.DARK_SAMPLE_VOXELS
    n_prev = np.prod([-(-v // (s - 1)) for v in (317, 480, 480)])
    assert n_prev > R.DARK_SAMPLE_VOXELS


# --------------------------------------------------------------------------- #
# B4.2 -- the Brightness histogram and its Auto ride the info worker
# --------------------------------------------------------------------------- #
def _nuclei_info(*, clahe=False, image=True, preset="nuclei", cancel_event=None):
    from .test_add_nuclei_golden import box_of, nuclei_fixture

    img, labels = nuclei_fixture(0)
    roi = box_of(labels.shape)
    crop = img[roi].astype(np.float64)
    info = AP.add_box_info(
        existing_roi=labels[roi].copy(), dark_sources=None, dark_rois=None,
        spacing=ISO, stride=1, dark_cache=None, source="paint",
        run_guide_roi=None, nuclei_roi=None,
        params={"preset": preset, "clahe": clahe, "clahe_clip": 0.01},
        image_roi=crop if image else None, cancel_event=cancel_event)
    return info, crop


def test_the_histogram_is_the_crops_in_the_threshold_space():
    from packages.epicycle_annotator.add_segment import add_threshold_space

    info, crop = _nuclei_info()
    assert info.status == "ok"
    proc = add_threshold_space(crop, clahe=False, clahe_clip=0.01)
    counts, edges = np.histogram(proc, bins=AP.ADD_HIST_BINS)
    assert info.histogram is not None
    assert list(info.histogram["counts"]) == counts.tolist()
    assert np.allclose(info.histogram["edges"], edges)
    assert sum(info.histogram["counts"]) == crop.size


def test_the_brightness_auto_is_the_otsu_of_the_crop_exactly():
    from skimage.filters import threshold_otsu

    info, crop = _nuclei_info()
    assert info.brightness_auto == float(threshold_otsu(crop))


def test_local_contrast_moves_the_histogram_and_the_auto():
    from packages.epicycle_annotator.add_segment import add_threshold_space

    info, crop = _nuclei_info(clahe=True)
    proc = add_threshold_space(crop, clahe=True, clahe_clip=0.01)
    assert info.brightness_auto == _resolve_threshold(proc, "otsu")
    assert 0.0 <= info.histogram["edges"][0]
    assert info.histogram["edges"][-1] <= 1.0 + 1e-9


def test_no_image_crop_no_histogram():
    info, _ = _nuclei_info(image=False)
    assert info.histogram is None
    assert info.brightness_auto is None


def test_the_cells_preset_has_no_brightness_card():
    info, _ = _nuclei_info(preset="cells")
    assert info.histogram is None
    assert info.brightness_auto is None


def test_a_flat_crop_has_a_histogram_but_no_auto():
    info = AP.add_box_info(
        existing_roi=np.zeros((4, 5, 6), np.int32), dark_sources=None,
        dark_rois=None, spacing=ISO, stride=1, dark_cache=None,
        source="paint", run_guide_roi=None, nuclei_roi=None,
        params={"preset": "nuclei"}, image_roi=np.full((4, 5, 6), 0.3))
    assert info.histogram is not None
    assert info.brightness_auto is None


def test_a_stop_before_the_histogram_is_cancelled():
    ev = threading.Event()
    ev.set()
    info, _ = _nuclei_info(cancel_event=ev)
    assert info.status == "cancelled"
    assert info.histogram is None
