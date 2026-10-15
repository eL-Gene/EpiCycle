"""``add_preview.run_add_preview`` -- the Add worker function.

Both presets, id minting, the dtype guard, the small / edge flags, the
below-min-size and smooth-emptied counts, the three zero-object answers,
``ridge=False`` reaching the engine, and a Stop answered as ``cancelled``.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import region as R

from .test_add_nuclei_golden import (
    ANISO,
    RECORDED,
    box_of,
    digest,
    nuclei_fixture,
)
from .test_reshape_region import _tissue

ISO = (1.0, 1.0, 1.0)


def _box(shape):
    return tuple(slice(0, int(n)) for n in shape)


def _nuclei_params(**kw):
    p = dict(clahe=False, clahe_clip=0.01, threshold="otsu", min_size=0,
             separate=True, min_distance=3, clear_border=True, edge_margin=3,
             auto_smooth=True, smooth_sigma=1.5)
    p.update(kw)
    return p


def _cells_params(source="labels", **kw):
    p = dict(source=source, wall_kind="membrane", wall_chain="none",
             dark_outside=True, dark_otsu_factor=0.5, dark_clearance_um=2.0,
             core_erode_um=1.0, click_radius_um=2.0, min_size=0,
             clear_border=True, edge_margin=3, auto_smooth=False,
             smooth_sigma=1.5)
    p.update(kw)
    return p


def _cells(source="labels", *, next_label=500, id_max=None, params=None,
           labels=None, cancel_event=None, progress=None, clicks=()):
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells) if labels is None else labels
    chans = {"membrane": mem, "nuclei": nuc}
    roi = _box(cells.shape)
    return AP.run_add_preview(
        preset="cells", roi=roi, existing_roi=labels, next_label=next_label,
        spacing=ISO, params=params or _cells_params(source),
        nuclei_roi=nuc, wall_roi=mem, dark_rois=chans,
        run_guide_roi=cells.copy(), click_points=clicks,
        dark_threshold=R.patch_dark_threshold(chans, spacing=ISO, stride=1),
        id_max=id_max, cancel_event=cancel_event, progress=progress), cells


# --------------------------------------------------------------------------- #
# Nuclei
# --------------------------------------------------------------------------- #
def _nuclei(seed=0, spacing=ISO, next_label=None, **kw):
    img, labels = nuclei_fixture(seed)
    roi = box_of(labels.shape)
    nl = int(labels.max()) + 1 if next_label is None else next_label
    return AP.run_add_preview(
        preset="nuclei", roi=roi, existing_roi=labels[roi].copy(),
        next_label=nl, spacing=spacing, params=_nuclei_params(**kw),
        image_roi=img[roi].astype(np.float64)), labels


@pytest.mark.parametrize("name,seed,spacing",
                         [("iso_seed0", 0, ISO), ("aniso_seed1", 1, ANISO)])
def test_the_nuclei_preset_reproduces_the_golden(name, seed, spacing):
    out, _labels = _nuclei(seed, spacing)
    assert out.ok
    d = digest(out.result)
    rec = RECORDED[name]
    assert d["sha256"] == rec["sha256"]
    assert sorted(out.edge_ids) == rec["edge"]
    assert out.ids == rec["ids"]


def test_the_nuclei_preset_shifts_ids_only_by_the_floor():
    """Decision 12: a deleted id above the volume max shifts every id by
    exactly ``floor - max(volume)``; the voxels do not move."""
    base, labels = _nuclei(0, ISO)
    shift = 40
    out, _ = _nuclei(0, ISO, next_label=int(labels.max()) + 1 + shift)
    assert out.ok
    assert np.array_equal(out.result > 0, base.result > 0)
    nz = base.result > 0
    assert np.all(out.result[nz] == base.result[nz] + shift)
    assert sorted(out.edge_ids) == sorted(i + shift for i in base.edge_ids)


def test_nuclei_found_nothing_is_its_own_sentence():
    img, labels = nuclei_fixture(0)
    roi = box_of(labels.shape)
    out = AP.run_add_preview(
        preset="nuclei", roi=roi, existing_roi=labels[roi].copy(),
        next_label=100, spacing=ISO, params=_nuclei_params(threshold=1e9),
        image_roi=img[roi].astype(np.float64))
    assert out.status == "refused"
    assert out.refusal == AP.ADD_S1_NOTHING_NUCLEI
    assert out.result is None or not np.any(out.result)


def test_nuclei_dtype_guard():
    img, labels = nuclei_fixture(0)
    roi = box_of(labels.shape)
    out = AP.run_add_preview(
        preset="nuclei", roi=roi, existing_roi=labels[roi].copy(),
        next_label=65530, spacing=ISO, params=_nuclei_params(),
        image_roi=img[roi].astype(np.float64), id_max=65535)
    assert out.status == "refused"
    assert out.refusal == AP.ADD_S1_IDS_FULL


def test_nuclei_small_ids_follow_the_widget_rule():
    out, _ = _nuclei(0, ISO)
    assert out.small_ids == AP.small_piece_ids(out.sizes)


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def test_cells_mints_final_ids_from_next_label():
    out, cells = _cells("labels", next_label=500)
    assert out.ok, out.refusal
    assert out.ids == list(range(500, 500 + len(out.ids)))
    assert set(np.unique(out.result)) - {0} == set(out.ids)
    assert out.result.dtype == np.int32


def test_cells_writes_only_on_empty_space():
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    labels[cells == 101] = 7
    out, _ = _cells("labels", labels=labels)
    assert out.ok
    assert not np.any(out.result[labels != 0])


def test_cells_dtype_guard():
    out, _ = _cells("labels", next_label=65530, id_max=65535)
    assert out.status == "refused"
    assert out.refusal == AP.ADD_S1_IDS_FULL


def test_cells_faint_pairs_are_remapped_to_final_ids():
    out, _ = _cells("labels", next_label=500)
    for a, b, _m in out.faint_pairs:
        assert a in out.ids and b in out.ids


def test_cells_below_min_size_is_counted_and_dropped():
    base, _ = _cells("labels")
    smallest = min(base.sizes.values())
    out, _ = _cells("labels", params=_cells_params("labels",
                                                    min_size=smallest + 1))
    assert out.ok
    assert out.n_below_min_size >= 1
    assert all(v >= smallest + 1 for v in out.sizes.values())


def test_every_object_filtered_is_all_filtered_not_no_markers():
    out, _ = _cells("labels", params=_cells_params("labels", min_size=10 ** 7))
    assert out.status == "refused"
    assert out.refusal == AP.ADD_S1_ALL_FILTERED
    assert out.n_below_min_size >= 1


def test_no_marker_at_all_is_the_engine_sentence():
    mem, nuc, cells = _tissue()
    labels = cells.copy()                   # every run object over kept space
    out, _ = _cells("labels", labels=labels)
    assert out.status == "refused"
    assert out.refusal == AP.ADD_ENGINE_NO_MARKERS


def test_the_budget_refusal_names_the_budget():
    out, _ = _cells("labels", params=_cells_params("labels", box_budget=100))
    assert out.status == "refused"
    assert out.refusal == R.too_big_sentence(100)


def test_ridge_false_reaches_the_engine(monkeypatch):
    seen = {}
    real = R.reshape_empty_region

    def spy(labels, images, spec, **kw):
        seen["ridge"] = spec.params["ridge"]
        seen["outside"] = spec.params["outside"]
        return real(labels, images, spec, **kw)

    monkeypatch.setattr(R, "reshape_empty_region", spy)
    out, _ = _cells("labels")
    assert out.ok
    assert seen == {"ridge": False, "outside": "faces+dark"}


def test_the_dark_tick_off_passes_faces(monkeypatch):
    seen = {}
    real = R.reshape_empty_region

    def spy(labels, images, spec, **kw):
        seen["outside"] = spec.params["outside"]
        seen["dark"] = kw.get("dark_channels")
        return real(labels, images, spec, **kw)

    monkeypatch.setattr(R, "reshape_empty_region", spy)
    out, _ = _cells("labels", params=_cells_params("labels", dark_outside=False))
    assert out.ok
    assert seen["outside"] == "faces" and seen["dark"] is None
    assert out.n_dropped_dark == 0
    assert out.unfilled_tissue_fraction is None


def test_nuclei_masks_source_runs():
    out, _ = _cells("nuclei_masks")
    assert out.ok, out.refusal
    assert out.source == "nuclei_masks"
    assert len(out.ids) >= 4


def test_clicks_are_made_box_local():
    mem, nuc, cells = _tissue()
    shape = cells.shape
    labels = np.zeros((shape[0] + 4,) + shape[1:], cells.dtype)
    roi = (slice(2, 2 + shape[0]),) + _box(shape[1:])
    chans = {"membrane": mem, "nuclei": nuc}
    # (6, 5, 5) box-local == (8, 5, 5) in the volume
    out = AP.run_add_preview(
        preset="cells", roi=roi, existing_roi=labels[roi].copy(),
        next_label=10, spacing=ISO,
        params=_cells_params("paint"), wall_roi=mem, dark_rois=chans,
        click_points=[(8, 5, 5)],
        dark_threshold=R.patch_dark_threshold(chans, spacing=ISO, stride=1))
    assert out.ok, out.refusal
    assert out.result[6, 5, 5] == 10


def test_smooth_emptied_objects_are_counted(monkeypatch):
    from packages.epicycle_annotator import add_segment

    def drop_first(result, sigma):
        ids = [int(v) for v in np.unique(result) if v]
        return np.where(result == ids[0], 0, result)

    monkeypatch.setattr(add_segment, "smooth_pieces", drop_first)
    out, _ = _cells("labels", params=_cells_params("labels", auto_smooth=True))
    assert out.ok
    assert out.n_emptied_by_smooth == 1


def test_a_stop_is_answered_as_cancelled_not_raised():
    ev = threading.Event()
    ev.set()
    out, _ = _cells("labels", cancel_event=ev)
    assert out.status == "cancelled"
    assert out.result is None


def test_a_stop_mid_engine_is_cancelled(monkeypatch):
    ev = threading.Event()

    def progress(stage, i, n):
        if stage == "add_cut":
            ev.set()

    out, _ = _cells("labels", cancel_event=ev, progress=progress)
    assert out.status == "cancelled"


def test_progress_names_arrive_at_stage_start():
    names = []
    out, _ = _cells("nuclei_masks", progress=lambda s, i, n: names.append(s))
    assert out.ok
    # the nuclei are this source's markers (live-shot fix V6)
    assert names[:1] == ["add_markers"]
    assert "add_find" not in names
    for s in R.REGION_STAGES:
        assert s in names


def test_the_log_line_carries_faint_and_unfilled():
    out, _ = _cells("labels")
    line = out.log_line()
    assert line.startswith("[ADD] preview ok")
    assert "faint=" in line and "unfilled=" in line


# --------------------------------------------------------------------------- #
# A5 -- click markers keep the id they reserved
# --------------------------------------------------------------------------- #
#: one click in the middle of three of _tissue's cells (box-local == volume)
_A, _B, _C = (6, 6, 5), (6, 6, 15), (6, 6, 25)


def _cells_reserved(points, reserved, id_max=None):
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    chans = {"membrane": mem, "nuclei": nuc}
    return AP.run_add_preview(
        preset="cells", roi=_box(cells.shape), existing_roi=labels,
        next_label=500, spacing=ISO, params=_cells_params("paint"),
        wall_roi=mem, dark_rois=chans, click_points=list(points),
        reserved_ids=list(reserved),
        dark_threshold=R.patch_dark_threshold(chans, spacing=ISO, stride=1),
        id_max=id_max)


def test_each_click_becomes_the_id_it_reserved():
    out = _cells_reserved([_A, _B, _C], [500, 503, 507])
    assert out.ok, out.refusal
    assert sorted(out.ids) == [500, 503, 507]
    assert out.result[_A] == 500
    assert out.result[_B] == 503
    assert out.result[_C] == 507
    assert out.absorbed_ids == []


def test_an_unticked_marker_leaves_a_gap_and_the_rest_keep_their_ids():
    """Invariant 20: the unticked marker's point is not handed on, its id is
    not written, and #3 keeps 507 (its colour)."""
    out = _cells_reserved([_A, _C], [500, 507])
    assert out.ok, out.refusal
    assert sorted(out.ids) == [500, 507]
    assert out.result[_C] == 507
    assert 503 not in set(np.unique(out.result).tolist())


def test_an_absorbed_marker_does_not_shift_the_later_ids():
    """Two clicks on the same voxel: the engine keeps the first ball, the
    second vanishes -- the third must still be 502, and 501 is reported."""
    out = _cells_reserved([_A, _A, _C], [500, 501, 502])
    assert out.ok, out.refusal
    assert sorted(out.ids) == [500, 502]
    assert out.result[_C] == 502
    assert out.absorbed_ids == [501]
    assert "501" in out.log_line()


def test_the_dtype_guard_reads_the_largest_reserved_id():
    out = _cells_reserved([_A, _C], [500, 70000], id_max=65535)
    assert not out.ok
    assert out.refusal == AP.ADD_S1_IDS_FULL
    ok = _cells_reserved([_A, _C], [500, 65535], id_max=65535)
    assert ok.ok, ok.refusal
    assert ok.result[_C] == 65535


def test_reserved_ids_must_match_the_points():
    with pytest.raises(ValueError):
        _cells_reserved([_A, _C], [500])


def test_without_reserved_ids_clicks_still_mint_from_next_label():
    out, _ = _cells("paint", clicks=[_A, _C], next_label=500)
    assert out.ok, out.refusal
    assert sorted(out.ids) == [500, 501]
