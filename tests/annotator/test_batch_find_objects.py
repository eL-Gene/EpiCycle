"""Item C — one ``find_objects`` pass per batch run instead of one
``argwhere`` per object.

Pins two invariants: geometry byte-identity and ``find_objects`` parity.

Pure pytest for the executor; the two annotator cases bind the unbound
methods to a plain fake object (no ``QWidget.__init__`` needed).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_annotator import batch_refine as br
from packages.epicycle_annotator.batch_refine import (
    RefineStep,
    compute_label_roi,
    label_bboxes,
    resolve_target_ids,
    resolve_target_labels,
    run_batch_recipe,
)
from packages.epicycle_annotator.smooth import apply_smooth_to_volume


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _argwhere_bbox(volume, lid):
    coords = np.argwhere(volume == lid)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _random_labels(shape, n_ids, dtype, seed, id_offset=0):
    """Random box-shaped objects (overlaps allowed — later ones win)."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, dtype=dtype)
    for k in range(1, n_ids + 1):
        lo = [int(rng.integers(0, max(1, s - 1))) for s in shape]
        hi = [min(s, a + int(rng.integers(1, 6))) for a, s in zip(lo, shape)]
        vol[tuple(slice(a, b) for a, b in zip(lo, hi))] = k + id_offset
    return vol


def _old_run(volume, ids, recipe):
    """``run_batch_recipe``'s compute + apply at ``d1cdc6bb`` (per-id argwhere)."""
    working = volume.copy()
    margin = br._recipe_margin(recipe, volume.shape)
    results = []
    for lid in sorted(int(i) for i in ids):
        coords = np.argwhere(volume == lid)
        if coords.size == 0:
            results.append((lid, None, None, 0))
            continue
        mins, maxs = coords.min(axis=0), coords.max(axis=0)
        roi = tuple(
            slice(max(0, int(mins[d]) - margin[d]),
                  min(volume.shape[d], int(maxs[d]) + 1 + margin[d]))
            for d in range(volume.ndim)
        )
        res, n = br.apply_recipe_to_label(volume, lid, roi, recipe)
        results.append((lid, roi, res, n))
    touched, n_total, u0, u1 = [], 0, None, None
    for lid, roi, res, n in results:
        if roi is None:
            continue
        applied = apply_smooth_to_volume(working, roi, res, lid)
        if n > 0 or applied > 0:
            touched.append(lid)
            n_total += n
            s, e = [x.start for x in roi], [x.stop for x in roi]
            if u0 is None:
                u0, u1 = s, e
            else:
                u0 = [min(a, b) for a, b in zip(u0, s)]
                u1 = [max(a, b) for a, b in zip(u1, e)]
    union = tuple(slice(a, b) for a, b in zip(u0, u1)) if u0 is not None else None
    return working, touched, union, len([r for r in results if r[1] is not None]), n_total


DTYPES = [np.uint8, np.uint16, np.int32, np.uint32, np.int64]


# --------------------------------------------------------------------------- #
# label_bboxes — parity with per-id argwhere (invariant 9)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(9, 11, 13), (1, 20, 20), (30, 4, 7)])
def test_label_bboxes_equals_argwhere(dtype, shape):
    vol = _random_labels(shape, 25, dtype, seed=hash((shape, np.dtype(dtype).str)) % 1000)
    got = label_bboxes(vol)
    present = [int(v) for v in np.unique(vol) if v > 0]
    assert sorted(got) == present
    for lid in present:
        assert got[lid] == _argwhere_bbox(vol, lid)


@pytest.mark.parametrize("dtype", DTYPES)
def test_label_bboxes_with_ids_drops_absent(dtype):
    vol = _random_labels((12, 12, 12), 10, dtype, seed=3)
    present = {int(v) for v in np.unique(vol) if v > 0}
    ask = sorted(present)[:4] + [200, 0]
    got = label_bboxes(vol, ask)
    assert set(got) == set(sorted(present)[:4])
    for lid, bb in got.items():
        assert bb == _argwhere_bbox(vol, lid)


@pytest.mark.parametrize("dtype", [np.uint32, np.int64])
def test_label_bboxes_fallback_for_a_huge_sparse_id(dtype):
    """max id far above 64 x the count → per-id argwhere, same answer."""
    vol = _random_labels((10, 10, 10), 5, dtype, seed=7)
    vol[0:2, 0:3, 5:9] = 5_000_000
    assert not br._use_find_objects(vol.dtype, 5_000_000, 6)
    got = label_bboxes(vol)
    for lid in [int(v) for v in np.unique(vol) if v > 0]:
        assert got[lid] == _argwhere_bbox(vol, lid)
    got_ids = label_bboxes(vol, [5_000_000, 1, 123])
    assert got_ids[5_000_000] == _argwhere_bbox(vol, 5_000_000)
    assert 123 not in got_ids


def _sparse_volume():
    """200 small objects plus one stray id 2**31 (audit A1-M1)."""
    vol = _random_labels((30, 30, 30), 200, np.int64, seed=21)
    vol[25:28, 1:4, 20:29] = 2**31
    return vol


def test_label_bboxes_sparse_fallback_never_argwheres(monkeypatch):
    """A stray huge id must not cost one full-volume argwhere per object:
    the fallback compacts the ids and takes ONE find_objects pass."""
    vol = _sparse_volume()
    present = [int(v) for v in np.unique(vol) if v > 0]
    assert not br._use_find_objects(vol.dtype, 2**31, len(present))
    expected = {lid: _argwhere_bbox(vol, lid) for lid in present}

    calls = []
    real_argwhere = np.argwhere
    monkeypatch.setattr(
        np, "argwhere", lambda *a, **k: calls.append(1) or real_argwhere(*a, **k)
    )
    got = label_bboxes(vol)
    got_ids = label_bboxes(vol, [2**31, present[0], 123_456])
    assert calls == [], "the sparse fallback must not argwhere per object"
    assert got == expected
    assert got_ids == {2**31: expected[2**31], present[0]: expected[present[0]]}


def test_label_bboxes_sparse_fallback_matches_the_direct_path():
    """Byte-identity: renumber the stray id into the dense range and the
    direct find_objects path gives the very same boxes."""
    vol = _sparse_volume()
    dense = vol.copy()
    dense[dense == 2**31] = 201
    assert br._use_find_objects(dense.dtype, int(dense.max()), 1)
    sparse_boxes = label_bboxes(vol)
    dense_boxes = label_bboxes(dense)
    assert sparse_boxes.pop(2**31) == dense_boxes.pop(201)
    assert sparse_boxes == dense_boxes


def test_find_objects_bound():
    i32 = np.dtype(np.int32)
    assert br._use_find_objects(i32, 1_000_000, 1)
    assert not br._use_find_objects(i32, 1_000_001, 1)
    assert br._use_find_objects(i32, 64 * 20_000, 20_000)
    assert not br._use_find_objects(i32, 64 * 20_000 + 1, 20_000)
    assert not br._use_find_objects(np.dtype(np.float32), 10, 10)


def test_label_bboxes_float_volume_uses_argwhere():
    vol = np.zeros((6, 6, 6), dtype=np.float32)
    vol[1:3, 2:4, 0:5] = 2.0
    got = label_bboxes(vol)
    assert got == {2: (slice(1, 3), slice(2, 4), slice(0, 5))}


def test_label_bboxes_empty_and_background_only():
    assert label_bboxes(np.zeros((4, 4, 4), dtype=np.int32)) == {}
    assert label_bboxes(np.zeros((4, 4, 4), dtype=np.int32), [1, 2]) == {}
    assert label_bboxes(np.zeros((0, 4, 4), dtype=np.int32)) == {}


def test_present_ids_equal_np_unique():
    vol = _random_labels((15, 15, 15), 40, np.int32, seed=11)
    assert frozenset(label_bboxes(vol)) == frozenset(
        int(v) for v in np.unique(vol) if v > 0
    )


# --------------------------------------------------------------------------- #
# compute_label_roi(bbox=) — byte-identical padded ROI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("margin", [0, 2, 15, (3, 7, 1), (40, 0, 9)])
def test_compute_label_roi_bbox_equals_no_bbox(dtype, margin):
    vol = _random_labels((14, 18, 11), 20, dtype, seed=5)
    boxes = label_bboxes(vol)
    for lid, bb in boxes.items():
        assert compute_label_roi(vol, lid, margin, bbox=bb) == compute_label_roi(
            vol, lid, margin
        )


def test_compute_label_roi_bbox_clamps_at_the_walls():
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[0:2, 8:10, 4:6] = 1
    bb = label_bboxes(vol)[1]
    assert compute_label_roi(vol, 1, 5, bbox=bb) == (
        slice(0, 7), slice(3, 10), slice(0, 10),
    )


def test_compute_label_roi_keeps_its_positional_signature():
    """reshape/engine.py calls compute_label_roi(labels, target_id, int(margin))."""
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[4:6, 4:6, 4:6] = 3
    assert compute_label_roi(vol, 3, 2) == (slice(2, 8),) * 3
    assert compute_label_roi(vol, 9, 2) is None


def test_compute_label_roi_bbox_wrong_length_raises():
    vol = np.zeros((5, 5, 5), dtype=np.int32)
    vol[1, 1, 1] = 1
    with pytest.raises(ValueError):
        compute_label_roi(vol, 1, 1, bbox=(slice(1, 2), slice(1, 2)))


# --------------------------------------------------------------------------- #
# run_batch_recipe — geometry golden (invariant 1) + absent_ids
# --------------------------------------------------------------------------- #
RECIPES = [
    [RefineStep("sdf", {"sigma_z": 1.0, "sigma_xy": 1.0})],
    [RefineStep("dilate_erode", {"radius": 2, "axis": "XY"}), RefineStep("fill_holes", {})],
    [RefineStep("dilate_erode", {"radius": 1}), RefineStep("sdf", {"sigma_z": 0.8, "sigma_xy": 1.2})],
]


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("n_workers", [1, 4])
def test_geometry_run_is_byte_identical(dtype, recipe, n_workers):
    vol = np.zeros((24, 24, 24), dtype=dtype)
    vol[3:9, 3:10, 3:8] = 1
    vol[9:15, 3:10, 3:8] = 2          # touches 1
    vol[15:21, 12:20, 12:21] = 5
    vol[17:19, 15:17, 15:18] = 0      # a hole in 5
    vol[1:6, 16:23, 1:6] = 9
    ids = [1, 2, 5, 9]
    res = run_batch_recipe(vol, ids, recipe, n_workers=n_workers)
    working, touched, union, n_proc, n_total = _old_run(vol, ids, recipe)
    assert res.emptied_ids == [], "fixture: invariant 1 is for volumes where nothing empties"
    assert res.working_volume.dtype == vol.dtype
    assert np.array_equal(res.working_volume, working)
    assert res.touched_ids == touched
    assert res.union_roi == union
    assert res.n_processed == n_proc
    assert res.n_changed_total == n_total
    assert res.absent_ids == []


def test_absent_targets_are_reported_and_never_written():
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    vol[6:14, 6:14, 6:14] = 1
    seen = []
    res = run_batch_recipe(
        vol, [1, 999, 42], [RefineStep("sdf", {})], n_workers=1,
        progress_cb=lambda d, t: seen.append((d, t)),
    )
    assert res.absent_ids == [42, 999]
    assert 999 not in res.touched_ids and 42 not in res.touched_ids
    assert not np.any(res.working_volume == 999)
    assert res.n_processed == 1
    # Progress still counts every target, as before.
    assert seen[-1] == (3, 3)


def test_run_uses_one_bbox_pass_not_argwhere_per_object(monkeypatch):
    """The worker computes the bboxes once and hands each to compute_label_roi."""
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    vol[2:6, 2:6, 2:6] = 1
    vol[10:15, 10:15, 10:15] = 2
    calls = {"bbox": 0, "none": 0}
    real = br.compute_label_roi

    def spy(volume, label_id, margin, *, bbox=None):
        calls["none" if bbox is None else "bbox"] += 1
        return real(volume, label_id, margin, bbox=bbox)

    monkeypatch.setattr(br, "compute_label_roi", spy)
    run_batch_recipe(vol, [1, 2], [RefineStep("fill_holes", {})], n_workers=1)
    assert calls == {"bbox": 2, "none": 0}


def test_cancelled_before_start_does_no_bbox_pass(monkeypatch):
    import threading

    vol = np.zeros((8, 8, 8), dtype=np.int32)
    vol[2:4, 2:4, 2:4] = 1
    ev = threading.Event()
    ev.set()
    monkeypatch.setattr(br, "label_bboxes", lambda *a, **k: pytest.fail("bbox pass ran"))
    res = run_batch_recipe(vol, [1], [RefineStep("sdf", {})], n_workers=1, cancel_event=ev)
    assert res.cancelled and res.absent_ids == []


# --------------------------------------------------------------------------- #
# resolve_target_ids — the pure resolve over a cached present set
# --------------------------------------------------------------------------- #
def test_resolve_target_ids_matches_resolve_target_labels():
    vol = _random_labels((12, 12, 12), 15, np.int32, seed=2)
    present = frozenset(label_bboxes(vol))
    assert resolve_target_ids(present, "all") == resolve_target_labels(vol, None, "all")
    sel = [3, 5, 999, 0, 5]
    assert resolve_target_ids(present, "selected", selected_ids=sel) == resolve_target_labels(
        vol, None, "selected", selected_ids=sel
    )
    assert resolve_target_ids(present, "selected") == []
    with pytest.raises(ValueError):
        resolve_target_ids(present, "tier")


# --------------------------------------------------------------------------- #
# The annotator: the cache taken at dialog open is what counts read
# --------------------------------------------------------------------------- #
def _fake_annotator(vol):
    return types.SimpleNamespace(
        _active_labels_name="lbl",
        sync_manager=types.SimpleNamespace(shared_labels={"lbl": vol}),
        _navigator_widget=types.SimpleNamespace(
            selected_label_ids=lambda: {2, 3, 77}, _items=[]
        ),
    )


def test_resolve_batch_targets_reads_the_cache():
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0], vol[1, 1, 1], vol[2, 2, 2] = 1, 2, 3
    ann = _fake_annotator(vol)
    # A cache taken before label 3 was painted: "All" follows the cache.
    ann._batch_present_ids = frozenset({1, 2})
    resolve = Curation3DAnnotator._resolve_batch_targets
    assert resolve(ann, {"mode": "all"}) == [1, 2]
    assert resolve(ann, {"mode": "selected"}) == [2]
    # No cache (e.g. never opened): computed from the volume.
    ann._batch_present_ids = None
    assert resolve(ann, {"mode": "all"}) == [1, 2, 3]
    assert resolve(ann, {"mode": "selected"}) == [2, 3]


def test_dialog_open_caches_bboxes_and_present_ids(monkeypatch):
    from napari_epicycle.annotator import batch_refine_dialog as brd
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    captured = {}

    class _StubDialog:
        def __init__(self, **kw):
            captured.update(kw)

        def show(self):
            pass

        def raise_(self):
            pass

    monkeypatch.setattr(brd, "BatchRefineDialog", _StubDialog)

    vol = np.zeros((6, 6, 6), dtype=np.int32)
    vol[0:2, 0:2, 0:2] = 4
    vol[3:5, 3:6, 1:2] = 9
    ann = _fake_annotator(vol)
    ann._load_locked = False
    ann._sync_refine_state = lambda: None
    ann._refine_state = types.SimpleNamespace(can_start_batch_refine=lambda: True)
    ann._refresh_batch_recipe_history = lambda dlg: None
    ann._refresh_merge_button_state = lambda: None
    ann._refresh_label_action_buttons_state = lambda: None
    ann._batch_last_recipe = None
    ann._on_batch_preview_one = ann._on_batch_run = ann._on_batch_cancel = None
    ann._on_batch_accept = ann._on_batch_reject = ann._on_batch_close = None
    ann._save_batch_recipe = None
    ann._resolve_batch_targets = lambda spec: Curation3DAnnotator._resolve_batch_targets(ann, spec)

    calls = []
    real_unique = np.unique
    monkeypatch.setattr(np, "unique", lambda *a, **k: calls.append(1) or real_unique(*a, **k))

    Curation3DAnnotator._open_batch_refine_dialog(ann)

    assert ann._batch_bboxes == {
        4: (slice(0, 2), slice(0, 2), slice(0, 2)),
        9: (slice(3, 5), slice(3, 6), slice(1, 2)),
    }
    assert ann._batch_present_ids == frozenset({4, 9})
    assert captured["all_count"] == 2
    assert captured["count_provider"]({"mode": "all"}) == 2
    assert calls == [], "the open and the counts must not re-run np.unique"


def test_teardown_drops_the_cache():
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    ann = types.SimpleNamespace(
        _batch_refine_dialog=None, _batch_present_ids=frozenset({1}),
        _batch_bboxes={1: (slice(0, 1),) * 3},
    )
    Curation3DAnnotator._teardown_batch_dialog(ann)
    assert ann._batch_present_ids is None and ann._batch_bboxes is None


def test_close_drops_the_cache():
    """A non-clean close (X button) drops the cache too (audit A2-L5)."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    ann = types.SimpleNamespace(
        _batch_refine_dialog=None, _batch_present_ids=frozenset({1}),
        _batch_bboxes={1: (slice(0, 1),) * 3}, _batch_staged=False,
        _batch_worker=None, _fire_batch_refine_lock=lambda on: None,
        _restore_ready_to_draw_state=lambda: None,
        _sync_refine_state=lambda: None,
    )
    Curation3DAnnotator._on_batch_close(ann)
    assert ann._batch_present_ids is None and ann._batch_bboxes is None


@pytest.mark.parametrize("mode", ["all", "selected"])
def test_run_includes_an_id_painted_after_open(mode):
    """The open-time cache serves the COUNT; Run re-resolves against the
    live volume, so an object painted while the dialog was open is not
    dropped (audit A2-M1)."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0], vol[1, 1, 1] = 1, 2
    ann = _fake_annotator(vol)
    ann._batch_present_ids = frozenset(label_bboxes(vol))  # taken at open
    ann._batch_explicit_targets = None
    ann._navigator_widget = types.SimpleNamespace(
        selected_label_ids=lambda: {2, 3}, _items=[]
    )
    ann._resolve_batch_targets = (
        lambda spec, **kw: Curation3DAnnotator._resolve_batch_targets(ann, spec, **kw)
    )
    vol[2, 2, 2] = 3  # painted while the dialog is open
    ran = []
    ann._run_batch_refine = ran.append
    ann._batch_refine_dialog = types.SimpleNamespace(
        target_spec=lambda: {"mode": mode}
    )
    # The count still reads the open-time cache ...
    assert 3 not in ann._resolve_batch_targets({"mode": mode})
    # ... but Run sees the object.
    Curation3DAnnotator._on_batch_run(ann)
    assert ran == [[1, 2, 3]] if mode == "all" else ran == [[2, 3]]


def test_roi_perf_print_is_env_gated(monkeypatch, capsys):
    """``[BATCH-PERF] roi:`` fires on every Add Preview, so it is silent
    unless ``EPICYCLE_SYNC_DEBUG=1`` (audit A2-L2)."""
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[1, 1, 1] = 1
    monkeypatch.delenv("EPICYCLE_SYNC_DEBUG", raising=False)
    label_bboxes(vol)
    assert "[BATCH-PERF] roi:" not in capsys.readouterr().out
    monkeypatch.setenv("EPICYCLE_SYNC_DEBUG", "1")
    label_bboxes(vol)
    assert "[BATCH-PERF] roi: find_objects n=1" in capsys.readouterr().out
