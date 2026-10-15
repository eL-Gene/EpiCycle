"""S1.5 / Phase B — chunk-parallel contact-graph family (the byte-identical gate).

The three VOLUME-reading graph passes (neighbour features / contact geometry /
axis-neighbour persistence) are embarrassingly parallel per cell: each cell reads
only its own halo crop off the label volume + its own df row. So
``run_adjacency_graph_passes`` splits ``df`` into contiguous row-chunks, runs the
EXISTING function on each chunk in a worker (which memmaps the labels once), and
merges back. The parallel output MUST be byte-identical to the serial output —
same feature values + same ``df.attrs["_adjacency_list"]`` (as ordered lists).

Gates proven here:
  1. ``run_adjacency_graph_passes(n_workers=1)`` == ``(n_workers=4)`` — every column
     via ``assert_frame_equal`` AND ``_adjacency_list`` equal as ordered lists + sets.
  2. ``extract_label_features(adjacency=True, n_workers=1)`` == ``(n_workers=4)``.
  3. In-memory array + ``n_workers=4`` falls back to serial (no crash, correct).
  4. ``FeatureStage(adjacency=True, n_workers=4)`` graph columns == ``n_workers=1``.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import find_objects

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.reconcile.features import extract_label_features


def _multi_touch_labels():
    """A 5x6 grid of touching cubes (1-voxel gaps -> adjacency at radius 1) in one
    z-slab + one isolated cube. ~31 cells so a 4-way split has multiple cells per
    chunk and an interior cell has 4 neighbours -> the contact-geometry /
    axis-persistence passes have non-trivial work in every chunk."""
    vol = np.zeros((10, 60, 70), np.int32)
    lab = 1
    for yi in range(5):
        for xi in range(6):
            y0 = 2 + yi * 6
            x0 = 2 + xi * 6
            vol[2:8, y0:y0 + 5, x0:x0 + 5] = lab   # 5-wide on a 6 stride -> 1-vox gap
            lab += 1
    vol[2:8, 50:56, 60:66] = lab                    # isolated (gap >> max_radius)
    return vol


def _assembled_base(path):
    """Labels-only per-cell table (no graph yet), label int-typed — the input the
    graph family runs on. Matches what extract_label_features hands to
    run_adjacency_graph_passes."""
    base = extract_label_features(str(path), n_workers=1)
    base["label"] = base["label"].astype(np.int64)
    return base.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. byte-identical gate — run_adjacency_graph_passes n_workers=1 vs 4         #
# --------------------------------------------------------------------------- #
def test_run_passes_serial_vs_parallel_byte_identical(tmp_path):
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    objs = find_objects(vol)

    base = _assembled_base(p)

    # SERIAL (n_workers=1 -> the unchanged byte-identical default path).
    df_serial = base.copy()
    mm1 = np.load(str(p), mmap_mode="r")
    qf.run_adjacency_graph_passes(
        df_serial, mm1, slices=objs, spacing=None, max_radius=5,
        n_workers=1, labels_path=str(p))

    # PARALLEL (n_workers=4 -> chunk-parallel path).
    df_par = base.copy()
    mm2 = np.load(str(p), mmap_mode="r")
    qf.run_adjacency_graph_passes(
        df_par, mm2, slices=objs, spacing=None, max_radius=5,
        n_workers=4, labels_path=str(p))

    pd.testing.assert_frame_equal(df_serial, df_par)
    # _adjacency_list equal as ORDERED lists (the strongest form) AND as sets.
    a_ser = df_serial.attrs["_adjacency_list"]
    a_par = df_par.attrs["_adjacency_list"]
    assert a_ser, "the grid really touches -> non-empty edge list"
    assert a_ser == a_par, "edge list ORDER must match the serial append order"
    _to_set = lambda L: {(int(r["label_a"]), int(r["label_b"]), int(r["contact_voxels"]))
                         for r in L}
    assert _to_set(a_ser) == _to_set(a_par)


def test_run_passes_byte_identical_scrambled_row_order(tmp_path):
    """The byte-identical gate with the df rows SCRAMBLED (not sorted by label) —
    this is what FeatureStage feeds (imap_unordered row order). A neighbour's row
    then lands in a DIFFERENT chunk than the cell, so the neighbour-SIZE-ratio
    columns (which need every neighbour's ``area``) only match serial if the FULL
    population area map is broadcast to each chunk. Catches the cross-chunk
    area-lookup class of bug at the function level (independent of FeatureStage)."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    objs = find_objects(vol)

    base = _assembled_base(p)
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(base))
    base = base.iloc[perm].reset_index(drop=True)   # scramble row order

    df_serial = base.copy()
    qf.run_adjacency_graph_passes(
        df_serial, np.load(str(p), mmap_mode="r"), slices=objs, spacing=None,
        max_radius=5, n_workers=1, labels_path=str(p))

    df_par = base.copy()
    qf.run_adjacency_graph_passes(
        df_par, np.load(str(p), mmap_mode="r"), slices=objs, spacing=None,
        max_radius=5, n_workers=4, labels_path=str(p))

    pd.testing.assert_frame_equal(df_serial, df_par)
    assert df_serial.attrs["_adjacency_list"] == df_par.attrs["_adjacency_list"]
    # the size-ratio columns are the ones that exposed the cross-chunk bug.
    for c in ("mean_neighbor_size_ratio", "neighbor_size_cv",
              "contact_weighted_neighbor_size_ratio"):
        np.testing.assert_array_equal(
            df_serial[c].to_numpy(), df_par[c].to_numpy(),
            err_msg=f"{c} differs under scrambled row order")


def test_run_passes_serial_vs_parallel_with_spacing(tmp_path):
    """Same gate with anisotropic spacing (exercises the physical-frame members of
    contact geometry + axis persistence)."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    objs = find_objects(vol)
    sp = (2.0, 0.5, 0.5)

    base = _assembled_base(p)

    df_serial = base.copy()
    qf.run_adjacency_graph_passes(
        df_serial, np.load(str(p), mmap_mode="r"), slices=objs, spacing=sp,
        max_radius=5, n_workers=1, labels_path=str(p))

    df_par = base.copy()
    qf.run_adjacency_graph_passes(
        df_par, np.load(str(p), mmap_mode="r"), slices=objs, spacing=sp,
        max_radius=5, n_workers=4, labels_path=str(p))

    pd.testing.assert_frame_equal(df_serial, df_par)
    assert df_serial.attrs["_adjacency_list"] == df_par.attrs["_adjacency_list"]


# --------------------------------------------------------------------------- #
# 2. extract_label_features parity — n_workers=1 vs 4                          #
# --------------------------------------------------------------------------- #
def test_extract_label_features_serial_vs_parallel(tmp_path):
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)

    df1 = extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)
    df4 = extract_label_features(str(p), n_workers=4, adjacency=True, max_radius=5)

    df1 = df1.sort_values("label").reset_index(drop=True)
    df4 = df4.sort_values("label").reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df4, check_dtype=False)

    a1 = df1.attrs.get("_adjacency_list")
    a4 = df4.attrs.get("_adjacency_list")
    assert a1, "the grid touches -> non-empty edge list"
    # The per-cell pool order can differ (imap_unordered), but both paths sort by
    # label before the graph pass, so the edge lists must be set-equal; and within
    # one path the order is deterministic. Compare as sets to be order-robust here.
    _to_set = lambda L: {(int(r["label_a"]), int(r["label_b"]), int(r["contact_voxels"]))
                         for r in L}
    assert _to_set(a1) == _to_set(a4)


def test_extract_label_features_serial_vs_parallel_spacing(tmp_path):
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    sp = (2.0, 0.5, 0.5)

    df1 = (extract_label_features(str(p), n_workers=1, adjacency=True,
                                  max_radius=5, spacing=sp)
           .sort_values("label").reset_index(drop=True))
    df4 = (extract_label_features(str(p), n_workers=4, adjacency=True,
                                  max_radius=5, spacing=sp)
           .sort_values("label").reset_index(drop=True))
    pd.testing.assert_frame_equal(df1, df4, check_dtype=False)


# --------------------------------------------------------------------------- #
# 3. in-memory array fallback — n_workers=4 with an array source -> serial    #
# --------------------------------------------------------------------------- #
def test_in_memory_array_falls_back_to_serial(tmp_path):
    """An in-memory array source has no durable file path for the workers' memmap,
    so the graph family must fall back to SERIAL — correct output, no crash."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)

    # Reference: the PATH path with n_workers=1 (byte-identical serial).
    ref = (extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)
           .sort_values("label").reset_index(drop=True))

    # In-memory array + n_workers=4 -> labels_path=None -> serial fallback.
    arr = (extract_label_features(vol, n_workers=4, adjacency=True, max_radius=5)
           .sort_values("label").reset_index(drop=True))

    pd.testing.assert_frame_equal(ref, arr, check_dtype=False)
    assert ref.attrs.get("_adjacency_list")


def test_run_passes_no_labels_path_falls_back(tmp_path):
    """run_adjacency_graph_passes with n_workers=4 but labels_path=None falls back
    to serial (the resident array IS available as graph_src) — same as n_workers=1."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    objs = find_objects(vol)
    base = _assembled_base(p)

    df_ref = base.copy()
    qf.run_adjacency_graph_passes(
        df_ref, vol, slices=objs, spacing=None, max_radius=5, n_workers=1)

    df_fb = base.copy()
    qf.run_adjacency_graph_passes(
        df_fb, vol, slices=objs, spacing=None, max_radius=5,
        n_workers=4, labels_path=None)   # no path -> serial fallback

    pd.testing.assert_frame_equal(df_ref, df_fb)
    assert df_ref.attrs["_adjacency_list"] == df_fb.attrs["_adjacency_list"]


# --------------------------------------------------------------------------- #
# 4. FeatureStage — adjacency graph columns n_workers=1 vs 4                   #
# --------------------------------------------------------------------------- #
_PROV_COLS = [
    "dense_id", "seam_status", "n_tiles", "min_match_iou", "voxel_count",
    "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
    "touches_border", "tiles", "constituents",
]


def _write_prov(path, vol, ids):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROV_COLS)
        w.writeheader()
        for lbl in ids:
            zs, ys, xs = np.nonzero(vol == lbl)
            w.writerow({
                "dense_id": lbl, "seam_status": "clean", "n_tiles": 1,
                "min_match_iou": "", "voxel_count": int((vol == lbl).sum()),
                "z_min": int(zs.min()), "z_max": int(zs.max()),
                "y_min": int(ys.min()), "y_max": int(ys.max()),
                "x_min": int(xs.min()), "x_max": int(xs.max()),
                "touches_border": 0, "tiles": "t0", "constituents": "",
            })


def _feature_stage_fixture(tmp: Path):
    stitch = tmp / "stage4_full_stitch"
    stitch.mkdir(parents=True, exist_ok=True)
    vol = _multi_touch_labels()
    ids = sorted(int(i) for i in np.unique(vol) if i != 0)
    np.save(str(stitch / "global_labels_clean.npy"), vol)
    _write_prov(stitch / "label_provenance_clean.csv", vol, ids)
    return stitch, vol


def test_feature_stage_adjacency_serial_vs_parallel(tmp_path):
    from packages.epicycle_scaledseg.core.feature_stage import FeatureStage, _FEATURE_FAMILIES

    stitch1, _ = _feature_stage_fixture(tmp_path / "w1")
    FeatureStage(input_dir=str(stitch1), n_workers=1, adjacency=True).run(
        [Path("x/tiles/t0")], n_gpus=0)
    cf1 = (pd.read_csv(stitch1 / "cell_features.csv")
           .sort_values("label").reset_index(drop=True))

    stitch4, _ = _feature_stage_fixture(tmp_path / "w4")
    FeatureStage(input_dir=str(stitch4), n_workers=4, adjacency=True).run(
        [Path("x/tiles/t0")], n_gpus=0)
    cf4 = (pd.read_csv(stitch4 / "cell_features.csv")
           .sort_values("label").reset_index(drop=True))

    graph_cols = [c for c in _FEATURE_FAMILIES["adjacency"]
                  if c in cf1.columns and c in cf4.columns]
    assert "neighbor_count" in graph_cols, "the family really landed"
    for c in graph_cols:
        np.testing.assert_allclose(
            cf1[c].to_numpy(dtype=float), cf4[c].to_numpy(dtype=float),
            equal_nan=True,
            err_msg=f"graph column {c} differs FeatureStage n_workers 1 vs 4")

    # edges.csv content must be identical (set-equal — order may differ across runs).
    e1 = pd.read_csv(stitch1 / "edges.csv")
    e4 = pd.read_csv(stitch4 / "edges.csv")
    s1 = {(int(r.label_a), int(r.label_b), int(r.contact_voxels)) for r in e1.itertuples()}
    s4 = {(int(r.label_a), int(r.label_b), int(r.contact_voxels)) for r in e4.itertuples()}
    assert s1 == s4 and s1


# --------------------------------------------------------------------------- #
# 5. H1 — failure isolation: a bad chunk must NOT abort the parallel pass      #
# --------------------------------------------------------------------------- #
# Module-level fake workers (picklable under the ``spawn`` start method — the
# function passed to ``pool.map`` is pickled by qualified name, so it MUST be
# importable in the child; a closure/lambda would fail to pickle). Each delegates
# to the REAL worker for every chunk except the one it is asked to fail.

# Captured at import, BEFORE any monkeypatch: under the ``fork`` start method the child
# inherits the patched ``qf._graph_chunk_worker`` (this wrapper), so calling it by
# attribute would recurse forever. Under ``spawn`` the child re-imports and gets the real one.
_REAL_GRAPH_CHUNK_WORKER = qf._graph_chunk_worker


def _worker_fail_chunk0(task):
    """Stand-in for the real worker that returns the CHUNK-ERROR sentinel for the
    chunk starting at row 0 (as the real worker's try/except would on a scipy crash)
    and delegates every other chunk to the real worker. Returning the sentinel here
    (rather than raising) exercises the DRIVER half of H1 — the spawn start method
    means a monkeypatch in the parent can't reach the children, so we inject at the
    dispatched-callable seam, which is the seam ``pool.map`` actually pickles. The
    WORKER half (turning a real exception INTO the sentinel) is covered by
    ``test_h1_worker_returns_sentinel_on_exception``."""
    chunk_pos = task[4]
    if len(chunk_pos) and int(chunk_pos[0]) == 0:
        return ("CHUNK-ERROR", chunk_pos, "ValueError('injected crash on chunk 0')")
    return _REAL_GRAPH_CHUNK_WORKER(task)


def test_h1_failed_chunk_does_not_abort_and_nans_its_rows(tmp_path, caplog):
    """H1 — if one chunk's worker raises, the pass COMPLETES: the failed chunk's
    rows get NaN in the pass's new columns, the OTHER chunks' columns are still
    byte-identical to serial, and a WARNING naming the chunk's row + cell range is
    logged. (One bad cell in ~1M must not lose a multi-hour job.)"""
    import logging

    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    objs = find_objects(vol)
    base = _assembled_base(p)

    # Serial reference (the byte-identical truth for the SURVIVING rows).
    df_serial = base.copy()
    qf.run_adjacency_graph_passes(
        df_serial, np.load(str(p), mmap_mode="r"), slices=objs, spacing=None,
        max_radius=5, n_workers=1, labels_path=str(p))

    # Parallel run with chunk-0's worker forced to raise. Patch the function the
    # pool dispatches; the real worker stays the delegate for the other chunks.
    df_par = base.copy()
    n = len(df_par)
    nw = 4
    # The first contiguous chunk == rows [chunk0_pos] — the ones that will be NaN.
    chunk0_pos = [p2 for p2 in np.array_split(np.arange(n), nw) if p2.size > 0][0]

    orig = qf._graph_chunk_worker
    qf._graph_chunk_worker = _worker_fail_chunk0
    try:
        with caplog.at_level(logging.WARNING, logger="packages.epicycle_scaledseg.core.qc_features"):
            qf.run_adjacency_graph_passes(
                df_par, np.load(str(p), mmap_mode="r"), slices=objs, spacing=None,
                max_radius=5, n_workers=nw, labels_path=str(p))
    finally:
        qf._graph_chunk_worker = orig

    # The run COMPLETED (no exception) and the warning fired for the failed chunk
    # of EVERY parallel pass that hit it (chunk 0 fails in all three).
    warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("chunk failed" in m for m in warns), warns
    assert any("compute_neighbor_features" in m for m in warns), warns

    # The columns written DIRECTLY by the three parallel passes (neighbour features
    # / contact geometry / axis persistence). The chunk-0 worker returned a sentinel
    # for each, so these MUST be NaN for chunk-0 rows. (Downstream SERIAL table
    # passes — graph_topology / local_comparative / motifs — derive from these and
    # may legitimately not be all-NaN, so we check only the parallel-pass columns.)
    parallel_cols = [c for c in (
        "neighbor_count", "contact_area_total",
        "axis_neighbor_persistence_mean",
    ) if c in df_par.columns]
    assert "neighbor_count" in parallel_cols, "the neighbour pass really ran"

    # Failed chunk's rows -> NaN in EVERY parallel-pass column (chunk 0 failed in
    # all three parallel passes).
    for c in parallel_cols:
        vals = df_par[c].to_numpy()[chunk0_pos]
        assert np.isnan(vals).all(), f"failed chunk rows should be NaN in {c}"

    # The SURVIVING rows are still COMPUTED (not NaN) — the failure did not abort
    # the pass for them. We use ``neighbor_count`` for the strict byte-identical
    # check: it is a per-cell crop-local count, independent of whether chunk 0 ran,
    # so survivors must match serial exactly. (The contact/axis columns read the
    # GLOBAL adjacency list, which is now missing chunk-0's edges, so a survivor
    # adjacent to a chunk-0 cell may legitimately differ there — that is the
    # expected downstream effect of a dropped chunk, not a contamination bug — so
    # we only assert those survivors are non-NaN, not equal.)
    survivors = np.setdiff1d(np.arange(n), np.asarray(chunk0_pos))
    np.testing.assert_array_equal(
        df_par["neighbor_count"].to_numpy()[survivors],
        df_serial["neighbor_count"].to_numpy()[survivors],
        err_msg="surviving rows' neighbor_count differs from serial")
    for c in parallel_cols:
        surv_vals = df_par[c].to_numpy()[survivors]
        assert not np.isnan(surv_vals).all(), \
            f"surviving rows should still be computed in {c} (not all NaN)"


def test_h1_worker_returns_sentinel_on_exception():
    """H1 (unit) — the worker itself converts ANY internal exception into a
    ``("CHUNK-ERROR", chunk_pos, repr)`` sentinel instead of propagating."""
    # Force a failure deterministically: an unknown fn_name -> getattr() raises.
    chunk_pos = np.array([0, 1, 2])
    bad_task = ("not_a_real_function", pd.DataFrame({"label": [1, 2, 3]}),
                None, 5, chunk_pos)
    res = qf._graph_chunk_worker(bad_task)
    assert isinstance(res, tuple) and res[0] == "CHUNK-ERROR"
    np.testing.assert_array_equal(res[1], chunk_pos)
    assert isinstance(res[2], str) and res[2]   # the repr of the exception


# --------------------------------------------------------------------------- #
# 6. H2 — the big shared objects are NOT pickled per task                      #
# --------------------------------------------------------------------------- #
def test_h2_task_payload_excludes_big_shared_objects(tmp_path, monkeypatch):
    """H2 — the per-chunk task tuple must carry ONLY the per-chunk data + cheap
    scalars; the volume-scale ``adjacency`` edge list + the full population
    ``area_lookup`` must travel via the worker INITARGS (one copy per process), NOT
    be embedded in every task. We intercept the tasks list handed to ``pool.map``
    and assert none of them contain a big dict/list."""
    import multiprocessing as mp

    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    objs = find_objects(vol)
    base = _assembled_base(p)

    captured = {"tasks": None, "initargs": None}

    real_pool_cls = mp.Pool

    def _spy_pool(*args, **kwargs):
        this_initargs = kwargs.get("initargs")
        pool = real_pool_cls(*args, **kwargs)
        real_map = pool.map

        def _map(fn, tasks, *a, **k):
            # Materialise + record the FIRST pass's tasks + initargs together (the
            # neighbour pass — the only one that builds an area_lookup), so the two
            # captured artefacts correspond to the same pass.
            tasks = list(tasks)
            if captured["tasks"] is None:
                captured["tasks"] = tasks
                captured["initargs"] = this_initargs
            return real_map(fn, tasks, *a, **k)

        pool.map = _map
        return pool

    monkeypatch.setattr(mp, "Pool", _spy_pool)

    df = base.copy()
    qf.run_adjacency_graph_passes(
        df, np.load(str(p), mmap_mode="r"), slices=objs, spacing=None,
        max_radius=5, n_workers=4, labels_path=str(p))

    tasks = captured["tasks"]
    assert tasks, "the parallel path ran and dispatched tasks"
    # Each task = (fn_name, df_chunk, spacing, max_radius, chunk_pos) — 5 elements,
    # no big dict/list among the trailing scalars.
    for t in tasks:
        assert len(t) == 5, f"task tuple grew a payload: len={len(t)}"
        fn_name, df_chunk, spacing, max_radius, chunk_pos = t
        assert isinstance(df_chunk, pd.DataFrame)
        # None of the carried members may be the big area_lookup / adjacency list.
        for member in (spacing, max_radius, chunk_pos):
            assert not isinstance(member, dict), "area_lookup leaked into the task"
        # The adjacency edge list is a list-of-dicts; a chunk_pos is a small array.
        assert not (isinstance(chunk_pos, list) and chunk_pos
                    and isinstance(chunk_pos[0], dict)), \
            "adjacency edge list leaked into the task"

    # The big objects DID travel via initargs (labels_path, slices, adjacency,
    # area_lookup). For the neighbour pass the area_lookup (a dict over the whole
    # population) is the 4th initarg and must be present + populated.
    initargs = captured["initargs"]
    assert initargs is not None and len(initargs) == 4
    area_lookup = initargs[3]
    assert isinstance(area_lookup, dict) and len(area_lookup) == len(base), \
        "the full population area_lookup must travel via initargs, once per worker"
