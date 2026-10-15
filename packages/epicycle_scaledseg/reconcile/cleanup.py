"""Descoped CLEANUP -> a CONSISTENT (cell, nucleus) consensus: every nucleus inside one cell.

This is the pragmatic "clean up the reconstruction" pass, NOT the hard merge-resolution (which
is parked as future work — likely a learned model). It TRUSTS the cell labels and makes the
nuclei conform to them. Pure, per-object primitives (no I/O, no Qt/GPU) so the driver can run
per-component, parallel OR sequential, on the HPC or a local laptop with any (cell, nucleus)
mask pair. Reuses the engine's SDF smoother.

The SHIPPED pipeline is `cleanup_volume`, which MAJORITY-ASSIGNS each nucleus to the single cell
it is mostly in — it does NOT fabricate a split (splitting a straddling nucleus is the parked
hard call). Consequence reported honestly: a nucleus that straddled two real cells is removed
from the loser cell, so that cell is counted `no_nucleus_strung_victim` (the cleanup's own
doing), separate from a genuinely anucleate cell.

Primitives:
  * recover_nucleus — for a cell with NO nucleus, try to recover a missed (dim/deep) nucleus from
    the nuclear stain by local contrast rescale + a peak-SNR threshold (the "crank the contrast
    in the deep slices" idea), with a boundary guard against a neighbour's nucleus bleeding in.
  * fill_smooth_cell — fill holes + (optional; OFF for a quantification mask) SDF-smooth one cell.
  * carve_nucleus_by_cells — an ALTERNATIVE explicit-split primitive (one piece per cell), NOT
    used by `cleanup_volume`; for callers who deliberately want to split a straddling nucleus.

Merged cells (1 cell, 2+ nuclei) are KEPT as-is and reported as the residual (the parked split).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _conn_full(ndim: int):
    from scipy.ndimage import generate_binary_structure
    return generate_binary_structure(ndim, ndim)


def _maybe_tqdm(iterable, total=None, desc=None, enable=False):
    """Wrap an iterable with a tqdm progress bar if available + enabled; else return it as-is."""
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def carve_nucleus_by_cells(nucleus_mask: np.ndarray, cell_labels: np.ndarray) -> dict:
    """Clip a nucleus to the cells it sits in; split it per cell (strung -> one piece per cell).

    Args:
        nucleus_mask: bool (Z,Y,X) one nucleus.
        cell_labels: int (Z,Y,X) the cell-label crop (same shape).

    Returns:
        ``{cell_id: bool mask}`` — the nucleus's piece inside each cell it overlaps. Voxels
        outside every cell are dropped (not returned). Empty dict if the nucleus is entirely
        outside all cells (an orphan).
    """
    N = np.asarray(nucleus_mask, dtype=bool)
    lab = np.asarray(cell_labels)
    out = {}
    for cid in np.unique(lab[N]):
        if cid == 0:
            continue
        out[int(cid)] = N & (lab == cid)
    return out


def recover_nucleus(
    nuclei_raw: np.ndarray,
    cell_mask: np.ndarray,
    k: float = 3.0,
    min_blob: int = 10,
    max_frac: float = 0.5,
    return_reason: bool = False,
):
    """Recover a missed nucleus inside a no-nucleus cell from the nuclear stain.

    A nucleus is a COMPACT bright PEAK above the cytoplasm — not just "the upper half" an Otsu
    split would hand back on flat noise or a gradient. So we gate on a real peak: contrast =
    (p99 − p50) relief over a blob-immune cytoplasm sigma (p50 − p16) must clear ``k``; then we
    take the largest connected blob above the half-relief threshold (a contrast-relative
    threshold — the "rescale then threshold" idea, done robustly), and require it to be compact
    (≥ ``min_blob`` and ≤ ``max_frac`` of the cell).

    Args:
        nuclei_raw: float (Z,Y,X) raw nuclear-stain crop.
        cell_mask: bool (Z,Y,X) the cell.
        k: minimum peak SNR (relief / cytoplasm-sigma) to accept a nucleus.
        min_blob: fewest voxels for a real nucleus.
        max_frac: reject a blob larger than this fraction of the cell (a gradient/halo, not a
            compact nucleus).

    Returns:
        bool (Z,Y,X) recovered-nucleus mask, or None. With ``return_reason=True`` returns
        ``(mask_or_None, reason)`` where ``reason`` ∈ {ok / cell_too_small / low_snr / no_bright /
        no_cc / too_small / too_big / on_boundary} — use it to see WHY a visibly-bright cell was
        rejected (e.g. ``low_snr`` = bright cytoplasm too, ``too_big`` = nucleus > max_frac of a
        small cell, ``on_boundary`` = blob hugs the wall, likely the neighbour's nucleus).
    """
    from scipy.ndimage import label as cc_label

    def _ret(mask, reason):
        return (mask, reason) if return_reason else mask

    m = np.asarray(cell_mask, dtype=bool)
    n = int(m.sum())
    cad = np.asarray(nuclei_raw, dtype=np.float64)
    if cad.shape != m.shape:
        raise ValueError("nuclei_raw and cell_mask must share a shape")
    if n < min_blob:
        return _ret(None, "cell_too_small")
    p16, p50, p99 = (float(np.percentile(cad[m], q)) for q in (16, 50, 99))
    relief = p99 - p50
    # cytoplasm sigma, blob-immune (high tail excluded); floor scale-RELATIVE so the SNR gate
    # still works on a [0,1]-normalized stain, not just raw integer intensities
    noise = max(p50 - p16, 1e-3 * relief, 1e-9)
    if relief / noise < k:                       # no clear peak -> no nucleus to recover
        return _ret(None, "low_snr")
    thr = p50 + 0.5 * relief                      # halfway up to the brightest -> the blob
    bright = m & (cad > thr)
    if int(bright.sum()) < min_blob:
        return _ret(None, "no_bright")
    lab, _ = cc_label(bright, structure=_conn_full(m.ndim))
    sizes = np.bincount(lab.ravel())[1:]
    if not len(sizes):
        return _ret(None, "no_cc")
    largest = lab == (int(np.argmax(sizes)) + 1)
    sz = int(largest.sum())
    if sz < min_blob:
        return _ret(None, "too_small")
    if sz > max_frac * n:                          # a whole-cell gradient, not a compact nucleus
        return _ret(None, "too_big")
    # boundary guard: a real nucleus sits in the cell INTERIOR. A bright blob hugging the cell
    # edge is most likely a neighbour's nucleus bleeding into the haloed crop -> reject.
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(m, structure=_conn_full(m.ndim), iterations=1, border_value=0)
    if interior.any() and (largest & interior).sum() < 0.5 * sz:
        return _ret(None, "on_boundary")
    return _ret(largest, "ok")


def fill_smooth_cell(
    cell_mask: np.ndarray,
    sigma_z: float = 1.5,
    sigma_xy: float = 1.5,
    smooth: bool = True,
) -> np.ndarray:
    """Fill holes + (optionally) SDF-smooth one cell. Reuses the engine's `smooth_label_sdf`.

    Args:
        cell_mask: bool (Z,Y,X) the cell.
        sigma_z, sigma_xy: smoothing sigma (equal = isotropic).
        smooth: if False, only fill holes.

    Returns:
        bool (Z,Y,X) cleaned cell mask.
    """
    from scipy.ndimage import binary_fill_holes

    m = binary_fill_holes(np.asarray(cell_mask, dtype=bool))
    if not smooth or not m.any():
        return m
    from packages.epicycle_scaledseg.core.smooth import smooth_label_sdf
    result, _, _ = smooth_label_sdf(m.astype(np.int32), 1, sigma_z=sigma_z, sigma_xy=sigma_xy)
    out = np.asarray(result) > 0
    return out if out.any() else m              # never return empty (smoothing a sliver can vanish)


def _smooth_carved_piece(piece, cell_final, on, sigma):
    """SDF-smooth ONE carved nucleus slice so its razor cut face rounds off into a nucleus-like blob.

    Clipped to its own cell AND to still-unwritten nucleus voxels, so the rounded piece can never
    cross the cell wall into the other half (which would re-merge the split) or overwrite a
    neighbouring nucleus in a merged cell. The cut face itself sits on the cell wall, so it stays
    ~planar (clamped) while the sharp edges round — a nucleus pressed against a wall. Each piece is
    smoothed INDEPENDENTLY, so asymmetric splits (one big + one small) each round on their own — no
    symmetry assumed. MAY return an empty mask in the rare case its voxels are all already taken by a
    neighbour nucleus in the cell; the caller re-checks and skips a piece that would write nothing (so
    no phantom zero-voxel label).
    """
    sm = fill_smooth_cell(piece, sigma_z=sigma, sigma_xy=sigma, smooth=True)
    out = sm & cell_final & (on == 0)
    return out if out.any() else (piece & (on == 0))


# --- per-cell worker (mirrors core.feature_stage: workers memmap the source ONCE in an
#     initializer + receive only lightweight (id, bbox) tasks, so the parent never pickles a
#     single voxel of crop to feed the pool). The serial path sets the same globals in-process.
_CL_CELLS = None     # cells label volume (memmap in a pool worker; the live array when serial)
_CL_NUCRAW = None    # raw nuclear-stain volume for recovery (serial path only; None in a pool)


def _init_cleanup_worker(cells_npy_path: str) -> None:
    """Pool initializer: memmap the cell-label volume once per worker (shared OS page cache)."""
    global _CL_CELLS, _CL_NUCRAW
    _CL_CELLS = np.load(cells_npy_path, mmap_mode="r")
    _CL_NUCRAW = None


def _clean_one_cell(args):
    """Worker: read this cell's bbox crop from the shared volume, fill+smooth, maybe recover."""
    cid, bbox, has_assigned, smooth, sigma, recover = args
    z0, z1, y0, y1, x0, x1 = bbox
    cellv = np.asarray(_CL_CELLS[z0:z1, y0:y1, x0:x1])
    m2 = fill_smooth_cell(cellv == cid, sigma_z=sigma, sigma_xy=sigma, smooth=smooth)
    rec = None
    if not has_assigned and recover and _CL_NUCRAW is not None:
        rec = recover_nucleus(np.asarray(_CL_NUCRAW[z0:z1, y0:y1, x0:x1]), m2)
    return cid, m2, rec


# --- majority-assignment worker (its own memmap pair: cells AND nuclei) ---------------------
_MA_CELLS = None
_MA_NUCLEI = None


def _init_majority_worker(cells_npy_path: str, nuclei_npy_path: str) -> None:
    """Pool initializer for the assign pass: memmap both label volumes once per worker."""
    global _MA_CELLS, _MA_NUCLEI
    _MA_CELLS = np.load(cells_npy_path, mmap_mode="r")
    _MA_NUCLEI = np.load(nuclei_npy_path, mmap_mode="r")


def _majority_one(args):
    """Worker: this nucleus's per-cell voxel votes. Returns (nid, vals_tuple, counts_tuple)."""
    nid, bbox = args
    z0, z1, y0, y1, x0, x1 = bbox
    nsub = np.asarray(_MA_NUCLEI[z0:z1, y0:y1, x0:x1])
    under = np.asarray(_MA_CELLS[z0:z1, y0:y1, x0:x1])[nsub == nid]
    under = under[under > 0]
    if under.size == 0:
        return nid, (), ()                            # orphan: no in-cell voxels
    vals, counts = np.unique(under, return_counts=True)
    return nid, tuple(int(v) for v in vals), tuple(int(c) for c in counts)


def _majority_cell_per_nucleus(cells, nuclei, *, parallel=False, n_workers=None, progress=False,
                               cells_path=None, nuclei_path=None, carve=False, min_piece_frac=0.2):
    """Assign each nucleus to the cell containing the MOST of its voxels.

    Returns ``(nuc_to_cell, victim_cells, n_orphan, cell_pieces)``:
      * ``nuc_to_cell`` — ``{nucleus_id: majority_cell_id}`` (the majority cell keeps the id),
      * ``victim_cells`` — cells that LOST (their share of) a nucleus to majority-assignment or to
        below-threshold clipping; these become ``no_nucleus_strung_victim`` if left empty,
      * ``n_orphan`` — nuclei entirely outside every cell (dropped),
      * ``cell_pieces`` — ``{}`` unless ``carve``; then ``{cell_id: [(fresh_id, source_nid), ...]}``
        for the NON-majority slices of a straddling nucleus whose share is >= ``min_piece_frac`` of
        the nucleus's in-cell voxels. Fresh ids are allocated deterministically (sorted) in the
        PARENT, disjoint from every existing cell AND nucleus id — so serial == parallel and no
        two workers can hand out the same id.

    Parallel (same memmap-pool idiom as the cell pass) when ``parallel`` and BOTH ``cells_path``
    and ``nuclei_path`` are given; otherwise serial. This used to be the silent single-core long
    pole — on a big volume it's a per-nucleus loop over tens of thousands of ids.
    """
    from scipy.ndimage import find_objects
    objs = find_objects(nuclei)
    ntasks = [(i + 1, (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, sl[2].start, sl[2].stop))
              for i, sl in enumerate(objs) if sl is not None]
    nuc_to_cell, victim_cells, n_orphan = {}, set(), 0
    carve_requests = []          # (source_nid, cid) for each qualifying non-majority slice

    def _handle(nid, vals, counts):
        """Fold one nucleus's per-cell votes into the assignment + (carve) split decision."""
        nonlocal n_orphan
        if not vals:
            n_orphan += 1
            return
        counts_a = np.asarray(counts)
        maj = int(vals[int(np.argmax(counts_a))])        # ties -> lowest cid (vals is sorted asc)
        nuc_to_cell[nid] = maj
        if carve:
            total = int(counts_a.sum())
            for v, c in zip(vals, counts):
                v = int(v)
                if v == maj:
                    continue
                # every non-majority cell is a potential victim; it ESCAPES that label only by
                # actually receiving its carved slice in the write-back. So a qualifying slice
                # that later vanishes (e.g. smoothing erodes it) is still honestly a victim.
                victim_cells.add(v)
                if total > 0 and (c / total) >= min_piece_frac:
                    carve_requests.append((int(nid), v))   # this cell keeps its own slice
        elif len(vals) >= 2:
            victim_cells.update(int(v) for v in vals if int(v) != maj)

    if parallel and len(ntasks) > 1 and cells_path is not None and nuclei_path is not None:
        import multiprocessing as mp
        from ..utils.resources import get_allocated_cpus
        nw = n_workers or max(1, get_allocated_cpus() - 1)
        nw = max(1, min(int(nw), len(ntasks)))
        with mp.Pool(nw, initializer=_init_majority_worker,
                     initargs=(str(cells_path), str(nuclei_path))) as pool:
            for nid, vals, counts in _maybe_tqdm(
                    pool.imap_unordered(_majority_one, ntasks, chunksize=64),
                    total=len(ntasks), desc=f"assign ({nw}w)", enable=progress):
                _handle(nid, vals, counts)
    else:
        for nid, bbox in _maybe_tqdm(ntasks, total=len(ntasks), desc="assign", enable=progress):
            z0, z1, y0, y1, x0, x1 = bbox
            nsub = nuclei[z0:z1, y0:y1, x0:x1]
            under = cells[z0:z1, y0:y1, x0:x1][nsub == nid]
            under = under[under > 0]
            if under.size == 0:
                _handle(nid, (), ())
            else:
                v_a, c_a = np.unique(under, return_counts=True)
                _handle(nid, tuple(int(v) for v in v_a), tuple(int(c) for c in c_a))

    # deterministic fresh-id allocation for carved slices (sorted -> serial == parallel)
    cell_pieces: dict = {}
    if carve and carve_requests:
        base = int(max(int(cells.max()), int(nuclei.max()))) + 1
        for k, (source_nid, cid) in enumerate(sorted(carve_requests)):
            cell_pieces.setdefault(cid, []).append((base + k, source_nid))
    return nuc_to_cell, victim_cells, n_orphan, cell_pieces


def cleanup_volume(
    cells: np.ndarray,
    nuclei: np.ndarray,
    nuclei_raw: Optional[np.ndarray] = None,
    *,
    smooth: bool = True,
    sigma: float = 1.5,
    min_cell_vox: int = 0,
    recover: bool = True,
    split_straddling_nuclei: bool = False,
    min_piece_frac: float = 0.2,
    smooth_split_nuclei: bool = False,
    parallel: bool = False,
    n_workers: Optional[int] = None,
    progress: bool = False,
    cells_path: Optional[str] = None,
    nuclei_path: Optional[str] = None,
    return_triage: bool = False,
):
    """Clean (cells, nuclei) into a CONSISTENT pair — every nucleus inside exactly one cell.

    Trusts the CELL labels; makes the nuclei conform:
      * each nucleus is assigned to the cell it's MOSTLY in and clipped to it. By default a nucleus
        that straddles a seam goes WHOLLY to its majority cell (the overhang is discarded and the
        loser cell is flagged ``no_nucleus_strung_victim``). With ``split_straddling_nuclei=True``
        (CARVE mode) the nucleus is instead cut at the cell boundary: each cell whose share is
        >= ``min_piece_frac`` of the nucleus's in-cell voxels keeps its own slice as a nucleus
        (majority cell keeps the original id; others get fresh globally-unique ids), so the loser
        cell gets its nucleus back and the QUALIFYING slices lose no volume. A slice BELOW the
        threshold is DISCARDED (its cell keeps nothing and is flagged a strung victim, exactly as
        in the default majority-assign) so registration slop doesn't spawn a sliver nucleus.
        With ``smooth_split_nuclei=True`` each carved slice (both halves of a split) is SDF-smoothed
        (``sigma``) clipped to its cell, so the razor cut face rounds into a nucleus-like blob
        (clean + recovered nuclei are NOT touched); carved slice ids are listed per cell in the
        triage ``carved_nucleus_ids`` for downstream shape-stat exclusion,
      * per cell: fill holes + (optional) SDF-smooth,
      * a cell with no assigned nucleus -> try to RECOVER one from ``nuclei_raw`` (local rescale),
      * fragment cells (< ``min_cell_vox``) and orphan nuclei (outside every cell) are dropped.
    Merged cells (2+ genuine nuclei, each mostly inside) are KEPT and counted as the residual.

    Nucleus ids are PRESERVED (each nucleus keeps its id on its majority cell); only recovered
    nuclei get fresh ids.

    Parallelism reuses the engine's proven idiom (``core.feature_stage._extract_parallel``):
    pass ``cells_path`` (the cell-label ``.npy``) and ``parallel=True`` and each worker MEMMAPS
    that file once via a pool initializer, then receives only lightweight ``(id, bbox)`` tasks —
    the parent never pickles crops, so the workers actually saturate their cores. Worker count is
    ``n_workers`` or the SLURM-aware ``get_allocated_cpus()-1`` (respects ``SLURM_CPUS_PER_TASK`` /
    cgroup affinity — never the bare node count). Without ``cells_path`` the run is SERIAL.
    Pass ``nuclei_path`` too to also parallelize the nucleus->cell assignment pass (the other
    big single-core phase). Recovery in the pooled path is deferred (do it as the separate
    per-bbox stain pass). With ``progress=True`` every phase prints a marker + a tqdm bar.

    Returns:
        ``(out_cells, out_nuclei, report)``, or ``(..., triage)`` when ``return_triage=True`` —
        ``triage`` is one dict per surviving cell: ``{cell_id, status, n_nuclei, nucleus_ids,
        cell_voxels}`` (the per-cell view of ``report``, for the feature/triage table).
        ``report`` counts: clean / merged_cell (the parked residual) / split_nucleus (a cell whose
        nucleus is a carved slice of a straddling nucleus — CARVE mode only) / recovered_nucleus /
        no_nucleus_strung_victim (a real cell the cleanup stripped of its nucleus) /
        no_nucleus_genuine / dropped_fragment / dropped_nucleus_with_fragment /
        orphan_nucleus_dropped (nuclei outside every cell = probable missed cells).
    """
    from collections import Counter, defaultdict
    from scipy.ndimage import find_objects

    def _log(msg):
        if progress:
            print(msg, flush=True)

    cells = np.asarray(cells)
    nuclei = np.asarray(nuclei)
    if cells.shape != nuclei.shape:
        raise ValueError("cells and nuclei must share a shape")
    # the pool workers read the *_path memmaps while the parent assigns/writes from the in-memory
    # arrays — if those disagree the result is silently wrong. Verify shape AND a sparse content
    # sample (catches "wrong file / mutated-after-load" cheaply without hashing the whole volume).
    for _p, _a, _n in ((cells_path, cells, "cells"), (nuclei_path, nuclei, "nuclei")):
        if _p is None:
            continue
        _mm = np.load(str(_p), mmap_mode="r")
        if _mm.shape != _a.shape:
            raise ValueError(f"{_n}_path shape {_mm.shape} != in-memory {_n} {_a.shape}")
        if _a.size:
            _idx = np.linspace(0, _a.size - 1, min(_a.size, 4096), dtype=np.int64)
            if not np.array_equal(np.asarray(_mm).flat[_idx], _a.flat[_idx]):
                raise ValueError(
                    f"{_n}_path content != in-memory {_n} (mutated after load, or wrong file). "
                    f"Pass the path the array was loaded from, before any in-place edits.")

    cdt0 = cells.dtype if cells.dtype.itemsize >= 4 else np.int32
    ndt0 = nuclei.dtype if nuclei.dtype.itemsize >= 4 else np.int32
    if cells.size == 0:                                   # degenerate empty volume -> nothing to do
        empty = (np.zeros(cells.shape, cdt0), np.zeros(nuclei.shape, ndt0), {})
        return (*empty, []) if return_triage else empty

    _log("[1/4] indexing nuclei (full-volume scan) + assigning each to its majority cell…")
    nuc_to_cell, victim_cells, n_orphan, cell_pieces = _majority_cell_per_nucleus(
        cells, nuclei, parallel=parallel, n_workers=n_workers, progress=progress,
        cells_path=cells_path, nuclei_path=nuclei_path,
        carve=split_straddling_nuclei, min_piece_frac=min_piece_frac)
    cell_to_nuc: dict = defaultdict(list)
    for nid, cid in nuc_to_cell.items():
        cell_to_nuc[cid].append(nid)

    # int32 out so fresh recovered ids can't wrap a uint8/uint16 input label volume
    cdt = cells.dtype if cells.dtype.itemsize >= 4 else np.int32
    ndt = nuclei.dtype if nuclei.dtype.itemsize >= 4 else np.int32
    out_cells = np.zeros(cells.shape, dtype=cdt)
    out_nuclei = np.zeros(nuclei.shape, dtype=ndt)
    report: Counter = Counter()
    report["orphan_nucleus_dropped"] = int(n_orphan)      # nuclei outside every cell (missed cells)

    # lightweight per-cell tasks (id + bbox + flags) — no crop travels to the workers
    _log("[2/4] indexing cells (full-volume scan) + bbox/fragment filter…")
    tasks, cid_to_sl = [], {}
    for idx, sl in _maybe_tqdm(list(enumerate(find_objects(cells))),
                               desc="scan cells", enable=progress):
        if sl is None:
            continue
        cid = idx + 1
        if int((cells[sl] == cid).sum()) < min_cell_vox:
            report["dropped_fragment"] += 1
            # count BOTH the majority nuclei AND any carved slices that would have landed here,
            # so a carve target lost to a fragment is not a silent disappearance
            report["dropped_nucleus_with_fragment"] += (len(cell_to_nuc.get(cid, []))
                                                        + len(cell_pieces.get(cid, [])))
            continue
        cid_to_sl[cid] = sl
        bbox = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, sl[2].start, sl[2].stop)
        tasks.append((cid, bbox, bool(cell_to_nuc.get(cid)), smooth, sigma, recover))

    n_tasks = len(tasks)
    _log(f"[3/4] cleaning {n_tasks:,} cells (fill{'+smooth' if smooth else ''})…")
    global _CL_CELLS, _CL_NUCRAW
    if parallel and n_tasks > 1 and cells_path is not None:
        import multiprocessing as mp
        from ..utils.resources import get_allocated_cpus
        if recover and nuclei_raw is not None:
            print("  WARNING: recovery is SKIPPED in the pooled (parallel) cell pass — run it as "
                  "the separate stain pass, or call with parallel=False.", flush=True)
        if n_workers is None:
            n_workers = max(1, get_allocated_cpus() - 1)   # SLURM-aware (cgroup affinity, not node)
        n_workers = max(1, min(int(n_workers), n_tasks))
        # workers memmap cells_path once (initializer) + pull only (id,bbox) tasks -> cores saturate
        with mp.Pool(n_workers, initializer=_init_cleanup_worker, initargs=(str(cells_path),)) as pool:
            results = list(_maybe_tqdm(
                pool.imap_unordered(_clean_one_cell, tasks, chunksize=16),
                total=n_tasks, desc=f"cleanup ({n_workers}w)", enable=progress))
    else:
        # serial (or parallel requested without cells_path): same worker, shared in-process volumes
        _CL_CELLS, _CL_NUCRAW = cells, nuclei_raw
        try:
            results = [_clean_one_cell(t)
                       for t in _maybe_tqdm(tasks, total=n_tasks, desc="cleanup", enable=progress)]
        finally:
            _CL_CELLS = _CL_NUCRAW = None

    _log("[4/4] writing the consensus volumes…")
    # recovery ids start ABOVE the highest carved id so the two allocators can't collide
    _carved_ids = [wid for ps in cell_pieces.values() for (wid, _src) in ps]
    # original nuclei that got split -> their MAJORITY slice is a carved piece too (both halves)
    _split_sources = {src for ps in cell_pieces.values() for (_wid, src) in ps}
    next_nuc = (max([int(nuclei.max())] + _carved_ids) + 1) if nuclei.size else 1
    triage = []                                           # one row per surviving cell (return_triage)
    for cid, m2, rec in _maybe_tqdm(results, total=len(results),
                                    desc="write-back", enable=progress):  # imap_unordered -> cid_to_sl
        sl = cid_to_sl[cid]
        oc = out_cells[sl]
        write = m2 & ((cells[sl] == cid) | (cells[sl] == 0)) & (oc == 0)
        oc[write] = cid
        cell_final = oc == cid
        vox = int(cell_final.sum())
        on = out_nuclei[sl]
        nucv = nuclei[sl]
        assigned = cell_to_nuc.get(cid, [])
        carved = cell_pieces.get(cid, [])
        nuc_ids = []; carved_ids = []
        # majority-assigned nuclei keep their original id (the MAJORITY slice of a split is carved too)
        for nid in assigned:
            piece = (nucv == nid) & cell_final
            is_carved = nid in _split_sources                  # this majority slice was cut at a wall
            if is_carved and smooth_split_nuclei and piece.any():
                piece = _smooth_carved_piece(piece, cell_final, on, sigma)
            if piece.any():                                    # re-check AFTER smoothing -> no phantom id
                on[piece] = nid
                nuc_ids.append(int(nid))
                if is_carved:
                    carved_ids.append(int(nid))
        # CARVE mode: a straddling nucleus's slice in this cell becomes its OWN fresh nucleus
        got_carved = False
        for write_id, source_nid in carved:                    # receiver slices are always carved
            piece = (nucv == source_nid) & cell_final
            if smooth_split_nuclei and piece.any():
                piece = _smooth_carved_piece(piece, cell_final, on, sigma)
            if piece.any():                                    # re-check AFTER smoothing -> no phantom id
                on[piece] = write_id
                nuc_ids.append(int(write_id))
                got_carved = True
                carved_ids.append(int(write_id))
        if carved_ids:
            report["carved_nucleus_pieces"] += len(carved_ids)
        # recover a missed nucleus ONLY when the cell still has none
        rec_written = False
        if not nuc_ids and rec is not None and (rec & cell_final).any():
            on[rec & cell_final] = next_nuc
            nuc_ids.append(int(next_nuc)); next_nuc += 1
            rec_written = True
        if len(nuc_ids) >= 2:
            status = "merged_cell"
        elif len(nuc_ids) == 1:
            if rec_written:
                status = "recovered_nucleus"
            elif got_carved and not assigned:
                status = "split_nucleus"          # this cell's only nucleus is a carved slice
            else:
                status = "clean"
        elif cid in victim_cells:
            # a real cell LEFT with no nucleus by majority-assignment/clipping (the cleanup's own
            # doing) — reported separately from a genuinely anucleate cell, for honesty
            status = "no_nucleus_strung_victim"
        else:
            status = "no_nucleus_genuine"
        report[status] += 1
        if return_triage:
            triage.append({"cell_id": int(cid), "status": status,
                           "n_nuclei": len(nuc_ids), "nucleus_ids": nuc_ids,
                           "carved_nucleus_ids": carved_ids, "cell_voxels": vox})

    # mass conservation: every cleaned cell got EXACTLY one status — no silent disappearance
    _accounted = sum(report[k] for k in ("clean", "merged_cell", "split_nucleus",
                                         "recovered_nucleus", "no_nucleus_strung_victim",
                                         "no_nucleus_genuine"))
    assert _accounted == len(results), (
        f"cell accounting leak: {_accounted} statuses != {len(results)} cleaned cells")

    if return_triage:
        return out_cells, out_nuclei, dict(report), triage
    return out_cells, out_nuclei, dict(report)


def tidy_labels(
    labels: np.ndarray,
    *,
    min_vox: int = 0,
    smooth: bool = False,
    sigma: float = 1.5,
    parallel: bool = False,
    n_workers: Optional[int] = None,
    labels_path: Optional[str] = None,
    progress: bool = False,
    return_report: bool = False,
):
    """Per-segmentation TIDY (channel-agnostic, single-seg): drop fragments (< ``min_vox``), fill
    holes, optional SDF smooth. No nucleus involved.

    This is the shared FRONT-HALF of the workflow: a single-channel run uses THIS and stops (then
    extract features); ``cleanup_volume`` (the cross-channel RECONCILE) does the same per-cell tidy
    and layers the cell<->nucleus consensus on top. Reuses the exact per-cell core
    (``_clean_one_cell`` + the ``_init_cleanup_worker`` memmap pool) — pass ``labels_path`` (the
    ``.npy``) + ``parallel=True`` to saturate cores; fill-only work is memmap-I/O-bound so the big
    CPU win is ``smooth=True``.

    Returns ``(out_labels, report)`` — ``report`` counts ``kept`` / ``dropped_fragment`` — or
    ``(out_labels, report, kept_records)`` when ``return_report=True`` (one ``{label, voxels}``
    row per surviving object).
    """
    from collections import Counter
    from scipy.ndimage import find_objects
    global _CL_CELLS, _CL_NUCRAW

    def _log(msg):
        if progress:
            print(msg, flush=True)

    labels = np.asarray(labels)
    cdt = labels.dtype if labels.dtype.itemsize >= 4 else np.int32
    if labels_path is not None and labels.size:               # path drives the pool memmap -> verify
        mm = np.load(str(labels_path), mmap_mode="r")
        if mm.shape != labels.shape:
            raise ValueError(f"labels_path shape {mm.shape} != in-memory labels {labels.shape}")
        si = np.linspace(0, labels.size - 1, min(labels.size, 4096), dtype=np.int64)
        if not np.array_equal(np.asarray(mm).flat[si], labels.flat[si]):
            raise ValueError("labels_path content != in-memory labels (wrong file / mutated)")
        del mm
    out = np.zeros(labels.shape, cdt)
    report: Counter = Counter()
    if labels.size == 0:
        return (out, dict(report), []) if return_report else (out, dict(report))

    _log("[1/3] indexing objects (full-volume scan) + fragment filter…")
    tasks, cid_to_sl = [], {}
    for idx, sl in _maybe_tqdm(list(enumerate(find_objects(labels))), desc="scan", enable=progress):
        if sl is None:
            continue
        cid = idx + 1
        if int((labels[sl] == cid).sum()) < min_vox:
            report["dropped_fragment"] += 1
            continue
        cid_to_sl[cid] = sl
        bbox = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, sl[2].start, sl[2].stop)
        tasks.append((cid, bbox, True, smooth, sigma, False))   # has_assigned=True -> no recovery

    n_tasks = len(tasks)
    _log(f"[2/3] cleaning {n_tasks:,} objects (fill{'+smooth' if smooth else ''})…")
    if parallel and n_tasks > 1 and labels_path is not None:
        import multiprocessing as mp
        from ..utils.resources import get_allocated_cpus
        nw = n_workers or max(1, get_allocated_cpus() - 1)
        nw = max(1, min(int(nw), n_tasks))
        with mp.Pool(nw, initializer=_init_cleanup_worker, initargs=(str(labels_path),)) as pool:
            results = list(_maybe_tqdm(pool.imap_unordered(_clean_one_cell, tasks, chunksize=16),
                                       total=n_tasks, desc=f"tidy ({nw}w)", enable=progress))
    else:
        _CL_CELLS, _CL_NUCRAW = labels, None
        try:
            results = [_clean_one_cell(t)
                       for t in _maybe_tqdm(tasks, total=n_tasks, desc="tidy", enable=progress)]
        finally:
            _CL_CELLS = _CL_NUCRAW = None

    _log("[3/3] writing…")
    kept = []
    for cid, m2, _rec in _maybe_tqdm(results, total=len(results), desc="write", enable=progress):
        sl = cid_to_sl[cid]
        oc = out[sl]
        write = m2 & ((labels[sl] == cid) | (labels[sl] == 0)) & (oc == 0)
        oc[write] = cid
        report["kept"] += 1
        if return_report:
            kept.append({"label": int(cid), "voxels": int((oc == cid).sum())})

    return (out, dict(report), kept) if return_report else (out, dict(report))


__all__ = ["carve_nucleus_by_cells", "recover_nucleus", "fill_smooth_cell", "cleanup_volume",
           "tidy_labels"]
