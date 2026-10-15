"""
Mode-B ("full") full-resolution IoU seam-stitch — PURE ALGORITHM CORE.

This is the centerpiece of the full-resolution path: instead of Mode A's lossy
"downsample-then-re-segment" final stage, Mode B keeps the per-tile instance
labels from Stages 0-2 and *stitches them at full resolution* by matching cells
across the tile-overlap seams via IoU, merging the matches with union-find, and
emitting a dense global relabel LUT. No second u-Segment3D solve, no downsample,
and — critically — **no whole full-Z volume is ever materialized in RAM.**

Everything here is pure (numpy / scipy / tifffile only). There is NO GPU/Qt/
napari surface and heavy imports are deferred to call time. The Stage orchestrator,
the writer-to-disk wiring, and the CLI ``--mode`` flag live in a SEPARATE task —
this module only provides the testable, RAM-bounded primitives those will call.

Memory contract (the make-or-break engineering): a full-length
vertical seam at Z=1000 is multi-GB *per side*. So the co-occurrence accumulator
is chunk-incremental from line 1 — it iterates Z in blocks of ``z_block`` and
reads ONLY the overlap sub-rect of each tile per block, never a single
``np.unique`` over a whole seam slab and never a full-Z load.

Geometry contract: a tile's geometry is either a :class:`~epicycle.utils.tile_grid.TileGeom`
or an equivalent dict (``{tile_name, origin:[y0,x0], extent:[h,w], grid_index,
is_edge, is_flush}``). Origins are STORED and flush-aware — a flush tile's origin
is ``dim - patch``, NOT ``grid_index * step`` — so they are used verbatim and
never regex-derived from the directory name.

Public surface
--------------
* :func:`overlap_rect`             — global intersection of two tile rects (or None).
* :func:`enumerate_adjacent_pairs` — all overlapping unordered tile pairs + rect.
* :func:`compute_global_offsets`   — running-counter offset map (no ``id*BIG`` overflow).
* :func:`tile_nmax`                — streaming per-Z-page max LOCAL label.
* :func:`overlap_iou_edges`        — THE CORE: chunk-incremental IoU seam matcher.
* :func:`union_find_merge`         — weighted union + path compression over edges.
* :func:`compact_roots`            — distinct roots -> dense 1..K (background 0 stays 0).
* :func:`compute_tile_cores`       — flush-aware disjoint per-tile "core" (midpoint rule).
* :func:`apply_relabel_to_slice`   — vectorized LUT remap of a 2D slice.
* :func:`build_lut_array`          — {global:dense} dict -> dense int32 LUT array.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Geometry inputs are accepted as TileGeom OR plain dicts. We never *require*
# the import (dicts work standalone); the type is only used for isinstance-free
# attribute access below.
GeomLike = Union["object", Dict]

Rect = Tuple[int, int, int, int]  # (y0, y1, x0, x1)


# ---------------------------------------------------------------------------
# Geometry accessors (TileGeom OR dict)
# ---------------------------------------------------------------------------


def _geom_name(g: GeomLike) -> str:
    if isinstance(g, dict):
        return g["tile_name"]
    return g.name


def _geom_origin(g: GeomLike) -> Tuple[int, int]:
    if isinstance(g, dict):
        y0, x0 = g["origin"]
        return int(y0), int(x0)
    y0, x0 = g.origin
    return int(y0), int(x0)


def _geom_extent(g: GeomLike) -> Tuple[int, int]:
    if isinstance(g, dict):
        h, w = g["extent"]
        return int(h), int(w)
    h, w = g.extent
    return int(h), int(w)


def _geom_rect(g: GeomLike) -> Rect:
    """Global (y0, y1, x0, x1) rectangle a tile occupies."""
    oy, ox = _geom_origin(g)
    h, w = _geom_extent(g)
    return (oy, oy + h, ox, ox + w)


# ---------------------------------------------------------------------------
# 1) overlap_rect
# ---------------------------------------------------------------------------


def overlap_rect(geomA: GeomLike, geomB: GeomLike) -> Optional[Rect]:
    """Global ``(y0, y1, x0, x1)`` intersection of two tile rects, or ``None``.

    A single rectangle-intersection covers horizontal, vertical AND 4-tile-corner
    overlaps with no special-casing. Returns ``None`` for a disjoint (or
    edge-touching, zero-area) pair.
    """
    ay0, ay1, ax0, ax1 = _geom_rect(geomA)
    by0, by1, bx0, bx1 = _geom_rect(geomB)
    y0 = max(ay0, by0)
    y1 = min(ay1, by1)
    x0 = max(ax0, bx0)
    x1 = min(ax1, bx1)
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1)


# ---------------------------------------------------------------------------
# 2) enumerate_adjacent_pairs
# ---------------------------------------------------------------------------


def enumerate_adjacent_pairs(
    geoms: Sequence[GeomLike],
) -> List[Tuple[str, str, Rect]]:
    """All unordered tile pairs with a non-empty overlap.

    Returns ``[(nameA, nameB, global_rect), ...]``. The pair order within each
    tuple follows the input order (i < j); the one rect-intersection covers
    horizontal / vertical / corner adjacency uniformly.
    """
    out: List[Tuple[str, str, Rect]] = []
    n = len(geoms)
    for i in range(n):
        for j in range(i + 1, n):
            rect = overlap_rect(geoms[i], geoms[j])
            if rect is not None:
                out.append((_geom_name(geoms[i]), _geom_name(geoms[j]), rect))
    return out


# ---------------------------------------------------------------------------
# 3) compute_global_offsets — running counter (no id*BIG overflow)
# ---------------------------------------------------------------------------


def compute_global_offsets(tile_nmax: Dict[str, int]) -> Dict[str, int]:
    """Running-counter global-id offsets, deterministic by sorted tile name.

    ``offset[t]`` = cumulative sum of all *prior* tiles' ``nmax``. The global id
    of local label ``L > 0`` in tile ``t`` is ``L + offset[t]``. Because tile
    ``t``'s id block is ``(offset[t], offset[t] + nmax[t]]`` (open on the low end —
    local labels start at 1), adjacent blocks never collide at the boundary.

    NOT ``tile_id * BIG`` (that overflows at scale).
    """
    offsets: Dict[str, int] = {}
    running = 0
    for name in sorted(tile_nmax):
        offsets[name] = running
        running += int(tile_nmax[name])
    return offsets


# ---------------------------------------------------------------------------
# 4) tile_nmax — STREAMING per-Z-page max (never loads whole full-Z volume)
# ---------------------------------------------------------------------------


def tile_nmax(mask_path: str) -> int:
    """Max LOCAL label in a tile mask, computed page-by-page (streaming).

    Reads one Z-page at a time via ``tifffile.TiffFile`` and keeps a running max,
    so a multi-GB full-Z tile never lands in RAM at once. Returns ``0`` for an
    all-background tile.
    """
    import tifffile  # deferred heavy import

    running = 0
    with tifffile.TiffFile(mask_path) as tf:
        for page in tf.series[0].pages:
            arr = page.asarray()
            m = int(arr.max())
            if m > running:
                running = m
    return running


# ---------------------------------------------------------------------------
# 5) overlap_iou_edges — THE CORE (chunk-incremental, RAM-bounded)
# ---------------------------------------------------------------------------


def _read_overlap_block(
    mask_path: str,
    rect_local: Rect,
    z_start: int,
    z_stop: int,
) -> np.ndarray:
    """Read ONLY ``rect_local`` of pages ``[z_start, z_stop)`` from a tile mask.

    Returns a ``(z_stop - z_start, dy, dx)`` array. This is the single named read
    site the memory-invariant spy wraps — it must never decode more than
    ``z_stop - z_start`` pages, and that span is bounded by ``z_block`` upstream.
    """
    import tifffile  # deferred heavy import

    y0, y1, x0, x1 = rect_local
    with tifffile.TiffFile(mask_path) as tf:
        series = tf.series[0]
        series_pages = series.pages
        if len(series_pages) >= z_stop:
            # Production case (Z >= ~5): one TIFF page per Z-slice -> read ONLY
            # the requested pages, RAM-bounded to (z_stop-z_start) pages.
            pages = [series_pages[z].asarray()[y0:y1, x0:x1]
                     for z in range(z_start, z_stop)]
            return np.stack(pages, axis=0)
        # Fallback: tifffile collapses a tiny contiguous volume (Z in {3,4}) into
        # a SINGLE page, so per-page indexing would IndexError. Read the whole
        # (necessarily tiny) series and slice by Z. Only triggers when the page
        # count < depth, i.e. volumes small enough that this is cheap.
        vol = series.asarray()
        return vol[z_start:z_stop, y0:y1, x0:x1]


def overlap_iou_edges(
    maskA_path: str,
    maskB_path: str,
    rectA_local: Rect,
    rectB_local: Rect,
    offsetA: int,
    offsetB: int,
    depth: int,
    *,
    tau: float = 0.25,
    vmin: int = 10,
    one_to_many: str = "max_iou",
    containment_rescue: bool = False,
    z_block: int = 64,
) -> List[Tuple[int, int, float]]:
    """Match instances across one tile-pair seam by overlap-restricted IoU.

    The accumulator is **chunk-incremental**: Z is walked in blocks of
    ``z_block``; each block reads ONLY the overlap sub-rect of each tile, packs the
    co-occurring ``(aL, bL)`` pairs into a bounded ``int64`` key, and runs
    ``np.unique`` on the BLOCK alone — never over a whole seam slab. The running
    dicts ``cooc``/``areaA``/``areaB`` accumulate intersection + per-label slab
    areas restricted to the overlap.

    IoU (restricted to the slab): ``iou = inter / (areaA[a] + areaB[b] - inter)``.

    Candidate edges clear ``iou >= tau AND inter >= vmin``. Resolution:

      * ``one_to_many="max_iou"`` (default, Q3 under-merge bias) — sort candidates
        by descending IoU, greedily accept, marking each local A and each local B
        consumed once; skip a candidate if either side is already matched.
      * ``one_to_many="greedy"`` — emit every above-threshold candidate (no
        consumed-once guard).

    ``containment_rescue`` (default OFF): a second pass accepting unmatched pairs
    whose ``inter / min(areaA, areaB) >= tau`` — implemented but off by default
    (defaulting it ON risks silent over-merge, which would compound the user's
    undersegmentation bias).

    Emits GLOBAL-id edges ``(aL + offsetA, bL + offsetB, iou)``.
    """
    rA_dims = (rectA_local[1] - rectA_local[0], rectA_local[3] - rectA_local[2])
    rB_dims = (rectB_local[1] - rectB_local[0], rectB_local[3] - rectB_local[2])
    if rA_dims != rB_dims:
        raise ValueError(
            f"overlap sub-rects must be the same shape; got {rA_dims} vs {rB_dims}"
        )

    cooc: Dict[Tuple[int, int], int] = {}
    areaA: Dict[int, int] = {}
    areaB: Dict[int, int] = {}

    z = 0
    while z < depth:
        z_stop = min(z + z_block, depth)
        blockA = _read_overlap_block(maskA_path, rectA_local, z, z_stop)
        blockB = _read_overlap_block(maskB_path, rectB_local, z, z_stop)

        fa = blockA.reshape(-1).astype(np.int64)
        fb = blockB.reshape(-1).astype(np.int64)

        # Per-label slab areas (foreground only).
        if fa.any():
            la_vals, la_counts = np.unique(fa[fa > 0], return_counts=True)
            for v, c in zip(la_vals.tolist(), la_counts.tolist()):
                areaA[v] = areaA.get(v, 0) + c
        if fb.any():
            lb_vals, lb_counts = np.unique(fb[fb > 0], return_counts=True)
            for v, c in zip(lb_vals.tolist(), lb_counts.tolist()):
                areaB[v] = areaB.get(v, 0) + c

        # Co-occurrence: only voxels where BOTH sides are foreground.
        #
        # We pack each (aL, bL) into a single int64 `aL * block_kmax + bL` so the
        # block-local np.unique is one-dimensional (bounded to THIS block, never a
        # whole-slab unique). `block_kmax` is recomputed per block and used only to
        # pack/unpack WITHIN the block — the running `cooc` dict is keyed by the
        # real (aL, bL) tuple, so a different block_kmax in a later block is
        # harmless (no global KMAX, no re-keying needed).
        both = (fa > 0) & (fb > 0)
        if both.any():
            aa = fa[both]
            bb = fb[both]
            block_kmax = int(bb.max()) + 1
            packed = aa * block_kmax + bb
            keys, counts = np.unique(packed, return_counts=True)
            for k, c in zip(keys.tolist(), counts.tolist()):
                key = (k // block_kmax, k % block_kmax)
                cooc[key] = cooc.get(key, 0) + int(c)

        z = z_stop

    # --- Build candidate IoU list ---------------------------------------------
    candidates: List[Tuple[float, int, int, int]] = []  # (iou, aL, bL, inter)
    for (aL, bL), inter in cooc.items():
        if inter < vmin:
            continue
        union = areaA.get(aL, 0) + areaB.get(bL, 0) - inter
        if union <= 0:
            continue
        iou = inter / union
        if iou >= tau:
            candidates.append((iou, aL, bL, inter))

    edges: List[Tuple[int, int, float]] = []
    matched_a = set()
    matched_b = set()

    if one_to_many == "greedy":
        # Emit every above-threshold candidate (deterministic order by -iou).
        for iou, aL, bL, _inter in sorted(candidates, key=lambda t: (-t[0], t[1], t[2])):
            edges.append((aL + offsetA, bL + offsetB, float(iou)))
            matched_a.add(aL)
            matched_b.add(bL)
    else:
        # 1:1 max-IoU (default): greedy descending, consume each side once.
        for iou, aL, bL, _inter in sorted(candidates, key=lambda t: (-t[0], t[1], t[2])):
            if aL in matched_a or bL in matched_b:
                continue
            edges.append((aL + offsetA, bL + offsetB, float(iou)))
            matched_a.add(aL)
            matched_b.add(bL)

    # --- Optional containment rescue (OFF by default) -------------------------
    if containment_rescue:
        for (aL, bL), inter in cooc.items():
            if inter < vmin:
                continue
            if aL in matched_a or bL in matched_b:
                continue
            smaller = min(areaA.get(aL, 0), areaB.get(bL, 0))
            if smaller <= 0:
                continue
            if inter / smaller >= tau:
                iou_val = inter / (areaA.get(aL, 0) + areaB.get(bL, 0) - inter)
                edges.append((aL + offsetA, bL + offsetB, float(iou_val)))
                matched_a.add(aL)
                matched_b.add(bL)

    return edges


# ---------------------------------------------------------------------------
# 6) union_find_merge — weighted union + path compression
# ---------------------------------------------------------------------------


def union_find_merge(
    ids: Sequence[int],
    edges: Sequence[Tuple[int, int, float]],
) -> Dict[int, int]:
    """Union-find over the global ids + edge list. Returns ``{global_id: root}``.

    Weighted union by rank + path compression; zero-dependency and transitive
    (A-B, B-C -> one component). Any id referenced by an edge but absent from
    ``ids`` is added defensively.
    """
    parent: Dict[int, int] = {int(i): int(i) for i in ids}
    rank: Dict[int, int] = {int(i): 0 for i in ids}

    def _find(x: int) -> int:
        # iterative path compression
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _ensure(x: int) -> None:
        if x not in parent:
            parent[x] = x
            rank[x] = 0

    for a, b, _w in edges:
        a, b = int(a), int(b)
        _ensure(a)
        _ensure(b)
        ra, rb = _find(a), _find(b)
        if ra == rb:
            continue
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    return {x: _find(x) for x in parent}


# ---------------------------------------------------------------------------
# 7) compact_roots — distinct roots -> dense 1..K (background 0 stays 0)
# ---------------------------------------------------------------------------


def compact_roots(roots: Dict[int, int]) -> Dict[int, int]:
    """Map distinct roots to DENSE ``1..K`` (sorted for determinism).

    Background ``0`` (if present as a key) stays ``0``. Returns the final LUT
    ``{global_id: dense_id}`` covering every key in ``roots``.
    """
    distinct = sorted({r for k, r in roots.items() if k != 0})
    root_to_dense = {r: i + 1 for i, r in enumerate(distinct)}
    lut: Dict[int, int] = {}
    for k, r in roots.items():
        lut[k] = 0 if k == 0 else root_to_dense[r]
    return lut


# ---------------------------------------------------------------------------
# 7b) Seam provenance — which final cells live in problem regions (QC keystone)
# ---------------------------------------------------------------------------


def classify_seam_status(n_tiles: int) -> str:
    """Map a cell's contributing-tile count to a seam-risk class.

    ``clean`` (1 tile) = entirely inside one tile's core, fully trusted.
    ``seam`` (2 tiles) = crossed one edge overlap.
    ``corner`` (3+ tiles) = involved a 3-/4-way corner overlap — the riskiest
    region for dense-tissue seam disagreement. ``seam``/``corner`` cells are the
    QC worklist; ``clean`` cells need no special review.
    """
    if n_tiles <= 1:
        return "clean"
    if n_tiles == 2:
        return "seam"
    return "corner"


def compute_label_provenance(
    present_ids: Sequence[int],
    gid_to_tile: Dict[int, str],
    offsets: Dict[str, int],
    dense_lut: Dict[int, int],
    edges: Optional[Sequence[Tuple[int, int, float]]] = None,
) -> Dict[int, dict]:
    """Per FINAL (dense) cell: which tiles contributed, seam-risk class + match quality.

    Pure bookkeeping over the union-find result — NO volume read. For each present
    per-tile global id, look up its dense label + its source tile; group by dense
    label to count distinct contributing tiles. A cell merged from 2+ tiles
    physically crossed a seam; 3+ means a corner.

    ``edges`` (the same ``(a_global, b_global, iou)`` list fed to ``union_find_merge``)
    is optional; when given, each multi-tile cell also gets ``min_match_iou`` — the
    WEAKEST seam-match IoU among the edges internal to that merged cell. This is the
    rank signal a QC consumer wants: low min-IoU = the stitch least confident this
    merge was correct, so review those first. ``clean`` cells (1 tile, no seam edge)
    get ``min_match_iou = None`` — there is nothing for a seam to have gotten wrong.

    Returns ``{dense_id: {'n_tiles', 'seam_status', 'tiles': sorted[str],
    'constituents': sorted[(tile, local_label)], 'min_match_iou': float|None}}``.
    Background (dense 0) excluded.
    """
    from collections import defaultdict

    acc = defaultdict(lambda: {"tiles": set(), "constituents": []})
    for gid in present_ids:
        dense = dense_lut.get(int(gid), 0)
        if dense == 0:
            continue
        tile = gid_to_tile[int(gid)]
        local = int(gid) - int(offsets[tile])
        acc[dense]["tiles"].add(tile)
        acc[dense]["constituents"].append((tile, local))

    # Weakest internal seam-match IoU per dense cell. Every edge's endpoints landed
    # in the same union-find group, so both map to the SAME dense id — key by either.
    min_iou: Dict[int, float] = {}
    if edges:
        for a, _b, w in edges:
            dense = dense_lut.get(int(a), 0)
            if dense == 0:
                continue
            cur = min_iou.get(dense)
            if cur is None or float(w) < cur:
                min_iou[dense] = float(w)

    out: Dict[int, dict] = {}
    for dense, d in acc.items():
        n = len(d["tiles"])
        out[int(dense)] = {
            "n_tiles": n,
            "seam_status": classify_seam_status(n),
            "tiles": sorted(d["tiles"]),
            "constituents": sorted(d["constituents"]),
            "min_match_iou": min_iou.get(int(dense)),
        }
    return out


# ---------------------------------------------------------------------------
# 8) compute_tile_cores — flush-aware disjoint cores (midpoint rule)
# ---------------------------------------------------------------------------


def _midpoint(a0: int, a1: int, b0: int, b1: int) -> int:
    """Seam midpoint between intervals ``[a0,a1)`` and ``[b0,b1)`` (spec rule)."""
    return (max(a0, b0) + min(a1, b1)) // 2


def compute_tile_cores(
    geoms: Sequence[GeomLike],
    full_size: Tuple[int, int],
    halo: int = 0,
) -> Dict[str, Dict[str, Rect]]:
    """Flush-aware per-tile disjoint "core" so every output voxel has one owner.

    Midpoint rule: a tile owns each seam up to the midpoint with its neighbor.
    Edge tiles keep their outer extent (the image border has no neighbor). The
    cores are pairwise-DISJOINT and together COVER the full ``(H, W)``.

    Returns per tile ``{dst_core: (y0,y1,x0,x1) global, src_inset: (y0,y1,x0,x1) local}``.
    The dst_core maps to src_inset by the tile origin (``dst = src + origin``).

    ``halo`` is accepted for forward-compat (a future thin-halo retiling); with
    ``halo=0`` the cores tile the plane exactly. It is currently advisory only —
    the midpoint cut already guarantees disjoint+covering cores for the 25%-
    overlap reuse case (Q5).

    REQUIRES a REGULAR LATTICE (every tile in a grid row shares its y-span, every
    tile in a column shares its x-span) — which is exactly what ``compute_tile_grid``
    emits. The per-axis separable midpoint cut is only valid for such a lattice;
    an irregular geom list (rows with differing y-spans) would silently
    under-cover. Always pass geometry produced by the tiler.
    """
    H, W = full_size

    # Per-axis: sort the distinct origins along each axis, cut each interior seam
    # at the midpoint of the two adjacent tile spans. This is separable because
    # the grid is a regular (flush-anchored) lattice — every tile in a column
    # shares an x-span, every tile in a row shares a y-span.
    rects = {_geom_name(g): _geom_rect(g) for g in geoms}

    # Collect the distinct y-spans and x-spans (one per grid row / column).
    y_spans = sorted({(r[0], r[1]) for r in rects.values()})
    x_spans = sorted({(r[2], r[3]) for r in rects.values()})

    def _axis_cuts(spans: List[Tuple[int, int]], dim: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Map each tile span -> its disjoint owned [lo, hi) along this axis."""
        owned: Dict[Tuple[int, int], Tuple[int, int]] = {}
        n = len(spans)
        for k, (s0, s1) in enumerate(spans):
            lo = 0 if k == 0 else _midpoint(spans[k - 1][0], spans[k - 1][1], s0, s1)
            hi = dim if k == n - 1 else _midpoint(s0, s1, spans[k + 1][0], spans[k + 1][1])
            owned[(s0, s1)] = (lo, hi)
        return owned

    y_owned = _axis_cuts(y_spans, H)
    x_owned = _axis_cuts(x_spans, W)

    cores: Dict[str, Dict[str, Rect]] = {}
    for g in geoms:
        name = _geom_name(g)
        oy, ox = _geom_origin(g)
        gy0, gy1, gx0, gx1 = rects[name]
        dy0, dy1 = y_owned[(gy0, gy1)]
        dx0, dx1 = x_owned[(gx0, gx1)]
        cores[name] = {
            "dst_core": (dy0, dy1, dx0, dx1),
            "src_inset": (dy0 - oy, dy1 - oy, dx0 - ox, dx1 - ox),
        }
    return cores


# ---------------------------------------------------------------------------
# 9) apply_relabel_to_slice — vectorized LUT remap
# ---------------------------------------------------------------------------


def apply_relabel_to_slice(slice_2d: np.ndarray, lut_array: np.ndarray) -> np.ndarray:
    """Vectorized ``lut_array[slice_2d]`` remap of a 2D (or N-D) global-id slice.

    ``slice_2d`` holds GLOBAL ids (``local + offset``, background 0); ``lut_array``
    is a dense int array indexed by global id (see :func:`build_lut_array`).

    WRITER CONTRACT (important): background must already be 0 in ``slice_2d``.
    A caller mapping a tile's LOCAL labels to global ids MUST preserve background,
    e.g. ``global = np.where(local > 0, local + offset, 0)`` — a naive
    ``local + offset`` turns a background voxel (local 0) into ``offset``, a REAL
    prior-tile global id, which this LUT would then relabel as a foreground cell.
    """
    return lut_array[slice_2d]


# ---------------------------------------------------------------------------
# 10) build_lut_array — {global:dense} dict -> dense int32 array
# ---------------------------------------------------------------------------


def build_lut_array(
    offsets: Dict[str, int],
    dense_lut: Dict[int, int],
    max_global: int,
) -> np.ndarray:
    """Turn the ``{global_id: dense_id}`` dict into a dense int32 LUT array.

    The returned array is indexed by global id, so a writer can do
    ``lut[local + offset]`` for a fast per-slice remap. Index 0 (background) maps
    to 0. Includes a fail-fast int32 overflow guard on the dense count.

    ``offsets`` is accepted for signature symmetry with the writer call path (and
    to size the array correctly if a caller passes a sparse lut); ``max_global``
    is the largest global id that may appear in any slice.
    """
    K = max((v for v in dense_lut.values()), default=0)
    assert K < np.iinfo(np.int32).max, (
        f"dense label count {K} exceeds int32 ceiling; bump dtype"
    )

    size = int(max_global) + 1
    # Also accommodate any global key larger than max_global (defensive).
    if dense_lut:
        size = max(size, max(dense_lut) + 1)

    lut = np.zeros(size, dtype=np.int32)
    for g, d in dense_lut.items():
        if g == 0:
            continue
        lut[int(g)] = int(d)
    return lut
