"""
Mode-B ("full") full-resolution IoU seam-stitch — STAGE ORCHESTRATOR + WRITER.

This module DRIVES the verified pure algorithm core (``epicycle.core.iou_stitch``)
end-to-end and writes the stitched global-label volume to disk. It is the Mode-B
counterpart to :class:`~epicycle.core.assembly.AssemblyStage` (Mode A's lossy
"stitch-by-averaging + re-segment" path): here we keep the per-tile instance
labels, match cells across the tile-overlap seams by IoU, merge with union-find,
and emit a dense global relabel — at full resolution, with no second u-Segment3D
solve and no downsample.

The make-or-break engineering is the **RAM-bounded writer**: the full
global-label volume is never materialized in RAM. A ``(depth, H, W)`` int32 memmap
is allocated on disk and filled ONE Z-slice at a time; for each slice, each tile
contributes ONLY its page-z cropped to its disjoint "core" rect (one ``(H, W)``
output slice + one tile page in RAM at any moment). The same slice is streamed to
a BigTIFF (the plugin ingest contract) so the ``.npy`` memmap and the ``.tif`` are
byte-identical arrays.

The background-0 trap (documented on ``apply_relabel_to_slice``): a tile's local
labels map to global as ``np.where(local > 0, local + offset, 0)`` — a naive
``local + offset`` would turn a background voxel (local 0) into ``offset`` (a REAL
prior-tile global id). The writer honors this in :func:`compose_slice`.

The heavy IO + per-slice composition is factored so a pure piece is testable:
:func:`compose_slice` composites one slice given a ``read_page_fn`` callback, so
the writer logic can be exercised with synthetic in-memory readers (no disk).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult
from packages.epicycle_scaledseg.core import iou_stitch as S

Rect = Tuple[int, int, int, int]  # (y0, y1, x0, x1)

# Bumped when the on-disk result_manifest schema or output layout changes.
_RESULT_MANIFEST_VERSION = 1


# ---------------------------------------------------------------------------
# Small pure helpers (rect conversion + present-id gathering)
# ---------------------------------------------------------------------------


def global_to_local_rect(global_rect: Rect, origin: Tuple[int, int]) -> Rect:
    """Convert a GLOBAL ``(y0, y1, x0, x1)`` rect to a tile's LOCAL rect.

    ``local = global - origin`` componentwise. The inverse of the
    ``dst = src + origin`` mapping the cores use.
    """
    gy0, gy1, gx0, gx1 = global_rect
    oy, ox = int(origin[0]), int(origin[1])
    return (gy0 - oy, gy1 - oy, gx0 - ox, gx1 - ox)


def _read_tile_page(mask_path: str, z: int, src_inset: Rect) -> np.ndarray:
    """Read ONE Z-page of a tile mask, cropped to ``src_inset`` (RAM-bounded).

    Mirrors the page-read strategy in ``iou_stitch._read_overlap_block`` so a
    multi-GB full-Z tile never lands in RAM — at most one ``(h, w)`` page is
    decoded. Handles the tifffile "tiny contiguous volume collapses to one page"
    fallback (Z in {3,4}) the same way the core does.
    """
    import tifffile  # deferred heavy import

    y0, y1, x0, x1 = src_inset
    with tifffile.TiffFile(mask_path) as tf:
        series = tf.series[0]
        pages = series.pages
        if len(pages) > z:
            arr = pages[z].asarray()
            if arr.ndim == 3:
                # tifffile collapsed a tiny contiguous volume into a SINGLE page
                # whose array IS the whole (Z,Y,X) stack — index Z into it. (A
                # normal one-page-per-Z mask returns a 2D page.) Mirrors the
                # core's _read_overlap_block fallback.
                return arr[z, y0:y1, x0:x1]
            return arr[y0:y1, x0:x1]
        # page index past the (collapsed) page count: read the small series + slice.
        vol = series.asarray()
        return vol[z, y0:y1, x0:x1]


def _present_global_ids_for_tile(mask_path: str, offset: int) -> List[int]:
    """Global ids for the labels ACTUALLY present in a tile (no phantoms).

    Streams the mask page-by-page accumulating a set of present LOCAL labels
    (never the whole volume in RAM, never a single whole-volume ``np.unique``),
    then maps each to global as ``local + offset``. Using the present labels —
    not ``range(1, nmax+1)`` — avoids inventing singleton components for unused
    ids between 1 and nmax.
    """
    import tifffile  # deferred heavy import

    present: set = set()
    with tifffile.TiffFile(mask_path) as tf:
        series = tf.series[0]
        pages = series.pages
        if len(pages) > 0:
            for page in pages:
                arr = page.asarray()
                vals = np.unique(arr)
                present.update(int(v) for v in vals.tolist() if v != 0)
        else:  # pragma: no cover - defensive; a series always has >= 1 page
            arr = series.asarray()
            present.update(int(v) for v in np.unique(arr).tolist() if v != 0)
    return [ell + offset for ell in sorted(present)]


# ---------------------------------------------------------------------------
# Seam provenance CSV — the QC worklist (one row per final cell)
# ---------------------------------------------------------------------------

_PROVENANCE_COLUMNS = [
    "dense_id", "seam_status", "n_tiles", "min_match_iou", "voxel_count",
    "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
    "touches_border", "tiles", "constituents",
]


def _write_label_provenance(path, provenance: Dict[int, dict], stats: Dict[int, dict]) -> None:
    """Write per-cell QC provenance as CSV (one row per final dense label).

    Merges ``provenance`` (seam classification + ``min_match_iou`` rank signal, from
    compute_label_provenance) with ``stats`` (size / ZYX bbox / border, accumulated
    during the write) by dense label. This is the QC worklist a plugin reads to
    surface seam/corner cells (and border-clipped ones) for review: the ZYX bbox
    centers a review crop, ``min_match_iou`` ranks seam cells by stitch confidence,
    ``voxel_count`` drops fragments. All extents are in VOXELS. A row with
    ``voxel_count == 0`` is a cell that merged/clipped entirely out of the written
    volume (no owned-core voxels) — present for provenance, filter it out of a
    worklist. CSV = pandas-friendly, no new dep (swap to parquet via the manifest's
    ``label_provenance`` path without touching the reader).
    """
    import csv

    all_ids = set(provenance) | set(stats)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROVENANCE_COLUMNS)
        w.writeheader()
        for d in sorted(all_ids):
            p = provenance.get(d, {})
            s = stats.get(d, {})
            miou = p.get("min_match_iou")
            w.writerow({
                "dense_id": d,
                "seam_status": p.get("seam_status", "clean"),
                "n_tiles": p.get("n_tiles", 1),
                "min_match_iou": "" if miou is None else f"{float(miou):.6f}",
                "voxel_count": s.get("voxel_count", 0),
                "z_min": s.get("z_min", -1),
                "z_max": s.get("z_max", -1),
                "y_min": s.get("y_min", -1),
                "y_max": s.get("y_max", -1),
                "x_min": s.get("x_min", -1),
                "x_max": s.get("x_max", -1),
                "touches_border": int(bool(s.get("touches_border", False))),
                "tiles": ";".join(p.get("tiles", [])),
                "constituents": ";".join(f"{t}:{l}" for t, l in p.get("constituents", [])),
            })


# ---------------------------------------------------------------------------
# Pure per-slice compositor (testable with a synthetic read_page_fn)
# ---------------------------------------------------------------------------


def compose_slice(
    z: int,
    geoms_cores: Sequence[Tuple[str, Rect, Rect]],
    offsets: Dict[str, int],
    lut_array: np.ndarray,
    full_size: Tuple[int, int],
    read_page_fn: Callable[[str, int, Rect], np.ndarray],
) -> np.ndarray:
    """Composite ONE global-label slice from each tile's core. Pure given the reader.

    Parameters
    ----------
    z : Z index of the slice being composited.
    geoms_cores : ``[(tile_name, dst_core_global, src_inset_local), ...]`` — each
        tile's disjoint output rect (global) and the matching local source rect
        (from :func:`iou_stitch.compute_tile_cores`).
    offsets : ``{tile_name: global_offset}`` — local label ``L`` -> global ``L + offset``.
    lut_array : dense int32 LUT indexed by global id (see ``build_lut_array``);
        index 0 maps to 0.
    full_size : ``(H, W)`` of the output slice.
    read_page_fn : ``(tile_name, z, src_inset) -> (h, w) array`` of LOCAL labels.
        For the real writer this reads one tile page cropped to ``src_inset``; for
        tests it can be a synthetic in-memory function (no disk).

    Returns
    -------
    A ``(H, W)`` int32 slice. Cores are disjoint + covering, so every output voxel
    has exactly one tile owner. Background-0 is preserved through the local->global
    map (``np.where(local > 0, local + offset, 0)``) BEFORE the LUT remap — the
    documented relabel trap.
    """
    H, W = full_size
    out = np.zeros((H, W), dtype=np.int32)
    for name, dst_core, src_inset in geoms_cores:
        dy0, dy1, dx0, dx1 = dst_core
        if dy1 <= dy0 or dx1 <= dx0:
            continue  # degenerate (empty) core — nothing to write
        local_page = read_page_fn(name, z, src_inset)
        offset = int(offsets.get(name, 0))
        # BACKGROUND-0 PRESERVATION: only foreground voxels get the offset.
        glob = np.where(local_page > 0, local_page.astype(np.int64) + offset, 0)
        dense = S.apply_relabel_to_slice(glob, lut_array)
        out[dy0:dy1, dx0:dx1] = dense.astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class IoUStitchStage(Stage):
    """Mode-B ("full") full-resolution IoU seam-stitch stage.

    Reads per-tile instance masks (``<recon_root>/<tile>/<tile>_mask.tif``),
    matches cells across seam overlaps by IoU, union-finds the matches into global
    instances, and writes:

    * ``<output_dir>/global_labels.npy`` — a ``(depth, H, W)`` int32 memmap,
    * ``<output_dir>/global_labels.tif`` — the byte-identical BigTIFF (plugin ingest),
    * ``<output_dir>/result_manifest.json`` — run metadata.

    Parameters
    ----------
    recon_root : root holding ``<tile>/<tile>_mask.tif`` (defaults to
        ``<tile_dir>.parent/stage2_reconstruction`` if unset).
    output_dir : directory for the stitched outputs (defaults to
        ``<recon_root>.parent/stage4_full_stitch``).
    tiling_config_path : explicit path to ``tiling_config.json`` (else auto-found
        next to the tile dirs).
    tau, vmin, one_to_many, containment_rescue, z_block : seam-matcher params,
        forwarded verbatim to :func:`iou_stitch.overlap_iou_edges`.
    """

    def __init__(
        self,
        recon_root: Optional[str] = None,
        output_dir: Optional[str] = None,
        tiling_config_path: Optional[str] = None,
        tau: float = 0.25,
        vmin: int = 10,
        one_to_many: str = "max_iou",
        containment_rescue: bool = False,
        z_block: int = 64,
        emit_provenance: bool = True,
    ):
        super().__init__("iou_stitch")
        self.recon_root = str(Path(recon_root).resolve()) if recon_root else None
        self.output_dir = str(Path(output_dir).resolve()) if output_dir else None
        self.tiling_config_path = tiling_config_path
        self.tau = tau
        self.vmin = vmin
        self.one_to_many = one_to_many
        self.containment_rescue = containment_rescue
        self.z_block = z_block
        # When False, skip the seam-provenance QC artifact entirely (no per-cell
        # classification, no per-slice stats accumulation, no CSV) — drops the
        # ~third of the write loop the stats pass costs for callers who don't QC.
        self.emit_provenance = emit_provenance
        # Populated by _load_tiling_config.
        self.full_size: Optional[Tuple[int, int]] = None
        self.depth: Optional[int] = None
        self._geoms: Optional[List[dict]] = None

    def get_params(self) -> dict:
        return {
            "tau": self.tau,
            "vmin": self.vmin,
            "one_to_many": self.one_to_many,
            "containment_rescue": self.containment_rescue,
            "z_block": self.z_block,
            "emit_provenance": self.emit_provenance,
            "full_size": self.full_size,
            "depth": self.depth,
        }

    # -- geometry loading (mirrors AssemblyStage._load_tiling_config) ----------

    def _recon_root(self, tile_dir: Path) -> Path:
        if self.recon_root:
            return Path(self.recon_root)
        return tile_dir.parent / "stage2_reconstruction"

    def _load_tiling_config(self, tile_dirs: List[Path]) -> bool:
        """Load ``full_size``, ``depth``, and the ``tile_geometry`` list.

        Returns True if a config with ``tile_geometry`` was found + loaded.
        """
        config_path = None
        if self.tiling_config_path and Path(self.tiling_config_path).exists():
            config_path = Path(self.tiling_config_path)
        elif tile_dirs:
            candidate = tile_dirs[0].parent / "tiling_config.json"
            if candidate.exists():
                config_path = candidate

        if config_path is None:
            return False

        with open(config_path) as f:
            cfg = json.load(f)

        if self.full_size is None and "full_size" in cfg:
            self.full_size = (int(cfg["full_size"][0]), int(cfg["full_size"][1]))
        if self.depth is None and "depth" in cfg:
            self.depth = int(cfg["depth"])

        geom = cfg.get("tile_geometry")
        if geom:
            self._geoms = list(geom)
            return True
        return False

    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        """At least one tile must have a mask AND geometry must be loadable."""
        if not tile_dirs:
            return False
        self._load_tiling_config(tile_dirs)
        if not self._geoms or self.full_size is None or self.depth is None:
            return False
        recon_root = self._recon_root(tile_dirs[0])
        for td in tile_dirs:
            if (recon_root / td.name / f"{td.name}_mask.tif").exists():
                return True
        return False

    # -- run -------------------------------------------------------------------

    def _mask_path(self, recon_root: Path, name: str) -> Path:
        return recon_root / name / f"{name}_mask.tif"

    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        logger = logging.getLogger("epicycle.iou_stitch")
        result = StageResult()

        if not tile_dirs:
            result.add_failure("(none)", "no tile directories provided")
            return result

        # --- 1) geometry --------------------------------------------------------
        self._load_tiling_config(tile_dirs)
        missing = []
        if not self._geoms:
            missing.append("tile_geometry")
        if self.full_size is None:
            missing.append("full_size")
        if self.depth is None:
            missing.append("depth")
        if missing:
            msg = (
                f"Mode-B stitch needs tiling_config.json with {', '.join(missing)}; "
                f"none found next to the tiles. Provide tiling_config_path or ensure "
                f"tiling_config.json exists in the tiles directory."
            )
            logger.error(msg)
            for td in tile_dirs:
                result.add_failure(td.name, msg)
            return result

        recon_root = self._recon_root(tile_dirs[0])
        H, W = self.full_size
        depth = self.depth

        # Only keep geoms whose tile_dir is present + has a mask on disk.
        present_names = {td.name for td in tile_dirs}
        geoms = [g for g in self._geoms if g.get("tile_name") in present_names]
        geoms = [
            g for g in geoms
            if self._mask_path(recon_root, g["tile_name"]).exists()
        ]
        if not geoms:
            msg = "no per-tile masks found on disk for any geometry tile"
            logger.error(msg)
            for td in tile_dirs:
                result.add_failure(td.name, msg)
            return result

        names = [g["tile_name"] for g in geoms]
        mask_paths = {n: str(self._mask_path(recon_root, n)) for n in names}
        origins = {S._geom_name(g): S._geom_origin(g) for g in geoms}

        logger.info(
            f"Mode-B IoU stitch: {len(geoms)} tile(s), full_size=({H},{W}), "
            f"depth={depth}, tau={self.tau}, vmin={self.vmin}, "
            f"one_to_many={self.one_to_many}"
        )

        # --- 2) per-tile nmax + global offsets ----------------------------------
        nmax = {n: S.tile_nmax(mask_paths[n]) for n in names}
        offsets = S.compute_global_offsets(nmax)

        # --- 3) seam edges ------------------------------------------------------
        edges: List[Tuple[int, int, float]] = []
        for (nameA, nameB, global_rect) in S.enumerate_adjacent_pairs(geoms):
            rectA_local = global_to_local_rect(global_rect, origins[nameA])
            rectB_local = global_to_local_rect(global_rect, origins[nameB])
            edges += S.overlap_iou_edges(
                mask_paths[nameA], mask_paths[nameB],
                rectA_local, rectB_local,
                offsets[nameA], offsets[nameB], depth,
                tau=self.tau, vmin=self.vmin,
                one_to_many=self.one_to_many,
                containment_rescue=self.containment_rescue,
                z_block=self.z_block,
            )

        # --- 4) present-id universe -> union-find -> dense LUT -------------------
        present_ids: List[int] = []
        gid_to_tile: Dict[int, str] = {}  # which tile each per-tile global id came from
        for n in names:
            gids = _present_global_ids_for_tile(mask_paths[n], offsets[n])
            present_ids += gids
            for g in gids:
                gid_to_tile[g] = n
        roots = S.union_find_merge(present_ids, edges)
        dense_lut = S.compact_roots(roots)
        max_global = max(present_ids, default=0)
        lut_array = S.build_lut_array(offsets, dense_lut, max_global=max_global)
        n_global_labels = max((v for v in dense_lut.values()), default=0)

        # Per-cell seam provenance (the QC keystone): which final cells live in a
        # seam/corner region + the weakest seam-match IoU that built them. Pure
        # bookkeeping over the merge result, no volume read. Skipped wholesale when
        # emit_provenance is off (also skips the write-time stats pass below).
        provenance = (
            S.compute_label_provenance(
                present_ids, gid_to_tile, offsets, dense_lut, edges=edges
            )
            if self.emit_provenance
            else {}
        )

        # --- 5) cores -----------------------------------------------------------
        # Compute cores from the FULL lattice (self._geoms), NOT the present-only
        # subset: compute_tile_cores requires a complete regular lattice, and a
        # missing tile would otherwise let a survivor's core stretch past its own
        # extent (a crash). With the full lattice, each present tile gets its
        # correct core and a missing tile's region is simply left as background 0
        # (graceful degradation) since we only write the present tiles below.
        cores = S.compute_tile_cores(self._geoms, (H, W))
        geoms_cores: List[Tuple[str, Rect, Rect]] = [
            (n, cores[n]["dst_core"], cores[n]["src_inset"]) for n in names
        ]

        # --- 6/7) writer + manifest (manifest in a finally so a dead job leaves one)
        out_dir = Path(self.output_dir) if self.output_dir else recon_root.parent / "stage4_full_stitch"
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / "global_labels.npy"
        tif_path = out_dir / "global_labels.tif"
        manifest_path = out_dir / "result_manifest.json"
        provenance_path = out_dir / "label_provenance.csv"

        # Seam summary (counts per risk class) — available before the write.
        seam_summary = {"clean": 0, "seam": 0, "corner": 0}
        for p in provenance.values():
            seam_summary[p["seam_status"]] += 1
        n_border = {"value": 0}  # filled after the write (needs the FOV scan)

        def _write_manifest(status: str):
            manifest = {
                "version": _RESULT_MANIFEST_VERSION,
                "mode": "full",
                "status": status,
                "output_npy": str(npy_path),
                "output_tif": str(tif_path),
                "full_size": [int(H), int(W)],
                "depth": int(depth),
                "n_global_labels": int(n_global_labels),
                "n_tiles": int(len(geoms)),
                "tau": self.tau,
                "vmin": self.vmin,
                "one_to_many": self.one_to_many,
                "containment_rescue": self.containment_rescue,
                "z_block": self.z_block,
                "dtype": "int32",
                "emit_provenance": self.emit_provenance,
                # QC keystone: the seam-provenance worklist + how many cells fall in
                # problem regions. Null when emit_provenance is off (no artifact).
                # All provenance extents (z/y/x_min/max) are in VOXELS.
                "label_provenance": str(provenance_path) if self.emit_provenance else None,
                "provenance_columns": _PROVENANCE_COLUMNS if self.emit_provenance else None,
                "provenance_units": "voxels" if self.emit_provenance else None,
                "seam_summary": (
                    {**seam_summary, "border_clipped": n_border["value"]}
                    if self.emit_provenance else None
                ),
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, default=str)

        status = "incomplete"
        try:
            # Write a provisional manifest up-front so a long job that dies still
            # leaves SOME manifest on disk; rewrite it "complete" on success.
            _write_manifest("incomplete")

            stats = self._write_volume(
                npy_path, tif_path, geoms_cores, offsets, lut_array,
                (H, W), depth, mask_paths, origins, names, cores, logger,
                collect_stats=self.emit_provenance,
            )
            if self.emit_provenance:
                n_border["value"] = sum(1 for s in stats.values() if s.get("touches_border"))
                _write_label_provenance(provenance_path, provenance, stats)
            status = "complete"
            for td in tile_dirs:
                if td.name in present_names and td.name in names:
                    result.add_success(td.name)
                else:
                    result.add_success(td.name)  # absent-from-geom tiles aren't a stitch failure
        except Exception as e:  # pragma: no cover - exercised via failure paths
            logger.error(f"Mode-B stitch failed: {e}")
            for td in tile_dirs:
                result.add_failure(td.name, str(e))
        finally:
            _write_manifest(status)

        logger.info(
            f"Mode-B stitch {status}: {n_global_labels} global label(s) -> "
            f"{npy_path.name} + {tif_path.name}"
        )
        return result

    def _write_volume(
        self,
        npy_path: Path,
        tif_path: Path,
        geoms_cores: List[Tuple[str, Rect, Rect]],
        offsets: Dict[str, int],
        lut_array: np.ndarray,
        full_size: Tuple[int, int],
        depth: int,
        mask_paths: Dict[str, str],
        origins: Dict[str, Tuple[int, int]],
        names: List[str],
        cores: Dict[str, Dict[str, Rect]],
        logger,
        collect_stats: bool = True,
    ) -> Dict[int, dict]:
        """RAM-bounded writer: one (H,W) slice + one tile page at a time.

        Allocates a ``(depth, H, W)`` int32 memmap, fills it slice-by-slice via
        :func:`compose_slice`, and streams each identical slice to a BigTIFF so the
        memmap and TIFF are byte-identical. The per-tile page read is cropped to
        ``src_inset`` (never a whole patch page, never a whole volume).

        When ``collect_stats`` is True, returns per-dense-label stats accumulated
        during the write (one sort + 4 edge-slices per Z slice — RAM-bounded):
        ``{dense_id: {voxel_count, z_min, z_max, y_min, y_max, x_min, x_max,
        touches_border}}``. The ZYX bbox centers a QC review crop; ``touches_border``
        flags cells clipped by the image FOV (Y/X edges or the first/last Z slice) —
        a QC exclusion flag. All extents are in VOXELS. When ``collect_stats`` is
        False the accumulation is skipped entirely (returns ``{}``) and the write is
        pure composite-and-stream — the only thing the writes themselves are unchanged
        either way (the slice is written THEN read-only accumulated).
        """
        import tifffile  # deferred heavy import

        H, W = full_size

        # The pure compositor reads one tile page via this closure; it is the only
        # disk-touching surface and is bounded to one (h, w) page per call.
        def read_page_fn(tile_name: str, z: int, src_inset: Rect) -> np.ndarray:
            return _read_tile_page(mask_paths[tile_name], z, src_inset)

        stats: Dict[int, dict] = {}

        def _accumulate(slice_2d: np.ndarray, z: int) -> None:
            # One sort yields per-label voxel_count AND the YX bbox together: sort the
            # foreground voxels by label, then reduceat over the group boundaries for
            # min/max row+col. Replaces a bare np.unique (so the bbox is ~free) and is
            # the same order as the write (bounded to this one slice's foreground).
            ys, xs = np.nonzero(slice_2d)
            if ys.size:
                vals = slice_2d[ys, xs]
                order = np.argsort(vals, kind="stable")
                vals_s = vals[order]
                ys_s = ys[order]
                xs_s = xs[order]
                uniq, start, counts = np.unique(
                    vals_s, return_index=True, return_counts=True
                )
                ymin = np.minimum.reduceat(ys_s, start)
                ymax = np.maximum.reduceat(ys_s, start)
                xmin = np.minimum.reduceat(xs_s, start)
                xmax = np.maximum.reduceat(xs_s, start)
                for i in range(uniq.size):
                    v = int(uniq[i])  # uniq excludes 0 (np.nonzero dropped background)
                    st = stats.get(v)
                    if st is None:
                        stats[v] = {
                            "voxel_count": int(counts[i]),
                            "z_min": z, "z_max": z,
                            "y_min": int(ymin[i]), "y_max": int(ymax[i]),
                            "x_min": int(xmin[i]), "x_max": int(xmax[i]),
                            "touches_border": False,
                        }
                    else:
                        st["voxel_count"] += int(counts[i])
                        st["z_min"] = min(st["z_min"], z)
                        st["z_max"] = max(st["z_max"], z)
                        st["y_min"] = min(st["y_min"], int(ymin[i]))
                        st["y_max"] = max(st["y_max"], int(ymax[i]))
                        st["x_min"] = min(st["x_min"], int(xmin[i]))
                        st["x_max"] = max(st["x_max"], int(xmax[i]))
                present_vals = uniq
            else:
                present_vals = np.empty(0, dtype=slice_2d.dtype)
            # Border: a label on any Y/X edge of this slice, or anywhere on the
            # first/last Z slice, is clipped by the FOV.
            edge = np.concatenate([slice_2d[0, :], slice_2d[-1, :],
                                   slice_2d[:, 0], slice_2d[:, -1]])
            border = set(np.unique(edge).tolist())
            if z == 0 or z == depth - 1:
                border |= set(present_vals.tolist())
            for v in border:
                if v != 0 and v in stats:
                    stats[v]["touches_border"] = True

        out = np.lib.format.open_memmap(
            str(npy_path), mode="w+", dtype=np.int32, shape=(depth, H, W)
        )
        try:
            with tifffile.TiffWriter(str(tif_path), bigtiff=True) as tif:
                for z in range(depth):
                    slice_2d = compose_slice(
                        z, geoms_cores, offsets, lut_array, full_size, read_page_fn
                    )
                    out[z] = slice_2d
                    tif.write(slice_2d, contiguous=True)
                    if collect_stats:
                        _accumulate(slice_2d, z)
            out.flush()
        finally:
            # Release the memmap handle so the file is fully written + closable.
            del out
        return stats


__all__ = [
    "IoUStitchStage",
    "compose_slice",
    "global_to_local_rect",
]
