"""
Shared pytest fixtures for the EpiCycle ScaledSegmentation test suite.

This module is the single source of synthetic tile data for the whole suite.
The fixtures here mirror the engine's REAL on-disk contract so that tests built
on them exercise the same shapes/dtypes/geometry the production pipeline emits:

  * per-tile mask  -> ``<tile>/<tile>_mask.tif``  : uint32, (Z, Y, X), BigTIFF,
    zero-padded to the full original Z (see ``recover_original_dimensions`` in
    ``epicycle/utils/array_ops.py`` and the write site at
    ``epicycle/core/reconstruction.py:900-905``).
  * per-tile config -> ``<tile>/<tile>_config.json`` : the Stage-0 tiling schema
    (``epicycle/utils/tiling.py:212-221``) — keys ``tile_name, row, col,
    truncated_range, original_z, patch_size, step_size, overlap``.
  * tile pixel origin -> ``(row * step, col * step)`` with
    ``step = int(patch * (1 - overlap))`` — matching the regex-based position
    recovery in ``epicycle/core/assembly.py:parse_tile_positions`` (``:28-60``).

Two fixtures are exported:

  * ``make_tile_mask``        — a factory to write one tile (arbitrary labels).
  * ``two_overlapping_tiles`` — two X-adjacent tiles sharing a seam, with a
    "straddle" cell present in BOTH tiles' overlap zone (the canonical Mode-B
    IoU-stitch scenario) plus two distinct singletons outside the overlap.

Fixtures are intentionally Qt/GPU/napari-free and write only under ``tmp_path``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pytest
import tifffile


# ---------------------------------------------------------------------------
# Geometry + on-disk handle for a synthetic per-tile mask
# ---------------------------------------------------------------------------


@dataclass
class TileSpec:
    """Everything a test needs about one synthetic tile, without re-deriving it.

    Attributes mirror both the in-memory array that was written and the
    on-disk artifacts, plus the pixel geometry that ``parse_tile_positions``
    would reconstruct from the tile's directory name.
    """

    name: str
    row: int
    col: int
    y0: int  # pixel origin row = row * step_size
    x0: int  # pixel origin col = col * step_size
    patch_size: int
    step_size: int
    overlap: float
    full_z: int
    z_range: Tuple[int, int]  # truncated_range [z_start, z_end)
    dir: Path
    mask_path: Path
    config_path: Path
    array: np.ndarray  # the (Z, Y, X) uint32 volume actually written to disk

    @property
    def position(self) -> Tuple[int, int, Path]:
        """``(y_pixel, x_pixel, tile_dir)`` — the tuple assembly consumes.

        NOTE on wiring into ``stitch_mask_volume``: assembly uses only
        ``tile_dir.name`` and re-prepends its ``recon_root`` argument
        (``assembly.py:119`` builds ``recon_root / td.name / f"{td.name}_mask.tif"``).
        Since this fixture writes to ``<base>/<name>/<name>_mask.tif``, a test
        must pass ``recon_root = spec.dir.parent`` (the fixture's ``base``), NOT
        ``tmp_path``. Getting it wrong yields a silently-skipped tile (assembly
        logs "missing mask" and continues), not a hard error.
        """
        return (self.y0, self.x0, self.dir)


def _write_tile(
    base_dir: Path,
    name: str,
    row: int,
    col: int,
    array: np.ndarray,
    *,
    patch_size: int,
    overlap: float,
    full_z: int,
    z_range: Tuple[int, int],
    config_overrides: Optional[Dict] = None,
    origin: Optional[Tuple[int, int]] = None,
) -> TileSpec:
    """Write ``<base>/<name>/<name>_mask.tif`` + ``_config.json`` and return a TileSpec.

    The mask is written whole-volume as a BigTIFF uint32 ``(Z, Y, X)`` array,
    exactly as the Stage-2 reconstruction worker does.

    The config JSON carries the **Stage-0 tiling/geometry** schema
    (``epicycle/utils/tiling.py:212-221``): ``{tile_name, row, col,
    truncated_range, original_z, patch_size, step_size, overlap}``. This is the
    config that assembly's position recovery and the Mode-B IoU stitcher read
    for tile GEOMETRY (origin/patch/overlap).

    NOTE — there are TWO real per-tile config schemas in the engine and they
    live in different directories:
      * Stage-0 tiling config (this one) — geometry, in the tile dir.
      * Stage-1 segmentation config — written by ``update_tile_config`` into the
        stage1 output dir (``segmentation.py:605-612``) with
        ``original_dimensions: {"z": ...}`` + ``truncated_range`` + seg params.
        The Stage-2 reconstruction worker reads THIS one
        (``reconstruction.py:710-711`` hard-subscripts
        ``meta["original_dimensions"]["z"]``).
    So a test that feeds this fixture's config to a reconstruction-style reader
    needs the ``original_dimensions`` key. Use ``config_overrides`` to add it,
    e.g. ``config_overrides={"original_dimensions": {"z": full_z}}``.
    """
    step_size = int(patch_size * (1 - overlap))
    tile_dir = Path(base_dir) / name
    tile_dir.mkdir(parents=True, exist_ok=True)

    arr = array.astype(np.uint32)
    mask_path = tile_dir / f"{name}_mask.tif"
    # Mirror the Stage-2 write site EXACTLY (reconstruction.py:901-905): a single
    # imwrite of the whole (Z, Y, X) volume as BigTIFF. Writing page-by-page with
    # TiffWriter instead produces a multi-series file that imread reads back as a
    # lone 2D page — not the engine's contract.
    tifffile.imwrite(str(mask_path), arr, bigtiff=True)

    config = {
        "tile_name": name,
        "row": int(row),
        "col": int(col),
        "truncated_range": [int(z_range[0]), int(z_range[1])],
        "original_z": int(full_z),
        "patch_size": int(patch_size),
        "step_size": int(step_size),
        "overlap": float(overlap),
    }
    if config_overrides:
        config.update(config_overrides)
    config_path = tile_dir / f"{name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Origin: explicit override (e.g. a flush-anchored compute_tile_grid origin)
    # or the legacy row*step / col*step regular-grid position.
    y0, x0 = origin if origin is not None else (row * step_size, col * step_size)

    return TileSpec(
        name=name,
        row=row,
        col=col,
        y0=y0,
        x0=x0,
        patch_size=patch_size,
        step_size=step_size,
        overlap=overlap,
        full_z=full_z,
        z_range=(int(z_range[0]), int(z_range[1])),
        dir=tile_dir,
        mask_path=mask_path,
        config_path=config_path,
        array=arr,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_tile_mask(tmp_path):
    """Factory fixture: write one synthetic per-tile mask + config.

    Usage::

        spec = make_tile_mask(
            row=0, col=0,
            labels={1: (slice(2, 6), slice(2, 6)),
                    3: (slice(4, 12), slice(10, 16))},
        )

    ``labels`` maps ``label_id -> region``. A region is EITHER:

      * a 2-tuple ``(y_slice, x_slice)`` — painted across the full truncated Z
        range ``z_range`` (the original, unchanged behavior); OR
      * a 3-tuple ``(z_slice, y_slice, x_slice)`` — painted with an EXPLICIT
        per-label Z extent (Stack-6 extension; unlocks the Z-band seam-mismatch
        + corner-only fixtures, where two tiles must disagree on a cell's Z span).

    The 3-tuple Z slice is absolute (in full-Z coordinates), not relative to
    ``z_range`` — e.g. ``(slice(10, 60), ys, xs)`` paints z∈[10, 60). It is the
    caller's job to keep it within ``[0, full_z)``; a slice is clamped by numpy
    indexing if it runs past the array, but no warning is emitted.

    Defaults give a single label ``1`` in a 4x4 block on a 16x16 patch,
    zero-padded to full_z=16 with a truncated range of [4, 12).

    ``config_overrides`` (optional dict) is merged into the written
    ``_config.json`` after the Stage-0 geometry keys. Use it to emit the
    reconstruction-input flavor a Stage-2 reader expects, e.g.
    ``config_overrides={"original_dimensions": {"z": full_z}}`` — see
    ``_write_tile`` for the two-schema explanation.
    """

    def _factory(
        *,
        row: int = 0,
        col: int = 0,
        patch_size: int = 16,
        overlap: float = 0.25,
        full_z: int = 16,
        z_range: Tuple[int, int] = (4, 12),
        labels: Optional[
            Dict[int, Union[Tuple[slice, slice], Tuple[slice, slice, slice]]]
        ] = None,
        base_dir: Optional[Path] = None,
        name: Optional[str] = None,
        config_overrides: Optional[Dict] = None,
        origin: Optional[Tuple[int, int]] = None,
    ) -> TileSpec:
        if labels is None:
            labels = {1: (slice(2, 6), slice(2, 6))}
        z0, z1 = z_range
        arr = np.zeros((full_z, patch_size, patch_size), dtype=np.uint32)
        for lab, region in labels.items():
            if len(region) == 3:
                zs, ys, xs = region  # explicit per-label Z extent
                arr[zs, ys, xs] = lab
            else:
                ys, xs = region  # paint across the whole truncated z_range
                arr[z0:z1, ys, xs] = lab
        nm = name or f"tile_{row}_{col}"
        base = Path(base_dir) if base_dir is not None else tmp_path
        return _write_tile(
            base,
            nm,
            row,
            col,
            arr,
            patch_size=patch_size,
            overlap=overlap,
            full_z=full_z,
            z_range=z_range,
            config_overrides=config_overrides,
            origin=origin,
        )

    return _factory


@pytest.fixture
def two_overlapping_tiles(tmp_path, make_tile_mask):
    """Two X-adjacent tiles (``tile_0_0``, ``tile_0_1``) sharing a seam.

    Geometry (self-consistent with ``parse_tile_positions``)::

        patch = 16, overlap = 0.25  -> step = 12, overlap_px = 4
        tile_0_0 spans global X [0, 16)   origin x0 = 0
        tile_0_1 spans global X [12, 28)  origin x0 = 12
        overlap zone = global X [12, 16)  (4 px wide, full Y, full Z range)

    Cells::

        label 3 (the straddle cell)  — present in BOTH tiles' overlap zone, i.e.
            the SAME physical voxels in global coords -> perfect co-occurrence,
            so an IoU seam-matcher must fuse it into one global instance.
        label 1 — only in tile_0_0, outside the overlap (global X [2, 6)).
        label 2 — only in tile_0_1, outside the overlap (global X [22, 26)).

    Returns ``(tile0_spec, tile1_spec)`` as :class:`TileSpec` instances. Both
    share full_z=16 and truncated_range [4, 12) so the straddle cell co-occurs
    in Z as well as Y/X.
    """
    patch = 16
    overlap = 0.25
    full_z = 16
    z_range = (4, 12)
    base = tmp_path / "tiles"

    # tile_0_0: origin x0=0. Overlap zone global X[12,16) == local x[12,16).
    # straddle label 3 spans local x[10,16) -> covers the overlap.
    t0 = make_tile_mask(
        row=0,
        col=0,
        patch_size=patch,
        overlap=overlap,
        full_z=full_z,
        z_range=z_range,
        base_dir=base,
        labels={
            1: (slice(4, 12), slice(2, 6)),
            3: (slice(4, 12), slice(10, 16)),
        },
    )

    # tile_0_1: origin x0=12. Overlap zone global X[12,16) == local x[0,4).
    # straddle label 3 spans local x[0,6) -> covers the overlap (same global voxels).
    t1 = make_tile_mask(
        row=0,
        col=1,
        patch_size=patch,
        overlap=overlap,
        full_z=full_z,
        z_range=z_range,
        base_dir=base,
        labels={
            2: (slice(4, 12), slice(10, 14)),
            3: (slice(4, 12), slice(0, 6)),
        },
    )

    return t0, t1


@pytest.fixture
def four_corner_tiles(tmp_path, make_tile_mask):
    """A 2×2 grid of tiles sharing ONE common global corner square.

    Geometry (flush-anchored ``compute_tile_grid``-style, explicit origins)::

        patch = 16, step = 12, overlap_px = 4
        tile_0_0 origin (0,  0)   spans global Y[0,16)  X[0,16)
        tile_0_1 origin (0, 12)   spans global Y[0,16)  X[12,28)
        tile_1_0 origin (12, 0)   spans global Y[12,28) X[0,16)
        tile_1_1 origin (12,12)   spans global Y[12,28) X[12,28)

    The 4-way corner square is global ``Y[12,16) × X[12,16)`` (4×4) — present in
    ALL FOUR tiles. In each tile it maps to a different local quadrant::

        tile_0_0: local y[12,16) x[12,16)   (bottom-right)
        tile_0_1: local y[12,16) x[ 0, 4)   (bottom-left)
        tile_1_0: local y[ 0, 4) x[12,16)   (top-right)
        tile_1_1: local y[ 0, 4) x[ 0, 4)   (top-left)

    Every tile paints the SAME global label-3 voxels in that corner square, so a
    cell present ONLY in the corner (absent from every 2-way edge slab) must still
    resolve to ONE global instance across all four tiles. Each tile also carries a
    private singleton well away from any seam.

    Returns ``[t00, t01, t10, t11]`` (:class:`TileSpec`), all full_z=16,
    z_range=[4,12), with the corner cell labelled ``3`` in every tile.
    """
    patch = 16
    overlap = 0.25
    full_z = 16
    z_range = (4, 12)
    base = tmp_path / "tiles"

    # Per-tile: (origin, local-corner-y-slice, local-corner-x-slice, private singleton)
    # The corner cell (label 3) is the SAME global voxels in all four tiles.
    # Private singletons are placed in each tile's seam-free interior quadrant
    # (the corner AWAY from the shared corner) so they never co-occur with a seam.
    specs = []
    layout = [
        # (row, col, origin, corner_ys, corner_xs, singleton_lab, sing_ys, sing_xs)
        (0, 0, (0, 0),   slice(12, 16), slice(12, 16), 11, slice(2, 5), slice(2, 5)),
        (0, 1, (0, 12),  slice(12, 16), slice(0, 4),   12, slice(2, 5), slice(11, 14)),
        (1, 0, (12, 0),  slice(0, 4),   slice(12, 16), 13, slice(11, 14), slice(2, 5)),
        (1, 1, (12, 12), slice(0, 4),   slice(0, 4),   14, slice(11, 14), slice(11, 14)),
    ]
    for (row, col, origin, cys, cxs, slab, sys_, sxs) in layout:
        specs.append(
            make_tile_mask(
                row=row,
                col=col,
                patch_size=patch,
                overlap=overlap,
                full_z=full_z,
                z_range=z_range,
                base_dir=base,
                origin=origin,
                labels={
                    3: (slice(4, 12), cys, cxs),   # the shared corner-only cell
                    slab: (sys_, sxs),             # private singleton, seam-free
                },
            )
        )
    return specs
