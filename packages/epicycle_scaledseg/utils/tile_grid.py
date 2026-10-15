"""
Pure tile-grid planner — flush-anchored, NO padding.

Replaces the old ``patchify`` + perfect-tiling-constraint approach. The previous
``ZStackTiler._compute_optimal_patch_and_step`` accepted a patch size ONLY if
``(dim - patch) % step == 0`` for both H and W — so on a realistic 7000px stack
it either crashed (``auto_correct=False`` -> ValueError) or silently mutated BOTH
your patch size and overlap (``auto_correct=True``). Neither is acceptable.

``compute_tile_grid`` instead tiles ANY ``(H, W)`` with the step grid and anchors
the LAST tile in each axis FLUSH to the boundary (``origin = dim - patch``). The
flush tile simply overlaps its neighbor a bit more — there is **zero padding**, so:
  * the segmentation model never sees a synthetic border, and
  * the assembled output is exactly the input ``(Z, Y, X)`` (no crop round-trip),
    so instances register 1:1 with the input stack for intensity-feature extraction.

This module is pure (no IO, no array slicing, no GPU/Qt) and is the explicit tile
geometry the assembly stage and the Mode-B IoU stitcher consume — the tile origin
is STORED, not regex-derived from the directory name (which is wrong for a flush
tile, whose true origin is ``dim - patch``, not ``grid_index * step``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass(frozen=True)
class TileGeom:
    """Geometry of one tile in the grid (pure data, no array)."""

    name: str
    origin: Tuple[int, int]      # (y0, x0) top-left pixel in the full image
    extent: Tuple[int, int]      # (h, w) — = patch, except a sub-patch single tile
    grid_index: Tuple[int, int]  # (i, j) row/col in the grid
    is_edge: bool                # touches the image border
    is_flush: bool               # the boundary-anchored last tile (origin != idx*step)


@dataclass(frozen=True)
class TileGrid:
    """The full tile grid + the parameters that produced it."""

    tiles: List[TileGeom]
    n_rows: int
    n_cols: int
    step: int
    patch: int
    overlap_px: int
    full_size: Tuple[int, int]   # (H, W)


def resolve_step(patch: int, overlap: Union[int, float]) -> Tuple[int, int]:
    """Return ``(step, overlap_px)`` from ``overlap`` given as a fraction or pixels.

    * ``0 <= overlap < 1`` -> FRACTION: ``step = int(patch * (1 - overlap))``
      (matches the legacy ``parse_tile_positions`` convention).
    * ``overlap >= 1`` (int) -> PIXELS: ``step = patch - overlap``.

    ``step`` is floored at 1 to avoid a degenerate zero/negative stride.
    """
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap < 1:
        step = int(patch * (1 - overlap))
    else:
        step = patch - int(overlap)
    step = max(step, 1)
    overlap_px = patch - step
    return step, overlap_px


def _axis_origins(dim: int, patch: int, step: int) -> List[int]:
    """Flush-anchored 1-D origin sequence covering ``[0, dim)`` with no padding.

    - ``dim <= patch`` -> ``[0]`` (single tile; its extent is ``dim``, sub-patch).
    - else step from 0 while ``origin + patch < dim``, then APPEND the flush
      anchor ``dim - patch`` if the last interior origin didn't already land there.
    """
    if dim <= patch:
        return [0]
    origins: List[int] = []
    o = 0
    while o + patch < dim:
        origins.append(o)
        o += step
    last = dim - patch
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


def compute_tile_grid(
    height: int,
    width: int,
    patch: int,
    overlap: Union[int, float],
) -> TileGrid:
    """Plan a flush-anchored, zero-padding tile grid over ``(height, width)``.

    Parameters
    ----------
    height, width : full image dimensions in pixels.
    patch : square tile size in pixels.
    overlap : fraction in ``[0, 1)`` OR pixel count (int ``>= 1``).

    Returns
    -------
    TileGrid with one :class:`TileGeom` per tile. Origins cover every pixel,
    never exceed the boundary, and never pad. The last tile in each axis is
    flush-anchored (``origin = dim - patch``), overlapping its neighbor more
    than ``overlap`` when ``(dim - patch)`` isn't a multiple of ``step``.
    """
    if patch <= 0:
        raise ValueError(f"patch must be > 0, got {patch}")
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be > 0, got {(height, width)}")

    step, overlap_px = resolve_step(patch, overlap)

    y_origins = _axis_origins(height, patch, step)
    x_origins = _axis_origins(width, patch, step)
    extent_y = min(patch, height)
    extent_x = min(patch, width)
    n_rows, n_cols = len(y_origins), len(x_origins)

    tiles: List[TileGeom] = []
    for i, y0 in enumerate(y_origins):
        y_flush = (y0 != i * step) and height > patch
        for j, x0 in enumerate(x_origins):
            x_flush = (x0 != j * step) and width > patch
            is_edge = (i == 0 or i == n_rows - 1 or j == 0 or j == n_cols - 1)
            tiles.append(TileGeom(
                name=f"tile_{i}_{j}",
                origin=(y0, x0),
                extent=(extent_y, extent_x),
                grid_index=(i, j),
                is_edge=is_edge,
                is_flush=(y_flush or x_flush),
            ))

    return TileGrid(
        tiles=tiles,
        n_rows=n_rows,
        n_cols=n_cols,
        step=step,
        patch=patch,
        overlap_px=overlap_px,
        full_size=(height, width),
    )
