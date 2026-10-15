"""The solve stage -- ported from the membrane sandbox's ``_solve``.

Three changes to the port, all required for correctness:

* the three ``print("[DIAG] ...")`` guard reports become
  :class:`~.notes.MessageSink` entries;
* ``random_walker`` is passed ``spacing=`` -- without it a soft cut is biased
  along the coarse axis;
* ``mode='cg_mg'`` is **forbidden**: ``pyamg`` is not installed in
  ``epicycle-env-py312`` and selecting it would raise at solve time, deep inside
  the worker.  The mode is pinned to :data:`~.spec.RANDOM_WALKER_MODE` and never
  exposed.

The two behaviour guards are ported verbatim: random walker with <2 seeds falls
back to watershed, and label diffusion that DROPS a cell keeps the watershed
split.
"""

from __future__ import annotations

import numpy as np

from .notes import MessageSink
from .optional_deps import import_diffuse_labels3D
from .preproc import norm
from .spec import RANDOM_WALKER_MODE


def solve(elev, seeds, region, solver="watershed", *, beta=130.0, compactness=0.0,
          spacing=(1, 1, 1), sink=None):
    """Assign each region voxel to a seed.

    - ``watershed``: floods the elevation -- the cut lands on the highest RIDGE
      between seeds, or the seed-midline (Voronoi) where the elevation is FLAT
      (a faint wall).  ``compactness`` > 0 regularizes basins toward equal
      territories -- the fix when a seed sitting on bright signal gets a starved
      sliver basin (0 = pure watershed).
    - ``random_walker``: diffuses each seed's label probability against the
      elevation gradients -- a SOFT cut that migrates toward whatever faint
      membrane exists, instead of snapping to the midline.  Needs >=2 seeds;
      else falls back to watershed.
    - ``label_diffusion``: watershed then u-Segment3D's ``diffuse_labels3D``
      refine -- GUARDED so the diffusion can't erase a whole cell (it would
      overrun a small basin on a faint inner wall); if a seed's label vanishes
      we keep the watershed split.
    """
    from skimage.segmentation import watershed

    sink = sink if sink is not None else MessageSink()

    def _ws(e):
        return watershed(e, markers=seeds, mask=region, compactness=float(compactness))

    if solver == "random_walker":
        n = len(set(int(v) for v in np.unique(seeds)) - {0})
        if n >= 2:
            try:
                from skimage.segmentation import random_walker
                markers = np.where(seeds > 0, seeds, 0).astype(np.int32)
                markers[~region] = -1                  # exclude outside the footprint
                rw = random_walker(
                    norm(elev), markers, beta=float(beta),
                    mode=RANDOM_WALKER_MODE,           # never 'cg_mg' (no pyamg)
                    spacing=tuple(float(s) for s in spacing),
                )
                return np.where(region, rw, 0).astype(np.int32)
            except Exception as exc:
                sink.emit(
                    "The soft cut failed ({!r}) -- used the sharp cut "
                    "instead.".format(exc)
                )
        else:
            sink.emit(
                "The soft cut needs at least 2 markers -- used the sharp "
                "cut instead."
            )

    elif solver == "label_diffusion":
        dense = _ws(elev)
        try:
            diffuse_labels3D = import_diffuse_labels3D()
        except ImportError as exc:
            sink.emit(str(exc))
            return dense
        try:
            guide = norm(elev).astype(np.float32)
            refined = diffuse_labels3D(dense.astype(np.int32), guide,
                                       clamp=0.5, n_iter=10, alpha=0.8,
                                       affinity_type="heat")
            out = np.where(region, np.asarray(refined), 0).astype(np.int32)
            dropped = ((set(int(v) for v in np.unique(dense)) - {0})
                       - set(int(v) for v in np.unique(out)))
            if dropped:
                sink.emit(
                    "Settling made object(s) {} disappear -- kept the sharp "
                    "cut instead.".format(sorted(dropped))
                )
                return dense
            return out
        except Exception as exc:
            sink.emit(
                "Settling failed ({!r}) -- used the sharp cut "
                "instead.".format(exc)
            )
            return dense

    return _ws(elev)
