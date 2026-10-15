"""The arbiter -- the membrane decides whether a proposed cut is real.

Cross-modal counts are **evidence, not instructions**:

    Two seeds inside one target:
      - a WALL between them  -> two cells wrongly merged  -> split it.
      - NO wall between them -> one dividing cell -> leave it whole, flag it.

Never split on a count alone.  A batch tool without this rule over-splits every
mitosis in the volume.

Mechanically, after the solver runs and **before anything is written**:

1. Take the interface between each adjacent piece pair: the voxels of piece A
   that touch piece B, dilated with a **physical-radius** structuring element.
2. Read the wall elevation -- the same array the solver flooded -- on that
   interface and take the **median**.
3. Compare against a threshold derived from **the target's own footprint**,
   never a global constant:
   ``median(interface) >= percentile(elevation inside target, q)``.
4. Above -> a real cut, keep both pieces.  Below -> merge the pair back and
   mark the merged piece ``"dividing"``.
5. If every pair merges, the target comes out as one piece.  **That is a
   successful run, not a failure.**

Two properties of step 3 that the tests pin down:

* the rule is **scale-relative**, so a bright cell raises its own bar.  That is
  why ``zdepth_correct`` is mandatory on the wall channel whenever the arbiter
  is on (enforced in :class:`~.spec.ReshapeSpec`), and why the synthetic fixture
  includes a bright-cell case.
* the interface band is built in **physical units**.  One ``binary_dilation``
  step is one *voxel* per axis, so at 5x Z anisotropy a voxel-unit band is 5x
  thicker in Z and the median is dominated by whichever axis the interface
  happens to lie along.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .spec import ArbiterPair, ArbiterReport


def physical_ball(radius, spacing):
    """A boolean structuring element of PHYSICAL radius ``radius``.

    The half-width in each axis is ``round(radius / spacing[axis])``, floored at
    1 so the element always connects a voxel to its face neighbours.  On
    ``(5,1,1)`` spacing with ``radius=1.0`` this is 1 voxel in Z and 1 in Y/X --
    i.e. the same physical reach, not the same voxel reach.
    """
    sp = np.asarray(spacing, float)
    half = [max(1, int(round(float(radius) / float(s)))) for s in sp]
    grids = np.ogrid[tuple(slice(-h, h + 1) for h in half)]
    dist2 = sum(((g * s) ** 2) for g, s in zip(grids, sp))
    ball = dist2 <= (float(radius) ** 2) + 1e-9

    # ALWAYS include the 6 face neighbours.  The `half = max(1, ...)` floor
    # gives the element a voxel of room in every axis, but the `dist2 <= r2`
    # test then removes it again whenever spacing[axis] > radius: at (5,1,1)
    # with radius 1 um the z-neighbour sits 5 um away and is excluded, so the
    # element has ZERO z-extent.  Two pieces stacked in z then never appear to
    # touch, `adjacent_pairs` returns nothing, and a real interface is silently
    # unmeasured -- in exactly the anisotropic regime this tool is built for.
    # The band stays physical in SIZE; this only guarantees it is never empty
    # along an axis the pieces genuinely meet across.
    centre = tuple(h for h in half)
    faces = ndi.generate_binary_structure(len(sp), 1)
    slicer = tuple(slice(c - 1, c + 2) for c in centre)
    ball[slicer] |= faces
    return ball


def interface_mask(result, a, b, *, spacing=(1, 1, 1), radius=1.0):
    """Voxels of piece ``a`` that touch piece ``b``, plus the mirror set.

    The band is symmetric (A-side and B-side voxels both included) so the median
    is not biased by which piece happens to be the dilation source.
    """
    se = physical_ball(radius, spacing)
    mask_a = result == a
    mask_b = result == b
    a_touching_b = mask_a & ndi.binary_dilation(mask_b, structure=se)
    b_touching_a = mask_b & ndi.binary_dilation(mask_a, structure=se)
    return a_touching_b | b_touching_a


def adjacent_pairs(result, *, spacing=(1, 1, 1), radius=1.0):
    """Every unordered pair of piece ids whose interface band is non-empty."""
    ids = sorted(int(v) for v in np.unique(result) if v != 0)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if interface_mask(result, a, b, spacing=spacing, radius=radius).any():
                pairs.append((a, b))
    return pairs


def _union_find_merge(pairs, ids):
    """Group ids joined by ``pairs`` -> ``{id: representative}`` (smallest id wins)."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo
    return {i: find(i) for i in ids}


def arbitrate(result, elevation, region, *, spacing=(1, 1, 1), percentile=75.0,
              low_evidence_percentile=90.0, band_um=1.0, sink=None):
    """Apply the rule.  Returns ``(new_result, report, flags)``.

    ``flags`` maps a SURVIVING piece id to ``"dividing"`` (it absorbed a pair the
    arbiter refused to cut) or ``"low-evidence"`` (the cut was kept, but the wall
    was marginal).  ``new_result`` is ``result`` with every refused pair merged
    back, ids preserved (the smallest id of each merged group survives).
    """
    ids = sorted(int(v) for v in np.unique(result) if v != 0)
    report = ArbiterReport(enabled=True, percentile=float(percentile))

    if len(ids) < 2:
        report.verdict = "no-pairs"
        return result, report, {}

    inside = np.asarray(elevation, np.float32)[np.asarray(region, bool)]
    if inside.size == 0:
        report.verdict = "no-pairs"
        return result, report, {}

    threshold = float(np.percentile(inside, float(percentile)))
    low_threshold = float(np.percentile(inside, float(low_evidence_percentile)))
    report.threshold = threshold

    pairs = adjacent_pairs(result, spacing=spacing, radius=band_um)
    if not pairs:
        # Two or more pieces, but no measurable interface between any of them.
        # This is NOT an evidenced split: reporting "split" here would be
        # indistinguishable from a cut the membrane actually justified.
        report.verdict = "unarbitrated"
        if sink is not None:
            sink.emit(
                "The pieces do not share a measurable surface, so there is no "
                "wall evidence to judge this cut on -- the split was kept "
                "as the solver placed it, unarbitrated."
            )
        return result, report, {}

    merge_pairs = []
    marginal = []
    for a, b in pairs:
        band = interface_mask(result, a, b, spacing=spacing, radius=band_um)
        values = np.asarray(elevation, np.float32)[band]
        median = float(np.median(values)) if values.size else 0.0
        decision = "split" if median >= threshold else "merge"
        report.pairs.append(ArbiterPair(
            a=a, b=b, interface_voxels=int(band.sum()),
            interface_median=median, threshold=threshold,
            low_evidence_threshold=low_threshold, decision=decision,
        ))
        if decision == "merge":
            merge_pairs.append((a, b))
        elif median < low_threshold:
            marginal.append((a, b))

    flags: dict = {}
    if not merge_pairs:
        report.verdict = "split"
        for a, b in marginal:
            flags.setdefault(a, "low-evidence")
            flags.setdefault(b, "low-evidence")
        if sink is not None and marginal:
            sink.emit(
                "The wall signal between some pieces was faint -- the cut was "
                "kept but flagged as low evidence."
            )
        return result, report, flags

    report.merged_pairs = list(merge_pairs)
    mapping = _union_find_merge(merge_pairs, ids)
    merged = np.zeros_like(result)
    for old, new in mapping.items():
        merged[result == old] = new
    survivors = sorted(set(mapping.values()))
    merged_survivors = {mapping[a] for a, _ in merge_pairs}
    for pid in survivors:
        if pid in merged_survivors:
            flags[pid] = "dividing"
    for a, b in marginal:
        for pid in (mapping[a], mapping[b]):
            flags.setdefault(pid, "low-evidence")

    report.verdict = "dividing" if len(survivors) == 1 else "split"
    if sink is not None:
        if len(survivors) == 1:
            sink.emit(
                "Kept whole -- no wall between the markers (possibly "
                "dividing)."
            )
        else:
            sink.emit(
                "{} proposed cut(s) had no wall behind them and were merged "
                "back (possibly dividing objects).".format(len(merge_pairs))
            )
    return merged, report, flags
