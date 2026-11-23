"""An ordinary Cells box -- the seeded flood -- is FROZEN.

Recorded on the unchanged engine, BEFORE the region target learnt a second
way to grow cells.  Each case composes exactly what the Add tab's Cells
preset hands to ``reshape_empty_region`` at its default settings (the
membrane walls with the z-correct chain, ridge off, dark space as outside at
0.5 x the patch Otsu, a 2 um clearance, run markers eroded 1 um, click
balls of 2 um, the 4 M box budget), on a deterministic synthetic tissue
with kept objects in the box.  The answer -- the partition, the markers,
the elevation and every count the Add tab reads -- must stay byte for byte
the same through every later engine edit that keeps the flood as the
default.

Two "hard" cases add what an ordinary box rarely shows -- a run object too
thin for the erosion, a nucleus pressed against a kept object (a sliver),
and a thin kept sheet that cuts the cells' pieces off below it -- so the
drop counts are frozen too.  Every case also freezes ``find_region_seeds``'
two counts (the Add box-info line).

The guard also covers what an ordinary box rarely holds (added with the
growth engine, recorded on the same unchanged flood): a run id
at the top of int32, uint32 labels with a kept id near 4e9, one-voxel run
markers (in the tissue and in the dark), the core erosion at 0, an int64
nuclei-pieces layer with a one-voxel piece -- and one digest of the WHOLE
Cells Preview at its defaults through ``run_add_preview(preset="cells")``
(the final ids, the edge flags, the smoothing, the sizes), so the worker's
hand-over is frozen too.

An optional case runs the same recipe on a crop of a real Wound patch when
the driver copy of that data is on this machine (skipped otherwise); the
driver root is ``EPICYCLE_DERIVE_DRIVERS`` (default
``~/Desktop/EpiCycle-drivers``).

To re-record after a DELIBERATE change to the flood (never for an
additive change): ``python -X utf8 tests/annotator/test_region_flood_golden.py
--record`` from the repository root, with the repository on PYTHONPATH.
``--record-new`` records only the cases the fixture does not hold yet and
leaves every recorded entry byte for byte as it is.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


FIXTURE = Path(__file__).with_name("fixtures") / "region_flood_golden.json"

ISO = (1.0, 1.0, 1.0)
ANISO = (2.0, 1.0, 1.0)
FINE = (1.0, 0.5, 0.5)

#: The Cells preset's defaults as the Add tab hands them over.
ADD_CELLS_DEFAULTS = {
    "wall_kind": "membrane", "wall_chain": "zcorrect", "dark_outside": True,
    "dark_otsu_factor": 0.5, "dark_clearance_um": 2.0, "core_erode_um": 1.0,
    "click_radius_um": 2.0, "box_budget": 4_000_000,
}

#: The real-data case: a read-only driver copy of a Wound patch.
DRIVERS = Path(os.environ.get(
    "EPICYCLE_DERIVE_DRIVERS", Path.home() / "Desktop" / "EpiCycle-drivers"))
WOUND_ROOT = DRIVERS / "derive_d0" / "ws"


#: The Cells Preview's params as the Add tab hands them over at its
#: defaults (the panel's shared Add knobs at their defaults, the Cells
#: preset, run markers).
ADD_CELLS_PREVIEW_DEFAULTS = {
    "channel": "", "clahe": False, "clahe_clip": 0.01, "min_size": 0,
    "separate": True, "min_distance": 3, "clear_border": True,
    "edge_margin": 3, "auto_smooth": True, "smooth_sigma": 1.5,
    "preset": "cells", "click_radius_um": 2.0, "core_erode_um": 1.0,
    "dark_clearance_um": 2.0, "dark_otsu_factor": 0.5,
    "nuclei_threshold": 0.0, "dark_outside": True, "source": "labels",
    "wall_kind": "membrane", "wall_chain": "zcorrect",
    "box_budget": 4_000_000,
}


def cells_spec(seeds, *, dark_outside=True, core_erode_um=None):
    """The ``ReshapeSpec`` the Cells preset builds at its defaults
    (``core_erode_um`` overrides the one knob the edge cases move)."""
    from packages.epicycle_annotator.reshape import ReshapeSpec

    p = dict(ADD_CELLS_DEFAULTS)
    p["dark_outside"] = dark_outside
    if core_erode_um is not None:
        p["core_erode_um"] = float(core_erode_um)
    return ReshapeSpec(
        target="region", seeds=seeds, walls=p["wall_kind"], solver="watershed",
        params={
            "wall_channel": "membrane",
            "wall_preproc": p["wall_chain"],
            "ridge": False,
            "outside": "faces+dark" if p["dark_outside"] else "faces",
            "dark_otsu_factor": p["dark_otsu_factor"],
            "dark_clearance_um": p["dark_clearance_um"],
            "core_erode_um": p["core_erode_um"],
            "click_radius_um": p["click_radius_um"],
            "box_budget": p["box_budget"],
        })


# --------------------------------------------------------------------------- #
# the synthetic tissue
# --------------------------------------------------------------------------- #
def tissue(seed, hard=False):
    """A 26 x 60 x 60 two-channel tissue.

    Membrane walls on a jittered grid (cells ~10 voxels across) over
    ``y < 42``; open dark space beyond; a nucleus in most cells (a few cells
    have none); three kept objects, one straddling a wall and one in the
    dark.  ``hard`` adds, after everything else: a run object 2 voxels thick
    in a cell with no nucleus, a nucleus piece pressed against the kept cell
    (one voxel outside it), and a kept sheet 2 slices thick low in the
    tissue.  Returns ``(nuc, mem, labels, nuclei_run, pieces, clicks)``.
    """
    rng = np.random.default_rng(seed)
    shape = (26, 60, 60)
    z, y, x = np.indices(shape)
    tissue_mask = y < 42
    mem = rng.normal(0.06, 0.015, shape).astype(np.float32)
    nuc = rng.normal(0.05, 0.015, shape).astype(np.float32)
    nuc[tissue_mask] += 0.25
    walls_x = [0, 10, 21, 30, 41, 50, 59]
    walls_y = [0, 11, 20, 31, 41]
    for wx in walls_x[1:-1]:
        mem[:, :42, wx] += 0.9
    for wy in walls_y[1:]:
        mem[:, wy, :] += 0.9 if wy < 41 else 0.35       # a faint basal wall
    mem[:, 31, 20:31] -= 0.6                             # a gap in one wall
    nuclei_run = np.zeros(shape, np.int32)
    pieces = np.zeros(shape, np.int32)
    clicks = []
    nid = 7
    for j in range(len(walls_y) - 1):
        for i in range(len(walls_x) - 1):
            if (i + 2 * j) % 7 == 3:
                continue                                 # a cell with no nucleus
            cy = (walls_y[j] + walls_y[j + 1]) / 2.0 + rng.uniform(-1, 1)
            cx = (walls_x[i] + walls_x[i + 1]) / 2.0 + rng.uniform(-1, 1)
            cz = 13 + rng.uniform(-3, 3)
            r = 2.6 + 0.4 * ((i + j) % 3)
            blob = ((z - cz) / 1.4) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r * r
            nuc[blob] += 0.7
            nuclei_run[blob] = nid * 1000 + 3            # large, sparse run ids
            pieces[blob] = nid
            clicks.append((int(round(cz)), int(round(cy)), int(round(cx))))
            nid += 1
    # a stray nucleus in the dark (dropped by the dark rule)
    stray = (z - 13) ** 2 + (y - 52) ** 2 + (x - 30) ** 2 <= 6
    nuc[stray] += 0.12
    nuclei_run[stray] = 999_999
    labels = np.zeros(shape, np.int32)
    labels[4:22, 1:10, 1:10] = 5                         # a kept cell
    labels[8:18, 13:19, 18:24] = 41                      # straddles x = 21
    labels[10:14, 48:54, 8:14] = 77                      # in the dark
    if hard:
        # too thin for the 1 um erosion (2 voxels across y), on empty space
        nuclei_run[8:18, 15:17, 13:18] = 123_456
        # pressed against the kept cell 5 (x 1..9): one voxel at x = 10
        sliver = (slice(10, 16), slice(3, 8), slice(3, 11))
        nuc[sliver] += 0.7
        pieces[sliver] = nid
        # a kept sheet 2 slices thick: no core after the erosion, so the
        # cells grow under it and their pieces there are cut off
        sheet = np.zeros(shape, bool)
        sheet[3:5, :42, :] = True
        labels[sheet & (labels == 0)] = 900
    return (np.clip(nuc, 0, None), np.clip(mem, 0, None), labels, nuclei_run,
            pieces, clicks)


ROI = (slice(1, 25), slice(1, 59), slice(2, 58))


def _digest_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def digest(case):
    """Everything the Add tab reads from a region answer, hashable.

    ``case`` is the answer, or ``(answer, seed report)`` -- the report of
    ``find_region_seeds`` over the same box adds its two counts."""
    res, report = case if isinstance(case, tuple) else (case, None)
    r = np.asarray(res.result) if res.result is not None else None
    out = {
        "status": res.status,
        "refusal_reason": res.refusal_reason,
        "n_seeds": int(res.n_seeds or 0),
        "n_seeds_found": int(res.n_seeds_found or 0),
        "n_dropped_dark": int(getattr(res, "n_dropped_dark", 0) or 0),
        "n_dropped_thin": int(getattr(res, "n_dropped_thin", 0) or 0),
        "n_nuclei_slivers": int(getattr(res, "n_nuclei_slivers", 0) or 0),
        "n_markers_over_kept": int(getattr(res, "n_markers_over_kept", 0) or 0),
        "n_fragments_dropped": int(getattr(res, "n_fragments_dropped", 0) or 0),
        "unfilled": (None if res.unfilled_tissue_fraction is None
                     else repr(float(res.unfilled_tissue_fraction))),
        "faint_pairs": [[int(a), int(b), repr(float(m))]
                        for a, b, m in (res.faint_pairs or [])],
    }
    if report is not None:
        out["find_status"] = report.status
        out["find_n_found"] = int(report.n_found or 0)
        out["find_n_in_region"] = int(report.n_in_region or 0)
    if r is not None:
        out.update({
            "shape": [int(v) for v in r.shape], "dtype": str(r.dtype),
            "result": _digest_array(r),
            "ids": [int(v) for v in np.unique(r) if v],
            "seeds": _digest_array(np.asarray(res.seeds)),
            "elevation": _digest_array(np.asarray(res.elevation, np.float32)),
        })
    return out


def _synthetic(seeds, spacing, dark_outside=True, seed=0, hard=False):
    from packages.epicycle_annotator.add_preview import dark_stride
    from packages.epicycle_annotator.reshape import region as R

    nuc, mem, labels, run, pieces, clicks = tissue(seed, hard=hard)
    channels = {"nuc": nuc, "mem": mem}
    threshold = R.patch_dark_threshold(
        channels, spacing=spacing, stride=dark_stride(nuc.shape))
    starts = [sl.start for sl in ROI]
    seed_layer = seed_points = None
    if seeds == "labels":
        seed_layer = run
    elif seeds == "nuclei_masks":
        seed_layer = pieces
    else:
        seed_points = [tuple(c - s for c, s in zip(p, starts)) for p in clicks]
    spec = cells_spec(seeds, dark_outside=dark_outside)
    kw = dict(roi=ROI, dark_channels=(channels if dark_outside else None),
              seed_layer=seed_layer, seed_points=seed_points,
              dark_threshold=threshold, spacing=spacing)
    res = R.reshape_empty_region(labels, {"membrane": mem}, spec, **kw)
    return res, R.find_region_seeds(labels, {"membrane": mem}, spec, **kw)


def _edge(seeds, spacing, *, seed=0, hard=False, dtype=np.int32,
          huge_ids=False, one_voxel=False, core_erode_um=None):
    """The ordinary recipe on the edge cases a box rarely holds: the
    labels / seed layer in ``dtype``; ``huge_ids`` -- the first run object
    at the top of the dtype (int32: 2**31 - 1), a kept object near it;
    ``one_voxel`` -- four one-voxel markers, two in the tissue, two in the
    dark."""
    from packages.epicycle_annotator.add_preview import dark_stride
    from packages.epicycle_annotator.reshape import region as R

    nuc, mem, labels, run, pieces, _clicks = tissue(seed, hard=hard)
    top = {np.dtype(np.int32): 2 ** 31 - 1, np.dtype(np.uint32): 4_000_000_000,
           np.dtype(np.int64): 2 ** 40}[np.dtype(dtype)]
    labels = labels.astype(dtype)
    layer = (run if seeds == "labels" else pieces).astype(dtype)
    if huge_ids:
        ids = np.unique(layer[layer > 0])
        layer[layer == ids[0]] = top
        labels[labels == 41] = top - 100
    if one_voxel:
        for k, (zz, yy, xx) in enumerate(((13, 22, 32), (12, 33, 43),
                                          (13, 50, 20), (12, 55, 40))):
            assert labels[zz, yy, xx] == 0 and layer[zz, yy, xx] == 0
            layer[zz, yy, xx] = top - 10 - k
    channels = {"nuc": nuc, "mem": mem}
    threshold = R.patch_dark_threshold(
        channels, spacing=spacing, stride=dark_stride(nuc.shape))
    spec = cells_spec(seeds, core_erode_um=core_erode_um)
    kw = dict(roi=ROI, dark_channels=channels, seed_layer=layer,
              dark_threshold=threshold, spacing=spacing)
    res = R.reshape_empty_region(labels, {"membrane": mem}, spec, **kw)
    return res, R.find_region_seeds(labels, {"membrane": mem}, spec, **kw)


def preview_digest(out):
    """Everything the Add tab reads from a Cells Preview, hashable."""
    r = None if out.result is None else np.asarray(out.result)
    d = {
        "status": out.status, "refusal": out.refusal, "source": out.source,
        "ids": [int(v) for v in out.ids],
        "sizes": {str(int(k)): int(v) for k, v in sorted(out.sizes.items())},
        "edge_ids": sorted(int(v) for v in out.edge_ids),
        "small_ids": sorted(int(v) for v in out.small_ids),
        "counts": {k: int(getattr(out, k, 0) or 0) for k in (
            "n_dropped_dark", "n_dropped_thin", "n_nuclei_slivers",
            "n_markers_over_kept", "n_fragments_dropped", "n_below_min_size",
            "n_emptied_by_smooth")},
        "unfilled": (None if out.unfilled_tissue_fraction is None
                     else repr(float(out.unfilled_tissue_fraction))),
        "faint_pairs": [[int(a), int(b), repr(float(m))]
                        for a, b, m in (out.faint_pairs or [])],
    }
    if r is not None:
        d.update({"shape": [int(v) for v in r.shape], "dtype": str(r.dtype),
                  "result": _digest_array(r),
                  "elevation": _digest_array(np.asarray(out.elevation,
                                                        np.float32))})
    return d


def _preview(spacing=ISO, seed=0, hard=False):
    """The whole Cells Preview at its defaults, run markers, on the
    synthetic tissue's box."""
    from packages.epicycle_annotator import add_preview as AP
    from packages.epicycle_annotator.add_preview import dark_stride
    from packages.epicycle_annotator.reshape import region as R

    nuc, mem, labels, run, _pieces, _clicks = tissue(seed, hard=hard)
    channels = {"nuc": nuc, "mem": mem}
    threshold = R.patch_dark_threshold(
        channels, spacing=spacing, stride=dark_stride(nuc.shape))
    return AP.run_add_preview(
        preset="cells", roi=ROI, existing_roi=labels[ROI].copy(),
        next_label=int(labels.max()) + 1, spacing=spacing,
        params=dict(ADD_CELLS_PREVIEW_DEFAULTS), wall_roi=mem[ROI].copy(),
        dark_rois={k: v[ROI].copy() for k, v in channels.items()},
        run_guide_roi=run[ROI].copy(), dark_threshold=threshold,
        id_max=int(np.iinfo(np.int32).max))


def _wound(seeds="labels"):
    """A box of Wound patch 1 (both channels; channel 1 is the membrane),
    the nuclei run as markers, every second curated cell kept."""
    import tifffile
    from packages.epicycle_annotator.add_preview import dark_stride
    from packages.epicycle_annotator.reshape import region as R

    img = tifffile.imread(str(WOUND_ROOT / "patches" / "Wound_patch_1.tiff"))
    cells = tifffile.imread(str(WOUND_ROOT / "default" / "Wound_patch_1" / "labels_3d.tif"))
    nuclei = tifffile.imread(str(WOUND_ROOT / "nuclei" / "Wound_patch_1" / "labels_3d.tif"))
    spacing = (0.5, 0.5, 0.5)
    channels = {"ch0": img[0], "ch1": img[1]}
    threshold = R.patch_dark_threshold(
        channels, spacing=spacing, stride=dark_stride(img[0].shape))
    labels = np.where(cells % 2 == 1, cells, 0).astype(cells.dtype)
    roi = (slice(0, img.shape[1]), slice(40, 136), slice(96, 192))
    kw = dict(roi=roi, dark_channels=channels, seed_layer=nuclei,
              dark_threshold=threshold, spacing=spacing)
    res = R.reshape_empty_region(labels, {"membrane": img[1]}, cells_spec(seeds), **kw)
    return res, R.find_region_seeds(labels, {"membrane": img[1]}, cells_spec(seeds), **kw)


CASES = {
    "run_iso": lambda: _synthetic("labels", ISO),
    "run_aniso": lambda: _synthetic("labels", ANISO, seed=1),
    "clicks_iso": lambda: _synthetic("paint", ISO),
    "nuclei_pieces_aniso": lambda: _synthetic("nuclei_masks", ANISO, seed=2),
    "run_iso_faces_only": lambda: _synthetic("labels", ISO, dark_outside=False),
    "run_iso_hard": lambda: _synthetic("labels", ISO, seed=3, hard=True),
    "nuclei_pieces_fine_hard": lambda: _synthetic("nuclei_masks", FINE, seed=4,
                                                  hard=True),
    # the edge cases: huge ids, uint32, one-voxel markers, erode 0
    "edge_run_huge_ids_int32": lambda: _edge("labels", ISO, seed=5,
                                             huge_ids=True),
    "edge_run_uint32_huge_one_voxel": lambda: _edge(
        "labels", ANISO, seed=6, dtype=np.uint32, huge_ids=True,
        one_voxel=True),
    "edge_run_erode0_one_voxel": lambda: _edge(
        "labels", ISO, seed=7, one_voxel=True, core_erode_um=0.0),
    "edge_pieces_int64_one_voxel_hard": lambda: _edge(
        "nuclei_masks", FINE, seed=8, hard=True, dtype=np.int64,
        huge_ids=True, one_voxel=True),
}

#: Whole Cells Previews through ``run_add_preview`` at the defaults.
PREVIEWS = {
    "cells_defaults_iso": lambda: _preview(ISO),
    "cells_defaults_aniso_hard": lambda: _preview(ANISO, seed=3, hard=True),
}


def _wound_present():
    return all((WOUND_ROOT / p).is_file() for p in (
        "patches/Wound_patch_1.tiff", "default/Wound_patch_1/labels_3d.tif",
        "nuclei/Wound_patch_1/labels_3d.tif"))


def _recorded():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_flood_answer_is_frozen(name):
    assert digest(CASES[name]()) == _recorded()["synthetic"][name]


@pytest.mark.parametrize("name", sorted(PREVIEWS))
def test_the_cells_preview_at_defaults_is_frozen(name):
    assert preview_digest(PREVIEWS[name]()) == _recorded()["preview"][name]


def test_the_edge_cases_hold_what_they_say():
    """The edge cases are not vacuous: the huge ids reach the engine, the
    one-voxel markers survive erode 0 (two in the tissue grow; two in the
    dark are dropped), and the uint32 kept id is never written."""
    rec = _recorded()["synthetic"]
    zero = rec["edge_run_erode0_one_voxel"]
    plain = _synthetic("labels", ISO, seed=7)[0]
    assert zero["n_seeds_found"] == plain.n_seeds_found + 4
    assert zero["n_dropped_dark"] >= 2
    one = rec["edge_run_uint32_huge_one_voxel"]
    assert one["n_seeds_found"] >= 8 + 4
    for name in ("edge_run_huge_ids_int32", "edge_run_uint32_huge_one_voxel",
                 "edge_pieces_int64_one_voxel_hard"):
        assert rec[name]["status"] == "ok", name
        assert rec[name]["dtype"] == "int32", name
    res, _rep = CASES["edge_run_uint32_huge_one_voxel"]()
    assert res.result.max() <= res.n_seeds


def test_the_preview_fixture_is_not_trivial():
    rec = _recorded()["preview"]
    for name, d in rec.items():
        assert d["status"] == "ok", name
        assert len(d["ids"]) >= 8, name
        assert min(d["ids"]) > 77, name         # final ids, past the kept
        assert d["edge_ids"], name
        assert d["counts"]["n_dropped_dark"] >= 1, name


def test_the_flood_answer_on_a_real_patch_is_frozen():
    rec = _recorded().get("wound")
    if rec is None or not _wound_present():
        pytest.skip("the Wound patch driver copy is not on this machine")
    assert digest(_wound()) == rec


def test_the_fixture_is_not_trivial():
    """The golden must exercise kept objects, the dark drop, several cells
    and the unfilled share -- otherwise it protects nothing."""
    rec = _recorded()["synthetic"]
    for name, d in rec.items():
        assert d["status"] == "ok", name
        assert len(d["ids"]) >= 8, name
    assert rec["run_iso"]["n_dropped_dark"] >= 1
    assert rec["run_iso"]["unfilled"] is not None
    assert rec["run_iso_faces_only"]["unfilled"] is None
    assert rec["run_iso"]["result"] != rec["run_iso_faces_only"]["result"]
    # the hard cases exercise every drop count
    assert rec["run_iso_hard"]["n_dropped_thin"] >= 1
    assert rec["run_iso_hard"]["n_fragments_dropped"] >= 1
    assert rec["nuclei_pieces_fine_hard"]["n_nuclei_slivers"] >= 1
    assert rec["nuclei_pieces_fine_hard"]["n_fragments_dropped"] >= 1
    # the box-info counts agree with the Preview's
    for name, d in rec.items():
        assert d["find_status"] == "ok", name
        assert d["find_n_in_region"] == d["n_seeds"], name
        assert d["find_n_found"] == d["n_seeds_found"], name


def _record():  # pragma: no cover - the recorder
    from packages.epicycle_annotator.reshape import region as R

    here = Path(__file__).resolve().parents[2]
    assert Path(R.__file__).resolve().is_relative_to(here), R.__file__
    out = {"synthetic": {name: digest(fn()) for name, fn in sorted(CASES.items())},
           "preview": {name: preview_digest(fn())
                       for name, fn in sorted(PREVIEWS.items())}}
    if _wound_present():
        out["wound"] = digest(_wound())
    FIXTURE.parent.mkdir(exist_ok=True)
    with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: (v["result"] if "result" in v else v)
                      for k, v in out["synthetic"].items()}, indent=1))
    if "wound" in out:
        print("wound", out["wound"]["result"], len(out["wound"]["ids"]), "ids")


def _record_new():  # pragma: no cover - the additive recorder
    """Record only the cases the fixture does not hold yet; every recorded
    entry is left exactly as it is."""
    from packages.epicycle_annotator.reshape import region as R

    here = Path(__file__).resolve().parents[2]
    assert Path(R.__file__).resolve().is_relative_to(here), R.__file__
    out = _recorded()
    added = []
    for section, cases, fn_digest in (("synthetic", CASES, digest),
                                      ("preview", PREVIEWS, preview_digest)):
        held = out.setdefault(section, {})
        for name, fn in sorted(cases.items()):
            if name not in held:
                held[name] = fn_digest(fn())
                added.append("{}/{}".format(section, name))
    with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print("added", added)


if __name__ == "__main__":  # pragma: no cover
    if "--record-new" in sys.argv:
        _record_new()
    elif "--record" in sys.argv:
        _record()
