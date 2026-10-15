"""S1 scaling — the contact-graph family off the memmap.

Two things to prove, both gates from the plan:

1. **Byte-identical.** Routing the graph family through per-cell halo crops on a
   memmap with a precomputed ``slices`` bbox list must produce EXACTLY the same
   feature values + the same ``_adjacency_list`` as the original resident-array +
   internal-``find_objects`` path. Proven at the function level (slices injected
   vs not) AND at the ``extract_label_features`` level (PATH/memmap branch vs
   in-memory-array branch).

2. **Actually scales.** On the adjacency-ONLY path with a PATH input, the graph
   functions must receive a ``np.memmap`` (crops stream from disk — peak RSS
   tracks crop x workers, not volume size), NOT a resident full-volume copy; and
   they must receive the precomputed ``slices`` (the redundant full-volume
   ``find_objects`` + int32 copy is skipped). When a truly-global pass
   (``tissue_depth``) ALSO runs, the resident volume it loaded is REUSED for the
   graph family (no second memmap), still with ``slices`` injected.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import find_objects

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.reconcile.features import extract_label_features


def _multi_touch_labels():
    """A 3x3 grid of touching cubes (1-voxel gaps -> adjacency at radius 1) in one
    z-slab + one isolated cube. Gives a real graph (interior cell has 4 neighbours)
    so the contact-geometry / axis-persistence passes have non-trivial work."""
    vol = np.zeros((10, 40, 40), np.int32)
    lab = 1
    for yi in range(3):
        for xi in range(3):
            y0 = 2 + yi * 6
            x0 = 2 + xi * 6
            vol[2:8, y0:y0 + 5, x0:x0 + 5] = lab   # 5-wide on a 6 stride -> 1-vox gap
            lab += 1
    vol[2:8, 30:36, 30:36] = lab                    # isolated (gap >> max_radius)
    return vol


# --------------------------------------------------------------------------- #
# 1. Byte-identical — function level (slices injected vs internal find_objects) #
# --------------------------------------------------------------------------- #
def test_neighbor_features_slices_injection_equivalent(tmp_path):
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    base = extract_label_features(str(p), n_workers=1)   # labels-only table
    base["label"] = base["label"].astype(np.int64)
    objs = find_objects(vol)

    # Reference: resident array + slices=None (the original internal scan).
    df_ref = base.copy()
    qf.compute_neighbor_features(vol, df_ref, max_radius=5, slices=None)

    # New: memmap + slices=objs (per-cell crops off disk, no redundant scan).
    mm = np.load(str(p), mmap_mode="r")
    df_new = base.copy()
    qf.compute_neighbor_features(mm, df_new, max_radius=5, slices=objs)

    pd.testing.assert_frame_equal(df_ref, df_new)
    assert df_ref.attrs["_adjacency_list"] == df_new.attrs["_adjacency_list"]
    assert df_ref.attrs["_adjacency_list"]   # non-empty (the grid really touches)


def test_contact_and_axis_slices_injection_equivalent(tmp_path):
    """The two OTHER volume-reading graph functions (contact geometry + axis
    persistence) ride on the same _adjacency_list — inject slices into the whole
    chain and assert the full assembled frame is identical to the resident path."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    base = extract_label_features(str(p), n_workers=1)
    base["label"] = base["label"].astype(np.int64)
    objs = find_objects(vol)
    mm = np.load(str(p), mmap_mode="r")

    def _run(src, slices):
        df = base.copy()
        qf.compute_neighbor_features(src, df, max_radius=5, slices=slices)
        qf.compute_contact_geometry(src, df, spacing=None, max_radius=5, slices=slices)
        qf.compute_axis_neighbor_persistence(
            src, df, spacing=None, max_radius=5, slices=slices)
        return df

    df_ref = _run(vol, None)     # resident + internal find_objects
    df_new = _run(mm, objs)      # memmap + injected slices
    pd.testing.assert_frame_equal(df_ref, df_new)


# --------------------------------------------------------------------------- #
# 1b. Byte-identical — extract_label_features (memmap branch vs array branch)  #
# --------------------------------------------------------------------------- #
def test_extract_path_vs_array_byte_identical(tmp_path):
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)

    df_path = extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)
    df_arr = extract_label_features(vol, n_workers=1, adjacency=True, max_radius=5)

    pd.testing.assert_frame_equal(
        df_path.reset_index(drop=True),
        df_arr.reset_index(drop=True),
        check_dtype=False,
    )
    assert df_path.attrs.get("_adjacency_list") == df_arr.attrs.get("_adjacency_list")
    assert df_path.attrs.get("_adjacency_list")   # the grid touches -> non-empty


def test_extract_with_spacing_path_vs_array_identical(tmp_path):
    """Same equivalence with anisotropic spacing (the contact-geometry physical
    members + axis persistence use spacing) — proves the memmap path doesn't
    perturb the physical-frame columns."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    sp = (2.0, 0.5, 0.5)

    df_path = extract_label_features(
        str(p), n_workers=1, adjacency=True, max_radius=5, spacing=sp)
    df_arr = extract_label_features(
        vol, n_workers=1, adjacency=True, max_radius=5, spacing=sp)
    pd.testing.assert_frame_equal(
        df_path.reset_index(drop=True),
        df_arr.reset_index(drop=True),
        check_dtype=False,
    )


# --------------------------------------------------------------------------- #
# 2. Actually scales — the adjacency-only path uses a memmap, not a copy        #
# --------------------------------------------------------------------------- #
def _spy_neighbor(monkeypatch, captured):
    """Patch qf.compute_neighbor_features (the symbol extract_label_features
    re-imports at call time) to record how it was invoked, then delegate."""
    orig = qf.compute_neighbor_features

    def spy(labels_3d, df, **kw):
        captured["is_memmap"] = isinstance(labels_3d, np.memmap)
        captured["slices_given"] = kw.get("slices") is not None
        return orig(labels_3d, df, **kw)

    monkeypatch.setattr(qf, "compute_neighbor_features", spy)


def test_adjacency_only_path_input_uses_memmap(tmp_path, monkeypatch):
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    captured = {}
    _spy_neighbor(monkeypatch, captured)

    extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)

    # PATH input + adjacency-only -> crops stream off a memmap (NOT a resident
    # np.array copy of the whole volume), and the redundant find_objects is skipped.
    assert captured["is_memmap"] is True
    assert captured["slices_given"] is True


def test_adjacency_with_tissue_depth_reuses_resident(tmp_path, monkeypatch):
    """When a truly-global pass (tissue_depth) also runs, it loads the resident
    full volume; the graph family must REUSE that resident array (no second
    memmap), still with slices injected."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    captured = {}
    _spy_neighbor(monkeypatch, captured)

    extract_label_features(
        str(p), n_workers=1, adjacency=True, tissue_depth=True, max_radius=5)

    assert captured["is_memmap"] is False     # reused the resident full_labels
    assert captured["slices_given"] is True


def test_adjacency_off_does_not_invoke_graph(tmp_path, monkeypatch):
    """adjacency=False -> the graph family never runs (no spy hit) and no
    _adjacency_list is attached — the labels-only contract is unchanged."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    captured = {}
    _spy_neighbor(monkeypatch, captured)

    df = extract_label_features(str(p), n_workers=1)   # adjacency off (default)

    assert captured == {}                              # spy never fired
    assert "_adjacency_list" not in df.attrs


# --------------------------------------------------------------------------- #
# 3. Regression locks (audit Q4): the byte-identical equivalence must hold on   #
#    gapped ids, int64 dtype, border-clamped crops, and the resident-reuse path #
# --------------------------------------------------------------------------- #
def _run_chain(base, src, slices):
    df = base.copy()
    qf.compute_neighbor_features(src, df, max_radius=5, slices=slices)
    qf.compute_contact_geometry(src, df, spacing=None, max_radius=5, slices=slices)
    qf.compute_axis_neighbor_persistence(src, df, spacing=None, max_radius=5, slices=slices)
    return df


def test_gapped_nonconsecutive_ids_equivalent(tmp_path):
    """Non-consecutive label ids {2, 5, 9} — pins the ``slices[id-1]`` indexing +
    the ``id > len(slices)`` bounds guard under the injection."""
    vol = np.zeros((8, 16, 30), np.int32)
    vol[1:6, 2:8, 2:8] = 2
    vol[1:6, 2:8, 8:14] = 5          # touches id 2
    vol[1:6, 2:8, 24:30] = 9         # isolated
    p = tmp_path / "l.npy"
    np.save(p, vol)
    base = extract_label_features(str(p), n_workers=1)
    base["label"] = base["label"].astype(np.int64)
    objs = find_objects(vol)
    mm = np.load(str(p), mmap_mode="r")

    df_ref = _run_chain(base, vol, None)
    df_new = _run_chain(base, mm, objs)
    pd.testing.assert_frame_equal(df_ref, df_new)
    edges = {(e["label_a"], e["label_b"]) for e in df_new.attrs["_adjacency_list"]}
    assert edges == {(2, 5)}


def test_int64_labels_equivalent(tmp_path):
    """int64 label volume (all tests/bench use int32) — injected int64 memmap path
    must match the resident path."""
    vol = _multi_touch_labels().astype(np.int64)
    p = tmp_path / "l64.npy"
    np.save(p, vol)
    base = extract_label_features(str(p), n_workers=1)
    base["label"] = base["label"].astype(np.int64)
    mm = np.load(str(p), mmap_mode="r")
    assert mm.dtype == np.int64
    df_ref = _run_chain(base, np.asarray(vol), None)
    df_new = _run_chain(base, mm, find_objects(vol))
    pd.testing.assert_frame_equal(df_ref, df_new)


def test_border_clamped_crop_equivalent(tmp_path):
    """A cell flush at the array corner (z=y=x=0) exercises the crop clamps
    ``max(start-r, 0)`` / ``min(stop+r, dim)`` — injected ≡ internal there too."""
    vol = np.zeros((8, 16, 20), np.int32)
    vol[0:5, 0:6, 0:6] = 1           # corner cell (touches Z=0/Y=0/X=0)
    vol[0:5, 0:6, 6:12] = 2          # touches cell 1 at x==6
    p = tmp_path / "lb.npy"
    np.save(p, vol)
    base = extract_label_features(str(p), n_workers=1)
    base["label"] = base["label"].astype(np.int64)
    mm = np.load(str(p), mmap_mode="r")
    df_ref = _run_chain(base, vol, None)
    df_new = _run_chain(base, mm, find_objects(vol))
    pd.testing.assert_frame_equal(df_ref, df_new)


def test_gapped_pair_found_by_graduated_dilation(tmp_path):
    """A 2-voxel GAP between cells: a single radius-1 shell finds NOTHING; the
    graduated 1..5 dilation finds the contact. Pins the gap-aware semantics (a
    revert to a single shell would report neighbor_count 0 here)."""
    vol = np.zeros((8, 16, 24), np.int32)
    vol[1:6, 2:8, 2:8] = 1
    vol[1:6, 2:8, 10:16] = 2         # x==8,9 empty -> a 2-voxel gap
    p = tmp_path / "lg.npy"
    np.save(p, vol)
    df = extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)
    by = df.set_index("label")
    assert by.loc[1, "neighbor_count"] == 1.0
    edges = {(e["label_a"], e["label_b"]): e["contact_voxels"]
             for e in df.attrs["_adjacency_list"]}
    assert (1, 2) in edges
    assert edges[(1, 2)] > 0          # contact recovered despite the gap


def test_tissue_depth_resident_reuse_matches_memmap_graph(tmp_path):
    """VALUE equivalence for the resident-reuse branch: with tissue_depth=True the
    graph family runs on the RESIDENT full_labels; the graph columns + edge list
    must equal the adjacency-only MEMMAP run (tissue_depth only ADDS columns)."""
    vol = _multi_touch_labels()
    p = tmp_path / "l.npy"
    np.save(p, vol)
    a = (extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)
         .sort_values("label").reset_index(drop=True))                   # memmap graph
    b = (extract_label_features(str(p), n_workers=1, adjacency=True,
                                tissue_depth=True, max_radius=5)
         .sort_values("label").reset_index(drop=True))                   # resident reuse

    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    graph_cols = [c for c in _FEATURE_FAMILIES["adjacency"]
                  if c in a.columns and c in b.columns]
    assert "neighbor_count" in graph_cols
    for c in graph_cols:
        np.testing.assert_allclose(
            a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True,
            err_msg=f"graph column {c} differs memmap vs resident-reuse")
    assert a.attrs["_adjacency_list"] == b.attrs["_adjacency_list"]
