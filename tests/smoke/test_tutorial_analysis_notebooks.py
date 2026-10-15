"""Run the five analysis tutorial notebooks end to end on a small synthetic table.

The notebooks are executed cell by cell in a fresh namespace (no Jupyter kernel
is needed). The EDIT cell is rewritten to point at files in a temporary folder.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = REPO_ROOT / "tutorials" / "analysis"
NB1 = NB_DIR / "01_cell_shape_morphospace.ipynb"
NB2 = NB_DIR / "02_cell_neighbourhood_niches.ipynb"
NB3 = NB_DIR / "03_figure_panels.ipynb"
NB4 = NB_DIR / "04_niche_figure_panels.ipynb"
NB5 = NB_DIR / "05_compare_two_groups_of_cells.ipynb"
ALL_NBS = (NB1, NB2, NB3, NB4, NB5)

# Built from pieces so this file does not itself contain the strings it forbids.
_BS = chr(92)
_WORDS = ["FT" + "36", "don" + "or", "plac" + "ode", _BS + "b" + "rid" + "ge", "wh" + "orl", "vol" + "ar",
          "dors" + "al", _BS + "b" + "ha" + "ir", "em" + "bryo", "CLA" + "UDE", "docs/" + "planning", "docs/" + "architecture",
          "GOT" + "CHAS", "spec " + chr(167), _BS + "bS" + "4" + _BS + "b", "Fig" + _BS + ". ?7", "pati" + "ent"]
FORBIDDEN = re.compile("|".join(["[A-Za-z]:" + _BS + _BS, "/ho" + "me/", "/Us" + "ers/", "Down" + "loads"] + _WORDS), re.I)
SCANNED_FILES = (NB_DIR / "_helpers.py", NB_DIR / "FEATURES.md")


def _cells(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return nb, [c for c in nb["cells"] if c["cell_type"] == "code"]


def _run(path: Path, override: str) -> dict:
    """Execute the code cells in order; ``override`` is appended to the EDIT cell."""
    _, code = _cells(path)
    ns: dict = {"__name__": "__main__"}
    for cell in code:
        src = "".join(cell["source"])
        if src.lstrip().startswith("# ---------------- EDIT ME"):
            src = src + "\n" + override + "\n"
        exec(compile(src, f"{path.name}[cell]", "exec"), ns)
    return ns


def _make_data(folder: Path) -> Path:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    rows = []
    stacks = ["stack_a", "stack_b"]
    n_per = 220
    for s_i, stack in enumerate(stacks):
        group = rng.integers(0, 3, n_per)
        size = np.exp(rng.normal(5 + 0.5 * group, 0.4))
        pos = rng.uniform(0, 100, (n_per, 3))
        feat = {"area": size, "area_filled": size * 1.02, "equivalent_diameter": size ** (1 / 3),
                "bbox_volume": size * (1.5 + 0.3 * group), "surface_area": size ** (2 / 3) * 5}
        for name in ("extent", "fill_ratio", "sphericity", "true_sphericity", "solidity",
                     "ellipsoid_solidity", "inertia_elongation", "inertia_flatness"):
            feat[name] = np.clip(rng.normal(0.5 + 0.12 * group, 0.08), 0.01, 1.0)
        for name in ("aspect_ratio", "z_span", "z_elongation", "xy_elongation", "axis_major_length",
                     "axis_minor_length", "apical_area", "basal_area", "cross_section_taper",
                     "cross_section_bulge", "skeleton_length"):
            feat[name] = np.exp(rng.normal(0.3 * group, 0.5))
        feat["euler_number"] = rng.integers(0, 3, n_per).astype(float)
        feat["major_axis_z_angle"] = rng.uniform(0, 90, n_per)
        feat["skeleton_branches"] = rng.integers(0, 6, n_per).astype(float)
        feat["skeleton_nodes"] = rng.integers(0, 8, n_per).astype(float)
        feat["apical_basal_area_ratio"] = np.exp(rng.normal(0, 0.3, n_per))
        feat["max_cross_section_z_relative"] = rng.uniform(0, 1, n_per)
        df = pd.DataFrame({"stack_name": stack, "label": np.arange(1, n_per + 1),
                           "centroid-0": pos[:, 0], "centroid-1": pos[:, 1], "centroid-2": pos[:, 2], **feat})
        rows.append(df)
        _, nn = cKDTree(pos).query(pos, k=5)
        pairs = {(min(i, j), max(i, j)) for i in range(n_per) for j in nn[i, 1:] if i != j}
        adj = pd.DataFrame(sorted(pairs), columns=["label_a", "label_b"]) + 1
        adj["contact_voxels"] = np.exp(rng.normal(5.5, 1.0, len(adj))).astype(int)
        adj.to_csv(folder / f"{stack}_adjacency.csv", index=False)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(folder / "all_stacks_features.csv", index=False)
    pd.DataFrame({"stack_name": stacks, "region": ["left", "right"]}).to_csv(folder / "metadata.csv", index=False)
    return folder / "all_stacks_features.csv"


def test_notebooks_have_no_outputs_and_plain_kernel():
    for path in ALL_NBS:
        nb, code = _cells(path)
        assert nb["metadata"]["kernelspec"]["name"] == "python3"
        for cell in code:
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
        text = " ".join("".join(c["source"]) for c in nb["cells"])
        assert not FORBIDDEN.search(text), FORBIDDEN.search(text).group(0)
    for path in SCANNED_FILES:
        text = path.read_text(encoding="utf-8")
        assert not FORBIDDEN.search(text), (path.name, FORBIDDEN.search(text).group(0))
        assert text.isascii(), path.name


def test_notebooks_run_on_synthetic_data(tmp_path, monkeypatch):
    import matplotlib

    try:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.close("all")
    except Exception as exc:  # no usable plotting backend on this machine
        pytest.skip(f"matplotlib backend not usable: {exc}")
    for mod in ("umap", "leidenalg", "igraph"):
        pytest.importorskip(mod)
    monkeypatch.chdir(REPO_ROOT)
    sys.modules.pop("_helpers", None)

    table = _make_data(tmp_path)
    out1 = tmp_path / "shape_output"
    out2 = tmp_path / "niche_output"

    _run(NB1, (
        f'TABLE_CSV = r"{table}"\n'
        f'METADATA_CSV = r"{tmp_path / "metadata.csv"}"\n'
        f'OUTPUT_DIR = r"{out1}"\n'
        'COLOR_BY = ["region"]\n'
        'N_PCS = 6\nLEIDEN_RESOLUTIONS = [0.5, 1]\nACTIVE_RES = 1\n'
    ))
    emb_path = out1 / "shape_embedding.csv"
    assert emb_path.exists()
    emb = pd.read_csv(emb_path)
    for col in ("stack_name", "label", "region", "PC1", "PC6", "umap_1", "umap_2", "leiden_r0.5", "leiden_r1.0"):
        assert col in emb.columns, col
    assert len(emb) == 440
    for name in ("pca_loadings.csv", "cluster_shape_stats.csv", "analysis_config.json", "representative_cells.csv"):
        assert (out1 / name).exists(), name

    _run(NB2, (
        f'SHAPE_EMBEDDING_CSV = r"{emb_path}"\n'
        f'ADJACENCY_DIR = r"{tmp_path}"\n'
        f'OUTPUT_DIR = r"{out2}"\n'
        'NICHE_CONFIGS = [{"name": "hop1_pure", "self_weight": 0.0, "n_iter": 1},'
        ' {"name": "hop3_pure", "self_weight": 0.0, "n_iter": 3}]\n'
        'PRIMARY_NICHE_RES = 1\nNICHE_LEIDEN_RES = [0.5, 1]\nMIN_CONTACT = 50\n'
    ))
    niche_path = out2 / "niche_embedding.csv"
    assert niche_path.exists()
    niche = pd.read_csv(niche_path)
    for col in ("niche_PC1", "niche_PC6", "niche_umap_1", "niche_umap_2", "niche_leiden_r0.5", "niche_leiden_r1.0",
                "leiden_r1.0"):
        assert col in niche.columns, col
    have_niche = niche["niche_PC1"].notna()
    assert have_niche.sum() > 200
    # the averaged value must differ from the cell's own value: contacts were really matched
    assert (niche.loc[have_niche, "niche_PC1"] - niche.loc[have_niche, "PC1"]).abs().gt(1e-9).mean() > 0.9
    assert (out2 / "niche_analysis_config.json").exists()

    # 03: figure panels from the shape embedding
    out3 = tmp_path / "figure_output"
    _run(NB3, (
        f'SHAPE_EMBEDDING_CSV = r"{emb_path}"\n'
        f'METADATA_CSV = r"{tmp_path / "metadata.csv"}"\n'
        f'OUTPUT_DIR = r"{out3}"\n'
        'COLOR_BY = ["region", "centroid-0", "not_a_column"]\n'
        'GROUP_COL = "region"\n'
    ))
    for name in ("umap_clusters", "umap_by_region", "umap_by_centroid-0", "cluster_share_by_region",
                 "heatmaps_region_by_cluster", "cluster_share_per_stack", "umap_cluster_badges", "umap_ghost"):
        assert (out3 / "figures" / f"{name}.svg").exists(), name
        assert (out3 / "figures" / f"{name}.png").exists(), name
    assert (out3 / "tables" / "typical_cells.csv").exists()

    # 04: niche figure panels (needs both embeddings)
    out4 = tmp_path / "niche_figure_output"
    _run(NB4, (
        f'NICHE_EMBEDDING_CSV = r"{niche_path}"\n'
        f'SHAPE_EMBEDDING_CSV = r"{emb_path}"\n'
        f'METADATA_CSV = r"{tmp_path / "metadata.csv"}"\n'
        f'OUTPUT_DIR = r"{out4}"\n'
        'NICHE_RESOLUTION = 1.0\nGROUP_COL = "region"\nMAX_CELLS_FOR_MIXING = 400\n'
    ))
    for name in ("niche_umap_clusters", "maps_shape_and_niche_cross", "niche_violins", "niche_share_by_region",
                 "enrichment_shape_by_niche", "do_replicates_mix"):
        assert (out4 / "figures" / f"{name}.svg").exists(), name
    assert (out4 / "tables" / "stack_mixing.csv").exists()

    # 05: compare two groups of cells; groups are four shape clusters picked by size
    counts = emb["leiden_r1.0"].value_counts()
    assert len(counts) >= 4, "the synthetic table should give at least four clusters"
    a, b, ra, rb = (int(c) for c in counts.index[:4])
    out5 = tmp_path / "comparison_output"
    _run(NB5, (
        f'EMBEDDING_CSV = r"{emb_path}"\n'
        f'METADATA_CSV = r"{tmp_path / "metadata.csv"}"\n'
        f'OUTPUT_DIR = r"{out5}"\n'
        f'GROUP_SPEC = {{"A": [{a}], "A_ref": [{ra}], "B": [{b}], "B_ref": [{rb}]}}\n'
        'N_BOOT = 50\nMIN_GROUP_CELLS = 20\n'
    ))
    tables = out5 / "comparison_tables"
    assert tables.is_dir()
    for name in ("summary_by_group", "departure_from_reference", "direct_comparison", "pc_separation_key",
                 "share_per_stack"):
        assert (tables / f"{name}.csv").exists(), name
    direct = pd.read_csv(tables / "direct_comparison.csv", header=[0, 1], index_col=0)
    assert direct.shape[0] >= 10
    for name in ("departure_from_reference", "direct_comparison", "violins", "map_groups", "pc_separation"):
        assert (out5 / "figures" / f"{name}.svg").exists(), name

    # 05 left at its defaults stops on purpose and says what to do
    with pytest.raises(SystemExit, match="Define at least groups A and B|define at least groups A and B"):
        _run(NB5, (
            f'EMBEDDING_CSV = r"{emb_path}"\n'
            f'OUTPUT_DIR = r"{tmp_path / "stopped"}"\n'
        ))

    # one stack, no metadata, no references
    one = emb[emb["stack_name"] == "stack_a"]
    one_path = tmp_path / "one_stack_embedding.csv"
    one.to_csv(one_path, index=False)
    out3b = tmp_path / "figure_output_one"
    _run(NB3, f'SHAPE_EMBEDDING_CSV = r"{one_path}"\nOUTPUT_DIR = r"{out3b}"\n')
    assert (out3b / "figures" / "umap_clusters.svg").exists()
    assert not (out3b / "figures" / "cluster_share_per_stack.svg").exists()
    out5b = tmp_path / "comparison_one"
    _run(NB5, (
        f'EMBEDDING_CSV = r"{one_path}"\n'
        f'OUTPUT_DIR = r"{out5b}"\n'
        f'GROUP_SPEC = {{"A": [{a}], "B": [{b}]}}\n'
        'N_BOOT = 20\nMIN_GROUP_CELLS = 10\n'
    ))
    assert (out5b / "comparison_tables" / "direct_comparison.csv").exists()
    assert not (out5b / "figures" / "departure_from_reference.svg").exists()


def test_group_builders_and_statistics(tmp_path):
    sys.path.insert(0, str(NB_DIR))
    try:
        import _helpers as h
    finally:
        sys.path.remove(str(NB_DIR))
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"stack_name": ["s"] * 60, "label": np.arange(1, 61), "x": rng.normal(size=60),
                       "kind": ["p"] * 20 + ["q"] * 20 + ["r"] * 20})
    # column mode and rule mode agree
    by_col = h.build_groups(df, "column", {"A": ["p"], "B": ["q"]}, column="kind")
    by_rule = h.build_groups(df, "rule", {"A": "kind == 'p'", "B": "kind == 'q'"})
    assert by_col.equals(by_rule) and by_col.notna().sum() == 40
    # csv mode
    pd.DataFrame({"stack_name": ["s"] * 3, "label": [1, 2, 41], "group": ["g1", "g1", "g2"]}).to_csv(
        tmp_path / "groups.csv", index=False)
    by_csv = h.build_groups(df, "csv", {"A": "g1", "B": "g2"}, csv=tmp_path / "groups.csv")
    assert (by_csv == "A").sum() == 2 and (by_csv == "B").sum() == 1
    # overlapping groups are refused
    with pytest.raises(ValueError):
        h.build_groups(df, "rule", {"A": "x > -10", "B": "x > 0"})
    # Cliff's delta: exact value against the definition, bootstrap interval brackets the estimate
    a, b = rng.normal(1.0, 1, 50), rng.normal(0.0, 1, 40)
    brute = np.mean([int(ai > bj) - int(ai < bj) for ai in a for bj in b])
    assert abs(h.cliffs_delta(a, b) - brute) < 1e-12
    d, lo, hi = h.delta_with_ci(a, b, n_boot=200)
    assert lo <= d <= hi and d > 0.3


def test_niche_diffuse_ignores_small_contacts(tmp_path):
    sys.path.insert(0, str(NB_DIR))
    try:
        import _helpers as h
    finally:
        sys.path.remove(str(NB_DIR))
    pcs = np.array([[0.0], [10.0], [20.0]])
    pd.DataFrame({"label_a": [1, 2], "label_b": [2, 3], "contact_voxels": [500, 5]}).to_csv(
        tmp_path / "s_adjacency.csv", index=False)
    out, stats = h.niche_diffuse(pcs, ["s"] * 3, [1, 2, 3], tmp_path, self_weight=0.0, n_iter=1, min_contact=100)
    assert out[0, 0] == 10.0 and out[1, 0] == 0.0
    assert np.isnan(out[2, 0])  # no usable contact: no neighbourhood, not a copy of its own value
    assert int(stats["edges_kept"].iloc[0]) == 1 and int(stats["cells_without_contact"].iloc[0]) == 1
    kept, _ = h.niche_diffuse(pcs, ["s"] * 3, [1, 2, 3], tmp_path, self_weight=0.0, n_iter=1, min_contact=100,
                              isolated="keep")
    assert kept[2, 0] == 20.0


def test_more_helpers(tmp_path):
    sys.path.insert(0, str(NB_DIR))
    try:
        import _helpers as h
    finally:
        sys.path.remove(str(NB_DIR))
    # number-like stack names stay text; resolution columns always carry a decimal point
    pd.DataFrame({"stack_name": ["001", "002"], "label": [1, 2], "x": [1.0, 2.0]}).to_csv(tmp_path / "t.csv", index=False)
    assert list(h.load_table(tmp_path / "t.csv")["stack_name"]) == ["001", "002"]
    assert h.res_name(1) == "leiden_r1.0" and h.res_name(1.5, "niche_leiden_r") == "niche_leiden_r1.5"
    # the contact-file search finds the plugin's folder and names the folders it tried otherwise
    ws = tmp_path / "ws"
    (ws / "artifacts" / "step4" / "qc").mkdir(parents=True)
    (ws / "artifacts" / "step4" / "exports").mkdir(parents=True)
    (ws / "artifacts" / "step4" / "qc" / "s_adjacency.csv").write_text("label_a,label_b,contact_voxels\n1,2,5\n")
    table = ws / "artifacts" / "step4" / "exports" / "features.csv"
    table.write_text("x\n")
    assert h.find_adjacency_dir(["s"], table) == (ws / "artifacts" / "step4" / "qc").resolve()
    with pytest.raises(FileNotFoundError, match="Folders tried"):
        h.find_adjacency_dir(["nope"], table)
    # a mostly empty feature is dropped, not the whole table
    d = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0), "c": [np.nan] * 15 + [1.0] * 5})
    assert h.drop_sparse_features(d, ["a", "b", "c"]) == ["a", "b"]
    # replicate dots: a stack without the group counts as 0
    df = pd.DataFrame({"stack_name": ["s1"] * 10 + ["s2"] * 10, "g": ["A"] * 10 + [None] * 10})
    _, tab = h.replicate_dotplot(df, "g", groups=["A"])
    assert sorted(tab["value"]) == [0.0, 1.0]
    # Cliff's delta ignores NaN; the PC separation is symmetric; rules need backticks for hyphens
    assert h.cliffs_delta([1, np.nan, 3], [2, 2]) == 0.0
    pc = pd.DataFrame({"grp": ["a"] * 30 + ["b"] * 30, "PC1": np.r_[np.zeros(30), np.ones(30) * 3] + np.arange(60) % 5 * 0.1,
                       "PC2": np.arange(60) % 7 * 0.1})
    raw, norm, _ = h.pc_separation(pc, "grp", ["a", "b"], ["PC1", "PC2"])
    assert abs(raw.loc["a", "b"] - raw.loc["b", "a"]) < 1e-12 and norm.loc["a", "b"] > 1
    cells = pd.DataFrame({"centroid-0": np.arange(10.0), "area": np.arange(10.0)})
    assert h.mask_from_rule(cells, "`centroid-0` < 3").sum() == 3
    with pytest.raises(ValueError):
        h.mask_from_rule(cells, "area")
