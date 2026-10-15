"""Helpers shared by the analysis tutorials.

The tutorials take the feature table written by the segmentation tutorial (or by
the plugin's Export window) and turn it into shape clusters and neighbourhood
clusters.  Everything that is not specific to one notebook lives here, so that
each notebook stays short and readable.

Table contract
--------------
One CSV, one row per cell.  Columns: ``stack_name`` (which image the cell
came from), ``label`` (the cell id inside that image), ``centroid-0/1/2``
(position) and the shape features.  Optionally, next to the CSV, one contact
file per stack named ``<stack_name>_adjacency.csv`` with the columns
``label_a, label_b, contact_voxels``.  The segmentation tutorial writes these
next to the table.  The plugin's Step 4 writes them under
``artifacts/step4/qc/`` of the workspace (its Export window writes only the
merged table, under ``artifacts/step4/exports/``).

Nothing in this module needs the plugin or napari.
"""

from __future__ import annotations

import colorsys
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------

#: The 31 shape features used by the morphospace notebook.  The notebook keeps
#: only the ones your table actually has.
TARGET_FEATURES: List[str] = [
    "area", "area_filled", "equivalent_diameter", "extent", "euler_number",
    "bbox_volume", "fill_ratio",
    "sphericity", "aspect_ratio", "z_span", "z_elongation", "xy_elongation",
    "ellipsoid_solidity", "inertia_elongation", "inertia_flatness",
    "axis_major_length", "axis_minor_length", "major_axis_z_angle",
    "surface_area", "true_sphericity", "solidity",
    "skeleton_branches", "skeleton_nodes", "skeleton_length",
    "apical_area", "basal_area", "apical_basal_area_ratio",
    "max_cross_section_z_relative", "cross_section_taper", "cross_section_bulge",
    "roughness",
]

#: Features that mostly measure size.  Used by the optional size correction.
SIZE_REGRESSORS: List[str] = [
    "area", "area_filled", "equivalent_diameter", "bbox_volume",
    "surface_area", "axis_major_length", "axis_minor_length",
]

# Okabe-Ito colour-blind-safe colours first, then more distinct colours.
_BASE_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#999999", "#882255", "#44AA99",
    "#117733", "#AA4499", "#332288", "#DDCC77", "#88CCEE",
    "#CC6677", "#661100", "#6699CC", "#AA4466", "#4477AA",
]

#: Features with a natural 0-to-1 range.  Under ``LOG_RULE = "workbench"`` these
#: are never log-transformed (the same rule as the plugin's analysis workbench).
BOUNDED_FEATURES = frozenset([
    "background_fraction", "compactness", "discrete_compactness", "ellipsoid_solidity", "extent",
    "fill_ratio", "max_contact_fraction", "max_cross_section_z_relative", "neighbor_persistence_ratio",
    "neighbor_z_turnover", "prob_boundary_entropy", "prob_boundary_mean", "prob_interior_mean",
    "prob_mask_entropy", "prob_mean", "prob_p10", "solidity", "sphericity", "true_sphericity",
])


def res_name(resolution, prefix: str = "leiden_r") -> str:
    """Column name for a clustering resolution, always with a decimal point.

    ``res_name(1)`` and ``res_name(1.0)`` both give ``'leiden_r1.0'``, so a
    notebook that writes the column and one that reads it always agree.
    ``res_name(1.5, "niche_leiden_r")`` gives ``'niche_leiden_r1.5'``.
    """
    return f"{prefix}{float(resolution)}"


def read_table_csv(path: PathLike) -> pd.DataFrame:
    """Read one of the notebooks' CSV files, keeping ``stack_name`` as text.

    Stack names such as ``001`` would otherwise be read as the number 1.
    """
    return pd.read_csv(path, dtype={"stack_name": str})


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def load_table(csv: PathLike, adjacency_dir: Optional[PathLike] = None) -> pd.DataFrame:
    """Read a feature table and remember where the contact files are.

    If the table has no ``stack_name`` column (one stack only), the file name
    is used as the stack name.  The folder with the ``<stack_name>_adjacency.csv``
    files defaults to the folder of the table; it is stored in
    ``df.attrs["adjacency_dir"]``.
    """
    csv = Path(csv)
    if not csv.exists():
        raise FileNotFoundError(f"Table not found: {csv}")
    df = read_table_csv(csv)
    if "stack_name" not in df.columns:
        df.insert(0, "stack_name", csv.stem)
    if "label" not in df.columns:
        raise ValueError("The table needs a 'label' column (the cell id inside each stack).")
    dup = int(df.duplicated(["stack_name", "label"]).sum())
    if dup:
        print(f"Warning: {dup} row(s) repeat a (stack_name, label) pair. Contacts can only be matched to the first.")
    df.attrs["adjacency_dir"] = str(Path(adjacency_dir) if adjacency_dir else csv.parent)
    return df


def attach_metadata(df: pd.DataFrame, metadata_csv: Optional[PathLike]) -> pd.DataFrame:
    """Add your own columns per stack (for example sample, region, age group).

    ``metadata_csv`` has one row per stack: a ``stack_name`` column plus any
    other columns.  Stacks missing from the file get an empty value.  Pass
    ``None`` to skip.
    """
    if not metadata_csv:
        return df
    meta = read_table_csv(metadata_csv)
    if "stack_name" not in meta.columns:
        raise ValueError("The metadata file needs a 'stack_name' column.")
    meta = meta.drop_duplicates("stack_name")
    new_cols = [c for c in meta.columns if c != "stack_name" and c not in df.columns]
    clash = [c for c in meta.columns if c != "stack_name" and c in df.columns]
    if clash:
        print(f"Metadata column(s) already in the table are kept as they are: {clash}")
    attrs = dict(df.attrs)
    out = df.merge(meta[["stack_name"] + new_cols], on="stack_name", how="left")
    out.attrs.update(attrs)
    missing = sorted(set(df["stack_name"]) - set(meta["stack_name"]))
    if missing:
        print(f"{len(missing)} stack(s) are not in the metadata file (their values stay empty).")
    return out


def subsample_cells(df: pd.DataFrame, max_cells: Optional[int], seed: int = 42) -> pd.DataFrame:
    """Return at most ``max_cells`` random rows (all rows if ``max_cells`` is None).

    Embeddings such as UMAP get slow above a few hundred thousand cells.  A
    random subset keeps the picture the same and the run short.
    """
    if max_cells is None or len(df) <= max_cells:
        return df
    attrs = dict(df.attrs)
    out = df.sample(n=int(max_cells), random_state=seed).sort_index().reset_index(drop=True)
    out.attrs.update(attrs)
    print(f"Using a random subset of {len(out):,} of {len(df):,} cells.")
    return out


def read_adjacency(stack_name: str, adjacency_dir: PathLike) -> Optional[pd.DataFrame]:
    """Read ``<stack_name>_adjacency.csv`` from ``adjacency_dir``; None if absent."""
    path = Path(adjacency_dir) / f"{stack_name}_adjacency.csv"
    if not path.exists():
        return None
    adj = pd.read_csv(path)
    need = {"label_a", "label_b", "contact_voxels"}
    if not need.issubset(adj.columns):
        raise ValueError(f"{path.name} needs the columns {sorted(need)}; found {list(adj.columns)}.")
    return adj


def drop_sparse_features(df: pd.DataFrame, features: Sequence[str], max_missing: float = 0.10) -> List[str]:
    """Keep the features that are missing in at most ``max_missing`` of the cells.

    A feature that is empty for a whole stack (for example because it was not
    measured there) would otherwise remove every cell of that stack.  The
    dropped features are printed with their share of missing values.
    """
    feats = [f for f in features if f in df.columns]
    miss = df[feats].replace([np.inf, -np.inf], np.nan).isna().mean()
    dropped = [f for f in feats if miss[f] > max_missing]
    for f in dropped:
        print(f"Dropped '{f}': {miss[f]:.0%} of the cells have no value for it.")
    return [f for f in feats if f not in dropped]


def find_adjacency_dir(stack_names: Sequence[str], table_path: PathLike, explicit: Optional[PathLike] = None,
                       recorded: Optional[PathLike] = None) -> Path:
    """Find the folder that holds the ``<stack_name>_adjacency.csv`` files.

    Tried in this order: ``explicit`` (your setting), ``recorded`` (the folder
    the first notebook saw its table in), the folder of ``table_path``, the folder
    above it, and ``artifacts/step4/qc/`` of a plugin workspace above the table.
    Prints every folder it tried when none works.
    """
    table_path = Path(table_path).resolve()
    cands: List[Path] = []
    for c in (explicit, recorded):
        if c:
            cands.append(Path(c))
    cands += [table_path.parent, table_path.parent.parent]
    cands += [a / "artifacts" / "step4" / "qc" for a in [table_path.parent] + list(table_path.parents)]
    seen, tried = set(), []
    for c in cands:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        tried.append(c)
        if c.is_dir() and any((c / f"{s}_adjacency.csv").exists() for s in stack_names):
            return c
    msg = "\n".join(f"  {c}" for c in tried[:8])
    raise FileNotFoundError(
        "No <stack_name>_adjacency.csv file was found for the stacks in your table. Folders tried:" + "\n" + msg +
        "\nSet ADJACENCY_DIR to the folder that holds them. The segmentation tutorial writes them next to its "
        "feature table; the plugin's Step 4 writes them to artifacts/step4/qc/.")


# ----------------------------------------------------------------------------
# Preprocessing and PCA
# ----------------------------------------------------------------------------


def preprocess_features(
    X: np.ndarray,
    features: Sequence[str],
    winsor_clip: Optional[float] = 5.0,
    skew_threshold: float = 1.0,
    verbose: bool = True,
    log_rule: str = "original",
) -> Tuple[np.ndarray, List[str]]:
    """Make features comparable: log, robust-scale, clip.

    1. ``log_rule="original"``: a feature that is never negative and strongly
       skewed to either side (|skew| above ``skew_threshold``) gets ``log1p``.
       ``log_rule="workbench"``: the plugin's rule.  Features with a 0-to-1
       range (``BOUNDED_FEATURES``) are never logged, and only right-skewed
       features (skew above ``skew_threshold``) get ``log1p`` (shifted up
       first if they have negative values).
    2. Every feature is centred on its median and divided by its inter-quartile
       range (``RobustScaler``), so a few huge cells do not dominate.
    3. Values beyond ``winsor_clip`` robust units are clipped to that limit.

    Returns the scaled matrix and the names of the log-transformed features.
    """
    from scipy.stats import skew
    from sklearn.preprocessing import RobustScaler

    if log_rule not in ("original", "workbench"):
        raise ValueError("log_rule must be 'original' or 'workbench'.")
    X = np.array(X, dtype=float, copy=True)
    log_applied: List[str] = []
    for i, feat in enumerate(features):
        col = X[:, i]
        if log_rule == "original":
            if col.min() >= 0 and abs(skew(col)) > skew_threshold:
                X[:, i] = np.log1p(col)
                log_applied.append(feat)
        elif feat not in BOUNDED_FEATURES and skew(col, bias=False) > skew_threshold:
            X[:, i] = np.log1p(col - min(col.min(), 0.0))
            log_applied.append(feat)
    X = RobustScaler().fit_transform(X)
    if winsor_clip is not None:
        X = np.clip(X, -winsor_clip, winsor_clip)
    if verbose:
        print(f"log1p applied to {len(log_applied)} feature(s): {', '.join(log_applied) or 'none'}")
    return X, log_applied


def residualise_size(
    X: np.ndarray,
    features: Sequence[str],
    size_features: Sequence[str] = SIZE_REGRESSORS,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Remove the part of every shape feature that just tracks cell size.

    A size score is the first principal component of the size features.  Each
    remaining feature is regressed on it and only the residual is kept.  The
    size features themselves are dropped.  ``X`` must already be scaled.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    features = list(features)
    size_idx = [features.index(f) for f in size_features if f in features]
    if not size_idx:
        print("No size features found; nothing was removed.")
        return X, features
    size_pc1 = PCA(n_components=1, random_state=seed).fit_transform(X[:, size_idx]).ravel()
    size_z = ((size_pc1 - size_pc1.mean()) / (size_pc1.std() + 1e-12)).reshape(-1, 1)
    keep_idx = [i for i, f in enumerate(features) if f not in size_features]
    out = X[:, keep_idx].copy()
    for i in range(out.shape[1]):
        out[:, i] -= LinearRegression().fit(size_z, out[:, i]).predict(size_z)
    kept = [features[i] for i in keep_idx]
    print(f"Size removed from {len(kept)} feature(s).")
    return out, kept


def run_pca(
    X: np.ndarray, features: Sequence[str], n_pcs: int = 10, seed: int = 42
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """PCA.  Returns (scores, loadings table, variance explained per PC)."""
    from sklearn.decomposition import PCA

    n = min(n_pcs, X.shape[1], X.shape[0])
    pca = PCA(n_components=n, random_state=seed)
    pcs = pca.fit_transform(X)
    loadings = pd.DataFrame(
        pca.components_.T, index=list(features), columns=[f"PC{i + 1}" for i in range(n)]
    )
    return pcs, loadings, pca.explained_variance_ratio_


# ----------------------------------------------------------------------------
# Graph, clusters, embeddings
# ----------------------------------------------------------------------------


def knn_graph(X: np.ndarray, k: int = 15, metric: str = "euclidean"):
    """Weighted k-nearest-neighbour graph (python-igraph).

    The edges are built with array operations, not a Python loop.  The weight
    of an edge is ``exp(-distance)``; a pair found from both sides is merged.
    """
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    k = min(k, n - 1)
    dists, idxs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X).kneighbors(X)
    src = np.repeat(np.arange(n), k)
    dst = idxs[:, 1:].ravel()
    w = np.exp(-dists[:, 1:].ravel())
    g = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.es["weight"] = w.tolist()
    return g.simplify(combine_edges="mean")


def leiden_multi(
    X: np.ndarray,
    resolutions: Sequence[float],
    k: int = 15,
    metric: str = "euclidean",
    seed: int = 42,
) -> Dict[float, np.ndarray]:
    """Leiden clusters at several resolutions on one shared kNN graph.

    A higher resolution gives more, smaller clusters.  Returns
    ``{resolution: cluster id per row}``.
    """
    import leidenalg

    g = knn_graph(X, k=k, metric=metric)
    out: Dict[float, np.ndarray] = {}
    for res in resolutions:
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=res, seed=seed,
        )
        out[res] = np.asarray(part.membership, dtype=int)
        print(f"resolution {res}: {len(set(part.membership))} clusters")
    return out


def run_umap(
    X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
    metric: str = "euclidean", seed: int = 42,
) -> np.ndarray:
    """2D UMAP embedding of ``X`` (rows = cells)."""
    from umap import UMAP

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return UMAP(
            n_components=2, n_neighbors=min(n_neighbors, len(X) - 1),
            min_dist=min_dist, metric=metric, random_state=seed,
        ).fit_transform(X)


def run_tsne(X: np.ndarray, perplexity: float = 50, metric: str = "euclidean", seed: int = 42) -> np.ndarray:
    """2D t-SNE embedding (optional alternative to UMAP; slower)."""
    from sklearn.manifold import TSNE

    perplexity = min(perplexity, max(5, (len(X) - 1) / 3))
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed, metric=metric).fit_transform(X)


def select_representatives(
    pcs: np.ndarray, df: pd.DataFrame, cluster_col: str, n_reps: int = 10
) -> pd.DataFrame:
    """For every cluster, pick the cells closest to its median in PC space.

    These are the "typical" cells of each cluster.  Returns a table with
    ``cluster, rank, stack_name, label, dist_to_median``.
    """
    rows = []
    clusters = df[cluster_col].to_numpy()
    for c in sorted(pd.unique(clusters)):
        idx = np.where(clusters == c)[0]
        sub = pcs[idx]
        dist = np.linalg.norm(sub - np.median(sub, axis=0), axis=1)
        for rank, j in enumerate(np.argsort(dist)[: min(n_reps, len(idx))]):
            r = df.iloc[idx[j]]
            rows.append({
                "cluster": int(c), "rank": rank, "stack_name": r["stack_name"],
                "label": int(r["label"]), "dist_to_median": float(dist[j]),
            })
    return pd.DataFrame(rows)


def cluster_summary(df: pd.DataFrame, cluster_col: str, features: Sequence[str]) -> pd.DataFrame:
    """Median of every feature per cluster, plus the cluster size."""
    feats = [f for f in features if f in df.columns]
    out = df.groupby(cluster_col)[feats].median()
    out.insert(0, "n_cells", df.groupby(cluster_col).size())
    out.index.name = "cluster"
    return out.reset_index()


def enrichment_log2(a: Sequence, b: Sequence) -> pd.DataFrame:
    """How over- or under-represented is each pair of groups?

    Rows are the groups in ``a``, columns the groups in ``b``.  Each value is
    ``log2(observed / expected)``: observed is the share of group-a cells that
    fall in group b; expected is the share of all cells in group b.  Zero means
    "as expected by chance", +1 means twice as often, -1 half as often.
    Pairs that never occur are shown as -3 (the lower limit).
    """
    a = pd.Series(np.asarray(a)).reset_index(drop=True)
    b = pd.Series(np.asarray(b)).reset_index(drop=True)
    obs = pd.crosstab(a, b)
    row_share = obs.div(obs.sum(axis=1), axis=0)
    col_share = obs.sum(axis=0) / obs.to_numpy().sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.log2(row_share / col_share)
    return e.replace([-np.inf, np.inf], np.nan).fillna(-3).clip(-3, 3)


# ----------------------------------------------------------------------------
# Neighbourhood diffusion
# ----------------------------------------------------------------------------


def niche_diffuse(
    pcs: np.ndarray,
    stack_names: Sequence[str],
    labels: Sequence[int],
    adjacency_dir: PathLike,
    self_weight: float = 0.5,
    n_iter: int = 1,
    min_contact: float = 100,
    weight_by_contact: bool = True,
    isolated: str = "nan",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Average each cell's PCs with those of its touching neighbours.

    For every stack the contact file ``<stack_name>_adjacency.csv`` in
    ``adjacency_dir`` gives the cells that touch.  One step replaces a cell's
    values by::

        self_weight * (own values) + (1 - self_weight) * (mean of neighbours)

    ``n_iter`` steps reach neighbours of neighbours ("hops").  With
    ``self_weight = 0`` the cell's own value is not added back at each step
    (after two or more hops it still returns a little, through its neighbours).
    Contacts smaller than ``min_contact`` voxels are ignored; with
    ``weight_by_contact`` bigger contacts count more.  Cells without a usable
    contact come back as NaN rows (``isolated="nan"``, the default, so they are
    not mistaken for cells that have a neighbourhood) or keep their own values
    (``isolated="keep"``).  Stacks without a contact file come back as NaN rows.
    The statistics table counts the cells without a usable contact.

    Returns (niche matrix with the shape of ``pcs``, per-stack statistics).
    """
    from scipy import sparse

    if isolated not in ("nan", "keep"):
        raise ValueError("isolated must be 'nan' or 'keep'.")
    pcs = np.asarray(pcs, dtype=float)
    stack_names = np.asarray(stack_names)
    labels = np.asarray(labels).astype(int)
    out = np.full_like(pcs, np.nan, dtype=float)
    stats = []
    for stack in pd.unique(stack_names):
        idx = np.where(stack_names == stack)[0]
        adj = read_adjacency(str(stack), adjacency_dir)
        if adj is None:
            stats.append({"stack_name": stack, "cells": len(idx), "edges_total": 0,
                          "edges_kept": 0, "cells_without_contact": len(idx), "note": "no contact file"})
            continue
        local = pd.Series(np.arange(len(idx)), index=labels[idx])
        local = local[~local.index.duplicated()]
        total = len(adj)
        adj = adj[adj["contact_voxels"] >= min_contact]
        ra = adj["label_a"].astype(int).map(local)
        rb = adj["label_b"].astype(int).map(local)
        ok = ra.notna() & rb.notna()
        ra, rb = ra[ok].astype(int).to_numpy(), rb[ok].astype(int).to_numpy()
        if weight_by_contact:
            w = adj.loc[ok, "contact_voxels"].astype(float).to_numpy()
        else:
            w = np.ones(len(ra))
        n = len(idx)
        W = sparse.coo_matrix(
            (np.concatenate([w, w]), (np.concatenate([ra, rb]), np.concatenate([rb, ra]))), shape=(n, n)
        ).tocsr()
        row_sum = np.asarray(W.sum(axis=1)).ravel()
        iso = row_sum == 0
        row_sum[iso] = 1.0
        Wn = sparse.diags(1.0 / row_sum) @ W
        x_self = pcs[idx]
        x = x_self.copy()
        for _ in range(n_iter):
            x = self_weight * x_self + (1.0 - self_weight) * (Wn @ x)
            if iso.any():
                x[iso] = x_self[iso]
        if isolated == "nan":
            x[iso] = np.nan
        out[idx] = x
        stats.append({"stack_name": stack, "cells": n, "edges_total": total,
                      "edges_kept": int(len(ra)), "cells_without_contact": int(iso.sum()), "note": ""})
    return out, pd.DataFrame(stats)


# ----------------------------------------------------------------------------
# Colours and figures
# ----------------------------------------------------------------------------


def assign_palette(values: Iterable, order: Optional[Sequence] = None) -> Dict:
    """Give every distinct value of a column its own colour.

    The first colours are a colour-blind-safe set; more are generated when
    needed.  Pass ``order`` to fix which value gets which colour.  Returns
    ``{value: hex colour}``.
    """
    if order is not None:
        uniq = list(order)
    else:
        uniq = sorted(pd.unique(pd.Series(list(values)).dropna()))
    colours = list(_BASE_PALETTE[: len(uniq)])
    for i in range(max(0, len(uniq) - len(_BASE_PALETTE))):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.65 + (i % 3) * 0.1, 0.8 + (i % 2) * 0.1)
        colours.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
    return dict(zip(uniq, colours))


#: Page presets in millimetres.
A4_PORTRAIT_MM = (210, 297)
A4_LANDSCAPE_MM = (297, 210)
PANEL_FULL_MM = (180, 130)
PANEL_HALF_MM = (86, 70)


def save_fig(fig, name: str, out_dir: PathLike = "figures", mm: Optional[Tuple[float, float]] = None,
             formats: Sequence[str] = ("svg", "png")) -> List[Path]:
    """Save a matplotlib figure as SVG (editable vector) and PNG (300 dpi).

    ``mm=(width, height)`` resizes the figure to a physical size first, for
    example ``mm=PANEL_FULL_MM`` for a full-width panel.  Legends and labels
    outside the axes are kept (``bbox_inches="tight"``), so the saved file can
    be a little larger than ``mm``.  Text stays real text
    in the SVG, so you can edit it in a vector editor.
    """
    import matplotlib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mm is not None:
        fig.set_size_inches(mm[0] / 25.4, mm[1] / 25.4)
    paths = []
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        for fmt in formats:
            p = out_dir / f"{name}.{fmt}"
            fig.savefig(p, format=fmt, dpi=300, bbox_inches="tight")
            paths.append(p)
            print(f"saved {p}")
    return paths


def scatter_by(ax, x, y, values, palette: Optional[Dict] = None, s: float = 3, alpha: float = 0.6,
               title: str = "", legend: bool = True, max_legend: int = 25):
    """Scatter plot with points coloured by a categorical column."""
    values = np.asarray(values)
    palette = palette or assign_palette(values)
    colours = [palette.get(v, "#999999") for v in values]
    ax.scatter(x, y, c=colours, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if legend and len(palette) <= max_legend:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c, markeredgecolor=c,
                          markersize=4, label=str(k)) for k, c in palette.items()]
        ax.legend(handles=handles, fontsize=6, frameon=False, ncol=2 if len(handles) > 12 else 1,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
    return ax


def scatter_continuous(ax, x, y, values, cmap: str = "viridis", s: float = 3, alpha: float = 0.6,
                       title: str = "", label: str = ""):
    """Scatter plot with points coloured by a number."""
    sc = ax.scatter(x, y, c=values, cmap=cmap, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return ax


def column_kind(series: pd.Series, max_categories: int = 12) -> str:
    """'categorical' for text columns or columns with few distinct values, else 'numeric'."""
    if series.dtype == object or str(series.dtype) in ("category", "bool"):
        return "categorical"
    return "categorical" if series.nunique() <= max_categories else "numeric"


def plot_scree_loadings(variance: np.ndarray, loadings: pd.DataFrame, max_pcs: int = 10):
    """Left: variance explained per PC.  Right: which features drive each PC."""
    import matplotlib.pyplot as plt

    n = len(variance)
    show = min(max_pcs, n)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, max(4, 0.22 * len(loadings))),
                                 gridspec_kw={"width_ratios": [1, 1.6]})
    a1.bar(range(1, n + 1), variance, color="#0072B2")
    a1.set_xlabel("Principal component")
    a1.set_ylabel("Share of variance")
    a1.set_title("Scree plot", fontsize=10)
    im = a2.imshow(loadings.to_numpy()[:, :show], cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    a2.set_xticks(range(show))
    a2.set_xticklabels(loadings.columns[:show], fontsize=7)
    a2.set_yticks(range(len(loadings)))
    a2.set_yticklabels(loadings.index, fontsize=6)
    a2.set_title("Loadings (red = feature rises with the PC, blue = falls)", fontsize=10)
    fig.colorbar(im, ax=a2, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_enrichment(table: pd.DataFrame, xlabel: str, ylabel: str, title: str = ""):
    """Heatmap of an ``enrichment_log2`` table."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * table.shape[1] + 2), max(4, 0.4 * table.shape[0] + 1.5)))
    im = ax.imshow(table.to_numpy(), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, fontsize=7)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log2(observed / expected)", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Meshes (optional)
# ----------------------------------------------------------------------------


def label_bboxes(labels_volume: np.ndarray) -> list:
    """Bounding box (three slices) of every label, found in a single pass over the volume.

    Entry ``i`` belongs to label ``i + 1``; labels that do not occur are None.
    Compute this once and pass it to ``mesh_from_labels`` so each cell is cut
    out of the volume instead of scanning the whole volume again.
    """
    from scipy import ndimage

    return ndimage.find_objects(np.asarray(labels_volume).astype(np.int64, copy=False))


def mesh_from_labels(labels_volume: np.ndarray, label: int, spacing: Sequence[float] = (1.0, 1.0, 1.0),
                     pad: int = 2, bboxes: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Build a surface mesh of one cell from a labels volume (marching cubes).

    ``labels_volume`` is a 3D array (z, y, x) where every cell has its own
    integer.  ``spacing`` is the voxel size (z, y, x).  ``bboxes`` (from
    ``label_bboxes``) avoids scanning the whole volume for every cell.  Returns
    ``(vertices, faces)``; vertices are in the units of ``spacing``.
    """
    from skimage.measure import marching_cubes

    vol = labels_volume
    if bboxes is not None:
        box = bboxes[int(label) - 1] if 0 < int(label) <= len(bboxes) else None
        if box is None:
            raise ValueError(f"Label {label} is not in the volume.")
        lo = np.array([sl.start for sl in box])
        hi = np.array([sl.stop for sl in box])
    else:
        zs, ys, xs = np.where(np.asarray(vol) == label)
        if len(zs) == 0:
            raise ValueError(f"Label {label} is not in the volume.")
        lo = np.array([zs.min(), ys.min(), xs.min()])
        hi = np.array([zs.max(), ys.max(), xs.max()]) + 1
    crop = np.asarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) == label
    crop = np.pad(crop, pad)
    sp = np.asarray(spacing, dtype=float)
    verts, faces, _, _ = marching_cubes(crop.astype(np.float32), level=0.5, spacing=tuple(sp))
    verts = verts + (lo - pad) * sp
    return verts, faces


def read_labels_volume(path: PathLike) -> np.ndarray:
    """Read a labels volume from a TIFF or NumPy file.

    A TIFF is read completely into memory, so for a whole-tissue volume use
    the ``.npy`` file (it is opened without loading it) or one stack's volume.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r")
    import tifffile
    return tifffile.imread(path)


def plot_cell_meshes(meshes: Sequence[Tuple[np.ndarray, np.ndarray]], titles: Sequence[str] = (),
                     color: str = "#0072B2", ncols: int = 5):
    """Draw cell meshes side by side (matplotlib, no extra packages).

    All cells are drawn at the same scale, so you can compare their sizes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n = len(meshes)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    allv = np.vstack([v for v, _ in meshes])
    half = float((allv.max(axis=0) - allv.min(axis=0)).max()) / 2 or 1.0
    fig = plt.figure(figsize=(2.6 * ncols, 2.6 * nrows))
    for i, (v, f) in enumerate(meshes):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(v[f][:, :, ::-1], facecolor=color, edgecolor="none",
                                             alpha=0.9, linewidths=0))
        c = v.mean(axis=0)[::-1]
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_zlim(c[2] - half, c[2] + half)
        ax.set_axis_off()
        if i < len(titles):
            ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    return fig


def read_obj(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Read vertices and triangles from a simple OBJ mesh file (as the plugin exports)."""
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def plot_cluster_gallery(
    reps: pd.DataFrame,
    palette: Dict,
    labels_volume: Optional[np.ndarray] = None,
    labels_stack: Optional[str] = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    meshes_dir: Optional[PathLike] = None,
    n_per_cluster: int = 4,
    max_clusters: int = 12,
    title_prefix: str = "cluster ",
):
    """One row of typical cells per cluster, all drawn at the same scale.

    Meshes come from ``meshes_dir`` (``<meshes_dir>/<stack_name>/cell_<label>.obj``,
    as written by the plugin's Step 4) when the file exists, otherwise from
    ``labels_volume`` for cells of the stack named ``labels_stack``.  Returns the
    figure, or None when no mesh could be made.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rows = []
    bboxes = label_bboxes(labels_volume) if labels_volume is not None else None
    for c in sorted(reps["cluster"].unique())[:max_clusters]:
        meshes = []
        for _, r in reps[reps["cluster"] == c].sort_values("rank").iterrows():
            if len(meshes) >= n_per_cluster:
                break
            mesh = None
            if meshes_dir is not None:
                p = Path(meshes_dir) / str(r["stack_name"]) / f"cell_{int(r['label'])}.obj"
                if p.exists():
                    mesh = read_obj(p)
            if mesh is None and labels_volume is not None and r["stack_name"] == labels_stack:
                try:
                    mesh = mesh_from_labels(labels_volume, int(r["label"]), spacing, bboxes=bboxes)
                except ValueError:
                    mesh = None
            if mesh is not None and len(mesh[0]) and len(mesh[1]):
                meshes.append(mesh)
        if meshes:
            rows.append((c, meshes))
    if not rows:
        return None
    allv = np.vstack([v for _, ms in rows for v, _ in ms])
    half = float((allv.max(axis=0) - allv.min(axis=0)).max()) / 2 or 1.0
    ncols = max(len(ms) for _, ms in rows)
    fig = plt.figure(figsize=(2.2 * ncols, 2.2 * len(rows)))
    for i, (c, ms) in enumerate(rows):
        for j, (v, f) in enumerate(ms):
            ax = fig.add_subplot(len(rows), ncols, i * ncols + j + 1, projection="3d")
            ax.add_collection3d(Poly3DCollection(v[f][:, :, ::-1], facecolor=palette.get(c, "#999999"),
                                                 edgecolor="none", alpha=0.95, linewidths=0))
            ctr = v.mean(axis=0)[::-1]
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            ax.set_zlim(ctr[2] - half, ctr[2] + half)
            ax.set_axis_off()
            if j == 0:
                ax.set_title(f"{title_prefix}{c}", fontsize=8, loc="left")
    fig.tight_layout()
    return fig


def write_json(path: PathLike, obj: dict) -> None:
    """Write a settings dictionary as readable JSON (numpy values allowed)."""
    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


# ============================================================================
# Plot kit (matplotlib only, so everything runs in the base install)
# ============================================================================

#: Type sizes (points).  One ladder for every figure, so panels from different
#: notebooks look alike when you put them side by side.
FONT_TITLE = 9
FONT_AXIS = 8
FONT_TICK = 7
FONT_LEGEND = 7
FONT_BADGE = 7
FONT_FAMILIES = ["Arial", "Helvetica", "DejaVu Sans"]

#: Figure size presets in millimetres (width, height).
FIGURE_SIZES_MM: Dict[str, Tuple[float, float]] = {
    "half": PANEL_HALF_MM,
    "full": PANEL_FULL_MM,
    "a4_portrait": A4_PORTRAIT_MM,
    "a4_landscape": A4_LANDSCAPE_MM,
}

NA_COLOUR = "#D9D9D9"


def figure_size(preset: Union[str, Tuple[float, float]] = "full", scale: float = 1.0) -> Tuple[float, float]:
    """Figure size in inches for a preset name ('half', 'full', 'a4_portrait',
    'a4_landscape') or a (width_mm, height_mm) pair."""
    mm = FIGURE_SIZES_MM[preset] if isinstance(preset, str) else tuple(preset)
    return (mm[0] / 25.4 * scale, mm[1] / 25.4 * scale)


def apply_style() -> None:
    """Set the shared look: small sans-serif type, no top/right frame, real text in SVG."""
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILIES,
        "font.size": FONT_TICK,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "figure.dpi": 100,
    })


def fmt_value(v) -> str:
    """Print 3.0 as '3' and everything else as text."""
    try:
        if float(v) == int(float(v)):
            return str(int(float(v)))
    except (TypeError, ValueError, OverflowError):
        pass
    return str(v)


def _new_axes(ax, size="full"):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figure_size(size))
    return fig, ax


def _umap_axes(ax, x_label="UMAP 1", y_label="UMAP 2", title: str = ""):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(x_label, fontsize=FONT_AXIS)
    ax.set_ylabel(y_label, fontsize=FONT_AXIS)
    if title:
        ax.set_title(title, fontsize=FONT_TITLE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend(ax, palette: Dict, counts: Optional[Dict] = None, title: Optional[str] = None, max_items: int = 30):
    from matplotlib.lines import Line2D

    if len(palette) > max_items:
        return
    handles = []
    for k, c in palette.items():
        lab = fmt_value(k) + (f" (n={counts[k]:,})" if counts and k in counts else "")
        handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c, markeredgecolor=c,
                              markersize=4, label=lab))
    ax.legend(handles=handles, title=title, fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND, frameon=False,
              ncol=2 if len(handles) > 14 else 1, loc="center left", bbox_to_anchor=(1.0, 0.5))


def apply_transform(values, method: str = "raw", winsorize_sigma: float = 5.0) -> np.ndarray:
    """Re-express a feature before colouring or plotting it (NaN stays NaN).

    ``method``: ``raw`` (unchanged), ``log1p``, ``sqrt``, ``rank`` (0 to 1, good for
    skewed features), ``zscore`` (mean 0, spread 1) or ``winsorize`` (clip beyond
    ``winsorize_sigma`` standard deviations).  ``log1p`` and ``sqrt`` shift the
    values up first if any are negative.
    """
    arr = np.asarray(values, dtype=float).copy()
    ok = np.isfinite(arr)
    if not ok.any() or method == "raw":
        return arr
    v = arr[ok]
    if method in ("log", "log1p", "sqrt"):
        if v.min() < 0:
            v = v - v.min()
        arr[ok] = np.sqrt(v) if method == "sqrt" else np.log1p(v)
    elif method == "rank":
        from scipy.stats import rankdata
        arr[ok] = rankdata(v) / len(v)
    elif method in ("zscore", "winsorize"):
        sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
        if sd > 0:
            z = (v - v.mean()) / sd
            if method == "winsorize":
                z = np.clip(z, -winsorize_sigma, winsorize_sigma) * sd + v.mean()
            arr[ok] = z
    else:
        raise ValueError(f"Unknown transform {method!r}. Use raw, log1p, sqrt, rank, zscore or winsorize.")
    return arr


def umap_scatter(df: pd.DataFrame, x: str, y: str, color_by: str, ax=None, kind: str = "auto",
                 palette: Optional[Dict] = None, order: Optional[Sequence] = None, transform: str = "raw",
                 cmap: str = "viridis", s: float = 3, alpha: float = 0.7, title: Optional[str] = None,
                 legend: bool = True, legend_counts: bool = False, seed: int = 0, size="full"):
    """A 2D map (UMAP or any two columns) coloured by one column.

    ``kind``: ``'categorical'`` (one colour per value, legend), ``'continuous'``
    (colour bar; ``transform`` re-expresses the numbers first, see
    ``apply_transform``) or ``'auto'`` (text and few-valued columns count as
    categorical).  Cells with a missing value are drawn first, in light grey.
    Draw order is shuffled so no group hides the others.  Returns ``(ax, palette)``;
    the palette is ``None`` for a continuous colouring.
    """
    fig, ax = _new_axes(ax, size)
    d = df[[x, y, color_by]].dropna(subset=[x, y])
    if kind == "auto":
        kind = column_kind(d[color_by])
    rng = np.random.default_rng(seed)
    title = color_by if title is None else title
    if kind in ("continuous", "numeric"):
        vals = apply_transform(d[color_by].to_numpy(dtype=float), transform)
        na = ~np.isfinite(vals)
        if na.any():
            ax.scatter(d.loc[na, x], d.loc[na, y], c=NA_COLOUR, s=s, linewidths=0, rasterized=True)
        sel = np.where(~na)[0]
        sel = sel[rng.permutation(len(sel))]
        sc = ax.scatter(d[x].to_numpy()[sel], d[y].to_numpy()[sel], c=vals[sel], cmap=cmap, s=s, alpha=alpha,
                        linewidths=0, rasterized=True)
        cb = ax.figure.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(color_by if transform == "raw" else f"{color_by} ({transform})", fontsize=FONT_TICK)
        cb.ax.tick_params(labelsize=FONT_TICK)
        _umap_axes(ax, title=title)
        return ax, None
    values = d[color_by]
    present = list(order) if order is not None else sorted(pd.unique(values.dropna()))
    palette = dict(palette) if palette else assign_palette(present, order=present)
    na = values.isna().to_numpy()
    if na.any():
        ax.scatter(d.loc[na, x], d.loc[na, y], c=NA_COLOUR, s=s, linewidths=0, rasterized=True)
    sel = np.where(~na)[0]
    sel = sel[rng.permutation(len(sel))]
    cols = [palette.get(v, "#999999") for v in values.to_numpy()[sel]]
    ax.scatter(d[x].to_numpy()[sel], d[y].to_numpy()[sel], c=cols, s=s, alpha=alpha, linewidths=0, rasterized=True)
    _umap_axes(ax, title=title)
    if legend:
        counts = values.value_counts().to_dict() if legend_counts else None
        _legend(ax, {k: palette[k] for k in present if k in palette}, counts)
    return ax, palette


def umap_highlight(df: pd.DataFrame, x: str, y: str, group_col: str, groups: Sequence, ax=None,
                   palette: Optional[Dict] = None, on_top: Optional[Sequence] = None, bg: str = "#DADADA",
                   s: float = 3, alpha: float = 0.8, title: Optional[str] = None, size="full"):
    """All cells in grey, the chosen ``groups`` of ``group_col`` in colour on top."""
    fig, ax = _new_axes(ax, size)
    d = df.dropna(subset=[x, y])
    groups = list(groups)
    palette = palette or assign_palette(groups, order=groups)
    ax.scatter(d[x], d[y], c=bg, s=s, linewidths=0, alpha=0.5, rasterized=True)
    ordered = [g for g in groups if g not in (on_top or [])] + [g for g in (on_top or []) if g in groups]
    for g in ordered:
        sub = d[d[group_col] == g]
        ax.scatter(sub[x], sub[y], c=palette.get(g, "#D55E00"), s=s * 1.6, alpha=alpha, linewidths=0,
                   rasterized=True, label=f"{fmt_value(g)} (n={len(sub):,})")
    _umap_axes(ax, title=group_col if title is None else title)
    ax.legend(fontsize=FONT_LEGEND, frameon=False, markerscale=2.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return ax, palette


def highlight_grid(df: pd.DataFrame, x: str, y: str, group_col: str, groups: Sequence,
                   palette: Optional[Dict] = None, ncols: int = 3, s: float = 2.5, panel_mm=(60, 55)):
    """One small map per group: that group in colour, every other cell in grey."""
    import matplotlib.pyplot as plt

    groups = list(groups)
    palette = palette or assign_palette(groups, order=groups)
    ncols = max(1, min(ncols, len(groups)))
    nrows = int(np.ceil(len(groups) / ncols))
    w, hgt = panel_mm[0] / 25.4, panel_mm[1] / 25.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, hgt * nrows), squeeze=False)
    d = df.dropna(subset=[x, y])
    for ax, g in zip(axes.ravel(), groups):
        ax.scatter(d[x], d[y], c="#DADADA", s=s, linewidths=0, alpha=0.5, rasterized=True)
        sub = d[d[group_col] == g]
        ax.scatter(sub[x], sub[y], c=palette.get(g, "#D55E00"), s=s * 1.6, linewidths=0, alpha=0.85, rasterized=True)
        _umap_axes(ax, x_label="", y_label="", title=f"{fmt_value(g)} (n={len(sub):,})")
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def _readable_text(hex_colour: str) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "#1A1A1A" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "#FFFFFF"


def umap_with_medians(df: pd.DataFrame, x: str, y: str, cluster_col: str, palette: Optional[Dict] = None, ax=None,
                      s: float = 3, bg_alpha: float = 0.45, badge_size: float = 190, prefix: str = "",
                      title: Optional[str] = None, size="full"):
    """Cells coloured by cluster (faint) with one numbered badge at each cluster's median."""
    fig, ax = _new_axes(ax, size)
    d = df.dropna(subset=[x, y, cluster_col])
    clusters = sorted(pd.unique(d[cluster_col]))
    palette = palette or assign_palette(clusters, order=clusters)
    sel = np.random.default_rng(0).permutation(len(d))
    ax.scatter(d[x].to_numpy()[sel], d[y].to_numpy()[sel],
               c=[palette.get(v, "#999999") for v in d[cluster_col].to_numpy()[sel]],
               s=s, alpha=bg_alpha, linewidths=0, rasterized=True)
    for c in clusters:
        sub = d[d[cluster_col] == c]
        col = palette.get(c, "#999999")
        ax.scatter([sub[x].median()], [sub[y].median()], s=badge_size, c=col, edgecolors="white", linewidths=1.2,
                   zorder=5)
        ax.text(sub[x].median(), sub[y].median(), f"{prefix}{fmt_value(c)}", ha="center", va="center",
                fontsize=FONT_BADGE, color=_readable_text(col), zorder=6, fontweight="bold")
    _umap_axes(ax, title=cluster_col if title is None else title)
    return ax, palette


def median_cells(df: pd.DataFrame, pc_cols: Sequence[str], cluster_col: str, x: str = "umap_1",
                 y: str = "umap_2") -> pd.DataFrame:
    """The single cell closest to each cluster's median in PC space, with its map position."""
    rows = []
    pcs = df[list(pc_cols)].to_numpy(dtype=float)
    cl = df[cluster_col].to_numpy()
    for c in sorted(pd.unique(df[cluster_col].dropna())):
        idx = np.where(cl == c)[0]
        d = np.linalg.norm(pcs[idx] - np.nanmedian(pcs[idx], axis=0), axis=1)
        r = df.iloc[idx[int(np.nanargmin(d))]]
        rows.append({"cluster": c, "stack_name": r["stack_name"], "label": int(r["label"]),
                     x: r[x], y: r[y], "dist_to_median": float(np.nanmin(d))})
    return pd.DataFrame(rows)


def umap_ghost(df: pd.DataFrame, x: str, y: str, cluster_col: str, medians: pd.DataFrame,
               palette: Optional[Dict] = None, ax=None, bg_alpha: float = 0.25, s: float = 3,
               badge_size: float = 150, prefix: str = "", size="full"):
    """A map for finishing in a vector editor.

    All cells are faint grey.  Only the typical cell of every cluster
    (``medians``, from ``median_cells``) is drawn as a coloured, outlined marker
    with a label.  In the SVG every marker and label has its own id, so you can
    move or replace them (for example with a picture of the cell) in Inkscape or
    Illustrator.
    """
    fig, ax = _new_axes(ax, size)
    d = df.dropna(subset=[x, y])
    clusters = sorted(pd.unique(df[cluster_col].dropna()))
    palette = palette or assign_palette(clusters, order=clusters)
    bg = ax.scatter(d[x], d[y], c="#BBBBBB", s=s, alpha=bg_alpha, linewidths=0, rasterized=True)
    bg.set_gid("background_cells")
    for _, r in medians.iterrows():
        col = palette.get(r["cluster"], "#999999")
        name = f"{prefix}{fmt_value(r['cluster'])}"
        mk = ax.scatter([r[x]], [r[y]], s=badge_size, c=col, edgecolors="#111111", linewidths=1.6, marker="D",
                        zorder=5)
        mk.set_gid(f"marker_{name}")
        tx = ax.text(r[x], r[y], f"  {name}", ha="left", va="center", fontsize=FONT_AXIS, color="#222222", zorder=6)
        tx.set_gid(f"label_{name}")
    _umap_axes(ax, title="")
    return ax, palette


def violin_panel(df: pd.DataFrame, feature: str, group_col: str, ax=None, palette: Optional[Dict] = None,
                 order: Optional[Sequence] = None, transform: str = "raw", show_points: bool = False,
                 title: Optional[str] = None, size="full", rotate: int = 0):
    """Distribution of one feature in every group: violin, median dot and inter-quartile bar."""
    fig, ax = _new_axes(ax, size)
    d = df[[feature, group_col]].dropna(subset=[group_col])
    d = d.assign(_v=apply_transform(d[feature].to_numpy(dtype=float), transform))
    groups = list(order) if order is not None else sorted(pd.unique(d[group_col]))
    palette = palette or assign_palette(groups, order=groups)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        v = d.loc[d[group_col] == g, "_v"].dropna().to_numpy()
        if len(v) == 0:
            continue
        col = palette.get(g, "#999999")
        if len(v) >= 3 and np.ptp(v) > 0:
            parts = ax.violinplot([v], positions=[i], showextrema=False, widths=0.85)
            for body in parts["bodies"]:
                body.set_facecolor(col)
                body.set_edgecolor("none")
                body.set_alpha(0.75)
        if show_points:
            n = min(len(v), 300)
            ax.scatter(i + rng.uniform(-0.15, 0.15, n), rng.choice(v, n, replace=False), s=1.5, c="#333333",
                       alpha=0.3, linewidths=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q1, q3, color="#222222", lw=2.2)
        ax.scatter([i], [med], s=14, c="white", edgecolors="#222222", linewidths=0.8, zorder=5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([fmt_value(g) for g in groups], fontsize=FONT_TICK, rotation=rotate,
                       ha="right" if rotate else "center")
    ax.set_xlabel(group_col, fontsize=FONT_AXIS)
    ax.set_ylabel(feature if transform == "raw" else f"{feature} ({transform})", fontsize=FONT_AXIS)
    ax.set_title(feature if title is None else title, fontsize=FONT_TITLE)
    return ax


def violin_grid(df: pd.DataFrame, features: Sequence[str], group_col: str, palette: Optional[Dict] = None,
                order: Optional[Sequence] = None, ncols: int = 4, transforms: Optional[Dict[str, str]] = None,
                panel_mm=(55, 48), rotate: int = 0):
    """``violin_panel`` for several features in one figure."""
    import matplotlib.pyplot as plt

    features = [f for f in features if f in df.columns]
    ncols = max(1, min(ncols, len(features)))
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_mm[0] / 25.4 * ncols, panel_mm[1] / 25.4 * nrows),
                             squeeze=False)
    for ax, f in zip(axes.ravel(), features):
        violin_panel(df, f, group_col, ax=ax, palette=palette, order=order,
                     transform=(transforms or {}).get(f, "raw"), rotate=rotate)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def composition_table(df: pd.DataFrame, group_col: str, cluster_col: str, normalize: bool = True) -> pd.DataFrame:
    """Cells per (group, cluster).  With ``normalize`` each row sums to 1 (shares within the group)."""
    d = df.dropna(subset=[group_col, cluster_col])
    tab = pd.crosstab(d[group_col], d[cluster_col])
    return tab.div(tab.sum(axis=1), axis=0) if normalize else tab


def composition_bar(df: pd.DataFrame, group_col: str, cluster_col: str, ax=None, palette: Optional[Dict] = None,
                    group_order: Optional[Sequence] = None, normalize: bool = True, horizontal: bool = False,
                    size="full"):
    """Stacked bars: for every group, which share of its cells is in each cluster."""
    fig, ax = _new_axes(ax, size)
    tab = composition_table(df, group_col, cluster_col, normalize)
    if group_order is not None:
        tab = tab.reindex([g for g in group_order if g in tab.index])
    palette = palette or assign_palette(list(tab.columns), order=list(tab.columns))
    pos = np.arange(len(tab))
    base = np.zeros(len(tab))
    for c in tab.columns:
        vals = tab[c].to_numpy(dtype=float)
        if horizontal:
            ax.barh(pos, vals, left=base, color=palette.get(c, "#999999"), edgecolor="white", linewidth=0.4,
                    label=fmt_value(c))
        else:
            ax.bar(pos, vals, bottom=base, color=palette.get(c, "#999999"), edgecolor="white", linewidth=0.4,
                   label=fmt_value(c))
        base += vals
    labels = [fmt_value(g) for g in tab.index]
    if horizontal:
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=FONT_TICK)
        ax.set_xlabel("share of cells" if normalize else "cells", fontsize=FONT_AXIS)
        ax.invert_yaxis()
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=FONT_TICK, rotation=45 if len(labels) > 6 else 0,
                           ha="right" if len(labels) > 6 else "center")
        ax.set_ylabel("share of cells" if normalize else "cells", fontsize=FONT_AXIS)
    ax.legend(title=cluster_col, fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND, frameon=False,
              ncol=2 if len(tab.columns) > 14 else 1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return ax


def enrichment_heatmap(df: pd.DataFrame, row_col: str, col_col: str, ax=None, vlim: float = 3.0,
                       annotate: bool = True, size="full"):
    """log2(observed / expected) of every (row group, column group) pair.

    Rows are the values of ``row_col``, columns those of ``col_col``.  Zero =
    as often as chance, +1 = twice as often, -1 = half as often.  Pairs that
    never occur are drawn grey and labelled "none" (the returned table holds
    -3 for them).  Returns ``(ax, table)``.
    """
    d = df.dropna(subset=[row_col, col_col])
    table = enrichment_log2(d[row_col].to_numpy(), d[col_col].to_numpy())
    counts = pd.crosstab(d[row_col].to_numpy(), d[col_col].to_numpy()).reindex(index=table.index, columns=table.columns)
    never = counts.to_numpy() == 0
    fig, ax = _new_axes(ax, size)
    cmap_obj = __import__("matplotlib").colormaps["RdBu_r"].copy()
    cmap_obj.set_bad("#BBBBBB")
    im = ax.imshow(np.ma.masked_where(never, table.to_numpy()), cmap=cmap_obj, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([fmt_value(c) for c in table.columns], fontsize=FONT_TICK)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([fmt_value(c) for c in table.index], fontsize=FONT_TICK)
    ax.set_xlabel(col_col, fontsize=FONT_AXIS)
    ax.set_ylabel(row_col, fontsize=FONT_AXIS)
    if annotate and table.size <= 400:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.iat[i, j]
                if never[i, j]:
                    ax.text(j, i, "none", ha="center", va="center", fontsize=FONT_TICK - 1, color="#444444")
                else:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=FONT_TICK - 1,
                            color="white" if abs(v) > vlim * 0.6 else "#222222")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("log2(observed / expected); grey = never occurs", fontsize=FONT_TICK)
    cb.ax.tick_params(labelsize=FONT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return ax, table


def replicate_dotplot(df: pd.DataFrame, group_col: str, value_col: Optional[str] = None, ax=None,
                      stack_col: str = "stack_name", groups: Optional[Sequence] = None,
                      palette: Optional[Dict] = None, min_cells: int = 10, size="full"):
    """One dot per stack, to see whether a result repeats across replicates.

    Without ``value_col``: for every group, the share of each stack's cells that
    belong to it (a stack without any cell of the group shows a dot at 0).  With ``value_col``: the median of that feature among the
    group's cells in each stack (stacks with fewer than ``min_cells`` cells in
    the group are left out).  Only meaningful when the table has several
    stacks; otherwise a message is printed and ``(None, None)`` returned.
    Returns ``(ax, per-stack table)``.
    """
    if df[stack_col].nunique() < 2:
        print("Only one stack in the table: a replicate plot needs at least two.")
        return None, None
    d = df.dropna(subset=[group_col])
    groups = list(groups) if groups is not None else sorted(pd.unique(d[group_col]))
    palette = palette or assign_palette(groups, order=groups)
    rows = []
    totals = df.groupby(stack_col).size()
    for g in groups:
        sub = d[d[group_col] == g]
        n_in = sub.groupby(stack_col).size()
        for stack in totals.index:
            n = int(n_in.get(stack, 0))
            if value_col is None:
                rows.append({"group": g, "stack_name": stack, "value": n / totals[stack], "n": n})
            elif n >= min_cells:
                part = sub[sub[stack_col] == stack]
                rows.append({"group": g, "stack_name": stack, "value": float(part[value_col].median()), "n": n})
    tab = pd.DataFrame(rows, columns=["group", "stack_name", "value", "n"])
    fig, ax = _new_axes(ax, size)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        t = tab[tab["group"] == g]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(t)), t["value"], s=18, c=palette.get(g, "#999999"),
                   edgecolors="#222222", linewidths=0.5, zorder=3)
        if len(t):
            ax.hlines(t["value"].mean(), i - 0.3, i + 0.3, color="#222222", lw=1.4, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([fmt_value(g) for g in groups], fontsize=FONT_TICK)
    ax.set_xlabel(group_col, fontsize=FONT_AXIS)
    ax.set_ylabel("share of the stack's cells" if value_col is None else f"{value_col} (median per stack)",
                  fontsize=FONT_AXIS)
    note = "" if value_col is None else f"; stacks with fewer than {min_cells} cells left out"
    ax.set_title(f"One dot per stack ({df[stack_col].nunique()} stacks); bar = mean{note}", fontsize=FONT_TITLE)
    return ax, tab


def matrix_heatmap(table: pd.DataFrame, ax=None, vmin: float = 0.0, vmax: Optional[float] = None,
                   cmap: str = "Blues", cbar_label: str = "", annotate: bool = True, size="full"):
    """A labelled heatmap of any small table (for example PC separation or stack overlap)."""
    fig, ax = _new_axes(ax, size)
    im = ax.imshow(table.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([fmt_value(c) for c in table.columns], fontsize=FONT_TICK, rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([fmt_value(c) for c in table.index], fontsize=FONT_TICK)
    if annotate and table.size <= 400:
        top = np.nanmax(table.to_numpy(dtype=float)) if vmax is None else vmax
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FONT_TICK - 1,
                        color="white" if v > 0.6 * top else "#222222")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(cbar_label, fontsize=FONT_TICK)
    cb.ax.tick_params(labelsize=FONT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return ax


# ============================================================================
# Groups of cells, statistics and "do replicates mix?" (for the comparison notebook)
# ============================================================================


def mask_from_column(df: pd.DataFrame, column: str, values: Optional[Sequence]) -> np.ndarray:
    """True for cells whose ``column`` equals one of ``values`` (text and numbers both work)."""
    if column not in df.columns:
        raise KeyError(f"Column {column!r} is not in the table.")
    if values is None or len(values) == 0:
        return np.zeros(len(df), dtype=bool)
    col = df[column]
    vals = list(values)
    mask = col.isin(vals)
    if int(mask.sum()) == 0 or any(not col.isin([v]).any() for v in vals):
        # a requested value did not match as typed: also compare as text ("2" against 2)
        mask = mask | col.astype(str).isin([str(v) for v in vals])
    return mask.to_numpy()


def mask_from_csv(df: pd.DataFrame, csv: PathLike, group_name: Optional[str]) -> np.ndarray:
    """True for cells listed in a CSV with the columns ``stack_name, label, group`` under ``group_name``."""
    if not group_name:
        return np.zeros(len(df), dtype=bool)
    table = pd.read_csv(csv)
    need = {"stack_name", "label", "group"}
    if not need.issubset(table.columns):
        raise ValueError(f"The group file needs the columns {sorted(need)}; found {list(table.columns)}.")
    keep = table[table["group"].astype(str) == str(group_name)]
    if keep.empty:
        print(f"Note: no row of the group file has group = {group_name!r}.")
    key = pd.MultiIndex.from_arrays([df["stack_name"].astype(str), df["label"].astype(int)])
    want = pd.MultiIndex.from_arrays([keep["stack_name"].astype(str), keep["label"].astype(int)])
    return np.asarray(key.isin(want))


def mask_from_rule(df: pd.DataFrame, rule: Optional[str]) -> np.ndarray:
    """True for cells matching a rule such as ``"`centroid-0` < 20"`` or ``"area > 500 and sphericity < 0.6"``."""
    if not rule:
        return np.zeros(len(df), dtype=bool)
    res = df.eval(rule, engine="python")
    res = np.asarray(res)
    if res.dtype != bool:
        raise ValueError(f"The rule {rule!r} must be a comparison such as \"area > 500\" (it gave {res.dtype} values).")
    return res


def build_groups(df: pd.DataFrame, mode: str, spec: Dict[str, object], column: Optional[str] = None,
                 csv: Optional[PathLike] = None) -> pd.Series:
    """Label every cell with the role it plays (or NaN).

    ``spec`` maps a role name (for example ``"A"``, ``"A_ref"``) to what defines it:
    values of ``column`` (mode ``'column'``), a group name in ``csv`` (mode
    ``'csv'``) or a rule string (mode ``'rule'``).  Roles set to ``None`` are
    left out.  A cell that belongs to more than one role is an error: groups
    must not overlap.
    """
    masks = {}
    for role, how in spec.items():
        if how is None:
            continue
        if mode == "column":
            masks[role] = mask_from_column(df, column, how)
        elif mode == "csv":
            masks[role] = mask_from_csv(df, csv, how)
        elif mode == "rule":
            masks[role] = mask_from_rule(df, how)
        else:
            raise ValueError("mode must be 'column', 'csv' or 'rule'.")
    roles = list(masks)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            if (masks[a] & masks[b]).any():
                raise ValueError(f"The groups {a!r} and {b!r} share cells, which is not allowed. "
                                 "Make their definitions disjoint.")
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for role, m in masks.items():
        out[m] = role
    return out


def cliffs_delta(a, b) -> float:
    """Cliff's delta = P(a > b) - P(a < b), from -1 to 1 (0 = no shift).

    Computed exactly from sorted values.  It depends only on the ranks, so a
    log transform of the feature does not change it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    b = np.sort(b)
    below = np.searchsorted(b, a, side="left")
    above = len(b) - np.searchsorted(b, a, side="right")
    return float((below.sum() - above.sum()) / (len(a) * len(b)))


def magnitude(delta: float) -> str:
    """Words for |delta| (Romano et al. 2006): negligible, small, medium, large."""
    d = abs(delta)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def delta_with_ci(a, b, n_boot: int = 1000, seed: int = 42, ci: float = 95.0,
                  max_cells: Optional[int] = None) -> Tuple[float, float, float]:
    """Cliff's delta with a bootstrap confidence interval.

    Cells are resampled (with replacement) within each group ``n_boot`` times.
    The interval therefore reflects the sampling of cells only, not differences
    between replicate stacks.  The estimate itself uses every cell.  With
    ``max_cells`` set, each group is first cut to at most that many randomly
    chosen cells for the bootstrap only (the interval gets a little wider, which
    is the honest direction) so that very large groups stay fast.
    Returns ``(delta, low, high)``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    est = cliffs_delta(a, b)
    if max_cells is not None:
        if len(a) > max_cells:
            a = a[rng.choice(len(a), max_cells, replace=False)]
        if len(b) > max_cells:
            b = b[rng.choice(len(b), max_cells, replace=False)]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = cliffs_delta(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return est, float(lo), float(hi)


def bootstrap_time_estimate(n_a: int, n_b: int, n_features: int, n_tables: int, n_boot: int,
                            max_cells: Optional[int] = None) -> float:
    """Rough number of seconds the bootstrap tables will take on this computer.

    Times a few resamples at the (capped) group sizes and scales up.
    """
    import time

    na = min(n_a, max_cells) if max_cells else n_a
    nb = min(n_b, max_cells) if max_cells else n_b
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=na), rng.normal(size=nb)
    t0 = time.perf_counter()
    for _ in range(3):
        cliffs_delta(a[rng.integers(0, na, na)], b[rng.integers(0, nb, nb)])
    return (time.perf_counter() - t0) / 3 * n_boot * n_features * n_tables


def delta_table(df: pd.DataFrame, group_col: str, group_a, group_b, features: Sequence[str],
                n_boot: int = 1000, seed: int = 42, max_cells: Optional[int] = None) -> pd.DataFrame:
    """Cliff's delta (``group_a`` vs ``group_b``) with bootstrap interval, one row per feature.

    Positive delta = the cells of ``group_a`` have larger values.  Columns:
    ``delta, ci_low, ci_high, n_a, n_b, size`` (``size`` in words).
    """
    A, B = df[df[group_col] == group_a], df[df[group_col] == group_b]
    rows = {}
    for f in features:
        d, lo, hi = delta_with_ci(A[f], B[f], n_boot=n_boot, seed=seed, max_cells=max_cells)
        rows[f] = {"delta": d, "ci_low": lo, "ci_high": hi, "n_a": int(A[f].notna().sum()),
                   "n_b": int(B[f].notna().sum()), "size": magnitude(d) if np.isfinite(d) else ""}
    return pd.DataFrame(rows).T.astype({"delta": float, "ci_low": float, "ci_high": float, "n_a": int, "n_b": int})


def ci_overlap(t1: pd.DataFrame, t2: pd.DataFrame) -> pd.Series:
    """For every feature: do the two tables' confidence intervals overlap?

    Overlapping intervals do not show that the two effects are equal, and
    separate intervals are a stricter sign than a difference needs.  Use it
    as a rough guide, not as a test.
    """
    return ~((t1["ci_low"] > t2["ci_high"]) | (t2["ci_low"] > t1["ci_high"]))


def departure_plot(dep_a: pd.DataFrame, dep_b: Optional[pd.DataFrame], name_a: str, name_b: str = "",
                   col_a: str = "#0072B2", col_b: str = "#D55E00", ax=None, xlabel: str = "Cliff's delta"):
    """Paired lollipop plot: one row per feature, delta with its interval, zero line.

    ``dep_a`` / ``dep_b`` come from ``delta_table`` (same features).  ``dep_b``
    may be None.  Dotted lines mark the small / medium / large boundaries.
    """
    import matplotlib.pyplot as plt

    feats = list(dep_a.index)
    if ax is None:
        fig, ax = plt.subplots(figsize=(figure_size("half")[0] * 1.6, max(3.0, 0.28 * len(feats) + 1.2)))
    y = np.arange(len(feats))[::-1]
    series = [(dep_a, name_a, col_a, 0.15 if dep_b is not None else 0.0)]
    if dep_b is not None:
        series.append((dep_b, name_b, col_b, -0.15))
    for dep, name, col, dy in series:
        d = dep.loc[feats]
        ax.hlines(y + dy, 0, d["delta"], color=col, lw=1.2)
        ax.errorbar(d["delta"], y + dy, xerr=[(d["delta"] - d["ci_low"]).clip(lower=0),
                                              (d["ci_high"] - d["delta"]).clip(lower=0)],
                    fmt="o", ms=4, color=col, ecolor=col, elinewidth=1.0, capsize=2, label=name,
                    markeredgecolor="white", markeredgewidth=0.5)
    ax.axvline(0, color="#555555", lw=0.9)
    for xv in (-0.474, -0.147, 0.147, 0.474):
        ax.axvline(xv, color="#CCCCCC", lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=FONT_TICK)
    ax.set_xlim(-1.02, 1.02)
    ax.set_xlabel(xlabel, fontsize=FONT_AXIS)
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return ax


def pc_separation(df: pd.DataFrame, group_col: str, groups: Sequence, pc_cols: Sequence[str]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """How far apart are groups in PC space, compared with how spread out each is?

    ``raw``: distance between group medians (coordinate by coordinate).
    ``normalised``: the same divided by the average of the two groups' spread
    (median distance of a group's cells to its own median).  The value grows
    with the number of PCs used, so it has no fixed meaning: compare only pairs
    computed with the same set of PCs, for example the pair of groups against
    the pair of their references.  Returns ``(raw, normalised, spread)``.
    """
    groups = list(groups)
    X = {g: df.loc[df[group_col] == g, list(pc_cols)].dropna().to_numpy(dtype=float) for g in groups}
    med = {g: np.median(X[g], axis=0) for g in groups}
    spread = {g: float(np.median(np.linalg.norm(X[g] - med[g], axis=1))) for g in groups}
    raw = pd.DataFrame([[np.linalg.norm(med[a] - med[b]) for b in groups] for a in groups], index=groups,
                       columns=groups)
    norm = pd.DataFrame([[raw.loc[a, b] / (0.5 * (spread[a] + spread[b]) + 1e-12) for b in groups]
                         for a in groups], index=groups, columns=groups)
    return raw, norm, spread


def summary_by_group(df: pd.DataFrame, group_col: str, groups: Sequence, features: Sequence[str]) -> pd.DataFrame:
    """n, median and inter-quartile range of every feature in every group (long table)."""
    rows = []
    for g in groups:
        sub = df[df[group_col] == g]
        for f in features:
            v = sub[f].dropna()
            rows.append({"group": g, "feature": f, "n": len(v), "median": v.median(),
                         "q1": v.quantile(0.25), "q3": v.quantile(0.75)})
    return pd.DataFrame(rows)


def cluster_overview(df: pd.DataFrame, cluster_col: str, features: Sequence[str] = ()) -> pd.DataFrame:
    """One row per cluster: cells, share, stacks it occurs in, and the median of a few features."""
    feats = [f for f in features if f in df.columns]
    g = df.dropna(subset=[cluster_col]).groupby(cluster_col)
    out = pd.DataFrame({"n_cells": g.size(), "share": g.size() / g.size().sum(),
                        "n_stacks": g["stack_name"].nunique()})
    if feats:
        out = out.join(g[feats].median().add_suffix("_median"))
    out.index.name = "cluster"
    return out.reset_index()


def stack_mixing_score(X: np.ndarray, stack_labels: Sequence, k: int = 15, n_perm: int = 5, seed: int = 42):
    """Does the embedding mix replicate stacks?

    For every cell: the fraction of its ``k`` nearest neighbours (Euclidean, the
    cell itself left out) that come from the same stack.  Low = stacks mix; high
    = cells sit with their own stack.  Also returns the same fraction after
    shuffling the stack labels (the value to expect if stacks mixed perfectly).
    Returns ``(score, baseline)``, one value per row (NaN for rows with a
    missing value).
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    labels = np.asarray(stack_labels)
    ok = np.isfinite(X).all(axis=1)
    k = min(k, int(ok.sum()) - 1)
    idx = NearestNeighbors(n_neighbors=k).fit(X[ok]).kneighbors(return_distance=False)
    lab = labels[ok]
    score = np.full(len(X), np.nan)
    score[ok] = (lab[idx] == lab[:, None]).mean(axis=1)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(lab) for _ in range(n_perm)]
    base = np.full(len(X), np.nan)
    base[ok] = np.mean([(p[idx] == p[:, None]).mean(axis=1) for p in perms], axis=0)
    return score, base


def cross_group_agreement(X: np.ndarray, cls: Sequence, group: Sequence, k: int = 50) -> pd.DataFrame:
    """Do cells of the same class sit together across replicates?

    For every cell, take its ``k`` nearest neighbours and keep those that come
    from a DIFFERENT group (another stack): how often do they share the cell's
    class (for example its cluster)?  ``expected`` is the same fraction if those
    neighbours were drawn at random from the other groups; ``ratio`` =
    observed / expected (1 = no better than chance).  One row per class.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    cls = np.asarray(cls)
    group = np.asarray(group)
    ok = np.isfinite(X).all(axis=1) & np.asarray(pd.notna(cls))
    X, cls, group = X[ok], cls[ok], group[ok]
    k = min(k, len(X) - 1)
    idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    valid = group[idx] != group[:, None]
    match = (cls[idx] == cls[:, None]) & valid
    n_valid = valid.sum(axis=1)
    have = n_valid > 0
    obs = np.where(have, match.sum(axis=1) / np.maximum(n_valid, 1), np.nan)
    cls_codes, cls_idx = np.unique(cls, return_inverse=True)
    grp_codes, grp_idx = np.unique(group, return_inverse=True)
    counts = np.zeros((len(grp_codes), len(cls_codes)))
    np.add.at(counts, (grp_idx, cls_idx), 1)
    total_c = counts.sum(axis=0)
    other_n = len(cls) - counts.sum(axis=1)[grp_idx]
    other_c = total_c[cls_idx] - counts[grp_idx, cls_idx]
    exp = np.where(other_n > 0, other_c / np.maximum(other_n, 1), np.nan)
    rows = {}
    for ci, c in enumerate(cls_codes):
        sel = (cls_idx == ci) & have
        rows[c] = {"observed": float(np.nanmean(obs[sel])) if sel.any() else np.nan,
                   "expected": float(np.nanmean(exp[sel])) if sel.any() else np.nan,
                   "cells": int(sel.sum()), "cells_without_such_neighbours": int(((cls_idx == ci) & ~have).sum())}
    out = pd.DataFrame(rows).T
    out["ratio"] = out["observed"] / out["expected"]
    out[["cells", "cells_without_such_neighbours"]] = out[["cells", "cells_without_such_neighbours"]].astype(int)
    return out


def sample_overlap(df: pd.DataFrame, cluster_col: str, stack_col: str = "stack_name") -> pd.DataFrame:
    """How alike are two stacks in their mix of clusters?

    Each stack has a share of its cells in every cluster.  The overlap of two
    stacks is the sum over clusters of the smaller of their two shares: 1 = the
    same mix, 0 = no cluster in common.  Returns a stack by stack table.
    """
    comp = composition_table(df, stack_col, cluster_col, normalize=True)
    P = comp.to_numpy()
    out = np.array([[np.minimum(P[i], P[j]).sum() for j in range(len(P))] for i in range(len(P))])
    return pd.DataFrame(out, index=comp.index, columns=comp.index)
