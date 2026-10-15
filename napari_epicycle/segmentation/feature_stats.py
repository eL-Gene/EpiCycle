"""Pure computation for EDA Workbench: sparklines, skewness, normality, transforms.

No Qt dependency — all functions operate on numpy arrays / pandas Series and
return plain Python dicts.  This keeps the logic testable without a display.

Phase 2: histogram, skewness, normality.
Phase 3: per-feature transforms (log, sqrt, rank, z-score, winsorize, auto).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Histogram (sparkline data)
# ---------------------------------------------------------------------------

def compute_histogram(values: np.ndarray, n_bins: int = 16) -> Dict:
    """Compute histogram bin counts for sparkline rendering.

    Parameters
    ----------
    values : array-like
        1-D numeric array (NaN values are dropped).
    n_bins : int
        Number of bins (default 16).

    Returns
    -------
    dict with keys:
        counts : np.ndarray of int — bin counts
        edges  : np.ndarray of float — bin edges (length n_bins + 1)
        max_count : int — max bin count (for normalization)
    """
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2:
        return {
            "counts": np.zeros(n_bins, dtype=int),
            "edges": np.linspace(0, 1, n_bins + 1),
            "max_count": 0,
        }
    counts, edges = np.histogram(clean, bins=n_bins)
    return {
        "counts": counts,
        "edges": edges,
        "max_count": int(counts.max()),
    }


# ---------------------------------------------------------------------------
# Skewness
# ---------------------------------------------------------------------------

def compute_skewness(values: np.ndarray) -> Dict:
    """Compute Fisher's skewness and classify severity.

    Returns
    -------
    dict with keys:
        value    : float — Fisher's skewness (NaN if insufficient data)
        arrow    : str — "↗" (right), "↙" (left), or "" (symmetric)
        severity : str — "low" (|s|<1), "moderate" (1-2), "high" (>2)
        color    : str — hex color for UI (#6abf69 / #d4a843 / #ff6b6b)
    """
    from scipy.stats import skew as _skew

    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return {"value": float("nan"), "arrow": "", "severity": "low",
                "color": "#6abf69"}

    # Guard: near-constant data causes catastrophic cancellation in moment calc
    if np.ptp(clean) < np.finfo(np.float64).eps * max(1.0, abs(np.mean(clean))):
        return {"value": 0.0, "arrow": "", "severity": "low",
                "color": "#6abf69"}

    s = float(_skew(clean, bias=False))

    abs_s = abs(s)
    if abs_s < 1.0:
        arrow, severity, color = "", "low", "#6abf69"
    elif abs_s < 2.0:
        arrow = "↗" if s > 0 else "↙"
        severity, color = "moderate", "#d4a843"
    else:
        arrow = "↗" if s > 0 else "↙"
        severity, color = "high", "#ff6b6b"

    return {"value": s, "arrow": arrow, "severity": severity, "color": color}


# ---------------------------------------------------------------------------
# Normality
# ---------------------------------------------------------------------------

def compute_normality(values: np.ndarray) -> Dict:
    """Test normality using Shapiro-Wilk (n<5000) or D'Agostino-Pearson.

    Returns
    -------
    dict with keys:
        p_value : float — p-value of the normality test (NaN if n < 8)
        normal  : bool — True if p > 0.05 (can't reject normality)
        symbol  : str — "✓" or "✗"
        color   : str — hex color (#6abf69 green / #ff6b6b red)
        test    : str — "shapiro" or "dagostino"
    """
    from scipy.stats import normaltest, shapiro

    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]

    # Shapiro-Wilk needs n >= 3; D'Agostino needs n >= 8.
    # Use n >= 8 as minimum for meaningful results.
    if len(clean) < 8:
        return {"p_value": float("nan"), "normal": True, "symbol": "—",
                "color": "#999999", "test": "none"}

    # Guard: near-constant data (range ≈ 0) causes unreliable results
    if np.ptp(clean) < np.finfo(np.float64).eps * max(1.0, abs(np.mean(clean))):
        return {"p_value": float("nan"), "normal": True, "symbol": "—",
                "color": "#999999", "test": "constant"}

    if len(clean) < 5000:
        stat, p = shapiro(clean)
        test_name = "shapiro"
    else:
        stat, p = normaltest(clean)
        test_name = "dagostino"

    p = float(p)
    is_normal = p > 0.05
    return {
        "p_value": p,
        "normal": is_normal,
        "symbol": "✓" if is_normal else "✗",
        "color": "#6abf69" if is_normal else "#ff6b6b",
        "test": test_name,
    }


# ---------------------------------------------------------------------------
# Transforms (Phase 3)
# ---------------------------------------------------------------------------

TRANSFORM_METHODS = ("raw", "log", "sqrt", "rank", "zscore", "winsorize", "auto")


def apply_transform(
    values: np.ndarray,
    method: str,
    *,
    feature_name: str = "",
    winsorize_sigma: float = 5.0,
) -> np.ndarray:
    """Apply a transform to a 1-D array of feature values.

    Parameters
    ----------
    values : array-like
        Raw feature values (NaN preserved in-place).
    method : str
        One of TRANSFORM_METHODS.
    feature_name : str
        Used by "auto" to check bounded features.
    winsorize_sigma : float
        Clipping threshold for "winsorize" (default ±5σ for EDA).

    Returns
    -------
    np.ndarray — transformed values (same length, NaN preserved).
    """
    arr = np.asarray(values, dtype=np.float64).copy()
    finite_mask = np.isfinite(arr)

    if method == "raw":
        return arr

    if method == "log":
        # log1p — handles zeros, but needs non-negative.
        # Shift if any negative values present.
        finite = arr[finite_mask]
        if len(finite) > 0 and finite.min() < 0:
            arr[finite_mask] = arr[finite_mask] - finite.min()
        arr[finite_mask] = np.log1p(arr[finite_mask])
        return arr

    if method == "sqrt":
        finite = arr[finite_mask]
        if len(finite) > 0 and finite.min() < 0:
            arr[finite_mask] = arr[finite_mask] - finite.min()
        arr[finite_mask] = np.sqrt(arr[finite_mask])
        return arr

    if method == "rank":
        from scipy.stats import rankdata
        finite = arr[finite_mask]
        if len(finite) > 0:
            arr[finite_mask] = rankdata(finite) / len(finite)
        return arr

    if method == "zscore":
        finite = arr[finite_mask]
        if len(finite) > 1:
            mu = np.mean(finite)
            std = np.std(finite, ddof=1)
            if std > 0:
                arr[finite_mask] = (finite - mu) / std
        return arr

    if method == "winsorize":
        finite = arr[finite_mask]
        if len(finite) > 1:
            mu = np.mean(finite)
            std = np.std(finite, ddof=1)
            if std > 0:
                z = (finite - mu) / std
                z = np.clip(z, -winsorize_sigma, winsorize_sigma)
                arr[finite_mask] = z * std + mu
        return arr

    if method == "auto":
        return _auto_transform(arr, feature_name=feature_name)

    return arr  # unknown method — passthrough


def _auto_transform(
    arr: np.ndarray,
    *,
    feature_name: str = "",
) -> np.ndarray:
    """Apply skewness-based auto-transform rules.

    Rules:
    - Bounded [0,1] features → raw (never log)
    - Right-skewed (skew > 1.0) → log
    - Everything else → raw
    """
    from napari_epicycle.fusion_reconstruction.qc_features import BOUNDED_FEATURES

    if feature_name in BOUNDED_FEATURES:
        return arr

    from scipy.stats import skew as _skew

    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    if len(finite) < 3:
        return arr

    # Guard: near-constant data
    if np.ptp(finite) < np.finfo(np.float64).eps * max(1.0, abs(np.mean(finite))):
        return arr

    s = float(_skew(finite, bias=False))
    if s > 1.0:
        return apply_transform(arr, "log")
    return arr


def resolve_auto_method(
    values: np.ndarray,
    feature_name: str = "",
) -> str:
    """Determine what "auto" resolves to for a given feature.

    Returns "log" or "raw" — the concrete method auto would apply.
    """
    from napari_epicycle.fusion_reconstruction.qc_features import BOUNDED_FEATURES

    if feature_name in BOUNDED_FEATURES:
        return "raw"

    from scipy.stats import skew as _skew

    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return "raw"

    # Guard: near-constant data
    if np.ptp(clean) < np.finfo(np.float64).eps * max(1.0, abs(np.mean(clean))):
        return "raw"

    s = float(_skew(clean, bias=False))
    return "log" if s > 1.0 else "raw"


# ---------------------------------------------------------------------------
# Detail stats (Phase 4)
# ---------------------------------------------------------------------------

def compute_detail_stats(values: np.ndarray) -> Dict:
    """Compute detailed summary statistics for a single feature.

    Returns
    -------
    dict with keys: n, mean, median, std, min, max, iqr, skew, kurtosis.
    All floats (NaN for insufficient data).
    """
    from scipy.stats import iqr as _iqr
    from scipy.stats import kurtosis as _kurtosis
    from scipy.stats import skew as _skew

    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    n = len(clean)

    if n == 0:
        nan = float("nan")
        return {
            "n": 0, "mean": nan, "median": nan, "std": nan,
            "min": nan, "max": nan, "iqr": nan, "skew": nan,
            "kurtosis": nan,
        }

    result = {
        "n": n,
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "std": float(np.std(clean, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "iqr": float(_iqr(clean)),
    }

    # Guard near-constant data
    if n < 3 or np.ptp(clean) < np.finfo(np.float64).eps * max(1.0, abs(result["mean"])):
        result["skew"] = 0.0
        result["kurtosis"] = 0.0
    else:
        result["skew"] = float(_skew(clean, bias=False))
        result["kurtosis"] = float(_kurtosis(clean, bias=False))

    return result


def compute_correlations(
    df: "pd.DataFrame",
    feature: str,
    feature_columns: List[str],
    *,
    top_n: int = 5,
    exclude_border: bool = True,
) -> Dict[str, List]:
    """Compute top Spearman and Pearson correlates for a feature.

    Parameters
    ----------
    df : DataFrame
    feature : str
        The target feature column.
    feature_columns : list of str
        Columns to correlate against (excludes *feature* itself).
    top_n : int
        Number of top correlates to return.
    exclude_border : bool
        If True, exclude border cells.

    Returns
    -------
    dict with keys:
        spearman : list of (feature_name, correlation) tuples, sorted by |r|
        pearson  : list of (feature_name, correlation) tuples, sorted by |r|
    """
    work = df
    if exclude_border and "is_border_cell" in df.columns:
        work = df[~df["is_border_cell"].astype(bool)]

    if feature not in work.columns:
        return {"spearman": [], "pearson": []}

    others = [c for c in feature_columns if c != feature and c in work.columns]
    if not others:
        return {"spearman": [], "pearson": []}

    target = work[feature].to_numpy()
    finite_target = np.isfinite(target)

    from scipy.stats import pearsonr, spearmanr

    spearman_results = []
    pearson_results = []

    for col in others:
        other = work[col].to_numpy()
        valid = finite_target & np.isfinite(other)
        if valid.sum() < 3:
            continue
        t, o = target[valid], other[valid]
        # Skip if either is constant
        if np.ptp(t) == 0 or np.ptp(o) == 0:
            continue
        try:
            rs, _ = spearmanr(t, o)
            spearman_results.append((col, float(rs)))
        except Exception:
            pass
        try:
            rp, _ = pearsonr(t, o)
            pearson_results.append((col, float(rp)))
        except Exception:
            pass

    # Sort by absolute correlation, descending
    spearman_results.sort(key=lambda x: abs(x[1]), reverse=True)
    pearson_results.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "spearman": spearman_results[:top_n],
        "pearson": pearson_results[:top_n],
    }


# ---------------------------------------------------------------------------
# Combined profile
# ---------------------------------------------------------------------------

def compute_feature_profile(
    values: np.ndarray,
    n_bins: int = 16,
) -> Dict:
    """Compute the full statistical profile for a single feature.

    Combines histogram, skewness, and normality into one dict.
    """
    return {
        "histogram": compute_histogram(values, n_bins=n_bins),
        "skewness": compute_skewness(values),
        "normality": compute_normality(values),
    }


# ---------------------------------------------------------------------------
# DataFrame-level profiling (batch)
# ---------------------------------------------------------------------------

def profile_dataframe(
    df: pd.DataFrame,
    feature_columns: List[str],
    *,
    exclude_border: bool = True,
    n_bins: int = 16,
) -> Dict[str, Dict]:
    """Compute profiles for all features in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain the feature columns.  May contain ``is_border_cell``.
    feature_columns : list of str
        Which columns to profile.
    exclude_border : bool
        If True and ``is_border_cell`` column exists, exclude border cells.
    n_bins : int
        Histogram bins.

    Returns
    -------
    dict mapping feature_name → profile dict (from compute_feature_profile).
    """
    work = df
    if exclude_border and "is_border_cell" in df.columns:
        work = df[~df["is_border_cell"].astype(bool)]

    profiles: Dict[str, Dict] = {}
    for col in feature_columns:
        if col not in work.columns:
            continue
        vals = work[col].to_numpy()
        profiles[col] = compute_feature_profile(vals, n_bins=n_bins)
    return profiles
