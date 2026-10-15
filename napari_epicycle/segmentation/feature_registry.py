"""Feature extraction registry for Step 4 (3D Stack Segmentation).

Provides a plug-and-play architecture for QC feature extraction.
Each feature group is a ``FeatureGroup`` entry with a name, callable,
required inputs, and a ``skip_after_smooth`` flag for post-smooth
re-extraction.

The underlying ``compute_*`` functions are shared with Step 3 (they
live in ``napari_epicycle.fusion_reconstruction.qc_features``).  Only
the orchestration layer is Step-4-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureGroup:
    """One pluggable feature extraction module."""

    name: str
    fn: Callable
    requires: frozenset
    skip_after_smooth: bool = False


# ---------------------------------------------------------------------------
# Built-in registry (order = execution order)
# ---------------------------------------------------------------------------

def _core_morphology(*, labels, df=None, intensity_image=None, spacing=None,
                     **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        extract_features,
    )
    return extract_features(labels, intensity_image=intensity_image,
                            spacing=spacing)


def _surface(*, labels, df, spacing=None, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_surface_features,
    )
    compute_surface_features(labels, df, spacing=spacing)
    return df


def _discrete_compactness(*, labels, df, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_discrete_compactness,
    )
    compute_discrete_compactness(labels, df)
    return df


def _boundary_interior(*, labels, df, intensity_image, erosion_iters=2, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_boundary_interior_features,
    )
    compute_boundary_interior_features(labels, intensity_image, df,
                                       erosion_iters=erosion_iters)
    return df


def _probability(*, labels, df, prob_volume, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_probability_features,
    )
    compute_probability_features(labels, df, prob_volume)
    return df


def _neighbor(*, labels, df, max_radius=5, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_neighbor_features,
    )
    compute_neighbor_features(labels, df, max_radius=max_radius)
    return df


def _skeleton(*, labels, df, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_skeleton_features,
    )
    compute_skeleton_features(labels, df)
    return df


def _cross_section(*, labels, df, invert_z=False, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_cross_section_features,
    )
    compute_cross_section_features(labels, df, invert_z=invert_z)
    return df


def _border_cells(*, labels, df, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_border_cells,
    )
    compute_border_cells(labels, df)
    return df


def _named_composites(*, df, labels=None, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_named_composites,
    )
    compute_named_composites(df)
    return df


def _composite(*, df, labels=None, exclude_border=False, **kw):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_composite_score,
    )
    compute_composite_score(df, exclude_border=exclude_border)
    return df


FEATURE_REGISTRY: list[FeatureGroup] = [
    FeatureGroup(
        name="core_morphology",
        fn=_core_morphology,
        requires=frozenset({"labels"}),
    ),
    FeatureGroup(
        name="surface",
        fn=_surface,
        requires=frozenset({"labels"}),
    ),
    FeatureGroup(
        name="discrete_compactness",
        fn=_discrete_compactness,
        requires=frozenset({"labels"}),
    ),
    # NOTE: boundary_interior removed from Step 4 registry.
    # Intensity features are now computed per-channel via compute_channel_intensity
    # (11 features each, including z_intensity_variation). Users select channels
    # in the UI; no implicit seg-channel assumption.
    FeatureGroup(
        name="probability",
        fn=_probability,
        requires=frozenset({"labels", "prob_volume"}),
        skip_after_smooth=True,
    ),
    FeatureGroup(
        name="neighbor",
        fn=_neighbor,
        requires=frozenset({"labels"}),
    ),
    FeatureGroup(
        name="skeleton",
        fn=_skeleton,
        requires=frozenset({"labels"}),
    ),
    FeatureGroup(
        name="cross_section",
        fn=_cross_section,
        requires=frozenset({"labels"}),
    ),
    FeatureGroup(
        name="border_cells",
        fn=_border_cells,
        requires=frozenset({"labels"}),
    ),
    FeatureGroup(
        name="named_composites",
        fn=_named_composites,
        requires=frozenset({"labels"}),
    ),
    FeatureGroup(
        name="composite",
        fn=_composite,
        requires=frozenset({"labels"}),
    ),
]


# ---------------------------------------------------------------------------
# Dynamic registry construction
# ---------------------------------------------------------------------------

def _make_channel_fn(channel_idx: int, prefix: str):
    """Create a channel intensity extraction function for the registry."""

    def _channel_intensity(*, labels, df, raw_stack, seg_channel=0, **kw):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_channel_intensity, extract_single_channel,
        )
        ch_image = extract_single_channel(
            raw_stack, labels.shape, seg_channel=channel_idx,
        )
        if ch_image is not None:
            compute_channel_intensity(labels, df, ch_image, prefix=prefix)
        return df

    return _channel_intensity


def build_registry(
    checked_channels: Optional[list] = None,
) -> list[FeatureGroup]:
    """Return registry with dynamic multi-channel entries appended.

    Args:
        checked_channels: List of ``{"index": int, "prefix": str}`` dicts
            from the Intensity Channels UI.  Each generates a FeatureGroup.

    Returns:
        Copy of FEATURE_REGISTRY + any channel-specific entries.
    """
    registry = list(FEATURE_REGISTRY)
    if checked_channels:
        for ch in checked_channels:
            idx = ch["index"]
            prefix = ch["prefix"]
            registry.append(FeatureGroup(
                name=f"intensity_{prefix}",
                fn=_make_channel_fn(idx, prefix),
                requires=frozenset({"labels", "raw_stack"}),
                skip_after_smooth=False,
            ))
    return registry


# ---------------------------------------------------------------------------
# Feature manifest (for Extract Features dialog)
# ---------------------------------------------------------------------------

# The 11 per-channel intensity stat suffixes (order matches compute_channel_intensity)
_CHANNEL_SUFFIXES = [
    "_intensity_mean", "_intensity_median", "_intensity_std",
    "_intensity_min", "_intensity_max", "_intensity_sum",
    "_intensity_cv",
    "_boundary_intensity_mean", "_interior_intensity_mean",
    "_boundary_interior_ratio", "_z_intensity_variation",
]


def get_feature_manifest(
    checked_channels: Optional[list] = None,
) -> dict[str, list[str]]:
    """Return ``{category: [feature_names]}`` for the Extract Features dialog.

    Core categories come from the static ``_FEATURE_CATEGORIES`` mapping in
    ``analysis_widget``.  Channel groups are generated dynamically from the
    user's checked channels.  Categories are returned in display order.

    Args:
        checked_channels: List of ``{"index": int, "prefix": str}`` dicts.

    Returns:
        OrderedDict-style dict (insertion-ordered) mapping category names to
        lists of feature column names.
    """
    from napari_epicycle.segmentation.analysis_widget import (
        _CATEGORY_ORDER, _FEATURE_CATEGORIES,
    )

    # Build {category: [features]} in display order
    manifest: dict[str, list[str]] = {}
    for cat in _CATEGORY_ORDER:
        feats = [f for f, c in _FEATURE_CATEGORIES.items() if c == cat]
        if feats:
            manifest[cat] = feats

    # Append dynamic channel groups
    if checked_channels:
        for ch in checked_channels:
            prefix = ch["prefix"]
            display_name = f"Channel: {prefix.replace('_', ' ').title()}"
            manifest[display_name] = [
                f"{prefix}{suffix}" for suffix in _CHANNEL_SUFFIXES
            ]

    return manifest


# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------

def run_extraction(
    registry: list[FeatureGroup],
    available_inputs: dict,
    *,
    post_smooth: bool = False,
    progress_cb=None,
) -> pd.DataFrame:
    """Iterate registry, skip groups whose inputs are missing or flagged.

    Args:
        registry: List of FeatureGroup entries (from ``build_registry``).
        available_inputs: Dict of available data keyed by name.
            Common keys: ``"labels"``, ``"intensity_image"``, ``"spacing"``,
            ``"prob_volume"``, ``"raw_stack"``, ``"seg_channel"``.
        post_smooth: If True, skip groups with ``skip_after_smooth=True``.
        progress_cb: Optional callback ``(int) -> None`` for progress
            reporting (0-100).

    Returns:
        DataFrame with all extracted features.
    """
    import time as _qcperf_time  # [QC-PERF] per-group extraction profiling

    df = None
    n_groups = len(registry)
    ran = 0

    for i, group in enumerate(registry):
        if post_smooth and group.skip_after_smooth:
            continue
        if not group.requires.issubset(available_inputs.keys()):
            continue

        # Build kwargs from available inputs
        kwargs = {k: available_inputs[k] for k in group.requires}
        # Always pass df and optional params that functions may accept
        kwargs["df"] = df
        for opt in ("spacing", "erosion_iters", "max_radius", "seg_channel",
                    "exclude_border", "invert_z", "intensity_image"):
            if opt in available_inputs:
                kwargs[opt] = available_inputs[opt]

        # [QC-PERF] time each feature group so the dominant cost is visible
        # in the console (the extraction is GIL-bound on a worker thread, so
        # this breakdown drives the fix).
        _grp_t0 = _qcperf_time.perf_counter()
        df = group.fn(**kwargs)
        _grp_ms = (_qcperf_time.perf_counter() - _grp_t0) * 1000.0
        print(
            f"[QC-PERF]   group {group.name} = {_grp_ms:.0f} ms "
            f"(rows={len(df) if df is not None else 0})"
        )
        ran += 1

        if progress_cb is not None:
            pct = int((i + 1) / n_groups * 100)
            progress_cb(pct)

    if progress_cb is not None:
        progress_cb(100)

    return df if df is not None else pd.DataFrame()
