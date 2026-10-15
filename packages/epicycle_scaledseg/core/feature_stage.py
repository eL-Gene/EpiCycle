"""Per-cell QC / morphometric feature extraction — STAGE ORCHESTRATOR + WRITER.

Runs AFTER the post-stitch cleanup on the single global-label volume
(``stage4_full_stitch/global_labels_clean.{npy,tif}`` + ``label_provenance_clean.csv``;
falls back to the raw stitch ``global_labels.npy`` + ``label_provenance.csv`` when no
clean output exists). It extracts Wave-1 cell-morphology features (see
:mod:`epicycle.core.qc_features`) into a
per-cell table written as **Parquet + CSV**, plus a ``feature_manifest.json``.

Design (mirrors :class:`~epicycle.core.postprocess_stage.PostprocessStage`):

* **Per-cell bbox-parallel.** The full volume is NEVER loaded. Each cell's bbox
  (from the provenance CSV) becomes one task; a worker ``Pool`` crops that bbox from
  the source memmap, isolates the cell (``crop == cell_id`` relabelled to a 1-cell
  volume), and runs the pure ``compute_*`` functions on it. Centroids + bboxes are
  offset back to GLOBAL coordinates. Robust per-cell isolation: a crop-read or
  compute failure on one cell is logged and skipped, never aborting the whole run.
* **Two passes.** The per-cell pass (parallel) fills morphology; then a single
  in-process ``cKDTree`` pass over all centroids adds neighbour-spacing / crowding
  (it needs every cell at once).
* **Border from provenance, not crops.** A per-cell bbox crop cannot see the tissue
  boundary (every cell touches its own crop face), so ``is_border_cell`` is taken
  from the provenance ``touches_border`` field — NOT from ``compute_border_cells``.
* **Physical units.** ``voxel_size_um`` (Z, Y, X) → physical-unit morphology
  (volume in µm³, lengths in µm, centroids/crowding in µm), exactly as the monorepo
  does via ``regionprops(spacing=...)``. ``None`` = voxel units.
* **Idempotent.** Skips if the features table exists AND the manifest status is
  ``complete`` (a truncated table from a crash never short-circuits a redo).
* **dry_run.** Logs the cell count + voxel-count distribution and writes nothing —
  to size the job before paying for the (long) extraction.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult
from packages.epicycle_scaledseg.core.cleanup import _ri
from packages.epicycle_scaledseg.utils.resources import get_allocated_cpus

_MANIFEST_VERSION = 1
_SCHEMA_VERSION = 1

# Per-process source label memmap handle for the extract workers (set by the
# initializer; module-level so it is picklable under spawn).
_WORKER_SRC: Optional[np.ndarray] = None

# M2 OPTIONAL sampler config (set by the initializer; module-level so it is
# picklable under spawn and shared by every task in the run — the dir paths +
# channel list + prob flag are the SAME for all cells). When all are unset the
# worker runs the labels-only path unchanged (byte-identical to pre-M2).
#   _WORKER_TILES_DIR : raw ZCYX channel tiles dir (read via raw_roi.read_global_roi)
#   _WORKER_RECON_DIR : per-tile probability dir (read via raw_roi.read_prob_roi)
#   _WORKER_CHANNELS  : list of (array_index, prefix) raw channels to sample,
#                       e.g. [(0, "nuc"), (1, "mem")] -> compute_channel_intensity
#   _WORKER_PROB      : sample the per-tile prob (sigmoid) -> compute_probability_features
_WORKER_TILES_DIR: Optional[str] = None
_WORKER_RECON_DIR: Optional[str] = None
_WORKER_CHANNELS: Optional[list] = None
_WORKER_PROB: bool = False

# The feature-set gate: "full" = every engine family (the in-code
# default, so every pre-gate caller is unchanged); "epicycle" = only the EpiCycle catalogue
# (the plugin's feature catalogue).  Set per worker by ``_init_feature_worker`` (initargs, NOT
# the frozen 5-tuple task) and reset in the serial ``finally`` blocks so a serial "epicycle" run
# cannot leak into a later "full" call in the same process.
FEATURE_SETS = ("epicycle", "full")
_WORKER_FEATURE_SET: str = "full"


def _check_feature_set(feature_set) -> str:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    return feature_set


# --------------------------------------------------------------------------- #
# Parallel per-cell worker (module-level for spawn-picklability)
# --------------------------------------------------------------------------- #

def _init_feature_worker(
    npy_path: str,
    tiles_dir: Optional[str] = None,
    recon_dir: Optional[str] = None,
    channels: Optional[list] = None,
    prob: bool = False,
    feature_set: str = "full",
) -> None:
    """Pool initializer: open the source label memmap ONCE per worker process.

    The OPTIONAL M2 sampler config (``tiles_dir`` / ``recon_dir`` / ``channels`` /
    ``prob``) is stashed in module-level globals so every task in the run shares it
    (it is identical for all cells) and so the worker stays picklable under spawn.
    Left at its defaults (all None / False) the worker runs the labels-only path
    unchanged.
    """
    # Per-cell extraction builds a 1-ROW DataFrame and appends ~300 feature
    # columns one at a time -> pandas raises PerformanceWarning("highly
    # fragmented") once per worker. For a single-row frame the consolidation is a
    # few hundred floats (negligible) — it is pure log noise that floods the SLURM
    # .out, so silence it INSIDE the worker only (the main-process global passes,
    # where fragmentation on the big assembled frame would actually matter, are
    # unaffected).
    import warnings as _warnings
    try:
        import pandas as _pd
        _warnings.filterwarnings("ignore", category=_pd.errors.PerformanceWarning)
    except Exception:
        pass

    global _WORKER_SRC, _WORKER_TILES_DIR, _WORKER_RECON_DIR, _WORKER_CHANNELS, _WORKER_PROB
    global _WORKER_FEATURE_SET
    _WORKER_FEATURE_SET = _check_feature_set(feature_set)
    _WORKER_SRC = np.load(npy_path, mmap_mode="r")
    _WORKER_TILES_DIR = str(tiles_dir) if tiles_dir else None
    _WORKER_RECON_DIR = str(recon_dir) if recon_dir else None
    _WORKER_CHANNELS = list(channels) if channels else None
    _WORKER_PROB = bool(prob)


def _extract_one(task):
    """Extract Wave-1 morphology for ONE cell from its bbox crop.

    ``task = (cell_id, (z0, z1, y0, y1, x0, x1), spacing_or_None, invert_z,
    include_surface)``.

    Reads the (half-open) bbox crop from the per-process source memmap, isolates
    ``cell_id`` (relabelled to a single-cell volume so the pure functions see exactly
    one object), runs the Wave-1 ``compute_*`` functions, and offsets centroid + bbox
    back to GLOBAL coordinates. Returns a flat ``{column: value}`` dict.

    Failure isolation (one bad cell must NEVER abort a multi-hour run):
    * cell genuinely ABSENT from its own bbox crop (CSV/volume drift) ->
      ``("MISSING", cell_id)``;
    * any compute/read exception -> ``("ERROR", cell_id, repr(e))``.
    The caller counts + logs these and continues.
    """
    from packages.epicycle_scaledseg.core import qc_features as qf

    cell_id, bbox, spacing, invert_z, include_surface = task
    z0, z1, y0, y1, x0, x1 = bbox
    try:
        crop = np.asarray(_WORKER_SRC[z0:z1, y0:y1, x0:x1])
        cell_mask = crop == cell_id
        if not cell_mask.any():
            return ("MISSING", int(cell_id))

        # Isolate the cell as a single-label (id == 1) volume so the pure
        # functions (which iterate df.label + index find_objects[label-1]) see
        # exactly this cell — neighbouring cells in the same bbox are excluded.
        single = cell_mask.astype(np.int32)
        sp = tuple(spacing) if spacing is not None else None

        df = qf.extract_features(single, spacing=sp)
        if include_surface:
            qf.compute_surface_features(single, df, spacing=sp)
        qf.compute_discrete_compactness(single, df)
        qf.compute_skeleton_features(single, df)
        qf.compute_cross_section_features(single, df, invert_z=invert_z)
        # feature_set == "epicycle": none of the engine-only families below are in the catalogue.
        if _WORKER_FEATURE_SET != "epicycle":
            # M0 additive families (labels-only, crop-local — same per-cell pass):
            qf.compute_cross_section_profile_stats(single, df, invert_z=invert_z)
            qf.compute_triaxial(single, df)
            # M1 additive families (labels-only, crop-local — same per-cell pass):
            #   EDT thickness field, EDT-anchored geodesic/tortuosity, and major-axis
            #   (tilt-robust) cross-section profile. ma_* reads the major_axis_* +
            #   centroid-* columns extract_features already emitted; it uses the
            #   crop-LOCAL centroid internally, so the global-offset step below is
            #   irrelevant to it (safe to call before that offset).
            qf.compute_edt_features(single, df, spacing=sp)
            qf.compute_geodesic_features(single, df, spacing=sp)
            qf.compute_ma_cross_section_features(single, df, spacing=sp)
            # M5 additive families (labels-only, crop-local — same per-cell pass):
            #   radial/shell decomposition (reuses the M1 EDT field) + surface
            #   curvature/concavity/symmetry (reuses the marching-cubes mesh, the
            #   convex hull, and the major-axis columns extract_features emitted).
            #   shell + symmetry read the centroid/major_axis_* columns using the
            #   crop-LOCAL coords internally, so calling them BEFORE the global
            #   centroid offset below is correct (same rationale as ma_*).
            qf.compute_shell_features(single, df, spacing=sp)
            qf.compute_curvature_features(single, df, spacing=sp)
            # Wave H additive family (labels-only, crop-local — same per-cell pass):
            #   multi-scale (zone-restricted) SHAPE templates across the SAME M5 EDT
            #   radial shells (inner core / outer rind) — discrete compactness per band
            #   + the outer-rind surface fraction. Intrinsic (object-only distance
            #   field), so no Wave-C gate; additive only.
            qf.compute_multiscale_shape(single, df, spacing=sp)
        # M2 OPTIONAL sampler families (intensity + probability). Only when the
        # initializer stashed a sampler config; the labels-only path is otherwise
        # byte-identical. Crops are read at the SAME (half-open) bbox as the label
        # crop, so they align voxel-for-voxel with `single`. Done HERE — on the
        # crop-local single-label mask, BEFORE the centroid/bbox global offset.
        if _WORKER_CHANNELS or _WORKER_PROB:
            _sample_m2_features(single, df, bbox, spacing=sp)
        # NB: compute_border_cells / compute_centroid_spacing are NOT run here —
        # they are global (the orchestrator sets is_border_cell from provenance
        # and runs the cKDTree crowding pass once on the assembled table).

        # Offset local crop coordinates -> global.
        #   centroid: regionprops reports it in (voxels × spacing); the crop
        #     origin is (z0, y0, x0) in voxels -> add origin × spacing.
        #   bbox: regionprops bbox is in PIXEL indices regardless of spacing
        #     (the monorepo scales it by hand) -> add the integer origin.
        if sp is not None:
            df["centroid-0"] = df["centroid-0"].values + z0 * float(sp[0])
            df["centroid-1"] = df["centroid-1"].values + y0 * float(sp[1])
            df["centroid-2"] = df["centroid-2"].values + x0 * float(sp[2])
        else:
            df["centroid-0"] = df["centroid-0"].values + z0
            df["centroid-1"] = df["centroid-1"].values + y0
            df["centroid-2"] = df["centroid-2"].values + x0
        for col, off in (
            ("bbox-0", z0), ("bbox-3", z0),
            ("bbox-1", y0), ("bbox-4", y0),
            ("bbox-2", x0), ("bbox-5", x0),
        ):
            if col in df.columns:
                df[col] = df[col].values + off

        df["label"] = int(cell_id)
        row = df.iloc[0].to_dict()
        return row
    except Exception as e:  # pragma: no cover - exercised via failure paths
        return ("ERROR", int(cell_id), repr(e))


def _sample_m2_features(single: np.ndarray, df, bbox, spacing=None) -> None:
    """Read the raw/prob crop(s) at ``bbox`` and add the M2 sampler columns to ``df``.

    Driven by the module-level sampler config (``_WORKER_TILES_DIR`` /
    ``_WORKER_RECON_DIR`` / ``_WORKER_CHANNELS`` / ``_WORKER_PROB``) the initializer
    stashed. ``single`` is the crop-local single-label mask (id==1) at ``bbox``;
    crops are read at the SAME (half-open) bbox so they align voxel-for-voxel.

    Per-channel raw intensity uses :func:`raw_roi.read_global_roi` (ZCYX raw tile,
    ``channel`` = array index). Probability uses :func:`raw_roi.read_prob_roi` with
    ``apply_sigmoid=True`` — the engine prob is a PER-TILE LOGIT (NOT a global
    consensus), so the sigmoid maps it to [0,1] before the prob_* math runs.

    ``spacing`` (the per-cell ``(Z, Y, X)`` voxel size, or None) is threaded through
    for the Wave-F geometry-conditioned families (channel localization + prob-
    geometry) whose interior-EDT shells respect voxel anisotropy.

    Schema stability: if a crop read fails the columns are still emitted (on a
    NaN crop) so every cell carries the same columns — a NaN value for a bad read
    rather than a missing column that would desync the assembled table.
    """
    from packages.epicycle_scaledseg.core import qc_features as qf
    from packages.epicycle_scaledseg.reconcile import raw_roi

    sp = tuple(spacing) if spacing is not None else None
    # "epicycle": keep ONLY compute_channel_intensity + compute_probability_features (catalogue
    # rows); the texture / localization / puncta / co-localization / prob-geometry extras are
    # engine-only and are skipped (never raised).
    _full = _WORKER_FEATURE_SET != "epicycle"
    shape = single.shape  # (Z, Y, X) — the label crop shape == read_*_roi(halo=0) shape
    single_mask = single == 1  # crop-local single-label mask (Wave-E/F + prob-geom)

    # --- per-channel raw intensity (B1 boundary/interior on each + B2 per-channel) ---
    if _WORKER_CHANNELS and _WORKER_TILES_DIR:
        sampled_crops = []  # (prefix, crop) — retained for the Wave-K pair co-loc
        for array_index, prefix in _WORKER_CHANNELS:
            try:
                ch = raw_roi.read_global_roi(_WORKER_TILES_DIR, bbox, int(array_index))
                ch = np.asarray(ch)
            except Exception:
                ch = np.full(shape, np.nan, dtype=np.float64)
            sampled_crops.append((str(prefix), ch))
            qf.compute_channel_intensity(single, df, ch, str(prefix))
            if not _full:
                continue
            # Wave E — per-channel TEXTURE on the SAME crop + mask. Returns a flat
            # {col: value} dict (single cell) -> write into the (single-row) df so
            # the texture columns ride alongside the intensity columns for this
            # channel. Appears ONLY when channels are sampled (default-off
            # byte-identical), exactly like compute_channel_intensity above.
            tex = qf.compute_channel_texture(ch, single_mask, str(prefix))
            for col, val in tex.items():
                df[col] = val
            # Wave F — per-channel SINGLE-VOLUME localization: membrane-vs-
            # core SHELL split + radial-intensity slope on the SAME crop + mask.
            # The apical/basal sub-block needs the Wave C OUTWARD normal, which the
            # per-cell pool runs BEFORE (a global pass) — so outward=None here and
            # those 3 columns stay NaN (filled by the paired/global path when the
            # outward normal is available). The compartmental nuclear/cytoplasm
            # split needs the nucleus mask -> the PAIRED path, NOT here.
            loc = qf.compute_channel_localization(
                ch, single_mask, str(prefix), spacing=sp, outward=None)
            for col, val in loc.items():
                df[col] = val
            # Wave G + Wave K — per-channel HCR PUNCTA: discrete spot detection
            # (blob_log) on the SAME crop + mask. The spot-COUNTING (transcript)
            # complement to the Wave-E continuous TEXTURE on this channel; Wave K adds
            # the spatial-distribution columns (radial position / clustering / intensity
            # cv / dispersion; polarity needs the Wave C outward normal -> NaN here).
            # Flat {col: value} dict -> write into the single-row df, alongside the
            # intensity / texture / localization columns. Appears ONLY when channels
            # are sampled (default-off byte-identical), exactly like the siblings.
            punc = qf.compute_channel_puncta(
                ch, single_mask, str(prefix), spacing=sp, outward=None)
            for col, val in punc.items():
                df[col] = val

        # Wave K — channel-PAIR CO-LOCALIZATION. For each UNORDERED
        # pair of sampled channels, measure Pearson / Manders / overlap / ICQ /
        # co-positive fraction over the in-mask voxels common to both crops. Runs
        # ONLY with >= 2 sampled channels (single-channel runs stay byte-identical to
        # the Wave-G path); a flat / failed-read channel NaN-guards inside the fn.
        if _full and len(sampled_crops) >= 2:
            for i in range(len(sampled_crops)):
                pa, ca = sampled_crops[i]
                for j in range(i + 1, len(sampled_crops)):
                    pb, cb = sampled_crops[j]
                    coloc = qf.compute_channel_colocalization(
                        ca, cb, single_mask, pa, pb)
                    for col, val in coloc.items():
                        df[col] = val

    # --- probability family (B3) — per-tile LOGIT -> sigmoid -> [0,1] ---
    if _WORKER_PROB and _WORKER_TILES_DIR and _WORKER_RECON_DIR:
        try:
            pr = raw_roi.read_prob_roi(
                _WORKER_TILES_DIR, _WORKER_RECON_DIR, bbox, apply_sigmoid=True
            )
            pr = np.asarray(pr, dtype=np.float32)
        except Exception:
            pr = np.full(shape, np.nan, dtype=np.float32)
        qf.compute_probability_features(single, df, pr)
        if not _full:
            return
        # Wave F — probability-GEOMETRY interaction: combine the SAME prob
        # crop with the cell's interior-EDT radial GEOMETRY (M5 shells). Runs right
        # after compute_probability_features so it can reuse the prob_weighted_volume
        # that pass just wrote on the row (keeps prob_area_normalized consistent with
        # the M2 soft volume). Flat {col: value} dict -> write into the single-row df.
        pwv = None
        if "prob_weighted_volume" in df.columns:
            try:
                pwv = float(df["prob_weighted_volume"].iloc[0])
            except (TypeError, ValueError, IndexError):
                pwv = None
        # Mop-up O — the surface-conditioned prob columns need the cell's
        # surface_area (the marching-cubes surface family ran earlier when
        # include_surface) + its prob_boundary_drop (compute_probability_features
        # just wrote it). Both optional -> NaN columns when absent.
        def _row_float(col):
            if col in df.columns:
                try:
                    v = float(df[col].iloc[0])
                    return v if np.isfinite(v) else None
                except (TypeError, ValueError, IndexError):
                    return None
            return None
        pg = qf.compute_prob_geometry(
            single_mask, pr, spacing=sp, prob_weighted_volume=pwv,
            surface_area=_row_float("surface_area"),
            prob_boundary_drop=_row_float("prob_boundary_drop"))
        for col, val in pg.items():
            df[col] = val


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _read_provenance(csv_path: Path) -> List[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _voxel_stats(vcs: np.ndarray) -> dict:
    if vcs.size == 0:
        return {}
    return {
        "n": int(vcs.size),
        "min": int(vcs.min()),
        "p1": int(np.percentile(vcs, 1)),
        "p5": int(np.percentile(vcs, 5)),
        "median": int(np.median(vcs)),
        "p95": int(np.percentile(vcs, 95)),
        "p99": int(np.percentile(vcs, 99)),
        "max": int(vcs.max()),
    }


# Per-family expected columns (for the manifest; recorded as present∩produced).
_FEATURE_FAMILIES = {
    "morphology": [
        "area", "area_filled", "equivalent_diameter", "extent", "euler_number",
        "sphericity", "aspect_ratio", "bbox_volume", "z_span", "z_elongation",
        "xy_elongation", "fill_ratio", "ellipsoid_solidity", "major_axis_z_angle",
        "inertia_elongation", "inertia_flatness", "axis_major_length",
        "axis_minor_length",
        # Principal-axis ORIENTATION (the inertia eigenvectors, ZYX-as-xyz) + the
        # ordered inertia EIGENVALUES. Emitted on every labels-only path (next to
        # major_axis_z_angle, which is derived from major_axis_z) but previously
        # in no family. role/frame mirror the other morphology columns.
        "major_axis_x", "major_axis_y", "major_axis_z",
        "mid_axis_x", "mid_axis_y", "mid_axis_z",
        "minor_axis_x", "minor_axis_y", "minor_axis_z",
        "inertia_eigval_major", "inertia_eigval_mid", "inertia_eigval_minor",
    ],
    "surface": ["surface_area", "true_sphericity", "roughness", "solidity"],
    "discrete_compactness": ["discrete_compactness"],
    "skeleton": ["skeleton_branches", "skeleton_nodes", "skeleton_length"],
    "cross_section": [
        "apical_area", "basal_area", "apical_basal_area_ratio",
        "max_cross_section_z_relative", "cross_section_taper", "cross_section_bulge",
    ],
    "cross_section_profile": [
        "min_cross_section_area", "cross_section_area_range",
        "cross_section_area_cv", "cross_section_area_skew",
        "cross_section_waist_depth", "cross_section_peak_count",
        # Wave I remainder.
        "cross_section_area_kurtosis", "min_cross_section_z_relative",
        "cross_section_constriction_index", "cross_section_n_lobes",
        # Wave N remainder — basal flare, peak/trough prominence + the
        # per-slice mask moments (centroid drift, eccentricity profile + taper,
        # orientation twist).
        "basal_flare_index",
        "cross_section_peak_prominence", "cross_section_trough_prominence",
        "slice_centroid_drift_mean", "slice_centroid_drift_max",
        "slice_eccentricity_mean", "slice_eccentricity_taper",
        "slice_orientation_twist",
    ],
    "triaxial": [
        "axis_mid_length", "triaxiality", "disc_vs_rod_score",
        # Wave I axis-ratio remainder.
        "axis_mid_to_major_ratio", "axis_minor_to_mid_ratio",
        "shape_anisotropy_3d",
    ],
    "edt": [
        "edt_max", "edt_mean", "edt_p90", "edt_cv", "edt_skew",
        # Wave I medial-axis remainder.
        "medial_radius_mean", "medial_radius_cv", "edt_core_fraction",
        # Wave N medial-axis remainder.
        "medial_radius_range",
    ],
    "geodesic": [
        "geodesic_max", "tortuosity",
        # Wave N skeleton main-path / medial-axis geometry.
        "skeleton_main_path_length", "skeleton_main_path_tortuosity",
        "endpoint_distance_ratio", "medial_axis_bend_energy",
        "medial_branch_count",
    ],
    "ma_cross_section": [
        "ma_min_cross_section_area", "ma_max_cross_section_area",
        "ma_cross_section_range", "ma_cross_section_taper",
        "ma_cross_section_waist_depth",
        # Wave I remainder.
        "ma_cross_section_cv",
        # Wave N remainder — widest-slab position, residual bulge, peak
        # count, profile skew.
        "ma_max_cross_section_axis_relative", "ma_cross_section_bulge",
        "ma_peak_count", "ma_cross_section_skew",
    ],
    # M5 additive families (labels-only, crop-local — same per-cell pass).
    # "shell" = C8 radial/shell decomposition (P7, from the M1 EDT field);
    # "curvature" = C13 surface curvature/concavity/symmetry (mesh + hull + axes).
    "shell": [
        "shell_volume_frac_inner", "shell_volume_frac_mid",
        "shell_volume_frac_outer", "core_surface_ratio",
        "mean_radial_position", "radial_position_std",
        # Wave N radial/shell remainder — per-band compactness +
        # per-band eccentricity + core offset + volume ratios.
        "shell_compactness_inner", "shell_compactness_mid",
        "shell_compactness_outer",
        "shell_eccentricity_inner", "shell_eccentricity_mid",
        "shell_eccentricity_outer",
        "core_offset",
        "shell_volume_ratio_inner_outer", "shell_volume_ratio_mid_outer",
    ],
    "curvature": [
        "mean_curvature_mean", "mean_curvature_std", "abs_curvature_mean",
        "convexity", "concavity_count", "max_concavity_depth",
        "bilateral_symmetry_score",
        # Wave N curvature/symmetry remainder.
        "radial_symmetry_score", "apical_basal_symmetry_score",
    ],
    # Wave H — multi-scale (zone-restricted) SHAPE templates (C17). A per-cell
    # crop pass (labels-only, ALWAYS produced like "shell"/"curvature") that applies a
    # MODEST curated set of shape stats across the M5 EDT radial shells (inner core /
    # outer rind): discrete compactness per band + the outer-rind surface fraction +
    # the inner/outer compactness ratio. frame="intrinsic" (object-only distance
    # field). Deliberately minimal — NOT a full feature×partition explosion.
    "multiscale_shape": [
        "shell_sphericity_inner", "shell_sphericity_outer",
        "shell_surface_ratio", "shell_compactness_ratio",
        # Wave N remainder — mid-band compactness + axis-quartile
        # volume-fraction template + the q1/q4 area ratio.
        "shell_sphericity_mid",
        "shape_axis_q1_fraction", "shape_axis_q4_fraction",
        "shape_axis_q1_q4_ratio",
    ],
    # Wave B — surface curvature REMAINDER. A SEPARATE family (not folded
    # into "curvature") so the M5 family registry test stays exact; same mesh,
    # same compute_curvature_features pass, additive columns only.
    "curvature_remainder": [
        "gaussian_curvature_mean", "gaussian_curvature_std", "bending_energy",
        "mean_curvature_skew", "neck_count",
        # Wave I remainder.
        "curvature_negative_fraction", "gaussian_curvature_skew",
    ],
    # M2 sampler families (OPTIONAL — present only when prob/raw inputs are given).
    # probability = B3. Per-channel intensity (B2) columns are <prefix>-prefixed and
    # dynamic, so they are recorded at manifest time from the actual df columns (see
    # _per_channel_families).
    # NOTE: the un-prefixed B1 "boundary_interior" family is DELIBERATELY ABSENT.
    # compute_boundary_interior_features (which would emit boundary_intensity_mean /
    # interior_intensity_mean / boundary_interior_ratio / z_intensity_variation) is
    # NEVER wired into the per-cell flow — the B1 signal arrives via the PREFIXED
    # compute_channel_intensity (e.g. mem_boundary_interior_ratio), recorded under
    # the dynamic channel_intensity_<prefix> families. Listing the un-prefixed
    # family here produced a family that was always EMPTY (dead). The function +
    # its tests remain for direct/un-prefixed callers; only the dead registry entry
    # was dropped.
    "probability": [
        "prob_mean", "prob_p10", "prob_std",
        "prob_boundary_mean", "prob_interior_mean", "prob_boundary_drop",
        "prob_weighted_volume", "prob_soft_hard_ratio",
        "prob_boundary_entropy", "prob_mask_entropy",
        "prob_edge_gradient", "prob_edge_gradient_std",
    ],
    # Wave F — probability-GEOMETRY interaction. The un-prefixed prob_*
    # geometry-conditioned columns (compute_prob_geometry): how much of the mask
    # is high-confidence + the inner/outer EDT-shell prob means/gradient + the
    # boundary-shell uncertainty. Present ONLY when prob is sampled (sits next to
    # the M2 "probability" family), default-off byte-identical.
    "probability_geometry": [
        "prob_area_normalized",
        "prob_shell_inner_mean", "prob_shell_outer_mean", "prob_shell_gradient",
        "prob_surface_uncertainty",
        # Mop-up O remainder — surface-normalized confidence + boundary-drop-
        # per-area (NaN unless include_surface + prob both ran).
        "prob_surface_normalized", "prob_boundary_drop_per_area",
        # P1a SIGNED remainder — apical-vs-basal confidence difference (NaN in
        # the per-cell pool; needs a trusted outward normal + centroid supplied).
        "prob_apical_basal_gradient",
    ],
    "crowding": [
        "nn_centroid_distance", "knn_mean_distance",
        "neighbor_count_in_radius", "local_density",
    ],
    # Wave B — reliability / confidence meta (C0). A TABLE pass over the
    # assembled frame (compute_reliability_meta) reading existing columns; ALWAYS
    # produced by the FeatureStage (like crowding), so it's recorded present∩
    # produced. role="qc" trust filters, not morphology.
    "reliability": [
        "tilt_reliability", "cross_section_reliability",
        "border_reliability", "feature_confidence_score",
        # Mop-up O remainder — neighbour-topology + probability-support
        # reliability (degrade gracefully on a labels-only table).
        "neighbor_topology_reliability", "probability_support_score",
    ],
    # M3 adjacency families (OPTIONAL — present only on the global adjacency
    # pass, i.e. extract_label_features(adjacency=True); the FeatureStage
    # per-cell flow does NOT produce them, so they are recorded as present∩
    # produced like the rest). "adjacency" = B4 core + filtration + quality;
    # "adjacency_topology" = B5 z-stratified; "named_composites" = B6.
    "adjacency": [
        "neighbor_count", "nearest_neighbor_distance",
        "max_contact_fraction", "background_fraction",
        "nn_count_r1", "nn_count_r3", "nn_birth_median", "nn_birth_std",
        "nn_accumulation_rate", "nn_accumulation_knee",
        "mean_neighbor_size_ratio", "neighbor_size_cv",
        "contact_weighted_neighbor_size_ratio",
    ],
    "adjacency_topology": [
        "basal_neighbor_count", "apical_neighbor_count",
        "neighbor_z_turnover", "neighbor_persistence_ratio",
    ],
    "named_composites": [
        "merge_signal", "fragment_signal",
        "boundary_confidence", "shape_abnormality",
    ],
    # M4 families (OPTIONAL — present only on the global adjacency pass, like the
    # M3 families above; the FeatureStage per-cell flow does NOT produce them).
    # "contact_geometry" = C3 contact-patch geometry (P3); "tissue_frame" = C4
    # local tissue frame + orientation coherence (P6).
    "contact_geometry": [
        "contact_area_total", "contact_area_max", "largest_contact_fraction",
        "contact_count", "multi_patch_neighbor_fraction", "contact_balance_index",
        # Mop-up M remainder — per-edge area + per-patch shape + centroid
        # offset + side asymmetry + per-neighbour patch-count.
        "contact_area_mean", "contact_area_std",
        "contact_patch_eccentricity_mean", "contact_patch_eccentricity_max",
        "contact_patch_compactness_mean",
        "contact_centroid_offset_mean", "contact_centroid_offset_max",
        "contact_side_asymmetry", "contact_patch_count_mean",
    ],
    "tissue_frame": [
        "neighbor_ring_planarity", "local_normal_alignment",
        "neighbor_axis_coherence", "neighbor_ring_anisotropy",
        # Mop-up O orientation-coherence remainder.
        "contact_weighted_axis_coherence", "neighbor_axis_dispersion",
        "major_axis_vs_local_normal_angle",
        "neighbor_polar_dispersion", "neighbor_azimuth_entropy",
        "local_packing_order", "coordination_anisotropy", "frame_stability",
        # P1a SIGNED remainder — the apical/basal-oriented frame (rides the
        # Wave C outward normal; only present on the tissue_depth=True re-call).
        "apical_basal_axis_alignment", "minor_axis_in_tangent_plane_fraction",
        "tangent_plane_anisotropy",
        "cell_tilt_polarity", "apical_vs_basal_elongation",
        "tangential_asymmetry_magnitude",
    ],
    # Wave A families (OPTIONAL — present only on the global adjacency pass, like
    # the M3/M4 families above; the FeatureStage per-cell flow does NOT produce
    # them). "graph_topology" = C5 ego-subgraph + second-ring; "axis_neighbor_
    # persistence" = C6 major-axis turnover/persistence; "graph_motif" = C16
    # triangles + degree relationships. The C12 local-comparative columns are
    # TEMPLATED per base feature, so they're recovered from the actual df columns
    # at manifest time (see _local_comparative_families), like the per-channel
    # intensity columns.
    "graph_topology": [
        "ego_density", "ego_clustering_coefficient",
        "second_ring_count", "mean_neighbor_degree",
        # Mop-up M remainder — neighbour-degree spread + ego components +
        # second-ring ratios.
        "neighbor_degree_std", "neighbor_degree_max",
        "ego_component_count",
        "first_to_second_ring_ratio", "second_ring_size_ratio",
        "second_ring_degree_mean",
    ],
    "axis_neighbor_persistence": [
        "axis_neighbor_turnover", "axis_neighbor_persistence_ratio",
        "axis_basal_neighbor_count", "axis_apical_neighbor_count",
        # Wave J remainder — per-contact span stats + events + bins.
        "axis_neighbor_persistence_mean", "axis_neighbor_persistence_max",
        "axis_neighbor_persistence_cv",
        "axis_neighbor_birth_median", "axis_neighbor_death_median",
        "axis_neighbor_span_cv",
        "axis_neighbor_events", "axis_neighbor_exchange_rate",
        "axis_neighbor_multispan_fraction", "axis_neighbor_span_count_mean",
        "axis_neighbor_reentry_count",
        "mid_axis_neighbor_count",
        "basal_to_mid_turnover", "mid_to_apical_turnover",
    ],
    "graph_motif": [
        "ego_triangle_count", "neighbor_count_cv",
        "degree_centrality", "neighbor_degree_assortativity",
        # Mop-up M remainder — higher-order motifs + node-attribute homophily
        # + ego-graph centrality trio (research tier).
        "ego_square_count", "ego_cycle_fraction",
        "neighbor_homophily_area", "neighbor_homophily_shape",
        "ego_betweenness", "ego_pagerank", "ego_eigenvector_centrality",
    ],
    # Wave C families (OPTIONAL — present only on the global tissue-depth pass,
    # i.e. extract_label_features(tissue_depth=True); the FeatureStage per-cell
    # flow does NOT produce them). "tissue_depth" = the "in/out" tissue-EDT field
    # (depth-from-surface) + the outward-normal sign basis; "spatial_context" =
    # the volume-relative position covariates. Both are frame="tissue".
    "tissue_depth": [
        "distance_to_tissue_surface", "tissue_depth_mean", "tissue_depth_max",
        "normalized_tissue_depth", "at_fov_border",
        "tissue_outward_z", "tissue_outward_y", "tissue_outward_x",
        "tissue_outward_confidence",
    ],
    "spatial_context": [
        "centroid_z_relative", "centroid_y_relative", "centroid_x_relative",
        "distance_to_tissue_centroid", "radial_tissue_position",
        # Wave I remainder.
        "distance_to_boundary_face_min",
        # Mop-up O remainder — the named per-cell local tissue depth.
        "local_tissue_depth",
    ],
    # Wave D families (OPTIONAL — present only on the global tissue-depth pass,
    # i.e. extract_label_features(tissue_depth=True), since they consume the Wave C
    # outward normal; the FeatureStage per-cell flow does NOT produce them).
    # "directional_contact" = apical/basal/lateral contact split (needs the
    # adjacency graph too); "surface_partition" = apical/basal/lateral mesh
    # surface areas. Both frame="tissue".
    "directional_contact": [
        "apical_contact_fraction", "basal_contact_fraction",
        "lateral_contact_fraction", "contact_polarity_bias",
        # Mop-up M remainder — contact-direction-cloud structure.
        "dominant_contact_axis", "contact_hemisphere_balance",
        "contact_direction_entropy", "contact_sector_count",
        "dominant_contact_neighbor_fraction",
    ],
    "surface_partition": [
        "apical_surface_area", "basal_surface_area", "lateral_surface_area",
        "apical_surface_fraction", "basal_surface_fraction",
        "lateral_surface_fraction", "apical_basal_surface_ratio",
        # Wave I cross ratios.
        "lateral_to_apical_surface_ratio", "lateral_to_basal_surface_ratio",
        # Wave J remainder — footprint contour + per-partition curvature/roughness.
        "apical_perimeter", "apical_circularity", "apical_aspect_ratio",
        "apical_compactness",
        "basal_perimeter", "basal_circularity", "basal_aspect_ratio",
        "basal_compactness",
        "apical_curvature_mean", "apical_curvature_std",
        "basal_curvature_mean", "lateral_curvature_mean",
        "apical_surface_roughness", "basal_surface_roughness",
    ],
    # Wave H signed-frame REMAINDER families (OPTIONAL — present only on the global
    # tissue-depth pass, like the Wave D families above, since they consume the Wave
    # C outward normal). "ma_apical_basal" = signed apical/basal split of the
    # M1 major-axis cross-section profile; "apical_junction" = epithelial
    # polygon topology from the cell's LATERAL contacts (needs the adjacency graph
    # too). Both frame="tissue".
    "ma_apical_basal": [
        "ma_apical_area", "ma_basal_area", "ma_apical_basal_ratio",
        "ma_apical_taper", "ma_basal_taper",
    ],
    "apical_junction": [
        "apical_polygon_sides", "apical_junction_length",
        "apical_polygon_irregularity",
        # Wave J remainder — belt vertex/edge stats + apical footprint polygon.
        "apical_junction_vertex_count", "apical_junction_edge_length_cv",
        "apical_vertex_angle_cv",
        "apical_polygon_area", "apical_polygon_perimeter",
        "apical_shape_index", "apical_polygon_convexity",
    ],
    # Wave G family (OPTIONAL — present only on the global watershed-territory pass,
    # i.e. extract_label_features(territory=True); the FeatureStage per-cell flow
    # does NOT produce it). "watershed_territory" = Voronoi-ish territory
    # volume + crowding ratio + territory shape (anisotropy / solidity).
    # frame="tissue" (the inter-cell space tessellation).
    "watershed_territory": [
        "ws_territory_volume", "territory_to_cell_ratio",
        "territory_anisotropy", "territory_solidity",
        # Mop-up M remainder — watershed adjacency + nearest gap + territory
        # z-bias + territory surface area.
        "ws_neighbor_count", "ws_nearest_gap",
        "territory_z_bias", "territory_surface_area",
    ],
    # P1b lumen-cavity family (OPTIONAL — present only on the global lumen pass,
    # i.e. extract_label_features(lumen=True); the FeatureStage per-cell flow does
    # NOT produce it). "lumen" = distance_to_lumen + lumen_facing_fraction
    # (the enclosed-cavity inner surface). frame="tissue" (a whole-tissue cavity
    # model). All-NaN for a tissue with no enclosed lumen (the correct answer).
    "lumen": [
        "distance_to_lumen", "lumen_facing_fraction",
    ],
    # M7 composite score family (OPTIONAL — present only on the global qc_score
    # pass, i.e. extract_label_features(qc_score=True); the FeatureStage per-cell
    # flow does NOT produce it). A single global-table outlier score.
    "composite_score": [
        "qc_score",
    ],
    "provenance": [
        "voxel_count", "is_border_cell", "seam_status", "min_match_iou", "n_tiles",
    ],
}

# Per-channel (B2) intensity suffixes — emitted by qc_features.compute_channel_intensity
# as ``<prefix>{suffix}``. Used to recover the dynamic per-channel families from the
# actual df columns at manifest time (the prefix is a runtime choice).
_CHANNEL_INTENSITY_SUFFIXES = (
    "_intensity_mean", "_intensity_median", "_intensity_std",
    "_intensity_min", "_intensity_max", "_intensity_sum", "_intensity_cv",
    "_boundary_intensity_mean", "_interior_intensity_mean",
    "_boundary_interior_ratio", "_z_intensity_variation",
)


def _per_channel_families(columns, channels) -> Dict[str, List[str]]:
    """Per-channel intensity (B2) families keyed ``channel_intensity_<prefix>``.

    ``channels`` = the run's ``[(array_index, prefix), ...]`` (or None). For each
    prefix, records the ``<prefix>{suffix}`` columns actually present in ``columns``.
    Empty dict when no channels were sampled.
    """
    if not channels:
        return {}
    cols = set(columns)
    fams: Dict[str, List[str]] = {}
    for _array_index, prefix in channels:
        prefix = str(prefix)
        present = [f"{prefix}{s}" for s in _CHANNEL_INTENSITY_SUFFIXES
                   if f"{prefix}{s}" in cols]
        if present:
            fams[f"channel_intensity_{prefix}"] = present
    return fams


# Wave E — per-channel TEXTURE suffixes — emitted by qc_features.
# compute_channel_texture as ``<prefix>{suffix}``. Like the intensity suffixes,
# used to recover the dynamic per-channel texture families from the actual df
# columns at manifest time (the prefix is a runtime choice).
_CHANNEL_TEXTURE_SUFFIXES = (
    "_texture_variance", "_texture_entropy", "_texture_mad",
    "_texture_granularity", "_texture_range_robust",
    "_glcm_contrast", "_glcm_homogeneity", "_glcm_correlation",
    "_glcm_energy", "_glcm_entropy", "_glcm_dissimilarity",
)


def _per_channel_texture_families(columns, channels) -> Dict[str, List[str]]:
    """Per-channel texture (Wave E) families keyed ``channel_texture_<prefix>``.

    ``channels`` = the run's ``[(array_index, prefix), ...]`` (or None). For each
    prefix, records the ``<prefix>{suffix}`` texture columns actually present in
    ``columns``. Empty dict when no channels were sampled. Mirrors
    :func:`_per_channel_families` — texture is the spatial-organization sibling of
    the B2 intensity family on the same channel.
    """
    if not channels:
        return {}
    cols = set(columns)
    fams: Dict[str, List[str]] = {}
    for _array_index, prefix in channels:
        prefix = str(prefix)
        present = [f"{prefix}{s}" for s in _CHANNEL_TEXTURE_SUFFIXES
                   if f"{prefix}{s}" in cols]
        if present:
            fams[f"channel_texture_{prefix}"] = present
    return fams


# Wave F — per-channel LOCALIZATION suffixes — emitted by qc_features.
# compute_channel_localization as ``<prefix>{suffix}``. The shell + radial-slope
# block is always present when channels are sampled (single-volume); the
# apical/basal block is present-but-NaN in the per-cell pool (no outward normal
# there), so it still shows up as a column. Recovered dynamically at manifest time
# like the intensity/texture per-channel families.
_CHANNEL_LOCALIZATION_SUFFIXES = (
    "_shell_inner_mean", "_shell_outer_mean", "_shell_gradient",
    "_radial_intensity_slope",
    # Mop-up O single-volume remainder (kept in lockstep with
    # qc_features._CHANNEL_LOCALIZATION_SHELL_SUFFIXES).
    "_shell_mid_mean", "_radial_bin_inner_mean", "_radial_bin_mid_mean",
    "_radial_bin_outer_mean",
    "_membrane_enrichment", "_outer_to_inner_ratio", "_fraction_above_threshold",
    "_apical_intensity_mean", "_basal_intensity_mean",
    "_apical_basal_intensity_ratio",
    # Mop-up O — lateral (equatorial) intensity band.
    "_lateral_intensity_mean",
)


def _per_channel_localization_families(columns, channels) -> Dict[str, List[str]]:
    """Per-channel localization (Wave F) families keyed
    ``channel_localization_<prefix>``.

    ``channels`` = the run's ``[(array_index, prefix), ...]`` (or None). For each
    prefix, records the ``<prefix>{suffix}`` localization columns actually present
    in ``columns``. Empty dict when no channels were sampled. Mirrors
    :func:`_per_channel_families` / :func:`_per_channel_texture_families` — these are
    the geometry-conditioned siblings of the B2 intensity family on the same channel.
    """
    if not channels:
        return {}
    cols = set(columns)
    fams: Dict[str, List[str]] = {}
    for _array_index, prefix in channels:
        prefix = str(prefix)
        present = [f"{prefix}{s}" for s in _CHANNEL_LOCALIZATION_SUFFIXES
                   if f"{prefix}{s}" in cols]
        if present:
            fams[f"channel_localization_{prefix}"] = present
    return fams


# Wave G — per-channel PUNCTA suffixes — emitted by qc_features.
# compute_channel_puncta as ``<prefix>{suffix}``. Like the intensity/texture/
# localization suffixes, used to recover the dynamic per-channel puncta families
# from the actual df columns at manifest time (the prefix is a runtime choice).
_CHANNEL_PUNCTA_SUFFIXES = (
    "_puncta_count", "_puncta_density", "_puncta_intensity_mean",
    "_puncta_mean_volume", "_puncta_nn_distance", "_punctate_fraction",
    # Wave K — puncta SPATIAL / distribution remainder.
    "_puncta_radial_position_mean", "_puncta_clustering_index",
    "_puncta_polarity", "_puncta_intensity_cv", "_puncta_dispersion",
)


def _per_channel_puncta_families(columns, channels) -> Dict[str, List[str]]:
    """Per-channel puncta (Wave G) families keyed ``channel_puncta_<prefix>``.

    ``channels`` = the run's ``[(array_index, prefix), ...]`` (or None). For each
    prefix, records the ``<prefix>{suffix}`` puncta columns actually present in
    ``columns``. Empty dict when no channels were sampled. Mirrors
    :func:`_per_channel_families` / :func:`_per_channel_texture_families` — puncta is
    the discrete spot-COUNTING (transcript) sibling of the intensity/texture family
    on the same channel.
    """
    if not channels:
        return {}
    cols = set(columns)
    fams: Dict[str, List[str]] = {}
    for _array_index, prefix in channels:
        prefix = str(prefix)
        present = [f"{prefix}{s}" for s in _CHANNEL_PUNCTA_SUFFIXES
                   if f"{prefix}{s}" in cols]
        if present:
            fams[f"channel_puncta_{prefix}"] = present
    return fams


# Wave K — channel-PAIR CO-LOCALIZATION suffixes — emitted by
# qc_features.compute_channel_colocalization as ``<a>_<b>{suffix}`` for each unordered
# pair of sampled channels. Recovered dynamically at manifest time like the
# per-channel families (the channel set + pairing is a runtime choice).
_CHANNEL_COLOC_SUFFIXES = (
    "_pearson", "_manders_m1", "_manders_m2", "_overlap_coefficient",
    "_icq", "_coloc_fraction",
    # Mop-up O co-localization remainder (kept in lockstep with
    # qc_features._CHANNEL_COLOC_SUFFIXES).
    "_mean_ratio", "_boundary_corr", "_interior_ratio_delta",
)


def _channel_pair_coloc_families(columns, channels) -> Dict[str, List[str]]:
    """Channel-pair co-localization (Wave K) families keyed
    ``channel_coloc_<a>_<b>``.

    ``channels`` = the run's ``[(array_index, prefix), ...]`` (or None). For each
    UNORDERED prefix pair, records the ``<a>_<b>{suffix}`` co-localization columns
    actually present in ``columns``. Empty dict when < 2 channels were sampled (no
    pair) or none present. Mirrors the per-channel family recovery helpers — these are
    the two-channel siblings of the single-channel intensity / texture / puncta
    families.
    """
    if not channels or len(channels) < 2:
        return {}
    cols = set(columns)
    prefixes = [str(p) for _i, p in channels]
    fams: Dict[str, List[str]] = {}
    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            pair = f"{prefixes[i]}_{prefixes[j]}"
            present = [f"{pair}{s}" for s in _CHANNEL_COLOC_SUFFIXES
                       if f"{pair}{s}" in cols]
            if present:
                fams[f"channel_coloc_{pair}"] = present
    return fams


# Wave A — local comparative (C12) columns are TEMPLATED per base feature
# (``<feature>_vs_neighbor_mean_ratio`` + ``<feature>_local_zscore``), so — like
# the per-channel intensity columns — they are recovered from the actual df
# columns at manifest time (the base-feature set is a runtime choice).
_LOCAL_COMPARATIVE_SUFFIXES = (
    "_vs_neighbor_mean_ratio", "_local_zscore",
    # Mop-up M — remainder suffixes (median ratio + signed difference +
    # contact-weighted-neighbour-mean ratio). Kept in lockstep with
    # qc_features._LOCAL_COMPARATIVE_SUFFIXES so the dynamic family recovery sees
    # the full templated set.
    "_vs_neighbor_median_ratio", "_minus_neighbor_mean",
    "_vs_contact_weighted_neighbor_mean",
)


def _local_comparative_families(columns) -> Dict[str, List[str]]:
    """Local-comparative (C12) families keyed ``local_comparative_<feature>``.

    Scans ``columns`` for ``<feature>_vs_neighbor_mean_ratio`` /
    ``<feature>_local_zscore`` pairs and groups each base feature's columns under
    ``local_comparative_<feature>``. Empty dict when none are present (the C12
    pass only runs on the opt-in global adjacency pass).
    """
    cols = list(columns)
    fams: Dict[str, List[str]] = {}
    for c in cols:
        for suffix in _LOCAL_COMPARATIVE_SUFFIXES:
            if c.endswith(suffix):
                feat = c[: -len(suffix)]
                fams.setdefault(f"local_comparative_{feat}", []).append(c)
                break
    return fams


# --------------------------------------------------------------------------- #
# Structured [FEATURES] logging helper
# --------------------------------------------------------------------------- #

def _peak_rss_gb() -> float:
    """True PEAK resident-set size of this process in GiB (NaN if unavailable).

    Used by the structured ``[FEATURES]`` per-family timing logs so the SLURM
    ``.out`` carries a RAM HIGH-WATER MARK next to each global pass's wall time —
    the "did this run blow up RAM" signal, which must be a peak, not the
    instantaneous value at the moment the timer happens to exit.

    H3 — platform-correct peak. On Windows, ``psutil``'s ``peak_wset`` is a true
    high-water mark. On Linux/macOS ``psutil`` only exposes the LIVE ``rss`` (its
    ``memory_info`` has no peak field there), so reporting ``rss`` mislabels every
    HPC ``.out`` peak-RSS line as instantaneous. Instead use
    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` — the process-lifetime peak
    (monotonic): Linux reports it in KiB (÷1024 -> GiB), macOS in BYTES (÷1024**3 ->
    GiB). Best-effort, never raises (a missing psutil/resource must not break
    extraction -> NaN). ``ru_maxrss`` covers only the PARENT process (the global
    field passes run single-process; the per-cell pool peak is captured by the
    workers' own resource accounting, not surfaced here)."""
    import sys
    if sys.platform.startswith("win"):
        try:
            import os
            import psutil
            mi = psutil.Process(os.getpid()).memory_info()
            return getattr(mi, "peak_wset", getattr(mi, "rss", 0)) / (1024 ** 3)
        except Exception:
            return float("nan")
    # POSIX (Linux / macOS): true lifetime peak via getrusage.
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KiB; macOS (Darwin): bytes.
        divisor = (1024 ** 3) if sys.platform == "darwin" else (1024 ** 2)
        return float(ru) / divisor
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

class FeatureStage(Stage):
    """Per-cell Wave-1 morphology extraction on the cleaned global labels.

    .. important:: CATALOGUE SUBSET — this stage (``--stages features`` /
       ``FeatureStage``) emits only the PER-CELL families: the bbox-parallel
       intrinsic morphology + the OPTIONAL per-cell samplers (channel intensity /
       texture / probability) + the in-process ``cKDTree`` crowding pass + the
       provenance / seam bookkeeping. The contact-graph GLOBAL family (adjacency /
       graph topology / contact geometry / axis-neighbour persistence / local-
       comparative / graph motifs / named composites) is OPT-IN via ``adjacency=
       True`` — it runs the shared :func:`~epicycle.core.qc_features.run_adjacency_graph_passes`
       on a MEMMAP of the global labels (per-cell halo crops; S1 scaling) and writes
       the contact graph to ``edges.csv``. The OTHER truly-global field families —
       the tissue depth / signed tissue frame / directional contact / surface
       partition / ma-apical-basal / apical-junction (``tissue_depth=True``), the
       watershed territory (``territory=True``), the lumen cavity (``lumen=True``),
       and the composite ``qc_score`` (``qc_score=True``) — are now ALSO opt-in here
       (Phase A): when requested they run the SAME sequence as
       :func:`epicycle.reconcile.features.extract_label_features` with the matching
       flags (byte-identical output) on a RESIDENT copy of the global label volume,
       loaded ONCE for all of them (these EDT / watershed / fill_holes passes need
       the whole volume, not per-cell crops — the Phase-A resident cost; Phase C
       downsamples it via ``field_downsample > 1``). So with all flags off (default)
       this stage emits the per-cell SUBSET; ``adjacency=True`` adds the contact
       graph (S1-scaled, memmap crops); the field flags add the resident global
       families (the engine's scaling stages S1-S3).

    .. warning:: LOW-TRUST SIGNED COLUMNS AT ``field_downsample > 1``. The downsampled
       field path (Phase C / S2-S3) computes the per-cell OUTWARD NORMAL from a coarse
       ``-∇EDT``; its DIRECTION drifts from the full-res normal on curved tissue
       (measured median ~11° / p90 ~57° at N=2). Everything it SIGNS — the apical/basal
       tissue frame, ``directional_contact``, ``surface_partition``, ``ma_apical_basal``,
       ``apical_junction`` — is therefore an APPROXIMATION at N>1; treat those columns as
       magnitude-only / low-trust unless the tissue geometry guarantees the sign. At
       N=1 the field path is byte-identical (full trust). The watershed-territory pass
       additionally seeds the coarse grid by per-block majority-of-nonzero, so a sub-N
       cell can drop from the territory columns (a WARNING reports the fraction).

    Parameters
    ----------
    input_dir : dir holding ``global_labels_clean.npy`` (or ``global_labels.npy``)
        + the matching provenance CSV (defaults to
        ``<tile_dirs[0].parent>/stage4_full_stitch`` — the stitch/clean output).
    output_dir : where to write the feature table (defaults to ``input_dir``).
    prefer_clean : use ``*_clean`` inputs when present (default True); set False to
        force extraction off the raw stitch.
    voxel_size_um : (Z, Y, X) physical voxel size → physical-unit features. None =
        voxel units.
    min_voxels : skip cells with ``voxel_count < min_voxels`` (and always skip the
        zero-voxel provenance ghosts). Default 1 = only drop ghosts.
    include_surface : compute the marching-cubes surface family (the most expensive
        per-cell op); set False for a fast morphology-only pass.
    invert_z : passed to the cross-section family (high Z = basal when True).
    k_neighbors / density_radius : crowding (cKDTree) params — see
        :func:`epicycle.core.qc_features.compute_centroid_spacing`.
    n_workers : parallel extract workers (defaults to allocated CPUs - 1).
    dry_run : log the cell count + size distribution; write nothing.
    """

    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        recon_root: Optional[str] = None,
        prefer_clean: bool = True,
        voxel_size_um: Optional[Tuple[float, float, float]] = None,
        min_voxels: int = 1,
        include_surface: bool = True,
        invert_z: bool = False,
        k_neighbors: int = 6,
        density_radius: Optional[float] = None,
        tiles_dir: Optional[str] = None,
        channels: Optional[List[Tuple[int, str]]] = None,
        prob: bool = False,
        adjacency: Optional[bool] = None,
        adjacency_max_radius: int = 5,
        tissue_depth: bool = False,
        territory: bool = False,
        lumen: bool = False,
        qc_score: Optional[bool] = None,
        field_downsample: int = 1,
        n_workers: Optional[int] = None,
        dry_run: bool = False,
        feature_set: str = "full",
    ):
        super().__init__("features")
        # feature_set "full" (in-code default) = every engine family, flags exactly as given
        # (None == off).  "epicycle" = the EpiCycle catalogue only: adjacency and qc_score are
        # forced ON unless the caller passes them explicitly False (closes the 22-row gap), and
        # the engine-only opt-ins (tissue_depth / territory / lumen) are skipped, never raised.
        self.feature_set = _check_feature_set(feature_set)
        self._fs_skipped: List[str] = []
        if self.feature_set == "epicycle":
            if adjacency is None:
                adjacency = True
            if qc_score is None:
                qc_score = True
            for _name, _on in (("tissue_depth", tissue_depth), ("territory", territory),
                               ("lumen", lumen)):
                if _on:
                    self._fs_skipped.append(_name)
            tissue_depth = territory = lumen = False
        self.input_dir = str(Path(input_dir).resolve()) if input_dir else None
        self.output_dir = str(Path(output_dir).resolve()) if output_dir else None
        self.recon_root = str(Path(recon_root).resolve()) if recon_root else None
        self.prefer_clean = prefer_clean
        self.voxel_size_um = tuple(float(v) for v in voxel_size_um) if voxel_size_um else None
        self.min_voxels = int(min_voxels)
        self.include_surface = include_surface
        self.invert_z = invert_z
        self.k_neighbors = int(k_neighbors)
        self.density_radius = density_radius
        # M2 OPTIONAL sampler inputs. tiles_dir = raw ZCYX channel tiles (read at
        # each cell's bbox); channels = [(array_index, prefix), ...] to sample;
        # prob = also sample the per-tile probability (sigmoid). recon_root already
        # locates the probability dir (stage2_reconstruction). All unset -> the
        # labels-only path runs unchanged.
        self.tiles_dir = str(Path(tiles_dir).resolve()) if tiles_dir else None
        self.channels = [(int(i), str(p)) for i, p in channels] if channels else None
        self.prob = bool(prob)
        # OPT-IN global adjacency / contact-graph pass. When
        # True, after the per-cell table is assembled the stage runs the SHARED
        # ``run_adjacency_graph_passes`` on a MEMMAP of the global labels (per-cell
        # halo crops — peak private RAM tracks crop, not volume; S1) and writes the
        # contact graph to ``edges.csv`` next to ``cell_features.csv``. Default off
        # -> the per-cell labels-only output is unchanged.
        self.adjacency = bool(adjacency)
        self.adjacency_max_radius = int(adjacency_max_radius)
        # OPT-IN truly-global field passes (Phase A — RESIDENT). When any of these
        # is True the stage loads the global label volume ONCE into RAM (a real
        # copy — these EDT / watershed / fill_holes passes need the whole volume,
        # not per-cell crops) and runs the SAME sequence as
        # ``reconcile.extract_label_features`` with the matching flags, so the
        # columns are byte-identical. Default off -> output unchanged.
        #   tissue_depth : 3D tissue-EDT "in/out" field + the signed apical/basal
        #                  frame + directional contact / surface partition /
        #                  ma-apical-basal / apical-junction (the last four self-
        #                  skip/NaN without the outward normal or the adjacency graph).
        #   territory    : watershed Voronoi-ish crowding territory.
        #   lumen        : enclosed-cavity distance + lining fraction.
        #   qc_score     : composite outlier score — a pure TABLE pass, runs
        #                  DEAD LAST (standardizes across the population).
        self.tissue_depth = bool(tissue_depth)
        self.territory = bool(territory)
        self.lumen = bool(lumen)
        self.qc_score = bool(qc_score)
        # RESERVED for Phase C (S2/S3): >1 will DOWNSAMPLE the resident field
        # volume for the EDT/watershed passes. Phase A keeps the field resident at
        # full resolution, so only 1 is meaningful here.
        self.field_downsample = int(field_downsample)
        self.n_workers = n_workers
        self.dry_run = dry_run

    def get_params(self) -> dict:
        return {
            "prefer_clean": self.prefer_clean,
            "voxel_size_um": self.voxel_size_um,
            "min_voxels": self.min_voxels,
            "include_surface": self.include_surface,
            "invert_z": self.invert_z,
            "k_neighbors": self.k_neighbors,
            "density_radius": self.density_radius,
            "tiles_dir": self.tiles_dir,
            "channels": self.channels,
            "prob": self.prob,
            "adjacency": self.adjacency,
            "adjacency_max_radius": self.adjacency_max_radius,
            "tissue_depth": self.tissue_depth,
            "territory": self.territory,
            "lumen": self.lumen,
            "qc_score": self.qc_score,
            "field_downsample": self.field_downsample,
            "n_workers": self.n_workers,
            "dry_run": self.dry_run,
            # A manifest written under another feature_set is NOT "complete" for this run.
            "feature_set": self.feature_set,
        }

    def _recon_root(self, tile_dirs: List[Path]) -> Path:
        if self.recon_root:
            return Path(self.recon_root)
        return tile_dirs[0].parent / "stage2_reconstruction"

    def _in_dir(self, tile_dirs: List[Path]) -> Path:
        if self.input_dir:
            return Path(self.input_dir)
        return self._recon_root(tile_dirs).parent / "stage4_full_stitch"

    def _resolve_inputs(self, in_dir: Path) -> Tuple[Optional[Path], Optional[Path], bool]:
        """Return (labels_npy, provenance_csv, used_clean).

        Prefers the ``*_clean`` pair when ``prefer_clean`` and both exist; else the
        raw stitch pair. Returns ``(None, None, ...)`` if no usable pair is found.
        """
        clean_npy = in_dir / "global_labels_clean.npy"
        clean_prov = in_dir / "label_provenance_clean.csv"
        raw_npy = in_dir / "global_labels.npy"
        raw_prov = in_dir / "label_provenance.csv"
        if self.prefer_clean and clean_npy.exists() and clean_prov.exists():
            return clean_npy, clean_prov, True
        if raw_npy.exists() and raw_prov.exists():
            return raw_npy, raw_prov, False
        # Last resort: clean pair exists but the raw pair doesn't. Use clean even
        # under prefer_clean=False (better than failing) — but say so, since it
        # overrides the user's explicit force-raw request.
        if clean_npy.exists() and clean_prov.exists():
            if not self.prefer_clean:
                logging.getLogger("epicycle.features").info(
                    "features: --no-feat-prefer-clean requested but no raw stitch "
                    "pair found — falling back to the *_clean inputs."
                )
            return clean_npy, clean_prov, True
        return None, None, False

    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        if not tile_dirs:
            return False
        npy, prov, _ = self._resolve_inputs(self._in_dir(tile_dirs))
        return npy is not None and prov is not None

    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        logger = logging.getLogger("epicycle.features")
        result = StageResult()
        if not tile_dirs:
            result.add_failure("(none)", "no tile directories provided")
            return result

        in_dir = self._in_dir(tile_dirs)
        out_dir = Path(self.output_dir) if self.output_dir else in_dir
        src_npy, prov_csv, used_clean = self._resolve_inputs(in_dir)

        if src_npy is None or prov_csv is None:
            msg = (f"features: no usable global labels + provenance in {in_dir} "
                   f"(run the iou_stitch / postprocess stage first)")
            logger.error(msg)
            for td in tile_dirs:
                result.add_failure(td.name, msg)
            return result

        out_parquet = out_dir / "cell_features.parquet"
        out_csv = out_dir / "cell_features.csv"
        out_edges = out_dir / "edges.csv"          # contact graph (adjacency=True)
        manifest_path = out_dir / "feature_manifest.json"

        # Idempotency: skip ONLY if a prior run COMPLETED (a partial table can exist
        # from a crash mid-write) — gate on the manifest status, never file existence.
        if out_parquet.exists() and not self.dry_run:
            prior_status = None
            prior_feature_set = "full"       # manifests that predate the gate were all "full"
            if manifest_path.exists():
                try:
                    _prior = json.loads(manifest_path.read_text())
                    prior_status = _prior.get("status")
                    prior_feature_set = (_prior.get("params") or {}).get("feature_set", "full")
                except Exception:
                    prior_status = None
            if prior_status == "complete" and prior_feature_set != self.feature_set:
                logger.info(f"features: prior manifest is complete for feature_set="
                            f"{prior_feature_set!r}, this run is {self.feature_set!r} "
                            f"— redoing.")
                prior_status = "complete (other feature_set)"
            if prior_status == "complete":
                logger.info(f"features: {out_parquet.name} exists + manifest complete "
                            f"— skipping (idempotent).")
                for td in tile_dirs:
                    result.add_success(td.name)
                return result
            logger.info(f"features: {out_parquet.name} exists but prior status is "
                        f"{prior_status!r} (not complete) — redoing.")

        src = np.load(str(src_npy), mmap_mode="r")
        depth, H, W = int(src.shape[0]), int(src.shape[1]), int(src.shape[2])
        rows = _read_provenance(prov_csv)

        # Build the work-list: real cells only (drop zero-voxel ghosts + below floor).
        sp = self.voxel_size_um
        floor = max(1, self.min_voxels)
        prov_by_id: Dict[int, dict] = {}
        tasks = []
        vcs_all = []
        for r in rows:
            cid = _ri(r, "dense_id")
            vc = _ri(r, "voxel_count")
            vcs_all.append(vc)
            prov_by_id[cid] = r
            if cid <= 0 or vc < floor:
                continue
            z0 = _ri(r, "z_min"); z1 = _ri(r, "z_max") + 1
            y0 = _ri(r, "y_min"); y1 = _ri(r, "y_max") + 1
            x0 = _ri(r, "x_min"); x1 = _ri(r, "x_max") + 1
            # Guard against a malformed (-1) provenance bbox.
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            z0 = max(0, z0); z1 = min(depth, z1)
            y0 = max(0, y0); y1 = min(H, y1)
            x0 = max(0, x0); x1 = min(W, x1)
            # A malformed (-1) bbox or an out-of-bounds row can degenerate to a
            # zero-size crop AFTER clamping — reject it (the pre-clamp guard above
            # misses a vc>0 row with a -1 bbox) rather than dispatch an empty task
            # that would only return a spurious MISSING.
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            tasks.append((cid, (z0, z1, y0, y1, x0, x1), sp, self.invert_z,
                          self.include_surface))

        vcs = np.array(vcs_all, dtype=np.int64)
        vstats = _voxel_stats(vcs[vcs > 0]) if vcs.size else {}
        n_cells = len(tasks)
        vox_um3 = (sp[0] * sp[1] * sp[2]) if sp else None

        logger.info(
            f"features: {len(rows)} provenance rows -> {n_cells} cells to extract "
            f"(min_voxels={floor}, surface={self.include_surface}, "
            f"clean_inputs={used_clean}); voxel_count {vstats}"
            + (f"; 1 voxel = {vox_um3:.4g} um^3" if vox_um3 else "")
        )

        if self.feature_set == "epicycle":
            logger.info(
                "[FEATURE-SET] epicycle: measuring the EpiCycle catalogue only; skipped the "
                "engine-only families (cross-section profile stats, triaxial, EDT, geodesic, "
                "major-axis cross-section, shell, curvature, multi-scale shape, centroid spacing, "
                "reliability meta, channel texture / localization / puncta / co-localization, "
                "prob geometry, contact geometry, graph topology / motifs / local-comparative, "
                "tissue depth / territory / lumen)"
                + (f"; requested but skipped: {', '.join(self._fs_skipped)}"
                   if self._fs_skipped else ""))

        # M5 — ONE up-front run summary so the operator can confirm the run config
        # from the HEAD of the .out (before any heavy work). The global-family flags
        # are on self.*; the effective worker count matches what the pools resolve
        # to (the per-cell pool clamps to len(tasks) later, but n_cells is the cap).
        _eff_nw = self.n_workers or max(1, get_allocated_cpus() - 1)
        _eff_nw = max(1, min(int(_eff_nw), max(1, n_cells)))
        _fam_flags = {
            "adjacency": self.adjacency, "tissue_depth": self.tissue_depth,
            "territory": self.territory, "lumen": self.lumen,
            "qc_score": self.qc_score,
        }
        _fam_on = ",".join(k for k, v in _fam_flags.items() if v) or "none"
        logger.info(
            f"[FEATURES] run: {n_cells} cells, vol ({depth}, {H}, {W}) {src.dtype}, "
            f"families={{{_fam_on}}}, field_downsample={self.field_downsample}, "
            f"workers={_eff_nw}, out={out_dir}"
        )

        def _manifest(status: str, extra: Optional[dict] = None) -> dict:
            m = {
                "version": _MANIFEST_VERSION, "stage": "features", "status": status,
                "schema_version": _SCHEMA_VERSION,
                "input_dir": str(in_dir), "source_npy": str(src_npy),
                "provenance_csv": str(prov_csv), "used_clean_inputs": used_clean,
                "output_parquet": str(out_parquet), "output_csv": str(out_csv),
                "full_size": [H, W], "depth": depth,
                "voxel_size_um": list(self.voxel_size_um) if self.voxel_size_um else None,
                "voxel_volume_um3": vox_um3,
                "n_provenance_rows": len(rows), "n_cells_requested": n_cells,
                "voxel_count_stats": vstats,
                "params": self.get_params(),
            }
            if extra:
                m.update(extra)
            return m

        out_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            with open(manifest_path, "w") as f:
                json.dump(_manifest("dry_run"), f, indent=2, default=str)
            logger.info(
                f"features DRY-RUN: would extract {n_cells} cells "
                f"(no table written). Re-run with dry_run=False."
            )
            for td in tile_dirs:
                result.add_success(td.name)
            return result

        if n_cells == 0:
            logger.warning("features: no cells to extract — writing empty table.")
            import pandas as pd
            empty = pd.DataFrame()
            self._write_outputs(empty, out_parquet, out_csv, logger)
            with open(manifest_path, "w") as f:
                json.dump(_manifest("complete", {
                    "n_cells_extracted": 0, "n_missing": 0, "n_failed": 0,
                    "feature_columns": [], "families": {},
                }), f, indent=2, default=str)
            for td in tile_dirs:
                result.add_success(td.name)
            return result

        # --- per-cell parallel extraction -------------------------------------
        status = "incomplete"
        n_edges = 0          # set by the opt-in adjacency pass (edges written)
        try:
            with open(manifest_path, "w") as f:
                json.dump(_manifest("incomplete"), f, indent=2, default=str)

            # M2: the per-tile probability dir (stage2_reconstruction) — only
            # needed when prob sampling is on.
            recon_dir = str(self._recon_root(tile_dirs)) if self.prob else None
            extracted, n_missing, n_failed = self._extract_parallel(
                tasks, str(src_npy), logger, recon_dir=recon_dir
            )

            import pandas as pd
            df = pd.DataFrame(extracted)

            # is_border_cell from provenance (a crop can't see the tissue boundary),
            # plus carry the worklist provenance fields by dense id.
            if not df.empty:
                # `label` round-trips through a per-row Series (-> float) in the
                # worker; restore it to int so it joins cleanly with provenance.
                df["label"] = df["label"].astype(np.int64)
                ids = df["label"].values
                df["voxel_count"] = [
                    _ri(prov_by_id.get(int(i), {}), "voxel_count") for i in ids
                ]
                df["is_border_cell"] = [
                    int(bool(_ri(prov_by_id.get(int(i), {}), "touches_border")))
                    for i in ids
                ]
                df["seam_status"] = [
                    (prov_by_id.get(int(i), {}) or {}).get("seam_status", "") for i in ids
                ]
                df["n_tiles"] = [
                    _ri(prov_by_id.get(int(i), {}), "n_tiles") for i in ids
                ]
                miou = []
                for i in ids:
                    v = (prov_by_id.get(int(i), {}) or {}).get("min_match_iou", "")
                    miou.append(float(v) if v not in ("", None) else np.nan)
                df["min_match_iou"] = miou

                # Global crowding pass (needs every centroid at once).
                if self.feature_set != "epicycle":
                    qf_centroid_spacing(df, self.k_neighbors, self.density_radius)

                # Wave B — reliability / confidence meta. A pure TABLE
                # pass over the assembled frame: reads major_axis_z_angle /
                # z_span (per-cell, already present) + is_border_cell (set above)
                # + prob_mean (set by the M2 sampler when prob=True), so it runs
                # HERE — after the table is complete — not in the per-cell worker
                # (which can't see is_border_cell). Degrades gracefully when an
                # input column is absent.
                if self.feature_set != "epicycle":
                    from packages.epicycle_scaledseg.core.qc_features import compute_reliability_meta
                    compute_reliability_meta(df)

                # --- structured [FEATURES] timing log helper (Phase A) ---------
                # Each global family logs a ``start``/``done`` pair (wall time +
                # peak RSS) so the SLURM .out carries a per-pass profile. INFO-level
                # (lands in the .out); the per-cell pass already has tqdm via the
                # extractor, so no per-cell heartbeat is added.
                _n_cells = int(len(df))
                _t_global0 = time.perf_counter()
                _on_families: List[str] = []

                from contextlib import contextmanager

                @contextmanager
                def _feat_timer(name: str):
                    logger.info(f"[FEATURES] {name} start (n_cells={_n_cells})")
                    _t0 = time.perf_counter()
                    yield
                    logger.info(
                        f"[FEATURES] {name} done in {time.perf_counter() - _t0:.1f}s "
                        f"(peak RSS {_peak_rss_gb():.1f} GB)")

                # The truly-global field passes (tissue_depth / territory / lumen)
                # need the WHOLE label volume. At field_downsample == 1 (Phase A) they
                # need it RESIDENT (3D EDT / watershed / fill_holes can't be tiled), so
                # load it ONCE here off the memmap as a real COPY and reuse it for every
                # field pass — exactly like reconcile.extract_label_features's
                # ``full_labels`` (so the columns are byte-identical).
                #
                # At field_downsample > 1 (Phase C / S2-S3) the field functions instead
                # build a COARSE foreground/labels volume STREAMING off the source — so
                # we pass the MEMMAP (NOT a resident copy): the field family never
                # loads the full-res volume resident (the whole point of the
                # downsample). The SIGNED per-cell families (directional_contact /
                # surface_partition / ma_apical_basal / apical_junction) still need
                # full-res crops, but they read each cell's bbox CROP off the array, so
                # a memmap is fine (they crop lazily — peak RAM tracks crop, not
                # volume). The adjacency graph deliberately does NOT use this handle —
                # it memmap-crops per cell (S1) — so it is opened only when a field
                # pass needs it.
                _need_field = self.tissue_depth or self.territory or self.lumen
                gvol = None
                # adj_slices is bound by the adjacency block (find_objects on the
                # memmap) when self.adjacency; the field block reuses it (or scans
                # once) so the 5 SIGNED families never re-run find_objects(...astype)
                # — that ``.astype`` on a memmap forces a FULL RESIDENT int32 copy
                # (C1 OOM at field_downsample > 1).
                adj_slices = None
                if _need_field:
                    if self.field_downsample > 1:
                        # MEMMAP — field passes stream the coarse volume; signed
                        # families crop lazily. No full-res resident load.
                        gvol = np.load(str(src_npy), mmap_mode="r")
                        logger.info(f"[FEATURES] field_downsample={self.field_downsample}")
                        # M2 — the downsampled OUTWARD NORMAL (and everything it signs)
                        # is LOW-TRUST on curved tissue: the coarse -∇EDT direction drifts
                        # from the full-res normal (measured median ~11deg / p90 ~57deg at
                        # N=2). Warn so the downstream apical/basal columns are not read as
                        # exact.
                        logger.warning(
                            "[FEATURES] field_downsample=%d: the downsampled OUTWARD "
                            "NORMAL is LOW-TRUST on curved tissue (measured median ~11deg "
                            "/ p90 ~57deg at N=2). Everything it SIGNS — the apical/basal "
                            "tissue frame, directional_contact, surface_partition, "
                            "ma_apical_basal, apical_junction — is approximate at N>1; "
                            "treat those columns as magnitude-only / low-trust.",
                            self.field_downsample)
                    else:
                        gvol = np.array(np.load(str(src_npy)))

                # --- OPT-IN global adjacency / contact-graph pass (S1) ---------
                # Runs the SHARED ``run_adjacency_graph_passes`` (the SAME sequence
                # as reconcile.extract_label_features(adjacency=True), so the graph
                # columns are byte-identical) on a MEMMAP of the global labels: each
                # cell reads only its halo crop, so peak PRIVATE RAM tracks crop
                # size, not volume size (scaling stage S1). Persists the
                # contact graph to ``edges.csv`` next to ``cell_features.csv``.
                # ``df["label"]`` was cast to int above (the graph functions index
                # find_objects with it). NOTE: with ``min_voxels > 1`` a neighbour
                # edge may reference a small cell dropped from the table (the graph
                # sees the WHOLE volume); at the default floor=1 only zero-voxel
                # ghosts are dropped (never neighbours), so every edge endpoint is a
                # table cell.
                if self.adjacency:
                    from scipy.ndimage import find_objects
                    from packages.epicycle_scaledseg.core.qc_features import (
                        run_adjacency_graph_passes, write_edges_csv)
                    with _feat_timer("adjacency"):
                        glab = np.load(str(src_npy), mmap_mode="r")
                        adj_slices = find_objects(glab)   # one memmap scan; bounded RAM
                        # S1.5 (Phase B): parallelise the three VOLUME graph passes
                        # on the SLURM allocation like the per-cell pool. Each worker
                        # opens its OWN memmap of ``src_npy`` (labels_path) so peak
                        # private RAM still tracks crop x workers (S1). n_workers=1
                        # keeps the byte-identical serial path.
                        _adj_nw = self.n_workers or max(1, get_allocated_cpus() - 1)
                        # M3 — the adjacency pass is otherwise silent (the only
                        # n_workers log is the per-cell pool); surface the worker
                        # count + cell count so the operator can confirm it ran
                        # parallel from the head of the .out.
                        logger.info(
                            f"[FEATURES] adjacency n_workers={_adj_nw} "
                            f"(n_cells={len(df)})")
                        run_adjacency_graph_passes(
                            df, glab, slices=adj_slices,
                            spacing=self.voxel_size_um,
                            max_radius=self.adjacency_max_radius,
                            n_workers=_adj_nw, labels_path=str(src_npy),
                            feature_set=self.feature_set)
                        # node_ids = the surviving table cells: with min_voxels > 1 a
                        # sub-floor cell dropped from the table can still be a graph
                        # neighbour, so filter its edges out of edges.csv (no dangling
                        # node). neighbor_count in the table keeps the true physical
                        # count. At the default floor=1 nothing real is dropped.
                        if write_edges_csv(df, out_edges, node_ids=df["label"].to_numpy()):
                            n_edges = len(df.attrs.get("_adjacency_list", []))
                            logger.info(
                                f"features: contact graph -> {out_edges.name} "
                                f"({n_edges} edges)")
                        del glab                          # release the memmap handle
                    _on_families.append("adjacency")

                # `label` round-trips through the per-cell worker's Series as a float;
                # the field passes index find_objects with the int id (the adjacency
                # block above already casts it when adjacency=True, but the field
                # passes can run with adjacency=False — cast here too, mirroring
                # reconcile.extract_label_features's per-block guards).
                if _need_field or self.qc_score:
                    df["label"] = df["label"].astype(np.int64)

                # --- OPT-IN Wave-C tissue-depth ("in/out") global pass ----------
                # 3D EDT of the whole-tissue mask -> per-cell depth-from-surface +
                # the spatial-context covariates + the per-cell OUTWARD normal.
                # compute_tissue_depth runs FIRST (it writes ``tissue_outward_*``);
                # then compute_tissue_frame_features is RE-CALLED so it ADDS the P1a
                # SIGNED apical/basal frame; then the four signed-frame remainders
                # (directional_contact / surface_partition / ma_apical_basal /
                # apical_junction). directional_contact + apical_junction reuse the
                # adjacency contact set, so they NaN every cell when adjacency was off
                # (no contacts to direction-classify) — that is expected. SAME order
                # + same args as reconcile.extract_label_features, so byte-identical.
                if self.tissue_depth and gvol is not None:
                    from scipy.ndimage import find_objects
                    from packages.epicycle_scaledseg.core.qc_features import (
                        compute_tissue_depth, compute_directional_contact,
                        compute_surface_partition, compute_ma_apical_basal,
                        compute_apical_junction, compute_tissue_frame_features,
                    )
                    sp = self.voxel_size_um
                    # C1 FIX — thread ONE bbox list into the 5 SIGNED families so
                    # they SKIP their internal find_objects(gvol.astype(int32)).
                    # That ``.astype`` on a memmap (gvol at field_downsample > 1)
                    # forces a FULL RESIDENT int32 copy -> OOM at scale. Reuse the
                    # adjacency block's scan when present, else scan once here.
                    # find_objects(gvol) STREAMS on a memmap (no materialisation;
                    # established by the S1 audit) and == find_objects(gvol.astype
                    # (int32)) for int-label ids < 2**31 (always true here), so the
                    # per-cell bboxes are unchanged (byte-identical output).
                    field_slices = (adj_slices if adj_slices is not None
                                    else find_objects(gvol))
                    # M4 — give EACH of the 6 tissue-frame functions its OWN
                    # _feat_timer line (was one combined "tissue_depth" timer). If a
                    # run stalls, the .out now pinpoints which of the 6 is running —
                    # in particular the full-volume EDT inside compute_tissue_depth.
                    with _feat_timer("tissue_depth.compute_tissue_depth"):
                        compute_tissue_depth(gvol, df, spacing=sp,
                                             downsample=self.field_downsample)
                    with _feat_timer("tissue_depth.tissue_frame_signed"):
                        compute_tissue_frame_features(
                            df, adjacency=df.attrs.get("_adjacency_list"),
                            spacing=sp, full_labels=gvol, slices=field_slices)
                    with _feat_timer("tissue_depth.directional_contact"):
                        compute_directional_contact(
                            gvol, df, spacing=sp, max_radius=self.adjacency_max_radius,
                            slices=field_slices)
                    with _feat_timer("tissue_depth.surface_partition"):
                        compute_surface_partition(gvol, df, spacing=sp,
                                                  slices=field_slices)
                    with _feat_timer("tissue_depth.ma_apical_basal"):
                        compute_ma_apical_basal(gvol, df, spacing=sp,
                                                slices=field_slices)
                    with _feat_timer("tissue_depth.apical_junction"):
                        compute_apical_junction(
                            gvol, df, spacing=sp, max_radius=self.adjacency_max_radius,
                            slices=field_slices)
                    _on_families.append("tissue_depth")

                # --- OPT-IN P1b lumen-cavity global pass ------------------------
                # binary_fill_holes(foreground) AND NOT foreground -> enclosed
                # cavities; drop border-touching components (external medium) -> each
                # cell's distance_to_lumen + lumen_facing_fraction. All-NaN for a
                # tissue with no enclosed lumen (the correct answer). Independent of
                # adjacency + tissue_depth. Same args as reconcile -> byte-identical.
                if self.lumen and gvol is not None:
                    from packages.epicycle_scaledseg.core.qc_features import compute_lumen_features
                    with _feat_timer("lumen"):
                        compute_lumen_features(gvol, df, spacing=self.voxel_size_um,
                                               downsample=self.field_downsample)
                    _on_families.append("lumen")

                # --- OPT-IN Wave-G watershed-territory ("crowding") global pass -
                # Background-EDT seeded watershed floods each cell to the inter-cell
                # midline -> a Voronoi-ish TERRITORY (breathing-room / crowding).
                # Inherently global (a per-cell crop can't see where territories
                # meet) + independent of adjacency + tissue_depth. Same args as
                # reconcile -> byte-identical.
                if self.territory and gvol is not None:
                    from packages.epicycle_scaledseg.core.qc_features import compute_watershed_territory
                    with _feat_timer("territory"):
                        compute_watershed_territory(gvol, df, spacing=self.voxel_size_um,
                                                    downsample=self.field_downsample)
                    _on_families.append("territory")

                # The resident copy is no longer needed (qc_score is a pure TABLE
                # pass); release it BEFORE _write_outputs to shed the Phase-A RAM.
                if gvol is not None:
                    del gvol

                # --- OPT-IN M7 composite qc_score global pass -------------------
                # Runs DEAD LAST (after every other family) because it standardizes
                # each feature ACROSS the population, so the whole table must exist
                # first. Pure TABLE pass -> needs no volume. Reads the curated
                # PCA_FEATURES subset; several columns come ONLY from the adjacency
                # pass, so warn (don't block) when adjacency was off — it still scores
                # on the labels-only subset. Same method="zscore" as reconcile ->
                # byte-identical.
                if self.qc_score:
                    from packages.epicycle_scaledseg.core.qc_features import (
                        compute_composite_score, PCA_FEATURES)
                    if not self.adjacency:
                        present = [c for c in PCA_FEATURES if c in df.columns]
                        logger.warning(
                            "features: qc_score requested with adjacency=False — "
                            "scoring on the labels-only subset (%d/%d curated "
                            "features present; the neighbour columns require "
                            "adjacency=True for the full score).",
                            len(present), len(PCA_FEATURES))
                    with _feat_timer("qc_score"):
                        compute_composite_score(df, method="zscore")
                    _on_families.append("qc_score")

                # One-line FINAL summary across all the global families.
                if _on_families:
                    logger.info(
                        f"[FEATURES] complete: {_n_cells} cells, "
                        f"families={{{', '.join(_on_families)}}}, "
                        f"total {time.perf_counter() - _t_global0:.1f}s, "
                        f"peak RSS {_peak_rss_gb():.1f} GB")

            self._write_outputs(df, out_parquet, out_csv, logger)

            families = {
                fam: [c for c in cols if c in df.columns]
                for fam, cols in _FEATURE_FAMILIES.items()
            }
            # Per-channel (B2) intensity families are dynamic (prefix is runtime);
            # recover them from the actual columns.
            families.update(_per_channel_families(df.columns, self.channels))
            # Wave E per-channel TEXTURE families are dynamic too (same prefix as
            # the intensity family); recover them alongside.
            families.update(_per_channel_texture_families(df.columns, self.channels))
            # Wave F per-channel LOCALIZATION families are dynamic too (same
            # prefix); recover them alongside.
            families.update(
                _per_channel_localization_families(df.columns, self.channels))
            # Wave G per-channel PUNCTA families are dynamic too (same
            # prefix); recover them alongside.
            families.update(
                _per_channel_puncta_families(df.columns, self.channels))
            # Wave K channel-PAIR co-localization families are
            # templated per unordered channel pair; recover them from the columns.
            families.update(
                _channel_pair_coloc_families(df.columns, self.channels))
            # Wave A local-comparative (C12) families are templated per base
            # feature; recover them from the actual columns too.
            families.update(_local_comparative_families(df.columns))
            with open(manifest_path, "w") as f:
                json.dump(_manifest("complete", {
                    "n_cells_extracted": int(len(df)),
                    "n_missing": int(n_missing), "n_failed": int(n_failed),
                    "feature_columns": list(df.columns),
                    "families": families,
                    "adjacency": self.adjacency,
                    "tissue_depth": self.tissue_depth,
                    "territory": self.territory,
                    "lumen": self.lumen,
                    "qc_score": self.qc_score,
                    "field_downsample": self.field_downsample,
                    "edges_csv": str(out_edges) if (self.adjacency and n_edges) else None,
                    "n_edges": int(n_edges),
                }), f, indent=2, default=str)
            status = "complete"
            logger.info(
                f"features complete: {len(df)} cells -> {out_parquet.name} + "
                f"{out_csv.name} ({n_missing} missing, {n_failed} failed)"
            )
            for td in tile_dirs:
                result.add_success(td.name)
        except Exception as e:  # pragma: no cover - exercised via failure paths
            logger.error(f"features failed: {e}")
            with open(manifest_path, "w") as f:
                json.dump(_manifest("failed", {"error": repr(e)}), f, indent=2, default=str)
            for td in tile_dirs:
                result.add_failure(td.name, str(e))
        return result

    # -- helpers ---------------------------------------------------------------

    def _extract_parallel(self, tasks, src_npy_path, logger, recon_dir=None):
        """Run the per-cell extraction across a worker Pool. Returns (rows, n_missing, n_failed).

        ``recon_dir`` = the per-tile probability dir, threaded into the worker
        initializer alongside the M2 sampler config (``tiles_dir`` / ``channels`` /
        ``prob``) so each worker reads the raw/prob crops itself. All unset -> the
        labels-only path runs unchanged.
        """
        from tqdm import tqdm

        n_workers = self.n_workers or max(1, get_allocated_cpus() - 1)
        n_workers = max(1, min(int(n_workers), len(tasks)))

        # M2 initargs: identical for every worker/cell (the dir paths + channel
        # list + prob flag). Empty when no sampler inputs were configured.
        m2_initargs = (self.tiles_dir, recon_dir, self.channels, self.prob,
                       self.feature_set)

        extracted: List[dict] = []
        n_missing = 0
        n_failed = 0

        def _collect(res):
            nonlocal n_missing, n_failed
            if res is None:
                return
            if isinstance(res, tuple):
                if res[0] == "MISSING":
                    n_missing += 1
                    return
                if res[0] == "ERROR":
                    n_failed += 1
                    logger.warning(f"features: extract failed for cell {res[1]}: {res[2]}")
                    return
            extracted.append(res)

        logger.info(f"features: extracting {len(tasks)} cell(s) on {n_workers} worker(s)")
        global _WORKER_SRC, _WORKER_TILES_DIR, _WORKER_RECON_DIR, _WORKER_CHANNELS, _WORKER_PROB
        global _WORKER_FEATURE_SET
        if n_workers <= 1:
            _init_feature_worker(src_npy_path, *m2_initargs)
            try:
                for t in tqdm(tasks, desc="features extract"):
                    _collect(_extract_one(t))
            finally:
                _WORKER_SRC = None
                _WORKER_TILES_DIR = None
                _WORKER_RECON_DIR = None
                _WORKER_CHANNELS = None
                _WORKER_PROB = False
                _WORKER_FEATURE_SET = "full"
        else:
            from multiprocessing import Pool
            with Pool(
                n_workers, initializer=_init_feature_worker,
                initargs=(src_npy_path, *m2_initargs),
            ) as pool:
                for res in tqdm(
                    pool.imap_unordered(_extract_one, tasks, chunksize=16),
                    total=len(tasks), desc="features extract",
                ):
                    _collect(res)

        if n_missing:
            logger.warning(f"features: {n_missing} cell(s) absent from their own bbox crop "
                           f"(provenance/volume drift) — skipped")
        if n_failed:
            logger.warning(f"features: {n_failed} cell(s) failed extraction — skipped "
                           f"(see warnings above)")
        return extracted, n_missing, n_failed

    def _write_outputs(self, df, out_parquet: Path, out_csv: Path, logger) -> None:
        """Write the per-cell table as Parquet (pyarrow) + CSV.

        CSV always; Parquet best-effort (logs + continues if pyarrow is missing so a
        bad install never loses the extracted table)."""
        # Deterministic row order (imap_unordered returns in completion order) +
        # stable, human-friendly column order: id + provenance first, then features.
        if not df.empty:
            if "label" in df.columns:
                df = df.sort_values("label").reset_index(drop=True)
            lead = [c for c in (
                "label", "voxel_count", "is_border_cell", "seam_status",
                "n_tiles", "min_match_iou",
                "centroid-0", "centroid-1", "centroid-2",
            ) if c in df.columns]
            rest = [c for c in df.columns if c not in lead]
            df = df[lead + rest]
        df.to_csv(out_csv, index=False)
        try:
            df.to_parquet(out_parquet, index=False)
        except Exception as e:  # pragma: no cover - pyarrow missing / write error
            logger.warning(f"features: Parquet write failed ({e!r}); CSV written at {out_csv}")


def qf_centroid_spacing(df, k_neighbors, density_radius):
    """Thin wrapper so the cKDTree pass is import-local (keeps module import light)."""
    from packages.epicycle_scaledseg.core.qc_features import compute_centroid_spacing
    return compute_centroid_spacing(df, k_neighbors=k_neighbors, density_radius=density_radius)


__all__ = ["FeatureStage"]
