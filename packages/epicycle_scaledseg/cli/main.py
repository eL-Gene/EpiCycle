"""
CLI entry point for EpiCycle ScaledSegmentation pipeline.

Usage:
    epicycle-scaledseg run --work-dir /data/tiles --lateral-model /models/xy.pth --axial-model /models/xz.pth
    epicycle-scaledseg run --config /path/to/config.json
    epicycle-scaledseg status --work-dir /data/tiles
    python -m packages.epicycle_scaledseg.cli run ...   (without the console script on PATH)
"""

import argparse
import json
import sys
from pathlib import Path


def _list_tile_dirs(root: Path):
    return sorted([
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(("stage1", "stage2", "stage3", "logs", ".", "epicycle"))
    ])


def _discover_tile_root(work_dir: Path) -> Path:
    tiles_root = work_dir / "tiles"
    if tiles_root.is_dir():
        tile_dirs = _list_tile_dirs(tiles_root)
        if tile_dirs:
            return tiles_root
    return work_dir


def _build_feature_stage(args):
    """Construct the opt-in ``FeatureStage`` from the parsed ``run`` arguments.

    Under ``--feat-feature-set epicycle`` (default) the global-family flags are passed as ``True``
    only when the user asked for them (else ``None``, so the stage forces adjacency + qc_score on);
    the stage skips the engine-only opt-ins itself."""
    from packages.epicycle_scaledseg.core.feature_stage import FeatureStage

    def _opt(flag):
        return True if (flag or args.feat_all) else (None if args.feat_feature_set == "epicycle" else False)

    return FeatureStage(
        input_dir=args.feat_input_dir or args.assembly_out,
        output_dir=args.feat_input_dir or args.assembly_out,
        recon_root=args.recon_out,
        prefer_clean=not args.no_feat_prefer_clean,
        voxel_size_um=tuple(args.feat_voxel_size) if args.feat_voxel_size else None,
        min_voxels=args.feat_min_voxels,
        include_surface=not args.no_feat_surface,
        invert_z=args.feat_invert_z,
        k_neighbors=args.feat_k_neighbors,
        density_radius=args.feat_density_radius,
        adjacency=_opt(args.feat_adjacency),
        adjacency_max_radius=args.feat_adjacency_max_radius,
        tissue_depth=args.feat_tissue_depth or args.feat_all,
        territory=args.feat_territory or args.feat_all,
        lumen=args.feat_lumen or args.feat_all,
        qc_score=_opt(args.feat_qc_score),
        field_downsample=args.feat_field_downsample,
        n_workers=args.feat_workers,
        dry_run=args.features_dry_run,
        feature_set=args.feat_feature_set,
    )


def _should_run_features(mode: str, stages_to_run) -> bool:
    """Whether to add the `features` stage. OPT-IN: full mode only, and NEVER part
    of `all` — it must be requested explicitly (`--stages features`)."""
    return mode == "full" and ("features" in set(stages_to_run))


def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI argument parser (separated from ``main`` so the
    argument surface — defaults, choices — is unit-testable without running)."""
    parser = argparse.ArgumentParser(
        prog="epicycle-scaledseg",
        description="EpiCycle ScaledSegmentation — large-scale 3D cell segmentation pipeline",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_p = sub.add_parser("run", help="Run the segmentation + reconstruction pipeline")
    run_p.add_argument("--work-dir", required=True, help="Root directory containing tile subdirectories (or parent for tiling output)")
    run_p.add_argument("--input-tif", default=None, help="Input Z-stack TIFF for tiling (enables Stage 0)")
    run_p.add_argument("--lateral-model", required=True, help="Path to CellPose lateral (XY) model")
    run_p.add_argument("--axial-model", required=True, help="Path to CellPose axial (XZ/YZ) model")
    run_p.add_argument("--n-gpus", type=int, default=None, help="Number of GPUs (auto-detect if omitted)")
    run_p.add_argument("--log-dir", default=None, help="Log directory (default: <work-dir>/logs)")
    run_p.add_argument("--seg-out", default=None, help="Stage1 segmentation output root")
    run_p.add_argument("--recon-out", default=None, help="Stage2 reconstruction output root")
    run_p.add_argument("--gpu-strategy", choices=["auto", "fixed", "round_robin"], default="auto",
                        help="GPU allocation strategy for segmentation (default: auto — uses fixed/3-GPU-per-tile when >= 3 GPUs, else round_robin)")
    # Segmentation (Cellpose) params — default to get_default_segment_params() if unset.
    run_p.add_argument("--channels", type=int, nargs=2, default=None, metavar=("SIGNAL", "NUCLEUS"),
                        help="Cellpose channels [signal nucleus] (default: 2 1). Use 0 0 for grayscale. "
                             "(Single-channel tiles auto-use 0 0 regardless.)")
    run_p.add_argument("--flow-threshold", type=float, default=None, help="Cellpose flow_threshold (default: 0.2). It only filters Cellpose's 2D masks, which the 3D reconstruction does not use; the paper's runs used 0.2.")
    run_p.add_argument("--cellprob-threshold", type=float, default=None, help="Cellpose cellprob_threshold (default: 0.0)")
    run_p.add_argument("--seg-batch-size", type=int, default=None, help="Cellpose batch_size (default: 1024)")
    run_p.add_argument("--recon-mode", choices=["sequential", "parallel"], default="sequential",
                        help="Reconstruction mode (default: sequential)")
    run_p.add_argument("--recon-max-workers", type=int, default=None,
                        help="Max concurrent tiles in parallel reconstruction. Default: "
                             "auto — sized from RAM + CPU (recon is CPU/RAM-bound, not "
                             "GPU-bound, so this can exceed the GPU count). An integer CAPS "
                             "the auto value.")
    # Per-tile reconstruction gradient-descent overrides (Mode B / full; None = tuned defaults)
    run_p.add_argument("--recon-n-iter", type=int, default=None,
                        help="Per-tile recon: gradient-descent iterations (default: 25)")
    run_p.add_argument("--recon-gradient-decay", type=float, default=None,
                        help="Per-tile recon: gradient decay (default: 0.001; lower = finer)")
    run_p.add_argument("--recon-momenta", type=float, default=None,
                        help="Per-tile recon: gradient-descent momentum (default: 0.98)")
    # Pre-smoothing options (reconstruction)
    run_p.add_argument("--pre-smooth-flows", action="store_true",
                        help="Anisotropic flow pre-smoothing before reconstruction (default: off)")
    run_p.add_argument("--pre-smooth-probs", action="store_true",
                        help="Anisotropic probability pre-smoothing before reconstruction (default: off). The tutorial and the paper's runs use sigma-perp 2.5 and sigma-inplane 1.0.")
    run_p.add_argument("--pre-smooth-sigma-perp", type=float, default=3.0,
                        help="Pre-smooth: sigma along perpendicular axis (default: 3.0)")
    run_p.add_argument("--pre-smooth-sigma-inplane", type=float, default=1.0,
                        help="Pre-smooth: sigma along in-plane axes (default: 1.0)")
    run_p.add_argument("--patch-size", type=int, default=1024, help="Tiling: target XY patch size (default: 1024)")
    run_p.add_argument("--z-threshold-method", choices=["hard", "percentile", "default"], default="hard",
                        help="Tiling: Z-range detection method (default: hard)")
    run_p.add_argument("--z-threshold-value", type=float, default=0.07,
                        help="Tiling: intensity threshold for Z-range (default: 0.07)")
    run_p.add_argument("--channel-index", type=int, default=1,
                        help="Tiling: channel for Z-range detection (default: 1)")
    run_p.add_argument("--input-axes", choices=["auto", "ZCYX", "CZYX", "ZYX"], default="auto",
                        help="Tiling: input Z-stack axis order (default: auto-resolve from "
                             "metadata/shape). Set explicitly for thin-Z stacks (Z<=C) that "
                             "lack axes metadata — auto-resolution is ambiguous for them.")
    run_p.add_argument("--auto-correct-patch", action="store_true",
                        help="Tiling: auto-find valid patch size if exact tiling fails")
    run_p.add_argument("--no-tile-ram-cap", action="store_true",
                        help="Tiling: do NOT shrink the XY tile to fit this machine's RAM "
                             "(default: shrink so one tile's recon fits — no-op on a fat "
                             "node, prevents OOM on a small box; logs when it shrinks).")
    run_p.add_argument("--stages", nargs="+",
                        choices=["tiling", "seg", "recon", "assembly", "global", "iou_stitch",
                                 "postprocess", "features", "all"],
                        default=["all"],
                        help="Which stages to run (default: all). NOTE: 'features' is "
                             "OPT-IN — it is NOT part of 'all'; request it explicitly "
                             "(e.g. --stages features) after postprocess is calibrated.")
    run_p.add_argument("--mode", choices=["coarse", "full"], default="coarse",
                        help="Stitch mode: 'coarse' (Mode A — downsample-then-re-segment "
                             "assembly + global recon) or 'full' (Mode B — full-resolution "
                             "IoU seam-stitch). Default: coarse.")
    run_p.add_argument("--no-binary", action="store_true", help="Skip binary mask extraction")
    run_p.add_argument("--no-prob3d", action="store_true", help="Skip 3D probability output")
    run_p.add_argument("--save-flow3d", action="store_true", help="Save 3D gradient volume")
    run_p.add_argument("--config", default=None, help="JSON config file (overrides CLI args)")
    run_p.add_argument("--resume", action="store_true", help="Resume from checkpoint (skip completed tiles)")
    # Assembly options
    run_p.add_argument("--full-size", type=int, nargs=2, default=[7680, 7680],
                        metavar=("H", "W"), help="Full volume size in pixels (default: 7680 7680)")
    run_p.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels (default: 512)")
    run_p.add_argument("--overlap", type=float, default=0.25, help="Tile overlap fraction (default: 0.25)")
    run_p.add_argument("--depth", type=int, default=None, help="Number of Z slices (auto-detect if omitted)")
    run_p.add_argument("--assembly-out", default=None, help="Stage3 assembly output directory")
    run_p.add_argument("--no-stitch-mask", action="store_true", help="Skip mask volume stitching")
    run_p.add_argument("--no-stitch-prob", action="store_true", help="Skip probability volume stitching")
    run_p.add_argument("--no-stitch-binary", action="store_true", help="Skip binary volume stitching")
    run_p.add_argument("--prob-backend", choices=["numpy", "zarr"], default="numpy",
                        help="Probability stitch backend (coarse mode): 'numpy' (default) "
                             "or 'zarr' (RAM-bounded chunked — avoids the full-volume OOM at "
                             "scale; writes full_probabilities.zarr). Nothing downstream reads "
                             "the stitched prob, so the format change is safe.")
    run_p.add_argument("--prob-stitch-mode", choices=["average", "max"], default="average",
                        help="Overlap blending for probabilities (default: average)")
    # Global reconstruction (Stage 4) options
    run_p.add_argument("--global-out", default=None, help="Stage4 global reconstruction output directory")
    run_p.add_argument("--downsample-factor", type=int, default=2,
                        help="Global recon: downsample factor (default: 2)")
    run_p.add_argument("--global-min-size", type=int, default=1000,
                        help="Global recon: min object size in voxels after downsampling (default: 1000)")
    run_p.add_argument("--no-precomputed-binary", action="store_true",
                        help="Global recon: don't use stitched binary as foreground constraint")
    run_p.add_argument("--global-n-iter", type=int, default=150,
                        help="Global recon: gradient descent iterations (default: 150)")
    run_p.add_argument("--global-gradient-decay", type=float, default=0.1,
                        help="Global recon: gradient decay (default: 0.1, lower = more oversegmentation)")
    # Mode-B full-resolution IoU seam-stitch (Stage iou_stitch) options
    run_p.add_argument("--iou-tau", type=float, default=0.25,
                        help="Mode-B stitch: IoU match threshold (default: 0.25)")
    run_p.add_argument("--iou-vmin", type=int, default=10,
                        help="Mode-B stitch: min intersection voxels for a match (default: 10)")
    run_p.add_argument("--iou-z-block", type=int, default=64,
                        help="Mode-B stitch: Z chunk size for the RAM-bounded seam matcher (default: 64)")
    run_p.add_argument("--iou-one-to-many", choices=["max_iou", "greedy"], default="max_iou",
                        help="Mode-B stitch: seam one-to-many policy — 'max_iou' (1:1 "
                             "under-merge bias, default) or 'greedy' (emit every above-threshold match)")
    # Post-stitch cleanup + smooth (Stage postprocess, Mode B only) options
    run_p.add_argument("--postprocess-dry-run", action="store_true",
                        help="postprocess: decide + log drop counts / voxel-count distribution "
                             "WITHOUT writing the volume (calibrate --pp-min-voxels first)")
    run_p.add_argument("--no-pp-remove-small", action="store_true",
                        help="postprocess: disable the small-fragment size filter")
    run_p.add_argument("--pp-min-voxels", type=int, default=200,
                        help="postprocess: drop cells below this voxel count (default: 200)")
    run_p.add_argument("--pp-remove-large", action="store_true",
                        help="postprocess: ENABLE the large-mask filter (off by default)")
    run_p.add_argument("--pp-max-voxels", type=int, default=0,
                        help="postprocess: drop cells ABOVE this voxel count (0 = off; calibrate via dry-run)")
    run_p.add_argument("--pp-min-percentile", type=float, default=0.0,
                        help="postprocess: drop the smallest N%% by percentile (overrides --pp-min-voxels; 0=off)")
    run_p.add_argument("--pp-max-percentile", type=float, default=0.0,
                        help="postprocess: drop the largest tail by percentile, e.g. 99.5 (overrides --pp-max-voxels; 0=off)")
    run_p.add_argument("--pp-voxel-size", type=float, nargs=3, default=None, metavar=("SZ", "SY", "SX"),
                        help="postprocess: physical voxel size in um (z y x) — reports cell sizes in um^3")
    run_p.add_argument("--no-pp-remove-islands", action="store_true",
                        help="postprocess: disable detached-island removal")
    run_p.add_argument("--pp-island-downsample", type=int, default=8,
                        help="postprocess: occupancy-grid stride for island detection (default: 8)")
    run_p.add_argument("--pp-island-min-frac", type=float, default=0.01,
                        help="postprocess: keep components >= this fraction of the largest (default: 0.01)")
    run_p.add_argument("--pp-island-dilate", type=int, default=1,
                        help="postprocess: occupancy dilation iters to bridge lobes (default: 1)")
    run_p.add_argument("--no-pp-smooth", action="store_true",
                        help="postprocess: disable the per-instance SDF smooth")
    run_p.add_argument("--pp-smooth-sigma", type=float, default=1.5,
                        help="postprocess: isotropic SDF smooth sigma (default: 1.5)")
    run_p.add_argument("--pp-smooth-sigma-z", type=float, default=None,
                        help="postprocess: override Z sigma (anisotropic; default: =sigma)")
    run_p.add_argument("--pp-smooth-sigma-xy", type=float, default=None,
                        help="postprocess: override XY sigma (anisotropic; default: =sigma)")
    run_p.add_argument("--no-pp-preserve-topology", action="store_true",
                        help="postprocess: disable skeleton-reconnect topology preservation")
    run_p.add_argument("--pp-workers", type=int, default=None,
                        help="postprocess: parallel smooth workers (default: allocated CPUs - 1)")
    # Per-cell feature extraction (Stage features, Mode B only, OPT-IN) options
    run_p.add_argument("--features-dry-run", action="store_true",
                        help="features: log the cell count + size distribution WITHOUT "
                             "writing the table (size the job before the long extraction)")
    run_p.add_argument("--feat-voxel-size", type=float, nargs=3, default=None, metavar=("SZ", "SY", "SX"),
                        help="features: physical voxel size in um (z y x) — reports volume in "
                             "um^3, lengths/centroids/crowding in um (default: voxel units)")
    run_p.add_argument("--feat-min-voxels", type=int, default=1,
                        help="features: skip cells below this voxel count (default: 1 = only "
                             "drop zero-voxel provenance ghosts)")
    run_p.add_argument("--no-feat-surface", action="store_true",
                        help="features: skip the marching-cubes surface family (the most "
                             "expensive per-cell op) for a fast morphology-only pass")
    run_p.add_argument("--feat-invert-z", action="store_true",
                        help="features: treat high Z as basal in the cross-section family "
                             "(for inverted stacks; default: low Z is basal)")
    run_p.add_argument("--feat-k-neighbors", type=int, default=6,
                        help="features: K for the centroid mean-KNN crowding distance (default: 6)")
    run_p.add_argument("--feat-density-radius", type=float, default=None,
                        help="features: radius (same unit as centroids) for the crowding "
                             "count/density (default: auto = 3x median nearest-neighbour distance)")
    run_p.add_argument("--feat-input-dir", default=None,
                        help="features: dir holding global_labels_clean.* + provenance "
                             "(default: the stitch/postprocess output, stage4_full_stitch)")
    run_p.add_argument("--no-feat-prefer-clean", action="store_true",
                        help="features: extract off the RAW stitch even when a *_clean output "
                             "exists (default: prefer clean)")
    run_p.add_argument("--feat-workers", type=int, default=None,
                        help="features: parallel extract workers (default: allocated CPUs - 1)")
    run_p.add_argument("--feat-adjacency", action="store_true",
                        help="features: ALSO compute the global contact-graph family "
                             "(adjacency / graph topology / contact geometry / axis "
                             "persistence / motifs / composites) off a memmap of the "
                             "global labels and write the cell contact graph to "
                             "edges.csv (default: off — per-cell families only)")
    run_p.add_argument("--feat-adjacency-max-radius", type=int, default=5,
                        help="features: graduated-dilation search radius (voxels) for "
                             "--feat-adjacency (default: 5, covers typical recon gaps)")
    run_p.add_argument("--feat-tissue-depth", action="store_true",
                        help="features: ALSO compute the truly-global tissue-depth "
                             "('in/out') field — depth-from-surface + spatial context + "
                             "the SIGNED apical/basal frame + directional contact / "
                             "surface partition / ma-apical-basal / apical-junction "
                             "(the contact families need --feat-adjacency too). At "
                             "--feat-field-downsample 1 runs on a RESIDENT copy of the "
                             "global labels; at N>1 streams a coarse field, never "
                             "materialising the full volume (default: off)")
    run_p.add_argument("--feat-territory", action="store_true",
                        help="features: ALSO compute the global watershed-territory "
                             "('crowding') family — a Voronoi-ish territory per cell "
                             "(breathing-room). Resident global pass at "
                             "--feat-field-downsample 1; streams-downsamples at N>1 "
                             "(default: off)")
    run_p.add_argument("--feat-lumen", action="store_true",
                        help="features: ALSO compute the global lumen-cavity family — "
                             "distance to the nearest enclosed cavity + the share of "
                             "the cell surface lining one (all-NaN with no lumen). "
                             "Resident global pass at --feat-field-downsample 1; "
                             "streams-downsamples at N>1 (default: off)")
    run_p.add_argument("--feat-qc-score", action="store_true",
                        help="features: ALSO compute the composite qc_score (a per-cell "
                             "outlier score, standardized across the population). Pure "
                             "table pass, runs last; most meaningful with "
                             "--feat-adjacency (default: off)")
    run_p.add_argument("--feat-field-downsample", type=int, default=1,
                        help="features: integer downsample factor N for the global FIELD "
                             "passes (tissue-depth / territory / lumen). N=1 (default) "
                             "runs them at FULL resolution (byte-identical, full trust). "
                             "N>1 computes the EDT / watershed / fill-holes on a "
                             "foreground downsampled by N, STREAMED off the labels so the "
                             "full-res volume is never resident (the scaling path for "
                             "larger-than-RAM tissue). CAVEAT: at N>1 the per-cell OUTWARD "
                             "NORMAL — and so the SIGNED apical/basal columns (tissue "
                             "frame, directional_contact, surface_partition, "
                             "ma_apical_basal, apical_junction) — is LOW-TRUST on curved "
                             "tissue (median ~11deg / p90 ~57deg drift at N=2); a sub-N "
                             "cell can also drop from the territory columns")
    run_p.add_argument("--feat-all", action="store_true",
                        help="features: convenience — turn ON every global family "
                             "(--feat-adjacency + --feat-tissue-depth + --feat-territory "
                             "+ --feat-lumen + --feat-qc-score). The per-cell families "
                             "always run; this adds the global ones.  Under "
                             "--feat-feature-set epicycle (the default) it means 'all of the "
                             "EpiCycle set' = adjacency + qc_score (already on by default); the "
                             "engine-only tissue-depth / territory / lumen are skipped, not run")
    run_p.add_argument("--feat-feature-set", choices=["epicycle", "full"], default="epicycle",
                        help="features: which columns to measure.  'epicycle' (default) = the "
                             "EpiCycle feature catalogue only (adjacency + qc_score forced on; the "
                             "engine-only families are skipped).  'full' = every engine family "
                             "(~400 columns; the opt-in global families stay opt-in)")

    # --- status ---
    status_p = sub.add_parser("status", help="Show pipeline progress for a work directory")
    status_p.add_argument("--work-dir", required=True, help="Root directory containing tile subdirectories")
    status_p.add_argument("--seg-root", default=None, help="Stage1 output root (default: <work-dir>/stage1_segmentation)")
    status_p.add_argument("--recon-root", default=None, help="Stage2 output root (default: <work-dir>/stage2_reconstruction)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "status":
        _cmd_status(args)


def _cmd_run(args):
    """Execute the pipeline."""
    from packages.epicycle_scaledseg.core.pipeline import EpiCyclePipeline
    from packages.epicycle_scaledseg.core.tiling import TilingStage
    from packages.epicycle_scaledseg.core.segmentation import SegmentationStage
    from packages.epicycle_scaledseg.core.reconstruction import ReconstructionStage
    from packages.epicycle_scaledseg.core.assembly import AssemblyStage
    from packages.epicycle_scaledseg.core.global_reconstruction import GlobalReconstructionStage
    from packages.epicycle_scaledseg.utils.resources import get_allocated_gpus

    # Load config file if provided (merges over CLI args)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        # Config values override CLI defaults but CLI explicit args win
        for key, val in cfg.items():
            cli_key = key.replace("-", "_")
            if hasattr(args, cli_key) and getattr(args, cli_key) is None:
                setattr(args, cli_key, val)

    work_dir = Path(args.work_dir)
    if not work_dir.is_dir():
        print(f"Error: work directory does not exist: {work_dir}", file=sys.stderr)
        sys.exit(1)

    n_gpus = args.n_gpus or get_allocated_gpus()
    log_dir = args.log_dir or str(work_dir / "logs")

    stages_to_run = set(args.stages)
    run_all = "all" in stages_to_run

    # Warn on cross-mode --stages requests that the active mode won't honor (they
    # would silently add no stitch stage and produce no global output).
    if not run_all:
        if args.mode == "coarse" and ({"iou_stitch", "postprocess", "features"} & stages_to_run):
            print("Warning: --stages iou_stitch/postprocess/features are ignored under "
                  "--mode coarse (use --mode full).", file=sys.stderr)
        if args.mode == "full" and ({"assembly", "global"} & stages_to_run):
            print("Warning: --stages assembly/global are ignored under --mode full "
                  "(full mode stitches via iou_stitch).", file=sys.stderr)

    pipeline = EpiCyclePipeline(
        work_dir=str(work_dir),
        n_gpus=n_gpus,
        log_dir=log_dir,
        mode=args.mode,
    )

    # Stage 0: Tiling (if --input-tif provided)
    has_tiling = args.input_tif is not None
    if has_tiling and (run_all or "tiling" in stages_to_run):
        tiles_dir = str(work_dir / "tiles")
        pipeline.add_stage(TilingStage(
            input_tif=args.input_tif,
            output_dir=tiles_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            z_threshold_method=args.z_threshold_method,
            threshold_value=args.z_threshold_value,
            channel_index=args.channel_index,
            auto_correct=args.auto_correct_patch,
            input_axes=args.input_axes,
            auto_ram_cap=not args.no_tile_ram_cap,
        ))

    # Discover tile directories (may be empty if tiling stage will create them)
    tile_root = _discover_tile_root(work_dir)
    tile_dirs = [str(p) for p in _list_tile_dirs(tile_root)]
    if not tile_dirs and not has_tiling:
        print(
            f"Error: no tile directories found in {tile_root}. "
            f"Use --input-tif to tile a Z-stack first.",
            file=sys.stderr,
        )
        sys.exit(1)
    # If tiling will produce tile dirs, seed with a dummy so pipeline can start
    if not tile_dirs and has_tiling:
        tile_dirs = [str(work_dir)]  # placeholder; TilingStage will replace via tile_dirs

    if run_all or "seg" in stages_to_run:
        from packages.epicycle_scaledseg.config.defaults import get_default_segment_params
        segment_params = get_default_segment_params()
        if args.channels is not None:
            segment_params["channels"] = list(args.channels)
        if args.flow_threshold is not None:
            segment_params["flow_threshold"] = args.flow_threshold
        if args.cellprob_threshold is not None:
            segment_params["cellprob_threshold"] = args.cellprob_threshold
        if args.seg_batch_size is not None:
            segment_params["batch_size"] = args.seg_batch_size
        pipeline.add_stage(SegmentationStage(
            lateral_model=args.lateral_model,
            axial_model=args.axial_model,
            segment_params=segment_params,
            gpu_strategy=args.gpu_strategy,
            out_root=args.seg_out,
        ))

    if run_all or "recon" in stages_to_run:
        pipeline.add_stage(ReconstructionStage(
            seg_root=args.seg_out,
            out_root=args.recon_out,
            mode=args.recon_mode,
            n_iter=args.recon_n_iter,
            gradient_decay=args.recon_gradient_decay,
            momenta=args.recon_momenta,
            save_prob3d=not args.no_prob3d,
            save_flow3d=args.save_flow3d,
            save_binary=not args.no_binary,
            max_workers=args.recon_max_workers,
            pre_smooth_flows=args.pre_smooth_flows,
            pre_smooth_probs=args.pre_smooth_probs,
            pre_smooth_sigma_perp=args.pre_smooth_sigma_perp,
            pre_smooth_sigma_inplane=args.pre_smooth_sigma_inplane,
        ))

    # NOTE: the post-recon stitch composition point is the ``if mode == coarse``
    # vs ``if mode == full`` gating below. We deliberately do NOT extract a single
    # ``build_pipeline_stages`` function — that would be too invasive/risky for the
    # Mode-A byte-identicality guarantee. Mode A (coarse) stays exactly as it was;
    # the only change is wrapping it in ``args.mode == "coarse"``. Mode B (full)
    # is a separate, additive branch that appends the IoU stitch stage instead.
    if args.mode == "coarse":
        if run_all or "assembly" in stages_to_run:
            pipeline.add_stage(AssemblyStage(
                recon_root=args.recon_out,
                output_dir=args.assembly_out,
                full_size=tuple(args.full_size),
                tile_size=args.tile_size,
                overlap=args.overlap,
                depth=args.depth,
                stitch_mask=not args.no_stitch_mask,
                stitch_prob=not args.no_stitch_prob and not args.no_prob3d,
                stitch_binary=not args.no_stitch_binary and not args.no_binary,
                prob_stitch_mode=args.prob_stitch_mode,
                prob_backend=args.prob_backend,
            ))

        if run_all or "global" in stages_to_run:
            assembly_out = args.assembly_out or str(work_dir / "stage3_assembly")
            pipeline.add_stage(GlobalReconstructionStage(
                assembly_dir=assembly_out,
                output_dir=args.global_out,
                downsample_factor=args.downsample_factor,
                min_size=args.global_min_size,
                use_precomputed_binary=not args.no_precomputed_binary,
                n_cpu=64,
                indirect_params=None,  # uses defaults; override n_iter/gradient_decay below
            ))

    if args.mode == "full" and (run_all or "iou_stitch" in stages_to_run):
        from packages.epicycle_scaledseg.core.iou_stitch_stage import IoUStitchStage
        pipeline.add_stage(IoUStitchStage(
            recon_root=args.recon_out,
            output_dir=args.assembly_out,
            tau=args.iou_tau,
            vmin=args.iou_vmin,
            one_to_many=args.iou_one_to_many,
            z_block=args.iou_z_block,
        ))

    if args.mode == "full" and (run_all or "postprocess" in stages_to_run):
        from packages.epicycle_scaledseg.core.postprocess_stage import PostprocessStage
        pipeline.add_stage(PostprocessStage(
            input_dir=args.assembly_out,
            output_dir=args.assembly_out,
            recon_root=args.recon_out,
            remove_small=not args.no_pp_remove_small,
            min_voxels=args.pp_min_voxels,
            min_percentile=args.pp_min_percentile,
            remove_large=args.pp_remove_large,
            max_voxels=args.pp_max_voxels,
            max_percentile=args.pp_max_percentile,
            voxel_size_um=tuple(args.pp_voxel_size) if args.pp_voxel_size else None,
            remove_islands=not args.no_pp_remove_islands,
            island_downsample=args.pp_island_downsample,
            island_min_component_frac=args.pp_island_min_frac,
            island_dilate=args.pp_island_dilate,
            smooth=not args.no_pp_smooth,
            smooth_sigma=args.pp_smooth_sigma,
            smooth_sigma_z=args.pp_smooth_sigma_z,
            smooth_sigma_xy=args.pp_smooth_sigma_xy,
            preserve_topology=not args.no_pp_preserve_topology,
            n_workers=args.pp_workers,
            dry_run=args.postprocess_dry_run,
        ))

    # Per-cell feature extraction is OPT-IN (NOT part of "all") — it runs only when
    # explicitly requested, on the cleaned global labels, after postprocess.
    if _should_run_features(args.mode, stages_to_run):
        pipeline.add_stage(_build_feature_stage(args))

    print(f"EpiCycle pipeline: {len(tile_dirs)} tiles, {n_gpus} GPUs")
    print(f"Stages: {[s.name for s in pipeline.stages]}")
    print(f"Logs: {log_dir}")
    print()

    result = pipeline.run(tile_dirs, resume=args.resume)
    print()
    print(result.summary())

    # Exit with non-zero if any failures
    if result.n_success < result.n_total:
        sys.exit(1)


def _cmd_status(args):
    """Report pipeline progress."""
    work_dir = Path(args.work_dir)
    if not work_dir.is_dir():
        print(f"Error: directory does not exist: {work_dir}", file=sys.stderr)
        sys.exit(1)

    tile_root = _discover_tile_root(work_dir)
    tile_dirs = _list_tile_dirs(tile_root)

    seg_root = Path(args.seg_root) if args.seg_root else work_dir / "stage1_segmentation"
    recon_root = Path(args.recon_root) if args.recon_root else work_dir / "stage2_reconstruction"

    total = len(tile_dirs)
    seg_done = 0
    seg_partial = 0
    recon_done = 0

    for td in tile_dirs:
        name = td.name
        seg_dir = seg_root / name

        if (seg_dir / "SEGMENTATION_DONE").exists():
            seg_done += 1
        else:
            # Check for partial views
            views = [v for v in ("xy", "xz", "yz") if (seg_dir / f"{v}.done").exists()]
            if views:
                seg_partial += 1

        recon_dir = recon_root / name
        if (recon_dir / f"{name}_mask.tif").exists():
            recon_done += 1

    print(f"Pipeline status: {work_dir}")
    print(f"  Total tiles: {total}")
    print()
    print(f"  Stage 1 — Segmentation:")
    print(f"    Complete:  {seg_done}/{total}")
    if seg_partial:
        print(f"    Partial:   {seg_partial}/{total} (will be re-segmented)")
    print(f"    Remaining: {total - seg_done}/{total}")
    print()
    print(f"  Stage 2 — Reconstruction:")
    print(f"    Complete:  {recon_done}/{total}")
    print(f"    Remaining: {total - recon_done}/{total}")


if __name__ == "__main__":
    main()
