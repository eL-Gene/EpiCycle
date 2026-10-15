"""Pure, Qt-free QC feature-extraction pipeline.

This module hosts the single canonical implementation of the Step 3 QC
feature-extraction compute chain — the body that previously lived inside
``controller._QCFeatureWorker.run()``.  It is deliberately free of Qt /
napari / signals so the SAME function can run in three contexts with
identical output:

  1. In the extraction **subprocess** (``_qc_extract_subprocess.main``)
     so the napari UI stays live during the 100-270s GIL-bound compute.
  2. As an **in-process fallback** on a QThread when subprocess spawn
     fails (env/launcher issues) — preserves the pre-subprocess behavior
     so extraction never just fails.
  3. Directly from **tests** to prove the subprocess output is
     byte-for-byte identical to in-process output (the equivalence test).

PERF-ONLY RELOCATION (2026-06-07, feature/step3-extract-subprocess): the
body below is moved VERBATIM from ``_QCFeatureWorker.run`` — same calls,
same order, same outputs.  The only structural change is that the Qt
``self.log.emit(...)`` lines became calls to an injected ``log`` callback
and the ``[QC-PERF]`` lap markers became calls to an injected ``progress``
callback.  No ``compute_*`` / ``run_extraction`` / ``save_patch_csv`` /
``persist_qc_artifacts`` logic changed — only WHERE the chain runs.

The extraction-context sidecar is intentionally NOT written here: it
depends on the controller's live ``_current_downsample_factor`` /
``state_manager`` and is written by ``_write_qc_extraction_context`` in
``_on_qc_extract_finished`` exactly as before.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ``progress(phase_name: str, ms: float)`` — emitted after each phase with
# the elapsed milliseconds, mirroring the old ``[QC-PERF] phase.<name>``
# rolling timer.  ``log(message: str)`` — free-text progress lines (the old
# ``self.log.emit`` surface).  Both optional; default to no-ops.
ProgressCb = Callable[[str, float], None]
LogCb = Callable[[str], None]


def run_full_qc_extraction(
    *,
    labels_path: str,
    source_stack_path: str = "",
    prob_path: str = "",
    csv_path: str,
    patch_name: str,
    spacing: Optional[Sequence[float]] = None,
    erosion_iters: int = 2,
    max_radius: int = 5,
    composite_method: str = "zscore",
    seg_channel: int = 0,
    pca_features: Optional[List[str]] = None,
    n_pcs: int = 5,
    compute_umap: bool = False,
    umap_min_dist: float = 0.10,
    umap_n_neighbors: int = 15,
    compute_tsne: bool = False,
    tsne_perplexity: float = 30.0,
    channel_configs: Optional[list] = None,
    exclude_border: bool = False,
    invert_z: bool = False,
    metric: str = "euclidean",
    patch_status: str = "unknown",
    progress: Optional[ProgressCb] = None,
    log: Optional[LogCb] = None,
    spacing_from_file: bool = False,
    spacing_workspace: str = "",
) -> Dict[str, Any]:
    """Run the full Step 3 QC feature-extraction chain on disk inputs.

    Loads the disk inputs, runs ``run_extraction`` + PCA + composite +
    optional UMAP / t-SNE, then ``save_patch_csv`` + ``persist_qc_artifacts``.
    Returns a summary dict the caller (worker shim / subprocess / test) uses
    to drive the post-extraction UI handler.

    Args mirror ``_QCFeatureWorker.__init__`` 1:1 (all disk paths + scalar
    config).  ``progress`` / ``log`` are optional callbacks for live markers.

    Returns:
        dict with keys:
            ``csv_path``       — str, the canonical features CSV written
            ``rows``           — int, number of label rows extracted
            ``n_labels``       — int, distinct labels in the volume (== rows)
            ``shape``          — tuple, the labels volume shape
            ``elapsed``        — float, total wall-clock seconds
            ``adjacency_list`` — list[dict], the worker's full neighbor edge
                                 list (from ``df.attrs["_adjacency_list"]``).
                                 The subprocess persists this so the
                                 controller can re-attach it to the re-read
                                 df and reproduce the adjacency_curated.csv
                                 behavior byte-for-byte.
            ``artifacts``      — dict, the ``persist_qc_artifacts`` result.
    """
    channel_configs = channel_configs or []
    # R7: a patch with no voxel size
    # of its own -- the caller passes ``spacing_from_file=True`` and its
    # fallback (QC Settings' value, or None) as *spacing*.  The patch
    # file's own tags are read HERE (a worker / the subprocess), never on
    # the main thread; what is found is used and returned in the summary
    # (``spacing_found``) for the main thread to store.
    spacing_found = resolve_patch_spacing_from_file(
        source_stack_path, spacing_workspace) if spacing_from_file else None
    if spacing_found is not None:
        spacing = list(spacing_found["zyx"])

    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    # ---- everything below is the VERBATIM body of _QCFeatureWorker.run ----
    import tifffile
    from .qc_features import (
        compute_composite_score,
        save_patch_csv,
        persist_qc_artifacts,
    )
    from napari_epicycle.segmentation.feature_registry import (
        build_registry, run_extraction,
    )

    t0 = time.time()
    # [QC-PERF] rolling per-phase timer.  The extraction runs GIL-bound
    # (per-label loops + column-by-column df build), so this breakdown
    # drives the fix.  Mirrored to the
    # injected ``progress`` callback.
    _qcp_prev = [t0]

    def _qcp_lap(_label):
        _now = time.time()
        _ms = (_now - _qcp_prev[0]) * 1000.0
        print(f"[QC-PERF] phase.{_label} = {_ms:.0f} ms", flush=True)
        if progress is not None:
            # NOTE: progress exceptions are intentionally NOT swallowed here —
            # the subprocess uses the progress callback to raise its
            # cancellation exception between phases.  Swallowing it would make
            # mid-extract cancellation impossible.  (The callback is the
            # caller's own code; it is responsible for its own error hygiene.)
            progress(_label, _ms)
        _qcp_prev[0] = _now

    # 2026-05-31 [QC-EXTRACT] transparency: tagged with patch_name so the
    # user can correlate to which patch the worker is operating on.
    _start_msg = (
        f"[QC-EXTRACT] Worker START patch={patch_name!r} "
        f"labels_path={Path(labels_path).name!r} "
        f"spacing={spacing} erosion={erosion_iters} "
        f"max_radius={max_radius}"
    )
    print(_start_msg, flush=True)
    _log(_start_msg)
    _log(f"Loading labels from {Path(labels_path).name}...")
    labels = tifffile.imread(labels_path)
    n_labels = len(np.unique(labels)) - 1  # exclude background
    shape = labels.shape
    voxels = labels.size
    _log(
        f"Volume: {shape[0]}x{shape[1]}x{shape[2]} "
        f"({voxels/1e6:.1f}M voxels, {n_labels} labels) — computing features..."
    )
    _vol_msg = (
        f"[QC-EXTRACT] Volume loaded patch={patch_name!r} "
        f"shape={shape} n_labels={n_labels} "
        f"load_time={time.time()-t0:.2f}s"
    )
    print(_vol_msg, flush=True)
    _log(_vol_msg)
    _qcp_lap("load_labels+count")  # imread + np.unique label count

    # Build the available_inputs dict the registry consumes.
    available: dict = {
        "labels": labels,
        "erosion_iters": erosion_iters,
        "max_radius": max_radius,
        "seg_channel": seg_channel,
    }
    if spacing is not None:
        available["spacing"] = spacing
    if exclude_border:
        available["exclude_border"] = True
    if invert_z:
        available["invert_z"] = True

    # Source stack (multi-channel TIFF).  Per-channel registry entries call
    # ``extract_single_channel`` internally on ``raw_stack``.
    if source_stack_path and Path(source_stack_path).exists():
        if channel_configs:
            _log("Loading source stack for per-channel intensity...")
            available["raw_stack"] = tifffile.imread(source_stack_path)
        else:
            _log("[info] No channels selected — skipping intensity")

    # Probability volume (gates the registry's "probability" group).
    if prob_path and Path(prob_path).exists():
        _log("Loading probability volume...")
        prob_3d = np.load(prob_path)
        _log(
            f"Probability volume: {prob_3d.shape}, dtype={prob_3d.dtype}"
        )
        available["prob_volume"] = prob_3d

    # Build registry with the user's checked channels.  Drop the
    # "composite" group — Step 3 invokes compute_composite_score
    # explicitly below with the dialog's method/features/n_pcs.
    registry = build_registry(
        checked_channels=channel_configs or None,
    )
    registry = [g for g in registry if g.name != "composite"]
    _log(
        f"Registry: {len(registry)} groups, inputs: "
        f"{sorted(available.keys())}"
    )

    _qcp_lap("load_source+prob+registry")

    df = run_extraction(
        registry,
        available,
        progress_cb=lambda pct: None,  # log per-group instead
    )
    _qcp_lap("run_extraction_all_groups")

    # Free large arrays held only for extraction.
    available.pop("prob_volume", None)
    available.pop("raw_stack", None)

    # Phase E — canonical PCA persistence.  Fit PCA once and write PC1..PCN
    # so downstream readers share the same coordinate system.  Must run
    # AFTER run_extraction (is_border_cell exists) and BEFORE
    # compute_composite_score (method='pca' L2-norm reads the PCs).
    from .qc_features import compute_pca_columns
    _log(
        f"Computing PCA (n_components={n_pcs}, "
        f"exclude_border={exclude_border})..."
    )
    compute_pca_columns(
        df,
        n_components=n_pcs,
        features=pca_features,
        exclude_border=exclude_border,
    )
    _qcp_lap("pca")

    # Composite QC score — runs outside the registry because n_pcs /
    # method / features come from the dialog.
    compute_composite_score(
        df,
        method=composite_method,
        features=pca_features,
        exclude_border=exclude_border,
        n_pcs=n_pcs,
    )
    _qcp_lap("composite")

    # Phase A.4 — opt-in 2D nonlinear embeddings.  Both share the n_pcs PCA
    # pre-reduction so the embedding lives in the same denoised space as
    # Leiden clustering.
    if compute_umap:
        _log(
            f"Computing UMAP (n_neighbors={umap_n_neighbors}, "
            f"min_dist={umap_min_dist})..."
        )
        from .qc_features import compute_umap_embedding
        compute_umap_embedding(
            df,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            features=pca_features,
            pca_dims=n_pcs,
            metric=metric,
        )
        _qcp_lap("umap")
    if compute_tsne:
        _log(f"Computing t-SNE (perplexity={tsne_perplexity})...")
        from .qc_features import compute_tsne_embedding
        compute_tsne_embedding(
            df,
            perplexity=tsne_perplexity,
            features=pca_features,
            pca_dims=n_pcs,
            metric=metric,
        )
        _qcp_lap("tsne")

    elapsed = time.time() - t0
    _log(f"Done in {elapsed:.1f}s — {len(df)} labels extracted")
    save_patch_csv(df, csv_path)
    # 2026-05-20 Step 3 restructure: persist net-new artifacts
    # (features_raw.csv backup, adjacency_raw.csv / adjacency_curated.csv).
    artifacts: Dict[str, Any] = {}
    try:
        artifacts = persist_qc_artifacts(
            df, csv_path,
            patch_status=patch_status,
        )
        if artifacts["raw_backup_written"]:
            _log("Wrote features_raw.csv backup (initial extraction)")
        if artifacts["raw_skipped_post_curation"]:
            _log(
                f"[NOTE] Skipped features_raw.csv backup — first extraction "
                f"occurred POST-curation (status={patch_status}); raw "
                f"pre-curation snapshot unavailable."
            )
        if artifacts["adjacency_written"]:
            kind = "raw" if artifacts["is_pre_curation"] else "curated"
            _log(
                f"Wrote adjacency_{kind}.csv "
                f"({Path(artifacts['adjacency_path']).name})"
            )
    except Exception as _persist_err:
        # Never let auxiliary persistence fail the extraction.
        _log(f"[WARN] QC artifact persistence raised: {_persist_err}")

    _qcp_lap("save_csv+persist_artifacts")
    _done_msg = (
        f"[QC-EXTRACT] Worker DONE patch={patch_name!r} "
        f"total_time={time.time()-t0:.2f}s rows={len(df)} "
        f"csv={Path(csv_path).name!r}"
    )
    print(_done_msg, flush=True)
    _log(_done_msg)

    # The full worker adjacency edge list — captured here so the subprocess
    # can persist it and the controller can re-attach it to the re-read df,
    # reproducing the adjacency_curated.csv re-extract behavior byte-for-byte
    # (the CSV does NOT carry df.attrs, so a naive re-read would lose it).
    adjacency_list = list(df.attrs.get("_adjacency_list") or [])

    return {
        "csv_path": csv_path,
        "rows": int(len(df)),
        "n_labels": int(n_labels),
        "shape": tuple(int(s) for s in shape),
        "elapsed": float(elapsed),
        "adjacency_list": adjacency_list,
        "artifacts": artifacts,
        "spacing_found": spacing_found,
    }


def resolve_patch_spacing_from_file(stack_path: str,
                                    workspace: str = "") -> Optional[dict]:
    """A patch's voxel size read with ``read_file=True`` (R7): Step 1's
    registry (a processed patch's store scale included), else the file's
    own tags.  ``{"zyx", "source"}`` or ``None``.  For a worker / the
    subprocess only -- it opens the file.  Never raises."""
    if not stack_path:
        return None
    try:
        from napari_epicycle.data_preparation._voxel_spacing import (
            read_step1_state, resolve_stack_spacing,
        )
        return resolve_stack_spacing(None, read_step1_state(workspace),
                                     stack_path, workspace, read_file=True)
    except Exception:      # noqa: BLE001 -- a size, never a crash
        return None


__all__ = ["run_full_qc_extraction", "resolve_patch_spacing_from_file"]
