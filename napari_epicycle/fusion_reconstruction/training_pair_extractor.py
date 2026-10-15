"""Extract 2D training pairs from curated 3D volumes for model fine-tuning.

Slices curated/finalized 3D label volumes along orthogonal axes (XY, XZ, YZ)
and pairs them with corresponding intensity images.  Output is organized as
spatial (XY) and axial (XZ/YZ) pairs, compatible with Cellpose training format.

Used by the Model Refinement section of Reconstruct + Curate 3D.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _paths


# ---------------------------------------------------------------------------
# Mask cleaning (ported from epicycle-training sampler.py)
# ---------------------------------------------------------------------------

def clean_mask_slice(
    mask: np.ndarray,
    *,
    remove_boundaries: bool = True,
    relabel: bool = True,
    min_size: int = 100,
    smoothing_disk_size: int = 1,
    expand_labels_distance: int = 0,
) -> np.ndarray:
    """Clean a 2D label mask for Cellpose training.

    Pipeline order (each step is independently toggleable):
      1. Remove boundaries — erode each label by 1px to separate touching cells
      2. Relabel — assign unique IDs to spatially disconnected regions
      3. Size filter — remove regions smaller than *min_size* pixels
      4. Smooth — morphological opening+closing per region
      5. Expand labels — grow each label outward by N px, stopping at
         adjacent labels (no merging).  Compensates for step 1's
         erosion bias.  Disabled by default (distance=0).

    Args:
        mask: 2D integer label array (H, W).
        remove_boundaries: Erode 1px inner boundaries between labels.
        relabel: Relabel connected components with unique IDs.
        min_size: Remove regions with fewer pixels (0 = keep all).
        smoothing_disk_size: Disk radius for morphological smoothing (0 = off).
        expand_labels_distance: Px to expand each label outward AFTER
            the size/smooth steps, using ``skimage.segmentation.expand_labels``.
            ``expand_labels`` stops at adjacent labels, so cells don't
            merge.  Default 0 (disabled).  Typical value: 1 to undo
            the 1px erosion from ``remove_boundaries=True``.

    Returns:
        Cleaned 2D label array with the same shape as *mask*.

    Why expand_labels exists (2026-05-21):
        HITL boundary-erosion bias.  ``remove_boundaries=True`` (the
        default, needed to separate touching cells for Cellpose) shaves
        1 px off every label's outer boundary.  Trained models then
        learn under-segmentation as the correct prediction.  Iterating
        over refinement rounds, masks shrink monotonically.
        ``expand_labels_distance=1`` adds that 1 px back AFTER
        relabeling, so cells stay separated but recover original size.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    if np.count_nonzero(mask) == 0:
        return mask.copy()

    # Fast path: everything disabled
    if (not remove_boundaries and not relabel and min_size <= 0
            and smoothing_disk_size <= 0 and expand_labels_distance <= 0):
        return mask.copy()

    from skimage.segmentation import find_boundaries, expand_labels
    from skimage.measure import label as sk_label, regionprops
    from skimage.morphology import opening, closing, disk

    cleaned = mask.copy()

    # Step 1: Remove boundaries
    if remove_boundaries:
        boundary = find_boundaries(cleaned, mode="inner")
        cleaned[boundary] = 0

    # Step 2: Relabel connected components
    if relabel:
        cleaned = sk_label(cleaned, connectivity=1).astype(np.int32)

    # Steps 3+4: Size filter and smoothing
    if min_size > 0 or smoothing_disk_size > 0:
        selem = disk(smoothing_disk_size) if smoothing_disk_size > 0 else None
        regions = regionprops(cleaned)
        final = np.zeros_like(cleaned)
        for region in regions:
            if min_size > 0 and region.area < min_size:
                continue
            region_mask = (cleaned == region.label)
            if selem is not None:
                region_mask = opening(region_mask, selem)
                region_mask = closing(region_mask, selem)
            final[region_mask] = region.label
        cleaned = final

    # Step 5: Expand labels — compensate for boundary erosion bias.
    # `expand_labels` stops at neighbors so labels never merge.
    if expand_labels_distance > 0:
        cleaned = expand_labels(
            cleaned, distance=int(expand_labels_distance),
        )

    return cleaned.astype(np.int32)


# ---------------------------------------------------------------------------
# Slice sampling
# ---------------------------------------------------------------------------

def _sample_with_min_distance(
    indices: List[int],
    num_samples: int,
    min_distance: int,
) -> List[int]:
    """Sample indices uniformly across the input range.

    2026-05-19: Replaces the previous greedy-from-bottom selector
    that silently dropped the top of every stack.  Pre-fix behavior:
    iterate ascending, take the lowest index that satisfies
    min_distance, stop when quota fills.  With quota=50 and 200
    valid indices, this captured indices[0]..indices[49] and missed
    indices[50]..indices[199] — the top three-quarters of the stack
    was never sampled.  For stratified epithelial stacks where
    apical and basal halves look genuinely different, this
    systematically underweighted the apical surface in training.

    Post-fix behavior: uniform-stride sampling picks *num_samples*
    evenly-spaced positions across the full sorted index list.  The
    ``(i + 0.5)/K`` centering avoids edge bias — for K=50 from
    N=200 the chosen list positions are {2, 6, 10, ..., 198}, which
    span the full range.

    ``min_distance`` is preserved as a soft floor.  If the natural
    stride exceeds it (the common case when the cap actually bites)
    this is a no-op.  When the cap is large relative to the valid
    set so stride < min_distance, over-dense picks are dropped and
    the returned list is shorter than ``num_samples``.

    When ``num_samples >= len(indices)`` the cap doesn't bite — all
    indices are returned without filtering, because the user asked
    for "everything that qualifies."
    """
    if not indices or num_samples <= 0:
        return []

    sorted_idx = sorted(indices)
    n = len(sorted_idx)

    if num_samples >= n:
        # Cap doesn't bite — caller asked for everything.
        return list(sorted_idx)

    # Uniform-stride pick: K evenly-spaced list positions.  The
    # (i + 0.5) centering means the first pick lands ~half a stride
    # past the bottom and the last pick lands ~half a stride before
    # the top, giving symmetric coverage.
    picks: List[int] = []
    for i in range(num_samples):
        pos = int((i + 0.5) * n / num_samples)
        pos = min(pos, n - 1)
        picks.append(sorted_idx[pos])

    # Enforce min_distance as a soft floor on the picked indices.
    # No-op when stride already exceeds min_distance (the common case).
    if min_distance <= 1:
        return picks
    filtered: List[int] = []
    for p in picks:
        if not filtered or p - filtered[-1] >= min_distance:
            filtered.append(p)
    return filtered


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_training_data(
    stems: List[str],
    *,
    train_frac: float = 0.8,
    spatial_split: bool = False,
    chunk_size: int = 10,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Split file stems into train and validation sets.

    Args:
        stems: List of file stem strings to split.
        train_frac: Fraction allocated to training (0.5–1.0).
        spatial_split: If True, split by contiguous chunks so adjacent
            slices stay together (prevents data leakage).
        chunk_size: Number of contiguous items per spatial chunk.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_stems, val_stems).
    """
    if not stems:
        return [], []

    rng = np.random.RandomState(seed)

    if not spatial_split:
        # Random shuffle split
        indices = list(range(len(stems)))
        rng.shuffle(indices)
        n_train = max(1, round(len(stems) * train_frac))
        if train_frac >= 1.0:
            n_train = len(stems)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        return (
            [stems[i] for i in sorted(train_idx)],
            [stems[i] for i in sorted(val_idx)],
        )

    # Spatial chunking: group into contiguous blocks, assign blocks
    n_chunks = max(1, (len(stems) + chunk_size - 1) // chunk_size)
    chunk_indices = list(range(n_chunks))
    rng.shuffle(chunk_indices)
    n_train_chunks = max(1, round(n_chunks * train_frac))
    if train_frac >= 1.0:
        n_train_chunks = n_chunks
    train_chunks = set(chunk_indices[:n_train_chunks])

    train_stems: List[str] = []
    val_stems: List[str] = []
    for i, stem in enumerate(stems):
        chunk_id = i // chunk_size
        if chunk_id in train_chunks:
            train_stems.append(stem)
        else:
            val_stems.append(stem)

    return train_stems, val_stems


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_training_pairs_from_3d(
    image_3d: np.ndarray,
    labels_3d: np.ndarray,
    output_dir: Union[str, Path],
    *,
    min_labels_per_slice: int = 3,
    views: Sequence[str] = ("xy", "xz", "yz"),
    # Slice sampling
    edge_buffer: int = 0,
    slices_per_patch: int = 0,
    min_distance: int = 1,
    # Mask cleaning
    remove_boundaries: bool = False,
    relabel: bool = False,
    min_size: int = 0,
    smoothing_disk_size: int = 0,
    expand_labels_distance: int = 0,
    # 2026-05-21 axes-aware input normalization.  Mirrors the
    # equivalent block in `epicycle_reconstruction.logic.segment_views`
    # for inference.  Pre-fix the extractor assumed 4D input was
    # always ZCYX; CZYX inputs would slice along the C axis instead
    # of Z, producing garbage XY slices.  Default "ZCYX" preserves
    # legacy behavior for callers that don't pass the kwarg.
    axes: str = "ZCYX",
    # 2026-05-19 (feature/refinement-extraction-redesign):
    # Per-orientation overrides.  When provided, override the scalar
    # params above for THAT orientation only.  Lateral = xy slices;
    # axial = xz + yz slices.  Each dict may contain any subset of
    # the scalar param names — unspecified keys fall back to the
    # scalar value.  None (default) means "use scalars uniformly."
    lateral_params: Optional[Dict[str, object]] = None,
    axial_params: Optional[Dict[str, object]] = None,
    # Progress callback: (current_slice, total_slices)
    progress_cb: Optional[object] = None,
) -> Dict[str, int]:
    """Extract orthogonal 2D slices as image+mask training pairs.

    Args:
        image_3d: 3D intensity volume — (Z, Y, X) or (Z, C, Y, X).
        labels_3d: 3D label volume — (Z, Y, X), int dtype.
        output_dir: Directory to save pairs into spatial/ and axial/ subdirs.
        min_labels_per_slice: Skip slices with fewer unique nonzero labels.
        views: Which orthogonal views to extract.
        edge_buffer: Skip slices within this distance of volume edges.
        slices_per_patch: Max slices per view (0 = all valid slices).
        min_distance: Min spacing between sampled slices.
        remove_boundaries: Erode 1px boundaries between touching labels.
        relabel: Relabel connected components with unique IDs.
        min_size: Remove regions smaller than this (pixels). 0 = keep all.
        smoothing_disk_size: Morphological smoothing disk radius. 0 = off.

    Returns:
        Dict with keys ``spatial``, ``axial``, ``total``, ``output_dir``.
    """
    import tifffile

    output_dir = Path(output_dir)
    spatial_dir = output_dir / "spatial"
    axial_dir = output_dir / "axial"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    axial_dir.mkdir(parents=True, exist_ok=True)

    # 2026-05-19: Resolve per-orientation effective params.  Scalar
    # kwargs serve as defaults; lateral_params / axial_params (if
    # given) override their corresponding keys.
    _scalar_defaults = {
        "min_labels_per_slice": min_labels_per_slice,
        "edge_buffer": edge_buffer,
        "slices_per_patch": slices_per_patch,
        "min_distance": min_distance,
        "remove_boundaries": remove_boundaries,
        "relabel": relabel,
        "min_size": min_size,
        "smoothing_disk_size": smoothing_disk_size,
        "expand_labels_distance": expand_labels_distance,
    }
    _lateral = {**_scalar_defaults, **(lateral_params or {})}
    _axial = {**_scalar_defaults, **(axial_params or {})}
    print(
        f"[PERVIEW-EXTRACT] resolved per-view params: "
        f"lateral.min_labels={_lateral['min_labels_per_slice']} "
        f"lateral.smooth={_lateral['smoothing_disk_size']} "
        f"lateral.min_size={_lateral['min_size']} "
        f"lateral.expand={_lateral['expand_labels_distance']} | "
        f"axial.min_labels={_axial['min_labels_per_slice']} "
        f"axial.smooth={_axial['smoothing_disk_size']} "
        f"axial.min_size={_axial['min_size']} "
        f"axial.expand={_axial['expand_labels_distance']}"
    )

    # Accept (Z, Y, X) or (Z, C, Y, X) / (C, Z, Y, X) images.
    img = np.asarray(image_3d)
    if img.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D image, got shape {img.shape}")

    # 2026-05-21 axes-aware normalization.  Mirror the equivalent in
    # `epicycle_reconstruction.logic.segment_views` so both pipelines
    # accept the same set of layouts.  After this block, multichannel
    # 4D input is ALWAYS (Z, C, Y, X) and the rest of the function
    # can assume that layout.
    #
    # 2026-05-21 audit fix: defensive `axes is None` handling.
    # Pre-fix `axes.upper()` would AttributeError on a None argument
    # — controller has try/except guards but direct callers don't.
    # Treat None as "use default ZCYX" for back-compat with any
    # caller that explicitly passes None.
    multichannel = img.ndim == 4
    if multichannel:
        if axes is None:
            axes_upper = "ZCYX"
        else:
            axes_upper = str(axes).upper()
        if axes_upper == "CZYX":
            # (C, Z, Y, X) → (Z, C, Y, X).  Pre-fix the slicing code
            # would treat axis 0 as Z but it's actually C — every XY
            # "slice" was a full Z stack, garbage training data.
            img = np.transpose(img, (1, 0, 2, 3))
            print(
                f"[PERVIEW-EXTRACT] normalized CZYX input to ZCYX "
                f"(shape {image_3d.shape if hasattr(image_3d, 'shape') else '?'} -> {img.shape})"
            )
        elif axes_upper != "ZCYX":
            raise ValueError(
                f"Unsupported axes {axes!r} for multichannel input; "
                f"expected 'ZCYX' or 'CZYX'."
            )

    lbl = np.asarray(labels_3d)
    if lbl.ndim != 3:
        raise ValueError(f"Expected 3D labels, got shape {lbl.shape}")

    spatial_count = 0
    axial_count = 0

    # Pass 1: build work list with per-view effective params.
    # work_list entries now carry the cleaning params they were
    # resolved with, so the threadpool workers below apply the
    # correct cleaning per view.
    work_list: List[Tuple[str, int, Path, Dict[str, object]]] = []

    for view in views:
        view_params = _lateral if view == "xy" else _axial
        if view == "xy":
            n_slices = lbl.shape[0]
            dest = spatial_dir
        elif view == "xz":
            n_slices = lbl.shape[1]
            dest = axial_dir
        elif view == "yz":
            n_slices = lbl.shape[2]
            dest = axial_dir
        else:
            raise ValueError(f"Unknown view '{view}', expected xy/xz/yz")

        v_edge_buf = int(view_params["edge_buffer"])
        v_min_lab = int(view_params["min_labels_per_slice"])
        v_slices_cap = int(view_params["slices_per_patch"])
        v_min_dist = int(view_params["min_distance"])

        # Build candidate index list with edge buffer
        lo = v_edge_buf
        hi = n_slices - v_edge_buf
        if hi <= lo:
            print(
                f"[PERVIEW-EXTRACT] view={view} edge_buffer={v_edge_buf} "
                f"leaves no candidate indices (n_slices={n_slices})"
            )
            continue
        candidate_indices = list(range(lo, hi))

        # Filter by min_labels
        valid_indices = []
        for idx in candidate_indices:
            if view == "xy":
                lbl_slice = lbl[idx]
            elif view == "xz":
                lbl_slice = lbl[:, idx, :]
            else:
                lbl_slice = lbl[:, :, idx]

            n_labels = len(np.unique(lbl_slice[lbl_slice > 0]))
            if n_labels >= v_min_lab:
                valid_indices.append(idx)

        # Apply sampling if slices_per_patch > 0
        n_before_cap = len(valid_indices)
        if v_slices_cap > 0 and len(valid_indices) > v_slices_cap:
            valid_indices = _sample_with_min_distance(
                valid_indices, v_slices_cap, v_min_dist,
            )
        print(
            f"[PERVIEW-EXTRACT] view={view} candidates={len(candidate_indices)} "
            f"valid_by_min_labels={n_before_cap} "
            f"after_cap(slices_per_patch={v_slices_cap})={len(valid_indices)}"
        )

        # Per-slice cleaning params are pre-resolved and bound into
        # each work-list entry so the threadpool workers don't need
        # to recompute or share state.
        cleaning_kwargs = {
            "remove_boundaries": bool(view_params["remove_boundaries"]),
            "relabel": bool(view_params["relabel"]),
            "min_size": int(view_params["min_size"]),
            "smoothing_disk_size": int(view_params["smoothing_disk_size"]),
            "expand_labels_distance": int(
                view_params.get("expand_labels_distance", 0)
            ),
        }
        for idx in valid_indices:
            work_list.append((view, idx, dest, cleaning_kwargs))

    total_slices = len(work_list)

    # Pre-slice all data in main thread (numpy indexing is fast, shared
    # memory).  Each slice_data tuple now carries the per-view
    # cleaning_kwargs resolved during work-list building, so workers
    # apply the correct preprocessing without revisiting view→params.
    slice_data: List[
        Tuple[np.ndarray, np.ndarray, str, Path, Dict[str, object]]
    ] = []
    for view, idx, dest, cleaning_kwargs in work_list:
        if view == "xy":
            # Source: (Z, C, Y, X) multichannel → img[idx] = (C, Y, X) ✓
            # Source: (Z, Y, X) single-channel → img[idx] = (Y, X) ✓
            img_slice = img[idx]
            lbl_slice = lbl[idx]
        elif view == "xz":
            if multichannel:
                # 2026-05-21 channel-axis bug fix: pre-fix this saved
                # (Z, C, X) — channels in the MIDDLE, not channels-
                # first.  tifffile interpreted Z as pages and (C, X)
                # as the 2D image → user saw "512×2 axial images".
                # Cellpose training silently ate the wrong shape via
                # smallest-dim auto-detect for our 2-channel data, but
                # the saved TIFF was visually wrong AND broke any tool
                # that doesn't auto-detect.  Inference at
                # `packages/epicycle_reconstruction/logic.py::segment_views`
                # already does the equivalent transpose.  Now extraction
                # matches: (C, Z, X) channels-first to mirror xy's
                # (C, Y, X).
                img_slice = np.transpose(
                    img[:, :, idx, :], (1, 0, 2),
                )  # (Z, C, X) → (C, Z, X)
            else:
                img_slice = img[:, idx, :]
            lbl_slice = lbl[:, idx, :]
        else:  # yz
            if multichannel:
                # Same fix as xz.  Source (Z, C, Y) → (C, Z, Y).
                img_slice = np.transpose(
                    img[:, :, :, idx], (1, 0, 2),
                )
            else:
                img_slice = img[:, :, idx]
            lbl_slice = lbl[:, :, idx]

        stem = f"{view}_{idx:04d}"
        # Copy slices to ensure contiguous memory for workers
        slice_data.append((
            np.ascontiguousarray(img_slice),
            np.ascontiguousarray(lbl_slice),
            stem, dest, cleaning_kwargs,
        ))

    # Process slices in parallel (mask cleaning + TIFF writing)
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    n_workers = max(1, min(os.cpu_count() or 1, 8) - 1)
    completed_count = 0
    count_lock = threading.Lock()

    def _process_slice(img_s, lbl_s, stem, dest_dir, cleaning_kwargs):
        """Apply per-view mask cleaning, write paired TIFFs."""
        do_clean = (
            cleaning_kwargs["remove_boundaries"]
            or cleaning_kwargs["relabel"]
            or cleaning_kwargs["min_size"] > 0
            or cleaning_kwargs["smoothing_disk_size"] > 0
            or cleaning_kwargs.get("expand_labels_distance", 0) > 0
        )
        if do_clean:
            lbl_s = clean_mask_slice(lbl_s, **cleaning_kwargs)
        tifffile.imwrite(str(dest_dir / f"{stem}_img.tif"), img_s)
        tifffile.imwrite(
            str(dest_dir / f"{stem}_mask.tif"),
            lbl_s.astype(np.int32),
        )
        return "xy" in stem  # True if spatial

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _process_slice, img_s, lbl_s, stem, dest, clean_kw,
            ): stem
            for img_s, lbl_s, stem, dest, clean_kw in slice_data
        }

        for future in as_completed(futures):
            is_spatial = future.result()
            if is_spatial:
                spatial_count += 1
            else:
                axial_count += 1
            completed_count += 1
            if progress_cb is not None:
                progress_cb(completed_count, total_slices)

    return {
        "spatial": spatial_count,
        "axial": axial_count,
        "total": spatial_count + axial_count,
        "output_dir": str(output_dir),
    }


def extract_from_workspace(
    workspace_dir: Union[str, Path],
    patch_names: Optional[List[str]] = None,
    *,
    image_dir: Optional[Union[str, Path]] = None,
    min_labels_per_slice: int = 3,
    views: Sequence[str] = ("xy", "xz", "yz"),
    edge_buffer: int = 0,
    slices_per_patch: int = 0,
    min_distance: int = 1,
    remove_boundaries: bool = False,
    relabel: bool = False,
    min_size: int = 0,
    smoothing_disk_size: int = 0,
    expand_labels_distance: int = 0,
    # 2026-05-21 axes-aware normalization (see
    # extract_training_pairs_from_3d for the contract).
    axes: str = "ZCYX",
    # 2026-05-19: per-orientation overrides.  See
    # extract_training_pairs_from_3d for the contract.
    lateral_params: Optional[Dict[str, object]] = None,
    axial_params: Optional[Dict[str, object]] = None,
    progress_cb: Optional[object] = None,
    # Multi-run path-id (Option A): None → flat layout (byte-identical to
    # today / the default run); a slug → ``runs/{run_id}/...`` for a
    # named run.  Threaded into the reconstruction-scan + refinement
    # output ``_paths`` calls below.
    run_id: Optional[str] = None,
) -> Dict[str, int]:
    """Extract training pairs from all reconstructed patches in a workspace.

    Scans ``{workspace}/artifacts/reconstruction/*/manifest.json`` for label
    volumes, loads corresponding intensity images from *image_dir*, and calls
    :func:`extract_training_pairs_from_3d` for each.

    Args:
        workspace_dir: Workspace root containing artifacts/.
        patch_names: If given, only process these patch stems. Otherwise
            processes all patches with reconstruction manifests.
        image_dir: Directory containing source z-stack TIFF images. If None,
            the ``source_stack_path`` from each manifest is used.
        min_labels_per_slice: Passed to extract_training_pairs_from_3d.
        views: Which orthogonal views to extract.
        edge_buffer: Skip slices within this distance of volume edges.
        slices_per_patch: Max slices per view (0 = all).
        min_distance: Min spacing between sampled slices.
        remove_boundaries: Erode 1px boundaries between touching labels.
        relabel: Relabel connected components with unique IDs.
        min_size: Remove regions smaller than this (pixels).
        smoothing_disk_size: Morphological smoothing disk radius.

    Returns:
        Summary dict with ``patches_processed``, ``spatial``, ``axial``.
    """
    import tifffile

    import shutil

    workspace_dir = Path(workspace_dir)
    recon_dir = _paths.reconstructions_dir(workspace_dir, run_id)
    output_root = _paths.refinement_pairs_root(workspace_dir, run_id)

    total_spatial = 0
    total_axial = 0
    patches_processed = 0
    per_patch_results: Dict[str, Dict[str, object]] = {}

    if not recon_dir.is_dir():
        return {"patches_processed": 0, "spatial": 0, "axial": 0}

    # 2026-05-19: replaced unconditional `shutil.rmtree(output_root)`
    # with per-patch clearing.  The old behavior silently deleted pairs
    # from patches NOT included in this run, which is a footgun for the
    # iterative-refinement workflow (extract A+B+C, re-extract A with
    # tweaked knobs → lose B + C pairs).  Now: only patches in this
    # job-set have their dirs cleared before re-extraction; other
    # patches' pairs are preserved on disk and in the merged manifest.
    output_root.mkdir(parents=True, exist_ok=True)

    # Load existing manifest if present.  Its per_patch dict is the
    # source of truth for "what's been extracted so far"; we merge new
    # entries in below.  Snapshots of extraction_settings + mask_cleaning
    # are persisted PER-PATCH so the controller's mixed-settings
    # detection has authoritative history per patch.
    #
    # Variable name `extraction_manifest_path` (not `manifest_path`) to
    # avoid shadowing the local `manifest_path` used inside the
    # patch-discovery loop below to read each reconstruction manifest.
    # 2026-05-20 audit fix (F1): use the canonical path helper.
    # Pre-fix the writer + reader + tests pointed at three different
    # locations.  Helper resolves to refinement/manifest.json (sibling
    # of refinement/pairs/, NOT inside it).  This aligns the runtime
    # with the migration script + the target layout.
    extraction_manifest_path = _paths.refinement_manifest_path(workspace_dir, run_id)
    existing_per_patch: Dict[str, Dict[str, object]] = {}
    if extraction_manifest_path.exists():
        try:
            existing_manifest = json.loads(
                extraction_manifest_path.read_text(encoding="utf-8")
            )
            existing_per_patch = existing_manifest.get("per_patch", {}) or {}
        except (json.JSONDecodeError, OSError):
            existing_per_patch = {}

    # Prune manifest entries whose patch directory no longer exists on
    # disk (stale references — e.g., user manually deleted a patch dir,
    # or the Clear-all action ran).  Keep the manifest honest.
    existing_per_patch = {
        stem: entry
        for stem, entry in existing_per_patch.items()
        if (output_root / stem).is_dir()
    }

    # Discover which patches we'll process
    patch_jobs = []
    for run_dir in sorted(recon_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        labels_path = manifest.get("labels_path", "")
        source_path = manifest.get("source_stack_path", "")

        if not labels_path or not Path(labels_path).exists():
            continue

        patch_stem = Path(source_path).stem if source_path else run_dir.name
        if patch_names is not None and patch_stem not in patch_names:
            continue

        img_path = source_path
        if image_dir is not None:
            img_dir = Path(image_dir)
            candidates = list(img_dir.glob(f"{patch_stem}.*"))
            if candidates:
                img_path = str(candidates[0])

        if not img_path or not Path(img_path).exists():
            continue

        patch_jobs.append((patch_stem, labels_path, img_path))

    # 2026-05-19 sub-commit 9 (progress-bar accuracy fix): resolve
    # per-view effective params BEFORE the pre-scan so the count
    # uses the SAME filters that extract_training_pairs_from_3d will
    # apply.  Previously the pre-scan used scalar `edge_buffer`,
    # `min_labels_per_slice`, and `slices_per_patch` for ALL views,
    # while actual extraction used `_eff_lateral` for xy and
    # `_eff_axial` for xz/yz — guaranteeing a mismatch any time the
    # per-orientation dialog had differentiated those values.  Net
    # effect: progress bar was inaccurate (typically finishing well
    # before or after 100%).  Now matches by construction.
    _scalar_defaults_ws = {
        "min_labels_per_slice": min_labels_per_slice,
        "edge_buffer": edge_buffer,
        "slices_per_patch": slices_per_patch,
        "min_distance": min_distance,
        "remove_boundaries": remove_boundaries,
        "relabel": relabel,
        "min_size": min_size,
        "smoothing_disk_size": smoothing_disk_size,
        "expand_labels_distance": expand_labels_distance,
    }
    _eff_lateral = {**_scalar_defaults_ws, **(lateral_params or {})}
    _eff_axial = {**_scalar_defaults_ws, **(axial_params or {})}

    def _params_for_view(view: str) -> dict:
        return _eff_lateral if view == "xy" else _eff_axial

    # Pre-scan: count total slices across all patches for accurate
    # progress.  Loads only the label volumes (not images) and uses
    # the per-view effective filters to mirror the real extraction.
    global_total = 0
    for _, lbl_path, _ in patch_jobs:
        lbl_vol = tifffile.imread(lbl_path)
        for view in views:
            vp = _params_for_view(view)
            v_edge_buf = int(vp["edge_buffer"])
            v_min_lab = int(vp["min_labels_per_slice"])
            v_cap = int(vp["slices_per_patch"])
            if view == "xy":
                n = lbl_vol.shape[0]
            elif view == "xz":
                n = lbl_vol.shape[1]
            else:
                n = lbl_vol.shape[2]
            lo, hi = v_edge_buf, n - v_edge_buf
            if hi <= lo:
                print(
                    f"[PERVIEW-PRESCAN] view={view} edge_buffer={v_edge_buf} "
                    f"leaves no candidates (n={n}) -> 0"
                )
                continue
            valid_count = 0
            for idx in range(lo, hi):
                if view == "xy":
                    sl = lbl_vol[idx]
                elif view == "xz":
                    sl = lbl_vol[:, idx, :]
                else:
                    sl = lbl_vol[:, :, idx]
                if len(np.unique(sl[sl > 0])) >= v_min_lab:
                    valid_count += 1
            # Apply per-view sampling cap (matches extraction loop).
            if v_cap > 0:
                valid_count = min(valid_count, v_cap)
            print(
                f"[PERVIEW-PRESCAN] view={view} candidates={hi - lo} "
                f"valid={valid_count} (edge_buffer={v_edge_buf}, "
                f"min_labels={v_min_lab}, cap={v_cap})"
            )
            global_total += valid_count
        del lbl_vol  # free memory before loading images

    print(f"[PERVIEW-PRESCAN] global_total predicted = {global_total}")

    # Global progress counter
    _global_done = [0]

    def _global_progress_cb(current: int, total: int) -> None:
        _global_done[0] += 1
        if progress_cb is not None:
            progress_cb(_global_done[0], global_total)

    # Per-patch settings snapshots — schema 1.2 (2026-05-19): separate
    # lateral and axial snapshots so the mixed-settings detector can
    # compare per-view.  Legacy `extraction_settings` / `mask_cleaning`
    # are also written for backward-compat reads.  _eff_lateral and
    # _eff_axial are now computed earlier (above the pre-scan, sub-
    # commit 9) so we just reuse them here.

    def _split_view_dict(eff):
        """Project effective view-params into the legacy
        (extraction_settings, mask_cleaning) pair stored in the
        manifest, so the per-view snapshots remain comparable to
        the global schema-1.1 fields."""
        return (
            {
                "views": list(views),
                "min_labels_per_slice": int(eff["min_labels_per_slice"]),
                "edge_buffer": int(eff["edge_buffer"]),
                "slices_per_patch": int(eff["slices_per_patch"]),
                "min_distance": int(eff["min_distance"]),
            },
            {
                "remove_boundaries": bool(eff["remove_boundaries"]),
                "relabel": bool(eff["relabel"]),
                "min_size": int(eff["min_size"]),
                "smoothing_disk_size": int(eff["smoothing_disk_size"]),
                # 2026-05-21: audit-flagged provenance gap — without
                # this, `_check_extraction_mixed_settings` would treat
                # patches with expand=0 vs expand=1 as identical.
                "expand_labels_distance": int(
                    eff.get("expand_labels_distance", 0)
                ),
            },
        )

    lateral_extraction_settings, lateral_mask_cleaning = _split_view_dict(_eff_lateral)
    axial_extraction_settings, axial_mask_cleaning = _split_view_dict(_eff_axial)
    # 2026-05-21 audit fix: record the axes value at the extraction-
    # settings level so `_check_extraction_mixed_settings` can detect
    # axes drift between extractions.  Stored as upper-case for
    # consistent comparison.  Workspace-global setting, so identical
    # for both lateral + axial.
    _axes_recorded = "ZCYX" if axes is None else str(axes).upper()
    lateral_extraction_settings["axes"] = _axes_recorded
    axial_extraction_settings["axes"] = _axes_recorded
    print(
        f"[PERVIEW-WS] resolved per-view settings: "
        f"lateral_min_size={lateral_mask_cleaning['min_size']} "
        f"axial_min_size={axial_mask_cleaning['min_size']} "
        f"axes={_axes_recorded}"
    )

    # Legacy combined dicts (schema 1.1 compat).  When per-view dicts
    # are absent (the typical pre-dialog state), these equal the
    # scalar-only resolution and behave exactly as before.
    current_extraction_settings = {
        "views": list(views),
        "min_labels_per_slice": min_labels_per_slice,
        "edge_buffer": edge_buffer,
        "slices_per_patch": slices_per_patch,
        "min_distance": min_distance,
        "axes": _axes_recorded,
    }
    current_mask_cleaning = {
        "remove_boundaries": remove_boundaries,
        "relabel": relabel,
        "min_size": min_size,
        "smoothing_disk_size": smoothing_disk_size,
        "expand_labels_distance": expand_labels_distance,
    }

    # Extract all patches with unified progress
    for patch_stem, labels_path, img_path in patch_jobs:
        labels_3d = tifffile.imread(labels_path)
        image_3d = tifffile.imread(img_path)

        # 2026-06-06 mixed-layout fix — handle ZCYX *and* CZYX per patch.
        # `axes` here is a SINGLE global value (the UI `axes_combo`, set
        # once).  In a set whose patches have different C/Z orderings, that
        # one value is wrong for whichever patches don't match it: with the
        # wrong assumption the 4D slicing below indexes the channel axis
        # (size ~2) with a Z index (up to ~hundreds) → IndexError, OR the
        # explicit ValueError in extract_training_pairs_from_3d for an
        # unsupported string — surfacing to the user as "failed to extract".
        #
        # Resolve the layout PER PATCH via the shared `resolve_patch_axes`
        # helper (metadata → deterministic labels-shape match → size
        # heuristic → global combo fallback).  The labels-shape match is
        # authoritative and, unlike the pure size heuristic, can't be fooled
        # by a thin-Z stack (Z <= C).  Only 4D images have a C/Z ambiguity;
        # 3D (ZYX) is unaffected (the extractor ignores `axes` for 3D).
        from .qc_features import resolve_patch_axes

        patch_axes = resolve_patch_axes(
            image_3d, labels=labels_3d, image_path=img_path, fallback=axes,
        )
        if getattr(image_3d, "ndim", 0) == 4:
            print(
                f"[EXTRACT-AXES] patch {patch_stem!r}: image shape "
                f"{tuple(image_3d.shape)} labels {tuple(labels_3d.shape)} -> "
                f"axes={patch_axes} (global combo was {axes})"
            )

        patch_out = output_root / patch_stem
        # Per-patch clean-slate: clear only THIS patch's dir before
        # re-extracting.  Other patches' dirs (not in patch_jobs) are
        # left intact — see merge logic below.
        if patch_out.exists():
            shutil.rmtree(patch_out)

        # Capture the source labels' mtime BEFORE running the
        # extraction so the staleness check has the timestamp of the
        # exact volume the pairs were derived from.
        try:
            source_mtime = Path(labels_path).stat().st_mtime
        except OSError:
            source_mtime = 0.0

        result = extract_training_pairs_from_3d(
            image_3d,
            labels_3d,
            patch_out,
            min_labels_per_slice=min_labels_per_slice,
            views=views,
            edge_buffer=edge_buffer,
            slices_per_patch=slices_per_patch,
            min_distance=min_distance,
            remove_boundaries=remove_boundaries,
            relabel=relabel,
            min_size=min_size,
            smoothing_disk_size=smoothing_disk_size,
            expand_labels_distance=expand_labels_distance,
            axes=patch_axes,
            lateral_params=lateral_params,
            axial_params=axial_params,
            progress_cb=_global_progress_cb,
        )
        per_patch_results[patch_stem] = {
            "spatial": result["spatial"],
            "axial": result["axial"],
            # Per-patch provenance (2026-05-19): the source labels' mtime
            # at extraction time + the settings snapshots used for this
            # patch.  Read by the controller for staleness detection
            # (`_check_extraction_staleness`) and mixed-settings detection
            # (`_check_extraction_mixed_settings`).
            "source_mtime": source_mtime,
            "source_path": str(labels_path),
            # Legacy schema-1.1 fields (kept for backward-compat reads).
            "extraction_settings": dict(current_extraction_settings),
            "mask_cleaning": dict(current_mask_cleaning),
            # New schema-1.2 fields — per-view snapshots.  Mixed-settings
            # detection prefers these when present.
            "lateral_extraction_settings": dict(lateral_extraction_settings),
            "lateral_mask_cleaning": dict(lateral_mask_cleaning),
            "axial_extraction_settings": dict(axial_extraction_settings),
            "axial_mask_cleaning": dict(axial_mask_cleaning),
        }
        total_spatial += result["spatial"]
        total_axial += result["axial"]
        patches_processed += 1
        print(
            f"[PERVIEW-WS] patch {patch_stem!r}: "
            f"+{result['spatial']} spatial, +{result['axial']} axial"
        )

    # Merge: new per-patch entries override existing entries for the
    # SAME patch stems; entries for patches NOT in this run are preserved.
    merged_per_patch = dict(existing_per_patch)
    merged_per_patch.update(per_patch_results)

    # Recompute totals across the merged set so the badge / summary
    # reflect EVERYTHING on disk, not just this run.
    merged_totals_spatial = sum(
        int(entry.get("spatial", 0)) for entry in merged_per_patch.values()
    )
    merged_totals_axial = sum(
        int(entry.get("axial", 0)) for entry in merged_per_patch.values()
    )

    # Write merged extraction manifest for provenance tracking.  The
    # top-level extraction_settings/mask_cleaning reflect THIS run; the
    # per-patch snapshots are the source of truth per patch (used for
    # mixed-settings detection on subsequent runs).
    from datetime import datetime, timezone
    manifest = {
        # Schema 1.2 (2026-05-19): adds per-view settings snapshots
        # in each per_patch entry.  Backward-compat: schema 1.1 entries
        # still load fine — readers fall back to the combined fields
        # when per-view fields are missing.
        "schema_version": "1.2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # 2026-05-19: `patches` now reflects EVERY patch with pairs on
        # disk (merged set), not just patches re-extracted in this run.
        "patches": sorted(merged_per_patch.keys()),
        "per_patch": merged_per_patch,
        "totals": {
            "spatial": merged_totals_spatial,
            "axial": merged_totals_axial,
            "total": merged_totals_spatial + merged_totals_axial,
        },
        # Legacy combined fields (schema 1.1) kept for backward-compat.
        "extraction_settings": current_extraction_settings,
        "mask_cleaning": current_mask_cleaning,
        # Schema 1.2 per-view top-level snapshots (this run's
        # effective lateral/axial settings).
        "lateral_extraction_settings": lateral_extraction_settings,
        "lateral_mask_cleaning": lateral_mask_cleaning,
        "axial_extraction_settings": axial_extraction_settings,
        "axial_mask_cleaning": axial_mask_cleaning,
        # Populated by training step when fine-tuning runs
        "training": None,
    }
    # 2026-05-20 audit fix (F2): atomic manifest write.
    # Pre-fix used plain write_text — crash mid-write left a torn
    # JSON file that the reader's swallowed ValueError silently
    # treated as "nothing extracted", losing all per-patch
    # provenance.  Now: tempfile + os.replace, parity with
    # state_manager.save().
    import os as _os
    import tempfile as _tempfile
    extraction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = _tempfile.mkstemp(
        suffix=".json", prefix=".manifest_", dir=str(extraction_manifest_path.parent),
    )
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        _os.replace(tmp_path, str(extraction_manifest_path))
    except BaseException:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return {
        "patches_processed": patches_processed,
        # `spatial`/`axial` in the return summary report THIS run's
        # additions (the user-visible "Extracted: N pairs..." line),
        # not the merged totals — the merged numbers live in the
        # manifest for the badge to read.
        "spatial": total_spatial,
        "axial": total_axial,
        "manifest_path": str(extraction_manifest_path),
    }
