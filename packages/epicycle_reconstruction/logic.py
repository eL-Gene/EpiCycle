"""
Reconstruction pipeline service.

Provides both placeholder (zero-filled) and real (Cellpose + u-Segment3D)
reconstruction modes. The real pipeline:
1. Segments a 3D volume in XY, XZ, YZ views using Cellpose
2. Aggregates 2D predictions to 3D using u-Segment3D
3. Optionally postprocesses with flow-consistency filtering

Ported from epicycle-training/epicycle_local/reconstruction/local.py
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile

from packages.epicycle_domain import ReconstructionJobSpec
from packages.epicycle_training.config import SegmentationConfig

from .config import ReconstructionConfig


# ---------------------------------------------------------------------------
# Anisotropic pre-smoothing helpers
# ---------------------------------------------------------------------------

def smooth_probs_3d(
    probs: np.ndarray,
    sigma: tuple = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Gaussian-smooth a 3-D probability volume with anisotropic sigma.

    Use a high sigma along the axis perpendicular to the slicing plane
    to suppress slice-to-slice discontinuities from independent 2D
    CellPose predictions.

    Parameters
    ----------
    probs : (Z, Y, X) probability map
    sigma : per-axis Gaussian sigma (Z, Y, X)

    Returns
    -------
    Smoothed probability array of same shape.
    """
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(probs, sigma=sigma)


def smooth_flows_3d(
    flows: np.ndarray,
    probs: np.ndarray = None,
    sigma: tuple = (1.0, 1.0, 1.0),
    mask_threshold: float = None,
    soft_weighting: bool = False,
) -> np.ndarray:
    """Gaussian-smooth 3-D flow fields, optionally masked/weighted by probs.

    Parameters
    ----------
    flows : (C, Z, Y, X) gradient field
    probs : (Z, Y, X) probability map (optional)
    sigma : per-axis Gaussian sigma
    mask_threshold : zero out flows where sigmoid(probs) < threshold
    soft_weighting : multiply flows by sigmoid(probs)

    Returns
    -------
    Smoothed flow array of same shape.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.special import expit

    flows_smooth = np.zeros_like(flows)
    for c in range(flows.shape[0]):
        flows_smooth[c] = gaussian_filter(flows[c], sigma=sigma)

    if probs is not None:
        prob_sigmoid = expit(probs)
        if mask_threshold is not None:
            mask = (prob_sigmoid >= mask_threshold).astype(np.float32)
            flows_smooth *= mask[np.newaxis, ...]
        if soft_weighting:
            flows_smooth *= prob_sigmoid[np.newaxis, ...]

    return flows_smooth


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Container for reconstruction outputs.

    Attributes:
        mask_3d: 3D instance segmentation mask (Z, Y, X)
        prob_3d: 3D cell probability volume (optional)
        binary_3d: 3D binary mask (optional)
        flows_3d: 3D gradient field (optional)
        stats: Dictionary of reconstruction statistics
    """
    mask_3d: np.ndarray
    prob_3d: Optional[np.ndarray] = None
    binary_3d: Optional[np.ndarray] = None
    flows_3d: Optional[np.ndarray] = None
    stats: Dict = field(default_factory=dict)

    def save(self, output_dir: Union[str, Path], prefix: str = "result"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tifffile.imwrite(
            str(output_dir / f"{prefix}_mask.tif"),
            self.mask_3d.astype(np.uint32),
            bigtiff=True,
        )
        if self.prob_3d is not None:
            np.save(output_dir / f"{prefix}_probabilities.npy", self.prob_3d.astype(np.float16))
        if self.binary_3d is not None:
            np.save(output_dir / f"{prefix}_binary.npy", self.binary_3d.astype(np.uint8))
        if self.flows_3d is not None:
            np.save(output_dir / f"{prefix}_flows3D.npy", self.flows_3d.astype(np.float16))


# ---------------------------------------------------------------------------
# Dense 2D segmentation (orthogonal views)
# ---------------------------------------------------------------------------

def segment_views(
    image: np.ndarray,
    lateral_model_path: Union[str, Path],
    axial_model_path: Union[str, Path],
    seg_config: Optional[SegmentationConfig] = None,
    axes: str = "ZCYX",
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Segment a 3D volume in XY, XZ, and YZ views sequentially.

    Models are loaded and released one at a time to conserve GPU memory.

    Args:
        image: 3D volume — (Z, C, H, W), (C, Z, H, W), or (Z, H, W)
        lateral_model_path: Path to Cellpose model for XY view
        axial_model_path: Path to Cellpose model for XZ/YZ views
        seg_config: Segmentation configuration
        axes: Axis order — 'ZCYX' (default) or 'CZYX'
        progress_cb: Optional callback(view_name, current_slice, total_slices)

    Returns:
        Dictionary mapping view names to (flows, probs, masks) tuples.
        flows: (2, Z, Y, X), probs: (Z, Y, X),
        masks: (Z, Y, X) int32 or None if compute_masks=False
    """
    from cellpose import models

    if seg_config is None:
        seg_config = SegmentationConfig()

    # Normalize input to ZCYX
    if image.ndim == 3:
        image = image[:, np.newaxis, :, :]
    elif image.ndim == 4:
        axes_upper = axes.upper()
        if axes_upper == "CZYX":
            image = image.transpose(1, 0, 2, 3)
        elif axes_upper != "ZCYX":
            raise ValueError(f"Unsupported axes '{axes}', expected 'ZCYX' or 'CZYX'")
    else:
        raise ValueError(f"Expected 3D or 4D image, got shape {image.shape}")

    results = {}

    # --- XY view (lateral model) ---
    lateral_model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(lateral_model_path),
    )
    xy_flows, xy_probs, xy_masks = _segment_view(
        image, lateral_model, "xy", seg_config, progress_cb,
    )
    results["xy"] = (xy_flows, xy_probs, xy_masks)
    del lateral_model
    _cleanup_gpu()

    # --- XZ and YZ views (axial model) ---
    axial_model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(axial_model_path),
    )
    xz_flows, xz_probs, xz_masks = _segment_view(
        image, axial_model, "xz", seg_config, progress_cb,
    )
    results["xz"] = (xz_flows, xz_probs, xz_masks)

    yz_flows, yz_probs, yz_masks = _segment_view(
        image, axial_model, "yz", seg_config, progress_cb,
    )
    results["yz"] = (yz_flows, yz_probs, yz_masks)
    del axial_model
    _cleanup_gpu()

    return results


def _segment_view(
    image: np.ndarray,
    model,
    view: str,
    config: SegmentationConfig,
    progress_cb: Optional[Callable[[str, int, int], None]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Segment all slices of a single view.

    Args:
        image: Volume in ZCYX order
        model: Loaded CellposeModel
        view: One of 'xy', 'xz', 'yz'
        config: Segmentation config
        progress_cb: Optional progress callback(view, current, total)

    Returns:
        (flows, probs, masks) — flows: (2, Z, Y, X), probs: (Z, Y, X),
        masks: (Z, Y, X) int32 or None if compute_masks=False
    """
    # Transpose so iteration axis is dim-0
    if view == "xy":
        stack = image  # (Z, C, H, W) — iterate over Z
    elif view == "xz":
        stack = np.transpose(image, (2, 1, 0, 3))  # (Y, C, Z, X)
    else:  # yz
        stack = np.transpose(image, (3, 1, 0, 2))  # (X, C, Z, Y)

    do_masks = getattr(config, "compute_masks", True)
    flows_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = [] if do_masks else None
    total = stack.shape[0]

    for i in range(total):
        if progress_cb is not None:
            progress_cb(view, i, total)

        slice_img = stack[i]  # (C, H, W)
        slice_masks, flow_out, _ = model.eval(
            slice_img,
            channels=config.channels,
            do_3D=False,
            normalize=config.normalize,
            rescale=config.rescale,
            compute_masks=do_masks,
            diameter=config.diameter,
            cellprob_threshold=config.cellprob_threshold,
            flow_threshold=config.flow_threshold,
        )
        _, flow_comp, cellprob = flow_out
        flows_list.append(flow_comp)
        probs_list.append(cellprob)
        if do_masks and slice_masks.ndim >= 2:
            masks_list.append(slice_masks)

    # Report final slice
    if progress_cb is not None:
        progress_cb(view, total, total)

    # Stack and transpose back to canonical (2, Z, Y, X) / (Z, Y, X)
    if view == "xy":
        flows = np.stack(flows_list, axis=0).transpose(1, 0, 2, 3)
        probs = np.stack(probs_list, axis=0)
        masks = np.stack(masks_list, axis=0) if do_masks else None
    elif view == "xz":
        flows = np.stack(flows_list, axis=0)
        flows = flows.transpose(1, 0, 2, 3).transpose(0, 2, 1, 3)
        probs = np.stack(probs_list, axis=0).transpose(1, 0, 2)
        masks = np.stack(masks_list, axis=0).transpose(1, 0, 2) if do_masks else None
    else:  # yz
        flows = np.stack(flows_list, axis=0).transpose(1, 2, 3, 0)
        probs = np.stack(probs_list, axis=0).transpose(1, 2, 0)
        masks = np.stack(masks_list, axis=0).transpose(1, 2, 0) if do_masks else None

    return flows.astype(np.float32), probs.astype(np.float32), masks


def _cleanup_gpu():
    """Release GPU memory between model loads."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# The reconstruction manifest
# ---------------------------------------------------------------------------

#: ``manifest.json``'s version.  1 (no key) recorded the channels and twelve
#: ``ReconstructionConfig`` keys; 2 records every field of both configs.
#: Additive: no v1 key is renamed or removed, so a v1 reader reads v2.
RECON_MANIFEST_VERSION = 2


def _json_default(obj):
    """``json.dumps`` fallback: numpy scalars / arrays and paths."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _config_dict(config) -> Dict[str, Any]:
    import dataclasses
    if config is None:
        return {}
    data = dataclasses.asdict(config)
    # JSON round trip: numpy scalars become plain numbers, tuples lists
    return json.loads(json.dumps(data, default=_json_default))


def build_real_manifest(
    *,
    run_id: str,
    source_stack_path: str,
    labels_path: str,
    labels_shape,
    num_cells_raw: int,
    num_cells_final: int,
    axes: str,
    seg_config: Optional[SegmentationConfig],
    recon_config: ReconstructionConfig,
    extra_manifest: Optional[Dict[str, Any]] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """``manifest.json`` of a real reconstruction (version 2).

    ``seg_config`` is every ``SegmentationConfig`` field (``channels`` as a
    list, ``rescale`` the value Cellpose got, ``None`` included);
    ``recon_config`` every ``ReconstructionConfig`` field
    (``dataclasses.asdict``).  The keys of *extra_manifest* are added as
    given; a key that is already a core key is dropped with a log line,
    never written over it.  JSON-safe."""
    seg = _config_dict(seg_config)
    if seg_config is None:
        seg["channels"] = [0, 0]
    manifest: Dict[str, Any] = {
        "manifest_version": RECON_MANIFEST_VERSION,
        "run_id": None if run_id is None else str(run_id),
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "mode": "real",
        # str(): a caller may hand Paths; the output stays JSON-safe
        # (R2c audit LOW-3)
        "source_stack_path": (None if source_stack_path is None
                              else str(source_stack_path)),
        "labels_path": str(labels_path),
        "labels_shape": [int(x) for x in labels_shape],
        "num_cells_raw": int(num_cells_raw),
        "num_cells_final": int(num_cells_final),
        "axes": axes,
        "seg_config": seg,
        "recon_config": _config_dict(recon_config),
    }
    for key, value in (extra_manifest or {}).items():
        if key in manifest:
            print(f"[RECON-MANIFEST] extra key {key!r} dropped: it is a "
                  f"core key of manifest.json")
            continue
        manifest[key] = json.loads(json.dumps(value, default=_json_default))
    return manifest


# ---------------------------------------------------------------------------
# Pipeline service
# ---------------------------------------------------------------------------

class ReconstructionPipelineService:
    """
    Reconstruction service with placeholder and real modes.

    - run_placeholder(): deterministic zero-filled labels (original Phase D)
    - run_real(): actual Cellpose + u-Segment3D reconstruction
    """

    def run_real(
        self,
        *,
        working_dir: str,
        source_stack_path: str,
        lateral_model_path: str,
        axial_model_path: str,
        seg_config: Optional[SegmentationConfig] = None,
        recon_config: Optional[ReconstructionConfig] = None,
        axes: str = "ZCYX",
        cached_views: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        output_base: Optional[str] = None,
        patch_name: Optional[str] = None,
        reconstructions_root: Optional[str] = None,
        extra_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run real 3D reconstruction via Cellpose + u-Segment3D.

        Args:
            working_dir: Workspace directory for artifact output
            source_stack_path: Path to input 3D image TIFF
            lateral_model_path: Path to Cellpose model for XY view
            axial_model_path: Path to Cellpose model for XZ/YZ views
            seg_config: Segmentation parameters
            recon_config: Reconstruction parameters
            axes: Axis order of the input TIFF
            cached_views: Pre-computed view predictions to skip segmentation
            progress_cb: Progress callback(stage, current, total)
            output_base: Legacy run_id-keyed output base (unchanged behavior).
            patch_name: Patch name → flat ``runs/{patch_name}`` output dir.
            reconstructions_root: Pre-computed parent dir for the per-patch
                output (a plain string — keeps this package napari-free /
                layout-dumb).  The napari controller computes this via
                ``_paths.reconstructions_dir(ws, run_id)`` and passes it down.
                When set, ``out_dir = Path(reconstructions_root) / patch_name``
                and it TAKES PRECEDENCE over ``output_base``.  For the default
                run the controller passes the flat ``runs/`` dir, so the
                output is byte-identical to the ``patch_name`` branch.
            extra_manifest: Keys the caller adds to ``manifest.json``,
                written as given, never over a core key (a clash is dropped
                with a log line; :func:`build_real_manifest`).

        Returns:
            Dict with run_id, labels_path, manifest_path, mode="real", result
        """
        import segment3D.usegment3d as uSegment3D
        import segment3D.parameters as uSegment3D_params

        if seg_config is None:
            seg_config = SegmentationConfig()
        if recon_config is None:
            recon_config = ReconstructionConfig()

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if reconstructions_root is not None:
            # 2026-06-27 Step 3 multi-run (Phase 1b-ii): the napari
            # controller computed the active run's reconstructions dir
            # via ``_paths.reconstructions_dir(ws, run_id)`` and passed
            # it as a plain string (this package stays layout-dumb +
            # napari-free).  Default run → flat ``runs/`` → byte-identical
            # to the ``patch_name`` branch below.  Takes precedence over
            # ``output_base``.
            out_dir = Path(reconstructions_root) / patch_name
        elif output_base:
            out_dir = Path(output_base) / run_id
        elif patch_name:
            # 2026-05-20 Step 3 directory restructure: each patch owns
            # its own ``artifacts/step3/runs/{patch_name}/`` dir.
            # Re-reconstruction overwrites in place (design Q2).
            # When this module folds into napari_epicycle/core/, switch
            # to ``_paths.patch_run_dir(working_dir, patch_name)``.
            out_dir = (
                Path(working_dir) / "artifacts" / "step3" / "runs"
                / patch_name
            )
        else:
            # Backward-compat: caller didn't pass patch_name (older
            # callers, tests).  Fall back to run_id-keyed subdir;
            # cleanup helper won't find it but reconstruction is still
            # discoverable via run-store glob.  Controller callers
            # post-2026-05-20 always pass patch_name.
            out_dir = (
                Path(working_dir) / "artifacts" / "step3" / "runs" / run_id
            )
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Dense 2D segmentation ---
        if cached_views is not None:
            views = cached_views
        else:
            image = tifffile.imread(source_stack_path)
            views = segment_views(
                image,
                lateral_model_path,
                axial_model_path,
                seg_config,
                axes=axes,
                progress_cb=progress_cb,
            )

        # Unpack views — supports both (flows, probs) and (flows, probs, masks)
        xy_data, xz_data, yz_data = views["xy"], views["xz"], views["yz"]
        xy_flows, xy_probs = xy_data[0], xy_data[1]
        xz_flows, xz_probs = xz_data[0], xz_data[1]
        yz_flows, yz_probs = yz_data[0], yz_data[1]

        # --- Optional anisotropic pre-smoothing (per-view, before aggregation) ---
        if recon_config.pre_smooth_probs or recon_config.pre_smooth_flows:
            s_perp = recon_config.pre_smooth_sigma_perp
            s_in = recon_config.pre_smooth_sigma_inplane
            sigma_xy = (s_perp, s_in, s_in)   # XY view: perpendicular = Z
            sigma_xz = (s_in, s_perp, s_in)   # XZ view: perpendicular = Y
            sigma_yz = (s_in, s_in, s_perp)   # YZ view: perpendicular = X

            if recon_config.pre_smooth_probs:
                xy_probs = smooth_probs_3d(xy_probs, sigma=sigma_xy)
                xz_probs = smooth_probs_3d(xz_probs, sigma=sigma_xz)
                yz_probs = smooth_probs_3d(yz_probs, sigma=sigma_yz)

            if recon_config.pre_smooth_flows:
                mt = recon_config.flow_mask_threshold
                sw = recon_config.flow_soft_weighting
                xy_flows = smooth_flows_3d(
                    xy_flows, probs=xy_probs, sigma=sigma_xy,
                    mask_threshold=mt, soft_weighting=sw,
                )
                xz_flows = smooth_flows_3d(
                    xz_flows, probs=xz_probs, sigma=sigma_xz,
                    mask_threshold=mt, soft_weighting=sw,
                )
                yz_flows = smooth_flows_3d(
                    yz_flows, probs=yz_probs, sigma=sigma_yz,
                    mask_threshold=mt, soft_weighting=sw,
                )

        # --- Build u-Segment3D params ---
        reconstruct_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
        reconstruct_params["combine_cell_gradients"].update(recon_config.combine_cell_gradients)
        reconstruct_params["combine_cell_probs"].update(recon_config.combine_cell_probs)
        reconstruct_params["postprocess_binary"].update(recon_config.postprocess_binary)
        reconstruct_params["gradient_descent"].update(recon_config.gradient_descent)
        reconstruct_params["connected_component"].update(recon_config.connected_component)
        reconstruct_params["indirect_method"].update(recon_config.indirect_method)

        # --- Aggregate 2D → 3D ---
        if progress_cb is not None:
            progress_cb("reconstruct", 0, 1)

        seg3D, (prob3D, grad3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(
            probs=[xy_probs, xz_probs, yz_probs],
            gradients=[xy_flows, xz_flows, yz_flows],
            params=reconstruct_params,
        )

        num_cells_raw = len(np.unique(seg3D)) - 1

        # --- Optional postprocessing ---
        if recon_config.skip_postprocess:
            seg3D_final = seg3D
        else:
            if progress_cb is not None:
                progress_cb("postprocess", 0, 1)

            postprocess_params = uSegment3D_params.get_postprocess_segmentation_params()
            postprocess_params["flow_consistency"].update(recon_config.flow_consistency)
            postprocess_params["flow_consistency"]["n_cpu"] = recon_config.n_cpu
            postprocess_params["size_filters"].update(recon_config.size_filters)

            seg3D_final, _ = uSegment3D.postprocess_3D_cell_segmentation(
                seg3D,
                aggregation_params=reconstruct_params,
                postprocess_params=postprocess_params,
                cell_gradients=grad3D,
            )

        num_cells_final = len(np.unique(seg3D_final)) - 1

        if progress_cb is not None:
            progress_cb("reconstruct", 1, 1)

        # --- Build result ---
        result = ReconstructionResult(
            mask_3d=seg3D_final,
            prob_3d=prob3D if recon_config.save_prob3d else None,
            flows_3d=grad3D if recon_config.save_flow3d else None,
            stats={
                "num_cells_raw": num_cells_raw,
                "num_cells_final": num_cells_final,
                "shape": list(seg3D_final.shape),
            },
        )

        # Extract binary if requested
        if recon_config.save_binary:
            binary = seg3D_final > 0
            if recon_config.binary_fill_holes:
                from scipy.ndimage import binary_fill_holes
                binary = binary_fill_holes(binary, structure=np.ones((3, 3, 3), dtype=bool))
            result.binary_3d = binary.astype(np.uint8)

        # --- Persist artifacts ---
        labels_path = out_dir / "labels_3d.tif"
        tifffile.imwrite(str(labels_path), seg3D_final.astype(np.uint32), bigtiff=True)

        result.save(out_dir, prefix=Path(source_stack_path).stem)

        manifest = build_real_manifest(
            run_id=run_id,
            source_stack_path=source_stack_path,
            labels_path=str(labels_path),
            labels_shape=seg3D_final.shape,
            num_cells_raw=num_cells_raw,
            num_cells_final=num_cells_final,
            axes=axes,
            seg_config=seg_config,
            recon_config=recon_config,
            extra_manifest=extra_manifest,
        )
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=_json_default),
            encoding="utf-8")

        _cleanup_gpu()

        return {
            "run_id": run_id,
            "labels_path": str(labels_path),
            "manifest_path": str(manifest_path),
            "mode": "real",
            "result": result,
        }

    # ------------------------------------------------------------------
    # Placeholder mode (original Phase D)
    # ------------------------------------------------------------------

    def run_placeholder(
        self,
        *,
        working_dir: str,
        spec: ReconstructionJobSpec,
        source_stack_path: Optional[str] = None,
        patch_name: Optional[str] = None,
        reconstructions_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # 2026-05-20 Step 3 directory restructure: prefer per-patch dir
        # (patch_name) over run_id-keyed subdir.  Backward-compat
        # fallback to run_id when patch_name not supplied.
        # 2026-06-27 Phase 1b-ii: ``reconstructions_root`` (a plain string
        # computed napari-side via ``_paths.reconstructions_dir(ws, run_id)``)
        # takes precedence — out_dir = root / patch_name.  Default run →
        # flat ``runs/`` → byte-identical to the ``patch_name`` branch.
        if reconstructions_root is not None:
            out_dir = Path(reconstructions_root) / patch_name
        elif patch_name:
            out_dir = (
                Path(working_dir) / "artifacts" / "step3" / "runs"
                / patch_name
            )
        else:
            out_dir = (
                Path(working_dir) / "artifacts" / "step3" / "runs" / run_id
            )
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = self._build_placeholder_labels(source_stack_path)
        labels_path = out_dir / "labels_3d.tif"
        tifffile.imwrite(str(labels_path), labels.astype(np.int32))

        manifest = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "placeholder",
            "source_stack_path": source_stack_path or "",
            "labels_path": str(labels_path),
            "labels_shape": [int(x) for x in labels.shape],
            "spec": spec.to_dict(),
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "run_id": run_id,
            "labels_path": str(labels_path),
            "manifest_path": str(manifest_path),
            "mode": "placeholder",
        }

    def _infer_3d_shape(self, arr: np.ndarray) -> Tuple[int, int, int]:
        if arr.ndim == 2:
            return (1, int(arr.shape[0]), int(arr.shape[1]))
        if arr.ndim == 3:
            return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        if arr.ndim == 4:
            if arr.shape[0] <= 4 and arr.shape[-1] > 4:
                return (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
            if arr.shape[-1] <= 4 and arr.shape[0] > 4:
                return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
            return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        return (16, 64, 64)

    def _build_placeholder_labels(self, source_stack_path: Optional[str]) -> np.ndarray:
        if not source_stack_path:
            return np.zeros((16, 64, 64), dtype=np.int32)
        path = Path(source_stack_path)
        if not path.exists():
            return np.zeros((16, 64, 64), dtype=np.int32)
        data = tifffile.imread(str(path))
        arr = np.asarray(data)
        zyx = self._infer_3d_shape(arr)
        return np.zeros(zyx, dtype=np.int32)
