"""SDF-based per-instance label smoothing for the post-stitch post-process stage.

MIRRORED from the monorepo's canonical Step-3 curation smoother
(``packages/epicycle_annotator/smooth.py::smooth_label_sdf``) so the engine's
smoothing is the SAME operation the EpiCycle GUI applies in 3D curation. Pure algorithm: ``scipy`` + ``skimage``
imported lazily; no Qt, no napari, no GPU.

Algorithm (``smooth_label_sdf``):
  1. ``mask = (volume_roi == label_id)``           — binary mask of the target
  2. ``sdf  = EDT(mask) - EDT(~mask)``              — signed distance, +inside
  3. ``sdf  = gaussian_filter(sdf, (sz, sxy, sxy))``
  4. ``mask = sdf > 0``                            — re-threshold
  5. topology: if smoothing SPLIT the label, dilate its skeleton to reconnect
  6. keep only the largest connected component (drop threshold-noise satellites)

"Isotropic" smoothing is simply ``sigma_z == sigma_xy`` — :func:`resolve_sigmas`
defaults to that; pass ``sigma_z``/``sigma_xy`` to override either axis (the
anisotropic ``σ_z > σ_xy`` lever for Z-staircase).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def resolve_sigmas(
    sigma: float,
    sigma_z: Optional[float] = None,
    sigma_xy: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve ``(sigma_z, sigma_xy)`` from an isotropic ``sigma`` + optional overrides.

    Isotropic by default: ``sigma_z == sigma_xy == sigma``. Either axis can be
    overridden independently (anisotropic).
    """
    sz = float(sigma if sigma_z is None else sigma_z)
    sxy = float(sigma if sigma_xy is None else sigma_xy)
    return sz, sxy


def smooth_label_sdf(
    volume_roi: np.ndarray,
    label_id: int,
    *,
    sigma_z: float = 1.5,
    sigma_xy: float = 1.5,
    preserve_topology: bool = True,
) -> Tuple[np.ndarray, int, str]:
    """Smooth a single label using SDF Gaussian filtering.

    Parameters
    ----------
    volume_roi : 3D label volume (or ROI crop). Only ``label_id`` voxels are
        smoothed; all other labels are ignored.
    label_id : the label ID to smooth.
    sigma_z, sigma_xy : Gaussian sigma along Z and along X/Y. Equal == isotropic.
    preserve_topology : if True and smoothing would disconnect the label, dilate
        the skeleton to maintain connectivity.

    Returns
    -------
    result_roi : volume of same shape as ``volume_roi`` containing only
        ``label_id`` (smoothed) and 0 elsewhere. Does NOT contain other labels.
    n_modified : number of voxels that changed (added + removed).
    message : human-readable status.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, label as nd_label

    mask = volume_roi == label_id
    if not np.any(mask):
        return np.zeros_like(volume_roi), 0, f"Label {label_id} not found in ROI."

    # Signed distance field: positive inside, negative outside.
    sdf = distance_transform_edt(mask).astype(np.float64)
    sdf -= distance_transform_edt(~mask).astype(np.float64)

    # Gaussian smoothing with (an)isotropic sigma.
    sdf_smooth = gaussian_filter(sdf, sigma=(sigma_z, sigma_xy, sigma_xy))

    # Threshold at zero to get the smoothed mask.
    smooth_mask = sdf_smooth > 0

    # Topology preservation: check if smoothing increased component count.
    if preserve_topology:
        _, n_orig = nd_label(mask)
        _, n_smooth = nd_label(smooth_mask)
        if n_smooth > n_orig:
            # Smoothing disconnected the label — dilate skeleton to reconnect.
            from skimage.morphology import binary_dilation
            try:
                from skimage.morphology import skeletonize_3d
                skeleton = skeletonize_3d(mask.astype(np.uint8)) > 0
            except ImportError:
                # skeletonize_3d removed in scikit-image >= 0.21;
                # use skeletonize with method='lee' instead.
                from skimage.morphology import skeletonize
                skeleton = skeletonize(mask, method="lee") > 0
            smooth_mask = smooth_mask | binary_dilation(skeleton)

    # Post-smooth cleanup: keep only the largest connected component.
    # Smoothing can create tiny satellite fragments from threshold noise.
    labeled_arr, n_components = nd_label(smooth_mask)
    if n_components > 1:
        component_sizes = np.bincount(labeled_arr.ravel())
        component_sizes[0] = 0  # ignore background
        largest = component_sizes.argmax()
        smooth_mask = labeled_arr == largest

    result = np.zeros_like(volume_roi)
    result[smooth_mask] = label_id

    n_modified = int(np.count_nonzero(smooth_mask != mask))
    if n_modified == 0:
        return result, 0, f"Label {label_id} is already smooth."

    frag_msg = f" ({n_components - 1} fragments removed)" if n_components > 1 else ""
    return result, n_modified, f"Smoothed label {label_id}: {n_modified} voxels changed.{frag_msg}"


def smooth_instance_mask(
    crop: np.ndarray,
    label_id: int,
    *,
    sigma: float = 1.5,
    sigma_z: Optional[float] = None,
    sigma_xy: Optional[float] = None,
    preserve_topology: bool = True,
) -> np.ndarray:
    """Smooth one instance inside a bbox ``crop`` and return its boolean mask.

    Thin wrapper over :func:`smooth_label_sdf` for the post-process scatter-writer:
    resolves the isotropic/anisotropic sigma, runs the SDF smooth, and returns the
    ``(smoothed == label_id)`` boolean footprint within the crop (the caller scatters
    it into the output volume with a safe-write). Returns the original mask unchanged
    on a degenerate (empty) crop.
    """
    sz, sxy = resolve_sigmas(sigma, sigma_z, sigma_xy)
    result, _n, _msg = smooth_label_sdf(
        crop, label_id, sigma_z=sz, sigma_xy=sxy, preserve_topology=preserve_topology
    )
    return result == label_id


__all__ = ["resolve_sigmas", "smooth_label_sdf", "smooth_instance_mask"]
