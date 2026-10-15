"""
Instance erosion for EpiCycle ScaledSegmentation.

Erodes individual cell instances in a 3D segmentation mask.
Uses batched 3D convolutions on GPU when CUDA is available,
otherwise falls back to scipy.ndimage.binary_erosion on CPU.
"""

import logging
import numpy as np
from scipy.ndimage import find_objects
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_instance_bboxes(labels: np.ndarray, min_voxels: int = 300):
    """
    Filter small instances and compute bounding-box slices.

    Parameters
    ----------
    labels : 3-D integer label array
    min_voxels : discard instances with fewer voxels

    Returns
    -------
    labels_filtered : label array with small instances zeroed
    slices : list from ``scipy.ndimage.find_objects``
    """
    ids, counts = np.unique(labels, return_counts=True)
    keep = set(ids[(counts >= min_voxels) & (ids != 0)])

    labels_f = labels.copy()
    labels_f[~np.isin(labels_f, list(keep))] = 0

    slices = find_objects(labels_f)
    return labels_f, slices


def _process_batch(batch, meta, out, kernel, pad):
    """
    Process a batch of instance crops with potentially different sizes.

    Pads all tensors to the max size in the batch before concatenation
    (fixes the v3 torch.cat shape mismatch bug).
    """
    import torch
    import torch.nn.functional as F

    shapes = [t.shape for t in batch]
    max_d = max(s[2] for s in shapes)
    max_h = max(s[3] for s in shapes)
    max_w = max(s[4] for s in shapes)

    padded_batch = []
    orig_shapes = []

    for t in batch:
        _, _, d, h, w = t.shape
        pad_d, pad_h, pad_w = max_d - d, max_h - h, max_w - w
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            t_padded = F.pad(t, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        else:
            t_padded = t
        padded_batch.append(t_padded)
        orig_shapes.append((d, h, w))

    x = torch.cat(padded_batch, dim=0).cuda()
    conv = F.conv3d(x, kernel)
    er = (conv == kernel.numel()).squeeze(1).cpu().numpy()

    for i, (label_id, slc) in enumerate(meta):
        d_orig, h_orig, w_orig = orig_shapes[i]
        k_shrink = kernel.shape[2] - 1

        d_final = d_orig - k_shrink
        h_final = h_orig - k_shrink
        w_final = w_orig - k_shrink

        er_crop = er[i, :d_final, :h_final, :w_final]

        if not er_crop.any():
            # Fallback: keep original mask if erosion removes everything
            orig_mask = batch[i].squeeze().cpu().numpy()[:d_final, :h_final, :w_final]
            er_crop = orig_mask > 0

        out[slc][er_crop] = label_id

    del x, conv, er, padded_batch


def _erode_instances_cpu(
    labels: np.ndarray,
    kernel_size: int = 3,
    min_voxels: int = 1000,
) -> np.ndarray:
    """
    Erode each labelled instance using scipy on CPU.

    Slower than GPU but works everywhere. Uses binary_erosion
    with a cubic structuring element per instance.
    """
    from scipy.ndimage import binary_erosion, generate_binary_structure

    labels_f, slices = get_instance_bboxes(labels, min_voxels)
    out = np.zeros_like(labels_f, dtype=np.int32)

    # Full connectivity 3D structuring element, then dilate to kernel_size
    iterations = kernel_size // 2
    struct = generate_binary_structure(3, 1)

    n_instances = sum(1 for s in slices if s is not None)
    logger.info(f"CPU erosion: {n_instances} instances, kernel_size={kernel_size}")

    for label_id, slc in tqdm(enumerate(slices, start=1), total=len(slices), desc="Eroding (CPU)", unit="inst"):
        if slc is None:
            continue

        mask = (labels_f[slc] == label_id)

        if iterations > 0:
            eroded = binary_erosion(mask, structure=struct, iterations=iterations)
        else:
            eroded = mask

        if not eroded.any():
            eroded = mask  # keep original if erosion removes everything

        out[slc][eroded] = label_id

    logger.info("CPU erosion complete")
    return out


def erode_instances_gpu(
    labels: np.ndarray,
    kernel_size: int = 3,
    min_voxels: int = 1000,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Erode each labelled instance via batched GPU 3-D convolution.

    Parameters
    ----------
    labels : (Z, Y, X) integer label array
    kernel_size : erosion kernel size (odd integer)
    min_voxels : skip instances smaller than this
    batch_size : number of instances per GPU batch
        (256 is tuned for H200 141 GB VRAM)

    Returns
    -------
    Eroded label array (int32).
    """
    import torch
    import torch.nn.functional as F

    labels_f, slices = get_instance_bboxes(labels, min_voxels)
    out = np.zeros_like(labels_f, dtype=np.int32)
    pad = kernel_size // 2

    n_instances = sum(1 for s in slices if s is not None)
    logger.info(f"GPU erosion: {n_instances} instances, kernel_size={kernel_size}, batch_size={batch_size}")

    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size, kernel_size), device="cuda"
    )

    batch = []
    meta = []

    for label_id, slc in tqdm(enumerate(slices, start=1), total=len(slices), desc="Eroding (GPU)", unit="inst"):
        if slc is None:
            continue

        sub = (labels_f[slc] == label_id)
        sub_t = torch.from_numpy(sub).float().unsqueeze(0).unsqueeze(0)

        if pad > 0:
            sub_t = F.pad(sub_t, (pad, pad, pad, pad, pad, pad))

        batch.append(sub_t)
        meta.append((label_id, slc))

        if len(batch) == batch_size:
            _process_batch(batch, meta, out, kernel, pad)
            batch, meta = [], []

    if batch:
        _process_batch(batch, meta, out, kernel, pad)

    logger.info("GPU erosion complete")
    return out


def erode_instances(
    labels: np.ndarray,
    kernel_size: int = 3,
    min_voxels: int = 1000,
    batch_size: int = 256,
    force_cpu: bool = False,
) -> np.ndarray:
    """
    Erode each labelled instance, auto-selecting GPU or CPU.

    Uses GPU (batched 3D convolution) when CUDA is available.
    Falls back to CPU (scipy binary_erosion) otherwise, or when
    ``force_cpu=True``.

    Parameters
    ----------
    labels : (Z, Y, X) integer label array
    kernel_size : erosion kernel size (odd integer)
    min_voxels : skip instances smaller than this
    batch_size : GPU batch size (ignored on CPU)
    force_cpu : force CPU path even if CUDA is available

    Returns
    -------
    Eroded label array (int32).
    """
    use_gpu = False
    if not force_cpu:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass

    if use_gpu:
        logger.info("Erosion: using GPU path")
        try:
            return erode_instances_gpu(labels, kernel_size, min_voxels, batch_size)
        except Exception as e:
            logger.warning(f"GPU erosion failed ({e}), falling back to CPU")

    logger.info("Erosion: using CPU path (scipy)")
    return _erode_instances_cpu(labels, kernel_size, min_voxels)


def protect_border_margin(
    eroded_labels: np.ndarray,
    original_labels: np.ndarray,
    margin: int = 2,
) -> np.ndarray:
    """
    Restore original labels within ``margin`` voxels of the volume border.

    Prevents erosion artefacts at tile boundaries.
    """
    if margin is None or margin <= 0:
        return eroded_labels
    m = int(margin)
    out = eroded_labels.copy()
    out[:m, :, :] = original_labels[:m, :, :]
    out[-m:, :, :] = original_labels[-m:, :, :]
    out[:, :m, :] = original_labels[:, :m, :]
    out[:, -m:, :] = original_labels[:, -m:, :]
    out[:, :, :m] = original_labels[:, :, :m]
    out[:, :, -m:] = original_labels[:, :, -m:]
    return out
