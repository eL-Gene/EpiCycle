import io
import math
import os
import re

import numpy as np
import tifffile

# NOTE: matplotlib + qtpy are imported LAZILY inside the two functions that need
# them (draw_boxes_on_projection_matplotlib, extract_patches_from_shapes) so this
# module stays import-time pure — the pure naming/index helpers below are imported
# by the headless _export.py, which must not transitively pull in Qt/matplotlib.
from ._paths import step1_patches_dir
from ._state import load_state


def _materialize(a):
    """Realize an array-like into RAM: ``.compute()`` a dask array, else
    ``np.asarray`` (a no-op view for an ndarray, a disk read for a memmap
    slice).  Kept trivial + import-free so ``logic.py`` stays import-pure for
    the headless ``_export.py`` (see module docstring)."""
    return a.compute() if hasattr(a, "compute") else np.asarray(a)


def extract_patches_from_shapes(image_data, bounding_boxes, progress_bar=None,
                                name_prefix="patch", start_index=1,
                                channel_indices=None, z_ranges=None):
    """
    Extracts rectangular/square patches from a 3D (Z, Y, X) or 4D (C, Z, Y, X) image using 2D bounding boxes.
    Pads with zeros if a box is partially outside the image.

    ``image_data`` may be a LAZY array-like (a dask array or a numpy memmap) —
    each box is sliced then materialized so ONLY the crop is read off disk (the
    L0 lazy-crop path).  For an in-RAM numpy array this is a no-op, so
    the result is byte-identical to the eager path.

    Parameters
    ----------
    image_data : array-like
        The image stack, either (Z, Y, X) or (C, Z, Y, X).  May be a numpy
        array, a numpy memmap, or a dask array.
    bounding_boxes : list of np.ndarray
        Each box is a (4, 2) array defining corners in (Y, X) format.
    progress_bar : QProgressBar, optional
        Progress bar to reflect patch extraction progress.
    name_prefix : str
        Prefix for patch names (default: "patch").
    start_index : int
        Starting index for patch numbering (default: 1).
    channel_indices : list of int, optional
        When set (4D input only), select these channel indices — IN ORDER — as
        part of the per-box slice, so a lazy source materializes only the
        chosen channels of the crop rather than the whole volume.  ``None``
        (default) keeps every channel; behaviour is unchanged for the callers
        that pass a pre-selected array.
    z_ranges : list, optional
        Per-box ``[z0, z1]`` half-open Z crop, aligned to ``bounding_boxes``
        (the box tool's editable-Z).  An entry may be ``None`` (or
        a full-Z ``[0, Z]``) to keep the whole depth — that path is
        BYTE-IDENTICAL to the pre-feature full-Z extract, so the default
        ``z_ranges=None`` is unchanged.  Applied as part of the per-box slice so
        a lazy source reads only the requested Z slab off disk.  Out-of-range
        values are clamped to ``[0, Z]``.

    Returns
    -------
    dict
        Mapping of patch names to extracted (padded) numpy arrays.
    """
    is_multichannel = image_data.ndim == 4  # (C, Z, Y, X) vs (Z, Y, X)
    patches = {}

    if progress_bar:
        progress_bar.setRange(0, len(bounding_boxes))
        progress_bar.setValue(0)

    # Get spatial dimensions
    if is_multichannel:
        C, Z, H, W = image_data.shape
    else:
        Z, H, W = image_data.shape

    for idx, box in enumerate(bounding_boxes):
        # have issues with the following not following user entered values for bbox
        # y0, y1 = int(np.floor(np.min(box[:, 0]))), int(np.ceil(np.max(box[:, 0])))
        # x0, x1 = int(np.floor(np.min(box[:, 1]))), int(np.ceil(np.max(box[:, 1])))

        # updated
        y0, y1 = int(np.floor(np.min(box[:, 0]))), int(np.floor(np.max(box[:, 0])))
        x0, x1 = int(np.floor(np.min(box[:, 1]))), int(np.floor(np.max(box[:, 1])))


        # Compute desired patch size
        patch_h, patch_w = y1 - y0, x1 - x0

        # Clip box to image bounds
        y0_clip, y1_clip = max(0, y0), min(H, y1)
        x0_clip, x1_clip = max(0, x0), min(W, x1)

        # Per-box Z crop (box tool's editable-Z).  A full-Z range
        # [0, Z] — the default box, and z_ranges=None — makes `zsl` equivalent
        # to the pre-feature `:` slice, so the crop is BYTE-IDENTICAL then.
        zsl = slice(None)
        if (z_ranges is not None and idx < len(z_ranges)
                and z_ranges[idx] is not None):
            z0 = max(0, min(int(z_ranges[idx][0]), Z))
            z1 = max(z0, min(int(z_ranges[idx][1]), Z))
            zsl = slice(z0, z1)

        # Slice the image, then materialize ONLY the crop (a no-op for an
        # in-RAM numpy array; a `.compute()`/disk read for a dask/memmap source).
        if is_multichannel:
            if channel_indices is not None:
                patch = image_data[channel_indices, zsl, y0_clip:y1_clip, x0_clip:x1_clip]
            else:
                patch = image_data[:, zsl, y0_clip:y1_clip, x0_clip:x1_clip]
            patch = _materialize(patch)
            cropped_shape = patch.shape  # (C, Z, h, w)
            pad_h = patch_h - cropped_shape[2]
            pad_w = patch_w - cropped_shape[3]
            padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w)]
        else:
            patch = image_data[zsl, y0_clip:y1_clip, x0_clip:x1_clip]
            patch = _materialize(patch)
            cropped_shape = patch.shape  # (Z, h, w)
            pad_h = patch_h - cropped_shape[1]
            pad_w = patch_w - cropped_shape[2]
            padding = [(0, 0), (0, pad_h), (0, pad_w)]

        # Pad with zeros if needed
        patch = np.pad(patch, pad_width=padding, mode='constant', constant_values=0)
        print(f"extracted patch of shape {patch.shape}")

        patches[f"{name_prefix}_{start_index + idx}"] = patch

        if progress_bar:
            progress_bar.setValue(idx + 1)

    return patches


def filter_extractable_boxes(bounding_boxes, image_shape):
    """Split *bounding_boxes* into ``(valid, n_dropped)``.

    A box is dropped if it has zero area (a click-box, ``y0==y1`` or ``x0==x1``)
    or no overlap with the image extent — either would extract an all-zero patch
    yet still consume a monotonic index/name.  A box that merely
    spills past the edge (partial overlap) is KEPT: extraction pads it, which is
    intended.

    *image_shape* is ``(C, Z, Y, X)`` or ``(Z, Y, X)`` — only the last two
    (spatial Y, X) dims are used, matching ``extract_patches_from_shapes``.
    """
    height, width = int(image_shape[-2]), int(image_shape[-1])
    valid = []
    dropped = 0
    for box in bounding_boxes:
        arr = np.asarray(box)
        y0 = int(np.floor(np.min(arr[:, 0])))
        y1 = int(np.floor(np.max(arr[:, 0])))
        x0 = int(np.floor(np.min(arr[:, 1])))
        x1 = int(np.floor(np.max(arr[:, 1])))
        if y1 <= y0 or x1 <= x0:  # zero area (degenerate / click box)
            dropped += 1
            continue
        if min(height, y1) <= max(0, y0) or min(width, x1) <= max(0, x0):  # no overlap
            dropped += 1
            continue
        valid.append(box)
    return valid, dropped


def box_signature(box, z_range=None):
    """Winding-invariant floored axis-aligned signature of a (4,2) Y,X box —
    ``(y0, y1, x0, x1)`` XY-only when *z_range* is None, or the Z-inclusive
    ``(y0, y1, x0, x1, z0, z1)`` when a ``[z0, z1]`` slab is supplied.

    Uses ``int(floor(min/max))`` — the SAME reduction ``extract_patches_from_shapes``
    applies (logic.py `y0,y1 = floor(min),floor(max)`), so two boxes that would
    extract to the same patch share a signature.  Robust to napari reordering the
    4 corners on re-add and to sub-pixel drift; JSON-round-trip stable (a box
    stored to a sidecar list and restored via ``np.array`` matches).

    The box's Z is now editable, so the XY footprint alone no longer identifies a
    patch: a full-Z box and a same-XY narrowed-Z slab are DIFFERENT ROIs.  Folding
    ``(z0, z1)`` into the identity keeps the dedup from silently dropping the
    Z-slab box as "already captured".  ``z_range=None`` preserves the legacy
    XY-only signature for callers that have no per-box Z (the legacy Shapes
    fallback + old sidecars).

    Used to exclude already-exported / restored boxes from re-extraction.
    """
    a = np.asarray(box, dtype=float)
    xy = (int(np.floor(a[:, 0].min())), int(np.floor(a[:, 0].max())),
          int(np.floor(a[:, 1].min())), int(np.floor(a[:, 1].max())))
    if z_range is None:
        return xy
    return xy + (int(z_range[0]), int(z_range[1]))


def select_uncommitted_boxes(boxes, committed_sigs, z_ranges=None):
    """``[(original_index, box), ...]`` for boxes whose :func:`box_signature` is
    NOT in *committed_sigs* — i.e. the boxes not already exported/restored.  The
    original layer index is preserved so callers can map back after gaps.

    *z_ranges*, when given, is a per-box ``[z0, z1]`` list aligned to *boxes*
    (a None entry, or z_ranges=None, falls back to the XY-only signature for
    that box) so a same-XY box at a different Z slab is compared Z-inclusively
    and no longer collides with a full-Z box already committed."""
    out = []
    for i, b in enumerate(boxes):
        zr = z_ranges[i] if z_ranges is not None else None
        if box_signature(b, zr) not in committed_sigs:
            out.append((i, b))
    return out

def generate_max_intensity_project(image_data):
    """
    Compute Z-axis maximum intensity projection.

    Parameters
    ----------
    image_data : np.ndarray
        The full stack, either (Z, Y, X) or (C, Z, Y, X).

    Returns
    -------
    np.ndarray
        2D projection (Y, X) or (C, Y, X) if multichannel.
    """
    if image_data.ndim == 4:
        return np.max(image_data, axis=1)  # (C, Y, X)
    else:
        return np.max(image_data, axis=0)  # (Y, X)


def downsampled_max_intensity_project(source_czyx, channel_indices=None,
                                      max_xy=1024, max_z=64):
    """Z-MIP recap thumbnail from a **downsampled** view of a LAZY 4D
    ``(C, Z, Y, X)`` source — the whole volume is NEVER read.

    Only a strided overview is materialized (via
    :func:`packages.epicycle_largedata.lazy.downsample_for_display`, which
    strides — no averaging — then reads just the strided elements off disk), so
    peak RAM is bounded by the caps below rather than the source size.  This is
    the large-data replacement for ``generate_max_intensity_project(whole
    volume)`` in the Step-1 extract recap.

    Parameters
    ----------
    source_czyx : array-like
        The lazy full-res handle reordered to ``(C, Z, Y, X)`` (memmap / dask /
        ndarray).  4D only — the Step-1 lazy crop source (:meth:
        ``_crop_source_for``) always presents a channel axis (a 3D source is
        exposed as ``(1, Z, Y, X)``).
    channel_indices : list of int, optional
        Channels to project, IN ORDER (the ticked channels).  ``None`` keeps
        every channel.  Selection happens on the small in-RAM overview, so it
        never widens the disk read.
    max_xy, max_z : int
        Per-axis overview caps.  A source already under a cap gets a stride
        factor of ``1`` on that axis; when Y, X ≤ ``max_xy`` **and** Z ≤
        ``max_z`` all factors are ``1`` and the projection is byte-identical to
        ``generate_max_intensity_project`` of the (channel-selected) full stack
        — so the small-file recap is unchanged.

    Returns
    -------
    (projection, (fy, fx))
        *projection* is the ``(C, Y', X')`` Z-max projection of the
        channel-selected overview; ``(fy, fx)`` are the applied Y / X stride
        factors, so the caller can scale box coords into the downsampled space
        (divide each box's ``(Y, X)`` by ``(fy, fx)``) to keep the overlay
        aligned.
    """
    from packages.epicycle_largedata.lazy import downsample_for_display

    shape = tuple(int(s) for s in source_czyx.shape)
    if len(shape) != 4:
        raise ValueError(
            f"downsampled_max_intensity_project expects 4D (C, Z, Y, X), "
            f"got shape {shape}"
        )
    _C, Z, Y, X = shape
    fz = max(1, -(-Z // int(max_z)))    # ceil-div: 1 while Z <= max_z
    fy = max(1, -(-Y // int(max_xy)))
    fx = max(1, -(-X // int(max_xy)))
    overview, (fz, fy, fx) = downsample_for_display(source_czyx, (fz, fy, fx))
    if channel_indices is not None:
        overview = overview[list(channel_indices)]
    projection = generate_max_intensity_project(overview)  # (C, Y', X')
    return projection, (int(fy), int(fx))


def draw_boxes_on_projection_matplotlib(image_rgb, boxes, labels):
    """
    Overlay bounding boxes and labels using matplotlib and return as QPixmap.

    Parameters
    ----------
    image_rgb : np.ndarray
        Image (Y, X, 3) to draw on
    boxes : list of np.ndarray
        Bounding boxes, each (4, 2) in (Y, X)
    labels : list of str
        Names of patches for display

    Returns
    -------
    QPixmap
        Annotated image as a QPixmap for QLabel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from qtpy.QtGui import QImage, QPixmap

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100, facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(image_rgb.astype(np.uint8))

    # Teal accent matching Step 1 theme.
    _TEAL = "#5ba4cf"
    _TEAL_FILL = (0.357, 0.643, 0.812, 0.10)

    for box, label in zip(boxes, labels):
        y_min = int(np.min(box[:, 0]))
        x_min = int(np.min(box[:, 1]))
        y_max = int(np.max(box[:, 0]))
        x_max = int(np.max(box[:, 1]))

        width = x_max - x_min
        height = y_max - y_min

        rect = plt.Rectangle(
            (x_min, y_min), width, height,
            edgecolor=_TEAL, facecolor=_TEAL_FILL,
            linewidth=1.5, linestyle="-",
        )
        ax.add_patch(rect)
        ax.text(
            x_min + 4, y_min + 14, label,
            color="white", fontsize=7, fontweight="bold",
            bbox=dict(facecolor=_TEAL, alpha=0.7, edgecolor="none",
                      boxstyle="round,pad=0.2"),
        )

    ax.axis('off')
    fig.tight_layout(pad=0)

    # Convert to QPixmap
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)

    image = QImage.fromData(buf.read())
    pixmap = QPixmap.fromImage(image)

    plt.close(fig)
    return pixmap


def prepare_rgb_projection(projection):
    """
    Converts a 2D or multi-channel (C, Y, X) projection to displayable RGB format.

    Parameters
    ----------
    projection : np.ndarray
        Max intensity projection: (Y, X) or (C, Y, X)

    Returns
    -------
    np.ndarray
        (Y, X, 3) uint8 RGB image
    """
    def normalize_to_uint8(arr):
        arr = arr.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr *= 255
        return arr.astype(np.uint8)

    if projection.ndim == 2:
        rgb = np.stack([normalize_to_uint8(projection)] * 3, axis=-1)

    elif projection.ndim == 3:
        rgb = np.moveaxis(projection, 0, -1)  # (C, Y, X) → (Y, X, C)
        rgb = normalize_to_uint8(rgb)

        c = rgb.shape[-1]
        if c == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        elif c == 2:
            blank = np.zeros_like(rgb[..., 0:1])
            rgb = np.concatenate([rgb, blank], axis=-1)
        elif c >= 3:
            rgb = rgb[..., :3]  # trim to 3 channels for RGB

    else:
        raise ValueError("Unexpected shape for projection.")

    return rgb



def detect_sample_z_range_by_intensity(
    stack,
    hard_threshold=None,
    percentile_threshold=None,
    use_default_threshold=False,
    scale_intensities=True,
    intensity_calc='mean'
):
    """
    Determine the range of z-slices where the sample is likely located based on intensity.

    Parameters
    ----------
    stack : np.ndarray
        3D array (Z, Y, X) representing an image stack.
    hard_threshold : float, optional
        Use a hard intensity threshold to define start/end slices.
    percentile_threshold : tuple(float, float), optional
        Use cumulative intensity percentiles (e.g., (5, 95)) to define range.
    use_default_threshold : bool
        If True, use mean + std as a threshold.
    scale_intensities : bool
        If True, normalize intensity values to [0, 1].
    intensity_calc : str
        One of 'mean' or 'median'. Determines how slice intensity is calculated.

    Returns
    -------
    start : int
        First z-slice with significant signal.
    end : int
        Last z-slice with significant signal.
    """

    # computes the mean or median for each slice in the image stack
    if intensity_calc == 'mean':
        intensities = np.array([np.mean(img) for img in stack])
    elif intensity_calc == 'median':
        intensities = np.array([np.median(img) for img in stack])
    else:
        raise ValueError("`intensity_calc` must be either 'mean' or 'median'.")

    # min max normalization
    if scale_intensities:
        min_val, max_val = np.min(intensities), np.max(intensities)
        intensities = (intensities - min_val) / (max_val - min_val + 1e-8)

    cumulative_distribution = np.cumsum(intensities) / np.sum(intensities)

    # Enforce single threshold method
    methods_used = sum([
        hard_threshold is not None,
        percentile_threshold is not None,
        use_default_threshold
    ])
    if methods_used != 1:
        raise ValueError("Specify exactly one thresholding method: `hard_threshold`, `percentile_threshold`, or `use_default_threshold`.")

    if hard_threshold is not None:
        sample_range = np.where(intensities > hard_threshold)[0]
        if sample_range.size == 0:
            raise ValueError(f"No intensities exceed the hard_threshold ({hard_threshold}).")
        start, end = sample_range[0], sample_range[-1]

    elif percentile_threshold is not None:
        lower = percentile_threshold[0] / 100.0
        upper = percentile_threshold[1] / 100.0
        start = np.searchsorted(cumulative_distribution, lower)
        end = np.searchsorted(cumulative_distribution, upper)

    elif use_default_threshold:
        threshold = np.mean(intensities) + np.std(intensities)
        sample_range = np.where(intensities > threshold)[0]
        if sample_range.size == 0:
            raise ValueError("No intensities exceed the default threshold (mean + std).")
        start, end = sample_range[0], sample_range[-1]

    return start, end

def detect_sample_range_all_channels(img_data):
    """
    Detect the union Z-range for all channels in a (C, Z, Y, X) image stack.

    Parameters
    ----------
    img_data : np.ndarray
        The image patch with shape (C, Z, Y, X)

    Returns
    -------
    z_start : int
        The minimum Z-slice index with signal across any channel
    z_end : int
        The maximum Z-slice index with signal across any channel
    """
    z_ranges = []
    for c in range(img_data.shape[0]):
        try:
            z0, z1 = detect_sample_z_range_by_intensity(
                img_data[c],  # (Z, Y, X)
                hard_threshold=0.1,
                intensity_calc="median",
                scale_intensities=True
            )
            z_ranges.append((z0, z1))
        except ValueError:
            # skip channels that don't return a valid range
            continue

    if not z_ranges:
        raise ValueError("No valid Z-range could be determined from any channel.")

    # basically take the union basically
    z_start = min(z[0] for z in z_ranges)
    z_end   = max(z[1] for z in z_ranges)

    return z_start, z_end


# ---------------------------------------------------------------------------
# Data browser helpers
# ---------------------------------------------------------------------------

def scan_tiff_folder(folder_path):
    """Scan a folder for TIFF files and read shapes from headers only.

    Parameters
    ----------
    folder_path : str
        Path to directory containing TIFF files.

    Returns
    -------
    list of dict
        Each dict has keys: ``path``, ``filename``, ``stem``, ``shape``,
        ``dtype`` and ``nbytes``.  Sorted alphabetically by filename.

        ``nbytes`` is the **decompressed** size, computed from the header's
        shape and dtype — never the file size, which DEFLATE under-reports by
        2-10x.  It is what the load-route decision must be made on.

    Raises
    ------
    FileNotFoundError
        If *folder_path* does not exist.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    results = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        fpath = os.path.join(folder_path, fname)
        dtype = None
        nbytes = None
        axes = None
        compression = None
        try:
            with tifffile.TiffFile(fpath) as tf:
                series = tf.series[0]
                shape = tuple(series.shape)
                try:
                    dtype = str(series.dtype)
                    # math.prod (pure-Python, arbitrary precision) NOT np.prod:
                    # np.prod accumulates in the platform default int (int32 on
                    # Windows), so a >2^31-voxel volume (e.g. a 35 GB uint16
                    # stack ~= 18e9 voxels) OVERFLOWS -> "overflow encountered in
                    # scalar multiply" + a wrapped/garbage nbytes that then
                    # mis-drives plan_load_route's >RAM decision.  math.prod
                    # never overflows.
                    nbytes = math.prod(shape) * int(series.dtype.itemsize)
                except Exception:
                    dtype, nbytes = None, None
                # The load plan wants axes and compression from the same header pass.
                # `compression` is what decides MEMMAP vs LAZY_ZARR: an
                # uncompressed page can be memory-mapped zero-copy, so a >RAM
                # raw TIFF must NOT be refused.
                axes = getattr(series, "axes", None)
                try:
                    page = tf.pages[0]
                    comp = getattr(page, "compression", None)
                    name = getattr(comp, "name", None) or str(comp)
                    compression = name
                except Exception:
                    compression = None
        except Exception:
            shape = None
        results.append({
            "path": fpath,
            "filename": fname,
            "stem": os.path.splitext(fname)[0],
            "shape": shape,
            "dtype": dtype,
            "nbytes": nbytes,
            "axes": axes,
            "compression": compression,
        })
    return results


def transpose_to_czyx(data):
    """Swap the first two axes of an array (e.g. ZCYX → CZYX).

    If the array has fewer than 2 dimensions it is returned unchanged.

    Parameters
    ----------
    data : np.ndarray
        Array to transpose.

    Returns
    -------
    np.ndarray
        Array with axes 0 and 1 swapped.
    """
    if data.ndim < 2:
        return data
    return np.swapaxes(data, 0, 1)


def split_channels(data, display_name):
    """Split a multi-channel volume into individual 3-D channel arrays.

    Parameters
    ----------
    data : np.ndarray
        4-D ``(C, Z, Y, X)`` or 3-D ``(Z, Y, X)`` array.
    display_name : str
        Human-readable name used as prefix for channel names.

    Returns
    -------
    list of (str, np.ndarray)
        Each tuple is ``(channel_name, 3d_array)``.

    Raises
    ------
    ValueError
        If *data* is not 3-D or 4-D.
    """
    if data.ndim == 4:
        return [(f"{display_name}_ch{i}", data[i]) for i in range(data.shape[0])]
    if data.ndim == 3:
        return [(display_name, data)]
    raise ValueError(
        f"Expected 3D or 4D array, got {data.ndim}D with shape {data.shape}"
    )


def _indices_for_prefix_in_dir(directory, name_prefix):
    """Collect indices N from ``{name_prefix}_{N}.tif(f)`` files in *directory*."""
    if not directory or not os.path.isdir(directory):
        return []
    pattern = re.compile(rf"^{re.escape(name_prefix)}_(\d+)\.tiff?$", re.IGNORECASE)
    out = []
    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if m:
            out.append(int(m.group(1)))
    return out


def _indices_for_prefix_in_registry(workspace_path, name_prefix, state=None):
    """Collect indices N from ``{name_prefix}_{N}`` patch_names in step1_state.json.

    Pass a pre-loaded *state* to avoid re-reading the JSON (the source badge
    refresh calls this once per source file — hoist the load out of that loop).
    """
    if state is None:
        state = load_state(workspace_path)
    pattern = re.compile(rf"^{re.escape(name_prefix)}_(\d+)$", re.IGNORECASE)
    out = []
    for entry in state.get("patches", []):
        m = pattern.match(str(entry.get("patch_name", "")))
        if m:
            out.append(int(m.group(1)))
    return out


def find_next_patch_index(workspace_path, name_prefix, state=None):
    """Next 1-based cumulative index for ``{name_prefix}_{N}``.

    Unions three sources so cumulative naming stays correct in ANY migration
    state (no overwrite, no index reset):

    1. the ``step1_state.json`` registry (authoritative when present),
    2. the canonical patches dir ``artifacts/step1/patches/``,
    3. the legacy workspace root (un-migrated workspaces).

    Returns ``max(found) + 1``, or ``1`` if nothing matches.

    Parameters
    ----------
    workspace_path : str
        Workspace root.
    name_prefix : str
        Prefix to match (e.g. ``"kidney_anterior_patch"``).
    state : dict, optional
        Pre-loaded step1_state.json (avoids a re-read when the caller loops).
    """
    indices = []
    indices += _indices_for_prefix_in_registry(workspace_path, name_prefix, state=state)
    indices += _indices_for_prefix_in_dir(
        str(step1_patches_dir(workspace_path)), name_prefix
    )
    indices += _indices_for_prefix_in_dir(workspace_path, name_prefix)
    if not indices:
        return 1
    return max(indices) + 1


def count_patches_for_prefix(workspace_path, name_prefix, state=None):
    """Count DISTINCT patches matching ``{name_prefix}_{N}`` across the registry,
    the patches dir, and the legacy root (deduplicated by index).

    Unlike ``find_next_patch_index() - 1`` (which returns the max index and so
    over-counts when indices have gaps), this returns the true patch count.
    Pass a pre-loaded *state* to avoid re-reading the JSON per call.
    """
    indices = set()
    indices.update(_indices_for_prefix_in_registry(workspace_path, name_prefix, state=state))
    indices.update(
        _indices_for_prefix_in_dir(str(step1_patches_dir(workspace_path)), name_prefix)
    )
    indices.update(_indices_for_prefix_in_dir(workspace_path, name_prefix))
    return len(indices)


def count_patches_for_source(state, source_id):
    """Number of registry patches belonging to *source_id* — the truly per-FILE
    count (a file's patches all share one source_id regardless of the prefix they
    were named under, so a rename doesn't undercount and two same-named files
    don't cross-count).  Returns 0 for an empty/falsy source_id (legacy patches
    carry ``source_id=""`` and must fall back to the prefix scan)."""
    if not source_id:
        return 0
    return sum(
        1 for p in state.get("patches", []) if p.get("source_id") == source_id
    )


# ---------------------------------------------------------------------------
# Unified naming authority (Commit 1)
#
# `_resolve_unique_name` (in _export.py) used to allocate collisions with a
# NESTED `{name}_{k}` scheme that the indexer regex `^{prefix}_(\d+)$` can never
# match — so a bumped patch became invisible to `find_next_patch_index` + the
# badge count forever.  These three functions replace it with a single authority
# that delegates collisions back to the cumulative index allocator, yielding
# indexer-visible `{prefix}_{max+1}` names.
# ---------------------------------------------------------------------------

_INDEXED_NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)$")


def split_indexed_name(patch_name):
    """Split ``{prefix}_{N}`` into ``(prefix, N)``; non-indexed -> ``(name, None)``.

    The prefix is greedy, so only the FINAL ``_{digits}`` group is the index:
    ``roi_2_patch_3`` -> ``("roi_2_patch", 3)``.  A leading-underscore number
    (``_5``) is non-indexed — the prefix must be non-empty.

    NOTE: the index is a plain ``int`` (``_007`` -> 7).  Step 1 only ever emits
    UN-padded indices, so the int-index scans (find_next / count) and the string
    name check agree.  A zero-padded name from migration / external tooling would
    diverge (``_007`` counted as index 7 but its file distinct from ``_7``) — add
    a normalization guard here if that path is ever introduced.
    """
    m = _INDEXED_NAME_RE.match(str(patch_name))
    if m:
        return m.group("prefix"), int(m.group("index"))
    return str(patch_name), None


def _casefold_listdir(directory):
    """Case-folded set of filenames in *directory* (``set()`` if not a dir).

    Case-folded so the on-disk collision check matches NTFS behavior on
    case-SENSITIVE filesystems too (ext4, where the test batch runs) — parity
    with the ``re.IGNORECASE`` index scans.  Deliberately NOT ``Path.exists()``
    (which is case-sensitive on ext4 and would be weaker than the indexer).
    """
    if not directory or not os.path.isdir(directory):
        return set()
    return {fname.casefold() for fname in os.listdir(directory)}


def patch_name_taken(workspace_path, patch_name, state=None):
    """True if *patch_name* already exists as a registry entry, an on-disk TIFF
    (``.tif`` OR ``.tiff``), or a ``_metadata.json`` sidecar — in the canonical
    patches dir OR the legacy workspace root.  Case-insensitive on every platform.

    Pass a pre-loaded *state* to avoid re-reading the registry JSON.
    """
    target = str(patch_name).casefold()
    if state is None:
        state = load_state(workspace_path)
    for entry in state.get("patches", []):
        if str(entry.get("patch_name", "")).casefold() == target:
            return True
    wanted = {f"{target}.tif", f"{target}.tiff", f"{target}_metadata.json"}
    for directory in (str(step1_patches_dir(workspace_path)), str(workspace_path)):
        if _casefold_listdir(directory) & wanted:
            return True
    return False


def allocate_unique_patch_name(workspace_path, patch_name, state=None):
    """THE single naming authority: return *patch_name* if free, else an
    indexer-compatible unique name.

    * indexed (``{prefix}_{N}``) -> ``{prefix}_{find_next_patch_index(prefix)}``,
      advanced past any already-taken index (incl. an orphan sidecar the TIFF
      scan can't see).  Stays visible to ``find_next_patch_index`` + the badge.
    * non-indexed -> first free ``{name}_{k}`` (k>=2), preserving the historical
      overwrite-protection behavior (``p`` -> ``p_2``).

    The trailing loop uses a LOCAL counter and NEVER re-invokes
    ``find_next_patch_index`` inside the loop (it cannot see an orphan sidecar,
    which would spin forever) — so it always terminates on the finite taken-set.

    Pass a pre-loaded *state* to avoid re-reading the registry per check.
    """
    if state is None:
        state = load_state(workspace_path)
    if not patch_name_taken(workspace_path, patch_name, state=state):
        return patch_name
    prefix, index = split_indexed_name(patch_name)
    if index is not None:
        k = find_next_patch_index(workspace_path, prefix, state=state)
        while patch_name_taken(workspace_path, f"{prefix}_{k}", state=state):
            k += 1
        return f"{prefix}_{k}"
    k = 2
    while patch_name_taken(workspace_path, f"{patch_name}_{k}", state=state):
        k += 1
    return f"{patch_name}_{k}"


def step1_sidecar_scan_dir(workspace_path):
    """Directory to scan for patch ``_metadata.json`` sidecars: the canonical
    ``artifacts/step1/patches/`` if it exists, else the legacy workspace root
    (un-migrated).  Used by the controller's bounding-box restore."""
    patches_dir = step1_patches_dir(workspace_path)
    return str(patches_dir) if os.path.isdir(patches_dir) else str(workspace_path)


# ---------------------------------------------------------------------------
# Large-data load routing
# ---------------------------------------------------------------------------

#: Fraction of free RAM a decompressed stack may occupy and still load eagerly.
EAGER_RAM_FRACTION = 0.6


#: Compression tags that mean "stored uncompressed", so pages can be
#: memory-mapped zero-copy.  tifffile reports these as NONE / 1.
_UNCOMPRESSED = {"NONE", "COMPRESSION.NONE", "1", "NONE (1)"}


def plan_load_route(nbytes, available_bytes, n_channels=1, compression=None,
                    fraction=EAGER_RAM_FRACTION):
    """Decide how a source must be loaded, from header numbers only.

    Pure and Qt-free so it can be unit-tested.  Implements the routing table
    verbatim, including the middle branch:

        view    = EAGER     if decompressed <= fraction * free_ram
                = MEMMAP    elif compression is raw   # zero-copy, no conversion
                = LAZY_ZARR else
        process = IN_MEMORY if the per-channel float32 peak fits else OUT_OF_CORE

    The MEMMAP branch matters: an UNCOMPRESSED >RAM TIFF can be read
    zero-copy, so refusing it would be a false refusal on exactly the file
    format large microscopy data usually ships in.

    ``eager`` means "safe to hand to `tifffile.imread`", which is true for
    EAGER only — MEMMAP needs `memmap()`, not `imread`.  ``needs_conversion``
    distinguishes the two so a caller can offer memory-mapping rather than
    only OME-Zarr.

    The process budget divides by ``n_channels`` because ops run one channel at
    a time, and multiplies by 4 for the float32 working copy plus headroom.
    NOTE: the original illustrative copy ("220 GB → ~1.3 TB") does not follow its own
    stated rule (220 GB at 4x, 2 channels, is 440 GB); the RULE is implemented
    here, not the example number.
    """
    if not nbytes or not available_bytes:
        # Unknown size — do not block, but record that the check could not run.
        return {"view": "UNKNOWN", "process": "UNKNOWN", "eager": True,
                "needs_conversion": False, "nbytes": nbytes or 0,
                "proc_peak": 0}
    budget = fraction * float(available_bytes)
    fits = float(nbytes) <= budget
    raw = str(compression or "").upper() in _UNCOMPRESSED
    if fits:
        view = "EAGER"
    elif raw:
        view = "MEMMAP"
    else:
        view = "LAZY_ZARR"
    per_channel = float(nbytes) / max(1, int(n_channels))
    proc_peak = per_channel * 4.0
    return {
        "view": view,
        "process": "IN_MEMORY" if proc_peak <= budget else "OUT_OF_CORE",
        "eager": fits,
        # Only LAZY_ZARR needs a re-save; MEMMAP is readable as-is.
        "needs_conversion": view == "LAZY_ZARR",
        "nbytes": int(nbytes),
        "proc_peak": int(proc_peak),
    }


def format_bytes(n):
    """Human-readable size, for the ① card and the load-guard message."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"
