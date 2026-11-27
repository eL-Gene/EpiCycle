"""De-novo region segmentation for 3D label curation (the "Add" tool).

Pure algorithm module — no Qt, no napari.  All heavy dependencies
(scipy, skimage) are imported lazily inside functions.

The "Add" tool segments a *missed* cell/nucleus from scratch inside a
user-drawn box: threshold an image channel, separate touching pieces with a
distance-watershed, and emit new labels **clipped to background** — existing
labels are never overwritten (Add only ever fills gaps).

Algorithm overview (``add_segment_region``):
  1. optional CLAHE on ``image_roi`` (contrast enhance; works in [0,1])
  2. ``binary = proc >= threshold``         (Otsu or explicit value)
  3. ``fg = binary & (existing_roi == 0)``   — segment ONLY background
  4. if ``separate``: ``peak_local_max(EDT(fg))`` seeds → ``watershed(-EDT)``
     else: connected components
  5. drop pieces ``< min_size`` voxels
  6. relabel survivors sequentially from ``next_label``

Output invariant: ``out`` is nonzero ONLY where ``existing_roi == 0`` (the
clipping rule holds *by construction*, step 3), so ``out`` and any existing
label never share a voxel.

Reuses the same primitives as ``split.py`` (``distance_transform_edt``,
``watershed``) and ``seed_detection.py`` (``peak_local_max``).
"""

from __future__ import annotations

import numbers

import numpy as np

#: At most this many threads smooth Add's new objects (and never more than
#: the cores this process may use minus two, so the window and the rest of
#: the work keep a core each).  Eight measured fastest on a 20-core machine:
#: 720 objects in 4.7 s against 20 s on one thread, while 18 threads took
#: 5.2 s.
ADD_SMOOTH_MAX_WORKERS = 8


def smooth_worker_count() -> int:
    """Threads for :func:`smooth_pieces`: ``min(8, usable cores - 2)``, at
    least 1 (see :mod:`.pool_workers`)."""
    from .pool_workers import pool_worker_count

    return pool_worker_count(ADD_SMOOTH_MAX_WORKERS)


def _normalize01(image: np.ndarray) -> np.ndarray:
    """Min-max scale a float image to [0, 1].  Constant images map to all-zeros."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float64)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float64)
    return ((image - lo) / (hi - lo)).astype(np.float64)


def _resolve_threshold(proc: np.ndarray, threshold) -> float | None:
    """Resolve ``threshold`` (``"otsu"`` or a number) to a scalar cut value.

    Returns ``None`` for a degenerate image (no finite range → nothing to
    threshold), which the caller treats as "no foreground".
    """
    if isinstance(threshold, str):
        method = threshold.lower()
        if method != "otsu":
            raise ValueError(f"unknown threshold method {threshold!r}")
        finite = proc[np.isfinite(proc)]
        if finite.size == 0 or float(finite.min()) == float(finite.max()):
            return None
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(proc))
    return float(threshold)


def add_threshold_space(image_roi, *, clahe=False, clahe_clip=0.01):
    """The image values the threshold is compared against.

    Raw image when ``clahe`` is off; min-max normalized to [0, 1] then CLAHE
    when on.  Shared by :func:`add_segment_region` (internally) and the Add
    tab's live in-box histogram so the draggable threshold line sits in exactly
    the value space the watershed will threshold — they never disagree.
    """
    image = np.asarray(image_roi, dtype=np.float64)
    if not clahe:
        return image
    from skimage.exposure import equalize_adapthist

    norm = _normalize01(image)
    if not np.any(norm):
        return norm
    return equalize_adapthist(norm, clip_limit=clahe_clip)


def add_segment_region(
    image_roi: np.ndarray,
    existing_roi: np.ndarray,
    *,
    clahe: bool = False,
    clahe_clip: float = 0.01,
    threshold="otsu",
    min_size: int = 0,
    separate: bool = True,
    min_distance: int = 3,
    next_label: int,
    spacing=None,
    cancel_event=None,
) -> np.ndarray:
    """Segment new label(s) from an image channel inside a box, clipped to gaps.

    Parameters
    ----------
    image_roi : np.ndarray
        Single-channel image cropped to the box (2D or 3D).  Multichannel
        images MUST be channel-collapsed by the caller first.
    existing_roi : np.ndarray
        Current label volume cropped to the same box.  Foreground is segmented
        ONLY where this is 0, so new pieces never overwrite a neighbour.
    clahe : bool
        Apply CLAHE (``equalize_adapthist``) before thresholding.  When on, the
        image is normalized to [0, 1] and ``threshold`` is interpreted in that
        space; when off, ``threshold`` is in the raw image's value space.
    clahe_clip : float
        CLAHE clip limit (skimage default 0.01).
    threshold : {"otsu"} or float
        ``"otsu"`` auto-thresholds; a number cuts at that value (``proc >= t``).
    min_size : int
        Drop connected pieces smaller than this many voxels (0 = keep all).
    separate : bool
        If True, distance-watershed splits touching pieces; else connected
        components (one label per blob).
    min_distance : int
        Minimum voxel separation between watershed seeds (``peak_local_max``).
    next_label : int
        First label ID to assign (typically ``max(volume) + 1``).  Survivors
        are numbered ``next_label, next_label + 1, ...``.
    spacing : sequence of float, optional
        Per-axis voxel spacing for the EDT (anisotropy-aware watershed).
    cancel_event : threading.Event, optional
        Polled BETWEEN the steps (the contrast, the threshold, the distance
        map, its blur, the peaks, the flood, the size floor); once set,
        ``ReshapeCancelled`` is raised at the next poll.  A step already
        running (the distance map, the flood) is never cut short.  ``None``
        polls nothing: the result is the one it always was.

    Returns
    -------
    np.ndarray
        ``int32`` array, same shape as *existing_roi*, nonzero ONLY in
        background (``existing_roi == 0``), relabeled from *next_label*.
    """
    image = np.asarray(image_roi, dtype=np.float64)
    existing = np.asarray(existing_roi)
    if image.shape != existing.shape:
        raise ValueError(
            f"image_roi shape {image.shape} != existing_roi shape "
            f"{existing.shape}"
        )

    out = np.zeros(existing.shape, dtype=np.int32)
    if image.size == 0:
        return out
    _poll_stop(cancel_event)

    # --- 1. optional CLAHE (requires [0,1]) ---
    proc = add_threshold_space(image, clahe=clahe, clahe_clip=clahe_clip)
    if clahe and not np.any(proc):  # constant image → nothing to enhance/segment
        return out
    _poll_stop(cancel_event)

    # --- 2. threshold ---
    thr = _resolve_threshold(proc, threshold)
    if thr is None:
        return out
    _poll_stop(cancel_event)
    binary = proc >= thr

    # --- 3. background-masked foreground (the clipping invariant) ---
    fg = binary & (existing == 0)
    if not np.any(fg):
        return out
    _poll_stop(cancel_event)

    # --- 4. separate touching pieces ---
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import label as cc_label

    if separate:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        from scipy.ndimage import gaussian_filter

        edt = distance_transform_edt(fg, sampling=spacing).astype(np.float64)
        _poll_stop(cancel_event)
        # Lightly smooth the EDT before peak detection so flat interior
        # plateaus collapse to a single maximum per blob — otherwise
        # peak_local_max returns many co-equal peaks across a plateau and the
        # watershed over-segments one cell into a dozen slivers.
        edt_smooth = gaussian_filter(edt, sigma=1.0)
        _poll_stop(cancel_event)
        peaks = peak_local_max(
            edt_smooth,
            min_distance=max(1, int(min_distance)),
            exclude_border=False,
            labels=fg,
        )
        _poll_stop(cancel_event)
        if len(peaks) >= 2:
            markers = np.zeros(fg.shape, dtype=np.int32)
            for i, coord in enumerate(peaks):
                markers[tuple(int(c) for c in coord)] = i + 1
            pieces = watershed(-edt, markers=markers, mask=fg)
        else:
            # 0 or 1 interior peak → no meaningful split; one piece per blob.
            pieces, _ = cc_label(fg)
    else:
        pieces, _ = cc_label(fg)
    _poll_stop(cancel_event)

    # --- 5 + 6. drop small pieces, relabel from next_label ---
    # One lookup-table pass (the pieces are numbered 1..K by the watershed or
    # the connected components): the same ids, in the same ascending order,
    # as one mask per piece -- which cost seconds on a whole patch.
    pieces = np.asarray(pieces)
    if not pieces.size or int(pieces.max()) <= 0:
        return out
    sizes = np.bincount(pieces.ravel().astype(np.int64, copy=False))
    lut = np.zeros(sizes.size, dtype=np.int64)
    new_id = int(next_label)
    for val in np.flatnonzero(sizes):
        if val == 0:
            continue
        if min_size and int(sizes[val]) < int(min_size):
            continue
        lut[val] = new_id
        new_id += 1
    # int32 as always; above the int32 maximum (a uint32 volume near 2**31)
    # int64, as the preview's _mint does -- an int32 cast would wrap those
    # ids negative
    top = new_id - 1
    dtype = np.int32 if top <= np.iinfo(np.int32).max else np.int64
    out = lut[pieces].astype(dtype, copy=False)
    return out


def _poll_stop(cancel_event):
    """Raise ``ReshapeCancelled`` once the Preview's Stop is pressed."""
    if cancel_event is not None and cancel_event.is_set():
        from .reshape.engine import ReshapeCancelled

        raise ReshapeCancelled()


def _climb(smooth, start, piece):
    """Steepest ascent of ``smooth`` from ``start`` over the 26 neighbours,
    inside ``piece``: the path, one face step at a time (one 6-connected
    run of voxels), ending on a voxel no neighbour of the piece exceeds."""
    p = tuple(int(v) for v in start)
    path = [p]
    shape = smooth.shape
    while True:
        lo = [max(0, p[a] - 1) for a in range(3)]
        hi = [min(shape[a], p[a] + 2) for a in range(3)]
        sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        win = np.where(piece[sl], smooth[sl], -1.0)
        idx = np.unravel_index(int(np.argmax(win)), win.shape)
        if win[idx] <= smooth[p]:
            return path
        q = tuple(lo[a] + int(idx[a]) for a in range(3))
        step = list(p)
        for a in range(3):
            if step[a] != q[a]:
                step[a] = q[a]
                path.append(tuple(step))
        p = q


def _cap(smooth, top, piece):
    """The flat top holding ``top``: the voxels of ``top``'s height joined
    to it (26-connected, inside ``piece``), as a boolean array over a window
    and the window's corner; the window grows until the cap does not touch
    it (or it is the whole piece's box)."""
    from scipy import ndimage as ndi

    level = float(smooth[top])
    shape = smooth.shape
    r = 4
    while True:
        lo = [max(0, top[a] - r) for a in range(3)]
        hi = [min(shape[a], top[a] + r + 1) for a in range(3)]
        sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        up = piece[sl] & (smooth[sl] == level)
        lab, _n = ndi.label(up, structure=np.ones((3, 3, 3), bool))
        cap = lab == lab[tuple(top[a] - lo[a] for a in range(3))]
        whole = all(lo[a] == 0 and hi[a] == shape[a] for a in range(3))
        edge = any(cap.take(0, axis=a).any() and lo[a] > 0
                   or cap.take(-1, axis=a).any() and hi[a] < shape[a]
                   for a in range(3))
        if whole or not edge:
            return cap, lo
        r *= 2


def _extra_start(smooth, own, vox, start):
    """Where a marker's later seed climbs from, in piece-box coordinates:
    its own click when that is on the piece, else its ball voxel highest
    on ``smooth`` (``vox`` are the ball's voxels on the piece, volume
    coordinates)."""
    if own is not None:
        return tuple(int(v) for v in np.asarray(own) - start)
    v = np.asarray(vox) - start
    return tuple(int(x) for x in v[int(np.argmax(smooth[tuple(v.T)]))])


def _gain_tops(edt, comp, markers, claimed, slices, clicked,
               cancel_event=None, extra=None):
    """``(gained, anchors)``: what each marker gains, ``{k: (N, 3) voxels}``,
    and the voxel each one climbed from.  In every bright piece holding two
    markers or more, each marker climbs the distance map from its click
    (its deepest voxel for a near miss).  Two climbs reach the same hill
    only when they end on one flat top (voxels of one height, joined); a
    marker gains its climb -- a path from its ball to its hill's top, so
    its object stays one piece -- only when no other marker reaches that
    hill.  A piece with one marker needs nothing: it floods all of it.
    ``extra`` ``{k: [(own click or None, ball voxels)]}``: a marker made of
    several seeds also climbs from each later seed, and gains every climb
    whose hill no other marker reaches.
    Stop is polled once per climb and once per top."""
    from scipy import ndimage as ndi

    by_piece = {}
    for k, c in claimed.items():
        by_piece.setdefault(c, []).append(k)
    multi = {c: ks for c, ks in by_piece.items() if len(ks) > 1}
    gained, anchors = {}, {}
    if not multi:
        return gained, anchors
    extra = extra or {}
    boxes = ndi.find_objects(comp, max_label=max(multi))
    for c, ks in sorted(multi.items()):
        sl = boxes[c - 1]
        start = np.asarray([s.start for s in sl])
        piece = comp[sl] == c
        smooth = np.where(piece, edt[sl], -1.0)
        climbs = {}
        more = {}
        for k in ks:
            for own, vox in extra.get(k, ()):
                _poll_stop(cancel_event)
                more.setdefault(k, []).append(_climb(
                    smooth, _extra_start(smooth, own, vox, start), piece))
        for k in ks:
            _poll_stop(cancel_event)
            bs = slices[k - 1]
            vox = np.argwhere(markers[bs] == k)
            if not len(vox):
                continue
            vox = vox + np.asarray([s.start for s in bs]) - start
            own = clicked.get(k)
            if own is not None:
                first = tuple(int(v) for v in np.asarray(own) - start)
            else:
                first = tuple(int(v) for v in
                              vox[int(np.argmax(smooth[tuple(vox.T)]))])
            anchors[k] = tuple(int(v) for v in np.asarray(first) + start)
            climbs[k] = _climb(smooth, first, piece)
        # one hill per flat top: a top whose cap holds another top joins
        # it.  A top already inside an earlier cap has that same cap, so
        # it is not labelled again
        tops = sorted({path[-1] for path in climbs.values()}
                      | {p[-1] for ps in more.values() for p in ps})
        hill = {t: t for t in tops}

        def root(t):
            while hill[t] != t:
                hill[t] = hill[hill[t]]
                t = hill[t]
            return t

        if len(tops) > 1:
            where = np.zeros(piece.shape, np.int32)
            for i, t in enumerate(tops):
                where[t] = i + 1
            done = set()
            for t in tops:
                _poll_stop(cancel_event)
                if t in done:
                    continue
                cap, lo = _cap(smooth, t, piece)
                win = tuple(slice(lo[a], lo[a] + cap.shape[a])
                            for a in range(3))
                for i in np.unique(where[win][cap]):
                    if i:
                        done.add(tops[int(i) - 1])
                        hill[root(tops[int(i) - 1])] = root(t)
        on_hill = {}
        for k, path in climbs.items():
            on_hill.setdefault(root(path[-1]), []).append(k)
        for k, paths in more.items():
            for path in paths:
                kk = on_hill.setdefault(root(path[-1]), [])
                if k not in kk:
                    kk.append(k)
        for top, kk in on_hill.items():
            if len(kk) == 1:
                k = kk[0]
                paths = [p for p in ([climbs[k]] if k in climbs else [])
                         + more.get(k, []) if root(p[-1]) == top]
                for path in paths:
                    path = np.asarray(path)
                    # a face step round a corner can leave the piece: not
                    # gained
                    got = path[piece[tuple(path.T)]] + start
                    gained[k] = (got if k not in gained
                                 else np.concatenate([gained[k], got]))
    return gained, anchors


def _split_ids(pieces):
    """The ids that come out as more than one 6-connected piece, with each
    piece's labels: ``{id: [region labels]}`` and the region map."""
    from skimage.measure import label as sk_label

    regions = sk_label(pieces, background=0, connectivity=1)
    n = int(regions.max())
    if n == 0:
        return {}, regions
    owner = np.zeros(n + 1, np.int64)
    flat_r = regions.ravel()
    nz = flat_r > 0
    owner[flat_r[nz]] = pieces.ravel()[nz]
    by_id = {}
    for r in range(1, n + 1):
        by_id.setdefault(int(owner[r]), []).append(r)
    return {k: rs for k, rs in by_id.items() if len(rs) > 1}, regions


#: ``segment_at_markers``' ``split``: ``"off"`` floods each clicked bright
#: piece from its markers alone (one click on touching nuclei gives them
#: all); ``"always"`` also cuts every clicked piece at its waists, so one
#: click gives the nucleus it sits on; ``"single"`` cuts only the pieces
#: that hold exactly one claimed click.
MARKER_SPLIT_MODES = ("off", "single", "always")
#: the markers' Spacing when none is given, in xy voxels: nuclei whose
#: middles lie closer than this count as one
ADD_MARKER_SPACING_DEFAULT = 5
#: two neighbouring basins are one nucleus when the lowest point of the
#: ridge between them is at least this share of the lower of their two tops
MARKER_SPLIT_SHARE = 0.8
#: the reach of the smoothing of the distance map, in sigmas (scipy's own
#: default, named because the crop's margin depends on it)
MARKER_GAUSS_TRUNCATE = 4.0
#: Stop is polled once per this many rows of the thinning, and per this
#: many steps of the merge
_THIN_POLL_ROWS = 4096
_MERGE_POLL_STEPS = 1024


def _marker_half_widths(peak_spacing_um, spacing):
    """``(n, (h_z, h_y, h_x))``: the markers' Spacing in xy voxels (from
    microns, or :data:`ADD_MARKER_SPACING_DEFAULT` for ``None``) and the
    half-width per axis of the box two peaks must not share -- the same
    physical distance on every axis, at least one voxel.  Python's ``round``
    (half to even); rounding to 9 places first only strips float noise."""
    s = tuple(float(v) for v in spacing)
    if peak_spacing_um is None:
        n = ADD_MARKER_SPACING_DEFAULT
    else:
        # a real number only: a tick (bool) or a typed word (str) is a bug
        # upstream, not a Spacing
        if (isinstance(peak_spacing_um, (bool, np.bool_))
                or not isinstance(peak_spacing_um, numbers.Real)):
            raise ValueError("peak_spacing_um must be a number > 0, got "
                             f"{peak_spacing_um!r}")
        um = float(peak_spacing_um)
        if not np.isfinite(um) or um <= 0:
            raise ValueError("peak_spacing_um must be a number > 0, got "
                             f"{peak_spacing_um!r}")
        n = round(um / s[-1], 9)
    h = tuple(max(1, int(round(round(n * s[-1] / s[a], 9))))
              for a in range(3))
    return n, h


def _marker_crop_margin(spacing):
    """Voxels per axis the split works beyond the clicked pieces' box: the
    smoothing's reach plus one (scipy's kernel radius,
    ``int(truncate * sigma + 0.5)``, with a sigma of one xy voxel; the
    sigma is computed first, as scipy computes it)."""
    s = tuple(float(v) for v in spacing)
    return tuple(int(MARKER_GAUSS_TRUNCATE * (s[-1] / s[a]) + 0.5) + 1
                 for a in range(3))


def _peak_order(heights):
    """The order of peaks by height, highest first; equal heights keep
    the order they came in (skimage's)."""
    heights = np.asarray(heights)
    return np.lexsort((np.arange(len(heights)), -heights))


def _thin_peaks(q, labs, n, cancel_event=None):
    """The rows kept when peaks (``q``, already highest first; coordinates
    in xy voxels) are thinned: a row is kept unless a row kept before it,
    in the same piece (``labs``), lies within ``n`` on every axis.  The
    same rows as the plain loop over the kept rows, in one pass over the
    close pairs a KD-tree finds.  Stop is polled every few thousand rows."""
    from scipy.spatial import cKDTree

    _poll_stop(cancel_event)
    q = np.asarray(q, dtype=np.float64).reshape(-1, 3)
    labs = np.asarray(labs)
    count = len(q)
    if count < 2:
        return np.arange(count, dtype=np.int64)
    tol = float(n) + 1e-9
    pairs = cKDTree(q).query_pairs(r=float(n) + 2e-9, p=np.inf,
                                   output_type="ndarray")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    i, j = pairs[:, 0], pairs[:, 1]
    close = ((labs[i] == labs[j])
             & (np.max(np.abs(q[i] - q[j]), axis=1) <= tol))
    i, j = i[close], j[close]
    a = np.concatenate([i, j])
    b = np.concatenate([j, i])
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    ptr = np.searchsorted(a, np.arange(count + 1))
    gone = np.zeros(count, dtype=bool)
    keep = []
    for row in range(count):
        if row % _THIN_POLL_ROWS == 0:
            _poll_stop(cancel_event)
        if gone[row]:
            continue
        keep.append(row)
        gone[b[ptr[row]:ptr[row + 1]]] = True
    return np.asarray(keep, dtype=np.int64)


def _hills_with_peaks(sm, edt, comp, markers, claimed, clicked, peaks,
                      half, n_points, cancel_event=None, extra=None):
    """The hills of the smoothed distance map ``sm``, per clicked piece.

    Every claimed marker climbs ``sm`` from its anchor (its click, or for a
    near miss its marker voxel highest on ``sm``) and every peak climbs from
    itself; tops whose flat caps meet are one hill (as :func:`_gain_tops`
    groups them).  A marker alone among markers on its hill gains its climb
    inside the piece.  A peak is dropped when its hill holds a marker or
    when its top lies within ``half`` on every axis of a marker's top in
    the same piece.  Returns ``(gained, anchors, tops, free)``: gained
    paths ``{k: (N, 3)}``, anchors ``{k: voxel}``, the marker tops
    ``{k: voxel}``, and the free peaks ``[(voxel, top height on edt)]`` in
    piece order, then peak order.  ``extra`` (see :func:`_gain_tops`): a
    marker's later seeds climb too; their hills are the marker's, and a
    peak near any of its tops is dropped.  Stop is polled once per climb
    and once per top."""
    from scipy import ndimage as ndi

    by_piece = {}
    for k, c in claimed.items():
        by_piece.setdefault(c, []).append(k)
    gained, anchors, mtop, free = {}, {}, {}, []
    if not by_piece:
        return gained, anchors, mtop, free
    extra = extra or {}
    slices = ndi.find_objects(markers, max_label=n_points)
    boxes = ndi.find_objects(comp, max_label=max(by_piece))
    plab = comp[tuple(peaks.T)] if len(peaks) else np.zeros(0, comp.dtype)
    peaks_of = {}
    for j, c in enumerate(plab.tolist()):
        peaks_of.setdefault(int(c), []).append(j)
    for c, ks in sorted(by_piece.items()):
        sl = boxes[c - 1]
        start = np.asarray([s.start for s in sl])
        piece = comp[sl] == c
        smooth = np.where(piece, sm[sl], -1.0)
        climbs = {}
        for k in ks:
            _poll_stop(cancel_event)
            bs = slices[k - 1]
            vox = np.argwhere(markers[bs] == k)
            if not len(vox):
                continue
            vox = vox + np.asarray([s.start for s in bs]) - start
            own = clicked.get(k)
            if own is not None:
                first = tuple(int(v) for v in np.asarray(own) - start)
            else:
                first = tuple(int(v) for v in
                              vox[int(np.argmax(smooth[tuple(vox.T)]))])
            anchors[k] = tuple(int(v) for v in np.asarray(first) + start)
            climbs[("m", k)] = _climb(smooth, first, piece)
        for k in ks:
            for i, (own, vox) in enumerate(extra.get(k, ())):
                _poll_stop(cancel_event)
                climbs[("e", k, i)] = _climb(
                    smooth, _extra_start(smooth, own, vox, start), piece)
        for j in peaks_of.get(c, ()):
            p = tuple(int(v) for v in peaks[j] - start)
            if not piece[p]:
                continue
            _poll_stop(cancel_event)
            climbs[("p", j)] = _climb(smooth, p, piece)
        tops = sorted({path[-1] for path in climbs.values()})
        hill = {t: t for t in tops}

        def root(t):
            while hill[t] != t:
                hill[t] = hill[hill[t]]
                t = hill[t]
            return t

        if len(tops) > 1:
            where = np.zeros(piece.shape, np.int32)
            for i, t in enumerate(tops):
                where[t] = i + 1
            done = set()
            for t in tops:
                _poll_stop(cancel_event)
                if t in done:
                    continue
                cap, lo = _cap(smooth, t, piece)
                win = tuple(slice(lo[a], lo[a] + cap.shape[a])
                            for a in range(3))
                for i in np.unique(where[win][cap]):
                    if i:
                        done.add(tops[int(i) - 1])
                        hill[root(tops[int(i) - 1])] = root(t)
        on_hill = {}
        for key, path in climbs.items():
            if key[0] in ("m", "e"):
                kk = on_hill.setdefault(root(path[-1]), [])
                if key[1] not in kk:
                    kk.append(key[1])
            if key[0] == "m":
                mtop[key[1]] = tuple(int(v) for v in
                                     np.asarray(path[-1]) + start)
        for top, kk in on_hill.items():
            if len(kk) == 1:
                for key, path in climbs.items():
                    if (key[0] not in ("m", "e") or key[1] != kk[0]
                            or root(path[-1]) != top):
                        continue
                    path = np.asarray(path)
                    # a face step round a corner can leave the piece: not
                    # gained
                    got = path[piece[tuple(path.T)]] + start
                    gained[kk[0]] = (
                        got if kk[0] not in gained
                        else np.concatenate([gained[kk[0]], got]))
        marker_tops = [path[-1] for key, path in climbs.items()
                       if key[0] in ("m", "e")]
        raw = edt[sl]
        for key, path in climbs.items():
            if key[0] != "p" or root(path[-1]) in on_hill:
                continue
            t = path[-1]
            if any(all(abs(t[a] - m[a]) <= half[a] for a in range(3))
                   for m in marker_tops):
                continue          # too close to a marker's top: its hill
            free.append((tuple(int(v) for v in peaks[key[1]]),
                         float(raw[t])))
    return gained, anchors, mtop, free


def _basin_saddles(ws, edt):
    """``(a, b, saddle)``: every pair of basins ``a < b`` of ``ws`` with two
    voxels one face apart, and the highest ``min(edt)`` over those face
    pairs.  One pass per axis; the largest label is read once and each pair
    is coded ``a * (L + 1) + b``."""
    ws = np.asarray(ws)
    big = int(ws.max()) if ws.size else 0
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64),
             np.zeros(0, np.float64))
    if big <= 0:
        return empty
    width = np.int64(big + 1)
    codes, vals = [], []
    for ax in range(ws.ndim):
        if ws.shape[ax] < 2:
            continue
        lo = [slice(None)] * ws.ndim
        hi = [slice(None)] * ws.ndim
        lo[ax] = slice(0, -1)
        hi[ax] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        one, two = ws[lo], ws[hi]
        sel = (one != two) & (one > 0) & (two > 0)
        if not sel.any():
            continue
        x = one[sel].astype(np.int64)
        y = two[sel].astype(np.int64)
        codes.append(np.minimum(x, y) * width + np.maximum(x, y))
        vals.append(np.minimum(edt[lo][sel], edt[hi][sel]))
    if not codes:
        return empty
    code = np.concatenate(codes)
    val = np.concatenate(vals)
    order = np.argsort(code, kind="stable")
    code, val = code[order], val[order]
    pairs, first = np.unique(code, return_index=True)
    saddle = np.maximum.reduceat(val, first)
    return pairs // width, pairs % width, saddle.astype(np.float64)


def _grow_groups(pairs, marker_tops, anon_tops, cancel_event=None):
    """``{j: k}``: the anonymous basins each marker's group takes.

    A group (a marker and the anonymous basins it took) grows best-first:
    the edge from a member ``m`` to an untaken anonymous basin ``j``
    qualifies when their saddle is at least :data:`MARKER_SPLIT_SHARE` of
    the lower of the two tops (``m``'s own top).  The highest qualifying
    saddle goes first (a tie to the lower marker id, then the lower
    ``j``); two markers never join.  Stop is polled every thousand steps."""
    import heapq

    a, b, s = pairs
    top = dict(marker_tops)
    top.update(anon_tops)
    near = {}
    for x, y, v in zip(np.asarray(a).tolist(), np.asarray(b).tolist(),
                       np.asarray(s).tolist()):
        if y in anon_tops:
            near.setdefault(x, []).append((y, v))
        if x in anon_tops:
            near.setdefault(y, []).append((x, v))
    owner = {}
    heap = []

    def push(m, k):
        tm = top[m]
        for j, v in near.get(m, ()):
            if j in owner:
                continue
            if v >= MARKER_SPLIT_SHARE * min(tm, top[j]):
                heapq.heappush(heap, (-v, k, j))

    for k in sorted(marker_tops):
        push(k, k)
    steps = 0
    while heap:
        if steps % _MERGE_POLL_STEPS == 0:
            _poll_stop(cancel_event)
        steps += 1
        _v, k, j = heapq.heappop(heap)
        if j in owner:
            continue
        owner[j] = k
        push(j, k)
    return owner


def _guard_markers(edt, seeds, mask, gained, anchors, keep,
                   cancel_event=None):
    """Flood ``seeds`` over ``mask`` and make every id in ``keep`` one
    6-connected piece: a gain cut off from its anchor is given back (its
    voxels zeroed in the seeds) and the flood re-run; otherwise each split
    id keeps the piece holding its anchor (or most of its seed), the rest
    is zeroed and flooded again from the labels around it.  Ids outside
    ``keep`` may stay in pieces.  Stop is polled after every flood."""
    from skimage.segmentation import watershed

    seeds = np.array(seeds, dtype=np.int32, copy=True)
    gained = dict(gained)
    pieces = None
    for _round in range(len(gained) + 1):
        pieces = np.asarray(watershed(-edt, seeds, mask=mask),
                            dtype=np.int32)
        _poll_stop(cancel_event)
        split, regions = _split_ids(pieces)
        split = {k: rs for k, rs in split.items() if k in keep}
        if not split:
            break
        back = [k for k in split if k in gained
                and (regions[tuple(gained[k].T)]
                     != regions[anchors[k]]).any()]
        if back:
            for k in back:
                seeds[tuple(gained.pop(k).T)] = 0
            continue
        for k, rs in split.items():
            if k in anchors:
                kept = int(regions[anchors[k]])
            else:
                held = regions[seeds == k]
                kept = max(rs, key=lambda r: int((held == r).sum()))
            pieces[np.isin(regions, [r for r in rs if r != kept])] = 0
        pieces = np.asarray(watershed(-edt, pieces, mask=mask),
                            dtype=np.int32)
        _poll_stop(cancel_event)
        break
    return pieces


def _join_group_pieces(ws, edt, anchors, groups, n_points):
    """Nothing between a marker's seeds is left out: for each marker in
    ``groups`` (one made of several seeds) whose basin of ``ws`` came out in
    several 6-connected pieces, the anonymous basins (ids above
    ``n_points``) on the shortest chain of basins from the piece holding its
    anchor to each other piece join it.  Only anonymous basins are crossed:
    another marker's basin is never taken.  ``ws`` with the joined basins
    relabelled; unchanged when no marker needs it."""
    from collections import deque

    from scipy import ndimage as ndi

    lab = np.array(ws, dtype=np.int64, copy=True)
    first = int(ws.max()) + 1 if ws.size else 1
    nxt = first
    pieces_of = {}
    for k in sorted(groups):
        m = ws == k
        if not m.any():
            continue
        reg, n = ndi.label(m)
        if n < 2:
            continue
        a = anchors.get(k)
        home = int(reg[a]) if a is not None and reg[a] else 0
        if not home:
            home = int(np.argmax(np.bincount(reg.ravel())[1:])) + 1
        ids = {}
        for i in range(1, n + 1):
            lab[reg == i] = nxt
            ids[i] = nxt
            nxt += 1
        pieces_of[k] = (ids[home], [ids[i] for i in range(1, n + 1)
                                    if i != home])
    if not pieces_of:
        return ws
    a, b, _s = _basin_saddles(lab, edt)
    near = {}
    for x, y in zip(np.asarray(a).tolist(), np.asarray(b).tolist()):
        near.setdefault(x, []).append(y)
        near.setdefault(y, []).append(x)
    taken = {}
    for k, (home, others) in pieces_of.items():
        parent = {home: None}
        todo = deque([home])
        want = set(others)
        while todo and want:
            u = todo.popleft()
            for v in sorted(near.get(u, ())):
                if v in parent:
                    continue
                anon = n_points < v < first and v not in taken
                if not anon and v not in want:
                    continue
                parent[v] = u
                want.discard(v)
                todo.append(v)
        for t in others:
            u = parent.get(t)
            while u is not None and u != home:
                if u < first:            # an anonymous basin, not a piece
                    taken[u] = k
                u = parent[u]
    out = np.array(ws, copy=True)
    if taken:
        lut = np.arange(int(ws.max()) + 1, dtype=out.dtype)
        for j, k in taken.items():
            lut[j] = k
        out = lut[out]
    return out


def _lost_seeds(pieces, later):
    """The later seeds of a marker none of whose marker voxels ended in
    their object (an object between them kept the way): they add nothing,
    and the search is made again as if they had never been placed.
    ``later`` -- ``(key, label, voxels)`` per later seed: its marker's
    voxels (``(n, 3)``, in ``pieces``' coordinates) and the label of its
    object in ``pieces``; returns the keys of the lost ones."""
    return {k for k, g, vox in later
            if not (pieces[tuple(np.asarray(vox).T)] == g).any()}


def _split_at_waists(edt, sm, comp, markers, claimed, clicked, mask,
                     n_points, spacing, n, half, cancel_event=None,
                     extra=None):
    """The split on the pieces of ``claimed`` (``mask`` holds exactly
    them): the peaks of ``sm`` (one per box of ``half`` per axis), thinned
    to ``n`` xy voxels; the hills; one flood of the raw distance map from
    the markers, their gained paths and the free peaks (anonymous seeds
    ``n_points + 1, ...``); the anonymous basins merged into the markers'
    groups; the one-piece guard; then every anonymous id zeroed."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    s = np.asarray(spacing, dtype=np.float64)
    lab = np.where(mask, comp, 0)
    footprint = np.ones(tuple(2 * h + 1 for h in half), dtype=bool)
    peaks = np.asarray(peak_local_max(sm, footprint=footprint, min_distance=1,
                                      exclude_border=False, labels=lab),
                       dtype=np.int64).reshape(-1, 3)
    if len(peaks):
        # highest first; a tie keeps the order skimage gave
        peaks = peaks[_peak_order(sm[tuple(peaks.T)])]
        kept = _thin_peaks(peaks * (s / s[-1]), lab[tuple(peaks.T)], n,
                           cancel_event)
        peaks = peaks[kept]
    found, anchors, mtop, free = _hills_with_peaks(
        sm, edt, comp, markers, claimed, clicked, peaks, half, n_points,
        cancel_event, extra=extra)
    seeds = np.where(mask, markers, 0).astype(np.int32)
    gained = {}
    for k, vox in found.items():
        new = vox[seeds[tuple(vox.T)] == 0]
        if len(new):
            seeds[tuple(new.T)] = k
            gained[k] = new
    anon = {}
    nid = n_points + 1
    for p, top in free:
        if seeds[p] == 0:
            seeds[p] = nid
            anon[nid] = top
            nid += 1
    ws = np.asarray(watershed(-edt, seeds, mask=mask), dtype=np.int32)
    _poll_stop(cancel_event)
    if anon:
        tops = {k: float(edt[t]) for k, t in mtop.items()}
        owner = _grow_groups(_basin_saddles(ws, edt), tops, anon,
                             cancel_event)
        if owner:
            lut = np.arange(int(ws.max()) + 1, dtype=np.int32)
            for j, k in owner.items():
                lut[j] = k
            ws = lut[ws]
    if extra:
        # a marker of several seeds takes the basins between its seeds
        ws = _join_group_pieces(ws, edt, anchors, set(extra), n_points)
    keep = set(range(1, n_points + 1))
    split, _regions = _split_ids(ws)
    if any(k in keep for k in split):
        ws = _guard_markers(edt, ws, mask, gained, anchors, keep,
                            cancel_event)
    ws[ws > n_points] = 0
    return ws


def _split_clicked_pieces(fg, comp, markers, claimed, clicked, n_points,
                          spacing, split, n, half, cancel_event=None,
                          extra=None):
    """The objects of every claimed click, for ``split`` ``"single"`` or
    ``"always"``.  ``"always"``: the split on every clicked piece.
    ``"single"``: a piece holding exactly one claimed click takes the
    split's result, any other the markers' own flood (as with ``"off"``).
    Both are run over ALL the clicked pieces, exactly as the two modes run
    them: the flood's queue breaks ties of equal height and age by its own
    layout, which the other pieces' seeds change, so a flood over fewer
    pieces could differ by a voxel.  The work runs on the clicked pieces'
    box grown by :func:`_marker_crop_margin` (the output equals the whole
    box's) and is pasted into a zero box."""
    from scipy import ndimage as ndi

    used = sorted(set(claimed.values()))
    if split == "always":
        rule = set(used)
    else:
        per = {}
        for c in claimed.values():
            per[c] = per.get(c, 0) + 1
        rule = {c for c in used if per[c] == 1}
    shape = comp.shape
    boxes = ndi.find_objects(comp, max_label=used[-1])
    margin = _marker_crop_margin(spacing)
    sl = tuple(
        slice(max(0, min(boxes[c - 1][a].start for c in used) - margin[a]),
              min(shape[a], max(boxes[c - 1][a].stop for c in used)
                  + margin[a]))
        for a in range(3))
    origin = np.asarray([v.start for v in sl])
    comp_c = comp[sl]
    markers_c = markers[sl]
    clicked_c = {k: tuple(int(v) for v in np.asarray(p) - origin)
                 for k, p in clicked.items()}
    extra_c = {k: [(None if own is None
                    else tuple(int(v) for v in np.asarray(own) - origin),
                    np.asarray(vox) - origin) for own, vox in seeds]
               for k, seeds in (extra or {}).items()} or None
    mask = fg[sl] & np.isin(comp_c, used)
    edt = ndi.distance_transform_edt(mask, sampling=spacing)
    _poll_stop(cancel_event)
    cut = flood = None
    if rule:
        # one xy voxel of smoothing: the same width in microns on each axis
        sm = ndi.gaussian_filter(
            edt, sigma=tuple(spacing[-1] / v for v in spacing),
            truncate=MARKER_GAUSS_TRUNCATE)
        _poll_stop(cancel_event)
        cut = _split_at_waists(edt, sm, comp_c, markers_c, claimed,
                               clicked_c, mask, n_points, spacing, n, half,
                               cancel_event, extra=extra_c)
    if len(rule) < len(used):
        # the markers' own flood, as with "off"
        seeds = np.array(markers_c, dtype=np.int32, copy=True)
        found, anchors = _gain_tops(
            edt, comp_c, seeds, claimed,
            ndi.find_objects(seeds, max_label=n_points), clicked_c,
            cancel_event, extra=extra_c)
        gained = {}
        for k, vox in found.items():
            new = vox[seeds[tuple(vox.T)] == 0]
            if len(new):
                seeds[tuple(new.T)] = k
                gained[k] = new
        _poll_stop(cancel_event)
        flood = _guard_markers(edt, seeds, mask, gained, anchors,
                               set(range(1, n_points + 1)), cancel_event)
    if flood is None:
        out = cut
    elif cut is None:
        out = flood
    else:
        out = np.where(np.isin(comp_c, sorted(rule)), cut, flood)
    pieces = np.zeros(shape, dtype=np.int32)
    pieces[sl] = out
    return pieces


def _marker_foreground(image, existing, *, clahe, clahe_clip, threshold):
    """``(empty, fg)`` for the markers' search: the voxels no kept object
    holds, and the bright ones among them.  The threshold is computed as
    :func:`add_segment_region` computes it (the same Local contrast space,
    the same flat-box guard); a flat box has no bright voxel."""
    empty = existing == 0
    proc = add_threshold_space(image, clahe=clahe, clahe_clip=clahe_clip)
    thr = None
    if image.size and not (clahe and not np.any(proc)):
        thr = _resolve_threshold(proc, threshold)
    if thr is None:
        fg = np.zeros(existing.shape, dtype=bool)
    else:
        fg = (proc >= thr) & empty
    return empty, fg


def _claim_piece(ball, sl, own, empty, fg, comp):
    """``(piece, why)``: the bright piece one click claims, from its ball
    (``ball``, boolean over the box ``sl``) -- or ``(0, "kept")`` when the
    ball lies wholly on kept objects, ``(0, "dim")`` when wholly off the
    stain.  ``own`` is the click's own voxel when it lies in the box and in
    its own ball (else ``None``): on the stain it names the piece;
    otherwise the piece holding most of the ball wins (a tie goes to the
    lower piece number)."""
    if not (ball & empty[sl]).any():
        return 0, "kept"
    on = ball & fg[sl]
    if not on.any():
        return 0, "dim"
    c = 0
    if own is not None and fg[own]:
        c = int(comp[own])
    if not c:
        vals, counts = np.unique(comp[sl][on], return_counts=True)
        c = int(vals[int(np.argmax(counts))])
    return c, None


def segment_at_markers(image_roi, existing_roi, points, *, radius_um,
                       spacing, clahe=False, clahe_clip=0.01,
                       threshold="otsu", min_size=0, cancel_event=None,
                       split="off", peak_spacing_um=None, groups=None,
                       _skip=(),
                       ) -> tuple:
    """One new object per click, from the bright pieces the clicks sit on.

    The threshold is computed exactly as :func:`add_segment_region` computes
    it (the same Local contrast space, the same flat-box guard), so it means
    the same voxels; stain is what passes it in empty space.

    Each click is a small ball of ``radius_um`` (the engine's balls: an
    earlier ball keeps a shared voxel).  A click claims ONE bright piece:
    the piece under its own voxel when that voxel is on stain, otherwise the
    piece holding most of its ball (a tie goes to the lower piece number).
    Its marker is its ball inside that piece.  A click makes nothing when
    its ball lies wholly on kept objects (``"kept"``) or wholly off stain
    (``"dim"``); a near miss whose ball still touches a nucleus claims it.

    Only clicked pieces are split: a distance watershed over them, seeded
    by the markers.  Before it, in a piece holding two markers or more,
    each marker climbs the distance map from its click to the top of its
    hill and takes that climb -- but only when no other marker reaches the
    same flat top (see :func:`_gain_tops`).  So a click
    anywhere on one of two touching nuclei gets its whole nucleus (split at
    the neck).  Two clicks in one nucleus share its hill, so neither takes
    it: the nucleus is cut between the two balls, and unless the clicks sit
    symmetrically about its middle the cut is very uneven -- the click
    nearer the middle takes most of it and the other keeps a small cap
    around itself.  One click on two touching nuclei floods both: one
    object (with ``split="off"``, the default; see below).  Every object
    comes back as ONE 6-connected piece (a climb cut
    off from its click is given back and the flood re-run; an object whose
    own ball was cut keeps the piece holding its click, and the rest goes
    to the objects around it), and a clicked piece keeps no unowned voxel.
    Unclicked pieces come back 0.  Then objects under ``min_size`` voxels
    are dropped (``"small"``).

    ``points`` are box-local ZYX, the ticked markers in placement order.
    Returns ``(pieces, why)``: ``pieces`` int32, non-zero only in empty
    space, the k-th click's object numbered ``k``; ``why`` ``{k: "kept" |
    "dim" | "small"}`` for every click that made nothing.  Stop is polled
    between the steps, once per climb and per top while the hills are
    grouped, and after every watershed call (each call itself cannot
    stop); a Stop raises ``ReshapeCancelled``.

    ``split`` (one of :data:`MARKER_SPLIT_MODES`) also cuts a clicked piece
    at its waists, so one click on a nucleus in a touching group gives that
    nucleus, not its neighbours: ``"off"`` (the default) is the flood above,
    unchanged; ``"always"`` cuts every clicked piece; ``"single"`` cuts
    only a piece that holds exactly one claimed click and floods the others
    as ``"off"`` does.  The cut: the distance map smoothed by one xy voxel
    (the same width in microns on every axis); its peaks, one per box of
    ``peak_spacing_um`` (in microns; ``None`` is
    :data:`ADD_MARKER_SPACING_DEFAULT` xy voxels) and thinned to that
    spacing within each piece; every peak not on a marker's hill seeds an
    anonymous basin of one flood of the raw map; each marker then takes,
    best first, the anonymous basins whose ridge with its group is at least
    :data:`MARKER_SPLIT_SHARE` of the lower top -- two markers never join
    -- and the rest is left out.  Two clicks on one lone nucleus still give
    two objects.  "dim" and "kept" are decided before the cut, "small"
    after it.  Stop is also polled in the thinning, the merge and after
    each flood.

    ``groups`` (one key per point; ``None``: every point alone) makes one
    object of every set of points with an equal key: ONE marker -- the
    union of its seeds' balls -- built before the flood and the cut, so
    the seeds of one object never compete and nothing between them is
    left out, and each seed also climbs to its own hill.  The group's
    bright piece is the one its earliest seed that claims one sits on; a
    later seed claiming another piece is refused (``"apart"``) and the
    call runs as if it had not been placed; so is a later seed none of
    whose marker ended in its object (another object kept the way between
    them).  Nothing between two seeds of one object is left out: the
    basins between them join it (another object's never do).  The size floor counts the
    group's object.  With ``groups`` the call returns ``(pieces, why,
    seed_why)``: ``pieces`` numbered by group, ``1..G`` in the order each
    group first appears; ``why`` ``{g: "kept" | "dim" | "small"}`` for each
    group that made nothing (a group none of whose seeds claimed a piece
    takes its first seed's reason); ``seed_why`` ``{k: "kept" | "dim" |
    "apart"}`` for every point (1-based) that added nothing to its group.
    """
    if split not in MARKER_SPLIT_MODES:
        raise ValueError(f"split must be one of {MARKER_SPLIT_MODES}, "
                         f"got {split!r}")
    if groups is not None:
        groups = list(groups)
        if len(groups) != len(list(points or ())):
            raise ValueError("groups has {} keys for {} points".format(
                len(groups), len(list(points or ()))))
    from scipy import ndimage as ndi

    from .reshape.region import rasterise_points

    image = np.asarray(image_roi, dtype=np.float64)
    existing = np.asarray(existing_roi)
    if image.shape != existing.shape:
        raise ValueError(
            f"image_roi shape {image.shape} != existing_roi shape "
            f"{existing.shape}"
        )
    if image.ndim != 3:
        raise ValueError("markers need a 3D box")
    pts = [tuple(p) for p in (points or ())]
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    spacing = tuple(float(v) for v in spacing)
    n_vox, half = None, None             # "off" never reads the Spacing
    if split != "off":
        n_vox, half = _marker_half_widths(peak_spacing_um, spacing)
    pieces = np.zeros(existing.shape, dtype=np.int32)
    why = {}
    seed_why = {}
    if groups is None:
        gidx = list(range(1, len(pts) + 1))
    else:
        first = {}
        gidx = [first.setdefault(g, len(first) + 1) for g in groups]
    n_groups = max(gidx or [0])

    def _out(p, w):
        return (p, w) if groups is None else (p, w, seed_why)

    if not pts:
        return _out(pieces, why)
    _poll_stop(cancel_event)

    # --- 1. the threshold, as Anywhere computes it ---
    empty, fg = _marker_foreground(image, existing, clahe=clahe,
                                   clahe_clip=clahe_clip,
                                   threshold=threshold)

    # --- 2. the bright pieces (6-connected: what the watershed can flood) ---
    comp, _n = ndi.label(fg)
    _poll_stop(cancel_event)

    # --- 3 + 4. the balls, and which piece each click claims ---
    def _claims(skip):
        """Each placed point's ball and claim, the points in ``skip`` left
        out (as if never placed): ``(balls, slices, claims, own, why)``,
        balls numbered by point."""
        live = [k for k in range(1, len(pts) + 1) if k not in skip]
        b = rasterise_points([pts[k - 1] for k in live], existing.shape,
                             radius_um=radius_um, spacing=spacing)
        if len(live) < len(pts):
            lut = np.zeros(len(live) + 1, np.int32)
            lut[1:] = live
            b = lut[b]
        s = ndi.find_objects(b, max_label=len(pts))
        got, mine, w = {}, {}, {}
        for k in live:
            sl = s[k - 1]
            if sl is None:
                w[k] = "kept"            # every voxel taken by earlier balls
                continue
            ball = b[sl] == k
            own = tuple(int(round(float(v))) for v in pts[k - 1][-3:])
            if not (all(0 <= own[a] < existing.shape[a] for a in range(3))
                    and b[own] == k):
                own = None
            c, reason = _claim_piece(ball, sl, own, empty, fg, comp)
            if reason is not None:
                w[k] = reason
                continue
            if own is not None and fg[own]:
                mine[k] = own
            got[k] = c
        return b, s, got, mine, w

    apart = set(int(k) for k in _skip)
    while True:
        balls, slices, got, mine, pwhy = _claims(apart)
        # a group's piece is its earliest claiming seed's; a later seed on
        # another piece is refused, and the claims are made again without
        # it (its ball may have kept a voxel a later ball needs)
        head = {}
        new = set()
        for k in sorted(got):
            g = gidx[k - 1]
            if g not in head:
                head[g] = k
            elif got[k] != got[head[g]]:
                new.add(k)
        if not new:
            break
        apart |= new
    for k in apart:
        seed_why[k] = "apart"
    markers = np.zeros(existing.shape, dtype=np.int32)
    claimed = {}
    clicked = {}     # g -> the group's first seed's own voxel, when it
    extra = {}       # claimed by it; g -> the later seeds [(own, voxels)]
    later = []       # the later seeds as (k, g, their marker voxels)
    for k in sorted(got):
        g = gidx[k - 1]
        c = got[head[g]]
        sl = slices[k - 1]
        mark = (balls[sl] == k) & (comp[sl] == c)
        markers[sl][mark] = g
        if head[g] == k:
            claimed[g] = c
            if k in mine:
                clicked[g] = mine[k]
        else:
            vox = np.argwhere(mark) + np.asarray([v.start for v in sl])
            own = mine.get(k)
            extra.setdefault(g, []).append(
                (own if own is not None and comp[own] == c else None, vox))
            later.append((k, g, vox))

    def _again(lost):
        return segment_at_markers(
            image_roi, existing_roi, points, radius_um=radius_um,
            spacing=spacing, clahe=clahe, clahe_clip=clahe_clip,
            threshold=threshold, min_size=min_size,
            cancel_event=cancel_event, split=split,
            peak_spacing_um=peak_spacing_um, groups=groups,
            _skip=apart | lost)
    if groups is None:
        why.update(pwhy)
    else:
        for k, w in pwhy.items():
            seed_why[k] = w
        for k in range(len(pts), 0, -1):
            g = gidx[k - 1]
            if g not in claimed and k in pwhy:
                why[g] = pwhy[k]          # the first seed's reason wins
    if not claimed:
        return _out(pieces, why)
    if groups is not None:
        # the group's marker is the union of its seeds' balls
        slices = ndi.find_objects(markers, max_label=n_groups)

    if split != "off":
        # --- the cut at the waists (steps 5-6b replaced), then step 7 ---
        pieces = _split_clicked_pieces(fg, comp, markers, claimed, clicked,
                                       n_groups, spacing, split, n_vox,
                                       half, cancel_event,
                                       extra=extra or None)
        lost = _lost_seeds(pieces, later)
        if lost:
            return _again(lost)
        sizes = np.bincount(pieces.ravel(), minlength=n_groups + 1)
        for k in claimed:
            if min_size and int(sizes[k]) < int(min_size):
                pieces[pieces == k] = 0
                why[k] = "small"
        return _out(pieces, why)

    # --- 5. the distance map over the clicked pieces only ---
    mask = fg & np.isin(comp, sorted(set(claimed.values())))
    edt = ndi.distance_transform_edt(mask, sampling=spacing)
    _poll_stop(cancel_event)

    # --- 6. each marker takes its hill's top; a shared hill stays free ---
    gained = {}
    found, anchors = _gain_tops(edt, comp, markers, claimed, slices,
                                clicked, cancel_event, extra=extra or None)
    for k, vox in found.items():
        free = vox[markers[tuple(vox.T)] == 0]
        if len(free):
            markers[tuple(free.T)] = k
            gained[k] = free
    _poll_stop(cancel_event)
    from skimage.segmentation import watershed

    # --- 6b. every object is one piece: a gain cut off from its click is
    # given back and the flood re-run; an object whose own ball was cut
    # keeps the piece holding its click, and the rest is flooded by the
    # neighbours around it ---
    for _round in range(len(gained) + 1):
        pieces = np.asarray(watershed(-edt, markers, mask=mask),
                            dtype=np.int32)
        _poll_stop(cancel_event)
        cut_ids, regions = _split_ids(pieces)
        if not cut_ids:
            break
        back = [k for k in cut_ids if k in gained
                and (regions[tuple(gained[k].T)]
                     != regions[anchors[k]]).any()]
        if back:
            for k in back:
                markers[tuple(gained.pop(k).T)] = 0
            continue
        for k, rs in cut_ids.items():
            if k in anchors:
                keep = int(regions[anchors[k]])
            else:
                # no climb (cannot happen: a lone marker floods its whole
                # piece): the piece holding most of its ball
                held = regions[markers == k]
                keep = max(rs, key=lambda r: int((held == r).sum()))
            pieces[np.isin(regions, [r for r in rs if r != keep])] = 0
        pieces = np.asarray(watershed(-edt, pieces, mask=mask),
                            dtype=np.int32)
        _poll_stop(cancel_event)
        break

    lost = _lost_seeds(pieces, later)
    if lost:
        return _again(lost)
    # --- 7. Smallest object ---
    sizes = np.bincount(pieces.ravel(), minlength=n_groups + 1)
    for k in claimed:
        if min_size and int(sizes[k]) < int(min_size):
            pieces[pieces == k] = 0
            why[k] = "small"
    return _out(pieces, why)


#: the six box faces, as ``open_faces`` names them (low / high end per axis)
BOX_FACES = ("z0", "z1", "y0", "y1", "x0", "x1")
#: ids up to this are compacted with one lookup table, larger ones by sorting
_EDGE_LUT_MAX_ID = 1 << 24


def _compact_ids(labels):
    """``(compact, ids)``: ``compact`` numbers the non-zero ids ``1..K`` in
    ascending order of ``ids`` (the original ids, ascending).  A lookup table
    when every id is small and non-negative, else a sorted search -- an id of
    2**31 - 1 must not make a table of that size."""
    lab = np.asarray(labels)
    if lab.size == 0:
        return np.zeros(lab.shape, np.int32), np.zeros(0, lab.dtype)
    lo = lab.min()
    hi = lab.max()
    if lo >= 0 and int(hi) <= _EDGE_LUT_MAX_ID:
        present = np.flatnonzero(np.bincount(
            lab.ravel().astype(np.int64, copy=False)))
        present = present[present > 0]
        lut = np.zeros(int(hi) + 1, np.int32)
        lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
        return lut[lab], present.astype(lab.dtype)
    vals = np.unique(lab)
    nz = vals[vals != 0]
    compact = (np.searchsorted(nz, lab) + 1).astype(np.int32)
    compact[lab == 0] = 0
    return compact, nz


def edge_flag_ids(labels, margin: int, *, open_faces=()) -> set:
    """IDs of pieces reaching the box EDGE ZONE -- within *margin* voxels of
    any of the 6 walls -- i.e. objects the box likely TRUNCATED.

    Flags on wall PROXIMITY (any voxel in the boundary shell), not centroid.
    A LONE piece is never flagged (a single tightly-boxed object stays
    ticked; grow the box if it is clipped).

    ``open_faces`` names faces (of :data:`BOX_FACES`) that cut nothing --
    on a whole patch every face is the edge of the data, not a cut by the
    box -- so a piece near only those faces is not flagged.

    One bounding-box pass over compacted ids: a piece reaches the zone of a
    face exactly when its bounding box starts within *margin* of it, so the
    answer is the per-piece mask test's, without a mask per piece.
    """
    if getattr(labels, "ndim", 0) != 3:
        return set()
    open_set = set(open_faces or ())
    unknown = open_set - set(BOX_FACES)
    if unknown:
        raise ValueError("unknown box faces: {}".format(sorted(unknown)))
    from scipy.ndimage import find_objects

    compact, ids = _compact_ids(labels)
    if ids.size <= 1:
        return set()  # single boxed object stays checked
    m = max(1, int(margin))
    shape = tuple(int(n) for n in np.asarray(labels).shape)
    flagged = set()
    for k, slices in enumerate(find_objects(compact), start=1):
        if slices is None:
            continue
        for axis, sl in enumerate(slices):
            low, high = BOX_FACES[2 * axis], BOX_FACES[2 * axis + 1]
            if ((low not in open_set and sl.start < m)
                    or (high not in open_set and sl.stop > shape[axis] - m)):
                flagged.add(int(ids[k - 1]))
                break
    return flagged


def smooth_pieces(result, sigma: float = 1.5, *, cancel_event=None):
    """Isotropic SDF surface-smooth every piece in *result*.

    Routed through ``run_batch_recipe`` -- the SAME backend as Batch Refine
    -- with ``empty="erase"``: a piece its smoothing empties is dropped, as
    it always was.  Moved verbatim from
    ``Curation3DAnnotator._smooth_add_result`` (Add rebuild).

    The pieces are smoothed on :func:`smooth_worker_count` threads.  Each
    piece's compute reads only the untouched *result*; the writes happen
    after every compute, in ascending id order (a lower id keeps a voxel two
    smoothed pieces both reach), so the output is the one-thread output
    whichever thread finishes first.

    ``cancel_event`` (a ``threading.Event``, the Preview's Stop) is checked
    as each piece finishes; once it is set no queued piece starts (the ones
    already running finish and are dropped) and ``ReshapeCancelled`` is
    raised, which the Preview answers as ``cancelled``.  ``None``: runs to
    the end, as before.  A piece that raises hands back *result*
    unsmoothed, as on one thread.
    """
    from .batch_refine import RefineStep, run_batch_recipe
    from .reshape.engine import ReshapeCancelled

    piece_ids = [int(v) for v in np.unique(result) if v != 0]
    if not piece_ids:
        return result
    recipe = [
        RefineStep(
            method="sdf",
            params={"sigma_z": float(sigma), "sigma_xy": float(sigma)},
        )
    ]
    try:
        res = run_batch_recipe(
            result, piece_ids, recipe, n_workers=smooth_worker_count(),
            empty="erase",
            cancel_event=cancel_event,
        )
    except Exception as exc:
        print(f"[ADD] smooth failed: {exc!r}")
        return result
    if getattr(res, "cancelled", False):
        raise ReshapeCancelled()
    return res.working_volume
