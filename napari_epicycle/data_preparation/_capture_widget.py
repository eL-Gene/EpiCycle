"""Step 1 ④ Review & Library — "patch capture" inspector dialog.

A standalone documentation / figure tool.  The user picks a saved patch and
gets a big GPU-rendered orthogonal view of it (three synced planes:
XY / XZ / YZ) with a click-to-move synced crosshair.

**Brick 1 of the vispy rewrite**: this replaces the old
matplotlib CPU rasteriser with a single standalone vispy ``SceneCanvas`` holding
a 2x2 grid of view-boxes — XY (top-left), YZ (top-right), XZ (bottom-left) and a
placeholder quadrant (bottom-right) where the 3-D volume lands in Brick 3.  Only
Point mode is built here; Scrub mode, the 3-D volume, the
render-mode combo and Save-figure are LATER bricks.

This module is deliberately self-contained: it imports only numpy, tifffile,
qtpy and vispy.  It does NOT import from any other ``napari_epicycle`` module so
it can be wired in from Step 1's UI without dragging in cross-step dependencies.

Pure, unit-tested, Qt-free / GL-free helpers live at the top (this is where the
interaction correctness is tested):

* :func:`volume_channel_limits` — per-channel ``(lo, hi)`` contrast bounds over
  the WHOLE volume, computed ONCE at load.
* :func:`compose_capture_rgb` — a PURE reduce-to-RGB function.
* :func:`plane_reslice` — the ground-truth reslice contract.
* :func:`point_click_to_cursor` — the click → cursor mapping (per panel).

The Qt / vispy shell (:class:`PatchCaptureDialog` + :func:`launch_patch_capture`)
is a floating, non-modal ortho inspector.  Every GL / vispy construction is
guarded so a bad file / render / driver can never crash the host napari process;
failures degrade to an in-dialog label instead.
"""

from __future__ import annotations

import os

import numpy as np


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _diag(msg):
    """Best-effort ``[CAPTURE]`` diagnostic print that can never raise.

    A non-ASCII character on a cp1252 Windows console must not crash the app,
    so we fall back to an ASCII-replaced print, then to silence.
    """
    try:
        print(f"[CAPTURE] {msg}")
    except Exception:
        try:
            print(str(msg).encode("ascii", "replace").decode("ascii"))
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Pure helpers (this is what the tests target)
# --------------------------------------------------------------------------- #
def _promote_to_4d(data):
    """Promote ``data`` to a ``(C, Z, Y, X)`` view.

    * ``(Y, X)``       -> ``(1, 1, Y, X)``
    * ``(Z, Y, X)``    -> ``(1, Z, Y, X)``
    * ``(C, Z, Y, X)`` -> unchanged
    * ``ndim > 4``     -> leading dims collapsed into the channel axis
    * ``ndim < 2``     -> ``None`` (nothing showable)
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr[None, None, ...]
    if arr.ndim == 3:
        return arr[None, ...]
    if arr.ndim == 4:
        return arr
    if arr.ndim > 4:
        return arr.reshape((-1,) + arr.shape[-3:])
    return None


def volume_channel_limits(data, percentile=(1.0, 99.0)):
    """Per-channel ``(lo, hi)`` contrast bounds over the ENTIRE volume.

    Computed ONCE (all Z), so every rendered slice can be normalized against
    the same fixed scale.  On a low-signal Z slice (noise / background beyond
    the tissue) this makes it render DARK — WYSIWYG — instead of stretching
    that slice's noise to full brightness (the "saturated/noisy Z-ends" bug).

    Parameters
    ----------
    data : np.ndarray
        ``(C, Z, Y, X)`` OR ``(Z, Y, X)`` (promoted to ``C=1``) OR ``(Y, X)``.
    percentile : (float, float)
        Low/high percentiles used for each channel's bounds.

    Returns
    -------
    list[tuple[float, float]]
        One ``(lo, hi)`` per channel.  Guard: when ``hi <= lo`` (flat /
        constant channel) returns ``(lo, lo + 1)`` so downstream normalization
        never divides by zero.  Returns ``[]`` for un-showable input.
    """
    try:
        arr = _promote_to_4d(data)
        if arr is None:
            return []
        C = int(arr.shape[0])
        try:
            plo, phi = float(percentile[0]), float(percentile[1])
        except Exception:
            plo, phi = 1.0, 99.0

        limits = []
        for c in range(C):
            try:
                chan = np.asarray(arr[c], dtype=np.float64)
                if chan.size == 0:
                    limits.append((0.0, 1.0))
                    continue
                lo = float(np.percentile(chan, plo))
                hi = float(np.percentile(chan, phi))
                if not np.isfinite(lo):
                    lo = 0.0
                # Guard hi <= lo (constant / flat channel) -> (lo, lo + 1).
                if not np.isfinite(hi) or hi <= lo:
                    hi = lo + 1.0
                limits.append((lo, hi))
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"channel-limits failed c={c} ({exc})")
                limits.append((0.0, 1.0))
        return limits
    except Exception as exc:  # pragma: no cover - defensive
        _diag(f"volume_channel_limits failed ({exc})")
        return []


def _channel_is_on(channels_on, c):
    """Whether channel ``c`` should contribute.  Defaults to True when the
    caller's list is missing / shorter than the channel count."""
    if channels_on is None:
        return True
    try:
        if c < len(channels_on):
            return bool(channels_on[c])
    except Exception:
        return True
    # Index beyond the provided list -> default on.
    return True


def _normalize_plane(plane, plo, phi):
    """Percentile-clip a 2D plane to the float range [0, 1] (PER-SLICE).

    Divide-by-zero guard: when the high percentile is <= the low percentile
    (flat / constant plane) return zeros rather than NaNs/inf.
    """
    try:
        arr = np.asarray(plane, dtype=np.float64)
        if arr.size == 0:
            return np.zeros(arr.shape, dtype=np.float64)
        lo = float(np.percentile(arr, plo))
        hi = float(np.percentile(arr, phi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros(arr.shape, dtype=np.float64)
        out = (arr - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)
    except Exception as exc:  # pragma: no cover - defensive
        _diag(f"normalize failed ({exc}); returning zeros")
        try:
            return np.zeros(np.asarray(plane).shape, dtype=np.float64)
        except Exception:
            return np.zeros((1, 1), dtype=np.float64)


def _normalize_plane_fixed(plane, lo, hi):
    """Clip a 2D plane to [0, 1] against FIXED ``(lo, hi)`` bounds.

    Used for volume-global normalization: the same per-channel bounds are
    applied to every slice, so a low-signal slice renders dark instead of
    being percentile-stretched.  Same divide-by-zero guard as
    :func:`_normalize_plane`.
    """
    try:
        arr = np.asarray(plane, dtype=np.float64)
        if arr.size == 0:
            return np.zeros(arr.shape, dtype=np.float64)
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros(arr.shape, dtype=np.float64)
        out = (arr - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)
    except Exception as exc:  # pragma: no cover - defensive
        _diag(f"normalize-fixed failed ({exc}); returning zeros")
        try:
            return np.zeros(np.asarray(plane).shape, dtype=np.float64)
        except Exception:
            return np.zeros((1, 1), dtype=np.float64)


def _norm_channel_plane(plane, c, limits, plo, phi):
    """Normalize one channel plane, preferring fixed volume-global limits.

    When ``limits`` provides a usable ``(lo, hi)`` for channel ``c`` we clip
    against those fixed bounds (WYSIWYG global contrast); otherwise we fall
    back to the legacy per-slice percentile stretch.
    """
    if limits is not None:
        try:
            lo, hi = limits[c]
            return _normalize_plane_fixed(plane, lo, hi)
        except Exception:
            pass  # fall through to per-slice
    return _normalize_plane(plane, plo, phi)


def compose_capture_rgb(data, z_index, mip, channels_on, limits=None,
                        percentile=(1.0, 99.0)):
    """Reduce a patch volume to an (H, W, 3) uint8 RGB image for display/save.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(C, Z, Y, X)`` OR ``(Z, Y, X)`` single-channel (promoted to
        ``C=1``).  A bare ``(Y, X)`` is also tolerated (single Z, single
        channel).
    z_index : int
        Z slice to show when ``mip`` is False.  Clamped into range.
    mip : bool
        If True, max-intensity-project over Z; else take ``data[:, z_index]``.
    channels_on : list[bool]
        Length ``C`` — a False channel contributes nothing.
    limits : list[tuple[float, float]] | None
        Optional per-channel ``(lo, hi)`` bounds (see
        :func:`volume_channel_limits`).  When provided, each channel is clipped
        against its FIXED bounds (volume-global, WYSIWYG contrast — low-signal
        slices render dark).  When ``None`` (default), the legacy per-slice
        percentile stretch is used (back-compat).
    percentile : (float, float)
        Low/high percentiles for the per-slice fallback path.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8.  Never raises for valid ndarray input.

    Channel mapping: 0->R, 1->G, 2->B (channels beyond 3 are ignored).  A
    single-channel patch is replicated across R/G/B (grayscale) so it is
    visible rather than red-only.
    """
    try:
        arr = np.asarray(data)

        # -- Promote to (C, Z, Y, X) ------------------------------------- #
        if arr.ndim == 2:            # (Y, X) -> (1, 1, Y, X)
            arr = arr[None, None, ...]
        elif arr.ndim == 3:          # (Z, Y, X) -> (1, Z, Y, X)
            arr = arr[None, ...]
        elif arr.ndim == 4:          # (C, Z, Y, X)
            pass
        elif arr.ndim > 4:
            # Collapse any leading extra dims into the channel axis best-effort.
            arr = arr.reshape((-1,) + arr.shape[-3:])
        else:  # ndim < 2 -> nothing to show
            return np.zeros((1, 1, 3), dtype=np.uint8)

        C, Z, Y, X = arr.shape
        if Y <= 0 or X <= 0:
            return np.zeros((max(Y, 1), max(X, 1), 3), dtype=np.uint8)

        out = np.zeros((Y, X, 3), dtype=np.float64)

        if Z <= 0:
            return out.astype(np.uint8)

        # -- Reduce over Z ----------------------------------------------- #
        if mip:
            planes = np.max(arr, axis=1)          # (C, Y, X)
        else:
            z = int(np.clip(int(z_index), 0, Z - 1))
            planes = arr[:, z]                    # (C, Y, X)

        try:
            plo, phi = float(percentile[0]), float(percentile[1])
        except Exception:
            plo, phi = 1.0, 99.0

        # -- Map channels to RGB ----------------------------------------- #
        if C == 1:
            # Single-channel patch -> grayscale (replicate across R/G/B) so
            # it renders as visible gray, not red-only.
            if _channel_is_on(channels_on, 0):
                norm = _norm_channel_plane(planes[0], 0, limits, plo, phi)
                out[..., 0] = norm
                out[..., 1] = norm
                out[..., 2] = norm
        else:
            for c in range(min(C, 3)):
                if _channel_is_on(channels_on, c):
                    out[..., c] = _norm_channel_plane(
                        planes[c], c, limits, plo, phi)
            # Channels beyond index 2 are ignored by design.

        return (out * 255.0).astype(np.uint8)
    except Exception as exc:  # pragma: no cover - defensive
        _diag(f"compose failed ({exc}); returning 1x1 black")
        return np.zeros((1, 1, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Ground-truth reslice + click mapping (pure, GL-free — TESTED)
# --------------------------------------------------------------------------- #
_PANELS = ("XY", "XZ", "YZ")


def panels_to_reslice(prev_cursor, new_cursor):
    """Which ortho panels need a fresh slice re-fetch when the cursor moves from
    ``prev_cursor`` to ``new_cursor`` (CW2 per-panel redraw).

    A panel's IMAGE depends only on its OWN slice axis — XY on z, XZ on y, YZ on
    x (the same mapping as :func:`plane_reslice` / ``_panel_slice_index``).  So a
    wheel-step over XY (z only) leaves the XZ and YZ *images* unchanged; only
    their crosshair moves (handled cheaply, always).  Returns the subset of
    :data:`_PANELS` whose slice index changed.  ``prev_cursor is None`` (initial
    load, a channel / contrast / mode change, or any forced full redraw) returns
    ALL panels.
    """
    if prev_cursor is None:
        return set(_PANELS)
    pz, py, px = prev_cursor
    nz, ny, nx = new_cursor
    changed = set()
    if nz != pz:
        changed.add("XY")
    if ny != py:
        changed.add("XZ")
    if nx != px:
        changed.add("YZ")
    return changed


def plane_reslice(vol, panel, index):
    """Reslice a ``(C, Z, Y, X)`` volume for an ortho ``panel`` -> ``(C, H, W)``.

    The single source of truth for the ortho orientation — every
    display path composes what this returns, so a wrong reslice can never
    silently diverge from a correct one:

    ======  ========================  =============  =====  =====  ==========
    panel   reslice                   image (C,H,W)  rows   cols   slice axis
    ======  ========================  =============  =====  =====  ==========
    XY      ``vol[:, index]``            (C, Y, X)   Y      X      Z
    XZ      ``vol[:, :, index, :]``      (C, Z, X)   Z      X      Y
    YZ      ``vol[:,:,:,index].T(Y,Z)``  (C, Y, Z)   Y      Z      X
    ======  ========================  =============  =====  =====  ==========

    ``H`` is the ROW axis (**Z for XZ, Y for XY/YZ**).  YZ uses the standard
    sagittal convention — **Y vertical (rows), Z horizontal (cols)** — so it
    reads distinctly from XZ; the raw ``(C, Z, Y)`` slice is transposed to
    ``(C, Y, Z)`` for that (XY/XZ are never transposed — axes-HIGH-1).  A 3-D
    ``(Z, Y, X)`` input is promoted to ``(1, Z, Y, X)``; a bare ``(Y, X)`` to
    ``(1, 1, Y, X)``.  ``index`` is clamped into range so an out-of-range slice
    request degrades gracefully.

    Raises
    ------
    ValueError
        If ``vol`` cannot be promoted to 4-D, or ``panel`` is not one of
        ``{"XY", "XZ", "YZ"}``.
    """
    arr = _promote_to_4d(vol)
    if arr is None:
        raise ValueError(f"plane_reslice: un-showable input ndim={np.asarray(vol).ndim}")
    C, Z, Y, X = arr.shape
    p = str(panel).upper()
    if p == "XY":
        i = int(np.clip(int(index), 0, max(Z - 1, 0)))
        return arr[:, i]            # (C, Y, X) — rows=Y, cols=X
    if p == "XZ":
        i = int(np.clip(int(index), 0, max(Y - 1, 0)))
        return arr[:, :, i, :]      # (C, Z, X) — rows=Z, cols=X
    if p == "YZ":
        i = int(np.clip(int(index), 0, max(X - 1, 0)))
        # Sagittal: rows=Y, cols=Z.  The raw slice is (C, Z, Y); transpose the
        # last two axes -> (C, Y, Z) so Y is vertical and Z horizontal.
        return arr[:, :, :, i].transpose(0, 2, 1)   # (C, Y, Z) — rows=Y, cols=Z
    raise ValueError(f"plane_reslice: unknown panel {panel!r}")


def point_click_to_cursor(panel, row, col, shape, cursor):
    """Map a Point-mode click at data ``(row, col)`` in ``panel`` -> new cursor.

    ``shape`` is ``(Z, Y, X)``; ``cursor`` is ``[z, y, x]``.  Per the design
    (the inverse of the table — ``row`` is the vertical/screen-down data
    axis, ``col`` the horizontal):

    * **XY** — ``col -> x``, ``row -> y`` (rows=Y, cols=X); ``z`` unchanged.
    * **XZ** — ``col -> x``, ``row -> z`` (rows=Z, cols=X); ``y`` unchanged.
    * **YZ** — ``col -> z``, ``row -> y`` (rows=Y, cols=Z, sagittal); ``x``
      unchanged.

    The panel's own slice axis is left untouched (you clicked *within* a slice,
    not through it).  All coordinates are clamped into ``[0, dim - 1]``.
    Returns a fresh ``[z, y, x]`` list.

    Raises
    ------
    ValueError
        If ``panel`` is not one of ``{"XY", "XZ", "YZ"}``.
    """
    Z, Y, X = int(shape[0]), int(shape[1]), int(shape[2])
    z, y, x = int(cursor[0]), int(cursor[1]), int(cursor[2])
    r = int(round(float(row)))
    c = int(round(float(col)))
    p = str(panel).upper()
    if p == "XY":
        x = int(np.clip(c, 0, max(X - 1, 0)))
        y = int(np.clip(r, 0, max(Y - 1, 0)))
    elif p == "XZ":
        x = int(np.clip(c, 0, max(X - 1, 0)))
        z = int(np.clip(r, 0, max(Z - 1, 0)))
    elif p == "YZ":
        z = int(np.clip(c, 0, max(Z - 1, 0)))   # col -> z (sagittal)
        y = int(np.clip(r, 0, max(Y - 1, 0)))   # row -> y
    else:
        raise ValueError(f"point_click_to_cursor: unknown panel {panel!r}")
    return [z, y, x]


def scrub_delta_to_cursor(panel, d_data, cursor, shape):
    """Map a Scrub-mode drag delta ``d_data`` in ``panel`` -> new cursor.

    CONVENTION (pin this — a test written in pixel-space would bake a
    compensating sign and hide a Z double-flip): ``d_data = (dx, dy)`` is the
    drag delta ALREADY in **flipped data-space**, i.e.

    * ``dx`` = change along the panel's COLUMN data-axis, and
    * ``dy`` = change along the panel's ROW data-axis,

    where **screen-down = +row = +dy** and **screen-right = +col = +dx** (the
    camera flip is applied UPSTREAM by the panel transform — this function
    NEVER re-applies it).  The axis map is therefore **routing only, sign =
    +identity**:

    ======  ==========  ==========
    panel   dx (→col)   dy (→row)
    ======  ==========  ==========
    XY      X           Y
    YZ      Z           Y
    XZ      X           Z
    ======  ==========  ==========

    (YZ is sagittal — col=Z, row=Y.)  Net on screen: **drag right → X+/Z+, drag
    down → Y+/Z+**.  Both components are applied (a diagonal drag steps BOTH
    in-plane axes), each clamped to ``shape=(Z, Y, X)``.  ``cursor`` is not
    mutated; a fresh ``[z, y, x]`` is returned.

    Raises
    ------
    ValueError
        If ``panel`` is not one of ``{"XY", "XZ", "YZ"}``.
    """
    Z, Y, X = int(shape[0]), int(shape[1]), int(shape[2])
    z, y, x = int(cursor[0]), int(cursor[1]), int(cursor[2])
    dx = int(round(float(d_data[0])))
    dy = int(round(float(d_data[1])))
    p = str(panel).upper()
    if p == "XY":
        x = int(np.clip(x + dx, 0, max(X - 1, 0)))
        y = int(np.clip(y + dy, 0, max(Y - 1, 0)))
    elif p == "YZ":
        z = int(np.clip(z + dx, 0, max(Z - 1, 0)))   # dx -> Z (col, sagittal)
        y = int(np.clip(y + dy, 0, max(Y - 1, 0)))   # dy -> Y (row)
    elif p == "XZ":
        x = int(np.clip(x + dx, 0, max(X - 1, 0)))
        z = int(np.clip(z + dy, 0, max(Z - 1, 0)))
    else:
        raise ValueError(f"scrub_delta_to_cursor: unknown panel {panel!r}")
    return [z, y, x]


def wheel_step(panel, direction, cursor, shape):
    """Step a panel's OWN slice axis by one voxel per wheel notch.

    Wheel over a panel walks *through depth* along that panel's slice axis:
    **XY→Z, XZ→Y, YZ→X**, by ``±1`` in the sign of ``direction`` (the primary
    Neuroglancer-style depth gesture).  ``shape`` is ``(Z, Y, X)``; the result
    is clamped; ``cursor`` is not mutated.

    Raises
    ------
    ValueError
        If ``panel`` is not one of ``{"XY", "XZ", "YZ"}``.
    """
    Z, Y, X = int(shape[0]), int(shape[1]), int(shape[2])
    z, y, x = int(cursor[0]), int(cursor[1]), int(cursor[2])
    try:
        d = float(direction)
    except Exception:
        d = 0.0
    step = 1 if d > 0 else (-1 if d < 0 else 0)
    p = str(panel).upper()
    if p == "XY":       # slice axis = Z
        z = int(np.clip(z + step, 0, max(Z - 1, 0)))
    elif p == "XZ":     # slice axis = Y
        y = int(np.clip(y + step, 0, max(Y - 1, 0)))
    elif p == "YZ":     # slice axis = X
        x = int(np.clip(x + step, 0, max(X - 1, 0)))
    else:
        raise ValueError(f"wheel_step: unknown panel {panel!r}")
    return [z, y, x]


def cut_plane_quads(cursor, shape):
    """The three 3-D cut-plane quads at ``cursor`` in the Volume's LOCAL frame.

    A vispy ``Volume`` takes a ``(Z, Y, X)`` array but its local frame is
    ``(x, y, z)`` (last array axis = local x), so these quads are authored
    in ``(x, y, z)``.  Each plane passes through the cursor with a **+0.5** offset
    on its perpendicular axis so it centres on the voxel the ortho panels show
    (axes-LOW-1):

    * **XY-cut** ⟂ Z at ``z + 0.5`` — spans ``x∈[0,X], y∈[0,Y]``
    * **XZ-cut** ⟂ Y at ``y + 0.5`` — spans ``x∈[0,X], z∈[0,Z]``
    * **YZ-cut** ⟂ X at ``x + 0.5`` — spans ``y∈[0,Y], z∈[0,Z]``

    ``cursor`` is ``[z, y, x]``; ``shape`` is ``(Z, Y, X)``.  Returns a dict
    ``{"XY"|"XZ"|"YZ": (4, 3) float32}`` of quad corner vertices (CCW), ready for
    a 2-triangle ``Mesh`` (faces ``[[0,1,2],[0,2,3]]``).  Pure / GL-free.
    """
    z, y, x = int(cursor[0]), int(cursor[1]), int(cursor[2])
    Z, Y, X = int(shape[0]), int(shape[1]), int(shape[2])
    zc, yc, xc = z + 0.5, y + 0.5, x + 0.5
    fX, fY, fZ = float(X), float(Y), float(Z)
    xy = np.array([[0.0, 0.0, zc], [fX, 0.0, zc],
                   [fX, fY, zc], [0.0, fY, zc]], dtype=np.float32)
    xz = np.array([[0.0, yc, 0.0], [fX, yc, 0.0],
                   [fX, yc, fZ], [0.0, yc, fZ]], dtype=np.float32)
    yz = np.array([[xc, 0.0, 0.0], [xc, fY, 0.0],
                   [xc, fY, fZ], [xc, 0.0, fZ]], dtype=np.float32)
    return {"XY": xy, "XZ": xz, "YZ": yz}


def _parse_voxel_size(voxel_size):
    """Coerce a ``(sz, sy, sx)`` voxel size to positive floats, or ``None``.

    Any non-3-tuple / non-finite / non-positive spacing collapses to ``None``.
    Accepted + stored but currently UNUSED — the panels render stretch-to-fill
    (matplotlib ``aspect='auto'``); anisotropy / true-proportion display is
    deferred to a later toggle.
    """
    if voxel_size is None:
        return None
    try:
        sz, sy, sx = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    except Exception:
        return None
    if not all(np.isfinite(v) and v > 0.0 for v in (sz, sy, sx)):
        return None
    return (sz, sy, sx)


# --------------------------------------------------------------------------- #
# Finer tick density for the ortho axes (Qt-free / GL-free math).
#
# vispy's ``AxisVisual`` picks tick locations with a Talbot locator whose
# density is HARD-CODED (``_get_ticks_talbot(..., density=2)``) and scaled by
# the axis's *physical* length in inches.  On our small, tiny-font panels that
# yields a coarse 0 / 50 / 100.  There is no public knob, so we override the
# per-axis ``Ticker._get_tick_frac_labels`` with a fixed-nice-step generator
# (~6-8 major ticks over the range) that ignores physical length.  It reproduces
# vispy's exact return contract (major_frac, minor_frac, labels) incl. the
# domain-flip handling, so ticks stay in DATA VOXEL coords and 0 stays at top.
# --------------------------------------------------------------------------- #
def _nice_tick_step(span, target=8):
    """A 'nice' tick step (1/2/2.5/5 x 10^k) giving ~``target`` intervals over
    ``span``.  E.g. span=128 -> 20 (0,20,..,120); span=143 -> 20 (0,..,140)."""
    import math
    try:
        span = abs(float(span))
    except Exception:
        return 1.0
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(int(target), 1)
    if raw <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def _make_fine_ticker(orig_method):
    """Build a replacement ``Ticker._get_tick_frac_labels`` bound-method body
    that uses :func:`_nice_tick_step`; falls back to ``orig_method`` on any
    surprise (non-linear scale, vispy-internal change)."""
    def _fine(self):
        try:
            if getattr(self.axis, "scale_type", "linear") != "linear":
                return orig_method()
            domain = self.axis.domain
            if domain[1] < domain[0]:
                flip = True
                dom = (domain[1], domain[0])
            else:
                flip = False
                dom = (domain[0], domain[1])
            offset = dom[0]
            scale = dom[1] - dom[0]
            step = _nice_tick_step(scale, target=8)
            if not np.isfinite(step) or step <= 0 or scale <= 0:
                return orig_method()
            start = np.ceil(dom[0] / step - 1e-9) * step
            stop = np.floor(dom[1] / step + 1e-9) * step
            count = int(round((stop - start) / step)) + 1
            if count < 2:
                return orig_method()
            major = start + np.arange(count) * step
            # Snap a near-zero tick to exactly 0.0 so it labels '0' not '-0'.
            major = np.where(np.abs(major) < step * 1e-6, 0.0, major)
            labels = ["%g" % x for x in major]

            minor_num = 4
            majstep = major[1] - major[0]
            try:
                minstart = 0 if self.axis._stop_at_major[0] else -1
                minstop = -1 if self.axis._stop_at_major[1] else 0
            except Exception:
                minstart, minstop = -1, 0
            minstep = majstep / (minor_num + 1)
            minor = []
            for i in range(minstart, len(major) + minstop):
                maj = major[0] + i * majstep
                minor.extend(np.linspace(maj + minstep,
                                         maj + majstep - minstep, minor_num))

            major_frac = (major - offset)
            minor_frac = (np.array(minor) - offset)
            if scale != 0:
                major_frac = major_frac / scale
                minor_frac = minor_frac / scale
            use_mask = (major_frac > -0.0001) & (major_frac < 1.0001)
            major_frac = major_frac[use_mask]
            labels = [l for li, l in enumerate(labels) if use_mask[li]]
            minor_frac = minor_frac[(minor_frac > -0.0001) &
                                    (minor_frac < 1.0001)]
            if flip:
                major_frac = 1 - major_frac
                minor_frac = 1 - minor_frac
            return major_frac, minor_frac, labels
        except Exception:
            try:
                return orig_method()
            except Exception:
                return np.array([]), np.array([]), []
    return _fine


def _install_fine_ticks(axis_widget):
    """Swap the fixed-nice-step ticker onto one ``AxisWidget`` (best-effort)."""
    try:
        import types
        ticker = axis_widget.axis.ticker
        orig = ticker._get_tick_frac_labels  # bound; captured before replacing
        ticker._get_tick_frac_labels = types.MethodType(
            _make_fine_ticker(orig), ticker)
    except Exception as exc:  # pragma: no cover - vispy-internal drift
        _diag(f"fine-ticks install failed {exc}")


# --------------------------------------------------------------------------- #
# Qt / vispy shell.  Imported lazily-tolerant so the pure functions above can
# be imported in a headless test environment even if Qt / vispy is unavailable.
# --------------------------------------------------------------------------- #
try:
    from qtpy.QtCore import Qt, QThread, QTimer, Signal
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except Exception as _qt_exc:  # pragma: no cover - only in headless installs
    _QT_AVAILABLE = False
    _diag(f"Qt unavailable, dialog disabled ({_qt_exc})")

# GL enum for the capability probe (not exposed as a named vispy constant).
_GL_MAX_3D_TEXTURE_SIZE = 0x8073

try:
    from vispy import scene as _vp_scene
    from vispy.scene import PanZoomCamera as _VpPanZoomCamera
    from vispy.scene import SceneCanvas as _VpSceneCanvas
    from vispy.scene import TurntableCamera as _VpTurntableCamera
    from vispy.color import Colormap as _VpColormap

    _VISPY_AVAILABLE = True
except Exception as _vp_exc:  # pragma: no cover - only in headless installs
    _VISPY_AVAILABLE = False
    _diag(f"vispy unavailable, GPU capture disabled ({_vp_exc})")


_DARK_BG = "#0e1a1a"
_CANVAS_BG = "black"          # image-forward: black everywhere behind the plots
_CROSSHAIR = (1.0, 1.0, 1.0, 0.8)  # thin but clearly-visible white crosshair
                                    # (matches the 164816 target).  Width >= 1.0
                                    # below — a sub-1px vispy GL line vanishes.
_AX_FG = "#9fd8d8"            # muted teal for the placeholder text
_AXIS_FG = "#c8d0d0"          # light grey for framed axes + tick numbers + X/Y/Z
_AXIS_BORDER = (0.72, 0.78, 0.78, 1.0)  # thin light-grey plot frame (spines)
_CLICK_PX = 3                # release within this pixel radius of press == a click
_CUT_COLOR = (0.25, 0.90, 0.90, 0.14)   # faint translucent cyan 3-D cut planes


def _query_max_3d_texture_size(canvas):
    """Read ``GL_MAX_3D_TEXTURE_SIZE`` from a LIVE context (capability probe).

    Returns the int limit, or ``None`` if it can't be read (unknown → caller
    attempts a guarded build).  Makes ``canvas`` current first — the value can
    only be read from a current context.
    """
    try:
        canvas.set_current()
    except Exception as exc:  # pragma: no cover - backend-specific
        _diag(f"set_current before GL probe failed ({exc})")
    try:
        from vispy.gloo import gl
        val = gl.glGetParameter(_GL_MAX_3D_TEXTURE_SIZE)
        try:
            val = int(np.ravel(np.asarray(val))[0])
        except Exception:
            val = int(val)
        if val <= 0:
            return None
        return val
    except Exception as exc:  # pragma: no cover - driver/backend-specific
        _diag(f"GL_MAX_3D_TEXTURE_SIZE query failed ({exc})")
        return None

_DARK_STYLE = """
QDialog { background-color: #0e1a1a; color: #dddddd; }
QLabel { color: #dddddd; background: transparent; }
QCheckBox { color: #dddddd; background: transparent; }
QPushButton {
    background-color: #16302f; color: #dddddd;
    border: 1px solid #3fb6b6; border-radius: 4px;
    padding: 5px 12px;
}
QPushButton:hover { background-color: #1f4544; }
QPushButton:pressed { background-color: #3fb6b6; color: #0e1a1a; }
QSlider::groove:horizontal {
    height: 6px; background: #16302f; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #3fb6b6; width: 14px; margin: -5px 0; border-radius: 7px;
}
"""


# Keep a strong reference to any load worker that outlives its dialog (a slow
# ``tifffile.imread`` that is still running at teardown).  Dropping the only
# Python ref to a *running* QThread risks "QThread destroyed while running"; we
# retain it here until it self-cleans on ``finished``.
_LIVE_DETACHED_WORKERS = set()


def _gl_disabled_by_env():
    """Out-of-band 2-D-only / no-canvas kill switch.

    A native segfault in *context creation itself* cannot be caught by
    ``try/except``; ``EPICYCLE_CAPTURE_NO_GL`` lets a user force the safe
    no-canvas path on a host where no GL context is creatable.
    """
    val = os.environ.get("EPICYCLE_CAPTURE_NO_GL", "")
    return str(val).strip().lower() not in ("", "0", "false", "no", "off")


def _is_deleted(obj):
    """Best-effort ``sip.isdeleted`` across PyQt (sip) and PySide (shiboken)."""
    for modname, attr, invert in (
        ("sip", "isdeleted", False),
        ("PyQt5.sip", "isdeleted", False),
        ("PyQt6.sip", "isdeleted", False),
        ("shiboken2", "isValid", True),
        ("shiboken6", "isValid", True),
    ):
        try:
            mod = __import__(modname, fromlist=[attr])
            res = bool(getattr(mod, attr)(obj))
            return (not res) if invert else res
        except Exception:
            continue
    return False


if _QT_AVAILABLE:

    class _PatchLoadWorker(QThread):
        """Read a patch TIFF off the UI thread (tens of MB).

        Emits ``done(ndarray)`` on success or ``failed(str)`` on any error.
        ``tifffile.imread`` does not poll interruption, so this thread cannot be
        force-stopped mid-read — the dialog's teardown disconnects
        the signals and lets a running read self-clean instead of waiting.
        """

        done = Signal(object, object)   # (data, limits) — limits computed off-UI
        failed = Signal(str)

        def __init__(self, path, parent=None):
            super().__init__(parent)
            self._path = path

        def run(self):  # noqa: D401 - QThread entry point
            try:
                _diag(f"load-start {self._path}")
                import tifffile

                data = tifffile.imread(self._path)
                data = np.asarray(data)
                # F2(a): compute the volume-global per-channel contrast HERE, off
                # the UI thread (two full-volume percentile passes per channel —
                # the launch-freeze audit flagged this as main-thread work with
                # no indicator).  _on_loaded consumes it, so the UI thread never
                # blocks on the percentile.  Best-effort: None -> _on_loaded
                # recomputes, keeping the old behaviour.
                limits = None
                try:
                    vol = _promote_to_4d(data)
                    if vol is not None:
                        limits = volume_channel_limits(vol)
                except Exception as exc:
                    _diag(f"worker limits failed {exc}")
                _diag(f"load-done shape={getattr(data, 'shape', '?')}")
                self.done.emit(data, limits)
            except Exception as exc:
                _diag(f"load-failed {exc}")
                try:
                    self.failed.emit(str(exc))
                except Exception:
                    pass

    class PatchCaptureDialog(QDialog):
        """Floating, non-modal orthogonal inspector for a single saved patch.

        Shows THREE synced vispy panels (XY / XZ / YZ) with a click-to-move
        crosshair and volume-global contrast, plus per-channel toggles and a Z
        slider (Point mode).  The bottom-right quadrant is a placeholder for the
        Brick-3 3-D volume.

        Parameters
        ----------
        parent : QWidget
            Parent widget (kept so the dialog is not a taskbar orphan).
        patch_name, tiff_path, channel_names
            Identify + describe the patch to load.
        voxel_size : (sz, sy, sx) | None
            Physical spacing for anisotropy-correct display.  When
            ``None``, the panels render isotropic with an honest "Z spacing
            unknown" stamp.  Optional (back-compat with the 4-arg call site).
        on_close : callable | None
            Zero-arg callback invoked EXACTLY ONCE when the dialog closes, so the
            controller can prune it from ``_capture_dialogs``.
        """

        def __init__(self, parent, patch_name, tiff_path, channel_names,
                     voxel_size=None, on_close=None, on_loaded=None):
            super().__init__(parent)
            self._patch_name = str(patch_name or "patch")
            self._tiff_path = tiff_path
            self._channel_names = list(channel_names or [])
            self._voxel_size = _parse_voxel_size(voxel_size)
            self._on_close = on_close if callable(on_close) else None
            self._on_close_fired = False
            # Fires ONCE when the patch data has loaded (or failed) — lets the
            # launcher clear a busy indicator (the ④ scan bar).
            self._on_loaded_cb = on_loaded if callable(on_loaded) else None
            self._on_loaded_cb_fired = False

            self._data = None            # raw array as loaded
            self._vol = None             # promoted (C, Z, Y, X) view
            self._limits = None          # per-channel volume-global (lo, hi)
            self._cursor = [0, 0, 0]     # [z, y, x] crosshair focus
            self._worker = None
            self._closing = False

            # CW1/CW2 render throttle (see `panels_to_reslice`): coalesce rapid
            # scrub/wheel cursor moves into ONE redraw per event-loop pass, and
            # redraw only the ortho panels whose slice index changed.
            self._render_timer = QTimer(self)
            self._render_timer.setSingleShot(True)
            self._render_timer.setInterval(0)   # fire on the next event-loop pass
            self._render_timer.timeout.connect(self._on_render_timer)
            self._render_pending = False
            self._last_rendered_cursor = None

            # vispy scene handles (populated on load).
            self._canvas = None
            self._ortho_views = {}       # panel -> ViewBox
            self._titles = {}            # panel -> Label (per-panel slice title)
            self._axis_widgets = []      # AxisWidgets (numbered L/B tick axes)
            self._images = {}            # panel -> Image visual
            self._crosshairs = {}        # panel -> Line visual
            self._overlay_visuals = []   # placeholder / stamp Text visuals
            self._panel_hw = {}          # panel -> (H, W) for click hit-test
            self._press_pos = None       # last mouse-press canvas pos (Point)
            self._display_mode = "ok"    # ok | gl_disabled | unavailable
            self._chan_checks = []

            # Brick 3 — the 3-D volume quadrant (bottom-right).
            self._vb_3d = None           # the 3-D ViewBox
            self._volumes = {}           # channel index -> Volume visual
            self._cut_meshes = {}        # panel -> cut-plane Mesh visual
            self._cut_visible = True     # "Show cut planes" state
            self._vol3d_label = None     # Text shown when there's no Volume
            self._volume_built = False   # True once the Volume(s) exist
            self._mode3d = "attenuated"  # "attenuated" | "mip"

            # Interaction mode (Brick 2).  Point = click-to-move (Brick 1);
            # Scrub = wheel-through-depth + drag-scrub with the crosshair locked
            # to each panel's screen centre.
            self._mode = "point"
            self._scrub_dragging = False       # a Scrub left-drag is in flight
            self._scrub_panel = None           # which panel the drag started in
            self._scrub_origin_pos = None      # canvas px at drag start
            self._scrub_origin_cursor = None   # cursor snapshot at drag start

            try:
                _diag(f"dialog open patch={self._patch_name!r} path={tiff_path} "
                      f"voxel_size={self._voxel_size}")
                self._build_ui()
                if self._display_mode == "ok":
                    self._start_load()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"dialog init failed {exc}")

        # -- UI construction ------------------------------------------------ #
        def _build_ui(self):
            try:
                self.setWindowTitle(f"Capture — {self._patch_name}")
                # Float as its own top-level window while keeping a parent so
                # it is not a taskbar orphan.
                self.setWindowFlags(Qt.Window)
                self.setModal(False)
                self.setStyleSheet(_DARK_STYLE)
                self.resize(820, 860)

                root = QVBoxLayout(self)

                self._title = QLabel(self._patch_name, self)
                self._title.setStyleSheet("font-size: 15px; font-weight: bold;")
                root.addWidget(self._title)

                # "Loading…" / error / disabled placeholder shown over the canvas.
                self._status = QLabel("Loading…", self)
                self._status.setAlignment(Qt.AlignCenter)
                self._status.setStyleSheet("font-size: 14px; color: #3fb6b6;")
                root.addWidget(self._status)

                # -- The GPU surface (build order) --------------- #
                # Build the 2-D canvas FIRST; the Brick-3 GL_MAX_3D_TEXTURE_SIZE
                # probe + Volume construction hook go where marked below.
                if _gl_disabled_by_env():
                    self._display_mode = "gl_disabled"
                    self._status.setText(
                        "Capture GL disabled (EPICYCLE_CAPTURE_NO_GL set).")
                    self._status.setStyleSheet("color: #d0b46a; font-size: 13px;")
                elif not _VISPY_AVAILABLE:
                    self._display_mode = "unavailable"
                    self._status.setText("Capture view unavailable (vispy).")
                    self._status.setStyleSheet("color: #e07a7a; font-size: 13px;")
                else:
                    try:
                        self._build_canvas(root)
                        self._display_mode = "ok"
                    except Exception as exc:  # pragma: no cover - GPU/driver
                        _diag(f"canvas build failed {exc}")
                        self._teardown_canvas()  # undo any partial build
                        self._display_mode = "unavailable"
                        self._status.setText("Capture view unavailable.")
                        self._status.setStyleSheet(
                            "color: #e07a7a; font-size: 13px;")

                # Controls row: mode toggle + Z slider + channel toggles.
                controls = QWidget(self)
                crow = QHBoxLayout(controls)
                crow.setContentsMargins(0, 0, 0, 0)

                # Point <-> Scrub mode toggle (checkable = clear indicator; the
                # mouse cursor also changes per mode).  Default = Point.
                self._mode_btn = QPushButton("Point mode", controls)
                self._mode_btn.setCheckable(True)
                self._mode_btn.setChecked(False)
                self._mode_btn.setToolTip(
                    "Point: click to move the crosshair.\n"
                    "Scrub: wheel = step depth; drag = scrub in-plane "
                    "(crosshair locked to centre).")
                self._mode_btn.toggled.connect(self._on_mode_toggled)
                crow.addWidget(self._mode_btn)

                self._z_label = QLabel("Z", controls)
                crow.addWidget(self._z_label)
                self._z_slider = QSlider(Qt.Horizontal, controls)
                self._z_slider.setMinimum(0)
                self._z_slider.setMaximum(0)
                self._z_slider.setEnabled(False)
                self._z_slider.valueChanged.connect(self._on_z_slider)
                crow.addWidget(self._z_slider, stretch=1)

                # Container for the per-channel checkboxes (populated on load).
                self._chan_container = QWidget(controls)
                self._chan_layout = QHBoxLayout(self._chan_container)
                self._chan_layout.setContentsMargins(0, 0, 0, 0)
                crow.addWidget(self._chan_container)

                # 3-D render-mode combo — default Attenuated.
                self._mode3d_combo = QComboBox(controls)
                self._mode3d_combo.addItem("3D: Attenuated", "attenuated")
                self._mode3d_combo.addItem("3D: MIP", "mip")
                self._mode3d_combo.setCurrentIndex(0)
                self._mode3d_combo.setToolTip(
                    "3-D render method: attenuated MIP (depth cue) or plain MIP.")
                self._mode3d_combo.currentIndexChanged.connect(
                    self._on_mode3d_changed)
                crow.addWidget(self._mode3d_combo)

                # "Show cut planes" — default on.
                self._cut_check = QCheckBox("Cut planes", controls)
                self._cut_check.setChecked(True)
                self._cut_check.setToolTip(
                    "Show the 3 translucent cut-plane quads at the cursor.")
                self._cut_check.stateChanged.connect(self._on_cut_toggled)
                crow.addWidget(self._cut_check)

                root.addWidget(controls)

                # Bottom button row (Save is a Brick-4 concern — not built here).
                btns = QHBoxLayout()
                btns.addStretch(1)
                close_btn = QPushButton("Close", self)
                close_btn.clicked.connect(self.close)
                btns.addWidget(close_btn)
                root.addLayout(btns)
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"build_ui failed {exc}")

        def _build_canvas(self, root_layout):
            """Build the single standalone ``SceneCanvas`` + 2x2 ortho grid.

            Each ortho cell is a framed sub-grid — titled header + numbered
            left/bottom tick axes (X/Y/Z labels) + a flipped,
            aspect-correct viewbox.  No image data yet — the per-panel ``Image``
            / crosshair ``Line`` are created in :meth:`_populate_panels` once the
            volume has loaded.  The canvas + every viewbox paint BLACK so empty
            margin around the non-square side panels reads black.
            """
            self._canvas = _VpSceneCanvas(bgcolor=_CANVAS_BG, keys=None,
                                          show=False)
            native = self._canvas.native
            try:
                native.setMinimumSize(460, 460)
            except Exception:
                pass
            root_layout.addWidget(native, stretch=1)

            outer = self._canvas.central_widget.add_grid(spacing=12, margin=8)
            # 2x2 layout:  XY | YZ  /  XZ | (3-D placeholder), square cells.
            cells = {"XY": (0, 0), "YZ": (0, 1), "XZ": (1, 0)}
            # Edge axis labels (bottom, left):
            #   XY: bottom=X, left=Y ;  XZ: bottom=X, left=Z ;  YZ: bottom=Z, left=Y
            #   (YZ sagittal — Y vertical, Z horizontal).
            axis_labels = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Z", "Y")}
            for panel, (r, c) in cells.items():
                xlabel, ylabel = axis_labels[panel]
                vb = self._build_ortho_cell(outer, r, c, panel, xlabel, ylabel)
                self._ortho_views[panel] = vb

            # Bottom-right = the 3-D volume quadrant.  A neutral 2-D camera +
            # "loading" Text hold it until the volume builds after load (or a
            # capability / flat-patch label replaces the text).
            vb_3d = outer.add_view(row=1, col=1, bgcolor=_CANVAS_BG)
            vb_3d.camera = _VpPanZoomCamera(rect=(0, 0, 1, 1))
            self._vol3d_label = _vp_scene.visuals.Text(
                "3-D view (loading…)",
                parent=vb_3d.scene, color=_AX_FG, font_size=9,
                anchor_x="center", anchor_y="center", pos=(0.5, 0.5),
            )
            self._vb_3d = vb_3d
            self._overlay_visuals.append(self._vol3d_label)

            # Mouse wiring.  Point mode: click on RELEASE (drag < _CLICK_PX).
            # Scrub mode: press starts a drag, move scrubs the in-plane axes,
            # wheel steps the panel's slice axis.  ALL four are disconnected in
            # the teardown (same disconnect list) so teardown stays crash-
            # safe.
            self._canvas.events.mouse_press.connect(self._on_mouse_press)
            self._canvas.events.mouse_release.connect(self._on_mouse_release)
            self._canvas.events.mouse_move.connect(self._on_mouse_move)
            self._canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
            self._apply_mode_cursor()  # default Point -> cross cursor

            # ----------------------------------------------------------------
            # TODO(Brick 3): GL capability probe.  The context is
            # LIVE now (2-D canvas built above), so read the 3-D texture limit
            # from ``self._canvas.context`` and only build the per-channel
            # ``Volume`` when GL_MAX_3D_TEXTURE_SIZE >= max(Z, Y, X); else show
            # "3-D volume unavailable on this GPU".  Brick 1 builds no Volume.
            # e.g.:
            #   gl_max = self._canvas.context.glGetParameter(GL_MAX_3D_TEXTURE_SIZE)
            # ----------------------------------------------------------------

        def _build_ortho_cell(self, outer, row, col, panel, xlabel, ylabel):
            """One framed ortho panel inside outer-grid cell ``(row, col)``.

            Sub-grid layout::

                +----------------------------+
                |        title (Label)       |   row 0, cols 0-1
                +------+---------------------+
                | Yaxis|      ViewBox        |   row 1
                +------+---------------------+
                |      |       Xaxis         |   row 2, col 1
                +------+---------------------+

            The camera is FLIPPED on Y (row 0 at the TOP) with NO aspect
            lock, so the image STRETCHES edge-to-edge to fill the plot rectangle
            (matplotlib ``aspect='auto'``; the fill rect is set on load in
            :meth:`_populate_panels`).  Both axes ``link_view`` the viewbox, so
            their numbers read DATA VOXEL coords and honour the flip
            (``AxisWidget`` derives its domain from the live scene transform, so
            "0" ends up where the old matplotlib figures had it).

            Anisotropy / ``voxel_size`` true-proportion is DEFERRED (a later
            toggle); ``voxel_size`` is still accepted + stored but unused."""
            sub = outer.add_grid(row=row, col=col, spacing=0, margin=0)

            title = _vp_scene.Label("", color=_AXIS_FG, font_size=5)
            try:
                title.height_max = 20
            except Exception:
                pass
            sub.add_widget(title, row=0, col=0, col_span=2)
            self._titles[panel] = title

            yaxis = _vp_scene.AxisWidget(
                orientation="left", axis_label=ylabel,
                axis_color=_AXIS_FG, tick_color=_AXIS_FG, text_color=_AXIS_FG,
                axis_font_size=4, tick_font_size=3, axis_label_margin=22,
                tick_label_margin=6, axis_width=1, tick_width=1,
                major_tick_length=4, minor_tick_length=0)
            try:
                yaxis.width_max = 40
            except Exception:
                pass
            sub.add_widget(yaxis, row=1, col=0)

            # Thin light-grey frame (the matplotlib spines) around the plot.
            vb = sub.add_view(row=1, col=1, bgcolor=_CANVAS_BG,
                              border_color=_AXIS_BORDER)
            cam = _VpPanZoomCamera()            # NO aspect lock -> stretch to fill
            cam.flip = (False, True, False)
            vb.camera = cam

            xaxis = _vp_scene.AxisWidget(
                orientation="bottom", axis_label=xlabel,
                axis_color=_AXIS_FG, tick_color=_AXIS_FG, text_color=_AXIS_FG,
                axis_font_size=4, tick_font_size=3, axis_label_margin=18,
                tick_label_margin=6, axis_width=1, tick_width=1,
                major_tick_length=4, minor_tick_length=0)
            try:
                xaxis.height_max = 28
            except Exception:
                pass
            sub.add_widget(xaxis, row=2, col=1)

            xaxis.link_view(vb)
            yaxis.link_view(vb)
            # Finer, fixed-nice-step ticks (~6-8 majors) instead of vispy's
            # length-scaled coarse 0/50/100 (see _install_fine_ticks above).
            _install_fine_ticks(xaxis)
            _install_fine_ticks(yaxis)
            self._axis_widgets.extend([xaxis, yaxis])
            return vb

        def _rebuild_channel_checks(self, n_channels):
            """Create one checkbox per real channel (all checked by default)."""
            try:
                # Clear any existing checkboxes.
                for cb in self._chan_checks:
                    try:
                        cb.stateChanged.disconnect()
                    except Exception:
                        pass
                    try:
                        self._chan_layout.removeWidget(cb)
                        cb.setParent(None)
                    except Exception:
                        pass
                self._chan_checks = []

                for c in range(max(int(n_channels), 0)):
                    if c < len(self._channel_names) and self._channel_names[c]:
                        label = str(self._channel_names[c])
                    else:
                        label = f"Ch {c}"
                    cb = QCheckBox(label, self._chan_container)
                    cb.setChecked(True)
                    cb.stateChanged.connect(self._on_render)
                    self._chan_layout.addWidget(cb)
                    self._chan_checks.append(cb)
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"rebuild channel checks failed {exc}")

        # -- Loading -------------------------------------------------------- #
        def _start_load(self):
            try:
                self._worker = _PatchLoadWorker(self._tiff_path, self)
                self._worker.done.connect(self._on_loaded)
                self._worker.failed.connect(self._on_load_failed)
                self._worker.start()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"start_load failed {exc}")
                self._on_load_failed(str(exc))

        def _on_loaded(self, data, limits=None):
            # a late emit after teardown-begins must be inert — never
            # touch a GL context / widget after closing.
            if self._closing or _is_deleted(self):
                _diag("on_loaded ignored (closing/deleted)")
                return
            try:
                self._data = np.asarray(data)
                self._vol = _promote_to_4d(self._data)
                if self._vol is None:
                    self._on_load_failed(
                        f"unsupported array shape {self._data.shape}")
                    return
                C, Z, Y, X = self._vol.shape

                self._rebuild_channel_checks(C)

                # Volume-global per-channel contrast, computed ONCE (perf
                # invariant — never per interaction).  F2(a): the load worker
                # already computed it OFF the UI thread; use that.  Fall back to
                # computing here only if the worker couldn't (keeps behaviour).
                self._limits = (limits if limits is not None
                                else volume_channel_limits(self._vol))
                _diag(f"volume limits (per-channel) = {self._limits}")

                # Crosshair starts at the volume centre.
                self._cursor = [
                    max(Z // 2, 0),
                    max(Y // 2, 0),
                    max(X // 2, 0),
                ]

                # H/W per panel for the click hit-test.
                # YZ is sagittal: rows=Y (H), cols=Z (W).
                self._panel_hw = {
                    "XY": (int(Y), int(X)),
                    "XZ": (int(Z), int(X)),
                    "YZ": (int(Y), int(Z)),
                }

                self._z_slider.blockSignals(True)
                self._z_slider.setMaximum(max(int(Z) - 1, 0))
                self._z_slider.setValue(int(self._cursor[0]))
                self._z_slider.setEnabled(int(Z) > 1)
                self._z_slider.blockSignals(False)

                # F2(a): the 3-D texture build + first render below run on the UI
                # thread (GL work can't move off it).  The status ROW stays up
                # through them, and we force-paint a "building" message FIRST so
                # the user sees honest progress rather than a frozen canvas — the
                # build never yields to the event loop, so a plain setText would
                # not repaint.  repaint() is synchronous (no event-loop re-entrancy,
                # unlike processEvents).  Hidden only once the view is ready.
                if self._canvas is not None:
                    try:
                        self._status.setText("Building 3-D view…")
                        self._status.repaint()
                    except Exception:
                        pass
                    self._populate_panels()
                    # If the user flipped to Scrub before the load finished,
                    # honour the cursor-preserving centre now.
                    if self._mode == "scrub":
                        for panel in _PANELS:
                            vb = self._ortho_views.get(panel)
                            try:
                                if vb is not None:
                                    vb.camera.interactive = False
                            except Exception:
                                pass
                        self._center_cameras_on_cursor()
                    self._render_all()

                self._status.setVisible(False)   # view ready -> reveal panels
                self._fire_loaded_cb()           # ready → clear the ④ scan cue
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"on_loaded failed {exc}")
                self._on_load_failed(str(exc))

        def _on_load_failed(self, msg):
            try:
                _diag(f"on_load_failed {msg}")
                self._status.setVisible(True)
                self._status.setText(f"Could not load patch:\n{msg}")
                self._status.setStyleSheet("color: #e07a7a; font-size: 13px;")
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"on_load_failed handler failed {exc}")
            self._fire_loaded_cb()   # clear the caller's busy cue on failure too

        def _fire_loaded_cb(self):
            """Fire the ``on_loaded`` callback exactly once (load resolved —
            success or failure), so the launcher can clear its busy indicator.
            Guarded — never raises."""
            if self._on_loaded_cb is None or self._on_loaded_cb_fired:
                return
            self._on_loaded_cb_fired = True
            try:
                self._on_loaded_cb()
            except Exception as exc:   # pragma: no cover - defensive
                _diag(f"on_loaded_cb failed {exc}")

        # -- Panel construction (needs data) -------------------------------- #
        def _populate_panels(self):
            """Create the per-panel ``Image`` + crosshair ``Line`` once the volume
            is loaded and STRETCH each image edge-to-edge to fill its plot.

            Image + crosshair sit directly under ``vb.scene`` in DATA VOXEL coords.
            The camera has no aspect lock, and the fill rect is set to the exact
            data bounds ``(0, 0, W, H)`` so the image fills its cell with no black
            bars (matplotlib ``aspect='auto'``).  The fill runs ONCE here; cursor
            moves never re-fit (a scrub must not re-zoom)."""
            if self._canvas is None or self._vol is None:
                return
            channels_on = self._current_channels_on()
            z, y, x = self._clamped_cursor()
            for panel in _PANELS:
                vb = self._ortho_views.get(panel)
                if vb is None:
                    continue
                try:
                    idx = self._panel_slice_index(panel, (z, y, x))
                    plane = plane_reslice(self._vol, panel, idx)  # (C, H, W)
                    rgb = compose_capture_rgb(
                        plane[:, None, :, :], 0, False, channels_on,
                        limits=self._limits)                       # (H, W, 3)

                    img = _vp_scene.visuals.Image(
                        rgb, parent=vb.scene, interpolation="nearest")
                    img.order = 0
                    self._images[panel] = img

                    line = _vp_scene.visuals.Line(
                        pos=self._crosshair_pos(panel, (z, y, x)),
                        color=_CROSSHAIR, width=1.2, connect="segments",
                        parent=vb.scene)
                    # THE crosshair fix: draw ON TOP of the Image via a very high
                    # order + NO depth test (box-overlay idiom).  Without the
                    # depth-test-off, the opaque Image occludes the Line and it
                    # never shows — order alone was not enough.
                    try:
                        line.order = 1000000
                        line.set_gl_state("translucent", depth_test=False)
                    except Exception as exc:
                        _diag(f"crosshair gl_state {panel} failed: {exc}")
                    self._crosshairs[panel] = line

                    title = self._titles.get(panel)
                    if title is not None:
                        try:
                            title.text = self._panel_title(panel, (z, y, x))
                        except Exception:
                            pass

                    # Fill-on-open: set the camera rect to the exact data bounds
                    # (0..W, 0..H) so the image fills edge-to-edge (no aspect
                    # lock, no letterbox).  Done ONCE; scrubbing never re-fits.
                    try:
                        H, W = self._panel_hw[panel]
                        vb.camera.rect = (0, 0, int(W), int(H))
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - GPU/driver
                    _diag(f"populate panel {panel} failed {exc}")

            # Build the 3-D volume quadrant (gated by the capability probe).
            self._build_volume()

        # -- 3-D volume (Brick 3) ------------------------------------------- #
        def _show_3d_message(self, text, color=_AX_FG):
            """Put ``text`` in the 3-D quadrant (a 2-D camera + centred Text) —
            used for the flat-patch / GPU-unavailable / error degrade states."""
            try:
                vb = self._vb_3d
                if vb is None:
                    return
                # A neutral 2-D camera so the Text sits still.
                try:
                    if not isinstance(vb.camera, _VpPanZoomCamera):
                        vb.camera = _VpPanZoomCamera(rect=(0, 0, 1, 1))
                except Exception:
                    pass
                if self._vol3d_label is None:
                    self._vol3d_label = _vp_scene.visuals.Text(
                        text, parent=vb.scene, color=color, font_size=9,
                        anchor_x="center", anchor_y="center", pos=(0.5, 0.5))
                    self._overlay_visuals.append(self._vol3d_label)
                else:
                    self._vol3d_label.text = text
                    self._vol3d_label.color = color
                    self._vol3d_label.visible = True
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"show-3d-message failed {exc}")

        def _channel_colormap(self, c, C):
            """Pure-primary colormap so a channel is the SAME colour in 2-D and
            3-D: ch0->red, ch1->green, ch2->blue; C==1 -> gray."""
            if C == 1:
                return _VpColormap([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
            prim = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0],
                    2: [0.0, 0.0, 1.0]}.get(c, [1.0, 1.0, 1.0])
            return _VpColormap([[0.0, 0.0, 0.0, 1.0], prim + [1.0]])

        def _build_volume(self):
            """Build one additive ``Volume`` per channel in the 3-D quadrant,
            GATED by the capability probe.  Never rebuilt after this —
            channel toggles / contrast / mode mutate properties in place.

            Degrade paths (all keep the 3 ortho panels working):
              * ``EPICYCLE_CAPTURE_NO_GL`` / no canvas -> not reached (no canvas).
              * Z < 2 (flat patch) -> "2-D patch — no 3-D view".
              * GL_MAX_3D_TEXTURE_SIZE < max(Z,Y,X) -> "3-D volume unavailable
                on this GPU".
              * any Python-catchable build failure -> "3-D volume unavailable".
            """
            if self._canvas is None or self._vol is None or self._vb_3d is None:
                return
            try:
                C, Z, Y, X = (int(self._vol.shape[0]), int(self._vol.shape[1]),
                              int(self._vol.shape[2]), int(self._vol.shape[3]))
            except Exception:
                return

            if Z < 2:
                _diag("3D skip: flat patch (Z<2)")
                self._show_3d_message("2-D patch — no 3-D view")
                return

            need = max(Z, Y, X)
            maxtex = _query_max_3d_texture_size(self._canvas)
            if maxtex is not None and need > maxtex:
                _diag(f"3D skip: GL_MAX_3D_TEXTURE_SIZE={maxtex} < need={need}")
                self._show_3d_message("3-D volume unavailable on this GPU",
                                      color="#e0a06a")
                return
            _diag(f"3D build: GL_MAX_3D_TEXTURE_SIZE={maxtex} need={need} "
                  f"(C={C},Z={Z},Y={Y},X={X})")

            try:
                vb = self._vb_3d
                channels_on = self._current_channels_on()
                method = ("mip" if self._mode3d == "mip" else "attenuated_mip")

                # Rotatable camera.  up='-z' so 3-D "up" matches the flipped
                # orthos (Z increases downward there — axes-LOW-3).
                try:
                    vb.camera = _VpTurntableCamera(
                        up="-z", fov=45.0, azimuth=130.0, elevation=25.0)
                except Exception:
                    vb.camera = _VpTurntableCamera()

                # One additive Volume per channel, in the Volume's (Z,Y,X) array
                # frame; colour = the matching pure primary; clim = the SAME
                # volume-global per-channel bounds used by the 2-D compose.
                lim = self._limits or []
                for c in range(C):
                    vol_c = np.ascontiguousarray(
                        self._vol[c], dtype=np.float32)     # (Z, Y, X)
                    clim = lim[c] if c < len(lim) else "auto"
                    vis = _vp_scene.visuals.Volume(
                        vol_c, clim=clim, method=method,
                        cmap=self._channel_colormap(c, C), parent=vb.scene)
                    try:
                        vis.set_gl_state("additive", depth_test=False,
                                         cull_face=False)
                    except Exception:
                        pass
                    on = True if channels_on is None else (
                        channels_on[c] if c < len(channels_on) else True)
                    vis.visible = bool(on)
                    self._volumes[c] = vis

                # 3 translucent cut-plane quads at the cursor.
                faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
                z, y, x = self._clamped_cursor()
                quads = cut_plane_quads((z, y, x), (Z, Y, X))
                for panel in _PANELS:
                    mesh = _vp_scene.visuals.Mesh(
                        vertices=quads[panel], faces=faces, color=_CUT_COLOR,
                        parent=vb.scene)
                    try:
                        mesh.set_gl_state("translucent", depth_test=False,
                                          cull_face=False)
                    except Exception:
                        pass
                    mesh.visible = self._cut_visible
                    self._cut_meshes[panel] = mesh

                # Hide the loading text; frame the data.
                if self._vol3d_label is not None:
                    try:
                        self._vol3d_label.visible = False
                    except Exception:
                        pass
                try:
                    vb.camera.set_range()
                except Exception:
                    pass

                self._volume_built = True
                _diag(f"3D built: {len(self._volumes)} volume(s), "
                      f"method={method}, cut_planes={len(self._cut_meshes)}")
            except Exception as exc:  # pragma: no cover - GPU/driver
                _diag(f"3D build failed ({exc}); showing unavailable label")
                self._teardown_3d_visuals()
                self._show_3d_message("3-D volume unavailable", color="#e0a06a")

        def _sync_3d(self):
            """Keep the 3-D quadrant in step with the cursor + channel state.
            Channel visibility + render mode are IN-PLACE property mutations —
            Volumes are NEVER added/removed here."""
            if not self._volume_built:
                return
            channels_on = self._current_channels_on()
            for c, vis in self._volumes.items():
                try:
                    on = True if channels_on is None else (
                        channels_on[c] if c < len(channels_on) else True)
                    vis.visible = bool(on)
                except Exception:
                    pass
            self._sync_cut_planes()

        def _sync_cut_planes(self):
            """Move the 3 cut-plane quads to the current cursor + apply the
            show/hide toggle.  Vertices only — never rebuild."""
            if not self._cut_meshes or self._vol is None:
                return
            try:
                z, y, x = self._clamped_cursor()
                C, Z, Y, X = self._vol.shape
                quads = cut_plane_quads((z, y, x), (int(Z), int(Y), int(X)))
                faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
                for panel, mesh in self._cut_meshes.items():
                    try:
                        mesh.visible = bool(self._cut_visible)
                        if self._cut_visible:
                            mesh.set_data(vertices=quads[panel], faces=faces,
                                          color=_CUT_COLOR)
                    except Exception:
                        pass
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"cut-plane sync failed {exc}")

        def _on_mode3d_changed(self, *args):
            """Render-mode combo -> mutate every Volume's method in place."""
            try:
                data = self._mode3d_combo.currentData()
                self._mode3d = "mip" if data == "mip" else "attenuated"
                method = "mip" if self._mode3d == "mip" else "attenuated_mip"
                _diag(f"3D mode -> {method}")
                for vis in self._volumes.values():
                    try:
                        vis.method = method
                    except Exception:
                        pass
                try:
                    if self._canvas is not None:
                        self._canvas.update()
                except Exception:
                    pass
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"mode3d change failed {exc}")

        def _on_cut_toggled(self, *args):
            """"Show cut planes" checkbox -> toggle the quad visibility."""
            try:
                self._cut_visible = bool(self._cut_check.isChecked())
                _diag(f"cut planes -> {'on' if self._cut_visible else 'off'}")
                self._sync_cut_planes()
                if self._canvas is not None:
                    self._canvas.update()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"cut toggle failed {exc}")

        def _teardown_3d_visuals(self):
            """Hide -> unparent -> drop the Volumes + cut-plane Meshes (used by
            the build-failure degrade path; teardown reuses the same lists)."""
            for vis in (list(self._volumes.values())
                        + list(self._cut_meshes.values())):
                try:
                    vis.visible = False
                except Exception:
                    pass
                try:
                    vis.parent = None
                except Exception:
                    pass
            self._volumes = {}
            self._cut_meshes = {}
            self._volume_built = False

        # -- Small pure-ish geometry helpers -------------------------------- #
        def _clamped_cursor(self):
            C, Z, Y, X = self._vol.shape
            z = int(np.clip(int(self._cursor[0]), 0, max(Z - 1, 0)))
            y = int(np.clip(int(self._cursor[1]), 0, max(Y - 1, 0)))
            x = int(np.clip(int(self._cursor[2]), 0, max(X - 1, 0)))
            return z, y, x

        @staticmethod
        def _panel_slice_index(panel, cursor):
            """Which cursor coord indexes ``panel``'s slice axis."""
            z, y, x = cursor
            return {"XY": z, "XZ": y, "YZ": x}[panel]

        @staticmethod
        def _panel_title(panel, cursor):
            """Header text showing the cursor's slice index for ``panel``:
            XY -> z, XZ -> y, YZ -> x."""
            z, y, x = cursor
            if panel == "XY":
                return f"XY  (z={z})"
            if panel == "XZ":
                return f"XZ  (y={y})"
            return f"YZ  (x={x})"

        def _panel_incol_row(self, panel, cursor):
            """(col, row, W, H) — the in-plane crosshair coords for ``panel``."""
            z, y, x = cursor
            H, W = self._panel_hw[panel]
            if panel == "XY":
                return x, y, W, H
            if panel == "XZ":
                return x, z, W, H
            # YZ (sagittal): col-axis = Z, row-axis = Y.
            return z, y, W, H

        def _crosshair_pos(self, panel, cursor):
            """A (4, 2) segment array: vertical line at col + horizontal at row.

            Coordinates are in data-voxel space (x = col-axis, y = row-axis) —
            the same frame as the panel's Image, so the crosshair tracks it.
            """
            col, row, W, H = self._panel_incol_row(panel, cursor)
            return np.array(
                [[col, 0.0], [col, float(H)],
                 [0.0, row], [float(W), row]], dtype=np.float32)

        # -- Rendering ------------------------------------------------------ #
        def _current_channels_on(self):
            if self._chan_checks:
                return [cb.isChecked() for cb in self._chan_checks]
            return None

        def _on_render(self, *args):
            """Signal-compatible entry point (channel checkboxes)."""
            self._render_all()

        def _on_z_slider(self, value):
            """Z slider drives the crosshair Z (Point mode)."""
            try:
                if self._vol is None:
                    return
                Z = self._vol.shape[1]
                self._cursor[0] = int(np.clip(int(value), 0, max(Z - 1, 0)))
                _diag(f"cursor -> ({self._cursor[0]},{self._cursor[1]},"
                      f"{self._cursor[2]})")
                self._render_all()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"z-slider failed {exc}")

        def _sync_z_slider(self):
            """Reflect the current cursor Z on the slider without re-rendering."""
            try:
                self._z_slider.blockSignals(True)
                self._z_slider.setValue(int(self._cursor[0]))
                self._z_slider.blockSignals(False)
            except Exception:
                pass

        def _render_all(self, only_changed=False):
            """Re-slice + upload the ortho panels and move the crosshairs.

            ``only_changed`` (CW2): when True, only the ortho panels whose slice
            index changed since ``_last_rendered_cursor`` are resliced +
            re-uploaded (see :func:`panels_to_reslice`) — the crosshair + title
            still move on every panel (cheap).  The default (False) redraws all
            three, so content changes (channel toggle / contrast / mode / initial
            load) stay correct.  The scrub/wheel path passes True via the
            throttle.  The volume-global ``limits`` are reused verbatim (never
            recomputed — perf invariant)."""
            if self._closing or self._vol is None or self._canvas is None:
                return
            try:
                channels_on = self._current_channels_on()
                z, y, x = self._clamped_cursor()
                self._update_title(z, y, x)
                prev = self._last_rendered_cursor if only_changed else None
                to_reslice = panels_to_reslice(prev, (z, y, x))
                for panel in _PANELS:
                    img = self._images.get(panel)
                    line = self._crosshairs.get(panel)
                    if img is None:
                        continue
                    if panel in to_reslice:
                        idx = self._panel_slice_index(panel, (z, y, x))
                        plane = plane_reslice(self._vol, panel, idx)  # (C, H, W)
                        rgb = compose_capture_rgb(
                            plane[:, None, :, :], 0, False, channels_on,
                            limits=self._limits)                       # (H, W, 3)
                        try:
                            img.set_data(rgb)
                        except Exception as exc:  # pragma: no cover - GPU/driver
                            _diag(f"set_data {panel} failed {exc}")
                    # Crosshair + slice title move on EVERY panel (cheap in-plane
                    # update, independent of whether the slice was re-fetched).
                    if line is not None:
                        try:
                            line.set_data(
                                pos=self._crosshair_pos(panel, (z, y, x)))
                        except Exception:
                            pass
                    title = self._titles.get(panel)
                    if title is not None:
                        try:
                            title.text = self._panel_title(panel, (z, y, x))
                        except Exception:
                            pass
                # 3-D quadrant: move cut planes to the cursor + apply channel
                # visibility in place (never rebuilds a Volume).
                self._sync_3d()
                try:
                    self._canvas.update()
                except Exception:
                    pass
                self._last_rendered_cursor = (z, y, x)
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"render failed {exc}")

        def _request_render(self):
            """CW1 throttle: coalesce rapid scrub-drag / wheel cursor moves into
            ONE redraw per event-loop pass.  A fast mouse fires 100+ moves/sec,
            each of which would otherwise reslice + re-upload the panels and kick
            the 3-D ray-march; the camera-follow + Z-slider stay synchronous in
            the caller (cheap), only the expensive redraw is deferred.  Guarded
            so no redraw fires after teardown (``_closing`` + the timer is stopped
            in ``_teardown_canvas``)."""
            if self._closing:
                return
            self._render_pending = True
            t = self._render_timer
            if t is not None and not t.isActive():
                t.start()

        def _on_render_timer(self):
            """Fire the coalesced redraw (CW1); only the ortho panels whose slice
            changed are resliced (``only_changed=True``)."""
            if self._closing or not self._render_pending:
                return
            self._render_pending = False
            self._render_all(only_changed=True)

        def _update_title(self, z, y, x):
            try:
                shape = tuple(int(s) for s in self._vol.shape)
                self._title.setText(
                    f"{self._patch_name}    (C,Z,Y,X)={shape}    "
                    f"cursor z={z} y={y} x={x}")
            except Exception:
                pass

        # -- Mode toggle (Point <-> Scrub) ----------------------- #
        def _on_mode_toggled(self, checked):
            self._set_mode("scrub" if checked else "point")

        def _set_mode(self, mode):
            """Switch Point <-> Scrub: cancel any in-flight drag, flip camera
            interactivity, (re)position the cameras, update the Z-slider
            visibility + the cursor icon + the button label, and redraw."""
            mode = "scrub" if mode == "scrub" else "point"
            self._mode = mode
            # Cancel any in-flight drag / pending click (UX-M2).
            self._scrub_dragging = False
            self._scrub_panel = None
            self._scrub_origin_pos = None
            self._scrub_origin_cursor = None
            self._press_pos = None
            _diag(f"mode -> {'Scrub' if mode == 'scrub' else 'Point'}")

            # Point: cameras interactive (pan/zoom).  Scrub: non-interactive so
            # left-drag routes to the scrub handler, not a camera pan (UX-H1).
            for panel in _PANELS:
                vb = self._ortho_views.get(panel)
                if vb is None:
                    continue
                try:
                    vb.camera.interactive = (mode == "point")
                except Exception:
                    pass

            if mode == "scrub":
                # Keep the current cursor; pan each panel so that voxel sits at
                # the screen centre (cursor-preserving, NOT a jump to shape//2 —
                # axes-HIGH-4).
                self._center_cameras_on_cursor()
            else:
                # Back to fill (Brick-1 look): rect = exact data bounds.
                for panel in _PANELS:
                    vb = self._ortho_views.get(panel)
                    if vb is None:
                        continue
                    try:
                        H, W = self._panel_hw[panel]
                        vb.camera.rect = (0, 0, int(W), int(H))
                    except Exception:
                        pass

            self._set_z_slider_visible(mode == "point")
            self._apply_mode_cursor()
            self._update_mode_button()
            self._render_all()

        def _update_mode_button(self):
            try:
                self._mode_btn.blockSignals(True)
                self._mode_btn.setChecked(self._mode == "scrub")
                self._mode_btn.setText(
                    "Scrub mode" if self._mode == "scrub" else "Point mode")
                self._mode_btn.blockSignals(False)
            except Exception:
                pass

        def _set_z_slider_visible(self, visible):
            """Z slider lives in Point mode; hidden in Scrub (UX-L3)."""
            for w in (getattr(self, "_z_label", None),
                      getattr(self, "_z_slider", None)):
                try:
                    if w is not None:
                        w.setVisible(bool(visible))
                except Exception:
                    pass

        def _apply_mode_cursor(self):
            """Distinct mouse cursor per mode (a visible indicator): a cross in
            Point, a size-all (move) cursor in Scrub."""
            if self._canvas is None:
                return
            try:
                native = self._canvas.native
                shape = (Qt.SizeAllCursor if self._mode == "scrub"
                         else Qt.CrossCursor)
                native.setCursor(shape)
            except Exception:
                pass

        def _center_cameras_on_cursor(self):
            """Scrub: pan each ortho camera so the cursor's in-plane voxel sits
            under the panel's screen centre, preserving the current zoom (rect
            SIZE).  Because the crosshair is drawn at the cursor's data coords,
            centring the camera there puts the crosshair at the screen centre."""
            if self._mode != "scrub" or self._vol is None:
                return
            z, y, x = self._clamped_cursor()
            for panel in _PANELS:
                vb = self._ortho_views.get(panel)
                if vb is None:
                    continue
                try:
                    col, row, _W, _H = self._panel_incol_row(panel, (z, y, x))
                    rect = vb.camera.rect
                    rw = float(rect.width)
                    rh = float(rect.height)
                    vb.camera.rect = (col - rw / 2.0, row - rh / 2.0, rw, rh)
                except Exception:
                    pass

        @staticmethod
        def _is_left_button(event):
            """vispy Qt maps the left mouse button to id 1."""
            try:
                return int(getattr(event, "button", 0)) == 1
            except Exception:
                return False

        # -- Mouse -------------------------------------------------------------
        def _on_mouse_press(self, event):
            try:
                if self._closing or self._vol is None:
                    return
                if self._mode == "scrub":
                    # Start a left-drag scrub if the press is over a panel.
                    if self._is_left_button(event):
                        hit = self._hit_test(event.pos)
                        if hit is not None:
                            panel, _row, _col = hit
                            self._scrub_panel = panel
                            self._scrub_origin_pos = (
                                float(event.pos[0]), float(event.pos[1]))
                            self._scrub_origin_cursor = list(self._cursor)
                            self._scrub_dragging = True
                    return
                # Point mode: remember the press to click-vs-pan on release.
                self._press_pos = (float(event.pos[0]), float(event.pos[1]))
            except Exception:
                self._press_pos = None
                self._scrub_dragging = False

        def _on_mouse_move(self, event):
            """Scrub left-drag: convert the pixel delta since press into a DATA
            (voxel) delta via the panel transform (zoom-independent — 1 voxel of
            drag = 1 voxel of scroll), then step the in-plane axes.

            The delta is computed as ``T(cur) - T(origin)`` through the SAME
            current transform, so the pan applied while dragging cancels out (the
            transform's linear part is constant — Scrub never zooms).  The result
            is applied to the drag-origin cursor (absolute), which avoids
            per-frame rounding drift."""
            try:
                if (self._closing or self._vol is None
                        or self._mode != "scrub" or not self._scrub_dragging):
                    return
                panel = self._scrub_panel
                img = self._images.get(panel) if panel else None
                if panel is None or img is None or self._scrub_origin_pos is None:
                    return
                try:
                    tr = img.get_transform("canvas", "visual")
                    d0 = tr.map([float(self._scrub_origin_pos[0]),
                                 float(self._scrub_origin_pos[1])])
                    d1 = tr.map([float(event.pos[0]), float(event.pos[1])])
                    d_data = (float(d1[0] - d0[0]), float(d1[1] - d0[1]))
                except Exception:
                    return
                C, Z, Y, X = self._vol.shape
                new_cursor = scrub_delta_to_cursor(
                    panel, d_data, self._scrub_origin_cursor, (Z, Y, X))
                if new_cursor != self._cursor:
                    self._cursor = new_cursor
                    _diag(f"scrub {panel} d_data=({d_data[0]:.2f},"
                          f"{d_data[1]:.2f}) -> cursor=({new_cursor[0]},"
                          f"{new_cursor[1]},{new_cursor[2]})")
                    self._sync_z_slider()
                    self._center_cameras_on_cursor()
                    self._request_render()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"mouse move failed {exc}")

        def _on_mouse_wheel(self, event):
            """Scrub wheel: step the panel-under-the-cursor's OWN slice axis
            (the primary depth gesture).  In Point mode the wheel is left
            to the camera (zoom)."""
            try:
                if self._closing or self._vol is None or self._mode != "scrub":
                    return
                try:
                    delta = event.delta
                    dy = float(delta[1]) if len(delta) > 1 else float(delta[0])
                except Exception:
                    return
                if dy == 0:
                    return
                direction = 1 if dy > 0 else -1
                hit = self._hit_test(event.pos)
                if hit is None:
                    return
                panel, _row, _col = hit
                C, Z, Y, X = self._vol.shape
                new_cursor = wheel_step(panel, direction, self._cursor, (Z, Y, X))
                if new_cursor != self._cursor:
                    self._cursor = new_cursor
                    _diag(f"wheel {panel} dir={direction} -> cursor="
                          f"({new_cursor[0]},{new_cursor[1]},{new_cursor[2]})")
                    self._sync_z_slider()
                    self._center_cameras_on_cursor()
                    self._request_render()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"mouse wheel failed {exc}")

        def _on_mouse_release(self, event):
            """Point mode: a release within ``_CLICK_PX`` of the press is a
            click.  Scrub mode: end the drag; a bare click is inert (UX-M2)."""
            try:
                if self._closing or self._vol is None:
                    return
                if self._mode == "scrub":
                    self._scrub_dragging = False
                    self._scrub_panel = None
                    self._scrub_origin_pos = None
                    self._scrub_origin_cursor = None
                    return  # bare click is inert in Scrub
                pp = self._press_pos
                self._press_pos = None
                if pp is None:
                    return
                pos = event.pos
                dx = float(pos[0]) - pp[0]
                dy = float(pos[1]) - pp[1]
                if (dx * dx + dy * dy) ** 0.5 >= _CLICK_PX:
                    return  # a pan/drag, not a click
                hit = self._hit_test(pos)
                if hit is None:
                    return
                panel, row, col = hit
                C, Z, Y, X = self._vol.shape
                self._cursor = point_click_to_cursor(
                    panel, row, col, (Z, Y, X), self._cursor)
                _diag(f"click panel={panel} row={row} col={col} -> "
                      f"cursor=({self._cursor[0]},{self._cursor[1]},"
                      f"{self._cursor[2]})")
                self._sync_z_slider()
                self._render_all()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"mouse release failed {exc}")

        def _hit_test(self, pos):
            """Which ortho panel (+ data row/col) contains canvas ``pos``.

            Inverts EACH panel image's ``canvas -> visual`` transform (which
            already accounts for that panel's on-canvas region, camera flip and
            anisotropy scale) and returns the first whose data-pixel coords fall
            inside ``[0, W) x [0, H)`` (the mandatory per-viewbox
            hit-test).  Returns ``(panel, row, col)`` or ``None``."""
            try:
                px = float(pos[0])
                py = float(pos[1])
            except Exception:
                return None
            for panel in _PANELS:
                img = self._images.get(panel)
                if img is None:
                    continue
                try:
                    tr = img.get_transform("canvas", "visual")
                    mapped = tr.map([px, py])
                    col = float(mapped[0])
                    row = float(mapped[1])
                except Exception:
                    continue
                H, W = self._panel_hw[panel]
                ci = int(np.floor(col))
                ri = int(np.floor(row))
                if 0 <= ci < W and 0 <= ri < H:
                    return panel, ri, ci
            return None

        # -- Teardown ------------------------------------------- #
        def _detach_worker(self):
            """cut the load worker's signals; never destroy a RUNNING
            QThread (that aborts with "QThread destroyed while running")."""
            w = self._worker
            self._worker = None
            if w is None:
                return
            for sig in ("done", "failed"):
                try:
                    getattr(w, sig).disconnect()
                except Exception:
                    pass
            try:
                running = bool(w.isRunning())
            except Exception:
                running = False
            if running:
                # Reparent + retain a strong ref until it self-cleans on finish;
                # a late emit is now inert (signals cut above).
                try:
                    w.setParent(None)
                except Exception:
                    pass
                _LIVE_DETACHED_WORKERS.add(w)

                def _cleanup(_w=w):
                    _LIVE_DETACHED_WORKERS.discard(_w)
                    try:
                        _w.deleteLater()
                    except Exception:
                        pass

                try:
                    w.finished.connect(_cleanup)
                    # If it finished in the gap between the isRunning() read and
                    # this connect, `finished` already fired and _cleanup won't
                    # run — clean up inline so it isn't stranded in the retain
                    # set forever (audit LOW).
                    if w.isFinished():
                        _cleanup()
                except Exception:
                    _LIVE_DETACHED_WORKERS.discard(w)
            else:
                try:
                    w.setParent(None)
                    w.deleteLater()
                except Exception:
                    pass

        def _teardown_canvas(self):
            """Teardown ORDER: (a) kill event/timer sources, (b) hide every
            visual, (c) unparent, (d) close the canvas, (e) drop refs.  No draw
            can be queued after (a); never ``processEvents()`` here."""
            c = self._canvas
            if c is None:
                # Still drop any partially-built refs.
                self._images = {}
                self._crosshairs = {}
                self._titles = {}
                self._axis_widgets = []
                self._overlay_visuals = []
                self._ortho_views = {}
                self._volumes = {}
                self._cut_meshes = {}
                self._volume_built = False
                self._vol3d_label = None
                self._vb_3d = None
                return
            # (a) disconnect canvas draw + all mouse callbacks + STOP camera
            #     timers.  The 3-D quadrant uses a TurntableCamera; the lifecycle
            #     reviewer's explicit reminder is that a running camera timer
            #     (spin/inertia) plus a freed 3-D texture is the crash window —
            #     stop any such timer BEFORE the volume textures are freed.  The
            #     3-D render-mode combo + cut-plane signals are cut here too.
            for evt in ("draw", "mouse_press", "mouse_release", "mouse_move",
                        "mouse_wheel"):
                try:
                    getattr(c.events, evt).disconnect()
                except Exception:
                    pass
            # CW1: stop the render-throttle timer so no coalesced redraw can fire
            # after the canvas + volume textures are freed (teardown step (a)).
            self._render_pending = False
            try:
                if self._render_timer is not None:
                    self._render_timer.stop()
            except Exception:
                pass
            for vb in ([self._vb_3d] + list(self._ortho_views.values())):
                try:
                    cam = getattr(vb, "camera", None)
                    tmr = getattr(cam, "_timer", None)
                    if tmr is not None:
                        tmr.stop()
                except Exception:
                    pass
            for sig in ((getattr(self, "_mode3d_combo", None),
                         "currentIndexChanged"),
                        (getattr(self, "_cut_check", None), "stateChanged")):
                try:
                    if sig[0] is not None:
                        getattr(sig[0], sig[1]).disconnect()
                except Exception:
                    pass

            visuals = (list(self._images.values())
                       + list(self._crosshairs.values())
                       + list(self._volumes.values())
                       + list(self._cut_meshes.values())
                       + list(self._overlay_visuals)
                       + list(self._titles.values())
                       + list(self._axis_widgets))
            # (b) visible = False on every visual (disconnect the vispy visual
            #     from the scene graph before any GPU resource is freed).  Volume
            #     3-D textures + the cut-plane meshes are included so no freed
            #     texture is referenced by a queued paint.
            for vis in visuals:
                try:
                    vis.visible = False
                except Exception:
                    pass
            # (c) parent = None.
            for vis in visuals:
                try:
                    vis.parent = None
                except Exception:
                    pass
            # (d) close the canvas (frees the GL context).
            try:
                c.close()
            except Exception:
                pass
            # (e) drop refs.
            self._images = {}
            self._crosshairs = {}
            self._volumes = {}
            self._cut_meshes = {}
            self._volume_built = False
            self._vol3d_label = None
            self._vb_3d = None
            self._titles = {}
            self._axis_widgets = []
            self._overlay_visuals = []
            self._ortho_views = {}
            self._canvas = None

        def _fire_on_close(self):
            """Invoke the controller's prune callback EXACTLY ONCE."""
            if self._on_close_fired:
                return
            self._on_close_fired = True
            cb = self._on_close
            self._on_close = None
            if cb is None:
                return
            try:
                cb()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"on_close callback failed {exc}")

        def closeEvent(self, event):
            # Idempotent: a second close (or a closeEvent after programmatic
            # teardown) must not re-run teardown or re-fire on_close.
            if self._closing:
                try:
                    super().closeEvent(event)
                except Exception:
                    pass
                return
            self._closing = True
            _diag("teardown begin")
            try:
                self._detach_worker()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"detach worker failed {exc}")
            try:
                self._teardown_canvas()
            except Exception as exc:  # pragma: no cover - defensive
                _diag(f"teardown canvas failed {exc}")
            self._fire_on_close()
            _diag("teardown done")
            try:
                super().closeEvent(event)
            except Exception:
                pass


def launch_patch_capture(parent, patch_name, tiff_path, channel_names,
                         voxel_size=None, on_close=None, on_loaded=None):
    """Create + show (non-modal) a :class:`PatchCaptureDialog`.

    Parameters mirror the dialog (``voxel_size`` / ``on_close`` / ``on_loaded``
    optional — the legacy 4-arg call still works).  ``on_loaded`` fires ONCE when
    the patch's data has finished loading (or failed), so a caller can clear a
    busy indicator.  Returns the dialog so the caller can keep a strong reference;
    returns ``None`` on any failure (including a headless install without Qt).
    """
    try:
        if not _QT_AVAILABLE:
            _diag("launch skipped — Qt unavailable")
            return None
        dlg = PatchCaptureDialog(parent, patch_name, tiff_path, channel_names,
                                 voxel_size=voxel_size, on_close=on_close,
                                 on_loaded=on_loaded)
        dlg.show()
        dlg.raise_()
        return dlg
    except Exception as exc:
        _diag(f"launch failed {exc}")
        return None
