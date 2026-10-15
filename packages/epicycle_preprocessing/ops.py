"""Step-1 preprocessing op math (Brick 2a) — the pure fns behind the registry.

napari-free BY DESIGN (mirrors the other ``packages/*``): no Qt, no napari, no
torch, no imports from ``napari_epicycle``.  Heavy deps (scipy / skimage) are
imported LAZILY inside the functions (same convention as
``packages/epicycle_annotator/add_segment.py``) so that importing the *registry*
stays cheap — the UI menu only needs the metadata table, not the op math.

Working-space contract
--------------------------------------------------------------------------------
* Every **INTENSITY** op (:func:`gaussian`, :func:`clahe`, :func:`background`,
  :func:`bleach`) takes a SINGLE channel's array (ZYX or 2D), assumes its input
  is already **normalized float32 in [0, 1]** (the hoisted, non-removable
  :func:`percentile_normalize` stage runs FIRST in :func:`execute`), returns
  float32 in [0, 1], and **never re-normalizes internally**.
* :func:`apply_geometry` is the ONE shared geometry step — it re-grids
  ALL channels identically on the shared spatial grid; it is NOT a per-channel
  intensity fn (it needs the spacing + every channel), so the registry's
  ``isotropic`` op carries :func:`apply_geometry` as a *marker/reference* and
  :func:`execute` routes geometry by ``kind == "geometry"`` rather than the
  per-channel fn signature.
* :func:`to_uint16` is the FIXED-scale ``[0, 1] → ×65535`` bake dtype (NOT
  per-channel min-max, which would re-stretch + undo dimming ops) — used by
  the whole-volume process (Brick 1b) + extract bake; provided + tested here.

Library choices (WRAP proven libs, don't reimplement)
-----------------------------------------------------
* :func:`gaussian`  → ``scipy.ndimage.gaussian_filter`` (lifted from
  ``packages/epicycle_reconstruction/logic.py::smooth_probs_3d``).
* :func:`clahe`     → ``skimage.exposure.equalize_adapthist`` (nD-capable;
  lifted from ``add_segment.py::add_threshold_space``).
* :func:`background` → **white top-hat** (``skimage.morphology.white_tophat``
  with a ball/disk footprint), NOT rolling-ball.  Rationale: ``rolling_ball``
  = ImageJ fidelity but is punishingly slow on 3D volumes; the morphological
  white top-hat is the accepted fast equivalent (top-hat = img − opening, which
  already IS the background-subtracted residual, clamped ≥ 0) and is the right
  default for an interactive per-channel 3D op.  We trade a little fidelity for
  large speed wins; rolling-ball can be added later as a second op.
* :func:`bleach`    → **CUSTOM** (~30 lines): per-Z-slice mean, log-linear fit
  of the exponential decay, per-slice rescale so every fitted level matches the
  **MEAN** fitted level (``mean(fitted)/fitted``, NOT the first/brightest
  slice — the latter multiplies deep slices by ``>1`` on ``[0, 1]``-normalized
  data and clips highlights).  2D / single-slice input is a graceful no-op; so
  is a non-positive / underflowed fit.  Whether a *non-decreasing* profile is
  corrected is the ``on_non_decreasing`` option (``"skip"`` default / ``"flatten"``).

Canonical geometry MODE / METHOD tokens (pinned here)
------------------------------------------------
The **stored** (recipe-config + executor) tokens are canonical and lowercase::

    mode   ∈ {"iso", "aniso"}
    method ∈ {"nearest", "linear", "cubic", "area"}

``recipe.py`` already normalizes to these, and the Brick-1 UI already writes
them (its own display↔token map: ``Isotropic/Per-axis → iso/aniso`` and
``Nearest/Linear/Cubic/Area-average → nearest/linear/cubic/area``).  The
``registry.isotropic`` ``ParamSpec`` choices are aligned to these same tokens so
the three surfaces never disagree.  Any prettier display label is a **UI**
concern (a display→token map lives in ``ui.py``); the tokens crossing the
recipe / persistence / executor boundary are always the canonical lowercase set.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .chunked import block_parallel_apply
from .exceptions import PreprocessingCancelled

# NB: :mod:`recipe` / :mod:`registry` are imported LAZILY inside :func:`execute`
# / :func:`_run_intensity_ops` (not at module top).  This keeps the import graph
# acyclic — ``registry`` wires the five fns from this module at its own import
# (``from . import ops``), so this module must be importable WITHOUT pulling
# ``registry`` back in mid-initialization.

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Channel parallelism — run independent per-channel work across cores
# --------------------------------------------------------------------------- #

#: Cap the pool so a many-channel image can't oversubscribe the box.  Channels
#: are few in practice (usually 1-5), so this rarely binds; it just protects the
#: pathological "40-channel source" case from spawning 40 threads.
_MAX_CHANNEL_WORKERS: int = min(8, os.cpu_count() or 1)


def _parallel_map(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    max_workers: int | None = None,
) -> List[Any]:
    """Map *fn* over *items* preserving order, on a thread pool when >1 item.

    The per-channel work (:func:`percentile_normalize`, the intensity ops, the
    geometry resample) is independent across channels and spends its time inside
    scipy / skimage C loops that RELEASE the GIL, so threads give a real
    multi-core speedup (measured ~3-4× on a 4-channel Gaussian) without the
    pickling / process-spawn cost of a process pool.  A single item (or none)
    runs INLINE — no pool overhead, and it keeps the single-channel path (and the
    order-sensitive monkeypatched executor tests) perfectly deterministic.

    Order is preserved (``pool.map`` yields results in input order).  *fn* must
    be self-contained + side-effect-free across items (each channel owns its own
    array); the callers here satisfy that.

    *max_workers* caps the pool width.  ``None`` → the default ``min(8, cpu)``
    (good for the small preview crops).  ``1`` runs INLINE / serial — one item's
    working set live at a time — which is how the whole-volume path (Slice 1)
    bounds peak memory on large arrays (channel-parallelism holds every channel's
    op-scratch at once, ~N× peak).
    """
    items = list(items)
    workers = min(max_workers or _MAX_CHANNEL_WORKERS, len(items))
    if workers <= 1:
        return [fn(x) for x in items]
    # CANCEL SEMANTICS.  A plain `with ThreadPoolExecutor` is correct HERE, unlike
    # `chunked.block_parallel_apply`, which manages its pool explicitly and always
    # calls `shutdown(wait=True, cancel_futures=True)`.  The difference is
    # granularity: this pool maps over CHANNELS (2-4 items, all submitted at
    # once), and each item polls `should_cancel` internally - between ops in
    # `_run_intensity_ops`, and once per tile wave inside a tiled op - so a cancel
    # is observed WITHIN the running items and `__exit__`s `shutdown(wait=True)`
    # has at most `workers` short-circuiting items to drain.  There is no queue of
    # not-yet-started work for `cancel_futures` to drop, which is precisely the
    # problem the block pool has (10^4-10^5 tiles) and the reason it is
    # hand-managed.  If this map is ever pointed at something finer-grained than
    # channels, it must adopt the explicit-pool pattern too.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def _warm_imports() -> None:
    """First-touch the heavy scipy / skimage submodules ONCE, in the calling
    thread, before any :func:`_parallel_map` fans work across the pool.

    The op fns import scipy / skimage LAZILY (the module docstring keeps the
    registry import cheap).  Without this, the first multi-channel :func:`execute`
    would first-touch those submodules concurrently from N pool threads; a
    transient concurrent-import hiccup raised inside an intensity op is swallowed
    by :func:`_run_intensity_ops`'s ``except Exception`` and would SILENTLY skip
    that op on one channel (a per-channel divergence on the training-data path).
    Warming once removes the race; it is effectively free once the modules are
    cached (a ``sys.modules`` dict hit).  Best-effort: a genuinely missing lib
    still surfaces as the pre-existing per-op skip, not a new execute() raise.
    """
    try:
        import scipy.ndimage  # noqa: F401
        import skimage.morphology  # noqa: F401
        import skimage.exposure  # noqa: F401
        import skimage.transform  # noqa: F401
    except Exception:  # noqa: BLE001 — best-effort warmup; ops re-import + guard
        pass


# --------------------------------------------------------------------------- #
# Working space — the hoisted, non-removable first stage
# --------------------------------------------------------------------------- #

def percentile_bounds(
    arr: np.ndarray, low: float = 0.1, high: float = 99.5
) -> Tuple[float, float] | None:
    """The ``(lo, hi)`` raw-intensity endpoints of :func:`percentile_normalize`'s
    linear map — factored out as the SINGLE source of truth.

    Computes the ``[low, high]`` percentiles over the FINITE voxels only.
    Returns ``None`` for a flat / empty / all-NaN / degenerate crop (``hi <= lo``
    or a non-finite percentile) so the caller decides the fallback:
    :func:`percentile_normalize` returns all-zeros; the Step-1 preview histogram
    (``controller._raw_percentile_bounds``) falls back to a min-max frame.  This
    is the ONE place the percentile / finite-filter / interpolation logic lives,
    so the preview's ``u`` frame can never silently drift off what normalize
    actually clips.  Pure — no Qt, no I/O.
    """
    a = np.asarray(arr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return lo, hi


def percentile_normalize(
    arr: np.ndarray,
    low: float = 0.1,
    high: float = 99.5,
    bounds: Tuple[float, float] | None = None,
) -> np.ndarray:
    """Clip *arr* to a ``(lo, hi)`` window and scale that window to ``[0, 1]``.

    The FIRST, non-removable stage of every channel's pipeline: it
    defines the ``[0, 1]`` domain that CLAHE / rolling-ball assume and that L2's
    ``map_overlap`` reuse requires.  **Never raises** on flat / empty / all-NaN
    input or a degenerate window (div-by-zero guarded → all-zeros).  Returns
    float32 in ``[0, 1]``.

    *bounds* — an OPTIONAL precomputed ``(lo, hi)`` window.  When given,
    it is used verbatim instead of this array's own ``[low, high]`` percentiles,
    so a REGION preview and the WHOLE-VOLUME process can normalize against the
    SAME shared window and agree (preview == process for intensity ops).  When
    ``None``, the window is computed from this array via :func:`percentile_bounds`
    (the single source of truth) — a ``None`` there is the flat / empty / all-NaN
    / degenerate case, returned here as all-zeros.  A passed-in degenerate window
    (``hi <= lo`` or non-finite) is likewise all-zeros.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return a.astype(np.float32, copy=False)
    lo = hi = None
    if bounds is not None:
        # A MALFORMED explicit window (scalar / 1-tuple / non-numeric) must NOT
        # raise out of execute() — it advertises "never raises"; fall back to the
        # array's own percentiles instead.
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError, IndexError):
            lo = hi = None
    if lo is None:
        computed = percentile_bounds(a, low, high)
        if computed is None:
            # flat / all-NaN / degenerate range → nothing to stretch; all-zeros.
            return np.zeros_like(a, dtype=np.float32)
        lo, hi = computed
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        # a caller-supplied degenerate window → nothing to stretch; all-zeros.
        return np.zeros_like(a, dtype=np.float32)
    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Exact-histogram percentile measure — streamable, order-independent
# --------------------------------------------------------------------------- #
# The streaming (out-of-core) whole-volume process cannot hold the array in RAM
# to compute the normalize window, and it reads the source block-by-block across
# threads.  A 65536-bin uint16 histogram is the EXACT, memory-light (~512 KB
# int64), order-independent alternative to a percentile over the resident array:
# each per-block histogram is summed (commutative → any read order gives the same
# total), and the percentile is reconstructed from the histogram to MATCH
# ``np.percentile(uint16_array, [low, high])`` bit-for-bit (verified — including
# the interpolated case).  This is ADDITIVE — ``execute`` / the in-memory measure
# are untouched (the controller's strided→histogram swap is a separate step).

def histogram_uint16(arr: np.ndarray) -> np.ndarray:
    """A length-65536 ``int64`` histogram of *arr*'s uint16 code values.

    ``np.bincount(arr.ravel(), minlength=65536)`` — one count per DN in
    ``[0, 65535]``.  Intended for the uint16 SOURCE the streaming measure reads;
    a non-uint16 input is rounded + clipped into the uint16 range defensively
    (the real streaming path always passes uint16 blocks).  Pure — no Qt / I/O.
    """
    flat = np.asarray(arr).reshape(-1)
    if flat.dtype != np.uint16:
        flat = np.clip(np.rint(np.asarray(flat, dtype=np.float64)), 0.0, 65535.0).astype(
            np.uint16
        )
    return np.bincount(flat, minlength=65536).astype(np.int64)


class Uint16HistogramAccumulator:
    """An order-independent running sum of per-block uint16 histograms.

    The streaming measure pass sums one :func:`histogram_uint16` per source block
    into a single 65536-bin accumulator, then derives the ``(lo, hi)`` normalize
    window from the total via :func:`percentile_bounds_from_hist`.  Summation is
    commutative, so the result is independent of the block read order (a
    correctness invariant — the parallel measure reads blocks in a nondeterministic
    order).

    NOT internally thread-safe — :meth:`add` mutates the accumulator in place; the
    streaming caller guards concurrent adds with a lock (the add itself is cheap).
    """

    __slots__ = ("_hist",)

    def __init__(self) -> None:
        self._hist = np.zeros(65536, dtype=np.int64)

    def add(self, arr: np.ndarray) -> "Uint16HistogramAccumulator":
        """Add *arr*'s uint16 histogram into the running total (in place)."""
        self._hist += histogram_uint16(arr)
        return self

    @property
    def histogram(self) -> np.ndarray:
        """The accumulated length-65536 ``int64`` histogram."""
        return self._hist


def percentile_bounds_from_hist(
    hist: np.ndarray, low: float = 0.1, high: float = 99.5
) -> Tuple[float, float] | None:
    """The ``(lo, hi)`` ``[low, high]``-percentile DN pair of a uint16 histogram.

    Reconstructs the exact linear-interpolated percentile numpy's default method
    computes: for percentile ``p`` over ``total`` values, the virtual rank is
    ``h = p/100 · (total-1)``; the sorted value at any rank ``k`` is recovered
    from the cumulative histogram as ``searchsorted(cdf, k, side="right")`` (an
    integer DN), and the result linearly interpolates between the two straddling
    ranks — so this EQUALS ``np.percentile(uint16_array, [low, high])`` bit-for-bit
    (the parity test in ``tests/data_preparation/test_streaming.py``).

    Returns ``None`` for a degenerate histogram — empty (``total == 0``) or flat
    (``hi <= lo``, a constant source) — mirroring :func:`percentile_bounds`, so a
    caller applies the same all-zeros fallback :func:`percentile_normalize` uses.
    Pure — no Qt / I/O.
    """
    h = np.asarray(hist, dtype=np.int64).reshape(-1)
    cdf = np.cumsum(h)
    total = int(cdf[-1]) if cdf.size else 0
    if total <= 0:
        return None

    def _pct(p: float) -> float:
        pos = (float(p) / 100.0) * (total - 1)
        lo_idx = int(np.floor(pos))
        frac = pos - lo_idx
        v_lo = float(np.searchsorted(cdf, lo_idx, side="right"))
        if frac == 0.0 or lo_idx + 1 >= total:
            return v_lo
        v_hi = float(np.searchsorted(cdf, lo_idx + 1, side="right"))
        return v_lo + frac * (v_hi - v_lo)

    lo = _pct(low)
    hi = _pct(high)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return lo, hi


# --------------------------------------------------------------------------- #
# Intensity ops — per channel, operate on normalized [0,1], return [0,1]
# --------------------------------------------------------------------------- #

def gaussian(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smooth (isotropic ``sigma`` in px).  ``sigma <= 0`` is a no-op.

    Thin wrap of ``scipy.ndimage.gaussian_filter``, **clipped back to
    ``[0, 1]``** so the op honours the section contract (normalized in,
    normalized out) for EVERY input it can actually receive.

    **Why the clip is not removable** (Brick 3a #4 removed it; Brick 3c #1 puts
    it back).  On ``[0, 1]`` input the filter is a convex combination and the
    clip is provably dead - that much was measured and is still asserted by
    `test_gaussian_on_in_range_input_equals_the_clipped_form`.  But the gaussian
    does NOT only see ``[0, 1]`` input: `apply_geometry` deliberately does not
    clip its resample, and ``method="cubic"`` OVERSHOOTS at sharp edges (measured
    range ``[-0.12, 1.23]`` on a step-edge volume), so a post-geometry gaussian
    is handed genuinely out-of-range data.  Its output then also leaves
    ``[0, 1]``, and the NEXT op need not be convex - ``background``
    (white-top-hat) reads those values through a morphological opening, so the
    error does not wash out: measured up to ~11000 DN difference in the uint16
    bake on a cubic-resize -> gaussian -> background recipe.  `execute`'s
    trailing clip cannot help - it runs after the whole chain.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(arr, dtype=np.float32)
    s = float(sigma)
    if s <= 0.0:
        return a.astype(np.float32, copy=False)
    out = gaussian_filter(a, sigma=s)
    return np.clip(out, 0.0, 1.0, out=out).astype(np.float32, copy=False)


def clahe(arr: np.ndarray, clip: float = 0.01, kernel: int = 64) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization (nD).

    Thin wrap of ``skimage.exposure.equalize_adapthist`` (input already
    ``[0, 1]``).  A degenerate (constant) input is returned unchanged — CLAHE is
    undefined there and skimage would map it to all-ones.
    """
    from skimage.exposure import equalize_adapthist

    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    finite = a[np.isfinite(a)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        return a.astype(np.float32, copy=False)  # constant / empty → no-op
    # Clamp the tile size per axis so it never exceeds the array dimensions.
    k = int(kernel)
    if k < 1:
        k = 1
    kernel_size = tuple(int(min(k, max(1, dim))) for dim in a.shape)
    out = equalize_adapthist(a, kernel_size=kernel_size, clip_limit=float(clip))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def background(arr: np.ndarray, radius: int = 15) -> np.ndarray:
    """Background subtraction via a morphological **white top-hat** (ball SE).

    ``white_tophat = img − grey_opening(img, ball(radius))`` keeps bright
    features SMALLER than the structuring element and removes the slowly-varying
    background (a flat / gradient background region → ~0).  The residual is
    already ``≥ 0``; on ``[0, 1]`` input it stays in ``[0, 1]``.  See the module
    docstring for why top-hat is chosen over rolling-ball (speed).

    **Footprint decomposition (perf, measured):** a SOLID ``ball(radius)`` in 3D
    is a non-separable neighborhood with ``O(N · radius³)`` cost — at the default
    ``radius=15`` a single preview-crop channel took ~4 MINUTES.  skimage's
    ``decomposition="sequence"`` approximates the ball with a short sequence of
    tiny structuring elements, cutting that to ~2 s (**~135× faster**) with a
    ``max|Δ|`` of ~0.02 on ``[0, 1]`` vs the solid ball — a negligible shape
    difference for background estimation.  The approximation is the right default
    for an interactive per-channel 3D op; an exact-ball option can be added later.
    """
    from skimage.morphology import white_tophat, ball, disk

    a = np.asarray(arr, dtype=np.float32)
    r = int(radius)
    if r <= 0:
        return a.astype(np.float32, copy=False)
    if a.ndim == 2:
        footprint = disk(r, decomposition="sequence")
    elif a.ndim == 3:
        footprint = ball(r, decomposition="sequence")
    else:
        # Fall back to a per-axis box for exotic ndim (undecomposed).
        footprint = np.ones((2 * r + 1,) * a.ndim, dtype=bool)
    out = white_tophat(a, footprint=footprint)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def bleach(
    arr: np.ndarray, model: str = "exp", on_non_decreasing: str = "skip"
) -> np.ndarray:
    """Depth/photo-bleach correction — flatten a per-Z-slice intensity decay.

    ImageJ-style, ``exp`` model to start.  Fit the per-slice mean decay
    log-linearly (``log(mean) = intercept + slope·z``) and rescale each slice so
    its fitted level matches the **MEAN** fitted level, then clip to ``[0, 1]``.
    The reference is the mean — NOT the first/brightest slice: on
    ``[0, 1]``-normalized data ``fitted[0]/fitted`` multiplies deep slices by
    ``> 1`` and clips highlights, whereas ``mean(fitted)/fitted`` boosts the dim
    slices and dims the bright ones symmetrically.  A residual ``clip(0, 1)``
    still runs (the working-space contract), so a small amount of clipping at the
    extremes is expected; the mean reference simply never overshoots MORE than
    the old first-slice reference.

    2D / single-slice input (no Z axis to correct) is a no-op, as is a
    non-positive fit or one that underflows to ``0`` at depth (see below).  Only
    ``"exp"`` is supported in v1; any other ``model`` is a no-op.

    ``on_non_decreasing`` controls what happens when the fitted profile is NOT
    decreasing with depth (``slope >= 0`` — flat or brightening, which is not a
    bleach signature):

    * ``"skip"`` (default) — return the input unchanged and emit a status line
      (``"[bleach] Z profile not decreasing — no correction applied"``).
    * ``"flatten"`` — apply the mean-reference rescale in BOTH directions anyway
      (the pre-fix behavior), flattening a brightening profile too.

    Divide-by-zero guard: a steep negative slope can make ``exp(intercept +
    slope·z)`` **underflow to 0.0** for large ``z`` — still *finite*, so a bare
    finiteness check passes, but the ensuing division yields ``inf``/``NaN``.
    We therefore require the WHOLE fitted profile to be finite AND ``> 0`` before
    dividing; otherwise we skip (return the input) with a status line.
    """
    a = np.asarray(arr, dtype=np.float32)
    if str(model).lower() != "exp":
        return a.astype(np.float32, copy=False)
    if a.ndim < 3 or a.shape[0] < 2:
        return a.astype(np.float32, copy=False)  # 2D / single slice → nothing to flatten

    z = a.shape[0]
    means = a.reshape(z, -1).mean(axis=1)
    valid = np.isfinite(means) & (means > 0)
    if int(valid.sum()) < 2:
        return a.astype(np.float32, copy=False)  # can't fit a decay

    zs = np.arange(z, dtype=np.float64)
    try:
        slope, intercept = np.polyfit(zs[valid], np.log(means[valid]), 1)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return a.astype(np.float32, copy=False)
    fitted = np.exp(intercept + slope * zs)

    # Divide-by-zero guard (defect 3): a steep negative slope underflows exp() to
    # 0.0 at depth (finite → an isfinite-only check passes) → the division below
    # would produce inf/NaN.  Require the ENTIRE fitted profile finite AND > 0.
    if not np.all(np.isfinite(fitted)) or not np.all(fitted > 0.0):
        logger.info(
            "[bleach] fitted Z profile has non-positive / underflowed levels "
            "— no correction applied"
        )
        return a.astype(np.float32, copy=False)

    # Non-decreasing guard (defect 2): only a profile that DIMS with depth is a
    # bleach signature.  With the log-linear fit, "decreasing with depth" ⇔
    # slope < 0.  A flat/brightening fit (slope >= 0) is handled per the option.
    if slope >= 0.0 and str(on_non_decreasing).lower() != "flatten":
        logger.info("[bleach] Z profile not decreasing — no correction applied")
        return a.astype(np.float32, copy=False)

    # Rescale every slice to the MEAN fitted level (defect 1) — not the
    # first/brightest slice.  A residual clip(0, 1) still applies.
    correction = (float(np.mean(fitted)) / fitted).astype(np.float32)
    correction = correction.reshape((z,) + (1,) * (a.ndim - 1))  # broadcast over Y,X(,…)
    out = a * correction
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Geometry — the ONE shared step (all channels, shared grid)
# --------------------------------------------------------------------------- #

#: Canonical geometry ``method`` token → ``scipy.ndimage.zoom`` spline order.
#: ``"area"`` is not a spline; it is handled by block-mean downscaling and only
#: falls back to order-1 zoom, so it maps to 1 here for that fallback.
_METHOD_ORDER: Dict[str, int] = {
    "nearest": 0,
    "linear": 1,
    "cubic": 3,
    "area": 1,
}


def method_to_order(method: str) -> int:
    """Map a canonical geometry ``method`` token to a ``zoom`` spline order.

    ``nearest → 0``, ``linear → 1``, ``cubic → 3``, ``area → 1`` (the fallback
    order area uses when it cannot block-mean).  Unknown tokens → ``1`` (linear).
    """
    return _METHOD_ORDER.get(str(method).lower(), 1)


def _resolve_factors(
    config: Dict[str, Any], spacing_axes: Tuple[float, ...], ndim: int
) -> Tuple[float, ...]:
    """Per-axis zoom factors for the resample, aligned to ``ndim`` trailing axes.

    ``iso`` → ``factor[axis] = spacing[axis] / target_um`` (an isotropic
    ``target_um`` voxel size; ``spacing > target`` ⇒ upsample, factor > 1).
    ``aniso`` → the config's ``(z, y, x)`` factors directly.
    """
    mode = str(config.get("mode", "iso")).lower()
    if mode == "aniso":
        raw = config.get("factors", (1.0,) * ndim)
        try:
            fseq = tuple(float(f) for f in raw)
        except (TypeError, ValueError):
            fseq = (1.0,) * ndim
        return _align_axes(fseq, ndim, fill=1.0)
    # isotropic
    try:
        target = float(config.get("target_um", 1.0))
    except (TypeError, ValueError):
        target = 1.0
    if not np.isfinite(target) or target <= 0.0:
        target = 1.0
    return tuple((s / target) if s > 0 else 1.0 for s in spacing_axes)


def _align_axes(
    seq: Sequence[float], ndim: int, fill: float = 1.0
) -> Tuple[float, ...]:
    """Return the trailing ``ndim`` entries of *seq* (front-padded with *fill*)."""
    seq = tuple(seq)
    if len(seq) >= ndim:
        return seq[-ndim:]
    return (fill,) * (ndim - len(seq)) + seq


def _resample_one(
    arr: np.ndarray, factors: Tuple[float, ...], method: str
) -> np.ndarray:
    """Resample a single array by *factors* using *method* (see :func:`method_to_order`)."""
    from scipy.ndimage import zoom

    m = str(method).lower()
    if m == "area" and any(f < 1.0 for f in factors):
        if all(f <= 1.0 for f in factors):
            from skimage.transform import downscale_local_mean

            blocks = tuple(
                max(1, int(round(1.0 / f))) if f < 1.0 else 1 for f in factors
            )
            return downscale_local_mean(arr, blocks).astype(np.float32)
        # Mixed up/down-scale → area block-mean can't upscale; fall back to linear.
        return zoom(arr, factors, order=1).astype(np.float32)
    return zoom(arr, factors, order=method_to_order(m)).astype(np.float32)


def apply_geometry(
    channels: Sequence[np.ndarray],
    spacing_zyx: Sequence[float],
    config: Dict[str, Any],
    *,
    max_workers: int | None = None,
) -> Tuple[List[np.ndarray], Tuple[float, ...]]:
    """Resample ALL *channels* identically on the shared spatial grid.

    ``config`` = ``{mode: "iso"|"aniso", target_um, factors:(z,y,x), method}``
    (canonical tokens — module docstring).  Returns ``(new_channels,
    new_spacing_zyx)``; every channel is re-gridded with the SAME factors so the
    channels stay co-registered.  Physical extent is preserved, so the new
    spacing is derived from the achieved shape change.  A no-op (all factors ≈ 1)
    returns float32 copies + the original spacing.  Preserves float32 dtype; does
    NOT clip (a pure resampler — :func:`execute` owns the ``[0, 1]`` contract).
    """
    chans = [np.asarray(c, dtype=np.float32) for c in channels]
    spacing = tuple(float(s) for s in spacing_zyx)
    if not chans:
        return [], spacing

    ndim = chans[0].ndim
    spacing_axes = _align_axes(spacing, ndim, fill=1.0)
    factors = _resolve_factors(config, spacing_axes, ndim)

    if all(abs(f - 1.0) < 1e-9 for f in factors):
        return [c.copy() for c in chans], spacing

    method = str(config.get("method", "linear"))
    resampled = _parallel_map(
        lambda c: _resample_one(c, factors, method), chans, max_workers=max_workers
    )

    old_shape = chans[0].shape
    new_shape = resampled[0].shape
    new_spacing_axes = tuple(
        spacing_axes[i] * old_shape[i] / new_shape[i]
        if new_shape[i]
        else spacing_axes[i]
        for i in range(ndim)
    )
    lead = spacing[: len(spacing) - ndim] if len(spacing) > ndim else ()
    return resampled, tuple(lead) + new_spacing_axes


# --------------------------------------------------------------------------- #
# Bake dtype — per-channel min-max → uint16
# --------------------------------------------------------------------------- #

def to_uint16(arr_1ch: np.ndarray) -> np.ndarray:
    """Fixed-scale ``[0, 1] → ×65535`` clip to ``uint16``.

    A DETERMINISTIC map of the working-space ``[0, 1]`` domain to the full
    ``uint16`` range: ``0.0 → 0``, ``1.0 → 65535``, ``0.5 → 32768``.  Values
    outside ``[0, 1]`` CLIP (so ``-inf`` / ``< 0 → 0`` and ``+inf`` / ``> 1 →
    65535``); ``NaN → 0``.  These branches are NOT decorative: ``np.clip``
    PROPAGATES NaN (``clip(nan, 0, 1) is nan``), so ``execute``'s trailing clip
    does not remove it - a NaN anywhere in the source survives
    ``percentile_normalize`` and the whole recipe and arrives here.  Without the
    ``NaN → 0`` branch, ``astype(uint16)`` on a NaN is UNDEFINED (it happens to
    give 0 on x86-64 today, but that is a platform accident, not a contract).

    **Why NOT per-channel min-max (the old behavior):** a per-channel min-max
    RE-STRETCHES every channel to fill 0..65535, which is a SECOND normalization
    on top of the recipe's own ``percentile_normalize`` — it silently UNDOES any
    op that dims the channel (background subtraction, gamma < 1), so the bake
    "looks like the op did nothing," and it makes the preview (float ``[0, 1]``)
    disagree with the bake.  The fixed scale preserves exactly what the recipe
    produced and keeps preview == process.  A uniformly dim channel stays
    dim; a genuinely constant channel maps to its constant's code (not zeroed).

    **Single-pass form (perf, byte-identical).**  The literal reading —
    ``round(clip(nan_to_num(a), 0, 1) * 65535)`` — allocates and walks FOUR
    full-size float32 arrays; measured 1.17 s for one 143x1024x1024 channel, of
    which ``nan_to_num`` alone is most (it materialises a copy just to replace
    NaN).  ``np.clip`` already maps ``-inf -> 0`` and ``+inf -> 65535``, so
    ``nan_to_num`` is needed for **NaN only**.  NaN is RARE on the real path but
    it is NOT impossible - `execute`'s trailing clip cannot remove it, since
    ``np.clip`` propagates NaN - so the branch is kept and merely made
    CONDITIONAL, not dropped.  Probe once with a cheap ``np.isnan(...).any()``
    (0.07 s) and take the ``nan_to_num`` branch only when it fires (paying the
    old cost exactly in the case that needs it); otherwise clip into ONE scratch
    buffer and do the
    scale + ``rint`` IN PLACE.  ``np.rint`` is what ``np.round`` calls (both
    round-half-to-even), so the result is bit-for-bit the old one — asserted by
    `test_preprocessing_ops.py` against a verbatim copy of the old expression on
    random float32 including exact 0.0 / 1.0 / +-inf / NaN.  Measured
    1.17 s -> 0.40 s per channel (2.9x).
    """
    a = np.asarray(arr_1ch, dtype=np.float32)
    if a.size and np.isnan(a).any():
        # Rare / defensive: NaN -> 0 exactly as before, then the same fast tail.
        a = np.nan_to_num(a, nan=0.0)
    out = np.clip(a, 0.0, 1.0)          # the ONE scratch buffer (also kills +-inf)
    np.multiply(out, 65535.0, out=out)
    np.rint(out, out=out)
    return out.astype(np.uint16)


# --------------------------------------------------------------------------- #
# halos — the op's TRUE neighborhood reach for block_parallel_apply (chunked.py).
# callable(**params) -> int.  MUST be the op's real reach, or the chunked result
# seams (a FAILED array_equal gate); see chunked.py.
# --------------------------------------------------------------------------- #

def _halo_gaussian(sigma: float = 1.5, **_: Any) -> int:
    """The Gaussian's TRUE reach — scipy ``gaussian_filter`` truncates the kernel
    at ``truncate=4.0`` σ, so weights are exactly zero beyond ``int(4σ + 0.5)``.

    Empirically verified (py312 env): halo ``int(4σ+0.5)`` → ``array_equal`` (0.0)
    chunked-vs-whole; the old ``ceil(3σ)`` is UNDERSIZED for every σ and seams
    (σ=3 → ``max|Δ|`` 1.2e-4).
    """
    return int(4.0 * float(sigma) + 0.5)


def _halo_background(radius: int = 15, **_: Any) -> int:
    """The white-top-hat's TRUE reach = ``2·radius``.

    ``white_tophat = img − opening`` and ``opening = erosion ∘ dilation``; each of
    the two morphology passes reaches ``radius``, so the composed reach is ``2r``.

    Empirically verified (py312 env): halo ``2r`` → ``array_equal`` (0.0) — and
    ``2r`` is the MINIMAL such halo (r=6 first matches at exactly 12); the old
    ``r`` is undersized ×2 and seams (r=6 → ``max|Δ|`` 1.65e-2).  Mirrors the op's
    own ``r = int(radius)`` footprint radius (truncation), so the halo tracks the
    exact footprint the op builds.
    """
    return 2 * int(float(radius))


def _halo_clahe(kernel: int = 64, **_: Any) -> int:
    """~one CLAHE tile — a conservative bound for tile-boundary continuity.

    UNUSED for chunking in this phase: CLAHE is ``chunk_class="adaptive"`` and
    runs WHOLE-array (never chunked here — chunked CLAHE is never bit-identical).
    Retained as the declared reach for a future seam-verified path.
    """
    return int(round(float(kernel)))


def max_local_tile_scratch_bytes(
    recipe_json: Any, *, dtype_bytes: int = 4, factor: int = 3
) -> int:
    """Conservative per-tile scratch (bytes) for the block-parallel path.

    Walks *recipe_json* for the largest LOCAL op (gaussian / background) and
    returns ``(block + 2·halo)³ · dtype_bytes · factor`` for it — the peak
    transient a single tile holds (input read + op scratch + result; the measured
    scipy / skimage factor is ≈3).  Lets the whole-volume caller size
    the block-pool width so ``workers × this ≤ RAM budget`` — i.e. run with FEWER
    cores instead of refusing on a RAM-tight machine.

    Returns 0 when the recipe has no local op (nothing tiles → no block scratch).
    Never raises (a malformed recipe → 0 → caller falls back to the default width).
    """
    from . import registry as _registry
    from .chunked import _adaptive_block_edge

    best = 0
    channels = recipe_json.get("channels", []) if isinstance(recipe_json, dict) else []
    for ch in channels or []:
        for step in ch or []:
            if not isinstance(step, dict):
                continue
            key = step.get("key")
            op = _registry.find(key) if key else None
            if (
                op is None
                or getattr(op, "chunk_class", None) != "local"
                or not callable(getattr(op, "halo", None))
            ):
                continue
            try:
                halo = int(op.halo(**(step.get("params") or {})))
            except Exception:  # noqa: BLE001 — a bad param → skip this op, not raise
                continue
            if halo <= 0:
                continue
            edge = _adaptive_block_edge(halo)
            tile = int(edge + 2 * halo)
            best = max(best, tile ** 3 * int(dtype_bytes) * int(factor))
    return int(best)


# --------------------------------------------------------------------------- #
# execute — the in-RAM pipeline (pure: arrays in, arrays out; no I/O, no bake)
# --------------------------------------------------------------------------- #

class _ProgressCtx:
    """Threads op-boundary + tile progress EVENTS to a caller callback.

    Emits two event kinds (plain dicts, no Qt) as the pipeline runs:
      * ``{"kind":"op", "op":key, "channel":ci, "n_channels":n, "index":i,
           "total":n_ops, "tiled":bool}`` — at the START of each op.
      * ``{"kind":"tile", "done":d, "total":t}`` — per wave inside a TILED op.

    The caller (the whole-volume worker) maps these to a cost-weighted percent +
    a plain step label.  Only constructed when the caller passes ``on_progress``
    (i.e. the whole-volume process, which runs channels serially so ``index`` is
    monotonic); ``None`` on the preview / serial-channel paths, so it costs
    nothing there.  Pure — no Qt / napari."""

    def __init__(self, on_progress, n_channels, total_ops):
        self._cb = on_progress
        self.n_channels = int(n_channels)
        self.total_ops = int(total_ops)
        self._i = 0
        # Channels may now run CONCURRENTLY when the recipe has a whole-array
        # adaptive op (Brick 3a #6), so the running index needs a lock — without
        # it two channels can read-modify-write `_i` to the same value and the
        # percent goes backwards.
        self._lock = threading.Lock()

    def op(self, key, channel_index, tiled):
        # Capture the index INSIDE the lock.  Reading `self._i` after releasing it
        # can observe the OTHER concurrent channel's increment, so two events
        # report the same index (or one is skipped) and the percent jitters.
        with self._lock:
            self._i += 1
            index = self._i
        if self._cb is not None:
            self._cb({
                "kind": "op", "op": key, "channel": channel_index,
                "n_channels": self.n_channels, "index": index,
                "total": self.total_ops, "tiled": bool(tiled),
            })

    def tile_cb(self):
        cb = self._cb
        if cb is None:
            return None
        return lambda done, total: cb({"kind": "tile", "done": done, "total": total})


#: Measured peak transient of ``skimage.exposure.equalize_adapthist`` as a
#: multiple of its float32 input (py312 env, k=64): 17.8x at 42 MB, 17.5x at
#: 118 MB, 15.3x at 600 MB.  CLAHE is the one op that is never chunked (chunking
#: it is not bit-identical), so it is also the one op whose peak is
#: proportional to the WHOLE channel — hence the deliberately conservative 20x.
_ADAPTIVE_PEAK_FACTOR: int = 20

#: Never run more than this many channels' whole-array adaptive ops at once, no
#: matter how much RAM is free — 2 already recovers most of the serial loss and
#: keeps the worst-case transient predictable.
_ADAPTIVE_MAX_CONCURRENT: int = 2


def _recipe_has_adaptive_op(norm_recipe: Any) -> bool:
    """True iff any channel runs a whole-array ``chunk_class == "adaptive"`` op.

    Those (CLAHE today) are the ops :func:`_run_intensity_ops` can NEVER route
    through the block pool, so they are the only ones that leave the cores idle
    while one channel is processed.
    """
    from . import registry as _registry

    for ch in getattr(norm_recipe, "channels", None) or []:
        for step in ch or []:
            op = _registry.find(getattr(step, "key", None) or "")
            if op is not None and getattr(op, "chunk_class", None) == "adaptive":
                return True
    return False


def adaptive_channel_workers(
    chans: Sequence[np.ndarray],
    *,
    cap: int = _ADAPTIVE_MAX_CONCURRENT,
    budget_fraction: float = 0.25,
) -> int:
    """How many channels may run their whole-array ADAPTIVE ops concurrently.

    The safe half of the CLAHE fix (Brick 3a #6).  CLAHE is routed whole-array,
    single call, and scales only ~3.8x internally, so a 2-channel stack pays it
    twice back-to-back (~36 s/channel on the wound stack).  Running the CHANNELS
    concurrently is BYTE-IDENTICAL — each channel owns its own array and the op
    is unchanged — unlike per-Z-slab chunking, which is NOT, and is
    deliberately not done here.

    The cost is RAM: the measured peak is ~:data:`_ADAPTIVE_PEAK_FACTOR` x the
    channel's float32 size, so a second concurrent channel needs another
    ~20x one channel.  Returns ``cap`` only when that EXTRA transient fits
    ``budget_fraction`` of available RAM; otherwise ``1`` (serial, exactly as
    today).  Returns ``1`` when psutil is unavailable — never guesses.  Pure /
    never raises.
    """
    n = len(chans)
    if n < 2:
        return 1
    try:
        import psutil

        avail = float(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 — no psutil / probe failed → stay serial
        return 1
    try:
        per_channel = max(
            int(np.asarray(c).size) * 4 for c in chans
        )
    except Exception:  # noqa: BLE001
        return 1
    if per_channel <= 0 or avail <= 0:
        return 1
    extra_per_worker = float(per_channel) * _ADAPTIVE_PEAK_FACTOR
    fits = int((budget_fraction * avail) // extra_per_worker)
    # `fits` is how many EXTRA concurrent channels the budget allows on top of
    # the one that runs anyway.
    return max(1, min(int(cap), n, 1 + fits))


def _run_intensity_ops(
    arr: np.ndarray,
    steps: Sequence[Any],
    *,
    parallel: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    block_workers: int | None = None,
    progress: "_ProgressCtx | None" = None,
    channel_index: int | None = None,
) -> np.ndarray:
    """Run an ordered list of :class:`recipe.IntensityStep` on *arr*.

    Skips + diagnoses (never raises) a step whose key isn't a runnable intensity
    op or whose fn raises.

    *parallel* — when ``True``, an op whose registry ``chunk_class == "local"``
    (a bounded-neighborhood op with a declared ``halo`` — gaussian, background)
    is run through :func:`chunked.block_parallel_apply` (block-parallel across
    cores, BYTE-IDENTICAL to the whole-array call).  ``block_parallel_apply``
    itself no-ops to a whole-array call on a small array, so the result is the
    same regardless.  Non-local ops (CLAHE ``adaptive``, bleach ``global_stat``)
    ALWAYS run whole-array — chunking them is not bit-identical / not applicable.
    Default ``False`` → every op runs whole-array exactly as before (byte-identical
    to the pre-parallel path).
    """
    from . import registry as _registry

    for step in steps:
        # Between-op cancel check — OUTSIDE the per-op try/except below so a cancel
        # is never swallowed (it is BaseException anyway, but keep the seam explicit
        # so a cancel between two whole-array ops — e.g. CLAHE then bleach — stops
        # promptly without entering the next op).
        if should_cancel is not None and should_cancel():
            raise PreprocessingCancelled("cancelled between ops")
        key = getattr(step, "key", None)
        op = _registry.find(key) if key else None
        if op is None or not op.implemented or op.fn is None or op.kind != "intensity":
            logger.warning(
                "preprocessing.execute: skipping non-runnable step %r", key
            )
            continue
        params = getattr(step, "params", {}) or {}
        tiled = bool(
            parallel
            and getattr(op, "chunk_class", None) == "local"
            and callable(getattr(op, "halo", None))
        )
        if progress is not None:
            progress.op(key, channel_index, tiled)
        try:
            if tiled:
                halo = int(op.halo(**params))
                boundary = getattr(op, "boundary", "none")
                op_fn = op.fn
                arr = block_parallel_apply(
                    lambda a, _fn=op_fn, _p=params: _fn(a, **_p),
                    arr,
                    halo,
                    boundary=boundary,
                    max_workers=block_workers,
                    should_cancel=should_cancel,
                    on_progress=(progress.tile_cb() if progress is not None else None),
                )
            else:
                arr = op.fn(arr, **params)
        except Exception:  # noqa: BLE001 — skip + diagnose, don't kill the pipeline
            logger.exception(
                "preprocessing.execute: op %r failed on shape %s; skipping",
                key,
                getattr(arr, "shape", None),
            )
            continue
        arr = np.asarray(arr, dtype=np.float32)
    return arr


def execute(
    recipe: Any,
    channels_data: Sequence[np.ndarray],
    spacing_zyx: Sequence[float],
    *,
    max_workers: int | None = None,
    bounds_per_channel: Sequence[Tuple[float, float] | None] | None = None,
    parallel: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    block_workers: int | None = None,
    on_progress: Callable[[Dict[str, Any]], None] | None = None,
) -> Tuple[List[np.ndarray], Tuple[float, ...], Dict[str, Any]]:
    """Run *recipe* over the in-memory *channels_data*.

    For EACH channel: :func:`percentile_normalize` FIRST (hoisted, non-removable
    working space), then that channel's intensity ops in order — with the ONE
    shared geometry step applied to ALL channels together at its shared
    ``boundary_index`` (via :func:`apply_geometry`, once, re-gridding every
    channel identically).  This realizes the flatten
    ``[pre-geometry intensity…, geometry, post-geometry intensity…]`` while
    keeping geometry a single shared-grid operation.

    Pure — takes in-memory arrays, returns arrays (no I/O, no uint16 bake).
    Returns ``(processed_channels, new_spacing, stamp)`` where
    ``processed_channels`` are float32 in ``[0, 1]`` and ``stamp`` records the
    normalized recipe json + the intended bake dtype / fixed-scale mode.

    *bounds_per_channel* — an OPTIONAL per-channel ``(lo, hi)`` normalization
    window, one entry per channel (or ``None`` to compute from the data).
    Threaded into each channel's hoisted :func:`percentile_normalize`, so a
    REGION preview and the WHOLE-VOLUME process normalize against the SAME shared
    whole-volume window and agree (preview == process for intensity ops).  A
    shorter/longer list or a ``None`` entry falls back to per-array percentiles
    for that channel.

    *parallel* — when ``True``, LOCAL intensity ops (registry ``chunk_class ==
    "local"`` — gaussian, background) run block-parallel across all cores via
    :func:`chunked.block_parallel_apply`, BYTE-IDENTICAL to the whole-array call
    (verified ``array_equal``).  This is the multi-core speed win for the
    whole-volume process; it does NOT change the output.  CHANNELS still run
    per-``max_workers`` (the whole-volume caller keeps ``max_workers=1`` → serial
    channels, memory-bounded; the block pool saturates cores per op, so there is
    no channel × block oversubscription).  CLAHE (``adaptive``), bleach
    (``global_stat``), normalize + geometry ALWAYS run whole-array.  Default
    ``False`` → every op runs whole-array (byte-identical to the pre-parallel path).

    Corrupt/edge recipes never raise — the recipe is passed through
    :func:`recipe.normalize_recipe` first (drops unknown / not-yet-implemented
    ops, clamps channel count, coerces params).
    """
    from . import recipe as _recipe

    chans = [np.asarray(c, dtype=np.float32) for c in channels_data]
    n = len(chans)
    if n > 1 or parallel:
        # Warm the lazy scipy/skimage imports once (serially) before any pool
        # first-touches them concurrently (see _warm_imports).  Cover BOTH the
        # channel pool (n>1) AND the per-op block pool (parallel=True) — a
        # single-channel parallel run would otherwise first-touch scipy/skimage
        # concurrently across block threads, and a swallowed concurrent-import
        # hiccup would silently skip an op on an unprocessed channel.
        _warm_imports()
    norm_recipe = _recipe.normalize_recipe(recipe, n)
    spacing = tuple(float(s) for s in spacing_zyx)

    geom = norm_recipe.geometry
    boundary = int(geom.boundary_index) if geom is not None else None

    bounds = list(bounds_per_channel) if bounds_per_channel is not None else None

    def _bounds_for(ci: int) -> Tuple[float, float] | None:
        return bounds[ci] if bounds is not None and ci < len(bounds) else None

    def _ops_for(ci: int) -> List[Any]:
        return list(norm_recipe.channels[ci]) if ci < len(norm_recipe.channels) else []

    # Progress plumbing (whole-volume path only — `on_progress` is None on the
    # preview / serial-channel paths, so `progress` stays None + costs nothing).
    # total_ops = per channel (1 normalize + its intensity ops) + 1 shared geometry.
    total_ops = sum(1 + len(_ops_for(ci)) for ci in range(n)) + (1 if geom is not None else 0)
    progress = (
        _ProgressCtx(on_progress, n, total_ops) if on_progress is not None else None
    )

    # CHANNEL pool width + the BLOCK pool width the stages hand down.
    #
    # Normally exactly what the caller asked for: the whole-volume process passes
    # `max_workers=1` so channels run serially and the BLOCK pool owns the cores.
    # The one exception (Brick 3a #6): a recipe with a whole-array ADAPTIVE op
    # (CLAHE) can never use the block pool, so a serial 2-channel run pays that op
    # twice back-to-back on essentially one core.  Widen to at most
    # `_ADAPTIVE_MAX_CONCURRENT` channels - byte-identical (each channel owns its
    # own array) - and ONLY when the measured ~20x-per-channel transient fits the
    # RAM budget; otherwise serial exactly as before.
    #
    # Two corrections from the Brick 3c audit:
    #
    #  * The gate is `max_workers == 1` EXPLICITLY, not `(max_workers or 1) <= 1`.
    #    `None` means "the caller has no opinion" (the preview path,
    #    `_compute_preview`, passes nothing) and already defaults to a WIDE
    #    channel pool - it must not be pulled into this branch.  Only the
    #    whole-volume process asks for serial channels, and it is the only caller
    #    this widening is for.
    #
    #  * The block pool is DIVIDED, not duplicated.  `stage_workers` applies to
    #    the WHOLE of stage 1 / stage 3, not just to the adaptive op: a widened
    #    run therefore has both channels inside `_run_intensity_ops` at once, and
    #    each would otherwise open a `block_parallel_apply` pool of the FULL
    #    `block_workers` - 2x the tile scratch `_derive_block_workers` budgeted.
    #    Handing the stages `block_workers // stage_workers` keeps the product,
    #    and therefore the peak block scratch, exactly where the caller sized it.
    stage_workers = max_workers
    stage_block_workers = block_workers
    if max_workers == 1 and n > 1 and _recipe_has_adaptive_op(norm_recipe):
        stage_workers = adaptive_channel_workers(chans)
        if stage_workers > 1:
            if block_workers is not None:
                stage_block_workers = max(1, int(block_workers) // stage_workers)
            logger.debug(
                "preprocessing.execute: adaptive op present -> %d channels "
                "concurrently (RAM budget allows); block pool %s -> %s",
                stage_workers, block_workers, stage_block_workers,
            )

    # Stage 1 - per-channel normalize + intensity ops BEFORE the geometry boundary
    # (independent across channels -> run them across cores; see _parallel_map).
    def _stage1(ci: int) -> np.ndarray:
        # Between-channel cancel seam: the whole-volume path runs channels inline
        # (max_workers=1), so `execute` never touches a channel between them — the
        # check must live HERE, at the top of the per-channel work.
        if should_cancel is not None and should_cancel():
            raise PreprocessingCancelled("cancelled")
        if progress is not None:
            progress.op("normalize", ci, False)
        arr = percentile_normalize(chans[ci], bounds=_bounds_for(ci))
        ch_ops = _ops_for(ci)
        return _run_intensity_ops(
            arr, ch_ops if geom is None else ch_ops[:boundary],
            parallel=parallel, should_cancel=should_cancel,
            block_workers=stage_block_workers,      # divided when channels widen
            progress=progress, channel_index=ci,
        )

    pre: List[np.ndarray] = _parallel_map(_stage1, range(n), max_workers=stage_workers)

    # Stage 2 — the ONE shared geometry step, applied to every channel at once
    # (apply_geometry itself resamples the channels in parallel).
    #
    # An IDENTITY geometry step (every resolved factor == 1 — the common case,
    # since the default recipe seeds a shared `isotropic` step and most sources
    # already sit at the target spacing) resamples NOTHING, yet `apply_geometry`
    # still runs a full `zoom` per channel.  Streaming has skipped it since the
    # assessment fix; the in-memory path did not (Brick 3a #5).  REUSE
    # `streaming._geometry_is_identity` rather than duplicating the predicate —
    # it must stay ONE definition or the two regimes can silently disagree about
    # what "identity" means.  Imported lazily (streaming imports this module at
    # module level, so a top-level import here would be circular).
    if geom is not None:
        if progress is not None:
            progress.op("geometry", None, False)
        from .streaming import _geometry_is_identity

        if _geometry_is_identity(geom, spacing):
            logger.debug(
                "preprocessing.execute: identity geometry step skipped "
                "(spacing %s already at target)", spacing,
            )
        else:
            pre, spacing = apply_geometry(
                pre, spacing, geom.config, max_workers=max_workers)

        # Stage 3 — intensity ops AFTER the geometry boundary, per channel (also
        # independent across channels).
        def _stage3(ci: int) -> np.ndarray:
            if should_cancel is not None and should_cancel():
                raise PreprocessingCancelled("cancelled")
            return _run_intensity_ops(
                pre[ci], _ops_for(ci)[boundary:],
                parallel=parallel, should_cancel=should_cancel,
                block_workers=stage_block_workers,  # divided when channels widen
                progress=progress, channel_index=ci,
            )

        processed = _parallel_map(_stage3, range(n), max_workers=stage_workers)
    else:
        processed = pre

    # Final [0,1] + float32 contract (cubic geometry can overshoot slightly).
    processed = [
        np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0).astype(np.float32)
        for a in processed
    ]

    stamp: Dict[str, Any] = {
        "recipe": _recipe.to_json(norm_recipe),
        "dtype": "uint16",
        "scale": "fixed_unit_x65535",  # to_uint16 = fixed [0,1]→×65535
    }
    return processed, spacing, stamp
