"""Resource-adaptive convert settings for the Step-1 large-data "Prepare for
large-data viewing" convert (feeds :func:`multiscale.convert_to_ome_zarr`).

napari-free BY DESIGN (mirrors the rest of this package): no Qt, no napari, no
imports from ``napari_epicycle``.  The GUI dialog
(``napari_epicycle/data_preparation/_largedata_config_dialog.py``) and the
Step-1 controller import these PURE helpers; the persistence layer that stores a
chosen config in a workspace's ``step1_state.json`` lives in the UI layer
(``napari_epicycle/data_preparation/_state.py``) so this module stays free of
the Step-1 path helpers.

Three ideas
-----------
* **Presets** — ``safe`` / ``balanced`` / ``fast`` scale the convert knobs
  (``num_workers``/``chunk``/``downsample``/``min_size``) to the machine:
  ``safe`` is the current portable behaviour (1 worker, small cubic chunk);
  ``balanced`` uses a few cores; ``fast`` uses all cores + a bigger chunk.
  ``custom`` is not a preset here — callers hold raw knobs for it.
* **RAM-safety clamp** — the dominant, file-size-INDEPENDENT cost of the convert
  is roughly ``num_workers x chunk**3 x itemsize`` (the in-flight cubic chunks;
  see ``convert_to_ome_zarr``'s ``num_workers`` docstring).  We clamp
  ``num_workers`` so that term stays under a fraction of AVAILABLE RAM, so
  ``fast`` degrades gracefully on a small box and uses every core on a big one.
* **Deterministic + pure** — :func:`recommend_convert_settings` takes the
  machine facts as explicit args (no hidden ``psutil`` call), so it is trivially
  unit-testable and reproducible.  :func:`detect_machine_resources` is the thin,
  ``psutil``-guarded probe the GUI uses to gather those facts at call time.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

# psutil gives us total/available RAM for the adaptive clamp + the dialog
# readout.  Guarded: on a host without psutil we fall back to an
# ``os.cpu_count()``-only recommendation (RAM unknown -> the clamp is skipped,
# so the preset's core count is trusted).
try:  # pragma: no cover - exercised implicitly; the fallback is unit-tested
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - psutil-missing fallback
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


#: The named presets :func:`recommend_convert_settings` understands.  ``custom``
#: is deliberately NOT here — a custom config carries raw user knobs and never
#: routes through the recommender (callers handle it, e.g. the dialog).
PRESET_NAMES = ("safe", "balanced", "fast")

#: The preset a caller falls back to when nothing is persisted.
DEFAULT_PRESET = "balanced"

#: The CUBIC chunk edges the "advanced" dialog option still offers (also the
#: accepted set for validation of the legacy int ``chunk`` field).  256 was
#: dropped in Brick 4b: at uint16 a cubic-256 chunk is 32 MB, so painting one
#: 2D slice would decompress GBs.
ALLOWED_CHUNKS = (32, 64, 128)

#: The DEFAULT store chunk shape ``(cz, cy, cx)`` for NEW stores (Brick 4b).
#: Thin in Z (a 2D XY slice decompresses 32 slices of depth, not 64) and big
#: in-plane (it touches 8x8 chunks, not 16x16) -> ~4x less decompressed per
#: slice than the old cubic 64, while KEEPING the in-plane 128 that made the
#: streaming write fast (Brick 3a).  One chunk at uint16 is exactly 1 MiB.
DEFAULT_CHUNK_ZYX = (32, 128, 128)

#: The ZYX chunk shapes the dialog offers.  All are thin-Z / big-in-plane; they
#: trade slice-read size against chunk count.
ALLOWED_CHUNK_ZYX = (
    (16, 128, 128),
    (32, 128, 128),
    (32, 256, 256),
    (64, 256, 256),
)

#: A convert may use at most this fraction of AVAILABLE RAM for its in-flight
#: cubic chunks (``num_workers x chunk**3 x itemsize``).
_RAM_SAFETY_FRACTION = 0.25

#: The fields a persisted / returned config dict carries.  ``chunk_zyx`` and
#: ``compressor`` are Brick 4b ADDITIONS: an older persisted config without them
#: still loads (``normalize_convert_config`` derives ``chunk_zyx`` from the
#: legacy cubic ``chunk`` and defaults ``compressor``), so the schema change is
#: strictly additive.
CONFIG_FIELDS = ("preset", "num_workers", "chunk", "chunk_zyx", "compressor",
                 "downsample", "min_size")

#: The compressors the store writer understands (see
#: :func:`multiscale._make_compressor`).  ``zstd`` clevel 3 + shuffle is the
#: Brick 4b default for NEW stores; ``lz4`` is zarr's historical default and the
#: faster-but-bigger alternative.  The codec is recorded per-array in
#: ``.zarray``, so a store written with either opens with either setting.
ALLOWED_COMPRESSORS = ("zstd", "lz4", "default")

#: The compressor NEW stores use unless a config says otherwise.
DEFAULT_COMPRESSOR = "zstd"


def normalize_chunk_zyx(value, fallback=DEFAULT_CHUNK_ZYX):
    """Coerce *value* into a ``(cz, cy, cx)`` tuple of positive ints.

    Accepts a 3-sequence (the new ``chunk_zyx``) or a plain positive int (the
    legacy cubic ``chunk`` -- ``N`` means ``(N, N, N)``).  Anything else returns
    *fallback*, so a corrupt persisted value degrades instead of raising.
    """
    if isinstance(value, bool):
        return tuple(int(v) for v in fallback)
    if isinstance(value, int) and value >= 1:
        return (int(value),) * 3
    try:
        vals = [int(v) for v in value]
    except (TypeError, ValueError):
        return tuple(int(v) for v in fallback)
    if len(vals) != 3 or any(v < 1 for v in vals):
        return tuple(int(v) for v in fallback)
    return tuple(vals)


def detect_machine_resources() -> Dict[str, int]:
    """Probe this host for ``{cpu_count, total_ram_bytes, avail_ram_bytes}``.

    Uses ``psutil.virtual_memory()`` for RAM when psutil is importable; otherwise
    falls back to ``os.cpu_count()`` only and reports ``0`` for both RAM figures
    (an "unknown" sentinel that makes :func:`recommend_convert_settings` skip the
    RAM clamp).  Never raises.
    """
    cpu = os.cpu_count() or 1
    total = 0
    avail = 0
    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            total = int(vm.total)
            avail = int(vm.available)
        except Exception:  # pragma: no cover - psutil runtime failure
            total = 0
            avail = 0
    return {
        "cpu_count": int(cpu),
        "total_ram_bytes": int(total),
        "avail_ram_bytes": int(avail),
    }


#: The two source-read regimes the convert can take.  They have very different
#: peak-RAM profiles, and which one applies is a property of the SOURCE FILE, not
#: of its shape -- see :func:`estimate_peak_ram_bytes`.
SOURCE_REGIMES = ("memmap", "compressed")


def resolve_source_regime(src_path):
    """Which read regime :func:`lazy.open_lazy` will take for *src_path*.

    ``"memmap"``     -- an uncompressed TIFF: ``tifffile.memmap`` succeeds, so the
                        converter reads it ONE CHUNK at a time through
                        ``multiscale._LazyReadProxy``.
    ``"compressed"`` -- a compressed TIFF (DEFLATE / LZW / ...): pages are not
                        memory-mappable, so the source opens as thin-Z dask
                        chunks and a whole Z-SLAB must be buffered per worker.
    ``None``         -- unknown (not a TIFF, missing, unreadable).  Callers treat
                        unknown as the conservative (compressed) case.

    Best-effort and NEVER raises: this only sharpens a RAM estimate.  It mirrors
    ``open_lazy``'s own decision by attempting the same ``tifffile.memmap`` call,
    rather than guessing from the extension -- guessing is what got the regime
    wrong in the first place.
    """
    if src_path is None:
        return None
    try:
        import os

        p = os.fspath(src_path)
        if not os.path.isfile(p):
            return None
        if not p.lower().endswith((".tif", ".tiff")):
            return None
        import tifffile

        try:
            tifffile.memmap(p)
            return "memmap"
        except Exception:
            return "compressed"
    except Exception:
        return None


def _zyx_from_shape(shape):
    """The trailing ``(Z, Y, X)`` of *shape*, or ``None`` if it is not usable.

    Accepts a 3-sequence (ZYX) or a 4-sequence (CZYX -- the channel axis does not
    enter the buffer, each channel's slab is assembled separately).  Anything
    else (``None``, a string, a 1- or 2-sequence, a non-positive extent) is
    "unknown", which makes the caller fall back to the chunk-only bound.
    """
    if shape is None or isinstance(shape, (str, bytes)):
        return None
    try:
        vals = [int(v) for v in shape]
    except (TypeError, ValueError):
        return None
    if len(vals) not in (3, 4):
        return None
    zyx = vals[-3:]
    if any(v < 1 for v in zyx):
        return None
    return tuple(zyx)


def estimate_peak_ram_bytes(num_workers: int, chunk,
                            itemsize: int = 2, shape=None,
                            regime=None) -> int:
    """Modelled peak RAM (bytes) of the convert.

    The convert has TWO read regimes with very different peaks, and *regime* --
    not the mere presence of a *shape* -- is what selects between them:

    * ``"memmap"`` (an UNCOMPRESSED TIFF) -> ``num_workers x prod(chunk_zyx) x
      itemsize``.  ``tifffile.memmap`` succeeds, so ``multiscale._LazyReadProxy``
      genuinely reads one chunk at a time; peak is ~one chunk per worker NO
      MATTER how big the XY field is.
    * ``"compressed"`` (a DEFLATE / LZW TIFF) -> ``num_workers x (chunk_z x Y x X)
      x itemsize`` plus a couple of chunks per worker for the pyramid read-back.
      Compressed pages are not memory-mappable, so the source opens as thin-Z
      ``(1, Y, X)`` dask chunks and a whole ``chunk_z`` SLAB must be buffered to
      assemble one row of cubic chunks.

    An earlier version inferred "slab" from ``shape is not None``.  That was
    wrong in a way that bites the good case: an 8192x8192 UNCOMPRESSED source
    was charged 4 GiB per worker and got clamped to a single core on a 32 GB
    box, when its real cost is 1 MiB per worker and it should use all of them.

    *regime* ``None`` with a known *shape* assumes the CONSERVATIVE (compressed)
    case -- under-promise rather than OOM.  No usable *shape* always gives the
    chunk bound (also what the dialog shows before a file is picked).
    Inputs are floored so a degenerate ``0`` never yields a nonsense ``0``.
    """
    nw = max(1, int(num_workers))
    it = max(1, int(itemsize))
    cz, cy, cx = normalize_chunk_zyx(chunk, fallback=DEFAULT_CHUNK_ZYX)
    chunk_bytes = cz * cy * cx * it

    zyx = _zyx_from_shape(shape)
    if zyx is None or regime == "memmap":
        return nw * chunk_bytes

    _z, y, x = zyx
    slab = min(cz, max(1, _z)) * y * x * it
    # + the coarser-level read-back buffers (a 2x2x2 neighbourhood of output
    # chunks per worker); small next to the slab but not zero.
    return nw * (slab + 8 * chunk_bytes)


def _clamp_workers_for_ram(num_workers: int, chunk,
                           avail_ram_bytes: Optional[int],
                           itemsize: int, shape=None, regime=None) -> int:
    """Clamp ``num_workers`` so the modelled per-worker buffer x workers stays under
    ``_RAM_SAFETY_FRACTION`` of ``avail_ram_bytes`` (never below 1).

    When ``avail_ram_bytes`` is unknown (``None`` / ``<= 0`` -- the psutil-less
    fallback) the clamp is a NO-OP: RAM is unknown, so the preset's core count is
    trusted unchanged.
    """
    nw = max(1, int(num_workers))
    if avail_ram_bytes is None or avail_ram_bytes <= 0:
        return nw
    per_worker = estimate_peak_ram_bytes(1, chunk, itemsize, shape=shape,
                                        regime=regime)
    budget = _RAM_SAFETY_FRACTION * float(avail_ram_bytes)
    max_workers = int(budget // per_worker)
    return max(1, min(nw, max_workers))


def recommend_convert_settings(
    preset: str,
    *,
    total_ram_bytes: int,
    avail_ram_bytes: int,
    cpu_count: int,
    dtype_itemsize: int = 2,
    shape=None,
    regime=None,
) -> Dict[str, object]:
    """Return the adaptive convert settings for ``preset`` on a machine with the
    given resources.

    Returns ``{"preset", "num_workers", "chunk", "downsample", "min_size"}``.

    Preset table (before the RAM clamp)::

        preset     num_workers      chunk  downsample  min_size
        safe       1                64     2           256
        balanced   min(4, cpu)      64     2           256
        fast       cpu (all cores)  128    2           256

    Brick 3a #2 briefly moved ``safe`` / ``balanced`` to a cubic 128 for WRITE
    throughput; Brick 3c #3 reverted it.  A prepared store is scrolled far more
    often than it is written, and with CUBIC chunks and no read cache a bigger
    edge makes 2D slice scrolling worse in proportion (a 128 edge decompresses
    256 MB to paint a 2 MB XY slice, vs 128 MB at 64).  ``fast`` keeps 128 as its
    deliberate "big box, throughput first" choice.  The write win returns in
    Brick 4b as an ANISOTROPIC (32, 128, 128) chunk plus an LRU read cache, which
    is the only shape that helps both directions.

    ``num_workers`` is then RAM-clamped (see :func:`_clamp_workers_for_ram`): so
    ``fast`` with ``chunk=128`` degrades on a small box but uses every core on a
    big one.  ``safe`` (1 worker) is unaffected by the clamp.

    *shape* is the SOURCE volume shape (``(Z, Y, X)`` or ``(C, Z, Y, X)``) when
    known.  It selects the honest compressed-path peak model (a ``chunk_z x Y x
    X`` slab per worker) instead of the chunk-only memmap bound -- see
    :func:`estimate_peak_ram_bytes`.  ``None`` keeps the pre-Brick-4b behaviour.

    *regime* is ``"memmap"`` / ``"compressed"`` / ``None`` (see
    :func:`resolve_source_regime`): it decides WHICH peak model applies, so an
    uncompressed source is not charged the compressed path's slab.

    ``total_ram_bytes`` is accepted for API symmetry / the dialog header; only
    ``avail_ram_bytes`` drives the clamp.  Raises ``ValueError`` for a preset
    outside :data:`PRESET_NAMES` (``custom`` is handled by callers, not here).
    """
    p = str(preset).lower()
    if p not in PRESET_NAMES:
        raise ValueError(
            f"recommend_convert_settings: unknown preset {preset!r}; "
            f"expected one of {PRESET_NAMES} (custom is caller-handled)"
        )
    cpu = max(1, int(cpu_count))
    itemsize = max(1, int(dtype_itemsize))

    chunk_zyx = DEFAULT_CHUNK_ZYX
    downsample, min_size = 2, 256
    if p == "safe":
        num_workers = 1
    elif p == "balanced":
        num_workers = min(4, cpu)
    else:  # fast
        num_workers = cpu

    num_workers = _clamp_workers_for_ram(
        num_workers, chunk_zyx, avail_ram_bytes, itemsize, shape=shape,
        regime=regime)

    return {
        "preset": p,
        "num_workers": int(num_workers),
        # Legacy cubic mirror, for an OLDER CHECKOUT of this app reading a
        # config this one wrote.  It is the Z axis (32), NOT the in-plane edge
        # (128), deliberately: an old checkout takes this int and prepares a
        # CUBIC store with it, and a cubic 128 is the worst case for 2D
        # scrolling -- precisely what Brick 3c had to revert.  Mirroring 32
        # degrades that checkout to a small, safe cube instead.  This checkout
        # never reads it for writing; the WRITER uses ``chunk_zyx``.
        "chunk": int(chunk_zyx[0]),
        "chunk_zyx": tuple(int(c) for c in chunk_zyx),
        "compressor": DEFAULT_COMPRESSOR,
        "downsample": int(downsample),
        "min_size": int(min_size),
    }


def normalize_convert_config(cfg: object) -> Optional[Dict[str, object]]:
    """Validate a (possibly persisted / partial) convert-config mapping.

    Returns a clean ``{"preset", "num_workers", "chunk", "downsample",
    "min_size"}`` copy, or ``None`` (NOT an exception) when *cfg* is not a dict,
    is missing a field, or carries a non-positive-int knob -- so a corrupt or
    partial persisted value makes the caller fall back to the adaptive default.

    ``preset`` must be a non-empty string (ANY name, including ``custom`` and
    future presets); the four knobs must be positive ``int`` (``bool`` rejected).
    """
    if not isinstance(cfg, dict):
        return None
    preset = cfg.get("preset")
    if not isinstance(preset, str) or not preset.strip():
        return None
    out: Dict[str, object] = {"preset": preset}
    for key in ("num_workers", "chunk", "downsample", "min_size"):
        val = cfg.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            return None
        out[key] = int(val)
    # Brick 4b ADDITIVE keys.  The fallback when ``chunk_zyx`` is absent or
    # corrupt depends on what the preset MEANS:
    #
    # * a NAMED preset (safe / balanced / fast) means "whatever the recommender
    #   currently says".  Every workspace that ran Prepare before Brick 4b has a
    #   saved config with ``chunk: 64`` and no ``chunk_zyx``; deriving
    #   ``(64, 64, 64)`` from it would PIN that workspace to the old cubic shape
    #   forever, so the user would keep the choppy scrolling while the dialog
    #   showed them the new default.  Named presets therefore adopt
    #   DEFAULT_CHUNK_ZYX.
    # * ``custom`` means the user typed those exact numbers -- honour them, so
    #   the cubic fallback stays.
    _named = preset.strip().lower() in PRESET_NAMES
    out["chunk_zyx"] = normalize_chunk_zyx(
        cfg.get("chunk_zyx"),
        fallback=(DEFAULT_CHUNK_ZYX if _named else (out["chunk"],) * 3))
    comp = cfg.get("compressor")
    out["compressor"] = (comp if isinstance(comp, str)
                         and comp.lower() in ALLOWED_COMPRESSORS
                         else DEFAULT_COMPRESSOR)
    return out
