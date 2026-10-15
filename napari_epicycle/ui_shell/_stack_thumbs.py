"""Step 4 Session thumbnails, never read on the main thread (an owner decision).

Shared with Step 3: moved here from ``segmentation/``, whose ``_stack_thumbs`` is now an alias
of THIS module object.

Invariant: ``tifffile`` is only called on the worker thread, never from the
main thread, so the session list never freezes while pictures load.

Two halves:

* **Worker side** (a ``QThread``): :class:`_StackThumbWorker` drains a queue of
  :class:`ThumbRequest` ``(name, path, axes, mtime_ns, st_size)``.  For each it
  reads ONLY the mid-Z plane of up to three channels through the TIFF's zarr
  view (``series[0].aszarr()``), indexed from the stack's RESOLVED axes -- so
  planar-RGB pages (what ``imwrite`` writes for C = 3 / 4 without
  ``photometric``), OME and multi-series files read the right plane.  It
  strides and percentile-normalises each plane, composes RGB, scales it to
  ``size`` px with a ``QImage`` (safe off the main thread; a ``QPixmap`` is
  not), encodes a PNG in memory and writes it atomically to
  ``<cache_dir>/<safe name>.<path hash>.<mtime_ns>.<st_size>.<style>.png``
  (``THUMB_STYLE``, the colour version: a PNG drawn in older colours is a
  cache miss and is swept).  It emits
  ``ready(name, stack_path, png_path, png_bytes)``; ``png_path`` is ``""``
  when the cache folder cannot be written, and the bytes still draw.  A PNG
  already on disk for that key is a cache hit and is emitted without opening
  the TIFF.  A stack that cannot be read, or whose resolved axes do not fit
  the file (or disagree with the file's own C/Z/Y/X axes), emits
  ``failed(name, stack_path, reason)`` and the row keeps its placeholder --
  never a wrong picture.  When a new PNG is written, that stack's older PNGs
  are deleted.

* **Main-thread side**: :class:`ThumbCache`, a 512-entry LRU keyed by
  ``(stack path, mtime_ns, st_size)`` that turns PNG bytes or a PNG path into
  a ``QPixmap``.  It refuses to run off the main thread.

Lifecycle: the ``QThread`` is never parented (a deleted parent widget would
destroy a running thread); every live thread sits in
:data:`_LIVE_THUMB_WORKERS` until it has finished, and holds its worker
(``thread._thumb_worker``).  A :func:`stop_thumb_worker` that times out leaves
the thread parked there.

Why the reader is not Step 1's: ``data_preparation/ui.py`` imports pyqtgraph
and builds widgets at import time, and its ``_read_patch_midz_planes`` trusts
the file's own axes with global page keys.  ``thumb_norm8`` / ``thumb_stride``
are copies of Step 1's ``_thumb_norm8`` / ``_thumb_stride`` (``thumb_norm8``
made NaN-safe).

Napari-free: qtpy + numpy + tifffile (+ zarr through tifffile) only.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from qtpy.QtCore import QBuffer, QByteArray, QIODevice, QObject, QThread, Qt, Signal, Slot
from qtpy.QtGui import QImage

#: The picture's edge on a WIDE row (Session, Curation, Step 4 1) and on a
#: NARROW / measure row (Predict + Reconstruct, Step 4 2 / 3) -- chosen
#: so a measure row fits its panel.  The painter's picture rect
#: and row height follow the mode; the PNG's name carries the size the
#: worker drew (``.s<px>``), so a change of a constant is a cache miss,
#: never a scaled-up old PNG.  The worker draws ONE picture per patch at the
#: wide size; a narrow row paints the same pixmap smoothly scaled to its
#: rect.
THUMB_WIDE_PX = 128
THUMB_NARROW_PX = 80
#: kept for older callers: the size a worker draws by default
THUMB_SIZE = THUMB_WIDE_PX
THUMB_CACHE_SIZE = 512
MAX_CHANNELS = 3
MAX_NAME_LEN = 80
TMP_MAX_AGE_S = 60.0
#: The colour of each channel, in channel order (decided 2026-10-06):
#: channel 0 magenta (nuclei by EpiCycle's
#: convention), channel 1 green (membrane), a third channel blue; added and
#: clipped.  One channel draws grey.  The colour follows the INDEX, not the
#: role.  (Until 2026-10-06: channel 0 red, 1 green, 2 blue.)
THUMB_COLOURS = ((255, 0, 255), (0, 255, 0), (0, 0, 255))
#: The colour version in every PNG's name: bump it whenever THUMB_COLOURS
#: changes, so no PNG drawn in the old colours is ever a cache hit.  Its
#: shape must stay ``c<digits>``: ``stale_pngs`` recognises only that shape,
#: so a token of another shape would orphan every PNG drawn with it.
THUMB_STYLE = "c2"
#: The one-channel colour rules ``compose_rgb(single=)`` knows.  ``None`` is
#: grey (Step 4); ``"inferno"`` (Step 3, owner 2026-10-10) maps the
#: normalised plane through ``THUMB_INFERNO``.  The rule is in the PNG's
#: name (``style_token``) so Step 3's and Step 4's picture of one file
#: never collide.
SINGLE_INFERNO = "inferno"
#: The old single amber outline colour (A1); the overlay below replaced it.
THUMB_OUTLINE_COLOUR = (255, 200, 60)
#: The overlay of a reconstructed patch's labels on its picture (A3, owner
#: 2026-10-10): each instance in the colour napari gives that id
#: (``label_colormap(49, 0.5)``; colour = ``colors[((id - 1) % 49) + 1]``) as
#: a translucent fill (``THUMB_OVERLAY_ALPHA``) plus a 1 px inner boundary
#: of the same colour at full strength.  The mode and the alpha are part of
#: the PNG's name (``overlay_token``).
OVERLAY_FILL_OUTLINE = "fill+outline"
OVERLAY_OUTLINE = "outline"
OVERLAY_FILL = "fill"
OVERLAY_MODES = (OVERLAY_FILL_OUTLINE, OVERLAY_OUTLINE, OVERLAY_FILL)
THUMB_OVERLAY_MODE = OVERLAY_FILL_OUTLINE
THUMB_OVERLAY_ALPHA = 0.45
_OVERLAY_CODES = {OVERLAY_FILL_OUTLINE: "fo", OVERLAY_OUTLINE: "ol",
                  OVERLAY_FILL: "fl"}
LABEL_COLOURS = 49
LABEL_SEED = 0.5

#: matplotlib 3.10's inferno, 256 entries, sampled ONCE (no matplotlib import at
#: call time): a single-channel Step 3 patch draws in this instead of grey.
THUMB_INFERNO = (
    (0,0,4), (1,0,5), (1,1,6), (1,1,8),
    (2,1,10), (2,2,12), (2,2,14), (3,2,16),
    (4,3,18), (4,3,20), (5,4,23), (6,4,25),
    (7,5,27), (8,5,29), (9,6,31), (10,7,34),
    (11,7,36), (12,8,38), (13,8,41), (14,9,43),
    (16,9,45), (17,10,48), (18,10,50), (20,11,52),
    (21,11,55), (22,11,57), (24,12,60), (25,12,62),
    (27,12,65), (28,12,67), (30,12,69), (31,12,72),
    (33,12,74), (35,12,76), (36,12,79), (38,12,81),
    (40,11,83), (41,11,85), (43,11,87), (45,11,89),
    (47,10,91), (49,10,92), (50,10,94), (52,10,95),
    (54,9,97), (56,9,98), (57,9,99), (59,9,100),
    (61,9,101), (62,9,102), (64,10,103), (66,10,104),
    (68,10,104), (69,10,105), (71,11,106), (73,11,106),
    (74,12,107), (76,12,107), (77,13,108), (79,13,108),
    (81,14,108), (82,14,109), (84,15,109), (85,15,109),
    (87,16,110), (89,16,110), (90,17,110), (92,18,110),
    (93,18,110), (95,19,110), (97,19,110), (98,20,110),
    (100,21,110), (101,21,110), (103,22,110), (105,22,110),
    (106,23,110), (108,24,110), (109,24,110), (111,25,110),
    (113,25,110), (114,26,110), (116,26,110), (117,27,110),
    (119,28,109), (120,28,109), (122,29,109), (124,29,109),
    (125,30,109), (127,30,108), (128,31,108), (130,32,108),
    (132,32,107), (133,33,107), (135,33,107), (136,34,106),
    (138,34,106), (140,35,105), (141,35,105), (143,36,105),
    (144,37,104), (146,37,104), (147,38,103), (149,38,103),
    (151,39,102), (152,39,102), (154,40,101), (155,41,100),
    (157,41,100), (159,42,99), (160,42,99), (162,43,98),
    (163,44,97), (165,44,96), (166,45,96), (168,46,95),
    (169,46,94), (171,47,94), (173,48,93), (174,48,92),
    (176,49,91), (177,50,90), (179,50,90), (180,51,89),
    (182,52,88), (183,53,87), (185,53,86), (186,54,85),
    (188,55,84), (189,56,83), (191,57,82), (192,58,81),
    (193,58,80), (195,59,79), (196,60,78), (198,61,77),
    (199,62,76), (200,63,75), (202,64,74), (203,65,73),
    (204,66,72), (206,67,71), (207,68,70), (208,69,69),
    (210,70,68), (211,71,67), (212,72,66), (213,74,65),
    (215,75,63), (216,76,62), (217,77,61), (218,78,60),
    (219,80,59), (221,81,58), (222,82,56), (223,83,55),
    (224,85,54), (225,86,53), (226,87,52), (227,89,51),
    (228,90,49), (229,92,48), (230,93,47), (231,94,46),
    (232,96,45), (233,97,43), (234,99,42), (235,100,41),
    (235,102,40), (236,103,38), (237,105,37), (238,106,36),
    (239,108,35), (239,110,33), (240,111,32), (241,113,31),
    (241,115,29), (242,116,28), (243,118,27), (243,120,25),
    (244,121,24), (245,123,23), (245,125,21), (246,126,20),
    (246,128,19), (247,130,18), (247,132,16), (248,133,15),
    (248,135,14), (248,137,12), (249,139,11), (249,140,10),
    (249,142,9), (250,144,8), (250,146,7), (250,148,7),
    (251,150,6), (251,151,6), (251,153,6), (251,155,6),
    (251,157,7), (252,159,7), (252,161,8), (252,163,9),
    (252,165,10), (252,166,12), (252,168,13), (252,170,15),
    (252,172,17), (252,174,18), (252,176,20), (252,178,22),
    (252,180,24), (251,182,26), (251,184,29), (251,186,31),
    (251,188,33), (251,190,35), (250,192,38), (250,194,40),
    (250,196,42), (250,198,45), (249,199,47), (249,201,50),
    (249,203,53), (248,205,55), (248,207,58), (247,209,61),
    (247,211,64), (246,213,67), (246,215,70), (245,217,73),
    (245,219,76), (244,221,79), (244,223,83), (244,225,86),
    (243,227,90), (243,229,93), (242,230,97), (242,232,101),
    (242,234,105), (241,236,109), (241,237,113), (241,239,117),
    (241,241,121), (242,242,125), (242,244,130), (243,245,134),
    (243,246,138), (244,248,142), (245,249,146), (246,250,150),
    (248,251,154), (249,252,157), (250,253,161), (252,255,164),
)

# Reasons emitted on ``failed`` (console / tooltip text, never shown as a picture).
THUMB_FAIL_MISSING = "The stack file is missing."
THUMB_FAIL_UNREADABLE = "The stack file could not be read."
THUMB_FAIL_AXES = "The stack's axes do not fit the file."
THUMB_FAIL_ENCODE = "The thumbnail could not be made."

#: Every running thumbnail thread, kept alive until it has finished so it is
#: never destroyed while running (the ``_LIVE_INTENSITY_WORKERS`` pattern,
#: ``data_preparation/controller.py``).  A thread removes itself on finished.
_LIVE_THUMB_WORKERS: set = set()


def _diag(msg: str) -> None:
    try:
        print(f"[STEP4-THUMB] {msg}", flush=True)
    except Exception:
        pass


class ThumbRequest(NamedTuple):
    name: str
    path: str
    axes: str
    mtime_ns: int
    st_size: int
    #: Step 3 (A1): the run's labels file whose MID-Z boundaries are drawn
    #: over the picture, and its mtime; ``""`` / 0 = no outline (Step 4)
    labels_path: str = ""
    labels_mtime_ns: int = 0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def thumb_norm8(arr: np.ndarray) -> np.ndarray:
    """Percentile-clip a 2D plane to uint8 [0, 255]; NaN draws black."""
    a = np.asarray(arr).astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, np.uint8)
    lo = float(np.nanpercentile(np.where(finite, a, np.nan), 1))
    hi = float(np.nanpercentile(np.where(finite, a, np.nan), 99))
    if hi <= lo:
        hi = lo + 1.0
    out = np.nan_to_num((a - lo) / (hi - lo) * 255.0, nan=0.0,
                        posinf=255.0, neginf=0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def thumb_stride(arr: np.ndarray, size: int) -> np.ndarray:
    """Stride-downsample a 2D plane so its longest edge is ~<= size (copy of
    Step 1's)."""
    h, w = arr.shape[:2]
    if h <= size and w <= size:
        return arr
    return arr[:: max(1, h // size), :: max(1, w // size)]


_INFERNO_ARR = None


def _inferno_array() -> np.ndarray:
    global _INFERNO_ARR
    if _INFERNO_ARR is None:
        _INFERNO_ARR = np.asarray(THUMB_INFERNO, dtype=np.uint8)
    return _INFERNO_ARR


def compose_rgb(planes: np.ndarray, size: int = THUMB_SIZE, *,
                colours=THUMB_COLOURS, single=None) -> np.ndarray:
    """``(n, Y, X)`` planes -> ``(h, w, 3)`` uint8.  One channel is grey; two
    or three are each normalised plane times its colour (``colours``,
    default ``THUMB_COLOURS``: magenta, green, blue), summed and clipped.
    ``single`` is the one-channel rule: ``None`` grey, ``"inferno"`` the
    ``THUMB_INFERNO`` table (an unknown rule raises ``ValueError``)."""
    if single not in (None, SINGLE_INFERNO):
        raise ValueError(f"unknown single-channel rule {single!r}")
    planes = np.asarray(planes)
    if planes.ndim == 2:
        planes = planes[None]
    if planes.ndim != 3 or planes.shape[0] == 0:
        raise ValueError(f"expected (n, Y, X) planes, got {planes.shape}")
    chans = [thumb_norm8(thumb_stride(planes[i], int(size)))
             for i in range(min(int(planes.shape[0]), MAX_CHANNELS))]
    h, w = chans[0].shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if len(chans) == 1:
        if single == SINGLE_INFERNO:
            out[...] = _inferno_array()[chans[0]]
        else:
            out[..., 0] = out[..., 1] = out[..., 2] = chans[0]
    else:
        acc = np.zeros((h, w, 3), dtype=np.int32)
        for p, colour in zip(chans, colours):
            acc += p.astype(np.int32)[..., None] * np.asarray(
                colour, dtype=np.int32)
        out[...] = np.clip((acc + 127) // 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


_UNSAFE = re.compile(r"[\\/:*?\"<>|\x00-\x1f\x7f]")
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_stack_name(name: str) -> str:
    """A stack name usable as a file name on Windows: reserved and control
    characters become ``_``, trailing dots / spaces go, a reserved device name
    (``CON``, ``LPT1``...) is prefixed, and the length is capped."""
    s = _UNSAFE.sub("_", str(name))[:MAX_NAME_LEN].rstrip(". ")
    if not s:
        return "_"
    if s.split(".")[0].upper() in _RESERVED:
        s = ("_" + s)[:MAX_NAME_LEN]
    return s


def path_hash(path) -> str:
    """Eight hex characters of the SHA-1 of the normalised absolute path."""
    norm = os.path.normcase(os.path.abspath(str(path)))
    return hashlib.sha1(norm.encode("utf-8", "surrogatepass")).hexdigest()[:8]


def style_token(single=None) -> str:
    """The colour version in a PNG's name: ``THUMB_STYLE``, plus ``i`` for
    the inferno rule (so Step 3's one-channel picture and Step 4's grey one
    of the same file are two files)."""
    return THUMB_STYLE + ("i" if single == SINGLE_INFERNO else "")


def overlay_token(mode=None, alpha=None) -> str:
    """The overlay rule in a PNG's name: ``o`` + the mode's two letters +
    the alpha in percent (``ofo45``).  A change of either is a new file."""
    mode = THUMB_OVERLAY_MODE if mode is None else mode
    alpha = THUMB_OVERLAY_ALPHA if alpha is None else alpha
    if mode not in _OVERLAY_CODES:
        raise ValueError(f"unknown overlay mode {mode!r}")
    return f"o{_OVERLAY_CODES[mode]}{int(round(float(alpha) * 100))}"


#: the tail of a PNG's name after ``<style>``: the size, the labels part
#: and the overlay rule -- each optional so a PNG drawn by an older build
#: still matches (and is swept as stale)
PNG_TAIL = r"(?:\.s\d+)?(?:\.l\d+)?(?:\.o[a-z]{2}\d+)?"


def thumb_png_path(cache_dir, name: str, stack_path, mtime_ns: int,
                   st_size: int, *, single=None,
                   labels_mtime_ns: int = 0, size: int = THUMB_SIZE,
                   overlay_mode=None, overlay_alpha=None) -> Path:
    """``<cache_dir>/<safe name>.<path hash>.<mtime_ns>.<st_size>.<style>
    .s<size>[.l<labels mtime_ns>.<overlay>].png`` (``THUMB_STYLE``; the
    ``l`` and overlay parts only when the picture carries the labels
    overlay)."""
    lab = ""
    if labels_mtime_ns:
        lab = (f".l{int(labels_mtime_ns)}."
               f"{overlay_token(overlay_mode, overlay_alpha)}")
    return Path(cache_dir) / (f"{safe_stack_name(name)}.{path_hash(stack_path)}"
                              f".{int(mtime_ns)}.{int(st_size)}"
                              f".{style_token(single)}.s{int(size)}{lab}.png")


def stale_pngs(cache_dir, name: str, stack_path, keep) -> list:
    """This stack's PNGs for OTHER mtimes / sizes / colour versions: same
    safe name AND same path hash, so ``stack`` never matches ``stack_2`` nor
    another folder's ``stack``; a name with no colour token (drawn before
    ``THUMB_STYLE``) or another token is stale too."""
    d = Path(cache_dir)
    if not d.is_dir():
        return []
    pat = re.compile(r"^" + re.escape(safe_stack_name(name)) + r"\."
                     + re.escape(path_hash(stack_path))
                     + r"\.\d+\.\d+(?:\.c\d+i?)?" + PNG_TAIL + r"\.png$")
    keep_name = Path(keep).name
    return [p for p in d.iterdir() if pat.match(p.name) and p.name != keep_name]


def sweep_stale_tmp(cache_dir, max_age_s: float = TMP_MAX_AGE_S) -> int:
    """Delete ``*.tmp`` left by an interrupted write (older than ``max_age_s``,
    so a concurrent writer's live temp file is left alone)."""
    d = Path(cache_dir)
    if not d.is_dir():
        return 0
    n = 0
    now = time.time()
    for p in d.glob("*.tmp"):
        try:
            if now - p.stat().st_mtime > max_age_s:
                p.unlink()
                n += 1
        except OSError:
            pass
    if n:
        _diag(f"swept {n} stale temp file(s)")
    return n


def n_channels(axes: str, shape) -> int:
    """How many channels to draw (<= 3).  Raises ``ValueError`` when the
    resolved axes do not fit the shape: they must be as long as the shape, end
    in ``YX``, have no repeats, and only ``C`` / ``Z`` may lead."""
    axes = str(axes or "").upper()
    shape = tuple(int(s) for s in shape)
    if len(axes) != len(shape):
        raise ValueError(f"axes {axes!r} do not fit shape {shape}")
    if not axes.endswith("YX") or len(set(axes)) != len(axes):
        raise ValueError(f"axes {axes!r} must end in YX with no repeats")
    if any(a not in "CZ" for a in axes[:-2]):
        raise ValueError(f"axes {axes!r}: only C and Z may lead YX")
    dims = dict(zip(axes, shape))
    return max(1, min(dims.get("C", 1), MAX_CHANNELS))


def midz_index(axes: str, shape, channel: int) -> tuple:
    """The index of channel ``channel``'s mid-Z plane in an array of
    ``shape`` laid out as ``axes``."""
    axes = str(axes or "").upper()
    dims = dict(zip(axes, (int(s) for s in shape)))
    z_mid = dims.get("Z", 1) // 2
    idx = []
    for a in axes:
        if a == "C":
            idx.append(int(channel))
        elif a == "Z":
            idx.append(z_mid)
        else:
            idx.append(slice(None))
    return tuple(idx)


def _file_axes_disagree(file_axes: str, axes: str) -> bool:
    """True when the file names its own axes with C / Z / Y / X only and they
    are a DIFFERENT order of the same letters (C1-2: never a wrong picture).
    Files whose axes use other letters (``Q``, ``S``, ``I``...) are unlabelled
    and trust the resolved axes."""
    fa = str(file_axes or "").upper()
    ra = str(axes or "").upper()
    if not fa or any(a not in "CZYX" for a in fa):
        return False
    return sorted(fa) == sorted(ra) and fa != ra


def read_midz_planes(path: str, axes: str) -> np.ndarray:
    """Read ONLY the mid-Z plane(s) -> ``(n, Y, X)``.  Worker thread only.

    Reads through the first series' zarr view (level 0 of a pyramid), so only
    the chunks holding those planes are decoded.  Raises ``ValueError`` when
    the resolved axes do not fit the file."""
    import tifffile
    import zarr

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
        if _file_axes_disagree(series.axes, axes):
            raise ValueError(f"file axes {series.axes!r} != resolved {axes!r}")
        n = n_channels(axes, shape)
        store = series.aszarr()
        try:
            z = zarr.open(store, mode="r")
            if not hasattr(z, "shape"):          # a pyramid: take level 0
                z = z[sorted(z.array_keys())[0]]
            if tuple(z.shape) != shape:
                raise ValueError(f"zarr shape {z.shape} != series {shape}")
            planes = np.stack([np.asarray(z[midz_index(axes, shape, c)])
                               for c in range(n)])
        finally:
            try:
                store.close()
            except Exception:
                pass
    if planes.ndim != 3 or planes.shape[-2:] != shape[-2:]:
        raise ValueError(f"plane shape {planes.shape} != {shape[-2:]}")
    return planes


def read_labels_midz(path: str) -> np.ndarray:
    """The MID-Z plane of a labels file -> ``(Y, X)``.  Worker thread only.

    A TIFF is read through its first series' zarr view (one plane, not the
    volume); a ``.zarr`` store through ``zarr`` (a group: its first array).
    Any leading axes before ``Y, X`` are cut at their middle.  Raises on an
    unreadable file."""
    import zarr

    path = str(path)
    if path.lower().endswith(".zarr") or os.path.isdir(path):
        z = zarr.open(path, mode="r")
        if not hasattr(z, "shape"):
            z = z[sorted(z.array_keys())[0]]
        shape = tuple(int(s) for s in z.shape)
        idx = tuple(d // 2 for d in shape[:-2]) + (slice(None), slice(None))
        return np.asarray(z[idx])
    import tifffile

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
        store = series.aszarr()
        try:
            z = zarr.open(store, mode="r")
            if not hasattr(z, "shape"):
                z = z[sorted(z.array_keys())[0]]
            idx = tuple(d // 2 for d in shape[:-2]) + (slice(None), slice(None))
            return np.asarray(z[idx])
        finally:
            try:
                store.close()
            except Exception:
                pass


def has_labels_part(png_path) -> bool:
    """Does this PNG's NAME end in the labels part (``.l<ns>.o..<pct>.png``)?
    The tail is tested, not a ``.l`` anywhere: a stack called ``a.l1`` has
    one in its stem (audit L1)."""
    return bool(re.search(r"\.l\d+\.o[a-z]{2}\d+\.png$",
                          Path(str(png_path)).name)) if png_path else False


@functools.lru_cache(maxsize=4)
def label_rgb_table(n: int = LABEL_COLOURS, seed: float = LABEL_SEED):
    """The viewer's own colour of each label id: ``(n + 1, 3)`` uint8, row
    ``k`` the colour of a label whose ``((id - 1) % n) + 1`` is ``k``
    (napari's ``label_colormap(49, 0.5)``)."""
    from napari.utils.colormaps import label_colormap

    cm = label_colormap(int(n), float(seed))
    cols = np.asarray(cm.colors, dtype=np.float32)[:, :3]
    return np.clip(np.rint(cols * 255.0), 0, 255).astype(np.uint8)


def overlay_labels(rgb: np.ndarray, plane: np.ndarray, *,
                   mode: str = THUMB_OVERLAY_MODE,
                   alpha: float = THUMB_OVERLAY_ALPHA,
                   size: int = THUMB_SIZE) -> np.ndarray:
    """Draw the labels *plane* over *rgb* (``(h, w, 3)`` uint8) -> a new
    array.  Each instance takes the colour napari gives its id; ``mode``
    ``"fill+outline"`` blends that colour at *alpha* over the picture inside
    the label and paints the 1 px inner boundary at full strength,
    ``"outline"`` only the boundary, ``"fill"`` only the blend.  The plane
    is strided the way the picture was (``thumb_stride``) and, when its
    shape still differs from the picture's (a downsampled labels file),
    nearest-resampled onto the picture's grid.  Background (0) is never
    touched."""
    if mode not in _OVERLAY_CODES:
        raise ValueError(f"unknown overlay mode {mode!r}")
    # (spelled this way: the shared-modules import guard reads any dotted
    # part called ``segmentation`` as Step 4's package)
    from skimage import segmentation as _sk_seg

    find_boundaries = _sk_seg.find_boundaries

    plane = np.asarray(plane)
    if plane.ndim != 2:
        raise ValueError(f"expected a (Y, X) labels plane, got {plane.shape}")
    sp = thumb_stride(plane, int(size))
    h, w = rgb.shape[:2]
    if sp.shape != (h, w):
        yi = np.minimum((np.arange(h) * sp.shape[0]) // h, sp.shape[0] - 1)
        xi = np.minimum((np.arange(w) * sp.shape[1]) // w, sp.shape[1] - 1)
        sp = sp[np.ix_(yi, xi)]
    out = np.array(rgb, dtype=np.uint8, copy=True)
    inside = sp != 0
    if not inside.any():
        return out
    table = label_rgb_table()
    ids = np.abs(sp.astype(np.int64))
    cols = table[((ids - 1) % LABEL_COLOURS) + 1]
    if mode in (OVERLAY_FILL, OVERLAY_FILL_OUTLINE):
        a = float(alpha)
        blend = np.rint(out.astype(np.float32) * (1.0 - a)
                        + cols.astype(np.float32) * a)
        out[inside] = np.clip(blend[inside], 0, 255).astype(np.uint8)
    if mode in (OVERLAY_OUTLINE, OVERLAY_FILL_OUTLINE):
        edge = find_boundaries(sp, mode="inner") & inside
        out[edge] = cols[edge]
    return out


def encode_png(rgb: np.ndarray, size: int = THUMB_SIZE) -> bytes:
    """Scale an RGB array to ``size`` px with a ``QImage`` and return PNG
    bytes.  Safe off the main thread."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    buf = rgb.tobytes()
    img = QImage(buf, w, h, w * 3, QImage.Format_RGB888).copy()
    if max(h, w) != int(size):
        img = img.scaled(int(size), int(size), Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
    ba = QByteArray()
    qbuf = QBuffer(ba)
    qbuf.open(QIODevice.WriteOnly)
    try:
        if not img.save(qbuf, "PNG"):
            raise OSError("QImage.save to PNG failed")
    finally:
        qbuf.close()
    data = bytes(ba)
    if not data:
        raise OSError("empty PNG")
    return data


def _write_atomic(dest: Path, data: bytes) -> None:
    """Temp file + ``os.replace``; the temp file never survives a failure."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _sweep_stale(cache_dir, req: ThumbRequest, dest) -> None:
    """Delete this stack's PNGs of other keys.  Also on a cache HIT, so an
    older outlined picture does not survive beside the raw one after a
    Reset (A1 audit LOW-2)."""
    try:
        old_pngs = stale_pngs(cache_dir, req.name, req.path, dest)
    except OSError:
        return
    for old in old_pngs:
        try:
            old.unlink()
            _diag(f"deleted stale {old.name}")
        except OSError as exc:
            _diag(f"could not delete stale {old.name}: {exc}")


def render_thumb(req: ThumbRequest, cache_dir, size: int = THUMB_SIZE, *,
                 single=None):
    """One request -> ``(True, png_path, png_bytes)`` or ``(False, reason,
    b"")``.  Never raises.  ``png_path`` is ``""`` when the PNG could be made
    but not written (a read-only cache folder).

    A PNG already on disk for this key is returned without opening the TIFF.

    ``single`` is ``compose_rgb``'s one-channel rule (Step 3: ``"inferno"``).
    A request with a ``labels_path`` gets that file's mid-Z boundaries drawn
    over the picture (``overlay_labels``, A3); a labels file that cannot be
    read gives the raw picture and one console line, never a failure -- and
    the PNG is then named WITHOUT the outline part, so the next run tries
    again."""
    req = ThumbRequest(*req)
    lab_ns = int(req.labels_mtime_ns or 0) if req.labels_path else 0
    dest = thumb_png_path(cache_dir, req.name, req.path, req.mtime_ns,
                          req.st_size, single=single, labels_mtime_ns=lab_ns,
                          size=size)
    if dest.is_file():
        try:
            data = dest.read_bytes()
            if data:
                _diag(f"cache hit {req.name!r} -> {dest.name}")
                _sweep_stale(cache_dir, req, dest)
                return True, str(dest), data
        except OSError as exc:
            _diag(f"cache hit unreadable {dest.name}: {exc}")
    if not os.path.isfile(req.path):
        _diag(f"failed {req.name!r}: missing {req.path}")
        return False, THUMB_FAIL_MISSING, b""
    try:
        planes = read_midz_planes(req.path, req.axes)
    except ValueError as exc:
        _diag(f"failed {req.name!r}: axes {req.axes!r}: {exc}")
        return False, THUMB_FAIL_AXES, b""
    except Exception as exc:
        _diag(f"failed {req.name!r}: read: {type(exc).__name__}: {exc}")
        return False, THUMB_FAIL_UNREADABLE, b""
    try:
        rgb = compose_rgb(planes, size, single=single)
        if lab_ns:
            try:
                rgb = overlay_labels(rgb, read_labels_midz(req.labels_path),
                                     size=size)
            except Exception as exc:
                _diag(f"{req.name!r}: labels unreadable "
                      f"({type(exc).__name__}: {exc}); raw picture")
                lab_ns = 0
                dest = thumb_png_path(cache_dir, req.name, req.path,
                                      req.mtime_ns, req.st_size, single=single,
                                      size=size)
        data = encode_png(rgb, size)
    except Exception as exc:
        _diag(f"failed {req.name!r}: encode: {type(exc).__name__}: {exc}")
        return False, THUMB_FAIL_ENCODE, b""
    try:
        _write_atomic(dest, data)
    except Exception as exc:
        _diag(f"could not write {dest.name} ({type(exc).__name__}: {exc}); "
              "drawing from memory")
        return True, "", data
    _sweep_stale(cache_dir, req, dest)
    _diag(f"wrote {dest.name} ({rgb.shape[1]}x{rgb.shape[0]} from "
          f"{planes.shape[0]} channel(s))")
    return True, str(dest), data


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _StackThumbWorker(QObject):
    """Drains a queue of :class:`ThumbRequest` on its own thread, then ends.

    Every exit of :meth:`run` emits ``finished`` (``try/finally``); an
    unexpected exception also emits ``error`` first.  Once the queue has been
    found empty, or :meth:`cancel` ran, the queue is CLOSED: :meth:`enqueue`
    returns False and the caller starts a new worker -- a request is refused,
    never silently dropped.  The item in flight completes (a few TIFF
    pages)."""

    ready = Signal(str, str, str, bytes)   # name, stack path, png path, png bytes
    failed = Signal(str, str, str)         # name, stack path, reason
    error = Signal(str)
    finished = Signal()

    def __init__(self, requests, cache_dir, size: int = THUMB_SIZE, parent=None,
                 single=None):
        super().__init__(parent)
        self._single = single
        self._cache_dir = str(cache_dir)
        self._size = int(size)
        self._lock = threading.Lock()
        self._queue: deque = deque(ThumbRequest(*r) for r in requests)
        self._cancel = threading.Event()
        self._closed = False

    def enqueue(self, req) -> bool:
        """Add a request; False once the queue is closed (start a new worker)."""
        with self._lock:
            if self._closed:
                return False
            self._queue.append(ThumbRequest(*req))
            return True

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            self._closed = True
            n = len(self._queue)
            self._queue.clear()
        _diag(f"cancel: drained {n} queued request(s)")

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def _pop(self) -> Optional[ThumbRequest]:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            self._closed = True
            return None

    @Slot()
    def run(self) -> None:
        _diag(f"run start: {self.pending()} request(s)")
        done = 0
        try:
            sweep_stale_tmp(self._cache_dir)
            while not self._cancel.is_set():
                req = self._pop()
                if req is None:
                    break
                ok, payload, data = render_thumb(req, self._cache_dir, self._size,
                                             single=self._single)
                if self._cancel.is_set():
                    break
                if ok:
                    self.ready.emit(req.name, req.path, payload, data)
                else:
                    self.failed.emit(req.name, req.path, payload)
                done += 1
        except Exception as exc:  # a bug, not a bad stack
            _diag(f"run error: {type(exc).__name__}: {exc}")
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._closed = True
            _diag(f"run end: {done} done, cancelled={self._cancel.is_set()}")
            self.finished.emit()


class _ThumbThread(QThread):
    """A never-parented ``QThread`` that holds its worker, records that
    ``quit`` was reached, and leaves :data:`_LIVE_THUMB_WORKERS` once it has
    finished."""

    quit_called = False
    _thumb_worker = None

    @Slot()
    def quit(self) -> None:  # noqa: A003 - Qt's name
        self.quit_called = True
        super().quit()

    @Slot()
    def _forget(self) -> None:
        # ``finished`` can be delivered a hair before the thread has fully
        # stopped; wait (microseconds) so dropping the last reference never
        # destroys a running QThread.
        self.wait(2000)
        _LIVE_THUMB_WORKERS.discard(self)
        _diag(f"thread finished; {len(_LIVE_THUMB_WORKERS)} live")


_CALLER_SLOTS = ("_on_ready", "_on_failed", "_on_finished", "_on_error")


def start_thumb_worker(parent, requests, cache_dir, size: int = THUMB_SIZE, *,
                       on_ready=None, on_failed=None, on_finished=None,
                       on_error=None, single=None):
    """Build, wire and start one worker -> ``(thread, worker)``.

    ``parent`` is IGNORED: the thread is never parented, because deleting a
    parent widget would destroy a running thread.  It is kept alive in
    :data:`_LIVE_THUMB_WORKERS` until it finishes, and it holds the worker.

    ``on_ready(name, stack_path, png_path, png_bytes)`` /
    ``on_failed(name, stack_path, reason)`` / ``on_finished()`` /
    ``on_error(message)`` are connected BEFORE ``start()`` so a fast worker
    cannot emit into nothing; pass bound methods, not lambdas (a slot held only
    by PyQt can lose its closure when napari garbage-collects during layer
    removal).  :func:`stop_thumb_worker` disconnects them.  ``finished``
    AND ``error`` reach ``thread.quit`` (direct connection, so a busy main
    thread cannot hold the thread open)."""
    if parent is not None:
        _diag(f"parent {type(parent).__name__} ignored (threads are never parented)")
    thread = _ThumbThread()
    worker = _StackThumbWorker(requests, cache_dir, size, single=single)
    worker.moveToThread(thread)
    thread._thumb_worker = worker
    for attr, sig, fn in (("_on_ready", worker.ready, on_ready),
                          ("_on_failed", worker.failed, on_failed),
                          ("_on_finished", worker.finished, on_finished),
                          ("_on_error", worker.error, on_error)):
        setattr(worker, attr, fn)
        if fn is not None:
            sig.connect(fn)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.DirectConnection)
    worker.error.connect(thread.quit, Qt.DirectConnection)
    thread.finished.connect(thread._forget)
    _LIVE_THUMB_WORKERS.add(thread)
    _diag(f"start: {worker.pending()} request(s) -> {cache_dir}")
    thread.start()
    return thread, worker


def stop_thumb_worker(thread, worker=None, wait_ms: int = 3000) -> bool:
    """Cancel, disconnect the caller's slots, quit and wait (the step's
    ``shutdown()``).  True if the thread has stopped.  On a timeout the thread
    stays parked in :data:`_LIVE_THUMB_WORKERS`, never destroyed while
    running."""
    if worker is None and thread is not None:
        worker = getattr(thread, "_thumb_worker", None)
    if worker is not None:
        try:
            worker.cancel()
        except RuntimeError:
            pass  # already deleted
        for attr, sig_name in zip(_CALLER_SLOTS, ("ready", "failed", "finished", "error")):
            fn = getattr(worker, attr, None)
            if fn is None:
                continue
            try:
                getattr(worker, sig_name).disconnect(fn)
            except (TypeError, RuntimeError):
                pass
    try:
        if thread is None or not thread.isRunning():
            return True
        thread.quit()
        ok = bool(thread.wait(int(wait_ms)))
        _diag(f"stop: waited, stopped={ok}"
              + ("" if ok else " -- left parked in _LIVE_THUMB_WORKERS"))
        return ok
    except RuntimeError:
        return True


# ---------------------------------------------------------------------------
# Main-thread side
# ---------------------------------------------------------------------------

def _on_main_thread() -> bool:
    from qtpy.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    if app is None:
        return threading.current_thread() is threading.main_thread()
    return QThread.currentThread() == app.thread()


class ThumbCache:
    """LRU of ``QPixmap`` keyed by ``(stack path, mtime_ns, st_size)``.  Main
    thread only."""

    def __init__(self, maxsize: int = THUMB_CACHE_SIZE):
        self._maxsize = int(maxsize)
        self._d: OrderedDict = OrderedDict()

    @staticmethod
    def key(path, mtime_ns, st_size, extra=0):
        """``extra`` (Step 3, A1: the labels file's mtime) joins the key
        only when non-zero, so Step 4's keys are the 3-tuples they were."""
        k = (os.path.normcase(os.path.abspath(str(path))), int(mtime_ns),
             int(st_size))
        return k + (int(extra),) if extra else k

    def __len__(self):
        return len(self._d)

    def get(self, path, mtime_ns, st_size, extra=0):
        k = self.key(path, mtime_ns, st_size, extra)
        pm = self._d.get(k)
        if pm is not None:
            self._d.move_to_end(k)
        return pm

    def load(self, path, mtime_ns, st_size, *, png_path=None, png_bytes=None,
             extra=0):
        """PNG bytes (preferred) or a PNG path -> cached ``QPixmap``, or None.

        A PNG FILE that loads as a null pixmap is deleted, so the next run
        makes it again."""
        if not _on_main_thread():
            raise RuntimeError("ThumbCache.load must run on the main thread")
        from qtpy.QtGui import QPixmap

        pm = QPixmap()
        if png_bytes:
            pm.loadFromData(bytes(png_bytes), "PNG")
        if pm.isNull() and png_path:
            pm = QPixmap(str(png_path))
            if pm.isNull():
                _diag(f"cache: {png_path} did not load; deleting it")
                try:
                    os.remove(str(png_path))
                except OSError:
                    pass
        if pm.isNull():
            return None
        k = self.key(path, mtime_ns, st_size, extra)
        self._d[k] = pm
        self._d.move_to_end(k)
        while len(self._d) > self._maxsize:
            self._d.popitem(last=False)
        return pm

    def clear(self):
        self._d.clear()

    def discard_path(self, path) -> int:
        """Drop every picture of *path* (any mtime / size) -> how many; a
        forgotten patch's entries (Step 3's Forget)."""
        p = os.path.normcase(os.path.abspath(str(path)))
        gone = [k for k in self._d if k[0] == p]
        for k in gone:
            del self._d[k]
        return len(gone)
