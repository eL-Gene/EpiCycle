"""The prediction window's reads and pictures (Qt-free).

A patch's prediction cache holds, per pass (``xy`` / ``xz`` / ``yz``), the
Cellpose cell-probability logits ``{stem}_{pass}_probs.npy`` (float16,
``(Z, Y, X)``) and the flows ``{stem}_{pass}_flows.npy`` (float16,
``(2, Z, Y, X)``).  All three passes are stored in the patch's own Z, Y, X
frame, so the plane a pass was predicted in is a different cut of the same
volume: XY is ``vol[z]`` (Y x X), XZ is ``vol[:, y]`` (Z x X), YZ is
``vol[:, :, x]``.  The two flow components are each pass's own in-plane
pair: (dy, dx) for XY, (dz, dx) for XZ, (dz, dy) for YZ.  The YZ plane is
shown turned to the annotator's YZ panel (Y down, Z across: its data is
the volume transposed (2, 1, 0)), its two components swapped to match.

Everything here runs on the window's worker thread:

* :func:`read_header` reads a ``.npy`` file's shape from its header (a
  ``with open()`` block), so a patch above ``PRED_MAX_VOXELS`` is refused
  before any array is read;
* :func:`load_map` reads ONE file fully into memory (never mapped: a mapped
  file cannot be replaced or moved on Windows while the window is open) and
  keeps it -- the probability as an 8-bit volume through the fixed
  ``PRED_PROB_LIMITS``, the flows as the stored two components with the
  1-99 % range of their length (``cellpose.plot.dx_to_circ``'s stretch,
  taken once for the volume);
* :func:`model_input` rebuilds what Cellpose was given from the recorded
  channel codes, by Cellpose's own rule (``cellpose.transforms.reshape``):
  code 0 is the mean of every channel, code k is channel k-1, a helper code
  h > 0 adds channel h-1, and a one-channel image is that channel whatever
  the codes say.  The brightness is stretched once for the whole patch
  (1-99 %), not slice by slice as Cellpose normalises.

The main thread only cuts one plane per pane (:func:`plane`,
:func:`flow_plane`), thins it to about twice the pane's size
(:func:`thin`: a step through the rows and columns, so a slider move costs
the same at any patch size) and colours it (:func:`render_prob`,
:func:`render_flow` -- the flow angle per drawn plane --,
:func:`render_image`); every picture is a C-contiguous
``uint8 (rows, cols, 3)`` array.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from . import _step3_pr_text as T

#: the passes, in the window's order
PASSES = ("xy", "xz", "yz")
#: the two maps of a pass (the file names' last word)
KINDS = ("probs", "flows")
#: the probability's fixed colour range (Cellpose logits)
PRED_PROB_LIMITS = (-6.0, 6.0)
#: the largest patch the window reads (Z x Y x X)
PRED_MAX_VOXELS = 200_000_000
#: the slider's axis per pass: the plane follows the pass
SLIDER_AXIS = {"xy": "z", "xz": "y", "yz": "x"}
_PLANE_AXIS = {"xy": 0, "xz": 1, "yz": 2}
#: planes per chunk for the volume-wide passes (the mean, the 8-bit cast)
_Z_CHUNK = 16
#: the brightness percentiles are taken over every Nth plane
_SAMPLE_STEP = 4


class PreviewTooLarge(ValueError):
    """The patch has more than ``PRED_MAX_VOXELS`` voxels; ``shape`` is its
    Z, Y, X."""

    def __init__(self, shape):
        super().__init__(tuple(shape))
        self.shape = tuple(int(v) for v in tuple(shape)[-3:])


def too_large_sentence(shape) -> str:
    """The refusal, with the patch's size and the limit."""
    z, y, x = (tuple(int(v) for v in tuple(shape)[-3:]) + (0, 0, 0))[:3]
    return T.S3_PRED_TOO_LARGE.format(
        size=f"{z} × {y} × {x}",
        limit=f"{PRED_MAX_VOXELS // 1_000_000:,}")


# ── files ──────────────────────────────────────────────────────────────────
def map_path(cache_dir, stem: str, pass_: str, kind: str) -> Path:
    """``{cache_dir}/{stem}_{pass}_{kind}.npy``."""
    if pass_ not in PASSES or kind not in KINDS:
        raise ValueError(f"unknown pass / map: {pass_!r} / {kind!r}")
    return Path(cache_dir) / f"{stem}_{pass_}_{kind}.npy"


def read_header(path) -> Tuple[int, ...]:
    """The array shape in a ``.npy`` file's header; the file is closed
    before this returns and no array data is read."""
    fmt = np.lib.format
    with open(path, "rb") as fh:
        version = fmt.read_magic(fh)
        if version == (1, 0):
            shape, _order, _dtype = fmt.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, _order, _dtype = fmt.read_array_header_2_0(fh)
        else:
            shape, _order, _dtype = fmt._read_array_header(fh, version)
    return tuple(int(s) for s in shape)


def voxels(shape: Sequence[int]) -> int:
    """Z x Y x X of a probability (3-D) or flow (4-D) shape."""
    dims = [int(s) for s in tuple(shape)[-3:]]
    return int(math.prod(dims)) if dims else 0


def check_size(shape: Sequence[int]) -> None:
    if voxels(shape) > PRED_MAX_VOXELS:
        raise PreviewTooLarge(tuple(shape))


def _load_full(path) -> np.ndarray:
    """The whole array in memory; the file is closed when this returns."""
    return np.load(str(path), allow_pickle=False)


# ── 8-bit volumes ──────────────────────────────────────────────────────────
def contrast_bounds(vol: np.ndarray, *, low: float = 1.0,
                    high: float = 99.0) -> Tuple[float, float]:
    """The 1-99 % brightness range over every ``_SAMPLE_STEP``-th plane."""
    sample = np.asarray(vol[::_SAMPLE_STEP], dtype=np.float32)
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(sample, (low, high))
    return float(lo), float(hi)


def to_uint8(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """*vol* stretched from [*lo*, *hi*] to 0-255, one chunk of planes at a
    time (no full-volume float copy)."""
    out = np.empty(vol.shape, dtype=np.uint8)
    span = float(hi) - float(lo)
    if span <= 0:
        out[...] = 0
        return out
    for z0 in range(0, vol.shape[0], _Z_CHUNK):
        chunk = np.asarray(vol[z0:z0 + _Z_CHUNK], dtype=np.float32)
        chunk = (chunk - float(lo)) * (255.0 / span)
        out[z0:z0 + _Z_CHUNK] = np.clip(np.rint(chunk), 0, 255)
    return out


def prob_to_uint8(probs: np.ndarray) -> np.ndarray:
    """The logits through the fixed ``PRED_PROB_LIMITS`` to 0-255."""
    lo, hi = PRED_PROB_LIMITS
    return to_uint8(probs, lo, hi)


def _norm99_bounds(mag_sample: np.ndarray) -> Tuple[float, float]:
    """``cellpose.transforms.normalize99``'s two percentiles."""
    if mag_sample.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(mag_sample, (1, 99))
    return float(lo), float(hi)


def flow_bounds(flows: np.ndarray) -> Tuple[float, float]:
    """The 1-99 % range of the flows' length (``normalize99``), over every
    ``_SAMPLE_STEP``-th plane of a pass's ``(2, Z, Y, X)`` components."""
    sample = flows[:, ::_SAMPLE_STEP].astype(np.float32)
    return _norm99_bounds(np.sqrt(np.sum(sample ** 2, axis=0)))


def load_map(cache_dir, stem: str, pass_: str, kind: str) -> Dict[str, Any]:
    """ONE map of one pass, as 8-bit volumes: ``{"pass", "kind", "shape"}``
    plus ``"prob"`` or ``"angle"`` / ``"mag"``.  The header is read first
    (the size limit), then the file in full; the float array is dropped
    before this returns."""
    path = map_path(cache_dir, stem, pass_, kind)
    check_size(read_header(path))
    arr = _load_full(path)
    out: Dict[str, Any] = {"pass": pass_, "kind": kind}
    if kind == "probs":
        if arr.ndim != 3:
            raise ValueError(f"probabilities of shape {arr.shape}")
        out["shape"] = tuple(int(s) for s in arr.shape)
        out["prob"] = prob_to_uint8(arr)
        del arr
    else:
        if arr.ndim != 4 or arr.shape[0] != 2:
            raise ValueError(f"flows of shape {arr.shape}")
        out["shape"] = tuple(int(s) for s in arr.shape[1:])
        # the components as stored (float16): the angle is taken per drawn
        # plane, the length's range once here
        out["comps"] = arr
        out["lo"], out["hi"] = flow_bounds(arr)
    return out


# ── what the model saw ─────────────────────────────────────────────────────
def to_zcyx(image: np.ndarray, axes: Optional[str]) -> np.ndarray:
    """*image* as Z, C, Y, X -- the order the predict worker hands Cellpose
    its slices in (a 3-D image gains a channel axis of 1)."""
    if image.ndim == 3:
        return image[:, np.newaxis, :, :]
    if image.ndim != 4:
        raise ValueError(f"expected a 3-D or 4-D image, got {image.shape}")
    order = (axes or "").upper()
    if order == "CZYX":
        return image.transpose(1, 0, 2, 3)
    if order == "ZCYX":
        return image
    raise ValueError(f"unsupported axes {axes!r}")


def _codes(codes) -> Tuple[Optional[int], int]:
    try:
        c1 = None if codes[0] is None else int(codes[0])
    except (TypeError, IndexError, ValueError):
        c1 = None
    try:
        c2 = int(codes[1])
    except (TypeError, IndexError, ValueError):
        c2 = 0
    return c1, max(c2, 0)


def _mean_channels(zcyx: np.ndarray) -> np.ndarray:
    """The mean of every channel in float32, a chunk of planes at a time
    (Cellpose casts to float32 before it averages)."""
    nz = zcyx.shape[0]
    out = np.empty((nz,) + zcyx.shape[2:], dtype=np.float32)
    for z0 in range(0, nz, _Z_CHUNK):
        out[z0:z0 + _Z_CHUNK] = np.mean(zcyx[z0:z0 + _Z_CHUNK], axis=1,
                                        dtype=np.float32)
    return out


def _u8(vol: np.ndarray) -> np.ndarray:
    lo, hi = contrast_bounds(vol)
    return to_uint8(vol, lo, hi)


def model_input(image: np.ndarray, axes: Optional[str], codes,
                one_channel: bool = False) -> Dict[str, Any]:
    """What Cellpose was given, as 8-bit ``(Z, Y, X)`` volumes:
    ``{"mode", "seg", "helper", "shape", "codes"}``.

    *mode* is ``"one"`` (a one-channel image, or a patch recorded as run as
    one channel: grey), ``"avg"`` (code 0: the mean of every channel; the
    helper is ignored), ``"seg"`` (code k: channel k-1, grey) or ``"pair"``
    (code k and helper h > 0: channel k-1 drawn green over channel h-1
    drawn red).  *codes* ``None`` (no channels recorded) is shown as code
    1, the first channel, and marked ``"guessed"``.
    """
    zcyx = to_zcyx(image, axes)
    n_ch = int(zcyx.shape[1])
    guessed = codes is None or _codes(codes)[0] is None
    c1, c2 = _codes(codes if codes is not None else (1, 0))
    if c1 is None:
        c1, c2 = 1, 0
    helper = None
    if n_ch == 1:
        mode, seg = "one", zcyx[:, 0]
    elif one_channel:
        mode, seg = "one", _mean_channels(zcyx)
    elif c1 == 0:
        mode, seg = "avg", _mean_channels(zcyx)
    else:
        if c1 - 1 >= n_ch:
            raise ValueError(
                f"channel code {c1} on an image with {n_ch} channels")
        seg = zcyx[:, c1 - 1]
        if 0 < c2 <= n_ch:
            mode, helper = "pair", zcyx[:, c2 - 1]
        else:
            mode = "seg"
    return {"mode": mode, "seg": _u8(seg),
            "helper": None if helper is None else _u8(helper),
            "shape": tuple(int(s) for s in seg.shape), "codes": (c1, c2),
            "guessed": bool(guessed)}


def load_image(stack_path, axes: Optional[str], codes,
               one_channel: bool = False,
               resolve: Optional[Callable] = None) -> Dict[str, Any]:
    """Read the patch (``tifffile.imread``) and rebuild the model's input.
    A 4-D image with no recorded axes is resolved as the predict worker
    resolves it (``qc_features.resolve_patch_axes``)."""
    import tifffile
    image = tifffile.imread(str(stack_path))
    if image.ndim == 4 and str(axes or "").upper() not in ("ZCYX", "CZYX"):
        if resolve is None:
            from .qc_features import resolve_patch_axes as resolve
        axes = resolve(image, image_path=str(stack_path), fallback="CZYX")
    out = model_input(image, axes, codes, one_channel)
    del image
    return out


# ── the window's words ─────────────────────────────────────────────────────
def _item(items, code) -> str:
    try:
        return str(items[int(code)])
    except (TypeError, IndexError, ValueError):
        return str(code)


def _when(stamp) -> str:
    """``predicted_at`` (UTC) in local time, ``YYYY-MM-DD HH:MM``."""
    try:
        t = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return T.PROV_EMPTY
    try:
        t = t.astimezone()
    except (OSError, ValueError, OverflowError):
        pass
    return t.strftime("%Y-%m-%d %H:%M")


def _model_name(manifest: Dict[str, Any]) -> str:
    from . import _model_provenance as MP
    block = manifest.get(MP.PROVENANCE_KEY)
    short = MP.short_name(block)
    if not short:
        return T.PROV_EMPTY
    lat = (block.get("views") or {}).get("lateral") \
        if isinstance(block, dict) else None
    if isinstance(lat, dict) and lat.get("kind") == "pretrained":
        return T.PRED_MODEL_BUILTIN.format(model=short)
    return short


def recorded(manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``{"codes", "axes", "one_channel"}`` from a prediction manifest
    (``None`` codes / axes when there is none, or when the manifest records
    the channel to segment as blank: ``channels: null``).  A manifest
    without the ``one_channel`` key ran on every channel it was asked to."""
    if not isinstance(manifest, dict):
        return {"codes": None, "axes": None, "one_channel": False}
    seg = manifest.get("seg_config") or {}
    codes = seg.get("channels") if isinstance(seg, dict) else None
    if not isinstance(codes, (list, tuple)) or len(codes) < 2:
        codes = None
    axes = manifest.get("axes")
    return {"codes": None if codes is None else list(codes),
            "axes": axes if isinstance(axes, str) else None,
            "one_channel": manifest.get("one_channel") is True}


def window_lines(manifest: Optional[Dict[str, Any]], run: str) -> Dict[str, str]:
    """The provenance's two lines: ``{"line1", "line2"}``.  *run* is the
    run's display name.  A patch with no prediction manifest (predicted
    before its settings were recorded) has line 1 only.  A one-channel
    patch's line 2 says what ran (0: gray), never the codes asked for.  A
    manifest whose channel to segment is blank names it "blank"."""
    run = str(run or T.PROV_EMPTY)
    rec = recorded(manifest)
    if not isinstance(manifest, dict):
        return {"line1": T.PRED_LINE_NO_RECORD.format(run=run), "line2": ""}
    line1 = T.PRED_LINE1.format(run=run, model=_model_name(manifest),
                                when=_when(manifest.get("predicted_at")))
    if rec["one_channel"]:
        return {"line1": line1, "line2": T.PRED_LINE2_ONE_CHANNEL}
    raw = (manifest.get("seg_config") or {}).get("channels") \
        if isinstance(manifest.get("seg_config"), dict) else None
    if rec["codes"] is not None:
        c1, c2 = _codes(rec["codes"])
    elif isinstance(raw, (list, tuple)):
        c1, c2 = _codes(list(raw) + [0, 0])
    else:
        c1, c2 = None, 0
    helper = _item(T.S3_CHAN2_ITEMS, c2)
    if c1 == 0 and c2 > 0:
        helper += T.PRED_HELPER_UNUSED
    seg = T.PRED_CHANNEL_BLANK if c1 is None else _item(T.S3_CHAN1_ITEMS, c1)
    line2 = T.PRED_LINE2.format(seg=seg, helper=helper)
    return {"line1": line1, "line2": line2}


def left_caption(image: Optional[Dict[str, Any]]) -> str:
    """The left pane's caption, in the channel boxes' words."""
    if not image:
        return ""
    mode = image.get("mode")
    c1, c2 = image.get("codes") or (None, 0)
    if mode == "one":
        return T.PRED_LEFT_ONE_CHANNEL
    if mode == "avg":
        return T.PRED_LEFT_GRAY_AVG
    seg = _item(T.S3_CHAN1_ITEMS, c1)
    if image.get("guessed"):
        return T.PRED_LEFT_NO_RECORD.format(seg=seg)
    if mode == "pair":
        return T.PRED_LEFT_CAPTION_PAIR.format(
            seg=seg, helper=_item(T.S3_CHAN2_ITEMS, c2))
    return T.PRED_LEFT_CAPTION.format(seg=seg)


def load_open(cache_dir, stem: str, stack_path, manifest_path,
              run: str, pass_: str = "xy", kind: str = "probs",
              resolve: Optional[Callable] = None) -> Dict[str, Any]:
    """The window's first read: the size limit (the chosen map's header),
    the manifest, the patch rebuilt as the model saw it and the chosen map.
    ``{"lines", "image", "caption", "map", "shape", "shape_changed"}``;
    a patch whose shape no longer matches its prediction keeps the map only
    (``image`` None, ``shape_changed`` True)."""
    from . import _step3_manifests as S
    first = map_path(cache_dir, stem, pass_, kind)
    check_size(read_header(first))
    manifest = S.read_prediction_manifest(manifest_path)
    rec = recorded(manifest)
    try:
        image = load_image(stack_path, rec["axes"], rec["codes"],
                           rec["one_channel"], resolve=resolve)
    except ValueError as exc:
        # e.g. a channel the patch no longer has: the patch changed
        print(f"[STEP3-PRED] {stem}: the patch no longer reads as predicted "
              f"({exc})")
        image = None
    the_map = load_map(cache_dir, stem, pass_, kind)
    changed = image is None or (
        tuple(image["shape"]) != tuple(the_map["shape"]))
    if changed:
        image = None
    return {"lines": window_lines(manifest, run), "image": image,
            "caption": left_caption(image), "map": the_map,
            "shape": tuple(the_map["shape"]), "shape_changed": changed}


# ── planes and pictures (main thread) ──────────────────────────────────────
def plane_count(shape: Sequence[int], pass_: str) -> int:
    """How many planes a pass has: Z for XY, Y for XZ, X for YZ."""
    return int(tuple(shape)[_PLANE_AXIS[pass_]])


def _clamp(shape, pass_: str, i: int) -> int:
    n = plane_count(shape, pass_)
    return min(max(int(i), 0), max(n - 1, 0))


def plane(vol: np.ndarray, pass_: str, i: int) -> np.ndarray:
    """The pass's plane *i* of a ``(Z, Y, X)`` volume as shown: XY
    ``vol[i]`` (Y down, X across), XZ ``vol[:, i]`` (Z down, X across), YZ
    ``vol[:, :, i]`` turned to the annotator's YZ panel (Y down, Z
    across)."""
    i = _clamp(vol.shape, pass_, i)
    if pass_ == "xy":
        return vol[i]
    if pass_ == "xz":
        return vol[:, i, :]
    return vol[:, :, i].T


def flow_plane(comps: np.ndarray, pass_: str, i: int):
    """``(down, across)`` flow components of the pass's plane *i*, in the
    plane as shown (the YZ pass's stored (dz, dy) become (dy, dz) once the
    plane is turned)."""
    a = plane(comps[0], pass_, i)
    b = plane(comps[1], pass_, i)
    return (b, a) if pass_ == "yz" else (a, b)


def thin_step(shape2d, max_px: int) -> int:
    """The row / column step that brings a plane to at most about
    *max_px* pixels a side."""
    longest = max(int(v) for v in tuple(shape2d)[:2]) if shape2d else 1
    return max(1, int(math.ceil(longest / float(max(int(max_px), 1)))))


def thin(arr: np.ndarray, step: int) -> np.ndarray:
    """Every *step*-th row and column of a plane."""
    return arr if step <= 1 else arr[::step, ::step]


def render_prob(plane_u8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """An 8-bit probability plane through a ``(256, 3)`` colour table."""
    return np.ascontiguousarray(lut[plane_u8], dtype=np.uint8)


def render_flow(down: np.ndarray, across: np.ndarray, lo: float,
                hi: float) -> np.ndarray:
    """``cellpose.plot.dx_to_circ``'s colours of one plane's two
    components, with the length's 1-99 % range *lo* / *hi* given (taken
    once for the volume): brightness = the stretched length / 2, hue = the
    angle, computed here for the drawn plane only."""
    d0 = np.asarray(down, dtype=np.float32)
    d1 = np.asarray(across, dtype=np.float32)
    length = np.sqrt(d0 ** 2 + d1 ** 2)
    if hi - lo > 1e-3:
        norm = (length - np.float32(lo)) / np.float32(hi - lo)
    else:
        norm = np.zeros_like(length)
    mag = 255 * np.clip(norm, 0, 1.)
    angles = np.arctan2(d1, d0) + np.pi
    mag /= 2
    rgb = np.zeros(d0.shape + (3,), "uint8")
    rgb[..., 0] = np.clip(mag * (np.cos(angles) + 1), 0, 255).astype("uint8")
    rgb[..., 1] = np.clip(mag * (np.cos(angles + 2 * np.pi / 3) + 1), 0,
                          255).astype("uint8")
    rgb[..., 2] = np.clip(mag * (np.cos(angles + 4 * np.pi / 3) + 1), 0,
                          255).astype("uint8")
    return rgb


def render_image(seg: np.ndarray, helper: Optional[np.ndarray] = None
                 ) -> np.ndarray:
    """The left pane: the channel to segment in grey, or -- with a helper
    -- the channel to segment green over the helper red."""
    if helper is None:
        return np.ascontiguousarray(np.repeat(seg[..., np.newaxis], 3,
                                              axis=-1), dtype=np.uint8)
    rgb = np.zeros(seg.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = helper
    rgb[..., 1] = seg
    return rgb
