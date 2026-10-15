"""The Step-1 preprocessing op registry — the single source of truth that drives
BOTH the ``+ add step`` menu and the executor.

napari-free BY DESIGN: pure metadata, no Qt, no napari, no op math.  The UI-side
placeholder ``napari_epicycle/data_preparation/ui.py::_OP_SUITE`` is what this
table replaces; the menu reads the *real* registry's visible set so the UI can
never offer an op the engine cannot run.

Design rules:

* **Steps bind by ``key``, not label** — a label rename never orphans a
  persisted recipe step, and the menu → recipe wiring maps display-label → key.
* **Show, don't hide** (decided 2026-08-06) — the menu shows ALL
  registered entries; the ``implemented=False`` ops render disabled / "— soon".
  Only the five v1 ops carry a runnable ``fn`` (added in Brick 2; ``None`` here).
* **Geometry → all channels** (``scope="all"``, one shared linked step);
  **intensity → per channel** (``scope="per_channel"``).

Brick 2a note: the FIVE v1 ops now carry a runnable ``fn`` (wired at the bottom
of this module from :mod:`ops`); the ~20 declared-only ops keep ``fn=None``.
Trivial ``halo`` callables are wired for the neighborhood ops (gaussian ~3σ,
background ~radius, clahe ~kernel) for L2's ``map_overlap``; the rest stay
``None``.  The geometry ``isotropic`` op's ``fn`` is a *reference* to
:func:`ops.apply_geometry` (a marker) — the executor routes geometry by
``kind == "geometry"`` (it needs the spacing + all channels), never via the
per-channel fn signature.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParamSpec:
    """One tunable knob of an op.

    ``type`` is one of ``"float"`` | ``"int"`` | ``"choice"``.  ``choices`` is a
    tuple of the allowed values for a ``"choice"`` param (``None`` otherwise).
    ``unit`` is a display-only suffix (e.g. ``"px"``, ``"µm"``).
    """

    name: str
    type: str
    default: object
    unit: str = ""
    choices: Optional[Tuple[object, ...]] = None


@dataclass
class OpSpec:
    """One registered preprocessing op — a row in :data:`REGISTRY`.

    ``key`` is the stable identity steps bind to; ``label`` is display-only.
    ``kind`` (``"geometry"`` | ``"intensity"``) drives all-channels vs
    per-channel; ``scope`` (``"all"`` | ``"per_channel"``) is the redundant,
    explicit projection of that same fact the executor keys on.  ``bucket`` is
    the menu group.  ``params`` is the ordered list of :class:`ParamSpec`.
    ``implemented`` is True only for ops that carry a runnable ``fn`` (Brick 2);
    ``fn`` / ``halo`` are ``None`` in Brick 1a.

    ``chunk_class`` is the op taxonomy (chunked-processing spec) that decides how
    the whole-volume process may parallelize the op: ``"local"`` (bounded
    neighborhood — gaussian, background — run block-parallel BYTE-IDENTICAL via
    ``chunked.block_parallel_apply`` when ``execute(parallel=True)``, needs a
    correct ``halo``); ``"pointwise"`` (per-voxel); ``"global_stat"`` (needs a
    whole-volume / per-plane statistic — bleach, min-max, histogram-equalize;
    NEVER a per-block stat); ``"adaptive"`` (CLAHE — whole-array, chunked CLAHE is
    never bit-identical); ``"geometry"`` (grid-changing resample, routed by
    ``kind``).  ``boundary`` is the true-edge contract for a ``"local"`` op
    (``"none"`` — defer the edge to the op's own native border; only consulted for
    ``"local"`` ops).
    """

    key: str
    label: str
    kind: str
    scope: str
    bucket: str
    params: List[ParamSpec] = field(default_factory=list)
    implemented: bool = False
    fn: Optional[Callable] = None
    halo: Optional[Callable] = None
    chunk_class: str = "adaptive"
    boundary: str = "none"


#: Menu group order.  ``by_bucket`` emits groups in this order; a bucket
#: with no ops is dropped from the result.
BUCKET_ORDER: Tuple[str, ...] = (
    "Geometry",
    "Denoise",
    "Contrast",
    "Background",
    "Depth / bleach",
    "Sharpen",
)


#: The registry table.  The FIVE v1 ops (``isotropic``, ``gaussian``, ``clahe``,
#: ``background``, ``bleach``) are ``implemented=True``; every other entry is
#: ported from the ``_OP_SUITE`` placeholder as ``implemented=False`` (declared,
#: shown-disabled).  Ordered by :data:`BUCKET_ORDER`.
REGISTRY: List[OpSpec] = [
    # ---- Geometry (all channels; the one shared linked step) -------------
    OpSpec(
        key="isotropic",
        label="Isotropic rescale",
        kind="geometry",
        scope="all",
        bucket="Geometry",
        params=[
            # Canonical tokens (see the ops.py docstring): the STORED
            # mode/method are lowercase — {"iso","aniso"} / {"nearest","linear",
            # "cubic","area"}.  Any prettier label is a UI-side display→token map.
            ParamSpec("mode", "choice", "iso", "", ("iso", "aniso")),
            ParamSpec("target_um", "float", 1.0, "µm"),
            ParamSpec("z", "float", 1.0, ""),
            ParamSpec("y", "float", 1.0, ""),
            ParamSpec("x", "float", 1.0, ""),
            ParamSpec(
                "method", "choice", "linear", "",
                ("nearest", "linear", "cubic", "area"),
            ),
        ],
        implemented=True,
    ),
    OpSpec(
        key="downsample",
        label="Downsample",
        kind="geometry",
        scope="all",
        bucket="Geometry",
        params=[
            ParamSpec("factor", "int", 2, "×"),
            ParamSpec("method", "choice", "mean", "", ("mean", "nearest")),
        ],
        implemented=False,
    ),
    # ---- Denoise (per channel) -------------------------------------------
    OpSpec(
        key="gaussian",
        label="Gaussian",
        kind="intensity",
        scope="per_channel",
        bucket="Denoise",
        params=[ParamSpec("sigma", "float", 1.5, "px")],
        implemented=True,
    ),
    OpSpec(
        key="median",
        label="Median",
        kind="intensity",
        scope="per_channel",
        bucket="Denoise",
        params=[ParamSpec("radius", "int", 2, "px")],
        implemented=False,
    ),
    OpSpec(
        key="bilateral",
        label="Bilateral",
        kind="intensity",
        scope="per_channel",
        bucket="Denoise",
        params=[
            ParamSpec("sigma_colour", "float", 0.1, ""),
            ParamSpec("sigma_space", "float", 2.0, "px"),
        ],
        implemented=False,
    ),
    OpSpec(
        key="nlm",
        label="Non-local means",
        kind="intensity",
        scope="per_channel",
        bucket="Denoise",
        params=[ParamSpec("h", "float", 0.8, "")],
        implemented=False,
    ),
    OpSpec(
        key="total_variation",
        label="Total variation",
        kind="intensity",
        scope="per_channel",
        bucket="Denoise",
        params=[ParamSpec("weight", "float", 0.1, "")],
        implemented=False,
    ),
    OpSpec(
        key="anisotropic_diffusion",
        label="Anisotropic diffusion",
        kind="intensity",
        scope="per_channel",
        bucket="Denoise",
        params=[ParamSpec("iterations", "int", 5, "")],
        implemented=False,
    ),
    OpSpec(
        key="wavelet",
        label="Wavelet",
        kind="intensity",
        scope="per_channel",
        bucket="Denoise",
        params=[ParamSpec("sigma", "float", 0.05, "")],
        implemented=False,
    ),
    # ---- Contrast / intensity (per channel) ------------------------------
    OpSpec(
        key="minmax_stretch",
        label="Min-max stretch",
        kind="intensity",
        scope="per_channel",
        bucket="Contrast",
        params=[
            ParamSpec("low_pct", "float", 0.1, ""),
            ParamSpec("high_pct", "float", 99.5, ""),
        ],
        implemented=False,
    ),
    OpSpec(
        key="gamma",
        label="Gamma",
        kind="intensity",
        scope="per_channel",
        bucket="Contrast",
        params=[ParamSpec("gamma", "float", 1.0, "")],
        implemented=False,
    ),
    OpSpec(
        key="log",
        label="Log",
        kind="intensity",
        scope="per_channel",
        bucket="Contrast",
        params=[],
        implemented=False,
    ),
    OpSpec(
        key="invert",
        label="Invert",
        kind="intensity",
        scope="per_channel",
        bucket="Contrast",
        params=[],
        implemented=False,
    ),
    OpSpec(
        key="histogram_equalize",
        label="Histogram equalize",
        kind="intensity",
        scope="per_channel",
        bucket="Contrast",
        params=[],
        implemented=False,
    ),
    OpSpec(
        key="clahe",
        label="CLAHE",
        kind="intensity",
        scope="per_channel",
        bucket="Contrast",
        params=[
            ParamSpec("clip", "float", 0.01, ""),
            ParamSpec("kernel", "int", 64, "px"),
        ],
        implemented=True,
    ),
    # ---- Background / illumination (per channel) -------------------------
    OpSpec(
        key="background",
        label="Rolling ball / white top-hat",
        kind="intensity",
        scope="per_channel",
        bucket="Background",
        params=[ParamSpec("radius", "int", 15, "px")],
        implemented=True,
    ),
    OpSpec(
        key="morphological_opening",
        label="Morphological opening",
        kind="intensity",
        scope="per_channel",
        bucket="Background",
        params=[ParamSpec("radius", "int", 3, "px")],
        implemented=False,
    ),
    OpSpec(
        key="flat_field",
        label="Flat-field correction",
        kind="intensity",
        scope="per_channel",
        bucket="Background",
        params=[ParamSpec("sigma", "float", 50.0, "px")],
        implemented=False,
    ),
    OpSpec(
        key="subtract_large_sigma_blur",
        label="Subtract large-sigma blur",
        kind="intensity",
        scope="per_channel",
        bucket="Background",
        params=[ParamSpec("sigma", "float", 30.0, "px")],
        implemented=False,
    ),
    # ---- Depth / bleach (per channel) ------------------------------------
    OpSpec(
        key="z_attenuation",
        label="Z-attenuation",
        kind="intensity",
        scope="per_channel",
        bucket="Depth / bleach",
        params=[ParamSpec("strength", "float", 1.0, "")],
        implemented=False,
    ),
    OpSpec(
        key="bleach",
        label="Bleach correction",
        kind="intensity",
        scope="per_channel",
        bucket="Depth / bleach",
        params=[
            ParamSpec("model", "choice", "exp", "", ("exp",)),
            # What to do when the fitted Z profile is NOT decreasing with depth
            # (flat / brightening — not a bleach signature): "skip" leaves it
            # unchanged (default), "flatten" applies the mean-reference rescale
            # in both directions anyway.  See ops.bleach.
            ParamSpec(
                "on_non_decreasing", "choice", "skip", "", ("skip", "flatten")
            ),
        ],
        implemented=True,
    ),
    OpSpec(
        key="per_slice_histogram_match",
        label="Per-slice histogram match",
        kind="intensity",
        scope="per_channel",
        bucket="Depth / bleach",
        params=[],
        implemented=False,
    ),
    # ---- Sharpen / feature enhancement (per channel) ---------------------
    OpSpec(
        key="unsharp_mask",
        label="Unsharp mask",
        kind="intensity",
        scope="per_channel",
        bucket="Sharpen",
        params=[
            ParamSpec("amount", "float", 1.0, ""),
            ParamSpec("sigma", "float", 1.0, "px"),
        ],
        implemented=False,
    ),
    OpSpec(
        key="high_pass",
        label="High-pass",
        kind="intensity",
        scope="per_channel",
        bucket="Sharpen",
        params=[ParamSpec("sigma", "float", 5.0, "px")],
        implemented=False,
    ),
    OpSpec(
        key="sobel",
        label="Sobel / gradient",
        kind="intensity",
        scope="per_channel",
        bucket="Sharpen",
        params=[],
        implemented=False,
    ),
    OpSpec(
        key="sato_frangi",
        label="Sato / Frangi ridge",
        kind="intensity",
        scope="per_channel",
        bucket="Sharpen",
        params=[ParamSpec("scale", "float", 2.0, "px")],
        implemented=False,
    ),
]


#: Op taxonomy (chunked-processing) — the ``chunk_class`` for every key,
#: applied to :data:`REGISTRY` below.  Only the IMPLEMENTED ops
#: (gaussian/background = ``"local"``, clahe = ``"adaptive"``, bleach =
#: ``"global_stat"``, isotropic = ``"geometry"``) affect runtime routing; the
#: rest are the taxonomy-in-code for future ops (inert — dropped by
#: ``normalize_recipe`` until implemented).  A key absent here keeps the field
#: default (``"adaptive"`` = whole-array, the safe class).
#:
#: WARNING for future implementers: the labels below for UNIMPLEMENTED ops are
#: PROVISIONAL and MUST be re-derived + seam-tested (byte-identical chunked==whole)
#: when the op gets a real ``fn``.  In particular ``nlm`` (non-local means — a
#: non-local SEARCH window, not a bounded neighborhood) and ``flat_field`` (an
#: image-ESTIMATED flat field is large-reach/global) are NOT truly ``"local"``;
#: do not trust the label — verify the reach before routing them block-parallel.
_CHUNK_CLASS: Dict[str, str] = {
    # geometry (grid-changing resample → routed by kind, not the block harness)
    "isotropic": "geometry",
    "downsample": "geometry",
    # local (bounded neighborhood → block-parallel byte-identical with a halo)
    "gaussian": "local",
    "median": "local",
    "bilateral": "local",
    "nlm": "local",
    "total_variation": "local",
    "anisotropic_diffusion": "local",
    "background": "local",
    "morphological_opening": "local",
    "flat_field": "local",
    "subtract_large_sigma_blur": "local",
    "unsharp_mask": "local",
    "high_pass": "local",
    "sobel": "local",
    "sato_frangi": "local",
    # pointwise (per-voxel → chunkable trivially; not yet parallelized)
    "gamma": "pointwise",
    "log": "pointwise",
    "invert": "pointwise",
    # global_stat (whole-volume / per-plane statistic → never a per-block stat)
    "minmax_stretch": "global_stat",
    "histogram_equalize": "global_stat",
    "wavelet": "global_stat",
    "z_attenuation": "global_stat",
    "bleach": "global_stat",
    "per_slice_histogram_match": "global_stat",
    # adaptive (CLAHE — whole-array; chunked CLAHE is never bit-identical)
    "clahe": "adaptive",
}
for _op in REGISTRY:
    _op.chunk_class = _CHUNK_CLASS.get(_op.key, _op.chunk_class)


#: key → OpSpec index (built once).  Guards against a duplicate key silently
#: shadowing an earlier row.
def _build_index(registry: List[OpSpec]) -> Dict[str, OpSpec]:
    index: Dict[str, OpSpec] = {}
    for op in registry:
        if op.key in index:
            raise ValueError(
                f"duplicate op key {op.key!r} in preprocessing REGISTRY"
            )
        index[op.key] = op
    return index


_BY_KEY: Dict[str, OpSpec] = _build_index(REGISTRY)


def get(key: str) -> OpSpec:
    """Return the :class:`OpSpec` for *key* (raises ``KeyError`` if unknown)."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"preprocessing registry has no op {key!r}") from None


def find(key: str) -> Optional[OpSpec]:
    """Tolerant lookup: the :class:`OpSpec` for *key*, or ``None`` if unknown.

    The executor + :func:`recipe.normalize_recipe` use this (never raise on a
    stale / newer-branch key — skip + diagnose instead).
    """
    return _BY_KEY.get(key)


def by_bucket() -> "OrderedDict[str, List[OpSpec]]":
    """Group the registry by menu bucket, ordered by :data:`BUCKET_ORDER`.

    Returns an ``OrderedDict`` of ``bucket_name -> [OpSpec, ...]`` (ops in
    registry order within each bucket).  Empty buckets are omitted; any bucket
    not in ``BUCKET_ORDER`` is appended after the known ones.
    """
    out: "OrderedDict[str, List[OpSpec]]" = OrderedDict(
        (b, []) for b in BUCKET_ORDER
    )
    for op in REGISTRY:
        out.setdefault(op.bucket, []).append(op)
    return OrderedDict((b, ops) for b, ops in out.items() if ops)


def implemented_keys() -> List[str]:
    """The keys of the runnable (``implemented=True``) ops — the v1 five."""
    return [op.key for op in REGISTRY if op.implemented]


# --------------------------------------------------------------------------- #
# Brick 2a — wire the runnable fns + trivial halos from :mod:`ops`.
# --------------------------------------------------------------------------- #
# Deferred to the bottom (after REGISTRY / _BY_KEY exist) so that the circular
# import resolves: ``ops`` imports this module for ``find`` at CALL time only,
# while this ``from . import ops`` sees an already-populated registry.  ``ops``
# imports scipy/skimage LAZILY, so importing the registry stays cheap.

def _wire_ops() -> None:
    from . import ops  # local import → breaks the import cycle cleanly

    _BY_KEY["gaussian"].fn = ops.gaussian
    _BY_KEY["gaussian"].halo = ops._halo_gaussian
    _BY_KEY["clahe"].fn = ops.clahe
    _BY_KEY["clahe"].halo = ops._halo_clahe
    _BY_KEY["background"].fn = ops.background
    _BY_KEY["background"].halo = ops._halo_background
    _BY_KEY["bleach"].fn = ops.bleach
    # Geometry is the shared, all-channels step — its fn is a *reference* to the
    # geometry executor; the executor routes it by kind, not per-channel call.
    _BY_KEY["isotropic"].fn = ops.apply_geometry


_wire_ops()
