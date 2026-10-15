"""The Step-1 preprocessing recipe data-model.

The **recipe is the persisted + executed source of truth**; the per-channel
column UI is a *projection* of it.  A recipe is:

* ``channels`` — one ordered list of :class:`IntensityStep` PER channel index
  (``len == n_channels``), and
* ``geometry`` — exactly ONE shared :class:`GeometryStep` (or ``None``), whose
  ``boundary_index`` is the shared split position: it inserts the geometry op at
  the *same* relative index in every channel.  Moving the linked chip
  re-segments all channels identically.

Flattening (``flatten_for_channel``) produces the deterministic, per-channel
execution order: intensity ops before the boundary, then the shared geometry
op, then the ops after.  The default recipe flattens to
``[gaussian, isotropic, background]`` when a channel is ``[gaussian, background]``
with a boundary of 1 (the proven Fiji order).

napari-free BY DESIGN: pure dataclasses + dict (de)serialization, no Qt / napari
/ numpy.  :func:`normalize_recipe` is the corrupt-tolerant validator (mirrors
``packages/epicycle_largedata/config.py::normalize_convert_config``): it drops
unknown / not-yet-implemented steps, clamps the channel count + boundary index,
coerces param types against the registry, and NEVER raises.

Brick 1a: model + validators only — no op math, no executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from . import registry as _registry
from .registry import OpSpec, ParamSpec


# --------------------------------------------------------------------------- #
# Step / recipe dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class IntensityStep:
    """A per-channel intensity op instance — binds to the registry by ``key``."""

    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryStep:
    """The ONE shared, linked geometry op instance.

    ``boundary_index`` is the shared split position across every channel.
    ``config`` holds the geometry-specific settings::

        {mode: "iso" | "aniso",
         target_um: float,
         factors: (z, y, x),
         method: str}
    """

    key: str = "isotropic"
    boundary_index: int = 0
    config: Dict[str, Any] = field(default_factory=lambda: default_geometry_config())


@dataclass
class Recipe:
    """A source's full preprocessing recipe: per-channel intensity lists + one
    shared geometry step."""

    channels: List[List[IntensityStep]] = field(default_factory=list)
    geometry: Optional[GeometryStep] = None


# --------------------------------------------------------------------------- #
# Seeds / defaults
# --------------------------------------------------------------------------- #

def default_geometry_config() -> Dict[str, Any]:
    """The default isotropic-rescale config (mode=iso, unit factors)."""
    return {
        "mode": "iso",
        "target_um": 1.0,
        "factors": (1.0, 1.0, 1.0),
        "method": "linear",
    }


def _registry_defaults(key: str) -> Dict[str, Any]:
    """Default param dict for *key* pulled straight from the registry (so the
    seed never drifts from the registry's declared defaults)."""
    op = _registry.find(key)
    if op is None:
        return {}
    return {p.name: p.default for p in op.params}


def default_recipe(n_channels: int) -> Recipe:
    """Seed recipe: per channel ``[gaussian]`` + a shared ``isotropic`` geometry
    step at boundary 1, so each channel flattens to ``[gaussian, isotropic]``
    (the default order: Gaussian → isotropic → …).  Add a per-channel
    background step later and it flattens to ``[gaussian, isotropic, background]``.
    """
    n = max(0, _as_int(n_channels, 0))
    channels: List[List[IntensityStep]] = [
        [IntensityStep(key="gaussian", params=_registry_defaults("gaussian"))]
        for _ in range(n)
    ]
    geometry = GeometryStep(
        key="isotropic",
        boundary_index=1,
        config=default_geometry_config(),
    )
    return Recipe(channels=channels, geometry=geometry)


# --------------------------------------------------------------------------- #
# Flatten (columns → flat per-channel execution order)
# --------------------------------------------------------------------------- #

def flatten_for_channel(
    recipe: Recipe, ci: int
) -> List[Union[IntensityStep, GeometryStep]]:
    """The channel ``ci``'s ordered execution pipeline: its intensity ops with
    the shared geometry step inserted at ``min(geometry.boundary_index, len(ops))``.

    Returns a mixed list of :class:`IntensityStep` / :class:`GeometryStep`; read
    ``.key`` off each element for the flat op sequence.  ``geometry is None`` →
    just the channel's intensity ops.
    """
    ops: List[Union[IntensityStep, GeometryStep]] = list(recipe.channels[ci])
    geom = recipe.geometry
    if geom is None:
        return ops
    idx = _as_int(geom.boundary_index, 0)
    idx = max(0, min(idx, len(ops)))
    return ops[:idx] + [geom] + ops[idx:]


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #

def to_json(recipe: Recipe) -> Dict[str, Any]:
    """Serialize *recipe* to a plain JSON-safe dict (tuples → lists)."""
    return {
        "channels": [
            [{"key": s.key, "params": dict(s.params)} for s in channel]
            for channel in recipe.channels
        ],
        "geometry": (
            None if recipe.geometry is None else {
                "key": recipe.geometry.key,
                "boundary_index": _as_int(recipe.geometry.boundary_index, 0),
                "config": _config_to_json(recipe.geometry.config),
            }
        ),
    }


def from_json(data: Any) -> Recipe:
    """Deserialize a dict (as produced by :func:`to_json`, incl. a real
    ``json`` round-trip) back into a :class:`Recipe`.

    Tolerant of missing / malformed pieces (returns a best-effort Recipe rather
    than raising) so it is safe to call on persisted or hand-edited state; the
    stricter policy (drop unknown ops, clamp) is :func:`normalize_recipe`.
    """
    if not isinstance(data, dict):
        return Recipe(channels=[], geometry=None)

    channels: List[List[IntensityStep]] = []
    raw_channels = data.get("channels", [])
    if isinstance(raw_channels, list):
        for raw_channel in raw_channels:
            steps: List[IntensityStep] = []
            if isinstance(raw_channel, list):
                for raw_step in raw_channel:
                    step = _intensity_step_from_json(raw_step)
                    if step is not None:
                        steps.append(step)
            channels.append(steps)

    geometry = _geometry_step_from_json(data.get("geometry"))
    return Recipe(channels=channels, geometry=geometry)


def _intensity_step_from_json(raw: Any) -> Optional[IntensityStep]:
    if not isinstance(raw, dict) or "key" not in raw:
        return None
    params = raw.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return IntensityStep(key=str(raw["key"]), params=dict(params))


def _geometry_step_from_json(raw: Any) -> Optional[GeometryStep]:
    if not isinstance(raw, dict) or "key" not in raw:
        return None
    config = raw.get("config", {})
    if not isinstance(config, dict):
        config = {}
    config = dict(config)
    if "factors" in config and isinstance(config["factors"], (list, tuple)):
        try:
            config["factors"] = tuple(config["factors"])
        except Exception:
            pass
    return GeometryStep(
        key=str(raw["key"]),
        boundary_index=_as_int(raw.get("boundary_index", 0), 0),
        config=config,
    )


def _config_to_json(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    factors = out.get("factors")
    if isinstance(factors, tuple):
        out["factors"] = list(factors)
    return out


# --------------------------------------------------------------------------- #
# normalize (corrupt-tolerant validator)
# --------------------------------------------------------------------------- #

def normalize_recipe(
    recipe: Any,
    n_channels: int,
    registry: Optional[List[OpSpec]] = None,
) -> Recipe:
    """Validate + clean *recipe* against ``n_channels`` and the registry.

    Mirrors ``normalize_convert_config``: corrupt-tolerant, **never raises**.
    Returns a clean :class:`Recipe`:

    * DROP any step whose ``key`` is unknown OR ``implemented=False`` (a step
      restored from a newer branch, or a typo'd key).
    * CLAMP the channel count to ``n_channels`` (truncate extras, pad with empty
      lists) — the recipe binds to channel INDEX.
    * CLAMP ``geometry.boundary_index`` to ``[0, max_channel_len]`` (negatives →
      0); drop geometry entirely if its key isn't a runnable geometry op.
    * COERCE each surviving step's params to its registry :class:`ParamSpec`
      types (drop unknown params, fill missing with the default).

    Accepts a :class:`Recipe`, a dict (as from :func:`to_json`), or garbage.
    """
    reg = registry if registry is not None else _registry.REGISTRY
    by_key: Dict[str, OpSpec] = {op.key: op for op in reg}
    n = max(0, _as_int(n_channels, 0))

    rec = _coerce_to_recipe(recipe)

    src_channels = rec.channels if isinstance(rec.channels, list) else []
    channels: List[List[IntensityStep]] = [
        _normalize_channel(channel, by_key) for channel in src_channels
    ]

    # Clamp channel count to n_channels.
    if len(channels) > n:
        channels = channels[:n]
    while len(channels) < n:
        channels.append([])

    geometry = _normalize_geometry(rec.geometry, by_key)
    if geometry is not None:
        max_len = max((len(ch) for ch in channels), default=0)
        clamped = max(0, min(_as_int(geometry.boundary_index, 0), max_len))
        geometry = GeometryStep(
            key=geometry.key,
            boundary_index=clamped,
            config=geometry.config,
        )

    return Recipe(channels=channels, geometry=geometry)


def _coerce_to_recipe(recipe: Any) -> Recipe:
    if isinstance(recipe, Recipe):
        return recipe
    if isinstance(recipe, dict):
        try:
            return from_json(recipe)
        except Exception:
            return Recipe(channels=[], geometry=None)
    return Recipe(channels=[], geometry=None)


def _normalize_channel(
    channel: Any, by_key: Dict[str, OpSpec]
) -> List[IntensityStep]:
    out: List[IntensityStep] = []
    if not isinstance(channel, list):
        return out
    for raw in channel:
        key = _step_key(raw)
        op = by_key.get(key) if key is not None else None
        if op is None or not op.implemented or op.kind != "intensity":
            continue  # unknown / not-yet-implemented / wrong kind → drop
        out.append(IntensityStep(key=op.key, params=_coerce_params(op, _step_params(raw))))
    return out


def _normalize_geometry(
    geometry: Any, by_key: Dict[str, OpSpec]
) -> Optional[GeometryStep]:
    if geometry is None:
        return None
    key = _step_key(geometry)
    op = by_key.get(key) if key is not None else None
    if op is None or not op.implemented or op.kind != "geometry":
        return None
    raw_config = getattr(geometry, "config", None)
    if not isinstance(raw_config, dict):
        raw_config = geometry.get("config") if isinstance(geometry, dict) else None
    if not isinstance(raw_config, dict):
        raw_config = {}

    mode = str(raw_config.get("mode", "iso"))
    if mode not in ("iso", "aniso"):
        mode = "iso"
    try:
        target_um = float(raw_config.get("target_um", 1.0))
    except Exception:
        target_um = 1.0
    factors = _coerce_factors(raw_config.get("factors"))
    method = str(raw_config.get("method", "linear"))

    boundary = _as_int(_step_boundary(geometry), 0)
    return GeometryStep(
        key=op.key,
        boundary_index=max(0, boundary),
        config={
            "mode": mode,
            "target_um": target_um,
            "factors": factors,
            "method": method,
        },
    )


# --------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------- #

def _step_key(raw: Any) -> Optional[str]:
    if isinstance(raw, (IntensityStep, GeometryStep)):
        return raw.key
    if isinstance(raw, dict):
        key = raw.get("key")
        return str(key) if key is not None else None
    return None


def _step_params(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, IntensityStep):
        return raw.params if isinstance(raw.params, dict) else {}
    if isinstance(raw, dict):
        params = raw.get("params", {})
        return params if isinstance(params, dict) else {}
    return {}


def _step_boundary(raw: Any) -> Any:
    if isinstance(raw, GeometryStep):
        return raw.boundary_index
    if isinstance(raw, dict):
        return raw.get("boundary_index", 0)
    return 0


def _coerce_params(op: OpSpec, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce *raw* to *op*'s ParamSpec types — drop unknown params, fill
    missing ones with the default."""
    if not isinstance(raw, dict):
        raw = {}
    out: Dict[str, Any] = {}
    for pspec in op.params:
        if pspec.name in raw:
            out[pspec.name] = _coerce_param(pspec, raw[pspec.name])
        else:
            out[pspec.name] = pspec.default
    return out


def _coerce_param(pspec: ParamSpec, value: Any) -> Any:
    try:
        if pspec.type == "float":
            return float(value)
        if pspec.type == "int":
            if isinstance(value, bool):
                return pspec.default
            return int(value)
        if pspec.type == "choice":
            s = str(value)
            if pspec.choices and s not in pspec.choices:
                return pspec.default
            return s
    except (TypeError, ValueError):
        return pspec.default
    return pspec.default


def _coerce_factors(value: Any) -> Tuple[float, float, float]:
    default = (1.0, 1.0, 1.0)
    if value is None:
        return default
    try:
        seq = list(value)
    except TypeError:
        return default
    if len(seq) != 3:
        return default
    try:
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
