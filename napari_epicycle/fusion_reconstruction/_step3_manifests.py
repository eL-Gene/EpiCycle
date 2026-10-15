"""Step 3's prediction manifest and the per-patch cell width
(bricks R2b part 2
and R2c).

Qt-free and napari-free; plain functions the controller calls from the
prediction worker and the main thread, each drivable by a test without a
viewer.

* **The cell width in µm** (row 10): ``patch_diameter`` turns the width the
  user typed (µm; 0 = Auto) and one patch's voxel size into the pixel
  number Cellpose gets for THAT patch -- Step 4's ``cell_width_px`` /
  ``_stack_px`` rule, copied here (Step 3 may not import a Step 4 module):
  ``max(1, round(width / mean Y-X voxel size))``; with no voxel size the
  number is taken as pixels, as typed.  Auto is ``None``, as before R2b.
* **The resize rule** (Step 4's F25 vocabulary, copied): which of
  Cellpose's three rules applies, and the factor when it is known without
  reading the model.
* **The prediction manifest** (row 26):
  ``prediction_cache/{stem}_prediction_manifest.json``, written atomically
  AFTER the arrays (mkstemp + fsync + ``os.replace``; "meta written last").
  Nothing needs it to restore the cache; it is deleted with the arrays
  (every deleter globs ``{stem}_*``).  ``spacing`` stays ``None`` until R7,
  ``model_provenance`` until R6.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: the prediction manifest's own version
PREDICTION_MANIFEST_VERSION = 1
PREDICTION_MANIFEST_SUFFIX = "_prediction_manifest.json"
VIEWS = ("xy", "xz", "yz")

#: How Cellpose 3.1.1.2 decides its resize (Step 4's F25 / N2 words): a
#: cell width (diameter > 0) always wins and the Rescale tick is ignored.
RESCALE_FROM_WIDTH = "from cell width"
RESCALE_NONE = "none"
RESCALE_TRAINED = "model's trained size"

#: where a patch's pixel number came from
DIAMETER_AUTO = "auto"
DIAMETER_FROM_SPACING = "voxel size"
DIAMETER_AS_PIXELS = "pixels"


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def json_safe(obj):
    """*obj* with numpy scalars / arrays, tuples and paths made plain, so
    ``json.dumps`` writes it as it reads back."""
    try:
        import numpy as np
    except Exception:      # noqa: BLE001
        np = None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return json_safe(obj.tolist())
        if isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def config_record(config) -> Dict[str, Any]:
    """Every field of a config dataclass, JSON-safe (``{}`` for None)."""
    if config is None:
        return {}
    if dataclasses.is_dataclass(config):
        return json_safe(dataclasses.asdict(config))
    if isinstance(config, dict):
        return json_safe(dict(config))
    return {}


# ---------------------------------------------------------------------------
# The cell width (row 10)
# ---------------------------------------------------------------------------


def xy_um(spacing: Optional[Dict[str, Any]]) -> Optional[float]:
    """The mean Y/X voxel size of a spacing, or None (Step 4's ``xy_um``)."""
    zyx = spacing.get("zyx") if isinstance(spacing, dict) else None
    try:
        v = (float(zyx[1]) + float(zyx[2])) / 2.0
    except (TypeError, ValueError, IndexError):
        return None
    return v if v > 0 else None


def cell_width_px(width_um, spacing: Optional[Dict[str, Any]]):
    """The cell width in pixels for a patch's Y/X spacing, or None when the
    width is Auto (0) or the patch has no spacing (Step 4's
    ``cell_width_px``)."""
    try:
        width = float(width_um or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0:
        return None
    um = xy_um(spacing)
    if um is None:
        return None
    return max(1, int(round(width / um)))


def patch_diameter(width_um, spacing) -> Tuple[Optional[float], str]:
    """``(diameter for Cellpose, where it came from)`` for one patch.

    Auto (0) -> ``(None, "auto")``; a voxel size -> its pixel number;
    none -> the typed number taken as pixels (``"pixels"``), as the row's
    warning says.  Never raises."""
    try:
        width = float(width_um or 0)
    except (TypeError, ValueError):
        width = 0.0
    if width <= 0:
        return None, DIAMETER_AUTO
    px = cell_width_px(width, spacing)
    if px is not None:
        return float(px), DIAMETER_FROM_SPACING
    return width, DIAMETER_AS_PIXELS


def with_diameter(seg_config, diameter):
    """A copy of *seg_config* with *diameter*; the input is never changed
    (one config is shared by every patch of a run).  *seg_config* that is
    not a dataclass comes back as it is."""
    if seg_config is None or not dataclasses.is_dataclass(seg_config):
        return seg_config
    return dataclasses.replace(seg_config, diameter=diameter)


# ---------------------------------------------------------------------------
# The resize rule (Step 4's F25 vocabulary)
# ---------------------------------------------------------------------------


def effective_rescale(rescale) -> Optional[float]:
    """The resize factor Cellpose really gets: ``None`` = the model's own
    trained cell size; ``1.0`` = no resize (the tick ON)."""
    if rescale is None or rescale is False:
        return None
    if rescale is True:
        return 1.0
    try:
        return float(rescale)
    except (TypeError, ValueError):
        return None


def rescale_rule(rescale, diameter) -> str:
    """Which of the three rules Cellpose applies."""
    try:
        if diameter is not None and float(diameter) > 0:
            return RESCALE_FROM_WIDTH
    except (TypeError, ValueError):
        pass
    return RESCALE_TRAINED if effective_rescale(rescale) is None \
        else RESCALE_NONE


def rescale_factor(rule: str) -> Optional[float]:
    """The factor when it is known without reading the model: 1.0 under
    "none"; otherwise it depends on the model's trained size, which is not
    read here (``None``)."""
    return 1.0 if rule == RESCALE_NONE else None


# ---------------------------------------------------------------------------
# The prediction manifest (row 26)
# ---------------------------------------------------------------------------


def prediction_manifest_path(cache_dir, patch_name: str) -> Path:
    return Path(cache_dir) / f"{Path(patch_name).stem}{PREDICTION_MANIFEST_SUFFIX}"


def build_prediction_manifest(*, patch_name: str, axes: str, seg_config,
                              cell_width_um=None,
                              diameter_from: Optional[str] = None,
                              predicted_at: Optional[str] = None,
                              ) -> Dict[str, Any]:
    """The record of one patch's Predict: every ``SegmentationConfig``
    field as Cellpose got it (the per-patch ``diameter`` included), the
    patch's pixel number, how it was found, and the resize rule.  JSON-safe.
    ``spacing`` (R7) and ``model_provenance`` (R6) are slots, ``None``."""
    seg = config_record(seg_config)
    if "channels" in seg:
        seg["channels"] = list(seg["channels"])
    diameter = seg.get("diameter")
    rule = rescale_rule(seg.get("rescale"), diameter)
    return json_safe({
        "manifest_version": PREDICTION_MANIFEST_VERSION,
        "patch_name": patch_name,
        "predicted_at": predicted_at
        or datetime.now(timezone.utc).isoformat(),
        "axes": axes,
        "seg_config": seg,
        "views": list(VIEWS),
        "diameter_px": diameter,
        "cell_width_um": cell_width_um,
        "diameter_from": diameter_from,
        "rescale_rule": rule,
        "rescale_factor": rescale_factor(rule),
        "spacing": None,             # R7
        "model_provenance": None,    # R6
    })


def write_json_atomic(path, data) -> None:
    """mkstemp in the target folder + fsync + ``os.replace``: a reader sees
    the old file or the whole new one, never half of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(json_safe(data), fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_prediction_manifest(path) -> Optional[Dict[str, Any]]:
    """The manifest, or None when it is missing, unreadable or not one.
    Never raises."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(
            data.get("seg_config"), dict):
        return None
    return data


def seg_config_from_record(record: Optional[Dict[str, Any]], cls):
    """A *cls* (``SegmentationConfig``) from a manifest's ``seg_config``,
    keeping only the fields *cls* has, or None.  Never raises."""
    if not isinstance(record, dict):
        return None
    try:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in record.items() if k in names})
    except Exception:      # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# "This run's last reconstruction" on a v2 manifest (row 9)
# ---------------------------------------------------------------------------


def is_full_record(manifest_version) -> bool:
    """A reconstruction manifest that records every setting (R2c, v2+)."""
    return (isinstance(manifest_version, int)
            and not isinstance(manifest_version, bool)
            and manifest_version >= 2)


def recorded_inference(seg_config: Optional[Dict[str, Any]],
                       recon_config: Optional[Dict[str, Any]],
                       cell_width_um=None) -> Dict[str, Any]:
    """Shared widget key -> value for the inference fields and the CPU
    cores a v2 reconstruction manifest records (the alias map
    ``STEP3_SETTING_KEYS`` turns them into Step 3's widgets).  Cellpose's
    two thresholds and "Compute masks" are hidden knobs and never written;
    the method has one item.  The cell width comes from ``cell_width_um``
    (the µm the user typed), never from the pixel ``diameter``."""
    seg = seg_config if isinstance(seg_config, dict) else {}
    rec = recon_config if isinstance(recon_config, dict) else {}
    out: Dict[str, Any] = {}
    if "batch_size" in seg and seg["batch_size"] is not None:
        out["batch_size_sb"] = seg["batch_size"]
    if "normalize" in seg and seg["normalize"] is not None:
        out["normalize_cb"] = bool(seg["normalize"])
    if "rescale" in seg:
        out["rescale_cb"] = effective_rescale(seg["rescale"]) is not None
    if cell_width_um is not None:
        try:
            out["cell_width_um_sb"] = float(cell_width_um)
        except (TypeError, ValueError):
            pass
    elif "diameter" in seg and seg["diameter"] is None:
        out["cell_width_um_sb"] = 0.0          # Auto, in any unit
    if rec.get("n_cpu") is not None:
        out["n_cpu_sb"] = rec["n_cpu"]
    return out
