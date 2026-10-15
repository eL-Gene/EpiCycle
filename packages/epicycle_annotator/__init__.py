"""Native annotator primitives for EpiCycle.

The public names below are re-exported **lazily** (PEP 562): the submodule is
imported on first attribute access, not at package import time.  The names, the
objects they resolve to, and ``__all__`` are exactly as before.

Why lazy: several of these modules import ``napari`` / ``vispy`` at module level
(``sync``, ``subviewers``, ``subviewer_mirror``, ``crosshairs``,
``box_overlay``), so an eager ``__init__`` made *every* module in this package
transitively napari-dependent -- including the pure ones.  ``reshape`` is
specified as napari-free and Qt-free, importable standalone, and that cannot be true while
the package's own ``__init__`` imports napari first.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

#: public name -> the submodule that defines it
_EXPORTS = {
    "CrosshairManager": "crosshairs",
    "DrawingHandler": "drawing",
    "SubviewerMirrorManager": "subviewer_mirror",
    "interpolate_label_all_orientations": "interpolation",
    "inverse_transpose_data": "orientation",
    "transpose_data": "orientation",
    "detect_seed_peaks": "seed_detection",
    "peaks_to_seed_volume": "seed_detection",
    "threshold_peaks": "seed_detection",
    "apply_split_to_volume": "split",
    "compute_edt_overlay": "split",
    "seeded_watershed_split": "split",
    "validate_split_seeds": "split",
    "SubviewerManager": "subviewers",
    "LabelSyncManager": "sync",
}

__all__ = [
    "CrosshairManager",
    "DrawingHandler",
    "SubviewerMirrorManager",
    "LabelSyncManager",
    "SubviewerManager",
    "apply_split_to_volume",
    "compute_edt_overlay",
    "detect_seed_peaks",
    "interpolate_label_all_orientations",
    "inverse_transpose_data",
    "peaks_to_seed_volume",
    "seeded_watershed_split",
    "threshold_peaks",
    "transpose_data",
    "validate_split_seeds",
]


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        )
    module = importlib.import_module("." + module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value          # cache: later access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # pragma: no cover - static analysers want the real names
    from .crosshairs import CrosshairManager
    from .drawing import DrawingHandler
    from .interpolation import interpolate_label_all_orientations
    from .orientation import inverse_transpose_data, transpose_data
    from .seed_detection import (
        detect_seed_peaks,
        peaks_to_seed_volume,
        threshold_peaks,
    )
    from .split import (
        apply_split_to_volume,
        compute_edt_overlay,
        seeded_watershed_split,
        validate_split_seeds,
    )
    from .subviewer_mirror import SubviewerMirrorManager
    from .subviewers import SubviewerManager
    from .sync import LabelSyncManager
