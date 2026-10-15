"""Cell <-> nucleus reconcile (read-only post-process; NOT a pipeline re-run).

This package reads raw channels + per-tile
probabilities at a GLOBAL bbox from an EXISTING engine run - it never re-tiles or
re-reconstructs, and it touches only the problematic cells.

The names of the optional layer (candidate scoring, consensus edits, rendering and the paired
cell-nucleus measurements) load on first access, so this package imports without those modules.
"""

from importlib import import_module as _import_module

from .raw_roi import (
    load_geometry,
    resolve_channel_layout,
    read_global_roi,
    read_prob_roi,
)
from .graph import (
    build_components,
    classify_component,
    component_census,
)
from .cleanup import (
    carve_nucleus_by_cells,
    recover_nucleus,
    fill_smooth_cell,
    cleanup_volume,
    tidy_labels,
)
from .features import (
    extract_label_features,
    assemble_cell_table,
    assemble_nucleus_table,
)

# name -> module that defines it, resolved lazily by __getattr__
_LAZY = {
    "build_candidate_cut": "adjudicate",
    "surface_band": "adjudicate",
    "membrane_support": "adjudicate",
    "interface_interior_masks": "calibrate",
    "cell_wall_reference": "calibrate",
    "measure_cut": "calibrate",
    "normalized_score": "calibrate",
    "score_merge_candidate": "calibrate",
    "score_cell_interface": "calibrate",
    "reference_by_depth": "calibrate",
    "split_cell_watershed": "realize",
    "boundary_continuity": "realize",
    "split_nucleus_at_seam": "realize",
    "nuclear_signal": "rescue",
    "cell_signal": "rescue",
    "next_free_id": "consensus",
    "apply_edits": "consensus",
    "reconcile_summary": "consensus",
    "combine_cell_nucleus_tables": "paired_features",
    "compute_paired_nucleus_features": "paired_features",
    "assemble_combined_table": "paired_features",
    "render_cell_ortho": "render",
    "render_gallery": "render",
    "render_cell_3d": "render",
    "render_gallery_3d": "render",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = _import_module(f".{mod}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{mod}":
            raise
        raise AttributeError(f"{name!r} is not part of this release") from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "load_geometry",
    "resolve_channel_layout",
    "read_global_roi",
    "read_prob_roi",
    "build_components",
    "classify_component",
    "component_census",
    "carve_nucleus_by_cells",
    "recover_nucleus",
    "fill_smooth_cell",
    "cleanup_volume",
    "tidy_labels",
    "extract_label_features",
    "assemble_cell_table",
    "assemble_nucleus_table",
]
