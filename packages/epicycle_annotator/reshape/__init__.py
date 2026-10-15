"""The Reshape engine -- one seeded partition of a bounded region.

``RESHAPE(target, seeds, walls, solver, refine, arbiter)`` -- the unified model
behind Split, Add and the membrane sandbox.

**Pure: no Qt, no napari.**  numpy / scipy / skimage only, with u-Segment3D as a
lazy optional import.  Nothing here imports from ``notebooks/``; this package IS
the former membrane-sandbox library, ported.

R0 ships the **split shape** (``target="label"``).  The add shape lands in R2.

>>> from packages.epicycle_annotator.reshape import ReshapeSpec, reshape_region
>>> spec = ReshapeSpec(target="label", seeds="labels", walls="membrane")
>>> out = reshape_region(volume, {"membrane": mem}, spec, target_id=412,
...                      seed_layer=nuclei, spacing=(0.5, 0.1, 0.1))  # doctest: +SKIP
"""

from __future__ import annotations

from .arbiter import adjacent_pairs, arbitrate, interface_mask, physical_ball
from .engine import (
    STAGES,
    ReshapeCancelled,
    SeedReport,
    apply_reshape_to_volume,
    find_seeds,
    remap_flags,
    reshape_region,
)
from .notes import MessageSink
from .optional_deps import (
    have_usegment3d,
    import_diffuse_labels3D,
    import_guidedfilter,
)
from .postprocess import (
    guided_filter_refine,
    merge_small_pieces,
    merge_to_count,
    refine_result_guided,
)
from .settle import (
    SettleResult,
    apply_settle_to_volume,
    roi_margin_voxels,
    settle_label,
)
from .quality import (
    QualityNote,
    coverage_percent,
    quality_warnings,
    shortfall_percent,
)
from .preproc import (
    apply_preproc,
    axis_sigma,
    background_subtract,
    clahe_stack,
    denoise_median,
    norm,
    preprocess_membrane,
    safe_otsu,
    strip_clahe,
    zdepth_correct,
)
from .seeds import (
    physical_to_voxel_scalar,
    seeds_blobs_dog,
    seeds_blobs_log,
    seeds_distance_maxima,
    seeds_from_labels,
    seeds_from_nuclei,
    seeds_hmaxima,
    seeds_membrane_minima,
    seeds_threshold_cc,
)
from .solvers import solve
from .spec import (
    DEFAULT_PARAMS,
    DEFAULT_SIZE_BUDGET,
    GEOMETRY_ONLY_WALLS,
    IMAGE_DERIVED_SEEDS,
    IMAGE_DERIVED_WALLS,
    REFINES,
    SEEDS,
    SOLVERS,
    TARGETS,
    WALLS,
    ArbiterPair,
    ArbiterReport,
    ReshapeResult,
    ReshapeSpec,
)
from .walls import build_elevation, raise_neighbour_walls

__all__ = [
    # contract
    "ReshapeSpec",
    "ReshapeResult",
    "ArbiterPair",
    "ArbiterReport",
    "ReshapeCancelled",
    "SeedReport",
    "MessageSink",
    # entry points
    "reshape_region",
    "find_seeds",
    "apply_reshape_to_volume",
    "remap_flags",
    "STAGES",
    # stages
    "build_elevation",
    "raise_neighbour_walls",
    "solve",
    "arbitrate",
    "adjacent_pairs",
    "interface_mask",
    "physical_ball",
    "guided_filter_refine",
    "refine_result_guided",
    "merge_small_pieces",
    "merge_to_count",
    # settle (the Refine tab's seventh method, REFINE)
    "settle_label",
    "apply_settle_to_volume",
    "roi_margin_voxels",
    "SettleResult",
    # the quality gate
    "QualityNote",
    "quality_warnings",
    "coverage_percent",
    "shortfall_percent",
    # seeds
    "seeds_from_labels",
    "seeds_from_nuclei",
    "seeds_membrane_minima",
    "seeds_distance_maxima",
    "seeds_blobs_log",
    "seeds_blobs_dog",
    "seeds_hmaxima",
    "seeds_threshold_cc",
    "physical_to_voxel_scalar",
    # preprocessing
    "norm",
    "axis_sigma",
    "safe_otsu",
    "clahe_stack",
    "zdepth_correct",
    "background_subtract",
    "denoise_median",
    "apply_preproc",
    "strip_clahe",
    "preprocess_membrane",
    # optional deps
    "have_usegment3d",
    "import_guidedfilter",
    "import_diffuse_labels3D",
    # enums / defaults
    "TARGETS",
    "SEEDS",
    "WALLS",
    "SOLVERS",
    "REFINES",
    "IMAGE_DERIVED_WALLS",
    "IMAGE_DERIVED_SEEDS",
    "GEOMETRY_ONLY_WALLS",
    "DEFAULT_PARAMS",
    "DEFAULT_SIZE_BUDGET",
]
