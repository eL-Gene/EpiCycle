"""
Configuration dataclasses for 3D reconstruction.

Ported from epicycle-training/epicycle_local/config.py — reconstruction-specific
configs only. SegmentationConfig lives in packages.epicycle_training.config.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class ReconstructionConfig:
    """Configuration for u-Segment3D 3D reconstruction.

    Attributes:
        method: Reconstruction method ('direct' or 'indirect')
        save_prob3d: Save 3D probability volume
        save_flow3d: Save 3D gradient volume
        save_binary: Save binary mask
        binary_erode: Erode instances before binary extraction
        binary_kernel_size: Erosion kernel size
        binary_fill_holes: Fill holes in binary mask
        n_cpu: Number of CPU threads for postprocessing
        skip_postprocess: Skip u-Segment3D flow-consistency postprocessing
    """
    method: Literal["direct", "indirect"] = "direct"
    save_prob3d: bool = True
    save_flow3d: bool = False
    save_binary: bool = True
    binary_erode: bool = True
    binary_kernel_size: int = 3
    binary_fill_holes: bool = True
    n_cpu: int = 6
    skip_postprocess: bool = True

    # Pre-smoothing — anisotropic Gaussian applied per-view BEFORE
    # u-Segment3D aggregation.  Suppresses slice-to-slice discontinuities
    # from independent 2D CellPose predictions (cross-hatch striations).
    # Disabled by default.
    pre_smooth_probs: bool = False
    pre_smooth_flows: bool = False
    pre_smooth_sigma_perp: float = 3.0
    pre_smooth_sigma_inplane: float = 1.0
    flow_mask_threshold: float = 0.2
    flow_soft_weighting: bool = True

    # u-Segment3D aggregation parameters — EpiCycle-tuned defaults
    # (from the whole-tissue engine's defaults)
    # Ordered by impact: highest → lowest.
    # See u-Segment3D documentation for full parameter details.

    # HIGHEST IMPACT — core flow-tracking that finds cell instances
    gradient_descent: Dict = field(default_factory=lambda: {
        "gradient_decay": 0.001,
        "n_iter": 25,
        "momenta": 0.98,
        "do_mp": True,
        "renorm_gradient": True,
        "tile_shape": [128, 256, 256],
        "tile_aspect": [1, 2, 2],
        "tile_overlap_ratio": 0.25,
    })

    # HIGH IMPACT — 3D binary mask from 2D probability maps
    combine_cell_probs: Dict = field(default_factory=lambda: {
        "ksize": 25,
        "alpha": 0.5,
        "smooth_sigma": 0.8,
        "threshold_level": 1,
        "threshold_n_levels": 3,
        "prob_thresh": None,
        "min_prob_thresh": 0.0,
        "apply_one_d_p_thresh": True,
    })

    # HIGH IMPACT — morphological cleanup of 3D binary mask
    postprocess_binary: Dict = field(default_factory=lambda: {
        "binary_closing": 1,
        "binary_dilation": 1,
        "binary_fill_holes": True,
        "remove_small_objects": 10000,
        "extra_erode": 0,
    })

    # MEDIUM IMPACT — merges 2D gradient fields into 3D
    combine_cell_gradients: Dict = field(default_factory=lambda: {
        "ksize": 25,
        "alpha": 0.5,
        "smooth_sigma": 1.5,
        "post_sigma": 0.5,
    })

    # MEDIUM IMPACT — density-based clustering of flow endpoints
    connected_component: Dict = field(default_factory=lambda: {
        "min_area": 5,
        "thresh_factor": 0.0,
        "smooth_sigma": 1.0,
    })

    # LOW IMPACT — optional post-hoc gradient consistency filter (off by default)
    flow_consistency: Dict = field(default_factory=lambda: {
        "do_flow_remove": False,
        "flow_threshold": 0.8,
        "dtform_method": "cellpose_improve",
        "edt_fixed_point_percentile": 0.01,
        "power_dist": None,
        "smooth_skel_sigma": 3.0,
    })

    # LOW IMPACT — size filtering (only when skip_postprocess=False)
    size_filters: Dict = field(default_factory=lambda: {
        "min_size": 200,
        "max_size": None,
        "max_size_factor": 10,
        "do_stats_filter": True,
    })

    # CONDITIONAL — only used when method="indirect"
    indirect_method: Dict = field(default_factory=lambda: {
        "dtform_method": "cellpose_improve",
        "smooth_binary": 1.0,
        "smooth_skel_sigma": 3.0,
        "edt_fixed_point_percentile": 0.01,
        "power_dist": None,
    })


def get_default_reconstruction_config() -> ReconstructionConfig:
    """Get default reconstruction configuration."""
    return ReconstructionConfig()
