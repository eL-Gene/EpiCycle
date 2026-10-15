"""The reconstruction settings' presets, shared by Steps 3 and 4.

Qt-free: widget -> value tables for "Start from" -- EpiCycle's defaults,
u-Segment3D's own ``parameters.py`` defaults (read, never edited) and the
keys a Step 3 reconstruction manifest records.  They were Step 4's
(``segmentation/_step4_models.py``), moved here
verbatim; ``_step4_models`` re-exports every name.  The widget names are
Step 4's; Step 3 maps its own onto them (``STEP3_SETTING_KEYS``, R2).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


#: widget -> value: ``ReconstructionConfig()`` plus today's Inference
#: widget defaults (parity-tested against ``ReconstructionConfig()``).
#: ``ccp_prob_thresh_sb`` -1 is "Auto" (None); ``sf_max_size_sb`` -1 is
#: "no limit" (None).
STEP4_PRESET_EPICYCLE = {
    "cell_width_um_sb": 0.0,
    "normalize_cb": True,
    "rescale_cb": True,
    "batch_size_sb": 128,
    "n_cpu_sb": 6,
    "skip_postprocess_cb": True,
    "ps_smooth_probs_cb": False,
    "ps_smooth_flows_cb": False,
    "ps_sigma_perp_sb": 3.0,
    "ps_sigma_inplane_sb": 1.0,
    "ps_mask_thresh_sb": 0.2,
    "ps_soft_weight_cb": True,
    "gd_decay_sb": 0.001,
    "gd_iters_sb": 25,
    "gd_momenta_sb": 0.98,
    "do_mp_cb": True,
    "gd_renorm_cb": True,
    "gd_tile_z_sb": 128,
    "gd_tile_y_sb": 256,
    "gd_tile_x_sb": 256,
    "gd_tile_overlap_sb": 0.25,
    "ccp_ksize_sb": 25,
    "ccp_alpha_sb": 0.5,
    "ccp_smooth_sigma_sb": 0.8,
    "ccp_threshold_level_sb": 1,
    "ccp_threshold_n_levels_sb": 3,
    "ccp_prob_thresh_sb": -1.0,
    "ccp_min_prob_thresh_sb": 0.0,
    "ccp_apply_one_d_p_cb": True,
    "post_closing_sb": 1,
    "post_dilation_sb": 1,
    "post_fill_cb": True,
    "post_small_obj_sb": 10000,
    "post_extra_erode_sb": 0,
    "ccg_ksize_sb": 25,
    "ccg_alpha_sb": 0.5,
    "ccg_smooth_sigma_sb": 1.5,
    "ccg_post_sigma_sb": 0.5,
    "cc_min_area_sb": 5,
    "cc_thresh_factor_sb": 0.0,
    "cc_smooth_sigma_sb": 1.0,
    "sf_min_size_sb": 200,
    "sf_max_size_sb": -1,
    "sf_max_size_factor_sb": 10.0,
    "sf_do_stats_cb": True,
}

#: widget -> (u-Segment3D group, key) for the ``parameters.py`` defaults
#: and the Step 3 manifest's ``recon_config`` (same group names).
USEG3D_WIDGET_KEYS = {
    "gd_decay_sb": ("gradient_descent", "gradient_decay"),
    "gd_iters_sb": ("gradient_descent", "n_iter"),
    "gd_momenta_sb": ("gradient_descent", "momenta"),
    "do_mp_cb": ("gradient_descent", "do_mp"),
    "gd_renorm_cb": ("gradient_descent", "renorm_gradient"),
    "gd_tile_z_sb": ("gradient_descent", ("tile_shape", 0)),
    "gd_tile_y_sb": ("gradient_descent", ("tile_shape", 1)),
    "gd_tile_x_sb": ("gradient_descent", ("tile_shape", 2)),
    "gd_tile_overlap_sb": ("gradient_descent", "tile_overlap_ratio"),
    "ccp_ksize_sb": ("combine_cell_probs", "ksize"),
    "ccp_alpha_sb": ("combine_cell_probs", "alpha"),
    "ccp_smooth_sigma_sb": ("combine_cell_probs", "smooth_sigma"),
    "ccp_threshold_level_sb": ("combine_cell_probs", "threshold_level"),
    "ccp_threshold_n_levels_sb": ("combine_cell_probs", "threshold_n_levels"),
    "ccp_prob_thresh_sb": ("combine_cell_probs", "prob_thresh"),
    "ccp_min_prob_thresh_sb": ("combine_cell_probs", "min_prob_thresh"),
    "ccp_apply_one_d_p_cb": ("combine_cell_probs", "apply_one_d_p_thresh"),
    "post_closing_sb": ("postprocess_binary", "binary_closing"),
    "post_dilation_sb": ("postprocess_binary", "binary_dilation"),
    "post_fill_cb": ("postprocess_binary", "binary_fill_holes"),
    "post_small_obj_sb": ("postprocess_binary", "remove_small_objects"),
    "post_extra_erode_sb": ("postprocess_binary", "extra_erode"),
    "ccg_ksize_sb": ("combine_cell_gradients", "ksize"),
    "ccg_alpha_sb": ("combine_cell_gradients", "alpha"),
    "ccg_smooth_sigma_sb": ("combine_cell_gradients", "smooth_sigma"),
    "ccg_post_sigma_sb": ("combine_cell_gradients", "post_sigma"),
    "cc_min_area_sb": ("connected_component", "min_area"),
    "cc_thresh_factor_sb": ("connected_component", "thresh_factor"),
    "cc_smooth_sigma_sb": ("connected_component", "smooth_sigma"),
    "sf_min_size_sb": ("size_filters", "min_size"),
    "sf_max_size_sb": ("size_filters", "max_size"),
    "sf_max_size_factor_sb": ("size_filters", "max_size_factor"),
    "sf_do_stats_cb": ("size_filters", "do_stats_filter"),
}

#: Widgets whose ``None`` is written as -1 ("Auto" / "no limit").
_NONE_AS_MINUS_ONE = ("ccp_prob_thresh_sb", "sf_max_size_sb")


def _useg3d_params() -> Optional[Dict[str, Any]]:
    try:
        from segment3D import parameters as P
    except Exception:      # noqa: BLE001
        return None
    try:
        params = dict(P.get_2D_to_3D_aggregation_params())
        params.update(P.get_postprocess_segmentation_params())
        return params
    except Exception:      # noqa: BLE001
        return None


def _key_present(group: Dict[str, Any], key) -> bool:
    return (key[0] if isinstance(key, tuple) else key) in group


def _lookup(group: Dict[str, Any], key):
    if isinstance(key, tuple):
        seq = group.get(key[0])
        return None if seq is None else seq[key[1]]
    return group.get(key)


def _preset_from_groups(groups: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for widget, (grp, key) in USEG3D_WIDGET_KEYS.items():
        group = groups.get(grp)
        if not isinstance(group, dict) or not _key_present(group, key):
            continue
        try:
            val = _lookup(group, key)
        except (IndexError, KeyError, TypeError):
            continue
        if val is None:
            if widget in _NONE_AS_MINUS_ONE:
                out[widget] = -1
            continue
        out[widget] = val
    return out


def useg3d_preset() -> Dict[str, Any]:
    """widget -> value from u-Segment3D's own ``parameters.py`` defaults
    (read, never edited).  Empty when the package cannot be imported."""
    params = _useg3d_params()
    return _preset_from_groups(params) if params else {}


#: The Step 3 manifest's top-level ``recon_config`` keys -> widgets.
_STEP3_TOP_KEYS = {
    "skip_postprocess": "skip_postprocess_cb",
    "pre_smooth_probs": "ps_smooth_probs_cb",
    "pre_smooth_flows": "ps_smooth_flows_cb",
    "pre_smooth_sigma_perp": "ps_sigma_perp_sb",
    "pre_smooth_sigma_inplane": "ps_sigma_inplane_sb",
    "flow_mask_threshold": "ps_mask_thresh_sb",
    "flow_soft_weighting": "ps_soft_weight_cb",
}


def step3_recon_preset(recon_config: Dict[str, Any]) -> Dict[str, Any]:
    """widget -> value for the keys a Step 3 manifest records (logic:533-546);
    every other widget is left as it is."""
    if not isinstance(recon_config, dict):
        return {}
    out = _preset_from_groups(recon_config)
    for key, widget in _STEP3_TOP_KEYS.items():
        if recon_config.get(key) is not None:
            out[widget] = recon_config[key]
    return out
