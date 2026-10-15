"""Tests for ReconstructionConfig dataclass."""

import dataclasses

from packages.epicycle_reconstruction.config import (
    ReconstructionConfig,
    get_default_reconstruction_config,
)


def test_default_config_fields():
    cfg = ReconstructionConfig()
    assert cfg.method == "direct"
    assert cfg.save_prob3d is True
    assert cfg.save_flow3d is False
    assert cfg.save_binary is True
    assert cfg.skip_postprocess is True
    # n_cpu is a fixed dataclass default (not derived from the machine); assert
    # it is a positive int equal to that declared default, not a literal.
    declared = {f.name: f.default for f in dataclasses.fields(ReconstructionConfig)}
    assert isinstance(cfg.n_cpu, int) and cfg.n_cpu > 0
    assert cfg.n_cpu == declared["n_cpu"]


def test_default_gradient_params():
    """EpiCycle-tuned defaults from epicycle-scaledsegmentation."""
    cfg = ReconstructionConfig()
    assert cfg.combine_cell_gradients["ksize"] == 25
    assert cfg.combine_cell_gradients["alpha"] == 0.5
    assert cfg.combine_cell_gradients["smooth_sigma"] == 1.5
    assert cfg.combine_cell_gradients["post_sigma"] == 0.5


def test_default_combine_cell_probs():
    cfg = ReconstructionConfig()
    assert cfg.combine_cell_probs["ksize"] == 25
    assert cfg.combine_cell_probs["apply_one_d_p_thresh"] is True


def test_default_postprocess_params():
    cfg = ReconstructionConfig()
    assert cfg.postprocess_binary["binary_closing"] == 1
    assert cfg.postprocess_binary["binary_dilation"] == 1
    assert cfg.postprocess_binary["binary_fill_holes"] is True
    assert cfg.postprocess_binary["remove_small_objects"] == 10000


def test_default_gradient_descent_params():
    """EpiCycle-tuned defaults: low decay, few iters, high momenta, MP on."""
    cfg = ReconstructionConfig()
    assert cfg.gradient_descent["gradient_decay"] == 0.001
    assert cfg.gradient_descent["n_iter"] == 25
    assert cfg.gradient_descent["momenta"] == 0.98
    assert cfg.gradient_descent["do_mp"] is True
    assert cfg.gradient_descent["renorm_gradient"] is True
    assert cfg.gradient_descent["tile_shape"] == [128, 256, 256]
    assert cfg.gradient_descent["tile_overlap_ratio"] == 0.25


def test_default_flow_consistency():
    cfg = ReconstructionConfig()
    assert cfg.flow_consistency["do_flow_remove"] is False
    assert cfg.flow_consistency["dtform_method"] == "cellpose_improve"
    assert cfg.flow_consistency["edt_fixed_point_percentile"] == 0.01
    assert cfg.flow_consistency["power_dist"] is None
    assert cfg.flow_consistency["smooth_skel_sigma"] == 3.0


def test_default_size_filters():
    cfg = ReconstructionConfig()
    assert cfg.size_filters["min_size"] == 200
    assert cfg.size_filters["max_size"] is None
    assert cfg.size_filters["max_size_factor"] == 10
    assert cfg.size_filters["do_stats_filter"] is True


def test_custom_config_override():
    cfg = ReconstructionConfig(
        method="indirect",
        skip_postprocess=True,
        combine_cell_gradients={"ksize": 5, "alpha": 0.3, "smooth_sigma": 2.0},
    )
    assert cfg.method == "indirect"
    assert cfg.skip_postprocess is True
    assert cfg.combine_cell_gradients["ksize"] == 5


def test_factory_function():
    cfg = get_default_reconstruction_config()
    assert isinstance(cfg, ReconstructionConfig)
    assert cfg.method == "direct"


def test_indirect_method_defaults():
    cfg = ReconstructionConfig()
    assert cfg.indirect_method["dtform_method"] == "cellpose_improve"
    assert cfg.indirect_method["smooth_binary"] == 1.0
    assert cfg.indirect_method["edt_fixed_point_percentile"] == 0.01
    assert cfg.indirect_method["power_dist"] is None


def test_connected_component_defaults():
    cfg = ReconstructionConfig()
    assert cfg.connected_component["min_area"] == 5
    assert cfg.connected_component["thresh_factor"] == 0.0
