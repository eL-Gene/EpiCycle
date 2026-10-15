"""Tests for epicycle.config.defaults module (non-segment3D-dependent parts)."""

from packages.epicycle_scaledseg.config.defaults import (
    get_default_segment_params,
    get_default_post_process_params,
    get_default_save_policy,
)


def test_segment_params_keys():
    p = get_default_segment_params()
    assert 'flow_threshold' in p
    assert 'channels' in p
    assert p['do_3D'] is False
    assert isinstance(p['batch_size'], int)


def test_post_process_params():
    p = get_default_post_process_params()
    assert p['erode_instances'] is False
    assert p['kernel_size'] == 3


def test_save_policy_defaults():
    p = get_default_save_policy()
    assert p['flows_compressed'] is False  # .npy preferred
    assert p['save_raw_masks'] is False    # masks not saved by default
    assert p['save_raw_flows'] is True
