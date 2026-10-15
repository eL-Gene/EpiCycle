"""
Default parameter dictionaries for EpiCycle ScaledSegmentation.

These are the empirically-tuned defaults for segmentation, reconstruction,
post-processing, and save policies.
"""


def get_default_segment_params() -> dict:
    """Default CellPose segmentation parameters (empirically tested)."""
    return {
        'normalize': True,
        'flow_threshold': 0.2,
        'cellprob_threshold': 0.0,
        'channels': [2, 1],        # [signal_channel, nucleus_channel]
        'resample': True,
        'rescale': True,
        'do_3D': False,
        'batch_size': 1024,
        # 2D masks OFF by default: the 3D reconstruction uses flows + probs, not
        # Cellpose's 2D masks, so skipping mask assembly saves GPU time. Set True
        # (or save_raw_masks) only if you actually want the per-view 2D masks.
        'compute_masks': False,
    }


def get_default_reconstruct_params(n_cpu: int = 8) -> dict:
    """
    Default USegment3D reconstruction parameters (direct method).

    Parameters
    ----------
    n_cpu : number of CPU threads per worker (default 8 = 64 cores / 8 GPUs).

    Returns
    -------
    USegment3D aggregation parameter dict with tuned values.

    Note
    ----
    Requires ``segment3D`` to be installed. Import is deferred so that
    this module can be loaded without the dependency for testing.
    """
    import segment3D.parameters as uSegment3D_params

    params = uSegment3D_params.get_2D_to_3D_aggregation_params()

    # Indirect method
    params['indirect_method']['dtform_method'] = 'edt'
    params['indirect_method']['smooth_skel_sigma'] = 1
    params['indirect_method']['n_cpu'] = n_cpu

    # Combine cell probabilities (direct mode)
    params['combine_cell_probs']['cellpose_prob_mask'] = True
    params['combine_cell_probs']['ksize'] = 25
    params['combine_cell_probs']['alpha'] = 0.5
    params['combine_cell_probs']['smooth_sigma'] = 0.8

    # Combine cell gradients
    params['combine_cell_gradients']['ksize'] = 25
    params['combine_cell_gradients']['smooth_sigma'] = 1.5
    params['combine_cell_gradients']['post_sigma'] = 0.5

    # Post-process binary (direct mode)
    params['postprocess_binary']['binary_fill_holes'] = True
    params['postprocess_binary']['remove_small_objects'] = True
    params['postprocess_binary']['min_size'] = 10000

    # Gradient descent field integration
    params['gradient_descent']['n_iter'] = 25
    params['gradient_descent']['momenta'] = 0.98
    params['gradient_descent']['gradient_decay'] = 0.001
    params['gradient_descent']['debug_viz'] = False
    params['gradient_descent']['do_mp'] = True
    # renorm_gradient=True matches the plugin's reconstruction settings, so Step 3/4
    # and the engine give the same result.
    params['gradient_descent']['renorm_gradient'] = True

    return params


def get_default_post_process_params() -> dict:
    """Default post-processing parameters (erosion settings)."""
    return {
        'erode_instances': False,
        'kernel_size': 3,
    }


def get_default_postprocess_segmentation_params(n_cpu: int = 8) -> dict:
    """
    Default USegment3D post-process segmentation parameters.

    Returns
    -------
    USegment3D postprocess parameter dict with tuned values.
    """
    import segment3D.parameters as uSegment3D_params

    post = uSegment3D_params.get_postprocess_segmentation_params()
    post['flow_consistency']['do_flow_remove'] = False
    post['flow_consistency']['dtform_method'] = 'cellpose_skel' # cellpose_improve # or use <- default  
    post['flow_consistency']['flow_threshold'] = 0.8
    post['flow_consistency']['n_cpu'] = n_cpu
    return post


def get_default_indirect_reconstruct_params(n_cpu: int = 64) -> dict:
    """
    Default USegment3D parameters for global indirect reconstruction (Stage 4).

    The indirect method takes label images from 3 orthogonal views and
    derives distance-transform gradients internally. Parameters are tuned
    for the full (downsampled) stitched volume.

    Parameters
    ----------
    n_cpu : CPU threads for multiprocess watershed (default 64 = full node).
    """
    import segment3D.parameters as uSegment3D_params

    params = uSegment3D_params.get_2D_to_3D_aggregation_params()

    # Indirect method — distance transform
    params['indirect_method']['dtform_method'] = 'edt'
    params['indirect_method']['smooth_skel_sigma'] = 1
    params['indirect_method']['n_cpu'] = n_cpu

    # Post-process binary
    params['postprocess_binary']['binary_fill_holes'] = True
    params['postprocess_binary']['remove_small_objects'] = True
    params['postprocess_binary']['min_size'] = 10000

    # Gradient descent field integration (indirect needs more iterations)
    params['gradient_descent']['n_iter'] = 150
    params['gradient_descent']['momenta'] = 0.98
    params['gradient_descent']['gradient_decay'] = 0.1
    params['gradient_descent']['debug_viz'] = False
    params['gradient_descent']['do_mp'] = True
    params['gradient_descent']['renorm_gradient'] = True

    return params


def get_default_save_policy() -> dict:
    """Default save policy for Stage 1 segmentation outputs."""
    return {
        "save_raw_flows": True,
        "save_raw_probs": True,
        "save_raw_masks": False,     # masks are large and usually not needed
        "save_styles": True,
        "flows_compressed": False,   # .npy is 7x faster to load than .npz
        "probs_compressed": False,
    }
