"""The reconstruction settings' words, shared by Steps 3 and 4.

Qt-free and imports nothing: the plain names, the seven u-Segment3D groups,
the hidden knobs and the help lines of the settings both steps build from
the same widgets.  They were Step 4's (``segmentation/_step4_text.py``), moved here verbatim; ``_step4_text``
re-exports every one under the same name, and its ``STEP4_HELP_LINES``
holds ``RECON_HELP_LINES`` among its own lines (same keys, same values,
same order).  A reword here changes BOTH steps' panels.
"""
from __future__ import annotations


# ── The settings column ─────────────────────────────────────────────
SEG_SETTINGS_TITLE = "Settings"
SEG_START_FROM_LABEL = "Start from:"
SEG_PRESET_EPICYCLE = "EpiCycle defaults"
SEG_PRESET_USEG3D = "u-Segment3D defaults (without its flow filter)"
SEG_PRESET_CUSTOM = "Custom"
SEG_INFERENCE_TITLE = "Inference settings"
SEG_MEMBRANE_LABEL = "Membrane channel:"
SEG_NUCLEI_LABEL = "Nuclei channel:"
SEG_CELL_WIDTH_LABEL = "Typical cell width:"
SEG_CELL_WIDTH_AUTO = "Auto"
SEG_CELL_WIDTH_SUFFIX = " µm"
SEG_CELL_WIDTH_PX = "≈ {px} pixels at {um} µm per pixel"
SEG_NORMALIZE_LABEL = "Even out brightness"
SEG_RESCALE_LABEL = "Rescale"
SEG_ADVANCED_TITLE = "Advanced"
SEG_METHOD_DIRECT = "Direct"
SEG_JOIN_TITLE = "Joining slices into cells"
SEG_JOIN_DESC = (
    "u-Segment3D, the engine that builds the 3D cells. Each group keeps its "
    "own help lines.")
SEG_SIZE_DESC = (
    "Cells outside this size range are dropped by the final clean-up after "
    "the 3D build, which runs only while “Skip the size limits” is unticked.")
SEG_SIZE_OFF_NOTE = "Size limits are off while this is ticked."
SEG_MAX_SIZE_NONE = "no limit"
SEG_PROB_THRESH_AUTO = "Auto"
#: F7: the dim note under the two fields "Smooth the flows" switches on
PS_FLOWS_OFF_NOTE = "The two fields above are used only with Smooth the flows."

# ── Help lines: the settings both steps share ─────────────────────
#: widget -> help line: the cell width, Even out brightness, Rescale and
#: the 43 Advanced controls (the 3 knobs, the 38 group controls,
#: ``do_mp_cb``, ``skip_postprocess_cb``).
RECON_HELP_LINES = {
    "cell_width_um_sb": (
        "About how wide one cell is, in micrometres; the model looks for "
        "cells near this size."),
    "normalize_cb": "Stretch each slice’s brightness before the model sees it.",
    "rescale_cb": (
        "Used only when no cell width is set: when ticked, the image is "
        "passed to the model at its own scale; when unticked, the model "
        "resizes it by its own trained cell size. The factor used is "
        "recorded with the result."),
    # the three visible knobs above the groups
    "batch_size_sb": (
        "How many slices go to the graphics card together; lower it if the "
        "card runs out of memory."),
    "method_combo": (
        "The way the three views are joined into 3D cells. Only Direct is "
        "offered."),
    "n_cpu_sb": "How many processor cores the 3D build uses.",
    # Smooth before joining
    "ps_smooth_probs_cb": (
        "Blur each view's cell-probability map before the views are joined, "
        "so cells do not step from slice to slice."),
    "ps_smooth_flows_cb": "Blur each view's flows before the views are joined.",
    "ps_sigma_perp_sb": (
        "How far the blur reaches across the slices of each view; higher "
        "removes stripes between slices."),
    "ps_sigma_inplane_sb": (
        "How far the blur reaches within a slice; keep it low to keep "
        "detail."),
    "ps_mask_thresh_sb": (
        "Flows where the cell probability is below this are set to zero. "
        "Used only with Smooth the flows."),
    "ps_soft_weight_cb": (
        "Scale the smoothed flows by how likely each voxel is inside a cell. "
        "Used only with Smooth the flows."),
    # Follow the flows
    "gd_decay_sb": (
        "How fast each step shrinks as voxels follow the flows; higher suits "
        "long, thin cells."),
    "gd_iters_sb": "How many steps each voxel takes along the flows.",
    "gd_momenta_sb": (
        "How much of the previous step each step keeps (0 to 1). Rarely "
        "changed."),
    "gd_renorm_cb": "Set the flows back to unit length at every step.",
    "gd_tile_z_sb": (
        "The depth of the pieces the volume is cut into for several cores. "
        "Used only with Use several cores."),
    "gd_tile_y_sb": (
        "The height of those pieces. Used only with Use several cores."),
    "gd_tile_x_sb": (
        "The width of those pieces. Used only with Use several cores."),
    "gd_tile_overlap_sb": (
        "How much neighbouring pieces overlap, as a fraction. Used only with "
        "Use several cores."),
    "do_mp_cb": (
        "Cut the volume into pieces and follow the flows on several cores at "
        "once."),
    # Combine the probabilities
    "ccp_ksize_sb": (
        "How many neighbouring voxels are pooled when the three views' "
        "probabilities are joined into 3D."),
    "ccp_alpha_sb": (
        "A small count added before pooling so no single view dominates; "
        "higher evens the views out more."),
    "ccp_smooth_sigma_sb": "Blur the joined 3D cell mask before it is cut.",
    "ccp_threshold_level_sb": (
        "Which automatic cut level decides whether a voxel is a cell; 0 is "
        "the lowest, and it must be below Number of levels minus 1."),
    "ccp_threshold_n_levels_sb": (
        "How many bands the automatic threshold splits the probabilities "
        "into; there is one fewer level to choose from."),
    "ccp_prob_thresh_sb": (
        "Set the cell-or-not cut by hand, from 0 to 1. Auto uses the two "
        "levels above."),
    "ccp_min_prob_thresh_sb": "The automatic cut never goes below this.",
    "ccp_apply_one_d_p_cb": "Round the automatic cut down to one decimal place.",
    # Clean the mask
    "post_small_obj_sb": (
        "Pieces of the 3D mask with fewer voxels than this are removed."),
    "post_fill_cb": "Fill holes inside the 3D mask.",
    "post_closing_sb": (
        "Close small gaps in the mask; the number is the size of the tool in "
        "voxels."),
    "post_dilation_sb": "Grow the mask outwards by this many voxels.",
    "post_extra_erode_sb": (
        "Shrink the mask by this many voxels after the steps above."),
    # Combine the flows
    "ccg_ksize_sb": (
        "How many neighbouring voxels are pooled when the three views' flows "
        "are joined into 3D."),
    "ccg_alpha_sb": (
        "A small count added before pooling so no single view dominates."),
    "ccg_smooth_sigma_sb": "Blur each view's flows before they are joined.",
    "ccg_post_sigma_sb": "Blur the 3D flows after they are joined.",
    # Group into cells
    "cc_min_area_sb": (
        "Clusters of fewer voxels than this are not made into cells."),
    "cc_thresh_factor_sb": (
        "Where points count as crowded: the average density plus this many "
        "spreads."),
    "cc_smooth_sigma_sb": (
        "How much the point density is blurred before clusters are found."),
    # Size limits
    "skip_postprocess_cb": (
        "When ticked (the default), the final clean-up after the 3D build "
        "does not run, so the size limits below drop nothing. Untick it to "
        "apply them."),
    "sf_min_size_sb": "Cells with fewer voxels than this are dropped.",
    "sf_max_size_sb": "Cells with more voxels than this are dropped.",
    "sf_max_size_factor_sb": (
        "With the tick below: a cell larger than the average size plus this "
        "many spreads is dropped."),
    "sf_do_stats_cb": (
        "Drop cells much larger than the typical cell, by the rule above."),
}

# ── The Advanced labels ───────────────────────────────────────────
#: widget -> (plain label, engine name), for every VISIBLE Advanced control
#: (3 knobs + the 38 group controls + ``do_mp_cb`` + ``skip_postprocess_cb``).
STEP4_ADVANCED_LABELS = {
    "batch_size_sb": ("Batch size (slices at once)", "batch_size"),
    "method_combo": ("How 2D becomes 3D", "method"),
    "n_cpu_sb": ("CPU cores", "n_cpu"),
    "ps_smooth_probs_cb": ("Smooth the cell maps", "pre_smooth_probs"),
    "ps_smooth_flows_cb": ("Smooth the flows", "pre_smooth_flows"),
    "ps_sigma_perp_sb": ("Blur across slices", "pre_smooth_sigma_perp"),
    "ps_sigma_inplane_sb": ("Blur within a slice", "pre_smooth_sigma_inplane"),
    "ps_mask_thresh_sb": ("Ignore flows below", "flow_mask_threshold"),
    "ps_soft_weight_cb": ("Weigh flows by cell probability",
                          "flow_soft_weighting"),
    "gd_decay_sb": ("Step fade", "gradient_decay"),
    "gd_iters_sb": ("Steps", "n_iter"),
    "gd_momenta_sb": ("Momentum", "momenta"),
    "gd_renorm_cb": ("Rescale the flows at each step", "renorm_gradient"),
    "gd_tile_z_sb": ("Piece depth", "tile_shape[0]"),
    "gd_tile_y_sb": ("Piece height", "tile_shape[1]"),
    "gd_tile_x_sb": ("Piece width", "tile_shape[2]"),
    "gd_tile_overlap_sb": ("Piece overlap", "tile_overlap_ratio"),
    "do_mp_cb": ("Use several cores", "do_mp"),
    "ccp_ksize_sb": ("Neighbourhood size", "ksize"),
    "ccp_alpha_sb": ("Balance between views", "alpha"),
    "ccp_smooth_sigma_sb": ("Smooth the joined mask", "smooth_sigma"),
    "ccp_threshold_level_sb": ("Threshold level", "threshold_level"),
    "ccp_threshold_n_levels_sb": ("Number of levels", "threshold_n_levels"),
    "ccp_prob_thresh_sb": ("Fixed cell threshold", "prob_thresh"),
    "ccp_min_prob_thresh_sb": ("Lowest cell threshold", "min_prob_thresh"),
    "ccp_apply_one_d_p_cb": ("Round the threshold down to 0.1",
                             "apply_one_d_p_thresh"),
    "post_small_obj_sb": ("Remove specks smaller than", "remove_small_objects"),
    "post_fill_cb": ("Fill holes", "binary_fill_holes"),
    "post_closing_sb": ("Close gaps", "binary_closing"),
    "post_dilation_sb": ("Grow the mask", "binary_dilation"),
    "post_extra_erode_sb": ("Shrink the mask", "extra_erode"),
    "ccg_ksize_sb": ("Neighbourhood size", "ksize"),
    "ccg_alpha_sb": ("Balance between views", "alpha"),
    "ccg_smooth_sigma_sb": ("Smooth the 2D flows first", "smooth_sigma"),
    "ccg_post_sigma_sb": ("Smooth the joined flows", "post_sigma"),
    "cc_min_area_sb": ("Smallest cluster", "min_area"),
    "cc_thresh_factor_sb": ("Density cut", "thresh_factor"),
    "cc_smooth_sigma_sb": ("Density blur", "smooth_sigma"),
    "skip_postprocess_cb": ("Skip the size limits", "skip_postprocess"),
    "sf_min_size_sb": ("Smallest cell", "min_size"),
    "sf_max_size_sb": ("Largest cell", "max_size"),
    "sf_max_size_factor_sb": ("Largest vs typical", "max_size_factor"),
    "sf_do_stats_cb": ("Drop cells far from the typical size",
                       "do_stats_filter"),
}

#: The seven plain-named groups, in order: (plain name, engine name, widgets).
STEP4_ADVANCED_GROUPS = (
    ("Smooth before joining", "pre_smoothing",
     ("ps_smooth_probs_cb", "ps_smooth_flows_cb", "ps_sigma_perp_sb",
      "ps_sigma_inplane_sb", "ps_mask_thresh_sb", "ps_soft_weight_cb")),
    ("Follow the flows", "gradient_descent",
     ("gd_decay_sb", "gd_iters_sb", "gd_momenta_sb", "gd_renorm_cb",
      "do_mp_cb", "gd_tile_z_sb", "gd_tile_y_sb", "gd_tile_x_sb",
      "gd_tile_overlap_sb")),
    ("Combine the probabilities", "combine_cell_probs",
     ("ccp_ksize_sb", "ccp_alpha_sb", "ccp_smooth_sigma_sb",
      "ccp_threshold_level_sb", "ccp_threshold_n_levels_sb",
      "ccp_prob_thresh_sb", "ccp_min_prob_thresh_sb",
      "ccp_apply_one_d_p_cb")),
    ("Clean the mask", "postprocess_binary",
     ("post_small_obj_sb", "post_fill_cb", "post_closing_sb",
      "post_dilation_sb", "post_extra_erode_sb")),
    ("Combine the flows", "combine_cell_gradients",
     ("ccg_ksize_sb", "ccg_alpha_sb", "ccg_smooth_sigma_sb",
      "ccg_post_sigma_sb")),
    ("Group into cells", "connected_component",
     ("cc_min_area_sb", "cc_thresh_factor_sb", "cc_smooth_sigma_sb")),
    ("Size limits", "size_filters",
     ("skip_postprocess_cb", "sf_min_size_sb", "sf_max_size_sb",
      "sf_max_size_factor_sb", "sf_do_stats_cb")),
)

#: The two knobs built hidden: no label, no help line.
STEP4_HIDDEN_KNOBS = ("flow_thresh_sb", "cellprob_sb")
