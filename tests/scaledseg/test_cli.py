"""CLI argument-surface guards.

Pins the Phase-1 ``--resume`` wiring: the flag must exist, default to False, and
flip to True when passed. The actual ``pipeline.run(resume=args.resume)`` glue is
a one-line pass-through over the already-tested pipeline resume behavior
(test_pipeline.py / test_integration.py); this guards the flag itself so a
regression that drops it is caught.
"""

from packages.epicycle_scaledseg.cli.main import build_parser, _should_run_features

_BASE = ["run", "--work-dir", "x", "--lateral-model", "a", "--axial-model", "b"]


def test_resume_flag_defaults_false():
    args = build_parser().parse_args(_BASE)
    assert args.resume is False


def test_resume_flag_sets_true():
    args = build_parser().parse_args(_BASE + ["--resume"])
    assert args.resume is True


def test_mode_default_is_coarse():
    # Guards the documented default so a stitch-mode flip is deliberate, not silent.
    assert build_parser().parse_args(_BASE).mode == "coarse"


def test_recon_mode_default_is_sequential():
    assert build_parser().parse_args(_BASE).recon_mode == "sequential"


# --- feature stage flags -----------------------------------------------------

def test_features_is_a_valid_stage_choice():
    args = build_parser().parse_args(_BASE + ["--stages", "features"])
    assert args.stages == ["features"]


def test_features_not_in_default_stages():
    # 'features' is opt-in — the default run must NOT include it.
    assert build_parser().parse_args(_BASE).stages == ["all"]


def test_feature_flag_defaults():
    args = build_parser().parse_args(_BASE)
    assert args.features_dry_run is False
    assert args.feat_voxel_size is None
    assert args.feat_min_voxels == 1
    assert args.no_feat_surface is False
    assert args.feat_k_neighbors == 6
    assert args.feat_density_radius is None
    assert args.no_feat_prefer_clean is False
    assert args.feat_feature_set == "epicycle"      # the product default; the in-code default stays "full"
    assert build_parser().prog == "epicycle-scaledseg"


def test_feature_voxel_size_parses_three_floats():
    args = build_parser().parse_args(_BASE + ["--feat-voxel-size", "2.0", "0.5", "0.5"])
    assert args.feat_voxel_size == [2.0, 0.5, 0.5]


def test_features_opt_in_gate():
    # The core invariant the whole stage hangs on: features is OPT-IN.
    # full + explicit features -> run
    assert _should_run_features("full", ["features"]) is True
    assert _should_run_features("full", ["recon", "features"]) is True
    # full + 'all' -> NOT run (opt-in, never part of 'all')
    assert _should_run_features("full", ["all"]) is False
    # coarse mode -> NEVER run (full only), even when explicitly requested
    assert _should_run_features("coarse", ["features"]) is False
    assert _should_run_features("coarse", ["all"]) is False
