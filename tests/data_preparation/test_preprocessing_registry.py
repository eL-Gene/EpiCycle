"""Brick 1a tests — the pure Step-1 preprocessing op registry.

Covers the preprocessing op set: the five v1 ops are
``implemented=True`` with the right kind/scope/params; the ported catalogue is
``implemented=False``; ``by_bucket`` groups; keys are unique; geometry ops have
scope ``"all"``, intensity ops scope ``"per_channel"``.

Pure — no napari / Qt.
"""

from __future__ import annotations

import pytest

from packages.epicycle_preprocessing import registry as reg
from packages.epicycle_preprocessing.registry import (
    BUCKET_ORDER,
    REGISTRY,
    OpSpec,
    ParamSpec,
    by_bucket,
    find,
    get,
    implemented_keys,
)

#: The five ops that carry a runnable fn in v1 (fn itself is Brick 2).
V1_IMPLEMENTED = {"isotropic", "gaussian", "clahe", "background", "bleach"}


def test_registry_is_pure_no_qt_napari():
    """The module must not drag in napari / Qt (napari-free by design)."""
    import sys

    reg_mod = sys.modules["packages.epicycle_preprocessing.registry"]
    # Importing the registry must not have pulled in napari/qtpy.
    # (They may be importable elsewhere; assert THIS module didn't import them.)
    src = reg_mod.__dict__
    assert "napari" not in src
    assert "qtpy" not in src


def test_every_key_is_unique():
    keys = [op.key for op in REGISTRY]
    assert len(keys) == len(set(keys)), "duplicate op key in REGISTRY"


def test_all_entries_are_opspecs_with_paramspecs():
    for op in REGISTRY:
        assert isinstance(op, OpSpec)
        assert isinstance(op.key, str) and op.key
        assert isinstance(op.label, str) and op.label
        assert op.kind in ("geometry", "intensity")
        assert op.scope in ("all", "per_channel")
        assert isinstance(op.params, list)
        for p in op.params:
            assert isinstance(p, ParamSpec)
            assert p.type in ("float", "int", "choice")
            if p.type == "choice":
                assert p.choices is not None and len(p.choices) >= 1
                assert p.default in p.choices


def test_the_five_are_implemented():
    assert set(implemented_keys()) == V1_IMPLEMENTED
    for key in V1_IMPLEMENTED:
        assert get(key).implemented is True


def test_ported_others_are_not_implemented():
    for op in REGISTRY:
        if op.key not in V1_IMPLEMENTED:
            assert op.implemented is False, f"{op.key} should be declared-only"


def test_implemented_ops_carry_a_runnable_fn():
    # Brick 2a wired the five v1 ops' fns; the declared-only rest stay fn=None.
    for op in REGISTRY:
        if op.key in V1_IMPLEMENTED:
            assert callable(op.fn), f"{op.key} should carry a runnable fn"
        else:
            assert op.fn is None, f"{op.key} is declared-only → fn must be None"


def test_neighborhood_ops_carry_a_halo_the_rest_none():
    # Trivial halos wired for the neighborhood ops (L2 map_overlap); others None.
    halo_ops = {"gaussian", "background", "clahe"}
    for op in REGISTRY:
        if op.key in halo_ops:
            assert callable(op.halo), f"{op.key} should carry a halo callable"
            assert isinstance(op.halo(), int)
        else:
            assert op.halo is None, f"{op.key} should have no halo"


def test_geometry_ops_are_scope_all():
    geometry = [op for op in REGISTRY if op.kind == "geometry"]
    assert geometry, "expected at least one geometry op"
    for op in geometry:
        assert op.scope == "all", f"{op.key} geometry must be scope=all"


def test_intensity_ops_are_per_channel():
    intensity = [op for op in REGISTRY if op.kind == "intensity"]
    assert intensity
    for op in intensity:
        assert op.scope == "per_channel", f"{op.key} intensity must be per_channel"


def test_isotropic_shape():
    op = get("isotropic")
    assert op.kind == "geometry" and op.scope == "all" and op.implemented
    names = [p.name for p in op.params]
    assert names == ["mode", "target_um", "z", "y", "x", "method"]
    by_name = {p.name: p for p in op.params}
    assert by_name["mode"].type == "choice"
    # Canonical stored tokens (grill note N2): lowercase {"iso","aniso"} and
    # {"nearest","linear","cubic","area"} — display labels live in the UI.
    assert set(by_name["mode"].choices) == {"iso", "aniso"}
    assert by_name["mode"].default == "iso"
    assert by_name["target_um"].type == "float"
    assert by_name["target_um"].unit == "µm"
    for axis in ("z", "y", "x"):
        assert by_name[axis].type == "float"
    assert by_name["method"].type == "choice"
    assert set(by_name["method"].choices) == {
        "nearest", "linear", "cubic", "area"
    }
    assert by_name["method"].default == "linear"


def test_gaussian_shape():
    op = get("gaussian")
    assert op.kind == "intensity" and op.scope == "per_channel" and op.implemented
    assert [p.name for p in op.params] == ["sigma"]
    assert op.params[0].type == "float" and op.params[0].unit == "px"


def test_clahe_shape():
    op = get("clahe")
    assert op.implemented and op.kind == "intensity"
    by_name = {p.name: p for p in op.params}
    assert set(by_name) == {"clip", "kernel"}
    assert by_name["clip"].type == "float"
    assert by_name["kernel"].type == "int" and by_name["kernel"].unit == "px"


def test_background_shape():
    op = get("background")
    assert op.implemented and op.kind == "intensity"
    assert [p.name for p in op.params] == ["radius"]
    assert op.params[0].type == "int"


def test_bleach_shape():
    op = get("bleach")
    assert op.implemented and op.kind == "intensity"
    assert [p.name for p in op.params] == ["model", "on_non_decreasing"]
    assert op.params[0].type == "choice"
    assert set(op.params[0].choices) == {"exp"}
    # The non-decreasing-profile policy is a first-class configurable choice.
    assert op.params[1].type == "choice"
    assert op.params[1].default == "skip"
    assert set(op.params[1].choices) == {"skip", "flatten"}


def test_ported_catalogue_present():
    # A representative sweep of the ported (declared-only) ops from _OP_SUITE.
    expected_declared = {
        "downsample", "median", "bilateral", "nlm", "total_variation",
        "anisotropic_diffusion", "wavelet", "minmax_stretch", "gamma", "log",
        "invert", "histogram_equalize", "morphological_opening", "flat_field",
        "subtract_large_sigma_blur", "z_attenuation", "per_slice_histogram_match",
        "unsharp_mask", "high_pass", "sobel", "sato_frangi",
    }
    keys = {op.key for op in REGISTRY}
    missing = expected_declared - keys
    assert not missing, f"missing ported ops: {missing}"
    for key in expected_declared:
        assert get(key).implemented is False


def test_by_bucket_groups_in_order():
    grouped = by_bucket()
    # Only buckets that actually have ops are returned, in BUCKET_ORDER.
    returned = list(grouped.keys())
    known_order = [b for b in BUCKET_ORDER if b in returned]
    assert returned[: len(known_order)] == known_order
    # Every registered op appears exactly once across the buckets.
    flat = [op for ops in grouped.values() for op in ops]
    assert len(flat) == len(REGISTRY)
    assert {op.key for op in flat} == {op.key for op in REGISTRY}


def test_by_bucket_geometry_holds_the_geometry_ops():
    grouped = by_bucket()
    assert "Geometry" in grouped
    geom_keys = {op.key for op in grouped["Geometry"]}
    assert "isotropic" in geom_keys and "downsample" in geom_keys
    # Every op in the Geometry bucket is a geometry op.
    for op in grouped["Geometry"]:
        assert op.kind == "geometry"


def test_get_unknown_raises_find_returns_none():
    with pytest.raises(KeyError):
        get("does_not_exist")
    assert find("does_not_exist") is None
    assert find("gaussian") is get("gaussian")
