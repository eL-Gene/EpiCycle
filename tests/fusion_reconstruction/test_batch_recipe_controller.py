"""Controller-level tests for the Batch-Refine recipe save/load callbacks.

Qt-free: builds a bare controller via ``__new__`` + stubs (same pattern as
``test_batch_refine_callback.py``).  Verifies the controller callbacks persist
to + read from the workspace JSON via ``_batch_recipe_store`` /
``_paths.batch_refine_recipes_path``.
"""

from __future__ import annotations


def _make_controller(working_dir):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.working_dir = str(working_dir)
    ctrl._append_log = lambda msg: None
    return ctrl


_RECIPE = [
    {"method": "sdf", "params": {"sigma_z": 2.0, "sigma_xy": 1.5, "preserve_topology": True}},
    {"method": "fill_holes", "params": {"axis": "3D"}},
]


def test_save_then_get_round_trip(tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = _make_controller(tmp_path)
    FusionReconstructionController._on_save_batch_recipe(ctrl, "Smooth + Fill", _RECIPE)
    got = FusionReconstructionController._get_saved_batch_recipes(ctrl)
    assert got == {"Smooth + Fill": _RECIPE}


def test_get_empty_on_fresh_workspace(tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = _make_controller(tmp_path)
    assert FusionReconstructionController._get_saved_batch_recipes(ctrl) == {}


def test_save_writes_to_canonical_path(tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    from napari_epicycle.fusion_reconstruction import _paths

    ctrl = _make_controller(tmp_path)
    FusionReconstructionController._on_save_batch_recipe(ctrl, "R", _RECIPE)
    expected = _paths.batch_refine_recipes_path(tmp_path)
    assert expected.exists()


def test_save_no_workspace_is_noop():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.working_dir = ""
    ctrl._append_log = lambda msg: None
    # Must not raise even with no workspace.
    FusionReconstructionController._on_save_batch_recipe(ctrl, "R", _RECIPE)
    assert FusionReconstructionController._get_saved_batch_recipes(ctrl) == {}


def test_save_collision_overwrites(tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = _make_controller(tmp_path)
    FusionReconstructionController._on_save_batch_recipe(ctrl, "X", _RECIPE)
    other = [{"method": "dilate_erode", "params": {"radius": 1, "axis": "XY"}}]
    FusionReconstructionController._on_save_batch_recipe(ctrl, "X", other)
    got = FusionReconstructionController._get_saved_batch_recipes(ctrl)
    assert got == {"X": other}


def test_launcher_passes_recipe_callbacks_source():
    """The controller launch call must wire both recipe callbacks."""
    import inspect
    from napari_epicycle.fusion_reconstruction import controller as ctrl_mod

    src = inspect.getsource(ctrl_mod)
    assert "on_save_batch_recipe=self._on_save_batch_recipe" in src
    assert "get_saved_batch_recipes=self._get_saved_batch_recipes" in src
