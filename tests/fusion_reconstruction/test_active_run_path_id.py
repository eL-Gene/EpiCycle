"""Phase 1b-i: ``_active_run_path_id()`` run-id → path-id mapping.

Option A (default run = flat layout, zero migration): the DEFAULT run's
path-id is ``None`` so the per-run ``_paths`` helpers reproduce today's
flat paths byte-for-byte; a NAMED run returns its slug so reads/writes
nest under ``runs/{slug}/...``.

Tested as a pure mapping via ``__new__`` + a tiny fake state_manager —
no Qt widget construction required (mirrors test_workspace_cache_reset).
"""

from __future__ import annotations

from napari_epicycle.fusion_reconstruction import _migrate_step3_runs
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


class _FakeStateManager:
    def __init__(self, run_id: str):
        self._run_id = run_id

    def get_active_run_id(self) -> str:
        return self._run_id


def _controller_with_run(run_id: str) -> FusionReconstructionController:
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c.state_manager = _FakeStateManager(run_id)
    return c


def test_default_run_maps_to_none():
    c = _controller_with_run(_migrate_step3_runs.DEFAULT_RUN_ID)
    assert c._active_run_path_id() is None


def test_default_literal_maps_to_none():
    # DEFAULT_RUN_ID is the literal "default" today; guard the literal too.
    c = _controller_with_run("default")
    assert c._active_run_path_id() is None


def test_named_run_maps_to_slug():
    c = _controller_with_run("cells")
    assert c._active_run_path_id() == "cells"


def test_other_named_run_maps_to_its_slug():
    c = _controller_with_run("nuclei")
    assert c._active_run_path_id() == "nuclei"


# ---------------------------------------------------------------------------
# Phase 7: default_run_nested flag flips the default run's path-id to "default"
# ---------------------------------------------------------------------------


class _FakeStateManagerWithFlag(_FakeStateManager):
    def __init__(self, run_id: str, nested: bool):
        super().__init__(run_id)
        self._nested = nested

    def get_default_run_nested(self) -> bool:
        return self._nested


def _controller_with_flag(run_id: str, nested: bool):
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c.state_manager = _FakeStateManagerWithFlag(run_id, nested)
    return c


def test_default_run_flat_when_flag_false():
    c = _controller_with_flag(_migrate_step3_runs.DEFAULT_RUN_ID, nested=False)
    assert c._active_run_path_id() is None


def test_default_run_nested_when_flag_true():
    c = _controller_with_flag(_migrate_step3_runs.DEFAULT_RUN_ID, nested=True)
    assert c._active_run_path_id() == _migrate_step3_runs.DEFAULT_RUN_ID


def test_named_run_unaffected_by_flag():
    c = _controller_with_flag("cells", nested=True)
    assert c._active_run_path_id() == "cells"


def test_absent_flag_getter_falls_back_to_flat():
    """A state_manager WITHOUT get_default_run_nested (the minimal fake)
    still resolves the default run to flat None — byte-identical to today."""
    c = _controller_with_run(_migrate_step3_runs.DEFAULT_RUN_ID)
    assert c._active_run_path_id() is None
