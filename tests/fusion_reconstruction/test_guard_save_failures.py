"""A guard that fails must stop the move it guards (a run switch, a layout
conversion, leaving Step 3), say so, and keep the edits marked unsaved."""
from __future__ import annotations

from unittest.mock import MagicMock

from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)

from .test_step3_switch_run_guard import (
    _fake_orchestrator_controller,
    _run_switch,
)


def _logged(fake):
    return " ".join(str(c.args[0]) for c in fake._append_log.call_args_list)


def test_a_guard_that_raises_aborts_the_run_switch_and_keeps_the_flags():
    fake = _fake_orchestrator_controller(active_run="run_a")
    fake._guard_unsaved_changes = MagicMock(side_effect=OSError("locked"))
    _run_switch(fake, "run_b")
    fake.state_manager.switch_active_run.assert_not_called()
    fake._close_3d_curation_annotator.assert_not_called()
    assert fake._run_epoch == 0
    assert fake._tiers_dirty is True
    assert fake._curation_dirty is True
    assert fake._qc_features_df == "STALE_DF"
    assert _pr_text.S3_SAVE_FAILED in _logged(fake)
