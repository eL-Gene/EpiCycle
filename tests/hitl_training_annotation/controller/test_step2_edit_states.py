"""Tests for Step 2 edit-states (2026-05-27 fix/step2-edit-states).

Covers:
  - _collect_prior_round_pairs: LATEST-round-wins dedup (re-edited slices
    overwrite their prior-round copy)
  - _process_draw_complete: accepted/staged demote to curating on draw;
    committed protected (no-op)
  - _unstage_selected_queue_items (generalized Revert): any finalized
    state -> curating; committed gated by confirm
  - annotator soft-lock visuals: _orient_button_style edit_state variants,
    _edit_state_for_orient mapping, _update_per_view_indicator enable logic
  - controller<->annotator bridge wiring (source inspection)

Pure-Python fakes — no Qt/napari instantiation except qtpy.QtCore.Qt
(import-only, no widget creation).
"""
from __future__ import annotations

import inspect
import numpy as np
import tifffile
from unittest.mock import MagicMock

import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)
from napari_epicycle.hitl_training_annotation import _paths as step2_paths


# ===========================================================================
# Piece 3 — LATEST-round-wins dedup in _collect_prior_round_pairs
# ===========================================================================


def _write_pair(accepted_dir, basename_stem, mask_value):
    """Write an img+mask tif pair under accepted_dir."""
    accepted_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((8, 8), dtype=np.uint8)
    mask = np.full((8, 8), mask_value, dtype=np.int32)
    tifffile.imwrite(str(accepted_dir / f"{basename_stem}_img.tif"), img)
    tifffile.imwrite(str(accepted_dir / f"{basename_stem}_mask.tif"), mask)


def _make_collect_controller(tmp_path):
    ctrl = MagicMock()
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = step2_paths.DEFAULT_RUN_ID
    ctrl.current_round_by_view = {"lateral": 2, "axial": 0}
    ctrl._current_round_key = lambda: "lateral"
    ctrl._current_round = lambda: 2
    ctrl._append_ui_log = lambda *a, **k: None
    # B-lite (2026-09-07): the collector filters on the slice record, so the
    # fake needs real records + the real label-parsing helpers.
    ctrl.sample_records = {}
    ctrl._infer_view_from_label = (
        TrainingAnnotationController._infer_view_from_label.__get__(ctrl)
    )
    ctrl._infer_slice_index = (
        TrainingAnnotationController._infer_slice_index.__get__(ctrl)
    )
    ctrl._slice_key_for_pair_filename = (
        TrainingAnnotationController._slice_key_for_pair_filename.__get__(ctrl)
    )
    ctrl._prior_pair_status_index = (
        TrainingAnnotationController._prior_pair_status_index.__get__(ctrl)
    )
    ctrl._TRAINABLE_PRIOR_STATUSES = (
        TrainingAnnotationController._TRAINABLE_PRIOR_STATUSES
    )
    ctrl._collect_prior_round_pairs = (
        TrainingAnnotationController._collect_prior_round_pairs.__get__(ctrl)
    )
    return ctrl


def test_dedup_latest_round_wins(tmp_path):
    """A slice in BOTH round 0 and round 1 → training uses round 1's copy."""
    ctrl = _make_collect_controller(tmp_path)
    stem = "patchA_XY_z0042"
    ctrl.sample_records[stem] = {"status": "accepted"}

    # Round 0: original version (mask value 11)
    r0 = step2_paths.round_accepted_dir(tmp_path, ctrl.active_run_id, "lateral", 0)
    _write_pair(r0, stem, mask_value=11)
    # Round 1: re-edited version (mask value 22)
    r1 = step2_paths.round_accepted_dir(tmp_path, ctrl.active_run_id, "lateral", 1)
    _write_pair(r1, stem, mask_value=22)

    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    # Exactly ONE pair for the slice (deduped).
    matching = [p for p in pairs if stem in p["mask_uri"]]
    assert len(matching) == 1, f"expected 1 deduped pair, got {len(matching)}"
    # And it must be ROUND 1's copy (the correction), not round 0's.
    mask = tifffile.imread(matching[0]["mask_uri"])
    assert int(mask.max()) == 22, (
        f"dedup used the wrong round — mask max={int(mask.max())} (expected "
        "22 = round 1's re-edited version; 11 would mean the stale round-0 "
        "copy shadowed the correction)"
    )
    assert "round_01" in matching[0]["mask_uri"] or "round_1" in matching[0]["mask_uri"]


def test_dedup_distinct_slices_all_kept(tmp_path):
    """Different slices across rounds are all retained."""
    ctrl = _make_collect_controller(tmp_path)
    r0 = step2_paths.round_accepted_dir(tmp_path, ctrl.active_run_id, "lateral", 0)
    _write_pair(r0, "patchA_XY_z0001", 5)
    r1 = step2_paths.round_accepted_dir(tmp_path, ctrl.active_run_id, "lateral", 1)
    _write_pair(r1, "patchA_XY_z0002", 6)
    ctrl.sample_records.update({
        "patchA_XY_z0001": {"status": "accepted"},
        "patchA_XY_z0002": {"status": "accepted"},
    })

    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")
    stems = {p["mask_uri"].split("\\")[-1].split("/")[-1] for p in pairs}
    assert any("z0001" in s for s in stems)
    assert any("z0002" in s for s in stems)


# ===========================================================================
# Piece 2 — _process_draw_complete demote branches
# ===========================================================================


def _make_draw_controller(status):
    ctrl = MagicMock()
    label = "patchA_XY_5"
    ctrl.sample_records = {label: {"status": status, "view": "XY"}}
    ctrl._touch_sample = MagicMock()
    ctrl._refresh_sample_queue = MagicMock()
    ctrl._schedule_curating_persist = MagicMock()
    ctrl._refresh_annotator_edit_state_visuals = MagicMock()
    ctrl._append_ui_log = MagicMock()
    ctrl._fresh_slice_has_content = MagicMock(return_value=True)
    ctrl._extract_input_stack_for_view = MagicMock(return_value=None)
    ctrl._process_draw_complete = (
        TrainingAnnotationController._process_draw_complete.__get__(ctrl)
    )
    return ctrl, label


def test_draw_on_accepted_demotes_to_curating():
    ctrl, label = _make_draw_controller("accepted")
    ctrl._process_draw_complete(label, "XY", 5)
    # Must demote to curating + schedule persist.
    ctrl._touch_sample.assert_called_once()
    _, kwargs = ctrl._touch_sample.call_args
    assert kwargs.get("status") == "curating"
    ctrl._schedule_curating_persist.assert_called_once()
    ctrl._refresh_annotator_edit_state_visuals.assert_called_once()


def test_draw_on_staged_demotes_to_curating():
    ctrl, label = _make_draw_controller("staged")
    ctrl._process_draw_complete(label, "XY", 5)
    ctrl._touch_sample.assert_called_once()
    _, kwargs = ctrl._touch_sample.call_args
    assert kwargs.get("status") == "curating"
    ctrl._schedule_curating_persist.assert_called_once()


def test_draw_on_committed_is_protected_noop():
    ctrl, label = _make_draw_controller("committed")
    ctrl._process_draw_complete(label, "XY", 5)
    # Committed is PROTECTED — no status change, no persist.
    ctrl._touch_sample.assert_not_called()
    ctrl._schedule_curating_persist.assert_not_called()


def test_draw_on_predicted_still_goes_to_curating():
    """Regression: the existing predicted->curating path still works."""
    ctrl, label = _make_draw_controller("predicted")
    ctrl._process_draw_complete(label, "XY", 5)
    ctrl._touch_sample.assert_called_once()
    _, kwargs = ctrl._touch_sample.call_args
    assert kwargs.get("status") == "curating"
    ctrl._schedule_curating_persist.assert_called_once()


# ===========================================================================
# Piece 4 — generalized Revert (_unstage_selected_queue_items)
# ===========================================================================


class _FakeItem:
    def __init__(self, sample_id):
        self._sid = sample_id
    def data(self, _role):
        return self._sid
    def text(self):
        return self._sid


class _FakeQueue:
    def __init__(self, items):
        self._items = items
    def selectedItems(self):
        return self._items


def _make_revert_controller(records, selected_ids, confirm_committed=True):
    ctrl = MagicMock()
    ctrl.sample_records = records
    ui = MagicMock()
    ui.staged_queue_list = _FakeQueue([_FakeItem(sid) for sid in selected_ids])
    ui.sample_queue_list = ui.staged_queue_list
    ctrl.ui = ui
    ctrl._touch_sample = MagicMock()
    ctrl._sync_data_to_add_list = MagicMock()
    ctrl._refresh_sample_queue = MagicMock()
    ctrl._refresh_round_status = MagicMock()
    ctrl._refresh_annotator_edit_state_visuals = MagicMock()
    ctrl._append_ui_log = MagicMock()
    ctrl._confirm_revert_committed = MagicMock(return_value=confirm_committed)
    ctrl._unstage_selected_queue_items = (
        TrainingAnnotationController._unstage_selected_queue_items.__get__(ctrl)
    )
    return ctrl


def test_revert_accepted_to_curating():
    records = {"p_XY_1": {"status": "accepted", "view": "XY"}}
    ctrl = _make_revert_controller(records, ["p_XY_1"])
    ctrl._unstage_selected_queue_items()
    ctrl._touch_sample.assert_called_once()
    _, kwargs = ctrl._touch_sample.call_args
    assert kwargs.get("status") == "curating"
    # Accepted does NOT trigger the committed confirm.
    ctrl._confirm_revert_committed.assert_not_called()


def test_revert_staged_to_curating():
    records = {"p_XY_1": {"status": "staged", "view": "XY"}}
    ctrl = _make_revert_controller(records, ["p_XY_1"])
    ctrl._unstage_selected_queue_items()
    _, kwargs = ctrl._touch_sample.call_args
    assert kwargs.get("status") == "curating"


def test_revert_committed_requires_confirm_yes():
    records = {"p_XY_1": {"status": "committed", "view": "XY"}}
    ctrl = _make_revert_controller(records, ["p_XY_1"], confirm_committed=True)
    ctrl._unstage_selected_queue_items()
    ctrl._confirm_revert_committed.assert_called_once()
    # Confirmed → reverted to curating.
    _, kwargs = ctrl._touch_sample.call_args
    assert kwargs.get("status") == "curating"


def test_revert_committed_declined_no_change():
    records = {"p_XY_1": {"status": "committed", "view": "XY"}}
    ctrl = _make_revert_controller(records, ["p_XY_1"], confirm_committed=False)
    ctrl._unstage_selected_queue_items()
    ctrl._confirm_revert_committed.assert_called_once()
    # Declined → NO status change.
    ctrl._touch_sample.assert_not_called()


def test_revert_already_editable_is_noop():
    records = {"p_XY_1": {"status": "curating", "view": "XY"}}
    ctrl = _make_revert_controller(records, ["p_XY_1"])
    ctrl._unstage_selected_queue_items()
    ctrl._touch_sample.assert_not_called()


def test_confirm_revert_committed_headless_returns_true():
    """In sync/headless mode, the confirm must not block — returns True."""
    ctrl = MagicMock()
    ctrl._use_sync_training = True
    ctrl._confirm_revert_committed = (
        TrainingAnnotationController._confirm_revert_committed.__get__(ctrl)
    )
    assert ctrl._confirm_revert_committed(["p_XY_1"]) is True


# ===========================================================================
# Annotator soft-lock visuals
# ===========================================================================


def test_orient_button_style_accepted_is_emerald():
    from napari_epicycle.annotator.curation_2d_widget import _orient_button_style
    style = _orient_button_style("XY", edit_state="accepted")
    assert "#2ecc71" in style or "46, 204, 113" in style


def test_orient_button_style_finalized_is_ice():
    from napari_epicycle.annotator.curation_2d_widget import _orient_button_style
    style = _orient_button_style("XY", edit_state="finalized")
    assert "#80deea" in style or "128, 222, 234" in style


def test_orient_button_style_editable_uses_orient_tint():
    from napari_epicycle.annotator.curation_2d_widget import _orient_button_style, _ORIENT_TINT
    style = _orient_button_style("XY", edit_state="editable")
    # Editable XY uses the yellow orient tint, NOT the edit-state accents.
    assert _ORIENT_TINT["XY"] in style
    assert "#2ecc71" not in style
    assert "#80deea" not in style


def _make_annotator_for_edit_state(status, idx=5):
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
    ann = MagicMock()
    ann._slice_status_callback = lambda view, i: status
    fake_vm = MagicMock()
    fake_vm.dims.current_step = (idx,)
    ann._vm_for_orient = lambda orient: fake_vm
    ann._edit_state_for_orient = (
        Curation2DAnnotator._edit_state_for_orient.__get__(ann)
    )
    return ann


def test_edit_state_for_orient_accepted():
    ann = _make_annotator_for_edit_state("accepted")
    assert ann._edit_state_for_orient("XY") == "accepted"


def test_edit_state_for_orient_staged_and_committed_are_finalized():
    assert _make_annotator_for_edit_state("staged")._edit_state_for_orient("XY") == "finalized"
    assert _make_annotator_for_edit_state("committed")._edit_state_for_orient("XY") == "finalized"


def test_edit_state_for_orient_editable_states():
    for s in ("predicted", "curating", ""):
        ann = _make_annotator_for_edit_state(s)
        assert ann._edit_state_for_orient("XY") == "editable"


def test_edit_state_for_orient_no_callback_fails_open():
    """No status callback → fail-open to editable (never stuck locked)."""
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
    ann = MagicMock()
    ann._slice_status_callback = None
    ann._edit_state_for_orient = (
        Curation2DAnnotator._edit_state_for_orient.__get__(ann)
    )
    assert ann._edit_state_for_orient("XY") == "editable"


# ===========================================================================
# Bridge wiring (source inspection)
# ===========================================================================


def test_controller_injects_status_callback_at_launch():
    src = inspect.getsource(
        TrainingAnnotationController._launch_multiview_annotator
    )
    assert "_slice_status_callback = self._status_for_view_slice" in src
    assert "refresh_edit_state_visuals" in src


def test_draw_complete_demotes_accepted_and_protects_committed():
    """Source guard: the demote branch + committed-protection exist."""
    src = inspect.getsource(TrainingAnnotationController._process_draw_complete)
    assert 'status in ("accepted", "staged")' in src
    assert "PROTECTED" in src or "committed" in src


def test_collect_prior_round_pairs_is_latest_wins():
    """Source guard against regression to first-occurrence-wins."""
    src = inspect.getsource(TrainingAnnotationController._collect_prior_round_pairs)
    assert "latest_by_basename" in src
    assert "LATEST-round-wins" in src or "latest" in src.lower()


def test_revert_handler_generalized():
    src = inspect.getsource(TrainingAnnotationController._unstage_selected_queue_items)
    assert "curating" in src
    assert "_confirm_revert_committed" in src


def test_apply_edit_constraints_override_still_present():
    """The training-lock _apply_edit_constraints override (merged from
    fix/step2-training-lock) must survive this branch."""
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
    assert "_apply_edit_constraints" in Curation2DAnnotator.__dict__
