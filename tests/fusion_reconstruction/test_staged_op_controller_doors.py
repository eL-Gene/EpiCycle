"""With an edit open in the annotator, every door that would save, persist,
undo, accept, leave the patch or forget it refuses first - with the one
sentence - and changes nothing."""
from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from napari_epicycle.annotator._staged_op_text import STAGED_WORK_OPEN
from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
    _refuse_if_staged,
)


def _ctrl(staged=True):
    c = C.__new__(C)
    c.logged = []
    c._append_log = c.logged.append
    c.cards, c.session_lines, c.statuses = [], [], []
    c._set_plot_status = lambda text, kind="info": c.cards.append((text, kind))
    c._session_say = c.session_lines.append
    c._pr_status = lambda text, kind="neutral": c.statuses.append(text)
    c._staged_annotator_work = lambda: staged
    c._curation_dirty = True
    c._tiers_dirty = True
    c._save_in_flight = False
    c._qc_features_df = object()
    c.ui = MagicMock()
    c.working_dir = "x"
    c.state_manager = MagicMock()
    for name in ("_save_tiers_csv", "_save_curation", "_save_curation_async",
                 "_commit_triage_prepare", "_commit_triage_single_patch"):
        setattr(c, name, lambda *a, _n=name, **k: pytest.fail(_n + " ran"))
    return c


def test_sentence_is_the_shared_one():
    assert _pr_text.S3_STAGED_WORK_OPEN == STAGED_WORK_OPEN


def test_save_click_refuses_and_writes_nothing():
    c = _ctrl()
    C._on_save_curation_clicked(c)
    assert (STAGED_WORK_OPEN, "warn") in c.cards


def test_commit_triage_refuses_before_anything():
    c = _ctrl()
    C._commit_triage(c)
    assert (STAGED_WORK_OPEN, "warn") in c.cards


def test_undo_refuses_and_keeps_the_stack():
    c = _ctrl()
    c._curation_undo_stack = MagicMock()
    C._undo_last_curation(c)
    c._curation_undo_stack.pop.assert_not_called()
    assert (STAGED_WORK_OPEN, "warn") in c.cards


def test_accept_and_finalize_refuse_without_a_save_failed_log():
    for fn, name in ((C._on_3d_curation_accept, "accept"),
                     (C._finalize_curation, "finalize")):
        c = _ctrl()
        import napari_epicycle.fusion_reconstruction.controller as mod
        mod_active = mod._active_qc_patch
        mod._active_qc_patch = lambda ctl: "P1.tif"
        try:
            fn(c)
        finally:
            mod._active_qc_patch = mod_active
        c.state_manager.set_patch_status.assert_not_called()
        assert not any("save failed" in line for line in c.logged)
        assert (STAGED_WORK_OPEN, "warn") in c.cards


def test_save_checked_refuses_before_the_clean_shortcut():
    c = _ctrl()
    c._curation_dirty = c._tiers_dirty = False
    assert C._save_curation_checked(c) is False


def test_guard_refuses_in_all_three_places_before_any_box():
    c = _ctrl()
    with pytest.MonkeyPatch.context() as mp:
        box = MagicMock()
        import napari_epicycle.fusion_reconstruction.controller as mod
        mp.setattr(mod, "QMessageBox", box)
        assert C._guard_unsaved_changes(c) is False
    box.information.assert_not_called()
    box.question.assert_not_called()
    assert (STAGED_WORK_OPEN, "warn") in c.cards
    assert c.session_lines == [STAGED_WORK_OPEN]
    assert c.statuses == [STAGED_WORK_OPEN]
    assert c._curation_dirty is True and c._tiers_dirty is True


def test_forget_in_view_refuses_in_session_and_the_other_two_places():
    c = _ctrl()
    c._pr_forget_refusal = lambda name: None
    c._selected_curation_patch_safe = lambda: "P1.tif"
    c._qc_view_patch_name = "P1.tif"
    c.working_dir = "x"
    assert C._forget_patch(c, "P1.tif") is False
    assert c.statuses == [STAGED_WORK_OPEN]
    assert c.session_lines == [STAGED_WORK_OPEN]
    assert (STAGED_WORK_OPEN, "warn") in c.cards


def test_an_idle_annotator_lets_the_guard_through_to_its_own_checks():
    c = _ctrl(staged=False)
    assert _refuse_if_staged(c, "x") is False
    assert c.cards == []


def test_a_raising_check_never_blocks():
    c = _ctrl()

    def boom():
        raise RuntimeError("x")

    c._staged_annotator_work = boom
    assert _refuse_if_staged(c, "x") is False


def test_undo_all_session_counts_nothing_when_an_edit_is_open():
    c = _ctrl()
    stack = MagicMock()
    stack.__len__.return_value = 2
    stack.peek.return_value = SimpleNamespace(action_type="add_denovo")
    c._curation_undo_stack = stack
    assert C._on_annotator_add_undo_session(c, 2) == 0
    stack.pop.assert_not_called()


def test_post_extract_switch_in_place_skips_silently_while_an_edit_is_open(
        monkeypatch):
    from qtpy import QtCore

    c = _ctrl()
    c._run_epoch = 0
    c._qc_view_patch_name = "P1.tif"
    c._switch_to_curation_patch = lambda *a, **k: pytest.fail("switched")
    c._update_flagged_labels_list = lambda: pytest.fail("refreshed")
    fire = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot",
                        staticmethod(lambda ms, fn: fire.append(fn)))
    C._switch_in_place_after_extract(c, "P1.tif")
    assert len(fire) == 1
    fire[0]()
    assert c.cards == [] and c.statuses == [] and c.session_lines == []
