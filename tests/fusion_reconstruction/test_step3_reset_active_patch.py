"""Reset to Predicted on the patch that is open in the viewer.

The reset deletes the patch's run folder.  If the patch stays loaded, its
labels path can no longer be resolved, so Discard, Save, step exit and a
patch switch all fail.  The open patch must therefore be unloaded first.
"""
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction.controller import (  # noqa: E402
    FusionReconstructionController as CTL,
)


class Bare:
    """A controller stand-in: unset attributes are MagicMocks."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        if name.startswith("__") or not hasattr(CTL, name):
            raise AttributeError(name)
        m = MagicMock(name=name)
        self.__dict__[name] = m
        return m


class _Item:
    def __init__(self, name):
        self._n = name

    def data(self, role):
        return self._n


class _SM:
    def __init__(self, statuses):
        self.statuses = dict(statuses)

    def get_all_patch_statuses(self):
        return dict(self.statuses)

    def get_patch_status(self, n):
        return self.statuses.get(n, "unprocessed")

    def reset_to_predicted(self, n):
        self.statuses[n] = "predicted"

    def flush(self):
        pass


def _make(selected, active="a.tif", **kw):
    base = dict(
        state_manager=_SM({"a.tif": "curating", "b.tif": "curating"}),
        _edited_patches=set(),
        _curation_dirty=True,
        _tiers_dirty=True,
        _qc_view_patch_name=active,
        _curation_undo_stack=["x"],
        _save_in_flight=False,
        _patch_load_in_flight=False,
        _staged_annotator_work=lambda: False,
        _refresh_patch_state_list=lambda: None,
        _refresh_patch_list_colors=lambda: None,
        _refresh_summary_dashboard=lambda: None,
        _refresh_refinement_patch_list=lambda: None,
        _refresh_start_from_items=lambda: None,
        _refresh_step3_thumbs_after_labels=lambda *_: None,
        _selected_curation_patch=lambda: active,
    )
    base.update(kw)
    b = Bare(**base)
    b.ui = MagicMock()
    b.ui._curation_bucket.selectedItems.return_value = [_Item(n) for n in selected]
    order = MagicMock()
    b._order = order
    b._close_3d_curation_annotator = order.close
    b._cleanup_old_reconstruction_folders = order.cleanup
    b._set_plot_status = MagicMock()
    b._hard_clear_curation_surfaces = MagicMock()
    b._refresh_save_button_state = MagicMock()
    for name in ("_pr_forget_reset_view", "_reset_patches_to_predicted",
                 "_pr_any_block_flag", "_clear_all_undo_stacks",
                 "_drop_tier_undo"):
        b.__dict__[name] = types.MethodType(getattr(CTL, name), b)
    return b


@pytest.fixture
def boxes(monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    q = MagicMock(return_value=QMessageBox.Yes)
    c = MagicMock()
    monkeypatch.setattr(QMessageBox, "question", staticmethod(q))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(c))
    return types.SimpleNamespace(question=q, critical=c, box=QMessageBox)


def test_reset_of_active_dirty_patch_unloads_it_first(boxes):
    b = _make(["a.tif"])
    CTL._on_reset_to_predicted(b)
    names = [c[0] for c in b._order.mock_calls]
    assert "close" in names and "cleanup" in names
    assert names.index("close") < names.index("cleanup")
    assert b._curation_dirty is False
    assert b._tiers_dirty is False
    assert b._qc_view_patch_name == ""
    assert b._curation_undo_stack == []
    b._hard_clear_curation_surfaces.assert_called_once()
    b._refresh_save_button_state.assert_called_once()
    assert b.state_manager.statuses["a.tif"] == "predicted"
    # the guard now lets the user move on without any box
    boxes.question.reset_mock()
    assert CTL._guard_unsaved_changes(b) is True
    boxes.question.assert_not_called()
    boxes.critical.assert_not_called()


def test_reset_of_other_patch_leaves_active_one_loaded(boxes):
    b = _make(["b.tif"], active="a.tif")
    CTL._on_reset_to_predicted(b)
    assert b._curation_dirty is True
    assert b._qc_view_patch_name == "a.tif"
    b._order.close.assert_not_called()
    assert b.state_manager.statuses["b.tif"] == "predicted"
    assert b.state_manager.statuses["a.tif"] == "curating"


@pytest.mark.parametrize("flag", ["_save_in_flight", "_patch_load_in_flight",
                                  "_curation_busy", "_qc_extract_running"])
def test_reset_refused_while_save_or_load_in_flight(boxes, flag):
    b = _make(["a.tif"], **{flag: True})
    CTL._on_reset_to_predicted(b)
    boxes.question.assert_not_called()
    b._order.cleanup.assert_not_called()
    b._order.close.assert_not_called()
    assert b._curation_dirty is True
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    b._set_plot_status.assert_called_once_with(P.S3_RESET_BUSY, "warn")
    assert len(P.S3_RESET_BUSY) <= 60
    assert b._append_log.called


def test_reset_answered_no_changes_nothing(boxes):
    boxes.question.return_value = boxes.box.No
    b = _make(["a.tif"])
    CTL._on_reset_to_predicted(b)
    b._order.cleanup.assert_not_called()
    b._order.close.assert_not_called()
    assert b._curation_dirty is True
    assert b._qc_view_patch_name == "a.tif"
    assert b._curation_undo_stack == ["x"]
    assert b.state_manager.statuses["a.tif"] == "curating"


def test_reset_refused_while_an_op_is_staged(boxes):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    b = _make(["a.tif"], _staged_annotator_work=lambda: True)
    CTL._on_reset_to_predicted(b)
    boxes.question.assert_not_called()
    b._order.close.assert_not_called()
    b._order.cleanup.assert_not_called()
    b._set_plot_status.assert_called_once_with(P.S3_STAGED_WORK_OPEN, "warn")


def test_reset_with_nothing_loaded_unloads_nothing(boxes):
    b = _make(["a.tif"], active="")
    b._qc_view_patch_name = ""
    b._selected_curation_patch = lambda: "a.tif"
    CTL._on_reset_to_predicted(b)
    b._order.close.assert_not_called()
    b._order.cleanup.assert_called_once()


def test_dialog_says_the_open_patch_will_be_closed(boxes):
    b = _make(["a.tif"])
    CTL._on_reset_to_predicted(b)
    assert "The patch open in the viewer will be closed." in boxes.question.call_args[0][2]
    boxes.question.reset_mock()
    b2 = _make(["b.tif"], active="a.tif")
    CTL._on_reset_to_predicted(b2)
    assert "will be closed" not in boxes.question.call_args[0][2]
