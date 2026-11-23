"""Merge Accept leaves the Isolate box the way the user had it."""
from __future__ import annotations

from types import SimpleNamespace

from .test_clear_edits_after_interp_reject import _drawn_scene

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K


class _Nav:
    def __init__(self, ticked):
        self.ticked = ticked
        self.set_calls = []

    def is_isolate_checked(self):
        return self.ticked

    def set_isolate_checked(self, v):
        self.set_calls.append(v)
        self.ticked = v

    def remove_item(self, _i):
        pass


def _accept(ticked):
    _, _, f = _drawn_scene()
    f._load_locked = False
    f._merge_dialog = None
    f._on_merge_cb = None
    f._navigator_widget = _Nav(ticked)
    f._merge_iso_box_was_ticked = ticked
    seen = []
    f._on_isolation_changed = lambda ids, **k: seen.append((set(ids), k))
    K._finalize_merge_accept(f, 5, [7])
    return f, seen


def test_box_off_stays_off_and_isolation_is_dropped():
    f, seen = _accept(False)
    assert f._navigator_widget.set_calls == []
    assert f._navigator_widget.ticked is False
    assert seen == [(set(), {"reason": "forget"})]


def test_box_on_keeps_the_merged_cell_isolated_and_ticked():
    f, seen = _accept(True)
    assert seen == [({5}, {})]
    assert f._navigator_widget.set_calls == [True]


def _open(ticked, monkeypatch):
    """Drive the real dialog-open path with a stub dialog."""
    import napari_epicycle.annotator.merge_dialog as md
    import napari_epicycle.fusion_reconstruction.neighbor_query as nq

    class _Dlg:
        def __init__(self, *a, **k):
            pass

        def show(self):
            pass

        def raise_(self):
            pass

    monkeypatch.setattr(md, "MergeDialog", _Dlg)
    monkeypatch.setattr(nq, "get_neighbor_ids", lambda *a, **k: [])
    layer = SimpleNamespace(data=object())
    f = SimpleNamespace(
        _sync_refine_state=lambda: None,
        _refine_state=SimpleNamespace(can_open_merge_dialog=lambda: True),
        _split_running=False,
        labels_combo=SimpleNamespace(currentText=lambda: "labels"),
        viewer=SimpleNamespace(layers={"labels": layer}),
        _get_label_metadata_for_ids=lambda ids: {},
        _merge_dialog=None,
        _navigator_widget=_Nav(ticked),
        _refresh_merge_button_state=lambda: None,
        _refresh_label_action_buttons_state=lambda: None,
        _execute_batch_merge=None, _finalize_merge_accept=None,
        _cancel_merge_dialog=None,
        _on_isolation_changed=lambda ids, **k: None,
    )
    K._open_merge_dialog(f, 5)
    return f


def test_opening_the_dialog_records_the_box_and_accept_honours_it(monkeypatch):
    f = _open(False, monkeypatch)
    assert f._merge_iso_box_was_ticked is False
    f._load_locked = False
    f._on_merge_cb = None
    f._merge_dialog = None
    f._clear_draw_snapshot = lambda: None
    seen = []
    f._on_isolation_changed = lambda ids, **k: seen.append((set(ids), k))
    K._finalize_merge_accept(f, 5, [7])
    assert f._navigator_widget.set_calls == []
    assert seen == [(set(), {"reason": "forget"})]
    g = _open(True, monkeypatch)
    assert g._merge_iso_box_was_ticked is True
