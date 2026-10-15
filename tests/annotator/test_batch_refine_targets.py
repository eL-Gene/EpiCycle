"""Tests for Curation3DAnnotator._resolve_batch_targets (all/selected/tier).

Proves the target-resolution logic in isolation so we can tell a code bug
apart from a runtime data/UX issue.  The method only touches
_active_labels_name, sync_manager.shared_labels, the present-id cache the
dialog open takes (_batch_present_ids), _batch_explicit_targets and
_navigator_widget (selected_label_ids() + _items), so a plain fake object
carrying exactly those attributes is bound to the UNBOUND method.  (It used
to build the annotator with ``Curation3DAnnotator.__new__`` and no
``QWidget.__init__``; any attribute read Qt does not know then raises
"super-class __init__() ... was never called".)
"""
import types

import numpy as np

from packages.epicycle_annotator.batch_refine import label_bboxes


class _FakeAnnotator(types.SimpleNamespace):
    """The attributes ``_resolve_batch_targets`` reads, and the method."""

    def _resolve_batch_targets(self, spec):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        return Curation3DAnnotator._resolve_batch_targets(self, spec)

    def _read_plot_selection(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        return Curation3DAnnotator._read_plot_selection(self)


def _ann(vol, nav):
    return _FakeAnnotator(
        _active_labels_name="lbl",
        sync_manager=types.SimpleNamespace(shared_labels={"lbl": vol}),
        _navigator_widget=nav,
        _batch_explicit_targets=None,
        # Seeded as _open_batch_refine_dialog seeds it.
        _batch_present_ids=frozenset(label_bboxes(vol)),
    )


def _vol():
    v = np.zeros((4, 4, 4), dtype=np.int32)
    v[0, 0, 0] = 1
    v[1, 1, 1] = 2
    v[2, 2, 2] = 3
    return v


def test_resolve_all():
    nav = types.SimpleNamespace(selected_label_ids=lambda: set(), _items=[])
    assert _ann(_vol(), nav)._resolve_batch_targets({"mode": "all"}) == [1, 2, 3]


def test_resolve_selected():
    nav = types.SimpleNamespace(selected_label_ids=lambda: {2, 3}, _items=[])
    assert _ann(_vol(), nav)._resolve_batch_targets({"mode": "selected"}) == [2, 3]


def test_resolve_selected_intersects_present():
    nav = types.SimpleNamespace(selected_label_ids=lambda: {2, 99}, _items=[])
    assert _ann(_vol(), nav)._resolve_batch_targets({"mode": "selected"}) == [2]


def test_resolve_selected_empty_with_no_selection():
    nav = types.SimpleNamespace(selected_label_ids=lambda: set(), _items=[])
    assert _ann(_vol(), nav)._resolve_batch_targets({"mode": "selected"}) == []


def test_resolve_tier():
    items = [
        {"label_id": 1, "tier": "good"},
        {"label_id": 2, "tier": "review"},
        {"label_id": 3, "tier": "review"},
    ]
    nav = types.SimpleNamespace(selected_label_ids=lambda: set(), _items=items)
    ann = _ann(_vol(), nav)
    assert ann._resolve_batch_targets({"mode": "tier", "tiers": ["review"]}) == [2, 3]
    assert ann._resolve_batch_targets({"mode": "tier", "tiers": ["good"]}) == [1]
    assert ann._resolve_batch_targets({"mode": "tier", "tiers": ["bad"]}) == []


def test_resolve_tier_multi():
    items = [
        {"label_id": 1, "tier": "good"},
        {"label_id": 2, "tier": "review"},
        {"label_id": 3, "tier": "bad"},
    ]
    nav = types.SimpleNamespace(selected_label_ids=lambda: set(), _items=items)
    got = _ann(_vol(), nav)._resolve_batch_targets(
        {"mode": "tier", "tiers": ["good", "bad"]}
    )
    assert got == [1, 3]


def test_resolve_plot_reads_the_live_getter():
    """The fourth target: what the QC plot shows
    highlighted, read through the controller's getter, ∩ present."""
    nav = types.SimpleNamespace(selected_label_ids=lambda: {1}, _items=[])
    ann = _ann(_vol(), nav)
    box = {"ids": [2, 99]}
    ann._get_plot_selection_cb = lambda: box["ids"]
    assert ann._resolve_batch_targets({"mode": "plot"}) == [2]
    box["ids"] = [1, 3]
    assert ann._resolve_batch_targets({"mode": "plot"}) == [1, 3]
    box["ids"] = None  # no QC plot on this patch
    assert ann._resolve_batch_targets({"mode": "plot"}) == []
